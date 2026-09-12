"""Compare archived responses with a frozen Phi teacher's last prompt query.

Top-k reads are diagnostic references using the full historical keys, not
bounded-memory serving results. No answer labels enter the attention calculation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer.model import QCCArchive
from qcc_transformer.associative import SetAssociativeLandmarkBank
from qcc_transformer.hf_loading import load_hf_causal_lm


def exact_prefill_output(module, inputs, kwargs, output):
    """Diagnostic intervention: preserve bounded QCC state, return exact prefill."""
    hidden = kwargs.get('hidden_states', inputs[0] if inputs else None)
    if hidden.shape[1] <= 1:
        return output
    exact, _ = module.base_attention(
        hidden_states=hidden,
        position_embeddings=kwargs['position_embeddings'],
        attention_mask=kwargs.get('attention_mask'),
        past_key_values=None,
    )
    return (exact, *output[1:])


def trace_candidate(model, tokenizer, record, args):
    """Observe answer-neighbour KV retention without feeding labels to selection."""
    from benchmarks.benchmark_hf_ruler import _run_model
    from qcc_transformer.hybrid_archive import patch_hf_model_hybrid
    text = record['input']
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    answer_tokens = set()
    for answer in record['outputs']:
        start = text.find(answer)
        if start < 0:
            raise ValueError('trace requires answers present in the input')
        answer_tokens.update(i for i, (a, b) in enumerate(offsets) if b > start and a < start+len(answer))
    tracked = set(i for token in answer_tokens for i in range(max(0, token-16), min(len(offsets), token+17)))
    records = [(args.record, record)]
    baseline = _run_model(model, tokenizer, records, torch.device('cuda'), 128,
                          offload_full_kv=True, full_kv_resident_layers=8)
    model.requires_grad_(False)
    config = dict(window_size=args.window_size, attention_sink_size=args.sink_size,
                  num_codes=args.num_codes, use_triton=False,
                  prefill_chunk_size=args.prefill_chunk_size,
                  hybrid_kwargs=dict(exact_num_sets=args.exact_num_sets,
                      exact_ways=args.exact_ways,
                      background_size=args.background_size,
                      block_size=args.block_size,
                      quality_query_tail=args.quality_query_tail,
                      quality_first=True, exact_attention=True))
    names = patch_hf_model_hybrid(model, **config)
    captures = {}
    traces = []
    handles = []

    def wrap_update(bank, captured):
        original = bank.update
        def observed(key, value, **kwargs):
            position = bank._step + args.sink_size
            if position in tracked:
                captured[position] = (key.detach().float().clone(), value.detach().float().clone())
            return original(key, value, **kwargs)
        return observed

    def summarize(module, phase):
        index = module.qcc._qcc_layer_index
        bank = module.qcc.archive.exact_bank
        positions = bank._ages.flatten(2, 3)-1+args.sink_size
        valid = torch.isfinite(bank._scores.flatten(2, 3))
        bg_valid = torch.arange(bank.background_size, device=positions.device)[None, None, :] < bank._background_count[..., None].clamp(max=bank.background_size)
        rows = []
        for token, (key, value) in sorted(captures[index].items()):
            fg_heads = ((positions == token) & valid).any(-1)[0]
            bg_heads = ((bank._background_keys == key.unsqueeze(2)).all(-1)
                        & (bank._background_values == value.unsqueeze(2)).all(-1)
                        & bg_valid).any(-1)[0]
            rows.append(dict(token=token, answer_token=token in answer_tokens,
                foreground_heads=fg_heads.nonzero().flatten().tolist(),
                background_matching_heads=bg_heads.nonzero().flatten().tolist()))
        traces.append(dict(layer=index, phase=phase, captured_tokens=len(rows), tokens=rows))
        print(f'retention trace layer {index}: {phase}', flush=True)

    def after_prefill(module, inputs, kwargs, output):
        hidden = kwargs.get('hidden_states', inputs[0] if inputs else None)
        if hidden is not None and hidden.shape[1] > 1:
            summarize(module, 'prefill')

    for name in names:
        module = model.get_submodule(name)
        captured = captures.setdefault(module.qcc._qcc_layer_index, {})
        module.qcc.archive.exact_bank.update = wrap_update(module.qcc.archive.exact_bank, captured)
        handles.append(module.register_forward_hook(after_prefill, with_kwargs=True))
        if args.exact_prefill_intervention:
            handles.append(module.register_forward_hook(exact_prefill_output, with_kwargs=True))
    try:
        candidate = _run_model(model, tokenizer, records, torch.device('cuda'), 128, qcc=True)
        for name in names:
            summarize(model.get_submodule(name), 'completion')
    finally:
        for handle in handles:
            handle.remove()
        for name in names:
            del model.get_submodule(name).qcc.archive.exact_bank.update
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(record=args.record, model=args.model,
        scope='passive label-aware diagnostic; labels never enter selection or generation',
        prefill_intervention='exact attention output with QCC state populated from teacher hidden states' if args.exact_prefill_intervention else None,
        config=config, answer_token_positions=sorted(answer_tokens),
        baseline=baseline, candidate=candidate, traces=traces), indent=2)+'\n')


def trace_teacher_heads(model, tokenizer, record, args):
    from benchmarks.benchmark_hf_ruler import _run_model
    from transformers.models.phi3.modeling_phi3 import ALL_ATTENTION_FUNCTIONS
    retention = json.loads(args.trace_teacher_heads.read_text())
    if retention['record'] != args.record or retention['model'] != args.model:
        raise ValueError('retention trace must match the teacher record and model')
    positions = retention['answer_token_positions']
    prefill = {x['layer']: x for x in retention['traces'] if x['phase'] == 'prefill'}
    prompt_length = retention['baseline'][0]['tokens']
    retention_config = retention['config']
    policy = retention_config['hybrid_kwargs']
    foreground_capacity = policy['exact_num_sets'] * policy['exact_ways']
    background_capacity = policy['background_size']
    observations = []
    offline_membership = {}
    replay_membership = {}
    generated_ids = []
    original_attention = ALL_ATTENTION_FUNCTIONS['sdpa']
    original_generate = model.generate

    def observed_attention(module, query, key, value, attention_mask, **kwargs):
        output = original_attention(module, query, key, value, attention_mask, **kwargs)
        if query.shape[2] > 1:
            start = retention_config['attention_sink_size']
            window = retention_config['window_size']
            end = key.shape[2] - window
            tail = policy.get('quality_query_tail')
            query_start = max(window, key.shape[2] - tail) if tail else window
            qp = torch.linspace(query_start, key.shape[2]-1,
                                min(32, key.shape[2]-query_start),
                                device=key.device).round().long().unique(sorted=True)
            sample_queries = query.index_select(2, qp).float()
            historical = key[:, :, start:end].float()
            salience = torch.matmul(F.normalize(sample_queries, dim=-1), F.normalize(historical, dim=-1).transpose(-1, -2))
            kp = torch.arange(start, end, device=key.device)
            salience = salience.masked_fill(kp[None, :] > qp[:, None]-window, -torch.inf).amax(2)
            generator = torch.Generator(device=key.device).manual_seed(11)
            priority = torch.rand(salience.shape, device=key.device, generator=generator)
            for size in (1, 32):
                if size == 1:
                    foreground = salience.topk(foreground_capacity, dim=-1).indices
                else:
                    blocks = (end-start)//size
                    block_scores = salience[..., :blocks*size].reshape(1, query.shape[1], blocks, size).amax(-1)
                    chosen = block_scores.topk(foreground_capacity//size, dim=-1).indices
                    foreground = (chosen[..., None]*size + torch.arange(size, device=key.device)).flatten(-2)
                background = priority.scatter(2, foreground, torch.inf).topk(background_capacity, largest=False, dim=-1).indices
                retained_positions = torch.cat((foreground, background), -1) + start
                target_positions = torch.tensor(positions, device=key.device)
                offline_membership[module.layer_idx, size] = (retained_positions[..., None] == target_positions).any(2)[0]
            if args.replay_teacher_retention and module.layer_idx in (15, 17, 19, 20):
                from qcc_transformer.hybrid_archive import HybridQCCArchive
                replay = HybridQCCArchive(
                    query.shape[1], key.shape[-1], num_codes=1,
                    window_size=window, use_triton=False,
                    **policy,
                ).to(key.device)
                replay.reset_state(1, device=key.device)
                bank = replay.exact_bank
                # Replay admission only; teacher attention output remains exact.
                for lo in range(start, end, 512):
                    hi = min(lo + 512, end)
                    tile_scores = replay._quality_first_salience(
                        key[:, :, lo:hi], query[:, :, query_start:],
                        key_start=lo, query_start=query_start-window,
                    )
                    for offset in range(hi-lo):
                        replay._admit_one(key[:, :, lo+offset], value[:, :, lo+offset],
                                          tile_scores[:, :, offset])
                fg_positions = bank._ages.flatten(2, 3)-1+start
                fg_valid = torch.isfinite(bank._scores.flatten(2, 3))
                bg_valid = torch.arange(bank.background_size, device=key.device)[None, None, :] < bank._background_count[..., None].clamp(max=bank.background_size)
                membership = []
                for token in positions:
                    fg = ((fg_positions == token) & fg_valid).any(-1)
                    bg = ((bank._background_keys == key[:, :, token:token+1].float()).all(-1)
                          & (bank._background_values == value[:, :, token:token+1].float()).all(-1)
                          & bg_valid).any(-1)
                    membership.append((fg | bg)[0])
                replay_membership[module.layer_idx] = torch.stack(membership, -1)
                print('teacher retention replay completed:', module.layer_idx, flush=True)
                del replay, bank
        scores = torch.matmul(query[:, :, -1:].float(), key.float().transpose(-1, -2)).squeeze(2)
        scores *= kwargs.get('scaling', module.scaling)
        if attention_mask is not None:
            mask = attention_mask[:, :, -1, :key.shape[2]]
            scores = scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
        probability = scores.softmax(-1)[0]
        masses = probability[:, positions]
        matched = torch.zeros_like(masses, dtype=torch.bool)
        tokens = {x['token']: x for x in prefill[module.layer_idx]['tokens']}
        for column, token in enumerate(positions):
            row = tokens.get(token)
            if row is not None:
                heads = sorted(set(row['foreground_heads']) | set(row['background_matching_heads']))
                matched[heads, column] = True
        observation = dict(layer=module.layer_idx,
            predicted_output_index=key.shape[2]-prompt_length,
            answer_mass_by_head=masses.sum(-1).tolist(),
            retained_answer_mass_by_head=(masses*matched).sum(-1).tolist())
        for size in (1, 32):
            observation[f'offline_block{size}_retained_mass_by_head'] = (masses*offline_membership[module.layer_idx, size]).sum(-1).tolist()
        if module.layer_idx in replay_membership:
            observation['replay_retained_mass_by_head'] = (masses*replay_membership[module.layer_idx]).sum(-1).tolist()
        observations.append(observation)
        if module.layer_idx == model.config.num_hidden_layers-1:
            print('teacher output query traced:', key.shape[2]-prompt_length, flush=True)
        return output

    def observed_generate(*inputs, **kwargs):
        output = original_generate(*inputs, **kwargs)
        generated_ids.extend(output[0, prompt_length:].tolist())
        return output

    ALL_ATTENTION_FUNCTIONS.register('sdpa', observed_attention)
    model.generate = observed_generate
    try:
        baseline = _run_model(model, tokenizer, [(args.record, record)], torch.device('cuda'), 128,
                              offload_full_kv=True, full_kv_resident_layers=8)
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', original_attention)
        del model.generate
    answer_steps = set()
    for answer in record['outputs']:
        ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        while ids and not tokenizer.decode(ids[:1]).strip():
            ids = ids[1:]
        if ids:
            for start in range(len(generated_ids)-len(ids)+1):
                if generated_ids[start:start+len(ids)] == ids:
                    answer_steps.update(range(start, start+len(ids)))
    selected = [row for row in observations if row['predicted_output_index'] in answer_steps]
    total = sum(sum(row['answer_mass_by_head']) for row in selected)
    retained = sum(sum(row['retained_answer_mass_by_head']) for row in selected)
    offline_fractions = {size: sum(sum(row[f'offline_block{size}_retained_mass_by_head']) for row in selected)/total
                         for size in (1, 32)} if total else {}
    replay_rows = [row for row in selected if 'replay_retained_mass_by_head' in row]
    replay_total = sum(sum(row['answer_mass_by_head']) for row in replay_rows)
    replay_summary = dict(layers=sorted(replay_membership), teacher_answer_mass=replay_total,
        replay_retained_mass=sum(sum(row['replay_retained_mass_by_head']) for row in replay_rows),
        candidate_retained_mass=sum(sum(row['retained_answer_mass_by_head']) for row in replay_rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(record=args.record, model=args.model,
        scope='teacher answer-token attention aligned with candidate prefill KV membership; descriptive, not a causal attribution',
        retention_trace=str(args.trace_teacher_heads), baseline=baseline,
        generated_token_ids=generated_ids, answer_generation_indices=sorted(answer_steps),
        teacher_answer_attention_mass=total, retained_teacher_answer_attention_mass=retained,
        retained_mass_fraction=retained/total if total else None,
        offline_retention_fractions=offline_fractions,
        teacher_retention_replay=replay_summary,
        offline_config=dict(foreground_capacity=foreground_capacity,
            background_capacity=background_capacity, quality_query_tail=policy.get('quality_query_tail'),
            query_samples=32, block_sizes=[1, 32]),
        offline_scope='teacher-prefill selection with candidate capacity and query interval; offline background sampling, not candidate-state or generation results',
        observations=selected), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--ruler-jsonl', type=Path, required=True)
    parser.add_argument('--record', type=int, default=3)
    parser.add_argument('--window-size', type=int, default=4096)
    parser.add_argument('--sink-size', type=int, default=32)
    parser.add_argument('--query-sampling', choices=('uniform', 'tail'), default='uniform')
    parser.add_argument('--tail-span', type=int, default=32)
    parser.add_argument('--selection-block-size', type=int, choices=(1, 16, 32), default=1)
    parser.add_argument('--num-codes', type=int, default=16)
    parser.add_argument('--prefill-chunk-size', type=int, default=512)
    parser.add_argument('--exact-num-sets', type=int, default=96)
    parser.add_argument('--exact-ways', type=int, default=4)
    parser.add_argument('--background-size', type=int, default=128)
    parser.add_argument('--block-size', type=int, default=1)
    parser.add_argument('--quality-query-tail', type=int, default=32)
    parser.add_argument('--trace-candidate', action='store_true')
    parser.add_argument('--exact-prefill-intervention', action='store_true',
                        help='diagnostic only: exact prefill outputs followed by bounded QCC decode')
    parser.add_argument('--trace-teacher-heads', type=Path)
    parser.add_argument('--replay-teacher-retention', action='store_true',
                        help='replay actual bounded admission on teacher KV at layers 15,17,19,20')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with args.ruler_jsonl.open() as stream:
        record = next(json.loads(line) for i, line in enumerate(stream, 1) if i == args.record)
    model = load_hf_causal_lm(args.model, dtype=torch.bfloat16, device='cuda', attn_implementation='sdpa')
    if args.trace_teacher_heads:
        trace_teacher_heads(model, tokenizer, record, args)
        return
    if args.trace_candidate:
        trace_candidate(model, tokenizer, record, args)
        return
    encoded = tokenizer(record['input'], return_tensors='pt', add_special_tokens=False).to('cuda')
    heads = model.config.num_attention_heads
    dim = model.config.hidden_size // heads
    rows = []

    def probe(module, inputs, kwargs):
        hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else inputs[0]
        n = hidden.shape[1]
        q, k, v = module.qkv_proj(hidden).split(model.config.hidden_size, dim=-1)
        q, k, v = (x.reshape(1, n, heads, dim).transpose(1, 2) for x in (q, k, v))
        all_qr, kr = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
        qr = all_qr[:, :, -1].float()
        scores = torch.einsum('bhd,bhtd->bht', qr, kr.float()) / math.sqrt(dim)
        probability = scores.softmax(-1)
        start, end = args.sink_size, n - args.window_size
        remote_mass = probability[:, :, start:end].sum(-1)
        remote_prob = scores[:, :, start:end].softmax(-1)
        values = v[:, :, start:end].float()
        target = torch.einsum('bht,bhtd->bhd', remote_prob, values)
        horizons = torch.logspace(math.log10(args.window_size), math.log10(model.config.max_position_embeddings), 4)
        rates = tuple(torch.exp(-math.log(2) / horizons).tolist())
        torch.manual_seed(123 + module.layer_idx)
        archive = QCCArchive(heads, dim, num_codes=16, decay_rates=rates, window_size=args.window_size, use_triton=False).cuda()
        archive.reset_state(1, device=hidden.device)
        content = torch.einsum('bhtd,hmd->bhmt', k[:, :, start:end].float(), archive.codes)
        content = (content / math.sqrt(dim)).clamp(-20, 10).exp()
        ages = (n - 1) - torch.arange(start, end, device=hidden.device)
        decay = archive.decay_rates[:, None].pow(ages[None, :])
        weight = content.unsqueeze(3) * decay[None, None, None]
        archive._denominator = weight.sum(-1)
        archive._numerator = torch.einsum('bhmst,bhtd->bhmsd', weight, values)
        candidate = archive.read(q[:, :, -1].float())

        def weighted_error(output):
            error = (output - target).square().sum(-1)
            energy = target.square().sum(-1)
            return float((error * remote_mass).sum() / (energy * remote_mass).sum().clamp_min(1e-12))

        row = {'layer': module.layer_idx, 'tokens': n,
               'remote_attention_mass_mean': float(remote_mass.mean()),
               'remote_attention_mass_max': float(remote_mass.max()),
               'archive_relative_squared_error': weighted_error(candidate),
               'archive_cosine_mean': float(F.cosine_similarity(candidate, target, dim=-1).mean())}
        for count in (1, 16, 128):
            top_weight, index = remote_prob.topk(min(count, end-start), dim=-1)
            selected = values.gather(2, index[..., None].expand(-1, -1, -1, dim))
            estimate = (selected * top_weight[..., None]).sum(2) / top_weight.sum(-1, keepdim=True)
            row[f'oracle_top{count}_relative_squared_error'] = weighted_error(estimate)
        # Retention uses earlier prompt queries; the final probe query is excluded.
        if args.query_sampling == 'tail':
            if args.tail_span <= 0:
                raise ValueError('tail-span must be positive')
            samples = torch.arange(max(args.window_size, n-1-args.tail_span), n-1, device=hidden.device)
        else:
            samples = torch.linspace(args.window_size, n-2, 32, device=hidden.device).long().unique()
        sample_q = all_qr.index_select(2, samples).float()
        historical_k = kr[:, :, start:end].float()
        key_positions = torch.arange(start, end, device=hidden.device)
        causal = key_positions[None, :] <= samples[:, None] - args.window_size
        for metric in ('cosine', 'dot'):
            qs, ks = sample_q, historical_k
            if metric == 'cosine':
                qs, ks = F.normalize(qs, dim=-1), F.normalize(ks, dim=-1)
            salience = torch.einsum('bhqd,bhtd->bhqt', qs, ks)
            salience = salience.masked_fill(~causal, -torch.inf).amax(2)
            block_size = args.selection_block_size
            if block_size == 1:
                selected_index = salience.topk(min(512, end-start), dim=-1).indices
            else:
                blocks = (end-start)//block_size
                block_scores = salience[..., :blocks*block_size].reshape(1, heads, blocks, block_size).amax(-1)
                selected_blocks = block_scores.topk(min(512//block_size, blocks), dim=-1).indices
                selected_index = (selected_blocks[..., None]*block_size + torch.arange(block_size, device=hidden.device)).flatten(-2)
            retained_scores = scores[:, :, start:end].gather(2, selected_index)
            retained_values = values.gather(2, selected_index[..., None].expand(-1, -1, -1, dim))
            bank = SetAssociativeLandmarkBank(heads, dim, num_sets=1, ways=selected_index.shape[-1], probe_sets=1)
            bank.reset_state(1, device=hidden.device)
            bank._keys[:, :, 0] = historical_k.gather(2, selected_index[..., None].expand(-1, -1, -1, dim))
            bank._values[:, :, 0] = retained_values
            bank._scores.fill_(0)
            estimate, _ = bank.read_attention(qr)
            direct = (retained_values * retained_scores.softmax(-1)[..., None]).sum(2)
            torch.testing.assert_close(estimate, direct, atol=1e-5, rtol=1e-4)
            row[f'retained512_{metric}_softmax_relative_squared_error'] = weighted_error(estimate)
            important_count = min(384, selected_index.shape[-1])
            important = selected_index[:, :, :important_count]
            important_logits = scores[:, :, start:end].gather(2, important)
            important_values = values.gather(2, important[..., None].expand(-1, -1, -1, dim))
            remainder = end-start-important_count
            background_count = min(128, remainder)
            errors = []
            centered_errors = []
            if background_count:
                background_mean = (values.sum(2) - important_values.sum(2)) / remainder
                important_output = (important_values * important_logits.softmax(-1)[..., None]).sum(2)
                for seed in (11, 29, 47):
                    generator = torch.Generator(device=hidden.device).manual_seed(seed)
                    priority = torch.rand(salience.shape, device=hidden.device, generator=generator)
                    priority.scatter_(2, important, torch.inf)
                    background = priority.topk(background_count, dim=-1, largest=False).indices
                    background_logits = scores[:, :, start:end].gather(2, background)
                    background_logits += math.log(remainder / background_count)
                    background_values = values.gather(2, background[..., None].expand(-1, -1, -1, dim))
                    merged_logits = torch.cat((important_logits, background_logits), -1)
                    merged_values = torch.cat((important_values, background_values), 2)
                    mixed = (merged_values * merged_logits.softmax(-1)[..., None]).sum(2)
                    errors.append(weighted_error(mixed))
                    sampled_output = (background_values * background_logits.softmax(-1)[..., None]).sum(2)
                    covariance_correction = background_count / max(1, background_count-1)
                    centered_output = background_mean + covariance_correction * (sampled_output - background_values.mean(2))
                    background_mass = torch.sigmoid(background_logits.logsumexp(-1) - important_logits.logsumexp(-1))[..., None]
                    centered = (1-background_mass)*important_output + background_mass*centered_output
                    centered_errors.append(weighted_error(centered))
            row[f'important384_background128_{metric}_errors'] = errors
            row[f'important384_centered_background128_{metric}_errors'] = centered_errors
        rows.append(row)
        print(json.dumps(row), flush=True)

    hooks = [model.model.layers[i].self_attn.register_forward_pre_hook(probe, with_kwargs=True)
             for i in (0, 7, 15, 23, 31)]
    class ReferenceCache(DynamicCache):
        def offload(self, layer_idx, only_non_sliding=True):
            super().offload(layer_idx, only_non_sliding)
            torch.cuda.synchronize()
    try:
        with torch.no_grad():
            model(**encoded, past_key_values=ReferenceCache(offloading=True), use_cache=True, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'record': args.record, 'model': args.model,
        'window_size': args.window_size, 'sink_size': args.sink_size,
        'query_sampling': args.query_sampling,
        'tail_span': args.tail_span,
        'selection_block_size': args.selection_block_size,
        'scope': 'teacher last-prompt-query diagnostic; oracle top-k and offline retention are not serving implementations',
        'layers': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
