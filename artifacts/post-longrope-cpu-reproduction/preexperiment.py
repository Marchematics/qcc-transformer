"""Post-LongRoPE QCC study, CPU reference experiments, not pretrained-LM scores.

Run: python preexperiment.py
Dependencies: torch, numpy, scipy. No network access, training, or GPU required.
The Phi factors below come from Microsoft's public Phi-3.5 config.
The dispatch function reproduces the two relevant length assignments in current
HFQCCAttention._bounded_prefill; it is not a full copy of the repository.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm, binom

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent
LONG = [1.0800000429153442,1.1100000143051147,1.1399999856948853,1.340000033378601,1.5899999141693115,1.600000023841858,1.6200000047683716,2.620000123977661,3.2300000190734863,3.2300000190734863,4.789999961853027,7.400000095367432,7.700000286102295,9.09000015258789,12.199999809265137,17.670000076293945,24.46000099182129,28.57000160217285,30.420001983642578,30.840002059936523,32.590003967285156,32.93000411987305,42.320003509521484,44.96000289916992,50.340003967285156,50.45000457763672,57.55000305175781,57.93000411987305,58.21000289916992,60.1400032043457,62.61000442504883,62.62000274658203,62.71000289916992,63.1400032043457,63.1400032043457,63.77000427246094,63.93000411987305,63.96000289916992,63.970001220703125,64.02999877929688,64.06999969482422,64.08000183105469,64.12000274658203,64.41000366210938,64.4800033569336,64.51000213623047,64.52999877929688,64.83999633789062]
SHORT = [1.0,1.0199999809265137,1.0299999713897705,1.0299999713897705,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0699999332427979,1.0999999046325684,1.1099998950958252,1.1599998474121094,1.1599998474121094,1.1699998378753662,1.2899998426437378,1.339999794960022,1.679999828338623,1.7899998426437378,1.8199998140335083,1.8499997854232788,1.8799997568130493,1.9099997282028198,1.9399996995925903,1.9899996519088745,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0799996852874756,2.0899996757507324,2.189999580383301,2.2199995517730713,2.5899994373321533,2.729999542236328,2.749999523162842,2.8399994373321533]
SCALE = math.sqrt(1.0 + math.log(131072 / 4096) / math.log(4096))

def rope(x: torch.Tensor, pos: torch.Tensor, context: int, *, scaled=True) -> torch.Tensor:
    """Microsoft half-split LongRoPE, x shape [heads,tokens,96]."""
    factor = torch.tensor(LONG if context > 4096 else SHORT, dtype=torch.float32)
    inv = 1.0 / (factor * 10000.0 ** (torch.arange(0,96,2).float()/96))
    angle = pos.float()[:,None] * inv[None,:]
    angle = torch.cat((angle,angle),-1)
    c = (angle.cos()*(SCALE if scaled else 1.)).to(x.dtype)
    s = (angle.sin()*(SCALE if scaled else 1.)).to(x.dtype)
    a,b=x.chunk(2,-1)
    return x*c+torch.cat((-b,a),-1)*s

def metrics(x: torch.Tensor,y: torch.Tensor) -> dict:
    x,y=x.double(),y.double()
    return {'cosine':float(F.cosine_similarity(x,y,dim=-1).mean()),
            'relative_l2':float((x-y).norm()/y.norm().clamp_min(1e-30)),
            'max_abs':float((x-y).abs().max())}

def attn(q,k,v,mask=None):
    logits=q.float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
    if mask is not None: logits=logits.masked_fill(~mask,-torch.inf)
    return logits.softmax(-1)@v.float()

def dispatch(x,pos,past,chunk,corrected):
    # Current _bounded_prefill passes length (new tokens) to all inner chunks.
    length=x.shape[1]
    context=past+length if corrected else length
    return torch.cat([rope(x[:,lo:lo+chunk],pos[lo:lo+chunk],context)
                      for lo in range(0,length,chunk)],1)

def longrope_experiments():
    gen=torch.Generator().manual_seed(20260913)
    n=7422
    q=torch.randn(4,n,96,generator=gen); k=torch.randn(4,n,96,generator=gen)
    v=torch.randn(4,n,96,generator=gen); p=torch.arange(n)
    qr=rope(q,p,n); kr=rope(k,p,n)
    q_old=torch.cat((rope(q[:,:4096],p[:4096],4096),rope(q[:,4096:],p[4096:],n)),1)
    k_old=torch.cat((rope(k[:,:4096],p[:4096],4096),rope(k[:,4096:],p[4096:],n)),1)
    ref=attn(qr[:,-16:],kr,v)
    rows={'old_per_position':{'q':metrics(q_old,qr),'k':metrics(k_old,kr),
                             'last16_attention':metrics(attn(q_old[:,-16:],k_old,v),ref)},
          'new_first_prompt_chunked':metrics(dispatch(q,p,0,512,False),qr)}
    continuations=[]
    for past,count in [(7422,8),(4000,256),(8192,512),(3500,128),(4095,2)]:
        x=torch.randn(4,count,96,generator=gen); positions=past+torch.arange(count)
        reference=rope(x,positions,past+count)
        continuations.append({'past':past,'new_tokens':count,
            'current_context':count,'native_eager_context':past+count,
            'current':metrics(dispatch(x,positions,past,128,False),reference),
            'corrected':metrics(dispatch(x,positions,past,128,True),reference)})
    rows['multi_token_continuations']=continuations
    # Model-level external chunking is different from slicing one full forward.
    # To match a native one-shot long prompt, all external chunks must receive
    # the same declared prompt extent, not merely the processed prefix extent.
    growing=[]
    for lo in range(0,n,512):
        hi=min(n,lo+512)
        growing.append(rope(q[:,lo:hi],p[lo:hi],hi))
    rows['model_level_chunks_growing_extent_vs_one_shot']=metrics(torch.cat(growing,1),qr)
    declared=torch.cat([rope(q[:,lo:lo+512],p[lo:lo+512],n) for lo in range(0,n,512)],1)
    rows['model_level_chunks_declared_full_extent_vs_one_shot']=metrics(declared,qr)
    # External native (cos,sin) is authoritative in the actual module and bypasses
    # internal selection; do not claim every earlier run reached the faulty branch.
    q_no=rope(q[:,-16:],p[-16:],n,scaled=False)
    k_no=rope(k,p,n,scaled=False)
    rows['missing_amplitude_scaling']={'q':metrics(q_no,qr[:,-16:]),
        'attention':metrics(attn(q_no,k_no,v),ref),'correct_factor':SCALE}
    # A complete request known to be long differs from a previously short request
    # later extended. This is a reference-semantics distinction, not a pruning bug.
    short_prefix=rope(q[:,:4000],p[:4000],4000)
    rows['native_length_metadata_changes_prefix']=metrics(short_prefix,qr[:,:4000])
    # Information loss: two old-basis vectors cancel in a mean, but cannot be
    # transformed to the new basis from that mean alone.
    p2=torch.tensor([1000,3000]); u=torch.zeros(1,2,96); u[:,:,0]=torch.tensor([1.,-1.])
    # Inverse rotation uses negative positions and divides the two scale factors.
    raw=rope(u,-p2,4000)/(SCALE*SCALE)
    old_mean=rope(raw,p2,4000).mean(1)
    new_mean=rope(raw,p2,7422).mean(1)
    rows['merged_key_basis_counterexample']={'old_mean_norm':float(old_mean.norm()),
                                           'new_mean_norm':float(new_mean.norm()),
                                           'scope':'mean alone cannot rephase position-dependent members'}
    return rows

def masking_and_normalization():
    # Two padded tokens plus one valid token, all equal logits.
    q=torch.zeros(1,1,4); k=torch.zeros(1,3,4)
    v=torch.tensor([[[0.,0.,0.,0.],[0.,0.,0.,0.],[1.,1.,1.,1.]]])
    ignored=attn(q,k,v); correct=attn(q,k,v,torch.tensor([[[False,False,True]]]))
    # Exact output-deletion identity: o - o_S = p_removed (o_R - o_S).
    g=torch.Generator().manual_seed(14)
    q2=torch.randn(12,8,generator=g); k2=torch.randn(128,8,generator=g); v2=torch.randn(128,8,generator=g)
    prob=(q2@k2.T/math.sqrt(8)).softmax(-1)
    selected=torch.arange(48); removed=torch.arange(48,128)
    full=prob@v2; mass=prob[:,removed].sum(-1)
    kept=(prob[:,selected]@v2[selected])/prob[:,selected].sum(-1,keepdim=True)
    other=(prob[:,removed]@v2[removed])/mass[:,None]
    identity=mass[:,None]*(other-kept)
    return {'padding_ignored_output':ignored.flatten().tolist(),
            'padding_native_output':correct.flatten().tolist(),
            'pruning_identity_max_error':float(((full-kept)-identity).abs().max()),
            'note':'identity assumes same Q,K,V before deletion, not a multilayer task bound'}

def pack4(x:torch.Tensor,group:int=32):
    """Symmetric group int4 with FP32 scales; explicit real packed storage."""
    if x.shape[-1]%group: raise ValueError('dimension must divide group size')
    y=x.float().reshape(*x.shape[:-1],-1,group)
    scale=y.abs().amax(-1,keepdim=True).clamp_min(1e-12)/7
    code=(y/scale).round().clamp(-7,7).to(torch.int16).reshape(x.shape)+8
    packed=(code[...,0::2] | (code[...,1::2]<<4)).to(torch.uint8)
    return packed,scale.squeeze(-1)

def unpack4(packed,scale,group=32):
    code=torch.stack((packed&15,packed>>4),-1).flatten(-2).float()-8
    return (code.reshape(*scale.shape,group)*scale[...,None]).flatten(-2)

def storage_experiments():
    g=torch.Generator().manual_seed(78)
    k=torch.randn(4,4096,96,generator=g).bfloat16(); v=torch.randn(4,4096,96,generator=g).bfloat16()
    q=torch.randn(4,32,96,generator=g).bfloat16()
    same=attn(q,k.float(),v.float()); bf=attn(q,k,v)
    pk,sk=pack4(k); pv,sv=pack4(v); k4=unpack4(pk,sk); v4=unpack4(pv,sv)
    quant=attn(q,k4,v4)
    # Near ties can flip retrieval even while reconstruction error is small.
    margins=[]
    for amplitude in [1.,2.,4.]:
        qs=k[:,:32].float()*amplitude
        base=(qs@k.float().transpose(-1,-2)/math.sqrt(96))
        changed=qs@k4.transpose(-1,-2)/math.sqrt(96)
        top=base.topk(2,-1).values
        margins.append({'query_scale':amplitude,'argmax_agreement':float((base.argmax(-1)==changed.argmax(-1)).float().mean()),
                        'min_top2_margin':float((top[...,0]-top[...,1]).min()),
                        'attention':metrics(attn(qs,k4,v4),attn(qs,k,v))})
    return {'bf16_source_fp32_vs_bf16_arithmetic':metrics(bf,same),
            'int4_output':metrics(quant,same),
            'bf16_kv_bytes':k.numel()*k.element_size()+v.numel()*v.element_size(),
            'int4_kv_bytes':sum(t.numel()*t.element_size() for t in (pk,sk,pv,sv)),
            'sharp_queries':margins,'note':'constructed tensors; not pretrained retrieval quality'}

class FixedSlotKV:
    """Minimal original-KV selection state, fixed capacity, no future queries.

    Incoming scores are inputs, not a trained importance predictor. Final retained
    items are the global largest immutable scores. The study does not claim this
    solves unknown-future-query ranking. Updates do not compute attention outputs.
    """
    def __init__(self,capacity,dim):
        self.capacity,self.dim=capacity,dim
        self.k=torch.zeros(capacity,dim); self.v=torch.zeros_like(self.k)
        self.s=torch.full((capacity,),-torch.inf); self.ids=torch.full((capacity,),-1,dtype=torch.int64)
        self.total=torch.zeros((),dtype=torch.int64)
    def update(self,k,v,s):
        # Top-k is globally correct for immutable per-entry scores; ties are absent
        # in the deterministic construction below. Production must define ties.
        n=k.shape[0]; ids=torch.arange(int(self.total),int(self.total)+n)
        new_s=torch.cat((self.s,s)); chosen=new_s.topk(self.capacity).indices
        self.k.copy_(torch.cat((self.k,k))[chosen]); self.v.copy_(torch.cat((self.v,v))[chosen])
        self.ids.copy_(torch.cat((self.ids,ids))[chosen]); self.s.copy_(new_s[chosen]); self.total.add_(n)
    def bytes(self):
        return sum(t.untyped_storage().nbytes() for t in (self.k,self.v,self.s,self.ids,self.total))

def million_stream():
    c,d,block=512,8,1024
    bank=FixedSlotKV(c,d); state={}; gen=torch.Generator().manual_seed(28)
    maxn=1048576; ktotal=0; t0=time.monotonic()
    # Analytically known, well-spread immutable priorities using a permutation.
    # They do not inspect future queries or encode semantic importance.
    for start in range(0,maxn,block):
        ids=torch.arange(start,start+block,dtype=torch.int64)
        scores=((ids*2654435761)%4294967296).double()/4294967296
        # Use float64 scores in both reference and bank for an exact ranking match.
        if bank.s.dtype!=torch.float64: bank.s=bank.s.double()
        k=torch.randn(block,d,generator=gen); v=torch.randn(block,d,generator=gen)
        bank.update(k,v,scores); ktotal+=block
        if ktotal in (131072,maxn):
            state[str(ktotal)]={'state_bytes':bank.bytes(),'valid_items':int((bank.ids>=0).sum())}
    all_ids=torch.arange(maxn,dtype=torch.int64)
    ref=(((all_ids*2654435761)%4294967296).double()/4294967296).topk(c).indices
    correct=torch.equal(torch.sort(bank.ids).values,torch.sort(ref).values)
    # A random independent target cannot be recalled merely from immutable
    # priorities: exact survival probability is C/N for this exchangeable setup.
    return {'events_processed':ktotal,'state':state,'growth':state[str(maxn)]['state_bytes']/state['131072']['state_bytes'],
            'global_topk_set_equal':correct,'elapsed_cpu_seconds':time.monotonic()-t0,
            'uniform_unseen_target_survival':c/maxn,
            'scope':'one head, dim8 state/update experiment; no LM and no GPU performance claim'}

def toy_native_length_transition():
    """Two random residual attention layers, dimension16, threshold32.

    Demonstrates why native incremental short->long and one long prefill need
    not agree. No pretrained parameters; length policy is an external input.
    """
    g=torch.Generator().manual_seed(52); dim=16; length=64; split=32
    x=torch.randn(length,dim,generator=g)
    weights=[tuple(torch.randn(dim,dim,generator=g)/math.sqrt(dim) for _ in range(4)) for _ in range(2)]
    inv0=10000.**(-torch.arange(0,dim,2).float()/dim); inv1=inv0/3.7
    def rr(y,pos,long):
        ang=pos[:,None]*(inv1 if long else inv0); ang=torch.cat((ang,ang),-1)
        a,b=y.chunk(2,-1); return y*ang.cos()+torch.cat((-b,a),-1)*ang.sin()
    def whole(inp,long):
        h=inp.clone()
        for wq,wk,wv,wo in weights:
            z=F.layer_norm(h,(dim,)); q=rr(z@wq,torch.arange(len(h)),long); k=rr(z@wk,torch.arange(len(h)),long); v=z@wv
            h=h+attn(q,k,v,torch.ones(len(h),len(h),dtype=torch.bool).tril())@wo
        return h
    def incremental(rephase):
        cached=[]; h=x[:split].clone()
        for wq,wk,wv,wo in weights:
            z=F.layer_norm(h,(dim,)); rawk=z@wk; v=z@wv
            q=rr(z@wq,torch.arange(split),False); k=rr(rawk,torch.arange(split),False)
            cached.append((rawk.clone(),k.clone(),v.clone()))
            h=h+attn(q,k,v,torch.ones(split,split,dtype=torch.bool).tril())@wo
        h=x[split:].clone(); p=split+torch.arange(length-split)
        for ix,(wq,wk,wv,wo) in enumerate(weights):
            z=F.layer_norm(h,(dim,)); q=rr(z@wq,p,True); k=rr(z@wk,p,True); v=z@wv
            raw_old,oldk,oldv=cached[ix]
            if rephase: oldk=rr(raw_old,torch.arange(split),True)
            keys=torch.cat((oldk,k)); values=torch.cat((oldv,v))
            mask=torch.arange(length)[None,:]<=p[:,None]
            h=h+attn(q,keys,values,mask)@wo
        return h
    target=whole(x,True)[split:]
    return {'direct_incremental_without_native_rebuild_vs_long_prefill':metrics(incremental(False),target),
            'rephase_keys_only_vs_long_prefill':metrics(incremental(True),target),
            'note':'reference schedules differ intentionally; not evidence native implementation is wrong'}

def feasibility():
    L,H,D,d,V,I=32,32,96,3072,32064,8192
    logical=2*V*d+L*(4*d*d+3*d*I+2*d)+d
    writer=L*(d*H+H)
    current=6589440
    capacity_rows=[]
    for capacity in [4096,8192,16384,32768]:
        # Real prototype int4 group32 uses FP32 scales, one int64 ID + FP32 score.
        bytes_per_pair=2*(D//2+(D//32)*4)+8+4
        hist=L*H*capacity*bytes_per_pair
        window=L*H*1024*D*2*2
        capacity_rows.append({'capacity_per_head':capacity,'history_bytes':hist,'recent_bf16_bytes':window,
                             'subtotal_bytes':hist+window,
                             'note':'excludes prefill scratch, model weights, server/runtime buffers'})
    stress=[]
    for C in [768,8192,16384]:
        for N in [131072,1048576]:
            threshold=norm.ppf(1-C/N)
            stress.append({'capacity':C,'distractor_count':N,
                'normal_score_threshold_approx':threshold,
                'positive_mean_for_995_recall_approx':threshold+norm.ppf(.995),
                'recall_if_positive_mean_38_approx':norm.cdf(3.8-threshold)})
    amdahl=[]
    for f in [.05,.1,.2,.3,.4]:
        amdahl.append({'nonattention_fraction':f,'maximum_speedup':1/f,
                      'with_attention_20x_and_2percent_overhead':1/(f+(1-f)/20+.02)})
    # Batch1 bandwidth model, not hardware timing: Qwen config from official source.
    qwen=[]; qL,qH,qD=28,4,128
    weight_bytes=7.61e9*2
    baseline=qL*qH*qD*2*2*131072
    candidate=qL*qH*8192*(2*(qD//2+(qD//32)*4)+12)+qL*qH*1024*qD*4
    for batch in [1,4,8,16]:
        qwen.append({'batch':batch,'ideal_common_bandwidth_ratio':(weight_bytes+batch*baseline)/(weight_bytes+batch*candidate)})
    return {'logical_phi_parameters':logical,'packed_array_count_in_report':2009140224,
            'existing_trainable_fraction':current/logical,'linear_writer_parameters':writer,
            'linear_writer_fraction':writer/logical,'fixed_capacity_geometry':capacity_rows,
            'normal_score_separation_model':stress,'amdahl':amdahl,
            'qwen128k_bandwidth_illustration':qwen,
            'zero_failure_trials_for_one_sided_95_lower_at_995':math.ceil(math.log(.05)/math.log(.995)),
            'four_successes_one_sided_95_lower':.05**.25,
            'at_1000_trials_5_failures_lower_one_sided':float(__import__('scipy').stats.beta.ppf(.05,995,6))}

# An intentionally small proof-of-concept for a different background model.
# Exact only for a joint Gaussian population, not an arbitrary finite cache.
def gaussian_background():
    g=torch.Generator().manual_seed(381); n,d=32768,16
    k=torch.randn(n,d,generator=g); mapv=torch.randn(d,d,generator=g)/math.sqrt(d)
    v=k@mapv+.2*torch.randn(n,d,generator=g)
    mu=k.mean(0); mv=v.mean(0); dk=k-mu; dv=v-mv
    cov=dk.T@dk/n; cross=dv.T@dk/n
    rows=[]
    for norm_scale in [.25,1.,2.,4.]:
        q=torch.randn(64,d,generator=g)*norm_scale; t=q/math.sqrt(d)
        estimate=mv+t@cross.T
        logz=math.log(n)+t@mu+.5*torch.einsum('bi,ij,bj->b',t,cov,t)
        scores=q@k.T/math.sqrt(d); reference=scores.softmax(-1)@v
        rows.append({'query_sd':norm_scale,'output':metrics(estimate,reference),
            'mean_log_partition_error':float((logz-scores.logsumexp(-1)).abs().mean())})
    # Same first/second moments, different MGF. This falsifies a universal guarantee.
    a=torch.tensor([-1.,1.]); b=torch.tensor([-math.sqrt(2),0.,0.,math.sqrt(2)])
    return {'joint_gaussian_constructed':rows,
        'same_moment_counterexample':{'mean_a':float(a.mean()),'mean_b':float(b.mean()),
            'second_moment_a':float(a.square().mean()),'second_moment_b':float(b.square().mean()),
            'exp_mean_at_q4_a':float((4*a).exp().mean()),'exp_mean_at_q4_b':float((4*b).exp().mean())},
        'verdict':'do not promote Gaussian moments as arbitrary unknown-query memory'}



def native_rebuild_protocol():
    """Execute the native prepare / QCC-owned-counter interaction on a stub.

    This reproduces control flow, not a Phi forward pass. The official native
    prepare drops the cache exactly once when a short request first becomes long.
    """
    from types import SimpleNamespace
    from functools import wraps
    class Cache:
        def __init__(self, owners): self.owners=owners
        @property
        def seen_tokens(self): return max(x.seen for x in self.owners)
        def __bool__(self): return True
    class Model:
        def __init__(self,past):
            self.owners=[SimpleNamespace(seen=past) for _ in range(3)]
            self.config=SimpleNamespace(original_max_position_embeddings=4096,rope_scaling=True)
            self.reset_calls=0
        def prepare_inputs_for_generation(self,input_ids,past_key_values=None,**kwargs):
            # Semantic excerpt of Microsoft's native Phi preparation.
            if past_key_values and self.config.rope_scaling and input_ids.shape[1]>=4097:
                if past_key_values.seen_tokens<=4096: past_key_values=None
            if past_key_values is not None:
                past=past_key_values.seen_tokens
                if past<input_ids.shape[1]: input_ids=input_ids[:,past:]
            return {'input_ids':input_ids,'past_key_values':past_key_values}
        def reset(self,batch):
            self.reset_calls+=1
            for x in self.owners: x.seen=0
    def install(model):
        original=model.prepare_inputs_for_generation
        @wraps(original)
        def prepared(input_ids,past_key_values=None,*args,**kwargs):
            output=original(input_ids,past_key_values,*args,**kwargs)
            if past_key_values is not None and output.get('past_key_values') is None:
                model.reset(input_ids.shape[0])
            return output
        model.prepare_inputs_for_generation=prepared
    rows=[]
    for past,whole in [(4096,4097),(4095,4096),(7422,7423)]:
        for corrected in [False,True]:
            model=Model(past)
            if corrected: install(model)
            cache=Cache(model.owners)
            prepared=model.prepare_inputs_for_generation(torch.zeros(1,whole,dtype=torch.long),past_key_values=cache)
            incoming=prepared['input_ids'].shape[1]
            for x in model.owners: x.seen+=incoming
            rows.append({'past':past,'complete_input_tokens':whole,'synchronization_enabled':corrected,
                         'native_cache_discarded':prepared['past_key_values'] is None,
                         'forward_input_tokens':incoming,'qcc_layer_lengths_after': [x.seen for x in model.owners],
                         'reset_calls':model.reset_calls,'expected_length':whole})
    assert rows[0]['qcc_layer_lengths_after']==[8193]*3
    assert rows[1]['qcc_layer_lengths_after']==[4097]*3
    assert all(r['reset_calls']==0 for r in rows[2:])
    return {'rows':rows,'scope':'native prepare + persistent counter control-flow reproduction; no pretrained model'}


def attention_arithmetic_ladder():
    """Native eager, current tiled reference, and SDPA have different rounding.

    Reproduces the exposed arithmetic operations; not a conclusion about row16's
    remaining next-token difference. All variants use identical Q/K/V and masks.
    """
    g=torch.Generator().manual_seed(103)
    h,n,d=4,7422,96
    k=torch.randn(h,n,d,generator=g).bfloat16()
    v=torch.randn(h,n,d,generator=g).bfloat16()
    outputs=[]
    for nq in [1,16]:
        q=torch.randn(h,nq,d,generator=g).bfloat16()
        offset=n-nq
        mask=torch.arange(n)[None,:]<=offset+torch.arange(nq)[:,None]
        reference=attn(q,k,v,mask).bfloat16()
        eager_logits=(q@k.transpose(-1,-2))/math.sqrt(d)
        eager_logits=eager_logits.masked_fill(~mask,-torch.inf)
        eager=eager_logits.float().softmax(-1).bfloat16()@v
        sdpa=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        variants={'native_bf16_eager':metrics(eager,reference),'bf16_sdpa':metrics(sdpa,reference)}
        for qk_fp32 in [False,True]:
            m=torch.full((h,nq),-torch.inf); z=torch.zeros(h,nq); num=torch.zeros(h,nq,d)
            for lo in range(0,n,1024):
                hi=min(n,lo+1024)
                logits=(q.float()@k[:,lo:hi].float().transpose(-1,-2)) if qk_fp32 else (q@k[:,lo:hi].transpose(-1,-2)).float()
                logits=(logits/math.sqrt(d)).masked_fill(~mask[:,lo:hi],-torch.inf)
                newm=torch.maximum(m,logits.amax(-1))
                w=(logits-newm[...,None]).exp().masked_fill(~torch.isfinite(logits),0)
                old=(m-newm).exp().masked_fill(~torch.isfinite(m),0)
                z=z*old+w.sum(-1); num=num*old[...,None]+w@v[:,lo:hi].float(); m=newm
            out=(num/z[...,None]).bfloat16()
            variants['tiled_fp32_qk' if qk_fp32 else 'current_tiled_bf16_qk']=metrics(out,reference)
        outputs.append({'query_tokens':nq,'variants_to_fp32_mathematical_reference':variants})
    return {'rows':outputs,'scope':'constructed BF16 tensors; kernel rounding isolation, not real-model error attribution'}


def quantization_error_split():
    """Measure K and V quantization separately, retaining every token."""
    def pack8(x):
        y=x.float().reshape(*x.shape[:-1],-1,32)
        scale=y.abs().amax(-1,keepdim=True).clamp_min(1e-12)/127
        return (y/scale).round().clamp(-127,127).to(torch.int8).reshape_as(x),scale.squeeze(-1)
    def unpack8(code,scale): return (code.float().reshape(*scale.shape,32)*scale[...,None]).flatten(-2)
    g=torch.Generator().manual_seed(198)
    k=torch.randn(4,4096,96,generator=g).bfloat16(); v=torch.randn(4,4096,96,generator=g).bfloat16()
    q=torch.randn(4,32,96,generator=g).bfloat16(); ref=attn(q,k,v)
    kp4,ks4=pack4(k); vp4,vs4=pack4(v); kp8,ks8=pack8(k); vp8,vs8=pack8(v)
    keys={'bf16':(k,k.numel()*2),'int4':(unpack4(kp4,ks4),kp4.numel()+ks4.numel()*4),
          'int8':(unpack8(kp8,ks8),kp8.numel()+ks8.numel()*4)}
    vals={'bf16':(v,v.numel()*2),'int4':(unpack4(vp4,vs4),vp4.numel()+vs4.numel()*4),
          'int8':(unpack8(vp8,vs8),vp8.numel()+vs8.numel()*4)}
    rows=[]
    for kb,vb in [('int4','bf16'),('bf16','int4'),('int4','int4'),('int8','int4'),('int4','int8'),('int8','int8')]:
        kk,bk=keys[kb]; vv,bv=vals[vb]
        rows.append({'key_bits':kb,'value_bits':vb,'stored_kv_bytes':bk+bv,
                     'output':metrics(attn(q,kk,vv),ref)})
    return {'rows':rows,'note':'simple group32 quantizer; no learnt rotations; no task score; not a claim that advanced quantization has the same errors'}


def finite_information_bound():
    """Fano-style necessary bits for independent random values and unknown index.

    Values have a 128-bit alphabet; index may be provided by the later query.
    No claim that this lower bound is tight for a retrofit implementation.
    """
    err=.005; h=-err*math.log2(err)-(1-err)*math.log2(1-err)
    n=1_000_000; bits=n*(128-h-err*math.log2(2**128-1))
    return {'independent_values':n,'bits_per_value':128,'mean_error_limit':err,
            'necessary_state_bits_lower_bound':bits,'necessary_state_MiB':bits/8/2**20,
            'interpretation':'finite 1M information bound is far below multi-GiB QCC states; it does not rule out the finite workload, but rules out uniform arbitrary-horizon lossless compression'}


class CausalBlockKV:
    """Original-KV bounded reference with fixed logical commit boundaries.

    Each query attends to the retained bank plus all uncommitted recent tokens.
    Only AFTER a boundary token has been answered are old recent entries scored
    for eviction. No future Q is used. Scores are supplied causal inputs, NOT
    an implemented or trained importance model. One head for CPU experiments.
    """
    def __init__(self,capacity:int,window:int,commit:int,dim:int):
        self.capacity,self.window,self.commit,self.dim=capacity,window,commit,dim
        self.bk=torch.zeros(capacity,dim); self.bv=torch.zeros_like(self.bk)
        self.bs=torch.full((capacity,),-torch.inf,dtype=torch.float64)
        self.bi=torch.full((capacity,),-1,dtype=torch.int64)
        self.rk=torch.zeros(window+commit,dim); self.rv=torch.zeros_like(self.rk)
        self.rs=torch.full((window+commit,),-torch.inf,dtype=torch.float64)
        self.ri=torch.full((window+commit,),-1,dtype=torch.int64)
        self.count=torch.zeros((),dtype=torch.int64)
        self.nb,self.nr,self.writes=0,0,0
    def state_bytes(self):
        return sum(x.untyped_storage().nbytes() for x in (self.bk,self.bv,self.bs,self.bi,self.rk,self.rv,self.rs,self.ri,self.count))
    def step(self,q,k,v,score):
        i=self.nr; self.rk[i].copy_(k); self.rv[i].copy_(v)
        self.rs[i]=score; self.ri[i]=self.count; self.nr+=1
        keys=torch.cat((self.bk[:self.nb],self.rk[:self.nr]))
        values=torch.cat((self.bv[:self.nb],self.rv[:self.nr]))
        output=attn(q[None],keys,values)[0]
        self.count.add_(1)
        if int(self.count)%self.commit==0 and self.nr>self.window:
            n=self.nr-self.window
            ks=torch.cat((self.bk[:self.nb],self.rk[:n])); vs=torch.cat((self.bv[:self.nb],self.rv[:n]))
            scores=torch.cat((self.bs[:self.nb],self.rs[:n])); ids=torch.cat((self.bi[:self.nb],self.ri[:n]))
            take=min(self.capacity,len(scores)); chosen=scores.topk(take).indices
            # Chronological layout is not required mathematically, but removes
            # needless permutation-rounding from the no-eviction control.
            chosen=chosen[ids[chosen].argsort()]
            self.bk[:take].copy_(ks[chosen]); self.bv[:take].copy_(vs[chosen])
            self.bs[:take].copy_(scores[chosen]); self.bi[:take].copy_(ids[chosen]); self.nb=take
            for tensor in (self.rk,self.rv,self.rs,self.ri): tensor[:self.window].copy_(tensor[n:self.nr].clone())
            self.nr=self.window; self.writes+=1
        return output
    def block(self,q,k,v,s):
        return torch.stack([self.step(q[i],k[i],v[i],s[i]) for i in range(len(q))])


def causal_block_operator():
    gen=torch.Generator().manual_seed(371); n,d=512,16
    q=torch.randn(n,d,generator=gen); k=torch.randn(n,d,generator=gen); v=torch.randn(n,d,generator=gen)
    scores=torch.rand(n,generator=gen,dtype=torch.float64)
    # Priority depends only on provided, already-arrived data in deployment.
    # Random priorities here intentionally do not pretend to be useful salience.
    one=CausalBlockKV(32,16,8,d); r1=[]; footprint={}
    for i in range(n):
        r1.append(one.step(q[i],k[i],v[i],scores[i]))
        if i+1 in (128,n): footprint[str(i+1)]=one.state_bytes()
    r1=torch.stack(r1)
    chunked=CausalBlockKV(32,16,8,d); pieces=[]; start=0
    for width in [7,17,31,2,89,101,265]:
        end=min(n,start+width)
        if end>start: pieces.append(chunked.block(q[start:end],k[start:end],v[start:end],scores[start:end]))
        start=end
    if start<n: pieces.append(chunked.block(q[start:],k[start:],v[start:],scores[start:]))
    r2=torch.cat(pieces)
    alternate=CausalBlockKV(32,16,8,d); q2=q.clone(); k2=k.clone(); v2=v.clone()
    q2[128:]=torch.randn(n-128,d,generator=gen); k2[128:]=torch.randn(n-128,d,generator=gen); v2[128:]=torch.randn(n-128,d,generator=gen)
    ra=alternate.block(q2,k2,v2,scores)
    exact=CausalBlockKV(n,16,8,d); all_kept=exact.block(q,k,v,scores)
    full=attn(q,k,v,torch.ones(n,n,dtype=torch.bool).tril())
    result={'token_vs_external_chunks':metrics(r2,r1),
            'same_prefix_changed_suffix':metrics(ra[:128],r1[:128]),
            'all_tokens_fit_vs_full_attention':metrics(all_kept,full),
            'fixed_capacity_footprint':footprint,'state_growth':footprint[str(n)]/footprint['128'],
            'logical_bank_commits':one.writes,'token_evictions_if_updated_one_by_one':n-16,
            'scope':'single-head preprojected tensors, fixed position regime; no pretrained model; writer quality not measured'}
    assert torch.equal(r1,r2)
    assert torch.equal(r1[:128],ra[:128])
    assert result['all_tokens_fit_vs_full_attention']['relative_l2']<1e-5
    return result


def memory_reclamation_geometry():
    L,H,d,W,B,C,G,P=32,32,96,4096,512,768,128,32
    parts={'recent_KV':L*H*W*d*4,'recent_raw_K':L*H*W*d*2,
           'per_layer_KV_scratch':L*H*(W+B)*d*4,'per_layer_rawK_scratch':L*H*(W+B)*d*2,
           'bank_KV':L*H*(C+G+P)*d*8,
           'bank_scores_ages':L*H*C*12+L*H*P*4,
           'recurrence':L*H*16*4*(d*4+4+8),'background_counts':L*H*8,'rng':512}
    measured=5899964928
    candidate=[]
    for c,bits in [(768,16),(4096,16),(8192,16),(16384,8),(32768,4)]:
        pair=4*d if bits==16 else 2*(d*bits//8+d//32*4)
        historical=L*H*(c+G+P)*pair
        metadata=L*H*c*12+L*H*P*4+L*H*8+512
        recent=L*H*W*d*4; shared=H*(W+B)*d*4
        subtotal = historical + metadata + recent + shared
        # Additional conservative allowance for the NEW causal block operator:
        # an entire BF16 pending block plus score/ID fields, scores/IDs for the
        # recent region, and an optional 32-token exact prefix. The legacy
        # pending32 and background remain counted above rather than subtracted.
        causal_allowance = L*H*B*(4*d+12) + L*H*W*12 + L*H*32*d*4
        candidate.append({'foreground_slots':c,'KV_bits':bits,'history_bytes':historical,
                          'score_id_bytes':metadata,'recent_bytes':recent,'shared_workspace_bytes':shared,
                          'subtotal_bytes':subtotal,
                          'additional_causal_pending_recent_metadata_prefix_bytes':causal_allowance,
                          'with_causal_allowance_bytes':subtotal+causal_allowance,
                          'note':'budget estimate only; retains current background/pending counts conservatively; separate allowance for new block operator; excludes weights and all other engine buffers'})
    logical=3821079552
    writer64=L*(3072*64+64+64*H+H)
    return {'reported_request_bytes':measured,'tensor_geometry_bytes':parts,
            'geometry_sum':sum(parts.values()),'unattributed_difference_bytes':measured-sum(parts.values()),
            'candidate_budgets':candidate,'width64_writer_parameters':writer64,
            'width64_writer_fraction':writer64/logical,
            'scope':'shape-derived accounting, not a profiler trace and not measured post-change GPU allocation'}

def main():
    report={'date':'2026-09-13','runtime':{'torch':torch.__version__,'device':'cpu','threads':torch.get_num_threads()},
            'scope':'No pretrained model loaded; no GPU/serving scores; no remote writes.', 'experiments':{}}
    jobs=[('causal_block_operator',causal_block_operator),('memory_reclamation',memory_reclamation_geometry),('native_cache_rebuild',native_rebuild_protocol),('rounding_ladder',attention_arithmetic_ladder),('quantization_split',quantization_error_split),('finite_information',finite_information_bound),('longrope',longrope_experiments),('mask_and_mass',masking_and_normalization),
          ('storage',storage_experiments),('million_stream',million_stream),
          ('native_transition',toy_native_length_transition),('feasibility',feasibility),
          ('background_candidate',gaussian_background)]
    for name,job in jobs:
        start=time.monotonic(); value=job(); report['experiments'][name]=value
        report['experiments'][name]['cpu_elapsed_seconds']=time.monotonic()-start
        (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(name,json.dumps(value,ensure_ascii=False),flush=True)
    print('Saved',OUT/'results.json')
if __name__=='__main__': main()
