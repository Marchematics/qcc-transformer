import torch

from qcc_transformer.admission_training import (
    balanced_admission_loss,
    salience_binary_labels,
    sampled_future_attention_salience,
)


def test_future_attention_salience_finds_long_range_retrieval_key() -> None:
    query = torch.zeros(1, 1, 6, 2)
    key = torch.zeros_like(query)
    key[0, 0, 0] = torch.tensor([1.0, 0.0])
    key[0, 0, 1:] = torch.tensor([0.0, 1.0])
    query[0, 0, 4:] = torch.tensor([10.0, 0.0])
    salience = sampled_future_attention_salience(
        query,
        key,
        window_size=2,
        num_queries=2,
        topk=1,
    )
    assert int(salience[0, 0].argmax()) == 0
    assert float(salience[0, 0, 0]) > 0.0


def test_salience_labels_keep_fixed_positive_budget() -> None:
    salience = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.0]]])
    labels = salience_binary_labels(
        salience, positive_fraction=0.4, min_positive=1
    )
    assert int(labels.sum()) == 2
    assert labels[0, 0, 1] == 1
    assert labels[0, 0, 3] == 1


def test_balanced_admission_loss_is_finite_and_differentiable() -> None:
    logits = torch.zeros(2, 3, 5, requires_grad=True)
    labels = torch.zeros_like(logits)
    labels[..., 0] = 1
    loss = balanced_admission_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
