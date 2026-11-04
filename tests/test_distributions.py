import torch

from core.agent.distributions import MaskedCategorical


def test_masked_actions_have_zero_probability_and_are_never_sampled():
    torch.manual_seed(0)
    logits = torch.randn(64, 9) * 5
    mask = torch.rand(64, 9) > 0.5
    mask[:, 0] = True  # guarantee a legal action per row
    dist = MaskedCategorical(logits, mask)

    assert torch.all(dist.probs[~mask] == 0)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(64))
    samples = dist.sample((2000,))  # (2000, 64)
    assert mask.gather(1, samples.T).all()


def test_masking_prefers_legal_action_even_with_huge_illegal_logit():
    logits = torch.tensor([[1e6, 0.0, 0.0]])
    mask = torch.tensor([[False, True, True]])
    dist = MaskedCategorical(logits, mask)
    assert dist.probs[0, 0] == 0
    assert torch.allclose(dist.probs[0, 1:], torch.tensor([0.5, 0.5]))


def test_entropy_is_finite_and_matches_uniform_over_legal():
    logits = torch.zeros(1, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False, False]])
    dist = MaskedCategorical(logits, mask)
    ent = dist.entropy()
    assert torch.allclose(ent, torch.log(torch.tensor([2.0])))
    ent.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_single_legal_action_is_deterministic():
    dist = MaskedCategorical(torch.randn(3, 5), torch.eye(5, dtype=torch.bool)[[1, 3, 4]])
    assert dist.sample().tolist() == [1, 3, 4]
    assert torch.allclose(dist.entropy(), torch.zeros(3))


def test_all_masked_row_raises():
    try:
        MaskedCategorical(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
