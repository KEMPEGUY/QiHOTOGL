"""Tests for experimental M2/M3 surrogate signals."""

import pytest
import torch
import torch.nn.functional as F

from torch_topological.nn.qihotogl import (
    filtration_surrogate_signal,
    prepare_filtration_teacher,
    single_step_filtration_teacher,
)


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("M2", [0., -2., 0.]),
        ("M3", [0., -2., 2.]),
    ],
)
def test_swap_signal(mode, expected):
    scores = torch.tensor([0., 0.2, 0.4], dtype=torch.float64)
    context = {
        "t": torch.tensor([1, 3, 4]),
        "sigma": torch.tensor([0, 1, 2]),
    }
    teacher = {
        "selected": (1, "+", 2.0),
        "displacement": torch.tensor([0, 1, -1]),
    }

    signal = filtration_surrogate_signal(
        scores, context, teacher, mode
    )
    assert signal.tolist() == expected
    assert not signal.requires_grad


@pytest.mark.parametrize("mode", ["M2", "M3"])
def test_direct_decrement_and_no_candidate(mode):
    scores = torch.tensor([0., 0.2, 0.4])
    context = {
        "t": torch.tensor([1, 3, 5]),
        "sigma": torch.tensor([0, 1, 2]),
    }
    direct = {
        "selected": (1, "-", 2.0),
        "displacement": torch.tensor([0, -1, 0]),
    }
    signal = filtration_surrogate_signal(
        scores, context, direct, mode
    )
    assert signal.tolist() == [0., 2., 0.]

    stopped = {
        "selected": None,
        "displacement": torch.zeros(3, dtype=torch.long),
    }
    signal = filtration_surrogate_signal(
        scores, context, stopped, mode
    )
    assert torch.count_nonzero(signal).item() == 0


@pytest.mark.parametrize("mode", ["M2", "M3"])
def test_original_vertex_index_routing(mode):
    scores = torch.tensor([0.4, 0., 0.2])
    context = {
        "t": torch.tensor([4, 1, 3]),
        "sigma": torch.tensor([1, 2, 0]),
    }
    teacher = {
        "selected": (1, "+", 2.0),
        "displacement": torch.tensor([-1, 0, 1]),
    }
    signal = filtration_surrogate_signal(
        scores, context, teacher, mode
    )
    expected = (
        [0., 0., -2.] if mode == "M2"
        else [2., 0., -2.]
    )
    assert signal.tolist() == expected


@pytest.mark.parametrize("mode", ["M2", "M3"])
def test_classification_teacher_scorer_connection(mode):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)

        # Shared neural scorer: every vertex uses the same weight.
        x = torch.tensor(
            [[0.], [0.2], [0.4], [0.7]],
            dtype=torch.float64,
        )
        scorer = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            scorer.weight.fill_(1.0)

        scores = scorer(x).flatten()
        context = prepare_filtration_teacher(
            scores,
            torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]]),
            k=8, q_values=(0, 1), z_max=4,
            moment_method="exact",
        )
        current_t = context["t"].clone()

        phi = context["phi"].detach().clone().requires_grad_(True)
        classifier = torch.nn.Linear(
            phi.numel(), 2, dtype=phi.dtype
        )
        logits = classifier(phi.reshape(1, -1))
        loss = F.cross_entropy(logits, torch.tensor([1]))
        loss.backward()

        # Classification has not crossed the hard decoder.
        assert scorer.weight.grad is None

        teacher = single_step_filtration_teacher(
            context, -phi.grad.detach()
        )
        assert teacher["selected"] is not None
        assert teacher["selected"][2] > 0

        signal = filtration_surrogate_signal(
            scores, context, teacher, mode
        )

        # For s_i = w*x_i, the injected parameter gradient is sum g_i*x_i.
        expected_gradient = (signal * x.flatten()).sum()
        scores.backward(gradient=signal)

        assert scorer.weight.grad is not None
        assert torch.isfinite(scorer.weight.grad).all().item()
        assert torch.allclose(
            scorer.weight.grad.reshape(()), expected_gradient
        )
        assert expected_gradient.abs().item() > 0

        optimizer = torch.optim.SGD(scorer.parameters(), lr=0.01)
        before = scorer.weight.detach().clone()
        optimizer.step()

        assert torch.allclose(
            scorer.weight.detach(),
            before - 0.01 * expected_gradient,
        )
        assert torch.equal(context["t"], current_t)


def test_unknown_surrogate_mode_rejected():
    with pytest.raises(ValueError):
        filtration_surrogate_signal(
            torch.zeros(2),
            {"t": torch.tensor([1, 2])},
            {"selected": None},
            mode="unknown",
        )
