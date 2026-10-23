"""Checks for preparation and a single temporary teacher move."""

import pytest
import torch

from torch_topological.nn.qihotogl import (
    prepare_filtration_teacher,
    single_step_filtration_teacher,
    validate_current_filtration,
)


def prepare_example(moment_method="exact"):
    edges = torch.tensor([[0, 0], [1, 2]])
    scores = torch.tensor([0., 0.2, 0.4], dtype=torch.float64)
    return prepare_filtration_teacher(
        scores, edges,
        k=6, q_values=(0, 1), z_max=4,
        moment_method=moment_method,
        n_samples=1000, seed=42,
    )


def positive_gamma(context):
    phi = context["phi"]
    gamma = torch.zeros_like(phi)

    for m in range(1, context["t"].numel()):
        level = int(context["t"][context["sigma"][m]].item())
        if level < context["k"]:
            change = phi[level - 2] - phi[level - 1]
            if change.square().sum().item() > 0:
                gamma[level - 1] = change
                return gamma

    raise AssertionError("Expected a nonzero proposal change.")


def test_exact_preparation_and_one_swap():
    context = prepare_example()
    assert context["t"].tolist() == [1, 3, 4]
    assert context["phi"].shape == (6, 8)

    snapshots = {
        key: context[key].clone()
        for key in ("t", "sigma", "phi")
    }

    result = single_step_filtration_teacher(
        context, positive_gamma(context)
    )

    m, direction, score = result["selected"]
    assert (m, direction) == (1, "+")
    assert score == pytest.approx(1.0)
    assert result["move_type"] == "swap"
    assert result["target_t"].tolist() == [1, 4, 3]
    assert torch.equal(
        result["target_sigma"],
        validate_current_filtration(result["target_t"], 6),
    )
    assert torch.equal(
        result["displacement"],
        result["target_t"] - context["t"],
    )

    for key, before in snapshots.items():
        assert torch.equal(context[key], before)


def test_zero_gradient_returns_independent_unchanged_target():
    context = prepare_example()
    result = single_step_filtration_teacher(
        context, torch.zeros_like(context["phi"])
    )

    assert result["selected"] is None
    assert result["move_type"] == "none"
    assert torch.equal(result["target_t"], context["t"])
    assert torch.equal(result["target_sigma"], context["sigma"])
    assert torch.count_nonzero(result["displacement"]).item() == 0

    # The returned target must not alias the current assignment.
    result["target_t"][0] = 99
    assert context["t"][0].item() == 1


def test_monte_carlo_preparation_and_selection():
    context = prepare_example("monte_carlo")
    repeated = prepare_example("monte_carlo")

    assert context["moment_method"] == "monte_carlo"
    assert torch.equal(context["phi"], repeated["phi"])
    assert torch.isfinite(context["phi"]).all().item()

    result = single_step_filtration_teacher(
        context, positive_gamma(context)
    )
    assert result["selected"] is not None
    assert result["selected"][2] > 0
    assert torch.equal(
        result["target_sigma"],
        validate_current_filtration(result["target_t"], 6),
    )


def test_direct_realisation_in_teacher():
    # Controlled row data isolates the wrapper's direct-move branch.
    # These rows are a synthetic selector fixture, not Hodge moments.
    t = torch.tensor([1, 3, 5])
    phi = torch.tensor([[0.], [0.], [1.], [1.], [1.], [1.]])
    gamma = torch.zeros_like(phi)
    gamma[1] = 1.0

    context = {
        "t": t,
        "sigma": validate_current_filtration(t, 6),
        "phi": phi,
        "k": 6,
    }
    result = single_step_filtration_teacher(context, gamma)

    assert result["selected"] == (1, "-", 1.0)
    assert result["move_type"] == "direct"
    assert result["target_t"].tolist() == [1, 2, 5]
    assert t.tolist() == [1, 3, 5]


def test_rank_baseline_and_empty_candidate_set():
    context = prepare_filtration_teacher(
        torch.tensor([0.]),
        torch.empty((2, 0), dtype=torch.long),
        k=1, q_values=(0,), z_max=2,
        decoder_mode="rank",
    )
    result = single_step_filtration_teacher(
        context, torch.ones_like(context["phi"])
    )
    assert context["t"].tolist() == [1]
    assert result["selected"] is None
    assert result["move_type"] == "none"


def test_unknown_moment_method_rejected():
    with pytest.raises(ValueError):
        prepare_filtration_teacher(
            torch.tensor([0., 1.]),
            torch.tensor([[0], [1]]),
            k=2, q_values=(0,), z_max=2,
            moment_method="unknown",
        )
