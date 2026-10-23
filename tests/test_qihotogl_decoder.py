"""Tests for hard rank and score-gap filtration decoders."""

import itertools

import pytest
import torch

from torch_topological.nn.qihotogl import (
    decode_filtration_scores,
    validate_current_filtration,
)


@pytest.mark.parametrize("mode", ["rank", "score_gaps"])
def test_decoder_invariants(mode):
    for n in range(1, 5):
        for k in range(n, n + 4):
            for values in itertools.product(
                (-2.0, 0.0, 0.4, 2.0), repeat=n
            ):
                scores = torch.tensor(values, dtype=torch.float64)
                original = scores.clone()
                t, sigma = decode_filtration_scores(
                    scores, k, mode=mode
                )

                assert t.dtype == torch.long
                assert t.device == scores.device
                assert torch.equal(
                    sigma, torch.argsort(scores, stable=True)
                )
                assert torch.equal(
                    sigma, validate_current_filtration(t, k)
                )
                assert torch.equal(scores, original)

                if mode == "rank" or k == n:
                    assert t[sigma].tolist() == list(
                        range(1, n + 1)
                    )


def test_known_gap_assignment():
    scores = torch.tensor([0.8, 0.1, 0.4], dtype=torch.float64)
    t, sigma = decode_filtration_scores(scores, 6)

    assert t.tolist() == [5, 1, 3]
    assert sigma.tolist() == [1, 2, 0]
    assert t[sigma].tolist() == [1, 3, 5]


def test_stable_ties_and_single_vertex():
    t, sigma = decode_filtration_scores(torch.ones(3), 6)
    assert t.tolist() == [1, 2, 3]
    assert sigma.tolist() == [0, 1, 2]

    t, sigma = decode_filtration_scores(torch.tensor([-7.]), 6)
    assert t.tolist() == [1]
    assert sigma.tolist() == [0]


def test_all_small_labeled_assignments_are_representable():
    # Construct scores strictly inside the desired rounding bins.
    for n in range(1, 5):
        for k in range(n, 8):
            extra = k - n
            for rest in itertools.combinations(
                range(2, k + 1), n - 1
            ):
                levels = (1,) + rest

                for order in itertools.permutations(range(n)):
                    scores = torch.empty(n, dtype=torch.float64)
                    expected = torch.empty(n, dtype=torch.long)

                    for m, vertex in enumerate(order):
                        offset = levels[m] - 1 - m
                        scores[vertex] = (
                            float(m) if extra == 0
                            else (offset + 0.1 * m / n) / extra
                        )
                        expected[vertex] = levels[m]

                    t, sigma = decode_filtration_scores(scores, k)
                    assert torch.equal(t, expected)
                    assert sigma.tolist() == list(order)


def test_tau_scaling_and_attained_maximum():
    scores = torch.tensor([0., 0.3, 0.7], dtype=torch.float64)
    first, _ = decode_filtration_scores(scores, 6, tau=1.0)
    second, _ = decode_filtration_scores(
        scores * 2, 6, tau=2.0
    )
    assert torch.equal(first, second)

    t, _ = decode_filtration_scores(
        torch.tensor([0., 0.5, 2.]), 6
    )
    assert t.max().item() == 6


def test_score_exchange_reproduces_swap():
    scores = torch.tensor([0., 0.2, 0.4], dtype=torch.float64)

    for mode in ("rank", "score_gaps"):
        before, _ = decode_filtration_scores(scores, 6, mode)
        exchanged = scores[[0, 2, 1]]
        after, _ = decode_filtration_scores(exchanged, 6, mode)

        assert torch.equal(after, before[[0, 2, 1]])


def test_input_preserved_and_output_detached():
    scores = torch.tensor([0.8, 0.1, 0.4], requires_grad=True)
    original = scores.detach().clone()

    t, sigma = decode_filtration_scores(scores, 6)

    assert torch.equal(scores.detach(), original)
    assert not t.requires_grad
    assert not sigma.requires_grad


@pytest.mark.parametrize(
    "scores, k, options, error",
    [
        ([0., 1.], 2, {}, TypeError),
        (torch.tensor([]), 2, {}, ValueError),
        (torch.zeros(2, 2), 4, {}, ValueError),
        (torch.tensor([0, 1]), 2, {}, TypeError),
        (torch.tensor([0., float("nan")]), 2, {}, ValueError),
        (torch.tensor([0., float("inf")]), 2, {}, ValueError),
        (torch.tensor([0., 1.]), 1, {}, ValueError),
        (torch.tensor([0., 1.]), True, {}, TypeError),
        (torch.tensor([0., 1.]), 2.0, {}, TypeError),
        (torch.tensor([0., 1.]), 2, {"mode": "unknown"}, ValueError),
        (torch.tensor([0., 1.]), 2, {"tau": 0}, ValueError),
        (torch.tensor([0., 1.]), 2, {"tau": -1}, ValueError),
        (
            torch.tensor([0., 1.]), 2,
            {"tau": float("nan")}, ValueError,
        ),
        (torch.tensor([0., 1.]), 2, {"tau": True}, TypeError),
    ],
)
def test_invalid_inputs(scores, k, options, error):
    with pytest.raises(error):
        decode_filtration_scores(scores, k, **options)
