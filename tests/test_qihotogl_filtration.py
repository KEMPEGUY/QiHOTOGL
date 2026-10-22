"""Checks for QiHOTOGL filtration validation, selection and realisation."""

import pytest
import torch

from torch_topological.nn.qihotogl import (
    validate_current_filtration,
    realise_filtration_direction,
    select_filtration_direction,
)


@pytest.mark.parametrize(
    "values,k,expected",
    [
        ([5, 1, 3], 6, [1, 2, 0]),
        ([3, 1, 2], 3, [1, 2, 0]),
        ([1], 4, [0]),
    ],
)
def test_valid_filtrations(values, k, expected):
    t = torch.tensor(values)
    original = t.clone()
    sigma = validate_current_filtration(t, k)
    assert sigma.tolist() == expected
    assert torch.equal(t, original)


@pytest.mark.parametrize(
    "values,k",
    [
        ([1, 1, 3], 4),
        ([2, 3, 4], 4),
        ([1, 3, 7], 6),
        ([1, 2, 3], 2),
        ([1.0, 2.0], 2),
    ],
)
def test_invalid_filtrations(values, k):
    with pytest.raises((TypeError, ValueError)):
        validate_current_filtration(torch.tensor(values), k)


@pytest.mark.parametrize(
    "values,m,direction,expected,kind",
    [
        ([5, 1, 3], 1, "-", [5, 1, 2], "direct"),
        ([5, 1, 3], 1, "+", [5, 1, 4], "direct"),
        ([4, 1, 3], 2, "-", [3, 1, 4], "swap"),
        ([4, 1, 3], 1, "+", [3, 1, 4], "swap"),
    ],
)
def test_realisation(values, m, direction, expected, kind):
    t = torch.tensor(values)
    sigma = validate_current_filtration(t, 6)
    old_t, old_sigma = t.clone(), sigma.clone()

    result, ordering, move_type = realise_filtration_direction(
        t, sigma, m, direction, 6
    )

    assert result.tolist() == expected
    assert move_type == kind
    assert torch.equal(ordering, validate_current_filtration(result, 6))
    assert torch.equal(t, old_t)
    assert torch.equal(sigma, old_sigma)


@pytest.mark.parametrize(
    "values,m,direction,k",
    [
        ([1, 3, 5], 0, "+", 6),
        ([1, 2, 5], 1, "-", 6),
        ([1, 3, 6], 2, "+", 6),
    ],
)
def test_inadmissible_directions(values, m, direction, k):
    t = torch.tensor(values)
    sigma = validate_current_filtration(t, k)
    with pytest.raises(ValueError):
        realise_filtration_direction(t, sigma, m, direction, k)


def test_selection_then_swap():
    t = torch.tensor([4, 1, 3])
    sigma = validate_current_filtration(t, 5)
    phi = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8]])
    gamma = torch.tensor([[0.0], [0.0], [-1.0], [0.0], [0.0]])

    m, direction, score = select_filtration_direction(
        t, sigma, phi, gamma, 5
    )

    assert (m, direction) == (1, "+")
    assert score == pytest.approx(0.2)

    result, ordering, kind = realise_filtration_direction(
        t, sigma, m, direction, 5
    )
    assert result.tolist() == [3, 1, 4]
    assert ordering.tolist() == [1, 0, 2]
    assert kind == "swap"


def test_selection_tie_breaking():
    t = torch.tensor([4, 1, 3])
    sigma = validate_current_filtration(t, 5)
    phi = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]])
    gamma = torch.tensor([[0.0], [1.0], [-1.0], [0.0], [0.0]])

    assert select_filtration_direction(
        t, sigma, phi, gamma, 5
    ) == (1, "-", 1.0)


def test_no_positive_score():
    t = torch.tensor([4, 1, 3])
    sigma = validate_current_filtration(t, 5)
    phi = torch.zeros(5, 1)
    assert select_filtration_direction(
        t, sigma, phi, torch.zeros_like(phi), 5
    ) is None


@pytest.mark.parametrize("values,k", [([1], 1), ([1, 2], 2)])
def test_empty_candidate_set(values, k):
    t = torch.tensor(values)
    sigma = validate_current_filtration(t, k)
    phi = torch.zeros(k, 1)
    assert select_filtration_direction(
        t, sigma, phi, torch.ones_like(phi), k
    ) is None
