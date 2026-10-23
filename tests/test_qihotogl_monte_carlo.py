"""Checks for sparse reflected Hodge operators and signed walks."""

import pytest
import torch

from torch_topological.nn.qihotogl import (
    filtered_clique_simplices,
    oriented_boundary_matrix,
    sparse_reflected_hodge_columns,
    prepare_hodge_walk_table,
    estimate_hodge_walk_moments,
    exact_hodge_moments_at_level,
)


def make_table(edges, t, level, q):
    sparse = sparse_reflected_hodge_columns(
        edges, t, level=level, q=q
    )
    return prepare_hodge_walk_table(sparse)


def dense_reflected_operator(edges, t, level, q):
    simplices = filtered_clique_simplices(
        edges, t, level=level, max_dim=q + 1
    )
    dimension = len(simplices[q])
    if dimension == 0:
        return torch.zeros((0, 0), dtype=torch.float64)

    bq = oriented_boundary_matrix(simplices, q)
    bnext = oriented_boundary_matrix(simplices, q + 1)
    laplacian = bq.T @ bq + bnext @ bnext.T

    return (
        torch.eye(dimension, dtype=torch.float64)
        - laplacian / len(simplices[0])
    )


@pytest.mark.parametrize("q", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "pairs, levels, level",
    [
        ([], [1, 3, 5], 5),
        ([(0, 1), (0, 2)], [1, 2, 3], 3),
        (
            [(0, 1), (0, 2), (1, 2), (2, 3)],
            [1, 3, 5, 7],
            7,
        ),
        (
            [(0, 1), (0, 2), (1, 2), (2, 3)],
            [5, 1, 3, 7],
            3,
        ),
        (
            [(0, 1), (0, 2), (0, 3),
             (1, 2), (1, 3), (2, 3)],
            [1, 2, 3, 4],
            4,
        ),
    ],
)
def test_sparse_operator_matches_dense(pairs, levels, level, q):
    edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T
    t = torch.tensor(levels)
    sparse = sparse_reflected_hodge_columns(
        edges, t, level=level, q=q
    )
    expected = dense_reflected_operator(edges, t, level, q)
    actual = torch.zeros_like(expected)

    for column, (rows, weights) in enumerate(
        zip(sparse["columns"], sparse["weights"])
    ):
        actual[rows, column] = weights

    assert torch.allclose(actual, expected, atol=1e-12, rtol=0)


def test_walk_table_probabilities_and_signs():
    edges = torch.tensor([[0, 0], [1, 2]])
    t = torch.tensor([1, 2, 3])
    sparse = sparse_reflected_hodge_columns(
        edges, t, level=3, q=1
    )
    table = prepare_hodge_walk_table(sparse)

    assert table["nstates"] == 2
    assert (table["signs"] < 0).any().item()

    for column in range(table["nstates"]):
        lo = int(table["offsets"][column].item())
        hi = int(table["offsets"][column + 1].item())
        cdf = table["cdf"][lo:hi]
        probabilities = torch.diff(
            torch.cat((torch.zeros(1, dtype=cdf.dtype), cdf))
        )

        assert cdf[-1].item() == 1.0
        assert (probabilities >= 0).all().item()

        # Probability times signed column norm reconstructs H.
        reconstructed = torch.zeros(2, dtype=torch.float64)
        reconstructed[table["destinations"][lo:hi]] = (
            probabilities
            * table["signs"][lo:hi]
            * table["column_norms"][column]
        )
        expected = torch.zeros(2, dtype=torch.float64)
        expected[sparse["columns"][column]] = sparse["weights"][column]
        assert torch.allclose(
            reconstructed, expected, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize(
    "pairs, levels, level, q, expected",
    [
        ([], [1, 3, 5], 5, 0, 1.0),
        ([(0, 1)], [1, 2], 2, 1, 0.0),
        ([(0, 1)], [1, 2], 1, 1, 0.0),
        (
            [(0, 1), (0, 2), (1, 2)],
            [1, 2, 3],
            3,
            2,
            0.0,
        ),
    ],
)
def test_deterministic_moments(pairs, levels, level, q, expected):
    edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T
    table = make_table(edges, torch.tensor(levels), level, q)
    result = estimate_hodge_walk_moments(
        table, z_max=4, n_samples=1000, seed=42
    )
    assert torch.equal(
        result, torch.full((4,), expected, dtype=torch.float64)
    )


def test_fixed_seed_reproducibility():
    edges = torch.tensor([[0, 0], [1, 2]])
    table = make_table(edges, torch.tensor([1, 2, 3]), 3, 1)

    first = estimate_hodge_walk_moments(
        table, z_max=4, n_samples=2000, seed=42
    )
    second = estimate_hodge_walk_moments(
        table, z_max=4, n_samples=2000, seed=42
    )
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "pairs, levels, q",
    [
        ([(0, 1), (0, 2)], [1, 2, 3], 1),
        ([(0, 1), (0, 2), (1, 2)], [1, 2, 3], 0),
    ],
)
def test_monte_carlo_against_exact(pairs, levels, q):
    edges = torch.tensor(pairs, dtype=torch.long).T
    t = torch.tensor(levels)
    level = max(levels)
    table = make_table(edges, t, level, q)
    n_samples = 20000

    estimate = estimate_hodge_walk_moments(
        table, z_max=4, n_samples=n_samples, seed=42
    )
    exact = exact_hodge_moments_at_level(
        edges, t, level=level, q=q, z_max=4
    )

    # In these specific examples every column norm is at most one.
    # Thus each signed walk contribution lies in [-1, 1].
    assert (table["column_norms"] <= 1.0 + 1e-12).all().item()

    # Hoeffding bound, union over four orders, failure budget 1e-6.
    tolerance = (
        2.0 * torch.log(torch.tensor(8.0 / 1e-6)) / n_samples
    ).sqrt().item()

    assert torch.isfinite(estimate).all().item()
    assert torch.all(
        (estimate - exact).abs() <= tolerance
    ).item()
