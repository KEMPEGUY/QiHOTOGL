"""Tests for Monte Carlo moment matrices across filtration levels."""

import math

import torch

from torch_topological.nn.qihotogl import (
    exact_hodge_moment_matrices,
    monte_carlo_hodge_moment_matrices,
)


def triangle_edges():
    return torch.tensor([[0, 0, 1], [1, 2, 2]])


def test_shapes_gaps_and_absent_dimension():
    t = torch.tensor([1, 3, 5])
    original = t.clone()

    matrices = monte_carlo_hodge_moment_matrices(
        triangle_edges(), t,
        k=6, q_values=(0, 1, 2, 3),
        z_max=4, n_samples=1000, seed=42,
    )

    assert list(matrices) == [0, 1, 2, 3]

    for matrix in matrices.values():
        assert matrix.shape == (6, 4)
        assert torch.isfinite(matrix).all().item()
        assert torch.equal(matrix[0], matrix[1])
        assert torch.equal(matrix[2], matrix[3])
        assert torch.equal(matrix[4], matrix[5])

    assert torch.equal(
        matrices[3], torch.zeros(6, 4, dtype=torch.float64)
    )
    assert torch.equal(
        matrices[0][0], torch.ones(4, dtype=torch.float64)
    )
    assert torch.equal(t, original)


def test_reproducibility_and_dimension_order():
    edges = triangle_edges()
    t = torch.tensor([1, 3, 5])
    options = dict(
        k=6, z_max=4, n_samples=1000, seed=42,
    )

    first = monte_carlo_hodge_moment_matrices(
        edges, t, q_values=(0, 1), **options
    )
    repeated = monte_carlo_hodge_moment_matrices(
        edges, t, q_values=(0, 1), **options
    )
    reversed_order = monte_carlo_hodge_moment_matrices(
        edges, t, q_values=(1, 0), **options
    )

    assert list(reversed_order) == [1, 0]
    for q in first:
        assert torch.equal(first[q], repeated[q])
        assert torch.equal(first[q], reversed_order[q])


def test_provisional_ties_and_original_labels():
    edges = triangle_edges()

    # Vertices 1 and 2 arrive together.
    tied = torch.tensor([1, 3, 3])
    matrices = monte_carlo_hodge_moment_matrices(
        edges, tied,
        k=6, q_values=(0, 1),
        z_max=4, n_samples=1000, seed=42,
    )
    assert torch.equal(matrices[0][0], matrices[0][1])
    assert torch.equal(matrices[0][2], matrices[0][-1])

    # Vertex indices need not follow arrival order.
    reordered = torch.tensor([5, 1, 3])
    matrices = monte_carlo_hodge_moment_matrices(
        edges, reordered,
        k=6, q_values=(0,),
        z_max=4, n_samples=1000, seed=42,
    )
    assert torch.equal(
        matrices[0][0], torch.ones(4, dtype=torch.float64)
    )
    assert reordered.tolist() == [5, 1, 3]


def test_matrices_against_exact():
    # Successive complexes: vertex, edge, filled triangle.
    # Their reflected operators have column norms at most one,
    # so every signed walk contribution lies in [-1, 1].
    edges = triangle_edges()
    t = torch.tensor([1, 3, 5])
    q_values = (0, 1, 2, 3)
    k = 6
    z_max = 4
    n_samples = 20000

    estimated = monte_carlo_hodge_moment_matrices(
        edges, t,
        k=k, q_values=q_values, z_max=z_max,
        n_samples=n_samples, seed=42,
    )
    exact = exact_hodge_moment_matrices(
        edges, t, k=k, q_values=q_values, z_max=z_max,
    )

    # Hoeffding bound with a union bound across all matrix entries.
    # This tolerance applies to this bounded-contribution example.
    comparisons = len(q_values) * k * z_max
    failure_budget = 1e-6
    tolerance = math.sqrt(
        2.0 * math.log(2.0 * comparisons / failure_budget)
        / n_samples
    )

    for q in q_values:
        error = (estimated[q] - exact[q]).abs()
        assert torch.all(error <= tolerance).item(), (
            f"q={q}: maximum error {error.max().item()}, "
            f"tolerance {tolerance}"
        )
