"""Known-answer checks for exact Hodge moment calculations."""

import torch

from torch_topological.nn.qihotogl import (
    filtered_clique_simplices,
    oriented_boundary_matrix,
    exact_hodge_moments_at_level,
    exact_hodge_moment_matrices,
)


def triangle():
    edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
    t = torch.tensor([1, 3, 5])
    return edges, t


def test_triangle_boundary_signs_and_chain_identity():
    edges, t = triangle()
    simplices = filtered_clique_simplices(edges, t, 5, 2)

    B0 = oriented_boundary_matrix(simplices, 0)
    B1 = oriented_boundary_matrix(simplices, 1)
    B2 = oriented_boundary_matrix(simplices, 2)

    expected_B1 = torch.tensor([
        [-1., -1.,  0.],
        [ 1.,  0., -1.],
        [ 0.,  1.,  1.],
    ], dtype=torch.float64)
    expected_B2 = torch.tensor([[1.], [-1.], [1.]], dtype=torch.float64)

    assert B0.shape == (0, 3)
    assert torch.equal(B1, expected_B1)
    assert torch.equal(B2, expected_B2)
    assert torch.equal(B1 @ B2, torch.zeros(3, 1, dtype=torch.float64))


def test_q0_graph_laplacian():
    edges, t = triangle()
    simplices = filtered_clique_simplices(edges, t, 5, 1)
    B1 = oriented_boundary_matrix(simplices, 1)

    expected = torch.tensor([
        [ 2., -1., -1.],
        [-1.,  2., -1.],
        [-1., -1.,  2.],
    ], dtype=torch.float64)

    assert torch.equal(B1 @ B1.T, expected)


def test_empty_boundary_shapes():
    edges, t = triangle()
    simplices = filtered_clique_simplices(edges, t, 1, 2)

    assert oriented_boundary_matrix(simplices, 0).shape == (0, 1)
    assert oriented_boundary_matrix(simplices, 1).shape == (1, 0)
    assert oriented_boundary_matrix(simplices, 2).shape == (0, 0)


def test_nonempty_zero_laplacian_moments():
    edges = torch.empty((2, 0), dtype=torch.long)
    t = torch.tensor([1, 2, 3])
    result = exact_hodge_moments_at_level(edges, t, 3, 0, 4)

    assert torch.equal(result, torch.ones(4, dtype=torch.float64))


def test_empty_chain_space_moments():
    edges, t = triangle()
    result = exact_hodge_moments_at_level(edges, t, 1, 1, 4)

    assert torch.equal(result, torch.zeros(4, dtype=torch.float64))


def test_full_triangle_q0_moments():
    edges, t = triangle()
    result = exact_hodge_moments_at_level(edges, t, 5, 0, 4)
    expected = torch.full((4,), 1.0 / 3.0, dtype=torch.float64)

    assert torch.allclose(result, expected, atol=1e-12, rtol=0)


def test_full_triangle_q1_moments():
    edges, t = triangle()
    result = exact_hodge_moments_at_level(edges, t, 5, 1, 4)

    assert torch.allclose(
        result, torch.zeros(4, dtype=torch.float64),
        atol=1e-12, rtol=0,
    )


def test_full_triangle_q2_moments():
    # The triangle has one 2-simplex, with Delta_2 = [3].
    # Active-vertex normalization therefore gives H_2 = [0].
    edges, t = triangle()
    result = exact_hodge_moments_at_level(edges, t, 5, 2, 4)

    assert torch.allclose(
        result, torch.zeros(4, dtype=torch.float64),
        atol=1e-12, rtol=0,
    )


def test_selected_dimensions_and_block_constant_rows():
    edges, t = triangle()
    matrices = exact_hodge_moment_matrices(
        edges, t, 6, (0, 3), 4
    )

    assert list(matrices) == [0, 3]
    assert matrices[0].shape == (6, 4)
    assert torch.equal(
        matrices[3], torch.zeros(6, 4, dtype=torch.float64)
    )

    for first, second in ((0, 1), (2, 3), (4, 5)):
        assert torch.equal(matrices[0][first], matrices[0][second])


def test_provisional_ties_and_original_labels():
    edges, _ = triangle()

    simplices = filtered_clique_simplices(
        edges, torch.tensor([5, 1, 3]), 3, 2
    )
    assert simplices[0] == [(1,), (2,)]
    assert simplices[1] == [(1, 2)]

    tied = exact_hodge_moment_matrices(
        edges, torch.tensor([1, 3, 3]), 6, (0,), 4
    )
    expected = torch.full((4,), 1.0 / 3.0, dtype=torch.float64)
    assert torch.allclose(tied[0][2], expected, atol=1e-12, rtol=0)
