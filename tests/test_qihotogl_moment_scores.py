"""Validate provisional row scores against recomputed exact moments."""

import pytest
import torch

from torch_topological.nn.qihotogl import (
    exact_hodge_moment_matrices,
    select_filtration_direction,
    validate_current_filtration,
)


@pytest.mark.parametrize(
    "values,k",
    [
        ([4, 1, 3, 6], 6),  # Gaps and consecutive occupied levels.
        ([2, 4, 1, 3], 4),  # k = n; collisions in proposals.
        ([1, 3, 5, 7], 8),  # Gaps, including above the last arrival.
    ],
)
def test_provisional_scores_and_argmax(values, k):
    # Triangle (0,1,2), with an additional edge (2,3).
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    t = torch.tensor(values)
    original = t.clone()
    sigma = validate_current_filtration(t, k)
    q_values = (0, 1, 2, 3)
    z_max = 4

    def representation(assignment):
        blocks = exact_hodge_moment_matrices(
            edges, assignment, k, q_values, z_max
        )
        # Concatenation makes the inner product sum across selected q.
        return torch.cat([blocks[q] for q in q_values], dim=1)

    phi = representation(t)

    # Independently recompute each provisional representation.
    candidates = []
    for m in range(1, t.numel()):
        vertex = int(sigma[m].item())
        s = int(t[vertex].item())

        for direction in ("-", "+"):
            if direction == "-" and not (m >= 2 or s > 2):
                continue
            if direction == "+" and not s < k:
                continue

            proposal = t.clone()
            proposal[vertex] += -1 if direction == "-" else 1
            difference = representation(proposal) - phi

            # Only the predicted row may change.
            changed_row = s - 2 if direction == "-" else s - 1
            other_rows = difference.clone()
            other_rows[changed_row] = 0
            assert torch.allclose(
                other_rows, torch.zeros_like(other_rows),
                atol=1e-12, rtol=0,
            )

            candidates.append((m, direction, s, difference))

    # Compare all scores and winners under several descent matrices.
    generator = torch.Generator().manual_seed(17)

    for _ in range(5):
        gamma = torch.randn(
            phi.shape, generator=generator, dtype=torch.float64
        )
        explicit_scores = []

        for m, direction, s, difference in candidates:
            explicit = torch.sum(gamma * difference).item()

            if direction == "-":
                local = torch.dot(
                    gamma[s - 2], phi[s - 1] - phi[s - 2]
                ).item()
            else:
                local = torch.dot(
                    gamma[s - 1], phi[s - 2] - phi[s - 1]
                ).item()

            assert local == pytest.approx(
                explicit, abs=1e-12, rel=1e-10
            )
            explicit_scores.append((m, direction, explicit))

        # Independent argmax, ordered by rank then decrement.
        expected = None
        best_score = 0.0
        for m, direction, score in explicit_scores:
            if score > best_score:
                best_score = score
                expected = (m, direction, score)

        selected = select_filtration_direction(t, sigma, phi, gamma, k)

        if expected is None:
            assert selected is None
        else:
            assert selected is not None
            assert selected[:2] == expected[:2]
            assert selected[2] == pytest.approx(
                expected[2], abs=1e-12, rel=1e-10
            )

    # Neither scoring nor moment construction should alter the assignment.
    assert torch.equal(t, original)
