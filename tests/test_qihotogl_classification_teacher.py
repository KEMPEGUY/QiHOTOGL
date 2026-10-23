"""Verify classification gradients reach the discrete teacher."""

import torch
import torch.nn.functional as F

from torch_topological.nn.qihotogl import (
    prepare_filtration_teacher,
    single_step_filtration_teacher,
    validate_current_filtration,
)


def test_classification_gradient_reaches_teacher():
    # Keep this test's random initialization local.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)

        edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
        scores = torch.tensor(
            [0., 0.2, 0.4, 0.7],
            dtype=torch.float64,
            requires_grad=True,
        )
        original_scores = scores.detach().clone()

        context = prepare_filtration_teacher(
            scores, edges,
            k=8, q_values=(0, 1), z_max=4,
            moment_method="exact",
        )
        snapshots = {
            key: context[key].clone()
            for key in ("t", "sigma", "phi")
        }

        phi = context["phi"].detach().clone().requires_grad_(True)
        classifier = torch.nn.Linear(
            phi.numel(), 2, dtype=phi.dtype
        )

        logits = classifier(phi.reshape(1, -1))
        loss = F.cross_entropy(logits, torch.tensor([1]))
        loss.backward()

        assert torch.isfinite(loss).item()
        assert phi.grad is not None
        assert phi.grad.shape == phi.shape
        assert torch.isfinite(phi.grad).all().item()
        assert phi.grad.norm().item() > 0

        for parameter in classifier.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all().item()

        gamma = -phi.grad.detach()
        result = single_step_filtration_teacher(context, gamma)

        # Verify against an independently enumerated candidate list,
        # using the manuscript's provisional row-score formulas.
        candidates = []
        t, sigma = context["t"], context["sigma"]

        for m in range(1, t.numel()):
            level = int(t[sigma[m]].item())

            if m >= 2 or level > 2:
                score = torch.dot(
                    gamma[level - 2],
                    phi.detach()[level - 1] - phi.detach()[level - 2],
                ).item()
                candidates.append((m, "-", score))

            if level < context["k"]:
                score = torch.dot(
                    gamma[level - 1],
                    phi.detach()[level - 2] - phi.detach()[level - 1],
                ).item()
                candidates.append((m, "+", score))

        # Python max preserves the first candidate in a tie.
        expected = max(candidates, key=lambda item: item[2])
        if expected[2] <= 0:
            expected = None

        assert result["selected"] == expected

        if expected is None:
            assert result["move_type"] == "none"
            assert torch.equal(result["target_t"], t)
        else:
            assert result["selected"][2] > 0
            assert result["move_type"] in ("direct", "swap")

        assert torch.equal(
            result["target_sigma"],
            validate_current_filtration(
                result["target_t"], context["k"]
            ),
        )
        assert torch.equal(
            result["displacement"], result["target_t"] - t
        )

        for key, before in snapshots.items():
            assert torch.equal(context[key], before)

        assert torch.equal(scores.detach(), original_scores)
        assert scores.grad is None
