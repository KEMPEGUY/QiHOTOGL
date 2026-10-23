"""Verify two backward passes accumulate through a shared GCN."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_topological.nn.qihotogl import (
    prepare_filtration_teacher,
    single_step_filtration_teacher,
    filtration_surrogate_signal,
)


@pytest.mark.parametrize("mode", ["M2", "M3"])
def test_shared_gcn_gradient_accumulation(mode):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)

        dtype = torch.float64
        edges = torch.tensor([
            [0, 1, 0, 2, 1, 2, 2, 3],
            [1, 0, 2, 0, 2, 1, 3, 2],
        ])
        x = torch.tensor(
            [[0.1], [0.3], [0.7], [1.2]], dtype=dtype
        )

        backbone = GCNConv(1, 2).double()
        scorer = torch.nn.Linear(2, 1, bias=False).double()

        h = torch.tanh(backbone(x, edges))
        scores = scorer(h).flatten()

        context = prepare_filtration_teacher(
            scores, edges,
            k=8, q_values=(0, 1), z_max=4,
            moment_method="exact",
        )
        original_t = context["t"].clone()
        phi = context["phi"].detach().clone().requires_grad_(True)

        # Construct a classification head with a moment-dependent
        # descent direction known to give a positive candidate.
        desired_gamma = torch.zeros_like(phi)
        found = False
        t, sigma = context["t"], context["sigma"]

        for m in range(1, t.numel()):
            level = int(t[sigma[m]].item())
            if level < context["k"]:
                change = phi.detach()[level - 2] - phi.detach()[level - 1]
                if change.square().sum().item() > 0:
                    desired_gamma[level - 1] = change
                    found = True
                    break

        assert found

        features = torch.cat((h.mean(dim=0), phi.flatten()))
        classifier = torch.nn.Linear(
            features.numel(), 2, dtype=dtype
        )

        # Controlled initialization tests gradient plumbing,
        # not prediction quality.
        with torch.no_grad():
            classifier.weight.zero_()
            classifier.bias.zero_()
            classifier.weight[1, :2] = torch.tensor(
                [0.2, -0.3], dtype=dtype
            )
            classifier.weight[1, 2:] = desired_gamma.flatten()

        logits = classifier(features.unsqueeze(0))
        loss = F.cross_entropy(logits, torch.tensor([1]))

        parameters = (
            list(backbone.parameters())
            + list(scorer.parameters())
            + list(classifier.parameters())
        )

        # Independent reference: classification contribution.
        classification_ref = torch.autograd.grad(
            loss, parameters,
            retain_graph=True, allow_unused=True,
        )
        phi_gradient = torch.autograd.grad(
            loss, phi, retain_graph=True
        )[0]

        teacher = single_step_filtration_teacher(
            context, -phi_gradient.detach()
        )
        assert teacher["selected"] is not None

        signal = filtration_surrogate_signal(
            scores, context, teacher, mode
        )
        assert signal.abs().sum().item() > 0

        # Independent reference: surrogate contribution.
        surrogate_ref = torch.autograd.grad(
            scores, parameters,
            grad_outputs=signal,
            retain_graph=True, allow_unused=True,
        )

        expected = []
        for parameter, first, second in zip(
            parameters, classification_ref, surrogate_ref
        ):
            total = torch.zeros_like(parameter)
            if first is not None:
                total = total + first
            if second is not None:
                total = total + second
            expected.append(total)

        optimizer = torch.optim.SGD(parameters, lr=0.01)
        optimizer.zero_grad(set_to_none=True)

        # Actual training sequence: retain the shared graph until
        # the scorer's surrogate backward has used it.
        loss.backward(retain_graph=True)

        # The hard decoder blocks classification gradients to scorer.
        for parameter in scorer.parameters():
            assert parameter.grad is None

        scores.backward(gradient=signal)

        for parameter, reference in zip(parameters, expected):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all().item()
            assert torch.allclose(
                parameter.grad, reference,
                atol=1e-10, rtol=1e-8,
            )

        # Both paths must actually contribute to the backbone.
        backbone_count = len(list(backbone.parameters()))
        assert any(
            gradient is not None and gradient.abs().sum().item() > 0
            for gradient in classification_ref[:backbone_count]
        )
        assert any(
            gradient is not None and gradient.abs().sum().item() > 0
            for gradient in surrogate_ref[:backbone_count]
        )

        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.step()

        for parameter, old, gradient in zip(
            parameters, before, expected
        ):
            assert torch.allclose(
                parameter.detach(), old - 0.01 * gradient,
                atol=1e-10, rtol=1e-8,
            )

        assert torch.equal(context["t"], original_t)
