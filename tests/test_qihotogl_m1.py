"""M1 formula and integrated backward checks."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

import torch_topological.nn.qihotogl as qi


@pytest.mark.parametrize("rank,direction,boundary", [
    (1, "+", 0.3),
    (2, "-", 0.3),
    (3, "+", 0.8),
])
def test_m1_boundary_and_signal(rank, direction, boundary):
    scores = torch.tensor(
        [0., .2, .4, .8],
        dtype=torch.float64,
        requires_grad=True,
    )
    context = {
        "t": torch.tensor([1, 3, 4, 7]),
        "sigma": torch.arange(4),
    }
    event = {"selected": (rank, direction, 2.)}

    actual = qi.filtration_surrogate_signal(
        scores, context, event, "M1",
        m1_alpha=10., m1_boundary_lambda=.5,
    )
    expected = -2 * torch.sigmoid(
        10 * (scores.detach() - boundary)
    )

    assert torch.allclose(
        actual, expected, atol=1e-14, rtol=1e-14
    )
    assert not actual.requires_grad
    assert scores.grad is None


def test_m1_no_selection_returns_zero():
    scores = torch.tensor([0., .2, .4])
    context = {"t": torch.tensor([1, 3, 5])}
    actual = qi.filtration_surrogate_signal(
        scores, context, {"selected": None}, "M1"
    )
    assert torch.equal(actual, torch.zeros_like(scores))


@pytest.mark.parametrize("option,value", [
    ("m1_alpha", 0.),
    ("m1_alpha", float("nan")),
    ("m1_boundary_lambda", -1.),
    ("m1_boundary_lambda", 2.),
])
def test_invalid_m1_options(option, value):
    with pytest.raises(ValueError):
        qi.QiHOTOGL(
            2, 2, 5, 6, "mean",
            backward_mode="M1",
            **{option: value},
        )


@pytest.mark.parametrize("method", ["exact", "monte_carlo"])
def test_m1_layer_gradient_and_sgd(method):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        batch = Batch.from_data_list([Data(
            x=torch.tensor([
                [.1, .3], [.2, .8], [.7, .4], [.6, .1],
            ], dtype=torch.float64),
            edge_index=torch.tensor([
                [0, 1, 0, 2, 1, 2, 2, 3],
                [1, 0, 2, 0, 2, 1, 3, 2],
            ]),
        )])

        layer = qi.QiHOTOGL(
            2, 2, 5, 6, "mean",
            k=8, q_values=(0, 1, 2), z_max=2,
            decoder_mode="rank",
            moment_method=method,
            backward_mode="M1",
            n_samples=100, seed=42,
        ).double()

        output = layer(batch.x, batch)
        contexts = layer._training_contexts

        # Controlled downstream loss checks routing, not accuracy.
        loss = output.square().sum() * 0
        for _, context in contexts:
            phi = context["phi"]
            desired = torch.zeros_like(phi)
            found = False

            for rank in range(1, context["t"].numel()):
                level = int(
                    context["t"][context["sigma"][rank]]
                )
                if level < layer.k:
                    change = (
                        phi.detach()[level - 2]
                        - phi.detach()[level - 1]
                    )
                    if change.norm().item() > 0:
                        desired[level - 1] = change
                        found = True
                        break

            assert found
            loss = loss + F.softplus(-(phi * desired).sum())

        loss.backward(retain_graph=True)
        parameters = list(layer.filtrations.parameters())
        assert all(p.grad is None for p in parameters)

        tensors, signals = [], []
        snapshots = [c["t"].clone() for _, c in contexts]

        for scores, context in contexts:
            event = qi.single_step_filtration_teacher(
                context, -context["phi"].grad.detach()
            )
            assert event["selected"] is not None
            rank, direction, score = event["selected"]
            i = int(context["sigma"][rank])
            neighbor_rank = rank + (-1 if direction == "-" else 1)

            boundary = scores[i].detach()
            if neighbor_rank < scores.numel():
                j = int(context["sigma"][neighbor_rank])
                boundary = (
                    (1 - layer.m1_boundary_lambda) * scores[i].detach()
                    + layer.m1_boundary_lambda * scores[j].detach()
                )

            tensors.append(scores)
            signals.append(
                -score * torch.sigmoid(
                    layer.m1_alpha * (scores.detach() - boundary)
                )
            )

        expected = torch.autograd.grad(
            tensors, parameters,
            grad_outputs=signals,
            retain_graph=True,
            allow_unused=True,
        )
        expected = [
            torch.zeros_like(p) if g is None else g.detach().clone()
            for p, g in zip(parameters, expected)
        ]
        before = [p.detach().clone() for p in parameters]

        layer.backward_surrogate()

        for p, reference in zip(parameters, expected):
            assert p.grad is not None
            assert torch.allclose(
                p.grad, reference, atol=1e-12, rtol=1e-9
            )

        assert sum(float(g.square().sum()) for g in expected) > 0

        torch.optim.SGD(parameters, lr=.01).step()
        for p, old, gradient in zip(parameters, before, expected):
            assert torch.allclose(
                p.detach(), old - .01 * gradient,
                atol=1e-12, rtol=1e-9,
            )

        for (_, context), old in zip(contexts, snapshots):
            assert torch.equal(context["t"], old)

        assert layer._training_contexts is None
