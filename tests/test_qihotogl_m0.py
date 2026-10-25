"""M0 fixed-random filtration control checks."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

import torch_topological.nn.qihotogl as qi


def graph_batch():
    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3), (3, 4),
    ]
    directed = [
        edge for a, b in pairs
        for edge in ((a, b), (b, a))
    ]
    return Batch.from_data_list([Data(
        x=torch.tensor([
            [.1, .3], [.2, .8], [.7, .4],
            [.6, .1], [1.2, .9],
        ], dtype=torch.float64),
        edge_index=torch.tensor(directed, dtype=torch.long).t(),
        num_nodes=5,
    )])


def make_layer(method="exact"):
    return qi.QiHOTOGL(
        2, 2, 5, 6, "mean",
        k=8, q_values=(0, 1, 2), z_max=2,
        backward_mode="M0", moment_method=method,
        n_samples=100, seed=42,
    ).double()


def test_fixed_scores_are_reproducible_without_global_rng_changes():
    batch = graph_batch()
    state = torch.random.get_rng_state().clone()

    first = qi.fixed_random_filtration_scores(
        batch.edge_index, 5, seed=42, view=0
    )
    reordered = qi.fixed_random_filtration_scores(
        torch.cat([
            batch.edge_index.flip(1),
            batch.edge_index,
        ], dim=1),
        5, seed=42, view=0,
    )
    other_view = qi.fixed_random_filtration_scores(
        batch.edge_index, 5, seed=42, view=1
    )

    assert torch.equal(first, reordered)
    assert not torch.equal(first, other_view)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("method", ["exact", "monte_carlo"])
def test_m0_fixed_assignments_and_zero_scorer_backward(method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("M0 must not run SELECT.")

    monkeypatch.setattr(
        qi, "single_step_filtration_teacher", forbidden
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        batch = graph_batch()
        layer = make_layer(method)
        classifier = torch.nn.Linear(2, 2).double()
        x = batch.x.clone().requires_grad_(True)

        optimizer = torch.optim.SGD(
            list(layer.parameters()) + list(classifier.parameters()),
            lr=0.01,
        )
        optimizer.zero_grad(set_to_none=True)
        scorer_before = [
            p.detach().clone() for p in layer.filtrations.parameters()
        ]

        output = layer(x, batch)
        assignments = [
            c["t"].clone() for _, c in layer._training_contexts
        ]
        logits = classifier(global_mean_pool(output, batch.batch))
        F.cross_entropy(logits, torch.tensor([1])).backward()

        events = layer.backward_surrogate()
        assert all(e["selected"] is None for e in events)
        assert all(
            p.grad is None for p in layer.filtrations.parameters()
        )
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.moment_projector.parameters()
        )
        assert layer._training_contexts is None

        optimizer.step()
        for old, p in zip(
            scorer_before, layer.filtrations.parameters()
        ):
            assert torch.equal(old, p.detach())

        # Even deliberate scorer-weight and embedding changes must
        # leave M0's actual assignments unchanged.
        with torch.no_grad():
            for p in layer.filtrations.parameters():
                p.add_(100)

        layer(batch.x * 100 + 50, batch)
        for (_, context), old in zip(
            layer._training_contexts, assignments
        ):
            assert torch.equal(context["t"], old)
            qi.validate_current_filtration(context["t"], layer.k)
        layer.clear_training_context()

        layer.eval()
        with torch.no_grad():
            assert torch.isfinite(layer(batch.x, batch)).all()
        assert layer._training_contexts is None


def test_m0_signal_is_zero():
    scores = torch.tensor([0., .2, .4], requires_grad=True)
    context = {"t": torch.tensor([1, 3, 5])}
    signal = qi.filtration_surrogate_signal(
        scores, context, {"selected": None}, mode="M0"
    )
    assert torch.equal(signal, torch.zeros_like(scores))
    assert not signal.requires_grad


@pytest.mark.parametrize("configured,requested", [
    ("M2", "M0"), ("M0", "M2"),
])
def test_fixed_control_cannot_be_switched_after_forward(configured, requested):
    layer = make_layer()
    layer.backward_mode = configured
    batch = graph_batch()
    layer(batch.x, batch)

    with pytest.raises(ValueError, match="before the forward"):
        layer.backward_surrogate(requested)
    layer.clear_training_context()
