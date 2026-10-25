"""M5 raw-degree filtration control."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

import torch_topological.nn.qihotogl as qi


def graph():
    # Triangle, one attached vertex, and one isolated vertex.
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    edges = [edge for a, b in pairs for edge in ((a, b), (b, a))]
    return Data(
        x=torch.tensor([
            [.1, .2], [.3, .8], [.7, .4], [1.2, .9], [.6, .5],
        ], dtype=torch.float64),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        num_nodes=5,
    )


def test_raw_degrees_duplicates_loops_and_zero_level():
    data = graph()
    context = qi.prepare_degree_filtration(
        data.edge_index, 5, 8, (0, 1, 2), 2
    )
    assert torch.equal(context["t"], torch.tensor([2, 2, 3, 1, 0]))
    assert torch.equal(context["levels"], torch.arange(8))
    assert context["phi"].shape == (8, 6)

    # At threshold zero, only the isolated vertex is present.
    torch.testing.assert_close(
        context["phi"][0],
        torch.tensor([1., 1., 0., 0., 0., 0.], dtype=torch.float64),
    )

    noisy_edges = torch.cat([
        data.edge_index.flip(1), data.edge_index,
        torch.tensor([[0, 4], [0, 4]]),
    ], dim=1)
    repeated = qi.prepare_degree_filtration(
        noisy_edges, 5, 8, (0, 1, 2), 2
    )
    assert torch.equal(context["t"], repeated["t"])
    torch.testing.assert_close(context["phi"], repeated["phi"])


def test_exact_rows_against_induced_complexes():
    data = graph()
    context = qi.prepare_degree_filtration(
        data.edge_index, 5, 8, (0, 1, 2), 2
    )
    degrees = torch.tensor([2, 2, 3, 1, 0])

    for level in range(8):
        active = torch.nonzero(degrees <= level).flatten().tolist()
        mapping = {v: i for i, v in enumerate(active)}
        pairs = [
            (mapping[a], mapping[b])
            for a, b in data.edge_index.t().tolist()
            if a in mapping and b in mapping
        ]
        edges = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs else torch.empty((2, 0), dtype=torch.long)
        )
        values = torch.ones(len(active), dtype=torch.long)
        expected = torch.cat([
            qi.exact_hodge_moments_at_level(edges, values, 1, q, 2)
            for q in (0, 1, 2)
        ])
        torch.testing.assert_close(context["phi"][level], expected)


def test_degree_filtration_relabels_with_vertices():
    data = graph()
    permutation = torch.tensor([3, 0, 4, 1, 2])
    first = qi.prepare_degree_filtration(
        data.edge_index, 5, 8, (0, 1, 2), 2
    )
    second = qi.prepare_degree_filtration(
        permutation[data.edge_index], 5, 8, (0, 1, 2), 2
    )
    assert torch.equal(second["t"][permutation], first["t"])
    torch.testing.assert_close(first["phi"], second["phi"])


@pytest.mark.parametrize("method", ["exact", "monte_carlo"])
def test_fixed_forward_and_classification_backward(method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("M5 must bypass scorer, decoder and SELECT.")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        batch = Batch.from_data_list([graph()])
        layer = qi.QiHOTOGL(
            2, 2, 5, 6, "mean",
            k=8, q_values=(0, 1, 2), z_max=2,
            backward_mode="M5", moment_method=method,
            n_samples=200, seed=42,
        ).double()
        layer.moment_projector[1] = torch.nn.LeakyReLU(.01)
        classifier = torch.nn.Linear(2, 2).double()

        monkeypatch.setattr(layer.filtrations, "forward", forbidden)
        monkeypatch.setattr(qi, "decode_filtration_scores", forbidden)
        monkeypatch.setattr(qi, "single_step_filtration_teacher", forbidden)

        parameters = list(layer.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.SGD(parameters, lr=.01)
        optimizer.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in layer.filtrations.parameters()]
        x = batch.x.clone().requires_grad_(True)

        output = layer(x, batch)
        contexts = list(layer._training_contexts)
        expected_t = torch.tensor([2, 2, 3, 1, 0])
        for _, context in contexts:
            assert torch.equal(context["t"], expected_t)
        torch.testing.assert_close(contexts[0][1]["phi"], contexts[1][1]["phi"])

        logits = classifier(global_mean_pool(output, batch.batch))
        F.cross_entropy(logits, torch.tensor([1])).backward()
        events = layer.backward_surrogate()
        assert all(e["selected"] is None for e in events)
        assert all(e["fixed_degree_control"] for e in events)
        assert layer._training_contexts is None
        assert all(p.grad is None for p in layer.filtrations.parameters())
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.moment_projector.parameters()
        )
        optimizer.step()
        for old, parameter in zip(before, layer.filtrations.parameters()):
            assert torch.equal(old, parameter.detach())

        with torch.no_grad():
            for parameter in layer.filtrations.parameters():
                parameter.add_(100)
        layer(batch.x * 100 + 50, batch)
        for _, context in layer._training_contexts:
            assert torch.equal(context["t"], expected_t)
        layer.clear_training_context()

        layer.eval()
        with torch.no_grad():
            assert torch.isfinite(layer(batch.x, batch)).all()
        assert layer._training_contexts is None


def test_m5_zero_signal_without_context():
    scores = torch.ones(5, requires_grad=True)
    signal = qi.filtration_surrogate_signal(
        scores, None, None, mode="M5"
    )
    assert torch.equal(signal, torch.zeros_like(scores))
    assert not signal.requires_grad


@pytest.mark.parametrize("configured,requested", [
    ("M5", "M2"), ("M2", "M5"), ("M5", "M0"),
])
def test_cannot_switch_degree_control_after_forward(configured, requested):
    layer = qi.QiHOTOGL(
        2, 2, 5, 6, "mean",
        k=8, backward_mode=configured,
    ).double()
    batch = Batch.from_data_list([graph()])
    layer(batch.x, batch)
    with pytest.raises(ValueError, match="before the forward"):
        layer.backward_surrogate(requested)
    layer.clear_training_context()
