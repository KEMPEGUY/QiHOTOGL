"""Integration checks for the moment-based QiHOTOGL layer."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from torch_topological.nn.qihotogl import QiHOTOGL


def graph_batch():
    return Batch.from_data_list([
        Data(
            x=torch.tensor([[0.1], [0.3], [0.7], [1.2]]),
            edge_index=torch.tensor([
                [0, 1, 0, 2, 1, 2, 2, 3],
                [1, 0, 2, 0, 2, 1, 3, 2],
            ]),
        ),
        Data(
            x=torch.tensor([[0.2], [0.6], [0.9]]),
            edge_index=torch.tensor([
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ]),
        ),
    ])


def make_layer(**options):
    return QiHOTOGL(
        4, 2, 8, 8, "mean",
        k=8, q_values=(0, 1), z_max=3,
        n_samples=500, **options,
    )


@pytest.mark.parametrize("mode", ["M2", "M3"])
@pytest.mark.parametrize("method", ["exact", "monte_carlo"])
def test_batched_training_and_inference(mode, method):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        batch = graph_batch()
        backbone = GCNConv(1, 4)
        topology = make_layer(
            backward_mode=mode, moment_method=method
        )
        classifier = torch.nn.Linear(4, 2)

        parameters = (
            list(backbone.parameters())
            + list(topology.parameters())
            + list(classifier.parameters())
        )
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        optimizer.zero_grad(set_to_none=True)

        h = torch.tanh(backbone(batch.x, batch.edge_index))
        output = topology(h, batch)
        assert output.shape == (7, 4)

        contexts = topology._training_contexts
        assert len(contexts) == 4
        snapshots = [
            context["t"].clone() for _, context in contexts
        ]

        logits = classifier(global_mean_pool(output, batch.batch))
        loss = F.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward(retain_graph=True)

        assert all(
            p.grad is None for p in topology.filtrations.parameters()
        )
        events = topology.backward_surrogate()
        assert len(events) == 4

        for parameter in parameters:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all().item()

        for (_, context), before in zip(contexts, snapshots):
            assert torch.equal(context["t"], before)

        optimizer.step()
        assert topology._training_contexts is None

        # A second training forward must work after consuming context.
        optimizer.zero_grad(set_to_none=True)
        h = torch.tanh(backbone(batch.x, batch.edge_index))
        output = topology(h, batch)
        logits = classifier(global_mean_pool(output, batch.batch))
        F.cross_entropy(logits, torch.tensor([0, 1])).backward(
            retain_graph=True
        )
        topology.backward_surrogate()
        optimizer.step()

        topology.eval()
        with torch.no_grad():
            h = torch.tanh(backbone(batch.x, batch.edge_index))
            output = topology(h, batch)

        assert output.shape == (7, 4)
        assert torch.isfinite(output).all().item()
        assert topology._training_contexts is None


def test_exact_batch_matches_separate_graphs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        layer = make_layer()
        layer.eval()
        batch = graph_batch()
        graphs = batch.to_data_list()

        with torch.no_grad():
            features = torch.randn(7, 4)
            together = layer(features, batch)
            separate = torch.cat([
                layer(features[:4], graphs[0]),
                layer(features[4:], graphs[1]),
            ])

        assert torch.allclose(
            together, separate, atol=1e-6, rtol=1e-5
        )


def test_context_lifecycle():
    batch = graph_batch()
    layer = make_layer()
    features = torch.randn(7, 4, requires_grad=True)

    with pytest.raises(RuntimeError, match="No pending"):
        layer.backward_surrogate()

    output = layer(features, batch)

    with pytest.raises(RuntimeError, match="pending"):
        layer(features, batch)

    with pytest.raises(RuntimeError, match="backward"):
        layer.backward_surrogate()

    # Explicitly discard an unused training forward.
    layer.clear_training_context()
    assert layer._training_contexts is None

    # No-grad execution must not retain training contexts.
    with torch.no_grad():
        output = layer(features, batch)
    assert output.shape == (7, 4)
    assert layer._training_contexts is None


def test_oversized_graph_rejected():
    layer = QiHOTOGL(
        4, 1, 8, 8, "mean",
        k=2, q_values=(0,), z_max=2,
    )
    graph = Data(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
    )
    with pytest.raises(ValueError, match="n <= k"):
        layer(torch.randn(3, 4), graph)
    assert layer._training_contexts is None
