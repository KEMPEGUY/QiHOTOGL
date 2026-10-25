"""Pure random backward control: no SELECT or filtration credit."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

import torch_topological.nn.qihotogl as qi


def test_formula_without_context_or_teacher():
    scores = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        expected = .2 * torch.randn_like(scores)
        torch.manual_seed(123)
        actual = qi.filtration_surrogate_signal(
            scores, None, None, mode="M4", m4_scale=.2
        )
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_score_values_do_not_determine_noise():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = qi.filtration_surrogate_signal(
            torch.zeros(8), None, None, mode="M4"
        )
        torch.manual_seed(123)
        second = qi.filtration_surrogate_signal(
            torch.arange(8).float() * 100, None, None, mode="M4"
        )
        third = qi.filtration_surrogate_signal(
            torch.zeros(8), None, None, mode="M4"
        )
    assert torch.equal(first, second)
    assert not torch.equal(second, third)


@pytest.mark.parametrize("scale", [0., -1., float("nan"), float("inf")])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        qi.filtration_surrogate_signal(
            torch.zeros(4), None, None, mode="M4", m4_scale=scale
        )


@pytest.mark.parametrize("method", ["exact", "monte_carlo"])
def test_integrated_gradient_and_sgd_without_select(method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("M4 must not execute SELECT.")

    monkeypatch.setattr(qi, "single_step_filtration_teacher", forbidden)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
        edges = [edge for a, b in pairs for edge in ((a, b), (b, a))]
        batch = Batch.from_data_list([Data(
            x=torch.tensor([
                [.1, .2], [.3, .8], [.7, .4], [1.2, .9],
            ], dtype=torch.float64),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            num_nodes=4,
        )])
        scale = .01
        layer = qi.QiHOTOGL(
            2, 2, 5, 6, "mean",
            k=8, q_values=(0, 1, 2), z_max=2,
            decoder_mode="rank", backward_mode="M4",
            moment_method=method, n_samples=100, seed=42,
            m4_scale=scale,
        ).double()
        classifier = torch.nn.Linear(2, 2).double()
        layer.train()

        output = layer(batch.x, batch)
        contexts = list(layer._training_contexts)
        logits = classifier(global_mean_pool(output, batch.batch))
        F.cross_entropy(logits, torch.tensor([1])).backward(
            retain_graph=True
        )

        parameters = tuple(layer.filtrations.parameters())
        assert all(p.grad is None for p in parameters)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in classifier.parameters()
        )

        # Even entirely zero moment gradients must not gate M4.
        for _, context in contexts:
            assert context["phi"].grad is not None
            context["phi"].grad.zero_()

        scores = [s for s, _ in contexts]
        torch.manual_seed(314)
        signals = [scale * torch.randn_like(s) for s in scores]
        reference = torch.autograd.grad(
            scores, parameters,
            grad_outputs=signals, retain_graph=True,
        )
        before = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.SGD(parameters, lr=.01)

        torch.manual_seed(314)
        events = layer.backward_surrogate()
        assert layer._training_contexts is None
        assert len(events) == len(contexts)
        assert all(
            e["selected"] is None and e["random_backward_control"]
            for e in events
        )

        for parameter, expected in zip(parameters, reference):
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, expected)
        assert sum(float(g.square().sum()) for g in reference) > 0

        optimizer.step()
        for parameter, old, gradient in zip(parameters, before, reference):
            torch.testing.assert_close(parameter, old - .01 * gradient)

        layer.eval()
        with torch.no_grad():
            assert torch.isfinite(layer(batch.x, batch)).all()
        assert layer._training_contexts is None
