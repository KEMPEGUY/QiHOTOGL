"""M2-pair signal, integrated gradient and SGD checks."""

import pytest
import torch
from torch_geometric.data import Data, Batch

import torch_topological.nn.qihotogl as qi


@pytest.mark.parametrize("direction", ["-", "+"])
@pytest.mark.parametrize("swap", [False, True])
def test_signal_matches_realised_move(direction, swap):
    # Selected rank 2 is original vertex 0.
    t = torch.tensor([4, 1, 3, 5] if swap else [4, 1, 2, 6])
    sigma = torch.argsort(t)
    scores = torch.tensor(
        [.4, .1, .3, .6], dtype=torch.float64, requires_grad=True
    )
    target, target_sigma, move_type = qi.realise_filtration_direction(
        t, sigma, 2, direction, 8
    )
    displacement = target.long() - t.long()
    event = {
        "selected": (2, direction, 2.0),
        "target_t": target,
        "target_sigma": target_sigma,
        "displacement": displacement,
        "move_type": move_type,
    }
    delta = -1 if direction == "-" else 1
    expected = torch.zeros_like(scores)
    expected[displacement != 0] = -2.0 * delta

    actual = qi.filtration_surrogate_signal(
        scores, {"t": t, "sigma": sigma}, event, mode="M2-pair"
    )
    assert (move_type == "swap") == swap
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad

    if not swap:
        ordinary = qi.filtration_surrogate_signal(
            scores, {"t": t, "sigma": sigma}, event, mode="M2"
        )
        torch.testing.assert_close(actual, ordinary)


def test_no_selection_returns_zero():
    scores = torch.tensor([.1, .2, .3], requires_grad=True)
    signal = qi.filtration_surrogate_signal(
        scores, {"t": torch.tensor([1, 2, 3])},
        {"selected": None}, mode="M2-pair",
    )
    assert torch.equal(signal, torch.zeros_like(scores))
    assert not signal.requires_grad


@pytest.mark.parametrize("method", ["exact", "monte_carlo"])
def test_integrated_gradient_and_sgd(method):
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
        layer = qi.QiHOTOGL(
            2, 2, 5, 6, "mean",
            k=8, q_values=(0, 1, 2), z_max=2,
            decoder_mode="rank", backward_mode="M2-pair",
            moment_method=method, n_samples=100, seed=42,
        ).double()
        layer.train()
        layer(batch.x, batch)
        contexts = list(layer._training_contexts)

        # Construct a diagnostic linear moment loss with a positive
        # provisional candidate. This tests backward plumbing, not accuracy.
        losses = []
        for _, context in contexts:
            phi = context["phi"]
            sigma = context["sigma"]
            candidates = []
            for m in range(1, sigma.numel()):
                level = int(context["t"][sigma[m]])
                if level < context["k"]:
                    gamma = torch.zeros_like(phi)
                    gamma[level - 1] = (
                        phi[level - 2] - phi[level - 1]
                    ).detach()
                    event = qi.single_step_filtration_teacher(context, gamma)
                    if event["selected"] is not None:
                        candidates.append((event["selected"][2], gamma))
            assert candidates, "Fixture produced no positive candidate."
            gamma = max(candidates, key=lambda item: item[0])[1]
            losses.append(-(phi * gamma).sum())

        sum(losses).backward(retain_graph=True)

        score_tensors = []
        expected_signals = []
        for scores, context in contexts:
            event = qi.single_step_filtration_teacher(
                context, -context["phi"].grad.detach()
            )
            assert event["selected"] is not None
            _, direction, strength = event["selected"]
            delta = -1 if direction == "-" else 1
            expected = torch.zeros_like(scores)
            # Independent formula: every realised moved vertex receives
            # the selected vertex's signed coefficient.
            expected[event["displacement"] != 0] = -float(strength) * delta
            score_tensors.append(scores)
            expected_signals.append(expected)

        parameters = tuple(layer.filtrations.parameters())
        reference = torch.autograd.grad(
            score_tensors, parameters,
            grad_outputs=expected_signals,
            retain_graph=True,
        )
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer = torch.optim.SGD(parameters, lr=.01)

        events = layer.backward_surrogate()
        assert len(events) == len(contexts)
        assert layer._training_contexts is None

        for parameter, expected in zip(parameters, reference):
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, expected)
        assert sum(float(g.square().sum()) for g in reference) > 0

        optimizer.step()
        for parameter, old, gradient in zip(parameters, before, reference):
            torch.testing.assert_close(parameter, old - .01 * gradient)
