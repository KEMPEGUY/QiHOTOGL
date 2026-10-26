# QiHOTOGL integration

Original plan: 22 October 2026  
Updated: 26 October 2026

The integration started from the simplified TOGL implementation in
`torch_topological/nn/graphs.py`. QiHOTOGL is implemented separately in
`torch_topological/nn/qihotogl.py`.

This document describes the current discrete implementation, the changes
made to the original plan, and the work still to do.

## 1. Architecture

The original TOGL pathway was:

Node features → GCN → neural filtration scores → persistence pairs →
DeepSet encoder → node residual → GCN → graph mean pooling.

The implemented QiHOTOGL pathway is:

Node features → first GCN → node embeddings H.

For each graph, a topology branch computes:

H → filtration construction → Hodge moment matrices →
flatten and concatenate → moment projector → graph correction c.

The correction is added to every node embedding in that graph:

    H'_v = H_v + c

Classification then follows:

H' → second GCN → graph mean pooling → linear classifier.

This differs from the original standalone plan, which fed the flattened
moment matrices directly to a graph classifier. The implemented model
retains a node residual and a downstream GCN, but replaces persistence
features with projected moment features.

The persistence-specific DeepSet encoder is not used in the QiHOTOGL
pathway. The original TOGL implementation remains a separate reference.

## 2. Neural filtration scorer

The neural scorer retains the structure:

    Linear(n_features, hidden_dim)
    ReLU()
    Linear(hidden_dim, n_filtrations)

It produces R scores per vertex:

    scores.shape = [N_batch, R]

Here, N_batch is the total number of vertices in the graph batch.
Scoring and decoding are separated by graph and view.

M1, M2, M2-pair, M3 and M4 use this neural scorer.
M0 and M5 bypass it.

The preceding GCN supplies the scorer's inputs. Its parameters can
receive both ordinary classification gradients through the residual
path and additional credit through the neural scorer.

## 3. Discrete decoding

For each graph and neural filtration view, the decoder sorts scores
and constructs an anchored, injective integer assignment:

    1 = t[sigma[0]] < ... < t[sigma[n-1]] <= k

The decoder requires k >= n, where n is the graph's vertex count.
It operates within each graph, never across the whole batch.

Two decoding modes are available:

- Rank decoding assigns consecutive levels according to sorted order.
- Score-gap decoding uses sorted score differences to allocate additional
  integer gaps when k > n.

The ordering permutation sigma records original vertex indices in
increasing filtration order. Sorting ties are resolved deterministically.

M0 applies the decoder to fixed random scores.
M5 uses raw degrees and does not apply this decoder.

## 4. Complexes and Hodge moments

The vertex assignment extends to each simplex by the lower-star rule:

    simplex_level = max(t[v] for v in simplex)

At each evaluation level, the active vertices induce a subgraph and
its clique complex.

The q-Hodge Laplacian includes both boundary contributions:

    Delta_q = B_q.T @ B_q + B_(q+1) @ B_(q+1).T

Computing dimension q therefore requires simplices through dimension
q + 1 when those simplices exist.

For a nonempty active q-chain space, the current normalization is:

    H_q = I - Delta_q / n_active
    M_q(z) = trace(H_q ** z) / d_q

Here, n_active is the number of active vertices, d_q is the number
of active q-simplices, and the power is a matrix power.
Empty q-chain spaces receive zero moment vectors.

Both exact and Monte Carlo moment computation are available.
The exact computation serves as a reference for the Monte Carlo
estimator and for small-graph experiments.

The QiHOTOGL pathway does not construct persistence pairs or use
persistence-generator indices.

## 5. Dimensions, views and feature size

The selected Hodge dimensions are controlled by q_values.
Only selected dimensions contribute feature blocks and SELECT scores.
Other simplex dimensions are constructed when needed by their operators.

The number of views R, maximum filtration level k and maximum moment
order z_max are configurable.

Each graph contributes:

    R * len(q_values) * k * z_max

moment features.

These are flattened and concatenated before projection to the node
embedding width. The resulting correction is broadcast to every node
of the graph.

Recent experiments use:

    R = 2
    q_values = (0, 1, 2)
    z_max = 4

They replace the moment projector's activation with LeakyReLU(0.01)
inside the experiment script. This replacement is separate from the
filtration scorer's ReLU.

M5 retains k rows, evaluated at degree thresholds 0, ..., k - 1.
Neural-decoded filtrations and M0 use levels 1, ..., k.

## 6. Classification and backward

Training uses one graph-classification objective:

    loss = CrossEntropy(logits, labels)

There is no auxiliary filtration loss and no teacher-student objective.

Classification backward trains the projector, classifier and surrounding
GNN layers. Computed moments are used as gradient-enabled leaves, allowing
the loss gradient with respect to the moment features to be collected.

The hard decoding and complex construction do not provide ordinary
autograd derivatives to the neural scores. A separate backward rule
supplies score-space credit:

    g_theta = J_theta.T @ g_s

where J_theta is the Jacobian of the neural scores with respect to the
scorer parameters.

When the neural scorer shares the first GCN's computation, classification
backward retains that graph for the subsequent surrogate backward.
Both backward passes are completed before one optimizer step.

The training pattern is:

    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = CrossEntropy(logits, batch.y)
    loss.backward(retain_graph=True)
    events = model.topology.backward_surrogate()
    optimizer.step()

Training contexts are cleared after surrogate backward.
Inference does not run SELECT or surrogate backward.

## 7. SELECT and realization

M1, M2, M2-pair and M3 use one SELECT iteration per graph/view
per training step.

The procedure:

1. Obtains the negative classification gradient at the moment features.
2. Scores admissible provisional unit increments and decrements.
3. Sums contributions over the selected Hodge dimensions.
4. Selects the largest strictly positive score.
5. Breaks ties by smallest sorted vertex rank, then decrement before
   increment.
6. Realizes the direction as a direct move or an adjacent swap.
7. Constructs the configured score-space signal.

Provisional proposals may contain ties. Realization preserves an
injective current-filtration target and protects the anchored vertex.

The provisional score and the realized swap's moment response are
different quantities. A positive provisional score does not guarantee
that the realized swap decreases the classification loss.

The target is temporary. It supplies backward credit and does not
overwrite the next forward filtration. After the optimizer step,
the network produces new scores and the decoder is applied again.

A selected event does not necessarily produce a change in the next
decoded filtration.

## 8. Implemented conditions

Seven conditions are selectable through backward_mode:

| Mode | Forward filtration | Scorer credit |
| --- | --- | --- |
| M0 | Fixed random graph/view scores followed by decoding | None |
| M1 | Neural scores followed by decoding | Soft-boundary credit |
| M2 | Neural scores followed by decoding | Selected-vertex directional credit |
| M2-pair | Neural scores followed by decoding | Same-sign pair credit for swaps |
| M3 | Neural scores followed by decoding | Realized-displacement credit |
| M4 | Neural scores followed by decoding | Independent Gaussian noise |
| M5 | Raw graph degrees; ties allowed | None |

For the SELECT-based rules, let S_star be the selected positive
provisional score, i the selected original vertex index, delta the
direction (-1 earlier, +1 later), and d_star = target_t - t.

The implemented signals are:

    M1:
        g_s = -S_star * sigmoid(alpha * (scores - boundary))

    M2:
        g_s = -S_star * delta * e_i

    M2-pair, swap:
        g_s = -S_star * delta * (e_i + e_j)

    M2-pair, direct move:
        same as M2

    M3:
        g_s = -S_star * d_star

    M4:
        g_s = m4_scale * standard_normal_noise

M1 interpolates its boundary between the selected score and its
directional neighbour. For a topmost increment, the boundary is the
selected score. It does not include a direction multiplier.

M1, M2, M2-pair and M3 return zero credit when no positive candidate
is selected.

M4 does not use SELECT, the selected target or the classification
moment-gradient values to construct its random signal.

M0 generates reproducible random scores from the labelled graph,
seed and view. The actual assignment is independent of embeddings,
scorer weights and batch position. Identical labelled graphs share
the same random scores for the same seed and view. An individual
assignment is not guaranteed to be equivariant under vertex relabelling.

M5 assigns:

    t_i = degree_G(v_i)

Degrees are taken from the simple undirected graph. Duplicate edges
and self-loops do not alter them. No ranking, integer decoder or SELECT
is used. Repeated degree views are identical.

M0 and M5 bypass scorer backward. The surrounding network still trains.

## 9. Tests and experiments

The reported QiHOTOGL regression suite passed 152 tests after M5
was added. Coverage includes decoding, selection, realization,
moments, surrogate formulas, batched layer behavior and shared-backbone
gradient accumulation.

Tests for the added controls check fixed assignments, scorer bypass,
random credit and exact/Monte Carlo integration.

Separate audits have checked:

- Nonzero moment gradients.
- Score-space signal routing.
- Scorer gradients against reference calculations.
- Optimizer parameter updates.
- Filtration changes after updates.
- Differences between provisional and realized moment responses.
- Decoder boundaries and repeated-update behavior.

These checks establish implementation behavior. They do not establish
that every update realizes the selected target or improves classification.

Classification experiments have compared all seven conditions and a
three-layer GCN baseline on small synthetic datasets. Recent comparisons
used exact moments and one seed.

On the marked four-cycle diagnostic, all five neural-score conditions,
including random-backward M4, reached 100% test accuracy. M0, M5 and
the tested GCN baseline remained at 50%.

This result supports the usefulness of the neural-filtration architecture
on that diagnostic, but does not isolate the benefit of SELECT-based
credit. The models were not parameter-matched.

## 10. Changes from the original plan

Three changes should be kept explicit:

1. The implemented architecture projects moments back to node embeddings
   and retains a downstream GCN, rather than using only a standalone
   moment classifier.
2. M0 uses fixed random graph/view scores instead of freezing a neural
   score-producing pathway.
3. M5 uses raw degree filtrations with ties, rather than decoded degree
   rankings.

M2-pair was added as a further backward experiment.

External training and audit scripts are available, but a training demo
also remains in the layer module. Moving that demo out is still cleanup
work.

## 11. Remaining work

- Add a control with the same neural forward construction but no injected
  scorer credit. Frozen scorer weights alone do not fix its filtration
  when the preceding GCN changes.
- Record actual filtration and moment changes alongside selection counts
  in the classification experiments.
- Report direct moves, swaps and realized-response alignment separately.
- Repeat comparisons across seeds and broader datasets.
- Evaluate classification training with Monte Carlo moments and report
  its variability against the exact reference.
- Check supported devices, dependencies, package exports and dimensions
  beyond those exercised in the current experiments.
- Move executable demonstration code out of the layer module.
- Keep the paper's architecture description consistent with the evaluated
  model.

A real-valued fine-grid filtration has also been discussed. It would
remove integer decoding when raw neural outputs are used directly,
while retaining finite-grid moment evaluation and an adapted surrogate
backward. It remains an extension to implement and test, not part of the
validated discrete pathway described above.
