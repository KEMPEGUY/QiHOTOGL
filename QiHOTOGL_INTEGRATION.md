# QiHOTOGL integration plan

Date: 22 October 2026

Starting point: `torch_topological/nn/graphs.py` in this repository.

## Purpose

Identify where the QiHOTOGL layer will be integrated and which parts of
the simplified TOGL implementation will be retained, replaced, or added.

This is an implementation plan. The QiHOTOGL layer has not yet been coded
or validated in this repository.

## 1. Current and proposed computation

The current model follows:

Node features → GCN → neural filtration scores → persistence pairs →
DeepSet encoder → node residual → GCN → graph mean pooling.

The proposed standalone QiHOTOGL model follows:

Node features → GCN feature extractor → neural filtration scores →
SORT and hard decoding → Monte Carlo Hodge moment matrices →
flatten and concatenate → graph classifier.

The GCN feature extractor and filtration network together form the
trainable scorer. The classifier uses one graph-classification loss.

## 2. Retain the neural scorer

Location: `TOGL.__init__`, attribute `self.filtrations`.

The existing scorer is:

    Linear(n_features, hidden_dim)
    ReLU()
    Linear(hidden_dim, n_filtrations)

Retain this structure to produce R scores per vertex, where
R = n_filtrations.

Keep R configurable. The existing example uses 16 filtration views.

The scorer output has shape:

    [N_batch, R]

N_batch is the total number of vertices in the current graph batch.

## 3. Insert SORT and hard decoding

Insertion point: immediately after the existing call:

    filtered_v = self.filtrations(x)

For each graph and filtration view:

1. Sort its vertex scores using a fixed ranking and tie-breaking convention.
2. Decode an injective integer filtration t.
3. Return the ordering permutation sigma.

The decoded state must satisfy:

    1 = t[sigma[0]] < ... < t[sigma[n-1]] <= k

Decoding is performed separately for each graph, not across all vertices
in the batch.

The lower-star rule remains:

    simplex_level = max(t[v] for v in simplex)

The allocation of levels when k > n must be specified before implementing
the decoder.

## 4. Replace persistent homology with Hodge moments

Locations:

- `TOGL.compute_persistent_homology`
- `TOGL._compute_persistent_homology`

Replace the persistence-pair computation with:

1. Construction of the graph's clique complex.
2. Assignment of simplex levels from the decoded vertex filtration.
3. Construction or application of the required Hodge operators.
4. Monte Carlo estimation of normalized spectral moments at each level.
5. Assembly of moment matrices for each graph, view, and selected dimension.

The following persistence-specific operations are not required in the
QiHOTOGL pathway:

- `st.persistence()`
- `st.lower_star_persistence_generators()`
- Persistence-generator indexing.
- Birth–death pair construction.
- The `persistence_diagrams` tensor.

GUDHI may still be used for complex construction; removing persistence
does not require removing the complex-construction library.

## 5. Make Hodge dimensions configurable

Add a user parameter such as:

    q_values = (0, 3)

Other examples include:

    q_values = (2,)
    q_values = (0, 1, 2, 3)

Validate the selection and use a consistent ordering of dimensions.

Derive:

    q_max = max(q_values)

The full q-Hodge Laplacian contains both boundary contributions:

    Delta_q = B_q.T @ B_q + B_(q+1) @ B_(q+1).T

Therefore, the construction corresponding to `st.expansion(2)` must
support simplices through dimension q_max + 1.

Only dimensions explicitly selected in q_values contribute moment blocks
and candidate-score contributions. Intermediate simplices are constructed
where required for those operators.

## 6. Replace the persistence encoder and node residual

Locations:

- Class `DeepSetLayer`
- `TOGL.__init__`: `self.set_fn` and `self.batch_norm`
- `TOGL.forward`: persistence reshape, encoder loop, and residual addition

The existing representation contains two persistence coordinates per
vertex and view:

    [R, N_batch, 2]

The encoder reshapes this into node features and returns a node correction.

For standalone QiHOTOGL, replace that pathway with moment blocks:

    [B_batch, R, len(q_values), k, z_max]

Flatten and concatenate the blocks for each graph:

    [B_batch, R * len(q_values) * k * z_max]

The DeepSet persistence encoder and its node residual are not used in this
standalone architecture.

## 7. Adapt the surrounding model

Location: `TopoGCN.__init__` and `TopoGCN.forward`.

Retain the first GCN as a feature extractor feeding the filtration scorer.

Replace the connection:

    TOGL → second GCN → node mean pooling

with:

    QiHOTOGL moment representation → graph classifier

The classifier input width is derived from:

    R * len(q_values) * k * z_max

The classifier outputs:

    [B_batch, num_classes]

This surrounding-model change is necessary because the existing TOGL
returns node features, whereas standalone QiHOTOGL produces graph features.

## 8. Add the configurable backward connection

The classification objective is:

    L = CrossEntropy(logits, labels)

Obtain ordinary classification-loss gradients with respect to the moment
matrices and classifier parameters.

A configurable backward operator then constructs the score-space signal.
Propagate that signal through the differentiable scorer using a
vector–Jacobian product.

There is no auxiliary filtration loss.

### SELECT-based methods: M1, M2, M3

For each graph and view:

1. Apply Algorithm 1 to all admissible provisional unit directions.
2. Sum score contributions over the selected Hodge dimensions.
3. Select the best strictly positive score.
4. Break ties by smallest vertex rank, then decrement before increment.
5. Apply Algorithm 2 to construct one direct move or adjacent swap.
6. Convert the selection information into the configured backward signal.

Use one iteration per graph/view/training step, initially with unit
filtration steps.

If no candidate has a positive score, return zero surrogate credit.

Keep separate:

- Selected vertex rank.
- Selected vertex index.
- Direction.
- Affected moment-row level.
- Provisional candidate score.
- Realization type: direct or swap.
- Swap partner, if present.
- Realized filtration displacement.

Provisional scoring allows ties. Algorithm 2 restores injectivity through
a direct update or adjacent swap. A swap changes two vertex assignments
but at most one moment row per selected Hodge dimension.

The provisional score need not equal the score of the realized swap.

The selected candidate supplies a learning signal. It does not permanently
overwrite the forward filtration. After the optimizer update, compute new
scores and decode again.

### Experimental modes

| Mode | Role |
| --- | --- |
| M0 | Frozen randomly initialized scorer; classifier learning only. |
| M1 | SELECT with soft-boundary credit. |
| M2 | SELECT with selected-index credit. |
| M3 | SELECT with task-weighted displacement credit. |
| M4 | Independent random score-space signal. |
| M5 | Fixed degree-based scores followed by hard decoding. |

M0 and M5 are filtration controls. M4 is a random backward control.
They do not require Algorithms 1–2.

For M0, freeze the entire score-producing pathway and control stochastic
layers so that its filtration remains fixed for a fixed input.

For M5, repeated identical degree views are not distinct learned views.

At inference, use score construction, hard decoding, moment estimation,
and classification only. No SELECT or surrogate backward is required.

## 9. Separate the example and training code

Location: the demonstration beginning with `B = 64`.

Move example execution out of the layer module and into a separate script.

Add:

- Graph labels and train/validation/test splits.
- Classification loss and optimizer.
- The configured backward connection.
- Reproducible random seeds.
- Separate counts of direct moves and swaps.
- Actual filtration-change and selected-target realization diagnostics.

The current demonstration performs forward calls only. It does not train
the model.

## 10. Implementation organization

Implement the new QiHOTOGL layer separately, for example in:

    torch_topological/nn/qihotogl.py

Separate responsibilities for:

- Hard decoding.
- Complex and Hodge operations.
- Monte Carlo moment estimation.
- Direction selection and realization.
- Configurable backward rules.

Retain the original TOGL source as a comparison reference; it is not part
of the standalone QiHOTOGL forward pathway.

Check package exports when adding the new layer.

## 11. Details to finalize during implementation

- Decoder ranking, deterministic ties, and level allocation for k > n.
- Monte Carlo walk estimator and sampling controls.
- Hodge normalization and empty-chain convention.
- Exact M1 and M2 credit formulas.
- Backward scaling and sign conventions.
- Dependency versions and device/batching behavior.

The decoder's reachable assignments must be checked against the selected
direct-move and swap targets.

These implementation details do not change the integration points
identified above.
