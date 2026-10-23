"""Experimental QiHOTOGL graph layer and small training demonstration.

The layer decodes anchored, injective neural vertex filtrations, computes
exact or Monte Carlo Hodge moments, and projects graph moment features
into residual node corrections.

Training uses classification backward followed by explicit M2/M3
surrogate backward before one optimizer step. Teacher moves provide
learning signals; they do not overwrite neural-decoded filtrations.
Inference uses neural decoding without teacher selection.

Candidate moves currently use epsilon = 1. M2/M3 are surrogate rules,
not exact derivatives of the hard decoder. Other backward modes remain
pending. Exact moments use dense reference calculations for small graphs.

Run the recorded demonstration with:
    python -m torch_topological.nn.qihotogl

The synthetic experiment checks training behavior and records metrics;
it does not establish a general classification advantage.

Original TOGL reference:
    https://github.com/BorgwardtLab/TOGL
"""

import torch
import torch.nn as nn

import gudhi as gd

from torch_geometric.data import Data

from torch_geometric.loader import DataLoader

from torch_geometric.utils import erdos_renyi_graph

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from torch_scatter import scatter

from torch_topological.utils import pairwise


def validate_current_filtration(t, k):
    """Validate a current injective assignment and return sigma.

    t[i] is the integer level of original vertex i.
    Gaps between occupied levels are allowed.
    """
    # Require one integer level per vertex.
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor.")

    if t.ndim != 1 or t.numel() == 0:
        raise ValueError("t must be a nonempty vector.")

    if t.dtype not in (
        torch.int8, torch.int16, torch.int32,
        torch.int64, torch.uint8,
    ):
        raise TypeError("t must have an integer dtype.")

    # Require enough available levels for an injective assignment.
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a Python integer.")

    if k < t.numel():
        raise ValueError("k must be at least the number of vertices.")

    # Sort without changing t; sigma records original vertex indices.
    ordered, sigma = torch.sort(t)

    # Check anchoring and the maximum level.
    if ordered[0].item() != 1:
        raise ValueError("The minimum filtration level must be 1.")

    if ordered[-1].item() > k:
        raise ValueError("Filtration levels must not exceed k.")

    # Strict increase permits gaps but rejects tied current levels.
    if not torch.all(ordered[1:] > ordered[:-1]).item():
        raise ValueError("Current filtration levels must be distinct.")

    return sigma


def realise_filtration_direction(t, sigma, m, direction, k):
    """Realise one selected direction according to Algorithm 2.

    Implements unit steps (epsilon = 1).
    General epsilon-perturbations are not yet implemented.

    m is the current arrival rank, not an original vertex index.
    Return (new_t, new_sigma, move_type), without modifying inputs.
    """
    # Check the current assignment and supplied ordering.
    expected_sigma = validate_current_filtration(t, k)

    if not isinstance(sigma, torch.Tensor):
        raise TypeError("sigma must be a torch.Tensor.")
    if sigma.dtype != torch.long or sigma.device != t.device:
        raise ValueError("sigma must be long and on the same device as t.")
    if not torch.equal(sigma, expected_sigma):
        raise ValueError("sigma does not match the ordering of t.")

    n = t.numel()

    # Rank zero is the anchored vertex and cannot be selected.
    if isinstance(m, bool) or not isinstance(m, int):
        raise TypeError("m must be a Python integer.")
    if not 1 <= m < n:
        raise ValueError("Selected rank must satisfy 1 <= m < n.")
    if direction not in ("-", "+"):
        raise ValueError("direction must be '-' or '+'.")

    # Work on copies; long integers avoid narrow-integer overflow.
    new_t = t.to(dtype=torch.long).clone()
    new_sigma = sigma.clone()

    a = int(sigma[m].item())
    s = int(t[a].item())
    neighbour_rank = None

    if direction == "-":
        # Exclude decrementing the first movable vertex into level 1.
        if m == 1 and s == 2:
            raise ValueError("Cannot decrement into the fixed level 1.")

        target = s - 1

        # A preceding movable vertex at the target causes a swap.
        if m >= 2:
            b = int(sigma[m - 1].item())
            if int(t[b].item()) == target:
                neighbour_rank = m - 1

    else:
        if s >= k:
            raise ValueError("Cannot increment above k.")

        target = s + 1

        # A following vertex at the target causes a swap.
        if m < n - 1:
            b = int(sigma[m + 1].item())
            if int(t[b].item()) == target:
                neighbour_rank = m + 1

    if neighbour_rank is None:
        # Empty target level: change only the selected vertex.
        new_t[a] = target
        move_type = "direct"
    else:
        # Occupied target: exchange levels and ordering positions.
        b = int(sigma[neighbour_rank].item())
        new_t[a] = int(t[b].item())
        new_t[b] = s
        new_sigma[m] = b
        new_sigma[neighbour_rank] = a
        move_type = "swap"

    # Verify Algorithm 2's final invariant.
    checked_sigma = validate_current_filtration(new_t, k)
    if not torch.equal(new_sigma, checked_sigma):
        raise RuntimeError("Updated ordering is inconsistent.")

    return new_t, new_sigma, move_type


@torch.no_grad()
def select_filtration_direction(t, sigma, phi, gamma, k):
    """Select the best positive provisional direction, Algorithm 1.

    Implements unit steps (epsilon = 1) for one graph and one view.
    General epsilon-perturbations are not yet implemented.
    phi and gamma have shape [k, z]; gamma is the NEGATIVE loss gradient.
    Moment columns for selected dimensions may be concatenated.

    Ties prefer the smallest arrival rank, then decrement over increment.
    Return (rank, direction, score), or None if no positive candidate.
    Inputs remain unchanged. Algorithm 2 realises the selected direction
    separately; provisional scoring and swap realisation are distinct.
    """
    expected_sigma = validate_current_filtration(t, k)

    if not isinstance(sigma, torch.Tensor):
        raise TypeError("sigma must be a torch.Tensor.")
    if sigma.dtype != torch.long or sigma.device != t.device:
        raise ValueError("sigma must be long and on the same device as t.")
    if not torch.equal(sigma, expected_sigma):
        raise ValueError("sigma does not match t.")

    if not isinstance(phi, torch.Tensor):
        raise TypeError("phi must be a torch.Tensor.")
    if not isinstance(gamma, torch.Tensor):
        raise TypeError("gamma must be a torch.Tensor.")
    if phi.ndim != 2 or phi.shape[0] != k or phi.shape[1] == 0:
        raise ValueError("phi must have shape [k, z] with z > 0.")
    if gamma.shape != phi.shape:
        raise ValueError("gamma and phi must have the same shape.")
    if not phi.is_floating_point() or not gamma.is_floating_point():
        raise TypeError("phi and gamma must be floating-point tensors.")
    if phi.device != gamma.device or phi.dtype != gamma.dtype:
        raise ValueError("phi and gamma must share device and dtype.")
    if not torch.isfinite(phi).all().item():
        raise ValueError("phi must contain finite values.")
    if not torch.isfinite(gamma).all().item():
        raise ValueError("gamma must contain finite values.")

    best = None
    best_score = 0.0

    # Increasing rank, with decrement considered first, resolves ties.
    for m in range(1, t.numel()):
        vertex = int(sigma[m].item())
        s = int(t[vertex].item())

        # Algorithm 1, lines 7-9: admissible provisional decrement.
        if m >= 2 or s > 2:
            delta = torch.dot(
                gamma[s - 2],
                phi[s - 1] - phi[s - 2],
            ).item()

            if delta > best_score:
                best_score = delta
                best = (m, "-", delta)

        # Algorithm 1, lines 11-13: admissible provisional increment.
        if s < k:
            delta = torch.dot(
                gamma[s - 1],
                phi[s - 2] - phi[s - 1],
            ).item()

            if delta > best_score:
                best_score = delta
                best = (m, "+", delta)

    return best


def filtered_clique_simplices(edge_index, t, level, max_dim):
    """Build the induced clique complex at one filtration level.

    Parameters
    ----------
    edge_index : integer tensor, shape [2, number_of_edges]
        Edges of one graph, with vertices numbered 0, ..., n-1.
        Reverse edges and duplicates are deduplicated.
        Self-loops do not contribute simplices.
    t : integer tensor, shape [n]
        Positive vertex filtration levels, with minimum 1.
        Ties are permitted for provisional scoring assignments.
    level : int
        Positive filtration threshold.
    max_dim : int
        Largest simplex dimension to construct.

    Returns
    -------
    dict
        Dimension -> sorted list of vertex tuples.
        Original vertex labels are preserved. Ascending vertex tuples
        provide the canonical orientation for boundary matrices.

    Notes
    -----
    This CPU enumeration is an exact reference for small graphs.
    It supplies no gradient through the discrete simplex construction.
    """
    from itertools import combinations

    integer_dtypes = (
        torch.int8, torch.int16, torch.int32,
        torch.int64, torch.uint8,
    )

    # Validate assignments without requiring injectivity.
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor.")
    if t.ndim != 1 or t.numel() == 0:
        raise ValueError("t must be a nonempty vector.")
    if t.dtype not in integer_dtypes:
        raise TypeError("t must have an integer dtype.")
    if t.min().item() != 1:
        raise ValueError("The minimum filtration level must be 1.")

    for name, value in (("level", level), ("max_dim", max_dim)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(name + " must be a Python integer.")

    if level < 1:
        raise ValueError("level must be positive.")
    if max_dim < 0:
        raise ValueError("max_dim must be nonnegative.")

    # Validate graph connectivity.
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a torch.Tensor.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.dtype not in integer_dtypes:
        raise TypeError("edge_index must have an integer dtype.")

    n = t.numel()
    if edge_index.numel():
        if edge_index.min().item() < 0:
            raise ValueError("Vertex indices must be nonnegative.")
        if edge_index.max().item() >= n:
            raise ValueError("An edge references a vertex outside t.")

    # Select active vertices, retaining original labels.
    assignments = t.detach().cpu().tolist()
    active = tuple(
        vertex for vertex, value in enumerate(assignments)
        if value <= level
    )
    active_set = set(active)

    # Interpret connectivity as a simple undirected graph.
    edges = set()
    for u, v in edge_index.detach().cpu().t().tolist():
        if u != v and u in active_set and v in active_set:
            edges.add(tuple(sorted((u, v))))

    simplices = {dimension: [] for dimension in range(max_dim + 1)}
    simplices[0] = [(vertex,) for vertex in active]

    if max_dim >= 1:
        simplices[1] = sorted(edges)

    # A higher simplex exists exactly when all its edges exist.
    largest_possible_dim = min(max_dim, len(active) - 1)
    for dimension in range(2, largest_possible_dim + 1):
        simplices[dimension] = [
            simplex
            for simplex in combinations(active, dimension + 1)
            if all(edge in edges for edge in combinations(simplex, 2))
        ]

    return simplices


def oriented_boundary_matrix(simplices, q):
    """Construct B_q: C_q -> C_(q-1) in canonical orientations.

    simplices is the dictionary returned by filtered_clique_simplices.
    Rows index (q-1)-simplices; columns index q-simplices.
    B_0 has zero rows, following ordinary unreduced homology.

    Returns a CPU float64 tensor for exact-reference calculations.
    """
    if isinstance(q, bool) or not isinstance(q, int):
        raise TypeError("q must be a Python integer.")
    if q < 0:
        raise ValueError("q must be nonnegative.")
    if q not in simplices:
        raise ValueError("Requested simplex dimension was not constructed.")
    if q > 0 and q - 1 not in simplices:
        raise ValueError("Required face dimension was not constructed.")

    upper = simplices[q]
    lower = [] if q == 0 else simplices[q - 1]

    # Check canonical simplex orientations and exclude duplicate entries.
    for dimension, entries in ((q, upper), (q - 1, lower)):
        if len(set(entries)) != len(entries):
            raise ValueError("Simplex lists must not contain duplicates.")
        for simplex in entries:
            if len(simplex) != dimension + 1:
                raise ValueError("Simplex has an incorrect dimension.")
            if any(
                simplex[i] >= simplex[i + 1]
                for i in range(len(simplex) - 1)
            ):
                raise ValueError("Simplex vertices must be strictly increasing.")

    boundary = torch.zeros(
        (len(lower), len(upper)), dtype=torch.float64
    )

    # Vertices have zero boundary in ordinary homology.
    if q == 0:
        return boundary

    row_index = {simplex: i for i, simplex in enumerate(lower)}

    for column, simplex in enumerate(upper):
        for position in range(q + 1):
            face = simplex[:position] + simplex[position + 1:]

            if face not in row_index:
                raise ValueError("A simplex is missing a required face.")

            boundary[row_index[face], column] = (
                1.0 if position % 2 == 0 else -1.0
            )

    return boundary


def exact_hodge_moments_at_level(edge_index, t, level, q, z_max):
    """Compute exact-reference reflected moments at one level.

    Uses H_q = I - Delta_q / n_r, where n_r counts active vertices.
    Returns CPU float64 moments of orders 1, ..., z_max.

    Empty q-chain spaces return zeros.
    A nonempty zero-Laplacian space returns moments equal to one.
    Provisional assignments with ties are supported.
    """
    for name, value in (("q", q), ("z_max", z_max)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(name + " must be a Python integer.")

    if q < 0:
        raise ValueError("q must be nonnegative.")
    if z_max < 1:
        raise ValueError("z_max must be positive.")

    # Full q-Hodge Laplacians require simplices through dimension q+1.
    simplices = filtered_clique_simplices(
        edge_index, t, level, max_dim=q + 1
    )

    n_r = len(simplices[0])
    d_q = len(simplices[q])

    # The manuscript defines an empty q-chain-space row as zero.
    if d_q == 0:
        return torch.zeros(z_max, dtype=torch.float64)

    B_q = oriented_boundary_matrix(simplices, q)
    B_next = oriented_boundary_matrix(simplices, q + 1)

    delta = B_q.T @ B_q + B_next @ B_next.T

    # Anchoring guarantees at least one active vertex at levels >= 1.
    identity = torch.eye(d_q, dtype=torch.float64)
    reflected = identity - delta / n_r

    # Repeated multiplication gives powers 1, ..., z_max.
    power = identity
    moments = torch.empty(z_max, dtype=torch.float64)

    for index in range(z_max):
        power = power @ reflected
        moments[index] = torch.trace(power) / d_q

    return moments


def exact_hodge_moment_matrices(edge_index, t, k, q_values, z_max):
    """Return exact-reference moment matrices for selected dimensions.

    Processes one graph and one filtration view.
    Returns {q: tensor of shape [k, z_max]} in the requested order.
    Uses active-vertex normalization and CPU float64 calculations.
    Ties are permitted in provisional assignments.

    This reference implementation prioritizes clarity over performance.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a Python integer.")
    if k < 1:
        raise ValueError("k must be positive.")

    if isinstance(z_max, bool) or not isinstance(z_max, int):
        raise TypeError("z_max must be a Python integer.")
    if z_max < 1:
        raise ValueError("z_max must be positive.")

    q_values = tuple(q_values)
    if not q_values:
        raise ValueError("Select at least one simplex dimension.")
    for q in q_values:
        if isinstance(q, bool) or not isinstance(q, int):
            raise TypeError("Each selected dimension must be a Python integer.")
        if q < 0:
            raise ValueError("Selected dimensions must be nonnegative.")
    if len(set(q_values)) != len(q_values):
        raise ValueError("Selected dimensions must not contain duplicates.")

    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor.")
    if t.ndim != 1 or t.numel() == 0:
        raise ValueError("t must be a nonempty vector.")
    if t.dtype not in (
        torch.int8, torch.int16, torch.int32,
        torch.int64, torch.uint8,
    ):
        raise TypeError("t must have an integer dtype.")
    if k < t.numel():
        raise ValueError("k must be at least the number of vertices.")
    if t.min().item() != 1 or t.max().item() > k:
        raise ValueError("Assignments must lie in [1, k] with minimum 1.")

    matrices = {}
    for q in q_values:
        rows = []
        previous_active = None
        previous_row = None

        for level in range(1, k + 1):
            active = tuple(
                torch.nonzero(t <= level, as_tuple=False)
                .flatten().cpu().tolist()
            )

            # No new vertex means the entire induced complex is unchanged.
            if active == previous_active:
                row = previous_row
            else:
                row = exact_hodge_moments_at_level(
                    edge_index, t, level, q, z_max
                )

            rows.append(row)
            previous_active = active
            previous_row = row

        matrices[q] = torch.stack(rows)

    return matrices


def sparse_reflected_hodge_columns(edge_index, t, level, q):
    """Construct sparse columns of H_q = I - Delta_q / n_r.

    Returns:
        faces: canonical q-simplex ordering
        columns: destination row indices for each column
        weights: corresponding signed H entries
        n_vertices: active vertex count

    No dense boundary or Hodge matrix is constructed.
    Provisional assignments with ties are supported.
    Simplex enumeration remains an exact-reference implementation.
    """
    if isinstance(q, bool) or not isinstance(q, int):
        raise TypeError("q must be a Python integer.")
    if q < 0:
        raise ValueError("q must be nonnegative.")

    simplices = filtered_clique_simplices(
        edge_index, t, level, max_dim=q + 1
    )
    active = tuple(simplex[0] for simplex in simplices[0])
    n_r = len(active)
    faces = simplices[q]
    face_index = {face: i for i, face in enumerate(faces)}
    edges = set(simplices[1])

    def connected(u, v):
        return u != v and tuple(sorted((u, v))) in edges

    columns = []
    weights = []

    for column, face in enumerate(faces):
        entries = {}

        if q == 0:
            # For vertices: Delta_0 = D - A.
            vertex = face[0]
            neighbours = [
                v for v in active if connected(vertex, v)
            ]
            entries[column] = 1.0 - len(neighbours) / n_r

            for v in neighbours:
                entries[face_index[(v,)]] = 1.0 / n_r

        else:
            # Diagonal: q+1 boundary faces plus upper cofaces.
            face_set = set(face)
            coface_count = sum(
                all(connected(v, w) for w in face)
                for v in active if v not in face_set
            )
            entries[column] = (
                1.0 - (q + 1 + coface_count) / n_r
            )

            for v in active:
                if v in face_set:
                    continue

                for removed_position, removed_vertex in enumerate(face):
                    # If the union is a coface, the lower and upper
                    # off-diagonal contributions cancel.
                    if connected(v, removed_vertex):
                        continue
                    if not all(
                        connected(v, w)
                        for w in face if w != removed_vertex
                    ):
                        continue

                    neighbour = tuple(sorted(
                        (face_set - {removed_vertex}) | {v}
                    ))
                    inserted_position = neighbour.index(v)

                    # Reflection negates the lower-boundary coefficient.
                    sign = (
                        -1.0
                        if (removed_position + inserted_position) % 2 == 0
                        else 1.0
                    )
                    row = face_index[neighbour]
                    entries[row] = entries.get(row, 0.0) + sign / n_r

        # Keep signed entries; discard only exact zeros.
        nonzero = sorted(
            (row, value)
            for row, value in entries.items()
            if value != 0.0
        )
        columns.append(torch.tensor(
            [row for row, _ in nonzero], dtype=torch.long
        ))
        weights.append(torch.tensor(
            [value for _, value in nonzero], dtype=torch.float64
        ))

    return {
        "faces": faces,
        "columns": columns,
        "weights": weights,
        "n_vertices": n_r,
    }


def prepare_hodge_walk_table(sparse, device="cpu"):
    """Prepare signed importance-sampling data from sparse H columns.

    Column j occupies entries offsets[j]:offsets[j+1].
    Its local cumulative probabilities run from 0 to 1.
    A zero column receives a self-transition with zero path weight.
    """
    columns = sparse["columns"]
    weights = sparse["weights"]
    nstates = len(sparse["faces"])

    if len(columns) != nstates or len(weights) != nstates:
        raise ValueError("Sparse column counts do not match the faces.")

    destinations = []
    signs = []
    cumulative = []
    norms = []
    offsets = [0]

    for j, (rows, values) in enumerate(zip(columns, weights)):
        if rows.ndim != 1 or values.ndim != 1:
            raise ValueError("Sparse columns must be vectors.")
        if rows.numel() != values.numel():
            raise ValueError("Rows and weights must have matching lengths.")
        if rows.dtype != torch.long:
            raise TypeError("Destination indices must have dtype long.")

        rows = rows.detach().cpu()
        values = values.detach().to(device="cpu", dtype=torch.float64)

        if not torch.isfinite(values).all().item():
            raise ValueError("Transition weights must be finite.")
        if rows.numel() and (
            rows.min().item() < 0 or rows.max().item() >= nstates
        ):
            raise ValueError("Destination index is out of bounds.")
        if torch.unique(rows).numel() != rows.numel():
            raise ValueError("Duplicate row entries must be combined first.")

        # Ignore exact zero entries, retaining both positive and negative ones.
        keep = values != 0
        rows = rows[keep]
        values = values[keep]

        norm = values.abs().sum()
        if not torch.isfinite(norm).item():
            raise ValueError("Column norm is not finite.")
        norms.append(norm.item())

        if norm.item() == 0:
            # The sampling probability is defined artificially here,
            # but the zero multiplier makes the contribution vanish.
            rows = torch.tensor([j], dtype=torch.long)
            column_signs = torch.zeros(1, dtype=torch.float64)
            cdf = torch.ones(1, dtype=torch.float64)
        else:
            probabilities = values.abs() / norm
            cdf = probabilities.cumsum(0)
            cdf[-1] = 1.0
            column_signs = values.sign()

        destinations.append(rows)
        signs.append(column_signs)
        cumulative.append(cdf)
        offsets.append(offsets[-1] + rows.numel())

    # Empty chain spaces produce empty sampling arrays.
    def join(parts, dtype):
        if parts:
            return torch.cat(parts).to(device=device)
        return torch.empty(0, dtype=dtype, device=device)

    return {
        "nstates": nstates,
        "offsets": torch.tensor(
            offsets, dtype=torch.long, device=device
        ),
        "destinations": join(destinations, torch.long),
        "signs": join(signs, torch.float64),
        "cdf": join(cumulative, torch.float64),
        "column_norms": torch.tensor(
            norms, dtype=torch.float64, device=device
        ),
    }


@torch.no_grad()
def estimate_hodge_walk_moments(
    table, z_max, n_samples, seed=0, batch_size=4096
):
    """Estimate normalized traces of H^z using signed closed walks.

    Return one estimate for every order from 1 through z_max.
    The table must come from prepare_hodge_walk_table().
    Empty chain spaces return zeros.
    """
    for name, value in (
        ("z_max", z_max),
        ("n_samples", n_samples),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(name + " must be a Python integer.")
        if value < 1:
            raise ValueError(name + " must be positive.")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a Python integer.")

    norms = table["column_norms"]
    device = norms.device
    totals = torch.zeros(z_max, dtype=torch.float64, device=device)
    nstates = table["nstates"]

    if nstates == 0:
        return totals

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    offsets = table["offsets"]
    destinations = table["destinations"]
    signs = table["signs"]
    cdf = table["cdf"]

    completed = 0
    while completed < n_samples:
        size = min(batch_size, n_samples - completed)

        start = torch.randint(
            nstates, (size,), device=device, generator=generator
        )
        current = start.clone()
        product = torch.ones(
            size, dtype=torch.float64, device=device
        )

        for order in range(z_max):
            # Read all transitions from the states before this step.
            previous = current
            next_states = previous.clone()

            uniforms = torch.rand(
                size,
                dtype=torch.float64,
                device=device,
                generator=generator,
            )

            # Sample within each column's own CDF.
            for state in torch.unique(previous).tolist():
                mask = previous == state
                lo = int(offsets[state].item())
                hi = int(offsets[state + 1].item())

                local_cdf = cdf[lo:hi]
                chosen = torch.searchsorted(
                    local_cdf, uniforms[mask], right=True
                )
                chosen = chosen.clamp(max=hi - lo - 1)
                positions = lo + chosen

                next_states[mask] = destinations[positions]
                product[mask] *= norms[state] * signs[positions]

            current = next_states

            contribution = torch.where(
                current == start, product, torch.zeros_like(product)
            )
            totals[order] += contribution.sum()

        completed += size

    return totals / n_samples


@torch.no_grad()
def monte_carlo_hodge_moment_matrices(
    edge_index,
    t,
    k,
    q_values,
    z_max,
    n_samples,
    seed=0,
    batch_size=4096,
    device="cpu",
):
    """Estimate moment matrices for one graph and one filtration.

    Return a dictionary mapping each requested q to a [k, z_max]
    tensor. Provisional tied levels are allowed. Unchanged complexes
    reuse the preceding row, preserving block-constant matrices.
    """
    for name, value in (
        ("k", k),
        ("z_max", z_max),
        ("n_samples", n_samples),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(name + " must be a Python integer.")
        if value < 1:
            raise ValueError(name + " must be positive.")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a Python integer.")

    q_values = tuple(q_values)
    if not q_values:
        raise ValueError("Select at least one simplex dimension.")

    for q in q_values:
        if isinstance(q, bool) or not isinstance(q, int):
            raise TypeError("Each q must be a Python integer.")
        if q < 0:
            raise ValueError("Simplex dimensions must be nonnegative.")

    if len(set(q_values)) != len(q_values):
        raise ValueError("Selected dimensions must be distinct.")

    # Reuse the complex builder's validation of edges and assignments.
    # It permits ties, which are needed for provisional move scoring.
    filtered_clique_simplices(
        edge_index, t, level=1, max_dim=0
    )

    if k < t.numel():
        raise ValueError("k must be at least the number of vertices.")
    if t.max().item() > k:
        raise ValueError("Filtration levels must not exceed k.")

    matrices = {}

    for q in q_values:
        rows = []
        previous_active = None
        previous_row = None

        for level in range(1, k + 1):
            active = tuple(
                torch.nonzero(t <= level, as_tuple=False)
                .flatten().tolist()
            )

            if active != previous_active:
                sparse = sparse_reflected_hodge_columns(
                    edge_index, t, level=level, q=q
                )
                table = prepare_hodge_walk_table(
                    sparse, device=device
                )

                # Deterministic seeds indexed by dimension and level.
                # The result does not depend on q_values ordering.
                local_seed = (
                    seed + q * (k + 1) + level
                ) % (2**63 - 1)

                previous_row = estimate_hodge_walk_moments(
                    table,
                    z_max=z_max,
                    n_samples=n_samples,
                    seed=local_seed,
                    batch_size=batch_size,
                )
                previous_active = active

            rows.append(previous_row)

        matrices[q] = torch.stack(rows, dim=0)

    return matrices


@torch.no_grad()
def decode_filtration_scores(
    scores, k, mode="score_gaps", tau=1.0
):
    """Decode one graph/view into an anchored injective filtration.

    Return (t, sigma), both integer tensors on scores.device.
    Stable ascending sorting resolves equal scores by vertex index.

    rank:
        Assign consecutive levels 1 through n.
    score_gaps:
        Add rounded, nondecreasing offsets determined by score
        distances from the minimum. tau controls the gap scale.

    This hard decoder supplies no ordinary gradient through its
    discrete decisions. Surrogate backward rules are separate.
    """
    import math

    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor.")
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty vector.")
    if not scores.is_floating_point():
        raise TypeError("scores must have a floating-point dtype.")
    if not torch.isfinite(scores).all().item():
        raise ValueError("scores must be finite.")

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a Python integer.")

    n = scores.numel()
    if k < n:
        raise ValueError("k must be at least the number of vertices.")
    if k > torch.iinfo(torch.int64).max:
        raise ValueError("k exceeds the supported integer range.")

    if mode not in ("rank", "score_gaps"):
        raise ValueError("mode must be 'rank' or 'score_gaps'.")

    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise TypeError("tau must be a Python number.")
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive.")

    sigma = torch.argsort(scores, stable=True)
    levels = torch.arange(
        1, n + 1, dtype=torch.long, device=scores.device
    )
    extra = k - n

    if mode == "score_gaps" and extra > 0:
        # Float64 reduces rounding error in score normalization.
        ordered = scores[sigma].to(torch.float64)
        distances = ordered - ordered[0]
        scaled = (distances / tau).clamp(min=0.0, max=1.0)

        # Avoid silently losing integer precision for huge grids.
        if extra > 2**52:
            raise ValueError(
                "score_gaps supports k - n up to 2**52."
            )

        offsets = torch.floor(extra * scaled + 0.5).long()
        offsets = offsets.clamp(min=0, max=extra)
        levels = levels + offsets

    t = torch.empty(n, dtype=torch.long, device=scores.device)
    t[sigma] = levels

    # Check the same admissibility contract used by the optimizer.
    checked_sigma = validate_current_filtration(t, k)
    if not torch.equal(sigma, checked_sigma):
        raise RuntimeError("Decoded ordering is inconsistent.")

    return t, sigma


@torch.no_grad()
def prepare_filtration_teacher(
    scores,
    edge_index,
    k,
    q_values,
    z_max,
    decoder_mode="score_gaps",
    tau=1.0,
    moment_method="exact",
    n_samples=10000,
    seed=0,
    batch_size=4096,
):
    """Prepare one graph/view before classification backward.

    phi concatenates the selected q blocks along the moment columns.
    This helper creates no autograd connection to the scores.
    """
    q_values = tuple(q_values)
    t, sigma = decode_filtration_scores(
        scores, k, mode=decoder_mode, tau=tau
    )

    if moment_method == "exact":
        matrices = exact_hodge_moment_matrices(
            edge_index, t, k, q_values, z_max
        )
    elif moment_method == "monte_carlo":
        matrices = monte_carlo_hodge_moment_matrices(
            edge_index, t, k, q_values, z_max,
            n_samples=n_samples,
            seed=seed,
            batch_size=batch_size,
            device=scores.device,
        )
    else:
        raise ValueError(
            "moment_method must be 'exact' or 'monte_carlo'."
        )

    phi = torch.cat(
        [matrices[q] for q in q_values], dim=1
    ).to(scores.device)

    return {
        "t": t,
        "sigma": sigma,
        "k": k,
        "q_values": q_values,
        "z_max": z_max,
        "phi": phi,
        "moment_method": moment_method,
    }


@torch.no_grad()
def single_step_filtration_teacher(context, gamma):
    """Select and realise at most one direction for one graph/view.

    gamma must be the NEGATIVE classification-loss gradient with
    respect to context["phi"], with matching shape, dtype and device.

    Selection scores provisional moves. A realised swap can have a
    different moment change. The selection score is not a guarantee
    of realised loss decrease.

    Monte Carlo input produces estimated, noise-sensitive scores.
    """
    t = context["t"]
    sigma = context["sigma"]
    phi = context["phi"]
    k = context["k"]

    selected = select_filtration_direction(
        t, sigma, phi, gamma, k
    )

    if selected is None:
        target_t = t.clone()
        target_sigma = sigma.clone()
        move_type = "none"
    else:
        m, direction, score = selected
        target_t, target_sigma, move_type = (
            realise_filtration_direction(
                t, sigma, m, direction, k
            )
        )

    return {
        "selected": selected,
        "target_t": target_t,
        "target_sigma": target_sigma,
        "displacement": target_t.long() - t.long(),
        "move_type": move_type,
    }


@torch.no_grad()
def filtration_surrogate_signal(
    scores, context, teacher, mode="M3"
):
    """Construct an experimental M2 or M3 score-gradient signal.

    M2: -S * direction * one_hot(selected_vertex).
    M3: -S * realised_displacement.

    S is the positive provisional score returned by SELECT.
    This is a detached surrogate, not an exact decoder derivative.
    """
    import math

    if mode not in ("M2", "M3"):
        raise ValueError("mode must be 'M2' or 'M3'.")

    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor.")
    if scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError("scores must be a floating-point vector.")
    if scores.numel() != context["t"].numel():
        raise ValueError("scores and filtration sizes must match.")
    if not torch.isfinite(scores).all().item():
        raise ValueError("scores must be finite.")

    signal = torch.zeros_like(scores)
    selected = teacher["selected"]

    if selected is None:
        return signal

    m, direction, score = selected
    score = float(score)

    if not math.isfinite(score) or score <= 0:
        raise ValueError("Selected score must be finite and positive.")
    if direction not in ("-", "+"):
        raise ValueError("Unknown selected direction.")
    if not 1 <= m < scores.numel():
        raise ValueError("Selected rank must be movable.")

    if mode == "M2":
        vertex = int(context["sigma"][m].item())
        delta = -1 if direction == "-" else 1
        signal[vertex] = -score * delta
    else:
        displacement = teacher["displacement"]
        if displacement.shape != scores.shape:
            raise ValueError("Displacement shape must match scores.")
        signal = -score * displacement.to(
            device=scores.device, dtype=scores.dtype
        )

    if not torch.isfinite(signal).all().item():
        raise ValueError("Surrogate signal overflowed.")

    return signal


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, dim_in, dim_out, aggregation_fn):
        """Create new deep set layer.

        Parameters
        ----------
        dim_in : int
            Input dimension

        dim_out : int
            Output dimension

        aggregation_fn : str
            Aggregation to use for the reduction step. Must be valid for
            the ``torch_scatter.scatter()`` function, i.e. one of "sum",
            "mul", "mean", "min" or "max".
        """
        super().__init__()

        self.Gamma = nn.Linear(dim_in, dim_out)
        self.Lambda = nn.Linear(dim_in, dim_out, bias=False)

        self.aggregation_fn = aggregation_fn

    def forward(self, x, batch):
        """Implement forward pass through layer."""
        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)

        x = self.Gamma(x)
        x = x - xm[batch, :]
        return x


class QiHOTOGL(nn.Module):
    """Experimental batched moment-based graph layer.

    Each graph/view produces an anchored hard filtration and moments.
    A graph-level correction is broadcast to its node embeddings.

    Training:
        loss.backward(retain_graph=True)
        layer.backward_surrogate()
        optimizer.step()

    M2/M3 are detached surrogate rules, not exact decoder gradients.
    Only one outstanding training forward is supported per instance.
    """

    def __init__(
        self,
        n_features,
        n_filtrations,
        hidden_dim,
        out_dim,
        aggregation_fn,
        *,
        k=128,
        q_values=(0, 1),
        z_max=4,
        decoder_mode="score_gaps",
        tau=1.0,
        moment_method="exact",
        backward_mode="M3",
        n_samples=10000,
        seed=0,
        batch_size=4096,
    ):
        super().__init__()

        for name, value in (
            ("n_features", n_features),
            ("n_filtrations", n_filtrations),
            ("hidden_dim", hidden_dim),
            ("out_dim", out_dim),
            ("k", k),
            ("z_max", z_max),
            ("n_samples", n_samples),
            ("batch_size", batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a Python integer.")
            if value < 1:
                raise ValueError(name + " must be positive.")

        q_values = tuple(q_values)
        if not q_values or any(
            isinstance(q, bool) or not isinstance(q, int) or q < 0
            for q in q_values
        ):
            raise ValueError("q_values must contain nonnegative integers.")
        if len(set(q_values)) != len(q_values):
            raise ValueError("q_values must be distinct.")
        if moment_method not in ("exact", "monte_carlo"):
            raise ValueError("Unknown moment method.")
        if backward_mode not in ("M2", "M3"):
            raise ValueError("This version supports M2 and M3.")
        if aggregation_fn != "mean":
            raise ValueError(
                "This graph-moment version currently supports 'mean'."
            )

        # Validate decoder options using a one-vertex assignment.
        decode_filtration_scores(
            torch.zeros(1), k, mode=decoder_mode, tau=tau
        )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be a Python integer.")

        self.n_features = n_features
        self.n_filtrations = n_filtrations
        self.k = k
        self.q_values = q_values
        self.z_max = z_max
        self.decoder_mode = decoder_mode
        self.tau = tau
        self.moment_method = moment_method
        self.backward_mode = backward_mode
        self.n_samples = n_samples
        self.seed = seed
        self.walk_batch_size = batch_size

        self.filtrations = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_filtrations),
        )

        feature_count = (
            n_filtrations * len(q_values) * k * z_max
        )
        self.moment_projector = nn.Sequential(
            nn.Linear(feature_count, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, n_features),
        )

        self._training_contexts = None

    def clear_training_context(self):
        """Explicitly discard an outstanding training context."""
        self._training_contexts = None

    def forward(self, x, data):
        """Return node embeddings with graph-level moment corrections."""
        if self._training_contexts is not None:
            raise RuntimeError(
                "Previous training context is pending. Call "
                "backward_surrogate() or clear_training_context()."
            )

        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError("Unexpected node-feature shape.")
        if x.shape[0] == 0 or not x.is_floating_point():
            raise ValueError("Expected nonempty floating node features.")

        edges = data.edge_index.to(device=x.device)
        if edges.dtype != torch.long:
            raise TypeError("edge_index must use torch.long.")
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if edges.numel() and (
            edges.min().item() < 0
            or edges.max().item() >= x.shape[0]
        ):
            raise ValueError("Edge vertex index is out of range.")

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                x.shape[0], dtype=torch.long, device=x.device
            )
        else:
            batch = batch.to(device=x.device)

        if batch.dtype != torch.long or batch.shape != (x.shape[0],):
            raise ValueError("batch must be a long vector per node.")

        graph_ids = torch.unique(batch, sorted=True)
        expected_ids = torch.arange(
            graph_ids.numel(), device=x.device
        )
        if not torch.equal(graph_ids, expected_ids):
            raise ValueError("Batch IDs must be consecutive from zero.")

        if edges.numel() and torch.any(
            batch[edges[0]] != batch[edges[1]]
        ).item():
            raise ValueError("Edges cannot connect different graphs.")

        # Reject oversized graphs before computing any moments.
        for graph_id in range(graph_ids.numel()):
            if (batch == graph_id).sum().item() > self.k:
                raise ValueError("Every graph must satisfy n <= k.")

        scores = self.filtrations(x)
        collect = self.training and torch.is_grad_enabled()
        contexts = []
        graph_features = []

        for graph_id in range(graph_ids.numel()):
            nodes = torch.nonzero(
                batch == graph_id, as_tuple=False
            ).flatten()

            # Convert batch-global vertex indices to local indices.
            mapping = torch.full(
                (x.shape[0],), -1,
                dtype=torch.long, device=x.device,
            )
            mapping[nodes] = torch.arange(
                nodes.numel(), device=x.device
            )
            edge_mask = batch[edges[0]] == graph_id
            local_edges = mapping[edges[:, edge_mask]]
            graph_scores = scores[nodes]
            views = []

            for view in range(self.n_filtrations):
                view_scores = graph_scores[:, view]
                context = prepare_filtration_teacher(
                    view_scores,
                    local_edges,
                    self.k,
                    self.q_values,
                    self.z_max,
                    decoder_mode=self.decoder_mode,
                    tau=self.tau,
                    moment_method=self.moment_method,
                    n_samples=self.n_samples,
                    seed=self.seed
                    + graph_id * self.n_filtrations + view,
                    batch_size=self.walk_batch_size,
                )

                # Cast BEFORE making the classifier-input leaf.
                phi = context["phi"].to(
                    device=x.device, dtype=x.dtype
                ).detach()
                if collect:
                    phi.requires_grad_(True)
                    context["phi"] = phi
                    contexts.append((view_scores, context))

                views.append(phi.flatten())

            graph_features.append(torch.cat(views))

        features = torch.stack(graph_features)
        correction = self.moment_projector(features)
        output = x + correction[batch]

        if collect:
            self._training_contexts = contexts

        return output

    def backward_surrogate(self, mode=None, retain_graph=False):
        """Inject teacher signals after classification backward.

        Call once, before optimizer.step(). Returns detached event data.
        All views/graphs are backpropagated together.
        """
        mode = self.backward_mode if mode is None else mode
        if mode not in ("M2", "M3"):
            raise ValueError("This version supports M2 and M3.")

        contexts = self._training_contexts
        if contexts is None:
            raise RuntimeError("No pending training forward.")

        # Check all views before starting the second backward pass.
        for scores, context in contexts:
            if context["phi"].grad is None:
                raise RuntimeError(
                    "Run loss.backward(retain_graph=True) first."
                )
            if not scores.requires_grad:
                raise RuntimeError("Scorer outputs require gradients.")

        score_tensors = []
        signals = []
        events = []

        for scores, context in contexts:
            gamma = -context["phi"].grad.detach()
            teacher = single_step_filtration_teacher(context, gamma)
            signal = filtration_surrogate_signal(
                scores, context, teacher, mode
            )
            score_tensors.append(scores)
            signals.append(signal)
            events.append(teacher)

        torch.autograd.backward(
            score_tensors,
            grad_tensors=signals,
            retain_graph=retain_graph,
        )
        self._training_contexts = None
        return events


class TopoGCN(torch.nn.Module):
    """Small GCN with an explicit QiHOTOGL training interface."""

    def __init__(self, backward_mode="M3"):
        super().__init__()
        self.backbone = GCNConv(1, 4)
        self.topology = QiHOTOGL(
            4, 2, 8, 8, "mean",
            k=8, q_values=(0, 1), z_max=3,
            moment_method="exact",
            backward_mode=backward_mode,
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, data):
        h = torch.tanh(self.backbone(data.x, data.edge_index))
        h = self.topology(h, data)
        return self.classifier(global_mean_pool(h, data.batch))


def run_training_demo():
    """Compare surrogate modes on six-node synthetic graphs."""
    import csv
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from torch_geometric.data import Batch

    torch.set_num_threads(1)
    dataset_seed = 1729
    model_seed = 42
    epochs = 15
    learning_rate = 0.01
    generator = torch.Generator().manual_seed(dataset_seed)

    def make_graph(label):
        # Both classes have six vertices, six edges and degree two.
        # Class 0: one six-cycle. Class 1: two disjoint triangles.
        if label == 0:
            pairs = [(i, (i + 1) % 6) for i in range(6)]
        else:
            pairs = [
                (0, 1), (1, 2), (2, 0),
                (3, 4), (4, 5), (5, 3),
            ]
        permutation = torch.randperm(6, generator=generator)
        directed = [
            (int(permutation[a]), int(permutation[b]))
            for u, v in pairs for a, b in ((u, v), (v, u))
        ]
        return Data(
            x=torch.rand((6, 1), generator=generator),
            edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
            y=torch.tensor([label], dtype=torch.long),
        )

    train = Batch.from_data_list([make_graph(i % 2) for i in range(12)])
    test = Batch.from_data_list([make_graph(i % 2) for i in range(8)])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path("runs") / ("qihotogl_small_" + stamp)
    directory.mkdir(parents=True, exist_ok=False)

    config = {
        "dataset_seed": dataset_seed,
        "model_seed": model_seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_graphs": 12,
        "test_graphs": 8,
        "moment_method": "exact",
        "k": 8,
        "q_values": [0, 1],
        "z_max": 3,
        "views": 2,
        "control": "Scorer weights fixed; shared backbone still learns.",
        "task": "Six-cycle versus two disjoint triangles.",
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    torch.save({"train": train, "test": test}, directory / "dataset.pt")

    def evaluate(model, batch):
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, batch.y).item()
            accuracy = (logits.argmax(1) == batch.y).float().mean().item()
        return loss, accuracy

    fields = [
        "mode", "epoch", "train_loss", "train_accuracy",
        "test_loss", "test_accuracy", "selected_moves",
        "scorer_grad_norm", "changed_views", "changed_vertices",
    ]

    with (directory / "metrics.csv").open("w", newline="") as handle, (
        directory / "filtrations.jsonl"
    ).open("w") as filtration_log:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for mode in ("no_surrogate", "M2", "M3"):
            # Identical model initialization in every arm.
            torch.manual_seed(model_seed)
            model = TopoGCN("M3" if mode == "no_surrogate" else mode)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            previous = None
            initial_scorer = [
                p.detach().clone() for p in model.topology.filtrations.parameters()
            ]

            train_loss, train_accuracy = evaluate(model, train)
            test_loss, test_accuracy = evaluate(model, test)
            writer.writerow(dict(
                mode=mode, epoch=0,
                train_loss=train_loss, train_accuracy=train_accuracy,
                test_loss=test_loss, test_accuracy=test_accuracy,
                selected_moves="", scorer_grad_norm="",
                changed_views="", changed_vertices="",
            ))

            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(train)

                contexts = model.topology._training_contexts
                current = torch.stack([
                    context["t"].detach().cpu().clone()
                    for _, context in contexts
                ])
                if previous is None:
                    changed_views = changed_vertices = 0
                else:
                    difference = current != previous
                    changed_views = int(difference.any(dim=1).sum())
                    changed_vertices = int(difference.sum())
                previous = current.clone()

                loss = nn.functional.cross_entropy(logits, train.y)
                loss.backward(retain_graph=(mode != "no_surrogate"))

                if mode == "no_surrogate":
                    model.topology.clear_training_context()
                    events = []
                else:
                    events = model.topology.backward_surrogate()

                selected = sum(event["selected"] is not None for event in events)
                grad_squared = sum(
                    float(p.grad.detach().square().sum())
                    for p in model.topology.filtrations.parameters()
                    if p.grad is not None
                )
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("Nonfinite gradient.")

                optimizer.step()

                if mode == "no_surrogate":
                    assert all(
                        torch.equal(before, after.detach())
                        for before, after in zip(
                            initial_scorer,
                            model.topology.filtrations.parameters(),
                        )
                    )

                # Metrics are evaluated after the optimizer update.
                train_loss, train_accuracy = evaluate(model, train)
                test_loss, test_accuracy = evaluate(model, test)
                writer.writerow(dict(
                    mode=mode, epoch=epoch,
                    train_loss=train_loss, train_accuracy=train_accuracy,
                    test_loss=test_loss, test_accuracy=test_accuracy,
                    selected_moves=selected,
                    scorer_grad_norm=grad_squared ** 0.5,
                    changed_views=changed_views,
                    changed_vertices=changed_vertices,
                ))
                filtration_log.write(json.dumps({
                    "mode": mode,
                    "epoch": epoch,
                    "timing": "training forward before optimizer update",
                    "t": current.tolist(),
                    "selected": [event["selected"] for event in events],
                }) + "\n")
                handle.flush()
                filtration_log.flush()

                print(
                    f"{mode:12s} epoch={epoch:02d} "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_accuracy:.3f} "
                    f"test_acc={test_accuracy:.3f} "
                    f"moves={selected:2d} changed_views={changed_views:2d}",
                    flush=True,
                )

            torch.save(model.state_dict(), directory / (mode + ".pt"))

    print("Results saved:", directory)


if __name__ == "__main__":
    run_training_demo()
