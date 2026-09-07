"""Differentiable 3-D vector FDFD on a Yee grid (E-field formulation).

Physics. Time-harmonic Maxwell with the e^{+j omega t} convention (the one
the 2-D H-plane solver and its referee use: outgoing waves go as
e^{-j beta z}, lossy media have eps_r = eps' - j eps''):

    curl(curl E) - k0^2 eps_r E = -j omega mu0 J,      mu_r = 1,
    H = -curl E / (j omega mu0).

Discretisation. Standard Yee staggering on a NONUNIFORM tensor grid with
steps ``dx (nx,)``, ``dy (ny,)``, ``dz (nz,)``: E components on the primal
edges, H components on the primal faces (= dual edges). The operator is
built literally as the product of the two discrete curls,

    A = Ch diag(1) Ce - k0^2 diag(eps_edge),

where ``Ce`` (edges -> faces) divides by the PRIMAL step of the edge it
differences across and ``Ch`` (faces -> edges) divides by the DUAL step
(half-cell to half-cell). Both curl matrices have a static integer COO
pattern (NumPy, built once in ``build``) and traced values; the product is
evaluated entry by entry from a precomputed list of (Ch entry, Ce entry)
pairs, so the pattern of ``A`` is static and its values are a JAX function
of the steps, the frequency and the permittivity. Interior rows of the
product are the usual 13-point curl-curl stencil.

The relative permittivity is given per CELL, ``eps_r (nx, ny, nz)``
complex, and averaged arithmetically onto each E edge from the (up to four)
cells that share it. Loss enters as the imaginary part; there is no
dispersion model -- ``eps_r`` is whatever it is at the frequency solved.

Boundaries. The outer boundary is PEC (tangential E fixed to zero, the same
identity-row / dropped-column trick as ``hplane.assemble``). On any of the
six faces a PML of ``n`` cells can be switched on: complex coordinate
stretching ``s_w = kappa + sigma_w / (j omega eps0)``, polynomial grading
of order ``pml_order`` from the interface into the wall (``sigma_max`` from
the nominal normal-incidence reflection ``pml_r0`` in vacuum), applied to
the primal steps at the cell centres and to the dual steps at the nodes.
Interior PEC objects are a boolean mask per E-edge set (static NumPy).

Shape parameters. ``dx``, ``dy``, ``dz`` may be traced JAX arrays, exactly
like the body-fitted transverse stretch of :mod:`rfx.fdfd.hplane`: the
grid topology is fixed, the node positions move, so a derivative with
respect to a step length is a smooth shape derivative. Everything on the
differentiable path is ``jax.numpy`` complex128; the sparse solve is
:func:`rfx.fdfd.linear_solve.sparse_solve` (forward AND reverse mode).

Rectangular-guide TE10 helpers. For validation (and reuse by the port /
conductor work) the module ships the standard FDFD "normalisation run"
port: a guide along z with PEC walls in x and y and PML at both z ends, an
electric current sheet ``J_y(x) = sin(pi x / a)`` on one transverse plane,
the modal amplitude read off by projecting ``E_y`` on the mode profile at a
reference plane, and S11 / S21 of an obstacle as ratios against the
amplitudes of an EMPTY-guide calibration solve on the same grid.

Scope fence. Outer boundary PEC (optionally PML-backed); mu_r = 1; one
frequency at a time with ``eps_r`` given at that frequency (no dispersion
fitting); no symmetry planes. Lumped ports and surface-impedance
conductors live in :mod:`rfx.fdfd.ports3d` and :mod:`rfx.fdfd.conductor`
and enter through the :class:`BoundaryTerms` hooks of :func:`assemble`.
S-parameter magnitudes are the validated quantities; phases are
referenced to the reference planes and rotate with them. The
``pml_r0`` default was tuned on the WR90 gate of ``tests/test_fdfd_yee3d.py``
(|Gamma_PML| ~ 2e-5 with 10 cells, 4e-6 with 16, at 10 GHz); other guides may
want a different value.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rfx.fdfd.linear_solve import sparse_matvec, sparse_solve

__all__ = [
    "Yee3DSpec", "Yee3DModel", "BoundaryTerms", "merge_terms",
    "build", "assemble", "solve", "h_from_e", "curl_h",
    "edge_shapes", "face_shapes", "split_edges", "flatten_edges",
    "pec_edges_from_cells", "positions",
    "te10_current", "te10_amplitude", "te10_discrete_beta", "te10_s_params",
    "wave_split", "C0", "MU0", "EPS0", "ETA0",
]

C0 = 299792458.0
MU0 = 4.0e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0 * C0)
ETA0 = MU0 * C0


@dataclass(frozen=True)
class Yee3DSpec:
    """Grid topology and PML layout (everything static).

    ``pml`` is the number of PML cells on (x_lo, x_hi, y_lo, y_hi, z_lo,
    z_hi); zero means a bare PEC wall on that face. ``pml_r0`` is the
    nominal normal-incidence reflection of the graded profile in vacuum,
    ``pml_kappa_max >= 1`` the real stretch at the outer wall.
    """
    nx: int
    ny: int
    nz: int
    pml: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    pml_order: int = 3
    pml_kappa_max: float = 1.0
    pml_r0: float = 1e-8


@dataclass(frozen=True)
class Yee3DModel:
    """Static part: index layout, wall mask, curl patterns, product map."""
    spec: Yee3DSpec
    n_edges: int
    n_faces: int
    edge_offsets: tuple[int, int, int]
    face_offsets: tuple[int, int, int]
    wall: np.ndarray            # (n_edges,) bool: tangential edges on the outer PEC box
    ce_rows: np.ndarray         # Ce pattern (faces x edges), value = ce_sign / primal_step
    ce_cols: np.ndarray
    ce_sign: np.ndarray
    ce_axis: np.ndarray         # 0, 1, 2 -> which primal step array
    ce_idx: np.ndarray          # index into that step array
    ch_rows: np.ndarray         # Ch pattern (edges x faces), value = ch_sign / dual_step
    ch_cols: np.ndarray
    ch_sign: np.ndarray
    ch_axis: np.ndarray
    ch_idx: np.ndarray
    rows: np.ndarray            # A pattern (edges x edges)
    cols: np.ndarray
    prod_a: np.ndarray          # A[prod_a] += Ch[prod_ch] * Ce[prod_ce]
    prod_ch: np.ndarray
    prod_ce: np.ndarray
    diag_a: np.ndarray          # (n_edges,) A entry of each diagonal element

    @property
    def n_unknowns(self) -> int:
        return self.n_edges

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.spec.nx, self.spec.ny, self.spec.nz)


@dataclass(frozen=True)
class BoundaryTerms:
    """Additive hooks for boundary models built on top of the core operator
    (lumped ports, surface-impedance conductors). All optional:

    ``pec`` -- bool ``(n_edges,)``, edges added to the PEC set;
    ``free`` -- bool ``(n_edges,)``, edges of the OUTER wall released from
    the PEC set (a lossy wall); an edge in ``pec`` (either kind) stays PEC;
    ``ch_scale`` -- ``(len(ch_rows),)`` multiplier on the dual-curl entries
    (cut-cell Ampere loops, dead H faces inside a conductor);
    ``diag_add`` -- ``(n_edges,)`` added to the diagonal of ``A`` (an
    admittance ``j omega mu0 sigma_eff``; discarded on PEC edges).
    """
    pec: np.ndarray | None = None
    free: np.ndarray | None = None
    ch_scale: jax.Array | None = None
    diag_add: jax.Array | None = None


def merge_terms(*terms) -> BoundaryTerms | None:
    """Combine several :class:`BoundaryTerms` (``None`` entries ignored):
    masks are OR-ed, ``ch_scale`` multiplied, ``diag_add`` summed."""
    ts = [t for t in terms if t is not None]
    if not ts:
        return None

    def _or(key):
        parts = [getattr(t, key) for t in ts if getattr(t, key) is not None]
        return np.logical_or.reduce(parts) if parts else None

    scales = [t.ch_scale for t in ts if t.ch_scale is not None]
    diags = [t.diag_add for t in ts if t.diag_add is not None]
    ch_scale = None
    for s in scales:
        ch_scale = s if ch_scale is None else ch_scale * s
    diag_add = None
    for d in diags:
        diag_add = d if diag_add is None else diag_add + d
    return BoundaryTerms(pec=_or("pec"), free=_or("free"), ch_scale=ch_scale, diag_add=diag_add)


def edge_shapes(nx: int, ny: int, nz: int):
    """Array shapes of (Ex, Ey, Ez): a component is cell-count long along
    its own axis and node-count long along the other two."""
    return ((nx, ny + 1, nz + 1), (nx + 1, ny, nz + 1), (nx + 1, ny + 1, nz))


def face_shapes(nx: int, ny: int, nz: int):
    """Array shapes of (Hx, Hy, Hz): node-count along the own axis, cell-count
    along the other two."""
    return ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1))


def _offsets(shapes):
    sizes = [int(np.prod(s)) for s in shapes]
    return (0, sizes[0], sizes[0] + sizes[1]), sum(sizes)


def _index_arrays(shapes, offsets):
    return [np.arange(int(np.prod(s))).reshape(s) + off for s, off in zip(shapes, offsets)]


def _shift(idx: np.ndarray, axis: int, lo_off: int, hi_off: int) -> np.ndarray:
    """``idx`` restricted along ``axis`` to the slice ``[lo_off, n - hi_off)``."""
    sl = [slice(None)] * 3
    sl[axis] = slice(lo_off, idx.shape[axis] - hi_off if hi_off else None)
    return idx[tuple(sl)]


def _axis_index(shape, axis: int) -> np.ndarray:
    """Broadcast array of the coordinate index along ``axis`` for ``shape``."""
    n = shape[axis]
    view = [1, 1, 1]
    view[axis] = n
    return np.broadcast_to(np.arange(n).reshape(view), shape)


def _curl_patterns(nx: int, ny: int, nz: int):
    """Static COO patterns of the primal curl Ce (edges -> faces) and the dual
    curl Ch (faces -> edges) with the axis / index of the step each entry
    divides by. Component c differences component b = c+2 along a = c+1 and
    component a along b (cyclic)."""
    e_shapes, f_shapes = edge_shapes(nx, ny, nz), face_shapes(nx, ny, nz)
    e_off, n_edges = _offsets(e_shapes)
    f_off, n_faces = _offsets(f_shapes)
    e_idx = _index_arrays(e_shapes, e_off)
    f_idx = _index_arrays(f_shapes, f_off)

    ce: list[tuple[np.ndarray, np.ndarray, int, int, np.ndarray]] = []
    ch: list[tuple[np.ndarray, np.ndarray, int, int, np.ndarray]] = []
    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        # Ce: (curl E)_c = d_a E_b - d_b E_a, forward difference from the face
        face = f_idx[c]
        for sign, comp, axis in ((+1, b, a), (-1, a, b)):
            step_i = _axis_index(face.shape, axis)
            ce.append((face, _shift(e_idx[comp], axis, 1, 0), +sign, axis, step_i))
            ce.append((face, _shift(e_idx[comp], axis, 0, 1), -sign, axis, step_i))
        # Ch: (curl H)_c = d_a H_b - d_b H_a, backward difference to the edge
        edge = e_idx[c]
        for sign, comp, axis in ((+1, b, a), (-1, a, b)):
            step_i = _axis_index(edge.shape, axis)
            rows_hi = _shift(edge, axis, 0, 1)          # H[idx] exists for idx_axis <= n-1
            ch.append((rows_hi, f_idx[comp], +sign, axis, _shift(step_i, axis, 0, 1)))
            rows_lo = _shift(edge, axis, 1, 0)          # H[idx - e_axis] exists for idx_axis >= 1
            ch.append((rows_lo, f_idx[comp], -sign, axis, _shift(step_i, axis, 1, 0)))

    def pack(entries):
        r = np.concatenate([e[0].ravel() for e in entries])
        cc = np.concatenate([e[1].ravel() for e in entries])
        s = np.concatenate([np.full(e[0].size, e[2], dtype=np.int8) for e in entries])
        ax = np.concatenate([np.full(e[0].size, e[3], dtype=np.int8) for e in entries])
        ix = np.concatenate([e[4].ravel() for e in entries])
        return r, cc, s, ax, ix

    return (e_shapes, e_off, n_edges, f_shapes, f_off, n_faces, pack(ce), pack(ch))


def build(spec: Yee3DSpec) -> Yee3DModel:
    """Static model: patterns of Ce, Ch and A = Ch Ce - k0^2 eps, the outer
    PEC wall mask and the product map. Geometry (interior PEC, eps, steps)
    is NOT part of the model, so one model serves the calibration and the
    obstacle solves of a port normalisation run."""
    nx, ny, nz = int(spec.nx), int(spec.ny), int(spec.nz)
    if min(nx, ny, nz) < 2:
        raise ValueError("need at least 2 cells per axis")
    if len(spec.pml) != 6 or any(p < 0 for p in spec.pml):
        raise ValueError("pml must be six non-negative cell counts")
    for ax, n in enumerate((nx, ny, nz)):
        if spec.pml[2 * ax] + spec.pml[2 * ax + 1] >= n:
            raise ValueError(f"PML on axis {ax} leaves no interior cells")
    if spec.pml_kappa_max < 1.0:
        raise ValueError("pml_kappa_max must be >= 1")

    (e_shapes, e_off, n_edges, _f_shapes, f_off, n_faces,
     (ce_r, ce_c, ce_s, ce_ax, ce_ix), (ch_r, ch_c, ch_s, ch_ax, ch_ix)) = _curl_patterns(nx, ny, nz)

    # outer PEC box: tangential edges on the six faces
    wall_parts = []
    for c, shp in enumerate(e_shapes):
        w = np.zeros(shp, dtype=bool)
        for axis in range(3):
            if axis == c:
                continue
            sl: list = [slice(None)] * 3
            sl[axis] = 0
            w[tuple(sl)] = True
            sl[axis] = shp[axis] - 1
            w[tuple(sl)] = True
        wall_parts.append(w.ravel())
    wall = np.concatenate(wall_parts)

    # product A = Ch Ce: for every Ch entry m (row e, col f) pair it with all
    # Ce entries n in row f; A[e, ce_c[n]] += Ch[m] Ce[n].
    order = np.argsort(ce_r, kind="stable")
    ce_r_sorted = ce_r[order]
    faces = np.arange(n_faces)
    starts = np.searchsorted(ce_r_sorted, faces, side="left")
    counts = np.searchsorted(ce_r_sorted, faces, side="right") - starts
    cnt_m = counts[ch_c]
    m_rep = np.repeat(np.arange(len(ch_r)), cnt_m)
    offs = np.arange(int(cnt_m.sum())) - np.repeat(np.cumsum(cnt_m) - cnt_m, cnt_m)
    n_idx = order[starts[ch_c[m_rep]] + offs]
    a_rows = np.concatenate([ch_r[m_rep], np.arange(n_edges)])
    a_cols = np.concatenate([ce_c[n_idx], np.arange(n_edges)])
    key = a_rows.astype(np.int64) * n_edges + a_cols
    uniq, inv = np.unique(key, return_inverse=True)
    rows = (uniq // n_edges).astype(np.int64)
    cols = (uniq % n_edges).astype(np.int64)
    n_prod = len(m_rep)
    return Yee3DModel(
        spec=spec, n_edges=n_edges, n_faces=n_faces,
        edge_offsets=e_off, face_offsets=f_off, wall=wall,
        ce_rows=ce_r, ce_cols=ce_c, ce_sign=ce_s, ce_axis=ce_ax, ce_idx=ce_ix,
        ch_rows=ch_r, ch_cols=ch_c, ch_sign=ch_s, ch_axis=ch_ax, ch_idx=ch_ix,
        rows=rows, cols=cols, prod_a=inv[:n_prod], prod_ch=m_rep, prod_ce=n_idx,
        diag_a=inv[n_prod:])


# ----------------------------------------------------------------------------
# field layout helpers

def split_edges(model: Yee3DModel, vec: jax.Array):
    """Flat edge vector ``(n_edges,)`` or ``(n_edges, m)`` -> (Ex, Ey, Ez)
    with the shapes of :func:`edge_shapes` (a trailing ``m`` axis if given)."""
    shapes = edge_shapes(*model.shape)
    out = []
    for shp, off in zip(shapes, model.edge_offsets):
        size = int(np.prod(shp))
        out.append(vec[off:off + size].reshape(shp + tuple(vec.shape[1:])))
    return tuple(out)


def split_faces(model: Yee3DModel, vec: jax.Array):
    """Flat face vector -> (Hx, Hy, Hz) with the shapes of :func:`face_shapes`."""
    shapes = face_shapes(*model.shape)
    out = []
    for shp, off in zip(shapes, model.face_offsets):
        size = int(np.prod(shp))
        out.append(vec[off:off + size].reshape(shp + tuple(vec.shape[1:])))
    return tuple(out)


def flatten_edges(model: Yee3DModel, ex, ey, ez) -> jax.Array:
    """(Ex, Ey, Ez) -> flat ``(n_edges,)`` complex128 vector."""
    shapes = edge_shapes(*model.shape)
    parts = []
    for arr, shp in zip((ex, ey, ez), shapes):
        arr = jnp.asarray(arr, dtype=jnp.complex128)
        if arr.shape != shp:
            raise ValueError(f"edge array has shape {arr.shape}, expected {shp}")
        parts.append(arr.ravel())
    return jnp.concatenate(parts)


def pec_edges_from_cells(spec: Yee3DSpec, cells: np.ndarray):
    """Boolean E-edge masks (Ex, Ey, Ez) of the edges touching any metal
    cell of the ``(nx, ny, nz)`` cell mask ``cells`` (the node-Dirichlet
    staircase: an edge is PEC when one of the cells around it is metal)."""
    cells = np.asarray(cells, dtype=bool)
    if cells.shape != (spec.nx, spec.ny, spec.nz):
        raise ValueError(f"cells must have shape {(spec.nx, spec.ny, spec.nz)}")
    out = []
    for c in range(3):
        # an edge along c has a cell index along c and node indices (j, k)
        # on the other axes; it touches the cells (j-1, j) x (k-1, k)
        a, b = (c + 1) % 3, (c + 2) % 3
        pad = [(0, 0)] * 3
        pad[a] = pad[b] = (1, 1)
        p = np.pad(cells, pad, mode="constant", constant_values=False)
        acc = np.zeros(edge_shapes(spec.nx, spec.ny, spec.nz)[c], dtype=bool)
        for da in (0, 1):
            for db in (0, 1):
                sl = [slice(None)] * 3
                sl[a] = slice(da, p.shape[a] - 1 + da)
                sl[b] = slice(db, p.shape[b] - 1 + db)
                acc |= p[tuple(sl)]
        out.append(acc)
    shapes = edge_shapes(spec.nx, spec.ny, spec.nz)
    for arr, shp in zip(out, shapes):
        if arr.shape != shp:
            raise AssertionError(f"mask bookkeeping: {arr.shape} vs {shp}")
    return tuple(out)


def positions(d) -> jax.Array:
    """Node coordinates ``(n + 1,)`` from the steps ``(n,)``, first node at 0."""
    d = jnp.asarray(d, dtype=jnp.float64)
    return jnp.concatenate([jnp.zeros((1,), jnp.float64), jnp.cumsum(d)])


# ----------------------------------------------------------------------------
# metric: PML-stretched primal and dual steps

def _stretch(spec: Yee3DSpec, axis: int, pos: jax.Array, x_nodes: jax.Array, omega) -> jax.Array:
    """Complex stretch ``s_w`` evaluated at the coordinates ``pos``."""
    n_lo, n_hi = spec.pml[2 * axis], spec.pml[2 * axis + 1]
    m = spec.pml_order
    s = jnp.ones(pos.shape, dtype=jnp.complex128)
    length = x_nodes[-1]
    for side, n in ((0, n_lo), (1, n_hi)):
        if n == 0:
            continue
        if side == 0:
            depth = x_nodes[n]
            rho = (depth - pos) / depth
        else:
            depth = length - x_nodes[-1 - n]
            rho = (pos - (length - depth)) / depth
        rho = jnp.clip(rho, 0.0, 1.0)
        profile = rho ** m
        sigma_max = -(m + 1) * np.log(spec.pml_r0) / (2.0 * ETA0 * depth)
        s = s + (spec.pml_kappa_max - 1.0) * profile + sigma_max * profile / (1j * omega * EPS0)
    return s


def _steps(model: Yee3DModel, freq, dx, dy, dz):
    """Stretched primal steps (cell centres) and dual steps (nodes; half
    cells at the two walls) per axis, complex128."""
    omega = 2.0 * jnp.pi * jnp.asarray(freq, dtype=jnp.float64)
    primal, dual = [], []
    for axis, d in enumerate((dx, dy, dz)):
        d = jnp.asarray(d, dtype=jnp.float64)
        if d.shape != (model.shape[axis],):
            raise ValueError(f"step array on axis {axis} must have shape {(model.shape[axis],)}")
        x = positions(d)
        centres = x[:-1] + 0.5 * d
        half = jnp.concatenate([0.5 * d[:1], 0.5 * (d[:-1] + d[1:]), 0.5 * d[-1:]])
        primal.append(_stretch(model.spec, axis, centres, x, omega) * d)
        dual.append(_stretch(model.spec, axis, x, x, omega) * half)
    return primal, dual, omega


def _curl_values(model: Yee3DModel, freq, dx, dy, dz):
    primal, dual, omega = _steps(model, freq, dx, dy, dz)
    ce = jnp.zeros(len(model.ce_rows), jnp.complex128)
    ch = jnp.zeros(len(model.ch_rows), jnp.complex128)
    for axis in range(3):
        sel = model.ce_axis == axis
        ce = ce.at[sel].set(model.ce_sign[sel] / primal[axis][model.ce_idx[sel]])
        sel = model.ch_axis == axis
        ch = ch.at[sel].set(model.ch_sign[sel] / dual[axis][model.ch_idx[sel]])
    return ce, ch, omega


def _eps_on_edges(model: Yee3DModel, eps_r) -> jax.Array:
    """Cell permittivity ``(nx, ny, nz)`` -> flat edge vector by arithmetic
    averaging over the cells sharing each edge (a wall edge is PEC, so the
    replicated value used there is never seen by the solve)."""
    nx, ny, nz = model.shape
    if eps_r is None:
        return jnp.ones(model.n_edges, jnp.complex128)
    eps_r = jnp.asarray(eps_r, dtype=jnp.complex128)
    if eps_r.shape != (nx, ny, nz):
        raise ValueError(f"eps_r must have shape {(nx, ny, nz)}, got {eps_r.shape}")
    parts = []
    for c in range(3):
        pad = [(0, 0)] * 3
        for axis in range(3):
            if axis != c:
                pad[axis] = (1, 1)
        p = jnp.pad(eps_r, pad, mode="edge")
        a, b = (c + 1) % 3, (c + 2) % 3
        acc = jnp.zeros(edge_shapes(nx, ny, nz)[c], jnp.complex128)
        for da in (0, 1):
            for db in (0, 1):
                sl = [slice(None)] * 3
                sl[a] = slice(da, p.shape[a] - 1 + da)
                sl[b] = slice(db, p.shape[b] - 1 + db)
                acc = acc + p[tuple(sl)]
        parts.append((0.25 * acc).ravel())
    return jnp.concatenate(parts)


def _pec_mask(model: Yee3DModel, pec, terms: BoundaryTerms | None = None) -> np.ndarray:
    """PEC edge set: ``(wall & ~terms.free) | pec | terms.pec``."""
    mask = model.wall.copy()
    if terms is not None and terms.free is not None:
        free = np.asarray(terms.free, dtype=bool)
        if free.shape != (model.n_edges,):
            raise ValueError(f"terms.free must have shape {(model.n_edges,)}")
        mask &= ~free
    if pec is not None:
        shapes = edge_shapes(*model.shape)
        if len(pec) != 3:
            raise ValueError("pec must be a triple of boolean edge masks (Ex, Ey, Ez)")
        for arr, shp in zip(pec, shapes):
            arr = np.asarray(arr, dtype=bool)
            if arr.shape != shp:
                raise ValueError(f"pec mask has shape {arr.shape}, expected {shp}")
        mask |= np.concatenate([np.asarray(a, dtype=bool).ravel() for a in pec])
    if terms is not None and terms.pec is not None:
        extra = np.asarray(terms.pec, dtype=bool)
        if extra.shape != (model.n_edges,):
            raise ValueError(f"terms.pec must have shape {(model.n_edges,)}")
        mask |= extra
    return mask


def _rhs(model: Yee3DModel, sources, omega, mask: np.ndarray) -> jax.Array:
    """``-j omega mu0 J`` per edge; ``sources`` is one (Jx, Jy, Jz) triple or
    a sequence of them (then the rhs is ``(n_edges, m)``)."""
    if len(sources) == 3 and not (isinstance(sources[0], (tuple, list))):
        cols = [flatten_edges(model, *sources)]
        block = False
    else:
        cols = [flatten_edges(model, *s) for s in sources]
        block = True
    j = jnp.stack(cols, axis=1) if block else cols[0]
    rhs = -1j * omega * MU0 * j
    keep = jnp.asarray(~mask)
    return jnp.where(keep[:, None] if block else keep, rhs, 0.0)


def _apply_ch_scale(model: Yee3DModel, ch: jax.Array, terms: BoundaryTerms | None) -> jax.Array:
    if terms is None or terms.ch_scale is None:
        return ch
    scale = jnp.asarray(terms.ch_scale)
    if scale.shape != (len(model.ch_rows),):
        raise ValueError(f"terms.ch_scale must have shape {(len(model.ch_rows),)}")
    return ch * scale


def assemble(model: Yee3DModel, freq, eps_r, dx, dy, dz, sources, pec=None,
             terms: BoundaryTerms | None = None):
    """(data, rhs) for the pattern ``model.rows/cols``.

    ``eps_r`` -- cell permittivity ``(nx, ny, nz)`` complex or ``None`` (vacuum);
    ``dx, dy, dz`` -- step arrays (traced OK); ``sources`` -- (Jx, Jy, Jz)
    edge arrays of electric current density, or a sequence of triples for a
    block right-hand side; ``pec`` -- optional (Ex, Ey, Ez) boolean masks of
    interior PEC edges (static), added to the outer walls; ``terms`` --
    optional :class:`BoundaryTerms` hooks (ports, surface impedance).
    """
    ce, ch, omega = _curl_values(model, freq, dx, dy, dz)
    ch = _apply_ch_scale(model, ch, terms)
    k0 = omega / C0
    eps_e = _eps_on_edges(model, eps_r)
    data = jnp.zeros(len(model.rows), jnp.complex128)
    data = data.at[model.prod_a].add(ch[model.prod_ch] * ce[model.prod_ce])
    data = data.at[model.diag_a].add(-(k0 * k0) * eps_e)
    if terms is not None and terms.diag_add is not None:
        extra = jnp.asarray(terms.diag_add, dtype=jnp.complex128)
        if extra.shape != (model.n_edges,):
            raise ValueError(f"terms.diag_add must have shape {(model.n_edges,)}")
        data = data.at[model.diag_a].add(extra)
    mask = _pec_mask(model, pec, terms)
    data = jnp.where(mask[model.rows] | mask[model.cols], 0.0, data)
    data = jnp.where((model.rows == model.cols) & mask[model.rows], 1.0, data)
    return data, _rhs(model, sources, omega, mask)


def solve(model: Yee3DModel, freq, eps_r, dx, dy, dz, sources, pec=None,
          terms: BoundaryTerms | None = None):
    """E field for one frequency: ``(Ex, Ey, Ez)`` on the edge shapes, with a
    trailing axis when ``sources`` is a sequence of triples (one LU for all
    of them). Differentiable in ``freq``, ``eps_r``, the steps, the sources
    and the traced parts of ``terms``."""
    data, rhs = assemble(model, freq, eps_r, dx, dy, dz, sources, pec, terms)
    x = sparse_solve(data, model.rows, model.cols, rhs)
    return split_edges(model, x)


def curl_h(model: Yee3DModel, freq, h, dx, dy, dz, terms: BoundaryTerms | None = None):
    """``Ch H`` on the edges: the discrete Ampere loop of ``H`` around every
    edge divided by the edge's dual area (with the same PML stretch and the
    same ``terms.ch_scale`` as the operator). ``h`` is the (Hx, Hy, Hz)
    triple of :func:`h_from_e`; returns a flat ``(n_edges,)`` or
    ``(n_edges, m)`` vector."""
    _, ch, _ = _curl_values(model, freq, dx, dy, dz)
    ch = _apply_ch_scale(model, ch, terms)
    hx, hy, hz = h
    if hx.ndim == 4:
        vec = jnp.concatenate([jnp.asarray(a, jnp.complex128).reshape(-1, a.shape[-1])
                               for a in (hx, hy, hz)])
    else:
        vec = jnp.concatenate([jnp.asarray(a, jnp.complex128).ravel() for a in (hx, hy, hz)])
    return sparse_matvec(ch, model.ch_rows, model.ch_cols, vec)


def h_from_e(model: Yee3DModel, freq, e, dx, dy, dz):
    """``H = -curl E / (j omega mu0)`` on the faces (e^{+j omega t}), using the
    same PML-stretched primal curl as the operator. ``e`` is the (Ex, Ey, Ez)
    triple returned by :func:`solve` (a trailing block axis is allowed)."""
    ce, _, omega = _curl_values(model, freq, dx, dy, dz)
    ex, ey, ez = e
    if ex.ndim == 4:
        vec = jnp.concatenate([jnp.asarray(a, jnp.complex128).reshape(-1, a.shape[-1])
                               for a in (ex, ey, ez)])
    else:
        vec = flatten_edges(model, ex, ey, ez)
    curl = sparse_matvec(ce, model.ce_rows, model.ce_cols, vec)
    return split_faces(model, curl / (-1j * omega * MU0))


# ----------------------------------------------------------------------------
# rectangular-guide TE10 normalisation-run port (guide along z, walls in x, y)

def _te10_profile(dx):
    """``sin(pi x / a)`` at the x nodes and the (unstretched) dual-x weights."""
    d = jnp.asarray(dx, dtype=jnp.float64)
    x = positions(d)
    a = x[-1]
    phi = jnp.sin(jnp.pi * x / a)
    w = jnp.concatenate([0.5 * d[:1], 0.5 * (d[:-1] + d[1:]), 0.5 * d[-1:]])
    return phi, w


def te10_current(model: Yee3DModel, dx, k_plane: int):
    """Electric current sheet ``J_y = sin(pi x / a)`` on the transverse
    z-node plane ``k_plane`` (all y cells), as a (Jx, Jy, Jz) triple."""
    nx, ny, nz = model.shape
    phi, _ = _te10_profile(dx)
    shapes = edge_shapes(nx, ny, nz)
    jx = jnp.zeros(shapes[0], jnp.complex128)
    jy = jnp.zeros(shapes[1], jnp.complex128)
    jy = jy.at[:, :, k_plane].set(jnp.broadcast_to(phi[:, None].astype(jnp.complex128), (nx + 1, ny)))
    jz = jnp.zeros(shapes[2], jnp.complex128)
    return jx, jy, jz


def te10_amplitude(ey, dx, k_plane: int):
    """Modal amplitude of ``E_y`` on the z-node plane ``k_plane``: the
    y-average of ``E_y`` projected on ``sin(pi x / a)`` with trapezoid
    weights (exactly orthogonal to the other TE_n0 modes on a uniform grid,
    so evanescent iris modes do not leak into it)."""
    phi, w = _te10_profile(dx)
    prof = jnp.mean(ey[:, :, k_plane], axis=1)
    return jnp.sum(w * phi * prof) / jnp.sum(w * phi * phi)


def te10_discrete_beta(freq, a: float, hx: float, hz: float):
    """Propagation constant of the DISCRETE TE10 mode of the uniform Yee grid
    (steps ``hx``, ``hz``): ``(2/hx sin(pi hx / 2a))^2 + (2/hz sin(beta hz/2))^2 = k0^2``.
    Real above cutoff; the grid's numerical dispersion is included exactly,
    which the wave-split gate needs."""
    k0 = 2.0 * np.pi * freq / C0
    lam = (2.0 / hx * np.sin(np.pi * hx / (2.0 * a))) ** 2
    kt = np.sqrt(k0 * k0 - lam)
    return 2.0 / hz * np.arcsin(hz * kt / 2.0)


def wave_split(amps, f, ks):
    """Least-squares split of modal amplitudes ``amps[k] = A f^k + B f^-k``
    sampled on the z-planes ``ks`` (``f = e^{-j beta hz}``): returns ``(A, B)``,
    the forward and backward wave amplitudes referenced to k = 0."""
    ks = jnp.asarray(ks, dtype=jnp.float64)
    f = jnp.asarray(f, dtype=jnp.complex128)
    basis = jnp.stack([f ** ks, f ** (-ks)], axis=1)
    sol, *_ = jnp.linalg.lstsq(basis, jnp.asarray(amps, dtype=jnp.complex128))
    return sol[0], sol[1]


def te10_s_params(model: Yee3DModel, freq, eps_r, dx, dy, dz, k_src: int,
                  k_ref1: int, k_ref2: int, pec=None, return_fields: bool = False):
    """S11 / S21 of an obstacle (``eps_r``, ``pec``) by the normalisation
    run: an empty-guide solve on the same grid gives the incident modal
    amplitudes at the two reference planes; then

        S11 = (a1 - a1_inc) / a1_inc,     S21 = a2 / a2_inc,

    with the source at ``k_src < k_ref1 < obstacle < k_ref2``. Phases are
    referenced to the reference planes. Differentiable in everything
    (both solves are on the differentiable path)."""
    src = te10_current(model, dx, k_src)
    e_cal = solve(model, freq, None, dx, dy, dz, src, None)
    e_obs = solve(model, freq, eps_r, dx, dy, dz, src, pec)
    a1_inc = te10_amplitude(e_cal[1], dx, k_ref1)
    a2_inc = te10_amplitude(e_cal[1], dx, k_ref2)
    a1 = te10_amplitude(e_obs[1], dx, k_ref1)
    a2 = te10_amplitude(e_obs[1], dx, k_ref2)
    s11 = (a1 - a1_inc) / a1_inc
    s21 = a2 / a2_inc
    if return_fields:
        return s11, s21, e_cal, e_obs
    return s11, s21
