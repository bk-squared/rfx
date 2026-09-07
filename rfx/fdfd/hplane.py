"""Differentiable 2-D H-plane FDFD (scalar Helmholtz, TE_n0) with exact
discrete transparent ports and body-fitted, continuously movable PEC irises.

Physics and discretisation follow the independent referee
``validation/crossval/comparators/fdfd_hplane.py`` (node-Dirichlet PEC,
5-point Laplacian, exact discrete DtN port built from the discrete transverse
eigenbasis): on the nominal grid the two agree to LU roundoff, and that
agreement is the correctness gate of this module (``tests/test_fdfd_hplane.py``).
What is new:

* the assembled system is a JAX function of ``freq`` (through k and the port
  DtN), of a complex relative permittivity map ``eps_r`` per interior node,
  and of the iris aperture widths ``apertures`` [m] as CONTINUOUS parameters;
* the solve is ``rfx.fdfd.linear_solve.sparse_solve``, so S11/S21 and any
  function of them can be differentiated in forward or reverse mode with
  respect to all three (one extra transposed solve per gradient).

Continuous aperture width -- how. The transverse grid is BODY-FITTED: its
nodes are placed so that every iris edge is always ON a node, and when a
width changes the nodes between the breakpoints (wall, edges, centre)
stretch uniformly with it. The node count, the PEC mask and the sparsity
pattern are static; only the node coordinates move. The x-stencil is the
standard nonuniform three-point one,

    u_xx ~ 2/(h_- + h_+) [ (u_+ - u)/h_+ - (u - u_-)/h_- ],

which is the uniform stencil at the nominal widths, so the referee is
reproduced exactly there. The transverse eigenbasis for the DtN port is the
discrete eigenbasis of that (self-adjoint, weight (h_- + h_+)/2) operator,
obtained by ``jnp.linalg.eigh`` and therefore differentiable too; port
transparency stays exact on the stretched grid. Because nothing switches
topology as the width moves, the width derivative is smooth and converges
with the mesh -- unlike a cut-cell / Shortley-Weller edge, whose derivative
was measured to jump ~50 % at every node crossing (see the module history).

Irises whose NOMINAL apertures coincide share a breakpoint and therefore a
width parameter (``HPlaneModel.width_groups``); ``apertures`` is given per
group. Widths must keep the nominal ordering of edges (a narrower nominal
iris must stay narrower) -- that is the only topological constraint.

    (Dxx + Dzz + k^2 eps_r) E = 0,  E = 0 on PEC (walls + irises)
    ghost at z=0 :  E_{-1}   = Q E_0 + (e^{+g1 h} - e^{-g1 h}) phi1
    ghost at z=L :  E_{nz+1} = Q E_{nz}
    Q = sum_n e^{-gt_n h} phi_n phi_n^T W,   cosh(gt_n h) = 1 - h^2 (lam_n + k^2)/2
    (evaluated as a matrix function of the transverse operator, see _spectral)
    S11 = <E_0, phi1>_W - 1,   S21 = <E_nz, phi1>_W e^{+gt_1 nz h}

Scope fence (inherited from the referee, still binding): H-plane only --
inductive irises, septa, width steps. Not capacitive / E-plane / posts.
Magnitudes are the validated quantities; the S-parameter phases are
referenced to the domain ends and rotate with ``margin_cells``. One grid is
not an answer: Richardson over >= 2 integer refinements is.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rfx.fdfd.linear_solve import sparse_solve

__all__ = ["HPlaneSpec", "HPlaneModel", "build", "assemble", "solve",
           "s_params", "richardson_first_order", "C0"]

C0 = 299792458.0


@dataclass(frozen=True)
class HPlaneSpec:
    """Geometry in BASE cells (referee condition 3: integer refinement keeps
    every level grid-exact, which is what makes Richardson valid).

    ``apertures_cells`` are the NOMINAL electrical aperture widths (distance
    between the bounding metal node planes); ``solve`` may override them
    with continuous widths in metres, one per width group.
    """
    a: float                              # guide width [m]
    base_cells: int                       # cells across a at refinement 1
    refinement: int = 1
    apertures_cells: tuple[int, ...] = ()  # nominal electrical aperture widths
    cavities_cells: tuple[int, ...] = ()   # electrical cavity lengths between irises
    thickness_cells: int = 1               # electrical iris thickness
    margin_cells: int = 8                  # uniform guide before/after the structure


@dataclass(frozen=True)
class HPlaneModel:
    """Static part of the problem: grid topology, PEC mask, COO pattern."""
    spec: HPlaneSpec
    h: float                # nominal (and z) spacing
    nx: int
    nz: int
    nxi: int
    nzi: int
    metal: np.ndarray       # (nxi, nzi) bool PEC mask (static: edges sit on nodes)
    breakpoints: np.ndarray  # (K+1,) node indices 0 = wall ... nx = wall
    bp_group: np.ndarray    # (K+1,) width group of each breakpoint, -1 for walls/centre
    bp_side: np.ndarray     # (K+1,) -1 left edge, +1 right edge, 0 fixed
    width_groups: tuple[tuple[int, ...], ...]  # iris indices per width parameter
    nominal_apertures: tuple[float, ...]       # [m], one per width group
    rows: np.ndarray        # COO pattern
    cols: np.ndarray
    kind: np.ndarray        # 0 x-minus, 1 x-plus, 2 z-neighbour, 3 diag, 4 port block

    @property
    def n_unknowns(self) -> int:
        return self.nxi * self.nzi

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nxi, self.nzi)


def build(spec: HPlaneSpec) -> HPlaneModel:
    r = int(spec.refinement)
    if r < 1 or r != spec.refinement:
        raise ValueError(f"refinement must be a positive integer, got {spec.refinement}")
    nx = spec.base_cells * r
    h = spec.a / nx
    tc = spec.thickness_cells * r + 1
    apertures = tuple(spec.apertures_cells)
    cavities = tuple(spec.cavities_cells)
    span = len(apertures) * (tc - 1) + sum(c * r for c in cavities)
    nz = span + 2 * spec.margin_cells * r
    ix = np.arange(1, nx)
    nxi, nzi = len(ix), nz + 1

    metal = np.zeros((nxi, nzi), dtype=bool)
    fcs: list[int] = []
    z = spec.margin_cells * r
    for i, d_c in enumerate(apertures):
        if (nx - d_c * r) % 2 != 0:
            raise ValueError(
                f"aperture {d_c} base cells is not symmetric at refinement {r}; "
                "re-snapping would break Richardson (referee condition 3)")
        fc = (nx - d_c * r) // 2
        if fc < 1:
            raise ValueError(f"aperture {d_c} base cells leaves no metal at the wall")
        fcs.append(fc)
        metal[(ix <= fc) | (ix >= nx - fc), z:z + tc] = True
        if i < len(cavities):
            z += (tc - 1) + cavities[i] * r
    if apertures and z + tc - 1 != spec.margin_cells * r + span:
        raise ValueError(f"span mismatch: {z} vs {span}")

    # Width groups: irises with equal nominal aperture share a breakpoint.
    groups: dict[int, list[int]] = {}
    for i, fc in enumerate(fcs):
        groups.setdefault(fc, []).append(i)
    group_fcs = sorted(groups)                       # left edges, wall -> centre
    width_groups = tuple(tuple(groups[fc]) for fc in group_fcs)
    nominal = tuple(float((nx - 2 * fc) * h) for fc in group_fcs)
    bps = [0] + group_fcs + [nx - fc for fc in reversed(group_fcs)] + [nx]
    bp_group = [-1] + list(range(len(group_fcs))) + list(reversed(range(len(group_fcs)))) + [-1]
    bp_side = [0] + [-1] * len(group_fcs) + [1] * len(group_fcs) + [0]
    if len(set(bps)) != len(bps):                     # an aperture of zero width
        raise ValueError("degenerate aperture (edges coincide)")

    # COO pattern, flat index = i * nzi + j. Every node carries all four
    # neighbour entries; PEC rows become identity rows at assembly.
    idx = np.arange(nxi * nzi).reshape(nxi, nzi)
    blocks = [
        (idx[1:, :], idx[:-1, :], 0),     # x-minus neighbour
        (idx[:-1, :], idx[1:, :], 1),     # x-plus neighbour
        (idx[:, 1:], idx[:, :-1], 2),     # z-minus
        (idx[:, :-1], idx[:, 1:], 2),     # z-plus
        (idx, idx, 3),                    # diagonal
    ]
    ii, kk = np.meshgrid(np.arange(nxi), np.arange(nxi), indexing="ij")
    for j in (0, nzi - 1):                # dense DtN port blocks
        blocks.append((idx[ii, j], idx[kk, j], 4))
    rows = np.concatenate([b[0].ravel() for b in blocks])
    cols = np.concatenate([b[1].ravel() for b in blocks])
    kind = np.concatenate([np.full(b[0].size, b[2], dtype=int) for b in blocks])
    return HPlaneModel(spec=spec, h=h, nx=nx, nz=nz, nxi=nxi, nzi=nzi, metal=metal,
                       breakpoints=np.asarray(bps), bp_group=np.asarray(bp_group),
                       bp_side=np.asarray(bp_side), width_groups=width_groups,
                       nominal_apertures=nominal, rows=rows, cols=cols, kind=kind)


def _node_positions(model: HPlaneModel, apertures) -> jax.Array:
    """Body-fitted transverse node positions, (nx + 1,) including the walls."""
    a = model.spec.a
    ws = [jnp.asarray(w, dtype=jnp.float64) for w in apertures]
    bp_x: list[jax.Array | None] = []
    for g, side in zip(model.bp_group, model.bp_side):
        if side == 0:
            bp_x.append(None)
        else:
            bp_x.append(0.5 * (a + side * ws[int(g)]))
    bp_x[0] = jnp.asarray(0.0)
    bp_x[-1] = jnp.asarray(a)
    bp_pos = jnp.stack([b for b in bp_x if b is not None])
    if len(bp_pos) != len(model.breakpoints):
        raise AssertionError("breakpoint bookkeeping")
    segs = []
    bps = model.breakpoints
    for k in range(len(bps) - 1):
        n_k = int(bps[k + 1] - bps[k])
        frac = jnp.arange(n_k) / n_k
        segs.append(bp_pos[k] + frac * (bp_pos[k + 1] - bp_pos[k]))
    return jnp.concatenate(segs + [bp_pos[-1:]])


def _transverse(model: HPlaneModel, x_all: jax.Array):
    """Nonuniform x-stencil coefficients, trapezoid weights and the
    symmetrised transverse operator ``S = W^-1/2 T W^-1/2`` (so that
    ``Dxx = W^-1 T`` has the eigenbasis ``W^-1/2 psi``).

    Returns ``(c_minus, c_plus, d_x, w, S)``.
    """
    hs = jnp.diff(x_all)                       # (nx,) cell widths
    h_m, h_p = hs[:-1], hs[1:]                 # left/right spacing of interior nodes
    s = h_m + h_p
    c_minus = 2.0 / (s * h_m)
    c_plus = 2.0 / (s * h_p)
    d_x = -2.0 / (h_m * h_p)
    w = 0.5 * s                                # trapezoid weights
    off = 1.0 / h_p[:-1]
    t = jnp.diag(-(1.0 / h_m + 1.0 / h_p)) + jnp.diag(off, 1) + jnp.diag(off, -1)
    ws = 1.0 / jnp.sqrt(w)
    return c_minus, c_plus, d_x, w, ws[:, None] * t * ws[None, :]


def _f_of_mu(mu: jax.Array) -> jax.Array:
    """``exp(-gamma h)`` for the outgoing/decaying discrete branch as a
    function of ``mu = cosh(gamma h)``: ``mu - sqrt(mu^2 - 1)``, principal
    sqrt, which is real in (0, 1) for evanescent modes (mu > 1) and
    ``exp(-j arccos mu)`` for propagating ones (|mu| <= 1) -- the referee's
    branch, in closed form."""
    mu_c = jnp.asarray(mu, dtype=jnp.complex128)
    return mu_c - jnp.sqrt(mu_c * mu_c - 1.0)


def _df_dmu(mu: jax.Array) -> jax.Array:
    mu_c = jnp.asarray(mu, dtype=jnp.complex128)
    return 1.0 - mu_c / jnp.sqrt(mu_c * mu_c - 1.0)


@jax.custom_jvp
def _spectral(sym: jax.Array, c: jax.Array, h: float):
    """Spectral quantities of the transverse operator with a robust derivative.

    ``mu_n = 1 - h^2 lam_n / 2 - c`` with ``c = h^2 k^2 / 2``. Returns
    ``(F, lam1, psi1)``: ``F = Psi diag(f(mu)) Psi^T`` (the DtN kernel in the
    symmetrised basis), and the eigenpair of the dominant TE10 mode.

    The derivative is NOT taken through ``eigh`` (whose eigenvector tangents
    divide by eigenvalue gaps and produce nan/garbage near degeneracies of
    the stretched grid's spectrum): ``F`` is differentiated as a matrix
    function (Daleckii-Krein divided differences, which stay bounded), and
    the TE10 pair through first-order perturbation theory, which is safe
    because that mode is isolated.
    """
    lam, psi = jnp.linalg.eigh(sym)
    mu = 1.0 - 0.5 * h * h * lam - c
    f = _f_of_mu(mu)
    big = (psi * f[None, :]) @ psi.T
    return big, lam[-1], psi[:, -1]


@_spectral.defjvp
def _spectral_jvp(primals, tangents):
    sym, c, h = primals
    d_sym, d_c, _ = tangents
    lam, psi = jnp.linalg.eigh(sym)
    mu = 1.0 - 0.5 * h * h * lam - c
    f = _f_of_mu(mu)
    fp = _df_dmu(mu)                                  # df/dmu
    big = (psi * f[None, :]) @ psi.T
    lam1, psi1 = lam[-1], psi[:, -1]

    a_mat = psi.T @ d_sym @ psi                       # projected tangent
    gap = lam[:, None] - lam[None, :]
    scale = jnp.max(jnp.abs(lam))
    close = jnp.abs(gap) < 1e-9 * scale
    safe_gap = jnp.where(close, 1.0, gap)
    g_dd = (f[:, None] - f[None, :]) / safe_gap       # divided differences
    g_diag = fp * (-0.5 * h * h)                      # dg/dlam on the diagonal
    g_mat = jnp.where(close, 0.5 * (g_diag[:, None] + g_diag[None, :]), g_dd)
    d_f_c = -fp * d_c                                 # df/dc dc
    d_big = psi @ (a_mat * g_mat + jnp.diag(d_f_c)) @ psi.T

    d_lam1 = a_mat[-1, -1]
    inv_gap = jnp.where(jnp.arange(lam.shape[0]) == lam.shape[0] - 1, 0.0,
                        1.0 / jnp.where(gap[-1] == 0, 1.0, (lam1 - lam)))
    d_psi1 = psi @ (a_mat[:, -1] * inv_gap)
    return (big, lam1, psi1), (d_big, d_lam1, d_psi1)


def _port_terms(model: HPlaneModel, freq, apertures):
    """k, the DtN kernel Q (nodes x nodes), the TE10 vector phi1 (W-orthonormal),
    its exp(-gamma1 h), the weights and the x-stencil coefficients."""
    h = model.h
    k = 2.0 * jnp.pi * jnp.asarray(freq, dtype=jnp.float64) / C0
    x_all = _node_positions(model, apertures)
    c_minus, c_plus, d_x, w, sym = _transverse(model, x_all)
    big, lam1, psi1 = _spectral(sym, 0.5 * h * h * k * k, h)
    ws = 1.0 / jnp.sqrt(w)
    q = (ws[:, None] * big) * (1.0 / ws)[None, :]     # W^-1/2 F W^1/2
    phi1 = ws * psi1
    f1 = _f_of_mu(1.0 - 0.5 * h * h * lam1 - 0.5 * h * h * k * k)
    return k, q, phi1, f1, w, (c_minus, c_plus, d_x)


def _apertures(model: HPlaneModel, apertures):
    if apertures is None:
        return model.nominal_apertures
    apertures = tuple(apertures)
    if len(apertures) != len(model.nominal_apertures):
        raise ValueError(
            f"expected {len(model.nominal_apertures)} apertures (one per width "
            f"group {model.width_groups}), got {len(apertures)}")
    return apertures


def assemble(model: HPlaneModel, freq, eps_r=None, apertures=None, *, pec: bool = True):
    """(data, rhs) as JAX arrays for the pattern ``model.rows/cols``.

    ``apertures`` -- electrical aperture width [m] per width group (traced
    OK); defaults to the nominal widths. ``pec=False`` drops the irises
    (empty guide on the same, possibly stretched, grid) -- a port gate.
    """
    h, nxi, nzi = model.h, model.nxi, model.nzi
    apertures = _apertures(model, apertures)
    k, q, phi1, f1, w, (c_minus, c_plus, d_x) = _port_terms(model, freq, apertures)
    if eps_r is None:
        eps_r = jnp.ones((nxi, nzi), dtype=jnp.complex128)
    eps_r = jnp.asarray(eps_r, dtype=jnp.complex128)
    if eps_r.shape != (nxi, nzi):
        raise ValueError(f"eps_r must have shape {(nxi, nzi)}, got {eps_r.shape}")

    kind = model.kind
    node_i = model.rows // nzi                 # transverse index of each entry's row
    diag = (d_x[:, None] - 2.0 / h ** 2 + k * k * eps_r).ravel()
    data = jnp.concatenate([
        c_minus[node_i[kind == 0]].astype(jnp.complex128),
        c_plus[node_i[kind == 1]].astype(jnp.complex128),
        jnp.full((int(np.sum(kind == 2)),), 1.0 / h ** 2, dtype=jnp.complex128),
        diag,
        (q / h ** 2).ravel(),
        (q / h ** 2).ravel(),
    ])
    metal = model.metal if pec else np.zeros_like(model.metal)
    row_metal = metal.ravel()[model.rows]
    col_metal = metal.ravel()[model.cols]
    data = jnp.where(row_metal | col_metal, 0.0, data)
    data = jnp.where((kind == 3) & row_metal, 1.0, data)

    rhs = jnp.zeros((nxi, nzi), dtype=jnp.complex128)
    rhs = rhs.at[:, 0].set(-(1.0 / f1 - f1) * phi1 / h ** 2)
    rhs = jnp.where(metal, 0.0, rhs).ravel()
    return data, rhs


def s_params(model: HPlaneModel, freq, e: jax.Array, apertures=None):
    """(S11, S21) read off a solution field ``e`` of shape (nxi, nzi)."""
    apertures = _apertures(model, apertures)
    _, _, phi1, f1, w, _ = _port_terms(model, freq, apertures)
    s11 = jnp.sum(phi1 * w * e[:, 0]) - 1.0
    s21 = jnp.sum(phi1 * w * e[:, -1]) * f1 ** (-model.nz)
    return s11, s21


def solve(model: HPlaneModel, freq, eps_r=None, apertures=None, *,
          pec: bool = True, return_field: bool = False):
    """Solve one frequency. Differentiable in ``freq``, ``eps_r`` and ``apertures``.

    Returns ``(s11, s21)``, or ``(s11, s21, E)`` with ``E`` of shape
    ``(nxi, nzi)`` when ``return_field`` is set.
    """
    data, rhs = assemble(model, freq, eps_r, apertures, pec=pec)
    x = sparse_solve(data, model.rows, model.cols, rhs)
    e = x.reshape(model.nxi, model.nzi)
    s11, s21 = s_params(model, freq, e, apertures)
    if return_field:
        return s11, s21, e
    return s11, s21


def richardson_first_order(value_coarse, r_coarse, value_fine, r_fine):
    """h ∝ 1/r, so f(r) = f_exact + c/r (first-order PEC staircase)."""
    if not r_fine > r_coarse >= 1:
        raise ValueError("need r_fine > r_coarse >= 1")
    return (r_fine * value_fine - r_coarse * value_coarse) / (r_fine - r_coarse)
