"""E6 — in-plane design-variable autodiff (Lane 2 of the NU full-functionality
program, docs/design_notes/20260913_nu_full_functionality_program.md section
5; this lane's own note is
docs/design_notes/20260913_nu_lane2_inplane_designvar_ad_predeclaration.md).

Question: is the reverse-mode gradient of a 3-D NU simulation correct along
two PHYSICAL in-plane design variables — the width ``w`` of a fine x band (a
trace-width surrogate: a PEC strip whose x edges are the band edges, attached
by node index) and its position ``x_c`` — including the CFL ``dt`` path
through the x tied set, for (a) a transient loss (L1 class) and (b) a smooth
spectral observable (physical-time Hann taper plus a 0.5 GHz comb integral)?

Judges are imported verbatim from ``adq_designvar`` (``fit_order``,
``fd_budget``, ``selfcheck``, ``coverage``, ``ulp``, ``losses_along``, ``HS``,
``N_QUANTA``, ``RHO_BAND``, ``FLOOR_HS``); ``adq_designvar.py`` and
``e4_diff_stackup.py`` are not edited by this lane. No ``rfx/`` change.

The mesh map (``MeshMap``) is a GENERAL differentiable edges -> cells map:
fixed per-segment cell counts, exact total length, every interface on a node
by construction, pinned end cells exactly fixed, materials attached by node
index. It lives here (validation/research), not in ``rfx/``.

Usage (declared in the note; run from the worktree with the pinned PYTHONPATH,
one attempt per arm — the ``.started`` claim file refuses a rerun)::

    python -m validation.research.multiband_nu.e6_inplane_designvar --arm model       (zero FDTD)
    python -m validation.research.multiband_nu.e6_inplane_designvar --arm map_checks  (E6-M, two L1 runs)
    python -m validation.research.multiband_nu.e6_inplane_designvar --arm l1          (E6-L1)
    python -m validation.research.multiband_nu.e6_inplane_designvar --arm revert_l1_nodt  (E6-R)
    python -m validation.research.multiband_nu.e6_inplane_designvar --arm l2s         (E6-L2s)
    python -m validation.research.multiband_nu.e6_inplane_designvar --arm coverage    (E6-C, zero FDTD)
    python -m validation.research.multiband_nu.e6_inplane_designvar --arm diag_revert_cellwise
        (second-pass producer of the E6-R cell-wise diagnostic: two gradients, zero ladder runs)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
import numpy as np
import rfx
from rfx.core.yee import MaterialArrays
from rfx.nonuniform import C0, make_band_profile, make_nonuniform_grid, run_nonuniform

from .adq_designvar import (
    FLOOR_HS, HS, N_QUANTA, RHO_BAND, coverage, fd_budget, fit_order, losses_along, selfcheck, ulp,
)
from .w6_band_builder import _git_dirty, _git_sha

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'validation/research/multiband_nu/results'

# --- fixture (program 5.2; every length in metres) ------------------------------
X_EDGES = (0.0, 8e-3, 10e-3, 18e-3)
X_SIZES = (0.5e-3, 0.1e-3, 0.5e-3)
X_PROTECTED = (False, True, False)
CAP = 1.3
PIN = 0.5e-3                     # boundary_cell on x (the x/y end-cell contract, G1/G11)
DYZ = 0.5e-3
NY, NZ = 12, 24
DOMAIN_XY = (18e-3, 6e-3)
K_S = 12                         # PEC strip z-node plane
SRC_IJK = (8, 6, 6)              # ez soft source, lead segment, by index
F0 = 20e9
SIGMA_T = 20e-12
T0 = 5 * SIGMA_T
N1 = 600                         # L1 steps
N2 = 2400                        # L2s steps
CHECKPOINT_EVERY = 100           # L2s only (E4's value)
T_W_FRACTION = 0.8               # T_w = 0.8 N2 dt(p0), fixed at the nominal design
COMB_DF = 0.5e9
COMB_K = 10                      # k = -10 .. 10
# --- design variables ---------------------------------------------------------------
W0 = 2e-3
XC0 = 9e-3
P0 = np.array([W0, XC0])
NAMES = ('w', 'x_c')
SEG_LEN0 = np.diff(np.asarray(X_EDGES))                       # (8, 2, 8) mm
A = np.array([[-0.5, 1.0], [1.0, 0.0], [-0.5, -1.0]])         # seg_len = SEG_LEN0 + A @ delta
LADDER_SCALE = W0                # absolute displacement h x w0 for BOTH controls
# --- map-check thresholds (program 5.3, instrument checks, not windows) ---------
M_LEN_TOL = 1e-9
M_IFACE_TOL = 1e-9
M_JAC_REL = 1e-5
M_DT_REL = 1e-6
M_STRIP_NODES = 21
M_FWD_REL = 1e-6


# ------------------------------------------------------------------------------------
# The map
# ------------------------------------------------------------------------------------
class MeshMap:
    """Edges -> cells with fixed per-segment cell counts (host-built, pure-jnp
    forward).

    ``cells(seg_len) = where(pinned, cells0, cells0 * (seg_len - pinned_len)
    [seg_id] / (seg_len0 - pinned_len)[seg_id])``: every free cell of a
    segment scales with the segment's free length, so the segment sums to
    ``seg_len`` exactly and every interface stays on the same node; pinned
    cells never move (boundary Jacobian exactly 0). Materials are attached by
    node index (``edge_nodes``), never by coordinate.
    """

    def __init__(self, cells0, seg_id, pinned, seg_len0):
        self.cells0 = np.asarray(cells0, np.float64)
        self.seg_id = np.asarray(seg_id, int)
        self.pinned = np.asarray(pinned, bool)
        self.seg_len0 = np.asarray(seg_len0, np.float64)
        self.n_seg = int(self.seg_len0.size)
        self.counts = np.array([int(np.sum(self.seg_id == s)) for s in range(self.n_seg)])
        self.pinned_len = np.array([float(self.cells0[self.pinned & (self.seg_id == s)].sum())
                                    for s in range(self.n_seg)])
        self.free_len0 = self.seg_len0 - self.pinned_len
        self.edge_nodes = np.concatenate([[0], np.cumsum(self.counts)]).astype(int)

    @classmethod
    def from_builder(cls, edges, cell_sizes, protected, max_ratio, boundary_cell):
        cells0 = make_band_profile(edges, cell_sizes, protected=protected, max_ratio=max_ratio,
                                   boundary_cell=boundary_cell)
        nodes = np.concatenate([[0.0], np.cumsum(cells0)])
        e = np.asarray(edges, np.float64)
        idx = [int(np.argmin(np.abs(nodes - x))) for x in e]
        seg_id = np.zeros(cells0.size, int)
        for s in range(e.size - 1):
            seg_id[idx[s]:idx[s + 1]] = s
        pinned = np.zeros(cells0.size, bool)
        if boundary_cell is not None:
            pinned[0] = pinned[-1] = True
        return cls(cells0, seg_id, pinned, np.diff(e))

    def cells(self, seg_len):
        seg_len = jnp.asarray(seg_len)
        dt = seg_len.dtype
        cells0 = jnp.asarray(self.cells0, dt)
        pinned_len = jnp.asarray(self.pinned_len, dt)
        seg_len0 = jnp.asarray(self.seg_len0, dt)
        scale = (seg_len - pinned_len) / (seg_len0 - pinned_len)
        return jnp.where(jnp.asarray(self.pinned), cells0, cells0 * scale[jnp.asarray(self.seg_id)])

    def cells_f64(self, seg_len):
        seg_len = np.asarray(seg_len, np.float64)
        scale = (seg_len - self.pinned_len) / self.free_len0
        return np.where(self.pinned, self.cells0, self.cells0 * scale[self.seg_id])

    def jacobian_seg(self):
        """d cells / d seg_len (n_cells x n_seg), exact float64."""
        j = np.zeros((self.cells0.size, self.n_seg))
        for s in range(self.n_seg):
            col = (self.seg_id == s) & ~self.pinned
            j[col, s] = self.cells0[col] / self.free_len0[s]
        return j


MAP = MeshMap.from_builder(X_EDGES, X_SIZES, X_PROTECTED, CAP, PIN)
I_L, I_R = int(MAP.edge_nodes[1]), int(MAP.edge_nodes[2])          # band edge nodes (20, 40)
PRB_IJK = (I_R + 3, 6, K_S + 2)
CELLS0_F32 = jnp.asarray(MAP.cells0, jnp.float32)
J_PARAM = MAP.jacobian_seg() @ A                                   # d cells / d p (60 x 2), float64


def seg_len_of(delta):
    """Segment lengths at design displacement ``delta = p - p0`` (pure jnp)."""
    delta = jnp.asarray(delta)
    return jnp.asarray(SEG_LEN0, delta.dtype) + jnp.asarray(A, delta.dtype) @ delta


def cells_of(delta):
    return MAP.cells(seg_len_of(delta))


def cells_f64_of(delta):
    return MAP.cells_f64(SEG_LEN0 + A @ np.asarray(delta, np.float64))


def dt_f64(cells):
    d = np.asarray(cells, np.float64)
    return 0.99 / (C0 * np.sqrt(1 / d.min() ** 2 + 1 / DYZ ** 2 + 1 / DYZ ** 2))


DT0 = dt_f64(MAP.cells0)
T_W = T_W_FRACTION * N2 * DT0
COMB = np.asarray([F0 + k * COMB_DF for k in range(-COMB_K, COMB_K + 1)])


def ddt_dw_analytic():
    """d(dt)/dw = d(dt)/d(dx_min) x d(dx_min)/dw; band cells scale by 1/free_len0."""
    d = float(MAP.cells0.min())
    s = 1 / d ** 2 + 2 / DYZ ** 2
    ddt_ddmin = 0.99 / C0 * s ** -1.5 / d ** 3
    ddmin_dw = d / MAP.free_len0[1] * A[1, 0]
    return float(ddt_ddmin * ddmin_dw)


# ------------------------------------------------------------------------------------
# The fixture and the losses
# ------------------------------------------------------------------------------------
def _grid(cells):
    return make_nonuniform_grid(DOMAIN_XY, np.full(NZ, DYZ), PIN, cpml_layers=0, dx_profile=cells)


def _vacuum(grid):
    shape = grid.shape
    return MaterialArrays(eps_r=jnp.ones(shape, jnp.float32), mu_r=jnp.ones(shape, jnp.float32),
                          sigma=jnp.zeros(shape, jnp.float32))


def strip_edge_masks(shape, i_l=I_L, i_r=I_R, k_s=K_S):
    """PEC strip on the z-node plane ``k_s`` for x nodes ``i_l .. i_r``, all y:
    tangential ``ex`` edges (between nodes i_l..i_r) and ``ey`` edges (on the
    nodes) zeroed; ``ez`` untouched. Attached BY NODE INDEX."""
    mx = np.zeros(shape, bool)
    my = np.zeros(shape, bool)
    mz = np.zeros(shape, bool)
    mx[i_l:i_r, :, k_s] = True
    my[i_l:i_r + 1, :, k_s] = True
    return (jnp.asarray(mx), jnp.asarray(my), jnp.asarray(mz))


GRID_SHAPE = (MAP.cells0.size + 1, NY + 1, NZ + 1)
MASKS = strip_edge_masks(GRID_SHAPE)


def _waveform(n_steps, dt):
    t = jnp.arange(n_steps, dtype=jnp.float32) * dt
    return (jnp.exp(-(((t - T0) / SIGMA_T) ** 2)) * jnp.sin(2 * jnp.pi * F0 * t)).astype(jnp.float32)


def _run(cells, n_steps, checkpoint_every=None, stop_dt=False):
    grid = _grid(cells)
    if stop_dt:
        grid = grid._replace(dt=jax.lax.stop_gradient(grid.dt))
    dt = grid.dt
    wf = _waveform(n_steps, dt)
    out = run_nonuniform(grid, _vacuum(grid), n_steps, pec_edge_masks=MASKS,
                         sources=[(*SRC_IJK, 'ez', wf)], probes=[(*PRB_IJK, 'ez')],
                         checkpoint_every=checkpoint_every)
    return out['time_series'][:, 0], dt


def l1_cells(cells, stop_dt=False):
    ts, _ = _run(cells, N1, stop_dt=stop_dt)
    return jnp.sum(ts ** 2)


def l2s_cells(cells, stop_dt=False):
    ts, dt = _run(cells, N2, CHECKPOINT_EVERY, stop_dt)
    t = jnp.arange(N2, dtype=jnp.float32) * dt
    tw = jnp.float32(T_W)
    hann = jnp.where(t < tw, 0.5 * (1 - jnp.cos(2 * jnp.pi * t / tw)), 0.0).astype(jnp.float32)
    phase = 2 * jnp.pi * jnp.asarray(COMB, jnp.float32)[:, None] * t[None, :]
    wts = hann * ts
    re = dt * jnp.sum(wts[None, :] * jnp.cos(phase), axis=1)
    im = dt * jnp.sum(wts[None, :] * jnp.sin(phase), axis=1)
    return jnp.sum(re ** 2 + im ** 2)


LOSSES = {'l1': l1_cells, 'l2s': l2s_cells}


# ------------------------------------------------------------------------------------
# Model (zero FDTD): the numbers the note freezes
# ------------------------------------------------------------------------------------
def ladder_geometry():
    """Per ladder point and control, both signs: seam ratios, tied set, dt, T_w
    margin — from the float32 map (what the traced solve sees) and float64."""
    rows = []
    for i, name in enumerate(NAMES):
        v = np.eye(2)[i] * LADDER_SCALE
        for h in HS:
            for sign in (+1, -1):
                d64 = cells_f64_of(sign * h * v)
                d32 = np.asarray(cells_of(jnp.asarray(sign * h * v, jnp.float32)), np.float64)
                f32 = d32.astype(np.float32)
                tied = np.flatnonzero(f32 == f32.min()).tolist()
                dt = dt_f64(d64)
                rows.append({'control': name, 'h': h, 'sign': sign,
                             'seam_ratio_lo': float(d64[I_L - 1] / d64[I_L]),
                             'seam_ratio_hi': float(d64[I_R] / d64[I_R - 1]),
                             'band_cell': float(d64[I_L]), 'first_ramp_lo': float(d64[I_L - 1]),
                             'tied_set': tied, 'tied_is_band': tied == list(range(I_L, I_R)),
                             'dt': dt, 'dt_over_dt0': dt / DT0,
                             'n2_dt_over_tw': N2 * dt / T_W,
                             'length_err_m': float(abs(d64.sum() - (X_EDGES[-1] - X_EDGES[0]))),
                             'length_err_f32_m': float(abs(d32.sum() - (X_EDGES[-1] - X_EDGES[0])))})
    return rows


def model():
    jac32 = np.asarray(jax.jacfwd(cells_of)(jnp.zeros(2, jnp.float32)), np.float64)
    tied = np.flatnonzero(MAP.cells0 == MAP.cells0.min())
    out = {'cells0': MAP.cells0.tolist(), 'seg_id': MAP.seg_id.tolist(), 'pinned': MAP.pinned.tolist(),
           'counts': MAP.counts.tolist(), 'edge_nodes': MAP.edge_nodes.tolist(),
           'seg_len0': SEG_LEN0.tolist(), 'free_len0': MAP.free_len0.tolist(),
           'pinned_len': MAP.pinned_len.tolist(), 'A': A.tolist(), 'p0': P0.tolist(),
           'ladder_scale': LADDER_SCALE, 'jacobian_param_f64': J_PARAM.tolist(),
           'jacobian_param_f32_ad': jac32.tolist(),
           'jacobian_f32_vs_f64_max_abs': float(np.max(np.abs(jac32 - J_PARAM))),
           'tied_cells_x': tied.tolist(), 'n_tied_xyz': [int(tied.size), NY, NZ],
           'tie_spread': [float(np.ptp(J_PARAM[tied, i])) for i in range(2)],
           'boundary_jacobian': J_PARAM[MAP.pinned].tolist(),
           'seam_ratio_nominal': float(MAP.cells0[I_L - 1] / MAP.cells0[I_L]),
           'dt0': DT0, 'dt0_grid_f32': float(_grid(MAP.cells0).dt),
           'ddt_dw_analytic': ddt_dw_analytic(),
           'grid_shape': list(GRID_SHAPE), 'strip_nodes': int(np.sum(np.asarray(MASKS[1])[:, 6, K_S])),
           'strip_ex_edges': int(np.sum(np.asarray(MASKS[0])[:, 6, K_S])),
           'i_l': I_L, 'i_r': I_R, 'k_s': K_S, 'src_ijk': list(SRC_IJK), 'prb_ijk': list(PRB_IJK),
           'f0': F0, 'sigma_t': SIGMA_T, 't0': T0, 'n1': N1, 'n2': N2, 't1_s': N1 * DT0,
           't2_s': N2 * DT0, 't_w': T_W, 'comb_hz': COMB.tolist(), 'comb_width_hz': float(COMB[-1] - COMB[0]),
           'one_over_tw_hz': 1 / T_W, 'checkpoint_every': CHECKPOINT_EVERY,
           'ladder': ladder_geometry()}
    return out


# ------------------------------------------------------------------------------------
# E6-M map checks (m1-m8; m8 needs two L1 solves)
# ------------------------------------------------------------------------------------
def map_checks():
    ch = {}
    rows = ladder_geometry()
    ch['m1_length_err_max_m'] = max(r['length_err_m'] for r in rows)
    ch['m1_pass'] = ch['m1_length_err_max_m'] <= M_LEN_TOL
    worst = 0.0
    for i, name in enumerate(NAMES):
        v = np.eye(2)[i] * LADDER_SCALE
        for h in HS:
            for sign in (+1, -1):
                d64 = cells_f64_of(sign * h * v)
                nodes = np.concatenate([[0.0], np.cumsum(d64)])
                edges = np.concatenate([[0.0], np.cumsum(SEG_LEN0 + A @ (sign * h * v))])
                worst = max(worst, float(np.max(np.abs(nodes[MAP.edge_nodes] - edges))))
    ch['m2_iface_err_max_m'] = worst
    ch['m2_pass'] = worst <= M_IFACE_TOL
    tied = np.flatnonzero(MAP.cells0 == MAP.cells0.min())
    ch['m3_tie_spread'] = [float(np.ptp(J_PARAM[tied, i])) for i in range(2)]
    ch['m3_tied_is_band_everywhere'] = all(r['tied_is_band'] for r in rows)
    ch['m3_tied_cells'] = tied.tolist()
    ch['m3_pass'] = (ch['m3_tie_spread'] == [0.0, 0.0] and ch['m3_tied_is_band_everywhere']
                     and tied.tolist() == list(range(I_L, I_R)))
    ch['m4_boundary_jacobian'] = J_PARAM[MAP.pinned].tolist()
    jac32 = np.asarray(jax.jacfwd(cells_of)(jnp.zeros(2, jnp.float32)), np.float64)
    ch['m4_boundary_jacobian_f32_ad'] = jac32[MAP.pinned].tolist()
    ch['m4_pass'] = bool(np.all(J_PARAM[MAP.pinned] == 0.0) and np.all(jac32[MAP.pinned] == 0.0))
    # m5: J^T g vs the direct parameter gradient (L1)
    g_cell = np.asarray(jax.jit(jax.grad(l1_cells))(CELLS0_F32), np.float64)
    g_p = np.asarray(jax.jit(jax.grad(lambda d: l1_cells(cells_of(d))))(jnp.zeros(2, jnp.float32)), np.float64)
    chain = J_PARAM.T @ g_cell
    ch['m5_g_cell'] = g_cell.tolist()
    ch['m5_g_param'] = g_p.tolist()
    ch['m5_chain_rule'] = chain.tolist()
    ch['m5_rel'] = [float(abs(c - g) / abs(g)) if g else None for c, g in zip(chain, g_p)]
    ch['m5_pass'] = all(r is not None and r <= M_JAC_REL for r in ch['m5_rel'])
    # m6: d(dt)/dw AD (float32, through make_nonuniform_grid) vs analytic
    ddt = np.asarray(jax.grad(lambda d: _grid(cells_of(d)).dt)(jnp.zeros(2, jnp.float32)), np.float64)
    ana = ddt_dw_analytic()
    ch['m6_ddt_dp_ad'] = ddt.tolist()
    ch['m6_ddt_dw_analytic'] = ana
    ch['m6_rel'] = float(abs(ddt[0] - ana) / abs(ana))
    ch['m6_xc_exactly_zero'] = bool(ddt[1] == 0.0)
    ch['m6_pass'] = ch['m6_rel'] <= M_DT_REL and ch['m6_xc_exactly_zero']
    # m7: strip occupancy and the kwarg
    ch['m7_strip_nodes'] = int(np.sum(np.asarray(MASKS[1])[:, 6, K_S]))
    ch['m7_kwarg'] = 'run_nonuniform(pec_edge_masks=(Mx, My, Mz)) accepted'
    ch['m7_topology_fixed'] = 'masks are index-attached constants; identical at every ladder point'
    ch['m7_pass'] = ch['m7_strip_nodes'] == M_STRIP_NODES
    # m8: traced loss at p0 vs the concrete float64-map loss at p0
    traced = float(jax.jit(lambda d: l1_cells(cells_of(d)))(jnp.zeros(2, jnp.float32)))
    grid_c = _grid(MAP.cells0)                                  # concrete path, float64 profile
    concrete = float(jax.jit(lambda: l1_cells_on_grid(grid_c))())
    ch['m8_loss_traced'] = traced
    ch['m8_loss_concrete'] = concrete
    ch['m8_rel'] = float(abs(traced - concrete) / abs(concrete))
    ch['m8_dt_traced_f32'] = float(jax.jit(lambda d: _grid(cells_of(d)).dt)(jnp.zeros(2, jnp.float32)))
    ch['m8_dt_concrete'] = float(grid_c.dt)
    ch['m8_pass'] = ch['m8_rel'] <= M_FWD_REL
    ch['all_pass'] = all(ch[f'm{k}_pass'] for k in range(1, 9))
    ch['ladder_geometry'] = rows
    return ch


def l1_cells_on_grid(grid):
    wf = _waveform(N1, grid.dt)
    out = run_nonuniform(grid, _vacuum(grid), N1, pec_edge_masks=MASKS,
                         sources=[(*SRC_IJK, 'ez', wf)], probes=[(*PRB_IJK, 'ez')])
    return jnp.sum(out['time_series'][:, 0] ** 2)


# ------------------------------------------------------------------------------------
# Floor and judging (AD-Q verbatim judges; floor = the L2 second-attempt method)
# ------------------------------------------------------------------------------------
def measure_floor(loss, v):
    hs = np.asarray(FLOOR_HS)
    ys = np.array([float(loss(jnp.asarray(h * v, jnp.float32))) for h in hs])
    res2 = ys - np.polyval(np.polyfit(hs, ys, 2), hs)
    res3 = ys - np.polyval(np.polyfit(hs, ys, 3), hs)
    sigma = float(np.sqrt(np.sum(res2 ** 2) / (len(hs) - 3)))
    rms2, rms3 = float(np.sqrt(np.mean(res2 ** 2))), float(np.sqrt(np.mean(res3 ** 2)))
    return {'floor_hs': hs.tolist(), 'floor_losses': ys.tolist(), 'sigma': sigma,
            'rms_quadratic': rms2, 'rms_cubic': rms3, 'sigma_reliable': bool(rms3 >= 0.9 * rms2)}


def judge(points, loss0, ad, sigma_eff):
    order = fit_order(points, loss0, ad, sigma=sigma_eff)
    fd = fd_budget(points, ad, LADDER_SCALE, sigma=sigma_eff)
    elig = [k for k, e in enumerate(order['eligible']) if e]
    bound = min((order['r1'][k] / (points[k]['h'] * abs(ad)) for k in elig), default=None) if ad else None
    return {'order': order, 'fd': fd,
            'order_1ulp': fit_order(points, loss0, ad), 'fd_1ulp': fd_budget(points, ad, LADDER_SCALE),
            'lower_side_diagnostic': (None if 'R1' not in order else bool(order['R1']['slope'] >= 1.8)),
            'empirical_rel_error_bound': bound,
            'both_held': bool(order['verdict'] == fd['verdict'] == 'HELD')}


def measure_arm(arm):
    cell_loss = LOSSES[arm]
    loss = jax.jit(lambda d: cell_loss(cells_of(d)))
    zero = jnp.zeros(2, jnp.float32)
    loss0 = float(loss(zero))
    g_cell = np.asarray(jax.jit(jax.grad(cell_loss))(CELLS0_F32), np.float64)
    g_p = np.asarray(jax.jit(jax.grad(lambda d: cell_loss(cells_of(d))))(zero), np.float64)
    tied = np.flatnonzero(MAP.cells0 == MAP.cells0.min())
    out = {'loss0': loss0, 'loss_ulp': ulp(loss0), 'n_steps': N1 if arm == 'l1' else N2,
           'p0': P0.tolist(), 'ladder_scale': LADDER_SCALE, 'cell0': MAP.cells0.tolist(),
           'jacobian': J_PARAM.tolist(), 'g_cell': g_cell.tolist(), 'g_param': g_p.tolist(),
           'chain_rule': (J_PARAM.T @ g_cell).tolist(), 'tied_cells_x': tied.tolist(),
           'n_tied_xyz': [int(tied.size), NY, NZ], 'directions': {}}
    held = []
    for i, name in enumerate(NAMES):
        v = np.eye(2)[i] * LADDER_SCALE
        floor = measure_floor(loss, v)
        sigma_eff = max(ulp(loss0), floor['sigma'])
        points = losses_along(loss, np.zeros(2), v, HS)
        ad = float(g_p[i] * LADDER_SCALE)
        spread = float(np.ptp(J_PARAM[tied, i]))
        assert spread == 0.
        rec = judge(points, loss0, ad, sigma_eff)
        rec.update({'floor': floor, 'sigma_eff': sigma_eff, 'sigma_in_ulp': floor['sigma'] / ulp(loss0),
                    'induced_direction': (J_PARAM[:, i] * LADDER_SCALE).tolist(),
                    'tie_spread_xyz': [spread, 0., 0.], 'ad_relative': ad, 'ad_physical': float(g_p[i])})
        out['directions'][name] = rec
        if rec['both_held']:
            held.append(i)
        print(arm, name, f"sigma={floor['sigma'] / ulp(loss0):.1f}ulp reliable={floor['sigma_reliable']}",
              rec['order'].get('R0', {}).get('slope'), rec['order'].get('R1', {}).get('slope'),
              rec['order']['verdict'], rec['fd']['verdict'], flush=True)
    out['verified_controls'] = [NAMES[i] for i in held]
    out['coverage_all'] = coverage(g_cell, J_PARAM)
    out['coverage_verified'] = coverage(g_cell, J_PARAM[:, held])
    out['coverage_dimensionless'] = coverage(g_cell * MAP.cells0, J_PARAM / MAP.cells0[:, None])
    return out


def measure_revert():
    """Revert-proof (AD-Q C, measure_revert pattern): TRUE L1 losses along both
    controls judged against the gradient WITHOUT its dt path. ``w`` moves the x
    tied set and must FIRE; ``x_c`` leaves the band cells unchanged (dt share
    exactly 0) and must return E6-L1's verdicts. The floor sigma is E6-L1's
    (same loss function, same nominal)."""
    first = json.loads((RESULTS / 'e6_l1.json').read_text())
    assert 'instrument_error' not in first
    loss = jax.jit(lambda d: l1_cells(cells_of(d)))
    zero = jnp.zeros(2, jnp.float32)
    g_true = np.asarray(jax.jit(jax.grad(lambda d: l1_cells(cells_of(d))))(zero), np.float64)
    g_nodt = np.asarray(jax.jit(jax.grad(lambda d: l1_cells(cells_of(d), stop_dt=True)))(zero), np.float64)
    loss0 = float(loss(zero))
    nodt0 = float(jax.jit(lambda d: l1_cells(cells_of(d), stop_dt=True))(zero))
    out = {'first_attempt_key': 'e6_l1', 'loss0': loss0, 'loss0_first': first['loss0'],
           'loss0_matches_first': bool(loss0 == first['loss0']),
           'forward_values_identical': bool(loss0 == nodt0),
           'g_param_true': g_true.tolist(), 'g_param_nodt': g_nodt.tolist(),
           'g_param_true_matches_first': bool(np.array_equal(g_true, np.asarray(first['g_param']))),
           'dt_path_share': [float((t - n) / t) if t else None for t, n in zip(g_true, g_nodt)],
           'directions': {}}
    for i, name in enumerate(NAMES):
        v = np.eye(2)[i] * LADDER_SCALE
        sigma_eff = first['directions'][name]['sigma_eff']
        points = losses_along(loss, np.zeros(2), v, HS)
        ad = float(g_nodt[i] * LADDER_SCALE)
        rec = judge(points, loss0, ad, sigma_eff)
        rec.update({'sigma_eff': sigma_eff, 'ad_relative_nodt': ad,
                    'ad_relative_true': float(g_true[i] * LADDER_SCALE),
                    'first_attempt_order': first['directions'][name]['order']['verdict'],
                    'first_attempt_fd': first['directions'][name]['fd']['verdict'],
                    'ladder_matches_first': bool(points == [{'h': s['h'], 'loss_plus': s['loss_plus'],
                                                             'loss_minus': s['loss_minus']}
                                                            for s in first['directions'][name]['fd']['steps']])})
        out['directions'][name] = rec
        print('revert', name, rec['order'].get('R1', {}).get('slope'), rec['order']['verdict'],
              rec['fd']['verdict'], flush=True)
    w, xc = out['directions']['w'], out['directions']['x_c']
    # Declared (program 5.4): w FIRED on order = R1 slope below 1.8, or no fit at all.
    out['w_fired_on_order'] = bool(w['order']['verdict'] != 'HELD'
                                   and ('R1' not in w['order'] or w['order']['R1']['slope'] < 1.8))
    out['xc_unchanged'] = bool(xc['order']['verdict'] == xc['first_attempt_order']
                               and xc['fd']['verdict'] == xc['first_attempt_fd'])
    out['requirement_met'] = bool(out['w_fired_on_order'] and out['xc_unchanged'])
    return out


def coverage_summary():
    out = {'arms': {}}
    for arm in ('l1', 'l2s'):
        d = json.loads((RESULTS / f'e6_{arm}.json').read_text())
        assert 'instrument_error' not in d
        g, jac, c0 = np.asarray(d['g_cell']), np.asarray(d['jacobian']), np.asarray(d['cell0'])
        held = [NAMES.index(n) for n in d['verified_controls']]
        out['arms'][arm] = {'verified_controls': d['verified_controls'],
                            'coverage_all': coverage(g, jac), 'coverage_verified': coverage(g, jac[:, held]),
                            'coverage_dimensionless': coverage(g * c0, jac / c0[:, None]),
                            'coverage_w_only': coverage(g, jac[:, :1]), 'coverage_xc_only': coverage(g, jac[:, 1:]),
                            'g_cell_norm': float(np.linalg.norm(g)),
                            'g_cell_norm_band': float(np.linalg.norm(g[I_L:I_R])),
                            'g_cell_norm_lead': float(np.linalg.norm(g[:I_L])),
                            'g_cell_norm_tail': float(np.linalg.norm(g[I_R:]))}
    return out


def diag_revert_cellwise():
    """Post-hoc diagnostic on E6-R (declared in the note's "Second pass"
    section): the cell-wise L1 gradient WITH and WITHOUT the dt path — two
    backward passes at the nominal, zero ladder runs, not a window. The first
    pass of this diagnostic (``4f2d22af``) was produced by a scratch script and
    is kept as ``e6_diag_revert_cellwise_firstpass_4f2d22af.json``; this arm is
    the committed producer and records whether it reproduces that file bit for
    bit."""
    first = json.loads((RESULTS / 'e6_l1.json').read_text())
    rv = json.loads((RESULTS / 'e6_revert_l1_nodt.json').read_text())
    assert 'instrument_error' not in first and 'instrument_error' not in rv
    zero = jnp.zeros(2, jnp.float32)
    g_true = np.asarray(jax.jit(jax.grad(l1_cells))(CELLS0_F32), np.float64)
    g_nodt = np.asarray(jax.jit(jax.grad(lambda c: l1_cells(c, stop_dt=True)))(CELLS0_F32), np.float64)
    p_true = np.asarray(jax.jit(jax.grad(lambda d: l1_cells(cells_of(d))))(zero), np.float64)
    p_nodt = np.asarray(jax.jit(jax.grad(lambda d: l1_cells(cells_of(d), stop_dt=True)))(zero), np.float64)
    tied = np.flatnonzero(MAP.cells0 == MAP.cells0.min())
    non = np.setdiff1d(np.arange(MAP.cells0.size), tied)
    diff = g_true - g_nodt
    chain_true, chain_nodt = J_PARAM.T @ g_true, J_PARAM.T @ g_nodt
    out = {'purpose': ('post-hoc diagnostic on E6-R: cell-wise L1 gradient with and without the dt path '
                       '(two backward passes, zero ladder runs); not a window'),
           'g_cell_true_reproduced_bitwise': bool(np.array_equal(g_true, np.asarray(first['g_cell']))),
           'param_ad_true_reproduced_bitwise': bool(np.array_equal(p_true, np.asarray(rv['g_param_true']))),
           'param_ad_nodt_reproduced_bitwise': bool(np.array_equal(p_nodt, np.asarray(rv['g_param_nodt']))),
           'g_cell_true': g_true.tolist(), 'g_cell_nodt': g_nodt.tolist(),
           'nontied_max_abs_diff': float(np.max(np.abs(diff[non]))),
           'nontied_max_rel_diff': float(np.max(np.abs(diff[non]) / np.abs(g_true[non]))),
           'nontied_diff_is_zero': bool(np.all(diff[non] == 0.0)),
           'tied_diff_mean': float(diff[tied].mean()), 'tied_diff_spread': float(np.ptp(diff[tied])),
           'chain_true': chain_true.tolist(), 'chain_nodt': chain_nodt.tolist(),
           'chain_xc_share': float(J_PARAM[:, 1] @ diff / chain_true[1]),
           'chain_w_share': float(J_PARAM[:, 0] @ diff / chain_true[0]),
           'param_ad_true': p_true.tolist(), 'param_ad_nodt': p_nodt.tolist(),
           'param_dt_share': [float((t - n) / t) for t, n in zip(p_true, p_nodt)],
           'xc_lead_contrib_true': float(J_PARAM[:I_L, 1] @ g_true[:I_L]),
           'xc_tail_contrib_true': float(J_PARAM[I_R:, 1] @ g_true[I_R:])}
    fp = RESULTS / 'e6_diag_revert_cellwise_firstpass_4f2d22af.json'
    if fp.exists():
        f = json.loads(fp.read_text())
        out['firstpass_file'] = fp.name
        out['firstpass_git_sha'] = f['git_sha']
        out['g_cell_nodt_matches_firstpass'] = bool(np.array_equal(g_nodt, np.asarray(f['g_cell_nodt'])))
        out['scalars_match_firstpass'] = {k: bool(out[k] == f[k]) for k in (
            'nontied_max_abs_diff', 'nontied_max_rel_diff', 'tied_diff_mean', 'tied_diff_spread',
            'chain_xc_share', 'chain_w_share', 'xc_lead_contrib_true', 'xc_tail_contrib_true',
            'param_dt_share')}
    return out


ARMS = {'model': model, 'map_checks': map_checks, 'l1': lambda: measure_arm('l1'),
        'revert_l1_nodt': measure_revert, 'l2s': lambda: measure_arm('l2s'), 'coverage': coverage_summary,
        'diag_revert_cellwise': diag_revert_cellwise}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(ARMS), required=True)
    args = parser.parse_args(argv)
    assert str(Path(rfx.__file__).resolve()).startswith(str(ROOT) + '/'), rfx.__file__
    assert all(d.platform == 'cpu' for d in jax.devices())
    out = RESULTS / f'e6_{args.arm}.json'
    if out.exists():
        raise FileExistsError(out)
    sha, dirty = _git_sha(), _git_dirty()          # read BEFORE the claim file exists
    # Refuse to overwrite evidence or rerun an already started arm.
    claim = out.with_suffix('.started')
    with claim.open('x') as f:
        f.write(f'{time.time()}\n')
    provenance = {'rfx_file': rfx.__file__, 'git_sha': sha, 'git_dirty': dirty,
                  'argv': sys.argv[1:], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'platform': 'cpu', 'arm': args.arm, 'hs': HS, 'floor_hs': list(FLOOR_HS),
                  'n_quanta': N_QUANTA, 'rho_band': RHO_BAND, 'fixture': {
                      'x_edges': X_EDGES, 'x_sizes': X_SIZES, 'x_protected': X_PROTECTED, 'cap': CAP,
                      'pin': PIN, 'dyz': DYZ, 'ny': NY, 'nz': NZ, 'k_s': K_S, 'src_ijk': SRC_IJK,
                      'prb_ijk': PRB_IJK, 'f0': F0, 'sigma_t': SIGMA_T, 't0': T0, 'n1': N1, 'n2': N2,
                      't_w': T_W, 'comb_hz': COMB.tolist(), 'checkpoint_every': CHECKPOINT_EVERY,
                      'p0': P0.tolist(), 'ladder_scale': LADDER_SCALE, 'dt0': DT0}}
    t = time.monotonic()
    try:
        sc = selfcheck()
        provenance['selfcheck'] = sc
        if not sc['all_pass']:
            raise RuntimeError(f'judge selfcheck failed: {sc}')
        provenance.update(ARMS[args.arm]())
    except Exception as exc:
        provenance['instrument_error'] = repr(exc)
        raise
    finally:
        provenance['elapsed_seconds'] = time.monotonic() - t
        out.write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
