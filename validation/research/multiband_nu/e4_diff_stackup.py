"""E4 — differentiable stackup prototype (research script, NO production change).

Pre-declaration (binding, every window frozen there before this file was
written): docs/design_notes/20260907_nu_exp4_diff_stackup_predeclaration.md.
This instrument is committed BEFORE its first measurement run.

Question: can a substrate thickness (h_thin) and layer permittivities
(eps_thin, eps_core) be design variables with correct gradients TODAY on
the committed NU solver, via (a) fixed-topology mesh stretching and (b) a
dual-cell fill-fraction eps on the interface nodes?

Map ``stackup(params) -> (dz_profile, eps_by_node)`` on the A1 stack of W7
(note 2.2): four thin cells ``h_thin / 4``, both cores absorb half the
change, the air run is fixed, the column stays 44 mm; node eps is the
dual-cell average. Materials, sources and probes are attached BY NODE
INDEX (coordinate rasterization has zero derivative — note 0 / 1c).

Observables (note 2.3 / 2.4): L1 the W7 AD1 probe energy at 120 steps;
L2 the narrowband DFT power at the analytic f_res (P_nom) plus the flank
asymmetry S from P(f_c +- Delta) at 8000 steps.

Falsifiers (note 3): E4-F1 / E4-F2 jax.grad vs central FD in the design
variable, 15 % with sign; E4-S sign(dS/dp) vs the transfer-matrix oracle;
E4-T FD+ vs FD- in h_thin (the four-way tie does not bite); E4-AD tracer
path; E4-M map invariants (--selfcheck).

Usage (note 4)::

    PYTHONPATH=. python -m validation.research.multiband_nu.e4_diff_stackup --selfcheck --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.e4_diff_stackup --smoke --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.e4_diff_stackup --arms l1 --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.e4_diff_stackup --arms l2 --out R
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box
from rfx.core.yee import MaterialArrays
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform

from . import w7_accuracy_ad as w7
from .analytic_dispersion import C0

# ---------------------------------------------------------------------------
# Fixture (note 2.1 / 2.2) — every length in metres
# ---------------------------------------------------------------------------
N_CORE, N_THIN, N_AIR = 20, 4, 20
T_CORE, H0, T_AIR = 14e-3, 2e-3, 14e-3
L_Z = 2 * T_CORE + H0 + T_AIR                             # 44 mm
A_X, B_Y = 30e-3, 3e-3
DXY = 0.5e-3
DOMAIN_XY = (A_X, B_Y)
PARAMS0 = (2e-3, 3.0, 4.3)                                # (h_thin, eps_thin, eps_core)
PARAM_NAMES = ("h_thin", "eps_thin", "eps_core")
K_IFACE = (N_CORE, N_CORE + N_THIN, 2 * N_CORE + N_THIN)  # 20, 24, 44
K_SRC = N_CORE + N_THIN                                   # 24
K_PRB_L1 = N_CORE                                         # 20
K_PRB_L2 = 2 * N_CORE + N_THIN                            # 44
N_CELLS = 2 * N_CORE + N_THIN + N_AIR                     # 64
KX2 = (np.pi / A_X) ** 2                                  # m = 1
# observables (note 2.3 / 2.4)
N1_STEPS = 120
N2_STEPS = 8000
CHECKPOINT_EVERY = 100
SIGMA_T = 200e-12
T0 = 5 * SIGMA_T
# FD steps (note 2.5)
FD_REL_L1 = {"h_thin": 1e-3, "eps_thin": 1e-2, "eps_core": 1e-2}
FD_REL_L2 = {"h_thin": 1e-3, "eps_thin": 1e-3, "eps_core": 1e-3}
FD_REL_L2_REPORTED = {"eps_thin": 1e-2, "eps_core": 1e-2}
FD_REL_CELL = 1e-3                                        # tie table, one cell at a time
# frozen windows (note 3)
TOL_REL = 0.15
TOL_KINK = 0.15
REF_FLOOR_QUANTA = 50.0
ORACLE_SIGNS = {"h_thin": +1, "eps_thin": -1, "eps_core": -1}
# selfcheck numbers (note 1a / 1b / 3 E4-M)
ORACLE_SLOPES = {"h_thin": 2.2275e10, "eps_thin": -2.3779e7, "eps_core": -8.3599e8}
ORACLE_SLOPE_REL = 1e-4
F_RES_REL = 1e-9
KOTTKE_COLUMN = {20: 2.825, 24: 2.825, 44: 2.65}
KOTTKE_TOL = 1e-6
MAP_LEN_TOL = 1e-9
MAP_JAC_REL = 1e-6
MAP_EPS_TOL = 1e-6
SELFCHECK_H = (1.8e-3, 2.0e-3, 2.2e-3)
DEFAULT_OUT = "validation/research/multiband_nu/results/e4_diff_stackup.json"
# smoke (note 4)
SMOKE_N1, SMOKE_N2 = 12, 200


# ---------------------------------------------------------------------------
# The map (note 2.2)
# ---------------------------------------------------------------------------
def stackup(params):
    """params = (h_thin, eps_thin, eps_core) -> (dz [64], eps_node [65]).

    Pure jnp, smooth in every entry, topology fixed. The node eps is the
    dual-cell average (the fill fraction of the dual cell centred on the
    node); the last node is the bounding / PEC node."""
    p = jnp.asarray(params)
    h, et, ec = p[0], p[1], p[2]
    d_thin = h / N_THIN
    d_core = (T_CORE - (h - H0) / 2.0) / N_CORE
    d_air = jnp.asarray(T_AIR / N_AIR, p.dtype)
    dz = jnp.concatenate([jnp.full((N_CORE,), d_core), jnp.full((N_THIN,), d_thin),
                          jnp.full((N_CORE,), d_core), jnp.full((N_AIR,), d_air)])
    eps_cell = jnp.concatenate([jnp.full((N_CORE,), ec), jnp.full((N_THIN,), et),
                                jnp.full((N_CORE,), ec), jnp.full((N_AIR,), 1.0, p.dtype)])
    return dz, dual_eps_from_cells(dz, eps_cell)


def dual_eps_from_cells(dz, eps_cell):
    """Node eps (len N + 1) from cell eps and cell sizes: dual-cell average
    at interior nodes, the end cells' eps at the two end nodes."""
    inner = (eps_cell[:-1] * dz[:-1] + eps_cell[1:] * dz[1:]) / (dz[:-1] + dz[1:])
    return jnp.concatenate([eps_cell[:1], inner, eps_cell[-1:]])


def cell_eps_pattern(params):
    p = jnp.asarray(params)
    return jnp.concatenate([jnp.full((N_CORE,), p[2]), jnp.full((N_THIN,), p[1]),
                            jnp.full((N_CORE,), p[2]), jnp.full((N_AIR,), 1.0, p.dtype)])


def declared_jacobian_h() -> np.ndarray:
    return np.asarray([-1.0 / (2 * N_CORE)] * N_CORE + [1.0 / N_THIN] * N_THIN
                      + [-1.0 / (2 * N_CORE)] * N_CORE + [0.0] * N_AIR)


def edges_of(h: float):
    return [0.0, 15e-3 - h / 2, 15e-3 + h / 2, 2 * T_CORE + H0, L_Z]


def dt_f64(h: float, dxy: float = DXY) -> float:
    return 0.99 / (C0 * np.sqrt(1 / dxy ** 2 + 1 / dxy ** 2 + 1 / (h / N_THIN) ** 2))


# ---------------------------------------------------------------------------
# Losses (note 2.3 / 2.4)
# ---------------------------------------------------------------------------
def _layout(domain_xy, dx):
    nx_cells = int(round(domain_xy[0] / dx))
    ny_cells = int(round(domain_xy[1] / dx))
    i1, i2 = nx_cells // 3, 2 * nx_cells // 3
    j = ny_cells // 2
    return i1, i2, j


def _materials(grid, eps_node):
    shape = grid.shape
    eps = jnp.broadcast_to(eps_node.astype(jnp.float32)[None, None, :], shape)
    return MaterialArrays(eps_r=eps, mu_r=jnp.ones(shape, jnp.float32),
                          sigma=jnp.zeros(shape, jnp.float32))


def _w5_waveform(n_steps: int):
    n = jnp.arange(n_steps, dtype=jnp.float32)
    return jnp.exp(-(((n - 15.0) / 5.0) ** 2)).astype(jnp.float32)


def _run(dz, eps_node, n_steps, waveform, k_prb, domain_xy, dx, checkpoint_every=None):
    grid = make_nonuniform_grid(domain_xy=domain_xy, dz_profile=dz, dx=dx, cpml_layers=0)
    i1, i2, j = _layout(domain_xy, dx)
    wf = waveform(n_steps, grid.dt)
    out = run_nonuniform(grid, _materials(grid, eps_node), n_steps,
                         sources=[(i1, j, K_SRC, "ey", wf), (i2, j, K_SRC, "ey", wf)],
                         probes=[(i1, j, k_prb, "ey")],
                         checkpoint_every=checkpoint_every)
    return out["time_series"][:, 0], grid.dt


def l1_from_dz_eps(dz, eps_node, n_steps=N1_STEPS, domain_xy=DOMAIN_XY, dx=DXY):
    ts, _ = _run(dz, eps_node, n_steps, lambda n, dt: _w5_waveform(n), K_PRB_L1, domain_xy, dx)
    return jnp.sum(ts ** 2)


def l1_factory(n_steps=N1_STEPS, domain_xy=DOMAIN_XY, dx=DXY):
    def loss(params):
        dz, eps_node = stackup(params)
        return l1_from_dz_eps(dz, eps_node, n_steps, domain_xy, dx)
    return loss


def l1_dz_factory(params0, n_steps=N1_STEPS, domain_xy=DOMAIN_XY, dx=DXY):
    """The same L1 written on the dz VECTOR: cell eps fixed by index from
    params0, node eps re-assembled from dz (note 2.5, the tie table)."""
    eps_cell = cell_eps_pattern(jnp.asarray(params0))

    def loss(dz):
        return l1_from_dz_eps(dz, dual_eps_from_cells(dz, eps_cell), n_steps, domain_xy, dx)
    return loss


def _drive(f_res: float):
    def waveform(n_steps, dt):
        t = jnp.arange(n_steps, dtype=jnp.float32) * dt
        return (jnp.exp(-(((t - T0) / SIGMA_T) ** 2)) * jnp.sin(2 * jnp.pi * f_res * t)).astype(jnp.float32)
    return waveform


def l2_factory(freqs, f_res: float, n_steps=N2_STEPS, domain_xy=DOMAIN_XY, dx=DXY,
               checkpoint_every=CHECKPOINT_EVERY):
    """(P(f_nom), P(f_up), P(f_dn)) from one run; DFT in-trace with the
    traced dt (note 2.4)."""
    freqs = jnp.asarray(np.asarray(freqs, np.float64), jnp.float32)

    def powers(params):
        dz, eps_node = stackup(params)
        ts, dt = _run(dz, eps_node, n_steps, _drive(f_res), K_PRB_L2, domain_xy, dx, checkpoint_every)
        t = jnp.arange(n_steps, dtype=jnp.float32) * dt
        phase = 2 * jnp.pi * freqs[:, None] * t[None, :]
        re = dt * jnp.sum(ts[None, :] * jnp.cos(phase), axis=1)
        im = dt * jnp.sum(ts[None, :] * jnp.sin(phase), axis=1)
        return re ** 2 + im ** 2
    return powers


def asym(p_up, p_dn):
    return (p_up - p_dn) / (p_up + p_dn)


def d_asym(p_up, p_dn, g_up, g_dn):
    """d S / d p by the quotient rule from the jacobian rows."""
    s = p_up + p_dn
    return ((g_up - g_dn) * s - (p_up - p_dn) * (g_up + g_dn)) / s ** 2


# ---------------------------------------------------------------------------
# Oracles (note 1a / 2.4)
# ---------------------------------------------------------------------------
def f_res(params) -> float:
    h, et, ec = (float(x) for x in params)
    return w7.root_near(lambda f: w7.lse_det(f, KX2, edges_of(h), [ec, et, ec, 1.0]),
                        w7.F_TRUE_DECLARED)


def oracle_slope(name: str, rel: float, params=PARAMS0) -> dict:
    i = PARAM_NAMES.index(name)
    p = np.asarray(params, np.float64)
    step = rel * p[i]
    pp, pm = p.copy(), p.copy()
    pp[i] += step
    pm[i] -= step
    fp, fm = f_res(pp), f_res(pm)
    return {"param": name, "rel_step": rel, "step": float(step), "f_plus": fp, "f_minus": fm,
            "slope": (fp - fm) / (2 * step), "moves_per_step_hz": (fp - fm) / 2}


def f_discrete(params, dxy: float = DXY) -> dict:
    """The discrete line on this mesh: W7's exact stratified model with the
    dual eps column, mu_x of the uniform x mesh, leap at dt(h)."""
    h = float(params[0])
    dz64 = _dz_f64(params)
    eps64 = _eps_f64(params)
    n_x = int(round(A_X / dxy))
    mu_x = w7.mu_1d(np.full(n_x, dxy), 1)
    lam = w7.discrete_lambda(dz64, eps64, mu_x)
    ft = f_res(params)
    et, ec = float(params[1]), float(params[2])
    f_cont = w7.root_near(lambda f: w7.lse_det(f, mu_x, edges_of(h), [ec, et, ec, 1.0]), ft)
    lam_t = (2 * np.pi * f_cont / C0) ** 2
    lam_p = float(lam[np.argmin(np.abs(lam - lam_t))])
    dt = dt_f64(h, dxy)
    return {"f_c": w7.leap(lam_p, dt), "f_res": ft, "f_cont_mux": f_cont, "mu_x": mu_x,
            "lambda": lam_p, "dt": dt}


def _dz_f64(params):
    h = float(params[0])
    d_thin = h / N_THIN
    d_core = (T_CORE - (h - H0) / 2.0) / N_CORE
    return np.asarray([d_core] * N_CORE + [d_thin] * N_THIN + [d_core] * N_CORE + [T_AIR / N_AIR] * N_AIR)


def _eps_f64(params):
    et, ec = float(params[1]), float(params[2])
    dz = _dz_f64(params)
    ce = np.asarray([ec] * N_CORE + [et] * N_THIN + [ec] * N_CORE + [1.0] * N_AIR)
    return _dual_np(dz, ce)


def _dual_np(dz, ce):
    inner = (ce[:-1] * dz[:-1] + ce[1:] * dz[1:]) / (dz[:-1] + dz[1:])
    return np.concatenate([ce[:1], inner, ce[-1:]])


def discrete_slope(name: str, rel: float, params=PARAMS0) -> dict:
    i = PARAM_NAMES.index(name)
    p = np.asarray(params, np.float64)
    step = rel * p[i]
    pp, pm = p.copy(), p.copy()
    pp[i] += step
    pm[i] -= step
    fp, fm = f_discrete(pp)["f_c"], f_discrete(pm)["f_c"]
    return {"param": name, "rel_step": rel, "step": float(step), "f_plus": fp, "f_minus": fm,
            "slope": (fp - fm) / (2 * step)}


def sinc2(x):
    return np.sinc(x) ** 2          # np.sinc(x) = sin(pi x)/(pi x)


def ds_ddelta(delta_hz: float, T: float, f_c: float, f_eval_c: float, d_line: float = 1e3) -> float:
    """d S / d(line offset) at the nominal from the rectangular-window
    sinc^2 model, S = (u - v)/(u + v), u = sinc^2((f_up - f_line) T),
    v = sinc^2((f_dn - f_line) T)."""
    def s_of(fl):
        u = sinc2((f_eval_c + delta_hz - fl) * T)
        v = sinc2((f_eval_c - delta_hz - fl) * T)
        return (u - v) / (u + v)
    return (s_of(f_c + d_line) - s_of(f_c - d_line)) / (2 * d_line)


# ---------------------------------------------------------------------------
# Section 1b / 1c reproductions (selfcheck)
# ---------------------------------------------------------------------------
def kottke_column() -> dict:
    from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform
    dz = _dz_f64(PARAMS0)
    g = make_nonuniform_grid(DOMAIN_XY, dz, DXY, cpml_layers=0)
    boxes = [(Box((0, 0, 0.0), (A_X, B_Y, 14e-3)), 4.3), (Box((0, 0, 14e-3), (A_X, B_Y, 16e-3)), 3.0),
             (Box((0, 0, 16e-3), (A_X, B_Y, 30e-3)), 4.3)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, ey, _ = compute_smoothed_eps_nonuniform(g, boxes, 1.0)
    col = np.asarray(ey[g.nx // 2, g.ny // 2, :], np.float64)
    dual = _eps_f64(PARAMS0)
    return {"kottke_ey": {str(k): float(col[k]) for k in KOTTKE_COLUMN},
            "dual": {str(k): float(dual[k]) for k in KOTTKE_COLUMN},
            "declared": {str(k): v for k, v in KOTTKE_COLUMN.items()},
            "max_abs_dev_from_declared": float(max(abs(col[k] - v) for k, v in KOTTKE_COLUMN.items()))}


def mask_grad_probe() -> dict:
    dz = jnp.asarray(_dz_f64(PARAMS0), jnp.float32)

    def f(dzv):
        cum = jnp.concatenate([jnp.zeros(1, dzv.dtype), jnp.cumsum(dzv)])
        m = Box((0, 0, 14e-3), (A_X, B_Y, 16e-3)).mask_on_coords(jnp.array([0.0]), jnp.array([0.0]), cum)
        return jnp.sum(jnp.where(m, 3.0, 4.3) * jnp.arange(m.shape[-1]))
    try:
        g = np.asarray(jax.grad(f)(dz), np.float64)
        return {"raised": False, "all_finite": bool(np.all(np.isfinite(g))), "max_abs_grad": float(np.max(np.abs(g)))}
    except Exception as exc:  # noqa: BLE001
        return {"raised": True, "error": repr(exc)}


# ---------------------------------------------------------------------------
# Selfcheck (note 3, E4-M)
# ---------------------------------------------------------------------------
def _rel(x, y):
    return abs(x - y) / max(abs(y), 1e-300)


def map_checks(hs=SELFCHECK_H) -> dict:
    """E4-M. The map is exact linear arithmetic; its 1e-9 m windows are
    float64-class statements (0.27 f32 ulp of the 44 mm column), so the
    gated rows are evaluated under the scoped x64 context (bring-up fix,
    see Results) and the default-dtype (float32, the solver's view) rows
    are recorded alongside as ``rows_f32``."""
    from tests._x64_compat import enable_x64
    rows_f32 = _map_rows(hs)
    with enable_x64():
        jax.clear_caches()
        ch = _map_checks_gated(hs)
    jax.clear_caches()
    ch["rows_f32"] = rows_f32
    ch["f32_length_err_max_m"] = float(max(r["length_err_m"] for r in rows_f32))
    ch["f32_iface_err_max_m"] = float(max(r["iface_err_m"] for r in rows_f32))
    ch["f32_ulp_of_column_m"] = float(np.spacing(np.float32(L_Z)))
    return ch


def _map_checks_gated(hs) -> dict:
    ch = {}
    jac = np.asarray(jax.jacfwd(lambda p: stackup(p)[0])(jnp.asarray(PARAMS0))[:, 0], np.float64)
    jd = declared_jacobian_h()
    ch["jacobian_h_worst_rel"] = float(max(_rel(a, b) if b != 0 else abs(a) for a, b in zip(jac, jd)))
    ch["jacobian_h_pass"] = ch["jacobian_h_worst_rel"] <= MAP_JAC_REL
    ch["jacobian_h_sum"] = float(jac.sum())
    rows = _map_rows(hs)
    ch["rows"] = rows
    ch["length_pass"] = all(r["length_err_m"] <= MAP_LEN_TOL for r in rows)
    ch["thin_equal_pass"] = all(r["thin_cells_bitwise_equal"] for r in rows)
    ch["iface_pass"] = all(r["iface_err_m"] <= MAP_LEN_TOL for r in rows)
    ch["eps_pass"] = all(r["eps_iface_err"] <= MAP_EPS_TOL for r in rows)
    ch["topology_pass"] = all(r["n_cells"] == N_CELLS and r["n_nodes"] == N_CELLS + 1 for r in rows)
    ch["all_pass"] = all(ch[k] for k in ("jacobian_h_pass", "length_pass", "thin_equal_pass",
                                          "iface_pass", "eps_pass", "topology_pass"))
    ch["dtype"] = str(jnp.asarray(PARAMS0).dtype)
    return ch


def _map_rows(hs) -> list[dict]:
    rows = []
    for h in hs:
        p = (h, PARAMS0[1], PARAMS0[2])
        dz, eps = stackup(jnp.asarray(p))
        dz = np.asarray(dz, np.float64)
        eps = np.asarray(eps, np.float64)
        zn = np.concatenate([[0.0], np.cumsum(dz)])
        thin = dz[N_CORE:N_CORE + N_THIN]
        e_hand = {20: (4.3 * dz[19] + 3.0 * dz[20]) / (dz[19] + dz[20]),
                  24: (3.0 * dz[23] + 4.3 * dz[24]) / (dz[23] + dz[24]),
                  44: (4.3 * dz[43] + 1.0 * dz[44]) / (dz[43] + dz[44])}
        want = (15e-3 - h / 2, 15e-3 + h / 2, 30e-3)
        rows.append({"h_thin": h, "n_cells": int(len(dz)), "n_nodes": int(len(eps)),
                     "length_err_m": float(abs(dz.sum() - L_Z)),
                     "thin_cells_bitwise_equal": bool(np.all(thin == thin[0])),
                     "iface_err_m": float(max(abs(zn[k] - w) for k, w in zip(K_IFACE, want))),
                     "eps_iface": {str(k): float(eps[k]) for k in K_IFACE},
                     "eps_iface_err": float(max(abs(eps[k] - e_hand[k]) for k in K_IFACE)),
                     "dz_min_is_thin": bool(np.isclose(dz.min(), thin[0]) and np.sum(dz == dz.min()) == N_THIN),
                     "dtype": str(np.asarray(stackup(jnp.asarray(p))[0]).dtype)})
    return rows


def selfcheck(verbose: bool = True) -> dict:
    t0 = time.time()
    sc = {"map": map_checks()}
    sc["oracle_w7"] = w7.oracle_selfcheck()
    sc["oracle_w7"]["all_pass"] = bool(sc["oracle_w7"]["i_pass"] and sc["oracle_w7"]["ip_pass"]
                                        and sc["oracle_w7"]["ipp_pass"]
                                        and sc["oracle_w7"]["f_true_matches_declared"])
    f0 = f_res(PARAMS0)
    sc["f_res_hz"] = f0
    sc["f_res_rel_to_w7_f_true"] = _rel(f0, w7.F_TRUE_DECLARED)
    sc["f_res_pass"] = sc["f_res_rel_to_w7_f_true"] <= F_RES_REL
    slopes = {}
    for name in PARAM_NAMES:
        row = oracle_slope(name, FD_REL_L1[name])
        row["declared"] = ORACLE_SLOPES[name]
        row["rel_dev"] = _rel(row["slope"], ORACLE_SLOPES[name])
        row["sign_declared"] = ORACLE_SIGNS[name]
        row["sign_pass"] = int(np.sign(row["slope"])) == ORACLE_SIGNS[name]
        slopes[name] = row
        slopes[name + "_1e-3"] = oracle_slope(name, 1e-3)
    sc["oracle_slopes"] = slopes
    sc["oracle_slopes_pass"] = all(slopes[n]["rel_dev"] <= ORACLE_SLOPE_REL and slopes[n]["sign_pass"]
                                   for n in PARAM_NAMES)
    disc = f_discrete(PARAMS0)
    disc["f_c_minus_f_res_hz"] = disc["f_c"] - disc["f_res"]
    disc["dt_grid_f32"] = float(make_nonuniform_grid(DOMAIN_XY, _dz_f64(PARAMS0), DXY, cpml_layers=0).dt)
    disc["delta_hz"] = 0.5 / (N2_STEPS * disc["dt"])
    disc["T_s"] = N2_STEPS * disc["dt"]
    disc["f_eval"] = {"nom": disc["f_res"], "up": disc["f_c"] + disc["delta_hz"], "dn": disc["f_c"] - disc["delta_hz"]}
    disc["slopes"] = {n: discrete_slope(n, 1e-3) for n in PARAM_NAMES}
    sc["discrete"] = disc
    sc["kottke"] = kottke_column()
    sc["kottke_pass"] = sc["kottke"]["max_abs_dev_from_declared"] <= KOTTKE_TOL
    sc["mask_grad"] = mask_grad_probe()
    sc["mask_grad_pass"] = (not sc["mask_grad"]["raised"]) and sc["mask_grad"]["max_abs_grad"] == 0.0
    sc["dtype_probe"] = {"loss_l1": str(jax.eval_shape(l1_factory(4), jnp.asarray(PARAMS0)).dtype),
                         "grad_l1": str(jax.eval_shape(jax.grad(l1_factory(4)), jnp.asarray(PARAMS0)).dtype)}
    sc["all_pass"] = bool(sc["map"]["all_pass"] and sc["oracle_w7"]["all_pass"] and sc["f_res_pass"]
                          and sc["oracle_slopes_pass"] and sc["kottke_pass"] and sc["mask_grad_pass"])
    sc["wallclock_s"] = time.time() - t0
    sc.update(w7.provenance())
    if verbose:
        print(f"selfcheck: map {sc['map']['all_pass']}  oracle(i/i') {sc['oracle_w7']['all_pass']}  "
              f"f_res {f0:.3f} Hz (rel {sc['f_res_rel_to_w7_f_true']:.1e})  slopes {sc['oracle_slopes_pass']}  "
              f"kottke {sc['kottke_pass']}  mask_grad {sc['mask_grad_pass']}  -> all_pass {sc['all_pass']}")
        print(f"  f_c (discrete) = {disc['f_c']:.3f} Hz, f_c - f_res = {disc['f_c_minus_f_res_hz']/1e6:.4f} MHz, "
              f"Delta = {disc['delta_hz']/1e6:.3f} MHz, T = {disc['T_s']*1e9:.3f} ns, dt = {disc['dt']:.5e}")
        for n in PARAM_NAMES:
            print(f"  d f_res / d {n} = {slopes[n]['slope']:+.5e} (oracle), d f_c / d {n} = "
                  f"{disc['slopes'][n]['slope']:+.5e} (discrete)")
        print(f"  {sc['wallclock_s']:.1f} s")
    return sc


# ---------------------------------------------------------------------------
# Gradient comparison (note 2.5 / 3)
# ---------------------------------------------------------------------------
def _ulp(x: float) -> float:
    return float(np.spacing(np.float32(abs(x))))


def _sgn(x) -> int:
    """Sign as an int; 0 for a non-finite value (a NaN is a recorded
    witness, never an exception in the judge — smoke bring-up fix)."""
    return int(np.sign(x)) if np.isfinite(x) else 0


def _fd_rows(loss_j, p0: np.ndarray, rel_steps: dict, names, n_out: int = 1) -> dict:
    """Central / one-sided FD of a (vector) loss in each design variable."""
    l0 = np.atleast_1d(np.asarray(loss_j(jnp.asarray(p0)), np.float64))
    rows = {}
    for name in names:
        i = PARAM_NAMES.index(name)
        step = rel_steps[name] * p0[i]
        pp, pm = p0.copy(), p0.copy()
        pp[i] += step
        pm[i] -= step
        lp = np.atleast_1d(np.asarray(loss_j(jnp.asarray(pp)), np.float64))
        lm = np.atleast_1d(np.asarray(loss_j(jnp.asarray(pm)), np.float64))
        rows[name] = {"rel_step": rel_steps[name], "step": float(step),
                      "loss_plus": lp.tolist(), "loss_minus": lm.tolist(),
                      "fd": ((lp - lm) / (2 * step)).tolist(),
                      "fd_plus": ((lp - l0) / step).tolist(),
                      "fd_minus": ((l0 - lm) / step).tolist(),
                      "quanta": [float(abs(a - b) / _ulp(c)) for a, b, c in zip(lp, lm, l0)]}
    return {"loss0": l0.tolist(), "rows": rows}


def judge_row(g_ad: float, fd: dict, k: int = 0, tol: float = TOL_REL) -> dict:
    g_fd, fp, fm, q = fd["fd"][k], fd["fd_plus"][k], fd["fd_minus"][k], fd["quanta"][k]
    resolved = q >= REF_FLOOR_QUANTA
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-300)
    sign_ok = bool(np.sign(g_ad) == np.sign(g_fd))
    kink = abs(fp - fm) / max(abs(g_fd), 1e-300)
    return {"g_ad": float(g_ad), "g_fd": float(g_fd), "fd_plus": float(fp), "fd_minus": float(fm),
            "quanta": float(q), "resolved": bool(resolved), "rel_err": float(rel),
            "sign_agreement": sign_ok, "kink": float(kink), "tol": tol,
            "verdict": ("INCONCLUSIVE" if not resolved else
                        ("FIRED" if (rel > tol or not sign_ok or not np.isfinite(g_ad)) else "HELD"))}


def measure_l1(smoke: bool = False) -> dict:
    t_unit = time.time()
    n_steps = SMOKE_N1 if smoke else N1_STEPS
    names = ("h_thin",) if smoke else PARAM_NAMES
    p0 = np.asarray(PARAMS0, np.float64)
    dz0, eps0 = stackup(jnp.asarray(PARAMS0))
    out = {"arm": "l1", "n_steps": n_steps, "smoke": bool(smoke), "params0": list(PARAMS0),
           "dz_m": np.asarray(dz0, np.float64).tolist(), "eps_node": np.asarray(eps0, np.float64).tolist(),
           "domain_xy": list(DOMAIN_XY), "dx": DXY, "layout_ij": list(_layout(DOMAIN_XY, DXY)),
           "k_src": K_SRC, "k_prb": K_PRB_L1, "fd_rel": dict(FD_REL_L1), "tol": TOL_REL,
           "ref_floor_quanta": REF_FLOOR_QUANTA}
    try:
        loss = l1_factory(n_steps)
        loss_j, grad_j = jax.jit(loss), jax.jit(jax.grad(loss))
        g_ad = np.asarray(grad_j(jnp.asarray(PARAMS0)), np.float64)
        out["g_ad"] = dict(zip(PARAM_NAMES, g_ad.tolist()))
        out["all_finite"] = bool(np.all(np.isfinite(g_ad)))
        fd = _fd_rows(loss_j, p0, FD_REL_L1, names)
        out["loss0"] = fd["loss0"][0]
        out["ulp_loss0"] = _ulp(fd["loss0"][0])
        out["rows"] = {n: judge_row(g_ad[PARAM_NAMES.index(n)], fd["rows"][n]) for n in names}
        out["dt_grid_f32"] = float(make_nonuniform_grid(DOMAIN_XY, np.asarray(dz0, np.float64), DXY, cpml_layers=0).dt)
        # E4-T on h_thin
        r = out["rows"]["h_thin"]
        out["E4_T"] = {"kink": r["kink"], "tol": TOL_KINK, "resolved": r["resolved"],
                       "verdict": "INCONCLUSIVE" if not r["resolved"] else ("FIRED" if r["kink"] > TOL_KINK else "HELD")}
        # tie table (note 2.5), not in smoke
        if not smoke:
            out["tie_table"] = tie_table(n_steps, g_ad[0])
        out["E4_F1"] = {n: out["rows"][n]["verdict"] for n in names}
        out["fired"] = any(v == "FIRED" for v in out["E4_F1"].values()) or out["E4_T"]["verdict"] == "FIRED"
    except Exception as exc:  # noqa: BLE001 — a tracer-path failure IS the fired witness (E4-AD)
        out["error"] = repr(exc)
        out["E4_AD"] = "FIRED"
        out["fired"] = True
    out["wallclock_s"] = time.time() - t_unit
    out.update(w7.provenance())
    return out


def tie_table(n_steps: int, g_h_direct: float) -> dict:
    """Per-cell d L1 / d dz at the nominal, one-sided FD on the four tied
    cells (one cell at a time), the equal-split model, and J^T g_dz."""
    loss = l1_dz_factory(PARAMS0, n_steps)
    loss_j, grad_j = jax.jit(loss), jax.jit(jax.grad(loss))
    dz0 = np.asarray(_dz_f64(PARAMS0))
    x0 = jnp.asarray(dz0)
    g_dz = np.asarray(grad_j(x0), np.float64)
    l0 = float(loss_j(x0))
    d32 = w7.f32_values(dz0)
    tied = np.nonzero(d32 == d32.min())[0]
    rows = []
    for k in tied:
        h = FD_REL_CELL * dz0[k]
        dp, dm = dz0.copy(), dz0.copy()
        dp[k] += h
        dm[k] -= h
        lp, lm = float(loss_j(jnp.asarray(dp))), float(loss_j(jnp.asarray(dm)))
        fp, fm = (lp - l0) / h, (l0 - lm) / h
        rows.append({"k": int(k), "d_m": float(dz0[k]), "g_ad": float(g_dz[k]), "fd_plus": fp, "fd_minus": fm,
                     "fd_central": (lp - lm) / (2 * h), "equal_split_model": fp + (fm - fp) / len(tied)})
    jt = float(np.dot(declared_jacobian_h(), g_dz))
    return {"n_tied": int(len(tied)), "tied_cells": tied.tolist(), "loss0": l0, "rows": rows,
            "g_dz": g_dz.tolist(), "jt_g_dz_h": jt, "g_h_direct": float(g_h_direct),
            "jt_vs_direct_rel": _rel(jt, g_h_direct)}


def measure_l2(smoke: bool = False, sc: dict | None = None) -> dict:
    t_unit = time.time()
    n_steps = SMOKE_N2 if smoke else N2_STEPS
    names = ("h_thin",) if smoke else PARAM_NAMES
    sc = selfcheck(verbose=False) if sc is None else sc
    disc = sc["discrete"]
    fr, fc = disc["f_res"], disc["f_c"]
    dt0 = disc["dt"]
    delta = 0.5 / (n_steps * dt0)
    freqs = np.asarray([fr, fc + delta, fc - delta])
    p0 = np.asarray(PARAMS0, np.float64)
    out = {"arm": "l2", "n_steps": n_steps, "smoke": bool(smoke), "params0": list(PARAMS0),
           "checkpoint_every": CHECKPOINT_EVERY, "f_res_hz": fr, "f_c_hz": fc, "delta_hz": delta,
           "f_eval_hz": {"nom": freqs[0], "up": freqs[1], "dn": freqs[2]}, "T_s": n_steps * dt0,
           "sigma_t": SIGMA_T, "t0": T0, "k_src": K_SRC, "k_prb": K_PRB_L2,
           "fd_rel": dict(FD_REL_L2), "fd_rel_reported": dict(FD_REL_L2_REPORTED), "tol": TOL_REL,
           "ref_floor_quanta": REF_FLOOR_QUANTA}
    try:
        powers = l2_factory(freqs, fr, n_steps)
        pow_j = jax.jit(powers)
        jac_j = jax.jit(jax.jacrev(powers))
        P0 = np.asarray(pow_j(jnp.asarray(PARAMS0)), np.float64)
        J = np.asarray(jac_j(jnp.asarray(PARAMS0)), np.float64)          # (3 freqs, 3 params)
        out["P0"] = {"nom": P0[0], "up": P0[1], "dn": P0[2]}
        out["S0"] = float(asym(P0[1], P0[2]))
        out["jac_ad"] = {n: {"nom": J[0, i], "up": J[1, i], "dn": J[2, i]} for i, n in enumerate(PARAM_NAMES)}
        out["all_finite"] = bool(np.all(np.isfinite(J)))
        gS_ad = {n: float(d_asym(P0[1], P0[2], J[1, i], J[2, i])) for i, n in enumerate(PARAM_NAMES)}
        out["dS_ad"] = gS_ad
        fd = _fd_rows(pow_j, p0, FD_REL_L2, names, n_out=3)
        out["ulp_P0"] = {"nom": _ulp(P0[0]), "up": _ulp(P0[1]), "dn": _ulp(P0[2])}
        rows, srows = {}, {}
        for n in names:
            i = PARAM_NAMES.index(n)
            r = fd["rows"][n]
            rows[n] = judge_row(J[0, i], r, k=0)
            lp, lm = np.asarray(r["loss_plus"]), np.asarray(r["loss_minus"])
            sp, sm = asym(lp[1], lp[2]), asym(lm[1], lm[2])
            step = r["step"]
            s_fd = (sp - sm) / (2 * step)
            q = float(abs(sp - sm) / _ulp(out["S0"])) if np.isfinite(out["S0"]) else float("nan")
            srows[n] = {"g_ad": gS_ad[n], "g_fd": float(s_fd), "s_plus": float(sp), "s_minus": float(sm),
                        "quanta": q, "resolved": bool(np.isfinite(q) and q >= REF_FLOOR_QUANTA),
                        "sign_ad": _sgn(gS_ad[n]), "sign_fd": _sgn(s_fd),
                        "sign_oracle": ORACLE_SIGNS[n],
                        "rel_err_ad_fd": float(abs(gS_ad[n] - s_fd) / max(abs(s_fd), 1e-300))}
            srows[n]["finite"] = bool(np.isfinite(gS_ad[n]) and np.isfinite(s_fd))
            srows[n]["verdict"] = ("FIRED" if not srows[n]["finite"] else
                                   "INCONCLUSIVE" if not srows[n]["resolved"] else
                                   ("HELD" if srows[n]["sign_ad"] == ORACLE_SIGNS[n] else "FIRED"))
            # reported: the up / dn rows themselves
            rows[n]["up"] = judge_row(J[1, i], r, k=1)
            rows[n]["dn"] = judge_row(J[2, i], r, k=2)
        out["rows"] = rows
        out["E4_F2"] = {n: rows[n]["verdict"] for n in names}
        out["S_rows"] = srows
        out["E4_S"] = {n: srows[n]["verdict"] for n in names}
        # sinc^2 / oracle prediction of dS/dp, discrete-model slope (reported)
        T = n_steps * dt0
        dsd = ds_ddelta(delta, T, fc, fc)
        pred = {}
        for n in PARAM_NAMES:
            o = sc["oracle_slopes"][n + "_1e-3"]["slope"]
            d = disc["slopes"][n]["slope"]
            pred[n] = {"dS_ddelta": dsd, "df_res_dp_oracle": o, "df_c_dp_discrete": d,
                       "pred_oracle": dsd * o, "pred_discrete": dsd * d,
                       "ad_over_pred_oracle": gS_ad[n] / (dsd * o) if dsd * o else float("nan"),
                       "ad_over_pred_discrete": gS_ad[n] / (dsd * d) if dsd * d else float("nan"),
                       "discrete_over_oracle": d / o}
        out["dS_prediction"] = pred
        out["literal_Pnom_sign_h_thin"] = {"sign_dPnom_dh_ad": _sgn(J[0, 0]),
                                           "sign_df_res_dh_oracle": ORACLE_SIGNS["h_thin"],
                                           "agree": _sgn(J[0, 0]) == ORACLE_SIGNS["h_thin"],
                                           "gated": False}
        out["E4_T"] = {"kink": rows["h_thin"]["kink"], "tol": TOL_KINK, "resolved": rows["h_thin"]["resolved"],
                       "verdict": ("INCONCLUSIVE" if not rows["h_thin"]["resolved"] else
                                   ("FIRED" if rows["h_thin"]["kink"] > TOL_KINK else "HELD"))}
        if not smoke:
            fdr = _fd_rows(pow_j, p0, FD_REL_L2_REPORTED, tuple(FD_REL_L2_REPORTED), n_out=3)
            rep = {}
            for n in FD_REL_L2_REPORTED:
                i = PARAM_NAMES.index(n)
                r = fdr["rows"][n]
                lp, lm = np.asarray(r["loss_plus"]), np.asarray(r["loss_minus"])
                rep[n] = {"nom": judge_row(J[0, i], r, k=0), "step": r["step"], "rel_step": r["rel_step"],
                          "S_fd": float((asym(lp[1], lp[2]) - asym(lm[1], lm[2])) / (2 * r["step"])),
                          "S_ad": gS_ad[n], "gated": False}
            out["reported_1e-2"] = rep
        out["fired"] = (any(v == "FIRED" for v in out["E4_F2"].values())
                        or any(v == "FIRED" for v in out["E4_S"].values())
                        or out["E4_T"]["verdict"] == "FIRED")
    except Exception as exc:  # noqa: BLE001 — E4-AD
        out["error"] = repr(exc)
        out["E4_AD"] = "FIRED"
        out["fired"] = True
    out["wallclock_s"] = time.time() - t_unit
    out.update(w7.provenance())
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def _save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, default=float)
    os.replace(tmp, path)


def _print_rows(r: dict) -> None:
    if "error" in r:
        print(f"  {r['arm']}: E4-AD FIRED — {r['error'][:300]}")
        return
    for n, row in r["rows"].items():
        print(f"  {r['arm']} {n:9s} g_ad {row['g_ad']:+.6e}  g_fd {row['g_fd']:+.6e}  rel {row['rel_err']:.3e}  "
              f"sign {row['sign_agreement']}  quanta {row['quanta']:.0f}  kink {row['kink']:.3e}  -> {row['verdict']}")
    if "S_rows" in r:
        for n, row in r["S_rows"].items():
            print(f"  {r['arm']} S {n:9s} dS_ad {row['g_ad']:+.6e}  dS_fd {row['g_fd']:+.6e}  "
                  f"signs ad/fd/oracle {row['sign_ad']:+d}/{row['sign_fd']:+d}/{row['sign_oracle']:+d}  "
                  f"quanta {row['quanta']:.0f}  -> {row['verdict']}")
    print(f"  {r['arm']} E4-T kink {r['E4_T']['kink']:.3e} -> {r['E4_T']['verdict']}   fired={r['fired']}   "
          f"wall {r['wallclock_s']:.1f} s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    print("rfx.__file__ =", __import__("rfx").__file__)
    data = _load(a.out)
    sc = selfcheck(verbose=True)
    data["selfcheck"] = sc
    _save(a.out, data)
    if not sc["all_pass"]:
        print("selfcheck FAILED — refusing to run any arm (E4-M)")
        return 2
    if a.smoke:
        data["smoke"] = {"l1": measure_l1(smoke=True), "l2": measure_l2(smoke=True, sc=sc)}
        _save(a.out, data)
        for k in ("l1", "l2"):
            _print_rows(data["smoke"][k])
    for arm in [x for x in a.arms.split(",") if x]:
        if arm == "l1":
            data["l1"] = measure_l1()
        elif arm == "l2":
            data["l2"] = measure_l2(sc=sc)
        else:
            raise SystemExit(f"unknown arm {arm}")
        _save(a.out, data)
        _print_rows(data[arm])
    return 0


if __name__ == "__main__":
    sys.exit(main())
