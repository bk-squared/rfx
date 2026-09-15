"""E2 — thin dielectric layer: resolve it, or average it?

Pre-declaration (binding, every window frozen there before this file was
written): docs/design_notes/20260907_nu_exp2_thin_layer_subpixel_predeclaration.md.
This instrument is committed BEFORE its first measurement run. It reuses
lane F's oracle, exact discrete model, harness, extraction and provenance
(``w7_accuracy_ad``) and adds one ingredient: the fill-fraction (subpixel)
cell eps of the S1 / S1b arms.

Arms (note section 0): S1, S1b, S0 on the uniform coarse mesh; R2 / R4 /
R8 builder bands with the dual rule; P4 the production column on the R4
mesh; every arm has a no-layer reference on the SAME mesh so the shift
error ``e_shift = (f_meas(layer) - f_meas(ref)) - (f_true(layer) - f_ref)``
(note 2.4) is defined from measured numbers only.

Usage (note section 4)::

    PYTHONPATH=. python -m validation.research.multiband_nu.e2_thin_layer_subpixel --selfcheck --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.e2_thin_layer_subpixel --ladders L1,L2,T --scales 2 --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.e2_thin_layer_subpixel --ladders L1,L2,T --scales 1 --out R
    PYTHONPATH=. python -m validation.research.multiband_nu.e2_thin_layer_subpixel --ladders L1 --scales 0.5 --arms s1,s1b,s0 --out R

``--out`` is merged per unit; a unit already in the JSON is skipped (one
attempt per unit); ``--force`` keeps the earlier row under
``<key>|superseded|<started_utc>``. ``--smoke`` shrinks the step count;
smoke rows are tagged and are not evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import jax.numpy as jnp

import rfx
from rfx import Box, Simulation
from rfx.nonuniform import make_band_profile, make_nonuniform_grid, run_nonuniform

from .analytic_dispersion import C0
from .harness import build_pec_fixture
from . import w7_accuracy_ad as w7

# ---------------------------------------------------------------------------
# Fixture (note 2.1 / 2.2) — every length in metres
# ---------------------------------------------------------------------------
Z1 = 15.4e-3                     # layer bottom face (S1 / S0 / R / P4) or centre (S1b)
Z_AIR = 29.4e-3
L_Z = 43.4e-3
A_X, B_Y = 30e-3, 3e-3
EPS_C, EPS_T = 4.3, 3.0
DC0 = 0.7e-3                     # coarse cell at s = 1
DXY0 = 0.25e-3                   # transverse cell at s = 1
R_CAP = 1.4
Z_SRC, Z_PRB = 35.0e-3, 32.2e-3
SCALES = (0.5, 1.0, 2.0)
THICK_UM = (175, 350, 700)
T_TOTAL, T_TRUNC = w7.T_TOTAL, w7.T_TRUNC
SIGMA_T = w7.SIGMA_T
E_FLOOR_HZ = w7.E_FLOOR_HZ                    # 0.3 MHz
INVARIANCE_HZ = w7.INVARIANCE_HZ              # 0.1 MHz
G3_MODEL_RESIDUAL_HZ = w7.G3_MODEL_RESIDUAL_HZ  # 0.15 MHz
ORACLE_REL = w7.ORACLE_REL
SELFCHECK_REL = w7.SELFCHECK_REL
CELL_TOL = w7.CELL_TOL
H1 = DC0
# frozen windows (note 3)
P_S1_WINDOW = (1.8, 2.2)
RHO_MAX = 2.0
LAW_REL = 0.25
LAW_ABS_HZ = 0.3e6
S1B_RATIO_MAX = 1.0
# declared numbers the selfcheck must reproduce (note 1b / 2.1 / 2.5)
K_PERT_DECLARED = 17.6079e6 / 1e-6           # Hz / m^2  (17.608 MHz / mm^2)
F_REF_DECLARED = 10_680_141_503.244
F_TRUE_DECLARED = {                          # (t_um, centred): Hz
    (175, False): 10_686_187_469.788, (350, False): 10_693_303_832.300, (700, False): 10_710_740_086.538,
    (175, True): 10_685_646_135.693, (350, True): 10_691_137_566.563, (700, True): 10_702_154_770.606,
}
MODEL_TABLE_MHZ = {                          # "arm|s|t": (err, err_ref, e_shift)   note 2.5
    "s1|0.5|175": (-8.0883, -8.6672, +0.5789), "s1b|0.5|175": (-8.6048, -8.6672, +0.0624),
    "s0|0.5|175": (-1.4893, -8.6672, +7.1779), "r2|0.5|175": (-8.1993, -8.2219, +0.0226),
    "r4|0.5|175": (-8.0481, -8.0690, +0.0209), "r8|0.5|175": (-8.1666, -8.1852, +0.0186),
    "p4|0.5|175": (+7.6871, +7.5191, +0.1680),
    "s1|1|175": (-33.0280, -34.7443, +1.7162), "s1b|1|175": (-34.5104, -34.7443, +0.2339),
    "r4|1|175": (-29.7267, -29.7712, +0.0445),
    "s1|1|350": (-32.3940, -34.7443, +2.3503), "s1b|1|350": (-34.2647, -34.7443, +0.4796),
    "s0|1|350": (-16.9394, -34.7443, +17.8049), "r2|1|350": (-31.2806, -31.4150, +0.1344),
    "r4|1|350": (-30.0489, -30.1509, +0.1020), "r8|1|350": (-31.1871, -31.2477, +0.0606),
    "p4|1|350": (-61.0682, -60.8899, -0.1784),
    "s1|1|700": (-34.3756, -34.7443, +0.3686), "s1b|1|700": (-33.8173, -34.7443, +0.9270),
    "r4|1|700": (-32.5163, -32.7204, +0.2040),
    "s1|2|175": (-136.5783, -140.1520, +3.5737), "s1b|2|175": (-139.3686, -140.1520, +0.7834),
    "r4|2|175": (-104.2638, -104.1469, -0.1169),
    "s1|2|350": (-134.0393, -140.1520, +6.1127),      # the sixth F3 law unit (note Results addendum)
    "s1|2|700": (-132.0695, -140.1520, +8.0825), "s1b|2|700": (-136.9034, -140.1520, +3.2486),
    "s0|2|700": (-170.7506, -140.1520, -30.5986), "r2|2|700": (-107.9900, -108.3166, +0.3266),
    "r4|2|700": (-98.5541, -98.5536, -0.0005), "r8|2|700": (-115.0258, -113.8394, -1.1865),
    "p4|2|700": (-47.2632, -39.7986, -7.4646),
}
MODEL_ORDERS = {"s1_L1": 1.902, "s1b_L1": 2.851, "s0_L1": 1.046, "s1_L2": 1.313,
                "s1_total_L1": 2.015, "r4_total_L1": 1.807}
MODEL_RHO_SCALE = {0.5: 1.005, 1.0: 1.078, 2.0: 1.340}
MODEL_RHO_FIT_H1 = 1.132
MODEL_RHO_T = {175: 1.111, 350: 1.078, 700: 1.057}
MODEL_S1B_RATIO = {0.5: 0.108, 1.0: 0.204, 2.0: 0.402}
MODEL_LAW_DEV_PCT = {"s1|0.5|175": +7.4, "s1|1|175": +6.1, "s1|1|350": +9.0,
                     "s1|2|175": -5.3, "s1|2|350": -5.5, "s1|2|700": -6.3}
P4_INTERFACE_TABLE = {0.5: (3.0, 4.3, 1.0), 1.0: (3.0, 4.3, 4.3), 2.0: (4.3, 4.3, 1.0)}   # note 2.3
MESH_NZ = {("u", 0.5): 124, ("u", 1.0): 62, ("u", 2.0): 31,
           ("r2", 0.5, 175): 130, ("r4", 0.5, 175): 136, ("r8", 0.5, 175): 143,
           ("r4", 1.0, 175): 78, ("r2", 1.0, 350): 68, ("r4", 1.0, 350): 74, ("r8", 1.0, 350): 81,
           ("r4", 1.0, 700): 69, ("r4", 2.0, 175): 51,
           ("r2", 2.0, 700): 37, ("r4", 2.0, 700): 43, ("r8", 2.0, 700): 50}
DEFAULT_OUT = "validation/research/multiband_nu/results/e2_thin_layer_subpixel.json"
LAYER_ARMS = ("s1", "s1b", "s0", "r2", "r4", "r8", "p4")
MESH_OF = {"s1": "u", "s1b": "u", "s0": "u", "r2": "r2", "r4": "r4", "r8": "r8", "p4": "p4"}
N_CELLS = {"r2": 2, "r4": 4, "r8": 8, "p4": 4}
KX2 = (np.pi / A_X) ** 2


# ---------------------------------------------------------------------------
# Stacks and oracle
# ---------------------------------------------------------------------------
def stack(t: float, centred: bool = False):
    """(edges, eps) of the layer stack; t <= 0 is the no-layer reference."""
    if t <= 0:
        return [0.0, Z_AIR, L_Z], [EPS_C, 1.0]
    z_lo = Z1 - t / 2 if centred else Z1
    return [0.0, z_lo, z_lo + t, Z_AIR, L_Z], [EPS_C, EPS_T, EPS_C, 1.0]


def f_true_of(t_um: int, centred: bool) -> float:
    edges, eps = stack(t_um * 1e-6, centred)
    guess = F_REF_DECLARED if t_um <= 0 else F_TRUE_DECLARED[(t_um, centred)]
    return w7.root_near(lambda f: w7.lse_det(f, KX2, edges, eps), guess)


def f_ref_true() -> float:
    return f_true_of(0, False)


def oracle_selfcheck() -> dict:
    """(i), (i'), (i'') of note 2.1 on the E2 box, plus the declared frequencies."""
    out = {"i_worst_rel": 0.0, "i_roots": 0, "ip_worst_rel": 0.0, "ip_roots": 0,
           "ipp_worst_rel": 0.0, "ipp_roots": 0}
    edges2 = [0.0, Z_AIR, L_Z]
    for a in (A_X, 2 * A_X):
        for m in (1, 5):
            kx2 = (m * np.pi / a) ** 2
            for r in w7.family_roots(w7.lse_det, kx2, edges2, [1.0, 1.0], 3e9, 16e9):
                p = int(round(np.sqrt(max((2 * np.pi * r / C0) ** 2 - kx2, 0.0)) * L_Z / np.pi))
                f_cf = w7.closed_form(m, p, a, L_Z)
                out["i_worst_rel"] = max(out["i_worst_rel"], abs(r - f_cf) / f_cf)
                out["i_roots"] += 1
        kx2 = (np.pi / a) ** 2
        for r in w7.family_roots(w7.lse_det, kx2, edges2, [4.3, 4.3], 2e9, 8e9):
            p = int(round(np.sqrt(max(4.3 * (2 * np.pi * r / C0) ** 2 - kx2, 0.0)) * L_Z / np.pi))
            f_cf = w7.closed_form(1, p, a, L_Z, 4.3)
            out["ip_worst_rel"] = max(out["ip_worst_rel"], abs(r - f_cf) / f_cf)
            out["ip_roots"] += 1
    for r in w7.family_roots(w7.lsm_det, KX2, edges2, [1.0, 1.0], 3e9, 16e9):
        p = int(round(np.sqrt(max((2 * np.pi * r / C0) ** 2 - KX2, 0.0)) * L_Z / np.pi))
        f_cf = w7.closed_form(1, p, A_X, L_Z)
        out["ipp_worst_rel"] = max(out["ipp_worst_rel"], abs(r - f_cf) / f_cf)
        out["ipp_roots"] += 1
    out["i_pass"] = bool(out["i_worst_rel"] <= ORACLE_REL and out["i_roots"] >= 10)
    out["ip_pass"] = bool(out["ip_worst_rel"] <= ORACLE_REL and out["ip_roots"] >= 3)
    out["ipp_pass"] = bool(out["ipp_worst_rel"] <= ORACLE_REL and out["ipp_roots"] >= 5)
    f_ref = f_ref_true()
    out["f_ref_hz"] = f_ref
    out["f_ref_matches_declared"] = bool(abs(f_ref - F_REF_DECLARED) <= 1e-2)
    out["f_true_hz"] = {}
    ok = True
    for (t_um, c), fd in F_TRUE_DECLARED.items():
        f = f_true_of(t_um, c)
        out["f_true_hz"][f"{t_um}|{'centred' if c else 'face'}"] = f
        ok = ok and abs(f - fd) <= 1e-2
    out["f_true_match_declared"] = bool(ok)
    # p = 5 half waves of the reference mode, neighbours, other families (t = 350 stack)
    zg = np.linspace(0, L_Z, 8001)
    pg = w7.lse_field(f_ref, KX2, *stack(0), zg)
    out["p_half_waves"] = int(np.sum(np.diff(np.sign(pg[1:-1])) != 0)) + 1
    psi = w7.lse_field(f_ref, KX2, *stack(0), np.array([14e-3, Z1, Z_PRB, Z_SRC]))
    out["psi_at_14_15p4_prb_src"] = psi.tolist()
    e350, p350 = stack(350e-6)
    f350 = f_true_of(350, False)
    roots = w7.lse_roots(KX2, e350, p350, 3e9, 13e9)
    others = [r for r in roots if abs(r - f350) > 1.0]
    out["neighbours_t350_hz"] = [max(r for r in others if r < f350), min(r for r in others if r > f350)]
    rows = []
    for m in (2, 3, 4, 6):
        amp = float(np.sin(m * np.pi / 3) + np.sin(2 * m * np.pi / 3))
        for r in w7.family_roots(w7.lse_det, (m * np.pi / A_X) ** 2, e350, p350, 9e9, 12.5e9):
            rows.append({"family": "Ey", "m": m, "f_hz": float(r), "rel_to_f_true": float((r - f350) / f350),
                         "source_pair_amplitude": amp})
    for m in range(0, 5):
        for r in w7.family_roots(w7.lsm_det, (m * np.pi / A_X) ** 2, e350, p350, 9e9, 12.5e9):
            rows.append({"family": "Hy", "m": m, "f_hz": float(r), "rel_to_f_true": float((r - f350) / f350),
                         "source_pair_amplitude": 0.0})
    rows.sort(key=lambda d: abs(d["rel_to_f_true"]))
    out["other_families_t350"] = rows
    return out


def k_pert() -> dict:
    """K_pert = f_ref |Delta| (psi^2)'(Z1) / (4 N) from the oracle field (note 1b)."""
    f_ref = f_ref_true()
    z = np.linspace(0, L_Z, 200001)
    psi = w7.lse_field(f_ref, KX2, *stack(0), z)
    eps_z = np.where(z < Z_AIR, EPS_C, 1.0)
    N = float(np.trapezoid(eps_z * psi ** 2, z))
    dpsi2 = np.gradient(psi ** 2, z)
    i1 = int(np.argmin(np.abs(z - Z1)))
    K = f_ref * abs(EPS_T - EPS_C) * dpsi2[i1] / (4 * N)
    return {"K_pert_hz_per_m2": float(K), "K_pert_mhz_per_mm2": float(K / 1e6 * 1e-6),
            "N_m": N, "dpsi2_dz_at_z1_per_m": float(dpsi2[i1]), "psi_at_z1": float(psi[i1]),
            "staircase_coeff_mhz_per_mm": float(f_ref * abs(EPS_T - EPS_C) * psi[i1] ** 2 / (2 * N) / 1e6 * 1e-3),
            "matches_declared": bool(abs(K - K_PERT_DECLARED) <= 1e-3 * K_PERT_DECLARED)}


# ---------------------------------------------------------------------------
# Meshes and columns (note 2.3)
# ---------------------------------------------------------------------------
def mesh_u(s: float) -> np.ndarray:
    dc = DC0 * s
    n = int(round(L_Z / dc))
    assert abs(n * dc - L_Z) < CELL_TOL
    return np.full(n, dc, np.float64)


def mesh_r(arm: str, s: float, t: float) -> np.ndarray:
    dc = DC0 * s
    return np.asarray(make_band_profile([0.0, Z1, Z1 + t, Z_AIR, L_Z], [dc, t / N_CELLS[arm], dc, dc],
                                        protected=[False, True, False, False], max_ratio=R_CAP), np.float64)


def mesh_of(mesh: str, s: float, t: float) -> np.ndarray:
    return mesh_u(s) if mesh == "u" else mesh_r(mesh, s, t)


def cell_eps_fill(prof, edges, eps) -> np.ndarray:
    """eps of every CELL by exact volume (length) fraction — the subpixel rule."""
    zn = np.concatenate([[0.0], np.cumsum(prof)])
    out = np.empty(len(prof))
    for k in range(len(prof)):
        lo, hi = zn[k], zn[k + 1]
        acc = 0.0
        for i in range(len(eps)):
            acc += eps[i] * max(0.0, min(hi, edges[i + 1]) - max(lo, edges[i]))
        out[k] = acc / (hi - lo)
    return out


def dual_from_cells(prof, ce) -> np.ndarray:
    """Dual-cell node average of a cell-eps vector (lane F's rule); end nodes
    take their cell (PEC for Ey anyway). Length N + 1."""
    n = len(prof)
    out = np.ones(n + 1)
    out[0], out[n] = ce[0], ce[-1]
    for k in range(1, n):
        out[k] = (ce[k - 1] * prof[k - 1] + ce[k] * prof[k]) / (prof[k - 1] + prof[k])
    return out


def production_column(prof, s: float, edges, eps, f_guess: float) -> dict:
    """The eps column ``Simulation._assemble_materials_nu`` assembles for this
    stack on this profile (lane F ``a1_production_column`` generalised): boxes
    for every non-air layer, column at (nx//2, ny//2), float32 values."""
    dx = DXY0 * s
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=2 * f_guess, domain=(A_X, B_Y, L_Z), boundary="pec", dx=dx,
                         dz_profile=np.asarray(prof, np.float64))
        sim.add_material("e2_core", eps_r=EPS_C)
        sim.add_material("e2_thin", eps_r=EPS_T)
        for i in range(len(eps)):
            if eps[i] == 1.0:
                continue
            sim.add(Box((0.0, 0.0, edges[i]), (A_X, B_Y, edges[i + 1])),
                    material=("e2_core" if eps[i] == EPS_C else "e2_thin"))
        grid = sim._build_nonuniform_grid()
        mats = sim._assemble_materials_nu(grid)[0]
    e = np.asarray(mats.eps_r, np.float64)
    col = e[grid.nx // 2, grid.ny // 2, :]
    zn = np.concatenate([[0.0], np.cumsum(np.asarray(prof, np.float64))])
    table = {}
    for z in edges[1:-1]:
        k = int(np.argmin(np.abs(zn - z)))
        table[f"{z*1e3:.3f}mm"] = {"k": k, "node_coord_repr": repr(float(zn[k])), "eps": float(col[k]),
                                   "node_err_m": float(abs(zn[k] - z))}
    return {"column": col.tolist(), "interface_table": table,
            "interior_transversely_uniform": bool(np.all(e[:-1, :-1, :] == col[None, None, :])),
            "warnings": sorted({str(x.message)[:200] for x in w})}


def column_for(arm: str, prof, edges, eps, s: float, f_guess: float) -> tuple[np.ndarray, dict]:
    """The node eps column of a unit and how it was assembled."""
    if arm in ("s1", "s1b"):
        ce = cell_eps_fill(prof, edges, eps)
        col = dual_from_cells(prof, ce)
        info = {"rule": "subpixel", "column_source": "fill-fraction cell eps, then dual-cell node average, float32",
                "cell_eps_nonpure": {int(k): float(v) for k, v in enumerate(ce) if v not in (EPS_C, EPS_T, 1.0)}}
    elif arm in ("s0", "r2", "r4", "r8", "u"):
        ce = w7.cell_eps(prof, edges, eps)
        col = dual_from_cells(prof, ce)
        info = {"rule": "dual", "column_source": "cell-midpoint eps (half-open), then dual-cell node average, float32"}
    elif arm == "p4":
        pc = production_column(prof, s, edges, eps, f_guess)
        col = np.asarray(pc["column"], np.float64)
        info = {"rule": "production", "column_source": "Simulation._assemble_materials_nu column at (nx//2, ny//2)",
                "interface_table": pc["interface_table"],
                "interior_transversely_uniform": pc["interior_transversely_uniform"],
                "assembly_warnings": pc["warnings"]}
    else:
        raise ValueError(arm)
    return w7.f32_values(col), info


# ---------------------------------------------------------------------------
# Exact discrete model (lane F 3.1 with the E2 stack)
# ---------------------------------------------------------------------------
def model(prof, col, dx: float, dt: float, edges, eps, f_true: float) -> dict:
    n_x = int(round(A_X / dx))
    mu_x = w7.mu_1d(np.full(n_x, dx), 1)
    lam = w7.discrete_lambda(prof, col, mu_x)
    f_cont_mux = w7.root_near(lambda f: w7.lse_det(f, mu_x, edges, eps), f_true)
    lam_target = (2 * np.pi * f_cont_mux / C0) ** 2
    lam_p = float(lam[np.argmin(np.abs(lam - lam_target))])
    f_full = w7.leap(lam_p, dt)
    f_no_z = w7.leap(lam_target, dt)
    f_no_xz = w7.leap((2 * np.pi * f_true / C0) ** 2, dt)
    e_z, e_x, e_t = f_full - f_no_z, f_no_z - f_no_xz, f_no_xz - f_true
    den = abs(e_z) + abs(e_x) + abs(e_t)
    return {"f_model": f_full, "e_z": e_z, "e_x": e_x, "e_t": e_t, "e_total_model": f_full - f_true,
            "z_fraction": abs(e_z) / den if den else float("nan"), "mu_x": mu_x, "lambda": lam_p, "dt": float(dt)}


# ---------------------------------------------------------------------------
# The declared unit set (note 2.5)
# ---------------------------------------------------------------------------
def unit_key(arm: str, s: float, t_um: int, layer: bool) -> str:
    if layer:
        return f"{arm}|{s:g}|{t_um}"
    mesh = MESH_OF[arm]
    return f"ref|u|{s:g}" if mesh == "u" else f"ref|{mesh}|{s:g}|{t_um}"


def unit_spec(arm: str, s: float, t_um: int, layer: bool) -> dict:
    mesh = MESH_OF[arm]
    centred = (arm == "s1b")
    if layer:
        edges, eps = stack(t_um * 1e-6, centred)
    else:
        edges, eps = stack(0)
    return {"key": unit_key(arm, s, t_um, layer), "arm": arm if layer else mesh, "mesh": mesh,
            "scale": s, "t_um": t_um if mesh != "u" or layer else 0, "t_mesh_um": t_um,
            "layer": layer, "centred": centred, "edges": edges, "eps": eps,
            "column_arm": (arm if layer else ("p4" if mesh == "p4" else "u")),
            "ref_key": (unit_key(arm, s, t_um, False) if layer else None)}


def declared_units(ladders=("L1", "L2", "T")) -> list[dict]:
    specs: dict[str, dict] = {}

    def add(arm, s, t_um):
        for layer in (True, False):
            u = unit_spec(arm, s, t_um, layer)
            specs.setdefault(u["key"], u)

    if "L1" in ladders:
        for s in SCALES:
            for arm in LAYER_ARMS:
                add(arm, s, int(round(350 * s)))
    if "L2" in ladders:
        for s in (1.0, 2.0):
            for arm in ("s1", "s1b", "r4"):
                add(arm, s, 175)
        for arm in ("s1", "s1b", "r4"):
            add(arm, 0.5, 175)
        add("s1", 2.0, 350)          # sixth S1 law unit (t < h), note Results addendum
    if "T" in ladders:
        for t_um in THICK_UM:
            for arm in ("s1", "s1b", "r4"):
                add(arm, 1.0, t_um)
    return list(specs.values())


def law_units() -> list[str]:
    """The six S1 units with t < h (note 3, E2-F3)."""
    return [k for k in MODEL_TABLE_MHZ if k.startswith("s1|")
            and int(k.split("|")[2]) * 1e-6 < DC0 * float(k.split("|")[1]) - 1e-15]


# ---------------------------------------------------------------------------
# One unit
# ---------------------------------------------------------------------------
def measure_unit(spec: dict, smoke: bool = False) -> dict:
    t_unit = time.time()
    s = spec["scale"]
    dx = DXY0 * s
    t_mesh = spec["t_mesh_um"] * 1e-6
    prof = mesh_of(spec["mesh"], s, t_mesh)
    rep = w7.profile_report(prof, spec["edges"])
    f_true = f_true_of(spec["t_um"] if spec["layer"] else 0, spec["centred"])
    col, column_info = column_for(spec["column_arm"], prof, spec["edges"], spec["eps"], s, f_true)
    zn = np.concatenate([[0.0], np.cumsum(prof)])
    planes = [Z_SRC, Z_PRB, Z_AIR] + ([spec["edges"][1]] if spec["mesh"] == "u" and spec["arm"] != "s1b" else [])
    if spec["mesh"] != "u":
        planes += list(spec["edges"][1:-1])
    node_err = float(max(np.min(np.abs(zn - z)) for z in planes))
    assert node_err <= CELL_TOL, (spec["key"], node_err)
    iface = {f"{z*1e3:.3f}mm": float(col[int(np.argmin(np.abs(zn - z)))]) for z in spec["edges"][1:-1]}
    grid, mats = build_pec_fixture(prof, (A_X, B_Y), dx)
    assert len(col) == grid.nz, (len(col), grid.nz)
    eps3 = np.broadcast_to(col[None, None, :], grid.shape)
    mats = mats._replace(eps_r=jnp.asarray(eps3, jnp.float32))
    dt = float(grid.dt)
    n_steps = 4000 if smoke else int(round(T_TOTAL / dt))
    t = np.arange(n_steps) * dt
    t0 = 5 * SIGMA_T
    wf = (np.exp(-((t - t0) / SIGMA_T) ** 2 / 2.0) * np.sin(2 * np.pi * f_true * (t - t0))).astype(np.float32)
    k_src = int(np.argmin(np.abs(zn - Z_SRC)))
    k_prb = int(np.argmin(np.abs(zn - Z_PRB)))
    i_a = int(round(A_X / 3 / dx))
    i_b = int(round(2 * A_X / 3 / dx))
    assert abs(i_a * dx - A_X / 3) < CELL_TOL and abs(i_b * dx - 2 * A_X / 3) < CELL_TOL
    j_c = grid.ny // 2
    t_run = time.time()
    out = run_nonuniform(grid, mats, n_steps,
                         sources=[(i_a, j_c, k_src, "ey", wf), (i_b, j_c, k_src, "ey", wf)],
                         probes=[(i_a, j_c, k_prb, "ey")])
    ts = np.asarray(out["time_series"][:, 0], np.float64)
    wall_run = time.time() - t_run
    n_skip = int((2 * t0) / dt)
    f_meas, lines, valid = w7.extract_a1(ts, dt, n_skip, f_true)
    n_trunc = min(int(round(T_TRUNC / dt)), n_steps)
    f_meas_10, _, valid_10 = w7.extract_a1(ts[:n_trunc], dt, n_skip, f_true)
    invariance = abs(f_meas - f_meas_10) if (valid and valid_10) else float("nan")
    m = model(prof, col, dx, dt, spec["edges"], spec["eps"], f_true)
    resid = (f_meas - m["f_model"]) if valid else float("nan")
    row = {
        "key": spec["key"], "arm": spec["arm"], "mesh": spec["mesh"], "scale": s, "t_um": spec["t_um"],
        "t_mesh_um": spec["t_mesh_um"], "layer": spec["layer"], "centred": spec["centred"],
        "ref_key": spec["ref_key"], "smoke": bool(smoke),
        "edges_m": list(spec["edges"]), "eps_layers": list(spec["eps"]),
        "profile_m": prof.tolist(), "nz": len(prof),
        "profile": {k: v for k, v in rep.items() if k != "cells_m"},
        "eps_column": col.tolist(), "interface_eps": iface, "node_err_m": node_err, **column_info,
        "dx": dx, "dt": dt, "n_steps": n_steps, "t_total_s": n_steps * dt,
        "cells": int(grid.nx * grid.ny * grid.nz), "grid_shape": list(grid.shape),
        "k_src": k_src, "k_prb": k_prb, "i_a": i_a, "i_b": i_b, "j_c": j_c,
        "n_skip": n_skip, "harminv_lines": lines,
        "f_true": f_true, "f_meas": f_meas, "valid": valid,
        "err_hz": (f_meas - f_true) if valid else float("nan"),
        "abs_err_hz": abs(f_meas - f_true) if valid else float("nan"),
        "f_meas_10ns": f_meas_10, "valid_10ns": valid_10,
        "invariance_hz": invariance,
        "invariance_pass": bool(np.isfinite(invariance) and invariance <= INVARIANCE_HZ),
        "f_model": m["f_model"], "e_z": m["e_z"], "e_x": m["e_x"], "e_t": m["e_t"],
        "e_total_model": m["e_total_model"], "z_fraction": m["z_fraction"], "mu_x": m["mu_x"],
        "model_residual_hz": resid,
        "g3_pass": bool(np.isfinite(resid) and abs(resid) <= G3_MODEL_RESIDUAL_HZ),
        "wallclock_run_s": wall_run, "wallclock_s": time.time() - t_unit,
        **w7.provenance(),
    }
    return row


# ---------------------------------------------------------------------------
# Judge (note 3), re-evaluable on any partial set
# ---------------------------------------------------------------------------
def _conclusive(r) -> bool:
    return bool(r is not None and not r.get("smoke") and r["valid"] and r["invariance_pass"])


def e_shift(units: dict, key: str, f_ref: float) -> float | None:
    r = units.get(key)
    if not _conclusive(r):
        return None
    ref = units.get(r["ref_key"])
    if not _conclusive(ref):
        return None
    return float((r["f_meas"] - ref["f_meas"]) - (r["f_true"] - f_ref))


def e_shift_model(units: dict, key: str, f_ref: float) -> float | None:
    r = units.get(key)
    ref = units.get(r["ref_key"]) if r else None
    if r is None or ref is None:
        return None
    return float((r["f_model"] - ref["f_model"]) - (r["f_true"] - f_ref))


def judge(units: dict, K: float | None = None, f_ref: float | None = None) -> dict:
    units = {k: v for k, v in units.items() if not v.get("smoke")}
    f_ref = f_ref_true() if f_ref is None else f_ref
    K = k_pert()["K_pert_hz_per_m2"] if K is None else K
    out = {"n_units": len(units), "units_present": sorted(units), "K_pert_hz_per_m2": K, "f_ref_hz": f_ref}
    # per-unit gates
    g3_sub = [abs(r["model_residual_hz"]) for r in units.values() if r["valid"] and r["rule"] != "production"]
    g3_prod = [abs(r["model_residual_hz"]) for r in units.values() if r["valid"] and r["rule"] == "production"]
    inconclusive = [k for k, r in units.items() if not (r["valid"] and r["invariance_pass"])]
    g3_fail = [k for k, r in units.items() if r["valid"] and not r["g3_pass"]]
    out["gates"] = {
        "g3_max_residual_subpixel_dual_hz": max(g3_sub, default=float("nan")),
        "g3_max_residual_production_hz": max(g3_prod, default=float("nan")),
        "g3_target_hz": G3_MODEL_RESIDUAL_HZ, "g3_failed_units": g3_fail,
        "g3_pass_all": bool(not g3_fail and len(units) > 0),
        "g3_pass_subpixel_dual": bool(not [k for k in g3_fail if units[k]["rule"] != "production"]),
        "inconclusive_units": inconclusive,
        "max_invariance_hz": max([r["invariance_hz"] for r in units.values() if np.isfinite(r["invariance_hz"])],
                                 default=float("nan")),
    }
    # e_shift for every layer unit
    es, esm = {}, {}
    for k, r in units.items():
        if r["layer"]:
            es[k] = e_shift(units, k, f_ref)
            esm[k] = e_shift_model(units, k, f_ref)
    out["e_shift_hz"] = es
    out["e_shift_model_hz"] = esm
    # --- F1: S1 order at fixed fill (L1) ---------------------------------------
    def ladder(arm, keys_h):
        pts = []
        for k, h in keys_h:
            e = es.get(k)
            if e is None or abs(e) < E_FLOOR_HZ:
                continue
            pts.append((h, abs(e), k))
        return pts
    L1 = [(f"{{arm}}|{s:g}|{int(round(350 * s))}", DC0 * s) for s in SCALES]
    L2 = [(f"{{arm}}|{s:g}|175", DC0 * s) for s in SCALES]
    fits = {}
    for arm in ("s1", "s1b", "s0"):
        pts = ladder(arm, [(k.format(arm=arm), h) for k, h in L1])
        fits[f"{arm}_L1"] = {"points": pts, "order": (w7.fit_line([(h, e) for h, e, _ in pts])[0] if len(pts) >= 3 else None)}
    pts = ladder("s1", [(k.format(arm="s1"), h) for k, h in L2])
    fits["s1_L2"] = {"points": pts, "order": (w7.fit_line([(h, e) for h, e, _ in pts])[0] if len(pts) >= 3 else None)}
    # total-error orders (reported) and the fitted rho at h1
    tot = {}
    for arm in ("s1", "r4"):
        pts = []
        for s in SCALES:
            r = units.get(f"{arm}|{s:g}|{int(round(350 * s))}")
            if _conclusive(r) and r["abs_err_hz"] >= E_FLOOR_HZ:
                pts.append((DC0 * s, r["abs_err_hz"]))
        tot[arm] = (w7.fit_line(pts) if len(pts) >= 3 else None)
    fits["s1_total_L1_order"] = tot["s1"][0] if tot["s1"] else None
    fits["r4_total_L1_order"] = tot["r4"][0] if tot["r4"] else None
    rho_fit = None
    if tot["s1"] and tot["r4"]:
        lh = np.log10(H1)
        rho_fit = float(10 ** ((tot["s1"][0] * lh + tot["s1"][1]) - (tot["r4"][0] * lh + tot["r4"][1])))
    fits["rho_fit_at_h1"] = rho_fit
    out["fits"] = fits
    p_s1 = fits["s1_L1"]["order"]
    out["e2_f1"] = {"window": list(P_S1_WINDOW), "p_s1": p_s1, "model": MODEL_ORDERS["s1_L1"],
                    "n_points": len(fits["s1_L1"]["points"]),
                    "pass": (None if p_s1 is None else bool(P_S1_WINDOW[0] <= p_s1 <= P_S1_WINDOW[1]))}
    # --- F2: |err_S1| / |err_R4| ---------------------------------------------------
    rho_s, rho_t = {}, {}
    for s in SCALES:
        a = units.get(f"s1|{s:g}|{int(round(350 * s))}")
        b = units.get(f"r4|{s:g}|{int(round(350 * s))}")
        if _conclusive(a) and _conclusive(b):
            rho_s[f"{s:g}"] = a["abs_err_hz"] / b["abs_err_hz"]
    for t_um in THICK_UM:
        a = units.get(f"s1|1|{t_um}")
        b = units.get(f"r4|1|{t_um}")
        if _conclusive(a) and _conclusive(b):
            rho_t[f"{t_um}"] = a["abs_err_hz"] / b["abs_err_hz"]
    allr = list(rho_s.values()) + list(rho_t.values())
    out["e2_f2"] = {"rho_max": RHO_MAX, "rho_per_scale": rho_s, "rho_per_thickness_s1": rho_t,
                    "rho_fit_at_h1": rho_fit, "model_per_scale": {f"{s:g}": v for s, v in MODEL_RHO_SCALE.items()},
                    "model_per_thickness": {f"{t}": v for t, v in MODEL_RHO_T.items()}, "model_fit": MODEL_RHO_FIT_H1,
                    "fired": (None if not allr else bool(max(allr) > RHO_MAX)), "n_ratios": len(allr)}
    # --- F3: the law ---------------------------------------------------------------
    law = {}
    fired = []
    for k in law_units():
        _, s, t_um = k.split("|")
        h, t = DC0 * float(s), int(t_um) * 1e-6
        e_law = K * t * (h - t)
        e = es.get(k)
        row = {"e_law_hz": e_law, "e_shift_hz": e, "e_shift_model_hz": esm.get(k),
               "tol_hz": max(LAW_ABS_HZ, LAW_REL * abs(e_law))}
        if e is not None:
            row["dev_hz"] = e - e_law
            row["dev_rel"] = (e - e_law) / e_law
            row["pass"] = bool(abs(e - e_law) <= row["tol_hz"])
            if not row["pass"]:
                fired.append(k)
        else:
            row["pass"] = None
        law[k] = row
    n_eval = sum(1 for v in law.values() if v["pass"] is not None)
    out["e2_f3"] = {"units": law, "fired_units": fired, "n_evaluated": n_eval,
                    "fired": (None if n_eval == 0 else bool(fired)),
                    "law": "e_shift = K_pert t (h - t)", "K_pert_mhz_per_mm2": K / 1e6 * 1e-6}
    # --- F4: S1b / S1 --------------------------------------------------------------
    ratio = {}
    for s in SCALES:
        a = es.get(f"s1b|{s:g}|{int(round(350 * s))}")
        b = es.get(f"s1|{s:g}|{int(round(350 * s))}")
        if a is not None and b is not None and abs(b) >= E_FLOOR_HZ:
            ratio[f"{s:g}"] = abs(a) / abs(b)
    out["e2_f4"] = {"ratio_max": S1B_RATIO_MAX, "ratio_per_scale": ratio,
                    "model": {f"{s:g}": v for s, v in MODEL_S1B_RATIO.items()},
                    "fired": (None if not ratio else bool(max(ratio.values()) > S1B_RATIO_MAX))}
    # --- reported ------------------------------------------------------------------
    rep = {"thickness_sweep_s1": {}, "p4": {}, "cost": {}, "r_arms_e_shift_hz": {}}
    for t_um in THICK_UM:
        for arm in ("s1", "s1b", "r4"):
            k = f"{arm}|1|{t_um}"
            r = units.get(k)
            if r and es.get(k) is not None:
                rep["thickness_sweep_s1"][k] = {"e_shift_hz": es[k], "shift_true_hz": r["f_true"] - f_ref,
                                                "e_shift_over_shift": es[k] / (r["f_true"] - f_ref),
                                                "err_hz": r["err_hz"], "e_shift_model_hz": esm.get(k)}
    for k, r in units.items():
        if r["layer"] and r["arm"] in ("r2", "r4", "r8"):
            rep["r_arms_e_shift_hz"][k] = {"meas": es.get(k), "model": esm.get(k)}
        if r["arm"] == "p4" or r["mesh"] == "p4":
            rep["p4"][k] = {"interface_table": r.get("interface_table"), "err_hz": r["err_hz"],
                            "e_shift_hz": es.get(k), "e_shift_model_hz": esm.get(k),
                            "model_residual_hz": r["model_residual_hz"]}
    for k, r in units.items():
        rep["cost"][k] = {"dt": r["dt"], "n_steps": r["n_steps"], "cells": r["cells"], "nz": r["nz"],
                          "dz_min_m": r["profile"]["d_min_m"], "wallclock_run_s": r["wallclock_run_s"]}
    dtr = {}
    for s in SCALES:
        for t_um in THICK_UM:
            a = units.get(f"s1|{s:g}|{t_um}")
            b = units.get(f"r4|{s:g}|{t_um}")
            if a and b:
                dtr[f"{s:g}|{t_um}"] = {"dt_s1_over_dt_r4": a["dt"] / b["dt"], "z_only": (DC0 * s) / (t_um * 1e-6 / 4),
                                        "cell_steps_r4_over_s1": (b["cells"] * b["n_steps"]) / (a["cells"] * a["n_steps"]),
                                        "wall_r4_over_s1": b["wallclock_run_s"] / a["wallclock_run_s"]}
    rep["dt_price"] = dtr
    out["reported"] = rep
    # --- verdict -------------------------------------------------------------------
    g = out["gates"]
    if not g["g3_pass_subpixel_dual"]:
        out["verdict"] = "STOP: G3 failed on a subpixel/dual unit: %s" % [k for k in g["g3_failed_units"] if units[k]["rule"] != "production"]
    elif p_s1 is None:
        out["verdict"] = "INCONCLUSIVE (fewer than 3 S1 L1 points above the floor / conclusive)"
    else:
        out["verdict"] = (f"p_s1={p_s1:.3f} F1_pass={out['e2_f1']['pass']} F2_fired={out['e2_f2']['fired']} "
                          f"(max rho {max(allr) if allr else float('nan'):.3f}) F3_fired={out['e2_f3']['fired']} "
                          f"({n_eval} law units) F4_fired={out['e2_f4']['fired']} "
                          f"G3_all={g['g3_pass_all']} inconclusive={len(inconclusive)}")
    return out


# ---------------------------------------------------------------------------
# Selfcheck (zero FDTD): oracle, meshes, nodes, every model number of note 2.5
# ---------------------------------------------------------------------------
def selfcheck(verbose: bool = True) -> dict:
    t0 = time.time()
    sc = {"oracle": oracle_selfcheck(), "k_pert": k_pert(), "checks": {}}
    ch = sc["checks"]
    o = sc["oracle"]
    ch["oracle_i"], ch["oracle_ip"], ch["oracle_ipp"] = o["i_pass"], o["ip_pass"], o["ipp_pass"]
    ch["f_ref_declared"], ch["f_true_declared"] = o["f_ref_matches_declared"], o["f_true_match_declared"]
    ch["mode_p5"] = bool(o["p_half_waves"] == 5)
    ch["k_pert_declared"] = sc["k_pert"]["matches_declared"]
    f_ref = o["f_ref_hz"]
    K = sc["k_pert"]["K_pert_hz_per_m2"]
    # every declared unit: mesh, nodes, column, model; the model table
    sc["units"] = {}
    sc["meshes"] = {}
    specs = {u["key"]: u for u in declared_units()}
    ch["n_declared_units"] = bool(len(specs) == 49)
    fm = {}
    for key, spec in specs.items():
        s, t_mesh = spec["scale"], spec["t_mesh_um"] * 1e-6
        prof = mesh_of(spec["mesh"], s, t_mesh)
        mk = f"{spec['mesh']}|{s:g}" + ("" if spec["mesh"] == "u" else f"|{spec['t_mesh_um']}")
        if mk not in sc["meshes"]:
            rep = w7.profile_report(prof, [0.0, Z1, Z1 + t_mesh, Z_AIR, L_Z] if spec["mesh"] != "u" else [0.0, Z_AIR, L_Z])
            zn = np.concatenate([[0.0], np.cumsum(prof)])
            nodes_ok = all(np.min(np.abs(zn - z)) <= CELL_TOL for z in (Z1, Z_AIR, Z_SRC, Z_PRB))
            if spec["mesh"] != "u":
                nodes_ok = bool(nodes_ok and np.min(np.abs(zn - (Z1 + t_mesh))) <= CELL_TOL)
            air = prof[zn[:-1] >= Z_AIR - 1e-12]
            nz_key = ("u", s) if spec["mesh"] == "u" else (spec["mesh"] if spec["mesh"] != "p4" else "r4", s, spec["t_mesh_um"])
            sc["meshes"][mk] = {**{k: v for k, v in rep.items() if k != "cells_m"}, "nodes_ok": nodes_ok,
                                "air_all_dc": bool(np.allclose(air, DC0 * s, atol=1e-12)),
                                "cells_um": (prof * 1e6).round(3).tolist()}
            ch[f"mesh_{mk}"] = bool(nodes_ok and rep["n"] == MESH_NZ[nz_key] and rep["max_ratio"] <= R_CAP + 1e-9
                                    and rep["interface_err_m"] <= CELL_TOL)
        dx = DXY0 * s
        grid = make_nonuniform_grid((A_X, B_Y), prof, dx, cpml_layers=0)
        dt = float(grid.dt)
        f_true = f_true_of(spec["t_um"] if spec["layer"] else 0, spec["centred"])
        col, info = column_for(spec["column_arm"], prof, spec["edges"], spec["eps"], s, f_true)
        m = model(prof, col, dx, dt, spec["edges"], spec["eps"], f_true)
        fm[key] = (m["f_model"], f_true)
        sc["units"][key] = {"dt": dt, "cells": int(grid.nx * grid.ny * grid.nz), "n_steps_15ns": int(round(T_TOTAL / dt)),
                            "f_true": f_true, "f_model": m["f_model"], "e_total_model_mhz": m["e_total_model"] / 1e6,
                            "z_fraction": m["z_fraction"], "column": col.tolist(),
                            **({"interface_table": info["interface_table"]} if "interface_table" in info else {}),
                            **({"cell_eps_nonpure": info["cell_eps_nonpure"]} if "cell_eps_nonpure" in info else {})}
        if spec["arm"] == "p4" and spec["layer"] and spec["t_mesh_um"] == int(round(350 * s)):
            tab = tuple(round(v["eps"], 1) for v in info["interface_table"].values())
            ch[f"p4_interface_table_{s:g}"] = bool(tab == P4_INTERFACE_TABLE[s])
            sc["units"][key]["interface_eps"] = list(tab)
    # model table of note 2.5
    esm = {}
    for key, (q_err, q_ref, q_shift) in MODEL_TABLE_MHZ.items():
        spec = specs[key]
        f_model, f_true = fm[key]
        f_model_ref, f_ref_u = fm[spec["ref_key"]]
        ok = (w7._rel(f_model, f_true + q_err * 1e6) <= SELFCHECK_REL
              and w7._rel(f_model_ref, f_ref_u + q_ref * 1e6) <= SELFCHECK_REL)
        esm[key] = (f_model - f_model_ref) - (f_true - f_ref)
        ok = ok and abs(esm[key] - q_shift * 1e6) <= 2e-4 * 1e6 + 2 * SELFCHECK_REL * f_true
        ch[f"model_{key}"] = bool(ok)
        sc["units"][key].update({"err_model_mhz": (f_model - f_true) / 1e6, "quoted_err_mhz": q_err,
                                 "err_ref_model_mhz": (f_model_ref - f_ref) / 1e6, "quoted_err_ref_mhz": q_ref,
                                 "e_shift_model_mhz": esm[key] / 1e6, "quoted_e_shift_mhz": q_shift})
        if verbose:
            print(f"  selfcheck {key:12s}: err {(f_model - f_true)/1e6:+9.4f} (quoted {q_err:+9.4f}) "
                  f"e_shift {esm[key]/1e6:+8.4f} (quoted {q_shift:+8.4f}) MHz ok={ok}", flush=True)
    # orders, ratios, law deviations
    h = np.log10([DC0 * s for s in SCALES])
    orders = {}
    for arm in ("s1", "s1b", "s0"):
        orders[f"{arm}_L1"] = float(np.polyfit(h, np.log10([abs(esm[f"{arm}|{s:g}|{int(round(350*s))}"]) for s in SCALES]), 1)[0])
    orders["s1_L2"] = float(np.polyfit(h, np.log10([abs(esm[f"s1|{s:g}|175"]) for s in SCALES]), 1)[0])
    tot = {}
    for arm in ("s1", "r4"):
        tot[arm] = np.polyfit(h, np.log10([abs(fm[f"{arm}|{s:g}|{int(round(350*s))}"][0] - fm[f"{arm}|{s:g}|{int(round(350*s))}"][1]) for s in SCALES]), 1)
        orders[f"{arm}_total_L1"] = float(tot[arm][0])
    lh = np.log10(H1)
    rho_fit = float(10 ** (np.polyval(tot["s1"], lh) - np.polyval(tot["r4"], lh)))
    rho_s = {s: abs((fm[f"s1|{s:g}|{int(round(350*s))}"][0] - fm[f"s1|{s:g}|{int(round(350*s))}"][1])
                    / (fm[f"r4|{s:g}|{int(round(350*s))}"][0] - fm[f"r4|{s:g}|{int(round(350*s))}"][1])) for s in SCALES}
    rho_t = {t: abs((fm[f"s1|1|{t}"][0] - fm[f"s1|1|{t}"][1]) / (fm[f"r4|1|{t}"][0] - fm[f"r4|1|{t}"][1])) for t in THICK_UM}
    s1b_ratio = {s: abs(esm[f"s1b|{s:g}|{int(round(350*s))}"] / esm[f"s1|{s:g}|{int(round(350*s))}"]) for s in SCALES}
    law_dev = {}
    for k in law_units():
        _, s, t_um = k.split("|")
        hh, tt = DC0 * float(s), int(t_um) * 1e-6
        law_dev[k] = (esm[k] - K * tt * (hh - tt)) / (K * tt * (hh - tt)) * 100
    sc["model"] = {"orders": orders, "rho_per_scale": {f"{s:g}": v for s, v in rho_s.items()}, "rho_fit_at_h1": rho_fit,
                   "rho_per_thickness": {f"{t}": v for t, v in rho_t.items()},
                   "s1b_over_s1": {f"{s:g}": v for s, v in s1b_ratio.items()}, "law_dev_pct": law_dev}
    ch["model_orders"] = bool(all(abs(orders[k] - MODEL_ORDERS[k]) <= 1e-3 for k in MODEL_ORDERS))
    ch["model_rho"] = bool(all(abs(rho_s[s] - MODEL_RHO_SCALE[s]) <= 1e-3 for s in SCALES)
                           and abs(rho_fit - MODEL_RHO_FIT_H1) <= 1e-3
                           and all(abs(rho_t[t] - MODEL_RHO_T[t]) <= 1e-3 for t in THICK_UM))
    ch["model_s1b_ratio"] = bool(all(abs(s1b_ratio[s] - MODEL_S1B_RATIO[s]) <= 1e-3 for s in SCALES))
    ch["model_law_dev"] = bool(all(abs(law_dev[k] - MODEL_LAW_DEV_PCT[k]) <= 0.1 for k in MODEL_LAW_DEV_PCT))
    sc["all_pass"] = bool(all(ch.values()))
    sc["failed"] = [k for k, v in ch.items() if not v]
    sc["wallclock_s"] = time.time() - t0
    sc.update(w7.provenance())
    if verbose:
        print(f"selfcheck: oracle (i) {o['i_worst_rel']:.2e}/{o['i_roots']} (i') {o['ip_worst_rel']:.2e}/{o['ip_roots']} "
              f"(i'') {o['ipp_worst_rel']:.2e}/{o['ipp_roots']}; f_ref={f_ref:.3f}; K_pert={K/1e6*1e-6:.4f} MHz/mm^2; "
              f"orders={ {k: round(v, 3) for k, v in orders.items()} }; all_pass={sc['all_pass']} failed={sc['failed']} "
              f"({sc['wallclock_s']:.0f} s)", flush=True)
    return sc


# ---------------------------------------------------------------------------
# JSON merge + CLI
# ---------------------------------------------------------------------------
def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"runs": [], "units": {}}


def _save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def _print_unit(r: dict) -> None:
    print(f"E2 {r['key']:16s}: f={r['f_meas']/1e9:.6f} GHz err={r['err_hz']/1e6:+9.4f} model={r['e_total_model']/1e6:+9.4f} "
          f"resid={r['model_residual_hz']/1e6:+.4f} inv={r['invariance_hz']/1e6:.4f} MHz valid={r['valid']} "
          f"nz={r['nz']} cells={r['cells']} steps={r['n_steps']} dt={r['dt']:.3e} wall={r['wallclock_run_s']:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders", default="L1,L2,T", help="comma list from L1,L2,T")
    ap.add_argument("--scales", default="2,1,0.5")
    ap.add_argument("--arms", default=",".join(LAYER_ARMS) + ",ref",
                    help="layer arms to run (s1,s1b,s0,r2,r4,r8,p4); 'ref' includes the references of the selected arms")
    ap.add_argument("--thicknesses", default="", help="comma list of t in um to keep (default: all declared)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    print("rfx.__file__ =", rfx.__file__, flush=True)
    data = _load(args.out)
    run = {**w7.provenance(), "ladders": args.ladders, "scales": args.scales, "arms": args.arms,
           "thicknesses": args.thicknesses, "smoke": args.smoke, "selfcheck_only": args.selfcheck}
    data.setdefault("runs", []).append(run)
    sc = selfcheck()
    data["selfcheck"] = sc
    _save(args.out, data)
    if not sc["all_pass"]:
        print("SELFCHECK FAILED:", sc["failed"], "- refusing to run any unit", flush=True)
        return 2
    if args.selfcheck:
        print("wrote", args.out)
        return 0

    ladders = [x.strip() for x in args.ladders.split(",") if x.strip()]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    thick = [int(x) for x in args.thicknesses.split(",") if x.strip()]
    units = data.setdefault("units", {})
    specs = declared_units(ladders)
    selected = []
    for u in specs:
        if u["scale"] not in scales:
            continue
        if thick and u["t_mesh_um"] not in thick:
            continue
        if u["layer"]:
            if u["arm"] not in arms:
                continue
        else:
            group = [a for a in LAYER_ARMS if MESH_OF[a] == u["mesh"]]
            if "ref" not in arms or not any(a in arms for a in group):
                continue
        selected.append(u)
    # references before their layer units within a scale, cheap scales first as given
    order = {s: i for i, s in enumerate(scales)}
    selected.sort(key=lambda u: (order[u["scale"]], u["layer"], u["key"]))
    print(f"{len(selected)} units selected:", [u["key"] for u in selected], flush=True)
    for u in selected:
        key = u["key"] + ("|smoke" if args.smoke else "")
        if key in units and not args.smoke:
            if not args.force:
                print(f"{key}: already measured (one attempt per unit) - skipping; --force keeps both", flush=True)
                continue
            tag = units[key].get("started_utc", "unknown")
            units[f"{key}|superseded|{tag}"] = units.pop(key)
            print(f"{key}: earlier attempt kept as {key}|superseded|{tag}", flush=True)
        row = measure_unit(u, smoke=args.smoke)
        units[key] = row
        _print_unit(row)
        data["judge"] = judge({k: v for k, v in units.items() if not v.get("smoke")},
                              K=sc["k_pert"]["K_pert_hz_per_m2"], f_ref=sc["oracle"]["f_ref_hz"])
        _save(args.out, data)
    if "judge" in data:
        print("E2 judge:", data["judge"]["verdict"], flush=True)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
