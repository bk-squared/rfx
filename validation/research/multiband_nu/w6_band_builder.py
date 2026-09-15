"""W6 — band-profile builder witnesses F7 (chain model, no FDTD) and F8
(FDTD, narrow fine band), pre-declared in
docs/design_notes/20260907_nu_band_profile_predeclaration.md §3.

F7: exact discrete scattering on the OLD
(main d990e18c ``_make_dz_profile``) and NEW auto-z PCB profiles embedded
in W2-count runways (140 lead cells of the profile's first cell, 150 tail
cells of its last). The whole-profile |R|, |T| come from
``chain_model.scattering``; the per-step |R_step| and the F7b cap come from
``chain_model.step_reflection``, the closed form of that same solve for a
single step (the dense float64 solve of a step carries ~1e-5 relative
noise on a 1e-7 reflection, and its last bits vary with the LAPACK build). Gates: F7a ``|R|^2 <= 1.5 (sum_steps |R_step|)^2`` on
the NEW profile; F7b max non-thirds step ``<= |R_single(r=1.4, d=max P)|``.
The OLD profile is regenerated from the committed source of d990e18c
(``git show``), never retyped.

F8: 2-run differencing + geometric time gating (the W2 method,
``w2_w3_reflection.py``) on profile A(n_b) built BY THE BUILDER:
``[1.96 mm] x 140 | 1.4 | [1.0 mm] x n_b | 1.4 | [1.96 mm] x 150``, incident
from the coarse side, PEC-closed, TE10 soft Ex source at K_SRC = 85, probe
K_PRB = 100 (coarse cells). B run: ``[1.96] x 400 + [1.0] x 4`` (dt pin).
Gate: the n_b = 4 row, window ``|R_meas - R_model| <= 0.20 R_model + 3e-5``
(amplitude). Rows 2, 8, 16 trace the band-width law.

Usage (declared in the note):
    PYTHONPATH=. python -m validation.research.multiband_nu.w6_band_builder \
        --widths 2,4,8,16 --out validation/research/multiband_nu/results/w6_band_builder.json
    add ``--f7-only`` to skip the FDTD rows; ``--f7-only --carry-f8``
    recomputes F7 in place and keeps the committed F8 block (whose rows
    hold FDTD measurements) with the provenance it was written under.

E1 (resolution x ratio sweep of the F8 witness, pre-declared in
docs/design_notes/20260907_nu_exp1_band_law_sweep_predeclaration.md):
    PYTHONPATH=. python -m validation.research.multiband_nu.w6_band_builder \
        --sweep --fine-cells-per-lambda 15,30,60 --ratio 1.2,1.4,2.0 \
        --widths 2,4,8,16,32 --out validation/research/multiband_nu/results/e1_band_law_sweep.json
    ``--model-only`` writes the chain-model predictions without FDTD;
    ``--resume`` skips cells already present in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types

import numpy as np

import rfx
from rfx.auto_config import _make_dz_profile
from rfx.nonuniform import make_band_profile, make_nonuniform_grid, run_nonuniform

from . import fixtures as fx
from .chain_model import bloch_kz, s0_sy, scattering, step_reflection
from .harness import build_pec_fixture
from .w2_w3_reflection import (
    F0, SIGMA_T, T0, FS2_FLOOR, dft_at, gaussian_sine, te10_sources, vg_of,
)

OLD_COMMIT = "d990e18ce0870ac7a893a86627e7232d1cf92c7b"

# --- F7 fixture -------------------------------------------------------------
PCB_FEATS = [(0.5e-3, 1.3e-3, 4.3), (1.3e-3, 1.4e-3, 4.3),
             (1.4e-3, 2.2e-3, 4.3), (2.2e-3, 2.3e-3, 4.3),
             (2.3e-3, 3.1e-3, 4.3)]
PCB_DOMAIN = 4.0e-3
PCB_DX = 0.2e-3
PCB_EDGES = [0.0, 0.5e-3, 1.3e-3, 1.4e-3, 2.2e-3, 2.3e-3, 3.1e-3, 4.0e-3]
F7_LEAD, F7_TAIL = 140, 150
F7_A = 4.5e-3            # transverse box; TE10 uses b = B_Y
F7_CAP_REF = 1.4


def _git_sha() -> str:
    """HEAD of the tree the run was made on (provenance, as W2/W4 record
    ``rfx.__file__``)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _git_dirty() -> bool:
    """True when the worktree had uncommitted changes at run time."""
    try:
        return subprocess.check_output(["git", "status", "--porcelain"], text=True) != ""
    except (subprocess.CalledProcessError, OSError):
        return True


def _old_make_dz_profile():
    """``_make_dz_profile`` exactly as committed on main d990e18c, loaded
    from ``git show`` into a throwaway module (provenance, not a retype)."""
    src = subprocess.check_output(
        ["git", "show", f"{OLD_COMMIT}:rfx/auto_config.py"], text=True)
    mod = types.ModuleType("_old_auto_config_d990e18c")
    mod.__file__ = f"git:{OLD_COMMIT}:rfx/auto_config.py"
    # the old module's @dataclass looks itself up in sys.modules
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod._make_dz_profile


def _embed(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64)
    return np.concatenate([np.full(F7_LEAD, p[0]), p, np.full(F7_TAIL, p[-1])])


def _thirds_pairs(profile: np.ndarray, edges) -> set[int]:
    c = np.asarray(profile, dtype=float)
    nodes = np.concatenate([[0.0], np.cumsum(c)])
    ex: set[int] = set()
    for lo, hi in zip(edges[1:-2], edges[2:-1]):    # the five blocks
        i = int(np.argmin(np.abs(nodes - lo)))
        j = int(np.argmin(np.abs(nodes - hi)))
        if j - i >= 3 and abs(c[i + 1] - 2 * c[i]) <= 1e-12 * c[i + 1]:
            ex.update((i, i + 1))
        if j - i >= 3 and abs(c[j - 2] - 2 * c[j - 1]) <= 1e-12 * c[j - 2]:
            ex.update((j - 3, j - 2))
    return ex


def f7_profile_report(name: str, profile: np.ndarray, dt: float, dy: float,
                      b: float) -> dict:
    emb = _embed(profile)
    R, T = scattering(emb, F7_LEAD, F7_TAIL, F0, dt, dy, b)
    steps = []
    ex = _thirds_pairs(profile, PCB_EDGES)
    for k in range(len(profile) - 1):
        d0, d1 = float(profile[k]), float(profile[k + 1])
        if d0 == d1:
            continue
        # closed form, not scattering() on a [d0]*lead + [d1]*tail runway:
        # same model, but the float64 dense solve of that runway carries
        # ~1e-5 relative noise on a 1e-7 reflection and its last bits are
        # LAPACK-build dependent (chain_model.step_reflection docstring).
        Rk = step_reflection(d0, d1, F0, dt, dy, b)
        steps.append({"index": k, "d_from_um": d0 * 1e6, "d_to_um": d1 * 1e6,
                      "ratio": max(d0 / d1, d1 / d0), "R_step": Rk,
                      "thirds_pair": k in ex})
    non_thirds = [s for s in steps if not s["thirds_pair"]]
    thirds = [s for s in steps if s["thirds_pair"]]
    sum_amp = sum(s["R_step"] for s in steps)
    sum_pow = sum(s["R_step"] ** 2 for s in steps)
    dmax = float(np.max(profile))
    R1 = step_reflection(dmax / F7_CAP_REF, dmax, F0, dt, dy, b)
    max_nt = max(s["R_step"] for s in non_thirds) if non_thirds else 0.0
    max_th = max(s["R_step"] for s in thirds) if thirds else 0.0
    rr = profile[1:] / profile[:-1]
    rr = np.maximum(rr, 1 / rr)
    return {
        "name": name, "nz": int(len(profile)),
        "dz_min_um": float(np.min(profile)) * 1e6,
        "dz_max_um": dmax * 1e6, "max_ratio": float(rr.max()),
        "n_ratio_over_1p4": int(np.sum(rr > 1.4 + 1e-9)),
        "R_total": abs(R), "R_total_sq": abs(R) ** 2, "T_total": abs(T),
        "n_steps": len(steps), "sum_abs_R_step": sum_amp,
        "sum_abs_R_step_sq": sum_amp ** 2, "sum_R_step_sq": sum_pow,
        "max_nonthirds_step": max_nt, "max_thirds_step": max_th,
        "R_single_cap_at_dmax": R1,
        "f7a_window": 1.5 * sum_amp ** 2,
        "f7a_fired": bool(abs(R) ** 2 > 1.5 * sum_amp ** 2),
        "f7b_fired": bool(max_nt > R1 * (1 + 1e-9)),
        "steps": steps, "cells_um": (profile * 1e6).tolist(),
    }


def run_f7() -> dict:
    old_fn = _old_make_dz_profile()
    old = np.asarray(old_fn(PCB_FEATS, PCB_DOMAIN, PCB_DX), dtype=np.float64)
    new = np.asarray(_make_dz_profile(PCB_FEATS, PCB_DOMAIN, PCB_DX), dtype=np.float64)
    grid_old = make_nonuniform_grid((F7_A, fx.B_Y), _embed(old), PCB_DX, cpml_layers=0)
    grid_new = make_nonuniform_grid((F7_A, fx.B_Y), _embed(new), PCB_DX, cpml_layers=0)
    dt_old, dt_new = float(grid_old.dt), float(grid_new.dt)
    dy = PCB_DX
    out = {"old_commit": OLD_COMMIT, "dt_old_s": dt_old, "dt_new_s": dt_new,
           "F0_Hz": F0, "dy_m": dy, "b_m": fx.B_Y,
           "lead_cells": F7_LEAD, "tail_cells": F7_TAIL}
    out["old"] = f7_profile_report("OLD (d990e18c _make_dz_profile)", old, dt_old, dy, fx.B_Y)
    out["new"] = f7_profile_report("NEW (_make_dz_profile on the band engine)", new, dt_new, dy, fx.B_Y)
    for key in ("old", "new"):
        r = out[key]
        print(f"F7 {r['name']}: nz={r['nz']} dz_min={r['dz_min_um']:.3f}um "
              f"max_ratio={r['max_ratio']:.3f} |R|={r['R_total']:.4e} "
              f"|R|^2={r['R_total_sq']:.4e} sum|Rstep|={r['sum_abs_R_step']:.4e} "
              f"(sum)^2={r['sum_abs_R_step_sq']:.4e} sum(Rstep^2)={r['sum_R_step_sq']:.4e} "
              f"max_nonthirds={r['max_nonthirds_step']:.4e} max_thirds={r['max_thirds_step']:.4e} "
              f"R_single(1.4,dmax)={r['R_single_cap_at_dmax']:.4e} "
              f"F7a_window={r['f7a_window']:.4e} fired={r['f7a_fired']} F7b_fired={r['f7b_fired']}",
              flush=True)
    return out


# --- F8 ----------------------------------------------------------------------
DC = fx.DZ_FINE * 1.4 ** 2         # 1.96 mm coarse cell
DR = fx.DZ_FINE * 1.4              # 1.4 mm ramp cell
N_LEAD_C = 140
N_TAIL_C = 150
K_SRC = 85
K_PRB = 100
N_STEPS = 1200
B_N_COARSE = 400
B_N_FINE_PIN = 4


def a_profile(n_b: int) -> np.ndarray:
    z1 = N_LEAD_C * DC + DR
    z2 = z1 + n_b * fx.DZ_FINE
    z3 = z2 + DR + N_TAIL_C * DC
    return make_band_profile([0.0, z1, z2, z3], [DC, fx.DZ_FINE, DC],
                             protected=[False, True, False], max_ratio=1.4)


def a_profile_expected(n_b: int) -> np.ndarray:
    return np.asarray([DC] * N_LEAD_C + [DR] + [fx.DZ_FINE] * n_b + [DR]
                      + [DC] * N_TAIL_C, dtype=np.float64)


def b_profile() -> np.ndarray:
    return np.asarray([DC] * B_N_COARSE + [fx.DZ_FINE] * B_N_FINE_PIN, dtype=np.float64)


def _run_probe(profile: np.ndarray, n_steps: int, k_src: int = K_SRC,
               k_prb: int = K_PRB):
    """TE10 soft source at cell ``k_src``, Ex probe at cell ``k_prb``
    (defaults: the lane-A F8 planes). E1 passes the setting's own
    cells — the first E1 attempt (a2cb6cf3) ran with the defaults and
    put the source of the ratio-2.0 A runs inside the fine tail / band;
    see the note's Results, first attempt."""
    grid, mats = build_pec_fixture(profile, (fx.A_X, fx.B_Y), fx.DXY)
    wf = gaussian_sine(n_steps, float(grid.dt), SIGMA_T, T0)
    srcs = te10_sources(grid, k_src, wf)
    out = run_nonuniform(grid, mats, n_steps, sources=srcs,
                         probes=[(1, grid.ny // 2, k_prb, "ex")])
    return grid, np.asarray(out["time_series"][:, 0], dtype=np.float64)


def _cell_delay(profile: np.ndarray, dt: float, dy: float, b: float) -> float:
    cache: dict[float, float] = {}
    tot = 0.0
    for d in profile:
        d = float(d)
        if d not in cache:
            cache[d] = vg_of(d, dt, dy, b)
        tot += d / cache[d]
    return tot


def f8_arm(n_b: int, trace_b: np.ndarray, grid_b) -> dict:
    prof = a_profile(n_b)
    expected = a_profile_expected(n_b)
    builder_exact = (len(prof) == len(expected)
                     and bool(np.max(np.abs(prof - expected)) <= 1e-12))
    grid_a, trace_a = _run_probe(prof, len(trace_b))
    dt = float(grid_a.dt)
    assert abs(dt - float(grid_b.dt)) < 1e-20, (dt, float(grid_b.dt))
    dy, b = float(grid_a.dy), fx.B_Y
    vg_c = vg_of(DC, dt, dy, b)
    z_src, z_prb = K_SRC * DC, K_PRB * DC
    z_tr = N_LEAD_C * DC
    t_r = T0 + (2 * z_tr - z_src - z_prb) / vg_c
    t_s = T0 + (z_src + 2 * z_tr - z_prb) / vg_c
    # far wall: through the band + tail and back
    beyond = prof[N_LEAD_C:]
    t_f = T0 + (z_tr - z_src) / vg_c + 2 * _cell_delay(beyond, dt, dy, b) + (z_tr - z_prb) / vg_c
    # last band-internal return: to the far ramp/band edge and back
    band_and_ramps = prof[N_LEAD_C:N_LEAD_C + 2 + n_b]
    t_band = T0 + (z_tr - z_src) / vg_c + 2 * _cell_delay(band_and_ramps, dt, dy, b) + (z_tr - z_prb) / vg_c
    gate_end = min(t_s, t_f) - 4 * SIGMA_T
    assert t_r + 4 * SIGMA_T < gate_end, (t_r, t_s, t_f)
    diff = trace_a - trace_b
    n_gate = int(gate_end / dt)
    refl = dft_at(diff, dt, F0, 0, n_gate)
    t_echo_prb = T0 + (z_src + z_prb) / vg_c
    t_inc_end = min(T0 + (z_prb - z_src) / vg_c + 8 * SIGMA_T, t_echo_prb - 4 * SIGMA_T)
    inc = dft_at(trace_b, dt, F0, 0, int(t_inc_end / dt))
    r_meas = abs(refl) / abs(inc)
    R, _ = scattering(prof, N_LEAD_C, N_TAIL_C, F0, dt, dy, b)
    r_model = abs(R)
    half = 0.20 * r_model + FS2_FLOOR
    return {
        "n_b": n_b, "band_mm": n_b * fx.DZ_FINE * 1e3,
        "builder_matches_declared_vector": builder_exact,
        "profile_cells_mm": (prof * 1e3).tolist(),
        "nz": int(len(prof)), "dt_s": dt,
        "R_model": r_model, "R_model_sq": r_model ** 2,
        "R_model_db": 20 * np.log10(r_model),
        "window": [r_model - half, r_model + half],
        "R_meas": r_meas, "R_meas_sq": r_meas ** 2,
        "R_meas_db": 20 * np.log10(max(r_meas, 1e-300)),
        "deviation": abs(r_meas - r_model),
        "f8_fired": bool(abs(r_meas - r_model) > half),
        "is_gate_row": n_b == 4,
        "gates_ns": {"t_r": t_r * 1e9, "t_band_last": t_band * 1e9,
                     "t_s": t_s * 1e9, "t_f": t_f * 1e9,
                     "gate_end": gate_end * 1e9, "gate_steps": n_gate,
                     "t_inc_end": t_inc_end * 1e9},
    }


def run_f8(widths: list[int]) -> dict:
    prof_b = b_profile()
    grid_b, trace_b = _run_probe(prof_b, N_STEPS)
    out = {"dt_b_s": float(grid_b.dt), "n_steps": N_STEPS, "K_SRC": K_SRC,
           "K_PRB": K_PRB, "coarse_cell_mm": DC * 1e3, "ramp_cell_mm": DR * 1e3,
           "fine_cell_mm": fx.DZ_FINE * 1e3, "rows": []}
    for n_b in widths:
        res = f8_arm(n_b, trace_b, grid_b)
        out["rows"].append(res)
        print(f"F8 n_b={n_b}: builder_exact={res['builder_matches_declared_vector']} "
              f"R_meas={res['R_meas']:.4e} ({res['R_meas_db']:.1f} dB) "
              f"R_model={res['R_model']:.4e} window=[{res['window'][0]:.4e}, "
              f"{res['window'][1]:.4e}] fired={res['f8_fired']} "
              f"gates(ns) t_r={res['gates_ns']['t_r']:.3f} "
              f"t_band={res['gates_ns']['t_band_last']:.3f} "
              f"gate_end={res['gates_ns']['gate_end']:.3f}", flush=True)
    return out

# --- E1: resolution x ratio sweep of the F8 witness ---------------------------
# Pre-declared in docs/design_notes/20260907_nu_exp1_band_law_sweep_predeclaration.md.
# The lane-A F8 layout is kept in PHYSICAL lengths (source 166.6 mm, probe
# 196.0 mm, transition 274.4 mm, tail 294.0 mm, B reference 784.0 mm, all
# multiples of the 1.96 mm coarse cell) and re-expressed in cells of the
# setting's coarse cell, so the 30 / 1.4 cell of the sweep reproduces the
# lane-A F8 arm bit for bit (K_SRC 85, K_PRB 100, 140 / 150 / 400 cells,
# 1200 steps) and every other cell keeps the same gate geometry in time.
E1_N_LAMBDA_REF = 30                    # lane-A fine resolution (1 mm cells)
E1_Z_SRC = K_SRC * DC                   # 166.6 mm
E1_Z_PRB = K_PRB * DC                   # 196.0 mm
E1_Z_LEAD = N_LEAD_C * DC               # 274.4 mm
E1_Z_TAIL = N_TAIL_C * DC               # 294.0 mm
E1_Z_B = B_N_COARSE * DC                # 784.0 mm
E1_DT_LANE_A = 2.4027649366612596e-12   # dt of the lane-A F8 runs (JSON dt_b_s)
E1_T_RUN = N_STEPS * E1_DT_LANE_A       # 2.883 ns, the lane-A run length
E1_T_RUN_PAD = 0.1e-9                   # run at least 0.1 ns past t_s
E1_LAW_SWEEP_MAX = 80                   # chain-model n_b range for the c fit
E1_C_FIT_MAX_DR = 3.0                   # c search range, in ramp cells
E1_BOUND_SLACK = 0.05                   # (ii): R_meas <= 2 R_single (1 + 0.05)
E1_SCALING_TOL = 0.25                   # (i): measured / model ratio within 25 %
E1_C_TOL_DR = 0.10                      # (iii): |c_meas - c_model| <= 0.10 DR
E1_CLASS_RATIO = 1.4                    # -54 dB accuracy class: ratio cap ...
E1_CLASS_N_LAMBDA = 30                  # ... and >= 30 fine cells per wavelength


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def e1_setting(n_lambda: int, ratio: float) -> dict:
    """Cell layout of one (resolution, ratio) setting. The fine cell is
    ``DZ_FINE x 30 / n_lambda`` (2.0 / 1.0 / 0.5 mm for 15 / 30 / 60), the
    coarse cell ``ratio^2 x fine``, the ramp cell ``ratio x fine``; every
    runway length is the lane-A physical length rounded to whole coarse
    (or, for the single-ramp tail, fine) cells."""
    dzf = fx.DZ_FINE * E1_N_LAMBDA_REF / n_lambda
    dc, dr = dzf * ratio * ratio, dzf * ratio
    return {
        "n_lambda": n_lambda, "ratio": ratio,
        "fine_cell_m": dzf, "coarse_cell_m": dc, "ramp_cell_m": dr,
        "fine_cells_per_lambda0": (C0_E1 / F0) / dzf,
        "coarse_cells_per_lambda0": (C0_E1 / F0) / dc,
        "k_src": _round_half_up(E1_Z_SRC / dc),
        "k_prb": _round_half_up(E1_Z_PRB / dc),
        "n_lead": _round_half_up(E1_Z_LEAD / dc),
        "n_tail": _round_half_up(E1_Z_TAIL / dc),
        "n_b_coarse": _round_half_up(E1_Z_B / dc),
        "n_tail_fine": _round_half_up(E1_Z_TAIL / dzf),
        "in_accuracy_class": bool(ratio <= E1_CLASS_RATIO + 1e-12
                                  and n_lambda >= E1_CLASS_N_LAMBDA),
    }


C0_E1 = 299792458.0


def e1_band_profile(st: dict, n_b: int) -> np.ndarray:
    dc, dzf, dr = st["coarse_cell_m"], st["fine_cell_m"], st["ramp_cell_m"]
    z1 = st["n_lead"] * dc + dr
    z2 = z1 + n_b * dzf
    z3 = z2 + dr + st["n_tail"] * dc
    return make_band_profile([0.0, z1, z2, z3], [dc, dzf, dc],
                             protected=[False, True, False], max_ratio=st["ratio"])


def e1_band_expected(st: dict, n_b: int) -> np.ndarray:
    dc, dzf, dr = st["coarse_cell_m"], st["fine_cell_m"], st["ramp_cell_m"]
    return np.asarray([dc] * st["n_lead"] + [dr] + [dzf] * n_b + [dr]
                      + [dc] * st["n_tail"], dtype=np.float64)


def e1_single_profile(st: dict) -> np.ndarray:
    """Single ramp coarse -> ramp -> fine, fine tail of the lane-A tail
    length: the R_single arm of law check (i)."""
    dc, dzf, dr = st["coarse_cell_m"], st["fine_cell_m"], st["ramp_cell_m"]
    z1 = st["n_lead"] * dc + dr
    z2 = z1 + st["n_tail_fine"] * dzf
    return make_band_profile([0.0, z1, z2], [dc, dzf], protected=[False, True],
                             max_ratio=st["ratio"])


def e1_single_expected(st: dict) -> np.ndarray:
    dc, dzf, dr = st["coarse_cell_m"], st["fine_cell_m"], st["ramp_cell_m"]
    return np.asarray([dc] * st["n_lead"] + [dr] + [dzf] * st["n_tail_fine"],
                      dtype=np.float64)


def e1_b_profile(st: dict) -> np.ndarray:
    return np.asarray([st["coarse_cell_m"]] * st["n_b_coarse"]
                      + [st["fine_cell_m"]] * B_N_FINE_PIN, dtype=np.float64)


def e1_fit_c(n_bs, r_vals, dzf: float, r_single: float, k_g: float,
             c_max: float) -> float:
    """Least-squares c in ``2 R_single |sin(k_g (n_b dzf + c))|`` over the
    given rows: coarse scan then two refinements (the objective is smooth
    and 1-D; the scan spacing is c_max / 4000)."""
    L = np.asarray(n_bs, dtype=np.float64) * dzf
    R = np.asarray(r_vals, dtype=np.float64)

    def err(c):
        return float(np.sum((R - 2 * r_single * np.abs(np.sin(k_g * (L + c)))) ** 2))

    lo, hi, n = 0.0, c_max, 4001
    for _ in range(3):
        cs = np.linspace(lo, hi, n)
        e = np.array([err(c) for c in cs])
        i = int(np.argmin(e))
        step = cs[1] - cs[0]
        lo, hi = max(0.0, cs[i] - step), min(c_max, cs[i] + step)
    return float(cs[i])


def e1_stopband_edge_hz(dc: float, dt: float, dy: float, b: float) -> float | None:
    """Lowest frequency at which the coarse lattice stops propagating the
    TE10 Bloch wave (``bloch_kz`` argument reaches 1); None if above the
    scan (60 GHz)."""
    for f in np.linspace(F0, 60e9, 5001):
        s0, sy = s0_sy(f, dt, dy, b)
        if s0 ** 2 - sy ** 2 < 0:
            continue
        if dc * np.sqrt(s0 ** 2 - sy ** 2) / 2 >= 1.0:
            return float(f)
    return None


def e1_model(st: dict, dt: float, widths: list[int]) -> dict:
    """Chain-model side of one setting: single-ramp R, k_g, the c fit over
    n_b = 0..80 (c_model), the width rows with their frozen windows."""
    dy, b = fx.DXY, fx.B_Y
    single = e1_single_profile(st)
    single_exact = (len(single) == len(e1_single_expected(st))
                    and bool(np.max(np.abs(single - e1_single_expected(st))) <= 1e-12))
    r_single = abs(scattering(single, st["n_lead"], st["n_tail_fine"], F0, dt, dy, b)[0])
    k_g = bloch_kz(F0, dt, dy, b, st["fine_cell_m"])
    k_c = bloch_kz(F0, dt, dy, b, st["coarse_cell_m"])
    sweep_n = list(range(0, E1_LAW_SWEEP_MAX + 1))
    sweep_r = []
    for n_b in sweep_n:
        prof = e1_band_expected(st, n_b)
        sweep_r.append(abs(scattering(prof, st["n_lead"], st["n_tail"], F0, dt, dy, b)[0]))
    c_model = e1_fit_c(sweep_n, sweep_r, st["fine_cell_m"], r_single, k_g,
                       E1_C_FIT_MAX_DR * st["ramp_cell_m"])
    fit = 2 * r_single * np.abs(np.sin(k_g * (np.asarray(sweep_n) * st["fine_cell_m"] + c_model)))
    rows = []
    for n_b in widths:
        prof = e1_band_profile(st, n_b)
        exp = e1_band_expected(st, n_b)
        exact = len(prof) == len(exp) and bool(np.max(np.abs(prof - exp)) <= 1e-12)
        r_model = abs(scattering(prof, st["n_lead"], st["n_tail"], F0, dt, dy, b)[0])
        half = 0.20 * r_model + FS2_FLOOR
        rows.append({"n_b": n_b, "band_mm": n_b * st["fine_cell_m"] * 1e3,
                     "builder_matches_declared_vector": exact, "nz": int(len(prof)),
                     "R_model": r_model, "R_model_db": 20 * np.log10(r_model),
                     "window": [r_model - half, r_model + half], "half": half,
                     "bound_2R1": 2 * r_single * (1 + E1_BOUND_SLACK),
                     "fp_form_model": 2 * r_single * abs(np.sin(k_g * (n_b * st["fine_cell_m"] + c_model)))})
    return {
        "dt_s": dt, "single_builder_matches_declared_vector": single_exact,
        "single_nz": int(len(single)),
        "R_single_model": r_single, "R_single_model_db": 20 * np.log10(r_single),
        "k_g_fine_per_m": k_g, "k_coarse_per_m": k_c,
        "lambda_g_fine_mm": 2 * np.pi / k_g * 1e3,
        "coarse_bloch_arg": st["coarse_cell_m"] * np.sqrt(
            s0_sy(F0, dt, dy, b)[0] ** 2 - s0_sy(F0, dt, dy, b)[1] ** 2) / 2,
        "vg_coarse_over_c": vg_of(st["coarse_cell_m"], dt, dy, b) / C0_E1,
        "stopband_edge_ghz": (lambda f: None if f is None else f / 1e9)(
            e1_stopband_edge_hz(st["coarse_cell_m"], dt, dy, b)),
        "c_model_m": c_model, "c_model_over_ramp_cell": c_model / st["ramp_cell_m"],
        "c_fit_rms_0_80": float(np.sqrt(np.mean((np.asarray(sweep_r) - fit) ** 2))),
        "c_fit_max_rel_0_80": float(np.max(np.abs(np.asarray(sweep_r) - fit)
                                           / np.maximum(np.asarray(sweep_r), 1e-300))),
        "chain_max_over_2R1": float(max(sweep_r) / (2 * r_single)),
        "chain_sweep_R": sweep_r,
        "rows": rows,
    }


def e1_gates(st: dict, prof: np.ndarray, kind: str, n_b: int, dt: float,
             n_steps: int) -> dict:
    """Gate geometry of one arm from the chain-model group velocities (the
    lane-A construction): arrival times in s, the gate end, and the five
    margins whose conjunction is ``gates_hold``. Pure geometry — used by
    the note's gate table before any run and by ``e1_arm`` after it."""
    dy, b = fx.DXY, fx.B_Y
    dc = st["coarse_cell_m"]
    vg_c = vg_of(dc, dt, dy, b)
    z_src, z_prb, z_tr = st["k_src"] * dc, st["k_prb"] * dc, st["n_lead"] * dc
    t_r = T0 + (2 * z_tr - z_src - z_prb) / vg_c
    t_s = T0 + (z_src + 2 * z_tr - z_prb) / vg_c
    beyond = prof[st["n_lead"]:]
    t_f = T0 + (z_tr - z_src) / vg_c + 2 * _cell_delay(beyond, dt, dy, b) + (z_tr - z_prb) / vg_c
    if kind == "band":
        inner = prof[st["n_lead"]:st["n_lead"] + 2 + n_b]
    else:
        inner = prof[st["n_lead"]:st["n_lead"] + 1]
    t_inner = T0 + (z_tr - z_src) / vg_c + 2 * _cell_delay(inner, dt, dy, b) + (z_tr - z_prb) / vg_c
    gate_end = min(t_s, t_f) - 4 * SIGMA_T
    n_gate = int(gate_end / dt)
    t_echo_prb = T0 + (z_src + z_prb) / vg_c
    t_inc_arr = T0 + (z_prb - z_src) / vg_c
    t_inc_end = min(t_inc_arr + 8 * SIGMA_T, t_echo_prb - 4 * SIGMA_T)
    # B reference: far wall / fine pin return must land after the gate
    t_bpin = T0 + (z_src + 2 * st["n_b_coarse"] * dc - z_prb) / vg_c
    margins = {
        "reflection_inside": gate_end - (t_r + 4 * SIGMA_T),
        "inner_return_inside": gate_end - (t_inner + 4 * SIGMA_T),
        "run_covers_gate": n_steps * dt - gate_end,
        "b_pin_return_after_gate": t_bpin - gate_end,
        "incident_inside": t_inc_end - (t_inc_arr + 4 * SIGMA_T),
    }
    return {
        "gates_ns": {"t_r": t_r * 1e9, "t_inner_last": t_inner * 1e9, "t_s": t_s * 1e9,
                     "t_f": t_f * 1e9, "gate_end": gate_end * 1e9, "gate_steps": n_gate,
                     "t_inc_end": t_inc_end * 1e9, "t_b_pin_return": t_bpin * 1e9,
                     "run_end": n_steps * dt * 1e9},
        "gate_margins_ns": {k: v * 1e9 for k, v in margins.items()},
        "gates_hold": bool(all(v > 0 for v in margins.values())),
        "_n_gate": n_gate, "_t_inc_end": t_inc_end,
    }


def e1_arm(st: dict, prof: np.ndarray, kind: str, n_b: int, trace_b: np.ndarray,
           grid_b, r_model: float, r_single: float) -> dict:
    """One FDTD arm (band or single ramp) with the lane-A gating, generalized
    to the setting's cells; every gate margin is reported in ns and
    ``gates_hold`` is their conjunction (nothing is asserted — a violated
    gate is a result)."""
    n_steps = len(trace_b)
    grid_a, trace_a = _run_probe(prof, n_steps, st["k_src"], st["k_prb"])
    dt = float(grid_a.dt)
    dt_match = abs(dt - float(grid_b.dt)) < 1e-20
    # the source and probe cells must be coarse lead cells of THIS profile
    # (identical in A and B) — the first attempt's defect, now checked
    planes_ok = bool(st["k_prb"] < st["n_lead"]
                     and abs(prof[st["k_src"]] - st["coarse_cell_m"]) <= 1e-12
                     and abs(prof[st["k_prb"]] - st["coarse_cell_m"]) <= 1e-12)
    g = e1_gates(st, prof, kind, n_b, dt, n_steps)
    diff = trace_a - trace_b
    refl = dft_at(diff, dt, F0, 0, min(g["_n_gate"], n_steps))
    inc = dft_at(trace_b, dt, F0, 0, int(g["_t_inc_end"] / dt))
    r_meas = abs(refl) / abs(inc)
    half = 0.20 * r_model + FS2_FLOOR
    dev = abs(r_meas - r_model)
    out = {
        "kind": kind, "n_b": n_b, "nz": int(len(prof)), "dt_s": dt, "dt_matches_b": dt_match,
        "R_model": r_model, "R_meas": r_meas, "R_meas_db": 20 * np.log10(max(r_meas, 1e-300)),
        "deviation": dev, "deviation_rel": dev / r_model,
        "window": [r_model - half, r_model + half],
        "fired": bool(dev > half),
        "gates_ns": g["gates_ns"], "gate_margins_ns": g["gate_margins_ns"],
        "k_src_used": st["k_src"], "k_prb_used": st["k_prb"],
        "source_probe_planes_mm": [st["k_src"] * st["coarse_cell_m"] * 1e3,
                                   st["k_prb"] * st["coarse_cell_m"] * 1e3],
        "source_probe_in_coarse_lead": planes_ok,
        "gates_hold": bool(g["gates_hold"] and dt_match and planes_ok),
    }
    if kind == "band":
        out["bound_2R1"] = 2 * r_single * (1 + E1_BOUND_SLACK)
        out["bound_fired"] = bool(r_meas > 2 * r_single * (1 + E1_BOUND_SLACK))
    return out


def e1_cell_key(n_lambda: int, ratio: float) -> str:
    return f"N{n_lambda}_r{ratio:g}"


def run_e1_cell(st: dict, widths: list[int], model_only: bool) -> dict:
    dzf = st["fine_cell_m"]
    grid_probe = make_nonuniform_grid((fx.A_X, fx.B_Y), e1_b_profile(st), fx.DXY, cpml_layers=0)
    dt = float(grid_probe.dt)
    model = e1_model(st, dt, widths)
    vg_c = vg_of(st["coarse_cell_m"], dt, fx.DXY, fx.B_Y)
    dc = st["coarse_cell_m"]
    t_s = T0 + (st["k_src"] * dc + 2 * st["n_lead"] * dc - st["k_prb"] * dc) / vg_c
    n_steps = int(np.ceil(max(E1_T_RUN, t_s + E1_T_RUN_PAD) / dt - 1e-9))
    for row in model["rows"]:
        g = e1_gates(st, e1_band_expected(st, row["n_b"]), "band", row["n_b"], dt, n_steps)
        row["gates_ns"], row["gate_margins_ns"], row["gates_hold"] = (
            g["gates_ns"], g["gate_margins_ns"], g["gates_hold"])
    g = e1_gates(st, e1_single_expected(st), "single", 0, dt, n_steps)
    model["single_gates_ns"], model["single_gate_margins_ns"], model["single_gates_hold"] = (
        g["gates_ns"], g["gate_margins_ns"], g["gates_hold"])
    cell = {"setting": st, "n_steps": n_steps, "run_ns": n_steps * dt * 1e9,
            "model": model}
    print(f"E1 {e1_cell_key(st['n_lambda'], st['ratio'])}: fine={dzf*1e3:.3f}mm coarse={dc*1e3:.3f}mm "
          f"K={st['k_src']}/{st['k_prb']} lead/tail/B={st['n_lead']}/{st['n_tail']}/{st['n_b_coarse']} "
          f"dt={dt:.4e} n_steps={n_steps} R1_model={model['R_single_model']:.4e} "
          f"({model['R_single_model_db']:.1f} dB) k_g={model['k_g_fine_per_m']/1e3:.5f}/mm "
          f"c_model={model['c_model_m']*1e3:.3f}mm (c/DR={model['c_model_over_ramp_cell']:.3f}) "
          f"vg_c/c={model['vg_coarse_over_c']:.3f} stopband={model['stopband_edge_ghz']} GHz "
          f"single_exact={model['single_builder_matches_declared_vector']}", flush=True)
    for row in model["rows"]:
        print(f"   model n_b={row['n_b']}: exact={row['builder_matches_declared_vector']} "
              f"R_model={row['R_model']:.4e} ({row['R_model_db']:.1f} dB) "
              f"window=[{row['window'][0]:.4e}, {row['window'][1]:.4e}] "
              f"fp_form={row['fp_form_model']:.4e} gates_hold={row['gates_hold']}", flush=True)
    if model_only:
        return cell
    t0 = time.time()
    grid_b, trace_b = _run_probe(e1_b_profile(st), n_steps, st["k_src"], st["k_prb"])
    assert abs(float(grid_b.dt) - dt) < 1e-20
    single = e1_arm(st, e1_single_profile(st), "single", 0, trace_b, grid_b,
                    model["R_single_model"], model["R_single_model"])
    print(f"   FDTD single ramp: R_meas={single['R_meas']:.4e} ({single['R_meas_db']:.1f} dB) "
          f"model={single['R_model']:.4e} dev={single['deviation_rel']*100:.2f}% "
          f"fired={single['fired']} gates_hold={single['gates_hold']}", flush=True)
    arms = []
    for row in model["rows"]:
        arm = e1_arm(st, e1_band_profile(st, row["n_b"]), "band", row["n_b"], trace_b,
                     grid_b, row["R_model"], model["R_single_model"])
        arm["builder_matches_declared_vector"] = row["builder_matches_declared_vector"]
        arms.append(arm)
        print(f"   FDTD n_b={row['n_b']}: R_meas={arm['R_meas']:.4e} ({arm['R_meas_db']:.1f} dB) "
              f"model={arm['R_model']:.4e} dev={arm['deviation_rel']*100:.2f}% "
              f"fired={arm['fired']} bound_fired={arm['bound_fired']} "
              f"gates_hold={arm['gates_hold']} margins(ns)="
              + ", ".join(f"{k}={v:.3f}" for k, v in arm["gate_margins_ns"].items()), flush=True)
    c_meas = e1_fit_c([a["n_b"] for a in arms], [a["R_meas"] for a in arms], dzf,
                      model["R_single_model"], model["k_g_fine_per_m"],
                      E1_C_FIT_MAX_DR * st["ramp_cell_m"])
    c_dev = abs(c_meas - model["c_model_m"])
    cell.update({
        "single": single, "arms": arms, "wallclock_s": time.time() - t0,
        "c_meas_m": c_meas, "c_model_m": model["c_model_m"],
        "c_dev_m": c_dev, "c_dev_over_ramp_cell": c_dev / st["ramp_cell_m"],
        "c_window_m": E1_C_TOL_DR * st["ramp_cell_m"],
        "law_iii_fired": bool(c_dev > E1_C_TOL_DR * st["ramp_cell_m"]),
        "any_width_fired": any(a["fired"] for a in arms),
        "any_bound_fired": any(a["bound_fired"] for a in arms),
        "all_gates_hold": bool(single["gates_hold"] and all(a["gates_hold"] for a in arms)),
    })
    cell["in_law_domain"] = bool(not cell["any_width_fired"] and not cell["any_bound_fired"]
                                 and not cell["law_iii_fired"] and not single["fired"])
    print(f"   c_meas={c_meas*1e3:.3f}mm c_model={model['c_model_m']*1e3:.3f}mm "
          f"dev={c_dev*1e3:.3f}mm window={E1_C_TOL_DR*st['ramp_cell_m']*1e3:.3f}mm "
          f"law_iii_fired={cell['law_iii_fired']} in_law_domain={cell['in_law_domain']} "
          f"in_accuracy_class={st['in_accuracy_class']} wallclock={cell['wallclock_s']:.1f}s", flush=True)
    return cell


def e1_scaling_checks(cells: dict, ratios: list[float], n_lambdas: list[int]) -> list[dict]:
    """Law check (i): single-ramp reflection ratio N / 30, measured vs
    model, within E1_SCALING_TOL; the model's own exponent vs 2 reported."""
    out = []
    for ratio in ratios:
        ref = cells.get(e1_cell_key(E1_N_LAMBDA_REF, ratio))
        if ref is None:
            continue
        for n_lambda in n_lambdas:
            if n_lambda == E1_N_LAMBDA_REF:
                continue
            c = cells.get(e1_cell_key(n_lambda, ratio))
            if c is None:
                continue
            m_ratio = c["model"]["R_single_model"] / ref["model"]["R_single_model"]
            rec = {"ratio": ratio, "n_lambda": n_lambda, "ref_n_lambda": E1_N_LAMBDA_REF,
                   "model_ratio": m_ratio,
                   "model_exponent": float(np.log(m_ratio) / np.log(E1_N_LAMBDA_REF / n_lambda)),
                   "dz_over_lambda_sq_ratio": (E1_N_LAMBDA_REF / n_lambda) ** 2}
            if "single" in c and "single" in ref:
                meas = c["single"]["R_meas"] / ref["single"]["R_meas"]
                rec.update({"meas_ratio": meas, "meas_over_model": meas / m_ratio,
                            "fired": bool(abs(meas / m_ratio - 1) > E1_SCALING_TOL)})
            out.append(rec)
    return out


def run_e1(n_lambdas: list[int], ratios: list[float], widths: list[int],
           out_path: str, model_only: bool, resume: bool) -> dict:
    results = {"rfx_file": rfx.__file__, "argv": sys.argv[1:], "git_sha": _git_sha(),
               "git_dirty": _git_dirty(),
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "model_only": model_only, "widths": widths, "n_lambdas": n_lambdas,
               "ratios": ratios, "F0_Hz": F0, "sigma_t_s": SIGMA_T, "dxy_m": fx.DXY,
               "b_m": fx.B_Y, "a_m": fx.A_X,
               "windows": {"per_arm": "|R_meas - R_model| <= 0.20 R_model + 3e-5",
                           "law_i_scaling_tol": E1_SCALING_TOL,
                           "law_ii_bound_slack": E1_BOUND_SLACK,
                           "law_iii_c_tol_ramp_cells": E1_C_TOL_DR},
               "cells": {}}
    if resume:
        try:
            with open(out_path) as fh:
                prev = json.load(fh)
            if prev.get("model_only") == model_only:
                results["cells"] = prev.get("cells", {})
                results["resumed_from"] = {"git_sha": prev.get("git_sha"),
                                           "started_utc": prev.get("started_utc")}
        except (OSError, ValueError):
            pass
    t0 = time.time()
    for n_lambda in n_lambdas:
        for ratio in ratios:
            key = e1_cell_key(n_lambda, ratio)
            if key in results["cells"] and (model_only or "arms" in results["cells"][key]):
                print(f"E1 {key}: already in {out_path}, skipped (resume)", flush=True)
                continue
            results["cells"][key] = run_e1_cell(e1_setting(n_lambda, ratio), widths, model_only)
            results["wallclock_s"] = time.time() - t0
            with open(out_path, "w") as fh:
                json.dump(results, fh, indent=1)
    # law (i) aggregates over EVERY cell in the JSON, not this call's lists
    # (a --resume call per resolution would otherwise drop the earlier pairs)
    results["n_lambdas_all"] = sorted({c["setting"]["n_lambda"] for c in results["cells"].values()})
    results["ratios_all"] = sorted({c["setting"]["ratio"] for c in results["cells"].values()})
    results["law_i"] = e1_scaling_checks(results["cells"], results["ratios_all"],
                                         results["n_lambdas_all"])
    for rec in results["law_i"]:
        print(f"law (i) r={rec['ratio']} N={rec['n_lambda']}/30: model_ratio={rec['model_ratio']:.4f} "
              f"(exponent {rec['model_exponent']:.3f}) "
              + (f"meas_ratio={rec['meas_ratio']:.4f} meas/model={rec['meas_over_model']:.4f} "
                 f"fired={rec['fired']}" if "meas_ratio" in rec else "(model only)"), flush=True)
    results["wallclock_s"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out_path)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="2,4,8,16")
    ap.add_argument("--out", default="validation/research/multiband_nu/results/w6_band_builder.json")
    ap.add_argument("--f7-only", action="store_true")
    ap.add_argument("--carry-f8", action="store_true",
                    help="with --f7-only: keep the f8 block already in --out "
                         "(its FDTD rows cannot be recomputed without re-running "
                         "the runs) and record the provenance it was written with")
    ap.add_argument("--sweep", action="store_true",
                    help="E1: resolution x ratio sweep of the F8 witness (no F7)")
    ap.add_argument("--fine-cells-per-lambda", default="30",
                    help="E1: fine cells per free-space wavelength, comma list of 15/30/60")
    ap.add_argument("--ratio", default="1.4", help="E1: band ratio, comma list")
    ap.add_argument("--model-only", action="store_true", help="E1: chain model only, no FDTD")
    ap.add_argument("--resume", action="store_true", help="E1: skip cells already in --out")
    args = ap.parse_args(argv)
    widths = [int(w) for w in args.widths.split(",") if w]
    if args.sweep:
        n_lambdas = [int(n) for n in args.fine_cells_per_lambda.split(",") if n]
        ratios = [float(r) for r in args.ratio.split(",") if r]
        run_e1(n_lambdas, ratios, widths, args.out, args.model_only, args.resume)
        return
    t0 = time.time()
    results = {"rfx_file": rfx.__file__, "argv": sys.argv[1:],
               "git_sha": _git_sha(), "git_dirty": _git_dirty(),
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    carried = None
    if args.carry_f8:
        if not args.f7_only:
            ap.error("--carry-f8 only makes sense with --f7-only")
        with open(args.out) as fh:
            carried = json.load(fh)
    results["f7"] = run_f7()
    if not args.f7_only:
        results["f8"] = run_f8(widths)
    elif carried is not None:
        results["f8"] = carried["f8"]
        # The F8 rows belong to the run that MEASURED them. On a file that
        # already carries an f8_provenance, that run is the one it names --
        # not this file's top-level keys, which a previous F7-only recompute
        # already replaced. Rebuilding from the top level here would hand
        # F8's FDTD runs to whichever tree last recomputed F7.
        results["f8_provenance"] = carried.get("f8_provenance") or {
            k: carried[k] for k in ("rfx_file", "argv", "git_sha", "git_dirty",
                                    "started_utc", "wallclock_s") if k in carried}
    results["wallclock_s"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
