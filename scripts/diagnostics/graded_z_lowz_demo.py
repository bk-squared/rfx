#!/usr/bin/env python3
"""graded-z efficiency demo — wide low-Z escape hatch (the ONE R2-tight attempt).

Pre-declaration (fixture, metrics, falsifiers, attempt-spend rule):
    docs/design_notes/graded_z_lowz_demo_predeclaration.md
That note is committed BEFORE this script ever ran FDTD. Do not edit the
thresholds here — they are the committed MSL thru envelope gates
(scripts/diagnostics/build_msl_broad_e5_envelope.py THRESH) plus the two
deterministic efficiency gates declared in the note.

Fixture: matched thru on the committed thru-fixture class, re-dimensioned
to the documented escape hatch — W = 6*h_sub = 1524 um (~25.4 Ohm HJ),
dx = W/8 = 190.5 um, eps_r 3.38, h_sub 254 um, L = 10 mm.

Arms (identical physical geometry, both on the NU code path):
  A: uniform-z baseline, dz = h_sub/4 everywhere (aligned mesh).
  B: graded-z from the PRODUCTION profile machinery
     (_make_dz_profile via analyze_features — the exact auto_configure
     step-4 call; no hand-tuned profile).

Usage:
  graded_z_lowz_demo.py --pilot          # timing-only pilot (arm B, np=2)
  graded_z_lowz_demo.py                  # the full attempt, both arms
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rfx.api import Simulation                          # noqa: E402
from rfx.auto_config import _make_dz_profile, analyze_features  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box                        # noqa: E402
from rfx.microstrip import microstrip_impedance         # noqa: E402

C0 = 299792458.0

# ---- fixture constants (pre-declared; do not tune) ----
EPS_R = 3.38
H_SUB = 254e-6
W_TRACE = 6 * H_SUB          # 1524 um — the escape hatch
DX = W_TRACE / 8             # 190.5 um — the escape hatch
T_TRACE = H_SUB / 4          # 63.5 um physical trace thickness, both arms
L_LINE = 10e-3
PORT_MARGIN = 2e-3
F_MAX = 5e9
GATE_F_LO, GATE_F_HI = 3.0e9, 4.5e9
N_FREQS = 30
NUM_PERIODS = 12.0           # committed CPU thru-gate precedent

LX = L_LINE + 2 * PORT_MARGIN
LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
LZ_NOM = H_SUB + 1.5e-3

Z0_HJ, EPS_EFF_HJ = microstrip_impedance(W_TRACE, H_SUB, EPS_R)

# ---- committed envelope thresholds (provenance in the pre-declaration) ----
THRU_MAX_S11 = 0.10
THRU_MEAN_S21_MIN = 0.95
THRU_Z0_REL_ERR_MED = 0.05
EPS_EFF_BOUNDS = (1.0, EPS_R)
ZRATIO_MIN = 2.0             # F5: below this the ~3x z-savings claim fails
SETTLING_DB_MAX = -40.0      # validity precondition, not a falsifier


def _geometry_tuples():
    y_c = LY / 2.0
    sub = Box((0.0, 0.0, 0.0), (LX, LY, H_SUB))
    trace = Box((0.0, y_c - W_TRACE / 2.0, H_SUB),
                (LX, y_c + W_TRACE / 2.0, H_SUB + T_TRACE))
    return sub, trace


def production_graded_profile() -> np.ndarray:
    """The exact auto_configure step-4 call, on this fixture's geometry."""
    sub, trace = _geometry_tuples()
    feats = analyze_features(
        [(sub, "sub"), (trace, "pec")],
        {"sub": {"eps_r": EPS_R}, "pec": {"sigma": 1e30}},
    )
    return _make_dz_profile(feats.z_features, LZ_NOM, DX)


def build_arm(dz_profile: np.ndarray) -> Simulation:
    lz = float(np.sum(dz_profile))
    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, LY, lz),
        dx=DX,
        cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        dz_profile=np.asarray(dz_profile, dtype=float),
    )
    sim.add_material("ro4003c", eps_r=EPS_R)
    sub, trace = _geometry_tuples()
    sim.add(sub, material="ro4003c")
    sim.add(trace, material="pec")
    y_c = LY / 2.0
    sim.add_msl_port(position=(PORT_MARGIN, y_c, 0.0),
                     width=W_TRACE, height=H_SUB, direction="+x",
                     impedance=float(Z0_HJ))
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_c, 0.0),
                     width=W_TRACE, height=H_SUB, direction="-x",
                     impedance=float(Z0_HJ))
    return sim


def run_arm(name: str, dz_profile: np.ndarray, num_periods: float,
            dump_path: Path | None, timing_only: bool = False) -> dict:
    import warnings as _warnings

    sim = build_arm(dz_profile)

    # Preflight ON and fully surfaced (never suppressed).
    print(f"\n===== arm {name}: preflight =====", flush=True)
    with _warnings.catch_warnings(record=True) as wrec:
        _warnings.simplefilter("always")
        pf = sim.preflight()
    for w in wrec:
        print(f"[preflight-warn:{name}] {w.category.__name__}: {w.message}",
              flush=True)
    print(f"[preflight:{name}] return: {pf}", flush=True)

    t0 = time.time()
    with _warnings.catch_warnings(record=True) as wrec2:
        _warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(
            n_freqs=N_FREQS,
            num_periods=num_periods,
            enforce_passivity=False,   # issue #470: diagnostics see RAW
            raw_3probe_dump_path=str(dump_path) if dump_path else None,
        )
    wall = time.time() - t0
    for w in wrec2:
        print(f"[run-warn:{name}] {w.category.__name__}: {w.message}",
              flush=True)

    if timing_only:
        # Declared pilot contract: wallclock only — S/Z0 never read.
        del res
        return {"arm": name, "wallclock_s": round(wall, 1),
                "timing_only": True}

    S = np.asarray(res.S)
    Z0 = np.asarray(res.Z0)
    beta = np.asarray(res.beta)
    freqs = np.asarray(res.freqs, dtype=float)
    reliable = (np.asarray(res.reliable) if res.reliable is not None
                else np.ones((S.shape[0], S.shape[2]), dtype=bool))
    settling = (np.asarray(res.settling_db) if res.settling_db is not None
                else None)

    gate = (freqs >= GATE_F_LO) & (freqs <= GATE_F_HI)
    rel_ok = np.all(reliable, axis=0)
    use = gate & rel_ok

    out = {
        "arm": name,
        "wallclock_s": round(wall, 1),
        "nz": int(len(dz_profile)),
        "dz_min_um": float(np.min(dz_profile) * 1e6),
        "lz_mm": float(np.sum(dz_profile) * 1e3),
        "n_gate_bins": int(np.sum(gate)),
        "n_gate_bins_reliable": int(np.sum(use)),
        "settling_db": None if settling is None else [float(x) for x in settling],
        "assembly": res.assembly,
    }
    if not np.any(use):
        out["invalid"] = "entire gate band unreliable"
        return out

    s11 = np.abs(S[0, 0, :])
    s21 = np.abs(S[1, 0, :])
    z0_re = Z0[0, :].real
    omega = 2 * np.pi * freqs
    eps_eff = (np.real(beta) * C0 / np.where(omega > 0, omega, np.nan)) ** 2

    out.update({
        "max_s11_gate": float(np.max(s11[use])),
        "mean_s21_gate": float(np.mean(s21[use])),
        "z0_med_gate": float(np.median(z0_re[use])),
        "z0_rel_err_med": float(abs(np.median(z0_re[use]) - Z0_HJ) / Z0_HJ),
        "eps_eff_gate_min": float(np.nanmin(eps_eff[use])),
        "eps_eff_gate_max": float(np.nanmax(eps_eff[use])),
        "trace": [
            {"f_ghz": round(f / 1e9, 3), "s11": round(float(a), 5),
             "s21": round(float(b), 5), "re_z0": round(float(z), 2),
             "eps_eff": round(float(e), 4), "reliable": bool(r)}
            for f, a, b, z, e, r in zip(freqs, s11, s21, z0_re, eps_eff, rel_ok)
        ],
    })

    # Boundedness protocol + sub-item (b) witness from the raw phasor dump.
    if dump_path is not None and dump_path.exists():
        with np.load(dump_path, allow_pickle=True) as npz:
            raw_v = npz["raw_v"]     # (n_driven, n_ports, n_probes_max, n_freqs)
            raw_i1 = npz["raw_i1"]   # (n_driven, n_ports, n_freqs)
        zin = raw_v[0, 0, 0, :] / raw_i1[0, 0, :]
        z0e = Z0[0, :]
        s11_reref = (zin - z0e) / (zin + z0e)
        out["re_vi_min_gate"] = float(np.min(np.real(zin)[use]))
        out["s11_reref_extracted_z0_gate_max"] = float(
            np.max(np.abs(s11_reref)[use]))
        out["s11_reref_trace"] = [
            {"f_ghz": round(f / 1e9, 3), "re_zin": round(float(z.real), 2),
             "im_zin": round(float(z.imag), 2),
             "abs_s11_reref": round(float(a), 5)}
            for f, z, a in zip(freqs, zin, np.abs(s11_reref))
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="timing-only pilot: arm B, num_periods=2; "
                         "S/Z0 outputs are neither printed nor saved")
    ap.add_argument("--output",
                    default="docs/design_notes/graded_z_lowz_demo_results.json")
    args = ap.parse_args()

    prof_b = production_graded_profile()
    nz_b = len(prof_b)
    nz_a = int(round(float(np.sum(prof_b)) / (H_SUB / 4)))
    prof_a = np.full(nz_a, H_SUB / 4)

    def dt_of(dzmin):
        return 0.99 / (C0 * np.sqrt(2 / DX**2 + 1 / dzmin**2))

    dt_a, dt_b = dt_of(H_SUB / 4), dt_of(float(np.min(prof_b)))
    z_ratio = nz_a / nz_b
    z_cost_ratio = (nz_b / dt_b) / (nz_a / dt_a)

    print(f"Z0_HJ={Z0_HJ:.4f} ohm  eps_eff_HJ={EPS_EFF_HJ:.4f}")
    print(f"arm A: nz={nz_a} dz=63.5um  dt={dt_a:.4g}")
    print(f"arm B: nz={nz_b} dz_min={np.min(prof_b)*1e6:.2f}um dt={dt_b:.4g}")
    print(f"F5 z-cell ratio A/B = {z_ratio:.3f} (falsified if < {ZRATIO_MIN})")
    print(f"F6 z-cost ratio B/A = {z_cost_ratio:.3f} (falsified if >= 1.0)")
    print(f"graded profile (um): {np.round(prof_b*1e6, 2).tolist()}")

    if args.pilot:
        print("\n[pilot] timing-only run, arm B, num_periods=2 "
              "(outputs discarded unread)")
        t0 = time.time()
        _ = run_arm("B-pilot", prof_b, 2.0, None, timing_only=True)
        print(f"[pilot] wallclock {time.time()-t0:.1f}s for num_periods=2; "
              f"full run scales ~x{NUM_PERIODS/2:.0f} per arm")
        return 0

    scratch = REPO / ".omx"
    scratch.mkdir(exist_ok=True)
    res_a = run_arm("A-uniform-fine", prof_a, NUM_PERIODS,
                    scratch / "graded_z_demo_armA_dump.npz")
    res_b = run_arm("B-graded", prof_b, NUM_PERIODS,
                    scratch / "graded_z_demo_armB_dump.npz")

    verdicts = {}
    for tag, r in (("A", res_a), ("B", res_b)):
        if "invalid" in r or r.get("max_s11_gate") is None:
            verdicts[f"F1_{tag}"] = verdicts[f"F2_{tag}"] = \
                verdicts[f"F3_{tag}"] = verdicts[f"F4_{tag}"] = "INVALID"
            continue
        verdicts[f"F1_{tag}_max_s11"] = (
            "FIRED" if r["max_s11_gate"] >= THRU_MAX_S11 else "held")
        verdicts[f"F2_{tag}_mean_s21"] = (
            "FIRED" if r["mean_s21_gate"] <= THRU_MEAN_S21_MIN else "held")
        verdicts[f"F3_{tag}_z0"] = (
            "FIRED" if r["z0_rel_err_med"] >= THRU_Z0_REL_ERR_MED else "held")
        verdicts[f"F4_{tag}_eps_eff"] = (
            "FIRED" if (r["eps_eff_gate_min"] <= EPS_EFF_BOUNDS[0]
                        or r["eps_eff_gate_max"] >= EPS_EFF_BOUNDS[1])
            else "held")
    verdicts["F5_z_cell_ratio"] = (
        "FIRED" if z_ratio < ZRATIO_MIN else "held")
    verdicts["F6_z_cost_ratio"] = (
        "FIRED" if z_cost_ratio >= 1.0 else "held")

    settling_ok = all(
        r.get("settling_db") is not None
        and max(r["settling_db"]) <= SETTLING_DB_MAX
        for r in (res_a, res_b))
    verdicts["settling_precondition"] = "OK" if settling_ok else "VIOLATED"

    result = {
        "fixture": {
            "eps_r": EPS_R, "h_sub_m": H_SUB, "w_trace_m": W_TRACE,
            "t_trace_m": T_TRACE, "dx_m": DX, "l_line_m": L_LINE,
            "lx_m": LX, "ly_m": LY, "lz_nominal_m": LZ_NOM,
            "z0_hj_ohm": float(Z0_HJ), "eps_eff_hj": float(EPS_EFF_HJ),
            "f_max_hz": F_MAX, "gate_band_hz": [GATE_F_LO, GATE_F_HI],
            "num_periods": NUM_PERIODS, "n_freqs": N_FREQS,
        },
        "efficiency": {
            "nz_A": nz_a, "nz_B": nz_b,
            "dt_A_s": float(dt_a), "dt_B_s": float(dt_b),
            "z_cell_ratio_A_over_B": float(z_ratio),
            "z_cost_ratio_B_over_A": float(z_cost_ratio),
            "graded_profile_um": np.round(prof_b * 1e6, 3).tolist(),
        },
        "arm_A": res_a,
        "arm_B": res_b,
        "verdicts": verdicts,
    }
    out = REPO / args.output
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("\n===== VERDICTS =====")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
