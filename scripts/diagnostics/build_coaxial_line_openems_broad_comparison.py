#!/usr/bin/env python3
"""Broad-E4 external comparison: rfx coaxial-LINE reflection vs openEMS FDTD.

This is the broad, calibrated-reference-plane successor to the narrow
``build_coaxial_gap_openems_sparameter_comparison.py`` (which is E4-enabling
only). It compares the PROMOTED rfx API ``Simulation.compute_coaxial_line_reflection``
against an independent openEMS full-wave coax line, for the canonical
termination panel (short/open/matched + resistive loads) across a broad band.

Both solvers model the SAME coax line (SMA pin/PTFE/shell) terminated at one
end; |Γ| is reference-plane-independent for a lossless line, so the magnitude
comparison is robust to the (different) reference planes the two solvers use.
The resistive-load cases (|Γ|≈0.23-0.35) are the non-trivial cross-solver
discriminators; short/open pin |Γ|=1; matched pins |Γ|≈0.

openEMS termination modelling:
  short   -> PEC cap (metal disk) shorting pin to shell at the far end
  open    -> pin ends short of the far wall (gap)
  matched -> a second AddLumpedPort (excite=0) with Z_ref = Z0 (acts as a load)
  load(R) -> a second AddLumpedPort (excite=0) with Z_ref = R

IMPORTANT: the openEMS half REQUIRES openEMS/CSXCAD and is intended to run on
the ``port_external_coaxial`` VESSL shard (remilab-c0). It is NOT exercised in
local CI. Use ``--stub-openems`` to validate the rfx + comparison + artifact
code paths locally (the stub substitutes the EXACT analytic |Γ| for openEMS, so
a local stub run checks plumbing, NOT the independent full-wave reference).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)
from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import (
    coaxial_tem_characteristic_impedance,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
)

DEFAULT_FREQS_HZ = np.linspace(4.0e9, 12.0e9, 9)
Z0 = coaxial_tem_characteristic_impedance(SMA_PIN_RADIUS, SMA_OUTER_RADIUS)

# (name, dut_impedance_ohm or None, analytic |Gamma|)
TERMINATIONS = [
    ("short", None, 1.0),
    ("open", None, 1.0),
    ("matched", None, 0.0),
    ("load25", 25.0, abs((25.0 - Z0) / (25.0 + Z0))),
    ("load100", 100.0, abs((100.0 - Z0) / (100.0 + Z0))),
]


def _ensure_openems_numpy_compat() -> None:
    for name, value in {"float": float, "int": int, "complex": complex}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _oneport(mag_or_complex: np.ndarray) -> np.ndarray:
    arr = np.asarray(mag_or_complex, dtype=np.complex128)
    s = np.zeros((1, 1, arr.size), dtype=np.complex128)
    s[0, 0, :] = arr
    return s


def _rfx_line_abs_gamma(term: str, R: float | None, *, freqs: np.ndarray,
                        freq_max: float, n_steps: int):
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=freq_max, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    kw: dict[str, Any] = dict(
        termination=("matched" if term == "matched" or term.startswith("load") else term),
        n_steps=n_steps, freqs=np.asarray(freqs, dtype=float),
    )
    if R is not None:
        kw["dut_impedance"] = R
    res = sim.compute_coaxial_line_reflection(**kw)
    return np.abs(np.asarray(res.s11)), float(np.max(res.recurrence_residual)), res.status


def _run_openems_line_reference(term: str, R: float | None, *, sim_dir: Path,
                                n_steps: int, freqs: np.ndarray) -> np.ndarray:
    """Independent openEMS coax-line |S11|. Cluster-only (needs openEMS/CSXCAD)."""
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS

    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    unit = 1.0e-3
    dx_m = 0.30e-3                         # ~4-5 cells across the SMA annulus
    cx = cy = 10.0                          # mm
    z_lo, z_hi = 4.0, 34.0                  # mm: coax line extent
    z_feed, z_dut = 30.0, 6.0               # mm: feed port near +z, DUT near -z
    domain = (20.0, 20.0, 38.0)             # mm

    fdtd = openEMS(NrTS=n_steps, EndCriteria=0)
    fdtd.SetGaussExcite(8.0e9, 6.0e9)
    fdtd.SetBoundaryCond(["PEC", "PEC", "PEC", "PEC", "MUR", "MUR"])  # absorb +-z
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)
    for axis, length in zip("xyz", domain):
        n = int(round((length * 1e-3) / dx_m))
        mesh.AddLine(axis, np.linspace(0.0, length, n + 1))

    out_r = SMA_OUTER_RADIUS / unit
    shell_in = out_r - dx_m / unit
    pin_r = SMA_PIN_RADIUS / unit
    # Outer PEC shell via priority: metal(out) < ptfe(shell_in) < pin
    outer = csx.AddMetal("outer")
    outer.AddCylinder([cx, cy, z_lo], [cx, cy, z_hi], radius=out_r, priority=1)
    ptfe = csx.AddMaterial("ptfe", epsilon=2.1)
    ptfe.AddCylinder([cx, cy, z_lo], [cx, cy, z_hi], radius=shell_in, priority=5)
    pin = csx.AddMetal("pin")
    pin_z_lo = z_dut if term != "open" else (z_dut + 3.0)   # open: pin retracts
    pin.AddCylinder([cx, cy, pin_z_lo], [cx, cy, z_hi], radius=pin_r, priority=10)

    if term == "short":                     # PEC cap pin->shell at z_dut
        cap = csx.AddMetal("short_cap")
        cap.AddCylinder([cx, cy, z_dut], [cx, cy, z_dut + dx_m / unit], radius=out_r, priority=20)

    # Feed lumped port (Z0) along +x radial edge pin->shell at z_feed.
    feed = fdtd.AddLumpedPort(1, float(Z0), [cx + pin_r, cy, z_feed],
                              [cx + shell_in, cy, z_feed], "x", excite=1.0)
    if term in ("matched",) or term.startswith("load"):
        R_load = float(R) if R is not None else float(Z0)
        fdtd.AddLumpedPort(2, R_load, [cx + pin_r, cy, z_dut],
                           [cx + shell_in, cy, z_dut], "x", excite=0.0)

    fdtd.Run(str(sim_dir), verbose=0, cleanup=True)
    feed.CalcPort(str(sim_dir), np.asarray(freqs, dtype=float))
    s11 = np.asarray(feed.uf_ref / feed.uf_inc, dtype=np.complex128)
    return np.abs(s11)


def build(output_dir: Path, *, freqs: np.ndarray, freq_max: float, rfx_n_steps: int,
          openems_n_steps: int, stub_openems: bool,
          terms: list[tuple[str, float | None, float]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_term: list[dict[str, Any]] = []
    worst_max = 0.0
    worst_mean = 0.0
    all_passed = True
    rfx_resid_max = 0.0
    for name, R, gamma_an in terms:
        rfx_mag, resid, status = _rfx_line_abs_gamma(
            name, R, freqs=freqs, freq_max=freq_max, n_steps=rfx_n_steps)
        rfx_resid_max = max(rfx_resid_max, resid)
        if stub_openems:
            ext_mag = np.full(freqs.shape, float(gamma_an))   # analytic stand-in
        else:
            ext_mag = _run_openems_line_reference(
                name, R, sim_dir=output_dir / f"openems_{name}_tmp",
                n_steps=openems_n_steps, freqs=freqs)
        cand = output_dir / f"rfx_{name}.npz"
        ref = output_dir / f"openems_{name}.npz"
        np.savez(cand, freqs_hz=freqs, s_params=_oneport(rfx_mag))
        np.savez(ref, freqs_hz=freqs, s_params=_oneport(ext_mag))
        cmp = compare_sparameter_datasets(
            load_sparameter_dataset(cand), load_sparameter_dataset(ref),
            terms="S11", comparison_mode="magnitude",
            max_abs_tol=0.20, mean_abs_tol=0.15, max_mag_abs_tol=0.10, mean_mag_abs_tol=0.06,
        )
        worst_max = max(worst_max, cmp["summary"]["max_mag_abs_diff"])
        worst_mean = max(worst_mean, cmp["summary"]["mean_mag_abs_diff_max_over_terms"])
        all_passed = all_passed and cmp["status"] == "passed" and status == "passed"
        per_term.append(dict(termination=name, dut_impedance=R, gamma_analytic_mag=round(gamma_an, 4),
                             rfx_abs_gamma=[round(float(x), 4) for x in rfx_mag],
                             openems_abs_gamma=[round(float(x), 4) for x in ext_mag],
                             max_mag_abs_diff=round(cmp["summary"]["max_mag_abs_diff"], 4),
                             rfx_recurrence_residual_max=round(resid, 5),
                             rfx_status=status, term_status=cmp["status"]))

    payload: dict[str, Any] = {
        "schema": "rfx.coaxial_line_openems_broad_comparison",
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        # Override the compare helper's default "E4-enabling": this is a broad,
        # calibrated-reference-plane comparison of the PROMOTED line API against an
        # independent full-wave solver across a termination panel and frequency band.
        "evidence_level": "E4-broad-coaxial-line-termination-openems-fdtd-comparison",
        "claim": (
            f"rfx Simulation.compute_coaxial_line_reflection |Γ| agrees with an independent "
            f"openEMS full-wave coax line to <= {worst_max:.3f} (mean <= {worst_mean:.3f}) across a "
            f"broad {freqs[0]/1e9:.0f}-{freqs[-1]/1e9:.0f} GHz band for the short/open/matched and "
            f"resistive (25/100Ω) termination panel; rfx single-TEM-mode recurrence residual "
            f"<= {rfx_resid_max:.4f}."),
        "claim_scope": (
            "broad external full-wave cross-solver comparison of the promoted rfx coaxial_port "
            "line reflection API vs openEMS over a frequency axis (4-12 GHz) and a termination "
            "panel (short/open/matched + resistive 25/100Ω, |Γ| spanning 0.23-1.0) on a matched "
            "coax line; |Γ| magnitude comparison (reference-plane independent for the lossless "
            f"line). Cross-solver tolerance max |Γ| diff {0.10}."),
        "cross_solver_max_mag_abs_diff": round(worst_max, 4),
        "cross_solver_mean_mag_abs_diff": round(worst_mean, 4),
        "tolerances": {"max_mag_abs_tol": 0.10, "mean_mag_abs_tol": 0.06},
        "rfx_recurrence_residual_max": round(rfx_resid_max, 5),
        "per_termination": per_term,
        "stub_openems": bool(stub_openems),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if stub_openems:
        payload["status"] = "stub"   # never claim a real external pass from the stub
        payload["evidence_level"] = "E4-enabling-stub-local-plumbing-check"
        payload["claim_scope"] += " STUB run: openEMS replaced by analytic |Γ| (plumbing only)."
    out_json = output_dir / "coaxial_line_openems_broad_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=".omx/physics-gate/latest-coaxial-line-openems-broad")
    p.add_argument("--rfx-n-steps", type=int, default=5000)
    p.add_argument("--openems-n-steps", type=int, default=60000)
    p.add_argument("--freq-max", type=float, default=40.0e9)
    p.add_argument("--stub-openems", action="store_true",
                   help="local plumbing check: substitute analytic |Γ| for openEMS")
    p.add_argument("--only", default="", help="comma-separated termination names to run")
    args = p.parse_args(argv)
    terms = TERMINATIONS
    if args.only:
        keep = set(args.only.split(","))
        terms = [t for t in TERMINATIONS if t[0] in keep]
    payload = build(_repo_path(args.output_dir), freqs=DEFAULT_FREQS_HZ, freq_max=args.freq_max,
                    rfx_n_steps=args.rfx_n_steps, openems_n_steps=args.openems_n_steps,
                    stub_openems=args.stub_openems, terms=terms)
    print(f"status={payload['status']} evidence_level={payload['evidence_level']} "
          f"cross_solver_max_mag_abs_diff={payload['cross_solver_max_mag_abs_diff']}")
    return 0 if payload["status"] in ("passed", "stub") else 1


if __name__ == "__main__":
    raise SystemExit(main())
