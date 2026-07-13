#!/usr/bin/env python3
"""Broad-E4 external comparison: rfx coaxial-LINE reflection vs openEMS FDTD.

SUPERSEDED (2026-06-08) by the MEEP path (meep_coax_line_reference.py +
build_coaxial_line_meep_broad_comparison.py): cluster openEMS v0.0.35 has no
AddCoaxialPort, and a radial AddLumpedPort cannot excite the coax TEM mode (3
runs failed: mesh-misalignment, MUR-on-PTFE instability, then weak/zero coupling
with uf_inc~6e-14). The stability fix (MUR->PML_8), the Z0 fix (dielectric to
b=SMA_OUTER_RADIUS=2.055mm) and the mesh-line snap below are KEPT for the record
and remain correct openEMS practice, but this script is not the coaxial_port
broad-E4 producer. The promoted broad-E4 evidence is the MEEP power-flux
comparison; see project_port_sparam_review memory + the manifest M74 note.

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
    """Independent openEMS coax-line |S11|. Cluster-only (needs openEMS/CSXCAD).

    Redesign (2026-06-08) — fixes the cluster blow-up + Z0 mismatch + bidirectional
    contamination of the prior version:

    * INSTABILITY: MUR on the +-z faces assumes phase velocity == c, but the PTFE
      (eps_r=2.1) TEM wave travels at c/sqrt(2.1) -> MUR-on-dielectric is unstable
      (exponential energy blow-up 5e-16 -> 2.8e13). Use PML_8 on +-z instead
      (same fix validation/crossval/05_patch_antenna.py applied for its resonance).
      PEC on the transverse faces, PML_8 on +-z. The full coax cross-section
      (pin+PTFE+shell) extends uniformly THROUGH both PML regions so PML is
      reflectionless (constant cross-section into the PML).
    * Z0 MISMATCH: the PTFE must fill the annulus OUT TO b = SMA_OUTER_RADIUS;
      the PEC outer conductor is a TUBE (>=2 cells thick) OUTSIDE b so the
      staircased shell is sealed. Build by priority: outer metal solid cylinder
      r=shell_outer (prio 1) < ptfe solid cylinder r=b (prio 5) < pin solid
      cylinder r=a (prio 10). Net: pin(r<a) / PTFE(a<r<b) / PEC(b<r<shell_outer)
      / vacuum. Z0(a,b,eps=2.1)=48.6 ohm = the port Z_ref = the rfx value.
    * BIDIRECTIONAL/STUB CONTAMINATION: the lumped port's uf_inc/uf_ref split
      (uf_inc=0.5*(uf_tot+if_tot*Z_ref)) only stays clean if NOTHING reflects back
      from the +z side. Place the feed a few cells BELOW the +z PML inner edge (in
      the clean line, PML behind it) and the DUT FAR at the -z end; the +z wave is
      absorbed by the +z PML, so uf_ref carries only the -z DUT reflection.
    """
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS

    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    unit = 1.0e-3
    dx_mm = 0.30                            # mm: >=4 cells across the b-a annulus
    a = SMA_PIN_RADIUS / unit              # pin radius (mm) = 0.635
    b = SMA_OUTER_RADIUS / unit           # dielectric/inner-shell radius (mm) = 2.055
    shell_outer = b + 2.0 * dx_mm         # outer shell wall: >=2 cells thick -> sealed
    domain_x = domain_y = 8.0             # mm: ~2*shell_outer + ~2mm margin. The
    #   8mm box gives transverse TE10 cutoff c/(2*8mm)=18.7 GHz > 12 GHz band top,
    #   so there is NO box-mode overmoding (the old 20mm box was overmoded at
    #   7.5 GHz, contaminating 7.5-12 GHz). Do NOT re-widen the box.
    domain_z = 30.0                        # mm
    cx = cy = domain_x / 2.0              # 4.0 mm (coax centered)
    open_gap = 3.0                         # mm: open-pin retract above z_dut
    # PML_8 = 8 cells = 2.4 mm at each +-z end. Feed a few cells BELOW the +z PML
    # inner edge (in the clean line, PML behind it); DUT above the -z PML inner
    # edge; ~19 mm clean line between them. Ports must NOT sit inside the PML.
    z_feed = domain_z - 5.0               # 25.0 mm: feed near +z (PML behind it)
    z_dut = 6.0                            # 6.0 mm: DUT near -z

    # PML lets the field DECAY -> stop early once it is -50 dB down (NrTS caps it).
    fdtd = openEMS(NrTS=n_steps, EndCriteria=1.0e-5)
    fdtd.SetGaussExcite(8.0e9, 6.0e9)
    # PEC transverse, PML_8 on +-z (MUR-on-dielectric was the blow-up cause).
    fdtd.SetBoundaryCond(["PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"])
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)

    # The radial (x-directed) lumped feed only couples if its edge coordinates
    # coincide with mesh lines. The SMA radii (a=0.635, b=2.055) are not round, so
    # a plain uniform linspace leaves the port edges BETWEEN lines -> openEMS drops
    # the excitation ("Unused primitive ... port_excite_1", uf_inc=0, S11=NaN). Pin
    # every conductor radius and port plane to a fixed mesh line, then
    # SmoothMeshLines fills the interior at <= dx_mm.
    fixed = {
        "x": [0.0, domain_x, cx,
              cx - a, cx + a, cx - b, cx + b,
              cx - shell_outer, cx + shell_outer],
        "y": [0.0, domain_y, cy,
              cy - a, cy + a, cy - b, cy + b,
              cy - shell_outer, cy + shell_outer],
        "z": [0.0, domain_z, z_feed, z_dut,
              z_dut + dx_mm, z_dut + open_gap],  # cap cell + open-pin retract plane
    }
    for axis in "xyz":
        mesh.AddLine(axis, sorted(set(fixed[axis])))
    mesh.SmoothMeshLines("all", dx_mm, 1.4)

    # Coax (pin+PTFE+shell) spans the FULL z = 0..domain_z, THROUGH both PMLs, so
    # the PML sees a constant cross-section and stays reflectionless. Build by
    # priority: outer metal (prio 1) < ptfe (prio 5) < pin (prio 10).
    outer = csx.AddMetal("outer")
    outer.AddCylinder([cx, cy, 0.0], [cx, cy, domain_z], radius=shell_outer, priority=1)
    ptfe = csx.AddMaterial("ptfe", epsilon=2.1)
    ptfe.AddCylinder([cx, cy, 0.0], [cx, cy, domain_z], radius=b, priority=5)
    pin = csx.AddMetal("pin")
    # For non-open terms the pin spans z_dut..domain_z; for open it retracts above
    # z_dut by open_gap, leaving a below-cutoff PTFE+shell stub -> |Gamma|~1.
    pin_z_lo = (z_dut + open_gap) if term == "open" else z_dut
    pin.AddCylinder([cx, cy, pin_z_lo], [cx, cy, domain_z], radius=a, priority=10)

    if term == "short":                     # PEC disk pin->shell at z_dut
        cap = csx.AddMetal("short_cap")
        cap.AddCylinder([cx, cy, z_dut], [cx, cy, z_dut + dx_mm], radius=shell_outer, priority=20)

    # Feed lumped port (Z0) along +x radial edge pin(a) -> dielectric/shell(b) at
    # z_feed.  Outer edge at cx+b (the conductor boundary), NOT cx+shell_outer.
    feed = fdtd.AddLumpedPort(1, float(Z0), [cx + a, cy, z_feed],
                              [cx + b, cy, z_feed], "x", excite=1.0)
    if term in ("matched",) or term.startswith("load"):
        R_load = float(R) if R is not None else float(Z0)
        fdtd.AddLumpedPort(2, R_load, [cx + a, cy, z_dut],
                           [cx + b, cy, z_dut], "x", excite=0.0)

    fdtd.Run(str(sim_dir), verbose=0, cleanup=True)
    feed.CalcPort(str(sim_dir), np.asarray(freqs, dtype=float))
    uf_inc = np.asarray(feed.uf_inc, dtype=np.complex128)
    # Fail loud and fast: a port that never coupled (mesh-misaligned excitation)
    # yields uf_inc=0 -> S11=0/0=NaN.  Catch it on the FIRST termination instead
    # of running the whole panel to silent NaN.
    inc_peak = float(np.max(np.abs(uf_inc))) if uf_inc.size else 0.0
    if not np.isfinite(inc_peak) or inc_peak <= 1.0e-9:
        raise RuntimeError(
            f"openEMS feed port injected ~0 incident energy (max|uf_inc|={inc_peak:.3e}) "
            f"for term={term!r}: excitation did not couple. Check that the radial port "
            f"edges land on mesh lines (SmoothMeshLines/fixed-line snap)."
        )
    s11 = np.asarray(feed.uf_ref / uf_inc, dtype=np.complex128)
    # Fail loud on a blown-up / non-physical field: a stable lossless line has
    # |Gamma| <= 1; anything >> 1 (or non-finite) means the run diverged.
    s11_peak = float(np.max(np.abs(s11))) if s11.size else float("nan")
    if not np.all(np.isfinite(s11)) or s11_peak > 2.0:
        raise RuntimeError(
            f"openEMS coax ref non-physical/unstable |S11| max={s11_peak:.3e} "
            f"for term={term!r} (field blew up?)"
        )
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
            max_abs_tol=0.20, mean_abs_tol=0.15, max_mag_abs_tol=0.08, mean_mag_abs_tol=0.05,
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
            f"line). Cross-solver tolerance max |Γ| diff {0.08}."),
        "cross_solver_max_mag_abs_diff": round(worst_max, 4),
        "cross_solver_mean_mag_abs_diff": round(worst_mean, 4),
        "tolerances": {"max_mag_abs_tol": 0.08, "mean_mag_abs_tol": 0.05},
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
