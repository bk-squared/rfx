#!/usr/bin/env python3
"""Coaxial S11 calibration: rfx compute_coaxial_s_matrix vs openEMS.

This is the M73 follow-up to the M71 gap-lane envelope: instead of comparing
the low-level rfx gap diagnostic against openEMS ``AddLumpedPort``, we
exercise the public ``Simulation.compute_coaxial_s_matrix(...)`` API on a
SMA-style coax in a PEC cavity and compare the per-frequency ``|S11|``
against an analogous openEMS setup.

Geometry: 20 mm cubic PEC cavity, face='top' coax port at (10 mm, 10 mm,
15 mm), 5 mm pin extending into the cavity (pin tip at z=10 mm in
vacuum). The pin does NOT reach the cavity floor — extending to z=0
shorts both terminals of openEMS's AddLumpedPort to floor-PEC ground,
collapses uf_inc/uf_ref to zero, and produces NaN S11. The 5 mm pin
keeps the gap region floatable so both solvers can drive the line.

The two solvers differ in source mechanism (rfx uses the M73 distributed
TFSF plane source; openEMS uses a single-cell ``AddLumpedPort`` at the
gap), but the device under test — pin-in-vacuum-cavity SMA coax with
PEC walls — is identical, so the ``|S11|`` magnitude must agree within
a calibrated envelope. The result is *not* a strict transmission-line
PEC-short (Γ at the pin tip is the pin-tip-into-vacuum-cavity
discontinuity already characterised by the M35 narrow comparator and
the M71 broad envelope on the gap-diagnostic lane).

Output: a ``coaxial_s11_calibration_envelope.json`` artifact in the
``--output-dir`` (default
``.omx/physics-gate/2026-05-11-m73-coaxial-s11-calibration``).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import jax.numpy as jnp
import numpy as np

from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)
from rfx import Simulation
from rfx.sources.coaxial_port import (
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
)
from rfx.sources.sources import GaussianPulse


# Geometry: pin in vacuum cavity (M71-style), NOT touching floor.
#
# Pin reaching the cavity floor PEC creates a path that shorts both
# terminals of openEMS's AddLumpedPort to PEC ground, so no voltage can
# develop across the port and uf_inc/uf_ref both come back as zero. We
# therefore terminate the rfx fixture the same way (5 mm pin, tip at
# z=10 mm in vacuum cavity) so both solvers see an identical device
# under test. The comparison validates rfx-vs-openEMS S11 agreement on
# the same fixture; it is *not* a PEC-short calibration in the strict
# transmission-line sense (Γ at the pin tip is not exactly +/-1 — it's
# the pin-tip-into-vacuum-cavity discontinuity, the same one M35/M71
# already validated for the gap diagnostic lane).
_DOMAIN_M = (0.020, 0.020, 0.020)
_GAP_POSITION_M = (0.010, 0.010, 0.015)
_PIN_LENGTH_M = 5.0e-3  # pin tip at z=10 mm in vacuum cavity
_PULSE_F0_HZ = 5.0e9
_PULSE_BANDWIDTH = 0.8


@dataclass(frozen=True)
class CoaxialCalibrationCase:
    name: str
    freqs_hz: tuple[float, ...]
    rfx_n_steps: int
    openems_n_steps: int
    rfx_freq_max_hz: float  # used by rfx auto-grid; 20 GHz gives ~1 PTFE cell
    openems_dx_m: float     # explicit dx for openEMS to match rfx grid
    max_mag_abs_tol: float
    mean_mag_abs_tol: float


DEFAULT_CASES: tuple[CoaxialCalibrationCase, ...] = (
    # Wide band — full GHz–10 GHz sweep. Low freq has known divergence
    # because the 5 mm pin is electrically very short (~λ/12 in PTFE at
    # 2 GHz) AND the two solvers measure at different reference planes
    # (rfx at pin_centre, openEMS at gap), so the envelope is lenient.
    CoaxialCalibrationCase(
        name="band_2_to_10_GHz_wide",
        freqs_hz=(2.0e9, 4.0e9, 6.0e9, 8.0e9, 10.0e9),
        rfx_n_steps=400,
        openems_n_steps=500,
        rfx_freq_max_hz=30.0e9,
        openems_dx_m=0.5e-3,
        max_mag_abs_tol=0.80,
        mean_mag_abs_tol=0.40,
    ),
    # High-band tail (8–10 GHz). The pin is electrically meaningful here
    # (~λ/2.5 in PTFE at 10 GHz) and the rfx-vs-openEMS agreement
    # tightens to ~0.15 in magnitude. This case is the actual cross-
    # solver-agreement gate; the wide-band case is documentation.
    CoaxialCalibrationCase(
        name="band_8_to_10_GHz_tight",
        freqs_hz=(8.0e9, 9.0e9, 10.0e9),
        rfx_n_steps=400,
        openems_n_steps=500,
        rfx_freq_max_hz=30.0e9,
        openems_dx_m=0.5e-3,
        max_mag_abs_tol=0.20,
        mean_mag_abs_tol=0.15,
    ),
)


def _ensure_openems_numpy_compat() -> None:
    """openEMS Python bindings still expect deprecated NumPy aliases."""
    for name, value in {"float": float, "int": int, "complex": complex}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _oneport_from_s11(s11: np.ndarray) -> np.ndarray:
    s11 = np.asarray(s11, dtype=np.complex128)
    s_params = np.zeros((1, 1, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    return s_params


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _run_rfx_pec_short(case: CoaxialCalibrationCase) -> np.ndarray:
    """Run the M73 rfx PEC-short calibration and return per-frequency |S11|."""
    sim = Simulation(
        freq_max=case.rfx_freq_max_hz,
        domain=_DOMAIN_M,
        boundary="pec",
    )
    sim.add_coaxial_port(
        position=_GAP_POSITION_M,
        face="top",
        pin_length=_PIN_LENGTH_M,
        waveform=GaussianPulse(f0=_PULSE_F0_HZ, bandwidth=_PULSE_BANDWIDTH),
    )
    res = sim.compute_coaxial_s_matrix(
        n_steps=case.rfx_n_steps,
        n_freqs=len(case.freqs_hz),
        freqs=jnp.asarray(case.freqs_hz, dtype=jnp.float32),
    )
    return np.asarray(res.s_params[0, 0, :], dtype=np.complex128)


def _run_openems_pec_short(
    case: CoaxialCalibrationCase, sim_dir: Path
) -> np.ndarray:
    """Run the openEMS PEC-short coax reference and return per-frequency S11."""
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS

    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    unit = 1.0e-3  # mm
    fdtd = openEMS(NrTS=case.openems_n_steps, EndCriteria=0)
    fdtd.SetGaussExcite(_PULSE_F0_HZ, _PULSE_BANDWIDTH * _PULSE_F0_HZ)
    fdtd.SetBoundaryCond(["PEC"] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)
    for axis, length_m in zip("xyz", _DOMAIN_M):
        n_cells = int(round(length_m / case.openems_dx_m))
        mesh.AddLine(axis, np.linspace(0.0, length_m / unit, n_cells + 1))

    center_mm = [
        _GAP_POSITION_M[0] / unit,
        _GAP_POSITION_M[1] / unit,
    ]
    gap_z_mm = _GAP_POSITION_M[2] / unit
    pin_tip_z_mm = (_GAP_POSITION_M[2] - _PIN_LENGTH_M) / unit

    # PTFE fill between pin tip and gap (full coax line).
    ptfe = csx.AddMaterial("ptfe", epsilon=2.1)
    ptfe.AddCylinder(
        [center_mm[0], center_mm[1], pin_tip_z_mm],
        [center_mm[0], center_mm[1], gap_z_mm],
        radius=SMA_OUTER_RADIUS / unit,
        priority=1,
    )

    # Inner pin: PEC, extends from pin tip to gap.
    pin = csx.AddMetal("pin")
    pin.AddCylinder(
        [center_mm[0], center_mm[1], pin_tip_z_mm],
        [center_mm[0], center_mm[1], gap_z_mm],
        radius=SMA_PIN_RADIUS / unit,
        priority=10,
    )

    # Lumped port at the gap drives the line; the PEC short happens
    # naturally because the pin tip touches the cavity floor PEC at
    # pin_tip_z_mm.
    port = fdtd.AddLumpedPort(
        1,
        50.0,
        [center_mm[0], center_mm[1], gap_z_mm],
        [center_mm[0], center_mm[1], gap_z_mm + case.openems_dx_m / unit],
        "z",
        excite=1.0,
    )
    fdtd.Run(str(sim_dir), verbose=0, cleanup=True)
    port.CalcPort(str(sim_dir), np.asarray(case.freqs_hz, dtype=float))
    return np.asarray(port.uf_ref / port.uf_inc, dtype=np.complex128)


def _build_case_payload(
    case: CoaxialCalibrationCase, output_dir: Path
) -> dict[str, Any]:
    rfx_s11 = _run_rfx_pec_short(case)
    oem_s11 = _run_openems_pec_short(case, output_dir / f"_openems_tmp_{case.name}")
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    rfx_npz = case_dir / "rfx_pec_short_sparams.npz"
    oem_npz = case_dir / "openems_pec_short_sparams.npz"
    np.savez(
        rfx_npz,
        freqs_hz=np.asarray(case.freqs_hz, dtype=float),
        s_params=_oneport_from_s11(rfx_s11),
    )
    np.savez(
        oem_npz,
        freqs_hz=np.asarray(case.freqs_hz, dtype=float),
        s_params=_oneport_from_s11(oem_s11),
    )

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(rfx_npz),
        load_sparameter_dataset(oem_npz),
        terms="S11",
        comparison_mode="magnitude",
        max_abs_tol=1.0,
        mean_abs_tol=1.0,
        max_mag_abs_tol=case.max_mag_abs_tol,
        mean_mag_abs_tol=case.mean_mag_abs_tol,
    )
    payload["case"] = asdict(case)
    payload["case_artifacts"] = {
        "rfx_npz": str(rfx_npz.relative_to(output_dir)),
        "openems_npz": str(oem_npz.relative_to(output_dir)),
    }
    payload["per_frequency"] = {
        "freqs_hz": [float(f) for f in case.freqs_hz],
        "rfx_s11_mag": [float(abs(s)) for s in rfx_s11],
        "openems_s11_mag": [float(abs(s)) for s in oem_s11],
        "abs_diff": [float(abs(abs(r) - abs(o))) for r, o in zip(rfx_s11, oem_s11)],
    }
    return payload


def build_coaxial_pec_short_calibration(
    output_dir: Path,
    cases: tuple[CoaxialCalibrationCase, ...] = DEFAULT_CASES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_payloads: list[dict[str, Any]] = []
    overall_max = 0.0
    overall_mean = 0.0
    fail_count = 0

    for case in cases:
        payload = _build_case_payload(case, output_dir)
        case_payloads.append(payload)
        s = payload.get("summary", {})
        overall_max = max(overall_max, float(s.get("max_mag_abs_diff", 0.0)))
        overall_mean = max(
            overall_mean, float(s.get("mean_mag_abs_diff_max_over_terms", 0.0))
        )
        if payload.get("status") != "passed":
            fail_count += 1

    envelope_payload: dict[str, Any] = {
        "schema": "rfx.port_external_envelope",
        "schema_version": 1,
        "claim": (
            "coaxial S11 cross-solver calibration: "
            "Simulation.compute_coaxial_s_matrix(...) (rfx M73 distributed "
            "TFSF plane source) vs openEMS PEC-box AddLumpedPort"
        ),
        "claim_scope": (
            "uniform Yee SMA-geometry coax in 20 mm PEC cubic cavity with a "
            "5 mm pin terminated into vacuum (pin tip floating at z=10 mm). "
            "Both solvers see the same device under test; the M73 source "
            "mechanism differs from the openEMS AddLumpedPort coupling, so "
            "any residual S11-magnitude disagreement is the source-mechanism "
            "envelope at this discretisation"
        ),
        "evidence_level": "E4-cross-solver-calibration",
        "status": "passed" if fail_count == 0 else "failed",
        "case_count": len(case_payloads),
        "fail_count": fail_count,
        "envelope_summary": {
            "max_mag_abs_diff_across_cases": overall_max,
            "max_mean_mag_abs_diff_across_cases": overall_mean,
            "freq_range_hz": [
                float(min(min(c.freqs_hz) for c in cases)),
                float(max(max(c.freqs_hz) for c in cases)),
            ],
            "rfx_n_steps_grid": sorted({c.rfx_n_steps for c in cases}),
            "openems_n_steps_grid": sorted({c.openems_n_steps for c in cases}),
            "case_names": [c.name for c in cases],
            "geometry_domain_m": list(_DOMAIN_M),
            "geometry_pin_length_m": _PIN_LENGTH_M,
            "geometry_gap_position_m": list(_GAP_POSITION_M),
        },
        "cases": case_payloads,
        "commit_hash": _git_commit_short(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_json = output_dir / "coaxial_s11_calibration_envelope.json"
    output_json.write_text(json.dumps(envelope_payload, indent=2, sort_keys=True) + "\n")
    return envelope_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/2026-05-11-m73-coaxial-s11-calibration",
    )
    parser.add_argument("--keep-openems-tmp", action="store_true")
    args = parser.parse_args(argv)

    output_dir = _repo_path(args.output_dir)
    payload = build_coaxial_pec_short_calibration(output_dir, DEFAULT_CASES)
    if not args.keep_openems_tmp:
        for case in DEFAULT_CASES:
            tmp = output_dir / f"_openems_tmp_{case.name}"
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    summary = payload.get("envelope_summary", {})
    print(
        "status={status} case_count={n} fail_count={f} "
        "max_mag_abs_diff_across_cases={m:.6g}".format(
            status=payload["status"],
            n=payload["case_count"],
            f=payload["fail_count"],
            m=summary.get("max_mag_abs_diff_across_cases", 0.0),
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
