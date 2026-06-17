#!/usr/bin/env python3
"""Compare current rfx MSL thru-line S-parameters to a stored openEMS reference.

This is an M3 external-reference smoke check, not a claims-bearing MSL-port
promotion.  It uses a stored openEMS artifact generated outside rfx and reruns
rfx on the same core microstrip dimensions, then compares S11/S21 magnitudes in
the 3.0--4.5 GHz quasi-TEM gate window.

Validation-honesty note: on a MATCHED thru-line |S11| is intrinsically small
(reference mean ~0.12), comparable to the comparison tolerance (0.15), so the
S11 channel cannot discriminate rfx physics from a constant (a degenerate
``S11==0`` would "pass" it).  The S11 channel is therefore reported as
INFORMATIONAL (``s11_gate_discriminating=False``) and does NOT gate; the status
then reads ``passed_transmission_only_s11_nondiscriminating`` so a green result
is never mis-read as an S11 validation.  Transmission (|S21|~1) is the real
discriminator here.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = "docs/research_notes/msl_thru_openems_80um.json"


def _complex_array(values: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(real, imag) for real, imag in values], dtype=complex)


def load_openems_reference(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"meta", "freqs_ghz", "s11", "s21"}
    missing = sorted(required - set(data))
    if missing:
        raise SystemExit(f"reference {path} missing keys: {missing}")
    return data


def run_rfx_msl_thru(
    meta: dict[str, Any],
    *,
    num_periods: float,
    raw_3probe_dump_path: Path | None = None,
) -> dict[str, Any]:
    eps_r = float(meta["eps_r"])
    h_sub = float(meta["h_sub_mm"]) * 1e-3
    w_trace = float(meta["w_trace_mm"]) * 1e-3
    l_line = float(meta["l_line_mm"]) * 1e-3
    port_margin = float(meta["port_margin_mm"]) * 1e-3
    lx = float(meta["lx_mm"]) * 1e-3
    ly = float(meta["ly_mm"]) * 1e-3
    lz = float(meta["lz_mm"]) * 1e-3
    dx = float(meta["dx_um"]) * 1e-6
    f_stop = float(meta["f_stop_ghz"]) * 1e9
    n_freqs = int(meta["n_freqs"])

    sim = Simulation(
        freq_max=f_stop,
        domain=(lx, ly, lz),
        dx=dx,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=eps_r)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, h_sub)), material="ro4350b")

    y_centre = ly / 2.0
    trace_y_lo = y_centre - w_trace / 2.0
    trace_y_hi = y_centre + w_trace / 2.0
    sim.add(
        Box((0.0, trace_y_lo, h_sub), (lx, trace_y_hi, h_sub + dx)),
        material="pec",
    )

    sim.add_msl_port(
        position=(port_margin, y_centre, 0.0),
        width=w_trace,
        height=h_sub,
        direction="+x",
        impedance=50.0,
    )
    sim.add_msl_port(
        position=(port_margin + l_line, y_centre, 0.0),
        width=w_trace,
        height=h_sub,
        direction="-x",
        impedance=50.0,
    )

    started = time.monotonic()
    result = sim.compute_msl_s_matrix(
        n_freqs=n_freqs,
        num_periods=num_periods,
        raw_3probe_dump_path=(
            None if raw_3probe_dump_path is None else str(raw_3probe_dump_path)
        ),
    )
    duration_s = time.monotonic() - started
    return {
        "duration_s": duration_s,
        "freqs_hz": np.asarray(result.freqs, dtype=float),
        "s11": np.asarray(result.S[0, 0, :], dtype=complex),
        "s21": np.asarray(result.S[1, 0, :], dtype=complex),
        "z0": np.asarray(result.Z0[0, :], dtype=complex),
    }


def is_pass_status(status: str) -> bool:
    """True for any PASS status, for exit-code purposes.

    Includes ``passed_transmission_only_s11_nondiscriminating`` — a GREEN result on
    a matched line where the |S11| channel is informational (too small to
    discriminate). That is a pass and must NOT turn the M3 VESSL lane red; the
    exit code keys off this helper, not an exact ``== "passed"`` match.
    """
    return status.startswith("passed")


def compare_magnitudes(
    freqs_hz: np.ndarray,
    rfx_s11: np.ndarray,
    rfx_s21: np.ndarray,
    ref_freqs_hz: np.ndarray,
    ref_s11: np.ndarray,
    ref_s21: np.ndarray,
    *,
    f_lo_hz: float,
    f_hi_hz: float,
) -> dict[str, Any]:
    mask = (ref_freqs_hz >= f_lo_hz) & (ref_freqs_hz <= f_hi_hz)
    if not np.any(mask):
        raise SystemExit("reference has no points in comparison window")

    ref_f = ref_freqs_hz[mask]
    ref_s11_mag = np.abs(ref_s11[mask])
    ref_s21_mag = np.abs(ref_s21[mask])
    rfx_s11_mag = np.interp(ref_f, freqs_hz, np.abs(rfx_s11))
    rfx_s21_mag = np.interp(ref_f, freqs_hz, np.abs(rfx_s21))

    s11_mean_abs_diff = float(np.mean(np.abs(rfx_s11_mag - ref_s11_mag)))
    s21_mean_abs_diff = float(np.mean(np.abs(rfx_s21_mag - ref_s21_mag)))
    s11_mean_rfx = float(np.mean(rfx_s11_mag))
    s11_mean_ref = float(np.mean(ref_s11_mag))
    s21_mean_rfx = float(np.mean(rfx_s21_mag))
    s21_mean_ref = float(np.mean(ref_s21_mag))
    mag_tol = 0.15
    # Validation-honesty guard: on a MATCHED thru-line |S11| is intrinsically tiny
    # (here ref mean ~0.12), so an absolute |S11| tolerance of 0.15 is LARGER than the
    # signal it is supposed to check — a degenerate rfx output (S11==0) would pass it,
    # i.e. the gate cannot distinguish rfx physics from a constant. The S11 channel is a
    # real discriminator ONLY when the reference |S11| comfortably exceeds the tolerance.
    # When it does not, we mark S11 INFORMATIONAL and let the gate rest on transmission,
    # so a green status is never mis-read as "rfx S11 validated".
    s11_gate_discriminating = s11_mean_ref >= 2.0 * mag_tol
    pass_s11 = (s11_mean_abs_diff <= mag_tol) if s11_gate_discriminating else None
    pass_s21 = s21_mean_abs_diff <= mag_tol
    pass_transmission = 0.85 <= s21_mean_rfx <= 1.10 and 0.85 <= s21_mean_ref <= 1.10

    if s11_gate_discriminating:
        status = "passed" if (pass_s11 and pass_s21 and pass_transmission) else "failed"
    else:
        status = (
            "passed_transmission_only_s11_nondiscriminating"
            if (pass_s21 and pass_transmission)
            else "failed"
        )

    return {
        "frequency_window_hz": [float(f_lo_hz), float(f_hi_hz)],
        "n_points": int(np.sum(mask)),
        "s11_mean_abs_diff": s11_mean_abs_diff,
        "s21_mean_abs_diff": s21_mean_abs_diff,
        "s11_mean_rfx": s11_mean_rfx,
        "s11_mean_openems": s11_mean_ref,
        "s21_mean_rfx": s21_mean_rfx,
        "s21_mean_openems": s21_mean_ref,
        "pass_s11_mean_abs_diff_le_0p15": pass_s11,
        "pass_s21_mean_abs_diff_le_0p15": pass_s21,
        "pass_transmission_magnitude_window": pass_transmission,
        "s11_gate_discriminating": bool(s11_gate_discriminating),
        "s11_discrimination_note": (
            "S11 channel discriminating (ref |S11| mean "
            f"{s11_mean_ref:.3f} >= 2x tol {2.0 * mag_tol:.2f})"
            if s11_gate_discriminating
            else (
                f"S11 channel NON-discriminating: ref |S11| mean {s11_mean_ref:.3f} < 2x "
                f"tol {2.0 * mag_tol:.2f}; a degenerate rfx S11 would pass it, so S11 is "
                "INFORMATIONAL only and does NOT gate. This is a transmission smoke "
                "check, not an S11 validation."
            )
        ),
        "status": status,
    }


def _to_pairs(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--output-json",
        default=".omx/physics-gate/m3-msl-openems-reference/msl_openems_comparison.json",
    )
    parser.add_argument("--num-periods", type=float, default=12.0)
    parser.add_argument("--f-lo-ghz", type=float, default=3.0)
    parser.add_argument("--f-hi-ghz", type=float, default=4.5)
    parser.add_argument(
        "--raw-3probe-dump",
        default=None,
        help="Optional .npz path for raw MSL 3-probe V/I phasors.",
    )
    args = parser.parse_args(argv)

    ref_path = Path(args.reference)
    if not ref_path.is_absolute():
        ref_path = REPO_ROOT / ref_path
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reference = load_openems_reference(ref_path)
    ref_meta = reference["meta"]
    ref_freqs_hz = np.asarray(reference["freqs_ghz"], dtype=float) * 1e9
    ref_s11 = _complex_array(reference["s11"])
    ref_s21 = _complex_array(reference["s21"])

    raw_dump_path = None
    if args.raw_3probe_dump:
        raw_dump_path = Path(args.raw_3probe_dump)
        if not raw_dump_path.is_absolute():
            raw_dump_path = REPO_ROOT / raw_dump_path
    rfx = run_rfx_msl_thru(
        ref_meta,
        num_periods=args.num_periods,
        raw_3probe_dump_path=raw_dump_path,
    )
    metrics = compare_magnitudes(
        rfx["freqs_hz"],
        rfx["s11"],
        rfx["s21"],
        ref_freqs_hz,
        ref_s11,
        ref_s21,
        f_lo_hz=args.f_lo_ghz * 1e9,
        f_hi_hz=args.f_hi_ghz * 1e9,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": metrics["status"],
        "evidence_level": "E4",
        "claim": "MSL thru-line magnitude cross-check against stored openEMS reference",
        "claim_scope": (
            "narrow magnitude-only smoke check over 3.0--4.5 GHz; not an E5 "
            "claims-bearing MSL envelope"
        ),
        "reference_artifact": str(ref_path.relative_to(REPO_ROOT)),
        "reference_solver": ref_meta.get("solver", "openEMS"),
        "reference_meta": ref_meta,
        "rfx_duration_s": rfx["duration_s"],
        "rfx_raw_3probe_dump": (
            None
            if raw_dump_path is None
            else str(raw_dump_path.relative_to(REPO_ROOT))
        ),
        "metrics": metrics,
        "rfx_freqs_hz": [float(x) for x in rfx["freqs_hz"]],
        "rfx_s11": _to_pairs(rfx["s11"]),
        "rfx_s21": _to_pairs(rfx["s21"]),
        "rfx_z0": _to_pairs(rfx["z0"]),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "metrics": metrics}, indent=2))
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0 if is_pass_status(payload["status"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
