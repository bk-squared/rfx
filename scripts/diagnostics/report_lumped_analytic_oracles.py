#!/usr/bin/env python3
"""Report closed-form lumped-port S11 oracle checks.

This diagnostic validates the lumped-port V/I-to-S11 extractor against
analytic load impedances: matched, short, open, resistive mismatch, capacitor,
inductor, and series/parallel RLC.  It is an E2 extractor-oracle artifact, not
a broad FDTD calibrated-port E5 proof by itself.

WHICH CONVENTION IT REPORTS. It reads `extract_lumped_s11`, the PASSIVE
port-branch algebra, on V/I phasors this script CONSTRUCTS from each load
impedance. That is the right reading here: the phasors are a consistent
(V across the load, I into it) pair by construction, and no port is being
driven. It is NOT the reading a driven port gets since 2026-09-21 — a driven
one-cell port uses `driven_port_reflection`, and applying the algebra below to
a driven port's V/I gives the reciprocal of the physical reflection
(scripts/diagnostics/lumped_port_known_load_line.py). Nothing here solves a
field, so nothing here can catch that; the payload says so in
`extractor_convention` for a reader who finds this report on its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rfx.probes.probes import extract_lumped_s11


Z0_OHM = 50.0
FREQS_HZ = np.asarray([0.75e9, 1.0e9, 1.5e9, 2.0e9, 3.0e9, 4.0e9], dtype=np.float64)


def _gamma_from_load(z_load: np.ndarray, *, z0: float = Z0_OHM) -> np.ndarray:
    """Closed-form reflection coefficient for load impedance ``z_load``."""

    with np.errstate(divide="ignore", invalid="ignore"):
        return (z_load - z0) / (z_load + z0)


def _phasors_for_load(z_load: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FDTD-sign V/I phasors whose input impedance is ``z_load``.

    The lumped extractor convention is ``Z_in = -V/I``.  For finite loads we
    choose ``I=1`` and ``V=-Z``.  For ideal open circuits we choose ``I=0`` and
    a nonzero voltage, producing Γ=+1 without relying on an infinite number.
    """

    z_load = np.asarray(z_load, dtype=np.complex128)
    currents = np.ones_like(z_load)
    voltages = -z_load
    open_mask = np.isinf(np.real(z_load)) | np.isinf(np.imag(z_load))
    if np.any(open_mask):
        currents = np.where(open_mask, 0.0 + 0.0j, currents)
        voltages = np.where(open_mask, -1.0 + 0.0j, voltages)
    return voltages, currents


def _case_impedances(freqs: np.ndarray) -> dict[str, np.ndarray]:
    w = 2.0 * np.pi * freqs
    r = 25.0
    c = 1.0e-12
    l_h = 10.0e-9
    r_series = 10.0
    c_series = 1.0e-12
    l_series = 10.0e-9
    r_parallel = 200.0
    c_parallel = 1.0e-12
    l_parallel = 10.0e-9

    z_c = 1.0 / (1j * w * c)
    z_l = 1j * w * l_h
    z_series = r_series + 1j * w * l_series + 1.0 / (1j * w * c_series)
    y_parallel = (1.0 / r_parallel) + (1.0 / (1j * w * l_parallel)) + (1j * w * c_parallel)

    return {
        "matched_50ohm": np.full_like(freqs, Z0_OHM, dtype=np.complex128),
        "short_0ohm": np.zeros_like(freqs, dtype=np.complex128),
        "open_infinite": np.full_like(freqs, np.inf + 0.0j, dtype=np.complex128),
        "resistor_25ohm": np.full_like(freqs, r, dtype=np.complex128),
        "capacitor_1pf": z_c,
        "inductor_10nh": z_l,
        "series_rlc_10ohm_10nh_1pf": z_series,
        "parallel_rlc_200ohm_10nh_1pf": 1.0 / y_parallel,
    }


def evaluate_lumped_analytic_oracles(
    *,
    freqs_hz: np.ndarray = FREQS_HZ,
    atol: float = 2.0e-7,
    rtol: float = 2.0e-6,
) -> dict:
    """Evaluate all analytic oracle cases and return a JSONable payload."""

    freqs = np.asarray(freqs_hz, dtype=np.float64)
    cases = []
    for name, z_load in _case_impedances(freqs).items():
        voltages, currents = _phasors_for_load(z_load)
        production = np.asarray(
            extract_lumped_s11(
                jnp.asarray(voltages, dtype=jnp.complex64),
                jnp.asarray(currents, dtype=jnp.complex64),
                z0=Z0_OHM,
            ),
            dtype=np.complex128,
        )
        if name == "open_infinite":
            expected = np.ones_like(production)
        else:
            expected = _gamma_from_load(z_load)
        diff = np.abs(production - expected)
        allowed = atol + rtol * np.maximum(np.abs(production), np.abs(expected))
        case_status = "passed" if bool(np.all(diff <= allowed)) else "failed"
        cases.append(
            {
                "name": name,
                "status": case_status,
                "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
                "max_allowed": float(np.max(allowed)) if allowed.size else float(atol),
                "expected_gamma_first": {
                    "re": float(np.real(expected[0])),
                    "im": float(np.imag(expected[0])),
                },
                "production_gamma_first": {
                    "re": float(np.real(production[0])),
                    "im": float(np.imag(production[0])),
                },
            }
        )

    return {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "claim_scope": "lumped-port V/I-to-S11 analytic extractor oracle, not broad FDTD E5",
        "extractor_convention": (
            "extract_lumped_s11, the passive port-branch algebra, on synthetic "
            "V/I phasors constructed from each load impedance. A DRIVEN "
            "one-cell port uses driven_port_reflection instead; this report "
            "solves no field and does not exercise that path."
        ),
        "z0_ohm": Z0_OHM,
        "freqs_hz": freqs.tolist(),
        "atol": float(atol),
        "rtol": float(rtol),
        "cases": cases,
        "max_abs_diff": max((case["max_abs_diff"] for case in cases), default=0.0),
        "max_allowed": max((case["max_allowed"] for case in cases), default=float(atol)),
    }


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Lumped-port analytic oracle report",
        "",
        f"- status: `{payload['status']}`",
        f"- claim_scope: {payload['claim_scope']}",
        f"- z0_ohm: `{payload['z0_ohm']}`",
        f"- frequency_count: `{len(payload['freqs_hz'])}`",
        f"- max_abs_diff: `{payload['max_abs_diff']:.6g}`",
        f"- max_allowed: `{payload['max_allowed']:.6g}`",
        "",
        "| Case | Status | Max abs diff | Max allowed | Γ_expected first | Γ_production first |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        exp = case["expected_gamma_first"]
        prod = case["production_gamma_first"]
        lines.append(
            "| "
            f"`{case['name']}` | `{case['status']}` | "
            f"`{case['max_abs_diff']:.6g}` | `{case['max_allowed']:.6g}` | "
            f"`{exp['re']:.6g}{exp['im']:+.6g}j` | "
            f"`{prod['re']:.6g}{prod['im']:+.6g}j` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2.0e-7)
    parser.add_argument("--rtol", type=float, default=2.0e-6)
    args = parser.parse_args(argv)

    payload = evaluate_lumped_analytic_oracles(atol=args.atol, rtol=args.rtol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "lumped_analytic_oracle_report.json"
    md_path = args.output_dir / "lumped_analytic_oracle_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(
        f"status={payload['status']} max_abs_diff={payload['max_abs_diff']:.6g} "
        f"max_allowed={payload['max_allowed']:.6g}"
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
