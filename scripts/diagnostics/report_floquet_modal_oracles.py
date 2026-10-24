#!/usr/bin/env python3
"""Report analytic/specular modal oracles for Floquet helper infrastructure.

This is an E2-modal diagnostic artifact for the internal Floquet/Bloch helper
math. It validates phase-vector consistency and the current specular TE
forward/backward decomposition using synthetic DFT accumulators. It does **not**
promote `add_floquet_port(...)` to a high-level Simulation S-parameter API and
is not an FDTD, RCWA, or external full-wave cross-validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0
from rfx.floquet import (
    compute_floquet_s_params,
    extract_floquet_modes,
    floquet_wave_vector,
    init_floquet_dft,
)
from rfx.measurement.floquet import (
    floquet_phase_shift,
)
from rfx.grid import C0


FREQS_HZ = np.asarray([5.0e9, 10.0e9, 15.0e9], dtype=np.float64)
ANGLES_DEG = (0.0, 15.0, 30.0, 45.0, 60.0)
DEFAULT_LX_M = 0.015
DEFAULT_LY_M = 0.012
DEFAULT_GAMMA = 0.3 + 0.2j
DEFAULT_TAU = math.sqrt(1.0 - abs(DEFAULT_GAMMA) ** 2) + 0.0j


def _complex_dict(value: complex) -> dict[str, float]:
    return {"re": float(value.real), "im": float(value.imag)}


def _make_specular_te_accumulator(
    *,
    n_freqs: int,
    plane_shape: tuple[int, int],
    forward_amplitude: complex,
    backward_amplitude: complex,
    theta_deg: float,
):
    eta0 = float(np.sqrt(MU_0 / EPS_0))
    cos_theta = max(math.cos(math.radians(theta_deg)), 1e-10)
    eta_te = eta0 / cos_theta
    e_tangential = forward_amplitude + backward_amplitude
    h_tangential = (forward_amplitude - backward_amplitude) / eta_te

    acc = init_floquet_dft(n_freqs, plane_shape)
    e_plane = jnp.full((n_freqs,) + plane_shape, e_tangential, dtype=jnp.complex64)
    h_plane = jnp.full((n_freqs,) + plane_shape, h_tangential, dtype=jnp.complex64)
    return acc._replace(e_tang1_dft=e_plane, h_tang2_dft=h_plane)


def evaluate_floquet_modal_oracles(
    *,
    freqs_hz: np.ndarray = FREQS_HZ,
    angles_deg: tuple[float, ...] = ANGLES_DEG,
    lx_m: float = DEFAULT_LX_M,
    ly_m: float = DEFAULT_LY_M,
    gamma: complex = DEFAULT_GAMMA,
    tau: complex = DEFAULT_TAU,
    amplitude: complex = 1.0 + 0.25j,
    plane_shape: tuple[int, int] = (8, 6),
) -> dict:
    # The current Floquet helper stack runs through JAX's default float32 mode.
    freqs = jnp.asarray(freqs_hz, dtype=jnp.float32)
    n_freqs = len(freqs_hz)

    phase_rows = []
    max_phase_abs_diff = 0.0
    max_kmag_abs_diff = 0.0
    max_ky_abs_diff = 0.0
    for theta_deg in angles_deg:
        for freq in freqs_hz:
            phase_x, phase_y = floquet_phase_shift(lx_m, ly_m, float(freq), theta_deg, 0.0)
            kx, ky, kz = floquet_wave_vector(float(freq), theta_deg, 0.0)
            expected_phase_x = complex(np.exp(1j * kx * lx_m))
            expected_phase_y = complex(np.exp(1j * ky * ly_m))
            k0 = 2.0 * math.pi * float(freq) / C0
            k_mag = math.sqrt(kx * kx + ky * ky + kz * kz)
            phase_abs_diff = max(abs(phase_x - expected_phase_x), abs(phase_y - expected_phase_y))
            kmag_abs_diff = abs(k_mag - k0)
            ky_abs_diff = abs(ky)
            max_phase_abs_diff = max(max_phase_abs_diff, phase_abs_diff)
            max_kmag_abs_diff = max(max_kmag_abs_diff, kmag_abs_diff)
            max_ky_abs_diff = max(max_ky_abs_diff, ky_abs_diff)
            phase_rows.append(
                {
                    "theta_deg": float(theta_deg),
                    "freq_hz": float(freq),
                    "phase_abs_diff": float(phase_abs_diff),
                    "kmag_abs_diff": float(kmag_abs_diff),
                    "ky_abs_diff_phi0": float(ky_abs_diff),
                }
            )

    modal_rows = []
    max_forward_rel_error = 0.0
    max_backward_rel_error = 0.0
    max_s11_abs_diff = 0.0
    max_s21_abs_diff = 0.0
    max_power_error = 0.0
    for theta_deg in angles_deg:
        acc_inc = _make_specular_te_accumulator(
            n_freqs=n_freqs,
            plane_shape=plane_shape,
            forward_amplitude=amplitude,
            backward_amplitude=0.0 + 0.0j,
            theta_deg=theta_deg,
        )
        acc_ref = _make_specular_te_accumulator(
            n_freqs=n_freqs,
            plane_shape=plane_shape,
            forward_amplitude=amplitude,
            backward_amplitude=gamma * amplitude,
            theta_deg=theta_deg,
        )
        acc_trans = _make_specular_te_accumulator(
            n_freqs=n_freqs,
            plane_shape=plane_shape,
            forward_amplitude=tau * amplitude,
            backward_amplitude=0.0 + 0.0j,
            theta_deg=theta_deg,
        )

        extracted = extract_floquet_modes(
            acc_ref,
            dx=0.001,
            Lx=lx_m,
            Ly=ly_m,
            freqs=freqs,
            theta_deg=theta_deg,
            phi_deg=0.0,
            n_modes=1,
        )
        sparams = compute_floquet_s_params(
            acc_inc,
            acc_ref,
            acc_trans,
            dx=0.001,
            Lx=lx_m,
            Ly=ly_m,
            freqs=freqs,
            theta_deg=theta_deg,
            phi_deg=0.0,
        )

        for idx, freq in enumerate(freqs_hz):
            forward = complex(extracted["forward_amplitude"][idx])
            backward = complex(extracted["backward_amplitude"][idx])
            s11 = complex(sparams["S11"][idx])
            s21 = complex(sparams["S21"][idx])
            forward_rel_error = abs(forward - amplitude) / max(abs(amplitude), 1e-30)
            expected_backward = gamma * amplitude
            backward_rel_error = abs(backward - expected_backward) / max(abs(expected_backward), 1e-30)
            s11_abs_diff = abs(s11 - gamma)
            s21_abs_diff = abs(s21 - tau)
            power_error = abs(abs(s11) ** 2 + abs(s21) ** 2 - 1.0)
            max_forward_rel_error = max(max_forward_rel_error, forward_rel_error)
            max_backward_rel_error = max(max_backward_rel_error, backward_rel_error)
            max_s11_abs_diff = max(max_s11_abs_diff, s11_abs_diff)
            max_s21_abs_diff = max(max_s21_abs_diff, s21_abs_diff)
            max_power_error = max(max_power_error, power_error)
            modal_rows.append(
                {
                    "theta_deg": float(theta_deg),
                    "freq_hz": float(freq),
                    "forward": _complex_dict(forward),
                    "expected_forward": _complex_dict(amplitude),
                    "forward_rel_error": float(forward_rel_error),
                    "backward": _complex_dict(backward),
                    "expected_backward": _complex_dict(expected_backward),
                    "backward_rel_error": float(backward_rel_error),
                    "s11": _complex_dict(s11),
                    "expected_s11": _complex_dict(gamma),
                    "s11_abs_diff": float(s11_abs_diff),
                    "s21": _complex_dict(s21),
                    "expected_s21": _complex_dict(tau),
                    "s21_abs_diff": float(s21_abs_diff),
                    "power_balance_abs_error": float(power_error),
                }
            )

    checks = [
        {
            "name": "phase_shift_matches_wave_vector",
            "value": float(max_phase_abs_diff),
            "limit": 1e-12,
            "status": "passed" if max_phase_abs_diff <= 1e-12 else "failed",
        },
        {
            "name": "wave_vector_magnitude_matches_k0",
            "value": float(max_kmag_abs_diff),
            "limit": 1e-5,
            "status": "passed" if max_kmag_abs_diff <= 1e-5 else "failed",
        },
        {
            "name": "phi0_has_zero_ky",
            "value": float(max_ky_abs_diff),
            "limit": 1e-12,
            "status": "passed" if max_ky_abs_diff <= 1e-12 else "failed",
        },
        {
            "name": "forward_amplitude_recovered",
            "value": float(max_forward_rel_error),
            "limit": 1e-5,
            "status": "passed" if max_forward_rel_error <= 1e-5 else "failed",
        },
        {
            "name": "backward_amplitude_recovered",
            "value": float(max_backward_rel_error),
            "limit": 1e-5,
            "status": "passed" if max_backward_rel_error <= 1e-5 else "failed",
        },
        {
            "name": "s11_matches_synthetic_gamma",
            "value": float(max_s11_abs_diff),
            "limit": 1e-5,
            "status": "passed" if max_s11_abs_diff <= 1e-5 else "failed",
        },
        {
            "name": "s21_matches_synthetic_tau",
            "value": float(max_s21_abs_diff),
            "limit": 1e-5,
            "status": "passed" if max_s21_abs_diff <= 1e-5 else "failed",
        },
        {
            "name": "lossless_synthetic_power_balance",
            "value": float(max_power_error),
            "limit": 1e-5,
            "status": "passed" if max_power_error <= 1e-5 else "failed",
        },
    ]
    status = "passed" if all(check["status"] == "passed" for check in checks) else "failed"
    return {
        "status": status,
        "claim_scope": (
            "Floquet/Bloch specular TE modal bookkeeping oracle only; no promoted "
            "Simulation Floquet S-parameter API, no FDTD validation, no external RCWA/full-wave reference"
        ),
        "freqs_hz": freqs_hz.tolist(),
        "angles_deg": [float(angle) for angle in angles_deg],
        "periods_m": {"Lx": float(lx_m), "Ly": float(ly_m)},
        "synthetic_coefficients": {
            "forward_amplitude": _complex_dict(amplitude),
            "gamma": _complex_dict(gamma),
            "tau": _complex_dict(tau),
        },
        "checks": checks,
        "phase_rows": phase_rows,
        "modal_rows": modal_rows,
    }


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Floquet modal oracle report",
        "",
        f"- status: `{payload['status']}`",
        f"- claim_scope: {payload['claim_scope']}",
        f"- angles_deg: `{payload['angles_deg']}`",
        f"- freqs_hz: `{payload['freqs_hz']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Value | Limit |",
        "|---|---:|---:|---:|",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| `{check['name']}` | `{check['status']}` | "
            f"`{check['value']:.6g}` | `{check['limit']:.6g}` |"
        )
    lines.extend([
        "",
        "## Scope note",
        "",
        "This report uses synthetic specular TE DFT accumulators to check the modal",
        "bookkeeping formulas. It does not validate `add_floquet_port(...)` FDTD",
        "fields or promote a public Floquet S-parameter calculator.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = evaluate_floquet_modal_oracles()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "floquet_modal_oracle_report.json"
    md_path = args.output_dir / "floquet_modal_oracle_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"status={payload['status']}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
