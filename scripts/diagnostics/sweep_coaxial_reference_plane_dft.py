#!/usr/bin/env python3
"""Sweep real-FDTD coaxial DFT reference planes along the coax axis.

M63 showed a blocked TEM replay at the pin-center reference plane.  This
diagnostic checks whether that was merely a bad plane choice by capturing the
same Cartesian DFT fields at every z-plane spanning the current coaxial helper
geometry from pin tip to gap cell, then replaying each plane through the
M61/M62 TEM V/I extractor.

This remains diagnostic infrastructure only.  It does not compare against an
external solver and must not be counted as E4/E5 evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import jax.numpy as jnp
import numpy as np

from generate_coaxial_reference_plane_dft_dump import (
    FREQS_HZ,
    classify_coaxial_reference_plane_replay,
    _make_coaxial_source_spec,
)
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.probes.probes import driven_port_reflection, init_dft_plane_probe
from rfx.simulation import LumpedPortSParamSpec, run
from rfx.sources.coaxial_port import (
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    CoaxialPort,
    _coaxial_port_geometry,
    coaxial_tem_characteristic_impedance,
    coaxial_tem_reference_plane_s11,
    coaxial_tem_reference_plane_vi_from_cartesian_plane,
    setup_coaxial_port,
)
from rfx.sources.sources import GaussianPulse


PLANE_COMPONENTS = ("ex", "ey", "hx", "hy")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def summarize_reference_plane_sweep(rows: list[dict]) -> dict:
    """Summarize per-plane replay rows without promoting diagnostic evidence."""

    passed = [row for row in rows if row["classification"]["status"] == "passed"]
    finite_rows = [row for row in rows if np.isfinite(row["signal_score"])]
    best_signal = max(finite_rows, key=lambda row: row["signal_score"]) if finite_rows else None
    best_s11 = min(
        finite_rows,
        key=lambda row: row["classification"]["max_plane_gap_s11_abs_diff"],
    ) if finite_rows else None
    blockers: list[str] = []
    if not passed:
        blockers.append("no swept real-FDTD reference plane produced usable TEM V/I")
    if best_signal is not None:
        blockers.append(
            "best signal-score plane still has "
            f"max_abs_voltage={best_signal['classification']['max_abs_voltage']:.3e}, "
            f"max_abs_current={best_signal['classification']['max_abs_current']:.3e}"
        )
    return {
        "status": "passed" if passed else "blocked",
        "evidence_level": (
            "E3-diagnostic-internal-consistency"
            if passed
            else "E3-diagnostic-blocked"
        ),
        "claim_scope": (
            "real-FDTD coaxial DFT-plane sweep replay only; no external full-wave "
            "comparison and not broad E5"
        ),
        "plane_count": len(rows),
        "passed_plane_count": len(passed),
        "best_signal_plane_index": (
            int(best_signal["plane_index"]) if best_signal is not None else None
        ),
        "best_signal_score": (
            float(best_signal["signal_score"]) if best_signal is not None else None
        ),
        "best_s11_plane_index": int(best_s11["plane_index"]) if best_s11 is not None else None,
        "best_s11_max_abs_diff": (
            float(best_s11["classification"]["max_plane_gap_s11_abs_diff"])
            if best_s11 is not None
            else None
        ),
        "blockers": blockers,
    }


def sweep_coaxial_reference_planes(
    *,
    output_dir: Path,
    n_steps: int = 160,
    plane_index_step: int = 1,
) -> dict:
    if plane_index_step < 1:
        raise ValueError(f"plane_index_step must be >= 1, got {plane_index_step}")

    domain = (0.020, 0.020, 0.020)
    dx = 0.5e-3
    grid = Grid(freq_max=10.0e9, domain=domain, dx=dx, cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=5.0e9, bandwidth=0.8, amplitude=1.0)
    port = CoaxialPort(
        position=(0.010, 0.010, 0.015),
        face="top",
        pin_length=5e-3,
        pin_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        impedance=50.0,
        excitation=pulse,
    )
    materials = setup_coaxial_port(grid, port, materials)
    axis, _direction, component, pin_center, pin_tip, gap_idx = _coaxial_port_geometry(
        grid,
        port,
    )
    if axis != "z":
        raise ValueError("this diagnostic currently expects the default z-axis coaxial port")
    pin_tip_idx = grid.position_to_index(pin_tip)[2]
    gap_axis_idx = gap_idx[2]
    lo = min(pin_tip_idx, gap_axis_idx)
    hi = max(pin_tip_idx, gap_axis_idx)
    plane_indices = list(range(lo, hi + 1, plane_index_step))
    if plane_indices[-1] != hi:
        plane_indices.append(hi)

    freqs = jnp.asarray(FREQS_HZ, dtype=jnp.float32)
    source = _make_coaxial_source_spec(
        grid,
        materials,
        port,
        component=component,
        gap_idx=gap_idx,
        n_steps=n_steps,
    )
    gap_spec = LumpedPortSParamSpec(
        i=gap_idx[0],
        j=gap_idx[1],
        k=gap_idx[2],
        component=component,
        freqs=freqs,
        impedance=port.impedance,
    )
    dft_planes = [
        init_dft_plane_probe(
            axis=2,
            index=plane_index,
            component=field_component,
            freqs=freqs,
            grid_shape=grid.shape,
            dft_total_steps=n_steps,
        )
        for plane_index in plane_indices
        for field_component in PLANE_COMPONENTS
    ]
    result = run(
        grid,
        materials,
        n_steps,
        boundary="pec",
        sources=[source],
        dft_planes=dft_planes,
        lumped_port_sparams=[gap_spec],
        return_state=False,
    )
    if result.dft_planes is None:
        raise RuntimeError("coaxial reference-plane sweep produced no DFT planes")
    if not result.lumped_port_sparams:
        raise RuntimeError("coaxial reference-plane sweep produced no gap V/I accumulator")

    _raw_spec, gap_accs = result.lumped_port_sparams[0]
    gap_v_dft, gap_i_dft = gap_accs[0], gap_accs[1]
    # A DRIVEN port's S11 is the terminal reflection.  This used to be
    # extract_lumped_s11, the passive port-branch reading, which on a driven
    # port is the reciprocal of the physical reflection
    # (scripts/diagnostics/lumped_port_known_load_line.py).
    gap_s11 = np.asarray(
        driven_port_reflection(gap_v_dft, gap_i_dft, port.impedance),
        dtype=np.complex128,
    )
    z0_tem = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS,
        SMA_OUTER_RADIUS,
        PTFE_EPS_R,
    )
    u_coords = (np.arange(grid.nx, dtype=np.float64) - grid.pad_x_lo) * grid.dx
    v_coords = (np.arange(grid.ny, dtype=np.float64) - grid.pad_y_lo) * grid.dx

    rows = []
    probes = list(result.dft_planes)
    for plane_offset, plane_index in enumerate(plane_indices):
        start = plane_offset * len(PLANE_COMPONENTS)
        plane_by_component = {
            probe.component: np.asarray(probe.accumulator, dtype=np.complex128)
            for probe in probes[start : start + len(PLANE_COMPONENTS)]
        }
        extracted = coaxial_tem_reference_plane_vi_from_cartesian_plane(
            u_coords,
            v_coords,
            plane_by_component["ex"],
            plane_by_component["ey"],
            plane_by_component["hx"],
            plane_by_component["hy"],
            center_u_m=port.position[0],
            center_v_m=port.position[1],
            inner_radius=SMA_PIN_RADIUS,
            outer_radius=SMA_OUTER_RADIUS,
            eps_r=PTFE_EPS_R,
        )
        plane_s11 = np.asarray(
            coaxial_tem_reference_plane_s11(
                extracted.vi.voltage,
                extracted.vi.current,
                z0_tem,
            ),
            dtype=np.complex128,
        )
        max_abs_voltage = float(np.max(np.abs(extracted.vi.voltage)))
        max_abs_current = float(np.max(np.abs(extracted.vi.current)))
        classification = classify_coaxial_reference_plane_replay(
            plane_s11=plane_s11,
            gap_s11=gap_s11,
            max_abs_voltage=max_abs_voltage,
            max_abs_current=max_abs_current,
        )
        rows.append(
            {
                "plane_index": int(plane_index),
                "plane_z_m": float((plane_index - grid.pad_z_lo) * grid.dx),
                "classification": classification,
                "signal_score": float(min(max_abs_voltage, max_abs_current)),
                "max_abs_plane_field_dft": {
                    name: float(np.max(np.abs(values)))
                    for name, values in plane_by_component.items()
                },
                "plane_s11_abs": [float(abs(value)) for value in plane_s11],
                "gap_s11_abs": [float(abs(value)) for value in gap_s11],
                "plane_gap_s11_abs_diff": [
                    float(abs(plane_s11[idx] - gap_s11[idx]))
                    for idx in range(len(gap_s11))
                ],
            }
        )

    summary = summarize_reference_plane_sweep(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **summary,
        "family": "coaxial_port",
        "diagnostic": "coaxial_reference_plane_dft_sweep",
        "commit_hash": _git_commit(),
        "n_steps": int(n_steps),
        "freqs_hz": FREQS_HZ.tolist(),
        "grid_shape": list(grid.shape),
        "dx_m": float(grid.dx),
        "dt_s": float(grid.dt),
        "plane_axis": "z",
        "plane_indices": plane_indices,
        "pin_tip_index": int(pin_tip_idx),
        "gap_index": list(gap_idx),
        "pin_center_m": list(pin_center),
        "pin_tip_m": list(pin_tip),
        "rows": rows,
    }
    return payload


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Coaxial real-FDTD reference-plane DFT sweep",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- claim_scope: {payload['claim_scope']}",
        f"- plane_count: `{payload['plane_count']}`",
        f"- passed_plane_count: `{payload['passed_plane_count']}`",
        f"- best_signal_plane_index: `{payload['best_signal_plane_index']}`",
        f"- best_signal_score: `{payload['best_signal_score']:.6g}`",
        f"- best_s11_plane_index: `{payload['best_s11_plane_index']}`",
        f"- best_s11_max_abs_diff: `{payload['best_s11_max_abs_diff']:.6g}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["blockers"])
    lines.extend(
        [
            "",
            "## Plane rows",
            "",
            "| plane index | z (m) | status | max |V| | max |I| | max plane/gap |ΔS11| |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        cls = row["classification"]
        lines.append(
            f"| `{row['plane_index']}` | `{row['plane_z_m']:.6g}` | "
            f"`{cls['status']}` | `{cls['max_abs_voltage']:.6g}` | "
            f"`{cls['max_abs_current']:.6g}` | "
            f"`{cls['max_plane_gap_s11_abs_diff']:.6g}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=160)
    parser.add_argument("--plane-index-step", type=int, default=1)
    args = parser.parse_args(argv)

    payload = sweep_coaxial_reference_planes(
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        plane_index_step=args.plane_index_step,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "coaxial_reference_plane_dft_sweep.json"
    md_path = args.output_dir / "coaxial_reference_plane_dft_sweep.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"status={payload['status']} passed_plane_count={payload['passed_plane_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
