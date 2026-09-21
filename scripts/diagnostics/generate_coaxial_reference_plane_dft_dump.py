#!/usr/bin/env python3
"""Generate a real-FDTD coaxial reference-plane DFT diagnostic.

This is an infrastructure diagnostic for the broad-E5 coaxial-port blocker.
It wires the current low-level ``add_coaxial_port`` material/source helper to
frequency-domain Cartesian field-plane probes, then routes those DFT fields
through the M61/M62 TEM reference-plane V/I extractor.

The diagnostic is intentionally not a promotion gate.  A finite run here only
proves that a real-FDTD DFT plane can be captured and replayed by an
independent post-processor.  Broad coaxial E5 still requires a calibrated TEM
reference-plane signal, matched/open/short/load reference fixtures, and an
external full-wave envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, init_materials
from rfx.grid import Grid
from rfx.probes.probes import driven_port_reflection, init_dft_plane_probe
from rfx.simulation import LumpedPortSParamSpec, SourceSpec, run
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
from rfx.sources.sources import GaussianPulse, port_d_parallel


warnings.filterwarnings(
    "ignore",
    message="Explicitly requested dtype .*jax_enable_x64.*",
    category=UserWarning,
)


FREQS_HZ = np.asarray([3.0e9, 5.0e9, 7.0e9], dtype=np.float64)
SIGNAL_FLOOR = 1e-12
PLANE_GAP_S11_ABS_LIMIT = 0.5


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _make_coaxial_source_spec(
    grid: Grid,
    materials,
    port: CoaxialPort,
    *,
    component: str,
    gap_idx: tuple[int, int, int],
    n_steps: int,
) -> SourceSpec:
    i, j, k = gap_idx
    eps = materials.eps_r[i, j, k] * EPS_0
    sigma = materials.sigma[i, j, k]
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)
    d_par = port_d_parallel(grid, gap_idx, component)
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = (cb / d_par) * jax.vmap(port.excitation)(times)
    return SourceSpec(i=i, j=j, k=k, component=component, waveform=waveform)


def classify_coaxial_reference_plane_replay(
    *,
    plane_s11,
    gap_s11,
    max_abs_voltage: float,
    max_abs_current: float,
    signal_floor: float = SIGNAL_FLOOR,
    plane_gap_s11_abs_limit: float = PLANE_GAP_S11_ABS_LIMIT,
) -> dict:
    """Classify whether the real-FDTD plane replay is usable as a TEM signal."""

    plane = np.asarray(plane_s11, dtype=np.complex128)
    gap = np.asarray(gap_s11, dtype=np.complex128)
    if plane.shape != gap.shape:
        raise ValueError(f"plane_s11 and gap_s11 shapes differ: {plane.shape} vs {gap.shape}")
    finite = bool(np.all(np.isfinite(plane)) and np.all(np.isfinite(gap)))
    diff = np.abs(plane - gap)
    max_diff = float(np.max(diff)) if diff.size else float("nan")
    blockers: list[str] = []
    if not finite:
        blockers.append("non-finite S11 values in plane or gap replay")
    if max_abs_voltage <= signal_floor:
        blockers.append(
            f"TEM reference-plane voltage below signal floor: {max_abs_voltage:.3e} <= {signal_floor:.3e}"
        )
    if max_abs_current <= signal_floor:
        blockers.append(
            f"TEM reference-plane current below signal floor: {max_abs_current:.3e} <= {signal_floor:.3e}"
        )
    if not np.isfinite(max_diff) or max_diff > plane_gap_s11_abs_limit:
        blockers.append(
            "plane replay does not match the current gap diagnostic within the "
            f"internal-consistency smoke limit: {max_diff:.3e} > {plane_gap_s11_abs_limit:.3e}"
        )
    status = "passed" if not blockers else "blocked"
    return {
        "status": status,
        "evidence_level": (
            "E3-diagnostic-internal-consistency"
            if status == "passed"
            else "E3-diagnostic-blocked"
        ),
        "claim_scope": (
            "real-FDTD DFT-plane replay infrastructure only; internal gap "
            "comparison is not external full-wave evidence and not broad E5"
        ),
        "max_plane_gap_s11_abs_diff": max_diff,
        "plane_gap_s11_abs_limit": float(plane_gap_s11_abs_limit),
        "signal_floor": float(signal_floor),
        "max_abs_voltage": float(max_abs_voltage),
        "max_abs_current": float(max_abs_current),
        "blockers": blockers,
    }


def generate_coaxial_reference_plane_dft_dump(
    *,
    output_dir: Path,
    n_steps: int = 160,
) -> dict:
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
    axis, direction, component, pin_center, pin_tip, gap_idx = _coaxial_port_geometry(grid, port)
    if axis != "z":
        raise ValueError("this diagnostic currently expects the default z-axis coaxial port")

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
    plane_index = grid.position_to_index(pin_center)[2]
    dft_planes = [
        init_dft_plane_probe(
            axis=2,
            index=plane_index,
            component=field_component,
            freqs=freqs,
            grid_shape=grid.shape,
            dft_total_steps=n_steps,
        )
        for field_component in ("ex", "ey", "hx", "hy")
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
        raise RuntimeError("coaxial reference-plane diagnostic produced no DFT planes")
    if not result.lumped_port_sparams:
        raise RuntimeError("coaxial reference-plane diagnostic produced no gap V/I accumulator")

    plane_by_component = {
        probe.component: np.asarray(probe.accumulator, dtype=np.complex128)
        for probe in result.dft_planes
    }
    u_coords = (np.arange(grid.nx, dtype=np.float64) - grid.pad_x_lo) * grid.dx
    v_coords = (np.arange(grid.ny, dtype=np.float64) - grid.pad_y_lo) * grid.dx
    z0_tem = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS,
        SMA_OUTER_RADIUS,
        PTFE_EPS_R,
    )
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
    _raw_spec, gap_accs = result.lumped_port_sparams[0]
    gap_v_dft, gap_i_dft = gap_accs[0], gap_accs[1]
    # A DRIVEN port's S11 is the terminal reflection.  This used to be
    # driven_port_reflection, the passive port-branch reading, which on a driven
    # port is the reciprocal of the physical reflection
    # (scripts/diagnostics/lumped_port_known_load_line.py).
    gap_s11 = np.asarray(
        driven_port_reflection(gap_v_dft, gap_i_dft, port.impedance),
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

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / "coaxial_reference_plane_dft_dump.npz"
    np.savez_compressed(
        dump_path,
        freqs_hz=np.asarray(FREQS_HZ, dtype=np.float64),
        u_coords_m=u_coords,
        v_coords_m=v_coords,
        ex_dft=plane_by_component["ex"],
        ey_dft=plane_by_component["ey"],
        hx_dft=plane_by_component["hx"],
        hy_dft=plane_by_component["hy"],
        plane_voltage=extracted.vi.voltage,
        plane_current=extracted.vi.current,
        plane_s11=plane_s11,
        gap_s11=gap_s11,
        metadata_json=json.dumps(
            {
                "commit_hash": _git_commit(),
                "grid_shape": list(grid.shape),
                "dx_m": float(grid.dx),
                "dt_s": float(grid.dt),
                "n_steps": int(n_steps),
                "boundary": "pec",
                "plane_axis": "z",
                "plane_index": int(plane_index),
                "plane_z_m": float(pin_center[2]),
                "gap_index": list(gap_idx),
                "pin_center_m": list(pin_center),
                "pin_tip_m": list(pin_tip),
            },
            sort_keys=True,
        ),
    )

    rows = []
    for idx, freq in enumerate(FREQS_HZ):
        rows.append(
            {
                "freq_hz": float(freq),
                "plane_s11": {
                    "re": float(plane_s11[idx].real),
                    "im": float(plane_s11[idx].imag),
                    "abs": float(abs(plane_s11[idx])),
                },
                "gap_s11": {
                    "re": float(gap_s11[idx].real),
                    "im": float(gap_s11[idx].imag),
                    "abs": float(abs(gap_s11[idx])),
                },
                "abs_diff": float(abs(plane_s11[idx] - gap_s11[idx])),
            }
        )
    payload = {
        **classification,
        "artifact": str(dump_path),
        "family": "coaxial_port",
        "diagnostic": "coaxial_reference_plane_dft_replay",
        "reference_impedance_ohm": float(port.impedance),
        "tem_z0_ohm": float(z0_tem),
        "n_steps": int(n_steps),
        "freqs_hz": FREQS_HZ.tolist(),
        "grid_shape": list(grid.shape),
        "dx_m": float(grid.dx),
        "dt_s": float(grid.dt),
        "plane_axis": "z",
        "plane_index": int(plane_index),
        "plane_z_m": float(pin_center[2]),
        "gap_index": list(gap_idx),
        "pin_center_m": list(pin_center),
        "pin_tip_m": list(pin_tip),
        "max_abs_plane_field_dft": {
            name: float(np.max(np.abs(values)))
            for name, values in plane_by_component.items()
        },
        "rows": rows,
    }
    return payload


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Coaxial real-FDTD reference-plane DFT diagnostic",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- claim_scope: {payload['claim_scope']}",
        f"- artifact: `{payload['artifact']}`",
        f"- max_abs_voltage: `{payload['max_abs_voltage']:.6g}`",
        f"- max_abs_current: `{payload['max_abs_current']:.6g}`",
        f"- max_plane_gap_s11_abs_diff: `{payload['max_plane_gap_s11_abs_diff']:.6g}`",
        "",
        "## Blockers",
        "",
    ]
    if payload["blockers"]:
        lines.extend(f"- {item}" for item in payload["blockers"])
    else:
        lines.append("- none for this internal diagnostic; still not E4/E5 evidence")
    lines.extend(
        [
            "",
            "## Frequency rows",
            "",
            "| f (Hz) | Plane |S11| | Gap |S11| | abs diff |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| `{row['freq_hz']:.6g}` | `{row['plane_s11']['abs']:.6g}` | "
            f"`{row['gap_s11']['abs']:.6g}` | `{row['abs_diff']:.6g}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=160)
    args = parser.parse_args(argv)

    payload = generate_coaxial_reference_plane_dft_dump(
        output_dir=args.output_dir,
        n_steps=args.n_steps,
    )
    json_path = args.output_dir / "coaxial_reference_plane_dft_replay.json"
    md_path = args.output_dir / "coaxial_reference_plane_dft_replay.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"status={payload['status']} evidence_level={payload['evidence_level']}")
    # Blocked is the expected current diagnostic outcome, not a script error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
