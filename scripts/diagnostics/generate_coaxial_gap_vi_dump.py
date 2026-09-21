#!/usr/bin/env python3
"""Generate a diagnostic coaxial-port gap V/I dump and replay report.

This uses the low-level `rfx.sources.coaxial_port` material/source helpers plus
the compiled runner's single-cell V/I DFT accumulator at the coaxial gap cell.
It is E3 gap-replay infrastructure evidence only: it does not validate a TEM
coaxial reference plane, does not calibrate a coaxial line mode, and does not
promote `add_coaxial_port(...)` to a high-level S-parameter API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from rfx import (
    PortDumpMetadata,
    compare_replayed_smatrix,
    load_port_vi_dump_npz,
    replay_smatrix_from_port_vi_dump,
    save_port_vi_dump_npz,
)
from rfx.core.yee import EPS_0, init_materials
from rfx.grid import Grid
from rfx.probes.probes import driven_port_reflection
from rfx.simulation import LumpedPortSParamSpec, SourceSpec, run
from rfx.sources.coaxial_port import (
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    CoaxialPort,
    _coaxial_port_geometry,
    setup_coaxial_port,
)
from rfx.sources.sources import GaussianPulse, port_d_parallel


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


def generate_coaxial_gap_vi_dump(
    *,
    dump_path: Path,
    summary_json: Path | None = None,
    replay_json: Path | None = None,
    n_steps: int = 200,
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
    freqs = jnp.asarray([3.0e9, 5.0e9, 7.0e9], dtype=jnp.float32)
    source = _make_coaxial_source_spec(
        grid,
        materials,
        port,
        component=component,
        gap_idx=gap_idx,
        n_steps=n_steps,
    )
    spec = LumpedPortSParamSpec(
        i=gap_idx[0],
        j=gap_idx[1],
        k=gap_idx[2],
        component=component,
        freqs=freqs,
        impedance=port.impedance,
    )
    result = run(
        grid,
        materials,
        n_steps,
        boundary="pec",
        sources=[source],
        lumped_port_sparams=[spec],
        return_state=False,
    )
    if not result.lumped_port_sparams:
        raise RuntimeError("coaxial diagnostic produced no V/I DFT accumulators")
    raw_spec, accs = result.lumped_port_sparams[0]
    v_dft, i_dft, v_ref_dft = accs[0], accs[1], accs[2]
    # This gap is a DRIVEN port, so its S11 is the terminal reflection.  It
    # used to be extract_lumped_s11 — the passive port-branch reading, which
    # on a driven port is the reciprocal of the physical reflection
    # (scripts/diagnostics/lumped_port_known_load_line.py).  The dump's
    # diagonal_frame below declares this frame, and a writer may only declare
    # the frame its recorded production S actually used.
    diagnostic_s11 = np.asarray(
        driven_port_reflection(v_dft, i_dft, port.impedance),
        dtype=np.complex128)

    # Convert from the runner's FDTD sign convention to the public dump replay
    # convention used by rfx.validation: voltage/current positive into the DUT.
    voltages = -np.asarray(v_dft, dtype=np.complex128).reshape(1, 1, -1)
    currents = np.asarray(i_dft, dtype=np.complex128).reshape(1, 1, -1)
    drive_ref_voltages = -np.asarray(v_ref_dft, dtype=np.complex128).reshape(1, 1, -1)
    production_smatrix = diagnostic_s11.reshape(1, 1, -1)

    metadata = PortDumpMetadata(
        commit_hash=_git_commit(),
        # The coaxial gap probe is a lumped port, whose production
        # diagonal is the driven terminal reflection.
        diagonal_frame="driven_terminal",
        geometry={
            "kind": "single_coaxial_gap_diagnostic_pec_cavity",
            "domain_m": list(domain),
            "dx_m": dx,
            "coaxial_axis": axis,
            "coaxial_direction": float(direction),
            "pin_center_m": list(pin_center),
            "pin_tip_m": list(pin_tip),
            "gap_index": list(gap_idx),
        },
        materials={
            "background_eps_r": 1.0,
            "ptfe_eps_r": 2.1,
            "pec_sigma_s_per_m": 1.0e10,
        },
        grid={
            "shape": list(grid.shape),
            "dx_m": float(grid.dx),
            "dt_s": float(grid.dt),
            "boundary": "pec",
            "cpml_layers": 0,
        },
        boundaries={"x": "pec", "y": "pec", "z": "pec"},
        port_definitions=(
            {
                "name": "coax_gap",
                "kind": "coaxial_gap_single_cell_diagnostic",
                "position_m": list(port.position),
                "face": port.face,
                "component": component,
                "impedance_ohm": float(port.impedance),
                "pin_radius_m": float(port.pin_radius),
                "outer_radius_m": float(port.outer_radius),
                "pin_length_m": float(port.pin_length),
            },
        ),
        waveform={"kind": "GaussianPulse", "f0_hz": 5.0e9, "bandwidth": 0.8},
        dt_s=float(grid.dt),
        frequency_grid_hz=tuple(float(f) for f in np.asarray(freqs)),
        notes=(
            "Diagnostic V/I replay at the coaxial gap cell using low-level coaxial "
            "material/source helpers. This is not calibrated TEM coaxial-port "
            "S-parameter evidence and is not a promoted high-level API."
        ),
    )

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    save_port_vi_dump_npz(
        dump_path,
        voltages=voltages,
        currents=currents,
        freqs=np.asarray(freqs, dtype=np.float64),
        port_impedances=np.asarray([port.impedance], dtype=np.float64),
        metadata=metadata,
        port_names=("coax_gap",),
        driven_port_indices=(0,),
        production_smatrix=production_smatrix,
        drive_ref_voltages=drive_ref_voltages,
    )

    dump = load_port_vi_dump_npz(dump_path)
    replayed = replay_smatrix_from_port_vi_dump(dump)
    production = type(
        "CoaxialGapDiagnosticS11",
        (),
        {"s_params": dump.production_smatrix, "freqs": dump.freqs},
    )()
    comparison = compare_replayed_smatrix(replayed, production)
    summary = {
        "status": "passed" if comparison.ok else "failed",
        "dump": str(dump_path),
        "claim_scope": "coaxial gap V/I replay diagnostic only; no promoted coaxial S-parameter API",
        "n_steps": int(n_steps),
        "grid_shape": list(grid.shape),
        "gap_index": list(gap_idx),
        "component": component,
        "freqs_hz": np.asarray(freqs, dtype=np.float64).tolist(),
        "max_abs_diff": comparison.max_abs_diff,
        "max_allowed": comparison.max_allowed,
        "max_abs_s11": float(np.max(np.abs(production_smatrix))),
        "max_abs_voltage": float(np.max(np.abs(voltages))),
        "max_abs_current": float(np.max(np.abs(currents))),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if replay_json:
        replay_json.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("replay_port_vi_dump.py")),
                str(dump_path),
                "--write-json",
                str(replay_json),
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "coaxial replay_port_vi_dump.py failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--replay-json", type=Path)
    parser.add_argument("--n-steps", type=int, default=200)
    args = parser.parse_args(argv)
    summary = generate_coaxial_gap_vi_dump(
        dump_path=args.dump,
        summary_json=args.summary_json,
        replay_json=args.replay_json,
        n_steps=args.n_steps,
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
