#!/usr/bin/env python3
"""Generate a real wire-port V/I dump for independent replay.

This diagnostic builds a small two-wire-port uniform-Yee PEC cavity, runs the
production wire S-matrix extractor with raw midpoint-cell V/I capture enabled,
and writes an ``rfx.wire_port_vi_dump`` file replayable by
``scripts/diagnostics/replay_wire_port_vi_dump.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.probes.probes import extract_s_matrix_wire
from rfx.sources.sources import GaussianPulse, WirePort

from replay_wire_port_vi_dump import replay_wire_port_vi_dump


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _metadata_json(*, grid: Grid, ports: list[WirePort], n_steps: int, freqs: np.ndarray) -> str:
    from rfx.sources.sources import _wire_port_cells

    metadata = {
        "schema": "rfx.wire_port_vi_dump",
        # Issue #770: dumps written by the current extractor record the
        # whole-port off-diagonal frame; absence of the tag marks a
        # pre-#770 dump (per-cell #308 frame replay).
        "offdiag_frame": "wholeport",
        "schema_version": 1,
        "commit_hash": _git_commit(),
        "geometry": {
            "kind": "two_wire_port_pec_cavity",
            "domain_m": list(grid.domain),
            "dx_m": float(grid.dx),
        },
        "grid": {
            "shape": list(grid.shape),
            "dt_s": float(grid.dt),
            "n_steps": int(n_steps),
            "boundary": "pec",
            "cpml_layers": 0,
        },
        "materials": {"background_eps_r": 1.0, "background_sigma_s_per_m": 0.0},
        "boundaries": {"x": "pec", "y": "pec", "z": "pec"},
        "port_definitions": [
            {
                "name": f"wire_{idx}",
                "kind": "wire_port_midpoint_vi",
                "start_m": list(port.start),
                "end_m": list(port.end),
                "component": port.component,
                "impedance_ohm": float(port.impedance),
                "cell_count": int(len(_wire_port_cells(grid, port))),
            }
            for idx, port in enumerate(ports)
        ],
        "waveform": {"kind": "GaussianPulse", "f0_hz": 2.5e9, "bandwidth": 0.8},
        "frequency_grid_hz": [float(f) for f in freqs],
        "raw_phasor_type": "wire midpoint V/I DFT",
        "voltage_convention": "FDTD sign V=-E*d_parallel at wire midpoint",
        "current_convention": "positive_into_dut midpoint Ampere-loop current",
        "production_smatrix_schema": "S[receiver_port, driven_port, frequency_index]",
        "notes": (
            "Wire-port replay intentionally mirrors the current legacy "
            "calibration convention: diagonal entries use total port "
            "impedance; off-diagonal entries use per-cell impedance."
        ),
    }
    return json.dumps(metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--freq-min", type=float, default=1.5e9)
    parser.add_argument("--freq-max", type=float, default=4.5e9)
    parser.add_argument("--n-freqs", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=0)
    parser.add_argument("--num-periods", type=float, default=35.0)
    args = parser.parse_args(argv)

    if args.n_freqs <= 0:
        raise ValueError("--n-freqs must be positive")

    domain = (0.03, 0.026, 0.012)
    dx = 2.0e-3
    grid = Grid(freq_max=5.0e9, domain=domain, dx=dx, cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=2.5e9, bandwidth=0.8, amplitude=1.0)
    ports = [
        WirePort(
            start=(0.010, 0.014, 0.004),
            end=(0.010, 0.014, 0.008),
            component="ez",
            impedance=50.0,
            excitation=pulse,
        ),
        WirePort(
            start=(0.020, 0.014, 0.004),
            end=(0.020, 0.014, 0.008),
            component="ez",
            impedance=50.0,
            excitation=pulse,
        ),
    ]
    freqs = jnp.linspace(args.freq_min, args.freq_max, args.n_freqs)
    n_steps = int(args.n_steps) if args.n_steps else grid.num_timesteps(args.num_periods)

    extraction = extract_s_matrix_wire(
        grid,
        materials,
        ports,
        freqs,
        n_steps=n_steps,
        boundary="pec",
        return_vi_dump=True,
    )

    args.dump.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.dump,
        metadata_json=np.asarray(
            _metadata_json(grid=grid, ports=ports, n_steps=n_steps, freqs=np.asarray(freqs))
        ),
        freqs_hz=np.asarray(extraction.freqs, dtype=np.float64),
        raw_voltages_fdt=np.asarray(extraction.raw_voltages_fdt, dtype=np.complex128),
        # Whole-port gap-voltage channel (issue #764): required to replay
        # the driven whole-port diagonal of a post-#764 production dump.
        raw_port_voltages_fdt=np.asarray(
            extraction.raw_port_voltages_fdt, dtype=np.complex128),
        raw_currents=np.asarray(extraction.raw_currents, dtype=np.complex128),
        port_impedances_ohm=np.asarray(extraction.port_impedances, dtype=np.float64),
        port_cell_counts=np.asarray(extraction.port_cell_counts, dtype=np.int64),
        production_smatrix=np.asarray(extraction.s_params, dtype=np.complex128),
        port_names=np.asarray(extraction.port_names, dtype=object),
        driven_port_indices=np.asarray(extraction.driven_port_indices, dtype=np.int64),
    )

    summary = replay_wire_port_vi_dump(args.dump)
    summary.update(
        {
            "claim_scope": "two-port uniform-Yee wire-port E3 replay diagnostic",
            "n_steps": n_steps,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
