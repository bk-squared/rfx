#!/usr/bin/env python3
"""Generate a real lumped-port V/I dump for independent S-matrix replay.

This diagnostic is intentionally narrow: it builds a small two-port uniform
Yee PEC cavity, runs the production lumped-port S-matrix extractor, and saves
the raw V/I phasors needed by ``scripts/diagnostics/replay_port_vi_dump.py``.
The dump is E3 infrastructure evidence only for this stated lane; it is not a
broad calibrated-port E5 envelope by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np

from rfx import (
    PortDumpMetadata,
    compare_replayed_smatrix,
    load_port_vi_dump_npz,
    replay_smatrix_from_port_vi_dump,
    save_port_vi_dump_npz,
)
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.probes.probes import extract_s_matrix
from rfx.sources.sources import GaussianPulse, LumpedPort


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _complex_metric(value: np.ndarray) -> float:
    arr = np.asarray(value)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        type=Path,
        required=True,
        help="output .npz path for the raw rfx.port_vi_dump",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="optional summary JSON with replay-vs-production metrics",
    )
    parser.add_argument("--freq-min", type=float, default=1.5e9)
    parser.add_argument("--freq-max", type=float, default=4.5e9)
    parser.add_argument("--n-freqs", type=int, default=9)
    parser.add_argument("--n-steps", type=int, default=0)
    parser.add_argument("--num-periods", type=float, default=60.0)
    args = parser.parse_args(argv)

    if args.n_freqs <= 0:
        raise ValueError("--n-freqs must be positive")

    domain = (0.05, 0.05, 0.025)
    dx = 2.5e-3
    grid = Grid(freq_max=5.0e9, domain=domain, dx=dx, cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=3.0e9, bandwidth=0.8, amplitude=1.0)
    ports = [
        LumpedPort(
            position=(domain[0] * 0.30, domain[1] * 0.50, domain[2] * 0.50),
            component="ez",
            impedance=50.0,
            excitation=pulse,
        ),
        LumpedPort(
            position=(domain[0] * 0.70, domain[1] * 0.50, domain[2] * 0.50),
            component="ez",
            impedance=50.0,
            excitation=pulse,
        ),
    ]
    freqs = jnp.linspace(args.freq_min, args.freq_max, args.n_freqs)
    n_steps = int(args.n_steps) if args.n_steps else grid.num_timesteps(args.num_periods)

    extraction = extract_s_matrix(
        grid,
        materials,
        ports,
        freqs,
        n_steps=n_steps,
        boundary="pec",
        return_vi_dump=True,
    )

    metadata = PortDumpMetadata(
        commit_hash=_git_commit(),
        # rfx's lumped production diagonal is the driven terminal
        # reflection, so the replay must read the dump in that frame.
        diagonal_frame="driven_terminal",
        geometry={
            "kind": "two_port_lumped_pec_cavity",
            "domain_m": list(domain),
            "dx_m": dx,
        },
        materials={"background_eps_r": 1.0, "background_sigma_s_per_m": 0.0},
        grid={
            "shape": list(grid.shape),
            "dx_m": float(grid.dx),
            "dt_s": float(grid.dt),
            "boundary": "pec",
            "cpml_layers": 0,
        },
        boundaries={"x": "pec", "y": "pec", "z": "pec"},
        port_definitions=tuple(
            {
                "name": name,
                "kind": "single_cell_lumped_port",
                "position_m": list(port.position),
                "component": port.component,
                "impedance_ohm": float(port.impedance),
            }
            for name, port in zip(extraction.port_names, ports)
        ),
        waveform={"kind": "GaussianPulse", "f0_hz": 3.0e9, "bandwidth": 0.8},
        dt_s=float(grid.dt),
        frequency_grid_hz=tuple(float(f) for f in np.asarray(freqs)),
        notes=(
            "Real uniform-Yee two-port lumped-port dump. Voltages are stored "
            "with the public replay sign convention (positive into DUT)."
        ),
    )

    args.dump.parent.mkdir(parents=True, exist_ok=True)
    save_port_vi_dump_npz(
        args.dump,
        voltages=extraction.voltages,
        currents=extraction.currents,
        freqs=np.asarray(extraction.freqs, dtype=np.float64),
        port_impedances=extraction.port_impedances,
        metadata=metadata,
        port_names=extraction.port_names,
        driven_port_indices=extraction.driven_port_indices,
        production_smatrix=np.asarray(extraction.s_params),
    )

    dump = load_port_vi_dump_npz(args.dump)
    replayed = replay_smatrix_from_port_vi_dump(dump)
    production = type(
        "ProductionSMatrix",
        (),
        {"s_params": dump.production_smatrix, "freqs": dump.freqs},
    )()
    comparison = compare_replayed_smatrix(replayed, production)

    summary = {
        "status": "passed" if comparison.ok else "failed",
        "dump": str(args.dump),
        "schema": dump.metadata.get("schema"),
        "claim_scope": "two-port uniform-Yee lumped-port E3 replay diagnostic",
        "n_steps": n_steps,
        "n_ports": comparison.n_ports,
        "n_freqs": comparison.n_freqs,
        "freq_min_hz": float(np.min(dump.freqs)),
        "freq_max_hz": float(np.max(dump.freqs)),
        "max_abs_diff": comparison.max_abs_diff,
        "max_allowed": comparison.max_allowed,
        "max_abs_s": _complex_metric(dump.production_smatrix),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return 0 if comparison.ok else 1


if __name__ == "__main__":
    sys.exit(main())
