#!/usr/bin/env python3
"""Run a small lumped-port replay / invariant sweep envelope.

This report is an E5-enabling artifact, not a completed E5 proof.  It sweeps
several uniform-Yee two-port lumped configurations, saves replayable V/I dumps,
and records independent replay agreement plus passivity / reciprocity metrics.
External-solver evidence is still required before broad calibrated-port E5
claims.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Iterable

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


@dataclass(frozen=True)
class LumpedSweepCase:
    name: str
    domain_m: tuple[float, float, float]
    dx_m: float
    freq_max_hz: float
    freqs_hz: tuple[float, ...]
    port1_pos_m: tuple[float, float, float]
    port2_pos_m: tuple[float, float, float]
    num_periods: float = 30.0
    n_steps: int | None = None


DEFAULT_CASES = (
    LumpedSweepCase(
        name="coarse_short_box",
        domain_m=(0.030, 0.020, 0.015),
        dx_m=5.0e-3,
        freq_max_hz=2.0e9,
        freqs_hz=(0.8e9, 1.0e9, 1.2e9, 1.5e9, 1.8e9),
        port1_pos_m=(0.010, 0.010, 0.005),
        port2_pos_m=(0.020, 0.010, 0.005),
    ),
    LumpedSweepCase(
        name="finer_short_box",
        domain_m=(0.030, 0.020, 0.015),
        dx_m=4.0e-3,
        freq_max_hz=2.0e9,
        freqs_hz=(0.8e9, 1.0e9, 1.2e9, 1.5e9, 1.8e9),
        port1_pos_m=(0.008, 0.012, 0.008),
        port2_pos_m=(0.020, 0.012, 0.008),
    ),
    LumpedSweepCase(
        name="longer_spacing_box",
        domain_m=(0.040, 0.024, 0.016),
        dx_m=5.0e-3,
        freq_max_hz=2.0e9,
        freqs_hz=(0.7e9, 1.0e9, 1.3e9, 1.6e9, 1.9e9),
        port1_pos_m=(0.010, 0.012, 0.005),
        port2_pos_m=(0.030, 0.012, 0.005),
    ),
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _case_metadata(case: LumpedSweepCase, grid: Grid, n_steps: int) -> PortDumpMetadata:
    return PortDumpMetadata(
        commit_hash=_git_commit(),
        geometry={
            "kind": "two_port_lumped_pec_cavity_sweep",
            "case": case.name,
            "domain_m": list(case.domain_m),
            "dx_m": case.dx_m,
        },
        materials={"background_eps_r": 1.0, "background_sigma_s_per_m": 0.0},
        grid={
            "shape": list(grid.shape),
            "dx_m": float(grid.dx),
            "dt_s": float(grid.dt),
            "n_steps": int(n_steps),
            "boundary": "pec",
            "cpml_layers": 0,
        },
        boundaries={"x": "pec", "y": "pec", "z": "pec"},
        # rfx's lumped production diagonal is the driven terminal
        # reflection, so the replay must read the dump in that frame.
        diagonal_frame="driven_terminal",
        port_definitions=(
            {
                "name": "port_0",
                "kind": "single_cell_lumped_port",
                "position_m": list(case.port1_pos_m),
                "component": "ez",
                "impedance_ohm": 50.0,
            },
            {
                "name": "port_1",
                "kind": "single_cell_lumped_port",
                "position_m": list(case.port2_pos_m),
                "component": "ez",
                "impedance_ohm": 50.0,
            },
        ),
        waveform={"kind": "GaussianPulse", "f0_hz": 1.0e9, "bandwidth": 0.8},
        dt_s=float(grid.dt),
        frequency_grid_hz=tuple(float(f) for f in case.freqs_hz),
        notes=(
            "Sweep case for lumped-port replay/passivity/reciprocity envelope. "
            "This is not external-solver evidence."
        ),
    )


def _run_case(case: LumpedSweepCase, output_dir: Path) -> dict:
    grid = Grid(
        freq_max=case.freq_max_hz,
        domain=case.domain_m,
        dx=case.dx_m,
        cpml_layers=0,
    )
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=1.0e9, bandwidth=0.8, amplitude=1.0)
    ports = [
        LumpedPort(case.port1_pos_m, "ez", 50.0, pulse),
        LumpedPort(case.port2_pos_m, "ez", 50.0, pulse),
    ]
    freqs = jnp.asarray(case.freqs_hz, dtype=jnp.float32)
    n_steps = int(case.n_steps) if case.n_steps is not None else grid.num_timesteps(case.num_periods)

    extraction = extract_s_matrix(
        grid,
        materials,
        ports,
        freqs,
        n_steps=n_steps,
        boundary="pec",
        return_vi_dump=True,
    )

    dump_path = output_dir / f"{case.name}_raw_vi_dump.npz"
    save_port_vi_dump_npz(
        dump_path,
        voltages=extraction.voltages,
        currents=extraction.currents,
        freqs=np.asarray(extraction.freqs, dtype=np.float64),
        port_impedances=extraction.port_impedances,
        metadata=_case_metadata(case, grid, n_steps),
        port_names=extraction.port_names,
        driven_port_indices=extraction.driven_port_indices,
        production_smatrix=np.asarray(extraction.s_params, dtype=np.complex128),
        # The production off-diagonal incident wave is built on the
        # pre-injection drive sample; without this channel the replay
        # cannot reproduce S21.
        drive_ref_voltages=extraction.drive_ref_voltages,
    )

    dump = load_port_vi_dump_npz(dump_path)
    replayed = replay_smatrix_from_port_vi_dump(dump)
    production = type(
        "ProductionSMatrix",
        (),
        {"s_params": dump.production_smatrix, "freqs": dump.freqs},
    )()
    comparison = compare_replayed_smatrix(replayed, production)

    s = np.asarray(dump.production_smatrix, dtype=np.complex128)
    column_power = np.sum(np.abs(s) ** 2, axis=0)
    reciprocity = np.abs(s[1, 0, :] - s[0, 1, :])
    return {
        "case": asdict(case),
        "dump": str(dump_path),
        "status": "passed" if comparison.ok else "failed",
        "n_steps": int(n_steps),
        "grid_shape": list(grid.shape),
        "replay_max_abs_diff": comparison.max_abs_diff,
        "replay_max_allowed": comparison.max_allowed,
        "max_column_power": float(np.max(column_power)) if column_power.size else 0.0,
        "mean_column_power": float(np.mean(column_power)) if column_power.size else 0.0,
        "max_reciprocity_abs_diff": float(np.max(reciprocity)) if reciprocity.size else 0.0,
        "max_abs_s": float(np.max(np.abs(s))) if s.size else 0.0,
    }


def run_lumped_replay_sweep(
    *,
    output_dir: Path,
    cases: Iterable[LumpedSweepCase] = DEFAULT_CASES,
    passivity_limit: float = 1.25,
    reciprocity_limit: float = 5.0e-2,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_run_case(case, output_dir) for case in cases]
    for row in rows:
        if row["max_column_power"] > passivity_limit:
            row["status"] = "failed"
            row["failure_reason"] = "passivity_limit"
        if row["max_reciprocity_abs_diff"] > reciprocity_limit:
            row["status"] = "failed"
            row["failure_reason"] = "reciprocity_limit"
    return {
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
        "claim_scope": (
            "lumped-port uniform-Yee replay/passivity/reciprocity sweep; "
            "external solver evidence still required for broad E5"
        ),
        "passivity_limit": float(passivity_limit),
        "reciprocity_limit": float(reciprocity_limit),
        "cases": rows,
        "max_replay_abs_diff": max((row["replay_max_abs_diff"] for row in rows), default=0.0),
        "max_replay_allowed": max((row["replay_max_allowed"] for row in rows), default=0.0),
        "max_column_power": max((row["max_column_power"] for row in rows), default=0.0),
        "max_reciprocity_abs_diff": max((row["max_reciprocity_abs_diff"] for row in rows), default=0.0),
    }


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Lumped-port replay sweep report",
        "",
        f"- status: `{payload['status']}`",
        f"- claim_scope: {payload['claim_scope']}",
        f"- passivity_limit: `{payload['passivity_limit']}`",
        f"- reciprocity_limit: `{payload['reciprocity_limit']}`",
        f"- max_replay_abs_diff: `{payload['max_replay_abs_diff']:.6g}`",
        f"- max_replay_allowed: `{payload['max_replay_allowed']:.6g}`",
        f"- max_column_power: `{payload['max_column_power']:.6g}`",
        f"- max_reciprocity_abs_diff: `{payload['max_reciprocity_abs_diff']:.6g}`",
        "",
        "| Case | Status | Grid | Steps | Replay diff | Replay allowed | Max column power | Max reciprocity diff | Dump |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["cases"]:
        lines.append(
            "| "
            f"`{row['case']['name']}` | `{row['status']}` | "
            f"`{row['grid_shape']}` | `{row['n_steps']}` | "
            f"`{row['replay_max_abs_diff']:.6g}` | `{row['replay_max_allowed']:.6g}` | "
            f"`{row['max_column_power']:.6g}` | `{row['max_reciprocity_abs_diff']:.6g}` | "
            f"`{row['dump']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--passivity-limit", type=float, default=1.25)
    parser.add_argument("--reciprocity-limit", type=float, default=5.0e-2)
    args = parser.parse_args(argv)

    payload = run_lumped_replay_sweep(
        output_dir=args.output_dir,
        passivity_limit=args.passivity_limit,
        reciprocity_limit=args.reciprocity_limit,
    )
    json_path = args.output_dir / "lumped_replay_sweep_report.json"
    md_path = args.output_dir / "lumped_replay_sweep_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(
        f"status={payload['status']} replay={payload['max_replay_abs_diff']:.6g}/"
        f"{payload['max_replay_allowed']:.6g} column_power={payload['max_column_power']:.6g} "
        f"reciprocity={payload['max_reciprocity_abs_diff']:.6g}"
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
