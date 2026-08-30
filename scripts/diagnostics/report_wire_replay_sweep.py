#!/usr/bin/env python3
"""Run a small wire-port replay / invariant sweep envelope.

This report extends the M12 single-case wire-port raw V/I replay into a small
internal uniform-Yee sweep over geometry, mesh, and frequency choices.  It is an
E5-enabling artifact, not a completed broad calibrated wire-port E5 proof:
external absolute S11/S21 evidence and a bounded calibration convention remain
required.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.probes.probes import extract_s_matrix_wire
from rfx.sources.sources import GaussianPulse, WirePort


def _load_wire_replay():
    path = Path(__file__).resolve().with_name("replay_wire_port_vi_dump.py")
    spec = importlib.util.spec_from_file_location("replay_wire_port_vi_dump", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("replay_wire_port_vi_dump", module)
    spec.loader.exec_module(module)
    return module.replay_wire_port_vi_dump


replay_wire_port_vi_dump = _load_wire_replay()


@dataclass(frozen=True)
class WireSweepCase:
    name: str
    domain_m: tuple[float, float, float]
    dx_m: float
    freq_max_hz: float
    freqs_hz: tuple[float, ...]
    port1_start_m: tuple[float, float, float]
    port1_end_m: tuple[float, float, float]
    port2_start_m: tuple[float, float, float]
    port2_end_m: tuple[float, float, float]
    num_periods: float = 25.0
    n_steps: int | None = None


DEFAULT_CASES = (
    WireSweepCase(
        name="coarse_three_cell",
        domain_m=(0.030, 0.026, 0.012),
        dx_m=2.0e-3,
        freq_max_hz=5.0e9,
        freqs_hz=(1.5e9, 2.0e9, 2.5e9, 3.0e9, 3.5e9),
        port1_start_m=(0.010, 0.014, 0.004),
        port1_end_m=(0.010, 0.014, 0.008),
        port2_start_m=(0.020, 0.014, 0.004),
        port2_end_m=(0.020, 0.014, 0.008),
    ),
    WireSweepCase(
        name="finer_three_cell",
        domain_m=(0.028, 0.024, 0.012),
        dx_m=1.5e-3,
        freq_max_hz=5.0e9,
        freqs_hz=(1.5e9, 2.0e9, 2.5e9, 3.0e9, 3.5e9),
        port1_start_m=(0.009, 0.012, 0.003),
        port1_end_m=(0.009, 0.012, 0.006),
        port2_start_m=(0.0195, 0.012, 0.003),
        port2_end_m=(0.0195, 0.012, 0.006),
    ),
    WireSweepCase(
        name="longer_spacing",
        domain_m=(0.040, 0.028, 0.012),
        dx_m=2.0e-3,
        freq_max_hz=5.0e9,
        freqs_hz=(1.2e9, 1.8e9, 2.4e9, 3.0e9, 3.6e9),
        port1_start_m=(0.010, 0.014, 0.004),
        port1_end_m=(0.010, 0.014, 0.008),
        port2_start_m=(0.030, 0.014, 0.004),
        port2_end_m=(0.030, 0.014, 0.008),
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


def _metadata_json(*, case: WireSweepCase, grid: Grid, ports: list[WirePort], n_steps: int) -> str:
    from rfx.sources.sources import _wire_port_cells

    return json.dumps(
        {
            "schema": "rfx.wire_port_vi_dump",
            "schema_version": 1,
            "commit_hash": _git_commit(),
            "geometry": {
                "kind": "two_wire_port_pec_cavity_sweep",
                "case": case.name,
                "domain_m": list(case.domain_m),
                "dx_m": float(case.dx_m),
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
            "frequency_grid_hz": [float(f) for f in case.freqs_hz],
            "raw_phasor_type": "wire midpoint V/I DFT",
            "voltage_convention": "FDTD sign V=-E*d_parallel at wire midpoint",
            "current_convention": "positive_into_dut midpoint Ampere-loop current",
            "production_smatrix_schema": "S[receiver_port, driven_port, frequency_index]",
            "notes": (
                "Internal sweep for wire-port replay/passivity/reciprocity. "
                "Replay mirrors the current legacy diagonal-total-impedance / "
                "off-diagonal per-cell convention."
            ),
        }
    )


def _run_case(case: WireSweepCase, output_dir: Path) -> dict:
    grid = Grid(
        freq_max=case.freq_max_hz,
        domain=case.domain_m,
        dx=case.dx_m,
        cpml_layers=0,
    )
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=2.5e9, bandwidth=0.8, amplitude=1.0)
    ports = [
        WirePort(case.port1_start_m, case.port1_end_m, "ez", 50.0, pulse),
        WirePort(case.port2_start_m, case.port2_end_m, "ez", 50.0, pulse),
    ]
    freqs = jnp.asarray(case.freqs_hz, dtype=jnp.float32)
    n_steps = int(case.n_steps) if case.n_steps is not None else grid.num_timesteps(case.num_periods)

    extraction = extract_s_matrix_wire(
        grid,
        materials,
        ports,
        freqs,
        n_steps=n_steps,
        boundary="pec",
        return_vi_dump=True,
    )

    dump_path = output_dir / f"{case.name}_raw_vi_dump.npz"
    np.savez(
        dump_path,
        metadata_json=np.asarray(_metadata_json(case=case, grid=grid, ports=ports, n_steps=n_steps)),
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

    replay = replay_wire_port_vi_dump(dump_path)
    s = np.asarray(extraction.s_params, dtype=np.complex128)
    column_power = np.sum(np.abs(s) ** 2, axis=0)
    reciprocity = np.abs(s[1, 0, :] - s[0, 1, :])
    return {
        "case": asdict(case),
        "dump": str(dump_path),
        "status": replay["status"],
        "n_steps": int(n_steps),
        "grid_shape": list(grid.shape),
        "replay_max_abs_diff": float(replay["max_abs_diff"]),
        "replay_max_allowed": float(replay["max_allowed"]),
        "max_column_power": float(np.max(column_power)) if column_power.size else 0.0,
        "mean_column_power": float(np.mean(column_power)) if column_power.size else 0.0,
        "max_reciprocity_abs_diff": float(np.max(reciprocity)) if reciprocity.size else 0.0,
        "max_abs_s": float(np.max(np.abs(s))) if s.size else 0.0,
    }


def run_wire_replay_sweep(
    *,
    output_dir: Path,
    cases: Iterable[WireSweepCase] = DEFAULT_CASES,
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
            "wire-port uniform-Yee replay/passivity/reciprocity sweep; "
            "external absolute S-parameter evidence still required for broad E5"
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
        "# Wire-port replay sweep report",
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

    payload = run_wire_replay_sweep(
        output_dir=args.output_dir,
        passivity_limit=args.passivity_limit,
        reciprocity_limit=args.reciprocity_limit,
    )
    json_path = args.output_dir / "wire_replay_sweep_report.json"
    md_path = args.output_dir / "wire_replay_sweep_report.md"
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
