"""Diagnostic only: isolate #931 volume-edge realization, without fixture tuning.

Run from this worktree: PYTHONPATH=$PWD JAX_PLATFORMS=cpu python
scripts/diagnostics/slow_931_farfield_attribution.py

Predeclared falsifier: if restoring ONLY the historical volume-edge map does
not bring this unchanged uniform/graded comparison back below 5%, volume-edge
ownership alone does not explain crossing the envelope. This is not a rollback
proposal or a newly qualified physical baseline. Four sequential 600-step runs.
Historical map is copied from a3e4dba4^:rfx/boundaries/pec.py tangential_edge_masks.
All patching is process-local, restored after each counterfactual.

Fixture repair qualification: --fixture repaired --operators current uses
identical 250 um radiator/source lattices and grades only the NTFF shoulders.
Its predeclared falsifier is local-cell error >= 5% OR >= scalar-cell error/2;
either miss remains a finding, with no envelope change or profile sweep.
Use --steps 1200 as an independently named duration-convergence witness.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import rfx
import rfx.boundaries.pec as pec
from rfx.farfield import compute_far_field, compute_far_field_jax
from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid, coords_from_uniform_grid
from rfx.nonuniform import NonUniformGrid, position_to_index
from rfx.sources import GaussianPulse
from tests._realized_geometry import realized
from tests.unit.farfield.test_farfield_inplane_nonuniform import _graded_profile, _sim


ROOT = Path.cwd().resolve()
assert Path(rfx.__file__).resolve().is_relative_to(ROOT), rfx.__file__
TH = np.radians(np.linspace(0.0, 180.0, 25))
PH = np.radians(np.linspace(0.0, 350.0, 12))
PROF = np.concatenate([np.full(30, 250e-6), np.full(8, 312.5e-6),
                       np.full(16, 125e-6), np.full(8, 312.5e-6),
                       np.full(30, 250e-6)])


def old_volume_edges(cells, periodic):
    return tuple(cells & (pec._shift(cells, a, periodic, +1)
                          | pec._shift(cells, a, periodic, -1))
                 for a in range(3))


@contextmanager
def operator(which):
    original = pec._volume_edge_masks
    if which == "legacy":
        pec._volume_edge_masks = old_volume_edges
    try:
        yield
    finally:
        pec._volume_edge_masks = original
        # Traces must not retain a previous operator for another case.
        jax.clear_caches()


def power(ff):
    u = np.abs(np.asarray(ff.E_theta)) ** 2 + np.abs(np.asarray(ff.E_phi)) ** 2
    p = np.sum(u * np.sin(TH)[None, :, None]
               * np.gradient(TH)[None, :, None]
               * np.gradient(PH)[None, None, :], axis=(1, 2))
    return u, float(p[0])


class NoInPlaneArrays:
    def __init__(self, grid):
        self.grid = grid

    def __getattr__(self, name):
        if name in ("dx_arr", "dy_arr"):
            raise AttributeError(name)
        return getattr(self.grid, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".validation-931-closures")
    parser.add_argument("--operators", nargs="+", default=["current", "legacy"],
                        choices=["current", "legacy"])
    parser.add_argument("--fixture", choices=["historical", "repaired"],
                        default="historical")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {"source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "rfx_import": rfx.__file__, "predeclared_falsifier": __doc__,
               "fixture": args.fixture, "steps": args.steps, "cases": {}}
    profile = _graded_profile() if args.fixture == "repaired" else PROF
    arrays = {"theta": TH, "phi": PH}
    for op in args.operators:
        for mesh in ("uniform", "graded"):
            key = op + "_" + mesh
            with operator(op):
                sim = _sim() if mesh == "uniform" else _sim(dx_profile=profile, dy_profile=profile)
                rz = realized(sim)
                grid = rz.grid
                nu = isinstance(grid, NonUniformGrid)
                coords = coords_from_nonuniform_grid(grid) if nu else coords_from_uniform_grid(grid)
                nodes = [np.asarray(v) for v in (coords.x, coords.y, coords.z)]
                old_cells = np.asarray(sim._geometry[0].shape.mask_on_coords(coords.x, coords.y, coords.z))
                source_idx = (position_to_index(grid, (11e-3, 11e-3, 9.5e-3)) if nu
                              else grid.position_to_index((11e-3, 11e-3, 9.5e-3)))
                pulse = GaussianPulse(f0=30e9, bandwidth=0.5)
                times = jnp.arange(args.steps, dtype=jnp.float32) * grid.dt
                waveform = np.asarray(pulse(times))
                source_dft = float(grid.dt) * np.sum(waveform * np.exp(-2j * np.pi * 30e9 * np.asarray(times)))
                if nu:
                    from rfx.nonuniform import e_node_dual_spacing_at
                    source_dv = (float(e_node_dual_spacing_at(np.asarray(grid.dx_arr), source_idx[0]))
                                 * float(e_node_dual_spacing_at(np.asarray(grid.dy_arr), source_idx[1]))
                                 * float(np.asarray(grid.dz)[source_idx[2]]))
                else:
                    source_dv = float(grid.dx) ** 3
                record = {"shape": list(grid.shape), "dt": float(grid.dt),
                          "duration": args.steps * float(grid.dt),
                          "occupied_cells": int(np.count_nonzero(rz.pec_mask)),
                          "node_vs_center_occupancy_xor": int(np.count_nonzero(old_cells != np.asarray(rz.pec_mask))),
                          "edge_counts": [int(np.count_nonzero(e)) for e in rz.edge_masks],
                          "wall_planes_m": [[float(nodes[a][i]) for i in rz.wall_planes(a)] for a in range(3)],
                          "source_index": [int(i) for i in source_idx],
                          "source_node_m": [float(nodes[a][source_idx[a]]) for a in range(3)],
                          "source_dual_volume_m3": source_dv,
                          "source_waveform_30ghz_dt_dft": [float(source_dft.real), float(source_dft.imag)],
                          "source_waveform_tail": float(waveform[-1])}
                assert record["node_vs_center_occupancy_xor"] == 0, (
                    "Legacy-edge-only attribution requires identical occupancy")
                arrays[key + "_source_times"] = np.asarray(times)
                arrays[key + "_source_waveform"] = waveform
                payload["cases"][key] = record
                if not args.structural_only:
                    res = sim.run(n_steps=args.steps)
                    assert res.grid.shape == grid.shape and res.grid.dt == grid.dt
                    final_coords = (coords_from_nonuniform_grid(res.grid) if nu
                                    else coords_from_uniform_grid(res.grid))
                    for before, after in zip(nodes, (final_coords.x, final_coords.y, final_coords.z)):
                        np.testing.assert_array_equal(before, np.asarray(after))
                    grid = res.grid
                    ff = compute_far_field_jax(res.ntff_data, res.ntff_box, grid, TH, PH)
                    u, record["power"] = power(ff)
                    ff_np = compute_far_field(res.ntff_data, res.ntff_box, grid, TH, PH)
                    u_np, record["numpy_power"] = power(ff_np)
                    record["numpy_jax_max_U_relative"] = float(np.max(np.abs(u_np - u)) / np.max(u_np))
                    old_ff = compute_far_field_jax(res.ntff_data, res.ntff_box,
                                                   NoInPlaneArrays(grid), TH, PH)
                    _, record["scalar_inplane_power"] = power(old_ff)
                    record["ntff_indices"] = [int(getattr(res.ntff_box, n)) for n in
                                               ("i_lo", "i_hi", "j_lo", "j_hi", "k_lo", "k_hi")]
                    record["ntff_coordinates_m"] = [[float(nodes[a][getattr(res.ntff_box, n)])
                                                     for n in pair] for a, pair in enumerate(
                                                     (("i_lo", "i_hi"), ("j_lo", "j_hi"), ("k_lo", "k_hi")))]
                    arrays[key + "_U"] = u
                    arrays[key + "_E_theta"] = np.asarray(ff.E_theta)
                    arrays[key + "_E_phi"] = np.asarray(ff.E_phi)
                    for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"):
                        arrays[key + "_ntff_" + face] = np.asarray(getattr(res.ntff_data, face))
                print(key, json.dumps(record), flush=True)
                suffix = "-structural" if args.structural_only else ""
                (out / ("farfield-attribution" + suffix + ".json")).write_text(json.dumps(payload, indent=2))
                np.savez_compressed(out / ("farfield-attribution" + suffix + ".npz"), **arrays)
    if not args.structural_only:
        for op in args.operators:
            u = payload["cases"][op + "_uniform"]["power"]
            g = payload["cases"][op + "_graded"]["power"]
            old = payload["cases"][op + "_graded"]["scalar_inplane_power"]
            uu = arrays[op + "_uniform_U"]
            ug = arrays[op + "_graded_U"]
            payload[op + "_errors"] = {"local": abs(g - u) / u, "scalar": abs(old - u) / u,
                                       "graded_over_uniform_power": g / u,
                                       "peak_normalized_pattern_max_abs": float(np.max(np.abs(
                                           ug / np.max(ug) - uu / np.max(uu)))),
                                       "power_normalized_pattern_relative_l2": float(np.linalg.norm(
                                           ug / g - uu / u) / np.linalg.norm(uu / u))}
            if args.fixture == "repaired":
                errors = payload[op + "_errors"]
                errors["unchanged_5pct_gate_passed"] = errors["local"] < 0.05
                errors["local_cell_discriminator_passed"] = errors["local"] < errors["scalar"] / 2
        (out / "farfield-attribution.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps({k: v for k, v in payload.items() if k.endswith("_errors")}), flush=True)


if __name__ == "__main__":
    main()
