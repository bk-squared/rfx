"""Issue #88 step-1 dump: dev-run vs ref-run wave amplitudes + materials slice
for the dielectric-slab iris on the NU waveguide path.

Investigates the issue body claim:
  "device-run b ≈ reference-run b, so b_dev - b_ref cancels"

For each num_periods in {2, 4, 8} we drive port 0 with port 1 passive, run
the NU path twice (dev with slab + ref with vacuum override), then dump:
  - raw complex a_inc_ref, b_ref_diag, b_dev_diag at the driven port
  - raw complex a_inc_other, b_ref_off, b_dev_off at the passive port
  - cancellation residual |b_dev - b_ref| / |a_inc| (= |S11| diagonal)
  - materials.eps_r slice through (y_port, z_port) to verify slab presence

The driver runs each num_periods in a fresh subprocess to isolate any
cross-call JAX/rfx state pollution that was observed when batching them in
one Python invocation.

Outputs:
  .omx/diagnostics/issue_88/probe_nu_nontrivial_b_cancellation.json
  .omx/diagnostics/issue_88/materials_slice.npz
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from dataclasses import replace as _dc_replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.auto_config import smooth_grading
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.runners.nonuniform import (
    assemble_materials_nu,
    build_nonuniform_grid,
    pos_to_nu_index,
    run_nonuniform_path,
)
from rfx.sources.waveguide_port import (
    extract_waveguide_port_waves,
    waveguide_plane_positions,
)


_A_WG = 0.02286
_B_WG = 0.01016
_F_MAX = 12e9
_SLAB_X_LO = 0.045
_SLAB_X_HI = 0.049
_SLAB_EPS_R = 4.0
_FREQS = jnp.linspace(8.2e9, 12.4e9, 5)


def _worker(num_periods: int, out_npz: str, want_materials: int):
    """Single-num_periods worker invoked in a subprocess via _WORKER_CMD."""
    dx_coarse = 1.5e-3
    dx_fine = 0.75e-3
    n_pre = int(round(0.030 / dx_coarse))
    n_fine = int(round(0.040 / dx_fine))
    n_post = int(round(0.030 / dx_coarse))
    raw = np.concatenate([
        np.full(n_pre, dx_coarse),
        np.full(n_fine, dx_fine),
        np.full(n_post, dx_coarse),
    ])
    dx_profile = smooth_grading(raw, max_ratio=1.3)
    domain_x = float(np.sum(dx_profile))

    sim = Simulation(
        freq_max=_F_MAX,
        domain=(domain_x, _A_WG, _B_WG),
        dx=dx_coarse,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
        dx_profile=dx_profile,
    )
    sim.add_material("diel_slab", eps_r=_SLAB_EPS_R, sigma=0.0)
    sim.add(
        Box((_SLAB_X_LO, 0.0, 0.0), (_SLAB_X_HI, _A_WG, _B_WG)),
        material="diel_slab",
    )
    sim.add_waveguide_port(
        0.015, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=0.020, name="left",
    )
    sim.add_waveguide_port(
        domain_x - 0.015, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=domain_x - 0.020, name="right",
    )

    _nz = int(round(float(sim._domain[2]) / float(sim._dx)))
    sim._dz_profile = np.full(max(_nz, 1), float(sim._dx))

    pec_set = sim._boundary_spec.pec_faces() or set()
    pmc_set = sim._boundary_spec.pmc_faces() or set()

    def _axis_fully_closed(ax):
        return {f"{ax}_lo", f"{ax}_hi"}.issubset(pec_set | pmc_set)

    cpml_axes = "".join(
        ax for ax in "xyz" if not _axis_fully_closed(ax)
    )

    grid = build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
        sim._dz_profile,
        dx_profile=sim._dx_profile,
        pec_faces=pec_set or None,
        pmc_faces=pmc_set or None,
        cpml_axes=cpml_axes,
    )
    n_steps = int(np.ceil(num_periods / sim._freq_max / float(grid.dt)))

    entries = list(sim._waveguide_ports)
    sim._waveguide_ports = [
        _dc_replace(e, amplitude=(e.amplitude if i == 0 else 0.0))
        for i, e in enumerate(entries)
    ]

    dev_mats, _, _, _ = assemble_materials_nu(sim, grid)
    vacuum_eps = jnp.ones_like(dev_mats.eps_r)
    vacuum_sigma = jnp.zeros_like(dev_mats.sigma)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dev_result = run_nonuniform_path(sim, n_steps=n_steps)
        ref_result = run_nonuniform_path(
            sim, n_steps=n_steps,
            eps_override=vacuum_eps,
            sigma_override=vacuum_sigma,
        )

    cfg_dev_left = dev_result.waveguide_ports["left"]
    cfg_dev_right = dev_result.waveguide_ports["right"]
    cfg_ref_left = ref_result.waveguide_ports["left"]
    cfg_ref_right = ref_result.waveguide_ports["right"]

    planes_left = waveguide_plane_positions(cfg_dev_left)
    planes_right = waveguide_plane_positions(cfg_dev_right)
    ref_shift_left = entries[0].reference_plane - planes_left["reference"]
    ref_shift_right = entries[1].reference_plane - planes_right["reference"]

    a_inc_left, _ = extract_waveguide_port_waves(cfg_ref_left, ref_shift=ref_shift_left)
    _, b_ref_left = extract_waveguide_port_waves(cfg_ref_left, ref_shift=ref_shift_left)
    _, b_dev_left = extract_waveguide_port_waves(cfg_dev_left, ref_shift=ref_shift_left)
    _, b_ref_right = extract_waveguide_port_waves(cfg_ref_right, ref_shift=ref_shift_right)
    _, b_dev_right = extract_waveguide_port_waves(cfg_dev_right, ref_shift=ref_shift_right)

    a = np.array(a_inc_left)
    br_d = np.array(b_ref_left)
    bd_d = np.array(b_dev_left)
    br_o = np.array(b_ref_right)
    bd_o = np.array(b_dev_right)

    save_payload = dict(
        num_periods=int(num_periods),
        n_steps=int(n_steps),
        dt=float(grid.dt),
        domain_x=float(domain_x),
        freqs=np.array(_FREQS),
        ref_shift_left=float(ref_shift_left),
        ref_shift_right=float(ref_shift_right),
        a_inc_left=a,
        b_ref_left=br_d,
        b_dev_left=bd_d,
        b_ref_right=br_o,
        b_dev_right=bd_o,
    )
    if want_materials:
        port_pos = (0.015, _A_WG * 0.5, _B_WG * 0.5)
        _, j_p, k_p = pos_to_nu_index(grid, port_pos)
        eps_dev_xline = np.array(dev_mats.eps_r[:, j_p, k_p])
        dx_arr_np = np.asarray(grid.dx_arr)
        pad_lo = int(getattr(grid, "pad_x_lo", 0))
        pad_hi = int(getattr(grid, "pad_x_hi", 0))
        edges_x = np.insert(np.cumsum(dx_arr_np), 0, 0.0)
        cell_centers_x_pad = 0.5 * (edges_x[:-1] + edges_x[1:])
        user_origin = edges_x[pad_lo]
        cell_centers_x_user = cell_centers_x_pad - user_origin
        save_payload.update(dict(
            dx_arr=dx_arr_np,
            eps_dev_xline=eps_dev_xline,
            edges_x_pad=edges_x,
            cell_centers_x_user=cell_centers_x_user,
            pad_lo=pad_lo,
            pad_hi=pad_hi,
        ))
    np.savez(out_npz, **save_payload)


def _driver():
    out_dir = Path(".omx/diagnostics/issue_88")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    for i, num_periods in enumerate((2, 4, 8)):
        out_npz = out_dir / f"_worker_N{num_periods}.npz"
        want_mat = 1 if i == 0 else 0
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                str(num_periods),
                str(out_npz),
                str(want_mat),
            ],
            check=True,
        )
        per_run.append(np.load(out_npz, allow_pickle=False))

    sweep = []
    for r in per_run:
        a = r["a_inc_left"]
        br_d = r["b_ref_left"]
        bd_d = r["b_dev_left"]
        br_o = r["b_ref_right"]
        bd_o = r["b_dev_right"]
        safe_a = np.where(np.abs(a) > 1e-30, a, 1.0 + 0.0j)
        s11_diag = (bd_d - br_d) / safe_a
        safe_br_o = np.where(np.abs(br_o) > 1e-60, br_o, 1.0 + 0.0j)
        s21_off = bd_o / safe_br_o
        sweep.append({
            "num_periods": int(r["num_periods"]),
            "n_steps": int(r["n_steps"]),
            "dt_ns": float(r["dt"]) * 1e9,
            "total_time_ns": float(r["dt"]) * float(r["n_steps"]) * 1e9,
            "freqs_ghz": (r["freqs"] / 1e9).tolist(),
            "ref_shift_left": float(r["ref_shift_left"]),
            "ref_shift_right": float(r["ref_shift_right"]),
            "abs_a_inc_left": np.abs(a).tolist(),
            "abs_b_ref_left": np.abs(br_d).tolist(),
            "abs_b_dev_left": np.abs(bd_d).tolist(),
            "abs_diff_b_left": np.abs(bd_d - br_d).tolist(),
            "S11_diag": np.abs(s11_diag).tolist(),
            "abs_b_ref_right": np.abs(br_o).tolist(),
            "abs_b_dev_right": np.abs(bd_o).tolist(),
            "S21_off": np.abs(s21_off).tolist(),
            "ratio_dev_over_ref_left": (np.abs(bd_d) / np.where(np.abs(br_d) > 1e-60, np.abs(br_d), 1.0)).tolist(),
            "ratio_dev_over_ref_right": (np.abs(bd_o) / np.where(np.abs(br_o) > 1e-60, np.abs(br_o), 1.0)).tolist(),
        })

    r0 = per_run[0]
    pad_lo = int(r0["pad_lo"])
    pad_hi = int(r0["pad_hi"])
    eps_xline = r0["eps_dev_xline"]
    cell_centers_x_user = r0["cell_centers_x_user"]
    dx_arr = r0["dx_arr"]
    edges_x = r0["edges_x_pad"]

    np.savez(
        out_dir / "materials_slice.npz",
        edges_x_pad=edges_x,
        cell_centers_x_user=cell_centers_x_user,
        dx_arr=dx_arr,
        eps_dev_xline=eps_xline,
        pad_lo=pad_lo,
        pad_hi=pad_hi,
    )

    slab_cells_idx = np.where(eps_xline > 1.5)[0]
    slab_user_x = cell_centers_x_user[slab_cells_idx]
    in_user_box = (
        (slab_user_x >= _SLAB_X_LO - 1e-4)
        & (slab_user_x <= _SLAB_X_HI + 1e-4)
    )

    summary = {
        "domain_x": float(r0["domain_x"]),
        "n_cells_x": int(dx_arr.shape[0]),
        "pad_lo": pad_lo,
        "pad_hi": pad_hi,
        "dx_min": float(np.min(dx_arr)),
        "dx_max": float(np.max(dx_arr)),
        "slab_cells_count": int(slab_cells_idx.size),
        "slab_cell_indices": slab_cells_idx.tolist(),
        "slab_cell_user_x": slab_user_x.tolist(),
        "slab_user_box_target": [_SLAB_X_LO, _SLAB_X_HI],
        "slab_rasterized_correctly": bool(
            in_user_box.all() and in_user_box.size > 0
        ),
        "sweep": sweep,
    }

    out_json = out_dir / "probe_nu_nontrivial_b_cancellation.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Cleanup the worker npz files (already merged into materials_slice + json).
    for num_periods in (2, 4, 8):
        p = out_dir / f"_worker_N{num_periods}.npz"
        if p.exists():
            p.unlink()

    print(f"wrote {out_json}")
    print(f"wrote {out_dir / 'materials_slice.npz'}")
    print()
    print("[MATERIALS]")
    print(f"  pad_x_lo={pad_lo}, pad_x_hi={pad_hi}")
    print(f"  slab cells (eps_r>1.5): {summary['slab_cells_count']} at user_x={summary['slab_cell_user_x']}")
    print(f"  user box target: {summary['slab_user_box_target']}")
    print(f"  rasterized correctly: {summary['slab_rasterized_correctly']}")
    print()
    print("[SWEEP]")
    print(f"{'N':>3}  {'dt(ps)':>7}  {'T(ns)':>7}  {'|a_inc|':>10}  "
          f"{'|b_ref_L|':>10}  {'|b_dev_L|':>10}  {'|diff_L|':>10}  {'S11_diag':>9}  {'b_dev/b_ref':>11}")
    for row in sweep:
        a = np.mean(row["abs_a_inc_left"])
        bL = np.mean(row["abs_b_ref_left"])
        bD = np.mean(row["abs_b_dev_left"])
        df = np.mean(row["abs_diff_b_left"])
        s11 = np.mean(row["S11_diag"])
        ratio = np.mean(row["ratio_dev_over_ref_left"])
        print(f"{row['num_periods']:>3}  {row['dt_ns']*1000:>7.2f}  {row['total_time_ns']:>7.3f}  "
              f"{a:>10.3e}  {bL:>10.3e}  {bD:>10.3e}  {df:>10.3e}  {s11:>9.4f}  {ratio:>11.4f}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    else:
        _driver()
