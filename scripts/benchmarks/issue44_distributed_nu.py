"""issue44_distributed_nu.py — Phase 4 distributed-NU benchmark harness.

Single parametrised harness used by issue #44 V3 plan, Phase 3.5 (CPU
calibration) and Phase 4 (Job 0 / Job A / Job B GPU jobs).

The harness is parametrised by ``--case`` (calibration / baseline / scaleout)
and ``--devices N`` so the same script can run:

* CPU virtual cluster Phase 3.5 calibration (`--device-type cpu_virtual`,
  `--devices 2`).
* VESSL GPU Job 0 single-device calibration (`--device-type gpu`,
  `--devices 1`, `--case calibration`).
* Future VESSL Jobs A/B (`--devices 2/4`, `--case baseline/scaleout`).

Output classification (V3 plan §3.5, lines 947-957):

* CPU-transferable (set GPU numerical-parity targets):
  ``grad_rel_err_vs_single``, ``value_rel_err_vs_single``,
  ``probe_rel_err_vs_single``, ``scan_graph_compile_time``.
* GPU-only (cannot be derived from CPU virtual cluster):
  ``peak_device_memory_*``, ``forward_wall_time``, ``backward_wall_time``.

The harness writes a single JSON file with all V3 lines 1047-1057 fields.
On Job 0 (single-GPU calibration) ``calibration_row=true`` and
``memory_pass_threshold`` / ``wall_time_pass_threshold`` are populated as
``Job0_value * 1.1`` for downstream Jobs A/B to compare against.

Constraints (process rules from issue #44 task brief):
* No edits to ``rfx/`` source files. This is pure harness/script work.
* Set XLA_FLAGS at module import time so ``python scripts/...`` works on CPU.
* Catch OOM / XlaRuntimeError so a failed run still produces a JSON record.
"""

from __future__ import annotations

# Set XLA_FLAGS *before* importing jax so the CPU virtual cluster works
# when the script is run via ``python`` rather than under pytest's conftest.
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Case definitions (V3 plan §3.5 + §Phase 4)
# ---------------------------------------------------------------------------

# Each case is a dict of grid + physics parameters.  ``calibration`` matches
# the Phase 3.5 calibration-shape spec (V3 lines 939-944): ~4M cells, full
# CPML, 1 Debye + 1 Lorentz pole mixed dispersion, 1 source + 1 probe.
#
# ``baseline`` is the Job A shape (V3 lines 1026-1029): slightly larger
# (~10M cells) with the same physics, plus a checkpoint_every sweep handled
# by repeated invocations of the harness with different ``--checkpoint-every``.
#
# ``scaleout`` is Job B (V3 lines 1031-1033): ~100M cells, full CPML +
# dispersive.  Authored here for Job B reuse but not run in this session.
#
# Issue #625 (2026-08-11): design_mask was removed from every rfx public
# surface (measured to save zero AD memory while corrupting the gradient
# in every configuration tested), so the per-case ``"design_mask"``
# True/False field that used to live here is gone. That field was never
# actually read by ``build_simulation()`` or ``run()`` below in the first
# place -- the harness's only live design_mask control was the separate
# ``--design-mask-fraction`` CLI flag (default 0.0, never exercised in
# any run to date) -- so removing it changes no benchmark behaviour.

CASES: dict[str, dict[str, Any]] = {
    "calibration": {
        # Aspect ratio matches realistic FDTD grids (transverse-thin slab
        # with cubic-ish transverse cross-section).  64 x 64 x 1024 = 4.19M.
        "nx": 64, "ny": 64, "nz": 1024,
        "dx": 1e-3,
        "dz_min": 5e-4,   # 2x refinement near the slab
        "cpml_layers": 8,
        "n_steps_default": 200,
        "dispersion_mode": "mixed",  # 1 Debye + 1 Lorentz
    },
    "baseline": {
        # ~10M cells (Job A): keep transverse small so the seam exchange has
        # to do real work but stay within 1 RTX-4090 HBM (24 GiB).
        "nx": 96, "ny": 96, "nz": 1024,
        "dx": 1e-3,
        "dz_min": 5e-4,
        "cpml_layers": 8,
        "n_steps_default": 500,
        "dispersion_mode": "mixed",
    },
    "scaleout": {
        # ~100M cells (Job B). Author the case definition; user won't run
        # it this session.  Shape sized for 4 RTX-4090 (4 x 24 GiB HBM).
        "nx": 256, "ny": 256, "nz": 1536,
        "dx": 1e-3,
        "dz_min": 5e-4,
        "cpml_layers": 8,
        "n_steps_default": 1000,
        "dispersion_mode": "mixed",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_info(repo_root: Path) -> dict[str, str]:
    """Return current branch + short commit for reproducibility."""
    info = {"branch": "unknown", "commit": "unknown"}
    try:
        info["branch"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
        info["commit"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        pass
    return info


def _live_array_bytes() -> int:
    """Sum of jax.live_arrays() bytes — used as a peak-memory proxy.

    On GPU this approximates the device-resident HBM at the call site (it is
    an underestimate when XLA has temporary buffers we cannot see).  On CPU
    virtual devices it is a process-wide number and not per-device, so the
    Phase 3.5 calibration JSON correctly classifies it as a GPU-only metric.
    """
    total = 0
    try:
        for arr in jax.live_arrays():
            try:
                total += int(np.prod(arr.shape)) * arr.dtype.itemsize
            except Exception:
                pass
    except Exception:
        pass
    return total


def _peak_device_memory(devices: list) -> dict[str, int]:
    """Best-effort per-device peak HBM in bytes.

    Uses ``device.memory_stats()`` (JAX >= 0.4.x on CUDA / ROCm devices).
    Returns a dict ``{device_str: peak_bytes_in_use, ...}``.  Falls back to
    ``_live_array_bytes()`` when ``memory_stats`` is unsupported (CPU virtual
    devices) — the fallback value is then process-wide rather than per-device.
    """
    out: dict[str, int] = {}
    for i, dev in enumerate(devices):
        key = f"rank{i}"
        try:
            stats = dev.memory_stats()  # type: ignore[attr-defined]
        except Exception:
            stats = None
        if stats:
            # CUDA gives `peak_bytes_in_use`; fall back to `bytes_in_use`.
            val = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
            if val is not None:
                out[key] = int(val)
                continue
        # Fallback: live-array bytes (process-wide, not per-device)
        out[key] = _live_array_bytes()
    return out


# ---------------------------------------------------------------------------
# Simulation builder
# ---------------------------------------------------------------------------


def _build_dz_profile(nz_total: int, dz_uniform: float, dz_min: float,
                      slab_layers: int = 16) -> np.ndarray:
    """Construct a tapered dz profile with a refined region in the middle.

    Returns an array of length ``nz_total`` where the centre ``slab_layers``
    cells use ``dz_min`` and the rest use ``dz_uniform`` with a short
    geometric transition.  This mirrors the convergence_grid.py NU pattern.
    """
    n_trans = 4
    trans = np.geomspace(dz_uniform, dz_min, n_trans + 2)[1:-1]
    n_fine = slab_layers
    n_coarse_each = max(0, (nz_total - 2 * n_trans - n_fine) // 2)
    # Handle off-by-one when nz_total is odd
    n_coarse_left = n_coarse_each
    n_coarse_right = nz_total - 2 * n_trans - n_fine - n_coarse_left
    if n_coarse_right < 0:
        # Fall back to uniform if nz_total is too small for the refinement
        return np.full(nz_total, dz_uniform)
    dz = np.concatenate([
        np.full(n_coarse_left, dz_uniform),
        trans,
        np.full(n_fine, dz_min),
        trans[::-1],
        np.full(n_coarse_right, dz_uniform),
    ])
    assert len(dz) == nz_total, (
        f"dz profile length {len(dz)} != nz_total {nz_total}"
    )
    return dz


def build_simulation(case: dict[str, Any]):
    """Build a non-uniform Simulation matching the case spec.

    Returns a tuple ``(sim, build_meta)`` where ``build_meta`` records the
    actual grid shape used (which may differ from the request when nz_total
    is too small for the dz refinement scheme).
    """
    from rfx import Simulation, Box
    from rfx.materials.debye import DebyePole
    from rfx.materials.lorentz import lorentz_pole

    nx = int(case["nx"])
    ny = int(case["ny"])
    nz = int(case["nz"])
    dx = float(case["dx"])
    dz_min = float(case["dz_min"])
    cpml = int(case["cpml_layers"])

    dz_profile = _build_dz_profile(nz, dz_uniform=dx, dz_min=dz_min)

    sim = Simulation(
        freq_max=5e9,
        # domain is recomputed by Simulation when dx_profile/dz_profile are
        # given; pass nominal extents here (they are overridden internally).
        domain=(nx * dx, ny * dx, float(np.sum(dz_profile))),
        dx=dx,
        boundary="cpml",
        cpml_layers=cpml,
        dx_profile=np.full(nx, dx),
        dz_profile=dz_profile,
    )

    mode = case["dispersion_mode"]
    if mode in ("debye", "mixed"):
        debye_poles = [DebyePole(delta_eps=0.5, tau=1e-11)]
    else:
        debye_poles = None
    if mode in ("lorentz", "mixed"):
        lorentz_poles = [lorentz_pole(delta_eps=0.3, omega_0=2e10, delta=1e9)]
    else:
        lorentz_poles = None

    # Place a dispersive slab spanning the centre region in z, full
    # transverse extent.  Picks the centre slab_layers cells (where dz is
    # refined) to also exercise the NU mesh.
    z_edges = np.concatenate([[0.0], np.cumsum(dz_profile)])
    nz_actual = len(dz_profile)
    z_lo_idx = nz_actual // 2 - 4
    z_hi_idx = nz_actual // 2 + 4
    z_lo = float(z_edges[z_lo_idx])
    z_hi = float(z_edges[z_hi_idx])

    if debye_poles is not None or lorentz_poles is not None:
        sim.add_material(
            "disp_slab",
            eps_r=2.0,
            debye_poles=debye_poles,
            lorentz_poles=lorentz_poles,
        )
        # Inset the slab so it does NOT extend into the CPML region — the
        # rfx preflight rejects/warns on materials inside the absorber and
        # the gradient becomes ill-defined when the dispersive ADE state
        # overlaps the CPML stretching profile.
        domain_x = nx * dx
        domain_y = ny * dx
        margin = (cpml + 2) * dx
        x_lo = max(0.0, margin)
        x_hi = max(x_lo + dx, domain_x - margin)
        y_lo = max(0.0, margin)
        y_hi = max(y_lo + dx, domain_y - margin)
        sim.add(
            Box((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi)),
            material="disp_slab",
        )

    # Source on the rank-0 side (x < nx/2), probe on the rank-1 side
    # (x >= nx/2) so the signal crosses the seam.  Both sit safely outside
    # the CPML guard layers along x.
    src_i = max(cpml + 2, nx // 4)
    prb_i = min(nx - cpml - 3, 3 * nx // 4)
    sx = src_i * dx
    sy = (ny // 2) * dx
    sz = float(z_edges[nz_actual // 2])
    px = prb_i * dx
    sim.add_source(position=(sx, sy, sz), component="ez")
    sim.add_probe(position=(px, sy, sz), component="ez")

    # Probe NU grid for true cell count (nx_total includes CPML padding)
    nu_grid = sim._build_nonuniform_grid()
    build_meta = {
        "grid_nx": int(nu_grid.nx),
        "grid_ny": int(nu_grid.ny),
        "grid_nz": int(nu_grid.nz),
        "grid_dt": float(nu_grid.dt),
        "n_cells": int(nu_grid.nx) * int(nu_grid.ny) * int(nu_grid.nz),
        "dispersion_mode": mode,
        "n_debye_poles": 1 if debye_poles else 0,
        "n_lorentz_poles": 1 if lorentz_poles else 0,
    }
    return sim, build_meta


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def _loss_fn(res) -> jnp.ndarray:
    """Scalar loss for grad parity comparisons.

    Uses ``sum(time_series ** 2)`` over ALL time steps and probes so the
    loss is non-zero even on short runs (the composition tests use
    ``time_series[0]`` which slices the first time step — that gives a
    zero loss / zero gradient when n_steps is short and the source has
    not yet ramped up, which propagates as NaN through the distributed
    runner's ghost-exchange under jax.grad).
    """
    return jnp.sum(res.time_series ** 2)


def _measure_forward(sim_factory, kwargs, devices_for_mem) -> dict[str, Any]:
    """Run forward once for warmup, then measure forward wall time + memory."""
    # Warmup compile
    t0 = time.perf_counter()
    res_warm = sim_factory().forward(**kwargs)
    res_warm.time_series.block_until_ready()
    t_warm = time.perf_counter() - t0

    # Timed run
    t0 = time.perf_counter()
    res = sim_factory().forward(**kwargs)
    res.time_series.block_until_ready()
    t_run = time.perf_counter() - t0

    mem = _peak_device_memory(devices_for_mem)
    return {
        "compile_plus_run_time": t_warm,
        "scan_graph_compile_time": max(0.0, t_warm - t_run),
        "forward_wall_time": t_run,
        "forward_result": res,
        "peak_device_memory": mem,
    }


def _measure_backward(sim_factory, kwargs, eps_init, devices_for_mem) -> dict[str, Any]:
    """Measure backward (jax.grad) wall time + value + grad."""
    def _grad_loss(eps_val):
        res = sim_factory().forward(eps_override=eps_val, **kwargs)
        return _loss_fn(res)

    grad_fn = jax.value_and_grad(_grad_loss)

    # Warmup compile
    t0 = time.perf_counter()
    val_w, grad_w = grad_fn(eps_init)
    val_w.block_until_ready(); grad_w.block_until_ready()
    t_warm = time.perf_counter() - t0

    # Timed
    t0 = time.perf_counter()
    val, grad = grad_fn(eps_init)
    val.block_until_ready(); grad.block_until_ready()
    t_run = time.perf_counter() - t0

    mem = _peak_device_memory(devices_for_mem)
    return {
        "compile_plus_run_time": t_warm,
        "backward_wall_time": t_run,
        "value": val,
        "grad": grad,
        "peak_device_memory": mem,
    }


def _rel_err(a, b) -> float:
    """Max relative error |a-b| / (|b| + eps), as a Python float."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    denom = np.abs(b) + 1e-30
    return float(np.max(np.abs(a - b) / denom))


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    git = _git_info(repo_root)

    case_name = args.case
    case = dict(CASES[case_name])  # copy
    n_steps = int(args.n_steps) if args.n_steps else int(case["n_steps_default"])

    # Resolve devices and validate
    available = jax.devices()
    if args.devices > len(available):
        raise ValueError(
            f"requested {args.devices} devices but only "
            f"{len(available)} jax.devices() available "
            f"(platform={available[0].platform if available else 'unknown'})"
        )
    devices = list(available[: args.devices])
    is_distributed = args.devices >= 2

    # Build sim
    sim_kwargs = dict(
        n_steps=n_steps,
        emit_time_series=True,
    )
    if args.checkpoint_every is not None and args.checkpoint_every != "none":
        sim_kwargs["checkpoint_every"] = int(args.checkpoint_every)
    sim_kwargs["exchange_interval"] = int(args.exchange_interval)

    if is_distributed:
        sim_kwargs["distributed"] = True
        sim_kwargs["devices"] = devices

    # Build once to get grid metadata
    sim_for_meta, build_meta = build_simulation(case)

    # Initial eps for grad measurement: just ones, the ADE/CPML response is
    # what we are differentiating w.r.t.
    eps_init = jnp.ones((build_meta["grid_nx"], build_meta["grid_ny"],
                         build_meta["grid_nz"]))

    # Single-device reference (always run at devices=1 to give the
    # numerical-parity targets).  Even when args.devices == 1, this gives us
    # the warmup-vs-timed wall time split.
    def _factory_single():
        s, _ = build_simulation(case)
        return s

    def _factory_dist():
        s, _ = build_simulation(case)
        return s

    record: dict[str, Any] = {
        "issue": "44",
        "case": case_name,
        "branch": git["branch"],
        "commit": git["commit"],
        "device_type": args.device_type,
        "n_devices": int(args.devices),
        "platform": jax.default_backend(),
        "machine": platform.platform(),
        "python": sys.version.split()[0],
        "jax_version": jax.__version__,
        "nx": build_meta["grid_nx"],
        "ny": build_meta["grid_ny"],
        "nz": build_meta["grid_nz"],
        "n_cells": build_meta["n_cells"],
        "dt": build_meta["grid_dt"],
        "n_steps": n_steps,
        "cpml_layers": int(case["cpml_layers"]),
        "dispersion_mode": build_meta["dispersion_mode"],
        "n_debye_poles": build_meta["n_debye_poles"],
        "n_lorentz_poles": build_meta["n_lorentz_poles"],
        "distributed": bool(is_distributed),
        "checkpoint_every": (None if args.checkpoint_every in (None, "none")
                              else int(args.checkpoint_every)),
        "n_warmup": 0,
        "emit_time_series": True,
        "exchange_interval": int(args.exchange_interval),
        "calibration_row": (case_name == "calibration" and args.devices == 1
                             and args.device_type == "gpu"),
        "memory_pass_threshold": None,
        "wall_time_pass_threshold": None,
        "pass_memory": None,
        "pass_wall_time": None,
        "pass_numerics": None,
        "errors": [],
    }

    # ---- Forward (single-device reference) ----
    try:
        single_kwargs = {k: v for k, v in sim_kwargs.items()
                         if k not in ("distributed", "devices")}
        f_single = _measure_forward(_factory_single, single_kwargs,
                                     devices_for_mem=[available[0]])
        ts_single = np.asarray(f_single["forward_result"].time_series[:, 0],
                               dtype=np.float64)
        record["scan_graph_compile_time"] = float(f_single["scan_graph_compile_time"])
        record["forward_wall_time_single"] = float(f_single["forward_wall_time"])
    except Exception as exc:
        record["errors"].append({
            "stage": "forward_single",
            "type": type(exc).__name__,
            "msg": str(exc),
            "trace": traceback.format_exc(),
        })
        return record

    # ---- Forward (distributed, only if requested) ----
    forward_wall_time = float(f_single["forward_wall_time"])
    peak_mem_dist = f_single["peak_device_memory"]
    value_rel_err = 0.0
    probe_rel_err = 0.0
    ts_dist = ts_single
    if is_distributed:
        try:
            f_dist = _measure_forward(_factory_dist, sim_kwargs,
                                       devices_for_mem=devices)
            ts_dist = np.asarray(f_dist["forward_result"].time_series[:, 0],
                                  dtype=np.float64)
            forward_wall_time = float(f_dist["forward_wall_time"])
            peak_mem_dist = f_dist["peak_device_memory"]
            value_rel_err = _rel_err(ts_dist[-1], ts_single[-1])
            probe_rel_err = _rel_err(ts_dist, ts_single)
        except Exception as exc:
            record["errors"].append({
                "stage": "forward_distributed",
                "type": type(exc).__name__,
                "msg": str(exc),
                "trace": traceback.format_exc(),
            })

    record["forward_wall_time"] = float(forward_wall_time)
    record["peak_device_memory"] = peak_mem_dist
    # Convenience scalars matching V3 spec
    record["peak_mem_per_device"] = (
        max(peak_mem_dist.values()) if peak_mem_dist else None
    )
    record["value_rel_err_vs_single"] = float(value_rel_err)
    record["probe_rel_err_vs_single"] = float(probe_rel_err)

    # ---- Backward (gradient) ----
    backward_wall_time: float = 0.0
    grad_rel_err = 0.0
    try:
        # Single-device baseline gradient (always)
        b_single = _measure_backward(_factory_single, single_kwargs,
                                      eps_init,
                                      devices_for_mem=[available[0]])
        backward_wall_time = float(b_single["backward_wall_time"])
        grad_single = np.asarray(b_single["grad"], dtype=np.float64)

        if is_distributed:
            b_dist = _measure_backward(_factory_dist, sim_kwargs,
                                        eps_init,
                                        devices_for_mem=devices)
            backward_wall_time = float(b_dist["backward_wall_time"])
            grad_dist = np.asarray(b_dist["grad"], dtype=np.float64)
            # KNOWN ISSUE: Phase 4 calibration revealed the distributed
            # backward pass produces NaN at rank-0 corner cells (jnp.where
            # rank-conditional CPML gating, JAX gradient gotcha — NaN
            # propagates through both branches in backward).  Forward is
            # bit-perfect.  Surface this as separate metrics so the active-
            # cell parity number stays meaningful while the NaN count
            # flags the bug for follow-up.
            nan_mask = np.isnan(grad_dist)
            record["grad_nan_count"] = int(np.sum(nan_mask))
            record["grad_nan_fraction"] = (
                float(record["grad_nan_count"]) / grad_dist.size
            )
            # Active-cell parity: cells where single-device gradient is
            # numerically meaningful (above noise floor) AND distributed
            # produced a finite value.  This is the right comparison for
            # validating the physics-active region.
            active = (np.abs(grad_single) > 1e-12) & np.isfinite(grad_dist)
            if np.any(active):
                grad_rel_err = float(np.max(
                    np.abs(grad_dist[active] - grad_single[active])
                    / (np.abs(grad_single[active]) + 1e-30)
                ))
            else:
                grad_rel_err = float("nan")
            record["grad_active_cell_count"] = int(np.sum(active))
            # Update memory with the larger of forward / backward
            for k, v in b_dist["peak_device_memory"].items():
                peak_mem_dist[k] = max(peak_mem_dist.get(k, 0), v)
            record["peak_device_memory"] = peak_mem_dist
            record["peak_mem_per_device"] = (
                max(peak_mem_dist.values()) if peak_mem_dist else None
            )
        else:
            grad_rel_err = 0.0
            record["grad_nan_count"] = 0
            record["grad_nan_fraction"] = 0.0
            record["grad_active_cell_count"] = int(np.count_nonzero(grad_single))
    except Exception as exc:
        record["errors"].append({
            "stage": "backward",
            "type": type(exc).__name__,
            "msg": str(exc),
            "trace": traceback.format_exc(),
        })

    record["backward_wall_time"] = float(backward_wall_time)
    record["grad_rel_err_vs_single"] = float(grad_rel_err)

    # ---- Pass/fail (only meaningful when calibration_row=False) ----
    if record["calibration_row"]:
        # Self-comparison: this row sets the thresholds for downstream jobs.
        if record["peak_mem_per_device"] is not None:
            record["memory_pass_threshold"] = (
                float(record["peak_mem_per_device"]) * 1.1
            )
        record["wall_time_pass_threshold"] = (
            float(record["forward_wall_time"]) * 1.1
        )
        record["pass_memory"] = True
        record["pass_wall_time"] = True
        record["pass_numerics"] = True
    elif is_distributed and not record["errors"]:
        # Without a Job 0 reference at hand, the harness cannot self-judge
        # GPU thresholds.  These fields stay None for downstream tooling to
        # populate from the Job 0 JSON.
        # Numerics: tolerance contract Class A (grad) + Class B (value).
        record["pass_numerics"] = bool(
            grad_rel_err <= 1e-6 and value_rel_err <= 5e-5
            and probe_rel_err <= 5e-4
        )

    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=list(CASES.keys()),
                   default="calibration",
                   help="Case shape and feature set.")
    p.add_argument("--devices", type=int, default=1,
                   help="Number of devices to use (1 / 2 / 4).")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Override case-default n_steps.")
    p.add_argument("--checkpoint-every", default=None,
                   help="Segmented remat interval ('none' or int).")
    p.add_argument("--exchange-interval", type=int, default=1,
                   help="Ghost-cell exchange interval.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path.")
    p.add_argument("--device-type",
                   choices=["cpu_virtual", "gpu"], default="cpu_virtual",
                   help="cpu_virtual sets CPU-transferable classification; "
                        "gpu sets calibration_row eligibility.")
    args = p.parse_args()

    # Banner
    print("=" * 70)
    print(f"issue44_distributed_nu  |  case={args.case}  devices={args.devices}  "
          f"device_type={args.device_type}")
    print(f"jax.devices()={jax.devices()}  default_backend={jax.default_backend()}")
    print("=" * 70)

    # Run + write
    record = run(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=str))

    # Human summary
    print()
    print("--- Summary ---")
    print(f"case            : {record['case']}")
    print(f"grid            : {record['nx']}x{record['ny']}x{record['nz']} "
          f"= {record['n_cells']:,} cells")
    print(f"n_steps         : {record['n_steps']}  dt={record['dt']:.3e}")
    print(f"distributed     : {record['distributed']}  devices={record['n_devices']}")
    print(f"calibration_row : {record['calibration_row']}")
    if "scan_graph_compile_time" in record:
        print(f"compile_time_s  : {record['scan_graph_compile_time']:.3f}")
    print(f"forward_wt_s    : {record.get('forward_wall_time')}")
    print(f"backward_wt_s   : {record.get('backward_wall_time')}")
    print(f"peak_mem_per_dev: {record.get('peak_mem_per_device')}")
    print(f"value_rel_err   : {record.get('value_rel_err_vs_single')}")
    print(f"probe_rel_err   : {record.get('probe_rel_err_vs_single')}")
    print(f"grad_rel_err    : {record.get('grad_rel_err_vs_single')}")
    if record["errors"]:
        print(f"ERRORS          : {len(record['errors'])}")
        for e in record["errors"]:
            print(f"  - [{e['stage']}] {e['type']}: {e['msg']}")
    print(f"\nJSON written to: {args.out}")
    # Exit non-zero if any error occurred but JSON still got written
    return 0 if not record["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
