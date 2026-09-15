"""A/B bit-identity harness for the #1038 consolidation of ``rfx/runners/``.

Issue #1038 moves duplicated helpers out of ``distributed.py`` /
``distributed_v2.py`` / ``distributed_nu.py`` into the existing shared
``rfx/runners/_distributed_common.py``, one leg at a time. Every leg is PURE
CODE MOTION, and ``docs/agent-memory/development_methodology.md`` §2.2 gates a
code-motion refactor on BIT IDENTITY -- ``np.array_equal`` on the raw arrays,
never a tolerance -- not on "the physics still looks right".

This module is that gate. It drives each runner end to end on the smallest
fixture that reaches it, snapshots every numeric array of the observable, and
compares the snapshot byte-exactly against a baseline captured on the SAME host
before the move.

Why the baseline is NOT committed
---------------------------------
The same FDTD code is measured to differ by up to 1.3e-3 relative between macOS
and the Linux cluster, and XLA is free to re-fuse a graph between versions. Bit
identity is therefore promised only for A/B on ONE host with ONE toolchain --
which is exactly the question a code-motion refactor asks. A committed baseline
would be a cross-platform pin the code never made, so the arrays live outside
the repo and the baseline tests skip when they are absent.

Why the SIX FINAL FIELD ARRAYS, not just the probe trace
--------------------------------------------------------
This is the hazard the gate exists for. Moving a helper out of a ``run_*``
closure changes what is a free variable and what is an argument, which changes
the jaxpr the tracer builds, which XLA is free to fuse differently -- and float
addition is not associative, so a different fusion can change the last bits.
Three of the helpers the first leg wants (``_shard_stacked``,
``_sample_probes_shmap``, ``_inject_sources_shmap``) are defined INSIDE
``run_distributed`` / ``run_nonuniform_distributed_pec`` and capture the mesh,
the sharding, and -- inside a Python loop unrolled at trace time -- the probe
and source specs, so the traced graph depends on those VALUES and not only on
their shapes. A fusion change of that kind shows up as a last-bit difference in
the field arrays long before it is visible in a probe trace, which is why every
snapshot here carries ``state.{ex,ey,ez,hx,hy,hz,step}`` and not just
``time_series``. If a leg does trip it, the mitigation is to keep the helper
nested and share only the inner ``shard_map``-decorated body -- the shape
``_distributed_common.exchange_component_shmap`` already uses, which takes
``field, mesh, n_devices`` explicitly and is proven bit-safe here by
``test_cpml_axis_params_refactor_bit_identical.py``.

This lock NEVER compares one runner to another
----------------------------------------------
Only each runner to its own baseline. On an identical model the v1 (``pmap``)
and v2 (``shard_map``) distributed runners are NOT bit-identical. Two fixtures
here make that concrete and reproducible: ``distributed_v1_cpml_small`` and
``distributed_v2_cpml`` are the SAME simulation at the same step count, one
through each runner, and their stored baselines differ by
``max|delta| = 2.7940e-09`` on a ``9.4145e-03`` peak (2.97e-07 relative) in
``time_series``, ``3.7253e-09`` on ``1.1388e-01`` in ``state.ez``, and by more
than 100% of the peak in ``state.hz`` (``1.6734e-11`` vs ``1.1815e-11``, a
component that is numerically noise here). They are two implementations of the
same physics, so cross-runner equality is a physics-parity question
(``test_distributed.py`` and the ``test_distributed_nu_*`` family own it, with
tolerances) and not an identity question. Anyone proposing to merge v1 into v2
"without changing the numbers" should read that measurement first.

Use
---
Capture, on the tree BEFORE the move::

    RFX_RUNNER_BASELINE_DIR=/path/to/baseline \\
    RFX_RUNNER_BASELINE_CAPTURE=1 \\
    pytest tests/locks/test_runner_split_bit_identity.py

Compare, on the tree AFTER the move (same host, same interpreter, same
XLA_FLAGS -- ``conftest.py`` sets
``XLA_FLAGS=--xla_force_host_platform_device_count=2`` by ``setdefault`` before
any ``import jax``, and the capture side must not override it)::

    RFX_RUNNER_BASELINE_DIR=/path/to/baseline \\
    pytest tests/locks/test_runner_split_bit_identity.py

Each fixture writes ``<fixture>.npz`` plus a ``<fixture>.sha256`` sidecar over
the canonical byte serialisation of the snapshot, so the two sides can also be
diffed by hand from a PR body.

Three test families here need NO baseline and never skip: the shared-helper
``is``-identity guards, the registry-completeness guard that forces a newly
extracted helper to acquire one, and the public entry-point surface lock.

Builders are REPLICATED here, not imported
------------------------------------------
Each fixture body is copied from the test module named above it, with the
source file and line. The sparams precedent's reason applies verbatim: a
baseline is meaningless if the geometry it was captured on can drift underneath
it, and importing a builder from a test module makes an edit to that module
silently invalidate every stored digest. Truncated step counts are deliberate
-- an identity check needs a deterministic record, not a settled one.

Determinism, measured on this host before the module was committed (2026-09-15,
python 3.11.16 / jax 0.10.2 / 2 virtual CPU devices): all thirteen fixtures
reproduce their full snapshot bit-for-bit back-to-back in ONE process, across
two processes (capture then compare, 43.0 s serial), and across four xdist
worker processes (``-n 4``, 17.3 s). No fixture was dropped. Cold wall time is
38.7 s for the whole set, warm 18.3 s; ``distributed_nu_pec_mask_seam`` is 9.8 s
of that and does not warm up, because each call builds a fresh ``Mesh`` and
retraces its two ``shard_map`` kernels.

Thirteen and not the inventory's twelve: §7.3 files the ``pad_x`` fixture under
"distributed v1", but v1 REFUSES an nx that is not divisible by n_devices
(``rfx/runners/distributed.py:1393``), so that geometry cannot reach it. The
geometry is kept, runs on v2 where the pad lane actually lives, and v1 gets a
second even-nx fixture of its own.

Every snapshot is checked for non-vacuity BEFORE it is stored
(``_assert_snapshot_can_bind``): an all-zero baseline would compare identical
to an all-zero result forever.

Multi-device fixtures skip when fewer than 2 JAX devices are visible. That is
not a silent hole: ``tests/unit/runners/test_device_count_sentinel.py`` FAILS
(does not skip) in that environment unless a lane sets
``RFX_ALLOW_SINGLE_DEVICE=1``.

No ``jax.config.update('jax_enable_x64', True)`` here: it is process-global and
would red every same-process pytest-split shard.
"""

from __future__ import annotations

LOCK_PROVENANCE = {
    "fixture": (
        "tests/locks/test_cpml_axis_params_refactor_bit_identical.py,"
        "tests/unit/nonuniform/test_nu_flux_monitor_finite_size.py,"
        "tests/unit/runners/test_distributed_nu_smoke.py,"
        "tests/unit/runners/test_distributed.py,"
        "tests/unit/runners/test_distributed_nu_kernel.py,"
        "tests/unit/runners/test_distributed_nu_pec_mask_lane_parity.py,"
        "tests/unit/subgrid/test_subgrid_source_injection_dtype.py"
    ),
    "generator": "tests/locks/test_runner_split_bit_identity.py (RFX_RUNNER_BASELINE_CAPTURE=1)",
    "commit": "9866bafd",
    "date": "2026-09-15",
    "run_id": "local",
    "host": "remilab pod linux x86_64, CPU 2 virtual devices, python 3.11.16, jax 0.10.2",
    "pinned_until": "2027-03-15",
}

import ast
import hashlib
import importlib
import os
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse, ModulatedGaussian

_BASELINE_ENV = "RFX_RUNNER_BASELINE_DIR"
_CAPTURE_ENV = "RFX_RUNNER_BASELINE_CAPTURE"

_SKIP_REASON = (
    f"set {_BASELINE_ENV} to run the #1038 A/B bit-identity gate. It compares "
    "raw runner output arrays against a baseline captured on the SAME host "
    "with the SAME interpreter and XLA flags, before the code motion -- bit "
    "identity across CPUs / XLA versions is not promised (the same FDTD code "
    "differs by up to 1.3e-3 relative between macOS and the Linux cluster, "
    "and XLA may re-fuse), so the baseline is deliberately not committed. "
    f"Capture it with {_BASELINE_ENV}=<dir> {_CAPTURE_ENV}=1 pytest "
    "tests/locks/test_runner_split_bit_identity.py on the pre-move tree."
)

_NEEDS_2DEV = (
    "needs >=2 JAX devices (XLA_FLAGS=--xla_force_host_platform_device_count=2). "
    "tests/unit/runners/test_device_count_sentinel.py FAILS in that environment, "
    "so this skip is not a silent hole."
)

_C0 = 2.998e8


# ===========================================================================
# Fixture builders. Replicated verbatim from the modules named per builder.
# ===========================================================================

# --- 1/5/7: tests/locks/test_cpml_axis_params_refactor_bit_identical.py:38
#     (_build_small_sim), used as-is and, for the distributed lanes, with the
#     x extent widened to 10.5 mm so nx is EVEN -- the v1 runner REFUSES an nx
#     that is not divisible by n_devices (rfx/runners/distributed.py:1393).
def _small_cpml_sim(lx=0.01):
    sim = Simulation(
        freq_max=10e9, domain=(lx, 0.01, 0.01), dx=0.5e-3,
        boundary="cpml", cpml_layers=6,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    return sim


def _f_uniform_cpml():
    return _small_cpml_sim().run(n_steps=200)


# --- 2: the same geometry on the OTHER uniform branch -- boundary="pec" with
#     a dielectric Box, so the run reaches the PEC face applier and the
#     material-aware E update instead of the CPML scan body.
def _small_pec_dielectric_sim():
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary="pec", cpml_layers=0,
    )
    sim.add_material("diel", eps_r=4.0)
    sim.add(Box((0.003, 0.003, 0.003), (0.007, 0.007, 0.007)), material="diel")
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    return sim


def _f_uniform_pec_dielectric():
    return _small_pec_dielectric_sim().run(n_steps=200)


# --- 3: tests/unit/nonuniform/test_nu_flux_monitor_finite_size.py:93
#     (_build_open_graded_sim) + the T1 monitor at :126. The graded NU runner
#     WITH a finite-region flux monitor, so the snapshot carries the flux DFT
#     accumulators as well as the fields.
def _open_graded_sim(dz_profile, nx=44, dx=0.5e-3, f0=9e9):
    dz = np.asarray(dz_profile, dtype=float)
    lz = float(np.sum(dz))
    ny = 28
    ly = ny * dx
    lx = nx * dx
    sim = Simulation(freq_max=0.5 * _C0 / dx, domain=(lx, ly, lz), dx=dx,
                     dz_profile=dz, cpml_layers=8, boundary="cpml")
    sim.add_source((lx * 0.30, ly * 0.5, lz * 0.5), "ez",
                   waveform=ModulatedGaussian(f0=f0, bandwidth=0.6,
                                              amplitude=1.0))
    # The source builder's ``add_probe=True`` branch. It is off by default
    # there because those tests read the flux monitor, not a probe -- but
    # without it this fixture's ``time_series`` is a (n_steps, 0) EMPTY slot,
    # which _assert_snapshot_can_bind rejects and which would have left the
    # nonuniform lane's probe path unwitnessed here. Found by that guard.
    sim.add_probe((lx * 0.72, ly * 0.5, lz * 0.5), "ez")
    return sim, (lx, ly, lz)


def _f_nonuniform_graded_flux():
    dz = np.array([0.25e-3] * 8 + [0.5e-3] * 8, dtype=float)
    sim, (lx, ly, lz) = _open_graded_sim(dz)
    freqs = jnp.asarray(np.linspace(5e9, 18e9, 6))
    sim.add_flux_monitor(axis="x", coordinate=lx * 0.6, freqs=freqs,
                         size=(ly * 0.5, lz * 0.5),
                         center=(ly * 0.5, lz * 0.5), name="fin")
    return sim.run(n_steps=200, compute_s_params=False)


# --- 4/8: tests/unit/runners/test_distributed_nu_smoke.py:26 (_make_sim_nu),
#     with the source/probe placement of its
#     test_degenerate_uniform_matches_single_device_nu at :62. Run
#     single-device it is the NU runner; run with devices= it is
#     distributed_v2's ``is_nu`` branch -- the branch no dedicated test file
#     owns, and the one leg 4 deletes clones from.
def _constant_profile_nu_sim():
    nx, ny, nz = 32, 16, 16
    dx = 1e-3
    sim = Simulation(freq_max=3e9, domain=(nx * dx, ny * dx, nz * dx), dx=dx,
                     boundary="pec")
    sim._dx_profile = np.full(nx, dx)
    sim._dy_profile = np.full(ny, dx)
    sim._dz_profile = np.full(nz, dx)
    sim.add_source(position=(4e-3, 8e-3, 8e-3), component="ez")
    sim.add_probe(position=(8e-3, 8e-3, 8e-3), component="ez")
    return sim


def _f_nonuniform_constant_profile():
    return _constant_profile_nu_sim().run(n_steps=100)


# --- 5: v1 (jax.pmap) is reachable from the public API only as v2's
#     n_devices == 1 fast path (rfx/runners/distributed_v2.py:626), so the
#     fixture calls it DIRECTLY at 2 devices.
def _f_distributed_v1_cpml_small():
    from rfx.runners.distributed import run_distributed as _v1
    return _v1(_small_cpml_sim(lx=0.0105), n_steps=100,
               devices=jax.devices()[:2])


# --- 6: tests/unit/runners/test_distributed.py:294
#     (TestDistributedCPML::test_distributed_cpml_matches_single) geometry,
#     again through the direct v1 entry point. Lx=0.13 realizes an EVEN nx,
#     which v1 requires.
def _wide_cpml_sim(lx=0.13):
    sim = Simulation(freq_max=3e9, domain=(lx, 0.04, 0.04), boundary="cpml")
    sim.add_source(position=(0.065, 0.02, 0.02), component="ez",
                   waveform=GaussianPulse(f0=1.5e9, bandwidth=1.5e9))
    sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")
    return sim


def _f_distributed_v1_cpml_wide():
    from rfx.runners.distributed import run_distributed as _v1
    return _v1(_wide_cpml_sim(), n_steps=100, devices=jax.devices()[:2])


# --- 7/9: sim.run(devices=[...]) is what the public API dispatches to
#     distributed_v2 (rfx/api/_execute.py:3692), for uniform AND non-uniform
#     grids. Two step bodies: step_fn_cpml here, step_fn_pec below.
def _f_distributed_v2_cpml():
    return _small_cpml_sim(lx=0.0105).run(n_steps=100,
                                          devices=jax.devices()[:2])


def _small_pec_sim(lx=0.0105):
    sim = Simulation(freq_max=10e9, domain=(lx, 0.01, 0.01), dx=0.5e-3,
                     boundary="pec", cpml_layers=0)
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    return sim


def _f_distributed_v2_pec():
    return _small_pec_sim().run(n_steps=100, devices=jax.devices()[:2])


def _f_distributed_v2_nu_branch():
    return _constant_profile_nu_sim().run(n_steps=40,
                                          devices=jax.devices()[:2])


# --- 10: tests/unit/runners/test_distributed.py:464
#     (test_distributed_cpml_pad_x_matches_single) geometry. Lx=0.125 realizes
#     an ODD nx=59, so pad_x = 1 at 2 devices.
#
#     The inventory (§7.3 row 6) files this fixture under "distributed v1".
#     It cannot be: v1 REFUSES an nx that is not divisible by n_devices
#     (rfx/runners/distributed.py:1393-1397, ValueError), and the source
#     test's own docstring says the pad lane runs through "distributed_v2's
#     shmap wrappers, which is what production run(devices=...) dispatches
#     to". The geometry is kept and the fixture runs on the runner that
#     actually reaches it, which is why this module carries thirteen fixtures
#     and not twelve.
#
#     skip_preflight mirrors the source test: the probe sits inside the x-hi
#     CPML zone on purpose, because that is where an absorber displacement is
#     visible. Preflight would otherwise advise moving it inward. This is a
#     boundary-MECHANISM fixture, not a claims-bearing measurement.
def _padx_sim():
    lx = 0.125
    sim = Simulation(freq_max=3e9, domain=(lx, 0.04, 0.04), boundary="cpml")
    sim.add_source(position=(lx / 2, 0.02, 0.02), component="ez",
                   waveform=GaussianPulse(f0=1.5e9, bandwidth=1.5e9))
    sim.add_probe(position=(lx - 0.006, 0.02, 0.02), component="ez")
    return sim


def _f_distributed_v2_pad_x():
    return _padx_sim().run(n_steps=100, devices=jax.devices()[:2],
                           skip_preflight=True)


# --- 11: tests/unit/runners/test_distributed_nu_kernel.py:594
#     (test_distributed_pec_only_2device_matches_single_device) -- the DIRECT
#     run_nonuniform_distributed_pec call. distributed_nu is reachable from
#     the public API only through forward(distributed=True); this is the
#     cheapest route that walks its whole step body. Helpers replicated from
#     :43 (_graded_profile), :531 (_phase2b_build_test_grid), :543
#     (_phase2b_make_materials), :553 (_phase2b_gauss_waveform) and :557
#     (_phase2b_shard_mat).
def _graded_profile(n_physical, dx0, ratio=1.5):
    prof = dx0 * ratio ** np.linspace(0, 1, n_physical)
    prof[0] = dx0
    prof[-1] = dx0
    return prof


def _nu_test_grid(nx_physical=16, ny=8, nz=8, dx0=1e-3, ratio=1.2):
    from rfx.nonuniform import make_nonuniform_grid
    return make_nonuniform_grid(
        (nx_physical * dx0, ny * dx0), np.full(nz, dx0), dx0,
        cpml_layers=0, dx_profile=_graded_profile(nx_physical, dx0, ratio),
    )


def _nu_vacuum_materials(grid):
    from rfx.core.yee import MaterialArrays
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    return MaterialArrays(
        eps_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
    )


def _nu_gauss(t, t0=8e-12, tau=2.5e-12):
    return jnp.exp(-((t - t0) ** 2) / (2.0 * tau ** 2))


def _nu_shard_materials(materials, sharded_grid):
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from rfx.core.yee import MaterialArrays
    from rfx.runners.distributed import _split_materials

    n_devices = sharded_grid.n_devices
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x
    mesh = Mesh(np.array(jax.devices()[:n_devices]), axis_names=("x",))
    shd = NamedSharding(mesh, P("x"))

    if pad_x > 0:
        pad = ((0, pad_x), (0, 0), (0, 0))
        materials = MaterialArrays(
            eps_r=jnp.pad(materials.eps_r, pad, constant_values=1.0),
            sigma=jnp.pad(materials.sigma, pad, constant_values=0.0),
            mu_r=jnp.pad(materials.mu_r, pad, constant_values=1.0),
        )

    mat_slabs = _split_materials(materials, n_devices, ghost)

    def _shard_stacked(arr):
        n_dev = arr.shape[0]
        rest = arr.shape[1:]
        return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)

    return MaterialArrays(
        eps_r=_shard_stacked(mat_slabs.eps_r),
        sigma=_shard_stacked(mat_slabs.sigma),
        mu_r=_shard_stacked(mat_slabs.mu_r),
    )


def _f_distributed_nu_direct_pec():
    from rfx.nonuniform import make_current_source
    from rfx.runners.distributed_nu import (
        build_sharded_nu_grid,
        run_nonuniform_distributed_pec,
    )
    from rfx.simulation import ProbeSpec, SourceSpec

    n_devices = 2
    n_steps = 60
    grid = _nu_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.2)
    materials = _nu_vacuum_materials(grid)
    si, sj, sk, comp, wf = make_current_source(
        grid, (4, 4, 4), "ez", _nu_gauss, n_steps, materials)
    src = SourceSpec(i=int(si), j=int(sj), k=int(sk), component=comp,
                     waveform=jnp.asarray(wf))
    prb = ProbeSpec(i=12, j=4, k=4, component="ez")
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)
    return run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=_nu_shard_materials(materials, sharded_grid),
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src],
        probes=[prb],
        n_devices=n_devices,
        devices=jax.devices()[:2],
    )


# --- 12: tests/unit/runners/test_distributed_nu_pec_mask_lane_parity.py:108
#     (_distributed_nnz) and :207 (_distributed_occ_nnz), at 2 ranks with a
#     body that straddles the shard seam. These two shmap kernels -- a hard
#     PEC cell mask and a soft occupancy field -- are the two features no
#     sibling runner has, so nothing else in this module would notice them
#     moving. The source tests compare zero COUNTS; the snapshot here keeps
#     the full output field arrays, which is strictly stronger.
_SEAM_NX, _SEAM_NY, _SEAM_NZ = 8, 6, 6
_SEAM_GHOST = 1


def _seam_plates(axis, idx):
    m = np.zeros((_SEAM_NX, _SEAM_NY, _SEAM_NZ), bool)
    sl = [slice(1, 5)] * 3
    for i in idx:
        sl[axis] = i
        m[tuple(sl)] = True
    return m


def _seam_slabs(gmask, n_devices, dtype, zero):
    nx_per = _SEAM_NX // n_devices
    nx_local = nx_per + 2 * _SEAM_GHOST
    slabs = np.zeros((n_devices * nx_local, _SEAM_NY, _SEAM_NZ), dtype)
    for d in range(n_devices):
        lo, hi = d * nx_per, (d + 1) * nx_per
        base = d * nx_local
        slabs[base + _SEAM_GHOST:base + _SEAM_GHOST + nx_per] = gmask[lo:hi]
        slabs[base] = gmask[lo - 1] if d > 0 else zero
        slabs[base + nx_local - 1] = gmask[hi] if d < n_devices - 1 else zero
    return slabs, nx_local


def _f_distributed_nu_pec_mask_seam():
    from jax.sharding import Mesh

    from rfx.core.yee import init_state
    from rfx.runners.distributed_nu import (
        _apply_pec_mask_nu_shmap,
        _apply_pec_occupancy_nu_shmap,
    )

    n_devices = 2
    xbar = np.zeros((_SEAM_NX, _SEAM_NY, _SEAM_NZ), bool)
    xbar[:, 2:4, 2:4] = True                 # spans x ACROSS the seam
    gmask = _seam_plates(2, [0, _SEAM_NZ - 1]) | xbar
    hard_slabs, nx_local = _seam_slabs(gmask, n_devices, bool, False)
    soft_slabs, _ = _seam_slabs(gmask.astype(np.float32), n_devices,
                                np.float32, 0.0)
    mesh = Mesh(np.asarray(jax.devices()[:n_devices]).reshape(n_devices),
                ("x",))
    shape = (n_devices * nx_local, _SEAM_NY, _SEAM_NZ)
    state = init_state(shape)
    one = jnp.ones(shape, jnp.float32)
    live = state._replace(ex=one, ey=one, ez=one)
    return {
        "hard": _apply_pec_mask_nu_shmap(live, jnp.asarray(hard_slabs), mesh,
                                         n_devices, nx_local),
        "soft": _apply_pec_occupancy_nu_shmap(live, jnp.asarray(soft_slabs),
                                              mesh, n_devices, nx_local),
    }


# --- 13: tests/unit/subgrid/test_subgrid_source_injection_dtype.py:123
#     (_coarse_shadow_source_subgrid_sim). Chosen over the inventory's
#     suggestion (test_subgrid_fine_shape_parity.py's builders) because those
#     build a validation REGION and never call the runner at all -- its
#     centered slab is refused by the production z-slab validator. This one
#     runs run_subgridded_path end to end, through both the coarse-shadow
#     source injection and the SBP-SAT coupling. n_steps=30 is load-bearing
#     in the source file: fewer steps leave the probe provably all zero.
def _subgrid_coarse_shadow_sim():
    sim = Simulation(freq_max=4e9, domain=(0.012, 0.012, 0.012),
                     boundary="pec", dx=0.004)
    sim.add_refinement(z_range=(0.004, 0.012), ratio=3, validation="research")
    sim._refinement["inject_sources_on_coarse_shadow"] = True
    sim.add_source((0.004, 0.004, 0.008), "ez")
    sim.add_probe((0.008, 0.008, 0.008), "ez")
    return sim


def _f_subgridded():
    return _subgrid_coarse_shadow_sim().run(n_steps=30, compute_s_params=False)


# ===========================================================================
# (fixture id, builder, runner it exercises, multi-device?)
# ===========================================================================
_FIXTURES = (
    ("uniform_cpml", _f_uniform_cpml, "uniform", False),
    ("uniform_pec_dielectric", _f_uniform_pec_dielectric, "uniform", False),
    ("nonuniform_graded_flux", _f_nonuniform_graded_flux, "nonuniform", False),
    ("nonuniform_constant_profile", _f_nonuniform_constant_profile,
     "nonuniform", False),
    ("distributed_v1_cpml_small", _f_distributed_v1_cpml_small,
     "distributed", True),
    ("distributed_v1_cpml_wide", _f_distributed_v1_cpml_wide,
     "distributed", True),
    ("distributed_v2_cpml", _f_distributed_v2_cpml, "distributed_v2", True),
    ("distributed_v2_pec", _f_distributed_v2_pec, "distributed_v2", True),
    ("distributed_v2_nu_branch", _f_distributed_v2_nu_branch,
     "distributed_v2", True),
    ("distributed_v2_pad_x", _f_distributed_v2_pad_x, "distributed_v2", True),
    ("distributed_nu_direct_pec", _f_distributed_nu_direct_pec,
     "distributed_nu", True),
    ("distributed_nu_pec_mask_seam", _f_distributed_nu_pec_mask_seam,
     "distributed_nu", True),
    ("subgridded_coarse_shadow", _f_subgridded, "subgridded", False),
)

_NONE_KEY = "__none_fields__"
_STATE_FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz", "step")


def _flatten(obj, prefix, out, missing):
    """Every leaf of ``obj`` as a raw numpy array, keyed by dotted path.

    Total by construction: an unhandled type RAISES rather than being
    skipped, so a result field cannot escape the witness by being a type
    this function forgot. ``None`` leaves are recorded by NAME, so a leg
    that starts (or stops) returning an optional accumulator is a field-set
    change and fails loudly instead of silently shrinking the snapshot.

    ``np.asarray`` on a jax array is a device-to-host copy, not a cast: the
    dtype and every bit of the mantissa survive, which is what the gate reads.
    """
    key = prefix.rstrip(".")
    if obj is None:
        missing.append(key)
        return
    if isinstance(obj, str):
        out[key] = np.asarray(obj, dtype="<U64")
        return
    if isinstance(obj, dict):
        for name in sorted(obj):
            _flatten(obj[name], f"{prefix}{name}.", out, missing)
        return
    if hasattr(obj, "_fields"):                       # NamedTuple
        for name in obj._fields:
            _flatten(getattr(obj, name), f"{prefix}{name}.", out, missing)
        return
    if isinstance(obj, (bool, int, float, complex, np.generic)):
        out[key] = np.asarray(obj)
        return
    if isinstance(obj, (np.ndarray, jax.Array)):
        out[key] = np.asarray(obj)
        return
    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
        if arr.dtype == object:
            raise TypeError(
                f"{key}: ragged/object sequence in the runner observable; "
                "this lock cannot serialise it byte-exactly")
        out[key] = arr
        return
    raise TypeError(
        f"{key}: unhandled type {type(obj).__name__} in the runner observable. "
        "Teach _flatten about it -- do NOT let it be skipped, or the gate goes "
        "vacuous for that field.")


def _observable(res) -> dict[str, np.ndarray]:
    """The numeric surface of one runner call.

    For the six runners that return a ``Result``: the probe ``time_series``,
    all six final ``state`` field arrays plus ``step``, and the flux-monitor /
    DFT-plane accumulators where the lane carries any. For
    ``distributed_nu.run_nonuniform_distributed_pec`` (which returns a dict)
    and for the raw shmap kernels: every entry of what came back.
    """
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    if isinstance(res, dict):
        _flatten(res, "", out, missing)
    else:
        for name in ("time_series", "state", "flux_monitors", "dft_planes"):
            assert hasattr(res, name), (
                f"{type(res).__name__} has no field {name!r}; this lock's "
                "observable has drifted from the Result contract")
        _flatten(res.time_series, "time_series.", out, missing)
        state = res.state
        for field in _STATE_FIELDS:
            assert hasattr(state, field), (
                f"final state has no field {field!r}; the six-field witness "
                "that catches an XLA re-fusion is incomplete")
        _flatten(state, "state.", out, missing)
        _flatten(res.flux_monitors, "flux_monitors.", out, missing)
        _flatten(res.dft_planes, "dft_planes.", out, missing)
    out[_NONE_KEY] = np.asarray(sorted(missing), dtype="<U64")
    return out


def _digest(snapshot: dict[str, np.ndarray]) -> str:
    """SHA256 over name + dtype + shape + raw bytes, in sorted key order."""
    h = hashlib.sha256()
    for key in sorted(snapshot):
        arr = np.ascontiguousarray(snapshot[key])
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _baseline_dir() -> Path | None:
    raw = os.environ.get(_BASELINE_ENV, "").strip()
    return Path(raw) if raw else None


def _capturing() -> bool:
    return os.environ.get(_CAPTURE_ENV, "").strip() not in ("", "0", "false")


def _assert_snapshot_can_bind(fixture: str, snapshot: dict[str, np.ndarray]):
    """The gate-can-bind-artifact rule, applied before anything is stored.

    An all-zero record compares bit-identical to another all-zero record, so
    a baseline captured from a silently-dead fixture would make this gate
    blind for the rest of the campaign. This runs on the CAPTURE side too,
    which is where that damage would be done.

    Two checks: some E-field array carries a non-zero value, and a probe
    trace, where the observable has one, is not identically zero. H is
    deliberately exempt -- the ``distributed_nu_pec_mask_seam`` fixture
    drives two PEC appliers over a state whose H is zero by construction,
    which is correct and not a dead instrument.
    """
    e_keys = [k for k in snapshot
              if k.rsplit(".", 1)[-1] in ("ex", "ey", "ez")]
    assert e_keys, f"{fixture}: snapshot carries no E-field array at all"
    assert any(np.any(np.asarray(snapshot[k]) != 0) for k in e_keys), (
        f"{fixture}: every E-field array in the snapshot is identically zero "
        f"({e_keys}). The fixture is not exciting the runner, so a bit-identity "
        "comparison against it would pass vacuously.")
    if "time_series" in snapshot:
        ts = np.asarray(snapshot["time_series"])
        assert ts.size > 0 and np.any(ts != 0), (
            f"{fixture}: time_series is empty or identically zero -- the probe "
            "records nothing, so it cannot witness a code-motion change.")


def _first_mismatch(a: np.ndarray, b: np.ndarray) -> str:
    flat_a, flat_b = np.ravel(a), np.ravel(b)
    for i in range(flat_a.size):
        x, y = flat_a[i], flat_b[i]
        if x != y and not (x != x and y != y):     # NaN == NaN for this report
            return f"index {np.unravel_index(i, a.shape)}: {x!r} != {y!r}"
    return "(none found - arrays differ only in a way this scan cannot see)"


@pytest.mark.parametrize(
    "fixture,build,runner,multi", _FIXTURES,
    ids=[f for f, _, _, _ in _FIXTURES],
)
def test_runner_is_bit_identical_to_the_pre_split_baseline(
        fixture, build, runner, multi):
    base = _baseline_dir()
    if base is None:
        pytest.skip(_SKIP_REASON)
    if multi and jax.device_count() < 2:
        pytest.skip(f"{fixture} ({runner}) {_NEEDS_2DEV}")

    with warnings.catch_warnings():
        # These records are deliberately truncated and several fixtures sit
        # on advisory geometry on purpose (see each builder). The warnings
        # are not what this gate measures; the advisory behaviour itself is
        # pinned in tests/unit/runners and tests/unit/preflight.
        warnings.simplefilter("ignore")
        result = build()
    got = _observable(result)
    _assert_snapshot_can_bind(fixture, got)
    got_sha = _digest(got)

    npz = base / f"{fixture}.npz"
    sidecar = base / f"{fixture}.sha256"

    if _capturing():
        base.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **got)
        sidecar.write_text(got_sha + "\n", encoding="utf-8")
        pytest.skip(f"captured baseline for {fixture}: {npz} sha256={got_sha}")

    assert npz.exists(), (
        f"no baseline for fixture {fixture!r} at {npz}. Capture it on the "
        f"pre-move tree with {_CAPTURE_ENV}=1.")
    with np.load(npz, allow_pickle=False) as data:
        ref = {k: data[k] for k in data.files}

    assert sorted(ref) == sorted(got), (
        f"{fixture}: observable field set changed -- baseline has "
        f"{sorted(ref)}, this tree produced {sorted(got)}. A code-motion "
        "refactor may not add or drop an output array.")
    for key in sorted(got):
        a, b = got[key], ref[key]
        assert a.dtype == b.dtype, f"{fixture}.{key}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{fixture}.{key}: shape {a.shape} != {b.shape}"
        # np.array_equal, no tolerance: methodology §2.2 gates code motion on
        # bit identity. equal_nan=True so a legitimate NaN slot compares equal
        # to the same NaN slot in the baseline.
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(
                a.dtype, np.complexfloating):
            same = np.array_equal(a, b, equal_nan=True)
        else:
            same = np.array_equal(a, b)
        assert same, (
            f"{fixture}.{key} is NOT bit-identical to the pre-move baseline "
            f"(runner: {runner}).\n"
            f"  baseline sha256 : "
            f"{sidecar.read_text().strip() if sidecar.exists() else '(none)'}\n"
            f"  this tree sha256: {got_sha}\n"
            f"  first mismatch  : {_first_mismatch(a, b)}\n"
            "  If this is a helper lifted out of a run_* closure, suspect an "
            "XLA re-fusion before suspecting the algebra: keep the helper "
            "nested and share only the inner shard_map body.")

    if sidecar.exists():
        assert sidecar.read_text().strip() == got_sha, (
            f"{fixture}: every array compared equal but the snapshot digest "
            f"moved ({sidecar.read_text().strip()} -> {got_sha}); the "
            "serialisation itself changed")


# ===========================================================================
# Shared-helper identity guards. No baseline; these run always.
# ===========================================================================
#
# Extends the pattern of
# tests/locks/test_cpml_axis_params_refactor_bit_identical.py:170-194 into a
# registry, so each #1038 leg adds its extracted helpers here as rows rather
# than as new hand-written tests. The point is mechanical: after an
# extraction, a sibling runner CANNOT drift, because both names resolve to
# the one function object.
#
# (name in _distributed_common, importing module, attribute it is bound to)
_SHARED_HELPER_BINDINGS = (
    ("cpml_coeff_e_vacuum", "rfx.runners.distributed", "cpml_coeff_e_vacuum"),
    ("cpml_coeff_h_vacuum", "rfx.runners.distributed", "cpml_coeff_h_vacuum"),
    ("cpml_coeff_e_vacuum", "rfx.runners.distributed_nu", "cpml_coeff_e_vacuum"),
    ("cpml_coeff_h_vacuum", "rfx.runners.distributed_nu", "cpml_coeff_h_vacuum"),
    ("exchange_component_shmap", "rfx.runners.distributed_v2",
     "_exchange_component_shmap"),
    ("exchange_component_shmap", "rfx.runners.distributed_nu",
     "_exchange_component_nu_shmap"),
    # #1038 leg 1 -- the x-slab sharding helpers. These were NESTED closures
    # over ``shd`` at every site, so the shared form takes ``shd`` explicitly
    # and each site keeps a same-named forwarder; the module-level import is
    # what this guard pins.
    ("shard_stacked", "rfx.runners.distributed_nu", "shard_stacked"),
    ("shard_stacked", "rfx.runners.distributed_v2", "shard_stacked"),
    ("shard_stacked_poles", "rfx.runners.distributed_nu",
     "shard_stacked_poles"),
    ("shard_stacked_psi", "rfx.runners.distributed_nu", "shard_stacked_psi"),
    ("shard_stacked_psi", "rfx.runners.distributed_v2", "shard_stacked_psi"),
    # #1038 leg 1 -- the two shard_map helpers called from inside the jitted
    # step body. These are the leg's real jaxpr-shape change: four locals
    # (mesh, n_{src,prb}, *_local_specs, *_device_ids) became parameters, and
    # the *_specs/*_ids Python loops are unrolled at trace time, so the traced
    # graph depends on their values. The 13 baseline rows above are what says
    # the arrays did not move with the jaxpr.
    ("inject_sources_shmap", "rfx.runners.distributed_nu",
     "inject_sources_shmap"),
    ("inject_sources_shmap", "rfx.runners.distributed_v2",
     "inject_sources_shmap"),
    ("sample_probes_shmap", "rfx.runners.distributed_nu",
     "sample_probes_shmap"),
    ("sample_probes_shmap", "rfx.runners.distributed_v2",
     "sample_probes_shmap"),
    # #1038 leg 2 (a) -- the stacked-CPML-psi allocator. Two nested ``_zeros``
    # closures over (n_devices, n) that differed only in how they spelled the
    # two face-parallel extents (inventory §2.3(b)). Both callers are
    # setup-time, so this row guards a de-duplication with no jaxpr exposure.
    ("zeros_psi_stacked", "rfx.runners.distributed", "zeros_psi_stacked"),
    ("zeros_psi_stacked", "rfx.runners.distributed_nu", "zeros_psi_stacked"),
    # #1038 leg 2 -- the x-slab primitives. Not a de-duplication: these were
    # single definitions in distributed.py that had to move BELOW the shared
    # module so that shared bodies calling them (leg 2b's unstack_and_gather,
    # leg 2c's split_poles_x) can import them without a cycle. distributed.py
    # re-exports both at their old position, so these rows are what says the
    # re-export is the same object and not a resurrected copy.
    ("split_array_x", "rfx.runners.distributed", "split_array_x"),
    ("gather_array_x", "rfx.runners.distributed", "gather_array_x"),
    ("gather_array_x", "rfx.runners.distributed_v2", "gather_array_x"),
    # #1038 leg 2 (b) -- the final-state gather. The two copies differed in the
    # trim bound (``sharded_grid.nx`` in the NU runner, the enclosing ``nx`` in
    # v2) and in the assert message; the bound is now an explicit argument each
    # caller supplies, so neither call site's expression moved.
    ("unstack_and_gather", "rfx.runners.distributed_nu", "unstack_and_gather"),
    ("unstack_and_gather", "rfx.runners.distributed_v2", "unstack_and_gather"),
    # #1038 leg 2 (c) -- the per-pole x-splitter. Three nested copies, all in
    # distributed.py, which leg 1 had to leave behind because their body calls
    # split_array_x and that name was then defined BELOW distributed.py's
    # import of this module. One importer only, so one row.
    ("split_poles_x", "rfx.runners.distributed", "split_poles_x"),
    # #1038 leg 3 -- the domain-face PEC kernel (inventory §2.3(c)). Unlike the
    # leg-1/leg-2 helpers these were already top-level functions taking every
    # value explicitly, so nothing became a parameter and the traced jaxpr is
    # structurally unchanged. The two copies differed in ONE token, each
    # runner's own spelling of the per-rank slab length; the NU side was
    # renamed to nx_local_with_ghost first, which made the normalised-AST
    # hashes equal. Two importers, two module-local aliases, two rows.
    ("apply_pec_face_shmap", "rfx.runners.distributed_v2",
     "_apply_pec_shmap"),
    ("apply_pec_face_shmap", "rfx.runners.distributed_nu",
     "_apply_pec_face_nu_shmap"),
    # #1038 leg 3 -- the domain-face PMC kernel (inventory §2.3(d)). Same story
    # as the PEC row above and the same single token of difference; after the
    # rename the two inner bodies were byte-identical, not merely
    # AST-equivalent. NOTE the kernel is shared but the HOOK POINT is not: v2
    # calls it after the H ghost exchange, the NU runner before it. That
    # divergence is leg 5 / inventory §3.2 and these rows do not speak to it.
    ("apply_pmc_face_shmap", "rfx.runners.distributed_v2",
     "_apply_pmc_shmap"),
    ("apply_pmc_face_shmap", "rfx.runners.distributed_nu",
     "_apply_pmc_face_nu_shmap"),
)


@pytest.mark.parametrize(
    "shared,module,attr", _SHARED_HELPER_BINDINGS,
    ids=[f"{m.rsplit('.', 1)[-1]}.{a}" for _, m, a in _SHARED_HELPER_BINDINGS],
)
def test_shared_runner_helper_is_the_same_object_in_every_importer(
        shared, module, attr):
    from rfx.runners import _distributed_common as common

    importer = importlib.import_module(module)
    assert hasattr(importer, attr), (
        f"{module} no longer binds {attr!r}. A #1038 leg either renamed it or "
        "made it a function-local import; either way the de-duplication is no "
        "longer mechanically enforced. Update this registry deliberately.")
    assert getattr(importer, attr) is getattr(common, shared), (
        f"{module}.{attr} is NOT _distributed_common.{shared} -- a sibling "
        "copy has been re-introduced, which is the exact drift #1038 exists "
        "to remove.")


def test_every_shared_helper_has_an_identity_guard():
    """A helper a leg extracts must acquire a guard in the SAME change.

    Without this, leg N can add a name to ``_distributed_common.__all__``,
    leave one runner still carrying its own private copy, and nothing here
    would notice -- the de-duplication would be cosmetic.
    """
    from rfx.runners import _distributed_common as common

    guarded = {shared for shared, _, _ in _SHARED_HELPER_BINDINGS}
    exported = set(common.__all__)
    assert exported - guarded == set(), (
        "_distributed_common exports helpers with no is-identity guard: "
        f"{sorted(exported - guarded)}. Add a row to "
        "_SHARED_HELPER_BINDINGS for every importer of each one.")
    assert guarded - exported == set(), (
        "this registry guards names _distributed_common no longer exports: "
        f"{sorted(guarded - exported)}")


# ===========================================================================
# Public entry-point surface lock. No baseline; these run always.
# ===========================================================================
#
# §1.10 of the inventory: every dispatch out of rfx/api/_execute.py is a
# function-local import of one of these names, and 104 import statements
# across 55 files resolve a rfx.runners.* name. A consolidation leg may move
# a BODY; it may not silently drop, rename, or relocate an entry point.
#
# (module, name, defining module, qualname). The defining module is pinned on
# purpose: a re-export keeps __module__ at the definition site, so this table
# also catches an entry point whose body moved into a shared module and came
# back as an alias. That matters beyond tidiness --
# tests/unit/materials/test_sheet_impedance.py keys its FENCE_REGISTRY on
# (path, ENCLOSING FUNCTION, message) and its _fence() asserts the raising
# TRACEBACK FRAME, so seven #677 refusals must keep being raised from inside
# run_distributed / run_uniform / run_nonuniform_path in their current files.
_RUNNER_ENTRY_POINTS = (
    ("uniform", "run_uniform", "rfx.runners.uniform", "run_uniform"),
    ("nonuniform", "run_nonuniform", "rfx.nonuniform", "run_nonuniform"),
    ("nonuniform", "run_nonuniform_path", "rfx.runners.nonuniform",
     "run_nonuniform_path"),
    ("nonuniform", "run_nonuniform_until_decay", "rfx.nonuniform",
     "run_nonuniform_until_decay"),
    ("subgridded", "run_subgridded_path", "rfx.runners.subgridded",
     "run_subgridded_path"),
    ("disjoint", "run_disjoint_stage2_path", "rfx.runners.disjoint",
     "run_disjoint_stage2_path"),
    ("distributed", "run_distributed", "rfx.runners.distributed",
     "run_distributed"),
    ("distributed_v2", "run_distributed", "rfx.runners.distributed_v2",
     "run_distributed"),
    ("distributed_nu", "run_nonuniform_distributed_pec",
     "rfx.runners.distributed_nu", "run_nonuniform_distributed_pec"),
)

#: rfx/runners/__init__.py's __all__ at 9866bafd.
_RUNNERS_ALL = ("run_uniform", "run_nonuniform_path", "run_subgridded_path",
                "run_distributed")

#: The module-level ``from rfx.runners.X import Y`` statements in
#: rfx/runners/__init__.py at 9866bafd, read with ast. §5e: __init__.py
#: eagerly imports four submodules at package import, and anything a new
#: shared module imports at module level joins that eager chain --
#: conftest.py:405 carries a repair routine for what happens when a test
#: evicts rfx.runners from sys.modules and that chain is not what it was.
_RUNNERS_EAGER_IMPORTS = (
    ("rfx.runners.uniform", "run_uniform"),
    ("rfx.runners.nonuniform", "run_nonuniform_path"),
    ("rfx.runners.subgridded", "run_subgridded_path"),
    ("rfx.runners.distributed", "run_distributed"),
)


@pytest.mark.parametrize(
    "module,name,defined_in,qualname", _RUNNER_ENTRY_POINTS,
    ids=[f"{m}.{n}" for m, n, _, _ in _RUNNER_ENTRY_POINTS],
)
def test_runner_module_still_exports_its_entry_point(
        module, name, defined_in, qualname):
    mod = importlib.import_module(f"rfx.runners.{module}")
    fn = getattr(mod, name, None)
    assert fn is not None, (
        f"rfx.runners.{module} no longer exports {name!r}. A #1038 leg may "
        "move a body; it may not drop or rename an entry point.")
    assert callable(fn), f"rfx.runners.{module}.{name} is not callable"
    assert fn.__module__ == defined_in, (
        f"rfx.runners.{module}.{name} is now DEFINED in {fn.__module__} "
        f"(was {defined_in}). If the move is deliberate, edit this table and "
        "check tests/unit/materials/test_sheet_impedance.py's FENCE_REGISTRY "
        "in the same commit -- its _fence() asserts the raising traceback "
        "frame, which a re-export cannot satisfy.")
    assert fn.__qualname__ == qualname, (
        f"rfx.runners.{module}.{name}.__qualname__ is {fn.__qualname__!r}, "
        f"expected {qualname!r}")


def test_runners_package_all_is_unchanged():
    import rfx.runners as pkg

    assert tuple(pkg.__all__) == _RUNNERS_ALL, (
        f"rfx.runners.__all__ is {tuple(pkg.__all__)}, expected "
        f"{_RUNNERS_ALL}. Inventory §8 leg 6 PROPOSES dropping "
        "'run_distributed' from it (retiring v1 as a public entry point); "
        "that is a deliberate surface change and must edit this pin in the "
        "same commit, not arrive as a side effect of code motion.")


def test_the_run_distributed_name_collision_still_resolves_to_v1():
    """``rfx.runners.run_distributed`` is v1; production ``run(devices=)`` is v2.

    Two modules export ``run_distributed`` (§1.1 / §1.10). The package-level
    name is ``distributed.py``'s pmap runner, while ``sim.run(devices=[...])``
    dispatches to ``distributed_v2.run_distributed`` and only falls back to v1
    at ``n_devices == 1``. Pinning which one the package re-exports keeps a
    consolidation from quietly swapping the public meaning of the name -- the
    two are NOT bit-identical (2.794e-09 on a 9.4145e-03 peak).
    """
    import rfx.runners as pkg
    import rfx.runners.distributed as v1
    import rfx.runners.distributed_v2 as v2

    assert pkg.run_distributed is v1.run_distributed
    assert pkg.run_distributed is not v2.run_distributed


def test_runners_package_eager_import_list_is_unchanged():
    """Read with ast, so it measures the SOURCE and not a polluted sys.modules."""
    init = Path(importlib.import_module("rfx.runners").__file__)
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    found = tuple(
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    assert found == _RUNNERS_EAGER_IMPORTS, (
        f"rfx/runners/__init__.py's eager imports are now {found}, expected "
        f"{_RUNNERS_EAGER_IMPORTS}. Adding a submodule to the package is safe, "
        "but anything a new shared module imports at module level joins this "
        "eager chain -- see conftest.py:405 and "
        "tests/unit/runners/test_runner_import_binding.py.")
