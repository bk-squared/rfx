"""A/B bit-identity harness for the #1038 consolidation of ``rfx/runners/``.

Issue #1038 moves duplicated helpers out of ``distributed.py`` /
``distributed_v2.py`` / ``distributed_nu.py`` into the existing shared
``rfx/runners/_distributed_common.py``, one leg at a time. Every leg is PURE
CODE MOTION, and the code-motion rule gates a
refactor on BIT IDENTITY before and after -- ``np.array_equal`` on the raw arrays,
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

Fifteen since #1053 leg 0. That issue ports ``distributed_nu``'s realized-PEC
mask stage onto ``distributed_v2``, and the two rows it adds
(``distributed_v2_pec_body_seam`` / ``…_interior``) pin the kernel on the body
geometry v2 acquires, captured on the pre-move tree. The thirteen #1038 rows
above are untouched by that capture and keep their original digests.

Thirteen since #1296, which removed the pmap runner and with it the two
``distributed_v1_*`` rows. The v1-versus-v2 measurement above is theirs and
stands as recorded; the other thirteen keep their digests.

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
        "tests/unit/nonuniform/test_nu_flux_monitor_finite_size.py,"
        "tests/unit/runners/test_distributed_nu_smoke.py,"
        "tests/unit/runners/test_distributed.py,"
        "tests/unit/runners/test_distributed_nu_kernel.py,"
        "tests/unit/runners/test_distributed_nu_pec_mask_lane_parity.py"
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
from rfx.runners._rank import mesh_ranks
from rfx.sources.sources import GaussianPulse, ModulatedGaussian
from tests._stated_plan import stated_plan

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


def _open_graded_sim(dz_profile, nx=44, dx=0.5e-3, f0=9e9):
    """The probe prevents an empty time_series from leaving the nonuniform probe path unwitnessed."""
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


def _padx_sim():
    lx = 0.125
    sim = Simulation(freq_max=3e9, domain=(lx, 0.04, 0.04), boundary="cpml")
    sim.add_source(position=(lx / 2, 0.02, 0.02), component="ez",
                   waveform=GaussianPulse(f0=1.5e9, bandwidth=0.8))
    sim.add_probe(position=(lx - 0.006, 0.02, 0.02), component="ez")
    assert sim._build_realized_grid().nx % 2 == 1, (
        "An odd x cell count exercises padding on two devices.")
    return sim


def _f_distributed_v2_pad_x():
    """Preflight is skipped to keep the probe where an absorber displacement is visible."""
    return _padx_sim().run(n_steps=100, devices=jax.devices()[:2],
                           skip_preflight=True)


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
    from rfx.runners._distributed_common import _split_materials

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
        measurement=stated_plan(sharded_grid, [prb], n_steps=n_steps),
    )


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
    )

    n_devices = 2
    xbar = np.zeros((_SEAM_NX, _SEAM_NY, _SEAM_NZ), bool)
    xbar[:, 2:4, 2:4] = True
    assert np.any(xbar[_SEAM_NX // n_devices - 1] & xbar[_SEAM_NX // n_devices]), (
        "The x bar must span both ranks across the shard seam.")
    gmask = _seam_plates(2, [0, _SEAM_NZ - 1]) | xbar
    hard_slabs, nx_local = _seam_slabs(gmask, n_devices, bool, False)
    from tests.unit.materials.test_occupancy_stage import step_occupancy_on_ones
    mesh = Mesh(np.asarray(jax.devices()[:n_devices]).reshape(n_devices),
                ("x",))
    shape = (n_devices * nx_local, _SEAM_NY, _SEAM_NZ)
    state = init_state(shape)
    one = jnp.ones(shape, jnp.float32)
    live = state._replace(ex=one, ey=one, ez=one)
    effective = step_occupancy_on_ones(gmask.astype(np.float32), n_devices)
    return {
        "hard": _apply_pec_mask_nu_shmap(live, jnp.asarray(hard_slabs), mesh,
                                         n_devices, nx_local, ranks=mesh_ranks(mesh)),
        "soft": live._replace(ex=effective.ex, ey=effective.ey, ez=effective.ez),
    }


def _v2_body_mask(x_lo, x_hi):
    """A solid rectangular PEC block, x in [x_lo, x_hi), y 2:5, z 1:4."""
    m = np.zeros((_SEAM_NX, _SEAM_NY, _SEAM_NZ), bool)
    m[x_lo:x_hi, 2:5, 1:4] = True
    return m


def _f_distributed_v2_pec_body(gmask):
    """The body fixtures stay kernel-level because re-capturing them through the public route would throw away their pre-move reference."""
    from jax.sharding import Mesh

    from rfx.core.yee import init_state
    from rfx.runners.distributed_nu import _apply_pec_mask_nu_shmap

    n_devices = 2
    slabs, nx_local = _seam_slabs(gmask, n_devices, bool, False)
    mesh = Mesh(np.asarray(jax.devices()[:n_devices]).reshape(n_devices),
                ("x",))
    shape = (n_devices * nx_local, _SEAM_NY, _SEAM_NZ)
    state = init_state(shape)
    one = jnp.ones(shape, jnp.float32)
    live = state._replace(ex=one, ey=one, ez=one)
    return {
        "hard": _apply_pec_mask_nu_shmap(live, jnp.asarray(slabs), mesh,
                                         n_devices, nx_local, ranks=mesh_ranks(mesh)),
    }


def _f_distributed_v2_pec_body_seam():
    return _f_distributed_v2_pec_body(_v2_body_mask(3, 5))


def _f_distributed_v2_pec_body_interior():
    """The interior fixture is the control for isolating a seam-only regression."""
    return _f_distributed_v2_pec_body(_v2_body_mask(1, 3))


def _subgrid_coarse_shadow_sim():
    sim = Simulation(freq_max=4e9, domain=(0.012, 0.012, 0.012),
                     boundary="pec", dx=0.004)
    sim.add_refinement(z_range=(0.004, 0.012), ratio=3, validation="research")
    sim._refinement["inject_sources_on_coarse_shadow"] = True
    sim.add_source((0.004, 0.004, 0.008), "ez")
    sim.add_probe((0.008, 0.008, 0.008), "ez")
    return sim


def _f_subgridded():
    """The 30 steps avoid an all-zero probe trace."""
    return _subgrid_coarse_shadow_sim().run(n_steps=30, compute_s_params=False)


_FIXTURES = (
    ("uniform_cpml", _f_uniform_cpml, "uniform", False),
    ("uniform_pec_dielectric", _f_uniform_pec_dielectric, "uniform", False),
    ("nonuniform_graded_flux", _f_nonuniform_graded_flux, "nonuniform", False),
    ("nonuniform_constant_profile", _f_nonuniform_constant_profile,
     "nonuniform", False),
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
    ("distributed_v2_pec_body_seam", _f_distributed_v2_pec_body_seam,
     "distributed_v2", True),
    ("distributed_v2_pec_body_interior", _f_distributed_v2_pec_body_interior,
     "distributed_v2", True),
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
    if hasattr(obj, "_fields"):
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
    from rfx.sparams._record_selection import SReadPorts
    if isinstance(obj, SReadPorts):
        out[f"{key}.ids"] = np.asarray(obj.ids, dtype=str)
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
        if x != y and not (x != x and y != y):
            return f"index {np.unravel_index(i, a.shape)}: {x!r} != {y!r}"
    return "(none found - arrays differ only in a way this scan cannot see)"


@pytest.mark.parametrize(
    "fixture,build,runner,multi", _FIXTURES,
    ids=[f for f, _, _, _ in _FIXTURES],
)
def test_runner_is_bit_identical_to_the_pre_split_baseline(
        fixture, build, runner, multi):
    """Advisory warnings are outside the bit-identity gate."""
    base = _baseline_dir()
    if base is None:
        pytest.skip(_SKIP_REASON)
    if multi and jax.device_count() < 2:
        pytest.skip(f"{fixture} ({runner}) {_NEEDS_2DEV}")

    with warnings.catch_warnings():
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


_SHARED_HELPER_BINDINGS = (
    ("exchange_component_shmap", "rfx.runners.distributed_v2",
     "_exchange_component_shmap"),
    ("exchange_component_shmap", "rfx.runners.distributed_nu",
     "_exchange_component_nu_shmap"),
    ("shard_stacked", "rfx.runners.distributed_nu", "shard_stacked"),
    ("shard_stacked", "rfx.runners.distributed_v2", "shard_stacked"),
    ("shard_stacked_poles", "rfx.runners.distributed_nu",
     "shard_stacked_poles"),
    ("inject_sources_shmap", "rfx.runners.distributed_nu",
     "inject_sources_shmap"),
    ("inject_sources_shmap", "rfx.runners.distributed_v2",
     "inject_sources_shmap"),
    ("sample_probes_shmap", "rfx.runners.distributed_nu",
     "sample_probes_shmap"),
    ("sample_probes_shmap", "rfx.runners.distributed_v2",
     "sample_probes_shmap"),
    ("_distributed_boundary_layers", "rfx.runners.distributed_v2", "_distributed_boundary_layers"),
    ("_distributed_boundary_layers", "rfx.runners.distributed_nu", "_distributed_boundary_layers"),
    ("split_array_x", "rfx.runners.distributed", "split_array_x"),
    ("split_array_x", "rfx.runners.distributed_v2", "split_array_x"),
    ("gather_array_x", "rfx.runners.distributed", "gather_array_x"),
    ("gather_array_x", "rfx.runners.distributed_v2", "gather_array_x"),
    ("unstack_and_gather", "rfx.runners.distributed_nu", "unstack_and_gather"),
    ("unstack_and_gather", "rfx.runners.distributed_v2", "unstack_and_gather"),
    ("split_poles_x", "rfx.runners.distributed", "split_poles_x"),
    ("apply_pec_face_shmap", "rfx.runners.distributed_v2",
     "_apply_pec_shmap"),
    ("apply_pec_face_shmap", "rfx.runners.distributed_nu",
     "_apply_pec_face_nu_shmap"),
    ("apply_pec_mask_shmap", "rfx.runners.distributed_nu",
     "_apply_pec_mask_nu_shmap"),
    ("apply_pec_mask_shmap", "rfx.runners.distributed_v2",
     "apply_pec_mask_shmap"),
    ("_update_h_local_nu", "rfx.runners.distributed_nu",
     "_update_h_local_nu"),
    ("_update_e_local_nu", "rfx.runners.distributed_nu",
     "_update_e_local_nu"),
    ("update_h_nu_shmap", "rfx.runners.distributed_nu", "update_h_nu_shmap"),
    ("update_e_nu_shmap", "rfx.runners.distributed_nu", "update_e_nu_shmap"),
    ("slab_e_component_materials", "rfx.runners.distributed_nu",
     "slab_e_component_materials"),
    ("slab_e_component_materials", "rfx.runners.distributed_v2",
     "slab_e_component_materials"),
    ("slab_e_materials_shmap", "rfx.runners.distributed_nu",
     "slab_e_materials_shmap"),
    ("_split_state", "rfx.runners.distributed_v2", "_split_state"),
    ("_split_state", "rfx.runners.distributed", "_split_state"),
    ("_split_materials", "rfx.runners.distributed_v2", "_split_materials"),
    ("_split_materials", "rfx.runners.distributed", "_split_materials"),
    ("_split_debye_coeffs", "rfx.runners.distributed_v2",
     "_split_debye_coeffs"),
    ("_split_debye_coeffs", "rfx.runners.distributed", "_split_debye_coeffs"),
    ("_split_debye_state", "rfx.runners.distributed_v2", "_split_debye_state"),
    ("_split_debye_state", "rfx.runners.distributed", "_split_debye_state"),
    ("_split_lorentz_coeffs", "rfx.runners.distributed_v2",
     "_split_lorentz_coeffs"),
    ("_split_lorentz_coeffs", "rfx.runners.distributed",
     "_split_lorentz_coeffs"),
    ("_split_lorentz_state", "rfx.runners.distributed_v2",
     "_split_lorentz_state"),
    ("_split_lorentz_state", "rfx.runners.distributed", "_split_lorentz_state"),
    ("_update_h_local", "rfx.runners.distributed_v2", "_update_h_local"),
    ("_update_h_local", "rfx.runners.distributed", "_update_h_local"),
    ("_update_e_local", "rfx.runners.distributed", "_update_e_local"),
    ("_update_e_debye_local", "rfx.runners.distributed",
     "_update_e_debye_local"),
    ("_update_e_lorentz_local", "rfx.runners.distributed",
     "_update_e_lorentz_local"),
    ("_update_e_local_with_dispersion", "rfx.runners.distributed_v2",
     "_update_e_local_with_dispersion"),
    ("_update_e_local_with_dispersion", "rfx.runners.distributed",
     "_update_e_local_with_dispersion"),
)


@pytest.mark.parametrize(
    "shared,module,attr", _SHARED_HELPER_BINDINGS,
    ids=[f"{m.rsplit('.', 1)[-1]}.{a}" for _, m, a in _SHARED_HELPER_BINDINGS],
)
def test_shared_runner_helper_is_the_same_object_in_every_importer(
        shared, module, attr):
    """The PEC cell-mask identity rows forbid v2 growing its own spelling of the four-incident-cell rule."""
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


_RUNNER_ENTRY_POINTS = (
    ("uniform", "run_uniform", "rfx.runners.uniform", "run_uniform"),
    ("nonuniform", "run_nonuniform", "rfx.stepping.graded_loop", "run_nonuniform"),
    ("nonuniform", "run_nonuniform_path", "rfx.runners.nonuniform",
     "run_nonuniform_path"),
    ("nonuniform", "run_nonuniform_until_decay", "rfx.stepping.graded_loop",
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

_RUNNERS_ALL = ("run_nonuniform_path", "run_subgridded_path", "run_uniform")

_RUNNERS_EAGER_IMPORTS = (
    ("rfx.runners.nonuniform", "run_nonuniform_path"),
    ("rfx.runners.subgridded", "run_subgridded_path"),
    ("rfx.runners.uniform", "run_uniform"),
)


@pytest.mark.parametrize(
    "module,name,defined_in,qualname", _RUNNER_ENTRY_POINTS,
    ids=[f"{m}.{n}" for m, n, _, _ in _RUNNER_ENTRY_POINTS],
)
def test_runner_module_still_exports_its_entry_point(
        module, name, defined_in, qualname):
    """The defining module is pinned because refusal guards assert the enclosing traceback frame."""
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
        f"{_RUNNERS_ALL}. Inventory §8 leg 6 dropped 'run_distributed' from it "
        "(retiring v1 as a public entry point); that was a deliberate surface "
        "change and it edited this pin in the same commit. Any further change "
        "-- including RE-ADDING 'run_distributed', which would restore the "
        "collision leg 6 removed -- must do the same, not arrive as a side "
        "effect of code motion.")


def test_the_package_no_longer_exports_run_distributed():
    """#1038 leg 6: neither ``run_distributed`` is a package-level name.

    Replaces ``test_the_run_distributed_name_collision_still_resolves_to_v1``,
    which pinned the PRE-leg-6 surface: ``rfx.runners.run_distributed`` used to
    resolve to ``distributed.py``'s pmap runner. That was the wrong runner to
    put behind the public name -- ``sim.run(devices=[...])`` dispatches to
    ``distributed_v2.run_distributed`` for uniform and non-uniform grids alike,
    and reaches v1 only as v2's ``n_devices == 1`` fast path
    (``distributed_v2.py:514-517``). Leg 6 resolved the collision by exporting
    NEITHER, rather than by swapping which one wins: the two are NOT
    bit-identical (2.794e-09 on a 9.4145e-03 peak) and they disagree on the
    odd-``nx`` rule, so a silent swap of the public meaning of the name would
    have changed numbers for anyone who had imported it.

    What this now pins: the package exports no ``run_distributed``, and both
    functions still exist as distinct callables reachable only by full module
    path. The v1 MODULE keeps its entry point -- see
    ``test_runner_module_still_exports_its_entry_point[distributed.run_distributed]``.
    """
    import rfx.runners as pkg
    import rfx.runners.distributed as v1
    import rfx.runners.distributed_v2 as v2

    assert not hasattr(pkg, "run_distributed"), (
        "rfx.runners.run_distributed is back. #1038 leg 6 retired v1 as a "
        "public entry point precisely so this name cannot point at a runner "
        "sim.run() does not dispatch to; re-exporting either one restores the "
        "collision. Migration for callers is Simulation.run(devices=[...]) or "
        "the full module path.")
    assert "run_distributed" not in pkg.__all__

    assert callable(v1.run_distributed)
    assert callable(v2.run_distributed)
    assert v1.run_distributed is not v2.run_distributed, (
        "rfx.runners.distributed.run_distributed and "
        "rfx.runners.distributed_v2.run_distributed collapsed to one object. "
        "v1 is NOT merged into v2 -- they are "
        "not bit-identical and they disagree on the odd-nx rule.")
    assert v1.run_distributed.__module__ == "rfx.runners.distributed"
    assert v2.run_distributed.__module__ == "rfx.runners.distributed_v2"

    with pytest.raises(RuntimeError) as excinfo:
        v1.run_distributed(object(), n_steps=1)
    msg = str(excinfo.value)
    for needle in ("was removed", "Simulation.run(",
                   "rfx.runners.distributed_v2.run_distributed"):
        assert needle in msg, msg


@pytest.mark.source_scan
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
