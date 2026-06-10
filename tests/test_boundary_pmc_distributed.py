"""T8 (2026-04) — sharded PMC runtime smoke tests.

Covers the three sharded runners:
  - ``rfx/runners/distributed_nu.py``   — NU path, reached via direct
                                          call to
                                          ``run_nonuniform_distributed_pec``
                                          (the PRD-intended route; the
                                          stateless ``ForwardResult`` from
                                          ``sim.forward(distributed=True)``
                                          does not carry ``state``).
  - ``rfx/runners/distributed_v2.py``  — default uniform distributed,
                                          reached via
                                          ``sim.run(..., devices=...)``.
  - ``rfx/runners/distributed.py``     — legacy uniform (pmap) path,
                                          reached via a direct call to
                                          ``run_distributed`` — this is
                                          the only code path that exercises
                                          the T8 (2026-04) ``_apply_pmc_local``
                                          hook inside its pmap scan body.

Case 2 includes a NEGATIVE assertion that non-owning ranks' x-face
slabs are NOT zeroed — catches "zero everywhere" bugs that pass the
positive assertion.
"""
# Simulate 2 devices on CPU. Must be set BEFORE importing JAX.
import os  # noqa: I001

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2"
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from rfx import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.core.yee import MaterialArrays  # noqa: E402


def _require_two_devices():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "need 2 virtual devices "
            "(XLA_FLAGS=--xla_force_host_platform_device_count=2)"
        )
    return list(devices[:2])


def _shard_materials_nu(materials, sharded_grid):
    """Mirror of ``tests/test_distributed_nu_kernel.py::_phase2b_shard_mat``.

    Shards a full-domain MaterialArrays for the NU sharded runner.
    """
    n_devices = sharded_grid.n_devices
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x

    devices = jax.devices()[:n_devices]
    mesh = Mesh(np.array(devices), axis_names=("x",))
    shd = NamedSharding(mesh, P("x"))

    if pad_x > 0:
        pad = ((0, pad_x), (0, 0), (0, 0))
        materials = MaterialArrays(
            eps_r=jnp.pad(materials.eps_r, pad, constant_values=1.0),
            sigma=jnp.pad(materials.sigma, pad, constant_values=0.0),
            mu_r=jnp.pad(materials.mu_r, pad, constant_values=1.0),
        )

    from rfx.runners.distributed import _split_materials
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


# ---------------------------------------------------------------------------
# Case 1 — NU path, z_lo PMC
# ---------------------------------------------------------------------------

def test_pmc_distributed_nu_z_lo():
    """PMC on z_lo via the sharded NU runner.

    z-face PMC is rank-invariant under x-slab decomposition, so every
    rank must zero tangential H (``hx``, ``hy``) at the z_lo face.
    Gathers ``final_state`` from the runner output and asserts.
    """
    devices = _require_two_devices()
    n_devices = len(devices)

    from rfx.nonuniform import position_to_index as _nu_pos_to_idx
    from rfx.runners.distributed_nu import (
        build_sharded_nu_grid,
        run_nonuniform_distributed_pec,
    )
    from rfx.simulation import ProbeSpec, SourceSpec
    from rfx.sources.sources import GaussianPulse

    dx = 5e-3
    nx, ny, nz = 24, 8, 8
    # NU path requires profiles on at least one axis.
    dx_profile = np.full(nx, dx, dtype=np.float64)
    dy_profile = np.full(ny, dx, dtype=np.float64)
    dz_profile = np.full(nz, dx, dtype=np.float64)

    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        dx_profile=dx_profile,
        dy_profile=dy_profile,
        dz_profile=dz_profile,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="cpml"),
        ),
    )
    src_pos = (nx // 4 * dx, ny // 2 * dx, nz // 2 * dx)
    prb_pos = ((nx // 4 + 2) * dx, ny // 2 * dx, nz // 2 * dx)
    sim.add_source(src_pos, "ez")
    sim.add_probe(prb_pos, "ez")

    grid = sim._build_nonuniform_grid()
    materials, _db, _lz, _pmask = sim._assemble_materials_nu(grid)

    n_steps = 5
    t = jnp.arange(n_steps, dtype=jnp.float32) * float(grid.dt)
    wf = GaussianPulse(f0=1e9, bandwidth=0.8)(t)

    src_ijk = _nu_pos_to_idx(grid, src_pos)
    prb_ijk = _nu_pos_to_idx(grid, prb_pos)
    sources = [SourceSpec(i=int(src_ijk[0]), j=int(src_ijk[1]),
                          k=int(src_ijk[2]),
                          component="ez", waveform=wf)]
    probes = [ProbeSpec(i=int(prb_ijk[0]), j=int(prb_ijk[1]),
                        k=int(prb_ijk[2]), component="ez")]

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)
    sharded_mat = _shard_materials_nu(materials, sharded_grid)

    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=sources,
        probes=probes,
        n_devices=n_devices,
        devices=devices,
        pmc_faces=frozenset({"z_lo"}),
    )
    state = out["final_state"]
    hx = np.asarray(state.hx)
    hy = np.asarray(state.hy)
    # z_lo PMC: tangential H (hx, hy) at k=0 must be zero on every rank.
    assert np.allclose(hx[:, :, 0], 0.0), (
        f"hx[:,:,0] nonzero on z_lo: max |hx| = "
        f"{float(np.max(np.abs(hx[:, :, 0]))):.3e}"
    )
    assert np.allclose(hy[:, :, 0], 0.0), (
        f"hy[:,:,0] nonzero on z_lo: max |hy| = "
        f"{float(np.max(np.abs(hy[:, :, 0]))):.3e}"
    )


# ---------------------------------------------------------------------------
# Case 2 — distributed_v2 (uniform), x_lo PMC — owning + non-owning
# ---------------------------------------------------------------------------

def test_pmc_distributed_v2_x_lo_owner_and_non_owner():
    """PMC on x_lo via ``sim.run(..., devices=...)`` -> distributed_v2.

    Rank 0 owns x_lo: ``hy[0, :, :]`` and ``hz[0, :, :]`` must be zero
    on the gathered full-domain state (global index 0 is rank 0's first
    real cell).

    Negative assertion: an interior global-x index that lives on rank 1
    (``nx // 2 + 2``) must NOT be zeroed. A "zero everywhere" bug would
    pass the positive assertion and fail this one.
    """
    devices = _require_two_devices()

    dx = 5e-3
    nx, ny, nz = 16, 8, 8
    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        boundary=BoundarySpec(
            x=Boundary(lo="pmc", hi="cpml"),
            y="cpml", z="cpml",
        ),
    )
    sim.add_source(
        (nx // 2 * dx, ny // 2 * dx, nz // 2 * dx), "ez"
    )
    sim.add_probe(
        ((nx // 2 + 1) * dx, ny // 2 * dx, nz // 2 * dx), "ez"
    )
    # Drive long enough for H at interior cells on rank 1 to be nonzero.
    result = sim.run(n_steps=30, devices=devices)
    hy = np.asarray(result.state.hy)
    hz = np.asarray(result.state.hz)

    # Owning rank (rank 0) — global x index 0 is the first real cell.
    assert np.allclose(hy[0, :, :], 0.0), (
        f"x_lo owning rank hy not zeroed: max |hy| = "
        f"{float(np.max(np.abs(hy[0, :, :]))):.3e}"
    )
    assert np.allclose(hz[0, :, :], 0.0), (
        f"x_lo owning rank hz not zeroed: max |hz| = "
        f"{float(np.max(np.abs(hz[0, :, :]))):.3e}"
    )

    # Non-owning rank (rank 1) — interior index well inside its slab.
    # Under 2-device x-slab decomposition rank 1 owns x in [nx//2, nx).
    non_owner_idx = nx // 2 + 2
    max_hy = float(np.max(np.abs(hy[non_owner_idx, :, :])))
    max_hz = float(np.max(np.abs(hz[non_owner_idx, :, :])))
    # At least one tangential H component must be nonzero on rank 1 —
    # otherwise a "zero every rank's x_lo face" bug would pass silently.
    assert max_hy > 1e-20 or max_hz > 1e-20, (
        f"non-owning rank (x={non_owner_idx}) has both |hy|={max_hy:.3e} "
        f"and |hz|={max_hz:.3e} at zero — x_lo PMC over-reaches across "
        f"ranks (catches 'zero everywhere' bug)"
    )


# ---------------------------------------------------------------------------
# Case 3 — distributed.py legacy pmap uniform path, mixed PMC + PEC on z
# ---------------------------------------------------------------------------

def test_pmc_distributed_legacy_mixed_z():
    """Mixed PMC (z_lo) + PEC (z_hi) via the legacy ``rfx.runners.distributed``
    pmap runner.

    ``sim.run(..., devices=[2])`` routes through
    ``rfx.runners.distributed_v2.run_distributed`` (the v2 shard_map
    runner), so to exercise the legacy pmap runner's
    ``_apply_pmc_local`` hook we call it directly with ``n_devices=1``.
    The hook still runs under pmap; n_devices=1 keeps the test CPU-cheap.

    Assertions:
      - z_lo PMC: tangential H (hx, hy) at k=0 is zero.
      - z_hi PEC: tangential E (ex, ey) at k=-1 is zero.
      - No interference between PMC and PEC hooks (different axis
        ends, different field types).
    """
    devices = _require_two_devices()

    dx = 5e-3
    nx, ny, nz = 16, 8, 24
    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="pec"),
        ),
    )
    sim.add_source(
        (nx // 2 * dx, ny // 2 * dx, nz // 2 * dx), "ex"
    )
    sim.add_probe(
        ((nx // 2 + 1) * dx, ny // 2 * dx, nz // 2 * dx), "ex"
    )

    # Direct call to the legacy distributed runner with n_devices=1 so
    # the _apply_pmc_local hook inside its pmap scan body is exercised.
    # (sim.run(..., devices=devices) would route to distributed_v2.)
    from rfx.runners.distributed import run_distributed as legacy_run
    result = legacy_run(sim, n_steps=30, devices=[devices[0]])

    hx = np.asarray(result.state.hx)
    hy = np.asarray(result.state.hy)
    ex = np.asarray(result.state.ex)
    ey = np.asarray(result.state.ey)

    # z_lo PMC: tangential H = 0
    assert np.allclose(hx[:, :, 0], 0.0), (
        f"z_lo PMC failed on hx: max |hx| = "
        f"{float(np.max(np.abs(hx[:, :, 0]))):.3e}"
    )
    assert np.allclose(hy[:, :, 0], 0.0), (
        f"z_lo PMC failed on hy: max |hy| = "
        f"{float(np.max(np.abs(hy[:, :, 0]))):.3e}"
    )
    # z_hi PEC: tangential E = 0
    assert np.allclose(ex[:, :, -1], 0.0), (
        f"z_hi PEC failed on ex: max |ex| = "
        f"{float(np.max(np.abs(ex[:, :, -1]))):.3e}"
    )
    assert np.allclose(ey[:, :, -1], 0.0), (
        f"z_hi PEC failed on ey: max |ey| = "
        f"{float(np.max(np.abs(ey[:, :, -1]))):.3e}"
    )
