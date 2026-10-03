"""B3b public distributed PMC refusal and the deferred B4 kernel rule.

The direct non-uniform kernel still pins its legacy physical H-zero plane.
Public multi-device execution refuses the declared-face image until B4.
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
    """Mirror of ``tests/unit/runners/test_distributed_nu_kernel.py::_phase2b_shard_mat``.

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

    from rfx.runners._distributed_common import _split_materials
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

@pytest.mark.parametrize("face", ["x_lo", "z_lo"])
@pytest.mark.parametrize("skip_preflight", [False, True])
def test_pmc_distributed_v2_refuses_declared_face(face, skip_preflight):
    """Public distributed kernels retain the half-cell rule and refuse until B4."""
    devices = _require_two_devices()
    spec = BoundarySpec(x=Boundary("pmc", "cpml") if face == "x_lo" else "cpml",
                        y="cpml", z=Boundary("pmc", "pec") if face == "z_lo" else "cpml")
    sim = Simulation(freq_max=5e9, domain=(.08, .04, .12), dx=.005, boundary=spec)
    sim.add_source((.04, .02, .06), "ex")
    sim.add_probe((.045, .02, .06), "ex")
    with pytest.raises(NotImplementedError, match=face + ".*distributed_v2.*magnetic image"):
        sim.run(n_steps=30, devices=devices, skip_preflight=skip_preflight)
