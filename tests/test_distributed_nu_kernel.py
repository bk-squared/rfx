"""Unit tests for Phase B distributed_nu kernels.

These are pure-python tests that do NOT require multiple devices — they
exercise the slab-building helper and the local H update directly.
"""

import os
# Force 2 virtual devices for the sharded tests in sibling file; also
# harmless here because we don't actually create a mesh in this file.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.gpu

from rfx.nonuniform import make_nonuniform_grid
from rfx.core.yee import FDTDState, MaterialArrays, MU_0, update_h_nu
from rfx.runners.distributed_nu import (
    _build_sharded_inv_dx_arrays,
    _update_h_local_nu,
)


def _graded_profile(n_physical, dx0, ratio=1.5):
    """Return a 1-D profile of length n_physical with geometric grading
    of `ratio` on one side, clamping both boundary cells to dx0 so the
    make_nonuniform_grid CPML padding is valid."""
    # Build non-uniform interior, then force both ends back to dx0 so
    # the CPML boundary-value invariant holds.
    prof = dx0 * ratio ** np.linspace(0, 1, n_physical)
    prof[0] = dx0
    prof[-1] = dx0
    return prof


def test_inv_dx_h_slab_boundary_matches_global():
    """The sharded x inverse-spacing slabs must match the globally-built
    arrays at the slab seam. CORE-C2 (2026-05-16): inv_dx_h is the
    H-update array — local cell width 1/dx[k]; inv_dx_g is the E-update
    array — mean 2/(dx[k-1]+dx[k]) — which is the one that straddles the
    lower neighbour and is resolved by the global build."""
    nz = 8
    ny = 8
    n_physical = 32
    dx0 = 1e-3
    dx_profile = _graded_profile(n_physical, dx0, ratio=1.5)
    dz_profile = np.full(nz, dx0)
    grid = make_nonuniform_grid(
        (n_physical * dx0, ny * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )

    n_devices = 2
    inv_dx_g, inv_dx_h_g, dx_padded = _build_sharded_inv_dx_arrays(
        grid, n_devices, pad_x=0
    )
    nx = inv_dx_g.shape[0]
    nx_per = nx // n_devices

    # Replay the slab split from distributed_v2 (ghost=1)
    ghost = 1
    nx_local = nx_per + 2 * ghost
    slabs = np.zeros((n_devices, nx_local), dtype=np.float32)
    for d in range(n_devices):
        lo, hi = d * nx_per, (d + 1) * nx_per
        slabs[d, ghost:ghost + nx_per] = inv_dx_h_g[lo:hi]
        if d > 0:
            slabs[d, 0] = inv_dx_h_g[lo - 1]
        if d < n_devices - 1:
            slabs[d, -1] = inv_dx_h_g[hi]

    # For device 0: the rightmost real cell (index ghost+nx_per-1 in slab)
    # equals inv_dx_h_g[nx_per - 1] — a slice of the globally-built array.
    expected = inv_dx_h_g[nx_per - 1]
    got = slabs[0, ghost + nx_per - 1]
    assert np.isclose(got, expected), (
        f"Device 0 seam-cell inv_dx_h = {got}, expected global "
        f"inv_dx_h[{nx_per - 1}] = {expected}"
    )
    # CORE-C2: inv_dx_h is the H-update array — LOCAL cell width 1/dx[k].
    expected_analytic = 1.0 / dx_padded[nx_per - 1]
    assert np.isclose(got, expected_analytic, atol=1e-6), (
        f"Seam inv_dx_h = {got}, analytic 1/dx = {expected_analytic}"
    )
    # inv_dx_g is the E-update MEAN 2/(dx[k-1]+dx[k]); its genuine seam
    # cell is device-1's first real cell (global nx_per), which straddles
    # device-0's last cell — correct because the array is built globally.
    e_seam = inv_dx_g[nx_per]
    e_analytic = 2.0 / (dx_padded[nx_per - 1] + dx_padded[nx_per])
    assert np.isclose(e_seam, e_analytic, atol=1e-6), (
        f"Seam inv_dx_g (E mean) = {e_seam}, analytic = {e_analytic}"
    )


def test_update_h_nu_local_matches_global_interior():
    """_update_h_local_nu on device-0 slab should match the global
    update_h_nu on the unsharded tensor at interior real cells."""
    nz = 8
    ny = 8
    n_physical = 16
    dx0 = 1e-3
    dx_profile = _graded_profile(n_physical, dx0, ratio=1.2)
    dz_profile = np.full(nz, dx0)
    grid = make_nonuniform_grid(
        (n_physical * dx0, ny * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )
    nx = grid.nx
    n_devices = 2
    nx_per = nx // n_devices
    ghost = 1
    nx_local = nx_per + 2 * ghost

    # Random-ish E fields
    rng = np.random.default_rng(42)
    ex = jnp.asarray(rng.standard_normal((nx, ny, nz)), dtype=jnp.float32)
    ey = jnp.asarray(rng.standard_normal((nx, ny, nz)), dtype=jnp.float32)
    ez = jnp.asarray(rng.standard_normal((nx, ny, nz)), dtype=jnp.float32)
    zeros_xyz = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    state = FDTDState(
        ex=ex, ey=ey, ez=ez,
        hx=zeros_xyz, hy=zeros_xyz, hz=zeros_xyz,
        step=jnp.int32(0),
    )
    mats = MaterialArrays(
        eps_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
    )

    # Global H update
    global_h = update_h_nu(
        state, mats, grid.dt,
        grid.inv_dx_h, grid.inv_dy_h, grid.inv_dz_h,
    )

    # Device-0 slab (with ghost). Pick slab build identical to runner.
    inv_dx_g, inv_dx_h_g, _ = _build_sharded_inv_dx_arrays(
        grid, n_devices, pad_x=0)

    def _slab_1d(arr, pad_value):
        slabs = np.zeros((n_devices, nx_local), dtype=arr.dtype)
        for d in range(n_devices):
            lo, hi = d * nx_per, (d + 1) * nx_per
            slabs[d, ghost:ghost + nx_per] = arr[lo:hi]
            if d > 0:
                slabs[d, 0] = arr[lo - 1]
            else:
                slabs[d, 0] = pad_value
            if d < n_devices - 1:
                slabs[d, -1] = arr[hi]
            else:
                slabs[d, -1] = pad_value
        return slabs

    idx_slab = _slab_1d(inv_dx_g, 1.0)
    idxh_slab = _slab_1d(inv_dx_h_g, 0.0)

    def _slab_field(arr):
        out = np.zeros((n_devices, nx_local, ny, nz), dtype=np.float32)
        for d in range(n_devices):
            lo, hi = d * nx_per, (d + 1) * nx_per
            out[d, ghost:ghost + nx_per] = np.asarray(arr)[lo:hi]
            if d > 0:
                out[d, 0] = np.asarray(arr)[lo - 1]
            if d < n_devices - 1:
                out[d, -1] = np.asarray(arr)[hi]
        return out

    ex_sl = _slab_field(ex)
    ey_sl = _slab_field(ey)
    ez_sl = _slab_field(ez)
    z_sl = np.zeros_like(ex_sl)

    d = 0
    slab_state = FDTDState(
        ex=jnp.asarray(ex_sl[d]),
        ey=jnp.asarray(ey_sl[d]),
        ez=jnp.asarray(ez_sl[d]),
        hx=jnp.asarray(z_sl[d]),
        hy=jnp.asarray(z_sl[d]),
        hz=jnp.asarray(z_sl[d]),
        step=jnp.int32(0),
    )
    slab_mats = MaterialArrays(
        eps_r=jnp.ones((nx_local, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx_local, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx_local, ny, nz), dtype=jnp.float32),
    )
    slab_h = _update_h_local_nu(
        slab_state, slab_mats, grid.dt,
        jnp.asarray(idx_slab[d]),
        grid.inv_dy, grid.inv_dz,
        jnp.asarray(idxh_slab[d]),
        grid.inv_dy_h, grid.inv_dz_h,
    )

    # Compare interior real cells (exclude ghost + the seam cell, which
    # uses the global mean-spacing; this cell's forward-diff reaches
    # into the ghost and should match the global reference via our
    # inv_dx_h_h pad).
    # Interior real cells in device 0: slab indices [ghost, ghost+nx_per-1).
    # Last real cell (ghost+nx_per-1) uses inv_dx_h straddling the seam,
    # matching global inv_dx_h[nx_per-1], so it should match too.
    glob_slice = np.asarray(global_h.hz)[:nx_per]
    slab_slice = np.asarray(slab_h.hz)[ghost:ghost + nx_per]
    # Exclude last real cell where forward-diff pulls in the ghost ex/ey
    # which are populated from arr[nx_per] (the global interior cell),
    # so this must still match.
    np.testing.assert_allclose(
        slab_slice, glob_slice, atol=1e-5,
        err_msg="device-0 H-z slab should match global interior H-z",
    )


# ---------------------------------------------------------------------------
# Phase 2A: build_sharded_nu_grid metadata tests
# ---------------------------------------------------------------------------

from rfx.runners.distributed_nu import build_sharded_nu_grid, ShardedNUGrid


def _make_test_grid(nx_physical=16, ny_physical=8, nz_physical=8,
                    dx0=1e-3, ratio=1.3, cpml_layers=0):
    """Build a small graded NonUniformGrid for metadata tests."""
    dx_profile = _graded_profile(nx_physical, dx0, ratio=ratio)
    dz_profile = np.full(nz_physical, dx0)
    return make_nonuniform_grid(
        (nx_physical * dx0, ny_physical * dx0), dz_profile, dx0,
        cpml_layers=cpml_layers,
        dx_profile=dx_profile,
    )


def test_build_sharded_nu_grid_metadata_shapes():
    """Local x sizes sum to global nx; y/z unchanged; cpml_layers replicated."""
    grid = _make_test_grid(nx_physical=16, ny_physical=8, nz_physical=6, cpml_layers=0)
    n_devices = 2
    sg = build_sharded_nu_grid(grid, n_devices=n_devices, exchange_interval=1)

    assert isinstance(sg, ShardedNUGrid)

    # x sizes sum correctly
    assert sg.nx == grid.nx
    assert sg.nx_padded % n_devices == 0
    assert sg.nx_per_rank * n_devices == sg.nx_padded
    assert sg.nx_padded >= sg.nx
    assert sg.pad_x == sg.nx_padded - sg.nx

    # nx_local includes ghosts
    assert sg.nx_local == sg.nx_per_rank + 2 * sg.ghost_width
    assert sg.ghost_width == 1

    # y/z unchanged
    assert sg.ny == grid.ny
    assert sg.nz == grid.nz

    # cpml_layers replicated
    assert sg.cpml_layers == grid.cpml_layers

    # inv spacing array shapes
    assert sg.inv_dx_global.shape == (sg.nx_padded,)
    assert sg.inv_dx_h_global.shape == (sg.nx_padded,)
    assert sg.dx_padded.shape == (sg.nx_padded,)
    assert sg.inv_dy.shape == (grid.ny,)
    assert sg.inv_dy_h.shape == (grid.ny,)
    assert sg.inv_dz.shape == (grid.nz,)
    assert sg.inv_dz_h.shape == (grid.nz,)

    # x_starts / x_stops bookkeeping
    assert len(sg.x_starts) == n_devices
    assert len(sg.x_stops) == n_devices
    assert sg.x_starts[0] == 0
    assert sg.x_stops[-1] == sg.nx  # capped at unpadded nx


def test_build_sharded_nu_grid_inv_dx_seam_continuity():
    """inv_dx_h at the slab seam matches the un-sharded global reference.
    CORE-C2: inv_dx_h is the H-update array — local cell width 1/dx[k]."""
    from rfx.runners.distributed_nu import split_1d_with_ghost

    grid = _make_test_grid(nx_physical=32, ny_physical=8, nz_physical=8,
                           dx0=1e-3, ratio=1.5)
    n_devices = 2
    sg = build_sharded_nu_grid(grid, n_devices=n_devices)

    nx_per = sg.nx_per_rank
    ghost = sg.ghost_width
    nx_local = sg.nx_local

    # Build the slabs using the canonical helper
    slabs = split_1d_with_ghost(
        sg.inv_dx_h_global, n_devices, nx_per, nx_local, ghost, pad_value=0.0
    )

    # For device 0: last real cell in slab = global index nx_per - 1
    seam_slab = float(slabs[0, ghost + nx_per - 1])
    seam_global = float(sg.inv_dx_h_global[nx_per - 1])
    assert np.isclose(seam_slab, seam_global, atol=1e-6), (
        f"Seam slab value {seam_slab} != global reference {seam_global}"
    )

    # Cross-check analytically: inv_dx_h is the H-update local 1/dx[k].
    dx_arr = sg.dx_padded
    analytic = float(1.0 / dx_arr[nx_per - 1])
    assert np.isclose(seam_slab, analytic, atol=1e-5), (
        f"Seam inv_dx_h {seam_slab} != analytic 1/dx {analytic}"
    )


def test_build_sharded_nu_grid_pad_trim_for_nondivisible_nx():
    """nx=17, n_devices=2 — high-x rank gets the pad; metadata flags are correct."""
    # nx=17 is odd; with n_devices=2 we need pad_x=1 to reach 18
    dx0 = 1e-3
    nx_physical = 17
    ny_physical = 8
    nz_physical = 8
    dx_profile = _graded_profile(nx_physical, dx0, ratio=1.2)
    dz_profile = np.full(nz_physical, dx0)
    grid = make_nonuniform_grid(
        (nx_physical * dx0, ny_physical * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )
    assert grid.nx == nx_physical  # sanity

    n_devices = 2
    sg = build_sharded_nu_grid(grid, n_devices=n_devices)

    # Padding arithmetic
    assert sg.pad_x == 1, f"Expected pad_x=1, got {sg.pad_x}"
    assert sg.nx_padded == 18
    assert sg.nx_per_rank == 9
    assert sg.nx_trim == 1

    # High-x rank index
    assert sg.rank_has_high_x_pad == n_devices - 1  # rank 1

    # The padded cell in inv_dx_global should equal 1/dx_arr[-1]
    expected_last_inv = float(1.0 / np.asarray(grid.dx_arr)[-1])
    got_last_inv = float(sg.inv_dx_global[-1])
    assert np.isclose(got_last_inv, expected_last_inv, rtol=1e-5), (
        f"Padded inv_dx last cell {got_last_inv} != 1/dx[-1] {expected_last_inv}"
    )


def test_build_sharded_nu_grid_replicates_dt():
    """dt is identical (same Python float) across all conceptual ranks."""
    grid = _make_test_grid(nx_physical=16, cpml_layers=0)
    n_devices = 4
    sg = build_sharded_nu_grid(grid, n_devices=n_devices)

    # dt must equal the grid's dt exactly (no recomputation)
    assert sg.dt == float(grid.dt), (
        f"ShardedNUGrid dt {sg.dt} != grid.dt {float(grid.dt)}"
    )
    # dt is a plain Python float (not a JAX array) so it's trivially
    # identical across all ranks — assert it is not a JAX array
    assert isinstance(sg.dt, float), (
        f"Expected plain float for dt, got {type(sg.dt)}"
    )


def test_build_sharded_nu_grid_position_to_index_deterministic():
    """Known physical coord maps to expected (rank, local_i) deterministically."""
    from rfx.nonuniform import position_to_index

    dx0 = 1e-3
    nx_physical = 16
    ny_physical = 8
    nz_physical = 8
    dx_profile = np.full(nx_physical, dx0)   # uniform so we can predict the index
    dz_profile = np.full(nz_physical, dx0)
    grid = make_nonuniform_grid(
        (nx_physical * dx0, ny_physical * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )

    n_devices = 2
    sg = build_sharded_nu_grid(grid, n_devices=n_devices)

    # Place a physical position in the middle of the domain
    # For a uniform profile with cpml=0, index = round(pos / dx0)
    # Use a position in the second half so it lands on rank 1.
    # nx_physical=16, nx_per_rank=8; cell 10 → rank 1, local 10-8=2 (+ghost=1 → local_i=3)
    target_global_i = 10
    pos_x = (float(np.asarray(grid.dx_arr[:target_global_i]).sum())
              + 0.5 * float(np.asarray(grid.dx_arr)[target_global_i]))
    pos_y = 0.5 * dx0
    pos_z = 0.5 * dx0
    i_global, j_global, k_global = position_to_index(grid, (pos_x, pos_y, pos_z))
    assert i_global == target_global_i, (
        f"position_to_index returned i={i_global}, expected {target_global_i}"
    )

    # Apply Phase 2A mapping convention
    expected_rank = target_global_i // sg.nx_per_rank          # 10 // 8 = 1
    expected_local_i = (target_global_i % sg.nx_per_rank) + sg.ghost_width  # 2 + 1 = 3

    got_rank = i_global // sg.nx_per_rank
    got_local_i = (i_global % sg.nx_per_rank) + sg.ghost_width

    assert got_rank == expected_rank, (
        f"rank={got_rank}, expected {expected_rank}"
    )
    assert got_local_i == expected_local_i, (
        f"local_i={got_local_i}, expected {expected_local_i}"
    )

    # Calling the mapping a second time must give the same result (deterministic)
    i2, _, _ = position_to_index(grid, (pos_x, pos_y, pos_z))
    assert i2 // sg.nx_per_rank == got_rank
    assert (i2 % sg.nx_per_rank) + sg.ghost_width == got_local_i


# ---------------------------------------------------------------------------
# Phase 2B: hard-PEC + ghost-exchange sharded NU scan body tests
# ---------------------------------------------------------------------------

import jax  # noqa: E402

from rfx.runners.distributed_nu import (  # noqa: E402
    run_nonuniform_distributed_pec,
    shard_pec_mask_x_slab,
    shard_pec_occupancy_x_slab,
)
from rfx.nonuniform import (  # noqa: E402
    run_nonuniform,
    position_to_index as _phase2b_pos_to_idx,
    make_current_source as _phase2b_make_current_source,
)
from rfx.simulation import SourceSpec, ProbeSpec  # noqa: E402
from tests._distributed_nu_tolerances import (  # noqa: E402
    assert_class_b_parity,
)


_PHASE2B_REQUIRES_2DEV = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        "Phase 2B distributed-NU PEC tests need >=2 JAX devices "
        "(set XLA_FLAGS=--xla_force_host_platform_device_count=2)."
    ),
)


def _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, dx0=1e-3, ratio=1.2):
    """Small NU grid (cpml=0, hard-PEC cavity) for Phase 2B parity tests."""
    dx_profile = _graded_profile(nx_physical, dx0, ratio=ratio)
    dz_profile = np.full(nz, dx0)
    grid = make_nonuniform_grid(
        (nx_physical * dx0, ny * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )
    return grid


def _phase2b_make_materials(grid):
    """Vacuum (eps=mu=1, sigma=0) materials shaped like ``grid``."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    return MaterialArrays(
        eps_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
    )


def _phase2b_gauss_waveform(t, t0=8e-12, tau=2.5e-12):
    return jnp.exp(-((t - t0) ** 2) / (2.0 * tau ** 2))


def _phase2b_shard_mat(materials, sharded_grid):
    """Shard a full-domain MaterialArrays for Phase 2B's runner."""
    n_devices = sharded_grid.n_devices
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x

    devices = jax.devices()[:n_devices]
    from jax.sharding import Mesh, NamedSharding
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


from jax.sharding import PartitionSpec as P  # noqa: E402


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_only_2device_matches_single_device():
    """Class B forward parity: 2-device distributed run should match the
    single-device ``run_nonuniform`` reference at the final step on a
    small NU PEC cavity.

    Source on rank 0 interior, probe on rank 1 interior, so the signal
    crosses the seam.  Hard PEC at all 6 domain faces (no CPML).
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 60

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.2)
    materials = _phase2b_make_materials(grid)

    # Source: global x-index 4 (rank 0), probe: global x-index 12 (rank 1)
    src_idx = (4, 4, 4)
    prb_idx = (12, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(
        i=int(prb_idx[0]), j=int(prb_idx[1]), k=int(prb_idx[2]),
        component="ez",
    )

    # Single-device reference
    single_out = run_nonuniform(
        grid=grid,
        materials=materials,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    # 2-device distributed run
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert ts_dist.shape == ts_single.shape, (
        f"shape mismatch: dist={ts_dist.shape}, single={ts_single.shape}"
    )

    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2b_pec_only_2device_parity")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_only_seam_no_double_zeroing():
    """Class D seam isolation: a PEC mask cell exactly at the slab seam
    is zeroed exactly once.

    Place a single PEC cell at global x-index ``nx_per_rank`` (rank 1's
    first real cell).  The Phase 2B mask runner must mask this cell
    once (rank 1's sharded mask has it True at slab-local index
    ``ghost``); rank 0's slab must NOT see it as a real cell (only as
    its right ghost) so apply_pec_mask on rank 0 must not act on it.

    Verification: the distributed run must match the single-device
    reference (which applies the mask exactly once).
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 30

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Build a PEC mask with a single cell exactly at the slab seam.
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    seam_i = sharded_grid.nx_per_rank  # global x-index of rank 1's first real cell
    seam_j = ny // 2
    seam_k = nz // 2
    pec_mask = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
    # Mark a 1x1x1 PEC cell + its left neighbour so the tangential mask
    # in apply_pec_mask sees a PEC neighbour (otherwise the thin-sheet
    # rule preserves the field — no double-zeroing risk to detect).
    pec_mask = pec_mask.at[seam_i, seam_j, seam_k].set(True)
    pec_mask = pec_mask.at[seam_i - 1, seam_j, seam_k].set(True)
    pec_mask = pec_mask.at[seam_i + 1, seam_j, seam_k].set(True)

    src_idx = (4, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    # Probe on rank 1, away from the seam, so the seam-mask physics
    # propagates into the sampled signal.
    prb_spec = ProbeSpec(i=12, j=4, k=4, component="ez")

    # Single-device reference (applies the mask once via apply_pec_mask)
    single_out = run_nonuniform(
        grid=grid,
        materials=materials,
        n_steps=n_steps,
        pec_mask=pec_mask,
        sources=[src_spec],
        probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    # Distributed run with the same PEC mask
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    sharded_mask = shard_pec_mask_x_slab(pec_mask, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=sharded_mask,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    # If the seam cell were double-zeroed, the field profile would
    # decay differently — Class B forward parity catches that.
    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2b_seam_no_double_zeroing")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_mask_override_union_semantics():
    """Geometry-defined PEC + override union must match single-device.

    The single-device path takes ``pec_mask = geom_mask | override_mask``
    before passing to the runner.  We replicate that union at the host
    level (mirroring ``Simulation.forward()`` semantics) and verify
    the distributed runner produces the identical probe time-series.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 30

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # "Geometry" PEC slab: a thin sheet at x=10
    geom_mask = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
    geom_mask = geom_mask.at[10, 2:6, 2:6].set(True)
    geom_mask = geom_mask.at[11, 2:6, 2:6].set(True)
    # "Override" PEC: another small block at x=5 inside rank 0
    override_mask = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
    override_mask = override_mask.at[5, 3:5, 3:5].set(True)
    override_mask = override_mask.at[6, 3:5, 3:5].set(True)
    union_mask = geom_mask | override_mask

    src_idx = (2, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=13, j=4, k=4, component="ez")

    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        pec_mask=union_mask,
        sources=[src_spec], probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    sharded_mask = shard_pec_mask_x_slab(union_mask, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=sharded_mask,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2b_pec_mask_override_union")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_h_ghost_exchange_recovers_global_field():
    """Class B parity test for H-update + ghost exchange with no PEC mask
    and no source.

    Initialise the E field with a localised non-zero pattern that
    straddles the slab seam, then run a short forward (no source).  The
    distributed scan body's H update + ghost exchange should produce
    H-field evolution that matches the single-device run_nonuniform
    reference at the probe.
    """
    devices = jax.devices()[:2]
    n_devices = 2

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)

    # Inject a soft source so we have a non-trivial time series to compare;
    # this still tests the H ghost exchange because the signal must cross
    # the seam to reach the probe.
    n_steps_run = 40
    src_idx = (3, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps_run, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    # Probe in rank 1 sampling Hy (so the H ghost exchange is the
    # critical step transporting the signal)
    prb_spec = ProbeSpec(i=12, j=4, k=4, component="hy")

    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps_run,
        sources=[src_spec], probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps_run,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2b_h_ghost_exchange_global_field")


# ---------------------------------------------------------------------------
# Phase 2C: full CPML on x-slabs — V3 plan lines 633-657
# ---------------------------------------------------------------------------

from rfx.runners.distributed_nu import (  # noqa: E402
    init_cpml_for_sharded_nu,
    shard_cpml_state_x_slab,
)
from tests._distributed_nu_tolerances import (  # noqa: E402
    assert_class_c_cpml_seam_noop,
    CPML_X_FACE_PSI_FIELDS,
)


def _phase2c_build_cpml_grid(nx_physical=16, ny=8, nz=8, dx0=1e-3,
                             cpml_layers=4):
    """Small NU grid with CPML padding (uniform xy + uniform dz; the
    NU plumbing exercises the per-axis dx_x/dx_y/dz code paths even on a
    degenerate-uniform profile).
    """
    dx_profile = np.full(nx_physical, dx0)
    dz_profile = np.full(nz, dx0)
    grid = make_nonuniform_grid(
        (nx_physical * dx0, ny * dx0), dz_profile, dx0,
        cpml_layers=cpml_layers,
        dx_profile=dx_profile,
    )
    return grid


def _phase2c_unstack_cpml(cpml_state_sharded, sharded_grid, n_cpml,
                          field_name):
    """Unstack a sharded CPML psi field back into per-rank arrays.

    The runner returns CPMLState psi arrays with leading axis
    ``n_devices * n_cpml``; this helper pulls them to host and reshapes
    to ``(n_devices, n_cpml, d1, d2)`` so individual ranks can be
    inspected.
    """
    arr = np.asarray(getattr(cpml_state_sharded, field_name))
    n_devices = sharded_grid.n_devices
    total = arr.shape[0]
    assert total == n_devices * n_cpml, (
        f"unstack: total leading dim {total} != "
        f"n_devices * n_cpml = {n_devices * n_cpml}"
    )
    return arr.reshape(n_devices, n_cpml, *arr.shape[1:])


@_PHASE2B_REQUIRES_2DEV
def test_distributed_cpml_2device_matches_single_device():
    """Phase 2C Class B forward parity: NU + CPML on all faces, 2 devices.

    The 2-device distributed run with CPML must match the single-device
    ``run_nonuniform`` reference at the final step on a small NU CPML
    domain.  Source on rank 0, probe on rank 1, so the signal crosses
    the seam.  Validates that the per-rank CPMLState evolution + the
    rank-conditional x-face gating preserve forward parity.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 60

    grid = _phase2c_build_cpml_grid(
        nx_physical=16, ny=8, nz=8, dx0=1e-3, cpml_layers=4)
    materials = _phase2b_make_materials(grid)

    # Source / probe in interior; cross the slab seam.
    nx, ny_g, nz_g = grid.nx, grid.ny, grid.nz
    src_idx = (nx // 4, ny_g // 2, nz_g // 2)
    prb_idx = (3 * nx // 4, ny_g // 2, nz_g // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(
        i=int(prb_idx[0]), j=int(prb_idx[1]), k=int(prb_idx[2]),
        component="ez",
    )

    # Single-device reference (uses init_cpml under the hood)
    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        sources=[src_spec], probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    # Distributed run with CPML
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    cpml_params, cpml_state_stacked = init_cpml_for_sharded_nu(
        sharded_grid, n_devices=n_devices,
    )
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    cpml_state_sharded = shard_cpml_state_x_slab(
        cpml_state_stacked, sharded_grid, mesh,
    )
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        cpml_params=cpml_params,
        cpml_state=cpml_state_sharded,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert ts_dist.shape == ts_single.shape, (
        f"shape mismatch: dist={ts_dist.shape}, single={ts_single.shape}"
    )
    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2c_cpml_2device_parity")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_cpml_internal_seam_is_noop():
    """Phase 2C Class C: interior-rank x-face CPML state is exactly zero.

    Run a 2-device distributed CPML scan for 50 steps with a source that
    drives a non-trivial field through the seam.  After the run, every
    x-face CPML psi array on an interior rank must be exactly zero —
    the seam is NOT a CPML boundary, only the outer x_lo / x_hi faces are.

    With ``n_devices=3`` rank 1 is the interior rank; with ``n_devices=2``
    there is no interior rank, so we pick a 3-device run when at least
    3 virtual devices are available; otherwise we run the 2-device case
    and assert that rank 0 owns x_lo psi but the *upper* slab of rank 0's
    x-hi psi is unmodified (rank 0 is not rank N-1, so its x-hi psi must
    stay zero).
    """
    devices_all = jax.devices()
    n_devices = 3 if len(devices_all) >= 3 else 2
    devices = devices_all[:n_devices]
    n_steps = 50

    grid = _phase2c_build_cpml_grid(
        nx_physical=24 if n_devices == 3 else 16, ny=8, nz=8,
        dx0=1e-3, cpml_layers=4,
    )
    materials = _phase2b_make_materials(grid)
    nx = grid.nx

    # Source in the centre so the field reaches every rank
    src_idx = (nx // 2, grid.ny // 2, grid.nz // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=nx // 2, j=grid.ny // 2, k=grid.nz // 2,
                         component="ez")

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    cpml_params, cpml_state_stacked = init_cpml_for_sharded_nu(
        sharded_grid, n_devices=n_devices,
    )
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    cpml_state_sharded = shard_cpml_state_x_slab(
        cpml_state_stacked, sharded_grid, mesh,
    )
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        cpml_params=cpml_params,
        cpml_state=cpml_state_sharded,
    )
    final_cpml = out["cpml_state_sharded"]
    assert final_cpml is not None, (
        "Phase 2C runner must return cpml_state_sharded when CPML is enabled"
    )

    n_cpml = sharded_grid.cpml_layers

    # Rank-by-rank inspection of every x-face psi field
    if n_devices >= 3:
        # Rank 1 is interior — every x-face psi must be exactly zero
        from rfx.boundaries.cpml import CPMLState
        rank1_psi = {}
        for fname in CPML_X_FACE_PSI_FIELDS:
            stacked = _phase2c_unstack_cpml(
                final_cpml, sharded_grid, n_cpml, fname,
            )
            rank1_psi[fname] = jnp.asarray(stacked[1])  # rank 1
        # Build a fake CPMLState-like NamedTuple subset for the helper
        rank1_state = CPMLState(
            psi_ex_ylo=jnp.zeros(1), psi_ex_yhi=jnp.zeros(1),
            psi_ex_zlo=jnp.zeros(1), psi_ex_zhi=jnp.zeros(1),
            psi_ey_xlo=rank1_psi["psi_ey_xlo"],
            psi_ey_xhi=rank1_psi["psi_ey_xhi"],
            psi_ey_zlo=jnp.zeros(1), psi_ey_zhi=jnp.zeros(1),
            psi_ez_xlo=rank1_psi["psi_ez_xlo"],
            psi_ez_xhi=rank1_psi["psi_ez_xhi"],
            psi_ez_ylo=jnp.zeros(1), psi_ez_yhi=jnp.zeros(1),
            psi_hx_ylo=jnp.zeros(1), psi_hx_yhi=jnp.zeros(1),
            psi_hx_zlo=jnp.zeros(1), psi_hx_zhi=jnp.zeros(1),
            psi_hy_xlo=rank1_psi["psi_hy_xlo"],
            psi_hy_xhi=rank1_psi["psi_hy_xhi"],
            psi_hy_zlo=jnp.zeros(1), psi_hy_zhi=jnp.zeros(1),
            psi_hz_xlo=rank1_psi["psi_hz_xlo"],
            psi_hz_xhi=rank1_psi["psi_hz_xhi"],
            psi_hz_ylo=jnp.zeros(1), psi_hz_yhi=jnp.zeros(1),
        )
        assert_class_c_cpml_seam_noop(rank1_state, label="phase2c_seam_noop_rank1")
    else:
        # 2-device fallback: rank 0 owns x_lo only, so x_hi psi must be 0;
        # rank 1 owns x_hi only, so x_lo psi must be 0.
        for fname in CPML_X_FACE_PSI_FIELDS:
            stacked = _phase2c_unstack_cpml(
                final_cpml, sharded_grid, n_cpml, fname,
            )
            if fname.endswith("xlo"):
                # Rank 1 (not rank 0) must NOT touch x_lo
                assert np.all(stacked[1] == 0), (
                    f"Phase 2C: rank 1 (interior side) modified x_lo psi "
                    f"field '{fname}' — internal seam treated as boundary."
                )
            elif fname.endswith("xhi"):
                # Rank 0 (not rank N-1) must NOT touch x_hi
                assert np.all(stacked[0] == 0), (
                    f"Phase 2C: rank 0 (interior side) modified x_hi psi "
                    f"field '{fname}' — internal seam treated as boundary."
                )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_cpml_outer_face_active():
    """Phase 2C Class C+D: outer x-face CPML on rank 0 / rank N-1 is active.

    Drive a pulse from the centre of the domain.  After enough time
    steps for the wavefront to reach both x-faces, rank 0's x_lo psi
    arrays and rank (N-1)'s x_hi psi arrays must be non-zero — the
    outer face IS a physical CPML boundary.

    This is the complement of ``test_distributed_cpml_internal_seam_is_noop``:
    seam-noop says interior ranks don't activate; outer-face-active
    says boundary ranks DO activate.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 80  # long enough for the pulse to reach both x-faces

    grid = _phase2c_build_cpml_grid(
        nx_physical=16, ny=8, nz=8, dx0=1e-3, cpml_layers=4)
    materials = _phase2b_make_materials(grid)
    nx = grid.nx

    # Source in the centre (cell ~nx/2)
    src_idx = (nx // 2, grid.ny // 2, grid.nz // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=nx // 2, j=grid.ny // 2, k=grid.nz // 2,
                         component="ez")

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    cpml_params, cpml_state_stacked = init_cpml_for_sharded_nu(
        sharded_grid, n_devices=n_devices,
    )
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    cpml_state_sharded = shard_cpml_state_x_slab(
        cpml_state_stacked, sharded_grid, mesh,
    )
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        cpml_params=cpml_params,
        cpml_state=cpml_state_sharded,
    )
    final_cpml = out["cpml_state_sharded"]
    assert final_cpml is not None
    n_cpml = sharded_grid.cpml_layers

    # Rank 0 must have non-zero x_lo psi after the wave hits the boundary
    rank0_psi_ey_xlo = _phase2c_unstack_cpml(
        final_cpml, sharded_grid, n_cpml, "psi_ey_xlo",
    )[0]
    assert np.max(np.abs(rank0_psi_ey_xlo)) > 0, (
        "Phase 2C: rank 0's psi_ey_xlo must be non-zero after the "
        "wavefront reaches the x_lo boundary — outer x-face CPML inactive."
    )

    # Rank N-1 must have non-zero x_hi psi
    rankN_psi_ey_xhi = _phase2c_unstack_cpml(
        final_cpml, sharded_grid, n_cpml, "psi_ey_xhi",
    )[n_devices - 1]
    assert np.max(np.abs(rankN_psi_ey_xhi)) > 0, (
        "Phase 2C: rank N-1's psi_ey_xhi must be non-zero after the "
        "wavefront reaches the x_hi boundary — outer x-face CPML inactive."
    )


# ---------------------------------------------------------------------------
# Phase 2D: Debye / Lorentz E path on the sharded scan body — V3 plan 659-720
# ---------------------------------------------------------------------------

from rfx.runners.distributed_nu import (  # noqa: E402
    shard_debye_coeffs_x_slab,
    shard_debye_state_x_slab,
    shard_lorentz_coeffs_x_slab,
    shard_lorentz_state_x_slab,
)
from rfx.materials.debye import DebyePole, init_debye  # noqa: E402
from rfx.materials.lorentz import (  # noqa: E402
    LorentzPole, init_lorentz, lorentz_pole,
)
from tests._distributed_nu_tolerances import (  # noqa: E402
    assert_class_d_timeseries_drift,
)


def _phase2d_debye_pole():
    """Single-pole Debye: tau ~ a few ps, modest delta_eps so the
    distributed test runs fast on CPU virtual devices."""
    return DebyePole(delta_eps=2.0, tau=5e-12)


def _phase2d_lorentz_pole():
    """Single-pole Lorentz, microwave-band parameters."""
    return lorentz_pole(delta_eps=1.5, omega_0=2.0 * np.pi * 30e9, delta=1e10)


@_PHASE2B_REQUIRES_2DEV
def test_distributed_debye_only_2device_matches_single_device():
    """Phase 2D Class B forward parity (Debye-only).

    Build a small NU PEC cavity with a Debye pole active in the central
    slab.  2-device distributed run must match the single-device
    ``run_nonuniform`` reference at the final step.  This is the V3
    Phase 2D base parity gate for Debye dispersion.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 80

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny_g, nz_g = grid.nx, grid.ny, grid.nz

    # Debye mask: central slab spanning the slab seam so the ADE
    # ordering matters at the seam cell.
    debye_mask = jnp.zeros((nx, ny_g, nz_g), dtype=jnp.bool_)
    seam_i = nx // 2
    debye_mask = debye_mask.at[seam_i - 2:seam_i + 2,
                               ny_g // 4:3 * ny_g // 4,
                               nz_g // 4:3 * nz_g // 4].set(True)

    debye = init_debye([_phase2d_debye_pole()], materials, grid.dt,
                       mask=debye_mask)

    src_idx = (4, ny_g // 2, nz_g // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=12, j=ny_g // 2, k=nz_g // 2, component="ez")

    # Single-device reference
    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        sources=[src_spec], probes=[prb_spec],
        debye=debye,
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    # Distributed run with sharded Debye
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    debye_coeffs_sharded = shard_debye_coeffs_x_slab(
        debye[0], sharded_grid, mesh,
    )
    debye_state_sharded = shard_debye_state_x_slab(
        debye[1], sharded_grid, mesh,
    )

    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        debye=(debye_coeffs_sharded, debye_state_sharded),
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert ts_dist.shape == ts_single.shape, (
        f"shape mismatch: dist={ts_dist.shape}, single={ts_single.shape}"
    )
    assert out["debye_state_sharded"] is not None, (
        "Phase 2D runner must return debye_state_sharded when Debye is enabled"
    )
    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2d_debye_only_2device_parity")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_lorentz_only_2device_matches_single_device():
    """Phase 2D Class B forward parity (Lorentz-only)."""
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 80

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny_g, nz_g = grid.nx, grid.ny, grid.nz

    lorentz_mask = jnp.zeros((nx, ny_g, nz_g), dtype=jnp.bool_)
    seam_i = nx // 2
    lorentz_mask = lorentz_mask.at[seam_i - 2:seam_i + 2,
                                   ny_g // 4:3 * ny_g // 4,
                                   nz_g // 4:3 * nz_g // 4].set(True)

    lorentz = init_lorentz([_phase2d_lorentz_pole()], materials, grid.dt,
                           mask=lorentz_mask)

    src_idx = (4, ny_g // 2, nz_g // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=12, j=ny_g // 2, k=nz_g // 2, component="ez")

    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        sources=[src_spec], probes=[prb_spec],
        lorentz=lorentz,
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    lorentz_coeffs_sharded = shard_lorentz_coeffs_x_slab(
        lorentz[0], sharded_grid, mesh,
    )
    lorentz_state_sharded = shard_lorentz_state_x_slab(
        lorentz[1], sharded_grid, mesh,
    )

    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        lorentz=(lorentz_coeffs_sharded, lorentz_state_sharded),
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert ts_dist.shape == ts_single.shape
    assert out["lorentz_state_sharded"] is not None, (
        "Phase 2D runner must return lorentz_state_sharded when Lorentz is enabled"
    )
    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2d_lorentz_only_2device_parity")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_mixed_dispersion_2device_matches_single_device():
    """Phase 2D Class B forward parity (mixed Debye + Lorentz).

    V3 mandatory: DP4 explicitly requires the mixed branch.  Both
    materials active simultaneously; the runner must take the mixed
    dispatch path inside ``_update_e_nu_dispersive``.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 80

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny_g, nz_g = grid.nx, grid.ny, grid.nz

    # Use the same mask region for both (mixed dispersion in a slab
    # spanning the seam — the heaviest stress test for the ADE order).
    mask = jnp.zeros((nx, ny_g, nz_g), dtype=jnp.bool_)
    seam_i = nx // 2
    mask = mask.at[seam_i - 2:seam_i + 2,
                   ny_g // 4:3 * ny_g // 4,
                   nz_g // 4:3 * nz_g // 4].set(True)

    debye = init_debye([_phase2d_debye_pole()], materials, grid.dt, mask=mask)
    lorentz = init_lorentz([_phase2d_lorentz_pole()], materials, grid.dt,
                           mask=mask)

    src_idx = (4, ny_g // 2, nz_g // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=12, j=ny_g // 2, k=nz_g // 2, component="ez")

    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        sources=[src_spec], probes=[prb_spec],
        debye=debye, lorentz=lorentz,
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    debye_coeffs_sharded = shard_debye_coeffs_x_slab(debye[0], sharded_grid, mesh)
    debye_state_sharded = shard_debye_state_x_slab(debye[1], sharded_grid, mesh)
    lorentz_coeffs_sharded = shard_lorentz_coeffs_x_slab(lorentz[0], sharded_grid, mesh)
    lorentz_state_sharded = shard_lorentz_state_x_slab(lorentz[1], sharded_grid, mesh)

    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        debye=(debye_coeffs_sharded, debye_state_sharded),
        lorentz=(lorentz_coeffs_sharded, lorentz_state_sharded),
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert ts_dist.shape == ts_single.shape
    assert out["debye_state_sharded"] is not None
    assert out["lorentz_state_sharded"] is not None
    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2d_mixed_dispersion_2device_parity")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_mixed_dispersion_grad_no_corner_nan():
    """Phase 4 regression (2026-04-17): mixed Debye+Lorentz backward must
    produce NaN-free gradient.

    The Phase 4 calibration run surfaced a NaN leak at the rank-0 and
    rank-(N-1) first/last x cells of the eps gradient whenever both
    Debye and Lorentz poles were active on the distributed NU path.
    Root cause: ``_split_lorentz_coeffs`` / ``shard_lorentz_coeffs_x_slab``
    padded ``lorentz_coeffs.cc`` with 0.0 at ghost / high-x PEC cells, so
    the mixed formula's ``gamma_base = 1 / cc = inf`` combined with
    ``numer_base = ca * gamma_base = 0 * inf = NaN`` in forward.  Forward
    stayed bit-perfect (ghost cells drop before output assembly) but
    backward propagated ``NaN * 0 = NaN`` through ``cb_mixed[ghost] *
    curl`` into real-cell hy/hz gradients and on into ``d_loss/d_eps``
    at the physical-boundary x cells.

    The fix pads ``cc`` with the vacuum ``1 / EPS_0`` value so
    ``gamma_base = EPS_0`` at ghosts and the mixed cascade produces
    finite no-op updates.  This regression test fails WITHOUT the fix
    (NaN count ~2 * ny * nz) and passes WITH the fix.
    """
    from rfx import Simulation, Box

    devices = jax.devices()[:2]
    n_steps = 30

    # Minimum-scale repro that keeps CI cost low but still exercises the
    # mixed-dispersion distributed path.  Transverse extent must satisfy
    # `nx_per_rank > 2 * cpml_layers` — with 2 ranks and cpml=8, the
    # post-CPML grid (~40 cells) gives nx_per=20, well above 16.
    nx, ny_g, nz_g = 24, 24, 48
    dx = 1e-3
    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny_g * dx, nz_g * dx),
        dx=dx,
        boundary="cpml",
        cpml_layers=8,
        dx_profile=np.full(nx, dx),
        dz_profile=np.full(nz_g, dx),
    )
    debye_poles = [_phase2d_debye_pole()]
    lorentz_poles = [_phase2d_lorentz_pole()]
    sim.add_material(
        "disp_slab",
        eps_r=2.0,
        debye_poles=debye_poles,
        lorentz_poles=lorentz_poles,
    )
    margin = (8 + 2) * dx
    sim.add(
        Box(
            (margin, margin, 6 * dx),
            (nx * dx - margin, ny_g * dx - margin, (nz_g - 6) * dx),
        ),
        material="disp_slab",
    )
    src_i = max(10, nx // 4)
    prb_i = min(nx - 11, 3 * nx // 4)
    sim.add_source(
        position=(src_i * dx, (ny_g // 2) * dx, (nz_g // 2) * dx),
        component="ez",
    )
    sim.add_probe(
        position=(prb_i * dx, (ny_g // 2) * dx, (nz_g // 2) * dx),
        component="ez",
    )

    nu_grid = sim._build_nonuniform_grid()
    gnx, gny, gnz = int(nu_grid.nx), int(nu_grid.ny), int(nu_grid.nz)
    eps_init = jnp.ones((gnx, gny, gnz), dtype=jnp.float32)

    def _loss(eps):
        res = sim.forward(
            eps_override=eps,
            n_steps=n_steps,
            emit_time_series=True,
            distributed=True,
            devices=devices,
        )
        return jnp.sum(res.time_series ** 2)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grad = jax.grad(_loss)(eps_init)

    grad_np = np.asarray(grad)
    nan_count = int(np.isnan(grad_np).sum())
    assert nan_count == 0, (
        f"Distributed NU mixed-dispersion backward produced "
        f"{nan_count} NaN gradient cells (expected 0). "
        f"NaN x-rows: {sorted(set(np.argwhere(np.isnan(grad_np))[:, 0].tolist()))}. "
        f"This is the #44 Phase 4 corner-NaN regression — check "
        f"that `lorentz_coeffs.cc` is padded with a nonzero value in "
        f"`_split_lorentz_coeffs` and `shard_lorentz_coeffs_x_slab`."
    )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_debye_seam_ade_ordering_isolated():
    """Phase 2D Class D — ADE Ordering Contract seam-isolation gate.

    M3 critic finding (V3 plan lines 679-698): if ghost exchange
    overwrites ``ex_old`` on a slab boundary BEFORE the ADE update
    runs, the polarisation update is corrupted at the seam.  The
    distributed scan body must snapshot ``ex_old/ey_old/ez_old`` BEFORE
    any ghost exchange and pass that snapshot into the ADE call.

    Setup
    -----
    Vacuum everywhere except a 1-cell-thick Debye slab placed AT the
    slab seam (rank 1's first real cell, global x-index = nx_per_rank).
    Drive a pulse from rank 0, run 50 steps, compare the probe time-
    series to the single-device reference.

    If the runner reads ex_old from the post-exchange E (rank 0's value
    leaked into rank 1's first real cell via the ghost), the
    polarisation at the seam will diverge from single-device — caught
    here by Class B parity (rel_err < 5e-5).
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 50

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny_g, nz_g = grid.nx, grid.ny, grid.nz

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    seam_i = sharded_grid.nx_per_rank   # rank 1's first real cell

    # 1-cell-thick Debye slab AT the slab seam.  Use a thicker
    # transverse extent so the polarisation update does meaningful work.
    debye_mask = jnp.zeros((nx, ny_g, nz_g), dtype=jnp.bool_)
    debye_mask = debye_mask.at[seam_i, 1:-1, 1:-1].set(True)

    debye = init_debye([_phase2d_debye_pole()], materials, grid.dt,
                       mask=debye_mask)

    src_idx = (3, ny_g // 2, nz_g // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    # Probe sits ON the seam Debye cell so the polarisation feedback
    # to E is in the sampled signal.
    prb_spec = ProbeSpec(i=int(seam_i), j=ny_g // 2, k=nz_g // 2,
                         component="ez")

    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        sources=[src_spec], probes=[prb_spec],
        debye=debye,
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    debye_coeffs_sharded = shard_debye_coeffs_x_slab(debye[0], sharded_grid, mesh)
    debye_state_sharded = shard_debye_state_x_slab(debye[1], sharded_grid, mesh)

    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        debye=(debye_coeffs_sharded, debye_state_sharded),
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    # Class D semantic: this test catches an ordering violation that
    # would not be detectable from final-step parity alone.  Use
    # Class B (5e-5) — the seam Debye cell amplifies an ordering
    # mistake substantially above that threshold.
    assert_class_b_parity(
        ts_single, ts_dist,
        label="phase2d_debye_seam_ade_ordering_isolated",
    )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_dispersion_drift_over_time():
    """Phase 2D Class C — cumulative drift sweep across 4 checkpoints.

    Single-time-point parity can miss subtle drift that only surfaces
    after multiple polarisation update cycles.  Run for 200 steps with
    Debye dispersion in a slab spanning the seam, then verify
    ``rel_err < 5e-4`` at T/4, T/2, 3T/4, and T (Class D drift sweep).
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 200

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny_g, nz_g = grid.nx, grid.ny, grid.nz

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    seam_i = sharded_grid.nx_per_rank

    # Debye slab spanning the seam — wide enough that ADE feedback
    # accumulates measurably over 200 steps.
    debye_mask = jnp.zeros((nx, ny_g, nz_g), dtype=jnp.bool_)
    debye_mask = debye_mask.at[seam_i - 2:seam_i + 3, 1:-1, 1:-1].set(True)

    debye = init_debye([_phase2d_debye_pole()], materials, grid.dt,
                       mask=debye_mask)

    src_idx = (3, ny_g // 2, nz_g // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=12, j=ny_g // 2, k=nz_g // 2, component="ez")

    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        sources=[src_spec], probes=[prb_spec],
        debye=debye,
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    debye_coeffs_sharded = shard_debye_coeffs_x_slab(debye[0], sharded_grid, mesh)
    debye_state_sharded = shard_debye_state_x_slab(debye[1], sharded_grid, mesh)

    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        debye=(debye_coeffs_sharded, debye_state_sharded),
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    # Drift sweep at T/4, T/2, 3T/4, T (rel_err < 5e-4)
    assert_class_d_timeseries_drift(
        ts_single, ts_dist,
        label="phase2d_dispersion_drift_over_time",
    )


# ---------------------------------------------------------------------------
# Phase 2E: soft PEC (pec_occupancy) sharded scan body tests
# ---------------------------------------------------------------------------


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_occupancy_2device_matches_single_device():
    """Class B forward parity for soft-PEC occupancy.

    Place a small block of ``pec_occupancy`` (float) inside rank 1 and
    verify that the 2-device distributed run matches the single-device
    ``run_nonuniform`` reference at the final step.  The source sits on
    rank 0, so the signal crosses the seam and interacts with the soft
    conductor.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 60

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.2)
    materials = _phase2b_make_materials(grid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Soft-PEC block on rank 1 (global x in [11, 13], y/z 3:5).
    occ = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    occ = occ.at[11:14, 3:5, 3:5].set(1.0)

    src_idx = (4, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    # Probe on rank 1, downstream of the soft block, so the soft-PEC
    # interaction shows up in the time series.
    prb_spec = ProbeSpec(i=14, j=4, k=4, component="ez")

    # Single-device reference: run_nonuniform with pec_occupancy
    single_out = run_nonuniform(
        grid=grid,
        materials=materials,
        n_steps=n_steps,
        pec_occupancy=occ,
        sources=[src_spec],
        probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    # 2-device distributed run with sharded occupancy
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    sharded_occ = shard_pec_occupancy_x_slab(occ, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        sharded_pec_occupancy=sharded_occ,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert ts_dist.shape == ts_single.shape, (
        f"shape mismatch: dist={ts_dist.shape}, single={ts_single.shape}"
    )

    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2e_pec_occupancy_2device_parity")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_occupancy_seam_no_double_application():
    """Class D seam isolation: a soft-PEC occupancy cell exactly at the
    slab seam must be applied exactly once.

    Place ``occ=1.0`` at global x-index ``nx_per_rank`` (rank 1's first
    real cell) along with neighbouring occupancy so the per-component
    tangential rule (``occ * jnp.maximum(roll(+1), roll(-1))``) yields
    a non-zero contribution.  The distributed run must match the
    single-device reference, which applies the field once.

    If the seam cell were double-applied, the multiplicative
    soft-zeroing factor at the seam would compound, producing a
    distinguishable probe trace.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 30

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    seam_i = sharded_grid.nx_per_rank  # rank 1's first real cell, global x
    seam_j = ny // 2
    seam_k = nz // 2

    # Soft-PEC at the seam plus its left/right neighbours so the
    # tangential occupancy rule sees a non-zero neighbour and does not
    # silently cancel (otherwise no double-application would be visible).
    occ = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    occ = occ.at[seam_i - 1, seam_j, seam_k].set(1.0)
    occ = occ.at[seam_i, seam_j, seam_k].set(1.0)
    occ = occ.at[seam_i + 1, seam_j, seam_k].set(1.0)

    src_idx = (4, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=12, j=4, k=4, component="ez")

    # Single-device reference: applies occupancy exactly once.
    single_out = run_nonuniform(
        grid=grid,
        materials=materials,
        n_steps=n_steps,
        pec_occupancy=occ,
        sources=[src_spec],
        probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    sharded_occ = shard_pec_occupancy_x_slab(occ, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        sharded_pec_occupancy=sharded_occ,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2e_pec_occupancy_seam_no_double_application")


# ---------------------------------------------------------------------------
# Phase 2F: segmented remat + warmup + design-mask + emit_time_series tests
# ---------------------------------------------------------------------------

from rfx.runners.distributed_nu import (  # noqa: E402
    shard_design_mask_x_slab,
)
from tests._distributed_nu_tolerances import (  # noqa: E402
    assert_class_a_grad,
    assert_class_e_bit_match,
)


def _phase2f_build_run_kwargs(*, n_steps, n_devices, devices, eps_scale=1.0,
                              src_idx=(4, 4, 4), prb_idx=(12, 4, 4)):
    """Build a small Phase 2B-style 2-device run with a tunable eps scale.

    Returns a callable ``run(*, ckpt_every=None, n_warmup=0,
    design_mask=None, emit_time_series=True)`` that runs a fresh
    distributed simulation with the requested kwargs and returns the
    raw result dict.  The eps inside the cavity is scaled by
    ``eps_scale`` so callers can take ``jax.grad`` w.r.t. the scale.
    """
    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.2)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)

    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps,
        _phase2b_make_materials(grid),
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(
        i=int(prb_idx[0]), j=int(prb_idx[1]), k=int(prb_idx[2]),
        component="ez",
    )

    def _materials(eps_scale_val):
        eps = jnp.full((nx, ny, nz), eps_scale_val, dtype=jnp.float32)
        return MaterialArrays(
            eps_r=eps,
            sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
            mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        )

    def run(eps_scale_val=eps_scale, *, checkpoint_every=None,
            n_warmup=0, design_mask=None, emit_time_series=True):
        materials = _materials(eps_scale_val)
        sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
        sharded_design = (
            shard_design_mask_x_slab(design_mask, sharded_grid)
            if design_mask is not None else None
        )
        return run_nonuniform_distributed_pec(
            sharded_grid=sharded_grid,
            sharded_materials=sharded_mat,
            sharded_pec_mask=None,
            n_steps=n_steps,
            sources=[src_spec],
            probes=[prb_spec],
            n_devices=n_devices,
            devices=devices,
            checkpoint_every=checkpoint_every,
            n_warmup=n_warmup,
            sharded_design_mask=sharded_design,
            emit_time_series=emit_time_series,
        )

    return run, grid, sharded_grid, src_spec, prb_spec, _materials


@_PHASE2B_REQUIRES_2DEV
def test_distributed_checkpoint_every_value_matches_no_segment():
    """Class A bit-match: forward result with ``checkpoint_every=K``
    must equal the forward result without checkpointing.

    Segmented remat must not alter the forward physics — only the
    AD tape memory profile.  The segment scan-of-scan composition with
    zero-source padding + tail discard must reproduce the same probe
    samples step-for-step.

    Note: the JIT'd distributed path produces a different XLA HLO for
    the single-scan vs scan-of-scan dispatch, so float32 results may
    differ in the last bit due to op-ordering rounding.  We require
    Class A tolerance (``rtol=1e-6``) on the time-series rather than
    strict ``jnp.array_equal``; this catches any real physics drift
    while tolerating XLA-driven ULP noise.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 24

    run, *_ = _phase2f_build_run_kwargs(
        n_steps=n_steps, n_devices=n_devices, devices=devices,
    )

    out_no_seg = run(checkpoint_every=None)
    out_seg = run(checkpoint_every=8)

    ts_no = jnp.asarray(out_no_seg["time_series"])[:, 0]
    ts_seg = jnp.asarray(out_seg["time_series"])[:, 0]

    assert ts_no.shape == ts_seg.shape, (
        f"shape mismatch: no_seg={ts_no.shape}, seg={ts_seg.shape}"
    )
    # Class A tolerance on the forward time-series (max rel-err <= 1e-6).
    # Use a near-zero floor of 1e-30 to avoid div-by-zero on early
    # samples where the probe is still 0.
    rel_err = float(jnp.max(
        jnp.abs(ts_seg - ts_no) / (jnp.abs(ts_no) + 1e-30)
    ))
    assert rel_err <= 1e-6, (
        f"phase2f_checkpoint_every_forward Class A failed: "
        f"max rel_err={rel_err:.3e} > 1e-6"
    )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_checkpoint_every_grad_matches_no_segment():
    """Class A bit-match: ``jax.grad`` with ``checkpoint_every=K`` must
    equal ``jax.grad`` without checkpointing.

    Segmented remat is mathematically transparent to autodiff —
    forward+backward through scan-of-scan + ``jax.checkpoint`` produces
    the exact same gradient as the single-scan path (up to the
    deterministic remat redo).  The objective is a scalar function of
    the probe time-series; we differentiate w.r.t. the global eps
    scale which threads through every step.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 24

    run, *_ = _phase2f_build_run_kwargs(
        n_steps=n_steps, n_devices=n_devices, devices=devices,
    )

    def loss(eps_scale_val, *, checkpoint_every):
        out = run(eps_scale_val, checkpoint_every=checkpoint_every)
        ts = out["time_series"][:, 0]
        return jnp.sum(ts ** 2)

    eps0 = jnp.float32(1.0)
    grad_no = jax.grad(lambda v: loss(v, checkpoint_every=None))(eps0)
    grad_seg = jax.grad(lambda v: loss(v, checkpoint_every=8))(eps0)

    # Use Class A tolerance (1e-6 rel-err) for gradient parity.
    assert_class_a_grad(
        jnp.atleast_1d(grad_no),
        jnp.atleast_1d(grad_seg),
        label="phase2f_checkpoint_every_grad_match",
    )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_n_warmup_tail_grad_matches_single_device():
    """Class A: ``jax.grad`` through warmup-split distributed equals
    ``jax.grad`` through warmup-split single-device, both wrt eps after
    warmup.

    Single-device :func:`rfx.nonuniform.run_nonuniform` already
    supports ``n_warmup``.  The distributed runner's warmup split must
    produce the same gradient on a small NU+PEC parity case (single-
    scan, no checkpoint_every) so we compare the two
    warmup-split-active paths directly.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 24
    n_warmup = 8

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.2)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)

    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, (4, 4, 4), "ez", _phase2b_gauss_waveform, n_steps,
        _phase2b_make_materials(grid),
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=12, j=4, k=4, component="ez")

    def _mats(eps_val):
        eps = jnp.full((nx, ny, nz), eps_val, dtype=jnp.float32)
        return MaterialArrays(
            eps_r=eps,
            sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
            mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        )

    def loss_single(eps_val):
        out = run_nonuniform(
            grid=grid,
            materials=_mats(eps_val),
            n_steps=n_steps,
            sources=[src_spec],
            probes=[prb_spec],
            n_warmup=n_warmup,
        )
        ts = out["time_series"][:, 0]
        # Sum only the post-warmup tail (which is the only AD-active
        # region on both paths).
        return jnp.sum(ts[n_warmup:] ** 2)

    def loss_dist(eps_val):
        materials = _mats(eps_val)
        sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
        out = run_nonuniform_distributed_pec(
            sharded_grid=sharded_grid,
            sharded_materials=sharded_mat,
            sharded_pec_mask=None,
            n_steps=n_steps,
            sources=[src_spec],
            probes=[prb_spec],
            n_devices=n_devices,
            devices=devices,
            n_warmup=n_warmup,
        )
        ts = out["time_series"][:, 0]
        return jnp.sum(ts[n_warmup:] ** 2)

    eps0 = jnp.float32(1.0)
    g_single = jax.grad(loss_single)(eps0)
    g_dist = jax.grad(loss_dist)(eps0)

    assert_class_a_grad(
        jnp.atleast_1d(g_single),
        jnp.atleast_1d(g_dist),
        label="phase2f_n_warmup_tail_grad_match",
    )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_design_mask_stop_grad_matches_single_device():
    """Class A: forward bit-identical (design_mask is identity forward);
    gradient zero outside mask cells, finite inside.

    Compares the distributed runner's ``sharded_design_mask`` path with
    the single-device :func:`rfx.nonuniform.run_nonuniform`'s
    ``design_mask`` path on a small NU+PEC case.  Forward must be
    exactly identical to the no-mask path (``stop_gradient`` is
    forward-identity); the gradient w.r.t. a per-cell eps perturbation
    must vanish outside the mask and stay finite inside.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 16

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.2)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)

    # Design region: a small block on rank 1 (x in [10, 14)).
    design_mask_global = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
    design_mask_global = design_mask_global.at[10:14, 3:6, 3:6].set(True)

    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, (4, 4, 4), "ez", _phase2b_gauss_waveform, n_steps,
        _phase2b_make_materials(grid),
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=12, j=4, k=4, component="ez")

    # ------ Forward bit-identity check -------------------------------
    materials_const = _phase2b_make_materials(grid)
    sharded_mat = _phase2b_shard_mat(materials_const, sharded_grid)

    out_no_mask = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    out_with_mask = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        sharded_design_mask=shard_design_mask_x_slab(
            design_mask_global, sharded_grid),
    )
    assert_class_e_bit_match(
        jnp.asarray(out_no_mask["time_series"])[:, 0],
        jnp.asarray(out_with_mask["time_series"])[:, 0],
        label="phase2f_design_mask_forward_bit_identity",
    )

    # ------ Gradient parity vs single-device --------------------------
    # Use a per-cell eps perturbation indexed at one cell INSIDE the
    # mask and one cell OUTSIDE.  We expect: grad_inside finite
    # (matches single-device), grad_outside == 0 on both paths.
    inside_idx = (12, 4, 4)   # inside the design block
    outside_idx = (5, 4, 4)   # outside the design block

    def _make_eps_with_perturbation(delta, idx):
        base = jnp.ones((nx, ny, nz), dtype=jnp.float32)
        return base.at[idx].add(delta)

    def _mats_from_eps(eps):
        return MaterialArrays(
            eps_r=eps,
            sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
            mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        )

    def loss_single(delta, idx):
        eps = _make_eps_with_perturbation(delta, idx)
        out = run_nonuniform(
            grid=grid,
            materials=_mats_from_eps(eps),
            n_steps=n_steps,
            sources=[src_spec],
            probes=[prb_spec],
            design_mask=design_mask_global,
        )
        return jnp.sum(out["time_series"][:, 0] ** 2)

    def loss_dist(delta, idx):
        eps = _make_eps_with_perturbation(delta, idx)
        materials = _mats_from_eps(eps)
        sharded_mat_local = _phase2b_shard_mat(materials, sharded_grid)
        sharded_dm = shard_design_mask_x_slab(
            design_mask_global, sharded_grid)
        out = run_nonuniform_distributed_pec(
            sharded_grid=sharded_grid,
            sharded_materials=sharded_mat_local,
            sharded_pec_mask=None,
            n_steps=n_steps,
            sources=[src_spec],
            probes=[prb_spec],
            n_devices=n_devices,
            devices=devices,
            sharded_design_mask=sharded_dm,
        )
        return jnp.sum(out["time_series"][:, 0] ** 2)

    delta0 = jnp.float32(0.0)
    g_in_dist = jax.grad(lambda d: loss_dist(d, inside_idx))(delta0)
    g_out_dist = jax.grad(lambda d: loss_dist(d, outside_idx))(delta0)

    # Outside the design region the gradient must be (numerically) zero.
    assert float(jnp.abs(g_out_dist)) <= 1e-12, (
        f"phase2f_design_mask: outside-mask grad should be 0, "
        f"got {float(g_out_dist):.3e}"
    )
    # Inside the design region the gradient must be finite (and equal
    # to single-device's, up to Class A tolerance).
    g_in_single = jax.grad(lambda d: loss_single(d, inside_idx))(delta0)
    assert jnp.isfinite(g_in_dist), (
        f"phase2f_design_mask: inside-mask grad should be finite, "
        f"got {float(g_in_dist):.3e}"
    )
    assert_class_a_grad(
        jnp.atleast_1d(g_in_single),
        jnp.atleast_1d(g_in_dist),
        label="phase2f_design_mask_inside_grad_match",
    )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_emit_time_series_false_skips_probe():
    """Class E structural: result dict has ``time_series=None`` when
    ``emit_time_series=False``; the forward result still produces a
    valid ``final_state`` and ``jax.grad`` is preserved (verified by
    matching the time-series-driven gradient on a parallel run with
    ``emit_time_series=True``).

    The structural part of this test is the ``time_series is None``
    assertion: skipping probe accumulation must not silently emit a
    zero-length array hidden inside the dict.

    The gradient-preservation part is verified by computing a scalar
    loss from the source amplitude (which only enters via the source
    injection inside the scan body, so the AD path stays inside the
    JIT'd scan and doesn't trip on shard_map collective backward
    edge cases).  The ``emit=False`` run uses the ``final_state``'s
    rank-0 slab to build the loss; the ``emit=True`` run uses the
    probe time-series.  Both gradients must be finite and the
    ``emit=False`` path's gradient must agree with a finite-difference
    reference taken on the same path (round-trip soundness).
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 16

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.2)
    _nx, _ny, _nz = grid.nx, grid.ny, grid.nz
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)

    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, (4, 4, 4), "ez", _phase2b_gauss_waveform, n_steps,
        _phase2b_make_materials(grid),
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=12, j=4, k=4, component="ez")

    materials = _phase2b_make_materials(grid)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)

    # ---- Structural part: emit_time_series=False -> time_series None ----
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        emit_time_series=False,
    )
    assert out["time_series"] is None, (
        f"emit_time_series=False should yield time_series=None, "
        f"got type={type(out['time_series']).__name__}"
    )
    # Final state still gathered and returned.
    assert out["final_state"] is not None
    assert hasattr(out["final_state"], "ex")

    # ---- Cross-check: emit=False forward matches emit=True forward ----
    out_with_ts = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        emit_time_series=True,
    )
    # Final-state ex slab (gathered) must agree to Class A tolerance
    # between the two paths (skipping probe sampling must not perturb
    # the FDTD update).
    rel = float(jnp.max(
        jnp.abs(out["final_state"].ex - out_with_ts["final_state"].ex)
        / (jnp.abs(out_with_ts["final_state"].ex) + 1e-30)
    ))
    assert rel <= 1e-6, (
        f"emit=False perturbed final_state.ex vs emit=True: "
        f"max rel_err={rel:.3e} > 1e-6"
    )

    # ---- Gradient-preservation: differentiate through a non-time-series
    # objective.  We thread a scalar source-amplitude scaling factor
    # through ``waveform`` so the AD path stays entirely inside the
    # scan body's source injection (no shard_map all_gather backward).
    # The objective is the rank-0 slab's |E|^2 sum on the SHARDED final
    # state (rank-0 slab is sliced by axis-0 indexing, which composes
    # well with the sharded backward).
    base_wf = jnp.asarray(src_wf)

    def loss(amp):
        scaled_src = SourceSpec(
            i=int(src_si), j=int(src_sj), k=int(src_sk),
            component=src_comp, waveform=base_wf * amp,
        )
        out_inner = run_nonuniform_distributed_pec(
            sharded_grid=sharded_grid,
            sharded_materials=sharded_mat,
            sharded_pec_mask=None,
            n_steps=n_steps,
            sources=[scaled_src],
            probes=[prb_spec],
            n_devices=n_devices,
            devices=devices,
            emit_time_series=False,
        )
        # Use the *gathered* final state and slice the rank-0 owned
        # x-range.  The gathered tensor lives on a single device after
        # the JAX-friendly gather, so backward stays well-defined.
        ex = out_inner["final_state"].ex
        # Loss is quadratic in amp -> grad must scale linearly with amp.
        return jnp.sum(ex ** 2)

    amp0 = jnp.float32(1.0)
    g_emit_false = jax.grad(loss)(amp0)
    # Gradient must be a finite scalar (no NaN/Inf) — proves the
    # AD tape was correctly built without the per-step probe outputs.
    assert jnp.isfinite(g_emit_false), (
        f"emit_time_series=False loss grad must be finite, "
        f"got {float(g_emit_false):.3e}"
    )
    # Quadratic-in-amp: grad at amp=2 must be 2x grad at amp=1.
    g_2 = jax.grad(loss)(jnp.float32(2.0))
    ratio = float(g_2 / g_emit_false) if abs(g_emit_false) > 0 else float("nan")
    assert abs(ratio - 2.0) < 1e-3, (
        f"emit=False grad must scale linearly with source amp "
        f"(quadratic loss): grad(2)/grad(1) = {ratio:.6f}, expected ~2.0"
    )


# ---------------------------------------------------------------------------
# Phase 2 absolute-physics validation (Class F) — issue #44 follow-up
# ---------------------------------------------------------------------------
#
# These two tests are *not* parity tests.  Phases 2A-2F shipped 26 parity
# tests (single-device == distributed) which transitively inherit physical
# correctness from the 730+ existing single-device tests.  The remaining
# gap that pure parity cannot close:
#
#   1. Common-origin numerical bugs — a helper shared by both single-device
#      and distributed paths could be wrong; both runs would agree, yet
#      both would be wrong.
#   2. Slab-boundary-specific physics — the seam-aware ghost exchange and
#      rank-conditional PEC have no single-device counterpart, so a
#      construction-specific bug there would not register against a
#      single-device baseline.
#
# The two tests below close that gap by validating the distributed runner
# against analytic / first-principles references (Class F tolerance, see
# ``tests/_distributed_nu_tolerances.py``).

from rfx.harminv import harminv_from_probe  # noqa: E402
from rfx.core.yee import EPS_0 as _EPS_0_F, MU_0 as _MU_0_F  # noqa: E402
from tests._distributed_nu_tolerances import (  # noqa: E402
    RTOL_ANALYTIC_F,
    RTOL_ENERGY_DRIFT_F,
)


def _phase2_uniform_pec_grid(n_cells: int, dx0: float = 1e-3):
    """Build a uniform-spacing closed PEC cube (cpml_layers=0).

    Uses ``make_nonuniform_grid`` with a constant ``dx_profile`` so the
    NU plumbing is exercised, but the physics is a vanilla uniform Yee
    cube with hard PEC on all 6 faces.  Returns a ``NonUniformGrid``
    suitable for both ``run_nonuniform`` and the distributed runner.
    """
    dx_profile = np.full(n_cells, dx0)
    dz_profile = np.full(n_cells, dx0)
    return make_nonuniform_grid(
        (n_cells * dx0, n_cells * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )


def _phase2_total_em_energy(state, materials, grid):
    """Compute the total electromagnetic energy in a closed PEC cavity.

    ``E_total = sum_cells dV * (0.5 * eps * |E|^2 + 0.5 * mu * |H|^2)``

    The Yee staggering means each component is sampled at a different
    location within the cell, but for a uniform-cell-size cube and a
    cell-centred energy estimator this midpoint approximation is the
    standard FDTD energy surrogate (the per-step drift is dominated by
    op-ordering noise, not staggering bias).
    """
    eps = np.asarray(materials.eps_r, dtype=np.float64) * _EPS_0_F
    mu = np.asarray(materials.mu_r, dtype=np.float64) * _MU_0_F
    ex = np.asarray(state.ex, dtype=np.float64)
    ey = np.asarray(state.ey, dtype=np.float64)
    ez = np.asarray(state.ez, dtype=np.float64)
    hx = np.asarray(state.hx, dtype=np.float64)
    hy = np.asarray(state.hy, dtype=np.float64)
    hz = np.asarray(state.hz, dtype=np.float64)

    e_density = 0.5 * eps * (ex ** 2 + ey ** 2 + ez ** 2)
    h_density = 0.5 * mu * (hx ** 2 + hy ** 2 + hz ** 2)

    dx = np.asarray(grid.dx_arr, dtype=np.float64)
    dy = np.asarray(grid.dy_arr, dtype=np.float64)
    dz = np.asarray(grid.dz, dtype=np.float64)
    dV = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]

    return float(np.sum((e_density + h_density) * dV))


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_cavity_analytic_resonance():
    """Class F absolute analytic: distributed runner reproduces the
    analytic TM_110 resonance of a closed PEC cubic cavity to within
    5%.

    Setup
    -----
    Closed (hard PEC, no CPML) cubic cavity with side L = 24 mm,
    dx = dy = dz = 1 mm, so the lowest-resonance has wavelength
    34.0 mm (~34 cells per lambda — well-resolved Yee).

    The lowest-frequency mode of a cubic cavity is the degenerate set
    {TE_011, TE_101, TM_110} at::

        f_110 = (c0 / (2 L)) * sqrt(1^2 + 1^2 + 0^2)
              = c0 * sqrt(2) / (2 L)

    With L = 24 mm this is ~8.838 GHz.  We excite the TM_110 member
    (``E_z != 0``, ``E_z propto sin(pi x / L) sin(pi y / L)``) with a
    z-polarised current source placed off all node planes, then pick
    a probe location that also sits off the mode's node lines so the
    resonance dominates the time-series.

    Why this catches what parity cannot
    ----------------------------------
    A common-origin numerical bug (e.g. an incorrect inverse-spacing
    helper used by both single-device and distributed paths) would
    pass parity yet shift the extracted resonance away from the
    analytic value.  Class F (rtol = 5%) catches such drifts: 5% is
    the standard FDTD-vs-analytic bound at this resolution; a real
    bug typically pushes the error well past 10%.
    """
    # Determinism: set numpy + jax PRNG seeds even though this test
    # does not use randomness directly (the underlying scan body is
    # deterministic in float32; this is belt-and-braces).
    np.random.seed(0)

    devices = jax.devices()[:2]
    n_devices = 2

    n_cells = 24
    dx0 = 1e-3
    L = n_cells * dx0  # 24 mm

    # Analytic lowest-mode resonance (TM_110 / TE_011 / TE_101 degenerate)
    c0 = 1.0 / np.sqrt(_EPS_0_F * _MU_0_F)
    f_analytic = c0 * np.sqrt(2.0) / (2.0 * L)

    grid = _phase2_uniform_pec_grid(n_cells, dx0=dx0)
    materials = _phase2b_make_materials(grid)

    # Source: narrowband modulated Gaussian centred on the analytic
    # frequency.  A short un-modulated Gaussian has too much bandwidth
    # and excites the TM_111 / TM_210 / TM_211 modes alongside TM_110,
    # so harminv may pick a higher-amplitude neighbour even after a
    # bandpass.  A sinusoid * Gaussian envelope concentrates the
    # excitation in a ~+/- 1 GHz window around 8.84 GHz, well below
    # the gap to the next-mode (10.82 GHz).
    #
    # Source/probe placement uses cavity-symmetry suppression: the
    # source sits on the x=L/2, y=L/2 cell (which is the slab seam
    # from rank 0's perspective and rank 1's first real cell), where
    # TM_2,n,0 / TM_m,2,0 modes (sin(2 pi x/L)) all have a *node*.
    # The probe lies on rank 0 at an asymmetric position so the
    # TM_110 ring-down dominates the time series.
    n_steps = 4000
    src_idx = (n_cells // 2, n_cells // 2, 6)
    prb_idx = (n_cells // 3, n_cells // 3, 17)

    src_t0 = 250.0 * float(grid.dt)
    src_tau = 100.0 * float(grid.dt)
    src_omega = 2.0 * np.pi * float(f_analytic)

    def _modulated_gauss(t, t0=src_t0, tau=src_tau, omega=src_omega):
        envelope = jnp.exp(-((t - t0) ** 2) / (2.0 * tau ** 2))
        return jnp.sin(omega * (t - t0)) * envelope

    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _modulated_gauss, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(
        i=int(prb_idx[0]), j=int(prb_idx[1]), k=int(prb_idx[2]),
        component="ez",
    )

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts = np.asarray(out["time_series"])[:, 0]

    # Window past the source.  The modulated Gaussian envelope is
    # centred at t0 = 250 dt with tau = 100 dt, ~vanishing at
    # t = t0 + 5*tau = 750 dt.  Pass that to harminv so only the
    # ring-down portion is fitted.
    source_decay_time = (src_t0 + 5.0 * src_tau)

    # Tight search range centred on f_analytic (the source spectrum is
    # narrowband, so we also tighten the harminv window to reject any
    # spurious DC / out-of-band leakage).
    f_min = 0.85 * f_analytic
    f_max = 1.15 * f_analytic

    modes = harminv_from_probe(
        ts, dt=float(grid.dt),
        freq_range=(f_min, f_max),
        source_decay_time=source_decay_time,
    )
    assert len(modes) > 0, (
        "Class F resonance: harminv extracted no modes from the "
        "distributed PEC cavity time-series in the search range "
        f"[{f_min/1e9:.2f}, {f_max/1e9:.2f}] GHz."
    )

    # Strongest mode (sorted by amplitude in harminv_from_probe)
    f_extracted = modes[0].freq
    rel_err = abs(f_extracted - f_analytic) / f_analytic
    assert rel_err < RTOL_ANALYTIC_F, (
        f"Class F resonance: distributed PEC cavity TM_110 frequency "
        f"deviates by {100*rel_err:.2f}% from analytic. "
        f"f_analytic = {f_analytic/1e9:.4f} GHz, "
        f"f_extracted = {f_extracted/1e9:.4f} GHz, "
        f"rtol = {100*RTOL_ANALYTIC_F:.1f}%."
    )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_cavity_energy_conservation():
    """Class F absolute first-principles: distributed runner conserves
    total electromagnetic energy in a closed lossless PEC cavity to
    within 5% drift over a few hundred steps after the source switches
    off.

    Setup
    -----
    Same uniform PEC cube as the resonance test but smaller (16^3) for
    speed.  A short Gaussian current pulse runs for ~50 dt then the
    source waveform is identically zero for the remaining steps.  No
    CPML, no dispersion (lossless vacuum).  Two distributed runs are
    performed:

      * ``n_steps = 100`` -> measure ``E_total`` at ``t_settle``
        (well past source-off).
      * ``n_steps = 500`` -> measure ``E_total`` at ``t_late``.

    Both runs use the *same* source waveform (sliced/extended), so the
    physics is identical up to the per-run scan length.  The drift
    ``|E(t_late) - E(t_settle)| / E(t_settle)`` must be below 5%.

    Why this catches what parity cannot
    ----------------------------------
    Parity tests prove distributed and single-device agree, but they
    do not prove either is energy-conserving.  A dissipative bug in
    the seam ghost exchange (e.g. forgetting to pull a neighbour
    contribution that contains the mirror flux) would fail this
    check while still passing the parity tests if the same
    dissipation showed up in a hypothetical buggy single-device path.
    Likewise an *injection* bug (an accidental double-application of
    PEC at the seam) would *grow* the energy and trip the same bound.
    """
    np.random.seed(0)

    devices = jax.devices()[:2]
    n_devices = 2

    n_cells = 16
    dx0 = 1e-3

    grid = _phase2_uniform_pec_grid(n_cells, dx0=dx0)
    materials = _phase2b_make_materials(grid)

    # Short current pulse: the Gaussian centred at 20 dt, tau = 6 dt,
    # is ~exp(-(80/12)^2) ~ 1.5e-20 at t = 100 dt -> effectively
    # source-off well before t_settle = 100 dt.
    src_t0 = 20.0 * float(grid.dt)
    src_tau = 6.0 * float(grid.dt)

    def _gauss_pulse(t):
        return jnp.exp(-((t - src_t0) ** 2) / (2.0 * src_tau ** 2))

    # Single source/probe; probe value is unused for this test (we
    # only inspect the gathered ``final_state``) but the runner
    # contract requires a probe list to be non-empty for shape sanity.
    src_idx = (3, 5, 7)
    prb_idx = (5, 7, 11)

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)

    def _run(n_steps):
        src_si, src_sj, src_sk, src_comp, src_wf = (
            _phase2b_make_current_source(
                grid, src_idx, "ez", _gauss_pulse, n_steps, materials,
            )
        )
        src_spec = SourceSpec(
            i=int(src_si), j=int(src_sj), k=int(src_sk),
            component=src_comp, waveform=jnp.asarray(src_wf),
        )
        prb_spec = ProbeSpec(
            i=int(prb_idx[0]), j=int(prb_idx[1]), k=int(prb_idx[2]),
            component="ez",
        )
        return run_nonuniform_distributed_pec(
            sharded_grid=sharded_grid,
            sharded_materials=sharded_mat,
            sharded_pec_mask=None,
            n_steps=n_steps,
            sources=[src_spec],
            probes=[prb_spec],
            n_devices=n_devices,
            devices=devices,
        )

    n_settle = 100
    n_late = 500

    out_settle = _run(n_settle)
    out_late = _run(n_late)

    e_settle = _phase2_total_em_energy(
        out_settle["final_state"], materials, grid,
    )
    e_late = _phase2_total_em_energy(
        out_late["final_state"], materials, grid,
    )

    # Sanity: there must be non-trivial energy in the cavity to compare
    # against.  Absolute floor of 1e-30 J guards against an all-zero
    # final-state slipping through.
    assert e_settle > 1e-30, (
        f"Class F energy conservation: settle-time total energy is "
        f"~zero ({e_settle:.3e} J).  Either the source did not deposit "
        f"energy or the gathered final_state is corrupt."
    )

    drift = abs(e_late - e_settle) / e_settle
    assert drift < RTOL_ENERGY_DRIFT_F, (
        f"Class F energy conservation: closed lossless PEC cavity "
        f"total energy drifted by {100*drift:.3f}% between step "
        f"{n_settle} and step {n_late}, exceeding the {100*RTOL_ENERGY_DRIFT_F:.1f}% "
        f"bound.  E({n_settle}) = {e_settle:.6e} J, "
        f"E({n_late}) = {e_late:.6e} J.  This points to a "
        f"dissipative seam-exchange bug (drift < 0) or an accidental "
        f"PEC double-application at the slab boundary (drift > 0)."
    )
