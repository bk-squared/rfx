"""Multi-GPU distributed FDTD runner using jax.jit + shard_map.

Replaces the pmap-based distributed.py with the modern JAX sharding API:
- jax.sharding.Mesh + NamedSharding for explicit device placement
- jax.experimental.shard_map for ghost cell exchange and device-conditional ops
- jax.jit with in_shardings / out_shardings for the top-level compiled scan

Uses 1D slab decomposition along the x-axis with ghost cell exchange via
lax.ppermute inside shard_map.  Supports PEC and CPML boundaries, soft
sources, point probes, lumped ports, and dispersive materials
(Debye / Lorentz / Drude).

Single-device fallback: when only 1 device is available, sharding is skipped
and a plain jit path is used.

Known limitations (transparent single-device fallback):
- TFSF plane-wave sources: require full-domain field injection, not
  compatible with slab decomposition.  Detected and warned at runtime.
- Waveguide / Floquet ports: modal decomposition needs the full
  cross-section on one device.  Detected and warned at runtime.
- Non-divisible nx: automatically padded to nearest multiple of n_devices
  with PEC-filled cells, then trimmed after gathering.

The public entry point ``run_distributed`` has an identical signature to the
pmap version in ``distributed.py`` so callers need no changes.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
)
from rfx.simulation import (
    make_source,
    make_j_source,
    make_probe,
    make_port_source,
)
from rfx.sources.sources import LumpedPort, setup_lumped_port
from rfx.materials.debye import DebyeCoeffs, DebyeState
from rfx.materials.lorentz import LorentzCoeffs, LorentzState

# Re-export domain splitting helpers from the original module so existing
# callers that import them directly continue to work.
from rfx.runners.distributed import (
    gather_array_x,
    _split_state,
    _split_materials,
    _split_debye_coeffs,
    _split_debye_state,
    _split_lorentz_coeffs,
    _split_lorentz_state,
    _update_h_local,
    _update_e_local_with_dispersion,
    _init_cpml_distributed,
    _apply_cpml_e_distributed,
    _apply_cpml_h_distributed,
)


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------

def _make_mesh(devices):
    """Create a 1-D x-axis mesh from a list of JAX devices."""
    return Mesh(np.array(devices), axis_names=("x",))


def _x_sharding(mesh):
    """NamedSharding that distributes the leading (x) axis across devices."""
    return NamedSharding(mesh, P("x"))


def _rep_sharding(mesh):
    """NamedSharding for replicated (non-distributed) arrays."""
    return NamedSharding(mesh, P())


def _shard_field_state(state: FDTDState, mesh: Mesh) -> FDTDState:
    """Place all FDTDState arrays onto the mesh, sharding along x."""
    shd = _x_sharding(mesh)
    return FDTDState(
        ex=jax.device_put(state.ex, shd),
        ey=jax.device_put(state.ey, shd),
        ez=jax.device_put(state.ez, shd),
        hx=jax.device_put(state.hx, shd),
        hy=jax.device_put(state.hy, shd),
        hz=jax.device_put(state.hz, shd),
        step=jax.device_put(state.step, _rep_sharding(mesh)),
    )


def _shard_materials(materials: MaterialArrays, mesh: Mesh) -> MaterialArrays:
    shd = _x_sharding(mesh)
    return MaterialArrays(
        eps_r=jax.device_put(materials.eps_r, shd),
        sigma=jax.device_put(materials.sigma, shd),
        mu_r=jax.device_put(materials.mu_r, shd),
    )


# ---------------------------------------------------------------------------
# Ghost cell exchange via shard_map + ppermute
# ---------------------------------------------------------------------------

def _exchange_component_shmap(field, mesh, n_devices):
    """Exchange ghost cells for one field component using shard_map.

    ``field`` has global shape ``(nx_with_ghost*n_devices, ny, nz)`` when
    viewed from outside shard_map; inside each shard sees
    ``(nx_local_with_ghost, ny, nz)``.

    The convention matches the pmap version in distributed.py:
    - field[0]  = left ghost  <- left neighbour's rightmost real cell
    - field[-1] = right ghost <- right neighbour's leftmost real cell
    - real cells: field[1:-1]
    """

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=P("x"),
        out_specs=P("x"),
        check_rep=False,
    )
    def _exchange(f):
        right_boundary = f[-2:-1, :, :]   # last real cell -> right neighbour's left ghost
        left_boundary  = f[1:2, :, :]     # first real cell -> left neighbour's right ghost

        perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
        left_ghost_recv = lax.ppermute(right_boundary, "x", perm=perm_right)

        perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
        right_ghost_recv = lax.ppermute(left_boundary, "x", perm=perm_left)

        device_idx = lax.axis_index("x")

        left_ghost_val = jnp.where(device_idx > 0,
                                   left_ghost_recv,
                                   f[0:1, :, :])
        right_ghost_val = jnp.where(device_idx < n_devices - 1,
                                    right_ghost_recv,
                                    f[-1:, :, :])

        f = f.at[0:1, :, :].set(left_ghost_val)
        f = f.at[-1:, :, :].set(right_ghost_val)
        return f

    return _exchange(field)


def _exchange_h_ghosts_shmap(state: FDTDState, mesh: Mesh, n_devices: int) -> FDTDState:
    return state._replace(
        hx=_exchange_component_shmap(state.hx, mesh, n_devices),
        hy=_exchange_component_shmap(state.hy, mesh, n_devices),
        hz=_exchange_component_shmap(state.hz, mesh, n_devices),
    )


def _exchange_e_ghosts_shmap(state: FDTDState, mesh: Mesh, n_devices: int) -> FDTDState:
    return state._replace(
        ex=_exchange_component_shmap(state.ex, mesh, n_devices),
        ey=_exchange_component_shmap(state.ey, mesh, n_devices),
        ez=_exchange_component_shmap(state.ez, mesh, n_devices),
    )


# ---------------------------------------------------------------------------
# Device-conditional PEC inside shard_map
# ---------------------------------------------------------------------------

def _apply_pec_shmap(state: FDTDState, mesh: Mesh, n_devices: int,
                     nx_local_with_ghost: int) -> FDTDState:
    """Apply PEC boundary conditions using shard_map for device identity."""

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("x"),  # ex
            P("x"),  # ey
            P("x"),  # ez
        ),
        out_specs=(
            P("x"),
            P("x"),
            P("x"),
        ),
        check_rep=False,
    )
    def _pec(ex, ey, ez):
        ghost = 1

        # Y-axis PEC (all devices)
        ex = ex.at[:, 0, :].set(0.0)
        ex = ex.at[:, -1, :].set(0.0)
        ez = ez.at[:, 0, :].set(0.0)
        ez = ez.at[:, -1, :].set(0.0)

        # Z-axis PEC (all devices)
        ex = ex.at[:, :, 0].set(0.0)
        ex = ex.at[:, :, -1].set(0.0)
        ey = ey.at[:, :, 0].set(0.0)
        ey = ey.at[:, :, -1].set(0.0)

        device_idx = lax.axis_index("x")

        # X-lo PEC: device 0 only
        is_first = (device_idx == 0)
        ey_xlo = jnp.where(is_first, 0.0, ey[ghost, :, :])
        ez_xlo = jnp.where(is_first, 0.0, ez[ghost, :, :])
        ey = ey.at[ghost, :, :].set(ey_xlo)
        ez = ez.at[ghost, :, :].set(ez_xlo)

        # X-hi PEC: device N-1 only
        is_last = (device_idx == n_devices - 1)
        last_real = nx_local_with_ghost - 1 - ghost
        ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
        ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
        ey = ey.at[last_real, :, :].set(ey_xhi)
        ez = ez.at[last_real, :, :].set(ez_xhi)

        return ex, ey, ez

    ex, ey, ez = _pec(state.ex, state.ey, state.ez)
    return state._replace(ex=ex, ey=ey, ez=ez)


# ---------------------------------------------------------------------------
# CPML inside shard_map
# ---------------------------------------------------------------------------

def _apply_cpml_e_shmap(state, cpml_params, cpml_state, n_cpml, dt, dx,
                         mesh, n_devices, ghost=1):
    """Apply CPML E-field correction using shard_map."""

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("x"),   # ex
            P("x"),   # ey
            P("x"),   # ez
            P("x"),   # hx
            P("x"),   # hy
            P("x"),   # hz
            # cpml psi arrays – all x-sharded
            P("x"), P("x"), P("x"), P("x"),  # ey_xlo/xhi, ez_xlo/xhi
            P("x"), P("x"), P("x"), P("x"),  # ex_ylo/yhi, ez_ylo/yhi
            P("x"), P("x"), P("x"), P("x"),  # ex_zlo/zhi, ey_zlo/zhi
        ),
        out_specs=(
            P("x"), P("x"), P("x"),           # ex, ey, ez
            P("x"), P("x"), P("x"), P("x"),   # psi ey/ez x-faces
            P("x"), P("x"), P("x"), P("x"),   # psi ex/ez y-faces
            P("x"), P("x"), P("x"), P("x"),   # psi ex/ey z-faces
        ),
        check_rep=False,
    )
    def _cpml_e(ex, ey, ez, hx, hy, hz,
                psi_ey_xlo, psi_ey_xhi, psi_ez_xlo, psi_ez_xhi,
                psi_ex_ylo, psi_ex_yhi, psi_ez_ylo, psi_ez_yhi,
                psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi):
        # Reconstruct minimal state and cpml_state objects
        from rfx.core.yee import FDTDState as _FS
        _st = _FS(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
        _cs = cpml_state._replace(
            psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
            psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
            psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
            psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
            psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
            psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
        )
        new_st, new_cs = _apply_cpml_e_distributed(
            _st, cpml_params, _cs, n_cpml, dt, dx,
            n_devices, ghost=ghost, axis_name="x")
        return (
            new_st.ex, new_st.ey, new_st.ez,
            new_cs.psi_ey_xlo, new_cs.psi_ey_xhi,
            new_cs.psi_ez_xlo, new_cs.psi_ez_xhi,
            new_cs.psi_ex_ylo, new_cs.psi_ex_yhi,
            new_cs.psi_ez_ylo, new_cs.psi_ez_yhi,
            new_cs.psi_ex_zlo, new_cs.psi_ex_zhi,
            new_cs.psi_ey_zlo, new_cs.psi_ey_zhi,
        )

    (ex, ey, ez,
     psi_ey_xlo, psi_ey_xhi, psi_ez_xlo, psi_ez_xhi,
     psi_ex_ylo, psi_ex_yhi, psi_ez_ylo, psi_ez_yhi,
     psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi) = _cpml_e(
        state.ex, state.ey, state.ez,
        state.hx, state.hy, state.hz,
        cpml_state.psi_ey_xlo, cpml_state.psi_ey_xhi,
        cpml_state.psi_ez_xlo, cpml_state.psi_ez_xhi,
        cpml_state.psi_ex_ylo, cpml_state.psi_ex_yhi,
        cpml_state.psi_ez_ylo, cpml_state.psi_ez_yhi,
        cpml_state.psi_ex_zlo, cpml_state.psi_ex_zhi,
        cpml_state.psi_ey_zlo, cpml_state.psi_ey_zhi,
    )
    new_state = state._replace(ex=ex, ey=ey, ez=ez)
    new_cpml = cpml_state._replace(
        psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
        psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
        psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
        psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
        psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
        psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
    )
    return new_state, new_cpml


def _apply_cpml_h_shmap(state, cpml_params, cpml_state, n_cpml, dt, dx,
                         mesh, n_devices, ghost=1):
    """Apply CPML H-field correction using shard_map."""

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("x"),   # ex
            P("x"),   # ey
            P("x"),   # ez
            P("x"),   # hx
            P("x"),   # hy
            P("x"),   # hz
            P("x"), P("x"), P("x"), P("x"),  # hy_xlo/xhi, hz_xlo/xhi
            P("x"), P("x"), P("x"), P("x"),  # hx_ylo/yhi, hz_ylo/yhi
            P("x"), P("x"), P("x"), P("x"),  # hx_zlo/zhi, hy_zlo/zhi
        ),
        out_specs=(
            P("x"), P("x"), P("x"),           # hx, hy, hz
            P("x"), P("x"), P("x"), P("x"),   # psi hy/hz x-faces
            P("x"), P("x"), P("x"), P("x"),   # psi hx/hz y-faces
            P("x"), P("x"), P("x"), P("x"),   # psi hx/hy z-faces
        ),
        check_rep=False,
    )
    def _cpml_h(ex, ey, ez, hx, hy, hz,
                psi_hy_xlo, psi_hy_xhi, psi_hz_xlo, psi_hz_xhi,
                psi_hx_ylo, psi_hx_yhi, psi_hz_ylo, psi_hz_yhi,
                psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi):
        from rfx.core.yee import FDTDState as _FS
        _st = _FS(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
        _cs = cpml_state._replace(
            psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
            psi_hz_xlo=psi_hz_xlo, psi_hz_xhi=psi_hz_xhi,
            psi_hx_ylo=psi_hx_ylo, psi_hx_yhi=psi_hx_yhi,
            psi_hz_ylo=psi_hz_ylo, psi_hz_yhi=psi_hz_yhi,
            psi_hx_zlo=psi_hx_zlo, psi_hx_zhi=psi_hx_zhi,
            psi_hy_zlo=psi_hy_zlo, psi_hy_zhi=psi_hy_zhi,
        )
        new_st, new_cs = _apply_cpml_h_distributed(
            _st, cpml_params, _cs, n_cpml, dt, dx,
            n_devices, ghost=ghost, axis_name="x")
        return (
            new_st.hx, new_st.hy, new_st.hz,
            new_cs.psi_hy_xlo, new_cs.psi_hy_xhi,
            new_cs.psi_hz_xlo, new_cs.psi_hz_xhi,
            new_cs.psi_hx_ylo, new_cs.psi_hx_yhi,
            new_cs.psi_hz_ylo, new_cs.psi_hz_yhi,
            new_cs.psi_hx_zlo, new_cs.psi_hx_zhi,
            new_cs.psi_hy_zlo, new_cs.psi_hy_zhi,
        )

    (hx, hy, hz,
     psi_hy_xlo, psi_hy_xhi, psi_hz_xlo, psi_hz_xhi,
     psi_hx_ylo, psi_hx_yhi, psi_hz_ylo, psi_hz_yhi,
     psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi) = _cpml_h(
        state.ex, state.ey, state.ez,
        state.hx, state.hy, state.hz,
        cpml_state.psi_hy_xlo, cpml_state.psi_hy_xhi,
        cpml_state.psi_hz_xlo, cpml_state.psi_hz_xhi,
        cpml_state.psi_hx_ylo, cpml_state.psi_hx_yhi,
        cpml_state.psi_hz_ylo, cpml_state.psi_hz_yhi,
        cpml_state.psi_hx_zlo, cpml_state.psi_hx_zhi,
        cpml_state.psi_hy_zlo, cpml_state.psi_hy_zhi,
    )
    new_state = state._replace(hx=hx, hy=hy, hz=hz)
    new_cpml = cpml_state._replace(
        psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
        psi_hz_xlo=psi_hz_xlo, psi_hz_xhi=psi_hz_xhi,
        psi_hx_ylo=psi_hx_ylo, psi_hx_yhi=psi_hx_yhi,
        psi_hz_ylo=psi_hz_ylo, psi_hz_yhi=psi_hz_yhi,
        psi_hx_zlo=psi_hx_zlo, psi_hx_zhi=psi_hx_zhi,
        psi_hy_zlo=psi_hy_zlo, psi_hy_zhi=psi_hy_zhi,
    )
    return new_state, new_cpml


# ---------------------------------------------------------------------------
# Sharded array initialisation helpers for CPML state
# ---------------------------------------------------------------------------

def _init_cpml_sharded(grid, nx_local, n_devices, mesh):
    """Initialize CPML params and per-slab state sharded along x.

    Returns ``(cpml_params, cpml_state)`` where all psi arrays in
    ``cpml_state`` have been placed on the mesh with x-sharding.
    Their global leading dim is ``n_devices * n_cpml`` (or ``n_devices *
    nx_local`` for y/z-indexed faces) but each shard sees the per-device
    slice.
    """
    cpml_params, cpml_state_stacked = _init_cpml_distributed(
        grid, nx_local, n_devices)
    shd = _x_sharding(mesh)

    def _shard_psi(arr):
        # arr: (n_devices, n_cpml, d1, d2) — merge device+cpml dims then shard
        # Re-interpret as (n_devices * n_cpml, d1, d2) so x-sharding distributes
        # the first axis across devices correctly.  Each device owns n_cpml rows.
        n_dev, n_c, d1, d2 = arr.shape
        merged = arr.reshape(n_dev * n_c, d1, d2)
        return jax.device_put(merged, shd)

    def _shard_psi_field(arr):
        # arr: (n_devices, n_cpml, d1, d2) same as above
        return _shard_psi(arr)

    # Build sharded CPMLState
    from rfx.boundaries.cpml import CPMLState
    cpml_state_sharded = CPMLState(
        psi_ex_ylo=_shard_psi(cpml_state_stacked.psi_ex_ylo),
        psi_ex_yhi=_shard_psi(cpml_state_stacked.psi_ex_yhi),
        psi_ex_zlo=_shard_psi(cpml_state_stacked.psi_ex_zlo),
        psi_ex_zhi=_shard_psi(cpml_state_stacked.psi_ex_zhi),
        psi_ey_xlo=_shard_psi(cpml_state_stacked.psi_ey_xlo),
        psi_ey_xhi=_shard_psi(cpml_state_stacked.psi_ey_xhi),
        psi_ey_zlo=_shard_psi(cpml_state_stacked.psi_ey_zlo),
        psi_ey_zhi=_shard_psi(cpml_state_stacked.psi_ey_zhi),
        psi_ez_xlo=_shard_psi(cpml_state_stacked.psi_ez_xlo),
        psi_ez_xhi=_shard_psi(cpml_state_stacked.psi_ez_xhi),
        psi_ez_ylo=_shard_psi(cpml_state_stacked.psi_ez_ylo),
        psi_ez_yhi=_shard_psi(cpml_state_stacked.psi_ez_yhi),
        psi_hx_ylo=_shard_psi(cpml_state_stacked.psi_hx_ylo),
        psi_hx_yhi=_shard_psi(cpml_state_stacked.psi_hx_yhi),
        psi_hx_zlo=_shard_psi(cpml_state_stacked.psi_hx_zlo),
        psi_hx_zhi=_shard_psi(cpml_state_stacked.psi_hx_zhi),
        psi_hy_xlo=_shard_psi(cpml_state_stacked.psi_hy_xlo),
        psi_hy_xhi=_shard_psi(cpml_state_stacked.psi_hy_xhi),
        psi_hy_zlo=_shard_psi(cpml_state_stacked.psi_hy_zlo),
        psi_hy_zhi=_shard_psi(cpml_state_stacked.psi_hy_zhi),
        psi_hz_xlo=_shard_psi(cpml_state_stacked.psi_hz_xlo),
        psi_hz_xhi=_shard_psi(cpml_state_stacked.psi_hz_xhi),
        psi_hz_ylo=_shard_psi(cpml_state_stacked.psi_hz_ylo),
        psi_hz_yhi=_shard_psi(cpml_state_stacked.psi_hz_yhi),
    )
    return cpml_params, cpml_state_sharded


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_distributed(sim, *, n_steps, devices=None, exchange_interval=1,
                    **kwargs):
    """Run FDTD simulation distributed across multiple devices.

    Uses 1D slab decomposition along the x-axis.  Supports PEC and
    CPML boundaries, soft sources, point probes, lumped ports, and
    Debye/Lorentz dispersive materials.

    TFSF plane-wave sources and waveguide ports are not yet supported
    in the distributed runner.  When detected, the runner issues a
    warning and transparently falls back to the single-device path.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance.
    n_steps : int
        Number of timesteps.
    devices : list of jax.Device or None
        If None, use all available devices.
    exchange_interval : int, optional
        How often (in timesteps) to perform ghost cell exchange.
        Default is 1 (every step).  Setting to 2 or 4 reduces
        synchronisation overhead at the cost of O(interval * dt)
        boundary error from stale ghost data.

    Returns
    -------
    Result
    """
    import warnings

    if exchange_interval > 1:
        warnings.warn(
            f"exchange_interval={exchange_interval}: ghost cells are stale "
            f"for {exchange_interval-1} steps between exchanges, introducing "
            f"O(dt*{exchange_interval}) boundary error. Use exchange_interval=1 "
            f"for physically accurate results.",
            stacklevel=2,
        )

    if sim._boundary == "upml":
        raise ValueError("boundary='upml' does not support distributed execution")

    # ------------------------------------------------------------------
    # Graceful fallback for features that require the full domain on a
    # single device.
    # ------------------------------------------------------------------
    if sim._tfsf is not None:
        warnings.warn(
            "Distributed runner does not yet support TFSF plane-wave "
            "sources. Falling back to single-device execution.",
            stacklevel=2,
        )
        return sim.run(n_steps=n_steps)

    if sim._waveguide_ports:
        warnings.warn(
            "Distributed runner does not yet support waveguide ports. "
            "Falling back to single-device execution.",
            stacklevel=2,
        )
        return sim.run(n_steps=n_steps)

    from rfx.api import Result

    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)

    # ------------------------------------------------------------------
    # Single-device fast path: skip all sharding overhead.
    # ------------------------------------------------------------------
    if n_devices == 1:
        from rfx.runners.distributed import run_distributed as _pmap_run
        return _pmap_run(sim, n_steps=n_steps, devices=devices,
                         exchange_interval=exchange_interval, **kwargs)

    # ------------------------------------------------------------------
    # Build grid and materials (full domain)
    # ------------------------------------------------------------------
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, _ = (
        sim._assemble_materials(grid)
    )
    materials = base_materials

    nx, ny, nz = grid.shape
    # Pad nx to nearest multiple of n_devices (PEC-filled padding cells)
    pad_x = 0
    if nx % n_devices != 0:
        pad_x = n_devices - (nx % n_devices)
    nx_padded = nx + pad_x

    if pad_x > 0:
        _pad_x = ((0, pad_x), (0, 0), (0, 0))
        materials = MaterialArrays(
            eps_r=jnp.pad(materials.eps_r, _pad_x, constant_values=1.0),
            sigma=jnp.pad(materials.sigma, _pad_x, constant_values=0.0),
            mu_r=jnp.pad(materials.mu_r, _pad_x, constant_values=1.0),
        )
        if pec_mask is not None:
            pec_mask = jnp.pad(pec_mask, _pad_x, constant_values=True)
        if debye_spec is not None:
            d_poles, d_masks = debye_spec
            d_masks = [jnp.pad(m, _pad_x, constant_values=False) for m in d_masks]
            debye_spec = (d_poles, d_masks)
        if lorentz_spec is not None:
            l_poles, l_masks = lorentz_spec
            l_masks = [jnp.pad(m, _pad_x, constant_values=False) for m in l_masks]
            lorentz_spec = (l_poles, l_masks)

    nx_per = nx_padded // n_devices
    ghost = 1
    nx_local = nx_per + 2 * ghost

    use_cpml = sim._boundary == "cpml" and grid.cpml_layers > 0
    n_cpml = grid.cpml_layers if use_cpml else 0

    dt = grid.dt
    dx = grid.dx

    # ------------------------------------------------------------------
    # Create mesh
    # ------------------------------------------------------------------
    mesh = _make_mesh(devices)

    # ------------------------------------------------------------------
    # Build sources and probes
    # ------------------------------------------------------------------
    sources = []
    probes = []
    for pe in sim._ports:
        if pe.impedance > 0.0 and pe.extent is None:
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            materials = setup_lumped_port(grid, lp, materials)
            sources.append(make_port_source(grid, lp, materials, n_steps))
        elif pe.impedance == 0.0:
            if sim._boundary == "cpml":
                sources.append(make_j_source(grid, pe.position, pe.component,
                                             pe.waveform, n_steps, materials))
            else:
                sources.append(make_source(grid, pe.position, pe.component,
                                           pe.waveform, n_steps))
    for pe in sim._probes:
        probes.append(make_probe(grid, pe.position, pe.component))

    # Map source/probe global indices to (device_id, local_index)
    src_device_ids = []
    src_local_specs = []
    for s in sources:
        dev_id = s.i // nx_per
        local_i = (s.i % nx_per) + ghost
        src_device_ids.append(dev_id)
        src_local_specs.append((local_i, s.j, s.k, s.component))

    prb_device_ids = []
    prb_local_specs = []
    for p in probes:
        dev_id = p.i // nx_per
        local_i = (p.i % nx_per) + ghost
        prb_device_ids.append(dev_id)
        prb_local_specs.append((local_i, p.j, p.k, p.component))

    # Precompute source waveforms: (n_steps, n_sources)
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Initialise state and split into per-device slabs with ghost cells,
    # then shard along x.
    # ------------------------------------------------------------------
    from rfx.core.yee import init_state
    full_state = init_state((nx_padded, ny, nz))
    state_slabs = _split_state(full_state, n_devices, ghost)
    materials_slabs = _split_materials(materials, n_devices, ghost)

    # Shard the stacked slabs: shape (n_devices, nx_local, ny, nz) ->
    # each device owns nx_local rows of the sharded (n_devices*nx_local, ny, nz) array.
    shd = _x_sharding(mesh)
    rep = _rep_sharding(mesh)

    def _shard_stacked(arr):
        """Merge device axis into x, then shard."""
        n_dev = arr.shape[0]
        rest = arr.shape[1:]
        return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)

    def _shard_stacked_5d(arr):
        """(n_devices, n_poles, nx_local, ny, nz) -> shard along device dim.

        Merges (n_dev, n_poles) into first axis so P("x") gives each
        device (n_poles, nx_local, ny, nz) — matching the layout expected
        by _update_e_debye_local / _update_e_lorentz_local.
        """
        n_dev, n_poles, nx_loc, ny_a, nz_a = arr.shape
        merged = arr.reshape(n_dev * n_poles, nx_loc, ny_a, nz_a)
        return jax.device_put(merged, shd)

    state_ex = _shard_stacked(state_slabs.ex)
    state_ey = _shard_stacked(state_slabs.ey)
    state_ez = _shard_stacked(state_slabs.ez)
    state_hx = _shard_stacked(state_slabs.hx)
    state_hy = _shard_stacked(state_slabs.hy)
    state_hz = _shard_stacked(state_slabs.hz)
    state_step = jax.device_put(jnp.int32(0), rep)

    sharded_state = FDTDState(
        ex=state_ex, ey=state_ey, ez=state_ez,
        hx=state_hx, hy=state_hy, hz=state_hz,
        step=state_step,
    )

    mat_eps_r = _shard_stacked(materials_slabs.eps_r)
    mat_sigma = _shard_stacked(materials_slabs.sigma)
    mat_mu_r  = _shard_stacked(materials_slabs.mu_r)
    sharded_materials = MaterialArrays(
        eps_r=mat_eps_r, sigma=mat_sigma, mu_r=mat_mu_r)

    # ------------------------------------------------------------------
    # Dispersive materials
    # ------------------------------------------------------------------
    _, debye_full, lorentz_full = sim._init_dispersion(
        materials, grid.dt, debye_spec, lorentz_spec)

    has_debye = debye_full is not None
    has_lorentz = lorentz_full is not None

    if has_debye:
        debye_coeffs_full, debye_state_full = debye_full
        debye_coeffs_slabs = _split_debye_coeffs(debye_coeffs_full, n_devices, ghost)
        debye_state_slabs = _split_debye_state(debye_state_full, n_devices, ghost)

        # Shard coefficients
        debye_coeffs_sharded = DebyeCoeffs(
            ca=_shard_stacked(debye_coeffs_slabs.ca),
            cb=_shard_stacked(debye_coeffs_slabs.cb),
            cc=_shard_stacked_5d(debye_coeffs_slabs.cc),
            alpha=_shard_stacked_5d(debye_coeffs_slabs.alpha),
            beta=_shard_stacked_5d(debye_coeffs_slabs.beta),
        )
        debye_state_sharded = DebyeState(
            px=_shard_stacked_5d(debye_state_slabs.px),
            py=_shard_stacked_5d(debye_state_slabs.py),
            pz=_shard_stacked_5d(debye_state_slabs.pz),
        )
    else:
        _total_x = n_devices * nx_local
        _dz = jnp.zeros((_total_x, ny, nz), dtype=jnp.float32)
        # 5D dummy: (n_dev * 1_pole, nx_local, ny, nz) so P("x") gives
        # each device (1, nx_local, ny, nz) matching (n_poles, nx_local, ny, nz) layout
        _dz5 = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.float32)
        debye_coeffs_sharded = DebyeCoeffs(
            ca=jax.device_put(_dz, shd),
            cb=jax.device_put(_dz, shd),
            cc=jax.device_put(_dz5, shd),
            alpha=jax.device_put(_dz5, shd),
            beta=jax.device_put(_dz5, shd),
        )
        debye_state_sharded = DebyeState(
            px=jax.device_put(_dz5, shd),
            py=jax.device_put(_dz5, shd),
            pz=jax.device_put(_dz5, shd),
        )

    if has_lorentz:
        lorentz_coeffs_full, lorentz_state_full = lorentz_full
        lorentz_coeffs_slabs = _split_lorentz_coeffs(lorentz_coeffs_full, n_devices, ghost)
        lorentz_state_slabs = _split_lorentz_state(lorentz_state_full, n_devices, ghost)

        lorentz_coeffs_sharded = LorentzCoeffs(
            ca=_shard_stacked(lorentz_coeffs_slabs.ca),
            cb=_shard_stacked(lorentz_coeffs_slabs.cb),
            cc=_shard_stacked(lorentz_coeffs_slabs.cc),
            a=_shard_stacked_5d(lorentz_coeffs_slabs.a),
            b=_shard_stacked_5d(lorentz_coeffs_slabs.b),
            c=_shard_stacked_5d(lorentz_coeffs_slabs.c),
        )
        lorentz_state_sharded = LorentzState(
            px=_shard_stacked_5d(lorentz_state_slabs.px),
            py=_shard_stacked_5d(lorentz_state_slabs.py),
            pz=_shard_stacked_5d(lorentz_state_slabs.pz),
            px_prev=_shard_stacked_5d(lorentz_state_slabs.px_prev),
            py_prev=_shard_stacked_5d(lorentz_state_slabs.py_prev),
            pz_prev=_shard_stacked_5d(lorentz_state_slabs.pz_prev),
        )
    else:
        _total_x = n_devices * nx_local
        _lz = jnp.zeros((_total_x, ny, nz), dtype=jnp.float32)
        # 5D dummy: (n_dev * 1_pole, nx_local, ny, nz)
        _lz5 = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.float32)
        lorentz_coeffs_sharded = LorentzCoeffs(
            ca=jax.device_put(_lz, shd),
            cb=jax.device_put(_lz, shd),
            cc=jax.device_put(_lz, shd),
            a=jax.device_put(_lz5, shd),
            b=jax.device_put(_lz5, shd),
            c=jax.device_put(_lz5, shd),
        )
        lorentz_state_sharded = LorentzState(
            px=jax.device_put(_lz5, shd),
            py=jax.device_put(_lz5, shd),
            pz=jax.device_put(_lz5, shd),
            px_prev=jax.device_put(_lz5, shd),
            py_prev=jax.device_put(_lz5, shd),
            pz_prev=jax.device_put(_lz5, shd),
        )

    # ------------------------------------------------------------------
    # CPML state
    # ------------------------------------------------------------------
    if use_cpml:
        cpml_params, cpml_state_sharded = _init_cpml_sharded(
            grid, nx_local, n_devices, mesh)
    else:
        from rfx.boundaries.cpml import CPMLState
        _total_x = n_devices * nx_local
        _z = jax.device_put(
            jnp.zeros((_total_x, 1, 1), dtype=jnp.float32), shd)
        cpml_state_sharded = CPMLState(
            psi_ex_ylo=_z, psi_ex_yhi=_z,
            psi_ex_zlo=_z, psi_ex_zhi=_z,
            psi_ey_xlo=_z, psi_ey_xhi=_z,
            psi_ey_zlo=_z, psi_ey_zhi=_z,
            psi_ez_xlo=_z, psi_ez_xhi=_z,
            psi_ez_ylo=_z, psi_ez_yhi=_z,
            psi_hx_ylo=_z, psi_hx_yhi=_z,
            psi_hx_zlo=_z, psi_hx_zhi=_z,
            psi_hy_xlo=_z, psi_hy_xhi=_z,
            psi_hy_zlo=_z, psi_hy_zhi=_z,
            psi_hz_xlo=_z, psi_hz_xhi=_z,
            psi_hz_ylo=_z, psi_hz_yhi=_z,
        )
        cpml_params = None

    # ------------------------------------------------------------------
    # Static metadata
    # ------------------------------------------------------------------
    n_src = len(sources)
    n_prb = len(probes)
    _exchange_interval = int(exchange_interval)

    # ------------------------------------------------------------------
    # Per-device source/probe injection via shard_map
    # ------------------------------------------------------------------

    def _inject_sources_shmap(st, src_vals_step):
        """Inject sources on their owning device using shard_map."""
        if n_src == 0:
            return st

        # Build per-device injection: shard_map gives each device its slab
        # We need device identity inside shard_map.
        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                P("x"),  # ex
                P("x"),  # ey
                P("x"),  # ez
                P(),     # src_vals_step: replicated scalar vector
            ),
            out_specs=(P("x"), P("x"), P("x")),
            check_rep=False,
        )
        def _inject(ex, ey, ez, sv):
            device_idx = lax.axis_index("x")
            for idx_s in range(n_src):
                li, lj, lk, lc = src_local_specs[idx_s]
                dev_id = src_device_ids[idx_s]
                val = jnp.where(device_idx == dev_id, sv[idx_s], 0.0)
                if lc == "ex":
                    ex = ex.at[li, lj, lk].add(val)
                elif lc == "ey":
                    ey = ey.at[li, lj, lk].add(val)
                elif lc == "ez":
                    ez = ez.at[li, lj, lk].add(val)
            return ex, ey, ez

        ex, ey, ez = _inject(st.ex, st.ey, st.ez, src_vals_step)
        return st._replace(ex=ex, ey=ey, ez=ez)

    def _sample_probes_shmap(st):
        """Sample probes on their owning devices, then sum across devices."""
        if n_prb == 0:
            return jnp.zeros(0, dtype=jnp.float32)

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(P("x"), P("x"), P("x"),
                      P("x"), P("x"), P("x")),
            out_specs=P(),
            check_rep=False,
        )
        def _sample(ex, ey, ez, hx, hy, hz):
            device_idx = lax.axis_index("x")
            samples = []
            for idx_p in range(n_prb):
                li, lj, lk, lc = prb_local_specs[idx_p]
                dev_id = prb_device_ids[idx_p]
                if lc == "ex":
                    raw = ex[li, lj, lk]
                elif lc == "ey":
                    raw = ey[li, lj, lk]
                elif lc == "ez":
                    raw = ez[li, lj, lk]
                elif lc == "hx":
                    raw = hx[li, lj, lk]
                elif lc == "hy":
                    raw = hy[li, lj, lk]
                else:
                    raw = hz[li, lj, lk]
                val = jnp.where(device_idx == dev_id, raw, 0.0)
                samples.append(val)
            return lax.psum(jnp.stack(samples), "x")

        return _sample(st.ex, st.ey, st.ez, st.hx, st.hy, st.hz)

    # ------------------------------------------------------------------
    # H and E local updates via shard_map
    # ------------------------------------------------------------------

    def _update_h_shmap(st, mat):
        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),  # ex, ey, ez
                P("x"), P("x"), P("x"),  # hx, hy, hz
                P(),                     # step (replicated)
                P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
            ),
            out_specs=(P("x"), P("x"), P("x"), P()),  # hx, hy, hz, step
            check_rep=False,
        )
        def _h(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r):
            _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
            _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
            new_st = _update_h_local(_st, _mat, dt, dx)
            return new_st.hx, new_st.hy, new_st.hz, new_st.step

        hx, hy, hz, step = _h(
            st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
            mat.eps_r, mat.sigma, mat.mu_r)
        return st._replace(hx=hx, hy=hy, hz=hz, step=step)

    def _update_e_shmap(st, mat, db_coeffs, db_st, lr_coeffs, lr_st):
        """E update (with optional dispersion) via shard_map."""

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),  # ex, ey, ez
                P("x"), P("x"), P("x"),  # hx, hy, hz
                P(),                     # step
                P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
                # debye coeffs
                P("x"), P("x"), P("x"), P("x"), P("x"),
                # debye state
                P("x"), P("x"), P("x"),
                # lorentz coeffs
                P("x"), P("x"), P("x"), P("x"), P("x"), P("x"),
                # lorentz state
                P("x"), P("x"), P("x"), P("x"), P("x"), P("x"),
            ),
            out_specs=(
                P("x"), P("x"), P("x"), P(),  # ex, ey, ez, step
                P("x"), P("x"), P("x"),        # new debye px, py, pz
                P("x"), P("x"), P("x"),        # new lorentz px, py, pz
                P("x"), P("x"), P("x"),        # new lorentz px_prev, py_prev, pz_prev
            ),
            check_rep=False,
        )
        def _e(ex, ey, ez, hx, hy, hz, step,
               eps_r, sigma, mu_r,
               d_ca, d_cb, d_cc, d_alpha, d_beta,
               d_px, d_py, d_pz,
               l_ca, l_cb, l_cc, l_a, l_b, l_c,
               l_px, l_py, l_pz, l_px_prev, l_py_prev, l_pz_prev):
            _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
            _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
            _db = (DebyeCoeffs(ca=d_ca, cb=d_cb, cc=d_cc, alpha=d_alpha, beta=d_beta),
                   DebyeState(px=d_px, py=d_py, pz=d_pz)) if has_debye else None
            _lr = (LorentzCoeffs(ca=l_ca, cb=l_cb, cc=l_cc, a=l_a, b=l_b, c=l_c),
                   LorentzState(px=l_px, py=l_py, pz=l_pz,
                                px_prev=l_px_prev, py_prev=l_py_prev, pz_prev=l_pz_prev)) \
                   if has_lorentz else None
            new_st, new_db, new_lr = _update_e_local_with_dispersion(
                _st, _mat, dt, dx, debye=_db, lorentz=_lr)
            # Unpack debye
            if new_db is not None:
                nd_px, nd_py, nd_pz = new_db.px, new_db.py, new_db.pz
            else:
                nd_px, nd_py, nd_pz = d_px, d_py, d_pz
            # Unpack lorentz
            if new_lr is not None:
                nl_px, nl_py, nl_pz = new_lr.px, new_lr.py, new_lr.pz
                nl_pxp, nl_pyp, nl_pzp = new_lr.px_prev, new_lr.py_prev, new_lr.pz_prev
            else:
                nl_px, nl_py, nl_pz = l_px, l_py, l_pz
                nl_pxp, nl_pyp, nl_pzp = l_px_prev, l_py_prev, l_pz_prev
            return (new_st.ex, new_st.ey, new_st.ez, new_st.step,
                    nd_px, nd_py, nd_pz,
                    nl_px, nl_py, nl_pz,
                    nl_pxp, nl_pyp, nl_pzp)

        (ex, ey, ez, step,
         nd_px, nd_py, nd_pz,
         nl_px, nl_py, nl_pz,
         nl_pxp, nl_pyp, nl_pzp) = _e(
            st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
            mat.eps_r, mat.sigma, mat.mu_r,
            db_coeffs.ca, db_coeffs.cb, db_coeffs.cc, db_coeffs.alpha, db_coeffs.beta,
            db_st.px, db_st.py, db_st.pz,
            lr_coeffs.ca, lr_coeffs.cb, lr_coeffs.cc, lr_coeffs.a, lr_coeffs.b, lr_coeffs.c,
            lr_st.px, lr_st.py, lr_st.pz, lr_st.px_prev, lr_st.py_prev, lr_st.pz_prev,
        )
        new_st = st._replace(ex=ex, ey=ey, ez=ez, step=step)
        new_db_st = db_st._replace(px=nd_px, py=nd_py, pz=nd_pz)
        new_lr_st = lr_st._replace(
            px=nl_px, py=nl_py, pz=nl_pz,
            px_prev=nl_pxp, py_prev=nl_pyp, pz_prev=nl_pzp)
        return new_st, new_db_st, new_lr_st

    # ------------------------------------------------------------------
    # Step function (operates on sharded arrays)
    # ------------------------------------------------------------------

    def step_fn_cpml(carry, xs):
        """Single FDTD step (CPML path) operating on sharded arrays."""
        _step_idx, src_vals = xs
        st = carry["fdtd"]
        cpml_st = carry["cpml"]
        db_st = carry["debye"]
        lr_st = carry["lorentz"]

        # 1. H update
        st = _update_h_shmap(st, sharded_materials)

        # 2. CPML H correction
        st, cpml_st = _apply_cpml_h_shmap(
            st, cpml_params, cpml_st, n_cpml, dt, dx,
            mesh, n_devices, ghost=ghost)

        # 3. Exchange H ghost cells (conditionally skip)
        do_exchange = (_step_idx % _exchange_interval == 0)
        st = lax.cond(
            do_exchange,
            lambda s: _exchange_h_ghosts_shmap(s, mesh, n_devices),
            lambda s: s,
            st,
        )

        # 4. E update
        st, db_st, lr_st = _update_e_shmap(
            st, sharded_materials,
            debye_coeffs_sharded, db_st,
            lorentz_coeffs_sharded, lr_st)

        # 5. CPML E correction
        st, cpml_st = _apply_cpml_e_shmap(
            st, cpml_params, cpml_st, n_cpml, dt, dx,
            mesh, n_devices, ghost=ghost)

        # 6. Exchange E ghost cells
        st = lax.cond(
            do_exchange,
            lambda s: _exchange_e_ghosts_shmap(s, mesh, n_devices),
            lambda s: s,
            st,
        )

        # 7. Source injection
        st = _inject_sources_shmap(st, src_vals)

        # 8. Probe sampling
        probe_out = _sample_probes_shmap(st)

        return {"fdtd": st, "cpml": cpml_st,
                "debye": db_st, "lorentz": lr_st}, probe_out

    def step_fn_pec(carry, xs):
        """Single FDTD step (PEC path) operating on sharded arrays."""
        _step_idx, src_vals = xs
        st = carry["fdtd"]
        db_st = carry["debye"]
        lr_st = carry["lorentz"]

        # 1. H update
        st = _update_h_shmap(st, sharded_materials)

        # 2. Exchange H ghost cells
        do_exchange = (_step_idx % _exchange_interval == 0)
        st = lax.cond(
            do_exchange,
            lambda s: _exchange_h_ghosts_shmap(s, mesh, n_devices),
            lambda s: s,
            st,
        )

        # 3. E update
        st, db_st, lr_st = _update_e_shmap(
            st, sharded_materials,
            debye_coeffs_sharded, db_st,
            lorentz_coeffs_sharded, lr_st)

        # 4. Exchange E ghost cells
        st = lax.cond(
            do_exchange,
            lambda s: _exchange_e_ghosts_shmap(s, mesh, n_devices),
            lambda s: s,
            st,
        )

        # 5. PEC boundaries
        st = _apply_pec_shmap(st, mesh, n_devices, nx_local)

        # 6. Source injection
        st = _inject_sources_shmap(st, src_vals)

        # 7. Probe sampling
        probe_out = _sample_probes_shmap(st)

        return {"fdtd": st, "debye": db_st, "lorentz": lr_st}, probe_out

    # ------------------------------------------------------------------
    # Build step indices and waveform scan inputs
    # ------------------------------------------------------------------
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    # src_waveforms shape: (n_steps, n_sources) — replicated, not sharded
    src_waveforms_rep = jax.device_put(src_waveforms, rep)
    xs = (step_indices, src_waveforms_rep)

    # ------------------------------------------------------------------
    # Run with jit + lax.scan
    # ------------------------------------------------------------------
    if use_cpml:
        carry_init = {
            "fdtd": sharded_state,
            "cpml": cpml_state_sharded,
            "debye": debye_state_sharded,
            "lorentz": lorentz_state_sharded,
        }
        run_fn = jax.jit(lambda carry, xs: lax.scan(step_fn_cpml, carry, xs))
        final_carry, probe_ts = run_fn(carry_init, xs)
        final_state_sharded = final_carry["fdtd"]
    else:
        carry_init = {
            "fdtd": sharded_state,
            "debye": debye_state_sharded,
            "lorentz": lorentz_state_sharded,
        }
        run_fn = jax.jit(lambda carry, xs: lax.scan(step_fn_pec, carry, xs))
        final_carry, probe_ts = run_fn(carry_init, xs)
        final_state_sharded = final_carry["fdtd"]

    # ------------------------------------------------------------------
    # Gather final state: sharded (n_devices*nx_local, ny, nz) ->
    # stacked (n_devices, nx_local, ny, nz) -> gathered (nx, ny, nz)
    # ------------------------------------------------------------------
    def _unstack_and_gather(sharded_arr):
        arr = np.array(sharded_arr)  # pull to host
        # arr: (n_devices * nx_local, ny, nz)
        total_x = arr.shape[0]
        assert total_x == n_devices * nx_local
        stacked = arr.reshape(n_devices, nx_local, *arr.shape[1:])
        gathered = jnp.array(gather_array_x(jnp.array(stacked), ghost))
        # Trim padding cells if nx was padded
        if pad_x > 0:
            gathered = gathered[:nx]
        return gathered

    final_state = FDTDState(
        ex=_unstack_and_gather(final_state_sharded.ex),
        ey=_unstack_and_gather(final_state_sharded.ey),
        ez=_unstack_and_gather(final_state_sharded.ez),
        hx=_unstack_and_gather(final_state_sharded.hx),
        hy=_unstack_and_gather(final_state_sharded.hy),
        hz=_unstack_and_gather(final_state_sharded.hz),
        step=int(final_state_sharded.step),
    )

    # ------------------------------------------------------------------
    # Probe time series: already summed across devices inside shard_map
    # probe_ts shape: (n_steps, n_probes) or (n_steps, 0)
    # ------------------------------------------------------------------
    if n_prb > 0:
        time_series = jnp.array(probe_ts)  # (n_steps, n_probes)
    else:
        time_series = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    return Result(
        state=final_state,
        time_series=time_series,
        s_params=None,
        freqs=None,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
