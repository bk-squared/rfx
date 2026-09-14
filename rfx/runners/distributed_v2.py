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

from rfx.core.jax_utils import is_tracer
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
from rfx.runners._distributed_common import exchange_component_shmap


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

# Exchange ghost cells for one field component using shard_map.
#
# The body lived here verbatim and is now the shared
# ``exchange_component_shmap`` in ``_distributed_common.py`` (the NU
# runner carried a byte-identical copy). Kept as a module-local alias so
# the existing call sites are unchanged.
_exchange_component_shmap = exchange_component_shmap


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
                     nx_local_with_ghost: int, pad_x: int = 0) -> FDTDState:
    """Apply PEC boundary conditions using shard_map for device identity.

    ``pad_x`` (#622): when the last rank's slab carries alignment-pad
    cells past the real x-hi face (global node ``nx - 1``), the face
    must act on that real node, not on the padded slab end.
    """

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

        # X-hi PEC: device N-1 only (skip ghost AND alignment pad, #622)
        is_last = (device_idx == n_devices - 1)
        last_real = nx_local_with_ghost - 1 - ghost - pad_x
        ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
        ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
        ey = ey.at[last_real, :, :].set(ey_xhi)
        ez = ez.at[last_real, :, :].set(ez_xhi)

        return ex, ey, ez

    ex, ey, ez = _pec(state.ex, state.ey, state.ez)
    return state._replace(ex=ex, ey=ey, ez=ez)


def _apply_pmc_shmap(state: FDTDState, mesh: Mesh, n_devices: int,
                     nx_local_with_ghost: int,
                     pmc_faces: frozenset, pad_x: int = 0) -> FDTDState:
    """PMC (``H_tangential = 0``) for the sharded uniform scan body.

    Dual of :func:`_apply_pec_shmap`. Hook point is the H-half of the
    scan body (after H ghost exchange, before E update) — OQ9 directive.
    PMC must fire before the next E-update reads H via curl, so it is
    NOT placed at the E-half mirror of PEC's call site.

    Y- and Z-face PMC is local to every rank; X-face PMC is rank-
    conditional and acts on the first / last real cell, not the seam
    ghost — nor the alignment-pad cells when ``pad_x > 0`` (#622).
    """
    if not pmc_faces:
        return state

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("x"),  # hx
            P("x"),  # hy
            P("x"),  # hz
        ),
        out_specs=(
            P("x"),
            P("x"),
            P("x"),
        ),
        check_rep=False,
    )
    def _pmc(hx, hy, hz):
        ghost = 1

        # Yee convention: _hi PMC acts on index -2 (0.5·dx INSIDE the
        # wall), not -1 (ghost outside). See rfx/boundaries/pmc.py.
        if "y_lo" in pmc_faces:
            hx = hx.at[:, 0, :].set(0.0)
            hz = hz.at[:, 0, :].set(0.0)
        if "y_hi" in pmc_faces:
            hx = hx.at[:, -2, :].set(0.0)
            hz = hz.at[:, -2, :].set(0.0)
        if "z_lo" in pmc_faces:
            hx = hx.at[:, :, 0].set(0.0)
            hy = hy.at[:, :, 0].set(0.0)
        if "z_hi" in pmc_faces:
            hx = hx.at[:, :, -2].set(0.0)
            hy = hy.at[:, :, -2].set(0.0)

        device_idx = lax.axis_index("x")
        is_first = (device_idx == 0)
        is_last = (device_idx == n_devices - 1)
        last_real = nx_local_with_ghost - 1 - ghost - pad_x
        last_inside = last_real - 1

        if "x_lo" in pmc_faces:
            hy_new = jnp.where(is_first, 0.0, hy[ghost, :, :])
            hz_new = jnp.where(is_first, 0.0, hz[ghost, :, :])
            hy = hy.at[ghost, :, :].set(hy_new)
            hz = hz.at[ghost, :, :].set(hz_new)
        if "x_hi" in pmc_faces:
            hy_new = jnp.where(is_last, 0.0, hy[last_inside, :, :])
            hz_new = jnp.where(is_last, 0.0, hz[last_inside, :, :])
            hy = hy.at[last_inside, :, :].set(hy_new)
            hz = hz.at[last_inside, :, :].set(hz_new)

        return hx, hy, hz

    hx, hy, hz = _pmc(state.hx, state.hy, state.hz)
    return state._replace(hx=hx, hy=hy, hz=hz)


# ---------------------------------------------------------------------------
# CPML inside shard_map
# ---------------------------------------------------------------------------

def _apply_cpml_e_shmap(state, cpml_params, cpml_state, n_cpml, dt, dx,
                         mesh, n_devices, ghost=1, eps_r=None, pad_x=0):
    """Apply CPML E-field correction using shard_map.

    ``eps_r`` (the x-sharded per-cell relative permittivity slab) is a new
    READ-ONLY input: it is threaded into the inner shard so the CPML
    correction is material-aware (#205).  It is NOT returned, so it gets an
    extra ``P("x")`` entry in ``in_specs`` only.  ``None`` reproduces the
    vacuum (pre-#205) coefficient bit-identically.

    ``pad_x`` (#623, same class as #622): forwarded to
    :func:`rfx.runners.distributed._apply_cpml_e_distributed` so the x-hi
    CPML window on the last rank targets the real physical face (global
    node ``nx - 1``) instead of the padded slab end.
    """

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
            P("x"),  # eps_r (read-only, material-aware coefficient)
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
                psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi,
                eps_r_slab):
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
            n_devices, ghost=ghost, axis_name="x", eps_r=eps_r_slab,
            pad_x=pad_x)
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
        eps_r,
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
                         mesh, n_devices, ghost=1, mu_r=None, pad_x=0):
    """Apply CPML H-field correction using shard_map.

    ``mu_r`` (the x-sharded per-cell relative permeability slab) is a new
    READ-ONLY input threaded into the inner shard for material-aware CPML
    (#205).  Like ``eps_r`` in the E-variant it gets one extra ``P("x")``
    in ``in_specs`` only and is not returned.  ``None`` reproduces the
    vacuum (pre-#205) coefficient bit-identically.

    ``pad_x`` (#623, same class as #622): forwarded to
    :func:`rfx.runners.distributed._apply_cpml_h_distributed` so the x-hi
    CPML window on the last rank targets the real physical face (global
    node ``nx - 1``) instead of the padded slab end.
    """

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
            P("x"),  # mu_r (read-only, material-aware coefficient)
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
                psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi,
                mu_r_slab):
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
            n_devices, ghost=ghost, axis_name="x", mu_r=mu_r_slab,
            pad_x=pad_x)
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
        mu_r,
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

# ---------------------------------------------------------------------------
# Distributed admission gate (B0, 2026-09-14)
# ---------------------------------------------------------------------------
# Five features reached this lane with no refusal and no warning, and came
# back wrong.  The research note
# ``rfx-research-notes/accel-import-20260913/DIRECTION-distributed-preflight.md``
# (S3-S4) and its survey (``decomposition-survey.md`` SC1) name them; what
# follows is that note's FIRST layer -- the position-INDEPENDENT admission
# check.  Every one fails closed and names both the feature and how to
# proceed; none of them warns and drops.
#
# Measured on main BEFORE these refusals existed (2026-09-14, 2 virtual CPU
# devices, 40 steps, 24x12x12 mm PEC box at dx=1 mm, source at (6, 6, 6) mm
# with ``amplitude_kind='field'``, TWO Ez probes at x=12 and x=22 mm -- the
# probe ROW matters: the energies below are sums over the whole trace, and a
# ONE-probe variant of the same fixture gives 9.951558e-04 and 1.982875e-03
# instead.  "energy" is ``np.sum(trace.astype(np.float64) ** 2)``; the f32
# sum differs in the 7th digit (9.953717e-04), which is why the convention
# is stated rather than left implicit.  The harness is
# ``tests/unit/runners/test_distributed_admission_refusals.py`` (its
# ``_build`` builds exactly this fixture) and the table is in
# ``docs/design_notes/2026-09-14_distributed_admission_refusals.md``:
#
#   periodic axes 'y'  distributed vs native max|dEz| = 1.963798e-04 on a
#                      1.090241e-03 peak (18.0% of peak), 0 warnings, no error
#   extent port        distributed probe energy 0.0 (max|E| 0.0) vs native
#                      9.953718e-04 (max|E| 1.412484e-02), 0 warnings
#   excite=False port  distributed energy 1.983971e-03 -- BIT-IDENTICAL to the
#                      same port with excite=True -- vs native 0.0; with the
#                      documented ``waveform=None`` default it instead died in
#                      ``make_port_source`` with "TypeError: Expected a
#                      callable value, got None", which names no feature
#   flux / NTFF        result.flux_monitors is None (native: 1 monitor named
#                      'flux_x_0'), result.ntff_data is None (native:
#                      NTFFData), 0 warnings
#
# The x-absorber class (#5) is position-DEPENDENT and lives in
# :func:`check_x_absorber_fits_ranks`, called from inside
# :func:`run_distributed` once the slab arithmetic is known.  Round 3
# narrowed its bound by one cell per face: ONE cell of window overflow is a
# measured no-op (the innermost CPML layer has sigma=0/kappa=1, so its
# correction is identically zero), and refusing it refused configurations
# origin/main runs byte-identically to the fitting run.  The phantom
# window that the one-cell fixtures were really wrong about is class #6,
# :func:`check_absorber_faces_are_absorbing`, which round 4 widened from
# the two x faces to all six -- the same defect is on y and z, and was
# ADMITTED until then (90.58% of the source probe's own peak, measured).
DISTRIBUTED_UNSUPPORTED_CODE = "distributed_unsupported_feature"


def refuse_unsupported_distributed_features(sim, *, lane, bloch=None):
    """Refuse the features the distributed lanes would silently drop.

    Four position-independent classes, in declaration order:

    1. **Periodic / Bloch boundaries.** The distributed local kernels are
       unconditionally non-periodic -- ``rfx/runners/distributed.py:351``
       ("non-periodic (ghost cells handle inter-device coupling)"), and
       the same sentence again at :380, :415 and :440 -- and
       no kernel or stepper in ``rfx/runners/distributed_v2.py`` reads
       ``sim._periodic_axes``: on main (d56f68eb) the string did not occur
       in the file at all, and on this branch every occurrence is inside
       this gate -- prose lines included, which is why an earlier draft of
       this sentence miscounted them as three: the substring also appears
       inside the test NAME
       ``test_the_runner_still_never_reads_periodic_axes``.  Do not trust
       a hand count here; that test compares the whole-file count against
       the gate count, so it stays correct as this docstring is edited.
       A periodic axis was therefore solved with the DECLARED
       non-periodic boundary instead -- PEC here, CPML if
       ``boundary='cpml'`` -- with nothing said.  Round 3 of the review
       corrected the earlier "open ghost-coupled axis" wording by
       measurement: y and z are not sharded at all, and on origin/main
       (7b511591) the distributed ``periodic_axes='y'`` probe trace is
       BYTE-IDENTICAL to the distributed PEC trace (``tobytes()`` equal,
       max|d| 0.0, and the distributed PEC trace is 2.9e-10 from native
       PEC), which is why the RED diff 1.963798e-04 equals
       native-periodic-vs-native-PEC to every digit.
    2. **Extended lumped ports** (``impedance > 0`` with ``extent`` set).
       ``distributed_v2.py`` forks on ``impedance > 0.0 and extent is
       None`` then ``elif impedance == 0.0``; a wire port satisfies
       neither, so it got no source, no resistive termination and no
       error.  The pmap twin in ``rfx/runners/distributed.py`` -- which
       is also the ``n_devices == 1`` delegate -- has the same fork, at
       the same two ``if pe.impedance > 0.0 and pe.extent is None:`` /
       ``elif pe.impedance == 0.0:`` lines.  Cited by symbol and not by
       line number on purpose: this branch inserts the two admission-gate
       blocks ABOVE that fork, which moved it from :1414/:1422 on
       origin/main to :1476/:1484 here, and an error message that points
       62 lines off is worse than one that points at a name.
    3. **Passive ports** (``excite=False``).  No port branch in either
       distributed runner reads ``pe.excite`` -- on main (d56f68eb) the
       name ``excite`` did not occur in either file, and on this branch
       its occurrences in ``distributed_v2.py`` are all inside this gate;
       ``rfx/runners/uniform.py:421,440`` honours it.  Every port was
       excited.
    4. **Flux monitors / NTFF boxes.**  Neither runner has an accumulator
       for them (the strings ``flux`` and ``ntff`` do not occur), so a
       registered monitor was dropped and the ``Result`` came back with
       the field set to ``None`` -- same class as the #579 DFT-plane
       refusal, so the same treatment.

    Parameters
    ----------
    sim : Simulation
        The simulation about to be dispatched to a distributed runner.
    lane : str
        Human name of the lane, used verbatim in the messages.
    bloch : object or None
        An explicit Bloch phase, when a caller has one.  **A placeholder
        today, and deliberately kept as one.**  ``run()`` has no ``bloch=``
        parameter, and ``sim._bloch`` is not a ``Simulation`` attribute at
        all: the only ``_bloch`` in the package is a LOCAL variable inside
        ``rfx/simulation.py`` (:890, assigned at :942 from
        ``bloch_phase_tuple`` and handed straight to the step context at
        :1117), reachable only from oblique TFSF -- which falls back to a
        single device before this gate runs.  So the ``getattr`` below
        always yields ``None`` today and the periodic-axis half of class 1
        is the only reachable one.  The parameter and the read stay so that
        an explicit Bloch phase cannot slip in BEHIND the refusal on the
        day one is added; no test can exercise this half without
        monkeypatching, and the refusal claims no measured coverage for
        it.

    Raises
    ------
    NotImplementedError
        Naming the feature, why it cannot ride this lane, and the two ways
        forward (drop the feature, or omit ``devices=...``).
    """
    periodic_axes = getattr(sim, "_periodic_axes", "") or ""
    if bloch is None:
        bloch = getattr(sim, "_bloch", None)
    if periodic_axes or bloch is not None:
        _what = []
        if periodic_axes:
            _what.append(
                "periodic axes "
                + ", ".join(repr(a) for a in periodic_axes)
            )
        if bloch is not None:
            _what.append("a Bloch phase")
        raise NotImplementedError(
            f"{' and '.join(_what)}: periodic / Bloch boundaries are not "
            f"supported on the {lane} path. The distributed local kernels "
            "are unconditionally non-periodic "
            "(rfx/runners/distributed.py:351, \"non-periodic (ghost cells "
            "handle inter-device coupling)\", and again at :380/:415/:440) "
            "and no kernel in rfx/runners/distributed_v2.py reads "
            "sim._periodic_axes (outside this admission gate the attribute "
            "does not appear in the file at all), so the axis is solved "
            "with the DECLARED non-periodic boundary instead -- PEC here, "
            "CPML if boundary='cpml' -- with nothing said (measured on main "
            "before this refusal: periodic_axes='y' gave a distributed "
            "trace 1.96e-04 away from the native periodic trace on a "
            "1.09e-03 peak -- 18% of peak -- with no error and no warning, "
            "and that trace was byte-identical to the distributed PEC run "
            "of the same model, which is how we know the axis is reflected "
            "and not left open). "
            "Drop the periodic axes (or the Bloch phase), or omit "
            "devices=... : the single-device run() lane honours both."
        )

    extended = [
        (i, pe) for i, pe in enumerate(getattr(sim, "_ports", ()) or ())
        if pe.impedance > 0.0 and pe.extent is not None
    ]
    if extended:
        _detail = ", ".join(
            f"port {i} ({pe.component} at {pe.position}, "
            f"impedance={pe.impedance} ohm, extent={pe.extent})"
            for i, pe in extended
        )
        raise NotImplementedError(
            f"add_port(..., extent=...) (extended / wire port) is not "
            f"supported on the {lane} path: the lane forks on "
            "`impedance > 0.0 and extent is None` and then on "
            "`impedance == 0.0` (the source/termination fork in "
            "rfx/runners/distributed_v2.py, and its twin at the same two "
            "`if pe.impedance > 0.0 and pe.extent is None:` / `elif "
            "pe.impedance == 0.0:` lines of rfx/runners/distributed.py), "
            "and a port with a "
            "positive impedance AND an extent satisfies NEITHER -- it gets "
            "no source, no resistive termination and no error. Declared "
            f"here: {_detail}. Measured on main before this refusal: the "
            "distributed probe energy was exactly 0.0 where the native run "
            "read 9.95e-04 (peak |Ez| 1.41e-02). Use a single-cell lumped "
            "port (drop extent=), or omit devices=... : the single-device "
            "run() lane realizes wire ports "
            "(rfx/runners/uniform.py:419-420, setup_wire_port)."
        )

    passive = [
        (i, pe) for i, pe in enumerate(getattr(sim, "_ports", ()) or ())
        if not pe.excite
    ]
    if passive:
        _detail = ", ".join(
            f"port {i} ({pe.component} at {pe.position})" for i, pe in passive
        )
        raise NotImplementedError(
            f"add_port(..., excite=False) (passive matched load) is not "
            f"supported on the {lane} path: no port branch in "
            "rfx/runners/distributed_v2.py or rfx/runners/distributed.py "
            "reads `excite` (outside this admission gate the name does not "
            "appear in either file), while "
            "rfx/runners/uniform.py:421,440 honours it (`if "
            "pe.excite:` on both the wire and the lumped branch), so every "
            f"port here is EXCITED. Declared here: {_detail}. Measured on "
            "main before this refusal: an excite=False port with an "
            "explicit waveform produced a distributed probe energy of "
            "1.983971e-03 -- bit-identical to the same port with "
            "excite=True -- where the native run read exactly 0.0; with the "
            "documented waveform=None default the lane instead died inside "
            "make_port_source with \"Expected a callable value, got None\". "
            "Excite the port (excite=True) and read the other ports' V/I, "
            "or omit devices=... : multi-port S-parameter extraction with "
            "passive loads belongs on the single-device run() lane."
        )

    _monitors = []
    if getattr(sim, "_flux_monitors", None):
        _monitors.append(
            f"{len(sim._flux_monitors)} flux monitor(s) from "
            "add_flux_monitor()"
        )
    if getattr(sim, "_ntff", None) is not None:
        _monitors.append("an NTFF box from add_ntff_box()")
    if _monitors:
        raise NotImplementedError(
            "add_flux_monitor() / add_ntff_box() are not supported on the "
            f"{lane} path (same class as the #579 DFT-plane refusal): "
            "neither rfx.runners.distributed_v2 nor rfx.runners.distributed "
            "accumulates a flux or NTFF surface DFT -- outside this "
            "admission gate neither file mentions flux or ntff at all: no "
            "accumulator and no reduce across ranks -- so a registered "
            f"monitor is silently dropped. Declared here: "
            f"{', and '.join(_monitors)}. Measured on main before this "
            "refusal: result.flux_monitors was None where the native run "
            "returned one monitor ('flux_x_0'), and result.ntff_data was "
            "None where the native run returned an NTFFData, with no "
            "warning either time. Drop the flux monitors / NTFF box, or "
            "omit devices=... (use a single-device run() instead)."
        )


def check_x_absorber_fits_ranks(*, nx, n_devices, nx_per, pad_x, ghost,
                                cpml_layers, pad_x_lo=None, pad_x_hi=None,
                                lane="distributed multi-device run()"):
    """Refuse an x CPML absorber whose window overflows a rank's slab by >1.

    Class 5 of the admission gate, and the one that is position-dependent:
    it depends on the slab arithmetic, so it runs inside
    :func:`run_distributed` rather than at the API boundary.

    The condition is derived from the windows
    :func:`rfx.runners.distributed._apply_cpml_e_distributed` (and its H
    twin) actually slice on a per-rank slab of length
    ``nx_local = nx_per + 2*ghost``::

        x-lo, rank 0     [ghost, ghost + n)
        x-hi, rank N-1   [nx_per + ghost - pad_x - n, nx_per + ghost - pad_x)

    (``xlo``/``xhi`` at rfx/runners/distributed.py:807-808; ``pad_x`` is the
    #623 alignment pad, which sits PAST the real x-hi face on the last
    rank.)  A rank owns the real cells ``[ghost, ghost + nx_per)`` of its
    own slab, so the windows stay inside owned cells exactly while
    ``n <= nx_per`` (x-lo) and ``n <= nx_per - pad_x`` (x-hi).

    **That is NOT the bound this check enforces**, and round 3 of the B0
    review is why.  ONE cell of overflow is provably and measurably a
    NO-OP, so refusing it refuses configurations main runs at parity::

        n <= nx_per + 1           (x-lo face)
        n <= nx_per - pad_x + 1   (x-hi face)

    Why one cell is free, from the code: ``_cpml_profile``
    (rfx/boundaries/cpml.py) grades ``rho = 1 - arange(n)/(n-1)``, so the
    INNERMOST layer (profile index ``n-1``, or index ``0`` of the flipped
    x-hi profile) has ``rho = 0``, hence ``sigma = 0``, ``kappa = 1`` and
    ``c = sigma*(b-1)/denom = 0``.  Its psi therefore stays at its zero
    initial value forever and the correction it applies,
    ``-ce*psi - ce*(1/kappa - 1)*curl``, is identically zero -- not small,
    exactly zero.  Both windows are ANCHORED (x-lo at ``ghost``, x-hi at
    ``-(ghost + pad_x)``), so one cell of overflow moves exactly that
    no-op layer into the halo and leaves layers ``0..n-2`` (x-hi:
    ``1..n-1``) on the very cells they occupy when the absorber fits.

    MEASURED on pristine origin/main (7b511591, 2026-09-14, symmetric
    ``boundary='cpml'``, ``cpml_layers=8``, dx = 1 mm, y/z 8 mm,
    ``amplitude_kind='field'`` Ez source at x = 3 mm, Ez probe row
    x = 1/3/5/6 mm, 120 steps), with
    ``XLA_FLAGS=--xla_force_host_platform_device_count=4``:

    * 11x8x8 mm -> nx = 28, 4 devices, ``pad_x=0``, ``nx_per=7``: ``n=8``
      overflows BOTH faces by one cell.  max|dEz| vs native
      9.536743e-07 on a 9.238437e+00 peak = **1.032290e-07 of peak**,
      0 warnings, and the probe trace is **byte-identical** to the
      2-device run (``nx_per=14``, absorber fits).
    * 8x8x8 mm -> nx = 25, 3 devices, ``pad_x=2``, ``nx_per=9``: the x-hi
      bound ``nx_per - pad_x = 7`` is exceeded by one.  1.430511e-06 on
      9.238317e+00 = **1.548455e-07 of peak**, 0 warnings, and the SAME
      REL to seven digits as the fitting 2-device run (1.548455e-07
      both).  Unlike the 11 mm / 4-device row above, this one is NOT
      byte-identical -- round 4 measured 3-dev vs 2-dev max|d| =
      9.536743e-07, one float32 last bit on a 9.24 peak.  Stated as
      "same rel", not as a second byte-identity claim.

    A 25-row sweep on the same tree (domains 6/8/11 mm, ``cpml_layers``
    8/12/16, ``n_devices`` 2..6) separates the bands exactly: every
    configuration this check now admits ran at 5.4e-08..1.1e-07 of peak
    with 0 warnings, and every configuration it refuses died with
    ``TypeError: mul got incompatible shapes for broadcasting`` -- 0
    mismatches in either direction.

    TWO cells of overflow move a layer with ``sigma > 0`` into the halo.
    At ``ghost == 1`` (what :func:`run_distributed` hardcodes) that is
    also where the window runs past the slab end, the clipped window is
    shorter than the psi arrays, and XLA raises a broadcasting error that
    names no feature: e.g. ``BoundarySpec(x=('pec','pec'), y=cpml,
    z=cpml)`` with ``cpml_layers=8`` at 6x8x8 mm (nx=7, pad_x=1,
    nx_per=4) dies with ``(8, 1, 1), (5, 25, 25)``, and at 8x8x8 mm
    (nx=9, nx_per=5) with ``(8, 1, 1), (6, 25, 25)``.  The general
    statement is per ``ghost``, not per two cells: the window CLIPS when
    ``n > nx_per + ghost`` (x-lo) / ``n > nx_per + ghost - pad_x``
    (x-hi), so at ``ghost > 1`` overflows of 2..ghost cells would be
    SILENT rather than fatal -- which is why this check refuses at
    ``> 1`` and not at the clip, and why the bound does not go stale the
    day ``ghost`` stops being 1.  ``ghost`` is still reported in the
    message and used to say which band the caller is in.

    **This bound is NECESSARY, NOT SUFFICIENT** (round-2 review,
    measured 2026-09-14 on pristine d56f68eb, 2 CPU devices, 60 steps).
    Satisfying it does not make the run right.  The ADMITTED boundary
    case ``n == nx_per - pad_x`` -- ``BoundarySpec(x=(cpml, pec),
    y=cpml, z=cpml)`` at domain 7x8x8 mm, ``cpml_layers=8`` (nx=16,
    pad_x=0, nx_per=8), source (3, 4, 4) mm, probe row x = 1/3/5/6 mm --
    is::

        native peak |Ez| (x=3 mm)       4.420052e+00
        max|dEz|                        2.040125e+00   (46.16% of peak)
        probe x=5 mm                    native 1.800406e-01 vs
                                        distributed 5.905863e-05  (99.97%)
        probe x=6 mm                    native 4.441055e-02 vs
                                        distributed 1.295477e-07  (100.0%)
        warnings raised                 0

    The dominant mechanism there is not the overflow but the PHANTOM
    window: the runner applies BOTH x-face windows whenever
    ``sim._boundary == "cpml"`` and ``grid.cpml_layers > 0``, without
    consulting ``grid.face_pads``, and ``ce_xhi = dt / (eps_r * EPS_0)``
    is non-zero at a PEC face, so the lane absorbs at the reflector.  (It
    does the same at y-lo/y-hi/z-lo/z-hi, which is why class 6 reads all
    six pads since round 4; this check stays x-only because the slab
    decomposition, and therefore the fit, is x-only.)
    Decided by measurement, not by reading: the same asymmetric model at
    24x8x8 mm reads max|dEz| = 1.610351e-04 on a 3.165971e-03 peak =
    5.086e-02 of peak at ``n_devices=2`` AND the identical figure at
    ``n_devices=1``, where there is no slab to overflow.  That class is
    refused separately and unconditionally by
    :func:`check_absorber_faces_are_absorbing` (class 6), which is what
    covers ``BoundarySpec(x=('pec', 'pec'), ...)`` and every asymmetric
    composition -- including ``_asym(8, 6)``, which this check admitted
    again the moment its bound was narrowed to the measurement.
    ``pad_x_lo`` / ``pad_x_hi`` are read here for the message only, so a
    caller who asked for no x absorber is not told to shrink one.

    Reachability, stated rather than implied: with a symmetric absorber at
    2 devices the bound is unreachable by construction
    (``nx_per >= n + 1``; confirmed at ``cpml_layers=16`` on a 2 mm
    interior, nx=35, which runs), and every asymmetric route is refused
    earlier-in-effect by class 6, so what this check actually catches on
    the CPU lane is ``n_devices >= 3`` with a deep absorber -- exactly the
    band that dies in XLA with no feature named.

    Parameters
    ----------
    pad_x_lo, pad_x_hi : int or None
        ``grid.pad_x_lo`` / ``grid.pad_x_hi`` -- the absorber padding
        outside the requested domain on each x face, 0 on a pec/pmc/
        periodic face.  Used for the message only (see above); ``None``
        means "not supplied", and the message then makes no claim about
        the declared faces.

    Raises
    ------
    ValueError
        Naming nx, n_devices, nx_per, cpml_layers and the face(s) that
        overflow, and how to proceed.
    """
    # ONE cell of overflow moves only the sigma=0 / kappa=1 no-op layer
    # into the halo, measured byte-identical to the fitting run -- see the
    # docstring. The binding depth is therefore one deeper than the
    # "stays inside owned cells" depth on each face.
    lo_limit = nx_per + 1
    hi_limit = nx_per - pad_x + 1
    faces = []
    if cpml_layers > lo_limit:
        faces.append(
            f"x-lo needs cells [{ghost}, {ghost + cpml_layers}) of rank 0's "
            f"slab but rank 0 owns [{ghost}, {ghost + nx_per}) and only its "
            f"innermost layer may fall outside"
        )
    if cpml_layers > hi_limit:
        _start = nx_per + ghost - pad_x - cpml_layers
        # The last rank's REAL cells stop pad_x short of its owned slab:
        # the trailing pad_x cells are #623 PEC-fill alignment cells that
        # sit past the physical x-hi face, which is exactly why the x-hi
        # limit is nx_per - pad_x (+1) and not nx_per (+1).
        _pad_note = (
            f" (its last {pad_x} cell(s) of "
            f"[{ghost}, {ghost + nx_per}) are #623 PEC-fill alignment cells "
            "past the x-hi face)"
        ) if pad_x else ""
        faces.append(
            f"x-hi needs cells [{_start}, {nx_per + ghost - pad_x}) of rank "
            f"{n_devices - 1}'s slab but that rank owns only "
            f"[{ghost}, {ghost + nx_per - pad_x}) real cells{_pad_note}, and "
            "only its innermost layer may fall outside"
        )
    if not faces:
        return
    # Which band: the window CLIPS (and XLA dies) past `ghost` cells of
    # overflow, and is silently wrong in between. At the hardcoded
    # ghost == 1 the two coincide, so the "died in XLA" sentence below is
    # true of everything this check refuses there -- but it is asserted
    # only when it is actually true, because a false claim about the
    # caller's own configuration is worse than a vaguer message.
    clips = (cpml_layers > nx_per + ghost
             or cpml_layers > nx_per + ghost - pad_x)
    _band = (
        "On main this configuration died inside XLA with a broadcasting "
        "error that named no feature (the clipped window is shorter than "
        "the psi profile)."
        if clips else
        f"With ghost={ghost} the window still fits the slab, so on main "
        "this configuration ran with no error and no warning while "
        f"{cpml_layers - min(lo_limit, hi_limit)} absorbing layer(s) "
        "updated ghost-halo cells instead of the owned cells they belong "
        "to."
    )
    # The largest FEWER device count that does fit, searched exactly rather
    # than estimated as nx // cpml_layers (pad_x is not monotone in N), and
    # against the SAME bound this check enforces.
    # n=1 is in this list whenever cpml_layers <= nx, i.e. for every real
    # grid, so the "no count fits" branch below is effectively unreachable
    # and fits == [1] is the common case -- and "use n_devices=1" is just
    # "omit devices=" said twice, so it gets the single-device wording
    # instead of a device-count recommendation.
    fits = [
        n for n in range(2, n_devices)
        if cpml_layers <= ((nx + (-nx) % n) // n) - ((-nx) % n) + 1
    ]
    remedy = (
        f"Use fewer devices (n_devices={max(fits)} is the largest count "
        f"above 1 that fits at nx={nx} and cpml_layers={cpml_layers})"
        if fits else
        f"No device count above 1 fits an absorber this deep at nx={nx}: "
        f"reduce cpml_layers (<= {max(hi_limit, 0)} fits at "
        f"n_devices={n_devices}) or increase nx"
    )
    # The caller may have declared no x absorber at all; say so rather
    # than blaming one (the runner applies the x windows regardless -- S5).
    # Name the faces that are OFF, and never assert "neither" when only
    # one of them is: for BoundarySpec(x=(cpml, pec), ...) the x-lo pad is
    # the full cpml_layers and only x-hi is off.
    _declared = ""
    if pad_x_lo is not None and pad_x_hi is not None:
        _faces_off = [
            f for f, pad in (("x-lo", pad_x_lo), ("x-hi", pad_x_hi))
            if pad == 0
        ]
        if _faces_off:
            _declared = (
                " NOTE: this model declares no CPML on "
                + " or ".join(_faces_off)
                + f" (grid.pad_x_lo={pad_x_lo}, grid.pad_x_hi={pad_x_hi}), "
                "but the distributed lane applies BOTH x-face CPML windows "
                "whenever boundary='cpml' and cpml_layers>0 -- it does not "
                "read grid.face_pads on any axis (rfx/runners/distributed.py: "
                "_init_cpml_distributed builds one scalar profile and the "
                "kernel drives it at every face). So the depth that has to "
                "fit is "
                f"cpml_layers={cpml_layers} on both x faces, including "
                + " and ".join(_faces_off)
                + ("." + (" Those faces are" if len(_faces_off) > 1
                            else " That face is"))
                + " also refused independently by "
                "check_absorber_faces_are_absorbing() (class 6)."
            )
    raise ValueError(
        f"the x CPML absorber does not fit inside one rank's slab on the "
        f"{lane} path: cpml_layers={cpml_layers} with nx={nx}, "
        f"n_devices={n_devices} gives nx_per={nx_per} owned cells per rank "
        f"(pad_x={pad_x} alignment cell(s) on the last rank, ghost={ghost}) "
        f"-- {'; '.join(faces)}. Exactly ONE cell of overflow is harmless "
        "(the innermost CPML layer has sigma=0 and kappa=1, so its "
        "correction is identically zero -- measured byte-identical to the "
        "fitting run on origin/main) and is ADMITTED; past that a layer "
        f"that really absorbs lands outside the owned slab. {_band} "
        f"{remedy}, or omit devices=... entirely (the single-device run() "
        "lane has no slab to overflow). NOTE ON MAGNITUDE: this bound is "
        "NECESSARY, NOT SUFFICIENT -- a configuration that satisfies it can "
        "still be wrong for a different reason, measured 46.16% of peak ONE "
        "CELL INSIDE this bound (x_lo='cpml'/x_hi='pec' with y/z CPML, "
        "cpml_layers=8, 7x8x8 mm, 2 devices, 60 steps, 0 warnings: nx=16, "
        "pad_x=0, nx_per=8, n=8 against an x-hi limit of "
        "nx_per-pad_x+1=9 -- it was the exact limit under the round-1/2 "
        "bound n <= nx_per-pad_x and is one cell inside it after round 3). "
        "That cause is the phantom window at a non-absorbing face, refused "
        "separately by check_absorber_faces_are_absorbing(); the "
        "measurements are in "
        "docs/design_notes/2026-09-14_distributed_admission_refusals.md "
        f"S2.5 and S2.6.{_declared}"
    )


def check_absorber_faces_are_absorbing(*, cpml_layers, face_pads, n_devices,
                                       lane="distributed multi-device run()"):
    """Refuse a CPML window driven at ANY face that declares no absorber.

    Class 6 of the admission gate, added in round 2 of the B0 review
    because the class-5 arithmetic bound turned out to be necessary and
    NOT sufficient: the configuration that satisfies it exactly is just as
    wrong, and for a different reason.  **Round 4 widened it from the two
    x faces to all six** -- see "Why all six faces" below; the y and z
    faces carry the identical defect and were ADMITTED until then.

    The distributed lane applies EVERY CPML face window whenever
    ``sim._boundary == "cpml"`` and ``grid.cpml_layers > 0``.  It never
    reads ``grid.face_pads``: ``_init_cpml_distributed``
    (``rfx/runners/distributed.py``) builds ONE scalar profile from
    ``_cpml_profile(grid.cpml_layers, ...)`` and the kernel applies it at
    x-lo/x-hi (on the owning ranks) and at y-lo/y-hi/z-lo/z-hi
    unconditionally, with a per-face E coefficient
    ``dt / (eps_r * EPS_0)`` that is non-zero at a PEC or PMC face.  So on
    a face whose ``BoundarySpec`` token is ``pec`` / ``pmc`` (or on an axis
    left out of ``cpml_axes``), where ``grid.face_pads`` is 0 because no
    absorber was padded outside the domain (``rfx/grid.py``,
    ``_face_pad``), the lane drives a ``cpml_layers``-deep absorber into
    the reflector.  The reflection the caller asked for is absorbed
    instead, with no error and no warning.

    The single-device ``run()`` lane does NOT have this defect: its
    ``init_cpml`` clamps each face's profile to that face's allocated pad
    (``rfx/boundaries/cpml.py``), which is why it is the reference every
    number below is measured against.

    MEASURED on the two x faces, pristine main (d56f68eb, 2026-09-14, 2
    virtual CPU devices; every row re-derived to the printed digit on
    ``origin/main`` 883615c6, 2026-09-15), all with 0 warnings:

    * ``BoundarySpec(x=(cpml, pec), y=(cpml, cpml), z=(cpml, cpml))``,
      ``cpml_layers=8``, 24x8x8 mm at dx=1 mm, field source (6, 6, 6) mm,
      Ez probes at x = 12 / 22 mm, 60 steps: max|dEz| **1.610351e-04** on
      a 3.165971e-03 row peak = **5.086e-02 of peak**, and the x = 22 mm
      probe (2 mm inside the ``x_hi='pec'`` face) is **99.46 % wrong on
      its own peak** (1.610351e-04 of 1.619078e-04).  At 100 steps the
      row figure grows to 7.890e-02 (the x = 22 mm probe 99.48 %).
    * the same spec at 7x8x8 mm -- the class-5 case one cell inside its
      bound -- is 46.16 % wrong at the source probe and 99.97 % / 100.0 %
      wrong at x = 5 / 6 mm.
    * ``BoundarySpec(x=(pec, pec), y=(cpml, cpml), z=(cpml, cpml))``,
      ``cpml_layers=8``, dx = 1 mm, y/z 8 mm, field Ez source at the
      domain centre, 60 steps.  The figure that does NOT depend on where
      the source sits is the one on the **face probes**, and it is what
      this row is quoted by (round 4: an earlier draft quoted
      "2.7-3.1 % wrong at the source", which is not re-derivable -- the
      measured source-probe figures for these three domains are 1.7610 %
      / 0.0225 % / 0.00002 %, *decaying* as the source moves away from a
      window that stays 8 cells deep)::

          15 mm (nx=16), source x=7 mm, probes x=2/7/13 mm:
              the x=2/13 mm probes 99.9986 % / 99.9319 % of their own
              peaks (max|dEz| 7.786655e-02 on a 4.421772e+00 row peak,
              1.760981e-02 of the row peak)
          19 mm (nx=20), source x=9 mm, probes x=2/9/17 mm:
              99.9917 % / 99.8366 % on the face probes
              (6.308181e-04 of the row peak, whose max sits on the
              x=2 mm FACE probe, not on the source probe)
          39 mm (nx=40), source x=19 mm, probes x=2/19/37 mm:
              99.9424 % / 99.4168 % on the face probes
              (3.307773e-05 of the row peak, again on the x=2 mm probe)

    * ``BoundarySpec(x=(pmc, cpml), y=cpml, z=cpml)``, dx=5 mm,
      16x8x8 cells, default ``cpml_layers`` (16) -- the configuration of
      ``tests/unit/boundaries/test_boundary_pmc_distributed.py``'s
      ``test_pmc_distributed_v2_x_lo_owner_and_non_owner`` -- is 54.08 %
      wrong at 30 steps and 93.48 % at 80 steps, and 100 % wrong at every
      probe within 5 cells of the PMC face.  That test asserts a
      zero-PATTERN on the PMC face and never a value, which is why it was
      green; it now composes its PMC face with PEC instead of CPML so the
      same structural claim is made on a configuration this lane can
      actually realize.

    **Why all six faces** (round 4, the consistency lens).  ``cpml.py``'s
    per-face clamp is what makes a reflector-plus-absorber composition
    work on the single-device lane, and the distributed lane has no
    equivalent on ANY axis -- not just x.  MEASURED on ``origin/main``
    883615c6 (identical digits on this branch's 461cfe53 and on the base
    7b511591), 24x8x8 mm at dx = 1 mm, ``cpml_layers=8``,
    ``amplitude_kind='field'`` Ez source at (6, 4, 4) mm, Ez probes at
    x = 6 / 12 / 20 mm, 60 steps, 2 virtual CPU devices, **0 warnings
    every time**:

    * ``BoundarySpec(x='cpml', y='cpml', z=Boundary(lo='pec', hi='cpml'))``
      -- the exact shape that
      ``tests/unit/boundaries/test_boundary_pmc_composition.py``'s OQ9 test
      was running at ``devices=devices[:2]``
      (``test_oq9_distributed_v2_cpml_path_enforces_pec_face_via_cpml_init``,
      split in round 4 into
      ``test_oq9_uniform_cpml_path_enforces_pec_face_via_cpml_init`` and
      ``test_oq9_distributed_v2_refuses_the_pec_face_composition``)
      -- is **90.58 % wrong at the source probe** (max|dEz|
      4.003862e+00 on a 4.420338e+00 peak) and **409.5 % / 281.3 % of
      their own peaks** at x = 12 / 20 mm.  Through the v1 pmap runner at
      ONE device the figures are bit-for-bit the same (90.5782 %,
      409.4823 %, 281.3395 %), which is how we know this is the window
      and not the decomposition -- the same proof of mechanism the x face
      has.
    * ``z=Boundary(lo='pec', hi='pec')``: max|dEz| **2.615769e+00** on a
      5.171041e-03 probe peak at x = 12 mm, i.e. 5.06e+04 % of that
      probe's own peak; 81.28 % at the source probe.
    * ``y=Boundary(lo='pmc', hi='cpml')``: **5.2x** the x = 12 mm probe's
      own peak (3.988669e-02 on 7.642152e-03); 48.62 % at the source.
    * ``y=Boundary(lo='pec', hi='pec')``: 373 % of the source probe's own
      peak (1.649596e+01 on 4.417783e+00).
    * SYMMETRIC control, same fixture, all six faces absorbing:
      **1.078195e-07** of peak (4.768372e-07 on 4.422548e+00) -- parity.

    And on its own fixture, the THIRD shipped false green of this class,
    found by widening the pre-PR re-run list to the whole fast suite.
    ``BoundarySpec(x='cpml', y='cpml', z=Boundary(lo='pmc', hi='pec'))``,
    dx=5 mm, 16x8x8x24 cells, default ``cpml_layers`` (16), Ex source at
    the domain centre, Ex probe one cell along x -- the configuration of
    ``tests/unit/boundaries/test_boundary_pmc_distributed.py``'s
    ``test_pmc_distributed_legacy_mixed_z``.  BOTH z pads are 0
    (``grid.face_pads == (16, 16, 16, 16, 0, 0)``) and the lane drives its
    16-layer window at the PMC z-lo AND the PEC z-hi.  MEASURED on
    883615c6 through the v1 pmap runner at one device, which is how that
    test calls it: **6.1030 %** of peak wrong at the test's own 30 steps
    (max|dEx| 2.869174e-02 on a 4.701230e-01 peak) and **15.4906 %** at
    80, with 0 warnings -- and all four of its zero-pattern assertions
    (``hx[:,:,0]``, ``hy[:,:,0]``, ``ex[:,:,-1]``, ``ey[:,:,-1]``) held on
    that run, which is exactly why it was green.  It now composes x and y
    with ``pec`` instead of ``cpml``: the PMC/PEC hooks, the pmap scan body
    and the axis ends under test are unchanged, the four zeros are exact,
    and the model runs at parity with the single-device lane
    (rel 1.361770e-07).  A liveness assertion was added with the fixture
    change, because four "is zero" claims with no energised-interior check
    also pass on a dead grid.

That OQ9 test asserted only a zero-pattern on the PEC face
    (``ex[:,:,0]``, ``ey[:,:,0]``) and never a value, so it stayed green on
    the 90.58 %-wrong run: the same false green the
    ``test_boundary_pmc_distributed.py`` rewrite above fixed on the x
    face.  Its structural claim -- PEC enforced through the per-face CPML
    profile, no scan-body hook -- is true on the lane that implements it,
    so it now makes that claim single-device with its assertions
    unchanged, and a sibling test pins what the distributed lane owes the
    same fixture: this refusal.  The defect is NOT a regression -- it is
    identical on the base tree -- but it is the same fail-closed violation
    as the x face, so it is refused by the same check rather than handed
    on as a "numbers-only gap" (the mistake round 2 had to undo for the
    x face).

    **2-D models.** ``rfx/grid.py`` drops ``z`` from ``cpml_axes`` when
    ``mode`` starts with ``2d``, so a 2-D CPML model has
    ``pad_z_lo == pad_z_hi == 0`` and is refused here.  That takes nothing
    away: MEASURED on 883615c6, ``mode='2d_tmz'`` and ``'2d_tez'``,
    24x8x8 mm at dx=1 mm, ``boundary='cpml'``, ``cpml_layers=8``, 2
    devices -- the run dies inside XLA with ``ValueError: Incompatible
    types for broadcasting: input type=float32[23,25,8] and requested
    type=float32[23,25,1]``, an error that names no feature.  This
    refusal replaces that with a named one.

    Negative control, same fixture with all six faces absorbing
    (``boundary='cpml'``, ``cpml_layers=8``, 24x8x8 mm, probes 12/22 mm,
    60 steps): max|dEz| 5.820766e-09 on a 3.170117e-03 peak =
    1.836e-06 of peak at 2 devices (1.395e-06 at 1 device) -- parity,
    three orders inside the shipped CPML tolerance of 1e-3
    (``tests/unit/runners/test_distributed.py``).  So this refusal costs
    the fully-absorbing model nothing, and that is the common case:
    ``boundary='cpml'`` pads all six faces outside the requested domain by
    construction.

    **Reachability -- why this refuses instead of warning.**  A caller who
    probes only far from the faces cannot see this class at any tolerance
    the suite uses, so no threshold would catch it.  MEASURED on 883615c6,
    ``x=(pec,pec)`` with y/z CPML, ``cpml_layers=8``, 39x8x8 mm, source
    x = 19 mm, the CENTRE probe alone and 200 steps (the shape of the
    shipped ``tests/unit/runners/test_distributed.py`` parity tests):
    8.356664e-04 on a 9.237972e+00 peak = **9.045994e-05 of peak**, which
    is INSIDE the shipped 1e-3 CPML tolerance -- while the face probes on
    the same run are 99.4-99.9 % wrong.  A warn-and-continue would be read
    as "passes" by exactly the fixtures most likely to be written.

    Scope. This is an admission refusal, not the fix.  The fix is to gate
    every face window on ``grid.face_pads`` and read ``grid.face_layers``
    for the depth, which is lane B's subject (the distributed CPML outer
    termination) and changes physics rather than admission; until it
    lands, the honest answer for these configurations is to refuse them.
    When lane B lands, delete this function and narrow class 5 to the
    faces that really are CPML.

    Two scope edges, recorded rather than implied (round 3, widened in
    round 4):

    * this check fires on ANY zero face pad, not only a ``pec``/``pmc``
      composition.  ``rfx/grid.py``'s ``_face_pad`` also returns 0 when
      the axis is simply left out of ``cpml_axes``, so such a model is
      refused too -- and that is the route by which 2-D models land here.
      Nothing in-tree builds a 3-D model that way (the api-level
      ``Simulation`` has no ``cpml_axes`` parameter; the route exists for
      callers that build a ``Grid`` themselves), but it is a behaviour
      change and the design note names it.
    * a gap this check does NOT close: it reads ``grid.cpml_layers`` for
      every face, matching the kernel, which also ignores per-face
      ``grid.face_layers`` (``rfx/grid.py``).  A model with
      ``face_layers`` ``x_lo=6`` / ``x_hi=10`` is therefore ADMITTED here
      (both pads are non-zero) while the lane drives ``cpml_layers`` deep
      at both faces regardless.  Unmeasured, and it hands to lane B with
      the phantom window: the same ``grid.face_pads`` gating that fixes
      one should read ``face_layers`` for the depth.

    Parameters
    ----------
    cpml_layers : int
        ``grid.cpml_layers`` -- the depth the lane will drive at EVERY
        face.  Only called when it is > 0 and the boundary is CPML.
    face_pads : tuple of six int
        ``grid.face_pads`` -- ``(pad_x_lo, pad_x_hi, pad_y_lo, pad_y_hi,
        pad_z_lo, pad_z_hi)``: the absorber padding outside the requested
        domain on each face, 0 exactly when that face is ``pec`` / ``pmc``
        / ``periodic`` or its axis is not in ``cpml_axes``.  This is the
        attribute the runner fails to read, so it is the attribute the
        check reads.
    n_devices : int
        Reported in the message only; the defect does not depend on it
        (measured identical at 1 and 2 devices, on the x face and on the
        z face).

    Raises
    ------
    ValueError
        Naming the face(s), the depth, and the ways forward.
    """
    _FACES = ("x-lo", "x-hi", "y-lo", "y-hi", "z-lo", "z-hi")
    if len(face_pads) != len(_FACES):
        raise ValueError(
            f"face_pads must be the six-tuple grid.face_pads, got "
            f"{len(face_pads)} entries: {face_pads!r}")
    off = [face for face, pad in zip(_FACES, face_pads) if pad == 0]
    if not off:
        return
    _faces = " and ".join(off) if len(off) < 3 else (
        ", ".join(off[:-1]) + " and " + off[-1])
    _plural = "faces" if len(off) > 1 else "face"
    _pads = ", ".join(
        f"grid.pad_{f.replace('-', '_')}={p}"
        for f, p in zip(_FACES, face_pads))
    raise ValueError(
        f"{_plural} {_faces} declare(s) no CPML absorber but the {lane} "
        f"path drives a {cpml_layers}-layer CPML window there anyway: the "
        "distributed lane applies EVERY CPML face window whenever "
        "boundary='cpml' and cpml_layers>0 and never reads "
        "grid.face_pads (_init_cpml_distributed builds ONE scalar "
        "_cpml_profile and the kernel applies it at x-lo/x-hi on the "
        "owning ranks and at y-lo/y-hi/z-lo/z-hi unconditionally), and "
        "its per-face E coefficient dt/(eps_r*EPS_0) is non-zero at a "
        "reflector, so the lane ABSORBS at a face you asked to reflect "
        f"({_pads}, grid.cpml_layers={cpml_layers}, "
        f"n_devices={n_devices}). Measured on main before this refusal, 2 "
        "CPU devices, 0 warnings every time -- x faces: "
        "x_lo='cpml'/x_hi='pec' with y/z CPML, cpml_layers=8, 24x8x8 mm "
        "at dx=1 mm, field source (6, 6, 6) mm, Ez probes at x=12/22 mm, "
        "60 steps gave max|dEz| 1.610351e-04 on a 3.165971e-03 row peak = "
        "5.09e-02 of peak, with the x=22 mm probe 99.46% wrong on its own "
        "peak (7.89e-02 of peak at 100 steps); x='pec'/'pec' with y/z "
        "CPML at cpml_layers=8, dx=1 mm, y/z 8 mm, 60 steps, source at "
        "the domain centre put the FACE probes 2 mm inside each x face at "
        "99.9986%/99.9319% (15 mm, nx=16, probes x=2/7/13 mm), "
        "99.9917%/99.8366% (19 mm, nx=20, probes 2/9/17 mm) and "
        "99.9424%/99.4168% (39 mm, nx=40, probes 2/19/37 mm) of their own "
        "peaks; x_lo='pmc'/x_hi='cpml' at dx=5 mm, 16x8x8 cells was 54.08% "
        "of peak at 30 steps and 93.48% at 80. y/z faces, same defect, "
        "measured at 24x8x8 mm, dx=1 mm, cpml_layers=8, field Ez source "
        "(6, 4, 4) mm, Ez probes x=6/12/20 mm, 60 steps: "
        "z=Boundary(lo='pec', hi='cpml') with x/y CPML is 90.58% wrong at "
        "the source probe (max|dEz| 4.003862e+00 on 4.420338e+00) and "
        "409%/281% of their own peaks at x=12/20 mm; z=('pec','pec') "
        "reads max|dEz| 2.615769e+00 against a 5.171041e-03 probe peak; "
        "y=('pmc','cpml') is 5.2x its x=12 mm probe peak. The SYMMETRIC "
        "control on that fixture -- all six faces absorbing -- is "
        "1.078195e-07 of peak. Two of these were re-measured through the "
        "v1 pmap runner at n_devices=1 -- the x=(cpml,pec) 24 mm row and "
        "the z=(pec,cpml) row -- and both are wrong by bit-for-bit the "
        "same amount as at 2 devices, so this is the window and not the "
        "decomposition. Ways forward: make ALL "
        "SIX faces absorbing (boundary='cpml' with x, y and z all in "
        "cpml_axes -- NOT just the x faces: a model with x='cpml' and a "
        "pec/pmc y or z face is the 90.58% row above), or drop the "
        "absorber entirely (boundary='pec' / cpml_layers=0, which stops "
        "the lane building any CPML window), or omit devices=... : the "
        "single-device run() lane clamps each face's CPML profile to that "
        "face's allocated pad (rfx/boundaries/cpml.py) and so realizes "
        "the reflector -- it is the reference every number above is "
        "measured against, and it is the only way forward for a model "
        "that CANNOT make all six absorbing, which includes every 2-D "
        "model (mode='2d_*' drops z from cpml_axes, so pad_z_lo = "
        "pad_z_hi = 0 by construction; on main such a model died inside "
        "XLA with 'Incompatible types for broadcasting: "
        "float32[23,25,8] and float32[23,25,1]', naming no feature)."
    )


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
    from rfx.materials.thin_conductor import refuse_f0_sheets as _refuse_f0
    _refuse_f0(sim._thin_conductors, "distributed (v2) runner")
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

    # ---- Admission gate (B0): the four position-independent classes ----
    # After the two FALLBACKS above on purpose: TFSF and waveguide ports
    # run the whole model on one device, which is a right answer rather
    # than a silently wrong one, and that single-device lane honours
    # periodic axes, wire ports, excite=False and the monitors. Before the
    # n_devices == 1 delegation below on purpose too: the pmap runner in
    # rfx/runners/distributed.py carries the same four gaps (its
    # `impedance > 0.0 and extent is None` / `elif impedance == 0.0` pair
    # is the same port fork; "excite", "flux" and "ntff" do not occur there
    # either).
    refuse_unsupported_distributed_features(
        sim, lane="distributed (v2) runner")

    from rfx.api import Result

    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)

    # Resolve PMC faces (T8, 2026-04). ``BoundarySpec.pmc_faces()`` returns
    # a set; freeze it so it is safely closed-over by the traced scan
    # bodies below.
    _pmc_faces_frozen = frozenset(
        sim._boundary_spec.pmc_faces()
        if getattr(sim, "_boundary_spec", None) is not None
        else ()
    )

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
    # Phase B: non-uniform grid detection. When any axis profile is
    # present on ``sim`` we build a NonUniformGrid and route updates
    # through the NU kernels in ``distributed_nu.py``.
    is_nu = (
        getattr(sim, "_dz_profile", None) is not None
        or getattr(sim, "_dx_profile", None) is not None
        or getattr(sim, "_dy_profile", None) is not None
    )
    # PRE-EXISTING GAP, not a #931 regression (verified 2026-09-07): this
    # lane assembles ``pec_mask`` and never applies it. Its step body only
    # calls the DOMAIN-FACE PEC (``_apply_pec_shmap``); no geometry PEC —
    # volume, sheet or wire — reaches the field update here. #931 did not
    # introduce this and does not fix it; threading sheets in would only
    # make the drop harder to see.
    #
    # What #931 DID introduce was a refusal that named "redraw it as a
    # volume" as the remedy — advice that on this lane produces a run with
    # the metal still missing and no sign of it (measured: a two-device run
    # probing inside a declared PEC Box returns a trace bit-identical to
    # the same model with the Box deleted). Since the drop is the same for
    # all three kinds, the refusal is the same for all three: a declared
    # PEC volume is refused here too rather than solved away.
    _d_pec_sheets: list = []
    _d_pec_wires: list = []
    if is_nu:
        # dz-profile synthesis happens locally inside
        # _build_nonuniform_grid() — no sim-state mutation here.
        grid = sim._build_nonuniform_grid()
        base_materials, debye_spec, lorentz_spec, pec_mask = (
            sim._assemble_materials_nu(grid, pec_sheets=_d_pec_sheets,
                                       pec_wires=_d_pec_wires)
        )
        pec_shapes = None
    else:
        grid = sim._build_grid()
        base_materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, *_ = (
            sim._assemble_materials(grid, pec_sheets=_d_pec_sheets,
                                    pec_wires=_d_pec_wires)
        )
    _d_pec_volume = (pec_mask is not None
                     and not is_tracer(pec_mask)
                     and bool(jnp.any(pec_mask)))
    if _d_pec_sheets or _d_pec_wires or _d_pec_volume:
        _d_declared = []
        if _d_pec_sheets:
            _d_declared.append(f"{len(_d_pec_sheets)} PEC sheet(s)")
        if _d_pec_wires:
            _d_declared.append(f"{len(_d_pec_wires)} sub-cell wire(s)")
        if _d_pec_volume:
            _d_declared.append(
                f"a PEC volume of {int(jnp.sum(pec_mask))} cell(s)")
        raise NotImplementedError(
            "run_distributed_v2() does not realize declared PEC geometry of "
            "ANY kind (#931): this lane carries geometry PEC only as a cell "
            "mask, its step body applies domain-face PEC alone, and a sheet "
            "or a wire owns no cell to begin with. Declared here: "
            f"{', '.join(_d_declared)} — all of it would be absent from "
            "every rank with no sign of it. Redrawing a sheet as a volume "
            "does NOT help on this lane (measured: a probe inside a declared "
            "PEC Box reads a trace bit-identical to empty geometry). Use "
            "sim.run() without devices=, which realizes all three, or model "
            "the conductor as a sigma fill, which rides in the material "
            "arrays this lane does shard.")
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

    # ---- Admission gate (B0) class 5: the x absorber must fit one rank ----
    # Here and not at the API boundary because it is position-dependent:
    # nx_per / pad_x / ghost are only known now, and this is the last point
    # before the state is split into slabs and sharded.
    if use_cpml:
        check_x_absorber_fits_ranks(
            nx=nx, n_devices=n_devices, nx_per=nx_per, pad_x=pad_x,
            ghost=ghost, cpml_layers=n_cpml,
            pad_x_lo=getattr(grid, "pad_x_lo", None),
            pad_x_hi=getattr(grid, "pad_x_hi", None),
            lane="distributed (v2) runner",
        )
        # ---- class 6: the phantom window at a non-absorbing face ----
        # AFTER class 5 on purpose: when both apply (the asymmetric
        # fixture at a too-small nx_per) the arithmetic message is the
        # more specific one, and it is the one the class-5 tests pin.
        # Class 6 is what catches the configurations class 5 ADMITS and
        # that are wrong anyway -- measured 46% one cell inside the
        # class-5 bound, and identical at n_devices=1.
        # Round 4 widened it from the two x faces to all six:
        # _init_cpml_distributed builds ONE scalar profile and the kernel
        # applies it at y-lo/y-hi/z-lo/z-hi unconditionally too, so a
        # pec/pmc y or z face composed with boundary='cpml' carries the
        # same phantom window. MEASURED on origin/main 883615c6 (identical
        # on the base 7b511591), 24x8x8 mm at dx=1 mm, cpml_layers=8,
        # field Ez source (6, 4, 4) mm, probes x=6/12/20 mm, 60 steps, 2
        # devices, 0 warnings: z=Boundary(lo='pec', hi='cpml') with x/y
        # CPML is 90.58% wrong at the source probe (max|dEz| 4.003862e+00
        # on 4.420338e+00) and 409%/281% of their own peaks at x=12/20 mm;
        # the symmetric control on the same fixture is 1.078195e-07 of
        # peak. grid.face_pads, not the two x pads, is therefore what is
        # passed -- it is also the attribute the runner fails to read.
        check_absorber_faces_are_absorbing(
            cpml_layers=n_cpml, face_pads=grid.face_pads,
            n_devices=n_devices, lane="distributed (v2) runner",
        )

    # Phase B: NU + CPML + distributed is not implemented yet.
    # The api.py-level guardrail already blocks this combination; assert
    # here as a defensive backstop in case a future caller bypasses it.
    if is_nu and use_cpml:
        raise NotImplementedError(
            "Phase B supports non-uniform grids on the distributed path "
            "with PEC boundaries only. boundary='cpml' with dx/dy/dz "
            "profile and devices>1 is a Phase C item."
        )
    if is_nu and debye_spec is not None:
        raise NotImplementedError(
            "Phase B does not support Debye dispersion on the NU "
            "distributed path yet."
        )
    if is_nu and lorentz_spec is not None:
        raise NotImplementedError(
            "Phase B does not support Lorentz dispersion on the NU "
            "distributed path yet."
        )

    dt = grid.dt
    dx = grid.dx

    # Phase B: pre-computed per-device inv-dx arrays for NU path.
    # Inner kernels receive the full-axis inv_dy/inv_dz replicated and
    # the per-device slab of inv_dx / inv_dx_h (length nx_local = nx_per
    # + 2*ghost constructed below).
    if is_nu:
        from rfx.runners.distributed_nu import (
            _build_sharded_inv_dx_arrays,
            _update_h_local_nu,
            _update_e_local_nu,
            split_1d_with_ghost as _split_1d_with_ghost_helper,
        )
        inv_dx_global, inv_dx_h_global, _dx_padded = (
            _build_sharded_inv_dx_arrays(grid, n_devices, pad_x=pad_x)
        )
        inv_dy_full = np.asarray(grid.inv_dy, dtype=np.float32)
        inv_dy_h_full = np.asarray(grid.inv_dy_h, dtype=np.float32)
        inv_dz_full = np.asarray(grid.inv_dz, dtype=np.float32)
        inv_dz_h_full = np.asarray(grid.inv_dz_h, dtype=np.float32)
    else:
        inv_dx_global = inv_dx_h_global = None
        inv_dy_full = inv_dy_h_full = None
        inv_dz_full = inv_dz_h_full = None

    # ------------------------------------------------------------------
    # Create mesh
    # ------------------------------------------------------------------
    mesh = _make_mesh(devices)

    # ------------------------------------------------------------------
    # Build sources and probes
    # ------------------------------------------------------------------
    sources = []
    probes = []
    if is_nu:
        # NU source/probe build uses cumulative position lookup and the
        # same dV-normalised current-source waveform as the single-device
        # NU runner (rfx.nonuniform.make_current_source) so distributed
        # and single-device results agree.
        from rfx.nonuniform import (
            position_to_index as _nu_pos_to_idx,
            make_current_source as _nu_make_current_source,
        )
        from rfx.simulation import SourceSpec, ProbeSpec
        for pe in sim._ports:
            if pe.impedance > 0.0:
                raise NotImplementedError(
                    "Phase B distributed+NU does not support lumped / wire "
                    "ports yet. Use single-device for ports on NU meshes."
                )
            if pe.impedance == 0.0:
                idx = _nu_pos_to_idx(grid, pe.position)
                si, sj, sk, sc, wf = _nu_make_current_source(
                    grid, idx, pe.component, pe.waveform, n_steps, materials,
                    amplitude_kind=pe.amplitude_kind)
                sources.append(SourceSpec(
                    i=int(si), j=int(sj), k=int(sk),
                    component=sc, waveform=jnp.asarray(wf),
                ))
        for pe in sim._probes:
            idx = _nu_pos_to_idx(grid, pe.position)
            probes.append(ProbeSpec(
                i=int(idx[0]), j=int(idx[1]), k=int(idx[2]),
                component=pe.component,
            ))
    else:
        for pe in sim._ports:
            if pe.impedance > 0.0 and pe.extent is None:
                lp = LumpedPort(
                    position=pe.position, component=pe.component,
                    impedance=pe.impedance, excitation=pe.waveform,
                )
                materials = setup_lumped_port(grid, lp, materials)
                sources.append(make_port_source(grid, lp, materials, n_steps))
            elif pe.impedance == 0.0:
                # issue #571: thread amplitude_kind; helper stays boundary-selected.
                if sim._boundary == "cpml":
                    sources.append(make_j_source(grid, pe.position, pe.component,
                                                 pe.waveform, n_steps, materials,
                                                 amplitude_kind=pe.amplitude_kind))
                else:
                    sources.append(make_source(grid, pe.position, pe.component,
                                               pe.waveform, n_steps,
                                               materials=materials,
                                               amplitude_kind=pe.amplitude_kind))
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

    # Phase B: shard per-device inv_dx / inv_dx_h slabs for NU updates.
    if is_nu:
        # Split the 1-D padded global arrays into per-device slabs with
        # ghost cells, matching the field slabbing convention.
        # inv_dx uses cell-local values (ghost cells at the domain
        # boundary carry the neighbour-cell value to avoid divide-by-zero
        # in update kernels); inv_dx_h uses mean-spacing values which
        # straddle adjacent cells.
        _inv_dx_slabs = _split_1d_with_ghost_helper(
            inv_dx_global, n_devices, nx_per, nx_local, ghost, pad_value=1.0)
        _inv_dx_h_slabs = _split_1d_with_ghost_helper(
            inv_dx_h_global, n_devices, nx_per, nx_local, ghost, pad_value=0.0)
        # Shard to (n_devices * nx_local,) along P("x")
        inv_dx_sharded = jax.device_put(
            _inv_dx_slabs.reshape(n_devices * nx_local), shd)
        inv_dx_h_sharded = jax.device_put(
            _inv_dx_h_slabs.reshape(n_devices * nx_local), shd)
        inv_dy_rep = jax.device_put(inv_dy_full, rep)
        inv_dy_h_rep = jax.device_put(inv_dy_h_full, rep)
        inv_dz_rep = jax.device_put(inv_dz_full, rep)
        inv_dz_h_rep = jax.device_put(inv_dz_h_full, rep)
    else:
        inv_dx_sharded = inv_dx_h_sharded = None
        inv_dy_rep = inv_dy_h_rep = None
        inv_dz_rep = inv_dz_h_rep = None

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
        if is_nu:
            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(
                    P("x"), P("x"), P("x"),  # ex, ey, ez
                    P("x"), P("x"), P("x"),  # hx, hy, hz
                    P(),                     # step
                    P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
                    P("x"), P(None), P(None),  # inv_dx, inv_dy, inv_dz
                    P("x"), P(None), P(None),  # inv_dx_h, inv_dy_h, inv_dz_h
                ),
                out_specs=(P("x"), P("x"), P("x"), P()),
                check_rep=False,
            )
            def _h_nu(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
                      invdx, invdy, invdz, invdxh, invdyh, invdzh):
                _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
                _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
                new_st = _update_h_local_nu(
                    _st, _mat, dt,
                    invdx, invdy, invdz, invdxh, invdyh, invdzh)
                return new_st.hx, new_st.hy, new_st.hz, new_st.step

            hx, hy, hz, step = _h_nu(
                st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
                mat.eps_r, mat.sigma, mat.mu_r,
                inv_dx_sharded, inv_dy_rep, inv_dz_rep,
                inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep)
            return st._replace(hx=hx, hy=hy, hz=hz, step=step)

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
        if is_nu:
            # NU path: no dispersion (blocked upstream).
            @partial(
                shard_map,
                mesh=mesh,
                in_specs=(
                    P("x"), P("x"), P("x"),
                    P("x"), P("x"), P("x"),
                    P(),
                    P("x"), P("x"), P("x"),
                    P("x"), P(None), P(None),
                ),
                out_specs=(P("x"), P("x"), P("x"), P()),
                check_rep=False,
            )
            def _e_nu(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
                      invdx, invdy, invdz):
                _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
                _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
                new_st = _update_e_local_nu(_st, _mat, dt, invdx, invdy, invdz)
                return new_st.ex, new_st.ey, new_st.ez, new_st.step

            ex, ey, ez, step = _e_nu(
                st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
                mat.eps_r, mat.sigma, mat.mu_r,
                inv_dx_sharded, inv_dy_rep, inv_dz_rep)
            new_st = st._replace(ex=ex, ey=ey, ez=ez, step=step)
            # db_st / lr_st are passthrough (dummies) in NU path.
            return new_st, db_st, lr_st

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

        # 2. CPML H correction (material-aware, #205)
        st, cpml_st = _apply_cpml_h_shmap(
            st, cpml_params, cpml_st, n_cpml, dt, dx,
            mesh, n_devices, ghost=ghost, mu_r=sharded_materials.mu_r,
            pad_x=pad_x)

        # 3. Exchange H ghost cells (conditionally skip)
        do_exchange = (_step_idx % _exchange_interval == 0)
        st = lax.cond(
            do_exchange,
            lambda s: _exchange_h_ghosts_shmap(s, mesh, n_devices),
            lambda s: s,
            st,
        )

        # 3b. PMC face (H-tangential = 0) — T8, 2026-04. H-half hook per
        #     OQ9: after H ghost exchange, before E update. PMC must
        #     fire before the next E-update reads H via curl.
        st = _apply_pmc_shmap(
            st, mesh, n_devices, nx_local, _pmc_faces_frozen, pad_x=pad_x)

        # 4. E update
        st, db_st, lr_st = _update_e_shmap(
            st, sharded_materials,
            debye_coeffs_sharded, db_st,
            lorentz_coeffs_sharded, lr_st)

        # 5. CPML E correction (material-aware, #205)
        st, cpml_st = _apply_cpml_e_shmap(
            st, cpml_params, cpml_st, n_cpml, dt, dx,
            mesh, n_devices, ghost=ghost, eps_r=sharded_materials.eps_r,
            pad_x=pad_x)

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

        # 2b. PMC face (H-tangential = 0) — T8, 2026-04. H-half hook per
        #     OQ9: after H ghost exchange, before E update.
        st = _apply_pmc_shmap(
            st, mesh, n_devices, nx_local, _pmc_faces_frozen, pad_x=pad_x)

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
        st = _apply_pec_shmap(st, mesh, n_devices, nx_local, pad_x=pad_x)

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
        # Stage 1.5b: must remain traceable under jax.grad. The prior
        # ``np.array(sharded_arr)`` host-pull raised
        # TracerArrayConversionError whenever a caller wrapped the runner
        # in ``jax.grad`` to drive an objective from the gathered
        # ``final_state``. Pure JAX reshape + ``gather_array_x`` (already
        # JAX-friendly) keeps the gather in the trace. Mirrors the
        # already-correct ``distributed_nu.py::_unstack_and_gather``.
        total_x = sharded_arr.shape[0]
        assert total_x == n_devices * nx_local
        stacked = jnp.reshape(
            sharded_arr,
            (n_devices, nx_local) + tuple(sharded_arr.shape[1:]),
        )
        gathered = gather_array_x(stacked, ghost)
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
