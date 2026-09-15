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
from rfx.runners._distributed_common import (
    apply_pec_face_shmap,
    apply_pmc_face_shmap,
    exchange_component_shmap,
    inject_sources_shmap,
    sample_probes_shmap,
    shard_stacked,
    shard_stacked_psi,
    unstack_and_gather,
    update_e_nu_shmap,
    update_h_nu_shmap,
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

# #1038 leg 3. ``_apply_pec_shmap``'s body now lives as
# ``apply_pec_face_shmap`` in ``_distributed_common.py`` (the NU runner
# carried a copy that differed only in its parameter spelling and three
# comments). Kept as a module-local alias so the existing call sites are
# unchanged.
_apply_pec_shmap = apply_pec_face_shmap


# #1038 leg 3. ``_apply_pmc_shmap``'s body now lives as
# ``apply_pmc_face_shmap`` in ``_distributed_common.py`` (the NU runner carried
# a byte-identical copy once the slab-length parameter was spelled the same).
# Kept as a module-local alias so the existing call sites are unchanged.
_apply_pmc_shmap = apply_pmc_face_shmap


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
        return shard_stacked_psi(arr, shd)

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
        return shard_stacked(arr, shd)

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
        return inject_sources_shmap(
            st, src_vals_step, mesh, n_src,
            src_local_specs, src_device_ids,
        )

    def _sample_probes_shmap(st):
        """Sample probes on their owning devices, then sum across devices."""
        return sample_probes_shmap(
            st, mesh, n_prb, prb_local_specs, prb_device_ids,
        )

    # ------------------------------------------------------------------
    # H and E local updates via shard_map
    # ------------------------------------------------------------------

    def _update_h_shmap(st, mat):
        if is_nu:
            # #1038 leg 4: this branch held a renamed copy (`_h_nu`) of
            # distributed_nu.py's own `_h` wrapper -- inventory §2.4, 0.884
            # similarity, the def name plus one re-wrapped argument list. One
            # shared body now, in _distributed_common; the uniform branch below
            # and the `is_nu` dispatch itself are untouched.
            return update_h_nu_shmap(
                st, mat, mesh, dt,
                inv_dx_sharded, inv_dy_rep, inv_dz_rep,
                inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep,
            )

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
            # #1038 leg 4: this branch held a renamed copy (`_e_nu`) of
            # distributed_nu.py's own `_e` wrapper -- inventory §2.4, 0.947
            # similarity, the def name and nothing else. One shared body now,
            # in _distributed_common; the `is_nu` dispatch, the dispersive
            # branch below and the (state, db_st, lr_st) return shape are
            # untouched.
            new_st = update_e_nu_shmap(
                st, mat, mesh, dt,
                inv_dx_sharded, inv_dy_rep, inv_dz_rep,
            )
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
        """Single FDTD step (CPML path) operating on sharded arrays.

        Stage order (#1041)::

            H -> CPML-H -> exch H -> PMC face -> E -> CPML-E
              -> sources -> exch E -> probes

        The E ghost exchange is the LAST stage of the E half-step, so a
        ghost row is always a copy of the owner's FINISHED real row. This
        is ``distributed_nu.py``'s order since ``ac782d4f`` (#931 T3) and
        the mirror of the H half, which applies the PMC face before the H
        exchange "so the zero propagates via the exchange". Until #1041
        this body exchanged BEFORE injecting, which lagged a seam-cell
        source by one step in the neighbour's copy -- see the comment on
        stage 7 for the measurement that moved it.
        """
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

        # 6. Source injection
        st = _inject_sources_shmap(st, src_vals)

        # 7. Exchange E ghost cells -- LAST stage of the E half-step, so a
        #    ghost row is a copy of the owner's FINISHED real row (#1041,
        #    the ordering distributed_nu.py took in ac782d4f / #931 T3).
        #    Exchanging before injection left a source at rank d's first
        #    real cell out of rank d-1's right ghost for one step, and
        #    rank d-1's next H update read the pre-injection plane.
        #    Measured by scripts/diagnostics/issue1041_v2_step_order.py on a
        #    seam-cell ez source, 2 ranks, 300 steps, boundary="cpml",
        #    against the SAME model on one device: the probe 4 cells into
        #    the neighbouring rank went from 1.653e-01 relative (first
        #    divergent step 4, the causal arrival) to 1.894e-06 (step 50),
        #    which is the interior-source control's own lane-difference
        #    floor of 1.629e-06 on this geometry.
        #
        #    The defect is ONE-SIDED, which is worth knowing before reading
        #    any lock result. Only the RIGHT E ghost is live: rank d-1's H
        #    at its last REAL cell consumes it. A rank's LEFT E ghost feeds
        #    only its own H at that same index, and the H exchange
        #    (stage 3 / stage 2) overwrites that H with the neighbour's
        #    authoritative value before anything reads it. So a source in
        #    rank d's LAST real cell was never affected -- measured
        #    bit-identical between the two orderings on both bodies -- and
        #    that is exactly where every distributed_v2_* fixture of the
        #    #1038 bit-identity lock puts its source, which is why that lock
        #    stays 13/13 green through this change.
        st = lax.cond(
            do_exchange,
            lambda s: _exchange_e_ghosts_shmap(s, mesh, n_devices),
            lambda s: s,
            st,
        )

        # 8. Probe sampling
        probe_out = _sample_probes_shmap(st)

        return {"fdtd": st, "cpml": cpml_st,
                "debye": db_st, "lorentz": lr_st}, probe_out

    def step_fn_pec(carry, xs):
        """Single FDTD step (PEC path) operating on sharded arrays.

        Stage order (#1041)::

            H -> exch H -> PMC face -> E -> sources -> PEC face
              -> exch E -> probes

        Same invariant as ``step_fn_cpml``: the E ghost exchange is last,
        so a ghost row is a copy of the owner's finished real row. The
        PEC face moved with it for lane parity with ``distributed_nu``;
        on THIS lane that half of the move is provably and measurably
        inert (stage 6 comment), because the only PEC here is the domain
        face.
        """
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

        # 4. Source injection
        st = _inject_sources_shmap(st, src_vals)

        # 5. PEC boundaries (domain faces only -- this lane refuses declared
        #    PEC geometry, see the NotImplementedError in run_distributed).
        #    Injection runs BEFORE the face, matching distributed_nu stage 7.
        #    The two lanes therefore both differ from the single-device lane,
        #    which applies the faces before its soft-source loop; the
        #    difference is observable only for a source placed ON a domain
        #    PEC face, where the tangential E is zeroed in the same step it
        #    is injected. #1041 measured no such fixture and did not change
        #    it -- a source on a PEC face is its own question.
        st = _apply_pec_shmap(st, mesh, n_devices, nx_local, pad_x=pad_x)

        # 6. Exchange E ghost cells -- LAST stage of the E half-step, so a
        #    ghost row is a copy of the owner's FINISHED real row (#1041,
        #    the ordering distributed_nu.py took in ac782d4f / #931 T3).
        #    Same measurement as step_fn_cpml above, boundary="pec": the
        #    probe 4 cells into the neighbouring rank went from 1.859e-01
        #    relative (first divergent step 4) to 4.858e-06 (step 43),
        #    against an interior-source floor of 5.100e-06. Same one-sided
        #    reachability as step_fn_cpml: a source in rank d's LAST real
        #    cell is unaffected either way (measured bit-identical).
        #    The PEC half of the move is inert on THIS lane and was measured
        #    to be so (#1041 fixture P, bit-identical): apply_pec_face_shmap
        #    zeroes the y/z faces on every x row INCLUDING the ghosts, and
        #    its x_lo / x_hi faces act on rank 0's first and rank N-1's last
        #    real cell, whose exchanged copies the receiving rank discards.
        st = lax.cond(
            do_exchange,
            lambda s: _exchange_e_ghosts_shmap(s, mesh, n_devices),
            lambda s: s,
            st,
        )

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
        # already-correct ``distributed_nu.py::_unstack_and_gather`` -- as of
        # #1038 leg 2b it IS that function, shared.
        return unstack_and_gather(
            sharded_arr, n_devices, nx_local, ghost, pad_x, nx,
        )

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
