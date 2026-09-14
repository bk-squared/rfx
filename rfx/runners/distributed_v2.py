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

Declared PEC (#1053): a PEC VOLUME is realized here.  ``pec_mask`` is
sharded with the material arrays and applied in both step bodies after
source injection and immediately before the E ghost exchange, which is
``distributed_nu``'s stage 8 at the #1041 ordering.  Declared SHEETS and
sub-cell WIRES own no cell, this lane has no other carrier for them, and
they are refused rather than silently dropped.  Note the single-device fast
path below delegates to the pmap runner, which still drops the mask and so
still refuses a volume (#1055); ``Simulation.run(devices=...)`` never reaches
it, because ``rfx/api/_execute.py`` routes here only for ``len(devices) > 1``.

Known limitations (transparent single-device fallback):
- TFSF plane-wave sources: require full-domain field injection, not
  compatible with slab decomposition.  Detected and warned at runtime.
- Waveguide / Floquet ports: modal decomposition needs the full
  cross-section on one device.  Detected and warned at runtime.
- Non-divisible nx: automatically padded to nearest multiple of n_devices
  with PEC-filled cells, then trimmed after gathering.
- Declared PEC sheets and sub-cell wires: refused (see above).

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
from rfx.runners._distributed_common import (
    apply_pec_face_shmap,
    apply_pec_mask_shmap,
    apply_pmc_face_shmap,
    exchange_component_shmap,
    inject_sources_shmap,
    sample_probes_shmap,
    shard_stacked,
    shard_stacked_psi,
    split_array_x,
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
# :func:`run_distributed` once the slab arithmetic is known.
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
       in the file at all, and on this branch its only three occurrences
       are inside this gate (the read below, this sentence and the refusal
       message), pinned by
       ``test_the_runner_still_never_reads_periodic_axes``.  A periodic
       axis therefore became an open ghost-coupled axis with nothing said.
    2. **Extended lumped ports** (``impedance > 0`` with ``extent`` set).
       ``distributed_v2.py`` forks on ``impedance > 0.0 and extent is
       None`` then ``elif impedance == 0.0``; a wire port satisfies
       neither, so it got no source, no resistive termination and no
       error.  ``rfx/runners/distributed.py:1414-1422`` (the pmap twin,
       which is also the ``n_devices == 1`` delegate) has the same fork.
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
            "does not appear in the file at all), so the axis is solved as "
            "an OPEN ghost-coupled axis with nothing said (measured on main "
            "before this refusal: periodic_axes='y' gave a distributed "
            "trace 1.96e-04 away from the native periodic trace on a "
            "1.09e-03 peak -- 18% of peak -- with no error and no warning). "
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
            "rfx/runners/distributed_v2.py, and its twin at "
            "rfx/runners/distributed.py:1414-1422), and a port with a "
            "positive impedance AND an extent satisfies NEITHER -- it gets "
            "no source, no resistive termination and no error. Declared "
            f"here: {_detail}. Measured on main before this refusal: the "
            "distributed probe energy was exactly 0.0 where the native run "
            "read 9.95e-04 (peak |Ez| 1.41e-02). Use a single-cell lumped "
            "port (drop extent=), or omit devices=... : the single-device "
            "run() lane realizes wire ports "
            "(rfx/runners/uniform.py:420, setup_wire_port)."
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
    """Refuse an x CPML absorber that does not fit inside one rank's slab.

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
    own slab, so the windows stay inside owned cells exactly while::

        n <= nx_per           (x-lo face)
        n <= nx_per - pad_x   (x-hi face)

    Past that the window slides into the ghost halo.  ONE cell of overflow
    keeps the window's LENGTH at ``n``, so every shape still matches,
    nothing raises, and the absorber updates a halo cell instead of the
    owned cell it should.

    MEASURED on pristine main (d56f68eb, 2026-09-14, 2 virtual CPU
    devices, 60 steps).  The measuring fixture is stated in full because
    the digits do not survive a change of source or probe placement; it is
    ``_asym(8, 6)`` in
    ``tests/unit/runners/test_distributed_admission_refusals.py``, i.e.
    ``BoundarySpec(x=(cpml, pec), y=(cpml, cpml), z=(cpml, cpml))``,
    ``cpml_layers=8``, domain 6x8x8 mm at dx=1 mm (so nx=15, pad_x=1,
    nx_per=8), an Ez soft source at (3, 4, 4) mm with
    ``amplitude_kind='field'``, and an Ez probe ROW at x = 1, 3, 5 and
    6 mm (y = z = 4 mm)::

        native peak |Ez|                4.416633e+00   (probe x=3 mm)
        distributed vs native max|dEz|  2.207244e+00   (49.98% of peak)
        probe x=5 mm (one cell inside
          the x-hi PEC face)            1.646758e-01 wrong on its own
                                        1.646768e-01 peak -- 99.9994%
        warnings raised                 0

    The probe ON the x-hi face (x=6 mm) reads exactly 0.0 on both lanes --
    it sits on the PEC plane -- so the "99.9994%" probe is the one a cell
    inside it, not the face probe.  The same fixture with the domain
    widened to 24x8x8 mm (nx=33, pad_x=1, nx_per=17, so the absorber
    fits) is at max|dEz| = 1.005828e-06 on a 4.422700e+00 peak =
    **2.274240e-07 of peak**, i.e. parity; that is the negative control,
    and it is the same fixture with one number changed.

    TWO or more cells of overflow run past the slab end, the clipped
    window is shorter than the psi arrays, and XLA raises a broadcasting
    error that names no feature.  That quote also needs its fixture: with
    the SAME boundary spec but ``cpml_layers=20`` and domain 7x12x12 mm at
    dx=1 mm (nx=28, pad_x=0, nx_per=14, ny=nz=53) main dies with "mul got
    incompatible shapes for broadcasting: (20, 1, 1), (15, 53, 53)" -- the
    ``(20, 1, 1)`` is the 20-layer psi profile and the ``15`` is the
    clipped window ``nx_local - (ghost + pad_x)``.

    The condition is on ``sim._boundary == "cpml"`` and
    ``grid.cpml_layers > 0``, NOT on the x faces actually being CPML, and
    that is deliberate: ``_apply_cpml_e_distributed`` (and its H twin)
    applies BOTH x-face windows unconditionally, without consulting
    ``grid.face_pads`` -- the S5 gap the direction note records.  So
    ``BoundarySpec(x=('pec', 'pec'), y=('cpml', 'cpml'), z=('cpml',
    'cpml'))`` with ``cpml_layers=8`` still drives an 8-layer x window into
    a slab with no x pad at all, and on pristine main (d56f68eb, 2
    devices) that died in XLA -- 6x8x8 mm at dx=1 mm gave "mul got
    incompatible shapes for broadcasting: (8, 1, 1), (5, 25, 25)" and
    8x8x8 mm gave "(8, 1, 1), (6, 25, 25)".  Refusing it is therefore not
    an over-refusal of a working case.  But the caller declared NO x
    absorber, so ``pad_x_lo`` / ``pad_x_hi`` are read here for the sole
    purpose of saying so in the message instead of blaming an absorber
    that was never asked for; the arithmetic itself is unchanged.  Once
    lane B closes S5 (x windows gated on ``face_pads``), this condition
    should be narrowed to the faces that really are CPML.

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
    faces = []
    if cpml_layers > nx_per:
        faces.append(
            f"x-lo needs cells [{ghost}, {ghost + cpml_layers}) of rank 0's "
            f"slab but rank 0 owns [{ghost}, {ghost + nx_per})"
        )
    if cpml_layers > nx_per - pad_x:
        _start = nx_per + ghost - pad_x - cpml_layers
        # The last rank's REAL cells stop pad_x short of its owned slab:
        # the trailing pad_x cells are #623 PEC-fill alignment cells that
        # sit past the physical x-hi face, which is exactly why the x-hi
        # bound is nx_per - pad_x and not nx_per.
        faces.append(
            f"x-hi needs cells [{_start}, {nx_per + ghost - pad_x}) of rank "
            f"{n_devices - 1}'s slab but that rank owns only "
            f"[{ghost}, {ghost + nx_per - pad_x}) real cells "
            f"(its last {pad_x} cell(s) of [{ghost}, {ghost + nx_per}) are "
            "#623 PEC-fill alignment cells past the x-hi face)"
        )
    if not faces:
        return
    # The largest FEWER device count that does fit, searched exactly rather
    # than estimated as nx // cpml_layers (pad_x is not monotone in N).
    # n=1 is in this list whenever cpml_layers <= nx, i.e. for every real
    # grid, so the "no count fits" branch below is effectively unreachable
    # and fits == [1] is the common case -- and "use n_devices=1" is just
    # "omit devices=" said twice, so it gets the single-device wording
    # instead of a device-count recommendation.
    fits = [
        n for n in range(2, n_devices)
        if cpml_layers <= ((nx + (-nx) % n) // n) - ((-nx) % n)
    ]
    remedy = (
        f"Use fewer devices (n_devices={max(fits)} is the largest count "
        f"above 1 that fits at nx={nx} and cpml_layers={cpml_layers})"
        if fits else
        f"No device count above 1 fits an absorber this deep at nx={nx}: "
        f"reduce cpml_layers (<= {max(nx_per - pad_x, 0)} fits at "
        f"n_devices={n_devices}) or increase nx"
    )
    # The caller may have declared no x absorber at all; say so rather
    # than blaming one (the runner applies the x windows regardless -- S5).
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
                "read grid.face_pads (rfx/runners/distributed.py, the "
                "xlo/xhi window slices). So the depth that has to fit is "
                f"cpml_layers={cpml_layers} on both x faces even though you "
                "asked for an absorber on neither; on main this same "
                "configuration died inside XLA with a broadcasting error "
                "that named no feature."
            )
    raise ValueError(
        f"the x CPML absorber does not fit inside one rank's slab on the "
        f"{lane} path: cpml_layers={cpml_layers} with nx={nx}, "
        f"n_devices={n_devices} gives nx_per={nx_per} owned cells per rank "
        f"(pad_x={pad_x} alignment cell(s) on the last rank, ghost={ghost}) "
        f"-- {'; '.join(faces)}. The distributed CPML window would reach "
        "into the ghost halo, which either silently absorbs in the wrong "
        "cell (measured on main, 2 devices, 60 steps, x_lo='cpml'/"
        "x_hi='pec', 6x8x8 mm at dx=1 mm, cpml_layers=8, field source at "
        "(3, 4, 4) mm, Ez probes at x=1/3/5/6 mm: max|dEz| 2.207244 on a "
        "4.416633 peak = 49.98%, and the probe one cell inside the x-hi "
        "face wrong by 99.9994% of its own peak, with no warning) or dies "
        f"in XLA with a broadcasting error that names no feature. {remedy}, "
        "or omit devices=... entirely (the single-device run() lane has no "
        f"slab to overflow).{_declared}"
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
    # rfx/runners/distributed.py carries the same four gaps (:1414-1422 is
    # the same port fork; "excite", "flux" and "ntff" do not occur there
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
        # This lane realizes a declared PEC volume (#1053) but the pmap
        # runner does not, so a volume is refused HERE and runs at
        # n_devices >= 2. Not reachable from Simulation.run(devices=...),
        # which dispatches here only for len(devices) > 1; a direct caller
        # passing one device gets the pmap refusal, which is honest about
        # that runner (#1055).
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
    # Declared PEC on this lane, after #1053: VOLUMES are realized, SHEETS
    # and WIRES are not.
    #
    # A volume is carried as ``pec_mask``, which #1053 legs 1-2 shard next
    # to the material arrays and apply in both step bodies through
    # ``apply_pec_mask_shmap`` — after source injection and the domain
    # faces, immediately before the E ghost exchange, i.e. distributed_nu's
    # stage 8 at the #1041 ordering. Gated end-to-end against the
    # single-device lane by tests/unit/runners/
    # test_distributed_v2_pec_body_seam.py (three bodies; seam-face body
    # 3.074e-05 relative against a 1e-3 gate, 3.203e-01 with the stage one
    # step late).
    #
    # A sheet and a sub-cell wire own no cell, so the mask carries neither.
    # Nothing else on this lane does either: the step body's only other PEC
    # is the DOMAIN FACE (``_apply_pec_shmap``). Declaring one here would
    # put metal in the model that is absent from every rank with no sign of
    # it, so both are still refused.
    #
    # The pre-#1053 refusal covered volumes too, because the mask was
    # assembled and then dropped (measured then: a two-device run probing
    # inside a declared PEC Box returned a trace bit-identical to the same
    # model with the Box deleted). It also had to warn that redrawing a
    # sheet as a volume did NOT help. Both statements are now false HERE
    # and both stay true of the pmap lane in ``distributed.py``, whose copy
    # of the message is a separate string and is out of #1053 scope (#1055).
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
    if _d_pec_sheets or _d_pec_wires:
        _d_declared = []
        if _d_pec_sheets:
            _d_declared.append(f"{len(_d_pec_sheets)} PEC sheet(s)")
        if _d_pec_wires:
            _d_declared.append(f"{len(_d_pec_wires)} sub-cell wire(s)")
        raise NotImplementedError(
            "run_distributed_v2() does not realize declared PEC SHEETS or "
            "sub-cell WIRES: this lane carries geometry PEC only as a cell "
            "mask, and a sheet or a wire owns no cell to begin with, so it "
            "would be absent from every rank with no sign of it. Declared "
            f"here: {', '.join(_d_declared)}. A declared PEC VOLUME is a "
            "different case and DOES run here: since #1053 this lane shards "
            "pec_mask and applies it in both step bodies, at the #1041 "
            "ordering. So redrawing a sheet as a volume does realize metal "
            "on this lane — but a volume is not the same object as a sheet "
            "(it shorts the normal E edge between its two faces, #690), so "
            "redraw only if that is the conductor you meant. Otherwise use "
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

    # #1053 leg 1: carry the realized-PEC cell mask into the sharded world.
    # ``pec_mask`` is the full-domain primal-cell occupancy of every declared
    # PEC VOLUME, already padded to ``nx_padded`` above (with ``True``, the
    # high-x alignment convention ``shard_pec_mask_x_slab`` also uses). Until
    # now this lane built it and dropped it on the floor.
    #
    # ``pad_value=False`` fills the PHYSICAL-boundary ghost rows -- device 0's
    # left ghost and device N-1's right ghost. It must be False and not True:
    # a True there puts a spurious PEC wall on the whole x_lo / x_hi node
    # plane of the outer ranks and shorts a CPML face (#689/#931, recorded at
    # distributed_nu.py:538-547). Interior ghost rows get the seam
    # neighbour's real value from the slicing, which is what lets a rank's
    # first/last real cell see its true x neighbour under the four-incident-
    # cell rule.
    #
    # ``split_array_x`` + ``shard_stacked`` reproduce
    # ``distributed_nu.shard_pec_mask_x_slab`` BIT-IDENTICALLY -- verified on
    # (nx, pad_x) in {(8,0), (7,1), (16,0)}, ny=nz=6, n_devices=2, random
    # masks, jnp.array_equal True on every case. So this lane needs no sharder
    # of its own, and the two lanes cannot disagree about slab layout.
    pec_mask_slabs = (
        None if pec_mask is None
        else split_array_x(pec_mask, n_devices, ghost, pad_value=False))

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

    # #1053 leg 1. ``None`` whenever the model declares no PEC volume, which
    # is every fixture of the #1038 bit-identity lock -- so the stage leg 2
    # hooks on this is a no-op branch there and the lock stays 15/15. The step
    # bodies read this as a CLOSURE VARIABLE, next to ``sharded_materials``,
    # not through ``run_distributed``'s ``**kwargs``: that kwargs bag is
    # forwarded only on the ``n_devices == 1`` fast path and is silently
    # discarded at exactly the device counts this stage exists for.
    sharded_pec_mask = (
        None if pec_mask_slabs is None else _shard_stacked(pec_mask_slabs))

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
              -> sources -> PEC mask -> exch E -> probes

        The E ghost exchange is the LAST stage of the E half-step, so a
        ghost row is always a copy of the owner's FINISHED real row -- a
        row the realized-PEC mask stage has already zeroed where the
        geometry says metal (#1053). This is ``distributed_nu.py``'s order
        since ``ac782d4f`` (#931 T3) and
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

        # 6b. Realized-PEC cell mask (#1053). Geometry PEC, as opposed to the
        #     domain faces step_fn_pec applies -- this body has no domain-face
        #     PEC at all, because its faces are CPML. Each rank realizes the
        #     per-component edge masks on its own slab by calling the one
        #     owner, rfx.boundaries.pec.realized_pec_edge_masks, and zeroes
        #     only its REAL cells; the ghost rows are forced False inside the
        #     kernel so a seam cell is zeroed exactly once, by the rank that
        #     owns it.
        #
        #     POSITION IS LOAD-BEARING: after injection, immediately BEFORE
        #     the E ghost exchange. The exchange then hands the neighbour a
        #     FINISHED row. Exchanging first leaves rank 0's right ghost
        #     carrying the un-zeroed Ey/Ez of the seam plane rank 1 zeroes a
        #     stage later, and rank 0's next H update at its last real cell
        #     reads that stale plane -- measured on the nu lane at 2.107e-01
        #     final-step error against a 5e-5 gate (ac782d4f, #931 T3), versus
        #     7.773e-08 in this order.
        if sharded_pec_mask is not None:
            st = apply_pec_mask_shmap(
                st, sharded_pec_mask, mesh, n_devices, nx_local)

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
              -> PEC mask -> exch E -> probes

        Same invariant as ``step_fn_cpml``: the E ghost exchange is last,
        so a ghost row is a copy of the owner's finished real row. The
        PEC face moved with it for lane parity with ``distributed_nu``;
        on THIS lane that half of the move is provably and measurably
        inert (stage 6 comment), because the DOMAIN-FACE PEC is the only
        thing it touches. The realized-PEC cell mask of stage 5b is a
        different matter entirely and is why the invariant is load-bearing
        rather than merely tidy (#1053).
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

        # 5. PEC boundaries (domain faces only; a declared PEC VOLUME is
        #    realized at stage 5b below, and declared sheets/wires are
        #    refused -- see the NotImplementedError in run_distributed).
        #    Injection runs BEFORE the face, matching distributed_nu stage 7.
        #    The two lanes therefore both differ from the single-device lane,
        #    which applies the faces before its soft-source loop; the
        #    difference is observable only for a source placed ON a domain
        #    PEC face, where the tangential E is zeroed in the same step it
        #    is injected. #1041 measured no such fixture and did not change
        #    it -- a source on a PEC face is its own question.
        st = _apply_pec_shmap(st, mesh, n_devices, nx_local, pad_x=pad_x)

        # 5b. Realized-PEC cell mask (#1053), AFTER the domain faces and
        #     immediately before the E ghost exchange -- distributed_nu's
        #     stage 8, in distributed_nu's position. See the long note on
        #     step_fn_cpml stage 6b: getting this one stage later, after the
        #     exchange, cost 2.107e-01 against a 5e-5 gate when the nu lane
        #     had it there. The domain-face PEC above did NOT move; its
        #     position relative to the exchange was measured inert (#1041
        #     fixture P, bit-identical), and that measurement does not
        #     transfer to a body mask, which is the whole reason this stage
        #     goes here and not next to it.
        if sharded_pec_mask is not None:
            st = apply_pec_mask_shmap(
                st, sharded_pec_mask, mesh, n_devices, nx_local)

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
