"""Multi-GPU distributed FDTD runner using jax.pmap.

Uses 1D slab decomposition along the x-axis with ghost cell
exchange via jax.lax.ppermute.  Supports PEC and CPML boundaries,
soft sources, point probes, lumped ports, and dispersive materials
(Debye / Lorentz / Drude).

Phase 1: PEC boundary (1D slab decomposition)
Phase 2: CPML absorbing boundary support — "apply everywhere, override
         with ghosts" strategy.  Each device runs CPML on all 6 faces of
         its local slab.  Ghost exchange after CPML overwrites x-face
         artifacts on interior devices; physical boundary devices (first /
         last) keep their CPML absorption intact.
Phase 3: Lumped ports + Debye/Lorentz dispersive materials.
         Port impedance is folded into materials before splitting.
         ADE (auxiliary differential equation) state is carried through
         the scan loop; updates are purely local (no cross-device
         exchange needed for polarization fields).

LEGACY LANE -- not the production distributed path (#1038 leg 6, PI decision
2026-09-15). ``Simulation.run(devices=...)`` dispatches to
``rfx.runners.distributed_v2.run_distributed`` (``shard_map``) for uniform and
non-uniform grids alike; v2 is the single distributed development trunk. This
module is NOT merged into v2: the two are not bit-identical (max |delta|
2.794e-09 on a 9.4145e-03 peak) and they disagree on the odd-``nx`` rule (this
module refuses an odd ``nx``, v2 pads it).

The module stays, and stays supported, for three live roles:

* ``distributed_v2.py:56`` imports twelve domain splitting, local-update and
  CPML names from here (``gather_array_x``, ``_split_state``,
  ``_split_materials``, the Debye/Lorentz splitters,
  ``_apply_cpml_{e,h}_distributed``, ...);
* ``rfx/api/_execute.py`` imports ``_split_materials`` from here for the
  distributed non-uniform forward path;
* ``distributed_v2.run_distributed`` delegates to this module's
  ``run_distributed`` verbatim as its single-device fast path
  (``distributed_v2.py:514-517``, ``if n_devices == 1``), so this code still
  runs under every one-device ``sim.run(devices=[...])`` call.

As of #1038 leg 6 it is no longer re-exported from ``rfx.runners``; import it
by full module path.

STEP ORDER: the same as v2 and ``distributed_nu`` since #1055. Both scan
bodies here inject sources (and, on the PEC path, apply the domain-face PEC)
BEFORE the E ghost exchange, so the exchange is the last stage of the E
half-step and a ghost row is always a copy of the owner's FINISHED real row.
Until #1055 this module exchanged first -- the defect #1041 measured and fixed
on v2 (#1056), reproduced here to four digits: a soft source in rank 1's FIRST
real cell read 1.859e-01 (pec) / 1.653e-01 (cpml) relative to the SAME model on
one device, and 0.000e+00 (pec) / 1.579e-06 (cpml) after the reorder -- the
interior-source control on the same geometry, which is order-insensitive, sits
at 0.000e+00 / 2.417e-06, so the corrected seam is at the lane floor
(``scripts/diagnostics/issue1055_v1_step_order.py``, gated by
``tests/unit/runners/test_distributed_v2_seam_source_order.py``, which
parametrises over both runners). The three step orders no longer diverge, so a
reader comparing the lanes does not have to work out which one is the odd one.
"""

from __future__ import annotations

from functools import partial

from rfx.runners._exchange_interval import validate_exchange_interval

import jax
import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer
from jax import lax

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    EPS_0,
    MU_0,
    _shift_fwd,
    _shift_bwd,
    cell_owned_component_materials,
    lumped_components,
    map_lumped,
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
from rfx.runners._distributed_common import (
    cpml_coeff_e_vacuum,
    cpml_coeff_h_vacuum,
    gather_array_x,
    split_array_x,
    split_poles_x,
    zeros_psi_stacked,
)
# Defined here until the pmap runner's retirement; moved verbatim to
# _distributed_common and re-exported, so ``rfx.runners.distributed.<name>``
# still resolves for every importer.
from rfx.runners._distributed_common import (
    _split_state,
    _split_materials,
    _split_debye_coeffs,
    _split_debye_state,
    _split_lorentz_coeffs,
    _split_lorentz_state,
    _update_h_local,
    _update_e_local,
    _update_e_debye_local,
    _update_e_lorentz_local,
    _update_e_local_with_dispersion,
    _init_cpml_distributed,
    _apply_cpml_e_distributed,
    _apply_cpml_h_distributed,
)


# ---------------------------------------------------------------------------
# Domain splitting / gathering
# ---------------------------------------------------------------------------

# #1038 leg 2: ``split_array_x`` and ``gather_array_x`` moved VERBATIM to
# rfx/runners/_distributed_common.py and are re-exported here, at the position
# they were defined, so ``rfx.runners.distributed.split_array_x`` and
# ``...gather_array_x`` keep resolving for the three external importers
# (tests/unit/runners/test_distributed.py, test_distributed_v2_gather_traceable.py,
# and distributed_v2.py:57). They had to move before any consolidation of the
# helpers that CALL them: ``_distributed_common`` is imported by distributed.py
# ten lines above their old definitions, so a shared body calling them could not
# import them from here. Measured, not assumed:
#   ImportError: cannot import name 'gather_array_x' from partially initialized
#   module 'rfx.runners.distributed' (most likely due to a circular import)
# Both are closure-free module-level functions whose only global is ``jnp``, so
# the move needs no signature change.


def _gather_state(state, ghost=1):
    """Gather per-device FDTDState slabs back into a single state."""
    return FDTDState(
        ex=gather_array_x(state.ex, ghost),
        ey=gather_array_x(state.ey, ghost),
        ez=gather_array_x(state.ez, ghost),
        hx=gather_array_x(state.hx, ghost),
        hy=gather_array_x(state.hy, ghost),
        hz=gather_array_x(state.hz, ghost),
        step=state.step[0],
    )


# ---------------------------------------------------------------------------
# Ghost cell exchange
# ---------------------------------------------------------------------------

def _exchange_component(field, n_devices, axis_name="devices"):
    """Exchange ghost cells for a single field component.

    field : shape (nx_local + 2*ghost, ny, nz)  -- per-device via pmap
    The first and last x-planes are ghost cells.

    After exchange:
    - field[0, :, :] = right-neighbor's field[1, :, :]  (NO -- left neighbor's boundary)

    Convention:
    - ghost[0]  = left ghost  <- should contain left neighbor's rightmost real cell
    - ghost[-1] = right ghost <- should contain right neighbor's leftmost real cell
    - real cells: field[1:-1]
    """
    # The rightmost real cell of each device -> left ghost of right neighbor
    right_boundary = field[-2:-1, :, :]  # last real cell: index -2, shape (1, ny, nz)

    # The leftmost real cell of each device -> right ghost of left neighbor
    left_boundary = field[1:2, :, :]   # first real cell: index 1, shape (1, ny, nz)

    # ppermute: send from device i to device (i+1) % n  (right shift)
    # This sends each device's right_boundary to its right neighbor's left ghost
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    left_ghost_recv = lax.ppermute(right_boundary, axis_name, perm=perm_right)

    # ppermute: send from device i to device (i-1) % n  (left shift)
    # This sends each device's left_boundary to its left neighbor's right ghost
    perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    right_ghost_recv = lax.ppermute(left_boundary, axis_name, perm=perm_left)

    # Determine device index to mask boundary devices
    device_idx = lax.axis_index(axis_name)

    # For device 0: don't overwrite left ghost (physical boundary, keep zero)
    # For device n-1: don't overwrite right ghost (physical boundary, keep zero)
    left_ghost_val = jnp.where(device_idx > 0,
                               left_ghost_recv,
                               field[0:1, :, :])
    right_ghost_val = jnp.where(device_idx < n_devices - 1,
                                right_ghost_recv,
                                field[-1:, :, :])

    field = field.at[0:1, :, :].set(left_ghost_val)
    field = field.at[-1:, :, :].set(right_ghost_val)
    return field


def _exchange_h_ghosts(state, n_devices, axis_name="devices"):
    """Exchange ghost cells for all H components."""
    return state._replace(
        hx=_exchange_component(state.hx, n_devices, axis_name),
        hy=_exchange_component(state.hy, n_devices, axis_name),
        hz=_exchange_component(state.hz, n_devices, axis_name),
    )


def _exchange_e_ghosts(state, n_devices, axis_name="devices"):
    """Exchange ghost cells for all E components."""
    return state._replace(
        ex=_exchange_component(state.ex, n_devices, axis_name),
        ey=_exchange_component(state.ey, n_devices, axis_name),
        ez=_exchange_component(state.ez, n_devices, axis_name),
    )


def _apply_pec_local(state, n_devices, nx_local_with_ghost, axis_name="devices",
                     pad_x: int = 0):
    """Apply PEC boundary on a local slab.

    - y and z PEC: always applied (all devices own the full y/z extent).
    - x PEC: only device 0 applies x-lo, only device N-1 applies x-hi.
      These operate on the first/last REAL cell (index ghost and
      nx_local+ghost-1-pad_x), not the ghost cell itself.

    ``pad_x`` (#623, same class as #622): when the last rank's slab
    carries alignment-pad cells past the real x-hi face (global node
    ``nx - 1``), the face must act on that real node, not on the
    padded slab end.
    """
    device_idx = lax.axis_index(axis_name)
    ghost = 1

    ex, ey, ez = state.ex, state.ey, state.ez

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

    # X-axis PEC: only at physical boundaries
    # Device 0: x-lo PEC at real cell index ghost (=1)
    # We zero tangential components (ey, ez) at x=0 of the global domain,
    # which is index `ghost` (first real cell) of device 0.
    is_first = (device_idx == 0)
    ey_xlo = jnp.where(is_first, 0.0, ey[ghost, :, :])
    ez_xlo = jnp.where(is_first, 0.0, ez[ghost, :, :])
    ey = ey.at[ghost, :, :].set(ey_xlo)
    ez = ez.at[ghost, :, :].set(ez_xlo)

    # Device N-1: x-hi PEC at last real cell (skip ghost AND alignment
    # pad, #623 — same class as #622)
    is_last = (device_idx == n_devices - 1)
    last_real = nx_local_with_ghost - 1 - ghost - pad_x  # last real cell index
    ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
    ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
    ey = ey.at[last_real, :, :].set(ey_xhi)
    ez = ez.at[last_real, :, :].set(ez_xhi)

    return state._replace(ex=ex, ey=ey, ez=ez)


def _apply_pmc_local(state, n_devices, nx_local_with_ghost, axis_name,
                     pmc_faces, pad_x: int = 0):
    """Apply PMC (``H_tangential = 0``) on a local slab.

    Electromagnetic dual of :func:`_apply_pec_local`. Hook point is the
    H-half of the scan body (after H ghost exchange, before E update)
    per OQ9 — PMC must fire before the next E-update reads H via curl.

    - y and z PMC: always applied on every device (all devices own the
      full y/z extent).
    - x PMC: only device 0 applies x-lo, only device N-1 applies x-hi.
      Acts on the first/last REAL cell (index ghost and
      ``nx_local_with_ghost - 1 - ghost - pad_x``), NOT the ghost cell
      nor the alignment-pad cells when ``pad_x > 0`` (#623, same class
      as #622).
    """
    if not pmc_faces:
        return state

    device_idx = lax.axis_index(axis_name)
    ghost = 1

    hx, hy, hz = state.hx, state.hy, state.hz

    # Yee-grid convention (see rfx/boundaries/pmc.py): _hi PMC acts on
    # index -2 (0.5·dx INSIDE the wall), not -1 (ghost 0.5·dx OUTSIDE).
    # For x_hi in distributed, the physical wall is local index last_real;
    # the INSIDE half-cell is last_real - 1.
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

    return state._replace(hx=hx, hy=hy, hz=hz)


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
        Ghost exchange interval in timesteps; only integer 1 is supported.

    Returns
    -------
    Result
    """
    validate_exchange_interval(exchange_interval)
    from rfx.materials.thin_conductor import refuse_f0_sheets as _refuse_f0
    _refuse_f0(sim._thin_conductors, "distributed (v1) runner")
    import warnings

    if sim._boundary == "upml":
        raise ValueError("boundary='upml' does not support distributed execution")

    # ------------------------------------------------------------------
    # Graceful fallback for features that require the full domain on a
    # single device (TFSF auxiliary grid, waveguide eigenmode solver).
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

    from rfx.runners.distributed_v2 import refuse_unsupported_distributed_features
    refuse_unsupported_distributed_features(
        sim, lane="distributed (v1) pmap runner", bloch=kwargs.get("bloch"))

    from rfx.api import Result

    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)

    # Resolve PMC faces (T8, 2026-04). ``BoundarySpec.pmc_faces()`` returns
    # a set; freeze it so it is safely closed-over by the pmap scan
    # bodies defined below. Empty frozenset == no-op inside the helper.
    _pmc_faces_frozen = frozenset(
        sim._boundary_spec.pmc_faces()
        if getattr(sim, "_boundary_spec", None) is not None
        else ()
    )

    # Build grid and materials (full domain)
    # PRE-EXISTING GAP, not a #931 regression (verified 2026-09-07): this
    # lane assembles ``pec_mask`` and never applies it. Its step body only
    # calls the DOMAIN-FACE PEC (``_apply_pec_local``); no geometry PEC —
    # volume, sheet or wire — reaches the field update here. #931 did not
    # introduce this and does not fix it; threading sheets in would only
    # make the drop harder to see.
    #
    # The refusal below used to name "redraw it as a volume" as the remedy,
    # which on this lane leaves the metal just as absent. Since the drop is
    # the same for all three kinds, so is the refusal.
    grid = sim._build_grid()
    _d_pec_sheets: list = []
    _d_pec_wires: list = []
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
            "run_distributed() does not realize declared PEC geometry of "
            "ANY kind (#931): this lane carries geometry PEC only as a cell "
            "mask, its step body applies domain-face PEC alone, and a sheet "
            "or a wire owns no cell to begin with. Declared here: "
            f"{', '.join(_d_declared)} — all of it would be absent from "
            "every rank with no sign of it. Redrawing a sheet as a volume "
            "does NOT help on this lane. Use sim.run() without devices=, "
            "which realizes all three, or model the conductor as a sigma "
            "fill, which rides in the material arrays this lane does shard.")
    materials = base_materials

    nx, ny, nz = grid.shape
    if nx % n_devices != 0:
        raise ValueError(
            f"Grid nx={nx} is not evenly divisible by {n_devices} devices. "
            f"Adjust domain size or dx so that nx is a multiple of n_devices."
        )

    nx_per = nx // n_devices
    ghost = 1
    nx_local = nx_per + 2 * ghost

    # Determine boundary type
    use_cpml = sim._boundary == "cpml" and grid.cpml_layers > 0
    n_cpml = grid.cpml_layers if use_cpml else 0

    # Build sources and probes on the full grid
    # First pass: fold lumped port impedances into materials (must happen
    # before splitting so the owning device's slab gets correct sigma).
    sources = []
    probes = []
    for pe in sim._ports:
        if pe.impedance > 0.0 and pe.extent is None:
            # Single-cell lumped port: fold impedance into materials
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
    # Source mapping
    src_device_ids = []
    src_local_specs = []  # (local_i, j, k, component, waveform)
    for s in sources:
        dev_id = s.i // nx_per
        local_i = (s.i % nx_per) + ghost  # offset by ghost
        src_device_ids.append(dev_id)
        src_local_specs.append((local_i, s.j, s.k, s.component))

    # Probe mapping
    prb_device_ids = []
    prb_local_specs = []
    for p in probes:
        dev_id = p.i // nx_per
        local_i = (p.i % nx_per) + ghost
        prb_device_ids.append(dev_id)
        prb_local_specs.append((local_i, p.j, p.k, p.component))

    # Precompute source waveform matrix: (n_steps, n_sources)
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # Replicate waveforms to all devices: (n_devices, n_steps, n_sources)
    src_waveforms_rep = jnp.broadcast_to(
        src_waveforms[None, :, :],
        (n_devices, n_steps, src_waveforms.shape[-1]),
    )

    # Build per-device source mask: (n_devices, n_sources) bool
    # device d only injects source s if src_device_ids[s] == d
    src_device_mask = jnp.array(
        [[1.0 if src_device_ids[s] == d else 0.0
          for s in range(len(sources))]
         for d in range(n_devices)],
        dtype=jnp.float32,
    ) if sources else jnp.zeros((n_devices, 0), dtype=jnp.float32)

    # Build per-device probe mask similarly
    prb_device_mask = jnp.array(
        [[1.0 if prb_device_ids[p] == d else 0.0
          for p in range(len(probes))]
         for d in range(n_devices)],
        dtype=jnp.float32,
    ) if probes else jnp.zeros((n_devices, 0), dtype=jnp.float32)

    # Split domain into per-device slabs
    from rfx.core.yee import init_state
    full_state = init_state(grid.shape)
    state_slabs = _split_state(full_state, n_devices, ghost)
    materials_slabs = _split_materials(materials, n_devices, ghost)

    # Initialize dispersion (Debye / Lorentz) on the full domain,
    # then split coefficients and state into per-device slabs.
    _, debye_full, lorentz_full = sim._init_dispersion(
        materials, grid.dt, debye_spec, lorentz_spec)

    has_debye = debye_full is not None
    has_lorentz = lorentz_full is not None

    if has_debye:
        debye_coeffs_full, debye_state_full = debye_full
        debye_coeffs_slabs = _split_debye_coeffs(debye_coeffs_full, n_devices, ghost)
        debye_state_slabs = _split_debye_state(debye_state_full, n_devices, ghost)
    else:
        _dz3 = jnp.zeros((n_devices, 1, nx_local, ny, nz), dtype=jnp.float32)
        _dz = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.float32)
        debye_coeffs_slabs = DebyeCoeffs(ca=_dz, cb=_dz, cc=_dz3, alpha=_dz3, beta=_dz3)
        debye_state_slabs = DebyeState(px=_dz3, py=_dz3.copy(), pz=_dz3.copy())

    if has_lorentz:
        lorentz_coeffs_full, lorentz_state_full = lorentz_full
        lorentz_coeffs_slabs = _split_lorentz_coeffs(lorentz_coeffs_full, n_devices, ghost)
        lorentz_state_slabs = _split_lorentz_state(lorentz_state_full, n_devices, ghost)
    else:
        _lz3 = jnp.zeros((n_devices, 1, nx_local, ny, nz), dtype=jnp.float32)
        _lz = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.float32)
        lorentz_coeffs_slabs = LorentzCoeffs(ca=_lz, cb=_lz, a=_lz3, b=_lz3, c=_lz3, cc=_lz)
        lorentz_state_slabs = LorentzState(
            px=_lz3, py=_lz3.copy(), pz=_lz3.copy(),
            px_prev=_lz3.copy(), py_prev=_lz3.copy(), pz_prev=_lz3.copy(),
        )

    # Initialize CPML if needed
    if use_cpml:
        cpml_params, cpml_state_slabs = _init_cpml_distributed(
            grid, nx_local, n_devices)

    # Static metadata captured by closure
    n_src = len(sources)
    n_prb = len(probes)

    dt = grid.dt
    dx = grid.dx

    # Capture exchange_interval for use inside pmap closures
    _exchange_interval = int(exchange_interval)

    if use_cpml:
        # CPML path: uses dict carry for state + cpml_state
        @partial(jax.pmap, axis_name="devices", devices=devices)
        def distributed_scan(state_slab, materials_slab, cpml_state_dev,
                             step_indices, src_waveforms_dev, src_mask,
                             prb_mask, debye_coeffs_dev, debye_state_dev,
                             lorentz_coeffs_dev, lorentz_state_dev):
            """Scan body over timesteps on one device (CPML path).

            Stage order (#1055)::

                H -> CPML-H -> exch H -> PMC face -> E -> CPML-E
                  -> sources -> exch E -> probes

            The E ghost exchange is the LAST stage of the E half-step, so a
            ghost row is always a copy of the owner's FINISHED real row --
            the mirror of the H half, which applies the PMC face before the
            H exchange so the zero propagates via the exchange. This is
            ``distributed_v2``'s order since #1056 and ``distributed_nu``'s
            since ``ac782d4f`` (#931 T3). Until #1055 this body exchanged
            BEFORE injecting, which lagged a seam-cell source by one step in
            the neighbour's copy -- see the comment on stage 7 for the
            measurement that moved it.
            """

            def step_fn(carry, xs):
                _step_idx, src_vals = xs
                st = carry["fdtd"]
                cpml_st = carry["cpml"]
                db_st = carry["debye"]
                lr_st = carry["lorentz"]

                # 1. H update (local)
                st = _update_h_local(st, materials_slab, dt, dx)

                # 2. CPML H correction (before ghost exchange) — material-aware (#205)
                st, cpml_st = _apply_cpml_h_distributed(
                    st, cpml_params, cpml_st, n_cpml, dt, dx,
                    n_devices, ghost=ghost, axis_name="devices",
                    mu_r=materials_slab.mu_r)

                # 3. Exchange H ghost cells (conditionally skip)
                do_exchange = (_step_idx % _exchange_interval == 0)
                st = lax.cond(
                    do_exchange,
                    lambda s: _exchange_h_ghosts(s, n_devices, "devices"),
                    lambda s: s,
                    st,
                )

                # 3b. PMC face (H-tangential = 0) — T8, 2026-04. H-half
                #     hook per OQ9: after H ghost exchange, before E
                #     update. PMC must fire before the next E-update
                #     reads H via curl.
                st = _apply_pmc_local(
                    st, n_devices, nx_local, "devices",
                    _pmc_faces_frozen)

                # 4. E update (local)
                _debye_arg = (debye_coeffs_dev, db_st) if has_debye else None
                _lorentz_arg = (lorentz_coeffs_dev, lr_st) if has_lorentz else None
                st, new_db, new_lr = _update_e_local_with_dispersion(
                    st, materials_slab, dt, dx,
                    debye=_debye_arg, lorentz=_lorentz_arg)
                if has_debye:
                    db_st = new_db
                if has_lorentz:
                    lr_st = new_lr

                # 5. CPML E correction (before ghost exchange) — material-aware (#205)
                st, cpml_st = _apply_cpml_e_distributed(
                    st, cpml_params, cpml_st, n_cpml, dt, dx,
                    n_devices, ghost=ghost, axis_name="devices",
                    eps_r=materials_slab.eps_r)

                # 6. Source injection (only on owning device)
                for idx_s in range(n_src):
                    li, lj, lk, lc = src_local_specs[idx_s]
                    val = src_vals[idx_s] * src_mask[idx_s]
                    field = getattr(st, lc)
                    field = field.at[li, lj, lk].add(val)
                    st = st._replace(**{lc: field})

                # 7. Exchange E ghost cells -- LAST stage of the E half-step,
                #    so a ghost row is a copy of the owner's FINISHED real row
                #    (#1055, matching distributed_v2 since #1056 and
                #    distributed_nu since ac782d4f / #931 T3). Exchanging
                #    before injection left a source at rank d's first real
                #    cell out of rank d-1's right ghost for one step, and
                #    rank d-1's next H update read the pre-injection plane.
                #    Measured by scripts/diagnostics/issue1055_v1_step_order.py
                #    on a seam-cell ez source, 2 ranks, 300 steps,
                #    boundary="cpml", against the SAME model on ONE device:
                #    the probe 4 cells into the neighbouring rank went from
                #    1.653e-01 relative (first divergent step 4, the causal
                #    arrival of the source's own wavefront) to 1.579e-06
                #    (step 50), which is BELOW this body's own lane floor --
                #    the interior-source control, source 8 cells from the
                #    seam, sits at 2.417e-06 / 2.894e-05 on the same geometry
                #    and is bit-identical between the two orderings.
                #
                #    The defect is ONE-SIDED. Only the RIGHT E ghost is live:
                #    rank d-1's H at its last REAL cell consumes it. A rank's
                #    LEFT E ghost feeds only its own H at that same index, and
                #    the H exchange (stage 3) overwrites that H with the
                #    neighbour's authoritative value before anything reads it.
                #    So a source in rank d's LAST real cell was never affected
                #    -- measured bit-identical between the two orderings --
                #    and that is exactly where both distributed_v1_* fixtures
                #    of the #1038 bit-identity lock put their source, which is
                #    why that lock stays 13/13 green through this change.
                st = lax.cond(
                    do_exchange,
                    lambda s: _exchange_e_ghosts(s, n_devices, "devices"),
                    lambda s: s,
                    st,
                )

                # 8. Probe sampling (only on owning device)
                samples = []
                for idx_p in range(n_prb):
                    li, lj, lk, lc = prb_local_specs[idx_p]
                    val = getattr(st, lc)[li, lj, lk] * prb_mask[idx_p]
                    samples.append(val)

                if samples:
                    probe_out = jnp.stack(samples)
                else:
                    probe_out = jnp.zeros(0, dtype=jnp.float32)

                return {"fdtd": st, "cpml": cpml_st,
                        "debye": db_st, "lorentz": lr_st}, probe_out

            xs = (step_indices, src_waveforms_dev)
            carry_init = {"fdtd": state_slab, "cpml": cpml_state_dev,
                          "debye": debye_state_dev, "lorentz": lorentz_state_dev}
            final_carry, probe_ts = lax.scan(step_fn, carry_init, xs)
            return final_carry["fdtd"], probe_ts

    else:
        # PEC path: original simple carry
        @partial(jax.pmap, axis_name="devices", devices=devices)
        def distributed_scan(state_slab, materials_slab, cpml_state_dev,
                             step_indices, src_waveforms_dev, src_mask,
                             prb_mask, debye_coeffs_dev, debye_state_dev,
                             lorentz_coeffs_dev, lorentz_state_dev):
            """Scan body over timesteps on one device (PEC path).

            Stage order (#1055)::

                H -> exch H -> PMC face -> E -> sources -> PEC face
                  -> exch E -> probes

            Same invariant as the CPML body: the E ghost exchange is last,
            so a ghost row is a copy of the owner's finished real row. The
            PEC face moved with it for parity with ``distributed_v2`` /
            ``distributed_nu``; on THIS lane that half of the move is
            measurably inert (stage 6 comment), because the DOMAIN-FACE PEC
            is the only PEC this lane applies at all.
            """

            def step_fn(carry, xs):
                _step_idx, src_vals = xs
                st = carry["fdtd"]
                db_st = carry["debye"]
                lr_st = carry["lorentz"]

                # 1. H update (local)
                st = _update_h_local(st, materials_slab, dt, dx)

                # 2. Exchange H ghost cells (conditionally skip)
                do_exchange = (_step_idx % _exchange_interval == 0)
                st = lax.cond(
                    do_exchange,
                    lambda s: _exchange_h_ghosts(s, n_devices, "devices"),
                    lambda s: s,
                    st,
                )

                # 2b. PMC face (H-tangential = 0) — T8, 2026-04. H-half
                #     hook per OQ9: after H ghost exchange, before E
                #     update.
                st = _apply_pmc_local(
                    st, n_devices, nx_local, "devices",
                    _pmc_faces_frozen)

                # 3. E update (local)
                _debye_arg = (debye_coeffs_dev, db_st) if has_debye else None
                _lorentz_arg = (lorentz_coeffs_dev, lr_st) if has_lorentz else None
                st, new_db, new_lr = _update_e_local_with_dispersion(
                    st, materials_slab, dt, dx,
                    debye=_debye_arg, lorentz=_lorentz_arg)
                if has_debye:
                    db_st = new_db
                if has_lorentz:
                    lr_st = new_lr

                # 4. Source injection (only on owning device)
                for idx_s in range(n_src):
                    li, lj, lk, lc = src_local_specs[idx_s]
                    val = src_vals[idx_s] * src_mask[idx_s]
                    field = getattr(st, lc)
                    field = field.at[li, lj, lk].add(val)
                    st = st._replace(**{lc: field})

                # 5. PEC boundaries (domain faces only; declared PEC geometry
                #    of any kind is REFUSED on this lane, see the
                #    NotImplementedError in run_distributed). Injection runs
                #    BEFORE the face, matching distributed_v2 stage 5 and
                #    distributed_nu stage 7. Both distributed lanes therefore
                #    differ from the single-device lane, which applies the
                #    faces before its soft-source loop; the difference is
                #    observable only for a source placed ON a domain PEC face,
                #    where the tangential E is zeroed in the same step it is
                #    injected. #1055 measured no such fixture and did not
                #    change it -- a source on a PEC face is its own question.
                st = _apply_pec_local(st, n_devices, nx_local, "devices")

                # 6. Exchange E ghost cells -- LAST stage of the E half-step,
                #    so a ghost row is a copy of the owner's FINISHED real row
                #    (#1055; the ordering distributed_v2 took in #1056 and
                #    distributed_nu in ac782d4f / #931 T3). Same measurement
                #    as the CPML body above, boundary="pec": the probe 4 cells
                #    into the neighbouring rank went from 1.859e-01 relative
                #    (first divergent step 4) to 0.000e+00 -- on THIS body the
                #    corrected lane is bit-identical to the single-device
                #    lane, as are both of its controls, so the floor is zero
                #    and the seam sits on it. Same one-sided reachability as
                #    the CPML body: a source in rank d's LAST real cell is
                #    unaffected either way (measured bit-identical).
                #    The PEC half of the move is inert on THIS lane and was
                #    measured to be so (#1055 fixture P, bit-identical):
                #    _apply_pec_local zeroes the y/z faces on every x row
                #    INCLUDING the ghosts, and its x_lo / x_hi faces act on
                #    rank 0's first and rank N-1's last real cell, whose
                #    exchanged copies the receiving rank discards.
                st = lax.cond(
                    do_exchange,
                    lambda s: _exchange_e_ghosts(s, n_devices, "devices"),
                    lambda s: s,
                    st,
                )

                # 7. Probe sampling (only on owning device)
                samples = []
                for idx_p in range(n_prb):
                    li, lj, lk, lc = prb_local_specs[idx_p]
                    val = getattr(st, lc)[li, lj, lk] * prb_mask[idx_p]
                    samples.append(val)

                if samples:
                    probe_out = jnp.stack(samples)
                else:
                    probe_out = jnp.zeros(0, dtype=jnp.float32)

                return {"fdtd": st, "debye": db_st, "lorentz": lr_st}, probe_out

            xs = (step_indices, src_waveforms_dev)
            carry_init = {"fdtd": state_slab, "debye": debye_state_dev,
                          "lorentz": lorentz_state_dev}
            final_carry, probe_ts = lax.scan(step_fn, carry_init, xs)
            return final_carry["fdtd"], probe_ts

    # Prepare scan inputs: replicate step indices across devices
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    step_indices_rep = jnp.broadcast_to(
        step_indices[None, :], (n_devices, n_steps)
    )

    # Build dummy CPML state for PEC path (keeps pmap signature uniform)
    if not use_cpml:
        from rfx.boundaries.cpml import CPMLState
        # Minimal dummy: all zeros, shape (n_devices, 0, ...) won't work
        # for NamedTuple, so use shape (n_devices, 1, 1, 1)
        _z = jnp.zeros((n_devices, 1, 1, 1), dtype=jnp.float32)
        cpml_state_slabs = CPMLState(
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

    # Run the distributed simulation
    final_state_slabs, probe_ts_all = distributed_scan(
        state_slabs,
        materials_slabs,
        cpml_state_slabs,
        step_indices_rep,
        src_waveforms_rep,
        src_device_mask,
        prb_device_mask,
        debye_coeffs_slabs,
        debye_state_slabs,
        lorentz_coeffs_slabs,
        lorentz_state_slabs,
    )

    # Gather final state
    final_state = _gather_state(final_state_slabs, ghost)

    # Aggregate probe time series: sum across devices
    # probe_ts_all: (n_devices, n_steps, n_probes)
    # Each probe is non-zero only on its owning device, so sum works
    if n_prb > 0:
        time_series = jnp.sum(probe_ts_all, axis=0)  # (n_steps, n_probes)
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
