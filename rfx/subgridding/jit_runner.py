"""JIT-compiled subgridded FDTD runner via jax.lax.scan.

Replaces the Python-loop runner (runner.py) with a fully JIT-compiled
version that achieves 50-100x speedup. Both coarse and fine grids
are updated per step with SBP-SAT interface coupling.  The public
``Simulation.add_refinement`` path is a full-x/y z-slab refinement, so only the
two z faces are artificial SAT interfaces; x/y faces remain physical
boundaries.

Handles both CPML (absorbing) and PEC (reflecting) coarse-grid
boundaries. When cpml_layers == 0 the CPML subsystem is skipped
entirely and PEC is applied on all domain faces.
"""

from __future__ import annotations

from typing import NamedTuple

from rfx import _realized

import jax
import jax.numpy as jnp

from rfx.core.drives import drive_layout, inject_drives

from rfx.core.yee import FDTDState, MaterialArrays, init_state
from rfx.stepping import update_h, update_e
from rfx.boundaries.pec import apply_pec, apply_pec_faces, apply_pec_mask
from rfx.grid import Grid
from rfx.subgridding.sbp_sat_3d import (
    SubgridConfig3D,
    _shared_node_coupling_3d,
    _shared_node_coupling_h_3d,
)
from rfx.subgridding.material_sat import interface_pair_deltas
from rfx.core.yee import EPS_0, MU_0, component_h_materials
from rfx.probes.probes import _ampere_loop
from rfx.geometry.rasterize_grid import unpadded_high_walls, fine_high_walls
from rfx.model.materials import sat_face_mu, subgrid_components


class SubgridResult(NamedTuple):
    """Result from JIT-compiled subgridded simulation."""
    state_c: FDTDState
    state_f: FDTDState
    time_series: jnp.ndarray
    config: SubgridConfig3D
    dt: float
    time_series_c: jnp.ndarray | None = None
    lumped_sparam_v_dft_f: jnp.ndarray | None = None
    lumped_sparam_i_dft_f: jnp.ndarray | None = None
    lumped_sparam_freqs_f: jnp.ndarray | None = None
    lumped_sparam_impedances_f: jnp.ndarray | None = None
    lumped_sparam_cell_counts_f: jnp.ndarray | None = None
    ntff_data_f: object | None = None
    ntff_box_f: object | None = None


from rfx.stepping.subgrid_coupling import ZSlabCouplingPlan


from rfx.model.subgrid_options import SubgridRunOptions


from rfx.stepping.subgrid_coupling import _uses_endpoint_fine_shape


from rfx.stepping.subgrid_coupling import _trapezoid_weights


from rfx.stepping.subgrid_coupling import _restrict_axis_linear_adjoint


from rfx.stepping.subgrid_coupling import _restrict_node_aligned_2d


from rfx.stepping.subgrid_coupling import _restrict_node_aligned_2d_sample


from rfx.stepping.subgrid_coupling import _prolong_axis_linear


from rfx.stepping.subgrid_coupling import _prolong_node_aligned_2d


from rfx.stepping.subgrid_coupling import MODAL_TRACE_DELTA_FILTERS


from rfx.stepping.subgrid_coupling import _z_slab_face_ks


from rfx.stepping.subgrid_coupling import _z_slab_coupling_plan


from rfx.stepping.subgrid_coupling import _z_slab_coupling_e_3d


from rfx.stepping.subgrid_coupling import _z_slab_coupling_h_3d


from rfx.stepping.subgrid_coupling import _box_node_aligned_coupling_e_3d


from rfx.stepping.subgrid_coupling import _box_node_aligned_coupling_h_3d


from rfx.stepping.subgrid_coupling import _sync_z_slab_coarse_tangential_from_fine


from rfx.stepping.subgrid_coupling import _restrict_node_aligned_3d_sample


from rfx.stepping.subgrid_coupling import _blend_to_sample


from rfx.stepping.subgrid_coupling import _sync_z_slab_coarse_shadow_from_fine


from rfx.stepping.subgrid_coupling import _mask_z_slab_coarse_shadow_interior


from rfx.stepping.subgrid_coupling import _mask_z_slab_coarse_shadow_all


from rfx.stepping.subgrid_coupling import _mask_box_coarse_shadow_all


from rfx.stepping.subgrid_coupling import _z_slab_material_coupling_h_3d


from rfx.stepping.subgrid_coupling import _z_slab_material_coupling_e_3d


from rfx.stepping.subgrid_jit import _STEP_FN_CAPTURES


from rfx.stepping.subgrid_jit import _STEP_FN_REQUIRED_CAPTURES


from rfx.stepping.subgrid_jit import _make_step_fn


def run_subgridded_jit(
    grid_c: Grid,
    mats_c: MaterialArrays,
    mats_f: MaterialArrays,
    config: SubgridConfig3D,
    n_steps: int,
    *,
    opts: SubgridRunOptions = SubgridRunOptions(),
) -> SubgridResult:
    """Run subgridded FDTD via jax.lax.scan.

    Parameters
    ----------
    grid_c : Grid -- coarse grid (full domain, with or without CPML)
    mats_c : MaterialArrays -- coarse materials
    mats_f : MaterialArrays -- fine materials
    config : SubgridConfig3D
    n_steps : int
    pec_mask_c, pec_mask_f : boolean arrays or None
    sources_f : list of (i, j, k, component, waveform_array)
    sources_c : list of (i, j, k, component, waveform_array)
        Diagnostic-only coarse-shadow soft sources paired with ``sources_f``.
    probe_indices_f : list of (i, j, k)
    probe_components : list of component names
    probe_indices_c : list of (i, j, k), optional
        Diagnostic-only coarse-grid point probes. The high-level Simulation API
        records fine probes only; this stream helps interface-transfer
        diagnostics inspect the coarse side without changing public results.
    probe_components_c : list of component names, optional
    use_material_sat : bool or None
        Static selector for material-weighted artificial-interface coupling.
        ``None`` preserves the historical auto-detection from interface
        material arrays.  Pass an explicit bool when differentiating/jitting
        through material arrays so the branch decision is not made from a JAX
        tracer.  For contained-material research gates whose artificial
        interfaces remain vacuum, this should be ``False``.
    sync_coarse_interface_from_fine : bool
        Diagnostic-only experiment: after z-slab H/E coupling, overwrite the
        coarse tangential z-interface traces from the restricted fine traces.
        This is not a public support mode.
    sync_coarse_shadow_from_fine : bool
        Diagnostic-only experiment: keep the overlapping coarse shadow slab
        synchronized from colocated fine fields before/after updates. This is
        not a public support mode.
    sync_box_coarse_shadow_from_fine : bool
        Diagnostic-only local x/y experiment: keep a non-full-x/y overlapping
        coarse shadow box synchronized from colocated fine fields before/after
        updates. This tests local-box transfer ownership and is not a public
        support mode.  The sync overwrites the full coarse-shadow volume from
        the colocated fine samples on every E/H component before and after the
        H and E updates.
    mask_coarse_shadow_interior : bool
        Diagnostic-only experiment: zero the overlapping coarse shadow interior
        while preserving the z-interface faces. This is not a public support
        mode.
    use_exterior_z_interfaces : bool
        Diagnostic-only experiment: couple fine z faces to the immediately
        exterior coarse planes and zero the full overlapping coarse shadow slab.
        This approximates disjoint coarse/fine ownership for falsification
        experiments and is not a public support mode.
    use_boundary_terminated_exterior_z_interfaces : bool
        Diagnostic-only experiment: exterior/disjoint-like ownership for slabs
        that touch a physical PEC z boundary.  SAT coupling is skipped on
        physical z-boundary faces and retained only on sides with an active
        exterior coarse plane.  This tests whether the 83.3% exterior-coverage
        ceiling is the remaining broad-waveform blocker; it is not a public
        support mode.
    ghost_exterior_coarse_shadow_from_fine : bool
        Diagnostic-only experiment for exterior/disjoint-like ownership: before
        each coarse H/E update, populate the otherwise fine-owned coarse shadow
        slab from colocated fine fields so coarse exterior curl stencils see
        fine-grid ghost values instead of zeros.  The shadow slab is masked
        again after the coarse update; it remains non-owned and is not a public
        support mode.
    material_sat_scale : float
        Diagnostic-only multiplier on the impedance-upwind/material SAT
        corrections.  The production value is 1.0; any other value is a
        claims-blocking research knob.
    material_sat_coarse_scale, material_sat_fine_scale : float
        Diagnostic-only side-specific multipliers on coarse/fine material SAT
        corrections.  These probe whether the exterior/two-interface residual
        is caused by an incorrect side norm in the current closure.  Production
        value is 1.0 for both.
    material_sat_e_coarse_scale, material_sat_e_fine_scale,
    material_sat_h_coarse_scale, material_sat_h_fine_scale : float
        Diagnostic-only E/H-specific side multipliers layered on top of the
        side scales.  These probe whether the centered residual is an E/H phase
        split rather than one scalar side norm.  Production value is 1.0 for all.
    material_sat_zlo_scale, material_sat_zhi_scale : float
        Diagnostic-only z-interface-specific multipliers layered on top of the
        side and E/H scales.  These test lower-vs-upper two-interface energy
        balance.  Production value is 1.0 for both.
    material_sat_e_zlo_scale, material_sat_e_zhi_scale,
    material_sat_h_zlo_scale, material_sat_h_zhi_scale : float
        Diagnostic-only E/H-specific z-interface multipliers layered on top of
        the z-interface scale.  These test lower-interface phase/field balance.
        Production value is 1.0 for all.
    material_sat_pair_a_zlo_scale, material_sat_pair_b_zlo_scale : float
        Diagnostic-only lower-interface tangential-pair multipliers.  Pair A is
        ``(E_x, H_y)`` and pair B is ``(E_y, -H_x)``.  These test whether the
        centered residual is a tangential transfer imbalance rather than an
        E/H-wide or interface-wide scalar.  Production value is 1.0 for both.
    material_sat_zlo_common_trace_projection : {"dual", "coarse", "fine", "average"}
        Diagnostic-only selector for the lower z-interface common trace used by
        material SAT.  ``"dual"`` is the current behavior: coarse and fine
        sides compute their own projected common states.  ``"coarse"`` computes
        the common trace on the coarse face and prolongs it to the fine side;
        ``"fine"`` computes the common trace on the fine face and restricts it
        to the coarse side.  ``"average"`` averages the projected coarse-face
        and fine-face common states before applying both sides.  This probes
        whether centered lower-interface residuals come from non-single-valued
        coarse/fine trace construction.  Production value is ``"dual"``.
    material_sat_zhi_common_trace_projection : {"dual", "coarse", "fine", "average"}
        Diagnostic-only selector for the upper z-interface common trace used by
        material SAT.  This mirrors ``material_sat_zlo_common_trace_projection``
        and probes whether material-jump phase drift comes from non-single-valued
        coarse/fine trace construction at the artificial dielectric/vacuum
        interface.  Production value is ``"dual"``.
    material_sat_normal_e_scale : float
        Diagnostic-only normal electric displacement continuity penalty on
        material z interfaces.  The current production material SAT couples
        tangential characteristic pairs only; nonzero values test whether
        dielectric/vacuum artificial-interface drift is dominated by missing
        normal-D transfer.  Production value is 0.0.
    material_sat_zhi_coarse_eps_blend : float
        Diagnostic-only blend for the coarse-side epsilon used by material SAT
        at the z-hi artificial interface.  ``0`` uses the coarse cell-center
        material, while ``1`` uses the restricted fine-side epsilon for the SAT
        impedance only.  This tests whether material-jump phase drift comes
        from coarse-side interface material aliasing.  Production value is 0.0.
    defer_material_h_sat_until_after_e : bool
        Diagnostic-only H-SAT ordering variant for material z slabs.  When
        true, material H-SAT is computed after the H update but withheld from
        the H state used by the subsequent E curl/E-SAT; the corrected H state
        is committed only after E-SAT.  This probes whether centered support
        needs a different time-centered two-interface operator.  Production
        value is False.
    material_sat_face_projection : {"node_adjoint", "sample"}
        Diagnostic-only material-SAT z-face restriction variant.  The
        production value is the current adjoint node-aligned projection;
        ``"sample"`` tests colocated fine-node restriction as a spatial
        transfer falsification.
    inject_sources_before_e_coupling : bool
        Diagnostic-only experiment: inject fine-grid electric sources before
        z-slab E SAT coupling instead of after. This tests source/SAT ordering
        as a blocker hypothesis and is not a public support mode.
    use_exterior_box_interfaces : bool
        Diagnostic-only local x/y experiment: for non-full-x/y boxes, couple
        fine faces to immediately exterior coarse planes and mask the coarse
        shadow volume, approximating disjoint local-box ownership.
    inject_sources_on_coarse_shadow : bool
        Diagnostic-only experiment: also inject soft source waveforms on the
        overlapping coarse grid when sources lie inside the refined region.
    """
    from rfx.sources.wire_radius import require_radius_update
    for mats in (mats_c, mats_f):
        require_radius_update(mats, lane="subgridded", unsupported=True)
    # Unpack the bundled options into same-named locals so the body below is
    # byte-unchanged from the historical keyword-argument signature.
    pec_mask_c = opts.pec_mask_c
    pec_mask_f = opts.pec_mask_f
    sources_f = opts.sources_f
    sources_c = opts.sources_c
    probe_indices_f = opts.probe_indices_f
    probe_components = opts.probe_components
    probe_indices_c = opts.probe_indices_c
    probe_components_c = opts.probe_components_c
    lumped_sparam_indices_f = opts.lumped_sparam_indices_f
    lumped_sparam_components_f = opts.lumped_sparam_components_f
    lumped_sparam_impedances_f = opts.lumped_sparam_impedances_f
    lumped_sparam_cell_counts_f = opts.lumped_sparam_cell_counts_f
    lumped_sparam_freqs_f = opts.lumped_sparam_freqs_f
    ntff_box_f = opts.ntff_box_f
    ntff_data_f = opts.ntff_data_f
    use_material_sat = opts.use_material_sat
    sync_coarse_interface_from_fine = opts.sync_coarse_interface_from_fine
    sync_coarse_shadow_from_fine = opts.sync_coarse_shadow_from_fine
    sync_box_coarse_shadow_from_fine = opts.sync_box_coarse_shadow_from_fine
    mask_coarse_shadow_interior = opts.mask_coarse_shadow_interior
    use_exterior_z_interfaces = opts.use_exterior_z_interfaces
    use_boundary_terminated_exterior_z_interfaces = (
        opts.use_boundary_terminated_exterior_z_interfaces
    )
    ghost_exterior_coarse_shadow_from_fine = (
        opts.ghost_exterior_coarse_shadow_from_fine
    )
    # The material_sat_* scale knobs are consumed directly from ``opts`` by
    # the _z_slab_material_coupling_{h,e}_3d helpers, so the runner body no
    # longer unpacks them here.
    defer_material_h_sat_until_after_e = opts.defer_material_h_sat_until_after_e
    inject_sources_before_e_coupling = opts.inject_sources_before_e_coupling
    use_exterior_box_interfaces = opts.use_exterior_box_interfaces
    inject_sources_on_coarse_shadow = opts.inject_sources_on_coarse_shadow
    sources_f = sources_f or []
    sources_c = sources_c or []
    probe_indices_f = probe_indices_f or []
    probe_components = probe_components or []
    probe_indices_c = probe_indices_c or []
    probe_components_c = probe_components_c or []
    lumped_sparam_indices_f = lumped_sparam_indices_f or []
    lumped_sparam_components_f = lumped_sparam_components_f or []
    lumped_sparam_impedances_f = lumped_sparam_impedances_f or []
    lumped_sparam_cell_counts_f = lumped_sparam_cell_counts_f or []
    use_ntff = ntff_box_f is not None and ntff_data_f is not None

    dt = config.dt
    dx_c = config.dx_c
    dx_f = config.dx_f

    mats_c, mats_f = subgrid_components(mats_c, mats_f, grid_c, config)

    # Override coarse grid dt to match subgrid global timestep
    grid_c.dt = dt

    # ---- Subsystem flags (resolved at trace time, not inside scan) ----
    use_cpml = grid_c.cpml_layers > 0
    pad_x_lo = int(getattr(grid_c, "pad_x_lo", grid_c.cpml_layers))
    pad_x_hi = int(getattr(grid_c, "pad_x_hi", grid_c.cpml_layers))
    pad_y_lo = int(getattr(grid_c, "pad_y_lo", grid_c.cpml_layers))
    pad_y_hi = int(getattr(grid_c, "pad_y_hi", grid_c.cpml_layers))
    pad_z_lo = int(getattr(grid_c, "pad_z_lo", grid_c.cpml_layers))
    pad_z_hi = int(getattr(grid_c, "pad_z_hi", grid_c.cpml_layers))
    is_full_xy_z_slab = (
        config.fi_lo == pad_x_lo
        and config.fj_lo == pad_y_lo
        and config.fi_hi == grid_c.nx - pad_x_hi
        and config.fj_hi == grid_c.ny - pad_y_hi
    )
    grid_nx = int(getattr(grid_c, "nx", config.nx_c))
    grid_ny = int(getattr(grid_c, "ny", config.ny_c))
    grid_nz = int(getattr(grid_c, "nz", config.nz_c))
    fine_physical_pec_faces = set()
    if not use_cpml:
        if config.fi_lo <= pad_x_lo:
            fine_physical_pec_faces.add("x_lo")
        if config.fi_hi >= grid_nx - pad_x_hi:
            fine_physical_pec_faces.add("x_hi")
        if config.fj_lo <= pad_y_lo:
            fine_physical_pec_faces.add("y_lo")
        if config.fj_hi >= grid_ny - pad_y_hi:
            fine_physical_pec_faces.add("y_hi")
        if config.fk_lo <= pad_z_lo:
            fine_physical_pec_faces.add("z_lo")
        if config.fk_hi >= grid_nz - pad_z_hi:
            fine_physical_pec_faces.add("z_hi")
    else:
        pec_faces = getattr(grid_c, "pec_faces", set())
        if "x_lo" in pec_faces and config.fi_lo <= pad_x_lo:
            fine_physical_pec_faces.add("x_lo")
        if "x_hi" in pec_faces and config.fi_hi >= grid_nx - pad_x_hi:
            fine_physical_pec_faces.add("x_hi")
        if "y_lo" in pec_faces and config.fj_lo <= pad_y_lo:
            fine_physical_pec_faces.add("y_lo")
        if "y_hi" in pec_faces and config.fj_hi >= grid_ny - pad_y_hi:
            fine_physical_pec_faces.add("y_hi")
        if "z_lo" in pec_faces and config.fk_lo <= pad_z_lo:
            fine_physical_pec_faces.add("z_lo")
        if "z_hi" in pec_faces and config.fk_hi >= grid_nz - pad_z_hi:
            fine_physical_pec_faces.add("z_hi")
    use_exterior_owned_z_slab = (
        use_exterior_z_interfaces
        or use_boundary_terminated_exterior_z_interfaces
    )
    use_exterior_owned_box = bool(use_exterior_box_interfaces) and not is_full_xy_z_slab
    sync_box_shadow_from_fine = (
        bool(sync_box_coarse_shadow_from_fine) and not is_full_xy_z_slab
    )

    def _sync_box_shadow(fields_c, fields_f, component_names):
        return _sync_z_slab_coarse_shadow_from_fine(
            fields_c,
            fields_f,
            config,
        )
    use_exterior_shadow_ghost = (
        ghost_exterior_coarse_shadow_from_fine
        and use_exterior_owned_z_slab
        and is_full_xy_z_slab
    )
    if use_exterior_z_interfaces and use_boundary_terminated_exterior_z_interfaces:
        raise ValueError(
            "use_exterior_z_interfaces and "
            "use_boundary_terminated_exterior_z_interfaces are mutually exclusive"
        )
    if use_exterior_z_interfaces:
        if not is_full_xy_z_slab:
            raise ValueError(
                "use_exterior_z_interfaces is only implemented for full-x/y z slabs"
            )
        if config.fk_lo <= 0 or config.fk_hi >= config.nz_c:
            raise ValueError(
                "use_exterior_z_interfaces needs active coarse cells immediately "
                "outside both z-slab faces"
            )
    if use_boundary_terminated_exterior_z_interfaces:
        if not is_full_xy_z_slab:
            raise ValueError(
                "use_boundary_terminated_exterior_z_interfaces is only "
                "implemented for full-x/y z slabs"
            )
        if use_cpml and not (
            ("z_lo" in fine_physical_pec_faces)
            ^ ("z_hi" in fine_physical_pec_faces)
        ):
            raise ValueError(
                "use_boundary_terminated_exterior_z_interfaces is only "
                "implemented when exactly one refined z face touches a PEC "
                "physical boundary; fine-grid CPML faces are not implemented"
            )
        if config.fk_lo > 0 and config.fk_hi < config.nz_c:
            raise ValueError(
                "use_boundary_terminated_exterior_z_interfaces requires the "
                "slab to touch at least one physical z boundary"
            )
        if config.fk_lo <= 0:
            fine_physical_pec_faces.add("z_lo")
        if config.fk_hi >= config.nz_c:
            fine_physical_pec_faces.add("z_hi")
    box_artificial_face_kwargs = {
        "exterior_interfaces": use_exterior_owned_box,
        "couple_xlo": (
            "x_lo" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fi_lo > 0)
        ),
        "couple_xhi": (
            "x_hi" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fi_hi < config.nx_c)
        ),
        "couple_ylo": (
            "y_lo" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fj_lo > 0)
        ),
        "couple_yhi": (
            "y_hi" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fj_hi < config.ny_c)
        ),
        "couple_zlo": (
            "z_lo" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fk_lo > 0)
        ),
        "couple_zhi": (
            "z_hi" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fk_hi < config.nz_c)
        ),
    }
    c_xy = (slice(config.fi_lo, config.fi_hi), slice(config.fj_lo, config.fj_hi))
    f_xy = (slice(None), slice(None))

    def _face_nonvacuum(mats, face_slice):
        return (
            jnp.any(jnp.abs(mats.eps_r[face_slice] - 1.0) > 1e-12)
            or jnp.any(jnp.abs(mats.mu_r[face_slice] - 1.0) > 1e-12)
        )

    # Material weighting is needed only when the artificial coarse/fine
    # interface itself sees non-vacuum impedance.  Material discontinuities
    # wholly inside the fine slab should use ordinary Yee material updates
    # internally and the established vacuum z-interface SAT at the slab faces.
    if use_material_sat is None:
        use_material_sat = bool(
            _face_nonvacuum(mats_c, (*c_xy, config.fk_lo))
            or _face_nonvacuum(mats_c, (*c_xy, config.fk_hi - 1))
            or _face_nonvacuum(mats_f, (*f_xy, 0))
            or _face_nonvacuum(mats_f, (*f_xy, -1))
        )

    cpml_params = None
    cpml_state = None
    if use_cpml:
        # apply_cpml_{e,h} are consumed by step_fn via the _make_step_fn ctx
        # dict (built from locals()), not referenced directly here.
        from rfx.boundaries.cpml import (  # noqa: F401
            init_cpml,
            apply_cpml_h,
            apply_cpml_e,
        )
        cpml_params, cpml_state = init_cpml(grid_c)

    # Initialize field states
    shape_c = (config.nx_c, config.ny_c, config.nz_c)
    shape_f = (config.nx_f, config.ny_f, config.nz_f)

    state_c = init_state(shape_c)
    state_f = init_state(shape_f)

    # Precompute source waveforms matrix (n_steps, n_sources)
    if sources_f:
        src_waveforms = jnp.stack([jnp.array(s[4]) for s in sources_f], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    if sources_c and inject_sources_on_coarse_shadow:
        src_waveforms_c = jnp.stack([jnp.array(s[4]) for s in sources_c], axis=-1)
    else:
        src_waveforms_c = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    src_meta = [(s[0], s[1], s[2], s[3]) for s in sources_f]
    src_meta_c = (
        [(s[0], s[1], s[2], s[3]) for s in sources_c]
        if inject_sources_on_coarse_shadow
        else []
    )
    prb_meta = [(p[0], p[1], p[2], c) for p, c in
                zip(probe_indices_f, probe_components)]
    n_probes = len(prb_meta)
    prb_meta_c = [(p[0], p[1], p[2], c) for p, c in
                  zip(probe_indices_c, probe_components_c)]
    n_probes_c = len(prb_meta_c)
    lumped_sparam_meta = [
        (p[0], p[1], p[2], c, float(z0))
        for p, c, z0 in zip(
            lumped_sparam_indices_f,
            lumped_sparam_components_f,
            lumped_sparam_impedances_f,
        )
    ]
    n_lumped_sparams = len(lumped_sparam_meta)
    if lumped_sparam_freqs_f is None:
        lumped_sparam_freqs_arr = jnp.zeros((0,), dtype=jnp.float32)
    else:
        lumped_sparam_freqs_arr = jnp.asarray(lumped_sparam_freqs_f, dtype=jnp.float32)
    lumped_sparam_impedances_arr = (
        jnp.asarray(lumped_sparam_impedances_f, dtype=jnp.float32)
        if n_lumped_sparams
        else jnp.zeros((0,), dtype=jnp.float32)
    )
    lumped_sparam_cell_counts_arr = (
        jnp.asarray(lumped_sparam_cell_counts_f, dtype=jnp.float32)
        if n_lumped_sparams
        else jnp.zeros((0,), dtype=jnp.float32)
    )

    use_pec_mask_c = pec_mask_c is not None
    use_pec_mask_f = pec_mask_f is not None

    # Pack carry — include CPML state only when CPML is active
    carry_init = {
        "c": (state_c.ex, state_c.ey, state_c.ez,
              state_c.hx, state_c.hy, state_c.hz),
        "f": (state_f.ex, state_f.ey, state_f.ez,
              state_f.hx, state_f.hy, state_f.hz),
    }
    if use_cpml:
        carry_init["cpml"] = cpml_state
    if n_lumped_sparams > 0:
        dft_shape = (n_lumped_sparams, lumped_sparam_freqs_arr.shape[0])
        carry_init["lumped_sparam_v"] = jnp.zeros(dft_shape, dtype=jnp.complex64)
        carry_init["lumped_sparam_i"] = jnp.zeros(dft_shape, dtype=jnp.complex64)
    if use_ntff:
        carry_init["ntff"] = ntff_data_f

    cpml_axes = "xyz"


    # Run scan
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms, src_waveforms_c)
    _step_locals = dict(locals())
    _step_ctx = {n: _step_locals[n] for n in _STEP_FN_CAPTURES if n in _step_locals}
    _missing_required = sorted(_STEP_FN_REQUIRED_CAPTURES - _step_ctx.keys())
    if _missing_required:
        raise RuntimeError(
            "run_subgridded_jit: step_fn capture(s) verified to be "
            f"unconditionally bound are absent from ctx: {_missing_required}. "
            "A capture in _STEP_FN_REQUIRED_CAPTURES was renamed or became "
            "conditionally bound -- ctx.get would otherwise mask it as None "
            "inside step_fn."
        )
    final_carry, probe_series = run_subgridded_scan(
        _step_ctx=_step_ctx,
        carry_init=carry_init,
        xs=xs,
    )
    time_series, time_series_c = probe_series

    # Unpack final state
    ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = final_carry["c"]
    ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = final_carry["f"]

    final_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                        hx=hx_c, hy=hy_c, hz=hz_c,
                        step=jnp.array(n_steps, dtype=jnp.int32))
    final_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                        hx=hx_f, hy=hy_f, hz=hz_f,
                        step=jnp.array(n_steps, dtype=jnp.int32))

    return SubgridResult(
        state_c=final_c,
        state_f=final_f,
        time_series=time_series,
        config=config,
        dt=dt,
        time_series_c=time_series_c,
        lumped_sparam_v_dft_f=(
            final_carry["lumped_sparam_v"] if n_lumped_sparams > 0 else None
        ),
        lumped_sparam_i_dft_f=(
            final_carry["lumped_sparam_i"] if n_lumped_sparams > 0 else None
        ),
        lumped_sparam_freqs_f=(
            lumped_sparam_freqs_arr if n_lumped_sparams > 0 else None
        ),
        lumped_sparam_impedances_f=(
            lumped_sparam_impedances_arr if n_lumped_sparams > 0 else None
        ),
        lumped_sparam_cell_counts_f=(
            lumped_sparam_cell_counts_arr if n_lumped_sparams > 0 else None
        ),
        ntff_data_f=final_carry["ntff"] if use_ntff else None,
        ntff_box_f=ntff_box_f if use_ntff else None,
    )


from rfx.stepping.subgrid_jit import run_subgridded_scan


from rfx.measurement.subgrid_lumped_dft import accumulate_lumped_sparam_dft
