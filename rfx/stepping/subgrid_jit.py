from __future__ import annotations

import jax
import jax.numpy as jnp
from rfx.core.drives import drive_layout, inject_drives
from rfx.core.yee import FDTDState
from rfx.stepping import update_h, update_e
from rfx.boundaries.pec import apply_pec, apply_pec_faces, apply_pec_mask
from rfx.subgridding.sbp_sat_3d import _shared_node_coupling_3d, _shared_node_coupling_h_3d
from rfx.geometry.rasterize_grid import unpadded_high_walls, fine_high_walls
from rfx.stepping.subgrid_coupling import _box_node_aligned_coupling_e_3d
from rfx.stepping.subgrid_coupling import _box_node_aligned_coupling_h_3d
from rfx.stepping.subgrid_coupling import _mask_box_coarse_shadow_all
from rfx.stepping.subgrid_coupling import _mask_z_slab_coarse_shadow_all
from rfx.stepping.subgrid_coupling import _mask_z_slab_coarse_shadow_interior
from rfx.stepping.subgrid_coupling import _sync_z_slab_coarse_shadow_from_fine
from rfx.stepping.subgrid_coupling import _sync_z_slab_coarse_tangential_from_fine
from rfx.stepping.subgrid_coupling import _uses_endpoint_fine_shape
from rfx.stepping.subgrid_coupling import _z_slab_coupling_e_3d
from rfx.stepping.subgrid_coupling import _z_slab_coupling_h_3d
from rfx.stepping.subgrid_coupling import _z_slab_material_coupling_e_3d
from rfx.stepping.subgrid_coupling import _z_slab_material_coupling_h_3d
from rfx.measurement.subgrid_lumped_dft import accumulate_lumped_sparam_dft


_STEP_FN_CAPTURES = (
    "_sync_box_shadow",
    "apply_cpml_e",
    "apply_cpml_h",
    "box_artificial_face_kwargs",
    "config",
    "cpml_axes",
    "cpml_params",
    "defer_material_h_sat_until_after_e",
    "dt",
    "dx_c",
    "dx_f",
    "fine_physical_pec_faces",
    "grid_c",
    "inject_sources_before_e_coupling",
    "inject_sources_on_coarse_shadow",
    "is_full_xy_z_slab",
    "lumped_sparam_freqs_arr",
    "lumped_sparam_meta",
    "mask_coarse_shadow_interior",
    "mats_c",
    "mats_f",
    "n_lumped_sparams",
    "n_probes",
    "n_probes_c",
    "ntff_box_f",
    "opts",
    "pec_mask_c",
    "pec_mask_f",
    "prb_meta",
    "prb_meta_c",
    "src_waveforms",
    "src_waveforms_c",
    "src_meta",
    "src_meta_c",
    "sync_box_shadow_from_fine",
    "sync_coarse_interface_from_fine",
    "sync_coarse_shadow_from_fine",
    "use_boundary_terminated_exterior_z_interfaces",
    "use_cpml",
    "use_exterior_owned_box",
    "use_exterior_owned_z_slab",
    "use_exterior_shadow_ghost",
    "use_exterior_z_interfaces",
    "use_material_sat",
    "use_ntff",
    "use_pec_mask_c",
    "use_pec_mask_f",
)


# Subset of _STEP_FN_CAPTURES verified to be assigned unconditionally in
# run_subgridded_jit.  Every name here is bound on all run configurations, so
# its absence from the _make_step_fn ctx signals a refactor regression rather
# than a legitimately-skipped optional feature.  Conditionally-bound captures
# (e.g. apply_cpml_e / apply_cpml_h, imported only under ``if use_cpml``) are
# deliberately excluded -- their absence from ctx is expected.
_STEP_FN_REQUIRED_CAPTURES = frozenset({
    "config",
    "cpml_axes",
    "cpml_params",
    "dt",
    "dx_c",
    "dx_f",
    "grid_c",
    "lumped_sparam_freqs_arr",
    "lumped_sparam_meta",
    "mats_c",
    "mats_f",
    "n_lumped_sparams",
    "n_probes",
    "n_probes_c",
    "opts",
    "pec_mask_c",
    "pec_mask_f",
    "prb_meta",
    "prb_meta_c",
    "src_waveforms",
    "src_waveforms_c",
    "src_meta",
    "src_meta_c",
    "use_cpml",
    "use_material_sat",
    "use_ntff",
    "use_pec_mask_c",
    "use_pec_mask_f",
})


def _make_step_fn(ctx):
    """Build the jax.lax.scan body for run_subgridded_jit.

    Hoisted out of run_subgridded_jit so the runner stays within the
    GATE-B2 G1 function-size budget. ``ctx`` carries the values the scan
    body closes over; ctx.get keeps closure semantics for captures that
    are only conditionally bound and only conditionally used.
    """
    _sync_box_shadow = ctx.get("_sync_box_shadow")
    apply_cpml_e = ctx.get("apply_cpml_e")
    apply_cpml_h = ctx.get("apply_cpml_h")
    box_artificial_face_kwargs = ctx.get("box_artificial_face_kwargs")
    config = ctx.get("config")
    cpml_axes = ctx.get("cpml_axes")
    cpml_params = ctx.get("cpml_params")
    defer_material_h_sat_until_after_e = ctx.get("defer_material_h_sat_until_after_e")
    dt = ctx.get("dt")
    dx_c = ctx.get("dx_c")
    dx_f = ctx.get("dx_f")
    fine_physical_pec_faces = ctx.get("fine_physical_pec_faces")
    grid_c = ctx.get("grid_c")
    inject_sources_before_e_coupling = ctx.get("inject_sources_before_e_coupling")
    inject_sources_on_coarse_shadow = ctx.get("inject_sources_on_coarse_shadow")
    is_full_xy_z_slab = ctx.get("is_full_xy_z_slab")
    lumped_sparam_freqs_arr = ctx.get("lumped_sparam_freqs_arr")
    lumped_sparam_meta = ctx.get("lumped_sparam_meta")
    mask_coarse_shadow_interior = ctx.get("mask_coarse_shadow_interior")
    mats_c = ctx.get("mats_c")
    mats_f = ctx.get("mats_f")
    high_wall_c, high_wall_f = unpadded_high_walls(grid_c), fine_high_walls(grid_c, config)
    n_lumped_sparams = ctx.get("n_lumped_sparams")
    n_probes = ctx.get("n_probes")
    n_probes_c = ctx.get("n_probes_c")
    ntff_box_f = ctx.get("ntff_box_f")
    opts = ctx.get("opts")
    pec_mask_c = ctx.get("pec_mask_c")
    pec_mask_f = ctx.get("pec_mask_f")
    prb_meta = ctx.get("prb_meta")
    prb_meta_c = ctx.get("prb_meta_c")
    src_meta = ctx.get("src_meta")
    src_meta_c = ctx.get("src_meta_c")
    drives_f = drive_layout(src_meta, ctx["src_waveforms"].dtype, electric_only_path="subgridded fine")
    drives_c = drive_layout(src_meta_c, ctx["src_waveforms_c"].dtype, electric_only_path="subgridded coarse")
    sync_box_shadow_from_fine = ctx.get("sync_box_shadow_from_fine")
    sync_coarse_interface_from_fine = ctx.get("sync_coarse_interface_from_fine")
    sync_coarse_shadow_from_fine = ctx.get("sync_coarse_shadow_from_fine")
    use_boundary_terminated_exterior_z_interfaces = ctx.get("use_boundary_terminated_exterior_z_interfaces")
    use_cpml = ctx.get("use_cpml")
    use_exterior_owned_box = ctx.get("use_exterior_owned_box")
    use_exterior_owned_z_slab = ctx.get("use_exterior_owned_z_slab")
    use_exterior_shadow_ghost = ctx.get("use_exterior_shadow_ghost")
    use_exterior_z_interfaces = ctx.get("use_exterior_z_interfaces")
    use_material_sat = ctx.get("use_material_sat")
    use_ntff = ctx.get("use_ntff")
    use_pec_mask_c = ctx.get("use_pec_mask_c")
    use_pec_mask_f = ctx.get("use_pec_mask_f")
    def step_fn(carry, xs):
        step_idx, src_vals, src_vals_c = xs
        ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = carry["c"]
        ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = carry["f"]

        def _inject_fields(ex, ey, ez, drives, values):
            from rfx.core.yee import FDTDState
            st = inject_drives(FDTDState(ex, ey, ez, None, None, None, None),
                               drives, values)
            return st.ex, st.ey, st.ez

        def _inject_fine_sources(ex, ey, ez):
            return _inject_fields(ex, ey, ez, drives_f, src_vals)

        def _inject_coarse_sources(ex, ey, ez):
            return _inject_fields(ex, ey, ez, drives_c, src_vals_c)

        if sync_coarse_shadow_from_fine and is_full_xy_z_slab:
            ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _sync_z_slab_coarse_shadow_from_fine(
                (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                (ex_f, ey_f, ez_f, hx_f, hy_f, hz_f),
                config,
            )
        if sync_box_shadow_from_fine:
            ex_c, ey_c, ez_c = _sync_box_shadow(
                (ex_c, ey_c, ez_c),
                (ex_f, ey_f, ez_f),
                ("ex", "ey", "ez"),
            )
            hx_c, hy_c, hz_c = _sync_box_shadow(
                (hx_c, hy_c, hz_c),
                (hx_f, hy_f, hz_f),
                ("hx", "hy", "hz"),
            )
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            if use_exterior_shadow_ghost:
                ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _sync_z_slab_coarse_shadow_from_fine(
                    (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                    (ex_f, ey_f, ez_f, hx_f, hy_f, hz_f),
                    config,
                )
            else:
                ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _mask_z_slab_coarse_shadow_all(
                    (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                    config,
                )
        elif mask_coarse_shadow_interior and is_full_xy_z_slab:
            ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _mask_z_slab_coarse_shadow_interior(
                (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                config,
            )
        elif use_exterior_owned_box:
            ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _mask_box_coarse_shadow_all(
                (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                config,
            )

        # === Coarse H update ===
        st_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                         hx=hx_c, hy=hy_c, hz=hz_c,
                         step=step_idx)
        st_c = update_h(st_c, mats_c, dt, dx_c, high_wall=high_wall_c)
        if use_cpml:
            # materials= keeps the CPML impedance-matched inside dielectrics
            # (issue #205; free-space coefficients diverge when a dielectric
            # overlaps the absorber — same mechanism as #203/#204).
            st_c, cpml_new = apply_cpml_h(st_c, cpml_params, carry["cpml"],
                                           grid_c, cpml_axes, materials=mats_c)
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            hx_c, hy_c, hz_c = _mask_z_slab_coarse_shadow_all(
                (st_c.hx, st_c.hy, st_c.hz),
                config,
            )
            st_c = st_c._replace(hx=hx_c, hy=hy_c, hz=hz_c)
        elif use_exterior_owned_box:
            hx_c, hy_c, hz_c = _mask_box_coarse_shadow_all(
                (st_c.hx, st_c.hy, st_c.hz),
                config,
            )
            st_c = st_c._replace(hx=hx_c, hy=hy_c, hz=hz_c)

        # === Fine H update ===
        st_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                         hx=hx_f, hy=hy_f, hz=hz_f,
                         step=step_idx)
        st_f = update_h(st_f, mats_f, dt, dx_f, high_wall=high_wall_f)

        # === SAT H-coupling (tangential H on all 6 faces) ===
        if is_full_xy_z_slab:
            if use_material_sat:
                (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
                    _z_slab_material_coupling_h_3d(
                        (st_c.ex, st_c.ey, st_c.ez, st_c.hx, st_c.hy, st_c.hz),
                        (st_f.ex, st_f.ey, st_f.ez, st_f.hx, st_f.hy, st_f.hz),
                        mats_c,
                        mats_f,
                        config,
                        opts=opts,
                    )
            else:
                (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
                    _z_slab_coupling_h_3d(
                        (st_c.hx, st_c.hy, st_c.hz),
                        (st_f.hx, st_f.hy, st_f.hz),
                        config,
                        use_exterior_z_interfaces=use_exterior_z_interfaces,
                        use_boundary_terminated_exterior_z_interfaces=(
                            use_boundary_terminated_exterior_z_interfaces
                        ),
                    )
        else:
            if _uses_endpoint_fine_shape(config):
                (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
                    _box_node_aligned_coupling_h_3d(
                        (st_c.hx, st_c.hy, st_c.hz),
                        (st_f.hx, st_f.hy, st_f.hz),
                        config,
                        **box_artificial_face_kwargs,
                    )
            else:
                (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
                    _shared_node_coupling_h_3d(
                        (st_c.hx, st_c.hy, st_c.hz),
                        (st_f.hx, st_f.hy, st_f.hz),
                        config,
                    )
        if sync_coarse_interface_from_fine and is_full_xy_z_slab:
            hx_c_new, hy_c_new = _sync_z_slab_coarse_tangential_from_fine(
                hx_c_new,
                hy_c_new,
                hx_f_new,
                hy_f_new,
                config,
                use_exterior_z_interfaces=use_exterior_z_interfaces,
                use_boundary_terminated_exterior_z_interfaces=(
                    use_boundary_terminated_exterior_z_interfaces
                ),
            )
        if sync_coarse_shadow_from_fine and is_full_xy_z_slab:
            hx_c_new, hy_c_new, hz_c_new = _sync_z_slab_coarse_shadow_from_fine(
                (hx_c_new, hy_c_new, hz_c_new),
                (hx_f_new, hy_f_new, hz_f_new),
                config,
            )
        if sync_box_shadow_from_fine:
            hx_c_new, hy_c_new, hz_c_new = _sync_box_shadow(
                (hx_c_new, hy_c_new, hz_c_new),
                (hx_f_new, hy_f_new, hz_f_new),
                ("hx", "hy", "hz"),
            )
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            hx_c_new, hy_c_new, hz_c_new = _mask_z_slab_coarse_shadow_all(
                (hx_c_new, hy_c_new, hz_c_new),
                config,
            )
        elif mask_coarse_shadow_interior and is_full_xy_z_slab:
            hx_c_new, hy_c_new, hz_c_new = _mask_z_slab_coarse_shadow_interior(
                (hx_c_new, hy_c_new, hz_c_new),
                config,
            )
        elif use_exterior_owned_box:
            hx_c_new, hy_c_new, hz_c_new = _mask_box_coarse_shadow_all(
                (hx_c_new, hy_c_new, hz_c_new),
                config,
            )
        defer_h_sat_for_e = (
            defer_material_h_sat_until_after_e
            and is_full_xy_z_slab
            and use_material_sat
        )
        if not defer_h_sat_for_e:
            st_c = st_c._replace(hx=hx_c_new, hy=hy_c_new, hz=hz_c_new)
            st_f = st_f._replace(hx=hx_f_new, hy=hy_f_new, hz=hz_f_new)
        if use_exterior_shadow_ghost and not defer_h_sat_for_e:
            ex_c_ghost, ey_c_ghost, ez_c_ghost, hx_c_ghost, hy_c_ghost, hz_c_ghost = (
                _sync_z_slab_coarse_shadow_from_fine(
                    (st_c.ex, st_c.ey, st_c.ez, st_c.hx, st_c.hy, st_c.hz),
                    (st_f.ex, st_f.ey, st_f.ez, st_f.hx, st_f.hy, st_f.hz),
                    config,
                )
            )
            st_c = st_c._replace(
                ex=ex_c_ghost,
                ey=ey_c_ghost,
                ez=ez_c_ghost,
                hx=hx_c_ghost,
                hy=hy_c_ghost,
                hz=hz_c_ghost,
            )

        # === Coarse E update + boundary ===
        st_c = update_e(st_c, mats_c, dt, dx_c, high_wall=high_wall_c)
        if use_cpml:
            st_c, cpml_new = apply_cpml_e(st_c, cpml_params, cpml_new,
                                           grid_c, cpml_axes, materials=mats_c)
        if use_pec_mask_c:
            # #931: ``apply_pec_mask`` IS the shared realization
            # (realized_pec_edge_masks, volume branch). This lane
            # carries VOLUMES only — run() refuses a declared sheet
            # or wire before it gets here (rfx/api/_execute.py) — and
            # has no periodic branch, so the default #689 flags are
            # the run's own.
            st_c = apply_pec_mask(st_c, pec_mask_c)
        st_c = apply_pec(st_c)
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            ex_c, ey_c, ez_c = _mask_z_slab_coarse_shadow_all(
                (st_c.ex, st_c.ey, st_c.ez),
                config,
            )
            st_c = st_c._replace(ex=ex_c, ey=ey_c, ez=ez_c)
        elif use_exterior_owned_box:
            ex_c, ey_c, ez_c = _mask_box_coarse_shadow_all(
                (st_c.ex, st_c.ey, st_c.ez),
                config,
            )
            st_c = st_c._replace(ex=ex_c, ey=ey_c, ez=ez_c)

        # === Fine E update + PEC mask ===
        st_f = update_e(st_f, mats_f, dt, dx_f, high_wall=high_wall_f)
        if use_pec_mask_f:
            # #931: ``apply_pec_mask`` IS the shared realization
            # (realized_pec_edge_masks, volume branch). This lane
            # carries VOLUMES only — run() refuses a declared sheet
            # or wire before it gets here (rfx/api/_execute.py) — and
            # has no periodic branch, so the default #689 flags are
            # the run's own.
            st_f = apply_pec_mask(st_f, pec_mask_f)
        if fine_physical_pec_faces:
            st_f = apply_pec_faces(st_f, fine_physical_pec_faces)
        if inject_sources_before_e_coupling:
            ex_src, ey_src, ez_src = _inject_fine_sources(st_f.ex, st_f.ey, st_f.ez)
            st_f = st_f._replace(ex=ex_src, ey=ey_src, ez=ez_src)

        # === SBP-SAT coupling ===
        if is_full_xy_z_slab:
            if use_material_sat:
                (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
                    _z_slab_material_coupling_e_3d(
                        (
                            st_c.ex,
                            st_c.ey,
                            st_c.ez,
                            st_c.hx,
                            st_c.hy,
                            st_c.hz,
                        ),
                        (
                            st_f.ex,
                            st_f.ey,
                            st_f.ez,
                            st_f.hx,
                            st_f.hy,
                            st_f.hz,
                        ),
                        mats_c,
                        mats_f,
                        config,
                        opts=opts,
                    )
            else:
                (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
                    _z_slab_coupling_e_3d(
                        (st_c.ex, st_c.ey, st_c.ez),
                        (st_f.ex, st_f.ey, st_f.ez),
                        config,
                        use_exterior_z_interfaces=use_exterior_z_interfaces,
                        use_boundary_terminated_exterior_z_interfaces=(
                            use_boundary_terminated_exterior_z_interfaces
                        ),
                    )
        else:
            if _uses_endpoint_fine_shape(config):
                (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
                    _box_node_aligned_coupling_e_3d(
                        (st_c.ex, st_c.ey, st_c.ez),
                        (st_f.ex, st_f.ey, st_f.ez),
                        config,
                        **box_artificial_face_kwargs,
                    )
            else:
                (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
                    _shared_node_coupling_3d(
                        (st_c.ex, st_c.ey, st_c.ez),
                        (st_f.ex, st_f.ey, st_f.ez),
                        config,
                    )
        if sync_coarse_interface_from_fine and is_full_xy_z_slab:
            ex_c_new, ey_c_new = _sync_z_slab_coarse_tangential_from_fine(
                ex_c_new,
                ey_c_new,
                ex_f_new,
                ey_f_new,
                config,
                use_exterior_z_interfaces=use_exterior_z_interfaces,
                use_boundary_terminated_exterior_z_interfaces=(
                    use_boundary_terminated_exterior_z_interfaces
                ),
            )

        if n_lumped_sparams > 0:
            v_dft, i_dft = accumulate_lumped_sparam_dft(
                carry=carry,
                dt=dt,
                dx_f=dx_f,
                ex_f_new=ex_f_new,
                ey_f_new=ey_f_new,
                ez_f_new=ez_f_new,
                lumped_sparam_freqs_arr=lumped_sparam_freqs_arr,
                lumped_sparam_meta=lumped_sparam_meta,
                st_f=st_f,
                step_idx=step_idx,
            )

        # === Source injection on fine grid ===
        if not inject_sources_before_e_coupling:
            ex_f_new, ey_f_new, ez_f_new = _inject_fine_sources(
                ex_f_new,
                ey_f_new,
                ez_f_new,
            )

        if not use_cpml:
            st_c_after = apply_pec(
                FDTDState(
                    ex=ex_c_new, ey=ey_c_new, ez=ez_c_new,
                    hx=st_c.hx, hy=st_c.hy, hz=st_c.hz,
                    step=step_idx,
                )
            )
            ex_c_new, ey_c_new, ez_c_new = st_c_after.ex, st_c_after.ey, st_c_after.ez
        if fine_physical_pec_faces:
            st_f_after = apply_pec_faces(
                FDTDState(
                    ex=ex_f_new, ey=ey_f_new, ez=ez_f_new,
                    hx=st_f.hx, hy=st_f.hy, hz=st_f.hz,
                    step=step_idx,
                ),
                fine_physical_pec_faces,
            )
            ex_f_new, ey_f_new, ez_f_new = st_f_after.ex, st_f_after.ey, st_f_after.ez
        if sync_coarse_shadow_from_fine and is_full_xy_z_slab:
            ex_c_new, ey_c_new, ez_c_new = _sync_z_slab_coarse_shadow_from_fine(
                (ex_c_new, ey_c_new, ez_c_new),
                (ex_f_new, ey_f_new, ez_f_new),
                config,
            )
        if sync_box_shadow_from_fine:
            ex_c_new, ey_c_new, ez_c_new = _sync_box_shadow(
                (ex_c_new, ey_c_new, ez_c_new),
                (ex_f_new, ey_f_new, ez_f_new),
                ("ex", "ey", "ez"),
            )
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            ex_c_new, ey_c_new, ez_c_new = _mask_z_slab_coarse_shadow_all(
                (ex_c_new, ey_c_new, ez_c_new),
                config,
            )
        elif mask_coarse_shadow_interior and is_full_xy_z_slab:
            ex_c_new, ey_c_new, ez_c_new = _mask_z_slab_coarse_shadow_interior(
                (ex_c_new, ey_c_new, ez_c_new),
                config,
            )
        elif use_exterior_owned_box:
            ex_c_new, ey_c_new, ez_c_new = _mask_box_coarse_shadow_all(
                (ex_c_new, ey_c_new, ez_c_new),
                config,
            )
        if inject_sources_on_coarse_shadow and src_meta_c:
            ex_c_new, ey_c_new, ez_c_new = _inject_coarse_sources(
                ex_c_new,
                ey_c_new,
                ez_c_new,
            )
        if defer_h_sat_for_e:
            st_c = st_c._replace(hx=hx_c_new, hy=hy_c_new, hz=hz_c_new)
            st_f = st_f._replace(hx=hx_f_new, hy=hy_f_new, hz=hz_f_new)

        # === Probe samples ===
        if n_probes > 0:
            def _get_field(comp, i, j, k):
                if comp == "ez":
                    return ez_f_new[i, j, k]
                if comp == "ex":
                    return ex_f_new[i, j, k]
                if comp == "ey":
                    return ey_f_new[i, j, k]
                if comp == "hx":
                    return st_f.hx[i, j, k]
                if comp == "hy":
                    return st_f.hy[i, j, k]
                return st_f.hz[i, j, k]

            samples = [_get_field(pc, pi, pj, pk)
                       for pi, pj, pk, pc in prb_meta]
            probe_out = jnp.stack(samples)
        else:
            probe_out = jnp.zeros(0, dtype=jnp.float32)

        if n_probes_c > 0:
            def _get_coarse_field(comp, i, j, k):
                if comp == "ez":
                    return ez_c_new[i, j, k]
                if comp == "ex":
                    return ex_c_new[i, j, k]
                if comp == "ey":
                    return ey_c_new[i, j, k]
                if comp == "hx":
                    return st_c.hx[i, j, k]
                if comp == "hy":
                    return st_c.hy[i, j, k]
                return st_c.hz[i, j, k]

            coarse_samples = [_get_coarse_field(pc, pi, pj, pk)
                              for pi, pj, pk, pc in prb_meta_c]
            probe_out_c = jnp.stack(coarse_samples)
        else:
            probe_out_c = jnp.zeros(0, dtype=jnp.float32)

        if use_ntff:
            from rfx.farfield import accumulate_ntff

            ntff_state_f = FDTDState(
                ex=ex_f_new,
                ey=ey_f_new,
                ez=ez_f_new,
                hx=st_f.hx,
                hy=st_f.hy,
                hz=st_f.hz,
                step=step_idx,
            )
            ntff_new = accumulate_ntff(
                carry["ntff"],
                ntff_state_f,
                ntff_box_f,
                dt,
                step_idx,
            )

        new_carry = {
            "c": (ex_c_new, ey_c_new, ez_c_new,
                  st_c.hx, st_c.hy, st_c.hz),
            "f": (ex_f_new, ey_f_new, ez_f_new,
                  st_f.hx, st_f.hy, st_f.hz),
        }
        if use_cpml:
            new_carry["cpml"] = cpml_new
        if n_lumped_sparams > 0:
            new_carry["lumped_sparam_v"] = v_dft
            new_carry["lumped_sparam_i"] = i_dft
        if use_ntff:
            new_carry["ntff"] = ntff_new

        return new_carry, (probe_out, probe_out_c)
    return step_fn


def run_subgridded_scan(*, _step_ctx, carry_init, xs):
    step_fn = _make_step_fn(_step_ctx)
    final_carry, probe_series = jax.lax.scan(step_fn, carry_init, xs)
    return final_carry, probe_series
