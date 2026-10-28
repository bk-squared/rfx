"""Distributed slab graded legacy loop."""
from __future__ import annotations

from functools import partial
from jax import lax
from jax.sharding import PartitionSpec as P
from rfx import _realized
from rfx.boundaries.cpml import apply_cpml_e
from rfx.boundaries.cpml import apply_cpml_h
from rfx.boundaries.cpml import cpml_grid_metadata
from rfx.core.drives import drive_layout
from rfx.core.yee import FDTDState
from rfx.core.yee import MaterialArrays
from rfx.core.yee import component_h_materials
from rfx.model.electric_metrics import SlabElectricMetrics
from rfx.model.occupancy import publish_slab_occupancy
from rfx.stepping import segmented_scan
from rfx.stepping.rank import mesh_ranks
from rfx.stepping.rank import rank_shard_map
from rfx.stepping.slab import owned_values
from rfx.stepping.slab_exchange import exchange_e_yee_shmap as _exchange_e_ghosts_nu
from rfx.stepping.slab_exchange import exchange_h_yee_shmap as _exchange_h_ghosts_nu
from rfx.stepping.slab_update import _concrete_on_mesh
from rfx.stepping.slab_update import _slab_curl_kwargs
from rfx.stepping.slab_update import _update_e_dispersive_local_nu
from rfx.stepping.slab_update import cpml_absorbing_faces
from rfx.stepping.slab_update import cpml_slab_frame
from rfx.stepping.slab_update import inject_sources_shmap
from rfx.stepping.slab_update import magnetic_low_ghost
from rfx.stepping.slab_update import material_drive_scales
from rfx.stepping.slab_update import sample_probes_shmap
from rfx.stepping.slab_update import slab_dispersion_coeffs
from rfx.stepping.slab_update import slab_e_materials_shmap
from rfx.stepping.slab_update import update_e_nu_shmap
from rfx.stepping.slab_update import update_h_nu_shmap
from rfx.stepping.slab_walls import apply_pec_face_shmap as _apply_pec_face_nu_shmap
from rfx.stepping.slab_walls import apply_pec_mask_shmap as _apply_pec_mask_nu_shmap
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np


def run_graded_legacy_slab_loop(
    *,
    cell_sizes,
    checkpoint_every,
    cpml_params,
    cpml_state,
    curl_boundary,
    debye_coeffs,
    debye_state_init,
    drive_columns,
    drives,
    dt,
    e_wall_faces,
    emit_time_series,
    ghost,
    high_wall,
    inv_dx_h_sharded,
    inv_dx_sharded,
    inv_dy_h_rep,
    inv_dy_rep,
    inv_dz_h_rep,
    inv_dz_rep,
    lorentz_coeffs,
    lorentz_state_init,
    mesh,
    n_devices,
    n_prb,
    n_src,
    n_steps,
    n_warmup,
    nx_local,
    nx_per,
    nx_real,
    pad_x,
    prb_device_ids,
    prb_local_specs,
    rep,
    sharded_grid,
    sharded_materials,
    sharded_pec_mask,
    sharded_pec_occupancy,
    sharded_state,
    source_nodes,
    src_device_ids,
    src_local_specs,
    src_waveforms_rep,
    use_cpml,
    use_debye,
    use_dispersion,
    use_lorentz,
):
    def _update_h_shmap(st, mat, spacings, *, ranks):
        (inv_dx_sharded, inv_dy_rep, inv_dz_rep,
         inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep) = spacings[:6]
        # #1038 leg 4: body moved VERBATIM to
        # _distributed_common.update_h_nu_shmap, which distributed_v2.py's
        # `if is_nu:` branch also calls now. The eight locals this closure read
        # are passed explicitly; the call sites below did not move.
        return update_h_nu_shmap(
            st, mat, mesh, dt,
            inv_dx_sharded, inv_dy_rep, inv_dz_rep,
            inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep,
            ranks=ranks, nx_per=nx_per, nx=nx_real, high_wall=high_wall, cell_sizes=spacings[6:])

    def _update_e_shmap(st, mat, spacings, e_materials, *, ranks):
        inv_dx_sharded, inv_dy_rep, inv_dz_rep = spacings[:3]
        # #1038 leg 4: body moved VERBATIM to
        # _distributed_common.update_e_nu_shmap, shared with the `is_nu`
        # branch of distributed_v2.run_distributed. #1303: this lane hands it
        # the four-cell means run_fn built once, before the loop.
        return update_e_nu_shmap(
            st, mat, mesh, dt,
            inv_dx_sharded, inv_dy_rep, inv_dz_rep,
            nx_per, nx_real, e_materials=e_materials,
            ranks=ranks, high_wall=high_wall, boundary=curl_boundary, ghost=ghost, pad_x=pad_x)

    # ------------------------------------------------------------------
    # Phase 2D: dispersive E shmap helper
    # ------------------------------------------------------------------
    # Mirrors ``_update_e_shmap`` but routes the local update through
    # :func:`_update_e_dispersive_local_nu` (which delegates to
    # :func:`rfx.nonuniform._update_e_nu_dispersive`).  Critical: the
    # caller MUST pass the pre-snapshot ``e_old_*`` arrays so the ADE
    # polarisation update reads pre-exchange E values (see ADE
    # Ordering Contract in the docstring of
    # ``run_nonuniform_distributed_pec``).
    #
    # Pole arrays use the (n_devices*n_poles, nx_local, ny, nz) layout
    # documented on ``shard_debye_*`` / ``shard_lorentz_*``.  Inside
    # ``shard_map`` each device sees ``(n_poles, nx_local, ny, nz)``.
    if use_dispersion:
        def _update_e_dispersive_shmap(
            st, mat, db_st_in, lr_st_in, e_old_ex, e_old_ey, e_old_ez,
            debye_coeffs, lorentz_coeffs, spacings, e_materials,
            *, ranks,
        ):
            inv_dx_sharded, inv_dy_rep, inv_dz_rep = spacings[:3]
            # Static booleans captured by closure: which poles are active.
            _has_db = use_debye
            _has_lr = use_lorentz

            # Build the in_specs / out_specs based on which dispersion
            # branches are active so we never wire dummy poles into
            # shard_map (keeps the JIT graph honest).

            db_in_count = 0
            if _has_db:
                # 5 coeff arrays + 3 state arrays = 8 P("x") inputs
                db_in_count = 8
            lr_in_count = 0
            if _has_lr:
                # 6 coeff arrays + 6 state arrays = 12 P("x") inputs
                lr_in_count = 12

            # Build in_specs as a flat tuple
            base_in = (
                P("x"), P("x"), P("x"),       # ex, ey, ez (current)
                P("x"), P("x"), P("x"),       # hx, hy, hz (curl source)
                P(),                          # step
                P("x"), P("x"), P("x"),       # eps_r, sigma, mu_r
                P("x"), P(None), P(None),     # inv_dx, inv_dy, inv_dz
                P("x"), P("x"), P("x"),       # ex_old, ey_old, ez_old (snapshots)
                P("x"),                      # shared component materials
            )
            disp_in = tuple(P("x") for _ in range(db_in_count + lr_in_count))
            in_specs_full = base_in + disp_in

            # Out: ex, ey, ez, step, [+ db_px, db_py, db_pz if Debye]
            #      [+ lr_px, lr_py, lr_pz, lr_pxp, lr_pyp, lr_pzp if Lorentz]
            base_out = (P("x"), P("x"), P("x"), P())
            db_out = (P("x"), P("x"), P("x")) if _has_db else ()
            lr_out = (
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"),
            ) if _has_lr else ()
            out_specs_full = base_out + db_out + lr_out + (P("x"),)

            @partial(
                rank_shard_map,
                mesh=mesh,
                in_specs=in_specs_full,
                out_specs=out_specs_full,
                check_rep=False,
            )
            def _e_disp(*args, rank):
                # Unpack base inputs
                idx = 0
                ex = args[idx]
                idx += 1
                ey = args[idx]
                idx += 1
                ez = args[idx]
                idx += 1
                hx = args[idx]
                idx += 1
                hy = args[idx]
                idx += 1
                hz = args[idx]
                idx += 1
                step = args[idx]
                idx += 1
                eps_r = args[idx]
                idx += 1
                sigma = args[idx]
                idx += 1
                mu_r = args[idx]
                idx += 1
                invdx = args[idx]
                idx += 1
                invdy = args[idx]
                idx += 1
                invdz = args[idx]
                idx += 1
                ex_old_local = args[idx]
                idx += 1
                ey_old_local = args[idx]
                idx += 1
                ez_old_local = args[idx]
                idx += 1
                means = args[idx]
                idx += 1

                # Debye
                if _has_db:
                    from rfx.materials.debye import DebyeCoeffs, DebyeState
                    d_ca = args[idx]
                    idx += 1
                    d_cb = args[idx]
                    idx += 1
                    d_cc = args[idx]
                    idx += 1
                    d_alpha = args[idx]
                    idx += 1
                    d_beta = args[idx]
                    idx += 1
                    d_px = args[idx]
                    idx += 1
                    d_py = args[idx]
                    idx += 1
                    d_pz = args[idx]
                    idx += 1
                    db_local = (
                        slab_dispersion_coeffs(
                            DebyeCoeffs(ca=d_ca, cb=d_cb, cc=d_cc,
                                        alpha=d_alpha, beta=d_beta),
                            means, dt, nx_per, nx_real, rank=rank),
                        DebyeState(px=d_px, py=d_py, pz=d_pz),
                    )
                else:
                    db_local = None

                if _has_lr:
                    from rfx.materials.lorentz import (
                        LorentzCoeffs, LorentzState,
                    )
                    l_ca = args[idx]
                    idx += 1
                    l_cb = args[idx]
                    idx += 1
                    l_cc = args[idx]
                    idx += 1
                    l_a = args[idx]
                    idx += 1
                    l_b = args[idx]
                    idx += 1
                    l_c = args[idx]
                    idx += 1
                    l_px = args[idx]
                    idx += 1
                    l_py = args[idx]
                    idx += 1
                    l_pz = args[idx]
                    idx += 1
                    l_pxp = args[idx]
                    idx += 1
                    l_pyp = args[idx]
                    idx += 1
                    l_pzp = args[idx]
                    idx += 1
                    lr_local = (
                        slab_dispersion_coeffs(
                            LorentzCoeffs(ca=l_ca, cb=l_cb, cc=l_cc,
                                          a=l_a, b=l_b, c=l_c),
                            means, dt, nx_per, nx_real, rank=rank),
                        LorentzState(px=l_px, py=l_py, pz=l_pz,
                                     px_prev=l_pxp, py_prev=l_pyp,
                                     pz_prev=l_pzp),
                    )
                else:
                    lr_local = None

                _st = FDTDState(ex=ex, ey=ey, ez=ez,
                                hx=hx, hy=hy, hz=hz, step=step)
                _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

                new_st, new_db, new_lr = _update_e_dispersive_local_nu(
                    _st, _mat, dt, invdx, invdy, invdz,
                    debye=db_local, lorentz=lr_local,
                    e_old=(ex_old_local, ey_old_local, ez_old_local),
                    **_slab_curl_kwargs(curl_boundary, hx.shape, ghost, pad_x, rank, n_devices),
                )

                from rfx.boundaries.electric_coefficient import dispersive_curl
                absorber_curl = dispersive_curl(
                    db_local[0] if db_local is not None else None,
                    lr_local[0] if lr_local is not None else None, dt)
                outs = (new_st.ex, new_st.ey, new_st.ez, new_st.step)
                if _has_db:
                    outs = outs + (new_db.px, new_db.py, new_db.pz)
                if _has_lr:
                    outs = outs + (
                        new_lr.px, new_lr.py, new_lr.pz,
                        new_lr.px_prev, new_lr.py_prev, new_lr.pz_prev,
                    )
                return outs + (absorber_curl,)

            # Build flat input tuple matching in_specs_full ordering
            call_args = [
                st.ex, st.ey, st.ez,
                st.hx, st.hy, st.hz,
                st.step,
                mat.eps_r, mat.sigma, mat.mu_r,
                inv_dx_sharded, inv_dy_rep, inv_dz_rep,
                e_old_ex, e_old_ey, e_old_ez,
                e_materials,
            ]
            if _has_db:
                call_args.extend([
                    debye_coeffs.ca, debye_coeffs.cb, debye_coeffs.cc,
                    debye_coeffs.alpha, debye_coeffs.beta,
                    db_st_in.px, db_st_in.py, db_st_in.pz,
                ])
            if _has_lr:
                call_args.extend([
                    lorentz_coeffs.ca, lorentz_coeffs.cb, lorentz_coeffs.cc,
                    lorentz_coeffs.a, lorentz_coeffs.b, lorentz_coeffs.c,
                    lr_st_in.px, lr_st_in.py, lr_st_in.pz,
                    lr_st_in.px_prev, lr_st_in.py_prev, lr_st_in.pz_prev,
                ])

            results = _e_disp(ranks, *call_args)

            ridx = 0
            new_ex = results[ridx]
            ridx += 1
            new_ey = results[ridx]
            ridx += 1
            new_ez = results[ridx]
            ridx += 1
            new_step = results[ridx]
            ridx += 1

            new_db_st = None
            new_lr_st = None
            if _has_db:
                from rfx.materials.debye import DebyeState
                new_db_st = DebyeState(
                    px=results[ridx], py=results[ridx + 1], pz=results[ridx + 2],
                )
                ridx += 3
            if _has_lr:
                from rfx.materials.lorentz import LorentzState
                new_lr_st = LorentzState(
                    px=results[ridx], py=results[ridx + 1], pz=results[ridx + 2],
                    px_prev=results[ridx + 3], py_prev=results[ridx + 4],
                    pz_prev=results[ridx + 5],
                )
                ridx += 6

            new_state = st._replace(ex=new_ex, ey=new_ey, ez=new_ez,
                                    step=new_step)
            return new_state, new_db_st, new_lr_st, results[-1]

    def _inject_sources_shmap(st, src_vals_step, *, ranks):
        # Construct host constants while tracing: the outer scan closes over
        # Python metadata only, never concrete device or NumPy arrays.
        source_drives = drive_layout(src_local_specs, src_vals_step.dtype,
                                     electric_only_path="distributed_nu")
        return inject_sources_shmap(
            st, src_vals_step, mesh, n_src,
            src_local_specs, src_device_ids,
            ranks=ranks, drives=source_drives)

    def _sample_probes_shmap(st, *, ranks):
        return sample_probes_shmap(
            st, mesh, n_prb, prb_local_specs, prb_device_ids,
            reduce_devices=False,
            ranks=ranks)

    # ------------------------------------------------------------------
    # Phase 2C: shmap-wrapped CPML helpers (no-op when CPML disabled)
    # ------------------------------------------------------------------
    global_grid = cpml_grid_metadata(sharded_grid.global_grid)

    # The face profiles are dynamic jit inputs; only the Python cell sizes
    # remain static, as required by the local kernels' float conversions.
    from rfx.boundaries.cpml import CPMLAxisParams

    cpml_spacings = {}
    magnetic_cpml_spacings = {}
    if isinstance(cpml_params, CPMLAxisParams):
        cpml_spacings = {name: getattr(cpml_params, name) for name in (
            "dx_x_lo", "dx_x_hi", "dx_y_lo", "dx_y_hi", "dz_lo", "dz_hi",
        )}
        cpml_params = cpml_params._replace(**dict.fromkeys(cpml_spacings))
        if cpml_params.magnetic is not None:
            magnetic_cpml_spacings = {
                name: getattr(cpml_params.magnetic, name) for name in cpml_spacings
            }
            cpml_params = cpml_params._replace(
                magnetic=cpml_params.magnetic._replace(
                    **dict.fromkeys(magnetic_cpml_spacings)))
        cpml_params = jax.tree_util.tree_map(
            lambda arr: _concrete_on_mesh(arr, rep), cpml_params)

    def _apply_cpml_h_shmap(st, cs, cpml_params, mu_r, widths, *, ranks):
        @partial(
            rank_shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),  # ex, ey, ez
                P("x"), P("x"), P("x"),  # hx, hy, hz
                # x-face psi (rank-conditional inside)
                P("x"), P("x"), P("x"), P("x"),
                # y-face psi (sliced per rank)
                P("x"), P("x"), P("x"), P("x"),
                # z-face psi (sliced per rank)
                P("x"), P("x"), P("x"), P("x"),
                P("x"),  # mu_r slab (material-aware CPML coeff, #205)
                (P("x"), P(None), P(None)),
            ),
            out_specs=(
                P("x"), P("x"), P("x"),               # hx, hy, hz
                P("x"), P("x"), P("x"), P("x"),       # x-face psi
                P("x"), P("x"), P("x"), P("x"),       # y-face psi
                P("x"), P("x"), P("x"), P("x"),       # z-face psi
            ),
            check_rep=False,
        )
        def _h(ex, ey, ez, hx, hy, hz,
               psi_hy_xlo, psi_hy_xhi, psi_hz_xlo, psi_hz_xhi,
               psi_hx_ylo, psi_hx_yhi, psi_hz_ylo, psi_hz_yhi,
               psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi,
               mu_r_slab, cell_sizes_slab, *, rank):
            _st = FDTDState(ex=ex, ey=ey, ez=ez,
                            hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
            _cs = cs._replace(
                psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
                psi_hz_xlo=psi_hz_xlo, psi_hz_xhi=psi_hz_xhi,
                psi_hx_ylo=psi_hx_ylo, psi_hx_yhi=psi_hx_yhi,
                psi_hz_ylo=psi_hz_ylo, psi_hz_yhi=psi_hz_yhi,
                psi_hx_zlo=psi_hx_zlo, psi_hx_zhi=psi_hx_zhi,
                psi_hy_zlo=psi_hy_zlo, psi_hy_zhi=psi_hy_zhi,
            )
            mu = magnetic_low_ghost(mu_r_slab, rank, ghost, nx_per=nx_per, nx=nx_real, high_wall=high_wall)
            from rfx.model.electric_metrics import slab_wall_view
            widths = (slab_wall_view(cell_sizes_slab[0], nx_per, nx_real, rank,
                                    high_wall=high_wall), *cell_sizes_slab[1:])
            means = component_h_materials(MaterialArrays(None, None, mu),
                high_wall=(False, *high_wall[1:]), cell_sizes=widths)
            material_view = SimpleNamespace(mu_r=mu, components=SimpleNamespace(mu_update=means))
            new_st, new_cs = apply_cpml_h(
                _st, cpml_params, _cs, global_grid, materials=material_view,
                faces=cpml_absorbing_faces(global_grid), frame=cpml_slab_frame(ex.shape, ghost, pad_x),
                owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})
            return (new_st.hx, new_st.hy, new_st.hz,
                    new_cs.psi_hy_xlo, new_cs.psi_hy_xhi,
                    new_cs.psi_hz_xlo, new_cs.psi_hz_xhi,
                    new_cs.psi_hx_ylo, new_cs.psi_hx_yhi,
                    new_cs.psi_hz_ylo, new_cs.psi_hz_yhi,
                    new_cs.psi_hx_zlo, new_cs.psi_hx_zhi,
                    new_cs.psi_hy_zlo, new_cs.psi_hy_zhi)

        (hx, hy, hz,
         psi_hy_xlo, psi_hy_xhi, psi_hz_xlo, psi_hz_xhi,
         psi_hx_ylo, psi_hx_yhi, psi_hz_ylo, psi_hz_yhi,
         psi_hx_zlo, psi_hx_zhi, psi_hy_zlo, psi_hy_zhi) = _h(
            ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
            cs.psi_hy_xlo, cs.psi_hy_xhi, cs.psi_hz_xlo, cs.psi_hz_xhi,
            cs.psi_hx_ylo, cs.psi_hx_yhi, cs.psi_hz_ylo, cs.psi_hz_yhi,
            cs.psi_hx_zlo, cs.psi_hx_zhi, cs.psi_hy_zlo, cs.psi_hy_zhi,
            mu_r, widths,
        )
        new_st = st._replace(hx=hx, hy=hy, hz=hz)
        new_cs = cs._replace(
            psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
            psi_hz_xlo=psi_hz_xlo, psi_hz_xhi=psi_hz_xhi,
            psi_hx_ylo=psi_hx_ylo, psi_hx_yhi=psi_hx_yhi,
            psi_hz_ylo=psi_hz_ylo, psi_hz_yhi=psi_hz_yhi,
            psi_hx_zlo=psi_hx_zlo, psi_hx_zhi=psi_hx_zhi,
            psi_hy_zlo=psi_hy_zlo, psi_hy_zhi=psi_hy_zhi,
        )
        return new_st, new_cs

    def _apply_cpml_e_shmap(st, cs, cpml_params, materials, e_materials, e_curl_coeff, *, ranks):
        # The psi coefficient takes the permittivity the E update of this
        # step used (#1043): each component's four-cell edge mean -- the one
        # run_fn built before the loop, including dispersive models.
        @partial(
            rank_shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"), P("x"),  # x-face psi
                P("x"), P("x"), P("x"), P("x"),  # y-face psi
                P("x"), P("x"), P("x"), P("x"),  # z-face psi
                P("x"),  # materials (material-aware CPML coeff, #205)
                P("x"),  # the four-cell eps/sigma means
                P("x"),  # the E update's in-loop dispersive curl coefficient
            ),
            out_specs=(
                P("x"), P("x"), P("x"),               # ex, ey, ez
                P("x"), P("x"), P("x"), P("x"),       # x-face psi
                P("x"), P("x"), P("x"), P("x"),       # y-face psi
                P("x"), P("x"), P("x"), P("x"),       # z-face psi
            ),
            check_rep=False,
        )
        def _e(ex, ey, ez, hx, hy, hz,
               psi_ey_xlo, psi_ey_xhi, psi_ez_xlo, psi_ez_xhi,
               psi_ex_ylo, psi_ex_yhi, psi_ez_ylo, psi_ez_yhi,
               psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi,
               mat_slab, means, curl, *, rank):
            _st = FDTDState(ex=ex, ey=ey, ez=ez,
                            hx=hx, hy=hy, hz=hz, step=jnp.int32(0))
            from rfx.boundaries.electric_coefficient import loss_terms
            eps_r_slab = means[0]
            loss = loss_terms(means, dt) if curl is None else None
            _cs = cs._replace(
                psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
                psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
                psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
                psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
                psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
                psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
            )
            new_st, new_cs = apply_cpml_e(
                _st, cpml_params, _cs, global_grid, materials=mat_slab,
                inv_eps_r_update=tuple(1 / e for e in eps_r_slab), e_loss=loss, e_curl_coeff=curl, boundary=curl_boundary,
                faces=cpml_absorbing_faces(global_grid), frame=cpml_slab_frame(ex.shape, ghost, pad_x),
                owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})
            return (new_st.ex, new_st.ey, new_st.ez,
                    new_cs.psi_ey_xlo, new_cs.psi_ey_xhi,
                    new_cs.psi_ez_xlo, new_cs.psi_ez_xhi,
                    new_cs.psi_ex_ylo, new_cs.psi_ex_yhi,
                    new_cs.psi_ez_ylo, new_cs.psi_ez_yhi,
                    new_cs.psi_ex_zlo, new_cs.psi_ex_zhi,
                    new_cs.psi_ey_zlo, new_cs.psi_ey_zhi)

        (ex, ey, ez,
         psi_ey_xlo, psi_ey_xhi, psi_ez_xlo, psi_ez_xhi,
         psi_ex_ylo, psi_ex_yhi, psi_ez_ylo, psi_ez_yhi,
         psi_ex_zlo, psi_ex_zhi, psi_ey_zlo, psi_ey_zhi) = _e(
            ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz,
            cs.psi_ey_xlo, cs.psi_ey_xhi, cs.psi_ez_xlo, cs.psi_ez_xhi,
            cs.psi_ex_ylo, cs.psi_ex_yhi, cs.psi_ez_ylo, cs.psi_ez_yhi,
            cs.psi_ex_zlo, cs.psi_ex_zhi, cs.psi_ey_zlo, cs.psi_ey_zhi,
            materials, e_materials, e_curl_coeff,
        )
        new_st = st._replace(ex=ex, ey=ey, ez=ez)
        new_cs = cs._replace(
            psi_ey_xlo=psi_ey_xlo, psi_ey_xhi=psi_ey_xhi,
            psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
            psi_ex_ylo=psi_ex_ylo, psi_ex_yhi=psi_ex_yhi,
            psi_ez_ylo=psi_ez_ylo, psi_ez_yhi=psi_ez_yhi,
            psi_ex_zlo=psi_ex_zlo, psi_ex_zhi=psi_ex_zhi,
            psi_ey_zlo=psi_ey_zlo, psi_ey_zhi=psi_ey_zhi,
        )
        return new_st, new_cs

    # ------------------------------------------------------------------
    # Per-step scan body (Phase 2B/2C/2D ordering — see docstring)
    # ------------------------------------------------------------------
    def step_fn(carry, xs, invariants, e_materials=None, *, ranks):
        (sharded_materials, sharded_pec_mask, edge_keep,
         debye_coeffs, lorentz_coeffs, cpml_params, spacings) = invariants
        if cpml_spacings:
            cpml_params = cpml_params._replace(**cpml_spacings)
        if magnetic_cpml_spacings:
            cpml_params = cpml_params._replace(
                magnetic=cpml_params.magnetic._replace(**magnetic_cpml_spacings))
        _step_idx, src_vals = xs
        st = carry["fdtd"]
        cs = carry.get("cpml")
        db_st = carry.get("debye")
        lr_st = carry.get("lorentz")

        # 0. Phase 2D ADE Ordering Contract — snapshot E BEFORE any
        #    ghost exchange / field update.  These references stay valid
        #    even when ``st`` is replaced because jnp arrays are
        #    immutable.  When dispersion is inactive the snapshot is
        #    unused (and the JIT will dead-code-eliminate it).
        ex_old_snapshot = st.ex
        ey_old_snapshot = st.ey
        ez_old_snapshot = st.ez

        # 1. H update (NU)
        st = _update_h_shmap(st, sharded_materials, spacings, ranks=ranks)

        # 2. Phase 2C: CPML H correction (after H, before E exchange).
        if use_cpml:
            st, cs = _apply_cpml_h_shmap(st, cs, cpml_params, sharded_materials.mu_r,
                                        spacings[6:], ranks=ranks)


        # 3. Ghost exchange of H so the upcoming E update sees the
        #    neighbour rank's H at the seam.
        st = _exchange_h_ghosts_nu(st, mesh, n_devices, ranks=ranks)

        # 4. E update — Phase 2D dispatch:
        #    - no dispersion: standard NU E update (Phase 2B path)
        #    - Debye-only / Lorentz-only / mixed: dispersive NU ADE
        #      with the snapshotted ex_old/ey_old/ez_old (NEVER the
        #      post-ghost-exchange E from the previous step's stage 9).
        absorber_curl = None
        if use_dispersion:
            st, db_st, lr_st, absorber_curl = _update_e_dispersive_shmap(
                st, sharded_materials, db_st, lr_st,
                ex_old_snapshot, ey_old_snapshot, ez_old_snapshot,
                debye_coeffs, lorentz_coeffs, spacings, e_materials,
                ranks=ranks)
        else:
            st = _update_e_shmap(st, sharded_materials, spacings, e_materials, ranks=ranks)

        # 5. Phase 2C: CPML E correction (after E, before sources/PEC).
        if use_cpml:
            st, cs = _apply_cpml_e_shmap(
                st, cs, cpml_params, sharded_materials,
                e_materials, absorber_curl, ranks=ranks)

        # 5b. Published soft-PEC factors; both ghost rows carry one.
        if edge_keep is not None:
            keep_x, keep_y, keep_z = edge_keep
            st = st._replace(ex=st.ex * keep_x, ey=st.ey * keep_y,
                             ez=st.ez * keep_z)

        # 6. Source injection (rank-conditional via shard_map)
        st = _inject_sources_shmap(st, src_vals, ranks=ranks)

        # 8. PEC mask zeroing (geometry + override union).  No-op when
        #    sharded_pec_mask is None.
        if sharded_pec_mask is not None:
            st = _apply_pec_mask_nu_shmap(
                st, sharded_pec_mask, mesh, n_devices, nx_local, ranks=ranks)

        # 7. PEC on physical domain faces (X-faces are rank-conditional).
        st = _apply_pec_face_nu_shmap(st, mesh, n_devices, nx_local, pad_x=pad_x, pec_faces=e_wall_faces, ranks=ranks)

        # 9. Ghost exchange of E so the next step's H update sees the
        #    neighbour rank's E at the seam.  LAST in the E half-step, after
        #    the PEC stages: the H update at a rank's last real cell reads
        #    Ey/Ez on its right ghost plane, and that plane's PEC edges are
        #    zeroed by the OWNER rank (stages 7/8 impose the hard conductors).
        #    Exchanging before the PEC stages handed the ghost the
        #    un-zeroed value — the #931 seam-cell divergence (2.107e-01
        #    final-step error on a one-cell body at rank 1's first real
        #    cell; see the docstring).
        st = _exchange_e_ghosts_nu(st, mesh, n_devices, ranks=ranks)

        # 10. Owner-masked samples; reduce devices once after all scans.
        #     Phase 2F emit_time_series=False: skip the probe-sample
        #     accumulation entirely so the AD tape is not loaded with
        #     per-step probe entries.  ``probe_out`` becomes a 0-length
        #     array (still a valid scan output, just empty).
        if emit_time_series:
            probe_out = _sample_probes_shmap(st, ranks=ranks)
        else:
            probe_out = jnp.zeros(0, dtype=jnp.float32)

        new_carry = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cs
        if use_debye:
            new_carry["debye"] = db_st
        if use_lorentz:
            new_carry["lorentz"] = lr_st
        return new_carry, probe_out

    # ------------------------------------------------------------------
    # JIT-compiled scan over n_steps
    # ------------------------------------------------------------------
    # Composition note (Phase 2F / M5 spike): an eager
    # ``jax.checkpoint(shard_map(...))`` raises
    # ``NotImplementedError: Eager evaluation of closed_call inside
    # shard_map isn't yet supported``.  The fix is to keep the outer
    # ``jax.jit`` boundary so that ``jax.checkpoint(segment_body)``
    # composes cleanly inside JIT (the shard_map calls live inside
    # ``step_fn``, which is reached via the ``lax.scan(seg_body, ...)``
    # call inside the JIT'd ``run_fn``).  Both Pattern D
    # ``jit(checkpoint(shard_map))`` and Pattern E
    # ``checkpoint(jit(shard_map))`` were verified to work; we use
    # Pattern D (jit-around-checkpoint) because the outer ``jax.jit``
    # boundary is already required for the scan dispatch and adding it
    # avoids re-jitting the shard_map step body once per call.
    step_indices = _concrete_on_mesh(np.arange(n_steps, dtype=np.int32), rep)
    xs_full = (step_indices, src_waveforms_rep)

    carry_init = {"fdtd": sharded_state}
    if use_cpml:
        carry_init["cpml"] = cpml_state
    if use_debye:
        carry_init["debye"] = debye_state_init
    if use_lorentz:
        carry_init["lorentz"] = lorentz_state_init

    # Phase 2F warmup split: first ``n_warmup`` steps run with
    # ``stop_gradient`` carry boundary so AD never sees that transient.
    # The optimize phase (``n_steps - n_warmup`` steps) runs afterwards
    # and may itself be segmented via ``checkpoint_every``.  When
    # ``n_warmup == 0`` this branch reduces to the prior single-scan
    # path (Phase 2A-2E semantics preserved).
    if n_warmup > 0:
        warmup_xs = (
            xs_full[0][:n_warmup],
            xs_full[1][:n_warmup],
        )
        opt_xs = (
            xs_full[0][n_warmup:],
            xs_full[1][n_warmup:],
        )
        n_opt = n_steps - n_warmup
    else:
        warmup_xs = None
        opt_xs = xs_full
        n_opt = n_steps

    use_segmented = (
        checkpoint_every is not None
        and 0 < int(checkpoint_every) < n_opt
    )

    # Per-cell arrays must enter the jit as arguments: captured concrete
    # arrays become whole-domain compiled constants on every device. Keep
    # their tracer-valued counterparts shared by all scans, including remat.
    def _drive_xs(xs, scales, ranks, weights=None):
        from rfx.model.source_coefficients import slab_drive_table
        return slab_drive_table(xs, mesh, src_device_ids, src_local_specs,
                                ranks=ranks, scales=scales, columns=drive_columns,
                                volumes=tuple(d[4] for d in drives), weights=weights)

    edge_keep = None
    if sharded_pec_occupancy is not None:
        edge_keep = publish_slab_occupancy(
            sharded_materials, sharded_pec_occupancy, mesh).edge_keep

    @jax.jit
    def run_fn(c0, invariants, warmup_xs, opt_xs, *, ranks, e_cell_sizes):
        electric_grid = SlabElectricMetrics(e_cell_sizes, nx_real, nx_per, nx_local, high_wall)
        scales = None
        if drives:
            # #1279: the drive sees the materials the E update sees -- the
            # override, on the tape when traced -- read in the program.
            scales = material_drive_scales(invariants[0].eps_r, invariants[0].sigma, mesh, drives, dt,
                ranks=ranks, grid=electric_grid, sigma_film=invariants[0].sigma_film)
            if _realized.ACTIVE is not None:
                scales = jnp.broadcast_to(_realized.runtime_drive(jnp.sum(scales, axis=0), tuple(drive_columns), volumes=tuple(d[4] for d in drives)), scales.shape)
        # Occupancy is the last float32 drive factor, read slab-locally on the
        # source's owner; scans only add the stored samples.
        weights = None
        edge_keep = invariants[2]
        if edge_keep is not None and n_src:
            weights = owned_values(edge_keep, mesh, source_nodes, ranks=ranks)
        if warmup_xs is not None:
            warmup_xs = _drive_xs(warmup_xs, scales, ranks, weights)
        opt_xs = _drive_xs(opt_xs, scales, ranks, weights)
        # #1303: each component's four-cell mean, built once for the
        # warm-up and the optimize scans (and every remat segment) -- see
        # slab_e_materials_shmap for why not inside the loop. Dispersive E
        # coefficients are finished from these same means inside the loop.
        e_materials = slab_e_materials_shmap(
            invariants[0], mesh, nx_per, nx_real, ranks=ranks, grid=electric_grid, high_wall=high_wall)
        scan_step = partial(step_fn, invariants=invariants, e_materials=e_materials, ranks=ranks)
        # Optional warmup scan: stop_gradient the carry at boundary so
        # AD does not see the warmup steps.  Probe samples from the
        # warmup phase are also stop_gradient'd (they're just metadata
        # at this point — gradient through them would be a tape leak).
        if warmup_xs is not None:
            warmup_final, warmup_ys = lax.scan(scan_step, c0, warmup_xs)
            c0_opt = jax.tree_util.tree_map(lax.stop_gradient, warmup_final)
            warmup_ys = lax.stop_gradient(warmup_ys)
        else:
            c0_opt = c0
            warmup_ys = None

        # Checkpoint full segments and the real, unpadded tail.
        if use_segmented:
            final_, opt_ys = segmented_scan(
                scan_step, c0_opt, opt_xs, int(checkpoint_every), scan=lax.scan
            )
        else:
            final_, opt_ys = lax.scan(scan_step, c0_opt, opt_xs)

        if warmup_ys is not None:
            full_ys = jnp.concatenate([warmup_ys, opt_ys], axis=0)
        else:
            full_ys = opt_ys
        if emit_time_series:
            # (time, device, probe) -> replicated (time, probe), including
            # empty probes. Keep warmup stop_gradient above.
            full_ys = jnp.sum(full_ys, axis=1)
        return final_, full_ys

    final_carry, probe_ts = run_fn(
        carry_init,
        (sharded_materials, sharded_pec_mask, edge_keep,
         debye_coeffs, lorentz_coeffs, cpml_params,
         (inv_dx_sharded, inv_dy_rep, inv_dz_rep,
          inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep, *cell_sizes)),
        warmup_xs, opt_xs,
        ranks=mesh_ranks(mesh), e_cell_sizes=sharded_grid.e_cell_sizes)
    final_state_sharded = final_carry["fdtd"]
    final_cpml_sharded = final_carry.get("cpml") if use_cpml else None
    final_debye_sharded = final_carry.get("debye") if use_debye else None
    final_lorentz_sharded = final_carry.get("lorentz") if use_lorentz else None

    return final_cpml_sharded, final_debye_sharded, final_lorentz_sharded, final_state_sharded, probe_ts

