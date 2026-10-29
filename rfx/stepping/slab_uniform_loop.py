"""Distributed slab uniform loop."""
from __future__ import annotations

from functools import partial
from jax import lax
from jax.sharding import PartitionSpec as P
from rfx.boundaries.cpml import cpml_grid_metadata
from rfx.core.drives import drive_layout
from rfx.core.yee import FDTDState
from rfx.core.yee import MaterialArrays
from rfx.materials.debye import DebyeCoeffs
from rfx.materials.debye import DebyeState
from rfx.materials.lorentz import LorentzCoeffs
from rfx.materials.lorentz import LorentzState
from rfx.stepping.rank import mesh_ranks
from rfx.stepping.rank import rank_shard_map
from rfx.stepping.slab import cut
from rfx.stepping.slab_cpml import _apply_cpml_e_shmap
from rfx.stepping.slab_cpml import _apply_cpml_h_shmap
from rfx.stepping.slab_exchange import exchange_component_shmap as _exchange_component_shmap
from rfx.stepping.slab_exchange import exchange_e_yee_shmap as _exchange_e_ghosts_shmap
from rfx.stepping.slab_exchange import exchange_h_yee_shmap as _exchange_h_ghosts_shmap
from rfx.stepping.slab_update import _slab_curl_kwargs
from rfx.stepping.slab_update import _update_h_local
from rfx.stepping.slab_update import inject_sources_shmap
from rfx.stepping.slab_update import magnetic_low_ghost
from rfx.stepping.slab_update import sample_probes_shmap
from rfx.stepping.slab_update import slab_e_materials_shmap
from rfx.stepping.slab_walls import apply_pec_face_shmap as _apply_pec_shmap
from rfx.stepping.slab_walls import apply_pec_mask_shmap
import jax
import jax.numpy as jnp


def run_uniform_slab_loop(
    *,
    SlabRecords,
    _pmc_faces_frozen,
    cpml_params,
    cpml_state_sharded,
    curl_boundary,
    debye_coeffs_sharded,
    debye_state_sharded,
    dt,
    dx,
    e_wall_faces,
    exchange_interval,
    ghost,
    grid,
    has_debye,
    has_lorentz,
    high_wall,
    layout,
    lorentz_coeffs_sharded,
    lorentz_state_sharded,
    mesh,
    n_cpml,
    n_devices,
    n_prb,
    n_src,
    n_steps,
    nx,
    nx_local,
    nx_per,
    pad_x,
    ports,
    prb_device_ids,
    prb_local_specs,
    rep,
    sharded_materials,
    sharded_pec_mask,
    sharded_state,
    sim,
    snapshot,
    src_device_ids,
    src_local_specs,
    src_waveforms,
    use_cpml,
):
    _exchange_interval = int(exchange_interval)

    def _exchange(exchange, components, st, step_idx, *, ranks):
        # The entry refuses any interval but 1, so production steps always
        # run the minimal exchange. The skipping branch is reachable only when
        # a test bypasses validate_exchange_interval (the K=2 negative control,
        # test_distributed.py test_pec_window_peak_invariant); it keeps the
        # legacy all-component exchange whose stale-seam growth that test
        # measures.
        if _exchange_interval == 1:
            return exchange(st, mesh, n_devices, ranks=ranks)

        def _legacy(s):
            return s._replace(**{
                name: _exchange_component_shmap(getattr(s, name), mesh, n_devices, ranks=ranks)
                for name in components
            })

        return lax.cond(
            step_idx % _exchange_interval == 0, _legacy, lambda s: s, st)

    # ------------------------------------------------------------------
    # Per-device source/probe injection via shard_map
    # ------------------------------------------------------------------

    def _inject_sources_shmap(st, src_vals_step, *, ranks):
        """Inject sources on their owning device using shard_map."""
        # Construct host constants while tracing: the outer scan closes over
        # Python metadata only, never concrete device or NumPy arrays.
        source_drives = drive_layout(src_local_specs, src_vals_step.dtype,
                                     electric_only_path="distributed_v2")
        return inject_sources_shmap(
            st, src_vals_step, mesh, n_src,
            src_local_specs, src_device_ids,
            ranks=ranks, drives=source_drives)

    def _sample_probes_shmap(st, *, ranks):
        """Keep masked samples local until the scan has finished."""
        return sample_probes_shmap(
            st, mesh, n_prb, prb_local_specs, prb_device_ids,
            reduce_devices=False,
            ranks=ranks)

    # ------------------------------------------------------------------
    # H and E local updates via shard_map
    # ------------------------------------------------------------------

    from rfx.boundaries.pmc import magnetic_update_factors
    magnetic_factors = magnetic_update_factors(_pmc_faces_frozen, grid.shape)
    magnetic_factors = tuple(
        cut(factor, layout, "pmc_h_factor", mesh=mesh) if axis == 0 and factor is not None
        else factor for axis, factor in enumerate(magnetic_factors))

    def _update_h_shmap(st, mat, *, ranks):

        @partial(
            rank_shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),  # ex, ey, ez
                P("x"), P("x"), P("x"),  # hx, hy, hz
                P(),                     # step (replicated)
                P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
                (P("x"), P(), P()),       # optional PMC axis factors
            ),
            out_specs=(P("x"), P("x"), P("x"), P()),  # hx, hy, hz, step
            check_rep=False,
        )
        def _h(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r, factors, *, rank):
            _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
            _mat = MaterialArrays(eps_r=eps_r, sigma=sigma,
                                 mu_r=magnetic_low_ghost(mu_r, rank, ghost, nx_per=nx_per, nx=nx, high_wall=high_wall))
            new_st = _update_h_local(_st, _mat, dt, dx, factors, high_wall=(False, *high_wall[1:]))
            return new_st.hx, new_st.hy, new_st.hz, new_st.step

        hx, hy, hz, step = _h(
            ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
            mat.eps_r, mat.sigma, mat.mu_r, magnetic_factors)
        return st._replace(hx=hx, hy=hy, hz=hz, step=step)

    def _update_e_shmap(st, mat, db_coeffs, db_st, lr_coeffs, lr_st, e_materials, *, ranks):
        """E update (with optional dispersion) via shard_map."""

        @partial(
            rank_shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),  # ex, ey, ez
                P("x"), P("x"), P("x"),  # hx, hy, hz
                P(),                     # step
                P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
                # #1236 lumped records: None or (x, y, z) of arrays / None
                P("x"), P("x"), P("x"),
                P("x"),  # shared component means
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
                P("x"),                         # in-loop dispersive curl coefficient
            ),
            check_rep=False,
        )
        def _e(ex, ey, ez, hx, hy, hz, step,
               eps_r, sigma, mu_r, sigma_lumped, eps_r_lumped, sigma_film, means,
               d_ca, d_cb, d_cc, d_alpha, d_beta,
               d_px, d_py, d_pz,
               l_ca, l_cb, l_cc, l_a, l_b, l_c,
               l_px, l_py, l_pz, l_px_prev, l_py_prev, l_pz_prev, *, rank):
            _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
            _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r,
                                  sigma_film=sigma_film,
                                  sigma_lumped=sigma_lumped,
                                  eps_r_lumped=eps_r_lumped)
            _db = (DebyeCoeffs(ca=d_ca, cb=d_cb, cc=d_cc, alpha=d_alpha, beta=d_beta),
                   DebyeState(px=d_px, py=d_py, pz=d_pz)) if has_debye else None
            _lr = (LorentzCoeffs(ca=l_ca, cb=l_cb, cc=l_cc, a=l_a, b=l_b, c=l_c),
                   LorentzState(px=l_px, py=l_py, pz=l_pz,
                                px_prev=l_px_prev, py_prev=l_py_prev, pz_prev=l_pz_prev)) \
                   if has_lorentz else None
            from rfx.stepping.absorber import local_e_with_absorber
            new_st, new_db, new_lr, absorber = local_e_with_absorber(
                _st, _mat, dt, dx, debye=_db, lorentz=_lr,
                slab=(nx_per, nx), e_materials=means, rank=rank, high_wall=high_wall,
                **_slab_curl_kwargs(curl_boundary, hx.shape, ghost, pad_x, rank, n_devices))
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
                    nl_pxp, nl_pyp, nl_pzp, absorber)

        (ex, ey, ez, step,
         nd_px, nd_py, nd_pz,
         nl_px, nl_py, nl_pz,
         nl_pxp, nl_pyp, nl_pzp, absorber) = _e(
            ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
            mat.eps_r, mat.sigma, mat.mu_r, mat.sigma_lumped, mat.eps_r_lumped, mat.sigma_film, e_materials,
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
        return new_st, new_db_st, new_lr_st, absorber

    records = SlabRecords(sim, grid, nx_per, mesh, sharded_state.ex.dtype, n_steps, snapshot)
    from rfx.stepping import slab_tfsf
    tfsf_cfg, tfsf_line, tfsf_tables = slab_tfsf.prepare(sim, grid, n_steps, layout, mesh)

    # ------------------------------------------------------------------
    # Step function (operates on sharded arrays)
    # ------------------------------------------------------------------

    cpml_grid = cpml_grid_metadata(grid)
    cpml_spacings = {}
    magnetic_cpml_spacings = {}
    if use_cpml:
        # Face widths are static grid data; only array leaves enter the jit.
        cpml_spacings = {name: float(getattr(cpml_params, name)) for name in (
            "dx_x_lo", "dx_x_hi", "dx_y_lo", "dx_y_hi", "dz_lo", "dz_hi")}
        cpml_params = cpml_params._replace(**dict.fromkeys(cpml_spacings))
        if cpml_params.magnetic is not None:
            magnetic_cpml_spacings = {
                name: float(getattr(cpml_params.magnetic, name)) for name in cpml_spacings}
            cpml_params = cpml_params._replace(magnetic=cpml_params.magnetic._replace(
                **dict.fromkeys(magnetic_cpml_spacings)))

    def step_fn_cpml(
        carry, xs, materials_arg, debye_coeffs_arg, lorentz_coeffs_arg,
        cpml_params_arg, pec_mask_arg, e_materials, tfsf_tables_arg, port_table_arg,
        *, ranks,
    ):
        """Single FDTD step (CPML path) operating on sharded arrays.

        Stage order (#1041)::

            H -> TF/SF-H -> CPML-H -> exch H -> E -> TF/SF-E -> CPML-E
              -> sources -> PEC face -> PEC mask -> exch E -> probes

        The E ghost exchange is the LAST stage of the E half-step, so a
        ghost row is always a copy of the owner's FINISHED real row -- a
        row the realized-PEC mask stage has already zeroed where the
        geometry says metal (#1053). This is ``distributed_nu.py``'s order
        since ``ac782d4f`` (#931 T3). The PMC coefficient holds the global
        high H ghost at zero; no exchange sends it. Until #1041
        this body exchanged BEFORE injecting, which lagged a seam-cell
        source by one step in the neighbour's copy -- see the comment on
        stage 7 for the measurement that moved it.
        """
        cpml_params_arg = cpml_params_arg._replace(**cpml_spacings)
        if magnetic_cpml_spacings:
            cpml_params_arg = cpml_params_arg._replace(
                magnetic=cpml_params_arg.magnetic._replace(**magnetic_cpml_spacings))
        _step_idx, src_vals = xs
        st = carry["fdtd"]
        record_previous = records.prepare(st, ranks=ranks)
        cpml_st = carry["cpml"]
        db_st = carry["debye"]
        lr_st = carry["lorentz"]

        # 1. H update
        st = _update_h_shmap(st, materials_arg, ranks=ranks)
        st, incident = slab_tfsf.apply_h(
            st, carry["tfsf"], tfsf_tables_arg[0], tfsf_cfg, mesh, ranks=ranks)

        # 2. CPML H correction (material-aware, #205)
        st, cpml_st = _apply_cpml_h_shmap(
            st, cpml_params_arg, cpml_st, n_cpml, dt, dx,
            mesh, n_devices, ghost=ghost, mu_r=materials_arg.mu_r,
            pad_x=pad_x, ranks=ranks, grid=cpml_grid, high_wall=high_wall, slab=(nx_per, nx))

        # 3. Exchange the live H ghosts (entry validates interval == 1)
        st = _exchange(_exchange_h_ghosts_shmap, ("hx", "hy", "hz"), st, _step_idx, ranks=ranks)


        # 4. E update
        st, db_st, lr_st, absorber = _update_e_shmap(
            st, materials_arg,
            debye_coeffs_arg, db_st,
            lorentz_coeffs_arg, lr_st, e_materials, ranks=ranks)
        st, incident = slab_tfsf.apply_e(
            st, incident, tfsf_tables_arg[1], tfsf_cfg, mesh, ranks=ranks)

        # 5. CPML E correction (material-aware, #205), with the permittivity
        #    the E update just used (#1043): each component's edge mean.
        st, cpml_st = _apply_cpml_e_shmap(
            st, cpml_params_arg, cpml_st, n_cpml, dt, dx,
            mesh, n_devices, ghost=ghost, materials=materials_arg,
            pad_x=pad_x,
            e_slab=(nx_per, nx), e_materials=e_materials,
            ranks=ranks, grid=cpml_grid, high_wall=high_wall,
            e_curl_coeff=absorber, boundary=curl_boundary)

        reference_out = ports.references(st, port_table_arg, ranks=ranks)
        st = _inject_sources_shmap(st, src_vals, ranks=ranks)

        # 6b. Realized conductor edges, distinct from domain walls. The
        #     global object already applied the shared PEC rule. Each rank
        #     zeros only owned rows; the kernel excludes ghosts so a seam
        #     edge is zeroed once and exchanged from its owner.
        #
        #     POSITION IS LOAD-BEARING: after injection, BEFORE the domain
        #     faces and E ghost exchange. The exchange hands the neighbour a
        #     FINISHED row. Exchanging first leaves rank 0's right ghost
        #     carrying the un-zeroed Ey/Ez of the seam plane rank 1 zeroes a
        #     stage later, and rank 0's next H update at its last real cell
        #     reads that stale plane -- measured on the nu lane at 2.107e-01
        #     final-step error against a 5e-5 gate (ac782d4f, #931 T3), versus
        #     7.773e-08 in this order.
        if pec_mask_arg is not None:
            st = apply_pec_mask_shmap(
                st, pec_mask_arg, mesh, n_devices, nx_local, ranks=ranks)

        # Every electric face: the declared walls and the conducting plane behind each absorbing face.
        st = _apply_pec_shmap(
            st, mesh, n_devices, nx_local, pad_x=pad_x,
            pec_faces=e_wall_faces, ranks=ranks)

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
        st = _exchange(_exchange_e_ghosts_shmap, ("ex", "ey", "ez"), st, _step_idx, ranks=ranks)

        # 8. Probe sampling
        probe_out = _sample_probes_shmap(st, ranks=ranks)

        updated = {"fdtd": st, "cpml": cpml_st,
                   "debye": db_st, "lorentz": lr_st}
        updated["tfsf"] = incident
        updated["records"] = records.update(carry["records"], st, dt, _step_idx,
                                             ranks=ranks, previous=record_previous)
        updated['ports'] = ports.update(carry['ports'], st, reference_out, _step_idx,
                                         port_table_arg, ranks=ranks)
        return updated, probe_out

    def step_fn_pec(
        carry, xs, materials_arg, debye_coeffs_arg, lorentz_coeffs_arg,
        pec_mask_arg, e_materials, tfsf_tables_arg, port_table_arg,
        *, ranks,
    ):
        """Single FDTD step (PEC path) operating on sharded arrays.

        Stage order (#1041)::

            H -> TF/SF-H -> exch H -> E -> TF/SF-E -> sources -> PEC face
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
        record_previous = records.prepare(st, ranks=ranks)
        db_st = carry["debye"]
        lr_st = carry["lorentz"]

        # 1. H update
        st = _update_h_shmap(st, materials_arg, ranks=ranks)
        st, incident = slab_tfsf.apply_h(
            st, carry["tfsf"], tfsf_tables_arg[0], tfsf_cfg, mesh, ranks=ranks)

        # 2. Exchange H ghost cells
        st = _exchange(_exchange_h_ghosts_shmap, ("hx", "hy", "hz"), st, _step_idx, ranks=ranks)


        # 3. E update
        st, db_st, lr_st, absorber = _update_e_shmap(
            st, materials_arg,
            debye_coeffs_arg, db_st,
            lorentz_coeffs_arg, lr_st, e_materials, ranks=ranks)
        st, incident = slab_tfsf.apply_e(
            st, incident, tfsf_tables_arg[1], tfsf_cfg, mesh, ranks=ranks)

        reference_out = ports.references(st, port_table_arg, ranks=ranks)
        st = _inject_sources_shmap(st, src_vals, ranks=ranks)

        # 5b. Realized-PEC cell mask (#1053), BEFORE the domain faces and
        #     E ghost exchange -- distributed_nu's stage 8, in its position.
        #     See the long note on step_fn_cpml stage 6b: getting this one
        #     stage later, after the exchange, cost 2.107e-01 against a 5e-5
        #     gate when the nu lane had it there. Moving domain faces across
        #     the exchange was measured inert (#1041 fixture P,
        #     bit-identical); that measurement does not transfer to a body
        #     mask, which must stay before the exchange.
        if pec_mask_arg is not None:
            st = apply_pec_mask_shmap(
                st, pec_mask_arg, mesh, n_devices, nx_local, ranks=ranks)

        # 5. PEC boundaries (domain faces only; conductor edges precede).
        #    Injection runs BEFORE the face, matching distributed_nu stage 7.
        #    The two lanes therefore both differ from the single-device lane,
        #    which applies the faces before its soft-source loop; the
        #    difference is observable only for a source placed ON a domain
        #    PEC face, where the tangential E is zeroed in the same step it
        #    is injected. #1041 measured no such fixture and did not change
        #    it -- a source on a PEC face is its own question.
        st = _apply_pec_shmap(
            st, mesh, n_devices, nx_local, pad_x=pad_x, pec_faces=e_wall_faces, ranks=ranks)

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
        st = _exchange(_exchange_e_ghosts_shmap, ("ex", "ey", "ez"), st, _step_idx, ranks=ranks)

        # 7. Probe sampling
        probe_out = _sample_probes_shmap(st, ranks=ranks)

        updated = {"fdtd": st, "debye": db_st, "lorentz": lr_st}
        updated["tfsf"] = incident
        updated["records"] = records.update(carry["records"], st, dt, _step_idx,
                                             ranks=ranks, previous=record_previous)
        updated['ports'] = ports.update(carry['ports'], st, reference_out, _step_idx,
                                         port_table_arg, ranks=ranks)
        return updated, probe_out

    # ------------------------------------------------------------------
    # Build step indices and waveform scan inputs
    # ------------------------------------------------------------------
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    # src_waveforms shape: (n_steps, n_sources) — replicated, not sharded
    xs = (step_indices, jax.device_put(src_waveforms, rep))
    from rfx.model.source_coefficients import slab_drive_table

    # ------------------------------------------------------------------
    # Run with jit + lax.scan
    # ------------------------------------------------------------------
    # Pass per-cell invariants as jit arguments on every topology. Closed-over
    # arrays become whole-domain compiled constants retained on every device,
    # including when a single process drives several devices.
    if use_cpml:
        carry_init = {
            "fdtd": sharded_state,
            "cpml": cpml_state_sharded,
            "debye": debye_state_sharded,
            "lorentz": lorentz_state_sharded,
        }
        carry_init["tfsf"] = tfsf_line
        carry_init["records"] = records.initial()
        carry_init['ports'] = ports.initial
        def _run_cpml(
            carry, scan_xs, materials_arg, debye_coeffs_arg, lorentz_coeffs_arg,
            cpml_params_arg, pec_mask_arg, tfsf_tables_arg, port_table_arg,
            *, ranks,
        ):
            e_materials = (slab_e_materials_shmap(materials_arg, mesh, nx_per, nx, ranks=ranks, high_wall=high_wall)
                           if has_debye or has_lorentz else None)
            final_carry, local_probe_ts = lax.scan(
                lambda scan_carry, scan_inputs: step_fn_cpml(
                    scan_carry, scan_inputs, materials_arg,
                    debye_coeffs_arg, lorentz_coeffs_arg,
                    cpml_params_arg, pec_mask_arg, e_materials, tfsf_tables_arg, port_table_arg,
                    ranks=ranks,
                ),
                carry,
                slab_drive_table(scan_xs, mesh, src_device_ids, src_local_specs, ranks=ranks),
            )

            # Each probe has exactly one owner; all other contributions are
            # +0. Sum once outside the time loop and replicate on the mesh,
            # including when the mesh spans multiple processes.
            probe_ts = lax.with_sharding_constraint(
                jnp.sum(local_probe_ts, axis=1), rep)
            final_carry['ports'] = ports.reduce(final_carry['ports'], ranks=ranks)
            return final_carry, probe_ts

        run_fn = jax.jit(_run_cpml)
        final_carry, probe_ts = run_fn(
            carry_init, xs, sharded_materials,
            debye_coeffs_sharded, lorentz_coeffs_sharded,
            cpml_params, sharded_pec_mask, tfsf_tables, ports.payload,
            ranks=mesh_ranks(mesh))
        final_state_sharded = final_carry["fdtd"]
    else:
        carry_init = {
            "fdtd": sharded_state,
            "debye": debye_state_sharded,
            "lorentz": lorentz_state_sharded,
        }
        carry_init["tfsf"] = tfsf_line
        carry_init["records"] = records.initial()
        carry_init['ports'] = ports.initial
        def _run_pec(
            carry, scan_xs, materials_arg, debye_coeffs_arg, lorentz_coeffs_arg,
            pec_mask_arg, tfsf_tables_arg, port_table_arg,
            *, ranks,
        ):
            e_materials = (slab_e_materials_shmap(materials_arg, mesh, nx_per, nx, ranks=ranks, high_wall=high_wall)
                           if has_debye or has_lorentz else None)
            final_carry, local_probe_ts = lax.scan(
                lambda scan_carry, scan_inputs: step_fn_pec(
                    scan_carry, scan_inputs, materials_arg,
                    debye_coeffs_arg, lorentz_coeffs_arg,
                    pec_mask_arg, e_materials, tfsf_tables_arg, port_table_arg,
                    ranks=ranks,
                ),
                carry,
                slab_drive_table(scan_xs, mesh, src_device_ids, src_local_specs, ranks=ranks),
            )

            # Each probe has exactly one owner; all other contributions are
            # +0. Sum once outside the time loop and replicate on the mesh,
            # including when the mesh spans multiple processes.
            probe_ts = lax.with_sharding_constraint(
                jnp.sum(local_probe_ts, axis=1), rep)
            final_carry['ports'] = ports.reduce(final_carry['ports'], ranks=ranks)
            return final_carry, probe_ts

        run_fn = jax.jit(_run_pec)
        final_carry, probe_ts = run_fn(
            carry_init, xs, sharded_materials,
            debye_coeffs_sharded, lorentz_coeffs_sharded,
            sharded_pec_mask, tfsf_tables, ports.payload,
            ranks=mesh_ranks(mesh))
        final_state_sharded = final_carry["fdtd"]

    return final_carry, probe_ts, final_state_sharded, records
