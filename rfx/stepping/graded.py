"""Graded Yee kernels and observations attached to the physical sequence.

The step is one pass over STEP_SEQUENCE. A kernel is the graded lane's own
arithmetic for one physical stage and may be absent. A hook point is a fixed
time in the step at which measurements or an adjoint attach, and where no
lane arithmetic runs. The frame carries the arrays of this step only.
Conductors, including the model's published occupancy weight, act on the
updated E before sheets, lumped elements and sources. A source's occupancy
weight is already in its stored drive table, so the loop only adds it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from rfx.boundaries.pec import apply_edge_keep, apply_pec_edges, apply_pec_faces
from rfx.core.drives import inject_drives
from rfx.core.yee import CurlBoundary
from rfx.stepping.yee import update_e_nu, update_h_nu
from .sequence import HookPoint, Kernel, compose

if TYPE_CHECKING:
    from rfx.boundaries.cpml import CPMLAxisParams
    from rfx.current_moments import CurrentMomentMonitor
    from rfx.farfield import NTFFBox
    from rfx.model.graded_grid import NonUniformGrid
    from rfx.sources.tfsf import TFSFConfig


@dataclass(frozen=True)
class GradedStepContext:
    """Static step inputs; lattice-sized operands remain in invariants."""

    use_lumped_rlc: bool
    rlc_metas: tuple
    use_current_moments: bool
    _slab_e_snapshot: Callable | None
    current_moments: CurrentMomentMonitor | None
    dt: float | jax.Array
    inv_dy_h: jax.Array
    inv_dz_h: jax.Array
    grid: NonUniformGrid
    use_tfsf: bool
    tfsf_cfg: TFSFConfig | None
    use_waveguide_ports: bool
    waveguide_meta: tuple
    waveguide_port_metrics: tuple
    use_cpml: bool
    apply_cpml_h: Callable
    cpml_params: CPMLAxisParams | None
    cpml_grid: NonUniformGrid | None
    cpml_axes_eff: str | None
    use_pmc_faces: bool
    _pmc_faces_frozen: frozenset[str]
    use_sheet_impedance: bool
    use_design_box: bool
    use_debye: bool
    use_lorentz: bool
    inv_dy: jax.Array
    inv_dz: jax.Array
    curl_boundary: CurlBoundary
    apply_cpml_e: Callable
    _pec_faces_frozen: frozenset[str]
    use_pec_edges: bool
    use_pec_occupancy: bool
    use_wire_ports: bool
    wp_meta: list
    sp_freqs: jax.Array | None
    use_dft_planes: bool
    dft_meta: tuple
    use_flux_monitors: bool
    flux_meta: tuple
    use_ntff: bool
    ntff_box: NTFFBox | None
    _accumulate_cm: Callable | None
    emit_time_series: bool
    x_face_frame: Mapping[str, tuple[int, int]] | None = None  # Axis -> static (lo, hi) physical rows.
    port_record_specs: tuple | None = None
    port_acc_indices: tuple = ()
    cpml_spacings: tuple | None = None


@dataclass
class GradedFrame:
    """One step's transient fields, input carry, and observer outputs."""

    carry: dict[str, Any]
    xs: tuple
    invariants: dict[str, Any]
    materials: Any = None
    pec_edge_masks: Any = None
    _cpml_inv_eps_r: Any = None
    debye_coeffs: Any = None
    lorentz_coeffs: Any = None
    aniso_eps: Any = None
    sheet_impedance: Any = None
    sheet_coeffs: Any = None
    design_box_coeffs: Any = None
    step_idx: Any = None
    src_vals: Any = None
    st: Any = None
    rlc_e_prev: Any = None
    e_prev_slab: Any = None
    tfsf_h_state: Any = None
    cpml_new: Any = None
    e_prev_sheet: Any = None
    st_prev_design: Any = None
    debye_new: Any = None
    lorentz_new: Any = None
    new_rlc_states: Any = None
    new_waveguide_port_accs: Any = None
    new_wire_sp: Any = None
    port_samples: Any = None
    port_refs: Any = None
    new_port_refs: Any = None
    new_dft_planes: Any = None
    new_flux_monitors: Any = None
    plane_samples: Any = None
    new_ntff: Any = None
    new_cm: Any = None
    tfsf_new: Any = None
    probe_out: Any = None


def make_graded_step(
    ctx: GradedStepContext,
    *,
    hooks: Mapping[HookPoint, tuple[Callable[[GradedFrame], None], ...]] | None = None,
) -> Callable:
    """Bind graded kernels and measurements, then user hooks at each point."""
    from rfx.stepping.graded_setup import _update_e_nu_dispersive
    from rfx.geometry.rasterize_grid import unpadded_high_walls
    high_wall = getattr(ctx.grid, "high_wall", unpadded_high_walls(ctx.grid))
    dy_cells, dz_cells = ctx.grid.dy_arr, ctx.grid.dz
    from rfx.measurement.projector import port_table, project, port_owners
    port_specs = ctx.port_record_specs if ctx.port_record_specs is not None else tuple(
        (m[3], m[13], 'wire', m[:3], i) for i, m in enumerate(ctx.wp_meta))
    acc_indices = ctx.port_acc_indices if ctx.port_record_specs is not None else tuple(range(len(ctx.wp_meta)))
    widths = tuple(3 if p[2] == 'wire' else 2 for p in port_specs)
    from rfx.measurement.port_sampler import (
        pre_injection_samples, ref_dft_update, wire_dft_update, stack_samples)
    port_tables = {}

    def graded_port_table(frame: GradedFrame):
        table = frame.invariants.get('port_table')
        dtype = frame.st.ex.dtype
        if table is None and dtype not in port_tables:
            # Graded widths are JAX arrays: evaluate the weights once, outside
            # the traced step, as they were before the table (a coefficient
            # formed inside the loop is rounded by its context).
            with jax.ensure_compile_time_eval():
                port_tables[dtype] = port_table(ctx.grid, port_specs, nu=True,
                                                boundary=ctx.curl_boundary, dtype=dtype)
        return port_tables[dtype] if table is None else table

    def before_sources(frame: GradedFrame):
        # The reference reads the V row's own weight, so a traced width
        # (a dz_profile design variable) stays a tracer.
        frame.port_refs = pre_injection_samples(frame.st, port_specs, None, table=graded_port_table(frame))
        frame.new_port_refs = [ref_dft_update(
            accs[4], frame.port_refs[acc_indices[p]], frame.step_idx, ctx.sp_freqs, ctx.dt)
            for p, accs in enumerate(frame.carry.get('wire_sparams', ()))]

    def prepare(frame: GradedFrame):
        frame.materials = frame.invariants["materials"]
        frame.pec_edge_masks = frame.invariants["pec_edge_masks"]
        frame._cpml_inv_eps_r = frame.invariants["cpml_inv_eps_r"]
        frame.debye_coeffs = frame.invariants["debye_coeffs"]
        frame.lorentz_coeffs = frame.invariants["lorentz_coeffs"]
        frame.aniso_eps = frame.invariants["aniso_eps"]
        frame.sheet_impedance = frame.invariants["sheet_impedance"]
        frame.sheet_coeffs = frame.invariants["sheet_coeffs"]
        frame.design_box_coeffs = frame.invariants["design_box_coeffs"]
        frame.step_idx, frame.src_vals = frame.xs
        frame.st = frame.carry["fdtd"]
        # #1163: E^n at each lumped RLC edge, read before the E update (the
        # series element is solved together with its edge field).
        frame.rlc_e_prev = (
            tuple(getattr(frame.st, m.component)[m.i, m.j, m.k] for m in ctx.rlc_metas)
            if ctx.use_lumped_rlc else ())

        # E^n on the slab, before anything in this step writes E (the H
        # update below does not touch it). Slab-sized so the reverse-mode
        # tape carries the slab and not the domain.
        if ctx.use_current_moments:
            frame.e_prev_slab = ctx._slab_e_snapshot(frame.st, ctx.current_moments)

    def h_update(frame: GradedFrame):
        # H update (non-uniform)
        frame.st = update_h_nu(frame.st, frame.materials, ctx.dt, frame.invariants["inv_dx_h"], ctx.inv_dy_h, ctx.inv_dz_h,
                         cell_sizes=(frame.invariants["dx_arr"], dy_cells, dz_cells),
                         high_wall=high_wall, periodic=ctx.curl_boundary.periodic)

    def h_boundary(frame: GradedFrame):
        frame.tfsf_h_state = None
        if ctx.use_tfsf:
            from rfx.sources.tfsf import apply_tfsf_h
            frame.st = apply_tfsf_h(frame.st, ctx.tfsf_cfg, frame.carry["tfsf"], ctx.grid.dx, ctx.dt)
        if ctx.use_waveguide_ports:
            from rfx.stepping.waveguide_source import apply_waveguide_port_h as _apply_wg_h_nu
            for cfg_meta, _m in zip(ctx.waveguide_meta,
                                    ctx.waveguide_port_metrics):
                frame.st = _apply_wg_h_nu(frame.st, cfg_meta, frame.step_idx, ctx.dt, _m[0])
        if ctx.use_cpml and 'cpml_x' in frame.invariants:
            from rfx.stepping.slab_cpml import apply
            frame.st, frame.cpml_new = apply(ctx.apply_cpml_h, ctx, frame.invariants,
                frame.st, frame.carry['cpml'], materials=frame.materials)
        elif ctx.use_cpml:
            frame.st, frame.cpml_new = ctx.apply_cpml_h(frame.st, ctx.cpml_params, frame.carry["cpml"],
                                         ctx.cpml_grid, ctx.cpml_axes_eff,
                                         materials=frame.materials)
        else:
            frame.cpml_new = None
        if ctx.use_tfsf:
            from rfx.sources.tfsf import update_tfsf_1d_h
            frame.tfsf_h_state = update_tfsf_1d_h(ctx.tfsf_cfg, frame.carry["tfsf"], ctx.grid.dx, ctx.dt)

    def e_update(frame: GradedFrame):
        # Snapshot E^n for the #677 sheet operator (it REPLACES the
        # standard update at masked tangential edges with A*E^n + B*curlH).
        frame.e_prev_sheet = (frame.st.ex, frame.st.ey, frame.st.ez) if ctx.use_sheet_impedance else None
        # #1183: the design box REDOES the E update at its own cells from
        # the pre-update state (E^n and the same H^{n+1/2}).
        frame.st_prev_design = frame.st if ctx.use_design_box else None

        # E update: use ADE-aware path when dispersive materials are present
        frame.debye_new = None
        frame.lorentz_new = None
        if ctx.use_debye or ctx.use_lorentz:
            frame.st, frame.debye_new, frame.lorentz_new = _update_e_nu_dispersive(
                frame.st, frame.materials, ctx.dt, frame.invariants["inv_dx"], ctx.inv_dy, ctx.inv_dz,
                debye=(frame.debye_coeffs, frame.carry["debye"]) if ctx.use_debye else None,
                lorentz=(frame.lorentz_coeffs, frame.carry["lorentz"]) if ctx.use_lorentz else None,
                boundary=ctx.curl_boundary,
            )
        elif frame.aniso_eps is not None:
            from rfx.stepping.yee import update_e_nu_aniso
            _eex, _eey, _eez = frame.aniso_eps
            frame.st = update_e_nu_aniso(
                frame.st, frame.materials, _eex, _eey, _eez, ctx.dt,
                frame.invariants["inv_dx"], ctx.inv_dy, ctx.inv_dz, boundary=ctx.curl_boundary,
                high_wall=high_wall,
            )
        else:
            frame.st = update_e_nu(frame.st, frame.materials, ctx.dt, frame.invariants["inv_dx"], ctx.inv_dy, ctx.inv_dz,
                             boundary=ctx.curl_boundary, high_wall=high_wall)

        # #1183 design box: redo the E update at the design cells from the
        # traced permittivity, on the graded-mesh curl. Same slot as the
        # uniform lane — immediately after the E update, on the same H,
        # before anything that reads or writes E.
        if ctx.use_design_box:
            from rfx.stepping.yee import update_e_box
            frame.st = update_e_box(
                frame.st, frame.st_prev_design, frame.design_box_coeffs.bounds,
                frame.design_box_coeffs.ca, frame.design_box_coeffs.cb, ctx.grid.dx,
                inv_d=(frame.invariants["inv_dx"], ctx.inv_dy, ctx.inv_dz), boundary=ctx.curl_boundary,
            )

    def e_boundary(frame: GradedFrame):
        if ctx.use_tfsf:
            from rfx.sources.tfsf import apply_tfsf_e
            frame.st = apply_tfsf_e(frame.st, ctx.tfsf_cfg, frame.tfsf_h_state, ctx.grid.dx, ctx.dt)
        if ctx.use_waveguide_ports:
            from rfx.stepping.waveguide_source import apply_waveguide_port_e as _apply_wg_e_nu
            for cfg_meta, _m in zip(ctx.waveguide_meta,
                                    ctx.waveguide_port_metrics):
                frame.st = _apply_wg_e_nu(frame.st, cfg_meta, frame.step_idx, ctx.dt, _m[1])
        if ctx.use_cpml and 'cpml_x' in frame.invariants:
            from rfx.stepping.slab_cpml import apply
            from rfx.boundaries.cpml import apply_cpml_e
            from rfx.boundaries.electric_coefficient import loss_terms, dispersive_curl, in_loop_loss
            means = (frame.materials.components.eps_update,
                     frame.materials.components.sigma_update)
            if frame._cpml_inv_eps_r is not None:
                loss = in_loop_loss(frame.invariants['e_loss'], ctx.dt)
                inverse = frame._cpml_inv_eps_r
            else:
                loss = loss_terms(means, ctx.dt) if means[0] else None
                inverse = tuple(1 / e for e in means[0]) or None
            frame.st, frame.cpml_new = apply(apply_cpml_e, ctx, frame.invariants,
                frame.st, frame.cpml_new, materials=frame.materials,
                e_loss=loss, inv_eps_r_update=inverse,
                boundary=ctx.curl_boundary,
                e_curl_coeff=dispersive_curl(frame.debye_coeffs, frame.lorentz_coeffs, ctx.dt))
        elif ctx.use_cpml:
            frame.st, frame.cpml_new = ctx.apply_cpml_e(frame.st, ctx.cpml_params, frame.cpml_new,
                                         ctx.cpml_grid, ctx.cpml_axes_eff,
                                         materials=frame.materials, e_loss=frame.invariants["e_loss"],
                                         inv_eps_r_update=frame._cpml_inv_eps_r,
                                         boundary=ctx.curl_boundary, debye=frame.debye_coeffs, lorentz=frame.lorentz_coeffs)

    def pec(frame: GradedFrame):
        """Enforce realized conductor edges and occupancy before sheets."""
        if ctx.use_pec_edges:
            frame.st = apply_pec_edges(frame.st, frame.pec_edge_masks)
        if ctx.use_pec_occupancy:
            frame.st = apply_edge_keep(frame.st, frame.materials.edge_keep)

    def constitutive(frame: GradedFrame):
        """Apply sheets, then domain faces, then lumped constitutive laws."""
        # #677 node-thin surface-impedance sheet operator. Contract slot:
        # AFTER apply_pec_mask/apply_pec_occupancy, BEFORE domain faces
        # (PEC wins on overlap), lumped updates, sources and port sampling.
        # curlH comes from
        # the SAME shared stencil helper update_e_nu uses, on the same
        # H^{n+1/2} the E update consumed.
        if ctx.use_sheet_impedance:
            from rfx.core.yee import curl_h_nu as _curl_h_nu
            from rfx.stepping.conductors import (
                apply_sheet_impedance_e as _apply_sheet_e)
            _scd = jnp.promote_types(frame.st.ex.dtype, jnp.float32)
            _curls = _curl_h_nu(
                frame.st.hx.astype(_scd), frame.st.hy.astype(_scd), frame.st.hz.astype(_scd),
                frame.invariants["inv_dx"], ctx.inv_dy, ctx.inv_dz, boundary=ctx.curl_boundary)
            frame.st = _apply_sheet_e(frame.st, frame.e_prev_sheet, _curls,
                                frame.sheet_impedance, frame.sheet_coeffs)

        # PEC, per face (#1164)
        frame.st = apply_pec_faces(frame.st, ctx._pec_faces_frozen,
                                   frame=ctx.x_face_frame, owned=frame.invariants.get("owns_x"))

        # Lumped RLC ADE update (after E update + boundaries, before sources)
        frame.new_rlc_states = None
        if ctx.use_lumped_rlc:
            from rfx.stepping.lumped import update_rlc_elements
            frame.st, frame.new_rlc_states = update_rlc_elements(
                frame.st, frame.carry["rlc_states"], ctx.rlc_metas, frame.rlc_e_prev)

    def sources(frame: GradedFrame):
        # Sources (point sources + wire port excitation)
        frame.st = inject_drives(frame.st, frame.invariants["drives"], frame.src_vals)

    def after_sources(frame: GradedFrame):
        # Waveguide-port injection + DFT probe accumulation. The dx
        # arg is unused by the per-cell-weighted integrals (cfg already
        # stores u_widths/v_widths), but kept in the function signature
        # for back-compat. It is handed the port's own primal cell rather
        # than the grid's boundary scalar so that nothing here reads a
        # scalar the port does not sit on; because the callee ignores it,
        # no recorded number moves either way.
        frame.new_waveguide_port_accs = None
        if ctx.use_waveguide_ports:
            from rfx.measurement.waveguide_modal import (
                update_waveguide_port_probe,
            )
            frame.new_waveguide_port_accs = []
            for accs, cfg_meta, _m in zip(
                frame.carry["waveguide_port_accs"], ctx.waveguide_meta,
                ctx.waveguide_port_metrics,
            ):
                cfg = cfg_meta._replace(
                    v_probe_t=accs[0],
                    v_ref_t=accs[1],
                    i_probe_t=accs[2],
                    i_ref_t=accs[3],
                    v_inc_t=accs[4],
                    n_steps_recorded=accs[5],
                )
                # TFSF-style corrections applied earlier at canonical slots.
                cfg_updated = update_waveguide_port_probe(cfg, frame.st, ctx.dt, _m[0])
                frame.new_waveguide_port_accs.append((
                    cfg_updated.v_probe_t,
                    cfg_updated.v_ref_t,
                    cfg_updated.i_probe_t,
                    cfg_updated.i_ref_t,
                    cfg_updated.v_inc_t,
                    cfg_updated.n_steps_recorded,
                ))

        # Wire port V/I DFT accumulation.
        #
        # Ordering: AFTER source injection.  Issue #683 is DECIDED (by
        # measurement, 2026-08-29 — docs/design_notes/
        # issue683_sampling_order_decision_protocol.md section 9): this
        # slot is the terminal-consistent, physically correct one.  The
        # known-load circuit law holds only for the post-injection pair
        # (n*a = +0.9987/+0.9950, n*|b| = 0.08/0.32 Ohm over a six-point
        # R_L sweep at both decision bins), and the post-injection E is
        # the true field level E^{n+1} of the discrete update
        # (Ampere-identity residual 2.3e-7 vs 3.25 pre-injection).  The
        # once-CONTESTED known-load sweep was settled by that protocol run
        # (gates G0-G2 all passed; the earlier independent repro's fixture
        # had failed to load the port), and the once-UNEXAMINED Ampere
        # identity was examined in the same run (section 7).
        #
        # The uniform lane (rfx/simulation.py wire block) flipped to this
        # same POST slot for its physical V/I/V_port channels, with the
        # pre-injection drive sample kept as a separate calibration
        # reference channel for its #308/#313 off-diagonal decomposition
        # (issue #683 x #764,
        # docs/design_notes/issue683_decomposer_flip_predeclaration.md).
        # This lane's decomposition was calibrated in the POST frame
        # directly, so it needs NO reference channel and NO change.
        #
        # PASSIVE ports are unaffected either way: I reads H only, and the
        # source loop writes E only, at the SOURCE cell.
        frame.new_wire_sp = None
        if port_specs:
            frame.new_wire_sp = []
            frame.port_samples = []
            port_values = project(frame.st, graded_port_table(frame))
            frame.port_samples = stack_samples(port_values, port_specs, frame.port_refs)
            for p, accs in enumerate(frame.carry.get("wire_sparams", ())):
                sample = frame.port_samples[acc_indices[p]]
                v, i_val = sample[:2]
                v_port = sample[2] if widths[acc_indices[p]] == 3 else v
                v_ref, phase = frame.new_port_refs[p]
                frame.new_wire_sp.append((*wire_dft_update(
                    accs, (v, i_val, v_port), frame.step_idx, ctx.sp_freqs, ctx.dt, phase), v_ref))

        from rfx.measurement.accumulators import planes, flux
        frame.new_dft_planes = frame.new_flux_monitors = None
        if ctx.use_dft_planes:
            frame.new_dft_planes, frame.plane_samples = planes(
                frame.st, frame.carry["dft_planes"], ctx.dft_meta, ctx.dt, frame.step_idx)
        if ctx.use_flux_monitors:
            frame.new_flux_monitors = flux(frame.st, frame.carry["flux_monitors"], ctx.flux_meta, ctx.dt, frame.step_idx)

        # NTFF: accumulate tangential E/H DFT on 6 box faces
        frame.new_ntff = None
        if ctx.use_ntff:
            from rfx.farfield import accumulate_ntff
            frame.new_ntff = accumulate_ntff(frame.carry["ntff"], frame.st, ctx.ntff_box, ctx.dt, frame.step_idx)

        # Block current moments — same slot as the NTFF box: E at (n+1)*dt,
        # H at (n+1/2)*dt, with the feed current already injected.
        frame.new_cm = None
        if ctx.use_current_moments:
            frame.new_cm = ctx._accumulate_cm(
                frame.carry["current_moments"], frame.st, frame.e_prev_slab, ctx.current_moments,
                ctx.dt, frame.step_idx)

        # TFSF 1D auxiliary E-field update (mirrors uniform scan body:
        # called AFTER sources, closes the leapfrog step).
        frame.tfsf_new = None
        if ctx.use_tfsf:
            from rfx.sources.tfsf import update_tfsf_1d_e
            t_tfsf = frame.step_idx.astype(jnp.float32) * ctx.dt
            frame.tfsf_new = update_tfsf_1d_e(ctx.tfsf_cfg, frame.tfsf_h_state, ctx.grid.dx, ctx.dt, t_tfsf)

        # Probes
        if ctx.emit_time_series and frame.invariants["probe_table"].n_channels:
            table = frame.invariants["probe_table"]
            nodes = _host_probe_nodes(table)
            if nodes is None:
                frame.probe_out = project(frame.st, table)
            else:
                samples = [getattr(frame.st, pc)[pi, pj, pk] for pc, pi, pj, pk in nodes]
                frame.probe_out = jnp.stack(samples)
        else:
            frame.probe_out = jnp.zeros(0)

    def finish(frame: GradedFrame):
        new_carry = {"fdtd": frame.st}
        if "dft_time_records" in frame.carry:
            new_carry["dft_time_records"] = tuple(
                jax.lax.stop_gradient(record.at[frame.step_idx].set(sample))
                for record, sample in zip(frame.carry["dft_time_records"], frame.plane_samples))
        if "sparam_time_records" in frame.carry:
            assert len(frame.carry["sparam_time_records"]) == len(frame.port_samples)
            new_carry["sparam_time_records"] = tuple(
                jax.lax.stop_gradient(record.at[frame.step_idx].set(sample))
                for record, sample in zip(frame.carry["sparam_time_records"], frame.port_samples))
        if ctx.use_cpml:
            new_carry["cpml"] = frame.cpml_new
        if ctx.use_debye and frame.debye_new is not None:
            new_carry["debye"] = frame.debye_new
        if ctx.use_lorentz and frame.lorentz_new is not None:
            new_carry["lorentz"] = frame.lorentz_new
        if ctx.use_wire_ports and frame.new_wire_sp is not None:
            new_carry["wire_sparams"] = tuple(frame.new_wire_sp)
        if ctx.use_dft_planes and frame.new_dft_planes is not None:
            new_carry["dft_planes"] = tuple(frame.new_dft_planes)
        if ctx.use_flux_monitors and frame.new_flux_monitors is not None:
            new_carry["flux_monitors"] = tuple(frame.new_flux_monitors)
        if ctx.use_lumped_rlc and frame.new_rlc_states is not None:
            new_carry["rlc_states"] = tuple(frame.new_rlc_states)
        if ctx.use_ntff and frame.new_ntff is not None:
            new_carry["ntff"] = frame.new_ntff
        if ctx.use_current_moments and frame.new_cm is not None:
            new_carry["current_moments"] = frame.new_cm
        if ctx.use_waveguide_ports and frame.new_waveguide_port_accs is not None:
            new_carry["waveguide_port_accs"] = tuple(frame.new_waveguide_port_accs)
        if ctx.use_tfsf and frame.tfsf_new is not None:
            new_carry["tfsf"] = frame.tfsf_new
        return new_carry, frame.probe_out

    attachments = {point: () for point in HookPoint}
    attachments[HookPoint.BEFORE_SOURCES] = (before_sources,) if port_specs else ()
    attachments[HookPoint.AFTER_SOURCES] = (after_sources,)
    for point, extra in (hooks or {}).items():
        attachments[point] += tuple(extra)
    advance = compose({
        Kernel.H_UPDATE: h_update,
        Kernel.H_BOUNDARY: h_boundary,
        Kernel.E_UPDATE: e_update,
        Kernel.E_BOUNDARY: e_boundary,
        Kernel.PEC: pec,
        Kernel.CONSTITUTIVE: constitutive,
        Kernel.SOURCES: sources,
    }, attachments)

    def step_fn(carry, xs, invariants):
        frame = GradedFrame(carry, xs, invariants)
        prepare(frame)
        advance(frame)
        return finish(frame)

    step_fn.context = ctx
    step_fn.port_owners = port_owners(ctx.grid, port_specs, nu=True, boundary=ctx.curl_boundary)
    step_fn.port_acc_indices = acc_indices
    return step_fn


def _host_probe_nodes(table):
    """Keep the pre-table one-device program and gradient; leave slab tables to project."""
    if any(not isinstance(index, np.ndarray) for terms in table.components for index in terms.indices):
        return None
    nodes = [None] * table.n_channels
    for terms in table.components:
        if isinstance(terms.channel_ids, jax.core.Tracer) or isinstance(terms.weights, jax.core.Tracer):
            return None
        for channel, i, j, k, weight in zip(np.asarray(terms.channel_ids), *terms.indices,
                                          np.asarray(terms.weights), strict=True):
            channel = int(channel)
            if weight != 1.0 or not 0 <= channel < table.n_channels or nodes[channel] is not None:
                return None
            nodes[channel] = (terms.component, int(i), int(j), int(k))
    return tuple(nodes) if all(node is not None for node in nodes) else None


def graded_probe_table(prb_meta, grid_shape, *, dtype):
    """Compile the graded point probes in their original record order."""
    from rfx.measurement.plan import Node, Owner, field_channel
    from rfx.measurement.projector import sample_table

    channels = []
    for number, (i, j, k, component) in enumerate(prb_meta):
        indices = []
        for axis, index, size in zip('xyz', (i, j, k), grid_shape, strict=True):
            if not -size <= index < size:
                raise IndexError(f'Probe {number} ({component}, {(i, j, k)}): '
                                 f'{axis} index {index} outside [-{size}, {size})')
            indices.append(index + size if index < 0 else index)
        channels.append(field_channel(component, (Node(component, *indices, 1.0),)))
    return sample_table((Owner('probes', 'probe', tuple(channels)),), dtype=dtype)
