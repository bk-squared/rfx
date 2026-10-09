"""Uniform Yee kernels and observations attached to the physical sequence.

The frame is Python trace-time bookkeeping, never a scan carry or a pytree.
Only its arrays leave a step, with the existing runner carry signature.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

from rfx.core.yee import FDTDState
from rfx.boundaries.pec import (
    apply_pec_occupancy, apply_pec_occupancy_box,
)
from rfx.probes import probes as observers
from rfx.probes.probes import _port_voltage_value
from rfx.core.dft_utils import port_dft_phase
from .sequence import HookPoint, Kernel, compose

if TYPE_CHECKING:
    from rfx.simulation import _StepContext


def apply_pec_faces(state, faces):
    """Retain both the lane and simulation-module wall observation points."""
    from rfx import simulation
    return simulation.apply_pec_faces(state, faces)


@dataclass
class UniformFrame:
    """One step's transient fields, input carry, and observer outputs."""

    st: FDTDState
    carry: dict[str, Any]
    step_idx: jax.Array
    src_vals: jax.Array
    mag_src_vals: jax.Array
    cm_new: Any = None
    cpml_new: Any = None
    debye_new: Any = None
    design_record: Any = None
    e_prev_kerr: Any = None
    e_prev_sheet: Any = None
    e_prev_slab: Any = None
    extras: Any = None
    lorentz_new: Any = None
    lumped_ref_samples: Any = None
    new_dft_planes: Any = None
    new_flux_accs: Any = None
    new_lumped_accs: Any = None
    new_lumped_refs: Any = None
    new_refplane_accs: Any = None
    new_rlc_states: Any = None
    new_waveguide_port_accs: Any = None
    new_wire_accs: Any = None
    new_wire_refs: Any = None
    ntff_new: Any = None
    plane_samples: Any = None
    port_samples: Any = None
    probe_out: Any = None
    refplane_samples: Any = None
    rlc_e_prev: Any = None
    st_prev_design: Any = None
    t: Any = None
    tfsf_h_state: Any = None
    tfsf_new: Any = None
    wire_ref_samples: Any = None


def make_uniform_step(
    ctx: _StepContext,
    invariants: dict[str, Any],
    *,
    design_hook: Callable | None = None,
    hooks: Mapping[HookPoint, tuple[Callable[[UniformFrame], None], ...]] | None = None,
) -> Callable:
    """Bind uniform physics and observations to the five physical hook points.

    ``design_hook`` retains the adjoint's (previous, updated) -> (state, record)
    contract, immediately after the E/design-box update. Additional ``hooks``
    consume the same transient frame after the built-in attachments at each
    point. The returned callable retains the simulation runner's signature.
    """
    from rfx import simulation
    from rfx.simulation import update_h, update_e_box, _update_e_with_optional_dispersion
    drives = None if ctx.drives is None else ctx.drives.electric
    mag_drives = None if ctx.drives is None else ctx.drives.magnetic
    if mag_drives is not None and not mag_drives.nodes:
        mag_drives = None
    materials = ctx.materials
    dt = ctx.dt
    dx = ctx.dx
    periodic = ctx.periodic
    grid = ctx.grid
    aniso_eps = ctx.aniso_eps
    aniso_inv_eps = ctx.aniso_inv_eps
    cpml_inv_eps_r = invariants["cpml_inv_eps_r"]
    _sheet_coeffs = invariants["sheet_coeffs"]

    if ctx.use_sheet_impedance:
        from rfx.materials.thin_conductor import (
            apply_sheet_impedance_e as _apply_sheet_e,
        )

    if ctx.use_current_moments:
        from rfx.current_moments import slab_e_snapshot as _slab_e_snapshot

    def prepare(frame: UniformFrame):
        """Retain E at n for reactive elements and current-moment observations."""
        frame.st = frame.carry["fdtd"]
        frame.tfsf_h_state = None
        # #1163: a series RLC element is solved together with its edge field
        # and needs E^n there; nothing before the E update writes E, so the
        # carry's value IS E^n on every path (fast H+E kernel included).
        frame.rlc_e_prev = (
            tuple(getattr(frame.st, m.component)[m.i, m.j, m.k]
                  for m in ctx.rlc_meta)
            if ctx.use_lumped_rlc else ())

        # E^n on the slab, taken before anything in this step writes E. The
        # H update below does not touch E, so this is the same array the
        # E update is about to consume — and the difference against the
        # post-update field is exactly one timestep, which is what makes
        # ``J = curl_h H - eps0 dE/dt`` the lattice's own current.
        # Slab-sized: the reverse-mode tape carries the slab, not the domain.
        if ctx.use_current_moments:
            frame.e_prev_slab = _slab_e_snapshot(frame.st, ctx.current_moments)

    def he_update(frame: UniformFrame):
        """Advance H and E together with conductor faces baked into coefficients."""
        from rfx import simulation
        frame.st = simulation.update_he_fast(
            frame.st, ctx.fast_coeffs, boundary=ctx.curl_boundary)

    def h_update(frame: UniformFrame):
        """Advance H to n+1/2 using the lane magnetic curl."""
        if not ctx.use_fast_he:
            if ctx.use_upml:
                frame.st = ctx.apply_upml_h(frame.st, ctx.upml_coeffs, periodic=periodic)
            else:
                frame.st = update_h(frame.st, materials, dt, dx, periodic=periodic,
                              stencil_order=ctx.stencil_order, bloch=ctx.bloch)

    def h_boundary(frame: UniformFrame):
        """Correct incident H, absorb outgoing H, and enforce magnetic walls."""
        if not ctx.use_fast_he:
            if ctx.use_tfsf:
                frame.st = ctx.apply_tfsf_h(frame.st, ctx.tfsf_cfg, frame.carry["tfsf"], dx, dt)
            if ctx.use_waveguide_ports:
                from rfx.sources.waveguide_port import apply_waveguide_port_h as _apply_wg_h
                for cfg_meta in ctx.waveguide_meta:
                    frame.st = _apply_wg_h(frame.st, cfg_meta, frame.step_idx, dt, dx)
            if ctx.use_cpml:
                frame.st, frame.cpml_new = ctx.apply_cpml_h(
                    frame.st, ctx.cpml_params, frame.carry["cpml"], grid, ctx.cpml_axes,
                    materials=materials, periodic=periodic)
            # Stage 2 H damping — applied AFTER CPML-H so CPML cannot
            # un-zero H at Kottke-frozen PEC cells.  Threshold rather
            # than ``== 0.0`` so smooth-Kottke (eps_inside = 1e10)
            # cells (inv ≈ 1e-10 at f=1) are caught alongside exact-PEC
            # cells (inv = 0 from binary `pec_shapes`).  Smooth-Kottke
            # path uses ONLY the full-PEC mask (all three inv below
            # threshold); the pairwise per-component masks would trigger
            # spuriously at sigmoid-edge cells (where 2 of 3 components
            # are frozen due to Kottke anisotropy but the cell is
            # legitimately a partial-fill interface, not full PEC) —
            # killing wave propagation INTO the stub region.  Binary
            # Stage 2 path keeps the pairwise masks as before (boundary
            # cells of binary PEC need the corner-specific H zero).
            if ctx.use_aniso_inv:
                from rfx.boundaries.pec import apply_pec_h_mask
                _inv_xx, _inv_yy, _inv_zz = aniso_inv_eps
                _PEC_INV_THRESHOLD = 1e-9
                _xx0 = (_inv_xx < _PEC_INV_THRESHOLD)
                _yy0 = (_inv_yy < _PEC_INV_THRESHOLD)
                _zz0 = (_inv_zz < _PEC_INV_THRESHOLD)
                if ctx.aniso_inv_eps_smooth:
                    frame.st = apply_pec_h_mask(
                        frame.st,
                        pec_mask=_xx0 & _yy0 & _zz0,
                    )
                else:
                    frame.st = apply_pec_h_mask(
                        frame.st,
                        pec_mask=_xx0 & _yy0 & _zz0,
                        mask_hx=_yy0 & _zz0,
                        mask_hy=_xx0 & _zz0,
                        mask_hz=_xx0 & _yy0,
                    )
            if ctx.use_pmc_faces:
                from rfx.boundaries.pmc import apply_pmc_faces
                frame.st = apply_pmc_faces(frame.st, ctx.pmc_faces_frozen, image=True)
            if ctx.use_tfsf:
                if ctx.tfsf_is_2d:
                    frame.tfsf_h_state = ctx.update_tfsf_2d_h(ctx.tfsf_cfg, frame.carry["tfsf"], dx, dt)
                else:
                    frame.tfsf_h_state = ctx.update_tfsf_1d_h(ctx.tfsf_cfg, frame.carry["tfsf"], dx, dt)

    def magnetic_drives(frame: UniformFrame):
        """Inject magnetic currents into the completed H half step."""
        # Magnetic current (Schelkunoff M / J-magnetic) injection —
        # applied after H update so the Yee leapfrog ordering is
        # H^{n+1/2} += -dt/mu · M^{n+1/2}. The coefficient is
        # pre-baked into the waveform values at construction time.
        if mag_drives is not None:
            frame.st = simulation.inject_drives(frame.st, mag_drives, frame.mag_src_vals)

    def e_update(frame: UniformFrame):
        """Advance E to n+1, including dispersive and design-box updates."""
        if not ctx.use_fast_he:
            # Snapshot E^n before the linear E-update for the reactive Kerr
            # increment (#437): E^{n+1} = E^n + (E_lin - E^n)/(1 + chi3|E^n|^2/eps_r).
            frame.e_prev_kerr = (frame.st.ex, frame.st.ey, frame.st.ez) if ctx.use_kerr else None
            # Snapshot E^n for the #677 sheet operator (it REPLACES the
            # standard update at masked tangential edges with
            # A*E^n + B*curlH, so it needs the pre-update E).
            frame.e_prev_sheet = (
                (frame.st.ex, frame.st.ey, frame.st.ez) if ctx.use_sheet_impedance else None)
            # #1179: the design box REDOES the E update at its own cells from
            # the pre-update state.  The whole state, because it needs E^n and
            # the same H^{n+1/2} the update below consumes; H is not touched
            # between the two, and a NamedTuple alias costs nothing.
            frame.st_prev_design = frame.st if ctx.use_design_box else None

            if ctx.use_upml:
                if ctx.use_debye or ctx.use_lorentz:
                    raise ValueError("boundary='upml' does not yet support dispersion")
                frame.st = ctx.apply_upml_e(frame.st, ctx.upml_coeffs, periodic=periodic,
                                      boundary=ctx.curl_boundary)
                frame.debye_new = None
                frame.lorentz_new = None
            else:
                frame.st, frame.debye_new, frame.lorentz_new = _update_e_with_optional_dispersion(
                    frame.st,
                    materials,
                    dt,
                    dx,
                    debye=(ctx.debye_coeffs, frame.carry["debye"]) if ctx.use_debye else None,
                    lorentz=(ctx.lorentz_coeffs, frame.carry["lorentz"]) if ctx.use_lorentz else None,
                    periodic=periodic,
                    aniso_eps=aniso_eps,
                    aniso_inv_eps=aniso_inv_eps,
                    stencil_order=ctx.stencil_order,
                    bloch=ctx.bloch,
                    boundary=ctx.curl_boundary,
                )

            # #1179 design box: redo the E update at the design cells with
            # coefficients built from the traced permittivity, leaving the
            # grid-wide ``materials`` constant.  Slot: immediately after the
            # E update, on the same H, before anything that reads or writes
            # E.  The fences in _build_step_setup guarantee that nothing
            # later in this step reads ``materials`` at a design cell.
            if ctx.use_design_box:
                frame.st = update_e_box(
                    frame.st, frame.st_prev_design, ctx.design_box.bounds,
                    ctx.design_box.ca, ctx.design_box.cb, dx,
                    periodic=periodic,
                    stencil_order=ctx.stencil_order,
                    bloch=ctx.bloch,
                    boundary=ctx.curl_boundary,
                )

    def design(frame: UniformFrame):
        """Expose the linear E update to the design adjoint before E treatment."""
        if design_hook is not None:
            frame.st, frame.design_record = design_hook(frame.st_prev_design, frame.st)

    def e_boundary(frame: UniformFrame):
        """Apply nonlinear/incident E corrections and the electric absorber."""
        if not ctx.use_fast_he:
            # Reactive Kerr correction: scale the E-increment by eps_r/eps_eff (#437).
            if ctx.use_kerr:
                frame.st = ctx.apply_kerr_ade(frame.st, frame.e_prev_kerr, ctx.kerr_chi3, materials.eps_r)

            if ctx.use_tfsf:
                frame.st = ctx.apply_tfsf_e(frame.st, ctx.tfsf_cfg, frame.tfsf_h_state, dx, dt)
            if ctx.use_waveguide_ports:
                from rfx.sources.waveguide_port import apply_waveguide_port_e as _apply_wg_e
                for cfg_meta in ctx.waveguide_meta:
                    frame.st = _apply_wg_e(frame.st, cfg_meta, frame.step_idx, dt, dx)
            if ctx.use_cpml:
                frame.st, frame.cpml_new = ctx.apply_cpml_e(
                    frame.st, ctx.cpml_params, frame.cpml_new, grid, ctx.cpml_axes,
                    materials=materials,
                    inv_eps_r_update=cpml_inv_eps_r,
                    boundary=ctx.curl_boundary)
            # Re-enforce Kottke-frozen E cells after CPML-E correction.

    def pec(frame: UniformFrame):
        """Enforce conductor faces, realized edges and occupancy after drives."""
        if not ctx.use_fast_he:
            # CPML adds a psi-driven correction that can thaw cells
            # where inv_eps==0; re-zero them here so the frozen
            # boundary condition is not violated.
            if ctx.use_aniso_inv:
                _inv_xx_r, _inv_yy_r, _inv_zz_r = aniso_inv_eps
                _PEC_INV_THRESHOLD = 1e-9
                frame.st = frame.st._replace(
                    ex=jnp.where(_inv_xx_r < _PEC_INV_THRESHOLD, 0.0, frame.st.ex),
                    ey=jnp.where(_inv_yy_r < _PEC_INV_THRESHOLD, 0.0, frame.st.ey),
                    ez=jnp.where(_inv_zz_r < _PEC_INV_THRESHOLD, 0.0, frame.st.ez),
                )

            if ctx.use_pec_faces:
                frame.st = apply_pec_faces(frame.st, ctx.pec_faces_frozen)

            if ctx.use_conformal and not ctx.use_aniso_inv:
                # Stage 1 path. Stage 2 (use_aniso_inv) skips this —
                # the inv-eps tensor encodes the fully-PEC-cell zero
                # already, so this would be redundant double-zeroing.
                from rfx.geometry.conformal import apply_conformal_pec
                frame.st = apply_conformal_pec(frame.st, ctx.conformal_weights[0], ctx.conformal_weights[1], ctx.conformal_weights[2])
            if ctx.use_pec_edges:
                # #931 §1.7: the (Mx, My, Mz) realized once at setup.
                #
                # NOT an ``elif`` on the conformal branch. Dey-Mittra is a
                # subpixel UPDATE-COEFFICIENT model, not a second geometry
                # realization: ``apply_conformal_pec`` zeroes only edges
                # whose weight is exactly 0, and no edge of a one-cell PEC
                # slab is fully covered — both its faces sit ON the slab's
                # own boundary, so w = 1/2 there. While the waveguide
                # S-matrix lane folded interior PEC into sigma=1e10 the
                # conductor survived anyway; with that fold deleted (#931)
                # the ``elif`` dropped it outright — measured on the
                # conformal PEC-short battery, min|S11| 0.2296 against a
                # gate of 0.99, restored to 0.9942 by applying both.
                frame.st = simulation.apply_pec_edges(frame.st, ctx.pec_edge_masks)

            # #1183: the design occupancy REDOES the scaling at its window
            # from the field BEFORE the grid-wide one, for the same reason
            # the design box keeps the pre-update state -- the multiply is
            # not invertible where the static factor is 0.
            st_prev_occ = frame.st if ctx.use_design_occupancy else None

            if ctx.use_pec_occupancy:
                frame.st = apply_pec_occupancy(
                    frame.st, ctx.pec_occupancy, ctx.periodic,
                    sheet_edge_masks=ctx.pec_static_edge_masks)

            # #1183 design occupancy: the traced 1 - M on its window, over
            # a field the static occupancy has not scaled. Outside the
            # window no design cell can reach, so the grid-wide factor
            # there is already the right one.
            if ctx.use_design_occupancy:
                frame.st = apply_pec_occupancy_box(
                    frame.st, st_prev_occ, ctx.design_occupancy.write,
                    ctx.design_occupancy.keep)

    def constitutive(frame: UniformFrame):
        """Apply sheet and lumped constitutive laws using retained E at n."""
        if not ctx.use_fast_he:
            # curlH comes from the SAME shared stencil helper update_e uses,
            # on the same H^{n+1/2} the E update consumed (H is unchanged
            # between the E update and this slot).
            if ctx.use_sheet_impedance:
                from rfx.core.yee import curl_h as _curl_h
                _scd = jnp.promote_types(frame.st.ex.dtype, jnp.float32)
                _curls = _curl_h(
                    frame.st.hx.astype(_scd), frame.st.hy.astype(_scd),
                    frame.st.hz.astype(_scd), dx, periodic,
                    ctx.stencil_order, ctx.bloch, boundary=ctx.curl_boundary)
                frame.st = _apply_sheet_e(
                    frame.st, frame.e_prev_sheet, _curls, ctx.sheet_impedance,
                    _sheet_coeffs)

        if ctx.use_lumped_rlc:
            frame.new_rlc_states = []
            for rlc_st, meta, e_prev in zip(
                    frame.carry["rlc_states"], ctx.rlc_meta, frame.rlc_e_prev):
                frame.st, rlc_st_new = ctx.update_rlc_element(
                    frame.st, rlc_st, meta, e_prev)
                frame.new_rlc_states.append(rlc_st_new)

    def before_sources(frame: UniformFrame):
        """Sample drive references before electric source injection."""
        # Wire-port DRIVE-REFERENCE DFT accumulation at the historical
        # PRE-injection slot (issue #683 x #764 decomposer recalibration,
        # docs/design_notes/issue683_decomposer_flip_predeclaration.md
        # section 3): the #308 receive-wave sign and Z0/n_cells
        # normalization were calibrated against the pre-injection drive
        # sample, so that sample is kept as its own channel `v_ref_dft` —
        # bit-identical to the historical `v_dft` — and feeds ONLY the
        # off-diagonal incident-wave denominator and the byte-frozen
        # legacy diagonal in decompose_wire_s_matrix. The physical
        # channels (v, i, v_port) are accumulated POST-injection below.
        if ctx.use_wire_sparams:
            frame.new_wire_refs = []
            frame.wire_ref_samples = []
            for accs, wp_meta in zip(frame.carry["wire_sparam_accs"], ctx.wire_sparam_meta):
                v_ref_dft = accs[4]
                mi, mj, mk = wp_meta.mid_i, wp_meta.mid_j, wp_meta.mid_k
                v_ref = -getattr(frame.st, wp_meta.component)[mi, mj, mk] * dx
                phase = port_dft_phase(frame.step_idx, wp_meta.freqs, dt, dtype=v_ref_dft.dtype)
                frame.new_wire_refs.append((v_ref_dft + v_ref * phase, phase))
                frame.wire_ref_samples.append(v_ref)

        # Lumped-port DRIVE-REFERENCE DFT accumulation at the historical
        # PRE-injection slot (issue #72): the #308 off-diagonal incident
        # wave is calibrated against this sample, so it is kept as its own
        # channel — bit-identical to the pre-decision `v_dft` — and feeds
        # ONLY the off-diagonal denominator in decompose_lumped_s_matrix.
        # The physical V/I are accumulated POST-injection below.  Mirrors
        # the wire-port block above (issue #683).
        if ctx.use_lumped_sparams:
            frame.new_lumped_refs = []
            frame.lumped_ref_samples = []
            for accs, lp_meta in zip(frame.carry["lumped_sparam_accs"], ctx.lumped_sparam_meta):
                v_ref_dft_l = accs[2]
                li, lj, lk = lp_meta.i, lp_meta.j, lp_meta.k
                v_ref_l = _port_voltage_value(getattr(frame.st, lp_meta.component)[li, lj, lk], dx)
                phase_l = port_dft_phase(frame.step_idx, lp_meta.freqs, dt, dtype=v_ref_dft_l.dtype)
                frame.new_lumped_refs.append((v_ref_dft_l + v_ref_l * phase_l, phase_l))
                frame.lumped_ref_samples.append(v_ref_l)

        # Reference-plane V/I DFT accumulation (issue #313 opt-in) — same
        # rect-DFT kernel as the port-cell channels.  This slot is before
        # the soft-source loop, but the planes sit >= 1 cell from every
        # source cell, so pre/post-injection sampling is identical here
        # and the #683 wire-port flip (which moved only the port-CELL
        # physical channels below the source loop) does not apply.
        if ctx.use_wire_refplanes:
            from rfx.probes.refplane import wire_refplane_step_vi
            frame.new_refplane_accs = []
            frame.refplane_samples = []
            for accs, rp_meta in zip(frame.carry["wire_refplane_accs"],
                                     ctx.wire_refplane_meta):
                v_dft_r, im_dft_r, ip_dft_r = accs
                v_r, im_r, ip_r = wire_refplane_step_vi(frame.st, rp_meta, dx)
                frame.refplane_samples.append(jnp.stack((v_r, im_r, ip_r)))
                phase_r = port_dft_phase(frame.step_idx, rp_meta.freqs, dt, dtype=v_dft_r.dtype)
                frame.new_refplane_accs.append((
                    v_dft_r + v_r * phase_r,
                    im_dft_r + im_r * port_dft_phase(frame.step_idx, rp_meta.freqs, dt, 'H', dtype=v_dft_r.dtype),
                    ip_dft_r + ip_r * port_dft_phase(frame.step_idx, rp_meta.freqs, dt, 'H', dtype=v_dft_r.dtype),
                ))

    def sources(frame: UniformFrame):
        """Inject the realized electric drives into E."""
        # Soft sources — cast source value to field dtype to avoid
        # mixed-precision scatter warnings (float32 -> float16).
        if drives is not None:
            inject = ctx.electric_injector or simulation.inject_drives
            frame.st = inject(frame.st, drives, frame.src_vals)

    def after_sources(frame: UniformFrame):
        """Read the completed physical fields for ports and observers."""
        frame.port_samples = []
        if ctx.use_wire_sparams:
            frame.new_wire_accs = []
            for accs, wp_meta, (v_ref_new, phase) in zip(
                    frame.carry["wire_sparam_accs"], ctx.wire_sparam_meta,
                    frame.new_wire_refs):
                v_dft, i_dft, vinc_dft, v_port_dft = accs[0], accs[1], accs[2], accs[3]
                mi, mj, mk = wp_meta.mid_i, wp_meta.mid_j, wp_meta.mid_k
                field_c = getattr(frame.st, wp_meta.component)
                v = -field_c[mi, mj, mk] * dx
                # Whole-port gap voltage (issue #764): the discrete line
                # integral of E across the LIVE run,
                # V_port = sum_live(-E_c*dx).  Only this SUM is
                # KVL/Faraday-constrained on the staggered grid; a single
                # cell is not (a PEC short forces sum V_c = 0 while V_mid
                # stays finite).  ``live_cells`` is a static tuple, so the
                # unroll resolves at trace time; an empty tuple (pre-#764
                # spec constructor) degenerates to the single midpoint cell.
                _lc = wp_meta.live_cells or ((mi, mj, mk),)
                v_port = -sum(
                    field_c[ci, cj, ck] for (ci, cj, ck) in _lc) * dx
                # #692: the SHARED loop, not a fourth inline copy. This block
                # used to spell the six branches verbatim with a raw
                # `h[i-1]`, so a port at index 0 read H from the OPPOSITE
                # face of the domain — and this lane is the one described at
                # the top of this file as "an AD-compatible alternative to
                # the Python-loop extract_s_matrix path", i.e. it must agree
                # with `probes.port_current` cell for cell. Measured against
                # the helper on random H: the two spellings disagreed at 9 of
                # 21 sampled (index, component) cells, every one of them at
                # an index with a zero coordinate on a back-read axis.
                # `_ampere_loop` is jit-safe here: `component` is a static
                # str and `mi/mj/mk` are static Python ints (see
                # WireSParamSpec), so every branch resolves at trace time.
                # (I reads H only, so it is identical on both sides of the
                # source loop; the flip moves V by the same-step injection
                # increment at driven cells, which #683's gate G2 measured
                # as EXACTLY the pre/post lane difference.)
                i_val = observers._ampere_loop(
                    frame.st, (mi, mj, mk), wp_meta.component, dx, periodic, boundary=ctx.curl_boundary)
                i_phase = port_dft_phase(frame.step_idx, wp_meta.freqs, dt, 'H', dtype=i_dft.dtype)
                frame.port_samples.append(jnp.stack((v, i_val, v_port,
                                                frame.wire_ref_samples[len(frame.new_wire_accs)])))
                frame.new_wire_accs.append((
                    v_dft + v * phase,
                    i_dft + i_val * i_phase,
                    vinc_dft,
                    v_port_dft + v_port * phase,
                    v_ref_new,
                ))

        # Lumped-port PHYSICAL V/I DFT accumulation AFTER source injection.
        # Decided by the lumped known-load decision run
        # (scripts/diagnostics/lumped_port_known_load_line.py), which the
        # 2026-09-05 scope note said was the missing input: on a
        # parallel-plate line terminated in a known R, the pre-injection
        # slot read |S11| 0.714 / 1.248 / 4.757 against a closed form of
        # 0.333 / 0 / 0.333, and its terminal V/(Zc·I) was -0.167 / +0.110
        # / +0.659 where the load is 0.5 / 1.0 / 2.0.  The one-cell WIRE
        # port on the SAME cell — same sigma (setup_wire_port with
        # n_live=1), same injection (apply_wire_port with n_live=1) — read
        # 0.334 / 0.0004 / 0.333 and 0.500 / 0.999 / 1.988.  The physics
        # was never the difference; the extraction lane was.  `t` stamping
        # is unchanged (phase computed at the pre slot and reused), so a
        # PASSIVE port reads bit-identically to the old slot in V.
        if ctx.use_lumped_sparams:
            frame.new_lumped_accs = []
            for accs, lp_meta, (v_ref_new_l, phase_l) in zip(
                    frame.carry["lumped_sparam_accs"], ctx.lumped_sparam_meta,
                    frame.new_lumped_refs):
                v_dft_l, i_dft_l = accs[0], accs[1]
                li, lj, lk = lp_meta.i, lp_meta.j, lp_meta.k
                v_l = _port_voltage_value(getattr(frame.st, lp_meta.component)[li, lj, lk], dx)
                # #692: shared loop — see the wire-port block above.
                i_val_l = observers._ampere_loop(
                    frame.st, (li, lj, lk), lp_meta.component, dx, periodic, boundary=ctx.curl_boundary)
                i_phase_l = port_dft_phase(frame.step_idx, lp_meta.freqs, dt, 'H', dtype=i_dft_l.dtype)
                frame.port_samples.append(jnp.stack((v_l, i_val_l,
                                                frame.lumped_ref_samples[len(frame.new_lumped_accs)])))
                frame.new_lumped_accs.append((
                    v_dft_l + v_l * phase_l,
                    i_dft_l + i_val_l * i_phase_l,
                    v_ref_new_l,
                ))

        if ctx.use_waveguide_ports:
            from rfx.sources.waveguide_port import (
                update_waveguide_port_probe,
            )

            frame.new_waveguide_port_accs = []
            for accs, cfg_meta in zip(frame.carry["waveguide_port_accs"], ctx.waveguide_meta):
                cfg = cfg_meta._replace(
                    v_probe_t=accs[0],
                    v_ref_t=accs[1],
                    i_probe_t=accs[2],
                    i_ref_t=accs[3],
                    v_inc_t=accs[4],
                    n_steps_recorded=accs[5],
                )
                # TFSF-style H and E corrections are applied earlier in
                # their respective Yee sub-steps (canonical TFSF slots).
                # NOTE: this samples `st` AFTER source injection above.
                # The same docstring-contract concern as wire/lumped
                # applies here, but waveguide-port is out of scope for
                # this fix (issue #29 OPEN tracks waveguide-port issues).
                cfg_updated = update_waveguide_port_probe(cfg, frame.st, dt, dx)
                frame.new_waveguide_port_accs.append(
                    (
                        cfg_updated.v_probe_t,
                        cfg_updated.v_ref_t,
                        cfg_updated.i_probe_t,
                        cfg_updated.i_ref_t,
                        cfg_updated.v_inc_t,
                        cfg_updated.n_steps_recorded,
                    )
                )

        # Probe samples
        samples = [getattr(frame.st, pc)[pi, pj, pk]
                   for pi, pj, pk, pc in ctx.prb_meta]
        frame.probe_out = jnp.stack(samples) if samples else jnp.zeros(0)

        # NTFF accumulation
        if ctx.use_ntff:
            frame.ntff_new = ctx.accumulate_ntff(
                frame.carry["ntff"], frame.st, ctx.ntff, dt, frame.step_idx)

        # Block current moments — same slot as the NTFF box, so the state
        # holds E at (n+1)*dt and H at (n+1/2)*dt, and the soft-source loop
        # above has already put the feed current into E.
        if ctx.use_current_moments:
            frame.cm_new = ctx.accumulate_current_moments(
                frame.carry["current_moments"], frame.st, frame.e_prev_slab,
                ctx.current_moments, dt, frame.step_idx)

        from rfx.measurement.accumulators import planes, flux
        if ctx.use_dft_planes:
            frame.new_dft_planes, frame.plane_samples = planes(
                frame.st, frame.carry["dft_planes"], ctx.dft_meta, dt, frame.step_idx)
        if ctx.use_flux_monitors:
            frame.new_flux_accs = flux(frame.st, frame.carry["flux_monitors"], ctx.flux_meta, dt,
                                 frame.step_idx, window_step=frame.st.step)

    def step_end(frame: UniformFrame):
        """Advance auxiliary incident E and expose end-of-step extras."""
        frame.t = frame.step_idx.astype(jnp.float32) * dt
        if ctx.use_tfsf:
            if ctx.tfsf_is_2d:
                frame.tfsf_new = ctx.update_tfsf_2d_e(ctx.tfsf_cfg, frame.tfsf_h_state, dx, dt, frame.t)
            else:
                frame.tfsf_new = ctx.update_tfsf_1d_e(ctx.tfsf_cfg, frame.tfsf_h_state, dx, dt, frame.t)

        # ---- per-step extras (caller-specific outputs) ----
        frame.extras: dict = {}
        if ctx.use_snapshot:
            frame.extras["snap_fields"] = ctx.snapshot_extractor(frame.st)
        if ctx.use_monitor:
            frame.extras["monitor_val"] = getattr(frame.st, ctx.monitor_component)[
                ctx.mon_idx[0], ctx.mon_idx[1], ctx.mon_idx[2]]

    def finish(frame: UniformFrame):
        """Assemble the existing runner carry and per-step outputs."""
        # Rebuild carry
        new_carry: dict = {"fdtd": frame.st}
        if "dft_time_records" in frame.carry:
            new_carry["dft_time_records"] = tuple(
                jax.lax.stop_gradient(record.at[frame.step_idx].set(sample))
                for record, sample in zip(frame.carry["dft_time_records"], frame.plane_samples))
        if ctx.use_wire_refplanes:
            frame.port_samples.extend(frame.refplane_samples)
        if "sparam_time_records" in frame.carry:
            new_carry["sparam_time_records"] = tuple(
                jax.lax.stop_gradient(record.at[frame.step_idx].set(sample))
                for record, sample in zip(frame.carry["sparam_time_records"], frame.port_samples))
        if ctx.use_cpml:
            new_carry["cpml"] = frame.cpml_new
        if ctx.use_debye:
            new_carry["debye"] = frame.debye_new
        if ctx.use_lorentz:
            new_carry["lorentz"] = frame.lorentz_new
        if ctx.use_tfsf:
            new_carry["tfsf"] = frame.tfsf_new
        if ctx.use_ntff:
            new_carry["ntff"] = frame.ntff_new
        if ctx.use_current_moments:
            new_carry["current_moments"] = frame.cm_new
        if ctx.use_dft_planes:
            new_carry["dft_planes"] = tuple(frame.new_dft_planes)
        if ctx.use_flux_monitors:
            new_carry["flux_monitors"] = tuple(frame.new_flux_accs)
        if ctx.use_waveguide_ports:
            new_carry["waveguide_port_accs"] = tuple(frame.new_waveguide_port_accs)
        if ctx.use_wire_sparams:
            new_carry["wire_sparam_accs"] = tuple(frame.new_wire_accs)
        if ctx.use_lumped_sparams:
            new_carry["lumped_sparam_accs"] = tuple(frame.new_lumped_accs)
        if ctx.use_wire_refplanes:
            new_carry["wire_refplane_accs"] = tuple(frame.new_refplane_accs)
        if ctx.use_lumped_rlc:
            new_carry["rlc_states"] = tuple(frame.new_rlc_states)

        if design_hook is not None:
            frame.extras["design_record"] = frame.design_record
        return new_carry, frame.probe_out, frame.extras

    attachments = {
        HookPoint.AFTER_H: (magnetic_drives,) if mag_drives is not None else (),
        HookPoint.AFTER_E_UPDATE: (design,) if design_hook is not None else (),
        HookPoint.BEFORE_SOURCES: ((before_sources,) if (ctx.use_wire_sparams
                                  or ctx.use_lumped_sparams or ctx.use_wire_refplanes) else ()),
        HookPoint.AFTER_SOURCES: (after_sources,),
        HookPoint.STEP_END: (step_end,),
    }
    for point, extra in (hooks or {}).items():
        attachments[point] += tuple(extra)
    kernels = {
        Kernel.H_UPDATE: h_update,
        Kernel.H_BOUNDARY: h_boundary,
        Kernel.E_UPDATE: e_update,
        Kernel.E_BOUNDARY: e_boundary,
        Kernel.CONSTITUTIVE: constitutive,
        Kernel.SOURCES: sources,
        Kernel.PEC: pec,
    }
    if ctx.use_fast_he:
        for phase in (Kernel.H_UPDATE, Kernel.H_BOUNDARY, Kernel.E_UPDATE, Kernel.PEC):
            del kernels[phase]
        kernels[Kernel.HE_UPDATE] = he_update
    advance = compose(kernels, attachments)

    def core_step(carry, step_idx, src_vals, mag_src_vals):
        frame = UniformFrame(carry["fdtd"], carry, step_idx, src_vals, mag_src_vals)
        prepare(frame)
        advance(frame)
        return finish(frame)

    return core_step
