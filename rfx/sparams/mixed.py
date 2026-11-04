"""Mixed-family S-matrix calculators bound onto the public Simulation API.

Drive-specific declarations and run state belong to private declaration
variants. The caller retains its declared port and observer containers.
"""

from __future__ import annotations

from rfx.sparams import _mixed_stages

from rfx.errors import AccuracyWarning, RfxRefusal, RfxUnsupported

from rfx._declarations import declaration_variant
from rfx._internal._precision import resolve_field_dtype
from rfx.preflight.realization import release_inspection
from rfx.boundaries.pmc import refuse_port_on_magnetic_face

from rfx.preflight.line_stub import line_stub_admission as _line_stub_admit
from rfx._diagnostic_transport import merge_diagnostics, report_diagnostics, diagnostic_refusal

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.sources.sources import GaussianPulse

from rfx.api._spec import MixedSMatrixResult

from rfx.sparams._mixed_checks import _mixed_refusal_masks, _warn_refused_mixed_entries
from rfx.sparams._port_reference import renormalize_power_waves

from rfx.sparams._common import (
    _msl_cell_profile,
    msl_modal_voltage,
    _msl_wave_split_reliability,
    _warn_msl_wave_split_unreliable,
    _warn_msl_beta_scan_railed,
    _warn_if_ringdown_truncated,
    _project_passive,
    _warn_if_passivity_projected,
    _warn_if_nonpassive_smatrix,
    _assemble_mixed_power_wave_s,
    _mixed_reciprocity_deviation,
    _mixed_flux_magnitude_override,
    _register_msl_h_planes,
    _collocated_msl_h,
)


def compute_mixed_s_matrix(
    self,
    *,
    n_steps: int | None = None,
    num_periods: float = 40.0,
    freqs: "jnp.ndarray | None" = None,
    n_freqs: int = 100,
    strict_extractor: bool = False,
    enforce_passivity: bool = True,
    skip_preflight: bool = False,
    return_diagnostics: bool = False,
    magnitude_channel: str = "flux",
    reciprocity_tol: float = 0.06,
) -> "MixedSMatrixResult":
    """Mixed-family S-matrix: lumped/wire ports + MSL ports.

    End-to-end S-parameters on ONE structure carrying two port
    families — the first supported pair is a homogeneous lumped OR
    wire set (``add_port``) together with MSL ports (``add_msl_port``),
    e.g. a vertical probe feed launching onto a microstrip line.

    Each port is driven in turn (lumped/wire ports first, then MSL
    ports, registration order within each family); the non-driven
    lumped/wire ports remain physical matched resistor loads and the
    non-driven MSL ports are passive probe columns. Extraction reuses
    the validated per-family wave machinery unchanged and combines the
    waves in the **Kurokawa power-wave convention** (each wave divided
    by ``sqrt(Re Z0)`` of its own port) — with unequal reference
    impedances across families a pseudo-wave ratio would be off by
    ``sqrt(Z_j/Z_i)``; reciprocity of a reciprocal
    structure is the committed internal falsifier for this choice.
    The returned ``S`` and ``S_wave`` are referenced to every port's
    declared ``impedance=`` (``z0_ref``): MSL waves are formed at
    the analytic line Z0 (``line_impedances``) and the matrix is then
    referred to ``z0_ref`` by ``renormalize_power_waves``.

    MAGNITUDE CHANNEL (``magnitude_channel``, default ``"flux"``):
    the mixed-family falsifier battery + an independent Poynting-flux referee
    measured the port-cell V*I accounting undercounting delivered
    power ~3x at a wire probe feed (near-field deflation) and
    the analytic Hammerstad-Jensen Z0 anchor diverging from the
    measured line ratio on an interface-aligned mesh. Off-diagonal
    MAGNITUDES are therefore taken from Poynting flux by default:
    auto-registered surfaces (a closed 5-face box around each
    lumped/wire port over the z-lo ground; a full cross-section plane
    at each MSL port's probe-0 x) give per-drive net powers, and

        |S_ij|^2 = P_arrive,i / (P_net,j / (1 - |S_jj|^2))

    where ``S_jj`` is the per-family diagonal supplying the reflection
    correction; no Z0 anchor enters the magnitude. Off-diagonal PHASE
    still comes from the wave channel (see below).
    ``magnitude_channel="wave"`` keeps the raw power-wave magnitudes
    (diagnostic; carries the near-field deflation at lumped/wire ports).

    HONESTY NOTES (read before quoting numbers):

    * **Neither diagonal is verified on this lane, and the returned
      diagonal is not always the measured one.** Two separate
      findings: (a) the wire port-cell V*I accounting was measured
      undercounting delivered power ~3x against an independent
      Poynting referee (at a near-field-dominated vertical probe), and on an end-fed
      fixture the MSL probe plane's local ``V/I`` was ~591 ohm and
      strongly reactive while the reported ``|S22|`` was 0.03 — those
      cannot both be right. (b) With ``enforce_passivity=True`` (the
      default), ``_project_passive`` is a JOINT SVD clip: when any
      entry is non-passive it rewrites others as a side effect.
      Measured on the committed test fixture, the shipped MSL
      diagonal came out ~4x its unprojected value. So ``result.S``'s
      diagonal is a projected quantity, not a raw per-family
      measurement — read ``S_raw`` and ``passivity_correction`` for
      what was actually measured, and treat any diagonal-derived
      conclusion accordingly.

    * The flux magnitudes inherit a flux-accounting envelope (box
      leakage + finite DFT; the Phase-0 referee class measured ~1.3%
      on the canonical thru) and an ill-conditioning guard: when
      ``1 - |S_jj|^2`` is small (near-total reflection at the driven
      port) the normalization is unreliable and a warning names the
      column.
    * **A per-column power sum near 1 is NOT evidence on this
      channel.** Substituting the definition gives
      ``sum_i |S_ij|^2 = |S_jj|^2 + (P_arr/P_net)(1 - |S_jj|^2)``,
      which is identically 1 whenever the arriving power equals the
      net launched power — for ANY value of the diagonal. The flux
      normalization makes passivity an identity, not a measurement,
      so do not quote column power as a passivity check here (it
      still detects power GAIN, i.e. ``P_arr > P_net``). The
      independent internal check on this channel is RECIPROCITY:
      ``S_ij`` and ``S_ji`` come from different runs with different
      normalizations, so their agreement is real evidence. It runs
      automatically on every extraction (see ``reciprocity_tol``),
      audits the RAW matrix, and is the check that a wrong diagonal
      trips.
    * The returned ``S`` warns (``AccuracyWarning``, nothing changed)
      where its largest singular value exceeds ``1 + 1e-2``; on a bin
      with a refused (NaN) column, the finite columns are judged as a
      lower bound. A projected ``S`` (``enforce_passivity=True``) is
      clipped to 1 first. Read ``S_raw`` for the measured matrix.
    * With ``magnitude_channel="wave"``, off-diagonal magnitudes that
      RECEIVE at a lumped/wire port cell inherit the
      near-field deflation of the default port-cell waves. The
      returned ``s21_power_witness`` cross-checks the MSL-receiving
      direction against a delivered-power normalization; quote it
      alongside ``|S|``.
    * Cross-family off-diagonal PHASE mixes two reference-plane
      conventions (port cell vs the MSL port's declared feed plane, to
      which the MSL rows/columns are shifted from probe 0) and a
      component-mixing ±1 (probes.py sign-convention fence);
      magnitude is the validated observable.
    * ``settling_db`` is the ring-down witness (above −40 dB =
      truncation suspect); preflight output is part of the result.

    v1 restrictions (loud ``NotImplementedError``): uniform mesh only,
    no waveguide/Floquet/coax/TFSF registrations, no bare sources or
    0-ohm ports (they would fire in every drive run), no
    ``reference_plane_cells`` wire ports, no mixed lumped+wire set
    (same fence as the production scan driver), imperative only (no
    ``eps_override`` AD channel). The default ``"flux"`` channel adds
    two more, because the per-port flux box omits its bottom face and
    treats the port extent as a height: a **PEC ``z_lo`` boundary**
    and **vertical (``component="ez"``) lumped/wire ports** are
    required. ``magnitude_channel="wave"`` makes neither assumption.

    Parameters
    ----------
    reciprocity_tol : float, default 0.06
        Relative ``|S_ij|`` vs ``|S_ji|`` disagreement above which the
        reciprocity witness warns. Deliberately set BELOW the 9%
        residual measured on this lane's own reference fixture, so
        that fixture warns rather than passing silently: a tolerance
        above the known residual would document the check and never
        fire it.

    Returns
    -------
    MixedSMatrixResult
    """
    resolve_field_dtype(self._precision)
    _diagnostics = ()
    import dataclasses as _dc

    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.sources.msl_port import (
        _msl_yz_cells,
        msl_cross_section_span,
        msl_h_plane_stencil,
        msl_loop_current,
        msl_probe_x_coords_n,
    )

    lw_entries, wire_mode = _mixed_stages._validate_mixed_family(
        RfxRefusal, RfxUnsupported, _diagnostics, diagnostic_refusal, self,
    )

    n_lw = len(lw_entries)
    from rfx.materials.thin_conductor import refuse_f0_sheets as _refuse_f0_hj
    _refuse_f0_hj(self._thin_conductors, "MSL junction S-parameter")
    from rfx.runners._admission import admit
    admit(self, "mixed_s_matrix")
    _line_stub_scope = _line_stub_admit(self, freqs)
    grid = self._build_grid()

    if freqs is None:
        freqs_arr = np.asarray(
            jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs)
        )
    else:
        freqs_arr = np.asarray(freqs)
    n_freqs_used = int(freqs_arr.shape[0])
    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=num_periods)

    entries, n_msl, msl_ports, probe_xs, h_stencils, probe_clearance = _mixed_stages._prepare_mixed_probes(
        grid, msl_h_plane_stencil, msl_probe_x_coords_n, self,
    )
    _mixed_stages._check_mixed_ladder(
        RfxRefusal, _diagnostics, diagnostic_refusal, entries, grid, probe_xs,
        self,
    )

    dy_arr, dz_arr, port_idx_meta = _mixed_stages._prepare_mixed_indices(
        _msl_cell_profile, grid, msl_cross_section_span, msl_ports,
    )

    # One materials assembly shared by the HJ eps anchor AND every
    # drive run (materials do not depend on excite flags).
    _mx_pec_sheets: list = []
    _mx_pec_wires: list = []
    materials, debye_spec, lorentz_spec, pec_mask, _, _, _ = \
        self._assemble_materials(
            grid, pec_sheets=_mx_pec_sheets, pec_wires=_mx_pec_wires)
    # realized wall planes locate the trace, not cells.
    from rfx.boundaries.pec import (
        realized_pec_edge_masks as _rpem_mx,
    )
    from rfx.probes.msl_wave_decomp import (
        realized_trace_planes_on_column as _trace_planes_mx,
    )
    _mx_pec_edge_masks = None
    if pec_mask is not None or _mx_pec_sheets or _mx_pec_wires:
        with jax.ensure_compile_time_eval():
            _mx_pec_edge_masks = _rpem_mx(
                pec_mask, sheets=tuple(_mx_pec_sheets),
                wires=tuple(_mx_pec_wires),
                periodic=self._periodic_flags())
    _mx_pec_edges_traced = (_mx_pec_edge_masks is not None
                            and any(is_tracer(mask) for mask in _mx_pec_edge_masks))

    z0_hj_per_port, beta0_per_port, eps_r_ref_per_port, trace_k_per_port, n_live_lw = _mixed_stages._prepare_mixed_anchors(
        RfxRefusal, _msl_yz_cells, _mx_pec_edge_masks, _mx_pec_edges_traced, _trace_planes_mx, entries, freqs_arr,
        grid, hammerstad_jensen_z0_eps_eff, lw_entries, materials, msl_ports, n_lw, n_msl,
        port_idx_meta, self, wire_mode,
    )

    if not skip_preflight:
        # One preflight for the full registration (run() would fire it
        # per drive run — 2*n_ports repeats of the same advisories).
        _diagnostics = report_diagnostics(self._auto_preflight(context="compute_mixed_s_matrix", check_ntff="advisory"))

    _mixed_stages._check_mixed_flux(
        RfxRefusal, RfxUnsupported, _diagnostics, diagnostic_refusal, lw_entries, magnitude_channel, refuse_port_on_magnetic_face,
        self,
    )

    drive_plan = [("lw", j) for j in range(n_lw)] + \
                 [("msl", d) for d in range(n_msl)]
    n_runs = len(drive_plan)
    _complex_dtype = (
        jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    )

    observers, _witness_base, _witness_total, flux_faces_lw, flux_names_msl = _mixed_stages._prepare_mixed_observers(
        declaration_variant, entries, freqs_arr, grid, jnp, lw_entries, magnitude_channel,
        probe_xs, self,
    )

    v_lw = np.zeros((n_runs, n_lw, n_freqs_used), dtype=np.complex128)
    vref_lw = np.zeros((n_runs, n_lw, n_freqs_used), dtype=np.complex128)
    i_lw = np.zeros((n_runs, n_lw, n_freqs_used), dtype=np.complex128)
    v0_msl = np.zeros((n_runs, n_msl, n_freqs_used), dtype=np.complex128)
    i_msl = np.zeros((n_runs, n_msl, n_freqs_used), dtype=np.complex128)
    _n_probes_max = max(len(pxs) for pxs in probe_xs)
    v_lad = np.zeros(
        (n_runs, n_msl, _n_probes_max, n_freqs_used),
        dtype=np.complex128,
    )
    settling_db_runs = np.full(n_runs, np.nan)
    settling_details = [{"status": "undetermined", "reason": "traced or unavailable S channels"} for _ in range(n_runs)]
    # Signed per-run flux accountings (magnitude_channel="flux"):
    # box_lw[run, p]  = net OUTWARD box flux at lumped/wire port p
    # plane_msl[run, p] = raw +x-directed flux at MSL port p's plane
    box_lw = np.zeros((n_runs, n_lw, n_freqs_used))
    plane_msl = np.zeros((n_runs, n_msl, n_freqs_used))

    for run_idx, (fam, loc) in enumerate(drive_plan):
        variant, names, h_names = _mixed_stages._prepare_mixed_drive(
            GaussianPulse, _dc, _register_msl_h_planes, declaration_variant, entries, fam, freqs_arr,
            h_stencils, jnp, loc, observers, probe_xs, run_idx, self,
        )

        _diagnostics, raw, planes, v_lw, i_lw, vref_lw, box_lw, plane_msl = _mixed_stages._run_mixed_drive(
            _diagnostics, _mx_pec_sheets, _mx_pec_wires, box_lw, debye_spec, flux_faces_lw, flux_names_msl,
            freqs_arr, grid, i_lw, lorentz_spec, magnitude_channel, materials, merge_diagnostics,
            n_freqs_used, n_lw, n_steps, pec_mask, plane_msl, run_idx,
            v_lw, variant, vref_lw, wire_mode,
        )

        settling_db_runs, settling_details = _mixed_stages._score_mixed_record(
            _witness_base, _witness_total, dy_arr, dz_arr, freqs_arr, grid, h_names,
            h_stencils, is_tracer, msl_ports, names, port_idx_meta, probe_xs,
            raw, run_idx, self, settling_db_runs, settling_details, trace_k_per_port, variant,
        )


        # MSL planes already carry physical E/H stamps from the shared DFT.
        for p_idx, meta in enumerate(port_idx_meta):
            nm = names[p_idx]
            for q_idx in range(len(probe_xs[p_idx])):
                ez_plane = jnp.asarray(
                    planes[nm + f"_ez{q_idx}"].accumulator
                )
                # Anchor the V-span top on the rasterized trace
                # node, not round(h_sub/dx) — see the sibling
                # comment in compute_msl_s_matrix.
                v_q = msl_modal_voltage(
                    ez_plane, j_centre=meta["j_centre"],
                    k_lo=meta["k_lo"],
                    k_hi=trace_k_per_port[p_idx][0],
                    dz_arr=dz_arr, dtype=_complex_dtype,
                )
                v_lad[run_idx, p_idx, q_idx, :] = np.asarray(v_q)
            v_f = jnp.asarray(v_lad[run_idx, p_idx, 0, :])
            hy_plane, hz_plane = _collocated_msl_h(
                planes, h_names[p_idx], h_stencils[p_idx]["weights"],
            )
            k_tr_lo, k_tr_hi = trace_k_per_port[p_idx]
            i_f = msl_loop_current(
                hy_plane, hz_plane,
                j_lo=meta["j_lo"], j_hi=meta["j_hi"],
                k_trace_lo=k_tr_lo, k_trace_hi=k_tr_hi,
                dy_arr=dy_arr, dz_arr=dz_arr,
                direction=msl_ports[p_idx].direction,
            )
            v0_msl[run_idx, p_idx, :] = np.asarray(v_f)
            i_msl[run_idx, p_idx, :] = np.asarray(i_f)

    # ---- Power-wave assembly (pure; unit-tested separately) ----
    S, s21_power = _assemble_mixed_power_wave_s(
        v_lw, i_lw, v0_msl, i_msl,
        np.asarray([pe.impedance for pe in lw_entries]),
        n_live_lw, np.asarray(z0_hj_per_port),
        wire_mode, drive_plan,
        v_ref_lw=vref_lw,
    )
    S = jnp.asarray(S, dtype=_complex_dtype)

    z0_msl_fit, beta_railed_msl = _mixed_stages._fit_mixed_impedance(
        beta0_per_port, entries, i_msl, jax, jnp, n_freqs_used,
        n_lw, n_msl, probe_xs, v_lad, z0_hj_per_port,
    )
    # β-scan rail flags for the diagnostic N-probe fit: a railed bin's |Zc|/β above
    # are the ±35% scan-window
    # limit, not a measurement. S is unaffected (the MSL diagonal
    # uses the analytic HJ anchor; the feed-plane shift below uses
    # the closed-form beta, never the fit).
    _warn_msl_beta_scan_railed(
        beta_railed_msl, freqs_arr,
        tuple(pe.name for pe in entries),
    )
    refused_flux, refused_wave = _mixed_refusal_masks(
        n_lw + n_msl, n_lw, n_live_lw, wire_mode, drive_plan)
    refused_positions = refused_flux if magnitude_channel == "flux" else refused_wave
    s_wave_full = None

    if magnitude_channel == "flux":
        import warnings as _wf
        s_wave_full = np.asarray(jax.lax.stop_gradient(S)).copy()
        s_wave_full[refused_wave] = np.nan
        msl_away = [
            (+1.0 if pe.direction == "+x" else -1.0) for pe in entries
        ]
        S, _ill_cond, _neg_power = _mixed_flux_magnitude_override(
            S, box_lw, plane_msl, drive_plan, msl_away, n_lw,
            refused_columns=np.flatnonzero(np.all(refused_flux, axis=0)),
        )
        S = jnp.asarray(S, dtype=_complex_dtype)
        for col in range(n_lw + n_msl):
            n_ill = int(_ill_cond[col].sum())
            if n_ill:
                _wf.warn(
                    f"compute_mixed_s_matrix: flux magnitude for "
                    f"driven port index {col} is ill-conditioned at "
                    f"{n_ill}/{n_freqs_used} bins "
                    f"(1 - |S_jj|^2 < 0.05 — near-total reflection; "
                    "the incident-power reconstruction divides by "
                    "almost nothing). Off-diagonals of that column "
                    "are UNRELIABLE at those bins.", AccuracyWarning,
                    stacklevel=2,
                )
            n_neg = int(_neg_power[col].sum())
            if n_neg:
                _wf.warn(
                    f"compute_mixed_s_matrix: flux power at port "
                    f"index {col} came out NEGATIVE at "
                    f"{n_neg}/{n_freqs_used} bins — a sign or "
                    "accounting defect in that port's flux "
                    "surface (net launched power cannot flow INTO "
                    "a driven port, and arriving power cannot be "
                    "negative at a receive port). Clipped to zero, "
                    "which reports |S| = 0 rather than failing: "
                    "treat those bins as an extraction failure, "
                    "not as an absence of coupling.", AccuracyWarning,
                    stacklevel=2,
                )

    S, s_wave_full, plane_offsets_all, shift_beta_all = _mixed_stages._shift_mixed_planes(
        S, _complex_dtype, entries, eps_r_ref_per_port, freqs_arr, grid, msl_ports,
        n_lw, n_msl, probe_xs, s_wave_full,
    )

    # MSL standing-wave-null reliability from each MSL port's own
    # driven run (mirrors compute_msl_s_matrix).
    reliable = None
    try:
        v_port = np.stack([
            v0_msl[n_lw + p, p, :] for p in range(n_msl)
        ])
        i_port = np.stack([
            i_msl[n_lw + p, p, :] for p in range(n_msl)
        ])
        reliable = _msl_wave_split_reliability(
            v_port, i_port, freqs_arr
        )
        _warn_msl_wave_split_unreliable(reliable, freqs_arr)
    except (ValueError, TypeError):
        pass

    port_names = tuple(
        [f"lw{k}" for k in range(n_lw)]
        + [pe.name for pe in entries]
    )
    port_families = tuple(
        [("wire" if wire_mode else "lumped")] * n_lw
        + ["msl"] * n_msl
    )
    # S is referenced to every port's declared impedance=. The
    # MSL waves were formed at the analytic line Z0 (line_impedances);
    # lumped/wire ports already use their declared impedance, so only
    # the MSL rows and columns move, once, before the projection.
    line_impedances = np.asarray(
        [float(pe.impedance) for pe in lw_entries]
        + z0_hj_per_port
    )
    z0_ref = np.asarray(
        [float(pe.impedance) for pe in lw_entries]
        + [float(pe.impedance) for pe in entries]
    )
    S = renormalize_power_waves(S, line_impedances, z0_ref)
    if s_wave_full is not None:
        s_wave_full = renormalize_power_waves(
            s_wave_full, line_impedances, z0_ref)

    S = jnp.where(refused_positions[..., None], jnp.nan, S)
    s_raw = None
    passivity_correction = None
    _warn_refused_mixed_entries(port_names, refused_flux, refused_wave, magnitude_channel)
    if enforce_passivity and not np.any(refused_positions) and not is_tracer(S):
        s_projected, correction = _project_passive(S)
        if bool(np.any(np.asarray(correction) > 0.0)):
            s_raw = S
            passivity_correction = correction
            S = s_projected

    result = MixedSMatrixResult(
        diagnostics=_diagnostics,
        S=S,
        freqs=np.asarray(freqs_arr),
        port_names=port_names,
        port_families=port_families,
        z0_ref=z0_ref,
        line_impedances=line_impedances,
        settling_db=settling_db_runs,
        settling_witness=tuple(settling_details),
        s21_power_witness=s21_power,
        reliable=reliable,
        S_raw=s_raw,
        passivity_correction=passivity_correction,
        S_wave=s_wave_full,
        magnitude_channel=magnitude_channel,
        beta_railed=beta_railed_msl,
        probe_clearance=probe_clearance,
        reference_plane_offsets_m=plane_offsets_all,
        shift_beta=shift_beta_all,
    )
    _warn_if_ringdown_truncated(
        settling_db_runs, port_names, num_periods=num_periods, witnesses=settling_details,
    )
    if passivity_correction is not None and not is_tracer(passivity_correction):
        _warn_if_passivity_projected(passivity_correction, freqs_arr)
    import dataclasses as _dc2
    audit_result = (
        result if s_raw is None else _dc2.replace(result, S=s_raw)
    )
    _warn_if_nonpassive_smatrix(
        audit_result,
        extractor="compute_mixed_s_matrix",
        strict=strict_extractor,
        passivity_tol=0.10,
        refused_positions=refused_positions,
    )
    from rfx.sparams.passivity_warning import warn_if_singular_value_exceeds_one
    warn_if_singular_value_exceeds_one(
        result.S, result.freqs, extractor="compute_mixed_s_matrix",
    )
    # RECIPROCITY WITNESS — the only independent runtime check on
    # the flux channel (review finding: the shared passivity audit
    # above is structurally inert here, because column power is an
    # identity under the flux normalization; and
    # validate_port_smatrix's own reciprocity option compares
    # COMPLEX S, which would misfire on this lane where
    # cross-family phase is provisional by construction).
    # S_ij and S_ji come from different drive runs with different
    # normalizations, so their MAGNITUDE agreement is real
    # evidence — and it is exactly the check that would have
    # caught a wrong diagonal feeding the flux normalization.
    #
    # Audited on the RAW (unprojected) matrix for the same reason
    # the passivity self-check above is: `_project_passive` is a
    # joint SVD clip, so it moves entries TOWARD each other and
    # would understate the disagreement actually measured. It
    # also rewrites diagonals as a side effect (measured on the
    # test fixture: a shipped MSL diagonal ~4x its raw value),
    # which is precisely what this witness exists to surface.
    _rec = _mixed_reciprocity_deviation(
        S if s_raw is None else s_raw
    )
    if _rec is not None:
        _pair, _dev_max = _rec
        if _dev_max > reciprocity_tol:
            import warnings as _wr
            _wr.warn(
                f"compute_mixed_s_matrix: reciprocity deviation "
                f"max {_dev_max * 100:.1f}% between |S[{_pair[0]},"
                f"{_pair[1]}]| and |S[{_pair[1]},{_pair[0]}]| "
                f"(tolerance {reciprocity_tol * 100:.0f}%). For a "
                "reciprocal structure these must agree; a "
                "disagreement means one DRIVEN-port diagonal is "
                "wrong (the diagonals set the incident-power "
                "normalization P_inc = P_net/(1-|S_jj|^2)), or a "
                "flux surface is mis-signed. Note the per-column "
                "power sum CANNOT detect this — it is an identity "
                "on this channel. Inspect the diagonals and "
                "settling_db before quoting any |S|.", AccuracyWarning,
                stacklevel=2,
            )
    if return_diagnostics:
        # R5 inspection surface: the raw per-run phasors behind
        # every wave, so a suspicious |S| can be traced to V/I
        # health (e.g. a broken Ampere loop shows as v0/i far
        # from the line Z0) without re-running.
        diagnostics = {
            "v_lw": v_lw, "i_lw": i_lw,
            "v0_msl": v0_msl, "i_msl": i_msl,
            "drive_plan": drive_plan,
            "z0_hj_msl": np.asarray(z0_hj_per_port),
            # |Zc| from the N-probe fit — DIAGNOSTIC ONLY, never
            # substituted into S (sign unstable; see the block
            # above). Magnitude only: no sign claim is made.
            "z0_msl_fit_abs": z0_msl_fit,
            "box_lw_flux": box_lw,
            "plane_msl_flux": plane_msl,
        }
        release_inspection(self)
        return result, diagnostics
    release_inspection(self)
    return result



# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# ``compute_mixed_s_matrix`` was a ``def`` in the ``_SparamMixin`` class body,
# so its ``__qualname__`` read ``_SparamMixin.compute_mixed_s_matrix``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``_SparamMixin.<name>`` -> ``Simulation.<name>`` at
# class-composition time so that a bad keyword argument reports
# ``Simulation.compute_mixed_s_matrix() got an unexpected keyword
# argument``, and it SKIPS any function whose qualname does not match that
# pattern. Leaving the bare name here would therefore change that TypeError
# message -- a user-visible behaviour change in a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` pins it.
# ---------------------------------------------------------------------------
compute_mixed_s_matrix.__qualname__ = "_SparamMixin.compute_mixed_s_matrix"
