"""S-parameter passivity checks, setup resolution and advisories; rfx.sparams._common and rfx.api._sparams re-export every name, and new code imports from here."""

from __future__ import annotations
from rfx._warning_origin import user_stacklevel
from rfx.errors import AccuracyWarning, ConfigurationWarning
from rfx.model.graded_admission import host_cells
import numpy as np
from rfx._internal._grid_metric import cells_crossed, distinct_cell_sizes
from rfx.core.jax_utils import is_tracer
from rfx.sparams._mixed_checks import _warn_partial_nonpassive
from rfx.sparams._grid_metrics import absorber_depth, junction_absorber_depth, node_distance
from rfx.model.graded_grid import interior_cells
from rfx.sparams._grid_metrics import _msl_axis_spacing, _msl_axis_cell_f64


def _warn_msl_wave_split_unreliable(
    reliable: np.ndarray, freqs: object
) -> None:
    """Emit one aggregate warning for unreliable MSL frequency bins."""
    affected_freqs = np.flatnonzero(np.any(~reliable, axis=0))
    if not affected_freqs.size:
        return

    import warnings

    freqs_arr = np.asarray(freqs)
    f1 = freqs_arr[int(affected_freqs[0])] / 1e9
    f2 = freqs_arr[int(affected_freqs[-1])] / 1e9
    warnings.warn(
        "low signal at an MSL port plane: " f"{affected_freqs.size} bins in [{f1:.4f}, {f2:.4f}] GHz "
        "have both |V| and |I| below 10% of their record's band medians. "
        "This flags relative signal strength, not proof of an incorrect "
        "S-matrix; a true transmission zero can also trigger it. Check "
        "signal uncertainty, settling, drive conditioning and probe "
        "geometry before using these bins.", AccuracyWarning,
        stacklevel=2,
    )


def _warn_msl_beta_scan_railed(
    beta_railed: np.ndarray, freqs: object, port_names: tuple
) -> None:
    """Emit one aggregate warning for β-scan rail-pinned bins.

    ``beta_railed`` is (n_ports, n_freqs) bool, True where a port's
    own-drive N-probe β scan failed to bracket its optimum — the fitted
    ``beta``/``Z0`` at that bin are the ±35% scan-window limit, not a
    measurement.  Mirrors ``_warn_msl_wave_split_unreliable``: silent
    while tracing (caller concretizes), one warning per extraction.
    """
    railed = np.asarray(beta_railed, dtype=bool)
    affected_freqs = np.flatnonzero(np.any(railed, axis=0))
    if not affected_freqs.size:
        return

    import warnings

    freqs_arr = np.asarray(freqs)
    f1 = freqs_arr[int(affected_freqs[0])] / 1e9
    f2 = freqs_arr[int(affected_freqs[-1])] / 1e9
    ports = ", ".join(
        repr(port_names[p]) for p in np.flatnonzero(np.any(railed, axis=1))
    )
    per_port = "; ".join(
        f"{port_names[p]!r}: "
        + ", ".join(f"{freqs_arr[k] / 1e9:.4f}" for k in np.flatnonzero(railed[p]))
        + " GHz"
        for p in np.flatnonzero(np.any(railed, axis=1))
    )
    warnings.warn(
        "N-probe beta scan pinned at its own window limit: "
        f"{affected_freqs.size} bins in [{f1:.4f}, {f2:.4f}] GHz at "
        f"port(s) {ports} minimized the fit residual at the edge of the "
        "±35% scan around the analytic Hammerstad-Jensen guess — the "
        "reported Z0/beta at those bins are the scan limit, NOT a "
        f"measurement. Railed bins per port: {per_port}. "
        "Fitted Z0/beta are not used in S11/S21, "
        "which use measured V/I and the analytic Z0 anchor; this does not "
        "certify those inputs. S is shifted from probe 0 to the declared "
        "feed plane with the closed-form beta (Hammerstad-Jensen with "
        "Kirschning-Jansen dispersion), not the fitted one. Common causes: "
        "a record too short or a probe ladder too coarse for the fit "
        "(check settling_db and the reliable mask), or the real eps_eff "
        "is far from the HJ estimate (wrong eps_r_sub / substrate not "
        "detected under the port) -- then the closed-form beta and the Z0 "
        "anchor of S are wrong as well. Check the result's beta_railed mask "
        "before quoting Z0 or beta.", AccuracyWarning,
        stacklevel=2,
    )


_SETTLING_WITNESS_DB = -40.0


def settling_verdict(settling_db) -> str:
    """The ONE -40 dB ring-down comparison in this codebase.

    Returns ``"pass"``, ``"fail"`` or ``"absent"``. ``"absent"`` covers
    ``None`` and every non-finite value (NaN, +/-inf) -- the states that mean
    "no witness was established", which a bare ``> -40`` comparison silently
    turns into a pass. That is not hypothetical: a campaign driver read a
    missing witness as ``np.nan`` through ``getattr(result, "settling_db",
    np.nan)``, compared it to the bar, and a completed 3-hour two-arm
    experiment came back INCONCLUSIVE for an instrument reason.

    Every caller that needs the decision calls this instead of writing the
    comparison again -- including the aggregate warner below and the
    ``run()`` post-run advisories in ``rfx/api/_execute.py``. A value exactly
    at the bar passes (the bar is the inclusive limit, unchanged).
    """
    if settling_db is None:
        return "absent"
    try:
        value = float(settling_db)
    except (TypeError, ValueError):
        return "absent"
    if not np.isfinite(value):
        return "absent"
    return "fail" if value > _SETTLING_WITNESS_DB else "pass"


def _validate_extra_flux_monitor_entries(entries, domain, fn_name):
    """Light re-validation of ``extra_flux_monitors=`` entries.

    Entries are the objects ``Simulation.add_flux_monitor`` registers,
    typically built on a scratch ``Simulation`` sharing this sim's domain —
    ``add_flux_monitor`` already validated them against THAT domain, so this
    only re-checks the one property that can silently diverge between the
    two sims (the normal-axis coordinate inside THIS domain). Energy-witness
    channel only: spectra come back on ``result.flux_monitors``; nothing
    here feeds the S-parameter math (the registered-monitor guard below is
    unchanged — this extractor still builds its own DFT planes).
    """
    if not entries:
        return
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    seen = set()
    for pe in entries:
        for attr in ("name", "axis", "coordinate"):
            if not hasattr(pe, attr):
                raise TypeError(
                    f"{fn_name}(): extra_flux_monitors entries must be the "
                    "objects Simulation.add_flux_monitor() registers "
                    f"(missing attribute {attr!r}); build them by calling "
                    "add_flux_monitor() on a scratch Simulation with the "
                    "same domain and passing its ._flux_monitors list."
                )
        ax = axis_to_index.get(pe.axis)
        if ax is None:
            raise ValueError(
                f"{fn_name}(): extra flux monitor {pe.name!r} has invalid "
                f"axis {pe.axis!r}."
            )
        if not (0.0 <= float(pe.coordinate) <= float(domain[ax])):
            raise ValueError(
                f"{fn_name}(): extra flux monitor {pe.name!r} coordinate "
                f"{pe.coordinate} m is outside this simulation's "
                f"{pe.axis}-domain [0, {domain[ax]}] m."
            )
        if pe.name in seen:
            raise ValueError(
                f"{fn_name}(): duplicate extra flux monitor name "
                f"{pe.name!r} — result spectra are name-keyed."
            )
        seen.add(pe.name)


def _nu_shift_span_cells(grid, cfg, desired_plane_m: float):
    """Cell sizes a waveguide port's reference-plane span crosses on a NU grid.

    The span is the hull of three planes along the port-normal axis: the
    user-facing port plane (``cfg.source_x_m``), the plane the modal V/I are
    recorded on (``cfg.reference_x_m``, ``ref_offset`` cells downstream) and
    the requested reference plane. ``_shift_modal_waves`` moves the recorded
    waves to the requested plane with ONE ``exp(-/+ j*beta*shift)`` whose beta
    is evaluated at a single cell size -- ``cfg.dx``, which on this lane is
    the grid's BOUNDARY cell (``NonUniformGrid.dx``) -- so the shift is exact
    only when every cell the span crosses has one size. Cell sizes are read
    from the grid's per-cell arrays (``dx_arr`` / ``dy_arr`` / ``dz``, exact
    float64 spine when present); a ``NonUniformGrid`` carries no
    ``*_profile`` attribute.

    Returns ``(axis, span_lo_m, span_hi_m, cell_sizes_m)``: ``cell_sizes_m``
    is the ordered float64 array of interior cell sizes the span crosses
    (empty when the three planes coincide). A cell counts as crossed when the
    span overlaps it by more than 1e-3 of the axis's smallest interior cell,
    so a plane sitting on a node -- up to the float32 quantisation of the
    recorded plane positions -- does not drag in its neighbour.
    """
    axis = str(cfg.normal_axis)
    d_full = host_cells(grid, axis)
    pad_lo, pad_hi = int(getattr(grid, f"pad_{axis}_lo")), int(getattr(grid, f"pad_{axis}_hi"))
    cells = np.asarray(interior_cells(np.asarray(d_full, dtype=np.float64), pad_lo, pad_hi, periodic=axis in grid.periodic_axes),
                       dtype=np.float64)
    planes = (float(cfg.source_x_m), float(cfg.reference_x_m), float(desired_plane_m))
    lo, hi = min(planes), max(planes)
    return axis, lo, hi, cells_crossed(cells, lo, hi)


def _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired_plane_m: float,
                                              port_name: str):
    """Refuse a NU reference-plane shift whose span crosses more than one cell size.

    The NU lane evaluates beta for the reference-plane shift at the boundary
    cell (``cfg.dx``), not the cell the plane sits in. Inside one uniform
    grading zone that is a documented second-order envelope
    (``(beta*dx)^2/24``; ``docs/guides/support_matrix.md``, nonuniform
    waveguide row, and ``tests/fixtures/waveguide_nu_beta_cell_size_envelope.json``).
    Across a graded span a single beta is right only by accident, and
    integrating beta cell by cell over the span is not implemented -- the case fails
    loudly here instead of silently applying one beta. Returns the
    ``_nu_shift_span_cells`` tuple when the span is admissible.
    """
    axis, lo, hi, sizes = _nu_shift_span_cells(grid, cfg, desired_plane_m)
    if sizes.size == 0:
        return axis, lo, hi, sizes
    distinct = distinct_cell_sizes(sizes)
    if len(distinct) > 1:
        raise ValueError(
            f"waveguide port {port_name!r}: the span from its port plane to its "
            f"reference plane along {axis} ({lo * 1e3:.4f} mm to {hi * 1e3:.4f} mm; "
            f"port plane {float(cfg.source_x_m) * 1e3:.4f} mm, modal record plane "
            f"{float(cfg.reference_x_m) * 1e3:.4f} mm, reference plane "
            f"{float(desired_plane_m) * 1e3:.4f} mm) crosses cells of "
            f"{len(distinct)} sizes: {[round(d * 1e3, 6) for d in distinct]} mm "
            f"(cells crossed in order: {[round(float(v) * 1e3, 6) for v in sizes]} mm). "
            "The reference-plane shift applies one exp(-/+ j*beta*shift) with beta "
            f"evaluated at the grid's boundary cell ({float(cfg.dx) * 1e3:.4f} mm), "
            "which is exact only inside one uniform grading zone; integrating beta "
            "over a graded span is not implemented on the nonuniform lane. Move the "
            "port and its reference plane into one uniform zone of the profile, or "
            "keep the graded block away from them."
        )
    return axis, lo, hi, sizes


def _resolve_msl_auto_offsets(sim, entries, grid):
    """solve the probe-offset interval for AUTO-offset ports.

    When the interval comes out EMPTY this is the third site that speaks
    about probe-clearance corruption, so it carries the same guidance
    text as ``compute_msl_s_matrix``'s Z0 guard and preflight's layout
    warning. Imported, not copied.

    ``add_msl_port``'s auto-default is the UPSTREAM-only lower edge
    (``offset_min = max(3, λ/(4π·dx), 5·h_sub/dx)``); the downstream
    constraint — the deepest probe ≥ λ_g/4 (at f_max) clear of the nearest
    reflector — needs the full registered geometry, which only exists at
    driver time. Per auto port on a uniform grid:

    * no downstream reflector  -> keep ``offset_min``;
    * finite reflector, interval ``[offset_min, offset_max]`` non-empty ->
      midpoint (on the Sheen measurement the old default sat at the
      contaminated near edge and the old advisory pushed PAST the far
      edge; the midpoint lands inside the measured-clean window);
    * interval EMPTY -> warn loudly (the feed line is too short for a
      clean N-probe measurement) and keep ``offset_min`` (upstream
      priority — the fringing transient is the historically dominant
      corruption).

    Explicit offsets are never touched. Automatic values are first re-derived
    from the full declaration into the bookkeeping dictionaries; repeated calls
    are idempotent. Returns a new entries list, leaving raw entries unchanged.

    Auto probe SPACING. A ~0.1·λ_g probe span leaves the
    N-probe β fit noise-fragile: the two model columns ``e^{∓jβx}`` are
    nearly collinear and the residual-vs-β curve is nearly flat, so probe
    noise walks the fitted β far from truth (measured, 500-trial
    Monte-Carlo at 1% probe noise, N=5: median β error 5.5% at a
    0.10 λ_g span vs 0.81% at 0.30 λ_g — a 6.8× degradation). For ports
    whose ``n_probe_spacing`` was auto (``sim._msl_auto_probe_spacing``,
    keyed like the offset bookkeeping) this solve therefore WIDENS the
    spacing from the conservative registration default toward

        ``spacing = λ_g(f_max)/4``  (λ_g from the registration HJ ε_eff),

    i.e. a total span of ``(N−1)·λ_g(f_max)/4`` (one full λ_g at f_max
    for the default N=5 — two periods of the λ_g/2 standing-wave
    pattern), capped by the SAME geometry this solve already knows:

    * spacing ≤ λ_g(f_max)/4 also keeps every probe pair far from the
      ``β ↔ 2π/s − β`` sampling alias (which enters the ±35% scan window
      only for s ≳ 0.42·λ_g);
    * with a downstream reflector, the span may consume at most HALF the
      compliant interval ``[offset_min, offset_max]`` — the other half
      stays with the midpoint rule, preserving the measured-clean
      upstream margin for probe 0 (the Sheen measurement showed the
      near edge contaminated);
    * the deepest probe stays ``λ_g/4``-clear of the absorbing boundary
      (an absorber face is a discontinuity like any reflector, and CPML
      cells are non-physical), i.e. inside
      ``domain_edge − clear − n_cpml·dx``.

    On a feed too short for any widening the caps floor at the hard
    minimum of 2 cells and the existing
    empty-interval warning still fires when even that does not fit.
    Explicit spacings are never touched. Graded/unevaluable propagation
    axes get no widening, via the same skip-and-warn path as the offset
    solve.

    Non-uniform meshes. The bail-out is per PORT and keys on
    that port's PROPAGATION axis, not on "is any axis graded". A
    cell-counted probe interval is ill-defined when the axis the probes
    march along has varying cell sizes — which is a statement about the
    propagation axis alone. A ``dz_profile`` does not grade dx, and a
    microstrip propagates along x or y, so the previous
    ``getattr(grid, "dz", None) is not None`` gate silently disabled the
    solve on exactly the boundary-fitted z-stackup meshes where it is
    both well defined and wanted. When the propagation axis IS graded (or
    is a traced profile the host cannot inspect) the stored value is kept
    as before — but the skip now WARNS once per call instead of being
    silent.

    Graded propagation axis. The automatic offset and spacing
    are LENGTHS (``lambda_eff/(4*pi)`` and ``5*h_sub`` for the offset,
    ``lambda_eff/8`` for the ladder span) that ``add_msl_port`` counted in
    the scalar cell. On a runway finer than that cell the count put probe 0
    at a fraction of the fringing length from the source (5 cells = 635 um
    of the 1.270 mm a 254 um board needs, on a 127 um runway under a 254 um
    boundary cell). For an automatic port on a graded axis the lengths are
    counted in the runway cell beside the node the source is stamped on,
    when the span from that node through the deepest probe lies in one
    uniform zone. When it would cross a grading ramp, no cell count names
    one length along it, and the stored counts are kept. Either way the interval solve and the widening, which count
    reflector and
    absorber distances in cells, do not run on a graded axis, and the skip
    warning says so. On an ungraded axis the same lengths are recounted in
    that axis's own cell before the interval solve starts from them, which
    matters where that cell is not the scalar: a uniform ``dy_profile`` finer
    than ``dx``, or an auto mesh that chose ``dx`` after the port.
    """
    from rfx.preflight._msl_probe_rules import admitted_msl_entries
    entries, _auto_spacing, _auto_lengths = admitted_msl_entries(sim, entries)
    if not sim._msl_auto_offset_min and not _auto_spacing:
        return entries

    import dataclasses
    from rfx.preflight.msl_codes import msl_text, msl_join, placement_warning
    import warnings

    from rfx.preflight._msl_probe_rules import (
        msl_min_probe_clearance,
    )
    from rfx.preflight.msl_reflector import msl_nearest_downstream_reflector
    from rfx.preflight._common import _fmt_len, local_cell
    from rfx.preflight.msl import (
        msl_auto_probe_ladder,
        msl_auto_probe_offset_cells,
        msl_auto_probe_spacing_cells,
    )
    from rfx.sources.msl_port import (
        _MSL_AXIS_INDEX as _MSL_AX,
        msl_axis_roles as _msl_axis_roles,
    )

    clear = msl_min_probe_clearance(float(sim._freq_max))
    resolved: list = []
    _graded_skips: list[str] = []
    for pe in entries:
        off_min = sim._msl_auto_offset_min.get(pe.name)
        sp_eps_eff = _auto_spacing.get(pe.name)
        if off_min is None and sp_eps_eff is None:
            resolved.append(pe)
            continue
        # project position/domain onto this port's propagation
        # and width axes before handing them to the x-frame helper.
        _prop_ax, _width_ax, _, _dir_sign = _msl_axis_roles(pe.direction)
        _ip = _MSL_AX[_prop_ax]
        _iw = _MSL_AX[_width_ax]
        # Grading gates propagation; tracing also gates width tolerances.
        dx_u, _graded, _evaluable = _msl_axis_spacing(grid, _ip)
        if not _evaluable or (not _graded and is_tracer(grid.cells(_iw))):
            _graded_skips.append(
                msl_text(
                    "placement_traced",
                    port_name=pe.name,
                    direction=pe.direction,
                    axis=_prop_ax if not _evaluable else _width_ax,
                )
            )
            resolved.append(pe)
            continue
        if _graded:
            # count the automatic LENGTHS in the runway cell at
            # the source node, where the whole ladder lies in one zone.
            _lengths = _auto_lengths.get(pe.name)
            if _lengths is None:
                _graded_skips.append(
                    msl_text(
                        "placement_missing_lengths",
                        port_name=pe.name,
                        direction=pe.direction,
                        propagation_axis=_prop_ax,
                    )
                )
                resolved.append(pe)
                continue
            from rfx.model.graded_grid import interior_cells as _interior_cells
            _pads = ((grid.pad_x_lo, grid.pad_x_hi),
                     (grid.pad_y_lo, grid.pad_y_hi),
                     (grid.pad_z_lo, grid.pad_z_hi))[_ip]
            _runway = np.asarray(
                _interior_cells(grid.cells(_ip), *_pads, periodic="xyz"[_ip] in grid.periodic_axes), dtype=np.float64)
            _off, _sp, _cell, _one_zone = msl_auto_probe_ladder(
                _runway, float(_runway[0]), float(pe.position[_ip]),
                _dir_sign, int(pe.n_probes), _lengths[0], float(pe.height),
                _lengths[1],
                n_probe_offset=(None if off_min is not None
                                else pe.n_probe_offset),
                n_probe_spacing=(None if sp_eps_eff is not None
                                 else pe.n_probe_spacing))
            _ladder_txt = msl_text(
                "placement_ladder",
                runway_offset_cells=_off,
                probe_intervals=int(pe.n_probes) - 1,
                runway_spacing_cells=_sp,
                runway_cell_m=_cell,
            )
            if not _one_zone:
                _graded_skips.append(
                    msl_text(
                        "placement_crosses_ramp",
                        port_name=pe.name,
                        direction=pe.direction,
                        propagation_axis=_prop_ax,
                        ladder=_ladder_txt,
                        stored_offset_cells=int(pe.n_probe_offset),
                        stored_spacing_cells=int(pe.n_probe_spacing),
                    )
                )
                resolved.append(pe)
                continue
            _fields = {}
            if off_min is not None and _off != int(pe.n_probe_offset):
                _fields["n_probe_offset"] = _off
            if sp_eps_eff is not None and _sp != int(pe.n_probe_spacing):
                _fields["n_probe_spacing"] = _sp
            _graded_skips.append(
                msl_text(
                    "placement_uniform_zone",
                    port_name=pe.name,
                    direction=pe.direction,
                    propagation_axis=_prop_ax,
                    ladder=_ladder_txt,
                    first_probe_distance_m=_off * _cell,
                )
            )
            resolved.append(
                dataclasses.replace(pe, **_fields) if _fields else pe)
            continue
        # A uniform propagation axis can still have a cell other than the one
        # add_msl_port counted in: a uniform dy_profile finer than dx, or an
        # auto mesh that settled on another dx after the port was registered.
        # The stored lengths are recounted in this axis's own cell, so the
        # interval solve below starts from the right lower edge and spacing.
        # Where the two cells are equal the counts are the stored ones.
        _lengths = _auto_lengths.get(pe.name)
        if _lengths is not None:
            _cell_u = _msl_axis_cell_f64(grid, _ip)
            _recount = {}
            if off_min is not None:
                off_min = msl_auto_probe_offset_cells(
                    _lengths[0], float(pe.height), _cell_u)
                if off_min != int(pe.n_probe_offset):
                    _recount["n_probe_offset"] = off_min
            if sp_eps_eff is not None:
                _sp_reg = msl_auto_probe_spacing_cells(
                    _lengths[1], int(pe.n_probes), _cell_u)
                if _sp_reg != int(pe.n_probe_spacing):
                    _recount["n_probe_spacing"] = _sp_reg
            if _recount:
                pe = dataclasses.replace(pe, **_recount)
        d_refl, _, _unevaluated = msl_nearest_downstream_reflector(
            getattr(sim, "_geometry", []),
            x_probe=float(pe.position[_ip]),
            x_feed=float(pe.position[_ip]),
            y_feed=float(pe.position[_iw]),
            w_trace=float(pe.width),
            dx=local_cell(grid, _ip, float(pe.position[_ip])),
            width_cell=min(local_cell(grid, _iw, float(pe.position[_iw]) - float(pe.width) / 2),
                           local_cell(grid, _iw, float(pe.position[_iw]) + float(pe.width) / 2)),
            domain_y=float(sim._domain[_iw]),
            direction=pe.direction, port_position=pe.position,
            # same conductor rule as the assembler, and thin
            # conductors included, so the solved offset is not derived
            # from a scan that was blind to most of the metal.
            resolve_material=sim._resolve_material,
            thin_conductors=getattr(sim, "_thin_conductors", ()),
            pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD,
        )
        if _unevaluated:
            _graded_skips.append(
                msl_text(
                    "placement_uninspected",
                    port_name=pe.name,
                    direction=pe.direction,
                    unevaluated_conductor_count=len(_unevaluated),
                )
                + "; ".join(_unevaluated)
                + msl_text("placement_uninspected_remedy")
            )
            resolved.append(pe)
            continue
        # --- Auto probe-spacing widening.
        # Target λ_g(f_max)/4 per probe step, capped by half the reflector
        # interval and by the absorber clearance; floors at the hard
        # 2-cell minimum. Explicit spacings (sp_eps_eff is None) pass
        # through untouched.
        spacing = int(pe.n_probe_spacing)
        off_base = int(off_min if off_min is not None else pe.n_probe_offset)
        if sp_eps_eff is not None:
            from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
            _c0 = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
            lam_g_fmax = _c0 / (
                float(sim._freq_max) * float(sp_eps_eff) ** 0.5
            )
            _target = max(2, int(round(0.25 * lam_g_fmax / dx_u)))
            _x_feed = float(pe.position[_ip])
            _dist_edge = (
                float(sim._domain[_ip]) - _x_feed
                if _dir_sign > 0 else _x_feed
            )
            _n_cpml = int(getattr(grid, "cpml_layers", 0) or 0)
            _b_dom = int((_dist_edge - clear) / dx_u) - _n_cpml - off_base
            if np.isfinite(d_refl):
                _b_refl = int((d_refl - clear) / dx_u) - off_base
                _span_budget = min(_b_refl // 2, _b_dom)
            else:
                _span_budget = _b_dom
            spacing = max(
                2,
                min(_target, _span_budget // (int(pe.n_probes) - 1)),
            )
        span = (int(pe.n_probes) - 1) * spacing
        _fields: dict = {}
        if spacing != int(pe.n_probe_spacing):
            _fields["n_probe_spacing"] = spacing

        if off_min is None or not np.isfinite(d_refl):
            # Explicit offset, or no downstream reflector: no offset
            # midpoint to solve — carry only the spacing resolution.
            resolved.append(
                dataclasses.replace(pe, **_fields) if _fields else pe
            )
            continue
        off_max = int((d_refl - clear) / dx_u) - span
        if off_max >= off_min:
            _fields["n_probe_offset"] = (off_min + off_max) // 2
            resolved.append(dataclasses.replace(pe, **_fields))
        else:
            from rfx.preflight.msl import (
                MSL_PROBE_CLEARANCE_EFFECT as _MSL_PROBE_CLEARANCE_EFFECT,
                MSL_PROBE_CLEARANCE_GUIDANCE as _MSL_PROBE_CLEARANCE_GUIDANCE,
            )
            warnings.warn(
                placement_warning(
                    msl_text(
                        "placement_empty_interval",
                        port_name=pe.name,
                        offset_min_cells=off_min,
                        offset_max_cells=off_max,
                        probe_count=int(pe.n_probes),
                        recommended_reflector_gap_m=clear,
                        feed_reflector_gap_m=d_refl,
                    )
                    + _MSL_PROBE_CLEARANCE_EFFECT
                    + " "
                    + _MSL_PROBE_CLEARANCE_GUIDANCE
                ),
                stacklevel=user_stacklevel(),
            )
            resolved.append(
                dataclasses.replace(pe, **_fields) if _fields else pe
            )
    if _graded_skips:
        warnings.warn(
            placement_warning(
                msl_text(
                    "placement_graded_skipped",
                    port_count=len(_graded_skips),
                    details=msl_join(
                        "; ",
                        _graded_skips,
                        indexed=True,
                    ),
                )
            ),
            stacklevel=user_stacklevel(),
        )
    return resolved


def _sigma_max_excess(S):
    """Per-frequency ``max(sigma_max(S(f)) - 1, 0)`` of an assembled S.

    The same quantity :func:`_project_passive` returns as its ``correction``,
    measured WITHOUT projecting — what a lane needs when it ships the raw
    extraction and still has to say how non-physical it is. Returns a
    ``(n_freqs,)`` host array in the input's real dtype; a bin whose matrix
    is non-finite stays NaN so the caller's finiteness audit still sees it.

    Concrete arrays only. A tracer has no value to take singular values of,
    so every caller gates on :func:`rfx.core.jax_utils.is_tracer` first.
    """
    s_t = np.asarray(S).transpose(2, 0, 1)  # (n_freqs, n_ports, n_ports)
    real_dtype = s_t.real.dtype
    work_dtype = np.complex128 if np.iscomplexobj(s_t) else np.float64
    finite = np.all(np.isfinite(s_t), axis=(1, 2))
    excess = np.full(s_t.shape[0], np.nan, dtype=real_dtype)
    if np.any(finite):
        sig = np.linalg.svd(s_t[finite].astype(work_dtype), compute_uv=False)
        excess[finite] = np.maximum(sig[:, 0] - 1.0, 0.0)
    return excess


def _warn_if_sigma_max_excess(
    excess, freqs, *, extractor: str, envelope: float = 0.05,
    projection_requested: bool = False,
) -> None:
    """One aggregate warning naming a RETURNED S that is not passive.

    The companion of :func:`_warn_if_passivity_projected`: that one reports
    what a projection removed, this one reports what was left in place. It
    fires on the lane that ships the raw extraction, so the message has to
    say three things the caller cannot see from ``S`` alone — how many bins
    exceed the bound and by how much, that ``S`` is the extraction itself
    with nothing clipped, and how to get the projected matrix instead.

    The cause sentence is the same one
    :func:`_warn_if_passivity_projected` gives and no stronger: a passive
    structure cannot scatter more power than it receives, so a raw
    ``sigma_max > 1`` is a measurement artifact — ``reliable`` and
    ``settling_db`` are the fields that say which one.
    """
    exc = np.asarray(excess)
    finite = np.isfinite(exc)
    # Same margin _project_passive clips to: a lossless thru reads
    # sigma_max = 1 + O(eps) in float32, and a warning that fires on
    # rounding teaches the caller to ignore it. The field keeps the raw
    # value; only the warning has a floor.
    floor = 64.0 * float(np.finfo(exc.dtype if exc.dtype.kind == "f"
                                  else np.float32).eps)
    if not np.any(exc[finite] > floor):
        return

    import warnings

    f = np.asarray(freqs)
    n_touched = int((exc[finite] > floor).sum())
    n_big = int((exc[finite] > envelope).sum())
    k = int(np.nanargmax(np.where(finite, exc, -np.inf)))
    warnings.warn(
        f"{extractor}: the returned S is not passive at {n_touched} of "
        f"{exc.size} frequency bins, worst sigma_max = {1.0 + exc[k]:.3f} at "
        f"{f[k] / 1e9:.3f} GHz. S is returned exactly as extracted — nothing "
        f"is clipped, and the per-bin amounts are in sigma_max_excess. "
        + (
            f"{n_big} bins exceed the {1.0 + envelope:.2f} extraction "
            f"envelope. "
            if n_big
            else ""
        )
        + "A passive structure cannot scatter more power than it receives, "
        "so these bins are a measurement artifact rather than physics — the "
        "usual causes are a record that ended before the structure rang "
        "down and a mesh too coarse for the geometry (see settling_db and "
        "reliable for which). "
        + (
            "enforce_passivity=True was passed, and the projection is skipped "
            "on the eps_override channel so that the measured and the "
            "differentiated S stay the same function."
            if projection_requested
            else "Pass enforce_passivity=True to get S projected onto the "
            "passive set instead; that projection is skipped on the "
            "eps_override channel, so the measured and the differentiated S "
            "stay the same function."
        ), AccuracyWarning,
        stacklevel=2,
    )


def _warn_if_passivity_projected(
    correction, freqs, *, envelope: float = 0.05
) -> None:
    """One aggregate warning stating exactly what the projection removed."""
    corr = np.asarray(correction)
    finite = np.isfinite(corr)
    if not np.any(corr[finite] > 0.0):
        return

    import warnings

    f = np.asarray(freqs)
    n_touched = int((corr[finite] > 0.0).sum())
    n_big = int((corr[finite] > envelope).sum())
    # nanargmax: a NaN raw bin would otherwise be selected and the message
    # would read "worst sigma_max = nan". NaN bins stay NaN in S (the
    # finiteness self-check flags them); the bound claim applies to finite bins.
    k = int(np.nanargmax(np.where(finite, corr, -np.inf)))
    warnings.warn(
        f"S-matrix projected onto the passive set (singular values clipped "
        f"to 1): {n_touched} of {corr.size} frequency bins were non-passive "
        f"as extracted, worst sigma_max = {1.0 + corr[k]:.3f} at "
        f"{f[k] / 1e9:.3f} GHz. "
        + (
            f"{n_big} bins exceeded the {1.0 + envelope:.2f} extraction "
            f"envelope — at those bins the RAW value is a measurement "
            f"artifact (see reliable / settling_db for the cause) and the "
            f"projected value inherits that uncertainty; do not quote them "
            f"as physics. "
            if n_big
            else ""
        )
        + "Raw values are preserved in S_raw; corrections per bin in "
        "passivity_correction.", AccuracyWarning,
        stacklevel=2,
    )


WAVEGUIDE_RECIPROCITY_ADVISORY_TOL = 0.011
"""Warn-only complex-reciprocity tolerance for the rectangular-waveguide
S-matrix extractor, in the dimensionless per-bin measure

    dev = max_f  max_{i<j} |S_ij(f) - S_ji(f)| / max_{p,q} |S_pq(f)|

DERIVED, not chosen: ``gate_from_envelope(6.9831664765629175e-3,
quantum=1000)`` = ``ceil(6.9831664765629175e-3 * 1.5 * 1000) / 1000`` =
``0.011``, with the repo-wide ``ENVELOPE_GATE_MULTIPLIER = 1.5`` of
``tests/_gate_policy.py``. ``quantum=1000`` (three decimals) is the coarsest
quantum that still resolves an envelope of order 7e-3; ``quantum=100`` rounds
to 0.02, which is 2.9x the measured worst case. The derivation is re-run
against the fixture, from outside this file, by
``tests/unit/sparams/test_waveguide_reciprocity_advisory.py::
test_advisory_tolerance_is_derived_from_the_measured_chain_battery_envelope``.

MEASURED ENVELOPE. Artifact:
``tests/fixtures/waveguide_chain_battery/fixture.json`` -- the single
pre-declared WR-90 chain-battery run. Per-cell numbers sit at
``physics_gates["<dut>|<rung>|<lane>"]["reciprocity_complex_max"]`` for the
pec_short and slab DUTs, and at ``cells[*]["reciprocity_complex_max"]`` for
all three DUTs (the thru control has no ``physics_gates`` row). The battery's
own measure is ``cell_metrics`` in
``tests/_waveguide_chain_battery_gates.py``, which is the formula above --
that is why this tolerance is expressed in it and not in an absolute
|S_ij - S_ji|.

Cells INCLUDED -- the claims rung (``physics_gates["claims_rung"] ==
"fine"``), both normalize lanes, both DUTs that actually transmit:

    slab | fine | normalize=False    6.9831664765629175e-3   <- the envelope
    slab | fine | normalize='flux'   3.2771666678166415e-4
    thru | fine | normalize=False    4.3316503043657606e-4
    thru | fine | normalize='flux'   3.2770621911991577e-4

All four settle well below -40 dB (``settling_db`` between -97.96 and
-100.43 dB over the eight drives), so none of the four is a record-truncation
artifact.

Cells EXCLUDED, and why:

  * ``pec_short``, every rung and lane. A full-height PEC short transmits
    nothing: ``S21`` and ``S12`` are zero to float32 denormal noise (worst
    ``max|S21|`` across its six cells is 3.4e-20), so its reciprocity
    deviation is 0 by construction and witnesses nothing. Its settling
    number is degenerate for the same reason -- the port-2 time records are
    identically zero, which is what puts ``+0.0 dB`` in its ``settling_db``
    and what sets the fixture's ``settling_all_below_minus_40_db`` false.
    A vacuous zero transmission cannot witness reciprocity, just as the
    empty-guide reflection identity cannot witness reflection accuracy.
  * the ``coarse`` and ``mid`` rungs. They are the mesh ladder, and their
    reciprocity deviation -- up to 6.758813e-2 at slab|coarse|normalize=False
    -- IS the discretization error this advisory exists to surface.
    Calibrating the tolerance from them would leave the advisory silent on
    exactly the under-resolved runs it is for. At 0.011 the advisory fires on
    slab|coarse|normalize=False (6.76e-2) and slab|mid|normalize=False
    (1.99e-2), and on nothing else among the fixture's 18 cells.

ONE tolerance covers both lanes. The two lanes' envelopes differ by ~21x
(6.98e-3 on normalize=False vs 3.28e-4 on normalize='flux'), so 0.011 is
loose for the flux lane. A per-lane split is deliberately NOT done here: it
would double the derivation surface for an advisory, and the conservative
single value cannot fire on a run the committed gate accepts.

This is NOT the chain battery's gate. That gate is
``RECIPROCITY_COMPLEX_MAX = 0.01`` in
``tests/_waveguide_chain_battery_gates.py``, pre-declared before the
measurement, hard, and untouched here. 0.011 > 0.01 on purpose: a warn-only
advisory must never fire on a run the committed gate would pass.

PORT COUNT. Every cell in the envelope is a 2-port. The check itself is
port-count-agnostic (``S_ij = S_ji`` is), and it is left enabled for 3+ port
waveguide results -- a T-junction reference sim is exactly where an extraction
asymmetry is most likely -- but applying a 2-port-measured tolerance there is
an EXTRAPOLATION. It is an acceptable one only because nothing gates on this
number: the advisory is informational on every path.

SCOPE. The waveguide extractor only. No other port family has a measured
complex-reciprocity envelope -- ``compute_mixed_s_matrix`` in particular
carries a documented ~9% reciprocity residual on its own experimental lane --
so the check stays off everywhere else.
"""


def _reciprocity_advisory_message(s_np, f_np, port_names, *, extractor, tol):
    """WARN-ONLY reciprocity advisory text, or ``None`` when there is nothing
    to say (fewer than two ports, non-finite data, or a deviation within
    ``tol``).

    This function NEVER raises and NEVER touches the returned S-parameters. A
    user whose structure is genuinely non-reciprocal -- magnetised ferrite, an
    active device -- must get a message, not a broken run.

    Detection is delegated to :func:`rfx.validation.validate_port_smatrix`
    rather than re-implemented. That validator compares ``|S - S^T|`` against
    ``atol + rtol * max(1, |S|, |S^T|)``, i.e. an ABSOLUTE difference for a
    passive S, where the scale floor of 1 always wins. The envelope this
    tolerance is derived from is the per-bin RELATIVE deviation
    ``|S_ij - S_ji| / max_pq |S_pq|``. Handing the validator the per-bin
    ``max|S|``-normalized view ``S_hat`` makes the two identical: by
    construction ``max_pq |S_hat_pq(f)| = 1``, so the validator's scale is
    exactly 1 in every bin and, with ``atol=0`` and ``rtol=tol``, its trigger
    condition ``max |S_hat - S_hat^T| > tol`` is the battery's
    ``reciprocity_complex_max > tol``. ``check_passivity=False`` here on
    purpose: passivity is the separate guard in
    :func:`_warn_if_nonpassive_smatrix`, and it must see the UNNORMALIZED S.
    """
    s_np = np.asarray(s_np)
    if s_np.ndim != 3 or s_np.shape[0] < 2 or s_np.shape[2] < 1:
        return None
    if not np.all(np.isfinite(s_np)):
        # Non-finite data is the passivity/finiteness guard's business.
        return None

    per_bin_max = np.max(np.abs(s_np), axis=(0, 1))
    s_hat = s_np / np.maximum(per_bin_max, 1e-12)[None, None, :]

    from rfx.validation import validate_port_smatrix

    report = validate_port_smatrix(
        s_params=s_hat,
        freqs=np.asarray(f_np),
        port_names=tuple(port_names),
        source=f"{extractor} (per-bin |S|-normalized reciprocity view)",
        check_passivity=False,
        check_reciprocity=True,
        reciprocity_atol=0.0,
        reciprocity_rtol=float(tol),
        require_positive_freqs=False,
        require_strictly_increasing_freqs=False,
    )
    issue = next(
        (i for i in report.issues if i.code == "reciprocity_violation"), None)
    if issue is None:
        return None

    dev = float(report.metrics.get("max_reciprocity_abs_diff", float("nan")))
    i, j = (issue.port_indices or (0, 1))[:2]
    k = int(issue.frequency_index or 0)
    names = tuple(port_names)
    n_i = names[i] if i < len(names) else str(i)
    n_j = names[j] if j < len(names) else str(j)
    f_flat = np.asarray(f_np).reshape(-1)
    f_ghz = float(f_flat[k]) / 1e9 if f_flat.size > k else float("nan")
    return (
        f"{extractor}: reciprocity ADVISORY (warn-only -- nothing failed, the "
        f"S-parameters are returned unchanged): worst complex reciprocity "
        f"deviation max|S_ij - S_ji| / max|S| = {dev:.4g} exceeds "
        f"{float(tol):g}, at ports ({n_i}, {n_j}), frequency index {k} "
        f"({f_ghz:.4f} GHz). The tolerance is derived from the WR-90 "
        f"chain-battery envelope -- see WAVEGUIDE_RECIPROCITY_ADVISORY_TOL in "
        f"rfx/api/_sparams.py for the cells it comes from. rfx models eps, "
        f"sigma and mu_r as isotropic scalars, and a structure built only "
        f"from those is reciprocal by Lorentz reciprocity, so on this lane a "
        f"deviation this large is normally a discretization or extraction "
        f"artifact: refine dx (the measured ladder runs 6.8e-2 coarse -> "
        f"7.0e-3 fine on normalize=False) or use normalize='flux', which "
        f"measured ~21x tighter at the same mesh. If your structure genuinely "
        f"IS non-reciprocal (magnetised ferrite, an active device), this "
        f"advisory is expected -- it is informational and changes nothing."
    )


def _warn_if_nonpassive_smatrix(
    result,
    *,
    extractor: str,
    strict: bool = False,
    passivity_tol: float = 0.10,
    amplitude_eps: float = 0.05,
    check_reciprocity: bool = False,
    refused_positions=None,
    reciprocity_tol: float = WAVEGUIDE_RECIPROCITY_ADVISORY_TOL,
) -> None:
    """Auto-run the passivity/finiteness self-check on a freshly-extracted
    S-matrix and surface a non-physical result as a warning (or raise when
    ``strict``).

    This operationalizes the R5 "no surface-metric verdict" discipline:
    a passive structure cannot scatter more power than it
    receives, so a per-column power > 1 (e.g. ``|S11| > 1`` on a one-port)
    means the *extractor* is wrong — mismeasured current sign/scale or a
    bad reference plane — and the S-parameters are untrustworthy, NOT that
    the device is exotic. The wire, waveguide, coaxial, and MSL extractors
    all route through this shared guard before returning. Wiring the existing
    :func:`rfx.validation.validate_port_smatrix` in here is the guard that
    would have short-circuited the multi-session WR-90 ``|S11|`` chase
    recorded in durable memory.

    Tracer-safe: under ``jax.grad`` / ``jax.jit`` tracing ``result.s_params``
    is an abstract tracer with no concrete value, so the numpy-based check is
    skipped entirely. The diagnostic is for the eager forward call (the
    common research-tool usage); it deliberately does not fire per optimizer
    iteration.
    """
    s = getattr(result, "s_params", None)
    if s is None:
        # MSLSMatrixResult uses the historical ``S`` field name.
        s = getattr(result, "S", None)
    if s is None:
        return
    if is_tracer(s) or is_tracer(result.freqs):
        return
    s_np = np.asarray(s)
    f_np = np.asarray(result.freqs)

    if refused_positions is not None and np.any(refused_positions):
        _warn_partial_nonpassive(
            s_np, refused_positions, extractor, strict, passivity_tol, amplitude_eps)
        return

    # WARN-ONLY reciprocity advisory, enabled per port family by the caller
    # (today: the waveguide extractor, the only family with a measured
    # complex-reciprocity envelope). Emitted as its OWN warning, before the
    # passivity guard below, so that it can neither be swallowed by that
    # guard's early return nor promoted into its ``strict`` raise: a
    # reciprocity finding must stay advisory on every path.
    if check_reciprocity:
        _recip_msg = _reciprocity_advisory_message(
            s_np, f_np, tuple(result.port_names),
            extractor=extractor, tol=reciprocity_tol,
        )
        if _recip_msg:
            import warnings as _w
            _w.warn(_recip_msg, AccuracyWarning, stacklevel=3)

    from rfx.validation import validate_port_smatrix

    report = validate_port_smatrix(
        s_params=s_np,
        freqs=f_np,
        port_names=tuple(result.port_names),
        source=extractor,
        check_passivity=True,
        passivity_limit=1.0,
        passivity_tol=float(passivity_tol),
    )

    # Independent per-frequency amplitude advisory.  Keep this
    # separate from the column-power gate above: normalize=False deliberately
    # has a loose column-power tolerance, but an individual |S_ij| materially
    # above unity is still useful evidence of an extraction/normalization
    # artifact.  One worst-bin warning keeps a broadband result actionable
    # without producing one warning per frequency.
    # Tolerance-class-aware threshold: the loose normalize=False waveguide
    # path (passivity_tol >= 2.0) has a DOCUMENTED validated |S11| envelope
    # reaching ~1.41 (Yee dispersion + band edge — see the normalize docstring
    # and the policy locks in test_sparam_passivity_guard.py), so
    # its amplitude advisory only fires above 1.5 (still catches the eager
    # ~1.98 spike class). Tight paths (lumped/MSL/coax/normalize=True) keep
    # the 1 + amplitude_eps (default 1.05) threshold — the MSL documented
    # envelope is 1.0063 and the measured field case was 1.36.
    eff_amplitude_eps = (float(amplitude_eps) if float(passivity_tol) < 2.0
                         else max(float(amplitude_eps), 0.5))
    amplitude_advisory = None
    finite_abs = np.where(np.isfinite(s_np), np.abs(s_np), -np.inf)
    if finite_abs.size:
        worst_flat = int(np.argmax(finite_abs))
        worst_index = np.unravel_index(worst_flat, finite_abs.shape)
        worst_value = float(finite_abs[worst_index])
        frequency_index = int(worst_index[-1])
        if worst_value > 1.0 + eff_amplitude_eps:
            amplitude_advisory = (
                f"{extractor}: per-frequency amplitude advisory at frequency "
                f"index {frequency_index}: max |S| = {worst_value:.4g}; "
                "passivity violated: extraction/normalization artifact — do "
                "not interpret as physics; see the normalize parameter "
                "docstring"
            )
    bad = [
        i for i in report.issues
        if i.code in ("passivity_violation", "nonfinite_sparams")
    ]
    if not bad:
        # SOFT ADVISORY: the hard check above
        # uses the caller's ``passivity_tol``. On the ``normalize=False``
        # waveguide path that tol is loose (2.0 -> column-power limit 3.0,
        # i.e. |S| <= 1.732 for a 1-port) to tolerate the DOCUMENTED single-run
        # Yee/near-cutoff over-unity: a validated strong reflector (PEC short)
        # sits at column power ~2.0 there (|S11| entry ~1.03 plus the
        # convention |S21| ~ 1 that the single-run decomposition does not
        # cancel — see test_waveguide_broad_e5_live_anchor / the tol=2.0 lock
        # in test_sparam_passivity_guard::
        # test_normalize_aware_tol_tolerates_documented_overshoot). That leaves
        # the window (~2.0, 3.0] UNGUARDED: a passive result whose column power
        # is materially above the documented envelope but below the
        # extractor-broken hard limit returns silently. Surface it as a
        # SEPARATE, humble advisory (never raise) so a naive caller does not
        # trust an over-unity |S| from a reference-plane / normalize choice or
        # an under-resolved mesh. The floor 2.25 (|S| ~ 1.5 for a 1-port)
        # clears the ~2.0 documented envelope AND the committed normalize=False
        # PEC-short column power (~2.00) with margin; the window is EMPTY on
        # the tight-tol path (tol=0.10 -> hard limit 1.10 < 2.25), so this
        # advisory only fires for the loose normalize=False tol.
        _SOFT_COLPOWER_FLOOR = 2.25
        hard_limit = 1.0 + float(passivity_tol)
        max_cp = report.metrics.get("max_column_power")
        soft_advisory = None
        if max_cp is not None and _SOFT_COLPOWER_FLOOR < float(max_cp) <= hard_limit:
            soft_advisory = (
                f"max column power {float(max_cp):.4g} exceeds 1 "
                f"(non-physical for a passive structure) but stays below the "
                f"tol={float(passivity_tol):g} extractor-broken hard limit "
                f"({hard_limit:.4g}). This is an ADVISORY, not an error: on the "
                f"normalize=False single-run path a modest over-unity is a "
                f"documented Yee / near-cutoff artifact (validated envelope "
                f"~2.0), but a column power materially above it can also signal "
                f"a reference-plane or normalize choice or an under-resolved "
                f"mesh. Treat these S-parameters with caution and cross-check "
                f"(normalize='flux', finer dx) before trusting them."
            )
        advisories = [x for x in (amplitude_advisory, soft_advisory) if x]
        if advisories:
            import warnings as _w
            _w.warn(" ".join(advisories), AccuracyWarning, stacklevel=3)
        return
    detail = "; ".join(f"{i.code}: {i.message}" for i in bad)
    msg = (
        f"{extractor}: extracted S-matrix failed a passivity/finiteness "
        f"self-check — {detail}. A passive structure cannot have column "
        f"power > 1; this almost always means the extractor (current "
        f"sign/scale or reference plane) is wrong and the S-parameters are "
        f"UNRELIABLE. Inspect the V/I dump via "
        f"rfx.validation.validate_port_smatrix / replay_smatrix_from_vi_dump "
        f"before trusting or optimizing against these numbers."
    )
    if amplitude_advisory:
        msg = f"{msg} {amplitude_advisory}"
    if strict:
        raise ValueError(msg)
    import warnings as _w
    _w.warn(msg, AccuracyWarning, stacklevel=3)


def _finalize_sparam_result(
    result,
    *,
    extractor: str,
    strict: bool,
    passivity_tol: float = 0.10,
    check_reciprocity: bool = False,
):
    """Shared two-run-S-param epilogue: run the passivity/finiteness guard on a
    freshly-assembled S-matrix result, then return it unchanged.

    This is the one genuinely-common piece of the per-family two-run S-param
    flow at the orchestration layer (W6.4): both
    :meth:`_SparamMixin.compute_waveguide_s_matrix` (NU, multi-mode, and
    single-mode return paths) and the coax lanes in :mod:`rfx.sparams.coax`
    assemble a family-specific result and then invoke
    :func:`_warn_if_nonpassive_smatrix` immediately before returning. The
    per-port drive loop, vacuum-reference override, and rectangular-DFT
    windowing live behind the family-specific extractors (waveguide:
    ``rfx.sources.waveguide_port``; coax: the coax-line fixtures in
    :mod:`rfx.sparams.coax`) and are deliberately *not* unified here — they share no code at this
    layer, so a wider scaffold would be a leaky abstraction.

    ``passivity_tol`` defaults to the tight 0.10 bound (matching the coax call
    site). The waveguide path passes a ``normalize``-aware tolerance.

    ``check_reciprocity`` is opt-in per family and WARN-ONLY: it emits the
    advisory built by :func:`_reciprocity_advisory_message` and never raises,
    never changes the result. Only the waveguide extractor turns it on, because
    it is the only family with a measured complex-reciprocity envelope (see
    ``WAVEGUIDE_RECIPROCITY_ADVISORY_TOL``).
    """
    _warn_if_nonpassive_smatrix(
        result,
        extractor=extractor,
        strict=strict,
        passivity_tol=passivity_tol,
        check_reciprocity=check_reciprocity,
    )
    from rfx.sparams.passivity_warning import warn_if_singular_value_exceeds_one
    s = getattr(result, "s_params", None)
    if s is None:
        s = getattr(result, "S", None)
    warn_if_singular_value_exceeds_one(s, result.freqs, extractor=extractor)
    return result


_C0_SPARAMS = 299792458.0

def _warn_junction_probe_clearance(
    grid, cfgs, device_sigma, ref_sigmas, freqs, *,
    device_pec_edges=None, ref_pec_edges=None,
):
    """Advisory: probe-plane clearance from a junction (pure NumPy, no FDTD).

    For each driven port, the junction is where the device materials differ
    from that port's straight-guide reference. We reduce the ``sigma`` and
    componentwise realized PEC edge differences over the two transverse axes
    to a 1-D profile along the port normal axis, find the nearest difference
    to the port's probe plane, and compare that clearance to evanescent decay
    lengths of the next higher mode (TE20, cutoff ``fc2 = C0 / a``).
    ``alpha = 2*pi*sqrt(fc2^2 -
    f^2)/C0`` is evaluated at the band-CENTRE frequency — the validated
    far-port campaign sized its arms in mid-band decay lengths, and a band-max
    evaluation diverges (L -> inf) as the band edge approaches ``fc2``, which
    would false-alarm on the validated geometry (its ports sit at 3.8-5.7
    mid-band decay lengths). The advisory FIRES below ``3 / alpha`` (the floor
    of the validated envelope) and the message RECOMMENDS ``>= 5 / alpha``. If
    the band max reaches ``fc2`` the next mode propagates in-band and this
    advisory is skipped (a separate preflight covers that). Emits
    ``warnings.warn`` per under-clearance port; does not raise.
    """
    import warnings
    from rfx.preflight._common import PreflightWarning

    axis_idx = {"x": 0, "y": 1, "z": 2}
    dev = np.asarray(device_sigma)
    f_arr = np.asarray(freqs, dtype=float)
    f_max = float(f_arr.max())
    f_cen = 0.5 * (float(f_arr.min()) + f_max)
    for i, cfg in enumerate(cfgs):
        a = float(cfg.a)
        fc2 = _C0_SPARAMS / a  # TE20 cutoff for a TE10 port of width a
        if f_max >= fc2:
            # Next higher mode propagates in-band; a separate preflight warns.
            continue
        ax = axis_idx[cfg.normal_axis]
        other_axes = tuple(j for j in range(3) if j != ax)
        diff_profile = np.any(np.asarray(ref_sigmas[i]) != dev, axis=other_axes)
        # PEC is represented by the solver's realized E edges, not sigma.
        # Reading sigma alone therefore sees identical vacuum in a PEC
        # junction and its straight references. Compare each component:
        # unioning the three masks first would hide differently oriented
        # sheets that occupy the same node plane.
        ref_edges = None if ref_pec_edges is None else ref_pec_edges[i]
        if device_pec_edges is not None or ref_edges is not None:
            for component in range(3):
                dev_edge = (False if device_pec_edges is None else
                            np.asarray(device_pec_edges[component], dtype=bool))
                ref_edge = (False if ref_edges is None else
                            np.asarray(ref_edges[component], dtype=bool))
                diff_profile |= np.any(dev_edge != ref_edge, axis=other_axes)
        differing = np.nonzero(diff_profile)[0]
        if differing.size == 0:
            continue
        probe = int(cfg.probe_x)
        clearance_m = min(node_distance(grid, ax, probe, int(node))
                          for node in differing)
        alpha = 2.0 * np.pi * np.sqrt(max(fc2 ** 2 - f_cen ** 2, 0.0)) / _C0_SPARAMS
        if alpha <= 0.0:
            continue
        minimum_m = 3.0 / alpha       # validated-envelope floor (fires below)
        recommended_m = 5.0 / alpha   # recommendation in the message
        if clearance_m < minimum_m:
            warnings.warn(PreflightWarning(
                f"port_reference_sims: port index {i} ({cfg.normal_axis}-normal) "
                f"probe plane is only {clearance_m * 1e3:.1f} mm from the "
                f"junction — below {minimum_m * 1e3:.1f} mm (3 mid-band "
                f"evanescent decay lengths of the next higher mode, the floor "
                f"of the validated far-port envelope); recommend "
                f">= {recommended_m * 1e3:.1f} mm (5 decay lengths). Compact "
                f"port-to-junction clearance can leave the matrix non-passive "
                f"even with matched straight-guide references — move "
                f"the probe plane farther from the junction.",
                code="port_junction_probe_clearance",
                loc=f"port:{i}", source="_warn_junction_probe_clearance",
            ),
                stacklevel=2,
            )


def _warn_junction_cpml_thickness(grid, cfgs, freqs, cpml_layers):
    """Advisory: CPML thickness vs guide wavelength (pure NumPy, no FDTD).

    For each port compute the TE10 guide wavelength at band centre,
    ``lambda_g = lambda0 / sqrt(1 - (fc1 / f)^2)`` with ``fc1 = C0 / (2 a)``
    and ``lambda0 = C0 / f``. If the thinnest active propagation pad is
    thinner than ``0.5 * lambda_g`` the absorber may under-drain guided energy.
    Heuristic from the validated campaign: 20 mm CPML produced |S11| ripple
    ~0.11, 48 mm passed. Emits ``warnings.warn`` per thin-CPML port; does not
    raise.
    """
    import warnings

    f_arr = np.asarray(freqs, dtype=float)
    f_cen = 0.5 * (float(f_arr.min()) + float(f_arr.max()))
    for i, cfg in enumerate(cfgs):
        cpml_m = junction_absorber_depth(grid, cfg.normal_axis, int(cpml_layers))
        a = float(cfg.a)
        fc1 = _C0_SPARAMS / (2.0 * a)  # TE10 cutoff
        if f_cen <= fc1:
            # Below cutoff at band centre; the guide wavelength is undefined.
            continue
        lambda0 = _C0_SPARAMS / f_cen
        lambda_g = lambda0 / np.sqrt(1.0 - (fc1 / f_cen) ** 2)
        if cpml_m < 0.5 * lambda_g:
            warnings.warn(
                f"port_reference_sims: port index {i} CPML stack is "
                f"{cpml_m * 1e3:.1f} mm, less than 0.5 guide-wavelength "
                f"({0.5 * lambda_g * 1e3:.1f} mm) at band centre. Thin CPML "
                f"under-drains guided energy — 20 mm CPML produced |S11| ripple "
                f"~0.11 in the validated campaign, 48 mm passed. Thicken the "
                f"absorber.",
                AccuracyWarning,
                stacklevel=2,
            )


def _warn_ntff_box_dropped(sim, method_name: str) -> None:
    """One warning per S-matrix call when an NTFF box would be dropped.

    ``add_ntff_box()`` registers a far-field monitor, but the S-matrix
    result classes (``MSLSMatrixResult``, ``WaveguideSMatrixResult``, and
    the coaxial results) carry no ``ntff_data``/``ntff_box`` fields, so
    whatever the per-drive solves record is discarded with nothing said. Called ONCE at each S-matrix
    entry (after the cheap guards, before any FDTD), never per port, so a
    call covers all its drives with a single message.

    The warning and no-box silence are covered by ``tests/unit/farfield/test_ntff_smatrix_drop_warning.py``.
    """
    if sim._ntff is None:
        return
    import warnings

    corner_lo, corner_hi, ntff_freqs = sim._ntff
    n_f = int(np.asarray(ntff_freqs).shape[0])
    warnings.warn(
        f"{method_name}: an NTFF box is registered on this simulation "
        f"(add_ntff_box, corners {tuple(float(c) for c in corner_lo)} -> "
        f"{tuple(float(c) for c in corner_hi)} m, {n_f} frequencies) but "
        "this S-matrix path returns NO far-field data — the registered "
        "monitor's recording is dropped. OBSERVED: the result class of "
        "this method carries no ntff_data/ntff_box fields, so the "
        "radiation pattern you asked for is lost for every driven-port "
        "solve in this call. WHY: threading NTFF data out of the "
        "per-drive S-matrix solves (one pattern per drive) is not "
        "implemented"
        ". REMEDY: call run() on this "
        "simulation for far-field patterns (Result.ntff_data/ntff_box "
        "from the same port drive), or drop the NTFF box from "
        "S-matrix-only runs to save its DFT cost. STALE-IF: this "
        "method's result class grows ntff_data/ntff_box"
        ", at which point remove this warning.",
        ConfigurationWarning,
        stacklevel=3,
    )

# Far-port discipline: minimum absorber depth as a fraction of the guide
# wavelength at the LOWEST measured frequency. The message quotes this value,
# so it MUST NOT be duplicated as a literal there — a mismatch would report one
# threshold while enforcing another.
_FAR_PORT_LAMBDA_G_FRACTION = 0.5


def _warn_thin_absorber_vs_guide_wavelength(
    grid, cfgs, freqs, cpml_layers, boundary_spec,
):
    """Advisory: absorber depth vs guide wavelength at the LOWEST measured freq.

    ``compute_waveguide_s_matrix``'s "Far-port discipline" requires an absorber
    ``>= ~0.5 * lambda_g``, but nothing checked it on the plain two-port path:
    the ``port_reference_sims`` sibling advisory
    (:func:`_warn_junction_cpml_thickness`) only runs on the junction path, and
    the functional entry points run no ``sim.preflight()`` at all. A gated
    revision of crossval case 18 therefore shipped a 0.30 ``lambda_g`` stack in
    silence, and the absorber — not discretization — set the reported accuracy
    envelope.

    Evaluated at the **lowest** measured frequency, where ``lambda_g`` is
    longest and the ``cpml_layers=16`` default is weakest, because
    ``lambda_g`` diverges as ``f`` approaches cutoff. Three deliberate
    false-positive fences, each of which makes this a *lower* bound on the
    real requirement:

    * Skipped when the lowest measured frequency is at or below the mode's
      own cutoff. ``lambda_g`` is undefined there and the band itself is
      invalid — ``_check_waveguide_port_evanescent`` (preflight code
      ``port_freqs_below_cutoff``) owns that failure, and warning about the
      absorber on top of it would be noise. Note this means a band that
      starts below cutoff gets **no** absorber advisory.
    * Skipped when the port's propagation axis carries no absorbing face
      (a PEC-closed or periodic axis has no absorber to under-drain).
    * Uses the port's lowest-cutoff mode, whose ``lambda_g`` is the shortest
      and so the least demanding; higher-order content sitting nearer its own
      cutoff needs a thicker stack than this check asks for.

    Emits one ``warnings.warn`` per distinct (propagation axis, cutoff);
    does not raise. Pure NumPy, no FDTD.
    """
    import warnings

    if boundary_spec is None:
        return
    f_arr = np.asarray(freqs, dtype=float)
    if f_arr.size == 0:
        return
    f_lo = float(f_arr.min())
    seen: set = set()
    for cfg in cfgs:
        # A multimode port arrives as a list of per-mode configs; the
        # lowest-cutoff mode carries the least demanding requirement.
        modes = cfg if isinstance(cfg, list) else [cfg]
        if not modes:
            continue
        c = min(modes, key=lambda m: float(m.f_cutoff))
        axis = str(c.normal_axis)
        fc = float(c.f_cutoff)
        key = (axis, round(fc, 3))
        if key in seen:
            continue
        seen.add(key)

        if fc <= 0.0 or f_lo <= fc:
            continue
        from rfx.boundaries.depths import grid_face_depths
        faces = [(record.name.split("_")[1], record.realized)
                 for record in grid_face_depths(grid, spec=boundary_spec, budget=cpml_layers)
                 if record.name.startswith(axis + "_") and record.realized > 0]
        if not faces:
            continue

        from rfx.model.waveguide_modes import port_light_speed
        lambda_g = (port_light_speed(c) / f_lo) / np.sqrt(1.0 - (fc / f_lo) ** 2)
        required_m = _FAR_PORT_LAMBDA_G_FRACTION * lambda_g
        depths = [(side, n, absorber_depth(grid, axis, side, n)) for side, n in faces]
        thin = [(side, n, depth) for side, n, depth in depths if depth < required_m]
        if not thin:
            continue
        detail = ", ".join(
            f"{axis}-{side} {n} cells = {depth * 1e3:.1f} mm "
            f"({depth / lambda_g:.2f} lambda_g)"
            for side, n, depth in thin
        )
        dx = min(float(host_cells(grid, axis)[0 if side == "lo" else -1]) for side, _, _ in thin)
        warnings.warn(
            f"compute_waveguide_s_matrix: absorber on the {axis} propagation "
            f"axis is thinner than the documented "
            f"{_FAR_PORT_LAMBDA_G_FRACTION:g} guide-wavelength "
            f"far-port discipline at the lowest measured frequency "
            f"{f_lo / 1e9:.3f} GHz (mode cutoff {fc / 1e9:.3f} GHz, "
            f"lambda_g = {lambda_g * 1e3:.1f} mm): {detail}, against a "
            f"required {required_m * 1e3:.1f} mm. A thin absorber reflects "
            f"guided energy and can set the accuracy envelope instead of "
            f"discretization: in the WR-90 iris lane residual "
            f"|S11| ripple was 0.0706 at 0.30 lambda_g, 0.0366 at 0.50, and "
            f"0.0093 at 0.75, so {_FAR_PORT_LAMBDA_G_FRACTION:g} lambda_g "
            f"is a floor and not a target. "
            f"Raise cpml_layers to at least "
            f"{int(np.ceil(required_m / dx))} (0.75 lambda_g needs "
            f"{int(np.ceil(0.75 * lambda_g / dx))}).",
            AccuracyWarning,
            stacklevel=2,
        )
