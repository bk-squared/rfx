"""The MSL S-matrix calculator, moved verbatim out of ``rfx.api._sparams``.

Issue #980 Phase 2. ``compute_msl_s_matrix`` below is the ``_SparamMixin``
method relocated byte for byte — same signature, same docstring, same logic,
dedented by exactly four spaces and nothing else. It is a MODULE-LEVEL
function whose first parameter is still named ``self`` and is still the
:class:`~rfx.api.Simulation` instance: ``rfx.api._sparams`` binds it back onto
the class (``compute_msl_s_matrix = _msl.compute_msl_s_matrix``), so
``sim.compute_msl_s_matrix(...)`` keeps its name, signature, ``__doc__`` and
bound-method behaviour. A class wrapper may replace that binding in a later
step; the move itself is gated on bit identity of the extracted S arrays
(``tests/locks/test_sparams_split_bit_identity.py``).

``_resolve_msl_probe_entries`` stays on the class and is reached through
``self``, as are ``run`` / ``forward`` / ``add_probe`` /
``add_dft_plane_probe`` and the six ``self._*`` port/probe lists.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.sources.sources import GaussianPulse

from rfx.api._spec import (
    MSLSMatrixResult,
    _MSLPortEntry,
)

from rfx.sparams._common import (
    _msl_cell_profile,
    msl_modal_voltage,
    msl_solve_s_from_waves,
    _msl_wave_split_reliability,
    _warn_msl_wave_split_unreliable,
    _warn_msl_beta_scan_railed,
    _warn_if_ringdown_truncated,
    _project_passive,
    _warn_if_passivity_projected,
    _warn_if_nonpassive_smatrix,
    _warn_ntff_box_dropped,
    _register_msl_h_planes,
    _collocated_msl_h,
    _msl_power_wave_scales,
)


def compute_msl_s_matrix(
    self,
    *,
    n_steps: int | None = None,
    num_periods: float = 40.0,
    freqs: jnp.ndarray | None = None,
    n_freqs: int = 100,
    raw_3probe_dump_path: str | None = None,
    strict_extractor: bool = False,
    eps_override: "jnp.ndarray | None" = None,
    checkpoint_every: int | None = None,
    checkpoint_segments: int | None = None,
    enforce_passivity: bool = True,
    report_every: int | None = None,
) -> "MSLSMatrixResult":
    """Compute MSL S from probe-plane V/I with fitted line diagnostics.

    Returned S uses power waves for the positive real per-port analytic
    references: a=(V+R*I)/(2*sqrt(R)), b=(V-R*I)/(2*sqrt(R)).
    ``reference_impedances`` records those R values separately from the
    fitted ``Z0`` and source/load resistances. For unequal references,
    transmission S differs from the legacy voltage ratio by
    sqrt(R_input/R_output); equal-reference results are unchanged.

    ``enforce_passivity=True`` (default) projects the assembled S(f) onto
    the passive set per frequency (singular values clipped to 1 — the
    nearest matrix in spectral norm with ``||S||_2 <= 1``), so the
    returned ``S`` satisfies the passive bound at every frequency on the
    plain measurement path. This is constraint enforcement, not a physics
    fix: the unprojected matrix is kept in ``S_raw``, the per-bin clip
    amount in ``passivity_correction``, and a warning names the touched
    bins. Bins with a large correction are measurement artifacts (see
    ``reliable`` / ``settling_db`` for the cause) — the projection bounds
    them, it does not make them trustworthy, and it is NOT small where
    the raw extraction is bad: on a thru fixture whose raw sigma_max ran
    1.19-1.91, projecting moved |S21| 1.000 -> 0.61-0.72 and rotated its
    phase by up to 17 degrees, while ``Z0`` and ``beta`` stay raw. Never
    quote projected values as physics where ``passivity_correction`` is
    large. Set ``False`` to get the raw extraction in ``S`` unchanged.

    EXEMPTION: no projection is applied on the ``eps_override`` channel
    (traced or concrete) so that finite-difference and ``jax.grad``
    objectives see the same raw function; ``S`` is then the raw
    extraction with ``S_raw``/``passivity_correction`` absent, no
    projection warning fires, and ``S`` may exceed the bound (measured
    sigma_max 1.18 on a coarse thru).

    Surface-impedance sheets (``add_thin_conductor(...,
    surface_impedance_f0=...)``) are supported on this lane (#677/#679):
    the sheet is realized node-thin by the per-step operator inside the
    ``run()``/``forward()`` device dispatches — no lane-level ctx is
    built here. Combination refusals fire downstream at whichever lane
    entry the call actually reaches, and the two channels carry their
    OWN copies: on ``run()`` the dispersive-substrate and
    ``boundary='upml'`` refusals come from ``run_uniform`` with run-lane
    wording, while the ``forward()`` / ``eps_override`` channel never
    enters ``run_uniform`` and raises instead from the forward-lane
    entry in ``rfx/api/_execute.py``, naming that lane. #679 added the
    ``upml`` half there: before it, an ``eps_override`` call on a
    ``boundary='upml'`` sim silently ran the very combination ``run()``
    refuses (the sheet operator overwriting UPML's split-field E update
    at its edges). ADI refuses on both channels through the shared
    ``refuse_f0_sheets`` helper; the subgridded and distributed lanes
    carry their own call sites at whichever entry reaches them —
    ``FENCE_REGISTRY`` in ``tests/unit/materials/test_sheet_impedance.py`` is the
    authoritative per-lane list, AST-guarded against drift.
    ``subpixel_smoothing`` / ``conformal_pec`` are ``run()``-only
    keywords, so that combination is unreachable on the ``eps_override``
    channel rather than refused there.
    The TRACE must remain PEC: an f0 sheet realizes no PEC edge, and
    the closed Ampere-loop current and the V span anchor on the
    realized PEC wall planes (#931 §1.9 — a PEC trace may be a volume
    Box or a sheet; both are found). The Hammerstad-Jensen beta/Z0 anchors
    and the real-beta N-probe fit assume a lossless line, so a sheet
    lying INSIDE a probed span adds per-length loss the fit cannot
    represent — reported ``Z0``/``q`` shift (the Z0 honesty guard may
    warn) while the V-I production S stays valid. A non-z-normal sheet
    (x/y-normal) carries Ez in its tangential edge set: crossing the
    feed plane or the V-integration column it legitimately modifies the
    measured/launched Ez — place sheets clear of feed and probe planes.

    For each registered MSL port, runs one FDTD simulation with that
    port driven and the others passive.  The passive ports are NOT
    assumed matched — ``|a_passive/a_driven| = 0.243-0.248`` at the
    shipped R = 50 Ω (0.19-0.93 across terminations), re-measured on
    current main 2026-08-30 (#524, VESSL 369367257265, PR #799; the
    July figure quoted here before was 0.07-0.51) — so the S-matrix is
    recovered by solving the
    full wave system ``S = B·A⁻¹`` over all drives rather than by the
    per-column ratio ``b_j/a_d`` (issue #507; the ratio reported the far
    port's echo as the structure's own reflection).  At each port
    ``n_probes`` downstream DFT plane probes record Ez and the first
    probe also records the transverse H components for a closed
    Ampere-loop current. Production S is formed from this first-plane
    V/I and the analytic Hammerstad-Jensen reference impedance. The
    N-probe least-squares fit of
    ``V_n = α e^{-jβx_n} + γ e^{+jβx_n}`` supplies only the reported
    beta/Z0 diagnostics. Those fitted values do not enter S. S is
    referenced to each first probe plane, without translation back to
    the physical feed planes. The two bracketing H planes are linearly
    interpolated to the E-node using their physical coordinates, then
    the leapfrog temporal offset is corrected. Comparing different
    observation offsets still requires accounting for reference planes
    and impedances. ``probe_clearance`` reports the separate downstream
    layout recommendation; it is not a signal or accuracy score.

    Parameters
    ----------
    n_steps : int or None
        Timesteps per FDTD run. ``None`` → auto from ``num_periods``.
    num_periods : float
        Source-period multiples used to derive ``n_steps`` when not
        provided. Default 40 (MSL transients are slow to drain).
    freqs : array, optional
        Frequency grid. Defaults to
        ``linspace(freq_max / 10, freq_max, n_freqs)``.
    n_freqs : int
        Number of frequencies if ``freqs`` is None.
    raw_3probe_dump_path : str or None
        Optional ``.npz`` path. When provided, write the real
        simulation-derived N-probe voltage/current phasors used by the
        extractor, together with the production power-wave S-matrix, so the
        de-embedding can be independently checked without rerunning
        FDTD. The dump schema is ``rfx.msl_nprobe_dump`` v4: it explicitly
        records the power-wave convention and actual reference impedances.
        Legacy v3 records use voltage waves. ``raw_v`` has shape
        ``(n_driven, n_ports, n_probes_max, n_freqs)``.
        ``scripts/diagnostics/replay_msl_3probe_dump.py`` is SUPERSEDED
        for modern dumps (it expects the retired 3-probe/single-ratio
        v1 schema). Use ``rfx.validation.load_port_vi_dump_npz`` and
        ``replay_smatrix_from_port_vi_dump`` for independent S replay.
    strict_extractor : bool
        Honesty guard for the de-embedding (issue #80 Fix A). After
        extraction, the per-frequency ``|q|`` and extracted ``Z0``
        are validated against physical bounds (``|q| <= 1`` for a passive
        line; extracted ``Z0`` within 10 % of the analytic
        Hammerstad-Jensen value). When ``False`` (default) a violation
        raises a loud :func:`warnings.warn`; when ``True`` it raises
        :class:`ValueError` instead. With the N-probe extractor (Fix C)
        ``|q|`` and ``Z0`` should be healthy so this rarely fires — it
        is the safety net for pathological geometries.
    checkpoint_segments : int or None
        Gradient-checkpointing segment count for the reverse-mode AD tape on
        the **uniform** mesh (the standard MSL path), forwarded to
        :meth:`forward` (only active on the differentiable ``eps_override``
        channel). Must DIVIDE the auto-computed ``n_steps`` exactly — padding
        is rejected because it would shift the DFT accumulator windows.
        Choose the divisor nearest ``sqrt(n_steps)`` so backward memory scales
        ~``sqrt(n_steps)*carry`` instead of ``n_steps*carry`` — required for
        converged ``num_periods>=20`` AD that otherwise OOMs (G-AD-CHECKPOINT).
        Default ``None`` leaves forward-only runs and small-period AD unchanged.
    checkpoint_every : int or None
        Non-uniform-mesh counterpart of ``checkpoint_segments`` (chunk size,
        not segment count; issue #73). Forwarded to :meth:`forward`; raises
        ``NotImplementedError`` on the uniform path — use
        ``checkpoint_segments`` there.
    report_every : int or None
        Issue #667 — progress reporting for long solves. Forwarded to
        each per-drive :meth:`run` call, tagged ``MSL drive pI/N`` so
        the drives are distinguishable in a log. ``None`` (default) is
        OFF and leaves the solve byte-identical. The measured case this
        exists for: a 42.15 M-cell / 225,000-step call whose last log
        line was written at second 0 and was still the last line 4 h
        10 min later, with no way to tell slow from wedged. Ignored
        (with a warning) on the differentiable ``eps_override``
        channel, which routes through the traced :meth:`forward`.

    Returns
    -------
    MSLSMatrixResult
    """
    from rfx.probes.msl_wave_decomp import extract_msl_nprobe
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.sources.msl_port import (
        msl_ampere_pair,
        msl_axis_roles,
        msl_cell,
        msl_cross_section_span,
        msl_h_plane_stencil,
        msl_loop_current,
        msl_physical_point,
        msl_port_from_entry,
        msl_probe_x_coords_n,
    )

    if not self._msl_ports:
        raise ValueError("No MSL ports registered. Call add_msl_port() first.")
    if self._ports or self._waveguide_ports or self._floquet_ports:
        raise NotImplementedError(
            "compute_msl_s_matrix() is defined only for add_msl_port(...) "
            "families in the current simulation. Use separate "
            "simulations for add_port(...), add_waveguide_port(...), "
            "or add_floquet_port(...) S-parameter workflows."
        )
    if self._tfsf is not None:
        raise NotImplementedError(
            "compute_msl_s_matrix() is not supported together with TFSF; "
            "TFSF is a plane-wave source, not an MSL port."
        )
    if self._coaxial_ports:
        raise NotImplementedError(
            "compute_msl_s_matrix() does not include add_coaxial_port(...); "
            "coaxial-port S-parameters need a separate validated V/I "
            "extraction and calibration contract."
        )
    is_nonuniform = (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    )
    if is_nonuniform and any(
        getattr(pe, "mode", "laplace") == "eigenmode"
        for pe in self._msl_ports
    ):
        raise NotImplementedError(
            "compute_msl_s_matrix() on a non-uniform mesh supports "
            "mode='laplace'/'uniform' (Ez static-Laplace feed) only; the "
            "eigenmode J+M launch needs the magnetic-source channel that "
            "the non-uniform runner does not carry. Use mode='laplace' "
            "(the add_msl_port default) on the graded-mesh lane."
        )
    if self._refinement is not None:
        raise NotImplementedError(
            "compute_msl_s_matrix() is not supported with SBP-SAT "
            "subgridding."
        )
    if self._solver == "adi":
        raise NotImplementedError(
            "compute_msl_s_matrix() is not supported with solver='adi'; "
            "use the uniform Yee solver."
        )

    # Issue #704: an NTFF box would be silently dropped on this path.
    _warn_ntff_box_dropped(self, "compute_msl_s_matrix()")

    entries = list(self._msl_ports)
    n_ports = len(entries)

    # Probe placement, material assembly and each run share the resolved
    # mesh. Missing z profiles are synthesized locally by the grid builder;
    # writing a derived profile into the declaration would freeze auto-mesh
    # state and change the resolved domain during the driver.
    grid = self._build_realized_grid()

    if freqs is None:
        freqs_arr = np.asarray(jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs))
    else:
        freqs_arr = np.asarray(freqs)
    n_freqs_used = int(freqs_arr.shape[0])

    # Issue #469: solve the probe-offset interval for AUTO ports (the
    # downstream reflector term is only computable here, with the full
    # geometry registered — see _resolve_msl_auto_offsets).
    entries = self._resolve_msl_probe_entries(grid)

    # Build MSLPort descriptors and probe coords once (geometry shared).
    # Issue #661: msl_port_from_entry projects ``position`` onto the
    # port frame for whichever in-plane axis ``direction`` names.
    msl_ports = [msl_port_from_entry(pe) for pe in entries]

    # N-probe placement (issue #80 Fix C). Probe n sits at
    # offset + n*spacing cells from the feed plane. N >= 3.
    n_probes_per_port = [int(pe.n_probes) for pe in entries]
    probe_xs = [
        msl_probe_x_coords_n(
            grid, mp,
            n_probes=n_probes,
            n_offset_cells=pe.n_probe_offset,
            n_spacing_cells=pe.n_probe_spacing,
        )
        for mp, pe, n_probes in zip(msl_ports, entries, n_probes_per_port)
    ]
    # Resolve before registering internal observations or advancing fields.
    # H[i] lies half a propagation cell beyond E[i], for either port sign.
    h_stencils = [msl_h_plane_stencil(grid, mp, xs[0])
                  for mp, xs in zip(msl_ports, probe_xs)]
    # Resolved in rfx.preflight.msl, NOT through the rfx.api._preflight
    # re-export, so that this reader and preflight's own reader
    # (_check_msl_port_geometry, which lives in that module and looks the
    # name up in its globals) share ONE lookup target. #980 Phase 3 leg 1
    # moved both; before it, a single monkeypatch on rfx.api._preflight
    # covered the two, and
    # tests/unit/ports/test_msl_clearance_diagnostic.py asserts that it
    # still does -- the re-export would have split them into two bindings
    # of which only one is ever patched.
    from rfx.preflight.msl import msl_probe_clearance_for_port
    probe_clearance = tuple(msl_probe_clearance_for_port(
        self, pe, grid, probe_coordinates=xs,
    ) for pe, xs in zip(entries, probe_xs))
    # ``probe_xs`` are the N physical x-coordinates fed to the
    # N-probe extractor (issue #80 Fix C), which fits
    # V_n = alpha*exp(-j*beta*x_n) + gamma*exp(+j*beta*x_n). The
    # coordinates are increasing for ``+x`` ports and decreasing for
    # ``-x`` ports; the extractor anchors the model at probe 0 and
    # only uses coordinate differences, so feeding raw physical x
    # keeps alpha = the +x-travelling wave for BOTH port directions
    # — matching the legacy 3-probe sign convention the S11 sign
    # was validated against.

    # Per-axis cell-size arrays for V/I integration. Both uniform and
    # non-uniform grids are supported — NonUniformGrid exposes per-cell
    # dx_arr/dy_arr/dz (NOT *_profile); see _msl_cell_profile.
    #
    # Issue #661: ports may point along different in-plane axes, so the
    # transverse profiles are resolved PER PORT. The substrate-normal
    # profile is always z (the normal axis is welded — see
    # msl_axis_roles), which is why the modal-voltage integration below
    # needs no per-port branch.
    _axis_n = {"x": grid.nx, "y": grid.ny, "z": grid.nz}

    def _prof(ax):
        return _msl_cell_profile(grid, ax, _axis_n[ax])

    dz_arr = _prof("z")     # substrate-normal profile (V integration)

    # Fixed cross-section indices per port (same across all runs).
    # Names are the historical x-frame names; their MEANING is
    # (width, substrate-normal): ``j_*`` indexes the trace-width axis
    # and ``k_*`` the normal axis. A DFT plane normal to the
    # propagation axis is stored as [freq, width, normal] for BOTH
    # "x" and "y" plane normals (probes.py: axis 0 -> (ny, nz),
    # axis 1 -> (nx, nz)), so these indices address the recorded
    # planes directly without a per-direction branch.
    port_idx_meta = []
    for mp in msl_ports:
        span = msl_cross_section_span(grid, mp)
        a_ax, b_ax = msl_ampere_pair(mp.direction)
        port_idx_meta.append(dict(
            j_lo=span["w_lo"], j_hi=span["w_hi"],
            k_lo=span["n_lo"], k_hi=span["n_hi"],
            j_centre=span["w_centre"], k_top=span["n_hi"],
            height=mp.z_hi - mp.z_lo,
            i_feed=span["i_feed"],
            prop_axis=span["prop_axis"], width_axis=span["width_axis"],
            normal_axis=span["normal_axis"], sign=span["sign"],
            prop_idx=span["prop_idx"], width_idx=span["width_idx"],
            normal_idx=span["normal_idx"],
            # Closed-Ampere-loop transverse pair (a_hat x b_hat = p_hat).
            a_axis=a_ax, b_axis=b_ax,
            a_is_width=(a_ax == span["width_axis"]),
            a_arr=_prof(a_ax), b_arr=_prof(b_ax),
            h_a=f"h{a_ax}", h_b=f"h{b_ax}",
        ))

    # Analytic Hammerstad-Jensen anchor per port (issue #80 Fix C).
    # ``beta0_per_port[p]`` is the (n_freqs,) propagation-constant
    # guess ``omega * sqrt(eps_eff) / c`` used to centre the N-probe
    # extractor's robust beta scan; ``z0_hj_per_port[p]`` is the
    # analytic Z0 used by the honesty guard.
    #
    # Substrate permittivity precedence (mirrors rfx/runners/uniform.py
    # so the beta anchor and the source see the SAME eps_r): explicit
    # add_msl_port(eps_r_sub=...) > the rasterised FDTD eps_r at the
    # trace-centre substrate cell. Reading the material array makes
    # this robust even when the user did not pass eps_r_sub — a plain
    # pe.eps_r_sub-or-1.0 fallback would anchor the scan on vacuum and
    # land beta outside the scan window for a loaded substrate.
    from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
    _C0_MSL = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
    # #679: surface_impedance_f0 sheets are supported on this lane. NO
    # ctx is built here — every device dispatch below goes through the
    # public run()/forward(), which assemble their own materials with
    # sheet_specs and build the ctx against their OWN final pec_mask
    # (run: _execute.py -> runners/uniform.py build_sheet_impedance_ctx;
    # forward: _execute.py; NU lanes: runners/nonuniform.py). The anchor
    # assembly below is deliberately sheet-free (apply_thin_conductor
    # emits nothing without a sheet_specs collector) — correct, because
    # an f0 sheet carries no eps and never enters pec_mask. This lane
    # has NO vacuum reference run, so there is no strip_sheet_impedance
    # analogue.
    _msl_pec_sheets: list = []
    _msl_pec_wires: list = []
    _msl_assembled = (
        self._assemble_materials_nu(
            grid, pec_sheets=_msl_pec_sheets, pec_wires=_msl_pec_wires)
        if is_nonuniform
        else self._assemble_materials(
            grid, pec_sheets=_msl_pec_sheets, pec_wires=_msl_pec_wires)
    )
    _msl_materials = _msl_assembled[0]
    # #931 §1.9: the trace is located by its REALIZED wall planes, not
    # by a pec_mask cell scan — a sheet-declared trace owns no cell.
    from rfx.boundaries.pec import (
        realized_pec_edge_masks as _rpem_msl,
    )
    from rfx.probes.msl_wave_decomp import (
        realized_trace_planes_on_column as _trace_planes,
    )
    _msl_pec_edge_masks = None
    if (_msl_assembled[3] is not None or _msl_pec_sheets
            or _msl_pec_wires):
        _msl_pec_edge_masks = _rpem_msl(
            _msl_assembled[3], sheets=tuple(_msl_pec_sheets),
            wires=tuple(_msl_pec_wires),
            periodic=self._periodic_flags())
    beta0_per_port: list[np.ndarray] = []
    z0_hj_per_port: list[float] = []
    for p_idx, pe in enumerate(entries):
        meta = port_idx_meta[p_idx]
        if pe.eps_r_sub is not None:
            eps_r_ref = float(pe.eps_r_sub)
        else:
            k_mid = (meta["k_lo"] + meta["k_hi"]) // 2
            eps_cell = msl_cell(
                pe.direction, meta["i_feed"], meta["j_centre"], k_mid
            )
            eps_r_ref = float(np.asarray(_msl_materials.eps_r[eps_cell]))
        z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(
            pe.width, pe.height, eps_r_ref
        )
        beta0_per_port.append(
            2.0 * np.pi * freqs_arr * float(np.sqrt(eps_eff_hj)) / _C0_MSL
        )
        z0_hj_per_port.append(float(z0_hj))

    # Trace-conductor z-cell span per port (issue #80 stage S1). The
    # closed Ampere-loop current needs the PEC trace cells; the trace
    # is the PEC run at/above the substrate top in the port's centre
    # column (the ground-plane PEC sits far below near k_lo).
    trace_k_per_port: list[tuple[int, int]] = []
    for p_idx in range(n_ports):
        meta = port_idx_meta[p_idx]
        # Walk UP the substrate-normal axis (always z) from the
        # substrate top, at the feed cell on the propagation axis and
        # the trace centre on the width axis (issue #661).
        _ij = tuple(
            meta["i_feed"] if c == meta["prop_idx"] else meta["j_centre"]
            for c in range(3) if c != meta["normal_idx"]
        )
        _k_lo_tr, _k_hi_tr = _trace_planes(
            _msl_pec_edge_masks, meta["normal_idx"], _ij, meta["k_top"],
            periodic=self._periodic_flags())
        if _k_lo_tr is None:
            raise RuntimeError(
                "compute_msl_s_matrix: no realized PEC trace conductor "
                "found above the substrate top for MSL port "
                f"{entries[p_idx].name!r}; the closed Ampere-loop "
                "current (issue #80 stage S1) needs the trace. Declare "
                "the microstrip trace as a Box(material='pec') (a "
                "volume) or as a zero-thickness Box / add_thin_conductor "
                "(a sheet, #931). A surface_impedance_f0 thin conductor "
                "is NOT a trace conductor here — it realizes no PEC "
                "edge, and the Ampere-loop current and V span anchor on "
                "realized PEC wall planes. Keep the trace PEC and use f0 "
                "sheets for auxiliary lossy metal only."
            )
        trace_k_per_port.append((_k_lo_tr, _k_hi_tr))

    # Stash existing add_dft_plane_probe registrations and restore on exit.
    saved_dft = list(self._dft_planes)
    _had_dft_regions = "_dft_plane_regions" in self.__dict__
    saved_dft_regions = dict(getattr(self, "_dft_plane_regions", {}))
    saved_msl = list(self._msl_ports)
    saved_ports = list(self._ports)
    saved_probes = list(self._probes)
    saved_internal_probes = set(self._internal_probe_indices)
    try:
        _complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
        power_scales = _msl_power_wave_scales(z0_hj_per_port, _complex_dtype)
        S = jnp.zeros((n_ports, n_ports, n_freqs_used), dtype=_complex_dtype)
        Z0_per_run = jnp.zeros((n_ports, n_freqs_used), dtype=_complex_dtype)
        beta_first = jnp.zeros(n_freqs_used, dtype=_complex_dtype)
        # N-probe extractor (issue #80 Fix C): store all N voltage
        # probe phasors. n_probes may differ per port — store the
        # max width and zero-pad shorter ports.
        n_probes_max = max(n_probes_per_port)
        raw_v = jnp.zeros(
            (n_ports, n_ports, n_probes_max, n_freqs_used), dtype=_complex_dtype
        )
        raw_i1 = jnp.zeros((n_ports, n_ports, n_freqs_used), dtype=_complex_dtype)
        # Optional evidence preserves both measured sides, so the old
        # same-index current and the corrected one can be compared from
        # ONE field record rather than another FDTD run.
        raw_i1_left = jnp.zeros_like(raw_i1) if raw_3probe_dump_path is not None else None
        raw_i1_same_index = jnp.zeros_like(raw_i1) if raw_3probe_dump_path is not None else None
        raw_z0 = jnp.zeros((n_ports, n_ports, n_freqs_used), dtype=_complex_dtype)
        raw_q = jnp.zeros((n_ports, n_ports, n_freqs_used), dtype=_complex_dtype)
        # β-scan rail flags per (driven, port) fit (issue #681).
        raw_beta_railed = jnp.zeros(
            (n_ports, n_ports, n_freqs_used), dtype=bool
        )
        # Wave amplitudes per (driven, port) for the multi-drive solve
        # (issue #507). Python lists of jnp arrays, not a stacked array,
        # so eps_override tracers stay on the AD tape.
        wave_a: list[list] = [[None] * n_ports for _ in range(n_ports)]
        wave_b: list[list] = [[None] * n_ports for _ in range(n_ports)]

        # Ring-down settling witness (project rule: fixed-length
        # open-domain records must quote end/peak energy before any
        # claims-bearing number). Point Ez time-series probes at EVERY
        # port probe plane, mid-substrate under the trace — a single
        # plane is standing-wave-node sensitive (measured on the thru
        # fixture at num_periods=6: 18.1 dB spread across planes, i.e.
        # PASS at one plane and FAIL at another for the same record), so
        # the witness takes the WORST plane. For the PASSIVE ports of a
        # run the whole record is response, so end/peak there is the
        # textbook ring-down witness.
        _witness_base = len(self._probes)
        _witness_counts: list[int] = []
        for pe_w, pxs_w, meta_w in zip(entries, probe_xs, port_idx_meta):
            _w_centre_m = float(pe_w.position[meta_w["width_idx"]])
            _n_lo_m = float(pe_w.position[meta_w["normal_idx"]])
            for _x_w in pxs_w:
                # ``_x_w`` is a coordinate on the PROPAGATION axis;
                # rebuild the physical point for this port's direction.
                self.add_probe(
                    position=msl_physical_point(
                        pe_w.direction,
                        float(_x_w),
                        _w_centre_m,
                        _n_lo_m + 0.5 * float(pe_w.height),
                    ),
                    component="ez",
                )
            _witness_counts.append(len(pxs_w))
        _witness_total = sum(_witness_counts)
        # Mark the witness probes as library-internal so probe-placement
        # preflight advisories and the #332 tail advisory skip them
        # (issue #470: 10 self-inflicted advisories per driven run on
        # the 2-port thru buried the genuine MSL port-clearance
        # advisories, and #332 double-fired next to settling_db).
        self._internal_probe_indices.update(
            range(_witness_base, _witness_base + _witness_total)
        )
        settling_db_runs = np.full(n_ports, np.nan)

        for driven in range(n_ports):
            # Re-instantiate a clean simulation by mutating in place:
            # use add_msl_port as the registration path, but here we
            # need finer control over excite=True/False per-run, so
            # rebuild ``self._msl_ports`` for this run.
            run_entries = []
            for idx, pe in enumerate(entries):
                new_excite = (idx == driven) and pe.excite
                if new_excite:
                    wf = pe.waveform if pe.waveform is not None else \
                        GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
                else:
                    wf = None
                # Carry the whole probe ladder: an omitted n_probes
                # silently reverted to the dataclass default (5) here,
                # so a port registered with n_probes=3 ran with a
                # five-rung bookkeeping entry (the reference-plane
                # crossing guard in _forward_from_materials walks it).
                run_entries.append(_MSLPortEntry(
                    name=pe.name, position=pe.position,
                    width=pe.width, height=pe.height,
                    direction=pe.direction, impedance=pe.impedance,
                    waveform=wf, excite=new_excite,
                    n_probe_offset=pe.n_probe_offset,
                    n_probe_spacing=pe.n_probe_spacing,
                    n_probes=pe.n_probes,
                    mode=pe.mode,
                    eps_r_sub=pe.eps_r_sub,
                ))
            self._msl_ports = run_entries

            # Register DFT plane probes for V (Ez) and I (Hy).
            self._dft_planes = list(saved_dft)
            self._dft_plane_regions = dict(saved_dft_regions)
            ez_probe_names: list[list[str]] = [[] for _ in range(n_ports)]
            h_probe_names = []
            for p_idx, (mp, pxs) in enumerate(zip(msl_ports, probe_xs)):
                # Plane normal = this port's PROPAGATION axis; the two
                # H components are the closed-Ampere-loop pair
                # (a_hat x b_hat = p_hat), not a fixed (hy, hz).
                _meta_p = port_idx_meta[p_idx]
                _plane_axis = _meta_p["prop_axis"]
                for q_idx, x_coord in enumerate(pxs):
                    nm = f"_msl_run{driven}_p{p_idx}_ez{q_idx}"
                    self.add_dft_plane_probe(
                        axis=_plane_axis, coordinate=float(x_coord),
                        component="ez", freqs=jnp.asarray(freqs_arr),
                        name=nm,
                    )
                    self._dft_plane_regions[nm] = (
                        _meta_p["j_centre"],
                        _meta_p["j_centre"] + 1,
                        _meta_p["k_lo"],
                        trace_k_per_port[p_idx][0],
                    )
                    ez_probe_names[p_idx].append(nm)
                _k_tr_lo, _k_tr_hi = trace_k_per_port[p_idx]
                _h_crop_region = (
                    _meta_p["j_lo"] - 1,
                    _meta_p["j_hi"] + 1,
                    _k_tr_lo - 1,
                    _k_tr_hi + 1,
                )
                h_probe_names.append(_register_msl_h_planes(
                    self, f"_msl_run{driven}_p{p_idx}", h_stencils[p_idx],
                    (_meta_p["h_a"], _meta_p["h_b"]), jnp.asarray(freqs_arr),
                    _h_crop_region,
                ))

            # G-AD-WIRE: when eps_override is provided use the
            # differentiable forward() path so jax.grad can flow
            # from eps_override through the DFT plane accumulators
            # into the V/I assembly. Otherwise fall back to run()
            # for imperative (non-AD) workflows.
            if eps_override is not None:
                if report_every is not None:
                    import warnings as _w667
                    _w667.warn(
                        "report_every is ignored on the "
                        "compute_msl_s_matrix(eps_override=...) channel: "
                        "that channel routes through forward(), which is "
                        "the differentiable (traced) path, and host-side "
                        "wall-clock progress reporting cannot run under a "
                        "trace (issue #667).",
                        UserWarning, stacklevel=2,
                    )
                fwd_result = self.forward(
                    eps_override=eps_override,
                    n_steps=n_steps,
                    num_periods=num_periods,
                    checkpoint_every=checkpoint_every,
                    checkpoint_segments=checkpoint_segments,
                )
                planes = fwd_result.dft_planes or {}
                _ts_result = fwd_result
            else:
                # Pass the progress kwargs ONLY when reporting is on:
                # default-off must preserve the CALL SIGNATURE, not just
                # the numbers. CI shard 2 proved why — test doubles that
                # monkeypatch sim.run with a fixed signature broke on the
                # unconditional kwarg, and in two tests the TypeError was
                # swallowed by a fallback except and surfaced as
                # DID-NOT-WARN instead (PR #555 class).
                result = self.run(
                    n_steps=n_steps,
                    num_periods=num_periods,
                    compute_s_params=False,
                    **({} if report_every is None else {
                        "report_every": report_every,
                        "report_label": f"MSL drive p{driven + 1}/{n_ports}",
                    }),
                )
                planes = result.dft_planes or {}
                _ts_result = result

            # Settling witness for this driven run: worst end/peak
            # Ez^2 ratio across the per-port witness probes. Host-side
            # numpy on concrete values only — on the eps_override AD
            # path time_series may be a tracer, in which case the
            # witness is skipped (NaN) rather than concretised.
            _ts = getattr(_ts_result, "time_series", None)
            if _ts is not None and not is_tracer(_ts):
                _ts_np = np.asarray(
                    _ts[:, _witness_base:_witness_base + _witness_total],
                    dtype=float,
                )
                if _ts_np.shape[0] >= 10 and _ts_np.shape[1] == _witness_total:
                    _p = _ts_np ** 2
                    _tail = max(1, _p.shape[0] // 10)
                    _end = _p[-_tail:, :].mean(axis=0)
                    _peak = _p.max(axis=0)
                    _tiny = np.finfo(float).tiny
                    _ratio_db = 10.0 * np.log10(
                        (_end + _tiny) / (_peak + _tiny)
                    )
                    settling_db_runs[driven] = float(np.max(_ratio_db))

            # Helper: integrate V and I per port from the recorded planes.
            v_per_port: list[list[np.ndarray]] = []
            i_first_per_port: list[np.ndarray] = []
            for p_idx, meta in enumerate(port_idx_meta):
                vs = []
                for nm in ez_probe_names[p_idx]:
                    ez_plane = jnp.asarray(planes[nm].accumulator)
                    _v_region = self._dft_plane_regions.get(nm)
                    # ez_plane shape: (n_freqs, ny, nz)
                    # Top of the V span = the RASTERIZED trace's bottom
                    # node, not round(h_sub/dx) (= meta["k_hi"]): Box
                    # rasterization is half-open over node coordinates,
                    # so for frac(h_sub/dx) in (0, 0.5) the trace lands
                    # at ceil = k_hi + 1 and the k_hi anchor is one
                    # substrate edge SHORT — V and the Ampere-loop
                    # current would reference different conductor
                    # planes (PR #516 review, finding F2; measured on
                    # the dx=80um gate fixture: trace node 4, k_hi 3).
                    # trace_k_per_port is the same PEC search the
                    # current integration uses, so V and I share one
                    # conductor plane by construction.
                    # ez_plane is [freq, width, normal] for an x- OR a
                    # y-normal plane alike (issue #661), so j_centre /
                    # k_lo / k_hi address it unchanged and the dz_arr
                    # here is always the substrate-normal profile.
                    if (
                        _v_region is not None
                        and ez_plane.shape[1:] == (
                            _v_region[1] - _v_region[0],
                            _v_region[3] - _v_region[2],
                        )
                    ):
                        vs.append(msl_modal_voltage(
                            ez_plane,
                            j_centre=0,
                            k_lo=0,
                            k_hi=_v_region[3] - _v_region[2],
                            dz_arr=dz_arr[_v_region[2]:_v_region[3]],
                            dtype=_complex_dtype,
                        ))
                    else:
                        # Replay fixtures and third-party test doubles
                        # may still return legacy full-plane arrays.
                        vs.append(msl_modal_voltage(
                            ez_plane, j_centre=meta["j_centre"],
                            k_lo=meta["k_lo"],
                            k_hi=trace_k_per_port[p_idx][0],
                            dz_arr=dz_arr, dtype=_complex_dtype,
                        ))
                v_per_port.append(vs)
                # G-AD-WIRE: keep on JAX tape when eps_override is
                # set. np.asarray() would concretise a JAX tracer and
                # break jax.grad. jnp.asarray() is a no-op on a real
                # jnp.ndarray and still works for numpy arrays.
                hy_plane, hz_plane = _collocated_msl_h(
                    planes, h_probe_names[p_idx], h_stencils[p_idx]["weights"],
                )
                _h_region = self._dft_plane_regions.get(
                    h_probe_names[p_idx][0][1]
                )
                _h_is_cropped = (
                    _h_region is not None
                    and hy_plane.shape[1:] == (
                        _h_region[1] - _h_region[0],
                        _h_region[3] - _h_region[2],
                    )
                    and hz_plane.shape == hy_plane.shape
                )
                # Leapfrog E/H half-step time correction. add_dft_plane_probe
                # timestamps EVERY component at t = step·dt
                # (rfx/probes/probes.py:457), but H lives half a step behind E
                # (H at t − dt/2), so the recorded Hy/Hz DFT is missing the
                # exp(+jω·dt/2) factor the flux monitor already applies
                # (rfx/simulation.py:1380-1382: phase_h = phase_e·exp(+jω·dt/2)).
                # Ez (→ V) is correctly at t and needs no correction. Without
                # this, I = ∮H·dl carries a spurious exp(−jω·dt/2) so the V·I
                # de-embedding sees Zin = V/I rotated by exp(+jω·dt/2) — a
                # frequency-dependent phase that can push Re(Zin) < 0 → a
                # non-physical |S11| > 1 near the passive boundary (the same
                # half-step class as the 2026-04-28 s11_from_dumps artefact).
                _hs_phase = jnp.exp(
                    1j * 2.0 * jnp.pi * jnp.asarray(freqs_arr)
                    * (float(grid.dt) * 0.5)
                ).astype(hy_plane.dtype)
                hy_plane = hy_plane * _hs_phase[:, None, None]
                hz_plane = hz_plane * _hs_phase[:, None, None]
                # Closed Ampere-loop current ∮H·dl around the trace
                # conductor (issue #80 stage S1). The pre-S1 inline
                # integral summed the bottom Hy leg only and undercounted
                # I by ~1.5x, inflating the de-embedded Z0 to ~74 vs the
                # ~48 ohm analytic value. msl_loop_current closes the
                # contour (bottom/top Hy legs + left/right Hz legs) and
                # carries the +x current-sign convention.
                k_tr_lo, k_tr_hi = trace_k_per_port[p_idx]
                if _h_is_cropped:
                    assert _h_region is not None
                    _w_lo, _w_hi, _n_lo, _n_hi = _h_region
                    _j_lo = meta["j_lo"] - _w_lo
                    _j_hi = meta["j_hi"] - _w_lo
                    _k_lo = k_tr_lo - _n_lo
                    _k_hi = k_tr_hi - _n_lo
                else:
                    _w_lo, _w_hi = 0, hy_plane.shape[1]
                    _n_lo, _n_hi = 0, hy_plane.shape[2]
                    _j_lo, _j_hi = meta["j_lo"], meta["j_hi"]
                    _k_lo, _k_hi = k_tr_lo, k_tr_hi
                # Issue #661: msl_loop_current wants the planes in the
                # right-handed transverse frame [freq, a, b] with
                # a_hat x b_hat = p_hat. The recorded planes are
                # [freq, width, normal]. For "+x"/"-x" the pair is
                # (y, z) = (width, normal) — already in frame, no
                # transpose, byte-identical to the pre-#661 call. For
                # "+y"/"-y" the pair is (z, x) = (normal, width), so
                # the two axes swap and the spans swap with them.
                # Do NOT "simplify" this into a plain x<->y rename:
                # that is a reflection, it flips the sign of I, and it
                # inverts S silently (see _MSL_CYCLIC_PAIR).
                if meta["a_is_width"]:
                    _ha, _hb = hy_plane, hz_plane
                    _a_lo, _a_hi = _j_lo, _j_hi
                    _b_lo, _b_hi = _k_lo, _k_hi
                    _a_arr = meta["a_arr"][_w_lo:_w_hi]
                    _b_arr = meta["b_arr"][_n_lo:_n_hi]
                else:
                    _ha = jnp.transpose(hy_plane, (0, 2, 1))
                    _hb = jnp.transpose(hz_plane, (0, 2, 1))
                    _a_lo, _a_hi = _k_lo, _k_hi
                    _b_lo, _b_hi = _j_lo, _j_hi
                    _a_arr = meta["a_arr"][_n_lo:_n_hi]
                    _b_arr = meta["b_arr"][_w_lo:_w_hi]
                i_f = msl_loop_current(
                    _ha, _hb,
                    j_lo=_a_lo, j_hi=_a_hi,
                    k_trace_lo=_b_lo, k_trace_hi=_b_hi,
                    dy_arr=_a_arr, dz_arr=_b_arr,
                    direction=msl_ports[p_idx].direction,
                )
                if raw_3probe_dump_path is not None:
                    assert raw_i1_left is not None and raw_i1_same_index is not None
                    side_currents = []
                    for side in (0, 1):
                        side_a = jnp.asarray(planes[h_probe_names[p_idx][0][side]].accumulator)
                        side_b = jnp.asarray(planes[h_probe_names[p_idx][1][side]].accumulator)
                        side_a = side_a * _hs_phase[:, None, None]
                        side_b = side_b * _hs_phase[:, None, None]
                        if not meta["a_is_width"]:
                            side_a = jnp.transpose(side_a, (0, 2, 1))
                            side_b = jnp.transpose(side_b, (0, 2, 1))
                        side_currents.append(msl_loop_current(
                            side_a, side_b, j_lo=_a_lo, j_hi=_a_hi,
                            k_trace_lo=_b_lo, k_trace_hi=_b_hi,
                            dy_arr=_a_arr, dz_arr=_b_arr,
                            direction=msl_ports[p_idx].direction,
                        ))
                    raw_i1_left = raw_i1_left.at[driven, p_idx, :].set(side_currents[0])
                    raw_i1_same_index = raw_i1_same_index.at[driven, p_idx, :].set(side_currents[1])
                i_first_per_port.append(i_f)
                # N-probe least-squares wave decomposition (issue #80
                # Fix C). Stack the N voltage probes into (n_freqs, N),
                # anchor the beta scan on the analytic HJ guess, and
                # solve the over-determined (alpha, gamma) system by
                # SVD lstsq — this removes the 3-probe q->1 singularity.
                n_probes_p = n_probes_per_port[p_idx]
                v_stack = jnp.stack(v_per_port[p_idx], axis=-1)  # (n_freqs, N)
                raw_v = raw_v.at[driven, p_idx, :n_probes_p, :].set(jnp.asarray(v_stack.T, dtype=_complex_dtype))
                raw_i1 = raw_i1.at[driven, p_idx, :].set(jnp.asarray(i_f, dtype=_complex_dtype))
                res_p = extract_msl_nprobe(
                    jnp.asarray(v_stack),
                    jnp.asarray(np.asarray(probe_xs[p_idx], dtype=float)),
                    jnp.asarray(i_f),
                    jnp.asarray(beta0_per_port[p_idx]),
                    z0_hj=z0_hj_per_port[p_idx],
                )
                # Normalize the REPORTED characteristic-impedance sign per port
                # (issue #140). msl_loop_current negates the loop current ONLY for
                # "+x" ports (rfx/sources/msl_port.py:947-948), so a "-x" port's
                # fitted z0 = (alpha - gamma)/I inherits a negative sign while a
                # physical Z0 is positive-real. Mirror that exact binary so BOTH
                # ports report a positive Z0. This touches ONLY the reported/dumped
                # Z0 (raw_z0, Z0_per_run) and the |Z0| honesty-guard; it never
                # enters S11/S21 (which use the static analytic Hammerstad-Jensen
                # z0_hj). It removes the spurious ~228% guard deviation on -x ports
                # while leaving the genuine ~20-27% 3-cell Yee-staircase Z0 warning on
                # both ports. NB: the raw current dump (raw_i1) intentionally keeps
                # its un-normalized sign; only the DERIVED Z0 is sign-normalized.
                dir_sign = float(
                    msl_axis_roles(msl_ports[p_idx].direction)[3]
                )
                z0_fit = jnp.asarray(res_p["z0"], dtype=_complex_dtype) * dir_sign
                raw_z0 = raw_z0.at[driven, p_idx, :].set(z0_fit)
                raw_q = raw_q.at[driven, p_idx, :].set(jnp.asarray(res_p["q"], dtype=_complex_dtype))
                raw_beta_railed = raw_beta_railed.at[driven, p_idx, :].set(
                    jnp.asarray(res_p["beta_railed"], dtype=bool)
                )
                if p_idx == driven:
                    # V·I single-plane wave split at probe 0 (issue #80
                    # stage S1): a=(V+Z0*I)/2, b=(V-Z0*I)/2, S11=b/a —
                    # the OpenEMS-style telegrapher de-embedding. With a
                    # real positive Z0 and a passive structure this is
                    # bounded |S11|<=1, unlike the Fix-C alpha/gamma
                    # spatial fit that blew up to |S11|>1 on a strong
                    # reflector. Z0 is analytic Hammerstad-Jensen; I is
                    # the closed Ampere loop.
                    v0_d = v_per_port[driven][0]
                    z0hj_d = z0_hj_per_port[driven]
                    a_fwd_d = 0.5 * (v0_d + z0hj_d * i_f)
                    b_ref_d = 0.5 * (v0_d - z0hj_d * i_f)
                    S = S.at[driven, driven, :].set(jnp.asarray(b_ref_d / (a_fwd_d + 1e-30), dtype=_complex_dtype))
                    Z0_per_run = Z0_per_run.at[driven, :].set(z0_fit)
                    alpha_d = a_fwd_d * power_scales[driven]
                    if driven == 0:
                        beta_first = jnp.asarray(res_p["beta"], dtype=_complex_dtype)

            # Off-diagonal S21: S[j,i] = b_j / a_i (issue #80 stage S1).
            # The wave received from the structure at a passive port is
            # its BACKWARD wave b=(V-Z0*I)/2, not the forward wave a it
            # would launch. For a transmitted wave arriving at a port
            # whose forward reference faces the other way, a~0 and b~V;
            # using a gave the non-physical |S21|~0.08, b gives ~1.
            #
            # RETAINED as the fallback only. This single-ratio rule is
            # exact only when a_j = 0 at every passive port, and it is
            # not: |a_passive/a_driven| = 0.243-0.248 at the shipped
            # R = 50 ohm (0.19-0.93 across terminations), re-measured
            # on current main 2026-08-30 (#524, VESSL 369367257265,
            # PR #799; the July figure here before was 0.07-0.51), so
            # the far port's echo is reported as the structure's own
            # reflection (issue #507). The wave
            # amplitudes recorded below feed the multi-drive solve that
            # replaces this after the drive loop.
            for j in range(n_ports):
                if j == driven:
                    continue
                v0_p = v_per_port[j][0]
                b_out_p = 0.5 * (
                    v0_p - z0_hj_per_port[j] * i_first_per_port[j]
                ) * power_scales[j]
                S = S.at[j, driven, :].set(jnp.asarray(b_out_p, dtype=_complex_dtype) / (jnp.asarray(alpha_d, dtype=_complex_dtype) + 1e-30))

            # Record the FULL (a, b) pair at every port for this drive
            # (issue #507). ``a`` at a passive port is what the
            # single-ratio rule above assumes away. Relative sqrt(R0/Rj)
            # row scales give power waves up to one shared sqrt(R0),
            # which cancels from both S and cond(A).
            for j in range(n_ports):
                v0_j = v_per_port[j][0]
                z0_j = z0_hj_per_port[j]
                i_j = i_first_per_port[j]
                wave_a[driven][j] = jnp.asarray(
                    0.5 * (v0_j + z0_j * i_j) * power_scales[j], dtype=_complex_dtype)
                wave_b[driven][j] = jnp.asarray(
                    0.5 * (v0_j - z0_j * i_j) * power_scales[j], dtype=_complex_dtype)

        # ---- Multi-drive S solve (issue #507) -----------------------
        # Every port was driven, so the full wave system is recorded:
        #     A[j, d] = a_j during drive d      B[j, d] = b_j during d
        #     b = S a   for every drive   =>    S = B · A⁻¹
        # The single-ratio rule above (S[j,d] = b_j/a_d) is the d-th
        # column of this only when a_j = 0 at every passive port. It is
        # not: the far port reflects, and the exact algebra
        # b_1/a_1 = S11 + S12·(a_2/a_1) holds to machine precision, so
        # the echo was reported as the structure's own reflection. That
        # also makes |S11|²+|S21|² = 1 + |S11|² — the same power counted
        # twice — which is the passivity violation #507 opened on.
        # Same algebra as the coax lane's
        # solve_two_port_from_wave_amplitudes (#489), generalised to n
        # ports. cond(A) bounds DEGENERACY only; it is not a
        # reliability score.
        msl_assembly: str | None = None
        msl_cond_a = None
        if all(wave_a[d][j] is not None
               for d in range(n_ports) for j in range(n_ports)):
            import warnings as _w507
            S_solved, cond_a = msl_solve_s_from_waves(wave_a, wave_b)
            _bad = False
            if cond_a is not None:
                _bad = bool(np.any(~np.isfinite(
                    np.asarray(jax.lax.stop_gradient(S_solved))
                )))
                if _bad:
                    _w507.warn(
                        "compute_msl_s_matrix: the multi-drive S solve "
                        f"(issue #507) produced non-finite entries with "
                        f"cond(A) up to {float(np.max(cond_a)):.3g}; "
                        "keeping the single-ratio S for this run, which "
                        "carries the far port's echo in S11 (so expect "
                        "|S11|^2+|S21|^2 above 1). The drive matrix is "
                        "degenerate — check that each drive actually "
                        "excited its port and that the ports are not "
                        "mutually shadowed.",
                        stacklevel=2,
                    )
                elif float(np.max(cond_a)) > 1.0e3:
                    _w507.warn(
                        "compute_msl_s_matrix: the multi-drive S solve "
                        f"(issue #507) has cond(A) up to "
                        f"{float(np.max(cond_a)):.3g}. That bounds "
                        "DEGENERACY of the drive system, not accuracy: "
                        "the drive columns are nearly dependent, so S is "
                        "sensitive to the recorded wave amplitudes. "
                        "Check port isolation and settling_db.",
                        stacklevel=2,
                    )
            if not _bad:
                S = S_solved.astype(_complex_dtype)
            # Persist WHICH rule produced S (issue #523). A transient
            # warning is not enough: with the default
            # enforce_passivity=True the projection clips away the
            # fallback's own symptom (column power > 1), so a fallback
            # result can look healthy in every number a caller reads.
            # None while tracing — _bad cannot be evaluated on a tracer,
            # so the solve result is taken as-is and no claim is made.
            msl_assembly = (
                None if cond_a is None
                else ("single_ratio_fallback" if _bad
                      else "multi_drive_solve")
            )
            msl_cond_a = cond_a

        # Preserve the existing relative low-signal screen and S exactly.
        # Small passive-port wave pairs may be a true transmission zero;
        # this flag alone establishes neither matrix corruption nor an
        # absolute error bound. It remains separate from layout/settling.
        reliable = None
        try:
            # Cover every drive/port record (issue #522). An error in
            # a passive record can affect the solve even when own-drive
            # records are healthy: the planted corruption test changes
            # S21 by0.92 with cond(A)=1.28. Small magnitude alone does
            # not establish that such corruption occurred; exact
            # transmission zeros can also trip this screen.
            # Output shape and threshold remain unchanged: a port/bin
            # is False if at least one drive has both relative phasors
            # below the existing ten-percent floor.
            v_all = np.stack([
                np.asarray(jax.lax.stop_gradient(raw_v[d, p, 0, :]))
                for d in range(n_ports) for p in range(n_ports)
            ])
            i_all = np.stack([
                np.asarray(jax.lax.stop_gradient(raw_i1[d, p, :]))
                for d in range(n_ports) for p in range(n_ports)
            ])
            reliable = np.all(
                _msl_wave_split_reliability(
                    v_all, i_all, freqs_arr
                ).reshape(n_ports, n_ports, -1),
                axis=0,
            )
            _warn_msl_wave_split_unreliable(reliable, freqs_arr)
        except (jax.errors.ConcretizationTypeError, TypeError):
            # Diagnostics cannot materialize phasors while tracing.  The
            # eager forward result still carries the reliability mask.
            pass

        # β-scan rail flags for the SHIPPED fit numbers (issue #681):
        # Z0[i, :] comes from port i's OWN-drive run and beta from run 0
        # / port 0, so the own-drive diagonal of raw_beta_railed is
        # exactly the provenance of every fitted number the result
        # carries. A railed bin's Z0/beta are the ±35% scan-window
        # limit, not a measurement — used to be returned silently
        # pinned (repro: eps_eff 6.30 line reported as 4.60 at
        # 0.974·rail with zero warnings). S11/S21 never ride on the
        # fitted beta. That dataflow separation does not rule out
        # contamination of the V/I shared with the fit.
        beta_railed = None
        try:
            beta_railed = np.stack([
                np.asarray(
                    jax.lax.stop_gradient(raw_beta_railed[p, p, :])
                )
                for p in range(n_ports)
            ]).astype(bool)
            _warn_msl_beta_scan_railed(
                beta_railed, freqs_arr,
                tuple(pe.name for pe in entries),
            )
        except (jax.errors.ConcretizationTypeError, TypeError):
            # Cannot materialize while tracing; the eager forward
            # result still carries the mask.
            beta_railed = None

        # --- Honesty guard (issue #80 Fix A, retargeted in stage S1) ---
        # S11/S21 come from the OpenEMS-style V·I wave amplitudes, now
        # combined by the multi-drive solve (issue #507). |S11| > 1 on a
        # passive structure remains the primary red flag, but read it
        # correctly: the single-plane split itself was bounded by
        # construction, whereas the solve is not — it can exceed 1 when
        # the recorded wave system is inconsistent (a mismeasured
        # current's sign or scale, a degenerate drive matrix, or an
        # under-settled record). That is what raises under
        # strict_extractor=True, and cond(A) plus settling_db are the
        # two handles for telling those apart. The reported Z0 and beta
        # still ride on the retained N-probe fit, which can be noisy
        # per-frequency on coarse meshes, so a Z0 deviation from
        # analytic Hammerstad-Jensen is reported as a SEPARATE, softer
        # caveat. The fitted numbers do not enter S, but the V/I
        # records and analytic reference still need their own checks.
        import warnings as _w

        # Issue #726: ONE text for what probe-clearance corruption does,
        # shared with the preflight layout warning that used to contradict
        # this guard about the same condition. Imported, not copied.
        from rfx.preflight.msl import (
            MSL_PROBE_CLEARANCE_EFFECT,
            MSL_PROBE_CLEARANCE_GUIDANCE,
        )

        _S11_MAX = 1.0 + 0.05
        _Z0_TOL = 0.10
        for driven in range(n_ports):
            pe = entries[driven]
            z0_hj = z0_hj_per_port[driven]
            s11_abs = np.abs(np.asarray(jax.lax.stop_gradient(S[driven, driven, :])))
            k_s = int(np.argmax(s11_abs))
            s11_max = float(s11_abs[k_s])
            z0_dev = np.abs(np.asarray(jax.lax.stop_gradient(raw_z0[driven, driven, :])) - z0_hj) / z0_hj
            k_z = int(np.argmax(z0_dev))
            z0_dev_max = float(z0_dev[k_z])
            # Primary — V·I-split S11 boundedness (extraction soundness).
            if s11_max > _S11_MAX:
                # Cross-reference the standing-wave-null reliability mask (computed above): if
                # the peak-|S11| bin is flagged, signal strength is an
                # additional concern, not a diagnosis of the root cause.
                at_null = reliable is not None and not bool(np.asarray(reliable)[driven, k_s])
                cause = (
                    "the relative low-signal mask also flags this bin; "
                    "check signal uncertainty and the drive system"
                    if at_null else
                    "check current sign/scale, mode validity, settling and drive conditioning"
                )
                msg = (
                    f"compute_msl_s_matrix: V·I-split |S11| = "
                    f"{s11_max:.3f} > 1 for MSL port {pe.name!r} at "
                    f"f = {freqs_arr[k_s] / 1e9:.4f} GHz — non-physical "
                    f"for a passive structure. {cause}; the extracted "
                    "S11/S21 at this bin are UNRELIABLE."
                )
                if strict_extractor:
                    raise ValueError(msg)
                _w.warn(msg, stacklevel=2)
            # Secondary — reported-Z0 sanity (retained N-probe fit).
            if z0_dev_max > _Z0_TOL:
                # Issue #726 review P1b: the clearance story is only ONE of
                # this guard's causes, and on a satisfied port it is the
                # WRONG advice — the common trigger is coarse-mesh staircase
                # bias (#752), where "lengthen the feed" does nothing. Read
                # the same per-port record the result carries, by the same
                # index the entries tuple uses.
                _cl = (probe_clearance[driven]
                       if probe_clearance is not None
                       and driven < len(probe_clearance) else None)
                _cl_status = getattr(_cl, "status", None)
                if _cl_status == "insufficient":
                    _clearance_clause = (
                        " This port's probe clearance is INSUFFICIENT, so "
                        "that is a live cause here: "
                        + MSL_PROBE_CLEARANCE_EFFECT + " "
                        + MSL_PROBE_CLEARANCE_GUIDANCE)
                elif _cl_status == "satisfied":
                    _clearance_clause = (
                        " This port's probe clearance is satisfied, so "
                        "standing-wave content at the probes is not the "
                        "cause; refine the mesh or reconcile the rasterized "
                        "board with the declared one before quoting Z0.")
                else:
                    _clearance_clause = (
                        " This port's probe clearance could not be "
                        "evaluated"
                        + (f" ({_cl.note})" if getattr(_cl, "note", None)
                           else "")
                        + ", so probe-clearance corruption cannot be ruled "
                        "out as a cause; read probe_clearance and "
                        "beta_railed before interpreting Z0.")
                _w.warn(
                    f"compute_msl_s_matrix: reported Z0 for MSL port "
                    f"{pe.name!r} = "
                    f"{float(np.asarray(jax.lax.stop_gradient(raw_z0[driven, driven, k_z])).real):.2f} ohm deviates "
                    f"{z0_dev_max * 100:.1f}% from analytic Hammerstad-"
                    f"Jensen {z0_hj:.2f} ohm at "
                    f"f = {freqs_arr[k_z] / 1e9:.4f} GHz. Z0 rides on the "
                    "retained N-probe fit (S1 transitional); this can "
                    "reflect Yee-staircase bias, or that the rasterized "
                    "board (h_sub/W snapped to the lattice; see "
                    "sim.fidelity_report()) differs from the "
                    "declared one — not necessarily an extraction "
                    "fault (issue #752). Fitted Z0/beta are not used in "
                    "S11/S21, which use measured V/I and the analytic "
                    "Z0 anchor; this does not certify those inputs."
                    + _clearance_clause +
                    " Check settling_db, probe_clearance, beta_railed and "
                    "observation-plane sensitivity before interpreting S.",
                    stacklevel=2,
                )

        if raw_3probe_dump_path is not None:
            import json
            from pathlib import Path

            path = Path(raw_3probe_dump_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "schema": "rfx.msl_nprobe_dump",
                "schema_version": 4,
                "s_wave_convention": "power",
                "wave_definition": "a=(V+R*I)/(2*sqrt(R)); b=(V-R*I)/(2*sqrt(R))",
                "solver_wave_common_scale": "sqrt(reference_impedances[0]); cancels from S and cond(A)",
                "current_spatial_alignment": "linear_bracketing_H_to_E_node",
                "current_plane_stencils": h_stencils,
                "s_reference_impedances_ohm": z0_hj_per_port,
                "production_smatrix_schema": "S[receiver_port, driven_port, frequency_index]",
                "production_smatrix_stage": (
                    "PRE-passivity-projection raw extraction; "
                    "MSLSMatrixResult.S is the post-projection value "
                    "when enforce_passivity=True (default)"
                ),
                # v3 (issue #523): production_smatrix is no longer always
                # the N-probe-fit-derived S. Record WHICH assembly made
                # it, so a replayed dump cannot be misattributed.
                #
                # NB production_smatrix is written PRE-projection, so a
                # fallback dump does still carry the >1 column power
                # (MSLSMatrixResult.S is post-projection and does not).
                # The marker is not a substitute for that symptom — it is
                # more specific: >1 column power has several causes, only
                # one of which is the fallback.
                "production_smatrix_assembly": (
                    "unknown" if msl_assembly is None else msl_assembly
                ),
                "raw_v_shape": "(n_driven, n_ports, n_probes_max, n_freqs)",
                "raw_i1_shape": "(n_driven, n_ports, n_freqs)",
                "n_probes_per_port": [int(n) for n in n_probes_per_port],
                "phase_convention": "DFT accumulator convention from add_dft_plane_probe",
                "current_convention": "native_msl_loop_current",
                "deembedding": (
                    "N equally spaced voltage probes plus current at "
                    "probe 0. The reported Z0/beta come from the N-probe "
                    "least-squares wave-decomposition extractor (issue #80 "
                    "Fix C), which fits V_n = alpha*exp(-j beta x_n) + "
                    "gamma*exp(+j beta x_n) by SVD lstsq. The production "
                    "S-matrix does NOT come from that fit: it is solved "
                    "from the probe-0 wave amplitudes over all drives, "
                    "S = B @ inv(A) (issue #507), at the first probe "
                    "planes without translation to the feed planes, "
                    "with the modal voltage "
                    "spanning ground to the rasterized trace node (#511) "
                    "-- see production_smatrix_assembly for which rule "
                    "actually produced this dump's S"
                ),
                "grid": {
                    "dx_m": float(grid.dx),
                    "dt_s": float(grid.dt),
                    "nx": int(grid.nx),
                    "ny": int(grid.ny),
                    "nz": int(grid.nz),
                },
                "simulation": {
                    "freq_max_hz": float(self._freq_max),
                    "num_periods": float(num_periods),
                    "n_steps": None if n_steps is None else int(n_steps),
                },
                "port_definitions": [
                    {
                        "name": str(pe.name),
                        "position_m": [float(x) for x in pe.position],
                        "width_m": float(pe.width),
                        "height_m": float(pe.height),
                        "direction": pe.direction,
                        "impedance_ohm": float(pe.impedance),
                        "n_probe_offset": int(pe.n_probe_offset),
                        "n_probe_spacing": int(pe.n_probe_spacing),
                        "n_probes": int(pe.n_probes),
                        "mode": pe.mode,
                    }
                    for pe in entries
                ],
            }
            np.savez(
                path,
                metadata_json=np.asarray(json.dumps(metadata)),
                freqs_hz=np.asarray(freqs_arr, dtype=np.float64),
                raw_v=raw_v,
                raw_i1=raw_i1,
                raw_i1_left=raw_i1_left,
                raw_i1_same_index=raw_i1_same_index,
                raw_z0=raw_z0,
                raw_q=raw_q,
                production_smatrix=S,
                production_z0=Z0_per_run,
                production_beta=beta_first,
                port_names=np.asarray(tuple(pe.name for pe in entries), dtype=object),
                driven_port_indices=np.arange(n_ports, dtype=np.int64),
            )

        s_raw = None
        passivity_correction = None
        # Projection runs on the CONCRETE MEASUREMENT channel only:
        # never under tracing (min(sigma,1) zeroes/deforms the objective
        # gradient wherever the clip is active — measured, it flipped the
        # committed d|S|^2/d-eps sign gate), and never on the
        # eps_override channel even when concrete — otherwise a finite-
        # difference objective sees the projected function while
        # jax.grad sees the raw one, and the committed AD==FD gates
        # compare two different functions (review finding, PR #468).
        if enforce_passivity and eps_override is None and not is_tracer(S):
            s_projected, correction = _project_passive(S)
            if bool(np.any(np.asarray(correction) > 0.0)):
                s_raw = S
                passivity_correction = correction
                S = s_projected

        result = MSLSMatrixResult(
            S=S,
            freqs=np.asarray(freqs_arr),
            Z0=Z0_per_run,
            beta=beta_first,
            port_names=tuple(pe.name for pe in entries),
            reliable=reliable,
            settling_db=settling_db_runs,
            S_raw=s_raw,
            passivity_correction=passivity_correction,
            assembly=msl_assembly,
            cond_a=msl_cond_a,
            beta_railed=beta_railed,
            probe_clearance=probe_clearance,
            reference_impedances=np.asarray(z0_hj_per_port, dtype=np.float64),
        )
        _warn_if_ringdown_truncated(
            settling_db_runs,
            tuple(pe.name for pe in entries),
            num_periods=num_periods,
        )
        if passivity_correction is not None and not is_tracer(passivity_correction):
            _warn_if_passivity_projected(passivity_correction, freqs_arr)
        # The raw-extraction self-check still audits what was MEASURED:
        # run it on the unprojected matrix so the projection can never
        # silence the artifact diagnosis.
        import dataclasses as _dc

        audit_result = (
            result if s_raw is None else _dc.replace(result, S=s_raw)
        )
        _warn_if_nonpassive_smatrix(
            audit_result,
            extractor="compute_msl_s_matrix",
            strict=strict_extractor,
            passivity_tol=0.10,
        )
        return result
    finally:
        self._dft_planes = saved_dft
        if _had_dft_regions:
            self._dft_plane_regions = saved_dft_regions
        else:
            # keep the constructor-time attribute set clean (design-IR
            # classification test) — the dict lives only during a run
            self.__dict__.pop("_dft_plane_regions", None)
        self._msl_ports = saved_msl
        self._ports = saved_ports
        self._probes = saved_probes
        self._internal_probe_indices = saved_internal_probes


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# ``compute_msl_s_matrix`` was a ``def`` in the ``_SparamMixin`` class body,
# so its ``__qualname__`` read ``_SparamMixin.compute_msl_s_matrix``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``_SparamMixin.<name>`` -> ``Simulation.<name>`` at
# class-composition time so that a bad keyword argument reports
# ``Simulation.compute_msl_s_matrix() got an unexpected keyword argument``,
# and it SKIPS any function whose qualname does not match that pattern.
# Leaving the bare name here would therefore change that TypeError message
# -- a user-visible behaviour change in a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` pins it.
# ---------------------------------------------------------------------------
compute_msl_s_matrix.__qualname__ = "_SparamMixin.compute_msl_s_matrix"
