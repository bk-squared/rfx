"""MSL S-matrix calculators bound onto the public Simulation API.

Drive-specific declarations and run state belong to private declaration
variants. The caller retains its declared port and observer containers.
"""

from __future__ import annotations

from rfx.sparams import _msl_stages

from rfx.errors import AccuracyWarning, ConfigurationWarning, RfxRefusal, RfxUnsupported

from rfx._declarations import declaration_variant
from rfx._internal._precision import resolve_field_dtype
from rfx.preflight.realization import release_inspection

from rfx.preflight.line_stub import line_stub_admission as _line_stub_admit
from rfx._diagnostic_transport import merge_diagnostics, report_diagnostics, diagnostic_refusal

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.measurement.msl_planes import readable_values


def _reported(x, shape):
    """``x`` as float64 for the result's reported fields; NaN where a transform
    leaves no single value (a ``vmap`` batch, ``jit``)."""
    value = readable_values(x)
    if value is None or tuple(np.shape(value)) != tuple(shape):
        return np.full(shape, np.nan)
    return np.asarray(value, dtype=np.float64)
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
    _sigma_max_excess,
    _warn_if_sigma_max_excess,
    _warn_if_passivity_projected,
    _warn_if_nonpassive_smatrix,
    _warn_ntff_box_dropped,
    _register_msl_h_planes,
    _collocated_msl_h,
    _msl_power_wave_scales,
)
from rfx.sparams._port_reference import renormalize_power_waves


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
    enforce_passivity: bool = False,
    report_every: int | None = None,
) -> "MSLSMatrixResult":
    """Compute MSL S from probe-plane V/I with fitted line diagnostics.

    Returned S uses power waves referenced to each port's declared
    ``impedance=`` (default 50 ohm): a=(V+R*I)/(2*sqrt(R)),
    b=(V-R*I)/(2*sqrt(R)) with R the declared impedance (#2014). The waves
    are formed at the analytic Hammerstad-Jensen line Z0 and the matrix is
    then referred to R by ``renormalize_power_waves`` (exact for real
    references; nothing is computed where R equals the line Z0).
    ``reference_impedances`` records R; ``line_impedances`` records the
    line Z0 the waves were formed at, separately from the fitted ``Z0``.

    ``S`` is the RAW extracted S-matrix on every channel — concrete,
    ``eps_override``, traced — and the default ``enforce_passivity=False``
    is what makes that one statement instead of three. The projection
    below cannot run under tracing or on the ``eps_override`` channel (it
    would zero the gradient wherever the clip bites), so while it was the
    default the S a caller measured and the S ``jax.grad`` differentiated
    were different functions wherever it bit. One function, chosen to be
    the measured one (PI decision, 2026-09-21; the contract is
    ``docs/design_notes/chain_closure_contract.md``).

    A raw S may exceed the passive bound, and the result says so rather
    than hiding it: ``sigma_max_excess[k] = max(sigma_max(S(f_k)) - 1, 0)``
    is filled on every concrete call (``None`` only under tracing), and a
    warning names the bin count and the worst ``sigma_max`` whenever any
    bin exceeds 0. A passive structure cannot scatter more power than it
    receives, so such a bin is a measurement artifact — the usual causes
    are a record that ended before the structure rang down and a mesh too
    coarse for the geometry (``settling_db`` and ``reliable`` say which).

    ``enforce_passivity=True`` opts back into the post-process: the
    assembled S(f) is projected onto the passive set per frequency
    (singular values clipped to 1 — the nearest matrix in spectral norm
    with ``||S||_2 <= 1``), the unprojected matrix is kept in ``S_raw``,
    the per-bin clip amount in ``passivity_correction``, and a warning
    names the touched bins. It is constraint enforcement, not a physics
    fix, and it is NOT small where the raw extraction is bad: on a thru
    fixture whose raw sigma_max ran 1.19-1.91, projecting moved |S21|
    1.000 -> 0.61-0.72 and rotated its phase by up to 17 degrees, while
    ``Z0`` and ``beta`` stay raw. Never quote projected values as physics
    where ``passivity_correction`` is large.

    EXEMPTION (unchanged): even with ``enforce_passivity=True`` no
    projection is applied under tracing or on the ``eps_override``
    channel (traced or concrete), so that finite-difference and
    ``jax.grad`` objectives see the same function; ``S`` is then the raw
    extraction with ``S_raw``/``passivity_correction`` absent and no
    projection warning. ``sigma_max_excess`` is still filled on the
    concrete ``eps_override`` channel, so the bound violation stays
    visible there (measured sigma_max 1.18 on a coarse thru).

    Surface-impedance sheets (``add_thin_conductor(...,
    surface_impedance_f0=...)``) are supported on this lane:
    the sheet is realized node-thin by the per-step operator inside the
    ``run()``/``forward()`` device dispatches — no lane-level ctx is
    built here. Combination refusals fire downstream at whichever lane
    entry the call actually reaches, and the two channels carry their
    OWN copies: on ``run()`` the dispersive-substrate and
    ``boundary='upml'`` refusals come from ``run_uniform`` with run-lane
    wording, while the ``forward()`` / ``eps_override`` channel never
    enters ``run_uniform`` and raises instead from the forward-lane
    entry in ``rfx/api/_execute.py``, naming that lane. The subgridded and distributed lanes
    carry their own call sites at whichever entry reaches them —
    ``FENCE_REGISTRY`` in ``tests/unit/materials/test_sheet_impedance.py`` is the
    authoritative per-lane list, AST-guarded against drift.
    ``subpixel_smoothing`` / ``conformal_pec`` are ``run()``-only
    keywords, so that combination is unreachable on the ``eps_override``
    channel rather than refused there.
    The TRACE must remain PEC: an f0 sheet realizes no PEC edge, and
    the closed Ampere-loop current and the V span anchor on the
    realized PEC wall planes (a PEC trace may be a volume
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
    current main 2026-08-30 (VESSL 369367257265) — so the S-matrix is
    recovered by solving the
    full wave system ``S = B·A⁻¹`` over all drives rather than by the
    per-column ratio ``b_j/a_d`` (the ratio reports the far
    port's echo as the structure's own reflection).  At each port
    ``n_probes`` downstream DFT plane probes record Ez and the first
    probe also records the transverse H components for a closed
    Ampere-loop current. Production S is formed from this first-plane
    V/I: the waves at the analytic Hammerstad-Jensen line Z0, then referred
    to each port's declared ``impedance=`` (see above). The
    N-probe least-squares fit of
    ``V_n = α e^{-jβx_n} + γ e^{+jβx_n}`` supplies only the reported
    beta/Z0 diagnostics. Those fitted values do not enter S. S is formed
    at each port's first probe plane and then referred back to the
    DECLARED FEED PLANE, ``S_feed[i, j] = D_i S[i, j] D_j`` with
    ``D_p = exp(-j β_p L_p)``, ``L_p`` the realized feed-to-probe-0
    distance (``reference_plane_offsets_m``) and ``β_p`` the closed-form
    β (Hammerstad-Jensen ε_eff(0) with Kirschning-Jansen dispersion, from
    the port's width, height and substrate εr; ``shift_beta``). β is
    real, so the shift moves phase only. Measured against the simulated
    line's two-line β: within 0.76 % on the resolved boards measured
    (chain-battery thru at 100 µm, 50 µm and z-graded; εr 10.2 / h 0.6 mm
    board), phase error at most 5.5e-3 rad; within 1.23 % on a 4-cell
    substrate (0.8 mm εr 2.2, dx 0.2 mm), phase error at most 1.9e-2 rad
    over 2-18 GHz (1.33e-2 up to 15 GHz) (rfx-archive MSL-PLANE.md,
    decision records 10-15). The phase-only shift is exact for a line whose
    Zc equals the reference R (the closed-form Z0). Under ``eps_override``
    β and that Z0 come from the permittivity the override sets under the
    port line (traced, so the gradient carries the plane term), the shift
    is applied in that line's reference, and S is then referred to
    ``impedance=``; an override that varies between a feed plane and its
    probe 0 is refused by name, as is a call under ``jax.jit``, where its
    values cannot be checked (#2199). The two bracketing H planes are linearly
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
        Honesty guard for the de-embedding. After
        extraction, the per-frequency ``|q|`` and extracted ``Z0``
        are validated against physical bounds (``|q| <= 1`` for a passive
        line; extracted ``Z0`` within 10 % of the analytic
        Hammerstad-Jensen value). When ``False`` (default) a violation
        raises a loud :func:`warnings.warn`; when ``True`` it raises
        :class:`ValueError` instead. With the N-probe extractor
        ``|q|`` and ``Z0`` should be healthy so this rarely fires — it
        is the safety net for pathological geometries.
    enforce_passivity : bool
        Opt into the passivity projection as a post-process. ``False``
        (default) returns the raw extracted ``S``; see the section above
        for what each setting costs and which channels the projection can
        reach. Changed from ``True`` on 2026-09-21.
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
        not segment count). Forwarded to :meth:`forward`; raises
        ``NotImplementedError`` on the uniform path — use
        ``checkpoint_segments`` there.
    report_every : int or None
        Progress reporting for long solves. Forwarded to
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
        Field meanings are on :class:`rfx.api._spec.MSLSMatrixResult`.

        **``assembly``: read this before trusting a fallback result.** The
        fallback's characteristic symptom is column power above 1, which the
        default ``enforce_passivity=False`` leaves standing in ``S`` and
        ``sigma_max_excess`` measures. Under ``enforce_passivity=True``
        the symptom is clipped out of ``S`` — but it is not erased from
        the result: ``passivity_correction`` records how much was clipped
        and ``S_raw`` keeps the unprojected matrix, and the run also emits
        both a fallback warning and a passivity-guard
        warning. So a fallback is not silent; ``assembly`` is simply the
        *specific* signal. Column power above 1 has several causes (an
        under-settled record, a standing-wave null, a mis-scaled current) and
        only one of them is the fallback — that is what ``assembly``
        disambiguates. It is ``None`` while tracing (the finiteness test
        cannot run on a tracer, so the solve result is taken as-is — see
        ``cond_a``).
    """
    resolve_field_dtype(self._precision)
    _diagnostics = ()
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

    is_nonuniform = _msl_stages._validate_msl_family(
        RfxRefusal, RfxUnsupported, _diagnostics, diagnostic_refusal, self,
    )

    # Issue #704: an NTFF box would be silently dropped on this path.
    _warn_ntff_box_dropped(self, "compute_msl_s_matrix()")

    entries = list(self._msl_ports)
    n_ports = len(entries)

    # Probe placement, material assembly and each run share the resolved
    # mesh. Missing z profiles are synthesized locally by the grid builder;
    # writing a derived profile into the declaration would freeze auto-mesh
    # state and change the resolved domain during the driver.
    _line_stub_scope = _line_stub_admit(self, freqs)
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

    msl_ports, n_probes_per_port, probe_xs, h_stencils, probe_clearance = _msl_stages._prepare_msl_probes(
        entries, grid, msl_h_plane_stencil, msl_port_from_entry, msl_probe_x_coords_n, self,
    )
    dz_arr, port_idx_meta, beta0_per_port, z0_hj_per_port, eps_r_ref_per_port, line_eps_per_port, declared_refs, trace_k_per_port = _msl_stages._prepare_msl_geometry(
        _msl_cell_profile, entries, eps_override, freqs_arr, grid, hammerstad_jensen_z0_eps_eff, is_nonuniform,
        jax, msl_ampere_pair, msl_cell, msl_cross_section_span, msl_ports, n_ports,
        self,
    )

    # Prepare private witness declarations, then a fresh variant per drive.
    observers = declaration_variant(
        self, _internal_probe_indices=set(self._internal_probe_indices))
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

    _witness_base, _witness_total, settling_db_runs, settling_details, observers = _msl_stages._prepare_msl_witnesses(
        entries, msl_physical_point, n_ports, observers, port_idx_meta, probe_xs,
    )

    for driven in range(n_ports):
        variant, ez_probe_names, h_probe_names = _msl_stages._prepare_msl_drive(
            GaussianPulse, _MSLPortEntry, _register_msl_h_planes, declaration_variant, driven, entries, freqs_arr,
            h_stencils, jnp, msl_ports, n_ports, observers, port_idx_meta, probe_xs,
            self, trace_k_per_port,
        )

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
                    "trace.",
                    ConfigurationWarning, stacklevel=2,
                )
            fwd_result = variant.forward(
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
            result = variant.run(
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

        _diagnostics = merge_diagnostics(_diagnostics, report_diagnostics(_ts_result))

        settling_db_runs, settling_details = _msl_stages._score_msl_record(
            _ts_result, _witness_base, _witness_total, driven, dz_arr, ez_probe_names, freqs_arr,
            grid, h_probe_names, h_stencils, is_tracer, msl_ports, port_idx_meta,
            self, settling_db_runs, settling_details, trace_k_per_port, variant,
        )



        # Helper: integrate V and I per port from the recorded planes.
        v_per_port: list[list[np.ndarray]] = []
        i_first_per_port: list[np.ndarray] = []
        for p_idx, meta in enumerate(port_idx_meta):
            i_f, raw_i1_left, raw_i1_same_index, v_per_port, i_first_per_port = _msl_stages._integrate_msl_port(
                _collocated_msl_h, _complex_dtype, driven, dz_arr, ez_probe_names, h_probe_names, h_stencils,
                i_first_per_port, jnp, meta, msl_loop_current, msl_modal_voltage, msl_ports, p_idx,
                planes, raw_3probe_dump_path, raw_i1_left, raw_i1_same_index, trace_k_per_port, v_per_port, variant,
            )
            raw_v, raw_i1, raw_z0, raw_q, raw_beta_railed, res_p, z0_fit = _msl_stages._fit_msl_port(
                _complex_dtype, beta0_per_port, driven, extract_msl_nprobe, i_f, jnp, msl_axis_roles,
                msl_ports, n_probes_per_port, p_idx, probe_xs, raw_beta_railed, raw_i1,
                raw_q, raw_v, raw_z0, v_per_port, z0_hj_per_port,
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
                    "compute_msl_s_matrix: the multi-drive S solve"
                    f" produced non-finite entries with "
                    f"cond(A) up to {float(np.max(cond_a)):.3g}; "
                    "keeping the single-ratio S for this run, which "
                    "carries the far port's echo in S11 (so expect "
                    "|S11|^2+|S21|^2 above 1). The drive matrix is "
                    "degenerate — check that each drive actually "
                    "excited its port and that the ports are not "
                    "mutually shadowed.", AccuracyWarning,
                    stacklevel=2,
                )
            elif float(np.max(cond_a)) > 1.0e3:
                _w507.warn(
                    "compute_msl_s_matrix: the multi-drive S solve"
                    f" has cond(A) up to "
                    f"{float(np.max(cond_a)):.3g}. That bounds "
                    "DEGENERACY of the drive system, not accuracy: "
                    "the drive columns are nearly dependent, so S is "
                    "sensitive to the recorded wave amplitudes. "
                    "Check port isolation and settling_db.", AccuracyWarning,
                    stacklevel=2,
                )
        if not _bad:
            S = S_solved.astype(_complex_dtype)
        # Persist WHICH rule produced S (issue #523). A transient
        # warning is not enough: under enforce_passivity=True the
        # projection clips away the fallback's own symptom (column
        # power > 1), so a fallback result can look healthy in every
        # number a caller reads.
        # None while tracing — _bad cannot be evaluated on a tracer,
        # so the solve result is taken as-is and no claim is made.
        msl_assembly = (
            None if cond_a is None
            else ("single_ratio_fallback" if _bad
                  else "multi_drive_solve")
        )
        msl_cond_a = cond_a

    S, msl_planes, s_line_refs = _msl_stages._shift_msl_reference_planes(
        S, _complex_dtype, _reported, entries, eps_r_ref_per_port, freqs_arr,
        grid, jnp, line_eps_per_port, msl_ports, probe_xs, renormalize_power_waves, z0_hj_per_port,
    )

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

    _msl_stages._check_msl_extraction(
        RfxRefusal, S, _diagnostics, diagnostic_refusal, entries, freqs_arr,
        jax, n_ports, probe_clearance, raw_z0, reliable, strict_extractor,
        z0_hj_per_port,
    )

    _msl_stages._dump_msl_record(
        S, Z0_per_run, _reported, beta_first, declared_refs, entries, freqs_arr,
        grid, h_stencils, msl_assembly, msl_planes, n_ports, n_probes_per_port, n_steps,
        num_periods, raw_3probe_dump_path, raw_i1, raw_i1_left, raw_i1_same_index, raw_q,
        raw_v, raw_z0, s_line_refs, self,
    )

    # #2014: refer S from the line Z0 the waves were formed at to the
    # declared impedance=, once, before the opt-in projection below.
    S = renormalize_power_waves(S, s_line_refs, declared_refs)

    s_raw = None
    passivity_correction = None
    # Measure the bound violation of the RAW extraction before any
    # opt-in projection can hide it. Concrete channels only — a
    # tracer has no singular values — but that includes the concrete
    # eps_override channel, which the projection below never reaches.
    sigma_max_excess = (
        None if is_tracer(S) else _sigma_max_excess(S)
    )
    # Projection runs on the CONCRETE MEASUREMENT channel only:
    # never under tracing (min(sigma,1) zeroes/deforms the objective
    # gradient wherever the clip is active — measured, it flipped the
    # committed d|S|^2/d-eps sign gate), and never on the
    # eps_override channel even when concrete — otherwise a finite-
    # difference objective sees the projected function while
    # jax.grad sees the raw one, and the committed AD==FD gates
    # compare two different functions (review finding, PR #468).
    # Those two exemptions are why it is no longer the default: a
    # post-process that CANNOT run on two of the three channels
    # cannot be the shipped definition of S (PI, 2026-09-21).
    if enforce_passivity and eps_override is None and not is_tracer(S):
        s_projected, correction = _project_passive(S)
        if bool(np.any(np.asarray(correction) > 0.0)):
            s_raw = S
            passivity_correction = correction
            S = s_projected

    result = MSLSMatrixResult(
        diagnostics=_diagnostics,
        S=S,
        freqs=np.asarray(freqs_arr),
        Z0=Z0_per_run,
        beta=beta_first,
        port_names=tuple(pe.name for pe in entries),
        reliable=reliable,
        settling_db=settling_db_runs,
        settling_witness=tuple(settling_details),
        S_raw=s_raw,
        passivity_correction=passivity_correction,
        sigma_max_excess=sigma_max_excess,
        assembly=msl_assembly,
        cond_a=msl_cond_a,
        beta_railed=beta_railed,
        probe_clearance=probe_clearance,
        reference_impedances=declared_refs.copy(),
        line_impedances=_reported(s_line_refs, (n_ports,)),
        reference_plane_offsets_m=np.asarray(msl_planes.offsets_m),
        shift_beta=np.asarray(msl_planes.shift_beta),
    )
    _warn_if_ringdown_truncated(
        settling_db_runs,
        tuple(pe.name for pe in entries),
        num_periods=num_periods, witnesses=settling_details,
    )
    if passivity_correction is not None and not is_tracer(passivity_correction):
        _warn_if_passivity_projected(passivity_correction, freqs_arr)
    elif sigma_max_excess is not None:
        # Nothing was clipped, so whatever excess the extraction has
        # is in the matrix being returned. One warning, not two: the
        # projection warning above already reports the same bins.
        _warn_if_sigma_max_excess(
            sigma_max_excess, freqs_arr,
            extractor="compute_msl_s_matrix",
            projection_requested=bool(enforce_passivity),
        )
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
    release_inspection(self)
    return result



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
