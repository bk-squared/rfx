"""Module-level S-matrix helpers, moved verbatim out of ``rfx.api._sparams``.

Issue #980 Phase 2, step 1. ``rfx/api/_sparams.py`` had grown to 8 647 lines:
2 590 lines of module-level helpers followed by one 5 957-line mixin class.
This module is the helper half, relocated byte for byte — same order, same
text, same indentation, same docstrings. Nothing was renamed, reordered,
cleaned up or rewritten, and the move is gated on bit identity of the S arrays
every public ``compute_*`` leg returns
(``tests/locks/test_sparams_split_bit_identity.py``).

``rfx.api._sparams`` re-exports every name below explicitly, so
``from rfx.api._sparams import <helper>`` — which 47 files do — keeps working,
and so do the string monkeypatches that patch ``"rfx.api._sparams.<name>"``.
New code should import from here.

Import contract, inherited from ``rfx.api._sparams``: this module must import
ONLY external ``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` (that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic). The helper bodies' function-local ``from rfx.api._preflight import
...`` imports are inside functions, exactly as they were, and so do not run at
import time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

try:  # Public API on current JAX; the 0.4.x GPU image uses the old location.
    from jax import enable_x64 as _enable_x64
except ImportError:
    from jax.experimental import enable_x64 as _enable_x64

from rfx.core.jax_utils import is_tracer

from rfx.nonuniform import NonUniformGrid, interior_cells


def _msl_cell_profile(grid, axis: str, n: int) -> np.ndarray:
    """Per-cell size array (length ``n``, full/padded) along ``axis`` for
    MSL V/I integration. Graded-mesh aware.

    ``NonUniformGrid`` (a NamedTuple) stores per-cell spacings as
    ``dx_arr`` / ``dy_arr`` / ``dz`` and exposes NO ``*_profile``
    attributes — so the legacy ``getattr(grid, "dy_profile", None)``
    fell through to ``np.full(n, grid.dx)``, i.e. the SCALAR boundary-x
    cell for every transverse cell (wrong axis AND scalar-not-per-cell).
    This reads the real per-cell array on a NU grid. On a uniform
    ``Grid`` it is byte-identical to the legacy path (``Grid`` is not a
    ``NonUniformGrid``, so the per-cell branch is never taken): the
    ``*_profile`` attr if present, else ``np.full(n, grid.dx)`` — the
    legacy behaviour of using ``grid.dx`` for every axis is preserved.
    """
    if isinstance(grid, NonUniformGrid):
        per_cell = {"x": grid.dx_arr, "y": grid.dy_arr, "z": grid.dz}[axis]
        a = np.asarray(per_cell, dtype=float)
        if a.shape != (n,):
            # The NU branch is authoritative — never silently fall back to a
            # scalar boundary-dx fill (that is the exact wrong-number bug this
            # helper exists to fix). A shape mismatch is a wiring error.
            raise ValueError(
                f"NonUniformGrid {axis} per-cell profile shape {a.shape} "
                f"!= expected ({n},)."
            )
        return a
    attr = {"x": "dx_profile", "y": "dy_profile", "z": "dz_profile"}[axis]
    prof = getattr(grid, attr, None)
    if prof is not None:
        return np.asarray(prof, dtype=float)
    return np.full(n, float(grid.dx), dtype=float)


def msl_modal_voltage(ez_plane, *, j_centre: int, k_lo: int, k_hi: int,
                      dz_arr, dtype=None):
    """Native MSL voltage ``V = +∫E·dz`` from ground to the trace underside.

    This is opposite the trace-to-ground potential convention in the
    quasi-TEM limit. The native positive-going MSL current is also opposite
    the physical conductor current; the paired signs preserve V*conj(I)
    and the wave ratios. Do not change only one of these signs.

    ``ez_plane`` is an ``(n_freqs, ny, nz)`` x-normal DFT accumulator.  The
    integral sums the z-edges ``k_lo .. k_hi-1`` — every Ez edge strictly
    below the trace conductor.

    ``k_hi`` is EXCLUSIVE and must be the **bottom node of the rasterized
    trace conductor** — in the S-matrix lanes, ``trace_k_per_port[p][0]``,
    i.e. the same PEC-mask search the Ampère-loop current uses, so V and I
    reference one conductor plane by construction.

    Do NOT pass ``round(h_sub/dx)`` (the port-height rounding,
    ``port_idx_meta["k_hi"]``) as a proxy.  ``Box`` rasterization is
    half-open over node coordinates, so for ``frac(h_sub/dx) ∈ (0, 0.5)``
    the trace lands at ``ceil(h_sub/dx) = round(h_sub/dx) + 1`` and the
    proxy is one substrate edge SHORT.  Measured on the dx = 80 µm gate
    fixture (``h_sub/dx = 3.175``): trace node 4, proxy 3 — the proxy span
    dropped the in-phase edge 3 and anchored V and I on different conductor
    planes (PR #516 review, finding F2).  On node-aligned meshes
    (``h_sub/dx`` integral, e.g. dx = h_sub/3 = 84.67 µm) the two agree.

    Issue #511: before this helper existed the span was
    ``range(k_lo, k_hi + 1)`` with the rounding proxy — on aligned meshes
    that is ``n+1`` edges for an ``n``-cell substrate, and the extra edge
    lies inside the one-cell PEC trace, where the realization leaves the
    NORMAL E component live (#931 §1.2: an E component is PEC iff its own
    location is inside the closed conductor region, so the normal edge of
    a SHEET stays live and the normal edge INSIDE a volume is shorted)
    — a correct boundary condition that is wrong to sum into a
    ground-to-trace potential difference.  It contributed roughly −12% at
    ``∠ ≈ 180°``, so every quantity derived from ``V`` (``Z0``, ``S11``,
    ``S21``, the N-probe fit, the two-plane invariant) carried a
    common-mode bias.  The extractor-independent witness is the Poynting
    flux: ``Re(V·conj(I)) / flux_spectrum`` measured 0.881-0.885 with the
    old span and 1.006-1.009 with the corrected span, over 7 planes × 12
    frequencies — **on the aligned dx = 84.67 µm mesh**; that identity is
    the falsifier for THIS span (ground→trace underside) and holds only
    when the top anchor is the true trace node.

    The BISECTING mesh (dx = 80 µm, the mesh class this span anchoring
    actually changes behaviour on) has since been measured too, on a
    properly terminated two-port fixture (issue #520 leg 1;
    ``scripts/diagnostics/msl_vi_flux_oracle.py`` + its committed JSON):
    HELD, ratio 1.0105-1.0118, 30/30 admissible cells. An earlier
    unterminated single-port reading of this same identity on the
    bisecting mesh (issue #525) read low (0.54-0.69) for two compounding,
    non-extractor reasons — a convention slip (see #525's own correction)
    and a reactive, non-travelling fixture (reactive fraction 0.963-0.995,
    PR #549 review reconstruction) — neither of which is present in the
    committed measurement.
    """
    if k_hi <= k_lo:
        raise ValueError(
            f"msl_modal_voltage: need at least one substrate edge, got "
            f"k_lo={k_lo}, k_hi={k_hi}. k_hi is the rasterized trace's "
            f"bottom node (exclusive), so a port whose height rasterises "
            f"to zero substrate cells cannot define a modal voltage — "
            f"refine the mesh or raise the port height."
        )
    v = jnp.zeros(ez_plane.shape[0],
                  dtype=ez_plane.dtype if dtype is None else dtype)
    for k in range(k_lo, k_hi):
        v = v + ez_plane[:, j_centre, k] * float(dz_arr[k])
    return v


def msl_solve_s_from_waves(wave_a, wave_b):
    """Solve ``S = B·A⁻¹`` from wave amplitudes recorded on every drive.

    ``wave_a[d][j]`` / ``wave_b[d][j]`` are ``(n_freqs,)`` forward / backward
    wave amplitudes at port ``j`` while port ``d`` was driven.  With every
    port driven the system ``b = S a`` is square:

        ``A[j, d] = a_j`` during drive ``d``,  ``B[j, d] = b_j`` likewise
        ``S = B · A⁻¹``

    Returns ``(S, cond_a)`` with ``S`` shaped ``(n_ports, n_ports, n_freqs)``
    indexed ``[receiver, driven, freq]``, and ``cond_a`` the per-frequency
    condition number of ``A`` (``None`` under tracing).

    Issue #507: the superseded rule ``S[j, d] = b_j / a_d`` is the ``d``-th
    column of this only when ``a_j = 0`` at every passive port.  It is not —
    ``|a_passive/a_driven| = 0.243-0.248`` at the shipped R = 50 Ω and
    0.19-0.93 across terminations, re-measured on current main 2026-08-30
    (#524, VESSL 369367257265, PR #799; the July pre-#511/#516 figure was
    0.07-0.51 and is superseded) — and
    the exact algebra ``b_1/a_1 = S11 + S12·(a_2/a_1)`` holds to machine
    precision, so the far port's echo was reported as the structure's own
    reflection.

    Unitarity is therefore lost.  How it is lost depends on phase: expanding
    for a symmetric ``S`` gives
    ``(1+γ²)(|S11|²+|S21|²) + 4γ·Re(S11·conj(S21))`` with ``γ = a_2/a_1``, so
    the old rule can push column power either side of 1.  In the NEAR-MATCHED
    case that the thru fixtures are (true ``S11 ≈ 0``) it reduces to
    ``|S11|² + |S21|² = 1 + γ²`` — the same power counted twice — which is the
    passivity violation #507 opened on.  Both cases are pinned in
    ``tests/unit/sparams/test_msl_modal_voltage_and_wave_solve.py``.

    ``cond_a`` bounds DEGENERACY of the drive system only — it is not a
    reliability score.  Same contract as the coax lane's
    :func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`
    (issue #489), generalised to ``n`` ports.
    """
    n_ports = len(wave_a)
    A = jnp.stack([
        jnp.stack([wave_a[d][j] for d in range(n_ports)], axis=-1)
        for j in range(n_ports)
    ], axis=-2)                                   # (n_freqs, n_ports, n_ports)
    B = jnp.stack([
        jnp.stack([wave_b[d][j] for d in range(n_ports)], axis=-1)
        for j in range(n_ports)
    ], axis=-2)
    # S = B A⁻¹  <=>  Sᵀ = (Aᵀ)⁻¹ Bᵀ, batched over frequency.
    S_solved = jnp.swapaxes(
        jnp.linalg.solve(jnp.swapaxes(A, -1, -2), jnp.swapaxes(B, -1, -2)),
        -1, -2,
    )
    cond_a = None
    if not is_tracer(A):
        # f64 BEFORE the ratio: for a complex64 A the SVD returns float32,
        # where the 1e-300 floor underflows to 0.0 (NEP-50 weak promotion
        # keeps float32) and a singular A divides by zero instead of
        # saturating. Same failure class as the #497 NEP-50 fallback.
        _sv = np.linalg.svd(
            np.asarray(jax.lax.stop_gradient(A)), compute_uv=False
        ).astype(np.float64)
        cond_a = np.asarray(_sv[..., 0] / np.maximum(_sv[..., -1], 1e-300))
    return jnp.moveaxis(S_solved, 0, -1), cond_a


def _msl_wave_split_reliability(
    voltages: object,
    currents: object,
    freqs: object,
) -> np.ndarray:
    """Return the per-port reliability mask for an MSL V·I wave split."""
    v_abs = np.abs(np.asarray(voltages))
    i_abs = np.abs(np.asarray(currents))
    freqs_arr = np.asarray(freqs)
    if v_abs.shape != i_abs.shape or v_abs.ndim != 2:
        raise ValueError(
            "MSL reliability phasors must have matching (n_records, n_freqs) "
            "shapes; the production caller passes n_records = n_ports**2, one "
            "row per (driven, port) pair"
        )
    if v_abs.shape[1:] != freqs_arr.shape:
        raise ValueError("MSL reliability phasors and frequency grid do not align")

    v_floor = 0.1 * np.median(v_abs, axis=1, keepdims=True)
    i_floor = 0.1 * np.median(i_abs, axis=1, keepdims=True)
    return ~((v_abs < v_floor) & (i_abs < i_floor))


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
        "low signal at an MSL port plane: "
        f"{affected_freqs.size} bins in [{f1:.4f}, {f2:.4f}] GHz "
        "have both |V| and |I| below 10% of their record's band medians. "
        "This flags relative signal strength, not proof of an incorrect "
        "S-matrix; a true transmission zero can also trigger it. Check "
        "signal uncertainty, settling, drive conditioning and probe "
        "geometry before using these bins.",
        stacklevel=2,
    )


def _warn_msl_beta_scan_railed(
    beta_railed: np.ndarray, freqs: object, port_names: tuple
) -> None:
    """Emit one aggregate warning for β-scan rail-pinned bins (issue #681).

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
    warnings.warn(
        "N-probe beta scan pinned at its own window limit: "
        f"{affected_freqs.size} bins in [{f1:.4f}, {f2:.4f}] GHz at "
        f"port(s) {ports} minimized the fit residual at the edge of the "
        "±35% scan around the analytic Hammerstad-Jensen guess — the "
        "reported Z0/beta at those bins are the scan limit, NOT a "
        "measurement (issue #681). Fitted Z0/beta are not used in S11/S21, "
        "which use measured V/I and the analytic Z0 anchor; this does not "
        "certify those inputs. Common causes: the real eps_eff is far from "
        "the HJ estimate (wrong eps_r_sub / substrate not detected under "
        "the port), or a contaminated/under-settled record at those bins "
        "(check settling_db and the reliable mask). Check the result's "
        "beta_railed mask before quoting Z0 or beta.",
        stacklevel=2,
    )


_SETTLING_WITNESS_DB = -40.0


def settling_verdict(settling_db) -> str:
    """The ONE -40 dB ring-down comparison in this codebase (issue #885).

    Returns ``"pass"``, ``"fail"`` or ``"absent"``. ``"absent"`` covers
    ``None`` and every non-finite value (NaN, +/-inf) -- the states that mean
    "no witness was established", which a bare ``> -40`` comparison silently
    turns into a pass. That is not hypothetical: a campaign driver read a
    missing witness as ``np.nan`` through ``getattr(result, "settling_db",
    np.nan)``, compared it to the bar, and a completed 3-hour two-arm
    experiment came back INCONCLUSIVE for an instrument reason (#885).

    Every caller that needs the decision calls this instead of writing the
    comparison again -- including the aggregate warner below and the
    ``run()`` post-run advisories in ``rfx/api/_execute.py``. A value exactly
    at the bar passes (the bar is the inclusive limit, unchanged since #538).
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
    """Light re-validation of ``extra_flux_monitors=`` entries (#589 opt-in).

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


def _warn_if_ringdown_truncated(
    settling_db: np.ndarray,
    port_names: tuple,
    *,
    num_periods: float | None = None,
    n_steps: int | None = None,
    drive_labels: tuple | None = None,
    consequence: str | None = None,
    quoted_thing: str = "any S value",
) -> None:
    """Emit one aggregate warning when a driven run's record is truncated.

    The witness makes the project's ring-down settling rule mechanical
    (docs/guides/simulation_methodology.md): end/peak
    Ez^2 at the port probe planes, per driven run. Above −40 dB the fixed
    record ended while the structure was still ringing, so
    the single-bin DFTs underlying V/I — and every S value of that run —
    integrate a truncated transient. Measured consequence on the Sheen-1990
    LPF (dx=200 µm, resonant stopband): num_periods=20 left the witness hot
    and produced |S| column-power poles up to ~1.8e3 that shrank
    monotonically as the record grew (20→60 periods: worst pole 62→8.8),
    while absorber depth (8→24 CPML layers) did not move them.

    Every lane that computes ``settling_db`` must route it through here
    (issue #662: the coax two-port and coax<->MSL transition lanes computed
    the witness, documented the −40 dB bar in their result docstrings, and
    never compared the two — a caller reading ``.s_params`` got a
    plausible-looking truncation artifact in silence). Pass whichever
    record-length knob the lane is actually driven by: ``num_periods`` for
    the waveguide/MSL/mixed lanes, ``n_steps`` for the coax lanes — the
    warning names that knob so its remedy is directly actionable.

    NaN entries are skipped by the finite mask, which is what keeps the
    differentiable paths quiet: they leave ``settling_db`` NaN on purpose
    (the witness needs a concrete time series, and a traced one cannot be
    Python-branched on). All callers pass a concrete host-side NumPy array,
    so nothing here branches on a tracer.

    All violating drives are named, not just the worst: the record length is
    a per-drive property with a per-drive remedy, and naming only the worst
    would hide a second drive needing the same fix. It stays ONE warning per
    call (issue #470: per-probe advisory flooding buried the genuine ones).
    """
    values = np.atleast_1d(np.asarray(settling_db, dtype=float))
    hot = np.array([settling_verdict(v) == "fail" for v in values], dtype=bool)
    if not bool(np.any(hot)):
        return

    import warnings

    if num_periods is not None:
        knob, record = "num_periods", f"num_periods={num_periods:g}"
    elif n_steps is not None:
        knob, record = "n_steps", f"n_steps={int(n_steps):d}"
    else:  # pragma: no cover - guarded by the call sites
        knob, record = "the record length", "fixed-length"
    def _label(i: int) -> str:
        if drive_labels is not None:
            return drive_labels[i] if i < len(drive_labels) else str(i)
        return f"port {port_names[i] if i < len(port_names) else i} driven"

    per_run = ", ".join(
        f"{_label(i)}: {values[i]:+.1f} dB" for i in np.flatnonzero(hot)
    )
    if consequence is None:
        consequence = (
            "the DFT-based S-parameters of the affected run(s) are "
            "truncation artifacts wherever the structure is resonant — expect "
            "spurious |S| poles and passivity violations"
        )
    warnings.warn(
        "ring-down settling witness FAILED (end/peak energy above "
        f"{_SETTLING_WITNESS_DB:.0f} dB): {per_run}. The {record} "
        "record ended while the structure was still "
        f"ringing, so {consequence}. Increase "
        f"{knob} until the witness is below −40 dB before quoting "
        f"{quoted_thing} (see the result's settling_db field).",
        stacklevel=2,
    )


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
    if axis == "x":
        d_full = grid.dx_arr_f64 if getattr(grid, "dx_arr_f64", None) is not None else grid.dx_arr
        pad_lo, pad_hi = int(grid.pad_x_lo), int(grid.pad_x_hi)
    elif axis == "y":
        d_full = grid.dy_arr_f64 if getattr(grid, "dy_arr_f64", None) is not None else grid.dy_arr
        pad_lo, pad_hi = int(grid.pad_y_lo), int(grid.pad_y_hi)
    else:
        d_full = grid.dz_f64 if getattr(grid, "dz_f64", None) is not None else grid.dz
        pad_lo, pad_hi = int(grid.pad_z_lo), int(grid.pad_z_hi)
    cells = np.asarray(interior_cells(np.asarray(d_full, dtype=np.float64), pad_lo, pad_hi),
                       dtype=np.float64)
    edges = np.insert(np.cumsum(cells), 0, 0.0)
    planes = (float(cfg.source_x_m), float(cfg.reference_x_m), float(desired_plane_m))
    lo, hi = min(planes), max(planes)
    tol = 1e-3 * float(np.min(cells)) if cells.size else 0.0
    crossed = (edges[:-1] < hi - tol) & (edges[1:] > lo + tol)
    return axis, lo, hi, cells[crossed]


def _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired_plane_m: float,
                                              port_name: str):
    """Refuse a NU reference-plane shift whose span crosses more than one cell size.

    The NU lane evaluates beta for the reference-plane shift at the boundary
    cell (``cfg.dx``), not the cell the plane sits in. Inside one uniform
    grading zone that is a documented second-order envelope
    (``(beta*dx)^2/24``; ``docs/guides/support_matrix.md``, nonuniform
    waveguide row, and ``tests/fixtures/waveguide_nu_beta_cell_size_envelope.json``).
    Across a graded span a single beta is right only by accident, and
    integrating beta cell by cell over the span is deferred (v1.8 plan WP4,
    decision 3; tracking issue #854 item 1) -- until then the case fails
    loudly here instead of silently applying one beta. Returns the
    ``_nu_shift_span_cells`` tuple when the span is admissible.
    """
    axis, lo, hi, sizes = _nu_shift_span_cells(grid, cfg, desired_plane_m)
    if sizes.size == 0:
        return axis, lo, hi, sizes
    distinct = [float(sizes[0])]
    for v in sizes[1:]:
        if not any(np.isclose(v, d, rtol=1e-9, atol=0.0) for d in distinct):
            distinct.append(float(v))
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


def _msl_axis_spacing(grid, axis: int):
    """Cell spacing along one grid axis, and whether that axis is GRADED.

    Returns ``(spacing_m, graded, evaluable)``:

    * uniform :class:`~rfx.grid.Grid` — ``(grid.dx, False, True)``: every
      axis carries the one scalar spacing.
    * :class:`~rfx.nonuniform.NonUniformGrid` — the axis's own interior
      cell-size array decides. ``graded`` is True when max/min differ by
      more than 1e-6 relative.  ``spacing_m`` is the (single) interior
      cell size when the axis is ungraded, ``None`` when it is graded.
    * traced (mesh-as-design-variable) profiles — ``(None, None, False)``:
      the answer is not available host-side.

    Issue #686: the #469 probe-offset interval solve used to bail on ANY
    non-uniform grid (``getattr(grid, "dz", None) is not None``). Its
    stated reason — "cell-counted intervals are ill-defined under graded
    dx" — is a statement about the PROPAGATION axis, and a ``dz_profile``
    does not grade dx. For a microstrip the propagation axis is x or y,
    so on a z-graded mesh (the boundary-fitted stackup case) the interval
    is perfectly well defined and the solve simply never ran.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.nonuniform import NonUniformGrid, interior_cells

    if not isinstance(grid, NonUniformGrid):
        return float(grid.dx), False, True
    arr = (grid.dx_arr, grid.dy_arr, grid.dz)[axis]
    pad_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[axis]
    pad_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)[axis]
    if is_tracer(arr):
        return None, None, False
    cells = np.asarray(interior_cells(np.asarray(arr), pad_lo, pad_hi),
                       dtype=np.float64)
    if cells.size == 0:
        return None, None, False
    lo, hi = float(cells.min()), float(cells.max())
    if lo <= 0.0:
        return None, None, False
    graded = (hi - lo) / lo > 1e-6
    return (None if graded else lo), graded, True


def _resolve_msl_auto_offsets(sim, entries, grid):
    """Issue #469: solve the probe-offset interval for AUTO-offset ports.

    When the interval comes out EMPTY this is the third site that speaks
    about probe-clearance corruption, so it carries the same #726 guidance
    text as ``compute_msl_s_matrix``'s Z0 guard and preflight's layout
    warning. Imported, not copied.

    ``add_msl_port``'s auto-default is the UPSTREAM-only lower edge
    (``offset_min = max(3, λ/(4π·dx), 5·h_sub/dx)``); the downstream
    constraint — the deepest probe ≥ λ_g/4 (at f_max) clear of the nearest
    reflector — needs the full registered geometry, which only exists at
    driver time. Per auto port on a uniform grid:

    * no downstream reflector  -> keep ``offset_min`` (byte-identical to
      the pre-#469 default);
    * finite reflector, interval ``[offset_min, offset_max]`` non-empty ->
      midpoint (on the #469 Sheen measurement the old default sat at the
      contaminated near edge and the old advisory pushed PAST the far
      edge; the midpoint lands inside the measured-clean window);
    * interval EMPTY -> warn loudly (the feed line is too short for a
      clean N-probe measurement) and keep ``offset_min`` (upstream
      priority — the fringing transient is the historically dominant
      corruption, issue #80).

    Explicit offsets are never touched. The solve always starts from the
    STORED lower edge (``sim._msl_auto_offset_min``), so repeated calls are
    idempotent. Returns a new entries list; ``sim`` is not mutated.

    Auto probe SPACING (issue #681). A ~0.1·λ_g probe span leaves the
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
      stays with the #469 midpoint rule, preserving the measured-clean
      upstream margin for probe 0 (the #469 Sheen measurement showed the
      near edge contaminated);
    * the deepest probe stays ``λ_g/4``-clear of the absorbing boundary
      (an absorber face is a discontinuity like any reflector, and CPML
      cells are non-physical), i.e. inside
      ``domain_edge − clear − n_cpml·dx``.

    On a feed too short for any widening the caps floor at the hard
    minimum of 2 cells — byte-identical to the pre-#681 defaults on the
    committed Sheen interval-test geometry — and the existing
    empty-interval warning still fires when even that does not fit.
    Explicit spacings are never touched. Graded/unevaluable propagation
    axes keep the stored registration value (short and safe) via the
    same skip-and-warn path as the offset solve.

    Non-uniform meshes (issue #686). The bail-out is per PORT and keys on
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
    """
    _auto_spacing = getattr(sim, "_msl_auto_probe_spacing", {}) or {}
    if not sim._msl_auto_offset_min and not _auto_spacing:
        return entries

    import dataclasses
    import warnings

    from rfx.api._preflight import (
        msl_min_probe_clearance,
        msl_nearest_downstream_reflector,
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
        # Issue #661: project position/domain onto this port's propagation
        # and width axes before handing them to the x-frame helper.
        _prop_ax, _width_ax, _, _dir_sign = _msl_axis_roles(pe.direction)
        _ip = _MSL_AX[_prop_ax]
        _iw = _MSL_AX[_width_ax]
        # Issue #686: the bail-out is about THIS port's propagation axis.
        dx_u, _graded, _evaluable = _msl_axis_spacing(grid, _ip)
        if not _evaluable:
            _graded_skips.append(
                f"{pe.name!r} (direction={pe.direction!r}): the "
                f"{_prop_ax}-axis cell sizes are a traced "
                f"mesh-as-design-variable profile and cannot be inspected "
                f"host-side")
            resolved.append(pe)
            continue
        if _graded:
            _graded_skips.append(
                f"{pe.name!r} (direction={pe.direction!r}): the "
                f"propagation axis {_prop_ax} is GRADED, so a cell-counted "
                f"probe interval has no single cell size to count in")
            resolved.append(pe)
            continue
        d_refl, _, _unevaluated = msl_nearest_downstream_reflector(
            getattr(sim, "_geometry", []),
            x_probe=float(pe.position[_ip]),
            x_feed=float(pe.position[_ip]),
            y_feed=float(pe.position[_iw]),
            w_trace=float(pe.width),
            dx=dx_u,
            domain_y=float(sim._domain[_iw]),
            direction=pe.direction,
            # Issue #685: same conductor rule as the assembler, and thin
            # conductors included, so the solved offset is not derived
            # from a scan that was blind to most of the metal.
            resolve_material=sim._resolve_material,
            thin_conductors=getattr(sim, "_thin_conductors", ()),
            pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD,
        )
        if _unevaluated:
            _graded_skips.append(
                f"{pe.name!r} (direction={pe.direction!r}): the downstream "
                f"reflector scan could not evaluate "
                f"{len(_unevaluated)} conductor(s) — "
                + "; ".join(_unevaluated))
            resolved.append(pe)
            continue
        # --- Auto probe-spacing widening (issue #681, docstring above).
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
                MSL_PROBE_CLEARANCE_GUIDANCE as _MSL_PROBE_CLEARANCE_GUIDANCE,
            )
            warnings.warn(
                f"MSL port {pe.name!r}: the upstream and downstream "
                f"probe clearances are mutually unsatisfiable on this "
                f"feed (upstream needs n_probe_offset >= {off_min} "
                f"cells = max(λ/4π, 5·h_sub)/dx; downstream needs "
                f"<= {off_max} cells to keep the deepest of "
                f"{int(pe.n_probes)} probes ≥ "
                f"{clear*1e6:.0f}µm (λ_g/4 at f_max) clear of the "
                f"reflector {d_refl*1e3:.2f}mm from the feed). "
                f"The feed line is too short for a clean N-probe "
                f"measurement — keeping the upstream-priority offset "
                f"{off_min}. Extend the feed line to fix (issue #469). "
                + _MSL_PROBE_CLEARANCE_GUIDANCE,
                stacklevel=3,
            )
            resolved.append(
                dataclasses.replace(pe, **_fields) if _fields else pe
            )
    if _graded_skips:
        warnings.warn(
            "MSL auto probe-offset interval solve (issue #469) SKIPPED for "
            + str(len(_graded_skips)) + " port(s); the stored upstream-only "
            "lower edge is kept (and any auto probe spacing keeps its "
            "conservative registration default — no #681 span widening), "
            "so the downstream reflector clearance is "
            "NOT enforced for them: " + "; ".join(_graded_skips)
            + ". This used to be silent for every non-uniform mesh (issue "
            "#686). Set n_probe_offset explicitly on these ports, or make "
            "the propagation axis uniform, if the deepest probe's "
            "clearance matters.",
            stacklevel=3,
        )
    return resolved


def _project_passive(S):
    """Project S(f) onto the passive set by singular-value clipping.

    Per frequency, ``S_pass = U · min(Σ, 1) · Vᴴ`` — the nearest matrix in
    spectral norm with ‖S‖₂ ≤ 1 (standard passivity enforcement, as used in
    macromodeling). Returns ``(S_pass, correction)`` where ``correction[k] =
    max(σ_max(S(f_k)) − 1, 0)`` is the amount clipped at each frequency —
    the honesty metric: 0 where the extraction was already passive, and
    exactly how non-physical the raw value was elsewhere.

    Concrete measurement postprocessing only: the AD (eps_override) path
    never calls this. The small S matrices use host LAPACK in double
    precision, then return to the input dtype and device placement. This
    avoids GPU SVD factor errors larger than the reconstruction margin;
    the FDTD fields and differentiable extraction keep their own dtype.
    """
    # A complex128 array may outlive the caller's scoped x64 context.
    # Preserve its dtype during both canonicalization and device_put.
    with _enable_x64():
        S = jnp.asarray(S, dtype=jnp.result_type(S, 1.0))
    s_t = np.asarray(S).transpose(2, 0, 1)  # (n_freqs, n_ports, n_ports)
    real_dtype = s_t.real.dtype
    work_dtype = np.complex128 if np.iscomplexobj(s_t) else np.float64
    finite = np.all(np.isfinite(s_t), axis=(1, 2))
    # Nonfinite bins must still reach the caller's finiteness audit. One
    # invalid frequency must not make batched LAPACK abort all other bins.
    s_pass = np.full(s_t.shape, np.nan, dtype=s_t.dtype)
    correction = np.full(s_t.shape[0], np.nan, dtype=real_dtype)
    # Clip a few ULPs below 1 so the bound still holds after the f32/f64
    # reconstruction round-trip (a bare min(sig, 1) reconstructs to
    # sigma_max = 1 + O(eps), which violates the strict bound this exists
    # to guarantee).
    eps = np.finfo(real_dtype).eps
    # 64*eps, not 8*eps: the reconstruction error grows with n_ports and a
    # measured f32 sweep showed 8*eps failing the strict bound from n=8
    # (1.0000000255) through n=32 (1.0000006584); 64*eps holds through n=32.
    if np.any(finite):
        u, sig, vh = np.linalg.svd(s_t[finite].astype(work_dtype), full_matrices=False)
        correction[finite] = np.maximum(sig[:, 0] - 1.0, 0.0)
        sig_c = np.minimum(sig, 1.0 - 64.0 * eps)
        s_pass[finite] = (u * sig_c[:, None, :]) @ vh
    with _enable_x64():
        return (jax.device_put(s_pass.transpose(1, 2, 0), S.sharding),
                jax.device_put(correction, S[0, 0, :].sharding))


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
        "passivity_correction.",
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
    Counting a vacuous zero as a witness is the same mistake #395 named for
    the empty-guide reflection identity, applied here to the transmission
    entries.
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
    try:
        if isinstance(s, jax.core.Tracer):
            return
    except Exception:
        pass
    try:
        s_np = np.asarray(s)
        f_np = np.asarray(result.freqs)
    except Exception:
        # Traced / non-materializable — never let a diagnostic break the
        # numeric return path.
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
            _w.warn(_recip_msg, stacklevel=3)

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

    # Independent per-frequency amplitude advisory (issue #337).  Keep this
    # separate from the column-power gate above: normalize=False deliberately
    # has a loose column-power tolerance, but an individual |S_ij| materially
    # above unity is still useful evidence of an extraction/normalization
    # artifact.  One worst-bin warning keeps a broadband result actionable
    # without producing one warning per frequency.
    # Tolerance-class-aware threshold: the loose normalize=False waveguide
    # path (passivity_tol >= 2.0) has a DOCUMENTED validated |S11| envelope
    # reaching ~1.41 (Yee dispersion + band edge — see the normalize docstring
    # and the 2026-06-21 policy locks in test_sparam_passivity_guard.py), so
    # its amplitude advisory only fires above 1.5 (still catches the eager
    # ~1.98 spike class). Tight paths (lumped/MSL/coax/normalize=True) keep
    # the 1 + amplitude_eps (default 1.05) threshold — the MSL documented
    # envelope is 1.0063 and the field case that motivated issue #337 was 1.36.
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
                "docstring and issue #337"
            )
    bad = [
        i for i in report.issues
        if i.code in ("passivity_violation", "nonfinite_sparams")
    ]
    if not bad:
        # SOFT ADVISORY (LLM-naive-usage audit item #5): the hard check above
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
            _w.warn(" ".join(advisories), stacklevel=3)
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
    _w.warn(msg, stacklevel=3)


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
    single-mode return paths) and :meth:`_SparamMixin.compute_coaxial_s_matrix`
    assemble a family-specific ``*SMatrixResult`` and then invoke
    :func:`_warn_if_nonpassive_smatrix` immediately before returning. The
    per-port drive loop, vacuum-reference override, and rectangular-DFT
    windowing live behind the family-specific extractors (waveguide:
    ``rfx.sources.waveguide_port``; coax: the inline single-run plane-source
    path) and are deliberately *not* unified here — they share no code at this
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
    return result


_C0_SPARAMS = 299792458.0

# Weak-signal phase mask: bins whose |S21| falls below this carry no usable
# phase and are excluded from the residual below. Same VALUE and same meaning
# as the repo's other phase masks — crossval 11's ``phase_mag_floor=0.30`` and
# ``scripts/diagnostics/build_waveguide_band_broad_e5_phase_envelope.py``'s
# ``PHASE_MAG_FLOOR`` — an ABSOLUTE floor on the 0..1 |S| scale, not a fraction
# of the band peak. It lives here because the library must not import a
# diagnostics script; the two are pinned equal by
# ``tests/unit/sparams/test_waveguide_s21_phase_residual.py``.
WAVEGUIDE_PHASE_MAG_FLOOR = 0.30

# The one string naming the beta this residual is measured against, carried on
# the result so a reader never has to guess which convention produced it.
WAVEGUIDE_PHASE_BETA_CONVENTION = "yee_discrete_at_port_f_cutoff"


def s21_phase_residual_deg_rms(
    s_params,
    freqs,
    *,
    f_cutoff_hz: float,
    dt: float,
    dx: float,
    length_m: float,
    mag_floor: float = WAVEGUIDE_PHASE_MAG_FLOOR,
):
    """RMS of ``wrap(angle(S21) + beta(f) * L)`` over the measured bins, degrees.

    Defined for an EMPTY or matched guide between the two reference planes:
    a device between them adds its own transmission phase to ``angle(S21)``,
    and this number then measures the device against ``-beta*L``, not the
    port. It also needs the modal phase intact -- ``normalize=True`` divides
    the reference run's propagation phase out of S21 and is refused upstream.

    The discretization witness for the waveguide port (#894). On the empty
    matched WR-90 guide the |S11| magnitude near cutoff carries a
    dx-independent term set by whether the far-boundary round trip fits inside
    the DFT record, so an order fitted to it is a clipping artefact. This phase
    residual does not pass through the absorber at all: measured invariant to
    absorber thickness (3x), record length (2.5x) and precision, and second
    order in dx at every band down to ``f/f_c = 1.010``
    (``tests/fixtures/waveguide_vi_envelope/s21_phase_residual_witness.json``).
    That is what makes it able to answer "is the port the problem?" when a
    reflection number looks wrong.

    ``beta`` is the EXTRACTOR'S OWN: ``_compute_beta`` at the port config's
    discrete ``f_cutoff``, ``dt`` and ``dx`` — the same call
    ``_shift_modal_waves`` uses to move a wave between reference planes. A
    continuous-medium ``sqrt(k^2 - kc^2)`` would fold the Yee dispersion of the
    grid into the residual and stop it being a statement about the port.

    ``length_m`` is the de-embedded separation of the two reference planes
    along the propagation axis, read off the result's ``reference_planes``.

    Bins whose ``|S21|`` falls below ``mag_floor`` carry no usable phase and
    are dropped from the RMS; ``masked_bins`` in the returned metadata says how
    many. Returns ``(None, meta)`` — never a vacuous zero — when the S-matrix
    is not 2x2, when a value is traced, or when no bin survives the mask. Pure
    host-side NumPy on arrays the caller already has; it perturbs nothing.
    """
    meta = {
        "beta_convention": WAVEGUIDE_PHASE_BETA_CONVENTION,
        "f_cutoff_hz": None,
        "L_m": None,
        "n_bins": 0,
        "masked_bins": 0,
    }
    if is_tracer(s_params) or is_tracer(freqs):
        meta["reason"] = "traced arrays (AD path); the witness is host-side"
        return None, meta
    s = np.asarray(s_params)
    if s.ndim != 3 or s.shape[0] != 2 or s.shape[1] != 2:
        meta["reason"] = (
            f"S-matrix shape {tuple(s.shape)} is not a two-port (2, 2, n_freqs)"
        )
        return None, meta
    f = np.asarray(freqs, dtype=float).ravel()
    s21 = np.asarray(s[1, 0], dtype=complex).ravel()
    if f.size == 0 or f.size != s21.size:
        meta["reason"] = "frequency grid and S21 column have different lengths"
        return None, meta

    f_c = float(f_cutoff_hz)
    L = float(length_m)
    meta["f_cutoff_hz"] = f_c
    meta["L_m"] = L
    if not np.isfinite(L) or L <= 0.0:
        meta["reason"] = "reference-plane separation is not a positive length"
        return None, meta

    from rfx.sources.waveguide_port import _compute_beta

    beta = np.real(np.asarray(
        _compute_beta(jnp.asarray(f), f_c, dt=float(dt), dx=float(dx)),
        dtype=complex,
    ))
    keep = np.isfinite(np.abs(s21)) & (np.abs(s21) >= float(mag_floor))
    keep &= np.isfinite(beta)
    meta["masked_bins"] = int((~keep).sum())
    meta["n_bins"] = int(keep.sum())
    if not keep.any():
        meta["reason"] = (
            f"every bin's |S21| fell below the phase mask floor "
            f"{float(mag_floor):g}; a fully masked band would report a "
            f"vacuous 0 degrees"
        )
        return None, meta
    resid = np.angle(s21[keep]) + beta[keep] * L
    resid = (resid + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.degrees(np.sqrt(np.mean(resid ** 2)))), meta


def _waveguide_s21_phase_residual(s_params, freqs, reference_planes, cfgs,
                                  *, normalize=None, announce: bool = True):
    """The S21 phase residual for one assembled two-port waveguide result.

    Reads the beta ingredients off the port configs the extractor itself ran
    with (``f_cutoff``, ``dt``, ``dx``) and the length off the reference planes
    the result reports, then prints ONE banner line after the settling line.

    Post-solve and report-only: no gate, no tolerance, no effect on any
    returned number. Silent (and ``None``) whenever the residual is not
    defined — a non-two-port result, a traced AD run, two ports on different
    axes or with different discrete cutoffs, a fully masked band, or a
    ``normalize=True`` lane: that lane divides the empty-guide reference
    run's propagation phase out of S21, so ``angle(S21) + beta*L`` is not
    the port residual there (``normalize="flux"`` and ``False`` keep the
    modal phase and are reported).

    Scope of the claim: a port witness only for an EMPTY or matched guide
    between the two reference planes. A device between them adds its own
    transmission phase to ``angle(S21)``, and the same number then reads the
    device against ``-beta*L``. The banner says so.
    """
    meta = {
        "beta_convention": WAVEGUIDE_PHASE_BETA_CONVENTION,
        "f_cutoff_hz": None,
        "L_m": None,
        "n_bins": 0,
        "masked_bins": 0,
        "normalize": normalize,
    }
    if normalize is True:
        meta["reason"] = (
            "normalize=True divides the empty-guide reference run's propagation "
            "phase out of S21, so wrap(angle(S21) + beta*L) is not the port "
            "residual on this lane; normalize='flux' or normalize=False keep it"
        )
        return None, meta
    cfgs = [c[0] if isinstance(c, list) else c for c in cfgs]
    if len(cfgs) != 2:
        meta["reason"] = f"{len(cfgs)} waveguide ports; the residual is two-port"
        return None, meta
    axes = {str(getattr(c, "normal_axis", "") or "") for c in cfgs}
    if len(axes) != 1:
        meta["reason"] = f"ports on different normal axes {sorted(axes)}"
        return None, meta
    fc = [float(c.f_cutoff) for c in cfgs]
    if fc[0] <= 0.0 or abs(fc[1] - fc[0]) > 1e-9 * fc[0]:
        meta["reason"] = (
            f"the two ports carry different discrete cutoffs "
            f"({fc[0]:.6g} Hz, {fc[1]:.6g} Hz), so one -beta*L is not defined"
        )
        return None, meta
    planes = np.asarray(reference_planes, dtype=float).ravel()
    if planes.size != 2:
        meta["reason"] = "reference_planes does not carry one plane per port"
        return None, meta
    rms, meta = s21_phase_residual_deg_rms(
        s_params, freqs,
        f_cutoff_hz=fc[0], dt=float(cfgs[0].dt), dx=float(cfgs[0].dx),
        length_m=abs(float(planes[1]) - float(planes[0])),
    )
    # The helper builds its own meta; carry the lane over so the record says
    # which normalisation the number was read under.
    meta["normalize"] = normalize
    if rms is not None and announce:
        print(
            f"  [WAVEGUIDE S-MATRIX] S21 phase residual vs -beta*L: "
            f"{rms:.4g} deg rms over {meta['n_bins']} bins (beta from the "
            f"port's discrete cutoff {meta['f_cutoff_hz'] / 1e9:.5g} GHz, "
            f"L = {meta['L_m']:.6g} m; port discretization witness for an "
            f"EMPTY or matched guide between the planes -- a device between "
            f"them adds its own transmission phase; not a gate)"
        )
    return rms, meta


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
    from rfx.api._preflight import PreflightWarning

    axis_idx = {"x": 0, "y": 1, "z": 2}
    dev = np.asarray(device_sigma)
    f_arr = np.asarray(freqs, dtype=float)
    f_max = float(f_arr.max())
    f_cen = 0.5 * (float(f_arr.min()) + f_max)
    dx = float(grid.dx)
    for i, cfg in enumerate(cfgs):
        a = float(cfg.a)
        fc2 = _C0_SPARAMS / a  # TE20 cutoff for a TE10 port of width a
        if f_max >= fc2:
            # Next higher mode propagates in-band; a separate preflight warns.
            continue
        ax = axis_idx[cfg.normal_axis]
        other_axes = tuple(j for j in range(3) if j != ax)
        diff_profile = np.any(np.asarray(ref_sigmas[i]) != dev, axis=other_axes)
        # #931 moved PEC out of sigma into the solver's realized E edges.
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
        nearest = int(differing[np.argmin(np.abs(differing - probe))])
        clearance_m = abs(nearest - probe) * dx
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
                f"port-to-junction clearance left residual max|S|~3.9 in the "
                f"2026-07-06 verification (necessary-but-not-sufficient) — move "
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
    and ``lambda0 = C0 / f``. If the CPML stack (``cpml_layers * dx``) is
    thinner than ``0.5 * lambda_g`` the absorber may under-drain guided energy.
    Heuristic from the validated campaign: 20 mm CPML produced |S11| ripple
    ~0.11, 48 mm passed. Emits ``warnings.warn`` per thin-CPML port; does not
    raise.
    """
    import warnings

    f_arr = np.asarray(freqs, dtype=float)
    f_cen = 0.5 * (float(f_arr.min()) + float(f_arr.max()))
    cpml_m = int(cpml_layers) * float(grid.dx)
    for i, cfg in enumerate(cfgs):
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
                UserWarning,
                stacklevel=2,
            )


def _warn_ntff_box_dropped(sim, method_name: str) -> None:
    """Issue #704 — one warning per S-matrix call when an NTFF box would be dropped.

    ``add_ntff_box()`` registers a far-field monitor, but the S-matrix
    result classes (``MSLSMatrixResult``, ``WaveguideSMatrixResult``,
    ``CoaxialSMatrixResult``) carry no ``ntff_data``/``ntff_box`` fields, so
    whatever the per-drive solves record is discarded with nothing said —
    the same silent-drop class as #695/#685. Called ONCE at each S-matrix
    entry (after the cheap guards, before any FDTD), never per port, so a
    call covers all its drives with a single message (#697 principle 8).

    Mutation falsification (both directions, run 2026-08-24 in this
    worktree against ``tests/unit/farfield/test_ntff_smatrix_drop_warning.py``):
    - warn DELETED (early ``return`` above the ``warnings.warn``):
      3 failed, 3 passed — ``test_msl_warns_with_ntff_box``,
      ``test_waveguide_warns_with_ntff_box``,
      ``test_coaxial_warns_with_ntff_box`` each red with verbatim
      ``Failed: DID NOT WARN. No warnings of type (<class 'UserWarning'>,)
      were emitted.``
    - warn made UNCONDITIONAL (``if sim._ntff is None: return`` deleted):
      the three ``*_silent_without_ntff_box`` tests each red with verbatim
      ``TypeError: cannot unpack non-iterable NoneType object`` from the
      ``sim._ntff`` unpack below on the no-box fixtures.
    Intact code: 6 passed.
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
        "implemented — issue #704 tracks that full fix; this warning "
        "closes only the silent part. REMEDY: call run() on this "
        "simulation for far-field patterns (Result.ntff_data/ntff_box "
        "from the same port drive), or drop the NTFF box from "
        "S-matrix-only runs to save its DFT cost. STALE-IF: this "
        "method's result class grows ntff_data/ntff_box (issue #704 "
        "full threading), at which point remove this warning.",
        UserWarning,
        stacklevel=3,
    )


def _assemble_mixed_power_wave_s(
    v_lw, i_lw, v0_msl, i_msl,
    z0_lw, n_live_lw, z0_hj_msl,
    wire_mode, drive_plan,
    v_ref_lw=None,
):
    """Assemble the mixed-family power-wave S-matrix (issue #488).

    Pure function of the recorded phasors so the cross-impedance
    normalization is unit-testable without an FDTD run.

    .. warning::

       SINGLE-RATIO ASSEMBLY, deliberately not changed here (issue #517).
       This assembly forms ``S[i, j] = b_i / a_j`` per drive — the
       single-ratio rule the pure-MSL lane replaced with the multi-drive
       solve ``S = B·A⁻¹``. That much is still true of the code below.

       What is NOT true is the leak this block used to allege. It claimed
       the driven-MSL diagonal was #507-contaminated by the passive port's
       echo and that the contamination propagated into the DEFAULT flux
       channel via ``P_inc = P_net / (1 - |S_jj|^2)``. That claim was
       measured in #517 step 1 (PR #543, 2026-08-03, evidence committed as
       ``scripts/diagnostics/i517_mixed_solve_vs_ratio_measurement.py`` +
       JSON) and REFUTED for this lane. The mixed lane's passive side is a
       lumped/wire port CELL, and that cell is structurally one-phasor: its
       ``b/a`` is the resistor-law constant ``s1 = -0.60000`` (to 5.8e-4 at
       ``n_live = 4``), and both shipped extraction candidates are the same
       identity paired oppositely, so no non-degenerate ``(a, b)`` pair
       exists at the port cell under any shipped formula. The apparent
       MSL-diagonal jump under the solve (0.03 -> 0.72) is the #507 echo
       correction ``-S10·γ`` applied at a passive port where it does not
       belong, acting on two already-#313/#507-polluted quantities
       (``|S10·γ|/|S11m|`` = 19.0-38.1×; closed form matches the solve to
       1.8e-7). Substituting the solved diagonal makes the flux channel
       WORSE, not better: reciprocity 10.53 % shipped -> 23.29 % solved
       (2026-08-03 measurement), and assembling the whole mixed lane by the
       solve moves the shipped flux witness 9.80 % -> 9.85 % — no physical
       gain. Extending the solve here is therefore inert, not blocked.

       The lane's reciprocity residual (9-10 %) is real and remains OPEN,
       but its adjudication lives in #498, not here: neither mixed-lane
       diagonal has an independent same-run check on main, and the declared
       next measurement is reference-plane-split waves
       (``rfx/probes/refplane.py``, ``add_port(reference_plane_cells=N)``)
       — a different measurement LOCATION, where ``a_passive != 0``
       genuinely — not another port-cell derivation. Lane remains fenced
       experimental with a running reciprocity witness.

    Wave conventions (each mirrored line-for-line from the validated
    per-family extractors — do NOT re-derive here):

    * lumped/wire drive wave  ``a = (-V + Z0c*I) / (2*sqrt(Z0c))`` —
      ``decompose_s_matrix`` probes.py:913 / ``decompose_wire_s_matrix``
      probes.py:1018, with ``Z0c = Z0/n_live`` (n_live=1 for lumped).
    * lumped/wire passive receive ``b = (V - Z0c*I) / (2*sqrt(Z0c))`` —
      the issue-#308 orthogonal receive channel, DC-falsifier-pinned
      (probes.py:925 / :1014).
    * lumped drive-port diagonal ``b = (-V - Z0*I) / (2*sqrt(Z0))`` —
      byte-frozen ``extract_lumped_s11`` algebra (probes.py:920).
    * wire drive-port diagonal ``S_ii = (Z_in - Z0)/(Z_in + Z0)`` with
      ``Z_in = -V/I`` and the FULL Z0 (probes.py:1001-1006).
    * MSL waves ``a = (V0 + Z0_hj*I)/2``, ``b = (V0 - Z0_hj*I)/2`` — the
      OpenEMS-style V*I split at probe 0 (compute_msl_s_matrix stage S1).

    The per-family extractors report pseudo-wave ``b/a`` ratios; those
    drop the ``sqrt(Re Z0)`` Kurokawa factor, which cancels only for
    equal-impedance ports (issue #460). Mixed families have unequal Z0 by
    construction, so every wave above is divided by ``sqrt(Z0_ref)`` of
    its own port BEFORE forming ``S[i, j] = b_i / a_j``: reciprocity
    ``S21 == S12`` of a reciprocal structure then holds and is the
    committed internal falsifier for this normalization.

    Also returns the extractor-independent |S21| power witness for
    lumped/wire-driven columns (issue #313 triangulation): the port-cell
    off-diagonal wave MAGNITUDES are near-field polluted on the default
    lumped/wire path (measured |S21| 0.52-0.67 vs flux-true 0.97-1.0 on
    the canonical thru), so ``|a_drive|`` is re-derived from delivered
    power ``P_del = 0.5*Re(Z_in)*|I|^2`` and ``(1 - |S_jj|^2)`` via the
    real-Z0 power-wave identity ``P_del = 0.5*|a|^2*(1 - |S11|^2)``.

    Parameters
    ----------
    v_lw, i_lw : (n_runs, n_lw, n_freqs) complex
        FDTD-sign V/I DFT phasors at each lumped/wire port per run.
    v0_msl, i_msl : (n_runs, n_msl, n_freqs) complex
        MSL probe-0 line voltage and closed-Ampere-loop current per run.
    z0_lw : (n_lw,) float — full registered port impedances.
    n_live_lw : (n_lw,) int — wire live-cell counts (1s for lumped).
    z0_hj_msl : (n_msl,) float — analytic Hammerstad-Jensen Z0 per port.
    wire_mode : bool — True when the lumped/wire family is wire ports.
    drive_plan : list of ("lw"|"msl", local_idx) — run order.

    Returns
    -------
    S : (n_tot, n_tot, n_freqs) complex — power-wave S-matrix.
    s21_power : (n_msl, n_lw, n_freqs) real — |S21| power witness.
    """
    v_lw = jnp.asarray(v_lw)
    # Issue #683 x #764 (wire mode): drive-sample reference — the
    # pre-injection drive sample every wave formula below was calibrated
    # against; v_lw now carries the physical POST samples.  None (lumped
    # mode / legacy callers) falls back to v_lw, which then IS the
    # reference.  Only the DRIVE port's own sample differs between the
    # two; passive receives are slot-invariant.
    v_ref_lw = v_lw if v_ref_lw is None else jnp.asarray(v_ref_lw)
    i_lw = jnp.asarray(i_lw)
    v0_msl = jnp.asarray(v0_msl)
    i_msl = jnp.asarray(i_msl)
    n_lw = int(v_lw.shape[1])
    n_msl = int(v0_msl.shape[1])
    n_tot = n_lw + n_msl
    n_freqs = int(v_lw.shape[-1])
    cdt = v_lw.dtype
    S = jnp.zeros((n_tot, n_tot, n_freqs), dtype=cdt)
    s21_power = np.zeros((n_msl, n_lw, n_freqs), dtype=np.float64)

    z0c_lw = np.asarray(z0_lw, dtype=np.float64) / np.maximum(
        np.asarray(n_live_lw, dtype=np.int64), 1
    )
    sq_lw = jnp.sqrt(jnp.asarray(z0c_lw))
    sq_msl = jnp.sqrt(jnp.asarray(np.asarray(z0_hj_msl, dtype=np.float64)))

    def _b_lw_passive(run, i_port):
        # #308 receive channel, power-wave normalized (probes.py:925/:1014).
        return (v_lw[run, i_port] - z0c_lw[i_port] * i_lw[run, i_port]) / (
            2.0 * sq_lw[i_port]
        )

    def _b_msl(run, p):
        return (v0_msl[run, p] - z0_hj_msl[p] * i_msl[run, p]) / (
            2.0 * sq_msl[p]
        )

    for run, (fam, loc) in enumerate(drive_plan):
        if fam == "lw":
            col = loc
            v_d, i_d = v_ref_lw[run, loc], i_lw[run, loc]
            # Drive wave — probes.py:913/:1018 (z0_cell for wire).
            a = (-v_d + z0c_lw[loc] * i_d) / (2.0 * sq_lw[loc])
            safe_a = jnp.where(jnp.abs(a) > 0, a, jnp.ones_like(a))
            # Diagonal first (byte-frozen legacy algebra, full Z0).
            if wire_mode:
                safe_i = jnp.where(
                    jnp.abs(i_d) > 0, i_d, jnp.ones_like(i_d) * 1e-30
                )
                z_in = -v_d / safe_i
                s_jj = (z_in - z0_lw[loc]) / (z_in + z0_lw[loc])
            else:
                b_diag = (-v_d - z0_lw[loc] * i_d) / (
                    2.0 * jnp.sqrt(jnp.asarray(z0_lw[loc]))
                )
                s_jj = b_diag / safe_a
            S = S.at[col, col, :].set(s_jj.astype(cdt))
            for ri in range(n_lw):
                if ri == loc:
                    continue
                S = S.at[ri, col, :].set(
                    (_b_lw_passive(run, ri) / safe_a).astype(cdt)
                )
            # Power witness: |a| from delivered power, not the port-cell
            # wave magnitude (#313 near-field pollution triangulation).
            safe_i = jnp.where(
                jnp.abs(i_d) > 0, i_d, jnp.ones_like(i_d) * 1e-30
            )
            z_in = -v_d / safe_i
            p_del = 0.5 * jnp.real(z_in) * jnp.abs(i_d) ** 2
            one_minus = jnp.clip(1.0 - jnp.abs(s_jj) ** 2, 1e-9, None)
            a_recon = jnp.sqrt(jnp.clip(2.0 * p_del, 0.0, None) / one_minus)
            safe_ar = jnp.where(a_recon > 0, a_recon, jnp.ones_like(a_recon))
            for p in range(n_msl):
                b_p = _b_msl(run, p)
                S = S.at[n_lw + p, col, :].set((b_p / safe_a).astype(cdt))
                s21_power[p, loc, :] = np.asarray(
                    jax.lax.stop_gradient(jnp.abs(b_p) / safe_ar),
                    dtype=np.float64,
                )
        else:
            col = n_lw + loc
            # MSL drive wave (stage-S1 V*I split), power-wave normalized.
            a = (v0_msl[run, loc] + z0_hj_msl[loc] * i_msl[run, loc]) / (
                2.0 * sq_msl[loc]
            )
            safe_a = jnp.where(jnp.abs(a) > 0, a, jnp.ones_like(a))
            for ri in range(n_lw):
                S = S.at[ri, col, :].set(
                    (_b_lw_passive(run, ri) / safe_a).astype(cdt)
                )
            for p in range(n_msl):
                S = S.at[n_lw + p, col, :].set(
                    (_b_msl(run, p) / safe_a).astype(cdt)
                )
    return S, s21_power


def _mixed_reciprocity_deviation(S):
    """Worst relative |S_ij| vs |S_ji| disagreement (issue #488).

    Magnitude-only by design: cross-family phase mixes two reference-plane
    conventions on this lane, so a complex comparison would misfire.
    Returns ``((i, j), max_relative_deviation)`` over all off-diagonal
    pairs and frequencies, or ``None`` for a 1-port / tracer input.
    """
    s = np.abs(np.asarray(jax.lax.stop_gradient(S)))
    n = int(s.shape[0])
    if n < 2 or not np.all(np.isfinite(s)):
        return None
    worst_pair, worst = None, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = s[i, j, :], s[j, i, :]
            denom = np.maximum(np.maximum(a, b), 1e-12)
            dev = float(np.max(np.abs(a - b) / denom))
            if dev >= worst:
                worst, worst_pair = dev, (i, j)
    return worst_pair, worst


def _mixed_flux_magnitude_override(
    S_wave, box_lw, plane_msl, drive_plan, msl_away_signs, n_lw,
    ill_cond_floor=0.05,
):
    """Replace off-diagonal MAGNITUDES with Poynting-flux ratios (issue #488).

    Pure function (unit-testable without FDTD). Per driven column j:

        P_inc,j  = max(P_net,j, 0) / max(1 - |S_jj|^2, floor)
        |S_ij|   = sqrt(max(P_arr,i, 0) / P_inc,j)
        S_out[i,j] = |S_ij| * exp(1j * arg(S_wave[i,j]))   (phase kept)

    with P_net,j the net flux leaving the driven port through its own
    surface (outward box for lumped/wire; away-signed plane for MSL) and
    P_arr,i the toward-port flux at receive port i (inward box / toward-
    signed plane). Diagonals are NOT touched — they stay on the validated
    per-family channels. The identity P_net = P_inc*(1-|S_jj|^2) holds for
    the net flow at ANY closed surface / cross-section around the driven
    port, so no Z0 anchor enters the magnitude (the point of arch A: both
    the #313 port-cell V*I accounting and the analytic-vs-measured MSL Z0
    divergence drop out).

    Returns (S_out, ill_cond, neg_power): boolean ``(n_ports, n_freqs)``
    masks. ``ill_cond[j]`` marks bins where ``1-|S_jj|^2 <
    ill_cond_floor`` (normalization unreliable — near-total reflection at
    driven port j). ``neg_power[i]`` marks bins where a raw power at port
    i came out negative and was clipped to zero — tracked for BOTH the
    driven port's net launched power and every receive port's arriving
    power, because either sign defect silently produces a plausible
    ``|S| = 0`` instead of failing loudly.
    """
    S_wave = jnp.asarray(S_wave)
    n_tot = int(S_wave.shape[0])
    n_freqs = int(S_wave.shape[-1])
    S_out = S_wave
    ill_cond = np.zeros((n_tot, n_freqs), dtype=bool)
    neg_power = np.zeros((n_tot, n_freqs), dtype=bool)
    for run_idx, (fam, loc) in enumerate(drive_plan):
        col = loc if fam == "lw" else n_lw + loc
        s_jj = np.asarray(jax.lax.stop_gradient(S_wave[col, col, :]))
        one_minus = 1.0 - np.abs(s_jj) ** 2
        ill_cond[col, :] = one_minus < ill_cond_floor
        if fam == "lw":
            p_net = box_lw[run_idx, loc, :]
        else:
            p_net = msl_away_signs[loc] * plane_msl[run_idx, loc, :]
        neg_power[col, :] |= p_net < 0.0
        p_inc = np.clip(p_net, 0.0, None) / np.clip(one_minus, ill_cond_floor, None)
        safe_pinc = np.where(p_inc > 0.0, p_inc, np.inf)
        for i in range(n_tot):
            if i == col:
                continue
            if i < n_lw:
                p_arr = -box_lw[run_idx, i, :]          # inward = -outward
            else:
                p_arr = -msl_away_signs[i - n_lw] * plane_msl[run_idx, i - n_lw, :]
            # Symmetric with the driven-port tracking above: a negative
            # ARRIVING power is also a sign/accounting defect, and
            # clipping it silently yields |S_ij| = 0 — a plausible-looking
            # number that reads as "no coupling" instead of "broken
            # measurement".
            neg_power[i, :] |= p_arr < 0.0
            mag = np.sqrt(np.clip(p_arr, 0.0, None) / safe_pinc)
            ph = jnp.exp(1j * jnp.angle(S_wave[i, col, :]))
            S_out = S_out.at[i, col, :].set(
                (jnp.asarray(mag) * ph).astype(S_wave.dtype)
            )
    return S_out, ill_cond, neg_power

# Far-port discipline: minimum absorber depth as a fraction of the guide
# wavelength at the LOWEST measured frequency. The message quotes this value,
# so it MUST NOT be duplicated as a literal there — a mismatch would report one
# threshold while enforcing another (PR #495 review, finding 5).
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
    envelope (issue #494).

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
    dx = float(grid.dx)
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
        axis_boundary = getattr(boundary_spec, axis, None)
        if axis_boundary is None:
            continue
        faces = []
        for side in ("lo", "hi"):
            if getattr(axis_boundary, side, None) not in ("cpml", "upml"):
                continue
            override = getattr(axis_boundary, f"{side}_thickness", None)
            n_cells = int(cpml_layers if override is None else override)
            if n_cells > 0:
                faces.append((side, n_cells))
        if not faces:
            continue

        lambda_g = (_C0_SPARAMS / f_lo) / np.sqrt(1.0 - (fc / f_lo) ** 2)
        required_m = _FAR_PORT_LAMBDA_G_FRACTION * lambda_g
        thin = [(side, n) for side, n in faces if n * dx < required_m]
        if not thin:
            continue
        detail = ", ".join(
            f"{axis}-{side} {n} cells = {n * dx * 1e3:.1f} mm "
            f"({n * dx / lambda_g:.2f} lambda_g)"
            for side, n in thin
        )
        warnings.warn(
            f"compute_waveguide_s_matrix: absorber on the {axis} propagation "
            f"axis is thinner than the documented "
            f"{_FAR_PORT_LAMBDA_G_FRACTION:g} guide-wavelength "
            f"far-port discipline at the lowest measured frequency "
            f"{f_lo / 1e9:.3f} GHz (mode cutoff {fc / 1e9:.3f} GHz, "
            f"lambda_g = {lambda_g * 1e3:.1f} mm): {detail}, against a "
            f"required {required_m * 1e3:.1f} mm. A thin absorber reflects "
            f"guided energy and can set the accuracy envelope instead of "
            f"discretization: in the WR-90 iris lane (issue #494) residual "
            f"|S11| ripple was 0.0706 at 0.30 lambda_g, 0.0366 at 0.50, and "
            f"0.0093 at 0.75, so {_FAR_PORT_LAMBDA_G_FRACTION:g} lambda_g "
            f"is a floor and not a target. "
            f"Raise cpml_layers to at least "
            f"{int(np.ceil(required_m / dx))} (0.75 lambda_g needs "
            f"{int(np.ceil(0.75 * lambda_g / dx))}).",
            UserWarning,
            stacklevel=2,
        )


def _assemble_coaxial_two_port_from_voltages(
    *,
    z_planes_bot_m,
    z_planes_top_m,
    ref_bot_m: float,
    ref_top_m: float,
    v_bot_by_drive,
    v_top_by_drive,
    cond_warn: float = 1.0e3,
    _prefer_jnp: bool = False,
):
    """Pure post-FDTD assembly: per-array V(z) -> (a_inc,b_out) -> 2x2 S (#489 stage 2).

    Isolated from :meth:`_SparamMixin.compute_coaxial_two_port` so the
    convention wiring below can be exercised with PLANTED analytic V(z)
    values (no FDTD) — see
    ``tests/unit/sparams/test_coax_two_port_smatrix.py::test_planted_voltages_recover_known_asymmetric_s_matrix``.

    Parameters
    ----------
    z_planes_bot_m, z_planes_top_m : (n_bot,), (n_top,) float
        Equally spaced axial probe-plane positions (metres) for the bottom
        (port 2) and top (port 1) arrays.
    ref_bot_m, ref_top_m : float
        Each port's own reference plane (its feed's axial position).
    v_bot_by_drive, v_top_by_drive : (2, n_bot, n_freqs), (2, n_top, n_freqs) complex
        Modal voltage ``V(z)`` at every probe plane, per drive (index 0 =
        port 1 driven, index 1 = port 2 driven) and frequency.
    cond_warn : float
        Forwarded to :func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`.
    _prefer_jnp : bool
        Force the jnp assembly path even when ``v_bot_by_drive`` /
        ``v_top_by_drive`` are CONCRETE (not ``jax.core.Tracer``) — set by
        :meth:`_SparamMixin.compute_coaxial_two_port` whenever its own
        ``eps_scale`` was provided, whether or not the call happens to be
        under ``jax.grad``. Without this, a concrete FD probe
        (``eps_scale`` given but called eagerly, e.g. a finite-difference
        AD-vs-FD cross-check) would silently fall through to the strict
        NumPy ``lstsq`` in ``coaxial_line_reflection_from_plane_voltages``
        instead of that function's own tolerant jnp lstsq — measured to
        raise ``numpy.linalg.LinAlgError: SVD did not converge`` on a
        marginal fit that the jnp path handles fine (the exact PR #468 /
        #559-B1 class: an AD-vs-FD comparator silently evaluating two
        different functions). See
        :func:`rfx.sources.coaxial_port.coaxial_line_reflection_from_plane_voltages`'s
        own ``_prefer_jnp`` docstring note for the identical precedent on
        the 1-port ``eps_scale`` path.

    Returns
    -------
    s_params, cond_a, recurrence_residual, fit_residual, gamma
        ``s_params`` is ``(2, 2, n_freqs)``; ``recurrence_residual`` /
        ``fit_residual`` / ``gamma`` are ``(2, 2, n_freqs)`` indexed
        ``[port_array, drive, freq]`` (port 0 = top/port1, port 1 = bot/port2).
        ``gamma`` is the matrix-pencil-fitted complex propagation constant
        from each array's OWN local probes during that drive (Z0-free,
        independent of the reference-plane extrapolation). Measured
        2026-08-02: the SAME array's ``Re(gamma)`` differs substantially
        (~2-8x, growing with frequency) between "own drive" (that array's
        own source active — dominant field is the large launched wave, only
        weakly perturbed by the nearby feed) and "other drive" (that array
        receiving the transmitted signal — dominant field has just crossed
        the whole line and is comparably perturbed by the array's OWN nearby
        feed, which is now absorbing much more incident power). The two
        "own-drive" fits agree with each other to ~5%; the two "other-drive"
        fits agree with each other to ~2%; recurrence_residual stays <0.003
        throughout (the two-wave fit itself is clean at every one of the 4
        measurements — this is not a bad fit, it is two different, both
        internally-consistent, local decay-rate estimates). A post-hoc
        consistency check (run after this measurement, with the
        all-4-average estimator chosen after seeing the own/other split —
        not predeclared) found `|S21|*exp(+Re(gamma_bar)*L12)` reproduces
        the measured `|S21|` to within 2.1% across 4-12 GHz (see
        ``tests/unit/sparams/test_coax_two_port_smatrix.py::
        test_matched_through_line_transmits_reciprocally``), consistent with
        combined bulk-line attenuation (captured by the lower, own-drive
        estimate) plus additional loss/scattering concentrated at the
        RECEIVING feed's own discontinuity (captured by the higher,
        other-drive estimate) — not a single uniform per-metre loss. This
        check is sensitive to SCALE-type deficits (amplitude
        mis-normalization, mode conversion, a bad wave split) but
        structurally BLIND to reference-plane referral errors: a referral
        error at either plane scales the wave amplitude by
        ``exp(+/-gamma*delta)`` while ``L12`` grows by the same ``delta``,
        so the compensation factor absorbs a referral error exactly.

    Notes
    -----
    **The sign convention this function encodes**: for BOTH port arrays,
    ``a_port = result.backward_amp`` and ``b_port = result.forward_amp``
    (from :func:`rfx.sources.coaxial_port.coaxial_line_reflection_from_plane_voltages`).
    This holds specifically for the dual-duty-feed geometry
    :meth:`compute_coaxial_two_port` builds, where each port's own feed sits
    exterior of its own probe array — on the scattered-only side of that
    drive's own TFSF boundary, so ``load_below`` evaluates False for the top
    array and True for the bottom array. The derivation (global "A-branch
    travels -z / B-branch travels +z" convention, applied per port's own
    "into/out of the network" direction) is in ``docs/design_notes/
    i489_stage2_two_port_fdtd_predeclaration.md``. This is NOT a general
    fact of the extractor — a different feed placement would need a
    different mapping.

    **The general rule that constant is an instance of (issue #822)**.
    ``forward_amp`` is the branch travelling TOWARD the reference plane
    (already resolved inside the extractor by its own ``load_below``
    test, for either ladder orientation). Which of ``a``/``b`` that is
    depends on ONE further fact the extractor cannot know: whether the
    reference plane is on the DUT side of the probes.

    * Reference plane on the DUT side -- the plane IS the DUT, as in
      :func:`_assemble_coax_msl_transition_from_voltages`, where both
      planes are the junction: the wave travelling toward it is the
      INCIDENT wave -> ``a = forward_amp``.
    * Reference plane on the FAR side of the probes from the DUT -- this
      lane: each plane is that port's own feed, with the through line
      beyond the probes on the other side, for the top and the bottom
      array alike: the wave travelling toward it is LEAVING the DUT ->
      ``a = backward_amp``.

    Each lane's answer is a constant of its geometry and each is written
    as one. The sibling lane copied THIS lane's constant onto the opposite
    geometry and inverted its whole S-matrix (#822).

    This function's constant is unchanged and numerically byte-identical
    through the #822 work: the lane is recorded as **validated with
    scope** (issue #489, PI decision 2026-08-06) in
    ``docs/guides/sparameter_support_matrix.md``, and any production edit
    to it is PI-gated. What the #822 work DOES change
    on this lane is the TEST contract: the planted fixtures below are now
    built from geometry by
    ``tests/_wave_convention.py::_plant_ladder_voltages_physical`` instead
    of by inverting the extractor's own labels, so the constant above is
    now PINNED by a test that would fail if it were wrong, and the
    wrong-convention mirror asserts the exact ``inv(S_true)`` signature
    instead of merely "not close".

    AD note (#489 leg 3)
    ---------------------
    ``s_params`` (and the other four returned arrays) are differentiable
    w.r.t. ``v_bot_by_drive`` / ``v_top_by_drive``: *traced* inputs (checked
    via :func:`rfx.core.jax_utils.is_tracer`) run the whole assembly on a
    ``jax.numpy`` core — ``coaxial_line_reflection_from_plane_voltages(...,
    _prefer_jnp=True)`` (already used by the GRAD_SAFE 1-port ``eps_scale``
    path) followed by :func:`rfx.sources.coaxial_port.
    solve_two_port_from_wave_amplitudes`'s own jnp dispatch — while concrete
    inputs keep the NumPy path below UNCHANGED. Reached by
    :meth:`_SparamMixin.compute_coaxial_two_port`'s own ``eps_scale``
    parameter.
    """
    from rfx.sources.coaxial_port import (
        coaxial_line_reflection_from_plane_voltages,
        solve_two_port_from_wave_amplitudes,
    )

    z_planes_bot_m = np.asarray(z_planes_bot_m, dtype=np.float64)
    z_planes_top_m = np.asarray(z_planes_top_m, dtype=np.float64)

    # Traced voltages (compute_coaxial_two_port(eps_scale=...), #489 AD leg 3)
    # -> jax.numpy assembly below; concrete -> NumPy, UNCHANGED (moved into the
    # `else` branch verbatim so the validated numeric path stays byte-identical).
    # ``_prefer_jnp`` (set by the eps_scale call site regardless of tracing)
    # is OR'd in for the same reason coaxial_line_reflection_from_plane_
    # voltages's own _prefer_jnp exists: a concrete FD probe on the
    # eps_scale path must hit the identical jnp branch jax.grad sees, not
    # silently fall back to NumPy just because it happens to run eagerly.
    _traced = _prefer_jnp or is_tracer(v_bot_by_drive) or is_tracer(v_top_by_drive)
    if not _traced:
        v_bot_by_drive = np.asarray(v_bot_by_drive, dtype=np.complex128)
        v_top_by_drive = np.asarray(v_top_by_drive, dtype=np.complex128)
    elif not hasattr(v_bot_by_drive, "shape") or not hasattr(v_top_by_drive, "shape"):
        # The traced/_prefer_jnp path indexes v_*_by_drive directly (no
        # np.asarray concretization step) — a plain Python list has no
        # .shape and would otherwise fail below with an opaque
        # AttributeError instead of a message naming the actual problem.
        raise ValueError(
            "v_bot_by_drive / v_top_by_drive must be array-like (numpy or "
            "jax.numpy) on the traced/_prefer_jnp path, not a plain Python "
            f"list; got {type(v_bot_by_drive).__name__} / "
            f"{type(v_top_by_drive).__name__}. Wrap with jnp.asarray(...) "
            "(or np.asarray(...) if concrete) before calling."
        )
    if v_bot_by_drive.shape[0] != 2 or v_top_by_drive.shape[0] != 2:
        raise ValueError(
            "v_bot_by_drive / v_top_by_drive must have a leading axis of "
            f"size 2 (one per drive); got {v_bot_by_drive.shape} / "
            f"{v_top_by_drive.shape}."
        )
    n_f = int(v_bot_by_drive.shape[-1])

    if _traced:
        # --- differentiable path: jnp assembly (AD moat, #489 leg 3) ---
        # Same per-(drive, frequency) recurrence-fit loop as the concrete path
        # below, routed through the extractor's own jnp core
        # (_coaxial_line_reflection_jnp via _prefer_jnp=True — the SAME
        # differentiable core the GRAD_SAFE 1-port eps_scale path already
        # uses) instead of concretizing. n_f/drive are static Python ints, so
        # this unrolls into a fixed-size jnp graph, mirroring
        # compute_coaxial_line_reflection's own `for fi in range(n_f):` jnp
        # branch.
        a_top_rows, b_top_rows = [], []
        a_bot_rows, b_bot_rows = [], []
        rec_top_rows, fit_top_rows, gamma_top_rows = [], [], []
        rec_bot_rows, fit_bot_rows, gamma_bot_rows = [], [], []
        for drive_idx in range(2):
            a_top_f, b_top_f, rec_top_f, fit_top_f, gamma_top_f = [], [], [], [], []
            a_bot_f, b_bot_f, rec_bot_f, fit_bot_f, gamma_bot_f = [], [], [], [], []
            for fi in range(n_f):
                out_bot = coaxial_line_reflection_from_plane_voltages(
                    z_planes_bot_m, v_bot_by_drive[drive_idx, :, fi],
                    reference_plane_m=ref_bot_m, _prefer_jnp=True,
                )
                out_top = coaxial_line_reflection_from_plane_voltages(
                    z_planes_top_m, v_top_by_drive[drive_idx, :, fi],
                    reference_plane_m=ref_top_m, _prefer_jnp=True,
                )
                a_top_f.append(out_top.backward_amp)
                b_top_f.append(out_top.forward_amp)
                rec_top_f.append(out_top.recurrence_residual)
                fit_top_f.append(out_top.fit_residual)
                gamma_top_f.append(out_top.gamma)
                a_bot_f.append(out_bot.backward_amp)
                b_bot_f.append(out_bot.forward_amp)
                rec_bot_f.append(out_bot.recurrence_residual)
                fit_bot_f.append(out_bot.fit_residual)
                gamma_bot_f.append(out_bot.gamma)
            a_top_rows.append(jnp.stack(a_top_f))
            b_top_rows.append(jnp.stack(b_top_f))
            rec_top_rows.append(jnp.stack(rec_top_f))
            fit_top_rows.append(jnp.stack(fit_top_f))
            gamma_top_rows.append(jnp.stack(gamma_top_f))
            a_bot_rows.append(jnp.stack(a_bot_f))
            b_bot_rows.append(jnp.stack(b_bot_f))
            rec_bot_rows.append(jnp.stack(rec_bot_f))
            fit_bot_rows.append(jnp.stack(fit_bot_f))
            gamma_bot_rows.append(jnp.stack(gamma_bot_f))

        # Port-array axis 0 = top/port1, 1 = bot/port2 (matches the concrete
        # path's a_inc[0]=out_top / a_inc[1]=out_bot assignment below).
        a_inc = jnp.stack([jnp.stack(a_top_rows), jnp.stack(a_bot_rows)], axis=0)
        b_out = jnp.stack([jnp.stack(b_top_rows), jnp.stack(b_bot_rows)], axis=0)
        rec_resid = jnp.stack([jnp.stack(rec_top_rows), jnp.stack(rec_bot_rows)], axis=0)
        fit_resid = jnp.stack([jnp.stack(fit_top_rows), jnp.stack(fit_bot_rows)], axis=0)
        gamma = jnp.stack([jnp.stack(gamma_top_rows), jnp.stack(gamma_bot_rows)], axis=0)

        solve = solve_two_port_from_wave_amplitudes(
            a_inc, b_out, cond_warn=float(cond_warn), _prefer_jnp=True,
        )
        return solve.s_params, solve.cond_a, rec_resid, fit_resid, gamma

    # --- concrete path: NumPy (byte-identical to the pre-AD code) ---
    a_inc = np.zeros((2, 2, n_f), dtype=np.complex128)
    b_out = np.zeros((2, 2, n_f), dtype=np.complex128)
    rec_resid = np.zeros((2, 2, n_f), dtype=np.float64)
    fit_resid = np.zeros((2, 2, n_f), dtype=np.float64)
    gamma = np.zeros((2, 2, n_f), dtype=np.complex128)

    for drive_idx in range(2):
        for fi in range(n_f):
            out_bot = coaxial_line_reflection_from_plane_voltages(
                z_planes_bot_m, v_bot_by_drive[drive_idx, :, fi],
                reference_plane_m=ref_bot_m,
            )
            out_top = coaxial_line_reflection_from_plane_voltages(
                z_planes_top_m, v_top_by_drive[drive_idx, :, fi],
                reference_plane_m=ref_top_m,
            )
            a_inc[0, drive_idx, fi] = out_top.backward_amp
            b_out[0, drive_idx, fi] = out_top.forward_amp
            a_inc[1, drive_idx, fi] = out_bot.backward_amp
            b_out[1, drive_idx, fi] = out_bot.forward_amp
            rec_resid[0, drive_idx, fi] = out_top.recurrence_residual
            fit_resid[0, drive_idx, fi] = out_top.fit_residual
            rec_resid[1, drive_idx, fi] = out_bot.recurrence_residual
            fit_resid[1, drive_idx, fi] = out_bot.fit_residual
            gamma[0, drive_idx, fi] = out_top.gamma
            gamma[1, drive_idx, fi] = out_bot.gamma

    solve = solve_two_port_from_wave_amplitudes(a_inc, b_out, cond_warn=float(cond_warn))
    return solve.s_params, solve.cond_a, rec_resid, fit_resid, gamma


def _ladder_split_witness(planes_m, v_by_drive, ref_m):
    """Disjoint-half self-consistency witness for ONE probe ladder (issue #823).

    Refit the SAME extractor
    (:func:`rfx.sources.coaxial_port.coaxial_line_reflection_from_plane_voltages`)
    on two DISJOINT CONTIGUOUS halves of the ladder, against the SAME reference
    plane, and report how far the two halves disagree. For ``N`` probes the
    halves are ``idx[0:N//2]`` and ``idx[N - N//2:N]`` -- for odd ``N`` the
    middle probe belongs to neither, so the two fits never share a plane.

    Why this exists: the lane's committed single-mode witness is
    ``fit_residual``, computed over the WHOLE ladder. A residual computed over
    a window that includes garbage cannot detect that the window is the
    problem. Measured on the settled attempt-3 run (VESSL 369367257533), the
    production 9-probe MSL ladder reports ``fit_residual`` 0.342/0.264/0.222 --
    large, but attributed for three attempts to "the field is not two-wave
    here" rather than to the ladder. Its two halves disagree about ``|Gamma|``
    by 4.379/4.487/4.321 DECADES; the compliant 8-probe subset of the very
    same dump disagrees by 0.005/0.001/0.002. That is three orders of
    separation on a quantity the residual could not resolve at all.

    Returns
    -------
    (gamma_dev, reflection_decades) : two ``(n_drives, n_freqs)`` float64 arrays
        ``gamma_dev = |g_A - g_B| / (0.5*|g_A + g_B|)`` -- the symmetric relative
        deviation of the complex propagation constant, and
        ``reflection_decades = |log10(|Gamma_A| / |Gamma_B|)|`` -- decades of
        disagreement in the reflection magnitude referred to ``ref_m``.
        ``NaN`` when the ladder carries fewer than 6 probes (each half needs
        >= 3 planes for the matrix pencil) or when a half's ``|Gamma|`` is zero
        or non-finite. NaN means "no witness", never "a small number".

    REPORT-ONLY. No gate, no refusal, no tolerance. The coax stub's own ladder
    reads ``gamma_dev`` 0.11-0.34 on every run of this family -- the already
    known 1 mm-span alpha-identification limit (#589) -- so a bar tight enough
    to catch the MSL ladder would refuse the coax ladder for a different
    defect. Gating is deliberately deferred (the PI sequencing puts the
    standoff rule and this witness BEFORE any ``msl_fit_residual_max``).
    """
    from rfx.sources.coaxial_port import coaxial_line_reflection_from_plane_voltages

    planes = np.asarray(planes_m, dtype=np.float64)
    v = np.asarray(v_by_drive, dtype=np.complex128)
    n_drives, n_planes, n_f = v.shape
    gamma_dev = np.full((n_drives, n_f), np.nan, dtype=np.float64)
    decades = np.full((n_drives, n_f), np.nan, dtype=np.float64)
    half = n_planes // 2
    if half < 3:
        return gamma_dev, decades
    lo = np.arange(half)
    hi = np.arange(n_planes - half, n_planes)
    for di in range(n_drives):
        for fi in range(n_f):
            try:
                out_a = coaxial_line_reflection_from_plane_voltages(
                    planes[lo], v[di, lo, fi], reference_plane_m=float(ref_m))
                out_b = coaxial_line_reflection_from_plane_voltages(
                    planes[hi], v[di, hi, fi], reference_plane_m=float(ref_m))
            except (ValueError, np.linalg.LinAlgError):
                continue
            g_a, g_b = complex(out_a.gamma), complex(out_b.gamma)
            g_mid = 0.5 * abs(g_a + g_b)
            if g_mid > 0.0 and np.isfinite(g_mid):
                dev = abs(g_a - g_b) / g_mid
                if np.isfinite(dev):
                    gamma_dev[di, fi] = float(dev)
            r_a, r_b = abs(complex(out_a.reflection)), abs(complex(out_b.reflection))
            if (np.isfinite(r_a) and np.isfinite(r_b) and r_a > 0.0 and r_b > 0.0):
                decades[di, fi] = float(abs(np.log10(r_a / r_b)))
    return gamma_dev, decades


def _assemble_coax_msl_transition_from_voltages(
    *,
    z_coax_planes_m,
    x_msl_planes_m,
    ref_coax_m: float,
    ref_msl_m: float,
    v_coax_by_drive,
    v_msl_by_drive,
    z0_coax: float,
    z0_msl: float,
    cond_warn: float = 1.0e3,
):
    """Pure post-FDTD assembly: coax + MSL modal-voltage ladders -> power-wave 2x2 S (#489 leg 4).

    Isolated from :meth:`_SparamMixin.compute_coax_msl_transition` so the
    cross-family normalization can be exercised with PLANTED analytic
    voltages (no FDTD) — see
    ``tests/unit/sparams/test_coax_msl_transition.py::test_planted_voltages_recover_known_s_matrix_with_unequal_z0``.
    Concrete NumPy only (no jnp/traced branch): AD is explicitly out of
    scope for this leg (see :class:`~rfx.api._spec.CoaxMSLTransitionResult`'s
    class docstring).

    Both ports' forward/backward modal-voltage wave amplitudes come from
    the SAME extractor,
    :func:`rfx.sources.coaxial_port.coaxial_line_reflection_from_plane_voltages`
    (a Z0-free matrix-pencil fit over >=3 equally spaced planes) — applied to
    the coax port's z-axis probe ladder AND, in place of the MSL lane's own
    diagnostic-only N-probe SVD fit, to the MSL port's x-axis probe ladder
    too (see the class docstring for why). That extractor returns RAW modal
    VOLTAGE wave amplitudes (volts, ``V(x)=A*exp(-gamma*x)+B*exp(+gamma*x)``
    Z0-free by construction) — each is converted to a POWER wave via
    ``a = V+ / sqrt(Z0)``, ``b = V- / sqrt(Z0)`` (the standard real-Z0
    Kurokawa identity, using each port's OWN reference impedance — ``Z0`` is
    already a real float here, both callers pass an analytic real
    impedance, so no ``Re()`` is taken anywhere in this function; a complex
    reference impedance is out of scope) before the two-drive solve. This
    division is the load-bearing fix for the pre-declared
    "impedance-convention mismatch" failure mode: solving directly on the
    raw volt-wave amplitudes would leave the diagonal correct but scale
    each off-diagonal entry by ``sqrt(Z0_i/Z0_j)`` — see
    :func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`'s
    own docstring for the generic two-drive solve this feeds.

    ``cond_a`` vs ``cond_a_equilibrated`` (issue #581 review, finding B2)
    -----------------------------------------------------------------------
    ``solve_two_port_from_wave_amplitudes``'s own ``cond_a`` is the RAW
    condition number of the 2x2 incident-wave matrix ``A``. On the coax-coax
    stage-2 lane that matrix's two columns are naturally comparable in scale
    (same TEM source construction, same ``field_scale``, both drives), so a
    large raw ``cond_a`` there really does mean "the two drives' incident
    waves are nearly parallel in port-space" (near-degenerate). On THIS
    mixed-family lane the two drives are built by unrelated source
    constructions with no reason to share an amplitude (a coax TEM plane
    source vs an MSL Ez injection) — measured on the committed fixture, the
    two columns differ by 5-9 orders of magnitude in norm, which alone
    inflates ``cond_a`` into the 1e3-1e7 range with NO implication about the
    two incident-wave DIRECTIONS. ``cond_a_equilibrated`` divides each
    column of ``A`` by its own norm before taking ``cond`` — invariant to
    the per-drive scale, so it isolates genuine geometric near-parallelism.
    Column equilibration does not change ``s_params`` (``S = B @ inv(A)`` is
    invariant under any per-column rescaling of ``(a_inc, b_out)`` pairs,
    since both matrices pick up the identical column scale factor and it
    cancels in ``B @ inv(A)``); it is a diagnostic-only recomputation.

    Parameters
    ----------
    z_coax_planes_m, x_msl_planes_m : (n_coax,), (n_msl,) float
        Equally spaced axial probe-plane positions (metres) for the coax
        port (port array 0, along z) and the MSL port (port array 1, along
        x). Each family's own axis — these are NOT on a shared coordinate.
    ref_coax_m, ref_msl_m : float
        Each port's own reference plane, in that port's own axis. Chosen by
        the caller to sit AT the physical coax<->MSL launch discontinuity
        (minimizing the reference-plane-mismatch failure mode by
        construction — see :meth:`_SparamMixin.compute_coax_msl_transition`).
    v_coax_by_drive, v_msl_by_drive : (2, n_coax, n_freqs), (2, n_msl, n_freqs) complex
        Modal voltage at every probe plane, per drive (index 0 = coax port
        driven, index 1 = MSL port driven) and frequency.
    z0_coax, z0_msl : float
        Real reference impedance (ohm) for the power-wave normalization:
        analytic coax TEM Z0 and analytic Hammerstad-Jensen microstrip Zc.
    cond_warn : float
        Forwarded to :func:`solve_two_port_from_wave_amplitudes`.

    Returns
    -------
    s_params, cond_a, cond_a_equilibrated, recurrence_residual, fit_residual, gamma, a_inc, b_out
        ``s_params`` is ``(2, 2, n_freqs)``, port order ``(coax, msl)``.
        ``cond_a`` / ``cond_a_equilibrated`` are ``(n_freqs,)`` (see above).
        ``recurrence_residual`` / ``fit_residual`` / ``gamma`` / ``a_inc`` /
        ``b_out`` are ``(2, 2, n_freqs)`` indexed ``[port_array, drive,
        freq]`` (port array 0 = coax, 1 = msl) — ``a_inc``/``b_out`` are the
        POWER-wave amplitudes actually fed to the two-drive solve (post
        ``sqrt(Z0)`` division), exposed for audit per issue #581 review
        finding B2.

    Notes
    -----
    **Wave-role convention (issue #822)**. ``forward_amp`` is the branch
    travelling TOWARD the reference plane -- the extractor's own contract,
    resolved inside it by its ``load_below = reference_plane_m <= z.mean()``
    test, for either orientation of the ladder. Which of ``a``/``b`` that
    branch is depends on ONE further fact the extractor cannot know:
    whether the reference plane is on the DUT side of the probes.

    * :meth:`_SparamMixin.compute_coax_msl_transition` (this function):
      both reference planes are placed AT the junction (``ref_coax_m`` =
      the junction z, ``ref_msl_m`` = ``float(junction_x)``) -- the
      reference plane IS the DUT. The wave travelling toward the reference
      plane is therefore the wave incident on the DUT, so
      ``a = forward_amp`` and ``b = backward_amp`` at BOTH ports. That is
      a constant of this lane and it is written as one. It does NOT depend
      on which side of its ladder the junction sits: the extractor already
      resolved "toward the reference plane" per call, so the same constant
      holds for both MSL facings the calling method supports (``"-x"``,
      junction below the ladder -- the committed attempt-2/3/3b fixture --
      and ``"+x"``, junction above it; see the method's
      ``(1 if msl_pe.direction == "+x" else -1)`` monotonicity branch).
      Pinned by ``tests/unit/sparams/test_coax_msl_transition_wave_roles.py::
      test_assembler_wave_roles_hold_on_a_mirrored_msl_ladder``, whose
      docstring records what reading ``a = backward_amp`` at the MSL port
      of the mirrored fixture does to ``S_code``.
      (An earlier revision of this fix wrote the choice as a per-port bit
      ``(ref_m - centroid) * sign(ref_m - centroid) > 0``. That expression
      is true whenever ``ref_m != centroid``, so it was this constant with
      an unreachable ``backward_amp`` branch; it is now written as the
      constant it always was.)
    * :func:`_assemble_coaxial_two_port_from_voltages`: each reference
      plane is that port's own feed, on the FAR side of the probes from
      the DUT, so the wave travelling toward it is LEAVING the DUT ->
      ``a = backward_amp``.

    Before #822 this function used the two-port lane's constant
    (``a = backward_amp``) on the opposite geometry, which swaps ``a`` and
    ``b`` at BOTH ports; since ``S = B inv(A)``, exchanging ``A`` and ``B``
    returns ``inv(S_true)``. Every number this function produced before the
    fix is that inverse. The regression gate is
    ``tests/unit/sparams/test_coax_msl_transition_wave_roles.py::
    test_assembler_wave_roles_follow_the_junction_side_reference_plane``,
    whose planted voltages are built from GEOMETRY
    (``tests/_wave_convention.py::_plant_ladder_voltages_physical``) and
    never from the extractor's own labels -- planting from the labels is
    why the pre-existing planted test passed under either assignment.
    """
    from rfx.sources.coaxial_port import (
        coaxial_line_reflection_from_plane_voltages,
        solve_two_port_from_wave_amplitudes,
    )

    z_coax = np.asarray(z_coax_planes_m, dtype=np.float64)
    x_msl = np.asarray(x_msl_planes_m, dtype=np.float64)
    v_coax_by_drive = np.asarray(v_coax_by_drive, dtype=np.complex128)
    v_msl_by_drive = np.asarray(v_msl_by_drive, dtype=np.complex128)
    if v_coax_by_drive.shape[0] != 2 or v_msl_by_drive.shape[0] != 2:
        raise ValueError(
            "v_coax_by_drive / v_msl_by_drive must have a leading axis of "
            f"size 2 (one per drive); got {v_coax_by_drive.shape} / "
            f"{v_msl_by_drive.shape}."
        )
    n_f = int(v_coax_by_drive.shape[-1])
    if int(v_msl_by_drive.shape[-1]) != n_f:
        raise ValueError(
            "v_coax_by_drive and v_msl_by_drive must share the same "
            f"trailing frequency axis; got {v_coax_by_drive.shape[-1]} vs "
            f"{v_msl_by_drive.shape[-1]}."
        )
    if not (np.isfinite(z0_coax) and z0_coax > 0.0):
        raise ValueError(f"z0_coax must be positive finite, got {z0_coax}")
    if not (np.isfinite(z0_msl) and z0_msl > 0.0):
        raise ValueError(f"z0_msl must be positive finite, got {z0_msl}")
    sqrt_z0 = np.array([np.sqrt(float(z0_coax)), np.sqrt(float(z0_msl))])

    a_inc = np.zeros((2, 2, n_f), dtype=np.complex128)
    b_out = np.zeros((2, 2, n_f), dtype=np.complex128)
    rec_resid = np.zeros((2, 2, n_f), dtype=np.float64)
    fit_resid = np.zeros((2, 2, n_f), dtype=np.float64)
    gamma = np.zeros((2, 2, n_f), dtype=np.complex128)

    for drive_idx in range(2):
        for fi in range(n_f):
            out_coax = coaxial_line_reflection_from_plane_voltages(
                z_coax, v_coax_by_drive[drive_idx, :, fi],
                reference_plane_m=float(ref_coax_m),
            )
            out_msl = coaxial_line_reflection_from_plane_voltages(
                x_msl, v_msl_by_drive[drive_idx, :, fi],
                reference_plane_m=float(ref_msl_m),
            )
            # Wave roles (#822): the extractor already resolved which branch
            # travels TOWARD each reference plane (forward_amp, by its own
            # load_below test, for either ladder orientation). On THIS lane
            # both reference planes ARE the junction, so that branch is the
            # wave incident on the DUT: a = forward_amp, b = backward_amp at
            # both ports -- a constant of the lane, written as one. See the
            # Notes section above.
            a_coax, b_coax = out_coax.forward_amp, out_coax.backward_amp
            a_msl, b_msl = out_msl.forward_amp, out_msl.backward_amp
            # Raw modal-voltage waves (volts, Z0-free) -> power waves.
            a_inc[0, drive_idx, fi] = a_coax / sqrt_z0[0]
            b_out[0, drive_idx, fi] = b_coax / sqrt_z0[0]
            a_inc[1, drive_idx, fi] = a_msl / sqrt_z0[1]
            b_out[1, drive_idx, fi] = b_msl / sqrt_z0[1]
            rec_resid[0, drive_idx, fi] = out_coax.recurrence_residual
            fit_resid[0, drive_idx, fi] = out_coax.fit_residual
            gamma[0, drive_idx, fi] = out_coax.gamma
            rec_resid[1, drive_idx, fi] = out_msl.recurrence_residual
            fit_resid[1, drive_idx, fi] = out_msl.fit_residual
            gamma[1, drive_idx, fi] = out_msl.gamma

    solve = solve_two_port_from_wave_amplitudes(a_inc, b_out, cond_warn=float(cond_warn))

    # Column-equilibrated condition number (issue #581 review, finding B2):
    # divide each drive's own incident-wave column by its own norm before
    # taking cond() so a per-drive amplitude-scale mismatch (routine on a
    # mixed-family lane, see the docstring above) cannot masquerade as
    # geometric near-parallelism. Does not touch s_params.
    cond_a_equilibrated = np.full(n_f, np.nan, dtype=np.float64)
    for fi in range(n_f):
        col_norms = np.linalg.norm(a_inc[:, :, fi], axis=0)
        safe_norms = np.where(col_norms > 0.0, col_norms, 1.0)
        a_eq = a_inc[:, :, fi] / safe_norms[None, :]
        cond_a_equilibrated[fi] = float(np.linalg.cond(a_eq))

    return (
        solve.s_params, solve.cond_a, cond_a_equilibrated,
        rec_resid, fit_resid, gamma, a_inc, b_out,
    )


def _register_msl_h_planes(sim, prefix, stencil, components, freqs, region=None):
    """Register the two bracketing samples of each transverse H component."""
    pairs = []
    for component in components:
        right = f"{prefix}_{component}"
        left = f"{right}_left"
        for name, coordinate in zip((left, right), stencil["registration_coordinates"]):
            sim.add_dft_plane_probe(
                axis=stencil["axis"], coordinate=coordinate,
                component=component, freqs=freqs, name=name,
            )
            if region is not None:
                sim._dft_plane_regions[name] = region
        pairs.append((left, right))
    return tuple(pairs)


def _collocated_msl_h(planes, names, weights):
    """Read each H pair without leaving the field's JAX differentiation tape."""
    from rfx.sources.msl_port import msl_collocate_h_planes

    missing = [name for pair in names for name in pair if name not in planes]
    if missing:
        raise ValueError(f"MSL current requires bracketing H-plane data; missing {missing}")
    return tuple(msl_collocate_h_planes(
        planes[left].accumulator, planes[right].accumulator, weights,
    ) for left, right in names)


def _msl_power_wave_scales(reference_impedances, dtype):
    """Relative power-wave row scales for positive real references.

    sqrt(R0)/sqrt(Rp) differs from canonical 1/sqrt(Rp) by one common
    scalar, which cancels from S and cond(A). Equal-reference records keep
    exactly unit scales, while unequal ports use the same power metric.
    """
    refs = np.asarray(reference_impedances)
    if (refs.ndim != 1 or refs.size == 0 or np.iscomplexobj(refs)
            or not np.all(np.isfinite(refs)) or not np.all(refs > 0)):
        raise ValueError("MSL S reference impedances must be finite positive real values")
    real_dtype = np.finfo(np.dtype(dtype)).dtype
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        roots = np.sqrt(refs.astype(np.float64))
        scales = (roots[0] / roots).astype(real_dtype)
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("MSL power-wave scaling is not representable at this precision")
    return jnp.asarray(scales)
