"""S-parameter extraction methods for :class:`Simulation`.

Import contract (Part B Stage 2 refactor):
  This module is a transitional mixin. It must import ONLY from
  ``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy.
  It must NEVER do ``from rfx.api import ...`` or ``from . import ...``
  the package, to keep ``rfx/api/__init__.py`` the sole composition point.

The methods here were moved verbatim out of ``rfx/api/__init__.py``'s
``class Simulation`` body. They are pure structural relocations — same
indentation, decorators, signatures, docstrings, and logic. ``Simulation``
inherits ``_SparamMixin`` so every method below remains a bound method on
``Simulation`` instances; all existing call-sites are unaffected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import CoaxialPort
from rfx.sources.waveguide_port import (
    extract_waveguide_s_matrix,
    extract_waveguide_s_matrix_flux,
    extract_waveguide_s_params_normalized,
    extract_multimode_s_matrix,
    extract_multimode_s_matrix_flux,
    waveguide_plane_positions,
)

from rfx.nonuniform import NonUniformGrid

from rfx.api._spec import (
    WaveguideSMatrixResult,
    CoaxialSMatrixResult,
    CoaxialLineReflectionResult,
    CoaxialTwoPortResult,
    MSLSMatrixResult,
    MixedSMatrixResult,
    CoaxMSLTransitionResult,
    _WaveguidePortEntry,
    _MSLPortEntry,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Type-only forward reference to the composed class for the
    # ``port_reference_sims: "list[Simulation] | None"`` annotation. This is
    # NOT a runtime import: the module-import contract above forbids a runtime
    # ``from rfx.api import ...`` (cycle / sole-composition-point), and a
    # TYPE_CHECKING guard never executes, so the contract is preserved.
    from rfx.api import Simulation


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
    """Modal voltage ``V = -∫E·dz`` from ground to the trace underside.

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
    lies inside the one-cell PEC trace, where
    :func:`rfx.boundaries.pec.apply_pec_mask` deliberately preserves the
    NORMAL E component as surface charge (``rfx/boundaries/pec.py:90-93``)
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
    measured ``|a_passive/a_driven| = 0.07-0.51`` across three fixtures — and
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
    ``tests/test_msl_modal_voltage_and_wave_solve.py``.

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
        "standing-wave null at the port plane: "
        f"{affected_freqs.size} bins in [{f1:.4f}, {f2:.4f}] GHz "
        "have |V|,|I| below 10% of band median — wave-split "
        "S-parameters are unreliable there (blind spot of single-run "
        "reflection measurements of strong reflectors); see "
        "rfx-known-issues standing-wave-null entry",
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
        "measurement (issue #681). S11/S21 are unaffected (they use the "
        "analytic Z0 anchor). Common causes: the real eps_eff is far from "
        "the HJ estimate (wrong eps_r_sub / substrate not detected under "
        "the port), or a contaminated/under-settled record at those bins "
        "(check settling_db and the reliable mask). Check the result's "
        "beta_railed mask before quoting Z0 or beta.",
        stacklevel=2,
    )


_SETTLING_WITNESS_DB = -40.0


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
    finite = np.isfinite(settling_db)
    hot = finite & (settling_db > _SETTLING_WITNESS_DB)
    if not bool(np.any(hot)):
        return

    import warnings

    if num_periods is not None:
        knob, record = "num_periods", f"num_periods={num_periods:g}"
    elif n_steps is not None:
        knob, record = "n_steps", f"n_steps={int(n_steps):d}"
    else:  # pragma: no cover - guarded by the call sites
        knob, record = "the record length", "fixed-length"
    per_run = ", ".join(
        f"port {port_names[i] if i < len(port_names) else i} driven: "
        f"{settling_db[i]:+.1f} dB"
        for i in np.flatnonzero(hot)
    )
    warnings.warn(
        "ring-down settling witness FAILED (end/peak energy above "
        f"{_SETTLING_WITNESS_DB:.0f} dB): {per_run}. The {record} "
        "record ended while the structure was still "
        "ringing, so the DFT-based S-parameters of the affected run(s) are "
        "truncation artifacts wherever the structure is resonant — expect "
        "spurious |S| poles and passivity violations. Increase "
        f"{knob} until the witness is below −40 dB before quoting any S "
        "value (see the result's settling_db field).",
        stacklevel=2,
    )


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
                f"{off_min}; expect standing-wave bias at the deep "
                f"probes. Extend the feed line to fix (issue #469).",
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

    jnp-native and batched. NOTE: the AD (eps_override) path never calls
    this — see the wiring comment at the call site.
    """
    s_t = jnp.transpose(S, (2, 0, 1))            # (n_freqs, n_ports, n_ports)
    u, sig, vh = jnp.linalg.svd(s_t, full_matrices=False)
    correction = jnp.maximum(sig[:, 0] - 1.0, 0.0)
    # Clip a few ULPs below 1 so the bound still holds after the f32/f64
    # reconstruction round-trip (a bare min(sig, 1) reconstructs to
    # sigma_max = 1 + O(eps), which violates the strict bound this exists
    # to guarantee).
    eps = jnp.finfo(sig.dtype).eps
    # 64*eps, not 8*eps: the reconstruction error grows with n_ports and a
    # measured f32 sweep showed 8*eps failing the strict bound from n=8
    # (1.0000000255) through n=32 (1.0000006584); 64*eps holds through n=32.
    sig_c = jnp.minimum(sig, 1.0 - 64.0 * eps)
    s_pass = jnp.einsum("fij,fj,fjk->fik", u, sig_c.astype(u.dtype), vh)
    return jnp.transpose(s_pass, (1, 2, 0)), correction


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


def _warn_if_nonpassive_smatrix(
    result,
    *,
    extractor: str,
    strict: bool = False,
    passivity_tol: float = 0.10,
    amplitude_eps: float = 0.05,
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
    """
    _warn_if_nonpassive_smatrix(
        result,
        extractor=extractor,
        strict=strict,
        passivity_tol=passivity_tol,
    )
    return result


_C0_SPARAMS = 299792458.0


def _warn_junction_probe_clearance(grid, cfgs, device_sigma, ref_sigmas, freqs):
    """Advisory: probe-plane clearance from a junction (pure NumPy, no FDTD).

    For each driven port, the junction is where the device materials differ
    from that port's straight-guide reference. We reduce the PEC-folded
    ``sigma`` difference over the two transverse axes to a 1-D profile along
    the port normal axis, find the nearest differing cell to the port's probe
    plane, and compare that clearance to evanescent decay lengths of the next
    higher mode (TE20, cutoff ``fc2 = C0 / a``). ``alpha = 2*pi*sqrt(fc2^2 -
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
            warnings.warn(
                f"port_reference_sims: port index {i} ({cfg.normal_axis}-normal) "
                f"probe plane is only {clearance_m * 1e3:.1f} mm from the "
                f"junction — below {minimum_m * 1e3:.1f} mm (3 mid-band "
                f"evanescent decay lengths of the next higher mode, the floor "
                f"of the validated far-port envelope); recommend "
                f">= {recommended_m * 1e3:.1f} mm (5 decay lengths). Compact "
                f"port-to-junction clearance left residual max|S|~3.9 in the "
                f"2026-07-06 verification (necessary-but-not-sufficient) — move "
                f"the probe plane farther from the junction.",
                UserWarning,
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
    worktree against ``tests/test_ntff_smatrix_drop_warning.py``):
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
):
    """Assemble the mixed-family power-wave S-matrix (issue #488).

    Pure function of the recorded phasors so the cross-impedance
    normalization is unit-testable without an FDTD run.

    .. warning::

       KNOWN #507 RESIDUE, deliberately not half-fixed here (issue #517).
       This assembly still forms ``S[i, j] = b_i / a_j`` per drive — the
       single-ratio rule the pure-MSL lane replaced with the multi-drive
       solve ``S = B·A⁻¹`` — so a passive port's echo is reported as the
       driven port's own response whenever ``a_passive != 0`` (measured
       0.07-0.51 on the pure-MSL fixtures). The driven-MSL diagonal is
       exactly that contaminated quantity, and it feeds the DEFAULT flux
       channel via ``P_inc = P_net / (1 - |S_jj|^2)``, so the residue
       propagates into the flux magnitudes too; it is a live candidate for
       this lane's 9% reciprocity residual (#488/#498). Extending the solve
       needs the Kurokawa cross-family ``sqrt(Z)`` composition worked out
       first — see #517 for the measurement-first plan. Lane remains fenced
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
            v_d, i_d = v_lw[run, loc], i_lw[run, loc]
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
    ``tests/test_coax_two_port_fdtd.py::test_planted_voltages_recover_known_asymmetric_s_matrix``.

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
        ``tests/test_coax_two_port_fdtd.py::
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
    ``tests/test_coax_msl_transition.py::test_planted_voltages_recover_known_s_matrix_with_unequal_z0``.
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
            # Raw modal-voltage waves (volts, Z0-free) -> power waves.
            a_inc[0, drive_idx, fi] = out_coax.backward_amp / sqrt_z0[0]
            b_out[0, drive_idx, fi] = out_coax.forward_amp / sqrt_z0[0]
            a_inc[1, drive_idx, fi] = out_msl.backward_amp / sqrt_z0[1]
            b_out[1, drive_idx, fi] = out_msl.forward_amp / sqrt_z0[1]
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


class _SparamMixin:
    """S-parameter extraction methods mixed into :class:`Simulation`."""

    def compute_waveguide_s_matrix(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        normalize: bool | str = False,
        subpixel_smoothing: bool | str = False,
        eps_override: "jnp.ndarray | None" = None,
        sigma_override: "jnp.ndarray | None" = None,
        checkpoint_segments: int | None = None,
        strict_passivity: bool = False,
        port_reference_sims: "list[Simulation] | None" = None,
    ) -> WaveguideSMatrixResult:
        """Compute a theoretically clean axis-normal boundary-aperture waveguide S-matrix.

        Parameters
        ----------
        num_periods : float
            Length of the FDTD run (in source-period multiples) used to
            derive ``n_steps`` when ``n_steps`` is not given. The
            spectra are computed POST-SCAN from the recorded modal V/I
            time series via a rectangular full-record DFT (matching
            OpenEMS's ``utilities.DFT_time2freq``); ``num_periods``
            therefore governs both the CPML drain horizon AND the DFT
            integration window. Phase 2 cleanup (2026-04-25) removed
            the legacy ``num_periods_dft`` early-gate knob — the rect
            full-record DFT is finite-energy on the recorded transient
            so no gating is needed even on strong reflectors.
        normalize : bool or "flux"
            Controls the S-parameter extraction algorithm:

            ``False`` (default) — modal V/I decomposition, no reference
            run.  Magnitude includes Yee impedance mismatch
            (Z_TE_num/Z_TE_exact ≈ 3 % at dx/λ = 0.07).  Use for
            |S11| of strong reflectors (PEC short, high-Q resonators)
            where this error is smaller than the ±10–20 % round-trip
            dispersion error introduced by ``normalize=True``.

            ``True`` — two-run modal normalization.  Cancels one-way
            Yee dispersion for **transmission** (off-diagonal) by
            dividing device outgoing waves by reference outgoing waves
            at the same port.  **Does not** cancel dispersion for
            reflection (round-trip vs one-way path mismatch); use
            ``normalize=False`` or ``normalize="flux"`` for S11 of
            strong reflectors.

            ``"flux"`` — hybrid power-flux extraction.  Magnitude from
            Poynting-vector DFT (|S|² = P_flux / P_inc), phase from
            modal V/I.  Corrects both the Z_TE impedance-mismatch error
            in S11 and the round-trip dispersion error in the
            ``normalize=True`` diagonal formula.  Costs 2 × N_ports
            FDTD runs (same as ``normalize=True``).
        checkpoint_segments : int or None
            Segmented gradient checkpointing for the **uniform** waveguide
            AD path (issue #73 / PR #125).  Splits the ``n_steps`` scan
            into ``K`` segments that are rematerialised via
            ``jax.checkpoint`` during the backward pass, reducing peak
            reverse-mode memory from O(n_steps·|carry|) to
            O((K + n_steps/K)·|carry|) (≈ O(√n_steps·|carry|) at the
            optimal K ≈ √n_steps, at ≈ 2× backward compute cost).
            ``K`` is forwarded to ``rfx.simulation.run`` with
            ``checkpoint=True``; ``K`` MUST exactly divide the
            auto-computed ``n_steps`` (the runner rejects non-divisors —
            choose the nearest divisor of √n_steps; padding is rejected
            because it would shift the V/I DFT windows).  Default
            ``None`` is byte-identical to the pre-checkpoint scan.

            On a NON-uniform mesh (``dx_profile`` / ``dy_profile``,
            issue #73) ``checkpoint_segments=K`` is now supported: ``K`` is
            translated to the NU runner's ``checkpoint_every`` chunk size — the
            divisor of ``n_steps`` nearest to ``n_steps/K`` — and applied to the
            *device* run only (the vacuum reference is constant in the design
            variable). The chunk MUST divide ``n_steps`` (same as the uniform
            path): a non-divisor chunk would let the NU runner's zero-padding add
            spurious ring-down steps to the carry-accumulated flux DFT and shift
            the S-matrix. With an exact divisor the result is forward-IDENTICAL
            and the ≈O(√n_steps) tape reduction is realised under ``jax.grad``
            with ``normalize='flux'`` + an ``eps_override`` / ``sigma_override``
            design variable.
        port_reference_sims : list[Simulation] or None
            Per-driven-port matched-straight-guide reference simulations for
            **interior-PEC multi-port** structures (T-junctions, branches,
            septa). ``port_reference_sims[i]`` is a ``Simulation`` describing
            the STRAIGHT continuation of driven port ``i``'s guide with **no
            junction** — same domain / ``dx`` / boundary, geometry = port
            ``i``'s guide walls extended straight through. Only valid with
            ``normalize='flux'`` (raises otherwise); single-mode ports only;
            uniform mesh only; not combinable with ``eps_override`` /
            ``sigma_override``.

            **Multi-port junctions (port_reference_sims).** The default flux
            path references incident power ``P_inc`` to a single shared VACUUM
            run (empty domain, no interior PEC). For a straight guide the walls
            come from ``pec_axes`` so that vacuum reference already carries the
            guided ``P_inc`` correctly. For a junction the interior PEC septum /
            branch is stripped from the vacuum reference, which then radiates
            into free space: ``P_inc`` is mis-normalized and every ``|S|``
            inflates hard (non-passive; ``normalize=True`` gave max|S|~230,
            ``normalize='flux'`` gave max|S|~9.8, |S11|~1.9 on the verified
            compact 3-port T-junction). Passing ``port_reference_sims[i]`` = the
            matched straight guide for driven arm ``i`` moves ``P_inc`` toward
            the true single-mode guided incident power (PEC-folded identically
            to the device run).

            **Far-port discipline (required for a physical S-matrix).** This
            plumbing is NECESSARY but NOT SUFFICIENT. A physical junction
            S-matrix additionally requires: (1) each port's probe plane placed
            >= 5 evanescent decay lengths of the next higher mode away from the
            junction; (2) CPML thickness >= ~0.5 guide wavelengths at band
            centre; (3) a converged mesh. On far-port geometry (arms 90/90/70 mm,
            48 mm CPML, dx 1.0/0.667 mm) the matched-reference flux path reaches
            passivity 1.006/1.002, reciprocity 0.001, mesh-convergence 0.0297
            and 0.087 vs MEEP (r2000, cross-device). On COMPACT geometry the
            matched reference fixes |S11| (1.86 → 0.49, physical) but the overall
            matrix stays non-physical (residual max|S|~3.9); the two in-method
            advisories (probe clearance, CPML thickness) warn when the far-port
            discipline is not met.

            The absorber half of that discipline is checked on **every**
            uniform path, not just this one: an in-method advisory (issue
            #494) fires whenever the absorber on a port's propagation axis is
            thinner than ``0.5 * lambda_g`` at the **lowest** measured
            frequency, which is where ``lambda_g`` is longest and the
            ``cpml_layers=16`` default weakest. It is checked in-method
            because the functional entry points run no ``sim.preflight()``.
            Treat ``0.5 * lambda_g`` as a floor, not a target: at the WR-90
            band edge the measured residual ``|S11|`` ripple was 0.0706 at
            0.30 ``lambda_g``, 0.0366 at 0.50 and 0.0093 at 0.75, so a
            0.5-``lambda_g`` absorber can still set the accuracy envelope
            instead of discretization. The advisory is silent on a
            non-uniform mesh (``cpml_layers * dx`` is ambiguous under a
            graded profile) and on a band that starts at or below cutoff
            (``lambda_g`` is undefined there; the ``port_freqs_below_cutoff``
            preflight owns that case).

            This enables junction measurements UNDER the documented
            discipline; it does NOT make arbitrary compact junctions
            valid. See the skipped ``test_api.py`` T-junction reciprocity test
            and the companion evidence gate test
            ``tests/test_waveguide_tjunction_e4e5_gates.py``.
        """
        if not normalize:
            import warnings
            warnings.warn(
                "compute_waveguide_s_matrix(normalize=False): S21 and "
                "S-parameter phase include Yee numerical dispersion. "
                "For S21 accuracy and reciprocity use normalize=True. "
                "For |S11| of strong reflectors (PEC short, resonators) "
                "normalize=False is more accurate — see the normalize "
                "parameter docstring.",
                stacklevel=2,
            )
        if self._ports or self._tfsf:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported together with lumped ports or TFSF"
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported with manual periodic-axis overrides"
            )
        if len(self._waveguide_ports) < 2:
            raise ValueError(
                "compute_waveguide_s_matrix() requires at least two waveguide ports"
            )

        entries = list(self._waveguide_ports)
        if any(entry.probe_plane is not None for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() does not use per-port probe_plane; use reference_plane only or leave probe_plane unset"
            )
        if any(entry.calibration_preset not in (None, "measured") for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() currently supports only measured/default reference planes or explicit reference_plane overrides"
            )

        # Per-port straight-guide references for interior-PEC junctions.
        # Cheap guards first (raise BEFORE any FDTD); the grid-match check
        # and PEC-fold happen after the device grid is assembled below.
        if port_reference_sims is not None:
            if normalize != "flux":
                raise ValueError(
                    "port_reference_sims requires normalize='flux' — the "
                    "per-port reference feeds the flux P_inc normalization"
                )
            if any(entry.n_modes > 1 for entry in entries):
                raise NotImplementedError(
                    "port_reference_sims is not supported with multimode ports "
                    "(n_modes>1): the multimode extractor has no per-port-"
                    "reference support"
                )
            if eps_override is not None or sigma_override is not None:
                raise NotImplementedError(
                    "port_reference_sims combined with eps_override / "
                    "sigma_override is an unvalidated combination and is not "
                    "supported"
                )
            if len(port_reference_sims) != len(entries):
                raise ValueError(
                    "port_reference_sims must supply one Simulation per "
                    f"waveguide port ({len(entries)}), got "
                    f"{len(port_reference_sims)}"
                )

        # Issue #704 audit: same silent NTFF drop class as the MSL path.
        _warn_ntff_box_dropped(self, "compute_waveguide_s_matrix()")

        # Non-uniform-mesh dispatch. Earlier the uniform scan ran with
        # the coarse boundary dx and silently ignored ``dx_profile`` /
        # ``dy_profile`` (handover v2 experiment 12). The dedicated NU
        # two-run extractor below is enabled when its supported scope
        # is met (``normalize=True``, single-mode ports); otherwise
        # raise so the user is not given silently-wrong numbers.
        # Frequency grid resolved HERE rather than after the non-uniform
        # dispatch below: the NU lane's absorber advisory needs it, and
        # duplicating the resolver into that branch would be one more hand copy
        # of production logic (the #576-review class of defect). Pure function of
        # the entries and freq_max, so moving it earlier only makes the
        # matching-grid check fail sooner.
        def _resolve_freqs(entry: _WaveguidePortEntry) -> jnp.ndarray:
            if entry.freqs is not None:
                return entry.freqs
            return jnp.linspace(self._freq_max / 10, self._freq_max, entry.n_freqs)

        freqs = _resolve_freqs(entries[0])
        for entry in entries[1:]:
            entry_freqs = _resolve_freqs(entry)
            if entry_freqs.shape != freqs.shape or not np.allclose(np.asarray(entry_freqs), np.asarray(freqs)):
                raise ValueError("waveguide S-matrix requires matching frequency grids on all ports")

        if self._dx_profile is not None or self._dy_profile is not None:
            if checkpoint_segments is not None and checkpoint_segments < 1:
                raise ValueError(
                    f"checkpoint_segments must be >= 1, got {checkpoint_segments}"
                )
            unsupported = []
            if normalize is not True and normalize != "flux":
                unsupported.append("normalize=True or normalize='flux' is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            # The differentiable eps/sigma AD channel is wired on the NU
            # path only for normalize='flux' (mirrors the uniform PR #172
            # flux-AD fix): the flux extractor is now jnp-native end-to-end
            # so a traced eps_override flows into the device Yee update and
            # back through the S-matrix. normalize=True is kept out of scope
            # — its diagonal a_inc_ref denominator carries the #88 band-edge
            # fragility, so accepting eps_override there could yield
            # silently-wrong gradients.
            if eps_override is not None and normalize != "flux":
                unsupported.append(
                    "eps_override (differentiable AD channel) on the NU path "
                    "requires normalize='flux'"
                )
            if sigma_override is not None and normalize != "flux":
                unsupported.append(
                    "sigma_override (differentiable AD channel) on the NU path "
                    "requires normalize='flux'"
                )
            if subpixel_smoothing:
                unsupported.append("subpixel_smoothing is not supported")
            if port_reference_sims is not None:
                unsupported.append(
                    "port_reference_sims (per-port straight-guide junction "
                    "references) is not supported on the non-uniform lane"
                )
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile) supports normalize=True or "
                    "normalize='flux' and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy profile to use the uniform lane."
                )
            # Far-port absorber advisory, on the NU lane too (#576 review F3).
            # It used to sit ~110 lines below this branch's return, i.e. it was
            # UNREACHABLE from here: the guard existed, its text named the exact
            # remedy, and no NU caller ever asked it. That silence is why both NU
            # fixture producers shipped 0.33 and 0.099 lambda_g stacks — patching
            # the two producers would have left the next one to repeat it. Pure
            # NumPy, no FDTD, so it costs a grid build and a few array ops.
            # No bare `except: pass` around it: an advisory that can fail
            # silently is the failure mode this finding IS. If the plumbing ever
            # breaks, the caller hears about it as a warning rather than getting
            # the same quiet all-clear that let 0.33 lambda_g ship.
            from rfx.runners.nonuniform import _build_waveguide_port_config_nu
            try:
                _nu_grid = self._build_nonuniform_grid()
                _nu_cfgs = [
                    # n_steps is still None here when the caller passed
                    # num_periods; the advisory reads only the config's
                    # propagation axis and modal cutoff, never anything derived
                    # from the step count, so a placeholder is honest rather
                    # than a lie about the run length.
                    _build_waveguide_port_config_nu(
                        self, _e, _nu_grid, jnp.asarray(freqs),
                        int(n_steps) if n_steps else 1)
                    for _e in entries
                ]
            except Exception as _exc:  # noqa: BLE001 - reported, not swallowed
                import warnings as _w
                _w.warn(
                    "compute_waveguide_s_matrix: could not evaluate the far-port "
                    f"absorber advisory on the non-uniform lane ({_exc!r}). The "
                    "0.5 guide-wavelength discipline is therefore UNCHECKED for "
                    "this run — verify the absorber depth by hand (#576).",
                    stacklevel=2,
                )
            else:
                _warn_thin_absorber_vs_guide_wavelength(
                    _nu_grid, _nu_cfgs, freqs, self._cpml_layers,
                    self._boundary_spec,
                )

            _res_nu = self._compute_waveguide_s_matrix_nu(
                n_steps=n_steps,
                num_periods=num_periods,
                normalize=normalize,
                eps_override=eps_override,
                sigma_override=sigma_override,
                checkpoint_segments=checkpoint_segments,
            )
            return _finalize_sparam_result(
                _res_nu,
                extractor="compute_waveguide_s_matrix",
                strict=strict_passivity,
                # normalize=False carries documented Yee-dispersion + band-edge
                # |S11| overshoot (validated paths reach ~1.4-1.7), so use a
                # loose bound there that still catches gross extractor bugs
                # (|S11|>>1); normalize=True/"flux" correct dispersion -> tight.
                passivity_tol=2.0 if normalize is False else 0.10,
            )

        grid = self._build_grid()
        _wg_sheet_specs: list = []
        base_materials, debye_spec, lorentz_spec, pec_mask_wg, pec_shapes, boundary_pec_shapes, _ = self._assemble_materials(
            grid, sheet_specs=_wg_sheet_specs)
        # #677: node-thin sheet ctx for the DEVICE runs of this lane. The
        # PEC here is folded to sigma=1e10 below rather than run as a
        # mask, but the edge exclusion still uses the assembled pec_mask
        # so sheet and PEC never contend for one edge; the vacuum
        # REFERENCE runs never receive the ctx (explicit strip at the
        # extractor call sites).
        from rfx.materials.thin_conductor import build_sheet_impedance_ctx as _build_sheet_ctx
        _wg_sheet_ctx = _build_sheet_ctx(_wg_sheet_specs, pec_mask=pec_mask_wg)
        if _wg_sheet_ctx is not None and subpixel_smoothing:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets are not "
                "supported with subpixel_smoothing / conformal on the "
                "waveguide S-matrix lane (#677 v1): the sheet operator "
                "assumes the plain isotropic E update at its edges.")
        # Waveguide S-matrix runner doesn't support pec_mask yet.
        # Fold PEC mask back into high sigma for compatibility.
        # **Stage 2 caveat**: when ``subpixel_smoothing="kottke_pec"`` is
        # active (use_kottke_pec, computed below), the inverse-eps
        # tensor encodes the PEC zero directly (inv = 0 freezes the
        # field). Folding pec_mask to sigma=1e10 then would conflict
        # with the Yee-stagger offsets in inv_xx/yy/zz: pec_mask is
        # per-cell-center, but inv_xx is at Ex(i+0.5, j, k) offsets,
        # so PEC boundary cells can have sigma=1e10 AND a fractional
        # inv > 0 — that combo blows up Ca ≈ -1 and field NaNs.
        # Skipped for Stage 2; the Kottke union (inv=0 inside PEC,
        # fractional at boundary) is the single source of truth.
        _use_kottke_pec_early = (subpixel_smoothing == "kottke_pec")
        if pec_mask_wg is not None and not _use_kottke_pec_early:
            base_materials = base_materials._replace(
                sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
        materials = base_materials
        # G-AD-WIRE-WG2: public eps_override / sigma_override channel.
        # Mirror the MSL pattern: replace eps_r / sigma on the assembled
        # materials *after* the PEC fold so PEC boundaries are untouched.
        if eps_override is not None:
            materials = materials._replace(eps_r=eps_override)
        if sigma_override is not None:
            materials = materials._replace(sigma=sigma_override)

        # Per-port straight-guide reference materials for interior-PEC
        # junctions. Each reference sim is a geometry carrier: assemble its
        # materials on a grid that must match the device grid, then fold its
        # interior PEC into sigma IDENTICALLY to the device path above (only
        # the plain path — no subpixel/conformal handling for references).
        ref_materials_per_port = None
        if port_reference_sims is not None:
            ref_materials_per_port = []
            for _i, _ref_sim in enumerate(port_reference_sims):
                _ref_grid = _ref_sim._build_grid()
                if _ref_grid.shape != grid.shape or float(_ref_grid.dx) != float(grid.dx):
                    raise ValueError(
                        f"port_reference_sims[{_i}] grid "
                        f"(shape={_ref_grid.shape}, dx={_ref_grid.dx}) must "
                        f"match the device grid (shape={grid.shape}, "
                        f"dx={grid.dx})"
                    )
                _ref_base, _, _, _ref_pec_mask, _, _, _ = _ref_sim._assemble_materials(_ref_grid)
                if _ref_pec_mask is not None:
                    _ref_base = _ref_base._replace(
                        sigma=jnp.where(_ref_pec_mask, 1e10, _ref_base.sigma))
                ref_materials_per_port.append(_ref_base)

        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)
        _, debye, lorentz = self._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)

        # Build configs — may be a single config or a list of configs per port
        has_multimode = any(entry.n_modes > 1 for entry in entries)
        raw_cfgs = [self._build_waveguide_port_config(entry, grid, freqs, n_steps) for entry in entries]

        # Unify source waveform across all ports so that the S-matrix
        # extraction uses identical excitation.  Different source spectra
        # (from mismatched f0/bandwidth) cause S11 ≠ S22 artifacts in the
        # unnormalized path because V/I decomposition error varies with
        # frequency.  Use port 0's waveform as the canonical source.
        def _flatten_cfgs(cfgs):
            out = []
            for c in cfgs:
                if isinstance(c, list):
                    out.extend(c)
                else:
                    out.append(c)
            return out

        flat0 = _flatten_cfgs(raw_cfgs)
        ref_t0 = flat0[0].src_t0
        ref_tau = flat0[0].src_tau
        need_unify = any(
            c.src_t0 != ref_t0 or c.src_tau != ref_tau for c in flat0[1:]
        )
        if need_unify:
            raw_cfgs = [
                cfg._replace(src_t0=ref_t0, src_tau=ref_tau)
                if not isinstance(cfg, list)
                else [c._replace(src_t0=ref_t0, src_tau=ref_tau) for c in cfg]
                for cfg in raw_cfgs
            ]

        # Far-port absorber advisory for EVERY uniform two-port path (issue
        # #494). Emitted here, before the FDTD runs and before the
        # single-mode / multimode split, because the functional entry points
        # run no sim.preflight() and the port_reference_sims advisory below
        # covers only the junction path.
        _warn_thin_absorber_vs_guide_wavelength(
            grid, raw_cfgs, freqs, self._cpml_layers, self._boundary_spec,
        )

        # Compute Kottke per-component smoothed permittivity if requested.
        # Shared by both single-mode and multi-mode paths.
        # Mirrors rfx/runners/uniform.py: shape_eps_pairs from sim geometry,
        # then compute_smoothed_eps. The reference run is vacuum and has no
        # ε interfaces, so it always passes aniso_eps=None inside the
        # extractor.
        # Stage 2 unified path: subpixel_smoothing="kottke_pec" routes
        # through compute_inv_eps_tensor_diag and skips the Stage 1
        # eps_correction + apply_conformal_pec chain entirely. Both
        # device and reference (vacuum) runs see the same boundary-
        # face PEC walls, so the inverse-permittivity tensor is
        # computed twice (once per material context).
        use_kottke_pec = (subpixel_smoothing == "kottke_pec")
        aniso_eps = None
        aniso_inv_eps = None
        ref_aniso_inv_eps = None
        if use_kottke_pec:
            from rfx.geometry.smoothing import compute_inv_eps_tensor_diag
            shape_eps_pairs = [
                (entry.shape, self._resolve_material(entry.material_name).eps_r)
                for entry in self._geometry
            ]
            aniso_inv_eps = compute_inv_eps_tensor_diag(
                grid,
                dielectric_shapes=shape_eps_pairs,
                pec_shapes=pec_shapes or [],
                background_eps=1.0,
            )
            # Reference run is empty guide with same boundary walls only.
            # Must NOT include interior PEC geometry (e.g. PEC short box):
            # if device and reference share the same obstacle, both DFTs
            # are identical and (device - reference) / incident = 0.
            ref_aniso_inv_eps = compute_inv_eps_tensor_diag(
                grid,
                dielectric_shapes=[],
                pec_shapes=boundary_pec_shapes,
                background_eps=1.0,
            )
            # Yee-stagger correction: the Kottke union reaches inv=0
            # on Yee-staggered components only when the cell-center
            # AND the offset position are both inside the PEC shape.
            # For thin PEC obstacles (e.g. a 1-cell-wide PEC short),
            # cell-center is inside but Ey/Ez Yee positions are at
            # cell-corner offsets that fall *outside* the box → inv
            # remains 1 (vacuum). That leaves the H field free to
            # propagate inside the PEC region and seeds late-time
            # exponential growth.
            #
            # Fix: where ``pec_mask`` (cell-center binary) is True,
            # force all three inv components to zero. This is the
            # cell-center analogue of Stage 1's sigma=1e10 fold,
            # without the Ca→-1 instability that the sigma fold has
            # at Yee-staggered cells where inv > 0.
            if pec_mask_wg is not None:
                inv_xx, inv_yy, inv_zz = aniso_inv_eps
                inv_xx = jnp.where(pec_mask_wg, 0.0, inv_xx)
                inv_yy = jnp.where(pec_mask_wg, 0.0, inv_yy)
                inv_zz = jnp.where(pec_mask_wg, 0.0, inv_zz)
                aniso_inv_eps = (inv_xx, inv_yy, inv_zz)
                # pec_mask_wg marks interior PEC geometry (e.g. the PEC
                # short). The reference run has no interior PEC — do NOT
                # apply pec_mask_wg to ref_aniso_inv_eps, or the reference
                # becomes identical to the device and S11 = 0.
        elif subpixel_smoothing:
            from rfx.geometry.smoothing import compute_smoothed_eps
            shape_eps_pairs = [
                (entry.shape, self._resolve_material(entry.material_name).eps_r)
                for entry in self._geometry
            ]
            if shape_eps_pairs:
                aniso_eps = compute_smoothed_eps(
                    grid, shape_eps_pairs, background_eps=1.0,
                )

        # Stage 1 conformal PEC: when BoundarySpec declares conformal
        # faces and pec_shapes was populated (boundary half-space +
        # any user PEC), compute Dey-Mittra weights and apply
        # eps_correction. Mirrors runners/uniform.py:96-124.
        # ``conformal_weights`` flows through extract_waveguide_*
        # into rfx.simulation.run, which already calls
        # ``apply_conformal_pec`` per step in its scan body.
        # Suppressed when use_kottke_pec — Stage 2 owns the PEC
        # tensor encoding and the eps_correction would double-correct.
        conformal_weights = None
        ref_aniso_eps = None
        if (self._boundary_spec.conformal_faces() and pec_shapes
                and not use_kottke_pec):
            from rfx.geometry.conformal import (
                compute_conformal_weights_sdf,
                clamp_conformal_weights,
                conformal_eps_correction,
            )
            w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, pec_shapes)
            w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, 0.1)
            conformal_weights = (w_ex, w_ey, w_ez)
            # Per-component conformal-corrected eps. Merge with the
            # smoothed eps (if any): conformal overrides at boundary
            # cells, smoothed survives in the interior.
            eps_base = materials.eps_r
            eps_ex_c, eps_ey_c, eps_ez_c = conformal_eps_correction(
                eps_base, w_ex, w_ey, w_ez,
            )
            if aniso_eps is not None:
                s_ex, s_ey, s_ez = aniso_eps
                boundary_ex = w_ex < 1.0
                boundary_ey = w_ey < 1.0
                boundary_ez = w_ez < 1.0
                eps_ex_c = jnp.where(boundary_ex, eps_ex_c, s_ex)
                eps_ey_c = jnp.where(boundary_ey, eps_ey_c, s_ey)
                eps_ez_c = jnp.where(boundary_ez, eps_ez_c, s_ez)
            aniso_eps = (eps_ex_c, eps_ey_c, eps_ez_c)
            # The reference run (vacuum) shares the same boundary
            # walls, so the conformal eps correction applies equally.
            # Build it from the ref vacuum eps so the only difference
            # ref-vs-device is the obstacle in ``materials.eps_r``.
            ref_eps_base = jnp.ones_like(eps_base)
            ref_ex, ref_ey, ref_ez = conformal_eps_correction(
                ref_eps_base, w_ex, w_ey, w_ez,
            )
            ref_aniso_eps = (ref_ex, ref_ey, ref_ez)

        if has_multimode and _wg_sheet_ctx is not None:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets are not "
                "supported on the multimode waveguide S-matrix path "
                "(#677 v1): the multimode extractors do not thread the "
                "sheet operator ctx, so the runs would silently simulate "
                "NO sheet. Use n_modes=1 ports or drop the f0 sheet.")
        if has_multimode:
            # Multi-mode path: each raw_cfg is a list of WaveguidePortConfig
            port_mode_cfgs: list[list] = []
            for entry, raw in zip(entries, raw_cfgs):
                if isinstance(raw, list):
                    port_mode_cfgs.append(raw)
                else:
                    port_mode_cfgs.append([raw])

            ref_shifts_mm = []
            desired_refs_mm = []
            for entry, mode_cfgs in zip(entries, port_mode_cfgs):
                first_cfg = mode_cfgs[0]
                planes = waveguide_plane_positions(first_cfg)
                desired_ref = (
                    entry.reference_plane
                    if entry.reference_plane is not None
                    else planes["source"]
                )
                ref_shifts_mm.append(desired_ref - planes["reference"])
                desired_refs_mm.append(desired_ref)

            mm_pec_axes = "".join(axis for axis in "xyz" if axis not in grid.cpml_axes)
            if normalize == "flux":
                from rfx.core.yee import init_materials as _init_vacuum_materials
                ref_materials = _init_vacuum_materials(grid.shape)
                s_params, mode_map = extract_multimode_s_matrix_flux(
                    grid,
                    materials,
                    ref_materials,
                    port_mode_cfgs,
                    n_steps,
                    boundary="cpml",
                    cpml_axes=grid.cpml_axes,
                    pec_axes=mm_pec_axes,
                    debye=debye,
                    lorentz=lorentz,
                    ref_shifts=ref_shifts_mm,
                    aniso_eps=aniso_eps,
                    conformal_weights=conformal_weights,
                    aniso_inv_eps=aniso_inv_eps,
                )
            elif normalize:
                # The two-run normalized extractor divides each receiving
                # channel by its own empty-guide outgoing wave
                # (b_dev/b_ref). For cross-mode channels the empty-guide
                # reference is ~0 (orthogonal modes do not couple in a
                # uniform guide), so the ratio blows up (measured
                # cross-mode |S| ~ 4.7 on an over-moded WR-90 slab).
                # Use normalize="flux" instead — power ratios referenced
                # to the always-nonzero incident modal power avoid the
                # 0/0 and also fix the reflection noise floor.
                raise ValueError(
                    "compute_waveguide_s_matrix(normalize=True) is not "
                    "supported with n_modes > 1 (cross-mode channels hit a "
                    "0/0 in the two-run normalization). Use "
                    "normalize='flux' for multi-mode S-matrices."
                )
            else:
                s_params, mode_map = extract_multimode_s_matrix(
                    grid,
                    materials,
                    port_mode_cfgs,
                    n_steps,
                    boundary="cpml",
                    cpml_axes=grid.cpml_axes,
                    pec_axes=mm_pec_axes,
                    debye=debye,
                    lorentz=lorentz,
                    ref_shifts=ref_shifts_mm,
                    aniso_eps=aniso_eps,
                    conformal_weights=conformal_weights,
                    aniso_inv_eps=aniso_inv_eps,
                )
            # Report the ABSOLUTE de-embed target plane (matches the single-mode + coax paths and
            # the WaveguideSMatrixResult schema), NOT the relative shift ref_shifts_mm — that is the
            # extractor's phase-shift input, not a plane coordinate (RF-audit 2026-07-23).
            reference_planes = np.array(desired_refs_mm, dtype=float)
            # Build port names including mode indices
            port_names_mm = []
            port_directions_mm = []
            for port_idx, mode_idx, mtype, m_n in mode_map:
                entry = entries[port_idx]
                port_names_mm.append(f"{entry.name}_mode{mode_idx}_{mtype}{m_n[0]}{m_n[1]}")
                port_directions_mm.append(entry.direction)
            _res_mm = WaveguideSMatrixResult(
                s_params=s_params,
                freqs=jnp.asarray(freqs),
                port_names=tuple(port_names_mm),
                port_directions=tuple(port_directions_mm),
                reference_planes=reference_planes,
            )
            return _finalize_sparam_result(
                _res_mm,
                extractor="compute_waveguide_s_matrix",
                strict=strict_passivity,
                passivity_tol=2.0 if normalize is False else 0.10,
            )

        # Single-mode path (original behavior)
        cfgs = raw_cfgs

        # Far-port discipline advisories for interior-PEC junction references.
        # Pure-numpy heuristics emitted BEFORE the FDTD runs; no simulation.
        if port_reference_sims is not None:
            _warn_junction_probe_clearance(
                grid, cfgs, materials.sigma,
                [m.sigma for m in ref_materials_per_port], freqs,
            )
            _warn_junction_cpml_thickness(
                grid, cfgs, freqs, self._cpml_layers,
            )

        def _slices_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return max(a[0], b[0]) < min(a[1], b[1])

        by_direction = {}
        for entry, cfg in zip(entries, cfgs):
            by_direction.setdefault(entry.direction, []).append(cfg)

        for direction, side_cfgs in by_direction.items():
            plane_indices = {cfg.x_index for cfg in side_cfgs}
            if len(plane_indices) != 1:
                raise ValueError(
                    f"waveguide ports on boundary {direction} must share one boundary plane"
                )
            for i in range(len(side_cfgs)):
                for j in range(i + 1, len(side_cfgs)):
                    if _slices_overlap((side_cfgs[i].u_lo, side_cfgs[i].u_hi), (side_cfgs[j].u_lo, side_cfgs[j].u_hi)) and _slices_overlap((side_cfgs[i].v_lo, side_cfgs[i].v_hi), (side_cfgs[j].v_lo, side_cfgs[j].v_hi)):
                        raise ValueError(
                            f"waveguide ports on the same {direction} boundary must have disjoint apertures"
                        )

        ref_shifts = []
        for entry, cfg in zip(entries, cfgs):
            # Default reference plane = the user-facing port plane
            # (snapped x_position). Previously defaulted to the internal
            # ``reference_x_m`` (= source + ref_offset·dx) which left the
            # returned S-matrix phase-shifted by `exp(-jβ·ref_offset·dx)`
            # relative to the physical port — a silent convention mismatch
            # vs. Meep, OpenEMS, and any analytic formula the user would
            # compare against. Keep the ``entry.reference_plane`` override
            # for explicit user control.
            planes = waveguide_plane_positions(cfg)
            desired_ref = (
                entry.reference_plane
                if entry.reference_plane is not None
                else planes["source"]
            )
            ref_shifts.append(desired_ref - planes["reference"])

        _pec_axes = "".join(axis for axis in "xyz" if axis not in grid.cpml_axes)
        # G-WI5 guardrail (mechanism corrected 2026-05-29): conformal=True
        # produces NaN S-params at fine mesh (min cell <= ~2 mm for WR-90).
        # The original 2026-05-24 diagnosis ("the two-run reference pass omits
        # conformal_weights") was FALSIFIED by two witnesses: threading
        # conformal_weights symmetrically into the reference run still NaNs,
        # and normalize=False (single-run, no reference) ALSO NaNs at dx=2 mm.
        # The real mechanism is the Dey-Mittra conformal-PEC run itself — the
        # E-update-only eps_eff=eps/w makes the update operator non-SPSD
        # (discrete-adjointness break), intrinsically unstable at fine dx and
        # independent of the normalisation mode.  Safe paths: conformal=False
        # (staircase PEC, the supported floor) or a coarser mesh.  The
        # ``conformal_nan`` preflight warning carries the same guidance;
        # the strict-xfail tracker is
        # tests/test_subpixel_pec.py::test_mesh_convergence_s21_with_conformal_pec.
        if conformal_weights is not None and normalize:
            import warnings as _w
            _w.warn(
                "compute_waveguide_s_matrix with conformal=True is KNOWN to "
                "produce NaN S-parameters at fine mesh spacings (min cell "
                "<= ~2 mm for WR-90): the Dey-Mittra conformal-PEC update is "
                "intrinsically unstable at fine dx (discrete-adjointness "
                "break) — this is independent of the normalisation mode, so "
                "normalize=False is NOT a safe workaround at fine dx.  Use "
                "conformal=False (staircase PEC) or a coarser mesh.  "
                "Tracked at https://github.com/bk-squared/rfx/issues.",
                UserWarning,
                stacklevel=3,
            )

        if normalize == "flux":
            from rfx.core.yee import init_materials as _init_vacuum_materials
            ref_materials = _init_vacuum_materials(grid.shape)
            s_params = extract_waveguide_s_matrix_flux(
                grid,
                materials,
                ref_materials,
                cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes=_pec_axes,
                debye=debye,
                lorentz=lorentz,
                ref_debye=None,
                ref_lorentz=None,
                ref_shifts=ref_shifts,
                aniso_eps=aniso_eps,
                ref_aniso_eps=ref_aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
                ref_aniso_inv_eps=ref_aniso_inv_eps,
                ref_materials_per_port=ref_materials_per_port,
                checkpoint_segments=checkpoint_segments,
                return_settling=True,
                sheet_impedance=_wg_sheet_ctx,
            )
            s_params, settling_db = s_params
        elif normalize:
            from rfx.core.yee import init_materials as _init_vacuum_materials
            ref_materials = _init_vacuum_materials(grid.shape)
            s_params = extract_waveguide_s_params_normalized(
                grid,
                materials,
                ref_materials,
                cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes=_pec_axes,
                debye=debye,
                lorentz=lorentz,
                ref_debye=None,
                ref_lorentz=None,
                ref_shifts=ref_shifts,
                aniso_eps=aniso_eps,
                ref_aniso_eps=ref_aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
                ref_aniso_inv_eps=ref_aniso_inv_eps,
                checkpoint_segments=checkpoint_segments,
                return_settling=True,
                sheet_impedance=_wg_sheet_ctx,
            )
            s_params, settling_db = s_params
        else:
            s_params, settling_db = extract_waveguide_s_matrix(
                grid,
                materials,
                cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes=_pec_axes,
                debye=debye,
                lorentz=lorentz,
                ref_shifts=ref_shifts,
                aniso_eps=aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
                checkpoint_segments=checkpoint_segments,
                return_settling=True,
                sheet_impedance=_wg_sheet_ctx,
            )
        reference_planes = np.array(
            [
                entry.reference_plane
                if entry.reference_plane is not None
                # Report the de-embed TARGET (the physical port/source plane, line 1042), not the
                # internal raw-extraction plane. Previously reported ["reference"] = source +
                # ref_offset·dx, so the metadata claimed a plane ref_offset cells off from where the
                # S-params are actually referenced (RF-audit 2026-07-23; matches the NU sibling).
                else waveguide_plane_positions(cfg)["source"]
                for entry, cfg in zip(entries, cfgs)
            ],
            dtype=float,
        )
        _port_names = tuple(entry.name for entry in entries)
        # Issue #538: the energy ring-down witness now reaches the waveguide
        # path — same aggregate truncation warning as the lumped/MSL path.
        # NaN entries (traced AD runs) are skipped by the warner's finite
        # mask; the array itself is always attached for the record.
        settling_db = np.asarray(settling_db, dtype=float)
        _warn_if_ringdown_truncated(
            settling_db, _port_names, num_periods=float(num_periods),
        )
        _res_sm = WaveguideSMatrixResult(
            s_params=s_params,
            freqs=jnp.asarray(freqs),
            port_names=_port_names,
            port_directions=tuple(entry.direction for entry in entries),
            reference_planes=reference_planes,
            settling_db=settling_db,
        )
        return _finalize_sparam_result(
            _res_sm,
            extractor="compute_waveguide_s_matrix",
            strict=strict_passivity,
            passivity_tol=2.0 if normalize is False else 0.10,
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
        """Compute the MSL S-matrix using N-probe numerical de-embedding.

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
        ``FENCE_REGISTRY`` in ``tests/test_sheet_lane_fences.py`` is the
        authoritative per-lane list, AST-guarded against drift.
        ``subpixel_smoothing`` / ``conformal_pec`` are ``run()``-only
        keywords, so that combination is unreachable on the ``eps_override``
        channel rather than refused there.
        The TRACE must remain PEC: an f0 sheet never enters
        ``pec_mask``, and the closed Ampere-loop current and the V span
        anchor on PEC trace nodes. The Hammerstad-Jensen beta/Z0 anchors
        and the real-beta N-probe fit assume a lossless line, so a sheet
        lying INSIDE a probed span adds per-length loss the fit cannot
        represent — reported ``Z0``/``q`` shift (the Z0 honesty guard may
        warn) while the V-I production S stays valid. A non-z-normal sheet
        (x/y-normal) carries Ez in its tangential edge set: crossing the
        feed plane or the V-integration column it legitimately modifies the
        measured/launched Ez — place sheets clear of feed and probe planes.

        For each registered MSL port, runs one FDTD simulation with that
        port driven and the others passive.  The passive ports are NOT
        assumed matched — measured ``|a_passive/a_driven| = 0.07-0.51``
        across three fixtures, so the S-matrix is recovered by solving the
        full wave system ``S = B·A⁻¹`` over all drives rather than by the
        per-column ratio ``b_j/a_d`` (issue #507; the ratio reported the far
        port's echo as the structure's own reflection).  At each port
        ``n_probes`` downstream DFT plane probes record Ez and the first
        probe also records Hy; β, Z0 and the wave amplitudes are extracted
        post-scan via the N-probe least-squares wave-decomposition
        extractor (issue #80 Fix C — SVD lstsq fit of
        ``V_n = α e^{-jβx_n} + γ e^{+jβx_n}`` anchored on the analytic
        Hammerstad-Jensen β guess) and assembled into the full S-matrix.
        The N-probe extractor removes the 3-probe quadratic's q→1
        singularity that produced wrong S11 resonances on thin-substrate
        patches.

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
            extractor, together with the production S-matrix, so the
            de-embedding can be independently checked without rerunning
            FDTD. The dump schema is ``rfx.msl_nprobe_dump`` v3 (issue #80
            Fix C; bumped 2->3 by issue #523 to add
            ``production_smatrix_assembly``); ``raw_v`` has shape
            ``(n_driven, n_ports, n_probes_max, n_freqs)``.
            ``scripts/diagnostics/replay_msl_3probe_dump.py`` is SUPERSEDED
            for v3 dumps (it expects the retired 3-probe/single-ratio v1
            schema); the current independent check is
            ``scripts/diagnostics/msl_vi_flux_oracle.py``.
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

        # Build the grid used for probe placement + the eps anchor. On the
        # non-uniform lane this MUST be the SAME grid run_nonuniform_path
        # builds (so probe_xs, port cells, dy/dz arrays and the eps anchor
        # align with the run's field planes). build_nonuniform_grid needs a
        # concrete dz_profile — synthesise from dx when absent into a LOCAL.
        # self._dz_profile is mutated only INSIDE the run try/finally below (so
        # the restore always runs even if probe placement / the trace-PEC scan
        # raises) — the subsequent self.run() then reads the same dz and builds
        # a byte-matching grid.
        _dz_profile_saved = self._dz_profile
        _dz_for_grid = self._dz_profile
        if is_nonuniform:
            from rfx.runners.nonuniform import build_nonuniform_grid
            if _dz_for_grid is None:
                _nz_syn = int(round(float(self._domain[2]) / float(self._dx)))
                _dz_for_grid = np.full(max(_nz_syn, 1), float(self._dx))
            grid = build_nonuniform_grid(
                self._freq_max, self._domain, self._dx, self._cpml_layers,
                _dz_for_grid,
                dx_profile=self._dx_profile, dy_profile=self._dy_profile,
                pec_faces=self._boundary_spec.pec_faces()
                    if self._boundary_spec is not None else None,
                pmc_faces=self._boundary_spec.pmc_faces()
                    if self._boundary_spec is not None else None,
                cpml_axes="".join(
                    ax for ax in "xyz"
                    if ax not in (self._periodic_axes or "")
                ),
            )
        else:
            grid = self._build_grid()

        if freqs is None:
            freqs_arr = np.asarray(jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs))
        else:
            freqs_arr = np.asarray(freqs)
        n_freqs_used = int(freqs_arr.shape[0])

        # Issue #469: solve the probe-offset interval for AUTO ports (the
        # downstream reflector term is only computable here, with the full
        # geometry registered — see _resolve_msl_auto_offsets).
        entries = _resolve_msl_auto_offsets(self, entries, grid)

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
        _msl_assembled = (
            self._assemble_materials_nu(grid) if is_nonuniform
            else self._assemble_materials(grid)
        )
        _msl_materials = _msl_assembled[0]
        _msl_pec_mask = (
            None if _msl_assembled[3] is None
            else np.asarray(_msl_assembled[3])
        )
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
            _sel = [0, 0, 0]
            _sel[meta["prop_idx"]] = meta["i_feed"]
            _sel[meta["width_idx"]] = meta["j_centre"]
            _sel[meta["normal_idx"]] = slice(meta["k_top"], None)
            col = (
                None if _msl_pec_mask is None
                else _msl_pec_mask[tuple(_sel)]
            )
            k_pec = np.array([], dtype=int) if col is None else np.where(col)[0]
            if k_pec.size == 0:
                raise RuntimeError(
                    "compute_msl_s_matrix: no PEC trace conductor found "
                    "above the substrate top for MSL port "
                    f"{entries[p_idx].name!r}; the closed Ampere-loop "
                    "current (issue #80 stage S1) needs the trace PEC. "
                    "Add the microstrip trace as a Box(material='pec'). "
                    "A surface_impedance_f0 thin conductor is NOT a trace "
                    "conductor here — it never enters pec_mask, and the "
                    "Ampere-loop current and V span anchor on PEC trace "
                    "nodes. Keep the trace PEC and use f0 sheets for "
                    "auxiliary lossy metal only."
                )
            trace_k_per_port.append((
                int(meta["k_top"] + int(k_pec.min())),
                int(meta["k_top"] + int(k_pec.max())),
            ))

        # Stash existing add_dft_plane_probe registrations and restore on exit.
        saved_dft = list(self._dft_planes)
        saved_msl = list(self._msl_ports)
        saved_ports = list(self._ports)
        saved_probes = list(self._probes)
        saved_internal_probes = set(self._internal_probe_indices)
        try:
            # Mutate self._dz_profile to the (possibly synthesised) grid dz only
            # now — inside the try — so the finally always restores it and the
            # subsequent self.run() builds a grid matching the one above.
            if is_nonuniform:
                self._dz_profile = _dz_for_grid
            _complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
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
                    run_entries.append(_MSLPortEntry(
                        name=pe.name, position=pe.position,
                        width=pe.width, height=pe.height,
                        direction=pe.direction, impedance=pe.impedance,
                        waveform=wf, excite=new_excite,
                        n_probe_offset=pe.n_probe_offset,
                        n_probe_spacing=pe.n_probe_spacing,
                        mode=pe.mode,
                        eps_r_sub=pe.eps_r_sub,
                    ))
                self._msl_ports = run_entries

                # Register DFT plane probes for V (Ez) and I (Hy).
                self._dft_planes = list(saved_dft)
                ez_probe_names: list[list[str]] = [[] for _ in range(n_ports)]
                hy_probe_names: list[str] = [None] * n_ports  # type: ignore
                hz_probe_names: list[str] = [None] * n_ports  # type: ignore
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
                        ez_probe_names[p_idx].append(nm)
                    nm_hy = f"_msl_run{driven}_p{p_idx}_{_meta_p['h_a']}"
                    self.add_dft_plane_probe(
                        axis=_plane_axis, coordinate=float(pxs[0]),
                        component=_meta_p["h_a"], freqs=jnp.asarray(freqs_arr),
                        name=nm_hy,
                    )
                    hy_probe_names[p_idx] = nm_hy
                    # H_b plane probe at probe 0 — the other leg pair of the
                    # closed Ampere-loop current (issue #80 stage S1).
                    nm_hz = f"_msl_run{driven}_p{p_idx}_{_meta_p['h_b']}"
                    self.add_dft_plane_probe(
                        axis=_plane_axis, coordinate=float(pxs[0]),
                        component=_meta_p["h_b"], freqs=jnp.asarray(freqs_arr),
                        name=nm_hz,
                    )
                    hz_probe_names[p_idx] = nm_hz

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
                    hy_plane = jnp.asarray(planes[hy_probe_names[p_idx]].accumulator)
                    hz_plane = jnp.asarray(planes[hz_probe_names[p_idx]].accumulator)
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
                        _a_lo, _a_hi = meta["j_lo"], meta["j_hi"]
                        _b_lo, _b_hi = k_tr_lo, k_tr_hi
                    else:
                        _ha = jnp.transpose(hy_plane, (0, 2, 1))
                        _hb = jnp.transpose(hz_plane, (0, 2, 1))
                        _a_lo, _a_hi = k_tr_lo, k_tr_hi
                        _b_lo, _b_hi = meta["j_lo"], meta["j_hi"]
                    i_f = msl_loop_current(
                        _ha, _hb,
                        j_lo=_a_lo, j_hi=_a_hi,
                        k_trace_lo=_b_lo, k_trace_hi=_b_hi,
                        dy_arr=meta["a_arr"], dz_arr=meta["b_arr"],
                        direction=msl_ports[p_idx].direction,
                    )
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
                        alpha_d = a_fwd_d
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
                # not: measured |a_passive/a_driven| = 0.07-0.51 across
                # three fixtures, so the far port's echo is reported as the
                # structure's own reflection (issue #507). The wave
                # amplitudes recorded below feed the multi-drive solve that
                # replaces this after the drive loop.
                for j in range(n_ports):
                    if j == driven:
                        continue
                    v0_p = v_per_port[j][0]
                    b_out_p = 0.5 * (
                        v0_p - z0_hj_per_port[j] * i_first_per_port[j]
                    )
                    S = S.at[j, driven, :].set(jnp.asarray(b_out_p, dtype=_complex_dtype) / (jnp.asarray(alpha_d, dtype=_complex_dtype) + 1e-30))

                # Record the FULL (a, b) pair at every port for this drive
                # (issue #507). ``a`` at a passive port is what the
                # single-ratio rule above assumes away.
                for j in range(n_ports):
                    v0_j = v_per_port[j][0]
                    z0_j = z0_hj_per_port[j]
                    i_j = i_first_per_port[j]
                    wave_a[driven][j] = jnp.asarray(
                        0.5 * (v0_j + z0_j * i_j), dtype=_complex_dtype)
                    wave_b[driven][j] = jnp.asarray(
                        0.5 * (v0_j - z0_j * i_j), dtype=_complex_dtype)

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

            # A deep standing-wave node can collapse both phasors at the
            # driven port plane.  The V·I ratio is then numerically ill
            # conditioned even though the underlying reflector is passive.
            # Preserve S exactly as computed and expose that blind spot as
            # per-port metadata (issue #337 follow-up).
            reliable = None
            try:
                # Cover EVERY (driven, port) record, not just the own-drive
                # diagonal (issue #522). The solve consumes the wave pair at
                # all n_drives x n_ports probe planes, so a collapse at a
                # PASSIVE port's plane during someone else's drive corrupts
                # the whole slice S[:, :, k] — and the diagonal-only mask
                # never saw it. Measured on a synthetic witness: poisoning
                # only the (drive 0, port 1) record moved |S21| by 0.92 at
                # that bin with the mask all-True, cond(A) = 1.28, S finite
                # and the honesty guard silent.
                #
                # Shape is unchanged, (n_ports, n_freqs), and so is the
                # meaning of the index: reliable[p, k] is False when PORT
                # p's plane collapsed at bin k in AT LEAST ONE drive. That
                # makes np.all(reliable, axis=0) genuinely sufficient for
                # "no plane the solve reads collapsed at this bin".
                #
                # The criterion is relative to each record's OWN band
                # median (see _msl_wave_split_reliability), so a uniformly
                # small passive record is not flagged wholesale — but deep
                # individual bins ARE. Live extractor runs on the two filter
                # geometries: 2/100 bins on msl_notch_e4 (and they are the
                # notch centre, 3.6273 GHz, recorded at -30.66 dB in the
                # committed fixture meta) and 12/120 on the Sheen LPF leg.
                # The counts need a re-run to check — the committed fixtures
                # store S magnitudes only, no V/I dump.
                # Correct behaviour — the split really is low-signal at a
                # -30 dB notch — but it costs a filter user their most
                # interesting bin; see the reliable docstring.
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
            # fitted β (analytic HJ anchor), so S is NOT condemned.
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
            # caveat — it does not impugn S11/S21.
            import warnings as _w

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
                    # the peak-|S11| bin is a flagged null, the correct root cause is the
                    # ill-conditioned V·I ratio there (both phasors collapse), NOT a current
                    # sign/scale error — attributing it to current mismeasurement misdiagnoses a
                    # legitimate passive strong reflector (RF-audit 2026-07-23). The guard still
                    # fires (the extracted value IS unreliable at that bin); only the cause differs.
                    at_null = reliable is not None and not bool(np.asarray(reliable)[driven, k_s])
                    cause = (
                        "that bin is flagged by the standing-wave-null reliability mask: a deep "
                        "node collapses both the V and I phasors, so the V·I ratio is "
                        "ill-conditioned (a numerical blind spot, not necessarily a current "
                        "sign/scale error)"
                        if at_null else
                        "the closed Ampere-loop current is likely mismeasured (sign/scale)"
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
                    _w.warn(
                        f"compute_msl_s_matrix: reported Z0 for MSL port "
                        f"{pe.name!r} = "
                        f"{float(np.asarray(jax.lax.stop_gradient(raw_z0[driven, driven, k_z])).real):.2f} ohm deviates "
                        f"{z0_dev_max * 100:.1f}% from analytic Hammerstad-"
                        f"Jensen {z0_hj:.2f} ohm at "
                        f"f = {freqs_arr[k_z] / 1e9:.4f} GHz. Z0 rides on the "
                        "retained N-probe fit (S1 transitional); on coarse "
                        "meshes this includes Yee-staircase bias. The V·I-"
                        "split S11/S21 are unaffected.",
                        stacklevel=2,
                    )

            if raw_3probe_dump_path is not None:
                import json
                from pathlib import Path

                path = Path(raw_3probe_dump_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "schema": "rfx.msl_nprobe_dump",
                    "schema_version": 3,
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
                    "current_convention": (
                        "line current sign normalized so +x and -x MSL ports "
                        "produce positive characteristic impedance on the "
                        "validated thru-line envelope"
                    ),
                    "deembedding": (
                        "N equally spaced voltage probes plus current at "
                        "probe 0. The reported Z0/beta come from the N-probe "
                        "least-squares wave-decomposition extractor (issue #80 "
                        "Fix C), which fits V_n = alpha*exp(-j beta x_n) + "
                        "gamma*exp(+j beta x_n) by SVD lstsq. The production "
                        "S-matrix does NOT come from that fit: it is solved "
                        "from the probe-0 wave amplitudes over all drives, "
                        "S = B @ inv(A) (issue #507), with the modal voltage "
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
            self._msl_ports = saved_msl
            self._ports = saved_ports
            self._probes = saved_probes
            self._internal_probe_indices = saved_internal_probes
            self._dz_profile = _dz_profile_saved

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
        """Mixed-family S-matrix: lumped/wire ports + MSL ports (issue #488).

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
        ``sqrt(Z_j/Z_i)`` (issue #460); reciprocity of a reciprocal
        structure is the committed internal falsifier for this choice.

        MAGNITUDE CHANNEL (``magnitude_channel``, default ``"flux"``):
        the #488 falsifier battery + an independent Poynting-flux referee
        measured the port-cell V*I accounting undercounting delivered
        power ~3x at a wire probe feed (the OPEN issue-#313 class) and
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
        (diagnostic; carries the #313 deflation at lumped/wire ports).

        HONESTY NOTES (read before quoting numbers):

        * **Neither diagonal is verified on this lane, and the returned
          diagonal is not always the measured one.** Two separate
          findings: (a) the wire port-cell V*I accounting was measured
          undercounting delivered power ~3x against an independent
          Poynting referee (the open issue #313 reaching the diagonal at
          a near-field-dominated vertical probe), and on an end-fed
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
        * With ``magnitude_channel="wave"``, off-diagonal magnitudes that
          RECEIVE at a lumped/wire port cell inherit the issue-#313
          near-field deflation of the default port-cell waves. The
          returned ``s21_power_witness`` cross-checks the MSL-receiving
          direction against a delivered-power normalization; quote it
          alongside ``|S|``.
        * Cross-family off-diagonal PHASE mixes two reference-plane
          conventions (port cell vs de-embedded MSL probe plane) and a
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
        import dataclasses as _dc

        from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
        from rfx.sources.msl_port import (
            MSLPort,
            _msl_yz_cells,
            msl_loop_current,
            msl_probe_x_coords_n,
        )

        # ---- Registration guards (v1 envelope) --------------------------
        if not self._msl_ports:
            raise ValueError(
                "compute_mixed_s_matrix() needs at least one add_msl_port() "
                "registration (for a pure lumped/wire multiport use the "
                "production scan driver / extract_s_matrix)."
            )
        lw_entries = [pe for pe in self._ports if pe.impedance != 0.0]
        if not lw_entries:
            raise ValueError(
                "compute_mixed_s_matrix() needs at least one sparam-eligible "
                "add_port() lumped/wire port (impedance != 0). For a pure "
                "MSL multiport use compute_msl_s_matrix()."
            )
        if any(pe.impedance == 0.0 for pe in self._ports):
            raise NotImplementedError(
                "compute_mixed_s_matrix() does not support bare sources / "
                "0-ohm ports (add_source or add_port(impedance=0)): they "
                "are not excite-gated and would fire in EVERY drive run, "
                "contaminating the single-drive S-parameter contract."
            )
        if self._waveguide_ports or self._floquet_ports:
            raise NotImplementedError(
                "compute_mixed_s_matrix() v1 covers lumped/wire + MSL only; "
                "waveguide/Floquet ports are not part of the validated "
                "mixed lane (issue #488)."
            )
        _mixed_non_x = [
            pe.name for pe in self._msl_ports
            if pe.direction not in ("+x", "-x")
        ]
        if _mixed_non_x:
            raise NotImplementedError(
                "compute_mixed_s_matrix() v1 covers '+x'/'-x' MSL ports "
                f"only; {_mixed_non_x} are not x-directed. The y-directed "
                "MSL lane landed in issue #661 for compute_msl_s_matrix(); "
                "this mixed lane is itself EXPERIMENTAL (issue #488) with "
                "both diagonals unverified, so it is fenced rather than "
                "extended untested. Use compute_msl_s_matrix() for a pure "
                "MSL multiport."
            )
        if self._coaxial_ports:
            raise NotImplementedError(
                "compute_mixed_s_matrix() v1 covers lumped/wire + MSL only; "
                "coaxial ports need a separate calibration contract."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_mixed_s_matrix() is not supported together with "
                "TFSF; TFSF is a plane-wave source, not a port."
            )
        is_wire = [pe.extent is not None for pe in lw_entries]
        if any(is_wire) and not all(is_wire):
            raise NotImplementedError(
                "compute_mixed_s_matrix(): mixed lumped + wire port sets "
                "are not supported (the off-diagonal wave-decomposition "
                "conventions differ — same fence as "
                "compute_lumped_wire_s_matrix_via_scan)."
            )
        wire_mode = all(is_wire)
        if any(getattr(pe, "reference_plane_cells", None) for pe in lw_entries):
            raise NotImplementedError(
                "compute_mixed_s_matrix() v1 does not support "
                "add_port(reference_plane_cells=...); the mixed lane uses "
                "the delivered-power witness for magnitude honesty instead."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_mixed_s_matrix() v1 supports the uniform mesh "
                "only (issue #488 scope: NU is explicitly out until the "
                "first pair ships)."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_mixed_s_matrix() is not supported with SBP-SAT "
                "subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_mixed_s_matrix() is not supported with "
                "solver='adi'; use the uniform Yee solver."
            )

        n_lw = len(lw_entries)
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

        # ---- MSL geometry prep (mirrors compute_msl_s_matrix; uniform
        # lane only, so the NU grid/dz-profile machinery is not needed) ----
        entries = _resolve_msl_auto_offsets(self, list(self._msl_ports), grid)
        n_msl = len(entries)
        msl_ports: list[MSLPort] = []
        for pe in entries:
            x_feed, y_centre, z_lo = pe.position
            msl_ports.append(MSLPort(
                feed_x=float(x_feed),
                y_lo=float(y_centre - pe.width / 2),
                y_hi=float(y_centre + pe.width / 2),
                z_lo=float(z_lo),
                z_hi=float(z_lo + pe.height),
                direction=pe.direction,
                impedance=pe.impedance,
                excitation=pe.waveform,
            ))
        # Probe-0 x-coordinate per port: the S1 V*I wave split lives at
        # probe 0 (the de-embedding reference plane); the mixed lane does
        # not run the N-probe spatial fit, so only probe 0 is recorded.
        probe_xs = [
            msl_probe_x_coords_n(
                grid, mp,
                n_probes=int(pe.n_probes),
                n_offset_cells=pe.n_probe_offset,
                n_spacing_cells=pe.n_probe_spacing,
            )
            for mp, pe in zip(msl_ports, entries)
        ]
        # Probe-LADDER validation (issue #488 attempt-1 defect D3). The S1
        # V*I split records probe 0 only, but v1 must not silently accept
        # a ladder the validated MSL lane rejects: attempt 1 registered a
        # "+x"-facing port whose default ladder (offset 31 + 4x12 cells)
        # ran past the declared domain; msl_probe_x_coords_n CLAMPS such
        # coordinates, and the surviving probe-0 plane sat 1.1 mm from
        # the trace's open end at the domain edge (a Box is rasterized
        # only inside the declared domain — the trace does NOT continue
        # into the CPML padding, so every boundary-touching trace has an
        # OPEN end there). The plane then measures the open-stub
        # standing-wave impedance -j*Z0*cot(beta*d) ~ 1/f (~1400 ohm at
        # 1 GHz vs the 48 ohm line) instead of a travelling wave.
        # Hard guard: every ladder coordinate strictly inside (0, lx)
        # and strictly monotonic (clamp shows up as duplicates).
        # Advisory: probe-0 closer than lambda_g/4 to a domain x-edge.
        from rfx.api._preflight import msl_min_probe_clearance
        _lx_dom = float(self._domain[0])
        _clear = msl_min_probe_clearance(float(self._freq_max))
        for pe, pxs in zip(entries, probe_xs):
            xs = [float(x) for x in pxs]
            mono = all(
                (xs[q + 1] - xs[q]) * (1 if pe.direction == "+x" else -1)
                > 0.5 * float(grid.dx)
                for q in range(len(xs) - 1)
            )
            if (not mono) or min(xs) <= 0.0 or max(xs) >= _lx_dom:
                raise ValueError(
                    f"compute_mixed_s_matrix: MSL port {pe.name!r} probe "
                    f"ladder ({', '.join(f'{x * 1e3:.2f}' for x in xs)} mm) "
                    f"leaves the declared x-domain (0, {_lx_dom * 1e3:.2f}) "
                    "mm or was clamped at its edge — the equivalent "
                    "registration is rejected by compute_msl_s_matrix, and "
                    "a plane near the trace's open end at the domain edge "
                    "measures the stub standing wave, not the line. Face "
                    "the port toward the DUT (direction), reduce "
                    "n_probe_offset/n_probe_spacing, or enlarge the domain."
                )
            _edge_d = min(xs[0], _lx_dom - xs[0])
            if _edge_d < _clear:
                import warnings as _w488
                _w488.warn(
                    f"compute_mixed_s_matrix: MSL port {pe.name!r} probe-0 "
                    f"plane is {_edge_d * 1e3:.2f} mm from a domain x-edge "
                    f"(< lambda_g/4 = {_clear * 1e3:.2f} mm at freq_max). "
                    "A boundary-touching trace has an OPEN end there; "
                    "standing waves from that discontinuity can bias the "
                    "V*I split (see the reliable mask).",
                    stacklevel=2,
                )

        dy_arr = _msl_cell_profile(grid, "y", grid.ny)
        dz_arr = _msl_cell_profile(grid, "z", grid.nz)
        port_idx_meta = []
        for mp in msl_ports:
            cells = _msl_yz_cells(grid, mp)
            j_set = sorted({c[1] for c in cells})
            k_set = sorted({c[2] for c in cells})
            j_lo, j_hi = j_set[0], j_set[-1]
            k_lo, k_hi = k_set[0], k_set[-1]
            port_idx_meta.append(dict(
                j_lo=j_lo, j_hi=j_hi, k_lo=k_lo, k_hi=k_hi,
                j_centre=(j_lo + j_hi) // 2, k_top=k_hi,
            ))

        # One materials assembly shared by the HJ eps anchor AND every
        # drive run (materials do not depend on excite flags).
        from rfx.materials.thin_conductor import refuse_f0_sheets as _refuse_f0_hj
        _refuse_f0_hj(self._thin_conductors, "MSL junction S-parameter")
        materials, debye_spec, lorentz_spec, pec_mask, _, _, _ = \
            self._assemble_materials(grid)
        pec_mask_np = None if pec_mask is None else np.asarray(pec_mask)

        # Analytic Hammerstad-Jensen anchor per MSL port (eps precedence
        # mirrors compute_msl_s_matrix: explicit eps_r_sub > rasterised
        # eps_r at the trace-centre substrate cell).
        z0_hj_per_port: list[float] = []
        beta0_per_port: list[np.ndarray] = []
        from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
        _c0_mixed = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
        for p_idx, pe in enumerate(entries):
            meta = port_idx_meta[p_idx]
            if pe.eps_r_sub is not None:
                eps_r_ref = float(pe.eps_r_sub)
            else:
                k_mid = (meta["k_lo"] + meta["k_hi"]) // 2
                i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
                eps_r_ref = float(np.asarray(
                    materials.eps_r[i_feed_p, meta["j_centre"], k_mid]
                ))
            z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(
                pe.width, pe.height, eps_r_ref
            )
            z0_hj_per_port.append(float(z0_hj))
            beta0_per_port.append(
                2.0 * np.pi * freqs_arr * float(np.sqrt(eps_eff_hj))
                / _c0_mixed
            )

        # Trace-conductor z-cell span (closed Ampere loop needs the PEC
        # trace; mirrors compute_msl_s_matrix issue #80 stage S1).
        trace_k_per_port: list[tuple[int, int]] = []
        for p_idx in range(n_msl):
            meta = port_idx_meta[p_idx]
            i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
            col = (
                None if pec_mask_np is None
                else pec_mask_np[i_feed_p, meta["j_centre"], meta["k_top"]:]
            )
            k_pec = np.array([], dtype=int) if col is None else np.where(col)[0]
            if k_pec.size == 0:
                raise RuntimeError(
                    "compute_mixed_s_matrix: no PEC trace conductor found "
                    "above the substrate top for MSL port "
                    f"{entries[p_idx].name!r}; the closed Ampere-loop "
                    "current needs the trace PEC. Add the microstrip trace "
                    "as a Box(material='pec')."
                )
            trace_k_per_port.append((
                int(meta["k_top"] + int(k_pec.min())),
                int(meta["k_top"] + int(k_pec.max())),
            ))

        # Wire live-cell counts for the per-cell impedance normalization
        # (mirrors compute_lumped_wire_s_matrix_via_scan, issue #318).
        n_live_lw = np.ones(n_lw, dtype=np.int64)
        if wire_mode:
            from rfx.sources.sources import WirePort, _wire_port_live_cells
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            for idx, pe in enumerate(lw_entries):
                end = list(pe.position)
                end[axis_map[pe.component]] += pe.extent
                wp = WirePort(
                    start=pe.position, end=tuple(end),
                    component=pe.component, impedance=pe.impedance,
                    excitation=pe.waveform,
                )
                n_live_lw[idx] = _wire_port_live_cells(grid, wp, pec_mask)[2]

        if not skip_preflight:
            # One preflight for the full registration (run() would fire it
            # per drive run — 2*n_ports repeats of the same advisories).
            self.preflight()

        if magnitude_channel not in ("flux", "wave"):
            raise ValueError(
                "compute_mixed_s_matrix: magnitude_channel must be 'flux' "
                f"(default) or 'wave', got {magnitude_channel!r}."
            )
        if magnitude_channel == "flux":
            # The per-port flux box is CLOSED by the z-lo PEC ground: it
            # has five faces and omits the bottom because flux through a
            # PEC face is identically zero. It also treats pe.extent as a
            # VERTICAL height. Both are physical preconditions, so check
            # them instead of assuming (review finding: a user with an
            # open z-lo boundary or a horizontal port would otherwise get
            # a silently wrong P_net on the DEFAULT channel).
            _pec_faces = (
                self._boundary_spec.pec_faces()
                if self._boundary_spec is not None else set()
            )
            if "z_lo" not in _pec_faces:
                raise NotImplementedError(
                    "compute_mixed_s_matrix(magnitude_channel='flux') "
                    "requires a PEC z_lo boundary: the per-port flux box "
                    "omits its bottom face because flux through a PEC "
                    "ground is identically zero. Your z_lo face is "
                    f"{sorted(_pec_faces) or 'not PEC'}, so the box would "
                    "not be closed and P_net would be wrong. Use "
                    "BoundarySpec(z=Boundary(lo='pec', ...)), or pass "
                    "magnitude_channel='wave' (which carries the #313 "
                    "port-cell deflation instead)."
                )
            _bad = [pe for pe in lw_entries if pe.component != "ez"]
            if _bad:
                raise NotImplementedError(
                    "compute_mixed_s_matrix(magnitude_channel='flux') "
                    "supports vertical (component='ez') lumped/wire ports "
                    "only: the flux box is built assuming the port extent "
                    "is a z height above the ground plane. Offending "
                    f"component(s): {sorted({pe.component for pe in _bad})}."
                )

        drive_plan = [("lw", j) for j in range(n_lw)] + \
                     [("msl", d) for d in range(n_msl)]
        n_runs = len(drive_plan)
        _complex_dtype = (
            jnp.complex128 if jax.config.x64_enabled else jnp.complex64
        )

        saved_dft = list(self._dft_planes)
        saved_msl = list(self._msl_ports)
        saved_ports = list(self._ports)
        saved_probes = list(self._probes)
        saved_internal = set(self._internal_probe_indices)
        saved_flux = list(self._flux_monitors)
        try:
            # Ring-down settling witness probes at every MSL probe plane,
            # mid-substrate (worst plane wins — a single plane is
            # standing-wave-node sensitive; mirrors compute_msl_s_matrix).
            _witness_base = len(self._probes)
            _witness_total = 0
            for pe_w, pxs_w in zip(entries, probe_xs):
                for _x_w in pxs_w:
                    self.add_probe(
                        position=(
                            float(_x_w),
                            float(pe_w.position[1]),
                            float(pe_w.position[2]) + 0.5 * float(pe_w.height),
                        ),
                        component="ez",
                    )
                    _witness_total += 1
            self._internal_probe_indices.update(
                range(_witness_base, _witness_base + _witness_total)
            )
            witness_probes = list(self._probes)

            # Flux surfaces for the magnitude channel (registered once —
            # geometry is drive-independent). Lumped/wire port p: a closed
            # 5-face box over the z-lo ground plane (flux through the PEC
            # ground face is identically zero); outward-signed face list.
            # MSL port p: one full-cross-section x-plane at probe 0 (its
            # de-embedding reference plane).
            flux_faces_lw: list[list[tuple[str, float]]] = []
            flux_names_msl: list[str] = []
            if magnitude_channel == "flux":
                _m = 3.0 * float(grid.dx)
                for p, pe in enumerate(lw_entries):
                    x0_p, y0_p, _z0p = (float(c) for c in pe.position)
                    z_top = float(pe.extent or grid.dx) + _m
                    faces = []
                    for ax, coord, size, center, sgn in (
                        ("x", x0_p - _m, (2 * _m, z_top), (y0_p, z_top / 2), -1.0),
                        ("x", x0_p + _m, (2 * _m, z_top), (y0_p, z_top / 2), +1.0),
                        ("y", y0_p - _m, (2 * _m, z_top), (x0_p, z_top / 2), -1.0),
                        ("y", y0_p + _m, (2 * _m, z_top), (x0_p, z_top / 2), +1.0),
                        ("z", z_top, (2 * _m, 2 * _m), (x0_p, y0_p), +1.0),
                    ):
                        nm = f"_mixed_flux_lw{p}_{ax}{'+' if sgn > 0 else '-'}{coord:.6g}"
                        self.add_flux_monitor(
                            axis=ax, coordinate=float(coord),
                            freqs=jnp.asarray(freqs_arr),
                            size=(float(size[0]), float(size[1])),
                            center=(float(center[0]), float(center[1])),
                            name=nm,
                        )
                        faces.append((nm, sgn))
                    flux_faces_lw.append(faces)
                for p, pxs in enumerate(probe_xs):
                    nm = f"_mixed_flux_msl{p}"
                    self.add_flux_monitor(
                        axis="x", coordinate=float(pxs[0]),
                        freqs=jnp.asarray(freqs_arr), name=nm,
                    )
                    flux_names_msl.append(nm)

            v_lw = np.zeros((n_runs, n_lw, n_freqs_used), dtype=np.complex128)
            i_lw = np.zeros((n_runs, n_lw, n_freqs_used), dtype=np.complex128)
            v0_msl = np.zeros((n_runs, n_msl, n_freqs_used), dtype=np.complex128)
            i_msl = np.zeros((n_runs, n_msl, n_freqs_used), dtype=np.complex128)
            _n_probes_max = max(len(pxs) for pxs in probe_xs)
            v_lad = np.zeros(
                (n_runs, n_msl, _n_probes_max, n_freqs_used),
                dtype=np.complex128,
            )
            settling_db_runs = np.full(n_runs, np.nan)
            # Signed per-run flux accountings (magnitude_channel="flux"):
            # box_lw[run, p]  = net OUTWARD box flux at lumped/wire port p
            # plane_msl[run, p] = raw +x-directed flux at MSL port p's plane
            box_lw = np.zeros((n_runs, n_lw, n_freqs_used))
            plane_msl = np.zeros((n_runs, n_msl, n_freqs_used))

            for run_idx, (fam, loc) in enumerate(drive_plan):
                # Excite exactly one port; every other port keeps its
                # physical termination (matched resistor cells for
                # lumped/wire, passive probe column for MSL). The driven
                # lumped/wire port's default waveform is synthesised by
                # the runner (issue #322); the driven MSL port mirrors the
                # compute_msl_s_matrix default.
                self._ports = [
                    _dc.replace(pe, excite=(fam == "lw" and k == loc))
                    for k, pe in enumerate(saved_ports)
                ]
                run_msl = []
                for k, pe in enumerate(saved_msl):
                    driven = fam == "msl" and k == loc
                    wf = (
                        (pe.waveform if pe.waveform is not None else
                         GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8))
                        if driven else None
                    )
                    run_msl.append(_dc.replace(pe, excite=driven, waveform=wf))
                self._msl_ports = run_msl
                self._probes = list(witness_probes)

                # Per-run named DFT planes: Ez at probe 0 (line voltage),
                # Hy+Hz at probe 0 (closed Ampere-loop current legs).
                self._dft_planes = list(saved_dft)
                names = []
                for p_idx, pxs in enumerate(probe_xs):
                    nm = f"_mixed_run{run_idx}_p{p_idx}"
                    # Full ez ladder: probe 0 feeds the V*I wave split;
                    # the N-probe least-squares fit (measured line Zc for
                    # the MSL diagonal — see the z0_fit block below) needs
                    # every plane.
                    for q_idx, x_c in enumerate(pxs):
                        self.add_dft_plane_probe(
                            axis="x", coordinate=float(x_c), component="ez",
                            freqs=jnp.asarray(freqs_arr),
                            name=nm + f"_ez{q_idx}",
                        )
                    self.add_dft_plane_probe(
                        axis="x", coordinate=float(pxs[0]), component="hy",
                        freqs=jnp.asarray(freqs_arr), name=nm + "_hy",
                    )
                    self.add_dft_plane_probe(
                        axis="x", coordinate=float(pxs[0]), component="hz",
                        freqs=jnp.asarray(freqs_arr), name=nm + "_hz",
                    )
                    names.append(nm)

                raw = self._forward_from_materials(
                    grid, materials, debye_spec, lorentz_spec,
                    n_steps=n_steps, checkpoint=False, pec_mask=pec_mask,
                    port_s11_freqs=freqs_arr,
                    _return_raw_port_sparams=True,
                )
                accs = raw["wire"] if wire_mode else raw["lumped"]
                if accs is None or len(accs) != n_lw:
                    raise RuntimeError(
                        "compute_mixed_s_matrix: production scan returned "
                        f"{0 if accs is None else len(accs)} "
                        f"{'wire' if wire_mode else 'lumped'} accumulators "
                        f"for run {run_idx}, expected {n_lw}."
                    )
                for i_port in range(n_lw):
                    _spec_i, vi = accs[i_port]
                    v_lw[run_idx, i_port, :] = np.asarray(vi[0])
                    i_lw[run_idx, i_port, :] = np.asarray(vi[1])

                planes = raw.get("dft_planes")
                if not planes:
                    raise RuntimeError(
                        "compute_mixed_s_matrix: the production scan "
                        "returned no DFT planes — the issue-#488 raw hook "
                        "is out of sync with _forward_from_materials."
                    )

                if magnitude_channel == "flux":
                    from rfx.probes.probes import flux_spectrum
                    fmon = raw.get("flux_monitors")
                    if not fmon:
                        raise RuntimeError(
                            "compute_mixed_s_matrix: the production scan "
                            "returned no flux monitors — the issue-#488 "
                            "flux hook is out of sync with "
                            "_forward_from_materials."
                        )
                    for p, faces in enumerate(flux_faces_lw):
                        acc = np.zeros(n_freqs_used)
                        for nm, sgn in faces:
                            acc = acc + sgn * np.asarray(
                                flux_spectrum(fmon[nm]), dtype=np.float64
                            )
                        box_lw[run_idx, p, :] = acc
                    for p, nm in enumerate(flux_names_msl):
                        plane_msl[run_idx, p, :] = np.asarray(
                            flux_spectrum(fmon[nm]), dtype=np.float64
                        )

                # Settling witness from the probe time series (worst
                # end/peak Ez^2 across the MSL probe planes).
                _ts = raw.get("time_series")
                if _ts is not None and not is_tracer(_ts):
                    _ts_np = np.asarray(
                        _ts[:, _witness_base:_witness_base + _witness_total],
                        dtype=float,
                    )
                    if (_ts_np.shape[0] >= 10
                            and _ts_np.shape[1] == _witness_total):
                        _p = _ts_np ** 2
                        _tail = max(1, _p.shape[0] // 10)
                        _end = _p[-_tail:, :].mean(axis=0)
                        _peak = _p.max(axis=0)
                        _tiny = np.finfo(float).tiny
                        settling_db_runs[run_idx] = float(np.max(
                            10.0 * np.log10((_end + _tiny) / (_peak + _tiny))
                        ))

                # MSL line V (probe-0 plane) + closed-loop I, with the
                # leapfrog E/H half-step correction (mirrors
                # compute_msl_s_matrix line-for-line; see that method for
                # the full derivation comments).
                _hs_phase = jnp.exp(
                    1j * 2.0 * jnp.pi * jnp.asarray(freqs_arr)
                    * (float(grid.dt) * 0.5)
                )
                for p_idx, meta in enumerate(port_idx_meta):
                    nm = names[p_idx]
                    for q_idx in range(len(probe_xs[p_idx])):
                        ez_plane = jnp.asarray(
                            planes[nm + f"_ez{q_idx}"].accumulator
                        )
                        # Anchor the V-span top on the rasterized trace
                        # node, not round(h_sub/dx) — see the sibling
                        # comment in compute_msl_s_matrix (PR #516 F2).
                        v_q = msl_modal_voltage(
                            ez_plane, j_centre=meta["j_centre"],
                            k_lo=meta["k_lo"],
                            k_hi=trace_k_per_port[p_idx][0],
                            dz_arr=dz_arr, dtype=_complex_dtype,
                        )
                        v_lad[run_idx, p_idx, q_idx, :] = np.asarray(v_q)
                    v_f = jnp.asarray(v_lad[run_idx, p_idx, 0, :])
                    hy_plane = jnp.asarray(planes[nm + "_hy"].accumulator)
                    hz_plane = jnp.asarray(planes[nm + "_hz"].accumulator)
                    hy_plane = hy_plane * _hs_phase[:, None, None].astype(hy_plane.dtype)
                    hz_plane = hz_plane * _hs_phase[:, None, None].astype(hz_plane.dtype)
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
            )
            S = jnp.asarray(S, dtype=_complex_dtype)

            # ---- N-probe line Zc: DIAGNOSTIC ONLY (issue #488) ---------
            # This lane does NOT substitute a fitted Zc into the MSL
            # diagonal. An earlier revision did, and it was withdrawn
            # after review: the fit's sign is not stable here, so the
            # substitution silently fell back to the analytic anchor and
            # looked like a confirmed measurement.
            #
            # Mechanism (why a sign fix would not be enough): the model
            # V_n = alpha*exp(-j beta x_n) + gamma*exp(+j beta x_n) is
            # invariant under (beta -> -beta, alpha <-> gamma), so a beta
            # scan that lands on the wrong branch SWAPS the two wave
            # roles and flips the sign of (alpha - gamma) — and therefore
            # of z0 = (alpha - gamma)/I1. Measured on one fixture: the
            # fitted sign differs between num_periods=4 and 20.
            #
            # The SECOND half of what this comment used to allege — that
            # `msl_loop_current`'s docstring ("the returned I is positive
            # for a forward quasi-TEM wave") and compute_msl_s_matrix's
            # #140 dir_sign comment ("a -x port's fitted z0 inherits a
            # negative sign") describe OPPOSITE conventions — was
            # measured during issue #661 and is a PROSE defect only, now
            # corrected in msl_loop_current's docstring. Own-drive
            # diagonal on the committed thru fixture:
            # Re((alpha-gamma)/I1) = +57.52 ohm at the "+x" port and
            # -57.56 ohm at the "-x" port (same magnitude to 0.08%), so
            # the #140 comment described the code and the docstring's
            # blanket claim was scoped wrong. The lane is self-consistent
            # about it: |S11|=|S22| to 5 decimals, reciprocity 1.27e-05,
            # column power <= 0.99998, both reported Z0 positive. No code
            # changed. The beta-branch instability above is separate and
            # still stands.
            #
            # Issue #524 remains open for its other two items (the
            # passive port's ~30 ohm termination reading and the
            # 0.194-vs-0.073 drive asymmetry, both orphaned from #507);
            # see #524.
            #
            # The fit is still computed and EXPOSED (return_diagnostics)
            # because it is the only handle on the open 30-vs-48 ohm
            # question, but it never feeds a shipped number, and its
            # magnitude is reported without a sign claim.
            from rfx.probes.msl_wave_decomp import extract_msl_nprobe
            z0_msl_fit = np.full((n_msl, n_freqs_used), np.nan)
            beta_railed_msl = np.zeros((n_msl, n_freqs_used), dtype=bool)
            for d in range(n_msl):
                run_d = n_lw + d
                n_p = len(probe_xs[d])
                res_fit = extract_msl_nprobe(
                    jnp.asarray(v_lad[run_d, d, :n_p, :].T),
                    jnp.asarray(np.asarray(probe_xs[d], dtype=float)),
                    jnp.asarray(i_msl[run_d, d, :]),
                    jnp.asarray(beta0_per_port[d]),
                    z0_hj=z0_hj_per_port[d],
                )
                z0_msl_fit[d, :] = np.abs(np.asarray(
                    jax.lax.stop_gradient(res_fit["z0"])
                ).astype(np.complex128))
                beta_railed_msl[d, :] = np.asarray(
                    jax.lax.stop_gradient(res_fit["beta_railed"])
                ).astype(bool)
                _dev = float(np.max(
                    np.abs(z0_msl_fit[d, :] - z0_hj_per_port[d])
                    / z0_hj_per_port[d]
                ))
                if _dev > 0.10:
                    import warnings as _wz
                    _wz.warn(
                        f"compute_mixed_s_matrix: the N-probe |Zc| fit for "
                        f"MSL port {entries[d].name!r} deviates up to "
                        f"{_dev * 100:.1f}% from analytic Hammerstad-Jensen "
                        f"{z0_hj_per_port[d]:.2f} ohm. The MSL diagonal "
                        "here uses the ANALYTIC anchor, so this does not "
                        "change the returned S — it is a diagnostic that "
                        "the record may be under-settled (check "
                        "settling_db) or the discretized line genuinely "
                        "differs (Yee staircase on coarse meshes).",
                        stacklevel=2,
                    )
            # β-scan rail flags for the diagnostic N-probe fit (issue
            # #681): a railed bin's |Zc|/β above are the ±35% scan-window
            # limit, not a measurement. S is unaffected (the MSL diagonal
            # uses the analytic HJ anchor).
            _warn_msl_beta_scan_railed(
                beta_railed_msl, freqs_arr,
                tuple(pe.name for pe in entries),
            )
            s_wave_full = None

            if magnitude_channel == "flux":
                import warnings as _wf
                s_wave_full = np.asarray(jax.lax.stop_gradient(S))
                msl_away = [
                    (+1.0 if pe.direction == "+x" else -1.0) for pe in entries
                ]
                S, _ill_cond, _neg_power = _mixed_flux_magnitude_override(
                    S, box_lw, plane_msl, drive_plan, msl_away, n_lw,
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
                            "are UNRELIABLE at those bins.",
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
                            "not as an absence of coupling.",
                            stacklevel=2,
                        )

            # MSL standing-wave-null reliability from each MSL port's own
            # driven run (mirrors compute_msl_s_matrix issue #337).
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
            z0_ref = np.asarray(
                [float(pe.impedance) for pe in lw_entries]
                + z0_hj_per_port
            )

            s_raw = None
            passivity_correction = None
            if enforce_passivity and not is_tracer(S):
                s_projected, correction = _project_passive(S)
                if bool(np.any(np.asarray(correction) > 0.0)):
                    s_raw = S
                    passivity_correction = correction
                    S = s_projected

            result = MixedSMatrixResult(
                S=S,
                freqs=np.asarray(freqs_arr),
                port_names=port_names,
                port_families=port_families,
                z0_ref=z0_ref,
                settling_db=settling_db_runs,
                s21_power_witness=s21_power,
                reliable=reliable,
                S_raw=s_raw,
                passivity_correction=passivity_correction,
                S_wave=s_wave_full,
                magnitude_channel=magnitude_channel,
                beta_railed=beta_railed_msl,
            )
            _warn_if_ringdown_truncated(
                settling_db_runs, port_names, num_periods=num_periods,
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
                        "settling_db before quoting any |S|.",
                        stacklevel=2,
                    )
            if return_diagnostics:
                # R5 inspection surface: the raw per-run phasors behind
                # every wave, so a suspicious |S| can be traced to V/I
                # health (e.g. a broken Ampere loop shows as v0/i far
                # from the line Z0) without re-running.
                return result, {
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
            return result
        finally:
            self._dft_planes = saved_dft
            self._msl_ports = saved_msl
            self._ports = saved_ports
            self._probes = saved_probes
            self._internal_probe_indices = saved_internal
            self._flux_monitors = saved_flux

    def compute_coaxial_s_matrix(
        self,
        *,
        n_steps: int = 320,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 21,
        field_scale: float = 1.0e4,
        magnetic_ratio: float = 1.0,
        signal_floor: float = 1.0e-12,
        reference_plane_axial_index_offset: int = 0,
        strict_passivity: bool = False,
    ) -> "CoaxialSMatrixResult":
        """Experimental coaxial S-matrix via distributed TEM plane sources.

        .. deprecated::
            This single-plane V/I path measures inside a closed PEC box around a
            short coaxial stub, which has no transmission line for a clean
            reflection — it reports non-physical ``|S11|>1`` for a lossless
            short (verified). Use :meth:`compute_coaxial_line_reflection`, which
            builds a real coax line with a matched CPML feed and extracts the
            reflection from a multi-plane matrix-pencil decomposition (validated
            short→Γ=-1, open→|Γ|=1, matched→0 across the band). This method is
            retained only for backward compatibility.

        For each registered ``add_coaxial_port(...)`` port, runs one FDTD
        simulation with that port driven and all other coaxial ports passive.
        A distributed transverse E/M plane source is injected on the port's
        cross-section (the M67 prototype scaffold promoted to the public
        API); DFT plane probes capture the resulting Ex/Ey/Hx/Hy on every
        coaxial port's reference plane; the V/I extractor recovers ``V`` and
        ``I`` via the radial line / azimuthal loop integrals; and the
        standard power-wave decomposition assembles the full S-matrix.

        Status: **experimental**. The plane source can produce a residual
        forward wave and the extracted reference-plane V/I has known
        amplitude bias for coarse grids; ``status="degraded"`` is reported
        when any V/I sample falls below ``signal_floor``. Use this API for
        development; do not promote claims beyond E2/E3 without an external
        cross-solver fixture (see ``port_external_reference_requirements``).

        Parameters
        ----------
        n_steps:
            FDTD timesteps per driven-port run. Default 320.
        freqs:
            Frequency grid (Hz). Defaults to a uniform grid covering
            ``[freq_max / 10, freq_max]``.
        n_freqs:
            Number of frequencies if ``freqs`` is None. Default 21.
        field_scale:
            Linear scale on the radial E waveform. Increase to lift the
            plane signal above DFT noise (V/I extraction is amplitude-linear
            so the S-matrix is invariant under this scale).
        magnetic_ratio:
            Multiplier on the ``H`` waveform after the analytic ``1/Z_TEM``
            factor. ``1.0`` injects the lossless-TEM Poynting-balanced
            amplitude; smaller values bias toward an E-only injection.
        signal_floor:
            Absolute V or I phasor magnitude below which the result is
            flagged as ``"degraded"``.
        reference_plane_axial_index_offset:
            Axial-index offset for the source/probe plane relative to the
            port pin centre.

        Returns
        -------
        CoaxialSMatrixResult
        """

        import warnings
        warnings.warn(
            "compute_coaxial_s_matrix() (single-plane V/I in a closed PEC box) is "
            "deprecated and reports non-physical |S11|>1 for a lossless short; use "
            "compute_coaxial_line_reflection() (validated coax-line method). "
            "It will be removed in rfx v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )

        from rfx.probes.probes import init_dft_plane_probe
        from rfx.simulation import run as _run
        from rfx.sources.coaxial_port import (
            build_coaxial_tem_plane_source_specs,
            extract_coaxial_plane_vi_from_dft,
        )

        if not self._coaxial_ports:
            raise ValueError(
                "No coaxial ports registered. Call add_coaxial_port() first."
            )
        if (
            self._ports
            or self._waveguide_ports
            or self._floquet_ports
            or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is defined only for "
                "add_coaxial_port(...) families in the current simulation."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with TFSF; "
                "TFSF is a plane-wave source, not a coaxial port."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() supports the uniform Yee lane only."
            )

        # Issue #704 audit: same silent NTFF drop class as the MSL path.
        _warn_ntff_box_dropped(self, "compute_coaxial_s_matrix()")

        ports: list[CoaxialPort] = list(self._coaxial_ports)
        n_ports = len(ports)

        # Build the working grid + materials with all coaxial geometries
        # stamped (PEC center pin, PTFE dielectric fill, PEC outer shell from
        # M66). ``_build_materials`` only assembles bulk materials and shapes;
        # ``add_coaxial_port`` only registers the port descriptor, so without
        # this loop the FDTD would run with the source dropped into pure
        # vacuum and the wave would radiate bidirectionally with no coax
        # structure to confine it (this is the real source of the
        # calibration-blocked status documented in the handover).
        from rfx.sources.coaxial_port import (
            setup_coaxial_port,
            add_coaxial_matched_termination,
            add_coaxial_open_termination,
            add_coaxial_pec_end_cap,
        )
        grid = self._build_grid()
        materials, _, _ = self._build_materials(grid)
        for p in ports:
            materials = setup_coaxial_port(grid, p, materials)
        for term_port_idx, term_R, term_offset_cells in self._coaxial_terminations:
            materials = add_coaxial_matched_termination(
                grid,
                ports[term_port_idx],
                materials,
                target_impedance=term_R,
                axial_offset_cells=term_offset_cells,
            )
        for term_port_idx, retract_cells in self._coaxial_open_terminations:
            materials = add_coaxial_open_termination(
                grid,
                ports[term_port_idx],
                materials,
                pin_retract_cells=retract_cells,
            )
        for cap_port_idx, cap_offset_cells in self._coaxial_pec_end_caps:
            materials = add_coaxial_pec_end_cap(
                grid,
                ports[cap_port_idx],
                materials,
                axial_offset_cells=cap_offset_cells,
            )

        # Frequency grid.
        if freqs is None:
            freqs = jnp.linspace(
                self._freq_max / 10.0,
                self._freq_max,
                int(n_freqs),
                dtype=jnp.float32,
            )
        else:
            freqs = jnp.asarray(freqs, dtype=jnp.float32)

        # Reference-plane axial indices per port (cross-section z-plane).
        from rfx.sources.coaxial_port import _coaxial_port_geometry
        plane_indices: list[int] = []
        for p in ports:
            _, _, _, pin_center, _, _ = _coaxial_port_geometry(grid, p)
            plane_indices.append(
                int(grid.position_to_index(pin_center)[2])
                + int(reference_plane_axial_index_offset)
            )

        # Output buffers.
        n_freqs_used = int(freqs.shape[0])
        s = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)
        z_tem_arr = np.zeros((n_ports, n_freqs_used), dtype=np.complex128)
        v_dump = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)
        i_dump = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)

        status = "passed"

        for driven in range(n_ports):
            spec = build_coaxial_tem_plane_source_specs(
                grid=grid,
                port=ports[driven],
                n_steps=int(n_steps),
                field_scale=float(field_scale),
                magnetic_ratio=float(magnetic_ratio),
                reference_plane_axial_index_offset=int(
                    reference_plane_axial_index_offset
                ),
            )
            z_tem_arr[driven, :] = complex(spec.z_tem_ohm)

            # DFT plane probes on every port's cross-section.
            dft_planes = []
            for p_idx, p in enumerate(ports):
                for component in ("ex", "ey", "hx", "hy"):
                    dft_planes.append(
                        init_dft_plane_probe(
                            axis=2,
                            index=plane_indices[p_idx],
                            component=component,
                            freqs=freqs,
                            grid_shape=grid.shape,
                            dft_total_steps=int(n_steps),
                        )
                    )

            result = _run(
                grid,
                materials,
                int(n_steps),
                # Honour the simulation boundary (was hardcoded "pec", the closed
                # box that is the documented root cause); self._boundary is always
                # a str ("pec"/"cpml"/"upml"), even for the BoundarySpec path.
                boundary=self._boundary,
                sources=list(spec.electric_sources),
                mag_sources=list(spec.magnetic_sources),
                dft_planes=dft_planes,
                return_state=False,
            )
            if result.dft_planes is None:
                raise RuntimeError(
                    "compute_coaxial_s_matrix(): runner returned no DFT planes"
                )

            # Slice DFT planes back into per-port (ex, ey, hx, hy) groups.
            per_port: list[dict[str, np.ndarray]] = []
            for p_idx in range(n_ports):
                start = p_idx * 4
                group = result.dft_planes[start : start + 4]
                comp_map = {
                    probe.component: np.asarray(probe.accumulator, dtype=np.complex128)
                    for probe in group
                }
                per_port.append(comp_map)

            # Extract V/I at each port's reference plane.
            voltages = []
            currents = []
            for p_idx, p in enumerate(ports):
                vi = extract_coaxial_plane_vi_from_dft(
                    grid=grid,
                    port=p,
                    plane_axial_index=plane_indices[p_idx],
                    ex_dft=per_port[p_idx]["ex"],
                    ey_dft=per_port[p_idx]["ey"],
                    hx_dft=per_port[p_idx]["hx"],
                    hy_dft=per_port[p_idx]["hy"],
                )
                v = np.asarray(vi.vi.voltage, dtype=np.complex128)
                i = np.asarray(vi.vi.current, dtype=np.complex128)
                voltages.append(v)
                currents.append(i)
                v_dump[driven, p_idx, :] = v
                i_dump[driven, p_idx, :] = i
                if (
                    float(np.max(np.abs(v))) <= float(signal_floor)
                    or float(np.max(np.abs(i))) <= float(signal_floor)
                ):
                    status = "degraded"

            # Power-wave decomposition at each receive port (a_j at driven, b_i
            # at receiver) using the analytic Z_TEM as Z0.
            z0 = complex(spec.z_tem_ohm)
            a_j = (voltages[driven] + z0 * currents[driven]) / (2.0 * np.sqrt(z0))
            for receiver in range(n_ports):
                b_i = (voltages[receiver] - z0 * currents[receiver]) / (
                    2.0 * np.sqrt(z0)
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    s[receiver, driven, :] = np.where(
                        np.abs(a_j) > 0.0,
                        b_i / a_j,
                        np.nan + 1j * np.nan,
                    )

        # Report the plane actually measured (``plane_indices``, derived from
        # each port's ``pin_center`` — see ``_coaxial_port_geometry``), not
        # ``port.position``: the two differ by ``direction*pin_length/2``
        # whenever ``pin_length != 0``, and ``position_to_index`` already adds
        # ``pad_z_lo``, so multiplying that padded index by ``dx`` directly
        # (the previous formula) double-counted the padding offset too.
        # Neither defect is pinned by a committed test (only the array SHAPE
        # is asserted in test_coaxial_s_matrix.py) — see #489 stage-2 design
        # note, incidental defect 1.
        reference_planes = np.asarray(
            [
                (float(plane_indices[p_idx]) - float(grid.pad_z_lo)) * float(grid.dx)
                for p_idx in range(n_ports)
            ],
            dtype=float,
        )

        _res_coax = CoaxialSMatrixResult(
            s_params=s,
            freqs=np.asarray(freqs, dtype=float),
            port_names=tuple(f"coax_{i}" for i in range(n_ports)),
            port_faces=tuple(p.face for p in ports),
            reference_planes=reference_planes,
            z_tem_ohm=z_tem_arr,
            voltages=v_dump,
            currents=i_dump,
            status=status,
        )
        return _finalize_sparam_result(
            _res_coax,
            extractor="compute_coaxial_s_matrix",
            strict=strict_passivity,
        )

    def compute_coaxial_line_reflection(
        self,
        *,
        termination: str = "short",
        n_steps: int = 6000,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 11,
        field_scale: float = 1.0e4,
        cpml_axes: str = "z",
        dut_offset_cells: int = 4,
        probe_count: int = 12,
        probe_start_cells: int = 8,
        probe_spacing_cells: int = 4,
        feed_impedance: float | None = None,
        dut_impedance: float | None = None,
        eps_scale: "jnp.ndarray | float | None" = None,
    ) -> "CoaxialLineReflectionResult":
        """One-port coaxial reflection on a real transmission line (broad-E5).

        Builds a coextensive coax line (face='top', pin along -z) terminated in a
        matched resistive feed near the +z boundary, drives a TEM TFSF source one
        cell below the feed, and reflects off a calibration ``termination`` at the
        -z end: ``"short"`` (Γ=-1), ``"open"`` (Γ=+1), or ``"matched"`` (Γ→0).
        With ``termination="matched"`` and ``dut_impedance=R`` the DUT is instead a
        known resistive load (analytic ``Γ=(R-Z0)/(R+Z0)``) — used by the broad-E5
        envelope to test non-trivial reflection magnitudes against exact truth.
        The reflection is read from the modal voltage ``V(z)=∫E_r dr`` sampled at
        ``probe_count`` equally spaced planes and a matrix-pencil estimate of the
        complex propagation constant (β self-measured, Z0-free).

        Unlike ``compute_coaxial_s_matrix`` (single-plane V/I in a hardcoded
        closed PEC box — non-physical |S11|>1), this uses an absorbing CPML feed
        so a real line exists. **Resolution recipe**: keep ≥~4 cells across the
        (outer-inner) annulus (raise ``freq_max`` to shrink ``dx``); the result
        ``status`` reports ``"under_resolved"`` below ~3.5 cells.

        The conductors deliberately stop ~2 cells short of the +z PML — running
        PEC into CPML is numerically unstable.

        The enclosing :class:`Simulation` must use float32 precision and the
        three-dimensional, second-order uniform Yee solver with
        ``boundary="cpml"`` and ``cpml_layers > 0``. Unsupported precision,
        solver, grid, boundary, TFSF, and refinement settings raise before the
        grid is built. The line feed requires positive CPML on both z faces,
        ``cpml_axes="z"``, and no periodic-axis override.

        This method constructs its own coaxial line, TEM source, DFT planes, and
        termination. Do not add separate geometry, thin conductors, lumped RLC
        elements, probes or field monitors, NTFF boxes, or ``add_coaxial_*``
        termination helpers; those registrations are rejected rather than
        ignored. Use the documented :class:`Simulation`, port, and method
        arguments instead.

        The registered coaxial port supplies its x/y centre, ``face``, inner
        and outer radii, and excitation waveform. The method derives its axial
        layout internally, so the port's z coordinate and ``pin_length`` do not
        place the line, and the port's ``impedance`` does not set either load.
        Use ``feed_impedance`` for the feed and ``dut_impedance`` only with
        ``termination="matched"``. ``probe_count`` must be an integer of at
        least three, and every requested plane must fit between the DUT and
        source; otherwise the method raises before starting the FDTD run.

        Differentiable (``eps_scale``)
        ------------------------------
        Pass ``eps_scale`` (a scalar or ``(nx, ny, nz)`` ``jnp`` array) to make
        the reflection differentiable w.r.t. the dielectric under ``jax.grad``.
        It MULTIPLIES the stamped ``eps_r`` (``eps_r <- eps_r * eps_scale``),
        applied AFTER the numpy conductor/dielectric stamps so the fixed geometry
        (PTFE fill in ``eps_r``, PEC pin/shell in ``sigma``) is preserved and only
        modulated — a well-conditioned design channel (unlike replacing the fill
        with air). When provided, the field→voltage→reflection assembly runs on
        the ``jax.numpy`` path (``coaxial_line_plane_voltage_jnp`` + the traced
        extractor) so the gradient flows design → FDTD → DFT planes → Γ. With
        ``eps_scale=None`` the result is byte-identical to the validated numpy
        path. The AD↔FD gate is ``tests/test_coax_end_to_end_ad.py``.
        """

        if self._boundary != "cpml" or self._cpml_layers <= 0:
            raise ValueError(
                "compute_coaxial_line_reflection() requires boundary='cpml' "
                "with cpml_layers > 0 for its absorbing feed."
            )
        z_boundary = self._boundary_spec.z
        if (
            z_boundary.lo != "cpml"
            or z_boundary.hi != "cpml"
            or z_boundary.resolved_lo_thickness(self._cpml_layers) <= 0
            or z_boundary.resolved_hi_thickness(self._cpml_layers) <= 0
        ):
            raise ValueError(
                "compute_coaxial_line_reflection() requires positive CPML "
                "thickness on both z faces."
            )
        if cpml_axes != "z":
            raise ValueError(
                "compute_coaxial_line_reflection() requires cpml_axes='z'."
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_coaxial_line_reflection() does not support periodic "
                "boundary axes."
            )
        if any(token != "cpml" for _, _, token in self._boundary_spec.faces()):
            raise ValueError(
                "compute_coaxial_line_reflection() requires CPML tokens on all "
                "six boundary faces; mixed BoundarySpec faces are not supported."
            )
        if self._mode != "3d":
            raise ValueError(
                "compute_coaxial_line_reflection() requires mode='3d'."
            )
        if self._solver != "yee":
            raise ValueError(
                "compute_coaxial_line_reflection() supports solver='yee' only; "
                "solver='adi' is not supported."
            )
        if self._precision != "float32":
            raise ValueError(
                "compute_coaxial_line_reflection() requires precision='float32'."
            )
        if self._stencil_order != 2:
            raise ValueError(
                "compute_coaxial_line_reflection() requires stencil_order=2."
            )
        if self._tfsf is not None:
            raise ValueError(
                "compute_coaxial_line_reflection() creates its own TEM TFSF "
                "source and does not accept an existing TFSF source."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise ValueError(
                "compute_coaxial_line_reflection() supports only a uniform Yee "
                "grid; dx_profile, dy_profile, and dz_profile are not supported."
            )
        if self._refinement is not None:
            raise ValueError(
                "compute_coaxial_line_reflection() does not support SBP-SAT "
                "refinement; remove add_refinement() from this simulation."
            )
        if self._geometry or self._thin_conductors:
            raise ValueError(
                "compute_coaxial_line_reflection() constructs the complete line "
                "geometry; registered geometry and thin conductors are not "
                "supported. Use the documented Simulation, port, and method "
                "arguments instead."
            )
        if self._lumped_rlc:
            raise ValueError(
                "compute_coaxial_line_reflection() does not support registered "
                "lumped RLC elements."
            )
        if self._probes or self._dft_planes or self._flux_monitors or self._ntff:
            raise ValueError(
                "compute_coaxial_line_reflection() does not consume registered "
                "probes, DFT planes, flux monitors, or NTFF boxes."
            )
        if (
            self._coaxial_terminations
            or self._coaxial_open_terminations
            or self._coaxial_pec_end_caps
        ):
            raise ValueError(
                "compute_coaxial_line_reflection() does not consume registered "
                "add_coaxial_* termination helpers; use termination= and "
                "dut_impedance= instead."
            )

        from rfx.probes.probes import init_dft_plane_probe
        from rfx.simulation import run as _run
        from rfx.sources.coaxial_port import (
            CoaxialPort as _CoaxPort,
            build_coaxial_tem_plane_source_specs,
            coaxial_line_plane_voltage,
            coaxial_line_plane_voltage_jnp,
            coaxial_line_reflection_from_plane_voltages,
            coaxial_tem_characteristic_impedance,
            stamp_coaxial_line,
            stamp_coaxial_short_plane,
            stamp_coaxial_annular_resistor,
        )

        if termination not in ("short", "open", "matched"):
            raise ValueError(
                f"termination must be 'short', 'open' or 'matched', got {termination!r}"
            )
        if dut_impedance is not None and termination != "matched":
            raise ValueError(
                "dut_impedance is used only with termination='matched'; remove "
                "it for short or open terminations."
            )
        if isinstance(probe_count, bool) or not isinstance(
            probe_count, (int, np.integer)
        ):
            raise ValueError("probe_count must be an integer of at least 3.")
        requested_probe_count = int(probe_count)
        if requested_probe_count < 3:
            raise ValueError("probe_count must be at least 3.")
        if len(self._coaxial_ports) != 1:
            raise ValueError(
                "compute_coaxial_line_reflection() is a one-port method; register "
                "exactly one add_coaxial_port()."
            )
        if (
            self._ports or self._waveguide_ports or self._floquet_ports or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_line_reflection() is defined only for a single "
                "add_coaxial_port(...) family."
            )
        port = self._coaxial_ports[0]
        if port.face != "top":
            raise NotImplementedError(
                "compute_coaxial_line_reflection() currently supports face='top' "
                "(pin along -z, DUT at the -z end); face='bottom' is symmetric and "
                "not yet wired."
            )

        grid = self._build_grid()
        nz = grid.shape[2]
        dz = float(grid.dx)
        center_xy = (float(port.position[0]), float(port.position[1]))
        a, b = float(port.pin_radius), float(port.outer_radius)

        # Axial layout: DUT just above the -z PML; coax runs up to ~2 cells short
        # of the +z PML; matched feed one cell below the coax top; source below it.
        # The +z offset uses pad_z_hi (not pad_z_lo) so an asymmetric BoundarySpec
        # cannot run the conductors into the +z PML (verified unstable).
        z_dut = int(grid.pad_z_lo) + int(dut_offset_cells)
        z_hi_coax = nz - int(grid.pad_z_hi) - 2
        z_feed = z_hi_coax - 1
        z_src = z_hi_coax - 3
        if not (z_dut + probe_start_cells + 2 * probe_spacing_cells < z_src):
            raise ValueError(
                "domain too short for the requested line layout; increase the z "
                "domain or reduce probe_start_cells/probe_count."
            )
        probes_z = [
            z_dut + int(probe_start_cells) + int(probe_spacing_cells) * k
            for k in range(requested_probe_count)
        ]
        probes_z = [z for z in probes_z if z < z_src - 4]
        if len(probes_z) != requested_probe_count:
            raise ValueError(
                f"only {len(probes_z)} of {requested_probe_count} requested "
                "probe planes fit before the source; increase the z domain or "
                "reduce probe_count, probe_start_cells, or probe_spacing_cells."
            )

        z_tem = coaxial_tem_characteristic_impedance(a, b)
        R_feed = float(feed_impedance) if feed_impedance is not None else float(z_tem)
        # For termination='matched', the DUT load resistance defaults to the feed
        # (Γ→0); override with dut_impedance to place a known mismatch
        # (Γ = (R-Z0)/(R+Z0)) — used by the broad-E5 envelope's non-trivial loads.
        R_dut = float(dut_impedance) if dut_impedance is not None else R_feed

        materials, _, _ = self._build_materials(grid)
        materials, shell_inner = stamp_coaxial_line(
            grid, materials, center_xy=center_xy, z_lo_index=z_dut,
            z_hi_index=z_hi_coax, pin_radius=a, outer_radius=b,
        )
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=z_feed, pin_radius=a,
            outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
        )
        if termination == "short":
            materials = stamp_coaxial_short_plane(
                grid, materials, center_xy=center_xy, z_index=z_dut, outer_radius=b,
            )
        elif termination == "matched":
            materials = stamp_coaxial_annular_resistor(
                grid, materials, center_xy=center_xy, z_index=z_dut, pin_radius=a,
                outer_radius=b, target_impedance=R_dut, shell_inner_radius=shell_inner,
            )
        # "open": conductors simply end at z_dut (open circuit) — no extra stamp.

        # Differentiable design channel: applied AFTER the (numpy) stamps as a
        # MULTIPLIER so the stamped dielectric (PTFE fill) + PEC (in sigma) are
        # preserved and only modulated — well-conditioned (vs replacing the fill).
        if eps_scale is not None:
            materials = materials._replace(eps_r=materials.eps_r * jnp.asarray(eps_scale))

        if freqs is None:
            freqs = jnp.linspace(
                0.1 * self._freq_max, 0.6 * self._freq_max, int(n_freqs), dtype=jnp.float32
            )
        else:
            freqs = jnp.asarray(freqs, dtype=jnp.float32)

        # TEM TFSF source at z_src (internal descriptor places pin_center there).
        src_port = _CoaxPort(
            position=(center_xy[0], center_xy[1], (z_src - grid.pad_z_lo) * dz),
            face="top", pin_length=dz, pin_radius=a, outer_radius=b,
            impedance=port.impedance, excitation=port.excitation,
        )
        spec = build_coaxial_tem_plane_source_specs(
            grid=grid, port=src_port, n_steps=int(n_steps), field_scale=float(field_scale),
            magnetic_ratio=1.0,
        )

        planes = []
        for z in probes_z:
            for comp in ("ex", "ey"):
                planes.append(
                    init_dft_plane_probe(
                        axis=2, index=int(z), component=comp, freqs=freqs,
                        grid_shape=grid.shape, dft_total_steps=int(n_steps),
                    )
                )
        result = _run(
            grid, materials, int(n_steps), boundary="cpml", cpml_axes=cpml_axes,
            sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
            dft_planes=planes, return_state=False,
        )
        if result.dft_planes is None:
            raise RuntimeError("compute_coaxial_line_reflection(): runner returned no DFT planes")

        # Modal voltage V(z) at every probe plane, per frequency.
        n_f = int(freqs.shape[0])
        z_planes_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_z], dtype=np.float64)
        ref_m = (z_dut - grid.pad_z_lo) * dz
        annulus_cells = float((b - a) / dz)

        if eps_scale is not None:
            # --- differentiable path: jnp voltage + traced extractor (AD moat) ---
            # Gradient flows: eps_scale -> Yee update -> DFT plane accumulators
            # -> modal voltage -> matrix-pencil reflection -> Γ.
            V = jnp.stack(
                [
                    coaxial_line_plane_voltage_jnp(
                        grid, result.dft_planes[pi * 2 + 0].accumulator,
                        center_xy=center_xy, pin_radius=a, outer_radius=b,
                    )
                    for pi in range(len(probes_z))
                ],
                axis=0,
            )  # (n_planes, n_freqs)
            s11_c, gamma_c, rec_c, fit_c, z0_c = [], [], [], [], []
            for fi in range(n_f):
                out = coaxial_line_reflection_from_plane_voltages(
                    z_planes_m, V[:, fi], reference_plane_m=ref_m, _prefer_jnp=True,
                )
                s11_c.append(out.reflection)
                gamma_c.append(out.gamma)
                rec_c.append(out.recurrence_residual)
                fit_c.append(out.fit_residual)
                z0_c.append(
                    R_dut * (1.0 - out.reflection) / (1.0 + out.reflection)
                    if termination == "matched"
                    else jnp.asarray(np.nan + 1j * np.nan)
                )
            # Status from concrete geometry only. NOTE: unlike the concrete path,
            # the traced rec_resid>0.1 "contaminated" check is NOT evaluated here
            # (can't Python-branch on a tracer), so ``"passed"`` on the eps_scale
            # path means geometry-resolved, NOT fit-clean — inspect
            # ``recurrence_residual`` (returned) if you need the contamination
            # signal. ``"differentiable"`` flags the AD path so it is not confused
            # with a fully-gated concrete ``"passed"``.
            status = "under_resolved" if annulus_cells < 3.5 else "differentiable"
            return CoaxialLineReflectionResult(
                s11=jnp.stack(s11_c),
                freqs=jnp.asarray(freqs),
                gamma=jnp.stack(gamma_c),
                recurrence_residual=jnp.stack(rec_c),
                fit_residual=jnp.stack(fit_c),
                annulus_cells=annulus_cells,
                z0_numerical_ohm=jnp.stack(z0_c),
                termination=termination,
                status=status,
            )

        # --- concrete path: NumPy (byte-identical to the validated result) ---
        v_by_plane = []
        for pi in range(len(probes_z)):
            ex = result.dft_planes[pi * 2 + 0].accumulator
            ey = result.dft_planes[pi * 2 + 1].accumulator
            v_by_plane.append(
                coaxial_line_plane_voltage(
                    grid, ex, ey, center_xy=center_xy, pin_radius=a, outer_radius=b,
                )
            )
        V = np.stack(v_by_plane, axis=0)          # (n_planes, n_freqs)

        s11 = np.zeros(n_f, dtype=np.complex128)
        gamma = np.zeros(n_f, dtype=np.complex128)
        rec_resid = np.zeros(n_f, dtype=np.float64)
        fit_resid = np.zeros(n_f, dtype=np.float64)
        z0_num = np.full(n_f, np.nan + 1j * np.nan, dtype=np.complex128)
        for fi in range(n_f):
            out = coaxial_line_reflection_from_plane_voltages(
                z_planes_m, V[:, fi], reference_plane_m=ref_m,
            )
            s11[fi] = out.reflection
            gamma[fi] = out.gamma
            rec_resid[fi] = out.recurrence_residual
            fit_resid[fi] = out.fit_residual
            if termination == "matched":
                G = out.reflection
                z0_num[fi] = R_dut * (1.0 - G) / (1.0 + G)

        if annulus_cells < 3.5:
            status = "under_resolved"
        elif float(np.max(rec_resid)) > 0.1:
            status = "contaminated"
        else:
            status = "passed"

        return CoaxialLineReflectionResult(
            s11=s11,
            freqs=np.asarray(freqs, dtype=float),
            gamma=gamma,
            recurrence_residual=rec_resid,
            fit_residual=fit_resid,
            annulus_cells=annulus_cells,
            z0_numerical_ohm=z0_num,
            termination=termination,
            status=status,
        )

    def compute_coaxial_two_port(
        self,
        *,
        n_steps: int = 6000,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 11,
        field_scale: float = 1.0e4,
        cpml_axes: str = "z",
        probe_count: int = 12,
        probe_start_cells: int = 8,
        probe_spacing_cells: int = 4,
        feed_impedance: float | None = None,
        cond_warn: float = 1.0e3,
        strict_passivity: bool = False,
        eps_scale: "jnp.ndarray | float | None" = None,
        extra_flux_monitors: "list | None" = None,
    ) -> "CoaxialTwoPortResult":
        """Two-drive coaxial 2-port S-parameters (#489 stage 2) — VALIDATED WITH SCOPE.

        .. note::
            **VALIDATED WITH SCOPE, not EXPERIMENTAL** (issue #489, PI
            decision 2026-08-06 —
            ``docs/guides/sparameter_support_matrix.md``, the S-parameter
            family companion where this row lives). Covered: the two-port
            through-line class on this single coax geometry family, bracketed
            by an external openEMS referee (crossval 21) on ``|S21|`` and,
            via the port's own measured ``beta``, phase; a mesh-refinement
            convergence witness (``p ~= 1.5``); an end-to-end ``eps_scale``
            AD gate (``GRAD_SAFE``); and this method's own measured
            reciprocity/``cond(A)``. NOT covered: every DUT this method can
            currently gate against is azimuthally symmetric (TM0n only) —
            transition discontinuities that excite TE11 are still outside
            this evidence (see :meth:`compute_coax_msl_transition`, which
            stays EXPERIMENTAL, diagnostic-only), nor does the evidence
            generalize to other coax geometry families. See the class
            docstring on the returned :class:`CoaxialTwoPortResult` for the
            full evidence chain and scope statement.

        Builds ONE through coax line spanning the z axis with a matched
        annular-resistor feed near EACH z end (mirroring the validated
        1-port :meth:`compute_coaxial_line_reflection` layout at both ends:
        each feed sits between that end's own TEM TFSF source and that end's
        own CPML, i.e. strictly on the scattered-only side of that drive's
        TFSF boundary — never in the path of that drive's own launched
        wave), then drives each end's source in turn (two separate FDTD
        runs). A probe array of ``probe_count`` equally spaced planes near
        each end recovers that array's local two-wave decomposition
        (matrix-pencil, Z0-free, same machinery as the 1-port method); the
        forward/back amplitudes are evaluated at each port's OWN reference
        plane (its feed's axial position) and assembled into a full 2x2
        S-matrix via :func:`rfx.sources.coaxial_port.
        solve_two_port_from_wave_amplitudes` — the two-drive solve that does
        NOT assume the non-driven port sees zero incident wave (unlike the
        naive ``S[j,i] = b_j/a_i`` ratio, which has a hard terminator-
        reflection floor on a through line; see that function's docstring
        and ``docs/research_notes/20260729_i489_coax_two_port_design.md``).

        Port 1 is the +z end (mirrors the 1-port fixture's own ``face='top'``
        orientation); port 2 is the -z end (mirror image, ``face='bottom'``).
        Both drives share the SAME registered ``add_coaxial_port(...)``
        geometry (x/y centre, pin/outer radii) and excitation waveform; the
        registered port's own ``position``/``face``/``pin_length`` do not
        place either end of the line (mirrors the 1-port method's own
        contract). Requires ``port.face == 'top'`` (arbitrary but consistent
        with the 1-port method; the value is not otherwise used to orient
        this method's own internally-built fixture).

        Unlike the 1-port method, the returned result is routed through
        :func:`_finalize_sparam_result` (which runs the passivity/finiteness
        self-check via :func:`_warn_if_nonpassive_smatrix`) before being
        returned — the 1-port ``CoaxialLineReflectionResult`` bypasses that
        check (design-note incidental defect 3); this result does not.

        This method constructs its own coaxial line, TEM sources, DFT
        planes, and feeds. Do not add separate geometry, thin conductors,
        lumped RLC elements, probes or field monitors, NTFF boxes, TFSF
        sources, or ``add_coaxial_*`` termination helpers; those
        registrations are rejected rather than ignored.

        Same solver/precision/boundary contract as
        :meth:`compute_coaxial_line_reflection` (float32, 3D uniform Yee,
        ``boundary='cpml'`` with positive CPML on all six faces,
        ``cpml_axes='z'``, no periodic axes, no non-uniform mesh, no
        refinement).

        Differentiable (``eps_scale``, #489 leg 3)
        -------------------------------------------
        Pass ``eps_scale`` (a scalar or ``(nx, ny, nz)`` ``jnp`` array) to make
        the S-matrix differentiable w.r.t. the dielectric under ``jax.grad`` —
        same name, semantics, and MULTIPLIES-the-stamped-``eps_r`` design as
        the 1-port :meth:`compute_coaxial_line_reflection`'s own ``eps_scale``
        (``eps_r <- eps_r * eps_scale``, applied once, after both feed stamps,
        so it reaches BOTH drives' FDTD runs — the through line has no DUT
        break to scope it to). When provided, both drives route their
        voltage extraction through :func:`rfx.sources.coaxial_port.
        coaxial_line_plane_voltage_jnp` (the same differentiable twin the
        1-port path uses) instead of the concrete
        :func:`~rfx.sources.coaxial_port.coaxial_line_plane_voltage`, and the
        assembly (:func:`_assemble_coaxial_two_port_from_voltages`) and the
        two-drive solve (:func:`~rfx.sources.coaxial_port.
        solve_two_port_from_wave_amplitudes`) both dispatch to their own jnp
        cores. With ``eps_scale=None`` the result is byte-identical to the
        validated numpy path (dual-path design, not a rewrite — mirrors the
        1-port method's own contract). The per-drive ring-down ``settling_db``
        witness needs a concrete time series and is skipped on this path
        (stays ``nan``, same reasoning as the 1-port method's own
        "can't Python-branch on a tracer" note on its ``rec_resid``
        contamination check); ``status`` is therefore ``"under_resolved"`` or
        ``"differentiable"`` here, never ``"contaminated"``/``"passed"``.
        ``cond_warn`` is also silently INERT on this path: the ill-
        conditioning warning it controls is Python control flow keyed on a
        concrete ``cond(A)`` value
        (:func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`'s
        NumPy branch), which the traced jnp core
        (``_solve_two_port_from_wave_amplitudes_jnp``) cannot evaluate and
        does not attempt to — this mirrors the ``settling_db``/``status``
        losses above (same "can't Python-branch on a tracer" reason), not a
        separate defect; ``cond_a`` is still RETURNED (as a tracer), so a
        caller can inspect it after concretizing the result if degeneracy
        matters to them. The AD gate is ``tests/test_coax_two_port_ad.py``.

        ``extra_flux_monitors`` (issue #589 flux-adjudication instrument):
        an ENERGY-WITNESS channel, not an extractor change. Pass the entry
        objects ``Simulation.add_flux_monitor`` registers (build them on a
        scratch ``Simulation`` with the same domain and hand over its
        ``._flux_monitors``); each internal drive run then accumulates the
        requested Poynting-flux planes, and per-drive spectra come back
        name-keyed on ``result.flux_monitors``
        (``{port_name: {monitor_name: (n_monitor_freqs,) float64}}``, net
        flux, positive = +axis). The S-parameter math is untouched — the
        non-perturbation witness (S bit-identical with and without
        monitors) is gated in
        ``tests/test_coax_msl_transition.py::test_extra_flux_monitors_do_not_perturb_s``.
        The registered-monitor guard is unchanged: monitors registered ON
        this sim still raise, because this method builds its own probes.
        """

        if self._boundary != "cpml" or self._cpml_layers <= 0:
            raise ValueError(
                "compute_coaxial_two_port() requires boundary='cpml' "
                "with cpml_layers > 0 for its absorbing feeds."
            )
        z_boundary = self._boundary_spec.z
        if (
            z_boundary.lo != "cpml"
            or z_boundary.hi != "cpml"
            or z_boundary.resolved_lo_thickness(self._cpml_layers) <= 0
            or z_boundary.resolved_hi_thickness(self._cpml_layers) <= 0
        ):
            raise ValueError(
                "compute_coaxial_two_port() requires positive CPML "
                "thickness on both z faces."
            )
        if cpml_axes != "z":
            raise ValueError(
                "compute_coaxial_two_port() requires cpml_axes='z'."
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_coaxial_two_port() does not support periodic "
                "boundary axes."
            )
        if any(token != "cpml" for _, _, token in self._boundary_spec.faces()):
            raise ValueError(
                "compute_coaxial_two_port() requires CPML tokens on all "
                "six boundary faces; mixed BoundarySpec faces are not "
                "supported."
            )
        if self._mode != "3d":
            raise ValueError(
                "compute_coaxial_two_port() requires mode='3d'."
            )
        if self._solver != "yee":
            raise ValueError(
                "compute_coaxial_two_port() supports solver='yee' only; "
                "solver='adi' is not supported."
            )
        if self._precision != "float32":
            raise ValueError(
                "compute_coaxial_two_port() requires precision='float32'."
            )
        if self._stencil_order != 2:
            raise ValueError(
                "compute_coaxial_two_port() requires stencil_order=2."
            )
        if self._tfsf is not None:
            raise ValueError(
                "compute_coaxial_two_port() creates its own TEM TFSF "
                "sources and does not accept an existing TFSF source."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise ValueError(
                "compute_coaxial_two_port() supports only a uniform Yee "
                "grid; dx_profile, dy_profile, and dz_profile are not "
                "supported."
            )
        if self._refinement is not None:
            raise ValueError(
                "compute_coaxial_two_port() does not support SBP-SAT "
                "refinement; remove add_refinement() from this simulation."
            )
        if self._geometry or self._thin_conductors:
            raise ValueError(
                "compute_coaxial_two_port() constructs the complete line "
                "geometry; registered geometry and thin conductors are not "
                "supported. Use the documented Simulation, port, and "
                "method arguments instead."
            )
        if self._lumped_rlc:
            raise ValueError(
                "compute_coaxial_two_port() does not support registered "
                "lumped RLC elements."
            )
        if self._probes or self._dft_planes or self._flux_monitors or self._ntff:
            raise ValueError(
                "compute_coaxial_two_port() does not consume registered "
                "probes, DFT planes, flux monitors, or NTFF boxes."
            )
        if (
            self._coaxial_terminations
            or self._coaxial_open_terminations
            or self._coaxial_pec_end_caps
        ):
            raise ValueError(
                "compute_coaxial_two_port() does not consume registered "
                "add_coaxial_* termination helpers; use feed_impedance= "
                "instead."
            )
        if isinstance(probe_count, bool) or not isinstance(
            probe_count, (int, np.integer)
        ):
            raise ValueError("probe_count must be an integer of at least 3.")
        requested_probe_count = int(probe_count)
        if requested_probe_count < 3:
            raise ValueError("probe_count must be at least 3.")
        if len(self._coaxial_ports) != 1:
            raise ValueError(
                "compute_coaxial_two_port() is built from exactly one "
                "add_coaxial_port() (its x/y centre, radii, and excitation "
                "waveform are shared by both drives); register exactly one."
            )
        if (
            self._ports or self._waveguide_ports or self._floquet_ports or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_two_port() is defined only for a single "
                "add_coaxial_port(...) family."
            )
        port = self._coaxial_ports[0]
        if port.face != "top":
            raise NotImplementedError(
                "compute_coaxial_two_port() currently requires the "
                "registered port's face='top' (the value is not otherwise "
                "used to orient this method's own internally-built "
                "two-ended fixture; kept for contract consistency with "
                "compute_coaxial_line_reflection)."
            )

        from rfx.probes.probes import init_dft_plane_probe
        from rfx.probes.probes import flux_spectrum as _flux_spectrum
        from rfx.runners.uniform import build_flux_monitor_cfgs
        from rfx.simulation import run as _run, ProbeSpec
        from rfx.sources.coaxial_port import (
            CoaxialPort as _CoaxPort,
            build_coaxial_tem_plane_source_specs,
            coaxial_line_plane_voltage,
            coaxial_line_plane_voltage_jnp,
            coaxial_tem_characteristic_impedance,
            stamp_coaxial_line,
            stamp_coaxial_annular_resistor,
        )
        _validate_extra_flux_monitor_entries(
            extra_flux_monitors, self._domain, "compute_coaxial_two_port"
        )
        flux_by_drive: dict = {}

        grid = self._build_grid()
        nz = grid.shape[2]
        dz = float(grid.dx)
        center_xy = (float(port.position[0]), float(port.position[1]))
        a, b = float(port.pin_radius), float(port.outer_radius)

        # Axial layout: two mirrored 1-port-style ends (source, feed, probe
        # array) sharing ONE continuous stamped line — no DUT break. Offsets
        # (2/1/3 cells) mirror compute_coaxial_line_reflection's own
        # (z_hi_coax, z_feed, z_src) spacing exactly, just doubled and
        # mirror-imaged. See docs/design_notes/
        # i489_stage2_two_port_fdtd_predeclaration.md for the derivation of
        # why each feed sits strictly on the scattered-only side of its own
        # drive's TFSF boundary.
        z_hi_coax_top = nz - int(grid.pad_z_hi) - 2
        z_feed_top = z_hi_coax_top - 1
        z_src_top = z_hi_coax_top - 3
        z_lo_coax_bot = int(grid.pad_z_lo) + 2
        z_feed_bot = z_lo_coax_bot + 1
        z_src_bot = z_lo_coax_bot + 3

        probes_top = sorted(
            z_src_top - int(probe_start_cells) - int(probe_spacing_cells) * k
            for k in range(requested_probe_count)
        )
        probes_bot = sorted(
            z_src_bot + int(probe_start_cells) + int(probe_spacing_cells) * k
            for k in range(requested_probe_count)
        )
        if z_lo_coax_bot >= z_hi_coax_top or probes_bot[0] <= z_lo_coax_bot:
            raise ValueError(
                "compute_coaxial_two_port(): domain too short for the "
                "two-feed line layout; increase the z domain."
            )
        if probes_bot[-1] >= probes_top[0]:
            raise ValueError(
                "compute_coaxial_two_port(): the two probe arrays overlap "
                f"(bottom array reaches index {probes_bot[-1]}, top array "
                f"starts at {probes_top[0]}); increase the z domain or "
                "reduce probe_count/probe_start_cells/probe_spacing_cells."
            )

        z_tem = coaxial_tem_characteristic_impedance(a, b)
        R_feed = float(feed_impedance) if feed_impedance is not None else float(z_tem)

        materials, _, _ = self._build_materials(grid)
        materials, shell_inner = stamp_coaxial_line(
            grid, materials, center_xy=center_xy, z_lo_index=z_lo_coax_bot,
            z_hi_index=z_hi_coax_top, pin_radius=a, outer_radius=b,
        )
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=z_feed_top, pin_radius=a,
            outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
        )
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=z_feed_bot, pin_radius=a,
            outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
        )

        # Differentiable design channel (#489 leg 3): applied AFTER the
        # (numpy) stamps as a MULTIPLIER, same contract as the 1-port
        # compute_coaxial_line_reflection's own eps_scale — the stamped
        # dielectric (PTFE fill) + PEC (in sigma) are preserved and only
        # modulated. Applied ONCE, before either drive's _run() call below,
        # so it reaches both (the through line has no DUT break to scope it
        # to one end).
        if eps_scale is not None:
            materials = materials._replace(eps_r=materials.eps_r * jnp.asarray(eps_scale))

        if freqs is None:
            freqs = jnp.linspace(
                0.1 * self._freq_max, 0.6 * self._freq_max, int(n_freqs), dtype=jnp.float32
            )
        else:
            freqs = jnp.asarray(freqs, dtype=jnp.float32)
        n_f = int(freqs.shape[0])

        # TEM TFSF sources: port 1 (+z end) mirrors the 1-port fixture's own
        # face='top' source exactly; port 2 (-z end) is the mirror image,
        # face='bottom'.
        src_port_top = _CoaxPort(
            position=(center_xy[0], center_xy[1], (z_src_top - grid.pad_z_lo) * dz),
            face="top", pin_length=dz, pin_radius=a, outer_radius=b,
            impedance=port.impedance, excitation=port.excitation,
        )
        src_port_bot = _CoaxPort(
            position=(center_xy[0], center_xy[1], (z_src_bot - grid.pad_z_lo) * dz),
            face="bottom", pin_length=dz, pin_radius=a, outer_radius=b,
            impedance=port.impedance, excitation=port.excitation,
        )
        spec_top = build_coaxial_tem_plane_source_specs(
            grid=grid, port=src_port_top, n_steps=int(n_steps),
            field_scale=float(field_scale), magnetic_ratio=1.0,
        )
        spec_bot = build_coaxial_tem_plane_source_specs(
            grid=grid, port=src_port_bot, n_steps=int(n_steps),
            field_scale=float(field_scale), magnetic_ratio=1.0,
        )

        n_bot = len(probes_bot)
        n_top = len(probes_top)
        all_probes_z = list(probes_bot) + list(probes_top)

        # Settling witness: one point probe per array (ex, mid-annulus on the
        # +x ray), at each array's middle plane. Same worst end/peak E^2 (dB)
        # convention as the MSL/mixed lanes (rfx.api._sparams module docstring
        # of _warn_if_ringdown_truncated); -40 dB is the project's ring-down
        # settling rule (docs/guides/simulation_methodology.md).
        x_mid = center_xy[0] + 0.5 * (a + b)
        i_probe = int(round(x_mid / dz)) + int(grid.pad_x_lo)
        j_probe = int(grid.pad_y_lo) + int(round(center_xy[1] / dz))
        witness_probes = [
            ProbeSpec(i=i_probe, j=j_probe, k=int(probes_bot[n_bot // 2]), component="ex"),
            ProbeSpec(i=i_probe, j=j_probe, k=int(probes_top[n_top // 2]), component="ex"),
        ]

        z_planes_bot_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_bot], dtype=np.float64)
        z_planes_top_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_top], dtype=np.float64)
        ref_top_m = (z_feed_top - grid.pad_z_lo) * dz
        ref_bot_m = (z_feed_bot - grid.pad_z_lo) * dz
        annulus_cells = float((b - a) / dz)

        _traced_eps = eps_scale is not None
        if _traced_eps:
            # jnp lists, one entry per drive; stacked into (2, n_planes, n_f)
            # arrays after the loop (traced values can't be assigned into a
            # preallocated numpy array in place).
            v_bot_list: list = [None, None]
            v_top_list: list = [None, None]
        else:
            v_bot_by_drive = np.zeros((2, n_bot, n_f), dtype=np.complex128)
            v_top_by_drive = np.zeros((2, n_top, n_f), dtype=np.complex128)
        settling_db = np.full(2, np.nan, dtype=np.float64)

        # drive_idx 0 drives port 1 (top); drive_idx 1 drives port 2 (bot).
        for drive_idx, spec in enumerate((spec_top, spec_bot)):
            planes = []
            for z in all_probes_z:
                for comp in ("ex", "ey"):
                    planes.append(
                        init_dft_plane_probe(
                            axis=2, index=int(z), component=comp, freqs=freqs,
                            grid_shape=grid.shape, dft_total_steps=int(n_steps),
                        )
                    )
            # #589 flux-adjudication opt-in: fresh accumulators PER DRIVE
            # (init_flux_monitor zeroes the DFT carries; sharing cfgs across
            # drives would co-accumulate both drives into one spectrum).
            _flux_run_kwargs = (
                {"flux_monitors": build_flux_monitor_cfgs(
                    self, grid, int(n_steps), entries=extra_flux_monitors)}
                if extra_flux_monitors else {}
            )
            result = _run(
                grid, materials, int(n_steps), boundary="cpml", cpml_axes=cpml_axes,
                sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
                probes=witness_probes, dft_planes=planes, return_state=False,
                **_flux_run_kwargs,
            )
            if result.dft_planes is None:
                raise RuntimeError(
                    "compute_coaxial_two_port(): runner returned no DFT planes"
                )
            if extra_flux_monitors:
                flux_by_drive[("port1", "port2")[drive_idx]] = {
                    entry.name: np.asarray(_flux_spectrum(fm, exact_f64=True), dtype=np.float64)
                    for entry, fm in zip(
                        extra_flux_monitors, result.flux_monitors or ()
                    )
                }

            top_off = n_bot * 2
            if _traced_eps:
                # --- differentiable path: jnp voltage (AD moat, #489 leg 3) ---
                # Only the ex plane is needed (mirrors compute_coaxial_line_
                # reflection's own eps_scale branch): coaxial_line_plane_
                # voltage_jnp integrates E_r along the +x ray, ey is unused
                # for the voltage line-integral either way.
                v_bot_list[drive_idx] = jnp.stack(
                    [
                        coaxial_line_plane_voltage_jnp(
                            grid, result.dft_planes[pi * 2 + 0].accumulator,
                            center_xy=center_xy, pin_radius=a, outer_radius=b,
                        )
                        for pi in range(n_bot)
                    ],
                    axis=0,
                )  # (n_bot, n_freqs)
                v_top_list[drive_idx] = jnp.stack(
                    [
                        coaxial_line_plane_voltage_jnp(
                            grid, result.dft_planes[top_off + pi * 2 + 0].accumulator,
                            center_xy=center_xy, pin_radius=a, outer_radius=b,
                        )
                        for pi in range(n_top)
                    ],
                    axis=0,
                )  # (n_top, n_freqs)
                # Settling witness needs a concrete time series (np.asarray on
                # a traced result.time_series would raise
                # TracerArrayConversionError) -- skipped on this path, same
                # "can't Python-branch on a tracer" reasoning as the 1-port
                # eps_scale branch's own skipped contamination check.
                # settling_db stays nan for both drives.
            else:
                v_bot_by_drive[drive_idx] = np.stack(
                    [
                        coaxial_line_plane_voltage(
                            grid, result.dft_planes[pi * 2 + 0].accumulator,
                            result.dft_planes[pi * 2 + 1].accumulator,
                            center_xy=center_xy, pin_radius=a, outer_radius=b,
                        )
                        for pi in range(n_bot)
                    ],
                    axis=0,
                )  # (n_bot, n_freqs)
                v_top_by_drive[drive_idx] = np.stack(
                    [
                        coaxial_line_plane_voltage(
                            grid, result.dft_planes[top_off + pi * 2 + 0].accumulator,
                            result.dft_planes[top_off + pi * 2 + 1].accumulator,
                            center_xy=center_xy, pin_radius=a, outer_radius=b,
                        )
                        for pi in range(n_top)
                    ],
                    axis=0,
                )  # (n_top, n_freqs)

                ts = np.asarray(result.time_series, dtype=float)
                if ts.ndim == 2 and ts.shape[0] >= 10 and ts.shape[1] == len(witness_probes):
                    power = ts ** 2
                    tail = max(1, power.shape[0] // 10)
                    end = power[-tail:, :].mean(axis=0)
                    peak = power.max(axis=0)
                    tiny = np.finfo(float).tiny
                    ratio_db = 10.0 * np.log10((end + tiny) / (peak + tiny))
                    settling_db[drive_idx] = float(np.max(ratio_db))

        if _traced_eps:
            v_bot_by_drive = jnp.stack(v_bot_list, axis=0)
            v_top_by_drive = jnp.stack(v_top_list, axis=0)

        s_params, cond_a, rec_resid, fit_resid, gamma = _assemble_coaxial_two_port_from_voltages(
            z_planes_bot_m=z_planes_bot_m, z_planes_top_m=z_planes_top_m,
            ref_bot_m=ref_bot_m, ref_top_m=ref_top_m,
            v_bot_by_drive=v_bot_by_drive, v_top_by_drive=v_top_by_drive,
            cond_warn=float(cond_warn), _prefer_jnp=_traced_eps,
        )

        if _traced_eps:
            # Status from concrete geometry only -- rec_resid is a jnp
            # tracer here, so the Python-branching "contaminated" check below
            # cannot run (mirrors the 1-port eps_scale branch's own status
            # derivation). "differentiable" flags the AD path so it is not
            # confused with a fully-gated concrete "passed".
            status = "under_resolved" if annulus_cells < 3.5 else "differentiable"
        elif annulus_cells < 3.5:
            status = "under_resolved"
        elif float(np.max(rec_resid)) > 0.1:
            status = "contaminated"
        else:
            status = "passed"

        reference_planes = np.asarray([ref_top_m, ref_bot_m], dtype=float)
        result_obj = CoaxialTwoPortResult(
            s_params=s_params,
            freqs=np.asarray(freqs, dtype=float),
            port_names=("port1", "port2"),
            reference_planes=reference_planes,
            cond_a=cond_a,
            recurrence_residual=rec_resid,
            fit_residual=fit_resid,
            gamma=gamma,
            annulus_cells=annulus_cells,
            settling_db=settling_db,
            status=status,
            flux_monitors=(flux_by_drive if extra_flux_monitors else None),
        )
        # Issue #662: the witness above is computed but was never compared to
        # the -40 dB bar this result's own docstring documents. NaN on the
        # eps_scale path is skipped by the warner's finite mask (this array is
        # host-side numpy on every path, so no tracer is branched on here).
        _warn_if_ringdown_truncated(
            settling_db, ("port1", "port2"), n_steps=int(n_steps),
        )
        return _finalize_sparam_result(
            result_obj,
            extractor="compute_coaxial_two_port",
            strict=strict_passivity,
        )

    def compute_coax_msl_transition(
        self,
        *,
        junction_x: float,
        eps_r_sub: float | None = None,
        n_steps: int | None = None,
        num_periods: float = 30.0,
        freqs: "jnp.ndarray | None" = None,
        n_freqs: int = 11,
        field_scale: float = 1.0e4,
        probe_count: int = 8,
        probe_start_cells: int = 6,
        probe_spacing_cells: int = 3,
        msl_probe_count: int | None = None,
        msl_probe_start_cells: int | None = None,
        msl_probe_spacing_cells: int | None = None,
        feed_impedance: float | None = None,
        cond_warn: float = 1.0e3,
        strict_passivity: bool = False,
        skip_preflight: bool = False,
        extra_flux_monitors: "list | None" = None,
    ) -> "CoaxMSLTransitionResult":
        """EXPERIMENTAL coax<->microstrip transition 2-port S-parameters (issue #489 leg 4).

        .. warning::
            **EXPERIMENTAL — not in the validated set**
            (``docs/guides/sparameter_support_matrix.md`` / ``.json``). One
            pre-declared fixture has been run against this method; see that
            fixture's own predeclaration
            (``tests/test_coax_msl_transition.py``) for the measured
            envelope. NOT_TRACEABLE (see :class:`~rfx.api._spec.
            CoaxMSLTransitionResult`'s class docstring for the full honesty
            contract, including why the MSL side is extracted via the coax
            matrix-pencil fit rather than the diagnostic-only N-probe fit
            #488 uses).

        Generalizes issue #488's mixed lumped/wire<->MSL assembler
        (:meth:`compute_mixed_s_matrix`) to a coax<->MSL pair by combining,
        UNCHANGED, the less-invasive half of each family's own validated
        machinery instead of writing a new geometry-specific extractor:

        * The **coax side** is built exactly like :meth:`compute_coaxial_two_port`'s
          own single-ended stub (CPML, TEM TFSF source, matched annular-
          resistor feed, then a probe array), reusing
          :func:`~rfx.sources.coaxial_port.stamp_coaxial_line`,
          :func:`~rfx.sources.coaxial_port.stamp_coaxial_annular_resistor`,
          and :func:`~rfx.sources.coaxial_port.build_coaxial_tem_plane_source_specs`
          verbatim — but only ONE end (this method has no second coax port).
        * The **MSL side** is consumed exactly like :meth:`compute_mixed_s_matrix`
          consumes its MSL ports: the caller registers arbitrary DUT
          geometry (substrate, trace, ground plane, and — unique to this
          transition — the ground-plane clearance hole and the vertical
          pin-to-trace post that connects the two families) via the
          ordinary ``sim.add(Box(...)/Cylinder(...), material=...)`` API,
          and this method reuses :func:`~rfx.sources.msl_port.compute_msl_mode_profile`,
          :func:`~rfx.sources.msl_port.setup_msl_port`, and
          :func:`~rfx.sources.msl_port.make_msl_port_sources` verbatim for
          the MSL port's own termination/excitation.

        This method does NOT build the junction geometry itself (the ground
        plane, its clearance hole, or the pin-to-trace post) — those are
        DUT-specific and belong to the caller's own registered geometry,
        exactly as :meth:`compute_mixed_s_matrix` never builds its own
        substrate/trace. What this method DOES fix, by construction, is
        WHERE each port's own S-parameter reference plane sits: both are
        placed AT the physical launch discontinuity (see ``junction_x``
        below and ``port.position[2]`` on the registered
        :meth:`add_coaxial_port`), specifically to minimize the
        pre-declared "reference-plane mismatch" failure mode (the coax's
        axial z-feed-plane convention has no direct analogue in the MSL's
        along-trace x reference plane — these are different geometric axes
        entirely).

        Registration contract
        ----------------------
        Exactly one :meth:`add_coaxial_port` (``face='bottom'`` — the
        physical convention this method assumes: the coax stub is built
        FROM the domain's low-z CPML face UP TO ``position[2]`` (rounded to
        the nearest grid z-node, :meth:`~rfx.grid.Grid.position_to_index`'s
        own convention), where ``position[0], position[1]`` is the coax
        axis centre (x, y) and ``position[2]`` is the physical height of
        the caller's OWN registered ground-plane conductor — i.e. this is
        the ONE parameter that ties this method's auto-built coax stub to
        the caller's own junction geometry; get it wrong and the pin will
        either dangle in a gap or overlap the caller's own PEC). Exactly
        one :meth:`add_msl_port`, whose ``position[2]`` (substrate bottom /
        ground height) MUST equal the coax port's ``position[2]`` to within
        one grid cell — both refer to the SAME physical ground plane. No
        other ports (lumped/wire/waveguide/Floquet), no TFSF, no lumped
        RLC, no pre-registered probes/DFT planes/flux monitors/NTFF (this
        method builds its own). ``self._geometry`` must be non-empty (the
        caller's substrate/trace/ground-plane/pin-post Boxes and
        Cylinders) — arbitrary, not validated here, same delegation
        :meth:`compute_mixed_s_matrix` uses for its own MSL DUT geometry.
        Same solver/precision/boundary contract as
        :meth:`compute_coaxial_two_port` (float32, 3D uniform Yee,
        ``boundary='cpml'`` with positive CPML on all six faces) EXCEPT
        this method needs absorption on all three axes (``cpml_axes="xyz"``,
        not ``"z"``): the coax's own far end needs an absorbing z face
        exactly like the coax lane, but the MSL trace radiates in x/y too,
        and the caller's own ground plane is an INTERNAL stamped/registered
        PEC layer rather than the domain's z boundary — this is also WHY
        this method cannot reuse a PEC ``z_lo`` domain boundary as the MSL
        ground reference the way :meth:`compute_mixed_s_matrix`'s
        ``magnitude_channel="flux"`` fixtures do (that would conflict with
        the coax's own need for an absorbing z_lo face).

        Two-drive extraction (never a naive single-ratio)
        ---------------------------------------------------
        Drives the coax source, then the MSL source, in two separate FDTD
        runs (never assuming the non-driven port sees zero incident wave —
        the terminator-floor problem :meth:`compute_coaxial_two_port`
        already solved for symmetric coax applies here too, and likely
        worse, since the coax feed resistor and the MSL Hammerstad-Jensen
        termination have no reason to share a termination quality). Each
        port's own forward/backward wave amplitudes are recovered from its
        own probe ladder via :func:`~rfx.sources.coaxial_port.
        coaxial_line_reflection_from_plane_voltages` (Z0-free matrix-pencil
        fit — see :class:`~rfx.api._spec.CoaxMSLTransitionResult` for why
        this is used for the MSL side too, not the lane's own N-probe fit),
        converted to POWER waves via each port's own analytic reference
        impedance, and assembled by
        :func:`~rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`
        (the same generic two-drive solve :meth:`compute_coaxial_two_port`
        uses). See :func:`_assemble_coax_msl_transition_from_voltages` for
        the full pure-assembly derivation.

        Parameters
        ----------
        junction_x : float
            Physical x-coordinate (metres) of the coax-to-trace launch
            discontinuity — the MSL side's own S-parameter reference plane.
            Must match wherever the caller's own registered pin-to-trace
            post / trace Box actually begins; this method does not infer it
            from geometry (mirrors the coax side's own reference plane
            being the registered port's ``position[2]``, not inferred).
        eps_r_sub : float, optional
            Substrate relative permittivity. If ``None``, taken from the
            registered :meth:`add_msl_port`'s own ``eps_r_sub`` (which must
            then be set explicitly — this method does not attempt the
            geometry-bounding-box auto-detection :meth:`add_msl_port`
            itself offers, to keep the eps anchor unambiguous for the
            Hammerstad-Jensen Z0 used in the power-wave normalization).
        probe_count, probe_start_cells, probe_spacing_cells : int
            The COAX side's own probe array (mirrors
            :meth:`compute_coaxial_two_port`'s identically-named
            parameters — this method's coax stub is short by
            construction, between the near-source feed and the junction,
            so these three rarely need to grow).
        msl_probe_count, msl_probe_start_cells, msl_probe_spacing_cells : int, optional
            The MSL side's OWN probe array, independent of the coax
            parameters above (added issue #489 leg 4 attempt 2 — the two
            families' probe ladders were coupled through one shared set
            of parameters through attempt 1, which is fine when both
            ladders are short but breaks as soon as the MSL side needs a
            ladder spanning a meaningful fraction of a guided wavelength,
            since the coax stub's own short z-extent cannot host the same
            span). Each defaults to ``None``, meaning "use the
            correspondingly-named coax parameter" — this preserves
            attempt 1's exact behavior (and its committed fixture's
            numbers) when left unset.

        ``extra_flux_monitors`` (issue #589 flux-adjudication instrument):
        an ENERGY-WITNESS channel, not an extractor change. Pass the entry
        objects ``Simulation.add_flux_monitor`` registers (build them on a
        scratch ``Simulation`` with the same domain and hand over its
        ``._flux_monitors``); each internal drive run then accumulates the
        requested Poynting-flux planes, and per-drive spectra come back
        name-keyed on ``result.flux_monitors``
        (``{"coax"|"msl": {monitor_name: (n_monitor_freqs,) float64}}``,
        net flux, positive = +axis). The S-parameter math is untouched —
        the non-perturbation witness (S bit-identical with and without
        monitors) is gated in
        ``tests/test_coax_msl_transition.py::test_extra_flux_monitors_do_not_perturb_s``.
        The registered-monitor guard is unchanged: monitors registered ON
        this sim still raise, because this method builds its own probes.

        Returns
        -------
        CoaxMSLTransitionResult
        """
        from rfx.sources.coaxial_port import (
            CoaxialPort as _CoaxPort,
            build_coaxial_tem_plane_source_specs,
            coaxial_line_plane_voltage,
            coaxial_tem_characteristic_impedance,
            stamp_coaxial_line,
            stamp_coaxial_annular_resistor,
        )
        from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
        from rfx.sources.msl_port import (
            MSLPort as _MSLPortLL,
            _msl_yz_cells,
            compute_msl_mode_profile,
            setup_msl_port,
            make_msl_port_sources,
            msl_probe_x_coords_n,
        )
        from rfx.probes.probes import init_dft_plane_probe
        from rfx.probes.probes import flux_spectrum as _flux_spectrum
        from rfx.runners.uniform import build_flux_monitor_cfgs
        from rfx.simulation import run as _run, ProbeSpec

        _validate_extra_flux_monitor_entries(
            extra_flux_monitors, self._domain, "compute_coax_msl_transition"
        )
        flux_by_drive: dict = {}

        # ---- Registration guards ----------------------------------------
        if self._boundary != "cpml" or self._cpml_layers <= 0:
            raise ValueError(
                "compute_coax_msl_transition() requires boundary='cpml' "
                "with cpml_layers > 0."
            )
        if any(token != "cpml" for _, _, token in self._boundary_spec.faces()):
            raise ValueError(
                "compute_coax_msl_transition() requires CPML tokens on all "
                "six boundary faces; mixed BoundarySpec faces are not "
                "supported (the coax stub needs an absorbing z_lo face and "
                "the MSL trace needs absorbing x/y faces; the ground plane "
                "is an internal registered/stamped PEC layer, not a domain "
                "boundary)."
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_coax_msl_transition() does not support periodic "
                "boundary axes."
            )
        if self._mode != "3d":
            raise ValueError("compute_coax_msl_transition() requires mode='3d'.")
        if self._solver != "yee":
            raise ValueError(
                "compute_coax_msl_transition() supports solver='yee' only."
            )
        if self._precision != "float32":
            raise ValueError(
                "compute_coax_msl_transition() requires precision='float32'."
            )
        if self._stencil_order != 2:
            raise ValueError(
                "compute_coax_msl_transition() requires stencil_order=2."
            )
        if self._tfsf is not None:
            raise ValueError(
                "compute_coax_msl_transition() creates its own coax TEM "
                "source and does not accept an existing TFSF source."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise ValueError(
                "compute_coax_msl_transition() supports only a uniform "
                "Yee grid; dx_profile/dy_profile/dz_profile are not "
                "supported."
            )
        if self._refinement is not None:
            raise ValueError(
                "compute_coax_msl_transition() does not support SBP-SAT "
                "refinement."
            )
        if self._lumped_rlc:
            raise ValueError(
                "compute_coax_msl_transition() does not support registered "
                "lumped RLC elements."
            )
        if self._probes or self._dft_planes or self._flux_monitors or self._ntff:
            raise ValueError(
                "compute_coax_msl_transition() does not consume registered "
                "probes, DFT planes, flux monitors, or NTFF boxes (it "
                "builds its own)."
            )
        if (
            self._coaxial_terminations
            or self._coaxial_open_terminations
            or self._coaxial_pec_end_caps
        ):
            raise ValueError(
                "compute_coax_msl_transition() does not consume registered "
                "add_coaxial_* termination helpers; use feed_impedance= "
                "instead."
            )
        if len(self._coaxial_ports) != 1:
            raise ValueError(
                "compute_coax_msl_transition() is built from exactly one "
                "add_coaxial_port()."
            )
        if len(self._msl_ports) != 1:
            raise ValueError(
                "compute_coax_msl_transition() is built from exactly one "
                "add_msl_port()."
            )
        if self._ports or self._waveguide_ports or self._floquet_ports:
            raise NotImplementedError(
                "compute_coax_msl_transition() is defined only for a "
                "coax + MSL port pair; no other port families."
            )
        if not self._geometry:
            raise ValueError(
                "compute_coax_msl_transition() consumes the caller's own "
                "registered DUT geometry (substrate, trace, ground plane, "
                "clearance hole, pin-to-trace post) — none is registered. "
                "This method builds only the coax stub; register the "
                "junction geometry via sim.add(...) first."
            )
        port = self._coaxial_ports[0]
        if port.face != "bottom":
            raise NotImplementedError(
                "compute_coax_msl_transition() currently requires the "
                "registered coax port's face='bottom' (the coax stub is "
                "built from the domain's low-z CPML face up to "
                "position[2])."
            )
        msl_pe = self._msl_ports[0]
        # The coax stub's own top face stops AT port.position[2] (rounded to
        # the nearest grid node); the caller's own registered ground-plane
        # conductor is expected to occupy that node and MSL's substrate
        # (msl_pe.position[2], its own z_lo / substrate-bottom convention)
        # to begin at or above it -- the exact gap is the caller's own
        # ground-plane thickness (fixture-specific, not knowable here), so
        # this only guards the ORDER and catches a grossly misaligned
        # ground reference (e.g. forgetting to raise msl_z_lo at all).
        if float(msl_pe.position[2]) < float(port.position[2]) - 1.5e-9:
            raise ValueError(
                "compute_coax_msl_transition(): the registered MSL port's "
                f"substrate-bottom height ({msl_pe.position[2]:.6g} m) sits "
                f"BELOW the coax port's own junction height "
                f"({port.position[2]:.6g} m) — both must reference the SAME "
                "physical ground plane, with the MSL substrate at or above "
                "it."
            )
        eps_r_sub_resolved = (
            float(eps_r_sub) if eps_r_sub is not None
            else (float(msl_pe.eps_r_sub) if msl_pe.eps_r_sub is not None else None)
        )
        if eps_r_sub_resolved is None:
            raise ValueError(
                "compute_coax_msl_transition() needs eps_r_sub, either "
                "passed directly or set on the registered add_msl_port() "
                "(this method does not auto-detect it from geometry)."
            )

        grid = self._build_grid()
        dz = float(grid.dx)
        _junction_gap_cells = (float(msl_pe.position[2]) - float(port.position[2])) / dz
        if _junction_gap_cells > 8.0:
            raise ValueError(
                "compute_coax_msl_transition(): the registered MSL port's "
                f"substrate-bottom height is {_junction_gap_cells:.1f} cells "
                f"above the coax port's junction height ({port.position[2]:.6g} "
                "m) — that is implausibly large for a single ground-plane "
                "layer; check both registrations reference the SAME "
                "physical ground plane."
            )
        materials, debye_spec, lorentz_spec, pec_mask, _, _, _ = \
            self._assemble_materials(grid)

        if freqs is None:
            freqs_arr = np.asarray(
                jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs)
            )
        else:
            freqs_arr = np.asarray(freqs)
        n_f = int(freqs_arr.shape[0])
        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)
        freqs_jnp = jnp.asarray(freqs_arr, dtype=jnp.float32)

        # ---- Coax stub (mirrors compute_coaxial_two_port's single end) --
        center_xy = (float(port.position[0]), float(port.position[1]))
        a, b = float(port.pin_radius), float(port.outer_radius)
        z_junction_idx = int(grid.position_to_index(port.position)[2])
        z_stub_lo = int(grid.pad_z_lo) + 2
        z_feed = z_stub_lo + 1
        z_src = z_stub_lo + 3
        z_stub_hi = z_junction_idx - 1
        if z_stub_hi <= z_src:
            raise ValueError(
                "compute_coax_msl_transition(): domain too short between "
                "the low-z CPML and the junction height for the coax "
                "stub's source/feed/probe layout; increase the z domain or "
                "lower position[2]."
            )
        probes_coax = sorted(
            z_src + int(probe_start_cells) + int(probe_spacing_cells) * k
            for k in range(int(probe_count))
        )
        if probes_coax[-1] >= z_stub_hi:
            raise ValueError(
                "compute_coax_msl_transition(): the coax probe array "
                f"reaches index {probes_coax[-1]}, at or past the junction "
                f"({z_stub_hi}); increase the z domain or reduce "
                "probe_count/probe_start_cells/probe_spacing_cells."
            )

        z_tem = coaxial_tem_characteristic_impedance(a, b)
        r_feed = float(feed_impedance) if feed_impedance is not None else float(z_tem)
        materials, shell_inner = stamp_coaxial_line(
            grid, materials, center_xy=center_xy, z_lo_index=z_stub_lo,
            z_hi_index=z_stub_hi, pin_radius=a, outer_radius=b,
        )
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=z_feed,
            pin_radius=a, outer_radius=b, target_impedance=r_feed,
            shell_inner_radius=shell_inner,
        )

        src_port = _CoaxPort(
            position=(center_xy[0], center_xy[1], (z_src - grid.pad_z_lo) * dz),
            face="bottom", pin_length=dz, pin_radius=a, outer_radius=b,
            impedance=port.impedance, excitation=port.excitation,
        )
        spec_coax = build_coaxial_tem_plane_source_specs(
            grid=grid, port=src_port, n_steps=int(n_steps),
            field_scale=float(field_scale), magnetic_ratio=1.0,
        )
        ref_coax_m = (z_junction_idx - grid.pad_z_lo) * dz
        z_planes_coax_m = np.array(
            [(z - grid.pad_z_lo) * dz for z in probes_coax], dtype=np.float64
        )
        annulus_cells = float((b - a) / dz)
        if annulus_cells < 3.5:
            import warnings as _wa
            _wa.warn(
                f"compute_coax_msl_transition(): coax annulus resolution "
                f"{annulus_cells:.2f} cells is below the documented "
                "under-resolved threshold (3.5 cells, same convention as "
                "compute_coaxial_line_reflection/compute_coaxial_two_port) "
                "— reflection accuracy degrades at high frequency.",
                stacklevel=2,
            )

        # ---- MSL side (mirrors compute_mixed_s_matrix's MSL consumption) ---
        x_feed, y_centre, msl_z_lo = (float(c) for c in msl_pe.position)
        msl_port_base = _MSLPortLL(
            feed_x=x_feed,
            y_lo=y_centre - msl_pe.width / 2, y_hi=y_centre + msl_pe.width / 2,
            z_lo=msl_z_lo, z_hi=msl_z_lo + msl_pe.height,
            direction=msl_pe.direction, impedance=msl_pe.impedance,
            excitation=None,
        )
        mode_profile = compute_msl_mode_profile(grid, msl_port_base, eps_r_sub_resolved)
        materials = setup_msl_port(grid, msl_port_base, materials, mode_profile=mode_profile)
        z0_msl, eps_eff_msl = hammerstad_jensen_z0_eps_eff(
            msl_pe.width, msl_pe.height, eps_r_sub_resolved
        )

        # Registered impedance= divergence advisory (issue #581 review N2):
        # add_coaxial_port(impedance=...) / add_msl_port(impedance=...) size
        # the feed resistor / termination sigma and (for coax) the TEM
        # source amplitude calibration — but the POWER-WAVE NORMALIZATION
        # (z0_ref, feeding sqrt(Z0) in the assembler) always uses the
        # ANALYTIC z_tem / z0_msl computed here, never the registered
        # impedance. A large silent divergence between the two is a
        # footgun: the source/termination is calibrated for one Z0 while
        # the extraction is normalized against another.
        for _label, _registered, _analytic in (
            ("coax", float(port.impedance), float(z_tem)),
            ("msl", float(msl_pe.impedance), float(z0_msl)),
        ):
            if _analytic > 0.0:
                _rel_dev = abs(_registered - _analytic) / _analytic
                if _rel_dev > 0.05:
                    import warnings as _wz
                    _wz.warn(
                        f"compute_coax_msl_transition(): the registered "
                        f"{_label} port impedance ({_registered:.2f} ohm) "
                        f"diverges {_rel_dev * 100:.1f}% from the analytic "
                        f"{_label} Z0 ({_analytic:.2f} ohm) this method "
                        "actually uses for the power-wave normalization "
                        "(z0_ref) and for sizing the feed resistor / "
                        "termination. The registered impedance= is NOT "
                        "the reference impedance of the returned "
                        "s_params; it only affects source/termination "
                        "sizing. Pass a matching pin_radius/outer_radius "
                        "(coax) or width/height/eps_r_sub (msl), or "
                        "reconcile the mismatch, before trusting a "
                        "specific reference-impedance interpretation.",
                        stacklevel=2,
                    )

        # MSL-side probe ladder is independent of the coax-side one (issue
        # #489 leg 4 attempt 2) -- default to the coax values so an
        # existing caller (attempt 1's committed fixture) sees byte-
        # identical behavior when these new parameters are left unset.
        _msl_probe_count = int(probe_count if msl_probe_count is None else msl_probe_count)
        _msl_probe_start_cells = int(
            probe_start_cells if msl_probe_start_cells is None else msl_probe_start_cells
        )
        _msl_probe_spacing_cells = int(
            probe_spacing_cells if msl_probe_spacing_cells is None else msl_probe_spacing_cells
        )
        probe_xs = msl_probe_x_coords_n(
            grid, msl_port_base, n_probes=_msl_probe_count,
            n_offset_cells=_msl_probe_start_cells,
            n_spacing_cells=_msl_probe_spacing_cells,
        )
        xs_ladder = [float(x) for x in probe_xs]
        lx_dom = float(self._domain[0])
        mono = all(
            (xs_ladder[q + 1] - xs_ladder[q]) * (1 if msl_pe.direction == "+x" else -1)
            > 0.5 * dz
            for q in range(len(xs_ladder) - 1)
        )
        if (not mono) or min(xs_ladder) <= 0.0 or max(xs_ladder) >= lx_dom:
            raise ValueError(
                "compute_coax_msl_transition(): the MSL probe ladder "
                f"({', '.join(f'{x * 1e3:.2f}' for x in xs_ladder)} mm) "
                f"leaves the declared x-domain (0, {lx_dom * 1e3:.2f}) mm "
                "or was clamped at its edge. Face the port toward the "
                "junction (direction), reduce n_probe_offset/spacing, or "
                "enlarge the domain."
            )
        # coaxial_line_reflection_from_plane_voltages requires STRICTLY
        # INCREASING plane positions; a "-x"-facing port's own ladder comes
        # back decreasing in x (probe n steps AWAY from feed_x, toward the
        # junction). Sort once here and use this order everywhere below so
        # DFT-plane construction and the voltage array stay index-consistent.
        xs_sorted = sorted(xs_ladder)

        pec_mask_np = None if pec_mask is None else np.asarray(pec_mask)
        cells = _msl_yz_cells(grid, msl_port_base)
        j_set = sorted({c[1] for c in cells})
        k_set = sorted({c[2] for c in cells})
        j_lo_msl, j_hi_msl = j_set[0], j_set[-1]
        k_lo_msl, k_hi_msl = k_set[0], k_set[-1]
        j_centre_msl = (j_lo_msl + j_hi_msl) // 2
        i_feed_msl = cells[0][0]
        if pec_mask_np is None:
            raise RuntimeError(
                "compute_coax_msl_transition(): no PEC geometry registered "
                "— the MSL trace conductor must be a registered PEC Box "
                "(pec_mask came back None)."
            )
        col = pec_mask_np[i_feed_msl, j_centre_msl, k_hi_msl:]
        k_pec = np.where(col)[0]
        if k_pec.size == 0:
            raise RuntimeError(
                "compute_coax_msl_transition(): no PEC trace conductor "
                "found above the substrate top at the registered MSL "
                "port's own feed plane; add the microstrip trace as a "
                "Box(material='pec')."
            )
        k_trace_lo = int(k_hi_msl + int(k_pec.min()))
        dz_arr = _msl_cell_profile(grid, "z", grid.nz)
        _complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64

        # ---- Two-drive FDTD run -------------------------------------------
        msl_waveform = (
            msl_pe.waveform if msl_pe.waveform is not None
            else GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
        )
        import dataclasses as _dc
        msl_port_driven = _dc.replace(msl_port_base, excitation=msl_waveform)

        v_coax_by_drive = np.zeros((2, len(probes_coax), n_f), dtype=np.complex128)
        v_msl_by_drive = np.zeros((2, len(xs_sorted), n_f), dtype=np.complex128)
        settling_db = np.full(2, np.nan, dtype=np.float64)

        x_mid_coax = center_xy[0] + 0.5 * (a + b)
        i_probe_coax = int(round(x_mid_coax / dz)) + int(grid.pad_x_lo)
        j_probe_coax = int(grid.pad_y_lo) + int(round(center_xy[1] / dz))
        i_probe_msl = int(grid.position_to_index(
            (xs_sorted[len(xs_sorted) // 2], y_centre, msl_z_lo + msl_pe.height * 0.5)
        )[0])
        j_probe_msl = int(grid.pad_y_lo) + int(round(y_centre / dz))
        k_probe_msl = int(round((msl_z_lo + 0.5 * msl_pe.height) / dz)) + int(grid.pad_z_lo)

        # drive_idx 0 drives the coax port; drive_idx 1 drives the MSL port.
        for drive_idx in range(2):
            if drive_idx == 0:
                sources = list(spec_coax.electric_sources)
                mag_sources = list(spec_coax.magnetic_sources)
            else:
                sources = make_msl_port_sources(
                    grid, msl_port_driven, materials, int(n_steps),
                    mode_profile=mode_profile,
                )
                mag_sources = []

            planes = []
            for z in probes_coax:
                for comp in ("ex", "ey"):
                    planes.append(init_dft_plane_probe(
                        axis=2, index=int(z), component=comp, freqs=freqs_jnp,
                        grid_shape=grid.shape, dft_total_steps=int(n_steps),
                    ))
            n_coax_planes = len(planes)
            for x in xs_sorted:
                i_x = int(grid.position_to_index((x, y_centre, msl_z_lo))[0])
                planes.append(init_dft_plane_probe(
                    axis=0, index=i_x, component="ez", freqs=freqs_jnp,
                    grid_shape=grid.shape, dft_total_steps=int(n_steps),
                ))

            witness_probes = [
                ProbeSpec(i=i_probe_coax, j=j_probe_coax,
                          k=int(probes_coax[len(probes_coax) // 2]), component="ex"),
                ProbeSpec(i=i_probe_msl, j=j_probe_msl, k=k_probe_msl, component="ez"),
            ]

            # #589 flux-adjudication opt-in: fresh accumulators PER DRIVE
            # (init_flux_monitor zeroes the DFT carries; sharing cfgs across
            # drives would co-accumulate both drives into one spectrum).
            _flux_run_kwargs = (
                {"flux_monitors": build_flux_monitor_cfgs(
                    self, grid, int(n_steps), entries=extra_flux_monitors)}
                if extra_flux_monitors else {}
            )
            result = _run(
                grid, materials, int(n_steps), boundary="cpml", cpml_axes="xyz",
                sources=sources, mag_sources=mag_sources, probes=witness_probes,
                dft_planes=planes, pec_mask=pec_mask, return_state=False,
                **_flux_run_kwargs,
            )
            if result.dft_planes is None:
                raise RuntimeError(
                    "compute_coax_msl_transition(): runner returned no DFT "
                    "planes"
                )
            if extra_flux_monitors:
                flux_by_drive[("coax", "msl")[drive_idx]] = {
                    entry.name: np.asarray(_flux_spectrum(fm, exact_f64=True), dtype=np.float64)
                    for entry, fm in zip(
                        extra_flux_monitors, result.flux_monitors or ()
                    )
                }

            for pi, z in enumerate(probes_coax):
                v_coax_by_drive[drive_idx, pi, :] = np.asarray(
                    coaxial_line_plane_voltage(
                        grid, result.dft_planes[pi * 2 + 0].accumulator,
                        result.dft_planes[pi * 2 + 1].accumulator,
                        center_xy=center_xy, pin_radius=a, outer_radius=b,
                    )
                )
            for pi in range(len(xs_sorted)):
                ez_plane = jnp.asarray(result.dft_planes[n_coax_planes + pi].accumulator)
                v_q = msl_modal_voltage(
                    ez_plane, j_centre=j_centre_msl, k_lo=k_lo_msl,
                    k_hi=k_trace_lo, dz_arr=dz_arr, dtype=_complex_dtype,
                )
                v_msl_by_drive[drive_idx, pi, :] = np.asarray(v_q)

            ts = np.asarray(result.time_series, dtype=float)
            if ts.ndim == 2 and ts.shape[0] >= 10 and ts.shape[1] == len(witness_probes):
                power = ts ** 2
                tail = max(1, power.shape[0] // 10)
                end = power[-tail:, :].mean(axis=0)
                peak = power.max(axis=0)
                tiny = np.finfo(float).tiny
                settling_db[drive_idx] = float(np.max(
                    10.0 * np.log10((end + tiny) / (peak + tiny))
                ))

        s_params, cond_a, cond_a_equilibrated, rec_resid, fit_resid, gamma, a_inc, b_out = \
            _assemble_coax_msl_transition_from_voltages(
                z_coax_planes_m=z_planes_coax_m, x_msl_planes_m=np.asarray(xs_sorted),
                ref_coax_m=ref_coax_m, ref_msl_m=float(junction_x),
                v_coax_by_drive=v_coax_by_drive, v_msl_by_drive=v_msl_by_drive,
                z0_coax=float(z_tem), z0_msl=float(z0_msl), cond_warn=float(cond_warn),
            )

        reference_planes = np.asarray([ref_coax_m, float(junction_x)], dtype=float)
        z0_ref = np.asarray([float(z_tem), float(z0_msl)], dtype=float)
        result_obj = CoaxMSLTransitionResult(
            s_params=s_params,
            freqs=np.asarray(freqs_arr, dtype=float),
            port_names=("coax", "msl"),
            reference_planes=reference_planes,
            z0_ref=z0_ref,
            cond_a=cond_a,
            cond_a_equilibrated=cond_a_equilibrated,
            recurrence_residual=rec_resid,
            fit_residual=fit_resid,
            gamma=gamma,
            a_inc=a_inc,
            b_out=b_out,
            settling_db=settling_db,
            status="experimental",
            flux_monitors=(flux_by_drive if extra_flux_monitors else None),
        )
        # Issue #662, same gap as compute_coaxial_two_port. ``n_steps`` here is
        # the RESOLVED record length (num_periods was folded into it above), and
        # it is the knob that overrides num_periods, so it is the actionable one.
        _warn_if_ringdown_truncated(
            settling_db, ("coax", "msl"), n_steps=int(n_steps),
        )
        return _finalize_sparam_result(
            result_obj,
            extractor="compute_coax_msl_transition",
            strict=strict_passivity,
        )

    def _compute_waveguide_s_matrix_nu(
        self,
        *,
        n_steps: int | None,
        num_periods: float,
        normalize: bool,
        eps_override=None,
        sigma_override=None,
        checkpoint_segments: int | None = None,
    ) -> WaveguideSMatrixResult:
        """Non-uniform-mesh two-run S-matrix extraction.

        Drives each port in turn, running device + vacuum-reference
        scans through ``run_nonuniform_path`` so ``dx_profile`` /
        ``dy_profile`` actually flow into the Yee update. The per-port
        drive is implemented by temporarily zeroing ``amplitude`` on
        non-driven entries; the original port list is restored in a
        ``finally`` block. Reference run uses ``eps_override`` /
        ``sigma_override`` to replace the assembled materials with
        vacuum before the scan launches.

        Current scope (matches the uniform path minus a few niceties):
          - ``normalize=True`` or ``normalize='flux'``.
          - Single-mode ports (``n_modes == 1``) only.
          - ``eps_override`` / ``sigma_override`` (the differentiable AD
            design variable) are wired only for ``normalize='flux'``: they
            are threaded into the *device* run so the traced eps flows
            through the jnp-native flux extraction and back to the
            S-matrix gradient. The *reference* run stays vacuum. They are
            rejected for ``normalize=True`` (its diagonal a_inc_ref
            denominator carries the #88 band-edge fragility).

        Extracts ``a_inc`` / ``b_out`` via the same
        ``extract_waveguide_port_waves`` helper as the uniform path and
        applies the same diagonal-subtraction + off-diagonal-division
        normalisation (see ``extract_waveguide_s_params_normalized``
        in ``rfx/sources/waveguide_port.py``).
        """
        from dataclasses import replace as _dc_replace
        from rfx.runners.nonuniform import (
            run_nonuniform_path,
            assemble_materials_nu,
        )
        from rfx.sources.waveguide_port import (
            extract_waveguide_port_waves,
            waveguide_plane_positions,
        )

        # ``normalize`` may be True (lumped V/I ratio) or "flux" (Poynting
        # power-ratio magnitude + modal phase). normalize=False is not
        # supported on the NU path.
        _flux_mode = (normalize == "flux")
        if not normalize:
            raise NotImplementedError(
                "compute_waveguide_s_matrix(normalize=False) is not yet "
                "supported on the non-uniform mesh path; use normalize=True, "
                "normalize='flux', or drop dx/dy_profile to stay on the "
                "uniform lane."
            )

        entries = list(self._waveguide_ports)
        if any(entry.n_modes > 1 for entry in entries):
            raise NotImplementedError(
                "Multi-mode waveguide ports are not yet supported on the "
                "non-uniform mesh path."
            )

        n_ports = len(entries)

        # ``_build_nonuniform_grid`` requires a concrete dz_profile.
        # Synthesise one from the scalar dx when the user did not supply
        # a dz_profile (same semantics as the uniform lane's implicit
        # z-resolution). Restored in the ``finally`` below.
        _dz_profile_saved = self._dz_profile
        if self._dz_profile is None:
            _nz = int(round(float(self._domain[2]) / float(self._dx)))
            self._dz_profile = np.full(max(_nz, 1), float(self._dx))

        # Build the grid directly so we can restrict ``cpml_axes`` to
        # axes that are not fully PEC/PMC-bounded. The rasteriser (see
        # ``rfx/geometry/rasterize_grid.py::coords_from_nonuniform_grid``)
        # uses a single ``grid.cpml_layers`` offset for every axis;
        # when a fully PEC-bounded axis is shorter than
        # ``cpml_layers + 1`` cells the offset slice hits IndexError.
        # Dropping that axis from ``cpml_axes`` keeps the physical
        # grid identical (PEC faces already have pad=0) but zeroes the
        # offset so the rasteriser snaps cells to 0 cleanly.
        from rfx.runners.nonuniform import build_nonuniform_grid
        pec_set = (self._boundary_spec.pec_faces()
                   if self._boundary_spec is not None else None) or set()
        pmc_set = (self._boundary_spec.pmc_faces()
                   if self._boundary_spec is not None else None) or set()

        def _axis_fully_closed(ax: str) -> bool:
            return {f"{ax}_lo", f"{ax}_hi"}.issubset(pec_set | pmc_set)

        cpml_axes = "".join(
            ax for ax in "xyz"
            if ax not in (self._periodic_axes or "")
            and not _axis_fully_closed(ax)
        )
        try:
            grid = build_nonuniform_grid(
                self._freq_max, self._domain, self._dx, self._cpml_layers,
                self._dz_profile,
                dx_profile=self._dx_profile,
                dy_profile=self._dy_profile,
                pec_faces=pec_set or None,
                pmc_faces=pmc_set or None,
                cpml_axes=cpml_axes,
            )
        except Exception:
            self._dz_profile = _dz_profile_saved
            raise
        if n_steps is None:
            # ``NonUniformGrid`` does not expose ``num_timesteps`` (known
            # asymmetry vs. ``Grid``); inline the same formula here.
            n_steps = int(np.ceil(num_periods / self._freq_max / float(grid.dt)))

        # Assemble device materials once to learn the full array shape;
        # vacuum reference is shape-matched onto that same array.
        dev_materials_concrete, _, _, _ = assemble_materials_nu(self, grid)
        vacuum_eps = jnp.ones_like(dev_materials_concrete.eps_r)
        vacuum_sigma = jnp.zeros_like(dev_materials_concrete.sigma)

        # Frequency grid must match across ports.
        port_freqs = entries[0].freqs
        if port_freqs is None:
            port_freqs = jnp.linspace(
                self._freq_max / 10, self._freq_max, entries[0].n_freqs,
            )
        for entry in entries[1:]:
            other = entry.freqs if entry.freqs is not None else jnp.linspace(
                self._freq_max / 10, self._freq_max, entry.n_freqs,
            )
            if other.shape != port_freqs.shape or not np.allclose(
                np.asarray(other), np.asarray(port_freqs)
            ):
                raise ValueError(
                    "waveguide S-matrix requires matching frequency grids on all ports"
                )

        # jnp-functional: collect per-drive columns; stack after loop
        s_columns: list[list] = []  # s_columns[drive_idx] = list of (n_freqs,) jnp arrays over recv_idx
        ref_shifts: tuple[float, ...] | None = None
        reference_planes_out: np.ndarray | None = None

        original_entries = list(entries)
        try:
            for drive_idx in range(n_ports):
                self._waveguide_ports = [
                    _dc_replace(
                        e,
                        amplitude=(e.amplitude if idx == drive_idx else 0.0),
                    )
                    for idx, e in enumerate(original_entries)
                ]

                # Device run: thread the public eps/sigma override (the
                # traced design variable) into the device Yee update so the
                # gradient flows from eps_override -> device fields -> flux
                # -> S-matrix (mirrors the uniform PR #172 flux-AD wiring).
                # When eps_override is None the assembled device materials
                # are used unchanged (device fields identical); the np->jnp
                # flux extraction below matches the prior np path to
                # rtol<=1e-5 (float reassociation only, per uniform PR #172).
                # issue #73: translate the uniform `checkpoint_segments` (K
                # segments) → the NU runner's `checkpoint_every` (chunk size).
                # The chunk MUST exactly divide n_steps (pad=0): the NU runner
                # zero-pads non-divisor chunks and those extra ring-down steps
                # would corrupt the carry-accumulated flux DFT (the time_series
                # is truncated to n_steps but the flux accumulator is NOT), so a
                # non-divisor chunk is NOT forward-identical for the flux
                # S-matrix — same divisor rule as the uniform V/I-DFT path. Pick
                # the divisor of n_steps nearest to n_steps/K. Checkpoint ONLY the
                # *device* run — the vacuum reference is constant in the design
                # variable so it carries no AD tape. The √N tape win is realised
                # under jax.grad (normalize='flux' + eps_override); plain forward
                # is identical.
                from rfx.simulation import _nearest_divisor
                _ckpt_every = None
                if checkpoint_segments is not None and checkpoint_segments > 1:
                    _ck = _nearest_divisor(n_steps, max(1, n_steps // int(checkpoint_segments)))
                    if 0 < _ck < n_steps:
                        _ckpt_every = _ck
                dev_result = run_nonuniform_path(
                    self, n_steps=n_steps,
                    eps_override=eps_override,
                    sigma_override=sigma_override,
                    attach_waveguide_flux=_flux_mode,
                    checkpoint_every=_ckpt_every,
                )
                # Reference run stays vacuum (incident-power reference) and is
                # independent of the design variable. ``strip_interior_pec``
                # drops the rasterized interior PEC (iris / wall / post) from
                # the reference so it is a clean empty guide: the boundary y/z
                # guide walls survive (they are enforced via pec_faces, not
                # pec_mask). Without this the vacuum override replaces only
                # eps/sigma and the reference retains the device's interior
                # PEC mask → device and reference DFTs are bit-identical →
                # (device - reference) = 0 → S11 = 0 for any PEC reflector.
                # This mirrors the uniform reference run, which builds the
                # reference with dielectric_shapes=[] + boundary-only PEC.
                ref_result = run_nonuniform_path(
                    self,
                    n_steps=n_steps,
                    eps_override=vacuum_eps,
                    sigma_override=vacuum_sigma,
                    attach_waveguide_flux=_flux_mode,
                    strip_interior_pec=True,
                    # #677: the surface-impedance sheet no longer rides
                    # materials.sigma, so sigma_override=vacuum does NOT
                    # strip it — the ctx must be stripped EXPLICITLY here,
                    # beside strip_interior_pec, or the "empty guide"
                    # reference would still carry the lossy sheet.
                    strip_sheet_impedance=True,
                )

                dev_wg = dev_result.waveguide_ports or {}
                ref_wg = ref_result.waveguide_ports or {}
                if len(dev_wg) != n_ports or len(ref_wg) != n_ports:
                    raise RuntimeError(
                        "NU waveguide S-matrix expected one final cfg per "
                        "port on both device and reference runs"
                    )

                # Compute ref_shifts from the first drive's configs (same
                # measured planes for every drive / run).
                if ref_shifts is None:
                    shifts = []
                    planes_out = []
                    for entry in original_entries:
                        cfg = dev_wg[entry.name]
                        planes = waveguide_plane_positions(cfg)
                        desired = (
                            entry.reference_plane
                            if entry.reference_plane is not None
                            else planes["source"]
                        )
                        shifts.append(desired - planes["reference"])
                        planes_out.append(desired)
                    ref_shifts = tuple(shifts)
                    reference_planes_out = np.asarray(planes_out, dtype=float)

                drive_name = original_entries[drive_idx].name
                a_inc_ref, _ = extract_waveguide_port_waves(
                    ref_wg[drive_name], ref_shift=ref_shifts[drive_idx],
                )
                safe_a_inc = jnp.where(
                    jnp.abs(a_inc_ref) > 1e-30,
                    a_inc_ref,
                    jnp.ones_like(a_inc_ref),
                )

                if _flux_mode:
                    # Power-flux magnitude + modal phase (mirrors the
                    # uniform extract_waveguide_s_matrix_flux). Immune to
                    # the band-edge a_inc_ref denominator collapse that
                    # makes the normalize=True diagonal blow up (issue #88):
                    # P_inc = |F_ref[drive]| is large and well-conditioned
                    # across the whole band, not source-spectrum-weighted.
                    F_ref = ref_result.waveguide_port_flux
                    F_dev = dev_result.waveguide_port_flux
                    if F_ref is None or F_dev is None:
                        raise RuntimeError(
                            "normalize='flux' on the NU path requires "
                            "per-port flux spectra; run_nonuniform_path did "
                            "not return waveguide_port_flux."
                        )
                    # jnp-native (mirrors the uniform PR #172 flux-AD fix):
                    # no np.asarray() concretization — keeps the whole flux
                    # extraction on the AD tape so an eps_override-traced
                    # device run yields finite gradients through the
                    # S-matrix. Uses the DOUBLE-WHERE trick at sqrt(0) /
                    # angle(0) (guard the INPUT, not just the output): a
                    # single jnp.where still leaks NaN grad through the dead
                    # branch (#171/#172/#148). Forward values are identical
                    # to the prior np version for P_inc / P > 0.
                    P_inc = jnp.abs(F_ref[drive_idx])
                    safe_P_inc = jnp.where(
                        P_inc > 1e-60, P_inc, jnp.ones_like(P_inc)
                    )
                    recv_col = []
                    for recv_idx in range(n_ports):
                        recv_name = original_entries[recv_idx].name
                        _, b_recv_dev = extract_waveguide_port_waves(
                            dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                        )
                        # AD-safe angle (double-where): angle() has an
                        # undefined gradient at 0; angle(1)=0 matches
                        # np.angle(0)=0 so the primal is unchanged.
                        ratio = b_recv_dev / safe_a_inc
                        ratio_ok = jnp.abs(ratio) > 0.0
                        phase = jnp.angle(
                            jnp.where(ratio_ok, ratio, jnp.ones_like(ratio))
                        )
                        if recv_idx == drive_idx:
                            P_num = jnp.abs(F_ref[drive_idx] - F_dev[drive_idx])
                        else:
                            P_num = jnp.abs(F_dev[recv_idx])
                        # AD-safe sqrt (double-where): a perfect match/null
                        # makes the power ratio exactly 0, where
                        # d(sqrt)/dx = inf would leak 0*inf=nan through the
                        # backward pass; primal stays exactly sqrt(x) for
                        # x>0 and exactly 0 at x=0.
                        p_ratio = P_num / safe_P_inc
                        p_ok = p_ratio > 0.0
                        mag = jnp.where(
                            p_ok,
                            jnp.sqrt(
                                jnp.where(p_ok, p_ratio, jnp.ones_like(p_ratio))
                            ),
                            0.0,
                        )
                        recv_col.append(mag * jnp.exp(1j * phase))
                    s_columns.append(recv_col)
                    continue

                recv_col: list = []
                for recv_idx in range(n_ports):
                    recv_name = original_entries[recv_idx].name
                    _, b_ref = extract_waveguide_port_waves(
                        ref_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    _, b_dev = extract_waveguide_port_waves(
                        dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    if recv_idx == drive_idx:
                        recv_col.append((b_dev - b_ref) / safe_a_inc)
                    else:
                        # Use a tighter guard than the diagonal safe_a_inc
                        # (1e-30): the NU path operates at lower float32
                        # signal levels (~1e-31) because the TFSF table
                        # injection scales with dt/dx. The reference
                        # outgoing wave b_ref at non-driven ports is
                        # proportional to the driven-port incident wave and
                        # can fall to ~1e-31 in float32. A 1e-30 guard
                        # fires falsely and replaces b_ref with 1.0, giving
                        # S21 = b_dev * 1e-31 instead of b_dev/b_ref ≈ 1.
                        # 1e-60 is safely below float32 underflow (~1e-38)
                        # so it only fires when b_ref is genuinely zero.
                        safe_b = jnp.where(
                            jnp.abs(b_ref) > 1e-60,
                            b_ref,
                            jnp.ones_like(b_ref),
                        )
                        recv_col.append(b_dev / safe_b)
                s_columns.append(recv_col)
        finally:
            self._waveguide_ports = original_entries
            self._dz_profile = _dz_profile_saved

        return WaveguideSMatrixResult(
            s_params=jnp.stack([jnp.stack(col) for col in s_columns], axis=1),
            freqs=jnp.asarray(port_freqs),
            port_names=tuple(e.name for e in original_entries),
            port_directions=tuple(e.direction for e in original_entries),
            reference_planes=reference_planes_out
            if reference_planes_out is not None
            else np.array(
                [
                    e.reference_plane if e.reference_plane is not None
                    else 0.0
                    for e in original_entries
                ],
                dtype=float,
            ),
        )
