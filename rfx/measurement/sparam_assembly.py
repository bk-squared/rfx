"""S-parameter assembly and wave split; rfx.sparams._common and rfx.api._sparams re-export every name, and new code imports from here."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
try:  # Public API on current JAX; the 0.4.x GPU image uses the old location.
    from jax import enable_x64 as _enable_x64
except ImportError:
    from jax.experimental import enable_x64 as _enable_x64
from rfx.core.jax_utils import complex_solve, is_tracer
from rfx.measurement.phasors import exp_j
from rfx.model.graded_grid import NonUniformGrid


def _msl_cell_profile(grid, axis: str, n: int) -> np.ndarray:
    """Per-cell size array (length ``n``, full/padded) along ``axis`` for
    MSL V/I integration. Graded-mesh aware.

    NU integration retains its solver-facing primal arrays. Uniform grids
    use their axis cell; optional legacy profiles still take precedence.
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
    return np.full(n, float(grid.boundary_cell(axis, "lo")), dtype=float)


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
    planes.  On node-aligned meshes
    (``h_sub/dx`` integral, e.g. dx = h_sub/3 = 84.67 µm) the two agree.

    before this helper existed the span was
    ``range(k_lo, k_hi + 1)`` with the rounding proxy — on aligned meshes
    that is ``n+1`` edges for an ``n``-cell substrate, and the extra edge
    lies inside the one-cell PEC trace, where the realization leaves the
    NORMAL E component live (an E component is PEC iff its own
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
    properly terminated two-port fixture (
    ``scripts/diagnostics/msl_vi_flux_oracle.py`` + its committed JSON):
    HELD, ratio 1.0105-1.0118, 30/30 admissible cells. An earlier
    unterminated single-port reading of this same identity on the
    bisecting mesh read low (0.54-0.69) for two compounding,
    non-extractor reasons — a convention slip
    and a reactive, non-travelling fixture (reactive fraction 0.963-0.995,
    review reconstruction) — neither of which is present in the
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

    the superseded rule ``S[j, d] = b_j / a_d`` is the ``d``-th
    column of this only when ``a_j = 0`` at every passive port.  It is not —
    ``|a_passive/a_driven| = 0.243-0.248`` at the shipped R = 50 Ω and
    0.19-0.93 across terminations, measured in VESSL 369367257265 — and
    the exact algebra ``b_1/a_1 = S11 + S12·(a_2/a_1)`` holds to machine
    precision, so the far port's echo was reported as the structure's own
    reflection.

    Unitarity is therefore lost.  How it is lost depends on phase: expanding
    for a symmetric ``S`` gives
    ``(1+γ²)(|S11|²+|S21|²) + 4γ·Re(S11·conj(S21))`` with ``γ = a_2/a_1``, so
    the old rule can push column power either side of 1.  In the NEAR-MATCHED
    case that the thru fixtures are (true ``S11 ≈ 0``) it reduces to
    ``|S11|² + |S21|² = 1 + γ²`` — the same power counted twice — which is the
    passivity violation.  Both cases are pinned in
    ``tests/unit/sparams/test_msl_modal_voltage_and_wave_solve.py``.

    ``cond_a`` bounds DEGENERACY of the drive system only — it is not a
    reliability score.  Same contract as the coax lane's
    :func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`, generalised to ``n`` ports.
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
        complex_solve(jnp.swapaxes(A, -1, -2), jnp.swapaxes(B, -1, -2)),
        -1, -2,
    )
    cond_a = None
    if not is_tracer(A):
        # f64 BEFORE the ratio: for a complex64 A the SVD returns float32,
        # where the 1e-300 floor underflows to 0.0 (NEP-50 weak promotion
        # keeps float32) and a singular A divides by zero instead of
        # saturating. Same failure class as the NEP-50 fallback.
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


def _assemble_mixed_power_wave_s(
    v_lw, i_lw, v0_msl, i_msl,
    z0_lw, n_live_lw, z0_hj_msl,
    wire_mode, drive_plan,
    v_ref_lw=None,
):
    """Assemble the mixed-family power-wave S-matrix.

    Lumped/wire-driven columns and their power witnesses are refused.
    MSL-driven waves retain the recorded-phasor algebra below. The caller
    refuses multi-cell wire receive waves after preserving their flux phase.

    .. warning::

       SINGLE-RATIO ASSEMBLY, deliberately not changed here.
       This assembly forms ``S[i, j] = b_i / a_j`` per drive — the
       single-ratio rule the pure-MSL lane replaced with the multi-drive
       solve ``S = B·A⁻¹``. That much is still true of the code below.

       ``scripts/diagnostics/i517_mixed_solve_vs_ratio_measurement.py`` and its JSON refute the claim that the
       passive port's echo contaminates the driven-MSL diagonal and propagates into the DEFAULT flux channel via
       ``P_inc = P_net / (1 - |S_jj|^2)``. The mixed lane's passive side is a
       lumped/wire port CELL, and that cell is structurally one-phasor: its
       ``b/a`` is the resistor-law constant ``s1 = -0.60000`` (to 5.8e-4 at
       ``n_live = 4``), and both shipped extraction candidates are the same
       identity paired oppositely, so no non-degenerate ``(a, b)`` pair
       exists at the port cell under any shipped formula. The apparent
       MSL-diagonal jump under the solve (0.03 -> 0.72) is the echo
       correction ``-S10·γ`` applied at a passive port where it does not
       belong, acting on two quantities based on near-field-dominated port-cell wave definitions
       (``|S10·γ|/|S11m|`` = 19.0-38.1×; closed form matches the solve to
       1.8e-7). Substituting the solved diagonal makes the flux channel
       WORSE, not better: reciprocity 10.53 % shipped -> 23.29 % solved,
       and assembling the whole mixed lane by the solve moves the shipped
       flux witness 9.80 % -> 9.85 % — no physical
       gain. Extending the solve here is therefore inert, not blocked.

       The lane's reciprocity residual (9-10 %) remains unexplained: neither mixed-lane
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
      the orthogonal receive channel, DC-falsifier-pinned
      (probes.py:925 / :1014).
    * lumped drive-port diagonal ``b = (-V - Z0*I) / (2*sqrt(Z0))`` —
      byte-frozen ``extract_lumped_s11`` algebra (probes.py:920).
    * wire drive-port diagonal ``S_ii = (Z_in - Z0)/(Z_in + Z0)`` with
      ``Z_in = -V/I`` and the FULL Z0 (probes.py:1001-1006).
    * MSL waves ``a = (V0 + Z0_hj*I)/2``, ``b = (V0 - Z0_hj*I)/2`` — the
      OpenEMS-style V*I split at probe 0 (compute_msl_s_matrix stage S1).

    The per-family extractors report pseudo-wave ``b/a`` ratios; those
    drop the ``sqrt(Re Z0)`` Kurokawa factor, which cancels only for
    equal-impedance ports. Mixed families have unequal Z0 by
    construction, so every wave above is divided by ``sqrt(Z0_ref)`` of
    its own port BEFORE forming ``S[i, j] = b_i / a_j``: reciprocity
    ``S21 == S12`` of a reciprocal structure then holds and is the
    committed internal falsifier for this normalization.

    Also returns the extractor-independent |S21| power witness for
    lumped/wire-driven columns: the port-cell
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
        # receive channel, power-wave normalized (probes.py:925/:1014).
        return (v_lw[run, i_port] - z0c_lw[i_port] * i_lw[run, i_port]) / (
            2.0 * sq_lw[i_port]
        )

    def _b_msl(run, p):
        return (v0_msl[run, p] - z0_hj_msl[p] * i_msl[run, p]) / (
            2.0 * sq_msl[p]
        )

    for run, (fam, loc) in enumerate(drive_plan):
        if fam == "lw":
            # The per-cell drive frame invalidates the entire driven column.
            S = S.at[:, loc, :].set(jnp.nan)
            s21_power[:, loc, :] = np.nan
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


def _mixed_flux_magnitude_override(
    S_wave, box_lw, plane_msl, drive_plan, msl_away_signs, n_lw,
    ill_cond_floor=0.05, refused_columns=(),
):
    """Replace off-diagonal MAGNITUDES with Poynting-flux ratios.

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
    the port-cell V*I accounting and the analytic-vs-measured MSL Z0
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
        safe_pinc = (np.full_like(p_inc, np.nan) if col in refused_columns
                     else np.where(p_inc > 0.0, p_inc, np.inf))
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
            S_out = S_out.at[i, col, :].set(
                (jnp.asarray(mag) * exp_j(jnp.angle(S_wave[i, col, :]), xp=jnp)).astype(S_wave.dtype)
            )
    return S_out, ill_cond, neg_power


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
    """Pure post-FDTD assembly: per-array V(z) -> (a_inc,b_out) -> 2x2 S.

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
        marginal fit that the jnp path handles fine (an AD-vs-FD comparator silently evaluating two different
        functions). See
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
        independent of the reference-plane extrapolation). Measured: the SAME array's ``Re(gamma)`` differs
        substantially
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

    **The general rule that constant is an instance of**.
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
    geometry and inverted its whole S-matrix.

    The lane is **validated with scope** in ``docs/guides/sparameter_support_matrix.md``. Production edits to this
    lane need the PI's approval. The geometry-based fixtures in
    ``tests/_wave_convention.py::_plant_ladder_voltages_physical`` pin the constant above independently of the
    extractor's own labels; the wrong-convention mirror asserts the exact ``inv(S_true)`` signature.

    AD note
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

    # Traced voltages (compute_coaxial_two_port(eps_scale=...))
    # -> jax.numpy assembly below; concrete -> NumPy.
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
        # --- differentiable path: jnp assembly ---
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
    """Disjoint-half self-consistency witness for ONE probe ladder.

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
    known 1 mm-span alpha-identification limit -- so a bar tight enough
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
    """Pure post-FDTD assembly: coax + MSL modal-voltage ladders -> power-wave 2x2 S.

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

    ``cond_a`` vs ``cond_a_equilibrated``
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
        ``sqrt(Z0)`` division), exposed for audit.

    Notes
    -----
    **Wave-role convention**. ``forward_amp`` is the branch
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

    Using the two-port lane's constant (``a = backward_amp``) on the opposite geometry swaps ``a`` and ``b`` at BOTH
    ports; since ``S = B inv(A)``, exchanging ``A`` and ``B``
    returns ``inv(S_true)``.  The regression gate is
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
            # Wave roles: the extractor already resolved which branch
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

    # Column-equilibrated condition number:
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
