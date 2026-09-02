#!/usr/bin/env python3
"""Label-independent ladder witnesses W1-W4 for the coax<->MSL transition (#589).

Pure NumPy post-processor for the per-probe modal-voltage ladders and the
flux-plane spectra that ``scripts/diagnostics/coax_msl_transition_settled_run.py
--dump-ladders --flux`` writes to ``<result>.ladders.npz``. Nothing here feeds
back into ``s_params``; every number is REPORT-ONLY and is stated next to the
convention it depends on so that a reader can re-derive the sign.

Sign conventions (frozen, unit-tested in ``tests/test_coax_msl_ladder_witnesses.py``)
--------------------------------------------------------------------------------
* DFT kernel is ``exp(-j 2 pi f t)`` (rfx/probes/probes.py). A wave travelling
  toward +axis, ``f(t - z/v)``, therefore has phasor ``F exp(-j beta z)``: its
  phase DECREASES with the coordinate, i.e. the adjacent-pair phase slope
  ``angle(V[p+1] conj(V[p])) / dz`` is NEGATIVE (= -beta). A -axis wave has
  slope +beta.
* Coax ladder: probes below the junction (junction at +z, "above"); MSL
  ladder: probes at larger x than the junction (junction at -x, "below").
  "toward the junction" is +axis on the coax ladder and -axis on the MSL one.
* Flux monitors: ``flux_spectrum`` returns ``integral Re(E x H*) . n dA`` with
  positive = power flowing toward +axis.

What each witness decides (review-repaired kill table)
------------------------------------------------------
W1 (phase-slope direction): SIGN only, AND ONLY BEHIND PREDECLARED
    PRECONDITIONS. On the coax ladder under the MSL drive, a positive mean
    slope means the dominant wave LEAVES the junction (-z); the POST-#822 assembler labels
    that wave ``b_out`` (outgoing), so a positive slope is the labels
    agreeing with the flow and KILLS H1. A NEGATIVE slope there -- the
    dominant wave arriving at an UNDRIVEN port, which the code would then
    be calling ``a_inc`` -- is what supports H1. (The pre-#822 assembler
    labelled that same -z wave ``a_inc``, so both conclusions were the
    other way round; the fix swapped the conclusions, not the geometry.)
    But a
    travel direction read out of extractor floor is not a measurement: the
    verdict is emitted only when the run is settled (settling_db <= -40 dB
    on both drives), the pairwise slopes agree in sign (>= 0.8), the mean
    slope is actually beta-sized (|mean|/beta_analytic in [0.7, 1.4]) and
    the same ladder's full-ladder pencil fit residual is < 0.02. Otherwise
    the row reads UNRESOLVED and names every precondition that failed.
    Magnitudes are otherwise printed next to the analytic beta but are not a
    kill: on a 1 mm ladder the pairwise slopes of an 11:1 two-wave field
    wander by +-9% and the pencil's own beta is 15-18% above analytic.
    On the MSL ladder the 15%-at-all-pairs criterion of the design is NOT
    decidable: a clean two-wave field with echo g has a worst pairwise slope
    deviation of ``2 g / (1 - g)`` (19.8% at g = 0.09), so the table prints
    the echo-aware tolerance ``2 g / (1 - g) + margin`` and the H4 kill is
    carried by W3 alone.
W2 (SWR |Gamma|): ``(max|V| - min|V|) / (max|V| + min|V|)`` is |Gamma| only
    when the ladder spans >= lambda/2; otherwise it is a LOWER BOUND. The
    coax ladder (1 mm vs lambda/2 = 17 mm) can therefore never bound the
    feed-end echo from above: Gamma_feed_end is NOT MEASURABLE on this
    fixture (the pencil's 0.09 is the only estimate) and W2 cannot kill
    H2/H3 -- the table says so instead of pretending.
W3 (subset two-wave fits): the unchanged production pencil
    ``coaxial_line_reflection_from_plane_voltages`` on ladder subsets.
    H4 kill (decidable): some MSL subset under the MSL drive has
    ``fit_residual < 0.02`` AND ``Im(gamma) / beta_HJ in [0.8, 1.3]``.
    H7 kill (decidable, impact test): |Gamma| at the coax reference plane
    refitted with alpha forced to 0 differs from the fitted-gamma value by
    < 1% at every bin (X = 1%, declared here) -- the referral bias is then
    immaterial whatever Re(gamma) the 1 mm ladder pretends to see.
W4 (flux planes): the H1 discriminator is the SIGN of ``msl_x20`` -- the
    NON-DRIVEN port -- under the coax drive, NOT the sign of ``coax_z22``.
    ``coax_z22 > 0`` is ENTAILED BY PASSIVITY and is confirmatory only: under
    the coax drive the feed sits BELOW the coax probes and the junction ABOVE
    them (ref_coax_m = 2.5 mm > max ladder z = 1.9 mm), so the net +z power at
    any plane between feed and junction is ``|a_phys|^2 - |b_phys|^2 > 0`` for
    ANY passive junction, whatever the code calls the two branches -- a witness
    keyed on it can only ever say "H1 supported". ``msl_x20`` is different:
    there is no source on the MSL side, so all the power crossing that plane
    came THROUGH the junction, and its sign is a property of the DUT rather
    than of the drive. Positive (+x, away from the junction) => the outgoing
    wave is the +x one, which the POST-#822 assembler labels ``b_out``
    (``a_inc[1] = out_msl.forward_amp``, and ``load_below`` is True on this
    lane so ``forward_amp`` is the -x wave, the one travelling toward the
    junction) => labels consistent, H1 KILLED. Negative => net power arrives
    at the junction from the undriven MSL side, so the wave the code calls
    ``a_inc`` is the one carrying power out => labels inverted, H1 supported
    (or the anomaly is genuine energy injection). Both conclusions are the
    reverse of what this file said before #822, for the same measured sign:
    the labels moved, the flux did not. The verdict carries the same predeclared
    preconditions as W1 (settled run; the box closes to within its declared
    band; |msl_x20| above the box's own closure error), and prints UNRESOLVED
    otherwise. ``R1 = coax_z22 / (|a_code|^2 - |b_code|^2)`` (POST-#822 label
    mapping; the pre-#822 dumps computed the same PHYSICAL quantity with the
    two labels exchanged) is a Z0
    CALIBRATION ratio (analytic z_tem = 45.46 ohm vs the numerical stub) with a
    predeclared >= 10% band, not an identity and NOT an H1 discriminator: its
    denominator is built from the extractor's GEOMETRIC branch identity ("coax:
    reference above => the branch travelling toward the reference plane is the
    +z wave"), which holds under H1 and under its negation alike, so R1 is
    label-blind by construction. ``R2 = msl_x20 / |MSL outgoing wave|^2``
    tests the MSL modal-voltage -> power conversion (the +x,
    away-from-junction wave on the MSL ladder -- ``b_out[1,0]`` post-#822,
    ``a_inc[1,0]`` before it, the SAME physical wave and the same number;
    the design wrote it |b_msl|^2 = 1.76e-15). Box closure ``coax_z22 = msl_x20 + top_z36 -
    xlo_x05 - ylo_y03 + yhi_y31`` (patch faces of one lossless box) is a
    port-model-free check; ``msl_x20_full`` (full plane) is reported next
    to it for the guided+radiated +x power.
Label-swap counterfactual: the two-drive solve re-run with a<->b equals
    inv(S_code) exactly (it is the algebraic inverse, not a new estimator).
    POST-#822 it is the reading the PRE-#822 assembler would have produced
    from the same field, not a prediction of an open hypothesis;
    printed as a PREDICTION of H1, never written into the legacy keys.

W4's box is NOT the predeclared #589 flux instrument -- read this before quoting it
-----------------------------------------------------------------------------------
``scripts/diagnostics/coax_msl_flux_adjudication.py`` (merged in #597/#599) is
the STANDING flux instrument on this issue. It implements, verbatim, the face
table pre-declared on issue #589 in the comment of 2026-08-07
("PRE-DECLARATION -- item 2 adjudication attempt") for the ATTEMPT-2 fixture:
xp x=2.2, xm x=0.3, yp y=3.1, ym y=0.3, zt z=3.4, zb z=0.9 mm, with explicit
outward signs, C1/C2 CONTROL runs that gate interpretation of the target
(rc=2 if a control fails), +-2-cell face-SHIFT invariance, and below-ground
``strip_{xp,xm,yp,ym}`` sub-strips that witness coax-shell tightness
("must read ~0").

W4 here is a SECOND, differently-placed box on the ATTEMPT-3 fixture
(coordinates in ``tests/test_coax_msl_transition.py``: FLUX_BOX_X_3/Y_3/Z_3,
FLUX_COAX_Z_3), and it does not supersede that predeclaration. Two of its
faces are forced by attempt 3: the clearance hole is OPEN, so the coax-shell
interior is on the power path and the box needs a face INSIDE the coax, and
that face must sit above the highest coax probe (z = 1.9 mm) and below
ref_coax_m = 2.5 mm for R1 to be comparable to the ladder's own pencil
amplitudes -- the predeclared zb = 0.9 mm is a full below-ground plane at the
BOTTOM of that ladder. The remaining coordinates are a tighter one-box choice,
not something attempt 3 forces; only the two y-face coordinates (0.3, 3.1 mm)
coincide with the predeclared ym/yp, and even they carry a different footprint.

Declared omissions (so their absence is not read as a result): this box has NO
control runs, NO face-shift invariance member -- ``top_z36`` = 3.6 mm sits 3
cells below the interior top (LZ_2 = 3.9 mm), tighter than the predeclared
zt = 3.4 mm, and the predeclaration restricted one-sided shifts exactly
because a face near the CPML inner edge is absorber-fringe contaminated, so a
settled closure residual here CANNOT be separated from face placement -- and
NO shell-tightness strips: it ASSUMES the coax-shell tightness the predeclared
instrument MEASURES. If any of those three matter to a decision, the answer
comes from ``coax_msl_flux_adjudication.py``, not from W4.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# ---- predeclared thresholds (report-only) -----------------------------------
H4_FIT_RESIDUAL_MAX = 0.02          # borrowed coax-lane convenience number
H4_BETA_RATIO_BAND = (0.8, 1.3)     # Im(gamma)/beta_HJ band, PREDECLARATION_ATTEMPT2's own
H7_ALPHA_IMPACT_MAX = 0.01          # X = 1%: |Gamma_ref| shift when alpha is forced to 0
W1_SLOPE_NOISE_MARGIN = 0.05        # added to 2g/(1-g) for the echo-aware slope tolerance
W4_R1_CALIBRATION_BAND = 0.10       # R1 is a Z0 calibration, >= 10% band (review blocker 2)
W4_CLOSURE_REL_MAX = 0.05           # "a few %" for a lossless box; qualitative if patches differ

# ---- predeclared PRECONDITIONS on the two H1 verdicts ------------------------
# A witness that emits a hypothesis verdict from numerical noise is not a
# falsifier. Every H1 verdict below is gated on ALL of the applicable
# preconditions and prints UNRESOLVED (naming the ones that failed) otherwise.
SETTLING_DB_MAX = -40.0             # this repo's ring-down settling witness rule
W1_MIN_SIGN_CONSISTENCY = 0.8       # fraction of pairwise slopes agreeing with the mean sign
W1_ABS_MEAN_OVER_BETA_BAND = (0.7, 1.4)   # |mean slope| / beta_analytic: is this a wave at all?
W1_MAX_FIT_RESIDUAL = 0.02          # the same (ladder, drive) full-ladder pencil residual
W4_MIN_SIGNAL_OVER_CLOSURE = 3.0    # |msl_x20| must exceed the box's own closure error by this

COAX_SUBSETS = ((0, 4), (2, 6))
MSL_SUBSETS = ((0, 3), (3, 6), (6, 9), (0, 5), (4, 9))
# The six faces of the attempt-3 W4 box (tests/test_coax_msl_transition.py:
# FLUX_BOX_X_3/Y_3/Z_3, FLUX_COAX_Z_3). NOT the face table pre-declared on #589
# on 2026-08-07 and implemented in scripts/diagnostics/coax_msl_flux_adjudication.py
# -- see the module docstring section "W4's box is NOT the predeclared #589 flux
# instrument" for what differs, why, and what this box does not carry (no
# controls, no shift-invariance member, no shell-tightness strips).
FLUX_BOX_FACES = ("coax_z22", "msl_x20", "top_z36", "xlo_x05", "ylo_y03", "yhi_y31")
FLUX_CLOSURE_SIGNS = {  # outflow through each face expressed in the monitor's +axis sign
    "msl_x20": +1.0, "top_z36": +1.0, "xlo_x05": -1.0, "ylo_y03": -1.0, "yhi_y31": +1.0,
}
JUNCTION_SIDE = {"coax": "above", "msl": "below"}   # where the junction sits on each ladder axis
DRIVE_ORDER = ("coax", "msl")


# =============================================================================
# Shared precondition: is the run settled?
# =============================================================================
def settling_precondition(settling_db):
    """``(ok, note)`` for the repo's ring-down settling rule.

    ``settling_db`` is the per-drive ring-down figure the driver already prints
    (``result.settling_db``). A truncated run is not in the time-harmonic
    steady state the whole witness set assumes, so no H1 verdict is emitted
    from one. ``None``/absent is NOT treated as pass: an unknown settling is a
    failed precondition, because the alternative is a verdict from an unknown
    state."""
    if settling_db is None:
        return False, "settling_db not in the dump (unknown)"
    arr = np.asarray(settling_db, dtype=float).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return False, f"settling_db not usable ({arr.tolist()})"
    worst = float(np.max(arr))
    ok = bool(worst <= SETTLING_DB_MAX)
    return ok, (f"settling_db worst {worst:+.1f} dB "
                f"{'<=' if ok else '>'} {SETTLING_DB_MAX:.0f} dB")


# =============================================================================
# W1 -- phase slopes
# =============================================================================
def phase_slopes(v, pos):
    """Adjacent-pair phase slope ``angle(V[p+1] conj(V[p])) / (pos[p+1]-pos[p])``.

    ``v`` is ``(n_probes, n_f)`` complex, ``pos`` ``(n_probes,)`` metres.
    Returns ``(n_probes - 1, n_f)`` rad/m (principal value; the largest
    physical step on either attempt-3 ladder is beta*dz <= 0.35 rad, so no
    unwrapping ambiguity).
    """
    v = np.asarray(v, dtype=np.complex128)
    pos = np.asarray(pos, dtype=np.float64)
    if v.ndim != 2 or pos.ndim != 1 or v.shape[0] != pos.shape[0] or pos.shape[0] < 2:
        raise ValueError(f"v must be (n_probes, n_f) with n_probes>=2 and pos (n_probes,); "
                         f"got {v.shape} / {pos.shape}")
    dz = np.diff(pos)
    if np.any(dz <= 0.0):
        raise ValueError("pos must be strictly increasing")
    return np.angle(v[1:, :] * np.conj(v[:-1, :])) / dz[:, None]


def travel_direction_from_slope(slope):
    """'+axis' when the phase decreases with the coordinate (slope < 0), else '-axis'."""
    return "+axis" if float(slope) < 0.0 else "-axis"


def toward_junction_axis(junction_side):
    """The travel direction that points AT the junction for a ladder whose
    junction is 'above' (+axis) or 'below' (-axis) the probes."""
    if junction_side == "above":
        return "+axis"
    if junction_side == "below":
        return "-axis"
    raise ValueError(f"junction_side must be 'above' or 'below', got {junction_side!r}")


def echo_aware_slope_tolerance(gamma_abs, margin=W1_SLOPE_NOISE_MARGIN):
    """Worst relative pairwise-slope deviation of a CLEAN two-wave field with
    reflection magnitude ``g``: the local phase velocity of ``e^{-jbz} + g e^{+jbz}``
    ranges over ``beta (1-g^2)/(1 -+ g)^2``, i.e. the slope exceeds beta by up
    to ``(1+g)/(1-g) - 1 = 2g/(1-g)`` -- 19.8% at g = 0.09. A 15%-at-all-pairs
    criterion can therefore never be met at that echo; this is the decidable
    replacement (plus a noise margin)."""
    g = float(np.clip(gamma_abs, 0.0, 0.999))
    return 2.0 * g / (1.0 - g) + float(margin)


def w1_ladder(v, pos, *, junction_side, beta_analytic, freqs):
    """W1 rows for one (ladder, drive): per-bin mean slope, sign consistency,
    travel direction, toward/away-from-junction, |slope|/beta."""
    s = phase_slopes(v, pos)                       # (n_pairs, n_f)
    mean = s.mean(axis=0)
    sign_mean = np.sign(mean)
    consistency = np.mean(np.sign(s) == sign_mean[None, :], axis=0)
    toward = toward_junction_axis(junction_side)
    rows = []
    for k in range(s.shape[1]):
        direction = travel_direction_from_slope(mean[k])
        rows.append({
            "freq_hz": float(freqs[k]),
            "slopes_rad_per_m": s[:, k].tolist(),
            "mean_slope_rad_per_m": float(mean[k]),
            "worst_pair_rel_dev_from_beta": float(
                np.max(np.abs(np.abs(s[:, k]) - beta_analytic[k]) / beta_analytic[k])),
            "sign_consistency": float(consistency[k]),
            "abs_mean_over_beta_analytic": float(abs(mean[k]) / beta_analytic[k]),
            "beta_analytic_rad_per_m": float(beta_analytic[k]),
            "dominant_travel": direction,
            "dominant_relative_to_junction": "toward" if direction == toward else "away",
        })
    return rows


def w1_h1_sign_witness(rows, *, fit_residual_by_bin, settling_db):
    """The W1 H1-sign reading on the coax ladder under the MSL drive, GATED.

    The sign of the mean phase slope only means something when the field at
    that ladder is a settled, sign-consistent, beta-sized propagating wave.
    On a truncated run the non-driven coax array sits at extractor floor --
    the driver's own A5 line says so -- and its pairwise slopes flip sign
    along the ladder while |mean|/beta is 0.04. Reading "H1 supported" out of
    that is reading noise. All four preconditions are predeclared at the top
    of this module; a row that fails any of them is UNRESOLVED and names them.
    """
    lo, hi = W1_ABS_MEAN_OVER_BETA_BAND
    s_ok, s_note = settling_precondition(settling_db)
    out = []
    for k, r in enumerate(rows):
        resid = (float(fit_residual_by_bin[k])
                 if fit_residual_by_bin is not None and k < len(fit_residual_by_bin)
                 else float("nan"))
        ratio = float(r["abs_mean_over_beta_analytic"])
        cons = float(r["sign_consistency"])
        failed = []
        if not s_ok:
            failed.append(f"not settled ({s_note})")
        if cons < W1_MIN_SIGN_CONSISTENCY:
            failed.append(f"sign_consistency {cons:.2f} < {W1_MIN_SIGN_CONSISTENCY} "
                          "(pairwise slopes disagree in sign along the ladder)")
        if not (lo <= ratio <= hi):
            failed.append(f"|mean slope|/beta_analytic {ratio:.3f} outside "
                          f"[{lo}, {hi}] (not a beta-sized propagating wave)")
        if not (np.isfinite(resid) and resid < W1_MAX_FIT_RESIDUAL):
            failed.append(f"full-ladder pencil fit_residual {resid:.2e} not < "
                          f"{W1_MAX_FIT_RESIDUAL} (the field here is not two-wave)")
        if failed:
            verdict = "UNRESOLVED -- precondition(s) failed: " + "; ".join(failed)
        elif r["dominant_relative_to_junction"] == "away":
            verdict = ("dominant wave LEAVES the junction (slope > 0): the post-#822 "
                       "assembler labels it b_out (outgoing) -> H1 KILLED")
        else:
            verdict = ("dominant wave ARRIVES at the junction (slope < 0) at an UNDRIVEN "
                       "port: the post-#822 assembler labels it a_inc (incident) -> H1 "
                       "supported")
        out.append({
            "freq_hz": r["freq_hz"],
            "coax_ladder_msl_drive_mean_slope": r["mean_slope_rad_per_m"],
            "dominant_travel": r["dominant_travel"],
            "relative_to_junction": r["dominant_relative_to_junction"],
            "sign_consistency": cons,
            "abs_mean_over_beta_analytic": ratio,
            "full_ladder_fit_residual": resid,
            "settling_note": s_note,
            "preconditions_failed": failed,
            "verdict_resolved": not failed,
            "verdict": verdict,
        })
    return out


# =============================================================================
# W2 -- SWR
# =============================================================================
def swr_reflection(v):
    """|Gamma|_SWR = (max|V| - min|V|) / (max|V| + min|V|) per frequency (n_f,)."""
    m = np.abs(np.asarray(v, dtype=np.complex128))
    hi, lo = m.max(axis=0), m.min(axis=0)
    return (hi - lo) / np.maximum(hi + lo, np.finfo(float).tiny)


def w2_ladder(v, pos, *, beta_analytic, freqs):
    pos = np.asarray(pos, dtype=float)
    span = float(pos[-1] - pos[0])
    g = swr_reflection(v)
    rows = []
    for k in range(len(freqs)):
        half_lambda = math.pi / float(beta_analytic[k])
        valid = span >= half_lambda
        rows.append({
            "freq_hz": float(freqs[k]),
            "swr_abs_gamma": float(g[k]),
            "swr": float((1.0 + g[k]) / max(1.0 - g[k], 1e-12)),
            "span_m": span,
            "half_lambda_m": half_lambda,
            "span_over_half_lambda": span / half_lambda,
            "valid_as_abs_gamma": bool(valid),
            "reading": ("|Gamma| (span >= lambda/2)" if valid
                        else "LOWER BOUND ONLY (span < lambda/2): cannot bound |Gamma| from above"),
        })
    return rows


# =============================================================================
# W3 -- two-wave fits on subsets (production pencil, unchanged) + alpha=0 refit
# =============================================================================
def two_wave_fit(pos, v, gamma, ref):
    """NumPy two-wave least squares at a GIVEN gamma (the pencil's own model
    ``V = A e^{+g(z-z0)} + B e^{-g(z-z0)}``, centred at the probe centroid).

    Returns the two travelling-wave amplitudes AT ``ref``: ``minus_axis``
    (A-branch, travels -axis) and ``plus_axis`` (B-branch, travels +axis),
    plus the relative fit residual. Label-free by construction -- the caller
    decides which one points at the junction."""
    z = np.asarray(pos, dtype=np.float64)
    V = np.asarray(v, dtype=np.complex128)
    z0 = float(z.mean())
    Phi = np.stack([np.exp(+gamma * (z - z0)), np.exp(-gamma * (z - z0))], axis=1)
    AB, *_ = np.linalg.lstsq(Phi, V, rcond=None)
    A, B = complex(AB[0]), complex(AB[1])
    resid = float(np.linalg.norm(Phi @ AB - V) / (np.linalg.norm(V) + 1e-300))
    zr = float(ref) - z0
    return {"minus_axis": A * np.exp(+gamma * zr), "plus_axis": B * np.exp(-gamma * zr),
            "fit_residual": resid}


def reflection_at_ref(fit, junction_side):
    """|wave leaving the junction| / |wave arriving at the junction| at ``ref``
    (identical to the production extractor's ``reflection`` definition: the
    incident wave is the one travelling from the probes toward the reference
    plane)."""
    toward = toward_junction_axis(junction_side)
    inc = fit["plus_axis"] if toward == "+axis" else fit["minus_axis"]
    out = fit["minus_axis"] if toward == "+axis" else fit["plus_axis"]
    return out / inc if abs(inc) > 0.0 else complex(np.nan, np.nan)


def _pencil(pos, v, ref):
    from rfx.sources.coaxial_port import coaxial_line_reflection_from_plane_voltages
    return coaxial_line_reflection_from_plane_voltages(
        np.asarray(pos, dtype=float), np.asarray(v, dtype=np.complex128),
        reference_plane_m=float(ref))


def w3_subsets(v, pos, *, ref, junction_side, beta_analytic, freqs, subsets, label):
    """Subset pencil fits (production extractor, unchanged) + the alpha-forced-
    zero refit at the same reference plane. One row per (subset, bin)."""
    pos = np.asarray(pos, dtype=float)
    v = np.asarray(v, dtype=np.complex128)
    rows = []
    for lo, hi in list(subsets) + [(0, len(pos))]:
        if hi > len(pos) or hi - lo < 3:
            continue
        name = f"{label}[{lo}:{hi}]" if (lo, hi) != (0, len(pos)) else f"{label}[all]"
        for k in range(len(freqs)):
            out = _pencil(pos[lo:hi], v[lo:hi, k], ref)
            g = complex(out.gamma)
            fit_full = two_wave_fit(pos[lo:hi], v[lo:hi, k], g, ref)
            fit_a0 = two_wave_fit(pos[lo:hi], v[lo:hi, k], 1j * g.imag, ref)
            refl_full = reflection_at_ref(fit_full, junction_side)
            refl_a0 = reflection_at_ref(fit_a0, junction_side)
            rows.append({
                "subset": name, "lo": int(lo), "hi": int(hi), "freq_hz": float(freqs[k]),
                "span_m": float(pos[hi - 1] - pos[lo]),
                "gamma_re": g.real, "gamma_im": g.imag,
                "im_gamma_over_beta_analytic": g.imag / float(beta_analytic[k]),
                "fit_residual": float(out.fit_residual),
                "recurrence_residual": float(out.recurrence_residual),
                "abs_forward_at_ref": abs(complex(out.forward_amp)),
                "abs_backward_at_ref": abs(complex(out.backward_amp)),
                "abs_reflection_at_ref": abs(complex(out.reflection)),
                "abs_reflection_at_ref_recomputed": abs(refl_full),
                "abs_reflection_at_ref_alpha0": abs(refl_a0),
                "alpha0_rel_shift": (abs(abs(refl_a0) - abs(refl_full)) / abs(refl_full)
                                     if abs(refl_full) > 0 else float("nan")),
                "abs_minus_axis_at_ref": abs(fit_full["minus_axis"]),
                "abs_plus_axis_at_ref": abs(fit_full["plus_axis"]),
            })
    return rows


def h4_kill_from_w3(rows, freqs):
    """H4 dies at a bin when SOME MSL subset (MSL drive) has fit_residual <
    0.02 and Im(gamma)/beta_HJ in [0.8, 1.3]. Returns per-bin dicts."""
    out = []
    lo_b, hi_b = H4_BETA_RATIO_BAND
    for f in freqs:
        cands = [r for r in rows if np.isclose(r["freq_hz"], f)]
        passing = [r["subset"] for r in cands
                   if r["fit_residual"] < H4_FIT_RESIDUAL_MAX
                   and lo_b <= r["im_gamma_over_beta_analytic"] <= hi_b]
        out.append({"freq_hz": float(f), "subsets_meeting_criterion": passing,
                    "h4_killed_at_bin": bool(passing)})
    return out


def h7_impact_from_w3(rows, freqs):
    """H7 dies when |Gamma_ref| shifts < 1% between fitted gamma and alpha=0
    for the full coax ladder AND every coax subset at every bin."""
    out = []
    for f in freqs:
        cands = [r for r in rows if np.isclose(r["freq_hz"], f)]
        worst = max((r["alpha0_rel_shift"] for r in cands), default=float("nan"))
        out.append({"freq_hz": float(f), "worst_alpha0_rel_shift": float(worst),
                    "subset_abs_gamma_spread": (
                        float(max(r["abs_reflection_at_ref"] for r in cands)
                              - min(r["abs_reflection_at_ref"] for r in cands)) if cands else None),
                    "h7_killed_at_bin": bool(np.isfinite(worst) and worst < H7_ALPHA_IMPACT_MAX)})
    return out


def nprobe_comparison(v, pos, *, beta_analytic, freqs, subsets):
    """Production N-probe extractor (``extract_msl_nprobe``) on MSL subsets
    vs the pencil, compared LABEL-FREE: |(-x wave)| / |(+x wave)| at probe 0
    of the subset. Needs jax; returns [] when unavailable."""
    try:
        from rfx.probes.msl_wave_decomp import extract_msl_nprobe
    except Exception as exc:  # pragma: no cover - environment-dependent
        return [{"error": f"extract_msl_nprobe unavailable: {exc}"}]
    pos = np.asarray(pos, dtype=float)
    v = np.asarray(v, dtype=np.complex128)
    rows = []
    for lo, hi in subsets:
        if hi > len(pos) or hi - lo < 3:
            continue
        x = pos[lo:hi]
        vv = v[lo:hi, :].T                          # (n_f, N)
        res = extract_msl_nprobe(vv, x, np.ones(len(freqs), dtype=complex),
                                 np.asarray(beta_analytic, dtype=float))
        alpha = np.asarray(res["alpha"], dtype=np.complex128)   # e^{-j beta x}: +x wave (beta > 0)
        gam = np.asarray(res["gamma"], dtype=np.complex128)     # e^{+j beta x}: -x wave
        beta = np.asarray(res["beta"], dtype=np.complex128)
        railed = np.asarray(res["beta_railed"], dtype=bool)
        resid = np.asarray(res["residual"], dtype=float)
        for k in range(len(freqs)):
            plus_x, minus_x = (alpha[k], gam[k]) if beta[k].real >= 0 else (gam[k], alpha[k])
            out = _pencil(x, v[lo:hi, k], x[0])
            fit = two_wave_fit(x, v[lo:hi, k], complex(out.gamma), x[0])
            fit_a0 = two_wave_fit(x, v[lo:hi, k], 1j * complex(out.gamma).imag, x[0])
            rows.append({
                "subset": f"msl[{lo}:{hi}]", "freq_hz": float(freqs[k]),
                "nprobe_beta": float(abs(beta[k].real)), "nprobe_beta_railed": bool(railed[k]),
                "nprobe_residual": float(resid[k]),
                "nprobe_ratio_minus_over_plus_at_probe0": float(abs(minus_x) / max(abs(plus_x), 1e-300)),
                "pencil_ratio_minus_over_plus_at_probe0": float(
                    abs(fit["minus_axis"]) / max(abs(fit["plus_axis"]), 1e-300)),
                "pencil_alpha0_ratio_minus_over_plus_at_probe0": float(
                    abs(fit_a0["minus_axis"]) / max(abs(fit_a0["plus_axis"]), 1e-300)),
                "pencil_im_gamma": float(complex(out.gamma).imag),
            })
    return rows


# =============================================================================
# W4 -- flux planes
# =============================================================================
def net_plus_axis_power(a_code, b_code, port_array, drive):
    """Net +axis power at a ladder in the code's OWN power-wave units.

    POST-#822 (``_assemble_coax_msl_transition_from_voltages``, Notes): on this lane both
    reference planes ARE the junction, so the branch travelling TOWARD the
    reference plane (the pencil's ``forward_amp``) is the one travelling
    toward the DUT, and the assembler labels it ``a_inc``; ``b_out`` is the
    branch travelling AWAY from the junction. Hence ``|a|^2 - |b|^2`` is the
    net power flowing TOWARD the junction at either ladder, and the +axis
    reading is that quantity signed by which axis direction points at the
    junction: +z on the coax ladder (junction above the probes) and -x on the
    MSL one (junction below them). Coax: net +z = |a|^2 - |b|^2. MSL:
    net +x = |b|^2 - |a|^2.

    BEFORE #822 the assembler used the opposite constant (``a_inc =
    backward_amp``) and this function carried the mirrored formula, so the
    NUMBER it returns is unchanged by the fix: the label swap in production
    and the sign swap here cancel, which is precisely why this file had to be
    changed WITH production rather than left alone (issue #822 review). What
    would have moved -- silently, and by a sign -- is what it returns when
    post-fix amplitudes are fed to the pre-fix formula: measured on the W5
    fixture, ``net_plus_z_power_code_units_coax`` +7.3612e-01/+8.1936e-01/
    +8.4814e-01 would have read -7.3612e-01/-8.1936e-01/-8.4814e-01, a coax
    drive net +z power that this module's own passivity check calls forbidden.

    NOTE this is derived from the extractor's GEOMETRIC branch identity (which
    of the two pencil modes travels toward the reference plane) COMPOSED with
    the assembler's geometric ``dut_sign``; both are true under H1 and under
    its negation alike. Anything built on it -- R1 in particular -- is
    therefore LABEL-BLIND and cannot discriminate H1."""
    a = abs(complex(a_code[port_array, drive])) ** 2
    b = abs(complex(b_code[port_array, drive])) ** 2
    return (a - b) if port_array == 0 else (b - a)


def h1_flux_verdict(*, msl_x20, coax_z22, closure_residual, closure_rel_residual,
                    settling_db):
    """The W4 H1 discriminator: the SIGN of ``msl_x20`` under the COAX drive.

    Why not ``coax_z22``: under the coax drive the feed sits BELOW the coax
    probes and the junction ABOVE them, so the physically incident wave travels
    +z and the net +z power at any plane between the two is
    ``|a_phys|^2 - |b_phys|^2 > 0`` for ANY passive junction -- positive
    regardless of which branch the code labels ``a`` and which ``b``. A verdict
    keyed on it can only ever come out FOR H1, which makes it not a falsifier.
    It is reported here as a passivity CHECK instead.

    Why ``msl_x20`` is different: nothing drives the MSL side, so every watt
    crossing that plane came THROUGH the junction and its sign is a property of
    the DUT, not of the drive. The POST-#822 assembler labels the +x branch
    ``b_out`` (``rfx/api/_sparams.py``: ``a_inc[1] = out_msl.forward_amp``,
    the branch travelling toward the reference plane, which IS the junction
    on this lane; ``rfx/sources/coaxial_port.py``: ``load_below`` is True
    here, so ``forward_amp`` is the -x wave). Measured +x net power therefore
    means the code's "outgoing" label sits on the wave that really is
    outgoing (H1 KILLED); measured -x net power means the code calls the
    power-carrying wave "incident" at an undriven port, i.e. the labels are
    inverted (H1 supported). Both readings are the REVERSE of what this
    function returned before #822 for the same measured sign, because the
    labels moved; the flux did not.

    Preconditions (predeclared): the run is settled; the flux box actually
    closes to within its declared band (an un-settled box's residual IS the
    stored energy, so its faces are not a steady-state power balance); and
    ``|msl_x20|`` exceeds the box's own closure error by ``W4_MIN_SIGNAL_OVER_
    CLOSURE``, so the discriminating quantity is bigger than the instrument's
    own inconsistency. Any failure -> UNRESOLVED, naming what failed.
    """
    s_ok, s_note = settling_precondition(settling_db)
    failed = []
    if not s_ok:
        failed.append(f"not settled ({s_note})")
    if not (np.isfinite(closure_rel_residual)
            and abs(closure_rel_residual) <= W4_CLOSURE_REL_MAX):
        failed.append(f"box closure {closure_rel_residual:+.2%} outside "
                      f"+-{W4_CLOSURE_REL_MAX:.0%} (not a steady-state power balance)")
    if not np.isfinite(msl_x20):
        failed.append("msl_x20 missing")
    elif not (np.isfinite(closure_residual)
              and abs(msl_x20) >= W4_MIN_SIGNAL_OVER_CLOSURE * abs(closure_residual)):
        failed.append(f"|msl_x20| {abs(msl_x20):.4e} < {W4_MIN_SIGNAL_OVER_CLOSURE:g} x "
                      f"|closure_residual| {abs(closure_residual):.4e} "
                      "(the discriminator is below the box's own error)")
    if failed:
        verdict = "UNRESOLVED -- precondition(s) failed: " + "; ".join(failed)
    elif msl_x20 > 0:
        verdict = ("msl_x20 > 0: net power at the NON-DRIVEN MSL port LEAVES the junction "
                   "(+x); the post-#822 assembler labels that same +x branch b_out "
                   "(a_inc[1] = out_msl.forward_amp = the -x wave, toward the junction) "
                   "=> the code's 'outgoing' wave IS the outgoing one => H1 KILLED")
    elif msl_x20 < 0:
        verdict = ("msl_x20 < 0: net power ARRIVES at the junction from the undriven MSL "
                   "side, so the wave the post-#822 assembler calls a_inc (-x, toward the "
                   "junction) is the one carrying power OUT => labels inverted (H1 "
                   "supported), unless the anomaly is genuine energy injection")
    else:
        verdict = "UNRESOLVED -- msl_x20 is exactly zero"
    if not np.isfinite(coax_z22):
        conf = "coax_z22 missing"
    elif coax_z22 > 0:
        conf = (f"coax_z22 {coax_z22:+.4e} > 0 as REQUIRED BY PASSIVITY (feed below the "
                "coax probes, junction above them): CONFIRMATORY ONLY -- this sign is "
                "entailed for any passive junction under either hypothesis and is NOT "
                "evidence for H1")
    else:
        conf = (f"coax_z22 {coax_z22:+.4e} <= 0, which passivity FORBIDS under the coax "
                "drive: the flux box or the run is unphysical. This is a failed sanity "
                "check on the instrument, not evidence about H1")
    return {
        "h1_discriminator": "sign(msl_x20) under the coax drive (the non-driven port)",
        "h1_flux_verdict": verdict,
        "h1_verdict_resolved": not failed,
        "h1_preconditions_failed": failed,
        "coax_z22_passivity_check": conf,
        "coax_z22_sign_is_passivity_entailed": True,
        "settling_note": s_note,
    }


def w4_flux(flux_by_drive, *, a_inc, b_out, freqs, settling_db=None):
    """Per (drive, bin): faces, closure residual, H1 flux verdict, R1, R2."""
    out = {"faces": list(FLUX_BOX_FACES), "closure_identity":
           "coax_z22 = msl_x20 + top_z36 - xlo_x05 - ylo_y03 + yhi_y31 (patch faces of one "
           "lossless box; +axis-positive flux)", "per_drive": {}}
    a_inc = np.asarray(a_inc, dtype=np.complex128)
    b_out = np.asarray(b_out, dtype=np.complex128)
    for d_idx, drive in enumerate(DRIVE_ORDER):
        spectra = flux_by_drive.get(drive) if flux_by_drive else None
        if not spectra:
            out["per_drive"][drive] = {"missing": True}
            continue
        rows = []
        for k in range(len(freqs)):
            faces = {n: float(np.asarray(spectra[n])[k]) for n in spectra}
            have_box = all(n in faces for n in FLUX_BOX_FACES)
            outflow = (sum(FLUX_CLOSURE_SIGNS[n] * faces[n] for n in FLUX_CLOSURE_SIGNS)
                       if have_box else float("nan"))
            closure_res = faces.get("coax_z22", float("nan")) - outflow
            scale = max([abs(faces[n]) for n in FLUX_BOX_FACES if n in faces] + [1e-300])
            coax_z22 = faces.get("coax_z22", float("nan"))
            msl_x20 = faces.get("msl_x20", float("nan"))
            net_coax = net_plus_axis_power(a_inc[:, :, k], b_out[:, :, k], 0, d_idx)
            net_msl = net_plus_axis_power(a_inc[:, :, k], b_out[:, :, k], 1, d_idx)
            # The MSL ladder's OUTGOING (+x, away-from-junction) wave. Named
            # by the physics, read out of whichever array the assembler's own
            # geometric split puts it in: post-#822 that is b_out (issue #822).
            msl_out_sq = abs(b_out[1, d_idx, k]) ** 2
            row = {
                "freq_hz": float(freqs[k]), "faces": faces,
                "box_outflow_sum": outflow, "closure_residual": closure_res,
                "closure_rel_residual": (closure_res / scale if have_box else float("nan")),
                "net_plus_z_power_code_units_coax": net_coax,
                "net_plus_x_power_code_units_msl": net_msl,
                "R1_coax_z22_over_net_code_coax": (coax_z22 / net_coax if net_coax != 0 else float("nan")),
                "R2_msl_x20_over_abs_msl_outgoing_sq": (msl_x20 / msl_out_sq if msl_out_sq > 0 else float("nan")),
                "R2net_msl_x20_over_net_code_msl": (msl_x20 / net_msl if net_msl != 0 else float("nan")),
                "S_side_ratio_abs_msl_outgoing_sq_over_net_coax": (msl_out_sq / net_coax if net_coax != 0 else float("nan")),
                "flux_ratio_msl_x20_over_coax_z22": (msl_x20 / coax_z22 if coax_z22 != 0 else float("nan")),
            }
            if drive == "coax":
                row.update(h1_flux_verdict(
                    msl_x20=msl_x20, coax_z22=coax_z22,
                    closure_residual=closure_res,
                    closure_rel_residual=row["closure_rel_residual"],
                    settling_db=settling_db))
                r1 = row["R1_coax_z22_over_net_code_coax"]
                row["R1_within_calibration_band"] = bool(
                    np.isfinite(r1) and abs(abs(r1) - 1.0) <= W4_R1_CALIBRATION_BAND)
            else:
                row["msl_drive_expectations"] = {
                    "msl_x20_negative": bool(np.isfinite(msl_x20) and msl_x20 < 0),
                    "coax_z22_negative": bool(np.isfinite(coax_z22) and coax_z22 < 0),
                    "abs_coax_z22_le_abs_msl_x20": bool(np.isfinite(coax_z22) and np.isfinite(msl_x20)
                                                        and abs(coax_z22) <= abs(msl_x20)),
                }
            rows.append(row)
        out["per_drive"][drive] = {"rows": rows}
    return out


# =============================================================================
# Label-swap counterfactual
# =============================================================================
def label_swap_counterfactual(a_inc, b_out):
    """Re-solve S with a<->b. POST-#822 this is no longer a prediction of a
    live hypothesis: it is what the PRE-#822 assembler returned, i.e. exactly
    inv(S_code), kept as a report-only bridge between the two conventions.
    Returns dict with
    ``s_swap`` (2,2,n_f), column powers, lambda_min(I - S^H S) and the max
    deviation from inv(S_code) -- algebraically identical, see test."""
    a = np.asarray(a_inc, dtype=np.complex128)
    b = np.asarray(b_out, dtype=np.complex128)
    n_f = a.shape[-1]
    s_code = np.full((2, 2, n_f), np.nan + 0j)
    s_swap = np.full((2, 2, n_f), np.nan + 0j)
    for k in range(n_f):
        try:
            s_code[:, :, k] = b[:, :, k] @ np.linalg.inv(a[:, :, k])
            s_swap[:, :, k] = a[:, :, k] @ np.linalg.inv(b[:, :, k])
        except np.linalg.LinAlgError:
            pass
    lam_min = np.array([float(np.min(np.linalg.eigvalsh(
        np.eye(2) - s_swap[:, :, k].conj().T @ s_swap[:, :, k]))) if np.all(np.isfinite(s_swap[:, :, k]))
        else np.nan for k in range(n_f)])
    with np.errstate(all="ignore"):
        inv_code = np.stack([np.linalg.inv(s_code[:, :, k]) if np.all(np.isfinite(s_code[:, :, k]))
                             else np.full((2, 2), np.nan) for k in range(n_f)], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        single_swapped = np.abs(a[0, 0] / b[0, 0])
        single_code = np.abs(b[0, 0] / a[0, 0])
    return {
        "s_code": s_code, "s_swap": s_swap,
        "col_power_coax_drive": (np.abs(s_swap[0, 0]) ** 2 + np.abs(s_swap[1, 0]) ** 2),
        "col_power_msl_drive": (np.abs(s_swap[0, 1]) ** 2 + np.abs(s_swap[1, 1]) ** 2),
        "lambda_min_I_minus_SHS": lam_min,
        "max_abs_dev_from_inv_s_code": float(np.nanmax(np.abs(s_swap - inv_code))),
        "single_ratio_abs_coax_refl_swapped": single_swapped,
        "single_ratio_abs_coax_refl_code": single_code,
    }


def ab_from_driver_ext(ext):
    """Rebuild a_inc/b_out (2,2,n_f) from a driver JSON's ``ext_589`` block."""
    def _arr(block):
        re, im = block["re"], block["im"]
        return np.array([
            [np.asarray(re["coax_array"]["coax_drive"]) + 1j * np.asarray(im["coax_array"]["coax_drive"]),
             np.asarray(re["coax_array"]["msl_drive"]) + 1j * np.asarray(im["coax_array"]["msl_drive"])],
            [np.asarray(re["msl_array"]["coax_drive"]) + 1j * np.asarray(im["msl_array"]["coax_drive"]),
             np.asarray(re["msl_array"]["msl_drive"]) + 1j * np.asarray(im["msl_array"]["msl_drive"])],
        ], dtype=np.complex128)
    return _arr(ext["a_inc"]), _arr(ext["b_out"])


# =============================================================================
# Orchestration + tables
# =============================================================================
def compute_witnesses(d):
    """``d`` is a dict/npz with keys: coax_ladder_v (2,n_c,n_f), coax_ladder_z_m,
    msl_ladder_v (2,n_m,n_f), msl_ladder_x_m, ref_coax_m, ref_msl_m, freqs,
    beta_coax_analytic, beta_msl_analytic, a_inc, b_out; optional
    flux_by_drive {drive: {name: (n_f,)}}, gamma, settling_db. Ladders may be absent
    (then W1-W3 are skipped and only W4 + the counterfactual are computed)."""
    freqs = np.asarray(d["freqs"], dtype=float)
    settling_db = d.get("settling_db")
    a_inc = np.asarray(d["a_inc"], dtype=np.complex128)
    b_out = np.asarray(d["b_out"], dtype=np.complex128)
    out = {"freqs_hz": freqs.tolist(), "conventions": {
        "dft_kernel": "exp(-j 2 pi f t): a +axis-travelling wave has NEGATIVE adjacent-pair phase slope",
        "coax_ladder": "junction ABOVE the probes (+z is toward the junction)",
        "msl_ladder": "junction BELOW the probes (-x is toward the junction)",
        "flux": "positive = power flowing toward +axis",
        "code_labels": "POST-#822: a_inc = pencil forward_amp (travels toward the reference "
                       "plane, which on this lane IS the junction); b_out = backward_amp. Runs "
                       "recorded BEFORE the #822 fix carry the opposite mapping under these same "
                       "key names -- see the Notes of rfx/api/_sparams.py::"
                       "_assemble_coax_msl_transition_from_voltages.",
    }}
    have_ladders = d.get("coax_ladder_v") is not None and d.get("msl_ladder_v") is not None
    if have_ladders:
        bc = np.asarray(d["beta_coax_analytic"], dtype=float)
        bm = np.asarray(d["beta_msl_analytic"], dtype=float)
        vc = np.asarray(d["coax_ladder_v"], dtype=np.complex128)
        vm = np.asarray(d["msl_ladder_v"], dtype=np.complex128)
        zc = np.asarray(d["coax_ladder_z_m"], dtype=float)
        xm = np.asarray(d["msl_ladder_x_m"], dtype=float)
        ref_c = float(d["ref_coax_m"])
        ref_m = float(d["ref_msl_m"])
        w1, w2, w3, w3_nprobe = {}, {}, {}, {}
        for di, drive in enumerate(DRIVE_ORDER):
            w1[f"coax_ladder/{drive}_drive"] = w1_ladder(
                vc[di], zc, junction_side="above", beta_analytic=bc, freqs=freqs)
            w1[f"msl_ladder/{drive}_drive"] = w1_ladder(
                vm[di], xm, junction_side="below", beta_analytic=bm, freqs=freqs)
            w2[f"coax_ladder/{drive}_drive"] = w2_ladder(vc[di], zc, beta_analytic=bc, freqs=freqs)
            w2[f"msl_ladder/{drive}_drive"] = w2_ladder(vm[di], xm, beta_analytic=bm, freqs=freqs)
            w3[f"coax_ladder/{drive}_drive"] = w3_subsets(
                vc[di], zc, ref=ref_c, junction_side="above", beta_analytic=bc, freqs=freqs,
                subsets=COAX_SUBSETS, label="coax")
            w3[f"msl_ladder/{drive}_drive"] = w3_subsets(
                vm[di], xm, ref=ref_m, junction_side="below", beta_analytic=bm, freqs=freqs,
                subsets=MSL_SUBSETS, label="msl")
            w3_nprobe[f"msl_ladder/{drive}_drive"] = nprobe_comparison(
                vm[di], xm, beta_analytic=bm, freqs=freqs, subsets=((6, 9), (4, 9)))
        # Echo-aware W1 tolerance for the MSL ladder from the full-ladder pencil's own |Gamma|.
        for key, rows in w1.items():
            fits = w3[key]
            for r in rows:
                full = [f for f in fits if f["subset"].endswith("[all]") and np.isclose(f["freq_hz"], r["freq_hz"])]
                g = full[0]["abs_reflection_at_ref"] if full else float("nan")
                g = min(g, 1.0 / g) if (np.isfinite(g) and g > 0) else g
                tol = echo_aware_slope_tolerance(g) if np.isfinite(g) else float("nan")
                r["pencil_abs_gamma_min_branch"] = float(g)
                r["echo_aware_slope_tolerance"] = float(tol)
                r["worst_pair_within_echo_aware_tolerance"] = bool(
                    np.isfinite(tol) and r["worst_pair_rel_dev_from_beta"] <= tol)
        out["W1_phase_slopes"] = w1
        full_resid = []
        for k in range(len(freqs)):
            hit = [f for f in w3["coax_ladder/msl_drive"]
                   if f["subset"].endswith("[all]") and np.isclose(f["freq_hz"], float(freqs[k]))]
            full_resid.append(hit[0]["fit_residual"] if hit else float("nan"))
        out["W1_h1_sign_witness"] = w1_h1_sign_witness(
            w1["coax_ladder/msl_drive"], fit_residual_by_bin=full_resid, settling_db=settling_db)
        out["W1_h1_sign_preconditions"] = (
            f"a verdict is emitted only when ALL hold: settling_db <= {SETTLING_DB_MAX:.0f} dB on "
            f"both drives; sign_consistency >= {W1_MIN_SIGN_CONSISTENCY}; |mean slope|/beta_analytic "
            f"in {list(W1_ABS_MEAN_OVER_BETA_BAND)}; full-ladder pencil fit_residual < "
            f"{W1_MAX_FIT_RESIDUAL}. Otherwise UNRESOLVED. Rationale: on a truncated run the "
            "non-driven coax array is extractor floor (the driver's own A5 line says so), its "
            "pairwise slopes flip sign along the ladder and |mean|/beta is ~0.04 -- a verdict "
            "there would be a reading of noise.")
        out["W2_swr"] = w2
        out["W2_note"] = ("coax ladder spans 1 mm vs lambda/2 ~ 17 mm: its SWR |Gamma| is a LOWER BOUND "
                          "and cannot bound the feed-end echo from above; Gamma_feed_end is NOT "
                          "MEASURABLE on this fixture (pencil 0.09 is the only estimate) -> W2 cannot "
                          "kill H2/H3. MSL ladder (8 mm) is a valid |Gamma| only where flagged.")
        out["W3_subset_fits"] = w3
        out["W3_nprobe_comparison"] = w3_nprobe
        out["W3_h4_kill"] = h4_kill_from_w3(w3["msl_ladder/msl_drive"], freqs)
        out["W3_h4_kill_rule"] = ("H4 killed at a bin iff some MSL subset under the MSL drive has "
                                  f"fit_residual < {H4_FIT_RESIDUAL_MAX} and Im(gamma)/beta_HJ in "
                                  f"{list(H4_BETA_RATIO_BAND)} (slope-at-all-pairs is NOT used: a clean "
                                  "two-wave field with echo g deviates by up to 2g/(1-g))")
        out["W3_h7_impact"] = h7_impact_from_w3(w3["coax_ladder/coax_drive"], freqs)
        out["W3_h7_rule"] = (f"H7 killed iff |Gamma_ref| shifts < {H7_ALPHA_IMPACT_MAX:.0%} between the "
                             "fitted gamma and alpha forced to 0 for the full coax ladder and every "
                             "subset, at every bin (impact test; Re(gamma) itself is not identifiable on "
                             "a 1 mm ladder). NOISE-LIMITED: at the settled run's own coax-ladder fit "
                             "residual (~2e-4) a synthetic LOSSLESS ladder already shows shifts of "
                             "0.05-1.1% (full 6-probe) and 0.6-4.8% (4-probe subsets) -- measured in "
                             "tests/test_coax_msl_ladder_witnesses.py::test_h7_impact_on_the_coax_ladder"
                             "_is_noise_limited_and_says_so -- so read the printed shift, not the "
                             "boolean, and treat a 'survives' at the 1-2% level as UNRESOLVED rather "
                             "than as evidence of real loss")
    else:
        out["W1_W2_W3"] = "SKIPPED: no per-probe ladders in this dump"
    flux = d.get("flux_by_drive")
    if flux:
        out["W4_flux"] = w4_flux(flux, a_inc=a_inc, b_out=b_out, freqs=freqs,
                                 settling_db=settling_db)
        out["W4_rules"] = {
            "predeclaration_relation":
                "THIS IS NOT THE PREDECLARED #589 FLUX INSTRUMENT. The standing one is "
                "scripts/diagnostics/coax_msl_flux_adjudication.py (merged #597/#599), which "
                "implements verbatim the face table pre-declared on #589 in the comment of "
                "2026-08-07 ('PRE-DECLARATION -- item 2 adjudication attempt') on the ATTEMPT-2 "
                "fixture (xp x=2.2, xm x=0.3, yp y=3.1, ym y=0.3, zt z=3.4, zb z=0.9 mm) with "
                "C1/C2 control runs that gate interpretation, +-2-cell face-shift invariance and "
                "below-ground shell-tightness strips. The attempt-3 box used here has DIFFERENT "
                "coordinates (its bottom face must sit inside the coax, above the top coax probe "
                "at 1.9 mm and below ref_coax_m = 2.5 mm, because the clearance hole is now open "
                "and R1 is compared to the ladder's own pencil amplitudes), and carries NO "
                "controls, NO shift-invariance member (top_z36 = 3.6 mm is 3 cells below the "
                "interior top, tighter than the predeclared zt = 3.4 mm, so a closure residual "
                "here cannot be separated from face placement) and NO shell-tightness strips (it "
                "ASSUMES the tightness the predeclared instrument MEASURES). These numbers are "
                "not directly comparable to that instrument's; the PI arbitrates.",
            "h1_sign": "coax drive, POST-#822: msl_x20 > 0 (net power LEAVES the junction at the "
                       "NON-DRIVEN port, whose +x branch the assembler now labels b_out) => labels "
                       "consistent, H1 KILLED; msl_x20 < 0 => labels inverted, H1 supported. NOT "
                       "keyed on coax_z22. Pre-#822 runs read this sign the other way round.",
            "h1_preconditions": f"the msl_x20 verdict is emitted only when settling_db <= "
                                f"{SETTLING_DB_MAX:.0f} dB on both drives, |closure_rel_residual| <= "
                                f"{W4_CLOSURE_REL_MAX} and |msl_x20| >= "
                                f"{W4_MIN_SIGNAL_OVER_CLOSURE:g} x |closure_residual|; otherwise "
                                f"UNRESOLVED.",
            "coax_z22": "PASSIVITY CHECK, not an H1 verdict: under the coax drive the feed is below "
                        "the coax probes and the junction above them, so net +z power there is "
                        "|a_phys|^2-|b_phys|^2 > 0 for ANY passive junction, whichever branch the "
                        "code labels a and which b. coax_z22 > 0 is therefore confirmatory only; "
                        "coax_z22 <= 0 would mean the box or the run is unphysical.",
            "R1": f"coax_z22 / (|a_code|^2 - |b_code|^2) on the coax ladder (POST-#822 label "
                  f"mapping; the pre-#822 dumps computed the same PHYSICAL quantity as "
                  f"|b_code|^2 - |a_code|^2): Z0 CALIBRATION ratio, predeclared band "
                  f"+-{W4_R1_CALIBRATION_BAND:.0%} (analytic z_tem vs numerical stub). LABEL-BLIND "
                  f"and therefore NOT an H1 discriminator: its denominator comes from the "
                  f"extractor's geometric branch identity ('coax: reference above => the branch "
                  f"toward the reference plane is the +z wave'), which holds under H1 and under "
                  f"its negation alike.",
            "R2": "msl_x20 / |MSL outgoing wave|^2: MSL modal-voltage -> power conversion. The "
                  "outgoing (+x, away-from-junction) wave is b_out[1,0] post-#822 and was "
                  "a_inc[1,0] before it; the physical quantity, and hence the number, is the same "
                  "(= the design's '|b_msl|^2 = 1.76e-15').",
            "closure": f"|closure_rel_residual| <= {W4_CLOSURE_REL_MAX} expected for a lossless box of "
                       "patch faces AT STEADY STATE ONLY: the identity is a time-harmonic power "
                       "balance, so on a truncated (un-settled) run the residual is the energy still "
                       "stored in the box, not a leak -- measured 78% on a 300-step attempt-3 smoke "
                       "(settling -1.6 dB) and expected to close only once settling_db clears -40 dB. "
                       "msl_x20_full is a full-plane comparator and is NOT one of the six faces.",
        }
    else:
        out["W4_flux"] = "SKIPPED: no flux spectra in this dump"
    cf = label_swap_counterfactual(a_inc, b_out)
    out["label_swap_counterfactual"] = {
        "note": "a<->b re-solved, NOT a measurement; equals inv(S_code) exactly. POST-#822 this "
                "is the S the PRE-#822 assembler would have reported from the same field, not a "
                "prediction of an open hypothesis (#822 settled H1 analytically).",
        "s_swap_re": cf["s_swap"].real.tolist(), "s_swap_im": cf["s_swap"].imag.tolist(),
        "s_swap_abs": np.abs(cf["s_swap"]).tolist(),
        "col_power_coax_drive": cf["col_power_coax_drive"].tolist(),
        "col_power_msl_drive": cf["col_power_msl_drive"].tolist(),
        "lambda_min_I_minus_SHS": cf["lambda_min_I_minus_SHS"].tolist(),
        "max_abs_dev_from_inv_s_code": cf["max_abs_dev_from_inv_s_code"],
        "single_ratio_abs_coax_refl_swapped": cf["single_ratio_abs_coax_refl_swapped"].tolist(),
        "single_ratio_abs_coax_refl_code": cf["single_ratio_abs_coax_refl_code"].tolist(),
    }
    return out


def _fg(f):
    return f"{f / 1e9:5.2f}"


def format_tables(w):
    """Human-readable tables (list of lines)."""
    L = []
    freqs = w["freqs_hz"]
    if "W1_phase_slopes" in w:
        L.append("=== W1: adjacent-pair phase slopes (rad/m; DFT exp(-jwt): +axis wave => slope < 0) ===")
        L.append("  ladder/drive           f[GHz]  mean slope   |mean|/beta  sign-cons  worst-pair dev  echo-tol  travel  vs junction")
        for key, rows in w["W1_phase_slopes"].items():
            for r in rows:
                L.append(f"  {key:22s} {_fg(r['freq_hz'])}  {r['mean_slope_rad_per_m']:+10.1f}  "
                         f"{r['abs_mean_over_beta_analytic']:10.3f}  {r['sign_consistency']:8.2f}  "
                         f"{r['worst_pair_rel_dev_from_beta']:13.1%}  {r.get('echo_aware_slope_tolerance', float('nan')):7.1%}  "
                         f"{r['dominant_travel']:6s}  {r['dominant_relative_to_junction']}")
                L.append(f"      pairs: {', '.join(f'{s:+.0f}' for s in r['slopes_rad_per_m'])}")
        L.append("  H1 sign witness (coax ladder, MSL drive):")
        L.append(f"    preconditions: {w.get('W1_h1_sign_preconditions', '')}")
        for r in w["W1_h1_sign_witness"]:
            L.append(f"    {_fg(r['freq_hz'])} GHz  slope {r['coax_ladder_msl_drive_mean_slope']:+.1f}  "
                     f"cons {r['sign_consistency']:.2f}  |mean|/beta "
                     f"{r.get('abs_mean_over_beta_analytic', float('nan')):.3f}  fit_res "
                     f"{r.get('full_ladder_fit_residual', float('nan')):.2e}  "
                     f"[{'RESOLVED' if r.get('verdict_resolved') else 'UNRESOLVED'}]  {r['verdict']}")
    if "W2_swr" in w:
        L.append("=== W2: SWR |Gamma| = (max|V|-min|V|)/(max|V|+min|V|) ===")
        L.append("  ladder/drive           f[GHz]  |Gamma|_SWR   SWR    span/(lambda/2)  reading")
        for key, rows in w["W2_swr"].items():
            for r in rows:
                L.append(f"  {key:22s} {_fg(r['freq_hz'])}  {r['swr_abs_gamma']:10.4f}  {r['swr']:6.2f}  "
                         f"{r['span_over_half_lambda']:14.3f}  {r['reading']}")
        L.append(f"  NOTE: {w['W2_note']}")
    if "W3_subset_fits" in w:
        L.append("=== W3: subset two-wave fits (production pencil, unchanged) at the reference plane ===")
        L.append("  ladder/drive           subset        f[GHz]  Re(g)     Im(g)    Im/beta  fit_res   rec_res   |fwd|       |bwd|       |Gamma|   |Gamma|a=0  a0 shift")
        for key, rows in w["W3_subset_fits"].items():
            for r in rows:
                L.append(f"  {key:22s} {r['subset']:12s}  {_fg(r['freq_hz'])}  {r['gamma_re']:7.1f}  "
                         f"{r['gamma_im']:8.1f}  {r['im_gamma_over_beta_analytic']:7.3f}  "
                         f"{r['fit_residual']:8.2e}  {r['recurrence_residual']:8.2e}  "
                         f"{r['abs_forward_at_ref']:10.3e}  {r['abs_backward_at_ref']:10.3e}  "
                         f"{r['abs_reflection_at_ref']:8.4f}  {r['abs_reflection_at_ref_alpha0']:9.4f}  "
                         f"{r['alpha0_rel_shift']:8.2%}")
        L.append("  N-probe (extract_msl_nprobe) vs pencil, |(-x wave)|/|(+x wave)| at probe 0 of the subset:")
        for key, rows in w["W3_nprobe_comparison"].items():
            for r in rows:
                if "error" in r:
                    L.append(f"  {key:22s} {r['error']}")
                    continue
                L.append(f"  {key:22s} {r['subset']:12s}  {_fg(r['freq_hz'])}  nprobe {r['nprobe_ratio_minus_over_plus_at_probe0']:.4f} "
                         f"(beta {r['nprobe_beta']:.1f}, railed {r['nprobe_beta_railed']}, res {r['nprobe_residual']:.2e})  "
                         f"pencil {r['pencil_ratio_minus_over_plus_at_probe0']:.4f} (a=0: {r['pencil_alpha0_ratio_minus_over_plus_at_probe0']:.4f}, "
                         f"Im g {r['pencil_im_gamma']:.1f})")
        L.append(f"  H4 kill rule: {w['W3_h4_kill_rule']}")
        for r in w["W3_h4_kill"]:
            L.append(f"    {_fg(r['freq_hz'])} GHz  subsets meeting criterion: {r['subsets_meeting_criterion'] or 'none'}  "
                     f"-> H4 {'KILLED' if r['h4_killed_at_bin'] else 'survives'} at this bin")
        L.append(f"  H7 rule: {w['W3_h7_rule']}")
        for r in w["W3_h7_impact"]:
            L.append(f"    {_fg(r['freq_hz'])} GHz  worst alpha=0 shift of |Gamma_ref| {r['worst_alpha0_rel_shift']:.3%}  "
                     f"subset |Gamma| spread {r['subset_abs_gamma_spread']:.4f}  -> H7 "
                     f"{'KILLED' if r['h7_killed_at_bin'] else 'survives'} at this bin")
    elif "W1_W2_W3" in w:
        L.append(f"=== W1-W3: {w['W1_W2_W3']} ===")
    w4 = w.get("W4_flux")
    if isinstance(w4, dict):
        L.append("=== W4: flux planes (W, +axis-positive; exact_f64 spectra) ===")
        for drive, blk in w4["per_drive"].items():
            if blk.get("missing"):
                L.append(f"  {drive} drive: missing")
                continue
            L.append(f"  --- {drive} drive ---")
            names = list(blk["rows"][0]["faces"].keys())
            L.append("  f[GHz]  " + "  ".join(f"{n:>12s}" for n in names) + "   closure_res(rel)")
            for r in blk["rows"]:
                L.append(f"  {_fg(r['freq_hz'])}  " + "  ".join(f"{r['faces'][n]:+12.4e}" for n in names)
                         + f"   {r['closure_residual']:+.3e} ({r['closure_rel_residual']:+.2%})")
            for r in blk["rows"]:
                L.append(f"  {_fg(r['freq_hz'])}  net code-units coax (|a|^2-|b|^2) {r['net_plus_z_power_code_units_coax']:+.4e}  "
                         f"R1 {r['R1_coax_z22_over_net_code_coax']:+.4f}  "
                         f"R2 {r['R2_msl_x20_over_abs_msl_outgoing_sq']:+.4f}  "
                         f"R2net {r['R2net_msl_x20_over_net_code_msl']:+.4f}  "
                         f"S-side |msl_out|^2/net_coax {r['S_side_ratio_abs_msl_outgoing_sq_over_net_coax']:+.4f}  "
                         f"flux msl_x20/coax_z22 {r['flux_ratio_msl_x20_over_coax_z22']:+.4f}")
                if "h1_flux_verdict" in r:
                    L.append(f"        H1 discriminator ({r['h1_discriminator']}) "
                             f"[{'RESOLVED' if r['h1_verdict_resolved'] else 'UNRESOLVED'}]: "
                             f"{r['h1_flux_verdict']}")
                    L.append(f"        passivity check: {r['coax_z22_passivity_check']}")
                    L.append(f"        R1 (Z0 calibration, label-blind, NOT an H1 verdict) within "
                             f"+-{W4_R1_CALIBRATION_BAND:.0%} band: {r['R1_within_calibration_band']}")
                if "msl_drive_expectations" in r:
                    L.append(f"        MSL-drive expectations: {r['msl_drive_expectations']}")
        for k, v in w["W4_rules"].items():
            L.append(f"  rule {k}: {v}")
    elif w4:
        L.append(f"=== W4: {w4} ===")
    cf = w["label_swap_counterfactual"]
    L.append("=== label-swap counterfactual (a<->b re-solved): the PRE-#822 reading of the same "
             "field, NOT a measurement ===")
    L.append(f"  max |S_swap - inv(S_code)| = {cf['max_abs_dev_from_inv_s_code']:.2e} (algebraic identity)")
    for k, f in enumerate(freqs):
        sa = np.asarray(cf["s_swap_abs"])[:, :, k]
        L.append(f"  {_fg(f)} GHz  |S_swap| = [[{sa[0, 0]:.4f}, {sa[0, 1]:.4f}], [{sa[1, 0]:.4f}, {sa[1, 1]:.4f}]]  "
                 f"colP coax {cf['col_power_coax_drive'][k]:.4f}  colP msl {cf['col_power_msl_drive'][k]:.4f}  "
                 f"lam_min {cf['lambda_min_I_minus_SHS'][k]:+.4g}  single-ratio |Gamma_coax| swapped "
                 f"{cf['single_ratio_abs_coax_refl_swapped'][k]:.4f} (code {cf['single_ratio_abs_coax_refl_code'][k]:.4f})")
    return L


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    return o


def load_npz(path):
    """Load a ``.ladders.npz`` written by the driver into the dict compute_witnesses() takes."""
    z = np.load(path, allow_pickle=False)
    d = {}
    for k in z.files:
        if k.startswith("flux__"):
            continue
        arr = z[k]
        d[k] = arr if arr.ndim else arr.item()
    flux = {}
    for k in z.files:
        if k.startswith("flux__"):
            _, drive, name = k.split("__", 2)
            flux.setdefault(drive, {})[name] = z[k]
    d["flux_by_drive"] = flux or None
    for k in ("coax_ladder_v", "msl_ladder_v"):
        if k not in d:
            d[k] = None
    return d


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", help="<result>.ladders.npz written by the settled-run driver")
    ap.add_argument("--json", default=None, help="write the witness dict here (default: <npz>.witnesses.json)")
    args = ap.parse_args(argv)
    d = load_npz(args.npz)
    w = compute_witnesses(d)
    for line in format_tables(w):
        print(line)
    out = Path(args.json) if args.json else Path(args.npz).with_suffix(".witnesses.json")
    out.write_text(json.dumps(_jsonable(w), indent=2))
    print(f"witness JSON written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
