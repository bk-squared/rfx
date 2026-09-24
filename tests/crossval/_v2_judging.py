"""The v2 accuracy bar's two judging rules, shared by every cross-validation case.

A case compares rfx's curve with a frozen external record in two steps, and both
steps are stated here once, as small pure functions on arrays (numpy only; no
rfx, no solver):

* :func:`mesh_statement` -- is rfx's own mesh ladder converged?  The judged
  feature's frequency (a notch, a null, a resonance) is taken along the ladder,
  coarse to fine.  A step no larger than ``FLAT_STEP`` of the frequency is flat.
  The ladder is monotone when every non-flat step has the same sign.  A ladder
  with a non-flat reversal, or whose last two rungs differ by ``LADDER_AGREEMENT``
  or more, is "not shown to converge" -- the case then states its numbers and
  xfails; it is not a hard failure.  (PI 2026-09-22 for the last-two-rungs rule,
  PI 2026-09-24 for the flat step and the monotone reading, rule R-a.)

* :func:`aligned_magnitude` -- the magnitude comparison, made after the
  frequency has been judged.  rfx's frequency axis is SCALED by
  ``f_ref_feature / f_our_feature`` so the judged features coincide (a scale
  keeps f = 0 fixed, and it is what a uniform phase-velocity error does; a
  translation is not), rfx's |S| in dB is interpolated onto the reference's
  bins, and ``MAG_BAR_DB`` applies over the bins where BOTH curves are at or
  above ``DEEP_NULL_DB``.  The unaligned maximum is returned beside it, to be
  reported and not judged.  (PI 2026-09-24, rule R-b.)

The frequency bar itself (``FREQ_BAR``) is applied by the case, to the judged
feature, before the magnitude comparison.
"""

from __future__ import annotations

import numpy as np

# The v2 accuracy bar (rfx CLAUDE.md, PI 2026-09-20).
FREQ_BAR = 0.01          # resonances, cutoffs, notches, band edges: within 1 %
MAG_BAR_DB = 2.0         # power and magnitude: within 2 dB
# Approved by the PI with the MSL notch filter's plan (2026-09-22).
LADDER_AGREEMENT = 0.01  # mesh statement: the last two rungs differ by < 1 %
DEEP_NULL_DB = -20.0     # a bin where either curve is below this is judged by
#                          position only, never by magnitude
# Approved by the PI on 2026-09-24 (rule R-a): a ladder step no larger than
# one tenth of the 1 % frequency bar counts as flat, not as a direction.
FLAT_STEP = 0.001

CONVERGED = "converged"
NOT_SHOWN_TO_CONVERGE = "not_shown_to_converge"


def mesh_statement(freqs_hz, rung_labels) -> dict:
    """The mesh statement of one ladder (rule R-a).

    ``freqs_hz`` is the judged feature's frequency at each rung, coarse to
    fine; ``rung_labels`` names the rungs (cell sizes) for the reason text.

    Returns ``steps_hz`` and ``steps_pct`` (rung i to rung i+1, relative to
    rung i), ``flat`` (``|step| <= FLAT_STEP`` of rung i's frequency),
    ``monotone`` (every non-flat step has one sign), ``last_two_pct`` (the
    last two rungs' distance relative to the second-to-last, ``nan`` for one
    rung), ``verdict`` (``converged`` or ``not_shown_to_converge``) and
    ``reason``.
    """
    f = [float(v) for v in freqs_hz]
    labels = [str(v) for v in rung_labels]
    if len(labels) != len(f):
        raise ValueError(f"{len(f)} frequencies but {len(labels)} rung labels")
    if not all(np.isfinite(v) and v > 0.0 for v in f):
        raise ValueError(f"feature frequencies must be finite and positive: {f}")

    steps_hz = [b - a for a, b in zip(f, f[1:])]
    steps_pct = [100.0 * s / a for s, a in zip(steps_hz, f)]
    flat = [abs(s) <= FLAT_STEP * a for s, a in zip(steps_hz, f)]
    signs = {float(np.sign(s)) for s, is_flat in zip(steps_hz, flat) if not is_flat}
    monotone = len(signs) <= 1
    last_two_pct = (100.0 * abs(f[-1] - f[-2]) / f[-2]) if len(f) >= 2 else float("nan")

    ladder = ", ".join(f"{v/1e9:.5f} GHz at {lab}" for v, lab in zip(f, labels))
    step_text = ", ".join(f"{p:+.3f} %" + (" (flat)" if fl else "")
                          for p, fl in zip(steps_pct, flat))
    if len(f) < 2:
        verdict = NOT_SHOWN_TO_CONVERGE
        reason = f"one rung ({ladder}); no step to judge"
    elif not monotone:
        verdict = NOT_SHOWN_TO_CONVERGE
        reason = (f"the feature reverses direction by more than "
                  f"{FLAT_STEP*100:.1f} % along the ladder: {ladder}; steps "
                  f"{step_text}")
    elif not last_two_pct < LADDER_AGREEMENT * 100.0:
        verdict = NOT_SHOWN_TO_CONVERGE
        reason = (f"the last two rungs differ by {last_two_pct:.3f} % "
                  f"(bar {LADDER_AGREEMENT*100:.0f} %): {ladder}; steps "
                  f"{step_text}")
    else:
        verdict = CONVERGED
        reason = (f"every step larger than {FLAT_STEP*100:.1f} % has one sign "
                  f"and the last two rungs differ by {last_two_pct:.3f} % "
                  f"(bar {LADDER_AGREEMENT*100:.0f} %): {ladder}; steps "
                  f"{step_text}")
    return {
        "freqs_hz": f,
        "rung_labels": labels,
        "steps_hz": steps_hz,
        "steps_pct": steps_pct,
        "flat": flat,
        "monotone": monotone,
        "last_two_pct": last_two_pct,
        "verdict": verdict,
        "reason": reason,
    }


def _on_reference_bins(ref_f, ref_db, in_band, our_f_axis, our_db, floor_db) -> dict:
    """rfx's dB curve, on the frequency axis given, sampled at the reference
    bins; the difference and the bins it is judged on."""
    ours = np.interp(ref_f, our_f_axis, our_db, left=np.nan, right=np.nan)
    keep = (in_band & (ref_db >= floor_db) & (ours >= floor_db)
            & np.isfinite(ours))
    delta = ours - ref_db
    if keep.any():
        worst = int(np.argmax(np.where(keep, np.abs(delta), -np.inf)))
        max_abs = float(abs(delta[worst]))
        f_at_max = float(ref_f[worst])
    else:
        worst, max_abs, f_at_max = -1, float("nan"), float("nan")
    return {"ours_db": ours, "delta_db": delta, "compared": keep,
            "n_compared": int(keep.sum()), "max_abs_delta_db": max_abs,
            "worst_index": worst, "f_at_max_hz": f_at_max}


def aligned_magnitude(ref_f, ref_db, our_f, our_db, f_ref_feature, f_our_feature,
                      band, floor_db=DEEP_NULL_DB) -> dict:
    """The magnitude comparison with the judged feature's frequency offset
    taken out (rule R-b).

    ``ref_f``/``ref_db`` and ``our_f``/``our_db`` are the two curves (Hz, dB);
    ``f_ref_feature`` and ``f_our_feature`` the judged feature's frequency on
    each; ``band`` the ``(lo, hi)`` range of reference bins that may be
    compared, inclusive.

    rfx's axis becomes ``our_f * scale`` with ``scale = f_ref_feature /
    f_our_feature``; rfx's dB curve is interpolated onto the reference bins
    (a reference bin outside rfx's scaled sweep is not compared); a bin is
    compared when it is in the band and BOTH curves are at or above
    ``floor_db``.

    Returns ``scale``, ``n_in_band``, ``n_compared``, ``max_abs_delta_db`` and
    ``f_at_max_hz`` (aligned: the value the bar judges), the same three for
    the unaligned axis under ``unaligned_*`` (reported, not judged), and the
    per-bin arrays of both under ``aligned`` and ``unaligned``.
    """
    ref_f = np.asarray(ref_f, dtype=float)
    ref_db = np.asarray(ref_db, dtype=float)
    our_f = np.asarray(our_f, dtype=float)
    our_db = np.asarray(our_db, dtype=float)
    f_ref_feature = float(f_ref_feature)
    f_our_feature = float(f_our_feature)
    if not (np.isfinite(f_ref_feature) and np.isfinite(f_our_feature)
            and f_ref_feature > 0.0 and f_our_feature > 0.0):
        raise ValueError(f"feature frequencies must be finite and positive: "
                         f"{f_ref_feature}, {f_our_feature}")
    if our_f.size < 2 or np.any(np.diff(our_f) <= 0.0):
        raise ValueError("rfx's frequency axis must be strictly increasing")

    lo, hi = band
    in_band = (ref_f >= lo) & (ref_f <= hi)
    scale = f_ref_feature / f_our_feature
    aligned_f = our_f * scale
    aligned = _on_reference_bins(ref_f, ref_db, in_band, aligned_f, our_db, floor_db)
    unaligned = _on_reference_bins(ref_f, ref_db, in_band, our_f, our_db, floor_db)
    return {
        "scale": scale,
        "n_in_band": int(in_band.sum()),
        "n_compared": aligned["n_compared"],
        "max_abs_delta_db": aligned["max_abs_delta_db"],
        "f_at_max_hz": aligned["f_at_max_hz"],
        "unaligned_n_compared": unaligned["n_compared"],
        "unaligned_max_abs_delta_db": unaligned["max_abs_delta_db"],
        "unaligned_f_at_max_hz": unaligned["f_at_max_hz"],
        "in_band": in_band,
        "aligned": aligned,
        "unaligned": unaligned,
    }
