"""Sub-bin spectral-feature estimators shared by the crossval notch/zero cases.

WHY THIS MODULE EXISTS (issue #812, mechanism P3 "estimator quantization").
--------------------------------------------------------------------------
cv06b and cv07 both located a transmission zero with a bare ``argmin`` over a
coarse frequency sweep, and then judged the result against a percentage
tolerance:

  * cv06b: ``argmin`` on a 63.636 MHz grid (``compute_msl_s_matrix(n_freqs=100)``
    over 0.7-7.0 GHz) = **1.754 %/bin** at the 3.627 GHz notch, judged at 15 %.
  * cv07:  ``argmin`` on a 163.866 MHz grid (``linspace(0.5, 20.0, 120)`` GHz)
    = **2.081 %/bin** at the 7.87 GHz zero, judged against a *declared* 1.0 %
    tolerance -- so the reported deviation could only ever be 0.000 % or
    >= 2.081 %, and the declared threshold was unexercisable in between.

A bin-quantised estimator makes the *reported* deviation a staircase, so a
sub-bin physics error reads exactly 0.00 % and the gate cannot fail for its own
stated reason.  The repair is to resolve better than a bin, then judge against
the refined estimator's own demonstrated precision.

The log-parabolic vertex refinement below is the SAME method already committed
in this repository at
``scripts/diagnostics/build_sheen_lpf_palace_referee.py::_min_in_window`` and
``scripts/diagnostics/build_msl_notch_palace_referee.py::_notch``; this module
factors it out so the crossval scripts and the referee producers cannot drift
apart.  ``refined_extremum`` reproduces the committed referee fixture's
``fdtd_doublet_ghz`` values bit-for-bit (locked by
``tests/test_spectral_feature_estimators.py``).

WHAT THE REFINEMENT ASSUMES, AND WHERE IT STOPS.
------------------------------------------------
A 3-point vertex is exact only if the sampled function is a parabola over the
stencil.  Near an isolated transmission zero displaced off the real frequency
axis by loss/radiation delta, |S21(f)|^2 ~ |A|^2 ((f-f0)^2 + delta^2) is
parabolic in f, and log|S21| is parabolic to the same order for delta not too
small; both estimators were measured on the committed cv07 legs and agree to
<= 0.02 % (see ``docs/design_notes/estimator_resolution_regate.md``).  The
residual error is set by the DEPARTURE from a parabola over one stencil -- for
the cv07 doublet, by the neighbouring zero's shoulder.  It is bounded, not
zero, and the honest way to state it is to MEASURE it, which is what
``half_grid_witness`` does in-run.

None of these functions gates anything.  Thresholds live in the calling case.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "refined_extremum",
    "half_grid_witness",
    "level_crossing",
    "band_at_level",
    "transmission_zeros",
]


def _vertex(y: np.ndarray, i: int) -> float:
    """Sub-bin offset (in bins) of the parabola through y[i-1:i+2]'s minimum.

    Returns 0.0 when the 3-point second difference is non-positive (no
    resolvable minimum) and clamps to +-1 bin so a near-degenerate stencil
    cannot throw the estimate outside its own bracket.
    """
    denom = float(y[i - 1] - 2.0 * y[i] + y[i + 1])
    if denom <= 0.0:
        return 0.0
    d = 0.5 * float(y[i - 1] - y[i + 1]) / denom
    return float(np.clip(d, -1.0, 1.0))


def refined_extremum(freqs, mag, lo=None, hi=None, *, transform="log"):
    """Deepest |S| bin in [lo, hi] plus a sub-bin parabolic vertex refinement.

    ``freqs`` and ``mag`` share units with the returned frequencies (the
    crossval callers pass GHz or Hz, whichever the leg carries).  ``transform``
    selects the domain the parabola is fitted in: ``"log"`` (log|S|, the method
    already committed in the two Palace referee producers) or ``"power"``
    (|S|^2, exactly parabolic near a simple transmission zero).

    Returns a dict with ``bin_f`` (the quantised argmin, i.e. what the old
    gates read), ``refined_f``, ``sub_bin_shift`` in bins, ``depth_db``,
    ``index``, and ``bin_width``.
    """
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(mag, dtype=float)
    if lo is None and hi is None:
        idx = np.arange(f.size)
    else:
        band = np.ones(f.size, dtype=bool)
        if lo is not None:
            band &= f >= lo
        if hi is not None:
            band &= f <= hi
        idx = np.where(band)[0]
        if idx.size == 0:
            idx = np.arange(f.size)
    i = int(idx[int(np.argmin(s[idx]))])
    h = float(f[i + 1] - f[i]) if i + 1 < f.size else float(f[i] - f[i - 1])
    if 0 < i < f.size - 1:
        y = np.log(np.maximum(s, 1e-300)) if transform == "log" else s ** 2
        d = _vertex(y, i)
    else:
        d = 0.0
    return {
        "index": i,
        "bin_f": float(f[i]),
        "refined_f": float(f[i]) + d * h,
        "sub_bin_shift": float(d),
        "depth_db": float(20.0 * np.log10(max(float(s[i]), 1e-300))),
        "bin_width": h,
    }


def half_grid_witness(freqs, mag, lo=None, hi=None, *, transform="log"):
    """In-run proof that the estimate is NOT bin-quantised.

    Split the sweep into its two interleaved half-density sub-grids (even and
    odd bins) and refine the same feature on each.  The two sub-grids are
    disjoint in frequency, so their ARGMIN bins are always at least one
    full-grid bin apart: any estimator that returns a bin centre fails a
    ``spread < 1 full-grid bin`` test BY CONSTRUCTION, while a genuinely
    sub-bin estimator's two answers converge on the same physical feature.

    Returns ``spread`` (max-min of the two refined estimates, same units as
    ``freqs``), ``spread_bins`` (in FULL-grid bins), the two refined values,
    and the same two numbers for the bare argmin as the contrast.
    """
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(mag, dtype=float)
    h = float(f[1] - f[0])
    ref, binned = [], []
    for phase in (0, 1):
        r = refined_extremum(f[phase::2], s[phase::2], lo, hi, transform=transform)
        ref.append(r["refined_f"])
        binned.append(r["bin_f"])
    spread = float(max(ref) - min(ref))
    return {
        "full_bin_width": h,
        "refined": [float(x) for x in ref],
        "binned": [float(x) for x in binned],
        "spread": spread,
        "spread_bins": spread / h,
        "argmin_spread": float(max(binned) - min(binned)),
        "argmin_spread_bins": float(max(binned) - min(binned)) / h,
    }


def level_crossing(freqs, mag, level, *, f_min=None, rising=False):
    """First frequency above ``f_min`` where |S| crosses ``level``, linearly
    interpolated between the two bracketing bins (sub-bin, unlike a bin index).

    ``rising=False`` finds a falling crossing (|S| above -> below).  Returns
    ``None`` if the sweep never crosses.
    """
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(mag, dtype=float)
    for k in range(1, f.size):
        if f_min is not None and f[k] < f_min:
            continue
        a, b = s[k - 1], s[k]
        hit = (a < level <= b) if rising else (a >= level > b)
        if hit and b != a:
            return float(f[k - 1] + (level - a) * (f[k] - f[k - 1]) / (b - a))
    return None


def band_at_level(freqs, mag, level_db, index):
    """Contiguous band around ``index`` where 20log10|S| <= ``level_db``.

    Both edges are located by linear interpolation of the dB curve between the
    bracketing bins, so the width is sub-bin rather than a bin count.  Returns
    ``(f_lo, f_hi, n_bins)``; edges fall back to the sweep ends when the band
    runs off the sweep.
    """
    f = np.asarray(freqs, dtype=float)
    y = 20.0 * np.log10(np.maximum(np.asarray(mag, dtype=float), 1e-300))
    if y[index] > level_db:
        return None
    k = index
    while k > 0 and y[k - 1] <= level_db:
        k -= 1
    m = index
    while m < f.size - 1 and y[m + 1] <= level_db:
        m += 1
    if k == 0:
        f_lo = float(f[0])
    else:
        f_lo = float(f[k - 1] + (level_db - y[k - 1]) * (f[k] - f[k - 1]) / (y[k] - y[k - 1]))
    if m == f.size - 1:
        f_hi = float(f[-1])
    else:
        f_hi = float(f[m] + (level_db - y[m]) * (f[m + 1] - f[m]) / (y[m + 1] - y[m]))
    return f_lo, f_hi, int(m - k + 1)


def transmission_zeros(freqs, mag, lo, hi, *, depth_db_max=-20.0,
                       prominence_db=3.0, transform="log"):
    """Structural transmission zeros of |S21| inside [lo, hi].

    A zero is a local minimum that is (a) deeper than ``depth_db_max`` and
    (b) more prominent than ``prominence_db`` against the shallower of its two
    flanking local maxima.  The prominence test is what separates a structural
    zero from the fine ripple a dense sweep (openEMS ships 801 bins here)
    carries on its shoulders; the depth test is what separates it from
    passband ripple.  Each returned zero carries its sub-bin refined frequency.
    """
    f = np.asarray(freqs, dtype=float)
    y = 20.0 * np.log10(np.maximum(np.asarray(mag, dtype=float), 1e-300))
    band = np.where((f >= lo) & (f <= hi))[0]
    out = []
    for i in band:
        if i == 0 or i == f.size - 1:
            continue
        if not (y[i] < y[i - 1] and y[i] < y[i + 1]):
            continue
        if y[i] > depth_db_max:
            continue
        left = y[i]
        k = i
        while k > 0 and y[k - 1] >= y[k]:
            k -= 1
            left = max(left, y[k])
        right = y[i]
        m = i
        while m < f.size - 1 and y[m + 1] >= y[m]:
            m += 1
            right = max(right, y[m])
        if min(left, right) - y[i] < prominence_db:
            continue
        r = refined_extremum(f, np.asarray(mag, dtype=float),
                             f[max(i - 2, 0)], f[min(i + 2, f.size - 1)],
                             transform=transform)
        out.append({"bin_f": float(f[i]), "refined_f": r["refined_f"],
                    "depth_db": float(y[i]),
                    "prominence_db": float(min(left, right) - y[i])})
    return out
