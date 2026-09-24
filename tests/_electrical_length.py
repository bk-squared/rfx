"""A line's electrical length: the v2 accuracy bar's phase item (PI 2026-09-24).

A lossless line between two reference planes transmits with
``S21 = exp(-j beta L)``. Its phase falls as the frequency rises, by ``2 pi``
times the line's delay per hertz, and ``|S21|`` carries none of it: a 5 %
permittivity rise in the coaxial battery's line lengthens it electrically by
about 2.4 % and leaves ``|S21|`` where it was. So the bar reads the delay
itself. It is the least-squares slope of ``unwrap(angle(S21))`` against
frequency over the bins where ``|S21|`` is above -20 dB, measured between the
planes the extractor references S to, and it has to be within 1 % of the
reference's slope over the same bins. A one-port reads its reflection phase the
same way, over the bins where it reflects. The phase at a single bin is not
judged (``docs/design_notes/chain_closure_contract.md``).

The batteries hold a stored S to its closed form with this; the drift locks in
``tests/locks/`` hold a live S to the stored one.
"""
from __future__ import annotations

import numpy as np

# The bar. Written here, not read from a fixture, so a fixture cannot move the
# bar it is judged by.
ELECTRICAL_LENGTH_FRAC = 0.01

# Below this a curve carries no phase worth reading (the bar's deep-null level).
LEVEL_DB = -20.0


def transmitting_bins(s, level_db: float = LEVEL_DB) -> np.ndarray:
    """The bins where ``|s|`` is above ``level_db``: where its phase is read."""
    mag = np.abs(np.asarray(s))
    return 20.0 * np.log10(np.maximum(mag, 1e-300)) > level_db


def electrical_length_ratio(freqs, s_measured, s_reference, mask) -> float:
    """``slope_measured / slope_reference - 1``, each slope the least-squares
    straight line through ``unwrap(angle(S))`` against frequency over the bins
    in ``mask``. On a line whose reference is ``exp(-j beta L)`` this is its
    electrical-length error.

    ``mask`` has to be one contiguous run of at least three bins. The angle is
    unwrapped along those bins only, and across a gap — the core of a
    transmission zero — the unwrap can take the zero's half-turn either way:
    on the microstrip notch record the band's phase reads -11.06 or -1.8 rad
    depending on it. A gap is refused rather than read.
    """
    f = np.asarray(freqs, dtype=float)
    keep = np.flatnonzero(np.asarray(mask, dtype=bool))
    if keep.size < 3:
        raise ValueError(f"{keep.size} bins in the mask; a slope needs three")
    if np.any(np.diff(keep) != 1):
        raise ValueError("the mask is not one contiguous run of bins: across a gap "
                         "the unwrapped phase can turn a half-turn either way")
    fk = f[keep]
    slopes = [float(np.polyfit(fk, np.unwrap(np.angle(np.asarray(s)[keep])), 1)[0])
              for s in (s_measured, s_reference)]
    if slopes[1] == 0.0:
        raise ValueError("the reference's phase does not change across the bins; "
                         "it has no electrical length to compare")
    return slopes[0] / slopes[1] - 1.0
