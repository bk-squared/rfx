"""Mode-RESOLVED identification of a rectangular patch's ring-down spectrum.

WHY THIS EXISTS (issue #812 audit, cv05/cv15 lane)
--------------------------------------------------
Both patch cross-vals selected their gated resonance as::

    modes.sort(key=lambda m: abs(m.freq - f_analytic));  f_res = modes[0].freq

and then reported ``f_res`` AGAINST that same ``f_analytic``.  That selector is
self-confirming: whichever ring-down mode happens to sit nearest the anchor is
promoted to "the" resonance, so the reported distance is bounded by the mode
spacing rather than by the physics, and a build whose design mode has drifted
far away silently re-anchors onto a DIFFERENT member of the patch's cavity
spectrum.  The anchor cannot be both the selector and the referee.

WHAT REPLACES IT
----------------
The patch is a rectangular cavity with magnetic side walls.  Its declared
geometry fixes a whole SPECTRUM of members, not one number::

    f_mn = (c/2) * hypot( m / (a_eff * sqrt(eps_a)) , n / (b_eff * sqrt(eps_b)) )

with the per-axis Balanis/Hammerstad transmission-line quantities

    eps_eff(width) = (er+1)/2 + (er-1)/2 * (1 + 12 h/width)^(-1/2)
    dL(width)      = 0.412 h (eps_eff+0.3)(width/h+0.264)
                            / [(eps_eff-0.258)(width/h+0.8)]
    a_eff = a + 2 dL(b)   (the a-mode's width is the OTHER in-plane dimension)
    b_eff = b + 2 dL(a)

``(1,0)`` and ``(0,1)`` reproduce each script's own single-mode closed form
exactly; ``(1,1)`` is the standard separable-cavity combination.

Identification then assigns EACH measured ring-down mode to its nearest
declared member and requires

  G1  every measured mode inside the identification span is assigned to a
      declared member within the tolerance, injectively (no two measured
      modes claim the same member, no measured mode is left unexplained);
  G2  the DESIGN member has exactly one mode assigned to it -- that mode,
      and only that mode, is the reported resonance;
  G3  at least one further member that resolves the OTHER in-plane axis is
      identified, so the verdict rests on a mode PAIR rather than a scalar.

A build whose design mode has drifted onto a neighbouring member's territory
now fails G2 by name ("declared member (1,0) has no mode") instead of being
re-anchored and reported as agreement.

THE TOLERANCE IS DERIVED, NOT FITTED
------------------------------------
``identification_tolerance()`` returns the LARGEST relative tolerance for
which "nearest declared member" is still unique for every frequency: for two
adjacent members f1 < f2 the windows [f1/(1+t), f1(1+t)] and [f2/(1+t),
f2(1+t)] stay disjoint iff (1+t)^2 < f2/f1, i.e.

    t < sqrt(min_adjacent(f2/f1)) - 1

so ``tol = sqrt(r_min) - 1`` with ``r_min`` the smallest adjacent member ratio
in band.  It depends ONLY on the declared geometry (patch dimensions, eps_r,
h) through the closed form -- no measured frequency enters it.  It is an
IDENTIFICATION tolerance, deliberately looser than the closed form's own 5-8%
accuracy: passing it is not an accuracy claim, and no accuracy claim may be
read from it.  Failing it means the mode could not be identified at all.

WHAT THIS INSTRUMENT CANNOT SEE (stated up front, measured in the design note)
-----------------------------------------------------------------------------
Every observable here is a RATIO of measured frequencies to declared ones, and
the identification residual of a common-mode dilation of the whole spectrum is
that dilation.  A realization error that scales the entire cavity spectrum by a
single factor -- issue #740's vacuum ground cell is one, measured at
1.068 +/- 0.003 across three members -- moves every member by the same few
percent and is INVISIBLE to any dimensionless spectral test (mode-pair ratio
included).  Such defects are the realized-geometry checks' domain
(``assert_realized_stack``), not this module's.  See
``docs/design_notes/20260901_patch_mode_identification_predeclaration.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

C0_DEFAULT = 2.99792458e8

DEFAULT_ORDERS = ((1, 0), (0, 1), (1, 1), (0, 2), (2, 0))


def microstrip_eps_eff_and_dl(eps_r: float, h: float, width: float):
    """Balanis (Antenna Theory, Ch. 14) per-axis transmission-line quantities.

    ``width`` is the NON-resonant in-plane dimension of the mode in question.
    """
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / width) ** -0.5
    dl = 0.412 * h * ((eps_eff + 0.3) * (width / h + 0.264)) / \
        ((eps_eff - 0.258) * (width / h + 0.8))
    return eps_eff, dl


def declared_cavity_spectrum(eps_r, h, a, b, *, orders=DEFAULT_ORDERS,
                             c0=C0_DEFAULT):
    """Declared TM_mn0 spectrum of an ``a`` x ``b`` patch (metres, Hz).

    ``a`` is the x dimension, ``b`` the y dimension; order ``(m, n)`` counts
    half-wavelengths along x and y.  ``(1, 0)`` is the a-resonant mode and
    reproduces the single-mode Balanis closed form for the a-axis exactly.
    """
    eps_a, dl_a = microstrip_eps_eff_and_dl(eps_r, h, b)
    eps_b, dl_b = microstrip_eps_eff_and_dl(eps_r, h, a)
    a_eff = a + 2 * dl_a
    b_eff = b + 2 * dl_b
    f10 = c0 / (2 * a_eff * math.sqrt(eps_a))
    f01 = c0 / (2 * b_eff * math.sqrt(eps_b))
    return {(m, n): math.hypot(m * f10, n * f01) for (m, n) in orders}


def members_in_band(members, f_lo, f_hi):
    """Declared members whose frequency lies inside the extraction band."""
    return {k: v for k, v in members.items() if f_lo <= v <= f_hi}


def identification_tolerance(members) -> float:
    """Largest relative tolerance keeping "nearest declared member" unique.

    Derived from the declared spectrum ALONE (see module docstring):
    ``tol = sqrt(min adjacent member ratio) - 1``.
    """
    fs = sorted(members.values())
    if len(fs) < 2:
        raise ValueError("identification_tolerance needs >= 2 declared members "
                         "in band; a single member cannot be identified against "
                         "its neighbours")
    r_min = min(f2 / f1 for f1, f2 in zip(fs, fs[1:]))
    return math.sqrt(r_min) - 1.0


@dataclass
class ModeIdentification:
    ok: bool
    tol: float
    span: tuple
    f_design: float | None
    design_order: tuple
    assignments: list = field(default_factory=list)   # (freq, order|None, rel)
    reasons: list = field(default_factory=list)

    def report_lines(self):
        lo, hi = self.span
        out = [f"identification tol = {self.tol * 100:.2f}% (derived: "
               f"sqrt(min adjacent declared ratio) - 1); span "
               f"[{lo / 1e9:.4f}, {hi / 1e9:.4f}] GHz"]
        for f, order, rel in self.assignments:
            tag = f"TM{order[0]}{order[1]}0" if order else "UNIDENTIFIED"
            rel_s = f"{rel * 100:+.2f}%" if rel is not None else "   n/a"
            out.append(f"  f = {f / 1e9:.4f} GHz  ->  {tag:<14} {rel_s}")
        for r in self.reasons:
            out.append(f"  ! {r}")
        return out


def identify_patch_modes(measured_freqs, members, *, design_order=(1, 0),
                         tol=None, require_second_axis=True):
    """Assign measured ring-down frequencies to declared cavity members.

    ``measured_freqs`` -- iterable of Hz (already filtered for Q / amplitude
    by the caller, which is the caller's physics decision, not this
    function's).  ``members`` -- ``{(m, n): f_Hz}`` restricted to the
    extraction band.  Returns a :class:`ModeIdentification`.

    Modes OUTSIDE the identification span (the declared members' own
    frequency range widened by the tolerance) are reported but not gated:
    they belong to higher members the declared set does not model.
    """
    if tol is None:
        tol = identification_tolerance(members)
    fs = sorted(members.values())
    span = (fs[0] / (1 + tol), fs[-1] * (1 + tol))
    log_tol = math.log(1 + tol)

    assignments = []
    reasons = []
    claims = {}
    for f in sorted(float(x) for x in measured_freqs):
        if not (span[0] <= f <= span[1]):
            assignments.append((f, None, None))
            continue
        order, fm = min(members.items(), key=lambda kv: abs(math.log(f / kv[1])))
        rel = f / fm - 1.0
        if abs(math.log(f / fm)) > log_tol:
            assignments.append((f, None, rel))
            reasons.append(
                f"measured mode {f / 1e9:.4f} GHz inside the identification span "
                f"matches NO declared member within {tol * 100:.2f}% (nearest is "
                f"TM{order[0]}{order[1]}0 at {fm / 1e9:.4f} GHz, {rel * 100:+.2f}%)")
            continue
        assignments.append((f, order, rel))
        claims.setdefault(order, []).append(f)

    for order, fl in claims.items():
        if len(fl) > 1:
            reasons.append(
                f"declared member TM{order[0]}{order[1]}0 is claimed by "
                f"{len(fl)} measured modes ({[round(x / 1e9, 4) for x in fl]} GHz) "
                "-- identification is ambiguous, refuse to name a resonance")

    design_hits = claims.get(design_order, [])
    f_design = design_hits[0] if len(design_hits) == 1 else None
    if not design_hits:
        near = [(f, o, r) for f, o, r in assignments if o is not None]
        detail = (", ".join(f"{f / 1e9:.4f} GHz -> TM{o[0]}{o[1]}0" for f, o, _ in near)
                  or "no mode identified at all")
        reasons.append(
            f"declared DESIGN member TM{design_order[0]}{design_order[1]}0 "
            f"({members[design_order] / 1e9:.4f} GHz) has NO measured mode within "
            f"{tol * 100:.2f}% -- the ring-down carries [{detail}]. The design "
            "resonance was not found; it was not merely mis-measured.")

    if require_second_axis:
        other = [o for o in claims if o != design_order and o[1] > 0]
        if not other:
            reasons.append(
                "no identified member resolves the second in-plane axis "
                "(need one of TM01/TM11/TM02) -- the verdict would rest on a "
                "single mode, which is what the anchored selector did")

    return ModeIdentification(ok=not reasons, tol=tol, span=span,
                              f_design=f_design, design_order=design_order,
                              assignments=assignments, reasons=reasons)
