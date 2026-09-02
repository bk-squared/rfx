#!/usr/bin/env python3
"""MSL modal V·I flux oracle — the standing falsifier for the extractor's
core physical claim (issue #520 top item; origin: #525).

PRE-DECLARATION (written before this script was run — R2-tight, one campaign)
-------------------------------------------------------------------------------
The extractor-independent oracle for the #511/#507-corrected modal voltage
and Ampère-loop current is

    Re(V · conj(I)) / flux_spectrum  ==  1

V and I are built from the trace-anchored span (``msl_modal_voltage`` /
``msl_loop_current``, ``rfx/api/_sparams.py`` / ``rfx/sources/msl_port.py``);
Poynting flux is accumulated by a DIFFERENT kernel entirely
(``rfx/probes/probes.py:flux_spectrum``, fed by ``rfx/simulation.py``'s own
H-field spatial-averaging + half-step phase correction — see
``rfx/simulation.py:1470-1523``). Agreement is not tautological: R is not
computed from its own numerator or denominator restated (unlike the
now-deleted convention check this PR replaced, see CONVENTION below). But
code-path separation is not the relevant strength claim, and this script
does not claim "genuine, independent" as if that closed the question --
``scripts/msl_flux_ratio_dof.py`` (the #525/#531 closure artifact) shows R
constrains exactly ONE real degree of freedom of (V, I) -- the product
``Re(V·conj(I))`` -- while the wave split the extractor actually uses
(``a, b = 0.5·(V ± Z0_ref·I)``) and the separately reported analytic Z0
both depend on V and I SEPARATELY. See DOF BLINDNESS below.

PR #516 measured this identity ONLY on the aligned mesh (dx = h_sub/3 =
84.667 µm, h_sub/dx = 3.0000) and got 1.006-1.009 — the EASY case, where the
rounding proxy and the rasterized trace node coincide (the PR's F2 anchoring
fix is measured bit-identical there). The mesh class the fix actually
changes behaviour on — dx = 80 µm, h_sub/dx = 3.175 (the committed
``test_msl_thru_line_passive_gate`` mesh, inside preflight's [0.10, 0.40]
mixed-cell danger zone) — had never had this identity measured; the PR body
said so verbatim ("expected ~1 by construction; not yet measured on this
mesh"). #525 then measured it on an UNTERMINATED single-port-into-CPML
fixture and found the ratio dominated by reactive residue (0.54-0.69, later
also found convention-contaminated by an asymmetric ½ factor — see
CONVENTION below) with 472% plane-to-plane spread at a −122.9 dB settling
witness — i.e. that fixture cannot see a travelling wave and its number is
not a valid oracle measurement. Neither harness was ever committed.

This script is the committed re-measurement #525's own follow-up comment
(2026-08-02 priority bump) specified as the bar for the artifact:
  1. state the convention and ASSERT it (not just comment it) — see
     CONVENTION below and the runtime ``assert`` in ``run_one``;
  2. an admissibility witness BEFORE the ratio — Re(E·H*) vs |E·H*| at each
     plane (a reactive fraction) plus plane-to-plane flux spread — and
     REFUSE to certify the ratio where the witness fails;
  3. a TERMINATED fixture (matched thru, not a bare port into CPML) so the
     plane sees a travelling wave — this script uses the SAME two-port thru
     geometry as ``tests/unit/sparams/test_msl_port_integration.py::
     test_msl_thru_line_passive_gate``, drives only the near port, and
     leaves the far port passively matched (``add_msl_port(..., excite=
     False)`` → resistive Z0 termination via the port's own σ distribution,
     ``rfx/api/__init__.py`` ``add_msl_port`` docstring: "``True`` →
     resistive termination + active source; ``False`` → passive matched
     termination only") — NOT relying on CPML alone to absorb the guided
     mode;
  4. BOTH mesh classes: aligned (dx = h_sub/3, frac(h_sub/dx) = 0) and
     bisecting (dx = 80 µm, frac = 0.175, the MESH-ALIGNMENT RULE danger
     zone — docs/agent-memory/rfx-known-issues.md, "Added 2026-07-30").

CONVENTION (assert, not comment)
---------------------------------
``flux_spectrum()`` returns ``sum(Re(E1·conj(H2) - E2·conj(H1)) * dA)`` with
NO ½ prefactor (``rfx/probes/probes.py`` docstring, verbatim: "∫ Re(E × H*)
· n̂ dA"). The oracle therefore compares ``Re(V·conj(I))`` — ALSO no ½ —
directly against ``flux_spectrum()``, via the single choke point
``_oracle_ratio()``. A "time-averaged power" reading would put ½ on BOTH
sides, which is a mathematical no-op on the RATIO — #525 root-caused the
prior session's factor-of-2 discrepancy to applying ½ to the V·I side only,
while the flux side stayed unhalved.

An earlier version of this script "verified" that by computing the ratio
both ways (no halves; halves on both) and asserting they matched — which is
a TAUTOLOGY (``X/Y == (0.5X)/(0.5Y)`` for any ``X, Y`` whatsoever) and was
removed on review (PR #549), the same defect class PR #531's review blocked
in the sibling ``scripts/msl_flux_ratio_dof.py`` arc. The real guard is
``_assert_half_factor_regression_is_caught()``, run once at the top of
``main()``. Its FIRST assertion — that the real, unmutated
``_oracle_ratio()`` reads 1.0 on an honest synthetic fixture — IS the
guard; everything after checks that the guard has discriminating power. It
plants TWO independent mutations, both routed through the real
``_oracle_ratio()`` call (not a hand-inlined duplicate expression): a ½ on
the ``v`` (numerator) side only, exactly #525's own bug, which must halve
the ratio (1.0 -> 0.5); and a ½ on the ``flux`` (denominator) side only —
the mirror-image bug an unhalved-numerator check alone would miss — which
must double it (1.0 -> 2.0). The second case matters because an unhalved
numerator over a halved denominator is exactly the shape of #525's own
"convention-corrected 2.01-2.02" misdiagnosis: a script-side factor bug
that would surface as a false PHYSICS over-report rather than a caught
regression. A second, independent defensive assertion greps
``flux_spectrum``'s own source for a stray ``0.5 *`` so a convention
change in the library itself cannot silently invalidate this script's
premise.

ADMISSIBILITY WITNESS
----------------------
Before trusting ``Re(V·conj(I)) / flux`` at a plane/frequency cell, this
script reports — computed from the SAME flux-monitor accumulator fields the
oracle divides by, so the witness and the ratio denominator can never
diverge:

    reactive_fraction = 1 − |Re(Σ integrand·dA)| / Σ |integrand|·dA
        (0 = fully travelling/real; 1 = fully reactive/cancelling)
    spread(f) = (max(flux_real) − min(flux_real)) / mean(|flux_real|)
        across the 5 planes, at each frequency.

These are NOT redundant checks of the same thing. ``spread`` follows from
LOSSLESSNESS (conservation of net guided power along the lossless line),
not from travelling-wave purity: a lossless line's net real Poynting flux
through any cross-section is constant WHETHER OR NOT a standing wave is
present (the standing wave lives in the REACTIVE part) — this is directly
measured on a healthy, reciprocating-wave fixture in
``scripts/msl_flux_ratio_dof.py`` ("R is flat because BOTH numerator and
denominator are conserved along a lossless line"). ``reactive_fraction`` is
the leg that actually flags a non-travelling, reactive-dominated plane —
verified empirically (PR #549 review): running this admissibility witness
on a reconstruction of #525's UNTERMINATED single-port fixture CLASS
(#525's own harness was never committed, so this is this script's own
geometry with the far port's termination removed, not a re-run of #525's
literal code) reads ``spread`` in [0.011, 0.067] (i.e. it PASSES the 0.5
threshold below — spread alone would not have refused that fixture) while
``reactive_fraction`` reads [0.963, 0.995] (decisively fails). So the
witness legs are complementary, not interchangeable, and this script's
AND-gate needs both; do not read either one alone as "the" admissibility
check.

A (plane, frequency) cell is ADMISSIBLE only if
``reactive_fraction <= REACTIVE_FRACTION_MAX`` (0.5 — majority-real) AND
``spread <= SPREAD_MAX`` (0.5 — 50%), both pre-declared below. The oracle
ratio is still printed at every cell for transparency, but INADMISSIBLE
cells are excluded from the pass/fail verdict and flagged in the JSON —
reporting a ratio computed from mostly-reactive power as a physics claim is
refused, per #525 comment 2.

DOF BLINDNESS (both witness legs, not just the ratio)
--------------------------------------------------------
``scripts/msl_flux_ratio_dof.py`` proves algebraically AND measures that
``R = Re(V·conj(I)) / flux`` is EXACTLY invariant under the power-preserving
rescale ``(V, I) -> (aV, I/a)`` — ``Re(aV·conj(I/a)) = Re(V·conj(I))``
identically — while the reported analytic Z0 and the wave split the
extractor actually uses move under the SAME rescale. R constrains ONE real
combination of (V, I); the extractor depends on the pair separately. This
oracle's admissibility witnesses do not close that gap — they make it
WORSE, structurally: both ``reactive_fraction`` and ``spread`` are computed
purely from the flux monitor's own ``e1/e2/h1/h2`` fields, never from V or
I at all, so they are ALSO exactly invariant under a ``(V, I) -> (aV, I/a)``
defect — flux does not see V or I, so nothing about V or I individually can
move a witness that never reads them. A defect that moves V and I in
exactly that opposing pattern (constant product) would pass this script's
ratio check AND both admissibility witnesses while corrupting every S value
downstream. This is stated as a known, structural blind direction, not
closed by this script; closing it needs a check on V and I SEPARATELY (the
``v_abs``/``i_abs`` fields recorded per cell in the committed JSON exist
for exactly that future purpose — e.g. against a closed-form Z0 on a
matched line).

FIXTURE
-------
Identical materials/geometry to the committed gate fixture
(``tests/unit/sparams/test_msl_port_integration.py``: EPS_R=3.66, H_SUB=254 µm,
W_TRACE=600 µm, L_LINE=10 mm, one-cell PEC trace, PEC ground + CPML), TWO
MSL ports (near driven ``+x``, far ``excite=False`` matched ``-x``), 5
Ez/Hy/Hz DFT-plane probes + flux monitors placed at 30/40/50/60/70% of
L_LINE — well clear of both the source-launch fringing zone (~5·h_sub ≈
1.27 mm) and the passive-port termination near field, on EITHER end.

PRE-DECLARED EXPECTATION (falsifiable, not adjusted after the run)
--------------------------------------------------------------------
On EVERY (plane, frequency) cell judged ADMISSIBLE by the witness above, on
BOTH mesh classes:

    0.85 <= Re(V·conj(I)) / flux_spectrum <= 1.15

(a wider band than PR #516's 1.006-1.009 aligned-only number, because this
is a fresh fixture/probe placement, not a replication of that run.)
FALSIFIED if any admissible cell falls outside that band on EITHER mesh
class — STOP and report the per-plane dump; a bisecting-mesh failure would
be a FINDING (the extractor is imperfect there, consistent with #525's
prior unterminated-fixture read of 0.54-0.69), not a script defect,
PROVIDED the admissibility witness passed at that cell. If fewer than half
of a mesh's (plane, frequency) cells are admissible, that mesh's identity
check is INCONCLUSIVE (not a pass or a fail) and is reported as such — that
would be a finding about this fixture's own regime, not a verdict on the
extractor.

RUNTIME
-------
2 mesh points x 1 FDTD run each (near port driven, far port passive;
n_freqs=6, num_periods=30). ~2-5 min/point on one CPU core.

    python scripts/diagnostics/msl_vi_flux_oracle.py

POST-RUN REVIEW NOTE (2026-08-04, appended after the run; not part of the
pre-declaration -- the committed JSON is the AS-RUN record, untouched)
--------------------------------------------------------------------------
OVERALL: HELD. Both mesh classes admitted all 30/30 (plane, frequency)
cells (reactive_fraction and spread both far inside the pre-declared
bounds on every cell -- the matched-thru fixture is nowhere near the
472%-spread, reactive-dominated regime #525 measured on the single-port
fixture) and the ratio landed inside the pre-declared [0.85, 1.15] band on
every admissible cell of both meshes:

    aligned  h_sub/3 (dx=84.667um, frac=0.0000): ratio 1.0083 .. 1.0090
    bisecting 80um   (dx=80.000um, frac=0.1750): ratio 1.0105 .. 1.0118

This is the first-COMMITTED measurement of the bisecting-mesh identity (PR
#516 shipped only the aligned-mesh number; #525 measured the bisecting mesh
twice in an uncommitted /tmp harness -- see CORRECTION below, not
"first-ever"). It does NOT reproduce #525's 0.54-0.69 as-posted read on the
bisecting class -- that number came from an unterminated
single-port-into-CPML fixture COMBINED with #525's own follow-up finding of
an asymmetric 1/2-factor convention bug in that earlier, never-committed
script (see CORRECTION below for the precise provenance and the
convention-corrected comparison -- an earlier version of this note
mis-attributed the 472%-spread/-122.9dB figures to the bisecting mesh --
they are the ALIGNED mesh's). With a properly terminated fixture (far port
`excite=False` -> resistive Z0 termination, not bare CPML absorption) and
the symmetric no-half convention asserted here, the extractor-independent
flux oracle holds tightly on BOTH mesh classes. This does not relitigate
#525's own numbers (that script was never committed and is not re-run
here); it establishes what the committed, falsifiable artifact reads on a
fixture built to satisfy the admissibility bar #525 itself set.

CORRECTION (PR #549 review, same day -- dated append, plus in-place fixes
to non-falsifiable explanatory prose only; every DECIDABLE pre-declaration
element -- the 0.85-1.15 band, the two mesh classes, the fixture, the
REACTIVE_FRACTION_MAX/SPREAD_MAX thresholds -- is untouched)
--------------------------------------------------------------------------
Five items from review (the reviewer reproduced this script's committed
numbers byte-for-byte, then ran an independent unterminated-fixture control
to check the provenance claims) -- two factual errors (1, 2), one code
defect (3), and two additions (4, 5):

1. PROVENANCE (MAJOR). The "472% plane-to-plane spread at a -122.9 dB
   settling witness" figure quoted in the PRE-DECLARATION above is #525's
   own measurement on the ALIGNED mesh (dx=84.67um), not the bisecting one
   -- #525's own comment states it verbatim: "at dx = 84.67 µm the net flux
   is ~1e-30 W with 472% / 147% / 304% plane-to-plane spread and random
   sign, on a record settled to −122.9 dB". #525's own EARLIER "Probe 1"
   comment independently confirms the bisecting column was the well-behaved
   one: "The dx=80 column is smooth, monotone in k_top, and consistent
   across frequency ... The dx=84.67 column is sign-flipping across
   frequency ... It is the signature of a contaminated denominator." So the
   reactive/spread pathology in #525's own record was an ALIGNED-mesh
   phenomenon, and the PRE-DECLARATION's phrasing ("that fixture cannot see
   a travelling wave") was correct about the OLD fixture in general but
   wrongly localized the specific 472%/-122.9dB evidence to the bisecting
   mesh discussion it appears next to.

   Second: the ADMISSIBILITY WITNESS leg that actually refuses #525's old
   UNTERMINATED fixture is REACTIVE_FRACTION, not spread. Reviewer's
   reproduction of this script's own witness computation on the same
   fixture CLASS, reconstructed (this script's geometry, far port
   termination removed -- #525's own harness was never committed, so this
   is not a re-run of its literal code): spread in [0.011, 0.067] (PASSES
   this script's 0.5 threshold -- spread alone would not have refused it)
   while reactive_fraction reads [0.963, 0.995] (decisively fails). The
   ADMISSIBILITY WITNESS section above has been corrected in place to
   state this (spread measures losslessness/conservation, not
   travelling-wave purity; reactive_fraction is the leg that catches a
   reactive, non-travelling plane).

2. COMPARISON TO #525's CORRECTED NUMBER (MAJOR). #525 comment 2 itself
   supplies the convention-corrected bisecting figure: the as-posted
   0.54-0.69 read, once the asymmetric-1/2 bug is removed (both sides
   un-halfed, doubling the ratio), becomes 1.20-1.34 -- table quoted
   verbatim from #525: "dx = 80 µm (bisecting) | 0.54 – 0.69 | 1.20 – 1.34".
   Review's own broader reconstruction across #525's full recorded data
   reads 1.076-1.387, and review's independent measurement on the same
   unterminated fixture CLASS, reconstructed (spread/reactive numbers in
   item 1 above) measured ratio 1.0898-1.3867 -- matching #525's corrected
   figure to ~1%. So
   #525's bisecting-mesh measurement is fully reproducible, and the
   original 0.54-0.69 headline is jointly explained by (a) the 1/2-factor
   convention bug and (b) the reactive, non-admissible fixture -- NOT by an
   extractor defect on the bisecting mesh. This script's own committed
   bisecting number (1.0105-1.0118, terminated fixture) is a DIFFERENT,
   cleaner measurement, not a replication of #525's fixture, and should not
   be read as "explaining away" #525's number by itself -- the
   reconciliation above is what does that.

   Also new: this script's ALIGNED-mesh number (1.0083-1.0090) resolves a
   question #525 comment 2 explicitly left open -- "PR #516's '1.006–1.009
   ⇒ the corrected V span is right' is only sound if that measurement
   paired ½ with ½ — which I cannot check, because its harness was never
   committed." This script's aligned measurement, built with the
   convention explicitly stated and asserted (not assumed), lands at
   1.0083-1.0090 -- matching PR #516's original 1.006-1.009 closely. That
   resolves #525's open doubt IN PR #516's FAVOUR: PR #516's original
   aligned-mesh measurement did use the correctly-paired convention: it was
   not a coincidence that it read close to 1.

3. THE TAUTOLOGICAL CONVENTION ASSERT (MAJOR). The CONVENTION section
   above has been corrected in place: this script no longer asserts
   ``p_vi/flux == (0.5*p_vi)/(0.5*flux)`` (true for any fixture, healthy or
   broken -- the exact defect class PR #531's review blocked in
   ``scripts/msl_flux_ratio_dof.py``). Removed; replaced by
   ``_assert_half_factor_regression_is_caught()``.

   Strengthened in a second review pass (same PR): the first version of the
   replacement still had two gaps -- the mutation was hand-inlined
   (``0.5 * np.real(v * np.conj(i))``) rather than routed through
   :func:`_oracle_ratio` itself, so the docstring's claim of exercising the
   real code path was not yet true; and only the numerator side was
   mutated, leaving a denominator-side (``flux``) half uncaught -- exactly
   the shape of #525's own "convention-corrected 2.01-2.02" misdiagnosis,
   where a script-side factor bug would surface as a false physics
   over-report instead of a caught regression. Both fixed: the guard now
   plants the numerator mutation via ``_oracle_ratio(0.5*v, i, flux)`` and
   the denominator mutation via ``_oracle_ratio(v, i, 0.5*flux)``, both
   through the real function, requiring the ratio to move to 0.5 and 2.0
   respectively.

4. THE FREE PROXY-SPAN MUTATION TWIN (MODERATE). The committed JSON now
   carries, per plane per frequency, ``ratio_proxy_span`` -- the retired
   ``round(h_sub/dx)`` (pre-#511/F2) span applied to the SAME
   already-recorded fields, zero additional FDTD -- plus a per-mesh
   ``proxy_ratio_range`` / ``proxy_span_falls_outside_band`` summary. On
   the aligned mesh the proxy span equals the corrected span by
   construction (k_top == trace_lo there), so this is an honest null
   result. On the bisecting mesh it demonstrably falls outside
   [0.85, 1.15] -- see the committed JSON for this run's exact numbers;
   the oracle can tell the retired span from the corrected one.

5. DOF BLINDNESS (MODERATE). Addressed in place above (new DOF BLINDNESS
   subsection): the admissibility witnesses are not tautological with the
   ratio's own numerator (they read the flux monitor's e1/e2/h1/h2 fields
   directly, never V or I), but they share the ratio's blindness to a
   product-preserving ``(V, I) -> (aV, I/a)`` defect, because neither
   witness reads V or I at all. ``|V|``/``|I|`` are now recorded per cell
   in the JSON (``v_abs``/``i_abs``) for a future check that closes this
   direction; this script does not close it.

Not relitigated here: #525's own numbers and its uncommitted harness are
not re-run by this script (per its own recommendation, item 3: "commit the
harness"). This script IS that committed harness, built to satisfy the bar
#525 set, and the reconciliation above is arithmetic on #525's OWN
published numbers plus the reviewer's independent reproduction -- not a new
FDTD investigation into #525's fixture.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rfx import Box, Simulation  # noqa: E402
from rfx.api._sparams import _msl_cell_profile, msl_modal_voltage  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.probes.probes import flux_spectrum  # noqa: E402
from rfx.sources.msl_port import MSLPort, _msl_yz_cells, msl_loop_current  # noqa: E402

# --- gate-fixture geometry, verbatim from tests/unit/sparams/test_msl_port_integration.py ---
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3
F_MAX = 5e9

FREQS = np.array([2.0e9, 2.5e9, 3.0e9, 3.5e9, 4.0e9, 4.5e9])
PLANE_FRACS = [0.30, 0.40, 0.50, 0.60, 0.70]
NUM_PERIODS = 30.0

REACTIVE_FRACTION_MAX = 0.5
SPREAD_MAX = 0.5
RATIO_LO, RATIO_HI = 0.85, 1.15

OUT_DIR = REPO / "scripts" / "diagnostics" / "msl_vi_flux_oracle"

MESH_CLASSES = [
    ("aligned h_sub/3", H_SUB / 3.0),
    ("bisecting 80um (gate mesh)", 80e-6),
]


def _assert_flux_convention_unchanged() -> None:
    """Defensive tripwire: fail loudly if ``flux_spectrum`` grows a stray
    ½ prefactor this script's convention assumption did not anticipate."""
    src = inspect.getsource(flux_spectrum)
    body = src.split("def flux_spectrum", 1)[1]
    assert "0.5 *" not in body and "* 0.5" not in body, (
        "flux_spectrum() source now contains a 0.5 factor — the no-half "
        "convention this script asserts against has changed upstream; "
        "re-derive the convention before trusting any ratio below."
    )


def _oracle_ratio(v, i, flux):
    """The oracle's core arithmetic, and its ONLY choke point: no half
    anywhere, on either side. ``run_one`` calls this rather than inlining
    the expression so the anti-regression guard below tests the same code
    path it is guarding, not a duplicate that could silently drift from
    it.

    Returns ``(p_vi, ratio)`` with ``p_vi = Re(V·conj(I))`` and
    ``ratio = p_vi / flux``.
    """
    p_vi = np.real(v * np.conj(i))
    ratio = p_vi / np.where(np.abs(flux) > 1e-300, flux, np.nan)
    return p_vi, ratio


def _assert_half_factor_regression_is_caught() -> None:
    """Anti-regression guard for the #525 / PR #531 factor-of-2 class.

    A prior version of this script asserted ``p_vi/flux == (0.5*p_vi)/(0.5*
    flux)`` — TAUTOLOGICALLY true for any V, I, flux whatsoever, since
    multiplying both the numerator and the denominator of a ratio by the
    same constant never changes it. PR #549 review (correctly) named this
    the exact defect class PR #531's review blocked in the sibling
    ``scripts/msl_flux_ratio_dof.py`` arc: a "falsifier" whose alternative
    outcome is algebraically unreachable is not a falsifier.

    THE ACTUAL GUARD is assertion 1 below: the real, unmutated
    :func:`_oracle_ratio` — the single choke point ``run_one`` calls — must
    read 1.0 on a fixture constructed so that is the honest answer. That is
    what "no half anywhere" MEANS operationally; everything after it is a
    discriminating-power check on the guard itself, not a second guard.

    Two mutations, BOTH routed through the real :func:`_oracle_ratio` (not
    a hand-inlined duplicate expression, which would test nothing about the
    actual code path):

    * NUMERATOR half (#525's own bug — a time-averaged
      ``0.5·Re(V·conj(I))`` divided by the un-averaged ``flux_spectrum``):
      call ``_oracle_ratio(0.5*v, i, flux)``. Since the function's
      numerator is ``Re(v·conj(i))``, halving ``v`` halves it — this
      exercises the function itself, not a copy of its formula.
    * DENOMINATOR half — the mirror-image bug (e.g. a call site dividing by
      ``0.5 * flux_spectrum(...)``): call ``_oracle_ratio(v, i, 0.5*flux)``.
      Mutation A alone does NOT cover this; an unhalved numerator over a
      halved flux reads ratio ≈ 2.0, which would surface downstream as a
      "physics over-report" finding rather than a caught script defect —
      exactly the shape of #525's own "convention-corrected 2.01-2.02"
      misdiagnosis class (see PRE-DECLARATION above). Both directions must
      be caught here, not discovered as a false physics claim later.
    """
    v = np.array([1.0 + 0.3j, 0.7 - 0.4j])
    i = np.array([0.9 - 0.2j, 0.5 + 0.1j])
    flux = np.real(v * np.conj(i))  # constructed so the healthy ratio is 1

    # THE ACTUAL GUARD: the real, unmutated _oracle_ratio must read 1.0.
    _, ratio = _oracle_ratio(v, i, flux)
    assert np.allclose(ratio, 1.0), (
        f"self-test fixture is not honest (expected ratio 1.0): {ratio}"
    )

    # Mutation A — numerator-side half, through the real function.
    _, ratio_num_half = _oracle_ratio(0.5 * v, i, flux)
    assert np.allclose(ratio_num_half, 0.5), (
        f"planting a half on the numerator (v) side should halve the "
        f"ratio; got {ratio_num_half} — this fixture no longer "
        "discriminates the bug"
    )
    assert not np.allclose(ratio_num_half, ratio), (
        "mutation A did not move the ratio relative to the healthy "
        "computation — it is not discriminating"
    )

    # Mutation B — denominator-side (flux) half, through the real function.
    _, ratio_den_half = _oracle_ratio(v, i, 0.5 * flux)
    assert np.allclose(ratio_den_half, 2.0), (
        f"planting a half on the flux (denominator) side should double "
        f"the ratio; got {ratio_den_half} — this fixture no longer "
        "discriminates the bug"
    )
    assert not np.allclose(ratio_den_half, ratio), (
        "mutation B did not move the ratio relative to the healthy "
        "computation — it is not discriminating"
    )


def run_one(label: str, dx: float) -> dict:
    lx = L_LINE + 2 * PORT_MARGIN
    ly = W_TRACE + 2 * (2 * H_SUB + 8 * dx)
    lz = H_SUB + 1.5e-3
    y_c = ly / 2.0

    sim = Simulation(
        freq_max=F_MAX, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="ro4350b")
    sim.add(Box((0.0, y_c - W_TRACE / 2.0, H_SUB),
                (lx, y_c + W_TRACE / 2.0, H_SUB + dx)), material="pec")
    # Near port: driven. Far port: excite=False -> passive matched
    # termination (resistive sigma at Z0, not bare CPML absorption) so the
    # planes between the two ports see a travelling wave, per #525 comment 2.
    sim.add_msl_port(position=(PORT_MARGIN, y_c, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_c, 0.0), width=W_TRACE,
                     height=H_SUB, direction="-x", impedance=50.0,
                     excite=False)

    plane_x = [PORT_MARGIN + f * L_LINE for f in PLANE_FRACS]
    fj = jnp.asarray(FREQS)
    for idx, x in enumerate(plane_x):
        for comp in ("ez", "hy", "hz"):
            sim.add_dft_plane_probe(axis="x", coordinate=float(x),
                                    component=comp, freqs=fj,
                                    name=f"p{idx}_{comp}")
        sim.add_flux_monitor(axis="x", coordinate=float(x), freqs=fj,
                             name=f"p{idx}_flux")
        # Settling-witness point probe at the same plane, mid-substrate
        # under the trace (project rule: quote end/peak energy for any
        # fixed-length open-domain claims-bearing record).
        sim.add_probe(position=(float(x), y_c, H_SUB / 2.0), component="ez")

    grid = sim._build_grid()
    mp0 = MSLPort(feed_x=PORT_MARGIN, y_lo=y_c - W_TRACE / 2.0,
                  y_hi=y_c + W_TRACE / 2.0, z_lo=0.0, z_hi=H_SUB,
                  direction="+x", impedance=50.0, excitation=None)
    cells = _msl_yz_cells(grid, mp0)
    j_set = sorted({c[1] for c in cells})
    k_set = sorted({c[2] for c in cells})
    j_lo, j_hi = j_set[0], j_set[-1]
    k_lo, k_top = k_set[0], k_set[-1]
    j_c = (j_lo + j_hi) // 2

    pec_mask = np.asarray(sim._assemble_materials(grid)[3])
    col = pec_mask[cells[0][0], j_c, k_top:]
    kp = np.where(col)[0]
    if kp.size == 0:
        raise RuntimeError(f"{label}: no PEC trace conductor found above "
                           "the substrate top — cannot locate trace node.")
    trace_lo = int(k_top + kp.min())
    trace_hi = int(k_top + kp.max())

    n_sub_exact = H_SUB / dx
    frac = n_sub_exact - int(n_sub_exact)

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.run(num_periods=NUM_PERIODS, compute_s_params=False)
    dt = time.time() - t0
    # Filter the generic "x64 disabled" JAX dtype-fallback spam (fires on
    # every accumulator allocation in this codebase when x64 is off, which
    # this script deliberately leaves off -- repo rule forbids module-level
    # x64) -- it is not a preflight/physics message and duplicates dozens of
    # times per run. Genuine preflight/physics warnings are kept verbatim.
    preflight_warnings = [
        str(w.message) for w in caught
        if "JAX_ENABLE_X64" not in str(w.message)
        and "current-gotchas" not in str(w.message)
    ]

    dy_arr = _msl_cell_profile(grid, "y", grid.ny)
    dz_arr = _msl_cell_profile(grid, "z", grid.nz)
    hs_phase = np.exp(1j * 2.0 * np.pi * FREQS * float(grid.dt) * 0.5)

    # Settling witness at every plane probe: PEAK-of-tail vs peak Ez^2, dB
    # (not mean-of-tail -- mean is the more lenient of the two, ~5.6 dB so
    # on the sibling msl_flux_ratio_dof.py fixture, PR #531 review; this is
    # the label's own convention, applied honestly rather than relabelled).
    ts = np.asarray(res.time_series, dtype=float)
    settling_db = []
    for idx in range(len(plane_x)):
        p = ts[:, idx] ** 2
        tail = max(1, p.shape[0] // 10)
        end = float(p[-tail:].max())
        peak = float(p.max())
        tiny = np.finfo(float).tiny
        settling_db.append(10.0 * np.log10((end + tiny) / (peak + tiny)))

    planes = res.dft_planes
    flux_mons = res.flux_monitors

    per_plane = []
    flux_real_by_plane = []  # (n_planes, n_freqs) for the spread witness
    for idx in range(len(plane_x)):
        ez_plane = np.asarray(planes[f"p{idx}_ez"].accumulator)
        hy_plane = np.asarray(planes[f"p{idx}_hy"].accumulator) * hs_phase[:, None, None]
        hz_plane = np.asarray(planes[f"p{idx}_hz"].accumulator) * hs_phase[:, None, None]

        v = np.asarray(msl_modal_voltage(
            jnp.asarray(ez_plane), j_centre=j_c, k_lo=k_lo, k_hi=trace_lo,
            dz_arr=dz_arr))
        i = np.asarray(msl_loop_current(
            jnp.asarray(hy_plane), jnp.asarray(hz_plane),
            j_lo=j_lo, j_hi=j_hi, k_trace_lo=trace_lo, k_trace_hi=trace_hi,
            dy_arr=dy_arr, dz_arr=dz_arr, direction="+x"))

        mon = flux_mons[f"p{idx}_flux"]
        flux_lib = np.real(np.asarray(flux_spectrum(mon)))

        e1 = np.asarray(mon.e1_dft); e2 = np.asarray(mon.e2_dft)
        h1 = np.asarray(mon.h1_dft); h2 = np.asarray(mon.h2_dft)
        dA = np.asarray(mon.dA)
        integrand = e1 * np.conj(h2) - e2 * np.conj(h1)
        flux_real_hand = np.real(np.sum(integrand * dA, axis=(-2, -1)))
        # Normalisation witness: the hand recompute from the monitor's own
        # accumulator fields must match the library call bit-for-bit (same
        # formula) -- this pins that reading mon.e1_dft/e2_dft/h1_dft/h2_dft
        # directly (for the admissibility witness) is not a second, silently
        # different convention from flux_spectrum() itself.
        assert np.allclose(flux_real_hand, flux_lib, rtol=1e-6, atol=1e-30), (
            f"{label} plane {idx}: hand Poynting recompute from mon.*_dft "
            f"disagrees with flux_spectrum() -- convention mismatch, not a "
            f"physics finding. hand={flux_real_hand} lib={flux_lib}"
        )
        flux_mag = np.sum(np.abs(integrand) * dA, axis=(-2, -1))
        reactive_fraction = 1.0 - np.abs(flux_real_hand) / np.where(
            flux_mag > 0, flux_mag, np.nan)

        p_vi, ratio_no_half = _oracle_ratio(v, i, flux_lib)

        # Free mutation twin (#520 leg-1 MODERATE 1, PR #549 review): the
        # RETIRED round(h_sub/dx) proxy span (k_hi=k_top, pre-#511/F2),
        # recomputed from the SAME already-recorded ez_plane/i of this run
        # -- zero additional FDTD. On the aligned mesh k_top == trace_lo so
        # this is a no-op (matches PR #516's own bit-identical finding); on
        # the bisecting mesh it drops the in-phase edge 3 and should read
        # far outside the admissible band, demonstrating the oracle CAN
        # tell the retired span from the corrected one.
        v_proxy = np.asarray(msl_modal_voltage(
            jnp.asarray(ez_plane), j_centre=j_c, k_lo=k_lo, k_hi=k_top,
            dz_arr=dz_arr))
        _, ratio_proxy = _oracle_ratio(v_proxy, i, flux_lib)

        flux_real_by_plane.append(flux_real_hand)
        per_plane.append(dict(
            plane_idx=idx, x_frac=PLANE_FRACS[idx],
            p_vi_real=[float(x) for x in p_vi],
            flux_real=[float(x) for x in flux_real_hand],
            reactive_fraction=[float(x) for x in reactive_fraction],
            ratio=[float(x) for x in ratio_no_half],
            ratio_proxy_span=[float(x) for x in ratio_proxy],
            # |V|, |I| per cell (PR #549 review MODERATE 2): scripts/
            # msl_flux_ratio_dof.py shows R = Re(V.conj(I))/flux is exactly
            # invariant under the power-preserving rescale (V,I) ->
            # (aV, I/a) -- and so, by construction, are reactive_fraction
            # and spread above (both built from flux's own e1/e2/h1/h2
            # fields, never from V or I). Recording |V|,|I| here is what a
            # future re-run needs to additionally check the pair
            # SEPARATELY (e.g. against a closed-form Z0) and close that
            # blind direction; this script does not close it.
            v_abs=[float(x) for x in np.abs(v)],
            i_abs=[float(x) for x in np.abs(i)],
        ))

    flux_stack = np.array(flux_real_by_plane)  # (n_planes, n_freqs)
    spread_per_freq = (
        (flux_stack.max(axis=0) - flux_stack.min(axis=0))
        / np.where(np.abs(flux_stack).mean(axis=0) > 0,
                  np.abs(flux_stack).mean(axis=0), np.nan)
    )

    admissible_cells = []
    all_cells = []
    all_ratio_proxy = []
    for idx, row in enumerate(per_plane):
        for f_idx in range(len(FREQS)):
            rf = row["reactive_fraction"][f_idx]
            sp = float(spread_per_freq[f_idx])
            ratio = row["ratio"][f_idx]
            ratio_proxy = row["ratio_proxy_span"][f_idx]
            all_ratio_proxy.append(ratio_proxy)
            ok = (np.isfinite(rf) and rf <= REACTIVE_FRACTION_MAX
                 and np.isfinite(sp) and sp <= SPREAD_MAX)
            cell = dict(plane_idx=idx, freq_ghz=float(FREQS[f_idx] / 1e9),
                       reactive_fraction=rf, spread=sp, ratio=ratio,
                       ratio_proxy_span=ratio_proxy, admissible=bool(ok))
            all_cells.append(cell)
            if ok:
                admissible_cells.append(cell)

    admissible_ratios = [c["ratio"] for c in admissible_cells]
    within_band = [RATIO_LO <= r <= RATIO_HI for r in admissible_ratios]
    inconclusive = len(admissible_cells) < 0.5 * len(all_cells)
    verdict = (
        "INCONCLUSIVE" if inconclusive else
        ("HELD" if admissible_ratios and all(within_band) else "FALSIFIED")
    )

    # Free mutation twin verdict (#520 leg-1 MODERATE 1): does the oracle
    # actually distinguish the retired proxy span from the corrected one?
    # On the aligned mesh (k_top == trace_lo) it CANNOT by construction --
    # that is the expected, honest null result there.
    proxy_ratio_range = [round(min(all_ratio_proxy), 4), round(max(all_ratio_proxy), 4)]
    proxy_falls_outside_band = any(
        r < RATIO_LO or r > RATIO_HI for r in all_ratio_proxy
    )

    return dict(
        label=label, dx_um=round(dx * 1e6, 4),
        n_sub_exact=round(n_sub_exact, 4), frac=round(frac, 4),
        mixed_cell_danger_zone=bool(0.10 <= frac <= 0.40),
        trace_node=trace_lo, k_lo=k_lo, k_top_proxy=k_top,
        wallclock_s=round(dt, 1),
        preflight_warnings=preflight_warnings,
        settling_db=[round(x, 1) for x in settling_db],
        spread_per_freq=[round(float(x), 4) for x in spread_per_freq],
        per_plane=per_plane,
        cells=all_cells,
        n_admissible=len(admissible_cells),
        n_total=len(all_cells),
        admissible_ratio_range=(
            [round(min(admissible_ratios), 4), round(max(admissible_ratios), 4)]
            if admissible_ratios else None
        ),
        proxy_ratio_range=proxy_ratio_range,
        proxy_span_falls_outside_band=bool(proxy_falls_outside_band),
        verdict=verdict,
    )


def main() -> int:
    _assert_flux_convention_unchanged()
    _assert_half_factor_regression_is_caught()

    rows = []
    for label, dx in MESH_CLASSES:
        print(f"\n=== {label} (dx={dx * 1e6:.3f} um) ===", flush=True)
        row = run_one(label, dx)
        rows.append(row)
        print(f"  h_sub/dx={row['n_sub_exact']:.4f} frac={row['frac']:.4f} "
              f"trace_node={row['trace_node']} (proxy k_top={row['k_top_proxy']})")
        print(f"  settling_db per plane: {row['settling_db']}")
        print(f"  spread_per_freq: {row['spread_per_freq']}")
        print(f"  admissible cells: {row['n_admissible']}/{row['n_total']}")
        print(f"  admissible ratio range: {row['admissible_ratio_range']}")
        print(f"  retired-proxy-span ratio range (free mutation twin, zero "
              f"new FDTD): {row['proxy_ratio_range']} "
              f"{'OUTSIDE' if row['proxy_span_falls_outside_band'] else 'inside'} "
              f"[{RATIO_LO}, {RATIO_HI}]")
        print(f"  VERDICT: {row['verdict']}")
        print("  --- preflight / warnings (verbatim) ---")
        for w in row["preflight_warnings"]:
            print(f"    {w}")

    overall = "HELD" if all(r["verdict"] == "HELD" for r in rows) else (
        "INCONCLUSIVE" if any(r["verdict"] == "INCONCLUSIVE" for r in rows)
        and all(r["verdict"] in ("HELD", "INCONCLUSIVE") for r in rows)
        else "FALSIFIED"
    )
    print(f"\nOVERALL: {overall}")
    if overall == "FALSIFIED":
        print("STOP: the flux identity failed on at least one admissible "
              "cell of at least one mesh class. Per-plane dumps are in the "
              "committed JSON below -- this is a finding, report it, do not "
              "adjust the pre-declared band.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "msl_vi_flux_oracle.json"
    out_path.write_text(json.dumps({
        "reactive_fraction_max": REACTIVE_FRACTION_MAX,
        "spread_max": SPREAD_MAX,
        "ratio_band": [RATIO_LO, RATIO_HI],
        "rows": rows,
        "overall": overall,
    }, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)
    return 0 if overall in ("HELD", "INCONCLUSIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
