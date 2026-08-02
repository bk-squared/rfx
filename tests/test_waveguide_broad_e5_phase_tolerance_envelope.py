"""Issue #490 Lane 1 -- the broad-E5 PHASE tolerance is a bounded, measured envelope.

Mirrors ``test_waveguide_broad_e5_tolerance_envelope.py`` (T2.4) for the
magnitude tolerance: ``MAX_PHASE_TOL_DEG`` must envelope every committed case
(never fail a validated case) AND not be slack (bounded margin, not a round
number picked to make things pass).

Pure-Python contract (no FDTD).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "waveguide_broad_e5"

sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_band_broad_e5_phase_envelope import MAX_PHASE_TOL_DEG  # type: ignore  # noqa: E402

# Same governance margin ceiling as the magnitude lane's MARGIN_CEIL=1.5.
# 15.0 / 11.99 = 1.25, well inside 1.5.
MARGIN_CEIL = 1.5


def _all_case_phase_diffs() -> list[float]:
    diffs: list[float] = []
    for f in sorted(FIXTURES.glob("waveguide_*_broad_e5_phase_envelope.json")):
        for c in json.loads(f.read_text())["cases"]:
            diffs.append(float(c["max_phase_diff_deg"]))
    return diffs


def test_max_phase_tol_is_a_bounded_measured_envelope():
    diffs = _all_case_phase_diffs()
    assert diffs, "no committed broad-E5 phase cases found"
    worst = max(diffs)
    assert worst <= MAX_PHASE_TOL_DEG, (
        f"MAX_PHASE_TOL_DEG={MAX_PHASE_TOL_DEG} is below the worst committed "
        f"case phase diff {worst:.3f} deg -- it would fail a validated case."
    )
    assert MAX_PHASE_TOL_DEG <= worst * MARGIN_CEIL, (
        f"MAX_PHASE_TOL_DEG={MAX_PHASE_TOL_DEG} exceeds worst case diff "
        f"{worst:.3f} deg x {MARGIN_CEIL} = {worst * MARGIN_CEIL:.3f} deg -- "
        f"the tolerance is slack; re-justify or tighten."
    )


def test_phase_residual_is_far_from_the_old_convention_masking_scale():
    """Tripwire: the worst-case phase residual must sit far below the OLD
    cv11 gate's 60-degree window, which existed specifically to mask a
    convention bug (not a physical residual) rather than measure one.

    M5 (adversarial review of PR #536): an earlier version of this docstring
    additionally claimed phase and magnitude residuals "share one mechanism"
    (interface discretization/staircasing). That claim is UNSUPPORTED and is
    retracted here rather than repeated: measured across the 20 committed
    cases, magnitude diff and phase diff are essentially uncorrelated
    (Pearson r = -0.010) and their worst-case orderings invert (the worst
    magnitude case is WR-15 dx=32um eps_r=4; the worst phase case is WR-28
    dx=100um eps_r=2 -- different cases entirely). The 60-degree tripwire
    below is kept on its own merits (it is far above the measured worst case
    and would catch a reintroduced convention bug), not because of a shared
    mechanism with magnitude.
    """
    diffs = _all_case_phase_diffs()
    worst = max(diffs)
    # 60 deg would suggest a genuinely different (larger) error mechanism is
    # active for phase than for magnitude (cf. the OLD unresolved cv11 gate,
    # which needed 60 deg precisely because it was masking a convention bug,
    # not a physical residual). Our worst case (11.99 deg) sits far below
    # that, consistent with "phase and magnitude residuals share one
    # mechanism", not "phase has its own unexplained larger error".
    assert worst < 60.0, (
        f"worst-case phase diff {worst:.2f} deg approaches the old cv11 "
        f"convention-masking scale (60 deg) -- re-verify the reference-plane "
        f"convention rather than assume it is real physics"
    )
