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


def test_phase_residual_same_order_as_magnitude_residual():
    """Cross-check: phase residual should track the KNOWN magnitude-residual
    story (T2.4: dielectric-contrast/staircasing dominated, not grid
    dispersion), not be a wildly different scale that suggests a separate,
    unexplained error source.

    The committed magnitude MAX_TOL is 0.05 (T2.4); a ~4% amplitude error and
    a several-degree phase error are the same order of magnitude for a
    Yee-discretized/staircased dielectric interface (both trace to the same
    interface-discretization mechanism), so this is a plausibility bound, not
    a tight physical derivation.
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
