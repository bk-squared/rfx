"""Issue #490 Lane 3 -- the near-cutoff group-delay tolerance is a bounded,
measured envelope, not a tautology.

Adversarial review of PR #536 (finding H1) caught that the original
``build_waveguide_group_delay_envelope.py`` computed ``tol_ns = max(max_diff_ns
* 1.3, 0.02)`` FROM the residual it then gated -- a test asserting
``x <= 1.3x`` from one fixture is vacuous. This mirrors
``test_waveguide_broad_e5_tolerance_envelope.py`` (T2.4, magnitude) and
``test_waveguide_broad_e5_phase_tolerance_envelope.py`` (Lane 1 phase) for the
group-delay lane: ``MAX_GROUP_DELAY_TOL_NS`` is now a PINNED module constant
(not derived on-the-fly), and this test independently re-derives the
like-for-like residual from the committed fixture's stored arrays and checks
the pinned constant envelopes it (never fails a validated case) without being
slack (bounded margin).

Pure-Python contract (no FDTD) -- replays the committed JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ENVELOPE = REPO / "tests/fixtures/waveguide_group_delay/wr340_near_cutoff_group_delay_envelope.json"

sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_group_delay_envelope import MAX_GROUP_DELAY_TOL_NS  # type: ignore  # noqa: E402

# Same governance margin ceiling as the magnitude (1.5) and phase (1.5) lanes.
# 0.042 / 0.0320 = 1.31, well inside 1.5.
MARGIN_CEIL = 1.5


def _load():
    return json.loads(ENVELOPE.read_text())


def _like_for_like_worst_diff_ns(env: dict) -> float:
    """Independently re-derive the GATED (like-for-like) residual from the
    committed fixture's stored arrays -- does not trust env['max_abs_diff_ns']
    at face value, recomputes it from tau_g_measured_ns and
    tau_g_analytic_via_stencil_ns at the interior indices."""
    meas = np.array(env["tau_g_measured_ns"])
    ana_stencil = np.array(env["tau_g_analytic_via_stencil_ns"])
    idx = env["interior_indices"]
    return float(np.abs(meas[idx] - ana_stencil[idx]).max())


def test_max_group_delay_tol_is_a_bounded_measured_envelope():
    env = _load()
    worst = _like_for_like_worst_diff_ns(env)
    # Cross-check: the producer's own reported max_abs_diff_ns must agree with
    # this independent recomputation (catches a doctored summary).
    assert worst == pytest.approx(env["max_abs_diff_ns"], rel=1e-9)
    assert worst <= MAX_GROUP_DELAY_TOL_NS, (
        f"MAX_GROUP_DELAY_TOL_NS={MAX_GROUP_DELAY_TOL_NS} is below the worst "
        f"committed like-for-like residual {worst:.4f} ns -- it would fail a "
        f"validated case (a real group-delay regression)."
    )
    assert MAX_GROUP_DELAY_TOL_NS <= worst * MARGIN_CEIL, (
        f"MAX_GROUP_DELAY_TOL_NS={MAX_GROUP_DELAY_TOL_NS} exceeds worst "
        f"like-for-like diff {worst:.4f} ns x {MARGIN_CEIL} = "
        f"{worst * MARGIN_CEIL:.4f} ns -- the tolerance is slack; re-justify "
        f"or tighten."
    )


def test_stencil_truncation_error_is_disclosed_not_hidden():
    """The comparator-neutrality fix (M3) must leave a visible trace: the
    fixture discloses both the gated like-for-like diff AND the (larger)
    stencil truncation error, so a reader cannot mistake the smaller
    vs-exact-oracle number for the gated one."""
    env = _load()
    assert "max_abs_diff_ns_vs_exact_oracle_not_gated" in env
    assert "stencil_truncation_error_ns_max_not_gated" in env
    assert "comparator_note" in env
    # The vs-exact-oracle number must be SMALLER than the gated like-for-like
    # number here (that is precisely the flattering-cancellation finding) --
    # if a future re-run makes them equal or inverts, the note text (which
    # explains the cancellation) would be stale and must be revisited, not
    # silently kept.
    assert env["max_abs_diff_ns_vs_exact_oracle_not_gated"] < env["max_abs_diff_ns"]
