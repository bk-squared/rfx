"""Issue #490 Lane 3 -- group delay near cutoff, gated against L_eff / v_g(f).

Replays the committed envelope produced by
``scripts/diagnostics/build_waveguide_group_delay_envelope.py`` from the 3
fresh FDTD runs of ``build_waveguide_near_cutoff_group_delay.py``
(baseline_np60 / settling_np120 / grown_domain_np60) on a purpose-built,
empty WR-340 guide near cutoff (f/fc in [1.152, 1.498] -- see that script's
frozen pre-declaration docstring for the fixture design, qualitative
expectations, and why this margin was chosen over an asymptotically-close one).

Convention (same family as Lane 1, see
``build_waveguide_band_broad_e5_phase_envelope.py`` for the full note):
e^{+jwt} time convention; S21 phase measured between the two declared
``reference_plane`` x-coordinates; ``tau_g = -d(unwrap(angle(S21)))/d(omega)``
via a central-difference stencil (O(domega^2)) at the 9 interior frequency
points, one-sided (O(domega)) at the 2 band-edge points (reported, not
gated).

Comparator (adversarial review of PR #536, finding M3): the gate compares
the measured finite-difference tau_g against the analytic phase run through
the IDENTICAL finite-difference stencil, not the exact closed-form
derivative -- otherwise the stencil's own truncation error (opposite sign to
the true residual here) partially cancels it and flatters the headline
number. See ``build_waveguide_group_delay_envelope.py`` for the derivation
and ``test_waveguide_group_delay_tolerance_envelope.py`` for the pinned,
non-tautological tolerance constant.

Falsifiers (finding M2): three genuinely independent defects -- skipping the
phase-unwrap step, dropping the tau_g leading minus sign, and using the WRONG
L_eff (domain length instead of reference-plane separation) in the analytic
comparator. An earlier ``conjugate_flip`` falsifier was found to be
IDENTICAL to ``wrong_sign`` on this derivation chain (6.7e-16) and has been
replaced.

Pure-Python contract (no FDTD) -- replays the committed JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENVELOPE = REPO / "tests/fixtures/waveguide_group_delay/wr340_near_cutoff_group_delay_envelope.json"


def _load():
    assert ENVELOPE.exists(), (
        "wr340_near_cutoff_group_delay_envelope.json is missing -- run "
        "scripts/diagnostics/build_waveguide_near_cutoff_group_delay.py "
        "for all 3 cases then "
        "scripts/diagnostics/build_waveguide_group_delay_envelope.py, and commit it"
    )
    return json.loads(ENVELOPE.read_text())


def test_group_delay_gate_passes():
    env = _load()
    assert env["status"] == "passed", env["claim"]
    assert env["max_abs_diff_ns"] <= env["max_group_delay_tol_ns"], env


def test_qualitative_expectations_hold():
    """Pre-declared expectations (frozen before the run): monotonic decrease,
    and the measured near/far-cutoff ratio is close to the analytic one.

    LOW(d), adversarial review of PR #536: the ratio check uses the INTERIOR
    bins (indices 1 and 9), not the band-edge endpoints (0 and 10) -- a
    sibling test (``test_endpoints_are_reported_but_not_gated``) asserts the
    endpoints use a lower-order stencil and are not gated, so this qualitative
    check should not lean on them either."""
    env = _load()
    q = env["qualitative_expectations"]
    assert q["monotonic_decrease"] is True, (
        "tau_g(f) is not monotonically decreasing -- pre-declared expectation "
        "failed; per the frozen stop rule this should have been reported, not "
        "silently tuned"
    )
    meas = q["measured_ratio_tau1_over_tau9_interior"]
    ana = q["analytic_exact_ratio_tau1_over_tau9_interior"]
    assert abs(meas - ana) / ana < 0.15, (
        f"measured interior tau1/tau9 ratio {meas:.3f} vs analytic {ana:.3f} -- "
        f">15% off the pre-declared order-of-magnitude/sign check"
    )


def test_endpoints_are_reported_but_not_gated():
    """Honesty check: the 2 lower-accuracy one-sided endpoints must be
    visible in the record (not silently dropped) but excluded from the gate
    (they use a first-order, not second-order, stencil)."""
    env = _load()
    assert env["endpoints_gated"] is False
    assert len(env["tau_g_measured_ns"]) == 11
    assert env["interior_indices"] == list(range(1, 10))


def test_settling_witness_is_recorded_and_tight():
    """Record-length convergence (num_periods 60 vs 120) substitutes for the
    MSL-style energy dB witness (compute_waveguide_s_matrix exposes no
    settling_db for the waveguide port family). If the num_periods=60 record
    were under-settled, doubling the record would move tau_g materially."""
    env = _load()
    w = env["settling_witness"]
    assert "record-length convergence" in w["method"]
    assert w["max_abs_diff_ns_np60_vs_np120"] <= env["max_group_delay_tol_ns"], (
        f"np=60 vs np=120 group delay differs by {w['max_abs_diff_ns_np60_vs_np120']:.4f} ns "
        f"-- the num_periods=60 baseline record looks under-settled"
    )


def test_domain_invariance_witness_does_not_flip_verdict():
    """Growing the domain by +100mm (symmetric buffer, L_eff held fixed)
    must not flip the group-delay gate verdict."""
    env = _load()
    inv = env["domain_invariance_witness"]
    assert inv["baseline_verdict"] == "passed", inv
    assert inv["grown_verdict"] == "passed", inv
    assert not inv["verdict_flipped"], inv


@pytest.mark.parametrize("name", ["skip_unwrap", "wrong_sign", "wrong_l_eff"])
def test_falsifier_reds_the_gate(name):
    """Mandate: a planted defect must red the gate. Three genuinely
    independent defects are checked -- skipping the unwrap step, dropping the
    tau_g leading minus sign, and using the WRONG L_eff (domain length
    instead of reference-plane separation) in the analytic comparator. (An
    earlier ``conjugate_flip`` falsifier was found, on adversarial review, to
    be identical to ``wrong_sign`` on this derivation chain -- 6.7e-16 apart,
    not an independent defect -- and was replaced with ``wrong_l_eff``.)"""
    env = _load()
    f = env["falsifiers"][name]
    assert f["gate_reds"] is True, (
        f"falsifier {name!r} worst diff {f['worst_abs_diff_ns']:.4f} ns did "
        f"NOT exceed the gate tolerance {env['max_group_delay_tol_ns']:.4f} ns"
    )
