"""Issue #498 / #517 — tests for the mixed-lane reference-plane measurement.

Four lanes, all cheap (pure NumPy, or ``num_periods <= 4``):

* **T3 / evaluators** — the F1 / F2 / F3 verdict functions on SYNTHETIC
  planted data. Every falsifier must be resolvable BOTH ways (each declared
  side reachable) and must return ``UNRESOLVED`` when its own precondition
  fails: the ring-down witness at or above -40 dB, or its discriminator below
  its own budget. ``NON_DISCRIMINATING`` is a legal, distinct outcome.
* **T3b / normalization identity** — ``|b_msl| == |out| / sqrt(Re Zc)`` on a
  synthetic single-travelling-wave line. Written against the WRONG
  ``|out|^2/(4 Re Zc)`` form first: that form under-reads by exactly 4 in
  power, and this test is what turns the corrected form green.
* **T1 / domain + absorber guard** — the mixed-lane reference-plane path must
  REJECT a plane that lands outside the declared domain, or within a face's
  absorber thickness of it, with a message class DISTINCT from "reach past
  another port".
* **T2 / crossing guard walks MSL probe ladders** — ``n_probes=5`` trips (the
  2N plane at 3.60 mm vs the last rung at 3.44 mm) and ``n_probes=3`` is
  silent, both at ``num_periods=4``.

FAIL-BEFORE-FIX (measured on this worktree, pristine ``HEAD`` = 3038f845 with
``rfx/api/_execute.py`` restored, same driver script)::

    T1 (direction='+x', n_probes=3): NO RAISE (silent)
    T2 (direction='-x', n_probes=5): NO RAISE (silent)
    CONTROL (direction='-x', n_probes=3): NO RAISE (silent)

and after the guards::

    T1 (direction='+x', n_probes=3): ValueError -> reference_plane_cells:
        reference plane slot 1 ... lands at index 13 on axis 0 (x), within the
        absorber thickness of a declared-domain face ...
    T2 (direction='-x', n_probes=5): ValueError -> reference_plane_cells: the
        reference planes ... reach past another port at (0.0055, 0.0015, 0.0)
        — its MSL de-embedding probe 4 at x = 0.00344 m (index 51) ...
    CONTROL (direction='-x', n_probes=3): NO RAISE (silent)

Nothing here pins a lumped/wire diagonal, moves a tolerance, or touches a
reference — see §10 of
``docs/design_notes/issue498_mixed_refplane_predeclaration.md``.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_DRIVER = REPO / "scripts" / "diagnostics" / "i498_mixed_refplane_measurement.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("_i498_driver", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load_driver()

_SETTLED = np.array([-122.5, -119.9])       # both drives below -40 dB
_HOT = np.array([-9.6, -11.2])              # the num_periods=4 smoke's own


# ===========================================================================
# T3 — F1 evaluator, every declared side reachable on planted data
# ===========================================================================

def _f1_call(**kw):
    base = dict(
        zc_im_re=np.full(5, 0.005),
        r1=np.full(5, 1.00),
        r1_xz=np.full(5, 1.00),
        y_face_contrib=np.zeros(5),
        box_net=np.full(5, 1e-28),
        settling_db=_SETTLED,
    )
    base.update(kw)
    return m.evaluate_f1(**base)


def test_f1_side_a_instrument_good():
    v = _f1_call(r1=np.array([0.96, 0.99, 1.00, 1.02, 1.029]))
    assert v["verdict"] == "SIDE_A_INSTRUMENT_GOOD"
    assert v["resolved"] and v["side"] is None


def test_f1_side_b_plane_bad_stops_the_plan():
    """The instrument itself failed: |Im Zc/Re Zc| above the class boundary."""
    v = _f1_call(zc_im_re=np.array([0.005, 0.005, 0.082, 0.005, 0.005]))
    assert v["verdict"] == "SIDE_B_PLANE_BAD"
    assert v["resolved"] and "plane" in v["side"]


def test_f1_side_c_box_undercapture_continues():
    """R1 misses, but the +/-y pair carries the miss and the x/z-only box
    reconciles — the declared box y half-width limitation."""
    box = np.full(5, 1.0)
    r1 = np.full(5, 0.80)                      # numerator = 0.80 * box
    miss = box - r1 * box                       # 0.20
    v = _f1_call(r1=r1, r1_xz=np.full(5, 1.00),
                 y_face_contrib=0.9 * miss, box_net=box)
    assert v["verdict"] == "SIDE_C_BOX_UNDERCAPTURE"
    assert v["resolved"] and "box" in v["side"]


def test_f1_side_d_unattributed_miss_is_a_declared_quadrant():
    """Blocker B2's fourth quadrant: R1 misses, y attribution INVALID."""
    box = np.full(5, 1.0)
    r1 = np.full(5, 0.80)
    miss = box - r1 * box
    v = _f1_call(r1=r1, r1_xz=np.full(5, 0.80),
                 y_face_contrib=0.10 * miss, box_net=box)
    assert v["verdict"] == "SIDE_D_UNATTRIBUTED_MISS"
    assert v["side"] == "unattributed"


def test_f1_side_d_when_yshare_ok_but_xz_box_still_misses():
    box = np.full(5, 1.0)
    r1 = np.full(5, 0.80)
    miss = box - r1 * box
    v = _f1_call(r1=r1, r1_xz=np.full(5, 0.60),
                 y_face_contrib=0.95 * miss, box_net=box)
    assert v["verdict"] == "SIDE_D_UNATTRIBUTED_MISS"


def test_f1_unresolved_when_settling_precondition_fails():
    v = _f1_call(settling_db=_HOT)
    assert v["verdict"] == "UNRESOLVED" and not v["resolved"]
    assert "settling" in v["reason"]


def test_f1_unresolved_when_discriminator_below_budget():
    """A vanishing box net is not a discriminator, whatever R1 says."""
    v = _f1_call(box_net=np.array([1.0, 1.0, 0.0, 1.0, 1.0]))
    assert v["verdict"] == "UNRESOLVED"
    assert "discriminator below budget" in v["reason"]


def test_f1_settling_precondition_dominates_a_would_be_conviction():
    """A hot run must never convict — truncation, not physics."""
    v = _f1_call(settling_db=_HOT, zc_im_re=np.full(5, 0.5))
    assert v["verdict"] == "UNRESOLVED"


# ===========================================================================
# T3 — F2 evaluator, both sides + the two non-discriminating outcomes
# ===========================================================================

def test_f2_budget_is_computed_from_the_run():
    b = m.f2_budget(58.0, 47.89479996289313)
    assert b == pytest.approx(abs((58.0 - 47.8948) / (58.0 + 47.8948)) + 0.01,
                              rel=1e-4)


def test_f2_consistent_is_explicitly_not_vindicated():
    v = m.evaluate_f2(m2=np.full(5, 0.021), s22=np.full(5, 0.020),
                      zc_meas_re_band_mean=48.5, settling_db=_SETTLED)
    assert v["verdict"] == "CONSISTENT_NON_DISCRIMINATING"
    assert not v["resolved"] and v["side"] is None
    assert "NOT 'the MSL diagonal is vindicated'" in v["reason"]


def test_f2_msl_diagonal_convicted_at_the_predeclared_alternative():
    """The other side: M2 lands on the reciprocity-implied ~0.43 row."""
    v = m.evaluate_f2(m2=np.asarray(m._F2_ALT_S22),
                      s22=np.array([0.0199, 0.0181, 0.0180, 0.0230, 0.0340]),
                      zc_meas_re_band_mean=48.5, settling_db=_SETTLED)
    assert v["verdict"] == "MSL_DIAGONAL_CONVICTED"
    assert v["resolved"] and "MSL diagonal" in v["side"]


def test_f2_reported_no_attribution_third_declared_possibility():
    v = m.evaluate_f2(m2=np.full(5, 0.20),
                      s22=np.array([0.0199, 0.0181, 0.0180, 0.0230, 0.0340]),
                      zc_meas_re_band_mean=48.5, settling_db=_SETTLED)
    assert v["verdict"] == "REPORTED_NO_ATTRIBUTION"
    assert not v["resolved"] and v["side"] is None


def test_f2_non_discriminating_anchor_when_budget_exceeds_its_own_cap():
    """B > 0.15: the discriminator is below its budget, so no attribution."""
    v = m.evaluate_f2(m2=np.asarray(m._F2_ALT_S22),
                      s22=np.full(5, 0.02),
                      zc_meas_re_band_mean=90.0, settling_db=_SETTLED)
    assert v["verdict"] == "NON_DISCRIMINATING_ANCHOR"
    assert not v["resolved"] and v["side"] is None
    assert v["numbers"]["B"] > m._F2_B_MAX


def test_f2_unresolved_when_settling_precondition_fails():
    v = m.evaluate_f2(m2=np.asarray(m._F2_ALT_S22), s22=np.full(5, 0.02),
                      zc_meas_re_band_mean=48.5, settling_db=_HOT)
    assert v["verdict"] == "UNRESOLVED" and "settling" in v["reason"]


# ===========================================================================
# T3 — F3 evaluator
# ===========================================================================

def test_f3_in_window_says_where_the_residual_is_not():
    v = m.evaluate_f3(r3=np.array([0.96, 0.99, 1.00, 1.02, 1.029]),
                      zc_meas_re_band_mean=48.5, settling_db=_SETTLED)
    assert v["verdict"] == "MSL_RECEIVE_AGREES"
    assert v["resolved"] and v["side"] is None


def test_f3_anchor_side_when_r3_equals_sqrt_zc_over_zhj():
    zc = 60.0
    anchor = float(np.sqrt(zc / m._Z0_HJ_COMMITTED))
    v = m.evaluate_f3(r3=np.full(5, anchor), zc_meas_re_band_mean=zc,
                      settling_db=_SETTLED)
    assert v["verdict"] == "MSL_ANCHOR_CONVICTED"
    assert "anchor" in v["side"]


def test_f3_extractor_side_when_r3_misses_both():
    v = m.evaluate_f3(r3=np.full(5, 0.50), zc_meas_re_band_mean=48.5,
                      settling_db=_SETTLED)
    assert v["verdict"] == "MSL_EXTRACTOR_CONVICTED"
    assert "extractor" in v["side"]


def test_f3_unresolved_when_settling_precondition_fails():
    v = m.evaluate_f3(r3=np.full(5, 0.50), zc_meas_re_band_mean=48.5,
                      settling_db=_HOT)
    assert v["verdict"] == "UNRESOLVED" and "settling" in v["reason"]


def test_f3_unresolved_when_the_discriminator_is_below_its_budget():
    """A plane-wave denominator at or below the signal floor is noise."""
    v = m.evaluate_f3(r3=np.full(5, 0.50), zc_meas_re_band_mean=48.5,
                      settling_db=_SETTLED,
                      denominator=np.array([1.0, 1.0, 1e-30, 1.0, 1.0]),
                      signal_floor=1e-20)
    assert v["verdict"] == "UNRESOLVED"
    assert "discriminator below budget" in v["reason"]


# ===========================================================================
# T3b — the normalization identity, pure NumPy (fail-first against ÷4)
# ===========================================================================

def test_b_msl_equals_out_over_sqrt_zc_on_a_single_travelling_wave():
    """``refplane_split``'s 0.5 and ``_b_msl``'s 2 already cancel.

    Plant a pure -x-travelling wave on a real line: V = V-, I = -V-/Z. Then
    ``out`` at a ``-x``-outboard plane is exactly V- and
    ``b_msl = V-/sqrt(Z)``, so ``|b_msl| == |out|/sqrt(Re Zc)`` EXACTLY —
    and NOT ``|out|/(2 sqrt(Re Zc))``, the form that under-reads by 4 in
    power.
    """
    from rfx.probes.refplane import refplane_split

    z = 47.89479996289313
    v_minus = np.array([1.0 + 0.5j, -0.3 + 2.0j, 4.0 - 1.0j])
    v = v_minus
    i = -v_minus / z
    # outboard_sign = +1 (a "-x" port pushes its planes toward +x), so the
    # OUTGOING wave at the plane is w_plus = 0.5*(v + z*i)... which is zero
    # for a purely -x-travelling wave. Use the -x-outboard orientation, where
    # `outgoing` is w_minus = 0.5*(v - z*i) = V-.
    out, inc = refplane_split(v, i, z, outboard_sign=-1)
    np.testing.assert_allclose(out, v_minus, rtol=0, atol=1e-12)

    b_msl = m.b_msl_probe0(v, i, z)
    np.testing.assert_allclose(np.abs(b_msl), np.abs(v_minus) / np.sqrt(z),
                               rtol=1e-12)

    correct = np.abs(out) / np.sqrt(z)
    np.testing.assert_allclose(np.abs(b_msl), correct, rtol=1e-12)

    # The reviewer's ÷4-in-power form is wrong by exactly 2 in amplitude.
    wrong = np.abs(out) / (2.0 * np.sqrt(z))
    ratio = np.abs(b_msl) / wrong
    np.testing.assert_allclose(ratio, np.full_like(ratio, 2.0), rtol=1e-12)
    np.testing.assert_allclose((np.abs(b_msl) / wrong) ** 2,
                               np.full_like(ratio, 4.0), rtol=1e-12)


def test_refplane_split_half_is_already_inside():
    """Guards the premise of the identity above against a silent refactor."""
    from rfx.probes.refplane import refplane_split
    v = np.array([2.0 + 0.0j])
    i = np.array([0.0 + 0.0j])
    out, inc = refplane_split(v, i, 50.0, outboard_sign=+1)
    assert out[0] == pytest.approx(1.0)      # 0.5 * v, not v
    assert inc[0] == pytest.approx(1.0)


# ===========================================================================
# Report-only guards
# ===========================================================================

@pytest.mark.parametrize("name", [
    "i498_reference_values.json", "i498_gate_record.json",
    "i498_snapshot.json", "i498_baseline.json", "i498_expected.json",
])
def test_driver_refuses_to_write_anything_that_reads_as_a_record(name, tmp_path):
    with pytest.raises(RuntimeError, match="report-only"):
        m._assert_report_only(tmp_path / name)


@pytest.mark.parametrize("rel", [
    "tests/x.json", "docs/x.json", "validation/x.json", "rfx/x.json",
])
def test_driver_refuses_to_write_into_committed_evidence_trees(rel):
    with pytest.raises(RuntimeError, match="report-only"):
        m._assert_report_only(REPO / rel)


def test_driver_allows_its_own_diagnostic_artifact(tmp_path):
    p = m._assert_report_only(tmp_path / "i498_mixed_refplane_measurement.json")
    assert p.name.endswith(".json")


def test_driver_refuses_to_pin_the_lw_diagonal():
    ok = {"s_matrix": {"abs_S00_recorded": [0.3814, 0.3863]}}
    m._assert_no_lw_diagonal_pin(ok)          # recording is fine
    bad = {"s_matrix": {"abs_S00_expected": [0.3814, 0.3863]}}
    with pytest.raises(RuntimeError, match="pin"):
        m._assert_no_lw_diagonal_pin(bad)


def test_f4_is_not_carried_as_a_falsifier():
    assert m.FALSIFIERS == ("F1", "F2", "F3")
    assert m._F4_PREDICTION_NOT_A_FALSIFIER["may_be_pinned"] is False


# ===========================================================================
# T1 / T2 — plumbing guards on the real mixed lane (num_periods = 4)
# ===========================================================================

def _run_mixed(direction: str, n_probes: int, num_periods: float = 4.0):
    """One instrumented two-drive call; returns the result or raises."""
    import contextlib
    import io

    sim = m.build_fixture(feed_direction=direction, n_probes=n_probes,
                          register_extra_flux=False)
    with m.refplane_instrumentation(m._REFPLANE_N):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                return sim.compute_mixed_s_matrix(
                    freqs=np.linspace(1e9, 4e9, 5),
                    num_periods=num_periods, skip_preflight=True,
                )


@pytest.mark.slow
def test_t1_mixed_refplane_path_rejects_absorber_adjacent_planes():
    """T1: ``direction="+x"`` puts slot 1 at index 13, six cells from the
    declared-domain lo face (index 8) inside the 8-cell absorber margin.

    On ``main`` this built SILENTLY (fail-before-fix evidence in the module
    docstring). The message class must be DISTINCT from the crossing guard's
    "reach past another port" — the two failures have different remedies.
    """
    with pytest.raises(ValueError) as ei:
        _run_mixed("+x", m._N_PROBES)
    msg = str(ei.value)
    assert "reference_plane_cells" in msg
    assert "absorber thickness of a declared-domain face" in msg or \
           "OUTSIDE the declared domain" in msg
    assert "reach past another port" not in msg


@pytest.mark.slow
def test_t2_crossing_guard_walks_msl_probe_ladders_n_probes_5_trips():
    """T2 (bookkeeping): the default ``n_probes=5`` ladder's last rung is at
    x = 3.44 mm (index 51), INBOARD of the 2N plane at 3.60 mm (index 53).

    The port position alone (index 77) can never see this, which is why the
    guard now walks ``self._msl_ports``. Message class stays "reach past
    another port".
    """
    with pytest.raises(ValueError) as ei:
        _run_mixed(m._FEED_DIRECTION, 5)
    msg = str(ei.value)
    assert "reach past another port" in msg
    assert "MSL de-embedding probe" in msg
    assert "0.00344" in msg          # the offending rung, named


@pytest.mark.slow
def test_t2_crossing_guard_silent_at_n_probes_3():
    """The declared deviation: ladder 4.72 / 4.40 / 4.08 mm, all outboard of
    the 2N plane. Must run clean — and must actually register the planes."""
    sim = m.build_fixture(feed_direction=m._FEED_DIRECTION,
                          n_probes=m._N_PROBES, register_extra_flux=False)
    import contextlib
    import io
    with m.refplane_instrumentation(m._REFPLANE_N) as cap:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                result = sim.compute_mixed_s_matrix(
                    freqs=np.linspace(1e9, 4e9, 5), num_periods=4.0,
                    skip_preflight=True)
    assert result is not None
    # B1 positive control: planes on BOTH drives, and the MSL-driven run
    # carries NO lw source.
    assert len(cap.positive_control) == 2
    assert all(pc["n_refplane_specs"] == 2 for pc in cap.positive_control)
    assert [pc["lw_excite_flags"] for pc in cap.positive_control] == \
        [[True], [False]]
    assert cap.positive_control[1]["drive_idx"] == m._NO_LW_DRIVE_SENTINEL


@pytest.mark.slow
def test_refplane_geometry_of_record_is_what_the_predeclaration_says():
    """Slot 0 at index 43 (x = 2.80 mm) and slot 1 at index 53 (3.60 mm),
    ``outboard_sign = +1`` for ``direction="-x"`` — the two deviations'
    measured geometry, frozen so a silent drift is loud."""
    sim = m.build_fixture(register_extra_flux=False)
    import contextlib
    import io
    with m.refplane_instrumentation(m._REFPLANE_N) as cap:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                sim.compute_mixed_s_matrix(
                    freqs=np.linspace(1e9, 4e9, 5), num_periods=1.0,
                    skip_preflight=True)
    rp = m.extract_refplane(cap.runs[0], np.linspace(1e9, 4e9, 5),
                            float(cap.grid.dt), float(cap.grid.dx))
    assert rp["outboard_sign"] == +1
    assert rp["slots"][0]["plane_index"] == 43
    assert rp["slots"][1]["plane_index"] == 53
    assert rp["separation_m"] == pytest.approx(10 * 80e-6)
