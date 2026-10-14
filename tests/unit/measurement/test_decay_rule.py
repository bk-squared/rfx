"""The post-drive decay rule on hand-written boundary energies.

Each sequence lists W at counts 10, 20, 30, ... (interval 10, threshold 0.1
unless stated, two consecutive boundaries). The expected stops, criteria and
Diagnostic values are worked out by hand in the comments (with the floor
constant 1e-12 written here, not read from rfx.measurement.decay); the rule
is fed through the same ``check`` the loops call, and the Diagnostic comes
from the same ``finish`` they call.
"""

import math
import warnings

import pytest

from rfx.measurement.decay import DecayStop, finish_decay


def _run(values, *, n_end=10, min_steps=0, consecutive=2):
    rule, asked = DecayStop(.1, min_steps, 10, consecutive, n_end), []
    for count, value in enumerate(values, 1):
        rule.check(count * 10, lambda value=value, count=count: asked.append(count * 10) or value)
    return rule, asked


def test_no_stop_before_drive_end_however_low():
    # n_end = 40: W is read at every boundary (the floor's largest W, 1000 at
    # 10), but the 1000 and the two 0.01 at 10..30 are not in ref and cannot
    # stop. From 40 on: ref = 10, then 0.5 < max(1, 1e-9) at 50 and 60 -> two
    # in a row, stop at 60.
    rule, asked = _run([1000, .01, .01, 10, .5, .5], n_end=40)
    assert asked == [10, 20, 30, 40, 50, 60]
    assert rule.wmax == 1000
    assert rule.stop == 60
    assert rule.ref == 10
    assert rule.decay_checks == [(40, 10.0, 10.0), (50, .5, 10.0), (60, .5, 10.0)]


def test_reference_rises_after_drive_end():
    # ref 10 at 10, 100 at 20; 5 < 0.1*100 at 30 and 40 -> stop at 40. With the
    # reference frozen at its first value, 5 < 1 would never hold.
    rule, _ = _run([10, 100, 5, 5])
    assert rule.stop == 40
    assert [row[2] for row in rule.decay_checks] == [10, 100, 100, 100]


def test_one_boundary_dip_recovers():
    # 1 < 10 at 20 (below = 1), 20 > 10 at 30 resets, 1 at 40 (below = 1),
    # 20 at 50 resets: never two in a row, no stop.
    rule, _ = _run([100, 1, 20, 1, 20])
    assert rule.stop is None
    assert rule.below == 0


def test_min_steps_beyond_crossing():
    # Below at 20, 30, 40, 50 (below = 4 at 50); counts before min_steps = 45
    # cannot stop, so the first stop is 50, the first boundary >= 45.
    rule, _ = _run([100, 1, 1, 1, 1], min_steps=45)
    assert rule.stop == 50
    assert [row[0] for row in rule.decay_checks] == [50]


def test_drive_never_ended_reads_nothing():
    rule, asked = _run([100, 1, 1, 1, 1], n_end=None)
    assert asked == []
    assert rule.stop is None and rule.decay_checks == []


def test_off_boundary_counts_are_not_read():
    rule = DecayStop(.1, 0, 10, 2, 0)
    rule.check(15, lambda: pytest.fail("W read off a k * interval boundary"))
    assert rule.decay_checks == []


@pytest.mark.parametrize("consecutive", [1, 0, -1])
def test_refuses_fewer_than_two_with_the_energy_criterion(consecutive):
    with pytest.raises(ValueError, match="decay_energy_consecutive must be >= 2"):
        DecayStop(.1, 0, 10, consecutive, 10)
    # Not the run's stop rule (forced length, the flux criterion): accepted.
    assert DecayStop(0., 0, 10, consecutive, None, enabled=False).stop is None


# --- how the run ended: the until_decay.stop Diagnostic (D3/D4) ----------------

FLOOR_TEXT = ("below about 1e-12 of the drive's peak energy the remaining field is at the "
              "single-precision rounding level of the drive's peak field; the requested "
              "threshold was not met")


def _end(values, *, thr, n_end, steps):
    rule = DecayStop(thr, 0, 10, 2, n_end)
    for count, value in enumerate(values, 1):
        if rule.check(count * 10, lambda value=value: value) is not None:
            break
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostic = finish_decay(rule, steps)
    return diagnostic, [str(w.message) for w in caught]


def test_threshold_end_diagnostic():
    # Drive (W 1000) at 10, n_end 20: ref 100 at 20. Below when W < max(0.1 * 100,
    # 1e-12 * 1000) = 10: 5 at 30 and 40 -> stop at 40 on the threshold (10 >= 1e-9).
    # reached = 5/100, floor_ratio = 1e-12 * 1000 / 100.
    d, warned = _end([1000, 100, 5, 5, 5], thr=.1, n_end=20, steps=40)
    assert (d.code, d.severity, d.subject) == ("until_decay.stop", "info", "threshold")
    v = d.values
    assert v["criterion"] == "threshold" and v["quantity"] == "conserved_energy"
    assert (v["requested_threshold"], v["stop_step"], v["drive_end_step"]) == (.1, 40, 20)
    assert v["reached"] == pytest.approx(5 / 100, rel=1e-12)
    assert v["floor_ratio"] == pytest.approx(1e-12 * 1000 / 100, rel=1e-12)
    assert not warned


def test_floor_end_diagnostic_and_one_warning():
    # Drive W 1e12 at 10, n_end 20, ref = W(20) = 1. Requested 1e-4 asks W < 1e-4;
    # 0.5 at 30 is under the floor 1e-12 * 1e12 = 1 but fell from 1 (0.5 < 0.95 * 1):
    # not below. 0.5 at 40 and 50 has stopped falling (>= 0.95 * 0.5): below on the
    # floor twice -> stop at 50 on the floor. reached 0.5, floor_ratio 1, 0.5e-12 of
    # the drive's peak energy.
    d, warned = _end([1e12, 1, .5, .5, .5], thr=1e-4, n_end=20, steps=50)
    assert (d.severity, d.subject, d.values["criterion"]) == (
        "warning", "float32_floor", "float32_floor")
    assert d.values["stop_step"] == 50
    assert d.values["reached"] == pytest.approx(.5, rel=1e-12)
    assert d.values["floor_ratio"] == pytest.approx(1.0, rel=1e-12)
    # The message: requested threshold, then the levels where W stopped falling, then
    # the floor text.
    msg = d.message
    assert (msg.index("requested threshold 0.0001") < msg.index("stopped falling at 0.5")
            < msg.index("(5e-13 of the drive's peak energy)") < msg.index(FLOOR_TEXT))
    assert warned == [msg]


def test_floor_does_not_stop_a_decay_below_it():
    # Drive W 1e12, then W falls 30 % per check from 1 (all under the floor 1, all above
    # the requested 1e-9 * ref): a decaying field, not rounding -> no stop.
    values = [1e12] + [0.7 ** k for k in range(12)]
    rule = DecayStop(1e-9, 0, 10, 2, 20)
    for count, value in enumerate(values, 1):
        rule.check(count * 10, lambda value=value: value)
    assert rule.stop is None


def test_floor_stops_a_flat_level_below_it():
    # The same drive, then W flat at 0.3 under the floor: stopped falling -> stop at
    # the second boundary after the first flat one (40: 0.3 against 1 is a fall).
    values = [1e12, 1, .3, .3, .3]
    rule = DecayStop(1e-9, 0, 10, 2, 20)
    for count, value in enumerate(values, 1):
        rule.check(count * 10, lambda value=value: value)
    assert (rule.stop, rule.criterion) == (50, "float32_floor")


def test_cap_end_diagnostic():
    # n_end 20, ref 10 (W at 20); W stays at 5 > max(1e-3 * 10, 1e-12 * 1000): no
    # stop. The run reaches its cap (steps 60): reached = 5/10 at the last boundary,
    # floor_ratio = 1e-12 * 1000 / 10.
    d, warned = _end([1000, 10, 5, 5, 5, 5], thr=1e-3, n_end=20, steps=60)
    assert (d.severity, d.subject, d.values["criterion"]) == ("warning", "cap", "cap")
    assert d.values["stop_step"] == 60
    assert d.values["reached"] == pytest.approx(.5, rel=1e-12)
    assert d.values["floor_ratio"] == pytest.approx(1e-12 * 1000 / 10, rel=1e-12)
    assert not warned


def test_drive_still_on_cap_has_no_level():
    d, _ = _end([1, 1, 1], thr=1e-3, n_end=None, steps=30)
    assert d.values["criterion"] == "cap"
    assert math.isnan(d.values["reached"]) and math.isnan(d.values["drive_end_step"])


def test_forced_length_has_no_record():
    assert finish_decay(DecayStop(0., 0, 10, 2, 10, enabled=False), 30) is None


def test_waveguide_port_without_its_applied_table_is_refused():
    # A port initialised without dt holds a (1,) zero sentinel instead of the
    # incident table it applies, and emits through the legacy soft-E path: its
    # drive end cannot be read, so until_decay refuses and names that.
    from types import SimpleNamespace

    import jax.numpy as jnp

    from rfx.measurement.decay import drive_end

    grid = SimpleNamespace(dt=1e-12)
    port = SimpleNamespace(src_amp=1.0, e_inc_table=jnp.zeros((1,)))
    with pytest.raises(ValueError, match=r"waveguide port\[0\] was initialised without dt"):
        drive_end(grid, 100, waveguide_ports=(port,))
    # The applied table, held at its last entry past its end, is read at distance 0.
    table = jnp.concatenate([jnp.ones(30), jnp.zeros(40)])
    assert drive_end(grid, 100, waveguide_ports=(SimpleNamespace(src_amp=1.0, e_inc_table=table),)) == 30
    held = jnp.concatenate([jnp.ones(30), jnp.full(40, .5)])
    assert drive_end(grid, 100, waveguide_ports=(SimpleNamespace(src_amp=1.0, e_inc_table=held),)) is None


def test_threshold_label_when_the_whole_final_run_is_below_the_threshold():
    # Interval 1, n_end 1: W 1 at 0 (drive), ref 1e-7 at 1. The threshold 1e-6 * 1e-7 =
    # 1e-13 is under the floor 1e-12 * 1, but 1e-14 and 1e-15 at 2 and 3 are below the
    # threshold itself: the stop at 3 is on the threshold.
    rule = DecayStop(1e-6, 0, 1, 2, 1)
    for count, w in enumerate([1.0, 1e-7, 1e-14, 1e-15]):
        rule.check(count, lambda w=w: w)
    assert (rule.stop, rule.criterion) == (3, "threshold")


def test_floor_flatness_is_per_50_steps_not_per_check():
    # Interval 10: W falling 2 % per check is 9.6 % per 50 steps, still decaying; it
    # must not stop on the floor (a per-check 0.95 would call it flat).
    values = [1e12] + [0.98 ** k for k in range(15)]
    rule = DecayStop(1e-9, 0, 10, 2, 20)
    for count, value in enumerate(values, 1):
        rule.check(count * 10, lambda value=value: value)
    assert rule.stop is None
