"""Forward settling diagnostics preserve records and survive JIT boundaries.

The tiny open box only distinguishes truncated, decayed and absent records;
these checks do not qualify an antenna, a mode or an RF extraction.
"""
import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._spec import ForwardResult
from rfx.api._sparams import settling_verdict


def _sim(*, probe=True, dft=False, ntff=False):
    sim = Simulation(freq_max=10e9, domain=(.004,) * 3, dx=.001,
                     boundary="cpml", cpml_layers=4)
    sim.add_source((.002,) * 3, "ez", amplitude_kind="field")
    if probe:
        sim.add_probe((.002, .002, .003), "ez")
    if dft:
        sim.add_dft_plane_probe(axis="z", coordinate=.002, component="ez", n_freqs=3)
    if ntff:
        sim.add_ntff_box((.001,) * 3, (.003,) * 3, freqs=np.array([5e9]))
    return sim


def _captured(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, caught


def _scoped_warnings(caught):
    return [str(w.message) for w in caught
            if "witness FAILED" in str(w.message)
            or "no ring-down settling witness" in str(w.message)]


def _independent_db(record):
    # A numerical oracle from returned samples, not the production scorer.
    power = np.abs(np.asarray(record).astype(np.float64)) ** 2
    tail = max(1, len(power) // 10)
    tiny = np.finfo(np.float64).tiny
    return 10 * np.log10((power[-tail:].mean() + tiny) / (power.max() + tiny))


def _assert_absent(result):
    assert result.settling_db is None
    witness = result.settling_witness
    assert witness["status"] == "absent" and witness["route"] is None
    assert witness["worst_record"] is None and witness["per_record_db"] == {}
    assert witness["reason"]
    assert settling_verdict(result.settling_db) == "absent"
    json.dumps(witness, allow_nan=False)


@pytest.fixture(scope="module")
def short_pair():
    # Both consumers requested together must still emit one aggregate warning.
    sim = _sim(dft=True, ntff=True)
    run, _ = _captured(lambda: sim.run(n_steps=60, compute_s_params=False, skip_preflight=True))
    forward, caught = _captured(lambda: sim.forward(n_steps=60, skip_preflight=True))
    return run, forward, caught


def test_short_forward_has_independent_numeric_witness_and_one_scoped_warning(short_pair):
    _, result, caught = short_pair
    expected = _independent_db(np.asarray(result.time_series)[:, 0])
    assert expected > -40 and settling_verdict(result.settling_db) == "fail"
    assert result.settling_db == pytest.approx(expected, abs=1e-10)
    witness = result.settling_witness
    assert witness["status"] == "measured" and witness["route"] == "probe_records"
    assert witness["worst_record"] == "probe0(ez)"
    assert witness["per_record_db"] == {"probe0(ez)": pytest.approx(expected, abs=1e-10)}
    assert witness["skipped_records"] == []
    messages = _scoped_warnings(caught)
    assert len(messages) == 1, messages
    assert "witness FAILED" in messages[0] and "probe0(ez)" in messages[0]
    assert "n_steps=60" in messages[0]
    json.dumps(witness, allow_nan=False)


def test_run_and_forward_score_the_same_record_and_labels(short_pair):
    run, forward, _ = short_pair
    # The unchanged baseline (99d0b5a0) already differs by up to 5.96e-8
    # between these float32 field records, even with checkpoint=False on
    # both lanes. Freeze the diagnostic's same-record contract separately
    # from field-lane equivalence; do not hide that difference with a wider
    # field tolerance.
    assert run.settling_db == pytest.approx(
        _independent_db(np.asarray(run.time_series)[:, 0]), abs=1e-10)
    same_record = forward._replace(time_series=run.time_series)
    assert same_record.settling_db == run.settling_db
    assert same_record.settling_witness == run.settling_witness
    assert tuple(tuple(map(int, item)) for item in forward.settling_probe_info) == ((0, 2),)


def test_diagnostic_finalization_does_not_change_recorded_fields(short_pair, monkeypatch):
    _, enabled, _ = short_pair
    sim = _sim(dft=True, ntff=True)
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **kwargs: result)
    bypassed, _ = _captured(lambda: sim.forward(n_steps=60, skip_preflight=True))
    for name in ("time_series", "ntff_data", "dft_planes"):
        left, left_tree = jax.tree_util.tree_flatten(getattr(enabled, name))
        right, right_tree = jax.tree_util.tree_flatten(getattr(bypassed, name))
        assert left_tree == right_tree
        for actual, original in zip(left, right):
            np.testing.assert_array_equal(actual, original)


def test_repeated_lazy_reads_do_not_reemit_forward_warnings(short_pair):
    _, result, _ = short_pair
    before = np.asarray(result.time_series).copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            assert result.settling_db is not None
            assert result.settling_witness["status"] == "measured"
    assert not caught, [str(w.message) for w in caught]
    np.testing.assert_array_equal(result.time_series, before)


def test_longer_actual_forward_has_a_decayed_probe_record():
    result, caught = _captured(lambda: _sim(dft=True).forward(n_steps=800, skip_preflight=True))
    expected = _independent_db(np.asarray(result.time_series)[:, 0])
    assert expected < -40
    assert result.settling_db == pytest.approx(expected, abs=1e-10)
    assert settling_verdict(result.settling_db) == "pass"
    assert not _scoped_warnings(caught)


@pytest.mark.parametrize("dft", [False, True])
def test_actual_probeless_forward_is_absent_and_only_dft_scope_warns(dft):
    result, caught = _captured(lambda: _sim(probe=False, dft=dft).forward(
        n_steps=60, skip_preflight=True))
    _assert_absent(result)
    messages = _scoped_warnings(caught)
    assert len(messages) == int(dft), messages
    if dft:
        assert "no ring-down settling witness" in messages[0]
        assert "unguarded" in messages[0]
    with warnings.catch_warnings(record=True) as repeated:
        warnings.simplefilter("always")
        _assert_absent(result)
        _assert_absent(result)
    assert not repeated, [str(w.message) for w in repeated]


def test_bare_short_forward_exposes_failure_without_scoped_dft_warning():
    result, caught = _captured(lambda: _sim().forward(n_steps=60, skip_preflight=True))
    assert settling_verdict(result.settling_db) == "fail"
    assert result.settling_witness["status"] == "measured"
    assert not _scoped_warnings(caught)


def test_actual_internal_only_probes_do_not_supply_user_coverage():
    sim = _sim(dft=True)
    sim._internal_probe_indices = {0}
    result, caught = _captured(lambda: sim.forward(n_steps=60, skip_preflight=True))
    assert np.max(np.abs(np.asarray(result.time_series))) > 0
    assert _independent_db(np.asarray(result.time_series)[:, 0]) > -40
    assert result.settling_probe_info == ()
    _assert_absent(result)
    # The enclosing driver owns this internal witness and its warnings.
    assert not _scoped_warnings(caught)


def test_public_metadata_excludes_internal_columns_and_outlives_sim_changes():
    sim = _sim()
    sim.add_probe((.003, .002, .002), "hy")
    sim._internal_probe_indices = {0}
    result, _ = _captured(lambda: sim.forward(n_steps=10, skip_preflight=True))
    assert tuple(tuple(map(int, item)) for item in result.settling_probe_info) == ((1, 4),)
    # Retain the real public metadata, but supply discriminating records:
    # the excluded internal column never decays; the selected one does.
    t = np.arange(100)
    records = np.column_stack((np.ones(100), np.exp(-t / 8))).astype(np.float32)
    result = result._replace(time_series=jnp.asarray(records))
    sim._probes.clear()
    sim._internal_probe_indices.clear()
    assert result.settling_db == pytest.approx(_independent_db(records[:, 1]), abs=1e-10)
    assert result.settling_witness["worst_record"] == "probe1(hy)"
    assert set(result.settling_witness["per_record_db"]) == {"probe1(hy)"}


@pytest.mark.parametrize("record", [
    None,
    np.empty((0, 1), np.float32),
    np.empty((100, 0), np.float32),
    np.ones((9, 1), np.float32),
    np.zeros((100, 1), np.float32),
    np.full((100, 1), 1e-37, np.float32),
])
def test_empty_short_or_underflowed_legacy_records_are_absent_and_lazy_reads_are_quiet(record):
    result = ForwardResult(time_series=record)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _assert_absent(result)
        _assert_absent(result)
    assert not caught, [str(w.message) for w in caught]


def test_underflowed_column_is_named_and_cannot_hide_a_live_record():
    t = np.arange(100)
    records = np.column_stack((np.exp(-t / 8), np.full(100, 1e-37))).astype(np.float32)
    result = ForwardResult(time_series=jnp.asarray(records),
                           settling_probe_info=((0, 2), (1, 4)))
    assert result.settling_db == pytest.approx(_independent_db(records[:, 0]), abs=1e-10)
    assert result.settling_witness["skipped_records"] == ["probe1(hy)"]
    assert set(result.settling_witness["per_record_db"]) == {"probe0(ez)"}


@pytest.mark.parametrize("bad_value", [np.nan, np.inf], ids=["nan", "inf"])
@pytest.mark.parametrize("bad_first", [False, True], ids=["good-first", "bad-first"])
def test_selected_nonfinite_record_invalidates_settled_companion_through_actual_finalizer(
    bad_value, bad_first,
):
    sim = _sim(dft=True)
    sim.add_probe((.003, .002, .002), "hy")
    sim.add_probe((.002, .003, .002), "hz")
    good = np.exp(-np.arange(100) / 8).astype(np.float32)
    assert _independent_db(good) < -40
    bad = good.copy()
    bad[15] = bad_value
    columns = [bad, good] if bad_first else [good, bad]
    # Invalid samples must not be confused with a separate underfloor skip.
    records = np.column_stack((*columns, np.full(100, 1e-37, np.float32)))
    result, caught = _captured(lambda: sim._attach_run_settling_witness(
        ForwardResult(time_series=jnp.asarray(records)), n_steps=100, context="forward"))
    _assert_absent(result)
    name = "probe0(ez)" if bad_first else "probe1(hy)"
    witness = result.settling_witness
    assert set(witness["invalid_records"]) == {name}
    assert "non-finite" in witness["invalid_records"][name]
    assert name in witness["reason"]
    assert witness["skipped_records"] == ["probe2(hz)"]
    assert len(_scoped_warnings(caught)) == 1
    assert "no ring-down settling witness" in _scoped_warnings(caught)[0]
    assert not any("witness FAILED" in str(w.message) for w in caught)
    invalid_messages = [str(w.message) for w in caught if "INVALID RECORDS" in str(w.message)]
    assert len(invalid_messages) == 1 and name in invalid_messages[0]
    with warnings.catch_warnings(record=True) as repeated:
        warnings.simplefilter("always")
        for _ in range(3):
            _assert_absent(result)
            assert result.settling_witness["invalid_records"] == witness["invalid_records"]
    assert not repeated, [str(w.message) for w in repeated]
    np.testing.assert_array_equal(result.time_series, records)


def test_none_metadata_scores_generic_columns_but_empty_selection_is_absent():
    records = np.exp(-np.arange(100) / 8).astype(np.float32)[:, None]
    generic = ForwardResult(time_series=jnp.asarray(records))
    assert generic.settling_db == pytest.approx(_independent_db(records[:, 0]), abs=1e-10)
    assert len(generic.settling_witness["per_record_db"]) == 1
    _assert_absent(generic._replace(settling_probe_info=()))


def test_full_forward_result_crosses_jit_then_scores_concrete_arrays():
    records = np.column_stack((np.ones(100), np.exp(-np.arange(100) / 8))).astype(np.float32)
    original = ForwardResult(time_series=jnp.asarray(records), settling_probe_info=((1, 5),))

    @jax.jit
    def roundtrip(result):
        # Diagnostics are absent while the samples are tracers; inspecting
        # them must not force host conversion or insert strings in the tree.
        assert result.settling_db is None
        assert result.settling_witness["status"] == "absent"
        return result._replace(time_series=result.time_series * 2)

    concrete = roundtrip(original)
    assert isinstance(concrete, ForwardResult) and concrete.grid is None
    np.testing.assert_array_equal(concrete.time_series, 2 * records)
    assert concrete.settling_db == pytest.approx(_independent_db(records[:, 1]), abs=1e-10)
    assert concrete.settling_witness["worst_record"] == "probe1(hz)"
    assert all(np.asarray(leaf).dtype.kind in "biufc" for leaf in jax.tree_util.tree_leaves(concrete))
    # A second complete-object JIT boundary also accepts the now-device-backed
    # numeric metadata; metadata must not depend on Python scalar identity.
    again = jax.jit(lambda value: value)(concrete)
    assert again.settling_witness == concrete.settling_witness


def test_actual_public_forward_returns_numeric_carrier_through_outer_jit():
    sim = _sim(dft=True)

    @jax.jit
    def recorded_forward():
        result = sim.forward(n_steps=60, skip_preflight=True)
        assert result.settling_db is None
        assert result.settling_witness["status"] == "absent"
        # Grid retains its existing non-JAX restriction. Return the actual
        # recorded samples and new numeric provenance, not the whole result.
        return result.time_series, result.settling_probe_info

    (samples, info), caught = _captured(recorded_forward)
    assert samples.shape == (60, 1)
    assert tuple(tuple(map(int, item)) for item in info) == ((0, 2),)
    assert not _scoped_warnings(caught)
    host_result = ForwardResult(time_series=samples, settling_probe_info=info)
    expected = _independent_db(np.asarray(samples)[:, 0])
    assert host_result.settling_db == pytest.approx(expected, abs=1e-10)
    assert settling_verdict(host_result.settling_db) == "fail"
    assert host_result.settling_witness["worst_record"] == "probe0(ez)"
    json.dumps(host_result.settling_witness, allow_nan=False)
