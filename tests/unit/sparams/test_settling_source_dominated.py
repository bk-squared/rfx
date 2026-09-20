"""A settling record on a source cell is not an independent witness (#1090).

``Result.settling_witness`` scores every user probe record by end/peak
energy and reports the worst. A record whose cell is also a drive cell has
the DRIVE PULSE as its peak, so its end/peak ratio measures how far the
source has turned off -- a number that barely moves with run length and
cannot fail. On the beam-steering fixture it read -79.4 dB at 20, 40 and 80
periods while a superstrate-edge probe read -23.6 / -45.3 / -77.9 dB.

These checks pin the marking, the selection rule and the qualifier. The
per-record ARITHMETIC is untouched and is pinned elsewhere
(``test_settling_witness.py``, ``test_forward_settling_witness.py``).
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._sparams import settling_verdict
from rfx.probes.settling import (
    SOURCE_DOMINATED_QUALIFIER,
    drive_cells,
    probe_record_info,
    probe_record_settling_witness,
    source_dominated_columns,
)


SOURCE = (.002, .002, .002)
AWAY = (.002, .002, .003)


def _sim(probe_positions, *, dft=True):
    sim = Simulation(freq_max=10e9, domain=(.004,) * 3, dx=.001,
                     boundary="cpml", cpml_layers=4)
    sim.add_source(SOURCE, "ez", amplitude_kind="field")
    for position in probe_positions:
        sim.add_probe(position, "ez")
    if dft:
        sim.add_dft_plane_probe(axis="z", coordinate=.002, component="ez",
                                n_freqs=3)
    return sim


def _run(sim, n_steps=60):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(n_steps=n_steps, compute_s_params=False,
                         skip_preflight=True)
    return result, [str(w.message) for w in caught]


def _records(*columns):
    return jnp.asarray(np.column_stack(columns).astype(np.float32))


# A record that never decays (0.0 dB, the worst possible score) and one that
# rings down hard. Synthetic records keep the SELECTION under test instead of
# the physics of a particular box.
FLAT = np.ones(200)
DECAYED = np.exp(-np.arange(200) / 6.0)


# --- 1. what counts as "the same cell" -------------------------------------

def test_co_location_is_decided_on_the_grid_node_spine():
    """Not dx-relative arithmetic: the same mapping that PLACES both."""
    sim = _sim([SOURCE, AWAY], dft=False)
    grid = sim._build_grid()
    cells = drive_cells(grid, sim._ports)
    assert cells == frozenset({grid.position_to_index(SOURCE)})
    assert grid.position_to_index(SOURCE) != grid.position_to_index(AWAY)
    assert source_dominated_columns(grid, sim._probes, cells) == frozenset({0})


def test_a_position_inside_the_same_cell_as_the_source_is_co_located():
    """A quarter-cell offset is the SAME Yee cell, so it is dominated."""
    sim = _sim([(.002, .002, .00225)], dft=False)
    grid = sim._build_grid()
    assert grid.position_to_index((.002, .002, .00225)) == grid.position_to_index(SOURCE)
    assert source_dominated_columns(
        grid, sim._probes, drive_cells(grid, sim._ports)) == frozenset({0})


def test_a_wire_port_extent_contributes_every_cell_it_spans():
    sim = Simulation(freq_max=10e9, domain=(.004,) * 3, dx=.001,
                     boundary="cpml", cpml_layers=4)
    sim.add_port(position=(.002, .002, .001), component="ez",
                 impedance=50.0, extent=.002)
    grid = sim._build_grid()
    i, j, k = grid.position_to_index((.002, .002, .001))
    assert drive_cells(grid, sim._ports) == frozenset(
        {(i, j, k), (i, j, k + 1), (i, j, k + 2)})


def test_no_grid_means_no_flags_rather_than_a_guess():
    sim = _sim([SOURCE], dft=False)
    assert drive_cells(None, sim._ports) == frozenset()
    assert source_dominated_columns(None, sim._probes, frozenset({(1, 2, 3)})) == frozenset()


# --- 2. the selection rule -------------------------------------------------

def test_the_only_record_on_the_source_cell_keeps_its_status_and_gains_the_qualifier():
    """Status and arithmetic unchanged; the number is labelled, not moved."""
    sim = _sim([SOURCE])
    result, _ = _run(sim)
    baseline = result.settling_witness["per_record_db"]["probe0(ez)"]
    witness = result.settling_witness
    assert witness["status"] == "measured" and witness["route"] == "probe_records"
    assert result.settling_db == pytest.approx(baseline, abs=0)
    assert witness["worst_record"] == "probe0(ez)"
    assert witness["source_dominated"] is True
    assert witness["source_dominated_records"] == ["probe0(ez)"]
    assert witness["qualifier"] == SOURCE_DOMINATED_QUALIFIER
    assert "not an independent ring-down witness" in witness["qualifier"]


def test_an_independent_record_takes_the_verdict_off_the_source_record():
    sim = _sim([SOURCE, AWAY])
    result, _ = _run(sim)
    # Real flags, synthetic records: the source column is the worse score,
    # so a selection that ignored the flag would report it.
    result = result._replace(time_series=_records(FLAT, DECAYED))
    witness = probe_record_settling_witness(
        result.time_series,
        sim._settling_probe_selection(result))[1]
    assert witness["per_record_db"]["probe0(ez)"] > witness["per_record_db"]["probe1(ez)"]
    assert witness["worst_record"] == "probe1(ez)"
    assert witness["source_dominated"] is False
    assert witness["source_dominated_records"] == ["probe0(ez)"]
    assert witness["qualifier"] == ""


def test_all_records_dominated_still_reports_the_worst_of_them():
    sim = _sim([SOURCE, (.002, .002, .00225)])
    result, _ = _run(sim)
    result = result._replace(time_series=_records(FLAT, DECAYED))
    value, witness = probe_record_settling_witness(
        result.time_series, sim._settling_probe_selection(result))
    assert witness["source_dominated_records"] == ["probe0(ez)", "probe1(ez)"]
    assert witness["worst_record"] == "probe0(ez)"
    assert value == pytest.approx(witness["per_record_db"]["probe0(ez)"], abs=0)
    assert witness["source_dominated"] is True


def test_an_independent_only_fixture_is_untouched():
    sim = _sim([AWAY])
    result, _ = _run(sim)
    witness = result.settling_witness
    assert witness["source_dominated"] is False
    assert witness["source_dominated_records"] == []
    assert witness["qualifier"] == ""
    assert witness["worst_record"] == "probe0(ez)"


# --- 3. what the user is told ---------------------------------------------

def test_a_source_dominated_pass_says_so_at_runtime():
    sim = _sim([SOURCE])
    result, messages = _run(sim, n_steps=800)
    assert settling_verdict(result.settling_db) == "pass"
    spoken = [m for m in messages if "source-dominated" in m]
    assert len(spoken) == 1, messages
    assert "is a source turn-off measurement" in spoken[0]
    assert "probe0(ez)" in spoken[0]


def test_an_independent_pass_is_quiet():
    sim = _sim([SOURCE, AWAY])
    result, messages = _run(sim, n_steps=800)
    assert settling_verdict(result.settling_db) == "pass"
    assert result.settling_witness["source_dominated"] is False
    assert not [m for m in messages if "source-dominated" in m]


def test_the_preflight_advice_names_the_drive_cell():
    sim = _sim([], dft=True)
    report = sim.preflight()
    spoken = [str(issue) for issue in
              report.by_code("settling_witness_will_be_absent")]
    assert len(spoken) == 1, [issue.code for issue in report]
    assert "NOT on a source/port drive cell" in spoken[0]
    assert "#1090" in spoken[0]


# --- 4. metadata ----------------------------------------------------------

def test_the_flag_rides_in_the_numeric_selection_metadata():
    sim = _sim([SOURCE, AWAY])
    result, _ = _run(sim)
    info = sim._settling_probe_selection(result)
    assert tuple(tuple(map(int, item)) for item in info) == ((0, 2, 1), (1, 2, 0))


def test_metadata_written_before_this_flag_existed_still_scores():
    """A 2-tuple selection is pre-#1090 and marks nothing dominated."""
    series = _records(FLAT, DECAYED)
    value, witness = probe_record_settling_witness(series, ((0, 2), (1, 2)))
    assert witness["worst_record"] == "probe0(ez)"
    assert witness["source_dominated"] is False
    assert witness["source_dominated_records"] == []
    assert np.isfinite(value)


def test_probe_record_info_marks_only_the_named_columns():
    entries = _sim([SOURCE, AWAY], dft=False)._probes
    series = _records(FLAT, DECAYED)
    assert probe_record_info(series, entries) == ((0, 2, 0), (1, 2, 0))
    assert probe_record_info(series, entries,
                             source_dominated={1}) == ((0, 2, 0), (1, 2, 1))


def test_a_backend_fallback_record_still_supplies_no_witness():
    """No user probe means no witness at all -- unchanged by #1090."""
    sim = _sim([])
    result, _ = _run(sim)
    assert result.settling_db is None
    assert result.settling_witness["status"] == "absent"
    assert result.settling_witness["source_dominated"] is False
