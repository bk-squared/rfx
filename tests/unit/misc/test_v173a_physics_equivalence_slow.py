"""Aligned V173A composition lock, with no antenna-matching claim.

The old volume, off-lattice board and unsupported S11 statistic are retired.
The historical JSON remains an archive; the replacement has its own drawing,
instrument and named capture. Frequency rtol=1e-6 and spectrum atol=0.1 dB
are unchanged drift tolerances, not physical-accuracy envelopes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.harnesses import v173a_physics_equivalence as v173a
from tests._realized_geometry import node_index, realized

BASELINE_PATH = Path(__file__).resolve().parents[2] / "data" / "v173a_aligned_composition.json"


def test_v173a_drawing_and_gap_measurements_are_realized():
    sim = v173a._build_sim()
    rz = realized(sim)
    assert rz.pec_mask is None
    assert len(rz.sheets) == 1
    sheet = rz.sheets[0]
    assert sheet.plane == node_index(rz.grid, 2, v173a._SUB_T)
    assert rz.wall_planes(2) == [sheet.plane]
    # Independent arithmetic: every physical face is an integer node index.
    for faces in ((0., .011, .039, .05), (0., .007, .043, .05), (0., .0015, .02)):
        np.testing.assert_allclose(np.array(faces) / v173a._DX,
                                   np.rint(np.array(faces) / v173a._DX), atol=1e-12, rtol=0)
    assert np.count_nonzero(rz.edge_masks[0]) == 56 * 73
    assert np.count_nonzero(rz.edge_masks[1]) == 57 * 72
    assert not np.any(rz.edge_masks[2])
    materials = sim._assemble_materials(rz.grid, pec_sheets=[], pec_wires=[])[0]
    i, j = (node_index(rz.grid, a, p) for a, p in enumerate(v173a._SRC_POS[:2]))
    lower = node_index(rz.grid, 2, 0.)
    upper = node_index(rz.grid, 2, v173a._SUB_T)
    np.testing.assert_allclose(np.asarray(materials.eps_r)[i, j, lower:upper], v173a._FR4_EPS)
    assert float(materials.eps_r[i, j, upper]) == 1.
    for position in (v173a._SRC_POS, v173a._PROBE_POS):
        idx = tuple(node_index(rz.grid, a, p) for a, p in enumerate(position))
        assert not rz.edge_masks[2][idx]
        assert 0 < position[2] + v173a._DX / 2 < v173a._SUB_T


def test_probe_spectrum_statistic_exposes_source_amplitude_dependence():
    # Falsifies the retired S11 interpretation: doubling a field trace raises
    # this statistic by 6.0206 dB. True linear reflection is scale-invariant.
    dt = 1e-11
    time = np.arange(2048) * dt
    trace = np.cos(2 * np.pi * 2.1e9 * time) * np.exp(-time / 1e-8)
    first, freq = v173a._probe_spectrum_min(trace, dt)
    doubled, doubled_freq = v173a._probe_spectrum_min(2 * trace, dt)
    assert doubled - first == pytest.approx(20 * np.log10(2), abs=1e-12)
    assert doubled_freq == freq
    with pytest.raises(ValueError, match="finite and nonzero"):
        v173a._probe_spectrum_min(np.zeros_like(trace), dt)


def test_mode_estimator_reads_ringdown_after_the_driven_transient():
    dt = 5e-12
    time = np.arange(1600) * dt
    wave = v173a._WAVEFORM
    arg = (time - wave.t0) / wave.tau
    forced = 1e3 * (-2 * arg) * np.exp(-arg**2)
    trace = forced + np.cos(2 * np.pi * 2.1e9 * time) * np.exp(-time / 1e-8)
    measured = v173a._measure(trace, dt)
    assert measured["dominant_probe_mode_hz"] == pytest.approx(2.1e9, rel=1e-3)
    assert abs(float(wave(measured["ringdown_start_s"]))) < 1e-12
    with pytest.raises(ValueError, match="before the source-free"):
        v173a._measure(trace[:100], dt)


@pytest.mark.slow_physics
def test_v173a_baseline_bit_identity():
    """Historical node ID retained; now an aligned synthetic composition lock."""
    baseline = json.loads(BASELINE_PATH.read_text())
    actual, _trace = v173a.run_capture()
    assert actual["fixture"] == baseline["fixture"]
    assert actual["n_steps"] == baseline["n_steps"]
    assert actual["ringdown_start_s"] == baseline["ringdown_start_s"]
    print(json.dumps(actual, indent=2))
    freq = actual["dominant_probe_mode_hz"]
    expected = baseline["dominant_probe_mode_hz"]
    assert abs(freq - expected) <= 1e-6 * expected, actual
    assert abs(actual["probe_spectrum_min_db"] - baseline["probe_spectrum_min_db"]) <= 0.1, actual
