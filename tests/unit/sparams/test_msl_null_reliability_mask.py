"""Regression coverage for the MSL standing-wave-null reliability mask."""

from pathlib import Path
import json

import numpy as np
import pytest

from rfx.api._spec import MSLSMatrixResult
from rfx.api._sparams import (
    _msl_wave_split_reliability,
    _warn_msl_wave_split_unreliable,
)
from rfx.validation import load_port_vi_dump_npz


_ROOT = Path(__file__).resolve().parents[3]


def _dump_path(name: str) -> Path:
    candidates = (
        _ROOT / "scripts" / "diagnostics" / "r45_vi_dumps" / name,
        _ROOT / "tests" / "fixtures" / "msl_null_mask" / name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip(f"real R45 regression dump is not present: {name}")


def _reliability_from_dump(name: str):
    # These historical v2 records lack actual S references. The low-signal
    # test needs only measured V/I; do not invent references to replay S.
    with np.load(_dump_path(name), allow_pickle=False) as dump:
        reliable = _msl_wave_split_reliability(
            dump['raw_v'][:, 0, 0, :], dump['raw_i1'][:, 0, :], dump['freqs_hz']
        )
        return dump['production_smatrix'], reliable


def test_dx98_real_dump_flags_null_and_nonpassive_peak():
    stored_s, reliable = _reliability_from_dump("vi_dump_dx98um.npz")
    unreliable = ~reliable[0]
    assert np.any(unreliable[57:60])

    s11 = np.abs(stored_s[0, 0])
    peak = int(np.argmax(s11))
    assert s11[peak] > 1.0
    assert unreliable[peak]


def test_dx197_real_dump_has_at_most_two_null_bins():
    _dump, reliable = _reliability_from_dump("vi_dump_dx197um.npz")
    assert np.count_nonzero(~reliable) <= 2


def test_result_plumbing_preserves_s_and_carries_mask():
    freqs = np.array([7.0e9, 8.0e9, 8.45e9, 9.0e9])
    voltage = np.array([[1.0, 0.8, 0.01, 1.2]])
    current = np.array([[1.1, 0.9, 0.02, 1.0]])
    reliable = _msl_wave_split_reliability(voltage, current, freqs)
    s = np.array([[[0.2, 0.8, 1.358, 0.3]]])

    result = MSLSMatrixResult(
        S=s.copy(),
        freqs=freqs,
        Z0=np.full((1, 4), 50.0),
        beta=np.ones(4),
        port_names=("msl_0",),
        reliable=reliable,
    )

    np.testing.assert_array_equal(result.reliable, [[True, True, False, True]])
    np.testing.assert_array_equal(result.S, s)


def test_loader_exposes_probe_zero_from_msl_raw_dump(tmp_path):
    path = tmp_path / "msl_raw.npz"
    raw_v = np.ones((1, 1, 3, 4), dtype=complex)
    raw_v[:, :, 0, 2] = 0.01
    raw_i1 = np.ones((1, 1, 4), dtype=complex)
    raw_i1[:, :, 2] = 0.01
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps({
            "schema": "rfx.msl_nprobe_dump", "schema_version": 4,
            "s_wave_convention": "power",
            "s_reference_impedances_ohm": [50.0],
            "port_definitions": [{"impedance_ohm": 50.0}],
        })),
        freqs_hz=np.arange(4, dtype=float),
        raw_v=raw_v,
        raw_i1=raw_i1,
        production_smatrix=np.zeros((1, 1, 4), dtype=complex),
        port_names=np.asarray(["msl_0"], dtype=object),
        driven_port_indices=np.asarray([0]),
    )

    dump = load_port_vi_dump_npz(path)
    assert dump.voltages.shape == (1, 1, 4)
    assert dump.voltages[0, 0, 2] == 0.01
    assert dump.currents[0, 0, 2] == 0.01


def test_voltage_only_dip_remains_reliable():
    reliable = _msl_wave_split_reliability(
        [[1.0, 0.01, 1.0]], [[1.0, 0.8, 1.0]], [1.0, 2.0, 3.0]
    )
    np.testing.assert_array_equal(reliable, True)


def test_unreliable_bins_emit_one_aggregate_warning():
    reliable = np.array([[True, False, False], [True, True, False]])
    with pytest.warns(UserWarning) as recorded:
        _warn_msl_wave_split_unreliable(
            reliable, np.array([8.4e9, 8.45e9, 8.5e9])
        )
    assert len(recorded) == 1
    assert str(recorded[0].message) == (
        "low signal at an MSL port plane: 2 bins in [8.4500, 8.5000] GHz "
        "have both |V| and |I| below 10% of their record's band medians. "
        "This flags relative signal strength, not proof of an incorrect "
        "S-matrix; a true transmission zero can also trigger it. Check "
        "signal uncertainty, settling, drive conditioning and probe "
        "geometry before using these bins."
    )
