"""Guard the #726 experiment against the confounds in its historical arms.

Build-only geometry and synthetic reporting tests; no RF accuracy claim.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/diagnostics/msl_probe_clearance_bias.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("_clearance_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def inputs(driver):
    from unittest.mock import patch
    from rfx import Simulation

    def forbidden(*args, **kwargs):
        raise AssertionError("field solve during input certification")

    with patch.object(Simulation, "run", forbidden), \
            patch.object(Simulation, "forward", forbidden), \
            patch.object(Simulation, "compute_msl_s_matrix", forbidden):
        cv, arms, receipt = driver.prepare_inputs()
        rz = driver._realized(arms["clean"])
    return cv, arms, receipt, rz


def test_only_observation_offset_changes(driver, inputs):
    cv, arms, receipt, rz = inputs
    clean, near = arms["clean"], arms["near"]
    # Compare every input in each port record, rather than a hand-picked
    # subset of source properties. Removing just the observation offset
    # must make the two complete records equal.
    for a, b in zip(clean._msl_ports, near._msl_ports):
        da, db = asdict(a), asdict(b)
        da.pop("n_probe_offset")
        db.pop("n_probe_offset")
        assert da == db
    assert clean._msl_ports[1] == near._msl_ports[1]
    assert repr(clean._geometry) == repr(near._geometry)
    assert clean._boundary == near._boundary
    assert clean._domain == near._domain
    assert clean._msl_auto_offset_min == near._msl_auto_offset_min == {}
    from rfx.api._sparams import _resolve_msl_auto_offsets
    for sim in arms.values():
        assert _resolve_msl_auto_offsets(sim, sim._msl_ports, rz.grid) == sim._msl_ports
    ports = receipt["ports"]
    assert ports["clean"][0]["deepest_gap_m"] > ports["clean"][0]["recommended_gap_m"]
    assert ports["near"][0]["deepest_gap_m"] == pytest.approx(cv.DX)
    assert ports["near"][0]["first_gap_m"] > cv.DX
    assert receipt["exact_full_wave_truth"] is False


@pytest.mark.parametrize("defect", ["touch", "cross", "source", "clamp"])
def test_invalid_ladder_refused_before_field_solve(driver, inputs, defect):
    _, arms, receipt, rz = inputs
    sim = copy.deepcopy(arms["near"])
    p = sim._msl_ports[0]
    offset = dict(touch=p.n_probe_offset + 1, cross=p.n_probe_offset + 2,
                  source=1, clamp=100000)[defect]
    sim._msl_ports[0] = replace(p, n_probe_offset=offset)
    with pytest.raises(ValueError, match="touches|crosses|standoff|clamps"):
        driver._certificate(sim, rz, receipt["stub_bounds_m"])


def _result():
    f = np.array([3.1e9, 3.7e9, 4.3e9, 4.9e9])
    s = np.full((2, 2, 4), 0.5 + 0j)
    s[1, 0] = [0.6, 0.05, 0.4, 0.7]
    return SimpleNamespace(freqs=f, S=s, Z0=np.ones((2, 4)) * 50,
                           settling_db=np.array([-60., -55.]),
                           reliable=np.ones((2, 4), dtype=bool))


@pytest.mark.parametrize("settling", [None, [np.nan, -60], [-39.9, -60], [0., 0.]])
def test_unsettled_or_unknown_results_never_get_a_numeric_verdict(driver, settling):
    clean, near = _result(), _result()
    near.settling_db = settling
    result = driver.compare(clean, near)
    assert result["status"] == "not_read"
    assert "delta_s11_db" not in result


def test_comparison_uses_same_control_notch_bin_without_truth_claim(driver):
    clean, near = _result(), _result()
    near.S[0, 0, 1] = 0.25
    near.S[1, 0] = [0.6, 0.2, 0.01, 0.7]  # different near-arm minimum
    result = driver.compare(clean, near)
    assert result["notch_bin_hz"] == 3.7e9
    assert result["delta_s11_db"] == pytest.approx(20 * np.log10(0.5))
    assert result["status"] == "paired_observation"
    assert "no universal bias" in result["interpretation"]


def test_low_signal_notch_is_not_reported(driver):
    clean, near = _result(), _result()
    near.reliable[1, 1] = False
    assert driver.compare(clean, near)["status"] == "not_read"


def test_record_is_saved_raw_and_failed_control_skips_second_arm(driver, monkeypatch, tmp_path):
    calls = []
    result = _result()
    result.settling_db[0] = -10

    def fake_solve(**kwargs):
        calls.append(kwargs)
        # A value above unity proves that the diagnostic does not clip its
        # saved result even though it refuses to interpret this record.
        result.S[0, 0, 1] = 1.2
        Path(kwargs["raw_3probe_dump_path"]).write_bytes(b"synthetic phasor receipt")
        return result

    sim = SimpleNamespace(compute_msl_s_matrix=fake_solve)
    monkeypatch.setattr(driver, "prepare_inputs", lambda: (
        SimpleNamespace(F_MAX=7e9), {"clean": sim, "near": sim}, {}))
    monkeypatch.setattr("sys.argv", ["x", "--out-dir", str(tmp_path)])
    assert driver.main() == 2
    assert len(calls) == 1
    assert calls[0]["enforce_passivity"] is False
    verdict = json.loads((tmp_path / "comparison.json").read_text())
    assert verdict["status"] == "not_read"
    assert "near arm skipped" in verdict["reason"]
    with np.load(tmp_path / "clean-result.npz") as data:
        assert data["S"][0, 0, 1] == 1.2
    assert (tmp_path / "clean-phasors.npz").is_file()
    with pytest.raises(FileExistsError):
        driver.main()
    assert len(calls) == 1


def test_voltage_path_pec_is_refused_even_before_the_junction(driver, inputs):
    _, arms, receipt, rz = inputs
    sim = arms["near"]
    entry = sim._msl_ports[0]
    span = driver.msl_cross_section_span(rz.grid, driver.msl_port_from_entry(entry))
    i = span["i_feed"] + entry.n_probe_offset
    masks = list(rz.edge_masks)
    masks[2] = np.array(masks[2], copy=True)
    masks[2][i, span["w_centre"], span["n_lo"]] = True
    bad = SimpleNamespace(grid=rz.grid, edge_masks=masks)
    with pytest.raises(ValueError, match="voltage path intersects realized PEC"):
        driver._certificate(sim, bad, receipt["stub_bounds_m"])


def test_retired_short_does_not_launch_another_invalid_experiment(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_retired_short", ROOT / "scripts/diagnostics/msl_probe_clearance_shorted_line.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def forbidden(*args, **kwargs):
        raise AssertionError("retired experiment tried to build or solve")

    monkeypatch.setattr(mod, "build", forbidden)
    with pytest.raises(SystemExit, match="Retired #726 accuracy fixture"):
        mod.main()
