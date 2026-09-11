"""cv02 must judge the retained analysis span, without executing its solver."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rfx.harminv import harminv_record_duration
from validation.crossval.comparators import ring_mode_judge

SCRIPT = Path(__file__).resolve().parents[2] / "validation/crossval/02_ring_resonator.py"


def _assignment(tree, name):
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def _execute_assignment(tree, name, namespace):
    code = ast.Module(body=[_assignment(tree, name)], type_ignores=[])
    exec(compile(code, str(SCRIPT), "exec"), namespace)


def test_cv02_judge_receives_the_filtered_duration():
    tree = ast.parse(SCRIPT.read_text())
    signal, dt, fmax = np.zeros(4096), 1 / (128e9), 1.4e9
    received = []

    def judge(_reference, _solver, duration, **kwargs):
        received.append(duration)

    ns = dict(signal=signal, dt=dt, fmax_hz=fmax, C0=299792458.0, a=1e-6,
              harminv_record_duration=harminv_record_duration,
              ring_mode_judge=SimpleNamespace(judge=judge), reference_modes=[],
              solver_modes=[], fmin_meep=0.1, fmax_meep=0.2)
    for name in ("analysis_duration", "record_T_meep", "verdict"):
        _execute_assignment(tree, name, ns)
    want = harminv_record_duration(len(signal), dt, fmax) * ns["C0"] / ns["a"]
    assert received == pytest.approx([want])
    assert want < (len(signal) - 1) * dt * ns["C0"] / ns["a"]


def test_cv02_ladder_admission_uses_the_filtered_duration():
    tree = ast.parse(SCRIPT.read_text())
    ts = np.zeros(4300)
    ns = dict(ts_r=ts, skip_r=204, dt_r=1 / 128e9, fmax_hz=1.4e9,
              harminv_record_duration=harminv_record_duration)
    _execute_assignment(tree, "free_r", ns)
    want = harminv_record_duration(4096, ns["dt_r"], ns["fmax_hz"])
    assert ns["free_r"] == want
    _execute_assignment(tree, "discarded_r", ns)
    assert ns["free_r"] + ns["discarded_r"] == pytest.approx(4095 * ns["dt_r"])


def test_cv02_settling_flag_uses_analysis_support_and_keeps_raw_measurement(capsys):
    tree = ast.parse(SCRIPT.read_text())
    n, dt, fmax = 4096, 1 / 128e9, 1.4e9
    raw_duration = (n - 1) * dt
    fit_duration = harminv_record_duration(n, dt, fmax)
    # Only the discarded time puts this mode above the existing Q floor.
    tau = (raw_duration + fit_duration) / (2 * ring_mode_judge.Q_RECORD_MIN_EFOLDS)
    freq, Q = 1e9, np.pi * 1e9 * tau
    times = np.arange(n) * dt
    signal = np.exp(-times / tau) * np.cos(2 * np.pi * freq * times)
    offset = 2e-9
    ns = dict(signal=signal, dt=dt, fmax_hz=fmax,
              harminv_record_duration=harminv_record_duration,
              ring_mode_judge=ring_mode_judge,
              witness_modes=[SimpleNamespace(freq=freq, Q=Q)],
              source_off_time=3e-9, peak_offset_after_source=offset,
              C0=299792458.0, a=1e-6)
    for name in ("analysis_duration", "record_after_source", "settling_rows", "signal_db"):
        _execute_assignment(tree, name, ns)
    (row,) = ns["settling_rows"]
    assert not row.observed
    assert row.t_over_tau == pytest.approx(fit_duration / tau)
    assert ring_mode_judge.mode_settling(freq, Q, raw_duration).observed
    assert ns["record_after_source"] == raw_duration
    assert ns["signal_db"] == ring_mode_judge.signal_settling_db(signal)

    (report,) = [node for node in ast.walk(tree)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "settling_rows"]
    exec(compile(ast.Module(body=[report], type_ignores=[]), str(SCRIPT), "exec"), ns)
    text = capsys.readouterr().out
    assert "truncation-susp" in text
    assert f"pole-fit span T = {fit_duration:.3e}" in text
    assert f"raw whole-signal span = {raw_duration:.3e}" in text
    assert f"starting {offset:.3e} after source-off" in text
    assert f"ALREADY-DECAYED peak {offset:.3e}" in text
    assert ns["need_gate"] == pytest.approx(ring_mode_judge.Q_RECORD_MIN_EFOLDS * tau)
    assert f"usable pole-fit span of {ns['need_gate'] * ns['C0'] / ns['a']:.0f}" in text
