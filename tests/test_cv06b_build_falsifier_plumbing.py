"""Plumbing test for cv06b's GPU-lane build falsifier (#812).

``scripts/diagnostics/cv06b_build_falsifiers.py`` runs three 5,729,080-cell
solves. A crash in its reporting or JSON path AFTER those solves costs the run,
so the non-FDTD half is exercised here with the solve stubbed out.

This test asserts NOTHING about physics. It checks only that the summary is
written, that it carries the keys the design note and the lane's prose cite by
name, and that gate verdicts survive as JSON booleans rather than being coerced
to 1.0/0.0 by a ``default=`` fallback (``evaluate()`` returns ``np.bool_``
whenever the analytic anchor arrives as ``np.float64``, which it does).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts/diagnostics/cv06b_build_falsifiers.py"
FIXTURE = REPO_ROOT / "tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_json_is_written_and_keeps_boolean_gates(tmp_path, monkeypatch):
    mod = _load(BUILDER, "_cv06b_build_falsifiers")
    d = json.loads(FIXTURE.read_text())
    f = np.asarray(d["freqs_ghz"], dtype=float) * 1e9
    s21 = np.asarray(d["s21_mag"], dtype=float)
    z0 = np.full_like(f, float(d["re_z0_median_ohm"]))

    def fake_solve(cv, label):
        # np.float64 anchor on purpose: that is what the real path passes, and
        # it is what makes evaluate()'s gate values np.bool_.
        f_an = np.float64(3.711e9)
        m = cv.evaluate(f, s21, z0, f_an)
        m.update(label=label, solve_s=0.0, stub_len_m=float(cv.STUB_LEN),
                 w_stub_m=float(cv.W_STUB), freqs_hz=f.tolist(),
                 s21_mag=s21.tolist(), re_z0=z0.tolist())
        return m

    monkeypatch.setattr(mod, "solve", fake_solve)
    monkeypatch.setattr("sys.argv", ["x", "--out-dir", str(tmp_path)])
    rc = mod.main()
    assert rc in (0, 1)

    summary = json.loads((tmp_path / "cv06b_build_falsifiers_summary.json")
                         .read_text())
    for key in ("criterion_A_baseline", "stub_1cell", "stub_narrow", "verdict"):
        assert key in summary
    for key in ("err_pct", "bw_ratio", "witness_bins", "notch_depth_db",
                "f_notch_refined_hz", "z0_median_ohm"):
        assert isinstance(summary["criterion_A_baseline"][key], float)
    for v in summary["criterion_A_baseline"]["gates"].values():
        assert isinstance(v, bool)
    for v in summary["verdict"].values():
        assert isinstance(v, bool)
    for key in ("true_shift_pct", "true_shift_bins", "bin_argmin_delta_pct",
                "refined_delta_pct"):
        assert isinstance(summary["stub_1cell"][key], float)
    assert isinstance(summary["stub_narrow"]["G2_fired"], bool)
    # the per-leg dumps must survive too — they carry the raw sweeps
    for label in ("baseline", "stub_1cell", "stub_narrow"):
        leg = json.loads((tmp_path / f"cv06b_falsifier_{label}.json").read_text())
        assert isinstance(leg["gates"]["G2 -10 dB stopband width"], bool)


def test_the_three_legs_differ_only_in_one_geometric_input():
    """stub_1cell changes STUB_LEN; stub_narrow changes W_STUB; nothing else."""
    src = BUILDER.read_text()
    assert 'setattr(cv, "STUB_LEN", cv.STUB_LEN - cv.DX)' in src
    assert 'setattr(cv, "W_STUB", 5 * cv.DX)' in src
