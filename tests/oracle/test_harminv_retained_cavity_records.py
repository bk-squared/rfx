"""Actual cv24 inputs against the exact lattice spectrum; no new FDTD solve.

The existing cv24 mode-count, lattice and stationarity windows are unchanged.
This is an estimator regression, not a replacement for cv24's energy audit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rfx.harminv import harminv
from validation.crossval.comparators import nu_cavity_gates as G

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "harminv_decimation"


def test_short_stage1_record_meets_its_existing_continuum_gate():
    from rfx.grid import C0
    from scripts.stage1_nu_cavity_physics_gate import _GATE_PCT

    directory = ROOT / "stage1"
    receipt = json.loads((directory / "receipt.json").read_text())
    path = directory / "input.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["input_sha256"]
    with np.load(path) as data:
        modes = harminv(data["signal"], float(data["dt"]),
                        float(data["fmin"]), float(data["fmax"]))
    continuum = C0 / 2 * np.sqrt(2) / 0.04  # TM110, a=b=40 mm, air
    mode = min(modes, key=lambda m: abs(m.freq - continuum))
    assert 100 * abs(mode.freq / continuum - 1) <= _GATE_PCT


@pytest.mark.parametrize("case,correct_metric", [
    ("cv24-uniform", True), ("cv24-single_band", True), ("cv24-metric_defect", False),
])
def test_automatic_harminv_meets_cavity_lattice_and_stationarity_gates(case, correct_metric):
    directory = ROOT / case
    capture = json.loads((directory / "capture.json").read_text())
    arm = capture["arm"]
    assert arm["metric_swap"] is (not correct_metric)
    profile = np.asarray(arm["profile_mm"]) * 1e-3
    a = (arm["nodes"][0] - 1) * arm["dx"]
    b = (arm["nodes"][1] - 1) * arm["dx"]
    band = (capture["records"][0]["f_min"], capture["records"][0]["f_max"])
    modes = G.declared_modes(band, a=a, b=b, d=float(profile.sum()))
    assert len(capture["records"]) == 3 * len(G.CHANNELS)
    windows = []
    for window in range(3):
        lines = []
        for c, channel in enumerate(G.CHANNELS):
            record = capture["records"][window * len(G.CHANNELS) + c]
            path = directory / record["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            with np.load(path) as data:
                signal = data["signal"]
            for m in harminv(signal, record["dt"], *band):
                lines.append(dict(f_hz=m.freq, amp=m.amplitude, error=m.error, channel=channel))
        found = G.identify_modes(lines, modes, band)
        assert found["n_clusters_in_band"] == len(modes) == 7
        assert not found["orphans"] and not found["ambiguous"]
        assert all(v is not None for v in found["per_mode"].values())
        windows.append(found["per_mode"])

    window = G.estimator_floor()  # The case's existing gate, not a re-pin.
    lattice_errors = []
    scatter_errors = []
    for mode in modes:
        name = mode["name"]
        measured = windows[0][name]["f_hz"]
        exact = G.lattice_freq(tuple(mode["mnl"]), profile, arm["dx"],
                               dt=arm["dt"], a=a, b=b)["f_lattice_hz"]
        lattice_errors.append(abs(measured / exact - 1))
        scatter = abs(windows[1][name]["f_hz"] - windows[2][name]["f_hz"]) / measured
        scatter_errors.append(scatter)
        assert scatter <= window, (case, name, scatter, window)
    print(f"{case}: max lattice error {max(lattice_errors) * 1e6:.6f} ppm; "
          f"max A/B scatter {max(scatter_errors) * 1e6:.6f} ppm; 7 modes in every window")
    if correct_metric:
        assert max(lattice_errors) <= window, (case, lattice_errors, window)
    else:
        # The wrong operator still produces seven stationary modes. Their
        # frequencies must be rejected against the intended lattice: a more
        # accurate extractor cannot turn wrong FDTD physics into a PASS.
        assert max(lattice_errors) > window, (case, lattice_errors, window)
