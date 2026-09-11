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


@pytest.mark.parametrize("case", ["cv24-uniform", "cv24-single_band"])
def test_automatic_harminv_meets_cavity_lattice_and_stationarity_gates(case):
    directory = ROOT / case
    capture = json.loads((directory / "capture.json").read_text())
    arm = capture["arm"]
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
    for mode in modes:
        name = mode["name"]
        measured = windows[0][name]["f_hz"]
        exact = G.lattice_freq(tuple(mode["mnl"]), profile, arm["dx"],
                               dt=arm["dt"], a=a, b=b)["f_lattice_hz"]
        assert abs(measured / exact - 1) <= window, (case, name, measured, exact)
        scatter = abs(windows[1][name]["f_hz"] - windows[2][name]["f_hz"]) / measured
        assert scatter <= window, (case, name, scatter, window)
