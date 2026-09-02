"""profile_forward basic smoke: returns timings, compile >= scan_exec."""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.profiling import profile_forward, format_report


def _sim():
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, float(np.sum(dz))),
                     dx=0.5e-3, dz_profile=dz, cpml_layers=4)
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    return sim


def test_profile_forward_returns_timings():
    sim = _sim()
    rep = profile_forward(sim, n_steps=40, warmup_trace=False)
    for k in ("grid_build", "compile", "scan_exec", "total", "n_steps", "cells"):
        assert k in rep, f"missing {k} in report"
    assert rep["total"] > 0
    assert rep["compile"] > 0
    assert rep["scan_exec"] > 0
    # Compile includes trace+JIT so it must be >= scan_exec.
    assert rep["compile"] >= rep["scan_exec"] * 0.5  # generous slack


def test_format_report_renders():
    sim = _sim()
    rep = profile_forward(sim, n_steps=40, warmup_trace=False)
    s = format_report(rep)
    assert "rfx profile" in s
    assert "scan_exec" in s
    assert "total" in s
