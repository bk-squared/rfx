"""Series RLC on a through TEM line, inverted with the exact pure-load network.

DFT kernel exp(-j omega t): Z=R+j omega L+1/(j omega C).
The pure-load answer and reference planes are never fitted to measurements.
"""
import json

import numpy as np
import pytest

from tests._interior_tem_line import build, extract_load, input_reflection, element_inductance

FREQS = np.linspace(1e9, 10e9, 10)
NUM_PERIODS = 40.0
MODEL_ALLOWANCE = 0.10


@pytest.mark.parametrize("r_ohm,l_h,c_f", [
    (300.0, 0.0, 1e-12),
    (300.0, 1e-9, 1e-12),
    (300.0, 1e-9, 0.0),
    (50.0, 0.0, 1e-12),
], ids=["RC-300", "RLC-300", "RL-300", "RC-50"])
def test_series_load_reads_its_closed_form(r_ohm, l_h, c_f, record_property):
    sim, line = build(dx=0.05e-3, cells=69, rlc=(r_ohm, l_h, c_f))
    assert line.port == (21, 3, 3) and line.load == (86, 3, 3)
    el, = sim._lumped_rlc
    assert (el.R, el.L, el.C, el.topology) == (r_ohm, l_h, c_f, "series")
    result = sim.forward(port_s11_freqs=FREQS, num_periods=NUM_PERIODS,
                         skip_preflight=True)
    s = np.asarray(result.s_params).reshape(-1)
    assert np.all(np.isfinite(s))
    z_meas = extract_load(line, FREQS, s)
    w = 2 * np.pi * FREQS
    z_true = r_ohm + 1j * w * l_h + (1 / (1j * w * c_f) if c_f else 0.)
    predicted_s = input_reflection(line, FREQS, z_true, element_l=element_inductance(line))
    predicted_z = extract_load(line, FREQS, predicted_s)
    predicted_error = np.maximum(abs((predicted_z - z_true).real), abs((predicted_z - z_true).imag))
    # D2: old 0.05*|Z| component bar -> 1.10*predicted_error(f).
    # Derivation: add Lcell=0.214*mu0*50um=13.446 pH at BOTH terminals,
    # invert the same pure network, then allow 10% of that predicted residual.
    # No rounded 13% blanket tolerance; every frequency keeps its own bar.
    tol = (1 + MODEL_ALLOWANCE) * predicted_error
    for name, values in (("z_measured", z_meas), ("z_closed_form", z_true), ("s11", s)):
        record_property(name, json.dumps([[float(z.real), float(z.imag)] for z in values]))
    error = np.maximum(np.abs(z_meas.real - z_true.real), np.abs(z_meas.imag - z_true.imag))
    print(f"R={r_ohm} L={l_h} C={c_f}: max component error/|Z|={max(error / abs(z_true)):.9g}")
    assert np.all(error <= tol), (
        f"1.10 times cell-inductance-predicted component bar: measured {z_meas}, closed form {z_true}, tolerance {tol}")
    assert np.all(abs(error - predicted_error) <= MODEL_ALLOWANCE * predicted_error), (
        f"50um coefficient validation: measured errors {error}, predicted {predicted_error}")
    record_property("predicted_component_error", json.dumps(predicted_error.tolist()))
