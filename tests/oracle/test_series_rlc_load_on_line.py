"""Series RLC impedance on the realized internal PEC TEM coax (#1163/#1162).

The former 120-cell, one-cell-wide PMC channel and short calibration assumed
half-cell magnetic walls. Here the internal transverse cells give C'=3.75 eps0
and Zc=eta0/3.75. At dx=50 um the (70,9,9) grid puts the port/load at
(1,3,3)/(66,3,3), a 3.25 mm TEM section with 0.05/0.15 mm open stubs.
The declared separation is 3.254 mm; only the realized cells enter the oracle.
Unequal exterior margins separate all transverse conductors from boundaries.

First remove the near open-stub admittance, invert the TEM line transform,
then remove the far open-stub admittance. No measured short, fitted phase
or reference impedance calibrates this extraction. The answer is
R+j*omega*L+1/(j*omega*C); the unchanged bar is 5% of |Z| separately on Re/Im.
Before #1163, subtracting the series current after the field update removed
the edge's own impedance from R; RC-50 exercised the resulting instability.
"""
import json

import numpy as np
import pytest

from tests._interior_tem_line import build, extract_load

FREQS = np.linspace(1e9, 10e9, 10)
NUM_PERIODS = 40.0
BAR = 0.05


@pytest.mark.parametrize("r_ohm,l_h,c_f", [
    (300.0, 0.0, 1e-12),
    (300.0, 1e-9, 1e-12),
    (300.0, 1e-9, 0.0),
    (50.0, 0.0, 1e-12),
], ids=["RC-300", "RLC-300", "RL-300", "RC-50"])
def test_series_load_reads_its_closed_form(r_ohm, l_h, c_f, record_property):
    sim, line = build(dx=0.05e-3, cells=69, rlc=(r_ohm, l_h, c_f))
    assert line.port == (1, 3, 3) and line.load == (66, 3, 3)
    el, = sim._lumped_rlc
    assert (el.R, el.L, el.C, el.topology) == (r_ohm, l_h, c_f, "series")
    result = sim.forward(port_s11_freqs=FREQS, num_periods=NUM_PERIODS,
                         skip_preflight=True)
    s = np.asarray(result.s_params).reshape(-1)
    assert np.all(np.isfinite(s))
    z_meas = extract_load(line, FREQS, s)
    w = 2 * np.pi * FREQS
    z_true = r_ohm + 1j * w * l_h + (1 / (1j * w * c_f) if c_f else 0.)
    tol = BAR * np.abs(z_true)
    for name, values in (("z_measured", z_meas), ("z_closed_form", z_true), ("s11", s)):
        record_property(name, json.dumps([[float(z.real), float(z.imag)] for z in values]))
    error = np.maximum(np.abs(z_meas.real - z_true.real), np.abs(z_meas.imag - z_true.imag))
    print(f"R={r_ohm} L={l_h} C={c_f}: max component error/|Z|={max(error / abs(z_true)):.9g}")
    assert np.all(error <= tol), (
        f"5% complex-Z component bar: measured {z_meas}, closed form {z_true}, tolerance {tol}")
