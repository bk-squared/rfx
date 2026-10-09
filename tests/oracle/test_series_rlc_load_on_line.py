"""Series RLC on a through TEM line, inverted with the exact pure-load network.

DFT kernel exp(-j omega t): Z=R+j omega L+1/(j omega C).
The pure-load answer and reference planes are never fitted to measurements.
"""
import json
from functools import lru_cache
from contextlib import nullcontext

from tests._x64_compat import enable_x64

import numpy as np
import pytest

from tests._interior_tem_line import (build, extract_load, input_reflection,
                                      element_inductance, assert_first_order, assert_solved_ports, residuals)

FREQS = np.linspace(1e9, 10e9, 10)
NUM_PERIODS = 40.0
MODEL_ALLOWANCE = 0.10
MESHES = (.05e-3, .025e-3, .0125e-3)
CASES = [(300., 0., 1e-12), (300., 1e-9, 1e-12),
         (300., 1e-9, 0.), (50., 0., 1e-12)]
IDS = ["RC-300", "RLC-300", "RL-300", "RC-50"]


@lru_cache(maxsize=None)
def measured_load(r_ohm, l_h, c_f, dx=MESHES[0], *, precision="float32"):
    with enable_x64() if precision == "float64" else nullcontext():
        sim, line = build(dx=dx, cells=round(3.45e-3/dx), rlc=(r_ohm, l_h, c_f),
                          axial_positions=(.05e-3, 3.30e-3), declared_separation=3.25e-3, precision=precision)
        el, = sim._lumped_rlc
        assert (el.R, el.L, el.C, el.topology) == (r_ohm, l_h, c_f, "series")
        result = sim.forward(port_s11_freqs=FREQS, num_periods=NUM_PERIODS,
                             skip_preflight=True)
        assert_solved_ports(result, line)
        s = np.asarray(result.s_params).reshape(-1)
        assert np.all(np.isfinite(s)), "non-finite series-load S11"
    return line, s


def load_curves(line, s, r_ohm, l_h, c_f, *, inductance_factor=1.):
    w = 2*np.pi*FREQS
    z_true = r_ohm + 1j*w*l_h + (1/(1j*w*c_f) if c_f else 0.)
    predicted_s = input_reflection(line, FREQS, z_true,
                                   element_l=inductance_factor*element_inductance(line))
    return extract_load(line, FREQS, s), z_true, extract_load(line, FREQS, predicted_s)


def assert_signed_residual(z_meas, z_true, predicted_z):
    # The bar is the measured one-cell element inductance (0.214*mu0*dx per
    # element). Old bar: 0.05*|Z| per component. New: predicted SIGNED residual
    # +/-10% of its magnitude, per component. At 50 um Lcell=13.44601656 pH
    # per element, wL=0.084484..0.844838 ohm over 1..10 GHz. Add it at BOTH
    # terminals and invert the same pure shunt network to derive predicted_z;
    # this accounts for that inverse's amplification, with no fitted tolerance.
    # Example from that formula, RC-300 at 50 um/10 GHz: old component bar
    # 15.0211 ohm; new residual center -35.61345+10.63533j ohm, with a
    # +/-3.71676 ohm budget on each signed component (0.10*|center|).
    measured, predicted = z_meas-z_true, predicted_z-z_true
    limit = MODEL_ALLOWANCE*abs(predicted)
    for name, delta in (("real", (measured-predicted).real),
                        ("imag", (measured-predicted).imag)):
        assert np.all(abs(delta) <= limit), (
            f"signed {name} residual differs by >10% of predicted magnitude: "
            f"measured={measured}; predicted={predicted}; limit={limit}")


def record_load(line, s, curves, record_property):
    record_property(f"realized_d_{line.dx}", line.length)
    pure_s = input_reflection(line, FREQS, curves[1])
    predicted_s = input_reflection(line, FREQS, curves[1], element_l=element_inductance(line))
    record_property(f"residuals_{line.dx}", json.dumps(residuals(s, pure_s, predicted_s)))
    for name, values in zip(("z_measured", "z_closed_form", "z_predicted", "s11"), (*curves, s)):
        record_property(f"{name}_{line.dx}", json.dumps(np.stack([values.real, values.imag], axis=-1).tolist()))


@pytest.mark.parametrize("r_ohm,l_h,c_f", CASES, ids=IDS)
def test_series_load_reads_its_closed_form(r_ohm, l_h, c_f, record_property):
    record_property("precision", "float32")
    record_property("num_periods", NUM_PERIODS)
    line, s = measured_load(r_ohm, l_h, c_f)
    curves = load_curves(line, s, r_ohm, l_h, c_f)
    record_load(line, s, curves, record_property)
    assert_signed_residual(*curves)


@pytest.mark.slow
@pytest.mark.parametrize("r_ohm,l_h,c_f", CASES, ids=IDS)
def test_series_load_three_mesh_trend(r_ohm, l_h, c_f, record_property):
    # Refined capacitive cells need a roundoff control: RC-300 at 25 um
    # has a 24.62% signed-residual mismatch in float32, unchanged at 80 vs 40
    # periods; float64 at the same 40 periods reads 4.43%. Use float64 on
    # EVERY rung to judge truncation order without changing the circuit,
    # frequencies, recording time, cell coefficient or any verdict bar.
    record_property("precision", "float64")
    record_property("num_periods", NUM_PERIODS)
    errors = []
    for dx in MESHES:
        line, s = measured_load(r_ohm, l_h, c_f, dx, precision="float64")
        curves = load_curves(line, s, r_ohm, l_h, c_f)
        record_load(line, s, curves, record_property)
        assert_signed_residual(*curves)
        errors.append(abs(curves[0]-curves[1]))
    record_property("complex_order", json.dumps(assert_first_order(MESHES, errors).tolist()))


@pytest.mark.parametrize("mutation", ["zero-L", "double-L", "negative-L", "sign-flip", "conjugate"])
def test_signed_residual_rejects_mutations(mutation):
    line, s = measured_load(*CASES[0])
    measured, pure, predicted = load_curves(line, s, *CASES[0])
    if mutation in ("sign-flip", "conjugate"):
        residual = measured-pure
        measured = pure + (-residual if mutation == "sign-flip" else residual.conjugate())
    else:
        factor = {"zero-L": 0., "double-L": 2., "negative-L": -1.}[mutation]
        predicted = load_curves(line, s, *CASES[0], inductance_factor=factor)[2]
    with pytest.raises(AssertionError, match="signed .* residual"):
        assert_signed_residual(measured, pure, predicted)
