"""Two radial shunt ports on a TEM line continued through both absorbers.

The pure-R network remains the oracle. The series-cell residual prediction
and the slow ladder's fitted order are separate checks. The receiving-port
S21 model is not established: its three predictions are reported only. The
residual verdict is the fixed-separation three-mesh trend; the pure-network
magnitude also retains its always-on 0.01 bar.
"""
from functools import lru_cache

import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

from tests._interior_tem_line import (build, two_port, element_inductance,
                                      assert_predicted_residual, assert_first_order, residuals, assert_solved_ports)

FREQS_HZ = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9
CLOSED_FORM_ATOL = 0.05


MESHES = (.1e-3, .05e-3, .025e-3)


def _mesh(kind, dx):
    return build(kind, two=True, dx=dx, cells=round(.9e-3 / dx),
                 axial_positions=(.1e-3, .6e-3), declared_separation=.5e-3)


def _build(kind):
    return _mesh(kind, MESHES[0])[0]


@lru_cache(maxsize=None)
def _s_matrix(kind, dx=.1e-3):
    sim, line = _mesh(kind, dx)
    forward = sim._forward_from_materials
    checked = []

    def checked_scan(*args, **kwargs):
        raw = forward(*args, **kwargs)
        if isinstance(raw, dict) and "lumped" in raw:
            assert_solved_ports(raw, line, kind, two=True)
            checked.append(True)
        return raw

    with patch.object(sim, "_forward_from_materials", checked_scan):
        res = sim.run(compute_s_params=True, s_param_freqs=FREQS_HZ,
                      skip_preflight=True)
    assert len(checked) == 2, "both independent port drives must expose their solved specs"
    return np.asarray(res.s_params)


def _expected(kind):
    _, line = _mesh(kind, MESHES[0])
    return two_port(line, FREQS_HZ)


@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_two_shunt_ports_match_the_pure_network_magnitude(kind, record_property):
    """Shunt ports reflect even when both reference resistances equal Zc."""
    s11 = np.abs(_s_matrix(kind)[0, 0])
    record_property("measured_s11", json.dumps(s11.tolist()))
    record_property("closed_form_s11", json.dumps(np.abs(_expected(kind)[0, 0]).tolist()))
    assert np.abs(s11 - np.abs(_expected(kind)[0, 0])).max() <= CLOSED_FORM_ATOL, (
        f"{kind}: TEM through line; read {np.round(s11, 5)} "
        f"at {FREQS_HZ / 1e9} GHz")


def test_the_two_lanes_agree_on_the_diagonal_of_the_same_cells():
    """Same cells, same sigma, same drive — so one diagonal, not two."""
    lumped = _s_matrix("lumped")[0, 0]
    wire = _s_matrix("wire")[0, 0]
    assert np.allclose(lumped, wire, rtol=0.0, atol=1e-6), (
        f"lumped {np.round(np.abs(lumped), 6)} vs wire "
        f"{np.round(np.abs(wire), 6)} on the identical cells")


def test_the_two_lanes_agree_on_every_entry_of_the_same_cells():
    """One port, one S-matrix — diagonal and off-diagonal alike.

    This is the gate that makes the two lanes one implementation rather than
    two that happen to agree: the lumped decomposition IS the wire
    decomposition at one live cell, so every entry must match, not just the
    diagonal. Measured: exactly 0.0 on complex S, all four entries.
    """
    lumped = _s_matrix("lumped")
    wire = _s_matrix("wire")
    assert np.allclose(lumped, wire, rtol=0.0, atol=1e-6), (
        f"lumped {np.round(np.abs(lumped), 6)} vs wire "
        f"{np.round(np.abs(wire), 6)}")


@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_diagonal_matches_the_predicted_cell_residual(kind, record_property):
    # S11/S22 retain D1(ii); the three S21 alternatives are reported below.
    _, line = _mesh(kind, MESHES[0])
    pure = two_port(line, FREQS_HZ)
    predicted = two_port(line, FREQS_HZ, element_l=element_inductance(line))
    measured = _s_matrix(kind)
    record_property("residuals", json.dumps(residuals(measured, pure, predicted)))
    record_transmission_predictions(line, measured, pure, predicted, record_property)
    for k in (0, 1):
        assert_predicted_residual(measured[k, k], pure[k, k], predicted[k, k])


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_two_port_three_mesh_trend(kind, record_property):
    errors, phases, curves = [], [], []
    for dx in MESHES:
        _, line = _mesh(kind, dx)
        pure = two_port(line, FREQS_HZ)
        predicted = two_port(line, FREQS_HZ, element_l=element_inductance(line))
        measured = _s_matrix(kind, dx)
        curves.append((measured, pure, predicted))
        record_property(f"residuals_{dx}", json.dumps(residuals(measured, pure, predicted)))
        errors.append(abs(measured - pure).reshape(-1))
        phases.append(abs(np.angle(measured / pure)).reshape(-1))
    # S21 receiving-port residual model is not established: measured -1.141
    # deg lies between -1.440 and -0.958 at 100 um, 10 GHz. No coefficient
    # is fitted; the three predictions are report-only. Its verdict is this trend.
    record_property("complex_order", json.dumps(assert_first_order(MESHES, errors).tolist()))
    record_property("phase_order", json.dumps(assert_first_order(MESHES, phases).tolist()))
    for dx, (measured, pure, predicted) in zip(MESHES, curves):
        _, line = _mesh(kind, dx)
        record_transmission_predictions(line, measured, pure, predicted, record_property)
        for k in (0, 1):
            assert_predicted_residual(measured[k, k], pure[k, k], predicted[k, k])


def record_transmission_predictions(line, measured, pure, predicted, record_property):
    """Receiver-only read alternatives; source denominator and loading fixed.

    DFT exp(-j omega t). With receiving branch current Ib, I_into=-Ib,
    b_R=(R Ib-R I_into)/(2 sqrt(R)); reading R+jwL instead multiplies b by
    1+jwL/(2R). The product reads -E dx and an Ampere loop, not an explicit
    added jwL voltage (rfx/probes/probes.py:258,253,1542). Its passive cell
    update gives -I_H/V_R=cos(w dt/2)/R+j*(2/dt)*sin(w dt/2)*eps0*dx.
    Therefore the R-only Ampere read multiplies b_R by
    (1+cos(w dt/2)+j*R*eps0*dx*(2/dt)*sin(w dt/2))/2.
    None of these receiving-port residual predictions is asserted.
    The pure-network |S21| bar remains 0.01 per bin; only the residual
    MODEL is report-only, with convergence judged by the slow ladder.
    """
    w = 2 * np.pi * FREQS_HZ
    dt = .99 * line.dx / (299792458. * np.sqrt(3))
    r_only = predicted[1, 0]
    r_plus_l = r_only * (1 + .5j * w * element_inductance(line) / line.zc)
    displacement = r_only * (1 + np.cos(w * dt / 2)
                             + 1j * line.zc * 8.8541878128e-12 * line.dx
                             * (2 / dt) * np.sin(w * dt / 2)) / 2
    values = dict(dx_m=line.dx, realized_d_m=line.length,
                  measured_abs_s21=abs(measured[1, 0]).tolist(),
                  pure_abs_s21=abs(pure[1, 0]).tolist(),
                  measured_phase_deg=np.degrees(np.angle(measured[1, 0] / pure[1, 0])).tolist())
    for name, curve in (("R_only", r_only), ("R_plus_jwL", r_plus_l),
                        ("R_only_displacement", displacement)):
        values[name] = residuals(measured[1, 0], pure[1, 0], curve)
    print(f"S21 report-only predictions: {values}", file=sys.stderr)
    record_property(f"s21_predictions_{line.dx}", json.dumps(values))


def assert_transmission_magnitude(measured, pure):
    assert np.all(abs(abs(measured) - abs(pure)) <= .01), "pure-network |S21| error exceeds 0.01"


@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_transmission_matches_pure_network_magnitude(kind, record_property):
    measured, pure = _s_matrix(kind)[1, 0], _expected(kind)[1, 0]
    record_property("s21_phase_residual_deg", json.dumps(np.degrees(np.angle(measured / pure)).tolist()))
    record_property("s21_magnitude_error", json.dumps((abs(measured) - abs(pure)).tolist()))
    assert_transmission_magnitude(measured, pure)


def test_transmission_magnitude_detects_removed_check():
    with pytest.raises(AssertionError, match="pure-network"):
        assert_transmission_magnitude(np.array([.52]), np.array([.5]))
