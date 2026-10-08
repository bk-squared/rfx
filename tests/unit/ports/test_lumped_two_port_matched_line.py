"""Two radial shunt ports on a TEM line continued through both absorbers.

The pure-R network remains the oracle. The series-cell residual prediction
and the slow ladder's fitted order are separate checks. S21's prediction
check currently exposes a discrepancy; first order alone does not excuse it.
"""
from functools import lru_cache

import json

import numpy as np
import pytest

from tests._interior_tem_line import (build, two_port, element_inductance,
                                      assert_predicted_residual, assert_first_order, residuals)

FREQS_HZ = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9
CLOSED_FORM_ATOL = 0.05


MESHES = (.1e-3, .05e-3, .025e-3)


def _mesh(kind, dx):
    return build(kind, two=True, dx=dx, cells=round(.9e-3 / dx),
                 axial_positions=(.123e-3, .631e-3))


def _build(kind):
    return _mesh(kind, MESHES[0])[0]


@lru_cache(maxsize=None)
def _s_matrix(kind, dx=.1e-3):
    res = _mesh(kind, dx)[0].run(compute_s_params=True, s_param_freqs=FREQS_HZ,
                           skip_preflight=True)
    return np.asarray(res.s_params)


def _expected(kind):
    _, line = build(kind, two=True, dx=0.1e-3)
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


def test_the_wire_off_diagonal_matches_the_closed_form(record_property):
    """Wire transmission agrees with the TEM network including both through branches."""
    s21 = np.abs(_s_matrix("wire")[1, 0])
    record_property("measured_s21", json.dumps(s21.tolist()))
    record_property("closed_form_s21", json.dumps(np.abs(_expected("wire")[1, 0]).tolist()))
    assert np.abs(s21 - np.abs(_expected("wire")[1, 0])).max() <= 0.01, (
        f"TEM through line; wire read {np.round(s21, 5)}")


def test_the_lumped_off_diagonal_matches_the_closed_form(record_property):
    """Lumped transmission retains the wire lane's original 0.01 bar."""
    s21 = np.abs(_s_matrix("lumped")[1, 0])
    record_property("measured_s21", json.dumps(s21.tolist()))
    record_property("closed_form_s21", json.dumps(np.abs(_expected("lumped")[1, 0]).tolist()))
    assert np.abs(s21 - np.abs(_expected("lumped")[1, 0])).max() <= 0.01, (
        f"TEM through line; lumped read {np.round(s21, 5)}")


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
def test_complex_s_matches_the_predicted_cell_residual(kind, record_property):
    # Old S21 phase bar: 1 degree. New bar: 10% of the residual predicted by
    # series L=0.214*mu0*dx at each port in the ABCD network (D1), in every
    # entry/bin, alongside the unchanged 0.05/0.01 magnitude guards above.
    _, line = _mesh(kind, MESHES[0])
    pure = two_port(line, FREQS_HZ)
    predicted = two_port(line, FREQS_HZ, element_l=element_inductance(line))
    measured = _s_matrix(kind)
    record_property("residuals", json.dumps(residuals(measured, pure, predicted)))
    assert_predicted_residual(measured, pure, predicted)


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
    record_property("complex_order", json.dumps(assert_first_order(MESHES, errors).tolist()))
    record_property("phase_order", json.dumps(assert_first_order(MESHES, phases).tolist()))
    for measured, pure, predicted in curves:
        assert_predicted_residual(measured, pure, predicted)
