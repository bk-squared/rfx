"""Two ports on the internal PEC TEM coax (the old PMC-width line is retired).

The transverse Dirichlet solution gives C'=3.75 epsilon0 and Zc=eta0/3.75. At dx=0.1 mm,
the (10,9,9) grid realizes port nodes (1,3,3) and (6,3,3): 0.5 mm
separation, with unequal 0.1/0.3 mm open end stubs. Declared separation
is 0.508 mm. Internal sheets and a filament define the cross-section;
no transverse boundary wall is part of the line. ABCD multiplication
includes both stubs. Bars remain 0.05 on reflection magnitude, 0.01 on
transmission magnitude and 1 degree on phase. No fitted reference offset.
"""
from functools import lru_cache

import json

import numpy as np
import pytest

from tests._interior_tem_line import build, two_port

FREQS_HZ = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9
CLOSED_FORM_ATOL = 0.05


def _build(kind):
    return build(kind, two=True, dx=0.1e-3)[0]


@lru_cache(maxsize=None)
def _s_matrix(kind):
    res = _build(kind).run(compute_s_params=True, s_param_freqs=FREQS_HZ,
                           skip_preflight=True)
    return np.asarray(res.s_params)


def _expected(kind):
    _, line = build(kind, two=True, dx=0.1e-3)
    return two_port(line, FREQS_HZ)


@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_a_line_matched_at_both_ends_reflects_nothing(kind, record_property):
    """Zc loads with the realized open stubs, not fictitious zero-length ends."""
    s11 = np.abs(_s_matrix(kind)[0, 0])
    record_property("measured_s11", json.dumps(s11.tolist()))
    record_property("closed_form_s11", json.dumps(np.abs(_expected(kind)[0, 0]).tolist()))
    assert np.abs(s11 - np.abs(_expected(kind)[0, 0])).max() <= CLOSED_FORM_ATOL, (
        f"{kind}: TEM with open stubs; read {np.round(s11, 5)} "
        f"at {FREQS_HZ / 1e9} GHz")


def test_the_two_lanes_agree_on_the_diagonal_of_the_same_cells():
    """Same cells, same sigma, same drive — so one diagonal, not two."""
    lumped = _s_matrix("lumped")[0, 0]
    wire = _s_matrix("wire")[0, 0]
    assert np.allclose(lumped, wire, rtol=0.0, atol=1e-6), (
        f"lumped {np.round(np.abs(lumped), 6)} vs wire "
        f"{np.round(np.abs(wire), 6)} on the identical cells")


def test_the_wire_off_diagonal_matches_the_closed_form(record_property):
    """Wire transmission agrees with the TEM network including its stubs."""
    s21 = np.abs(_s_matrix("wire")[1, 0])
    record_property("measured_s21", json.dumps(s21.tolist()))
    record_property("closed_form_s21", json.dumps(np.abs(_expected("wire")[1, 0]).tolist()))
    assert np.abs(s21 - np.abs(_expected("wire")[1, 0])).max() <= 0.01, (
        f"TEM with open stubs; wire read {np.round(s21, 5)}")


def test_the_lumped_off_diagonal_matches_the_closed_form(record_property):
    """Lumped transmission retains the wire lane's original 0.01 bar."""
    s21 = np.abs(_s_matrix("lumped")[1, 0])
    record_property("measured_s21", json.dumps(s21.tolist()))
    record_property("closed_form_s21", json.dumps(np.abs(_expected("lumped")[1, 0]).tolist()))
    assert np.abs(s21 - np.abs(_expected("lumped")[1, 0])).max() <= 0.01, (
        f"TEM with open stubs; lumped read {np.round(s21, 5)}")


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


PHASE_GATE_DEG = 1.0


@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_s21_lags_by_the_electrical_length_of_the_line(kind, record_property):
    """Complex TEM transmission, including both unequal open end stubs."""
    s21 = _s_matrix(kind)[1, 0]
    expected = _expected(kind)[1, 0]
    err_deg = np.degrees(np.angle(s21 * np.conj(expected)))
    record_property("phase_error_deg", json.dumps(err_deg.tolist()))
    print(f"{kind}: maximum transmission phase error {max(abs(err_deg)):.9g} deg")
    assert np.abs(err_deg).max() <= PHASE_GATE_DEG, (
        f"{kind}: phase errors {err_deg} deg, bar {PHASE_GATE_DEG}")
