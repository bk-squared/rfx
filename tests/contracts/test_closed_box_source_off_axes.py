"""A normally incident closed-box plane wave reads the same face-read level on both signs and every polarization.

Ring-down judges a closed-box plane wave off by the level ``r(n)`` its box
faces read from the 1-D incident line. Normal incidence (realized direction
(+-1, 0, 0)) is admitted for ``ez``, ``ey`` and ``s``/``p`` at theta 0 because
each such line reads the reference line's level. Each row here is the
reference declaration with one change; its line is recorded for 32,000 steps
with ``record_face_values``, and the windows ring-down reads,
[round(0.5 N), N) and [round(0.25 N), N) for N = 2,000, 8,000 and 32,000, are
taken from that record's prefixes. Every window that opens after the row's own
pulse has left the read nodes is at most 1e-6, and every window falls in the
reference row's class (at most 1e-6, a pulse refusal, or a floor).
"""
import jax
import numpy as np
import pytest

from tests._closed_box_plane_wave import (declare, level, line, pulse_end, recorded_maxima,
                                          windows)
from rfx.stepping.incident_line import record_face_values

LONGEST = 32000
LENGTHS = (2000, 8000, 32000)
ROWS = {
    "plus_x_ey": ({"polarization": "ey"}, (1, 0, 0)),
    "plus_x_s_phi_30": ({"polarization": "s", "phi_deg": 30}, (1, 0, 0)),
    "minus_x_ez": ({"direction": "-x"}, (-1, 0, 0)),
    "minus_x_ey": ({"direction": "-x", "polarization": "ey"}, (-1, 0, 0)),
    "minus_x_s_phi_30": ({"direction": "-x", "polarization": "s", "phi_deg": 30}, (-1, 0, 0)),
    "plus_x_p_phi_60": ({"polarization": "p", "phi_deg": 60}, (1, 0, 0)),
}


def table(i_h, i_e):
    return windows(i_h, i_e, LENGTHS, pulse_end(level(i_h, i_e)))


@pytest.fixture(scope="module")
def reference():
    i_h, i_e, m = recorded_maxima(declare(), LONGEST)
    assert m == (1, 0, 0)
    return i_h, i_e


def test_reference_row_is_off_after_its_pulse(reference):
    found = table(*reference)
    floors = {key: value for key, value in found.items() if value[2] == "floor"}
    assert not floors, f"a window opening after the pulse reads above 1e-6: {floors}"
    assert found[(2000, "early")][2] == "pulse", found[(2000, "early")]
    assert all(value[2] == "off" for key, value in found.items() if key != (2000, "early")), found


def test_reference_record_does_not_depend_on_its_length(reference):
    sim = declare()
    cfg, state, grid = line(sim, 2000)
    (e_line, h_line), _ = record_face_values(cfg, state, grid.dx, grid.dt)
    short_e = np.max(np.abs(np.asarray(e_line)), axis=1)
    short_h = np.max(np.abs(np.asarray(h_line)), axis=1)
    i_h, i_e = reference
    assert short_e.tobytes() == i_e[:2000].tobytes()
    assert short_h.tobytes() == i_h[:2000].tobytes()


@pytest.mark.parametrize("row", list(ROWS))
def test_normal_row_reads_the_reference_level(row, reference):
    change, expected = ROWS[row]
    i_h, i_e, m = recorded_maxima(declare(change), LONGEST)
    assert m == expected, f"{row}: realized direction {m}, expected {expected}"
    found, ref = table(i_h, i_e), table(*reference)
    floors = {key: value for key, value in found.items() if value[2] == "floor"}
    assert not floors, f"{row}: a window opening after the pulse reads above 1e-6: {floors}"
    differ = {key: (found[key], ref[key]) for key in found if found[key][2] != ref[key][2]}
    assert not differ, f"{row}: windows in another class than the reference row's: {differ}"


@pytest.mark.gpu
def test_reference_row_on_a_gpu_reads_the_cpu_level():
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        pytest.skip("needs a GPU")
    with jax.default_device(jax.devices("cpu")[0]):
        cpu = table(*recorded_maxima(declare(), LONGEST)[:2])
    with jax.default_device(gpus[0]):
        gpu = table(*recorded_maxima(declare(), LONGEST)[:2])
    for key in cpu:
        assert gpu[key][2] == cpu[key][2], (key, gpu[key], cpu[key])
        assert 0.5 <= gpu[key][1] / cpu[key][1] <= 2.0, (key, gpu[key], cpu[key])
