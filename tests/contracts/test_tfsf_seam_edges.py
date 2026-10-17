"""Full-span PEC sheets close the transverse seams of a TF/SF plane wave.

Every scene has a (57, 27, 27) field grid (41,553 values per component).
Edge comparisons stop at the uniform step's own input, before compilation;
the independent transmission check evolves two 700-step probe records.
"""

import rfx.stepping.uniform_loop as uniform_loop
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.stepping import uniform


def _scene(kind="sheet", polarization="ez", method="bloch", angle_deg=0):
    sim = Simulation(freq_max=20e9, domain=(0.040, 0.010, 0.010),
                     dx=0.001, cpml_layers=8)
    sim.add_tfsf_source(f0=10e9, bandwidth=0.5, polarization=polarization,
                        direction="+x", method=method, angle_deg=angle_deg)
    if kind == "sheet":
        sim.add_thin_conductor(Box((0.022, 0, 0), (0.022, 0.010, 0.010)))
    elif kind == "box":
        sim.add(Box((0.022, 0, 0), (0.023, 0.010, 0.010)), material="pec")
    sim.add_probe((0.030, 0.005, 0.005), component=polarization)
    return sim


def _step_edges(monkeypatch, sim, entry):
    captured = []

    class CapturedStep(Exception):
        pass

    def capture(ctx, invariants, **kwargs):
        # simulation.make_core_step passes invariants["ctx"] directly to
        # this constructor; the conductor stage reads ctx.pec_edge_masks.
        assert ctx is invariants["ctx"]
        captured.append(tuple(np.asarray(mask).copy() for mask in ctx.pec_edge_masks))
        raise CapturedStep

    with monkeypatch.context() as patch:
        patch.setattr(uniform, "make_uniform_step", capture)
        with pytest.raises(CapturedStep):
            getattr(sim, entry)(n_steps=2)
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("kind,polarization,method,angle", [
    ("sheet", "ez", "bloch", 0),
    ("sheet", "ey", "bloch", 0),
    ("sheet", "ez", "methodB", 20),
    ("box", "ez", "bloch", 0),
])
def test_plane_wave_step_edges_match(monkeypatch, kind, polarization, method, angle):
    run = _step_edges(monkeypatch, _scene(kind, polarization, method, angle), "run")
    forward = _step_edges(monkeypatch, _scene(kind, polarization, method, angle), "forward")
    for component, actual, expected in zip(("ex", "ey", "ez"), run, forward):
        print(f"{kind}/{polarization}/{method} {component}: "
              f"shape={actual.shape}, run={actual.sum()}, forward={expected.sum()}")
        assert actual.dtype == expected.dtype == np.bool_
        np.testing.assert_array_equal(actual, expected, err_msg=component)
    if kind == "sheet" and angle == 0:
        # Independent of the shared realization: a sheet wrapped on y and z owns
        # every tangential edge of its plane, pads included (ny * nz of the grid).
        ny, nz = run[0].shape[1:]
        assert [int(m.sum()) for m in run] == [0, ny * nz, ny * nz]


def test_plane_wave_decay_step_edges(monkeypatch):
    """run(until_decay=) steps through its own host loop; its step input carries the same edges."""
    import rfx.simulation as kernel
    captured = []
    original = uniform_loop.core_step_invariants

    def capture(ctx):
        captured.append(tuple(np.asarray(mask).copy() for mask in ctx.pec_edge_masks))
        return original(ctx)

    monkeypatch.setattr(uniform_loop, "core_step_invariants", capture)
    _scene().run(n_steps=40, until_decay=1e-3, decay_monitor_component="ez",
                 decay_monitor_position=(0.030, 0.005, 0.005))
    assert captured
    ny, nz = captured[0][0].shape[1:]
    for edges in captured:
        assert [int(m.sum()) for m in edges] == [0, ny * nz, ny * nz]


def test_full_span_sheet_transmission_below_minus_40_db():
    reference = _scene("vacuum")
    dt = float(reference._build_grid().dt)
    incident = np.asarray(reference.run(n_steps=700).time_series).reshape(-1)
    transmitted = np.asarray(_scene().run(n_steps=700).time_series).reshape(-1)
    # 5--15 GHz source band, excluding its upper sixth: 5--13 GHz.
    # Literal independent opacity bar; no forward() result sets it. With the
    # seam open this reads -13 to -21 dB; closed, the level (-51 to -59 dB) is
    # set by the 8-layer x absorber returning the shadow wave to the probe
    # (16 layers: -87 to -98 dB), so an absorber regression can also trip it.
    frequencies = np.linspace(5e9, 13e9, 17)
    phase = np.exp(-2j * np.pi * frequencies[:, None] * np.arange(700) * dt)
    incident_dft = phase @ incident
    assert np.all(np.abs(incident_dft) > 0)
    transmission_db = 20 * np.log10(np.abs(phase @ transmitted) / np.abs(incident_dft))
    print(f"frequencies_hz={frequencies.tolist()}")
    print(f"transmission_db={transmission_db.tolist()}")
    assert np.all(transmission_db < -40.0), transmission_db
