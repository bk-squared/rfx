"""A graded waveguide's empty reference must remove the device's μ slab."""
import numpy as np
import pytest

from rfx import Simulation, Box, GaussianPulse
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners import nonuniform


def test_stripping_reference_mu_matches_empty_guide_fields():
    def model(magnetic):
        sim = Simulation(freq_max=30e9, domain=(.006,) * 3, dx=.001,
                         dz_profile=np.full(6, .001), boundary='pec', cpml_layers=0)
        if magnetic:
            sim.add_material('magnetic', eps_r=1., mu_r=4.)
            sim.add(Box((.002,) * 3, (.004,) * 3), material='magnetic')
        sim.add_source((.003,) * 3, 'ey', waveform=GaussianPulse(f0=15e9),
                       amplitude_kind='field')
        sim.add_probe((.003, .003, .004), 'ey')
        return sim
    empty = nonuniform.run_nonuniform_path(model(False), n_steps=30)
    stripped = nonuniform.run_nonuniform_path(
        model(True), n_steps=30, strip_magnetic_materials=True)
    device = nonuniform.run_nonuniform_path(model(True), n_steps=30)
    assert np.max(abs(np.asarray(empty.time_series))) > 0
    assert np.asarray(empty.time_series).tobytes() == np.asarray(stripped.time_series).tobytes()
    assert not np.array_equal(device.time_series, stripped.time_series)


def test_s_matrix_strips_mu_only_in_reference_call(monkeypatch):
    sim = Simulation(
        freq_max=11e9, domain=(.024, .004, .096), dx=.001,
        dz_profile=np.full(96, .001), cpml_layers=4,
        boundary=BoundarySpec(x=Boundary('pec', 'pec'), y=Boundary('pec', 'pec'),
                              z=Boundary('cpml', 'cpml')))
    sim.add_material('magnetic', eps_r=1., mu_r=4.)
    sim.add(Box((0., 0., .036), (.024, .004, .046)), material='magnetic')
    for position, direction in ((.016, '+z'), (.080, '-z')):
        sim.add_waveguide_port(position, direction=direction, mode=(1, 0),
                               freqs=np.linspace(8e9, 10e9, 21), f0=9e9)
    calls = []

    class ReferenceReached(Exception):
        pass

    def capture(_sim, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise ReferenceReached
        return None

    monkeypatch.setattr(nonuniform, 'run_nonuniform_path', capture)
    with pytest.raises(ReferenceReached):
        sim.compute_waveguide_s_matrix(num_periods=20, normalize='flux')
    assert not calls[0].get('strip_magnetic_materials', False)
    assert calls[1]['strip_magnetic_materials'] is True
