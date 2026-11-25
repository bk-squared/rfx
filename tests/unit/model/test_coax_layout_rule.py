"""Coaxial port planes, line limits and source clearance through both entries."""
import re

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.errors import RfxUnsupported
from rfx.model.coax_line import coax_port_layout
from rfx.nonuniform import make_nonuniform_grid
from tests.contracts.test_coax_port_model_and_drive import capture


def scene(z, face='top', *, freq_min=None):
    sim = Simulation(freq_max=4.5e9, domain=(.040, .040, z), dx=.001,
                     boundary='cpml', cpml_layers=4, snap='declared')
    at = .002 if face == 'top' else round(z - .002, 9)
    sim.add(Box((0., 0., at), (.040, .040, at)), material='pec')
    sim.add_coaxial_port((.020, .020, at), face=face, pin_length=0.,
                         pin_radius=1e-3, outer_radius=6e-3, freq_min=freq_min)
    return sim


def sma_scene(z, face):
    sim = Simulation(freq_max=18e9, domain=(.012, .012, z), dx=.00025,
                     boundary='cpml', cpml_layers=4, snap='declared')
    at = .002 if face == 'top' else round(z - .002, 9)
    sim.add(Box((0., 0., at), (.012, .012, at)), material='pec')
    sim.add_coaxial_port((.006, .006, at), face=face, pin_length=0.)
    return sim


def clearance_scene(z, face):
    sim = Simulation(freq_max=10e9, domain=(.012, .012, z), dx=.0005,
                     boundary='cpml', cpml_layers=4, snap='declared')
    at = .002 if face == 'top' else round(z - .002, 9)
    sim.add_coaxial_port((.006, .006, at), face=face, pin_length=0)
    return sim


def admitted_layout(monkeypatch, sim, entry):
    capture(monkeypatch, sim, entry)
    return sim.realized_geometry().conductors.coax_bore_release.layout


def refused_text(sim, entry, capsys):
    with pytest.raises(NotImplementedError) as excinfo:
        getattr(sim, entry)(n_steps=3, skip_preflight=True)
    with capsys.disabled():
        print(f'{entry}: {type(excinfo.value).__name__}: {excinfo.value}')
    return str(excinfo.value)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_length_remedies(monkeypatch, entry, capsys):
    text = refused_text(scene(.103), entry, capsys)
    assert 'needs at least 105 line cells (105 mm) for the band 0.45 to 4.5 GHz' in text
    assert 'the line has 100 cells (100 mm) from the reference to the z absorber' in text
    assert 'lengthen the line by 5 cells (5 mm), or raise freq_min=' in text
    match = re.search(r'lengthen the line by (\d+) cells \(([0-9.]+) mm\), or raise freq_min=', text)
    assert match is not None
    N = int(match.group(1))
    X = match.group(2)
    assert float(X) == N * 1.0
    lengthened = admitted_layout(monkeypatch, scene(round(.103 + N * .001, 9)), entry)
    assert lengthened.plane_indices == tuple(6 + 33 + 4 * k for k in range(12))
    raised = admitted_layout(monkeypatch, scene(.103, freq_min=0.9e9), entry)
    assert raised.plane_indices == tuple(34 + 4 * k for k in range(12))


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_length_edge_bottom(monkeypatch, entry, capsys):
    text = refused_text(scene(.107, 'bottom'), entry, capsys)
    assert 'needs at least 105 line cells (105 mm) for the band 0.45 to 4.5 GHz' in text
    assert 'lengthen the line by 1 cells (1 mm), or raise freq_min=' in text
    layout = admitted_layout(monkeypatch, scene(.108, 'bottom'), entry)
    assert layout.reference_index == 110
    assert layout.plane_indices == tuple(110 - (33 + 4 * k) for k in range(12))
    assert layout.source_index == 15
    assert layout.feed_index == 13


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('face', ['top', 'bottom'])
def test_length_edge_sma(monkeypatch, entry, face, capsys):
    text = refused_text(sma_scene(.03525, face), entry, capsys)
    assert 'needs at least 133 line cells (33.25 mm) for the band 1.8 to 18 GHz' in text
    assert 'lengthen the line by 1 cells (0.25 mm), or raise freq_min=' in text
    layout = admitted_layout(monkeypatch, sma_scene(.0355, face), entry)
    if face == 'top':
        assert layout.plane_indices == tuple(61 + 4 * k for k in range(12))
        assert layout.source_index == 132
        assert layout.feed_index == 134
    else:
        assert layout.reference_index == 138
        assert layout.plane_indices == tuple(138 - (49 + 4 * k) for k in range(12))
        assert layout.source_index == 18
        assert layout.feed_index == 16


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('face', ['top', 'bottom'])
def test_ntff_end(entry, face, capsys):
    sim = clearance_scene(.052, face)
    if face == 'top':
        sim.add_ntff_box((.001, .001, .001), (.011, .011, .045), freqs=[5e9])
    else:
        sim.add_ntff_box((.001, .001, .007), (.011, .011, .051), freqs=[5e9])
    text = refused_text(sim, entry, capsys)
    assert 'the line has 84 cells (42 mm) from the reference to the NTFF box; lengthen the line by 8 cells (4 mm), or raise freq_min=' in text


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_plane_wiring(monkeypatch, entry):
    layout = admitted_layout(monkeypatch, scene(.108), entry)
    assert layout.reference_index == 6
    assert layout.plane_indices == tuple(6 + 33 + 4 * k for k in range(12))


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('face', ['top', 'bottom'])
def test_source_clearance(monkeypatch, entry, face):
    layout = admitted_layout(monkeypatch, clearance_scene(.0485, face), entry)
    direction = 1 if face == 'top' else -1
    if face == 'top':
        assert layout.source_index == 90
        assert layout.plane_indices[-1] == 77
        assert direction * (layout.source_index - layout.plane_indices[-1]) == 13
    else:
        assert direction * (layout.source_index - layout.plane_indices[-1]) == 14


@pytest.mark.parametrize('z_profile', [
    np.full(52, .5e-3),
    np.concatenate([np.full(26, .5e-3), np.full(26, .25e-3)]),
], ids=['uniform-profile', 'graded-profile'])
def test_refuses_nonuniform_grid(z_profile):
    grid = make_nonuniform_grid((.04, .04), z_profile, .5e-3, 8)
    port = scene(.108)._coaxial_ports[0]
    with pytest.raises(RfxUnsupported) as excinfo:
        coax_port_layout(grid, port, absorber_pads=(8, 8, 8, 8, 8, 8), freq_max=4.5e9)
    assert 'needs the uniform grid; got NonUniformGrid' in str(excinfo.value)
