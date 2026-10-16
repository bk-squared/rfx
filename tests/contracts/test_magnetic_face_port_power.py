"""Magnetic-face ports describe the full object; mixed flux cannot use them."""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pmc import refuse_port_on_magnetic_face
from rfx.boundaries.spec import Boundary, BoundarySpec


@pytest.mark.parametrize('face', ['x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi'])
@pytest.mark.parametrize('wire', [False, True])
@pytest.mark.parametrize('offset', [0., .49, .51])
def test_refusal_uses_realized_source_edge(face, wire, offset):
    dx, length = .001, .008
    axis = 'xyz'.index(face[0])
    boundaries = dict(x='pec', y='pec', z='pec')
    boundaries[face[0]] = Boundary('pmc', 'pec') if face.endswith('lo') else Boundary('pec', 'pmc')
    sim = Simulation(freq_max=20e9, domain=(length,) * 3, dx=dx,
                     boundary=BoundarySpec(**boundaries), cpml_layers=0)
    position = [.003] * 3
    position[axis] = offset * dx if face.endswith('lo') else length - offset * dx
    component = 'e' + 'xyz'[(axis + 1) % 3]
    sim.add_port(tuple(position), component, extent=2 * dx if wire else None)
    if offset > .5:
        refuse_port_on_magnetic_face(sim, 'contract', reason='Power comparison.')
        return
    with pytest.raises(ValueError) as caught:
        refuse_port_on_magnetic_face(sim, 'contract', reason='Power comparison.')
    message = str(caught.value)
    for text in ('contract', 'port 0', str(tuple(position)), face,
                 'full symmetric object', 'physical half', "magnitude_channel='wave'",
                 'move the port off the face', 'Power comparison.'):
        assert text in message


def _mixed_scene(wire):
    sim = Simulation(freq_max=20e9, domain=(.024, .016, .010), dx=.001,
                     cpml_layers=4, boundary=BoundarySpec(
                         x='cpml', y=Boundary('pmc', 'cpml'), z=Boundary('pec', 'cpml')))
    sim.add_material('substrate', eps_r=3.)
    sim.add(Box((0., 0., 0.), (.024, .016, .002)), material='substrate')
    sim.add(Box((0., .007, .002), (.014, .009, .002)), material='pec')
    sim.add_port((.008, .00049, 0.), 'ez', extent=.002 if wire else None)
    sim.add_msl_port(position=(.014, .008, 0.), width=.002, height=.002,
                     direction='-x', impedance=50., n_probe_offset=3, n_probe_spacing=2)
    return sim


class _StepReached(Exception):
    pass


@pytest.mark.parametrize('wire', [False, True])
def test_mixed_flux_refuses_before_monitor_or_step(monkeypatch, wire):
    sim = _mixed_scene(wire)
    steps = []

    def step(*args, **kwargs):
        steps.append(True)
        raise _StepReached

    def monitor(*args, **kwargs):
        pytest.fail('registered a flux monitor before refusing magnetic-face port')

    monkeypatch.setattr(sim, '_forward_from_materials', step)
    monkeypatch.setattr(sim, 'add_flux_monitor', monitor)
    from types import SimpleNamespace
    from rfx.diagnostic_records import Diagnostic
    prior = Diagnostic('contract.preflight', 'advisory', 'port 0', 'Prior observation', {})
    monkeypatch.setattr(sim, '_auto_preflight', lambda **kw: SimpleNamespace(diagnostics=(prior,)))
    with pytest.raises(ValueError, match='port 0.*magnetic face y_lo') as caught:
        sim.compute_mixed_s_matrix(freqs=np.array([8e9, 10e9]), n_steps=4,
                                   skip_preflight=False, magnitude_channel='flux')
    assert caught.value.diagnostics[0] == prior
    assert caught.value.diagnostics[-1].severity == 'refusal'
    assert caught.value.diagnostics[-1].message == str(caught.value)
    assert not steps
    assert not sim._flux_monitors


@pytest.mark.parametrize('wire', [False, True])
def test_mixed_wave_admits_magnetic_face_port(monkeypatch, wire):
    sim = _mixed_scene(wire)
    steps = []

    def step(*args, **kwargs):
        steps.append(True)
        raise _StepReached

    monkeypatch.setattr(sim, '_forward_from_materials', step)
    with pytest.raises(_StepReached):
        sim.compute_mixed_s_matrix(freqs=np.array([8e9, 10e9]), n_steps=4,
                                   skip_preflight=True, magnitude_channel='wave')
    assert len(steps) == 1


def _radiator_power(full, *, wire=False, high_face=False):
    """Use public rfx.run: one SimResult carries raw V/I and flux DFTs."""
    import time
    from rfx import GaussianPulse, make_port_source, make_probe, run
    from rfx.measurement.plan import build_measurement_plan
    from rfx.probes.probes import flux_spectrum
    from rfx.runners.uniform import build_flux_monitor_cfgs
    from rfx.simulation import LumpedPortSParamSpec, WirePortSParamSpec, make_wire_port_sources
    from rfx.sources.sources import (LumpedPort, WirePort, setup_lumped_port,
                                     setup_wire_port, _wire_port_live_cells)

    started = time.perf_counter()
    # Full physical y=-Ly..Ly is translated by Ly in the grid's zero frame.
    yc = .012 if full or high_face else 0.
    half_boundary = Boundary('cpml', 'pmc') if high_face else Boundary('pmc', 'cpml')
    sim = Simulation(freq_max=20e9, domain=(.024, .024 if full else .012, .024),
                     dx=.001, cpml_layers=4, boundary=BoundarySpec(
                         x='cpml', y='cpml' if full else half_boundary, z='cpml'))
    pulse = GaussianPulse(f0=10e9, bandwidth=.8)
    position = (.012, yc, .011 if wire else .012)
    # Wire spans a two-cell gap centred at z=12 mm.
    sim.add_port(position, 'ez', impedance=50., waveform=pulse, extent=.002 if wire else None)
    bins = np.array([8e9, 10e9, 12e9])
    low = np.array([.008, yc - .004 if full or high_face else 0., .008])
    high = np.array([.016, yc + .004 if full or not high_face else yc, .016])
    signs = []
    for axis in range(3):
        tangential = [a for a in range(3) if a != axis]
        for side, sign in ((0, -1), (1, 1)):
            if not full and axis == 1 and side == int(high_face):
                continue  # PMC carries no normal flux.
            sim.add_flux_monitor(axis='xyz'[axis], coordinate=(low if side == 0 else high)[axis],
                                 freqs=bins, size=tuple((high-low)[tangential]),
                                 center=tuple(((high+low)/2)[tangential]))
            signs.append(sign)
    grid = sim._build_realized_grid()
    materials = sim._assemble_materials(grid)[0]
    if wire:
        port = WirePort(position, (.012, yc, .013), 'ez', 50., pulse)
        materials = setup_wire_port(grid, port, materials)
        cells, flags, _ = _wire_port_live_cells(grid, port, None)
        live = tuple(tuple(int(v) for v in cell) for cell, flag in zip(cells, flags) if flag)
        assert len(live) == 2
        spec = WirePortSParamSpec(*live[len(live)//2], 'ez', bins, 50., live_cells=live)
        sources = make_wire_port_sources(grid, port, materials, 768)
        port_specs = dict(wire_port_sparams=[spec])
    else:
        port = LumpedPort(position, 'ez', 50., pulse)
        materials = setup_lumped_port(grid, port, materials)
        spec = LumpedPortSParamSpec(*grid.position_to_index(position), 'ez', bins, 50.)
        sources = [make_port_source(grid, port, materials, 768)]
        port_specs = dict(lumped_port_sparams=[spec])
    result = run(grid, materials, 768, boundary='cpml', pec_axes='', sources=sources,
                 probes=[make_probe(grid, (.014, yc + (-.002 if high_face else .002), .013), 'ez')],
                 flux_monitors=build_flux_monitor_cfgs(sim, grid, 768), **port_specs)
    # Independently transform the public records by channel name and the Yee
    # physical clock. No S, port-power helper, or mirror factor enters P.
    owner = next(o for o in build_measurement_plan(sim, grid, n_steps=768, path='run_uniform').owners
                 if o.id == 'port:0:mode:0')
    record = result.sparam_time_records[0]
    assert record.channels == tuple(c.name for c in owner.channels)
    channels = {name: np.asarray(record[name]) for name in record.channels}
    voltage = 'V_port' if wire else 'V'  # Whole-gap wire voltage, not midpoint voltage.
    spectra = {name: (np.exp(-2j*np.pi*bins[:, None]*(np.arange(record.samples.shape[0])+offset)*result.dt)
                     @ channels[name]) * result.dt for name, offset in [(voltage, 1.), ('I', .5)]}
    power = .5*np.real(spectra[voltage]*np.conj(spectra['I']))
    # flux_spectrum returns Re(E x H*); .5 puts it in the same power units.
    flux = .5*sum(sign*flux_spectrum(mon, exact_f64=True)
                  for sign, mon in zip(signs, result.flux_monitors))
    return dict(power=power, flux=flux, field=np.asarray(result.time_series),
                seconds=time.perf_counter()-started)


@pytest.mark.parametrize('wire', [False, True], ids=['lumped', 'wire'])
@pytest.mark.parametrize('high_face', [False, True], ids=['low', 'high'])
def test_half_vs_full_port_power(wire, high_face):
    """The face port describes the same full symmetric object as the interior port."""
    import json
    import os
    from pathlib import Path

    full = _radiator_power(True, wire=wire, high_face=high_face)
    half = _radiator_power(False, wire=wire, high_face=high_face)
    if directory := os.environ.get('RFX_MIRROR_OUTPUT'):
        record = {label: {name: value.tolist() if isinstance(value, np.ndarray) else value
                          for name, value in row.items() if name != 'field'}
                  for label, row in [('full', full), ('half', half)]}
        record['field_relative_l2'] = float(np.linalg.norm(half['field']-full['field'])
                                            / np.linalg.norm(full['field']))
        Path(directory, f"port-power-{'wire' if wire else 'lumped'}-{'high' if high_face else 'low'}.json").write_text(json.dumps(record, indent=2)+'\n')
    peak = np.max(np.abs(full['power']))
    assert peak > 0
    assert np.max(np.abs(half['power']-full['power'])) <= 1e-4*peak
