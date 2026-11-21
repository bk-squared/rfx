"""Review regressions: every unsupported input is named before stepping."""
from dataclasses import replace
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box
from rfx.errors import RfxUnsupported
from tests.contracts.test_coax_port_model_and_drive import scene, capture, Captured


def check_refusal(error, pattern):
    assert isinstance(error, RfxUnsupported), repr(error)
    assert re.search(pattern, str(error)), str(error)


def refusal(monkeypatch, sim, entry, pattern, **options):
    def kernel(*args, **kwargs):
        raise Captured('reached time stepping')
    monkeypatch.setattr('rfx.simulation.run', kernel)
    error = None
    try:
        getattr(sim, entry)(n_steps=3, skip_preflight=True, **options)
    except Exception as caught:
        error = caught
    check_refusal(error, pattern)


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('kind', ['source', 'polarized', 'zero_port', 'passive_port'])
def test_other_excitation(monkeypatch, entry, kind):
    sim = scene()
    point = (.010, .006, .010)
    if kind == 'source':
        sim.add_source(point)
    elif kind == 'polarized':
        sim.add_polarized_source(point, polarization='ex')
    else:
        sim.add_port(point, impedance=50, excite=kind == 'zero_port')
        if kind == 'zero_port':
            sim._ports[-1] = replace(sim._ports[-1], impedance=0, amplitude_kind='current')
    refusal(monkeypatch, sim, entry, r'other excitations.*_ports\[0\]',
            **({'port_s11_freqs': np.array([5e9])} if entry == 'forward' else {}))


@pytest.mark.parametrize('kind', ['mu', 'sheet_bore', 'sheet_aperture', 'film', 'rlc'])
def test_bore_material(monkeypatch, kind):
    sim = scene('bottom' if kind == 'sheet_aperture' else 'top')
    pattern = {'mu': 'mu_r != 1', 'sheet_bore': 'surface_impedance_f0',
               'sheet_aperture': 'surface_impedance_f0', 'film': 'sigma_film', 'rlc': 'lumped RLC stamp 0'}[kind]
    if kind == 'mu':
        sim.add_material('magnetic', eps_r=2., mu_r=2.)
        sim.add(Box((.007, .005, .010), (.0075, .007, .011)), material='magnetic')
    elif kind == 'rlc':
        sim.add_lumped_rlc((.0075, .006, .010), R=50., topology='parallel', component='ex')
    else:
        z = .038 if kind == 'sheet_aperture' else .010
        sim.add_thin_conductor(Box((.005, .005, z), (.008, .008, z)),
            sigma_bulk=1e4 if kind.startswith('sheet') else 1 / (377 * 35e-6), thickness=35e-6,
            **({'surface_impedance_f0': 9e9} if kind.startswith('sheet') else {}))
    refusal(monkeypatch, sim, 'run', pattern)


def test_mu_override(monkeypatch):
    refusal(monkeypatch, scene(), 'forward', 'mu_r_override', mu_r_override=jnp.ones((1, 1, 1)))


def test_off_record_steps(monkeypatch):
    refusal(monkeypatch, scene(), 'run', 's_param_n_steps=5', s_param_n_steps=5)


@pytest.mark.parametrize('face', ['top', 'bottom'])
def test_pin_absorber(monkeypatch, face):
    sim = scene(face)
    sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(pin_length=.002)
    refusal(monkeypatch, sim, 'run', 'pin_length=.*1 cell.*pin_length <=')


def test_design_box(monkeypatch):
    refusal(monkeypatch, scene(), 'forward', 'design_box.*port-released cells',
            design_box=((.007, .0055, .010), (.0075, .0065, .011)),
            design_eps_override=jnp.ones((1, 2, 2)))


def test_traced_stamp_material():
    from rfx.core.yee import init_materials
    from rfx.model.coax_stamp import stamp_coaxial_line
    sim = scene()
    grid = sim._build_grid()
    materials = init_materials(grid.shape)
    error = None
    def stamp(eps):
        return stamp_coaxial_line(grid, materials._replace(eps_r=eps), center_xy=(.006, .006),
            z_lo_index=8, z_hi_index=82)[0].eps_r
    try:
        jax.jit(stamp)(materials.eps_r)
    except Exception as caught:
        error = caught
    check_refusal(error, 'traced material eps_r')


def check_jit(actual, expected):
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - expected)) <= 1e-4


def test_user_jit():
    sim = scene()
    bins = np.array([5e9])
    def solve():
        return sim.forward(n_steps=900, port_s11_freqs=bins, skip_preflight=True).s_params
    eager = np.asarray(solve())
    actual = np.asarray(jax.jit(solve)())
    check_jit(actual, eager)


def test_refinement_s_request(monkeypatch):
    sim = scene()
    sim.add_refinement((.010, .015), ratio=2, validation='research')
    refusal(monkeypatch, sim, 'run', 'add_coaxial_port.*not carried', compute_s_params=True)


def check_probe_records(actual, expected):
    assert actual == expected


def test_settling_keeps_user_probes():
    sim = scene()
    point = np.array((.010, .006, .001))
    sim.add_probe(tuple(point), 'ex')
    result = sim.forward(n_steps=900, port_s11_freqs=np.array([5e9]), skip_preflight=True)
    grid = sim._build_grid()
    dt = result.dt
    times = jnp.arange(result.time_series.shape[0], dtype=jnp.float32) * dt
    from rfx import GaussianPulse
    waveform = GaussianPulse(f0=5e9, bandwidth=.8)
    source = int(round(((grid.shape[2] - grid.pad_z_hi - 5 - grid.pad_z_lo) * grid.dx
                        - grid.dx / 2) / float(np.asarray(grid.cells(2))[0]))) + grid.pad_z_lo
    shift = jnp.float32(.5 * dt + .5 * grid.dx * np.sqrt(2.1) / 299792458)
    expected = 0
    for offset, plane in ((0., source + 1), (shift, source)):
        samples = np.abs(np.asarray(jax.vmap(waveform)(times + offset), dtype=np.float64))
        off = int(np.flatnonzero(samples > 1e-6 * samples.max())[-1]) + 1
        start = np.array((.006, .006, (plane - grid.pad_z_lo) * grid.dx))
        expected = max(expected, off + int(np.ceil(np.linalg.norm(start - point) / (299792458 * dt))))
    assert expected != result.settling_source_end_index
    print(f'user-probe source end={expected}, coax-plane source end={result.settling_source_end_index}')
    check_probe_records(result.settling_probe_source_end_index, expected)


def check_release(warnings, occupancy):
    assert any('Port occupancy release:' in str(w.message) for w in warnings)
    assert occupancy[16, 16, 30] == 0
    assert occupancy[5, 5, 30] == 1


def test_occupancy_warns(monkeypatch):
    import warnings
    sim = scene()
    seen = {}
    def kernel(*args, **kwargs):
        seen.update(kwargs)
        raise Captured
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with warnings.catch_warnings(record=True) as caught, pytest.raises(Captured):
        warnings.simplefilter('always')
        sim.forward(n_steps=3, skip_preflight=True, pec_occupancy_override=jnp.ones(sim._build_grid().shape))
    check_release(caught, np.asarray(seen['pec_occupancy']))


def check_plate(root, captured, ref, face):
    cells = np.asarray(root.pec_cells)
    assert not cells[19, 16, ref if face == 'top' else ref - 1]
    assert not np.asarray(captured['pec_edge_masks'][0])[19, 16, ref]
    assert not np.asarray(captured['pec_edge_masks'][1])[19, 16, ref]
    if face == 'bottom':
        assert root.coax_bore_release.edges[2].size == 0 or not np.any(root.coax_bore_release.edges[2][:, 2] == ref)


@pytest.mark.parametrize('face', ['top', 'bottom'])
def test_plate_at_reference(monkeypatch, face):
    sim = scene(face)
    ref_z = .002 if face == 'top' else .038
    lo, hi = (ref_z, ref_z+.001) if face == 'top' else (ref_z-.001, ref_z)
    sim.add(Box((.003, .003, lo), (.009, .009, hi)), material='pec')
    captured = capture(monkeypatch, sim, 'run')
    root = sim.realized_geometry().conductors
    check_plate(root, captured, 8 if face == 'top' else 80, face)


def check_report(report, expected_cells, expected_edges):
    assert set(report.cleared_bore_cells) == expected_cells
    assert tuple(map(set, report.cleared_bore_edges)) == expected_edges


def test_bore_report(monkeypatch):
    sim = scene()
    sim.add(Box((.003, .003, .010), (.009, .009, .011)), material='pec')
    captured = capture(monkeypatch, sim, 'run')
    record = sim.realized_geometry()
    grid = captured['grid']
    xy = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
          if ((i-4)*.0005-.006)**2 + ((j-4)*.0005-.006)**2 < .002055**2]
    cells = {(i, j, k) for i, j in xy for k in (24, 25)}
    edges = tuple({(i, j, k) for i, j in xy for k in ((24, 25, 26) if a < 2 else (24, 25))} for a in range(3))
    check_report(record.ports[0], cells, edges)


def check_forward_report(record):
    assert record.lane == 'fwd_uniform'
    assert any(e.entity_id.endswith('/shell') for e in record.conductors.stamped_entities)
    assert record.edge_masks and any(np.any(e) for e in record.edge_masks)


def test_forward_diagnostic_arrays():
    sim = scene()
    result = sim.forward(n_steps=3, skip_preflight=True)
    assert result.realized_geometry.lane == 'fwd_uniform'
    from rfx.model.realization_report import rebuild_view
    check_forward_report(rebuild_view(sim, result.realized_geometry, False))
