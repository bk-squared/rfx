"""S1 model and drive identity at the low-level runner handover."""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.geometry import Box


class Captured(Exception):
    pass


def scene(face='top', **kwargs):
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .040), dx=.0005,
                     boundary='cpml', cpml_layers=4, snap='declared', **kwargs)
    sim.add_coaxial_port((.006, .006, .002 if face == 'top' else .038),
                         face=face, pin_length=0)
    return sim


def capture(monkeypatch, sim, entry):
    seen = {}
    def kernel(grid, materials, n_steps, **kwargs):
        seen.update(grid=grid, materials=materials, **kwargs)
        raise Captured
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with pytest.raises(Captured):
        if entry == 'calculator':
            sim.compute_coaxial_line_reflection(termination='open', n_steps=3,
                                                freqs=np.array([5e9]))
        else:
            getattr(sim, entry)(n_steps=3, skip_preflight=True)
    return seen


def equal(actual, expected):
    for field in actual['materials']._fields:
        left, right = getattr(actual['materials'], field), getattr(expected['materials'], field)
        assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right), field
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
            assert np.array_equal(a, b), field
    for name in ('pec_edge_masks', 'sources', 'mag_sources'):
        left, right = actual[name], expected[name]
        assert len(left) == len(right), name
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
            assert np.array_equal(a, b), name
        print(f'S1 {name}: array_equal', file=sys.stderr)


def test_run_against_calculator(monkeypatch):
    expected = capture(monkeypatch, scene(), 'calculator')
    actual = capture(monkeypatch, scene(), 'run')
    equal(actual, expected)
    grid = actual['grid']
    ref = grid.pad_z_lo + 4
    hi = grid.shape[2] - grid.pad_z_hi - 2
    source = int(round(((hi - 3 - grid.pad_z_lo) * grid.dx - grid.dx / 2)
                       / float(np.asarray(grid.cells(2))[0]))) + grid.pad_z_lo
    print(f'S1 B1 calculator: ref={ref}, span=({ref},{hi}), feed={hi-1}, source={source}, planes={tuple(ref+8+4*k for k in range(12))}', file=sys.stderr)
    layout = actual_layout(actual, scene())
    assert (layout.reference_index, layout.z_lo_index, layout.z_hi_index,
            layout.feed_index, layout.source_index, layout.plane_indices) == (
                ref, ref, hi, hi-1, source, tuple(ref+8+4*k for k in range(12)))
    print(f'S1 B1 model: {layout}', file=sys.stderr)


def actual_layout(captured, sim):
    from rfx.model.coax_line import coax_port_layout
    grid = captured['grid']
    pads = tuple(getattr(grid, f'pad_{a}_{s}') for a in 'xyz' for s in ('lo', 'hi'))
    return coax_port_layout(grid, sim._coaxial_ports[0], absorber_pads=pads, ntff=sim._ntff)


def test_forward_against_run(monkeypatch):
    sim = scene()
    expected = capture(monkeypatch, sim, 'run')
    actual = capture(monkeypatch, scene(), 'forward')
    equal(actual, expected)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_bore_clears_plate(monkeypatch, entry):
    sim = scene()
    sim.add(Box((.001, .001, .010), (.011, .011, .012)), material='pec')
    sim.add_probe((.009, .006, .005), 'ez')
    captured = capture(monkeypatch, sim, entry)
    record = sim.realized_geometry()
    root = record.conductors
    release = root.coax_bore_release
    assert release is not None
    port_report = next(p for p in record.ports if p.kind == 'coaxial')
    assert port_report.error is None
    assert port_report.aperture[2] == (release.layout.z_lo_index, release.layout.z_hi_index+1)
    assert len(release.cells) > 0
    grid = captured['grid']
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * grid.dx - .006
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * grid.dx - .006
    radial = x[:, None] ** 2 + y[None, :] ** 2 < .002055 ** 2
    region = np.zeros(grid.shape, bool)
    region[:, :, release.layout.z_lo_index:release.layout.z_hi_index+1] = radial[:, :, None]
    pin = next(e for e in root.stamped_entities if e.entity_id.endswith('/pin')).cells
    shell = next(e for e in root.stamped_entities if e.entity_id.endswith('/shell')).cells
    from tests.contracts.test_coax_conductors_known import _edges
    owned = _edges(pin | shell)
    check_bore(root, captured, region, pin, shell, owned)
    assert set(map(tuple, release.cells)).issubset(set(map(tuple, root.released_cells)))
    print(f'S1 bore {entry}: cleared cells={len(release.cells)}, edges={tuple(len(e) for e in release.edges)}, occupancy={len(release.occupancy_cells)}', file=sys.stderr)


def check_bore(root, captured, region, pin, shell, owned):
    assert not np.any(np.asarray(root.pec_cells) & region & ~pin & ~shell)
    for actual, stamped in zip(captured['pec_edge_masks'], owned):
        assert not np.any(np.asarray(actual) & region & ~stamped)


def test_bottom_is_node_mirror(monkeypatch):
    top_sim, bottom_sim = scene(), scene('bottom')
    top = capture(monkeypatch, top_sim, 'run')
    bottom = capture(monkeypatch, bottom_sim, 'run')
    roots = [s.realized_geometry().conductors for s in (top_sim, bottom_sim)]
    layouts = [r.coax_bore_release.layout for r in roots]
    a, b = layouts
    assert (b.z_lo_index, b.z_hi_index) == (b.reference_index - (a.z_hi_index-a.reference_index) - 1, b.reference_index-1)
    assert b.feed_index-b.reference_index == -(a.feed_index-a.reference_index)
    assert b.source_index-b.reference_index == -(a.source_index-a.reference_index)
    assert tuple(p-b.reference_index for p in b.plane_indices) == tuple(a.reference_index-p for p in a.plane_indices)
    shell = next(e.cells for e in roots[1].stamped_entities if e.entity_id.endswith('/shell'))
    assert np.where(shell)[2].max()+1 == b.reference_index
    assert np.where(next(e.cells for e in roots[0].stamped_entities if e.entity_id.endswith('/shell')))[2].min() == a.reference_index
    assert top['mag_sources'] and bottom['mag_sources']


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('gate,pattern', [
    ('precision', 'float32'), ('stencil_order', 'stencil_order=2'), ('mode', "mode='3d'"),
    ('boundary', 'CPML on all six faces'), ('zero_cpml', 'CPML on all six faces'),
    ('periodic', 'periodic'), ('tfsf', 'TF/SF'),
    ('face', 'face='), ('count', 'exactly one'), ('mixed', 'other (S-port|excitations)'),
    ('conformal_pec', 'conformal_pec'), ('subpixel_smoothing', 'subpixel_smoothing'),
    ('short', '61 line cells'), ('lateral', 'absorber'), ('front', 'user PEC in front'),
    ('kerr', 'Kerr material inside the bore'),
])
def test_s1_refusals(monkeypatch, entry, gate, pattern):
    sim = scene(**({'precision': 'float64'} if gate == 'precision' else
                   {'stencil_order': 4} if gate == 'stencil_order' else {}))
    options = {}
    if gate == 'mode':
        sim._mode = '2d_tmz'
    elif gate == 'boundary':
        sim._boundary = 'pec'
    elif gate == 'zero_cpml':
        sim._cpml_layers = 0
    elif gate == 'periodic':
        sim._periodic_axes = 'x'
    elif gate == 'tfsf':
        sim.add_tfsf_source()
    elif gate == 'face':
        sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(face='left')
    elif gate == 'count':
        sim.add_coaxial_port((.006, .006, .002))
    elif gate == 'mixed':
        sim.add_port((.008, .006, .004), 'ez')
    elif gate in ('conformal_pec', 'subpixel_smoothing'):
        if entry == 'forward' and gate == 'conformal_pec':
            from rfx.runners._admission import admit
            with pytest.raises(NotImplementedError, match=pattern):
                admit(sim, 'fwd_uniform', run_args={'conformal_pec': True})
            return
        options[gate] = True
    elif gate == 'short':
        sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(position=(.006, .006, .025))
    elif gate == 'lateral':
        sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(position=(.0025, .006, .002))
    elif gate == 'front':
        sim.add(Box((.005, .005, .0005), (.007, .007, .0015)), material='pec')
    elif gate == 'kerr':
        sim.add_material('kerr', eps_r=2., chi3=1e-20)
        sim.add(Box((.005, .005, .010), (.007, .007, .011)), material='kerr')
    if gate not in ('short', 'lateral', 'front', 'kerr'):
        def unexpected(*args, **kwargs):
            pytest.fail('declaration refusal must precede mesh planning')
        monkeypatch.setattr(sim, '_auto_configure_mesh', unexpected)
        monkeypatch.setattr(sim, '_build_realized_grid', unexpected)
        monkeypatch.setattr(sim, '_build_grid', unexpected)
    with pytest.raises((ValueError, NotImplementedError), match=pattern):
        getattr(sim, entry)(n_steps=3, skip_preflight=True, **options)


@pytest.mark.parametrize('name', ['eps_override', 'sigma_override'])
def test_forward_material_override_refuses(name):
    sim = scene()
    with pytest.raises(NotImplementedError, match=name):
        sim.forward(n_steps=3, skip_preflight=True, **{name: jnp.ones((1, 1, 1))})


@pytest.mark.parametrize('face', ['top', 'bottom'])
def test_pin_only_continues_in_front(monkeypatch, face):
    sim = scene(face)
    port = sim._coaxial_ports[0]
    sim._coaxial_ports[0] = port._replace(pin_length=.001)
    captured = capture(monkeypatch, sim, 'run')
    root = sim.realized_geometry().conductors
    ref = root.coax_bore_release.layout.reference_index
    pin = next(e.cells for e in root.stamped_entities if e.entity_id.endswith('/pin_extension'))
    indices = np.where(pin)[2]
    assert (indices.min(), indices.max()) == ((ref-2, ref-1) if face == 'top' else (ref, ref+1))
    shell = next(e.cells for e in root.stamped_entities if e.entity_id.endswith('/shell'))
    front = slice(None, ref) if face == 'top' else slice(ref, None)
    assert not shell[:, :, front].any()
    assert captured['mag_sources']


@pytest.mark.parametrize('face', ['top', 'bottom'])
def test_ntff_limits_realized_far_node(monkeypatch, face):
    sim = scene(face)
    sim.add_ntff_box((.001, .001, .001), (.011, .011, .039), freqs=[5e9])
    captured = capture(monkeypatch, sim, 'run')
    root = sim.realized_geometry().conductors
    layout = root.coax_bore_release.layout
    grid = captured['grid']
    assert (layout.z_lo_index, layout.z_hi_index) == ((8, 80) if face == 'top' else (7, 79))
    last_node = layout.z_hi_index+1 if face == 'top' else layout.z_lo_index
    assert (last_node-grid.pad_z_lo)*grid.dx == pytest.approx(.0385 if face == 'top' else .0015)


def test_forward_releases_occupancy(monkeypatch):
    sim = scene()
    baseline = capture(monkeypatch, sim, 'run')
    seen = {}
    def kernel(grid, materials, n_steps, **kwargs):
        seen.update(kwargs)
        raise Captured
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with pytest.raises(Captured):
        scene().forward(n_steps=3, skip_preflight=True,
                        pec_occupancy_override=jnp.ones(baseline['grid'].shape))
    occupancy = np.asarray(seen['pec_occupancy'])
    assert occupancy[16, 16, 30] == 0
    assert occupancy[5, 5, 30] == 1


def test_dispersive_bore_refuses():
    from rfx.materials.debye import DebyePole
    sim = scene()
    sim.add_material('dispersive', eps_r=2., debye_poles=[DebyePole(1., 1e-10)])
    sim.add(Box((.005, .005, .010), (.007, .007, .011)), material='dispersive')
    with pytest.raises(NotImplementedError, match='dispersive or Kerr material inside the bore'):
        sim.run(n_steps=3, skip_preflight=True)


def test_named_pin_contact_is_allowed(monkeypatch):
    sim = scene()
    contact = Box((.0055, .0055, .001), (.0065, .0065, .002))
    sim.add(contact, material='pec')
    from rfx.geometry.port_termination import resolve_terminates
    sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(pin_length=.001,
        terminates=resolve_terminates(sim, contact, port='test contact'))
    assert capture(monkeypatch, sim, 'run')['sources']


def test_ntff_must_contain_realized_shell(monkeypatch):
    sim = scene()
    sim.add_ntff_box((.003, .003, .001), (.009, .009, .039), freqs=[5e9])
    with pytest.raises(NotImplementedError, match='NTFF box must contain the realized shell'):
        capture(monkeypatch, sim, 'run')


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_each_boundary_face_must_be_cpml(entry):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    sim = scene()
    sim._boundary_spec = BoundarySpec(x=Boundary(lo='pec', hi='cpml'),
                                    y=Boundary(lo='cpml', hi='cpml'),
                                    z=Boundary(lo='cpml', hi='cpml'))
    with pytest.raises(NotImplementedError, match='CPML on all six faces'):
        getattr(sim, entry)(n_steps=3, skip_preflight=True)
