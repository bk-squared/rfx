"""Review regressions: every unsupported input is named before stepping."""
from dataclasses import replace
from types import SimpleNamespace
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation, drude_pole
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
    return error


def check_drude_admitted(captured, error, box):
    """The pole couples only to Yee edges incident on the plasma box's cells."""
    assert error is None, repr(error)
    assert captured['lorentz'] is not None
    coeffs, _ = captured['lorentz']
    cells = np.asarray(box.mask(captured['grid']))
    assert cells.any()
    for axis, component in enumerate(coeffs.c):
        coupling = np.asarray(component)
        assert coupling.shape == (1,) + cells.shape
        assert np.all(coupling[0][cells] > 0)
        edges = cells.copy()
        for transverse in range(3):
            if transverse != axis:
                edges |= np.roll(edges, 1, axis=transverse)
        assert np.all(coupling[0][edges] > 0)
        assert np.all(coupling[0][~edges] == 0)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_drude_outside_bore_admitted(monkeypatch, entry):
    """The box ends at x = y = .003, beyond the bore centred at .006."""
    sim = scene()
    sim.add_material('plasma', eps_r=1., lorentz_poles=[drude_pole(2*np.pi*5e9, 1e9)])
    box = Box((.001, .001, .020), (.003, .003, .022))
    sim.add(box, material='plasma')
    port = sim._coaxial_ports[0]
    assert np.hypot(.006 - .003, .006 - .003) > port.outer_radius
    captured, error = None, None
    try:
        captured = capture(monkeypatch, sim, entry)
    except Exception as caught:
        error = caught
    check_drude_admitted(captured, error, box)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_drude_inside_bore_refused(monkeypatch, entry):
    sim = scene()
    sim.add_material('plasma', eps_r=1., lorentz_poles=[drude_pole(2*np.pi*5e9, 1e9)])
    sim.add(Box((.005, .005, .010), (.007, .007, .011)), material='plasma')
    refusal(monkeypatch, sim, entry, r"material 'plasma' has a Drude pole")


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


def test_plain_source_message(monkeypatch):
    sim = scene()
    sim.add_source((.010, .006, .010))
    refusal(monkeypatch, sim, 'run', 'compute_s_params=False')


def check_substrate_admitted(error):
    assert isinstance(error, Captured), repr(error)


@pytest.mark.parametrize('face', ['top', 'bottom'])
@pytest.mark.parametrize('kind', ['debye', 'magnetic', 'kerr'])
@pytest.mark.parametrize('inside', [False, True])
def test_reference_substrate(monkeypatch, face, kind, inside):
    sim = scene(face)
    sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(pin_length=.001)
    properties = {'debye': {'debye_poles': [DebyePole(delta_eps=1., tau=8e-12)]},
                  'magnetic': {'mu_r': 2.}, 'kerr': {'chi3': 1e-20}}[kind]
    sim.add_material('substrate', eps_r=2., **properties)
    lo, hi = ((.001, .002) if face == 'top' else (.038, .039))
    if inside:
        lo, hi = (.010, .011)
    sim.add(Box((.003, .003, lo), (.009, .009, hi)), material='substrate')
    if inside:
        refusal(monkeypatch, sim, 'run', 'mu_r != 1' if kind == 'magnetic' else 'dispersive or Kerr')
        return
    def kernel(*args, **kwargs):
        raise Captured('reached time stepping')
    monkeypatch.setattr('rfx.simulation.run', kernel)
    error = None
    try:
        sim.run(n_steps=3, skip_preflight=True)
    except Exception as caught:
        error = caught
    check_substrate_admitted(error)


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


@pytest.mark.parametrize('kind', ['film', 'sheet', 'rlc'])
@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('mask_axis', range(3))
def test_edge_material_component(monkeypatch, kind, axis, mask_axis):
    from rfx._internal._grid_metric import field_index
    from rfx.model.coax_admission import refuse_bore_materials
    sim = scene()
    grid = sim._build_grid()
    component = ('ex', 'ey', 'ez')[axis]
    position = (.0075, .006, .010)
    edge = field_index(grid, position, component)
    bore = np.zeros(grid.shape, dtype=bool)
    edge_bore = [bore.copy() for _ in range(3)]
    edge_bore[mask_axis][edge] = True
    parts = [None] * 3
    sheets = []
    if kind == 'film':
        parts[axis] = np.zeros(grid.shape)
        parts[axis][edge] = 1.
    elif kind == 'sheet':
        sheets = [(0, component, np.array([edge]))]
    else:
        sim.add_lumped_rlc(position, R=50., topology='parallel', component=component)
    monkeypatch.setattr('rfx.model.design_admission.sheet_edges', lambda *args: sheets)
    materials = SimpleNamespace(mu_r=np.ones(grid.shape), sigma_film=tuple(parts),
                                _fields=('mu_r', 'sigma_film'))
    root = SimpleNamespace(grid=grid)
    if axis == mask_axis:
        with pytest.raises(RfxUnsupported, match={'film': 'sigma_film',
                'sheet': 'surface_impedance_f0', 'rlc': 'lumped RLC'}[kind]):
            refuse_bore_materials(sim, root, materials, bore, edge_bore)
    else:
        refuse_bore_materials(sim, root, materials, bore, edge_bore)


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
    port = sim._coaxial_ports[0]
    s = max(2, int(np.ceil(port.outer_radius / grid.dx))) + 1
    source += -s if port.face == 'top' else s
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
    root = captured['conductors']
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


def enclosed_line_scene(*, name_contact=True):
    from rfx.geometry.port_termination import resolve_terminates
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .048), dx=.0005,
                     boundary='cpml', cpml_layers=4, snap='declared')
    bottom = Box((.003, .003, .0375), (.009, .009, .038))
    contact = Box((.004, .0055, .040), (.008, .0065, .0405))
    top = Box((.003, .003, .043), (.009, .009, .0435))
    walls = (Box((.003, .003, .038), (.0035, .009, .043)),
             Box((.0085, .003, .038), (.009, .009, .043)),
             Box((.003, .003, .038), (.009, .0035, .043)),
             Box((.003, .0085, .038), (.009, .009, .043)))
    for box in (bottom, contact, top, *walls):
        sim.add(box, material='pec')
    sim.add_coaxial_port((.006, .006, .038), face='bottom', pin_length=.002)
    sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(
        terminates=resolve_terminates(sim, (bottom, contact) if name_contact else (bottom,),
                                     port='enclosed line'))
    return sim, top


def check_enclosed_line(captured, error, sim, top):
    assert error is None, repr(error)
    assert captured['sources'] and captured['mag_sources']
    grid = captured['grid']
    port = sim._coaxial_ports[0]
    tip = grid.position_to_index((.006, .006, .040))[2]
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * grid.dx - port.position[0]
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * grid.dx - port.position[1]
    radial = x[:, None] ** 2 + y[None, :] ** 2 < port.outer_radius ** 2
    cells = np.asarray(top.mask(grid))
    beyond = cells[:, :, tip + 1:] & radial[:, :, None]
    assert beyond.any()
    assert np.all(np.asarray(captured['pec_edge_masks'][2])[:, :, tip + 1:][beyond])
    assert np.where(cells)[2].min() >= tip + 4


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_enclosed_line_admitted(monkeypatch, entry):
    sim, top = enclosed_line_scene()
    captured, error = None, None
    try:
        captured = capture(monkeypatch, sim, entry)
    except Exception as caught:
        error = caught
    check_enclosed_line(captured, error, sim, top)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_span_metal_refused(monkeypatch, entry):
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.model.coax_line import coax_port_layout
    from rfx.model.coax_port import stamp_coax_pin
    sim, _ = enclosed_line_scene()
    metal = Box((.0075, .006, .0385), (.008, .0065, .039))
    sim.add(metal, material='pec')
    grid = sim._build_grid()
    port = sim._coaxial_ports[0]
    pads = tuple(getattr(grid, f'pad_{a}_{s}') for a in 'xyz' for s in ('lo', 'hi'))
    layout = coax_port_layout(grid, port, absorber_pads=pads)
    pin_edges = realized_pec_edge_masks(stamp_coax_pin(grid, port, layout))
    metal_edges = realized_pec_edge_masks(np.asarray(metal.mask(grid)))
    assert any(np.any(e) for e in metal_edges)
    assert not any(np.any(a & b) for a, b in zip(pin_edges, metal_edges))
    ref = layout.reference_index
    tip = grid.position_to_index((.006, .006, .040))[2]
    message = (f"add_coaxial_port(): user PEC in front of the reference plane inside the bore, "
               f"between the reference node {ref} and the pin's tip node {tip}; "
               "name the pin contact in terminates=; a named entry that also reaches the outer "
               "conductor terminates the line in a short at the reference plane")
    error = refusal(monkeypatch, sim, entry, re.escape(message))
    print(str(error))


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_unnamed_tip_contact_refused(monkeypatch, entry):
    sim, _ = enclosed_line_scene(name_contact=False)
    refusal(monkeypatch, sim, entry, r"user PEC in front.*between the reference node .*pin's tip node")


def check_realized_coax(before, after, metadata, pin_length):
    assert before.error is None
    assert isinstance(before.z_tem_ohm, float)
    assert np.isfinite(before.z_tem_ohm) and before.z_tem_ohm > 0
    assert bool(before.pin_cells) == bool(pin_length)
    assert before == after
    assert before.z_tem_ohm == metadata['z_tem_ohm']


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('pin_length', [0., .0005])
def test_realized_coax_before_run(entry, pin_length):
    from dataclasses import fields
    from rfx.realized_geometry import realized_geometry
    from rfx.model.realization_report import rebuild_view
    sim = scene()
    sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(pin_length=pin_length)
    before = realized_geometry(sim).ports[0]
    assert getattr(sim, '_coax_geometry', None) is None
    assert realized_geometry(sim).ports[0] == before
    options = (dict(s_param_freqs=np.array([5e9])) if entry == 'run'
               else dict(port_s11_freqs=np.array([5e9])))
    result = getattr(sim, entry)(n_steps=3, skip_preflight=True, **options)
    after = rebuild_view(sim, result.realized_geometry, False).ports[0]
    metadata = result.coax_port_metadata['coax_port:0:mode:0']
    print({f.name: (len(getattr(before, f.name)) if f.name in ('pin_cells', 'cleared_bore_cells')
           else tuple(map(len, before.cleared_bore_edges)) if f.name == 'cleared_bore_edges'
           else getattr(before, f.name)) for f in fields(before)})
    check_realized_coax(before, after, metadata, pin_length)


def check_realized_coax_refusal(row, error):
    assert row.error == str(error)
    assert row.z_tem_ohm is None
    assert row.pin_cells == ()
    assert row == type(row)('coaxial_port[0]', 'coaxial', None, (), error=str(error))


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_realized_coax_refusal_before_run(monkeypatch, entry):
    from rfx.realized_geometry import realized_geometry
    sim, _ = enclosed_line_scene()
    sim.add(Box((.0075, .006, .0385), (.008, .0065, .039)), material='pec')
    record = realized_geometry(sim)
    assert record.conductors is not None
    assert record.realization_table is not None
    row = record.ports[0]
    error = refusal(monkeypatch, sim, entry, r'user PEC in front.*pin.s tip node')
    print(row)
    check_realized_coax_refusal(row, error)
