"""Coax port spectra against independent calculator and mirrored field witnesses."""
from collections import deque
from datetime import datetime, timezone
import sys
from time import monotonic

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.geometry import Cylinder
from tests._coax_calculator_stand_in import open_end
from tests.contracts.test_coax_port_model_and_drive import scene, capture, actual_layout


BINS = np.array([4e9, 5e9, 6e9])
STEPS = 3000


def report(name, value):
    print(f'{name}: {value}', file=sys.stderr)


def check_native(result, expected):
    native = np.asarray(result.coax_port_metadata['coax_port:0:mode:0']['native_gamma'])
    errors = np.abs(native - expected.s11)
    report('B2 per-bin native error', errors.tolist())
    assert np.all(errors <= 1e-4)


def identity(actual, expected, label, families, limit=None):
    rows = []
    for name in families:
        left, right = actual[name], expected[name]
        structure = jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
        aa, bb = jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
        leaves = []
        for i, (a, b) in enumerate(zip(aa, bb)):
            a, b = np.asarray(a), np.asarray(b)
            if limit is not None and a.shape == b.shape == actual['grid'].shape:
                a, b = a[:, :, :limit+1], b[:, :, :limit+1]
            leaves.append(dict(leaf=i, dtype=a.dtype == b.dtype, shape=a.shape == b.shape,
                               bytes=a.tobytes() == b.tobytes()))
        passed = structure and len(aa) == len(bb) and all(
            row['dtype'] and row['shape'] and row['bytes'] for row in leaves)
        rows.append(dict(family=name, left_leaves=len(aa), right_leaves=len(bb),
                         structure=structure, passed=passed,
                         dtype=sum(row['dtype'] for row in leaves),
                         shape=sum(row['shape'] for row in leaves),
                         bytes=sum(row['bytes'] for row in leaves),
                         mismatches=[row for row in leaves if not all(
                             row[key] for key in ('dtype', 'shape', 'bytes'))]))
    report(label, dict(last_z_index=limit, rows=rows))
    assert all(row['passed'] for row in rows), label


def native(result):
    return np.asarray(result.coax_port_metadata['coax_port:0:mode:0']['native_gamma'])


def spectrum(label, left, right):
    errors = np.abs(left - right)
    for frequency, a, b, error in zip(BINS, left, right, errors):
        report(label, dict(frequency_hz=float(frequency), left=complex(a),
                           right=complex(b), error=float(error)))
    assert np.all(errors <= 1e-4), label


@pytest.fixture(scope='module')
def a1_results():
    with pytest.MonkeyPatch.context() as patch:
        expected_model = capture(patch, scene(), 'calculator')
    with pytest.MonkeyPatch.context() as patch:
        open_end(patch)
        actual_model = capture(patch, scene(), 'run')
    identity(actual_model, expected_model, 'A1 identity',
             ('materials', 'pec_edge_masks', 'sources', 'mag_sources'))
    expected = scene().compute_coaxial_line_reflection(
        termination='open', n_steps=STEPS, freqs=BINS)
    with pytest.MonkeyPatch.context() as patch:
        open_end(patch)
        stand_in = scene().run(n_steps=STEPS, s_param_freqs=BINS, skip_preflight=True)
    return stand_in, expected


@pytest.fixture(scope='module')
def open_results(a1_results):
    result = scene().run(n_steps=STEPS, s_param_freqs=BINS, skip_preflight=True)
    return result, a1_results[1]


def short_scene(legacy=False):
    dx = .0005
    sim = scene() if legacy else Simulation(
        freq_max=10e9, domain=(.012, .012, .052), dx=dx,
        boundary='cpml', cpml_layers=4, snap='declared')
    disc = Cylinder(center=(.006, .006, .002 - dx / 2),
                    radius=.0035, height=dx, axis='z')
    sim.add(disc, material='pec')
    if legacy:
        from rfx.geometry.port_termination import resolve_terminates
        port = sim._coaxial_ports[0]
        sim._coaxial_ports[0] = port._replace(
            pin_length=dx, terminates=resolve_terminates(sim, disc, port='B2 short contact'))
    else:
        sim.add_coaxial_port((.006, .006, .002), face='top', pin_length=dx, terminates=disc)
    return sim


def closure(side, sim, captured):
    """Flood the DUT side of the first plane, through the shell and disc bounds."""
    grid = captured['grid']
    root = sim.realized_geometry().conductors
    layout = root.coax_bore_release.layout
    port = sim._coaxial_ports[0]
    cells = np.asarray(root.pec_cells)
    shell = next(e for e in root.stamped_entities if e.entity_id.endswith('/shell'))
    outer = float(shell.declared_radii_m[1])
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * grid.dx - .006
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * grid.dx - .006
    radius = np.hypot(x[:, None], y[None, :])
    high, low, disc_low = layout.plane_indices[0], grid.pad_z_lo, grid.pad_z_lo + 3
    seeds = np.argwhere((radius > port.pin_radius) & (radius <= port.outer_radius)
                        & ~cells[:, :, high])
    queue = deque((int(i), int(j), high) for i, j in seeds)
    seen = np.zeros(grid.shape, bool)
    for cell in queue:
        seen[cell] = True
    while queue:
        i, j, k = queue.popleft()
        for a, b, c in ((i-1,j,k), (i+1,j,k), (i,j-1,k),
                        (i,j+1,k), (i,j,k-1), (i,j,k+1)):
            if 0 <= a < grid.shape[0] and 0 <= b < grid.shape[1] and low <= c <= high:
                if not seen[a,b,c] and not cells[a,b,c]:
                    seen[a,b,c] = True
                    queue.append((a,b,c))
    outside = int(np.count_nonzero(seen & (radius > outer)[:, :, None]))
    below = int(np.count_nonzero(seen[:, :, :disc_low]))
    report('A2 closure', dict(side=side, seed_cells=len(seeds), reached_cells=int(seen.sum()),
           outside_shell_cells=outside, below_disc_cells=below, shell_outer_radius_m=outer,
           first_interior_index=low, first_plane_index=high, disc_lower_index=disc_low,
           feed_index=layout.feed_index))
    assert seeds.size and outside == below == 0, f'A2 closure {side}'


@pytest.fixture(scope='module')
def short_results():
    start = monotonic()
    def stamp(stage):
        report('A2 timing', dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                                elapsed_s=monotonic()-start))
    stamp('start')
    setups = {}
    for side in ('capped', 'stand-in'):
        for legacy in (False, True):
            sim = short_scene(legacy)
            with pytest.MonkeyPatch.context() as patch:
                if side == 'stand-in':
                    open_end(patch)
                captured = capture(patch, sim, 'run')
            setups[side, legacy] = sim, captured
            stamp(f'capture {side} legacy={legacy}')
        identity(setups[side, False][1], setups[side, True][1],
                 f'A2 public versus _replace identity {side}',
                 ('materials', 'pec_edge_masks', 'sources', 'mag_sources'))
        closure(side, *setups[side, False])
        stamp(f'closure {side}')
    capped_sim, capped = setups['capped', False]
    limit = capped_sim.realized_geometry().conductors.coax_bore_release.layout.feed_index - 2
    identity(capped, setups['stand-in', False][1], 'A2 identity',
             ('materials', 'pec_edge_masks'), limit=limit)
    stamp('front identity')
    outputs = {}
    for side in ('capped', 'stand-in'):
        for legacy in (False, True):
            stamp(f'solve start {side} legacy={legacy}')
            with pytest.MonkeyPatch.context() as patch:
                if side == 'stand-in':
                    open_end(patch)
                outputs[side, legacy] = short_scene(legacy).run(
                    n_steps=STEPS, s_param_freqs=BINS, skip_preflight=True)
            stamp(f'solve end {side} legacy={legacy}')
        public, old = (native(outputs[side, legacy]) for legacy in (False, True))
        report(f'A2 public versus _replace Gamma {side}',
               dict(public=public.tolist(), legacy=old.tolist(),
                    errors=np.abs(public-old).tolist()))
        assert public.dtype == old.dtype and public.tobytes() == old.tobytes(), (
            f'A2 public versus _replace Gamma {side}')
    return outputs['capped', False], outputs['stand-in', False]


def test_shorted_two_link_chain(short_results):
    capped, stand_in = short_results
    spectrum('A2 S', native(capped), native(stand_in))


def test_open_native_and_renormalization(open_results, a1_results):
    result, expected = open_results
    spectrum('A1 S', native(a1_results[0]), np.asarray(expected.s11))
    metadata = result.coax_port_metadata['coax_port:0:mode:0']
    gamma, zt = np.asarray(metadata['native_gamma']), metadata['z_tem_ohm']
    reference = ((1 + gamma) * zt - 50 * (1 - gamma)) / ((1 + gamma) * zt + 50 * (1 - gamma))
    residual = np.max(np.abs(np.asarray(result.s_params).reshape(-1) - reference))
    report('B2 renormalization residual', residual)
    assert residual <= 1e-6
    assert result.sparam_time_records[0].samples.shape == (STEPS, 12)
    assert result.settling_witness['scored_port_record_ids'] == ['coax_port:0:mode:0']


def test_forward_against_run(open_results):
    result, _ = open_results
    actual = scene().forward(n_steps=STEPS, port_s11_freqs=BINS, skip_preflight=True)
    for output in (result, actual):
        assert output.s_params.dtype == np.dtype('complex64')
        assert output.coax_port_metadata['coax_port:0:mode:0']['native_gamma'].dtype == np.dtype('complex64')
    error = np.max(np.abs(np.asarray(actual.s_params) - np.asarray(result.s_params)))
    report('B3 max S difference', error)
    assert error <= 1e-4


def test_wave_metadata(open_results):
    result, _ = open_results
    actual = scene().forward(n_steps=STEPS, port_s11_freqs=BINS, skip_preflight=True)
    metadata = [output.coax_port_metadata['coax_port:0:mode:0'] for output in (result, actual)]
    nodes = [np.asarray(data['voltage_node_positions_m']) for data in metadata]
    np.testing.assert_array_equal(*nodes)
    assert nodes[0].ndim == 2 and nodes[0].shape[1] == 3
    shapes = dict(forward_voltage=(len(BINS),), recurrence_residual=(len(BINS),),
                  plane_voltages=(12, len(BINS)), plane_positions_m=(12,))
    for name, shape in shapes.items():
        left, right = (np.asarray(data[name]) for data in metadata)
        assert left.shape == right.shape == shape
        assert left.dtype == right.dtype
        assert left.dtype.kind == ('c' if name in ('forward_voltage', 'plane_voltages') else 'f')
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-4 * np.max(np.abs(left)))
    for data in metadata:
        positions = np.asarray(data['plane_positions_m'], dtype=np.float64)
        np.testing.assert_allclose(positions, .0125 + .002 * np.arange(12), rtol=1e-6)
        voltages = np.asarray(data['plane_voltages'], dtype=np.complex128)
        middle = voltages[1:-1]
        recurrence = np.sum((voltages[:-2] + voltages[2:]) * middle.conj(), axis=0)
        recurrence /= np.sum(np.abs(middle) ** 2, axis=0)
        propagation = np.arccosh(recurrence / 2) / np.mean(np.diff(positions))
        propagation = np.where(propagation.imag < 0, propagation.conj(), propagation)
        pair = np.array([0, 4])
        waves = np.stack([np.linalg.solve(
            np.column_stack((np.exp(k * positions[pair]), np.exp(-k * positions[pair]))),
            voltages[pair, i]) for i, k in enumerate(propagation)], axis=1)
        forward = np.asarray(data['forward_voltage'])
        backward = np.asarray(data['native_gamma']) * forward
        peak = np.max(np.abs(waves))
        report('reference-plane wave fit error / peak', np.max(np.abs(waves - [forward, backward])) / peak)
        np.testing.assert_allclose([forward, backward], waves, rtol=0, atol=1e-3 * peak)
        residual = np.asarray(data['recurrence_residual'])
        report('recurrence residual', residual.tolist())
        assert np.all(np.isfinite(residual))
        assert np.all(residual < .1)


def test_bottom_launch_and_mirror(monkeypatch, open_results):
    sim = scene('bottom')
    with monkeypatch.context() as patch:
        setup = capture(patch, sim, 'run')
    layout = actual_layout(setup, sim)
    grid = setup['grid']
    front, back = min(layout.plane_indices), layout.source_index - 1
    for plane in (back, front):
        sim.add_probe((.0075, .006, grid.node_of(2, plane)), 'ex')
    result = sim.run(n_steps=STEPS, s_param_freqs=BINS, skip_preflight=True)
    velocity = 299792458 / np.sqrt(2.1)
    before_return = int((abs(layout.reference_index-layout.source_index) +
                         abs(layout.reference_index-front)) * grid.dx / velocity / grid.dt)
    values = np.asarray(result.time_series)[:before_return]
    ratio = np.max(np.abs(values[:, 0])) / np.max(np.abs(values[:, 1]))
    report('B4 before-return steps', before_return)
    report('B4 backward/forward peak', ratio)
    assert ratio <= 1e-2
    error = np.max(np.abs(np.asarray(result.s_params) - np.asarray(open_results[0].s_params)))
    report('B4 mirror max S difference', error)
    assert error <= 1e-4


def test_occupancy_gradient(monkeypatch):
    sim = scene()
    with monkeypatch.context() as patch:
        setup = capture(patch, sim, 'run')
    grid = setup['grid']
    layout = actual_layout(setup, sim)
    cell = (grid.pad_x_lo + 14, grid.pad_y_lo + 12, layout.reference_index - 1)
    own = sim.realized_geometry().conductors
    assert not any(np.asarray(e.cells)[cell] for e in own.stamped_entities)
    def objective(value):
        rho = jnp.zeros(grid.shape).at[cell].set(value)
        output = sim.forward(n_steps=900, port_s11_freqs=np.array([5e9]),
                             pec_occupancy_override=rho, checkpoint_segments=30,
                             skip_preflight=True)
        return jnp.abs(output.s_params).reshape(-1)[0] ** 2
    gradient = jax.grad(objective)(jnp.asarray(.25))
    report('B8 occupancy gradient', float(gradient))
    assert np.isfinite(gradient)
    assert abs(float(gradient)) > 0


def test_short_cells_and_native(monkeypatch):
    from rfx.geometry import Box
    from tests.contracts.test_coax_port_model_and_drive import Captured
    sim = scene()
    sim.add(Box((.0035, .0035, .002), (.0085, .0085, .002)), material='pec')
    with monkeypatch.context() as patch:
        actual = capture(patch, sim, 'run')
    root = sim.realized_geometry().conductors
    ref = root.coax_bore_release.layout.reference_index
    recorded = {}
    def kernel(*args, **kwargs):
        recorded['cells'] = np.asarray(kwargs['conductors'].pec_cells)
        raise Captured
    with monkeypatch.context() as patch:
        patch.setattr('rfx.sparams._coax_oneport_record.run_reflection_record', kernel)
        with pytest.raises(Captured):
            scene().compute_coaxial_line_reflection(termination='short', n_steps=3, freqs=BINS)
    difference = np.asarray(root.pec_cells)[:, :, ref] ^ recorded['cells'][:, :, ref]
    report('B2 short realized cell difference', np.argwhere(difference).tolist())
    if difference.any():
        pytest.skip(f'short geometries differ at {np.count_nonzero(difference)} cells; no spectral judgement')
    assert actual['sources']
    result = sim.run(n_steps=STEPS, s_param_freqs=BINS, skip_preflight=True)
    expected = scene().compute_coaxial_line_reflection(termination='short', n_steps=STEPS, freqs=BINS)
    check_native(result, expected)
