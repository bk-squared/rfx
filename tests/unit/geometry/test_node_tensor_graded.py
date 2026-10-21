"""#1632 graded Eq. 10: independent metrics, energy form and public routes."""
from functools import wraps
from itertools import product
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
from rfx import Box, GaussianPulse, Simulation, Sphere
from rfx.geometry.node_tensor import compute_node_tensor_eps, node_metrics
from rfx.geometry.triplet_tensor import triplet_weights
from rfx.grid import Grid
from rfx.model.graded_grid import make_nonuniform_grid


SIGNS = tuple(product((0, 1), repeat=3))


def graded_grid(profiles, periodic=''):
    return make_nonuniform_grid(
        tuple(np.sum(p) for p in profiles[:2]), profiles[2], float(profiles[0][0]),
        cpml_layers=0, cpml_axes='', dx_profile=profiles[0], dy_profile=profiles[1],
        periodic_axes=periodic)


@pytest.mark.parametrize('periodic', ['', 'x', 'y', 'z', 'xyz'])
def test_w0_metrics_and_weights(periodic):
    profile = np.array([1., 1.5, 2.25, 1.5, 1.]) / 1024
    profiles = [np.full(5, profile[0]) if a in periodic else profile for a in 'xyz']
    grid = graded_grid(profiles, periodic)
    lengths, fill = node_metrics(grid)
    expected = []
    for axis, (name, profile) in enumerate(zip('xyz', profiles)):
        plus = profile if name in periodic else np.r_[profile, profile[-1]]
        minus = np.roll(plus, 1) if name in periodic else np.r_[plus[0], plus[:-1]]
        shape = tuple(len(plus) if a == axis else 1 for a in range(3))
        pair = (minus.reshape(shape), plus.reshape(shape))
        expected.append(pair)
        for actual, reference in zip(lengths[axis], pair):
            assert actual.dtype == np.float64
            np.testing.assert_array_equal(actual, reference)
        dual = (minus + plus) / 2
        # Solver metrics are stored in float32, independently of geometry.
        inverse = getattr(grid, f'inv_d{name}')
        np.testing.assert_array_equal(inverse, (1 / dual).astype(np.float32))
    duals = [(m + p) / 2 for m, p in expected]
    np.testing.assert_array_equal(fill, (duals[0] * duals[1] * duals[2]) ** (1 / 3))
    weights = triplet_weights(np, lengths)
    maximum = 0.
    for a in range(3):
        b, c = [axis for axis in range(3) if axis != a]
        for t, signs in enumerate(SIGNS):
            reference = (expected[b][1 - signs[b]] * expected[c][1 - signs[c]]
                         / (2 * sum(expected[b]) * sum(expected[c])))
            np.testing.assert_array_equal(weights[a][t], reference)
        for end in (0, 1):
            total = sum(weights[a][t] for t, signs in enumerate(SIGNS) if signs[a] == end)
            maximum = max(maximum, float(np.max(np.abs(2 * total - 1))))
    print(f'W0 periodic={periodic!r}: weight-sum error={maximum:.17g}, bar=1e-15')
    assert maximum <= 1e-15


# Main's uniform NodeTensor leaves on this scene (Mac, main 01216a729, whose leaves'
# SHA-256 the branch matched bit for bit there): (sum, max |value|) per leaf in
# tree order, and the three edge diagonals' sums. A bit hash does not hold
# across platforms (the eigensolve and inverse round differently on Linux).
MAIN_NODE_LEAVES = (
    (681.202942282, 1.0), (681.016578823, 1.0), (681.016576231, 1.0),
    (-0.099418484, 0.105664045), (0.036361065, 0.118183427), (-0.068163625, 0.124520838),
    (581.0, 1.0), (575.0, 1.0), (574.0, 1.0), (148.0, 1.0), (154.0, 1.0), (155.0, 1.0),
    (16330.410661399, 1.0), (0.038351786, 0.030762693), (0.039108522, 0.030395823),
    (-0.102327479, 0.031641316), (0.0, 0.0))
MAIN_DIAGONAL_SUMS = (884.308642, 883.174998, 881.296438)


def test_w1_uniform_builder_is_main():
    assert Path(rfx.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[3])
    grid = Grid(domain=(.008,) * 3, dx=.001, freq_max=10e9, cpml_axes='')
    eps = compute_node_tensor_eps(grid, [(Sphere((.0041, .0038, .0043), .0023), 4.)])
    assert eps.nodes.cross_reverse is None
    leaves = [np.asarray(a, dtype=np.float64) for a in jax.tree.leaves(eps.nodes)]
    assert len(leaves) == len(MAIN_NODE_LEAVES)
    for leaf, (total, peak) in zip(leaves, MAIN_NODE_LEAVES):
        assert abs(leaf.sum() - total) <= 1e-6 * max(abs(total), peak, 1e-30) + 1e-9
        assert abs(np.abs(leaf).max() - peak) <= 1e-6 * peak
    for diagonal, total in zip(eps.diagonal, MAIN_DIAGONAL_SUMS):
        assert abs(np.asarray(diagonal, dtype=np.float64).sum() - total) <= 1e-6 * total


def test_w2_constant_profile_coefficients():
    profile = np.full(8, .001)
    uniform = Grid(domain=(.008,) * 3, dx=.001, freq_max=10e9, cpml_axes='')
    graded = graded_grid([profile] * 3)
    shapes = [(Sphere((.0041, .0038, .0043), .0023), 4.)]
    reference, actual = [compute_node_tensor_eps(g, shapes) for g in (uniform, graded)]
    maximum = 0.
    for arrays in ((actual.diagonal, reference.diagonal),
                   (actual.nodes.cross, reference.nodes.cross),
                   (actual.nodes.cross_reverse, reference.nodes.cross)):
        for value, expected in zip(*arrays):
            value, expected = np.asarray(value), np.asarray(expected)
            error = np.abs(value.astype(np.float64) - expected)
            ulp = np.abs(np.spacing(expected))
            maximum = max(maximum, float(np.max(error / ulp)))
            np.testing.assert_array_max_ulp(value, expected, maxulp=2)
    print(f'W2 coefficient max ULP={maximum}, bar=2')


def sphere_scene(graded=False, *, varying=False):
    profile = .001 * np.array([1., 1., 1.5, 1.5, 1., 1., 1.])
    mesh = {'dz_profile': profile if varying else np.full(8, .001)} if graded else {}
    sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001,
                     boundary='pec', cpml_layers=0, **mesh)
    sim.add_material('sphere', eps_r=4.)
    sim.add(Sphere((.0041, .0038, .0043), .0023), material='sphere')
    sim.add_source((.0021, .0022, .0023), 'ez',
                   waveform=GaussianPulse(f0=10e9, bandwidth=.8), amplitude_kind='current')
    sim.add_probe((.0052, .0041, .0044), 'ez')
    return sim


def record_e_fields(monkeypatch, *, peak_only=False):
    """Replace only the test's probe output with the actual post-step E."""
    import rfx.stepping.graded as graded
    import rfx.stepping.uniform as uniform

    def observe(factory):
        @wraps(factory)
        def build(*args, **kwargs):
            step = factory(*args, **kwargs)
            @wraps(step)
            def measured(*args, **kwargs):
                carry, _, *rest = step(*args, **kwargs)
                state = carry['fdtd']
                fields = jnp.stack((state.ex, state.ey, state.ez))
                output = jnp.max(jnp.abs(fields))[None] if peak_only else fields.ravel()
                return (carry, output, *rest)
            return measured
        return build
    monkeypatch.setattr(graded, 'make_graded_step', observe(graded.make_graded_step))
    monkeypatch.setattr(uniform, 'make_uniform_step', observe(uniform.make_uniform_step))


def test_w2_constant_profile_400_steps(monkeypatch, tmp_path):
    record_e_fields(monkeypatch)
    runs = {}
    for smoothing in (True, 'full_tensor'):
        for graded in (False, True):
            result = sphere_scene(graded).run(
                n_steps=400, skip_preflight=True, subpixel_smoothing=smoothing)
            runs[smoothing, graded] = np.asarray(result.time_series)
    diagonal = np.max(np.abs(runs[True, True] - runs[True, False]), axis=1)
    difference = np.max(np.abs(runs['full_tensor', True] - runs['full_tensor', False]), axis=1)
    peak = np.maximum(np.max(np.abs(runs['full_tensor', True]), axis=1),
                      np.max(np.abs(runs['full_tensor', False]), axis=1))
    bar = np.maximum(2 * diagonal, 9 * np.spacing(peak))
    trace = tmp_path / 'w2-per-step.csv'
    np.savetxt(trace, np.column_stack((np.arange(1, 401), difference, diagonal, peak, bar)),
               delimiter=',', header='step,full_difference,diagonal_difference,peak,bar')
    print(f'W2 max difference={difference.max()}, max difference/bar='
          f'{np.max(difference / np.maximum(bar, np.finfo(np.float32).tiny))}, trace={trace}')
    assert np.all(np.isfinite(difference))
    assert np.all(difference <= bar), f'W2 failed steps={np.flatnonzero(difference > bar) + 1}; {trace}'


def weighted_operator():
    """W3 scene: the update operator assembled from the published arrays only."""
    profiles = [.001 * np.array(p) for p in (
        [1, 1, 1.5, 1.5, 1, 1], [1, 1.5, 2.25, 1.5, 1], [1, 1, 1.5, 1, 1])]
    with jax.enable_x64():
        grid = graded_grid(profiles)
        eps = compute_node_tensor_eps(
            grid, [(Sphere((.0031, .0034, .0027), .00165), 4.)], field_dtype=jnp.float64)
    # Assemble from the published diagonal/couplings, never off_diagonal,
    # coefficients, node_metrics or a solver material-volume helper.
    sizes = [np.r_[p, p[-1]] for p in profiles]
    dual = [(np.r_[p[0], p[:-1]] + p) / 2 for p in sizes]
    active = np.stack(eps.nodes.active)
    for a in range(3):
        sl = [slice(None)] * 4
        sl[0], sl[a + 1] = a, -1
        active[tuple(sl)] = False
    edges = list(zip(*np.nonzero(active)))
    index = {edge: i for i, edge in enumerate(edges)}
    operator = np.zeros((len(edges), len(edges)), dtype=np.float64)
    volumes = np.empty(len(edges), dtype=np.float64)
    for edge, row in index.items():
        a, *node = edge
        operator[row, row] = 1 / np.asarray(eps.diagonal[a])[tuple(node)]
        volumes[row] = sizes[a][node[a]]
        for b in range(3):
            if b != a:
                volumes[row] *= dual[b][node[b]]
    for node in np.ndindex(grid.shape):
        for pair, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            for sa, sb in product((0, 1), repeat=2):
                ea, eb = list(node), list(node)
                ea[a] -= sa
                eb[b] -= sb
                row, col = index.get((a, *ea)), index.get((b, *eb))
                if row is not None and col is not None:
                    operator[row, col] += np.asarray(eps.nodes.cross[pair])[(sa, sb, *node)]
                    operator[col, row] += np.asarray(eps.nodes.cross_reverse[pair])[(sa, sb, *node)]
    return grid, eps, edges, operator, volumes


def test_w3_weighted_operator_spd():
    _, _, edges, operator, volumes = weighted_operator()
    assert np.count_nonzero(operator - np.diag(np.diag(operator))) > 0
    energy = volumes[:, None] * operator
    relative = np.max(np.abs(energy - energy.T)) / np.max(np.abs(energy))
    print(f'W3 active edges={len(edges)}, symmetry relative={relative:.17g}, bar=1e-12')
    assert relative <= 1e-12
    chol = np.linalg.cholesky(energy)
    print(f'W3 Cholesky min pivot={np.min(np.diag(chol)):.17g}')
    assert np.all(np.diag(chol) > 0)


def test_w3_kernel_applies_the_assembled_couplings():
    """The step's off-diagonal increment is the W3 operator's off-diagonal part.

    On a graded mesh E_a<-b and E_b<-a carry different weights, so a kernel
    that reads the two coupling sets the wrong way round fails here.
    """
    from rfx.core.node_tensor import off_diagonal

    grid, eps, edges, operator, _ = weighted_operator()
    rng = np.random.default_rng(1632)
    values = rng.standard_normal(len(edges))
    curls = [np.zeros(grid.shape) for _ in range(3)]
    for (a, *node), value in zip(edges, values):
        curls[a][tuple(node)] = value
    with jax.enable_x64():
        result = off_diagonal(tuple(jnp.asarray(c) for c in curls), eps.nodes,
                              periodic=(False,) * 3)
        actual = np.array([np.asarray(result[a])[tuple(node)] for a, *node in edges])
    expected = (operator - np.diag(np.diag(operator))) @ values
    error = np.max(np.abs(actual - expected)) / np.max(np.abs(expected))
    print(f'W3 kernel vs assembled off-diagonal: relative={error:.3g}, bar=1e-12')
    assert error <= 1e-12


def test_w4_internal_pec_energy(monkeypatch, tmp_path):
    record_e_fields(monkeypatch, peak_only=True)
    profile = .001 * np.array([1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1, 1, 1])
    sim = Simulation(freq_max=20e9, domain=(.012,) * 3, dx=.001,
                     dz_profile=profile, boundary='pec', cpml_layers=0)
    sim.add_material('sphere', eps_r=12.)
    sim.add(Box((0, 0, .002), (.012, .012, .004)), material='pec')
    sim.add(Sphere((.0061, .0058, .0052), .0029), material='sphere')
    pulse = GaussianPulse(f0=10e9, bandwidth=.8)
    dt = sim._build_nonuniform_grid().dt
    assert float(pulse(400 * dt)) == 0.
    sim.add_source((.0031, .0042, .0083), 'ez', waveform=pulse, amplitude_kind='current')
    sim.add_probe((.0072, .0051, .0064), 'ex')
    started = time.monotonic()
    result = sim.run(n_steps=20400, skip_preflight=True, subpixel_smoothing='full_tensor')
    amplitude = np.asarray(result.time_series).ravel()[400:]
    elapsed = time.monotonic() - started
    trace = tmp_path / 'w4-per-step.csv'
    np.savetxt(trace, np.column_stack((np.arange(401, 20401), amplitude)),
               delimiter=',', header='step,global_peak_E')
    first, last = np.max(amplitude[:5000]), np.max(amplitude[-5000:])
    print(f'W4 seconds={elapsed:.3f}, first quarter={first}, last quarter={last}, '
          f'ratio={last / first}, bar=1.05, trace={trace}')
    assert np.all(np.isfinite(amplitude))
    assert first > 0
    assert last <= 1.05 * first, str(trace)


def test_w5_forward_run_and_kernel_operands(monkeypatch):
    import rfx.core.yee as yee
    from tests.contracts.run_forward_census import build_c_full_tensor

    calls = []
    original = yee.update_e_nu_aniso
    @wraps(original)
    def observe(state, materials, *args, **kwargs):
        nodes = materials.components.tensor.node_smoothing
        assert nodes is not None and nodes.cross_reverse is not None
        assert nodes.pec_keep is not None
        calls.append(True)
        return original(state, materials, *args, **kwargs)
    monkeypatch.setattr(yee, 'update_e_nu_aniso', observe)
    freqs = np.array([8e9, 10e9, 12e9])
    options = dict(n_steps=400, skip_preflight=True, subpixel_smoothing='full_tensor')
    run = build_c_full_tensor('graded').run(
        compute_s_params=True, s_param_freqs=freqs, **options)
    count = len(calls)
    assert count > 0
    forward = build_c_full_tensor('graded').forward(
        port_s11_freqs=freqs, checkpoint=False, **options)
    assert len(calls) > count
    for name in ('s_params', 'time_series'):
        expected, actual = np.asarray(getattr(run, name)), np.asarray(getattr(forward, name))
        assert expected.size and np.all(np.isfinite(expected))
        np.testing.assert_array_equal(actual, expected)
        assert actual.tobytes() == expected.tobytes()
    print(f'W5 run/forward S and probes: byte equal; kernel observations={len(calls)}')


def test_design_box_full_tensor_shared_refusal():
    messages = []
    for graded in (False, True):
        with pytest.raises(NotImplementedError, match='design-box.*aniso_eps') as info:
            sphere_scene(graded).forward(
                n_steps=1, skip_preflight=True, subpixel_smoothing='full_tensor',
                design_box=((.003,) * 3, (.005,) * 3),
                design_eps_override=jnp.ones((3, 3, 3)) * 3)
        messages.append(str(info.value))
    assert messages[0] == messages[1]
    print(f'A2 design box: {messages[0]}')


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_graded_tfsf_keeps_oblique_refusal(entry):
    from rfx import Cylinder
    messages = []
    for graded in (False, True):
        mesh = {'dz_profile': np.full(12, .001)} if graded else {}
        sim = Simulation(freq_max=20e9, domain=(.024, .012, .012), dx=.001,
                         cpml_layers=2, **mesh)
        sim.add_material('rod', eps_r=4.)
        sim.add(Cylinder((.0121, .0058, .006), .0029, .1, 'z'), material='rod')
        sim.add_tfsf_source(f0=10e9, bandwidth=.8, direction='+x', polarization='ez', angle_deg=30.)
        message = ("TFSF oblique incidence.*not yet supported on nonuniform" if graded else
                   "full_tensor's diagonal is not invariant")
        with pytest.raises(ValueError, match=message) as info:
            getattr(sim, entry)(n_steps=1, skip_preflight=True, subpixel_smoothing='full_tensor')
        messages.append(str(info.value))
    print(f'A2 TF/SF {entry}: uniform={messages[0]}; graded={messages[1]}')


@pytest.mark.parametrize('dimension', ['radius', 'centre', 'face'])
def test_graded_full_tensor_adds_no_shape_tangent(dimension):
    """Forward model only: a shape tangent through graded full_tensor stays refused."""
    def objective(value):
        sizes = {'radius': .0023, 'centre': .0041, 'face': .006, dimension: value}
        sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001, cpml_layers=4,
                         dz_profile=.001 * np.array([1, 1, 1.5, 1.5, 1, 1, 1]))
        sim.add_material('d', eps_r=4.)
        sim.add(Sphere((sizes['centre'], .0038, .0043), sizes['radius']), material='d')
        sim.add(Box((.001, .001, .001), (sizes['face'], .002, .002)), material='d')
        sim.add_source((.0021, .0022, .0023), 'ez', waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        sim.add_probe((.0052, .0041, .0044), 'ez')
        result = sim.forward(n_steps=20, skip_preflight=True, subpixel_smoothing='full_tensor')
        return jnp.sum(result.time_series ** 2)
    start = {'radius': .0023, 'centre': .0041, 'face': .006}[dimension]
    with pytest.raises(NotImplementedError, match='A gradient with respect to a declared shape dimension'):
        jax.grad(objective)(start)
