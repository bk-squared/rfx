"""Domain walls restrict node activation without becoming material conductors."""
import sys

import numpy as np

from rfx import Box, GaussianPulse, Simulation


def observe_nodes(monkeypatch):
    import rfx.model.node_pec as node_pec

    observed = []
    original = node_pec.restrict_node_pec

    def observe(*args, **kwargs):
        materials = original(*args, **kwargs)
        observed.append(materials.components.tensor.node_smoothing)
        return materials

    monkeypatch.setattr(node_pec, 'restrict_node_pec', observe)
    return observed


def slab_scene(*, plane_wave):
    sim = Simulation(freq_max=20e9, domain=(.024, .012, .012), dx=.001,
                     dz_profile=np.r_[np.full(6, .0009), np.full(6, .0011)],
                     boundary='cpml' if plane_wave else 'pec',
                     cpml_layers=2 if plane_wave else 0)
    sim.add_material('slab', eps_r=4.)
    sim.add(Box((.0103, 0., 0.), (.0137, .012, .012)), material='slab')
    if plane_wave:
        sim.add_tfsf_source(f0=10e9, bandwidth=.8, polarization='ez', direction='+x', margin=3)
    else:
        sim.add_source((.0071, .0042, .0043), 'ez',
                       waveform=GaussianPulse(f0=10e9, bandwidth=.8), amplitude_kind='current')
    sim.add_probe((.0161, .0052, .0063), 'ez')
    return sim


def test_graded_plane_wave_admits_full_tensor(monkeypatch):
    observed = observe_nodes(monkeypatch)
    # Include public preflight, as well as the step setup's admission checks.
    result = slab_scene(plane_wave=True).run(n_steps=40, subpixel_smoothing='full_tensor')
    assert observed
    expected = np.ones(result.grid.shape, dtype=bool)
    for nodes in observed:
        for keep in nodes.pec_keep:
            np.testing.assert_array_equal(keep, expected)
    assert result.time_series.shape == (40, 1)
    assert np.all(np.isfinite(result.time_series))


def test_graded_wall_edges_stay_inactive_and_zero(monkeypatch):
    observed = observe_nodes(monkeypatch)
    result = slab_scene(plane_wave=False).run(
        n_steps=40, skip_preflight=True, subpixel_smoothing='full_tensor')
    assert observed
    shape = result.grid.shape
    wall_edges = np.zeros((3, *shape), dtype=bool)
    # Independent tangential-edge ownership: component != wall normal.
    for normal in range(3):
        for plane in (0, shape[normal] - 1):
            for component in range(3):
                if component != normal:
                    cut = [component, slice(None), slice(None), slice(None)]
                    cut[normal + 1] = plane
                    wall_edges[tuple(cut)] = True
    fields = np.stack([np.asarray(getattr(result.state, name)) for name in ('ex', 'ey', 'ez')])
    wall_e = fields[wall_edges]
    print(f'wall E max after 40 steps: {np.max(np.abs(wall_e))}', file=sys.stderr)
    np.testing.assert_array_equal(wall_e, np.zeros(wall_e.shape))
    for nodes in observed:
        active = np.stack(nodes.active)[wall_edges]
        np.testing.assert_array_equal(active, np.zeros(active.shape, dtype=bool))


def test_uniform_conductor_keep_is_exactly_box_edges(monkeypatch):
    observed = observe_nodes(monkeypatch)
    sim = Simulation(freq_max=20e9, domain=(.008, .007, .006), dx=.001,
                     boundary='pec', cpml_layers=0)
    sim.add(Box((.002, .002, .002), (.004, .004, .004)), material='pec')
    sim.add_material('slab', eps_r=4.)
    sim.add(Box((.0051, 0., 0.), (.0063, .007, .006)), material='slab')
    sim.add_source((.0011, .0032, .0033), 'ez',
                   waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    sim.add_probe((.0052, .0041, .0044), 'ez')
    result = sim.run(n_steps=1, skip_preflight=True, subpixel_smoothing='full_tensor')
    assert observed
    expected = np.ones((3, *result.grid.shape), dtype=bool)
    # The closed box owns cells 2:4. Along an E edge's own axis use
    # cell centres (2:4); transverse node coordinates include both faces (2:5).
    for component in range(3):
        cut = [component, slice(2, 5), slice(2, 5), slice(2, 5)]
        cut[component + 1] = slice(2, 4)
        expected[tuple(cut)] = False
    for nodes in observed:
        np.testing.assert_array_equal(np.stack(nodes.pec_keep), expected)


def traced_plane_scene(eps, *, on_plane=False):
    from rfx import Cylinder
    from rfx.boundaries.spec import BoundarySpec

    sim = Simulation(freq_max=20e9, domain=(.024, .012, .012), dx=.001,
                     boundary=BoundarySpec(x='cpml', y='periodic', z='periodic'), cpml_layers=2)
    sim.add_material('slab', eps_r=eps)
    shape = (Cylinder((.0031, .006, .0061), .0029, .1, 'y') if on_plane else
             Box((.0103, 0., 0.), (.0137, .012, .012)))
    sim.add(shape, material='slab')
    sim.add_tfsf_source(f0=10e9, bandwidth=.8, polarization='ez', direction='+x', margin=3)
    sim.add_probe((.0161, .0052, .0063), 'ez')
    return sim


def test_r1_traced_nonzero_plane_coupling_raises():
    """A traced eps on the injection plane reaches the plane check and is refused.

    Until #2436 a traced eps stopped earlier, in the smoother (an unhashable
    tracer in _kottke_smooth). The smoother now carries the trace, so the
    curved face on the plane is judged by the TF/SF plane check at run time,
    as a concrete eps is: the same named refusal, never a raw error. Smoothing
    True: full_tensor refuses a traced eps at admission (#2441).
    """
    import jax
    import jax.numpy as jnp
    import pytest
    from rfx.errors import RfxRefusal

    def objective(eps):
        result = traced_plane_scene(eps, on_plane=True).forward(
            n_steps=2, checkpoint=False, skip_preflight=True, subpixel_smoothing=True)
        return jnp.sum(jnp.square(result.time_series))

    with pytest.raises(RfxRefusal, match='requires vacuum'):
        objective(4.)
    with pytest.raises(RfxRefusal, match='requires vacuum'):
        jax.grad(objective)(4.)
    # JIT wraps a callback's RfxRefusal in the backend runtime exception.
    with pytest.raises(jax.errors.JaxRuntimeError, match='requires vacuum'):
        jax.block_until_ready(jax.jit(objective)(jnp.float32(4.)))


def test_masked_couplings_match_numpy_output_edges():
    from dataclasses import replace
    import jax.numpy as jnp
    from rfx.geometry.node_tensor import NodeTensor
    from rfx.model.operands import node_tensor_operands

    shape = (4, 5, 6)
    coords = np.indices(shape)
    active = tuple((coords[a] % 2) == 0 for a in range(3))
    raw = tuple(np.arange(4 * np.prod(shape), dtype=np.float32).reshape(2, 2, *shape) + p
                for p in (1, 1001, 2001))
    reverse = tuple(-v - 7 for v in raw)
    nodes = NodeTensor(xi=(), bulk=(), active=tuple(map(jnp.asarray, active)),
                       triplets=None, cross=tuple(map(jnp.asarray, raw)), guard_count=0,
                       cross_reverse=tuple(map(jnp.asarray, reverse)))
    for periodic in ((False,) * 3, (True,) * 3):
        for backward in (reverse, None):
            value = node_tensor_operands(replace(nodes, cross_reverse=backward), periodic=periodic)
            assert 'active' not in value
            for pair, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                for direction, output, coupling in (('forward', a, raw),
                                                     ('reverse', b, raw if backward is None else backward)):
                    for sa in (0, 1):
                        for sb in (0, 1):
                            expected = np.zeros(shape, dtype=np.float32)
                            # Enumerate output edges, not the helper's shift implementation.
                            for edge in np.ndindex(shape):
                                source = list(edge)
                                source[output] += sa if direction == 'forward' else sb
                                if source[output] == shape[output]:
                                    if not periodic[output]:
                                        continue
                                    source[output] = 0
                                if active[output][edge]:
                                    expected[edge] = coupling[pair][sa, sb][tuple(source)]
                            np.testing.assert_array_equal(
                                value['masked_cross'][direction][pair][sa][sb], expected)


def test_traced_masked_coupling_warns_on_uniformity():
    import jax
    import jax.numpy as jnp
    import pytest
    from rfx.boundaries.tfsf import check_arrays, UnjudgedTFSFInvarianceWarning
    from rfx.geometry.node_tensor import NodeTensor
    from rfx.sources.tfsf import tfsf_from_declaration

    sim = slab_scene(plane_wave=True)
    grid = sim._build_nonuniform_grid()
    cfg = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=1)[0]
    shape = grid.shape
    raw = tuple(jnp.ones((2, 2, *shape)) for _ in range(3))

    def checked(active):
        nodes = NodeTensor(xi=(), bulk=(), active=(active,) * 3, triplets=None,
                           cross=raw, guard_count=0)
        check_arrays(cfg, grid, {'nodes': nodes}, nonuniform=True)
        return jnp.sum(active)

    with pytest.warns(UnjudgedTFSFInvarianceWarning, match='masked_cross'):
        jax.block_until_ready(jax.jit(checked)(jnp.ones(shape, dtype=bool)))


def test_r2_curved_face_reaching_y_absorber_refuses():
    import pytest
    from rfx import Cylinder
    from rfx.errors import RfxRefusal
    from rfx.geometry.node_tensor import compute_node_tensor_eps

    sim = Simulation(freq_max=20e9, domain=(.024, .012, .012), dx=.001,
                     dz_profile=np.r_[np.full(6, .0009), np.full(6, .0011)],
                     boundary='cpml', cpml_layers=2)
    face = Cylinder((.0121, .006, .0061), .0029, .1, 'y')
    sim.add_material('rod', eps_r=4.)
    sim.add(face, material='rod')
    sim.add_tfsf_source(f0=10e9, bandwidth=.8, polarization='ez', direction='+x', margin=3)
    sim.add_probe((.0181, .0052, .0063), 'ez')
    grid = sim._build_nonuniform_grid()
    nodes = compute_node_tensor_eps(grid, [(face, 4.)]).nodes
    for raw in (*nodes.cross, *nodes.cross_reverse):
        raw = np.asarray(raw)
        np.testing.assert_array_equal(raw, np.broadcast_to(raw[:, :, :, :1, :], raw.shape))
    # The plus/plus x-z coefficient reaches the x output edge without a shift.
    # NumPy constructs the applied coefficient independently of admission.
    masked = np.where(np.asarray(nodes.active[0]), np.asarray(nodes.cross[1][0, 0]), 0)
    assert not np.array_equal(masked, np.broadcast_to(masked[:, :1, :], masked.shape))
    with pytest.raises(RfxRefusal, match=(
            r'TF/SF z_lo, z_hi: update\.materials\.eps_r prevents an invariant periodic replacement')):
        sim.run(n_steps=2, subpixel_smoothing='full_tensor')


def test_r2_uniform_raw_coupling_with_absorber_mask_refuses():
    """The graded TF/SF refuses eps that is non-uniform along y or z,
    a structure uniform along both has only x-normal faces, and such faces
    give zero couplings. Thus no physical scene reaches the coupling check
    alone; Amendment 2 calls for a hand-built node tensor contract.
    """
    import jax.numpy as jnp
    import pytest
    from rfx.boundaries.tfsf import check_arrays, invariant
    from rfx.core.yee import init_materials
    from rfx.errors import RfxRefusal
    from rfx.geometry.node_tensor import NodeTensor
    from rfx.model.materials import with_components
    from rfx.model.tensor_operands import TensorOperands
    from rfx.sources.tfsf import tfsf_from_declaration

    sim = Simulation(freq_max=20e9, domain=(.008, .007, .006), dx=.001,
                     dz_profile=np.r_[np.full(3, .0009), np.full(3, .0011)],
                     boundary='cpml', cpml_layers=2)
    sim.add_tfsf_source(f0=10e9, bandwidth=.8, polarization='ez', direction='+x', margin=3)
    grid = sim._build_nonuniform_grid()
    cfg = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=1)[0]
    shape = grid.shape
    materials = init_materials(shape)
    pad = np.zeros(shape, dtype=bool)
    for axis, name in enumerate('xyz'):
        for side in ('lo', 'hi'):
            count = getattr(grid, f'pad_{name}_{side}')
            assert count > 0
            cut = [slice(None)] * 3
            cut[axis] = slice(0, count) if side == 'lo' else slice(-count, None)
            pad[tuple(cut)] = True
    active = ~pad
    assert np.any(active) and np.any(pad)
    cross = tuple(np.full((2, 2, *shape), v) for v in (.125, .25, .375))
    reverse = tuple(-2 * value for value in cross)
    for value in (*cross, *reverse):
        assert np.all(value != 0)
        for axis in (3, 4):
            np.testing.assert_array_equal(
                value, np.broadcast_to(np.take(value, [0], axis=axis), value.shape))

    # Match compute_node_tensor_eps: floating arrays use field_dtype;
    # boolean arrays and the guard counter keep their own dtypes.
    def cast(values):
        return tuple(jnp.asarray(value, dtype=materials.eps_r.dtype) for value in values)

    nodes = NodeTensor(
        xi=cast(tuple(np.ones(shape) for _ in range(6))),
        bulk=tuple(jnp.asarray(np.zeros(shape, dtype=bool)) for _ in range(3)),
        active=tuple(jnp.asarray(active) for _ in range(3)),
        triplets=jnp.asarray(np.broadcast_to(np.eye(3), (8, *shape, 3, 3)),
                             dtype=materials.eps_r.dtype),
        cross=cast(cross), cross_reverse=cast(reverse), guard_count=jnp.asarray(0),
        pec_keep=tuple(jnp.asarray(np.ones(shape, dtype=bool)) for _ in range(3)))
    materials = with_components(
        materials, grid, periodic=(False,) * 3,
        tensor=TensorOperands.of(eps=cast(tuple(np.ones(shape) for _ in range(3))),
                                 node_smoothing=nodes))
    values = {'materials': materials}
    # Raw leaves and eps are uniform; the applied, masked coefficients are not.
    with pytest.raises(RfxRefusal, match=r'update\.materials\.components\.tensor\.node_smoothing\.masked_cross'):
        check_arrays(cfg, grid, values, nonuniform=True)
    for axis in (1, 2):
        okay, feature = invariant(values, axis, shape)
        assert okay is False and 'masked_cross' in feature, (okay, feature)
