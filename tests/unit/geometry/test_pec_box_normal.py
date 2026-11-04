"""Box PEC normal operands are series capacitors, parallel across a dual cell."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation, Sphere
from rfx.geometry import pec_normal, smoothing

H = 1 / 1024
ER = 4.4


def scene(kind, fraction, boundary='pec'):
    sim = Simulation(freq_max=10e9, domain=(8*H, 8*H, 8*H), dx=H,
                     boundary=boundary, cpml_layers=4 if boundary == 'cpml' else 0)
    sim.add_material('substrate', eps_r=ER)
    face = (5 - fraction)*H if kind == 'lower' else 4*H
    dielectrics = []
    if kind != 'air':
        slab = Box((-10*H, -10*H, -10*H), (20*H, 20*H, face))
        sim.add(slab, material='substrate')
        dielectrics = [(slab, ER)]
    metal = (Box((H, H, face), (7*H, 7*H, 7*H)) if kind == 'lower'
             else Box((H, H, 4*H), (7*H, (4+fraction)*H, 7*H)))
    sim.add(metal, material='pec')
    return sim, dielectrics, metal


def operands(sim, dielectrics, metal):
    return smoothing.compute_inv_eps_tensor_diag(sim.freeze_mesh(), dielectrics, [metal])


@pytest.mark.parametrize('fraction', [.25, .5, .625])
def test_lower_face_hand_value(fraction):
    """A metal/slab series capacitor stores field only in its er=4.4 length."""
    sim, dielectrics, metal = scene('lower', fraction)
    result = operands(sim, dielectrics, metal)
    dielectric_length = metal.corner_lo[2] - 4*H
    expected = dielectric_length / H / ER
    np.testing.assert_allclose(result[2][4, 4, 4], expected, rtol=1e-6, atol=0)
    assert all(a.dtype == jnp.float32 for a in result)


@pytest.mark.parametrize('fraction', [.125, .25, .5])
def test_side_face_hand_value(fraction):
    """Normal Ey survives on a metal face: its air series length is nonzero."""
    sim, dielectrics, metal = scene('air', fraction)
    air_length = 5*H - metal.corner_hi[1]
    np.testing.assert_allclose(operands(sim, dielectrics, metal)[1][4, 4, 5],
                               air_length / H, rtol=1e-6, atol=0)


@pytest.mark.parametrize('fraction', [.125, .25, .5])
def test_corner_row_hand_value(fraction):
    """Half substrate and half metal/air series capacitors are in parallel."""
    sim, dielectrics, metal = scene('corner', fraction)
    air_fraction = (5*H - metal.corner_hi[1]) / H
    expected = 1 / (.5*ER + .5/air_fraction)
    np.testing.assert_allclose(operands(sim, dielectrics, metal)[1][4, 4, 4],
                               expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize('kind,fraction', [
    ('lower', .25), ('lower', .5), ('lower', .625),
    ('air', .125), ('air', .25), ('air', .5),
    ('corner', .125), ('corner', .25), ('corner', .5)])
def test_other_samples_keep_legacy_limit(kind, fraction):
    """PEC tangential E stays frozen; pure cells keep their bulk constitutive law."""
    sim, dielectrics, metal = scene(kind, fraction)
    grid = sim.freeze_mesh()
    actual = operands(sim, dielectrics, metal)
    nodes = np.indices(grid.shape, dtype=float) * H
    centre = (np.array(metal.corner_lo) + np.array(metal.corner_hi)) / 2
    half = (np.array(metal.corner_hi) - np.array(metal.corner_lo)) / 2
    for axis in range(3):
        points = nodes.copy()
        points[axis] += H / 2
        face_distances = np.abs(points - centre[:, None, None, None]) - half[:, None, None, None]
        # Orthogonal outside distances combine by Pythagoras; inside, the
        # largest signed distance is the closest face. argmax ties x,y,z.
        distance = (np.sqrt(np.sum(np.maximum(face_distances, 0)**2, axis=0))
                    + np.minimum(np.max(face_distances, axis=0), 0))
        selected = np.argmax(face_distances, axis=0) == axis
        fill = np.clip(.5 - distance/H, 0, 1)
        eps = np.ones(grid.shape)
        if dielectrics:
            slab_fraction = np.clip(.5 - (points[2] - dielectrics[0][0].corner_hi[2])/H, 0, 1)
            eps = (1/(slab_fraction/ER + (1-slab_fraction)) if axis == 2
                   else slab_fraction*ER + (1-slab_fraction))
        # Kottke's perpendicular inverse is the series mean; parallel
        # inverse is zero for any metal. This does not call rfx helpers.
        expected = np.where(selected, (1-fill)/eps, np.where(fill > 0, 0., 1/eps))
        expected = np.where(distance <= 0, 0., expected)
        expected = np.minimum((1/eps).astype(np.float32), expected).astype(np.float32)
        unchanged = ~(selected & (fill > 0) & (fill < 1))
        np.testing.assert_array_equal(np.asarray(actual[axis])[unchanged], expected[unchanged])


@pytest.mark.parametrize('mode', [False, True])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_plain_modes_never_build_pec_tensor(monkeypatch, mode, entry):
    """Staircase and dielectric-only updates cannot acquire a new PEC capacitor."""
    calls = []
    def spy(*args, **kwargs):
        calls.append(True)
        pytest.fail('False/True reached the PEC tensor builder')
    monkeypatch.setattr(smoothing, 'compute_inv_eps_tensor_diag', spy)
    sim, _, _ = scene('lower', .25)
    kwargs = {'checkpoint': False} if entry == 'forward' else {}
    getattr(sim, entry)(n_steps=1, skip_preflight=True, subpixel_smoothing=mode, **kwargs)
    assert not calls


@pytest.mark.parametrize('kind', ['cylinder', 'sphere'])
def test_non_box_samples_do_not_enter_r1(monkeypatch, kind):
    """A cylindrical cap and a spherical supporting dielectric retain the old law."""
    sim = Simulation(freq_max=10e9, domain=(8*H,)*3, dx=H, boundary='pec', cpml_layers=0)
    if kind == 'cylinder':
        metal = Cylinder((4*H, 4*H, 4*H), 2*H, 3*H)
        dielectrics = []
    else:
        metal = Box((3.25*H,)*3, (4.75*H,)*3)
        dielectric = Sphere((4*H,)*3, 3*H)
        sim.add_material('sphere', eps_r=ER)
        sim.add(dielectric, material='sphere')
        dielectrics = [(dielectric, ER)]
    sim.add(metal, material='pec')
    calls = []
    def spy(*args, **kwargs):
        calls.append(True)
        pytest.fail('a non-Box sample entered the exact Box integral')
    monkeypatch.setattr(pec_normal, 'exact_box_inverse', spy)
    operands(sim, dielectrics, metal)
    assert not calls


def test_corner_integral_is_not_clipped_by_smoothed_dielectric():
    """A dielectric corner's exact parallel area differs from its planar SDF model."""
    sim = Simulation(freq_max=10e9, domain=(8*H,)*3, dx=H, boundary='pec', cpml_layers=0)
    slab = Box((0.,)*3, (4.125*H, 4.125*H, 8*H))
    metal = Box((H, H, 4.875*H), (7*H,)*3)
    sim.add_material('corner', eps_r=ER)
    sim.add(slab, material='corner')
    sim.add(metal, material='pec')
    # Each transverse interval is .625 h dielectric. The metal occupies
    # .125 h along z across the entire area, leaving .875 h in series.
    expected = .875 / (.625**2*ER + (1-.625**2))
    np.testing.assert_allclose(operands(sim, [(slab, ER)], metal)[2][4, 4, 4],
                               expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize('traced', [False, True], ids=['host', 'jnp'])
def test_backend_and_overlap_precedence(traced):
    """Overlapping declared slabs share one material union; later groups own it."""
    sim, _, metal = scene('lower', .25)
    grid = sim.freeze_mesh()
    def evaluate(face):
        slab = Box((-10*H,)*3, (20*H, 20*H, face))
        pairs = [(slab, ER), (slab, 7.), (slab, ER)]
        pec = Box((H, H, face), (7*H,)*3)
        return smoothing.compute_inv_eps_tensor_diag(grid, pairs, [pec])[2][4, 4, 4]
    face = metal.corner_lo[2]
    actual = jax.jit(evaluate)(jnp.float32(face)) if traced else evaluate(face)
    np.testing.assert_allclose(actual, .75/7., rtol=1e-6, atol=0)


@pytest.mark.parametrize('gap', [5e-10, 2e-9])
def test_normal_freeze_uses_existing_threshold(gap):
    """A nearly filled series capacitor is frozen only by the existing 1e-9 fence."""
    from rfx.boundaries.pec import PEC_INV_THRESHOLD, kottke_fenced_edge_masks
    sim = Simulation(freq_max=10e9, domain=(8*H,)*3, dx=H, boundary='pec', cpml_layers=0)
    metal = Box((H, H, (4+gap)*H), (7*H,)*3)
    sim.add(metal, material='pec')
    inverse = operands(sim, [], metal)
    expected = (metal.corner_lo[2] - 4*H) / H
    np.testing.assert_allclose(inverse[2][4, 4, 4], expected, rtol=1e-6, atol=0)
    frozen = kottke_fenced_edge_masks(tuple(jnp.ones_like(v, dtype=bool) for v in inverse), inverse)
    assert bool(frozen[2][4, 4, 4]) == (expected < PEC_INV_THRESHOLD)


def test_non_box_elsewhere_does_not_disable_normal_box():
    """A remote sphere cannot change the series capacitance of a Box/air face."""
    sim, _, metal = scene('air', .5)
    sphere = Sphere((6.5*H, 6.5*H, H), .25*H)
    sim.add_material('remote', eps_r=ER)
    sim.add(sphere, material='remote')
    np.testing.assert_allclose(operands(sim, [(sphere, ER)], metal)[1][4, 4, 5],
                               .5, rtol=1e-6, atol=0)


@pytest.mark.parametrize('kind', ['sphere', 'cylinder', 'unbounded'])
@pytest.mark.parametrize('fraction', [.25, .5])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'jnp'])
def test_non_box_pec_contact_keeps_independent_legacy_limit(kind, fraction, traced):
    """A second PEC cap in the dual cell excludes the Box's normal sample."""
    sim, dielectrics, metal = scene('lower', fraction)
    grid = sim.freeze_mesh()

    class MaskOnly:
        def mask(self, grid):
            # Even a remote occupancy-only PEC excludes every R1 sample.
            return Box((H,)*3, (2*H,)*3).mask(grid)

    cap = {'sphere': Sphere((4*H, 4*H, 3.8*H), .4*H),
           'cylinder': Cylinder((4*H, 4*H, 3.5*H), .3*H, 1.2*H),
           'unbounded': MaskOnly()}[kind]
    # Independent legacy construction at q=(4,4,4.5)h: the slab is a
    # series dielectric/air cell; both PEC normals there are along z.
    # The caps end at 4.2h / 4.1h, inside the closed dual cell [4h,5h].
    qz = 4.5*H
    face = metal.corner_lo[2]
    slab_fraction = np.clip(.5 - (qz-face)/H, 0, 1)
    eps = 1 / (slab_fraction/ER + 1-slab_fraction)
    box_distance = face-qz
    box_fill = np.clip(.5-box_distance/H, 0, 1)
    box_inverse = 0. if box_distance <= 0 else (1-box_fill)/eps
    cap_top = {'sphere': 4.2*H, 'cylinder': 4.1*H, 'unbounded': 2*H}[kind]
    cap_fill = np.clip(.5-(qz-cap_top)/H, 0, 1)
    expected = np.float32(min(np.float32(1/eps), box_inverse, (1-cap_fill)/eps))

    def evaluate(face):
        slab = Box(dielectrics[0][0].corner_lo, (20*H, 20*H, face))
        pec = Box((H, H, face), metal.corner_hi)
        return smoothing.compute_inv_eps_tensor_diag(grid, [(slab, ER)], [pec, cap])[2][4, 4, 4]

    actual = jax.jit(evaluate)(jnp.float32(face)) if traced else evaluate(face)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('arrangement', ['side', 'sandwich', 'overlap'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'jnp'])
def test_multiple_pec_boxes_use_one_union_integral(arrangement, reverse, traced):
    """All metal has zero inverse epsilon, independent of PEC list order."""
    sim, _, _ = scene('lower', .25)
    grid = sim.freeze_mesh()
    other = {
        'side': Box((H,)*3, (4.3*H, 7*H, 4.4*H)),
        'sandwich': Box((H,)*3, (7*H, 7*H, 4.2*H)),
        'overlap': Box((H, H, 4.9*H), (7*H,)*3),
    }[arrangement]
    # Ez's cell spans x,y in [3.5,4.5]h and z in [4,5]h.
    # Top metal removes .2h. Side metal removes another .4h over .8
    # of the area: .8/.4 + .2/.8 = 2.25 parallel capacitance.
    # A lower slab leaves a .6h gap; overlapping top slabs remove .2h once.
    expected = {'side': 1/(.8/.4 + .2/.8), 'sandwich': .6, 'overlap': .8}[arrangement]

    def evaluate(face):
        upper = Box((H, H, face), (7*H,)*3)
        pecs = [other, upper] if reverse else [upper, other]
        return smoothing.compute_inv_eps_tensor_diag(grid, [], pecs)[2][4, 4, 4]

    face = 4.8*H
    actual = jax.jit(evaluate)(jnp.float32(face)) if traced else evaluate(face)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def test_chunked_traced_values_and_face_gradient():
    """A partial final block preserves every operand and the series derivative."""
    grid = Simulation(freq_max=10e9, domain=(9*H,)*3, dx=H,
                      boundary='pec', cpml_layers=0).freeze_mesh()

    def evaluate(face):
        slab = Box((-10*H,)*3, (20*H, 20*H, face))
        pec = Box((H, H, face), (8*H,)*3)
        return smoothing.compute_inv_eps_tensor_diag(grid, [(slab, ER)], [pec])

    assert np.prod(grid.shape) % 256 != 0
    face = 4.6*H
    host = evaluate(face)
    traced = jax.jit(evaluate)(jnp.float32(face))
    for actual, expected in zip(traced, host):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    gradient = jax.jit(jax.grad(lambda f: evaluate(f)[2][4, 4, 4]))(jnp.float32(face))
    np.testing.assert_allclose(gradient, 1/(H*ER), rtol=1e-6, atol=0)
