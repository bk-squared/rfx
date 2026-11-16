"""Full-tensor smoothing is continuous where a box edge meets a grid node.

The planar triplet rule gave each node one plane, the nearest face; at a box
edge on a node (or on the bisector of one) float rounding chose the face and
the model jumped. Near an axis-aligned Box each edge support now takes the
exact painted content of the Boxes; next to a non-Box shape the exact and the
planar triplets are blended over one cell.

Expected values are closed forms or an independent numpy integration of the
support (edge x dual face), never the builder's helpers.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, Sphere
from rfx.errors import RfxUnsupported
import rfx.geometry.node_tensor as NT

EPS = 10.0
H = .001
ONNODE = ((.003, .004, .004), (.0077, .008, .008))      # x-lo face on node 3, y-lo on node 4
BISECTOR = ((.0026, .0036, .0033), (.0077, .0082, .0083))  # x-lo and y-lo 0.4 cell inside node (3, 4)
R_UNIFORM = 1.25 * np.sqrt(1.5) * H                     # the reach on a 1 mm mesh


def _sim(items, n=12):
    sim = Simulation(freq_max=20e9, domain=(n * H,) * 3, dx=H, boundary="pec")
    for k, (shape, eps) in enumerate(items):
        sim.add_material(f"m{k}", eps_r=eps)
        sim.add(shape, material=f"m{k}")
    sim.add_source((.0031, .0042, .0043), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8),
                   amplitude_kind="current")
    sim.add_probe((.0072, .0051, .0064), "ez")
    return sim


class _Stop(Exception):
    pass


def _build(items, monkeypatch, n=12, planar=False):
    """The realized node tensor; ``planar`` builds it with the Box rule switched off."""
    got = {}
    real = NT.compute_node_tensor_eps

    def spy(grid, shapes, background_eps=1.0, **kw):
        got["eps"] = real(grid, shapes, background_eps, **kw)
        raise _Stop

    with monkeypatch.context() as m:
        m.setattr(NT, "compute_node_tensor_eps", spy)
        if planar:
            import rfx.geometry.box_content as BC
            m.setattr(BC, "box_scope", lambda *a, **k: None)
        with pytest.raises(_Stop):
            _sim(items, n).forward(n_steps=2, skip_preflight=True, subpixel_smoothing="full_tensor")
    return got["eps"]


def _diagonal(items, monkeypatch, **kw):
    return [np.asarray(a, dtype=np.float64) for a in _build(items, monkeypatch, **kw).diagonal]


def _ulps(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))


def _single_box_content(n, lo, hi, eps):
    """Independent exact inverse permittivity [a][sign] of one box on a uniform 1 mm mesh."""
    x = np.arange(n + 1) * H
    P = np.meshgrid(x, x, x, indexing="ij")

    def overlap(a0, a1, b0, b1):
        return np.maximum(0., np.minimum(a1, b1) - np.maximum(a0, b0))

    out = []
    for a in range(3):
        b, c = [k for k in range(3) if k != a]
        per = []
        for s in (0, 1):
            e0, e1 = (P[a], P[a] + H) if s == 0 else (P[a] - H, P[a])
            length = overlap(e0, e1, lo[a], hi[a]) / H
            area = (overlap(P[b] - H / 2, P[b] + H / 2, lo[b], hi[b])
                    * overlap(P[c] - H / 2, P[c] + H / 2, lo[c], hi[c]) / H ** 2)
            per.append(length / (area * eps + 1 - area) + (1 - length))
        out.append(per)
    return out


@pytest.mark.parametrize("d", [0.0, -1e-12, 1e-12])
def test_box_edge_cells_take_the_exact_content(d, monkeypatch):
    """An edge in a box face, half the dual face inside: (10 + 1) / 2; a quarter
    inside on the box's corner line: (10 + 3) / 4; a support that touches only
    the face: 1."""
    lo, hi = ONNODE
    diagonal = _diagonal([(Box((lo[0] + d, lo[1], lo[2]), hi), EPS)], monkeypatch)
    cells = [(0, (3, 4, 6)), (1, (3, 4, 6)), (1, (3, 3, 6)), (0, (2, 4, 6)), (0, (3, 4, 4))]
    expected = [(EPS + 1) / 2, (EPS + 1) / 2, 1.0, 1.0, (EPS + 3) / 4]
    np.testing.assert_allclose([diagonal[a][n] for a, n in cells], expected, rtol=1e-6)


def test_bisector_diagonal_does_not_jump(monkeypatch):
    lo, hi = BISECTOR
    below, above = (_diagonal([(Box((lo[0] + d, lo[1], lo[2]), hi), EPS)], monkeypatch)
                    for d in (-1e-12, 1e-12))
    for a in range(3):
        assert _ulps(below[a], above[a]).max() <= 1, a


@pytest.mark.parametrize("scene", [ONNODE, BISECTOR], ids=["on-node", "bisector"])
def test_every_cell_is_the_exact_content(scene, monkeypatch):
    """Every diagonal cell of a one-box scene is the assembled exact content,
    1 / ((v_a+(n) + v_a-(n + e_a)) / 2), also next to the box's edges where the
    face's extended plane gave content to a support no box reaches."""
    lo, hi = scene
    diagonal = _diagonal([(Box(lo, hi), EPS)], monkeypatch)
    value = _single_box_content(12, lo, hi, EPS)
    for a in range(3):
        nxt = np.take(value[a][1], np.minimum(np.arange(13) + 1, 12), axis=a)
        expected = 1 / ((value[a][0] + nxt) / 2)
        assert _ulps(diagonal[a], expected).max() <= 1, a


def _loss(lo, hi, d):
    result = _sim([(Box((lo[0] + d, lo[1], lo[2]), hi), EPS)]).forward(
        n_steps=20, skip_preflight=True, subpixel_smoothing="full_tensor")
    return float(np.sum(np.asarray(result.time_series, dtype=np.float64) ** 2))


@pytest.mark.parametrize("scene", [ONNODE, BISECTOR], ids=["on-node", "bisector"])
def test_face_moved_by_rounding_keeps_the_loss(scene):
    """At +-1e-12 m the relative change of the 20-step loss is at most 1e-6."""
    base = _loss(*scene, 0.0)
    for d in (-1e-12, 1e-12):
        assert abs(_loss(*scene, d) - base) / base <= 1e-6, d


def test_blend_next_to_a_sphere_adds_no_jump(monkeypatch):
    """The sphere's bounding box halfway between R and R + h from the node row
    y = 8 mm; a 1e-9 m move of its centre changes no diagonal cell by more than
    the planar build changes it, plus 1e-5 of the cell, and some node blends."""
    import rfx.geometry.box_content as BC
    seen = {}
    real = BC.box_scope

    def spy(*args, **kwargs):
        seen["scope"] = real(*args, **kwargs)
        return seen["scope"]

    monkeypatch.setattr(BC, "box_scope", spy)
    builds = {}
    for planar in (False, True):
        for dd in (-1e-9, 1e-9):
            cy = .008 + R_UNIFORM + H / 2 + .0015 + dd
            items = [(Box(*ONNODE), EPS), (Sphere((.008, cy, .006), .0015), EPS)]
            builds[planar, dd] = _diagonal(items, monkeypatch, n=16, planar=planar)
            if not planar:
                weight, region = seen["scope"].weight, seen["scope"].region
                assert int(((weight > 0) & (weight < 1) & region).sum()) > 0
    for a in range(3):
        head = np.abs(builds[False, 1e-9][a] - builds[False, -1e-9][a])
        planar = np.abs(builds[True, 1e-9][a] - builds[True, -1e-9][a])
        assert np.max((head - planar) / np.abs(builds[False, -1e-9][a])) <= 1e-5, a


@pytest.mark.parametrize("pair", [(3.38, 10.2), (4.0, 4.0)], ids=["3.38|10.2", "4|4"])
def test_abutting_and_overlapping_boxes_agree(pair, monkeypatch):
    """The upper box's lower face at the interface, or 1e-12 m below it: the
    later box wins the overlap, so the diagonal moves by at most 1e-6."""
    lower, upper = pair
    out = []
    for dd in (0.0, 1e-12):
        out.append(_diagonal([(Box((.0022, .0022, .002), (.0097, .0097, .0053)), lower),
                              (Box((.0022, .0022, .0053 - dd), (.0097, .0097, .009)), upper)],
                             monkeypatch))
    for a in range(3):
        assert np.max(np.abs(out[0][a] - out[1][a]) / out[0][a]) <= 1e-6, a


def test_a_later_box_paints_over_an_earlier_one(monkeypatch):
    """A (eps 4), then B (eps 10.2), then C (eps 4) inside B: as the assembled
    model paints them, every triplet of a node whose supports lie in C is 1/4."""
    eps = _build([(Box((.0022, .0022, .002), (.0097, .0097, .0053)), 4.0),
                  (Box((.0022, .0022, .0053), (.0097, .0097, .0105)), 10.2),
                  (Box((.0032, .0032, .0062), (.0088, .0088, .0098)), 4.0)], monkeypatch)
    triplets = np.asarray(eps.nodes.triplets)
    inside = triplets[:, 5:8, 5:8, 8]       # nodes 5..7 mm in x and y, 8 mm in z
    np.testing.assert_array_equal(inside, np.broadcast_to(np.eye(3, dtype=np.float32) / 4, inside.shape))


@pytest.mark.parametrize("wrapped", ["box", "sphere"], ids=["box-content", "sphere-blocker"])
def test_periodic_images_reach_the_lower_x_face(wrapped, monkeypatch):
    """Wrapped content keeps declared order; a wrapped sphere keeps the planar rule."""
    from rfx.grid import Grid

    grid = Grid(domain=(.012, .008, .008), dx=H, freq_max=10e9,
                cpml_axes="", periodic_axes="x")
    first = (Box((.0105, .0023, .0023), (.0126, .0057, .0057)) if wrapped == "box"
             else Sphere((.0118, .004, .004), .001))
    items = [(first, 9.0), (Box((.0003, .0023, .0023), (.0034, .0057, .0057)), 4.0)]
    triplets = np.asarray(NT.compute_node_tensor_eps(grid, items).nodes.triplets)
    if wrapped == "box":
        # As in _single_box_content, axis 0 / sign 0 is x+; triplets 0..3 have x+.
        # B covers 0..0.6 mm, then A paints 0.3..1 mm of the x+ support at node 0.
        np.testing.assert_allclose(triplets[:4, 0, 4, 4, 0, 0], .3 / 9 + .7 / 4, rtol=1e-6)
    else:
        import rfx.geometry.box_content as BC
        with monkeypatch.context() as m:
            m.setattr(BC, "box_scope", lambda *a, **k: None)
            planar = np.asarray(NT.compute_node_tensor_eps(grid, items).nodes.triplets)
        np.testing.assert_array_equal(triplets[:, 0, 4, 4].view(np.uint32),
                                      planar[:, 0, 4, 4].view(np.uint32))


def test_traced_coordinate_beside_a_box_is_refused():
    def loss(radius):
        sim = _sim([(Box(*ONNODE), 4.0), (Sphere((.0095, .0095, .0095), radius), 2.0)])
        result = sim.forward(n_steps=20, skip_preflight=True, subpixel_smoothing="full_tensor")
        return jnp.sum(jnp.asarray(result.time_series) ** 2)

    with pytest.raises(RfxUnsupported, match="traced shape coordinate .*'m1'.* 9.7e-4"):
        jax.jit(loss)(.0015)


def test_more_than_eight_boxes_on_one_support_are_refused():
    from rfx.geometry.node_tensor import compute_node_tensor_eps
    from rfx.grid import Grid
    grid = Grid(domain=(.004, .004, .004), dx=H, freq_max=10e9, cpml_axes='')
    boxes = [(Box((.0021 + .0001 * k, .0019, .0019), (.00215 + .0001 * k, .0021, .0021)), 4.0)
             for k in range(9)]
    with pytest.raises(RfxUnsupported, match="9 boxes .* at most 8"):
        compute_node_tensor_eps(grid, boxes)
