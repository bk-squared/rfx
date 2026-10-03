"""Material images fill a periodic cell; directed ports retain their extent.

These compare public declarations at an asymmetric seam, on a 13.7 mm
period with 17 cells. Material samples and owned PEC edges are the judges;
there is no absorber-dependent field threshold or fitted reference.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, PolylineWire, Simulation, Sphere
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.boundaries.spec import Boundary, BoundarySpec

L = .0137
DX = L / 17


def model():
    return Simulation(20e9, (L, 7.3 * DX, 5.6 * DX), dx=DX, cpml_layers=0,
                      boundary=BoundarySpec(x='periodic',
                                            y=Boundary(lo='pec', hi='pmc'), z='pec'))


def box(lo, hi, *, sheet=False):
    return Box((lo, .9 * DX, 2 * DX if sheet else .6 * DX),
               (hi, 5.3 * DX, 2 * DX if sheet else 4.1 * DX))


def realized(shapes, material='dielectric'):
    sim = model()
    sim.add_material('dielectric', eps_r=3.7)
    for shape in shapes:
        sim.add(shape, material=material)
    grid = sim._build_grid()
    sheets, wires = [], []
    built = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    if material == 'pec':
        return tuple(np.asarray(m) for m in realized_pec_edge_masks(
            built[3], sheets, wires, periodic=(True, False, False)))
    return (np.asarray(built[0].eps_r),)


def same_fill(actual, reference):
    assert any(np.any(m if m.dtype == bool else m != 1.) for m in reference)
    for a, b in zip(actual, reference):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('material', ['dielectric', 'pec'])
def test_box_spanning_many_periods_fills_the_cell(material):
    reference = realized([box(0., L)], material)
    # The second drawing is wholly outside [0,L]: clipping to k=0 cannot
    # pass just because the first drawing happens to cover the base cell.
    for lo, hi in [(-2 * L, 3.5 * L), (2 * L, 7.5 * L)]:
        same_fill(realized([box(lo, hi)], material), reference)


@pytest.mark.parametrize('material', ['dielectric', 'pec'])
def test_box_seam_fill_equals_two_halves(material):
    lo, hi = L - 3.2 * DX, L + 2.4 * DX
    same_fill(realized([box(lo, hi)], material),
              realized([box(lo, L), box(0., hi - L)], material))


@pytest.mark.parametrize('material', ['dielectric', 'pec'])
@pytest.mark.parametrize('kind', ['sphere', 'cylinder', 'wire_volume', 'mesh'])
def test_curved_seam_fill_equals_two_translated_bodies(material, kind):
    center = (L - .6 * DX, 3.1 * DX, 2.7 * DX)
    def body(x):
        c = (x, *center[1:])
        if kind == 'sphere':
            return Sphere(c, 1.8 * DX)
        if kind == 'cylinder':
            return Cylinder(c, 1.8 * DX, 2.7 * DX)
        if kind == 'wire_volume':
            return PolylineWire(((x - DX, *c[1:]), (x + DX, *c[1:])), 1.1 * DX)
        trimesh = pytest.importorskip('trimesh')
        pytest.importorskip('rtree')
        from rfx.geometry.mesh_import import MeshShape
        mesh = trimesh.creation.box(extents=(3.6 * DX, 2.8 * DX, 2.3 * DX))
        mesh.apply_translation(c)
        return MeshShape(mesh)
    shape, translated = body(center[0]), body(center[0] - L)
    same_fill(realized([shape], material), realized([shape, translated], material))


def test_sheet_footprint_seam_fill_equals_two_halves():
    lo, hi = L - 3.2 * DX, L + 2.4 * DX
    same_fill(realized([box(lo, hi, sheet=True)], 'pec'),
              realized([box(lo, L, sheet=True), box(0., hi - L, sheet=True)], 'pec'))


def test_wire_across_seam_is_still_a_directed_interval():
    sim = model()
    sim.add_port((L - 1.2 * DX, 3.1 * DX, 2.4 * DX), component='ex',
                 extent=2.7 * DX, impedance=50., excite=False)
    sim.add_source((.31 * L, 2.2 * DX, 3.3 * DX), 'ez', amplitude_kind='field')
    sim.add_probe((.68 * L, 4.3 * DX, 1.2 * DX), 'ez')
    with pytest.raises(ValueError, match="axis 'x'.*interval.*L=0.0137"):
        sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)


@pytest.mark.parametrize('kind', ['via', 'curved_patch'])
@pytest.mark.parametrize('material', ['dielectric', 'pec'])
def test_composite_thin_boxes_translate_without_extra_endpoint_cells(kind, material):
    from rfx.geometry.curved import CurvedPatch
    from rfx.geometry.via import Via
    def arrays(shift):
        x = L + .2 * DX - shift * DX
        shape = (Via(center=(x, 3.1 * DX), drill_radius=.2 * DX,
                     pad_radius=1.6 * DX, layers=[(1.2 * DX, 4.4 * DX)])
                 if kind == 'via' else
                 CurvedPatch(center=(x, 3.1 * DX, 1.2 * DX), length=4.6 * DX,
                             width=3.2 * DX, radius=2.7 * DX))
        return realized([shape], material)
    # A composite's sub-cell Boxes must choose their nearest node on the
    # complete image lattice, as a directly declared thin Box does.
    same_fill(tuple(np.roll(a, -8, axis=0) for a in arrays(0)), arrays(8))


def test_shape_without_bounds_uses_adjacent_images():
    class UnboundedSampler:
        def mask_on_coords(self, x, y, z):
            return Sphere((L - .6 * DX, 3.1 * DX, 2.7 * DX), 1.8 * DX).mask_on_coords(x, y, z)
    class BoundedSampler(UnboundedSampler):
        def bounding_box(self):
            return Sphere((L - .6 * DX, 3.1 * DX, 2.7 * DX), 1.8 * DX).bounding_box()
    # Both custom shapes use node sampling; built-in Sphere materials use
    # cell centres, so they would be a different physical reference here.
    same_fill(realized([UnboundedSampler()]),
              realized([BoundedSampler()]))


@pytest.mark.parametrize('rim', ['low', 'high'])
def test_sheet_endpoint_roundoff_does_not_fill_the_opposite_gap(rim):
    if rim == 'high':
        actual = box(DX, np.nextafter(L, np.inf), sheet=True)
        reference = box(DX, L, sheet=True)
    else:
        actual = box(np.nextafter(0., -np.inf), L - DX, sheet=True)
        reference = box(0., L - DX, sheet=True)
    same_fill(realized([actual], 'pec'), realized([reference], 'pec'))


def test_traced_zero_thickness_box_keeps_plane_admission():
    def material(x):
        sim = model()
        sim.add_material('dielectric', eps_r=3.7)
        sim.add(box(x, x), material='dielectric')
        return sim._build_materials(sim._build_grid())[0].eps_r
    # Staircase bounds, including a plane, must remain concrete.
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(material)(jnp.float32(1.5 * L)).block_until_ready()
