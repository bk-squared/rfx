"""#1826: declarations decide partial-aperture faces on both meshes."""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.depths import grid_face_depths
from rfx.boundaries.features import (
    WaveguideBoundaryWarning, admit_ntff, admit_waveguide, full_aperture,
)
from rfx.measurement.waveguide_planes import admit_waveguide_planes


def draw_aperture_walls(sim):
    """Make these face-allocation fixtures legal guides under rule 4."""
    for port in sim._waveguide_ports:
        normal = "xyz".index(port.direction[1])
        spans = [getattr(port, f"{axis}_range") or (0., sim._domain[a])
                 for a, axis in enumerate("xyz")]
        spans[normal] = (port.x_position - .002, port.x_position + .002)
        for a in range(3):
            if a == normal:
                continue
            for plane in spans[a]:
                lo, hi = [s[0] for s in spans], [s[1] for s in spans]
                lo[a] = hi[a] = plane
                sim.add(Box(tuple(lo), tuple(hi)), material="pec")


def partial_guide(*, graded=False, declared=True):
    kwargs = {"boundary": "cpml"} if declared else {}
    if graded:
        kwargs["dz_profile"] = np.full(16, .00254)
    sim = Simulation(12e9, (.08128, .06096, .04064), dx=.00254,
                     cpml_layers=4, **kwargs)
    sim.add_waveguide_port(.01016, direction="+x", f0=10e9,
                           freqs=np.array([9e9, 10e9, 11e9]),
                           y_range=(.01778, .04064), z_range=(.01524, .0254),
                           ref_offset=1, probe_offset=2)
    draw_aperture_walls(sim)
    return sim


def table(grid):
    return {face.name: (face.kind.value, face.realized)
            for face in grid_face_depths(grid)}


@pytest.mark.parametrize("boundary,axes,shape,kinds,depths", [
    ("pec", "xyz", (21, 11, 7),
     ("PEC", "PEC", "PEC", "PEC", "PEC", "PEC"), (0, 0, 0, 0, 0, 0)),
    ({"x": "cpml", "y": "pec", "z": "pec"}, "xyz", (29, 11, 7),
     ("ABSORBER", "ABSORBER", "PEC", "PEC", "PEC", "PEC"), (4, 4, 0, 0, 0, 0)),
    ("cpml", "xyz", (29, 19, 15),
     ("ABSORBER", "ABSORBER", "ABSORBER", "ABSORBER", "ABSORBER", "ABSORBER"),
     (4, 4, 4, 4, 4, 4)),
    ({"x": "periodic", "y": "cpml", "z": "cpml"}, "yz", (20, 19, 15),
     ("PERIODIC", "PERIODIC", "ABSORBER", "ABSORBER", "ABSORBER", "ABSORBER"),
     (0, 0, 4, 4, 4, 4)),
])
def test_no_port_grid_keeps_main_axes_shape_and_faces(boundary, axes, shape, kinds, depths):
    sim = Simulation(12e9, (.04, .02, .012), dx=.002, cpml_layers=4, boundary=boundary)
    grid = sim._build_grid()
    assert grid.cpml_axes == axes
    assert grid.shape == shape
    assert table(grid) == dict(zip(
        ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"), zip(kinds, depths)))


@pytest.mark.parametrize("declared", [False, True])
def test_partial_aperture_declared_faces(declared):
    expected = {
        "x_lo": ("ABSORBER", 4), "x_hi": ("ABSORBER", 4),
        "y_lo": ("ABSORBER", 4), "y_hi": ("ABSORBER", 4),
        "z_lo": ("ABSORBER", 4), "z_hi": ("ABSORBER", 4),
    }
    for graded in (False, True):
        sim = partial_guide(graded=graded, declared=declared)
        admit_waveguide(sim)
        grid = sim._build_realized_grid()
        assert table(grid) == expected
        assert full_aperture(sim, sim._waveguide_ports[0], grid) is False


def test_partial_aperture_ntff_has_absorbing_exterior():
    sim = partial_guide()
    sim.add_ntff_box((.00508, .00508, .00508), (.0762, .05588, .03556), freqs=[10e9])
    admit_ntff(sim, lane="contract")


@pytest.mark.parametrize("second", [None, "bend", "opposite-partial"])
def test_full_aperture_default_and_port_axis_precedence(second):
    sim = Simulation(12e9, (.04, .02, .012), dx=.002, cpml_layers=4)
    sim.add_waveguide_port(.008, direction="+x", f0=10e9,
                           ref_offset=1, probe_offset=2)
    if second == "bend":
        sim.add_waveguide_port(.004, direction="+y", f0=10e9,
                               x_range=(.008, .032), z_range=(.002, .01),
                               ref_offset=1, probe_offset=2)
    elif second == "opposite-partial":
        sim.add_waveguide_port(.032, direction="-x", f0=10e9,
                               y_range=(.002, .018), z_range=(.002, .01),
                               ref_offset=1, probe_offset=2)
    if second is not None:
        draw_aperture_walls(sim)
    expected = {
        "x_lo": ("ABSORBER", 4), "x_hi": ("ABSORBER", 4),
        "y_lo": ("PEC", 0), "y_hi": ("PEC", 0),
        "z_lo": ("PEC", 0), "z_hi": ("PEC", 0),
    }
    if second == "bend":
        expected["y_lo"] = expected["y_hi"] = ("ABSORBER", 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        admit_waveguide(sim)
        admit_waveguide(sim)
    assert sum(issubclass(w.category, WaveguideBoundaryWarning) for w in caught) == 1
    assert table(sim._build_grid()) == expected


@pytest.mark.parametrize("declared", [False, True])
def test_measurement_planes_keep_port_axis_indices(declared, monkeypatch):
    sim = partial_guide(declared=declared)
    build = sim._build_grid
    grids = []

    def capture(**kwargs):
        grid = build(**kwargs)
        grids.append(grid)
        return grid

    monkeypatch.setattr(sim, "_build_grid", capture)
    admit_waveguide_planes(sim, axis_idx=0, x_position=.01016, direction="+x",
                           ref_offset=1, probe_offset=2)
    grid, = grids
    assert grid.pad_x_lo == grid.pad_x_hi == 4
    source = grid.position_to_index((.01016, 0., 0.))[0]
    assert (source, source + 1, source + 2) == (8, 9, 10)
    assert ((source + 1 - 4) * grid.dx, (source + 2 - 4) * grid.dx) == (5 * .00254, 6 * .00254)


@pytest.mark.parametrize("declaration", [
    "wr90-nonwhole-cells", "not-passed", "cpml", "transverse-absorber", "pec-walls",
    "unknown-provenance", "imported-graded-cpml", "perpendicular-pec-z",
    "perpendicular-cpml", "periodic-admission", "pmc-admission",
])
@pytest.mark.parametrize("graded", [False, True])
def test_both_meshes_scene_aperture_answers_unchanged(declaration, graded):
    from tests.contracts.test_guide_admission_both_meshes import guide
    sim = guide(declaration, graded=graded)
    if declaration == "wr90-nonwhole-cells":
        # Main refuses this lattice's wall-length error before guide admission.
        # Accept that existing snap solely to inspect the same aperture spans.
        sim._snap = "declared"
    grid = sim._build_realized_grid()
    expected = [True] if declaration == "wr90-nonwhole-cells" else [True, True]
    assert [full_aperture(sim, entry, grid) for entry in sim._waveguide_ports] == expected


@pytest.mark.parametrize("boundary", [None, "cpml", {"x": "cpml", "y": "pec", "z": "pec"}])
@pytest.mark.parametrize("partial", [False, True])
def test_boundary_admission_scene_aperture_answers_unchanged(boundary, partial):
    from tests.contracts.test_waveguide_boundary_admission import guide
    sim = guide(boundary, partial=partial)
    assert full_aperture(sim, sim._waveguide_ports[0], sim._build_grid()) is (not partial)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("boundary", [None, "cpml", {"x": "cpml", "y": "pec", "z": "pec"}])
@pytest.mark.parametrize("dielectric", [False, True])
def test_graded_default_scene_aperture_answers_unchanged(graded, boundary, dielectric):
    from tests.contracts.test_graded_guide_default import guide
    sim = guide(graded=graded, boundary=boundary, dielectric=dielectric)
    grid = sim._build_realized_grid()
    assert [full_aperture(sim, entry, grid) for entry in sim._waveguide_ports] == [True, True]


@pytest.mark.parametrize("case", ["waveguide-cpml", "waveguide-pec", "waveguide-default"])
@pytest.mark.parametrize("entry", ["run", "wire-fast", "forward", "sweep", "nonuniform",
                                  "subgridded", "distributed", "gpu-query"])
def test_boundary_cases_aperture_answers_unchanged(case, entry):
    from tests.contracts.boundary_cases import build
    sim, _ = build(case, entry)
    assert full_aperture(sim, sim._waveguide_ports[0], sim._build_realized_grid()) is True
