"""Synthetic inset patch; asymmetric, off-node in-plane coordinates only."""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.preflight.msl import msl_probe_clearance_for_port

DX = 0.002


def synthetic_patch(direction="+y", *, front=0.0706, back_to_edge=True,
                    offset=10, feed=.0246, spacing=2):
    prop = 0 if direction[-1] == "x" else 1
    width = 1 - prop
    sign = 1 if direction[0] == "+" else -1
    domain = [0.100, 0.100, 0.024]
    domain[prop] = 0.150
    def point(p, w, z):
        xyz = [0., 0., z]
        xyz[prop] = p if sign > 0 else domain[prop] - p
        xyz[width] = w
        return xyz
    sim = Simulation(freq_max=3e9, domain=tuple(domain), dx=DX,
                     boundary="cpml", cpml_layers=4, snap="declared")
    sim.add_material("substrate", eps_r=3.2)
    def box(a, b, material):
        a, b = np.array(point(*a)), np.array(point(*b))
        sim.add(Box(tuple(np.minimum(a, b)), tuple(np.maximum(a, b))),
                material=material)
    box((0, .0186, .004), (.1374, .0786, .008), "substrate")
    box((0, .0186, .004), (.1374, .0786, .004), "pec")
    box((0 if back_to_edge else feed, .0433, .008),
        (front + .006, .0533, .008), "pec")
    # A rectangular patch with a rectangular inset around the feed.
    box((front, .0273, .008), (front + .006, .0393, .008), "pec")
    box((front, .0573, .008), (front + .006, .0713, .008), "pec")
    box((front + .006, .0273, .008), (front + .034, .0713, .008), "pec")
    sim.add_msl_port(position=tuple(point(feed, .0483, .004)),
                     width=.010, height=.004, direction=direction,
                     eps_r_sub=3.2, mode="uniform", name="synthetic",
                     n_probe_offset=offset, n_probe_spacing=spacing, n_probes=3)
    return sim


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
@pytest.mark.parametrize("offset,status", [(14, "insufficient"), (3, "satisfied")])
def test_finite_ground_patch_clearance_and_message(direction, offset, status):
    sim = synthetic_patch(direction, offset=offset)
    record = msl_probe_clearance_for_port(sim, sim._msl_ports[0],
                                          sim._build_realized_grid())
    # Rounded feed node is 24 mm; deepest node is 24+(offset+4)*2 mm.
    expected_gap = .0706 - (.024 + (offset + 4) * DX)
    assert record.deepest_gap_m == pytest.approx(expected_gap, abs=1e-8)
    assert record.status == status
    assert "70.60" in record.reflector or "79.40" in record.reflector
    messages = [str(i) for i in sim.preflight(strict=False, check_ntff=False)]
    hits = [m for m in messages if "from a strong reflector candidate" in m]
    if status == "insufficient":
        assert len(hits) == 1
        assert f"deepest probe at {direction[-1]}=" in hits[0]
        assert f"{expected_gap*1e6:.0f}µm" in hits[0]
    else:
        assert not hits


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
def test_auto_placement_also_passes_ground_reference(direction):
    # Both production scan callers must exclude the finite ground.
    sim = synthetic_patch(direction, offset=None)
    grid = sim._build_realized_grid()
    finite = sim._resolve_msl_probe_entries(grid)[0]
    sim._geometry.pop(1)  # geometry-only control: remove the ground sheet
    absent = sim._resolve_msl_probe_entries(grid)[0]
    assert finite.n_probe_offset == absent.n_probe_offset
    assert finite.n_probe_spacing == absent.n_probe_spacing


@pytest.mark.parametrize("ground_top,expected", [(.004, .046), (.003, .046), (.006, 0.)])
@pytest.mark.parametrize("thin", [False, True])
def test_ground_upper_surface_and_wide_top_metal(ground_top, expected, thin):
    from types import SimpleNamespace
    from rfx.preflight.msl_reflector import msl_nearest_downstream_reflector
    ground = Box((0, .01, .002), (.14, .07, ground_top))
    # A very wide conductor ABOVE ground is still a reflector.
    patch = Box((.070, 0, .008), (.104, .100, .008))
    geometry = [SimpleNamespace(shape=patch, material_name="pec")]
    sheets = []
    if thin:
        sheets.append(SimpleNamespace(shape=ground))
    else:
        geometry.append(SimpleNamespace(shape=ground, material_name="pec"))
    distance, _, _ = msl_nearest_downstream_reflector(
        geometry, x_probe=.024, x_feed=.024, y_feed=.0483,
        w_trace=.010, dx=DX, domain_y=.100, direction="+x",
        thin_conductors=sheets, ground_plane=.004)
    assert distance == pytest.approx(expected)
    # A caller without a port reference retains the legacy width rule.
    legacy, _, _ = msl_nearest_downstream_reflector(
        geometry[:1], x_probe=.024, x_feed=.024, y_feed=.0483,
        w_trace=.010, dx=DX, domain_y=.100, direction="+x")
    assert np.isinf(legacy)


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
@pytest.mark.parametrize("ground_width", [.006, .010, .020, .044, .066],
                         ids=["narrower-than-strip", "strip-width", "between",
                              "patch-width", "wider-than-patch"])
@pytest.mark.parametrize("ground_top", [.004, .002], ids=["on-plane", "below-plane"])
def test_ground_side_metal_never_limits_auto_ladder_or_clearance(
        direction, ground_width, ground_top):
    """Geometry-only comparison: removing ground cannot move either reader."""
    from dataclasses import replace

    sim = synthetic_patch(direction, offset=None, spacing=None)
    width_axis = 1 if direction[-1] == "x" else 0
    ground = sim._geometry[1]
    lo, hi = list(ground.shape.corner_lo), list(ground.shape.corner_hi)
    lo[width_axis] = .0483 - ground_width / 2
    hi[width_axis] = .0483 + ground_width / 2
    lo[2] = hi[2] = ground_top
    sim._geometry[1] = replace(ground, shape=Box(tuple(lo), tuple(hi)))
    grid = sim._build_realized_grid()
    with_ground, = sim._resolve_msl_probe_entries(grid)
    clearance = msl_probe_clearance_for_port(sim, with_ground, grid)
    sim._geometry.pop(1)
    without_ground, = sim._resolve_msl_probe_entries(grid)
    reference = msl_probe_clearance_for_port(sim, without_ground, grid)
    assert with_ground.n_probe_offset == without_ground.n_probe_offset == 11
    assert with_ground.n_probe_spacing == without_ground.n_probe_spacing == 2
    assert clearance == reference
    assert clearance.deepest_gap_m == pytest.approx(.0166)
    assert clearance.status == "satisfied"


@pytest.mark.parametrize("rise", [np.spacing(.004), .4 * DX])
def test_ground_a_fraction_of_a_cell_above_the_reference_is_still_ground(rise):
    # A ground top a rounding error or a rasterization residual above the port
    # reference must not become "a conductor containing the feed" (it did:
    # distance -52 mm to a domain-wide ground one ULP above the plane).
    from types import SimpleNamespace
    from rfx.preflight.msl_reflector import msl_nearest_downstream_reflector
    ground = Box((0, 0, .002), (.14, .100, .004 + rise))
    patch = Box((.070, .030, .008), (.104, .070, .008))
    geometry = [SimpleNamespace(shape=patch, material_name="pec"),
                SimpleNamespace(shape=ground, material_name="pec")]
    distance, _, _ = msl_nearest_downstream_reflector(
        geometry, x_probe=.024, x_feed=.024, y_feed=.0483,
        w_trace=.010, dx=DX, domain_y=.100, direction="+x",
        ground_plane=.004, ground_cell=DX)
    assert distance == pytest.approx(.046)
