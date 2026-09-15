"""MSL geometry advisories distinguish conductor gaps from material extents.

Build-only checks of actual PEC volume/sheet attachment on uniform and NU
grids. No source profile solve, field evolution or Z0 accuracy claim.
"""
from dataclasses import replace
import copy
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


U = 2.0**-13


def _point(direction, prop, width, normal):
    return (prop, width, normal) if direction[-1] == "x" else (width, prop, normal)


def _port(sim, direction, feed, centre, ground, height, width, name="checked"):
    sim.add_msl_port(
        position=_point(direction, feed, centre, ground), direction=direction,
        height=height, width=width, name=name,
        n_probe_offset=4, n_probe_spacing=2, n_probes=3,
    )


def _legacy(dx, direction="+x", *, height=254e-6, trace=True):
    """Only the declared board geometry is copied from the historical sweep."""
    width = 600e-6
    lateral = width + 2 * (2 * height + 8 * dx)
    domain = _point(direction, .006, lateral, height + .0015)
    sim = Simulation(freq_max=5e9, domain=domain, dx=dx, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("substrate", eps_r=3.66)
    sim.add(Box((0., 0., 0.), (domain[0], domain[1], height)), material="substrate")
    centre = lateral / 2
    if trace:
        sim.add(Box(_point(direction, 0., centre - width/2, height),
                    _point(direction, .006, centre + width/2, height + dx)), material="pec")
    _port(sim, direction, .002 if direction[0] == "+" else .004,
          centre, 0., height, width)
    return sim


def _offset_stack(direction, nonuniform, ground_kind, trace_kind):
    widths = np.full(16, U)
    if nonuniform:
        widths[4:7] *= [1.25, 1.5, 1.25]
    ground = 4 * U
    top = (8 if nonuniform else 7) * U
    domain = _point(direction, 32 * U, 24 * U, float(widths.sum()))
    sim = Simulation(freq_max=5e9, domain=domain, dx=U, cpml_layers=4,
                     boundary="cpml", **({"dz_profile": widths} if nonuniform else {}))
    # Both conductor kinds meet the same substrate-facing planes. The
    # lower volume extends DOWN from the ground, never into the source.
    bottom = ground - 2 * U if ground_kind == "volume" else ground
    trace_top = top + U if trace_kind == "volume" else top
    ground_shape = Box((0., 0., bottom), (domain[0], domain[1], ground))
    if ground_kind == "lossy-sheet":
        sim.add_thin_conductor(ground_shape, sigma_bulk=5.8e7,
                               thickness=10e-6, surface_impedance_f0=5e9)
    else:
        sim.add(ground_shape, material="pec")
    sim.add_material("substrate", eps_r=3.66)
    sim.add(Box((0., 0., ground), (domain[0], domain[1], top)), material="substrate")
    sim.add(Box(_point(direction, 0., 10 * U, top),
                _point(direction, 32 * U, 14 * U, trace_top)), material="pec")
    _port(sim, direction, (8 if direction[0] == "+" else 24) * U,
          12 * U, ground, top - ground, 4 * U)
    return sim, ground, top


def _messages(sim):
    """Invoke the actual MSL preflight check without unrelated check families."""
    grid = sim._build_realized_grid()
    low = [getattr(grid, f"pad_{axis}_lo") * grid.dx for axis in "xyz"]
    high = [getattr(grid, f"pad_{axis}_hi") * grid.dx for axis in "xyz"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim._check_msl_port_geometry(grid.dx, low, high)
    return [w.message for w in caught]


def _no_old_bbox_claims(messages):
    text = "\n".join(map(str, messages))
    for phrase in ("THICKER", "board-thickening", "worth roughly", "Z0 moves about",
                   "extractor tracks the realized-board", "0.60%", "<5% Z0 bias"):
        assert phrase not in text


@pytest.mark.parametrize("direction", ["+x", "-y"])
@pytest.mark.parametrize("dx,material_um,n_material,frac,n_gap", [
    (80e-6, 320., 4, .175, 3),
    (60e-6, 300., 5, 254/60 - 4, 4),
])
def test_legacy_misaligned_material_extent_is_not_the_conductor_gap(
    direction, dx, material_um, n_material, frac, n_gap,
):
    sim = _legacy(dx, direction)
    assembled = sim._msl_assemble_once()
    pe = sim._msl_ports[0]
    material = sim._msl_realized_substrate(pe, 2, assembled=assembled)
    gap = sim._msl_conductor_gap(pe, assembled)
    assert material["n"] == n_material
    assert material["h_real"] * 1e6 == pytest.approx(material_um, abs=1e-8)
    assert material["frac"] == pytest.approx(frac, abs=1e-10)
    assert gap["ground"] == 0.
    assert gap["trace"] == pytest.approx(240e-6, abs=1e-16)
    assert gap["h"] == pytest.approx(240e-6, abs=1e-16) and gap["n"] == n_gap
    messages = _messages(sim)
    assert not [m for m in messages if getattr(m, "code", "") == "msl_port_conductor_planes"]
    mixed, = [str(m) for m in messages if "mixed-cell danger zone" in str(m)]
    assert f"{frac:.3f} of a cell" in mixed
    assert "Validated conductor-plane gap=240.0µm" in mixed
    assert f"{n_material} same-permittivity sample slot(s), extent {material_um:.1f}µm" in mixed
    resolution = [str(m) for m in messages if "only " in str(m) and "normal interval(s)" in str(m)]
    assert len(resolution) == int(n_gap < 4)
    if resolution:
        assert "only 3 normal interval(s)" in resolution[0]
    _no_old_bbox_claims(messages)


@pytest.mark.parametrize("material_cells", [5, 8])
def test_snap_suggestions_follow_declared_face_when_material_continues_above_trace(material_cells):
    dx = 80e-6
    sim = _legacy(dx)
    sim.add(Box((0., 0., 254e-6),
                (sim._domain[0], sim._domain[1], material_cells * dx)), material="substrate")
    assembled = sim._msl_assemble_once()
    pe = sim._msl_ports[0]
    material = sim._msl_realized_substrate(pe, 2, assembled=assembled)
    gap = sim._msl_conductor_gap(pe, assembled)
    assert material["n"] == material_cells
    assert material["h_real"] == pytest.approx(material_cells * dx, abs=1e-16)
    assert material["frac"] == pytest.approx(.175, abs=1e-10)
    assert material["declared_n_above"] == 4
    assert gap["n"] == 3 and gap["h"] == pytest.approx(240e-6, abs=1e-16)
    mixed, = [str(m) for m in _messages(sim) if "mixed-cell danger zone" in str(m)]
    assert "set dx = 63.5µm (= h_sub/4) or 84.7µm (= h_sub/3)" in mixed
    assert f"{material_cells} same-permittivity sample slot(s)" in mixed


def test_translated_declared_faces_use_absolute_nodes_and_do_not_offer_unverified_dx():
    sim, _, _ = _offset_stack("+x", False, "sheet", "sheet")
    pe = sim._msl_ports[0]
    pos = (*pe.position[:2], 4.2 * U)
    sim._msl_ports[0] = replace(pe, position=pos)
    assembled = sim._msl_assemble_once()
    gap = sim._msl_conductor_gap(sim._msl_ports[0], assembled)
    assert gap["ground"] == 4 * U and gap["trace"] == 7 * U
    material = sim._msl_realized_substrate(sim._msl_ports[0], 2, assembled)
    assert material["n"] == 3 and material["h_real"] == 3 * U
    assert material["frac"] == pytest.approx(.2, abs=1e-14)
    mixed, = [str(m) for m in _messages(sim) if "mixed-cell danger zone" in str(m)]
    assert "0.200 of a cell" in mixed
    assert f"ground z={4.2*U*1e6:.1f}µm" in mixed
    assert f"trace z={7.2*U*1e6:.1f}µm" in mixed
    assert "set dx =" not in mixed  # height/n alone cannot align this translation
    assert "fraction alone does not establish material/PEC overlap" in mixed


def test_known_nu_geometry_does_not_fall_back_to_scalar_dx_when_material_is_absent():
    sim, ground, top = _offset_stack("+x", True, "sheet", "sheet")
    sim._geometry.pop(1)  # remove only the dielectric, retaining both conductor planes
    sim._dx = 3 * U      # base in-plane spacing is not the graded normal spacing
    assembled = sim._msl_assemble_once()
    pe = sim._msl_ports[0]
    assert sim._msl_realized_substrate(pe, 2, assembled) is None
    face = sim._msl_declared_face_geometry(pe, assembled[0])
    assert face["nonuniform"] and face["frac"] == 0.
    assert face["trace"] == top and face["d_iface"] == U
    gap = sim._msl_conductor_gap(pe, assembled)
    assert gap["h"] == top - ground == 4 * U and gap["n"] == 3
    messages = _messages(sim)
    assert not any("mixed-cell danger zone" in str(m) for m in messages)
    assert not any("scalar estimate, run grid unavailable" in str(m) for m in messages)
    assert not [m for m in messages if getattr(m, "code", "") == "msl_port_conductor_planes"]


@pytest.mark.parametrize("ground_node", [4, 5])
@pytest.mark.parametrize("direction", ["+x", "-y"])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_layered_material_walk_uses_the_ports_lower_tie_ground(ground_node, direction, nonuniform):
    domain = _point(direction, 32 * U, 24 * U, 16 * U)
    sim = Simulation(freq_max=5e9, domain=domain, dx=U, cpml_layers=4,
                     boundary="cpml", **({"dz_profile": np.full(16, U)} if nonuniform else {}))
    ground, interface, top = ground_node * U, (ground_node + 1) * U, 8 * U
    sim.add(Box((0., 0., ground), (domain[0], domain[1], ground)), material="pec")
    sim.add_material("first_layer", eps_r=3.)
    sim.add_material("second_layer", eps_r=5.)
    sim.add(Box((0., 0., ground), (domain[0], domain[1], interface)), material="first_layer")
    sim.add(Box((0., 0., interface), (domain[0], domain[1], top)), material="second_layer")
    sim.add(Box(_point(direction, 0., 10 * U, top),
                _point(direction, 32 * U, 14 * U, top)), material="pec")
    # #931 ties select the lower node at BOTH parities. Banker's rounding
    # incorrectly starts the material walk in the second layer at 5.5U.
    declared_ground = (ground_node + .5) * U
    _port(sim, direction, (8 if direction[0] == "+" else 24) * U,
          12 * U, declared_ground, top - declared_ground, 4 * U)
    assembled = sim._msl_assemble_once()
    pe = sim._msl_ports[0]
    gap = sim._msl_conductor_gap(pe, assembled)
    assert gap["ground"] == ground and gap["trace"] == top
    assert gap["n"] == 8 - ground_node
    material = sim._msl_realized_substrate(pe, 2, assembled)
    assert material["ground_cells"] == 0
    assert material["n"] == 1 and material["h_real"] == U
    assert material["frac"] == 0.
    messages = _messages(sim)
    assert any(f"1 same-permittivity sample slot(s), extent {U*1e6:.1f}µm" in str(m)
               for m in messages)
    assert not [m for m in messages if getattr(m, "code", "") == "msl_port_conductor_planes"]


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("ground_kind,trace_kind", [("volume", "sheet"), ("sheet", "volume")])
def test_gap_uses_substrate_facing_planes_and_actual_profile_intervals(
    direction, nonuniform, ground_kind, trace_kind,
):
    sim, ground, top = _offset_stack(direction, nonuniform, ground_kind, trace_kind)
    assembled = sim._msl_assemble_once()
    gap = sim._msl_conductor_gap(sim._msl_ports[0], assembled)
    assert gap["n"] == 3
    assert gap["ground"] == pytest.approx(ground, abs=1e-16)
    assert gap["trace"] == pytest.approx(top, abs=1e-16)
    assert gap["h"] == pytest.approx((4 if nonuniform else 3) * U, abs=1e-16)
    messages = _messages(sim)
    resolution, = [str(m) for m in messages if "only 3 normal interval(s)" in str(m)]
    assert f"gap={(top-ground)*1e6:.1f}µm" in resolution
    assert not [m for m in messages if getattr(m, "code", "") == "msl_port_conductor_planes"]
    _no_old_bbox_claims(messages)


@pytest.mark.parametrize("failure", ["missing-trace", "inside-trace", "unavailable-assembly"])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_invalid_attachment_has_no_fabricated_or_automatically_adapted_gap(
    failure, nonuniform, monkeypatch,
):
    if nonuniform:
        sim, _, _ = _offset_stack("+y", True, "volume", "volume")
        if failure == "missing-trace":
            sim._geometry.pop()
    else:
        sim = _legacy(80e-6, trace=failure != "missing-trace")
    if failure == "inside-trace":
        # The upper trace wall at 320um exists, but moving there would put
        # the source through the PEC volume. The valid lower wall is 240um;
        # the diagnostic must not silently replace this declared interval.
        height = sim._msl_ports[0].height + U if nonuniform else 334e-6
        sim._msl_ports[0] = replace(sim._msl_ports[0], height=height)
    if failure == "unavailable-assembly":
        monkeypatch.setattr(sim, "_msl_assemble_once", lambda: None)
    else:
        with pytest.raises(ValueError):
            sim._msl_conductor_gap(sim._msl_ports[0], sim._msl_assemble_once())
    messages = _messages(sim)
    errors = [m for m in messages if getattr(m, "code", "") == "msl_port_conductor_planes"]
    assert len(errors) == 1 and errors[0].severity == "error"
    assert "gap is unavailable" in str(errors[0])
    assert not any("Validated conductor-plane gap=" in str(m) for m in messages)
    assert not any("only " in str(m) and "normal interval(s)" in str(m) for m in messages)


@pytest.mark.parametrize("relative_shift,expected_warning", [(.08, False), (.09, True)])
def test_existing_gap_difference_threshold_is_preserved_without_z0_prediction(
    relative_shift, expected_warning,
):
    # Both interfaces snap UP to four normal intervals; the declared-face
    # fractions (~.704/.670) lie outside check 2b, and check 2 is silent.
    dx = 80e-6
    height = 4 * dx / (1 + relative_shift)
    sim = _legacy(dx, height=height)
    messages = _messages(sim)
    hits = [str(m) for m in messages if "geometry-advisory threshold" in str(m)]
    assert len(hits) == int(expected_warning)
    if hits:
        assert "existing 8.3%" in hits[0]
        assert "gap=320.0µm over 4 normal interval(s)" in hits[0]
        assert "does not predict a Z0 change" in hits[0]
    _no_old_bbox_claims(messages)


def test_all_ports_share_one_assembly_for_material_and_conductor_diagnostics(monkeypatch):
    sim = _legacy(80e-6)
    first = sim._msl_ports[0]
    _port(sim, "-x", .004, first.position[1], 0., first.height, first.width, name="second")
    assemblies, passed_carriers = [], []
    original_assemble = sim._assemble_realized
    original_gap = sim._msl_conductor_gap

    def assemble(*args, **kwargs):
        result = original_assemble(*args, **kwargs)
        assemblies.append(result)
        return result

    def gap(pe, assembled):
        passed_carriers.append(assembled[4])
        return original_gap(pe, assembled)

    monkeypatch.setattr(sim, "_assemble_realized", assemble)
    monkeypatch.setattr(sim, "_msl_conductor_gap", gap)
    messages = _messages(sim)
    assert len(assemblies) == 1
    assert len(passed_carriers) == 2 and all(r is assemblies[0] for r in passed_carriers)
    assert sum("only 3 normal interval(s)" in str(m) for m in messages) == 2


@pytest.mark.parametrize("nonuniform", [False, True])
def test_lossy_sheet_ground_is_observed_without_becoming_a_pec_plane(nonuniform):
    sim, ground, top = _offset_stack("-y", nonuniform, "lossy-sheet", "sheet")
    assembled = sim._msl_assemble_once()
    realized = assembled[4]
    assert len(realized.sheet_specs) == 1
    before = tuple(np.asarray(edge).copy() for edge in realized.edges)
    gap = sim._msl_conductor_gap(sim._msl_ports[0], assembled)
    assert gap["n"] == 3
    assert gap["ground"] == pytest.approx(ground, abs=1e-16)
    assert gap["trace"] == pytest.approx(top, abs=1e-16)
    # Removing only the observational f0 sheet removes the ground; it was
    # never present in the hard PEC masks supplied to the source operator.
    without_sheet = copy.copy(realized)
    without_sheet.sheet_specs = ()
    with pytest.raises(ValueError, match="ground"):
        sim._msl_conductor_gap(sim._msl_ports[0], (*assembled[:4], without_sheet))
    for actual, original in zip(realized.edges, before):
        np.testing.assert_array_equal(actual, original)
    messages = _messages(sim)
    assert not [m for m in messages if getattr(m, "code", "") == "msl_port_conductor_planes"]
    assert any(f"gap={(top-ground)*1e6:.1f}µm" in str(m) for m in messages)
