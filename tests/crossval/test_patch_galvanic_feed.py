"""The actual cv05/cv15 wire must reach both sheets before a field solve."""
from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import runpy
import sys

import pytest


CROSSVAL = Path(__file__).resolve().parents[2] / "validation" / "crossval"


def _forbid_solve(*args, **kwargs):
    raise AssertionError("a galvanic build-only check reached FDTD")


def _load_cv15():
    spec = importlib.util.spec_from_file_location(
        "_cv15_feed_contract", CROSSVAL / "15_patch_antenna_rt5880.py")
    case = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = case
    spec.loader.exec_module(case)
    return case


@pytest.fixture(scope="module", params=["cv05", "cv15"])
def patch_case(request):
    """Use the real case builders; cv05 stops at its existing build-only exit."""
    from rfx import Simulation

    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(CROSSVAL))
        mp.setattr(Simulation, "run", _forbid_solve)
        if request.param == "cv15":
            case = _load_cv15()
            sim, _, geom = case.build_rfx_sim(do_gain=False)
            grid = sim._build_grid()
            stack = case.assert_realized_stack(sim, grid)

            def check(candidate):
                # This metadata deliberately remains unchanged in the
                # registration mutations below: it used to hide the gap.
                return case.assert_galvanic_feed(candidate, grid, geom, stack_check=stack)

            n_sub = case.N_SUB
        else:
            import _patch_feed_contract as contract

            original = contract.assert_galvanic_patch_feed
            seen = []

            def capture(sim, grid, **terminals):
                result = original(sim, grid, **terminals)
                seen.append((sim, grid, terminals))
                return result

            mp.setattr(contract, "assert_galvanic_patch_feed", capture)
            mp.setenv("RFX_CV05_BUILD_ONLY", "1")
            for key in ("RFX_CV05_REALIZED_JSON", "RFX_CV05_SHEET_PLANE_DELTA",
                        "RFX_CV05_PATCH_L_MM"):
                mp.delenv(key, raising=False)
            with pytest.raises(SystemExit) as stopped:
                runpy.run_path(str(CROSSVAL / "05_patch_antenna.py"))
            assert stopped.value.code == 0
            assert len(seen) == 1
            sim, grid, terminals = seen[0]

            def check(candidate):
                return original(candidate, grid, **terminals)

            n_sub = 6
    return request.param, sim, grid, check, n_sub


def _copy_case(sim):
    clone = copy.copy(sim)
    clone._ports = list(sim._ports)
    clone._geometry = list(sim._geometry)
    clone._thin_conductors = list(sim._thin_conductors)
    return clone


def test_registered_feed_touches_both_realized_sheets(patch_case):
    _, sim, _, check, n_sub = patch_case
    result = check(sim)
    assert result["galvanic"]
    assert result["z1_node_k"] - result["z0_node_k"] == n_sub
    assert result["port_z0"] == sim._ports[0].position[2]
    assert result["port_extent"] == sim._ports[0].extent


@pytest.mark.parametrize("extent", [None, 0., float("nan")])
def test_case_requires_a_positive_finite_wire_extent(patch_case, extent):
    _, base, _, check, _ = patch_case
    sim = _copy_case(base)
    sim._ports[0] = replace(sim._ports[0], extent=extent)
    with pytest.raises(RuntimeError, match="positive finite extent"):
        check(sim)


@pytest.mark.parametrize("gap_cells", [2, 3])
@pytest.mark.parametrize("side", ["ground", "patch"])
def test_registered_gap_is_rejected_even_when_geometry_metadata_is_unchanged(
        patch_case, gap_cells, side):
    _, base, _, check, n_sub = patch_case
    sim = _copy_case(base)
    port = sim._ports[0]
    gap = gap_cells * port.extent / n_sub
    start = list(port.position)
    if side == "ground":
        start[2] += gap
    sim._ports[0] = replace(port, position=tuple(start), extent=port.extent-gap)
    with pytest.raises(RuntimeError, match="actual source endpoints"):
        check(sim)


@pytest.mark.parametrize("which", ["ground", "patch"])
def test_a_missing_conductor_cannot_count_as_contact(patch_case, which):
    name, base, _, check, _ = patch_case
    sim = _copy_case(base)
    port = sim._ports[0]
    z = port.position[2] + (port.extent if which == "patch" else 0.)
    if name == "cv05":
        sim._thin_conductors = [e for e in sim._thin_conductors
                                if abs(e.shape.corner_lo[2]-z) > 1e-12]
        assert len(sim._thin_conductors) == 1
    else:
        sim._geometry = [e for e in sim._geometry
                         if not (e.material_name == "pec"
                                 and abs(e.shape.corner_lo[2]-z) < 1e-12)]
        assert len(sim._geometry) == len(base._geometry)-1
    with pytest.raises(RuntimeError, match="actual source endpoints"):
        check(sim)


def test_cv05_rejects_a_bad_feed_before_its_first_ringdown_run(monkeypatch):
    """The normal script must refuse before Harminv, not only before Part 3."""
    from rfx import Simulation

    original = Simulation.add_port

    def shorten(sim, *args, **kwargs):
        result = original(sim, *args, **kwargs)
        port = sim._ports[-1]
        sim._ports[-1] = replace(port, extent=port.extent/2)
        return result

    monkeypatch.syspath_prepend(str(CROSSVAL))
    monkeypatch.setattr(Simulation, "add_port", shorten)
    monkeypatch.setattr(Simulation, "run", _forbid_solve)
    for key in ("RFX_CV05_BUILD_ONLY", "RFX_CV05_REALIZED_JSON",
                "RFX_CV05_SHEET_PLANE_DELTA", "RFX_CV05_PATCH_L_MM"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="actual source endpoints"):
        runpy.run_path(str(CROSSVAL / "05_patch_antenna.py"))


def test_cv05_cannot_substitute_an_intermediate_sheet_for_the_patch(monkeypatch):
    """The centre-column stack remains valid; the feed must reach its patch."""
    from rfx import Box, Simulation

    original = Simulation.add_port

    def short_to_other_sheet(sim, *args, **kwargs):
        result = original(sim, *args, **kwargs)
        port = sim._ports[-1]
        x, y, z = port.position
        middle = z + port.extent/2
        d = sim._dx
        sim.add(Box((x-d, y-d, middle), (x+d, y+d, middle)), material="pec")
        sim._ports[-1] = replace(port, extent=port.extent/2)
        return result

    monkeypatch.syspath_prepend(str(CROSSVAL))
    monkeypatch.setattr(Simulation, "add_port", short_to_other_sheet)
    monkeypatch.setattr(Simulation, "run", _forbid_solve)
    monkeypatch.setenv("RFX_CV05_BUILD_ONLY", "1")
    for key in ("RFX_CV05_REALIZED_JSON", "RFX_CV05_SHEET_PLANE_DELTA",
                "RFX_CV05_PATCH_L_MM"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="stack ground/patch"):
        runpy.run_path(str(CROSSVAL / "05_patch_antenna.py"))


def test_cv15_rejects_a_wire_wholly_inside_metal_even_with_valid_stack(monkeypatch):
    from rfx import Box, Simulation
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.sources.sources import WirePort, _wire_port_live_cells

    monkeypatch.syspath_prepend(str(CROSSVAL))
    monkeypatch.setattr(Simulation, "run", _forbid_solve)
    case = _load_cv15()
    sim, patch, geom = case.build_rfx_sim(do_gain=False)
    port = sim._ports[0]
    # Outside the patch footprint, so its stack remains unmodified. The
    # generated block spans the SAME ground/patch heights but is one metal
    # body: both wall contacts alone cannot establish a driven gap.
    x, y, z = 3*case.DX, port.position[1], port.position[2]
    sim.add(Box((x-case.DX, y-case.DX, z),
                (x+case.DX, y+case.DX, z+port.extent)), material="pec")
    sim._ports[0] = replace(port, position=(x, y, z))
    grid = sim._build_grid()
    case.assert_realized_stack(sim, grid, patch)
    sheets, wires = [], []
    material = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    edges = realized_pec_edge_masks(material[3], sheets, wires,
                                    periodic=sim._periodic_flags())
    with pytest.raises(ValueError, match="all .*PEC"):
        _wire_port_live_cells(
            grid, WirePort((x, y, z), (x, y, z+port.extent), "ez"), edges)
    with pytest.raises(RuntimeError, match="no live source edge"):
        case.assert_galvanic_feed(sim, grid, geom)
