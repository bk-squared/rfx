"""cv07 -- the build-time realized-metal gate (#931 lattice ownership).

`07_sheen_lpf.py` declares its three metal footprints as SHEETS (zero-
thickness Boxes on the substrate-top node plane). Before #931 they were
one-cell Boxes, which the contract realizes as a 200 um filled metal slab
with electric walls on BOTH bounding planes -- a quarter of the substrate
thickness of metal on a board whose openEMS comparator leg draws the same
metal with zero thickness.

Every assertion here is a BUILD, no time stepping: the whole file runs in
seconds. The geometry comes from the script's own `build_rfx_sim`, never
from a copy -- a test-local mirror is how cv15's equivalent check stayed
green after the fix it tested had been deleted (see
tests/crossval/test_crossval_cv15_wall_planes.py's docstring).

The negative arm is the mesh the design note's "redraw on-lattice" rule
would pick, dx = H_SUB/4 = 198.5 um: it realizes the two nominally
identical 50-ohm feeds 12 and 13 node rows wide, and the gate must refuse
it. That is the measurement behind cv07 keeping dx = 200 um.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CV07 = REPO_ROOT / "validation/crossval/07_sheen_lpf.py"

DX_COMMITTED = 200e-6


def _load_cv07():
    spec = importlib.util.spec_from_file_location("_cv07_realized_metal", CV07)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv07():
    return _load_cv07()


@pytest.fixture(scope="module")
def measured(cv07):
    return cv07.assert_realized_metal(cv07.build_rfx_sim(DX_COMMITTED),
                                      DX_COMMITTED)


def test_metal_is_three_sheets_on_one_plane(measured):
    assert measured["n_sheets"] == 3
    assert measured["n_volume_cells"] == 0
    # One tangential wall plane, not two: two is what a one-cell VOLUME
    # trace realizes, and it is what this script drew before #931.
    assert measured["plane_k"] == measured["sub_top_k"]


def test_sheet_sits_on_the_realized_laminate_face(cv07, measured):
    """The foil is on the substrate TOP, not buried in the laminate.

    h_sub = 794 um at dx = 200 um realizes 4 cells = 800 um, so the top
    face the mesh builds is z = 800 um and the sheet lands there: +6.0 um
    from the declared face, and no vacuum slot between metal and laminate
    (the #702 geometry). The declared-vs-realized offset is reported, not
    absorbed.
    """
    assert measured["plane_z"] == pytest.approx(800e-6, rel=1e-12)
    assert measured["plane_z"] - cv07.H_SUB == pytest.approx(6e-6, abs=1e-9)


def test_both_50_ohm_feeds_realize_the_same_width(measured):
    """They are the same declared object; a mesh that realizes them one row
    apart makes a symmetric board asymmetric."""
    assert measured["in_feed"]["n_rows"] == measured["out_feed"]["n_rows"] == 12
    assert measured["feed_w"] == pytest.approx(2200e-6, rel=1e-12)
    assert measured["out_feed"]["trv"] == pytest.approx(2200e-6, rel=1e-12)


def test_realized_patch_is_one_cell_under_declared_on_each_axis(cv07, measured):
    """A sheet IS its edge set: n footprint nodes carry n-1 edges, so every
    in-plane extent realizes one cell smaller than the pre-#931 table."""
    dx = DX_COMMITTED
    assert measured["patch_trv"] == pytest.approx(20200e-6, rel=1e-12)
    assert measured["patch_prop"] == pytest.approx(2400e-6, rel=1e-12)
    # pre-#931 this board reported 20400 / 2600 um -- exactly one cell more.
    assert measured["patch_trv"] + dx == pytest.approx(20400e-6, rel=1e-12)
    assert measured["patch_prop"] + dx == pytest.approx(2600e-6, rel=1e-12)


def test_shared_helper_agrees_with_the_case_gate(cv07):
    """The case's own gate and the branch-wide helper must say the same
    thing. `tests/_realized_geometry` is the single shared spelling of the
    build-time check (#931); this case carries its own `realized_metal`
    because a crossval script cannot import from `tests/`."""
    from tests._realized_geometry import assert_sheet_planes, assert_wall_planes

    sim = cv07.build_rfx_sim(DX_COMMITTED)
    # 800 um = the realized laminate face, one plane, not the two a one-cell
    # volume trace would stand.
    assert_wall_planes(sim, 2, [800e-6], what="cv07 metal")
    assert_sheet_planes(sim, 2, [800e-6] * 3, what="cv07 feeds + patch")


def test_port_cross_section_row_mismatch_is_reported_not_hidden(measured):
    """#931 §1.9 / #729, left OPEN by this case: the MSL port rounds each
    face to the nearest node while a sheet footprint is closed [lo, hi], so
    each port integrates one row that carries no metal. The gate must say
    so in its report rather than pass silently."""
    for (w_lo, w_hi), feed in zip(measured["port_rows"],
                                  (measured["in_feed"], measured["out_feed"])):
        assert w_hi - w_lo + 1 == feed["n_rows"] + 1
        assert w_lo == feed["j"][0] - 1
    assert any("OPEN DEFECT" in line for line in measured["report"])


def test_the_on_lattice_mesh_is_refused_with_its_measurement(cv07):
    """dx = H_SUB/4 = 198.5 um puts the sheet on the DECLARED interface but
    realizes the two feeds 12 and 13 rows wide. The gate refuses it, and
    the refusal names the asymmetry -- this is the falsifier arm for the
    mesh decision recorded in the script's #931 docstring section."""
    dx = cv07.H_SUB / 4.0
    with pytest.raises(RuntimeError) as exc:
        cv07.assert_realized_metal(cv07.build_rfx_sim(dx), dx)
    msg = str(exc.value)
    assert "12 and 13 node rows" in msg
    assert "symmetric board asymmetric" in msg


def test_the_msl_trace_detector_finds_a_sheet_trace(cv07):
    """The stage-C detector reads realized wall planes instead of scanning
    pec_mask cells (#931 §1.9). A sheet trace owns no cell, so this is the
    coupling that has to hold before the case can be solved at all."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.probes.msl_wave_decomp import realized_trace_planes_on_column
    from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry

    sim = cv07.build_rfx_sim(DX_COMMITTED)
    grid = sim._build_grid()
    sheets: list = []
    wires: list = []
    assembled = sim._assemble_materials(grid, pec_sheets=sheets,
                                        pec_wires=wires)
    periodic = sim._periodic_flags()
    edges = realized_pec_edge_masks(assembled[3], sheets=tuple(sheets),
                                    wires=tuple(wires), periodic=periodic)
    for pe in sim._msl_ports:
        span = msl_cross_section_span(grid, msl_port_from_entry(pe))
        ij = tuple(span["i_feed"] if c == span["prop_idx"] else span["w_centre"]
                   for c in range(3) if c != span["normal_idx"])
        k_lo, k_hi = realized_trace_planes_on_column(
            edges, span["normal_idx"], ij, span["n_hi"], periodic=periodic)
        assert k_lo is not None, "the sheet trace was not found"
        assert k_lo == k_hi == span["n_hi"]
