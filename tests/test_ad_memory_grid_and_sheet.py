"""Issue #696 — estimate against the grid the solve runs, count the sheet.

Two independent under/over-statements, both of which put a real run on the
wrong side of a GPU budget:

1. The estimate re-derived the grid shape itself
   (``ceil(extent/dx) + 1 + 2*cpml_layers`` per axis) instead of reading
   the grid the solve will build. That re-derivation models neither real
   grid wherever the shape depends on something it does not know about:
   per-face pads (a ``pec``/``pmc``/``periodic`` face allocates 0 cells on
   that side) and 2-D mode (``nz == 1``). ``SimConfig.grid_shape``, the
   planner-side twin, ignored ``dz_profile`` entirely and so described the
   UNIFORM grid for a graded-z stackup — the "handed the uniform grid
   while the solve steps the NU grid" shape of the report, and it under-
   states, which is the dangerous direction.
2. The ``surface_impedance_f0`` sheet operator's three boolean edge masks
   plus ``sigma_sheet`` were counted NOWHERE. Since #677 the sheet is in
   neither ``pec_mask`` nor ``materials.sigma``, so a lossy board
   estimated identically to the same board WITHOUT loss.

Measured (dx=0.2mm, cpml_layers=8, 20x20mm):

  case                            legacy shape -> real shape        ratio
  PEC y faces, graded dz          0.780M -> 0.674M cells            1.16x high
  mode="2d_tmz"                   1.602M -> 0.014M cells             117x high
  SimConfig, 200-cell z stackup   0.370M -> 2.971M cells            8.04x LOW
  f0 sheet on a graded-dz board   forward_gb 0.0487 -> 0.0542       11.3% low

The NU fixtures grade z with a ratio unlike anything on x/y: a
uniform-valued profile takes the NU path without exercising an NU metric.
"""

import math
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.auto_config import SimConfig
from rfx.boundaries.spec import Boundary, BoundarySpec

DX = 0.2e-3
CPML = 8
DZ = 0.05e-3 * 1.06 ** np.arange(40, dtype=float)   # ratio 1.06, z only


def _legacy_shape(sim):
    """The pre-#696 private re-derivation, verbatim."""
    dx = sim._dx or (2.998e8 / sim._freq_max / 20.0)

    def _nx(extent, prof):
        if prof is not None:
            return len(prof) + 1 + 2 * sim._cpml_layers
        return int(math.ceil(extent / dx)) + 1 + 2 * sim._cpml_layers

    return (_nx(sim._domain[0], sim._dx_profile),
            _nx(sim._domain[1], sim._dy_profile),
            _nx(sim._domain[2], sim._dz_profile))


def _acc_shape(sim):
    a = sim._ad_memory_static_accounting()
    return (a["nx"], a["ny"], a["nz"])


def test_accounting_matches_the_nonuniform_grid_the_solve_builds():
    sim = Simulation(freq_max=20e9, domain=(0.02, 0.02, float(DZ.sum())),
                     dx=DX, boundary="cpml", cpml_layers=CPML, dz_profile=DZ)
    assert float(DZ.max() / DZ.min()) > 5.0, "fixture dz is not graded"
    nu = tuple(int(v) for v in sim._build_nonuniform_grid().shape)
    uni = tuple(int(v) for v in sim._build_grid().shape)
    assert nu != uni, "fixture does not separate the two grids"
    assert _acc_shape(sim) == nu
    est = sim.estimate_ad_memory(1000)
    assert est.grid_kind == "nonuniform"
    assert est.grid_source == "built"
    assert tuple(est.grid_shape) == nu


def test_per_face_pads_are_no_longer_over_counted():
    """PEC y faces allocate 0 cells there; the re-derivation added 2*8."""
    spec = BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                        y=Boundary(lo="pec", hi="pec"),
                        z=Boundary(lo="cpml", hi="cpml"))
    sim = Simulation(freq_max=20e9, domain=(0.02, 0.02, float(DZ.sum())),
                     dx=DX, boundary=spec, cpml_layers=CPML, dz_profile=DZ)
    real = tuple(int(v) for v in sim._build_nonuniform_grid().shape)
    legacy = _legacy_shape(sim)
    assert _acc_shape(sim) == real
    assert legacy != real
    legacy_cells = legacy[0] * legacy[1] * legacy[2]
    real_cells = real[0] * real[1] * real[2]
    assert legacy_cells == 780_273
    assert real_cells == 673_569
    assert 1.15 < legacy_cells / real_cells < 1.17


def test_two_dimensional_mode_is_not_counted_as_a_cube():
    sim = Simulation(freq_max=20e9, domain=(0.02, 0.02, 0.02), dx=DX,
                     boundary="cpml", cpml_layers=CPML, mode="2d_tmz")
    real = tuple(int(v) for v in sim._build_grid().shape)
    assert real[2] == 1
    assert _acc_shape(sim) == real
    legacy = _legacy_shape(sim)
    assert legacy[2] == 117
    assert (legacy[0] * legacy[1] * legacy[2]) / (real[0] * real[1] * real[2]) \
        == pytest.approx(117.0)


def test_sim_config_grid_shape_counts_the_z_profile():
    """The planner-side twin described the UNIFORM grid — and under-stated
    by 8x on a refined stackup, which is the direction that kills a run."""
    dz = np.full(200, 1e-5)
    kw = dict(dx=DX, domain=(0.02, 0.02, float(dz.sum())), cpml_layers=CPML,
              n_steps=1000, freq_range=(1e9, 20e9), margin=0.0, dt=1e-13,
              accuracy="standard")
    with_prof = SimConfig(dz_profile=dz, **kw)
    without = SimConfig(**kw)
    assert with_prof.grid_shape == (117, 117, 217)
    assert without.grid_shape == (117, 117, 27)
    ratio = (np.prod(with_prof.grid_shape) / np.prod(without.grid_shape))
    assert ratio == pytest.approx(8.037, rel=1e-3)
    assert with_prof.estimated_memory_mb > 6.9 * without.estimated_memory_mb


def _sheet_sim():
    sim = Simulation(freq_max=20e9, domain=(0.02, 0.02, float(DZ.sum())),
                     dx=DX, boundary="cpml", cpml_layers=CPML, dz_profile=DZ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((0.002, 0.002, 0.002), (0.018, 0.018, 0.002)),
            sigma_bulk=5.8e7, thickness=35e-6, surface_impedance_f0=20e9)
    return sim


def test_f0_sheet_operator_is_counted():
    plain = Simulation(freq_max=20e9, domain=(0.02, 0.02, float(DZ.sum())),
                       dx=DX, boundary="cpml", cpml_layers=CPML,
                       dz_profile=DZ)
    sheet = _sheet_sim()
    assert _acc_shape(plain) == _acc_shape(sheet), "same grid, only loss added"

    a_plain = plain._ad_memory_static_accounting()
    a_sheet = sheet._ad_memory_static_accounting()
    assert a_plain["sheet_bytes"] == 0
    # 3 boolean edge masks (1 byte) + sigma_sheet (float32)
    assert a_sheet["sheet_bytes"] == a_sheet["cells"] * 7
    assert a_sheet["forward_bytes"] > a_plain["forward_bytes"]

    e_plain = plain.estimate_ad_memory(1000)
    e_sheet = sheet.estimate_ad_memory(1000)
    assert e_plain.sheet_gb == 0.0
    assert e_sheet.sheet_gb > 0.0
    assert e_sheet.forward_gb == pytest.approx(0.054151, rel=1e-4)
    assert e_plain.forward_gb == pytest.approx(0.048689, rel=1e-4)
    assert e_sheet.forward_gb / e_plain.forward_gb == pytest.approx(1.113,
                                                                   rel=1e-2)


def test_pec_sheet_is_not_charged_for_the_f0_operator():
    """Only f0 sheets build the operator; a PEC sheet is in pec_mask."""
    sim = Simulation(freq_max=20e9, domain=(0.02, 0.02, float(DZ.sum())),
                     dx=DX, boundary="cpml", cpml_layers=CPML, dz_profile=DZ)
    sim.add_thin_conductor(Box((0.002, 0.002, 0.002), (0.018, 0.018, 0.002)),
                           sigma_bulk=5.8e7, thickness=35e-6)
    assert sim._ad_memory_static_accounting()["sheet_bytes"] == 0


def test_report_and_estimate_describe_the_same_grid():
    sim = _sheet_sim()
    report = sim.mesh_intelligence_report(n_steps=1000)
    assert tuple(report.grid_shape) == _acc_shape(sim)
    assert tuple(report.grid_shape) == tuple(report.ad_memory.grid_shape)
    assert report.ad_memory.grid_kind == "nonuniform"


def test_explain_lists_the_sheet_component():
    sim = _sheet_sim()
    explanation = sim.explain_ad_memory(n_steps=1000)
    comps = {c.name: c for c in explanation.components}
    assert "surface_impedance_sheet_state" in comps
    assert comps["surface_impedance_sheet_state"].memory_gb > 0.0


# ---------------------------------------------------------------------------
# The grid-build FAILURE branch and its label.
#
# ``_ad_memory_static_accounting`` wraps the grid build in ``except
# Exception`` so a planning helper never raises, and labels the fallback
# ``grid_source="estimated_from_domain"`` so the artifact does not present
# the pre-#696 arithmetic as the real grid. Every test above takes the
# success branch, so mutating that literal to ``"built"`` — collapsing the
# two labels into one and making the honesty field say nothing — changed no
# result anywhere. Both tests below fail under that mutation, and they also
# pin WHAT the fallback computes, not only what it is called.
# ---------------------------------------------------------------------------

def test_unbuildable_profile_still_estimates_and_says_it_estimated():
    """A user-reachable trigger: a profile the grid builder rejects.

    ``dx_profile[0]`` must equal the boundary ``dx`` (the CPML cells use
    the boundary spacing), so this profile is accepted by the constructor
    and rejected by ``make_nonuniform_grid``. Planning must survive it —
    with the number labelled as the estimate it is.
    """
    bad = np.full(20, DX)
    bad[0] = DX * 3.0
    sim = Simulation(freq_max=20e9, domain=(4e-3, 4e-3, 3e-3), dx=DX,
                     boundary="cpml", cpml_layers=CPML, dx_profile=bad)
    with pytest.raises(ValueError, match="must equal boundary"):
        sim._build_nonuniform_grid()

    acc = sim._ad_memory_static_accounting()
    assert acc["grid_source"] == "estimated_from_domain"
    assert acc["grid_kind"] == "nonuniform"
    assert (acc["nx"], acc["ny"], acc["nz"]) == _legacy_shape(sim)
    est = sim.estimate_ad_memory(1000)
    assert est.grid_source == "estimated_from_domain"
    assert est.to_dict()["grid_source"] == "estimated_from_domain"
    assert est.forward_gb > 0.0


def test_fallback_label_and_number_both_differ_from_the_built_grid():
    """One fixture, both branches — the label is not the only difference.

    PEC y faces allocate no pad on y, which the pre-#696 arithmetic does
    not model. So on THIS sim the fallback is measurably wrong (ratio
    pinned below) as well as differently labelled: a reader who trusts
    ``grid_source`` learns something real, and a mutation that makes both
    branches say ``"built"`` hands them a 1.76x over-estimate wearing the
    label of a measured one.
    """
    spec = BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                        y=Boundary(lo="pec", hi="pec"),
                        z=Boundary(lo="cpml", hi="cpml"))
    sim = Simulation(freq_max=20e9, domain=(4e-3, 4e-3, float(DZ.sum())),
                     dx=DX, boundary=spec, cpml_layers=CPML, dz_profile=DZ)

    built = sim._ad_memory_static_accounting()
    assert built["grid_source"] == "built"
    real = tuple(int(v) for v in sim._build_nonuniform_grid().shape)
    assert (built["nx"], built["ny"], built["nz"]) == real

    def _boom(*_a, **_k):
        raise RuntimeError("forced grid-build failure")

    sim._build_nonuniform_grid = _boom
    fell_back = sim._ad_memory_static_accounting()

    assert fell_back["grid_source"] == "estimated_from_domain"
    assert fell_back["grid_source"] != built["grid_source"], (
        "the fallback must not wear the same label as a measured grid")
    assert (fell_back["nx"], fell_back["ny"],
            fell_back["nz"]) == _legacy_shape(sim)
    assert fell_back["cells"] > built["cells"]
    # Hand arithmetic (dx=0.2mm, 4mm y extent, cpml_layers=8): the real
    # grid has ny = 20 + 1 + 0 = 21 (PEC faces take no pad); the pre-#696
    # arithmetic adds 2*8 unconditionally for ny = 37. x and z agree, so
    # the whole over-count is 37/21.
    assert built["ny"] == 21 and fell_back["ny"] == 37
    assert fell_back["nx"] == built["nx"] and fell_back["nz"] == built["nz"]
    ratio = fell_back["cells"] / built["cells"]
    assert abs(ratio - 37 / 21) < 1e-9, (
        f"pinned pre-#696 over-count on PEC y faces drifted: {ratio:.4f}")
    # The byte totals ride on the cell count, so the honesty label is
    # carrying a real difference in the number a user sizes a GPU with.
    assert fell_back["forward_bytes"] > built["forward_bytes"]


# ---------------------------------------------------------------------------
# ``auto_configure(max_memory_mb=...)`` — the DECISION half must use the same
# grid and the same byte rule as the number it reports.
#
# ``SimConfig.grid_shape`` (fixed above) is the reporting half. The coarsening
# loop that PICKS dx had its own hand-written copy of both rules: it counted z
# as ``ceil(domain_z/dx)`` and its byte formula omitted the NTFF and
# dispersion terms the property adds. So the loop stopped at a number
# ``estimated_memory_mb`` never reproduced, and the documented postcondition
# — "dx is automatically coarsened until estimated_memory_mb <=
# max_memory_mb" — failed on the non-uniform-z boards this whole issue is
# about. Fixing only the reporting half made that disagreement WIDER, which
# is how this test came to exist. Both halves now go through
# ``_auto_grid_shape`` / ``_auto_memory_mb``.
# ---------------------------------------------------------------------------

def _fr4_board():
    """A board that forces a non-uniform z: 0.508 mm substrate, 35 um trace."""
    h = 0.508e-3
    return [
        (Box((0, 0, 0), (0.040, 0.030, h)), "fr4"),
        (Box((0.005, 0.014, h), (0.035, 0.016, h + 35e-6)), "pec"),
    ]


@pytest.mark.parametrize("budget_mb", [200.0, 500.0, 2000.0])
def test_auto_configure_budget_holds_on_a_nonuniform_z_board(budget_mb):
    """The postcondition, on the case that used to break it.

    Measured on this fixture before the decision half was moved onto the
    same grid: 200 MB budget -> 399.8 MB reported, 500 MB -> 653.1 MB.
    (On main, which had neither half fixed: 240.6 MB and 436.2 MB — so the
    200 MB case was already broken and fixing one half broke the 500 MB
    case too.)
    """
    from rfx.auto_config import auto_configure

    cfg = auto_configure(_fr4_board(), (1e9, 10e9), max_memory_mb=budget_mb)
    assert cfg.dz_profile is not None, "fixture must need a non-uniform z"
    assert not any("Could not fit" in w for w in cfg.warnings), cfg.warnings
    assert cfg.estimated_memory_mb <= budget_mb, (
        f"auto_configure accepted dx={cfg.dx*1e3:.4f} mm whose own "
        f"SimConfig reports {cfg.estimated_memory_mb:.1f} MB against a "
        f"{budget_mb:.0f} MB budget")
    # And the z count it was decided from is the profile's, not the
    # uniform re-derivation's.
    assert cfg.grid_shape[2] == len(cfg.dz_profile) + 1 + 2 * cfg.cpml_layers
    assert cfg.grid_shape[2] != int(math.ceil(cfg.domain[2] / cfg.dx)) \
        + 1 + 2 * cfg.cpml_layers


def test_auto_configure_budget_decision_and_report_agree_exactly():
    """The loop's accept test and the reported number are ONE computation.

    Tighter than the postcondition above: re-running the loop's own
    accept test on the returned config must reproduce the reported MB
    bit-for-bit. Two hand-written copies of the byte rule cannot do that.
    """
    from rfx.auto_config import (
        _auto_grid_shape,
        _auto_memory_mb,
        auto_configure,
    )

    cfg = auto_configure(_fr4_board(), (1e9, 10e9), max_memory_mb=500.0)
    shape = _auto_grid_shape(cfg.domain, cfg.dx, cfg.cpml_layers,
                             cfg.dz_profile)
    assert shape == cfg.grid_shape
    assert _auto_memory_mb(int(np.prod(shape)), cfg.cpml_layers) == \
        cfg.estimated_memory_mb


def test_auto_configure_budget_still_coarsens_a_uniform_board():
    """The uniform lane is unchanged in behaviour, only in plumbing."""
    from rfx.auto_config import auto_configure

    geom = [(Box((0, 0, 0), (0.1, 0.1, 0.05)), "dielectric")]
    loose = auto_configure(geom, (1e9, 3e9))
    assert loose.dz_profile is None, "fixture must stay on the uniform lane"
    budget = loose.estimated_memory_mb * 0.3
    tight = auto_configure(geom, (1e9, 3e9), max_memory_mb=budget)
    assert tight.dx > loose.dx
    assert tight.estimated_memory_mb <= budget
