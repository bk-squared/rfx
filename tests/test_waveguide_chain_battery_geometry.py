"""Guard (iii) of v1.8 plan decision 6 for the WR-90 chain battery: every
ladder rung realizes ONE guide and ONE DUT.

Pre-declaration: ``docs/design_notes/waveguide_chain_battery_predeclaration.md``.
Builder: ``tests/_waveguide_chain_battery_fixture.py``. Nothing here runs an
FDTD step; the test reads the grid, the production rasterizer
(``Box.mask``), preflight's transverse-span reader and ``fidelity_report``.

Why it exists: a ladder whose rungs do not divide ``a`` realizes a different
guide at every rung (the #703 class — 40 mm at dx ∈ {3, 2, 1.5} mm is
13.33 / 20 / 26.67 cells), and a DUT whose faces do not land on nodes
realizes a different obstacle at every rung, one level down. Either makes a
"mesh convergence" ladder compare three fixtures under one name. This test
pins, per rung:

* guide cell counts ``a/dx`` = 9 / 18 / 36 and ``b/dx`` = 4 / 8 / 16 exactly,
  with the realized aperture and wall-to-wall guide equal to 22.86 x 10.16 mm;
* the numerical TE10 cutoff (the quantity the absorber rule consumes) and the
  derived CPML layer count 17 / 34 / 68 — a constant 43.18 mm of absorber;
* the rasterized PEC-short and slab cell counts, per axis, scale exactly
  with ``1/dx`` (72 → 576 → 4608 and 144 → 1152 → 9216), and
  ``fidelity_report`` agrees with the rasterizer mask and raises no
  extent / absent finding;
* the port source, reference and probe planes and the θ window land on the
  pre-declared absolute coordinates at every rung.

Fast lane; measured well under the 20 s budget on CPU (the whole module,
nine builds, was 6 s at authoring).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from tests import _waveguide_chain_battery_fixture as F
from rfx.fidelity import fidelity_report


C0 = 299_792_458.0
FC_TE10_ANALYTIC_HZ = C0 / (2.0 * F.A_M)          # 6.557140 GHz
CPML_LAYERS_EXPECTED = {0.00254: 17, 0.00127: 34, 0.000635: 68}
CPML_THICKNESS_M = 17 * 0.00254                  # 0.04318 at every rung
GUIDE_CELLS_COARSE = (9, 4)                       # (a/dx, b/dx) at 2.54 mm
DUT_RUNS_COARSE = {                               # (nx, ny, nz) at 2.54 mm
    "pec_short": (2, 9, 4),
    "slab": (4, 9, 4),
}
DUT_MATERIAL = {"pec_short": "pec_like", "slab": "diel"}


def _scale(dx: float) -> int:
    s = F.DX_COARSE / dx
    assert abs(s - round(s)) < 1e-12, dx
    return int(round(s))


@pytest.fixture(scope="module")
def sims():
    """One build per (dut, dx); the absorber is derived inside the builder."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dx in F.DX_LADDER:
            for dut in F.DUTS:
                out[(dut, dx)] = F.build_simulation(dut, dx)
    return out


@pytest.mark.parametrize("dx", F.DX_LADDER)
def test_rung_realizes_the_same_guide(sims, dx):
    s = _scale(dx)
    sim = sims[("thru", dx)]
    ny, nz = F.realized_guide_nodes(sim)
    assert (ny - 1, nz - 1) == (GUIDE_CELLS_COARSE[0] * s, GUIDE_CELLS_COARSE[1] * s), (
        f"dx={dx}: guide realizes {ny - 1} x {nz - 1} cells, expected "
        f"{GUIDE_CELLS_COARSE[0] * s} x {GUIDE_CELLS_COARSE[1] * s} — this rung is a "
        "different guide (the #703 class)"
    )
    for port_index in (0, 1):
        sp = F.transverse_spans(sim, port_index)
        assert sp.a_aperture_m == pytest.approx(F.A_M, abs=1e-12)
        assert sp.b_aperture_m == pytest.approx(F.B_M, abs=1e-12)
        assert sp.a_guide_m == pytest.approx(F.A_M, abs=1e-12)
        assert sp.b_guide_m == pytest.approx(F.B_M, abs=1e-12)
        assert sp.guide_source == ("domain_faces", "domain_faces"), sp


@pytest.mark.parametrize("dx", F.DX_LADDER)
def test_numerical_cutoff_and_derived_absorber(sims, dx):
    """The absorber rule consumes the NUMERICAL TE10 cutoff of the realized
    guide. On a commensurate grid that cutoff equals c/2a; the rule then
    gives 17 / 34 / 68 layers, i.e. the same 43.18 mm at every rung."""
    sim = sims[("thru", dx)]
    fc = F.numerical_te10_cutoff_hz(sim)
    assert fc == pytest.approx(FC_TE10_ANALYTIC_HZ, rel=1e-12), (
        f"dx={dx}: numerical fc_TE10={fc / 1e9:.6f} GHz != c/2a on a grid "
        "that divides a — the realized guide is not 22.86 mm wide"
    )
    layers = F.cpml_layers_for(dx, fc)
    assert layers == CPML_LAYERS_EXPECTED[dx], (dx, layers)
    assert sim._cpml_layers == layers
    assert layers * dx == pytest.approx(CPML_THICKNESS_M, abs=1e-12)
    grid = sim._build_grid()
    assert grid.face_pads[0] == layers and grid.face_pads[1] == layers
    assert grid.face_pads[2:] == (0, 0, 0, 0), "PEC transverse faces must carry no pad"


@pytest.mark.parametrize("dut", ("pec_short", "slab"))
@pytest.mark.parametrize("dx", F.DX_LADDER)
def test_dut_cell_counts_scale_exactly_with_inverse_dx(sims, dut, dx):
    s = _scale(dx)
    sim = sims[(dut, dx)]
    mat = DUT_MATERIAL[dut]
    masks = F.dut_masks(sim)
    assert set(masks) == {mat}
    runs = F.axis_run_lengths(masks[mat])
    expected = tuple(r * s for r in DUT_RUNS_COARSE[dut])
    assert runs == expected, (
        f"{dut} at dx={dx}: rasterized (nx, ny, nz)={runs}, expected {expected} — "
        "the DUT does not scale with 1/dx (a face left the node lattice)"
    )
    assert int(masks[mat].sum()) == math.prod(expected)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = fidelity_report(sim, print_report=False)
    rows = [r for r in rep if r.get("entity", "").startswith("geometry[")]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["n_cells"] == int(masks[mat].sum()), row
    kinds = {f["kind"] for f in row["findings"]}
    # pec_like (sigma=1e10) is realized as PEC on purpose — that finding is the
    # construction, not a defect. Anything else on the DUT row is a defect.
    allowed = {"declared-lossy-realized-pec"} if dut == "pec_short" else set()
    assert kinds <= allowed, f"{dut} at dx={dx}: fidelity findings {kinds}"
    # The domain rows must not report a cell size that fails to divide a length.
    domain_rows = [r for r in rep if r.get("entity", "").startswith("domain")]
    assert len(domain_rows) == 1, [r.get("entity") for r in rep]
    for r in domain_rows:
        findings = list(r.get("findings", []))
        for ax in r.get("axes", []):
            findings.extend(ax.get("findings", []))
        for f in findings:
            assert "does not divide" not in f.get("detail", ""), (dx, f)


@pytest.mark.parametrize("dx", F.DX_LADDER)
def test_port_planes_and_theta_window_sit_on_declared_coordinates(sims, dx):
    s = _scale(dx)
    sim = sims[("slab", dx)]
    planes = F.snapped_planes(sim)
    assert planes["left"].source_m == pytest.approx(F.PORT_LEFT_X_M, abs=1e-12)
    assert planes["right"].source_m == pytest.approx(F.PORT_RIGHT_X_M, abs=1e-12)
    assert planes["left"].reference_m == pytest.approx(F.REF_LEFT_DEFAULT_M, abs=1e-12)
    assert planes["right"].reference_m == pytest.approx(F.REF_RIGHT_DEFAULT_M, abs=1e-12)
    assert planes["left"].probe_m == pytest.approx(F.PROBE_LEFT_M, abs=1e-12)
    assert planes["right"].probe_m == pytest.approx(F.PROBE_RIGHT_M, abs=1e-12)
    # Probe and reference planes are D_PROBE / D_REF inward at every rung, so
    # the cell offsets scale with 1/dx (10 / 20 / 40 and 3 / 6 / 12).
    for e in sim._waveguide_ports:
        assert e.probe_offset == 10 * s and e.ref_offset == 3 * s, (dx, e.name)

    for dut in ("slab", "pec_short"):
        sim_d = sims[(dut, dx)]
        i_lo, i_hi = F.design_region_index_range(sim_d, dut)
        assert i_hi - i_lo == 4 * s, (dut, dx, i_lo, i_hi)
        grid = sim_d._build_grid()
        lo_m = (i_lo - grid.axis_pads[0]) * dx
        assert lo_m == pytest.approx(F.design_region_x_m(dut)[0], abs=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ov = np.asarray(F.design_override(sim_d, dut, 1.0))
            base = np.asarray(sim_d._assemble_materials(grid)[0].eps_r)
        assert ov.shape == tuple(grid.shape)
        diff = ov - base
        assert np.all(diff[i_lo:i_hi] == 1.0) and np.all(diff[:i_lo] == 0.0) and np.all(diff[i_hi:] == 0.0)
        if dut == "slab":
            # The base must carry the slab itself: eps_r=4 on the slab's own cells.
            assert int((base == F.SLAB_EPS_R).sum()) == 144 * s ** 3

    # The shifted plane pair and the DUT faces are multiples of the coarse cell.
    for x in (F.REF_LEFT_SHIFTED_M, F.REF_RIGHT_SHIFTED_M, *F.PEC_SHORT_X_M, *F.SLAB_X_M,
              *F.PEC_SHORT_WINDOW_X_M):
        k = x / F.DX_COARSE
        assert abs(k - round(k)) < 1e-9, x
