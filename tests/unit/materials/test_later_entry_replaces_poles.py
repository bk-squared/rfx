"""A later geometry entry replaces the cells it covers — their dispersion poles included.

Drawing a block of another material inside a Debye/Lorentz/Drude body used to replace the body's permittivity
on those cells but keep the body's poles there, so a "hole" still dispersed like the body (probe 98 % of its
peak away from the same body declared with the hole left out).
"""
import warnings

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import drude_pole, lorentz_pole

LORENTZ = lorentz_pole(3.0, 2 * np.pi * 8e9, 2 * np.pi * 8e8)
OTHER = lorentz_pole(1.0, 2 * np.pi * 5e9, 2 * np.pi * 5e8)
DEBYE = DebyePole(delta_eps=2.0, tau=2e-11)
BODY, HOLE = ((.006, 0, 0), (.018, .012, .012)), ((.010, 0, 0), (.014, .012, .012))


def _sim(kind, poles, graded=False):
    # ``graded`` declares the same 1 mm cells as a profile, which routes to the graded-mesh assembler.
    kwargs = {"dx_profile": np.full(24, 1e-3)} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.024, .012, .012), dx=.001, boundary="pec", **kwargs)
    sim.add_material("body", eps_r=2.0, **poles)
    sim.add_material("air", eps_r=1.0)
    if kind == "drawn":
        sim.add(Box(*BODY), material="body")
        sim.add(Box(*HOLE), material="air")
    else:
        sim.add(Box(BODY[0], (HOLE[0][0], .012, .012)), material="body")
        sim.add(Box((HOLE[1][0], 0, 0), BODY[1]), material="body")
    sim.add_source((.003, .006, .006), "ez", waveform=GaussianPulse(f0=8e9, bandwidth=.8))
    sim.add_probe((.012, .006, .006), "ez")
    sim.add_probe((.021, .005, .006), "ez")
    return sim


def _assembled(sim, graded=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if graded:
            return sim._assemble_materials_nu(sim._build_nonuniform_grid(), pec_sheets=[], pec_wires=[])
        return sim._assemble_materials(sim._build_grid(), pec_sheets=[], pec_wires=[])


@pytest.mark.parametrize("poles,index", [({"lorentz_poles": [LORENTZ]}, 2), ({"debye_poles": [DEBYE]}, 1)],
                         ids=["lorentz", "debye"])
def test_a_block_drawn_into_a_dispersive_body_removes_its_poles_there(poles, index):
    drawn, declared = _assembled(_sim("drawn", poles)), _assembled(_sim("declared", poles))
    mask_drawn, mask_declared = (np.asarray(r[index][1][0]) for r in (drawn, declared))
    hole = slice(10, 14)                         # x cells 10..13 mm, written by hand
    assert not mask_drawn[hole].any()
    assert mask_drawn[6:10, :12, :12].all() and mask_drawn[14:18, :12, :12].all()
    assert np.array_equal(mask_drawn, mask_declared)
    assert np.array_equal(np.asarray(drawn[0].eps_r), np.asarray(declared[0].eps_r))


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("poles", [{"lorentz_poles": [LORENTZ]},
                                   {"lorentz_poles": [drude_pole(2 * np.pi * 20e9, 2 * np.pi * 2e9)]},
                                   {"debye_poles": [DEBYE]}], ids=["lorentz", "drude", "debye"])
def test_a_drawn_hole_is_solved_like_the_declared_one(poles, graded):
    # Probes: inside the hole and behind the body. With the body's poles left in the hole the two
    # runs differed by 48 % to 98 % of the peak (Lorentz, Drude, Debye; uniform and graded mesh).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = (np.asarray(_sim(k, poles, graded).run(n_steps=600, compute_s_params=False,
                                                      skip_preflight=True).time_series)
                for k in ("drawn", "declared"))
    assert np.max(np.abs(b)) > 0
    assert np.array_equal(a, b)


def test_a_later_dispersive_body_brings_only_its_own_poles():
    sim = Simulation(freq_max=10e9, domain=(.024, .012, .012), dx=.001, boundary="pec")
    sim.add_material("first", eps_r=2.0, lorentz_poles=[LORENTZ])
    sim.add_material("second", eps_r=3.0, lorentz_poles=[OTHER])
    sim.add(Box(*BODY), material="first")
    sim.add(Box(*HOLE), material="second")
    poles, masks = _assembled(sim)[2]
    by_pole = {p: np.asarray(m) for p, m in zip(poles, masks)}
    assert not by_pole[LORENTZ][10:14].any() and by_pole[LORENTZ][6:10, :12, :12].all()
    assert by_pole[OTHER][10:14, :12, :12].all() and not by_pole[OTHER][6:10].any()


def test_the_same_material_drawn_twice_keeps_its_pole_once():
    sim = Simulation(freq_max=10e9, domain=(.024, .012, .012), dx=.001, boundary="pec")
    sim.add_material("body", eps_r=2.0, lorentz_poles=[LORENTZ])
    sim.add(Box(*BODY), material="body")
    sim.add(Box(*HOLE), material="body")
    poles, masks = _assembled(sim)[2]
    assert len(poles) == 1
    assert np.asarray(masks[0])[6:18, :12, :12].all()


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("metal", [((.006, .002, .006), (.018, .010, .006)),      # a sheet: no thickness
                                   ((.010, .003, .003), (.014, .008, .009))],     # a block
                         ids=["sheet", "block"])
def test_metal_drawn_into_a_dispersive_body_leaves_its_poles(metal, graded):
    # PEC writes no permittivity into the cells, so it removes no poles either: a ground plane or
    # a patch on a dispersive substrate must not strip the dispersion from the cells beside it.
    def masks(with_metal):
        sim = _sim("drawn", {"debye_poles": [DEBYE], "lorentz_poles": [LORENTZ]}, graded)
        sim._geometry = sim._geometry[:1]                    # the body alone
        if with_metal:
            sim.add(Box(*metal), material="pec")
        result = _assembled(sim, graded)
        return np.asarray(result[1][1][0]), np.asarray(result[2][1][0])
    for with_metal, alone in zip(masks(True), masks(False)):
        assert alone.any()
        assert np.array_equal(with_metal, alone)


def _chi3_sim(kind):
    sim = Simulation(freq_max=10e9, domain=(.024, .012, .012), dx=.001, boundary="pec")
    sim.add_material("body", eps_r=2.0, chi3=1e-10)
    sim.add_material("air", eps_r=1.0)
    if kind == "drawn":
        sim.add(Box(*BODY), material="body")
        sim.add(Box(*HOLE), material="air")
    else:
        sim.add(Box(BODY[0], (HOLE[0][0], .012, .012)), material="body")
        sim.add(Box((HOLE[1][0], 0, 0), BODY[1]), material="body")
    return sim


def test_a_block_drawn_into_a_kerr_body_removes_its_nonlinearity_there():
    drawn, declared = (np.asarray(_assembled(_chi3_sim(k))[6]) for k in ("drawn", "declared"))
    assert not drawn[10:14].any()                # the hole, x cells 10..13 mm
    assert (drawn[6:10, :12, :12] == np.float32(1e-10)).all()
    assert np.array_equal(drawn, declared)


def test_the_shared_rasterizer_follows_the_same_rule():
    # ``rasterize_geometry`` (fine-grid and interface-permittivity sampling) keeps the declaration
    # rule of the cell assembler: poles and the Kerr coefficient leave the cells a later entry writes.
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid, rasterize_geometry
    sim = Simulation(freq_max=10e9, domain=(.024, .012, .012), dx=.001, boundary="pec")
    sim.add_material("body", eps_r=2.0, chi3=1e-10, debye_poles=[DEBYE], lorentz_poles=[LORENTZ])
    sim.add_material("air", eps_r=1.0)
    sim.add(Box(*BODY), material="body")
    sim.add(Box(*HOLE), material="air")
    _, debye, lorentz, _, _, chi3 = rasterize_geometry(
        sim._geometry, sim._resolve_material, coords_from_uniform_grid(sim._build_grid()))
    for cells in (np.asarray(debye[1][0]), np.asarray(lorentz[1][0]), np.asarray(chi3) != 0):
        assert not cells[10:14].any()
        assert cells[6:10, :12, :12].all() and cells[14:18, :12, :12].all()


@pytest.mark.parametrize("metal", [((.006, .002, .006), (.018, .010, .006)),
                                   ((.010, .003, .003), (.014, .008, .009))], ids=["sheet", "block"])
def test_metal_drawn_into_a_kerr_body_leaves_its_nonlinearity(metal):
    def chi3(with_metal):
        sim = _chi3_sim("drawn")
        sim._geometry = sim._geometry[:1]                    # the body alone
        if with_metal:
            sim.add(Box(*metal), material="pec")
        return np.asarray(_assembled(sim)[6])
    assert chi3(False).any()
    assert np.array_equal(chi3(True), chi3(False))
