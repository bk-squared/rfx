"""Study P (``validation/fdfd/invariant_ladder.py``): the level-invariant
spiral fixture and its joint GPU convergence ladder.

What runs live here (CPU, SuperLU, well under the file's budget):

I1  the new ``SpiralSpec`` options are additive: the default spec is the
    original cell-defined build (``invariant`` False) and it reproduces the
    committed ``spiral_convergence.json`` W/1 grid line for line.
I2  ``pad_to`` lands EXACTLY on the fixed wall with every neighbour ratio in
    ``[1/1.5, 1.5]`` and ``subdivide_lines`` keeps every input line bit-exactly.
I3  gate P1 from built models at levels 1, 2, 3: every wall / lid / post /
    bridge / column / port / interface coordinate read back from the cell
    masks is identical across the levels, every level-1 line is a line of
    every finer level -- and the same read-back on the OLD fixture fails.
I4  the referee (one number for every level) recomputed from scratch equals
    the JSON and the old study's de-embedded referee.
I5  the level-1 primary fixture solved live reproduces the JSON's cuDSS
    level-1 ``L_dut`` and its ``jax.grad`` (backend-to-backend agreement).

I8  gate P6's object read-back live: of the old fixture's three
    level-dependences, (a) walls and (b) short standard move a physical
    object, (c) the vertical grid refined moves none (it only adds z lines).

What is read from the JSON (the GPU ladder cannot run here): every gate
verdict and every derived block (extrapolation checks, residual, resolution)
re-derives from its recorded raw numbers (I6) and the headline numbers are
asserted as hard values (I7). Nothing needs the GPU backend; a test
that would is not written -- the GPU measurements are in the harvested lane
JSONs under ``validation/vessl/runs/fdfd-gpu-p*``.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from tests._x64_compat import enable_x64

pytest.importorskip("shapely")

REPO = pathlib.Path(__file__).resolve().parents[3]
STUDY_PATH = REPO / "validation" / "fdfd" / "invariant_ladder.py"
JSON_PATH = REPO / "validation" / "fdfd" / "invariant_ladder.json"
PNG_PATH = REPO / "validation" / "fdfd" / "invariant_ladder.png"
SC_JSON = REPO / "validation" / "fdfd" / "spiral_convergence.json"

_CACHE: dict = {}


def _il():
    if "il" not in _CACHE:
        spec = importlib.util.spec_from_file_location("invariant_ladder_test", STUDY_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["invariant_ladder_test"] = mod
        spec.loader.exec_module(mod)
        _CACHE["il"] = mod
    return _CACHE["il"]


def _study() -> dict:
    if "json" not in _CACHE:
        _CACHE["json"] = json.loads(JSON_PATH.read_text())
    return _CACHE["json"]


def test_i1_options_are_additive_and_default_is_the_old_build():
    """The default spec takes the original build path, and the old fixture
    (built through the study's ``spec_old``) reproduces the committed W/1
    grid of ``spiral_convergence.json`` exactly -- shape, unknowns, walls,
    lid, number of z lines."""
    from rfx.fdfd import spiral as sm
    assert sm.SpiralSpec().invariant is False
    assert sm.SpiralSpec(refine=2).invariant is True
    assert sm.SpiralSpec(short_gap_m=1e-6).invariant is True
    rec = json.loads(SC_JSON.read_text())["levels"]["1"]["grid"]
    with enable_x64():
        model = _il().build(_il().spec_old(1))
        assert not model.spec.invariant
        assert list(model.shape) == rec["shape"]
        assert model.n_unknowns == rec["n_unknowns"]
        assert float(model.x_nom[-1]) == rec["wall_x"][1]
        assert float(model.y_nom[0]) == rec["wall_y"][0]
        assert float(model.z[-1]) == rec["lid_z"]
        assert len(model.z) == rec["n_z_lines"]
        assert model.fixture == {}


def test_i2_pad_to_lands_on_the_wall_and_subdivision_nests():
    from rfx.fdfd import spiral as sm
    core = np.linspace(-62e-6, 62e-6, 13)             # a uniform 10.33 um core
    for wall in (100e-6, 152e-6, 452e-6, 862e-6):
        out = sm.pad_to(sm.pad_to(core, wall, 1.5, hi=True), -wall, 1.5, hi=False)
        assert out[-1] == wall and out[0] == -wall            # bit-exact, both ends
        d = np.diff(out)
        r = d[1:] / d[:-1]
        assert np.all(r <= 1.5 * (1 + 1e-12)) and np.all(r >= (1 / 1.5) * (1 - 1e-12))
        for m in (2, 3, 4, 6):
            fine = sm.subdivide_lines(out, m)
            assert len(fine) == m * (len(out) - 1) + 1
            assert np.array_equal(fine[::m], out)            # every level-1 line, bit-exact
    with pytest.raises(ValueError):
        sm.pad_to(core, core[-1] + 1e-6, 1.5)                # closer than end cell / ratio


def test_i3_p1_fixture_invariance_live():
    """Gate P1 on models built here at levels 1-3 (the JSON's P1 adds
    levels 4 and 6, 1.10 M and 3.66 M unknowns): all coordinates identical
    to 0 m (the tolerance is 1e-12 m), metal volumes to 1e-15 relative,
    nesting distance 0 m. The OLD fixture's read-back moves by 81.25 um
    (its walls) between W/1 and W/3 -- the gate discriminates."""
    il = _il()
    with enable_x64():
        models = {m: il.build(il.spec_new(m, il.WALL_W)) for m in (1, 2, 3)}
        g = il.p1_gate(models)
        old = il.p1_gate({m: il.build(il.spec_old(m)) for m in (1, 2, 3)})
    assert g["passed"]
    assert g["worst_coordinate_deviation_m"] <= 1e-12
    assert g["nesting_worst_distance_m"] <= 1e-12
    assert g["worst_volume_rel_deviation"] <= 1e-15
    c = g["coordinates"]["1"]
    assert c["post"][:2] == pytest.approx([22e-6, 32e-6], abs=1e-15)      # the physical post
    assert c["ports"]["outer"][4:] == pytest.approx([0.0, 10e-6], abs=1e-15)  # the port gap
    assert not old["passed"]
    assert old["worst_coordinate_deviation_m"] == pytest.approx(81.25e-6, rel=1e-9)
    js = _study()["p1"]
    assert js["passed"] and js["levels"] == [1, 2, 3, 4, 6]
    assert js["coordinates"]["1"] == g["coordinates"]["1"]


def test_i4_referee_is_one_number():
    il = _il()
    ref = il.referee_block()
    js = _study()["referee"]
    assert ref["total"] == js["total"]
    old = json.loads(SC_JSON.read_text())["referee"]["deembedded"]["total"]
    assert ref["total"] == old                    # the bridge split at the post centre (0.5)
    assert ref["total"] == pytest.approx(333.055e-12, abs=1e-15)
    # sensitivities recorded beside it (measured here): the M2 -> M1 transition
    # anywhere across the 10 um post 0.0382 %; the underpass one 2 um layer
    # up / down -0.521 % / +0.445 %; the omitted via at most 0.0342 %
    assert ref["post_split_band_rel"] == pytest.approx(3.8199e-4, rel=1e-3)
    assert ref["underpass_depth"]["one_layer_up"]["rel"] == pytest.approx(-5.2092e-3, rel=1e-3)
    assert ref["underpass_depth"]["one_layer_down"]["rel"] == pytest.approx(4.4542e-3, rel=1e-3)
    assert ref["via"]["bound_rel"] < 5e-4


def test_i5_level1_live_matches_the_gpu_record():
    """The primary level 1 (N = 18643) through SuperLU here against the
    cuDSS record: L_dut to 1e-7 and the gradient to 1e-6 relative (measured
    agreement 1.8e-9 and 1.5e-8 when this test was written)."""
    import jax
    import jax.numpy as jnp
    il = _il()
    rec = _study()["levels"]["1"]
    with enable_x64():
        model = il.build(il.spec_new(1, il.WALL_W))
        assert model.n_unknowns == rec["grid"]["n_unknowns"] == 18643
        val, grad = jax.value_and_grad(lambda t: jnp.real(il.l_dut(model, t)))(
            jnp.asarray(il.THETA0, dtype=jnp.float64))
    assert float(val) == pytest.approx(rec["L_dut"], rel=1e-7)
    for a, b in zip(np.asarray(grad), rec["grad"]):
        assert float(a) == pytest.approx(b, rel=1e-6)


def test_i6_json_verdicts_rederive_from_the_raw_numbers():
    """Every derived block of the JSON (family Richardson, corrections, the
    resolution estimate, the six gate verdicts) is recomputed here from the
    JSON's own raw measurements and must be identical: the verdicts cannot
    be edited without the numbers, and no failing gate can be widened."""
    il = _il()
    study = _study()
    fam_levels = {k: {"m": v["m"], "grid": v["grid"], "L_dut": v["L_dut"], "grad": v["grad"]}
                  for k, v in study["levels"].items()}
    fam = il.family(fam_levels)
    assert fam["L_dut"] == study["family"]["L_dut"]
    assert fam["richardson"] == study["family"]["richardson"]
    again = json.loads(json.dumps(il.evaluate_gates(study)))
    assert again == study["gates"]
    assert json.loads(json.dumps(il.corrections(study))) == study["corrections"]
    assert json.loads(json.dumps(il.resolution_needed(study))) == study["resolution_for_3_percent"]
    assert json.loads(json.dumps(il.extrapolation_checks(study))) == study["extrapolation_checks"]
    assert json.loads(json.dumps(il.residual_block(study))) == study["residual"]
    assert PNG_PATH.exists()
    runs = study["runs"]
    assert runs and all(r["id"] and r["state"] and r["artifacts"] for r in runs)
    for r in runs:
        assert (REPO / r["artifacts"]).exists()


def _pH(v: float) -> float:
    return v * 1e12


def test_i7_recorded_numbers_are_the_docstring_numbers():
    """The study's headline numbers, asserted at the precision its module
    docstring states them (so the docstring, the JSON and this file cannot
    drift apart). Every gate verdict is asserted as measured, failing or
    not."""
    st = _study()
    g = st["gates"]
    fam = st["family"]
    assert fam["m"][:4] == [1, 2, 3, 4]
    assert [round(_pH(v), 3) for v in fam["L_dut"][:4]] == [259.223, 298.525, 308.631, 312.675]
    assert fam["n_unknowns"][:4] == [18643, 140990, 466833, 1095964]
    # P2
    assert g["P2"]["passed"] and g["P2"]["monotone_all"] and g["P2"]["contracting"]
    assert g["P2"]["observed_order"] == pytest.approx(1.627, abs=5e-4)
    assert [round(_pH(v), 3) for v in g["P2"]["range"]] == [317.874, 324.806]
    # P3: inside 5 %, NOT inside 3 %
    assert g["P3"]["passed"] and g["P3"]["within_5_percent"] and not g["P3"]["within_3_percent"]
    assert [round(100 * v, 3) for v in g["P3"]["rel_range"]] == [-4.558, -2.477]
    assert round(_pH(g["P3"]["referee"]), 3) == 333.055
    # P4, P5
    assert g["P4"]["passed"] and g["P4"]["level"] == 4
    assert g["P4"]["rel"] == pytest.approx(5.45e-7, rel=2e-3)
    assert g["P5"]["passed"]
    assert [round(v * 1e6, 2) for v in g["P5"]["abs_changes"]] == [2.02, 1.34, 0.73]
    # P6
    p6 = g["P6"]["per_level"]
    assert g["P6"]["passed"]
    assert round(100 * p6["1"]["a_walls"], 3) == 0.099
    assert abs(p6["1"]["b_short"]) < 1e-8 and abs(p6["1"]["c_vertical"]) < 1e-8
    assert [round(100 * p6["2"][k], 3) for k in ("a_walls", "b_short", "c_vertical", "abc", "new")] \
        == [0.283, -1.043, 2.777, 1.854, 2.258]
    assert round(100 * p6["2"]["objects_ab"], 3) == -0.760
    # the same pairs read as L_old / L_case - 1 (how far the OLD L sits off)
    lv2 = st["p6"]["levels"]["2"]
    assert [round(100 * lv2[k]["old_rel_to_case"], 2) for k in ("a_walls", "b_short", "c_vertical")] \
        == [-0.28, 1.05, -2.70]
    assert g["P6"]["changes_a_physical_object_at_level_2"] == {
        "a_walls": True, "b_short": True, "c_vertical": False}
    ob = st["p6"]["objects"]["per_div"]["2"]
    assert round(ob["a_walls"]["worst_coordinate_deviation_m"] * 1e6, 2) == 129.06
    assert round(ob["b_short"]["worst_coordinate_deviation_m"] * 1e6, 2) == 5.00
    assert round(100 * ob["b_short"]["worst_volume_rel_deviation"], 1) == 29.1
    assert ob["c_vertical"]["worst_coordinate_deviation_m"] < 1e-20
    assert ob["c_vertical"]["worst_volume_rel_deviation"] < 1e-15
    assert (ob["old_z_lines"], ob["c_vertical"]["z_lines"]) == (16, 31)
    # supporting blocks
    w = st["walls"]
    assert w["chosen_margin_w"] == 10.0 and w["margins_w"] == [10.0, 20.0, 40.0, 80.0]
    assert [round(100 * v, 3) for v in w["rel_change_to_next"]] == [0.183, -0.006, 0.009]
    pl = st["sigma_plateau"]
    assert round(100 * pl["rc_twin"]["rel_to_chosen"], 3) == 0.089
    c = st["corrections"]
    assert round(100 * c["walls"]["rel"], 3) == 0.186
    assert [round(100 * v, 3) for v in c["all"]["rel_range"]] == [-4.646, -2.535]
    res = st["resolution_for_3_percent"]
    r = res["cells_across_width_needed_range"]
    assert [round(v, 2) for v in r] == [2.95, 4.98]
    # m = 4 is under 3 % at the observed order and p = 2, NOT at p = 1: the
    # first level under 3 % at every order in [1, 2] is m = 5
    po = res["per_order"]
    assert [round(100 * po[k]["error_at_finest"], 2) for k in ("p1", "observed", "p2")] \
        == [3.74, 2.12, 1.64]
    assert [po[k]["finest_within_target"] for k in ("p1", "observed", "p2")] == [False, True, True]
    assert [po[k]["first_level_within_target"] for k in ("p1", "observed", "p2")] == [5, 4, 3]
    assert res["finest_level"] == 4 and res["first_level_within_target_every_order"] == 5
    assert not res["finest_within_target_every_order"]
    # is [p = 2, p = 1] a bracket: rising observed order, pair estimates
    # closing from both sides, the two other fits inside the range
    ex = st["extrapolation_checks"]
    assert [round(t["observed_order"], 3) for t in ex["triples"].values()] == [1.447, 1.627]
    assert ex["observed_order_rising"] and ex["bracket_closes_from_both_sides"]
    assert [round(_pH(v["p1"]), 3) for v in ex["pairs"].values()] == [337.827, 328.842, 324.806]
    assert [round(_pH(v["p2"]), 3) for v in ex["pairs"].values()] == [311.626, 316.715, 317.874]
    f = ex["fits"]
    assert round(_pH(f["lsq_h_h2_all_levels"]["L_inf"]), 3) == 323.273
    assert round(100 * f["lsq_h_h2_all_levels"]["rel_to_referee"], 3) == -2.937
    assert round(_pH(f["exact_cubic_finest_four"]["L_inf"]), 3) == 319.577
    assert round(100 * f["exact_cubic_finest_four"]["rel_to_referee"], 3) == -4.047
    assert all(v["inside_p3_range"] for v in f.values())
    # the residual is NOT explained by any measured model difference
    rs = st["residual"]
    assert not rs["explained"]
    assert round(100 * rs["measured_model_differences_abs_sum"], 3) == 0.709
    assert round(100 * rs["unexplained_at_least"], 3) == 2.462
    assert round(100 * rs["finest_estimates_below_referee_by_at_least"], 3) == 2.477
    assert round(100 * rs["richardson_p_spread_rel"], 3) == 2.081
    assert round(100 * rs["referee_convention_bounds_rel"]["via"], 3) == 0.034
    assert round(100 * rs["referee_convention_bounds_rel"]["post_split_band"], 3) == 0.038
    assert round(100 * rs["signed_corrections_rel"]["rc"], 3) == 0.095
    assert round(100 * max(abs(v) for v in rs["post_short_rel_range"]), 3) == 0.356
    assert round(st["plans"]["W10_mc1_m5"]["permanent_device_memory_gb"], 2) == 91.87
    assert round(st["plans"]["W10_mc1_m6"]["permanent_device_memory_gb"], 2) == 193.78
    pa = st["paper"]
    assert [round(100 * v["gap"], 2) for v in pa["levels"].values()] == [-13.71, -6.27]
    assert round(100 * pa["plateau"]["rc_twin"]["rel_to_chosen"], 3) == 0.195


def test_i8_p6_objects_live():
    """P6's classification from models built here (old fixture, W/2 and
    W/3): the physical walls and the physical short standard each move an
    object read back from the built masks; refining the vertical grid moves
    none (tolerance 1e-12 m, volumes 1e-12) -- so P6's (c) is a
    discretisation left un-refined, not a fixture change. Matches the JSON."""
    il = _il()
    with enable_x64():
        ob = il.p6_objects()
    for div in ("2", "3"):
        r = ob["per_div"][div]
        assert r["a_walls"]["changes_a_physical_object"] and r["a_walls"]["objects_moved"] == ["walls"]
        assert r["b_short"]["changes_a_physical_object"]
        assert r["b_short"]["objects_moved"] == ["bridge_m1", "bridge_m2", "post"]
        assert not r["c_vertical"]["changes_a_physical_object"]
        assert r["c_vertical"]["objects_moved"] == []
        assert r["c_vertical"]["z_lines"] > r["old_z_lines"]
    assert json.loads(json.dumps(ob)) == _study()["p6"]["objects"]
