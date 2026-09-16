"""The slab family's E2 windows are derived per arm, and nothing borrows cv04.

Pre-declared in
``docs/design_notes/slab_family_per_arm_lattice_window_predeclaration.md``
(issue #928 item 2). cv22 and cv23 used to size the per-bin window their ``G1``
gates enforce from cv04's per-bin maximum ``|R+T-1|`` -- an energy-leak
quantity, standing in for lattice dispersion, which conserves energy, measured
on a different board. This file is the guarantee that it no longer does.

Three properties, each falsifiable:

1. **Re-derived from OUTSIDE.** Every window the consumers' evaluators enforce
   is recomputed here from ``dispersive_eps`` and ``lattice_witness``
   directly -- the two leaves the pre-declaration's section 2 names -- without
   calling the module under test, and compared against what the evaluator
   actually used. A derivation that drifted from its declaration reds here
   rather than passing because both sides changed together.
2. **cv04 cannot reach an E2 window.** Doubling the cv04-adopted scalars moves
   the Meep-leg windows and leaves every E2 window bit-identical. The paired
   half -- that the mutation is live at all -- is asserted too, because "the
   number did not move" proves nothing unless the mutation could have moved it.
3. **Every committed record keeps its verdict** where the pre-declaration said
   it would: 12 declared-arm records pass, 13 wrong-model falsifier records
   fail, and the four Meep-leg falsifiers keep passing E2 and failing E4.

No number in this file is written down. Every expected value is recomputed
from the committed record or from the shared gate policy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_CROSSVAL = _REPO / "validation" / "crossval"
_COMPARATORS = _CROSSVAL / "comparators"
for _p in (str(_COMPARATORS), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dispersive_eps as de  # noqa: E402
import lattice_witness as LW  # noqa: E402
import slab_family as SF  # noqa: E402
import slab_arm_windows as SAW  # noqa: E402
from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _load("cv22_gates_for_window_contract",
          "validation/crossval/comparators/cv22_dispersive_gates.py")
L = _load("cv23_gates_for_window_contract",
          "validation/crossval/comparators/cv23_lossy_gates.py")

_CV22_DIR = _CROSSVAL / "_22_dispersive_results"
_CV23_DIR = _CROSSVAL / "_23_lossy_results"


def _records():
    """(case, relative path, arm name, arm block) for every committed record."""
    out = []
    for case, directory in (("cv22", _CV22_DIR), ("cv23", _CV23_DIR)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("rfx*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            falsifier = doc.get("falsifier")
            for arm, block in doc["arms"].items():
                out.append({"case": case, "rel": str(path.relative_to(_REPO)),
                            "arm": arm, "doc": block, "falsifier": falsifier,
                            # A Meep-leg falsifier runs the DECLARED rfx arm; only
                            # its Meep leg is defective, so its E2 side is a
                            # declared record.
                            "declared": falsifier is None or str(falsifier).startswith("meep_")})
    return out


_RECORDS = _records()
_IDS = [f"{r['case']}:{Path(r['rel']).stem}:{r['arm']}" for r in _RECORDS]

if not _RECORDS:  # pragma: no cover - the artifacts are committed
    pytest.skip("no committed cv22/cv23 record to judge", allow_module_level=True)


def _evaluate(record, *, windows=None):
    """The consumer's own evaluator on one committed record."""
    block = record["doc"]
    if record["case"] == "cv22":
        win = G.arm_windows(block) if windows is None else windows
        return G.evaluate_e2(block["freqs_hz"], block["R_rfx"], block["T_rfx"],
                             block["model"], block["params"], block["dt_s"],
                             tail=block["tail"], require_complete=True, windows=win), win
    win = L.arm_windows(block) if windows is None else windows
    return L.evaluate_e2(block["freqs_hz"], block["R_rfx"], block["T_rfx"],
                         block["params"], block["dt_s"], tail=block["tail"],
                         dx=block["run"]["dx_m"], require_complete=True, windows=win), win


def _uncapped_witness(block: dict, tail: dict | None = None):
    """``W_wit`` WITHOUT the ceiling cap -- the shape shipped before the
    post-review amendment, kept here so the cap's effect is measurable rather
    than asserted."""
    freqs = np.asarray(block["freqs_hz"], dtype=float)
    run = block["run"]
    dx, dt = float(run["dx_m"]), float(block["dt_s"])
    tail = tail if tail is not None else block["tail"]
    R_lat, T_lat = de.yee_lattice_slab_rt_model(freqs, block["model"], block["params"],
                                                SF.D_SLAB_M, dx, dt)
    rate, _src = LW.ringdown_rate(run["record"], tail)
    terms = LW.budget_terms(freqs, block["inc_amp_rel"], dt=dt, n_steps=int(run["n_steps"]),
                            scat_tail_rel=tail["scat_refl_rel"],
                            trans_tail_rel=tail["total_trans_rel"],
                            purity_rel=tail["purity_inc_rel"], rate_1_s=rate)
    return LW.windows_from_terms(R_lat, T_lat, terms)


def _rederive_from_outside(block: dict):
    """Section 2's window, rebuilt here from the two leaves it names.

    Deliberately does NOT call ``slab_arm_windows``: this is the independent
    half. ``params`` is the DECLARED set -- ``params_run`` holds a falsifier's
    defect and must never size the window that catches it.
    """
    freqs = np.asarray(block["freqs_hz"], dtype=float)
    model, params = block["model"], block["params"]
    run = block["run"]
    dx, dt = float(run["dx_m"]), float(block["dt_s"])

    R_an, T_an = de.tmm_slab_rt(freqs, de.eps_analytic(freqs, model, params), SF.D_SLAB_M)
    R_lat, T_lat = de.yee_lattice_slab_rt_model(freqs, model, params, SF.D_SLAB_M, dx, dt)
    lat_R = np.abs(R_lat - R_an)
    lat_T = np.abs(T_lat - T_an)
    lat_A = np.abs((1.0 - R_lat - T_lat) - (1.0 - R_an - T_an))

    tail = block["tail"]
    rate, _src = LW.ringdown_rate(run["record"], tail)
    terms = LW.budget_terms(freqs, block["inc_amp_rel"], dt=dt,
                            n_steps=int(run["n_steps"]),
                            scat_tail_rel=tail["scat_refl_rel"],
                            trans_tail_rel=tail["total_trans_rel"],
                            purity_rel=tail["purity_inc_rel"], rate_1_s=rate)
    wit_R, wit_T, wit_A = LW.windows_from_terms(R_lat, T_lat, terms)
    # The cap (post-review amendment): the a-priori ceiling, built here from
    # the standard's own function with the DECLARED bars and the DERIVED ring
    # rate rather than taken from the module under test.
    ceil_R, ceil_T, ceil_A = _ceiling(block, R_lat, T_lat)
    wit_R = np.minimum(wit_R, ceil_R)
    wit_T = np.minimum(wit_T, ceil_T)
    wit_A = np.minimum(wit_A, ceil_A)

    m = ENVELOPE_GATE_MULTIPLIER
    return {"R": m * (lat_R + wit_R), "T": m * (lat_T + wit_T), "A": m * (lat_A + wit_A)}


def _ceiling(block: dict, R_lat, T_lat):
    """``lattice_witness.ceiling_windows`` on this record: the DECLARED bars and
    the DERIVED ring rate, none of it measured on the run it bounds."""
    run = block["run"]
    return LW.ceiling_windows(np.asarray(block["freqs_hz"], dtype=float),
                              block["inc_amp_rel"], R_lat, T_lat,
                              dt=float(block["dt_s"]), n_steps=int(run["n_steps"]),
                              rate_1_s=float(run["record"]["rate_ring_1_s"]))


# ---------------------------------------------------------------------------
# 1. The window the evaluator enforces IS the declared derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("record", _RECORDS, ids=_IDS)
def test_the_enforced_window_is_the_one_section_2_declares(record):
    out, _win = _evaluate(record)
    expected = _rederive_from_outside(record["doc"])
    for key, enforced in (("R", "window_R"), ("T", "window_T"), ("A", "window_A")):
        if enforced not in out:
            continue            # cv22 gates no absorption
        got = np.asarray(out[enforced], dtype=float)
        assert got.shape == expected[key].shape
        np.testing.assert_allclose(
            got, expected[key], rtol=1e-12, atol=0.0,
            err_msg=(f"{record['rel']}::{record['arm']} enforces a {key} window that is "
                     f"not MULTIPLIER * (W_lat + W_wit) re-derived from "
                     f"dispersive_eps + lattice_witness"))


@pytest.mark.parametrize("record", _RECORDS, ids=_IDS)
def test_the_band_mean_window_is_the_gated_mean_of_the_per_bin_one(record):
    out, _win = _evaluate(record)
    gated = np.asarray(out["gated"], dtype=bool)
    expected = _rederive_from_outside(record["doc"])
    for key, enforced in (("R", "mean_window_R"), ("T", "mean_window_T"),
                          ("A", "mean_window_A")):
        if enforced not in out:
            continue
        assert out[enforced] == pytest.approx(float(expected[key][gated].mean()), rel=1e-12)


@pytest.mark.parametrize("record", _RECORDS, ids=_IDS)
def test_the_ade_term_is_not_added_on_top_of_the_lattice_window(record):
    """Section 2.4. ``W_lat`` is built on the discrete-time permittivity, so the
    ADE term is inside it; the enforced window must not carry a second copy.

    Checked where it is decidable: ``w_ade`` is strictly positive somewhere in
    every gated band, so an additive term would show up as a per-bin excess.
    """
    out, _win = _evaluate(record)
    expected = _rederive_from_outside(record["doc"])
    w_ade_R = np.asarray(out["w_ade_R"], dtype=float)
    assert np.any(w_ade_R > 0.0), "w_ade is identically zero; this test would prove nothing"
    np.testing.assert_allclose(np.asarray(out["window_R"], dtype=float),
                               expected["R"], rtol=1e-12, atol=0.0)
    assert not np.allclose(np.asarray(out["window_R"], dtype=float),
                           expected["R"] + w_ade_R, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 2. cv04 cannot reach an E2 window (the mutation, both halves)
# ---------------------------------------------------------------------------

def _doubled_scalar_windows(module):
    return SF.Windows(2.0 * module.W_BIN, 2.0 * module.W_MEAN_R, 2.0 * module.W_MEAN_T)


@pytest.mark.parametrize("record", _RECORDS[:6], ids=_IDS[:6])
def test_doubling_the_cv04_adopted_scalars_does_not_move_an_e2_window(record, monkeypatch):
    """The borrowing is gone: the E2 window does not read the envelope.

    Mutating the module-level scalars is the shape a re-adoption (or a plant)
    takes. The E2 windows must be bit-identical across it.
    """
    before, _ = _evaluate(record)
    module = G if record["case"] == "cv22" else L
    monkeypatch.setattr(module, "W_BIN", 2.0 * module.W_BIN)
    monkeypatch.setattr(module, "W_MEAN_R", 2.0 * module.W_MEAN_R)
    monkeypatch.setattr(module, "W_MEAN_T", 2.0 * module.W_MEAN_T)
    monkeypatch.setattr(module, "WINDOWS", _doubled_scalar_windows(module))
    if record["case"] == "cv23":
        monkeypatch.setattr(module, "W_BIN_A", 2.0 * module.W_BIN_A)
        monkeypatch.setattr(module, "W_MEAN_A", 2.0 * module.W_MEAN_A)
    after, _ = _evaluate(record)
    for key in ("window_R", "window_T", "window_A", "mean_window_R",
                "mean_window_T", "mean_window_A"):
        if key not in before:
            continue
        assert json.dumps(before[key]) == json.dumps(after[key]), (
            f"{record['rel']}::{record['arm']} moved {key} when the cv04-adopted "
            f"scalars moved: an E2 window still borrows from cv04")
    assert before["gates"] == after["gates"]


def test_the_doubling_mutation_is_live_on_the_meep_leg():
    """The paired half: the same mutation DOES move a window somewhere.

    Without this, the test above would also pass against a window that reads
    nothing at all -- an evaluator that returned a constant, say.
    """
    record = next(r for r in _RECORDS
                  if r["case"] == "cv22" and Path(r["rel"]).name == "rfx.json")
    block = record["doc"]
    meep = block.get("meep")
    if not meep or not meep.get("present"):
        pytest.skip("no Meep leg committed for this arm")
    meep_doc = {"freqs_hz": block["freqs_hz"], "R": meep["R_meep"], "T": meep["T_meep"],
                "dt_meep_s": meep["dt_meep_s"], "meep_params": meep["meep_params_reported"],
                "precheck": meep.get("precheck")}
    e2, _ = _evaluate(record)
    base = G.evaluate_e4(e2, meep_doc, windows=G.WINDOWS)
    moved = G.evaluate_e4(e2, meep_doc, windows=_doubled_scalar_windows(G))
    assert not np.allclose(np.asarray(base["window4_R"]), np.asarray(moved["window4_R"])), (
        "doubling the cv04-adopted scalars did not move the Meep-leg window either; "
        "the mutation is inert and proves nothing about the E2 side")


@pytest.mark.parametrize("record", _RECORDS[:6], ids=_IDS[:6])
def test_the_legacy_scalar_shape_is_still_distinguishable(record):
    """A regression that reverted to the borrowed scalar must be visible.

    Handing the evaluator the old ``slab_family.Windows`` reproduces the old
    flat window; if that were indistinguishable from the derivation, the
    property above would be vacuous.
    """
    module = G if record["case"] == "cv22" else L
    derived, _ = _evaluate(record)
    legacy, _ = _evaluate(record, windows=module.WINDOWS)
    assert not np.allclose(np.asarray(derived["window_R"]),
                           np.asarray(legacy["window_R"]))
    flat = np.asarray(legacy["window_R"]) - np.asarray(legacy["w_ade_R"])
    assert np.allclose(flat, module.W_BIN), (
        "the legacy path no longer produces W_BIN + w_ade, so this control is stale")


def test_the_shared_multiplier_moves_every_derived_window(monkeypatch):
    """The one reviewer-visible object: scaling it scales the whole family."""
    record = _RECORDS[0]
    base = SAW.from_arm_doc(record["doc"])
    monkeypatch.setattr("tests._gate_policy.ENVELOPE_GATE_MULTIPLIER",
                        2.0 * ENVELOPE_GATE_MULTIPLIER)
    moved = SAW.from_arm_doc(record["doc"])
    for key in ("R", "T", "A"):
        np.testing.assert_allclose(np.asarray(getattr(moved, key)),
                                   2.0 * np.asarray(getattr(base, key)), rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. A falsifier cannot size the window that catches it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "record",
    [r for r in _RECORDS if not r["declared"]],
    ids=[i for r, i in zip(_RECORDS, _IDS) if not r["declared"]])
def test_a_wrong_model_does_not_size_its_own_window(record):
    """The window is built from ``params`` (declared), never ``params_run``.

    Both are committed in every falsifier record, so the claim is checkable
    rather than asserted: the enforced window must equal the DECLARED
    derivation, and must differ from the one the defect would have produced.
    """
    block = record["doc"]
    assert block["params"] != block["params_run"], "this record plants no defect"
    out, _ = _evaluate(record)
    declared = _rederive_from_outside(block)
    np.testing.assert_allclose(np.asarray(out["window_R"], dtype=float),
                               declared["R"], rtol=1e-12, atol=0.0)
    defective = _rederive_from_outside({**block, "params": block["params_run"]})
    assert not np.allclose(declared["R"], defective["R"], rtol=1e-9), (
        f"{record['rel']}: the defect does not change the window it would have "
        f"sized, so this record cannot demonstrate the property")


# ---------------------------------------------------------------------------
# 4. Every committed record keeps the verdict the pre-declaration expected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("record", _RECORDS, ids=_IDS)
def test_declared_arms_pass_and_wrong_model_falsifiers_fail(record):
    out, _ = _evaluate(record)
    if record["declared"]:
        assert out["e2_ok"] is True, (
            f"{record['rel']}::{record['arm']} is a DECLARED arm and fails E2 under "
            f"the per-arm window. Section 4 of the pre-declaration says STOP: do not "
            f"widen, do not re-pin. gates={out['gates']}")
        assert out["incomplete_gates"] == []
    else:
        assert out["e2_ok"] is False, (
            f"{record['rel']}::{record['arm']} is a wrong-model falsifier and PASSES "
            f"E2 under the per-arm window. A window that a planted defect walks "
            f"through is not a gate. gates={out['gates']}")


@pytest.mark.parametrize("record", _RECORDS, ids=_IDS)
def test_the_committed_verdicts_are_still_reproduced_by_the_old_windows(record):
    """The baseline the before/after table rests on.

    Every committed record's gate dict is reproducible from the artifact under
    the windows it was judged by, so a difference in the new column is the
    derivation and not a replay drift.
    """
    module = G if record["case"] == "cv22" else L
    out, _ = _evaluate(record, windows=module.WINDOWS)
    committed = record["doc"]["gates"]
    for name, value in out["gates"].items():
        if name not in committed:
            continue
        assert bool(value) is bool(committed[name]), (
            f"{record['rel']}::{record['arm']} gate {name}: replay {value}, "
            f"committed {committed[name]}")


# ---------------------------------------------------------------------------
# 5. The settling bar these windows stand on, and the ceiling that caps them
# ---------------------------------------------------------------------------
#
# `W_wit` is built from tails MEASURED on the record it then judges, so the
# self-certification question is real: could a worse-settled record buy itself
# a wider continuum window and walk a defect through? Two things answer it, and
# GL1 is NOT one of them -- GL1 is judged against the same `W_wit`, so it moves
# with it and bounds nothing independently.
#
#   (a) The record law. `G3_tail` is `tail["ok"]`, the record's own settling
#       witness against `tail["limit"]`, which the r3 recipe sets to
#       `slab_family.SETTLING_LIMIT` = 1e-2 (`slab_rig.py:109`) and then
#       EXTENDS the record until it is met, growing the box rather than
#       clipping (`slab_rig.py:190`). A record above the bar is not a wider
#       window, it is a red gate.
#   (b) The ceiling cap (post-review amendment). `W_wit` is capped per bin at
#       `lattice_witness.ceiling_windows`, the same budget with the DECLARED
#       bars and the DERIVED ring rate -- so above the bar the window stops
#       responding to the record's tail at all.
#
# Both are pinned below, and both are shown to bite by measurement rather than
# by assertion.

_R3_SETTLING_BAR = 1e-2
_FALSIFIER_RECORDS = [r for r in _RECORDS if not r["declared"]]
_FALSIFIER_IDS = [i for r, i in zip(_RECORDS, _IDS) if not r["declared"]]


def _with_tails_at(block: dict, bar: float, frac: float = 0.999) -> dict:
    """The same record with both settling tails and the purity pushed to
    ``frac`` of ``bar`` -- the most contaminated record that still passes."""
    tail = dict(block["tail"])
    tail["scat_refl_rel"] = bar * frac
    tail["total_trans_rel"] = bar * frac
    tail["purity_inc_rel"] = SF.TAIL_PURITY_LIMIT * frac
    return dict(block, tail=tail)


def test_the_settling_bar_these_windows_assume_is_the_r3_one():
    """The bar is a value, pinned, not a name resolved at read time.

    `slab_family.SETTLING_LIMIT` is what the r3 record recipe extends against
    AND what `lattice_witness`'s a-priori ceiling is built from, so the cap in
    `witness_term` inherits it. Moving it moves both halves of the defence at
    once, which is the visibility this pin is for.
    """
    assert SF.SETTLING_LIMIT == _R3_SETTLING_BAR
    assert LW.SETTLING_BAR == SF.SETTLING_LIMIT, (
        "the ceiling the windows are capped at is no longer built from the same "
        "settling bar the record law extends against")
    assert SF.TAIL_LIMIT > SF.SETTLING_LIMIT, (
        "TAIL_LIMIT is the looser non-r3 bar; if it stopped being looser this "
        "test's falsifier arm below would prove nothing")


@pytest.mark.parametrize("record", _RECORDS, ids=_IDS)
def test_every_committed_record_sits_under_the_settling_bar_it_is_judged_on(record):
    tail = record["doc"]["tail"]
    worst = max(float(tail["scat_refl_rel"]), float(tail["total_trans_rel"]))
    assert tail["limit"] == pytest.approx(SF.SETTLING_LIMIT)
    assert bool(tail["ok"]) is True
    assert worst < SF.SETTLING_LIMIT, (record["rel"], record["arm"], worst)
    print(f"settling headroom {record['case']}:{Path(record['rel']).stem}:{record['arm']} "
          f"x{SF.SETTLING_LIMIT / worst:.1f}")


@pytest.mark.parametrize("record", _FALSIFIER_RECORDS, ids=_FALSIFIER_IDS)
def test_a_falsifier_cannot_buy_its_way_out_with_a_worse_settled_record(record):
    """The adversarial arm: give the defective record the worst tails the
    settling bar admits and it must STILL fail."""
    contaminated = _with_tails_at(record["doc"], SF.SETTLING_LIMIT)
    out, _ = _evaluate(dict(record, doc=contaminated))
    assert out["e2_ok"] is False, (
        f"{record['rel']}::{record['arm']} reaches PASS once its tails are pushed to "
        f"the settling bar: the window is buyable. gates={out['gates']}")


@pytest.mark.parametrize("record", _RECORDS[:8], ids=_IDS[:8])
def test_the_capped_window_is_bounded_by_an_a_priori_ceiling(record):
    """What the cap buys, measured on both sides.

    However bad the record's own tail gets, the capped window cannot exceed
    ``MULTIPLIER x (W_lat + W_ceiling)``, every input of which is fixed before
    the run. The uncapped window has no such bound: at ten times the bar it
    runs away, which is the channel the cap closes.
    """
    freqs = np.asarray(record["doc"]["freqs_hz"], dtype=float)
    run = record["doc"]["run"]
    R_lat, T_lat = de.yee_lattice_slab_rt_model(
        freqs, record["doc"]["model"], record["doc"]["params"], SF.D_SLAB_M,
        float(run["dx_m"]), float(record["doc"]["dt_s"]))
    lat_R, _lat_T, _lat_A = SAW.lattice_term(
        freqs, record["doc"]["model"], record["doc"]["params"],
        float(run["dx_m"]), float(record["doc"]["dt_s"]))
    ceil_R, _ceil_T, _ceil_A = _ceiling(record["doc"], R_lat, T_lat)
    bound = ENVELOPE_GATE_MULTIPLIER * (lat_R + ceil_R)

    over_bar = _with_tails_at(record["doc"], SF.TAIL_LIMIT)     # 10x the bar
    capped_over = _rederive_from_outside(over_bar)["R"]
    assert np.all(capped_over <= bound * (1.0 + 1e-12)), (
        "the capped window exceeded its own a-priori ceiling bound")

    unc_over = _uncapped_witness(record["doc"], over_bar["tail"])[0]
    unc_bound = ENVELOPE_GATE_MULTIPLIER * (lat_R + unc_over)
    assert np.max(unc_bound / bound) > 2.0, (
        "the uncapped window does not run away above the bar either, so this "
        "control shows nothing about what the cap closes")


def test_without_the_cap_a_looser_bar_would_let_falsifiers_through():
    """Why the bar and the cap are both named: with neither, the window is
    buyable; with the r3 bar alone it is not; with the cap it does not even
    depend on the bar's value.

    Measured, not argued: at the non-r3 `TAIL_LIMIT` = 0.10 an UNCAPPED window
    lets some wrong-model falsifiers reach PASS, and the shipped capped window
    lets none through at either bar.
    """
    walked_uncapped, walked_capped = [], []
    for record in _FALSIFIER_RECORDS:
        block = _with_tails_at(record["doc"], SF.TAIL_LIMIT)
        out, _ = _evaluate(dict(record, doc=block))
        if out["e2_ok"]:
            walked_capped.append(f"{record['rel']}::{record['arm']}")
        # the uncapped window, judged by hand against the same residuals
        wR, wT, _wA = _uncapped_witness(record["doc"], block["tail"])
        lat = _rederive_from_outside(record["doc"])          # capped, for its lattice part
        g = np.asarray(record["doc"]["gated"], dtype=bool)
        m = ENVELOPE_GATE_MULTIPLIER
        latR, latT, _latA = SAW.lattice_term(
            np.asarray(record["doc"]["freqs_hz"], dtype=float), record["doc"]["model"],
            record["doc"]["params"], float(record["doc"]["run"]["dx_m"]),
            float(record["doc"]["dt_s"]))
        assert lat["R"].shape == wR.shape
        dR = np.abs(np.asarray(record["doc"]["R_rfx"], dtype=float)
                    - np.asarray(record["doc"]["R_tmm"], dtype=float))
        dT = np.abs(np.asarray(record["doc"]["T_rfx"], dtype=float)
                    - np.asarray(record["doc"]["T_tmm"], dtype=float))
        if (np.all(dR[g] <= (m * (latR + wR))[g])
                and np.all(dT[g] <= (m * (latT + wT))[g])):
            walked_uncapped.append(f"{record['rel']}::{record['arm']}")
    print(f"at TAIL_LIMIT={SF.TAIL_LIMIT}: uncapped G1 would pass "
          f"{len(walked_uncapped)}/{len(_FALSIFIER_RECORDS)} falsifiers "
          f"({sorted(walked_uncapped)}); capped lets {len(walked_capped)} through")
    assert walked_capped == [], (
        f"the shipped window is buyable at the looser bar: {walked_capped}")
    assert walked_uncapped, (
        "the uncapped window is not buyable at the looser bar either, so the cap "
        "cannot be justified by this measurement -- re-derive the claim")
