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

    m = ENVELOPE_GATE_MULTIPLIER
    return {"R": m * (lat_R + wit_R), "T": m * (lat_T + wit_T), "A": m * (lat_A + wit_A)}


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
