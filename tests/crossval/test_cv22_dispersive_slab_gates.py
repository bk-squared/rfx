"""cv22 dispersive slab -- gate replay and window-derivation witnesses.

Two kinds of test live here:

1. **Artifact-free** (always run): the windows are derived from the committed
   cv04 envelope by the repo's shared rule; the ADE discrete-time transfer
   functions used in the windows reproduce the LIVE ``init_debye`` /
   ``init_lorentz`` coefficient recurrences; every pre-declared falsifier
   exceeds its window analytically (the margins table of the note, §6,
   recomputed); the Meep wrong-convention falsifiers exceed the E4 windows.

2. **Artifact replay** (``pytest.skip`` while the VESSL artifacts are absent):
   ``validation/crossval/_22_dispersive_results/rfx.json`` is replayed
   through the same evaluators from its raw per-bin R/T and must reproduce
   its stored gate verdicts, all passing; each ``rfx__falsifier_<name>.json``
   must FAIL for the pre-declared reason; each
   ``meep_lorentz__falsifier_<name>.json`` must fail E4 against the baseline
   and carry ``precheck.passed == False``.

No FDTD runs here. Pre-declaration:
``docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER

_REPO = Path(__file__).resolve().parents[2]
_RESULTS = _REPO / "validation/crossval/_22_dispersive_results"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


de = _load("cv22_dispersive_eps", "validation/crossval/comparators/dispersive_eps.py")
G = _load("cv22_dispersive_gates", "validation/crossval/comparators/cv22_dispersive_gates.py")
_SF = _load("cv22_slab_family", "validation/crossval/comparators/slab_family.py")

_ENVELOPE = _REPO / "validation/crossval/_04_fresnel_results/envelope.json"
_COMPARATORS = _REPO / "validation/crossval/comparators"


def _e2(*args, **kw):
    """cv22's evaluator, handed cv22's own windows.

    The evaluator takes them as a required argument now (#928 round 2): it used
    to read cv22's module-level constants, which meant cv23 -- a different
    adopter calling the same function -- was judged by cv22's adoption. Every
    call in this file is cv22 judging cv22, so the record is named once here.
    """
    return G.evaluate_e2(*args, windows=G.WINDOWS, **kw)


def _e4(e2, meep_doc):
    return G.evaluate_e4(e2, meep_doc, windows=G.WINDOWS)


def _round_up(value: float, multiplier: float, quantum: float) -> float:
    """The repo's envelope->gate arithmetic, written out again on purpose.

    ``tests/_gate_policy.gate_from_envelope`` is the implementation under test
    on one side of every comparison below, so the other side must not be it.
    """
    return math.ceil(value * multiplier * quantum) / quantum


@contextlib.contextmanager
def _tmp_tree():
    """A scratch copy of the comparator package + the producer results dir.

    The point is to re-import the CONSUMER against a different artifact: a
    module that hard-codes its windows behaves identically to one that derives
    them until the artifact underneath it changes.
    """
    root = Path(tempfile.mkdtemp(prefix="cv22_envelope_"))
    try:
        comparators = root / "validation/crossval/comparators"
        comparators.mkdir(parents=True)
        (root / "validation/crossval/_04_fresnel_results").mkdir(parents=True)
        for name in ("cv22_dispersive_gates.py", "cv23_lossy_gates.py",
                     "dispersive_eps.py", "slab_family.py"):
            shutil.copy(_COMPARATORS / name, comparators / name)
        shutil.copy(_ENVELOPE, root / "validation/crossval/_04_fresnel_results/envelope.json")
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_envelope(tree: Path, doc: dict) -> None:
    (tree / "validation/crossval/_04_fresnel_results/envelope.json").write_text(
        json.dumps(doc, indent=1), encoding="utf-8")


def _repoint(src: Path, adopt: tuple[str, str] | None) -> None:
    """Re-point a copied module's adoption record: the deliberate, reviewed
    edit the real thing would be."""
    if adopt is None:
        return
    revision, digest = adopt
    text = src.read_text(encoding="utf-8")
    text, n1 = re.subn(r'"adopted_revision": "r1"',
                       f'"adopted_revision": "{revision}"', text)
    text, n2 = re.subn(r'"revision_sha256": "sha256:[0-9a-f]+"',
                       f'"revision_sha256": "{digest}"', text)
    assert (n1, n2) == (1, 1), (
        f"{src.name}'s adoption record changed shape; this scratch re-adoption "
        f"edits it textually and must be updated with it")
    src.write_text(text, encoding="utf-8")


def _load_consumers_from(tree: Path, *, adopt_cv22=None, adopt_cv23=None):
    """Import FRESH cv22 AND cv23 gate modules out of *tree*.

    Both are loaded because they are two INDEPENDENT adopters of one artifact:
    re-pointing cv22 alone must move cv22 and leave cv23 where its own record
    says it is. cv23 binds the scratch cv22 (registered under its plain name
    first), so the pair is self-consistent inside the tree.
    """
    comparators = tree / "validation/crossval/comparators"
    _repoint(comparators / "cv22_dispersive_gates.py", adopt_cv22)
    _repoint(comparators / "cv23_lossy_gates.py", adopt_cv23)
    names = ("slab_family", "cv22_dispersive_gates", "cv23_lossy_gates")
    saved_modules = {k: sys.modules.get(k) for k in names}
    saved_path = list(sys.path)
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(
                name, comparators / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module          # so the next one binds THIS copy
            spec.loader.exec_module(module)
            loaded[name] = module
        return loaded["cv22_dispersive_gates"], loaded["cv23_lossy_gates"]
    finally:
        for key, value in saved_modules.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        sys.path[:] = saved_path


def _load_cv22_from(tree: Path, *, adopt: tuple[str, str] | None = None):
    return _load_consumers_from(tree, adopt_cv22=adopt, adopt_cv23=adopt)[0]


def _rig_dt() -> float:
    from rfx.grid import Grid
    grid = Grid(freq_max=20e9, domain=(G.NX_INTERIOR * G.DX_M, 0.004, G.DX_M),
                dx=G.DX_M, cpml_layers=G.N_CPML, mode="2d_tmz")
    return float(grid.dt)


def _r3_records(dt):
    return {arm: G.derive_record_length(G.ARMS[arm]["model"], G.ARMS[arm]["params"], dt)
            for arm in G.ARM_ORDER}


def _rfx_bins():
    """The rfx rFFT bin grid of the r3 recipe (Lorentz record 1228 steps ->
    nfft 16384; all three arms share it, see test_r3_record_lengths_are_derived)."""
    dt = _rig_dt()
    n_steps = _r3_records(dt)["lorentz"]["n_steps"]
    nfft = int(2 ** np.ceil(np.log2(n_steps)) * G.NFFT_OVERSAMPLE)
    f = np.fft.rfftfreq(nfft, d=dt)
    f = f[(f > G.MASK_F_LO_HZ) & (f < G.MASK_F_HI_HZ)]
    return f, dt


# ---------------------------------------------------------------------------
# 1. Artifact-free witnesses
# ---------------------------------------------------------------------------

def test_windows_are_rederived_from_the_producer_artifact_outside_the_consumer():
    """The windows this case enforces, re-derived from the PRODUCER's artifact.

    Nothing about the derivation is taken from cv22: the values are read out of
    `_04_fresnel_results/envelope.json`, the round-up-and-quantize arithmetic is
    written out again below rather than borrowed from ``gate_from_envelope``
    (one of the two sides must not be the helper under test), and the result is
    compared with the numbers the evaluator actually uses. The multiplier is
    read from the shared policy and cross-checked against the one the adoption
    record was made under, so a policy change that nobody re-adopted is visible.

    This replaces two guards deleted in #928: an equality pin against the
    STUDIO UI fixture (which made a display fixture a physics source) and a
    grep for `max|R+T-1| = 0.0487` in cv04's source comment.
    """
    doc = json.loads(_ENVELOPE.read_text())
    assert doc["producer"] == "04_multilayer_fresnel"
    adoption = G.CV04_ADOPTION
    rev = doc["revisions"][adoption["adopted_revision"]]
    assert rev["status"] == "active"

    # Policy: live vs the one this adoption was made under.
    assert ENVELOPE_GATE_MULTIPLIER == adoption["gate_policy"]["multiplier"], (
        "the shared envelope->gate multiplier moved since this case adopted its "
        "envelope; re-adopt deliberately (the windows all move) rather than "
        "letting the change ride in"
    )
    mult = adoption["gate_policy"]["multiplier"]
    quantum = adoption["gate_policy"]["quantum"]

    values = rev["values"]
    assert G.W_BIN == _round_up(values["per_bin_max_RT_closure"], mult, quantum)
    assert G.W_MEAN_R == _round_up(values["mean_dR"], mult, quantum)
    assert G.W_MEAN_T == _round_up(values["mean_dT"], mult, quantum)
    # ... and the consumer's copy of the values is the artifact's, not a copy.
    assert G.CV04_ENVELOPE == values

    # Staleness: REPORTED, never enforced -- a newer revision is evidence, and
    # evidence does not move a gate on its own.
    info = G.CV04_ADOPTED
    print(f"cv22 calibration: adopted {info['revision']} "
          f"(latest {info['latest_revision']}, newer available: "
          f"{info['newer_revisions'] or 'none'}); witness_status "
          f"{info['witness_status']!r}")

    # cv04 witness constants carried unchanged (structure, not envelope).
    assert (G.TAIL_WINDOW, G.TAIL_PURITY_LIMIT, G.TAIL_LIMIT, G.CONS_MAX_LIMIT) == (50, 1e-3, 0.10, 0.06)


def test_the_evaluator_applies_exactly_the_rederived_windows():
    """Not only the three scalars: the per-bin windows the evaluator adds the
    ADE term to, the gated mask, and the pass/fail boundary itself."""
    f, dt = _rfx_bins()
    doc = json.loads(_ENVELOPE.read_text())
    values = doc["revisions"][G.CV04_ADOPTION["adopted_revision"]]["values"]
    mult = G.CV04_ADOPTION["gate_policy"]["multiplier"]
    quantum = G.CV04_ADOPTION["gate_policy"]["quantum"]
    w_bin = _round_up(values["per_bin_max_RT_closure"], mult, quantum)
    w_mean_R = _round_up(values["mean_dR"], mult, quantum)
    w_mean_T = _round_up(values["mean_dT"], mult, quantum)

    arm = "lorentz"
    model, params = G.ARMS[arm]["model"], G.ARMS[arm]["params"]
    # The oracle and the ADE window term, re-derived HERE from the leaf
    # `dispersive_eps`, not taken from G.analytic_rt / G.ade_window: the module
    # under test must not supply both sides of its own comparison.
    eps_exact = de.eps_analytic(f, model, params)
    R_an, T_an = de.tmm_slab_rt(f, eps_exact, _SF.D_SLAB_M)
    eps_ade = de.eps_numerical_ade(f, model, params, dt)
    R_ade, T_ade = de.tmm_slab_rt(f, eps_ade, _SF.D_SLAB_M)
    w_ade_R, w_ade_T = np.abs(R_ade - R_an), np.abs(T_ade - T_an)
    e2 = _e2(f, R_an, T_an, model, params, dt)

    # the gated mask, recomputed from the declared band
    g_expected = (f >= G.BAND_GATED_HZ[0]) & (f <= G.BAND_GATED_HZ[1])
    assert np.array_equal(np.asarray(e2["gated"], dtype=bool), g_expected)
    # the complete per-bin windows, and the band-mean ones
    assert np.array_equal(np.asarray(e2["window_R"]), w_bin + w_ade_R)
    assert np.array_equal(np.asarray(e2["window_T"]), w_bin + w_ade_T)
    assert e2["mean_window_R"] == w_mean_R + float(np.mean(w_ade_R[g_expected]))
    assert e2["mean_window_T"] == w_mean_T + float(np.mean(w_ade_T[g_expected]))
    assert e2["e2_ok"] is False or e2["gates"]["G1_R"]   # zero-error input passes G1

    # the pass/fail boundary sits where the re-derived window says it does
    i = int(np.argmax(g_expected))
    win_i = w_bin + w_ade_R[i]
    just_inside = np.array(R_an, dtype=float)
    just_inside[i] = R_an[i] + win_i * (1 - 1e-9)
    just_outside = np.array(R_an, dtype=float)
    just_outside[i] = R_an[i] + win_i * (1 + 1e-6)
    assert _e2(f, just_inside, T_an, model, params, dt)["gates"]["G1_R"]
    assert not _e2(f, just_outside, T_an, model, params, dt)["gates"]["G1_R"]
    # and the band-mean gate's boundary, on the same principle
    mean_win = w_mean_R + float(np.mean(w_ade_R[g_expected]))
    n_g = int(g_expected.sum())
    over = np.array(R_an, dtype=float)
    over[g_expected] = R_an[g_expected] + mean_win * (1 + 1e-6)
    assert not _e2(f, over, T_an, model, params, dt)["gates"]["G2_R"]
    under = np.array(R_an, dtype=float)
    under[g_expected] = R_an[g_expected] + mean_win * (1 - 1e-6)
    assert _e2(f, under, T_an, model, params, dt)["gates"]["G2_R"]
    assert n_g >= 100


def test_unadopted_new_evidence_leaves_the_windows_alone():
    """(a) The producer appends a revision. Nothing moves.

    This is principle 2 (no self-certification) as a mechanical fact: a
    producer re-run cannot widen the gates that judge its own family, because
    the consumer derives from the revision its own declaration names.
    """
    with _tmp_tree() as tree:
        doc = json.loads(_ENVELOPE.read_text())
        r2 = json.loads(json.dumps(doc["revisions"]["r1"]))
        r2["bootstrap"] = False
        r2["values"] = {k: v * 3.0 for k, v in r2["values"].items()}
        r2["revision_hash"] = _SF.revision_hash({k: v for k, v in r2.items()
                                                 if k != "revision_hash"})
        doc["revisions"]["r2"] = r2
        doc["latest_revision"] = "r2"
        _write_envelope(tree, doc)
        mod = _load_cv22_from(tree)
        assert mod.W_BIN == G.W_BIN
        assert mod.W_MEAN_R == G.W_MEAN_R
        assert mod.W_MEAN_T == G.W_MEAN_T
        assert mod.CV04_ADOPTED["newer_revisions"] == ["r2"]


def test_editing_adopted_evidence_fails_integrity():
    """(b) The adopted revision is altered in place. Two shapes, both refused."""
    # (b1) values edited, recorded hash left behind: the artifact fails against
    # itself, so nobody has to notice the consumer.
    with _tmp_tree() as tree:
        doc = json.loads(_ENVELOPE.read_text())
        doc["revisions"]["r1"]["values"]["mean_dR"] = 0.05
        _write_envelope(tree, doc)
        with pytest.raises(ValueError, match="edited after it was written"):
            _load_cv22_from(tree)
    # (b2) values edited AND the artifact's own hash re-stamped: the artifact is
    # self-consistent, and the consumer's pin is what refuses it.
    with _tmp_tree() as tree:
        doc = json.loads(_ENVELOPE.read_text())
        block = doc["revisions"]["r1"]
        block["values"]["mean_dR"] = 0.05
        block["revision_hash"] = _SF.revision_hash({k: v for k, v in block.items()
                                                    if k != "revision_hash"})
        _write_envelope(tree, doc)
        with pytest.raises(ValueError, match="adoption record pins"):
            _load_cv22_from(tree)


def _expected_windows(values: dict, mult: float, quantum: float) -> dict:
    """Every window either consumer derives, re-derived HERE from the adopted
    values. Named individually because a review appended `W_BIN_A = 148 / 1000`
    to cv23 -- a GATED window, hard-coded in expression form -- and the suite
    stayed green: nothing compared cv23's A windows against its own adoption.
    """
    w_bin = _round_up(values["per_bin_max_RT_closure"], mult, quantum)
    w_mean_R = _round_up(values["mean_dR"], mult, quantum)
    w_mean_T = _round_up(values["mean_dT"], mult, quantum)
    return {
        "W_BIN": w_bin, "W_MEAN_R": w_mean_R, "W_MEAN_T": w_mean_T,
        "W_BIN_A": 2.0 * w_bin, "W_MEAN_A": w_mean_R + w_mean_T,
        "W_BIN_A_TIGHT": w_bin,
        "W_MEAN_A_TIGHT": _round_up(values["mean_closure"], mult, quantum),
    }


_CV22_WINDOW_NAMES = ("W_BIN", "W_MEAN_R", "W_MEAN_T")
_CV23_WINDOW_NAMES = _CV22_WINDOW_NAMES + ("W_BIN_A", "W_MEAN_A",
                                           "W_BIN_A_TIGHT", "W_MEAN_A_TIGHT")


def _realized(mod, case: str, f, dt) -> dict:
    """What the EVALUATOR actually applies, not what the module declares.

    A constant can be right while the evaluator reads someone else's: that was
    round-2 item 1, where cv23's R/T verdicts came out of cv22's windows.
    """
    if case == "cv22":
        arm = mod.ARMS["lorentz"]
        R, T = mod.analytic_rt(f, arm["model"], arm["params"])
        out = mod.evaluate_e2(f, R, T, arm["model"], arm["params"], dt,
                              windows=mod.WINDOWS)
    else:
        params = mod.ARMS["tand1"]["params"]
        R, T, _A = mod.analytic_rta(f, params)
        out = mod.evaluate_e2(f, R, T, params, dt)
    got = {"window_R_first": out["window_R"][0],
           "mean_window_R": out["mean_window_R"],
           "mean_window_T": out["mean_window_T"]}
    if case == "cv23":
        got["window_A_first"] = out["window_A"][0]
        got["mean_window_A"] = out["mean_window_A"]
    return got


def _assert_consumer(mod, case: str, values: dict, mult, quantum, f, dt):
    """Every declared window of *mod*, and every window its evaluator applies,
    against a local re-derivation from the values *mod* adopted."""
    want = _expected_windows(values, mult, quantum)
    names = _CV22_WINDOW_NAMES if case == "cv22" else _CV23_WINDOW_NAMES
    for name in names:
        assert getattr(mod, name) == want[name], (case, name)
    realized = _realized(mod, case, f, dt)
    # the per-bin window is w_bin + the model-error term, so compare the part
    # that comes from the adoption by subtracting the term the evaluator adds
    assert realized["mean_window_R"] >= want["W_MEAN_R"]
    assert realized["mean_window_R"] - want["W_MEAN_R"] < 2e-3
    assert realized["mean_window_T"] >= want["W_MEAN_T"]
    assert realized["window_R_first"] >= want["W_BIN"]
    assert realized["window_R_first"] - want["W_BIN"] < 2e-3
    if case == "cv23":
        assert realized["window_A_first"] >= want["W_BIN_A"]
        assert realized["mean_window_A"] >= want["W_MEAN_A"]
    return realized


def test_adopting_a_new_revision_moves_the_adopting_consumer_and_nothing_else():
    """(c) An explicit re-adoption, judged on the evaluator's output.

    Both consumers are loaded from the scratch tree; the one that re-adopts
    must move -- every declared window AND every window its evaluator applies
    -- and the one that did not must sit exactly where its own record says. It
    is also the check that no consumer hard-codes a window: a module carrying
    0.074, or `148 / 1000`, sails through (a) and (b) and fails here.
    """
    live_before = (G.W_BIN, G.W_MEAN_R, G.W_MEAN_T)
    f, dt = _rfx_bins()
    mult = G.CV04_ADOPTION["gate_policy"]["multiplier"]
    quantum = G.CV04_ADOPTION["gate_policy"]["quantum"]
    doc = json.loads(_ENVELOPE.read_text())
    r1_values = doc["revisions"]["r1"]["values"]

    def _tree_with_r2(tree, factor):
        d = json.loads(_ENVELOPE.read_text())
        r2 = json.loads(json.dumps(d["revisions"]["r1"]))
        r2["bootstrap"] = False
        r2["values"] = {k: v * factor for k, v in r2["values"].items()}
        r2["revision_hash"] = _SF.revision_hash({k: v for k, v in r2.items()
                                                 if k != "revision_hash"})
        d["revisions"]["r2"] = r2
        d["latest_revision"] = "r2"
        _write_envelope(tree, d)
        return r2

    # --- cv22 re-adopts r2; cv23's own record still says r1 -----------------
    with _tmp_tree() as tree:
        r2 = _tree_with_r2(tree, 2.0)
        mod22, mod23 = _load_consumers_from(tree, adopt_cv22=("r2", r2["revision_hash"]))
        assert mod22.CV04_ADOPTION["adopted_revision"] == "r2"
        assert mod23.CV04_ADOPTION["adopted_revision"] == "r1"
        moved = _assert_consumer(mod22, "cv22", r2["values"], mult, quantum, f, dt)
        stayed = _assert_consumer(mod23, "cv23", r1_values, mult, quantum, f, dt)
        # and the two are genuinely different now, in the evaluator's output
        assert moved["mean_window_R"] != stayed["mean_window_R"]
        assert mod22.W_BIN != mod23.W_BIN

    # --- the mirror image, in a FRESH tree (the first tree's cv22 source has
    #     been re-pointed in place): cv23 re-adopts, cv22 does not ------------
    with _tmp_tree() as tree2:
        r2b = _tree_with_r2(tree2, 3.0)
        mod22b, mod23b = _load_consumers_from(tree2, adopt_cv23=("r2", r2b["revision_hash"]))
        assert mod22b.CV04_ADOPTION["adopted_revision"] == "r1"
        assert mod23b.CV04_ADOPTION["adopted_revision"] == "r2"
        stayed22 = _assert_consumer(mod22b, "cv22", r1_values, mult, quantum, f, dt)
        moved23 = _assert_consumer(mod23b, "cv23", r2b["values"], mult, quantum, f, dt)
        assert moved23["mean_window_A"] != stayed22["mean_window_R"]
        assert mod23b.W_BIN != mod22b.W_BIN

    # the live modules did not move with either scratch tree
    assert (G.W_BIN, G.W_MEAN_R, G.W_MEAN_T) == live_before


def test_a_missing_required_witness_cannot_be_a_pass():
    """Completeness (#928): an absent gate is None, and the claims-bearing
    aggregate refuses to call a two-of-three verdict PASS.

    The live instance this closes: `G3_tail` is None when no tail witness is
    handed in, and the default aggregate skipped it -- so a run whose settling
    witness never arrived would have read PASS on the rest. The default is
    kept for the analytic falsifier checks, which have no run behind them and
    legitimately carry no witness; the case scripts pass require_complete.
    """
    f, dt = _rfx_bins()
    arm = "lorentz"
    model, params = G.ARMS[arm]["model"], G.ARMS[arm]["params"]
    R_an, T_an = G.analytic_rt(f, model, params)

    # no witness handed in: passing on the other gates, INCOMPLETE
    lenient = _e2(f, R_an, T_an, model, params, dt)
    assert lenient["gates"]["G3_tail"] is None
    assert lenient["incomplete_gates"] == ["G3_tail"]
    assert lenient["gates_complete"] is False
    assert lenient["e2_ok"] is True, "the diagnostic aggregate is unchanged"

    strict = _e2(f, R_an, T_an, model, params, dt, require_complete=True)
    assert strict["e2_ok"] is False, (
        "a required witness that never arrived must not leave a PASS standing")
    assert strict["incomplete_gates"] == ["G3_tail"]

    # and with the witness present the two agree again, so the rule costs
    # nothing on a complete run
    tail = {"ok": True}
    complete = _e2(f, R_an, T_an, model, params, dt, tail=tail,
                             require_complete=True)
    assert complete["gates_complete"] and complete["e2_ok"]
    assert complete["e2_ok"] == _e2(f, R_an, T_an, model, params, dt,
                                              tail=tail)["e2_ok"]
    # a failing witness still fails, in both modes
    bad = _e2(f, R_an, T_an, model, params, dt, tail={"ok": False},
                        require_complete=True)
    assert bad["e2_ok"] is False and bad["gates_complete"]

    # ... and a declared gate that is ABSENT is incomplete too, not passing.
    # Without the declared set the aggregate can only see a key that is present
    # and None, so an evaluator that stops inserting a key -- an early return, a
    # dropped branch -- would read as a complete PASS.
    full = {name: True for name in G.DECLARED_GATES}
    assert G.aggregate_gates(full, declared=G.DECLARED_GATES,
                             require_complete=True)["e2_ok"] is True
    for dropped in G.DECLARED_GATES:
        partial = {k: v for k, v in full.items() if k != dropped}
        out = G.aggregate_gates(partial, declared=G.DECLARED_GATES,
                                require_complete=True)
        assert out["e2_ok"] is False, dropped
        assert out["incomplete_gates"] == [dropped]
        assert out["gates_complete"] is False


def test_a_non_active_revision_cannot_be_adopted():
    """A withdrawn or superseded revision is history, not calibration."""
    for status in ("withdrawn", "superseded", "scope-limited"):
        with _tmp_tree() as tree:
            doc = json.loads(_ENVELOPE.read_text())
            block = doc["revisions"]["r1"]
            block["status"] = status
            block["revision_hash"] = _SF.revision_hash({k: v for k, v in block.items()
                                                        if k != "revision_hash"})
            _write_envelope(tree, doc)
            adopt = ("r1", block["revision_hash"])
            with pytest.raises(ValueError, match="only an active revision"):
                _load_cv22_from(tree, adopt=adopt)


@pytest.mark.parametrize("arm", ["debye", "lorentz", "drude"])
def test_arms_have_strong_dispersion_in_the_gated_band(arm):
    f, _ = _rfx_bins()
    g = G.gated_mask(f)
    model, params = G.ARMS[arm]["model"], G.ARMS[arm]["params"]
    eps = de.eps_analytic(f[g], model, params)
    mag = np.abs(eps)
    assert (mag.max() - mag.min()) / mag.max() >= 0.30
    tand = -eps.imag / eps.real
    assert np.any((tand >= 0.1) & (tand <= 1.0)) or (tand.min() <= 1.0 <= tand.max())
    assert np.all(eps.imag < 0)  # passive in the rfx convention


def test_r3_record_lengths_are_derived_from_the_slab_ringdown():
    """§13 recipe: n_steps_min = n_pulse_end + max_f ln(100 w(f))/rate(f)/dt +
    TAIL_WINDOW over the incident ring band, per arm, inside the CPML
    round-trip gate of the nx 1000 rig; all three arms land on the same nfft
    so the gated bin grid is shared."""
    dt = _rig_dt()
    recs = _r3_records(dt)
    # §13: incident-weighted ring band (w >= 0.5 -> 1.13-15 GHz). Debye's slowest
    # component is the 1.4 GHz etalon (w 0.62, 1.18e10/s), not the 4 GHz one
    # r3 used (1.84e10/s, 1066 steps, witness -36 dB); Lorentz/Drude unchanged.
    assert {a: r["n_steps"] for a, r in recs.items()} == {"debye": 1108, "lorentz": 1228, "drude": 1168}
    f_lo, f_hi = G.ring_band_hz()
    assert 1.1e9 < f_lo < 1.2e9 and f_hi == 15e9
    for arm, r in recs.items():
        assert r["cpml_gate_ok"] and r["t_safe_cpml_steps"] == 1262
        assert r["n_pulse_end"] == 908 and r["settling_limit"] == 1e-2
        assert r["nx_interior"] == G.NX_INTERIOR_R3 == 1000
        assert r["ring_band_hz"] == [f_lo, f_hi]
    assert recs["debye"]["rate_ring_1_s"] < recs["debye"]["rate_material_1_s"]
    assert 1.4e9 < recs["debye"]["f_ring_hz"] < 1.5e9
    for arm in ("lorentz", "drude"):
        assert recs[arm]["rate_ring_1_s"] == recs[arm]["rate_material_1_s"]
    nffts = {int(2 ** np.ceil(np.log2(r["n_steps"])) * G.NFFT_OVERSAMPLE) for r in recs.values()}
    assert nffts == {16384}
    # cv04's own 719-step gate truncates every arm's ring-down (the r2 finding).
    assert all(r["n_steps"] > 719 for r in recs.values())


def test_ade_stability_constraints_of_the_note():
    dt = _rig_dt()
    tau = G.ARMS["debye"]["params"]["tau"]
    alpha = (2 * tau - dt) / (2 * tau + dt)
    assert abs(alpha) < 1
    lp = G.ARMS["lorentz"]["params"]
    w0 = 2 * np.pi * lp["f0"]
    assert w0 * dt < 2.0 and lp["delta"] >= 0
    dp = G.ARMS["drude"]["params"]
    assert dp["gamma"] / 2 * dt < 2.0
    # Meep-side mapped Debye pole.
    dt_meep = G.MEEP_COURANT * G.DX_M / de.C0
    assert 2 * np.pi * G.DEBYE_MEEP_MAP_FN_HZ * dt_meep < 2.0


@pytest.mark.parametrize("arm", ["debye", "lorentz", "drude"])
def test_discrete_transfer_function_matches_live_ade_recurrence(arm):
    """Drive the P recurrence built from the LIVE rfx coefficient arrays with
    a sinusoidal E and recover chi(omega); it must equal the closed-form
    eps_numerical_ade - eps_inf used in the windows."""
    from rfx.core.yee import EPS_0, init_materials
    from rfx.materials.debye import DebyePole, init_debye
    from rfx.materials.lorentz import drude_pole, init_lorentz, lorentz_pole

    dt = _rig_dt()
    model, params = G.ARMS[arm]["model"], G.ARMS[arm]["params"]
    args = de.rfx_pole_args(model, params)
    mats = init_materials((1, 1, 1))
    mats = mats._replace(eps_r=mats.eps_r * params["eps_inf"])
    if model == "debye":
        coeffs, _ = init_debye([DebyePole(**args)], mats, dt)
        alpha = float(coeffs.alpha[0, 0, 0, 0]); beta = float(coeffs.beta[0, 0, 0, 0])
        assert abs(alpha - (2 * params["tau"] - dt) / (2 * params["tau"] + dt)) < 1e-6
    else:
        pole = lorentz_pole(**args) if model == "lorentz" else drude_pole(**args)
        coeffs, _ = init_lorentz([pole], mats, dt)
        a = float(coeffs.a[0, 0, 0, 0]); b = float(coeffs.b[0, 0, 0, 0]); c = float(coeffs.c[0, 0, 0, 0])

    for f in (4.5e9, 7.0e9, 9.5e9):
        w = 2 * np.pi * f
        n_per = int(round(1.0 / (f * dt)))
        n_tr, n_meas = 40 * n_per, 20 * n_per   # settle, then project
        N = n_tr + n_meas
        t = np.arange(N + 1) * dt
        E = np.cos(w * t)
        P = np.zeros(N + 1)
        if model == "debye":
            for n in range(N):
                P[n + 1] = alpha * P[n] + beta * (E[n + 1] + E[n])
        else:
            for n in range(1, N):
                P[n + 1] = a * P[n] + b * P[n - 1] + c * E[n]
        # complex amplitude of P at omega on the steady-state window, by a
        # least-squares fit to [cos, sin] (exact for a pure sinusoid; no
        # integer-period assumption). E = cos(wt) has amplitude 1 + 0j.
        # The constant column absorbs the Drude recurrence's non-decaying DC
        # mode (omega_0 = 0 puts a root at z = 1: free carriers keep a
        # constant polarization offset from the transient, which is physical).
        seg = slice(n_tr, n_tr + n_meas)
        basis = np.column_stack([np.cos(w * t[seg]), np.sin(w * t[seg]), np.ones(n_meas)])
        (pa, pb, _dc), *_ = np.linalg.lstsq(basis, P[seg], rcond=None)
        chi_meas = (pa - 1j * pb) / EPS_0          # P = Re[(A - jB) e^{jwt}]
        chi_pred = de.eps_numerical_ade(f, model, params, dt) - params["eps_inf"]
        chi_cont = de.eps_analytic(f, model, params) - params["eps_inf"]
        # 2e-5: the live coefficients are float32 (x64 off), so ~1e-6 is the floor.
        assert abs(chi_meas - chi_pred) / abs(chi_pred) < 2e-5, (arm, f, chi_meas, chi_pred)
        # and the discrete form is what makes it agree: the continuous form
        # is measurably further away at the band top than the discrete one.
        assert abs(chi_meas - chi_pred) <= abs(chi_meas - chi_cont) + 1e-12


def test_ade_window_term_is_named_and_small_at_this_dt():
    f, dt = _rfx_bins()
    g = G.gated_mask(f)
    for arm in G.ARM_ORDER:
        model, params = G.ARMS[arm]["model"], G.ARMS[arm]["params"]
        wR, wT, _, _ = G.ade_window(f, model, params, dt)
        assert 0 < wR[g].max() < 2e-3 and 0 < wT[g].max() < 2e-3
        assert wR[g].max() < G.W_BIN / 10


@pytest.mark.parametrize("name", sorted(G.FALSIFIERS))
def test_rfx_falsifiers_exceed_the_windows_analytically(name):
    """§6 margins: every F1/F2 defect must fail G2 (band-mean) on R or T and
    the named per-bin failures must exist, before any FDTD is run."""
    f, dt = _rfx_bins()
    arm, model, bad = G.apply_falsifier(name)
    good = G.ARMS[arm]["params"]
    R, T = G.analytic_rt(f, model, good)
    Rb, Tb = G.analytic_rt(f, model, bad)
    e2 = _e2(f, Rb, Tb, model, good, dt)   # defective "measurement" vs the true oracle
    assert not (e2["gates"]["G2_R"] and e2["gates"]["G2_T"]), name
    # margin >= 2x on at least one band-mean window (coin-toss guard)
    ratio = max(e2["mean_dR_gated"] / e2["mean_window_R"], e2["mean_dT_gated"] / e2["mean_window_T"])
    assert ratio >= 2.0, (name, ratio)
    if name != "debye_tau_x2":
        assert e2["n_bins_R_over_window"] + e2["n_bins_T_over_window"] > 0
    else:
        assert e2["n_bins_T_over_window"] >= 60   # 72 in the note; T fails per-bin above ~7 GHz


def test_debye_tau_x1p3_would_be_a_coin_toss_and_is_not_a_falsifier():
    f, dt = _rfx_bins()
    p = G.ARMS["debye"]["params"]
    bad = {**p, "tau": 1.3 * p["tau"]}
    Rb, Tb = G.analytic_rt(f, "debye", bad)
    e2 = _e2(f, Rb, Tb, "debye", p, dt)
    assert e2["mean_dR_gated"] / e2["mean_window_R"] < 1.5
    assert "debye_tau_x1p3" not in G.FALSIFIERS


@pytest.mark.parametrize("name", sorted(G.MEEP_FALSIFIERS))
def test_meep_falsifiers_exceed_the_e4_windows_analytically(name):
    f, dt = _rfx_bins()
    lp = G.ARMS["lorentz"]["params"]
    good = de.to_meep("lorentz", lp, a_m=G.MEEP_A_M)
    bad = G.apply_meep_falsifier(good, name)
    dt_meep = G.MEEP_COURANT * G.DX_M / de.C0
    # A perfect Meep of the WRONG material, against a perfect rfx of the right one.
    eps_bad = np.conj(de.eps_meep_convention(f, bad))
    Rm, Tm = de.tmm_slab_rt(f, eps_bad, G.D_SLAB_M)
    R, T = G.analytic_rt(f, "lorentz", lp)
    e2 = _e2(f, R, T, "lorentz", lp, dt)
    meep_doc = {"freqs_hz": f.tolist(), "R": Rm.tolist(), "T": Tm.tolist(),
                "dt_meep_s": dt_meep, "meep_params": bad, "precheck": {"passed": False}}
    e4 = _e4(e2, meep_doc)
    assert not e4["e4_ok"], name
    assert not (e4["gates"]["G4_mean_R"] and e4["gates"]["G4_mean_T"]), name
    assert not (e4["gates"]["G5_mean_R"] and e4["gates"]["G5_mean_T"]), name
    # and the control: the RIGHT mapping passes the same E4 gates.
    eps_ok = np.conj(de.eps_meep_convention(f, good))
    Ro, To = de.tmm_slab_rt(f, eps_ok, G.D_SLAB_M)
    e4_ok = _e4(e2, {**meep_doc, "R": Ro.tolist(), "T": To.tolist(), "meep_params": good,
                               "precheck": {"passed": True}})
    assert e4_ok["e4_ok"]


# ---------------------------------------------------------------------------
# 2. Artifact replay (skips until the VESSL run lands)
# ---------------------------------------------------------------------------

def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _replay_e2(arm_doc: dict) -> dict:
    return _e2(arm_doc["freqs_hz"], arm_doc["R_rfx"], arm_doc["T_rfx"],
                         arm_doc["model"], arm_doc["params"], arm_doc["dt_s"], tail=arm_doc["tail"])


def _baseline() -> dict:
    p = _RESULTS / "rfx.json"
    if not p.is_file():
        pytest.skip(f"cv22 baseline artifact absent: {p.relative_to(_REPO)} (VESSL run pending)")
    doc = _read(p)
    assert doc["schema"] == "cv22-dispersive-slab/v1"
    if doc.get("smoke"):
        pytest.skip("baseline artifact is a --smoke run, not evidence")
    return doc


def test_baseline_artifact_replays_and_passes_e2_on_all_arms():
    doc = _baseline()
    assert set(doc["arms"]) == set(G.ARM_ORDER), "all three arms must be present"
    assert doc["falsifier"] is None
    for arm, ad in doc["arms"].items():
        assert ad["model"] == G.ARMS[arm]["model"]
        assert ad["params"] == pytest.approx(G.ARMS[arm]["params"])
        assert ad["params_run"] == pytest.approx(G.ARMS[arm]["params"])
        assert ad["run"]["recipe"] == G.RECIPE_R3, "the baseline must be the r3 (derived-record) recipe"
        rec = ad["run"]["record"]
        want = G.derive_record_length(ad["model"], ad["params"], ad["dt_s"], nx_interior=ad["run"]["nx_interior"])
        assert rec["n_steps_min"] == want["n_steps"], arm
        assert ad["run"]["n_steps"] == rec["n_steps"] == rec["n_steps_min"] + rec["extensions"] * G.RECORD_EXTEND_STEPS
        assert rec["n_steps"] <= rec["t_safe_cpml_steps"]
        assert ad["run"]["nx_interior"] >= G.NX_INTERIOR_R3
        assert ad["tail"]["limit"] == G.SETTLING_LIMIT and ad["tail"]["ok"], (arm, ad["tail"])
        # the stored envelope refits to the recorded rate, and the rate is a
        # physical decay (positive, finite)
        assert ad["tail"]["fit_start_step"] == rec["n_pulse_end"] + G.TAIL_WINDOW, "the fit must start after the pulse"
        refit = G.refit_tail(ad["tail"], ad["dt_s"], ad["run"]["n_steps"], rec["n_pulse_end"])
        rate, nb = refit["fitted_rate_scat_refl_1_s"], refit["fitted_rate_blocks"]
        assert nb >= 3 and rate == pytest.approx(ad["tail"]["fitted_rate_scat_refl_1_s"])
        assert refit["fitted_rate_total_trans_1_s"] == pytest.approx(ad["tail"]["fitted_rate_total_trans_1_s"])
        assert np.isfinite(rate) and rate > 0
        print(f"r4-summary rfx {arm}: n_steps_min {rec['n_steps_min']} reached {rec['n_steps']} "
              f"(+{rec['extensions']} ext, box grows {len(rec.get('nx_grows', []))}); tail scat/trans "
              f"{ad['tail']['scat_refl_rel']:.2e}/{ad['tail']['total_trans_rel']:.2e}; fitted rate scat/trans "
              f"{ad['tail']['fitted_rate_scat_refl_1_s']:.3e}/{ad['tail']['fitted_rate_total_trans_1_s']:.3e} /s "
              f"vs derived ring rate {rec['rate_ring_1_s']:.3e} (component {rec['f_ring_hz']/1e9:.2f} GHz)")
        assert ad["band_inc_ok"]
        re = _replay_e2(ad)
        assert re["gates"] == {k: v for k, v in ad["gates"].items() if k in re["gates"]}, arm
        assert re["e2_ok"], (arm, re["gates"], re["max_dR_gated"], re["max_dT_gated"])
        assert abs(re["mean_dR_gated"] - ad["mean_dR_gated"]) < 1e-12
        assert re["n_bins_gated"] >= 100
    assert doc["verdict"]["rfx_self_ok"]


def test_baseline_artifact_e4_against_the_committed_meep_jsons():
    doc = _baseline()
    missing = [arm for arm in G.ARM_ORDER if not (_RESULTS / G.meep_json_name(arm)).is_file()]
    if missing:
        pytest.skip(f"Meep JSON absent for {missing}; E4 not replayable (exit 2 class)")
    for arm, ad in doc["arms"].items():
        md = _read(_RESULTS / G.meep_json_name(arm))
        assert md["falsifier"] is None and md["arm"] == arm
        assert md["resolution"] == G.MEEP_PRIMARY_RESOLUTION == 40, "§12: the converged Meep reference"
        assert md["run"]["finite"]
        assert md["precheck"]["passed"], (arm, md["precheck"]["max_rel_err"])
        assert md["precheck"]["max_rel_err"] < 1e-9
        e4 = _e4(_replay_e2(ad), md)
        assert e4["e4_ok"], (arm, e4["gates"], e4["max_dR_rfx_meep_gated"], e4["max_dT_rfx_meep_gated"])
        assert e4["gates"]["precheck_passed"]
        # the committed r4 artifact predates the precheck_passed gate key
        # (review finding 2); every stored gate must still replay identically
        assert ad["meep"]["present"]
        assert {k: v for k, v in e4["gates"].items() if k in ad["meep"]["gates"]} == ad["meep"]["gates"]
    assert doc["verdict"]["exit_code"] == 0


@pytest.mark.parametrize("name", sorted(G.FALSIFIERS))
def test_rfx_falsifier_artifacts_fail_for_the_declared_reason(name):
    p = _RESULTS / G.rfx_json_name(name)
    if not p.is_file():
        pytest.skip(f"falsifier artifact absent: {p.name}")
    doc = _read(p)
    assert doc["falsifier"] == name and not doc.get("smoke")
    arm = G.FALSIFIERS[name][0]
    ad = doc["arms"][arm]
    # The artifact records the DEFECTIVE params the FDTD ran with (params_run)
    # and the DECLARED material it was judged against (params); replay both.
    _, model, bad = G.apply_falsifier(name)
    assert ad["params_run"] == pytest.approx(bad)
    assert ad["params"] == pytest.approx(G.ARMS[arm]["params"])
    re = _e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], model, G.ARMS[arm]["params"],
                       ad["dt_s"], tail=ad["tail"])
    assert re["gates"] == {k: v for k, v in ad["gates"].items() if k in re["gates"]}
    assert not re["e2_ok"]
    assert not (re["gates"]["G2_R"] and re["gates"]["G2_T"]), "must fail on the band-mean, not only a witness"
    assert doc["verdict"]["exit_code"] == 1


@pytest.mark.parametrize("name", sorted(G.MEEP_FALSIFIERS))
def test_meep_falsifier_artifacts_fail_e4_against_the_baseline(name):
    mp_path = _RESULTS / G.meep_json_name(G.MEEP_FALSIFIER_ARM, name)
    if not mp_path.is_file():
        pytest.skip(f"Meep falsifier artifact absent: {mp_path.name}")
    doc = _baseline()
    md = _read(mp_path)
    assert md["falsifier"] == name
    assert md["precheck"]["passed"] is False, "the 1e-9 pre-check must have caught the wrong convention"
    e4 = _e4(_replay_e2(doc["arms"][G.MEEP_FALSIFIER_ARM]), md)
    assert not e4["e4_ok"], e4["gates"]
    assert not (e4["gates"]["G4_mean_R"] and e4["gates"]["G4_mean_T"])


# ---------------------------------------------------------------------------
# 3. Round-2 replay (note section 11.2): scaling ratios are COMPUTED and
#    PRINTED against the pre-declared predictions; only structural
#    acceptances are asserted. Skips until the r2 artifacts exist.
# ---------------------------------------------------------------------------

_R2_RFX = {"lorentz": ["lorentz_dx2", "lorentz_dx4", "lorentz_nx1500"],
           "debye": ["debye_dx2", "debye_dx4"]}
_R2_MEEP_TAGS = ["res20", "res40", "decay1e-6", "src7"]


def _r2_rfx_doc(tag: str) -> dict:
    p = _RESULTS / f"rfx__{tag}.json"
    if not p.is_file():
        pytest.skip(f"r2 artifact absent: {p.name}")
    return _read(p)


@pytest.mark.parametrize("arm", sorted(_R2_RFX))
def test_r2_rfx_refinement_ratios_against_predeclared_predictions(arm):
    base = _baseline()["arms"][arm]
    m0 = base["mean_dR_gated"]
    lines = [f"r2-summary rfx {arm}: baseline mean|dR| {m0:.4f} mean|dT| {base['mean_dT_gated']:.4f} "
             f"(window {base['mean_window_R']:.4f}/{base['mean_window_T']:.4f})"]
    for tag in _R2_RFX[arm]:
        p = _RESULTS / f"rfx__{tag}.json"
        if not p.is_file():
            lines.append(f"r2-summary rfx {tag}: ABSENT")
            continue
        d = _read(p)["arms"][arm]
        assert d["params"] == pytest.approx(G.ARMS[arm]["params"])
        re = _replay_e2(d)                      # same evaluators, its own dt
        assert re["gates"] == {k: v for k, v in d["gates"].items() if k in re["gates"]}
        ratio = d["mean_dR_gated"] / m0 if m0 > 0 else float("nan")
        reading = ("first-order" if 0.20 <= ratio <= 0.35 and tag.endswith("dx4") else
                   "second-order" if ratio <= 0.12 and tag.endswith("dx4") else
                   "no-fall" if ratio >= 0.7 else "unresolved")
        lines.append(f"r2-summary rfx {tag}: dx_div={d['run']['dx_div']} nx={d['run']['nx_interior']} "
                     f"n_steps={d['run']['n_steps']} mean|dR| {d['mean_dR_gated']:.4f} (x{ratio:.2f}) "
                     f"mean|dT| {d['mean_dT_gated']:.4f} max|dR| {d['max_dR_gated']:.4f} "
                     f"E2 {'PASS' if re['e2_ok'] else 'FAIL'}"
                     + (f" -> {reading}" if tag.endswith("dx4") else ""))
    print("\n".join(lines))


@pytest.mark.parametrize("arm", ["lorentz", "drude"])
def test_r2_meep_diagnostics_against_predeclared_predictions(arm):
    base_path = _RESULTS / G.meep_json_name(arm)
    if not base_path.is_file():
        pytest.skip("baseline Meep leg absent")
    doc = _baseline()["arms"][arm]
    e2 = _replay_e2(doc)
    e4b = _e4(e2, _read(base_path))
    lines = [f"r2-summary meep {arm}: baseline (10 px/cm) Meep-vs-TMM mean|dR| {e4b['mean_dR_meep_tmm_gated']:.4f} "
             f"mean|dT| {e4b['mean_dT_meep_tmm_gated']:.4f}"]
    for tag in _R2_MEEP_TAGS:
        p = _RESULTS / f"meep_{arm}__{tag}.json"
        if not p.is_file():
            lines.append(f"r2-summary meep {arm}__{tag}: ABSENT")
            continue
        md = _read(p)
        assert md["run"]["finite"] and md["precheck"]["passed"], tag
        e4 = _e4(e2, md)
        rT = e4["mean_dT_meep_tmm_gated"] / max(e4b["mean_dT_meep_tmm_gated"], 1e-12)
        lines.append(f"r2-summary meep {arm}__{tag}: res={md['resolution']} decay={md['decay']} "
                     f"fcen={md['fcen_meep']:.3f} Meep-vs-TMM mean|dR| {e4['mean_dR_meep_tmm_gated']:.4f} "
                     f"mean|dT| {e4['mean_dT_meep_tmm_gated']:.4f} (T x{rT:.2f}) max|dT| {e4['max_dT_meep_tmm_gated']:.4f} "
                     f"G4_mean_T={'pass' if e4['gates']['G4_mean_T'] else 'FAIL'}")
    print("\n".join(lines))


def test_r2_meep_debye_res40_primary_is_the_predeclared_fix():
    """Section 11.2(c) acceptance: the primary Debye leg is the 40 px/cm one,
    ran finite, passed the 1e-9 pre-check, and its mapped pole sits where the
    note says (omega_n dt = 0.262, eps_num(Nyquist) > 1)."""
    p = _RESULTS / G.meep_json_name("debye")
    if not p.is_file():
        pytest.skip("Meep Debye leg absent")
    md = _read(p)
    assert md["resolution"] == 40 and md["run"]["finite"] and md["precheck"]["passed"]
    assert md["fn_debye_map_hz"] == pytest.approx(G.DEBYE_MEEP_MAP_FN_HZ)
    x = 2 * np.pi * md["fn_debye_map_hz"] * md["dt_meep_s"]
    assert x == pytest.approx(0.262, abs=0.005)
    dp = G.ARMS["debye"]["params"]
    eps_nyq = dp["eps_inf"] - dp["delta_eps"] * x ** 2 / (4 - x ** 2)
    assert eps_nyq > 1.0
    print(f"r2-summary meep debye primary: res 40, omega_n dt {x:.3f}, eps_num(Nyq) {eps_nyq:.3f}, "
          f"precheck max_rel_err {md['precheck']['max_rel_err']:.2e}")
    q = _RESULTS / "meep_debye__fn40.json"
    if q.is_file():
        m40 = _read(q)
        assert m40["run"]["finite"]
        e4 = _e4(_replay_e2(_baseline()["arms"]["debye"]), m40)
        print(f"r2-summary meep debye fn40 cross-check: mean|dT| vs TMM {e4['mean_dT_meep_tmm_gated']:.4f} "
              f"(W_map mean T carried {np.mean(np.asarray(e4['w_map_T'])[np.asarray(_baseline()['arms']['debye']['gated'])]):.2e})")


# ---------------------------------------------------------------------------
# 4. Meep ladder (measured-in-r2 evidence of Meep's first-order term; §12)
# ---------------------------------------------------------------------------

def test_meep_ladder_summary_locks_the_measured_first_order_term():
    p = _RESULTS / "meep_ladder_summary.json"
    if not p.is_file():
        pytest.skip("meep_ladder_summary.json absent")
    summ = _read(p)
    doc = _baseline()
    # The committed summary must be what the committed rungs replay to.
    fresh = G.meep_ladder_summary(str(_RESULTS), doc)
    for arm, v in summ["arms"].items():
        assert v["rungs"].keys() == fresh["arms"][arm]["rungs"].keys()
        for res, rung in v["rungs"].items():
            if rung.get("finite"):
                assert rung["mean_dT_meep_tmm_gated"] == pytest.approx(
                    fresh["arms"][arm]["rungs"][res]["mean_dT_meep_tmm_gated"], rel=1e-9)
    for arm in ("lorentz", "drude"):
        o = summ["arms"][arm]["orders"]
        # first order per doubling (r2 measured 0.94-1.06 across R/T and both doublings)
        for k, val in o.items():
            assert 0.8 <= val <= 1.3, (arm, k, val)
        r40 = summ["arms"][arm]["rungs"]["40"]
        assert r40["mean_dR_meep_tmm_gated"] <= G.W_MEAN_R and r40["mean_dT_meep_tmm_gated"] <= G.W_MEAN_T
    # F-B witness (unconditional, review finding 7): the overdamped Debye pole
    # is unstable at 10 px/cm (eps_num(Nyq) = 0.486 < 1); the committed
    # meep_debye__res10.json is built from the run log (no JSON is written by a
    # leg that raises inside Meep) and the ladder summary carries it as
    # finite = false.
    w = _read(_RESULTS / "meep_debye__res10.json")
    assert w["run"]["finite"] is False and w["rc"] == 1 and w["resolution"] == 10
    assert w["omega_n_dt"] == pytest.approx(1.048, abs=0.002) and w["eps_num_nyq"] == pytest.approx(0.486, abs=0.002)
    assert summ["arms"]["debye"]["rungs"]["10"]["finite"] is False
    print("r3-summary meep ladder:", {a: v["orders"] for a, v in summ["arms"].items()})
