"""Plumbing test for cv06b's GPU-lane build falsifier (#812).

``scripts/diagnostics/cv06b_build_falsifiers.py`` runs three 5,729,080-cell
solves. A crash in its reporting or JSON path AFTER those solves costs the run,
so the non-FDTD half is exercised here with the solve stubbed out.

The input contracts use the production geometry assembly but do not establish
RF accuracy. The reporting test checks that summary keys survive and gate
verdicts remain JSON booleans rather than 1.0/0.0 (``evaluate()`` returns
``np.bool_`` whenever its analytic anchor arrives as ``np.float64``).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "scripts/diagnostics/cv06b_build_falsifiers.py"
FIXTURE = REPO_ROOT / "tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_json_is_written_and_keeps_boolean_gates(tmp_path, monkeypatch):
    mod = _load(BUILDER, "_cv06b_build_falsifiers")
    d = json.loads(FIXTURE.read_text())
    f = np.asarray(d["freqs_ghz"], dtype=float) * 1e9
    s21 = np.asarray(d["s21_mag"], dtype=float)
    z0 = np.full_like(f, float(d["re_z0_median_ohm"]))

    def fake_solve(cv, label, **_):
        # np.float64 anchor on purpose: that is what the real path passes, and
        # it is what makes evaluate()'s gate values np.bool_.
        f_an = np.float64(3.711e9)
        m = cv.evaluate(f, s21, z0, f_an)
        m.update(label=label, solve_s=0.0, stub_len_m=float(cv.STUB_LEN),
                 w_stub_m=float(cv.W_STUB), freqs_hz=f.tolist(),
                 s21_mag=s21.tolist(), re_z0=z0.tolist())
        return m

    monkeypatch.setattr(mod, "solve", fake_solve)
    monkeypatch.setattr("sys.argv", ["x", "--out-dir", str(tmp_path)])
    rc = mod.main()
    assert rc in (0, 1)

    summary = json.loads((tmp_path / "cv06b_build_falsifiers_summary.json")
                         .read_text())
    for key in ("criterion_A_baseline", "stub_1cell", "stub_narrow", "verdict"):
        assert key in summary
    for key in ("err_pct", "bw_ratio", "witness_bins", "notch_depth_db",
                "f_notch_refined_hz", "z0_median_ohm"):
        assert isinstance(summary["criterion_A_baseline"][key], float)
    for v in summary["criterion_A_baseline"]["gates"].values():
        assert isinstance(v, bool)
    for v in summary["verdict"].values():
        assert isinstance(v, bool)
    for key in ("true_shift_pct", "true_shift_bins", "bin_argmin_delta_pct",
                "refined_delta_pct"):
        assert isinstance(summary["stub_1cell"][key], float)
    assert isinstance(summary["stub_narrow"]["G2_fired"], bool)
    # the per-leg dumps must survive too — they carry the raw sweeps
    for label in ("baseline", "stub_1cell", "stub_narrow"):
        leg = json.loads((tmp_path / f"cv06b_falsifier_{label}.json").read_text())
        assert isinstance(leg["gates"]["G2 -10 dB stopband width"], bool)


def test_three_arms_preserve_realized_centre_and_the_unmodified_dimensions():
    mod = _load(BUILDER, "_cv06b_realized_inputs")
    arms = {label: (cv, sim, geometry) for label, cv, sim, geometry in mod.prepare_inputs()}
    baseline_cv, _, baseline = arms["baseline"]
    dx = baseline_cv.DX
    for label in ("stub_1cell", "stub_narrow"):
        _, _, geometry = arms[label]
        assert geometry["plane_z"] == baseline["plane_z"]
        assert geometry["trace_y"] == baseline["trace_y"]
        assert sum(geometry["stub_x"]) == pytest.approx(sum(baseline["stub_x"]), abs=1e-15)
    one = arms["stub_1cell"][2]
    narrow = arms["stub_narrow"][2]
    assert one["stub_w"] == pytest.approx(baseline["stub_w"], abs=1e-15)
    assert one["stub_len"] == pytest.approx(baseline["stub_len"] - dx, abs=1e-15)
    assert narrow["stub_len"] == pytest.approx(baseline["stub_len"], abs=1e-15)
    assert narrow["stub_w"] == pytest.approx(5 * dx, abs=1e-15)
    assert narrow["n_cols"] == 6  # five intervals require six endpoint nodes


@pytest.mark.parametrize("fault", ["discard_bounds", "shift_centre"])
def test_bad_narrow_input_is_rejected_before_any_solve(tmp_path, monkeypatch, fault):
    mod = _load(BUILDER, "_cv06b_reject_bad_input")
    original_load = mod._load

    def load_faulty_case():
        cv = original_load()
        build = cv._build_sim

        def faulty_build(*, stub_x_bounds=None):
            if stub_x_bounds is not None and fault == "shift_centre":
                from rfx.geometry.rasterize_grid import coords_from_uniform_grid

                complete = build(stub_x_bounds=stub_x_bounds)
                nodes = np.asarray(coords_from_uniform_grid(complete._build_grid()).x)
                indices = [int(np.argmin(abs(nodes - value))) for value in stub_x_bounds]
                stub_x_bounds = tuple(float(nodes[i + 1]) for i in indices)
            if fault == "discard_bounds":
                stub_x_bounds = None
            return build(stub_x_bounds=stub_x_bounds)

        cv._build_sim = faulty_build
        return cv

    calls = []
    monkeypatch.setattr(mod, "_load", load_faulty_case)
    monkeypatch.setattr(mod, "solve", lambda *args, **kwargs: calls.append(True))
    monkeypatch.setattr("sys.argv", ["x", "--out-dir", str(tmp_path)])
    reason = "stub width" if fault == "discard_bounds" else "stub centre changed"
    with pytest.raises(RuntimeError, match=reason):
        mod.main()
    assert not calls
    assert not list(tmp_path.glob("cv06b_falsifier_*.json"))


def test_centre_parity_conflict_is_refused_instead_of_silently_shifting(monkeypatch):
    mod = _load(BUILDER, "_cv06b_parity_conflict")
    original_load = mod._load

    def load_even_span():
        cv = original_load()
        # This placement realizes ten intervals on both trace and stub.
        # Its centre is a node, incompatible with an odd five-interval span.
        cv.W_TRACE = cv.W_STUB = 10.7 * cv.DX
        return cv

    monkeypatch.setattr(mod, "_load", load_even_span)
    with pytest.raises(RuntimeError, match="cannot place a 5-cell stub"):
        mod.prepare_inputs()


def test_committed_gpu_summary_records_criterion_a_and_the_falsifier_lane():
    """The committed own-board summary is the evidence for cv06b's criterion
    (A) and for BOTH build-level (B) legs. Pin what it says so a regenerated
    file cannot silently flip a verdict.

    REGENERATED 2026-09-07 on the #931 sheet board (VESSL 369367259191). The
    previous state was VESSL 369367257702 (#812 round 2), measured when the
    trace and stub were one-cell PEC Boxes. Two verdicts flipped, both toward
    MORE falsifier sensitivity, and both are pinned below in their new
    direction rather than relaxed:

      stub_1cell   visible False -> True. Against a true stub-length shift of
                   0.5320 % the refined estimator read 0.1447 % on the Box
                   board (27 % of it -- it could not see a sub-bin change) and
                   reads 0.8228 % on the sheet board: it sees the shift now,
                   and overshoots it by 55 %. The bare bin argmin went
                   0.0 -> 1.6949 %, so it is the quantised estimator that is
                   furthest off, which is the point of the lane.
                   verdict.criterion_B_sub_bin_visible and verdict.all_ok both
                   False -> True.
      stub_narrow  G1 "notch freq vs analytic" True -> False (err_pct
                   0.208 -> 6.439). The deliberately-narrow stub is now caught
                   by G1 as well as G2. The arm exists to show that G2 fires
                   where the depth witness stays blind; a second gate also
                   firing is extra coverage, so what is pinned is that G2
                   fires, the depth witness stays blind, and at least one gate
                   fires -- not that G1 in particular stays silent.

    NO cv06b gate window moved. G1 is still < 4.0 %, G2 still (0.80, 1.20),
    G3 still < 1.0 bin, G4 still (40, 65) ohm. The measurement, its falsified
    pre-declarations and the width-convention finding are in
    validation/crossval/_06b_msl_notch_results/RECOMPUTE.md.
    """
    import json
    from pathlib import Path
    path = (Path(__file__).resolve().parents[2]
            / "validation/crossval/_06b_msl_notch_results/cv06b_build_falsifiers_summary.json")
    s = json.loads(path.read_text())
    a = s["criterion_A_baseline"]
    assert a["all_pass"] is True and all(a["gates"].values())
    assert a["err_pct"] < 4.0 and 0.80 < a["bw_ratio"] < 1.20 and a["witness_bins"] < 1.0

    n = s["stub_narrow"]
    assert n["G2_fired"] is True and n["depth_witness_still_passes"] is True
    assert not all(n["gates"].values()), (
        "the narrow-stub arm is the deliberately-broken one: some gate must "
        "fire on it")

    c = s["stub_1cell"]
    true_pct = abs(c["true_shift_pct"])
    refined_pct = abs(c["refined_delta_pct"])
    bin_pct = abs(c["bin_argmin_delta_pct"])
    assert c["visible"] is True
    assert 0.0 < true_pct < 1.0, "the arm only means something sub-bin"
    # The refined estimator must land nearer the true shift than the bare
    # bin argmin does -- that comparison, not either number alone, is what
    # the sub-bin lane claims.
    assert abs(refined_pct - true_pct) < abs(bin_pct - true_pct)

    assert s["verdict"]["criterion_A"] is True
    assert s["verdict"]["criterion_B_G2_fires_on_narrow_stub"] is True
    assert s["verdict"]["criterion_B_sub_bin_visible"] is True
    assert s["verdict"]["all_ok"] is True
