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
from types import SimpleNamespace

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
        m = cv.evaluate(f, s21, z0, f_an, frequency_gate=(label == "baseline"))
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
        if label == "baseline":
            assert "G1 notch freq vs analytic" in leg["gates"]
            assert "frequency_diagnostic" not in leg
        else:
            assert "G1 notch freq vs analytic" not in leg["gates"]
            assert "err_pct" not in leg
            assert isinstance(leg["frequency_diagnostic"]["err_pct"], float)
            assert "G1 notch freq vs analytic" not in summary[label]["gates"]
            assert "err_pct" not in summary[label]
            assert summary[label]["frequency_diagnostic"] == leg["frequency_diagnostic"]


@pytest.mark.parametrize("filename", [
    "cv06b_build_falsifiers_summary.json", "cv06b_falsifier_baseline.json",
    "cv06b_falsifier_stub_1cell.json", "cv06b_falsifier_stub_narrow.json",
])
def test_existing_evidence_is_refused_before_build_or_solve(tmp_path, monkeypatch, filename):
    mod = _load(BUILDER, "_cv06b_preserve_reports")
    path = tmp_path / filename
    original = b'{"retained": "historical spectrum or partial run"}\n'
    path.write_bytes(original)
    monkeypatch.setattr(mod, "prepare_inputs", lambda: pytest.fail("must refuse before building"))
    monkeypatch.setattr("sys.argv", ["x", "--out-dir", str(tmp_path)])
    with pytest.raises(FileExistsError, match="refusing to overwrite retained falsifier evidence"):
        mod.main()
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_frequency_comparison_is_a_gate_only_for_the_baseline(capsys):
    mod = _load(BUILDER, "_cv06b_frequency_scope")
    cv = mod._load()
    data = json.loads(FIXTURE.read_text())
    f = np.asarray(data["freqs_ghz"]) * 1e9
    s21 = np.asarray(data["s21_mag"])
    z0 = np.full_like(f, data["re_z0_median_ohm"])
    reference = 1.25 * f[np.argmin(s21)]
    baseline = cv.evaluate(f, s21, z0, reference)
    assert not baseline["gates"]["G1 notch freq vs analytic"]
    assert cv.NOTCH_FREQ_TOL_PCT == 4.0
    diagnostic = cv.evaluate(f, s21, z0, reference, frequency_gate=False)
    assert "G1 notch freq vs analytic" not in diagnostic["gates"]
    assert "err_pct" not in diagnostic
    assert diagnostic["frequency_diagnostic"]["err_pct"] == baseline["err_pct"]
    assert diagnostic["frequency_diagnostic"]["err_pct"] > 4.0
    cv.report(diagnostic)
    output = capsys.readouterr().out
    assert "diagnostic only" in output and "not an accuracy verdict" in output
    assert "G1 Notch freq" not in output


def test_narrow_frequency_reference_uses_its_geometry_without_rescaling_g2():
    mod = _load(BUILDER, "_cv06b_reference_contract")
    cv = mod._load()
    geometry = dict(trace_w_elec=10 * cv.DX, stub_w=5 * cv.DX,
                    stub_len=cv.STUB_LEN + 1.5e-6)
    baseline = mod.frequency_reference(cv, "baseline", geometry)
    narrow = mod.frequency_reference(cv, "stub_narrow", geometry)
    assert baseline["width_m"] == geometry["trace_w_elec"]
    assert baseline["length_m"] == cv.STUB_LEN
    assert narrow["width_m"] == geometry["stub_w"]
    assert narrow["length_m"] == geometry["stub_len"]
    assert narrow["role"] == "diagnostic_only"
    assert not narrow["discrete_electrical_width_certified"]
    # For this arm W/h=5/4; the independent closed form must use that ratio,
    # not the main-line 10/4 row-pitch ratio or the six-node count.
    eps = (cv.EPS_R + 1) / 2 + (cv.EPS_R - 1) / (2 * np.sqrt(1 + 12 / 1.25))
    expected = cv.C0 / (4 * geometry["stub_len"] * np.sqrt(eps))
    assert narrow["frequency_hz"] == pytest.approx(expected, rel=1e-14)
    assert narrow["frequency_hz"] != pytest.approx(baseline["frequency_hz"], rel=1e-3)
    data = json.loads(FIXTURE.read_text())
    f = np.asarray(data["freqs_ghz"]) * 1e9
    s21 = np.asarray(data["s21_mag"])
    z0 = np.full_like(f, data["re_z0_median_ohm"])
    a = mod.evaluate_arm(cv, "baseline", geometry, f, s21, z0)
    b = mod.evaluate_arm(cv, "stub_narrow", geometry, f, s21, z0)
    assert b["bw_ratio"] == a["bw_ratio"]
    assert b["gates"]["G2 -10 dB stopband width"] == a["gates"]["G2 -10 dB stopband width"]
    assert "G1 notch freq vs analytic" not in b["gates"]
    d = b["frequency_diagnostic"]
    assert d["reference"] == narrow
    assert d["legacy_main_line_reference"]["frequency_hz"] == baseline["frequency_hz"]
    assert d["legacy_main_line_err_pct"] == a["err_pct"]


def test_one_cell_prediction_retains_the_baseline_reference_convention():
    mod = _load(BUILDER, "_cv06b_one_cell_reference")
    cv = mod._load()
    geometry = dict(trace_w_elec=10 * cv.DX, stub_w=9 * cv.DX,
                    stub_len=cv.STUB_LEN + 1.5e-6)
    baseline = mod.frequency_reference(cv, "baseline", geometry)
    original_length = cv.STUB_LEN
    cv.STUB_LEN -= cv.DX
    one = mod.frequency_reference(cv, "stub_1cell", geometry)
    assert one["role"] == "length_shift_prediction"
    assert one["width_convention"] == baseline["width_convention"]
    assert one["width_m"] == baseline["width_m"]
    assert one["length_convention"] == baseline["length_convention"]
    assert one["frequency_hz"] / baseline["frequency_hz"] == pytest.approx(
        original_length / (original_length - cv.DX), rel=1e-14)


@pytest.mark.parametrize("label", ["stub_1cell", "stub_narrow"])
def test_real_solve_reports_frequency_diagnostics_with_only_fields_stubbed(label, capsys):
    mod = _load(BUILDER, "_cv06b_real_solve_reporting")
    cv = mod._load()
    if label == "stub_1cell":
        cv.STUB_LEN -= cv.DX
    else:
        cv.W_STUB = 5 * cv.DX
    data = json.loads(FIXTURE.read_text())
    freqs = np.asarray(data["freqs_ghz"]) * 1e9
    s = np.zeros((2, 2, len(freqs)), dtype=complex)
    s[1, 0] = data["s21_mag"]
    result = SimpleNamespace(freqs=freqs, S=s,
                             Z0=np.full((2, len(freqs)), data["re_z0_median_ohm"]))
    calls = []

    def fields(**kwargs):
        calls.append(kwargs)
        return result

    sim = SimpleNamespace(preflight=lambda **kwargs: None,
                          compute_msl_s_matrix=fields)
    geometry = dict(trace_w_elec=10 * cv.DX, stub_w=5 * cv.DX,
                    stub_len=cv.STUB_LEN + 1.5e-6)
    metrics = mod.solve(cv, label, sim=sim, geometry=geometry)
    persisted = json.loads(json.dumps(mod._plain(metrics)))
    assert calls == [dict(n_freqs=100, num_periods=20.0)]
    assert "G1 notch freq vs analytic" not in persisted["gates"]
    assert "err_pct" not in persisted
    diagnostic = persisted["frequency_diagnostic"]
    assert diagnostic["reference"] == mod.frequency_reference(cv, label, geometry)
    assert diagnostic["f_notch_refined_hz"] == persisted["f_notch_refined"]
    output = capsys.readouterr().out
    assert "diagnostic only" in output and "G1 Notch freq" not in output


def test_three_arms_preserve_realized_centre_and_the_unmodified_dimensions():
    mod = _load(BUILDER, "_cv06b_realized_inputs")
    arms = {label: (cv, sim, geometry) for label, cv, sim, geometry in mod.prepare_inputs()}
    baseline_cv, _, baseline = arms["baseline"]
    dx = baseline_cv.DX
    for label in ("stub_1cell", "stub_narrow"):
        _, sim, geometry = arms[label]
        assert geometry["plane_z"] == baseline["plane_z"]
        assert geometry["trace_y"] == baseline["trace_y"]
        assert sum(geometry["stub_x"]) == pytest.approx(sum(baseline["stub_x"]), abs=1e-15)
        assert geometry["comparison_environment"] == baseline["comparison_environment"]
        assert not sim._msl_auto_offset_min
        assert not sim._msl_auto_probe_spacing
    one = arms["stub_1cell"][2]
    narrow = arms["stub_narrow"][2]
    assert one["stub_w"] == pytest.approx(baseline["stub_w"], abs=1e-15)
    assert one["stub_len"] == pytest.approx(baseline["stub_len"] - dx, abs=1e-15)
    assert narrow["stub_len"] == pytest.approx(baseline["stub_len"], abs=1e-15)
    assert narrow["stub_w"] == pytest.approx(5 * dx, abs=1e-15)
    assert narrow["n_cols"] == 6  # five intervals require six endpoint nodes


def test_explicit_baseline_probe_settings_preserve_its_resolved_inputs():
    mod = _load(BUILDER, "_cv06b_explicit_baseline")
    cv = mod._load()
    automatic = cv._build_sim()
    before = mod.environment_signature(automatic)
    settings = tuple({key: port[key] for key in ("n_probe_offset", "n_probe_spacing", "n_probes")}
                     for port in before["ports"])
    explicit = cv._build_sim(domain_y=automatic._domain[1], probe_settings=settings)
    assert mod.environment_signature(explicit) == before
    assert not explicit._msl_auto_offset_min and not explicit._msl_auto_probe_spacing
    assert all(port.eps_r_sub is None for port in explicit._msl_ports)


def test_fixed_environment_preserves_the_three_measured_metal_geometries():
    mod = _load(BUILDER, "_cv06b_fixed_environment_metal")
    previous = json.loads((REPO_ROOT / "docs/research_notes/issue953/gpu-369367260669/artifacts/plan.json").read_text())
    for label, _, _, geometry in mod.prepare_inputs():
        metal = {key: value for key, value in geometry.items() if key != "comparison_environment"}
        assert mod._plain(metal) == previous["inputs"][label]["realized_metal"]


@pytest.mark.parametrize("fault, expected", [
    ("domain", "grid"), ("substrate", "materials"), ("auto_probes", "automatic probe"),
    ("material_array", "materials"), ("waveform", "ports"), ("cpml", "options"),
])
def test_hidden_comparison_changes_are_rejected_before_any_solve(tmp_path, monkeypatch, fault, expected):
    import dataclasses

    mod = _load(BUILDER, "_cv06b_reject_environment")
    original_load = mod._load

    def load_faulty_case():
        cv = original_load()
        build = cv._build_sim

        def faulty_build(**kwargs):
            # Only perturb the arms that were given explicit baseline controls.
            if not kwargs:
                return build()
            if fault == "domain":
                kwargs.pop("domain_y")
            if fault == "auto_probes":
                kwargs.pop("probe_settings")
            sim = build(**kwargs)
            if fault == "substrate":
                from rfx import Box
                entry = sim._geometry[0]
                # Keep Simulation.domain fixed while silently shortening the
                # actual dielectric carrier, reproducing the hidden dependency.
                old = entry.shape
                hi = tuple(old.corner_hi[i] - (cv.DX if i == 1 else 0) for i in range(3))
                sim._geometry[0] = dataclasses.replace(entry, shape=Box(old.corner_lo, hi))
            elif fault == "material_array":
                assemble = sim._assemble_materials

                def wrong_materials(*args, **kw):
                    values = list(assemble(*args, **kw))
                    mats = values[0]
                    values[0] = mats._replace(eps_r=mats.eps_r.at[10, 10, 10].add(.5))
                    return tuple(values)

                sim._assemble_materials = wrong_materials
            elif fault == "waveform":
                entry = sim._msl_ports[0]
                wave = dataclasses.replace(entry.waveform, amplitude=entry.waveform.amplitude * .9)
                sim._msl_ports[0] = dataclasses.replace(entry, waveform=wave)
            elif fault == "cpml":
                sim._cpml_kappa_max += .25
            return sim

        cv._build_sim = faulty_build
        return cv

    monkeypatch.setattr(mod, "_load", load_faulty_case)
    monkeypatch.setattr(mod, "solve", lambda *a, **kw: pytest.fail("field solve must not start"))
    monkeypatch.setattr("sys.argv", ["x", "--out-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match=expected):
        mod.main()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ["discard_bounds", "shift_centre"])
def test_bad_narrow_input_is_rejected_before_any_solve(tmp_path, monkeypatch, fault):
    mod = _load(BUILDER, "_cv06b_reject_bad_input")
    original_load = mod._load

    def load_faulty_case():
        cv = original_load()
        build = cv._build_sim

        def faulty_build(*, stub_x_bounds=None, **kwargs):
            if stub_x_bounds is not None and fault == "shift_centre":
                from rfx.geometry.rasterize_grid import coords_from_uniform_grid

                complete = build(stub_x_bounds=stub_x_bounds, **kwargs)
                nodes = np.asarray(coords_from_uniform_grid(complete._build_grid()).x)
                indices = [int(np.argmin(abs(nodes - value))) for value in stub_x_bounds]
                stub_x_bounds = tuple(float(nodes[i + 1]) for i in indices)
            if fault == "discard_bounds":
                stub_x_bounds = None
            return build(stub_x_bounds=stub_x_bounds, **kwargs)

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
