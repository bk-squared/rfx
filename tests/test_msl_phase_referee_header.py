"""Header/record/contract checks for the openEMS MSL phase referee (#490 lane 2).

``validation/research/msl_phase_referee/openems_msl_phase_referee.py`` is a
COMPARATOR-leg-only, VESSL-only harness (see its module docstring for the
full scope fence): openEMS is not installed in THIS test environment, and
this test does not need it -- it only loads the module and inspects
Python-level data, plus exercises the openEMS-free pure-arithmetic helpers
(``_stage_b_layout``, ``_self_consistency_witness``, ``_check_excitation_
and_trace``, ``_passivity_witness``, ``_non_physical_guard``) directly,
mirroring ``tests/test_coax_two_port_referee_header.py``'s design.

Design (fail-loud-honest, per the coax referee's own precedent): the
reproduce-gate-record tests pass on EITHER the never-run UNRUN placeholder
OR a legitimately-filled state -- they check the record's CONTRACT shape,
not one pinned finding.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
from types import ModuleType
from typing import Final

import numpy as np
import pytest

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_PATH: Final = (
    REPO_ROOT / "validation" / "research" / "msl_phase_referee" / "openems_msl_phase_referee.py"
)
RFX_FIXTURE_PATH: Final = REPO_ROOT / "tests" / "fixtures" / "msl_phase_referee" / "msl_thru_rfx_dx50.json"


def _load_referee_module() -> ModuleType:
    """Load the referee script as a throwaway module (not sys.modules-registered).

    Must succeed WITHOUT openEMS installed -- ``_import_openems`` defers its
    openEMS import into a function specifically so this works.
    """
    assert SCRIPT_PATH.exists(), f"missing referee script {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("_msl_phase_referee", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Module / reproduce-gate-record contract
# ---------------------------------------------------------------------------
def test_module_imports_without_openems():
    module = _load_referee_module()
    assert hasattr(module, "REPRODUCE_GATE_RECORD")


def test_reproduce_gate_record_has_required_fields():
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD
    required_fields = {
        "stage", "tutorial", "do_not_repeat", "geometry", "documented_check",
        "expected_f_notch_an_hz", "gate", "status", "reproduced_f_notch_hz",
        "reproduced_f_notch_dev_pct", "log_path", "vessl_run_id",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"REPRODUCE_GATE_RECORD missing fields: {missing}"

    tutorial = record["tutorial"]
    tutorial_fields = {"repo", "path", "verified_present_on", "verified_via", "submodule_pin_note"}
    missing_tutorial = tutorial_fields - set(tutorial.keys())
    assert not missing_tutorial, f"tutorial sub-record missing fields: {missing_tutorial}"


def test_tutorial_citation_is_verifiable():
    module = _load_referee_module()
    tutorial = module.REPRODUCE_GATE_RECORD["tutorial"]
    assert tutorial["repo"], "tutorial citation needs a repo"
    assert tutorial["path"].endswith(".py") or tutorial["path"].endswith(".m"), (
        "tutorial citation should point at a real source file"
    )
    assert tutorial["verified_present_on"], "tutorial citation needs a verification date"
    assert tutorial["verified_via"], "tutorial citation needs a verification method"
    assert "MSL_NotchFilter" in tutorial["path"]


def test_do_not_repeat_cites_the_dx80_mixed_cell_trap():
    """R1/R2 class: the rebuild must name the specific recorded failure it
    avoids repeating (the dx=80um mixed-cell openEMS non-physical result on
    the identical RO4350B substrate), not just assert dx=50um in the abstract.
    """
    module = _load_referee_module()
    do_not_repeat = module.REPRODUCE_GATE_RECORD["do_not_repeat"]
    assert "build_msl_notch_openems_comparison.py" in do_not_repeat
    assert "3.175" in do_not_repeat or "80" in do_not_repeat
    assert "5.08" in do_not_repeat or "50" in do_not_repeat


def test_reproduce_gate_record_is_committed_unrun_and_self_consistent():
    """Fail-loud-honest invariant: UNRUN <=> no numbers, no log path. A
    filled (status != 'UNRUN') record's log_path must live under a
    GIT-TRACKED prefix (PR #548 lesson -- .omx/ and docs/research_notes/
    vessl_logs/ are both gitignored and unreadable by a reviewer outside
    the machine that ran the job).
    """
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD

    if record["status"] == "UNRUN":
        assert record["reproduced_f_notch_hz"] is None
        assert record["reproduced_f_notch_dev_pct"] is None
        assert record["log_path"] is None
    else:
        assert record["reproduced_f_notch_hz"] is not None
        assert record["reproduced_f_notch_dev_pct"] is not None
        log_path_str = record["log_path"]
        assert log_path_str, "a filled-in reproduce_gate_record needs a log_path"

        gitignored_prefixes = (".omx/", "docs/research_notes/vessl_logs/")
        tracked_prefixes = ("validation/research/msl_phase_referee/logs/",)
        assert not log_path_str.startswith(gitignored_prefixes), (
            f"log_path {log_path_str!r} lives under a GITIGNORED prefix -- "
            f"a FILLED record needs a log a reviewer OUTSIDE this machine "
            f"can open; use {tracked_prefixes!r} instead (PR #548 lesson)"
        )
        assert log_path_str.startswith(tracked_prefixes), (
            f"log_path {log_path_str!r} must live under a TRACKED prefix {tracked_prefixes!r}"
        )
        log_path = REPO_ROOT / log_path_str
        assert log_path.exists(), (
            f"reproduce_gate_record claims status={record['status']!r} but its "
            f"log_path {log_path} does not exist"
        )


def test_declared_question_and_governance_notes_present():
    module = _load_referee_module()
    assert "reference-plane" in module.DECLARED_QUESTION.lower()
    assert "validation/crossval/" in module.MUST_MOVE_WHEN_VALIDATED
    assert "REPRODUCE_GATE_RECORD" in module.MUST_MOVE_WHEN_VALIDATED


def test_f_notch_an_matches_cv06b_closed_form():
    """Independently recompute the SAME Hammerstad-Jensen quarter-wave-notch
    closed form ``validation/crossval/06b_msl_notch_filter_uniform.py`` uses
    for this identical substrate/trace/stub combination, and check the two
    land on the same value (5 sig figs) -- a regression lock on the
    reproduce-gate's own oracle, and a cross-check that this script did not
    silently diverge from the repo's existing validated formula.
    """
    module = _load_referee_module()
    c0 = 2.998e8
    stub_len = 12.0e-3
    w_trace = 600e-6
    h_sub = 254e-6
    eps_r = 3.66
    u = w_trace / h_sub
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 / u) ** -0.5
    expected = c0 / (4.0 * stub_len * np.sqrt(eps_eff))

    assert abs(module.F_NOTCH_AN_HZ - expected) / expected < 1e-9
    # sanity: this repo's own recorded "fringing-free analytic 3.69 GHz"
    # (docs/agent-memory/rfx-known-issues.md, cv06b/msl_notch_e4 entries)
    assert 3.6e9 < module.F_NOTCH_AN_HZ < 3.8e9
    assert module.REPRODUCE_GATE_RECORD["expected_f_notch_an_hz"] == float(module.F_NOTCH_AN_HZ)


def test_stage_a_gate_band_is_one_sided_low_biased_and_contains_expected():
    module = _load_referee_module()
    gate = module.REPRODUCE_GATE_RECORD["gate"]
    assert gate["f_notch_lo_hz"] < module.F_NOTCH_AN_HZ < gate["f_notch_hi_hz"]
    # one-sided-low-biased: the low margin must be wider than the high
    # margin (precedent: openEMS reads notch frequency LOW vs analytic).
    lo_margin = module.F_NOTCH_AN_HZ - gate["f_notch_lo_hz"]
    hi_margin = gate["f_notch_hi_hz"] - module.F_NOTCH_AN_HZ
    assert lo_margin > hi_margin


def test_stage_a_matches_tutorial_constants():
    """Regression lock on MSL_NotchFilter.py's own literal parameters."""
    module = _load_referee_module()
    assert module.A_MSL_LENGTH_UM == 50000.0
    assert module.A_MSL_WIDTH_UM == 600.0
    assert module.A_SUBSTRATE_THICKNESS_UM == 254.0
    assert module.A_SUBSTRATE_EPR == 3.66
    assert module.A_STUB_LENGTH_UM == 12.0e3
    assert module.A_F_MAX_HZ == 7.0e9
    assert module.A_N_FREQS == 1601


# ---------------------------------------------------------------------------
# rfx fixture loading + layout arithmetic (openEMS-free)
# ---------------------------------------------------------------------------
def test_load_rfx_fixture_missing_raises_filenotfound(tmp_path):
    module = _load_referee_module()
    with pytest.raises(FileNotFoundError):
        module._load_rfx_fixture(str(tmp_path / "does_not_exist.json"))


def test_rfx_fixture_is_committed_and_has_required_schema():
    """The committed, regenerable rfx-side reference fixture must exist and
    carry the fields Stage B's layout/comparison code reads."""
    assert RFX_FIXTURE_PATH.exists(), (
        f"missing committed rfx fixture {RFX_FIXTURE_PATH} -- regenerate "
        f"with scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py"
    )
    fixture = json.loads(RFX_FIXTURE_PATH.read_text())
    for key in ("meta", "reference_plane_geometry", "freqs_hz", "s11", "s21", "beta_first_port"):
        assert key in fixture, f"rfx fixture missing top-level key {key!r}"
    geom = fixture["reference_plane_geometry"]
    for port_name in ("msl_0", "msl_1"):
        assert port_name in geom
        for field in ("feed_x_m", "direction", "probe0_x_m"):
            assert field in geom[port_name]
    assert "_grid" in geom
    grid = geom["_grid"]
    for field in ("dx", "pad_x_lo", "pad_x_hi", "pad_y_lo", "pad_z_hi", "rasterized_clear_m"):
        assert field in grid


def test_assert_matches_rfx_fixture_accepts_the_committed_fixture():
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    module._assert_matches_rfx_fixture(fixture)  # must not raise


def test_assert_matches_rfx_fixture_rejects_drift():
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    drifted = json.loads(json.dumps(fixture))
    drifted["meta"]["dx_m"] = fixture["meta"]["dx_m"] * 2.0
    with pytest.raises(AssertionError):
        module._assert_matches_rfx_fixture(drifted)


def test_stage_b_layout_is_self_consistent_and_openems_free():
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)  # must not raise

    required = {
        "lx_clear_m", "ly_clear_m", "lz_clear_m", "pml_x_m", "pml_y_m", "pml_z_hi_m",
        "feed_x_port0_m", "feed_x_port1_m", "probe0_x_port0_m", "probe0_x_port1_m",
        "ref_plane_shift_port0_m", "ref_plane_shift_port1_m", "l12_m",
    }
    assert required <= set(layout.keys())
    assert layout["feed_x_port0_m"] < layout["feed_x_port1_m"]
    assert layout["probe0_x_port0_m"] < layout["probe0_x_port1_m"]
    assert layout["l12_m"] > 0.0
    # ref_plane_shift is a positive magnitude for BOTH ports (mirror-
    # symmetric layout, module docstring CONVENTION NOTE part (c)).
    assert layout["ref_plane_shift_port0_m"] > 0.0
    assert layout["ref_plane_shift_port1_m"] > 0.0


def test_ref_plane_shift_transform_independently_rederived():
    """The (target - start) * direction transform, independently re-derived
    against a SYNTHETIC geometry dict (not the real fixture) -- proves the
    formula itself, not merely that it runs on today's fixture numbers.
    Matches the discipline the waveguide Lane-1 phase gate's own convention
    note used ('independently re-derived in review to 1.1e-14 deg').
    """
    module = _load_referee_module()
    synthetic_fixture = {
        "meta": {
            "eps_r": module.B_EPS_R, "h_sub_m": module.B_H_SUB_M,
            "w_trace_m": module.B_W_TRACE_M, "l_line_m": module.B_L_LINE_M,
            "port_margin_m": module.B_PORT_MARGIN_M, "dx_m": module.B_DX_M,
            "f_max_hz": module.B_F_MAX_HZ,
        },
        "reference_plane_geometry": {
            "msl_0": {"feed_x_m": 0.002, "direction": "+x", "probe0_x_m": 0.0045},
            "msl_1": {"feed_x_m": 0.012, "direction": "-x", "probe0_x_m": 0.0095},
            "_grid": {
                "dx": module.B_DX_M, "pad_x_lo": 8, "pad_x_hi": 8,
                "pad_y_lo": 8, "pad_y_hi": 8, "pad_z_lo": 0, "pad_z_hi": 8,
                "rasterized_clear_m": {"x": 0.01405, "y": 0.0025, "z": 0.00185},
            },
        },
    }
    layout = module._stage_b_layout(synthetic_fixture)
    # port0: direction +1, start=2.0mm, target=4.5mm -> shift=(4.5-2.0)*1=2.5mm
    assert abs(layout["ref_plane_shift_port0_m"] - 0.0025) < 1e-12
    # port1: direction -1, start=12.0mm, target=9.5mm -> shift=(9.5-12.0)*-1=2.5mm
    assert abs(layout["ref_plane_shift_port1_m"] - 0.0025) < 1e-12
    assert abs(layout["l12_m"] - 0.005) < 1e-12

    # Discrimination check: a sign error in the transform must NOT
    # coincidentally reproduce these exact numbers.
    wrong_shift_port1 = (0.0095 - 0.012) * 1.0  # forgetting the direction sign
    assert abs(wrong_shift_port1 - 0.0025) > 1e-6, (
        "sign-error variant coincidentally matches the correct shift -- "
        "this discrimination check is not testing anything"
    )


# ---------------------------------------------------------------------------
# Sanity-check helpers (scale-free excitation guard, passivity, non-physical,
# self-consistency witness) -- all openEMS-free, synthetic inputs.
# ---------------------------------------------------------------------------
def test_check_excitation_and_trace_is_scale_free_not_an_absolute_floor():
    """Mirrors the coax lane's own PR #547 lesson: no absolute floor, only
    exact-zero/non-finite is a defect signature."""
    module = _load_referee_module()

    class _FakePort:
        U_filenames: list = []

        def __init__(self, uf_inc):
            self.uf_inc = np.array([uf_inc + 0j])

    tiny_but_real = _FakePort(1e-14)
    peak, _ = module._check_excitation_and_trace(tiny_but_real, "/tmp/_unused", "tiny")
    assert peak == 1e-14

    for broken_value in (0.0, float("nan")):
        broken = _FakePort(broken_value)
        with pytest.raises(RuntimeError):
            module._check_excitation_and_trace(broken, "/tmp/_unused", "broken")

    assert not hasattr(module, "_EXCITATION_ENERGY_FLOOR")


def test_non_physical_guard_raises_above_two():
    module = _load_referee_module()
    module._non_physical_guard(np.array([0.1, 0.9, 1.5]), "ok")  # must not raise
    with pytest.raises(RuntimeError):
        module._non_physical_guard(np.array([0.1, 2.5]), "bad")
    with pytest.raises(RuntimeError):
        module._non_physical_guard(np.array([0.1, float("nan")]), "nan")


def test_passivity_witness_raises_above_tolerance():
    module = _load_referee_module()
    s11 = np.array([0.1, 0.1])
    s21_ok = np.array([0.9, 0.9])
    result = module._passivity_witness(s11, s21_ok, "ok")
    assert result["passed"] is True

    s21_bad = np.array([1.5, 1.5])
    with pytest.raises(RuntimeError):
        module._passivity_witness(s11, s21_bad, "bad")


def test_self_consistency_witness_passes_on_a_synthetic_matched_wave():
    """Feed S21 = exp(-j*beta*L) (a perfectly matched, lossless, non-
    dispersive line by construction) with beta EXACTLY equal to what the
    witness is told -- must pass with ~zero phase deviation, proving the
    witness's own sign convention matches the module's stated
    E(x,t)=Re[exp(j(wt-beta*x))] forward-wave convention (CONVENTION NOTE
    part (a))."""
    module = _load_referee_module()
    freqs_hz = np.linspace(3.0e9, 4.5e9, 9)
    l12_m = 5.0e-3
    beta = 2.0 * np.pi * freqs_hz * np.sqrt(2.9) / 2.998e8
    s21 = np.exp(-1j * beta * l12_m)

    result = module._self_consistency_witness(
        freqs_hz, s21, beta, l12_m=l12_m, mag_band=(0.5, 1.1),
        phase_tol_deg=1.0, gd_tol_ps=50.0, label="synthetic",
    )
    assert result["passed"] is True
    assert result["max_phase_dev_deg"] < 1e-6


def test_self_consistency_witness_fails_on_wrong_sign_beta():
    """Discrimination check: flipping the SIGN of beta (the exact class of
    bug a reference-plane/convention error would introduce) must fail the
    witness, proving it actually catches a convention regression rather
    than passing unconditionally."""
    module = _load_referee_module()
    freqs_hz = np.linspace(3.0e9, 4.5e9, 9)
    l12_m = 5.0e-3
    beta = 2.0 * np.pi * freqs_hz * np.sqrt(2.9) / 2.998e8
    s21 = np.exp(-1j * beta * l12_m)

    with pytest.raises(RuntimeError):
        module._self_consistency_witness(
            freqs_hz, s21, -beta, l12_m=l12_m, mag_band=(0.5, 1.1),
            phase_tol_deg=1.0, gd_tol_ps=50.0, label="wrong_sign",
        )


# ---------------------------------------------------------------------------
# main() exit-code contract
# ---------------------------------------------------------------------------
def test_openems_unavailable_exits_2():
    module = _load_referee_module()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("CSXCAD.CSXCAD", "openEMS.openEMS", "openEMS.ports", "CSXCAD", "openEMS"):
            raise ImportError("simulated: openEMS not installed")
        return real_import(name, *args, **kwargs)

    orig = builtins.__import__
    builtins.__import__ = fake_import
    try:
        rc = module.main(["--rfx-fixture", str(RFX_FIXTURE_PATH)])
    finally:
        builtins.__import__ = orig
    assert rc == 2


def test_missing_rfx_fixture_exits_3():
    module = _load_referee_module()
    rc = module.main(["--rfx-fixture", "/tmp/_definitely_missing_msl_phase_fixture.json"])
    assert rc == 3


def test_config_error_exits_3_distinct_from_physics_failure(monkeypatch):
    module = _load_referee_module()

    def _broken_layout(fixture):
        raise AssertionError("simulated: rfx geometry drifted")

    monkeypatch.setattr(module, "_stage_b_layout", _broken_layout)
    rc = module.main(["--rfx-fixture", str(RFX_FIXTURE_PATH)])
    assert rc == 3


def test_main_writes_valid_json_on_stage_b_physics_gate_failure(monkeypatch, tmp_path):
    """Forensics pattern (coax lane B3 fix): a Stage B physics-gate failure
    must still leave a JSON artifact a reviewer can open, carrying whatever
    partial data was computed before the guard tripped."""
    module = _load_referee_module()

    def fake_stage_a(*, sim_root, threads):
        return {
            "passed": True, "f_notch_hz": module.F_NOTCH_AN_HZ, "notch_depth_db": -30.0,
            "f_notch_expected_hz": module.F_NOTCH_AN_HZ, "f_notch_dev_pct": 0.0,
            "f_notch_ok": True, "max_uf_inc": 1e-12, "n_trace_samples": 100,
            "truncated_suspected": False, "elapsed_s": 1.0,
        }

    def fake_stage_b(*, sim_root, threads, nrts, end_criteria, rfx_fixture):
        exc = RuntimeError("[stage_b_s11] simulated non-physical field")
        exc.partial_stage_b_data = {
            "s11_mag": [300.0], "s21_mag": [0.9], "drive_diagnostics": {"max_uf_inc": 1e-12},
        }
        raise exc

    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))
    monkeypatch.setattr(module, "_run_stage_a_reproduce_gate", fake_stage_a)
    monkeypatch.setattr(module, "_run_stage_b", fake_stage_b)

    out_path = tmp_path / "out.json"
    rc = module.main(["--output", str(out_path), "--rfx-fixture", str(RFX_FIXTURE_PATH)])

    assert rc == 1, f"expected exit 1 (physics-gate failure), got {rc}"
    assert out_path.exists()

    artifact = json.loads(out_path.read_text())
    assert artifact["error"]
    assert "stage_b_partial" in artifact
    assert artifact["stage_b_partial"]["s11_mag"] == [300.0]


def test_main_writes_valid_json_on_overall_pass(monkeypatch, tmp_path):
    module = _load_referee_module()

    def fake_stage_a(*, sim_root, threads):
        return {
            "passed": True, "f_notch_hz": module.F_NOTCH_AN_HZ, "notch_depth_db": -30.0,
            "f_notch_expected_hz": module.F_NOTCH_AN_HZ, "f_notch_dev_pct": 0.0,
            "f_notch_ok": True, "max_uf_inc": 1e-12, "n_trace_samples": 100,
            "truncated_suspected": False, "elapsed_s": 1.0,
        }

    def fake_stage_b(*, sim_root, threads, nrts, end_criteria, rfx_fixture):
        return {
            "s21_mag": [0.95, 0.96], "s11_mag": [0.05, 0.06],
            "self_consistency_openems": {"passed": True},
            "self_consistency_rfx": {"passed": True},
            "sanity_passed": True,
        }

    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))
    monkeypatch.setattr(module, "_run_stage_a_reproduce_gate", fake_stage_a)
    monkeypatch.setattr(module, "_run_stage_b", fake_stage_b)

    out_path = tmp_path / "out_pass.json"
    rc = module.main(["--output", str(out_path), "--rfx-fixture", str(RFX_FIXTURE_PATH)])

    assert rc == 0
    artifact = json.loads(out_path.read_text())
    assert artifact["overall_passed"] is True
    assert artifact["stage_b"]["sanity_passed"] is True


def test_freqs_grid_shared_with_rfx_fixture_no_interpolation():
    """Stage B must evaluate openEMS's own CalcPort at the EXACT frequency
    grid the committed rfx fixture used (module docstring CONVENTION NOTE
    'UNWRAPPING') -- pinned here by checking the fixture's own freqs_hz
    against the constants Stage B is built from (F_MAX/N points match)."""
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    freqs = np.asarray(fixture["freqs_hz"])
    assert freqs[-1] == pytest.approx(module.B_F_MAX_HZ, rel=1e-9)
    assert freqs.min() > 0.0
    assert np.all(np.diff(freqs) > 0), "fixture freqs_hz must be strictly increasing"


# ---------------------------------------------------------------------------
# M1 review fix: gate must be PROVEN to discriminate a realistic referral
# defect (the #529 resolving-power pattern), not just pass a clean run.
# ---------------------------------------------------------------------------
def _rfx_fixture_s21_beta_l12(module):
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    freqs = np.asarray(fixture["freqs_hz"], dtype=float)
    s21 = np.asarray([complex(re, im) for re, im in fixture["s21"]], dtype=np.complex128)
    beta = np.asarray([complex(re, im) for re, im in fixture["beta_first_port"]], dtype=np.complex128)
    layout = module._stage_b_layout(fixture)
    return freqs, s21, beta, layout


def test_gate_budget_is_derived_not_a_round_number():
    """Regression lock on the M1 fix itself: the phase gate must be the
    derived, tight value (3 deg), not the pre-review 30 deg (which admitted
    a 3.372mm/67%-of-L12 plane error and let a real referral-drop defect
    pass -- see the module docstring 'GATE-BUDGET DERIVATION')."""
    module = _load_referee_module()
    assert module.B_PHASE_TOL_DEG == pytest.approx(3.0)
    assert module.B_PHASE_TOL_DEG < 10.0, (
        "phase gate drifted back toward the refuted 30 deg round number"
    )


def test_self_consistency_witness_resolving_power_single_port_referral_drop():
    """The #529 resolving-power pattern: a gate must be PROVEN to
    discriminate a real defect on committed data, not just documented as
    doing so. Plants a synthetic single-port referral drop -- S21 rotated
    by exp(+j*beta*ref_shift) for ONE port's own ref_plane_shift (2.5mm on
    this fixture) -- on the REAL committed rfx fixture's own S21/beta, and
    asserts:
      (a) the UNPERTURBED fixture passes the current B_PHASE_TOL_DEG gate
          with wide margin (measured max_phase_dev_deg ~0.12, ~25x under
          the 3 deg gate);
      (b) the PERTURBED (defect-planted) fixture REDS the same gate.
    This is what makes "the gate resolves a referral-plane error of this
    size" a MEASURED claim, not an assertion -- the pre-review 30 deg gate
    passed this exact perturbation (measured ~16-22 deg, well under 30),
    which is the defect this review caught (module docstring 'GATE-BUDGET
    DERIVATION').
    """
    module = _load_referee_module()
    freqs, s21, beta, layout = _rfx_fixture_s21_beta_l12(module)

    # (a) nominal fixture: must PASS with real margin under the current gate.
    result_nominal = module._self_consistency_witness(
        freqs, s21, beta, l12_m=layout["l12_m"], mag_band=module.B_S21_MAG_BAND,
        phase_tol_deg=module.B_PHASE_TOL_DEG, gd_tol_ps=module.B_GD_TOL_PS,
        label="resolving_power_nominal",
    )
    assert result_nominal["passed"] is True
    assert result_nominal["max_phase_dev_deg"] < module.B_PHASE_TOL_DEG / 5.0, (
        f"nominal fixture's own phase deviation "
        f"({result_nominal['max_phase_dev_deg']:.3f} deg) does not leave "
        f"the expected wide margin under the gate ({module.B_PHASE_TOL_DEG} deg) "
        f"-- the resolving-power claim needs BOTH a passing nominal case "
        f"AND a reding perturbed case to mean anything"
    )

    # (b) planted defect: ONE port's own ref_plane_shift (2.5mm) is
    # rotated back in as if it had never been applied.
    ref_shift_one_port_m = layout["ref_plane_shift_port0_m"]
    beta_re = np.real(beta)
    s21_perturbed = s21 * np.exp(1j * beta_re * ref_shift_one_port_m)

    with pytest.raises(RuntimeError) as excinfo:
        module._self_consistency_witness(
            freqs, s21_perturbed, beta, l12_m=layout["l12_m"], mag_band=module.B_S21_MAG_BAND,
            phase_tol_deg=module.B_PHASE_TOL_DEG, gd_tol_ps=module.B_GD_TOL_PS,
            label="resolving_power_planted_defect",
        )
    assert "self-consistency witness failed" in str(excinfo.value)

    # Measure (not just assert-raises) the actual deviation the planted
    # defect produces, over the gate band, so the docstring's own cited
    # "15.99-22.33 deg" claim is a checked number, not folklore.
    mask = (freqs >= module.B_GATE_F_LO_HZ) & (freqs <= module.B_GATE_F_HI_HZ)
    expected_phase = -beta_re * layout["l12_m"]
    measured_phase = np.unwrap(np.angle(s21_perturbed))
    phase_dev_deg = np.degrees(np.angle(np.exp(1j * (measured_phase - expected_phase))))
    max_dev = float(np.max(np.abs(phase_dev_deg[mask])))
    assert 10.0 < max_dev < 35.0, (
        f"planted single-port referral-drop deviation ({max_dev:.2f} deg) "
        f"fell outside the expected ballpark (10-35 deg) -- re-derive the "
        f"docstring's own cited numbers if this genuinely moved"
    )
    assert max_dev > 5.0 * module.B_PHASE_TOL_DEG, (
        f"planted defect ({max_dev:.2f} deg) is not comfortably (5x) past "
        f"the gate ({module.B_PHASE_TOL_DEG} deg) -- the resolving-power "
        f"margin this docstring claims ('~7-10x') would not hold"
    )


def test_gd_gate_is_honestly_inert_for_the_referral_drop_class():
    """M1 honesty fix: the group-delay gate (200ps) must NOT catch the same
    single-port referral-drop defect the phase gate catches -- pins the
    docstring's own 'GROUP-DELAY GATE HONESTY' claim (~14ps for a
    single-port 2.5mm drop) as a checked number instead of prose."""
    module = _load_referee_module()
    freqs, s21, beta, layout = _rfx_fixture_s21_beta_l12(module)

    ref_shift_one_port_m = layout["ref_plane_shift_port0_m"]
    beta_re = np.real(beta)
    s21_perturbed = s21 * np.exp(1j * beta_re * ref_shift_one_port_m)

    omega = 2.0 * np.pi * freqs
    mask = (freqs >= module.B_GATE_F_LO_HZ) & (freqs <= module.B_GATE_F_HI_HZ)
    expected_phase = -beta_re * layout["l12_m"]
    measured_phase = np.unwrap(np.angle(s21_perturbed))
    gd_measured = -np.gradient(measured_phase, omega)
    gd_expected = -np.gradient(expected_phase, omega)
    gd_dev_ps = np.abs(gd_measured[mask] - gd_expected[mask]) * 1e12

    assert float(np.max(gd_dev_ps)) < module.B_GD_TOL_PS, (
        "the planted single-port referral-drop defect unexpectedly TRIPPED "
        "the group-delay gate -- the docstring's inertness claim is wrong, "
        "fix the prose (or, if this is now doing useful work, say so)"
    )
    # Analytic cross-check: dl*sqrt(eps_eff)/c0 for this substrate.
    eps_eff = (module.B_EPS_R + 1.0) / 2.0 + (module.B_EPS_R - 1.0) / 2.0 * (
        1.0 + 12.0 / (module.B_W_TRACE_M / module.B_H_SUB_M)
    ) ** -0.5
    analytic_gd_ps = ref_shift_one_port_m * np.sqrt(eps_eff) / 2.998e8 * 1e12
    assert abs(float(np.mean(gd_dev_ps)) - analytic_gd_ps) < 1.0, (
        f"measured group-delay deviation ({float(np.mean(gd_dev_ps)):.2f} ps) "
        f"does not match the analytic dl*sqrt(eps_eff)/c0 prediction "
        f"({analytic_gd_ps:.2f} ps) -- the docstring's own physical "
        f"explanation would be wrong"
    )


# ---------------------------------------------------------------------------
# m6 review fix: substrate-top mesh line + rasterized cell count.
# ---------------------------------------------------------------------------
def test_stage_b_substrate_z_mesh_rasterized_cell_count():
    """#325 class: the substrate's ACTUAL rasterized cell count must be
    asserted, not left to an unguided uniform arange's arbitrary rounding.
    At this fixture's own h_sub=254um, dx=50um, 254/50=5.08 -> round=5."""
    module = _load_referee_module()
    z_lines, n_sub = module._stage_b_substrate_z_mesh(254.0, 50.0)  # um units
    assert n_sub == 5
    assert z_lines[0] == pytest.approx(0.0)
    assert z_lines[-1] == pytest.approx(254.0)
    assert len(z_lines) == n_sub + 1
    # Cell count must be derived from h_sub/dx, not hard-coded independent
    # of the inputs -- discrimination check with a different dx.
    _, n_sub_alt = module._stage_b_substrate_z_mesh(254.0, 80.0)
    assert n_sub_alt == 3  # round(254/80) = round(3.175) = 3
    assert n_sub_alt != n_sub


def test_build_stage_b_asserts_substrate_cell_count_matches_constants():
    """The wiring inside _build_stage_b (not just the pure helper above)
    must assert n_sub == round(B_H_SUB_M/B_DX_M) -- pinned by reading the
    source for the assertion rather than only exercising the pure
    function, since a future edit could call the helper with the wrong
    arguments and still "pass" the isolated unit test above."""
    module = _load_referee_module()
    import inspect
    src = inspect.getsource(module._build_stage_b)
    assert "_stage_b_substrate_z_mesh" in src
    assert "n_sub == round(B_H_SUB_M / B_DX_M)" in src or "n_sub ==" in src


# ---------------------------------------------------------------------------
# M3/m5/NEW-1/NEW-3 review fixes: Stage B's trace span and MeasPlaneShift
# targets are now sourced from the PURE, openEMS-free
# _stage_b_port_placement -- _build_stage_b calls it directly (wired, not
# duplicated), so these are NUMERIC assertions on the actual values used,
# not inspect.getsource string-counting (the NEW-3 fix).
# ---------------------------------------------------------------------------
def test_stage_b_port_placement_trace_span_is_feed_to_feed_not_pml():
    """M3 fix, numeric: the trace span must be EXACTLY
    (feed_x_port0_m, feed_x_port1_m) -- NOT the pre-review version's
    domain-edge (PML-side) x0/x1 coordinates (the parallel-impedance
    defect this review caught: Feed_R=50 in parallel with a PML-matched
    continuation, predicted |S11|~=0.33)."""
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)
    placement = module._stage_b_port_placement(layout)

    assert placement["trace_x_start_m"] == layout["feed_x_port0_m"]
    assert placement["trace_x_stop_m"] == layout["feed_x_port1_m"]
    # Must NOT be the PML-side domain-edge coordinates (x0 = -pml_x_m,
    # x1 = lx_clear_m + pml_x_m) -- a numeric, not textual, guard against
    # the M3 regression class.
    x0_pml_side = -layout["pml_x_m"]
    x1_pml_side = layout["lx_clear_m"] + layout["pml_x_m"]
    assert placement["trace_x_start_m"] != pytest.approx(x0_pml_side)
    assert placement["trace_x_stop_m"] != pytest.approx(x1_pml_side)
    # Sanity: the trace sits strictly inside the clear (non-PML) region.
    assert 0.0 < placement["trace_x_start_m"] < placement["trace_x_stop_m"] < layout["lx_clear_m"]


def test_stage_b_port_placement_measplaneshift_is_explicit_and_far_from_feed():
    """m5 fix, numeric: MeasPlaneShift's own TARGET must equal this
    fixture's own ref_plane_shift (2.5mm) -- NOT the class default (half
    the port's own 300um span, 150um/3 cells from the excitation+Feed_R,
    near-field contaminated)."""
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)
    placement = module._stage_b_port_placement(layout)

    assert placement["measplane_shift_target_port0_m"] == placement["ref_plane_shift_port0_m"]
    assert placement["measplane_shift_target_port1_m"] == placement["ref_plane_shift_port1_m"]
    # Comfortably (>3x) past the near-field-contaminated class-default
    # radius (half the port's own 300um span = 150um).
    half_port_span_m = 0.5 * module.B_PORT_W_CELLS * module.B_DX_M
    assert placement["measplane_shift_target_port0_m"] > 3.0 * half_port_span_m
    assert placement["measplane_shift_target_port1_m"] > 3.0 * half_port_span_m


def test_predict_measplane_snap_matches_uniform_grid_rounding():
    """_predict_measplane_snap must round to the nearest INTEGER multiple
    of dx -- pinned with both an exact (on-grid) and an off-grid target so
    the rounding behavior itself is checked, not just one lucky case."""
    module = _load_referee_module()
    dx = 50e-6
    assert module._predict_measplane_snap(2.5e-3, dx) == pytest.approx(2.5e-3)  # exact: 50 cells
    assert module._predict_measplane_snap(2.51e-3, dx) == pytest.approx(2.5e-3)  # rounds down
    assert module._predict_measplane_snap(2.53e-3, dx) == pytest.approx(2.55e-3)  # rounds up


def test_stage_b_port_placement_effective_shift_is_zero_by_construction():
    """NEW-1 fix (the one that matters, review 2026-08-04): on the
    COMMITTED fixture, the predicted effective CalcPort shift
    (ref_plane_shift - predicted measplane_shift) must be (numerically)
    ZERO -- proving, not merely asserting in prose, that the referral
    ROTATION is a no-op here and the referee's own phase-plane
    correctness comes from MeasPlaneShift's STENCIL PLACEMENT instead.
    See DECLARED_QUESTION / _stage_b_port_placement's own docstring for
    the full, non-overclaiming statement this test backs."""
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)
    placement = module._stage_b_port_placement(layout)

    assert abs(placement["effective_shift_predicted_port0_m"]) < 1e-12
    assert abs(placement["effective_shift_predicted_port1_m"]) < 1e-12

    # Discrimination check: an OFF-GRID target (not this fixture's own
    # 2.5mm, which lands exactly on a mesh line) must NOT read zero --
    # proving this test can actually tell "no-op" from "real rotation".
    synthetic_layout = dict(layout)
    synthetic_layout["feed_x_port0_m"] = 0.002
    synthetic_layout["ref_plane_shift_port0_m"] = 2.517e-3  # deliberately off-grid
    off_grid_placement = module._stage_b_port_placement(synthetic_layout)
    assert abs(off_grid_placement["effective_shift_predicted_port0_m"]) > 1e-6, (
        "off-grid target unexpectedly read zero shift -- this test is not discriminating"
    )


def test_build_stage_b_asserts_port_start_matches_feed_x():
    """n8 fix: the silent coupling between _stage_b_layout's ref_plane_
    shift derivation (which assumes start[prop]==feed_x) and the actual
    port construction, ~250 lines away, must be an explicit assertion."""
    module = _load_referee_module()
    import inspect
    src = inspect.getsource(module._build_stage_b)
    assert "port0.start[0]" in src and "port1.start[0]" in src, (
        "expected explicit start[prop]==feed_x assertions on both ports (n8 fix)"
    )
