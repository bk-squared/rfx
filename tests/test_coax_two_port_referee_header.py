"""Header/record consistency checks for the openEMS coax two-port referee.

``validation/research/coax_two_port/openems_coax_two_port_referee.py`` is a
COMPARATOR-leg-only, VESSL-only harness for rfx issue #489 stage 3 (see its
module docstring for the full scope fence): openEMS is not installed in
this environment, so it has never actually run. This test does NOT require
openEMS -- it checks that the script's reproduce-gate record is committed
in an honest, fail-loud state, that its Stage B layout arithmetic is
self-consistent (testable without openEMS -- see ``_stage_b_layout()``),
and that the exit-code split between a physics/self-check failure and an
internal config bug is real (not just documented).

Design (per the task's fail-loud-honest requirement, M1/M3 review fixes):
this test PASSES on the current, never-run placeholder state. It checks
the record's FIELDS' semantics (UNRUN <=> no numbers, no log path; a
filled record needs a log path under .omx/ or
docs/research_notes/vessl_logs/) rather than pinning a specific finding
as fact -- the record's own content (which tutorial, which geometry) is
free to evolve without this test needing to be rewritten each time.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType
from typing import Final

import numpy as np

REFEREE_DIR: Final = (
    pathlib.Path(__file__).resolve().parent.parent
    / "validation" / "research" / "coax_two_port"
)
SCRIPT_PATH: Final = REFEREE_DIR / "openems_coax_two_port_referee.py"
REPO_ROOT: Final = pathlib.Path(__file__).resolve().parent.parent


def _load_referee_module() -> ModuleType:
    """Load the referee script as a throwaway module (not sys.modules-registered).

    Must succeed WITHOUT openEMS installed -- the referee script defers its
    openEMS import into functions specifically so this works (see its
    ``_import_openems`` docstring).
    """
    assert SCRIPT_PATH.exists(), f"missing referee script {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("_coax_two_port_referee", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_openems():
    """The module itself must not require openEMS at import time."""
    module = _load_referee_module()
    assert hasattr(module, "REPRODUCE_GATE_RECORD")


def test_reproduce_gate_record_has_required_fields():
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD
    required_fields = {
        "stage", "tutorial", "do_not_repeat", "geometry", "documented_check",
        "expected_zl_a_ohm", "gate", "status", "reproduced_zl_mean_ohm",
        "reproduced_zl_max_dev_ohm", "log_path", "vessl_run_id",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"REPRODUCE_GATE_RECORD missing fields: {missing}"

    tutorial = record["tutorial"]
    tutorial_fields = {"repo", "path", "verified_present_on", "verified_via", "submodule_pin_note"}
    missing_tutorial = tutorial_fields - set(tutorial.keys())
    assert not missing_tutorial, f"tutorial sub-record missing fields: {missing_tutorial}"


def test_tutorial_citation_is_verifiable():
    """The reproduce-gate must cite a REAL, checkable tutorial path -- not
    an assertion that no tutorial exists (M1 fix: the first version wrongly
    claimed no canonical openEMS coax tutorial exists; PR #540 review
    found matlab/examples/waveguide/Coax.m). This test checks the citation
    has the shape of something a reviewer could independently verify
    (repo + path + how it was checked + when), not that any PARTICULAR
    tutorial is named -- the record's own content should be free to
    evolve without this test needing a rewrite."""
    module = _load_referee_module()
    tutorial = module.REPRODUCE_GATE_RECORD["tutorial"]
    assert tutorial["repo"], "tutorial citation needs a repo"
    assert tutorial["path"].endswith(".m") or tutorial["path"].endswith(".py"), (
        "tutorial citation should point at a real source file"
    )
    assert tutorial["verified_present_on"], "tutorial citation needs a verification date"
    assert tutorial["verified_via"], "tutorial citation needs a verification method"


def test_do_not_repeat_cites_the_recorded_failure():
    """R1/R2 class: the rebuild must name the specific recorded failure it
    avoids repeating, not just assert a new approach in the abstract."""
    module = _load_referee_module()
    do_not_repeat = module.REPRODUCE_GATE_RECORD["do_not_repeat"]
    assert "build_coaxial_line_openems_broad_comparison.py" in do_not_repeat
    assert "AddLumpedPort" in do_not_repeat


def test_reproduce_gate_record_is_committed_unrun_and_self_consistent():
    """Fail-loud-honest invariant: UNRUN <=> no numbers, no log path.

    This is the test that must go RED if someone later claims reproduced
    numbers without a log path pointing at the run that produced them --
    not a pinned number that rots after the first real VESSL run. M3 fix:
    a filled-in log_path must live under .omx/ or
    docs/research_notes/vessl_logs/ AND actually exist on disk.
    """
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD

    if record["status"] == "UNRUN":
        assert record["reproduced_zl_mean_ohm"] is None
        assert record["reproduced_zl_max_dev_ohm"] is None
        assert record["log_path"] is None
    else:
        assert record["reproduced_zl_mean_ohm"] is not None
        assert record["reproduced_zl_max_dev_ohm"] is not None
        log_path_str = record["log_path"]
        assert log_path_str, "a filled-in reproduce_gate_record needs a log_path"
        assert log_path_str.startswith(".omx/") or log_path_str.startswith(
            "docs/research_notes/vessl_logs/"
        ), (
            f"log_path {log_path_str!r} must live under .omx/ or "
            f"docs/research_notes/vessl_logs/ (M3 fix)"
        )
        log_path = REPO_ROOT / log_path_str
        assert log_path.exists(), (
            f"reproduce_gate_record claims status={record['status']!r} but its "
            f"log_path {log_path} does not exist -- a claimed number needs a "
            f"real log, per external_solver_comparator.md step 2"
        )


def test_declared_question_and_governance_notes_present():
    module = _load_referee_module()
    assert "reference-plane referral" in module.DECLARED_QUESTION
    assert "validation/crossval/" in module.MUST_MOVE_WHEN_VALIDATED
    assert "REPRODUCE_GATE_RECORD" in module.MUST_MOVE_WHEN_VALIDATED


def test_stage_a_matches_coaxm_tutorial_constants():
    """Regression lock on Coax.m's own literal parameters -- a future edit
    that silently drifts Stage A away from the tutorial (defeating the
    whole point of a reproduce-gate) should fail this test."""
    module = _load_referee_module()
    assert module.A_LENGTH_MM == 1000.0
    assert module.A_R_I_MM == 100.0
    assert module.A_R_O_MM == 230.0
    assert module.A_R_OS_MM == 240.0
    assert module.A_MESH_RES_MM == 5.0
    assert module.A_F0_HZ == module.A_FC_HZ == 0.5e9
    assert module.A_N_FREQS == 201
    assert module.A_NUM_TS == 5000
    assert module.A_END_CRITERIA == 1.0e-5
    assert module.A_FEEDSHIFT_MM == 50.0  # 10 * mesh_res(1)


def test_zl_a_analytic_matches_coaxm_formula():
    """Independently recompute Coax.m's own ZL_a = Z0/2/pi/sqrt(epsR)*log(r_o/r_i)
    (epsR=1, r_o=230, r_i=100) and check it matches the module's constant --
    a regression lock on the reproduce-gate's own oracle."""
    module = _load_referee_module()
    eta0 = 376.73031346177066
    expected = (eta0 / (2.0 * np.pi)) * np.log(230.0 / 100.0)
    assert abs(module.ZL_A_ANALYTIC_OHM - expected) < 1e-9
    assert 45.0 < module.ZL_A_ANALYTIC_OHM < 55.0  # sanity: coax Z0 is a ~50ohm-class number
    assert module.REPRODUCE_GATE_RECORD["expected_zl_a_ohm"] == float(module.ZL_A_ANALYTIC_OHM)


def test_geometry_constants_match_the_rasterized_rfx_fixture():
    """Pins the numbers this script's header claims were read off the live
    rfx grid-construction path -- including the RASTERIZED cell-count fix
    (B3 review fix: the nominal domain= argument, not the rasterized
    interior, was the pre-review bug)."""
    module = _load_referee_module()
    assert module.B_A_MM == 0.635
    assert module.B_B_MM == 2.055
    assert module.B_PTFE_EPS_R == 2.1
    assert abs(module.B_DX_MM - 3.7474057249999997e-4 * 1e3) < 1e-9
    assert module.B_CPML_CELLS == 16
    assert abs(module.B_L12_MM - 58.4595293) < 1e-4

    # B3 fix: rasterized interior cell counts, NOT the nominal domain= arg
    # (23 transverse / 162 axial cells at dz=0.37474mm).
    assert module.B_INTERIOR_X_CELLS == 23
    assert module.B_INTERIOR_Z_CELLS == 162
    assert abs(module.B_CLEAR_X_MM - 23 * module.B_DX_MM) < 1e-9
    assert abs(module.B_CLEAR_Z_MM - 162 * module.B_DX_MM) < 1e-9
    # the rasterized value must NOT equal the nominal domain= argument --
    # if it did, the B3 fix would have silently regressed back to the bug.
    assert abs(module.B_CLEAR_X_MM - 8.0) > 0.1
    assert abs(module.B_CLEAR_Z_MM - 60.0) > 0.1

    # Z0 = sqrt(L'/C') closed form must land close to the standard SMA/PTFE
    # value (~48.6 ohm) this repo's other coax lanes all cite.
    assert 48.0 < module.B_Z0_OHM < 49.0


def test_stage_b_layout_is_self_consistent_and_openems_free():
    """``_stage_b_layout()`` must be callable (and its own assertions must
    pass) WITHOUT openEMS -- this is the function whose failure should map
    to exit 3 (config bug), distinct from a physics-gate failure (exit 1)."""
    module = _load_referee_module()
    layout = module._stage_b_layout()  # must not raise

    required = {
        "lx_mm", "ly_mm", "lz_mm", "cx_mm", "cy_mm",
        "z_port1_start_mm", "z_port1_stop_mm", "z_port2_start_mm", "z_port2_stop_mm",
        "z_feed_bot_mm", "z_feed_top_mm", "z_split_mm",
        "ref_plane_shift_port1_mm", "ref_plane_shift_port2_mm",
        "measplane_port1_mm", "feedshift_port1_mm",
        "measplane_port2_mm", "feedshift_port2_mm",
    }
    assert required <= set(layout.keys())

    # H2' fix: both ports' OUTER ends now sit exactly at the domain edges
    # (into the PML), not retracted from it.
    assert layout["z_port1_start_mm"] == 0.0
    assert layout["z_port2_stop_mm"] == layout["lz_mm"]

    # The TARGET reference planes (not the bare conductor extent, which
    # legitimately reaches the domain edges per H2') must stay safely
    # inside the clear, non-PML region.
    assert layout["z_feed_bot_mm"] > module.B_PML_DEPTH_MM
    assert layout["z_feed_top_mm"] < layout["lz_mm"] - module.B_PML_DEPTH_MM
    assert layout["z_port1_start_mm"] < layout["z_split_mm"] < layout["z_port2_stop_mm"]

    # FeedShift/MeasPlaneShift must land strictly inside each port's own
    # span and stay separated from each other.
    port1_span = layout["z_port1_stop_mm"] - layout["z_port1_start_mm"]
    port2_span = layout["z_port2_stop_mm"] - layout["z_port2_start_mm"]
    assert 0.0 < layout["measplane_port1_mm"] < port1_span
    assert 0.0 < layout["feedshift_port1_mm"] < port1_span
    assert 0.0 < layout["measplane_port2_mm"] < port2_span
    assert 0.0 < layout["feedshift_port2_mm"] < port2_span
    assert abs(layout["measplane_port1_mm"] - layout["feedshift_port1_mm"]) > module.B_DX_MM
    assert abs(layout["measplane_port2_mm"] - layout["feedshift_port2_mm"]) > module.B_DX_MM


def test_stage_b_port_directions_are_both_positive():
    """B4' fix (round-2 review, BLOCKING, hole-closer): both Stage B ports
    must be built stop>start (direction=+1), matching Coax.m's own
    same-direction layout -- CoaxialPort's current probe is direction-
    signed, so a mirror-symmetric (direction=-1) port 2 makes
    |uf_inc2/uf_inc1| read ~0 for a genuine forward wave instead of ~1
    (verified by the reviewer against openEMS's own CalcPort formulas on
    a synthetic forward TEM wave). All 13 round-1 tests stayed GREEN
    after the reviewer mutated port 2's orientation back to direction=-1
    -- this test closes that hole by pinning the sign directly from the
    SAME layout fields ``_build_stage_b_drive`` consumes to build the
    real ``CoaxialPort`` objects (single source of truth, not a
    duplicated/independent computation that could itself drift)."""
    module = _load_referee_module()
    layout = module._stage_b_layout()

    direction_port1 = 1.0 if (layout["z_port1_stop_mm"] - layout["z_port1_start_mm"]) > 0 else -1.0
    direction_port2 = 1.0 if (layout["z_port2_stop_mm"] - layout["z_port2_start_mm"]) > 0 else -1.0
    assert direction_port1 == 1.0, "port1 direction must be +1"
    assert direction_port2 == 1.0, "port2 direction must be +1 (B4' regression)"

    # Mutation check: flipping port2's start/stop (the EXACT round-1
    # regression) must be caught -- both by _stage_b_layout()'s own
    # assertion (AssertionError, exit 3) and by this test reading the
    # same fields a different way.
    mutated_direction = (
        1.0 if (layout["z_port2_start_mm"] - layout["z_port2_stop_mm"]) > 0 else -1.0
    )
    assert mutated_direction == -1.0, "mutation sanity check itself is broken"


def test_extract_two_port_s_synthetic_forward_and_reverse_drive():
    """Round-3 fix (BLOCKING): the round-2 fix flipped only the ``s_thru``
    numerator for the reverse drive and left ``s_self`` as ``uf_ref_self/
    uf_inc_self`` unconditionally -- WRONG, since ``uf_inc_self`` is not
    the reverse drive's own launched wave. The reviewer verified this with
    openEMS's own ``CalcPort`` formulas on synthetic wave amplitudes,
    entirely WITHOUT openEMS installed (plain complex arithmetic): the
    round-2 code reported S22 as 1/S22_true and S12 as S21_true/S22_true
    (both traced to the same wrong denominator). This test imitates that
    approach as a real, permanent unit test feeding fake port objects
    with known ``uf_inc``/``uf_ref`` values through ``_extract_two_port_s``
    for BOTH drives (item (c) of the round-3 review -- the previous
    ``reverse_drive``/``thru_uses_ref`` flag was unpinned by any test;
    mutating it stayed green).

    Wave-amplitude construction (matches ``_extract_two_port_s``'s own
    docstring derivation): for a KNOWN reciprocal 2x2 S-matrix (S12=S21),
    normalize each drive's own launched-wave amplitude to 1 --
      forward (a1=1, a2=0): b1=S11*a1, b2=S21*a1 (b2 is +z at port2, so
        IS uf_inc2 under the shared +z convention).
      reverse (a2'=1, a1'=0): b2'=S22*a2', b1'=S12*a2' (a2' is -z at
        port2, so IS uf_ref2; b1' is -z at port1, so IS uf_ref1).
    """
    module = _load_referee_module()

    class _FakePort:
        def __init__(self, uf_inc, uf_ref):
            self.uf_inc = np.asarray(uf_inc, dtype=np.complex128)
            self.uf_ref = np.asarray(uf_ref, dtype=np.complex128)

    s11_true = 0.15 + 0.05j
    s21_true = 0.85 - 0.30j   # |s21_true| ~= 0.90, mirrors the reviewer's "0.9"
    s12_true = s21_true       # reciprocal
    s22_true = 0.03 + 0.04j   # |s22_true| == 0.05, mirrors the reviewer's "0.051"

    # forward drive: a1=1 (port1's own launched wave, +z=uf_inc1), a2=0
    uf_inc1_fwd = np.array([1.0 + 0j])
    uf_ref1_fwd = s11_true * uf_inc1_fwd
    uf_inc2_fwd = s21_true * uf_inc1_fwd
    uf_ref2_fwd = np.array([0.0 + 0j])
    port1_fwd = _FakePort(uf_inc1_fwd, uf_ref1_fwd)
    port2_fwd = _FakePort(uf_inc2_fwd, uf_ref2_fwd)

    # reverse drive: a2'=1 (port2's own launched wave, -z=uf_ref2), a1'=0
    uf_ref2_rev = np.array([1.0 + 0j])
    uf_inc2_rev = s22_true * uf_ref2_rev
    uf_ref1_rev = s12_true * uf_ref2_rev
    uf_inc1_rev = np.array([0.0 + 0j])
    port2_rev = _FakePort(uf_inc2_rev, uf_ref2_rev)
    port1_rev = _FakePort(uf_inc1_rev, uf_ref1_rev)

    fwd = module._extract_two_port_s(port1_fwd, port2_fwd, reverse_drive=False)
    rev = module._extract_two_port_s(port2_rev, port1_rev, reverse_drive=True)

    tol = 1e-9
    assert abs(fwd["s_self"][0] - s11_true) < tol, f"S11: got {fwd['s_self'][0]}, want {s11_true}"
    assert abs(fwd["s_thru"][0] - s21_true) < tol, f"S21: got {fwd['s_thru'][0]}, want {s21_true}"
    assert abs(rev["s_self"][0] - s22_true) < tol, f"S22: got {rev['s_self'][0]}, want {s22_true}"
    assert abs(rev["s_thru"][0] - s12_true) < tol, f"S12: got {rev['s_thru'][0]}, want {s12_true}"

    # Discrimination check: the OLD (round-2) buggy formula -- unconditional
    # uf_ref_self/uf_inc_self and thru/uf_inc_self -- must NOT match the
    # truth for the reverse drive (proving this test actually catches the
    # regression, not just tautologically confirms the current code).
    old_buggy_s22 = port2_rev.uf_ref[0] / port2_rev.uf_inc[0]
    old_buggy_s12 = port1_rev.uf_ref[0] / port2_rev.uf_inc[0]
    assert abs(old_buggy_s22 - (1.0 / s22_true)) < tol, "old-bug reconstruction itself is wrong"
    assert abs(old_buggy_s12 - (s21_true / s22_true)) < tol, "old-bug reconstruction itself is wrong"
    assert abs(old_buggy_s22 - s22_true) > 1.0, (
        "old buggy S22 formula coincidentally matches truth -- test not discriminating"
    )
    assert abs(old_buggy_s12 - s12_true) > 1.0, (
        "old buggy S12 formula coincidentally matches truth -- test not discriminating"
    )

    # reverse_drive is recorded so a reviewer can tell which branch a
    # given result came from without re-deriving it.
    assert fwd["reverse_drive"] is False
    assert rev["reverse_drive"] is True


def test_stage_b_matched_through_band_and_group_delay():
    """B5' fix (round-2 review, BLOCKING): Stage B's own matched-through
    band must differ from Stage A's ideal-lossless band (rfx's raw |S21|
    profile is 0.960->0.737, a real lossy line), and the expected group
    delay (L12*sqrt(eps_r)/c) must land near the reviewer's own
    independently-stated ~282 ps for L12=58.4595mm, eps_r=2.1."""
    module = _load_referee_module()
    assert module.B_S21_THRU_BAND == (0.5, 1.1)
    assert module.B_S21_THRU_BAND != module.REPRODUCE_GATE_RECORD["gate"]["s21_thru_band"]

    expected_gd_s = module.B_L12_MM * 1e-3 * (module.B_PTFE_EPS_R ** 0.5) / module._C0
    assert abs(expected_gd_s * 1e12 - 282.0) < 5.0


def test_stage_a_wires_its_own_num_ts_and_end_criteria_into_openems(monkeypatch):
    """H1' fix (round-2 review, BLOCKING): A_NUM_TS/A_END_CRITERIA were
    DEFINED and pinned by test_stage_a_matches_coaxm_tutorial_constants
    (above) but never actually reached ``openEMS(...)`` -- Stage A silently
    ran whatever ``--nrts``/``--end-criteria`` Stage B's CLI flags carried
    (default 200000/1e-4) against Coax.m's MUR boundaries, where run
    length controls reflected-energy accumulation. That constants test
    passed before AND after the bug existed -- it only checked the
    constants' OWN values, never that anything downstream used them. This
    test asserts the WIRING: it captures what
    ``_build_stage_a_coax_tutorial`` is actually called with, via
    ``_run_stage_a_reproduce_gate`` (which, post-fix, no longer even
    accepts nrts/end_criteria as its own parameters -- there is no longer
    a code path for a caller-supplied CLI value to reach Stage A at all).
    """
    module = _load_referee_module()
    calls = []

    class _FakeFdtd:
        def Run(self, *args, **kwargs):
            pass  # no-op: _run_openems_capturing_stdout just needs this to not raise

    def fake_build(ContinuousStructure, openEMS, CoaxialPort, *, nrts, end_criteria, use_pml):
        calls.append({"nrts": nrts, "end_criteria": end_criteria})
        if len(calls) < 2:
            # let the smoke-run call through so the real (second) call --
            # the one this test actually cares about -- is reached too.
            return _FakeFdtd(), None, None
        raise RuntimeError("stub: stop here, only checking the wiring")

    monkeypatch.setattr(module, "_build_stage_a_coax_tutorial", fake_build)
    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))

    try:
        module._run_stage_a_reproduce_gate(sim_root="/tmp/_unused_h1prime", threads=1, use_pml=False)
    except RuntimeError:
        pass

    assert calls, "_build_stage_a_coax_tutorial was never called"
    assert any(c["end_criteria"] == module.A_END_CRITERIA for c in calls), (
        f"no call used A_END_CRITERIA={module.A_END_CRITERIA}; got {calls}"
    )
    assert any(c["nrts"] == module.A_NUM_TS for c in calls), (
        f"no call used A_NUM_TS={module.A_NUM_TS}; got {calls}"
    )
    # the smoke run must NOT silently use Stage B's 200000/1e-4 CLI
    # defaults either -- it is min(200, A_NUM_TS), always derived FROM
    # A_NUM_TS, never an independent value.
    assert all(c["nrts"] in (module.A_NUM_TS, min(200, module.A_NUM_TS)) for c in calls), (
        f"a call used an nrts unrelated to A_NUM_TS; got {calls}"
    )
    assert 200000 not in [c["nrts"] for c in calls], (
        "Stage A used Stage B's CLI --nrts default (200000) -- H1' regression"
    )
    assert 1e-4 not in [c["end_criteria"] for c in calls], (
        "Stage A used Stage B's CLI --end-criteria default (1e-4) -- H1' regression"
    )


def test_openems_unavailable_exits_2(monkeypatch):
    """main() must return exit code 2 (declared skip), not raise, when the
    openEMS Python bindings are absent -- lane convention (vessl_external_
    referee_lane.md): 0=self-checks passed, 1=self-check failed, 2=missing,
    3=internal config error."""
    module = _load_referee_module()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("CSXCAD.CSXCAD", "openEMS.openEMS", "openEMS.ports", "CSXCAD", "openEMS"):
            raise ImportError("simulated: openEMS not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert module.main([]) == 2


def test_config_error_exits_3_distinct_from_physics_failure(monkeypatch):
    """M2 review fix: an internal config/layout bug must NOT be reported
    with the same exit code as a physics/self-check gate failure. This
    simulates a layout bug (by monkeypatching ``_stage_b_layout`` to raise
    AssertionError, as it would if rfx's geometry drifted and the
    hard-coded constants above were not updated) and checks main() reports
    exit 3, not 1 or a raw traceback."""
    module = _load_referee_module()

    def _broken_layout():
        raise AssertionError("simulated: rfx geometry drifted")

    monkeypatch.setattr(module, "_stage_b_layout", _broken_layout)
    assert module.main([]) == 3


def test_freqs_overlap_the_rfx_band():
    """The referee's Stage B frequency grid must cover the rfx BAND points
    ([4,6,8,10,12] GHz, tests/test_coax_two_port_fdtd.py) it will be
    compared against by hand."""
    module = _load_referee_module()
    freqs_ghz = set(round(float(f), 6) for f in module.B_FREQS_GHZ)
    for f in (4.0, 6.0, 8.0, 10.0, 12.0):
        assert f in freqs_ghz, f"{f} GHz missing from referee frequency grid"
