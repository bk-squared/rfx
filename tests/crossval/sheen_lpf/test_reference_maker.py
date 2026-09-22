"""The reference maker stays runnable, and honest, on a machine with no openEMS.

``reference/make_openems_reference.py`` is the script that produces this case's
openEMS record: openEMS's own MSL notch filter tutorial as the reproduce gate,
then the Sheen low-pass filter on three meshes.  It is never run by CI -- the
solver runs by hand, on the cluster.  What CI can afford to check, and what
these tests check, is that the script still imports, still agrees with itself
about what it changes, still prints the plan it would build, and still carries
the Sheen script's own builder unedited -- all without openEMS.

The byte-identity pin on the thirteen sanity helpers is NOT repeated here.  It
lives once, in ``tests/crossval/msl_notch_filter/test_reference_maker.py``,
pointed at the shared module both cases load.

None of these steps an FDTD, reads a reference record, or asserts anything about
physics.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MAKER = HERE / "reference" / "make_openems_reference.py"
REPO_ROOT = HERE.parents[2]
SHARED_GATE = REPO_ROOT / "tests" / "crossval" / "_openems_tutorial_gate.py"
SHEEN_SCRIPT = REPO_ROOT / "validation" / "crossval" / "07_sheen_lpf.py"

# Owned by the MSL notch filter case's test, listed here only so this module can
# assert that this maker keeps no second copy of any of them.
SHARED_HELPERS = [
    "_ensure_openems_numpy_compat",
    "_import_openems",
    "_BAD_STDOUT_PATTERNS",
    "_TRUNCATION_STDOUT_PATTERNS",
    "_ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES",
    "_scan_stdout_for_bad_patterns",
    "_run_openems_capturing_stdout",
    "_END_CRITERIA_NOT_REACHED_TEXT",
    "_log_indicates_truncation",
    "_check_excitation_and_trace",
    "_non_physical_guard",
    "_passivity_witness",
    "_build_stage_a_notch_tutorial",
]


def _load_maker():
    spec = importlib.util.spec_from_file_location("_sheen_lpf_reference_maker", MAKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str, script: Path = MAKER) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=120,
    )


def test_self_check_passes_without_openems():
    result = _run("--self-check")
    assert result.returncode == 0, (
        f"--self-check exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "SELF-CHECK PASSED" in result.stdout


def test_dry_run_prints_the_plan_and_the_delta_list():
    result = _run("--dry-run", "--stage", "both")
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    out = result.stdout
    assert "MSL_NotchFilter.py" in out, "the dry run does not name the reproduce gate's tutorial"
    assert "Sheen" in out, "the dry run does not name the structure"
    for stage in ("stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine"):
        assert stage in out, f"the dry run does not plan {stage}"
    for entry in _load_maker().DELTA_LIST:
        assert entry in out, f"the dry run does not print the delta entry {entry[:60]!r}"


def test_delta_list_says_what_changes():
    m = _load_maker()
    delta = m.DELTA_LIST
    assert len(delta) == 14, f"expected fourteen delta entries, got {len(delta)}"

    # 1-5: the structure and the box.
    assert "eps_r 2.2" in delta[0] and "0.794 mm" in delta[0]
    assert "2.413 mm" in delta[1] and "20.320 x 2.540 mm" in delta[1]
    assert "no open stub" in delta[1], "delta 2 does not say the stub is gone"
    assert "9.8565" in delta[2] and "16.4635" in delta[2]
    assert "27.472 x 26.320 x 3.794 mm" in delta[3]
    assert "axis map" in delta[4]

    # 6-9: the ports and the run parameters, all the Sheen script's own.
    assert "FeedShift = PORT_MARGIN = 2.5 mm" in delta[5]
    assert "5.610 mm" in delta[5]
    assert "linspace(0.5e9, 20e9, 801)" in delta[6]
    assert "ref_impedance" in delta[7] and "DEPARTURE" in delta[7], (
        "delta 8 does not declare the two-pass CalcPort as a departure from the "
        "precedent's own tick"
    )
    assert "library defaults" in delta[8] and "NOT carried" in delta[8], (
        "delta 9 does not say the real pass runs at openEMS's own defaults"
    )
    assert "30000" in delta[8], "delta 9 does not name the script cap it refuses"
    # And the maker really does pass nothing, so the defaults really do apply.
    assert m.B_REAL_NRTS is None and m.B_REAL_END_CRITERIA is None
    assert m.SCRIPT_NRTS_CAP == 30000 and m.SCRIPT_END_CRITERIA_CAP == 1e-4

    # 10-11: the mesh rule and the rungs.
    assert "h_sub / 4" in delta[9] and "198.5" in delta[9]
    assert "1.0" in delta[10] and "0.70711" in delta[10] and "0.5" in delta[10]
    assert "z lines" in delta[10], "delta 11 does not say the substrate z lines scale"
    assert "4, 6 and 8 cells" in delta[10], (
        "delta 11 does not give the substrate cell count per rung"
    )

    # 4: the tutorial's own box, which is 100 x 30 x 3 mm (x -50..+50,
    # y -9..+21, z 0..3), not 100 x 21 x 3.
    assert "100 x 30 x 3 mm" in delta[3], (
        "delta 4 misstates the tutorial's box"
    )

    # 14: the mesh refinement around the metal, inherited from the script.
    assert delta[12].startswith("DELTA 14 (mesh refinement around the metal)")
    assert "resolution/4" in delta[12] and "smooths ONCE per axis" in delta[12], (
        "delta 14 does not contrast the tutorial's second smoothing with this "
        "builder's single one"
    )
    assert "PATCH_X0 and PATCH_X1" in delta[12], (
        "delta 14 does not name the wide section's two impedance steps"
    )
    assert "feed_edges_realized" in delta[12], (
        "delta 14 does not say where the realized feed widths are recorded"
    )
    assert "no DELTA 13" in delta[12], (
        "delta 14 does not account for the missing number"
    )

    # 15th slot is the closing entry: what does NOT change.
    assert "['PML_8','PML_8','MUR','MUR','PEC','MUR']" in delta[13]
    assert "see delta 14" in delta[13], (
        "the closing entry still claims the thirds-rule placement is unchanged"
    )


def test_the_shared_helpers_are_not_copied_into_this_maker():
    """One copy, in the shared module.

    The byte-identity pin lives in the MSL notch filter case's test and watches
    ``_openems_tutorial_gate.py``.  A helper re-defined here would be unwatched.
    """
    shared_names = {
        node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        else node.targets[0].id
        for node in ast.parse(SHARED_GATE.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name))
    }
    assert set(SHARED_HELPERS) <= shared_names, (
        "the shared module no longer carries every helper this case relies on"
    )
    offenders = [
        node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        else node.targets[0].id
        for node in ast.parse(MAKER.read_text()).body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in SHARED_HELPERS)
        or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in SHARED_HELPERS)
    ]
    assert not offenders, (
        f"{MAKER.name} carries its own copy of {offenders}; the pin watches "
        f"{SHARED_GATE.name}, so that copy would be unwatched."
    )


def test_stage_a_is_the_shared_tutorial_gate_not_a_local_one():
    m = _load_maker()
    assert m.F_NOTCH_AN_HZ is m._gate.F_NOTCH_AN_HZ
    assert m.STAGE_A_GATE is m._gate.STAGE_A_GATE
    assert m.REPRODUCE_GATE_RECORD is m._gate.REPRODUCE_GATE_RECORD
    # And the gate's analytic frequency is the one 07_sheen_lpf.py's own
    # reproduce-gate record says this case must be checked against.
    assert abs(m.F_NOTCH_AN_HZ - m.REPRODUCE_GATE_RECORD["analytic_f_notch_hz"]) < 1.0


def test_the_reported_features_carry_no_verdict():
    """The stopband minimum, the passband mean and the corner are reported.

    Nothing in what ``features`` returns says pass, fail or ok -- a gate that
    grew here would be a gate inside a case, which the crossval role redesign
    forbids.
    """
    import numpy as np

    m = _load_maker()
    sf = m._gate.load_spectral_features()
    f_ghz = np.linspace(0.5, 20.0, m.B_N_FREQS)
    mag = np.where(f_ghz <= 5.0, 0.95, 0.95 / (1.0 + ((f_ghz - 5.0) / 0.8) ** 2))
    mag = mag * (1.0 - 0.999 * np.exp(-((f_ghz - 7.6) / 0.06) ** 2))
    feats = m._stage_b_features(sf)(f_ghz, np.full_like(f_ghz, 0.2), mag)

    def keys(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield f"{prefix}{k}"
                yield from keys(v, f"{prefix}{k}.")

    verdicts = [k for k in keys(feats)
                if k.rsplit(".", 1)[-1] in ("passed", "ok", "gate", "tol", "verdict")]
    assert not verdicts, f"a verdict field appeared in the reported features: {verdicts}"
    assert feats["null"]["refined_f_ghz"] == pytest.approx(7.6, abs=0.05)
    assert feats["passband"]["mean_s21_linear"] == pytest.approx(0.95, abs=0.01)
    assert feats["cutoff_3db"]["f_ghz"] is not None


# ---------------------------------------------------------------------------
# The copy proof is only worth having if it reddens.  Each mutation below
# leaves every helper call in place and changes one thing the proof exists to
# catch: the builder drifting from the script it was copied from.
# ---------------------------------------------------------------------------
def _self_check_on_a_mutant(old: str, new: str) -> subprocess.CompletedProcess:
    """Run --self-check on a copy of the maker with ``old`` replaced by ``new``.

    The mutant is written BESIDE the maker, not into a tmpdir: the maker finds
    the shared module and the Sheen script by walking up from its own path, and
    a copy somewhere else would fail for that reason instead of for the reason
    under test.
    """
    src = MAKER.read_text()
    assert src.count(old) >= 1, f"the mutation target is not in the maker: {old!r}"
    mutant = MAKER.with_name(f"_mutant_{abs(hash((old, new))) % 10**8}.py")
    mutant.write_text(src.replace(old, new, 1))
    try:
        return _run("--self-check", script=mutant)
    finally:
        mutant.unlink(missing_ok=True)


def test_copy_proof_reddens_when_the_copied_builder_drifts():
    """Mutation (b): revive a drift the proof exists to catch.

    ``FeedShift=PORT_MARGIN`` is the script's; move the excitation plane and the
    builder is no longer the script's builder, while every helper call, every
    constant and the whole plan still work.
    """
    result = _self_check_on_a_mutant(
        "        FeedShift=PORT_MARGIN, MeasPlaneShift=0.45 * PATCH_X0, priority=10)",
        "        FeedShift=2.0 * PORT_MARGIN, MeasPlaneShift=0.45 * PATCH_X0, priority=10)")
    assert result.returncode != 0, (
        "the self-check passed on a builder whose FeedShift no longer matches "
        "07_sheen_lpf.py's:\n" + result.stdout
    )
    assert "IS the script's build block" in result.stdout


def test_copy_proof_reddens_when_the_rung_builder_drifts():
    """Mutation (b): the rung builder stops scaling the substrate's z lines.

    Freezing the substrate at 4 cells is exactly the defect DELTA 11 exists to
    prevent, and it leaves the resolution factor, the plan and every gate call
    untouched -- so only the character-for-character derivation can see it.
    """
    result = _self_check_on_a_mutant(
        "    mesh.AddLine('z', np.linspace(0, H_SUB, substrate_z_cells(resolution_factor) + 1))",
        "    mesh.AddLine('z', np.linspace(0, H_SUB, 5))")
    assert result.returncode != 0, (
        "the self-check passed on a rung builder that no longer refines the "
        "substrate:\n" + result.stdout
    )
    assert "two declared rung substitutions" in result.stdout


def test_the_excitation_is_the_scripts_own():
    """The launched spectrum is derived, not written by hand.

    ``SetGaussExcite`` sits outside the build block -- the script sets it in
    ``_openems_common_setup``, which the block excludes on purpose -- so it gets
    its own derivation from the same renames.
    """
    m = _load_maker()
    proof = m._copy_proof()
    assert proof["excitation_matches"] and proof["excitation_rung_matches"]
    assert proof["excitation_script"].strip() == "fdtd.SetGaussExcite(F_MAX / 2, F_MAX / 2)"
    # And the record carries the source line, not a number formatted from a
    # constant that nothing compares against.
    art = m._build_artifact({}, {}, {}, ["stage_a"])
    assert "SetGaussExcite(F_MAX / 2, F_MAX / 2)" in art["meta"]["excitation"]


def test_excitation_proof_reddens_when_the_launched_band_moves():
    """Mutation (b): halve the excitation centre and corner.

    Every helper call, every constant and the whole plan still work; only the
    frequencies openEMS launches change. Before the derivation existed this
    mutation left the self-check green.
    """
    result = _self_check_on_a_mutant(
        "    fdtd.SetGaussExcite(F_MAX / 2, F_MAX / 2)",
        "    fdtd.SetGaussExcite(F_MAX / 4, F_MAX / 4)")
    assert result.returncode != 0, (
        "the self-check passed on a builder that launches a different band:\n"
        + result.stdout
    )
    assert "SetGaussExcite statement IS the script's" in result.stdout


def test_the_check_itself_can_fail():
    """Mutation (a): disable the comparison and the self-check must go red.

    (b) above is the one that matters -- a tautological check survives it -- but
    a check that cannot fail at all is worth ruling out too.
    """
    result = _self_check_on_a_mutant(
        '        check(proof["copy_matches"],',
        '        check(not proof["copy_matches"],')
    assert result.returncode != 0
    assert "SELF-CHECK FAILED" in result.stdout


def test_the_sheen_script_is_not_edited_by_this_case():
    """The builder is copied from ``07_sheen_lpf.py``; that file is not ours.

    The copy proof reads it at run time, so a session that "fixed" the script to
    make the proof pass would be editing another case's file.  This test states
    the direction of the dependency; git states whether the file moved.
    """
    assert SHEEN_SCRIPT.is_file()
    m = _load_maker()
    assert m.SCRIPT_REL_PATH == "validation/crossval/07_sheen_lpf.py"
    assert m.SCRIPT_FUNCTION == "run_openems"
    # The proof runs against the file on disk, not a pasted copy: the slice it
    # compares against is a substring of that file, with the script's own names.
    slice_ = m._script_builder_slice()
    assert slice_ in SHEEN_SCRIPT.read_text()
    assert "FDTD.AddMSLPort" in slice_ and "fdtd." not in slice_
    # Delta 9 declines the script's stop criteria, so the call that sets them is
    # outside the compared block -- otherwise the proof would be asserting that
    # this maker adopts the cap.
    assert m.SCRIPT_SETUP_FUNCTION not in slice_
    setup = m._script_function_source(m.SCRIPT_SETUP_FUNCTION)
    assert "NrTS=30000" in setup and "EndCriteria=1e-4" in setup, (
        "the cap delta 9 declines is no longer where the maker reads it"
    )


# ---------------------------------------------------------------------------
# One rung per cluster job, and putting the parts back together
# ---------------------------------------------------------------------------
def test_rung_selector_parses_and_refuses():
    m = _load_maker()
    assert m.parse_rungs(m.DEFAULT_RUNGS) == list(m.RUNG_ORDER_STAGES)
    assert m.parse_rungs("fine,coarse") == ["stage_b_coarse", "stage_b_fine"], (
        "a subset must come back in rung order, not in the order it was typed"
    )
    for spec in ("", "   ", "middle", "coarse,middle", ",,"):
        with pytest.raises(ValueError):
            m.parse_rungs(spec)


def test_stage_a_runs_in_every_rung_job():
    """Splitting the rungs across jobs does not make the gate optional."""
    m = _load_maker()
    for spec in ("coarse", "mid", "fine", "coarse,fine"):
        stages = m._stages_for("both", m.parse_rungs(spec))
        assert stages[0] == "stage_a", f"--rungs {spec} dropped the reproduce gate"
    # --stage B still skips it, deliberately, and warns at run time.
    assert m._stages_for("B", m.parse_rungs("mid")) == ["stage_b_mid"]


def test_dry_run_honours_the_rung_selection():
    result = _run("--dry-run", "--stage", "both", "--rungs", "mid")
    assert result.returncode == 0, result.stderr
    assert "rungs requested mid" in result.stdout
    # The delta list names all three rungs wherever it is printed, so scope the
    # check to the stage plan -- which is the part that says what gets solved.
    plan = result.stdout.split("STAGE PLAN", 1)[1]
    assert "stage_a" in plan, "the gate is still planned"
    assert "stage_b_mid" in plan
    assert "stage_b_coarse" not in plan and "stage_b_fine" not in plan, (
        "the stage plan planned rungs this job will not solve"
    )


def test_the_stop_criteria_note_says_what_ran_and_why():
    m = _load_maker()
    default = m.stop_criteria_note(None, None)
    assert "No override was given" in default
    assert "library defaults" in default

    over = m.stop_criteria_note(1e-4, None)
    assert over.startswith("made with --real-end-criteria ")
    # The measurement that motivated the override travels with the record, so a
    # reader of a looser run does not have to go and find it.
    assert "107 minutes on the coarsest rung on 8 threads" in over
    assert "369367263243" in over and "369367263269" in over
    assert "-40 dB of the post-source peak" in over

    both = m.stop_criteria_note(1e-4, 60000)
    assert "--real-end-criteria" in both and "--real-nrts 60000" in both


def test_the_declared_design_is_still_the_library_defaults():
    """An override is a CLI choice; it does not move what the script declares."""
    m = _load_maker()
    assert m.B_REAL_NRTS is None and m.B_REAL_END_CRITERIA is None
    assert "library defaults" in m.DELTA_LIST[8]
    assert "--real-end-criteria" in m.DELTA_LIST[8], (
        "delta 9 does not mention that the override exists"
    )


def _part(tmp_path, name, rung, *, run_id, s21_bin17=0.5, build="bld-1",
          commit="cafe1234", note="made with --real-end-criteria 0.0001: ...",
          record_length_s=7.4e-8, witness=None):
    """A minimal record shaped like one rung job's output."""
    stage_a = {
        "freqs_ghz": [1.0, 2.0, 3.0],
        "s11_mag": [0.1, 0.2, 0.3],
        "s11_deg": [0.0, 1.0, 2.0],
        "s21_mag": [0.9, s21_bin17, 0.8],
        "s21_deg": [0.0, 1.0, 2.0],
        "energy_sum": [0.82, 0.29, 0.73],
        "notch": {"refined_f_ghz": 3.6711, "depth_db": -53.16},
    }
    rec = {
        "meta": {
            "tutorial_source": "thliebig/openEMS python/Tutorials/MSL_NotchFilter.py",
            "rfx_openems_image": "ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8",
            "rfx_openems_commit": build,
            "rfx_commit": commit,
            "stop_criteria_note": note,
            "rungs_in_this_record": [rung],
            "stages": {"stage_a": {"stage": "stage_a"},
                       m_rung(rung): {"stage": m_rung(rung),
                                      "record_length_s": record_length_s}},
        },
        "stage_a": stage_a,
        m_rung(rung): dict({"null": {"refined_f_ghz": 7.9}, "rung": rung},
                           **({"witness_2n": witness} if witness else {})),
        "run_id": run_id,
        "run_id_note": "filled by the submitter",
    }
    p = tmp_path / name
    p.write_text(json.dumps(rec, indent=1))
    return p


def m_rung(short):
    return {"coarse": "stage_b_coarse", "mid": "stage_b_mid",
            "fine": "stage_b_fine"}[short]


def test_merge_combines_three_rung_parts(tmp_path):
    parts = [_part(tmp_path, f"{r}.json", r, run_id=f"3693672633{i:02d}")
             for i, r in enumerate(("coarse", "mid", "fine"))]
    out = tmp_path / "full.json"
    result = _run("--merge", *[str(p) for p in parts], "--output", str(out))
    assert result.returncode == 0, result.stdout + result.stderr

    merged = json.loads(out.read_text())
    for stage in ("stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine"):
        assert isinstance(merged[stage], dict), f"{stage} did not survive the merge"
    assert merged["meta"]["rungs_in_this_record"] == ["coarse", "mid", "fine"]
    assert merged["run_id"] is None
    assert "merged_from" in merged["run_id_note"]
    entries = merged["meta"]["merged_from"]
    assert len(entries) == 3
    assert [e["rungs"] for e in entries] == [["coarse"], ["mid"], ["fine"]]
    assert [e["run_id"] for e in entries] == ["369367263300", "369367263301",
                                              "369367263302"]
    for e in entries:
        assert e["maker_commit"] == "cafe1234"
        assert e["stop_criteria_note"].startswith("made with --real-end-criteria")
        assert Path(e["path"]).name in {"coarse.json", "mid.json", "fine.json"}
    assert merged["meta"]["maker_commits_agree"] is True
    # Each rung block came from its own part, not from the first one.
    assert [merged[m_rung(r)]["rung"] for r in ("coarse", "mid", "fine")] == \
        ["coarse", "mid", "fine"]


@pytest.mark.parametrize("what,build,bin17,drop,dup", [
    ("a rung missing", "bld-1", 0.5, True, False),
    ("a rung twice", "bld-1", 0.5, False, True),
    ("stage_a differs by one bin", "bld-1", 0.4242, False, False),
    ("a different openEMS build", "bld-2", 0.5, False, False),
])
def test_merge_refuses_what_it_cannot_reconcile(tmp_path, what, build, bin17,
                                                drop, dup):
    """A merge does not average, choose or reconcile; it refuses and says why.

    Stage A is the check that costs nothing: every rung job runs the same
    tutorial on the same mesh and grid, so two parts that disagree on it bin for
    bin did not come from the same solver.
    """
    parts = [_part(tmp_path, "coarse.json", "coarse", run_id="a"),
             _part(tmp_path, "mid.json", "mid", run_id="b", s21_bin17=bin17,
                   build=build)]
    if not drop:
        parts.append(_part(tmp_path, "fine.json", "fine", run_id="c"))
    if dup:
        parts.append(_part(tmp_path, "coarse2.json", "coarse", run_id="d"))
    out = tmp_path / "full.json"
    result = _run("--merge", *[str(p) for p in parts], "--output", str(out))
    assert result.returncode == 3, (
        f"{what}: expected a refusal, got rc={result.returncode}\n{result.stdout}"
    )
    assert "MERGE REFUSED" in result.stderr
    assert not out.exists(), f"{what}: a refused merge still wrote a record"


# ---------------------------------------------------------------------------
# A run that is stopped by its step cap
# ---------------------------------------------------------------------------
def test_accept_truncation_is_off_by_default():
    """A truncated spectrum is not a reference unless someone asks for one."""
    m = _load_maker()
    import inspect

    assert m.ACCEPT_TRUNCATION_DEFAULT is False
    assert inspect.signature(m._run_stage_b).parameters["accept_truncation"].default \
        is False
    # And the flag cannot reach Stage A even by mistake: its runner has no such
    # parameter, so the reproduce gate's end-criteria check is not overridable.
    assert "accept_truncation" not in inspect.signature(
        m._gate.run_stage_a).parameters
    # And a run that was not given the flag does not mention it.
    result = _run("--dry-run", "--stage", "both")
    assert result.returncode == 0
    assert "--accept-truncation was given" not in result.stdout


def test_the_stop_criteria_note_names_the_flag_when_it_is_used():
    m = _load_maker()
    off = m.stop_criteria_note(1e-4, 60000)
    on = m.stop_criteria_note(1e-4, 60000, True)
    assert "--accept-truncation" not in off
    assert "--accept-truncation was given" in on
    assert "truncated: true" in on
    assert "not decided here" in on, (
        "the note draws a conclusion about what a truncated record is worth"
    )


def test_openems_progress_lines_are_parsed(tmp_path):
    """openEMS's own progress print is where the ring-down depth comes from.

    The pattern is written against openEMS's print statement, not against a
    capture -- no real log was available -- so the shapes it must tolerate are
    pinned here: the space ``setw`` can leave after the minus sign, a space
    before ``dB``, and ``inf`` while the peak is still zero.
    """
    gate = _load_maker()._gate
    log = "\n".join([
        "openEMS v0.0.36",
        "[@   0s] Timestep:        0 || Speed: 0.0 MC/s || Energy: ~0.0e+00 (-infdB)",
        "[@   4s] Timestep:     4000 || Speed: 25.5 MC/s (1e-4 s/TS) || Energy: ~1.1e-12 (-12.40dB)",
        "[@1m40s] Timestep:    50000 || Speed: 25.1 MC/s (1e-4 s/TS) || Energy: ~9.9e-16 (-36.80 dB)",
        "[@2m00s] Timestep:    60000 || Speed: 25.0 MC/s (1e-4 s/TS) || Energy: ~6.1e-16 (- 8.92dB)",
        "RunFDTD: Warning: Max. number of timesteps was reached before the end-criteria of 0.0001 was reached",
        "Timestep: 200",
    ])
    p = gate._energy_progress(log)
    assert p["final_timestep"] == 60000
    assert p["final_energy_db"] == pytest.approx(-8.92)
    assert p["energy_db_trace"] == [[4000, -12.40], [50000, -36.80], [60000, -8.92]]
    assert p["energy_db_trace_points"] == 3
    assert "Timestep" in p["energy_db_source"]
    # A log with no progress lines yields None, not a crash and not a number.
    empty = gate._energy_progress("openEMS v0.0.36\nnothing to see here")
    assert empty["final_timestep"] is None and empty["final_energy_db"] is None
    assert empty["energy_db_trace"] == []


def test_a_truncated_pass_still_leaves_its_arrays(tmp_path):
    """The evidence file from a capped run carries what the solver produced.

    Measured 2026-09-22 (VESSL 369367263382): the gate fired as designed but the
    stage block in the evidence file was null, because the raise came before the
    arrays were computed. It now comes after. The gate itself is unchanged --
    it still raises and the record is still refused.
    """
    import numpy as np

    m = _load_maker()
    gate = m._gate
    capped = "\n".join([
        "openEMS v0.0.36",
        "FDTD timestep is: 0.00 s; Nyquist rate: 149 timesteps @20006339607.00 Hz",
        "Max. number of timesteps: 60000 ( --> 35.1 * Excitation signal length)",
        "[@4s] Timestep: 4000 || Energy: ~1.1e-12 (-12.40dB)",
        # The last progress line is NOT the cap: openEMS prints every few seconds.
        "[@2m] Timestep: 58120 || Energy: ~6.1e-16 (-38.92dB)",
        "RunFDTD: Warning: Max. number of timesteps was reached before the end-criteria of 0.0001 was reached",
    ])
    dt = 1.0 / (2.0 * 20006339607.0 * 149)

    class _Port:
        def __init__(s, n, nf):
            s.n, s.nf = n, nf
            s.U_filenames = []
            s.feed_shift = 0.0025
            s.measplane_shift = 0.0056
            s.Z_ref = np.full(nf, 50.5 + 0.1j)

        def CalcPort(s, path, f, ref_impedance=None):
            s.uf_inc = np.ones(s.nf, dtype=complex)
            g = f / 1e9
            mag = (np.where(g <= 5, 0.95, 0.95 / (1 + ((g - 5) / 0.8) ** 2))
                   * (1 - 0.999 * np.exp(-((g - 7.9) / 0.06) ** 2))) \
                if s.n == 1 else np.full(s.nf, 0.2)
            s.uf_ref = mag * np.exp(-1j * np.linspace(0, 40, s.nf))

    nf = m.B_N_FREQS
    lines = {"x": np.linspace(0.0, m.LX, 140), "y": np.linspace(0.0, m.LY, 140),
             "z": np.concatenate([np.linspace(0.0, m.H_SUB, 5),
                                  np.linspace(m.H_SUB, m.LZ, 17)[1:]])}
    ports = [_Port(0, nf), _Port(1, nf)]

    class _Grid:
        def __init__(s, l): s.l = l
        def GetLines(s, a): return s.l[a]

    class _CSX:
        def __init__(s, l): s._g = _Grid(l)
        def GetGrid(s): return s._g

    class _FDTD:
        def __init__(s, l): s._c = _CSX(l)
        def GetCSX(s): return s._c

    gate._import_openems = lambda: (object, object, object)
    gate._run_openems_capturing_stdout = lambda fd, p, threads=0: capped
    gate._openems_version = lambda log: {"version": "0.0.36", "source": "banner"}
    gate._check_excitation_and_trace = lambda *a, **k: (1.2e-13, 4096)
    m._build_sheen_board_at_rung = lambda *a, **k: (_FDTD(lines), ports[0], ports[1])

    def solve(accept):
        return m._run_stage_b(
            label="stage_b_coarse", sim_root=str(tmp_path), threads=1,
            resolution_factor=1.0, sf=gate.load_spectral_features(),
            real_nrts=60000, real_end_criteria=1e-4, accept_truncation=accept)

    # Gate ON (the default): it still raises, and the partial record is no
    # longer empty.
    with pytest.raises(gate.StageFailure) as excinfo:
        solve(False)
    partial = excinfo.value.partial
    for key in ("freqs_ghz", "s11_mag", "s11_deg", "s21_mag", "s21_deg",
                "energy_sum"):
        assert key in partial, f"the evidence file still lost {key}"
        assert len(partial[key]) == nf
    assert partial["truncated"] is True
    assert partial["final_energy_db"] == pytest.approx(-38.92)
    assert partial["final_timestep"] == 58120
    assert "null" in partial and "passband" in partial and "cutoff_3db" in partial
    assert "max_energy_sum_band" in partial
    assert "end-criteria" in str(excinfo.value)

    # Gate ACCEPTED: no raise, and the record says what it is.
    rec, meta = solve(True)
    assert rec["truncated"] is True
    assert rec["final_energy_db"] == pytest.approx(-38.92)
    assert rec["truncation_note"] == (
        "record length declared by --real-nrts 60000; the box energy had decayed "
        "to -38.92 dB at the cap; see stop_criteria_note")
    assert meta["end_criteria_reached"] is False
    assert meta["truncation_accepted"] is True
    assert meta["final_timestep"] == 58120
    assert meta["energy_db_trace"] == [[4000, -12.40], [58120, -38.92]]
    assert "--accept-truncation was given" in meta["stop_criteria_note"]
    # The record is as long as the CAP, not as long as the last progress line.
    assert meta["timesteps_executed"] == 58120
    assert meta["max_timesteps_declared"] == 60000
    assert meta["record_length_steps"] == 60000, (
        "a truncated run's record length came from the last progress line"
    )
    assert meta["record_length_s"] == pytest.approx(60000 * dt)
    assert "the CAP" in meta["record_length_steps_basis"]
    assert "58120" in meta["record_length_steps_basis"], (
        "the basis does not say what the last progress line read"
    )
    assert meta["dt_s"] == pytest.approx(dt)
    assert meta["dt_s_uncertainty_rel"] == pytest.approx(1.0 / 149)


def test_a_clean_run_is_untouched_by_all_of_this(tmp_path):
    """No truncation warning: no truncated key, and end_criteria_reached True."""
    import numpy as np

    m = _load_maker()
    gate = m._gate
    clean = "\n".join([
        "openEMS v0.0.36",
        "FDTD timestep is: 0.00 s; Nyquist rate: 149 timesteps @20006339607.00 Hz",
        "Max. number of timesteps: 60000 ( --> 35.1 * Excitation signal length)",
        "[@4s] Timestep: 4000 || Energy: ~1.1e-12 (-12.40dB)",
        "[@1m28s] Timestep: 44120 || Energy: ~1.0e-16 (-50.02dB)",
    ])

    class _Port:
        def __init__(s, n, nf):
            s.n, s.nf = n, nf
            s.U_filenames = []
            s.feed_shift = 0.0025
            s.measplane_shift = 0.0056
            s.Z_ref = np.full(nf, 50.5 + 0.1j)

        def CalcPort(s, path, f, ref_impedance=None):
            s.uf_inc = np.ones(s.nf, dtype=complex)
            s.uf_ref = np.full(s.nf, 0.9 if s.n == 1 else 0.2, dtype=complex)

    class _Grid:
        def __init__(s, l): s.l = l
        def GetLines(s, a): return s.l[a]

    class _CSX:
        def __init__(s, l): s._g = _Grid(l)
        def GetGrid(s): return s._g

    class _FDTD:
        def __init__(s, l): s._c = _CSX(l)
        def GetCSX(s): return s._c

    nf = m.B_N_FREQS
    lines = {"x": np.linspace(0.0, m.LX, 140), "y": np.linspace(0.0, m.LY, 140),
             "z": np.concatenate([np.linspace(0.0, m.H_SUB, 5),
                                  np.linspace(m.H_SUB, m.LZ, 17)[1:]])}
    ports = [_Port(0, nf), _Port(1, nf)]
    gate._import_openems = lambda: (object, object, object)
    gate._run_openems_capturing_stdout = lambda fd, p, threads=0: clean
    gate._openems_version = lambda log: {"version": "0.0.36", "source": "banner"}
    gate._check_excitation_and_trace = lambda *a, **k: (1.2e-13, 4096)
    m._build_sheen_board_at_rung = lambda *a, **k: (_FDTD(lines), ports[0], ports[1])

    rec, meta = m._run_stage_b(
        label="stage_b_coarse", sim_root=str(tmp_path), threads=1,
        resolution_factor=1.0, sf=gate.load_spectral_features())
    assert "truncated" not in rec
    assert "truncation_note" not in rec
    assert meta["end_criteria_reached"] is True
    assert "truncation_accepted" not in meta
    assert meta["final_energy_db"] == pytest.approx(-50.02)
    # A run that reached its own end criterion stopped where the last progress
    # line is, so THAT is the record length -- not the cap it never reached.
    assert meta["record_length_steps"] == 44120
    assert meta["max_timesteps_declared"] == 60000
    assert "reached its end criterion" in meta["record_length_steps_basis"]
    assert meta["record_length_s"] == pytest.approx(
        44120 / (2.0 * 20006339607.0 * 149))


# ---------------------------------------------------------------------------
# A declared record length, and the witness that measures what it costs
# ---------------------------------------------------------------------------
def test_record_length_witness_needs_a_declared_length():
    """Two lengths to compare, and a run at the shorter one that is recordable."""
    m = _load_maker()
    assert m.RECORD_LENGTH_WITNESS_DEFAULT is False
    for args in (("--record-length-witness",),
                 ("--record-length-witness", "--real-nrts", "60000"),
                 ("--record-length-witness", "--accept-truncation")):
        result = _run("--dry-run", *args)
        assert result.returncode == 3, f"{args} was accepted"
        assert "--record-length-witness needs" in result.stderr
    ok = _run("--dry-run", "--stage", "both", "--rungs", "coarse",
              "--real-nrts", "60000", "--accept-truncation",
              "--record-length-witness")
    assert ok.returncode == 0, ok.stderr


def test_dry_run_says_a_witnessed_rung_costs_two_solves():
    result = _run("--dry-run", "--stage", "both", "--rungs", "coarse,mid",
                  "--real-nrts", "60000", "--accept-truncation",
                  "--record-length-witness")
    assert result.returncode == 0, result.stderr
    assert "solved TWICE, at 60000 and 120000 timesteps" in result.stdout
    assert "costs 4 Stage B solves for 2 rung(s)" in result.stdout
    off = _run("--dry-run", "--stage", "both", "--rungs", "coarse,mid")
    assert "record-length witness   off: 2 Stage B solve(s)" in off.stdout


def test_the_witness_measures_only_where_both_curves_are_above_the_floor():
    """A deep null moves for reasons that are not record length.

    The floor is what keeps the witness a measure of truncation rather than of
    the null, so the arithmetic is planted here rather than assumed.
    """
    import numpy as np

    m = _load_maker()
    f = np.linspace(0.5, 20.0, m.B_N_FREQS)
    base = np.full_like(f, 0.5)                       # -6.02 dB, well above the floor
    other = base.copy()
    # One in-band bin moved by a known amount, above the floor.
    i_in = int(np.argmin(np.abs(f - 6.0)))
    other[i_in] = base[i_in] * 10 ** (0.25 / 20.0)    # +0.25 dB
    # One in-band bin far below the floor, moved by a lot. It must be ignored.
    i_null = int(np.argmin(np.abs(f - 8.0)))
    base[i_null] = 10 ** (-60.0 / 20.0)
    other[i_null] = 10 ** (-30.0 / 20.0)              # 30 dB apart, below the floor
    # One bin outside the witness band, moved by a lot. Also ignored.
    i_out = int(np.argmin(np.abs(f - 18.0)))
    other[i_out] = base[i_out] * 10 ** (9.0 / 20.0)

    rec_n = {"freqs_ghz": f.tolist(), "s21_mag": base.tolist(),
             "s11_mag": np.full_like(f, 0.3).tolist()}
    rec_2n = {"freqs_ghz": f.tolist(), "s21_mag": other.tolist(),
              "s11_mag": np.full_like(f, 0.3).tolist()}
    w = m.record_length_witness(rec_n, rec_2n, n_steps=60000, n2_steps=120000)

    assert w["n_steps"] == 60000 and w["n2_steps"] == 120000
    assert w["floor_db"] == -20.0
    assert w["band_ghz"] == [2.0, 12.0]
    assert w["max_abs_delta_s21_db"] == pytest.approx(0.25, abs=1e-6), (
        "the witness picked up the null or the out-of-band bin"
    )
    assert w["f_ghz_at_max_abs_delta_s21"] == pytest.approx(6.0, abs=0.05)
    assert w["max_abs_delta_s11_db"] == pytest.approx(0.0, abs=1e-9)
    assert w["s21_bins_compared"] < f.size, "the floor and the band excluded nothing"
    # No verdict anywhere in what it returns.
    assert not [k for k in w if k in ("passed", "ok", "gate", "verdict")]


def test_the_witness_says_so_when_the_grids_do_not_match():
    m = _load_maker()
    w = m.record_length_witness({"freqs_ghz": [1.0, 2.0]},
                                {"freqs_ghz": [1.0, 2.0, 3.0]},
                                n_steps=1, n2_steps=2)
    assert "error" in w and "frequency grid" in w["error"]
    assert "max_abs_delta_s21_db" not in w


# The lines the container actually prints, copied from the persisted log of run
# 369367263401 (stage_b_coarse_real_openems_stdout.log, lines 32-33, 61-63,
# 70-71) plus openEMS's own truncation warning. Every parser below is pinned
# against THIS text, not against openEMS's source.
REAL_LOG = "\n".join([
    "Timestep (s)\t\t: 0.00",
    "Timestep method name\t: Rennings_2",
    "FDTD timestep is: 0.00 s; Nyquist rate: 149 timesteps @20006339607.00 Hz",
    "Excitation signal length is: 1708 timesteps (0.00s)",
    "Max. number of timesteps: 3000 ( --> 1.76 * Excitation signal length)",
    "[@        4s] Timestep:         1110 || Speed:  122.2 MC/s "
    "(3.632e-03 s/TS) || Energy: ~1.21e-14 (- 0.00dB)",
    "[@        8s] Timestep:         2146 || Speed:  112.4 MC/s "
    "(3.948e-03 s/TS) || Energy: ~3.11e-15 (- 5.91dB)",
    "RunFDTD: Warning: Max. number of timesteps was reached before the "
    "end-criteria of 0.0001 was reached",
])


def test_the_solver_timestep_is_read_from_its_own_banner():
    gate = _load_maker()._gate
    # A build that prints enough digits is used as printed.
    assert gate._timestep_seconds("FDTD timestep is: 1.2345e-12 s; Nyquist rate: 3e-11 s") \
        == pytest.approx(1.2345e-12)
    assert gate._timestep_seconds("Used timestep: 4.567e-13 s") == pytest.approx(4.567e-13)
    # A progress line is NOT a timestep declaration: no seconds after the count.
    assert gate._timestep_seconds(
        "[@ 4s] Timestep: 4000 || Energy: ~1e-12 (-12.4dB)") is None
    assert gate._timestep_seconds("openEMS v0.0.36\nnothing here") is None
    # And THIS container prints two decimals, so its own line is unusable.
    assert gate._timestep_seconds(REAL_LOG) is None


def test_dt_is_derived_from_the_nyquist_line_when_the_print_is_too_coarse():
    """openEMS prints "0.00 s" for a 1.7e-13 s timestep.

    It also prints the Nyquist rate as an integer count of timesteps per half
    period at f_max, N = floor(1 / (2 f dt)), which inverts to dt = 1 / (2 f N)
    and is good to one step in N.
    """
    gate = _load_maker()._gate
    s = gate._solver_setup(REAL_LOG)
    assert s["nyquist_steps"] == 149
    assert s["nyquist_f_hz"] == pytest.approx(20006339607.0)
    assert s["dt_s"] == pytest.approx(1.0 / (2.0 * 20006339607.0 * 149))
    assert s["dt_s"] == pytest.approx(1.677e-13, rel=1e-3)
    assert s["dt_s_uncertainty_rel"] == pytest.approx(1.0 / 149)
    assert "DERIVED" in s["dt_s_source"] and "Nyquist" in s["dt_s_source"]
    assert s["excitation_length_steps"] == 1708
    assert s["max_timesteps_declared"] == 3000
    # The floor's bound is well inside what a merge allows between rungs.
    assert s["dt_s_uncertainty_rel"] < _load_maker().MERGE_RECORD_LENGTH_TOL

    # An exact print wins over the derivation, and says so.
    exact = gate._solver_setup(
        "FDTD timestep is: 1.6773e-13 s; Nyquist rate: 149 timesteps @2.0e10 Hz")
    assert exact["dt_s"] == pytest.approx(1.6773e-13)
    assert exact["dt_s_uncertainty_rel"] == 0.0
    assert "read directly" in exact["dt_s_source"]
    assert exact["nyquist_steps"] == 149, "the Nyquist numbers are recorded either way"

    # Neither line present: None, and the source says what was tried.
    none = gate._solver_setup("openEMS v0.0.36\nnothing useful")
    assert none["dt_s"] is None and "UNAVAILABLE" in none["dt_s_source"]


def test_the_truncation_warning_is_not_read_as_a_cap_declaration():
    """"Max. number of timesteps: 3000" declares; "... was reached" warns."""
    gate = _load_maker()._gate
    warn = ("RunFDTD: Warning: Max. number of timesteps was reached before the "
            "end-criteria of 0.0001 was reached")
    assert gate._first_int(gate._MAX_TIMESTEPS_RE, warn) is None
    assert gate._first_int(gate._MAX_TIMESTEPS_RE,
                           "Max. number of timesteps: 3000 ( --> 1.76 * ...)") == 3000


def test_the_progress_lines_parse_in_the_containers_own_form():
    """"(- 5.91dB)": a space after the minus, none before dB."""
    gate = _load_maker()._gate
    p = gate._energy_progress(REAL_LOG)
    assert p["final_timestep"] == 2146
    assert p["final_energy_db"] == pytest.approx(-5.91)
    assert p["energy_db_trace"] == [[1110, -0.0], [2146, -5.91]]
    assert p["energy_db_trace_points"] == 2


def test_a_truncated_runs_record_length_is_the_cap_not_the_last_print():
    """openEMS prints a progress line every few seconds, not at the cap.

    On the real log the last one says 2146 while the run was capped at 3000, so
    a truncated pass's record is 3000 steps long and the field says which number
    it used.
    """
    gate = _load_maker()._gate
    assert gate._timesteps_executed(REAL_LOG) == 2146, (
        "timesteps_executed is the largest PROGRESS count"
    )
    assert gate._solver_setup(REAL_LOG)["max_timesteps_declared"] == 3000
    assert gate._log_indicates_truncation(REAL_LOG) is True


def test_merge_refuses_rungs_of_different_record_lengths(tmp_path):
    """A step cap is not a record length: dt shrinks with the cell."""
    parts = [
        _part(tmp_path, "coarse.json", "coarse", run_id="a", record_length_s=7.4e-8),
        _part(tmp_path, "mid.json", "mid", run_id="b", record_length_s=7.4e-8),
        # 20 % short: capped at the same N as the coarse rung on a finer mesh.
        _part(tmp_path, "fine.json", "fine", run_id="c", record_length_s=5.9e-8),
    ]
    out = tmp_path / "full.json"
    result = _run("--merge", *[str(p) for p in parts], "--output", str(out))
    assert result.returncode == 3
    assert "recorded different lengths of time" in result.stderr
    assert "60000 / factor" in result.stderr, (
        "the refusal does not say how to fix it"
    )
    assert not out.exists()


def test_merge_refuses_when_a_rung_cannot_say_how_long_it_recorded(tmp_path):
    parts = [_part(tmp_path, "coarse.json", "coarse", run_id="a"),
             _part(tmp_path, "mid.json", "mid", run_id="b", record_length_s=None),
             _part(tmp_path, "fine.json", "fine", run_id="c")]
    out = tmp_path / "full.json"
    result = _run("--merge", *[str(p) for p in parts], "--output", str(out))
    assert result.returncode == 3
    assert "record_length_s is missing" in result.stderr
    assert "record_length_source" in result.stderr, (
        "the refusal does not say where to look"
    )


def test_merge_carries_each_rungs_witness(tmp_path):
    def w(delta21, delta11):
        return {"n_steps": 60000, "n2_steps": 120000, "floor_db": -20.0,
                "band_ghz": [2.0, 12.0],
                "max_abs_delta_s21_db": delta21,
                "f_ghz_at_max_abs_delta_s21": 6.1, "s21_bins_compared": 410,
                "max_abs_delta_s11_db": delta11,
                "f_ghz_at_max_abs_delta_s11": 3.2, "s11_bins_compared": 400}

    parts = [
        _part(tmp_path, "coarse.json", "coarse", run_id="a", witness=w(0.03, 0.02)),
        _part(tmp_path, "mid.json", "mid", run_id="b", witness=w(0.05, 0.04)),
        _part(tmp_path, "fine.json", "fine", run_id="c", witness=w(0.07, 0.06)),
    ]
    out = tmp_path / "full.json"
    result = _run("--merge", *[str(p) for p in parts], "--output", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    merged = json.loads(out.read_text())
    rw = merged["meta"]["record_length_witness"]
    assert set(rw) == {"coarse", "mid", "fine"}
    assert [rw[r]["max_abs_delta_s21_db"] for r in ("coarse", "mid", "fine")] == \
        [0.03, 0.05, 0.07]
    assert all(rw[r]["record_length_s"] == 7.4e-8 for r in rw)
    assert merged["meta"]["record_length_s_spread_pct"] == pytest.approx(0.0)
    assert merged["meta"]["record_length_s_tolerance_pct"] == 5.0
    assert "Reported, not gated" in result.stdout

    # Parts made without the flag leave the field null rather than an empty dict.
    plain = [_part(tmp_path, f"p_{r}.json", r, run_id=r)
             for r in ("coarse", "mid", "fine")]
    out2 = tmp_path / "full2.json"
    r2 = _run("--merge", *[str(p) for p in plain], "--output", str(out2))
    assert r2.returncode == 0, r2.stderr
    assert json.loads(out2.read_text())["meta"]["record_length_witness"] is None
