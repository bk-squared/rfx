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
