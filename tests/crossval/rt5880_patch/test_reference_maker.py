"""The RT5880 patch's reference maker stays runnable, and honest, with no openEMS.

``reference/make_openems_reference.py`` produces this case's openEMS record:
openEMS's own Simple Patch Antenna tutorial as the reproduce gate, then the
RT/Duroid 5880 probe-fed patch on three meshes. It is never run by CI -- the
solver runs by hand, on the cluster. These tests check what CI can afford to:
the maker imports, its two builders are still the texts they were copied from,
its constants are the retired script's, its dry run still plans the three rungs,
and its reproduce gate still refuses a curve with no resonance. Every check that
matters is paired with a mutation that re-introduces the defect it guards while
keeping every helper call in place, and the mutation must turn it red.

None of these steps an FDTD, reads a reference record, or asserts anything about
the board's physics.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MAKER = HERE / "reference" / "make_openems_reference.py"
REPO_ROOT = HERE.parents[2]
SHARED_GATE = REPO_ROOT / "tests" / "crossval" / "_openems_tutorial_gate.py"
RETIRED = REPO_ROOT / "validation" / "crossval" / "15_patch_antenna_rt5880.py"

# Owned by the MSL notch filter case's test (the byte-identity pin); listed here
# only to assert this maker keeps no second copy of any of them.
SHARED_HELPERS = [
    "_ensure_openems_numpy_compat",
    "_import_openems",
    "_BAD_STDOUT_PATTERNS",
    "_TRUNCATION_STDOUT_PATTERNS",
    "_scan_stdout_for_bad_patterns",
    "_run_openems_capturing_stdout",
    "_END_CRITERIA_NOT_REACHED_TEXT",
    "_log_indicates_truncation",
    "_check_excitation_and_trace",
    "_non_physical_guard",
    "_passivity_witness",
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_maker():
    return _load(MAKER, "_rt5880_patch_reference_maker")


def _run(*args: str, script: Path = MAKER) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, timeout=180)


class _Mutant:
    """A copy of the maker with one text replaced, written BESIDE the maker.

    Beside, not in a tmpdir: the maker finds the shared module and the retired
    script by walking up from its own path, and a copy elsewhere would fail for
    that reason instead of for the reason under test.
    """

    def __init__(self, old: str, new: str):
        src = MAKER.read_text()
        assert src.count(old) == 1, f"the mutation target is not unique in the maker: {old!r}"
        self.path = MAKER.with_name(f"_mutant_{abs(hash((old, new))) % 10**8}.py")
        self.text = src.replace(old, new, 1)

    def __enter__(self) -> Path:
        self.path.write_text(self.text)
        return self.path

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _self_check_on(old: str, new: str) -> subprocess.CompletedProcess:
    with _Mutant(old, new) as path:
        return _run("--self-check", script=path)


# ---------------------------------------------------------------------------
# The self-check and the dry run.
# ---------------------------------------------------------------------------
def test_self_check_passes_without_openems():
    result = _run("--self-check")
    assert result.returncode == 0, (
        f"--self-check exited {result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")
    assert "SELF-CHECK PASSED" in result.stdout


def test_the_self_check_itself_can_fail():
    """Mutation (a): invert the Stage B comparison; the self-check must go red."""
    result = _self_check_on(
        '        check(proof["matches"],',
        '        check(not proof["matches"],')
    assert result.returncode != 0
    assert "SELF-CHECK FAILED" in result.stdout


_SUBSTRATE_LINE = re.compile(
    r"substrate z cells\s+(\d+) of [0-9.]+ um across the 3\.175 mm board; the planned lines hold (\d+)")


def _planned_substrate_cells(stdout: str) -> list:
    return [(int(a), int(b)) for a, b in _SUBSTRATE_LINE.findall(stdout)]


def test_dry_run_plans_three_rungs_that_refine_the_substrate():
    result = _run("--dry-run", "--stage", "both")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "Simple_Patch_Antenna.py" in out, "the dry run does not name the reproduce gate"
    for stage in ("stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine"):
        assert stage in out, f"the dry run does not plan {stage}"
    for entry in _load_maker().DELTA_LIST:
        assert entry in out, f"the dry run does not print the delta entry {entry[:60]!r}"
    assert re.search(r"TOTAL \(the requested stages, smoke passes excluded\): [0-9,]+ s", out), (
        "the dry run prints no cost estimate")
    # The substrate is the direction that sets the effective permittivity; a
    # ladder that refines x and y only says nothing about where TM010 sits.
    assert _planned_substrate_cells(out) == [(4, 4), (6, 6), (8, 8)]


def test_dry_run_plan_reddens_when_the_planned_lines_stop_refining_the_substrate():
    """Mutation (b): the plan's z lines freeze at four cells; every call stays."""
    old = "    z += list(np.linspace(0, h, substrate_z_cells(resolution_factor) + 1))"
    with _Mutant(old, "    z += list(np.linspace(0, h, N_SUB + 1))") as path:
        result = _run("--dry-run", "--stage", "B", script=path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _planned_substrate_cells(result.stdout) != [(4, 4), (6, 6), (8, 8)], (
        "the dry-run test cannot see a plan whose lines no longer refine the substrate")


def test_delta_list_says_what_changes():
    m = _load_maker()
    d = m.DELTA_LIST
    assert [e.split(":")[0] for e in d] == [
        "DELTA 1 (boundaries)", "DELTA 2 (stop criteria)", "DELTA 3 (mesh rungs)",
        "DELTA 4 (frequency grid)", "DELTA 5 (resonance estimator)", "DELTA 6 (far field)",
        "DELTA 7 (how the solver runs)", "NOTHING ELSE"]
    assert "['MUR'] * 6 becomes ['PML_8'] * 6" in d[0] and "Sheen" in d[0]
    assert "NrTS=30000, EndCriteria=1e-4" in d[1] and "1e-5" in d[1]
    assert "0.70711" in d[2] and "4, 6 and 8 cells" in d[2] and "patch_res" in d[2]
    assert "901 points" in d[3] and "1.6-3.4 GHz" in d[3]
    assert "refined_extremum" in d[4] and "band_at_level" in d[4]
    assert "-6 dB" in d[5] and "Reported only" in d[5]
    # And the maker does what the list says.
    assert m.B_BOUNDARY == ["PML_8"] * 6
    assert m.B_REAL_NRTS is None and m.B_REAL_END_CRITERIA is None
    assert m.RETIRED_NRTS_CAP == 30000 and m.RETIRED_END_CRITERIA_CAP == 1e-4
    assert m.B_N_FREQS == 901 and (m.F_LO, m.F_HI) == (1.6e9, 3.4e9)
    assert [m.substrate_z_cells(f) for f in m.STAGE_B_FACTORS.values()] == [4, 6, 8]


# ---------------------------------------------------------------------------
# The constants and the Stage B copy proof.
# ---------------------------------------------------------------------------
def _module_assignments(text: str) -> dict:
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(t, ast.Name):
                out[t.id] = value
            elif isinstance(t, ast.Tuple) and all(isinstance(e, ast.Name) for e in t.elts):
                out[", ".join(e.id for e in t.elts)] = value
    return out


@pytest.mark.skipif(not RETIRED.is_file(), reason="the retired script has been removed; "
                    "the maker's frozen copy is then the only record")
def test_constants_and_frozen_slice_are_the_retired_scripts_own():
    """Read independently of the maker's own proof: ast on the retired file."""
    m = _load_maker()
    text = RETIRED.read_text()
    theirs = _module_assignments(text)
    for name in m.RETIRED_CONSTANT_LINES:
        mine = (m.F_LO, m.F_HI) if name == "F_LO, F_HI" else getattr(m, name)
        assert theirs[name] == mine, f"{name}: retired {theirs[name]!r}, maker {mine!r}"
    lines = text.split("\n")
    lo, hi = m.RETIRED_FROZEN_LINE_RANGE
    assert "\n".join(lines[lo - 1:hi]) == m._FROZEN_BUILDER_SLICE, (
        f"the frozen slice is not lines {lo}-{hi} of {RETIRED.name}")


def test_constants_check_reddens_when_a_constant_drifts():
    """Mutation (b): the substrate thickness drifts; the builder text is untouched."""
    result = _self_check_on("H_SUB = 3.175e-3            # 1/8 inch\nL_PATCH",
                            "H_SUB = 3.2e-3            # 1/8 inch\nL_PATCH")
    assert result.returncode != 0
    assert "[FAIL] H_SUB equals the retired line" in result.stdout


def test_copy_proof_reddens_when_the_board_builder_drifts():
    """Mutation (b): the probe moves; every constant and helper call is unchanged."""
    result = _self_check_on(
        "    feed = FEED_OFFSET_X * 1e3\n    f0, fc = 2.4e9, 1.2e9\n\n    FDTD = openEMS(**kw)",
        "    feed = 0.5 * FEED_OFFSET_X * 1e3\n    f0, fc = 2.4e9, 1.2e9\n\n    FDTD = openEMS(**kw)")
    assert result.returncode != 0
    assert "[FAIL] _build_patch_board_at_rung's block IS the frozen slice" in result.stdout


def test_copy_proof_reddens_when_a_mur_face_comes_back():
    """Mutation (b): the builder goes back to MUR walls, delta 1's whole reason."""
    result = _self_check_on(
        "    FDTD.SetGaussExcite(f0, fc)\n    FDTD.SetBoundaryCond(['PML_8'] * 6)\n",
        "    FDTD.SetGaussExcite(f0, fc)\n    FDTD.SetBoundaryCond(['MUR'] * 6)\n")
    assert result.returncode != 0
    assert "[FAIL] the builder carries neither the retired cap nor a MUR face" in result.stdout


def test_copy_proof_reddens_when_the_rung_stops_refining_the_substrate():
    """Mutation (b): the rung builder freezes the substrate at four cells."""
    result = _self_check_on(
        "    mesh.AddLine('z', np.linspace(0, h, substrate_z_cells(resolution_factor) + 1))\n"
        "    gnd = CSX.AddMetal('gnd')",
        "    mesh.AddLine('z', np.linspace(0, h, N_SUB + 1))\n"
        "    gnd = CSX.AddMetal('gnd')")
    assert result.returncode != 0
    assert "[FAIL] _build_patch_board_at_rung's block IS the frozen slice" in result.stdout


# ---------------------------------------------------------------------------
# The Stage A copy proof.
# ---------------------------------------------------------------------------
def test_stage_a_copy_proof_reddens_when_the_tutorial_builder_drifts():
    """Mutation (b): the tutorial's feed moves by one millimetre."""
    result = _self_check_on("    feed_pos = -6 #feeding position in x-direction",
                            "    feed_pos = -5 #feeding position in x-direction")
    assert result.returncode != 0
    assert "[FAIL] _build_stage_a_tutorial's block IS the frozen tutorial lines" in result.stdout


def test_stage_a_stop_criteria_are_read_from_the_tutorial_not_remembered():
    """Mutation (b): the real pass's step cap drifts from the tutorial's 30000."""
    result = _self_check_on("A_REAL_NRTS = 30000", "A_REAL_NRTS = 60000")
    assert result.returncode != 0
    assert "[FAIL] the real pass's NrTS / EndCriteria are read from the tutorial's own line" \
        in result.stdout


def test_the_frozen_tutorial_block_is_what_the_builder_runs():
    m = _load_maker()
    proof = m._copy_proof_a()
    assert proof["matches"] and proof["import_line_present"]
    assert "FDTD = openEMS(**kw)" in proof["mine"]
    assert "NrTS=30000" in m._FROZEN_TUTORIAL_BUILD
    assert m._free_names(m._FROZEN_TUTORIAL_BUILD) == m.A_FREE_NAMES
    assert len(m.A_TUTORIAL_SHA256) == 64 and int(m.A_TUTORIAL_SHA256, 16) >= 0


# ---------------------------------------------------------------------------
# The reproduce gate, on synthetic inputs.
# ---------------------------------------------------------------------------
def _flat_curve(m):
    """No resonance: |S11| 0.995 with a 0.1 dB ripple at the documented frequency."""
    f = m.stage_a_freqs_hz()
    s11 = 0.995 * np.exp(-1j * f / 1e9) * (1 - 0.01 * np.exp(-((f - 2.43e9) / 5e7) ** 2))
    return f, s11


def _gate_on_curve(m, f, s11):
    sf = m._gate.load_spectral_features()
    feats = m._one_port_features(sf, m.STAGE_A_BAND_HZ, with_tutorial_pick=True)(
        f, s11, np.full(f.shape, 50.0 + 0j))
    r = feats["resonance"]
    return m.stage_a_gate_verdict(r["refined_f_ghz"] * 1e9, r["depth_db"]), feats


def test_the_reproduce_gate_on_synthetic_inputs():
    m = _load_maker()
    doc = m.STAGE_A_DOCUMENTED
    assert m.stage_a_gate_verdict(doc["f_dip_hz"], doc["depth_db"])["passed"]
    off = m.stage_a_gate_verdict(2.52e9, -30.0)
    assert not off["passed"] and not off["f_ok"] and off["depth_ok"]
    shallow = m.stage_a_gate_verdict(2.43e9, -8.0)
    assert not shallow["passed"] and shallow["f_ok"] and not shallow["depth_ok"]
    verdict, feats = _gate_on_curve(m, *_flat_curve(m))
    assert not verdict["passed"], "a curve with no resonance passed the reproduce gate"
    assert feats["tutorial_pick"]["f_hz"] is None
    assert feats["resonance"]["band_minus10db"] is None


def test_the_reproduce_gate_reddens_when_it_stops_reading_the_depth():
    """Mutation (b): the verdict drops its depth term; a curve with no dip passes."""
    old = '        "passed": bool(f_ok and depth_ok),\n    }\n\n\n# ----'
    new = '        "passed": bool(f_ok),\n    }\n\n\n# ----'
    with _Mutant(old, new) as path:
        mutant = _load(path, "_rt5880_patch_reference_maker_mutant")
        verdict, _ = _gate_on_curve(mutant, *_flat_curve(mutant))
        result = _run("--self-check", script=path)
    assert verdict["passed"], "the mutation did not re-open the hole the test guards"
    assert result.returncode != 0
    assert "[FAIL] a curve with no dip" in result.stdout


# ---------------------------------------------------------------------------
# What the maker does not own, and what the record says about itself.
# ---------------------------------------------------------------------------
def test_the_shared_helpers_are_not_copied_into_this_maker():
    shared = {
        node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else node.targets[0].id
        for node in ast.parse(SHARED_GATE.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name))
    }
    assert set(SHARED_HELPERS) <= shared
    offenders = [
        node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else node.targets[0].id
        for node in ast.parse(MAKER.read_text()).body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in SHARED_HELPERS)
        or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in SHARED_HELPERS)
    ]
    assert not offenders, f"{MAKER.name} carries its own copy of {offenders}"


def test_the_reported_features_carry_no_verdict():
    m = _load_maker()
    sf = m._gate.load_spectral_features()
    f = m.stage_b_freqs_hz()
    x = 2 * 10.0 * (f - 2.33e9) / 2.33e9
    g = -1j * x / (2.0 + 1j * x)
    feats = m._one_port_features(sf, m.B_RESONANCE_BAND_HZ)(f, g, 50 * (1 + g) / (1 - g))

    def keys(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield f"{prefix}{k}"
                yield from keys(v, f"{prefix}{k}.")

    verdicts = [k for k in keys(feats)
                if k.rsplit(".", 1)[-1] in ("passed", "ok", "gate", "tol", "verdict")]
    assert not verdicts, f"a verdict field appeared in the reported features: {verdicts}"
    assert feats["resonance"]["refined_f_ghz"] == pytest.approx(2.33, abs=1e-3)


def test_the_record_names_its_provenance():
    m = _load_maker()
    art = m._build_artifact({}, {}, {}, ["stage_a", "stage_b_coarse"])
    meta = art["meta"]
    for key in ("rfx_openems_commit", "rfx_openems_image", "stage_a_tutorial",
                "stage_a_documented", "stage_a_gate_rule", "delta_list", "copy_proof"):
        assert key in meta, f"the record's meta has no {key}"
    assert meta["stage_a_tutorial"]["sha256"] == m.A_TUTORIAL_SHA256
    assert meta["stage_a_tutorial"]["openems_commit"] == m.A_TUTORIAL_OPENEMS_COMMIT
    assert art["run_id"] is None and meta["ci_runs_this"] is False
    assert set(m.STAGE_NAMES) <= set(art)
