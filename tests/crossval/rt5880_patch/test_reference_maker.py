"""The RT5880 patch's reference maker stays runnable, and honest, with no openEMS.

``reference/make_openems_reference.py`` produces this case's openEMS record:
openEMS's own Simple Patch Antenna tutorial as the reproduce gate, then the
RT/Duroid 5880 probe-fed patch on three meshes. It is never run by CI -- the
solver runs by hand, on the cluster. These tests check what CI can afford to:
the maker imports; its two builders are still the texts they were copied from;
its substitutions, its reproduce-gate window and its record's provenance keys
are the ones written here as literals; its builders, run against the maker's
own recording stand-in for openEMS and CSXCAD, still hand the solver the
declared stop criteria, boundaries and materials, and still put the probe
port, the thirds-rule lines and the absorber faces on exact lines on every
rung. Every check is paired with a mutation that re-introduces the defect it
guards while keeping every helper call in place, and the mutation turns it red.

None of these steps an FDTD, reads a reference record, or asserts anything about
the board's physics.
"""

from __future__ import annotations

import ast
import importlib.util
import os
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

# ---------------------------------------------------------------------------
# THE PINS. Written out here, not read from the maker: a change to any of them
# has to be made twice, once in the maker and once here, on purpose.
# ---------------------------------------------------------------------------
PINNED_STAGE_A_WINDOW_REL = 0.01
PINNED_A_COPY_SUBSTITUTIONS = [
    ("FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)", "FDTD = openEMS(**kw)"),
]
PINNED_B_COPY_SUBSTITUTIONS = [
    ("    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)",
     "    FDTD = openEMS(**kw)"),
    ("    FDTD.SetBoundaryCond(['MUR'] * 6)",
     "    FDTD.SetBoundaryCond(['PML_8'] * 6)"),
]
PINNED_B_RUNG_SUBSTITUTIONS = [
    ("    mesh_res = C0 / (f0 + fc) / unit / 30        # ~lambda/30 in air (well-resolved)",
     "    mesh_res = C0 / (f0 + fc) / unit / 30 * resolution_factor  # ~lambda/30 in air, x rung"),
    ("    patch_res = 1.2                              # dense lines across patch (converged ref)",
     "    patch_res = 1.2 * resolution_factor          # dense lines across patch, x rung"),
    ("    mesh.AddLine('z', np.linspace(0, h, N_SUB + 1))",
     "    mesh.AddLine('z', np.linspace(0, h, substrate_z_cells(resolution_factor) + 1))"),
]
PINNED_B_EDGE_SUBSTITUTIONS = [
    ("    mesh.AddLine('x', np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res))",
     "    mesh.AddLine('x', _clear_comb(np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, "
     "patch_res), 'x', patch_res))"),
    ("    mesh.AddLine('y', np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res))",
     "    mesh.AddLine('y', _clear_comb(np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, "
     "patch_res), 'y', patch_res))"),
    ("    mesh.SmoothMeshLines('all', mesh_res, 1.4)",
     "    mesh.SmoothMeshLines('all', mesh_res, 1.4)\n    _absorber_cells_outside(mesh, B_PML_CELLS)"),
]
PINNED_PROVENANCE_KEYS = {
    "openems", "rfx_openems_commit", "rfx_openems_image", "rfx_commit",
    "stage_a_tutorial", "stage_a_tutorial_in_this_container", "stage_a_documented",
    "stage_a_recorded_reproduction", "stage_a_gate_rule", "stage_a_gate",
    "reproduce_gate_ran", "stages", "delta_list", "copy_proof", "stop_criteria",
    "line_check", "record_complete", "stages_completed", "produced_by", "ci_runs_this",
}
PINNED_TUTORIAL_SHA256 = "33017faa27dc22134fe56964b5cb56623c4043916df01921f27ee54ad1b6a79a"
PINNED_TUTORIAL_OPENEMS_COMMIT = "2000574e8785667a4ad107ad32a304a5034d872a"
PINNED_RUNGS = {"stage_b_coarse": 1.0, "stage_b_mid": 2 ** -0.5, "stage_b_fine": 0.5}
PINNED_B_STOP = {"NrTS": 1_000_000_000, "EndCriteria": 1e-5}
PINNED_A_STOP = {"NrTS": 30000, "EndCriteria": 1e-4}
PINNED_ABSORBER_FACES = {"x": [-88.0, 88.0], "y": [-93.0, 93.0], "z": [-40.0, 90.0]}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_maker():
    return _load(MAKER, "_rt5880_patch_reference_maker")


def _run(*args: str, script: Path = MAKER) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "RFX_CSXCAD_SMOOTHMESHLINES"}
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, timeout=300, env=env)


class _Mutant:
    """A copy of the maker with texts replaced, written BESIDE the maker.

    Beside, not in a tmpdir: the maker finds the shared module and the retired
    script by walking up from its own path, and a copy elsewhere would fail for
    that reason instead of for the reason under test.
    """

    def __init__(self, *pairs):
        src = MAKER.read_text()
        for old, new in pairs:
            assert src.count(old) == 1, f"the mutation target is not unique in the maker: {old!r}"
            src = src.replace(old, new, 1)
        self.path = MAKER.with_name(f"_mutant_{abs(hash(pairs)) % 10**8}.py")
        self.text = src

    def __enter__(self) -> Path:
        self.path.write_text(self.text)
        return self.path

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _self_check_on(*pairs) -> subprocess.CompletedProcess:
    with _Mutant(*pairs) as path:
        return _run("--self-check", script=path)


def _on_mutant(check, *pairs):
    """Run ``check`` on a mutant module; return the AssertionError it raised."""
    with _Mutant(*pairs) as path:
        mutant = _load(path, f"_rt5880_mutant_{abs(hash(pairs)) % 10**8}")
        with pytest.raises(AssertionError) as caught:
            check(mutant)
    return caught.value


# ---------------------------------------------------------------------------
# The checks, as functions, so each runs on the maker AND on its mutant.
# ---------------------------------------------------------------------------
def _check_window(m) -> None:
    assert m.STAGE_A_WINDOW_REL == PINNED_STAGE_A_WINDOW_REL
    f = m.STAGE_A_DOCUMENTED["f_dip_hz"]
    assert m.stage_a_gate_verdict(f * 1.009, -20.0)["passed"]
    assert not m.stage_a_gate_verdict(f * 1.015, -20.0)["passed"], (
        "a dip 1.5 % off the documented one passed the reproduce gate")


def _check_substitutions(m) -> None:
    assert m.A_COPY_SUBSTITUTIONS == PINNED_A_COPY_SUBSTITUTIONS
    assert m.B_COPY_SUBSTITUTIONS == PINNED_B_COPY_SUBSTITUTIONS
    assert m.B_RUNG_SUBSTITUTIONS == PINNED_B_RUNG_SUBSTITUTIONS
    assert m.B_EDGE_SUBSTITUTIONS == PINNED_B_EDGE_SUBSTITUTIONS


def _check_provenance(m) -> None:
    art = m._build_artifact({}, {}, {}, ["stage_a", "stage_b_coarse"])
    missing = PINNED_PROVENANCE_KEYS - set(art["meta"])
    assert not missing, f"the record's meta lost {sorted(missing)}"
    assert art["meta"]["stage_a_tutorial"]["sha256"] == PINNED_TUTORIAL_SHA256
    assert art["meta"]["stage_a_tutorial"]["openems_commit"] == PINNED_TUTORIAL_OPENEMS_COMMIT
    assert art["meta"]["stop_criteria"]["stage_b_real"] == PINNED_B_STOP
    assert art["meta"]["stop_criteria"]["stage_a_real"] == PINNED_A_STOP
    assert art["run_id"] is None and art["meta"]["ci_runs_this"] is False


def _check_stand_in(m) -> None:
    """The builders, run against the maker's recording stand-in."""
    plans = m._stage_plans(list(m.STAGE_NAMES))
    a = plans["stage_a"]
    assert a["stand_in"]["kw"] == PINNED_A_STOP
    assert a["stand_in"]["boundary"] == ["MUR"] * 6
    assert a["stand_in"]["materials"]["substrate"]["epsilon"] == 3.38
    assert a["stand_in"]["ports"][0]["start"] == [-6.0, 0.0, 0.0]
    assert a["line_check"]["passed"], a["line_check"]["failures"]
    for name, f in PINNED_RUNGS.items():
        b = plans[name]
        assert b["resolution_factor"] == pytest.approx(f, rel=1e-15)
        assert b["stand_in"]["kw"] == PINNED_B_STOP, (
            f"{name}: handed {b['stand_in']['kw']}, pinned {PINNED_B_STOP}")
        assert b["stand_in"]["boundary"] == ["PML_8"] * 6, f"{name}: {b['stand_in']['boundary']}"
        assert b["stand_in"]["excite"] == (2.4e9, 1.2e9)
        assert b["stand_in"]["materials"]["sub"]["epsilon"] == 2.2
        port = b["stand_in"]["ports"][0]
        # The probe at -8.73125 mm, a node on every rfx rung (PI 2026-09-24,
        # delta 9), not the retired -9.0 mm.
        assert port["start"] == [-8.73125, 0.0, 0.0] and port["stop"] == [-8.73125, 0.0, 3.175]
        x = np.asarray(b["_lines"]["x"])
        pitch = float(np.median(np.diff(x[(x > -18.0) & (x < -10.0)])))
        assert pitch == pytest.approx(1.2 * f, rel=1e-9), (
            f"{name}: the lines across the patch are {pitch} mm apart, not 1.2 x {f}")
        assert b["absorber_inner_faces_mm"] == PINNED_ABSORBER_FACES, name
        assert b["line_check"]["passed"], f"{name}: {b['line_check']['failures']}"
        z = np.asarray(b["_lines"]["z"])
        assert int(np.sum((z >= 0.0) & (z <= 3.175))) - 1 == round(4 / f)


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
    result = _self_check_on(('        check(proof["matches"],',
                             '        check(not proof["matches"],'))
    assert result.returncode != 0
    assert "SELF-CHECK FAILED" in result.stdout


def test_dry_run_prints_the_realized_line_tables_and_the_cost():
    result = _run("--dry-run", "--stage", "both")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "Simple_Patch_Antenna.py" in out, "the dry run does not name the reproduce gate"
    for stage in ("stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine"):
        assert stage in out, f"the dry run does not plan {stage}"
    for entry in _load_maker().DELTA_LIST:
        assert entry in out, f"the dry run does not print the delta entry {entry[:60]!r}"
    assert out.count("realized-line check     PASSED") == 4
    for what in ("lines near patch x-", "lines near patch x+", "lines near port y",
                 "worst neighbour ratio", "absorber inner faces"):
        assert what in out, f"the dry run does not print {what!r}"
    assert re.findall(r"substrate +(\d+) cells of", out) == ["4", "6", "8"]
    assert re.search(r"TOTAL \(the requested stages, smoke passes excluded\): [0-9,]+ s", out)


def test_dry_run_fails_when_a_rung_stops_refining_the_substrate():
    """Mutation (b): the rung builder freezes the substrate at four cells (the
    copy proof would say so too; here the dry run's own line check does)."""
    with _Mutant(("    mesh.AddLine('z', np.linspace(0, h, substrate_z_cells(resolution_factor) + 1))\n"
                  "    gnd = CSX.AddMetal('gnd')",
                  "    mesh.AddLine('z', np.linspace(0, h, N_SUB + 1))\n"
                  "    gnd = CSX.AddMetal('gnd')")) as path:
        result = _run("--dry-run", "--stage", "B", script=path)
    assert result.returncode != 0
    assert "the substrate carries 4 z cells, not 6" in result.stdout


def test_delta_list_says_what_changes():
    m = _load_maker()
    d = m.DELTA_LIST
    assert [e.split(":")[0] for e in d] == [
        "DELTA 1 (boundaries)", "DELTA 2 (stop criteria)", "DELTA 3 (mesh rungs)",
        "DELTA 4 (frequency grid)", "DELTA 5 (resonance estimator)", "DELTA 6 (far field)",
        "DELTA 7 (how the solver runs)",
        "DELTA 8 (the patch lines keep clear of the port and the edges)",
        "DELTA 9 (probe position)", "NOTHING ELSE"]
    assert "['MUR'] * 6 becomes ['PML_8'] * 6" in d[0] and "OUTSIDE" in d[0]
    assert "EndCriteria = 1e-5" in d[1] and "1e-6" in d[1] and "openems.cpp:117" in d[1]
    assert "0.70711" in d[2] and "4, 6 and 8 cells" in d[2]
    assert "901 points" in d[3] and "1.6-3.4 GHz" in d[3]
    assert "refined_extremum" in d[4] and "band_at_level" in d[4]
    assert "-6 dB" in d[5] and "Reported only" in d[5]
    assert "half a patch cell" in d[7] and "y = 0" in d[7]
    assert "-9.0e-3 becomes -8.73125e-3" in d[8] and "-11, -22 and -33 cells" in d[8]
    was, now, _ = m.DECLARED_CONSTANT_DEPARTURES["FEED_OFFSET_X"]
    assert (was, now) == (-9.0e-3, -8.73125e-3) and m.FEED_OFFSET_X == now
    assert m.RETIRED_CONSTANT_LINES["FEED_OFFSET_X"].startswith("FEED_OFFSET_X = -9.0e-3")
    assert m.COMB_CLEARANCE_CELLS == 0.5
    assert m.RETIRED_NRTS_CAP == 30000 and m.RETIRED_END_CRITERIA_CAP == 1e-4
    assert m.OPENEMS_UNSET_END_CRITERIA == 1e-6


# ---------------------------------------------------------------------------
# The pins, and the reviewer's five mutations (M1-M5), each turned red.
# ---------------------------------------------------------------------------
def test_the_pins_hold_on_the_maker():
    m = _load_maker()
    _check_window(m)
    _check_substitutions(m)
    _check_provenance(m)
    _check_stand_in(m)


def test_m1_a_wider_reproduce_window_is_red():
    err = _on_mutant(_check_window, ("STAGE_A_WINDOW_REL = 0.01", "STAGE_A_WINDOW_REL = 0.03"))
    assert "0.03" in str(err) or "passed the reproduce gate" in str(err)


def test_m2_unscaled_patch_lines_are_red_even_with_their_substitution_removed():
    """The copy proof stays green on this mutant -- builder and declared list were
    changed together -- so only the pins and the stand-in can see it."""
    pairs = (
        ('''    ("    patch_res = 1.2                              # dense lines across patch (converged ref)",
     "    patch_res = 1.2 * resolution_factor          # dense lines across patch, x rung"),
''', ""),
        ("    patch_res = 1.2 * resolution_factor          # dense lines across patch, x rung",
         "    patch_res = 1.2                              # dense lines across patch (converged ref)"),
    )
    _on_mutant(_check_substitutions, *pairs)
    err = _on_mutant(_check_stand_in, *pairs)
    assert "apart" in str(err)
    with _Mutant(*pairs) as path:
        assert _load(path, "_rt5880_m2")._copy_proof_b()["matches"], (
            "the mutant was meant to keep the copy proof green")


def test_m3_a_changed_tutorial_permittivity_is_red():
    pairs = (
        ('''A_COPY_SUBSTITUTIONS = [
    ("FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)", "FDTD = openEMS(**kw)"),
]''', '''A_COPY_SUBSTITUTIONS = [
    ("FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)", "FDTD = openEMS(**kw)"),
    ("substrate_epsR   = 3.38", "substrate_epsR   = 3.48"),
]'''),
        ("    substrate_epsR   = 3.38\n", "    substrate_epsR   = 3.48\n"),
    )
    _on_mutant(_check_substitutions, *pairs)
    _on_mutant(_check_stand_in, *pairs)
    with _Mutant(*pairs) as path:
        assert _load(path, "_rt5880_m3")._copy_proof_a()["matches"], (
            "the mutant was meant to keep the Stage A copy proof green")


def test_m4_a_record_without_the_rfx_commit_is_red():
    err = _on_mutant(_check_provenance,
                     ('        "rfx_commit": os.environ.get("RFX_COMMIT"),\n', ""))
    assert "rfx_commit" in str(err)


def test_m5_a_thinner_absorber_is_red_even_with_its_substitution():
    pairs = (
        ('''    ("    FDTD.SetBoundaryCond(['MUR'] * 6)",
     "    FDTD.SetBoundaryCond(['PML_8'] * 6)"),''',
         '''    ("    FDTD.SetBoundaryCond(['MUR'] * 6)",
     "    FDTD.SetBoundaryCond(['PML_4'] * 6)"),'''),
        ("    FDTD.SetGaussExcite(f0, fc)\n    FDTD.SetBoundaryCond(['PML_8'] * 6)\n",
         "    FDTD.SetGaussExcite(f0, fc)\n    FDTD.SetBoundaryCond(['PML_4'] * 6)\n"),
    )
    _on_mutant(_check_substitutions, *pairs)
    err = _on_mutant(_check_stand_in, *pairs)
    assert "PML_4" in str(err)


# ---------------------------------------------------------------------------
# The realized-line check (review P1 and P2a) and the stop criteria (P2b).
# ---------------------------------------------------------------------------
def test_the_line_check_catches_the_lost_probe_line_without_delta_8():
    """The reviewer's P1, reproduced: the retired comb at the finest rung puts a
    line at y = 6.4e-14 mm, CSXCAD's rule deletes the lower line of the pair --
    the probe's y = 0 -- and the check must say so."""
    m = _load_maker()
    bare = m._without_delta_8_realized(0.5)
    check = m._line_check(bare["lines"], m._line_spec_b(0.5))
    assert not check["passed"]
    assert any(f.startswith("probe port y: no realized y line at 0.0 mm") for f in check["failures"])
    assert m._csxcad_unique([0.0, 6.4e-14, 1.0, 2.0]).tolist() == [6.4e-14, 1.0, 2.0], (
        "the near-duplicate rule must delete the LOWER line, as CSXCAD's Unique does")


def test_delta_8_removed_from_the_builder_is_red():
    """Mutation (b): drop the clearance from both the builder and its declared
    list; the copy proof stays green, the stand-in's line check does not."""
    pairs = (
        ("    mesh.AddLine('x', _clear_comb(np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res), 'x', patch_res))\n"
         "    mesh.AddLine('y', _clear_comb(np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res), 'y', patch_res))\n\n"
         "    patch = CSX",
         "    mesh.AddLine('x', np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res))\n"
         "    mesh.AddLine('y', np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res))\n\n"
         "    patch = CSX"),
        ('''    ("    mesh.AddLine('x', np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res))",
     "    mesh.AddLine('x', _clear_comb(np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res), 'x', patch_res))"),
    ("    mesh.AddLine('y', np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res))",
     "    mesh.AddLine('y', _clear_comb(np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res), 'y', patch_res))"),
''', ""),
    )
    err = _on_mutant(_check_stand_in, *pairs)
    assert "probe port y" in str(err) or "closer than" in str(err)
    with _Mutant(*pairs) as path:
        assert _load(path, "_rt5880_d8")._copy_proof_b()["matches"]


def test_absorber_carved_inside_the_box_is_red():
    """Mutation (b): the eight absorber cells are no longer laid outside."""
    pairs = (
        ("    mesh.SmoothMeshLines('all', mesh_res, 1.4)\n    _absorber_cells_outside(mesh, B_PML_CELLS)\n"
         "    nf2ff = FDTD.CreateNF2FFBox() if do_gain else None\n    return FDTD, port, nf2ff",
         "    mesh.SmoothMeshLines('all', mesh_res, 1.4)\n"
         "    nf2ff = FDTD.CreateNF2FFBox() if do_gain else None\n    return FDTD, port, nf2ff"),
        ('''    ("    mesh.SmoothMeshLines('all', mesh_res, 1.4)",
     "    mesh.SmoothMeshLines('all', mesh_res, 1.4)\\n    _absorber_cells_outside(mesh, B_PML_CELLS)"),
''', ""),
    )
    err = _on_mutant(_check_stand_in, *pairs)
    assert "stage_b" in str(err) or "absorber" in str(err)


def test_leaving_end_criteria_unset_is_red():
    """Mutation (b): back to passing nothing, which in the pinned build means 1e-6."""
    err = _on_mutant(_check_stand_in, ("B_REAL_END_CRITERIA = 1e-5 ", "B_REAL_END_CRITERIA = None "))
    assert "EndCriteria" in str(err)


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
    result = _self_check_on(("H_SUB = 3.175e-3            # 1/8 inch\nL_PATCH",
                             "H_SUB = 3.2e-3            # 1/8 inch\nL_PATCH"))
    assert result.returncode != 0
    assert "[FAIL] H_SUB equals the retired line" in result.stdout


def test_copy_proof_reddens_when_the_board_builder_drifts():
    """Mutation (b): the probe moves; every constant and helper call is unchanged."""
    result = _self_check_on(
        ("    feed = FEED_OFFSET_X * 1e3\n    f0, fc = 2.4e9, 1.2e9\n\n    FDTD = openEMS(**kw)",
         "    feed = 0.5 * FEED_OFFSET_X * 1e3\n    f0, fc = 2.4e9, 1.2e9\n\n    FDTD = openEMS(**kw)"))
    assert result.returncode != 0
    assert "[FAIL] _build_patch_board_at_rung's block IS the frozen slice" in result.stdout


def test_copy_proof_reddens_when_a_mur_face_comes_back():
    """Mutation (b): the builder goes back to MUR walls, delta 1's whole reason."""
    result = _self_check_on(
        ("    FDTD.SetGaussExcite(f0, fc)\n    FDTD.SetBoundaryCond(['PML_8'] * 6)\n",
         "    FDTD.SetGaussExcite(f0, fc)\n    FDTD.SetBoundaryCond(['MUR'] * 6)\n"))
    assert result.returncode != 0
    assert "[FAIL] the builder carries neither the retired cap nor a MUR face" in result.stdout


# ---------------------------------------------------------------------------
# The Stage A copy proof and the container check.
# ---------------------------------------------------------------------------
def test_stage_a_copy_proof_reddens_when_the_tutorial_builder_drifts():
    """Mutation (b): the tutorial's feed moves by one millimetre."""
    result = _self_check_on(("    feed_pos = -6 #feeding position in x-direction",
                             "    feed_pos = -5 #feeding position in x-direction"))
    assert result.returncode != 0
    assert "[FAIL] _build_stage_a_tutorial's block IS the frozen tutorial lines" in result.stdout


def test_stage_a_stop_criteria_are_read_from_the_tutorial_not_remembered():
    """Mutation (b): the real pass's step cap drifts from the tutorial's 30000."""
    result = _self_check_on(("A_REAL_NRTS = 30000", "A_REAL_NRTS = 60000"))
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
    assert m.A_TUTORIAL_SHA256 == PINNED_TUTORIAL_SHA256


def test_a_container_without_the_tutorial_is_refused(tmp_path):
    m = _load_maker()
    info = m._tutorial_source_in_image(str(tmp_path / "absent.py"))
    assert not info["present"]
    assert "carries no tutorial" in (m._tutorial_source_refusal(info) or "")


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
    with _Mutant((old, new)) as path:
        mutant = _load(path, "_rt5880_patch_reference_maker_mutant")
        verdict, _ = _gate_on_curve(mutant, *_flat_curve(mutant))
        result = _run("--self-check", script=path)
    assert verdict["passed"], "the mutation did not re-open the hole the test guards"
    assert result.returncode != 0
    assert "[FAIL] a curve with no dip" in result.stdout


# ---------------------------------------------------------------------------
# What the maker does not own.
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
