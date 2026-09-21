"""docs/research_notes/ and docs/agent-memory/ are local-only. Keep them that way.

This is a PUBLIC repository. ``docs/.gitignore`` ignores both directories, and
the repo's own operating standard says so in prose: agent-memory is "NOT
version-controlled here", research_notes is "Local only -- gitignored in this
public repo". Neither statement is self-enforcing. ``git add -f`` overrides a
gitignore without warning, and between 2026-09-08 and 2026-09-14 fifteen
commits force-added 1060 files (356.3 MiB, of which 344.8 MiB is 209 .npz/.npy
arrays)
under ``docs/research_notes/`` -- publishing internal chronology, and naming
1060 public paths after private issue numbers, without anyone deciding to.

Nobody chose that; the absence of a check let it accumulate. The content now
lives in the private workspace repo under ``docs/research-archive/rfx/``, kept
current by ``scripts/sync_research_archive.sh``; oversize artifacts the mirror
filters out are bundled on NFS. This gate keeps the public side clean.

If a file under either directory is load-bearing for a committed test or a
gated citation, that file is not a note -- move it to ``tests/fixtures/`` or
next to its consumer and cite it there. The numeric-provenance gate already
says as much when it refuses an untracked artifact: "Commit the artifact or
cite one that is committed."

WHERE THERE IS NO REPOSITORY, THIS GATE SKIPS. The GPU suite runs against a
``git archive`` export with no ``.git``, and the question here -- is this file
TRACKED though ignored -- has no oracle without git. The ignored files are
simply absent from an export, so their absence is evidence of nothing and a
pass would be a guess. Skipping says so.

Its sibling ``tests/contracts/test_guide_citations_resolve.py`` does NOT skip
there, and the difference is the question rather than the tooling: that gate
asks whether a link target is PRESENT for someone who clones, and an export is
a valid witness for that, holding the tracked files and nothing else. Same
missing tool, two different questions (review of PR #1135, B).
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tests._git_tracked import git_available

REPO = Path(__file__).resolve().parents[2]

LOCAL_ONLY = ("docs/research_notes", "docs/agent-memory")


def _tracked_under(repo: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Paths git tracks under *prefixes*, in *repo*. Empty when none."""
    result = subprocess.run(
        ["git", "ls-files", "--", *prefixes],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"git ls-files failed: {result.stderr.strip()}"
    return [line for line in result.stdout.splitlines() if line]


@pytest.mark.skipif(
    not git_available(REPO),
    reason="not a git checkout (wheel install, source tarball, or no git binary) -- "
           "tracking is not a question that can be answered here",
)
def test_local_only_doc_directories_are_not_tracked() -> None:
    tracked = _tracked_under(REPO, LOCAL_ONLY)
    assert not tracked, (
        f"{len(tracked)} file(s) under {LOCAL_ONLY} are git-tracked in this PUBLIC "
        f"repo. docs/.gitignore ignores both; these got in past it, which takes "
        f"`git add -f`. First five: {tracked[:5]}. Untrack with "
        f"`git rm -r --cached docs/research_notes docs/agent-memory` (the files stay "
        f"on disk). If one of them is load-bearing for a test or a gated citation, "
        f"move that file to tests/fixtures/ or beside its consumer and re-point the "
        f"reference -- do not re-add the directory."
    )


@pytest.mark.skipif(
    not git_available(REPO),
    reason="not a git checkout -- gitignore behaviour is not observable here",
)
def test_gitignore_still_ignores_both_directories() -> None:
    """Untracking is only durable while the ignore rule stands behind it.

    Checked through ``git check-ignore`` rather than by grepping the file, so a
    later negation elsewhere in the ignore chain is caught too.
    """
    for prefix in LOCAL_ONLY:
        probe = f"{prefix}/__ignore_probe__.md"
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", probe],
            cwd=REPO, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"{probe} is NOT ignored (git check-ignore returned "
            f"{result.returncode}). docs/.gitignore must keep ignoring {prefix}/ or "
            f"the next write there lands in the public repo by default."
        )


@pytest.mark.skipif(
    not git_available(REPO),
    reason="not a git checkout -- a scratch repo cannot be built without git",
)
def test_the_gate_fires_on_a_repository_that_does_track_them(tmp_path: Path) -> None:
    """Criterion (B): a gate that cannot go red is not a gate.

    The real check reads green on a clean tree, which is also what a broken
    check reads. This builds the failure case -- a repository that force-adds a
    file under each local-only directory -- and asserts the same predicate
    reports it.
    """
    for argv in (["git", "init", "-q"],
                 ["git", "config", "user.email", "probe@example.invalid"],
                 ["git", "config", "user.name", "probe"]):
        assert subprocess.run(argv, cwd=tmp_path, capture_output=True).returncode == 0

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / ".gitignore").write_text("/research_notes/\n/agent-memory/\n")
    for prefix in LOCAL_ONLY:
        target = tmp_path / prefix
        target.mkdir(parents=True)
        (target / "note.md").write_text("forced past the ignore rule\n")

    assert _tracked_under(tmp_path, LOCAL_ONLY) == [], (
        "the ignore rule should keep these out until someone forces them"
    )
    forced = subprocess.run(
        ["git", "add", "-f", "docs/research_notes", "docs/agent-memory"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert forced.returncode == 0, forced.stderr

    assert sorted(_tracked_under(tmp_path, LOCAL_ONLY)) == [
        "docs/agent-memory/note.md",
        "docs/research_notes/note.md",
    ], "the predicate did not see force-added files -- it cannot protect this repo"


# ---------------------------------------------------------------------------
# The committed sweep.
#
# Untracking the directories is half the job. The other half is that no
# committed code may RESOLVE a path inside them, because such a path exists in
# one checkout and in no clone -- green here, red for everyone else, and
# invisible to a local test run because `git rm --cached` leaves the files on
# disk.
#
# The 2026-09-15 pass first looked for those readers with a line grep for
# "research_notes" next to an open/load/import call. It missed
# tests/unit/sparams/test_msl_dump_power_replay.py, which splits both of its
# reads across two lines, and it could not have seen a path assembled segment by
# segment (`ROOT / "docs" / "research_notes" / name`) at all, because no line
# contains the joined string. This sweep replaces the grep: it parses each file
# and looks for the BARE SEGMENT in any string constant that is not a docstring,
# which catches both forms.
#
# It is not exhaustive over the repository. It reads Python only, so a path
# built in a shell script, a YAML, or from a variable is outside it, and a
# constant assembled at runtime from pieces that are individually innocuous
# would slip through. What it does guarantee is that a literal segment cannot
# appear in committed Python without a reviewer having written down why.
# ---------------------------------------------------------------------------

# Every tracked .py, with no root filter. An earlier revision listed roots
# (tests/, validation/, scripts/, examples/, rfx/) and thereby missed the two
# files that sit outside them: the repo-root conftest.py, whose breakage would
# red every fresh clone, and docs/design_notes/931_migration/t3_remeasure.py.
# Both are clean today. A root list has to be extended every time someone adds a
# top-level directory, and nothing makes that failure visible -- so there is no
# root list.
SEGMENTS = ("research_notes", "agent-memory", "agent_memory")

# path -> (how many segment literals it may contain, why each is not a read of a
# committed artifact). A count, not just a name: allow-listing a whole file
# forever would let a real reader be added to it unseen. Adding a usage to a
# file below fails this gate until someone bumps the number and extends the
# reason.
REVIEWED = {
    # Reject-lists: these name the path so that a log_path may NOT start with it.
    "tests/crossval/test_coax_two_port_referee_header.py":
        (1, "forbidden-prefix guard, not a read"),
    "tests/crossval/test_msl_phase_referee_header.py":
        (1, "forbidden-prefix guard, not a read"),
    "tests/crossval/test_sheen_lpf_header.py":
        (1, "forbidden-prefix guard, not a read"),
    "tests/crossval/test_waveguide_bend_header.py":
        (1, "forbidden-prefix guard, not a read"),

    # Assertion / warning text that happens to cite a note. No path resolution.
    "tests/unit/api/test_api.py":
        (1, "prose inside an assertion message"),
    "tests/unit/ports/test_msl_port.py":
        (1, "prose inside a NotImplementedError message"),
    "validation/crossval/07_sheen_lpf.py":
        (1, "prose pointing a maintainer at a task recipe"),
    "validation/crossval/21_coax_two_port_referee.py":
        (2, "prose citing the known-issues ledger"),
    "validation/research/floquet/rcwa_referee.py":
        (1, "prose citing the known-issues ledger"),
    "scripts/capture_msl_replay_fixture.py":
        (1, "prose inside a retirement message"),
    "scripts/diagnostics/probe_fed_msl_openems_referee.py":
        (2, "prose stating that the named log is LOCAL-ONLY"),
    "tests/contracts/test_guide_citations_resolve.py":
        (4, "sample TEXT fed to that file's own predicate -- the two dead "
            "links it was written to catch, the prose that replaced them, "
            "and the reference-style [id]: and angle-bracket (<path>) forms "
            "of the same link, one per link syntax the gate must not miss; "
            "it resolves none of them and reads no file there (#752)"),
    "scripts/diagnostics/build_msl_broad_e5_envelope.py":
        (2, "scope sentence copied into the emitted JSON"),

    # WRITE targets. The script creates the directory and writes into it; it
    # never expects to find anything there, so a clone losing the path is fine.
    "scripts/archive/cpml_reflectivity_sweep.py":
        (1, "output path, written after os.makedirs"),
    "scripts/archive/issue31_patch_validation.py":
        (1, "figure output directory, written after os.makedirs"),
    "scripts/archive/issue48_uniform_patch_ffrp.py":
        (1, "figure output directory, written after os.makedirs"),
    "scripts/memory_reduction_planning_artifact.py":
        (1, "argparse default for --output, a write target"),
    "validation/research/subgrid/13_subgrid_material_validation.py":
        (1, "ARTIFACT_PATH write target, built segment by segment, never tracked"),

    # OPTIONAL reads of artifacts that were NEVER tracked, from before this
    # cleanup. Each already tolerates absence, so untracking changed nothing for
    # them; they are listed so the next reader does not re-derive that.
    "examples/inverse_design/multilayer_ar_coating.py":
        (1, "load_meep_reference returns None when the file is absent"),
    "scripts/diagnostics/build_msl_openems_sparameter_comparison.py":
        (1, "argparse default for an external reference that was never tracked"),
    "scripts/diagnostics/compare_msl_thru_openems_reference.py":
        (1, "argparse default for an external reference that was never tracked"),
}


def segment_literals(source: str) -> list[tuple[int, str]]:
    """Non-docstring string constants naming a local-only doc directory.

    Matching the BARE segment, not the joined ``docs/research_notes`` prefix, is
    the whole point: ``ROOT / "docs" / "research_notes" / name`` puts the
    segment in a string of its own, and a joined-prefix search never sees it.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings \
                and any(segment in node.value for segment in SEGMENTS):
            found.append((node.lineno, node.value))
    return sorted(found)


SELF = "tests/contracts/" + Path(__file__).name


def _swept_files(repo: Path) -> list[str]:
    """Every tracked Python file, minus this one.

    This file is excluded because it is the rule, not a subject of it: it names
    both directories a dozen times in its own messages and mutants, and pinning
    a count for that would break on every wording change while protecting
    nothing.
    """
    result = subprocess.run(["git", "ls-files", "--", "*.py"],
                            cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return [p for p in result.stdout.split() if p != SELF]


@pytest.mark.skipif(
    not git_available(REPO),
    reason="not a git checkout -- the tracked file list is unavailable",
)
def test_no_unreviewed_reference_to_a_local_only_doc_directory() -> None:
    counted = {}
    for rel in _swept_files(REPO):
        hits = segment_literals((REPO / rel).read_text(encoding="utf-8", errors="replace"))
        if hits:
            counted[rel] = hits

    unreviewed = sorted(set(counted) - set(REVIEWED))
    assert not unreviewed, (
        "committed Python names a local-only doc directory in a file nobody has "
        f"signed off: {unreviewed}. If it RESOLVES a path there, the file exists "
        "in this checkout and in no clone -- move it under tests/fixtures/ or "
        "beside its consumer instead. If it is prose or a write target, add the "
        "file to REVIEWED with a count and a one-line reason."
    )

    drifted = {rel: (len(hits), REVIEWED[rel][0])
               for rel, hits in counted.items() if len(hits) != REVIEWED[rel][0]}
    assert not drifted, (
        "a reviewed file's segment count changed (file: found vs allowed) "
        f"{drifted}. A new usage in an already-reviewed file is exactly what a "
        "name-only allow-list would hide. Check what was added; if it is safe, "
        "bump the count and extend that entry's reason."
    )

    stale = sorted(set(REVIEWED) - set(counted))
    assert not stale, (
        f"REVIEWED lists files that no longer name either directory: {stale}. "
        "Drop the entries -- a stale allow-list silently widens over time."
    )


def test_the_sweep_sees_both_reader_shapes_and_leaves_prose_alone() -> None:
    """Criterion (B): mutants red, control green.

    The first mutant is the shape that shipped broken on 2026-09-15 (a read
    split across two lines); the second is the segment-built path, which no
    joined-string search can see. The control carries the same words in a
    docstring and a comment, where they are documentation.
    """
    split_across_lines = (
        "from pathlib import Path\n"
        "def load(root):\n"
        "    path = (Path(root)\n"
        "            / 'docs/research_notes/issue726/collocation/raw.npz')\n"
        "    return path.read_bytes()\n"
    )
    built_from_segments = (
        "from pathlib import Path\n"
        "ARTIFACT = Path(__file__).parent / 'docs' / 'research_notes' / 'x.json'\n"
    )
    prose_only = (
        '"""Provenance: docs/research_notes/20260915_note.md.\n\n'
        'Background lives under docs/agent-memory/.\n"""\n'
        "# see docs/research_notes/ for the trail\n"
        "VALUE = 3\n"
    )

    assert [v for _, v in segment_literals(split_across_lines)] == [
        "docs/research_notes/issue726/collocation/raw.npz"]
    assert [v for _, v in segment_literals(built_from_segments)] == ["research_notes"]
    assert segment_literals(prose_only) == []
