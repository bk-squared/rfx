"""Every ``git_sha`` a multiband-NU results record carries must resolve.

A one-shot FDTD ladder cannot be re-measured, so the commit hash the
instrument stamped into its JSON is the only link from the stored numbers to
the solver that produced them. A length check (``len(sha) == 40``) passes on
a hash that points at nothing, which is how fifteen E5/E6 records came to cite
commits that a rebase had orphaned before PR #1034 was opened (review,
2026-09-22). This test resolves each hash in the repository instead.

A hash that is KNOWN not to resolve is listed in ``ORPHANED`` together with
the note that records the interval it falls in; the listing is tied to the
record both ways (an unlisted dangling hash fails, a listed hash that resolves
again fails, a listed hash whose note does not mention it fails). Outside a
git checkout (a ``git archive`` export, the GPU suite) the test skips: there
is nothing to resolve against, and so does a shallow (depth-1) CI clone; the
gate lives in the full clone a developer or ``scripts/ci/local.sh`` runs on.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "validation/research/multiband_nu/results"
NOTES = REPO / "docs/design_notes"

#: sha -> the note whose "Provenance addendum" bounds the run by an interval.
ORPHANED = {
    # E5 (lane 1): rebased onto the AD-Q tip before PR #1034; runs 2026-09-13.
    "d6bc5ddeaaf677f4451b6ce32a62a8fafd9118b2": "20260913_nu_lane1_multilevel_xyz_predeclaration.md",
    "ea2080242722771c012b267d169350d50347e173": "20260913_nu_lane1_multilevel_xyz_predeclaration.md",
    "554d4b0f48c4bca1cc2343b1d02f3f9ce24ec8c5": "20260913_nu_lane1_multilevel_xyz_predeclaration.md",
    "b46465286110415752396526507b3a087ec438a4": "20260913_nu_lane1_multilevel_xyz_predeclaration.md",
    # E6 (lane 2): same rebase; runs 2026-09-14.
    "847269a5628c607cf7a776d4f5ea30c3dda5a09d": "20260913_nu_lane2_inplane_designvar_ad_predeclaration.md",
    "bd1ea2626ca856400d4d88bd31fb54931c4b2669": "20260913_nu_lane2_inplane_designvar_ad_predeclaration.md",
    "60781cccd3568940a2b2d2e164e935b8a6152287": "20260913_nu_lane2_inplane_designvar_ad_predeclaration.md",
    "fa2eae4dae5b25130de5ff90414f70cbdde6b06d": "20260913_nu_lane2_inplane_designvar_ad_predeclaration.md",
    "17ece76115d2fd9f891802128e0ee5ba2868036b": "20260913_nu_lane2_inplane_designvar_ad_predeclaration.md",
    "dd24ec3aee6b759d8d7849e5fa38963f5f651873": "20260913_nu_lane2_inplane_designvar_ad_predeclaration.md",
    "4f2d22af40d9d61ae63180fee928049ca18f3df7": "20260913_nu_lane2_inplane_designvar_ad_predeclaration.md",
    "99b510b3a813ca01d7406db118a650c46e2c2f5b": "20260913_nu_lane2_inplane_designvar_ad_predeclaration.md",
}

_KEY_SUFFIXES = ("git_sha", "sha", "commit")  # git_sha, firstpass_git_sha, e1_git_sha, old_commit, ...
_MAX_DEPTH = 3


def collect_shas(obj, depth: int = 0) -> set:
    """Every hex string stored under a git-sha-like key, a few levels deep."""
    out: set = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.endswith(_KEY_SUFFIXES) and isinstance(v, str) and len(v) >= 7 and all(c in "0123456789abcdef" for c in v):
                out.add(v)
            elif isinstance(v, (dict, list)) and depth < _MAX_DEPTH:
                out |= collect_shas(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:50]:
            if isinstance(v, (dict, list)):
                out |= collect_shas(v, depth + 1)
    return out


def resolves(sha: str, repo: Path = REPO) -> bool:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode == 0


def audit(results_dir: Path, orphaned: dict, repo: Path = REPO) -> list:
    """Return the problems, one string each; empty means every hash is accounted for."""
    problems = []
    seen: set = set()
    for path in sorted(results_dir.glob("*.json")):
        for sha in sorted(collect_shas(json.loads(path.read_text()))):
            seen.add(sha)
            listed = sha in orphaned
            ok = resolves(sha, repo)
            if ok and listed:
                problems.append(f"{path.name}: {sha[:12]} resolves again but is listed as orphaned")
            elif not ok and not listed:
                problems.append(f"{path.name}: {sha[:12]} resolves to no commit and is not recorded as orphaned")
    for sha in orphaned:
        if sha not in seen:
            problems.append(f"ORPHANED lists {sha[:12]}, which no record carries")
    return problems


#: The commit that created ``results/`` (#785, 2026-08-31). Every hash a record
#: can legitimately cite is younger, so a clone whose history reaches this commit
#: can resolve them all; a clone that cannot (the pull-request CI checkout is a
#: depth-1 clone, and ``rev-parse --is-shallow-repository`` is also true of a
#: deep-but-shallow development clone) has nothing to audit against.
RESULTS_DIR_CREATED = "5c3e658c656bff4f7c900c7522d9bae476dff1ce"


def history_available(repo: Path = REPO) -> bool:
    """A checkout whose history reaches the commit that created ``results/``."""
    return (repo / ".git").exists() and resolves(RESULTS_DIR_CREATED, repo)


@pytest.fixture(scope="module")
def _git_checkout():
    if not history_available():
        pytest.skip(f"history does not reach {RESULTS_DIR_CREATED[:8]} (archive export or "
                    "shallow CI clone): nothing to resolve against; a development clone runs this")


def test_every_recorded_sha_resolves_or_is_recorded_as_orphaned(_git_checkout):
    assert audit(RESULTS, ORPHANED) == []


def addendum(note: Path) -> str:
    """The text from the note's provenance addendum heading to its end."""
    text = note.read_text()
    marker = "## Provenance addendum"
    assert marker in text, note.name
    return text[text.index(marker):]


@pytest.mark.docs_consistency
def test_every_orphaned_sha_is_named_in_its_note():
    for sha, note in ORPHANED.items():
        assert sha[:8] in addendum(NOTES / note), (
            f"{note} does not name {sha[:8]} in its provenance addendum (earlier sections do not count)")


def test_audit_fires_on_a_listed_sha_no_record_carries(tmp_path, _git_checkout):
    (tmp_path / "empty.json").write_text(json.dumps({"provenance": {}}))
    assert audit(tmp_path, {"0123456789abcdef0123456789abcdef01234567": "x.md"}) == [
        "ORPHANED lists 0123456789ab, which no record carries"]


def test_audit_fires_on_an_unlisted_dangling_sha(tmp_path, _git_checkout):
    (tmp_path / "fake.json").write_text(json.dumps(
        {"provenance": {"git_sha": "0123456789abcdef0123456789abcdef01234567"}}))
    assert audit(tmp_path, {}) == [
        "fake.json: 0123456789ab resolves to no commit and is not recorded as orphaned"]


def test_audit_fires_when_a_listed_sha_resolves(tmp_path, _git_checkout):
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "live.json").write_text(json.dumps({"git_sha": head}))
    assert audit(tmp_path, {head: "x.md"}) == [
        f"live.json: {head[:12]} resolves again but is listed as orphaned"]
    assert audit(tmp_path, {}) == []
