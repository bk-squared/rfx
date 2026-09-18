"""tests/_fixture_provenance.py

One shared answer to "which code produced this committed number, and can a
reviewer still get at it?" (issue #1013).

``rfx_commit`` alone cannot answer it. A PR branch's commits stop being
reachable from ``main`` the moment the PR is squash-merged, so every fixture
that records the branch sha becomes a dangling pointer on the day it lands.
#1005 is the worked example: ``6a369c27`` is the sha in
``tests/fixtures/rcs280_reference_subtraction/fixture.json`` and
``git merge-base --is-ancestor 6a369c27 origin/main`` exits 1.

The obvious repair -- also record the root tree, ``HEAD^{tree}``, on the theory
that a squash preserves it -- is FALSE and was measured false before this
module was written. A squash replays the branch diff onto main's *current*
tip, so any main commit that landed while the PR was open changes the squash's
tree. For #1005 four commits touching ``rfx/sparams/`` landed in that window:
``tree(6a369c27)=d71fce95`` and ``tree(8586f549)=2da7bcfe``, and d71fce95
matches 0 of the 1905 distinct root trees on ``origin/main``. Across the whole
fixture backlog the root tree matched a main commit for 0 of 8 resolvable
shas. (Every count in this module was measured at ``origin/main`` = 541f703f;
they drift as main grows, the conclusions do not.)

What this module records instead is ``rfx_code_tree`` -- the sha of the
``rfx/`` subtree, ``git rev-parse HEAD:rfx``. It survives whenever main's
``rfx/`` did not move while the PR was open, which is most of the time but not
always. Measured over the same backlog: 8 of 10 (6 of the 8 resolvable in a
fresh clone, plus both of the two that survive only in the primary checkout).
It does NOT rescue every sha, and two of the backlog's own entries are misses:

  * ``6a369c27`` (#1005's own fixture) -> ``rfx/`` subtree ``2a076e71``,
    on NO commit of origin/main.
  * ``98d31987`` (harminv stage1 receipt) -> ``d4702ee9``, likewise.

So this is a strictly better anchor than the root tree, not a closed hole.
Anything it does not rescue must be annotated with a written reason in the
contract's table, never admitted by widening the check.

Two further limits, both measured, both deliberately not papered over:

  * RESOLVING POWER. 787 distinct ``rfx/`` subtrees cover 1936 main commits:
    median 1 commit per subtree, but the worst single subtree covers 95.
    ``5b588f3c`` and ``f914a7ca`` are two different fixture-producing commits
    that map to the same subtree ``7d93cbe0``; the anchor cannot tell them
    apart. ``generator_blob`` is recorded alongside for that reason -- it
    breaks such ties -- but it is deliberately NOT a rescue arm in the gate: a
    producing script being on main says nothing about whether the physics code
    that produced the number is.
  * SCOPE. ``rfx_code_tree`` says "the rfx package content that produced this
    number is on main". It does not say the fixture is byte-reproducible: a
    fixture whose generator or test-side inputs changed in the same PR can
    match on ``rfx/`` and still not replay. A green provenance gate is an E0
    schema/reachability assertion and is never evidence for a physics claim.

FAIL-CLOSED. Every capture subprocess here checks its return code and raises
``ProvenanceUnavailable`` rather than writing ``""``. The generators this
replaces did not: ``subprocess.run(["git","rev-parse","HEAD"], ...).stdout
.strip()`` returns ``''`` with ``returncode=128`` outside a work tree, and
``scripts/vessl_gpu_suite.yaml`` runs from a ``git archive`` export in /tmp
that has no ``.git`` at all, so that silent empty string was reachable. An
explicit ``commit_override`` remains the one supported way to stamp a record
produced outside a checkout (a tarball fetch); it is recorded as such.

The reader half is used by ``tests/contracts/test_fixture_provenance_
reachability.py``. It deliberately classifies from the STORED
``rfx_code_tree`` value rather than from a live ``<sha>:rfx`` lookup, because
whether an object is present is a property of the CLONE, not of the
repository: 6 of the 11 bad shas only resolve here because someone fetched
them by hand, and 2 (``c3189960``, ``296cabad``) are refused by origin
outright ("upload-pack: not our ref"). A gate that asked git to resolve the
sha would pass on one machine and fail on the next.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Key the code-tree witness is stored under, beside the commit sha it rescues.
CODE_TREE_KEY = "rfx_code_tree"
#: Key naming the producing script's blob, for disambiguation only.
GENERATOR_BLOB_KEY = "generator_blob"
#: The subtree whose sha is the witness.
CODE_SUBTREE = "rfx"
#: Environment override for the reference ref the gate measures against.
REF_ENV = "RFX_PROVENANCE_REF"
#: Ordered fallbacks when the environment does not name one.
REF_CANDIDATES = ("origin/main", "main")


class ProvenanceUnavailable(RuntimeError):
    """Git could not answer a provenance question and no override was given.

    Raised instead of returning ``""`` or ``"unknown"``. A record stamped with
    an empty or invented sha is worse than a record that was never written:
    the gate cannot tell it from a real one.
    """


# ---------------------------------------------------------------------------
# writer half -- every call fails closed
# ---------------------------------------------------------------------------

def _git(args: list[str], repo: Path | None = None) -> str:
    """Run git and RAISE on any non-zero exit. Never returns an empty answer."""
    cwd = repo or REPO
    try:
        result = subprocess.run(["git", *args], cwd=str(cwd),
                                capture_output=True, text=True, timeout=60)
    except OSError as exc:  # git missing entirely
        raise ProvenanceUnavailable(f"git {' '.join(args)} in {cwd}: {exc}") from exc
    if result.returncode != 0:
        raise ProvenanceUnavailable(
            f"git {' '.join(args)} in {cwd} exited {result.returncode}: "
            f"{result.stderr.strip() or '<no stderr>'}")
    out = result.stdout.strip()
    if not out:
        raise ProvenanceUnavailable(
            f"git {' '.join(args)} in {cwd} exited 0 with empty output")
    return out


def commit(repo: Path | None = None, rev: str = "HEAD") -> str:
    """Full 40-hex sha of *rev*. Short shas are refused on purpose: the
    backlog's one unrecoverable entry is a 7-char ``6fd6ea0`` written by
    ``git rev-parse --short HEAD``, which no clone can resolve today."""
    sha = _git(["rev-parse", rev], repo)
    if len(sha) != 40:
        raise ProvenanceUnavailable(f"git rev-parse {rev} returned {sha!r}, not a 40-hex sha")
    return sha


def code_tree(repo: Path | None = None, rev: str = "HEAD") -> str:
    """Sha of ``<rev>:rfx`` -- the durable half of the provenance record."""
    return _git(["rev-parse", f"{rev}:{CODE_SUBTREE}"], repo)


def blob(rel_path: str | Path, repo: Path | None = None, rev: str = "HEAD") -> str:
    """Sha of the committed blob at *rel_path* in *rev*."""
    return _git(["rev-parse", f"{rev}:{rel_path}"], repo)


def worktree_dirty(repo: Path | None = None) -> bool:
    """True when tracked files differ from *rev* HEAD.

    A dirty worktree means ``rfx_code_tree`` describes the committed code, not
    the code that ran. The flag is recorded rather than raised on, so a
    deliberate scratch run is still stampable and still legible as one.
    """
    cwd = repo or REPO
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=str(cwd), capture_output=True, text=True, timeout=60)
    except OSError as exc:
        raise ProvenanceUnavailable(f"git status in {cwd}: {exc}") from exc
    if result.returncode != 0:
        raise ProvenanceUnavailable(
            f"git status in {cwd} exited {result.returncode}: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def capture(repo: Path | None = None, *, commit_key: str = "rfx_commit",
            commit_override: str | None = None,
            generator: str | Path | None = None) -> dict:
    """The provenance block every fixture writer should embed.

    ``commit_key`` lets a writer keep the key name its schema already uses --
    the diagnostics lanes store ``commit``, the fixture generators store
    ``rfx_commit``, and the gate reads both. The code-tree witness is always
    stored under :data:`CODE_TREE_KEY` beside it, because the gate looks for
    that exact sibling.

    ``commit_override`` is the supported escape hatch for a run whose tree is
    not a checkout (a tarball or ``git archive`` export). It stamps the sha the
    caller vouches for and records ``<commit_key>_source`` so a reader can see
    the sha was asserted rather than read. When it is given and the tree cannot
    resolve it, the code-tree witness is honestly absent rather than invented.

    With no override, every field is read from git and any failure raises.
    """
    cwd = Path(repo) if repo is not None else REPO
    if commit_override is not None:
        if len(commit_override) != 40 or not all(c in "0123456789abcdef" for c in commit_override):
            raise ProvenanceUnavailable(
                f"commit_override={commit_override!r} is not a 40-hex sha; a short or "
                "invented sha is not resolvable in any clone")
        block: dict = {commit_key: commit_override, f"{commit_key}_source": "override"}
        try:
            block[CODE_TREE_KEY] = code_tree(cwd, rev=commit_override)
        except ProvenanceUnavailable:
            # The override names a commit this tree cannot resolve. Say so.
            block[CODE_TREE_KEY] = None
            block[f"{CODE_TREE_KEY}_absent_reason"] = (
                f"commit_override {commit_override} is not resolvable in {cwd}")
        return block

    block = {
        commit_key: commit(cwd),
        f"{commit_key}_source": "git",
        CODE_TREE_KEY: code_tree(cwd),
        "rfx_worktree_dirty": worktree_dirty(cwd),
    }
    if generator is not None:
        block[GENERATOR_BLOB_KEY] = blob(generator, cwd)
    return block


# ---------------------------------------------------------------------------
# reader half -- used by the contract gate
# ---------------------------------------------------------------------------

def _git_try(args: list[str], repo: Path | None = None) -> tuple[int, str]:
    """Run git where a non-zero exit is a legitimate ANSWER, not a failure."""
    cwd = repo or REPO
    try:
        result = subprocess.run(["git", *args], cwd=str(cwd),
                                capture_output=True, text=True, timeout=120)
    except OSError:
        return 127, ""
    return result.returncode, result.stdout.strip()


def git_available(repo: Path | None = None) -> bool:
    rc, out = _git_try(["rev-parse", "--is-inside-work-tree"], repo)
    return rc == 0 and out == "true"


def is_shallow(repo: Path | None = None) -> bool:
    """True when the clone is truncated.

    A reachability assertion in a shallow clone cannot distinguish "this commit
    is not on main" from "this clone was never told about it". Every CI
    checkout in this repository was depth-1 until #1013 added
    ``fetch-depth: 0`` to the job that runs ``tests/contracts/``.
    """
    rc, out = _git_try(["rev-parse", "--is-shallow-repository"], repo)
    return rc == 0 and out == "true"


def resolve_reference_ref(repo: Path | None = None) -> str | None:
    """The ref reachability is measured against, or None if none resolves.

    Policy, stated once so both CI events behave predictably:

      * ``RFX_PROVENANCE_REF`` wins when set.
      * otherwise ``origin/main``, then ``main``.

    The gate ALSO admits a site whose sha is reachable from ``HEAD``. That arm
    is what keeps a pull request green while it is still in flight: a PR that
    changes ``rfx/`` and regenerates a fixture stamps a sha and a code tree
    that are on the branch and on no main commit yet, and on the
    ``pull_request`` event HEAD is the merge commit whose history contains
    both. On the ``push: branches: [main]`` event HEAD *is* main, so the arm
    collapses and adds nothing -- which is the point: this class of breakage is
    only detectable after the squash, so the gate is a post-merge detector and
    its true positive lands on main.
    """
    override = os.environ.get(REF_ENV)
    candidates = (override,) if override else REF_CANDIDATES
    for ref in candidates:
        rc, _ = _git_try(["rev-parse", "--verify", f"{ref}^{{commit}}"], repo)
        if rc == 0:
            return ref
    return None


def commit_exists(sha: str, repo: Path | None = None) -> bool:
    rc, out = _git_try(["cat-file", "-t", sha], repo)
    return rc == 0 and out == "commit"


def is_reachable(sha: str, ref: str, repo: Path | None = None) -> bool:
    if not commit_exists(sha, repo):
        return False
    rc, _ = _git_try(["merge-base", "--is-ancestor", sha, ref], repo)
    return rc == 0


_CODE_TREE_CACHE: dict[tuple[str, str], frozenset[str]] = {}


def code_trees_on(ref: str, repo: Path | None = None) -> frozenset[str]:
    """Every distinct ``<commit>:rfx`` sha over the history of *ref*.

    One ``rev-list`` plus one batched ``cat-file``; measured at ~0.03 s over
    1936 commits, and memoised per (repo, ref) so a shard pays it once.
    """
    cwd = str(repo or REPO)
    key = (cwd, ref)
    if key in _CODE_TREE_CACHE:
        return _CODE_TREE_CACHE[key]
    rc, out = _git_try(["rev-list", ref], repo)
    if rc != 0:
        _CODE_TREE_CACHE[key] = frozenset()
        return _CODE_TREE_CACHE[key]
    commits = out.split()
    if not commits:
        _CODE_TREE_CACHE[key] = frozenset()
        return _CODE_TREE_CACHE[key]
    stdin = "".join(f"{c}:{CODE_SUBTREE}\n" for c in commits)
    try:
        result = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
            cwd=cwd, input=stdin, capture_output=True, text=True, timeout=300)
    except OSError:
        _CODE_TREE_CACHE[key] = frozenset()
        return _CODE_TREE_CACHE[key]
    trees = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "tree":
            trees.add(parts[0])
    _CODE_TREE_CACHE[key] = frozenset(trees)
    return _CODE_TREE_CACHE[key]


def code_tree_of(sha: str, repo: Path | None = None) -> str | None:
    """``<sha>:rfx``, or None when this CLONE cannot resolve *sha*.

    Used by the one-off annotation script, never by the gate: see the module
    docstring on clone state versus repo state.
    """
    rc, out = _git_try(["rev-parse", f"{sha}:{CODE_SUBTREE}"], repo)
    return out if rc == 0 and len(out) == 40 else None


def landing_commit(rel_path: str, ref: str, repo: Path | None = None) -> str | None:
    """The commit on *ref* that last touched *rel_path*.

    Recorded in failure messages as the only anchor that always resolves. It
    is NOT a pass condition: every tracked file has a landing commit by
    construction, so an assertion on it can never fail and carries no evidence.
    It is an anchor only in combination with a code tree.
    """
    rc, out = _git_try(["log", "-1", "--format=%H", ref, "--", rel_path], repo)
    return out or None if rc == 0 else None


#: The rfx/ subtree of every commit a committed fixture records, with the clone
#: the tree was read from. This exists because the binding it feeds CANNOT be
#: derived in CI: a fresh clone of main does not carry a branch-only commit, so
#: ``code_tree_of(sha)`` returns None for exactly the shas that matter and any
#: check conditioned on it silently does nothing there. Recording the mapping is
#: what makes "this witness belongs to THIS commit" answerable in a clone that
#: never had the object.
#:
#: Two entries are marked NOT on any main commit. They are recorded anyway: a
#: tree that is provably absent is more legible than a missing key, and
#: recording it does not make the site pass.
#:
#: ``test_recorded_code_trees_match_git`` re-derives every entry that resolves
#: locally, so the table cannot drift from git.
RECORDED_CODE_TREES: dict[str, tuple[str, str]] = {

    "088281899727fcc644814f0ae9451b6b89a26af8":
        ("f72547caacbd8b18690eebd5e8c6266a5e2a61c3", "this clone"),
    "8206031de9923dfaddc17f7060def466a7906915":
        ("3d2176e84bfee18b699d0976726bce27ff507bbe", "this clone"),
    "6216e06fff27d6b351457cb4807379bec14ced15":
        ("9ec8308106a26deb04787fce2f919b466d9fb22f", "this clone"),
    "ca168584e35b145fcbeb86dc11350a7304daf024":
        ("0cd2ab4aa36cb29ed763cca34668619ce0c0a5c6", "this clone"),
    "5b588f3c6cdc2aeeb9af1eb528889cec3813525d":
        ("7d93cbe022da003de9ed375f809cdecadf72ce73", "this clone"),
    "f914a7caf1ff8c63cac6f5f8c975b7f9f420a0c7":
        ("7d93cbe022da003de9ed375f809cdecadf72ce73", "this clone"),
    "c3189960504b5fa570e944c074801594e6caf9cc":
        ("192bbdab1f23bcaa8e3ee4d33c066001c1d82200",
         "read-only primary checkout; origin refuses the object"),
    "296cabada23343eede7beb05a0a4f98f7bb64adf":
        ("f74a17ce0386afcf88c4f23b7b540a0e209a5157",
         "read-only primary checkout; origin refuses the object"),
    "6a369c2730c4904f71187f3cd4bddff943e91f22":
        ("2a076e7146a21e0b07794fc82f179126e0c605d5",
         "refs/pull/1005/head; NOT on any main commit"),
    "98d319877dc50cec43bca950b163ebec029cf13e":
        ("d4702ee977fbb34aebcde063dfac591ea9c2238b",
         "this clone; NOT on any main commit"),
}
