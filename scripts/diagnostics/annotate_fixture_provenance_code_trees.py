#!/usr/bin/env python3
"""Add the rfx/ code-tree witness to fixture provenance sites, once (#1013).

The 29 provenance sites that existed when
``tests/contracts/test_fixture_provenance_reachability.py`` landed were written
before any producer emitted ``rfx_code_tree``. Their producers now do; this
script is the one-off that back-fills the records already committed, so the
derivation is a script a reviewer can re-run rather than a hand edit nobody can
check.

WHY A TEXT INSERTION AND NOT A JSON ROUND-TRIP. These fixtures hold measured
physics. ``json.dumps(json.load(...))`` re-renders every float and every bit of
whitespace, so a diff would be unreadable and a real numeric change could hide
inside it. This inserts one line after the line that carries the sha, at the
same indentation, and touches nothing else -- then re-parses and PROVES that
every pre-existing leaf compares float.hex-identical and that the only new keys
are the ones this script claims to add. It refuses to write otherwise.

WHERE THE TREES COME FROM. Eight of the ten resolvable shas are in this clone
or can be fetched from origin. Two -- ``c3189960`` and ``296cabad`` -- are
refused by origin outright ("upload-pack: not our ref") and survive only in the
read-only primary checkout on NFS. Their trees are recorded in :data:`TREES`
with the donor named, because a clone that never had those objects still has to
be able to check the annotation against main, and the check that matters is
"is this tree on main", which any full clone can answer.

TWO SHAS ARE NOT RESCUED. ``6a369c27`` (#1005's own fixture, the one the issue
is about) and ``98d31987`` have rfx/ subtrees on NO main commit: main's rfx/
moved while their PRs were open, so the squash tree differs on the rfx/ subtree
too. Their trees are recorded anyway -- a recorded tree that is provably absent
is more legible than a missing key -- and both sites are ANNOTATED in the gate
with the measured reason. Recording them does not make them pass.

Re-runnable: a site that already carries the key is left alone.

    PYTHONPATH=. python scripts/diagnostics/annotate_fixture_provenance_code_trees.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests._fixture_provenance import (
    RECORDED_CODE_TREES,  # noqa: E402
    CODE_TREE_KEY, code_tree_of, is_reachable, resolve_reference_ref)
from tests.contracts.test_fixture_provenance_reachability import SHA_KEYS, _sites  # noqa: E402

#: commit sha -> (rfx/ subtree sha, where it was resolved)

#: Shas with no tree at all, and why. These get no key: an invented or
#: guessed value would be worse than the honest absence.
NO_TREE: dict[str, str] = {
    "6fd6ea0": ("7-char short sha from `git rev-parse --short HEAD`; resolves in no "
                "clone available here, so no tree can be derived"),
}


def _flat(node, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            _flat(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _flat(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = out.get(prefix, node)
    return out


def _hexed(mapping: dict) -> dict:
    return {k: (float.hex(v) if isinstance(v, float) else v) for k, v in mapping.items()}


def annotate(rel: str, sites: list[tuple[str, str, str, str | None]]) -> tuple[int, list[str]]:
    """Insert the witness beside every sha line in *rel*. Returns (n, notes)."""
    path = REPO / rel
    before_text = path.read_text()
    before = _hexed(_flat(json.loads(before_text)))
    wanted = [(kp, sha) for _, kp, sha, witness in sites if witness is None]
    notes: list[str] = []
    text = before_text
    inserted = 0
    for sha in sorted({s for _, s in wanted}):
        if sha in NO_TREE:
            notes.append(f"    {sha}: SKIPPED -- {NO_TREE[sha]}")
            continue
        tree, where = RECORDED_CODE_TREES[sha]
        live = code_tree_of(sha, REPO)
        if live and live != tree:
            raise SystemExit(f"{rel}: {sha[:8]}:rfx is {live} here but the table says {tree}")
        n_expected = sum(1 for _, s in wanted if s == sha)
        # Only the key names the gate actually enumerates. This file has a
        # third line carrying the same sha under `post_fix_commit`, which is
        # not a gate site; matching "any key" would have annotated it silently.
        keys = "|".join(sorted(SHA_KEYS))
        pattern = re.compile(
            r'^(?P<indent>[ \t]*)"(?P<key>' + keys + r')": "' + sha + r'",?[ \t]*$',
            re.MULTILINE)
        hits = list(pattern.finditer(text))
        if len(hits) != n_expected:
            raise SystemExit(f"{rel}: {len(hits)} text line(s) carry {sha[:8]} but the parsed "
                             f"JSON has {n_expected} site(s); refusing to guess")

        def _insert(m: re.Match) -> str:
            indent = m.group("indent")
            line = f'{indent}"{CODE_TREE_KEY}": "{tree}",'
            return f'{indent}"{m.group("key")}": "{sha}",\n{line}'

        text = pattern.sub(_insert, text)
        inserted += n_expected
        notes.append(f"    {sha[:8]} -> {tree[:8]}  x{n_expected}  ({where})")
    if not inserted:
        return 0, notes
    after_obj = json.loads(text)          # refuses to write unparseable JSON
    after = _hexed(_flat(after_obj))
    moved = {k: (before[k], after[k]) for k in before if k in after and before[k] != after[k]}
    lost = sorted(set(before) - set(after))
    gained = sorted(set(after) - set(before))
    if moved or lost:
        raise SystemExit(f"{rel}: refusing to write -- moved={moved} lost={lost}")
    if any(not g.endswith(CODE_TREE_KEY) for g in gained):
        raise SystemExit(f"{rel}: refusing to write -- unexpected new keys {gained}")
    path.write_text(text)
    notes.append(f"    {len(before)} pre-existing leaves float.hex-identical; "
                 f"{len(gained)} new key(s), all {CODE_TREE_KEY}")
    return inserted, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    ref = resolve_reference_ref(REPO)
    if ref is None:
        raise SystemExit("no reference ref resolves; cannot tell which sites need a witness")

    # Only the sites that are actually broken. A sha still reachable from main
    # needs no witness, and stamping one into another lane's fixture to no
    # purpose is churn, not repair.
    by_file: dict[str, list] = {}
    for site in _sites():
        rel, keypath, sha, witness = site
        if witness is None and (is_reachable(sha, ref, REPO) or is_reachable(sha, "HEAD", REPO)):
            continue
        by_file.setdefault(rel, []).append(site)

    total = 0
    for rel in sorted(by_file):
        todo = [s for s in by_file[rel] if s[3] is None]
        if not todo:
            continue
        if args.check:
            print(f"{rel}: {len(todo)} site(s) without {CODE_TREE_KEY}")
            continue
        n, notes = annotate(rel, by_file[rel])
        if n or notes:
            print(f"{rel}: +{n}")
            for note in notes:
                print(note)
        total += n
    print(f"\ninserted {total} {CODE_TREE_KEY} value(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
