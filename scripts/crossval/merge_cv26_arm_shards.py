#!/usr/bin/env python3
"""Assemble cv26's ``rfx.json`` from per-arm shard artifacts.

Why this exists: cv26's baseline is ten arms, and running them in ONE process is a
single serial VESSL job whose wall clock is the sum of the arms (round 1: 21 h). The
lane runs one job per arm or arm group instead, so the wall clock is the slowest arm.
Each shard writes ``rfx__shard_<name>.json`` with ``--tag shard_<name>``; this merges
them into the single ``rfx.json`` the gate-replay test and the manifest cite.

The merge is a concatenation with checks, never a recomputation: every arm block is
copied verbatim from the shard that produced it. What is checked is that the shards
describe ONE run of ONE code state -- same schema, same commit, same rig, no falsifier,
no smoke, no depth or dx override -- and that their arm sets are disjoint and complete.
The merged verdict is derived from the arms' own recorded verdicts with the same rule
the case uses, and every shard's file name, tag, exit code and run id is kept in
``shards`` so the merged document says where each arm came from.

    python scripts/crossval/merge_cv26_arm_shards.py --in <dir> --out <dir> [--run-ids <json>]

Exits 0 when ``rfx.json`` was written, 1 when the shards do not form one complete run.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _load("cv26_merge_oblique_fresnel", "validation/crossval/comparators/oblique_fresnel.py")

# the keys that must agree across shards: a difference in any of them means the shards
# are not one run, and the merged document would be a composite of two rigs
_MUST_AGREE = ("schema", "case_id", "commit", "arm_dx_div")


def merge(shard_paths: list, run_ids: dict | None = None) -> dict:
    if not shard_paths:
        raise SystemExit("no shard artifacts given")
    run_ids = run_ids or {}
    docs = []
    for p in sorted(shard_paths):
        with open(p) as fh:
            docs.append((os.path.basename(p), json.load(fh)))

    ref_name, ref = docs[0]
    problems = []
    arms: dict = {}
    owner: dict = {}
    shards = []
    for name, d in docs:
        if d.get("schema") != O.SCHEMA:
            problems.append(f"{name}: schema {d.get('schema')!r}, expected {O.SCHEMA!r}")
        if d.get("falsifier") is not None:
            problems.append(f"{name}: carries falsifier {d['falsifier']!r}; a baseline shard must not")
        if d.get("smoke"):
            problems.append(f"{name}: smoke artifact is never evidence")
        if d.get("n_cpml_override") is not None or d.get("dx_div_override") is not None:
            problems.append(f"{name}: diagnostic overrides (n_cpml={d.get('n_cpml_override')}, "
                            f"dx_div={d.get('dx_div_override')}); a baseline shard runs the declared recipe")
        if not str(d.get("commit", "")).strip() or d.get("commit") == "unknown":
            problems.append(f"{name}: no commit recorded")
        for k in _MUST_AGREE:
            if d.get(k) != ref.get(k):
                problems.append(f"{name}: {k} = {d.get(k)!r} but {ref_name} has {ref.get(k)!r}")
        if d.get("rig") != ref.get("rig"):
            differing = sorted(k for k in set(d.get("rig", {})) | set(ref.get("rig", {}))
                               if d.get("rig", {}).get(k) != ref.get("rig", {}).get(k))
            problems.append(f"{name}: rig differs from {ref_name} in {differing}")
        for arm, block in d.get("arms", {}).items():
            if arm in arms:
                problems.append(f"{arm} appears in both {owner[arm]} and {name}")
                continue
            arms[arm] = block
            owner[arm] = name
        shards.append({"file": name, "tag": d.get("tag"), "arms": sorted(d.get("arms", {})),
                       "exit_code": (d.get("verdict") or {}).get("exit_code"),
                       "date_utc": d.get("date_utc"), "run_id": run_ids.get(name)})

    expected = set(O.ARM_ORDER + O.GRAZE_ARMS)
    missing = sorted(expected - set(arms))
    extra = sorted(set(arms) - expected)
    if missing:
        problems.append(f"arms missing from the shard set: {missing}")
    if extra:
        problems.append(f"arms not in the declared set: {extra}")
    for arm, block in arms.items():
        if "arm_ok" not in block:
            problems.append(f"{arm} ({owner[arm]}): no per-arm verdict (arm_ok) in the artifact")
    if problems:
        for p in problems:
            print(f"  MERGE REFUSED: {p}")
        raise SystemExit(1)

    # the case's own rule, applied to the arms' recorded verdicts
    any_fail = any(not block["arm_ok"] for block in arms.values())
    any_meep_missing = False
    for arm in O.MEEP_ARMS:
        m = arms[arm].get("meep") or {}
        if m.get("present"):
            any_fail = any_fail or not m.get("e4_ok")
        else:
            any_meep_missing = True
    if any_fail:
        rc, summary = 1, "rfx accuracy: FAIL -- a gate failed on at least one arm (exit 1)"
    elif any_meep_missing:
        rc, summary = 2, ("[SKIP] the Meep reference is UNAVAILABLE (absent, or written and rejected by the leg's "
                          "own acceptance) for at least one Meep arm -- inconclusive, NOT a disagreement (exit 2)")
    else:
        rc, summary = 0, ("ALL CHECKS PASSED -- E2 (Fresnel at the realized angle), lattice-gated grazing arms "
                          "and E4 (Meep) (exit 0)")

    merged = {k: ref.get(k) for k in ("schema", "case_id", "commit", "rig", "arm_dx_div")}
    # The sha every shard agreed on, named as what it is: the staged code state the
    # whole lane ran at. ``commit`` already carries it, but a reader of the merged
    # document has no way to tell that it was CHECKED across the shards rather than
    # copied from the first one -- _MUST_AGREE is what makes it a lane property.
    merged["staged_commit"] = ref.get("commit")
    # ...and the evidence that it WAS checked rather than copied: the shard files whose
    # commit was compared against it. Without this the key is indistinguishable from a
    # copy of the first shard's commit.
    merged["staged_commit_checked_across"] = sorted(name for name, _ in docs)
    merged.update({
        "date_utc": max(d.get("date_utc", "") for _, d in docs),
        "falsifier": None, "smoke": False, "tag": None,
        "n_cpml_override": None, "dx_div_override": None,
        "merged_from_shards": True,
        "shards": sorted(shards, key=lambda s: s["file"]),
        "arms": {arm: arms[arm] for arm in O.ARM_ORDER + O.GRAZE_ARMS},
        "verdict": {"rfx_self_ok": not any_fail, "meep_present": not any_meep_missing,
                    "exit_code": rc, "summary": summary},
    })
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True, help="directory holding rfx__shard_*.json")
    ap.add_argument("--out", dest="out_dir", default=None, help="where to write rfx.json (default: --in)")
    ap.add_argument("--run-ids", default=None,
                    help="JSON file mapping shard file name -> VESSL run id (provenance, optional)")
    a = ap.parse_args(argv)
    out_dir = a.out_dir or a.in_dir
    paths = sorted(glob.glob(os.path.join(a.in_dir, "rfx__shard_*.json")))
    print(f"merging {len(paths)} shard artifacts from {a.in_dir}")
    for p in paths:
        print(f"  {os.path.basename(p)}")
    run_ids = json.load(open(a.run_ids)) if a.run_ids else None
    merged = merge(paths, run_ids)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rfx.json")
    with open(out_path, "w") as fh:
        json.dump(merged, fh, indent=1)
    print(f"\n  {out_path}: {len(merged['arms'])} arms, commit {merged['commit'][:12]}, "
          f"exit_code {merged['verdict']['exit_code']}")
    print(f"  {merged['verdict']['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
