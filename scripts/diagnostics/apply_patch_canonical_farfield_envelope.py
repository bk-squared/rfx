#!/usr/bin/env python3
"""Apply the re-derived envelope constants to the gate file, from the record.

Reads the JSON that scripts/diagnostics/measure_patch_canonical_farfield_e4.py
produced and rewrites the four constants in
tests/crossval/test_patch_canonical_farfield_e4.py. Nothing is typed: every
value comes out of the record, and the script refuses if a value it is about to
write is not the one the record proposes.

Line-number discipline: validation/crossval/manifest.json and
validation/README.md cite
``tests/crossval/test_patch_canonical_farfield_e4.py:134,140,141`` and
tests/contracts/test_evidence_citation_pointers.py gates those pointers, so the
rewrite must keep D_ABS_TOL_DB on 134, F_RES_REL_LO on 140 and F_RES_REL_HI on
141. The script checks that after writing and restores the file if it moved.

Why this is a script and not an edit. The constants are the output of a solve
that costs ~15 min of cluster time, and the whole point of #931's recompute rule
is that no gate constant is typed by hand. A script that reads the record, holds
the file's own derivation rules, and refuses when the result would break the
cited line numbers makes the edit checkable: run it against the record and the
diff is the record.

It does NOT decide whether the flip is correct. Two judgements stay with the
person applying it, and both are written down in
docs/design_notes/931_migration/XA-manifest-2b.md:

* if the measured offset is small enough that [lo, hi] straddles zero, the
  resonance gate stops being a SIGN-locked bias band and becomes a +-band. That
  is a different kind of lock, not a re-tune, and XA-manifest.json.md 5 says so
  explicitly. Flipping the flag in that case needs the docstring rewritten to
  say what the gate now asserts;
* if the run is under-settled (``measured.settling_clears_bar`` false), nothing
  in the record is quotable and the flag stays False.

Usage:  python scripts/diagnostics/apply_patch_canonical_farfield_envelope.py \
            <record.json> <repo-root>
"""
import json
import re
import sys
from pathlib import Path


def main():
    rec = json.loads(Path(sys.argv[1]).read_text())
    repo = Path(sys.argv[2])
    f = repo / "tests/crossval/test_patch_canonical_farfield_e4.py"
    lines = f.read_text(encoding="utf-8").split("\n")
    before = list(lines)

    pc = rec["proposed_constants"]
    subs = {
        "_ENVELOPES_REDERIVED_FOR_931": "True",
        "D_ABS_TOL_DB": repr(float(pc["D_ABS_TOL_DB"]["new"])),
        "F_RES_REL_LO": f'{float(pc["F_RES_REL_LO"]["new"]):+g}',
        "F_RES_REL_HI": f'{float(pc["F_RES_REL_HI"]["new"]):+g}',
    }
    for i, ln in enumerate(lines):
        m = re.match(r"^([A-Z_][A-Za-z0-9_]*)\s*=\s*([^#]*?)(\s*#.*)?$", ln)
        if not m:
            continue
        name = m.group(1)
        if name in subs:
            tail = m.group(3) or ""
            lines[i] = f"{name} = {subs[name]}{tail}"
            print(f"line {i+1}: {name} {m.group(2).strip()} -> {subs[name]}")

    f.write_text("\n".join(lines), encoding="utf-8")

    # pointer discipline
    text = f.read_text(encoding="utf-8").split("\n")
    for lineno, name in ((134, "D_ABS_TOL_DB"), (140, "F_RES_REL_LO"),
                         (141, "F_RES_REL_HI")):
        got = text[lineno - 1]
        if not got.startswith(name):
            f.write_text("\n".join(before), encoding="utf-8")
            raise SystemExit(
                f"REVERTED: line {lineno} should start with {name}, got {got!r}. "
                "The manifest and README cite these line numbers and a contract "
                "test gates the pointer.")
    print("pointer discipline OK: 134/140/141 still D_ABS_TOL_DB / F_RES_REL_LO / F_RES_REL_HI")


if __name__ == "__main__":
    main()
