#!/usr/bin/env python3
"""Phase-C gate for the gallery v2 precompute run.

Asserts every committed manifest is schema ``rfx-gallery-manifest-v2``, has
``validation.passed`` in ``{True, null}`` (null only for ``--quick`` smokes,
which must never be committed — see INV-4), and carries a
``gradient_validation`` block on the three S-parameter cases. Exits non-zero on
any issue so the VESSL job (and any local pre-commit check) fails visibly.

Kept as a committed script rather than an inline heredoc because the VESSL
runner shell is busybox-ash, which rejects the ``python - <<'PY'`` heredoc form.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

SCHEMA = "rfx-gallery-manifest-v2"
NON_GRADIENT_CASES: set[str] = set()  # cases exempt from the AD panel (none currently)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-root", default="docs/public/gallery/assets")
    args = ap.parse_args()

    mfs = sorted(glob.glob(os.path.join(args.assets_root, "*", "manifest.json")))
    if not mfs:
        print(f"no manifests under {args.assets_root}")
        return 1

    bad: list[str] = []
    for mf in mfs:
        m = json.load(open(mf))
        cid = m.get("case_id")
        v = m.get("validation", {})
        sv = m.get("schema_version")
        passed = v.get("passed")
        grad = "gradient_validation" in m
        print(
            f"{cid}: schema={sv} passed={passed} gradient_validation={grad} "
            f"application={m.get('application')} capability={m.get('capability')}"
        )
        if sv != SCHEMA:
            bad.append(f"{cid}: schema {sv!r} != {SCHEMA}")
        if passed not in (True, None):
            bad.append(f"{cid}: validation.passed={passed!r} (expected True or null)")
        if cid not in NON_GRADIENT_CASES and not grad:
            bad.append(f"{cid}: missing gradient_validation")

    if bad:
        print("VALIDATION ISSUES:", *bad, sep="\n  ")
        return 1
    print(
        "manifests OK (all v2; validation.passed in {True, null}; "
        "gradient_validation on the 3 S-param cases)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
