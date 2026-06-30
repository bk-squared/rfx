#!/usr/bin/env python3
"""Reconcile committed gallery manifests against the files actually on disk.

After any pipeline run (precompute + v3 fig scripts + gradient emission), the
manifest.json in each case directory may contain stale sha256/size values,
entries for files that no longer exist, or ``type`` strings that disagree with
the authoritative vocabulary. This script is the FINAL step before committing
the bundle: it rebuilds assets[] to exactly match the files on disk and applies
the canonical filename → type vocabulary authoritatively.

Vocabulary design choices
--------------------------
* ``field_anim.gif``   — the PUBLIC-PAGE-REFERENCED field animation filename
  for the 3 S-param cases (fresnel / patch / waveguide). The precompute builder
  writes ``fields.gif``; if only ``fields.gif`` is present for an S-param case
  and ``field_anim.gif`` is absent, it is renamed automatically so the page URL
  is satisfied.
* ``fields.gif``       — the time-domain field animation for the AR case. For
  ar_coating_design, BOTH ``fields.gif`` AND ``design_field_coevolution.gif``
  are canonical; they keep their own distinct types.
* All non-canonical files are excluded from the manifest (but left on disk).

Usage
-----
::

    # reconcile ALL committed cases (default assets root)
    python scripts/reconcile_gallery_manifests.py

    # reconcile a specific assets root (e.g. a /tmp smoke bundle)
    python scripts/reconcile_gallery_manifests.py --assets-root /tmp/g

    # reconcile a single case within the default or given root
    python scripts/reconcile_gallery_manifests.py --case multilayer_fresnel

Importable
----------
::

    from scripts.reconcile_gallery_manifests import reconcile_case, reconcile_all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Canonical filename → type vocabulary
#
# Files whose filenames appear in CANONICAL_BY_CASE[case_id] are included.
# Files whose filenames appear in CANONICAL_COMMON are included for ALL cases.
# Everything else on disk is excluded (but left on disk untouched).
#
# The keys are exact filenames (case-sensitive). Values are the authoritative
# ``type`` string that overrides whatever the emitting script wrote.
# ---------------------------------------------------------------------------

# Files common to the 3 S-param cases (fresnel / waveguide / patch).
_SPARAM_COMMON: dict[str, str] = {
    "sparams.json": "sparams-json",
    "sparams.s2p": "touchstone",
    "sparams.s1p": "touchstone",
    "geometry.png": "geometry-png",
    # field_anim.gif is the page-referenced field animation for S-param cases.
    # precompute also emits a raw fields.gif, but it is NOT registered here — it
    # is redundant with the curated field_anim.gif and only bloats the bundle
    # (patch's was 8 MB). The rename fallback below still promotes a lone
    # fields.gif -> field_anim.gif when field_anim.gif is absent.
    "field_anim.gif": "field-animation-gif",
    "autodiff.png": "autodiff-png",
    "validation.png": "validation-png",
    "gradient.json": "gradient-json",
}

# Fresnel-specific extras.
_FRESNEL_EXTRA: dict[str, str] = {
    "rt_overlay.png": "rt-overlay-png",
    "field_xt.png": "field-xt-png",
    # sparams.png/smith.png are consistent provenance plots for slab/waveguide
    # (single run). They are NOT in _SPARAM_COMMON because the patch case runs a
    # separate precompute sim and would carry STALE ones — excluded there.
    "sparams.png": "sparam-plot-png",
    "smith.png": "smith-png",
}

# Patch-specific extras.
_PATCH_EXTRA: dict[str, str] = {
    "s11_db.png": "s11-db-png",
    "field_resonance.png": "field-resonance-png",
}

# Waveguide-specific extras.
_WAVEGUIDE_EXTRA: dict[str, str] = {
    "field_te10.png": "field-te10-png",
    "sparams.png": "sparam-plot-png",
    "smith.png": "smith-png",
}

# AR optimization case.
_AR_CANON: dict[str, str] = {
    "geometry.png": "geometry-png",
    "convergence.png": "convergence-png",
    "design_evolution.png": "design-evolution-png",
    "result_spectrum.png": "result-spectrum-png",
    "optimization.json": "optimization-json",
    # The design+field co-evolution GIF is the AR field animation. precompute
    # also emits a raw time-domain fields.gif, but the AR domain is 1-D (single
    # cell thick transversely) so it renders as a ~1px unreadable strip — it is
    # NOT registered; the field story is the co-evolution below.
    "design_field_coevolution.gif": "design-field-coevolution-gif",
    "gradient.json": "gradient-json",
}

# Assemble per-case canonical maps.
CANONICAL: dict[str, dict[str, str]] = {
    "multilayer_fresnel": {**_SPARAM_COMMON, **_FRESNEL_EXTRA},
    "patch_antenna": {**_SPARAM_COMMON, **_PATCH_EXTRA},
    "waveguide_wr90": {**_SPARAM_COMMON, **_WAVEGUIDE_EXTRA},
    "ar_coating_design": _AR_CANON,
}

# For unknown case ids fall back to accepting any file on disk with a generic
# type derived from its extension.
_EXTENSION_FALLBACK: dict[str, str] = {
    ".png": "figure-png",
    ".gif": "field-animation-gif",
    ".json": "data-json",
    ".s1p": "touchstone",
    ".s2p": "touchstone",
    ".s4p": "touchstone",
}

SERVED_URL_PREFIX = "/rfx/gallery/assets"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically (temp file + os.replace) to avoid partial writes."""
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    dir_ = path.parent
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".manifest_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def reconcile_case(
    case_dir: Path,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Reconcile assets[] in ``<case_dir>/manifest.json`` against disk.

    Steps
    -----
    1. Determine the canonical filename→type map for this case.
    2. For S-param cases: if ``field_anim.gif`` is absent but ``fields.gif``
       is present, rename ``fields.gif`` → ``field_anim.gif`` so the
       page-referenced URL is satisfied.
    3. Walk the canonical set: include only files that exist on disk; skip
       absent files (log a warning).
    4. Recompute sha256 + size for every included file.
    5. Preserve all non-assets blocks (``validation``, ``provenance``,
       ``gradient_validation``, ``application``, ``capability``,
       ``schema_version``, etc.).
    6. Write atomically.

    Returns the updated manifest dict.
    """
    man_path = case_dir / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {man_path}")

    with open(man_path) as fh:
        man = json.load(fh)

    cid = case_id or man.get("case_id") or case_dir.name
    canon = CANONICAL.get(cid)

    # ---- Step 2: field_anim.gif rename for S-param cases -------------------
    if cid in ("multilayer_fresnel", "patch_antenna", "waveguide_wr90"):
        fa = case_dir / "field_anim.gif"
        ff = case_dir / "fields.gif"
        if not fa.exists() and ff.exists():
            ff.rename(fa)
            print(f"  [{cid}] renamed fields.gif -> field_anim.gif")

    # ---- Step 3-4: build asset list from disk --------------------------------
    new_assets: list[dict[str, Any]] = []

    if canon is not None:
        # Canonical case: walk the canonical map in definition order.
        # For S-param cases, if BOTH fields.gif and field_anim.gif exist in the
        # canon map and on disk, we only include field_anim.gif (the page
        # reference). After the rename in step 2 only field_anim.gif will exist,
        # so this naturally falls out. But be defensive in case both somehow
        # survive.
        seen_filenames: set[str] = set()
        for filename, ftype in canon.items():
            if filename in seen_filenames:
                continue
            path = case_dir / filename
            if not path.exists():
                # For S-param cases fields.gif is expected to be renamed by
                # this point; don't warn for it separately.
                if not (cid in ("multilayer_fresnel", "patch_antenna", "waveguide_wr90")
                        and filename == "fields.gif"):
                    print(f"  [{cid}] skip {filename!r} (not on disk)")
                continue
            # Dedup: for S-param cases both "fields.gif" and "field_anim.gif"
            # appear in the canon map mapping to the same type; skip the
            # duplicate once we've added the preferred one.
            if any(a["filename"] == filename for a in new_assets):
                continue
            new_assets.append({
                "filename": filename,
                "type": ftype,
                "served_url": f"{SERVED_URL_PREFIX}/{cid}/{filename}",
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            })
            seen_filenames.add(filename)
    else:
        # Fallback for unknown case ids: include all files present.
        for p in sorted(case_dir.iterdir()):
            if p.name == "manifest.json" or p.name.startswith("."):
                continue
            ftype = _EXTENSION_FALLBACK.get(p.suffix, "data-file")
            new_assets.append({
                "filename": p.name,
                "type": ftype,
                "served_url": f"{SERVED_URL_PREFIX}/{cid}/{p.name}",
                "sha256": _sha256(p),
                "size_bytes": p.stat().st_size,
            })

    # ---- Step 5: preserve all non-assets blocks and update assets ----------
    man["assets"] = new_assets

    # ---- Step 6: atomic write -----------------------------------------------
    _atomic_write_json(man_path, man)
    print(f"  [{cid}] reconciled {len(new_assets)} assets -> {man_path}")
    return man


def reconcile_all(assets_root: Path) -> dict[str, dict[str, Any]]:
    """Reconcile all case directories found under ``assets_root``."""
    results: dict[str, dict[str, Any]] = {}
    known_ids = set(CANONICAL.keys())
    for case_dir in sorted(assets_root.iterdir()):
        if not case_dir.is_dir():
            continue
        man_path = case_dir / "manifest.json"
        if not man_path.exists():
            continue
        cid = case_dir.name
        try:
            man = reconcile_case(case_dir, case_id=cid)
            results[cid] = man
        except Exception as exc:
            print(f"  [{cid}] ERROR: {exc}")
    return results


def verify_sha256(assets_root: Path, case_id: str | None = None) -> bool:
    """Verify that every manifest.assets[] sha256 matches the on-disk file.

    Returns True iff every entry passes. Prints failures.
    """
    ok = True
    case_dirs = (
        [assets_root / case_id] if case_id
        else [d for d in sorted(assets_root.iterdir()) if d.is_dir()]
    )
    for case_dir in case_dirs:
        man_path = case_dir / "manifest.json"
        if not man_path.exists():
            continue
        with open(man_path) as fh:
            man = json.load(fh)
        cid = man.get("case_id", case_dir.name)
        for asset in man.get("assets", []):
            path = case_dir / asset["filename"]
            if not path.exists():
                print(f"  MISSING [{cid}] {asset['filename']}")
                ok = False
                continue
            actual = _sha256(path)
            if actual != asset.get("sha256"):
                print(f"  SHA256 MISMATCH [{cid}] {asset['filename']}: "
                      f"manifest={asset.get('sha256','?')[:16]}… "
                      f"actual={actual[:16]}…")
                ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    DEFAULT_ASSETS = ROOT / "docs" / "public" / "gallery" / "assets"

    parser = argparse.ArgumentParser(
        description="Reconcile gallery manifests: rebuild assets[] from disk with "
                    "canonical types, atomic write."
    )
    parser.add_argument(
        "--assets-root", type=Path, default=DEFAULT_ASSETS,
        help=f"Root assets directory (default: {DEFAULT_ASSETS})."
    )
    parser.add_argument(
        "--case", default=None,
        help="Single case id to reconcile (default: all cases found)."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After reconciling, verify all sha256 hashes match on-disk files."
    )
    args = parser.parse_args(argv)

    if not args.assets_root.exists():
        print(f"assets root not found: {args.assets_root}")
        return 1

    if args.case:
        case_dir = args.assets_root / args.case
        if not case_dir.is_dir():
            print(f"case dir not found: {case_dir}")
            return 1
        try:
            reconcile_case(case_dir, case_id=args.case)
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1
    else:
        reconcile_all(args.assets_root)

    if args.verify:
        ok = verify_sha256(args.assets_root, args.case)
        if ok:
            print("sha256 verification: ALL PASS")
        else:
            print("sha256 verification: FAILURES (see above)")
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
