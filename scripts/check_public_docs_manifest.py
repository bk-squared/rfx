#!/usr/bin/env python3
"""Verify that the public site map resolves to canonical source pages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PRIMARY_ROUTE_PREFIXES = ("rfx/showcase", "rfx/guide", "rfx/gallery", "rfx/examples", "rfx/api",
                          "rfx/validation")
LEGACY_GUIDE_QUARANTINE = {
    "documentation_architecture.md": "legacy docs/guide architecture note kept outside the public route manifest",
    "inverse_design_cookbook.md": "legacy cookbook retained until examples hub lands",
    "rf_backend_workflow.md": "legacy backend workflow note retained until API/support split lands",
}


def check_legacy_guide_retired(repo_root: Path) -> tuple[list[str], list[str]]:
    legacy_dir = repo_root / "docs" / "guide"
    quarantined: list[str] = []
    unexpected: list[str] = []
    for path in sorted(legacy_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(legacy_dir).as_posix()
        if rel == "index.md":
            continue
        if rel in LEGACY_GUIDE_QUARANTINE:
            quarantined.append(rel)
        else:
            unexpected.append(rel)
    return quarantined, unexpected


def resolve_slug(repo_root: Path, slug: str) -> bool:
    rel = "index" if slug == "rfx" else slug.removeprefix("rfx/")
    candidates = [
        repo_root / "docs" / "public" / f"{rel}.md",
        repo_root / "docs" / "public" / f"{rel}.mdx",
        repo_root / "docs" / "public" / rel / "index.md",
        repo_root / "docs" / "public" / rel / "index.mdx",
    ]
    return any(path.exists() for path in candidates)


def is_primary_route_slug(slug: str) -> bool:
    if not re.fullmatch(r"rfx(?:/[a-z0-9_-]+)*", slug):
        return False
    return slug == "rfx" or any(
        slug == prefix or slug.startswith(f"{prefix}/") for prefix in PRIMARY_ROUTE_PREFIXES
    )


def authored_routes(repo_root: Path) -> set[str]:
    """Every authored public page participates in the same navigation contract."""
    public = repo_root / "docs" / "public"
    routes: set[str] = set()
    for path in public.rglob("*"):
        if path.suffix not in {".md", ".mdx"} or "assets" in path.relative_to(public).parts:
            continue
        rel = path.relative_to(public).with_suffix("")
        if rel.name == "index":
            rel = rel.parent
        routes.add("rfx" if str(rel) == "." else f"rfx/{rel.as_posix()}")
    return routes


def check_curated_media(repo_root: Path) -> list[str]:
    """Check the bounded historical bundle without claiming physics validation."""
    root = repo_root / "docs/public/gallery/assets/showcase-20260927"
    catalog_path = root / "media-catalog.json"
    errors: list[str] = []
    if not catalog_path.is_file():
        return ["missing historical media catalog"]
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema_version") != "rfx-public-media-catalog-v1":
        errors.append("unsupported historical media catalog schema")
    seen: set[str] = set()
    allowed_kinds = {"optimization-history", "optimization-history-with-interpolation", "phasor-replay"}
    for asset in catalog.get("assets", []):
        if asset.get("media_kind") not in allowed_kinds or not asset.get("limitations"):
            errors.append(f"{asset.get('id')}: missing animation type or limitations")
        for role in ("video", "poster"):
            entry = asset.get(role, {})
            name = entry.get("path", "")
            if not re.fullmatch(r"[a-z0-9_-]+\.(?:mp4|jpg)", name):
                errors.append(f"{asset.get('id')}: unsafe or unsupported {role} path")
                continue
            if name in seen:
                errors.append(f"duplicate media file: {name}")
            seen.add(name)
            path = root / name
            if path.is_symlink() or not path.is_file():
                errors.append(f"missing regular media file: {name}")
                continue
            data = path.read_bytes()
            if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                errors.append(f"media byte count or SHA-256 mismatch: {name}")
    media_files = {p.name for p in root.iterdir() if p.suffix in {".mp4", ".jpg"}}
    for name in sorted(media_files - seen):
        errors.append(f"uncataloged media file: {name}")
    if not seen:
        errors.append("empty historical media catalog")
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    site_map = json.loads((repo_root / "docs" / "public" / "site_map.json").read_text())

    missing: list[str] = []
    invalid_primary_routes: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    media_errors = check_curated_media(repo_root)
    quarantined_legacy, unexpected_legacy = check_legacy_guide_retired(repo_root)

    for group in site_map["groups"]:
        for slug in group["items"]:
            if slug in seen:
                duplicates.add(slug)
            seen.add(slug)
            if not is_primary_route_slug(slug):
                invalid_primary_routes.append(slug)
            if is_primary_route_slug(slug) and not resolve_slug(repo_root, slug):
                missing.append(slug)

    unlisted = authored_routes(repo_root) - seen

    if duplicates:
        print("duplicate slugs:")
        for slug in sorted(duplicates):
            print(f"  - {slug}")
    if invalid_primary_routes:
        print("site_map primary-route policy violations:")
        for slug in sorted(invalid_primary_routes):
            print(f"  - {slug}")
    if missing:
        print("missing slugs:")
        for slug in missing:
            print(f"  - {slug}")
    if unlisted:
        print("authored public pages missing from site_map:")
        for slug in sorted(unlisted):
            print(f"  - {slug}")
    if media_errors:
        print("curated media issues:")
        for error in media_errors:
            print(f"  - {error}")
    if quarantined_legacy:
        print("quarantined legacy docs/guide files:")
        for rel in quarantined_legacy:
            print(f"  - {rel}: {LEGACY_GUIDE_QUARANTINE[rel]}")
    if unexpected_legacy:
        print("unexpected legacy docs/guide files:")
        for rel in unexpected_legacy:
            print(f"  - {rel}")

    if duplicates or invalid_primary_routes or missing or unexpected_legacy or unlisted or media_errors:
        return 1

    if quarantined_legacy:
        print(
            "site_map OK: "
            f"{len(seen)} slugs resolve; {len(quarantined_legacy)} legacy docs/guide files quarantined"
        )
    else:
        print(f"site_map OK: {len(seen)} slugs resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
