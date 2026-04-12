#!/usr/bin/env python3
"""Verify that the public site map resolves to canonical source pages."""

from __future__ import annotations

import json
from pathlib import Path

PRIMARY_ROUTE_PREFIXES = ("rfx/guide", "rfx/agent")
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
    rel = slug.removeprefix("rfx/")
    candidates = [
        repo_root / "docs" / "public" / f"{rel}.md",
        repo_root / "docs" / "public" / f"{rel}.mdx",
        repo_root / "docs" / "public" / rel / "index.md",
        repo_root / "docs" / "public" / rel / "index.mdx",
        repo_root / "docs" / "agent" / f"{rel.removeprefix('agent/')}.md",
        repo_root / "docs" / "agent" / f"{rel.removeprefix('agent/')}.mdx",
    ]
    return any(path.exists() for path in candidates)


def is_primary_route_slug(slug: str) -> bool:
    return any(slug == prefix or slug.startswith(f"{prefix}/") for prefix in PRIMARY_ROUTE_PREFIXES)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    site_map = json.loads((repo_root / "docs" / "public" / "site_map.json").read_text())

    missing: list[str] = []
    invalid_primary_routes: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    quarantined_legacy, unexpected_legacy = check_legacy_guide_retired(repo_root)

    for group in site_map["groups"]:
        for slug in group["items"]:
            if slug in seen:
                duplicates.add(slug)
            seen.add(slug)
            if not is_primary_route_slug(slug):
                invalid_primary_routes.append(slug)
            if not resolve_slug(repo_root, slug):
                missing.append(slug)

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
    if quarantined_legacy:
        print("quarantined legacy docs/guide files:")
        for rel in quarantined_legacy:
            print(f"  - {rel}: {LEGACY_GUIDE_QUARANTINE[rel]}")
    if unexpected_legacy:
        print("unexpected legacy docs/guide files:")
        for rel in unexpected_legacy:
            print(f"  - {rel}")

    if duplicates or invalid_primary_routes or missing or unexpected_legacy:
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
