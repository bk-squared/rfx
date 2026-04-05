#!/usr/bin/env python3
"""Verify that the public site map resolves to canonical source pages."""

from __future__ import annotations

import json
from pathlib import Path


def check_legacy_guide_retired(repo_root: Path) -> list[str]:
    legacy_dir = repo_root / "docs" / "guide"
    unexpected: list[str] = []
    for path in sorted(legacy_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(legacy_dir).as_posix()
        if rel != "index.md":
            unexpected.append(rel)
    return unexpected


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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    site_map = json.loads((repo_root / "docs" / "public" / "site_map.json").read_text())

    missing: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    legacy_files = check_legacy_guide_retired(repo_root)

    for group in site_map["groups"]:
        for slug in group["items"]:
            if slug in seen:
                duplicates.add(slug)
            seen.add(slug)
            if not resolve_slug(repo_root, slug):
                missing.append(slug)

    if duplicates:
        print("duplicate slugs:")
        for slug in sorted(duplicates):
            print(f"  - {slug}")
    if missing:
        print("missing slugs:")
        for slug in missing:
            print(f"  - {slug}")
    if legacy_files:
        print("unexpected legacy docs/guide files:")
        for rel in legacy_files:
            print(f"  - {rel}")

    if duplicates or missing or legacy_files:
        return 1

    print(f"site_map OK: {len(seen)} slugs resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
