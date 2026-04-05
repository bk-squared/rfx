#!/usr/bin/env python3
"""Verify that the public site map resolves to canonical source pages."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_slug(repo_root: Path, slug: str) -> bool:
    rel = slug.removeprefix("rfx/")
    candidates = [
        repo_root / "docs" / "public" / f"{rel}.md",
        repo_root / "docs" / "public" / f"{rel}.mdx",
        repo_root / "docs" / "public" / rel / "index.md",
        repo_root / "docs" / "public" / rel / "index.mdx",
        repo_root / "docs" / f"{rel}.md",
        repo_root / "docs" / f"{rel}.mdx",
    ]
    return any(path.exists() for path in candidates)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    site_map = json.loads((repo_root / "docs" / "public" / "site_map.json").read_text())

    missing: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()

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

    if duplicates or missing:
        return 1

    print(f"site_map OK: {len(seen)} slugs resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
