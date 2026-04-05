#!/usr/bin/env python3
"""Export canonical public RFX docs from research/rfx into the gitops snapshot."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to research/rfx.",
    )
    parser.add_argument(
        "--gitops-root",
        type=Path,
        default=None,
        help="Path to remilab-sites-gitops repo root.",
    )
    parser.add_argument("--check", action="store_true", help="Only verify source paths exist.")
    return parser.parse_args()


def default_gitops_root(repo_root: Path) -> Path:
    workspace = repo_root.parent.parent
    return workspace / "infra" / "remilab-sites-gitops"


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    gitops_root = (args.gitops_root or default_gitops_root(repo_root)).resolve()

    src_index = repo_root / "docs" / "public" / "index.mdx"
    src_guide = repo_root / "docs" / "public" / "guide"
    src_agent = repo_root / "docs" / "agent"

    dst_root = (
        gitops_root
        / "deploy"
        / "obsidian-stack"
        / "astro-starlight-presets"
        / "public"
        / "seed-pages"
        / "rfx"
    )
    dst_guide = dst_root / "guide"
    dst_agent = dst_root / "agent"

    required = [src_index, src_guide, src_agent, gitops_root]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))

    if args.check:
        print("source paths verified")
        print(f"index: {src_index}")
        print(f"guide: {src_guide}")
        print(f"agent: {src_agent}")
        print(f"gitops: {gitops_root}")
        return 0

    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_index, dst_root / "index.mdx")
    copy_tree(src_guide, dst_guide)
    copy_tree(src_agent, dst_agent)

    print("exported public docs to gitops snapshot")
    print(f"target: {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
