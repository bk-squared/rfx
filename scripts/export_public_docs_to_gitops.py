#!/usr/bin/env python3
"""Export canonical public RFX docs from research/rfx into the gitops snapshot."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DOC_EXTS = {".md", ".mdx"}
SKIP_PUBLIC_ROOT_FILES = {"site_map.json"}


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


def sync_generated_api_assets(src: Path, dst: Path) -> None:
    """Sync generated API assets into a curated api/generated subtree.

    Curated doc pages such as ``api/generated/index.mdx`` may already exist in
    ``dst`` and must be preserved. Generated non-doc assets should, however, be
    replaced exactly so stale HTML/JS files do not linger after regeneration.
    """

    dst.mkdir(parents=True, exist_ok=True)
    source_files = sorted(path for path in src.rglob("*") if path.is_file())
    source_rel_files = {path.relative_to(src).as_posix() for path in source_files}

    for path in sorted(dst.rglob("*"), reverse=True):
        rel = path.relative_to(dst).as_posix()
        if path.is_file():
            if path.suffix in DOC_EXTS:
                continue
            if rel not in source_rel_files:
                path.unlink()
        elif path.is_dir():
            try:
                next(path.iterdir())
            except StopIteration:
                path.rmdir()

    for src_file in source_files:
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)


def iter_public_root_docs(public_root: Path) -> list[Path]:
    return sorted(
        path
        for path in public_root.iterdir()
        if path.is_file() and path.suffix in DOC_EXTS and path.name not in SKIP_PUBLIC_ROOT_FILES
    )


def discover_public_subtrees(public_root: Path) -> list[Path]:
    return sorted(path for path in public_root.iterdir() if path.is_dir())


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    gitops_root = (args.gitops_root or default_gitops_root(repo_root)).resolve()

    public_root = repo_root / "docs" / "public"
    source_root_docs = iter_public_root_docs(public_root)
    public_subtrees = discover_public_subtrees(public_root)
    src_agent = repo_root / "docs" / "agent"
    src_generated_api = repo_root / "docs" / "api"

    dst_root = (
        gitops_root
        / "deploy"
        / "obsidian-stack"
        / "astro-starlight-presets"
        / "public"
        / "seed-pages"
        / "rfx"
    )
    dst_generated_api = dst_root / "api" / "generated"

    required = [
        public_root,
        src_agent,
        src_generated_api,
        gitops_root,
        *source_root_docs,
        *public_subtrees,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))

    if args.check:
        print("source paths verified")
        print(f"public_root: {public_root}")
        print(f"root_docs: {[path.name for path in source_root_docs]}")
        print(f"public_subtrees: {[path.name for path in public_subtrees]}")
        print(f"agent: {src_agent}")
        print(f"generated_api: {src_generated_api}")
        print(f"gitops: {gitops_root}")
        return 0

    dst_root.mkdir(parents=True, exist_ok=True)

    managed_root_docs = {
        path.name
        for path in dst_root.iterdir()
        if path.is_file() and path.suffix in DOC_EXTS and path.name not in SKIP_PUBLIC_ROOT_FILES
    }
    source_root_doc_names = {path.name for path in source_root_docs}
    for stale_name in sorted(managed_root_docs - source_root_doc_names):
        (dst_root / stale_name).unlink()
    for src_doc in source_root_docs:
        shutil.copy2(src_doc, dst_root / src_doc.name)

    for subtree in public_subtrees:
        copy_tree(subtree, dst_root / subtree.name)

    copy_tree(src_agent, dst_root / "agent")
    sync_generated_api_assets(src_generated_api, dst_generated_api)

    print("exported public docs to gitops snapshot")
    print(f"target: {dst_root}")
    print(f"root_docs: {[path.name for path in source_root_docs]}")
    print(f"public_subtrees: {[path.name for path in public_subtrees]}")
    print("generated_api_target: api/generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
