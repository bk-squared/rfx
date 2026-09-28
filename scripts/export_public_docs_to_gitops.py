#!/usr/bin/env python3
"""Export tracked public sources and a verified generated bundle into GitOps."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

DOC_EXTS = {".md", ".mdx"}
SKIP_PUBLIC_ROOT_FILES = {"site_map.json"}
# Authored inputs that a supplied bundle publishes itself, hashed in its manifest.
BUNDLE_PUBLISHED_SOURCES = {"showcase/showcase.json"}


def default_gitops_root(repo_root: Path) -> Path:
    return repo_root.parent.parent / "infra/remilab-sites-gitops"


def get_tracked_files(repo_root: Path, *rel_dirs: str) -> frozenset[Path]:
    result = subprocess.run(["git", "-C", str(repo_root), "ls-files", "-z", "--", *rel_dirs],
                            capture_output=True, text=True, check=True)
    return frozenset(repo_root / name for name in result.stdout.split("\0") if name)


def check_no_symlinks(root: Path) -> None:
    """Reject symlinks including a source/destination ancestor redirected elsewhere."""
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise SystemExit(f"refusing to export symlink: {parent}")
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames + dirnames:
            child = Path(dirpath) / name
            if child.is_symlink():
                raise SystemExit(f"refusing to export symlink: {child}")


def remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def transformed_source(text: str, base_url: str, sha: str) -> str:
    from build_public_docs_bundle import SOURCE_REPOSITORY
    text = re.sub(r"https://github.com/bk-squared/rfx/(blob|tree)/(main|master)/",
                  rf"{SOURCE_REPOSITORY}/\1/{sha}/", text)
    text = re.sub(r"https://raw.githubusercontent.com/bk-squared/rfx/(main|master)/",
                  f"https://raw.githubusercontent.com/bk-squared/rfx/{sha}/", text)
    # Replace only URL occurrences, not arbitrary prose or code string fragments.
    text = re.sub(r"https://remilab\.ai/rfx(?=[/#)\"'\s]|$)", base_url, text)
    from urllib.parse import urlsplit
    prefix = urlsplit(base_url).path
    text = re.sub(r"(?<=[(\"'])/rfx(?=[/#)\"'])", prefix, text)
    return text



def contained_destination(parent: Path, site_prefix: str) -> Path:
    """Resolve a validated RFX target beneath a fixed, caller-owned directory."""
    from build_public_docs_bundle import validate_site_prefix
    prefix = validate_site_prefix(site_prefix)
    check_no_symlinks(parent)
    parent = parent.resolve()
    destination = (parent / prefix).resolve()
    if destination == parent or not destination.is_relative_to(parent):
        raise ValueError("public export destination escapes its parent")
    return destination


def export_snapshot(repo_root: Path, dst_root: Path, bundle_dir: Path | None = None) -> dict | None:
    from build_public_docs_bundle import safe_relative, validate_bundle, validate_public_identity
    if ".." in dst_root.parts:
        raise ValueError("public export destination contains traversal")
    manifest = validate_bundle(bundle_dir, repo_root) if bundle_dir else None
    public_root = repo_root / "docs/public"
    check_no_symlinks(public_root)
    check_no_symlinks(dst_root)
    tracked = sorted(get_tracked_files(repo_root, "docs/public"))
    # Validate every path before replacing any owned destination subtree.
    for src in tracked:
        if src.name != ".gitignore":
            safe_relative(src.relative_to(public_root).as_posix())
        if not src.is_file():
            raise ValueError(f"tracked public source missing: {src}")
    bundled = BUNDLE_PUBLISHED_SOURCES if manifest else set()
    if manifest:
        authored = {p.relative_to(public_root).as_posix() for p in tracked} - bundled
        overlap = authored & (set(manifest["files"]) | {"docs-manifest.json"})
        if overlap:
            raise ValueError(f"generated bundle collides with authored source: {sorted(overlap)}")
        prefix = validate_public_identity(manifest["base_url"], manifest["channel"])
        if dst_root.parts[-len(prefix.parts):] != prefix.parts:
            raise ValueError("bundle base URL and export site prefix disagree")
    dst_root.mkdir(parents=True, exist_ok=True)
    # Preserve immutable release snapshots and the infra-owned dev redirect.
    # All other content in this RFX destination is owned by this exporter.
    for child in dst_root.iterdir():
        if child.name not in {"versions", "dev"}:
            remove_tree(child)
    for src in tracked:
        rel = src.relative_to(public_root)
        if rel.name == ".gitignore" or rel.as_posix() in SKIP_PUBLIC_ROOT_FILES | bundled:
            continue
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if manifest and src.suffix in DOC_EXTS:
            target.write_text(transformed_source(src.read_text(), manifest["base_url"], manifest["source_sha"]))
        else:
            shutil.copyfile(src, target)
    if manifest:
        for source in sorted((bundle_dir / "files").rglob("*")):
            if source.is_file():
                target = dst_root / source.relative_to(bundle_dir / "files")
                if target.exists():
                    raise ValueError(f"generated bundle collides with authored source: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gitops-root", type=Path)
    parser.add_argument("--bundle-dir", type=Path, help="verified build_public_docs_bundle output")
    parser.add_argument("--site-prefix", default="rfx", help="rfx or rfx/versions/<tag>")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    gitops_root = args.gitops_root or default_gitops_root(repo_root)
    dst = contained_destination(
        gitops_root / "deploy/obsidian-stack/astro-starlight-presets/public/seed-pages", args.site_prefix
    )
    if args.check:
        check_no_symlinks(repo_root / "docs/public")
        if args.bundle_dir:
            from build_public_docs_bundle import validate_bundle
            validate_bundle(args.bundle_dir, repo_root)
        print("public sources and supplied bundle verified")
        return 0
    if not gitops_root.is_dir():
        raise ValueError(f"missing GitOps checkout: {gitops_root}")
    manifest = export_snapshot(repo_root, dst, args.bundle_dir)
    print(f"exported public docs: {dst}")
    print(f"generated bundle: {manifest['source_sha'] if manifest else 'absent (source-only export)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
