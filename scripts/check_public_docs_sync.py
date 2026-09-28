#!/usr/bin/env python3
"""Audit authored pages AND generated static bytes against a pinned GitOps snapshot."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from export_public_docs_to_gitops import (
    check_no_symlinks, contained_destination, default_gitops_root, export_snapshot,
)


def file_map(root: Path) -> dict[str, bytes]:
    check_no_symlinks(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and p.relative_to(root).parts[0] not in {"versions", "dev"}}


def make_report(repo_root: Path, deploy_root: Path, bundle_dir: Path | None = None) -> dict:
    # Never derive a deletion-capable export destination from unvalidated JSON.
    prefix = "rfx"
    if bundle_dir:
        from build_public_docs_bundle import validate_bundle, validate_public_identity
        manifest = validate_bundle(bundle_dir, repo_root)
        prefix = str(validate_public_identity(manifest["base_url"], manifest["channel"]))
    with tempfile.TemporaryDirectory(prefix="rfx-docs-sync-") as tmp:
        expected_root = contained_destination(Path(tmp) / "snapshot", prefix)
        export_snapshot(repo_root, expected_root, bundle_dir)
        source, deployed = file_map(expected_root), file_map(deploy_root)
    report = {"repo_root": str(repo_root), "deploy_root": str(deploy_root),
              "source_files": len(source), "deploy_files": len(deployed),
              "source_only": sorted(source.keys() - deployed.keys()),
              "deploy_only": sorted(deployed.keys() - source.keys()),
              "content_drift": sorted(p for p in source.keys() & deployed.keys() if source[p] != deployed[p])}
    report["has_drift"] = bool(report["source_only"] or report["deploy_only"] or report["content_drift"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--deploy-root", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    deploy = args.deploy_root or default_gitops_root(repo) / "deploy/obsidian-stack/astro-starlight-presets/public/seed-pages/rfx"
    report = make_report(repo, deploy, args.bundle_dir)
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{'drift_detected' if key == 'has_drift' else key}: {value}")
    return int(args.strict and report["has_drift"])


if __name__ == "__main__":
    raise SystemExit(main())
