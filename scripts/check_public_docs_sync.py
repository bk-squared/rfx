#!/usr/bin/env python3
"""Audit drift between research/rfx public-doc sources and the gitops snapshot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

DOC_EXTS = {".md", ".mdx"}
SKIP_PUBLIC_ROOT_FILES = {"site_map.json"}


@dataclass
class SectionReport:
    name: str
    source_dir: str
    deploy_dir: str
    source_exists: bool
    deploy_exists: bool
    source_files: list[str]
    deploy_files: list[str]
    source_only: list[str]
    deploy_only: list[str]
    content_drift: list[str]


@dataclass
class AuditReport:
    repo_root: str
    deploy_root: str
    sections: list[SectionReport]
    unmanaged_deploy_entries: list[str]

    @property
    def has_drift(self) -> bool:
        return bool(self.unmanaged_deploy_entries) or any(
            (not s.source_exists)
            or (not s.deploy_exists)
            or s.source_only
            or s.deploy_only
            or s.content_drift
            for s in self.sections
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to research/rfx.",
    )
    parser.add_argument(
        "--deploy-root",
        type=Path,
        default=None,
        help="Path to gitops seed-pages/rfx root.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def default_deploy_root(repo_root: Path) -> Path:
    workspace = repo_root.parent.parent
    return (
        workspace
        / "infra"
        / "remilab-sites-gitops"
        / "deploy"
        / "obsidian-stack"
        / "astro-starlight-presets"
        / "public"
        / "seed-pages"
        / "rfx"
    )


def is_doc_file(path: Path) -> bool:
    return path.suffix in DOC_EXTS and path.name not in SKIP_PUBLIC_ROOT_FILES


def is_generated_api_asset(path: Path) -> bool:
    return not is_doc_file(path)


def iter_files(
    directory: Path,
    *,
    recursive: bool,
    include: Callable[[Path], bool],
    ignore_rel_prefixes: tuple[str, ...] = (),
) -> Iterable[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    for path in sorted(iterator):
        if not path.is_file() or not include(path):
            continue
        rel = path.relative_to(directory).as_posix()
        if any(rel == prefix or rel.startswith(f"{prefix}/") for prefix in ignore_rel_prefixes):
            continue
        yield path


def load_files(
    directory: Path,
    *,
    recursive: bool,
    include: Callable[[Path], bool],
    ignore_rel_prefixes: tuple[str, ...] = (),
) -> dict[str, Path]:
    return {
        path.relative_to(directory).as_posix(): path
        for path in iter_files(
            directory,
            recursive=recursive,
            include=include,
            ignore_rel_prefixes=ignore_rel_prefixes,
        )
    }


def compare_section(
    name: str,
    source_dir: Path,
    deploy_dir: Path,
    *,
    recursive: bool = True,
    include_source: Callable[[Path], bool] = is_doc_file,
    include_deploy: Callable[[Path], bool] = is_doc_file,
    ignore_source_rel_prefixes: tuple[str, ...] = (),
    ignore_deploy_rel_prefixes: tuple[str, ...] = (),
) -> SectionReport:
    source_exists = source_dir.exists()
    deploy_exists = deploy_dir.exists()
    source_files_map = (
        load_files(
            source_dir,
            recursive=recursive,
            include=include_source,
            ignore_rel_prefixes=ignore_source_rel_prefixes,
        )
        if source_exists
        else {}
    )
    deploy_files_map = (
        load_files(
            deploy_dir,
            recursive=recursive,
            include=include_deploy,
            ignore_rel_prefixes=ignore_deploy_rel_prefixes,
        )
        if deploy_exists
        else {}
    )
    source_files = sorted(source_files_map)
    deploy_files = sorted(deploy_files_map)
    source_only = sorted(set(source_files) - set(deploy_files))
    deploy_only = sorted(set(deploy_files) - set(source_files))
    content_drift = [
        rel
        for rel in sorted(set(source_files) & set(deploy_files))
        if source_files_map[rel].read_bytes() != deploy_files_map[rel].read_bytes()
    ]
    return SectionReport(
        name=name,
        source_dir=str(source_dir),
        deploy_dir=str(deploy_dir),
        source_exists=source_exists,
        deploy_exists=deploy_exists,
        source_files=source_files,
        deploy_files=deploy_files,
        source_only=source_only,
        deploy_only=deploy_only,
        content_drift=content_drift,
    )


def discover_public_subtrees(public_root: Path) -> list[Path]:
    return sorted(path for path in public_root.iterdir() if path.is_dir())


def managed_public_entries(public_root: Path, public_subtrees: list[Path]) -> set[str]:
    entries = {"agent", *(subtree.name for subtree in public_subtrees)}
    entries.update(
        path.name
        for path in public_root.iterdir()
        if path.is_file() and is_doc_file(path)
    )
    return entries


def find_unmanaged_deploy_entries(deploy_root: Path, managed_entries: set[str]) -> list[str]:
    if not deploy_root.exists():
        return []
    unmanaged: list[str] = []
    for path in sorted(deploy_root.iterdir()):
        if path.name in managed_entries:
            continue
        if path.is_dir() or is_doc_file(path):
            unmanaged.append(path.name)
    return unmanaged


def make_report(repo_root: Path, deploy_root: Path) -> AuditReport:
    public_root = repo_root / "docs" / "public"
    public_subtrees = discover_public_subtrees(public_root)
    generated_api_source = repo_root / "docs" / "api"

    sections = [
        compare_section("top-level", public_root, deploy_root, recursive=False),
        *(
            compare_section(subtree.name, subtree, deploy_root / subtree.name)
            for subtree in public_subtrees
        ),
        compare_section("agent", repo_root / "docs" / "agent", deploy_root / "agent"),
    ]

    if generated_api_source.exists() or (deploy_root / "api" / "generated").exists():
        sections.append(
            compare_section(
                "api-generated-assets",
                generated_api_source,
                deploy_root / "api" / "generated",
                include_source=is_generated_api_asset,
                include_deploy=is_generated_api_asset,
            )
        )

    return AuditReport(
        repo_root=str(repo_root),
        deploy_root=str(deploy_root),
        sections=sections,
        unmanaged_deploy_entries=find_unmanaged_deploy_entries(
            deploy_root,
            managed_public_entries(public_root, public_subtrees),
        ),
    )


def format_text(report: AuditReport) -> str:
    lines: list[str] = [
        f"repo_root:   {report.repo_root}",
        f"deploy_root: {report.deploy_root}",
        "",
    ]
    for section in report.sections:
        lines.extend(
            [
                f"[{section.name}]",
                f"  source_exists: {section.source_exists}",
                f"  deploy_exists: {section.deploy_exists}",
                f"  source_files:  {len(section.source_files)}",
                f"  deploy_files:  {len(section.deploy_files)}",
            ]
        )
        if section.source_only:
            lines.append("  source_only:")
            lines.extend(f"    - {item}" for item in section.source_only)
        if section.deploy_only:
            lines.append("  deploy_only:")
            lines.extend(f"    - {item}" for item in section.deploy_only)
        if section.content_drift:
            lines.append("  content_drift:")
            lines.extend(f"    - {item}" for item in section.content_drift)
        if not (
            section.source_only
            or section.deploy_only
            or section.content_drift
            or not section.source_exists
            or not section.deploy_exists
        ):
            lines.append("  status: in sync")
        lines.append("")
    if report.unmanaged_deploy_entries:
        lines.append("[unmanaged_deploy_entries]")
        lines.extend(f"  - {item}" for item in report.unmanaged_deploy_entries)
        lines.append("")
    lines.append(f"drift_detected: {report.has_drift}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    deploy_root = (args.deploy_root or default_deploy_root(repo_root)).resolve()
    report = make_report(repo_root, deploy_root)
    if args.format == "json":
        print(json.dumps({"has_drift": report.has_drift, **asdict(report)}, indent=2))
    else:
        print(format_text(report))
    return 1 if args.strict and report.has_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
