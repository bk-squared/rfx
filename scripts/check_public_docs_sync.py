#!/usr/bin/env python3
"""Audit drift between research/rfx public-doc sources and the gitops snapshot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DOC_EXTS = {".md", ".mdx"}


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

    @property
    def has_drift(self) -> bool:
        return any(
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


def iter_doc_files(directory: Path, *, recursive: bool = True) -> Iterable[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix in DOC_EXTS:
            yield path


def load_docs(directory: Path, *, recursive: bool = True) -> dict[str, Path]:
    return {
        path.relative_to(directory).as_posix(): path
        for path in iter_doc_files(directory, recursive=recursive)
    }


def compare_section(name: str, source_dir: Path, deploy_dir: Path, *, recursive: bool = True) -> SectionReport:
    source_exists = source_dir.exists()
    deploy_exists = deploy_dir.exists()
    source_docs = load_docs(source_dir, recursive=recursive) if source_exists else {}
    deploy_docs = load_docs(deploy_dir, recursive=recursive) if deploy_exists else {}
    source_files = sorted(source_docs)
    deploy_files = sorted(deploy_docs)
    source_only = sorted(set(source_files) - set(deploy_files))
    deploy_only = sorted(set(deploy_files) - set(source_files))
    content_drift = [
        rel
        for rel in sorted(set(source_files) & set(deploy_files))
        if source_docs[rel].read_text() != deploy_docs[rel].read_text()
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


def make_report(repo_root: Path, deploy_root: Path) -> AuditReport:
    return AuditReport(
        repo_root=str(repo_root),
        deploy_root=str(deploy_root),
        sections=[
            compare_section("top-level", repo_root / "docs" / "public", deploy_root, recursive=False),
            compare_section("guide", repo_root / "docs" / "public" / "guide", deploy_root / "guide"),
            compare_section("agent", repo_root / "docs" / "agent", deploy_root / "agent"),
        ],
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
        if not (section.source_only or section.deploy_only or section.content_drift or not section.source_exists or not section.deploy_exists):
            lines.append("  status: in sync")
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
