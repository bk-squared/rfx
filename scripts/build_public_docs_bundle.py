#!/usr/bin/env python3
"""Build a SHA-pinned public docs bundle from tracked sources (never private notes).

The output is disposable. Use the same interpreter/dependency environment for
byte-for-byte reproduction, then pass --bundle-dir to the public-docs exporter.
--repo-root also supports a clean historical release worktree: imports and pdoc
run from that worktree, not from the checkout that contains this script.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import inspect
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit

import yaml

from export_public_docs_to_gitops import check_no_symlinks, get_tracked_files

SOURCE_REPOSITORY = "https://github.com/bk-squared/rfx"
SUPPORT_FILES = ("support_matrix.json", "sparameter_support_matrix.json")
FORBIDDEN_PARTS = {"agent", "agent-memory", "agent_memory", "research_notes", ".env", ".omx", ".omc"}
GENERATOR_VERSION = 1
GENERATOR_ROOT = Path(__file__).resolve().parents[1]


def toolchain() -> dict:
    requirements = GENERATOR_ROOT / "scripts/requirements-public-docs.txt"
    pins = dict(line.split("==", 1) for line in requirements.read_text().splitlines()
                if line and not line.startswith("#"))
    actual = {name: importlib.metadata.version(name) for name in pins}
    if actual != pins:
        changed = [name for name in pins if actual[name] != pins[name]]
        raise ValueError(f"documentation dependency pins differ: {changed}; install {requirements}")
    return {"python_version": sys.version.split()[0], "dependencies": actual}



def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"unsafe bundle path: {name}")
    if any(part.startswith(".") or part in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError(f"excluded publication path: {name}")
    return path



def validate_site_prefix(prefix: str) -> PurePosixPath:
    """Accept only the owned root or one named immutable release subtree."""
    if not isinstance(prefix, str) or not re.fullmatch(r"rfx(?:/versions/[A-Za-z0-9][A-Za-z0-9._-]*)?", prefix):
        raise ValueError("invalid public site prefix")
    return safe_relative(prefix)


def validate_public_identity(base_url: str, channel: str) -> PurePosixPath:
    """Validate untrusted manifest identity before using it as a filesystem path."""
    if not isinstance(channel, str) or channel not in {"development", "release"}:
        raise ValueError("invalid documentation channel")
    if not isinstance(base_url, str):
        raise ValueError("invalid documentation base URL")
    parsed = urlsplit(base_url)
    if (parsed.scheme != "https" or parsed.netloc != "remilab.ai"
            or parsed.query or parsed.fragment or base_url != "https://remilab.ai" + parsed.path):
        raise ValueError("invalid documentation base URL origin or components")
    return validate_site_prefix(parsed.path.removeprefix("/"))


def allowed_artifact(name: str) -> bool:
    path = safe_relative(name)
    if name in {"llms.txt", "llms-full.txt", "api/inventory.json"}:
        return True
    if name in {f"api/support/{f}" for f in SUPPORT_FILES}:
        return True
    if path.parts[0] == "markdown" and path.suffix == ".md":
        return True
    if path.parts[:2] == ("api", "generated"):
        return path.suffix in {".html", ".js", ".css", ".svg", ".png", ".woff2"}
    return False


def source_inputs(root: Path) -> list[Path]:
    tracked = get_tracked_files(root, "docs/public", "rfx", "docs/pdoc_templates", "docs/guides", "pyproject.toml")
    result = []
    for path in sorted(tracked):
        rel = path.relative_to(root)
        if rel.parts[:2] in {("docs", "public"), ("docs", "pdoc_templates")}:
            result.append(path)
        elif rel.parts[0] == "rfx" and path.suffix == ".py":
            result.append(path)
        elif rel.as_posix() == "pyproject.toml" or rel.as_posix() in {f"docs/guides/{name}" for name in SUPPORT_FILES}:
            result.append(path)
    return result


def check_source(root: Path, source_sha: str | None = None) -> str:
    sha = git(root, "rev-parse", "HEAD")
    if source_sha and git(root, "rev-parse", source_sha) != sha:
        raise ValueError("source SHA must be the checked-out HEAD")
    paths = ["docs/public", "rfx", "docs/pdoc_templates", "docs/guides", "pyproject.toml"]
    if git(root, "status", "--porcelain", "--untracked-files=no", "--", *paths):
        raise ValueError("commit source inputs before building a SHA-pinned bundle")
    for rel in ("docs/public", "rfx", "docs/pdoc_templates"):
        check_no_symlinks(root / rel)
    check_no_symlinks(root / "pyproject.toml")
    for name in SUPPORT_FILES:
        check_no_symlinks(root / "docs/guides" / name)
    return sha


def page_route(path: Path) -> str:
    route = path.with_suffix("").as_posix()
    return re.sub(r"(^|/)index$", "", route).rstrip("/")


def version_url(url: str, base_url: str, source_sha: str) -> str:
    """Scope RFX links to this channel and pin repository main/master links."""
    url = re.sub(r"https://github.com/bk-squared/rfx/(blob|tree)/(main|master)/",
                 rf"{SOURCE_REPOSITORY}/\1/{source_sha}/", url)
    url = re.sub(r"https://raw.githubusercontent.com/bk-squared/rfx/(main|master)/",
                 f"https://raw.githubusercontent.com/bk-squared/rfx/{source_sha}/", url)
    if base_url != "https://remilab.ai/rfx" and (url == base_url or url.startswith(base_url + "/")):
        return url
    for prefix in ("https://remilab.ai/rfx", "/rfx"):
        if url == prefix or url.startswith(prefix + "/") or url.startswith(prefix + "#"):
            return base_url + url[len(prefix):]
    return url


def clean_markdown(text: str, *, route: str, base_url: str, source_sha: str) -> tuple[dict, str]:
    """Remove presentation-only MDX while retaining code, captions and media URLs."""
    metadata = {}
    if text.startswith("---\n"):
        front, text = text[4:].split("\n---", 1)
        metadata = yaml.safe_load(front) or {}
    page_url = base_url + "/" + (route + "/" if route else "")

    def target(url: str) -> str:
        url = version_url(html.unescape(url), base_url, source_sha)
        if url.startswith("#") or urlsplit(url).scheme:
            return url
        return urljoin(page_url, url)

    # Code fences are copied verbatim: removing tags/imports inside Python or
    # shell snippets changes the documented program.
    pieces = re.split(r"(^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*$)", text,
                      flags=re.M | re.S)
    for idx in range(0, len(pieces), 2):
        prose = pieces[idx]
        prose = re.sub(r"^import\s+.*?from\s+['\"].*?['\"];?\s*$", "", prose, flags=re.M)
        prose = re.sub(r"\{/\*.*?\*/\}", "", prose, flags=re.S)
        prose = re.sub(r"<!--.*?-->", "", prose, flags=re.S)
        prose = re.sub(r"<(style|script)\b[^>]*>.*?</\1>", "", prose, flags=re.S)

        def component(match: re.Match) -> str:
            tag, attrs = match.group(1), match.group(2)
            attributes = dict(re.findall(r"([\w-]+)=[\"']([^\"']*)[\"']", attrs))
            label = attributes.get("title") or attributes.get("alt") or tag
            url = attributes.get("href") or attributes.get("src")
            output = f"[{label}]({target(url)})\n" if url else ""
            if "poster" in attributes:
                output += f"![Video poster]({target(attributes['poster'])})\n"
            if not url and "title" in attributes:
                output += f"\n### {label}\n"
            return output

        prose = re.sub(r"<(LinkCard|Card|video|source|img|a)\b([^>]*)>", component, prose)
        prose = re.sub(r"</?[A-Za-z][\w.:]*(?:\s[^<>]*?)?/?>", "", prose)
        prose = re.sub(r"(!?\[[^\]]*\]\()([^\s)]+)(\))",
                       lambda m: m[1] + target(m[2]) + m[3], prose)
        pieces[idx] = html.unescape(prose)
    return metadata, re.sub(r"\n{3,}", "\n\n", "".join(pieces)).strip() + "\n"


def readable(value: object) -> str | None:
    if value is inspect.Signature.empty:
        return None
    if isinstance(value, str):
        return value
    if inspect.isclass(value) or inspect.isfunction(value):
        return f"{value.__module__}.{value.__qualname__}"
    rendered = repr(value)
    return re.sub(r" at 0x[0-9a-fA-F]+", "", rendered)


def symbol_entry(obj: object, name: str, root: Path, sha: str, base_url: str) -> dict:
    entry = {"name": name, "docstring": inspect.getdoc(obj) or ""}
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        entry["signature"] = re.sub(r" at 0x[0-9a-fA-F]+", "", str(sig))
        entry["parameters"] = [
            {"name": p.name, "kind": p.kind.name,
             "annotation": readable(p.annotation), "has_default": p.default is not p.empty,
             "default": (repr(p.default) if isinstance(p.default, str) else readable(p.default)) if p.default is not p.empty else None}
            for p in sig.parameters.values() if p.name not in {"self", "cls"}
        ]
        entry["returns"] = {"annotation": readable(sig.return_annotation)}
    try:
        file = Path(inspect.getsourcefile(obj)).resolve()
        source = file.relative_to(root).as_posix()
        line = inspect.getsourcelines(obj)[1]
        entry["source_url"] = f"{SOURCE_REPOSITORY}/blob/{sha}/{source}#L{line}"
    except (TypeError, OSError, ValueError):
        entry["source_url"] = None
    if name.startswith("rfx.Simulation."):
        anchor = name.removeprefix("rfx.")
        entry["reference_url"] = f"{base_url}/api/generated/rfx.html#{anchor}"
    else:
        entry["reference_url"] = f"{base_url}/api/generated/rfx.html#{name.removeprefix('rfx.')}"
    entry["support_url"] = f"{base_url}/api/support-boundaries/"
    return entry


def api_inventory(root: Path, sha: str, base_url: str, channel: str) -> dict:
    # Import the gate from this generator, then put the selected source tree
    # first. A historical build must never reuse an already-imported rfx.
    import check_api_reference
    if "rfx" in sys.modules:
        raise RuntimeError("rfx was imported before selecting the source worktree")
    sys.path.insert(0, str(root))
    import rfx
    if Path(rfx.__file__).resolve().parent != root / "rfx":
        raise RuntimeError("rfx imported from a different source worktree")
    surface = check_api_reference.build_inventory()
    symbols = []
    for name, item in surface["rfx_exports"].items():
        symbols.append({"kind": item["kind"], **symbol_entry(getattr(rfx, name), f"rfx.{name}", root, sha, base_url)})
    for name in surface["simulation_methods"]:
        symbols.append({"kind": "method", **symbol_entry(getattr(rfx.Simulation, name), f"rfx.Simulation.{name}", root, sha, base_url)})
    return {"schema_version": 1, "source_sha": sha, "package_version": rfx.__version__,
            "channel": channel, "base_url": base_url,
            "support_policy": "Importable does not imply validated. Consult the unchanged support contracts and prose for the chosen configuration; no per-symbol support level is inferred.",
            "support_contracts": [f"{base_url}/api/support/{f}" for f in SUPPORT_FILES],
            "symbols": symbols}



def finish_generated_index(generated: Path, *, has_authored_index: bool, base_url: str,
                           channel: str, package_version: str, source_sha: str) -> None:
    """Keep pdoc's module menu and supply an entry point for older source trees."""
    (generated / "index.html").rename(generated / "index-pdoc.html")
    if has_authored_index:
        return
    target = html.escape(f"{base_url}/api/generated/rfx.html", quote=True)
    source = html.escape(f"{SOURCE_REPOSITORY}/tree/{source_sha}", quote=True)
    landing = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        f'<link rel="canonical" href="{target}">\n'
        '<title>rfx generated API reference</title></head><body><main>\n'
        '<h1>Generated API reference</h1>\n'
        f'<p>{html.escape(channel.capitalize())} documentation · '
        f'rfx {html.escape(package_version)} · '
        f'<a href="{source}">{html.escape(source_sha)}</a></p>\n'
        f'<p><a href="{target}">Open the complete Python API reference</a>.</p>\n'
        '</main></body></html>\n'
    )
    (generated / "index.html").write_text(landing)


def build(root: Path, output: Path, base_url: str, channel: str, source_sha: str | None) -> dict:
    root = root.resolve()
    sha = check_source(root, source_sha)
    base_url = base_url.rstrip("/")
    validate_public_identity(base_url, channel)
    environment = toolchain()
    import pdoc
    inventory = api_inventory(root, sha, base_url, channel)
    inputs = source_inputs(root)
    with tempfile.TemporaryDirectory(prefix="rfx-docs-bundle-") as scratch:
        files = Path(scratch) / "files"
        files.mkdir()
        pages = []
        full = []
        public = root / "docs/public"
        for source in inputs:
            if not source.is_relative_to(public) or source.suffix not in {".md", ".mdx"}:
                continue
            rel = source.relative_to(public)
            safe_relative(rel.as_posix())
            route = page_route(rel)
            metadata, body = clean_markdown(source.read_text(), route=route, base_url=base_url, source_sha=sha)
            name = f"markdown/{route or 'index'}.md"
            title = metadata.get("title", route or "rfx")
            source_path = source.relative_to(root).as_posix()
            source_url = f"{SOURCE_REPOSITORY}/blob/{sha}/{source_path}"
            url = base_url + "/" + (route + "/" if route else "")
            header = f"# {title}\n\nChannel: {channel}; package: {inventory['package_version']}; source: {sha}\n\nPage: {url}\nSource: {source_url}\n\n"
            target = files / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(header + body)
            full.append(header + body)
            pages.append({"route": route, "title": title, "description": metadata.get("description", ""),
                          "url": url, "markdown_url": f"{base_url}/{name}", "source_path": source_path,
                          "source_url": source_url, "source_sha256": digest(source)["sha256"]})
        support = []
        for name in SUPPORT_FILES:
            source = root / "docs/guides" / name
            target = files / "api/support" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            support.append({"url": f"{base_url}/api/support/{name}", "source_path": f"docs/guides/{name}",
                            "source_url": f"{SOURCE_REPOSITORY}/blob/{sha}/docs/guides/{name}", **digest(source)})
        generated = files / "api/generated"
        # The historical source may predate the inherited-method rendering fix.
        # A renderer changes presentation, not the selected API implementation.
        templates = GENERATOR_ROOT / "docs/pdoc_templates"
        check_no_symlinks(templates)
        env = {**os.environ, "PYTHONPATH": str(root), "JAX_PLATFORMS": "cpu", "PYTHONHASHSEED": "0"}
        subprocess.run([sys.executable, "-m", "pdoc", "-t", str(templates),
                        "--no-show-source", "--edit-url", f"rfx={SOURCE_REPOSITORY}/blob/{sha}/rfx/",
                        "--footer-text", f"{channel} | rfx {inventory['package_version']} | {sha[:12]}",
                        "-o", str(generated), "rfx", "!rfx.dashboard"],
                       cwd=root, env=env, check=True)
        import check_api_reference
        errors = check_api_reference.check_html(generated, check_api_reference.build_inventory())
        if errors:
            raise ValueError("\n".join(errors))
        finish_generated_index(
            generated, has_authored_index=any(page["route"] == "api/generated" for page in pages),
            base_url=base_url, channel=channel, package_version=inventory["package_version"], source_sha=sha,
        )
        anchors = set(re.findall(r'id="([^"\n]+)"', (generated / "rfx.html").read_text()))
        for symbol in inventory["symbols"]:
            if symbol["name"].removeprefix("rfx.") not in anchors:
                symbol["reference_url"] = None
        write_json(files / "api/inventory.json", inventory)
        # Pin GitHub links in generated docstrings too, and omit source-code
        # panels (the pinned GitHub source is authoritative).
        for file in generated.rglob("*"):
            if file.suffix in {".html", ".js"}:
                text = file.read_text()
                text = text.replace('href="index.html"', 'href="index-pdoc.html"')
                text = text.replace('/index.html"', '/index-pdoc.html"')
                text = re.sub(r"https://github.com/bk-squared/rfx/(blob|tree)/(main|master)/",
                              rf"{SOURCE_REPOSITORY}/\1/{sha}/", text)
                if file.suffix == ".html":
                    banner = (f'<aside aria-label="Documentation version" style="padding:1rem;border:1px solid #bbb;margin-bottom:1rem">'
                              f'<strong>{channel.capitalize()} documentation</strong> · rfx {inventory["package_version"]} · '
                              f'<a href="{SOURCE_REPOSITORY}/tree/{sha}">{sha[:12]}</a><br>'
                              f'<a href="{base_url}/">Manual</a> · <a href="{base_url}/llms.txt">LLM index</a> · '
                              f'<a href="{base_url}/api/inventory.json">Typed inventory</a></aside>')
                    text = text.replace('<main class="pdoc">', '<main class="pdoc">' + banner, 1)
                file.write_text(text)
        nav = json.loads((public / "site_map.json").read_text())
        for group in nav.get("groups", []):
            group["items"] = [base_url + "/" if slug == "rfx" else
                              f"{base_url}/{slug.removeprefix('rfx/').rstrip('/')}/"
                              for slug in group["items"]]
        llms = ["# rfx", "", "> JAX-native electromagnetic simulation: manuals, reproducible examples and generated API reference.", "",
                f"Channel: {channel}. Package version: {inventory['package_version']}. Source commit: {sha}.",
                "", "Read the support boundaries before proposing a configuration. Importability, a successful run,",
                "and a historical visualization do not establish accuracy or a supported gradient.", "", "## API and support", "",
                f"- [Typed API inventory]({base_url}/api/inventory.json): signatures, defaults, annotations, docstrings and pinned sources.",
                f"- [Deep API reference]({base_url}/api/generated/rfx.html)",
                f"- [Support boundaries]({base_url}/markdown/api/support-boundaries.md)",
                *[f"- [{s['source_path']}]({s['url']})" for s in support],
                f"- [Build manifest]({base_url}/docs-manifest.json): source SHA and file hashes.",
                f"- [Combined manual]({base_url}/llms-full.txt)", "", "## Manual and examples", ""]
        llms += [f"- [{p['title']}]({p['markdown_url']}): {p['description']}" for p in pages]
        (files / "llms.txt").write_text("\n".join(llms) + "\n")
        (files / "llms-full.txt").write_text("\n\n---\n\n".join(full))
        check_no_symlinks(files)
        hashes = {}
        for path in sorted(files.rglob("*")):
            if path.is_file():
                name = path.relative_to(files).as_posix()
                if not allowed_artifact(name):
                    raise ValueError(f"unexpected generated artifact: {name}")
                hashes[name] = digest(path)
        manifest = {"schema_version": 1, "source_sha": sha, "package_version": inventory["package_version"],
                    "channel": channel, "base_url": base_url, "source_repository": SOURCE_REPOSITORY,
                    "generator": {"name": "build_public_docs_bundle.py", "version": GENERATOR_VERSION,
                                  "pdoc_version": pdoc.__version__,
                                  "source_sha256": digest(Path(__file__))["sha256"],
                                  "requirements_sha256": digest(GENERATOR_ROOT / "scripts/requirements-public-docs.txt")["sha256"],
                                  "template_sha256": {p.relative_to(templates).as_posix(): digest(p)["sha256"]
                                                      for p in sorted(templates.rglob("*")) if p.is_file()},
                                  "toolchain": environment},
                    "pages": pages, "navigation": nav, "support": support, "files": hashes,
                    "source_inputs": {p.relative_to(root).as_posix(): digest(p) for p in inputs}}
        write_json(files / "docs-manifest.json", manifest)
        if output.exists():
            raise ValueError(f"output already exists; choose a new directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(scratch), output)
    return manifest


def validate_bundle(bundle: Path, root: Path | None = None) -> dict:
    check_no_symlinks(bundle)
    files = bundle / "files"
    manifest = json.loads((files / "docs-manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("invalid docs manifest object")
    validate_public_identity(manifest.get("base_url"), manifest.get("channel"))
    if manifest.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_sha", "")):
        raise ValueError("invalid docs manifest identity")
    expected = manifest["files"]
    actual = {p.relative_to(files).as_posix() for p in files.rglob("*") if p.is_file()}
    if actual != set(expected) | {"docs-manifest.json"}:
        raise ValueError("bundle contains missing or unlisted files")
    for name, record in expected.items():
        if not allowed_artifact(name) or digest(files / name) != record:
            raise ValueError(f"bundle checksum or allowlist failed: {name}")
    if root is not None:
        check_source(root, manifest["source_sha"])
        current = {p.relative_to(root).as_posix(): digest(p) for p in source_inputs(root)}
        if current != manifest["source_inputs"]:
            raise ValueError("bundle source inputs differ from the checked-out source")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("docs/_build/public-docs"))
    parser.add_argument("--base-url", default="https://remilab.ai/rfx")
    parser.add_argument("--channel", choices=("development", "release"), default="development")
    parser.add_argument("--source-sha")
    parser.add_argument("--verify", type=Path, help="validate an existing bundle instead of building")
    args = parser.parse_args()
    if args.verify:
        manifest = validate_bundle(args.verify, args.repo_root.resolve())
    else:
        manifest = build(args.repo_root, args.output_dir.resolve(), args.base_url, args.channel, args.source_sha)
    print(f"public docs bundle: {len(manifest['pages'])} pages, {len(manifest['files'])} files, source {manifest['source_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
