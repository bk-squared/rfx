"""Publication-boundary and version-scope contracts; no prose equality gate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import build_public_docs_bundle as bundle
import check_public_docs_sync as sync
from export_public_docs_to_gitops import contained_destination, export_snapshot, transformed_source


@pytest.mark.parametrize("path", ["../secret.md", "/etc/passwd", "markdown/../../.env",
                                  "api/generated/agent/data.html", "markdown/.env", "api\\secret.json"])
def test_untrusted_bundle_paths_are_rejected(path):
    with pytest.raises(ValueError):
        bundle.safe_relative(path)


def test_mdx_conversion_preserves_programs_and_scopes_media_and_links():
    source = '''---
title: Example
---
import { Card } from '@astrojs/starlight/components';
<Card title="Example"><a href="/rfx/api/">API</a></Card>
<video controls src="/rfx/examples/assets/demo.mp4" poster="/rfx/examples/assets/poster.png"></video>
[Related](../other/)
```python
import numpy as np
value = "<div>preserve program bytes</div>"
```
'''
    _, markdown = bundle.clean_markdown(source, route="examples/demo", base_url="https://remilab.ai/rfx/versions/v1.8.0", source_sha="a" * 40)
    assert "@astrojs" not in markdown
    assert '<video' not in markdown and '<Card' not in markdown
    assert 'value = "<div>preserve program bytes</div>"' in markdown
    assert "https://remilab.ai/rfx/versions/v1.8.0/examples/assets/demo.mp4" in markdown
    assert "https://remilab.ai/rfx/versions/v1.8.0/examples/other/" in markdown
    assert "/versions/v1.8.0/versions/" not in markdown


def test_source_link_scope_does_not_pin_unrelated_repositories():
    text = '[A](/rfx/api/) [B](https://github.com/bk-squared/rfx/blob/main/rfx/api.py) [C](https://github.com/other/repo/blob/main/x)'
    changed = transformed_source(text, "https://remilab.ai/rfx/versions/v1.8.0", "a" * 40)
    assert "(/rfx/versions/v1.8.0/api/)" in changed
    assert f"/rfx/blob/{'a' * 40}/rfx/api.py" in changed
    assert "other/repo/blob/main/x" in changed


def fixture_bundle(tmp_path):
    root = tmp_path / "bundle"
    files = root / "files"
    files.mkdir(parents=True)
    (files / "llms.txt").write_text("Public index")
    manifest = {"schema_version": 1, "source_sha": "a" * 40,
                "base_url": "https://remilab.ai/rfx", "channel": "development",
                "files": {"llms.txt": bundle.digest(files / "llms.txt")}}
    bundle.write_json(files / "docs-manifest.json", manifest)
    return root


def test_bundle_tamper_extra_file_and_symlink_are_rejected(tmp_path):
    root = fixture_bundle(tmp_path)
    bundle.validate_bundle(root)
    file = root / "files/llms.txt"
    file.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        bundle.validate_bundle(root)
    file.write_text("Public index")
    extra = root / "files/unlisted.txt"
    extra.write_text("private")
    with pytest.raises(ValueError, match="unlisted"):
        bundle.validate_bundle(root)
    extra.unlink()
    file.unlink()
    file.symlink_to(tmp_path / "outside")
    with pytest.raises(SystemExit, match="symlink"):
        bundle.validate_bundle(root)


def test_export_keeps_immutable_releases_but_removes_stale_content(tmp_path):
    repo = tmp_path / "repo"
    public = repo / "docs/public"
    public.mkdir(parents=True)
    (public / "index.mdx").write_text("public")
    (public / "untracked-secret.md").write_text("private")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "docs/public/index.mdx"], check=True)
    destination = tmp_path / "rfx"
    release = destination / "versions/v1.8.0"
    release.mkdir(parents=True)
    (release / "index.mdx").write_text("immutable")
    (destination / "obsolete.html").write_text("old")
    export_snapshot(repo, destination)
    assert (release / "index.mdx").read_text() == "immutable"
    assert (destination / "index.mdx").is_file()
    assert not (destination / "obsolete.html").exists()
    assert not (destination / "untracked-secret.md").exists()


def test_export_rejects_source_symlink_before_mutation(tmp_path):
    repo = tmp_path / "repo"
    public = repo / "docs/public"
    public.mkdir(parents=True)
    (public / "escape").symlink_to(tmp_path)
    destination = tmp_path / "rfx"
    destination.mkdir()
    (destination / "index.mdx").write_text("still here")
    with pytest.raises(SystemExit, match="symlink"):
        export_snapshot(repo, destination)
    assert (destination / "index.mdx").read_text() == "still here"


def test_manifest_disallows_unsafe_path_even_with_valid_hash(tmp_path):
    root = fixture_bundle(tmp_path)
    files = root / "files"
    private = files / "markdown/agent/internal.md"
    private.parent.mkdir(parents=True)
    private.write_text("private")
    manifest = json.loads((files / "docs-manifest.json").read_text())
    manifest["files"]["markdown/agent/internal.md"] = bundle.digest(private)
    bundle.write_json(files / "docs-manifest.json", manifest)
    with pytest.raises(ValueError, match="excluded publication path"):
        bundle.validate_bundle(root)


@pytest.mark.parametrize("base_url,channel", [
    ("https://remilab.ai/rfx/../../outside", "development"),
    ("https://remilab.ai/rfx/versions/..", "release"),
    ("https://remilab.ai/rfx/versions/%2e%2e", "release"),
    ("https://remilab.ai/rfx//versions/v1.8.0", "release"),
    ("https://evil.example/rfx", "development"),
    ("https://remilab.ai.evil.example/rfx", "development"),
    ("https://user@remilab.ai/rfx", "development"),
    ("https://remilab.ai:443/rfx", "development"),
    ("http://remilab.ai/rfx", "development"),
    ("https://remilab.ai/other", "development"),
    ("https://remilab.ai/rfx/agent", "development"),
    ("https://remilab.ai/rfx?target=outside", "development"),
    ("https://remilab.ai/rfx#outside", "development"),
    ("https://remilab.ai/rfx", "unknown"),
    ("https://remilab.ai/rfx", ["development"]),
    (None, "development"),
])
def test_manifest_identity_is_rejected_before_export_or_temp_creation(tmp_path, monkeypatch, base_url, channel):
    root = fixture_bundle(tmp_path)
    manifest_path = root / "files/docs-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(base_url=base_url, channel=channel)
    bundle.write_json(manifest_path, manifest)
    sentinel = tmp_path / "outside/keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("owned by another task")

    def must_not_call(*args, **kwargs):
        raise AssertionError("invalid identity reached a temporary/export operation")

    monkeypatch.setattr(sync.tempfile, "TemporaryDirectory", must_not_call)
    monkeypatch.setattr(sync, "export_snapshot", must_not_call)
    with pytest.raises(ValueError):
        bundle.validate_bundle(root)
    with pytest.raises(ValueError):
        sync.make_report(tmp_path / "unused-source", sentinel.parent, root)
    assert sentinel.read_text() == "owned by another task"


@pytest.mark.parametrize("url,channel,prefix", [
    ("https://remilab.ai/rfx", "development", "rfx"),
    ("https://remilab.ai/rfx/versions/v1.8.0", "release", "rfx/versions/v1.8.0"),
    ("https://remilab.ai/rfx/versions/v2.0.0-rc.1", "release", "rfx/versions/v2.0.0-rc.1"),
])
def test_valid_identity_resolves_within_fixed_parent(tmp_path, url, channel, prefix):
    parsed = bundle.validate_public_identity(url, channel)
    assert str(parsed) == prefix
    parent = tmp_path / "fixed-snapshot"
    destination = contained_destination(parent, str(parsed))
    assert destination.is_relative_to(parent.resolve())
    assert destination == parent / prefix


@pytest.mark.parametrize("prefix", ["../rfx", "rfx/../../outside", "/rfx", "rfx/versions/..", "other"])
def test_export_destination_cannot_escape_fixed_parent(tmp_path, prefix):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError):
        contained_destination(tmp_path / "fixed-snapshot", prefix)
    assert sentinel.read_text() == "keep"


def test_export_destination_rejects_containment_escape_through_symlink(tmp_path):
    parent = tmp_path / "fixed-snapshot"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (parent / "rfx").symlink_to(outside, target_is_directory=True)
    with pytest.raises((SystemExit, ValueError)):
        contained_destination(parent, "rfx")
    assert (outside / "keep.txt").read_text() == "keep"


def test_direct_export_rejects_traversal_before_reading_or_mutating(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="traversal"):
        export_snapshot(tmp_path / "unused-source", tmp_path / "rfx/../outside")
    assert (outside / "keep.txt").read_text() == "keep"


def test_builder_uses_the_same_identity_validation_before_loading_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "check_source", lambda *args: "a" * 40)
    monkeypatch.setattr(bundle, "toolchain", lambda: pytest.fail("invalid identity reached runtime loading"))
    with pytest.raises(ValueError):
        bundle.build(tmp_path, tmp_path / "output", "https://remilab.ai/rfx/../outside", "development", None)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("has_authored_index", [False, True])
def test_generated_api_index_preserves_authoring_or_supplies_release_fallback(tmp_path, has_authored_index):
    generated = tmp_path / "api/generated"
    generated.mkdir(parents=True)
    original_menu = b"<html><body><a href='rfx.html'>module</a></body></html>"
    (generated / "index.html").write_bytes(original_menu)
    base = "https://remilab.ai/rfx/versions/v1.8.0"
    sha = "7" * 40
    bundle.finish_generated_index(
        generated, has_authored_index=has_authored_index, base_url=base,
        channel="release", package_version="1.8.0", source_sha=sha,
    )
    assert (generated / "index-pdoc.html").read_bytes() == original_menu
    assert (generated / "index.html").exists() is not has_authored_index
    if not has_authored_index:
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []
                self.refresh = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if "href" in attrs:
                    self.links.append(attrs["href"])
                if tag == "meta" and attrs.get("http-equiv") == "refresh":
                    self.refresh = attrs["content"]

        landing = (generated / "index.html").read_text()
        parsed = Links()
        parsed.feed(landing)
        assert parsed.refresh == f"0; url={base}/api/generated/rfx.html"
        assert f"{base}/api/generated/rfx.html" in parsed.links
        assert f"https://github.com/bk-squared/rfx/tree/{sha}" in parsed.links
        assert all(url.startswith(base + "/") or sha in url for url in parsed.links)
        assert "1.8.0" in landing and sha in landing
        assert bundle.allowed_artifact("api/generated/index.html")


def test_clean_markdown_removes_jsx_snippet_markers_and_preserves_python():
    source = """{/* intro-snippet:start */}
```python
print("hello")
```
{/* intro-snippet:end */}
"""
    _, markdown = bundle.clean_markdown(
        source, route="guide/first-run", base_url="https://remilab.ai/rfx", source_sha="a" * 40,
    )
    assert "intro-snippet" not in markdown
    assert '{/*' not in markdown and '*/}' not in markdown
    assert '```python\nprint("hello")\n```' in markdown
