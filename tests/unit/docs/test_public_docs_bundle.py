"""Publication-boundary and version-scope contracts; no prose equality gate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import build_public_docs_bundle as bundle
from export_public_docs_to_gitops import export_snapshot, transformed_source


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
