"""Publication-boundary and version-scope contracts; no prose equality gate."""
from __future__ import annotations

import json
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


def test_export_keeps_immutable_releases_but_removes_stale_content(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    public = repo / "docs/public"
    public.mkdir(parents=True)
    (public / "index.mdx").write_text("public")
    (public / "untracked-secret.md").write_text("private")
    monkeypatch.setattr('export_public_docs_to_gitops.get_tracked_files',
                        lambda *args: frozenset({public / 'index.mdx'}))
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


SHARE = "https://remilab.cnu.ac.kr/share/7c02ad43c580/"


def showcase_catalog(*case_ids):
    return {"schema": "rfx-showcase-catalog/1", "cases": [
        {"id": case, "title": f"Case {case}", "sources": [
            {"folder": folder, "result_json": f"{SHARE}{folder}/result.json"}
            for folder in (f"{case}-a", f"{case}-b")]}
        for case in case_ids]}


def test_llms_showcase_section_lists_each_case_page_catalog_entry_and_records():
    base = "https://remilab.ai/rfx"
    pages = [{"route": route, "markdown_url": f"{base}/markdown/{route}.md"}
             for route in ("showcase/one", "showcase/two", "guide/other")]
    lines = bundle.showcase_index(showcase_catalog("one", "two"), pages, base)
    assert lines[0] == "## Showcase results"
    assert any(f"({base}/showcase/showcase.json)" in line for line in lines[:4])
    for case in ("one", "two"):
        [entry] = [line for line in lines if f"({base}/markdown/showcase/{case}.md)" in line]
        assert f"`{case}`" in entry and f"({base}/showcase/showcase.json)" in entry
        for folder in (f"{case}-a", f"{case}-b"):
            assert f"({SHARE}{folder}/result.json)" in entry
    assert not any("guide/other" in line for line in lines)


@pytest.mark.parametrize("catalog", [
    {**showcase_catalog("one"), "schema": "rfx-showcase-catalog/2"},
    showcase_catalog("no-page"),
])
def test_llms_showcase_section_refuses_an_unknown_schema_or_a_case_without_a_page(catalog):
    pages = [{"route": "showcase/one", "markdown_url": "https://remilab.ai/rfx/markdown/showcase/one.md"}]
    with pytest.raises(ValueError):
        bundle.showcase_index(catalog, pages, "https://remilab.ai/rfx")


def test_the_showcase_catalog_is_the_one_publishable_authored_json():
    assert bundle.allowed_artifact("showcase/showcase.json")
    for name in ("showcase/other.json", "showcase.json", "showcase/showcase.json.bak"):
        assert not bundle.allowed_artifact(name)


def _public_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    public = repo / "docs/public"
    (public / "showcase").mkdir(parents=True)
    (public / "index.mdx").write_text("public")
    (public / "site_map.json").write_text("{}")
    (public / "showcase/showcase.json").write_text('{"authored": true}\n')
    tracked = frozenset(path for path in public.rglob('*') if path.is_file())
    monkeypatch.setattr('export_public_docs_to_gitops.get_tracked_files', lambda *args: tracked)
    return repo


def test_export_takes_the_showcase_catalog_from_the_bundle_when_one_is_given(tmp_path, monkeypatch):
    repo = _public_repo(tmp_path, monkeypatch)
    root = fixture_bundle(tmp_path)
    files = root / "files"
    (files / "showcase").mkdir()
    (files / "showcase/showcase.json").write_text('{"bundled": true}\n')
    manifest = json.loads((files / "docs-manifest.json").read_text())
    manifest["files"]["showcase/showcase.json"] = bundle.digest(files / "showcase/showcase.json")
    bundle.write_json(files / "docs-manifest.json", manifest)
    # The fixture repository has no commit to pin, so the source-identity half
    # of validation is skipped; the bundle's own checksums are still verified.
    manifest = bundle.validate_bundle(root)
    monkeypatch.setattr(bundle, "validate_bundle", lambda *args: manifest)
    destination = tmp_path / "snapshot/rfx"
    export_snapshot(repo, destination, root)
    assert (destination / "showcase/showcase.json").read_text() == '{"bundled": true}\n'
    assert not (destination / "site_map.json").exists()
    source_only = tmp_path / "source-only/rfx"
    export_snapshot(repo, source_only)
    assert (source_only / "showcase/showcase.json").read_text() == '{"authored": true}\n'


def test_raw_anchors_keep_their_own_text_and_urls():
    source = ('<a href="/rfx/showcase/">See the showcase →</a>'
              '<a href="https://github.com/bk-squared/rfx">Repository</a>'
              '<a href="../guide/"><strong>Read</strong> the <em>guide</em></a>')
    _, markdown = bundle.clean_markdown(
        source, route="examples/demo", base_url="https://remilab.ai/rfx", source_sha="a" * 40,
    )
    assert markdown == (
        '[See the showcase →](https://remilab.ai/rfx/showcase/)'
        '[Repository](https://github.com/bk-squared/rfx)'
        '[Read the guide](https://remilab.ai/rfx/examples/guide/)\n'
    )


def test_linked_images_and_attribute_order_keep_their_urls():
    # Gallery cards wrap an <img> in an <a>: the link keeps the image's alt and
    # source; href is matched case-insensitively and not inside data-href.
    source = ('<a href="../gallery/boundary/"><img src="../assets/b.png" alt="Boundary reflection"/></a>'
              '<a data-href="/wrong/" HREF="/rfx/right/">Right</a>')
    _, markdown = bundle.clean_markdown(
        source, route="examples/demo", base_url="https://remilab.ai/rfx", source_sha="a" * 40,
    )
    assert "[![Boundary reflection](" in markdown
    assert "b.png)](https://remilab.ai/rfx/examples/gallery/boundary/)" in markdown
    assert "[Right](https://remilab.ai/rfx/right/)" in markdown
    assert "wrong" not in markdown


def test_agent_publication_is_explicit_and_private_paths_stay_blocked(monkeypatch, tmp_path):
    sources = {tmp_path / 'docs/agent' / name for name in bundle.PUBLIC_AGENT_PAGES}
    private = {tmp_path / 'docs/agent' / name for name in (
        'working-on-rfx.mdx', 'agent-runbook.mdx', 'repo-map.mdx',
        'recipe-waveguide-sparams.mdx', 'gpu-throughput.mdx', 'new-private-page.mdx')}
    nested = tmp_path / 'docs/agent/internal/overview.mdx'
    monkeypatch.setattr(bundle, 'get_tracked_files', lambda *args: sources | private | {nested})
    assert set(bundle.source_inputs(tmp_path)) == sources
    for source in sources:
        assert bundle.allowed_artifact(f'markdown/agent/{source.stem}.md')
    for source in private:
        with pytest.raises(ValueError, match='excluded publication path'):
            bundle.allowed_artifact(f'markdown/agent/{source.stem}.md')


def test_agent_index_has_descriptions_and_omits_manual_pages():
    pages = [dict(route=route, title=route, markdown_url=f'https://example/{route}.md',
                  description=f'About {route}')
             for route in ('guide/start', 'agent/auto-config', 'agent/overview')]
    lines = bundle.agent_index(pages)
    assert lines[0] == '## For coding agents'
    assert 'agent/overview' in lines[2]
    assert 'About agent/overview' in lines[2]
    assert not any('guide/start' in line for line in lines)


def test_agent_links_resolve_to_published_alternates_and_pinned_guides():
    _, markdown = bundle.clean_markdown(
        '[Ports](./port-selection) [Design](./design-workflows.mdx#sweep-template) '
        '[Contract](../guides/support_matrix.md)', route='agent/overview',
        base_url='https://remilab.ai/rfx/versions/v2.0.0', source_sha='a' * 40,
    )
    assert '(https://remilab.ai/rfx/versions/v2.0.0/markdown/agent/port-selection.md)' in markdown
    assert '(https://remilab.ai/rfx/versions/v2.0.0/markdown/agent/design-workflows.md#sweep-template)' in markdown
    assert f'(https://github.com/bk-squared/rfx/blob/{"a" * 40}/docs/guides/support_matrix.md)' in markdown
    with pytest.raises(ValueError, match='excluded source'):
        bundle.clean_markdown('[Private](./working-on-rfx.mdx)', route='agent/overview',
                              base_url='https://remilab.ai/rfx', source_sha='a' * 40)


def test_build_indexes_only_allowlisted_agents_and_combines_the_same_pages(tmp_path, monkeypatch):
    import types
    import check_api_reference

    root = tmp_path / 'source'
    public = root / 'docs/public'
    agents = root / 'docs/agent'
    public.mkdir(parents=True)
    agents.mkdir(parents=True)
    manual = public / 'index.mdx'
    manual.write_text('---\ntitle: Manual\ndescription: Start here.\n---\nManual body.')
    agent = agents / 'overview.mdx'
    agent.write_text('---\ntitle: Agent overview\ndescription: Plan a simulation.\n---\nAgent body.')
    private = agents / 'working-on-rfx.mdx'
    private.write_text('Internal coordination must not publish.')
    (public / 'site_map.json').write_text('{"groups": []}')
    for name in bundle.SUPPORT_FILES:
        path = root / 'docs/guides' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('{}')
    monkeypatch.setattr(bundle, 'check_source', lambda *args: 'a' * 40)
    monkeypatch.setattr(bundle, 'get_tracked_files', lambda *args: {manual, agent, private})
    monkeypatch.setattr(bundle, 'toolchain', lambda: {})
    monkeypatch.setitem(sys.modules, 'pdoc', types.SimpleNamespace(__version__='fixture'))
    monkeypatch.setattr(bundle, 'api_inventory', lambda *args: {'package_version': 'fixture', 'symbols': []})
    monkeypatch.setattr(check_api_reference, 'check_html', lambda *args: [])
    monkeypatch.setattr(check_api_reference, 'build_inventory', lambda: {})

    def render(command, **kwargs):
        target = Path(command[command.index('-o') + 1])
        target.mkdir(parents=True)
        (target / 'index.html').write_text('<html>Index</html>')
        (target / 'rfx.html').write_text('<main class="pdoc">API</main>')

    monkeypatch.setattr(bundle.subprocess, 'run', render)
    output = tmp_path / 'bundle'
    manifest = bundle.build(root, output, 'https://remilab.ai/rfx', 'development', None)
    assert {p['route'] for p in manifest['pages']} == {'', 'agent/overview'}
    files = output / 'files'
    index = (files / 'llms.txt').read_text()
    assert index.index('## For coding agents') < index.index('## Manual and examples')
    assert index.count('[Agent overview]') == 1
    assert 'Plan a simulation.' in index
    full = (files / 'llms-full.txt').read_text()
    for page in manifest['pages']:
        path = page['markdown_url'].removeprefix('https://remilab.ai/rfx/')
        assert (files / path).read_text() in full
    assert 'Internal coordination' not in full
    bundle.validate_bundle(output)
