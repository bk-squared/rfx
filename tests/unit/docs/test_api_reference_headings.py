"""Rendered API page heading checks without importing pdoc."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from check_api_reference import check_html, check_page_headings


def test_one_heading_passes():
    assert check_page_headings("<h1>rfx</h1><h2>Parameters</h2>", "rfx.html") == []


def test_two_headings_fail_with_all_text():
    errors = check_page_headings(
        "<h1>rfx</h1><h1><code>123</code> &amp; fragment</h1>", "rfx.html")
    assert len(errors) == 1
    assert "rfx.html" in errors[0]
    assert "found 2" in errors[0]
    assert "['rfx', '123 & fragment']" in errors[0]


def test_missing_heading_fails():
    assert "found 0" in check_page_headings("<h2>rfx</h2>", "empty.html")[0]


@pytest.mark.parametrize("heading", ["<p>No title</p>", "<h1>extra</h1>"])
def test_every_nested_page_is_checked(tmp_path, heading):
    (tmp_path / "rfx.html").write_text(
        "<h1>rfx</h1>add_msl_port port_s11_freqs distributed")
    nested = tmp_path / "nested"
    nested.mkdir()
    page = nested / "module.html"
    page.write_text(heading + heading)
    errors = check_html(tmp_path, {"simulation_methods": {}}, headings=True)
    assert len(errors) == 1
    assert str(page) in errors[0]


@pytest.mark.parametrize("name", ["index.html", "index-pdoc.html"])
def test_index_pages_need_no_title(tmp_path, name):
    (tmp_path / "rfx.html").write_text(
        "<h1>rfx</h1>add_msl_port port_s11_freqs distributed")
    (tmp_path / name).write_text("<nav>rfx</nav>")
    assert check_html(tmp_path, {"simulation_methods": {}}, headings=True) == []


def test_release_builds_of_old_tags_are_not_held_to_the_title_rule(tmp_path):
    (tmp_path / "rfx.html").write_text(
        "<h1>rfx</h1><h1>488 uses)</h1>add_msl_port port_s11_freqs distributed")
    assert check_html(tmp_path, {"simulation_methods": {}}) == []
