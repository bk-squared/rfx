"""A WR-90 magnitude table cannot certify an unrecorded extractor.

The WR-90 waveguide-port case that produced the stdout was removed on
2026-09-21. What is pinned here is the SURVIVING half of that chain: the
committed broad-E4 comparison artifact under tests/fixtures/waveguide_broad_e5/
and the builder that writes it, which refuses a stdout whose extractor record
is invalid or self-contradictory. The one arm that drove the removed case's
own ``run_rfx_*`` entry points went with the case; nothing else moved.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
STDOUT = ROOT / "tests/fixtures/waveguide_broad_e5/cv11_current_main_8206031d_stdout.txt"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/diagnostics"))
    return _load("cv11_evidence_builder", ROOT / "scripts/diagnostics/build_waveguide_wr90_rectangular_broad_e4_comparison.py")


def test_historical_tables_do_not_imply_flux_or_ad_coverage(monkeypatch, tmp_path):
    payload = _builder(monkeypatch).build_rectangular_broad_e4_comparison(
        STDOUT, tmp_path, reference_column="Palace_r_h2")
    assert set(payload["normalization_by_geometry"].values()) == {None}
    assert "normalize='flux'" not in payload["claim"]
    current = json.loads(STDOUT.with_name("wr90_rectangular_broad_e4_comparison.json").read_text())
    assert payload["pairs"] == current["pairs"]
    assert payload["summary"] == current["summary"]


def test_retrospective_source_audit_is_bound_to_the_retained_run():
    current = json.loads(STDOUT.with_name("wr90_rectangular_broad_e4_comparison.json").read_text())
    audit = current["provenance"]["normalization_source_audit_2026_09_13"]
    source = ROOT / current["source_cv11_stdout"]
    assert audit["producer_commit"] == current["setup"]["commit"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == audit["source_cv11_stdout_sha256"]
    # These are retrospectively audited values, not recorded stdout fields.
    assert audit["normalization_by_geometry"] == {"empty": True, "pec_short": False, "slab": True}
    assert set(current["normalization_by_geometry"].values()) == {None}


@pytest.mark.parametrize("mode", [0, 1, "False", None])
def test_invalid_extractor_metadata_is_rejected(monkeypatch, tmp_path, mode):
    output = tmp_path / "invalid.stdout"
    output.write_text("CV11_EXTRACTION_MODE " + json.dumps({
        "geometry": "slab", "normalize": mode}) + "\n" + STDOUT.read_text())
    with pytest.raises(ValueError, match="invalid cv11 extraction record"):
        _builder(monkeypatch).build_rectangular_broad_e4_comparison(output, tmp_path)


def test_mixed_run_extractor_records_are_rejected(monkeypatch, tmp_path):
    output = tmp_path / "mixed.stdout"
    output.write_text('CV11_EXTRACTION_MODE {"geometry":"slab","normalize":true}\n'
                      'CV11_EXTRACTION_MODE {"geometry":"slab","normalize":"flux"}\n'
                      + STDOUT.read_text())
    with pytest.raises(ValueError, match="duplicate cv11 extraction record"):
        _builder(monkeypatch).build_rectangular_broad_e4_comparison(output, tmp_path)
