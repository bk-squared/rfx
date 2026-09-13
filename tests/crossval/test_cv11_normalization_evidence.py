"""A cv11 magnitude table cannot certify an unrecorded extractor."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_actual_producer_modes_reach_the_artifact(monkeypatch, capsys, tmp_path):
    producer = _load("cv11_modes", ROOT / "validation/crossval/11_waveguide_port_wr90.py")
    calls = []

    def solve(**kwargs):
        calls.append(kwargs["normalize"])
        return SimpleNamespace(s_params=np.zeros((2, 2, 1)),
                               port_names=["left", "right"], freqs=np.array([1.]))

    monkeypatch.setattr(producer, "_build_sim", lambda *a, **kw: SimpleNamespace(
        compute_waveguide_s_matrix=solve))
    monkeypatch.setattr(producer, "assert_realized_short", lambda sim: {
        "planes_m": [0.145, 0.147], "n_cells": 1, "n_sheets": 0})
    producer.run_rfx_empty()
    producer.run_rfx_pec_short()
    producer.run_rfx_slab(2., 0.01)
    assert calls == [True, False, True]
    output = tmp_path / "producer.stdout"
    output.write_text(capsys.readouterr().out + STDOUT.read_text())
    payload = _builder(monkeypatch).build_rectangular_broad_e4_comparison(
        output, tmp_path / "artifact", reference_column="Palace_r_h2")
    assert payload["normalization_by_geometry"] == {"empty": True, "pec_short": False, "slab": True}
    assert "normalize='flux'" not in payload["claim"]


def test_historical_tables_do_not_imply_flux_or_ad_coverage(monkeypatch, tmp_path):
    payload = _builder(monkeypatch).build_rectangular_broad_e4_comparison(
        STDOUT, tmp_path, reference_column="Palace_r_h2")
    assert set(payload["normalization_by_geometry"].values()) == {None}
    assert "normalize='flux'" not in payload["claim"]
    current = json.loads(STDOUT.with_name("wr90_rectangular_broad_e4_comparison.json").read_text())
    assert payload["pairs"] == current["pairs"]
    assert payload["summary"] == current["summary"]


def test_mixed_run_extractor_records_are_rejected(monkeypatch, tmp_path):
    output = tmp_path / "mixed.stdout"
    output.write_text('CV11_EXTRACTION_MODE {"geometry":"slab","normalize":true}\n'
                      'CV11_EXTRACTION_MODE {"geometry":"slab","normalize":"flux"}\n'
                      + STDOUT.read_text())
    with pytest.raises(ValueError, match="duplicate cv11 extraction record"):
        _builder(monkeypatch).build_rectangular_broad_e4_comparison(output, tmp_path)
