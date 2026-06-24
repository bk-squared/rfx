"""Tests for the precomputed gallery artifact pipeline.

Fast (default) tests exercise the pure ``build_manifest`` helper and the case
registry integrity without running any simulation. The single ``@pytest.mark.slow``
test runs ``--quick`` for the fastest case and asserts a real bundle is emitted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "precompute_gallery_artifacts.py"
)
_spec = importlib.util.spec_from_file_location("precompute_gallery_artifacts", _SCRIPT)
assert _spec is not None and _spec.loader is not None
pg = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass forward-ref resolution can find the module.
sys.modules[_spec.name] = pg
_spec.loader.exec_module(pg)


# ---------------------------------------------------------------------------
# Case-registry integrity
# ---------------------------------------------------------------------------


def test_registry_ids_unique():
    ids = [c.id for c in pg.CASE_REGISTRY]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    assert set(ids) == set(pg.CASE_BY_ID), "CASE_BY_ID must mirror the registry"


def test_registry_has_three_mvp_cases():
    ids = {c.id for c in pg.CASE_REGISTRY}
    assert {"multilayer_fresnel", "waveguide_wr90", "patch_antenna"} <= ids


def test_registry_required_fields_present():
    for case in pg.CASE_REGISTRY:
        assert case.id and isinstance(case.id, str)
        assert case.title and case.description
        assert case.validation_tier in {"E4", "E5", "OPT"}
        assert case.metric and case.tolerance and case.reference_solver
        assert callable(case.builder), f"{case.id} builder must be callable"
        assert case.n_ports >= 1


def test_registry_has_optimization_case():
    opt = pg.CASE_BY_ID.get("ar_coating_design")
    assert opt is not None, "ar_coating_design optimization case must be registered"
    assert opt.is_optimization is True
    assert opt.has_smith is False
    assert opt.has_animation is True


def test_optimization_case_has_no_touchstone_tier():
    # Validation tiers in {E4, E5} carry a Touchstone; the OPT case does not.
    opt = pg.CASE_BY_ID["ar_coating_design"]
    assert opt.validation_tier == "OPT"


# ---------------------------------------------------------------------------
# build_manifest — pure helper
# ---------------------------------------------------------------------------


def _sample_case():
    return pg.CASE_BY_ID["waveguide_wr90"]


def _sample_assets():
    return [
        {"filename": "sparams.png", "type": "sparam-plot-png", "sha256": "a" * 64, "size_bytes": 100},
        {"filename": "sparams.s2p", "type": "touchstone", "sha256": "b" * 64, "size_bytes": 200},
    ]


def test_build_manifest_provenance_fields_present():
    m = pg.build_manifest(
        _sample_case(),
        assets=_sample_assets(),
        runtime_seconds=12.5,
        passed=True,
        metric_value="max|S11|=0.01",
        params={"dx_m": 0.001},
        quick_smoke=False,
        rfx_version="9.9.9",
        git_sha="deadbeef",
        hostname="vessl-node",
        backend="gpu",
        generated_at="2026-01-01T00:00:00Z",
    )
    prov = m["provenance"]
    for key in ("rfx_version", "git_sha", "params", "runtime_seconds", "hostname", "backend"):
        assert key in prov, f"provenance missing {key}"
    assert prov["rfx_version"] == "9.9.9"
    assert prov["git_sha"] == "deadbeef"
    assert prov["hostname"] == "vessl-node"
    assert prov["backend"] == "gpu"
    assert prov["runtime_seconds"] == 12.5
    assert prov["params"] == {"dx_m": 0.001}
    assert m["schema_version"] == pg.MANIFEST_SCHEMA_VERSION
    assert m["generated_at"] == "2026-01-01T00:00:00Z"


def test_build_manifest_reproducibility_status_is_provenance_only():
    m = pg.build_manifest(
        _sample_case(),
        assets=_sample_assets(),
        runtime_seconds=1.0,
        passed=True,
        metric_value="x",
        params={},
        quick_smoke=False,
    )
    assert m["reproducibility"]["status"] == "provenance-only"


def test_build_manifest_served_url_shape_and_sha256():
    case = _sample_case()
    m = pg.build_manifest(
        case,
        assets=_sample_assets(),
        runtime_seconds=1.0,
        passed=True,
        metric_value="x",
        params={},
        quick_smoke=False,
    )
    assert len(m["assets"]) == 2
    for asset in m["assets"]:
        assert asset["served_url"].startswith("/rfx/gallery/assets/")
        assert asset["served_url"] == f"/rfx/gallery/assets/{case.id}/{asset['filename']}"
        assert asset["sha256"] is not None
        assert len(asset["sha256"]) == 64


def test_build_manifest_validation_tier_and_metric():
    case = _sample_case()
    m = pg.build_manifest(
        case,
        assets=_sample_assets(),
        runtime_seconds=1.0,
        passed=True,
        metric_value="max|S11|=0.01",
        params={},
        quick_smoke=False,
    )
    val = m["validation"]
    assert val["tier"] == case.validation_tier
    assert val["tier"] in {"E4", "E5"}
    assert val["metric"] == case.metric
    assert val["tolerance"] == case.tolerance
    assert val["reference_solver"] == case.reference_solver
    assert val["metric_value"] == "max|S11|=0.01"


def test_build_manifest_quick_smoke_forces_passed_null():
    m = pg.build_manifest(
        _sample_case(),
        assets=_sample_assets(),
        runtime_seconds=1.0,
        passed=True,  # builder claimed pass...
        metric_value="x",
        params={},
        quick_smoke=True,  # ...but quick smoke must override to null
    )
    assert m["quick_smoke"] is True
    assert m["validation"]["passed"] is None
    limitations = " ".join(m["reproducibility"]["limitations"]).lower()
    assert "not validated" in limitations


def test_build_manifest_non_quick_preserves_passed():
    m_pass = pg.build_manifest(
        _sample_case(), assets=_sample_assets(), runtime_seconds=1.0,
        passed=True, metric_value="x", params={}, quick_smoke=False,
    )
    m_fail = pg.build_manifest(
        _sample_case(), assets=_sample_assets(), runtime_seconds=1.0,
        passed=False, metric_value="x", params={}, quick_smoke=False,
    )
    assert m_pass["validation"]["passed"] is True
    assert m_fail["validation"]["passed"] is False


# ---------------------------------------------------------------------------
# CLI — fail loud on unknown case id
# ---------------------------------------------------------------------------


def test_main_unknown_case_id_fails_loud():
    with pytest.raises(SystemExit) as exc:
        pg.main(["--case", "does_not_exist"])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Slow smoke: run --quick for the fastest case, assert a real bundle.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_quick_smoke_emits_bundle(tmp_path):
    rc = pg.main(["--case", "multilayer_fresnel", "--quick", "--out", str(tmp_path)])
    assert rc == 0

    case_dir = tmp_path / "multilayer_fresnel"
    manifest_path = case_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json must be emitted"

    pngs = list(case_dir.glob("*.png"))
    assert pngs, "at least one PNG must be emitted"

    touchstones = list(case_dir.glob("*.s2p"))
    assert touchstones, "a Touchstone file must be emitted"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["quick_smoke"] is True
    assert manifest["validation"]["passed"] is None
    assert manifest["reproducibility"]["status"] == "provenance-only"
    assert manifest["case_id"] == "multilayer_fresnel"
    assert manifest["assets"], "manifest must list assets"
    for asset in manifest["assets"]:
        assert (case_dir / asset["filename"]).exists()
        assert asset["served_url"].startswith("/rfx/gallery/assets/multilayer_fresnel/")

    asset_types = {a["type"] for a in manifest["assets"]}
    assert "geometry-png" in asset_types, "a geometry image must be emitted"
    assert "animation" in asset_types, "a field animation must be emitted"
    geom = next(a for a in manifest["assets"] if a["type"] == "geometry-png")
    assert geom["filename"] == "geometry.png"
    anim = next(a for a in manifest["assets"] if a["type"] == "animation")
    # Animation is .mp4 when ffmpeg is present, otherwise .gif (Pillow fallback).
    assert anim["filename"].split(".")[-1] in {"mp4", "gif"}


@pytest.mark.slow
def test_quick_smoke_optimization_case_emits_bundle(tmp_path):
    rc = pg.main(["--case", "ar_coating_design", "--quick", "--out", str(tmp_path)])
    assert rc == 0

    case_dir = tmp_path / "ar_coating_design"
    manifest = json.loads((case_dir / "manifest.json").read_text())
    assert manifest["case_id"] == "ar_coating_design"
    assert manifest["quick_smoke"] is True
    assert manifest["validation"]["passed"] is None

    asset_types = {a["type"] for a in manifest["assets"]}
    assert {
        "convergence-png",
        "design-evolution-png",
        "result-spectrum-png",
        "optimization-json",
        "geometry-png",
    } <= asset_types, f"optimization assets missing: got {asset_types}"
    # The optimization case carries no S-matrix / Touchstone / Smith.
    assert "touchstone" not in asset_types
    assert "smith-png" not in asset_types

    for asset in manifest["assets"]:
        assert (case_dir / asset["filename"]).exists()

    opt = json.loads((case_dir / "optimization.json").read_text())
    assert opt["columns"][:2] == ["iter", "cost"]
    assert opt["rows"], "optimization.json must have at least one row"
