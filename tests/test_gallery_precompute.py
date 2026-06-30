"""Tests for the precomputed gallery artifact pipeline.

Fast (default) tests exercise the pure ``build_manifest`` helper and the case
registry integrity without running any simulation. The single ``@pytest.mark.slow``
test runs ``--quick`` for the fastest case and asserts a real bundle is emitted.
"""

from __future__ import annotations

import hashlib
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
# Schema v2 — gradient_validation + application/capability taxonomy
# ---------------------------------------------------------------------------


def test_schema_version_is_v2():
    assert pg.MANIFEST_SCHEMA_VERSION == "rfx-gallery-manifest-v2"
    m = pg.build_manifest(
        _sample_case(), assets=_sample_assets(), runtime_seconds=1.0,
        passed=True, metric_value="x", params={}, quick_smoke=False,
    )
    assert m["schema_version"] == "rfx-gallery-manifest-v2"


def test_application_and_capability_are_lists():
    for case in pg.CASE_REGISTRY:
        m = pg.build_manifest(
            case, assets=_sample_assets(), runtime_seconds=1.0,
            passed=True, metric_value="x", params={}, quick_smoke=False,
        )
        assert isinstance(m["application"], str) and m["application"]
        assert isinstance(m["capability"], list) and m["capability"]


def test_inverse_design_capability_only_on_ar_coating():
    for case in pg.CASE_REGISTRY:
        m = pg.build_manifest(
            case, assets=_sample_assets(), runtime_seconds=1.0,
            passed=True, metric_value="x", params={}, quick_smoke=False,
        )
        if case.id == "ar_coating_design":
            assert "inverse-design" in m["capability"]
            assert m["application"] == "inverse-design"
        else:
            assert "inverse-design" not in m["capability"]


def test_autodiff_gradient_capability_on_three_sparam_cases():
    for cid in ("multilayer_fresnel", "waveguide_wr90", "patch_antenna"):
        case = pg.CASE_BY_ID[cid]
        m = pg.build_manifest(
            case, assets=_sample_assets(), runtime_seconds=1.0,
            passed=True, metric_value="x", params={}, quick_smoke=False,
        )
        assert "autodiff-gradient" in m["capability"], cid


def test_gradient_validation_threaded_when_provided():
    grad = {
        "param": "slab_thickness_d",
        "ad_value": -1.0,
        "fd_value": -1.0,
        "fd_step_h": 0.05e-3,
        "rel_err_vs_fd": 2e-5,
        "rel_err_vs_analytic": 0.026,
        "sign_agreement": True,
        "gate_threshold": 0.05,
    }
    case = pg.CASE_BY_ID["multilayer_fresnel"]
    m = pg.build_manifest(
        case, assets=_sample_assets(), runtime_seconds=1.0,
        passed=True, metric_value="x", params={}, quick_smoke=False,
        gradient_validation=grad,
    )
    gv = m["gradient_validation"]
    for key in ("param", "ad_value", "fd_value", "fd_step_h", "rel_err_vs_fd",
                "rel_err_vs_analytic", "sign_agreement", "gate_threshold"):
        assert key in gv, f"gradient_validation missing {key}"
    # slab keeps BOTH numbers distinct: AD-vs-FD machinery check and the
    # AD-vs-analytic mesh-limited cross-check.
    assert gv["rel_err_vs_fd"] != gv["rel_err_vs_analytic"]
    assert gv["gate_threshold"] == 0.05  # never the MSL 0.10


def test_gradient_validation_absent_when_not_provided():
    case = pg.CASE_BY_ID["multilayer_fresnel"]
    m = pg.build_manifest(
        case, assets=_sample_assets(), runtime_seconds=1.0,
        passed=True, metric_value="x", params={}, quick_smoke=False,
    )
    assert "gradient_validation" not in m


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
    assert "field-animation-gif" in asset_types, "a field animation must be emitted"
    geom = next(a for a in manifest["assets"] if a["type"] == "geometry-png")
    assert geom["filename"] == "geometry.png"
    anim = next(a for a in manifest["assets"] if a["type"] == "field-animation-gif")
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
        # the time-domain final-design E-field GIF and the design+field
        # co-evolution GIF are distinct assets with distinct types (T4.2)
        "field-animation-gif",
        "design-field-coevolution-gif",
    } <= asset_types, f"optimization assets missing: got {asset_types}"
    # The optimization case carries no S-matrix / Touchstone / Smith.
    assert "touchstone" not in asset_types
    assert "smith-png" not in asset_types

    # The two GIFs are distinct files (fields.gif must not be clobbered).
    by_type = {a["type"]: a["filename"] for a in manifest["assets"]}
    assert by_type["field-animation-gif"] == "fields.gif"
    assert by_type["design-field-coevolution-gif"] == "design_field_coevolution.gif"
    assert by_type["field-animation-gif"] != by_type["design-field-coevolution-gif"]

    # All AR asset files exist on disk (full AR bundle).
    for asset in manifest["assets"]:
        assert (case_dir / asset["filename"]).exists()

    # The co-evolution GIF has >= 2 frames (PIL seek(1) succeeds).
    from PIL import Image
    im = Image.open(case_dir / "design_field_coevolution.gif")
    im.seek(1)

    opt = json.loads((case_dir / "optimization.json").read_text())
    assert opt["columns"][:2] == ["iter", "cost"]
    assert opt["rows"], "optimization.json must have at least one row"


# ---------------------------------------------------------------------------
# Reconcile pass — pure logic (no simulation required)
# ---------------------------------------------------------------------------

_RECONCILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reconcile_gallery_manifests.py"
)
_rspec = importlib.util.spec_from_file_location("reconcile_gallery_manifests", _RECONCILE)
assert _rspec is not None and _rspec.loader is not None
rc = importlib.util.module_from_spec(_rspec)
sys.modules[_rspec.name] = rc
_rspec.loader.exec_module(rc)


def test_reconcile_canonical_vocab_defined():
    """All four known case ids have a canonical map with at least 3 filenames."""
    for cid in ("multilayer_fresnel", "patch_antenna", "waveguide_wr90", "ar_coating_design"):
        assert cid in rc.CANONICAL, f"{cid} not in CANONICAL"
        assert len(rc.CANONICAL[cid]) >= 3, f"{cid} canonical map too short"


def test_reconcile_atomic_write(tmp_path):
    """_atomic_write_json writes correct content and leaves no temp file behind."""
    p = tmp_path / "test.json"
    data = {"key": "value", "num": 42}
    rc._atomic_write_json(p, data)
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded == data
    # No leftover .tmp files.
    tmps = list(tmp_path.glob("*.tmp"))
    assert not tmps, f"leftover temp files: {tmps}"


def _make_minimal_manifest(case_dir: Path, case_id: str, assets_filenames: list[str]) -> None:
    """Write a minimal manifest.json and create stub asset files."""
    case_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for fn in assets_filenames:
        p = case_dir / fn
        p.write_bytes(b"stub content for " + fn.encode())
        assets.append({
            "filename": fn,
            "type": "stale-type",
            "sha256": "0" * 64,  # deliberately stale
            "size_bytes": 0,
        })
    man = {
        "schema_version": "rfx-gallery-manifest-v2",
        "case_id": case_id,
        "validation": {"tier": "E5", "passed": True, "metric": "x",
                       "metric_value": "x", "tolerance": "x",
                       "reference_solver": "x"},
        "application": "test",
        "capability": ["test"],
        "assets": assets,
    }
    (case_dir / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


def test_reconcile_case_rebuilds_sha256(tmp_path):
    """After reconcile, every asset sha256 in the manifest matches the on-disk file."""
    case_dir = tmp_path / "multilayer_fresnel"
    # Write canonical files that the reconcile pass should include.
    canonical_files = ["sparams.json", "geometry.png", "rt_overlay.png", "field_anim.gif",
                       "autodiff.png", "gradient.json"]
    _make_minimal_manifest(case_dir, "multilayer_fresnel", canonical_files)

    man = rc.reconcile_case(case_dir, case_id="multilayer_fresnel")

    # Every asset entry in the result must have correct sha256.
    for asset in man["assets"]:
        path = case_dir / asset["filename"]
        assert path.exists(), f"{asset['filename']} missing after reconcile"
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert asset["sha256"] == actual_sha, (
            f"{asset['filename']}: manifest sha256 {asset['sha256'][:16]}… "
            f"!= actual {actual_sha[:16]}…"
        )


def test_reconcile_case_applies_canonical_types(tmp_path):
    """After reconcile, every asset type matches the canonical vocabulary."""
    case_dir = tmp_path / "waveguide_wr90"
    canonical_files = ["sparams.json", "sparams.s2p", "geometry.png", "validation.png",
                       "autodiff.png", "field_anim.gif"]
    _make_minimal_manifest(case_dir, "waveguide_wr90", canonical_files)

    man = rc.reconcile_case(case_dir, case_id="waveguide_wr90")

    by_name = {a["filename"]: a["type"] for a in man["assets"]}
    assert by_name.get("sparams.json") == "sparams-json"
    assert by_name.get("sparams.s2p") == "touchstone"
    assert by_name.get("geometry.png") == "geometry-png"
    assert by_name.get("validation.png") == "validation-png"
    assert by_name.get("autodiff.png") == "autodiff-png"
    assert by_name.get("field_anim.gif") == "field-animation-gif"


def test_reconcile_case_drops_absent_files(tmp_path):
    """Assets listed in the manifest but absent on disk are excluded."""
    case_dir = tmp_path / "patch_antenna"
    # Only write some canonical files; the rest are intentionally absent.
    present = ["sparams.json", "geometry.png"]
    _make_minimal_manifest(case_dir, "patch_antenna",
                           present + ["s11_db.png"])  # s11_db.png listed but NOT created
    (case_dir / "s11_db.png").unlink(missing_ok=True)  # ensure it's gone

    man = rc.reconcile_case(case_dir, case_id="patch_antenna")

    filenames_in_manifest = {a["filename"] for a in man["assets"]}
    assert "s11_db.png" not in filenames_in_manifest, \
        "absent file must not appear in reconciled manifest"
    assert "sparams.json" in filenames_in_manifest
    assert "geometry.png" in filenames_in_manifest


def test_reconcile_case_preserves_non_assets_blocks(tmp_path):
    """Non-assets blocks (validation, gradient_validation, etc.) are preserved."""
    case_dir = tmp_path / "multilayer_fresnel"
    _make_minimal_manifest(case_dir, "multilayer_fresnel", ["sparams.json"])
    # Inject extra blocks into the manifest.
    man_path = case_dir / "manifest.json"
    man = json.loads(man_path.read_text())
    man["gradient_validation"] = {"param": "slab_thickness_d", "gate_threshold": 0.05}
    man["custom_block"] = {"preserved": True}
    man_path.write_text(json.dumps(man, indent=2) + "\n")

    result = rc.reconcile_case(case_dir, case_id="multilayer_fresnel")

    assert "gradient_validation" in result
    assert result["gradient_validation"]["gate_threshold"] == 0.05
    assert result.get("custom_block", {}).get("preserved") is True


def test_reconcile_case_renames_fields_gif_for_sparam_cases(tmp_path):
    """fields.gif is renamed to field_anim.gif for S-param cases if field_anim.gif is absent."""
    for cid in ("multilayer_fresnel", "patch_antenna", "waveguide_wr90"):
        case_dir = tmp_path / cid
        # Write fields.gif (precompute output) but NOT field_anim.gif.
        _make_minimal_manifest(case_dir, cid, ["sparams.json", "fields.gif"])

        man = rc.reconcile_case(case_dir, case_id=cid)

        assert (case_dir / "field_anim.gif").exists(), \
            f"[{cid}] fields.gif should be renamed to field_anim.gif"
        assert not (case_dir / "fields.gif").exists(), \
            f"[{cid}] fields.gif should no longer exist after rename"
        filenames = {a["filename"] for a in man["assets"]}
        assert "field_anim.gif" in filenames
        assert "fields.gif" not in filenames


def test_reconcile_ar_keeps_coevolution_drops_raw_fields_gif(tmp_path):
    """For ar_coating_design the design+field co-evolution GIF is canonical, but the
    raw time-domain fields.gif is NOT registered: the AR domain is 1-D (single cell
    thick transversely) so fields.gif renders as a ~1px unreadable strip."""
    case_dir = tmp_path / "ar_coating_design"
    _make_minimal_manifest(case_dir, "ar_coating_design",
                           ["fields.gif", "design_field_coevolution.gif", "optimization.json"])

    man = rc.reconcile_case(case_dir, case_id="ar_coating_design")

    by_name = {a["filename"]: a["type"] for a in man["assets"]}
    assert "design_field_coevolution.gif" in by_name
    assert by_name["design_field_coevolution.gif"] == "design-field-coevolution-gif"
    assert "fields.gif" not in by_name, "raw 1-D fields.gif must NOT be registered for AR"


def test_verify_sha256_passes_after_reconcile(tmp_path):
    """verify_sha256 returns True when run immediately after reconcile."""
    case_dir = tmp_path / "waveguide_wr90"
    _make_minimal_manifest(case_dir, "waveguide_wr90", ["sparams.json", "validation.png"])

    rc.reconcile_case(case_dir, case_id="waveguide_wr90")
    ok = rc.verify_sha256(tmp_path, case_id="waveguide_wr90")
    assert ok, "verify_sha256 should pass immediately after reconcile"


@pytest.mark.slow
def test_reconcile_after_quick_smoke(tmp_path):
    """Full pipeline: quick smoke -> reconcile -> verify_sha256 all pass."""
    rc_code = pg.main(["--case", "multilayer_fresnel", "--quick", "--out", str(tmp_path)])
    assert rc_code == 0

    case_dir = tmp_path / "multilayer_fresnel"
    man = rc.reconcile_case(case_dir, case_id="multilayer_fresnel")

    # All types must be canonical.
    for asset in man["assets"]:
        expected = rc.CANONICAL.get("multilayer_fresnel", {}).get(asset["filename"])
        if expected is not None:
            assert asset["type"] == expected, (
                f"{asset['filename']}: type {asset['type']!r} != canonical {expected!r}"
            )

    # sha256 verification must pass.
    ok = rc.verify_sha256(tmp_path, case_id="multilayer_fresnel")
    assert ok, "sha256 verification failed after reconcile"

    # Committed assets tree unchanged.
    import subprocess
    result = subprocess.run(
        ["git", "status", "--short",
         "docs/public/gallery/assets/"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", \
        f"committed assets tree modified: {result.stdout}"
