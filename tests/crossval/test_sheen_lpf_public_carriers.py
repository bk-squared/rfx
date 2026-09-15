"""cv07 Sheen LPF -- public-carrier / docstring number correctness (issue #729, item D1).

The #931 regeneration updated the leg and referee while earlier carrier tests
still required pre-regeneration prose. Carrier expectations below are derived
from the committed artifacts at each carrier's displayed precision. This test locks:

  1. The fixture's own producer formula reproduces the committed
     ``structure_distance_pct`` field (guards against re-deriving it wrong).
  2. ``structure_distance_pct`` and ``argmin_first_null.distances_pct`` are
     NOT interchangeable -- a future "correction" must not conflate them
     (that conflation is what sank an earlier #729 proposal).
  3. The Re(Z0) windows a corrected carrier may quote are pinned exactly, so
     a future edit cannot silently re-quote the wrong window.
  4. Each public carrier now quotes the fixture-derived numbers, and no
     longer carries the stale pre-regeneration figures outside of text that
     is explicitly framed as history.

Fail-closed by design: every regex assert requires a match, so a carrier
rewording that drops the number entirely fails this test rather than
silently passing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json"
RFX_LEG_PATH = REPO_ROOT / "validation/crossval/_07_sheen_results/rfx.json"


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def rfx_leg():
    return json.loads(RFX_LEG_PATH.read_text())


def test_structure_distance_reproduces_from_committed_doublets(fixture):
    ref = fixture["referee"]
    p_lo = ref["palace_doublet_mid_ghz"]["lower"]
    p_hi = ref["palace_doublet_mid_ghz"]["upper"]
    for tag in ("rfx", "openems"):
        fd = ref["fdtd_doublet_ghz"][tag]
        e_lo = abs(fd["lower_ghz"] - p_lo) / p_lo * 100.0
        e_hi = abs(fd["upper_ghz"] - p_hi) / p_hi * 100.0
        expect = round(max(e_lo, e_hi), 4)
        assert expect == pytest.approx(ref["structure_distance_pct"][tag], abs=1e-3)
    assert ref["structure_distance_pct"]["rfx"] == pytest.approx(2.8668, abs=1e-3)
    assert ref["structure_distance_pct"]["openems"] == pytest.approx(0.6644, abs=1e-3)


def test_structure_distance_and_argmin_distance_are_not_interchangeable(fixture):
    ref = fixture["referee"]
    struct_rfx = ref["structure_distance_pct"]["rfx"]
    argmin_rfx = ref["argmin_first_null"]["distances_pct"]["rfx"]
    # These are DIFFERENT quantities (currently 2.8668% and 2.3756%) computed
    # from different windows; a carrier correction must not swap one in for
    # the other believing they are the same number.
    assert struct_rfx != argmin_rfx
    assert abs(struct_rfx - argmin_rfx) > 0.02


def test_leg_derived_re_z0_windows_pinned(rfx_leg):
    freq_ghz = np.asarray(rfx_leg["freqs_hz"]) / 1e9
    re_z0 = np.asarray(rfx_leg["re_z0"])
    passband = (freq_ghz >= 0.5) & (freq_ghz <= 3.0)
    inband = (freq_ghz >= 5.0) & (freq_ghz <= 15.0)
    assert int(passband.sum()) == 16
    assert int(inband.sum()) == 61
    assert float(np.median(re_z0[passband])) == pytest.approx(51.91227, abs=1e-3)
    assert float(np.median(re_z0[inband])) == pytest.approx(54.73019, abs=1e-3)


def test_leg_derived_passivity_and_column_power(rfx_leg):
    corr = np.asarray(rfx_leg["passivity_correction"])
    # #931: three bins, ALL above 17 GHz and none inside the 5-15 GHz null
    # band the case makes its structure statement in (the script's
    # rfx_corr_bins_in_null_band stays 0 and gate D5 is untouched). A
    # narrower strip in cells on the same dx = 200 um mesh is coarser, and
    # the top of the 20 GHz band was already artifact-class here. Reported
    # as a finding by the case itself, not smoothed over.
    assert int((corr > 0.05).sum()) == 3
    assert float(np.asarray(rfx_leg["freqs_hz"])[corr > 0.05].min()) > 17e9
    assert float(corr.max()) == pytest.approx(0.6571609, abs=1e-6)
    s11 = np.asarray(rfx_leg["s11_mag"])
    s21 = np.asarray(rfx_leg["s21_mag"])
    col_power = s11**2 + s21**2
    assert float(col_power.max()) == pytest.approx(0.999513, abs=1e-5)


def _must_find(pattern: str, text: str, where: str):
    m = re.search(pattern, text)
    assert m is not None, f"expected pattern {pattern!r} not found in {where}"
    return m


def _must_not_find(pattern: str, text: str, where: str):
    m = re.search(pattern, text)
    assert m is None, f"stale pattern {pattern!r} still present (unlabelled) in {where}"


@pytest.fixture(scope="module")
def quoted(fixture, rfx_leg):
    """Display values from each quantity's own source and frequency window."""
    ref = fixture["referee"]
    oems = json.loads((RFX_LEG_PATH.parent / "openems.json").read_text())
    result = {}
    for tag, leg in (("rfx", rfx_leg), ("openems", oems)):
        freq = np.asarray(leg["freqs_hz"]) / 1e9
        null_band = (freq >= 5.0) & (freq <= 15.0)
        null_bin = np.argmin(np.asarray(leg["s21_mag"])[null_band])
        result[f"{tag}_raw"] = f"{freq[null_band][null_bin]:.3f}"
        result[f"{tag}_structure"] = f"{ref['structure_distance_pct'][tag]:.2f}"
        fdtd = ref["fdtd_doublet_ghz"][tag]
        result[f"{tag}_refined"] = f"{fdtd['argmin_first_null_ghz']:.6f}"
        result[f"{tag}_lower"] = f"{fdtd['lower_ghz']:.3f}"
        result[f"{tag}_upper"] = f"{fdtd['upper_ghz']:.3f}"
    freq = np.asarray(rfx_leg["freqs_hz"]) / 1e9
    z0 = np.asarray(rfx_leg["re_z0"])
    for name, lo, hi in (("passband", 0.5, 3.0), ("inband", 5.0, 15.0)):
        median = np.median(z0[(freq >= lo) & (freq <= hi)])
        result[name] = f"{median:.1f}"
        result[f"{name}_precise"] = f"{median:.2f}"
    corr = np.asarray(rfx_leg["passivity_correction"])
    result["footprint"] = f"{np.count_nonzero(corr > 0.05)}/{len(corr)}"
    result["worst"] = f"{corr.max():.4f}"
    result["worst_freq"] = f"{freq[corr.argmax()]:.3f}"
    power = np.asarray(rfx_leg["s11_mag"]) ** 2 + np.asarray(rfx_leg["s21_mag"]) ** 2
    result["power"] = f"{power.max():.4f}"
    result["argmin_distance"] = f"{ref['argmin_first_null']['distances_pct']['rfx']:.4f}"
    return result


def _literal(value):
    return re.escape(value)


def _structure_pattern(quoted):
    return (r"structure distance " + _literal(quoted["rfx_structure"])
            + r"% vs OpenEMS " + _literal(quoted["openems_structure"]))


def _raw_pattern(quoted, *, argmin=False):
    label = " argmin" if argmin else ""
    return (rf"rfx{label}\s*" + _literal(quoted["rfx_raw"])
            + rf" GHz, openEMS{label} " + _literal(quoted["openems_raw"]) + " GHz")


def _assert_public_leg_values(text, path, quoted):
    _must_find(_literal(quoted["passband"]) + r" .*" + _literal(quoted["inband"]), text, path)
    _must_find(r"all three (?:are )?above 17 GHz", text, path)
    _must_find(r"none inside the 5[–-]15 GHz null band", text, path)
    _must_find(r"coarser realized strip at the same dx = 200 µm", text, path)
    _must_find(_literal(quoted["rfx_refined"] + " / " + quoted["openems_refined"]) + " GHz", text, path)
    _must_find(_literal(quoted["argmin_distance"]) + "%", text, path)
    _must_find("raw-bin rfx argmin " + _literal(quoted["rfx_raw"])
               + " GHz vs OpenEMS " + _literal(quoted["openems_raw"]) + " GHz", text, path)


def test_validation_readme_carrier_corrected(quoted):
    path = "validation/README.md"
    text = (REPO_ROOT / path).read_text()
    _must_find(_structure_pattern(quoted), text, path)
    _must_find(_literal(quoted["footprint"] + ", worst " + quoted["worst"]
                        + " at " + quoted["worst_freq"] + " GHz"), text, path)
    _must_not_find(r"structure distance 1\.91%", text, path)
    _must_not_find(r"reads ~67 ", text, path)
    _assert_public_leg_values(text, path, quoted)


def test_benchmarks_mdx_carrier_corrected(quoted):
    path = "docs/public/guide/benchmarks.mdx"
    text = (REPO_ROOT / path).read_text()
    _must_find(_structure_pattern(quoted), text, path)
    _must_find("max column power " + _literal(quoted["power"]), text, path)
    _must_find(_literal(quoted["footprint"] + " bins > 0.05 on the current leg, worst "
                        + quoted["worst"] + " at " + quoted["worst_freq"] + " GHz"), text, path)
    _must_not_find(r"max column power 0\.9938", text, path)
    _assert_public_leg_values(text, path, quoted)


def test_palace_sheen_referee_readme_carrier_corrected(quoted):
    path = "scripts/diagnostics/palace_sheen_referee/README.md"
    text = (REPO_ROOT / path).read_text()
    _must_find(_raw_pattern(quoted), text, path)
    _must_find(r"\| " + _literal(quoted["rfx_lower"]) + r"\s+\| "
               + _literal(quoted["rfx_upper"]) + r"\s+\| "
               + _literal(quoted["rfx_raw"]) + r"\s+\| \*\*"
               + _literal(quoted["rfx_structure"]) + r" %\*\*", text, path)
    _must_find(_literal(quoted["openems_refined"] + " / " + quoted["rfx_refined"]), text, path)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "7.218" in line:
            window = " ".join(lines[max(0, i - 1):i + 2])
            assert re.search(r"earlier|history|historical|figure above|regenerat", window, re.I), (
                f"unlabelled stale 7.218 in {path}: {line!r}"
            )


def test_build_referee_producer_docstring_corrected(quoted):
    path = "scripts/diagnostics/build_sheen_lpf_palace_referee.py"
    text = (REPO_ROOT / path).read_text()
    _must_find(_raw_pattern(quoted), text, path)
    _must_find("structure_distance_pct " + _literal(quoted["rfx_structure"]) + "%", text, path)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "7.218" in line:
            window = " ".join(lines[max(0, i - 1):i + 2])
            assert re.search(r"earlier|regenerat", window, re.I), (
                f"unlabelled stale 7.218 in {path}: {line!r}"
            )


def test_check_sparams_runtime_print_corrected(quoted):
    path = "scripts/diagnostics/palace_sheen_referee/check_sparams.py"
    text = (REPO_ROOT / path).read_text()
    _must_find("rfx " + _literal(quoted["rfx_raw"]) + r" \| openEMS "
               + _literal(quoted["openems_raw"]) + " GHz", text, path)


def test_mesh_sheen_docstring_corrected(quoted):
    path = "scripts/diagnostics/palace_sheen_referee/mesh_sheen.py"
    text = (REPO_ROOT / path).read_text()
    _must_find(_raw_pattern(quoted), text, path)
    for line in text.splitlines():
        if "7.218" in line:
            assert re.search(r"earlier", line, re.I), (
                f"unlabelled stale 7.218 in {path}: {line!r}"
            )


def test_gate_test_docstring_corrected(quoted):
    path = "tests/crossval/test_sheen_lpf_palace_referee_gates.py"
    text = (REPO_ROOT / path).read_text()
    _must_find(_raw_pattern(quoted, argmin=True), text, path)
    _must_find("structure_distance_pct: openEMS " + _literal(quoted["openems_structure"])
               + "%, rfx " + _literal(quoted["rfx_structure"]) + "%", text, path)


def test_case_docstring_current_evidence_matches_artifacts(quoted, fixture):
    path = "validation/crossval/07_sheen_lpf.py"
    text = (REPO_ROOT / path).read_text()
    current = text.split("CURRENT DOCS TRUTH", 1)[1].split("#931 LATTICE OWNERSHIP", 1)[0]
    _must_find("structure_distance_pct.rfx="
               + _literal(f"{fixture['referee']['structure_distance_pct']['rfx']:.4f}"), current, path)
    _must_find("argmin_first_null.distances_pct.rfx=" + _literal(quoted["argmin_distance"]), current, path)
    _must_find(_literal(quoted["footprint"]) + " passivity_correction bins > 0.05", current, path)
    _must_find("worst " + _literal(quoted["worst"] + " at " + quoted["worst_freq"] + " GHz"), current, path)
    _must_find(_literal("Passband median Re(Z0)=" + quoted["passband_precise"] + " ohm"), current, path)
    _must_find(_literal("in-band median Re(Z0)=" + quoted["inband_precise"] + " ohm"), current, path)


def test_estimator_falsifier_carriers_label_pre_regeneration_evidence():
    for path in ("validation/README.md", "docs/public/guide/benchmarks.mdx",
                 "scripts/diagnostics/cv07_estimator_falsifiers.py"):
        text = (REPO_ROOT / path).read_text()
        _must_find(r"pre-#931 (?:leg|calibration)", text, path)
    tool = (REPO_ROOT / "scripts/diagnostics/cv07_estimator_falsifiers.py").read_text()
    _must_find(r"claimed isolated defects require a separate\npost-#931 remeasurement", tool,
               "scripts/diagnostics/cv07_estimator_falsifiers.py")


def test_unremeasured_extractor_prose_labels_historical_legs():
    estimators = (REPO_ROOT / "validation/crossval/comparators/spectral_features.py").read_text()
    assert "pre-#931 cv07 legs" in estimators
    assert "<= 0.02 % (historical evidence" in estimators
    guide = (REPO_ROOT / "docs/public/guide/probes-sparams.mdx").read_text()
    assert "historical pre-#931 filter geometries" in guide
    assert "old runs, not the regenerated #931 legs" in guide
