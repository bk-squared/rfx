"""cv07 Sheen LPF -- public-carrier / docstring number correctness (issue #729, item D1).

Several committed docs and diagnostic-tool docstrings quoted "first null" and
"structure distance" numbers that pre-dated the leg regeneration (PR #468/#516)
or, in one case (validation/README.md / benchmarks.mdx), never matched the
committed Palace referee fixture at all. This test locks:

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
    assert ref["structure_distance_pct"]["rfx"] == pytest.approx(1.5195, abs=1e-3)
    assert ref["structure_distance_pct"]["openems"] == pytest.approx(0.6644, abs=1e-3)


def test_structure_distance_and_argmin_distance_are_not_interchangeable(fixture):
    ref = fixture["referee"]
    struct_rfx = ref["structure_distance_pct"]["rfx"]
    argmin_rfx = ref["argmin_first_null"]["distances_pct"]["rfx"]
    # Both round to "1.5%"-ish but they are DIFFERENT quantities computed
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
    assert float(np.median(re_z0[passband])) == pytest.approx(50.30264, abs=1e-3)
    assert float(np.median(re_z0[inband])) == pytest.approx(52.37558, abs=1e-3)


def test_leg_derived_passivity_and_column_power(rfx_leg):
    corr = np.asarray(rfx_leg["passivity_correction"])
    assert int((corr > 0.05).sum()) == 0
    assert float(corr.max()) == pytest.approx(0.0145084, abs=1e-6)
    s11 = np.asarray(rfx_leg["s11_mag"])
    s21 = np.asarray(rfx_leg["s21_mag"])
    col_power = s11**2 + s21**2
    assert float(col_power.max()) == pytest.approx(0.999514, abs=1e-5)


def _must_find(pattern: str, text: str, where: str):
    m = re.search(pattern, text)
    assert m is not None, f"expected pattern {pattern!r} not found in {where}"
    return m


def _must_not_find(pattern: str, text: str, where: str):
    m = re.search(pattern, text)
    assert m is None, f"stale pattern {pattern!r} still present (unlabelled) in {where}"


def test_validation_readme_carrier_corrected():
    path = "validation/README.md"
    text = (REPO_ROOT / path).read_text()
    _must_find(r"structure distance 1\.5\d?%? vs OpenEMS 0\.66", text, path)
    _must_not_find(r"structure distance 1\.91%", text, path)
    _must_not_find(r"\(84/120,", text, path)
    _must_find(r"0/120, worst 0\.0145", text, path)
    _must_not_find(r"reads ~67 ", text, path)
    _must_find(r"50\.3 .*52\.4", text, path)


def test_benchmarks_mdx_carrier_corrected():
    path = "docs/public/guide/benchmarks.mdx"
    text = (REPO_ROOT / path).read_text()
    _must_find(r"max column power 0\.9995", text, path)
    _must_not_find(r"max column power 0\.9938", text, path)
    _must_not_find(r"84/120 bins > 0\.05, worst 0\.365", text, path)
    _must_find(r"0/120 bins > 0\.05 on the current leg, worst 0\.0145", text, path)
    _must_find(r"50\.3 .*52\.4", text, path)


def test_palace_sheen_referee_readme_carrier_corrected():
    path = "scripts/diagnostics/palace_sheen_referee/README.md"
    text = (REPO_ROOT / path).read_text()
    _must_find(r"rfx 7\.874 GHz, openEMS 7\.983 GHz", text, path)
    _must_not_find(r"~9-10% first-S21-null\nsplit: \*\*rfx 7\.218", text, path)
    _must_not_find(r"\| ~7\.28\s+\| \(unresolved\)", text, path)
    _must_find(r"\| 6\.944\s+\| 7\.926\s+\| 7\.874\s+\| \*\*1\.52 %\*\*", text, path)
    # every surviving "7.218" must be explicitly framed as historical, within
    # a one-line window either side (the prose wraps across lines).
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "7.218" in line:
            window = " ".join(lines[max(0, i - 1):i + 2])
            assert re.search(r"earlier|history|historical|figure above|regenerat", window, re.I), (
                f"unlabelled stale 7.218 in {path}: {line!r}"
            )


def test_build_referee_producer_docstring_corrected():
    path = "scripts/diagnostics/build_sheen_lpf_palace_referee.py"
    text = (REPO_ROOT / path).read_text()
    _must_find(r"rfx\s*\n7\.874 GHz, openEMS 7\.983 GHz", text, path)
    _must_not_find(r"locks a ~10% split.*rfx 7\.218", text, path)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "7.218" in line:
            window = " ".join(lines[max(0, i - 1):i + 2])
            assert re.search(r"earlier|regenerat", window, re.I), (
                f"unlabelled stale 7.218 in {path}: {line!r}"
            )


def test_check_sparams_runtime_print_corrected():
    path = "scripts/diagnostics/palace_sheen_referee/check_sparams.py"
    text = (REPO_ROOT / path).read_text()
    _must_find(r'rfx 7\.874 \| openEMS 7\.983 GHz', text, path)
    _must_not_find(r'rfx 7\.218 \| openEMS 7\.983 GHz', text, path)


def test_mesh_sheen_docstring_corrected():
    path = "scripts/diagnostics/palace_sheen_referee/mesh_sheen.py"
    text = (REPO_ROOT / path).read_text()
    _must_find(r"rfx 7\.874 GHz, openEMS 7\.983 GHz", text, path)
    for line in text.splitlines():
        if "7.218" in line:
            assert re.search(r"earlier", line, re.I), (
                f"unlabelled stale 7.218 in {path}: {line!r}"
            )


def test_gate_test_docstring_corrected():
    path = "tests/crossval/test_sheen_lpf_palace_referee_gates.py"
    text = (REPO_ROOT / path).read_text()
    _must_find(r"rfx argmin 7\.874 GHz, openEMS argmin 7\.983 GHz", text, path)
    _must_not_find(r"rfx argmin 7\.218 GHz, openEMS argmin 7\.983 GHz", text, path)
