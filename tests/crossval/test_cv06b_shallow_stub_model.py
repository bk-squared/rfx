"""cv06b's shallow-notch falsifier must be independent of the gate it falsifies.

Round 1 of #812 built the shallow-notch defect by rescaling the measured sweep
with the ratio of the SAME ideal-shunt-stub closed form that G2's window is
derived from, so G2 firing on it was an algebraic identity. These tests pin the
round-2 replacement:

 1. the construction reads no cv06b gate constant (mechanical, source-level);
 2. it does not reproduce the gate's reference bandwidth (numeric — a model
    that did would be the gate's own formula in disguise);
 3. it does reproduce the COMMITTED MEASURED sweep of the same board, which is
    what makes it a stand-in for that board rather than an arbitrary curve;
 4. criterion (B) is two-sided on it: G2 passes at the shipped stub width and
    fails at a narrowed one, with the retained -10 dB depth witness passing on
    both — the blindness #812 measured.

Every number asserted here is re-derived in-test and cross-checked against the
committed artifact
``tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json``, so a
stale artifact fails the suite rather than sitting quietly.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "scripts/diagnostics/cv06b_shallow_stub_model.py"
SF = REPO_ROOT / "validation/crossval/comparators/spectral_features.py"
CV06B = REPO_ROOT / "validation/crossval/06b_msl_notch_filter_uniform.py"
FIXTURE = REPO_ROOT / "tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json"
ARTIFACT = (REPO_ROOT
            / "tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json")

DX50 = 50e-6
H_SUB = 300e-6          # the dx=50um producer REALIZES 300um (h_sub_cells 5.08)
W_LINE_CELLS = 12
BOARD = dict(h_sub=H_SUB, eps_r=3.66, tan_d=0.0037, sigma=5.8e7,
             l_stub=12e-3, l_line=5e-3, z_ref=50.0)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def model():
    return _load(MODEL, "_cv06b_shallow_stub_model")


@pytest.fixture(scope="module")
def sf():
    return _load(SF, "_sf")


@pytest.fixture(scope="module")
def cv():
    return _load(CV06B, "_cv06b")


@pytest.fixture(scope="module")
def sweep():
    d = json.loads(FIXTURE.read_text())
    return np.asarray(d["freqs_ghz"], dtype=float) * 1e9, \
        np.asarray(d["s21_mag"], dtype=float)


@pytest.fixture(scope="module")
def artifact():
    return json.loads(ARTIFACT.read_text())


def _s21(model, f, n_cells):
    return np.abs(model.stub_board_s21(
        f, w_line=W_LINE_CELLS * DX50, w_stub=n_cells * DX50, **BOARD))


def _features(sf, f, s21, level_db=-10.0):
    est = sf.refined_extremum(f, s21)
    band = sf.band_at_level(f, s21, level_db, est["index"])
    assert band is not None
    lo, hi, _ = band
    return est["refined_f"], (hi - lo) / est["refined_f"], est["depth_db"]


# ---------------------------------------------------------------- 1. source

def _code_only(path: Path) -> str:
    """Module source with every docstring removed, so prose that NAMES a gate
    constant while explaining why it is absent does not read as a use of it."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            body.pop(0)
    return ast.unparse(tree)


def test_construction_reads_no_cv06b_gate_constant():
    """The defect must not be built out of the quantity being judged."""
    code = _code_only(MODEL)
    for forbidden in ("STOPBAND", "0.210274", "NOTCH_FREQ_TOL",
                      "HALF_GRID_WITNESS", "06b_msl_notch", "atan"):
        assert forbidden not in code, (
            f"{MODEL.name} uses {forbidden!r} in CODE: the shallow-notch "
            f"construction must not read cv06b's gate constants nor rebuild "
            f"the closed form they come from")


# --------------------------------------------------------------- 2. numeric

def test_model_does_not_reproduce_the_gate_reference(model, sf, cv, sweep,
                                                     artifact):
    """At the shipped width the model must DISAGREE with G2's reference.

    A construction that returned (4/pi)*atan(r/6) exactly would be the gate's
    own formula wearing a different name — round 1's defect.
    """
    f, _ = sweep
    _, bw_frac, _ = _features(sf, f, _s21(model, f, W_LINE_CELLS))
    dep = (bw_frac - cv.STOPBAND_BW_FRAC_IDEAL) / cv.STOPBAND_BW_FRAC_IDEAL
    assert abs(dep) > 0.01, (
        "the geometry model reproduces the gate's closed form to better than "
        "1 %; it carries no independent physics")
    ind = artifact["case_C_shallow_notch_from_geometry"]["independence"]
    assert ind["model_bw_frac_at_shipped_width"] == pytest.approx(bw_frac,
                                                                  rel=1e-12)
    assert ind["model_departure_from_gate_reference_pct"] == pytest.approx(
        dep * 100.0, rel=1e-9)


# ------------------------------------------------------- 3. board validity

def test_model_reproduces_the_committed_measured_sweep(model, sf, sweep,
                                                       artifact):
    """The model is a stand-in for THIS board, not an arbitrary notch."""
    f, meas = sweep
    f0_m, bw_m, depth_m = _features(sf, f, meas)
    f0_g, bw_g, depth_g = _features(sf, f, _s21(model, f, W_LINE_CELLS))
    assert abs(f0_g - f0_m) / f0_m < 0.02          # notch frequency, 2 %
    assert abs(bw_g - bw_m) / bw_m < 0.05          # -10 dB width, 5 %
    assert abs(depth_g - depth_m) < 1.0            # sampled depth, 1 dB
    ind = artifact["case_C_shallow_notch_from_geometry"]["independence"]
    assert ind["model_vs_measured_f_notch_pct"] == pytest.approx(
        (f0_g - f0_m) / f0_m * 100.0, rel=1e-9)


# ------------------------------------------------------- 4. two-sided (B)

@pytest.mark.parametrize("n_cells,expect_g2", [(12, True), (10, True),
                                               (8, False), (6, False),
                                               (5, False)])
def test_g2_is_two_sided_on_the_geometry_ladder(model, sf, cv, sweep,
                                                n_cells, expect_g2):
    f, _ = sweep
    s = _s21(model, f, n_cells)
    _, bw_frac, depth_db = _features(sf, f, s)
    lo, hi = cv.STOPBAND_BW_RATIO_WINDOW
    ratio = bw_frac / cv.STOPBAND_BW_FRAC_IDEAL
    assert (lo < ratio < hi) is expect_g2, (
        f"{n_cells}-cell stub: BW ratio {ratio:.4f} vs window ({lo}, {hi})")
    # the gate the audit found blind must keep PASSING on every row, including
    # the ones G2 rejects — that is the blindness, shown next to its fix.
    assert depth_db < -10.0


def test_artifact_rows_match_a_fresh_derivation(model, sf, cv, sweep,
                                                artifact):
    f, _ = sweep
    for row in artifact["case_C_shallow_notch_from_geometry"]["rows"]:
        s = _s21(model, f, row["stub_cells"])
        f0, bw_frac, depth = _features(sf, f, s)
        assert row["bw_frac"] == pytest.approx(bw_frac, rel=1e-12)
        assert row["f_notch_refined_hz"] == pytest.approx(f0, rel=1e-12)
        assert row["notch_depth_db"] == pytest.approx(depth, rel=1e-12)
        assert row["G2_pass"] is bool(
            cv.STOPBAND_BW_RATIO_WINDOW[0]
            < bw_frac / cv.STOPBAND_BW_FRAC_IDEAL
            < cv.STOPBAND_BW_RATIO_WINDOW[1])
