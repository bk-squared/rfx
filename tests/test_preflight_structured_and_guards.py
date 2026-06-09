"""Tier-2: structured preflight records + two new setup guards.

(a) preflight() returns PreflightIssue (a str subclass — fully back-compatible
    with the old list[str]) carrying .severity / .code so an agent can gate:
    errors = [i for i in sim.preflight() if i.severity == "error"].
(b) conformal-PEC + fine dx (<=2mm) is a KNOWN NaN; surface it at setup as an
    error-severity PreflightErrorWarning instead of wasting a run.
(c) all-lossless dielectric in an open domain => artificial-Q trap warning
    (design-guide Anti-Pattern #1), narrow + single-shot to avoid noise.
"""
import warnings
from types import SimpleNamespace

import pytest

from rfx.api import Simulation
from rfx.api._preflight import (
    _PreflightMixin,
    PreflightErrorWarning,
    PreflightIssue,
    _preflight_code_for,
)
from rfx.geometry.csg import Box


# ---------------------------------------------------------------- (a) records
def test_preflight_returns_back_compatible_structured_issues():
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.018), component="ez")   # near CPML => issue
    sim.add_probe((0.01, 0.01, 0.019), component="ez")
    report = sim.preflight()
    assert report, "expected a preflight finding for a source/probe in CPML"
    for issue in report:
        assert isinstance(issue, PreflightIssue)
        assert isinstance(issue, str)          # back-compat: still a string
        assert issue.severity in ("warning", "error")
        assert isinstance(issue.code, str) and issue.code
    # Back-compat operations the old list[str] supported still work.
    assert isinstance("\n".join(report), str)


def test_preflight_issue_is_a_real_string():
    pi = PreflightIssue("ERROR: x", severity="error", code="conformal_nan")
    assert pi == "ERROR: x" and pi.startswith("ERROR") and pi.severity == "error"


def test_code_inference():
    assert _preflight_code_for("probe near CPML region") == "absorber"
    assert _preflight_code_for("conformal PEC NaN") == "conformal_nan"
    assert _preflight_code_for("4.1 cells per lambda") == "resolution"


def test_error_severity_mapping_end_to_end():
    """A PreflightErrorWarning emitted by any validator must surface as a
    severity='error' PreflightIssue (not masking other checks)."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim._validate_ntff_inverse_design = lambda: warnings.warn(
        "forced known-bad config", PreflightErrorWarning
    )
    report = sim.preflight()
    errs = [i for i in report if i.severity == "error"]
    assert any("forced known-bad" in i for i in errs)


# --------------------------------------------------------- (b) conformal guard
def _fake_conformal(dx, dy=None, dz=None, faces=("z_lo", "z_hi")):
    spec = SimpleNamespace(conformal_faces=lambda: set(faces))
    return SimpleNamespace(_boundary_spec=spec, _dx=dx, _dy=dy, _dz=dz)


def test_conformal_fine_dx_warns_error_severity():
    fake = _fake_conformal(1e-3)
    with pytest.warns(PreflightErrorWarning, match="KNOWN"):
        _PreflightMixin._validate_cfg_conformal_fine_dx(fake, 1e-3)


def test_conformal_coarse_dx_silent():
    fake = _fake_conformal(3e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _PreflightMixin._validate_cfg_conformal_fine_dx(fake, 3e-3)


def test_no_conformal_silent():
    fake = _fake_conformal(1e-3, faces=())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _PreflightMixin._validate_cfg_conformal_fine_dx(fake, 1e-3)


# --------------------------------------------------- (c) lossless-resonator
def _box_sim(material):
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add(Box((0.005,) * 3, (0.015,) * 3), material=material)
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.012), component="ez")
    return sim


def _lossless_warnings(sim):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _PreflightMixin._validate_cfg_lossless_resonator_in_absorber(sim, warnings)
    return [w for w in rec if "artificially" in str(w.message).lower()]


def test_lossless_dielectric_in_cpml_warns():
    # alumina: eps_r 9.8, sigma 0 (built-in, resolved via MATERIAL_LIBRARY).
    assert len(_lossless_warnings(_box_sim("alumina"))) == 1


def test_lossy_dielectric_silent():
    # fr4: sigma 0.025 => not the artificial-Q trap, no false positive.
    assert len(_lossless_warnings(_box_sim("fr4"))) == 0
