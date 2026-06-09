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
    PreflightReport,
    PreflightWarning,
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


def test_preflight_report_is_a_list_with_canonical_api():
    """PreflightReport IS a list (back-compat) AND mirrors the in-repo report
    idiom (.issues/.errors/.warnings/.ok/.format()/.to_dict()/.to_json())."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.018), component="ez")   # near CPML => issue
    sim.add_probe((0.01, 0.01, 0.019), component="ez")
    report = sim.preflight()
    assert isinstance(report, PreflightReport) and isinstance(report, list)
    # list[str] ops the 65 legacy call sites rely on
    assert isinstance("\n".join(report), str)
    assert len(report) == len(list(report)) and bool(report)
    # canonical report API
    assert report.issues == list(report)
    assert report.ok == (not report.errors)
    assert set(report.warnings) | set(report.errors) == set(report)
    assert "preflight:" in report.format()


def test_codes_set_at_check_site():
    """Codes come from the checks themselves (PreflightWarning instance /
    PreflightConfigError), NOT from text inference of the message.

    Confirms the deleted ``_preflight_code_for`` is not relied on: a probe in
    CPML carries the absorber_overlap slug set in
    ``_validate_cfg_absorber_placement`` and the source object identifies the
    emitting check.
    """
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.018), component="ez")
    sim.add_probe((0.01, 0.01, 0.019), component="ez")
    report = sim.preflight()
    absorber = report.by_code("absorber_overlap")
    assert absorber, f"expected absorber_overlap code, got {[i.code for i in report]}"
    for issue in absorber:
        assert issue.code == "absorber_overlap"
        assert issue.source == "_validate_cfg_absorber_placement"


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


def test_conformal_fine_dx_warns():
    # WARNING severity (not error/forbid): conformal-fine-dx is a known,
    # development-coupled bug; convergence tests must still RUN it, so it must
    # not hard-fail. Agents gate on the code, not a hard-stop.
    fake = _fake_conformal(1e-3)
    with pytest.warns(UserWarning, match="KNOWN"):
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


# ------------------------------------------------ (d) Phase A meta-coverage
def _bad_sim_probe_in_cpml():
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.018), component="ez")   # absorber_overlap
    sim.add_probe((0.01, 0.01, 0.019), component="ez")
    return sim


def _bad_sim_no_sources():
    # no_sources advisory
    return Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="pec")


def _bad_sim_lossless_q():
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add(Box((0.005,) * 3, (0.015,) * 3), material="alumina")  # lossless_q
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.012), component="ez")
    return sim


def _bad_sim_under_resolved_dielectric():
    # mesh_resolution: a fat high-eps slab under-resolved at coarse dx.
    sim = Simulation(domain=(0.04, 0.04, 0.04), freq_max=20e9, dx=2e-3,
                     boundary="pec")
    sim.add_material("hi", eps_r=12.0)
    sim.add(Box((0.005, 0.005, 0.005), (0.035, 0.035, 0.035)), material="hi")
    sim.add_source((0.02, 0.02, 0.02), component="ez")
    sim.add_probe((0.025, 0.025, 0.025), component="ez")
    return sim


def _bad_sim_upml_refinement():
    # upml_refinement: structurally-impossible config (error severity).
    from rfx import GaussianPulse
    sim = Simulation(freq_max=6e9, domain=(0.04, 0.04, 0.02),
                     boundary="upml", cpml_layers=6, dx=0.002)
    sim.add_refinement((0.004, 0.008), ratio=2)
    sim.add_source((0.01, 0.02, 0.01), "ez",
                   waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.026, 0.02, 0.01), "ez")
    return sim


def _all_emitted_issues(report):
    return list(report)


def test_every_emitted_issue_carries_a_check_site_code():
    """Phase A meta-test: across a battery of deliberately-bad sims, EVERY
    emitted preflight issue must carry a non-empty code != 'uncoded' (codes are
    set at the check site, not inferred)."""
    builders = (
        _bad_sim_probe_in_cpml,
        _bad_sim_no_sources,
        _bad_sim_lossless_q,
        _bad_sim_under_resolved_dielectric,
        _bad_sim_upml_refinement,
    )
    total = 0
    for build in builders:
        report = build().preflight()
        for issue in _all_emitted_issues(report):
            total += 1
            assert isinstance(issue, PreflightIssue)
            assert issue.code and issue.code != "uncoded", (
                f"{build.__name__}: uncoded issue {str(issue)!r}"
            )
            assert issue.severity in ("warning", "error")
    assert total > 0, "battery emitted no issues — meta-test is vacuous"


def test_error_severity_config_issue_is_coded():
    """The structurally-impossible upml+refinement raise surfaces as an
    error-severity issue with its check-site slug (not 'uncoded')."""
    report = _bad_sim_upml_refinement().preflight()
    errs = report.errors
    assert errs, "expected an error-severity issue for upml+refinement"
    assert any(i.code == "upml_refinement" for i in errs), (
        f"codes: {[i.code for i in errs]}"
    )
    assert not report.ok


def test_to_dict_and_to_json_roundtrip_carry_code_and_severity():
    """Real serialization: to_dict()/to_json() carry code + severity per issue
    (the str subclass alone is NOT json-dumpable with its attrs)."""
    import json

    report = _bad_sim_probe_in_cpml().preflight()
    assert report, "expected at least one issue to serialize"
    d = report.to_dict()
    assert d["n_issues"] == len(report)
    for src, rec in zip(report, d["issues"]):
        assert rec["code"] == src.code
        assert rec["severity"] == src.severity
        assert rec["message"] == str(src)
    back = json.loads(report.to_json())
    assert back["issues"][0]["code"] == report[0].code
    assert back["issues"][0]["severity"] == report[0].severity


# ------------------------------------------------ Phase C-full: aggregate strict
def test_strict_aggregates_all_issues_in_one_raise():
    """strict=True escalates EVERY finding in ONE ValueError (aggregate-then-
    raise), not fail-on-first — preserving the historical 'strict escalates any
    issue' contract while reporting all problems at once."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.018), component="ez")   # near CPML
    sim.add_probe((0.01, 0.01, 0.019), component="ez")     # near CPML
    # strict=False shows >= 2 findings...
    report = sim.preflight()
    assert len(report) >= 2, f"need a multi-issue config; got {list(report)}"
    # ...and strict raises ONCE listing all of them.
    with pytest.raises(ValueError) as exc:
        sim.preflight(strict=True)
    text = str(exc.value)
    assert text.count("\n  - ") >= 2, f"expected aggregated list, got: {text}"


def test_raise_for_failure_is_errors_only_gate():
    """report.raise_for_failure() is the SOFTER pre-launch gate: it raises only
    on error-severity, letting advisory warnings through (unlike strict=True)."""
    report = _bad_sim_probe_in_cpml().preflight()   # warnings only, no errors
    assert report and report.ok          # ok == no error-severity issues
    report.raise_for_failure()           # must NOT raise on warning-only report


# ------------------------------------------------ Phase D: validator crash is loud
def test_validator_crash_propagates_not_swallowed():
    """A validator raising a NON-ValueError is a bug, not a finding — it must
    propagate (loud), not degrade to a soft advisory that hides it."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.012), component="ez")

    def _boom():
        raise RuntimeError("validator bug")
    sim._validate_ntff_inverse_design = _boom

    # Via the auto-preflight path (run) the bug must surface, not be swallowed.
    with pytest.raises(RuntimeError, match="validator bug"):
        sim.run(n_steps=5)
