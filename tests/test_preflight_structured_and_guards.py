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
    # Target a validator run() actually invokes (run() uses check_ntff=False, so
    # the NTFF inverse-design check is NOT on its path).
    sim._validate_simulation_config = _boom

    # Via the auto-preflight path (run) the bug must surface, not be swallowed.
    with pytest.raises(RuntimeError, match="validator bug"):
        sim.run(n_steps=5)


# ----------------------------------------- run() error-severity + NTFF surface
def test_run_skips_ntff_inverse_design_check_but_forward_runs_it():
    """run() must NOT hard-fail on the inverse-design NTFF check (it historically
    never ran it — run() uses check_ntff=False), while forward()/optimize (the
    inverse-design entry points) still do. Regression lock for the NTFF/PEC
    behavior change flagged in cold review."""
    def _sim():
        s = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
        s.add_source((0.01, 0.01, 0.01), component="ez")
        s.add_probe((0.01, 0.01, 0.012), component="ez")
        # Stand in for an NTFF-box-crosses-PEC error-severity finding.
        s._validate_ntff_inverse_design = lambda: (_ for _ in ()).throw(
            ValueError("NTFF box face crosses PEC")
        )
        return s

    # run(): check_ntff=False -> the NTFF validator is not invoked -> no hard-fail
    _sim().run(n_steps=5)
    # forward(): check_ntff=True -> invoked -> error-severity -> re-raised
    with pytest.raises(ValueError, match="NTFF box face crosses PEC"):
        _sim().forward(n_steps=5)


def test_run_hard_fails_on_error_severity_and_skip_bypasses():
    """run() re-raises on a structurally-impossible (error-severity) config, and
    skip_preflight=True is the documented escape hatch."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.012), component="ez")
    sim._validate_simulation_config = lambda: (_ for _ in ()).throw(
        ValueError("structurally impossible config")
    )
    with pytest.raises(ValueError, match="structurally impossible config"):
        sim.run(n_steps=5)
    # escape hatch: skip_preflight bypasses the preflight (and its re-raise)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # no preflight warning/raise should fire
        sim.run(n_steps=5, skip_preflight=True)


# ------------------------------------------------- issue #166: 2D z + units
def test_absorber_overlap_no_false_positive_on_2d_collapsed_z():
    """2D modes collapse z to a single cell with NO absorber (Grid strips z
    from cpml_axes and sets pad_z=0). Every 2D source/probe necessarily sits
    at z=0, so the z-axis proximity check must not fire (issue #166: cv03
    emitted one absorber_overlap line per line-source point)."""
    sim = Simulation(domain=(16e-6, 9e-6, 1e-7), freq_max=7.5e13, dx=1e-7,
                     boundary="upml", cpml_layers=20, mode="2d_tmz")
    # mid-domain in x and y — only the collapsed z coordinate (0) is "near"
    # the phantom z absorber the old mirror logic assumed.
    sim.add_source((8e-6, 4.5e-6, 0), component="ez")
    sim.add_probe((9e-6, 4.5e-6, 0), component="ez")
    report = sim.preflight()
    overlap = report.by_code("absorber_overlap")
    assert not overlap, f"false positive on collapsed z axis: {list(overlap)}"


def test_absorber_overlap_still_fires_on_2d_xy():
    """The 2D-z exemption must not silence real x/y absorber overlap."""
    sim = Simulation(domain=(16e-6, 9e-6, 1e-7), freq_max=7.5e13, dx=1e-7,
                     boundary="upml", cpml_layers=20, mode="2d_tmz")
    sim.add_source((1e-7, 4.5e-6, 0), component="ez")   # deep in x_lo absorber
    report = sim.preflight()
    overlap = report.by_code("absorber_overlap")
    assert overlap, "expected absorber_overlap for a source at the x_lo edge"
    assert any("x-thickness" in str(i) for i in overlap)


def test_unit_adaptive_formatting_helpers():
    """_fmt_len/_fmt_freq pick units that keep digits visible at any scale
    (issue #166: fixed mm/GHz rendered 0.1µm as 0.000mm and 74.95THz as
    74950.00GHz)."""
    from rfx.api._preflight import _fmt_len, _fmt_freq
    assert _fmt_len(1e-7) == "100nm"
    assert _fmt_len(2e-6) == "2µm"
    assert _fmt_len(0.002) == "2mm"
    assert _fmt_len(0.02286) == "22.86mm"
    assert _fmt_len(1.5) == "1.5m"
    assert _fmt_len(5e-10) == "0.5nm"
    assert _fmt_len(0.0) == "0mm"
    assert _fmt_freq(7.495e13) == "74.95THz"
    assert _fmt_freq(10e9) == "10GHz"
    assert _fmt_freq(9.322e9) == "9.322GHz"
    assert _fmt_freq(2.45e6) == "2.45MHz"


def test_mesh_warning_uses_adaptive_units_at_optical_scale():
    """The cv03-class mesh-resolution warning must print THz/µm, not
    0.000mm / five-digit GHz, at optical scale."""
    sim = Simulation(domain=(16e-6, 9e-6, 1e-7), freq_max=7.495e13, dx=1e-7,
                     boundary="upml", cpml_layers=20, mode="2d_tmz")
    sim.add_material("wg", eps_r=12.0)
    sim.add(Box((0, 4e-6, 0), (16e-6, 5e-6, 1e-7)), material="wg")
    sim.add_source((8e-6, 4.5e-6, 0), component="ez")
    report = sim.preflight()
    mesh = [str(i) for i in report.by_code("mesh_resolution")
            if "cells per λ_eff" in str(i)]
    assert mesh, "expected the cells-per-λ_eff warning at 11.5 cells/λ_eff"
    assert any("74.95THz" in m for m in mesh), mesh
    assert any("100nm" in m for m in mesh), mesh
    assert not any("0.000mm" in m for m in mesh), mesh
