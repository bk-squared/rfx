"""Traced advisory outcomes without an FDTD step (P1–P5 and T-table)."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap
from typing import NamedTuple
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.errors import RfxError
from rfx.geometry.rasterize_grid import TracedShapeDimensionError
from rfx.preflight import _not_judged, _registry
from rfx.preflight._common import PreflightReport, PreflightWarning
from rfx.preflight._not_judged import (
    CONSTRUCTOR_PARAMETERS, CONTAINER_METHODS, current_names, judge_or_report, not_judged,
    preflight_scope, traced_input_names,
)
from tests.unit.preflight._not_judged_scenes import SCENES, build_scene, issue_rows


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("transform", ["make_jaxpr", "grad"])
def test_p1_material(mesh, transform):
    reports = []

    def objective(eps):
        options = {} if mesh == "uniform" else {
            "dz_profile": np.array([.0009] * 6 + [.0011] * 6),
        }
        sim = Simulation(domain=(.012, .012, .012), dx=.001,
                         freq_max=10e9, boundary="cpml", cpml_layers=4, **options)
        sim.add_material("dielectric", eps_r=eps)
        sim.add(Box((.003, .003, .003), (.008, .008, .008)), material="dielectric")
        reports.append(sim.preflight())
        return eps * eps

    getattr(jax, transform)(objective)(4.)
    assert reports
    for report in reports:
        assert any(
            issue.code == "not_judged_traced"
            and issue.severity == "warning"
            and str(issue) == "not judged: _validate_mesh_quality reads a traced input (add_material(eps_r))"
            for issue in report
        )


def _extra(monkeypatch, name, check):
    monkeypatch.setattr(_registry, "EXTRA_CONFIG_CHECKS", [
        _registry.ConfigCheck(name, check, "test"),
    ])


def test_p2_rfx_error_precedes_jax_error(monkeypatch):
    refusal = TracedShapeDimensionError("named shape refusal")
    assert isinstance(refusal, RfxError)
    assert isinstance(refusal, jax.errors.ConcretizationTypeError)

    def check(sim, ctx):
        assert _not_judged._has_tracer(_not_judged._scope.get().declarations._materials)
        raise refusal

    _extra(monkeypatch, "refusal", check)
    reports = []

    def objective(value):
        sim = build_scene("vacuum_without_source")
        sim.add_material("unused", eps_r=value)
        reports.append(sim.preflight(check_resolution=False))
        return value

    jax.make_jaxpr(objective)(2.)
    errors = [i for i in reports[0] if i.severity == "error"]
    assert len(errors) == 1
    assert str(errors[0]) == "ERROR: named shape refusal"
    assert not any(i.code == "not_judged_traced" for i in reports[0])


def test_p3_key_error_propagates(monkeypatch):
    failure = KeyError("validator bug")

    def check(sim, ctx):
        assert _not_judged._has_tracer(_not_judged._scope.get().declarations._materials)
        raise failure

    _extra(monkeypatch, "bug", check)

    def objective(value):
        sim = build_scene("vacuum_without_source")
        sim.add_material("unused", eps_r=value)
        sim.preflight(check_resolution=False)
        return value

    with pytest.raises(KeyError) as caught:
        jax.make_jaxpr(objective)(2.)
    assert caught.value is failure
    assert current_names() == ()


@pytest.mark.parametrize("scene", SCENES)
def test_p4_concrete_issues_frozen_on_main(scene):
    path = Path(__file__).with_name("data") / "not_judged_p4_frozen.json"
    # The frozen main artifact must exist for this comparison.
    frozen = json.loads(path.read_text())
    assert frozen["base_commit"] == "084b5032b7d014f12b65b4b947cf61a840970bf4"
    actual = json.loads(json.dumps(issue_rows(build_scene(scene).preflight())))
    assert actual == frozen["scenes"][scene]


def test_p5_forward_override_reaches_advisory(monkeypatch):
    # An opt-in scalar-override advisory exercises the public extra-check API.
    # Native waveguide checks deliberately use drawn geometry for traced
    # overrides; TF/SF already reports its own unjudged-invariance warning.
    # Stop at the gate, after real forward -> auto_preflight, before dispatch.
    class PreflightFinished(Exception):
        pass

    reports = []

    def scalar_override_check(sim, ctx):
        np.asarray(sim._waveguide_call_overrides[0])

    _extra(monkeypatch, "scalar_override_check", scalar_override_check)

    def gate(self, issues, **kwargs):
        reports.append(issues)
        raise PreflightFinished

    monkeypatch.setattr(Simulation, "_run_preflight_gate", gate)
    sim = Simulation(freq_max=12e9, domain=(.09, .04, .02), dx=.005,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=4)
    sim.add_waveguide_port(.015, direction="+x", freqs=jnp.array([8e9, 9e9]),
                           f0=8.5e9, bandwidth=.5)

    def objective(value):
        with pytest.raises(PreflightFinished):
            sim.forward(n_steps=1, eps_override=value)
        return value

    jax.make_jaxpr(objective)(jnp.ones((1, 1, 1)))
    assert reports
    assert any(
        i.code == "not_judged_traced"
        and str(i) == "not judged: scalar_override_check reads a traced input (forward(eps_override))"
        for i in reports[0]
    )
    assert current_names() == ()


def _assert_container_table(table):
    # Derive coverage independently from constructor source AND a real instance.
    # Annotations catch optional containers whose default is None. Destructured
    # assignments such as domain, self._nominal_profiles are seen by ast.walk.
    tree = ast.parse(textwrap.dedent(inspect.getsource(Simulation.__init__)))
    sim = build_scene("vacuum_without_source")
    required = {name for name, value in vars(sim).items()
                if isinstance(value, (list, dict, tuple))}
    # This unannotated optional dict is assigned by admit_graded_profiles;
    # the default scene receives None, so its type is not observable there.
    required.add("_nominal_profiles")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Dict, ast.Tuple)):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    required.add(target.attr)
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            if any(isinstance(part, ast.Name) and part.id in {"list", "dict", "tuple"}
                   for part in ast.walk(node.annotation)):
                required.add(target.attr)
    assert required <= table.keys(), sorted(required - table.keys())


@pytest.mark.source_scan
def test_t_table_covers_init_containers():
    _assert_container_table(CONTAINER_METHODS)


@pytest.mark.source_scan
@pytest.mark.parametrize("container", ["_materials", "_nominal_profiles", "_ntff"])
def test_t_table_rejects_missing_container(container):
    incomplete = dict(CONTAINER_METHODS)
    del incomplete[container]
    with pytest.raises(AssertionError, match=container):
        _assert_container_table(incomplete)


def test_scan_first_level_names_without_retaining_tracers():
    @dataclass
    class Record:
        eps_r: object
        shape: object

    class Waveform(NamedTuple):
        f0: object

    results = []

    def objective(value):
        sim = SimpleNamespace(
            _materials={"one": Record(value, None), "two": Record(value, None)},
            _geometry=[Record(None, {"nested": [Waveform(value)]})],
            _ntff=((0., 0., 0.), (1., 1., 1.), (value,)),
            _new_container=[Waveform(value)],
            _realized_geometry_record={"cached": Waveform(value)},
            _msl_default_memo={"cached": Waveform(value)},
            _domain=(value, 1., 1.),
            _dz_profile=value,
            _ports=[{"waveform": Waveform(value)}],
        )
        results.append(traced_input_names(sim, {"eps_override": value}))
        return value

    jax.make_jaxpr(objective)(2.)
    assert results == [(
        "Simulation(domain)", "Simulation(dz_profile)", "add(shape)", "add_material(eps_r)",
        "add_ntff_box(freqs)", "add_port(waveform)", "forward(eps_override)",
    )]
    assert all(isinstance(name, str) for name in results[0])
    assert traced_input_names(SimpleNamespace(_materials={"x": {"eps_r": np.ones(3)}}), {}) == ()


@pytest.mark.parametrize("error_name", [
    "ConcretizationTypeError", "TracerArrayConversionError",
    "TracerBoolConversionError", "TracerIntegerConversionError",
])
def test_conversion_types_and_internal_fallback(error_name):
    reports = []
    later = []

    def objective(value):
        report = PreflightReport()

        def check():
            raise getattr(jax.errors, error_name)(value)

        with preflight_scope(SimpleNamespace()):
            judge_or_report("internal_check", check, report)
            judge_or_report("later_check", lambda: later.append(True), report)
        reports.append(report)
        return value

    jax.make_jaxpr(objective)(1.)
    assert later == [True]
    assert [(i.code, i.severity, str(i)) for i in reports[0]] == [(
        "not_judged_traced", "warning",
        "not judged: internal_check reads an input traced inside rfx",
    )]


def test_warning_collector_preserves_structured_code_only(monkeypatch):
    plain = UserWarning("plain warning with a code")
    plain.code = "plain_code"

    def check(sim, ctx):
        ctx.warn.warn(plain)
        ctx.warn.warn(PreflightWarning("structured warning", code="structured_code"))
        not_judged(ctx.warn, "warning_check")

    _extra(monkeypatch, "warning_channel", check)
    report = build_scene("vacuum_without_source").preflight(check_resolution=False)
    assert [(i.code, str(i)) for i in report if str(i) in {
        "plain warning with a code", "structured warning",
        "not judged: warning_check reads an input traced inside rfx",
    }] == [
        ("uncoded", "plain warning with a code"),
        ("structured_code", "structured warning"),
        ("not_judged_traced", "not judged: warning_check reads an input traced inside rfx"),
    ]


def test_not_judged_deduplicates_per_call_and_resets():
    for _ in range(2):
        report = PreflightReport()
        with warnings.catch_warnings(record=True) as caught, preflight_scope(SimpleNamespace()):
            warnings.simplefilter("always")
            not_judged(report, "check")
            not_judged(warnings, "check")
            not_judged(report, "other_check")
        assert len(report) == 2
        assert not caught
        assert current_names() == ()


def test_report_scope_resets_on_exception():
    with pytest.raises(KeyError), preflight_scope(SimpleNamespace()):
        raise KeyError("stop")
    assert current_names() == ()


def test_skipped_profile_reports_but_none_does_not():
    from rfx.preflight._common import profile_span_is_uniform

    with warnings.catch_warnings(record=True) as concrete:
        warnings.simplefilter("always")
        assert profile_span_is_uniform(.001, None, .001, .002)
    assert not concrete

    def objective(profile):
        with warnings.catch_warnings(record=True) as caught, preflight_scope(SimpleNamespace()):
            warnings.simplefilter("always")
            assert profile_span_is_uniform(.001, profile, .001, .002)
            assert profile_span_is_uniform(.001, profile, .001, .002)
        assert len(caught) == 1
        assert caught[0].message.code == "not_judged_traced"
        assert str(caught[0].message) == (
            "not judged: profile_span_is_uniform reads an input traced inside rfx"
        )
        return profile

    jax.make_jaxpr(objective)(jnp.ones(4) * .001)


def test_retained_region_failure_has_a_finding(monkeypatch):
    from rfx.preflight import _realization_subjects as subjects

    def objective(value):
        selection = subjects._Selection(SimpleNamespace(), ("geometry[0]",), lambda: None)
        finding = SimpleNamespace(_subjects=selection, _region=subjects._Region(selection, None))
        monkeypatch.setattr(subjects, "_selection", lambda *args: selection)
        monkeypatch.setattr(subjects._Selection, "resolve", lambda self: ("geometry[0]",))

        def resolve_region(self):
            np.asarray(value)

        monkeypatch.setattr(subjects._Region, "resolve", resolve_region)
        with warnings.catch_warnings(record=True) as caught, preflight_scope(SimpleNamespace()):
            warnings.simplefilter("always")
            subjects.bind_retained_subjects(finding, SimpleNamespace())
        assert finding._subjects == ("geometry[0]",)
        assert finding._region is None
        assert len(caught) == 1
        assert caught[0].message.code == "not_judged_traced"
        assert str(caught[0].message) == (
            "not judged: bind_retained_subjects reads an input traced inside rfx"
        )
        return value

    jax.make_jaxpr(objective)(1.)


def _count_scans(monkeypatch):
    calls = []
    original = _not_judged.traced_input_names

    def scan(*args):
        calls.append(None)
        return original(*args)

    monkeypatch.setattr(_not_judged, "traced_input_names", scan)
    return calls


def test_concrete_preflight_never_scans(monkeypatch):
    from rfx import GaussianPulse

    calls = _count_scans(monkeypatch)
    sim = Simulation(freq_max=10e9, domain=(.12, .12, .04), dx=.002, boundary="pec")
    for layer in range(2):
        for iy in range(10):
            for ix in range(10):
                index = layer * 100 + iy * 10 + ix
                name = f"dielectric-{index}"
                sim.add_material(name, eps_r=2.0 + .001 * index)
                lo = (.01 + .01 * ix, .01 + .01 * iy, .008 + .012 * layer)
                sim.add(Box(lo, tuple(value + .006 for value in lo)), material=name)
    for index in range(20):
        sim.add_probe((.012 + .004 * index, .11, .015), "ez")
    for index in range(4):
        sim.add_port((.02 + .02 * index, .11, .02), "ez", impedance=50.0,
                     extent=.006, radius=.0002, waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    for coordinate in (.006, .114):
        sim.add_flux_monitor(axis="x", coordinate=coordinate, freqs=(4e9, 5e9, 6e9))
    sim.add_ntff_box((.008, .008, .006), (.112, .112, .034), freqs=(4e9, 5e9, 6e9))
    for _ in range(2):
        report = sim.preflight()
        assert not report.by_code("not_judged_traced")
    assert calls == []


def test_traced_preflight_scans_once(monkeypatch):
    from tests.unit.runners.test_shape_gradient_refusal import shape_sim

    calls = _count_scans(monkeypatch)
    reports = []

    def objective(radius):
        reports.append(shape_sim(radius).preflight())
        return radius

    jax.make_jaxpr(objective)(.0023)
    findings = reports[0].by_code("not_judged_traced")
    assert len({str(issue).split(" reads ")[0] for issue in findings}) >= 2
    assert calls == [None]


@pytest.mark.parametrize("raises", [False, True])
def test_scope_contexts_reset_on_exit(raises):
    def exercise(value):
        try:
            with preflight_scope(SimpleNamespace(_materials={"d": {"eps_r": value}})):
                with _not_judged.reporting_to(PreflightReport()):
                    def check():
                        not_judged(None, "scope_check")
                        assert current_names() == ("add_material(eps_r)",)
                        assert _not_judged._check.get() == "scope_check"
                        assert _not_judged._report.get() is not None
                        if raises:
                            raise KeyError("stop")
                    judge_or_report("scope_check", check, None)
        except KeyError:
            assert raises
        for variable in (_not_judged._scope, _not_judged._check, _not_judged._report):
            assert variable.get() is None
        return value

    jax.make_jaxpr(exercise)(2.)


def _assert_constructor_table(table):
    tree = ast.parse(textwrap.dedent(inspect.getsource(Simulation.__init__)))
    parameters = set(inspect.signature(Simulation.__init__).parameters) - {"self"}
    sim = build_scene("vacuum_without_source")
    required = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        inputs = {part.id for part in ast.walk(node.value)
                  if isinstance(part, ast.Name)} & parameters
        if not inputs:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for part in ast.walk(target):
                if not (isinstance(part, ast.Attribute) and isinstance(part.value, ast.Name)
                        and part.value.id == "self"):
                    continue
                value = getattr(sim, part.attr)
                if part.attr not in CONTAINER_METHODS and (
                    value is None or isinstance(value, (str, bool, int, float, complex, np.ndarray, np.generic, jax.Array))
                ):
                    required[part.attr] = inputs
    assert required.keys() <= table.keys(), sorted(required.keys() - table.keys())
    for attr, inputs in required.items():
        assert table[attr] in inputs, (attr, inputs)


@pytest.mark.source_scan
def test_t_table_covers_init_parameters():
    _assert_constructor_table(CONSTRUCTOR_PARAMETERS)


@pytest.mark.source_scan
@pytest.mark.parametrize("attribute", sorted(CONSTRUCTOR_PARAMETERS))
def test_t_table_rejects_missing_parameter(attribute):
    incomplete = dict(CONSTRUCTOR_PARAMETERS)
    del incomplete[attribute]
    with pytest.raises(AssertionError, match=attribute):
        _assert_constructor_table(incomplete)


def _graded_profile_scene(profile):
    from rfx import GaussianPulse

    sim = Simulation(freq_max=10e9, domain=(.012,) * 3, dx=.001, boundary="pec",
                     dz_profile=profile, dt=.2*.001/299792458., dt_min_cell=.0005)
    sim.add_source((.004, .006, .006), "ez", waveform=GaussianPulse(f0=5e9, bandwidth=.8),
                   amplitude_kind="current")
    return sim


def test_traced_dz_profile_names_constructor_parameter():
    reports = []

    def objective(profile):
        reports.append(_graded_profile_scene(profile).preflight())
        return profile

    jax.make_jaxpr(objective)(jnp.full(12, .001))
    findings = reports[0].by_code("not_judged_traced")
    assert findings
    assert all("Simulation(dz_profile)" in str(issue) for issue in findings)


def test_scope_snapshot_keeps_only_declared_references():
    def objective(value):
        materials = {"d": {"eps_r": value}}
        sim = SimpleNamespace(_materials=materials, _dz_profile=value,
                              _realized_geometry_record={"cache": value})
        entry = {"eps_override": value}
        with preflight_scope(sim, entry):
            scope = _not_judged._scope.get()
            assert scope.declarations._materials is materials
            assert scope.declarations._dz_profile is value
            assert not hasattr(scope.declarations, "_realized_geometry_record")
            sim._materials = {}
            sim._dz_profile = None
            entry.clear()
            not_judged(PreflightReport(), "snapshot")
            assert current_names() == (
                "Simulation(dz_profile)", "add_material(eps_r)", "forward(eps_override)",
            )
        return value

    jax.make_jaxpr(objective)(2.)


@pytest.mark.parametrize("skip_preflight", [True, False])
def test_traced_sphere_reports_only_inside_preflight(monkeypatch, skip_preflight):
    from tests.unit.runners.test_shape_gradient_refusal import objective

    reports = []
    original = Simulation._run_preflight_gate

    def gate(self, report, **kwargs):
        reports.append(report)
        return original(self, report, **kwargs)

    monkeypatch.setattr(Simulation, "_run_preflight_gate", gate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jax.make_jaxpr(lambda radius: objective(radius, skip_preflight=skip_preflight))(.0023)
    assert not [w for w in caught if getattr(w.message, "code", None) == "not_judged_traced"]
    if skip_preflight:
        assert not reports
    else:
        assert len(reports) == 1
        assert reports[0].by_code("not_judged_traced")


def test_not_judged_outside_scope_ignores_report():
    report = PreflightReport()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        not_judged(report, "outside")
        not_judged(warnings, "outside")
        not_judged(None, "outside")
    assert len(report) == 0
    assert not caught


def test_profile_helper_uses_calling_check_name():
    from rfx.preflight._common import profile_span_is_uniform

    def objective(profile):
        report = PreflightReport()
        with preflight_scope(SimpleNamespace(_dz_profile=profile)), _not_judged.reporting_to(report):
            def check():
                profile_span_is_uniform(.001, profile, .001, .002)
                _not_judged.traced_skip(profile, "_waveguide_setup_planes")
                _not_judged.report_tracer_error(
                    jax.errors.TracerArrayConversionError(profile), "_port_transverse_spans")
                not_judged(report, "calling_check")
            judge_or_report("calling_check", check, report)
        assert len(report) == 1
        assert str(report[0]) == (
            "not judged: calling_check reads a traced input (Simulation(dz_profile))"
        )
        return profile

    jax.make_jaxpr(objective)(jnp.full(12, .001))


def test_strict_traced_profile_escalates_only_not_judged():
    profile = np.full(12, .001)
    assert len(_graded_profile_scene(profile).preflight(strict=True)) == 0

    def objective(value):
        with pytest.raises(ValueError, match=r"preflight \(strict\)") as caught:
            _graded_profile_scene(value).preflight(strict=True)
        diagnostics = caught.value.diagnostics
        assert diagnostics
        assert {item.code for item in diagnostics} == {"not_judged_traced"}
        assert "Simulation(dz_profile)" in str(caught.value)
        return value

    jax.make_jaxpr(objective)(jnp.asarray(profile))
