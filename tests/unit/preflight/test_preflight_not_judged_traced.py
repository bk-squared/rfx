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
from rfx.preflight import _registry
from rfx.preflight._common import PreflightReport, PreflightWarning
from rfx.preflight._not_judged import (
    CONTAINER_METHODS, current_names, judge_or_report, not_judged,
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
        assert current_names() == ("add_material(eps_r)",)
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
        assert current_names() == ("add_material(eps_r)",)
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
    # Intentionally fail if the leader has not yet supplied the main artifact.
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
            _ports=[{"waveform": Waveform(value)}],
        )
        results.append(traced_input_names(sim, {"eps_override": value}))
        return value

    jax.make_jaxpr(objective)(2.)
    assert results == [(
        "_new_container(f0)", "add(shape)", "add_material(eps_r)",
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

        with preflight_scope(()):
            judge_or_report("internal_check", check, report, ())
            judge_or_report("later_check", lambda: later.append(True), report, ())
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
        not_judged(ctx.warn, "warning_check", ())

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
        with warnings.catch_warnings(record=True) as caught, preflight_scope(("add_material(eps_r)",)):
            warnings.simplefilter("always")
            not_judged(report, "check", current_names())
            not_judged(warnings, "check", current_names())
            not_judged(report, "check", ("forward(eps_override)",))
        assert len(report) == 2
        assert not caught
        assert current_names() == ()


def test_report_scope_resets_on_exception():
    with pytest.raises(KeyError), preflight_scope(("add_material(eps_r)",)):
        raise KeyError("stop")
    assert current_names() == ()


def test_skipped_profile_reports_but_none_does_not():
    from rfx.preflight._common import profile_span_is_uniform

    with warnings.catch_warnings(record=True) as concrete:
        warnings.simplefilter("always")
        assert profile_span_is_uniform(.001, None, .001, .002)
    assert not concrete

    def objective(profile):
        with warnings.catch_warnings(record=True) as caught, preflight_scope(()):
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
        with warnings.catch_warnings(record=True) as caught, preflight_scope(()):
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
