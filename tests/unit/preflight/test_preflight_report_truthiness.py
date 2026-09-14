"""``bool(PreflightReport)`` raises, and nothing else list-like changed (#980).

PreflightReport subclasses ``list``, so inherited truthiness is inverted for
what a report means: an EMPTY (clean) report is falsy, and a report carrying
only advisories is truthy. The gates that read most naturally —
``if not sim.preflight(): raise`` and ``assert sim.preflight()`` — therefore
fire backwards, and the failure is silent: the assert passes exactly when
there ARE issues and fails on a clean pass.

``__bool__`` raises ``TypeError`` so that trap surfaces at the call site with
the replacement named. This file pins three things:

1. every boolean context raises, on a clean report as well as a dirty one —
   including the one that matters, a real ``sim.preflight()`` under ``not``;
2. the raised message names ``.ok`` and ``.raise_for_failure`` so the caller
   can fix the line without reading the source;
3. every other list operation the legacy ``list[str]`` call sites rely on
   still works, and the report API (``.ok`` / ``.errors`` / ``.issues`` /
   ``.format()`` / ``.to_dict()``) is unchanged.
"""
import json

import pytest

from rfx import Simulation
from rfx.api._preflight import PreflightIssue, PreflightReport


U = 2.0**-10


def _clean_sim():
    """Smallest sim in this directory whose preflight finds nothing."""
    sim = Simulation(domain=(20 * U,) * 3, freq_max=1e9, dx=U,
                     cpml_layers=8, boundary="cpml")
    sim.add_source((5 * U,) * 3, "ez", amplitude_kind="current")
    return sim


def _clean_report():
    report = _clean_sim().preflight(check_ntff=False)
    assert len(report) == 0, f"fixture drifted, expected a clean report: {report!r}"
    return report


def _advisory_report():
    return PreflightReport([
        PreflightIssue("probe sits in the CPML", severity="warning", code="probe_in_cpml"),
        PreflightIssue("mesh is 12 cells per wavelength", severity="info", code="resolution"),
    ])


def _error_report():
    return PreflightReport([
        PreflightIssue("port overlaps PEC", severity="error", code="port_in_pec"),
        PreflightIssue("probe sits in the CPML", severity="warning", code="probe_in_cpml"),
    ])


def _all_reports():
    return [
        ("empty", PreflightReport()),
        ("clean-from-sim", _clean_report()),
        ("advisory-only", _advisory_report()),
        ("with-error", _error_report()),
    ]


# --------------------------------------------------------------- (a) it raises

@pytest.mark.parametrize("label,report", _all_reports(), ids=lambda v: v if isinstance(v, str) else "")
def test_bool_raises_for_every_report_severity_mix(label, report):
    """Empty, advisory-only and error reports all refuse bool().

    The empty case is the point: that is the one inherited truthiness gets
    backwards, and it is also the case a passing run produces.
    """
    with pytest.raises(TypeError) as excinfo:
        bool(report)
    assert "PreflightReport cannot be evaluated as a boolean" in str(excinfo.value)


@pytest.mark.parametrize("label,report", _all_reports(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_implicit_boolean_context_raises(label, report):
    """``bool()`` is not a special case — the implicit contexts raise too.

    These are the shapes the defect actually took in the repo: ``if report``,
    ``if not report``, ``and``/``or`` chains, a ternary, and an ``assert``.
    """
    with pytest.raises(TypeError):
        if report:
            pass
    with pytest.raises(TypeError):
        if not report:
            pass
    with pytest.raises(TypeError):
        _ = report and "x"
    with pytest.raises(TypeError):
        _ = report or []
    with pytest.raises(TypeError):
        _ = "dirty" if report else "clean"
    with pytest.raises(TypeError):
        assert report
    with pytest.raises(TypeError):
        while report:
            break
    with pytest.raises(TypeError):
        list(filter(None, [report]))


def test_message_names_the_replacements():
    """The message has to carry the fix, not just the refusal."""
    with pytest.raises(TypeError) as excinfo:
        bool(PreflightReport())
    text = str(excinfo.value)
    for needle in ("report.ok", "report.errors", "report.raise_for_failure()",
                   "len(report)", "report.issues"):
        assert needle in text, f"{needle!r} missing from:\n{text}"
    # It must also explain WHY, or the reader just swaps one guess for another.
    assert "EMPTY" in text and "advisories" in text


# ------------------------------------------- (b) the real-simulation call shape

def test_the_inverted_gate_on_a_real_simulation_raises():
    """``if not sim.preflight(): raise ...`` — the exact reported defect.

    Written against a clean sim, because that is where the old behaviour was
    silently wrong: the report was falsy, the body ran, and the caller
    reported a preflight failure on a simulation that had passed.
    """
    sim = _clean_sim()
    with pytest.raises(TypeError, match="cannot be evaluated as a boolean"):
        if not sim.preflight(check_ntff=False):
            raise RuntimeError("preflight failed")


def test_the_inverted_assert_on_a_real_simulation_raises():
    """``assert sim.preflight()`` used to pass only when preflight found
    problems. It now raises instead of asserting the opposite of its text."""
    sim = _clean_sim()
    with pytest.raises(TypeError, match="cannot be evaluated as a boolean"):
        assert sim.preflight(check_ntff=False)


def test_the_supported_gates_still_work_on_a_real_simulation():
    """The replacements the message points at do what the caller wanted."""
    report = _clean_sim().preflight(check_ntff=False)
    assert report.ok
    assert report.errors == []
    assert len(report) == 0
    assert report.raise_for_failure() is report


# --------------------------------------------------- (c) list behaviour intact

@pytest.mark.parametrize("label,report", _all_reports(), ids=lambda v: v if isinstance(v, str) else "")
def test_list_and_report_api_survive(label, report):
    """Everything except bool() is untouched."""
    items = list(report)
    assert len(report) == len(items)
    assert [i for i in report] == items            # iteration
    assert report.issues == items
    if items:
        assert report[0] is items[0]               # indexing
        assert report[-1] is items[-1]
        assert report[:1] == items[:1]             # slicing
        assert items[0] in report                  # membership
    assert isinstance("\n".join(report), str)      # the legacy list[str] op
    assert report.errors == [i for i in report if i.severity == "error"]
    assert report.warnings == [i for i in report if i.severity not in ("error", "info")]
    assert report.infos == [i for i in report if i.severity == "info"]
    assert report.ok == (not report.errors)
    assert isinstance(report.flux_regions, list)


def test_iteration_helpers_over_items_are_unaffected():
    """``any``/``all`` consume the ITEMS (strings), never the report itself."""
    report = _error_report()
    assert any("PEC" in i for i in report)
    assert all(isinstance(i, str) for i in report)
    assert any(report)                    # every issue is a non-empty string
    assert bool(report.errors)            # a plain list of items is still a list
    assert not bool(PreflightReport().errors)
    assert sorted(report.by_code("port_in_pec")) == ["port overlaps PEC"]


def test_mutation_and_copy_do_not_trip_the_guard():
    report = PreflightReport()
    report.append(PreflightIssue("late finding", severity="error", code="late"))
    report.extend(_advisory_report())
    assert len(report) == 3
    assert len(report.errors) == 1
    assert list(reversed(report))[0] == "mesh is 12 cells per wavelength"


# ------------------------------------------------------------- (d) equality

def test_equality_against_reports_and_plain_lists_still_works():
    """``==`` never consults ``__bool__``, so comparisons are unaffected."""
    a = _advisory_report()
    b = _advisory_report()
    assert a == b
    assert a == ["probe sits in the CPML", "mesh is 12 cells per wavelength"]
    assert a == list(a)
    assert a != _error_report()
    assert PreflightReport() == []
    assert PreflightReport() == PreflightReport()
    assert not (a == _error_report())      # `not (x == y)` is fine: bool of a bool


def test_equality_result_is_usable_in_a_condition():
    """The comparison RESULT is a plain bool, so it gates normally."""
    report = PreflightReport()
    if report == []:
        gated = True
    else:                                   # pragma: no cover - guard
        gated = False
    assert gated


# ------------------------------------------------ (e) serialization / format

def test_to_dict_and_format_on_an_empty_report():
    report = PreflightReport()
    assert report.to_dict() == {"ok": True, "n_issues": 0, "n_errors": 0, "issues": []}
    assert report.format() == "preflight: PASS (no issues)"
    assert json.loads(json.dumps(report.to_dict()))["ok"] is True


def test_to_dict_and_format_on_a_populated_report():
    report = _error_report()
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["n_issues"] == 2 and payload["n_errors"] == 1
    assert [r["code"] for r in payload["issues"]] == ["port_in_pec", "probe_in_cpml"]
    text = report.format()
    assert text.startswith("preflight: FAIL (2 issue(s))")
    assert "ERROR [port_in_pec] port overlaps PEC" in text


def test_format_from_a_real_clean_preflight():
    """The clean-report label comes from len(self), not truthiness."""
    assert _clean_report().format() == "preflight: PASS (no issues)"
