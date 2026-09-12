"""Finite flux-window reporting uses the runners' actual cells, without FDTD."""
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._preflight import PreflightIssue, PreflightReport
from rfx.probes.flux_region import flux_region_message
from rfx.runners.uniform import build_flux_monitor_cfgs


U = 2.0**-10


def _sim(*, graded=False):
    profiles = {}
    domain = (20 * U,) * 3
    if graded:
        widths = np.full(20, U)
        widths[8:11] *= [1.125, 1.25, 1.125]
        profiles = {f"d{axis}_profile": widths.copy() for axis in "xyz"}
        domain = (20.5 * U,) * 3
    sim = Simulation(domain=domain, freq_max=1e9, dx=U,
                     cpml_layers=8, boundary="cpml", **profiles)
    # Keep the source on a locally uniform node; the flux window alone
    # exercises the graded cells without an unrelated source advisory.
    sim.add_source((5 * U,) * 3, "ez", amplitude_kind="current")
    return sim


def _add(sim, axis="x", **kwargs):
    config = dict(coordinate=10 * U, size=(4 * U, 8 * U),
                  center=(10 * U, 10 * U), name="window", n_freqs=2)
    config.update(kwargs)
    sim.add_flux_monitor(axis=axis, **config)


def _report(sim, **kwargs):
    return sim.preflight(check_ntff=False, **kwargs)


def _nu_runner_monitor(sim, monkeypatch):
    """Capture the real NU runner's final monitor args before the stepper."""
    import rfx.runners.nonuniform as runner

    class ReachedStepper(BaseException):
        pass

    captured = []

    def stop(grid, materials, n_steps, **kwargs):
        captured.extend(kwargs["flux_monitors"])
        raise ReachedStepper

    monkeypatch.setattr(runner, "run_nonuniform", stop)
    with pytest.raises(ReachedStepper):
        runner.run_nonuniform_path(sim, n_steps=1, compute_s_params=False)
    monitor, = captured
    return monitor


@pytest.mark.parametrize("axis", list("xyz"))
@pytest.mark.parametrize("graded", [False, True])
def test_metadata_matches_consumed_windows_without_strict_findings(axis, graded, monkeypatch, capsys):
    sim = _sim(graded=graded)
    if graded:
        # Independent cumulative geometry: edges 8, 9.125, 10.375, 11.5,
        # 12.5 U. These endpoints bracket cells 8:12 and 6:14 exactly.
        _add(sim, axis, size=(4.5 * U, 8.5 * U), center=(10.25 * U,) * 2)
        expected_bounds = np.array([[8, 12.5], [6, 14.5]]) * U
        expected_normal = 10.375 * U
    else:
        _add(sim, axis)
        expected_bounds = np.array([[8, 12], [6, 14]]) * U
        expected_normal = 10 * U
    report = _report(sim, strict=True)
    assert not report and report == [] and report.ok
    assert not report.issues and not report.errors and not report.warnings and not report.infos
    assert report.raise_for_failure() is report
    record, = report.flux_regions
    assert record["lane"] == ("nonuniform" if graded else "uniform")
    assert record["tangential_axes"] == [a for a in "xyz" if a != axis]
    assert not record["clamped"]
    np.testing.assert_array_equal(record["requested_bounds_m"], expected_bounds)
    np.testing.assert_array_equal(record["realized_bounds_m"], expected_bounds)
    assert record["requested_coordinate_m"] == 10 * U
    assert record["realized_coordinate_m"] == expected_normal
    assert record["cell_slices"] == [[16, 20], [14, 22]]
    if graded:
        monitor = _nu_runner_monitor(sim, monkeypatch)
    else:
        monitor, = build_flux_monitor_cfgs(sim, sim._build_realized_grid(), 1)
    assert record["normal_index"] == monitor.index
    assert record["cell_slices"] == [[monitor.lo1, monitor.hi1], [monitor.lo2, monitor.hi2]]
    # Verify the physical area independently of the shared region resolver.
    expected_area = np.prod(expected_bounds[:, 1] - expected_bounds[:, 0])
    area = np.broadcast_to(np.asarray(monitor.dA), monitor.e1_dft.shape[1:]).sum()
    assert area == pytest.approx(expected_area, rel=1e-6)
    assert json.loads(report.to_json())["flux_regions"] == report.flux_regions
    assert "FLUX REGION" in report.format()
    printed = capsys.readouterr().out
    assert "[FLUX REGION]" in printed and "requested" in printed and "realized" in printed


@pytest.mark.parametrize("graded", [False, True])
def test_clamping_is_an_advisory_with_metres_and_strict_compatibility(graded, monkeypatch):
    sim = _sim(graded=graded)
    _add(sim, size=(60 * U, 8 * U))
    report = _report(sim)
    record, = report.flux_regions
    issue, = report.by_code("flux_region_clamped")
    assert report.ok and report.raise_for_failure() is report
    assert issue.severity == "warning" and issue.loc == "window"
    assert str(issue) == flux_region_message(record)
    assert record["clamped_ends"] == [[True, True], [False, False]]
    assert record["requested_bounds_m"][0] == [-20 * U, 40 * U]
    assert record["realized_bounds_m"][0] == [0, (20.5 if graded else 20) * U]
    with pytest.warns(UserWarning, match="CLAMPED"):
        monitor = (_nu_runner_monitor(sim, monkeypatch) if graded else
                   build_flux_monitor_cfgs(sim, sim._build_realized_grid(), 1)[0])
    assert record["cell_slices"] == [[monitor.lo1, monitor.hi1], [monitor.lo2, monitor.hi2]]
    with pytest.raises(ValueError, match="strict.*1 issue"):
        _report(sim, strict=True)


@pytest.mark.parametrize("overflow", [0., 1e-14])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_decimal_boundary_roundoff_is_distinct_from_physical_overflow(overflow, nonuniform):
    profiles = {f"d{axis}_profile": np.full(20, .001) for axis in "xyz"} if nonuniform else {}
    sim = Simulation(domain=(.020,) * 3, freq_max=1e9, dx=.001,
                     cpml_layers=8, boundary="cpml", **profiles)
    sim.add_source((.005,) * 3, "ez", amplitude_kind="current")
    lo, hi = .015, .020 + overflow
    sim.add_flux_monitor(axis="x", coordinate=.010, name="decimal",
                         size=(hi - lo, .004), center=((lo + hi) / 2, .010), n_freqs=2)
    report = _report(sim, strict=not overflow)
    record, = report.flux_regions
    assert record["cell_slices"] == [[23, 28], [16, 20]]
    np.testing.assert_allclose(record["realized_bounds_m"], [[.015, .020], [.008, .012]],
                               rtol=0, atol=1e-17)
    assert record["clamped"] == bool(overflow)
    assert len(report.by_code("flux_region_clamped")) == bool(overflow)
    if overflow:
        with pytest.raises(ValueError, match="strict.*1 issue"):
            _report(sim, strict=True)


@pytest.mark.parametrize("overrides,match", [
    ({"size": (0, U)}, "positive"),
    ({"size": (U,)}, "exactly two"),
    ({"center": (np.nan, U)}, "finite"),
    ({"size": (.1 * U, .1 * U)}, "no interior cell"),
    ({"center": (-10 * U, 10 * U)}, "no interior cell"),
])
def test_preflight_rejects_invalid_private_entries_without_losing_valid_metadata(overrides, match):
    sim = _sim()
    _add(sim)
    # Exercise restored/direct entries which bypass public registration guards.
    sim._flux_monitors.append(replace(sim._flux_monitors[0], name="bad", **overrides))
    report = _report(sim)
    issue, = report.by_code("flux_region_invalid")
    assert not report.ok and issue.severity == "error" and issue.loc == "bad"
    assert match in str(issue)
    assert [r["name"] for r in report.flux_regions] == ["window"]
    with pytest.raises(ValueError, match=match):
        report.raise_for_failure()
    with pytest.raises(ValueError, match=match):
        _report(sim, strict=True)


def test_full_plane_and_empty_report_keep_existing_serialization_and_list_contract():
    sim = _sim()
    _add(sim, size=None)
    report = _report(sim, strict=True)
    assert report.flux_regions == [] and not report
    assert report.to_dict() == {"ok": True, "n_issues": 0, "n_errors": 0, "issues": []}
    assert report.format() == "preflight: PASS (no issues)"
    copied = PreflightReport([PreflightIssue("existing", code="old")])
    assert copied == ["existing"] and "flux_regions" not in copied.to_dict()
    report.flux_regions.append({"name": "local"})
    assert copied.flux_regions == []


def test_metadata_recomputes_after_monitor_change():
    sim = _sim()
    _add(sim)
    first = _report(sim, strict=True)
    sim._flux_monitors[0] = replace(sim._flux_monitors[0], center=(12 * U, 10 * U))
    second = _report(sim, strict=True)
    assert first.flux_regions[0]["realized_bounds_m"][0] == [8 * U, 12 * U]
    assert second.flux_regions[0]["realized_bounds_m"][0] == [10 * U, 14 * U]


def test_sparameter_report_retains_included_general_metadata():
    sim = _sim()
    _add(sim)
    expected = _report(sim).flux_regions
    report = sim.preflight_sparameters(include_general=True)
    assert report.flux_regions == expected
    assert report.to_dict()["flux_regions"] == expected
    assert not report.by_code("flux_region_clamped")


def test_unevaluable_selected_grid_is_explicit_without_guessing_bounds(monkeypatch):
    sim = _sim(graded=True)
    _add(sim)

    def unavailable():
        raise NotImplementedError("test grid unavailable")

    monkeypatch.setattr(sim, "_build_realized_grid", unavailable)
    report = _report(sim)
    issue, = report.by_code("flux_region_unavailable")
    assert issue.loc == "window" and issue.severity == "warning"
    assert "test grid unavailable" in str(issue)
    assert not report.flux_regions and "flux_regions" not in report.to_dict()


def test_preflight_inside_material_trace_keeps_geometry_concrete_and_tape_live():
    sim = _sim(graded=True)
    _add(sim, size=(4.5 * U, 8.5 * U), center=(10.25 * U,) * 2)
    expected = _report(sim, strict=True).flux_regions

    def material_objective(eps):
        # forward's eps_override is independent of the static declaration;
        # this checks its outer jit/grad context, without stepping fields.
        report = _report(sim, strict=True)
        assert report.flux_regions == expected
        return jnp.sum(eps**2)

    eps = jnp.array([1.5, 2.0, 2.5], dtype=jnp.float32)
    np.testing.assert_array_equal(jax.jit(jax.grad(material_objective))(eps), 2 * np.asarray(eps))


@pytest.mark.parametrize("target", ["mesh", "region"])
def test_traced_geometry_has_no_fabricated_metres(target):
    sim = _sim()
    _add(sim)

    def objective(value):
        # Isolate the new collector: other preflight validators have their own
        # traced-geometry contracts, unrelated to the finite-window metadata.
        if target == "mesh":
            sim._dx = value
        else:
            sim._flux_monitors[0] = replace(sim._flux_monitors[0], size=(value, U))
        report = PreflightReport()
        sim._collect_flux_regions(report)
        assert len(report.by_code("flux_region_unavailable")) == 1
        assert report.flux_regions == []
        return value**2

    assert jax.grad(objective)(jnp.float32(U)) == 2 * U
