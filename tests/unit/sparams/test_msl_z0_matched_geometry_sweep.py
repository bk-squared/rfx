"""Build-only contracts for the fresh #752 foil/port geometry and recorder."""

import builtins
import functools
import importlib.util
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

import rfx.simulation as engine
from rfx.geometry.rasterize_grid import coords_from_uniform_grid


DRIVER = (
    Path(__file__).resolve().parents[3]
    / "scripts/diagnostics/msl_z0_matched_geometry_sweep.py"
)
LABELS = ("h3", "h4", "h5", "h6", "dx80", "dx60")


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location("fresh_msl_sweep_test", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def forbid_fields_and_historical_inputs(monkeypatch):
    # Preserve the real signature: run_case binds the actual engine arguments
    # before dry_run stops. A missing dry-run stop must never execute FDTD.
    for name in ("run", "run_until_decay"):
        original = getattr(engine, name)

        @functools.wraps(original)
        def forbidden(*args, **kwargs):
            pytest.fail("this build-only test attempted field evolution")

        monkeypatch.setattr(engine, name, forbidden)
    protected = {
        "msl_z0_bias_floor_sweep.json",
        "msl_z0_bias_floor_sweep_realized_anchor.json",
    }
    path_open, builtin_open = Path.open, builtins.open

    def check(path):
        if isinstance(path, (str, bytes, Path)):
            assert Path(path).name not in protected, (
                "fresh geometry read historical numerical inputs"
            )

    def guarded_path(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def guarded_builtin(path, *args, **kwargs):
        check(path)
        return builtin_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path)
    monkeypatch.setattr(builtins, "open", guarded_builtin)


@pytest.mark.parametrize("label", LABELS)
def test_each_case_coincides_on_actual_foil_edges_material_interface_and_port_planes(
    driver, label
):
    assert {label for label, _ in driver.CASES} == set(LABELS)
    sim, plan = driver.build_case(label)
    grid = sim._build_realized_grid()
    realized = sim._assemble_realized(grid, nonuniform=False)
    nodes = coords_from_uniform_grid(grid)
    eps = np.asarray(realized.materials.eps_r)
    ey = np.asarray(realized.edges[1])
    assert realized.pec_mask is None and len(realized.sheets) == 1
    assert "z_lo" in grid.pec_faces and grid.pad_z_lo == 0
    i = (grid.pad_x_lo + grid.shape[0] - grid.pad_x_hi - 1) // 2
    planes = np.flatnonzero(ey[i].any(axis=0))
    assert len(planes) == 1
    k = int(planes[0])
    width_edges = np.flatnonzero(ey[i, :, k])
    assert len(width_edges) > 0 and np.all(np.diff(width_edges) == 1)
    # Ey owns intervals, unlike the inclusive longitudinal Ex node rows.
    actual_width = np.diff(nodes.y)[width_edges].sum()
    actual_height = nodes.z[k] - nodes.z[0]
    assert plan["trace_thickness_m"] == 0.0
    assert plan["width_m"] == pytest.approx(actual_width, abs=1e-16)
    assert plan["height_m"] == pytest.approx(actual_height, abs=1e-16)
    assert plan["width_intervals"] == len(width_edges)
    assert plan["height_intervals"] == k
    # Nearest realizable dimensions: the half-cell geometric bound follows
    # from the declared integer-node design; no measured-Z0 numbers enter it.
    assert abs(actual_width - 600e-6) <= grid.dx / 2 + 1e-16
    assert abs(actual_height - 254e-6) <= grid.dx / 2 + 1e-16
    for pe in sim._msl_ports:
        ip, jp, kp = grid.position_to_index(pe.position)
        assert kp == 0 and nodes.x[ip] == pytest.approx(pe.position[0], abs=1e-16)
        np.testing.assert_array_equal(np.flatnonzero(ey[ip, :, k]), width_edges)
        assert pe.height == pytest.approx(actual_height, abs=1e-16)
        assert pe.width == pytest.approx(actual_width, abs=1e-16)
        assert pe.position[2] + pe.height == pytest.approx(nodes.z[k], abs=1e-16)
        assert pe.impedance == 50.0 and pe.eps_r_sub == 3.66
        np.testing.assert_array_equal(eps[ip, jp, :k], np.full(k, np.float32(3.66)))
        np.testing.assert_array_equal(eps[ip, jp, k:], np.ones(eps.shape[2] - k))
    assert [pe.direction for pe in sim._msl_ports] == ["+x", "-x"]
    assert np.isfinite(
        [plan["repo_simplified_reference_ohm"], plan["hj1980_reference_ohm"]]
    ).all()


@pytest.mark.parametrize("lower", [1, 2, 7, 8, 31])
def test_integer_width_half_ties_select_lower_with_correct_neighbors(driver, lower):
    tie = lower + 0.5
    assert driver.nearest_lower_tie(tie) == lower
    assert driver.nearest_lower_tie(np.nextafter(tie, -np.inf)) == lower
    assert driver.nearest_lower_tie(np.nextafter(tie, np.inf)) == lower + 1


def test_hj_vacuum_special_point_and_narrow_strip_limit(driver):
    eta = 376.730313668
    # At u=30.666 the HJ geometry correction exponential is exactly exp(-1).
    u = 30.666
    fu = 6 + (2 * math.pi - 6) / math.e
    expected = eta / (2 * math.pi) * math.log(fu / u + math.hypot(1.0, 2 / u))
    z, ee = driver.hj1980(u * 1e-3, 1e-3, 1.0)
    assert ee == 1.0 and z == pytest.approx(expected, rel=2e-14)
    # Independent narrow-strip asymptote: eta/(2pi) log(8h/w). This rejects
    # accidentally substituting the repository's approximate coefficient 60.
    z, ee = driver.hj1980(1e-6, 1e-3, 1.0)
    assert ee == 1.0
    assert z == pytest.approx(eta / (2 * math.pi) * math.log(8000), rel=1e-8)


@pytest.mark.parametrize("eps", [2.0, 3.66, 10.0])
def test_hj_material_dependence_and_length_scale_invariance(driver, eps):
    w, h = 600e-6, 254e-6
    vacuum_z, _ = driver.hj1980(w, h, 1.0)
    z, ee = driver.hj1980(w, h, eps)
    assert (eps + 1) / 2 < ee < eps
    assert z * math.sqrt(ee) == pytest.approx(vacuum_z, rel=2e-14)
    np.testing.assert_allclose(
        driver.hj1980(100 * w, 100 * h, eps), (z, ee), rtol=2e-14
    )


@pytest.mark.parametrize("mutation", ["foil-plane", "dielectric-interface"])
def test_builder_rejects_actual_geometry_mismatch_before_fields(
    driver, monkeypatch, mutation
):
    original = driver.Box
    altered, assembled = [], []
    dx = dict(driver.CASES)["h3"]
    original_assemble = driver.Simulation._msl_assemble_once

    def capture(sim):
        result = original_assemble(sim)
        assembled.append(result)
        return result

    def mismatch(lo, hi):
        lo, hi = list(lo), list(hi)
        if mutation == "foil-plane" and lo[2] == hi[2] and lo[2] > 0:
            lo[2] += dx
            hi[2] += dx
            altered.append(True)
        elif mutation == "dielectric-interface" and lo[2] == 0 and hi[2] > 0:
            # h-dx can round ABOVE 2*dx and retain the old half-open
            # dielectric row. Move to the actual integer-node coordinate.
            hi[2] = 2 * dx
            altered.append(True)
        return original(tuple(lo), tuple(hi))

    monkeypatch.setattr(driver, "Box", mismatch)
    monkeypatch.setattr(driver.Simulation, "_msl_assemble_once", capture)
    with pytest.raises(AssertionError):
        driver.build_case("h3")
    assert altered
    assert len(assembled) == 1
    grid, mats, _, _, realized = assembled[0]
    if mutation == "dielectric-interface":
        assert np.asarray(mats.eps_r)[grid.pad_x_lo + 1, grid.pad_y_lo + 1, 2] == 1.0
    else:
        actual_planes = np.flatnonzero(np.any(realized.edges[1], axis=(0, 1)))
        assert actual_planes.tolist() == [4]


def test_existing_output_refusal_precedes_build_or_field_execution(
    driver, tmp_path, monkeypatch
):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "keep.bin"
    sentinel.write_bytes(b"existing scientific record")
    monkeypatch.setattr(
        driver,
        "build_case",
        lambda *_: pytest.fail("output refusal must precede build"),
    )
    with pytest.raises(FileExistsError):
        driver.run_case("h3", out)
    assert sentinel.read_bytes() == b"existing scientific record"
    assert list(out.iterdir()) == [sentinel]


@pytest.mark.parametrize("mutation", [None, "eps_r", "mu_r", "pec", "n_steps"])
def test_actual_consumed_plan_guard_rejects_upstream_drift_without_fields(
    driver,
    tmp_path,
    monkeypatch,
    mutation,
):
    import rfx.runners.uniform as uniform

    signature = inspect.signature(engine.run)
    protected_engine = engine.run
    injected = []

    class UpstreamEngine:
        def __getattr__(self, name):
            return getattr(engine, name)

        def run(self, *args, **kwargs):
            # This is the ACTUAL runner handoff, upstream of the driver's
            # recorded guard. Do not change its stored plan/signature helper.
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            inp = bound.arguments
            grid = inp["grid"]
            index = (grid.pad_x_lo + 1, grid.pad_y_lo + 1, grid.pad_z_lo + 1)
            if mutation in ("eps_r", "mu_r"):
                mats = inp["materials"]
                value = getattr(mats, mutation).at[index].add(0.25)
                inp["materials"] = mats._replace(**{mutation: value})
            elif mutation == "pec":
                edges = list(inp["pec_edge_masks"])
                edges[0] = np.array(edges[0], copy=True)
                edges[0][index] = not edges[0][index]
                inp["pec_edge_masks"] = tuple(edges)
            elif mutation == "n_steps":
                inp["n_steps"] -= 1
            injected.append(True)
            return engine.run(*bound.args, **bound.kwargs)

    monkeypatch.setattr(uniform, "_simulation", UpstreamEngine())
    out = tmp_path / "fresh"
    if mutation is None:
        driver.run_case("h3", out, dry_run=True)
        plan = json.loads((out / "plan.json").read_text())
        calls = json.loads((out / "consumed-plans.json").read_text())
        assert len(calls) == 1 and calls[0]["n_steps"] == plan["n_steps"]
        assert calls[0]["sources"] and calls[0]["dft_planes"]
        assert not (out / "result.npz").exists()
    else:
        with pytest.raises(AssertionError):
            driver.run_case("h3", out, dry_run=True)
    assert len(injected) == 1
    assert engine.run is protected_engine  # restored after both success and refusal
