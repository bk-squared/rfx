"""Numerical E-kernel and port-current comparisons."""


import jax
import jax.numpy as jnp
import pytest
import numpy as np
import sys

from rfx import Simulation, GaussianPulse
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.grid import Grid

from rfx.core import yee
import rfx.stepping.yee as stepping_yee  # the moved E kernels read curl_h here
from rfx.boundaries import upml
import rfx.measurement.graded_wire as graded_wire
from tests.contracts.curl_cases import model


def kernel_inputs():
    shape = (5, 4, 3)
    state = yee.init_state(shape)
    state = state._replace(hy=jnp.arange(60, dtype=jnp.float32).reshape(shape))
    return state, yee.init_materials(shape), yee.CurlBoundary(
        frozenset({"x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}), frozenset({"x_lo"}))


def _measure(row, quantity, measured, bar, **details):
    print(f"B1 {row}: {quantity}={measured:.12g} bar={bar:.12g} {details}", file=sys.stderr)


def _max_relative_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = np.divide(difference, np.abs(expected), out=np.zeros_like(difference),
                         where=expected != 0)
    relative[(expected == 0) & (difference != 0)] = np.inf
    return float(np.max(relative))


@pytest.mark.parametrize("branch", ["plain", "aniso", "inverse", "fast_e", "fast_he", "box", "upml"])
def test_e_kernels_match_plain_pmc_update(branch):
    state, mats, boundary = kernel_inputs()
    dt, dx = 1e-12, 1e-3
    high_wall = (False, False, False)
    if branch == "upml":
        grid = Grid(freq_max=20e9, domain=(4e-3, 3e-3, 2e-3), dx=dx,
                    cpml_layers=0, cpml_axes="", pec_faces=boundary.pec_faces,
                    pmc_faces=boundary.pmc_faces)
        assert grid.shape == state.ex.shape
        dt = grid.dt
    expected = stepping_yee.update_e(state, mats, dt, dx, boundary=boundary, high_wall=high_wall)
    plain_wall = stepping_yee.update_e(state, mats, dt, dx,
                                     boundary=yee.CurlBoundary(), high_wall=high_wall)
    separation = float(np.max(np.abs(np.asarray(expected.ez[0] - plain_wall.ez[0]))))
    _measure(branch, "PMC-minus-default", separation, 0)
    assert separation > 0, "PMC x_lo must change the ramp's E update"
    coeffs = yee.precompute_coeffs(mats, dt, dx, high_wall=high_wall)
    ca, cb = yee.e_component_coeffs(mats, dt, (False,)*3, high_wall=high_wall)
    calls = {
        "plain": lambda: expected,
        "aniso": lambda: stepping_yee.update_e_aniso(state, mats, mats.eps_r, mats.eps_r,
            mats.eps_r, dt, dx, boundary=boundary, high_wall=high_wall),
        "inverse": lambda: stepping_yee.update_e_aniso_inv(state, mats, 1/mats.eps_r,
            1/mats.eps_r, 1/mats.eps_r, dt, dx, boundary=boundary, high_wall=high_wall),
        "fast_e": lambda: stepping_yee.update_e_fast(state, coeffs.ca_ex, coeffs.ca_ey,
            coeffs.ca_ez, coeffs.cb_ex, coeffs.cb_ey, coeffs.cb_ez, boundary=boundary),
        # E is zero initially, so the preceding H half-step leaves H unchanged.
        "fast_he": lambda: yee.update_he_fast(state, coeffs, boundary=boundary),
        # The grid-wide update above is the first update; redo the entire box.
        "box": lambda: stepping_yee.update_e_box(expected, state, (0, 5, 0, 4, 0, 3),
            ca, cb, dx, boundary=boundary),
    }
    if branch == "upml":
        calls[branch] = lambda: upml.apply_upml_e(state, upml.init_upml(grid, mats), boundary=boundary)
    actual = calls[branch]()
    # Repository kernel-family bar (arithmetic order differs between adapters):
    # tests/unit/geometry/test_update_e_aniso_inv.py:69-86; Design A28.
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        got, want = np.asarray(getattr(actual, component)), np.asarray(getattr(expected, component))
        error = _max_relative_error(got, want)
        _measure(f"{branch}/{component}", "max-element-relative-error", error,
                 0 if branch == "plain" else 5e-6, atol=0 if branch == "plain" else 1e-7,
                 mesh=state.ex.shape, steps=1)
        if branch == "plain":
            assert got.tobytes() == want.tobytes(), f"{branch}/{component}: not bitwise equal"
        else:
            np.testing.assert_allclose(got, want, rtol=5e-6, atol=1e-7,
                                       err_msg=f"{branch}/{component}")


def _random_h():
    state, _, boundary = kernel_inputs()
    rng = np.random.default_rng(692)
    state = state._replace(**{c: jnp.asarray(rng.standard_normal(state.ex.shape), dtype=jnp.float32)
                              for c in ("hx", "hy", "hz")})
    return state, boundary


def _sample_equal(row, actual, expected):
    actual, expected = float(actual), float(expected)
    assert np.isfinite(actual) and np.isfinite(expected) and expected != 0
    error = abs(actual - expected) / abs(expected)
    _measure(row, "relative-error", error, 1e-5, actual=actual, expected=expected,
             mesh=(5, 4, 3), steps=0)
    assert error <= 1e-5, f"{row}: current={actual}, curl*area={expected}, relative error={error} > 1e-5"


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("idx", [(0, 0, 0), (2, 1, 1), (4, 3, 2)])
@pytest.mark.parametrize("periodic", [(False, False, False), (True, True, True)])
def test_uniform_port_current_matches_curl_area(component, idx, periodic):
    from rfx.probes.probes import _ampere_loop
    from rfx.measurement.projector import port_table, project
    state, boundary = _random_h()
    boundary = boundary._replace(periodic=periodic)
    dx = 1e-3
    grid = Grid(freq_max=20e9, domain=(4e-3, 3e-3, 2e-3), dx=dx,
                cpml_layers=0, cpml_axes="", pec_faces=boundary.pec_faces,
                pmc_faces=boundary.pmc_faces)
    assert grid.shape == state.ex.shape
    table = port_table(grid, ((component, (idx,), "lumped", idx, 0),),
                       nu=False, boundary=boundary, dtype=np.float32)
    curl = yee.curl_h(state.hx, state.hy, state.hz, dx, periodic, boundary=boundary)
    expected = float(curl["xyz".index(component[1])][idx]) * dx**2
    row = f"uniform/{component}/{idx}/{periodic[0]}"
    _sample_equal(row + "/ampere", _ampere_loop(state, idx, component, dx, periodic, boundary), expected)
    # Single-device steps and public port_current sample these compiled owners.
    _sample_equal(row + "/projector", project(state, table)[1], expected)
    if idx == (0, 0, 0):
        alternate = yee.curl_h(state.hx, state.hy, state.hz, dx, (False,)*3,
                               boundary=yee.CurlBoundary())
        if periodic[0] or component != "ex":
            separation = abs(float(alternate["xyz".index(component[1])][idx]) * dx**2 - expected)
            _measure(row, "boundary-separation", separation, abs(expected)*1e-5)
            assert separation > abs(expected)*1e-5


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("idx", [(0, 0, 0), (2, 1, 1), (4, 3, 2)])
def test_graded_wire_current_matches_curl_area(component, idx):
    state, boundary = _random_h()
    duals = (.9e-3, 1.1e-3, 1.3e-3)
    inverse = tuple(jnp.full(n, 1/d, dtype=jnp.float32) for n, d in zip(state.ex.shape, duals))
    curl = yee.curl_h_nu(state.hx, state.hy, state.hz, *inverse, boundary=boundary)
    axis = "xyz".index(component[1])
    area = np.prod([d for a, d in enumerate(duals) if a != axis])
    expected = float(curl[axis][idx]) * area
    actual = graded_wire.wire_port_current(state.hx, state.hy, state.hz, component, *idx,
                                           *duals, boundary=boundary)
    _sample_equal(f"graded/{component}/{idx}", actual, expected)
    if idx == (0, 0, 0) and component != "ex":
        zero = yee.curl_h_nu(state.hx, state.hy, state.hz, *inverse, boundary=yee.CurlBoundary())
        separation = abs(float(zero[axis][idx])*area - expected)
        _measure(component, "graded-boundary-separation", separation, abs(expected)*1e-5)
        assert separation > abs(expected)*1e-5


def _port_model(mesh, feature):
    profile = np.r_[1e-3, np.linspace(.87, 1.13, 11)*1e-3, 1e-3]
    sim = Simulation(freq_max=20e9, domain=(13e-3, 10e-3, 8e-3), dx=1e-3,
        boundary=BoundarySpec(x=Boundary(lo="pmc", hi="pec"), y="pec", z="pec"),
        cpml_layers=0, **({"dx_profile": profile} if mesh == "graded" else {}))
    sim.add_port((0., 3e-3, 2e-3), "ez", impedance=50.,
                 waveform=GaussianPulse(f0=12e9, bandwidth=1.2),
                 **({"extent": 2e-3} if feature == "wire" else {}))
    sim.add_probe((4e-3, 3e-3, 2e-3), "ez")
    return sim


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("feature", ["lumped", "wire"])
def test_port_records_match_forward_and_final_ampere_loop(mesh, feature):
    sim = _port_model(mesh, feature)
    grid = sim.freeze_mesh()
    bins = np.array([8e9, 12e9, 16e9], dtype=np.float32)
    run = sim.run(n_steps=200, skip_preflight=True, compute_s_params=True, s_param_freqs=bins)
    forward = sim.forward(n_steps=200, skip_preflight=True, checkpoint=False, port_s11_freqs=bins)
    # Existing pair bars: tests/contracts/test_port_projector.py:243 (S11),
    # tests/contracts/test_run_port_records.py:39 (uniform/graded lumped/wire records).
    # Both compare exactly; keep that bar rather than introducing a looser one.
    assert np.isfinite(np.asarray(run.s_params)).all()
    assert np.isfinite(np.asarray(forward.s_params)).all()
    assert np.max(np.abs(np.asarray(run.s_params))) > 0
    error = float(np.max(np.abs(np.asarray(run.s_params) - np.asarray(forward.s_params))))
    row = f"{mesh}/{feature}"
    _measure(row, "S11-max-difference", error, 0, mesh=grid.shape, steps=200)
    np.testing.assert_array_equal(run.s_params, forward.s_params, err_msg=row + " S11")
    for lane, result in (("run", run), ("forward", forward)):
        record, = result.sparam_time_records
        assert np.max(np.abs(np.asarray(record["I"]))) > 0
        # Ez contour at the port midpoint. These indices are read from the
        # realized public grid, not from the measurement table being checked.
        position = (0., 3e-3, (3 if feature == "wire" else 2)*1e-3)
        idx = tuple(int(grid.index_of(a, p)) for a, p in enumerate(position))
        i, j, k = idx
        assert i == 0, (grid.shape, idx)
        # ForwardResult publishes records but no final state. Both records
        # are checked against the run's final H state.
        hx, hy = np.asarray(run.state.hx), np.asarray(run.state.hy)
        # The x_lo PMC image is odd for tangential H: form the ghost
        # explicitly, without consulting a curl helper or measurement owner.
        hy_back = -hy[idx]
        hx_back = 0. if j == 0 else hx[i, j-1, k]
        # Compute dual edge lengths from realized primal cells, independently.
        widths = [np.asarray(grid.cells(a)) for a in (0, 1)]
        dx, dy = [float(w[p] if p == 0 else (w[p-1] + w[p])*.5)
                  for w, p in zip(widths, (i, j))]
        terms = np.array([hy[idx]*dy, -hy_back*dy, -hx[idx]*dx, hx_back*dx], dtype=np.float32)
        expected = float(np.sum(terms, dtype=np.float32))
        actual = float(record["I"][-1])
        # Existing table-versus-loop rounding bar:
        # tests/contracts/test_port_projector.py:94 (9 float32 peak ULP).
        bar = 9*float(np.spacing(np.float32(abs(expected))))
        zero_ghost_terms = terms.copy()
        zero_ghost_terms[1] = 0.
        zero_ghost_loop = float(np.sum(zero_ghost_terms, dtype=np.float32))
        ghost_difference = abs(expected - zero_ghost_loop)
        _measure(row + "/" + lane, "zero-ghost-loop-difference", ghost_difference, bar,
                 ghost_term=abs(float(terms[1])), zero_ghost_loop=zero_ghost_loop)
        assert ghost_difference > bar, f"{row}/{lane}: PMC ghost contribution {ghost_difference} <= {bar}"
        difference = abs(actual - expected)
        _measure(row + "/" + lane, "final-current-difference", difference, bar,
                 actual=actual, expected=expected, mesh=grid.shape, idx=idx, steps=200)
        assert difference <= bar, f"{row}/{lane}: current={actual}, hand loop={expected}, error={difference} > {bar}"
    run_record, forward_record = run.sparam_time_records[0], forward.sparam_time_records[0]
    record_error = float(np.max(np.abs(np.asarray(run_record.samples) - np.asarray(forward_record.samples))))
    _measure(row, "record-max-difference", record_error, 0, mesh=grid.shape, steps=200)
    np.testing.assert_array_equal(run_record.samples, forward_record.samples, err_msg=row + " records")


def test_gpu_backend_query_preserves_pec_and_pmc_records(monkeypatch):
    records = {}
    for wall in ("pec", "pmc"):
        cpu = model(wall=wall, solved_domain=True).run(n_steps=40, skip_preflight=True, compute_s_params=False)
        with monkeypatch.context() as environment:
            environment.setattr(jax, "default_backend", lambda: "gpu")
            gpu = model(wall=wall, solved_domain=True).run(n_steps=40, skip_preflight=True, compute_s_params=False)
        error = float(np.max(np.abs(np.asarray(cpu.time_series) - np.asarray(gpu.time_series))))
        _measure(wall, "backend-record-difference", error, 0, mesh=cpu.state.ex.shape, steps=40)
        np.testing.assert_array_equal(cpu.time_series, gpu.time_series, err_msg=wall + " backend query")
        records[wall] = np.asarray(cpu.time_series)
    separation = float(np.max(np.abs(records["pmc"] - records["pec"])))
    bar = 9*float(np.spacing(np.float32(np.max(np.abs(records["pmc"])))))
    _measure("backend", "PMC-minus-PEC", separation, bar)
    assert separation > bar, "PMC and PEC records must separate"
