"""Runtime routing contract for B3a; no inspection of implementation text.

Each adapter must reach the shared curl with the step's resolved boundary.
The inline negative control also checks that disabling the assertion is red.
Numerical byte comparisons against 5c477d59 live in the B3a lane records.
"""
from contextlib import ExitStack
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from rfx.core import yee
from rfx.materials import debye, lorentz
from rfx.boundaries import upml
import rfx.simulation as uniform
import rfx.nonuniform as nu
from tests.contracts.curl_cases import model


class Reached(Exception):
    pass


def require_shared(call, *, helper="curl_h", indexed=False, minimum_calls=1):
    """Observe the actual call during execution/tracing, including JIT bodies."""
    seen = []
    original = getattr(yee, helper)

    def witness(*args, **kwargs):
        if indexed and kwargs.get("index") is None:
            return original(*args, **kwargs)
        boundary = kwargs.get("boundary")
        assert isinstance(boundary, yee.CurlBoundary), "realized boundary missing"
        seen.append(boundary)
        if len(seen) >= minimum_calls:
            raise Reached
        return original(*args, **kwargs)

    # ADE/UPML historically imported the helper into their module namespace.
    jax.clear_caches()
    with ExitStack() as stack:
        for module in (yee, uniform, debye, lorentz, upml):
            if hasattr(module, helper):
                stack.enter_context(patch.object(module, helper, witness))
        try:
            call()
        except Reached:
            pass
    assert len(seen) >= minimum_calls, f"E update bypassed shared {helper}"
    return seen[0]


@pytest.mark.parametrize("lane", ["run", "forward"])
@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("feature", ["plain", "lumped", "wire", "msl", "waveguide",
                                      "debye", "lorentz", "mixed"])
def test_simulation_uses_realized_curl(lane, mesh, feature):
    sim = model(feature, mesh)
    kwargs = dict(n_steps=2, skip_preflight=True)
    if lane == "run":
        kwargs["compute_s_params"] = False
    boundary = require_shared(lambda: getattr(sim, lane)(**kwargs),
                              helper="curl_h_nu" if mesh == "graded" else "curl_h")
    if feature not in ("msl", "waveguide"):
        assert boundary.pmc_faces == frozenset({"x_lo"})
        assert "x_hi" in boundary.pec_faces
        assert "x_lo" not in boundary.pec_faces


@pytest.mark.parametrize("lane", ["run", "forward"])
@pytest.mark.parametrize("mode", ["3d", "2d_tmz", "2d_tez"])
@pytest.mark.parametrize("wall", ["pec", "cpml", "periodic"])
def test_modes_and_periodicity(lane, mode, wall):
    sim = model(mode=mode, wall=wall)
    kwargs = dict(n_steps=2, skip_preflight=True)
    if lane == "run":
        kwargs["compute_s_params"] = False
    boundary = require_shared(lambda: getattr(sim, lane)(**kwargs))
    assert boundary.periodic[0] == (wall == "periodic")
    assert boundary.periodic[2] == (mode != "3d")


@pytest.mark.parametrize("feature,mesh", [("subpixel", "uniform"), ("subpixel", "graded"),
                                           ("conformal", "uniform"),
                                           ("tfsf", "uniform"), ("tfsf", "graded")])
def test_special_run_branches(feature, mesh):
    sim = model(feature, mesh)
    kwargs = dict(n_steps=2, skip_preflight=True, compute_s_params=False)
    if feature == "subpixel":
        kwargs["subpixel_smoothing"] = True
    require_shared(lambda: sim.run(**kwargs),
                   helper="curl_h_nu" if mesh == "graded" else "curl_h")


@pytest.mark.parametrize("mesh,feature", [("uniform", "lumped"), ("uniform", "wire"),
                                         ("graded", "wire")])
def test_scan_port_current_uses_boundary_neighbour(mesh, feature):
    sim = model(feature, mesh)
    if mesh == "graded":
        call = lambda: sim.run(n_steps=2, skip_preflight=True, compute_s_params=True,
                               s_param_freqs=jnp.array([12e9]))
    else:
        call = lambda: sim.forward(n_steps=2, skip_preflight=True,
                                   port_s11_freqs=jnp.array([12e9]))
    require_shared(call, helper="h_neighbor", indexed=True)


def kernel_inputs():
    shape = (5, 4, 3)
    state = yee.init_state(shape)
    state = state._replace(hy=jnp.arange(60, dtype=jnp.float32).reshape(shape))
    return state, yee.init_materials(shape), yee.CurlBoundary(
        frozenset({"x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}), frozenset({"x_lo"}))


@pytest.mark.parametrize("branch", ["plain", "aniso", "inverse", "fast_e", "fast_he", "box", "upml"])
def test_kernel_adapters(branch):
    state, mats, boundary = kernel_inputs()
    dt, dx = 1e-12, 1e-3
    coeffs = yee.precompute_coeffs(mats, dt, dx)
    calls = {
        "plain": lambda: yee.update_e(state, mats, dt, dx, boundary=boundary),
        "aniso": lambda: yee.update_e_aniso(state, mats, mats.eps_r, mats.eps_r,
                                           mats.eps_r, dt, dx, boundary=boundary),
        "inverse": lambda: yee.update_e_aniso_inv(state, mats, mats.eps_r, mats.eps_r,
                                                  mats.eps_r, dt, dx, boundary=boundary),
        "fast_e": lambda: yee.update_e_fast(state, coeffs.ca_ex, coeffs.ca_ey, coeffs.ca_ez,
                                            coeffs.cb_ex, coeffs.cb_ey, coeffs.cb_ez,
                                            boundary=boundary),
        "fast_he": lambda: yee.update_he_fast(state, coeffs, boundary=boundary),
        "box": lambda: yee.update_e_box(state, state, (1, 2, 1, 2, 1, 2),
                                         1., .1, dx, boundary=boundary),
    }
    if branch == "upml":
        from types import SimpleNamespace
        coeff = SimpleNamespace(inv_dx=1/dx, inv_dy=1/dx, inv_dz=1/dx)
        calls[branch] = lambda: upml.apply_upml_e(state, coeff, boundary=boundary)
    assert require_shared(calls[branch]) == boundary


def test_disabled_contract_is_detected():
    # An E update deliberately bypassing the helper. No source inspection and
    # no monkeypatch of the helpers: disabling require_shared's assertion
    # makes this negative-control test fail with "DID NOT RAISE".
    state, mats, boundary = kernel_inputs()
    def inline_update():
        curl_z = (state.hy - jnp.roll(state.hy, 1, axis=0)) / 1e-3
        return state._replace(ez=state.ez + 1e-12 / yee.EPS_0 * curl_z)
    with pytest.raises(AssertionError, match="bypassed shared"):
        require_shared(inline_update)


def test_cpml_uses_boundary_neighbours():
    from rfx.boundaries.cpml import init_cpml, apply_cpml_e
    sim = model(wall="cpml")
    grid = sim._build_grid()
    params, psi = init_cpml(grid)
    from rfx.boundaries.pec import resolve_wall_faces
    boundary = yee.CurlBoundary(*resolve_wall_faces(grid, (False, False, False)))
    import rfx.boundaries.cpml as cpml
    original = cpml.h_neighbor
    def sentinel(*args, **kwargs):
        assert kwargs["boundary"] == boundary
        raise Reached
    with patch.object(cpml, "h_neighbor", sentinel), pytest.raises(Reached):
        apply_cpml_e(yee.init_state(grid.shape), params, psi, grid, boundary=boundary)
    assert cpml.h_neighbor is original


@pytest.mark.parametrize("graded", [False, True])
def test_design_box_second_update_uses_boundary(graded):
    state, mats, boundary = kernel_inputs()
    inv_d = tuple(jnp.ones(n) / 1e-3 for n in state.ex.shape) if graded else None
    require_shared(lambda: yee.update_e_box(
        state, state, (1, 2, 1, 2, 1, 2), 1., .1, 1e-3,
        inv_d=inv_d, boundary=boundary), helper="curl_h_nu" if graded else "curl_h")


def test_coaxial_injection_parent_uses_shared_curl():
    from rfx.sources.coaxial_port import CoaxialPort, build_coaxial_tem_plane_source_specs
    from rfx import GaussianPulse
    sim = model(wall="pec")
    grid = sim._build_grid()
    port = CoaxialPort(position=(6e-3, 5e-3, 4e-3), face="top",
                       pin_length=2e-3, impedance=50.0, pin_radius=.5e-3,
                       outer_radius=2e-3, excitation=GaussianPulse(f0=12e9, bandwidth=1.2))
    sources = build_coaxial_tem_plane_source_specs(grid=grid, port=port, n_steps=2)
    require_shared(lambda: uniform.run(
        grid, yee.init_materials(grid.shape), 2,
        sources=list(sources.electric_sources), mag_sources=list(sources.magnetic_sources)))


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("lane", ["run", "forward"])
def test_cpml_driver_passes_realized_boundary(mesh, lane):
    from rfx.boundaries import cpml
    sim = model(mesh=mesh, wall="cpml")
    def correction(*args, **kwargs):
        assert isinstance(kwargs.get("boundary"), yee.CurlBoundary)
        raise Reached
    kwargs = dict(n_steps=2, skip_preflight=True)
    if lane == "run":
        kwargs["compute_s_params"] = False
    jax.clear_caches()
    with patch.object(cpml, "apply_cpml_e", correction), pytest.raises(Reached):
        getattr(sim, lane)(**kwargs)


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("lane", ["run", "forward"])
def test_sheet_replacement_update_also_uses_shared_curl(mesh, lane):
    from rfx import Box
    sim = model(mesh=mesh, wall="pec")
    sim.add_thin_conductor(Box((2e-3, 2e-3, 4e-3), (8e-3, 6e-3, 4e-3)),
                           sigma_bulk=1e4, surface_impedance_f0=12e9)
    kwargs = dict(n_steps=2, skip_preflight=True)
    if lane == "run":
        kwargs["compute_s_params"] = False
    require_shared(lambda: getattr(sim, lane)(**kwargs), minimum_calls=2,
                   helper="curl_h_nu" if mesh == "graded" else "curl_h")


def test_tfsf_forward_uses_shared_curl():
    sim = model("tfsf")
    boundary = require_shared(lambda: sim.forward(n_steps=2, skip_preflight=True))
    assert boundary.periodic == (False, True, True)


def test_baked_dispatch_passes_realized_boundary_on_cpu():
    # Exercise the production dispatch predicate and baked implementation on
    # CPU. This is not a native GPU measurement.
    sim = model(wall="pec")
    invoked = []
    original = uniform.update_he_fast
    def baked(*args, **kwargs):
        invoked.append(True)
        return original(*args, **kwargs)
    with patch.object(jax, "default_backend", lambda: "gpu"), \
            patch.object(uniform, "update_he_fast", baked):
        boundary = require_shared(lambda: sim.run(
            n_steps=2, skip_preflight=True, compute_s_params=False))
    assert invoked
    assert boundary.pec_faces == frozenset(
        f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi"))
