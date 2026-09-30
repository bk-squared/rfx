"""A 1 pF capacitor in a pulsed PEC box belongs to one E edge.

Compare each accepted permittivity lane with the ordinary uniform run.
Read all three permittivities at the capacitor node and the first E update
from an identical initial H field, before subsequent field propagation.
"""

from contextlib import ExitStack
import os
import sys
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx import simulation
from rfx import nonuniform as nu_core
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import EPS_0, component_e_materials, init_state
from rfx.runners import nonuniform as nu_runner


COMPONENTS = ("ex", "ey", "ez")
LANES = ("uniform", "smooth", "kottke", "conformal", "conformal_smooth",
         "uniform_dual", "nu", "nu_smooth", "nu_dual", "forward",
         "forward_occ", "materials_occ")
ACCEPTED = tuple(lane for lane in LANES if lane not in ("uniform_dual", "nu_dual"))
DX = 1e-3
CAPACITANCE = 1e-12
LOAD = (0.006, 0.006, 0.004)
FLOOR = 32 * np.finfo(np.float32).eps


def _model(lane, component=None, *, fractional=False, seeded=False):
    kwargs = {"dz_profile": np.full(11, DX)} if lane.startswith("nu") else {}
    if lane.endswith("dual"):
        kwargs["interface_eps"] = "dual_average"
    boundary = (BoundarySpec(x=Boundary(lo="pec", hi="pec", conformal=True),
                             y="pec", z="pec")
                if lane.startswith("conformal") else "pec")
    sim = Simulation(freq_max=10e9, domain=(0.009, 0.010, 0.011), dx=DX,
                     boundary=boundary, **kwargs)
    sim.add_material("diel", eps_r=4.0)
    sim.add(Box((-0.002,) * 3, (0.013,) * 3), material="diel")
    # Boundary(conformal=True) enters the conformal builder; the capacitor
    # is away from that wall, with w=1. A second wall at x=5.7 mm gives
    # fractional weights at the same off-centre capacitor node.
    if fractional:
        sim.add(Box((-0.003,) * 3, (0.0057, 0.014, 0.014)), material="pec")
    if not seeded:
        sim.add_source((0.003, 0.004, 0.005), "ez", amplitude_kind="field",
                       waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    if component is not None:
        sim.add_lumped_rlc(LOAD, component=component, C=CAPACITANCE,
                           topology="parallel")
    for comp in COMPONENTS:
        sim.add_probe(LOAD, comp)
    options = dict(subpixel_smoothing=("kottke_pec" if lane == "kottke"
                                      else "smooth" in lane),
                   conformal_pec=lane.startswith("conformal"))
    return sim, options


def _seed_h(shape, **kwargs):
    state = init_state(shape, **kwargs)
    rng = np.random.default_rng(1263)
    return state._replace(**{
        comp: jnp.asarray(rng.standard_normal(shape), dtype=state.ex.dtype)
        for comp in ("hx", "hy", "hz")})


def _run_case(lane, component=None, *, fractional=False, seeded=False):
    sim, options = _model(lane, component,
                          fractional=fractional and not lane.endswith("occ"), seeded=seeded)
    module = nu_runner if lane.startswith("nu") else simulation
    name = "run_nonuniform" if lane.startswith("nu") else "run"
    original = getattr(module, name)
    observed = {}

    def capture(grid, materials, n_steps, **kwargs):
        cell = (nu_core.position_to_index(grid, LOAD) if lane.startswith("nu")
                else grid.position_to_index(LOAD))
        eps = kwargs.get("aniso_eps")
        inverse = kwargs.get("aniso_inv_eps")
        if inverse is not None:
            values = [1.0 / float(e[cell]) for e in inverse]
        else:
            if eps is None:
                eps = component_e_materials(materials)[0]
            values = [float(e[cell]) for e in eps]
        observed["eps"] = np.asarray(values)
        weights = kwargs.get("conformal_weights")
        observed["weights"] = (None if weights is None
                               else np.asarray([float(w[cell]) for w in weights]))
        return original(grid, materials, n_steps, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(module, name, capture))
        if seeded:
            stack.enter_context(patch.object(simulation, "init_state", _seed_h))
            stack.enter_context(patch.object(nu_core, "init_state", _seed_h))
        n_steps = 1 if seeded else 240
        if lane in ("forward", "forward_occ", "materials_occ"):
            stack.enter_context(patch.dict(os.environ, RFX_PEC_OCC_KOTTKE="1"))
            grid = sim._build_grid()
            occupancy = jnp.zeros(grid.shape, dtype=jnp.float32)
            if fractional:
                occupancy = occupancy.at[grid.position_to_index(LOAD)].set(0.2)
            if lane == "materials_occ":
                materials, debye, lorentz, pec, _, _, _ = sim._assemble_materials(grid)
                result = sim._forward_from_materials(
                    grid, materials, debye, lorentz, pec_mask=pec,
                    pec_occupancy=occupancy, n_steps=n_steps)
            else:
                result = sim.forward(
                    n_steps=n_steps, skip_preflight=True,
                    **({"pec_occupancy_override": occupancy}
                       if lane == "forward_occ" else {}))
        else:
            result = sim.run(n_steps=n_steps, skip_preflight=True,
                             compute_s_params=False, **options)
    observed["trace"] = np.asarray(result.time_series)
    return observed


def _relative(actual, reference):
    return float(np.max(np.abs(actual - reference)) / np.max(np.abs(reference)))


@pytest.fixture(scope="module")
def reference_traces():
    return {comp: _run_case("uniform", comp)["trace"]
            for comp in (None,) + COMPONENTS}


@pytest.fixture(scope="module", params=ACCEPTED)
def lane_control(request, reference_traces):
    lane = request.param
    no_c = _run_case(lane)["trace"]
    residual = _relative(no_c, reference_traces[None])
    # Bound the control too: at most one float32 epsilon per timestep.
    assert residual < 240 * np.finfo(np.float32).eps, (lane, residual)
    # Expected residual is the lane's own float32 difference from the
    # reference, measured by the no-C control. Factor 4 covers the measured
    # with-C/no-C ratio 2.24 on conformal paths and up to 3.34 on forward
    # paths; 240 epsilons allow one per timestep.
    # No with-C result sets its own acceptance tolerance.
    tolerance = max(4 * residual, 240 * np.finfo(np.float32).eps)
    print(f"{lane}: no_C_residual={residual:.9g}, tolerance={tolerance:.9g}",
          file=sys.stderr)
    return lane, tolerance


@pytest.mark.parametrize("component", COMPONENTS)
def test_homogeneous_trace(lane_control, reference_traces, component):
    lane, tolerance = lane_control
    actual = _run_case(lane, component)["trace"]
    reference = reference_traces[component]
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    assert _relative(reference, reference_traces[None]) > 0.1
    error = _relative(actual, reference)
    print(f"{lane} {component}: with_C_residual={error:.9g}", file=sys.stderr)
    assert error <= tolerance, (lane, component, error, tolerance)


@pytest.mark.parametrize("component", COMPONENTS)
def test_capacitor_owns_one_edge(lane_control, component):
    lane, tolerance = lane_control
    _assert_one_edge(lane, component, tolerance)


def _assert_one_edge(lane, component, tolerance, *, fractional=False):
    without = _run_case(lane, fractional=fractional, seeded=True)
    with_c = _run_case(lane, component, fractional=fractional, seeded=True)
    axis = COMPONENTS.index(component)
    others = [c for c in range(3) if c != axis]
    # Independent physical stamp: C*d/(epsilon_0*dual_area), cubic 1 mm.
    stamp = CAPACITANCE / (EPS_0 * DX)
    expected = np.zeros(3)
    expected[axis] = stamp
    delta = with_c["eps"] - without["eps"]
    np.testing.assert_allclose(delta, expected, rtol=tolerance,
                               atol=tolerance * 4.0)
    a, b = with_c["trace"][0], without["trace"][0]
    assert np.all(np.abs(b) > 0), (lane, b)
    # Only the first E update is compared: longer traces may change on
    # transverse components after the capacitor field propagates.
    np.testing.assert_allclose(a[others], b[others], rtol=tolerance, atol=0)
    assert abs(a[axis] - b[axis]) > 0.5 * abs(b[axis])
    print(f"{lane} {component} fractional={fractional}: eps_delta={delta.tolist()}, "
          f"first_E_relative={((a-b)/b).tolist()}", file=sys.stderr)
    if lane.startswith("conformal"):
        weights = without["weights"]
        assert weights is not None
        if fractional:
            assert np.any((weights > 0) & (weights < 1)), weights
            np.testing.assert_allclose(without["eps"], 4.0 / weights, rtol=FLOOR)
        else:
            np.testing.assert_array_equal(weights, np.ones(3))
    if lane.endswith("occ") and fractional:
        # At this node one incident cell has occupancy 0.2: eps=4/(1-0.2).
        np.testing.assert_allclose(without["eps"], 5.0, rtol=FLOOR)


@pytest.mark.parametrize("lane", ["conformal", "conformal_smooth"])
@pytest.mark.parametrize("component", COMPONENTS)
def test_fractional_conformal_edge(lane, component):
    _assert_one_edge(lane, component, FLOOR, fractional=True)


@pytest.mark.parametrize("lane", ["forward_occ", "materials_occ"])
@pytest.mark.parametrize("component", COMPONENTS)
def test_fractional_occupancy_edge(lane, component):
    _assert_one_edge(lane, component, FLOOR, fractional=True)


@pytest.mark.parametrize("lane", ["uniform_dual", "nu_dual"])
def test_dual_average_refuses_lumped_capacitor(lane):
    with pytest.raises((ValueError, NotImplementedError), match="interface_eps"):
        _run_case(lane, "ez", seeded=True)


def _waveguide_model(lane, *, capacitor=False):
    sim = Simulation(
        freq_max=25e9, domain=(0.040, 0.012, 0.008), dx=0.002,
        cpml_layers=4,
        boundary=BoundarySpec(
            x="cpml", y=Boundary(lo="pec", hi="pec", conformal=lane == "conformal"),
            z="pec"))
    for x, direction in ((0.008, "+x"), (0.032, "-x")):
        sim.add_waveguide_port(
            x, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray([18e9, 20e9, 22e9]), f0=20e9,
            ref_offset=1, probe_offset=2)
    if capacitor:
        sim.add_lumped_rlc((0.022, 0.006, 0.004), "ez", C=CAPACITANCE,
                           topology="parallel")
    return sim


@pytest.mark.parametrize("lane", ["uniform", "smooth", "kottke", "conformal"])
def test_waveguide_s_matrix_refuses_rlc(lane):
    sim = _waveguide_model(lane, capacitor=True)
    with pytest.raises(NotImplementedError, match=r"lumped RLC.*#1263.*run\(\)"):
        sim.compute_waveguide_s_matrix(
            n_steps=120, subpixel_smoothing=("kottke_pec" if lane == "kottke"
                                           else lane == "smooth"))


def test_waveguide_reference_refuses_rlc():
    sim = _waveguide_model("uniform")
    references = [_waveguide_model("uniform", capacitor=True),
                  _waveguide_model("uniform")]
    with pytest.raises(NotImplementedError, match=r"lumped RLC.*#1263.*run\(\)"):
        sim.compute_waveguide_s_matrix(
            n_steps=120, normalize="flux", port_reference_sims=references)


@pytest.mark.parametrize("smoothing", [True, "kottke_pec"])
def test_capacitor_changes_probe_with_dielectric_cube(smoothing):
    traces = []
    for capacitor in (False, True):
        sim = Simulation(freq_max=10e9, domain=(0.015,) * 3, dx=DX, boundary="pec")
        sim.add_material("diel", eps_r=4.0)
        sim.add(Box((0.0023,) * 3, (0.0047,) * 3), material="diel")
        sim.add_port((0.007,) * 3, "ez", impedance=50.0,
                     waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
        if capacitor:
            sim.add_lumped_rlc((0.010, 0.007, 0.007), "ez", C=CAPACITANCE,
                               topology="parallel")
        sim.add_probe((0.010, 0.007, 0.007), "ez")
        traces.append(np.asarray(sim.run(n_steps=600, skip_preflight=True,
                                         subpixel_smoothing=smoothing).time_series))
    effect = _relative(traces[1], traces[0])
    assert effect > FLOOR
