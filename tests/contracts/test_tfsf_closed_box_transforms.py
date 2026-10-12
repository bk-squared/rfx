"""Summed scattered-field observables, not ULPs of a cancellation residue.

The plain forward is the reference; central differences independently check AD.
N=192 (0.732 ns) keeps this 29-cubed CPU fixture bounded. Longer-record gradient
witnesses and the sphere/plate measurements belong to the separate design record.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.farfield import NTFFBox
from rfx.rcs import compute_rcs_jax
from rfx.sources.sources import CustomWaveform
from rfx.sources.tfsf import init_tfsf, measure_normal_incident_spectrum


N_STEPS = 192
P0 = 4.0
FACES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
RUN_REFUSAL = (
    "CustomWaveform and closed_box TFSF cannot be passed through run() "
    "under external JAX transformations; differentiate forward() instead.")


@pytest.fixture(scope="module")
def observable():
    sim = Simulation(freq_max=11e9, domain=(.032,) * 3, dx=.002,
                     boundary="cpml", cpml_layers=6)
    sim.add_tfsf_source(f0=10e9, bandwidth=.5, polarization="ez",
                        direction="+x", margin=4, closed_box=True)
    freqs = np.array([9e9, 10e9, 11e9])
    sim.add_ntff_box((.004,) * 3, (.028,) * 3, freqs=freqs)
    grid = sim._build_grid()
    assert grid.shape == (29, 29, 29)
    # Hand-written padded indices: CPML ends at 6/23, TF at 10/18,
    # NTFF at 8/20, and the four design cells are 12..15 on every axis.
    # These objects construct the observable; none supplies an expected value.
    box = NTFFBox(8, 20, 8, 20, 8, 20, jnp.asarray(freqs),
                  cpml_lo_x=6, cpml_hi_x=6, cpml_lo_y=6, cpml_hi_y=6,
                  cpml_lo_z=6, cpml_hi_z=6, face_centre=True)
    cfg, state = init_tfsf(
        29, .002, grid.dt, ny=29, nz=29, cpml_layers=6, tfsf_margin=4,
        f0=10e9, bandwidth=.5, polarization="ez", direction="+x", closed_box=True)
    incident = jnp.asarray(measure_normal_incident_spectrum(
        cfg, state, N_STEPS, freqs, grid.dt))
    occupancy = jnp.ones((4, 4, 4), dtype=jnp.float32)

    def objective(p):
        result = sim.forward(
            n_steps=N_STEPS, checkpoint=True, skip_preflight=True,
            design_box=((.012,) * 3, (.018,) * 3),
            design_eps_override=1 + (p - 1) * occupancy)
        sigma = compute_rcs_jax(
            result.ntff_data, box, grid, jnp.array([jnp.pi / 2]),
            jnp.array([jnp.pi]), incident).reshape(3)
        return jnp.mean(sigma), (result.ntff_data, sigma)

    return sim, objective


@pytest.fixture(scope="module")
def plain(observable):
    _, objective = observable
    return [jax.device_get(objective(jnp.float32(p))) for p in (P0, 1.01 * P0)]


def assert_summed(actual, expected):
    actual_j, (actual_ntff, actual_sigma) = actual
    expected_j, (expected_ntff, expected_sigma) = expected
    arrays = [(name, getattr(actual_ntff, name), getattr(expected_ntff, name))
              for name in FACES]
    arrays.extend((("sigma", actual_sigma, expected_sigma),
                   ("J", actual_j, expected_j)))
    for name, value, reference in arrays:
        value, reference = np.asarray(value), np.asarray(reference)
        peak = np.max(np.abs(reference))
        difference = np.max(np.abs(value - reference))
        assert np.isfinite(value).all() and np.isfinite(reference).all(), name
        assert peak > 0, f"{name}: zero reference cannot normalize the comparison"
        assert difference <= 1e-4 * peak, (
            f"{name}: difference={difference}, reference_peak={peak}, "
            f"relative={difference / peak} > 1e-4")


@pytest.mark.parametrize("transform", (
    "value_and_grad", "jvp", "jit", "checkpoint", "vmap", "scan"))
def test_summed_observables(observable, plain, transform):
    _, objective = observable
    p = jnp.float32(P0)
    if transform == "value_and_grad":
        actual, _ = jax.value_and_grad(objective, has_aux=True)(p)
    elif transform == "jvp":
        actual, _ = jax.jvp(objective, (p,), (jnp.float32(1),))
    elif transform == "jit":
        actual = jax.jit(objective)(p)
    elif transform == "checkpoint":
        actual, _ = jax.value_and_grad(jax.checkpoint(objective), has_aux=True)(p)
    else:
        parameters = jnp.array([P0, 1.01 * P0], dtype=jnp.float32)
        if transform == "vmap":
            actual = jax.vmap(objective)(parameters)
        else:
            _, actual = jax.lax.scan(
                lambda carry, value: (carry, objective(value)), None, parameters)
        for lane, reference in enumerate(plain):
            assert_summed(jax.tree.map(lambda x: x[lane], actual), reference)
        return
    assert_summed(actual, plain[0])


@pytest.mark.parametrize("p0", [P0], ids=["AD-vs-two-FDs"])
def test_gradient_matches_resolved_finite_difference(observable, p0):
    _, objective = observable

    def scalar(p):
        return objective(p)[0]

    ad = float(jax.grad(scalar)(jnp.float32(p0)))
    fds = []
    for delta in (.01 * p0, .02 * p0):
        plus = float(scalar(jnp.float32(p0 + delta)))
        minus = float(scalar(jnp.float32(p0 - delta)))
        fds.append((plus - minus) / (2 * delta))
    assert np.isfinite([ad, *fds]).all()
    assert abs(fds[1] - fds[0]) <= 1e-2 * abs(fds[0]), f"FD unresolved: {fds}"
    assert abs(ad - fds[0]) <= 1e-2 * abs(fds[0]), f"AD={ad}, FD={fds}"


@pytest.mark.parametrize("transform", ["jit"])
def test_run_keeps_its_refusal(observable, transform):
    sim, _ = observable
    with pytest.raises(NotImplementedError) as exc:
        jax.jit(lambda: sim.run(n_steps=2, skip_preflight=True).time_series)()
    assert str(exc.value) == RUN_REFUSAL


@pytest.mark.parametrize("lane", ["distributed", "graded", "subgridded"])
@pytest.mark.parametrize("source", ["closed_box", "custom"])
def test_unsupported_forward_lane_still_refuses_under_grad(lane, source):
    options = dict(freq_max=11e9, domain=(.032,) * 3, dx=.002,
                   boundary="cpml", cpml_layers=6)
    if lane == "graded":
        options["dz_profile"] = [.002] * 16
    sim = Simulation(**options)
    sim.add_tfsf_source(f0=10e9, closed_box=source == "closed_box",
                        waveform=(CustomWaveform(lambda t: jnp.sin(t * 1e11))
                                  if source == "custom" else "differentiated_gaussian"))
    if lane == "subgridded":
        sim.add_refinement(z_range=(.008, .016), ratio=2)

    def objective(p):
        return jnp.sum(sim.forward(
            n_steps=2, skip_preflight=True, distributed=lane == "distributed",
            design_box=((.012,) * 3, (.018,) * 3),
            design_eps_override=p * jnp.ones((4, 4, 4))).time_series)

    with pytest.raises(NotImplementedError) as exc:
        jax.grad(objective)(jnp.float32(P0))
    assert str(exc.value) == (
        "CustomWaveform and closed_box TFSF require a uniform, single-device, "
        "3D second-order Yee run/forward with no subgrids")
