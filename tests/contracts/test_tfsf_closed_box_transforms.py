"""Summed scattered-field observables, not ULPs of a cancellation residue.

The plain forward is the reference; central differences independently check AD.
N=192 (0.732 ns) keeps this 29-cubed CPU fixture bounded. Longer-record gradient
witnesses and the sphere/plate measurements belong to the separate design record.
"""

import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.farfield import NTFFBox
from rfx.rcs import compute_rcs_jax
from rfx.sources.sources import CustomWaveform
from rfx.sources.tfsf import init_tfsf
from rfx.measurement.incident import measure_normal_incident_spectrum


N_STEPS = 192
P0 = 4.0
FACES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
RUN_REFUSAL = (
    "CustomWaveform and closed_box TFSF cannot be passed through run() "
    "under external JAX transformations; differentiate forward() instead.")


@pytest.fixture(scope="module")
def scenes():
    # Parametrizing a scene together with a transform may recreate fixtures.
    # Keep the same scene object and its plain reference across those rows.
    return functools.lru_cache(maxsize=3)(make_observable)


@pytest.fixture(scope="module")
def observable(request, scenes):
    return scenes(getattr(request, "param", "default"))


def make_observable(scene):
    closed_box = scene != "slab-custom"
    sim = Simulation(freq_max=11e9, domain=(.032,) * 3, dx=.002,
                     boundary="cpml", cpml_layers=6)
    grid = sim._build_grid()
    waveform = "differentiated_gaussian"
    if scene != "default":
        # A pulse that differs from the built-in one (delay 3.5 tau, width
        # 1.1 tau), so a transform that silently ran the built-in pulse instead
        # would change every custom row. Fixed in p.
        tau = 1 / (np.pi * 10e9 * .5)
        waveform = CustomWaveform(
            lambda t: -2 * ((t - 3.5 * tau) / (1.1 * tau))
            * jnp.exp(-((t - 3.5 * tau) / (1.1 * tau)) ** 2))
    sim.add_tfsf_source(f0=10e9, bandwidth=.5, polarization="ez",
                        direction="+x", margin=4, closed_box=closed_box,
                        waveform=waveform)
    if not closed_box:
        cfg, _ = init_tfsf(
            29, .002, grid.dt, ny=29, nz=29, cpml_layers=6, tfsf_margin=4,
            f0=10e9, bandwidth=.5, polarization="ez", direction="+x",
            closed_box=False, waveform=waveform)
        # Full transverse extent preserves the slab's lateral invariance.
        # Include x_lo+2 through x_hi-2; every injection-plane cell is vacuum.
        mask = jnp.zeros(grid.shape).at[cfg.x_lo + 2:cfg.x_hi - 1, :, :].set(1)
        probe_x = (cfg.x_lo - 2 - 6) * .002
        sim.add_probe((probe_x, .016, .016), "ez")

        def objective(p):
            result = sim.forward(n_steps=N_STEPS, checkpoint=True, skip_preflight=True,
                                 eps_override=1 + (p - 1) * mask)
            series = result.time_series[:, 0]
            return jnp.sum(series ** 2), series

        return sim, objective

    freqs = np.array([9e9, 10e9, 11e9])
    sim.add_ntff_box((.004,) * 3, (.028,) * 3, freqs=freqs)
    assert grid.shape == (29, 29, 29)
    # Hand-written padded indices: CPML ends at 6/23, TF at 10/18,
    # NTFF at 8/20, and the four design cells are 12..15 on every axis.
    # These objects construct the observable; none supplies an expected value.
    box = NTFFBox(8, 20, 8, 20, 8, 20, jnp.asarray(freqs),
                  cpml_lo_x=6, cpml_hi_x=6, cpml_lo_y=6, cpml_hi_y=6,
                  cpml_lo_z=6, cpml_hi_z=6, face_centre=True)
    cfg, state = init_tfsf(
        29, .002, grid.dt, ny=29, nz=29, cpml_layers=6, tfsf_margin=4,
        f0=10e9, bandwidth=.5, polarization="ez", direction="+x",
        closed_box=closed_box, waveform=waveform, n_steps=N_STEPS)
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
def reference_cache():
    return {}


@pytest.fixture(scope="module")
def plain(observable, reference_cache):
    _, objective = observable
    if objective not in reference_cache:
        reference_cache[objective] = [
            jax.device_get(objective(jnp.float32(p))) for p in (P0, 1.01 * P0)]
    return reference_cache[objective]


def assert_summed(actual, expected):
    comparisons = {}
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
        comparisons[name] = dict(difference=float(difference),
                                 reference_peak=float(peak), relative=float(difference / peak))
        assert difference <= 1e-4 * peak, (
            f"{name}: difference={difference}, reference_peak={peak}, "
            f"relative={difference / peak} > 1e-4")
    return comparisons


@pytest.mark.parametrize("observable,transform", [
    ("default", transform) for transform in
    ("value_and_grad", "jvp", "jit", "checkpoint", "vmap", "scan")
] + [("closed-custom", "jit"),
     ("closed-custom", "value_and_grad")],
    indirect=["observable"], scope="module")
def test_summed_observables(observable, plain, transform, record_property):
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
        comparisons = [assert_summed(jax.tree.map(lambda x: x[lane], actual), reference)
                       for lane, reference in enumerate(plain)]
        record_property("comparisons", json.dumps(comparisons))
        return
    record_property("comparisons", json.dumps(assert_summed(actual, plain[0])))


@pytest.mark.parametrize("observable", [
    "default", "closed-custom", "slab-custom"],
                         indirect=True, scope="module")
def test_gradient_matches_resolved_finite_difference(observable, record_property):
    p0 = P0
    _, objective = observable

    def scalar(p):
        return objective(p)[0]

    ad = float(jax.grad(scalar)(jnp.float32(p0)))
    fds = []
    for delta in (.01 * p0, .02 * p0):
        plus = float(scalar(jnp.float32(p0 + delta)))
        minus = float(scalar(jnp.float32(p0 - delta)))
        fds.append((plus - minus) / (2 * delta))
    record_property("gradient", json.dumps(dict(ad=ad, fds=fds,
        fd_relative=abs(fds[1] - fds[0]) / abs(fds[0]),
        ad_relative=abs(ad - fds[0]) / abs(fds[0]))))
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
    with pytest.raises(NotImplementedError) as exc:
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

        jax.grad(objective)(jnp.float32(P0))
    if lane == "distributed":
        assert ("open-slab" if source == "custom" else "design_box") in str(exc.value)
    else:
        assert str(exc.value) == (
            "CustomWaveform and closed_box TFSF require a uniform, "
            "3D second-order Yee run/forward with no subgrids")


@pytest.fixture(scope="module")
def plain_gradients(observable):
    _, objective = observable
    return np.array([float(jax.grad(lambda p: objective(p)[0])(jnp.float32(p)))
                     for p in (P0, 1.01 * P0)])


@pytest.mark.parametrize("transform", [
    "jit-grad", "value_and_grad", "jvp", "grad-checkpoint", "vmap-grad"])
def test_returned_gradients(observable, plain_gradients, transform, record_property):
    _, objective = observable
    scalar = lambda p: objective(p)[0]
    p = jnp.float32(P0)
    if transform == "jit-grad":
        actual = jax.jit(jax.grad(scalar))(p)
    elif transform == "value_and_grad":
        _, actual = jax.value_and_grad(objective, has_aux=True)(p)
    elif transform == "jvp":
        _, actual = jax.jvp(scalar, (p,), (jnp.float32(1),))
    elif transform == "grad-checkpoint":
        actual = jax.grad(jax.checkpoint(scalar))(p)
    else:
        actual = jax.vmap(jax.grad(scalar))(jnp.array([P0, 1.01 * P0], dtype=jnp.float32))
    actual = np.atleast_1d(np.asarray(actual))
    expected = plain_gradients[:len(actual)]
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert np.all(np.abs(expected) > 0)
    relative = np.abs(actual - expected) / np.abs(expected)
    record_property("gradient", json.dumps(dict(actual=actual.tolist(),
        reference=expected.tolist(), relative=relative.tolist())))
    assert np.all(relative <= 1e-4), f"actual={actual}, reference={expected}, relative={relative}"


@pytest.mark.parametrize("transform", ["grad", "jit"])
def test_traced_shell_requires_vacuum(observable, transform, record_property):
    sim, _ = observable

    def objective(p):
        # Whole-grid override: exactly one cell on x_lo, inside y/z faces.
        eps = jnp.ones((29, 29, 29)).at[10, 14, 14].set(p)
        result = sim.forward(n_steps=2, eps_override=eps, skip_preflight=True)
        return jnp.sum(result.ntff_data.x_lo.real)

    with pytest.raises(Exception) as exc:
        jax.block_until_ready(getattr(jax, transform)(objective)(jnp.float32(P0)))
    message = ("closed_box TFSF requires vacuum on and adjacent to all six "
               "faces; non-vacuum material found at x=10")
    record_property("refusal", str(exc.value))
    assert message in str(exc.value)


def test_custom_waveform_cannot_close_over_traced_parameter(record_property):
    def objective(p):
        sim = Simulation(freq_max=11e9, domain=(.032,) * 3, dx=.002,
                         boundary="cpml", cpml_layers=6)
        sim.add_tfsf_source(f0=10e9, margin=4, closed_box=True,
                            waveform=CustomWaveform(lambda t: p * jnp.sin(t * 1e11)))
        return jnp.sum(sim.forward(n_steps=2, skip_preflight=True).time_series)

    with pytest.raises(ValueError) as exc:
        jax.grad(objective)(jnp.float32(P0))
    message = ("TFSF CustomWaveform is a fixed input: it cannot depend on a traced "
               "parameter (differentiate the materials, not the source waveform)")
    record_property("refusal", str(exc.value))
    assert str(exc.value) == message


@pytest.mark.parametrize("observable", ["slab-custom"], indirect=True, scope="module")
def test_slab_custom_waveform_jit(observable, record_property):
    _, objective = observable
    p = jnp.float32(P0)
    _, reference = jax.device_get(objective(p))
    _, actual = jax.device_get(jax.jit(objective)(p))
    peak = float(np.max(np.abs(reference)))
    difference = float(np.max(np.abs(actual - reference)))
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    assert peak > 0
    record_property("probe", json.dumps(dict(difference=difference,
        reference_peak=peak, relative=difference / peak)))
    assert difference <= 1e-4 * peak, f"difference={difference}, reference_peak={peak}"
