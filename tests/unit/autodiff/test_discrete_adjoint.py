"""#1424 F1: exact production-scan design adjoint, CPU-sized gates."""
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box
from rfx.farfield import compute_far_field_jax
from tests._x64_compat import enable_x64
from tests.unit.autodiff.test_design_box_tape import (
    _sim, _box_cells, _eps_design, BOX_LO, BOX_HI, F0, DX,
)

STEPS = 48


def fixture(precision="float32", objective="probe_dft"):
    sim = _sim(precision=precision)
    if objective == "ntff":
        sim.add_ntff_box((2e-3, 2e-3, 2e-3), (22e-3, 18e-3, 14e-3),
                         freqs=jnp.array([F0]))
    else:
        sim.add_dft_plane_probe(axis="x", coordinate=18e-3, component="ez",
                                freqs=jnp.array([F0]), name="out")
    eps = jnp.asarray(_eps_design(_box_cells(sim)[1]), dtype=precision)
    return sim, eps


def objective_fn(sim, objective="probe_dft", **options):
    def loss(eps, sigma):
        result = sim.forward(design_box=(BOX_LO, BOX_HI),
                             design_eps_override=eps,
                             design_sigma_override=sigma,
                             n_steps=STEPS, skip_preflight=True, **options)
        if objective == "ntff":
            ff = compute_far_field_jax(result.ntff_data, result.ntff_box, result.grid,
                                   jnp.array([0.9]), jnp.array([0.2, 0.7]))
            return jnp.sum(jnp.abs(ff.E_theta)**2 + jnp.abs(ff.E_phi)**2)
        # Both per-step output cotangents and final DFT-carry cotangents.
        phase = jnp.exp(-1j * jnp.arange(STEPS) * 0.23)
        probe = jnp.sum(result.time_series[:, 0] * phase)
        plane = result.dft_planes["out"].accumulator
        return jnp.abs(probe)**2 + jnp.sum(jnp.abs(plane)**2)
    return loss


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("objective", ["probe_dft", "ntff"])
def test_gradient(precision, lossy, objective):
    with enable_x64() if precision == "float64" else nullcontext():
        sim, eps = fixture(precision, objective)
        sigma = jnp.full_like(eps, 0.02) if lossy else None
        args = (0, 1) if lossy else 0
        reference = jax.jit(jax.value_and_grad(objective_fn(
            sim, objective, checkpoint_segments=4), argnums=args))(eps, sigma)
        actual = jax.jit(jax.value_and_grad(objective_fn(
            sim, objective, gradient="adjoint", checkpoint_segments=4),
            argnums=args))(eps, sigma)
        np.testing.assert_allclose(actual[0], reference[0], rtol=1e-6)
        for a, b in zip(jax.tree.leaves(actual[1]), jax.tree.leaves(reference[1])):
            a, b = np.asarray(a), np.asarray(b)
            peak = np.max(np.abs(b))
            assert peak > 0 and np.all(np.isfinite(a))
            error = np.max(np.abs(a-b)) / peak
            print(f"{precision} {lossy=} {objective} peak-relative error={error:.9g}")
            assert error < 1e-4


def test_default_unchanged():
    sim, eps = fixture()
    implicit = objective_fn(sim)
    explicit = objective_fn(sim, gradient="autodiff")
    assert str(jax.make_jaxpr(implicit)(eps, None)) == str(
        jax.make_jaxpr(explicit)(eps, None))
    np.testing.assert_array_equal(jax.jit(implicit)(eps, None),
                                  jax.jit(explicit)(eps, None))


@pytest.mark.parametrize("path", ["graded", "distributed", "ringdown", "ports",
                                  "debye", "lorentz", "kerr", "upml",
                                  "occupancy", "whole_grid", "missing_box",
                                  "tfsf", "precision", "solver", "invalid",
                                  "sheet", "current_moments", "stencil"])
def test_refusals(path):
    sim, eps = fixture()
    options = dict(gradient="adjoint", design_box=(BOX_LO, BOX_HI),
                   design_eps_override=eps, n_steps=4, skip_preflight=True)
    match = "adjoint"
    if path == "graded":
        sim._dz_profile = np.full(8, DX)
    elif path == "distributed":
        options["distributed"] = True
    elif path == "ringdown":
        options["ringdown"] = object()
    elif path == "ports":
        sim.add_port(position=(4e-3, 10e-3, 8e-3), component="ez", impedance=50)
    elif path in ("debye", "lorentz", "kerr"):
        from rfx.materials.debye import DebyePole
        from rfx.materials.lorentz import LorentzPole
        extra = {"debye_poles": [DebyePole(1., 1e-11)]} if path == "debye" else (
            {"lorentz_poles": [LorentzPole(1., 2e10, 1e9)]} if path == "lorentz"
            else {"chi3": 1e-20})
        sim.add_material("disp", eps_r=3., **extra)
        sim.add(Box(BOX_LO, BOX_HI), material="disp")
        match = "Debye/Lorentz" if path != "kerr" else "Kerr"
    elif path == "upml":
        sim = _sim(boundary="upml")
        match = "upml"
    elif path == "occupancy":
        options["design_occupancy_override"] = jnp.zeros_like(eps)
    elif path == "whole_grid":
        options["eps_override"] = jnp.ones(sim._build_grid().shape)
    elif path == "missing_box":
        options["design_box"] = None
    elif path == "tfsf":
        # Public registration fixture for an unsupported source family.
        sim._ports.clear()
        sim.add_tfsf_source(f0=F0, closed_box=True)
    elif path == "precision":
        sim = _sim(precision="mixed")
    elif path == "current_moments":
        sim.add_current_moment_monitor(BOX_LO, BOX_HI, block_size=4e-3,
                                       freqs=[F0])
    elif path == "sheet":
        with pytest.warns(UserWarning, match="Leontovich"):
            sim.add_thin_conductor(
                Box((10e-3, 8e-3, 8e-3), (14e-3, 12e-3, 8e-3)),
                sigma_bulk=5.8e7, thickness=35e-6, surface_impedance_f0=F0)
        match = "sheet"
    elif path == "stencil":
        sim._stencil_order = 4
        match = "stencil_order"
    elif path == "solver":
        sim._solver = "adi"
    elif path == "invalid":
        options["gradient"] = "unknown"
        match = "gradient must"
    with pytest.raises((NotImplementedError, ValueError), match=match):
        sim.forward(**options)


def test_checkpoint_independence():
    sim, eps = fixture()
    functions = [jax.jit(jax.value_and_grad(objective_fn(
        sim, gradient="adjoint", checkpoint_segments=k))) for k in (None, 4, 8)]
    results = [f(eps, None) for f in functions]
    for result in results[1:]:
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(results[0])):
            np.testing.assert_array_equal(a, b)
    # The differentiated scan must not contain a checkpoint replay primitive.
    text = str(jax.make_jaxpr(jax.value_and_grad(objective_fn(
        sim, gradient="adjoint")))(eps, None))
    assert "remat" not in text


def test_local_residuals():
    from rfx.ad_diagnostics import inspect_ad_saved_residuals
    sim, eps = fixture()
    records = inspect_ad_saved_residuals(
        objective_fn(sim, gradient="adjoint"), eps, None).records
    history = [r.shape for r in records if r.shape and r.shape[0] == STEPS]
    window = tuple(n + 1 for n in eps.shape)
    assert history.count((STEPS,) + window) == 6
    assert not any(r.shape and tuple(sim._build_grid().shape) == r.shape[-3:]
                   and len(r.shape) > 3 for r in records)


def test_edge_sigma_gradient():
    sim, eps = fixture()
    sigma = tuple(jnp.full_like(eps, s) for s in (0.01, 0.02, 0.03))
    grads = [jax.jit(jax.grad(objective_fn(sim, gradient=mode), argnums=(0, 1)))(eps, sigma)
             for mode in ("autodiff", "adjoint")]
    for a, b in zip(jax.tree.leaves(grads[0]), jax.tree.leaves(grads[1])):
        assert np.max(np.abs(a)) > 0
        assert np.max(np.abs(a-b)) < 1e-4 * np.max(np.abs(a))
