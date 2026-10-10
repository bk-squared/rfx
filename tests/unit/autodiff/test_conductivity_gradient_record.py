"""Conductivity record metadata does not change the differentiated solve."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx.api._execute as execute
from rfx.api._gradient_witness import conductivity_gradient_record
from rfx.core.yee import EPS_0
from tests.unit.autodiff.test_reciprocity_adjoint import fixture, BOX_LO, BOX_HI
from tests.unit.autodiff.test_design_box_tape_nu_occupancy import (
    _nu_sim, _nu_box_cells, _eps_design, NU_BOX_LO, NU_BOX_HI,
)
from tests.contracts.path_equivalence.test_source_on_pec import model


CODE = "conductivity_gradient_record_length"
STEPS = 48
MESSAGE = (
    "A gradient with respect to conductivity keeps a static part on cells where the "
    "charge relaxation time eps/sigma is not short against the record, so it can depend on "
    "the record length's phase; check it with gradient_record_length_witness. "
    "x_min reports record time over relaxation time; no threshold is applied."
)


def record(records):
    selected = [d for d in records if d.code == CODE]
    assert len(selected) == 1
    d = selected[0]
    assert d.severity == "advisory"
    assert d.subject in (None, "design conductivity")
    assert d.message == MESSAGE
    assert set(d.values) == {"record_time_s", "n_steps", "x_min", "x_definition", "permittivity_used"}
    assert d.values["x_definition"] in (
        "minimum over the design cells of record time * sigma / (eps0 * eps_r)",
        "lower bound of that minimum: record time * min(sigma) / (eps0 * max(eps_r))")
    assert not jax.tree.leaves(d)
    assert all(type(value) in (int, float, str) for value in d.values.values())
    return d


def design_fixture(lane="uniform"):
    if lane == "uniform":
        sim, eps = fixture("float32")
        bounds = (BOX_LO, BOX_HI)
        dt = sim._build_grid().dt
    else:
        sim = _nu_sim()
        eps = jnp.asarray(_eps_design(_nu_box_cells(sim)[1]), dtype=jnp.float32)
        bounds = (NU_BOX_LO, NU_BOX_HI)
        dt = sim._build_nonuniform_grid().dt

    def forward(sigma):
        return sim.forward(design_box=bounds, design_eps_override=eps,
                           design_sigma_override=sigma, n_steps=STEPS,
                           gradient="autodiff", checkpoint=False, skip_preflight=True)

    return forward, eps, dt


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_lossless_design(lane):
    forward, eps, dt = design_fixture(lane)
    d = record(forward(jnp.zeros_like(eps)).diagnostics)
    assert d.values["x_min"] == 0.0
    assert d.values["n_steps"] == STEPS
    assert d.values["record_time_s"] == STEPS * dt
    assert d.values["permittivity_used"] == "design_eps_override"


def test_concrete_design_ratio():
    forward, eps, dt = design_fixture()
    sigma = jnp.linspace(0.02, 0.2, eps.size).reshape(eps.shape)
    # The per-cell ratio's own minimum, formed here cell by cell in float64.
    ratios = [float(s) / float(e) for s, e in zip(np.asarray(sigma).ravel(), np.asarray(eps).ravel())]
    expected = STEPS * dt * min(ratios) / EPS_0
    d = record(forward(sigma).diagnostics)
    assert d.values["x_min"] == pytest.approx(expected, rel=1e-6)
    assert d.values["x_definition"].startswith("minimum over the design cells")
    # Closed-over concrete JAX arrays stay host metadata under an outer jit.
    closed = jax.jit(lambda: conductivity_gradient_record(sigma, None, eps, STEPS, dt))()
    assert record(closed).values["x_min"] == pytest.approx(expected, rel=1e-6)


def identity(result, **kwargs):
    return result


def assert_bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def test_grad_record_and_unchanged_numbers(monkeypatch):
    forward, eps, _ = design_fixture()
    sigma = jnp.full_like(eps, 0.02)

    def loss(s):
        r = forward(s)
        return jnp.sum(r.time_series ** 2), (r.time_series, r.diagnostics)

    (value, (series, records)), gradient = jax.value_and_grad(loss, has_aux=True)(sigma)
    assert record(records).values["x_min"] == "not evaluated (traced)"
    monkeypatch.setattr(execute, "with_conductivity_record", identity)
    (reference, (plain_series, plain_records)), plain_gradient = jax.value_and_grad(loss, has_aux=True)(sigma)
    assert not any(d.code == CODE for d in plain_records)
    assert_bits_equal(gradient, plain_gradient)
    assert_bits_equal(series, plain_series)
    assert_bits_equal(value, reference)


def test_no_conductivity_design_unchanged(monkeypatch):
    forward, _, _ = design_fixture()
    result = forward(None)
    assert not any(d.code == CODE for d in result.diagnostics)
    monkeypatch.setattr(execute, "with_conductivity_record", identity)
    plain = forward(None)
    assert result.diagnostics == plain.diagnostics
    for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(plain), strict=True):
        if hasattr(actual, "dtype"):
            assert_bits_equal(actual, expected)


def test_jit_time_series_preserves_record():
    forward, eps, dt = design_fixture()
    with jax.checking_leaks():
        metadata = jax.jit(lambda s: conductivity_gradient_record(s, None, eps, STEPS, dt))(jnp.zeros_like(eps))
    assert record(metadata).values["x_min"] == "not evaluated (traced)"

    def series(sigma):
        result = forward(sigma)
        assert record(result.diagnostics).values["x_min"] == "not evaluated (traced)"
        return result.time_series

    compiled = jax.jit(series)
    first = compiled(jnp.zeros_like(eps))
    second = compiled(jnp.zeros_like(eps))
    assert_bits_equal(first, second)


def test_distributed_traced_sigma_record():
    if len(jax.devices()) < 2:
        pytest.skip(f"needs two devices on one backend, found {len(jax.devices())}")
    sim = model("distributed_graded", adjacent=True)
    grid = sim._build_nonuniform_grid()

    def loss(sigma):
        result = sim.forward(sigma_override=sigma, n_steps=3, skip_preflight=True,
                             distributed=True, devices=jax.devices()[:2])
        return jnp.sum(result.time_series ** 2), result.diagnostics

    (_, records), _ = jax.value_and_grad(loss, has_aux=True)(jnp.zeros(grid.shape))
    d = record(records)
    assert d.values["x_min"] == "not evaluated (traced)"
    assert d.values["permittivity_used"] == "not available"
    assert d.values["record_time_s"] == 3 * grid.dt


def test_per_edge_and_override_trigger():
    sigma = (np.array([0.2, 0.3]), np.array([0.1]), np.array([0.4, 0.5, 0.6]))
    eps = np.array([2., 3.])
    d = record(conductivity_gradient_record(sigma, None, eps, 10, 1e-12))
    expected = 10 * 1e-12 * min(np.min(s) for s in sigma) / (EPS_0 * np.max(eps))
    assert d.values["x_min"] == expected
    assert d.values["x_definition"].startswith("lower bound")
    # Cell arrays of one shape: the minimum of the ratio, not min(sigma) / max(eps).
    cell = record(conductivity_gradient_record(np.array([0.01, 1.]), None, np.array([1., 10.]), 10, 1e-12))
    assert cell.values["x_min"] == pytest.approx(10 * 1e-12 * 0.01 / EPS_0, rel=1e-12)
    assert cell.values["x_min"] > 5 * (10 * 1e-12 * 0.01 / (EPS_0 * 10.))
    assert conductivity_gradient_record(None, np.array([0.1]), eps, 10, 1e-12) == ()

    def traced(s):
        d = record(conductivity_gradient_record(None, s, None, 10, 1e-12))
        assert d.values["x_min"] == "not evaluated (traced)"
        assert d.values["permittivity_used"] == "not available"
        return jnp.sum(s)

    jax.grad(traced)(jnp.array([0.1]))


@pytest.mark.parametrize("traced", ["sigma", "eps"])
def test_amendment_traced_string(traced):
    def loss(value):
        sigma = (np.array([0.2]), value, np.array([0.3])) if traced == "sigma" else np.array([0.1])
        eps = value if traced == "eps" else None
        d = record(conductivity_gradient_record(sigma, None, eps, 10, 1e-12))
        assert d.values["x_min"] == "not evaluated (traced)"
        assert d.values["permittivity_used"] == ("design_eps_override" if traced == "eps" else "not available")
        return jnp.sum(value)

    jax.grad(loss)(jnp.array([2.]))


def test_amendment_no_permittivity_string():
    d = record(conductivity_gradient_record(np.array([0.1]), None, None, 10, 1e-12))
    assert d.values["x_min"] == "not evaluated (no permittivity)"
    assert d.values["permittivity_used"] == "not available"


def test_a_result_without_a_grid_does_not_fail_the_solve():
    """The multi-device lane reads the time step from the result's grid."""
    d = record(conductivity_gradient_record(np.array([0.1]), None, np.array([2.]), 10, None))
    assert d.values["record_time_s"] == d.values["x_min"] == "not available (no time step)"
    assert conductivity_gradient_record(None, None, None, 10, None) == ()
