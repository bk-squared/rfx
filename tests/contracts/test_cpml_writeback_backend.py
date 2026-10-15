"""Backend choice and the padded form against the pre-slab update oracle."""

import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries import _cpml_slab
from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h
from tests._x64_compat import enable_x64
from tests.contracts.test_cpml_slab_equivalence import (
    FACES, _assert_equal, _scene, _whole_array,
)
from tests.contracts.path_equivalence.comparison import compare, component_peaks


@pytest.mark.parametrize("wall", ("z_hi", "x_lo"))
@pytest.mark.parametrize("varying_mu", (False, True))
@pytest.mark.parametrize("magnetic", (False, True))
@pytest.mark.parametrize("wide", (False, True))
def test_padded_eager_equals_pinned_oracle(monkeypatch, wall, varying_mu, magnetic, wide):
    """All 64 face subsets; the oracle shares no production update helpers."""
    monkeypatch.setattr(_cpml_slab, "writeback_form", lambda: "padded")
    with enable_x64(), jax.disable_jit():
        grid, params, psi, state, materials = _scene(wall, varying_mu)
        if wide:
            materials = materials._replace(
                eps_r=materials.eps_r.astype(jnp.float64),
                mu_r=materials.mu_r.astype(jnp.float64))
        apply = apply_cpml_h if magnetic else apply_cpml_e
        for enabled in itertools.product((False, True), repeat=6):
            faces = tuple(f for f, active in zip(FACES, enabled) if active)
            actual = apply(state, params, psi, grid, materials=materials, faces=faces)
            expected = _whole_array(state, params, psi, grid, materials, faces, magnetic)
            assert all(v.dtype == jnp.float32 for v in actual[0])
            _assert_equal(actual, expected)


def _primitives(jaxpr):
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    names = set()
    for eqn in jaxpr.eqns:
        names.add(eqn.primitive.name)
        for value in eqn.params.values():
            for sub in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(sub, "eqns") or hasattr(getattr(sub, "jaxpr", None), "eqns"):
                    names.update(_primitives(sub))
    return names


@pytest.mark.parametrize("backend,form", (("cpu", "slab"), ("gpu", "padded"),
                                         ("tpu", "slab")))
def test_backend_choice_is_python_at_trace_time(monkeypatch, backend, form):
    grid, params, psi, state, materials = _scene()
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    assert _cpml_slab.writeback_form() == form
    traced = jax.make_jaxpr(lambda s: apply_cpml_e(
        s, params, psi, grid, materials=materials, faces=("x_lo",)))(state)
    names = _primitives(traced)
    assert ("dynamic_update_slice" in names) == (form == "slab"), names
    assert ("select_n" in names) == (form == "padded"), names
    assert "cond" not in names, names


def _emit(measurements, record_property):
    for m in measurements:
        m["ulp_at_peak"] = m["difference"] / float(np.spacing(np.float32(m["peak"])))
    record_property("w2_measurements", json.dumps(measurements))


@pytest.mark.gpu_gate
@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
@pytest.mark.parametrize("steps", (120, 240, 480))
def test_selected_backend_probe_plain_vs_jit(graded, steps, record_property):
    from tests.unit.autodiff.test_forward_jit_contract import _lumped_open, _run

    sim = _lumped_open(graded)

    def solve():
        # Retain the contract's summed output in the traced program as well.
        # Returning only the probe permits different compiler elimination.
        return _run(sim, graded, None if graded else "s", "box", steps, None)

    measurements = []
    try:
        plain, _ = solve()
        compiled, _ = jax.jit(solve)()
        compare(plain, compiled, kind="step", measurements=measurements,
                record=f"{jax.default_backend()}/{_cpml_slab.writeback_form()}/"
                       f"{'graded' if graded else 'uniform'}/probe/{steps}")
    finally:
        _emit(measurements, record_property)
        jax.clear_caches()


def _field_comparison(a, b, label, record_property, *, assert_bar, kind="step"):
    peaks = component_peaks(a, b)
    measurements, errors = [], []
    for key in a:
        try:
            compare(a[key], b[key], record=f"{label}/{key}", kind=kind,
                    peak=peaks[key], measurements=measurements)
        except AssertionError as exc:
            errors.append(str(exc))
    _emit(measurements, record_property)
    if assert_bar:
        assert not errors, "\n".join(errors)


@pytest.mark.gpu_gate
@pytest.mark.parametrize("lane", ("uniform", "graded"))
def test_selected_backend_six_fields_against_other_form(monkeypatch, lane, record_property):
    from tests.contracts._cpml_writeback_scene import build

    backend = jax.default_backend()
    selected = _cpml_slab.writeback_form()
    other = "slab" if selected == "padded" else "padded"

    def fields(form, eager):
        jax.clear_caches()
        with monkeypatch.context() as patch, jax.disable_jit(eager):
            patch.setattr(_cpml_slab, "writeback_form", lambda: form)
            sim, _ = build(lane)
            result = sim.run(n_steps=steps, compute_s_params=False)
            return {key: np.asarray(getattr(result.state, key))
                    for key in ("ex", "ey", "ez", "hx", "hy", "hz")}

    # On GPU the two compiled forms are compared (measured bitwise on seven card
    # classes). On CPU this scene is a poor compiled witness whatever the form:
    # the current source leaves a static ez of 3e5 at its cell, and the slab
    # form alone differs from its own uncompiled result by 0.1-0.2 % of the E
    # peak and a tenth or more of the H peak (measured, Mac arm64, 400 steps).
    # There the two forms are compared uncompiled, where they are bit-equal.
    eager = backend == "cpu"
    steps = 40 if eager else 400     # uncompiled steps cost 0.3 s each
    try:
        _field_comparison(fields(selected, eager), fields(other, eager),
                          f"{backend}/{lane}/{'eager' if eager else 'compiled'}/{steps}",
                          record_property, assert_bar=True,
                          kind="exact" if eager else "step")
    finally:
        jax.clear_caches()
