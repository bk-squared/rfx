"""#1439 G1–G4: public forward references, settling first, then reciprocity.

All long solves are slow; the small local transpose/source gate is always on.
Measurements are JUnit properties, including every attempted reference length.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, gradient_record_length_witness
from rfx.core.yee import MU_0
from rfx.measurement.farfield import (
    NTFFBox, accumulate_ntff, init_ntff_data, compute_far_field_jax,
)
from tests._x64_compat import enable_x64
from tests.unit.autodiff.test_reciprocity_adjoint import fixture, errors
from tests.unit.autodiff.test_design_box_tape import BOX_LO, BOX_HI, F0

FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")
CASES = [("h", "face_centre"), ("ntff", "face_centre"), ("ntff", "node")]


@contextmanager
def collocation_mode(mode):
    # The public registration has no collocation switch. Change the measurement
    # box factory, identically for both forward modes, before plan assembly.
    import rfx.farfield as farfield
    original = farfield.make_ntff_box
    def make(*args, **kwargs):
        return original(*args, **{**kwargs, "collocation": mode})
    with patch.object(farfield, "make_ntff_box", make):
        yield


def scene(kind, precision, lossy):
    sim, eps = fixture(precision, point=False)
    if lossy:
        sim.add_material("background_loss", eps_r=1., sigma=0.2)
        sim.add(Box((-1., -1., -1.), (1., 1., 1.)), material="background_loss")
    if kind == "h":
        sim._dft_planes[0] = replace(sim._dft_planes[0], component="hy")
    else:
        sim._dft_planes.clear()
        sim.add_ntff_box((4e-3, 4e-3, 4e-3), (20e-3, 16e-3, 12e-3),
                         freqs=np.array([0.8*F0, 1.05*F0]))
    return sim, eps


def loss_function(sim, kind, mode):
    def loss(eps, steps):
        result = sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=eps,
                             n_steps=steps, checkpoint_segments=8,
                             gradient=mode, skip_preflight=True)
        if kind == "h":
            spectrum = result.dft_planes["out"].accumulator / result.dt
            weights = jnp.arange(spectrum.size).reshape(spectrum.shape) + 1
            return jnp.sum(weights * jnp.abs(spectrum)**2)
        # Three polar samples, two bins; a discrete sphere quadrature defines
        # radiated power. This objective depends on both E and H NTFF channels.
        theta = jnp.array([np.pi/6, np.pi/2, 5*np.pi/6])
        phi = jnp.arange(8) * (2*np.pi/8)
        ff = compute_far_field_jax(result.ntff_data, result.ntff_box,
                                   result.grid, theta, phi)
        intensity = jnp.abs(ff.E_theta)**2 + jnp.abs(ff.E_phi)**2
        power = jnp.sum(intensity * jnp.sin(theta)[None, :, None], axis=(1, 2))
        power = power * (np.pi/3) * (2*np.pi/8)
        directivity = 4*np.pi*intensity[:, :, 0] / power[:, None]
        return jnp.sum(directivity * jnp.array([[1., 2., 4.], [3., 1., 2.]]))
    return loss


def settled_reference(loss, eps, record_property):
    """Lengthen the comparator before admitting any adjoint comparison."""
    steps = 2400
    grad = jax.jit(jax.grad(loss), static_argnums=1)
    reference = grad(eps, steps)
    for _ in range(4):
        doubled = grad(eps, 2*steps)
        move = errors(doubled, reference)[0]
        record_property(f"reference_{steps}_{2*steps}", move)
        assert np.isfinite(move) and np.max(np.abs(reference)) > 0
        if move <= 1e-8:
            record_property("record_steps", steps)
            record_property("reference_move", move)
            return steps, reference
        steps *= 2
        reference = doubled
    pytest.fail("reference has not settled to 1e-8; no adjoint verdict is valid")


@pytest.mark.slow
@pytest.mark.parametrize("kind,collocation", CASES)
@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_g1_g2_g3_g4(kind, collocation, lossy, precision, record_property):
    start = time.monotonic()
    with enable_x64() if precision == "float64" else nullcontext():
        with collocation_mode(collocation):
            # Amendment 2: both dtypes use the same settled float64 comparator.
            # Materialize on the host before leaving x64 so G3 keeps its precision.
            with enable_x64():
                reference_sim, reference_eps = scene(kind, "float64", lossy)
                reference_loss = loss_function(reference_sim, kind, "autodiff")
                steps, reference = settled_reference(reference_loss, reference_eps, record_property)
                reference = np.asarray(reference)
            record_property("reference_precision", str(reference.dtype))
            sim, eps = scene(kind, precision, lossy)
            adjoint_loss = loss_function(sim, kind, "adjoint")
            # Explicit user jit is covered for both H-plane and NTFF-only calls.
            actual = jax.jit(jax.grad(adjoint_loss), static_argnums=1)(eps, steps)
            error = errors(actual, reference)[0]
            record_property("gradient_error", error)
            record_property("gate", "G1" if kind == "h" else "G2")
            assert error <= (1e-5 if precision == "float64" else 1e-4), "G1/G2/G3 gradient agreement"
            witness = gradient_record_length_witness(
                adjoint_loss, eps, steps, tol=1e-2, factor=2.0)
            record_property("witness_worst", witness.worst)
            assert witness.passed, "G4 record-length witness"
    record_property("seconds", time.monotonic() - start)
    # These parameterized traces have different static boxes, records and dtypes.
    jax.clear_caches()
    gc.collect()


def test_local_transpose_and_magnetic_sources(record_property):
    """Full-grid production VJP is an independent oracle for cropped targets.

    The source assertion independently Fourier-transforms actual AFTER_H
    increments at their integer-step stamps; neither expected result uses a
    new helper. Nonzero compensation cotangents must be immaterial.
    """
    from rfx.measurement.adjoint_targets import ntff_target_spectra
    from rfx.stepping.adjoint_monitors import magnetic_wavelet_hook
    start = time.monotonic()
    with enable_x64():
        rng = np.random.default_rng(1439)
        shape = (7, 8, 9)
        fields = tuple(jnp.asarray(rng.normal(size=shape)) for _ in FIELDS)
        for centre in (False, True):
            box = NTFFBox(2, 5, 2, 6, 2, 7, np.zeros(2), face_centre=centre)
            empty = init_ntff_data(box, field_dtype=jnp.float64)
            bar = jax.tree.map(lambda a: jnp.asarray(
                rng.normal(size=a.shape) + 1j*rng.normal(size=a.shape)), empty)
            electric, magnetic = ntff_target_spectra(box, bar, jnp.float64)
            actual = {c: jnp.zeros((2,)+shape, jnp.complex128) for c in FIELDS}
            for c, slot, target in electric + magnetic:
                assert min(target.shape[1:]) <= 2, "transpose must stay on face slabs"
                actual[c] = actual[c].at[(slice(None),)+slot].add(target)
            def increment(*values):
                return accumulate_ntff(empty, SimpleNamespace(**dict(zip(FIELDS, values))), box, 1., 0)
            _, pullback = jax.vjp(increment, *fields)
            for f in range(2):
                selected = jax.tree.map(lambda a: jnp.zeros_like(a).at[f].set(a[f]), bar)
                for part in (jnp.real, jnp.imag):
                    expected = pullback(jax.tree.map(lambda a: part(a).astype(a.dtype), selected))
                    for c, ref in zip(FIELDS, expected):
                        np.testing.assert_allclose(part(actual[c][f]), ref, rtol=1e-13, atol=1e-13)
        freqs = jnp.array([.063, .078])
        target = jnp.array([1+2j, -2+.3j])[:, None, None]
        # Non-unit, different component permeabilities exercise Db's edge choice.
        ctx = SimpleNamespace(dt=1., materials=SimpleNamespace(
            components=SimpleNamespace(mu_update=(2., 3., 5.))))
        slot = (1, slice(1, 2), slice(1, 2))
        hook = magnetic_wavelet_hook(ctx, [("hy", slot, target)], freqs, 128, jnp.float64)
        def sample(n):
            from rfx.core.yee import FDTDState
            state = FDTDState(*(jnp.zeros((3, 3, 3)) for _ in FIELDS), step=jnp.int32(0))
            frame = SimpleNamespace(st=state, step_idx=n)
            hook(frame)
            return frame.st.hy[1, 1, 1]
        n = jnp.arange(128)
        wave = jax.vmap(sample)(n)
        actual = jnp.exp(-2j*jnp.pi*freqs[:, None]*(n+1)) @ wave
        expected = -target[:, 0, 0] * jnp.exp(-1j*jnp.pi*freqs) / (3*MU_0)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-6)
        # Trace the public VJP as well: an NTFF-only jit must not need a plane
        # to obtain concrete bins, and the second scan must close its carry.
        for kind in ("h", "ntff"):
            sim, eps = scene(kind, "float64", False)
            loss = loss_function(sim, kind, "adjoint")
            abstract = jax.eval_shape(jax.grad(lambda e: loss(e, 256)), eps)
            assert abstract.shape == eps.shape
    record_property("seconds", time.monotonic()-start)
