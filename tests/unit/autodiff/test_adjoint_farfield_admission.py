"""NTFF/H admission boundaries and compensation."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement.farfield import NTFFBox
from rfx.stepping.adjoint_monitors import monitor_frequencies
from tests._x64_compat import enable_x64
from tests.unit.autodiff.test_adjoint_farfield import scene, collocation_mode
from tests.unit.autodiff.test_design_box_tape import _sim, _box_cells, BOX_LO, BOX_HI


def context(box):
    return SimpleNamespace(dft_meta=(), use_ntff=True, ntff=box,
                           grid=SimpleNamespace(shape=(20,)*3, cpml_layers=2),
                           use_cpml=True, dt=1.)


@pytest.mark.parametrize("bins", [[], [0.], [-.1], [.5], [.1, .1], [np.nan]])
def test_ntff_only_bad_bins(bins):
    ctx = context(NTFFBox(4, 14, 4, 14, 4, 14, np.array(bins), face_centre=True))
    with pytest.raises(ValueError, match="distinct positive sub-Nyquist"):
        monitor_frequencies(ctx)


def test_ntff_halo_and_identical_plane_bins():
    from rfx.measurement.plan import field_channel
    ctx = context(NTFFBox(4, 14, 4, 14, 4, 14, np.array([.1, .2]), face_centre=True))
    np.testing.assert_array_equal(monitor_frequencies(ctx), [.1, .2])
    ctx.ntff = ctx.ntff._replace(i_lo=3)
    with pytest.raises(NotImplementedError, match="interior monitor"):
        monitor_frequencies(ctx)
    ctx.ntff = ctx.ntff._replace(face_centre=False)
    monitor_frequencies(ctx)  # node layout has no backward halo
    ctx.dft_meta = ((field_channel("hy"), 0, 10, np.array([.1]), (4, 8, 4, 8)),)
    with pytest.raises(NotImplementedError, match="identical monitor"):
        monitor_frequencies(ctx)
    ctx.dft_meta = ((field_channel("hy"), 0, 10, np.array([.1, .2]), (4, 8, 4, 8)),)
    monitor_frequencies(ctx)


@pytest.mark.slow
def test_compensation_objective_zero_and_adjoint_carry(record_property):
    from unittest.mock import patch
    seen = []
    # Inspect the production second scan without changing its carry or sources.
    from rfx.stepping import uniform_setup
    original = uniform_setup.make_core_step
    def make(ctx, *args, **kwargs):
        core = original(ctx, *args, **kwargs)
        def step(carry, *row):
            if ctx.drives is None:
                seen.append((ctx.use_ntff, "ntff" in carry, "dft_planes" in carry))
            return core(carry, *row)
        return step
    with enable_x64(), collocation_mode("face_centre"), patch.object(uniform_setup, "make_core_step", make):
        sim, eps = scene("ntff", "float64", False)
        def loss(e):
            result = sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=e,
                                 n_steps=256, gradient="adjoint", skip_preflight=True)
            return sum(jnp.real(jnp.sum(a)) for a in result.ntff_data[6:])
        grad = jax.jit(jax.grad(loss))(eps)
        np.testing.assert_array_equal(grad, jnp.zeros_like(eps))
        assert seen and all(row == (False, False, False) for row in seen)
        record_property("adjoint_carry", seen)


def test_float64_ntff_plane_bins_require_float32_rounding():
    """Trace both public monitor objectives without executing a settled record."""
    with enable_x64():
        for rounded in (False, True):
            sim = _sim(precision="float64")
            sim.add_ntff_box((4e-3, 4e-3, 4e-3), (20e-3, 16e-3, 12e-3),
                             freqs=np.asarray([2.45e9], np.float64))
            plane_freqs = (np.asarray([2.45e9], np.float32).astype(np.float64)
                           if rounded else np.asarray([2.45e9], np.float64))
            ix = sim._build_grid().position_to_index((18e-3, 10e-3, 8e-3))
            sim.add_dft_plane_probe(axis="x", coordinate=18e-3, component="ez",
                freqs=plane_freqs, name="out", region=(ix[1], ix[1]+1, ix[2], ix[2]+1))
            params = jnp.full(_box_cells(sim)[1], 2., dtype=jnp.float64)

            def loss(eps):
                result = sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=eps,
                                     n_steps=32, gradient="adjoint", skip_preflight=True)
                return (jnp.sum(jnp.abs(result.dft_planes["out"].accumulator)**2)
                        + sum(jnp.real(jnp.sum(a)) for a in result.ntff_data[:6]))

            if rounded:
                jax.jit(jax.grad(loss)).lower(params)
            else:
                with pytest.raises(NotImplementedError, match="float32-rounded"):
                    jax.jit(jax.grad(loss)).lower(params)
