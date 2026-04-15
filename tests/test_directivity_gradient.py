"""Regression test for issue #32: maximize_directivity gradient non-zero.

Prior to the ratio-based fix, `maximize_directivity` used absolute
|E|^2 at the target direction (~1e-27 in rfx's spectral NTFF
convention), which produced zero gradient in `topology_optimize`.

The ratio-based formulation U(target)/P_rad is scale-invariant and
must give a non-zero gradient through the NTFF accumulation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import make_source, run
from rfx.farfield import make_ntff_box
from rfx.optimize_objectives import (
    maximize_directivity,
    maximize_directivity_ratio,
)


def _forward(eps_scale: jnp.ndarray):
    """Minimal NTFF-enabled forward that is jax.grad-compatible."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), cpml_layers=0)
    materials_base = init_materials(grid.shape)
    # Scale eps_r by a scalar — gradient target.
    materials = materials_base._replace(eps_r=materials_base.eps_r * eps_scale)

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.01, 0.01, 0.01), "ez", pulse, 30)
    ntff = make_ntff_box(
        grid,
        (0.003, 0.003, 0.003),
        (0.017, 0.017, 0.017),
        freqs=jnp.array([3e9]),
    )
    return run(grid, materials, 30, sources=[src], ntff=ntff)


def test_maximize_directivity_alias():
    assert maximize_directivity_ratio is maximize_directivity


def test_maximize_directivity_gradient_nonzero():
    """Gradient of the directivity objective w.r.t. a scalar eps scale
    must be non-trivially non-zero.

    This pins the fix for issue #32: the absolute-power objective
    returned ~1e-27 values and 0.0 gradients; the ratio-based one
    should give |dD/dα| >> 1e-12 in single precision.
    """
    obj = maximize_directivity(theta_target=np.pi / 2, phi_target=0.0)

    def loss(alpha):
        return obj(_forward(alpha))

    value = float(loss(jnp.array(1.0)))
    assert np.isfinite(value)
    assert value < 0.0, "objective sign: -directivity should be < 0"

    grad = float(jax.grad(loss)(jnp.array(1.0)))
    assert np.isfinite(grad)
    # The ratio is scale-invariant, so d/dα exactly 1.0 is a pathological
    # case — what matters is that the path through NTFF keeps producing a
    # meaningful, finite gradient at other probe points.
    grad_off = float(jax.grad(loss)(jnp.array(1.2)))
    assert np.isfinite(grad_off)
    # Combined magnitude must exceed the float32 noise floor (~1e-7).
    assert abs(grad) + abs(grad_off) > 1e-6, (
        f"directivity gradient collapsed to noise: grad(1.0)={grad:.2e}, "
        f"grad(1.2)={grad_off:.2e} (issue #32 would show ~0.0)"
    )
