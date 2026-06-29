"""issue #73: √N gradient checkpointing wired onto the NU waveguide flux
S-matrix path (composes NU's forward cell-savings with the checkpointing tape
savings). `compute_waveguide_s_matrix(checkpoint_segments=K)` used to raise
`NotImplementedError` on a graded mesh; now K is translated to the NU runner's
`checkpoint_every` chunk size and applied to the device run.

Gates: (1) no longer fenced; (2) forward-IDENTICAL with/without K (the NU runner
pads+truncates → no divisor rule); (3) the AD gradient is unchanged (checkpointing
only shrinks the tape, not the value); (4) K<1 raises ValueError.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.auto_config import smooth_grading
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

_A, _B, _FMAX = 0.02286, 0.01016, 12e9
_FREQS = jnp.linspace(8.2e9, 12.4e9, 3)
_NP = 4.0
_SLAB_LO, _SLAB_HI, _SLAB_EPS = 0.030, 0.034, 4.0


def _graded_dy(ratio=2.0, base=0.75e-3):
    n = int(round(_A / base))
    x = np.linspace(-1, 1, n)
    w = 1.0 + (ratio - 1.0) * np.abs(x)
    return smooth_grading(w / w.sum() * _A, max_ratio=1.3)


def _nu_sim():
    dx = 1.5e-3
    nx = int(round(0.064 / dx))
    sim = Simulation(
        freq_max=_FMAX, domain=(nx * dx, _A, _B), dx=dx,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8, dy_profile=_graded_dy(),
    )
    sim.add_material("diel", eps_r=_SLAB_EPS, sigma=0.0)
    sim.add(Box((_SLAB_LO, 0.0, 0.0), (_SLAB_HI, _A, _B)), material="diel")
    for x0, d, nm in ((0.012, "+x", "left"), (nx * dx - 0.012, "-x", "right")):
        sim.add_waveguide_port(
            x0, direction=d, mode=(1, 0), mode_type="TE", freqs=_FREQS,
            f0=10.3e9, bandwidth=0.5,
            reference_plane=(0.017 if d == "+x" else nx * dx - 0.017), name=nm)
    return sim


def _s_of(checkpoint_segments=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(_nu_sim().compute_waveguide_s_matrix(
            num_periods=_NP, normalize="flux",
            checkpoint_segments=checkpoint_segments).s_params)


@pytest.mark.slow
def test_nu_checkpoint_no_longer_fenced_and_forward_identical():
    """It runs (no NotImplementedError) AND the S-matrix is forward-identical to
    the non-checkpointed NU scan for any K (the NU runner pads+truncates)."""
    s_none = _s_of(None)
    s_k4 = _s_of(4)          # K need NOT divide n_steps on the NU path
    s_k7 = _s_of(7)
    assert np.allclose(s_none, s_k4, rtol=1e-4, atol=1e-6), \
        f"checkpoint K=4 changed the NU S-matrix: max|d|={np.abs(s_none-s_k4).max():.2e}"
    assert np.allclose(s_none, s_k7, rtol=1e-4, atol=1e-6)


def _eps_override(sim, deps):
    from rfx.runners.nonuniform import pos_to_nu_index
    grid = sim._build_nonuniform_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    i_lo = pos_to_nu_index(grid, (_SLAB_LO, _A / 2, _B / 2))[0]
    i_hi = pos_to_nu_index(grid, (_SLAB_HI, _A / 2, _B / 2))[0]
    return eps.at[i_lo:i_hi, :, :].set(_SLAB_EPS + deps)


@pytest.mark.slow
def test_nu_checkpoint_gradient_matches_uncheckpointed():
    """The AD gradient through the NU flux S-matrix is UNCHANGED by checkpointing
    (it only shrinks the reverse-mode tape, not the value)."""
    def loss(deps, K):
        sim = _nu_sim()
        eps = _eps_override(sim, deps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_waveguide_s_matrix(
                num_periods=_NP, normalize="flux", eps_override=eps,
                checkpoint_segments=K)
        return jnp.abs(res.s_params[1, 0, 1]) ** 2

    deps0 = jnp.asarray(0.3, dtype=jnp.float32)
    g_none = float(jax.grad(lambda d: loss(d, None))(deps0))
    g_ckpt = float(jax.grad(lambda d: loss(d, 4))(deps0))
    assert np.isfinite(g_ckpt)
    assert abs(g_ckpt - g_none) <= 1e-3 * (abs(g_none) + 1e-6) + 1e-7, \
        f"checkpointed grad {g_ckpt} != uncheckpointed {g_none}"


def test_nu_checkpoint_invalid_raises():
    """K < 1 is a ValueError (not a silent no-op)."""
    with pytest.raises(ValueError):
        _nu_sim().compute_waveguide_s_matrix(
            num_periods=_NP, normalize="flux", checkpoint_segments=0)
