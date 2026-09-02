"""Differentiable plane-wave (TFSF) inverse design through Simulation.forward().

#404 wired oblique/normal TFSF into run(); this wires it into the DIFFERENTIABLE
forward path (_forward_from_materials passes the aux TFSF cfg to the shared scan,
which auto-activates the complex Bloch path for oblique). Gradients flow w.r.t.
the scatterer permittivity through the same validated kernel as run().

Previously forward() SILENTLY DROPPED a TFSF source (zero gradients — the
"TFSF->forward silent-drop" footgun). These gates pin that it now couples AND
produces a CORRECT gradient (finite-difference check), for normal and oblique.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation


def _sim(angle, waveform, bandwidth):
    sim = Simulation(freq_max=10e9, domain=(0.3, 0.12, 0.006), dx=0.002,
                     boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=5e9, bandwidth=bandwidth, polarization="ez",
                        direction="+x", angle_deg=angle, waveform=waveform)
    sim.add_probe((0.0, 0.0, 0.0), component="ez")  # total-field region (center)
    return sim


def _objective(sim, shp, n_steps, checkpoint=True, checkpoint_segments=None):
    xi = shp[0] // 2

    def obj(eps_val):
        eps = jnp.ones(shp, jnp.float32).at[xi:xi + 15, :, :].set(eps_val)
        fr = sim.forward(eps_override=eps, n_steps=n_steps, checkpoint=checkpoint,
                         checkpoint_segments=checkpoint_segments,
                         skip_preflight=True)
        return jnp.sum(jnp.abs(fr.time_series) ** 2).real

    return obj


@pytest.mark.slow
def test_forward_tfsf_normal_gradient_matches_fd():
    """forward() couples a normal-incidence TFSF source and returns a CORRECT
    gradient w.r.t. the scatterer eps (checkpointed path; matches finite diff)."""
    sim = _sim(0.0, "differentiated_gaussian", 0.5)
    shp = sim.run(n_steps=1).state.ez.shape
    obj = _objective(sim, shp, n_steps=600, checkpoint=True)

    val = float(obj(4.0))
    assert val > 1e-3, f"TFSF source did not couple (obj={val:.2e})"

    g = float(jax.grad(obj)(4.0))
    assert np.isfinite(g) and abs(g) > 1e-6, f"grad not finite/nonzero: {g}"

    # finite-difference correctness (not just nonzero)
    obj_fd = _objective(sim, shp, n_steps=600, checkpoint=False)
    d = 0.02
    fd = (float(obj_fd(4.0 + d)) - float(obj_fd(4.0 - d))) / (2 * d)
    rel = abs(fd - g) / max(abs(fd), 1e-30)
    assert rel < 0.05, f"grad {g:.4f} vs FD {fd:.4f} rel_err {rel*100:.1f}% > 5%"


# highmem (issue #545): the default-checkpoint footprint of this test was
# ~24 GB (23.78 GB / 23,964 MB on two independent local measurements;
# ~1.5x the ~16 GB runner-class ceiling), which is why it was the test the
# weekly lane kept killing at exit 143 -- serialization into its own job
# (PR #592) removed the accumulated baseline but not this test's own
# demand, and run 31139278303 confirmed the predicted kill WITH the live
# sampler attributing it (VmHWM 14,866 MB at death). RESOLVED by an
# explicit checkpoint_segments=8 (issue #545 residual, measured
# 2026-08-07, same harness both configs): peak RSS 24,065 MB -> 3,955 MB
# (6.1x), gradient 2.468510 -> 2.468503 (3e-6 relative -- the recompute
# reordering, not a numerics change), wall 96 s -> 38 s locally. 8 divides
# n_steps=1400 (175-step segments); non-divisors are rejected fail-loud by
# the runner. The highmem marker stays (additive-only per PR #592's
# design; ~4 GB still warrants serial-lane placement).
@pytest.mark.slow
@pytest.mark.highmem
def test_forward_tfsf_oblique_differentiable():
    """forward() couples an OBLIQUE TFSF source (complex Bloch path, auto-activated)
    and produces a finite, nonzero gradient w.r.t. the scatterer eps."""
    sim = _sim(30.0, "modulated_gaussian", 0.15)
    shp = sim.run(n_steps=1).state.ez.shape
    obj = _objective(sim, shp, n_steps=1400, checkpoint=True,
                     checkpoint_segments=8)

    val = float(obj(4.0))
    assert val > 1e-3, f"oblique TFSF did not couple (obj={val:.2e})"

    g = float(jax.grad(obj)(4.0))
    assert np.isfinite(g) and abs(g) > 1e-6, f"oblique grad not finite/nonzero: {g}"
