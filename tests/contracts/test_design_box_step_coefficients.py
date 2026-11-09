"""Design-window material values and gradients equal whole-grid overrides."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from tests._x64_compat import enable_x64


@pytest.mark.parametrize("design_sigma", [False, True], ids=["background-sigma", "design-sigma"])
def test_design_box_value_and_gradients_match_whole_grid(design_sigma):
    with enable_x64():
        sim = Simulation(freq_max=10e9, domain=(.014, .013, .012), dx=.001,
                         boundary="pec", precision="float64")
        rng = np.random.default_rng(1179)
        # Independent, random lossy materials in eight regions, including every wall.
        split = np.array((7, 6, 6))
        shape = (15, 14, 13)
        indices = np.indices(shape)
        background_eps, background_sigma = np.empty(shape), np.empty(shape)
        for octant in itertools.product((0, 1), repeat=3):
            e, s = float(np.float32(rng.uniform(1., 4.))), float(np.float32(rng.uniform(0., .05)))
            lower = tuple(split[a] * .001 if octant[a] else -1. for a in range(3))
            upper = tuple(1. if octant[a] else split[a] * .001 for a in range(3))
            name = 'background' + ''.join(map(str, octant))
            sim.add_material(name, eps_r=e, sigma=s)
            sim.add(Box(lower, upper), material=name)
            mask = np.logical_and.reduce([(indices[a] >= split[a]) == bool(octant[a]) for a in range(3)])
            background_eps[mask], background_sigma[mask] = e, s
        sim.add_source((.002, .004, .003), 'ez', amplitude_kind='field',
                       waveform=GaussianPulse(f0=8e9, bandwidth=.8))
        sim.add_probe((.010, .008, .007), 'ez')
        assert sim.freeze_mesh().shape == shape
        box = (slice(4, 9), slice(3, 8), slice(5, 9))
        eps = jnp.asarray(rng.uniform(1., 6., (5, 5, 4)), jnp.float64)
        sigma = jnp.asarray(rng.uniform(0., .2, (5, 5, 4)), jnp.float64)
        bg_e, bg_s = jnp.asarray(background_eps), jnp.asarray(background_sigma)
        def loss(e, s, window):
            options = dict(n_steps=80, checkpoint=False, skip_preflight=True)
            if window:
                options.update(design_box=((.004, .003, .005), (.008, .007, .008)),
                               design_eps_override=e)
                if design_sigma:
                    options['design_sigma_override'] = s
            else:
                options.update(eps_override=bg_e.at[box].set(e),
                               sigma_override=bg_s.at[box].set(s) if design_sigma else bg_s)
            return jnp.sum(sim.forward(**options).time_series ** 2)
        argnums = (0, 1) if design_sigma else (0,)
        actual, gradients = jax.value_and_grad(lambda e, s: loss(e, s, True), argnums=argnums)(eps, sigma)
        expected, references = jax.value_and_grad(lambda e, s: loss(e, s, False), argnums=argnums)(eps, sigma)
        assert np.isfinite(expected) and float(expected) > 0, 'reference objective must be live'
        comparisons = [('value', np.asarray(actual), np.asarray(expected))]
        comparisons += list(zip(('eps-gradient', 'sigma-gradient'), map(np.asarray, gradients), map(np.asarray, references)))
        # rfx-archive rfx/records/20261109-test-convert/DESIGN.md A12: lossy background.
        # Measured: the two AD paths differ by 1e-10 to 1e-9 in the loss-term derivative.
        for name, got, ref in comparisons:
            norm = float(np.linalg.norm(ref))
            assert norm > 0 and np.isfinite(got).all(), f'{name}: vanished reference or nonfinite result'
            relative = float(np.linalg.norm(got - ref)) / norm
            print(f'design_sigma={design_sigma} {name}: mesh={shape} steps=80 expected_norm={norm:.17g} measured_norm={np.linalg.norm(got):.17g} relative={relative:.17g} bar=1e-8')
            assert relative <= 1e-8, f'{name}: relative error {relative:.17g} exceeds 1e-8'
