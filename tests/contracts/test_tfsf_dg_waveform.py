"""TF/SF's DG and GaussianPulse have one waveform definition."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.sources.sources import GaussianPulse
from rfx.sources.tfsf import init_tfsf, line_source_value


def test_dg_line_matches_gaussian_pulse():
    """T2: expected samples come from GaussianPulse, never from the line."""
    f0, bandwidth, amplitude, dt = 8e9, 0.5, 1.7, 1.9065748695310057e-12
    cfg, _ = init_tfsf(61, 1e-3, dt, cpml_layers=8, tfsf_margin=4,
                       f0=f0, bandwidth=bandwidth)
    times = jnp.arange(2001, dtype=jnp.float32) * dt
    expected = jax.vmap(GaussianPulse(f0, bandwidth, amplitude, cutoff=3))(times)
    actual = jax.vmap(lambda t: line_source_value(cfg, t, amplitude))(times)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def test_dg_line_refuses_cutoff_below_gaussian_minimum():
    cfg, _ = init_tfsf(61, 1e-3, 1e-12, cpml_layers=8, tfsf_margin=4)
    cfg = cfg._replace(src_t0=2.4 * cfg.src_tau)
    with pytest.raises(ValueError, match=r"GaussianPulse cutoff=.*is below 2.5"):
        line_source_value(cfg, 0., cfg.src_amp)
