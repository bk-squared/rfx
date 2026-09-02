"""Issue #304: flux_spectrum must not return exactly 0.0 SILENTLY when the
per-cell E x H* products underflow float32 (XLA flushes subnormals).

The guard is eager-only and value-preserving: it warns (never raises, never
changes the returned flux) when flux == 0.0 everywhere, the accumulators are
nonzero, and a float64 recompute of the identical sum is nonzero — the
decisive artefact-vs-physics witness. Under jit/grad the check is skipped
entirely (tracer-safe; flux_spectrum is on the AD tape via normalize='flux',
PR #172), so no optimization path is perturbed.

Magnitudes below reproduce the empirically confirmed flush: complex64
1e-20 * conj(1e-22) == exactly 0j (product 1e-42 < min normal 1.1755e-38)
while 1e-15 * 1e-22 = 1e-37 survives.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.probes.probes import FluxMonitor, flux_spectrum


def _monitor(e_scale: float, h_scale: float, n_freqs: int = 3) -> FluxMonitor:
    shape = (n_freqs, 4, 4)
    c64 = jnp.complex64
    return FluxMonitor(
        e1_dft=jnp.full(shape, e_scale + 0j, dtype=c64),
        e2_dft=jnp.zeros(shape, dtype=c64),
        h1_dft=jnp.zeros(shape, dtype=c64),
        h2_dft=jnp.full(shape, h_scale + 0j, dtype=c64),
        freqs=jnp.linspace(1e9, 3e9, n_freqs),
        axis=0, index=5,
        dA=jnp.asarray(1.0, dtype=jnp.float32),
        total_steps=100, window="rect", window_alpha=0.25,
    )


def test_flushed_flux_warns_and_value_unchanged():
    """Subnormal-flushed monitor: warn, but still return the flushed zeros."""
    mon = _monitor(1e-20, 1e-22)  # product 1e-42 -> flushed to exactly 0
    with pytest.warns(UserWarning, match="issue #304"):
        flux = flux_spectrum(mon)
    assert np.all(np.asarray(flux) == 0.0), (
        "guard must WARN only — the returned value stays the flushed zeros")


def test_healthy_flux_does_not_warn():
    mon = _monitor(1e-3, 1e-3)  # products far above min normal
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        flux = flux_spectrum(mon)
    assert float(np.max(np.abs(np.asarray(flux)))) > 0


def test_empty_monitor_does_not_warn():
    """All-zero accumulators = genuinely zero flux, not an artefact."""
    mon = _monitor(0.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flux = flux_spectrum(mon)
    assert np.all(np.asarray(flux) == 0.0)


def test_traced_call_skips_guard_and_matches_eager():
    """Under jit the guard must not fire and must not perturb the value.

    Traces the ARRAY leaves only (static str/int fields stay Python values,
    matching production usage where flux_spectrum runs inside jitted
    extraction with traced accumulators, PR #172).
    """
    mon = _monitor(1e-20, 1e-22)

    def _f(e1_dft, h2_dft):
        return flux_spectrum(mon._replace(e1_dft=e1_dft, h2_dft=h2_dft))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a traced warn would fail here
        jitted = jax.jit(_f)(mon.e1_dft, mon.h2_dft)
    with pytest.warns(UserWarning, match="issue #304"):
        eager = flux_spectrum(mon)
    np.testing.assert_array_equal(np.asarray(jitted), np.asarray(eager))
