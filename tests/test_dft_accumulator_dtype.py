"""DFT accumulator dtype follows the x64 state (issue #477 residual).

Hardcoded complex64 accumulators broke the scan-carry dtype contract the
moment x64 was enabled (the per-step update `field * exp(-j w t)` promotes
to complex128), which blocked any f64 gradient referee. The constructors
now use the same conditional dtype idiom as compute_msl_s_matrix.

x64 is scoped per-test via a context manager — NEVER flipped at module
level (process-global; reds every same-process pytest shard).

``jax.experimental.enable_x64`` was removed in newer JAX releases (this
venv: JAX 0.10.2); the fallback below is the same scoped-flip-with-restore
shim tests/_x64_compat.py provides, applied inline here (folded pre-
existing fix, discovered while landing #579's own tests/_x64_compat usage
— see the #579 PR body).
"""
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.probes.probes import init_dft_probe, init_dft_plane_probe

try:
    from jax import enable_x64
except ImportError:  # older JAX (< ~0.4.31) -- see tests/_x64_compat.py
    from tests._x64_compat import enable_x64


def _grid():
    return Grid(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=1e-3,
                cpml_layers=4)


def test_accumulators_are_complex64_by_default():
    """f32 mode (the default): dtypes unchanged from the pre-#477 behaviour
    — the production path is BIT-identical (verified against pre-change
    code on a thru-line S-matrix during the #477 E3 enablement)."""
    g = _grid()
    freqs = jnp.asarray([5e9], dtype=jnp.float32)
    p = init_dft_probe(g, (0.005, 0.005, 0.005), "ez", freqs)
    pl = init_dft_plane_probe(axis=2, index=5, component="ez", freqs=freqs, grid_shape=g.shape)
    assert p.accumulator.dtype == jnp.complex64
    assert pl.accumulator.dtype == jnp.complex64


def test_accumulators_follow_x64_when_enabled():
    """Under scoped x64 the accumulators must be complex128, so the scan
    carry stays dtype-consistent and an f64 referee can run."""
    with enable_x64():
        g = _grid()
        freqs = jnp.asarray([5e9])
        p = init_dft_probe(g, (0.005, 0.005, 0.005), "ez", freqs)
        pl = init_dft_plane_probe(axis=2, index=5, component="ez", freqs=freqs, grid_shape=g.shape)
        assert p.accumulator.dtype == jnp.complex128
        assert pl.accumulator.dtype == jnp.complex128
    # and back to the default outside the scope
    g2 = _grid()
    p2 = init_dft_probe(g2, (0.005, 0.005, 0.005), "ez",
                        jnp.asarray([5e9], dtype=jnp.float32))
    assert p2.accumulator.dtype == jnp.complex64
