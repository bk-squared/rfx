"""The ``normalize="flux"`` waveguide S-matrix dtype follows ``JAX_ENABLE_X64``.

Until v1.8 the uniform flux lane hard-cast its assembled S column to
complex64 (``extract_waveguide_s_matrix_flux``), so it ignored the
precision knob that ``normalize=False`` honours through ``_rect_dft``
and that the non-uniform flux lane never overrode. Decision 1 of the
v1.8 chain-closure plan (``docs/design_notes/v18_waveguide_s_chain_plan.md``,
Appendix B) removed the cast. This test pins the outcome on both
settings so the cast cannot come back silently: complex64 with x64 off,
complex128 with x64 on, for ``False`` and ``"flux"`` alike.

x64 is scoped per test through the context manager, never flipped at
module level (that leaks into every same-process shard).
"""

from __future__ import annotations

import contextlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import enable_x64 as _enable_x64  # jax >= 0.8: top-level, takes a bool
except ImportError:
    try:
        from jax.experimental import enable_x64 as _enable_x64  # jax 0.6.x (CI pin)
    except ImportError:  # neither: same semantics, scoped flip with restore

        @contextlib.contextmanager
        def _enable_x64(flag: bool):
            prev = bool(jax.config.read("jax_enable_x64"))
            jax.config.update("jax_enable_x64", flag)
            try:
                yield
            finally:
                jax.config.update("jax_enable_x64", prev)

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec, Boundary


def _wr90_sim():
    """Tiny 2-port WR-90, the golden fixture of tests/test_waveguide_sparam_ad.py."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=0.003,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )
    freqs = jnp.linspace(5e9, 6.5e9, 4)
    sim.add_waveguide_port(0.010, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=freqs, f0=6e9, bandwidth=0.5, name="left")
    sim.add_waveguide_port(0.090, direction="-x", mode=(1, 0), mode_type="TE",
                           freqs=freqs, f0=6e9, bandwidth=0.5, name="right")
    return sim


@pytest.mark.parametrize("x64, expected", [(False, "complex64"), (True, "complex128")])
@pytest.mark.parametrize("normalize", [False, "flux"])
def test_waveguide_s_dtype_follows_x64(normalize, x64, expected):
    with _enable_x64(x64):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _wr90_sim().compute_waveguide_s_matrix(num_periods=4, normalize=normalize)
        dtype = str(res.s_params.dtype)
        finite = bool(np.all(np.isfinite(np.asarray(res.s_params))))
    print(f"\n[normalize={normalize}] x64={x64}: s_params.dtype={dtype}")
    assert dtype == expected, (
        f"normalize={normalize} under JAX_ENABLE_X64={x64} returned {dtype}, "
        f"expected {expected}: a dtype pin is back on this lane."
    )
    assert finite
