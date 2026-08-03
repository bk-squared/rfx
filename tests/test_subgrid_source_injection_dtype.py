"""Subgrid fine-source injection stays dtype-safe under scoped x64 (issue #485).

``_inject_fine_sources`` (``rfx/subgridding/jit_runner.py``) scatters the
per-step source waveform value into the fine E-field arrays with
``.at[...].add(...)``. The fine field arrays are built by ``init_state``
with its explicit ``field_dtype=jnp.float32`` default (independent of x64),
while the waveform array is built via ``jnp.array(...)`` (jit_runner.py
~2615), which DOES track x64: float32 with x64 off, float64 once
``jax.experimental.enable_x64()`` is scoped on. So under scoped x64 the
scatter mixed float64 values into a float32 array, which JAX flags with:

    FutureWarning: scatter inputs have incompatible types: cannot safely
    cast value from dtype=float64 to dtype=float32 ... In future JAX
    releases this will result in an error.

Non-fatal today (JAX still performs the cast), but the FutureWarning means a
future JAX release turns this into a hard error. Same class as #14 (NTFF
MLIR scan-carry mismatch) and #483 (DFT-carry hardcoded dtype).

Fix: the injection site now casts the source value explicitly to the
destination array's own dtype (``.astype(ez_arr.dtype)`` etc.) before the
scatter-add. Under the float32 default this is a no-op (verified
bit-identical against pre-fix ``time_series`` output, byte-for-byte, on this
same fixture — see ``docs/agent-memory`` / PR record for #485). Under x64 it
becomes an explicit, safe downcast, so JAX no longer treats it as an unsafe
implicit promotion.

Verified BOTH directions locally before landing this test:
  * with the pre-fix code (bare ``src_vals[idx_s]``, no ``.astype``):
    ``test_fine_source_injection_no_dtype_futurewarning_under_x64`` FAILED
    (the ``warnings.simplefilter("error", FutureWarning)`` context turned
    the warning into a raised ``FutureWarning`` at
    ``rfx/subgridding/jit_runner.py:1758``, inside ``_inject_fine_sources``).
  * with the fix (``.astype(ez_arr.dtype)`` / ``ex_arr.dtype`` /
    ``ey_arr.dtype``) applied: the same test PASSED, no FutureWarning raised.

x64 is scoped per-test via a context manager here — NEVER flip
jax_enable_x64 at module level (process-global; reds every same-process
pytest-split shard, see repo memory
``feedback_jax_x64_module_level_tests``).
"""
from __future__ import annotations

import warnings

try:  # modern JAX: scoped x64 promoted to top-level (experimental removed v0.8.0)
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX (< ~0.4.31)
    from jax.experimental import enable_x64 as _enable_x64

from rfx import GaussianPulse, Simulation


def _research_port_subgrid_sim() -> Simulation:
    """Minimal subgrid + excited-port fixture that exercises
    ``_inject_fine_sources`` (mirrors ``test_subgrid_port_research.py``'s
    ``_research_port_subgrid_sim``, trimmed to the excited-port-only case
    that is sufficient to reach the scatter site)."""
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002, cpml_layers=0)
    sim.add_refinement(z_range=(0.002, 0.024), ratio=2, validation="research")
    sim._refinement["use_boundary_terminated_exterior_z_interfaces"] = True
    sim.add_port(
        (0.04 / 3.0, 0.04 / 3.0, 0.002 + 0.45 * 0.022),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3.5e9, bandwidth=0.8),
        excite=True,
    )
    sim.add_probe((0.02, 0.02, 0.002 + 0.55 * 0.022), "ez")
    return sim


def test_fine_source_injection_no_dtype_futurewarning_under_x64():
    """Under scoped x64, running the subgrid excited-port fixture must NOT
    raise/emit the scatter dtype-cast FutureWarning from
    ``_inject_fine_sources``. Red before the #485 fix, green after."""
    with _enable_x64():
        sim = _research_port_subgrid_sim()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = sim.run(n_steps=20, compute_s_params=False)
    assert result.time_series.shape == (20, 1)


def test_fine_source_injection_default_precision_still_float32():
    """x64 OFF (the default): the fine-source injection cast is a no-op —
    time series stays float32, matching pre-#485 behaviour."""
    import numpy as np

    sim = _research_port_subgrid_sim()
    result = sim.run(n_steps=20, compute_s_params=False)
    assert np.asarray(result.time_series).dtype == np.float32
