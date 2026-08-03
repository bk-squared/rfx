"""Subgrid source injection stays dtype-safe under scoped x64 (issue #485).

Both ``_inject_fine_sources`` AND ``_inject_coarse_sources``
(``rfx/subgridding/jit_runner.py``) scatter a per-step source waveform value
into an E-field array with ``.at[...].add(...)``. The field arrays are built
by ``init_state`` with its explicit ``field_dtype=jnp.float32`` default
(independent of x64), while the waveform arrays (``src_vals`` /
``src_vals_c``) are built via ``jnp.array(...)`` (jit_runner.py ~2615 /
~2627), which DOES track x64: float32 with x64 off, float64 once
``jax.experimental.enable_x64()`` is scoped on. So under scoped x64 each
scatter mixed float64 values into a float32 array, which JAX flags with:

    FutureWarning: scatter inputs have incompatible types: cannot safely
    cast value from dtype=float64 to dtype=float32 ... In future JAX
    releases this will result in an error.

Non-fatal today (JAX still performs the cast), but the FutureWarning means a
future JAX release turns this into a hard error. Same class as #14 (NTFF
MLIR scan-carry mismatch) and #483 (DFT-carry hardcoded dtype).

Fix: both injection sites now cast the source value explicitly to the
destination array's own dtype (``.astype(ez_arr.dtype)`` etc.) before the
scatter-add. Under the float32 default this is a no-op (verified
bit-identical against pre-fix ``time_series`` output, byte-for-byte, on each
fixture below). Under x64 it becomes an explicit, safe downcast, so JAX no
longer treats it as an unsafe implicit promotion.

Coverage note on the two paths:
  * ``_inject_fine_sources`` fires for any excited port/source that lands in
    the fine region — cheap and always exercised by a normal subgrid+port
    fixture (``_research_port_subgrid_sim`` below).
  * ``_inject_coarse_sources`` only fires when ``src_meta_c`` is non-empty,
    which requires (a) a ``impedance=0.0`` soft source (``Simulation.add_source``,
    NOT ``add_port`` — lumped/wire ports never populate ``sources_c``, see
    ``rfx/runners/subgridded.py`` around the ``pe.impedance == 0.0`` branch)
    AND (b) ``inject_sources_on_coarse_shadow`` enabled (default-on once
    ``xy_margin`` is set, or settable directly on ``sim._refinement`` under
    ``validation="research"``, mirroring how other tests in this repo poke
    diagnostic subgrid knobs). Both conditions are cheap to satisfy — no
    special coarse-fed configuration was needed — so this file gates the
    coarse path with the same live ``simplefilter("error", ...)`` pattern as
    the fine path, not a cast-by-symmetry-only claim.

Verified BOTH directions locally before landing this test, for BOTH paths:
  * pre-fix code (bare ``src_vals[idx_s]`` / ``src_vals_c[idx_s]``, no
    ``.astype``):
    ``test_fine_source_injection_no_dtype_futurewarning_under_x64`` FAILED at
    ``rfx/subgridding/jit_runner.py:1758`` inside ``_inject_fine_sources``;
    ``test_coarse_source_injection_no_dtype_futurewarning_under_x64`` FAILED
    at ``rfx/subgridding/jit_runner.py:1776`` inside ``_inject_coarse_sources``.
  * fixed code (``.astype(ez_arr.dtype)`` / ``ex_arr.dtype`` /
    ``ey_arr.dtype`` at each of the six scatter sites): both tests PASSED,
    no FutureWarning raised.

x64 is scoped per-test via a context manager here — NEVER flip
jax_enable_x64 at module level (process-global; reds every same-process
pytest-split shard, see repo memory
``feedback_jax_x64_module_level_tests``).
"""
from __future__ import annotations

import warnings

import numpy as np

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


def _coarse_shadow_source_subgrid_sim() -> Simulation:
    """Minimal subgrid + soft-source fixture that exercises
    ``_inject_coarse_sources``. A soft source (``add_source``, impedance 0)
    is required — lumped/wire ports (``add_port``) never populate
    ``sources_c`` — plus ``inject_sources_on_coarse_shadow`` explicitly
    enabled on the research-validated refinement dict."""
    sim = Simulation(freq_max=4e9, domain=(0.012, 0.012, 0.012), boundary="pec", dx=0.004)
    sim.add_refinement(z_range=(0.004, 0.012), ratio=3, validation="research")
    sim._refinement["inject_sources_on_coarse_shadow"] = True
    sim.add_source((0.004, 0.004, 0.008), "ez")
    sim.add_probe((0.008, 0.008, 0.008), "ez")
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
    sim = _research_port_subgrid_sim()
    result = sim.run(n_steps=20, compute_s_params=False)
    assert np.asarray(result.time_series).dtype == np.float32


def test_coarse_source_injection_no_dtype_futurewarning_under_x64():
    """Under scoped x64, running the coarse-shadow soft-source fixture must
    NOT raise/emit the scatter dtype-cast FutureWarning from
    ``_inject_coarse_sources``. Red before the #485 follow-up fix, green
    after."""
    with _enable_x64():
        sim = _coarse_shadow_source_subgrid_sim()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = sim.run(n_steps=5, compute_s_params=False)
    assert result.time_series.shape == (5, 1)


def test_coarse_source_injection_default_precision_still_float32():
    """x64 OFF (the default): the coarse-source injection cast is a no-op —
    time series stays float32, matching pre-#485 behaviour."""
    sim = _coarse_shadow_source_subgrid_sim()
    result = sim.run(n_steps=5, compute_s_params=False)
    assert np.asarray(result.time_series).dtype == np.float32
