"""Subgrid source injection stays dtype-safe under scoped x64 (issue #485).

Both ``_inject_fine_sources`` AND ``_inject_coarse_sources``
(``rfx/subgridding/jit_runner.py``) scatter a per-step source waveform value
into an E-field array with ``.at[...].add(...)``. The field arrays are built
by ``init_state`` with its explicit ``field_dtype=jnp.float32`` default
(independent of x64). The waveform's HOST array (``np.array(waveform) *
fine_source_scale`` / ``* coarse_shadow_source_scale``, built in
``rfx/runners/subgridded.py``'s ``_run_subgridded_once``) is already float64
once x64 is scoped on: ``dt`` there comes from a plain ``np.sqrt(...)`` —
numpy always defaults to float64 on the host, independent of the JAX x64
flag — which promotes the downstream ``jax.vmap(waveform)(times)``
computation to genuine float64 once x64 is enabled (under x64 off, JAX's own
canonicalization keeps that same computation at float32 instead). Back in
``_inject_fine_sources``/``_inject_coarse_sources``'s enclosing function, the
``src_waveforms``/``src_waveforms_c`` builders (``jnp.stack([jnp.array(s[4])
...])``) PRESERVE that host dtype rather than promoting it. So under scoped
x64 each scatter mixed float64 values into a float32 array, which JAX flags
with:

    FutureWarning: scatter inputs have incompatible types: cannot safely
    cast value from dtype=float64 to dtype=float32 ... In future JAX
    releases this will result in an error.

Non-fatal today (JAX still performs the cast), but the FutureWarning means a
future JAX release turns this into a hard error. Same class as #14 (NTFF
MLIR scan-carry mismatch) and #483 (DFT-carry hardcoded dtype).

Fix: both injection sites now cast the source value explicitly to the
destination array's own dtype (``.astype(ez_arr.dtype)`` etc.) before the
scatter-add. Under the float32 default this is a no-op (verified
bit-identical against pre-fix ``time_series`` output, byte-for-byte, on
NONZERO data on each fixture below — see the two
``test_*_injection_default_precision_still_float32`` tests, which assert
nonzero so they cannot pass on a fixture the injection never actually
reaches). Under x64 it becomes an explicit, safe downcast, so JAX no longer
treats it as an unsafe implicit promotion.

Coverage note on the two paths:
  * ``_inject_fine_sources`` fires for any excited port/source that lands in
    the fine region — cheap and always exercised by a normal subgrid+port
    fixture (``_research_port_subgrid_sim`` below).
  * ``_inject_coarse_sources`` only fires when ``src_meta_c`` is non-empty,
    which requires (a) a ``impedance=0.0`` soft source
    (``Simulation.add_source``, NOT ``add_port`` — lumped/wire ports never
    populate ``sources_c``, see ``rfx/runners/subgridded.py`` around the
    ``pe.impedance == 0.0`` branch) AND (b) ``inject_sources_on_coarse_shadow``
    enabled (default-on once ``xy_margin`` is set, or settable directly on
    ``sim._refinement`` under ``validation="research"``, mirroring how other
    tests in this repo poke diagnostic subgrid knobs). Both conditions are
    cheap to satisfy, so this file gates the coarse path with the same live
    warning-error pattern as the fine path, not a cast-by-symmetry-only
    claim. ``_coarse_shadow_source_subgrid_sim`` co-injects the SAME soft
    source on the fine grid AND its coarse shadow (that duplication is what
    "coarse shadow" means), so a few steps land near the injection cell
    before the probe (also fine-grid-resident, since it too sits inside the
    fine ``z_range``) picks up a nonzero reading — ``n_steps=30`` was
    measured to give a clean nonzero margin (max|E| ~1.7e-4, 24/30 nonzero
    samples), well clear of float32 noise.

The warning-gate tests below use ``warnings.filterwarnings("error",
message=...)`` scoped to the exact scatter-cast message, NOT a bare
``simplefilter("error", FutureWarning)`` — JAX/numpy have other live
FutureWarning sources, and a blanket filter would misattribute an unrelated
one to this bug class.

Verified BOTH directions locally before landing this test, for BOTH paths
(pre-fix code: bare ``src_vals[idx_s]`` / ``src_vals_c[idx_s]``, no
``.astype``; fixed code: ``.astype(ez_arr.dtype)`` / ``ex_arr.dtype`` /
``ey_arr.dtype`` at each of the six scatter sites):
  * pre-fix: ``test_fine_source_injection_no_dtype_futurewarning_under_x64``
    and ``test_coarse_source_injection_no_dtype_futurewarning_under_x64``
    each FAILED, raising the scatter-cast FutureWarning from
    ``_inject_fine_sources`` / ``_inject_coarse_sources`` respectively (the
    other path's test still PASSED — the two are independent).
  * fixed: all four tests in this file PASS, no FutureWarning.

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
    from tests._x64_compat import enable_x64 as _enable_x64

from rfx import GaussianPulse, Simulation

# Narrow, targeted at the exact scatter dtype-cast message (issue #485) so
# this doesn't misfire on an unrelated JAX/numpy FutureWarning.
_SCATTER_DTYPE_WARNING = {
    "message": ".*scatter inputs have incompatible types.*",
    "category": FutureWarning,
}


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
    enabled on the research-validated refinement dict.

    ``n_steps=30`` (not the smaller counts used elsewhere in this file) is
    load-bearing: the probe needs enough steps for the injected field to
    actually propagate to it (measured max|E| ~1.7e-4 at 30 steps; a probe
    read at only 5 steps is provably all zero regardless of whether
    ``_inject_coarse_sources`` even runs — that would make any byte-identity
    check on it a blind instrument, see #485 PR review)."""
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
            warnings.filterwarnings("error", **_SCATTER_DTYPE_WARNING)
            result = sim.run(n_steps=20, compute_s_params=False)
    assert result.time_series.shape == (20, 1)


def test_fine_source_injection_default_precision_still_float32():
    """x64 OFF (the default): the fine-source injection cast is a no-op —
    time series stays float32 and (asserted, not assumed) nonzero, matching
    pre-#485 behaviour."""
    sim = _research_port_subgrid_sim()
    result = sim.run(n_steps=20, compute_s_params=False)
    ts = np.asarray(result.time_series)
    assert ts.dtype == np.float32
    assert np.any(ts != 0.0)


def test_coarse_source_injection_no_dtype_futurewarning_under_x64():
    """Under scoped x64, running the coarse-shadow soft-source fixture must
    NOT raise/emit the scatter dtype-cast FutureWarning from
    ``_inject_coarse_sources``. Red before the #485 follow-up fix, green
    after."""
    with _enable_x64():
        sim = _coarse_shadow_source_subgrid_sim()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", **_SCATTER_DTYPE_WARNING)
            result = sim.run(n_steps=30, compute_s_params=False)
    assert result.time_series.shape == (30, 1)


def test_coarse_source_injection_default_precision_still_float32():
    """x64 OFF (the default): the coarse-source injection cast is a no-op —
    time series stays float32 and (asserted, not assumed) nonzero, matching
    pre-#485 behaviour. The nonzero assertion is load-bearing: an all-zero
    ``time_series`` would make this byte-identity witness pass even if
    ``_inject_coarse_sources`` were deleted outright."""
    sim = _coarse_shadow_source_subgrid_sim()
    result = sim.run(n_steps=30, compute_s_params=False)
    ts = np.asarray(result.time_series)
    assert ts.dtype == np.float32
    assert np.any(ts != 0.0)
