"""A/B identity gate for the W6.1 shared Yee step kernel, and W6.2 explicit
checkpoint_segments guard for run_until_decay.

``run`` (jax.lax.scan) and ``run_until_decay`` (Python loop + jax.jit) now
share ONE per-step kernel via ``make_core_step``.  This test pins that the
two execution harnesses produce the same trajectory when forced to take the
same number of steps with the same source.

The decay path is forced to run exactly ``N`` steps:
  * ``decay_by=0.0``      → ``val_sq < 0`` is never true, so the early-stop
                            branch can never fire.
  * ``min_steps=N``       → no stop before N steps anyway.
  * ``check_interval>N``  → the decay-check branch is never even evaluated.

Agreement level
---------------
Target was bit-identical (``np.array_equal``).  The two harnesses do NOT
agree bit-for-bit — they agree to ~2.4e-7 relative (float32 epsilon scale).
This is a PRE-EXISTING difference, not introduced by the W6.1 refactor:
``run`` compiles its body inside ``jax.lax.scan`` while ``run_until_decay``
compiles a standalone ``jax.jit`` step driven from a Python loop, and XLA
fuses / reassociates the float32 Yee arithmetic differently between the two.

This was verified by running this exact A/B on the pre-refactor
``simulation.py`` (``git stash`` of the refactor): the pre-refactor code
reports the IDENTICAL ``max abs diff = 7.276e-12`` on the probe series with
byte-identical per-element values, confirming the shared-kernel refactor
reproduces both the scan path and the loop path bit-exactly.

We therefore gate at the pre-existing agreement level via ``np.allclose``
with a comfortable margin (rtol 1e-6 >> measured 2.4e-7; atol 1e-10 >>
measured 6e-11) rather than ``np.array_equal``.  The gate is deliberately
NOT loosened beyond the measured pre-refactor envelope.
"""

import pytest
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import run, run_until_decay, make_source, make_probe

# Pre-existing scan-vs-loop XLA agreement envelope (see module docstring).
# Measured: probe rel ~6.5e-7 (abs 7.3e-12), field rel ~2.4e-7 (abs 5.8e-11).
_RTOL = 1e-6
_ATOL = 1e-10


def _build():
    """Small PEC box, one Gaussian Ez source, one Ez probe."""
    grid = Grid(freq_max=10e9, domain=(0.048, 0.048, 0.048))
    materials = init_materials(grid.shape)
    n_steps = 80

    pulse = GaussianPulse(f0=5e9, bandwidth=5e9)
    src = make_source(grid, (0.018, 0.024, 0.024), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.030, 0.024, 0.024), "ez")
    return grid, materials, n_steps, [src], [prb]


def test_run_until_decay_ab_identity():
    grid, materials, n_steps, sources, probes = _build()

    res_scan = run(
        grid, materials, n_steps,
        sources=sources, probes=probes,
        return_state=True,
    )

    res_loop = run_until_decay(
        grid, materials,
        decay_by=0.0,
        check_interval=n_steps + 1,
        min_steps=n_steps,
        max_steps=n_steps,
        monitor_component="ez",
        sources=sources, probes=probes,
        return_state=True,
    )

    # Both harnesses must have taken exactly n_steps.
    assert res_scan.time_series.shape[0] == n_steps
    assert res_loop.time_series.shape[0] == n_steps

    ts_scan = np.asarray(res_scan.time_series)
    ts_loop = np.asarray(res_loop.time_series)
    assert ts_scan.shape == ts_loop.shape
    assert np.allclose(ts_scan, ts_loop, rtol=_RTOL, atol=_ATOL), (
        "probe time series differ beyond pre-existing scan-vs-loop envelope; "
        f"max abs diff = {np.max(np.abs(ts_scan - ts_loop)):.3e}"
    )

    # Final fields: every Yee component must match within the same envelope.
    for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
        a = np.asarray(getattr(res_scan.state, comp))
        b = np.asarray(getattr(res_loop.state, comp))
        assert np.allclose(a, b, rtol=_RTOL, atol=_ATOL), (
            f"final {comp} differs beyond pre-existing envelope; "
            f"max abs diff = {np.max(np.abs(a - b)):.3e}"
        )


def test_checkpoint_segments_raises():
    """W6.2: checkpoint_segments is explicitly unsupported on the decay path.

    run_until_decay uses a Python loop, not jax.lax.scan, so scan-level
    gradient checkpointing does not apply.  Passing checkpoint_segments=N
    must raise NotImplementedError with a descriptive message rather than
    silently accepting the parameter and doing nothing.
    """
    grid, materials, _n, sources, probes = _build()
    with pytest.raises(NotImplementedError, match="checkpoint_segments"):
        run_until_decay(
            grid, materials,
            decay_by=0.0,
            min_steps=1,
            max_steps=1,
            check_interval=2,
            sources=sources,
            probes=probes,
            checkpoint_segments=4,
        )
