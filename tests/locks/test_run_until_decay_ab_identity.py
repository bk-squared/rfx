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

We therefore gate at the reassociation agreement level via ``np.allclose``
rather than ``np.array_equal``.  ``rtol`` stays at the tight pre-existing 1e-6
(>> the CPU-measured 2.4e-7 relative), which binds the large elements on CPU.
The absolute floor is no longer the flat ``atol = 1e-10`` -- that literal was
calibrated to the CPU reassociation scale and is EXCEEDED on GPU (RTX4090:
max abs diff 1.037e-10 > 1e-10, ~64*eps_f32 relative, a CPU-vs-GPU
reduction-order artefact).  It is replaced by ``_reassoc_atol``, an absolute
floor DERIVED AT RUNTIME from the working dtype's epsilon, the step count, and
the field magnitude (``C * n_steps * eps * max|field|``); see the block above
that helper for the full root-cause derivation.  The floor is set from a
numerical bound with margin, not reverse-engineered from the observed diff, and
``test_reassoc_floor_still_reds_on_physical_perturbation`` proves it still reds
on a physically meaningful (0.1%) divergence.
"""

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "hand-derived",
    "commit": "7fb7dcf",
    "date": "2026-06-11",
    "run_id": "unknown",
    "host": "unknown",
    "pinned_until": "2026-12-08",
}

import pytest
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import run, run_until_decay, make_source, make_probe

# Pre-existing scan-vs-loop XLA agreement envelope (see module docstring).
# Measured: probe rel ~6.5e-7 (abs 7.3e-12), field rel ~2.4e-7 (abs 5.8e-11).
_RTOL = 1e-6

# --- Derived scan-vs-jitloop reassociation floor -----------------------------
#
# (Replaces the old flat ``_ATOL = 1e-10``.  That literal was calibrated to the
#  CPU reassociation scale and is EXCEEDED on GPU: VESSL 369367258329 / RTX4090
#  reports max abs diff 1.037e-10 > 1e-10 on the probe series, at a field peak
#  ~1.3e-5, i.e. ~64*eps_f32 relative -- a CPU-vs-GPU reduction-order artefact,
#  not a physics regression.)
#
# Root cause: ``run`` compiles the shared Yee kernel inside ``jax.lax.scan``
# while ``run_until_decay`` compiles a standalone ``jax.jit`` step driven from a
# Python loop.  XLA fuses / reassociates the float32 reductions differently
# between the two harnesses, so their trajectories differ purely by
# floating-point REASSOCIATION -- the SAME kernel, two execution harnesses, no
# physics difference.  On GPU the reduction order differs again from CPU and the
# difference is larger (GPU coalesces reductions across lanes, so per-step
# rounding adds more coherently than the CPU's near-random-walk order).
#
# Bound (numerical limitation, stated so the floor generalizes instead of being
# pinned to this fixture): the forward rounding error of an N-step float32
# recurrence grows AT MOST LINEARLY in the step count N (worst case: per-step
# rounding accumulates coherently rather than as a sqrt(N) random walk).  The
# GPU reduction reordering realizes close to this coherent worst case -- measured
# ~0.8*eps per step (64*eps over N=80) -- while CPU stays near the sqrt(N) walk
# (~6.9*eps ~= 0.77*sqrt(80)).  The rounding scales against the LARGEST field
# excursion in the compared state, because roundoff in the dominant component
# propagates into every component (including numerically-zero ones such as hz
# here) through the Yee curl coupling.  Hence the absolute floor is
#
#     atol = C * n_steps * finfo(dtype).eps * max|field over compared arrays|
#
# with C an O(1) per-step coherence constant.  We set C = 4: ~5x margin over the
# measured GPU coherence (~0.8), also absorbing the factor 2 from differencing
# two independently-rounded harnesses (triangle inequality on both forward
# errors).  Everything is read at RUNTIME -- eps and dtype from the array,
# n_steps from the run, the magnitude from the arrays -- so NOTHING is pinned to
# the observed 1.037e-10; the floor tracks dtype, problem magnitude and run
# length instead of this one fixture.  K = C*n_steps stays O(hundreds) of eps
# here (=> ~3.8e-5 relative), still >20x below any physically meaningful
# (>=1e-3 relative) change, which the regression test below proves it still reds.
_REASSOC_COHERENCE_C = 4.0


def _reassoc_atol(arrays, n_steps):
    """Magnitude- and dtype-relative absolute floor for scan-vs-jitloop float
    reassociation.

    ``arrays`` are compared against each other and share ONE floor set by their
    largest excursion -- the coupled-state roundoff scale -- so a numerically
    zero component (e.g. hz for an Ez source) is bounded by the dominant field's
    noise, not by its own (meaningless) ~0 magnitude.
    """
    arrays = [np.asarray(a) for a in arrays]
    dtype = arrays[0].dtype
    eps = float(np.finfo(dtype).eps)
    scale = max(float(np.max(np.abs(a))) for a in arrays)
    return _REASSOC_COHERENCE_C * float(n_steps) * eps * scale


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
    atol_ts = _reassoc_atol([ts_scan, ts_loop], n_steps)
    assert np.allclose(ts_scan, ts_loop, rtol=_RTOL, atol=atol_ts), (
        "probe time series differ beyond derived scan-vs-jitloop reassociation "
        f"floor (atol={atol_ts:.3e}); "
        f"max abs diff = {np.max(np.abs(ts_scan - ts_loop)):.3e}"
    )

    # Final fields: every Yee component must match within the same envelope.
    # One floor is shared across all components (set by the largest field
    # excursion), so numerically-zero components (e.g. hz) are bounded by the
    # coupled-state roundoff scale rather than their own ~0 magnitude.
    comps = ("ex", "ey", "ez", "hx", "hy", "hz")
    field_arrays = (
        [np.asarray(getattr(res_scan.state, c)) for c in comps]
        + [np.asarray(getattr(res_loop.state, c)) for c in comps]
    )
    atol_field = _reassoc_atol(field_arrays, n_steps)
    for comp in comps:
        a = np.asarray(getattr(res_scan.state, comp))
        b = np.asarray(getattr(res_loop.state, comp))
        assert np.allclose(a, b, rtol=_RTOL, atol=atol_field), (
            f"final {comp} differs beyond derived reassociation floor "
            f"(atol={atol_field:.3e}); "
            f"max abs diff = {np.max(np.abs(a - b)):.3e}"
        )


def test_reassoc_floor_still_reds_on_physical_perturbation():
    """The derived reassociation floor must NOT hide a real divergence.

    Loosening the A/B floor from the flat 1e-10 to the magnitude/step-derived
    ``_reassoc_atol`` is only defensible if it still reds on a physically
    meaningful difference.  This pins both ends:

      (a) the GPU-OBSERVED reassociation magnitude (1.037e-10, VESSL
          369367258329 / RTX4090, quoted here as evidence not as a threshold)
          PASSES under the derived floor with margin -- the whole point of the
          fix; and
      (b) a physically meaningful perturbation (0.1% of the field peak,
          ~1e4x the reassociation noise and >20x the derived floor) FAILS.

    (a) also demonstrates the derived floor -- computed here on the CPU-scale
    probe peak, which is SMALLER than the GPU peak -- already covers the larger
    GPU-observed diff, so it covers both the CPU (~1e-11) and GPU (1.037e-10)
    reassociation levels with margin.
    """
    grid, materials, n_steps, sources, probes = _build()
    res = run(
        grid, materials, n_steps,
        sources=sources, probes=probes,
        return_state=True,
    )
    ts = np.asarray(res.time_series)
    atol_ts = _reassoc_atol([ts], n_steps)
    imax = int(np.argmax(np.abs(ts)))
    peak = float(np.max(np.abs(ts)))

    # (a) The GPU-observed reassociation diff is below the derived floor.
    gpu_observed = 1.037e-10  # evidence, not a hardcoded tolerance
    assert atol_ts > gpu_observed, (
        f"derived floor {atol_ts:.3e} must cover the GPU-observed "
        f"reassociation diff {gpu_observed:.3e}"
    )
    reassoc = ts.copy()
    reassoc[imax] += np.asarray(gpu_observed, dtype=ts.dtype)
    assert np.allclose(ts, reassoc, rtol=_RTOL, atol=atol_ts), (
        "derived floor should accept a GPU-scale reassociation diff"
    )

    # (b) A physically meaningful perturbation must still red the lock.
    delta = 1e-3 * peak  # 0.1% of the field peak
    assert delta > 10.0 * atol_ts, (
        f"perturbation {delta:.3e} must sit well above the reassociation "
        f"floor {atol_ts:.3e} to be a fair regression probe"
    )
    perturbed = ts.copy()
    perturbed[imax] += np.asarray(delta, dtype=ts.dtype)
    assert not np.allclose(ts, perturbed, rtol=_RTOL, atol=atol_ts), (
        "derived reassociation floor swallowed a 0.1% physical perturbation -- "
        "it would hide a real scan-vs-jitloop divergence"
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
