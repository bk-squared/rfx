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
fuses / contracts the float32 Yee arithmetic differently between the two.

This was verified by running this exact A/B on the pre-refactor
``simulation.py`` (``git stash`` of the refactor): the pre-refactor code
reports the IDENTICAL ``max abs diff = 7.276e-12`` on the probe series with
byte-identical per-element values, confirming the shared-kernel refactor
reproduces both the scan path and the loop path bit-exactly.

We therefore gate at the reassociation agreement level via ``np.allclose``
rather than ``np.array_equal``.  ``rtol`` is left at the pre-existing 1e-6;
it is retained only so the gate keeps a relative component if the fixture
ever grows a much larger dynamic range.  Measured on this fixture it binds
NOTHING: ``rtol*|b| < atol`` on every element of every compared array, under
the old flat floor as well as under the derived one, so the gate is in
practice pure-atol.  Do not read "rtol unchanged" as a statement that the
gate's sensitivity is unchanged — the atol is what gates here.

The absolute floor is no longer the flat ``atol = 1e-10``: that literal was
calibrated to the CPU reassociation scale and is EXCEEDED on GPU (RTX4090:
max abs diff 1.037e-10 > 1e-10 on the probe series).  It is replaced by
floors DERIVED AT RUNTIME from the working dtype's epsilon, the step count,
and the magnitude of EACH COMPARED COMPONENT (with the Yee curl's own E↔H
coupling coefficient carrying one group's roundoff into the other); see the
block above ``_curl_roundoff_coupling`` for the full derivation.  Per
component, not one shared floor: the E and H groups differ by ~1/Z0 in SI
units, so a single E-scale floor would leave the H checks inert (it did —
that is the defect this file's second revision fixes).

Sensitivity of the derived floors (see ``_field_reassoc_atols``):

* for any component whose floor is set by its OWN magnitude, the floor sits
  ``C*n_steps*eps / 1e-3`` = 26x below a 0.1%-relative change — an identity
  in C, n_steps and dtype, independent of the fixture;
* for a component whose floor is set by the CROSS-COUPLED roundoff (here
  hx/hy, whose own peaks are below ``k*max|E|``) the margin is smaller and
  is computed at runtime; measured on this fixture 3.1x (hx) / 3.6x (hy), so
  a 0.1% change in H still reds and a 1% change reds at ~30x;
* for a numerically-null component (hz here) no relative statement exists —
  its floor is the coupled-state roundoff by construction, which is exactly
  what keeps it from redding on pure noise.

``test_field_reassoc_floors_still_red_physical_perturbations`` asserts all of
that on the fields themselves (perturbing hx, not only the probe series), and
red-lines the previous shared-E-floor behaviour so it cannot come back.
"""

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "hand-derived",
    "commit": "f019c89",
    "date": "2026-09-05",
    "run_id": "local (CPU derivation) + VESSL 369367258329 / 369367258350 (GPU evidence)",
    "host": "linux x86_64 jax 0.11.1 cpu; GPU evidence remilab RTX4090",
    "pinned_until": "2027-03-05",
}

import pytest
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import init_materials, EPS_0, MU_0
from rfx.sources.sources import GaussianPulse
from rfx.simulation import run, run_until_decay, make_source, make_probe

# Pre-existing scan-vs-loop XLA agreement envelope (see module docstring).
# Measured: probe rel ~6.5e-7 (abs 7.3e-12), field rel ~2.4e-7 (abs 5.8e-11).
_RTOL = 1e-6

# GPU-observed scan-vs-jitloop probe diff (VESSL 369367258329 / RTX4090).
# Quoted as EVIDENCE that the old flat 1e-10 floor is too tight on GPU, and
# used below as the magnitude of a realistic reassociation perturbation. It is
# never used as a threshold.
_GPU_OBSERVED_TS_DIFF = 1.037e-10

# --- Derived scan-vs-jitloop reassociation floors ----------------------------
#
# (Replaces the old flat ``_ATOL = 1e-10``.  That literal was calibrated to the
#  CPU reassociation scale and is EXCEEDED on GPU: VESSL 369367258329 / RTX4090
#  reports max abs diff 1.037e-10 > 1e-10 on the probe series, at a probe peak
#  ~1.3e-5, i.e. ~64*eps_f32 relative.)
#
# Root cause: ``run`` compiles the shared Yee kernel inside ``jax.lax.scan``
# while ``run_until_decay`` compiles a standalone ``jax.jit`` step driven from a
# Python loop.  XLA emits different fusions for the two programs, so the
# float32 curl arithmetic is contracted/associated differently (FMA contraction
# and operand order inside the stencil) -- the SAME kernel, two execution
# harnesses, no physics difference.  Note what this is NOT: the compared
# quantities contain no reduction at all (the probe is a single point sample,
# ``ProbeSpec(i,j,k,component)``, and the Yee update is an elementwise
# stencil), so the CPU-vs-GPU gap is a fusion/contraction-order effect, not a
# reduction-order one.  The GPU number is larger than the CPU one; with a
# single measured N we have no evidence for HOW it grows with N, so the floor
# below uses the worst-case bound rather than a fitted growth law.
#
# Bound (numerical limitation, stated so the floor generalizes instead of being
# pinned to this fixture): the forward rounding error of an N-step float32
# recurrence grows AT MOST LINEARLY in the step count N (worst case: per-step
# rounding accumulates coherently rather than as a sqrt(N) random walk).  So
#
#     atol(component) = C * n_steps * eps * scale(component)
#
# with C an O(1) per-step coherence constant.  We set C = 4: ~5x margin over the
# GPU-observed per-step coherence (1.037e-10 / (80 * eps * 1.3e-5) ~ 0.84 eps
# per step), also absorbing the factor 2 from differencing two independently
# rounded harnesses (triangle inequality on both forward errors) and the
# handful of roundings each curl component costs per step.
#
# ``scale(component)`` -- this is where the first revision of this file was
# wrong.  It used ONE scale, ``max|field|`` over all twelve arrays, arguing
# that roundoff in the dominant component reaches every component through the
# Yee curl.  The propagation is real but it has a COEFFICIENT, and in SI units
# that coefficient is far from 1:
#
#     rfx/core/yee.py:  h = h - (dt / (mu_r*MU_0)) * curl(E),  curl ~ 1/d
#                       e = e + (dt / (eps_r*EPS_0)) * curl(H) (+ loss terms)
#
# so an E-side roundoff ``eps*|E|`` enters H as ``k_he * eps*|E|`` with
# ``k_he = dt/(mu_r*MU_0*d)`` (= 1.517e-3 on this grid), and an H-side roundoff
# enters E as ``k_eh * eps*|H|`` with ``k_eh = dt/(eps_r*EPS_0*d)`` (= 215.3).
# Equivalently ``k_he = S/Z0`` and ``k_eh = S*Z0`` with S the Courant number,
# which is why the two directions are reciprocal and neither is 1.
#
# With one shared E-scale floor the H checks were inert: the floor sat ~4800x
# above the H group's own ``C*N*eps*max|H|`` and ~660x above the correctly
# coupled bound, i.e. at 22% (hx) / 18% (hy) of those components' own peaks --
# a 10% single-element error and a 1% whole-array scale error in hx both
# PASSED.  Each component therefore gets its own scale:
#
#     scale(c) = max( max|c| ,  k_cross * max|partner group| )
#
# which (a) tracks each component's own roundoff, (b) preserves the property
# the shared floor was introduced for -- a numerically-null component (hz for
# an Ez source here; ez for a 2D TE case) is bounded by the coupled-state
# roundoff, not by its own meaningless ~0 magnitude -- and (c) does so with the
# kernel's own coefficient instead of an implicit 1.  Sibling precedent:
# tests/unit/nonuniform/test_nonuniform_until_decay.py keeps separate E and H
# scales for exactly this reason.
#
# Everything is read at RUNTIME -- eps and dtype from the arrays, n_steps from
# the run, the magnitudes from the arrays, dt / cell size / mu_r / eps_r from
# the grid and material arrays -- so NOTHING is pinned to the observed
# 1.037e-10 or to this fixture's geometry.
_REASSOC_COHERENCE_C = 4.0

_E_COMPS = ("ex", "ey", "ez")
_H_COMPS = ("hx", "hy", "hz")


def _curl_roundoff_coupling(grid, materials):
    """Per-step Yee-curl coupling coefficients ``(k_he, k_eh)``.

    ``k_he`` carries an E-side absolute roundoff into H per step
    (``dt / (mu_r*MU_0*d)``); ``k_eh`` carries an H-side roundoff into E
    (``dt / (eps_r*EPS_0*d)``).  Both are read from the grid and the material
    arrays, and both take the worst (largest-coefficient) cell: the smallest
    cell size present and the smallest relative material constant.

    Axis-aware: ``d`` is the minimum of whatever cell sizes the grid exposes,
    so a rectilinear grid is not assumed cubic.
    """
    dx = float(grid.dx)
    d = min(dx, float(getattr(grid, "dy", dx)), float(getattr(grid, "dz", dx)))
    mu_min = float(np.min(np.asarray(materials.mu_r))) * MU_0
    eps_min = float(np.min(np.asarray(materials.eps_r))) * EPS_0
    dt = float(grid.dt)
    return dt / (mu_min * d), dt / (eps_min * d)


def _reassoc_atol(arrays, n_steps):
    """Reassociation floor for ONE group of mutually compared arrays.

    ``C * n_steps * finfo(dtype).eps * max|arrays|``.  Used for the probe time
    series (a single scalar observable).  For the Yee state use
    :func:`_field_reassoc_atols`, which gives each component its own scale.
    """
    arrays = [np.asarray(a) for a in arrays]
    dtype = arrays[0].dtype
    eps = float(np.finfo(dtype).eps)
    scale = max(float(np.max(np.abs(a))) for a in arrays)
    return _REASSOC_COHERENCE_C * float(n_steps) * eps * scale


def _field_reassoc_atols(arrays_by_comp, n_steps, grid, materials):
    """Per-component reassociation floors for the six Yee components.

    ``arrays_by_comp`` maps a component name to the list of arrays being
    compared for it (one per harness).  Each component's floor is set by its
    OWN peak, raised to the cross-coupled roundoff scale
    (``k * max|partner group|``) when that is larger -- which is what bounds a
    numerically-null component.

    Returns ``(atols, scales, peaks)``: the floor, the scale it was built from
    and the component's own peak, all per component, so callers can report
    which term set each floor.
    """
    peaks = {}
    dtype = None
    for comp, arrays in arrays_by_comp.items():
        arrays = [np.asarray(a) for a in arrays]
        if dtype is None:
            dtype = arrays[0].dtype
        peaks[comp] = max(float(np.max(np.abs(a))) for a in arrays)
    eps = float(np.finfo(dtype).eps)
    k_he, k_eh = _curl_roundoff_coupling(grid, materials)

    max_e = max((peaks[c] for c in _E_COMPS if c in peaks), default=0.0)
    max_h = max((peaks[c] for c in _H_COMPS if c in peaks), default=0.0)

    atols, scales = {}, {}
    for comp, peak in peaks.items():
        if comp in _H_COMPS:
            cross = k_he * max_e
        elif comp in _E_COMPS:
            cross = k_eh * max_h
        else:  # pragma: no cover - only the six Yee components are compared
            cross = 0.0
        scales[comp] = max(peak, cross)
        atols[comp] = _REASSOC_COHERENCE_C * float(n_steps) * eps * scales[comp]
    return atols, scales, peaks


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

    # Final fields: every Yee component must match within ITS OWN derived
    # envelope.  A single shared floor would be set by the dominant E
    # component and would leave the H checks inert (see the derivation above).
    comps = _E_COMPS + _H_COMPS
    arrays_by_comp = {
        c: [np.asarray(getattr(res_scan.state, c)),
            np.asarray(getattr(res_loop.state, c))]
        for c in comps
    }
    atols, scales, peaks = _field_reassoc_atols(
        arrays_by_comp, n_steps, grid, materials
    )
    for comp in comps:
        a, b = arrays_by_comp[comp]
        atol = atols[comp]
        floored_by = "own magnitude" if scales[comp] <= peaks[comp] else "curl coupling"
        assert np.allclose(a, b, rtol=_RTOL, atol=atol), (
            f"final {comp} differs beyond its derived reassociation floor "
            f"(atol={atol:.3e}, scale={scales[comp]:.3e} set by {floored_by}, "
            f"own peak={peaks[comp]:.3e}); "
            f"max abs diff = {np.max(np.abs(a - b)):.3e}"
        )


def test_reassoc_floor_still_reds_on_physical_perturbation():
    """The derived reassociation floor must NOT hide a real divergence.

    Loosening the A/B floor from the flat 1e-10 to the magnitude/step-derived
    ``_reassoc_atol`` is only defensible if it still reds on a physically
    meaningful difference.  This pins both ends on the PROBE TIME SERIES (the
    field components are covered by
    ``test_field_reassoc_floors_still_red_physical_perturbations``):

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
    assert atol_ts > _GPU_OBSERVED_TS_DIFF, (
        f"derived floor {atol_ts:.3e} must cover the GPU-observed "
        f"reassociation diff {_GPU_OBSERVED_TS_DIFF:.3e}"
    )
    reassoc = ts.copy()
    reassoc[imax] += np.asarray(_GPU_OBSERVED_TS_DIFF, dtype=ts.dtype)
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


def test_field_reassoc_floors_still_red_physical_perturbations():
    """The per-component FIELD floors must red real field divergences.

    The time-series guard above never touches a field component, so it cannot
    see the failure mode this file was revised for: a floor derived from the
    dominant E magnitude and shared with the H group is ~4800x above the H
    group's own roundoff and swallows gross H errors.  This test perturbs the
    fields themselves and pins, at runtime:

      (a) a reassociation-scale perturbation of hx PASSES -- taken as the
          GPU-observed probe (E-scale) diff pushed through the Yee curl with
          the kernel's own coefficient ``k_he = dt/(mu_r*MU_0*d)``, which is
          what an E-side reassociation of that size does to H in one step;
      (b) a 1% whole-array scale error in hx FAILS;
      (c) a 10% single-element error in hx FAILS;
      (d) the RETIRED shared-E-scale floor would have ACCEPTED both (b) and
          (c) -- the regression witness, so the shared floor cannot return
          unnoticed; and
      (e) every component that carries signal above its own floor still reds a
          0.1%-of-own-peak single-element change.  A component whose peak sits
          at or below its floor is numerically null (hz for this Ez source):
          no relative statement is possible for it, and the test asserts
          instead that its floor came from the coupled roundoff rather than
          from its own ~0 magnitude -- the null-handling property that
          motivated the shared floor in the first place.

    (e) is also the self-guard on the linear-in-N growth of the floor: a future
    fixture edit that pushed ``n_steps`` past ~2100 (where ``C*n_steps*eps``
    reaches 1e-3) would red it rather than silently gate at the 0.1% level.
    """
    grid, materials, n_steps, sources, probes = _build()
    res = run(
        grid, materials, n_steps,
        sources=sources, probes=probes,
        return_state=True,
    )
    comps = _E_COMPS + _H_COMPS
    arrays_by_comp = {
        c: [np.asarray(getattr(res.state, c))] for c in comps
    }
    atols, scales, peaks = _field_reassoc_atols(
        arrays_by_comp, n_steps, grid, materials
    )
    k_he, _k_eh = _curl_roundoff_coupling(grid, materials)

    hx = arrays_by_comp["hx"][0]
    atol_hx = atols["hx"]
    ihx = int(np.argmax(np.abs(hx)))
    hx_peak = peaks["hx"]
    assert hx_peak > atol_hx, (
        f"hx carries no signal above its own floor (peak {hx_peak:.3e}, "
        f"atol {atol_hx:.3e}); this fixture cannot probe the H gate"
    )

    # (a) Reassociation-scale noise on H, derived from the GPU-observed E-scale
    #     diff and the curl coupling coefficient -- must PASS.
    h_reassoc = k_he * _GPU_OBSERVED_TS_DIFF
    assert atol_hx > h_reassoc, (
        f"derived hx floor {atol_hx:.3e} must cover the curl-coupled "
        f"GPU-scale reassociation noise {h_reassoc:.3e} "
        f"(k_he={k_he:.3e} x {_GPU_OBSERVED_TS_DIFF:.3e})"
    )
    noisy = hx.copy()
    noisy[np.unravel_index(ihx, hx.shape)] += np.asarray(h_reassoc, dtype=hx.dtype)
    assert np.allclose(hx, noisy, rtol=_RTOL, atol=atol_hx), (
        "derived hx floor should accept curl-coupled GPU-scale reassociation "
        f"noise {h_reassoc:.3e} (atol {atol_hx:.3e})"
    )

    # (b) 1% whole-array scale error in hx -- must FAIL.
    scaled = (hx * np.asarray(1.01, dtype=hx.dtype)).astype(hx.dtype)
    assert not np.allclose(hx, scaled, rtol=_RTOL, atol=atol_hx), (
        f"derived hx floor {atol_hx:.3e} swallowed a 1% whole-array hx scale "
        f"error (max diff {np.max(np.abs(hx - scaled)):.3e}) -- the H checks "
        "are inert"
    )

    # (c) 10% single-element error in hx -- must FAIL.
    spike = hx.copy()
    spike[np.unravel_index(ihx, hx.shape)] += np.asarray(0.1 * hx_peak, dtype=hx.dtype)
    assert not np.allclose(hx, spike, rtol=_RTOL, atol=atol_hx), (
        f"derived hx floor {atol_hx:.3e} swallowed a 10% single-element hx "
        f"error ({0.1 * hx_peak:.3e}) -- the H checks are inert"
    )

    # (d) Regression witness: the RETIRED shared floor (one E-scale floor for
    #     all six components) accepted both of those. Recomputed here at
    #     runtime, not quoted, so the witness cannot go stale.
    shared_floor = _reassoc_atol([arrays_by_comp[c][0] for c in comps], n_steps)
    assert shared_floor > atol_hx, (
        f"shared E-scale floor {shared_floor:.3e} is not looser than the "
        f"per-component hx floor {atol_hx:.3e}; the regression witness below "
        "would be vacuous"
    )
    assert np.allclose(hx, scaled, rtol=_RTOL, atol=shared_floor), (
        "the retired shared floor is expected to ACCEPT a 1% hx scale error; "
        "if it now reds, this witness needs rewriting"
    )
    assert np.allclose(hx, spike, rtol=_RTOL, atol=shared_floor), (
        "the retired shared floor is expected to ACCEPT a 10% single-element "
        "hx error; if it now reds, this witness needs rewriting"
    )

    # (e) Every signal-carrying component still reds a 0.1% own-peak change;
    #     numerically-null components are floored by the coupled roundoff.
    for comp in comps:
        a = arrays_by_comp[comp][0]
        atol = atols[comp]
        peak = peaks[comp]
        if peak <= atol:
            # Numerically null: no relative statement exists. Assert only that
            # the floor is the coupled-state roundoff, not this component's own
            # (meaningless) magnitude.
            own_floor = _REASSOC_COHERENCE_C * float(n_steps) * float(
                np.finfo(a.dtype).eps
            ) * peak
            assert atol > own_floor, (
                f"{comp} is numerically null (peak {peak:.3e} <= atol "
                f"{atol:.3e}) but its floor was set by its own magnitude "
                f"({own_floor:.3e}); a null component must be bounded by the "
                "curl-coupled roundoff of the driven components"
            )
            continue
        delta = 1e-3 * peak
        perturbed = a.copy()
        idx = np.unravel_index(int(np.argmax(np.abs(a))), a.shape)
        perturbed[idx] += np.asarray(delta, dtype=a.dtype)
        assert not np.allclose(a, perturbed, rtol=_RTOL, atol=atol), (
            f"derived {comp} floor {atol:.3e} swallowed a 0.1%-of-own-peak "
            f"perturbation ({delta:.3e}, peak {peak:.3e}) -- the {comp} check "
            "no longer detects a physically meaningful divergence"
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
