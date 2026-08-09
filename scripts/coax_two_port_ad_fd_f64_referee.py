#!/usr/bin/env python3
"""Float64 finite-difference referee for the coax two-port AD gate (#489 leg 3).

WHY THIS EXISTS
---------------
``tests/test_coax_two_port_ad.py::
test_compute_coaxial_two_port_ad_grad_finite_and_fd_consistent`` originally
compared reverse-mode AD (float32) against a central finite difference whose
two loss evaluations were ALSO computed in float32. An adversarial review of
the PR that added the gate found this was the exact #527 comparator-class
mistake MSL hit: a float32 FD reference has quantized resolving power, and
the review measured this fixture's f32 comparator at ``span x threshold =
92.06`` -- just under this repo's own ``>= 100`` floor
(``tests/test_msl_ad_fd_converged.py::
test_comparator_floor_rejects_the_f32_reference_that_caused_527``).

The PR's own docstring additionally claimed a genuine float64 comparator was
"not achievable" for this method because ``compute_coaxial_two_port`` hard-
requires ``precision='float32'``. That claim was WRONG, and the review
demonstrated it directly: ``precision='float32'`` governs only the FDTD
field/CPML carry arrays (which DO stay float32, exactly like MSL's own f64
referee -- see ``scripts/msl_ad_fd_f64_referee.py``, "the shipped gate's AD
is f32, so f32 is the default"). The LOSS dtype instead follows the DFT
plane accumulator, which ``rfx/probes/probes.py``'s ``init_dft_plane_probe``
keys off ``jax.config.x64_enabled`` INDEPENDENTLY of ``self._precision``::

    _cdtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64

So a scoped ``enable_x64()`` around the forward call is sufficient --
``eps_scale``/``freqs`` do not even need to be passed as float64 themselves
(and forcing them to float64 is actively counterproductive: it makes
``materials.eps_r`` float64 while the field carry stays float32, producing a
harmless-but-noisy ``FutureWarning`` at ``rfx/boundaries/cpml.py``'s CPML
scatter -- that warning is what the PR's original attempt misread as "the
whole computation reverted to float32". It does not; ``loss.dtype`` is
verified ``== float64`` below with ``eps_scale`` left at its natural
float32).

WHAT THIS SCRIPT DOES
----------------------
Mirrors ``scripts/msl_ad_fd_f64_referee.py`` at reduced scale (this is a
first gate being added, not an existing gate being replaced under a
documented incident -- so no VESSL/GPU run is required; CPU is sufficient
and this script runs there). Evaluates the AD gradient in float32 (as
shipped) and a central finite difference in float64 (scoped
``enable_x64()``, never module-level -- see
``feedback_jax_x64_module_level_tests``: a module-level flip is process-
global and reds every same-process pytest-split shard) over a 7-point h
sweep (``DEFAULT_H``, ``2e-4`` to ``2e-2``, including ``h=5e-3``), reports
the resolving power (ULPs of the loss dtype) at every h, identifies the
tightest-agreeing adjacent pair (the converged window a gate should
actually be derived from), and prints a verdict. Report-only. Modifies no
gate --
the actual gate threshold is a constant hand-derived from a specific run
of this script and committed in the test file, same discipline as
``scripts/msl_ad_fd_f64_referee.py``.

This script runs against whatever checkout is on ``sys.path`` (import-time
``REPO`` resolution below).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

DEFAULT_H = (2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2)


def _fd_ulp_span(f_plus: float, f_minus: float, dtype) -> float:
    """Resolving power of a central difference, in ULPs of ``dtype``.

    Mirrors ``tests/test_msl_ad_fd_converged.py::_fd_ulp_span`` exactly
    (``dtype`` must be the dtype the LOSS was computed in, not the Python
    float container both values arrived in as -- ``float(jnp_scalar)``
    always widens to a Python float, so keying off the value alone would
    silently measure float64 resolution even for a float32 loss, which is
    how the #527 comparator passed the check it should have failed).
    """
    ulp = float(np.spacing(np.asarray(abs(0.5 * (f_plus + f_minus)), dtype=dtype)))
    return abs(f_plus - f_minus) / ulp


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, nargs="*", default=list(DEFAULT_H))
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    try:
        from jax import enable_x64
    except ImportError:  # older JAX (< ~0.4.31)
        from tests._x64_compat import enable_x64

    import rfx
    print(f"rfx     : {rfx.__file__}")
    print(f"jax     : {jax.__version__}   devices: {jax.devices()}")
    if not str(Path(rfx.__file__).resolve()).startswith(str(REPO)):
        print(f"FATAL: imported rfx is not this checkout ({REPO})")
        return 2

    sys.path.insert(0, str(REPO / "tests"))
    from test_coax_two_port_ad import (  # noqa: E402
        FREQ, N_STEPS, PROBE_COUNT, PROBE_SPACING_CELLS, PROBE_START_CELLS,
        _band_mean_s21_sq, _build_small_two_port_sim,
    )

    print(f"fixture : N_STEPS={N_STEPS} freqs={np.asarray(FREQ)} "
          f"probe_count={PROBE_COUNT}")

    # --- forward sanity: settling witness on the CONCRETE (eps_scale=None)
    # twin -- catches a fixture that regressed below the -40 dB ring-down
    # rule before any AD/FD time is spent (R5: inspect the trace, not just
    # the final metric).
    sim0 = _build_small_two_port_sim()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res0 = sim0.compute_coaxial_two_port(
            n_steps=N_STEPS, freqs=FREQ, probe_count=PROBE_COUNT,
            probe_start_cells=PROBE_START_CELLS,
            probe_spacing_cells=PROBE_SPACING_CELLS,
        )
    print(f"forward sanity (eps_scale=None): status={res0.status} "
          f"settling_db={np.asarray(res0.settling_db)}")
    if np.any(np.asarray(res0.settling_db) > -40.0):
        print("WARNING: concrete twin is under-settled (> -40 dB) -- "
              "raise N_STEPS before trusting the numbers below.")

    # ---- AD gradient: float32, as shipped ------------------------------
    print("\n--- AD (float32, as shipped) ---", flush=True)
    t0 = time.perf_counter()
    loss32, g32 = jax.value_and_grad(_band_mean_s21_sq)(jnp.asarray(1.0, dtype=jnp.float32))
    g_ad, loss32 = float(g32), float(loss32)
    ulp32 = float(np.spacing(np.float32(loss32)))
    print(f"  loss = {loss32:.8f}   g_ad = {g_ad:.6e}   ({time.perf_counter()-t0:.1f}s)")
    print(f"  float32 ULP of this loss = {ulp32:.4e}")

    # ---- central finite differences, float64 loss -----------------------
    # eps_scale is deliberately left float32 (its NATURAL dtype) -- only the
    # global x64 flag is scoped on. See this module's docstring for why
    # that is sufficient (the DFT accumulator dtype follows
    # jax.config.x64_enabled, not the FDTD field precision).
    print("\n--- central FD, float64 loss (eps_scale stays float32) ---", flush=True)
    rows = []
    with enable_x64():
        print(f"  {'h':>9} {'g_fd':>18} {'rel_err':>10} {'loss.dtype':>12} "
              f"{'ULP span':>12} {'s':>6}")
        for h in sorted(args.h):
            t0 = time.perf_counter()
            fp = _band_mean_s21_sq(jnp.asarray(1.0 + h, dtype=jnp.float32))
            fm = _band_mean_s21_sq(jnp.asarray(1.0 - h, dtype=jnp.float32))
            assert fp.dtype == jnp.float64 and fm.dtype == jnp.float64, (
                f"FD reference did NOT run in float64 (got {fp.dtype}). JAX "
                "truncates a float64 request to float32 when x64 is off -- "
                "check jax.experimental.enable_x64 and the "
                "jax.config.x64_enabled keying in rfx/probes/probes.py."
            )
            fp, fm = float(fp), float(fm)
            g_fd = (fp - fm) / (2.0 * h)
            span = _fd_ulp_span(fp, fm, np.float64)
            rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-300)
            rows.append((h, g_fd, rel, span))
            print(f"  {h:9.1e} {g_fd:18.10e} {rel:10.5f} {'float64':>12} "
                  f"{span:12.3g} {time.perf_counter()-t0:6.1f}", flush=True)

    hs = np.array([r[0] for r in rows])
    gs = np.array([r[1] for r in rows])
    rels = np.array([r[2] for r in rows])
    spans = np.array([r[3] for r in rows])
    spread = float(np.ptp(gs) / max(abs(np.mean(gs)), 1e-300))
    worst_rel = float(rels.max())
    min_span = float(spans.min())
    print(f"\n  g_ad (f32)              : {g_ad:.10e}")
    print(f"  f64 FD spread over full sweep : {spread*100:.3f}%")
    print(f"  worst rel_err over full sweep : {worst_rel:.5f}")
    print(f"  min resolving power           : {min_span:.3g} ULP")

    # The CONVERGED WINDOW (R5 -- inspect the trace, not one summary number):
    # a well-behaved central difference should have two adjacent h values
    # agreeing closely (negligible truncation AND negligible noise at both).
    # Report the tightest-agreeing adjacent pair and a windowed spread/worst
    # rel_err over [that pair's h, 2x that pair's larger h] as the basis a
    # gate should actually be derived from -- NOT the full sweep, which by
    # design also probes h values expected to be floor-noise- or truncation-
    # dominated (see this script's module docstring / the test file's own
    # "GATE DERIVATION" section for the reasoning).
    if len(hs) >= 2:
        adj_diff = np.abs(np.diff(gs)) / np.maximum(np.abs(gs[:-1]), 1e-300)
        i = int(np.argmin(adj_diff))
        h_lo, h_hi = hs[i], hs[i + 1]
        in_window = (hs >= h_lo) & (hs <= h_hi)
        print(f"\n  tightest-agreeing adjacent pair: h={h_lo:.1e} vs h={h_hi:.1e} "
              f"(agree to {adj_diff[i]*100:.4f}%)")
        print(f"  window [{h_lo:.1e}, {h_hi:.1e}] worst rel_err: "
              f"{float(rels[in_window].max()):.5f}")
        print("  -- a LOWER BOUND on the converged window, not the gate's own "
              "window: the committed gate (tests/test_coax_two_port_ad.py) "
              "derives its threshold from the WIDER {1e-3, 2e-3, 5e-3} window "
              "(worst rel_err 0.01053, not this pair's 0.00508) -- a "
              "deliberately more conservative choice than the bare minimum "
              "this auto-detected pair would justify, since h=1e-3 is still "
              "reasonably well-behaved (not part of the tightest pair, but "
              "not floor-noise-dominated the way h<=5e-4 is either). h values "
              "outside the gate's own window are diagnostic, not gate inputs.")

    stable = spread <= 0.05
    if stable:
        print("\nVERDICT R1: f64 FD is h-stable over the FULL sweep -> AD is "
              "correct on this fixture within the measured rel_err; a gate "
              "derived from gate_from_envelope(worst_rel_err, quantum=100) "
              "over the full sweep is sound.")
    else:
        print("\nVERDICT: full-sweep spread exceeds 5% -- expected if the "
              "sweep intentionally probes both a floor-noise-dominated "
              "small-h region and a truncation-dominated large-h region "
              "(see the converged-window analysis above); this is NOT "
              "automatically R3 (comparator failure) the way a flat/scatter "
              "pattern within a single well-chosen h decade would be. "
              "Inspect the full table by hand before trusting or rejecting "
              "a gate derived from any subset of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
