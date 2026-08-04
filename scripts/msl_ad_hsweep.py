#!/usr/bin/env python3
"""Sweep the finite-difference step h for the MSL AD-vs-FD gate (issue #527).

``test_msl_ad_fd_converged_tight`` compares a reverse-mode AD gradient against a
central finite difference at a SINGLE ``h = 1e-3``. A central difference carries
two competing errors — truncation growing as ``h^2`` and evaluation noise
growing as ``noise/h`` — so a single h is only a valid comparator if it sits in
the plateau between them. That has never been checked on this fixture at the
gate's own settings.

A reduced-fixture sweep (num_periods=4, CPU) showed the FD reference spreading
698% across ``h in [3e-4, 1e-2]``, flipping SIGN at h = 1e-3 (the gate's value)
and converging to the AD gradient within 4.2% at h = 1e-2. That fixture cannot
speak for the gate's converged settings; this script runs the identical sweep on
the gate's OWN fixture (``_build_msl_sim``, ``num_periods=20``, ``n_freqs=8``).

Reads, pre-declared. "Stable" below means the FD reference agrees with ITSELF
across h (spread < 15%) — the reference's own self-consistency, which has to be
established before its disagreement with AD means anything:
  * FD stable AND min rel_err <= 0.10
        -> the AD is fine; #527 is an h-selection defect in the comparator.
  * FD stable AND rel_err >= ~0.5 at every h
        -> the disagreement is real; the AD path is the suspect.
  * FD UNSTABLE but some h lands inside 10%
        -> that agreement is a coincidence of scatter, not convergence. Read it
           as the unstable case, not as a pass.
  * FD unstable at every h
        -> this fixture cannot support a finite-difference reference at all;
           the gate needs a different comparator (complex-step, f64 referee).

Prints ``cond_a`` too: the multi-drive solve's VJP goes through ``A^-T``, so an
ill-conditioned drive matrix is the one mechanism that would attenuate the
gradient specifically. On the reduced fixture it measured 1.09-1.16 (well
conditioned), which refuted that hypothesis there.

Does NOT modify any gate. Report-only.

OBJECTIVE UPDATE (issue #530, 2026-08-04): the gate this script sweeps now
differentiates band-mean |S21|^2 (tests._msl_ad_objective.
msl_band_mean_s21_sq), not sum_ij|S_ij|^2 -- this script's objective was
updated to match so it stays a faithful h-sweep of the CURRENT gate. The 10%
literal in this script's own verdict logic below is this investigation's
own historical threshold (pre-dates tests._gate_policy.gate_from_envelope)
and is independent of the gate's live _REL_ERR_THRESHOLD (now 0.03) -- read
this script's printed numbers, not its verdict labels, against the live gate.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

H_LIST = [3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]


def main() -> None:
    import importlib

    import jax
    import jax.numpy as jnp

    import rfx

    print(f"rfx: {rfx.__file__}")
    print(f"jax devices: {jax.devices()}")

    from tests._msl_ad_objective import msl_band_mean_s21_sq

    t = importlib.import_module("test_msl_ad_fd_converged")
    sim = t._build_msl_sim()
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    n_steps = int(grid.num_timesteps(num_periods=t._NUM_PERIODS))
    seg = t._closest_divisor(n_steps, int(np.sqrt(n_steps)))
    print(f"gate fixture: n_steps={n_steps} checkpoint_segments={seg} "
          f"num_periods={t._NUM_PERIODS} n_freqs={t._N_FREQS} "
          f"(gate h = {t._FD_H})")

    def objective(alpha):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sim.compute_msl_s_matrix(
                n_freqs=t._N_FREQS, num_periods=t._NUM_PERIODS,
                eps_override=eps_base * alpha, checkpoint_segments=seg,
            )
        return msl_band_mean_s21_sq(r.S)

    a0 = jnp.float32(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd = sim.compute_msl_s_matrix(
            n_freqs=t._N_FREQS, num_periods=t._NUM_PERIODS,
            eps_override=eps_base * a0, checkpoint_segments=seg,
        )
    S = np.abs(np.asarray(fwd.S))
    print(f"forward |S| range: [{S.min():.4f}, {S.max():.4f}]")
    print(f"assembly = {getattr(fwd, 'assembly', '<pre-#523 build>')}")
    ca = getattr(fwd, "cond_a", None)
    if ca is not None:
        ca = np.asarray(ca)
        print(f"cond_a: min {ca.min():.4g} max {ca.max():.4g}"
              f"   {'ILL-CONDITIONED' if ca.max() > 1e3 else 'well conditioned'}")

    t0 = time.time()
    g_ad = float(jax.grad(objective)(a0))
    print(f"\ng_ad = {g_ad:+.6e}   ({time.time() - t0:.1f}s)")

    print(f"\n{'h':>10} {'g_fd':>16} {'rel_err vs AD':>15} {'wall(s)':>9}")
    fds = []
    for h in H_LIST:
        s = time.time()
        fp = float(objective(jnp.float32(1.0 + h)))
        fm = float(objective(jnp.float32(1.0 - h)))
        g = (fp - fm) / (2.0 * h)
        fds.append(g)
        rel = abs(g_ad - g) / max(abs(g), abs(g_ad), 1e-30)
        print(f"{h:10.1e} {g:+16.6e} {rel:15.4f} {time.time() - s:9.1f}")

    fds = np.array(fds)
    rels = np.abs(g_ad - fds) / np.maximum(
        np.maximum(np.abs(fds), abs(g_ad)), 1e-30)
    spread = (fds.max() - fds.min()) / max(abs(np.median(fds)), 1e-30)
    print(f"\nFD spread across h : {spread * 100:.1f}%")
    print(f"median g_fd        : {np.median(fds):+.6e}")
    print(f"best rel_err       : {rels.min():.4f} at h = {H_LIST[int(rels.argmin())]:.1e}")
    print(f"gate's h rel_err   : {rels[H_LIST.index(1e-3)]:.4f}")
    print(f"g_ad inside FD band: {bool(fds.min() <= g_ad <= fds.max())}")
    # BOTH halves of the pre-declared read, not just the rel_err half: a lone
    # h landing inside 10% while the FD scatters wildly is a coincidence, not
    # a convergence. Require the reference to be stable too (PR #526
    # re-review, finding 5).
    if rels.min() <= 0.10 and spread < 0.15:
        print("\nVERDICT: the FD is stable across h AND some h agrees with AD "
              "inside the gate's own 10% bound -> the AD is sound and the "
              "gate's SINGLE h = 1e-3 is the defect.")
    elif rels.min() <= 0.10:
        print(f"\nVERDICT: an h agrees within 10% but the FD scatters "
              f"{spread*100:.0f}% across h -> that agreement is a coincidence, "
              "not convergence. Treat as FD-unstable.")
    elif spread < 0.15:
        print("\nVERDICT: FD is stable across h and still disagrees "
              "-> the disagreement is real; the AD path is the suspect.")
    else:
        print("\nVERDICT: FD unstable at every h -> this fixture cannot support a "
              "finite-difference reference; the gate needs a different comparator.")


if __name__ == "__main__":
    main()
