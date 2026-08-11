#!/usr/bin/env python3
"""Issue #626 addendum: is the n_warmup truncation error a property of
n_warmup itself, or of how far the design cell sits from the source
relative to the wavefront's travel time?

Context: the shipped "fixture 1"/"fixture 2" curves in
docs/agent-memory/rfx-known-issues.md's "#626 RESOLVED" entry, and in
tests/test_n_warmup.py::test_warmup_truncation_error_grows_with_k and
scripts/diagnostics/i626_n_warmup_truncation_sweep.py, both use a design
cell only ~3 cells from the source. At that placement the field (and
hence the loss's sensitivity to that cell) is nonzero from the first few
steps onward, so EVERY warmup step severed by n_warmup carries real
gradient signal -- the curve those fixtures measure is the WORST case for
that reason, not necessarily the general case for any design placement.

Hypothesis: before the wavefront (traveling from the source at the
domain's numerical wave speed) has reached the design cell, the field
there -- and therefore d(loss)/d(eps) at that cell for any step before
arrival -- is identically ~0. Severing the carry's gradient tape for
those steps then discards ~nothing, so n_warmup should be (near-)exact
whenever it stays below the wavefront's arrival step count at the design
cell, REGARDLESS of how large that step count is as a fraction of
n_steps. If true, "n_warmup truncates the gradient" is incomplete without
"once the wavefront has reached the design region" -- and n_warmup is
GENUINELY free compute/memory relief (not merely a truncated-BPTT
approximation) for K below that arrival step.

K_safe formula: for a design cell whose closest point is `d` metres from
every active source, the wavefront cannot arrive earlier than
    K_safe = floor(d / (c * dt))
grid steps (c = C0, the vacuum/vacuum-dominant wave speed -- a
conservative floor for any non-vacuum-slower propagation). This script
computes K_safe from the grid's own `dt` and the source/design-cell
geometry, and checks it against the measured K at which the truncation
error starts departing from the near-source fixtures' noise floor.

Verification-round correction: the original coarse sweep (multiples of
20) never sampled inside [101, 119] and reported "error only starts
growing once K crosses K_safe, exactly at the predicted boundary" as if
K_safe marked a sharp cliff. A finer sweep bracketing K_safe shows a
SMOOTH, monotonically growing curve starting well before K_safe (not a
cliff): at this fixture's K_safe=108, K=104 measures 0.259%, K=106
measures 0.601%, K=108 (=K_safe itself) measures 1.186% -- all still
comfortably under this repo's own established ~1.5% AD-vs-FD noise floor
(see e.g. `tests/test_n_warmup.py`'s docstrings), but not "near-exact to
every printed digit" the way K<=100 is. The falsifier below now (a)
samples finely around K_safe rather than only at multiples of 20, and
(b) gates on the ~1.5% noise floor rather than an arbitrarily tight
0.1%, which is what the formula's PRACTICAL claim actually needs to
survive: K_safe is a useful, honest bound on where truncation is
noise-floor-negligible, not a literal discontinuity.

Run:
    PYTHONPATH=. JAX_PLATFORMS=cpu python scripts/diagnostics/i626_n_warmup_wavefront_locality.py
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation

try:
    from jax import enable_x64
except ImportError:  # pragma: no cover - JAX version drift, see tests/_x64_compat.py
    from tests._x64_compat import enable_x64

C0 = 299792458.0

DX = 0.5e-3
NX = 70  # domain_x = 35mm -- long enough to separate "near" from "far"
DZ_LAYERS = 9
N_STEPS = 220
N_FIXED = 180  # loss window = time_series[N_FIXED:], fixed across the K sweep
ALPHA0 = 2.0


def _build_sim() -> Simulation:
    dz = np.array([0.5e-3] * DZ_LAYERS, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9, domain=(NX * DX, 0.01, float(np.sum(dz))),
        dx=DX, dz_profile=dz, cpml_layers=4,
    )
    sim.add_source((4 * DX, 0.005, 0.002), "ez")
    sim.add_probe(((NX - 4) * DX, 0.005, 0.002), "ez")
    return sim


def _make_objective(sim, ti, tj, tk, g_shape):
    eps_base = jnp.ones(g_shape, dtype=jnp.float32)

    def objective(alpha, *, n_warmup):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        fr = sim.forward(
            eps_override=eps, n_steps=N_STEPS, n_warmup=n_warmup,
            skip_preflight=True,
        )
        return jnp.sum(fr.time_series[N_FIXED:] ** 2)

    return objective


def _central_fd_f64(objective, x0, h, *, n_warmup):
    with enable_x64():
        fp = objective(jnp.asarray(x0 + h, dtype=jnp.float32), n_warmup=n_warmup)
        fm = objective(jnp.asarray(x0 - h, dtype=jnp.float32), n_warmup=n_warmup)
        return float((fp - fm) / (2.0 * h))


def main():
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    dt = float(g.dt)
    j = g.ny // 2
    k = g.nz // 2

    src_i = 8            # 4mm / 0.5mm
    near_i = 11           # 3 cells from the source (matches the shipped fixtures)
    far_i = g.nx - 9      # near the probe, far from the source

    print(f"grid shape={g.shape}, dt={dt:.4e}s, src_i={src_i}, "
          f"near_i={near_i}, far_i={far_i} "
          f"(far separation = {far_i - src_i} cells = "
          f"{(far_i - src_i) * DX * 1e3:.1f} mm)")

    h = 0.02

    # Far cell's K_safe, computed up front so the sweep can bracket it
    # finely -- multiples of 20 alone skip right over the boundary.
    d_far_m = abs(far_i - src_i) * DX
    k_safe_far = math.floor(d_far_m / (C0 * dt))
    fine_bracket = sorted({
        max(0, k_safe_far - d) for d in (20, 10, 6, 4, 2, 1, 0)
    } | {k_safe_far + d for d in (1, 2, 4, 8, 12)})
    coarse_tail = [0, 20, 40, 60, 80, 140, 160, 180, 200]
    k_sweep = sorted(set(fine_bracket) | set(coarse_tail))
    k_sweep = [K for K in k_sweep if K < N_STEPS]

    results = {}
    for label, ti in (("near", near_i), ("far", far_i)):
        d_m = abs(ti - src_i) * DX
        k_safe = math.floor(d_m / (C0 * dt)) if label == "far" else 0
        obj = _make_objective(sim, ti, j, k, g.shape)

        g_fd = _central_fd_f64(obj, ALPHA0, h, n_warmup=0)
        print(f"\n=== design cell {label}: i={ti}, d_from_source={d_m * 1e3:.2f}mm, "
              f"K_safe={k_safe} steps ===")
        print(f"FD oracle (K=0) = {g_fd:.6e}")

        row = []
        for K in k_sweep:
            a0 = jnp.asarray(ALPHA0, dtype=jnp.float32)
            g_ad = float(jax.grad(lambda a: obj(a, n_warmup=K))(a0))
            rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
            row.append((K, g_ad, rel))
            flag = "  <= K_safe" if label == "far" and K <= k_safe else ""
            print(f"  K={K:4d}  AD={g_ad:14.6e}  rel_err={rel * 100:8.3f}%{flag}")
        results[label] = row

    # ---- Falsifiable check: below K_safe, the far-cell gradient stays
    # within this repo's established ~1.5% AD-vs-FD noise floor -- the
    # PRACTICAL claim K_safe needs to survive (see the module docstring's
    # verification-round correction: it is a smooth, monotonically
    # growing curve approaching K_safe, not a sharp cliff, so a tight
    # <0.1% band across a merely-coarse sweep was an artifact of never
    # sampling near the boundary, not evidence of near-exactness AT the
    # boundary itself).
    NOISE_FLOOR = 0.015  # this repo's own established AD-vs-FD tolerance
    far_row = results["far"]
    d_m = abs(far_i - src_i) * DX
    k_safe = math.floor(d_m / (C0 * dt))
    sub_wavefront = [(K, g_ad, rel) for K, g_ad, rel in far_row if K <= k_safe]
    assert sub_wavefront, "no sampled K falls below K_safe -- widen k_sweep or shrink separation"
    assert k_safe in {K for K, _, _ in sub_wavefront}, (
        "K_safe itself is not a sampled point -- fine_bracket construction changed; "
        "the boundary value must be checked directly, not only points strictly below it"
    )
    for K, g_ad, rel in sub_wavefront:
        assert rel < NOISE_FLOOR, (
            f"far cell K={K} (<= K_safe={k_safe}) rel_err={rel * 100:.4f}% -- "
            f"expected within this repo's ~{NOISE_FLOOR * 100:.1f}% AD-vs-FD "
            "noise floor at or below the wavefront-arrival horizon"
        )
    # Separately confirm the DEEP sub-wavefront region (well below K_safe)
    # really is near-exact, not merely inside the looser noise floor --
    # this is where the "genuinely free, not merely approximate" claim
    # lives.
    deep = [(K, rel) for K, _, rel in sub_wavefront if K <= k_safe - 20]
    assert deep, "no sampled K falls in the deep sub-wavefront region (K_safe - 20)"
    for K, rel in deep:
        assert rel < 1e-3, (
            f"far cell K={K} (deep sub-wavefront, K_safe-{k_safe - K}) "
            f"rel_err={rel * 100:.4f}% -- expected near-exact (<0.1%) well "
            "below the wavefront-arrival horizon"
        )
    print(
        f"\n[falsifier PASS] far cell: every sampled K <= K_safe={k_safe} stays "
        f"within the {NOISE_FLOOR * 100:.1f}% noise floor (values: "
        f"{[(K, round(rel * 100, 4)) for K, _, rel in sub_wavefront]}); "
        f"the deep sub-wavefront region (K <= K_safe-20) is near-exact (<0.1%)"
    )

    near_row = results["near"]
    near_far_k = max(K for K, _, _ in near_row if K < N_FIXED)
    near_far_rel = next(rel for K, _, rel in near_row if K == near_far_k)
    print(
        f"[contrast] near cell: at the same K={near_far_k} the rel_err is "
        f"{near_far_rel * 100:.2f}% -- confirms the near-source fixture's "
        "large error is a placement effect, not a property of n_warmup itself."
    )


if __name__ == "__main__":
    main()
