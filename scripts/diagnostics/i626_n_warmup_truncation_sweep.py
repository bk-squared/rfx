#!/usr/bin/env python3
"""Issue #626 part 2: what does n_warmup do NUMERICALLY to the gradient on
the non-uniform lane where it IS implemented (rfx/nonuniform.py)?

This is the committed artifact behind the "fixture 2" curve quoted in
docs/agent-memory/rfx-known-issues.md's "Added 2026-08-11 -- #626 RESOLVED"
entry and the CHANGELOG.md entry of the same name. Fixture 1 (the smaller,
faster sweep) is committed directly as a pytest regression test instead --
see tests/unit/autodiff/test_n_warmup.py::test_warmup_truncation_error_grows_with_k --
because it is cheap enough to run on every CI pass; this script's fixture 2
(larger grid, more steps) is a slower confirmatory sweep kept here as a
reproducible artifact rather than as a standing test.

Design (see tests/unit/autodiff/test_n_warmup.py's docstring for the full rationale): the
LOSS WINDOW is held FIXED at time_series[N_FIXED:] for every K in the
sweep, and K (n_warmup) is varied independently -- conflating the two (as
the pre-existing test_warmup_grad_finite_and_same_sign does) cannot isolate
the truncation effect from "how much of the tail is scored". The oracle is
an independent central FD at n_warmup=0 (forward output is provably
n_warmup-invariant, bit-identical, so the oracle is unaffected by K),
comparison arithmetic in float64 (repo convention, tests/_x64_compat.py +
test_jacobian_fwd.py's _central_fd_f64).

Run:
    PYTHONPATH=. JAX_PLATFORMS=cpu python scripts/diagnostics/i626_n_warmup_truncation_sweep.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation

try:
    from jax import enable_x64
except ImportError:  # pragma: no cover - JAX version drift, see tests/_x64_compat.py
    from tests._x64_compat import enable_x64


def _build_sim():
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3, dz_profile=dz, cpml_layers=4,
    )
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    return sim


# Fixture 2: larger grid / longer run than the committed pytest fixture, a
# different design-cell placement than fixture 1's, kept as a slower
# confirmatory sweep (not a standing CI test).
N_STEPS = 200
N_FIXED = 160  # loss window = time_series[N_FIXED:], fixed across the K sweep
ALPHA0 = 2.0


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
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2
    print(f"grid shape={g.shape}, design cell=({ti},{tj},{tk})")

    objective = _make_objective(sim, ti, tj, tk, g.shape)

    # Forward-value identity sanity check (must hold per
    # tests/unit/autodiff/test_n_warmup.py::test_forward_matches_plain).
    eps_probe = jnp.ones(g.shape, dtype=jnp.float32).at[ti, tj, tk].set(ALPHA0)
    ts0 = np.asarray(sim.forward(eps_override=eps_probe, n_steps=N_STEPS, n_warmup=0,
                                  skip_preflight=True).time_series)
    ts60 = np.asarray(sim.forward(eps_override=eps_probe, n_steps=N_STEPS, n_warmup=60,
                                   skip_preflight=True).time_series)
    fwd_rel = np.max(np.abs(ts0 - ts60)) / max(np.max(np.abs(ts0)), 1e-30)
    print(f"[sanity] forward value n_warmup=0 vs 60: max abs diff rel to peak = {fwd_rel:.3e}")
    assert fwd_rel == 0.0, "forward output must be exactly n_warmup-invariant"

    h = 0.02
    g_fd = _central_fd_f64(objective, ALPHA0, h, n_warmup=0)
    print(f"[oracle] central FD (h={h}, x64 arithmetic) gradient = {g_fd:.6e}")

    k_sweep = [0, 10, 20, 40, 60, 80, 100, 120, 140, 155, 170, 190]
    print(f"{'K':>5} {'K/N_FIXED':>10} {'AD grad':>16} {'rel_err vs FD':>16}")
    results = []
    for k in k_sweep:
        alpha0 = jnp.asarray(ALPHA0, dtype=jnp.float32)
        g_ad = float(jax.grad(lambda a: objective(a, n_warmup=k))(alpha0))
        rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
        results.append((k, g_ad, rel))
        print(f"{k:5d} {k / N_FIXED:10.3f} {g_ad:16.6e} {rel * 100:15.3f}%")

    print("\nRaw tuples (K, AD_grad, rel_err_vs_FD):")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
