"""Issue #40: n_warmup splits the scan into a gradient-free warmup phase
and a full-AD optimize phase.

Guarantees:
  1. Forward output (time_series) is unchanged vs. a plain n_warmup=0
     run — the warmup phase still runs the same physics.
  2. jax.grad of a loss that depends only on post-warmup samples
     returns the same value with or without warmup (stop_gradient only
     kills tape for steps before n_warmup; physical forward is
     identical).
  3. n_warmup >= n_steps is a clean ValueError.

Issue #626 additions:
  4. The uniform (single-device) forward lane never implemented n_warmup
     (no warmup-split parameter on `_forward_from_materials`) but silently
     accepted and dropped it — measured bit-identical gradient at
     n_warmup=0 vs n_warmup=60. It now raises NotImplementedError instead
     (test_uniform_forward_rejects_n_warmup).
  5. n_warmup IS implemented on this file's non-uniform lane. Severing the
     scan carry at the warmup boundary severs the gradient path from a
     design variable's influence during the warmup window -- but ONLY for
     steps after the wavefront has reached the design region; before that
     the field there is ~0, so severing those steps' carry costs nothing
     and the truncation is (near-)exact. This file's fixture places the
     design cell only ~3 cells from the source (the worst case: the
     wavefront is present from step 0), so
     test_warmup_truncation_error_grows_with_k pins that worst-case curve
     specifically, NOT a universal property of n_warmup -- see
     `scripts/diagnostics/i626_n_warmup_wavefront_locality.py` for a
     far-from-source counter-fixture (K_safe formula + near-exact
     gradient below it) and `rfx/nonuniform.py`'s `n_warmup split`
     comment for both curves side by side (issue #626 part 2 /
     addendum). test_warmup_truncation_error_grows_with_k itself pins the
     measured error curve against an independent central-FD oracle
     (held-fixed loss window, K varied independently -- see the docstring
     there for why that isolation matters).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_forward_matches_plain():
    sim = _build_sim()
    ts_plain = np.asarray(sim.forward(n_steps=100).time_series)
    ts_warm = np.asarray(
        sim.forward(n_steps=100, n_warmup=40).time_series
    )
    assert ts_plain.shape == ts_warm.shape
    np.testing.assert_allclose(ts_plain, ts_warm, rtol=1e-5, atol=1e-10)


def test_warmup_grad_finite_and_same_sign():
    """n_warmup cuts gradient contribution from the warmup window (by
    design — that's the memory/speed trade). We only require the
    warmup'd gradient to be finite and agree on sign with the full
    gradient.
    """
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2
    n_steps = 60

    def loss(alpha, *, warmup):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        fr = sim.forward(
            eps_override=eps, n_steps=n_steps, n_warmup=warmup,
        )
        return jnp.sum(fr.time_series[warmup:] ** 2)

    a0 = jnp.float32(2.0)
    g_plain = float(jax.grad(lambda a: loss(a, warmup=0))(a0))
    g_warm = float(jax.grad(lambda a: loss(a, warmup=20))(a0))
    assert np.isfinite(g_plain) and np.isfinite(g_warm)
    assert np.sign(g_plain) == np.sign(g_warm), (
        f"grad signs disagree: plain={g_plain}, warmup={g_warm}"
    )


def test_n_warmup_ge_n_steps_raises():
    sim = _build_sim()
    with pytest.raises(ValueError, match="n_warmup"):
        sim.forward(n_steps=30, n_warmup=30)


def test_n_warmup_composes_with_checkpoint_every():
    sim = _build_sim()
    ts_plain = np.asarray(sim.forward(n_steps=80).time_series)
    ts_combo = np.asarray(
        sim.forward(n_steps=80, n_warmup=16, checkpoint_every=16).time_series
    )
    np.testing.assert_allclose(ts_plain, ts_combo, rtol=1e-5, atol=1e-10)


def test_uniform_forward_rejects_n_warmup():
    """Issue #626: forward() declared and documented n_warmup but only
    forwarded it to the NU lanes; `_forward_from_materials` (the uniform
    lane) has no such parameter, so a nonzero n_warmup was silently
    accepted and ignored (measured bit-identical gradient at n_warmup=0
    vs n_warmup=60). It must now fail loud instead, matching its two
    remaining uniform-lane siblings (emit_time_series=False,
    checkpoint_every) -- design_mask, the third historical sibling, was
    removed from every public surface entirely (issue #625) rather than
    fenced, so it is no longer part of this taxonomy. This is the
    regression witness: a future silent re-drop would make this raise
    disappear (n_warmup=0 still runs fine below), not merely "both run"
    -- the required difference is raise vs. no-raise."""
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                      cpml_layers=4)
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")

    # n_warmup=0 (the default / a no-op value) must still run cleanly.
    fr = sim.forward(n_steps=20, n_warmup=0)
    assert np.all(np.isfinite(np.asarray(fr.time_series)))

    with pytest.raises(NotImplementedError, match="n_warmup"):
        sim.forward(n_steps=20, n_warmup=5)


def test_warmup_truncation_error_grows_with_k():
    """Issue #626 part 2: measure what n_warmup does NUMERICALLY to the
    gradient on the non-uniform lane where it IS implemented, against an
    INDEPENDENT central-FD oracle -- not merely "finite and same sign"
    (that was test_warmup_grad_finite_and_same_sign's weaker guarantee).

    NOTE (issue #626 addendum): this fixture's design cell sits only ~3
    cells from the source, which is the WORST case for n_warmup (the
    wavefront is already present at every step, so every severed warmup
    step carries real gradient signal). This test intentionally keeps
    pinning that worst-case curve as a regression lock -- it is not a
    claim that n_warmup always costs this much accuracy. See
    `scripts/diagnostics/i626_n_warmup_wavefront_locality.py` for the
    complementary far-from-source case (near-exact below K_safe) and
    `rfx/nonuniform.py`'s `n_warmup split` comment for the K_safe formula.

    Design: the LOSS WINDOW is held FIXED at time_series[N_FIXED:] for
    every K in the sweep, and K (n_warmup) is varied independently. This
    is deliberately different from test_warmup_grad_finite_and_same_sign,
    which lets the loss window itself track `warmup` -- that conflates
    "how much of the tail is scored" with "how much AD tape is kept" and
    cannot isolate the truncation effect. The FD oracle uses n_warmup=0
    (forward output is provably n_warmup-invariant per
    test_forward_matches_plain, so the oracle is unaffected by K) with
    comparison arithmetic in float64 (repo convention, tests/_x64_compat.py
    + test_jacobian_fwd.py's _central_fd_f64).

    Measured (this fixture, N_STEPS=100, N_FIXED=80, design cell offset
    from source/probe): rel_err vs. FD oracle is roughly
    K=0: ~0.2-1%, K=10: ~1-1.5%, K=30: 3.3%, K=40 (half the
    pre-loss-window): 6.6%, K=50: 12.1%, K=70: 35.2% -- growing as K
    approaches the loss window, consistent with a second independent
    design-cell placement measured out-of-band (see
    docs/agent-memory/rfx-known-issues.md, issue #626 part 2, and the
    forward()/nonuniform.py docstrings this test backs). The K=0/K=10
    figures are hedged deliberately: an h-sweep of the FD oracle shows
    those two points sit INSIDE the oracle's own noise band (at h=0.05,
    K=10's error reads slightly below K=0's), so a strict monotonicity
    assertion starting at K=0 would bind that noise-band coincidence
    rather than the truncation mechanism -- this repo's own do-not-repeat
    lesson (feedback_gate_can_bind_artifact). K>=30 is unambiguous and is
    where the assertions below get strict. This is NOT a free memory
    lever: severing the carry at the warmup boundary severs the gradient
    path from the design variable's influence during the warmup window,
    and that path is not negligible for a static material parameter
    present throughout the whole run.
    """
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2
    n_steps = 100
    n_fixed = 80
    alpha0 = 2.0
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)

    def objective(alpha, *, n_warmup):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        fr = sim.forward(
            eps_override=eps, n_steps=n_steps, n_warmup=n_warmup,
            skip_preflight=True,
        )
        return jnp.sum(fr.time_series[n_fixed:] ** 2)

    h = 0.02
    with enable_x64():
        fp = objective(jnp.asarray(alpha0 + h, dtype=jnp.float32), n_warmup=0)
        fm = objective(jnp.asarray(alpha0 - h, dtype=jnp.float32), n_warmup=0)
        g_fd = float((fp - fm) / (2.0 * h))
    assert abs(g_fd) > 0.0, "FD oracle gradient is exactly zero -- fixture did not couple"

    k_sweep = [0, 10, 30, 40, 50, 70]
    rel_errs = []
    for k in k_sweep:
        a0 = jnp.asarray(alpha0, dtype=jnp.float32)
        g_ad = float(jax.grad(lambda a: objective(a, n_warmup=k))(a0))
        rel_errs.append(abs(g_ad - g_fd) / max(abs(g_fd), 1e-30))

    # K=0/K=10 (no or minimal truncation) sit inside the FD oracle's own
    # noise band at this h (measured via an h-sweep -- see the docstring),
    # so these are loose ceilings, not a tight/ordered pair.
    assert rel_errs[0] < 0.05, f"K=0 AD vs FD mismatch: {rel_errs[0] * 100:.2f}%"
    assert rel_errs[1] < 0.05, f"K={k_sweep[1]} rel_err unexpectedly large: {rel_errs[1] * 100:.2f}%"
    # Large K (approaching the loss window) is a substantial, non-noise
    # bias -- if this ever drops back near the noise floor, the truncation
    # mechanism itself has silently changed and needs re-investigation,
    # not a gate loosening.
    assert rel_errs[-1] > 0.15, f"K={k_sweep[-1]} rel_err unexpectedly small: {rel_errs[-1] * 100:.2f}%"
    # Strict monotonicity from K=30 upward ONLY: below K=30 the true
    # signal is not resolved above the FD oracle's own noise floor at
    # this h (measured), so asserting order there would bind a noise-band
    # coincidence rather than the truncation mechanism -- exactly the
    # failure mode feedback_gate_can_bind_artifact warns about. From
    # K=30 up the bias is unambiguous and the growth is decisive.
    k30_idx = k_sweep.index(30)
    trustworthy = rel_errs[k30_idx:]
    for a, b in zip(trustworthy, trustworthy[1:]):
        assert b >= a - 1e-6, f"non-monotone truncation error from K=30 up: {trustworthy}"
