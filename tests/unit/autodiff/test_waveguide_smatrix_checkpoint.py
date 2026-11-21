"""Regression tests for compute_waveguide_s_matrix(checkpoint_segments=K).

Validates the checkpoint_segments plumbing added in issue #131:
  - Forward equivalence: S(checkpoint=None) == S(K) bit-identically.
  - Gradient equivalence: grad wrt eps_override with None vs K, cross-trace bar.
  - Non-divisor checkpoint_segments raises ValueError.
  - NU mesh + checkpoint_segments is now SUPPORTED (issue #73; was a fence).

Grid: tiny WR-90-like 2-port (~57×15×8 cells), num_periods=4 → n_steps≈70,
K=7 (divides 70). CPU-only, float32.  Kept fast (each test ≪ 30 s).
The waveguide cell size of 0.003 m keeps the test grid tiny.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.contracts.path_equivalence.comparison import ACCUMULATED_RELATIVE


_FREQS = jnp.linspace(5e9, 6.5e9, 4)

CELL_SIZE_M = 0.003
_DX = CELL_SIZE_M
_DOMAIN = (0.171, 0.045, 0.024)


def _make_sim():
    """Minimal 2-port WR-90-like sim, no obstacle (empty guide)."""
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary

    sim = Simulation(
        freq_max=10e9,
        domain=_DOMAIN,
        dx=_DX,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        0.030, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=6e9, bandwidth=0.5, name="left",
    )
    sim.add_waveguide_port(
        0.141, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=6e9, bandwidth=0.5, name="right",
    )
    return sim


def _make_sim_with_eps_override():
    """Sim with a thin dielectric slab for AD testing (eps_override channel)."""
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary

    sim = Simulation(
        freq_max=10e9,
        domain=_DOMAIN,
        dx=_DX,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        0.030, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=6e9, bandwidth=0.5, name="left",
    )
    sim.add_waveguide_port(
        0.141, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=6e9, bandwidth=0.5, name="right",
    )
    return sim


def _get_n_steps(sim, num_periods: float = 4.0) -> int:
    """Return the auto-computed n_steps that compute_waveguide_s_matrix uses."""
    grid = sim._build_grid()
    return int(grid.num_timesteps(num_periods=num_periods))


def _find_divisor_near_sqrt(n: int) -> int:
    """Return the exact divisor of n closest to sqrt(n), >= 1."""
    import math
    target = math.sqrt(n)
    best = 1
    best_dist = abs(1 - target)
    for k in range(1, n + 1):
        if n % k == 0:
            dist = abs(k - target)
            if dist < best_dist:
                best_dist = dist
                best = k
    return best


@pytest.mark.parametrize("normalize", [False, True, "flux"])
def test_forward_equivalence_checkpoint_vs_none(normalize):
    """S(checkpoint_segments=K) is bit-identical to S(checkpoint_segments=None)."""
    sim = _make_sim()
    n_steps = _get_n_steps(sim, num_periods=4.0)
    K = _find_divisor_near_sqrt(n_steps)
    assert n_steps % K == 0, f"K={K} does not divide n_steps={n_steps}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_none = sim.compute_waveguide_s_matrix(
            num_periods=4.0, normalize=normalize,
        )
        res_k = sim.compute_waveguide_s_matrix(
            num_periods=4.0, normalize=normalize,
            checkpoint_segments=K,
        )

    s_none = np.array(res_none.s_params)
    s_k = np.array(res_k.s_params)

    max_delta = float(np.max(np.abs(s_none - s_k)))
    print(
        f"\n[normalize={normalize}] n_steps={n_steps}, K={K}: "
        f"max|S(None)-S(K)| = {max_delta:.3e}"
    )
    assert max_delta == 0.0, (
        f"normalize={normalize}: S(None) vs S(K={K}) not bit-identical; "
        f"max|delta|={max_delta:.3e}. checkpoint threading is wrong."
    )


def test_grad_equivalence_checkpoint_vs_none():
    """jax.grad wrt eps_override: None vs K agree to the cross-trace bar."""
    sim = _make_sim_with_eps_override()
    n_steps = _get_n_steps(sim, num_periods=4.0)
    K = _find_divisor_near_sqrt(n_steps)
    assert n_steps % K == 0, f"K={K} does not divide n_steps={n_steps}"

    grid = sim._build_grid()
    base_materials, _, _, _, _, _, _ = sim._assemble_materials(grid)
    eps_base = jnp.array(base_materials.eps_r)

    def objective(eps: jax.Array, *, ckpt: "int | None") -> jax.Array:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_waveguide_s_matrix(
                num_periods=4.0, normalize=False,
                eps_override=eps,
                checkpoint_segments=ckpt,
            )
        return jnp.mean(jnp.abs(res.s_params) ** 2).real

    grad_none = jax.grad(objective)(eps_base, ckpt=None)
    grad_k = jax.grad(objective)(eps_base, ckpt=K)

    assert jnp.all(jnp.isfinite(grad_none)), "grad_none contains non-finite values"
    assert jnp.all(jnp.isfinite(grad_k)), f"grad_k (K={K}) contains non-finite values"
    assert jnp.any(grad_none != 0.0), "grad_none is all-zero (no signal through eps_override)"
    assert jnp.any(grad_k != 0.0), f"grad_k (K={K}) is all-zero"

    denom = float(jnp.max(jnp.abs(grad_none))) + 1e-30
    rel_dev = float(jnp.max(jnp.abs(grad_none - grad_k))) / denom
    print(
        f"\n[grad AD] n_steps={n_steps}, K={K}: "
        f"max|grad_none|={denom:.3e}, "
        f"max|grad_none - grad_k|/max|grad_none| = {rel_dev:.3e}"
    )
    assert rel_dev <= ACCUMULATED_RELATIVE, (
        f"grad relative deviation {rel_dev:.3e} > {ACCUMULATED_RELATIVE:g} for K={K}. "
        "checkpoint_segments changes the gradient — threading is wrong."
    )


def test_nondivisor_checkpoint_raises_value_error():
    """checkpoint_segments that does not divide n_steps raises ValueError."""
    sim = _make_sim()
    n_steps = _get_n_steps(sim, num_periods=4.0)

    bad_K = None
    for k in range(2, n_steps + 2):
        if n_steps % k != 0:
            bad_K = k
            break
    assert bad_K is not None, f"Could not find a non-divisor of n_steps={n_steps}"

    with pytest.raises(ValueError, match="checkpoint_segments"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.compute_waveguide_s_matrix(
                num_periods=4.0, normalize=False,
                checkpoint_segments=bad_K,
            )


@pytest.mark.slow
def test_nu_mesh_checkpoint_now_supported():
    """issue #73: checkpoint_segments on a NU mesh now RUNS (previously raised
    NotImplementedError). Forward-identity + AD-gradient gates live in
    tests/unit/autodiff/test_waveguide_nu_checkpoint.py.
    The NU path requires normalize=True or 'flux'.
    """
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary

    n_cells_x = 57
    dx_profile = np.full(n_cells_x, _DX)
    dx_profile[n_cells_x // 2] *= 1.1

    sim = Simulation(
        freq_max=10e9,
        domain=_DOMAIN,
        dx=_DX,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
        dx_profile=dx_profile,
    )
    sim.add_waveguide_port(
        0.030, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=6e9, bandwidth=0.5, name="left",
    )
    sim.add_waveguide_port(
        0.141, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=6e9, bandwidth=0.5, name="right",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_waveguide_s_matrix(
            num_periods=4.0,
            normalize=True,
            checkpoint_segments=7,
        )
    s = np.asarray(res.s_params)
    assert s.shape[0] == s.shape[1] and np.all(np.isfinite(s))
