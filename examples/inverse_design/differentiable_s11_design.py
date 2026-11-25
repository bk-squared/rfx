"""Differentiable S11 design via end-to-end AD through the rfx S-parameter API.

This example demonstrates rfx's unique capability: ``jax.grad`` flows
end-to-end through the *public* ``sim.compute_waveguide_s_matrix`` API.
A scalar dielectric design variable (permittivity of a small plug region
inside the waveguide) is differentiated against the |S11|^2 objective in
one ``jax.grad`` call — no finite differences are needed to obtain the
gradient.  A central finite-difference cross-check is then computed and
compared, confirming the AD gradient is accurate to within 5%.

Run:  python examples/inverse_design/differentiable_s11_design.py

``_WR90_DX`` must divide both ``_WR90_A`` and ``_WR90_B`` to preserve the declared WR-90 aperture.
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation


_WR90_A = 22.86e-3
_WR90_B = 10.16e-3
_WR90_DX = 1.27e-3
_WR90_LX = 0.10

WR90_FREQS = jnp.linspace(8e9, 11.5e9, 8)
_WR90_F0 = 9.75e9


def build_simulation() -> Simulation:
    """Two-port WR-90 rectangular waveguide sim (CPU-fast, above-cutoff).

    Returns the Simulation without solving it. Both ports sample the design
    band ``WR90_FREQS`` (8.0-11.5 GHz, 8 points, all above the TE10 cutoff
    and below 0.90 x the TE20 cutoff); ``sim.preflight()`` inspects it.
    """
    sim = Simulation(
        freq_max=12e9,
        domain=(_WR90_LX, _WR90_A, _WR90_B),
        dx=_WR90_DX,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        direction="+x",
        x_position=0.024,
        y_range=(0.0, _WR90_A),
        z_range=(0.0, _WR90_B),
        n_modes=1,
        freqs=WR90_FREQS,
        f0=_WR90_F0,
    )
    sim.add_waveguide_port(
        direction="-x",
        x_position=_WR90_LX - 0.024,
        y_range=(0.0, _WR90_A),
        z_range=(0.0, _WR90_B),
        n_modes=1,
        freqs=WR90_FREQS,
        f0=_WR90_F0,
    )
    return sim



def _make_objective(sim: Simulation, eps_base: jnp.ndarray, design: jnp.ndarray):
    """Return a scalar objective(alpha) = sum_f |S11(f)|^2.

    ``alpha`` scales the permittivity of the mid-guide slab ``design`` via
    eps_override, acting as a single differentiable design degree of freedom.
    The guide at both ports stays empty: a waveguide port models a vacuum
    guide, so a filled port segment is refused (rfx#1994). The target
    frequency index is the midpoint of the frequency array.
    """
    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.compute_waveguide_s_matrix(
                n_steps=200,
                normalize=False,
                eps_override=eps_base + design * (alpha - 1.0),
            )
        S = result.s_params
        k0 = S.shape[-1] // 2
        return jnp.real(jnp.sum(jnp.abs(S[:, :, k0]) ** 2))

    return objective


if __name__ == "__main__":
    print("=" * 70)
    print("Differentiable S11 Design — end-to-end AD through rfx S-param API")
    print("=" * 70)

    sim = build_simulation()
    issues = sim.preflight()
    if len(issues):
        raise SystemExit(f"preflight reported {len(issues)} issue(s) — fix the setup first")
    grid = sim.freeze_mesh()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    lo, hi = (grid.index_of("x", x) for x in (0.040, 0.060))
    design = jnp.zeros(grid.shape, dtype=jnp.float32).at[lo:hi].set(1.0)
    alpha0 = jnp.float32(1.0)

    objective = _make_objective(sim, eps_base, design)

    print("\n[1/3] Forward pass …")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd_result = sim.compute_waveguide_s_matrix(
            n_steps=200,
            normalize=False,
            eps_override=eps_base + design * (alpha0 - 1.0),
        )
    S_fwd = np.asarray(fwd_result.s_params)
    s_max = float(np.max(np.abs(S_fwd)))
    obj_val = float(objective(alpha0))
    print(f"  Objective |S11|^2 sum = {obj_val:.6e}")
    print(f"  |S|_max = {s_max:.4f}  (must be in [0, 1.2])")
    assert s_max <= 1.2, f"|S| = {s_max:.4f} physically implausible"
    assert s_max > 0.0, "|S| = 0 everywhere — broken forward pass"

    print("\n[2/3] AD gradient via jax.grad …")
    loss_val, g = jax.value_and_grad(objective)(alpha0)
    g_ad = float(g)
    print(f"  objective = {float(loss_val):.6e}")
    print(f"  AD gradient = {g_ad:.6e}")
    assert jnp.isfinite(g), f"AD gradient is not finite: {g}"
    assert abs(g_ad) > 1e-10, f"AD gradient is effectively zero: {g_ad:.3e}"

    print("\n[3/3] Central finite-difference cross-check …")
    h = 1e-3
    f_plus = float(objective(jnp.float32(alpha0 + h)))
    f_minus = float(objective(jnp.float32(alpha0 - h)))
    g_fd = (f_plus - f_minus) / (2.0 * h)
    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)
    print(f"  FD gradient = {g_fd:.6e}  (h={h})")
    print(f"  Relative error (AD vs FD) = {rel_err:.3e}")

    wall = time.time() - t0
    print(f"\n  Wall time: {wall:.1f} s")

    assert rel_err < 0.05, (
        f"AD vs FD mismatch: AD={g_ad:.4e} FD={g_fd:.4e} "
        f"rel_err={rel_err:.3e} (threshold 5%)"
    )
    assert g_ad * g_fd > 0, (
        f"Sign disagreement: AD={g_ad:.4e} FD={g_fd:.4e}"
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  Objective value : {obj_val:.6e}")
    print(f"  AD gradient     : {g_ad:.6e}")
    print(f"  FD gradient     : {g_fd:.6e}")
    print(f"  Relative error  : {rel_err:.3e}  (< 0.05 required)")
    print(f"  Sign agreement  : {'YES' if g_ad * g_fd > 0 else 'NO'}")
    print(f"  Wall time       : {wall:.1f} s")
    print("\n  PASS — jax.grad flows end-to-end through compute_waveguide_s_matrix")
