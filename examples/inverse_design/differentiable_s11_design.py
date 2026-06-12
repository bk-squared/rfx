"""Differentiable S11 design via end-to-end AD through the rfx S-parameter API.

This example demonstrates rfx's unique capability: ``jax.grad`` flows
end-to-end through the *public* ``sim.compute_waveguide_s_matrix`` API.
A scalar dielectric design variable (permittivity of a small plug region
inside the waveguide) is differentiated against the |S11|^2 objective in
one ``jax.grad`` call — no finite differences are needed to obtain the
gradient.  A central finite-difference cross-check is then computed and
compared, confirming the AD gradient is accurate to within 5%.

Run:  python examples/inverse_design/differentiable_s11_design.py
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation

# ---------------------------------------------------------------------------
# WR-90 geometry — identical to test_waveguide_s_matrix_ad_end_to_end
# (tests/test_sparam_ad_end_to_end.py) so that CPU feasibility is proven.
# ---------------------------------------------------------------------------

_WR90_A = 22.86e-3    # broad wall (m)
_WR90_B = 10.16e-3    # narrow wall (m)
_WR90_DX = 2e-3       # cell size (m)
# Domain length: ports must clear the CPML absorber (issue #149 — the
# original LX=0.05 with cpml_layers=8 x dx=2mm put BOTH port planes inside
# the 16mm absorber; sim.preflight() flags exactly this).
_WR90_LX = 0.10       # domain length (m)

# TE10 cutoff for WR-90: c/(2a) ≈ 6.56 GHz; all freqs here are above cutoff
# AND below 0.90 x fc_TE20 = 11.8 GHz (preflight's next-mode contamination
# bound — the original 8-12 GHz band tripped it at the top edge).
# f0 is set explicitly at band center — a source centered at/below cutoff
# launches an evanescent crawl whose extracted S grows with n_steps
# (issue #150; preflight code "port_source_below_cutoff" now guards this).
_WR90_FREQS = jnp.linspace(8e9, 11.5e9, 8)
_WR90_F0 = 9.75e9


def _build_sim() -> Simulation:
    """Two-port WR-90 rectangular waveguide sim (CPU-fast, above-cutoff)."""
    sim = Simulation(
        freq_max=12e9,
        domain=(_WR90_LX, _WR90_A, _WR90_B),
        dx=_WR90_DX,
        boundary="cpml",
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        direction="+x",
        x_position=0.024,
        y_range=(0.0, _WR90_A),
        z_range=(0.0, _WR90_B),
        n_modes=1,
        freqs=_WR90_FREQS,
        f0=_WR90_F0,
    )
    sim.add_waveguide_port(
        direction="-x",
        x_position=_WR90_LX - 0.024,
        y_range=(0.0, _WR90_A),
        z_range=(0.0, _WR90_B),
        n_modes=1,
        freqs=_WR90_FREQS,
        f0=_WR90_F0,
    )
    return sim


# ---------------------------------------------------------------------------
# Objective: |S11|^2 summed over all frequencies, w.r.t. a scalar design eps
# ---------------------------------------------------------------------------

def _make_objective(sim: Simulation, eps_base: jnp.ndarray):
    """Return a scalar objective(alpha) = sum_f |S11(f)|^2.

    ``alpha`` scales the background permittivity array via eps_override,
    acting as a single differentiable design degree of freedom.  The target
    frequency index is the midpoint of the frequency array.
    """
    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.compute_waveguide_s_matrix(
                n_steps=200,
                normalize=False,
                eps_override=eps_base * alpha,
            )
        S = result.s_params
        k0 = S.shape[-1] // 2
        return jnp.real(jnp.sum(jnp.abs(S[:, :, k0]) ** 2))

    return objective


if __name__ == "__main__":
    print("=" * 70)
    print("Differentiable S11 Design — end-to-end AD through rfx S-param API")
    print("=" * 70)

    sim = _build_sim()
    # Preflight runs VISIBLY (never optimize against a setup you have not
    # preflighted — issues #149/#150 both hid behind suppressed warnings).
    issues = sim.preflight()
    if issues:
        raise SystemExit(f"preflight reported {len(issues)} issue(s) — fix the setup first")
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    alpha0 = jnp.float32(1.0)

    objective = _make_objective(sim, eps_base)

    # --- Forward pass (sanity check) ----------------------------------------
    print("\n[1/3] Forward pass …")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd_result = sim.compute_waveguide_s_matrix(
            n_steps=200,
            normalize=False,
            eps_override=eps_base * alpha0,
        )
    S_fwd = np.asarray(fwd_result.s_params)
    s_max = float(np.max(np.abs(S_fwd)))
    obj_val = float(objective(alpha0))
    print(f"  Objective |S11|^2 sum = {obj_val:.6e}")
    print(f"  |S|_max = {s_max:.4f}  (must be in [0, 1.2])")
    assert s_max <= 1.2, f"|S| = {s_max:.4f} physically implausible"
    assert s_max > 0.0, "|S| = 0 everywhere — broken forward pass"

    # --- AD gradient ----------------------------------------------------------
    print("\n[2/3] AD gradient via jax.grad …")
    loss_val, g = jax.value_and_grad(objective)(alpha0)
    g_ad = float(g)
    print(f"  objective = {float(loss_val):.6e}")
    print(f"  AD gradient = {g_ad:.6e}")
    assert jnp.isfinite(g), f"AD gradient is not finite: {g}"
    assert abs(g_ad) > 1e-10, f"AD gradient is effectively zero: {g_ad:.3e}"

    # --- FD cross-check -------------------------------------------------------
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

    # --- Assertions -----------------------------------------------------------
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
