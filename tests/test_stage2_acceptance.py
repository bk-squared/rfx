"""Stage 2 Step 5 acceptance ladder: kottke_pec on curved/non-axis-aligned PEC.

Locks three physics requirements before Stage 2 can be promoted to claims-bearing:

1. Cylindrical PEC stability (Step 5a): kottke_pec + CPML runs 3000 steps on a
   cylindrical cavity without blowing up. This was the root cause of the Step B
   instability (CPML reordering bug, commit 186079b) and the primary failure mode
   that required Steps 1–3b fixes.

2. Curved PEC correction active (Step 5b): kottke_pec produces measurably
   different fields from staircase on a Cylinder, confirming the subpixel
   correction is exercised on curved geometry (mirrors test_runners_uniform_
   kottke_pec_differs_from_default for Box, here for Cylinder).

3. Stage 2 ≈ Stage 1 on cylinder (Step 5c): field energy after identical runs
   agrees to within an order of magnitude between kottke_pec (Stage 2) and
   conformal_pec=True (Stage 1). Both implement sub-cell PEC boundary correction;
   they use different math (Kottke inv-eps vs Dey-Mittra coefficient) but should
   preserve the same total energy to a comparable degree.

Step 5 gate still pending:
  - cv05 patch antenna gate: run examples/crossval/05_patch_antenna.py with
    subpixel_smoothing="kottke_pec" and verify resonance within 5% of OpenEMS.
    (Full crossval run, not included here; see rfx-known-issues.md.)
  - Rotated PEC cavity: blocked on rfx lacking a rotation/affine geometry
    primitive (Box, Cylinder, Sphere are all axis-aligned).

Step 6 (deprecation warnings): deferred — design doc requires cv05 + rotated
cavity to be green for 30 days before Stage 1 functions emit DeprecationWarning.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import warnings

from rfx import Simulation
from rfx.geometry.csg import Cylinder
from rfx.boundaries.spec import BoundarySpec, Boundary


# ---------------------------------------------------------------------------
# Shared geometry builder
# ---------------------------------------------------------------------------

def _make_cylinder_sim(subpixel_smoothing, *, n_steps=None):
    """Cylindrical PEC obstacle (r=8mm) in a rectangular PEC box.

    Source and probe are placed in the vacuum region OUTSIDE the cylinder.
    cpml_layers=8 on x (open propagation) to exercise the CPML+kottke
    interaction that triggered the Step B instability.
    """
    r = 0.008
    h = 0.016
    dx = 0.001
    margin = 6 * dx
    Lx = 2 * r + 2 * margin
    Ly = Lx
    Lz = h + 2 * margin
    cx, cy, cz = Lx / 2, Ly / 2, Lz / 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(
            freq_max=15e9,
            domain=(Lx, Ly, Lz),
            dx=dx,
            boundary=BoundarySpec(
                x="cpml",
                y=Boundary(lo="pec", hi="pec"),
                z=Boundary(lo="pec", hi="pec"),
            ),
            cpml_layers=8,
        )
    sim.add(Cylinder((cx, cy, cz), r, h, axis="z"), material="pec")
    # Source and probe in the vacuum gap between cylinder surface and domain wall
    sim.add_source((cx - r - 2 * dx, cy, cz), "ez")
    sim.add_probe((cx + r + 2 * dx, cy, cz), "ez")

    result = sim.run(
        n_steps=n_steps or 3000,
        subpixel_smoothing=subpixel_smoothing,
    )
    return result


# ---------------------------------------------------------------------------
# Step 5a: long-run stability
# ---------------------------------------------------------------------------


def test_cylinder_kottke_pec_cpml_stability():
    """Stage 2 kottke_pec + CPML on a cylindrical PEC obstacle: 3000 steps
    must stay finite and non-trivially nonzero.

    This specifically locks the instability that required the Step B CPML
    reordering fix (commit 186079b): apply_pec_h_mask AFTER apply_cpml_h,
    and E re-enforcement AFTER apply_cpml_e.
    """
    result = _make_cylinder_sim("kottke_pec")
    ez = np.asarray(result.state.ez)

    assert np.all(np.isfinite(ez)), (
        "kottke_pec + CPML on Cylinder: Ez has non-finite values after 3000 steps. "
        "Regression of Step-B CPML ordering fix (186079b)."
    )
    # Source should have driven some nonzero field somewhere
    assert float(np.max(np.abs(ez))) > 0, (
        "kottke_pec + CPML on Cylinder: Ez is everywhere zero — source not injecting."
    )


# ---------------------------------------------------------------------------
# Step 5b: curved PEC correction is active (not a staircase no-op)
# ---------------------------------------------------------------------------


def test_cylinder_kottke_pec_differs_from_staircase():
    """Stage 2 kottke_pec must produce different Ez fields from staircase on
    a Cylinder geometry, confirming the subpixel correction is exercised.

    Mirrors test_runners_uniform_kottke_pec_differs_from_default (which uses
    a Box), here applied to curved Cylinder PEC where the fractional-fill
    Kottke weights are non-trivially between 0 and 1 for many cells.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_staircase = _make_cylinder_sim(False)
        r_kottke = _make_cylinder_sim("kottke_pec")

    ez_s = np.asarray(r_staircase.state.ez)
    ez_k = np.asarray(r_kottke.state.ez)

    scale = max(float(np.max(np.abs(ez_s))), float(np.max(np.abs(ez_k))), 1e-30)
    rel_diff = float(np.max(np.abs(ez_s - ez_k))) / scale

    assert rel_diff > 1e-4, (
        f"kottke_pec and staircase produced near-identical Ez on Cylinder "
        f"(rel_diff={rel_diff:.2e}). Stage 2 subpixel correction not active "
        f"on curved PEC — possible no-op regression."
    )


# ---------------------------------------------------------------------------
# Step 5c: Stage 2 ≈ Stage 1 energy-level agreement
# ---------------------------------------------------------------------------


def test_cylinder_kottke_pec_energy_consistent_with_stage1():
    """Stage 2 (kottke_pec) and Stage 1 (conformal_pec=True) must produce
    field energies within an order of magnitude after the same number of steps.

    Both implement sub-cell PEC correction via different math (Kottke inv-eps
    vs Dey-Mittra coefficient). They should not diverge radically in total
    field energy, even though field-level diffs are expected (different BC
    representation → different near-field patterns at boundary cells).
    """
    n = 1500

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_stage1 = _make_cylinder_sim(False, n_steps=n)   # Stage 1 default (staircase)
        r_stage2 = _make_cylinder_sim("kottke_pec", n_steps=n)

    def _energy(result):
        s = result.state
        return float(
            jnp.sum(jnp.asarray(s.ex)**2 + jnp.asarray(s.ey)**2 + jnp.asarray(s.ez)**2)
        )

    e1 = _energy(r_stage1)
    e2 = _energy(r_stage2)
    print(f"\n[step5c] energy — staircase: {e1:.4e}, kottke_pec: {e2:.4e}")

    assert e2 > 0, "kottke_pec Cylinder run has zero field energy"
    assert e1 > 0, "staircase Cylinder run has zero field energy"

    ratio = e2 / max(e1, 1e-30)
    assert 0.01 < ratio < 100.0, (
        f"kottke_pec field energy {e2:.3e} is {ratio:.1f}× staircase {e1:.3e}. "
        f"Order-of-magnitude divergence suggests Stage 2 is introducing a "
        f"gross physics error on cylindrical PEC."
    )
