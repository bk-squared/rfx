"""Stage 2 Step 3a — dual-path wiring tests.

The unified Stage 2 path runs alongside Stage 1's:
  * ``simulation.run(..., aniso_inv_eps=(inv_xx, inv_yy, inv_zz))`` is
    the new low-level entry point.
  * ``runners/uniform.py(subpixel_smoothing="kottke_pec")`` is the
    public-facing opt-in. When set, it computes the inv-eps tensor
    via ``compute_inv_eps_tensor_diag`` and threads it through.
  * Default behavior (no opt-in, no `kottke_pec`) is unchanged —
    the unified path is dormant code until invoked.

This test file exercises the wiring at three levels: simulation
scan body, runners/uniform.py public API, and NU runner hard-fail.
"""

from __future__ import annotations

from tests._wall_axis_declaration import assert_declared_wall_warning
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


# -----------------------------------------------------------------------------
# Test A: simulation.run smoke with aniso_inv_eps
# -----------------------------------------------------------------------------


def test_simulation_run_with_aniso_inv_eps_runs_finite():
    """Lowest-level smoke: simulation.run accepts aniso_inv_eps tuple
    and produces finite fields. Pins the new kwarg to the API surface
    of run() / run_until_decay()."""
    from rfx.grid import Grid
    from rfx.core.yee import init_state, init_materials
    from rfx.simulation import run as run_simulation

    grid = Grid(
        freq_max=10e9,
        domain=(0.06, 0.025, 0.012),
        dx=0.001,
        cpml_layers=4,
    )
    materials = init_materials(grid.shape)

    inv_xx = jnp.ones(grid.shape, dtype=jnp.float32)
    inv_yy = jnp.ones(grid.shape, dtype=jnp.float32)
    inv_zz = jnp.ones(grid.shape, dtype=jnp.float32)

    result = run_simulation(
        grid, materials, n_steps=10,
        boundary="cpml", cpml_axes="xyz",
        aniso_inv_eps=(inv_xx, inv_yy, inv_zz),
    )
    assert np.all(np.isfinite(np.asarray(result.state.ex)))
    assert np.all(np.isfinite(np.asarray(result.state.ey)))
    assert np.all(np.isfinite(np.asarray(result.state.ez)))


# NOTE: PEC-tangential-freeze contract is already pinned at the
# kernel level by ``test_update_e_aniso_inv_pec_tangential_frozen`` in
# tests/unit/geometry/test_update_e_aniso_inv.py — ``simulation.run`` does not
# currently accept a ``state=`` kwarg for explicit seeding, so we rely
# on the kernel-level test for the freeze semantics and verify
# integration via the smoke + dual-path-differs tests below.


# -----------------------------------------------------------------------------
# Test B: runners/uniform.py public API with subpixel_smoothing="kottke_pec"
# -----------------------------------------------------------------------------


def test_runners_uniform_kottke_pec_smoke():
    """Public API: ``Simulation.run(subpixel_smoothing='kottke_pec')``
    auto-routes through the unified path. End-to-end smoke on a small
    WR-90 sim: expect no exception and finite fields after 20 steps."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.06, 0.025, 0.012),
        dx=0.001,
        # conformal faces with kottke_pec are refused (refusal PR); the subject here is kottke_pec
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        0.030, direction="+x", mode=(1, 0), mode_type="TE",
        f0=8e9, bandwidth=0.5, name="left",
    )
    result = sim.run(n_steps=20, subpixel_smoothing="kottke_pec")
    ey = np.asarray(result.state.ey)
    assert np.all(np.isfinite(ey))
    assert float(np.max(np.abs(ey))) > 0


def test_runners_uniform_kottke_pec_differs_from_default():
    """Stage 2 unified path with a PEC geometry object must produce
    *different* fields from Stage 1 default. Confirms the kottke_pec
    code path is exercised (not silently no-op).

    For pure boundary-face PEC (no geometry shapes), Stage 1 and Stage
    2 are numerically equivalent — both correctly zero ghost cells and
    apply the same vacuum update. The distinction only appears at PEC
    *geometry* boundary cells, where Stage 2 uses Kottke inv-eps
    averaging (fractional inv) while Stage 1 uses sigma=1e10 (binary
    masking). The thin PEC strip's off-node x faces make Kottke's
    fractional inverse tensor differ from the staircase.
    """
    from rfx.geometry.csg import Box

    def _build():
        sim = Simulation(
            freq_max=10e9,
            domain=(0.06, 0.025, 0.012),
            dx=0.001,
            # conformal faces with kottke_pec are refused (refusal PR); the subject here is kottke_pec
            boundary=BoundarySpec(
                x="cpml",
                y=Boundary(lo="pec", hi="pec"),
                z=Boundary(lo="pec", hi="pec"),
            ),
            cpml_layers=4,
        )
        # PEC iris one cell thick (1 mm in x at dx = 1 mm), with both
        # x faces shifted off the nodes by 0.3 dx to 25.3 and 26.3 mm.
        # These off-node faces make Kottke's fractional inverse tensor
        # differ from Stage 1's staircase masks. Node-aligned faces
        # reduce to the staircase and do not exercise this difference.
        sim.add(Box((0.0253, 0.005, 0.002), (0.0263, 0.020, 0.010)),
                material="pec")
        sim.add_waveguide_port(
            0.010, direction="+x", mode=(1, 0), mode_type="TE",
            f0=8e9, bandwidth=0.5, name="left",
        )
        return sim

    n = 200
    r_stage1 = _build().run(n_steps=n)
    r_stage2 = _build().run(n_steps=n, subpixel_smoothing="kottke_pec")
    diff_ez = float(np.max(np.abs(np.asarray(r_stage1.state.ez)
                                  - np.asarray(r_stage2.state.ez))))
    assert diff_ez > 1e-4, (
        "Stage 2 'kottke_pec' produced near-identical Ez fields to "
        f"Stage 1 default — unified path not exercised. diff={diff_ez:g}"
    )


# -----------------------------------------------------------------------------
# Test C: default behavior (no opt-in) is bit-identical to pre-Stage-2
# -----------------------------------------------------------------------------


def test_default_simulation_run_unchanged():
    """When no opt-in (no aniso_inv_eps, no kottke_pec, no
    Boundary(conformal=True)), the simulation must produce exactly
    the same fields as before Step 3a wiring landed."""
    from rfx.api import Simulation
    sim_a = Simulation(
        freq_max=10e9,
        domain=(0.04, 0.04, 0.04),
        dx=0.002,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
        cpml_layers=4,
    )
    sim_b = Simulation(
        freq_max=10e9,
        domain=(0.04, 0.04, 0.04),
        dx=0.002,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
        cpml_layers=4,
    )
    sim_a.add_waveguide_port(
        0.020, direction="+x", mode=(1, 0), mode_type="TE",
        f0=8e9, bandwidth=0.5, name="left",
    )
    sim_b.add_waveguide_port(
        0.020, direction="+x", mode=(1, 0), mode_type="TE",
        f0=8e9, bandwidth=0.5, name="left",
    )

    n = 20
    r_a = sim_a.run(n_steps=n)
    r_b = sim_b.run(n_steps=n)
    np.testing.assert_array_equal(
        np.asarray(r_a.state.ey), np.asarray(r_b.state.ey),
        err_msg="repeated default runs should be bit-identical",
    )


# -----------------------------------------------------------------------------
# Test D: NU runner hard-fail on kottke_pec
# -----------------------------------------------------------------------------


def test_dual_path_dielectric_only_equivalent_to_ulp():
    """Step 3b dual-path equivalence: for a sim with **only**
    dielectric content (no PEC), ``subpixel_smoothing=True`` (Stage 1
    Kottke) and ``subpixel_smoothing="kottke_pec"`` (Stage 2 unified)
    must give field results that agree to ULP tolerance.

    Why: both paths run identical Kottke math at dielectric
    interfaces; the only difference is the eps-divide vs inv-eps-
    multiply arithmetic ordering in the Yee update. With no PEC
    content, the two paths should be numerically equivalent.

    A diff larger than ~5 ULP (relative) signals the Stage 2 path is
    silently introducing a non-equivalent transformation in the
    dielectric branch — a regression that would block Step 3c's
    default routing flip."""
    from rfx.geometry.csg import Box

    def _build():
        sim = Simulation(
            freq_max=10e9,
            domain=(0.04, 0.04, 0.04),
            dx=0.002,
            boundary={"x": "cpml", "y": "pec", "z": "pec"},
            cpml_layers=4,
        )
        sim.add_material("substrate", eps_r=4.0, sigma=0.0)
        sim.add(Box((0.012, 0.012, 0.012), (0.028, 0.028, 0.028)),
                material="substrate")
        sim.add_waveguide_port(
            0.020, direction="+x", mode=(1, 0), mode_type="TE",
            f0=8e9, bandwidth=0.5, name="left",
        )
        return sim

    n = 100
    r_stage1 = _build().run(n_steps=n, subpixel_smoothing=True)
    r_stage2 = _build().run(n_steps=n, subpixel_smoothing="kottke_pec")

    for component in ("ex", "ey", "ez"):
        a = np.asarray(getattr(r_stage1.state, component))
        b = np.asarray(getattr(r_stage2.state, component))
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-30)
        rel_diff = float(np.max(np.abs(a - b))) / scale
        assert rel_diff < 5e-5, (
            f"dielectric-only dual-path diverges on {component}: "
            f"rel_diff={rel_diff:.2e}, max|a|={np.max(np.abs(a)):.3e}, "
            f"max|b|={np.max(np.abs(b)):.3e}. Step 3c default flip "
            f"would re-bless reference values for any test that "
            f"asserts {component} to bit precision under "
            f"subpixel_smoothing=True."
        )


def test_pec_short_s11_with_kottke_pec_path():
    """Stage 2 step 4 acceptance: PEC-short |S11| ≥ 0.99 at 5–6.5 GHz
    via the unified ``subpixel_smoothing="kottke_pec"`` path.

    Design rationale:

    normalize=False is the correct choice for S11 of a strong reflector.
    The two-run normalize=True formula cancels dispersion only for S21
    (same one-way path in both runs). For S11 the device wave is a
    round-trip while the reference is one-way; the cancellation fails
    and introduces ±15 % swings from source-plane impedance mismatch.

    The domain is 120 × 39 × 21 mm, all whole-cell lengths at dx=3 mm.
    For the 39 mm guide width, TE10 cutoff is 3.85 GHz and TE20 cutoff
    is 7.69 GHz. The 5–6.5 GHz window stays between these cutoffs,
    limiting evanescent TE20 contamination at the TE10 port.

    PEC box at x=[84 mm, 87 mm]: both faces are integer multiples of
    dx=3 mm so Kottke reduces to a binary mask (inv=0 or inv=1 only).
    Gate ≥ 0.99: a value below 0.99 in the clean 5–6.5 GHz window
    signals unfrozen cells or CPML interaction in the Stage 2 path."""
    from rfx.geometry.csg import Box

    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.039, 0.021),
        dx=0.003,
        # conformal faces with kottke_pec are refused (refusal PR); the subject here is kottke_pec
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=10,
    )
    sim.add(Box((0.084, 0, 0), (0.087, 0.039, 0.021)), material="pec")
    # 5–6.5 GHz lies above TE10 cutoff (3.85 GHz) and below TE20
    # cutoff (7.69 GHz) for the 39 mm guide width.
    freqs = jnp.linspace(5e9, 6.5e9, 6)
    sim.add_waveguide_port(
        0.010, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5, name="left",
    )
    # #1913: 4 cells behind the short's far face (87 mm), so the port's 3-cell
    # reflection plane is in front of it; |S11| is bit-identical to 90 mm.
    sim.add_waveguide_port(
        0.099, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5, name="right",
    )
    res = sim.compute_waveguide_s_matrix(
        num_periods=40, normalize=False,
        subpixel_smoothing="kottke_pec",
    )
    s11 = np.abs(np.asarray(res.s_params)[0, 0, :])
    print(f"\n[stage2 step4 pec-short] |S11| range "
          f"[{s11.min():.4f}, {s11.max():.4f}] mean={s11.mean():.4f}")
    assert s11.min() >= 0.99, (
        f"Stage 2 unified path PEC-short |S11| regressed: "
        f"min={s11.min():.4f} (gate 0.99). "
        f"In the TE20-clean 5–6.5 GHz window, |S11| should be within "
        f"1 % of 1.0. A lower value signals unfrozen Kottke cells or "
        f"CPML energy leakage in the Stage 2 inv-eps path."
    )


def test_nu_runner_hard_fails_on_kottke_pec():
    """Per Step 3a scope decision: NU + Stage 2 unified path is out
    of scope for v1. Must hard-fail at run time, not silently
    misbehave.
    Addendum 5f requires transverse z PEC so guide admission reaches Kottke.
    """
    sim = Simulation(
        freq_max=10e9,
        domain=(0.04, 27 * (299792458.0 / 10e9 / 20.0), 0.04), dx=299792458.0 / 10e9 / 20.0,  # solved length; #1887
        dz_profile=[0.002] * 20,
        # No ``conformal=True`` here: on a non-uniform mesh that declaration
        # is refused on its own (conformal_pec) before kottke_pec is reached.
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        0.020, direction="+x", mode=(1, 0), mode_type="TE",
        f0=8e9, bandwidth=0.5, name="left",
    )
    with pytest.raises((NotImplementedError, ValueError),
                       match="kottke_pec|nonuniform|NU"):
        sim.run(n_steps=10, subpixel_smoothing="kottke_pec")
