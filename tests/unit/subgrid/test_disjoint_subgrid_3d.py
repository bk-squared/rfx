"""Stage 2 disjoint-domain subgrid prototype tests."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.subgridding.disjoint_3d import (
    ALL_FACES,
    Z0,
    _apply_faces_maxwell_sat_from_snapshot,
    apply_all_faces_maxwell_sat,
    apply_xlo_maxwell_sat,
    compute_disjoint_energy_3d,
    init_disjoint_subgrid_3d,
    prolong_face_linear,
    prolong_face_repeat,
    restrict_face_mean,
    step_disjoint_sat_3d,
    step_disjoint_xlo_sat_3d,
    step_disjoint_topology_3d,
    step_disjoint_z_slab_sat_3d,
    zero_coarse_hole,
)


def test_face_projection_norm_compatible_ratio2():
    """Mean restriction + repeat prolongation are adjoint under face norms."""
    ratio = 2
    dx_f = 0.5
    dx_c = ratio * dx_f
    coarse = jnp.arange(1, 7, dtype=jnp.float32).reshape(2, 3)
    fine = jnp.arange(1, 25, dtype=jnp.float32).reshape(4, 6)

    restricted = restrict_face_mean(fine, ratio)
    prolonged = prolong_face_repeat(coarse, ratio)

    coarse_inner = float(jnp.sum(coarse * restricted) * dx_c ** 2)
    fine_inner = float(jnp.sum(prolonged * fine) * dx_f ** 2)
    np.testing.assert_allclose(coarse_inner, fine_inner, rtol=1e-6)


def test_linear_face_projection_preserves_coarse_samples_at_ratio_nodes():
    coarse = jnp.arange(1, 10, dtype=jnp.float32).reshape(3, 3)

    fine = prolong_face_linear(coarse, ratio=2)

    assert fine.shape == (6, 6)
    np.testing.assert_allclose(fine[::2, ::2], coarse, rtol=0.0, atol=0.0)
    assert float(fine[1, 0]) == 0.5 * float(coarse[0, 0] + coarse[1, 0])
    assert float(fine[0, 1]) == 0.5 * float(coarse[0, 0] + coarse[0, 1])


def test_endpoint_node_disjoint_shape_uses_node_adjoint_projection():
    config, state = init_disjoint_subgrid_3d(
        shape_c=(10, 10, 10),
        fine_region=(0, 10, 0, 10, 3, 7),
        ratio=2,
        sat_strength=0.02,
        face_projection="node_adjoint",
        shape_convention="endpoint_node",
    )

    assert config.shape_f == (19, 19, 7)
    assert config.face_projection == "node_adjoint"
    assert config.shape_convention == "endpoint_node"

    state = state._replace(
        hy_c=state.hy_c.at[:, :, config.fine_region[4] - 1].set(1.0 / Z0),
    )
    stepped = step_disjoint_z_slab_sat_3d(zero_coarse_hole(state, config), config)

    assert float(jnp.max(jnp.abs(stepped.ex_f[:, 1:-1, :]))) > 0.0
    for arr in (
        stepped.ex_c,
        stepped.ey_c,
        stepped.ez_c,
        stepped.hx_c,
        stepped.hy_c,
        stepped.hz_c,
    ):
        hole = arr[
            config.fine_region[0] : config.fine_region[1],
            config.fine_region[2] : config.fine_region[3],
            config.fine_region[4] : config.fine_region[5],
        ]
        assert float(jnp.max(jnp.abs(hole))) == 0.0


def test_disjoint_energy_ignores_coarse_hole_and_counts_fine_volume():
    config, state = init_disjoint_subgrid_3d(
        shape_c=(10, 10, 10), fine_region=(3, 7, 3, 7, 3, 7), ratio=2,
    )
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    state = state._replace(
        ez_c=state.ez_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo:fk_hi].set(99.0)
    )
    assert compute_disjoint_energy_3d(state, config) == 0.0

    state = state._replace(ez_f=state.ez_f.at[2, 2, 2].set(1.0))
    fine_only = compute_disjoint_energy_3d(state, config)
    assert fine_only > 0.0

    state = state._replace(ez_c=state.ez_c.at[1, 1, 1].set(1.0))
    assert compute_disjoint_energy_3d(state, config) > fine_only


def test_disjoint_topology_step_keeps_coarse_hole_zero_and_finite():
    config, state = init_disjoint_subgrid_3d(
        shape_c=(12, 12, 12), fine_region=(4, 8, 4, 8, 4, 8), ratio=2,
    )
    state = state._replace(
        ez_c=state.ez_c.at[2, 6, 6].set(1.0),
        ez_f=state.ez_f.at[4, 4, 4].set(0.5),
    )
    state = zero_coarse_hole(state, config)
    initial_energy = compute_disjoint_energy_3d(state, config)

    for _ in range(20):
        state = step_disjoint_topology_3d(state, config)
        energy = compute_disjoint_energy_3d(state, config)
        assert np.isfinite(energy)
        assert energy >= 0.0

    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    for arr in (state.ex_c, state.ey_c, state.ez_c, state.hx_c, state.hy_c, state.hz_c):
        hole = arr[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo:fk_hi]
        assert float(jnp.max(jnp.abs(hole))) == 0.0

    final_energy = compute_disjoint_energy_3d(state, config)
    # Topology-only independent PEC blocks should remain bounded.  This
    # is not the final SBP-SAT energy proof; the next Stage-2 slice adds
    # Maxwell cross-coupled interface SAT and tightens the energy gate.
    assert final_energy <= initial_energy * 1.05


def test_fine_interface_faces_are_not_clamped_as_pec_boundaries():
    """The fine block boundary is an interface, not a physical PEC wall."""
    config, state = init_disjoint_subgrid_3d(
        shape_c=(12, 12, 12), fine_region=(4, 8, 4, 8, 4, 8), ratio=2,
    )
    state = state._replace(
        # Ey is tangential on the x-lo fine face and would be forced to zero by
        # apply_pec(..., axes="x").  It must survive topology stepping so the
        # subsequent SAT coupling can transport interface fields.
        ey_f=state.ey_f.at[0, 2:6, 2:6].set(0.25),
    )

    stepped = step_disjoint_topology_3d(state, config)

    assert float(jnp.max(jnp.abs(stepped.ey_f[0, :, :]))) > 0.0


def test_xlo_maxwell_sat_cross_couples_h_to_e():
    """Tangential E must receive tangential-H mismatch, not E mismatch."""
    config, state = init_disjoint_subgrid_3d(
        shape_c=(10, 10, 10),
        fine_region=(4, 7, 3, 7, 3, 7),
        ratio=2,
        sat_strength=0.01,
    )
    fi_lo, _fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    c_sl = (fi_lo - 1, slice(fj_lo, fj_hi), slice(fk_lo, fk_hi))

    # Coarse Hz mismatch should create Ey on the fine x-lo face through
    # the Maxwell curl pairing. Same-kind Ey fields start at zero.
    state = state._replace(hz_c=state.hz_c.at[c_sl].set(1.0 / Z0))
    updated = apply_xlo_maxwell_sat(state, config)

    fine_ey = updated.ey_f[0, :, :]
    fine_ez = updated.ez_f[0, :, :]
    assert float(jnp.max(jnp.abs(fine_ey))) > 0.0
    assert float(jnp.max(jnp.abs(fine_ez))) == 0.0


def test_xlo_sat_transfers_signal_to_fine_and_keeps_energy_bounded():
    """Prototype physical smoke: interface signal enters the fine block."""
    config, state = init_disjoint_subgrid_3d(
        shape_c=(14, 12, 12),
        fine_region=(6, 10, 4, 8, 4, 8),
        ratio=2,
        sat_strength=0.01,
    )
    fi_lo, _fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    c_sl = (fi_lo - 1, slice(fj_lo, fj_hi), slice(fk_lo, fk_hi))
    state = state._replace(hz_c=state.hz_c.at[c_sl].set(1.0 / Z0))
    state = zero_coarse_hole(state, config)
    initial_energy = compute_disjoint_energy_3d(state, config)

    max_energy = initial_energy
    for _ in range(30):
        state = step_disjoint_xlo_sat_3d(state, config)
        energy = compute_disjoint_energy_3d(state, config)
        assert np.isfinite(energy)
        max_energy = max(max_energy, energy)

    fine_signal = float(jnp.max(jnp.abs(state.ey_f)))
    assert fine_signal > 0.0
    assert max_energy <= initial_energy * 2.0

    # The coarse-owned hole remains inactive even with interface SAT.
    for arr in (state.ex_c, state.ey_c, state.ez_c, state.hx_c, state.hy_c, state.hz_c):
        hole = arr[fi_lo:_fi_hi, fj_lo:fj_hi, fk_lo:fk_hi]
        assert float(jnp.max(jnp.abs(hole))) == 0.0


def test_all_faces_maxwell_sat_transfers_signal_to_each_fine_boundary():
    """Six-face SAT must transfer coarse tangential-H signals to fine E."""
    config, state = init_disjoint_subgrid_3d(
        shape_c=(14, 14, 14),
        fine_region=(5, 9, 5, 9, 5, 9),
        ratio=2,
        sat_strength=0.02,
    )
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    # x faces: Hz mismatch drives Ey.
    state = state._replace(
        hz_c=state.hz_c.at[fi_lo - 1, fj_lo:fj_hi, fk_lo:fk_hi].set(1.0 / Z0),
    )
    state = state._replace(
        hz_c=state.hz_c.at[fi_hi, fj_lo:fj_hi, fk_lo:fk_hi].set(1.0 / Z0),
    )
    # y faces: Hz mismatch drives Ex.
    state = state._replace(
        hz_c=state.hz_c.at[fi_lo:fi_hi, fj_lo - 1, fk_lo:fk_hi].set(1.0 / Z0),
    )
    state = state._replace(
        hz_c=state.hz_c.at[fi_lo:fi_hi, fj_hi, fk_lo:fk_hi].set(1.0 / Z0),
    )
    # z faces: Hy mismatch drives Ex.
    state = state._replace(
        hy_c=state.hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1].set(1.0 / Z0),
    )
    state = state._replace(
        hy_c=state.hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi].set(1.0 / Z0),
    )

    state = apply_all_faces_maxwell_sat(state, config)

    assert set(ALL_FACES) == {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}
    assert float(jnp.max(jnp.abs(state.ey_f[0, :, :]))) > 0.0
    assert float(jnp.max(jnp.abs(state.ey_f[-1, :, :]))) > 0.0
    assert float(jnp.max(jnp.abs(state.ex_f[:, 0, :]))) > 0.0
    assert float(jnp.max(jnp.abs(state.ex_f[:, -1, :]))) > 0.0
    assert float(jnp.max(jnp.abs(state.ex_f[:, :, 0]))) > 0.0
    assert float(jnp.max(jnp.abs(state.ex_f[:, :, -1]))) > 0.0


def test_all_faces_maxwell_sat_is_order_invariant_from_snapshot():
    """All-face SAT deltas must be accumulated from one frozen snapshot."""
    config, base = init_disjoint_subgrid_3d(
        shape_c=(14, 14, 14),
        fine_region=(5, 9, 5, 9, 5, 9),
        ratio=2,
        sat_strength=0.2,
    )
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    state = base._replace(
        hz_c=base.hz_c.at[fi_lo - 1, fj_lo:fj_hi, fk_lo:fk_hi].set(1.0 / Z0),
        hx_c=base.hx_c.at[fi_lo:fi_hi, fj_lo - 1, fk_lo:fk_hi].set(0.7 / Z0),
        hy_c=base.hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1].set(0.4 / Z0),
        ez_f=base.ez_f.at[:, :, -1].set(0.2),
    )
    state = zero_coarse_hole(state, config)

    forward = _apply_faces_maxwell_sat_from_snapshot(state, config, ALL_FACES)
    reverse = _apply_faces_maxwell_sat_from_snapshot(
        state, config, tuple(reversed(ALL_FACES)),
    )

    for name in (
        "ex_c", "ey_c", "ez_c", "hx_c", "hy_c", "hz_c",
        "ex_f", "ey_f", "ez_f", "hx_f", "hy_f", "hz_f",
    ):
        np.testing.assert_allclose(
            getattr(forward, name),
            getattr(reverse, name),
            rtol=0.0,
            atol=0.0,
        )


def test_six_face_sat_longer_run_bounded_and_ad_finite():
    """Full prototype gate: six-face coupling is bounded and differentiable."""
    config, base = init_disjoint_subgrid_3d(
        shape_c=(14, 14, 14),
        fine_region=(5, 9, 5, 9, 5, 9),
        ratio=2,
        sat_strength=0.02,
    )
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region

    def _seed(alpha):
        state = base._replace(
            hz_c=base.hz_c.at[fi_lo - 1, fj_lo:fj_hi, fk_lo:fk_hi].set(alpha / Z0),
            hx_c=base.hx_c.at[fi_lo:fi_hi, fj_lo - 1, fk_lo:fk_hi].set(alpha / Z0),
            hy_c=base.hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1].set(alpha / Z0),
        )
        return zero_coarse_hole(state, config)

    state = _seed(1.0)
    initial_energy = compute_disjoint_energy_3d(state, config)
    max_energy = initial_energy
    for _ in range(80):
        state = step_disjoint_sat_3d(state, config)
        energy = compute_disjoint_energy_3d(state, config)
        assert np.isfinite(energy)
        max_energy = max(max_energy, energy)

    assert max_energy <= initial_energy * 1.25
    assert float(jnp.max(jnp.abs(state.ex_f))) > 0.0
    for arr in (state.ex_c, state.ey_c, state.ez_c, state.hx_c, state.hy_c, state.hz_c):
        hole = arr[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo:fk_hi]
        assert float(jnp.max(jnp.abs(hole))) == 0.0

    def loss(alpha):
        state_ad = _seed(alpha)
        for _ in range(5):
            state_ad = step_disjoint_sat_3d(state_ad, config)
        return jnp.sum(state_ad.ex_f**2 + state_ad.ey_f**2 + state_ad.ez_f**2)

    grad = float(jax.grad(loss)(jnp.float32(1.0)))
    assert np.isfinite(grad)


def _disjoint_longrun_final_energy(stepper, config, state, n_steps=3000):
    """Roll the disjoint state ``n_steps`` via a compiled scan; return energy.

    ``compute_disjoint_energy_3d`` reduces to a Python ``float`` and so cannot
    run inside ``lax.scan``; the scan only advances the state and energy is
    measured on the concrete final state. (The earlier per-step-energy scan
    body raised ``ConcretizationTypeError`` and the xfail masked it -- the
    long-run lock was never actually measuring the energy it claims to.)
    """
    final_state, _ = jax.lax.scan(
        lambda carry, _: (stepper(carry, config), None),
        state,
        None,
        length=n_steps,
    )
    return float(compute_disjoint_energy_3d(final_state, config))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known energy instability of the disjoint Stage-2 six-face SAT "
        "prototype: the closed-domain energy diverges ~14x over a 3000-step "
        "horizon. research-only — see handover SUBGRID_HANDOVER_2026-05-16.md "
        "section 3. This xfail(strict=True) locks the known instability: if a "
        "future change stabilizes the prototype the test XPASSes and strict "
        "flips it to a failure, so the change cannot land silently."
    ),
)
def test_disjoint_subgrid_3d_longrun_instability():
    """Long-run energy lock: the six-face disjoint prototype is NOT stable.

    ``test_six_face_sat_longer_run_bounded_and_ad_finite`` runs only 80 steps
    — inside the bounded-transient window. Over a 3000-step horizon the
    closed-domain energy diverges. This test asserts long-time boundedness,
    which fails by design; it documents the instability instead of hiding it
    behind a short window.
    """
    config, base = init_disjoint_subgrid_3d(
        shape_c=(14, 14, 14),
        fine_region=(5, 9, 5, 9, 5, 9),
        ratio=2,
        sat_strength=0.02,
    )
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    state = base._replace(
        hz_c=base.hz_c.at[fi_lo - 1, fj_lo:fj_hi, fk_lo:fk_hi].set(1.0 / Z0),
        hx_c=base.hx_c.at[fi_lo:fi_hi, fj_lo - 1, fk_lo:fk_hi].set(1.0 / Z0),
        hy_c=base.hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1].set(1.0 / Z0),
    )
    state = zero_coarse_hole(state, config)
    initial_energy = float(compute_disjoint_energy_3d(state, config))

    final_energy = _disjoint_longrun_final_energy(
        step_disjoint_sat_3d, config, state
    )

    # A genuine SBP-SAT closure keeps closed-domain energy bounded. This
    # prototype does not — the bound assertion fails by design (the xfail
    # lock); the finiteness assertion above it must still hold.
    assert np.isfinite(final_energy)
    assert final_energy <= initial_energy * 1.25


def test_z_slab_sat_handles_full_xy_fine_block_without_side_coarse_cells():
    """Centered z-slab topology has z interfaces and physical x/y faces."""
    config, state = init_disjoint_subgrid_3d(
        shape_c=(10, 10, 10),
        fine_region=(0, 10, 0, 10, 3, 7),
        ratio=2,
        sat_strength=0.02,
    )
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    state = state._replace(
        # z-normal SAT: coarse Hy mismatch drives fine Ex on the z-lo face.
        hy_c=state.hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1].set(1.0 / Z0),
    )
    state = zero_coarse_hole(state, config)
    initial_energy = compute_disjoint_energy_3d(state, config)
    max_energy = initial_energy

    for _ in range(20):
        state = step_disjoint_z_slab_sat_3d(state, config)
        energy = compute_disjoint_energy_3d(state, config)
        assert np.isfinite(energy)
        max_energy = max(max_energy, energy)

    assert float(jnp.max(jnp.abs(state.ex_f[:, 1:-1, :]))) > 0.0
    assert max_energy <= initial_energy * 2.0
    for arr in (state.ex_c, state.ey_c, state.ez_c, state.hx_c, state.hy_c, state.hz_c):
        hole = arr[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo:fk_hi]
        assert float(jnp.max(jnp.abs(hole))) == 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known energy instability of the disjoint z-slab keep-path stepper "
        "(step_disjoint_z_slab_sat_3d, the stepper US-11 kept as the disjoint "
        "research keep-path): the closed-domain energy diverges ~33x over a "
        "3000-step horizon. research-only. This xfail(strict=True) locks the "
        "instability so a future stabilization XPASSes and strict flips it to "
        "a failure."
    ),
)
def test_disjoint_z_slab_sat_3d_longrun_instability():
    """Long-run energy lock for the disjoint z-slab keep-path stepper.

    ``test_z_slab_sat_handles_full_xy_fine_block_without_side_coarse_cells``
    runs only 20 steps — inside the bounded-transient window.
    ``step_disjoint_z_slab_sat_3d`` is the stepper US-11 kept as the disjoint
    research keep-path, so its long-run behaviour needs an explicit lock too
    (the existing US-06 lock exercises only ``step_disjoint_sat_3d``). Over a
    3000-step horizon the closed-domain energy diverges; this test asserts
    long-time boundedness, which fails by design.
    """
    config, state = init_disjoint_subgrid_3d(
        shape_c=(10, 10, 10),
        fine_region=(0, 10, 0, 10, 3, 7),
        ratio=2,
        sat_strength=0.02,
    )
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    state = state._replace(
        hy_c=state.hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1].set(1.0 / Z0),
    )
    state = zero_coarse_hole(state, config)
    initial_energy = float(compute_disjoint_energy_3d(state, config))

    final_energy = _disjoint_longrun_final_energy(
        step_disjoint_z_slab_sat_3d, config, state
    )

    # The keep-path stepper is not long-time energy-stable either — the bound
    # assertion fails by design (the xfail lock).
    assert np.isfinite(final_energy)
    assert final_energy <= initial_energy * 2.0

