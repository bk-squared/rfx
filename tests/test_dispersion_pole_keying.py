"""Locks for dispersion-pole mask keying (issue #274).

The per-pole mask dicts in ``Simulation._assemble_materials``
(``rfx/api/_compile.py``) and ``rasterize_geometry``
(``rfx/geometry/rasterize.py``) key by pole VALUE when the pole is
hashable, falling back to ``id(pole)`` ONLY for unhashable poles
(JAX-traced fields). These tests lock both directions of that contract:

1. Value-dedupe (double-count lock): equal concrete poles registered
   through separate ``add_material`` calls on overlapping boxes merge
   into ONE (pole, mask) entry — ``init_debye``/``init_lorentz`` sum
   contributions over entries, so a duplicate entry would silently
   double delta_eps on overlap cells. Plain id-keying is the recorded
   do-not-repeat from the closed PR #272 branch (overlap-cell beta
   ratio 2.000 vs 1.000 on main).
2. Traced poles compile: a pole carrying a traced field must not raise
   TypeError in ``_assemble_materials``, and ``jax.grad`` flows through
   the full ``forward()`` pass w.r.t. a pole parameter (prerequisite
   for the differentiable-dispersion track, #273). Verified 2026-07-19:
   the full forward grad works — no deeper blocker past compile.
3. Value-dedupe caveat: poles that differ by any representable amount
   (e.g. tau perturbed by a relative 1e-6) stay DISTINCT entries.
   Over-parameterization demos that need genuinely redundant poles must
   perturb a field to keep the poles value-distinct.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.geometry.csg import Box
from rfx.geometry.rasterize import (
    coords_from_uniform_grid,
    rasterize_geometry,
)
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz, lorentz_pole


# Shared toy geometry: two boxes overlapping on x in [8, 12] mm inside a
# (20 x 10 x 10) mm PEC domain (imitates
# test_lorentz_poles_stay_scoped_to_their_material).
_DOMAIN = (0.02, 0.01, 0.01)
_FREQ_MAX = 8e9
_BOX_A = ((0.000, 0.000, 0.000), (0.012, 0.010, 0.010))
_BOX_B = ((0.008, 0.000, 0.000), (0.020, 0.010, 0.010))
_BOX_UNION = ((0.000, 0.000, 0.000), (0.020, 0.010, 0.010))


def _overlap_cell(grid):
    """Index of one cell inside BOX_A ∩ BOX_B."""
    mask_a = np.array(Box(*_BOX_A).mask(grid))
    mask_b = np.array(Box(*_BOX_B).mask(grid))
    overlap = mask_a & mask_b
    assert overlap.any(), "test geometry must overlap"
    return tuple(np.argwhere(overlap)[0])


def test_equal_debye_poles_on_overlapping_boxes_apply_once():
    """PUBLIC-API double-count lock (Debye).

    Two ``add_material`` calls with EQUAL Debye poles on overlapping
    boxes: on an overlap cell the applied beta equals the single-pole
    value (ratio 1.0). Under the PR #272-branch id-keying regression the
    ratio was 2.0 (delta_eps applied twice).
    """
    pole_kwargs = dict(delta_eps=1.5, tau=8e-12)

    sim = Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, boundary="pec")
    sim.add_material("mat_a", eps_r=2.0,
                     debye_poles=[DebyePole(**pole_kwargs)])
    sim.add_material("mat_b", eps_r=2.0,
                     debye_poles=[DebyePole(**pole_kwargs)])
    sim.add(Box(*_BOX_A), material="mat_a")
    sim.add(Box(*_BOX_B), material="mat_b")

    grid = sim._build_grid()
    _, debye, _ = sim._build_materials(grid)
    coeffs, _state = debye

    # Structural lock: ONE merged pole entry, not two.
    assert coeffs.beta.shape[0] == 1, (
        f"equal poles must merge into one entry, got "
        f"{coeffs.beta.shape[0]}")

    # Physics lock: applied beta on an overlap cell == single-pole value.
    ref = Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, boundary="pec")
    ref.add_material("mat", eps_r=2.0, debye_poles=[DebyePole(**pole_kwargs)])
    ref.add(Box(*_BOX_UNION), material="mat")
    ref_grid = ref._build_grid()
    _, ref_debye, _ = ref._build_materials(ref_grid)
    ref_coeffs, _ = ref_debye

    cell = _overlap_cell(grid)
    beta_two_mats = float(jnp.sum(coeffs.beta, axis=0)[cell])
    beta_single = float(jnp.sum(ref_coeffs.beta, axis=0)[cell])
    assert beta_single > 0.0
    ratio = beta_two_mats / beta_single
    assert ratio == pytest.approx(1.0, rel=1e-6), (
        f"overlap-cell beta ratio {ratio} != 1.0 — equal poles were "
        f"double-applied (the #272-branch id-keying regression)")


def test_equal_lorentz_poles_on_overlapping_boxes_apply_once():
    """PUBLIC-API double-count lock (Lorentz analog)."""
    def make_pole():
        return lorentz_pole(1.0, 2 * np.pi * 3e9, 1e8)

    sim = Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, boundary="pec")
    sim.add_material("mat_a", eps_r=2.0, lorentz_poles=[make_pole()])
    sim.add_material("mat_b", eps_r=2.0, lorentz_poles=[make_pole()])
    sim.add(Box(*_BOX_A), material="mat_a")
    sim.add(Box(*_BOX_B), material="mat_b")

    grid = sim._build_grid()
    _, _, lorentz = sim._build_materials(grid)
    coeffs, _state = lorentz

    assert coeffs.c.shape[0] == 1, (
        f"equal poles must merge into one entry, got {coeffs.c.shape[0]}")

    ref = Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, boundary="pec")
    ref.add_material("mat", eps_r=2.0, lorentz_poles=[make_pole()])
    ref.add(Box(*_BOX_UNION), material="mat")
    ref_grid = ref._build_grid()
    _, _, ref_lorentz = ref._build_materials(ref_grid)
    ref_coeffs, _ = ref_lorentz

    cell = _overlap_cell(grid)
    c_two_mats = float(jnp.sum(coeffs.c, axis=0)[cell])
    c_single = float(jnp.sum(ref_coeffs.c, axis=0)[cell])
    assert c_single > 0.0
    ratio = c_two_mats / c_single
    assert ratio == pytest.approx(1.0, rel=1e-6), (
        f"overlap-cell Lorentz c ratio {ratio} != 1.0 — equal poles "
        f"were double-applied")


def test_distinct_but_equal_poles_merge_into_one_entry():
    """Distinct-but-equal pole OBJECTS still dedupe (value keying).

    The spec must contain one pole whose mask is the UNION of both
    boxes' masks — including the non-overlapping regions.
    """
    sim = Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, boundary="pec")
    sim.add_material("mat_a", eps_r=2.0,
                     debye_poles=[DebyePole(delta_eps=1.5, tau=8e-12)])
    sim.add_material("mat_b", eps_r=2.0,
                     debye_poles=[DebyePole(delta_eps=1.5, tau=8e-12)])
    sim.add(Box(*_BOX_A), material="mat_a")
    sim.add(Box(*_BOX_B), material="mat_b")

    grid = sim._build_grid()
    _, debye_spec, _, *_ = sim._assemble_materials(grid)
    poles, masks = debye_spec

    assert len(poles) == 1
    assert len(masks) == 1
    expected = np.array(Box(*_BOX_A).mask(grid)) | np.array(
        Box(*_BOX_B).mask(grid))
    assert np.array_equal(np.array(masks[0]), expected), (
        "merged mask must be the union of both boxes' masks")


def test_value_perturbed_poles_stay_distinct():
    """Value-dedupe caveat: a relative-1e-6 tau perturbation keeps two
    entries.

    Over-parameterization demos that need genuinely redundant poles must
    perturb a field like this — deliberately duplicated identical poles
    merge under value keying.
    """
    tau = 8e-12
    sim = Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, boundary="pec")
    sim.add_material("mat_a", eps_r=2.0,
                     debye_poles=[DebyePole(delta_eps=1.5, tau=tau)])
    sim.add_material("mat_b", eps_r=2.0,
                     debye_poles=[DebyePole(delta_eps=1.5,
                                            tau=tau * (1.0 + 1e-6))])
    sim.add(Box(*_BOX_A), material="mat_a")
    sim.add(Box(*_BOX_B), material="mat_b")

    grid = sim._build_grid()
    _, debye_spec, _, *_ = sim._assemble_materials(grid)
    poles, masks = debye_spec

    assert len(poles) == 2, (
        f"value-distinct poles must stay separate entries, got "
        f"{len(poles)}")
    # Each pole's mask stays scoped to its own box.
    mask_a = np.array(Box(*_BOX_A).mask(grid))
    mask_b = np.array(Box(*_BOX_B).mask(grid))
    assert np.array_equal(np.array(masks[0]), mask_a)
    assert np.array_equal(np.array(masks[1]), mask_b)


def _build_traced_debye_sim(delta_eps):
    """Sim whose Debye pole carries a (possibly traced) delta_eps."""
    sim = Simulation(freq_max=15e9, domain=(0.012, 0.012, 0.012),
                     boundary="pec")
    sim.add_material("m", eps_r=2.0,
                     debye_poles=[DebyePole(delta_eps=delta_eps,
                                            tau=8e-12)])
    sim.add(Box((0.002, 0.002, 0.002), (0.010, 0.010, 0.010)),
            material="m")
    return sim


def test_traced_debye_pole_compiles():
    """A DebyePole with a traced field must not raise TypeError at
    compile (``_assemble_materials``); grad flows through the ADE
    coefficients.

    On main this raised ``TypeError: unhashable type: 'JVPTracer'`` at
    the ``pole in debye_masks_by_pole`` dict lookup.
    """
    def loss(delta_eps):
        sim = _build_traced_debye_sim(delta_eps)
        grid = sim._build_grid()
        materials, debye_spec, _, *_ = sim._assemble_materials(grid)
        assert debye_spec is not None
        coeffs, _ = init_debye(debye_spec[0], materials, grid.dt,
                               mask=debye_spec[1])
        return jnp.sum(coeffs.beta)

    g = jax.grad(loss)(jnp.float32(3.0))
    assert np.isfinite(float(g))
    assert float(g) != 0.0, "grad through init_debye beta must be nonzero"


def test_traced_debye_pole_grad_through_forward():
    """Full ``jax.grad`` smoke through ``forward()`` w.r.t. a pole field.

    Scoping note (#274 -> #273): the compile-layer fix is sufficient —
    no deeper blocker was found past compile. ``skip_preflight=True``
    keeps the traced test scoped to the solver path (preflight also
    traces cleanly today, but its advisory text is not this lock's
    subject).
    """
    def loss(delta_eps):
        sim = _build_traced_debye_sim(delta_eps)
        sim.add_source((0.006, 0.006, 0.006), "ez")
        sim.add_probe((0.009, 0.006, 0.006), "ez")
        res = sim.forward(n_steps=24, skip_preflight=True)
        return jnp.sum(res.time_series ** 2)

    g = jax.grad(loss)(jnp.float32(3.0))
    assert np.isfinite(float(g))
    assert float(g) != 0.0, (
        "grad through the full forward pass w.r.t. delta_eps must be "
        "nonzero")


def test_traced_lorentz_pole_compiles():
    """Lorentz analog of the traced-pole compile lock (log-space param,
    imitating ``differentiable_material_fit``)."""
    def loss(log_kappa):
        kappa = jnp.exp(log_kappa)
        sim = Simulation(freq_max=15e9, domain=(0.012, 0.012, 0.012),
                         boundary="pec")
        sim.add_material(
            "m", eps_r=2.0,
            lorentz_poles=[LorentzPole(omega_0=2 * np.pi * 3e9,
                                       delta=1e8, kappa=kappa)])
        sim.add(Box((0.002, 0.002, 0.002), (0.010, 0.010, 0.010)),
                material="m")
        grid = sim._build_grid()
        materials, _, lorentz_spec, *_ = sim._assemble_materials(grid)
        assert lorentz_spec is not None
        coeffs, _ = init_lorentz(lorentz_spec[0], materials, grid.dt,
                                 mask=lorentz_spec[1])
        return jnp.sum(coeffs.c)

    g = jax.grad(loss)(jnp.float32(np.log((2 * np.pi * 3e9) ** 2)))
    assert np.isfinite(float(g))
    assert float(g) != 0.0


def test_rasterize_geometry_path_value_dedupe_and_traced():
    """The shared NU/subgridded rasterizer (``rasterize_geometry``) has
    the same keying: equal poles merge, traced poles do not raise."""
    sim = Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, boundary="pec")
    sim.add_material("mat_a", eps_r=2.0,
                     debye_poles=[DebyePole(delta_eps=1.5, tau=8e-12)])
    sim.add_material("mat_b", eps_r=2.0,
                     debye_poles=[DebyePole(delta_eps=1.5, tau=8e-12)])
    sim.add(Box(*_BOX_A), material="mat_a")
    sim.add(Box(*_BOX_B), material="mat_b")
    grid = sim._build_grid()
    coords = coords_from_uniform_grid(grid)

    _, debye_spec, _, _, _, _ = rasterize_geometry(
        sim._geometry, sim._resolve_material, coords)
    assert len(debye_spec[0]) == 1, "equal poles must merge (rasterize)"

    # Traced pole through the same rasterizer path.
    def loss(delta_eps):
        t_sim = _build_traced_debye_sim(delta_eps)
        t_grid = t_sim._build_grid()
        t_coords = coords_from_uniform_grid(t_grid)
        materials, t_debye_spec, _, _, _, _ = rasterize_geometry(
            t_sim._geometry, t_sim._resolve_material, t_coords)
        coeffs, _ = init_debye(t_debye_spec[0], materials, t_grid.dt,
                               mask=t_debye_spec[1])
        return jnp.sum(coeffs.beta)

    g = jax.grad(loss)(jnp.float32(3.0))
    assert np.isfinite(float(g))
    assert float(g) != 0.0
