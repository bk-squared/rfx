"""Issue #1183 — the design box on a graded mesh, and on a traced PEC occupancy.

Two design regions rfx's own inverse design uses. A thin substrate is meshed
with a graded ``dz_profile``, so its dielectric design region sits on the
non-uniform lane; a metal shape is designed through the relaxed-conductor
occupancy, not through a permittivity. #1179/#1180 kept the gradient tape
box-shaped only for a permittivity on the uniform mesh, and both of these
raised.

The permittivity box is the same argument as #1179 with one substitution:
``E <- Ca·E + Cb·curl(H)`` is linear in the fields on a graded mesh too, and
the only thing that changes is which curl — ``curl_h_nu`` differences two H
cell centres whose separation is the MEAN of the two adjacent cell widths.
So the box redo takes the graded curl and nothing else moves.

The occupancy box is the same argument applied one stage later. The relaxed
conductor scales every E component by ``1 - M``, ``M`` the noisy-OR of the
component's incident cells' occupancy. That multiply is where a traced
whole-grid occupancy puts three grid-sized primal E arrays per timestep, and
an occupancy cell reaches ``M`` only at itself and its plus-side neighbours —
so a box plus one cell is the whole set of cells it can move, and ``1 - M``
over that window is built once, outside the time loop, because an occupancy
is a design variable and not a field.

What is pinned:

* graded mesh — value and gradient agree with the permittivity written into
  ``materials`` to 1e-10 relative in float64 on the step loop, and through
  ``Simulation.forward`` the two formulations stay within 3x of that lane's
  own float32 rounding (the lane stores float32 fields whatever
  ``precision`` says, #630, so 1e-10 is not available there and the honest
  bar is the size of a float32 rounding on the same formulation);
* occupancy — value and gradient agree with ``pec_occupancy_override`` to
  1e-10 relative in float64, alone and composed with a static occupancy, and
  the window rule is BIT-identical to the grid-wide one at four placements
  including the grid faces;
* nothing on either tape is (steps x grid), and the bytes each tape gains per
  extra timestep do not follow the grid when the grid grows;
* the design occupancy still reaches the fields on the GPU baked-coefficient
  path, where dropping it would be a silent zero gradient;
* every combination neither box carries raises.

Like #1180 this is NOT the removed ``design_mask``: nothing is masked and no
field state is ``stop_gradient``'d.

Why the graded-mesh tape is measured on the step loop and not through
``forward()``: the non-uniform grid build calls ``float()`` on a grid array
(``rfx/boundaries/cpml.py::_get_axis_cell_sizes``), so the lane cannot be
traced by ``jax.make_jaxpr`` — which is what ``saved_residuals`` needs. The
loop below runs the lane's own kernels in the lane's own order.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import GaussianPulse, Simulation
from rfx.ad_diagnostics import inspect_ad_saved_residuals
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.boundaries.pec import (
    apply_pec, apply_pec_occupancy, apply_pec_occupancy_box,
    pec_occupancy_box_keep,
)
from rfx.core.yee import (
    MaterialArrays, e_update_coeffs, init_state, init_materials,
    update_e_box, update_e_nu, update_h_nu,
)
from rfx.nonuniform import make_nonuniform_grid, position_to_index
from rfx.simulation import _design_box_edge_coeffs
from tests._x64_compat import enable_x64

F0 = 8e9
DX = 2e-3
CPML = 5
DOMAIN = (24e-3, 20e-3, 16e-3)
N_STEPS = 80

# Graded z mesh: eight uniform cells, then a 1.06 ramp — a thin-substrate
# profile, the case #1183 exists for.
DZ_PROFILE = np.concatenate([
    np.full(8, 1.6e-3), 1.6e-3 * 1.06 ** np.arange(1, 5, dtype=float)])

# Both corners on a cell centre, so the realized box does not depend on how
# round() breaks a tie (same discipline as test_design_box_tape.py).
NU_BOX_LO = (10e-3, 8e-3, 4e-3)
NU_BOX_HI = (14e-3, 12e-3, 7e-3)
BOX_LO = (10e-3, 8e-3, 6e-3)
BOX_HI = (14e-3, 12e-3, 10e-3)


# ---------------------------------------------------------------------------
# Part A — the graded mesh
# ---------------------------------------------------------------------------

def _nu_sim(domain=DOMAIN, **kwargs):
    sim = Simulation(freq_max=2 * F0, domain=domain, dx=DX, boundary="cpml",
                     cpml_layers=CPML, dz_profile=DZ_PROFILE, **kwargs)
    sim.add_source((4e-3, 10e-3, 6e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 6e-3), "ez")
    return sim


def _nu_box_cells(sim):
    """(slice, shape) of the realized design box on the graded grid."""
    grid = sim._build_nonuniform_grid()
    lo = position_to_index(grid, NU_BOX_LO)
    hi = position_to_index(grid, NU_BOX_HI)
    sl = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
    return sl, tuple(hi[d] - lo[d] + 1 for d in range(3))


def _eps_design(shape, seed=1):
    rng = np.random.default_rng(seed)
    return 2.0 + 2.0 * rng.random(shape)


def _nu_base_eps(sim):
    from rfx.runners.nonuniform import assemble_materials_nu
    return assemble_materials_nu(sim, sim._build_nonuniform_grid())[0].eps_r


def _nu_loss_box(sim, eps_design, n_steps=N_STEPS, **kwargs):
    result = sim.forward(design_box=(NU_BOX_LO, NU_BOX_HI),
                         design_eps_override=eps_design,
                         n_steps=n_steps, checkpoint=False,
                         skip_preflight=True, **kwargs)
    return jnp.sum(result.time_series ** 2)


def _nu_loss_override(sim, eps_design, n_steps=N_STEPS, **kwargs):
    sl, _ = _nu_box_cells(sim)
    full = _nu_base_eps(sim).astype(
        jnp.result_type(eps_design)).at[sl].set(eps_design)
    result = sim.forward(eps_override=full, n_steps=n_steps, checkpoint=False,
                         skip_preflight=True, **kwargs)
    return jnp.sum(result.time_series ** 2)


def _nu_step_loop(box=(8, 12, 8, 12, 6, 10), n_steps=120, nxy=8e-3,
                  field_dtype=jnp.float32):
    """The NU lane's own kernels, in the lane's own order, as one loss.

    ``update_h_nu -> apply_cpml_h -> update_e_nu -> [design box] ->
    apply_cpml_e -> apply_pec -> source`` is the order
    ``rfx/nonuniform.py``'s ``step_fn`` runs (its TFSF, waveguide, sheet and
    RLC slots are empty here). Driving the kernels directly is what lets the
    field dtype be float64 — the runner hardcodes float32 state (#630) — and
    what lets ``saved_residuals`` trace the tape at all.

    Returns ``(make_loss, grid, box_shape)``; ``make_loss(split, remat)``
    gives a loss of the design permittivity, ``split=False`` writing it into
    ``materials`` and ``split=True`` handing it to ``update_e_box``.
    """
    grid = make_nonuniform_grid((nxy, nxy), DZ_PROFILE / 4.0, 0.5e-3,
                                cpml_layers=4)
    shape = grid.shape
    ones = jnp.ones(shape, field_dtype)
    mats = MaterialArrays(eps_r=ones, sigma=0.0 * ones, mu_r=ones)
    dt = float(grid.dt)
    sl = tuple(slice(box[2 * d], box[2 * d + 1]) for d in range(3))
    box_shape = tuple(box[2 * d + 1] - box[2 * d] for d in range(3))
    cpml_params, cpml_init = init_cpml(grid, field_dtype=field_dtype)
    src = position_to_index(grid, (0.004, 0.004, 0.004))
    prb = position_to_index(grid, (0.005, 0.005, 0.006))
    t = np.arange(n_steps) * dt
    drive = jnp.asarray(
        np.exp(-((t - 1.0e-10) / 3e-11) ** 2) * np.sin(2 * np.pi * 10e9 * t),
        field_dtype)

    def make_loss(split, remat=False):
        def loss(eps_design):
            m = mats if split else mats._replace(
                eps_r=ones.astype(jnp.result_type(eps_design))
                .at[sl].set(eps_design))
            if split:
                # #1210: the box's coefficients come from the product's own
                # builder, not a hand-written e_update_coeffs on the box
                # alone. A Yee E component on the box's minus face is shared
                # with a background cell, so its coefficient is the mean of
                # the two -- and the design material reaches one cell past
                # the box on the plus side, which is why the redo writes a
                # window instead of the declared box. Building it here by
                # hand was a second spelling of the material-to-edge rule,
                # and it is the spelling #1210 corrected.
                box_w, ca, cb = _design_box_edge_coeffs(
                    box, eps_design, jnp.zeros_like(eps_design), m, dt, shape)

            def step(carry, n):
                st, cpml, acc = carry
                st = update_h_nu(st, m, dt, grid.inv_dx_h, grid.inv_dy_h,
                                 grid.inv_dz_h)
                st, cpml = apply_cpml_h(st, cpml_params, cpml, grid, "xyz",
                                        materials=m)
                prev = st
                st = update_e_nu(st, m, dt, grid.inv_dx, grid.inv_dy,
                                 grid.inv_dz)
                if split:
                    st = update_e_box(
                        st, prev, box_w, ca, cb, grid.dx,
                        inv_d=(grid.inv_dx, grid.inv_dy, grid.inv_dz))
                st, cpml = apply_cpml_e(st, cpml_params, cpml, grid, "xyz",
                                        materials=m)
                st = apply_pec(st)
                st = st._replace(ez=st.ez.at[src].add(drive[n]))
                return (st, cpml, acc + st.ez[prb] ** 2), None

            body = jax.checkpoint(step) if remat else step
            init = (init_state(shape, field_dtype=field_dtype), cpml_init,
                    jnp.zeros((), field_dtype))
            (_, _, acc), _ = jax.lax.scan(body, init, jnp.arange(n_steps))
            return acc
        return loss

    return make_loss, grid, box_shape


def test_nu_design_box_equals_the_whole_grid_permittivity_in_float64():
    """On a graded mesh the two formulations are the same computation.

    The graded curl is the only difference from #1179's uniform claim, and
    it enters both arms identically — ``update_e_nu`` and ``update_e_box``
    call the same ``curl_h_nu`` on the same H.
    """
    with enable_x64():
        make_loss, _, box_shape = _nu_step_loop(field_dtype=jnp.float64)
        eps = jnp.asarray(_eps_design(box_shape, seed=2), jnp.float64)
        loss_ref, loss_box = make_loss(False), make_loss(True)

        value_ref, value_box = float(loss_ref(eps)), float(loss_box(eps))
        assert value_ref != 0.0, "fixture excites nothing"
        assert abs(value_box - value_ref) / abs(value_ref) < 1e-10

        grad_ref = np.asarray(jax.grad(loss_ref)(eps))
        grad_box = np.asarray(jax.grad(loss_box)(eps))
        assert np.all(np.isfinite(grad_box))
        assert np.linalg.norm(grad_ref) > 0.0, "reference gradient vanished"
        rel = np.linalg.norm(grad_box - grad_ref) / np.linalg.norm(grad_ref)
        assert rel < 1e-10, f"gradient differs by {rel:.3e} relative"


def test_nu_design_box_through_forward_sits_below_the_lane_float32_floor():
    """End to end on the graded lane, against ``eps_override``.

    This lane stores float32 fields whatever ``precision`` asks for (#630
    threads no field dtype through the NU runner), so 1e-10 is not available
    here and the honest bar is the lane's own float32 floor. The floor is
    measured on the SAME formulation: ``eps_override`` given the identical
    permittivity as float32 and as float64 — pure material rounding, no
    change of formulation. The two formulations have to differ by less
    than that.
    """
    with enable_x64():
        sim = _nu_sim()
        _, shape = _nu_box_cells(sim)
        eps64 = jnp.asarray(_eps_design(shape, seed=3), jnp.float64)
        eps32 = eps64.astype(jnp.float32)

        value_box = float(_nu_loss_box(sim, eps64))
        value_ref = float(_nu_loss_override(sim, eps64))
        assert value_ref != 0.0, "fixture excites nothing"

        grad_box = np.asarray(jax.grad(
            lambda e: _nu_loss_box(sim, e))(eps64))
        grad_ref = np.asarray(jax.grad(
            lambda e: _nu_loss_override(sim, e))(eps64))
        grad_f32 = np.asarray(jax.grad(
            lambda e: _nu_loss_override(sim, e))(eps32)).astype(np.float64)
        norm = np.linalg.norm(grad_ref)
        assert norm > 0.0, "reference gradient vanished"
        assert np.all(np.isfinite(grad_box))

        floor = np.linalg.norm(grad_f32 - grad_ref) / norm
        rel = np.linalg.norm(grad_box - grad_ref) / norm
        assert 0.0 < floor < 1e-4, (
            f"the float32 floor witness itself is {floor:.3e} — the fixture "
            f"stopped measuring what it is for")
        assert rel < 3 * floor, (
            f"the two formulations differ by {rel:.3e} relative, more than "
            f"3x the same formulation's own float32 material rounding "
            f"({floor:.3e}) — that is no longer a rounding difference")
        assert rel < 1e-5, f"gradient differs by {rel:.3e} relative"
        assert abs(value_box - value_ref) / abs(value_ref) < 1e-5


def _tape_bytes_per_step(loss_of, eps_design, n_steps=120):
    """Bytes the tape gains per extra timestep.

    A difference of two step counts, so every one-off array — the final
    fields, the CPML profiles — cancels and what is left is the per-step
    history, which is the quantity the design box changes.
    """
    x = jnp.asarray(eps_design, jnp.float32)
    one = inspect_ad_saved_residuals(
        loss_of(n_steps), x).total_estimated_bytes
    two = inspect_ad_saved_residuals(
        loss_of(2 * n_steps), x).total_estimated_bytes
    assert one and two
    return (two - one) / n_steps


def test_nu_no_residual_carries_a_per_step_history_of_the_grid():
    """Nothing on the graded-mesh tape is (steps x grid)."""
    make_loss, grid, box_shape = _nu_step_loop()
    cells = int(np.prod(grid.shape))
    assert int(np.prod(box_shape)) * 8 < cells, "fixture box is not small"

    records = inspect_ad_saved_residuals(
        make_loss(True), jnp.asarray(_eps_design(box_shape), jnp.float32)
    ).records
    assert records, "nothing was traced"
    biggest = max(records, key=lambda r: r.size or 0)
    assert (biggest.size or 0) < 2 * cells, (
        f"a residual of {biggest.size} elements ({biggest.aval}) is "
        f"{(biggest.size or 0) / cells:.0f} whole grids of {cells} cells: "
        f"{biggest.source}")


def test_nu_tape_growth_per_step_does_not_follow_the_grid():
    """Widen the transverse extent; the per-step tape stays where it is."""
    make_small, grid_small, box_shape = _nu_step_loop(nxy=8e-3)
    make_large, grid_large, _ = _nu_step_loop(nxy=14e-3)
    cells_small = int(np.prod(grid_small.shape))
    cells_large = int(np.prod(grid_large.shape))
    assert cells_large > 1.8 * cells_small, "fixture grids are not 2x apart"

    eps = _eps_design(box_shape)
    slope_small = _tape_bytes_per_step(
        lambda n: _nu_step_loop(n_steps=n, nxy=8e-3)[0](True), eps)
    slope_large = _tape_bytes_per_step(
        lambda n: _nu_step_loop(n_steps=n, nxy=14e-3)[0](True), eps)
    assert slope_small > 0, "nothing accumulates per step — fixture is dead"
    assert slope_large < 1.2 * slope_small, (
        f"the tape followed the grid: {slope_small:.0f} -> {slope_large:.0f} "
        f"bytes per step for {cells_small} -> {cells_large} cells")
    assert slope_small < 4 * cells_small, (
        f"{slope_small:.0f} bytes per step is a whole float32 grid "
        f"({4 * cells_small} bytes) or more")


def test_nu_permittivity_in_materials_is_the_grid_sized_counterexample():
    """The same fixture with the permittivity in ``materials``: grid-sized.

    Without this the two tests above would also pass on a fixture that saves
    nothing either way.
    """
    _, grid, box_shape = _nu_step_loop()
    cells = int(np.prod(grid.shape))
    slope = _tape_bytes_per_step(
        lambda n: _nu_step_loop(n_steps=n)[0](False), _eps_design(box_shape))
    assert slope > 4 * cells, (
        f"{slope:.0f} bytes per step is less than one float32 grid "
        f"({4 * cells} bytes) — the counterexample stopped being one")


def test_nu_design_box_leaves_the_grid_wide_permittivity_concrete():
    """The one thing the graded lane's tape claim rests on, gated end to end.

    What makes the tape box-shaped is that ``update_e_nu`` keeps building
    its grid-wide Ca/Cb from a CONCRETE ``materials.eps_r`` while the design
    value reaches only ``update_e_box``. Revive the defect — build the
    background coefficients from the traced permittivity too — and the
    value and the gradient do not move at all, only the tape does, so no
    equality gate can see it. The saved-residual report cannot see it
    either: the non-uniform grid build calls ``float()`` on a grid array,
    so this lane cannot be traced by ``jax.make_jaxpr``. What is left, and
    what this checks, is the invariant itself.
    """
    sim = _nu_sim()
    _, shape = _nu_box_cells(sim)
    eps = jnp.asarray(_eps_design(shape, seed=9), jnp.float32)
    traced = []
    real_update_e_nu = update_e_nu

    def spy(state, materials, *args, **kwargs):
        traced.append(isinstance(materials.eps_r, jax.core.Tracer))
        return real_update_e_nu(state, materials, *args, **kwargs)

    with mock.patch("rfx.nonuniform.update_e_nu", spy):
        grad = jax.grad(lambda e: _nu_loss_box(sim, e, n_steps=12))(eps)
    assert traced, "the grid-wide E update never ran"
    assert float(jnp.max(jnp.abs(grad))) > 0.0, "the design box reached nothing"
    assert not any(traced), (
        "the grid-wide E update saw a TRACED permittivity, so every cell's "
        "Ca/Cb is a design quantity again and the tape is grid-sized")


def _nu_forward_with_box(sim, **kwargs):
    _, shape = _nu_box_cells(sim)
    return sim.forward(design_box=(NU_BOX_LO, NU_BOX_HI),
                       design_eps_override=jnp.ones(shape, jnp.float32) * 2.0,
                       n_steps=4, skip_preflight=True, **kwargs)


def test_nu_fence_waveguide_port():
    """A modal port writes its own field over a whole plane."""
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX, boundary="cpml",
                     cpml_layers=CPML, dz_profile=DZ_PROFILE)
    # No point source: a waveguide port refuses to share a run with one.
    sim.add_waveguide_port(6e-3, mode=(1, 0), f0=F0,
                           probe_offset=2, ref_offset=1)
    sim.add_probe((20e-3, 10e-3, 6e-3), "ez")
    with pytest.raises(NotImplementedError, match="waveguide port"):
        _nu_forward_with_box(sim)


def _nu_scan_case(n_steps=8):
    """A graded grid, materials and one source, for the resolver fences.

    ``Simulation.forward`` cannot reach these two: the corner resolver
    CLAMPS a position outside the domain, so no declared corner lands in
    the absorber, and ``forward`` never sets ``subpixel_smoothing`` on this
    lane. Driving ``run_nonuniform`` is what shows ``_build_nu_scan`` calls
    the shared resolver at all.
    """
    from rfx.nonuniform import make_current_source

    grid = make_nonuniform_grid((8e-3, 8e-3), DZ_PROFILE / 4.0, 0.5e-3,
                                cpml_layers=4)
    mats = init_materials(grid.shape)
    src_idx = position_to_index(grid, (4e-3, 4e-3, 4e-3))
    src = make_current_source(grid, src_idx, "ez",
                              GaussianPulse(f0=10e9, bandwidth=0.5),
                              n_steps, mats)
    probe = position_to_index(grid, (5e-3, 5e-3, 6e-3)) + ("ez",)
    return grid, mats, [src], [probe]


def test_nu_fence_box_reaching_into_the_absorber():
    """The step-level absorber fence runs on this lane.

    The CPML E correction builds its coefficient from the background
    permittivity, which the box no longer carries — the same reason as on
    the uniform lane, read off the graded grid's own per-face pads.
    """
    from rfx.nonuniform import run_nonuniform
    from rfx.simulation import DesignBoxSpec

    grid, mats, sources, probes = _nu_scan_case()
    assert grid.pad_x_lo == 4
    spec = DesignBoxSpec(bounds=(1, 4, 8, 11, 8, 11),
                         eps_r=jnp.ones((3, 3, 3), jnp.float32) * 2.0)
    with pytest.raises(ValueError, match="CPML absorber"):
        run_nonuniform(grid, mats, 8, sources=sources, probes=probes,
                       design_box=spec)


def test_nu_fence_subpixel_anisotropic_permittivity():
    """Subpixel eps on the graded mesh routes E through ``update_e_nu_aniso``.

    The box rebuilds the plain lossy update, so the resolver refuses it —
    reached here through the ``aniso_eps`` the NU subpixel path builds.
    """
    from rfx.nonuniform import run_nonuniform
    from rfx.simulation import DesignBoxSpec

    grid, mats, sources, probes = _nu_scan_case()
    ones = jnp.ones(grid.shape, jnp.float32)
    spec = DesignBoxSpec(bounds=(8, 11, 8, 11, 8, 11),
                         eps_r=jnp.ones((3, 3, 3), jnp.float32) * 2.0)
    with pytest.raises(NotImplementedError, match="aniso_eps"):
        run_nonuniform(grid, mats, 8, sources=sources, probes=probes,
                       aniso_eps=(ones, ones, ones), design_box=spec)


def test_nu_fence_box_holding_the_source_cell():
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX, boundary="cpml",
                     cpml_layers=CPML, dz_profile=DZ_PROFILE)
    sim.add_source((12e-3, 10e-3, 6e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 6e-3), "ez")
    with pytest.raises(ValueError, match="source cell"):
        _nu_forward_with_box(sim)


def test_nu_fence_until_decay():
    """The decay stop is a forward-only host loop — no tape to keep small."""
    from rfx.runners.nonuniform import run_nonuniform_path
    from rfx.simulation import DesignBoxSpec

    sim = _nu_sim()
    with pytest.raises(NotImplementedError, match="until_decay"):
        run_nonuniform_path(
            sim, n_steps=8, until_decay=1e-3,
            design_box=DesignBoxSpec(bounds=(10, 13, 9, 12, 6, 9),
                                     eps_r=jnp.ones((3, 3, 3), jnp.float32)))


# ---------------------------------------------------------------------------
# Part B — the traced PEC occupancy
# ---------------------------------------------------------------------------

def _sim(domain=DOMAIN, boundary="cpml", **kwargs):
    sim = Simulation(freq_max=2 * F0, domain=domain, dx=DX,
                     boundary=boundary, cpml_layers=CPML, **kwargs)
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    return sim


def _box_cells(sim):
    grid = sim._build_grid()
    lo = grid.position_to_index(BOX_LO)
    hi = grid.position_to_index(BOX_HI)
    sl = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
    return sl, tuple(hi[d] - lo[d] + 1 for d in range(3))


def _occ_design(shape, seed=3):
    return np.random.default_rng(seed).random(shape)


def _loss_occ_box(sim, occ, n_steps=N_STEPS, **kwargs):
    result = sim.forward(design_box=(BOX_LO, BOX_HI),
                         design_occupancy_override=occ,
                         n_steps=n_steps, checkpoint=False,
                         skip_preflight=True, **kwargs)
    return jnp.sum(result.time_series ** 2)


def _loss_occ_override(sim, occ, n_steps=N_STEPS, static=None, **kwargs):
    sl, _ = _box_cells(sim)
    grid = sim._build_grid()
    base = (jnp.zeros(grid.shape, jnp.result_type(occ)) if static is None
            else jnp.asarray(static).astype(jnp.result_type(occ)))
    result = sim.forward(pec_occupancy_override=base.at[sl].set(occ),
                         n_steps=n_steps, checkpoint=False,
                         skip_preflight=True, **kwargs)
    return jnp.sum(result.time_series ** 2)


def test_occupancy_window_reproduces_the_grid_wide_rule_everywhere():
    """The window and the grid-wide rule agree BIT for bit, at the faces too.

    The window is the box grown one cell on the plus side (the reach of the
    incident rule) and computed with one cell of context on the minus side.
    Four placements: interior, against the lo corner, against the hi corner,
    and the whole grid. A body on face 0 or n-1 of a non-periodic axis is
    where the zero pad decides the answer, which is why they are here.
    """
    with enable_x64():
        rng = np.random.default_rng(0)
        shape = (11, 9, 8)
        state = init_state(shape, field_dtype=jnp.float64)
        state = state._replace(
            ex=jnp.asarray(rng.random(shape)),
            ey=jnp.asarray(rng.random(shape)),
            ez=jnp.asarray(rng.random(shape)))
        for box in ((3, 6, 2, 5, 1, 4), (0, 3, 0, 2, 0, 2),
                    (8, 11, 6, 9, 5, 8), (0, 11, 0, 9, 0, 8)):
            sl = tuple(slice(a, b) for a, b in zip(box[::2], box[1::2]))
            box_shape = tuple(b - a for a, b in zip(box[::2], box[1::2]))
            static = jnp.asarray(rng.random(shape) * 0.5)
            design = jnp.asarray(rng.random(box_shape))
            ref = apply_pec_occupancy(state, static.at[sl].set(design))
            write, keep = pec_occupancy_box_keep(
                box, design, shape=shape, dtype=jnp.float64,
                pec_occupancy=static)
            got = apply_pec_occupancy_box(
                apply_pec_occupancy(state, static), state, write, keep)
            for comp in ("ex", "ey", "ez"):
                assert np.array_equal(
                    np.asarray(getattr(ref, comp)),
                    np.asarray(getattr(got, comp))), (box, comp)


def test_design_occupancy_gradient_equals_pec_occupancy_override_in_float64():
    """The two formulations are the same computation, to round-off."""
    with enable_x64():
        sim = _sim(precision="float64")
        _, shape = _box_cells(sim)
        occ = jnp.asarray(_occ_design(shape), jnp.float64)

        value_box = float(_loss_occ_box(sim, occ))
        value_ref = float(_loss_occ_override(sim, occ))
        assert value_ref != 0.0, "fixture excites nothing"
        assert abs(value_box - value_ref) / abs(value_ref) < 1e-10

        grad_box = np.asarray(jax.grad(lambda o: _loss_occ_box(sim, o))(occ))
        grad_ref = np.asarray(
            jax.grad(lambda o: _loss_occ_override(sim, o))(occ))
        assert np.all(np.isfinite(grad_box))
        assert np.linalg.norm(grad_ref) > 0.0, "reference gradient vanished"
        rel = np.linalg.norm(grad_box - grad_ref) / np.linalg.norm(grad_ref)
        assert rel < 1e-10, f"gradient differs by {rel:.3e} relative"


def test_design_occupancy_composes_with_a_static_occupancy():
    """A fixed conductor elsewhere on the grid, a traced patch in the box.

    The realistic case — a ground plane or a fixed feed the design does not
    touch — and the one where the window has to carry the static occupancy
    into its own noisy-OR rather than assume vacuum around the box.
    """
    with enable_x64():
        sim = _sim(precision="float64")
        grid = sim._build_grid()
        _, shape = _box_cells(sim)
        occ = jnp.asarray(_occ_design(shape, seed=5), jnp.float64)
        static = np.zeros(grid.shape)
        static[6:9, 6:9, 6:9] = 0.7
        static = jnp.asarray(static)

        value_box = float(_loss_occ_box(sim, occ, pec_occupancy_override=static))
        value_ref = float(_loss_occ_override(sim, occ, static=static))
        assert value_ref != 0.0, "fixture excites nothing"
        assert abs(value_box - value_ref) / abs(value_ref) < 1e-10

        grad_box = np.asarray(jax.grad(
            lambda o: _loss_occ_box(sim, o, pec_occupancy_override=static))(occ))
        grad_ref = np.asarray(jax.grad(
            lambda o: _loss_occ_override(sim, o, static=static))(occ))
        assert np.linalg.norm(grad_ref) > 0.0
        rel = np.linalg.norm(grad_box - grad_ref) / np.linalg.norm(grad_ref)
        assert rel < 1e-10, f"gradient differs by {rel:.3e} relative"


def _occ_tape_bytes_per_step(loss_of, occ):
    x = jnp.asarray(occ, jnp.float32)
    one = inspect_ad_saved_residuals(loss_of(N_STEPS), x).total_estimated_bytes
    two = inspect_ad_saved_residuals(
        loss_of(2 * N_STEPS), x).total_estimated_bytes
    assert one and two
    return (two - one) / N_STEPS


def test_occupancy_no_residual_carries_a_per_step_history_of_the_grid():
    """Nothing on the occupancy tape is (steps x grid)."""
    sim = _sim()
    _, shape = _box_cells(sim)
    cells = int(np.prod(sim._build_grid().shape))
    assert int(np.prod(shape)) * 8 < cells, "fixture box is not small"

    records = inspect_ad_saved_residuals(
        lambda o: _loss_occ_box(sim, o),
        jnp.asarray(_occ_design(shape), jnp.float32)).records
    assert records, "nothing was traced"
    biggest = max(records, key=lambda r: r.size or 0)
    assert (biggest.size or 0) < 2 * cells, (
        f"a residual of {biggest.size} elements ({biggest.aval}) is "
        f"{(biggest.size or 0) / cells:.0f} whole grids of {cells} cells: "
        f"{biggest.source}")


def test_occupancy_tape_growth_per_step_does_not_follow_the_grid():
    """Widen the domain until the grid has twice the cells; the per-step
    tape stays where it is."""
    small = _sim()
    large = _sim(domain=(3 * DOMAIN[0], DOMAIN[1], DOMAIN[2]))
    cells_small = int(np.prod(small._build_grid().shape))
    cells_large = int(np.prod(large._build_grid().shape))
    assert cells_large > 1.8 * cells_small, "fixture grids are not 2x apart"

    _, shape = _box_cells(small)
    assert _box_cells(large)[1] == shape, "the design box moved with the grid"
    occ = _occ_design(shape)
    slope_small = _occ_tape_bytes_per_step(
        lambda n: (lambda o: _loss_occ_box(small, o, n_steps=n)), occ)
    slope_large = _occ_tape_bytes_per_step(
        lambda n: (lambda o: _loss_occ_box(large, o, n_steps=n)), occ)
    assert slope_small > 0, "nothing accumulates per step — fixture is dead"
    assert slope_large < 1.2 * slope_small, (
        f"the tape followed the grid: {slope_small:.0f} -> {slope_large:.0f} "
        f"bytes per step for {cells_small} -> {cells_large} cells")
    assert slope_small < 4 * cells_small, (
        f"{slope_small:.0f} bytes per step is a whole float32 grid "
        f"({4 * cells_small} bytes) or more")


def test_pec_occupancy_override_is_the_grid_sized_counterexample():
    """The same fixture through ``pec_occupancy_override``: grid-sized."""
    sim = _sim()
    cells = int(np.prod(sim._build_grid().shape))
    _, shape = _box_cells(sim)
    slope = _occ_tape_bytes_per_step(
        lambda n: (lambda o: _loss_occ_override(sim, o, n_steps=n)),
        _occ_design(shape))
    assert slope > 4 * cells, (
        f"{slope:.0f} bytes per step is less than one float32 grid "
        f"({4 * cells} bytes) — the counterexample stopped being one")


def test_design_occupancy_survives_the_gpu_baked_coefficient_path():
    """On a GPU backend the design occupancy must still reach the fields.

    ``run()`` swaps in ``update_he_fast`` when the step body is only H + E +
    PEC and the backend is not CPU. That path has no occupancy slot at all,
    and a run whose ONLY occupancy is the design box would otherwise be
    eligible — the design variable would reach nothing and ``jax.grad``
    would return zeros with no error. CI runs on CPU, where the fast path is
    never eligible, so the backend is faked rather than left to chance.
    """
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX, boundary="pec")
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    _, shape = _box_cells(sim)
    occ = jnp.asarray(_occ_design(shape, seed=7), jnp.float32)

    grad_cpu = jax.grad(lambda o: _loss_occ_box(sim, o))(occ)
    with mock.patch("jax.default_backend", return_value="gpu"):
        grad_gpu = jax.grad(lambda o: _loss_occ_box(sim, o))(occ)
    assert float(jnp.max(jnp.abs(grad_cpu))) > 0.0
    assert np.array_equal(np.asarray(grad_cpu), np.asarray(grad_gpu))


def test_occupancy_fence_periodic_axis():
    """The incident rule wraps across a periodic seam, so the window is no
    longer the whole set of cells the box's occupancy can move."""
    from rfx.boundaries.spec import BoundarySpec

    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX, cpml_layers=CPML,
                     boundary=BoundarySpec(x="cpml", y="periodic", z="pec"))
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    _, shape = _box_cells(sim)
    with pytest.raises(NotImplementedError, match="periodic"):
        sim.forward(design_box=(BOX_LO, BOX_HI),
                    design_occupancy_override=jnp.zeros(shape, jnp.float32),
                    n_steps=4, skip_preflight=True)


def test_occupancy_fence_combined_with_a_design_permittivity():
    sim = _sim()
    _, shape = _box_cells(sim)
    with pytest.raises(NotImplementedError, match="one call"):
        sim.forward(design_box=(BOX_LO, BOX_HI),
                    design_eps_override=jnp.ones(shape, jnp.float32) * 2.0,
                    design_occupancy_override=jnp.zeros(shape, jnp.float32),
                    n_steps=4, skip_preflight=True)


def test_occupancy_fence_box_holding_a_port_cell():
    """A port forces the occupancy to zero at and around its own cell.

    The drive edge cannot stand inside a conductor, so the port setup
    clears the occupancy at its cell, its six face neighbours and an MSL
    port's in-plane diagonal owners. A design box writing over those cells
    realizes metal the identical occupancy handed to
    ``pec_occupancy_override`` would not have — measured before the fence:
    a 50 ohm Ez port at (12, 10, 8) mm inside the box moved the objective
    by 99 % and the gradient by 98 %, with no error raised.
    """
    sim = _sim()
    sim.add_port(position=(12e-3, 10e-3, 8e-3), component="ez",
                 impedance=50.0, excite=False)
    _, shape = _box_cells(sim)
    with pytest.raises(ValueError, match="port-cleared cell"):
        sim.forward(design_box=(BOX_LO, BOX_HI),
                    design_occupancy_override=jnp.ones(shape, jnp.float32) * 0.5,
                    n_steps=4, skip_preflight=True)


def test_occupancy_fence_kottke_occupancy_lane(monkeypatch):
    """``RFX_PEC_OCC_KOTTKE=1`` moves the occupancy into the E update.

    On that lane the occupancy becomes an inverse-eps tensor built from the
    STATIC array and ``pec_occupancy_for_run`` is set to None, so the
    design values never reach the update and the box's own ``1 - M``
    window corrects a field the tensor already handled — the double
    correction ``_forward_from_materials``'s own comment names. Measured
    before the fence: value 19x and gradient 2.8x off
    ``pec_occupancy_override``, silently.
    """
    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "1")
    sim = _sim()
    grid = sim._build_grid()
    _, shape = _box_cells(sim)
    static = jnp.zeros(grid.shape, jnp.float32).at[6:9, 6:9, 6:9].set(0.7)
    with pytest.raises(NotImplementedError, match="Kottke occupancy lane"):
        sim.forward(design_box=(BOX_LO, BOX_HI),
                    design_occupancy_override=jnp.ones(shape, jnp.float32) * 0.5,
                    pec_occupancy_override=static,
                    n_steps=4, skip_preflight=True)


def test_occupancy_fence_one_argument_without_the_other():
    sim = _sim()
    _, shape = _box_cells(sim)
    with pytest.raises(ValueError, match="used together"):
        sim.forward(design_occupancy_override=jnp.zeros(shape, jnp.float32),
                    n_steps=4, skip_preflight=True)


def test_occupancy_fence_shape_must_match_the_realized_box():
    sim = _sim()
    _, shape = _box_cells(sim)
    wrong = jnp.zeros((shape[0], shape[1], shape[2] + 1), jnp.float32)
    with pytest.raises(ValueError, match="realizes"):
        sim.forward(design_box=(BOX_LO, BOX_HI),
                    design_occupancy_override=wrong,
                    n_steps=4, skip_preflight=True)


def test_occupancy_fence_design_sigma_without_a_design_permittivity():
    sim = _sim()
    _, shape = _box_cells(sim)
    with pytest.raises(ValueError, match="design_eps_override"):
        sim.forward(design_box=(BOX_LO, BOX_HI),
                    design_occupancy_override=jnp.zeros(shape, jnp.float32),
                    design_sigma_override=jnp.zeros(shape, jnp.float32),
                    n_steps=4, skip_preflight=True)
