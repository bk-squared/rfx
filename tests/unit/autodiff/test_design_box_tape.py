"""Issue #1179 — a design region's permittivity, kept off the whole-grid tape.

The Yee E update is ``E <- Ca·E + Cb·curl(H)``, linear in the fields, so the
backward recursion needs the primal field only in the cells whose Ca/Cb
depend on a design variable. ``eps_override`` makes the whole grid's
coefficients traced, so reverse-mode AD keeps six grid-sized arrays per
timestep; ``design_box`` leaves them constant and redoes the update in the box
alone, so it keeps box-sized ones. Same physics, same gradient — measured here
against ``eps_override`` in float64.

What is pinned:

* the gradient and the forward value agree with ``eps_override`` to 1e-10
  relative in float64 (the formulations are equal, not approximately equal) —
  through a bare probe, through a 50 ohm port's |S11|, and on a lossy box;
* nothing on the tape is (steps x grid), and the bytes the tape gains per
  extra timestep do not follow the grid when the grid triples — the
  INVARIANT, not a byte count;
* the design variable still reaches the fields on the GPU baked-coefficient
  path, where dropping it would be a silent zero gradient;
* every combination the box does not carry raises.

This is NOT the removed ``design_mask`` (ledger; gated by
``test_design_mask_removed.py``). Nothing is masked and no field state is
``stop_gradient``'d: the arrays the tape holds are themselves box-shaped.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.ad_diagnostics import inspect_ad_saved_residuals
from tests._x64_compat import enable_x64

F0 = 8e9
DX = 2e-3
CPML = 5
DOMAIN = (24e-3, 20e-3, 16e-3)
# Both corners and every source/probe position sit on a cell centre, so the
# realized box does not depend on how round() breaks a tie at dx/2.
BOX_LO = (10e-3, 8e-3, 6e-3)
BOX_HI = (14e-3, 12e-3, 10e-3)
N_STEPS = 80


def _sim(domain=DOMAIN, boundary="cpml", **kwargs):
    sim = Simulation(freq_max=2 * F0, domain=domain, dx=DX,
                     boundary=boundary, cpml_layers=CPML, **kwargs)
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    return sim


def _box_cells(sim):
    """(slice, shape) of the realized design box — the cells, not the metres."""
    grid = sim._build_grid()
    lo = grid.position_to_index(BOX_LO)
    hi = grid.position_to_index(BOX_HI)
    sl = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
    return sl, tuple(hi[d] - lo[d] + 1 for d in range(3))


def _eps_design(shape, seed=1):
    rng = np.random.default_rng(seed)
    return 2.0 + 2.0 * rng.random(shape)


def _loss_box(sim, eps_design, n_steps=N_STEPS, **kwargs):
    # checkpoint=False throughout: forward()'s default per-step remat saves
    # the scan CARRY (six grid-sized field arrays) once per step whatever the
    # coefficients are, which hides the term this design removes -- measured
    # 0.59 MB of residuals here against 26.2 MB with the default on.
    result = sim.forward(design_box=(BOX_LO, BOX_HI),
                         design_eps_override=eps_design,
                         n_steps=n_steps, checkpoint=False,
                         skip_preflight=True, **kwargs)
    return jnp.sum(result.time_series ** 2)


def _loss_override(sim, eps_design, n_steps=N_STEPS, **kwargs):
    sl, _ = _box_cells(sim)
    base = sim._assemble_materials(sim._build_grid())[0].eps_r
    full = base.astype(jnp.result_type(eps_design)).at[sl].set(eps_design)
    result = sim.forward(eps_override=full, n_steps=n_steps, checkpoint=False,
                         skip_preflight=True, **kwargs)
    return jnp.sum(result.time_series ** 2)


# ---------------------------------------------------------------------------
# Equality with the whole-grid formulation
# ---------------------------------------------------------------------------

def test_design_box_gradient_equals_eps_override_in_float64():
    """The two formulations are the same computation, to round-off."""
    with enable_x64():
        sim = _sim(precision="float64")
        _, shape = _box_cells(sim)
        eps = jnp.asarray(_eps_design(shape), jnp.float64)

        value_box = float(_loss_box(sim, eps))
        value_ref = float(_loss_override(sim, eps))
        assert value_ref != 0.0, "fixture excites nothing"
        assert abs(value_box - value_ref) / abs(value_ref) < 1e-10

        grad_box = np.asarray(jax.grad(lambda e: _loss_box(sim, e))(eps))
        grad_ref = np.asarray(jax.grad(lambda e: _loss_override(sim, e))(eps))
        assert np.all(np.isfinite(grad_box))
        assert np.linalg.norm(grad_ref) > 0.0, "reference gradient vanished"
        rel = np.linalg.norm(grad_box - grad_ref) / np.linalg.norm(grad_ref)
        assert rel < 1e-10, f"gradient differs by {rel:.3e} relative"


def test_design_box_matches_eps_override_through_a_driven_port():
    """Same equality for an |S11| objective, with a 50 ohm port driving.

    The port folds its impedance into sigma at its own cell and reads the
    material there to build its drive coefficient, which is the read the
    box fences against — so a port OUTSIDE the box has to keep agreeing.
    ``DesignRegion`` is passed straight in as the box, the other accepted
    spelling.
    """
    from rfx import DesignRegion

    with enable_x64():
        sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX,
                         boundary="cpml", cpml_layers=CPML,
                         precision="float64")
        sim.add_port(position=(4e-3, 10e-3, 8e-3), component="ez",
                     impedance=50.0,
                     waveform=GaussianPulse(f0=F0, bandwidth=0.8))
        region = DesignRegion(corner_lo=BOX_LO, corner_hi=BOX_HI)
        sl, shape = _box_cells(sim)
        eps = jnp.asarray(_eps_design(shape, seed=5), jnp.float64)
        freqs = jnp.asarray([F0], jnp.float64)
        base = sim._assemble_materials(sim._build_grid())[0].eps_r

        def s11_box(e):
            r = sim.forward(design_box=region, design_eps_override=e,
                            n_steps=N_STEPS, checkpoint=False,
                            skip_preflight=True, port_s11_freqs=freqs)
            return jnp.sum(jnp.abs(r.s_params) ** 2)

        def s11_ref(e):
            full = base.astype(jnp.float64).at[sl].set(e)
            r = sim.forward(eps_override=full, n_steps=N_STEPS,
                            checkpoint=False, skip_preflight=True,
                            port_s11_freqs=freqs)
            return jnp.sum(jnp.abs(r.s_params) ** 2)

        value_box, value_ref = float(s11_box(eps)), float(s11_ref(eps))
        assert 0.0 < value_ref
        assert abs(value_box - value_ref) / value_ref < 1e-10
        grad_box = np.asarray(jax.grad(s11_box)(eps))
        grad_ref = np.asarray(jax.grad(s11_ref)(eps))
        assert np.linalg.norm(grad_ref) > 0.0
        rel = np.linalg.norm(grad_box - grad_ref) / np.linalg.norm(grad_ref)
        assert rel < 1e-10, f"gradient differs by {rel:.3e} relative"


def test_design_box_carries_a_lossy_background_and_a_traced_sigma():
    """A conductive design region: sigma defaults to the run's own, or is traced."""
    with enable_x64():
        sim = _sim(precision="float64")
        sim.add_material("lossy", eps_r=3.0, sigma=0.05)
        sim.add(Box(BOX_LO, BOX_HI), material="lossy")
        sl, shape = _box_cells(sim)
        eps = jnp.asarray(_eps_design(shape, seed=2), jnp.float64)

        # sigma=None keeps the declared 0.05 S/m of the box cells: the box
        # update must reproduce the reference, which reads that same sigma.
        rel_default = abs(float(_loss_box(sim, eps))
                          - float(_loss_override(sim, eps)))
        assert rel_default / abs(float(_loss_override(sim, eps))) < 1e-10

        # An explicit conductivity is differentiable on the same footing.
        sigma = jnp.asarray(0.02 + 0.01 * np.arange(np.prod(shape))
                            .reshape(shape), jnp.float64)
        grad = jax.grad(lambda s: _loss_box(sim, eps, design_sigma_override=s))(sigma)
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.max(np.abs(np.asarray(grad))) > 0.0


# ---------------------------------------------------------------------------
# The invariant: what the tape holds scales with the box, not with the grid
# ---------------------------------------------------------------------------

def _residuals(sim, eps_design, **kwargs):
    return inspect_ad_saved_residuals(
        lambda e: _loss_box(sim, e, **kwargs), jnp.asarray(eps_design, jnp.float32))


def _tape_bytes_per_step(loss_of, eps_design):
    """Bytes the tape gains per extra timestep.

    A difference of two step counts, so every one-off array — the scan's
    final fields, the CPML profiles — cancels and what is left is the
    per-step history, which is the quantity this design changes.
    """
    x = jnp.asarray(eps_design, jnp.float32)
    one = inspect_ad_saved_residuals(loss_of(N_STEPS), x).total_estimated_bytes
    two = inspect_ad_saved_residuals(loss_of(2 * N_STEPS), x).total_estimated_bytes
    assert one and two
    return (two - one) / N_STEPS


def test_no_residual_carries_a_per_step_history_of_the_grid():
    """Nothing on the tape is (steps x grid).

    A handful of single grid-sized arrays survive — the scan's own final H
    and E — and they do not grow with the step count, which is the whole
    point: what the design box removes is the per-step history. The bound is
    two grids, so a per-step grid array (here 80 of them) fails it by 40x.
    """
    sim = _sim()
    _, shape = _box_cells(sim)
    cells = int(np.prod(sim._build_grid().shape))
    box_cells = int(np.prod(shape))
    assert box_cells * 8 < cells, "fixture box is not small against its grid"

    records = _residuals(sim, _eps_design(shape)).records
    assert records, "nothing was traced"
    biggest = max(records, key=lambda r: r.size or 0)
    assert (biggest.size or 0) < 2 * cells, (
        f"a residual of {biggest.size} elements ({biggest.aval}) is "
        f"{(biggest.size or 0) / cells:.0f} whole grids of {cells} cells: "
        f"{biggest.source}")


def test_tape_growth_per_step_does_not_follow_the_grid():
    """Widen the domain until the grid has twice the cells; the per-step
    tape stays where it is.

    The slope in steps is the quantity that decides whether a long solve
    fits: it is what ``checkpoint_segments`` trades compute for. Holding it
    fixed against a doubled grid is the claim, and it is a ratio, not a byte
    count. The domain triples because the absorber pad does not scale with
    it — 3x the x extent is 2.04x the cells here.
    """
    small = _sim()
    large = _sim(domain=(3 * DOMAIN[0], DOMAIN[1], DOMAIN[2]))
    cells_small = int(np.prod(small._build_grid().shape))
    cells_large = int(np.prod(large._build_grid().shape))
    assert cells_large > 1.8 * cells_small, "fixture grids are not 2x apart"

    _, shape = _box_cells(small)
    assert _box_cells(large)[1] == shape, "the design box moved with the grid"
    eps = _eps_design(shape)
    slope_small = _tape_bytes_per_step(
        lambda n: (lambda e: _loss_box(small, e, n_steps=n)), eps)
    slope_large = _tape_bytes_per_step(
        lambda n: (lambda e: _loss_box(large, e, n_steps=n)), eps)
    assert slope_small > 0, "nothing accumulates per step — fixture is dead"
    assert slope_large < 1.2 * slope_small, (
        f"the tape followed the grid: {slope_small:.0f} -> {slope_large:.0f} "
        f"bytes per step for {cells_small} -> {cells_large} cells")
    assert slope_small < 4 * cells_small, (
        f"{slope_small:.0f} bytes per step is a whole float32 grid "
        f"({4 * cells_small} bytes) or more")


def test_eps_override_is_the_grid_sized_counterexample():
    """The same fixture through eps_override: the tape IS grid-sized.

    Without this, the two tests above would also pass on a fixture that
    saves nothing either way. Measured here: six float32 grids per step.
    """
    sim = _sim()
    cells = int(np.prod(sim._build_grid().shape))
    _, shape = _box_cells(sim)
    slope = _tape_bytes_per_step(
        lambda n: (lambda e: _loss_override(sim, e, n_steps=n)),
        _eps_design(shape))
    assert slope > 4 * cells, (
        f"{slope:.0f} bytes per step is less than one float32 grid "
        f"({4 * cells} bytes) — the counterexample stopped being one")


def _mixed_boundary_sim(**kwargs):
    """One absorbing face, five reflecting ones — a patch on a ground plane.

    ``x_hi`` absorbs; ``x_lo``, y and z are PEC walls that allocate no pad,
    so the design box may sit right against them.
    """
    from rfx.boundaries.spec import Boundary, BoundarySpec

    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX, cpml_layers=CPML,
                     boundary=BoundarySpec(x=Boundary(lo="pec", hi="cpml"),
                                           y="pec", z="pec"), **kwargs)
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((16e-3, 10e-3, 8e-3), "ez")
    return sim


def test_design_box_on_a_reflecting_wall_is_accepted_and_exact():
    """A box against a PEC face is not inside an absorber.

    The absorber window the kernel writes is the axis maximum of the two
    pads, but a face with no allocated pad carries the all-no-op profile and
    adds exactly zero. Reading the axis maximum refused this configuration —
    a design region sitting on a ground plane, which is most of them.
    """
    with enable_x64():
        sim = _mixed_boundary_sim(precision="float64")
        grid = sim._build_grid()
        assert (grid.pad_x_lo, grid.pad_x_hi) == (0, CPML)
        assert (grid.pad_y_lo, grid.pad_z_lo) == (0, 0)

        # Corner at the origin: cell 0 on all three axes, flush against the
        # PEC x_lo, y_lo and z_lo walls.
        lo_m, hi_m = (0.0, 0.0, 0.0), (4e-3, 4e-3, 4e-3)
        lo = grid.position_to_index(lo_m)
        hi = grid.position_to_index(hi_m)
        assert lo == (0, 0, 0), lo
        sl = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
        shape = tuple(hi[d] - lo[d] + 1 for d in range(3))
        eps = jnp.asarray(_eps_design(shape, seed=11), jnp.float64)
        base = sim._assemble_materials(grid)[0].eps_r

        def box(e):
            r = sim.forward(design_box=(lo_m, hi_m), design_eps_override=e,
                            n_steps=N_STEPS, checkpoint=False,
                            skip_preflight=True)
            return jnp.sum(r.time_series ** 2)

        def ref(e):
            full = base.astype(jnp.float64).at[sl].set(e)
            r = sim.forward(eps_override=full, n_steps=N_STEPS,
                            checkpoint=False, skip_preflight=True)
            return jnp.sum(r.time_series ** 2)

        value_box, value_ref = float(box(eps)), float(ref(eps))
        assert value_ref != 0.0, "fixture excites nothing"
        assert abs(value_box - value_ref) / abs(value_ref) < 1e-10
        grad_box = np.asarray(jax.grad(box)(eps))
        grad_ref = np.asarray(jax.grad(ref)(eps))
        assert np.linalg.norm(grad_ref) > 0.0
        rel = np.linalg.norm(grad_box - grad_ref) / np.linalg.norm(grad_ref)
        assert rel < 1e-10, f"gradient differs by {rel:.3e} relative"


def test_fence_still_catches_the_one_absorbing_face():
    """The same grid, a box run into the x_hi CPML: still refused."""
    sim = _mixed_boundary_sim()
    grid = sim._build_grid()
    n_x = grid.shape[0]
    # Last interior cell on x, then three more into the absorber.
    x_lo_m = (n_x - grid.pad_x_hi - 1) * DX
    lo_m = (x_lo_m, 0.0, 0.0)
    hi_m = (x_lo_m + 3 * DX, 4e-3, 4e-3)
    assert grid.position_to_index(hi_m)[0] > n_x - grid.pad_x_hi - 1
    shape = tuple(grid.position_to_index(hi_m)[d]
                  - grid.position_to_index(lo_m)[d] + 1 for d in range(3))
    with pytest.raises(ValueError, match="CPML absorber"):
        sim.forward(design_box=(lo_m, hi_m),
                    design_eps_override=jnp.ones(shape, jnp.float32) * 2.0,
                    n_steps=4, skip_preflight=True)


def test_design_box_survives_the_gpu_baked_coefficient_path():
    """On a GPU backend the design variable must still reach the fields.

    ``run()`` swaps in ``update_he_fast`` when the step body is only H + E +
    PEC and the backend is not CPU. That path bakes its coefficients from
    ``materials`` and has no slot for the box redo, so the design
    permittivity would reach nothing and ``jax.grad`` would return zeros
    with no error — measured: exactly 0.0 with the eligibility term
    removed, against 1.35e-06 with it. CI runs on CPU, where the fast path
    is never eligible, so the backend is faked rather than left to chance.
    """
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX, boundary="pec")
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    _, shape = _box_cells(sim)
    eps = jnp.asarray(_eps_design(shape, seed=7), jnp.float32)

    grad_cpu = jax.grad(lambda e: _loss_box(sim, e))(eps)
    with mock.patch("jax.default_backend", return_value="gpu"):
        grad_gpu = jax.grad(lambda e: _loss_box(sim, e))(eps)
    assert float(jnp.max(jnp.abs(grad_cpu))) > 0.0
    assert np.array_equal(np.asarray(grad_cpu), np.asarray(grad_gpu))


# ---------------------------------------------------------------------------
# Fences
# ---------------------------------------------------------------------------

def _forward_with_box(sim, **kwargs):
    _, shape = _box_cells(sim)
    return sim.forward(design_box=(BOX_LO, BOX_HI),
                       design_eps_override=jnp.ones(shape, jnp.float32) * 2.0,
                       n_steps=4, skip_preflight=True, **kwargs)


def test_fence_one_argument_without_the_other():
    sim = _sim()
    _, shape = _box_cells(sim)
    with pytest.raises(ValueError, match="used together"):
        sim.forward(design_box=(BOX_LO, BOX_HI), n_steps=4,
                    skip_preflight=True)
    with pytest.raises(ValueError, match="used together"):
        sim.forward(design_eps_override=jnp.ones(shape), n_steps=4,
                    skip_preflight=True)


def test_fence_combined_with_a_whole_grid_override():
    sim = _sim()
    grid = sim._build_grid()
    for name in ("eps_override", "sigma_override", "mu_r_override"):
        with pytest.raises(ValueError, match="does not combine with"):
            _forward_with_box(sim, **{name: jnp.ones(grid.shape)})


def test_fence_shape_must_match_the_realized_box():
    sim = _sim()
    _, shape = _box_cells(sim)
    wrong = jnp.ones((shape[0], shape[1], shape[2] + 1), jnp.float32)
    with pytest.raises(ValueError, match="realizes"):
        sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=wrong,
                    n_steps=4, skip_preflight=True)


def test_fence_box_reaching_into_the_absorber():
    """The absorber is the pad OUTSIDE the declared domain, so this takes a
    negative corner to reach — which is exactly the mistake worth catching."""
    sim = _sim()
    assert sim._build_grid().pad_x_lo == CPML
    with pytest.raises(ValueError, match="CPML absorber"):
        sim.forward(design_box=((-6e-3, 8e-3, 6e-3), (-2e-3, 12e-3, 10e-3)),
                    design_eps_override=jnp.ones((3, 3, 3), jnp.float32),
                    n_steps=4, skip_preflight=True)


def test_fence_box_holding_the_source_cell():
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_source((12e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    with pytest.raises(ValueError, match="source cell"):
        _forward_with_box(sim)


def test_fence_box_holding_a_passive_port():
    """A load port leaves no source and no accumulator — the box still
    has to see it, because its impedance was folded into sigma there."""
    sim = _sim()
    sim.add_port(position=(12e-3, 10e-3, 8e-3), component="ez",
                 impedance=50.0, excite=False)
    with pytest.raises(ValueError, match="holds port cells"):
        _forward_with_box(sim)


def test_fence_box_holding_a_port_cell():
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    with pytest.warns(UserWarning, match="Series topology"):
        sim.add_lumped_rlc((12e-3, 10e-3, 8e-3), "ez", R=50.0)
    with pytest.raises(ValueError, match="lumped RLC element cell"):
        _forward_with_box(sim)


def test_fence_sheet_inside_the_box():
    sim = _sim()
    with pytest.warns(UserWarning, match="Leontovich"):
        sim.add_thin_conductor(
            Box((10e-3, 8e-3, 8e-3), (14e-3, 12e-3, 8e-3)),
            sigma_bulk=5.8e7, thickness=35e-6, surface_impedance_f0=F0)
    with pytest.raises(ValueError, match="surface_impedance_f0 sheet"):
        _forward_with_box(sim)


def test_fence_upml():
    sim = _sim(boundary="upml")
    with pytest.raises(NotImplementedError, match="upml"):
        _forward_with_box(sim)


def test_fence_dispersive_material():
    from rfx.materials.debye import DebyePole

    sim = _sim()
    sim.add_material("dispersive", eps_r=3.0,
                     debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
    sim.add(Box(BOX_LO, BOX_HI), material="dispersive")
    with pytest.raises(NotImplementedError, match="Debye/Lorentz"):
        _forward_with_box(sim)


def test_fence_kerr_material():
    sim = _sim()
    sim.add_material("kerr", eps_r=2.0, chi3=1e-20)
    sim.add(Box(BOX_LO, BOX_HI), material="kerr")
    with pytest.raises(NotImplementedError, match="Kerr"):
        _forward_with_box(sim)


def test_fence_pec_occupancy_override():
    sim = _sim()
    grid = sim._build_grid()
    with pytest.raises(NotImplementedError, match="pec_occupancy_override"):
        _forward_with_box(
            sim, pec_occupancy_override=jnp.zeros(grid.shape, jnp.float32))


def test_fence_fourth_order_stencil():
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX, boundary="pec",
                     stencil_order=4)
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    with pytest.raises(NotImplementedError, match="stencil_order=4"):
        _forward_with_box(sim)


def test_fence_nonuniform_lane():
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX, boundary="cpml",
                     cpml_layers=CPML,
                     dz_profile=1.0e-3 * 1.05 ** np.arange(16, dtype=float))
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((20e-3, 10e-3, 8e-3), "ez")
    with pytest.raises(NotImplementedError, match="uniform single-device"):
        sim.forward(design_box=(BOX_LO, BOX_HI),
                    design_eps_override=jnp.ones((3, 3, 3), jnp.float32),
                    n_steps=4, skip_preflight=True)


def test_fence_adi_solver():
    sim = _sim(solver="adi")
    with pytest.raises(NotImplementedError, match="Yee solver"):
        _forward_with_box(sim)


def test_fence_anisotropic_and_bloch_paths():
    """The resolved-flag fences, at the one place that sees the flags.

    ``aniso_eps`` / ``aniso_inv_eps`` / the Bloch phase are not reachable
    through ``forward(design_box=...)`` today, so they are checked on the
    resolver itself rather than through a fixture that cannot exist.
    """
    from rfx.simulation import DesignBoxSpec, _resolve_design_box

    sim = _sim()
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid)[0]
    spec = DesignBoxSpec(bounds=(10, 13, 9, 12, 8, 11),
                         eps_r=jnp.ones((3, 3, 3), jnp.float32))
    common = dict(grid=grid, materials=materials, dt=grid.dt, use_cpml=True,
                  use_upml=False, cpml_axes="xyz", use_debye=False,
                  use_lorentz=False, use_kerr=False, aniso_eps=None,
                  aniso_inv_eps=None, stencil_order=2, bloch=None,
                  sheet_impedance=None, cell_metas=())
    # The same arguments without a flag resolve cleanly.
    assert _resolve_design_box(spec, **common).bounds == spec.bounds
    ones = jnp.ones(grid.shape, jnp.float32)
    for flag, value in (("aniso_eps", (ones, ones, ones)),
                        ("aniso_inv_eps", (ones, ones, ones)),
                        ("bloch", (1.0 + 0j, 1.0 + 0j, 1.0 + 0j))):
        with pytest.raises(NotImplementedError, match="design-box"):
            _resolve_design_box(spec, **{**common, flag: value})


def test_fence_on_an_axis_the_grid_does_not_absorb_but_the_run_does():
    """A grid built with ``cpml_axes="z"`` allocates no pad on x, yet a direct
    ``run(cpml_axes="xyz")`` still writes a real absorber there. A pad of 0
    on that axis means "no padding cells", not "no absorber", so the fence
    must fall back to the axis window and refuse a box at the x face.
    Only ``rfx.simulation.run`` can reach this; ``Simulation.forward``
    always passes the grid's own ``cpml_axes``.
    """
    from rfx.grid import Grid
    from rfx.simulation import DesignBoxSpec, _resolve_design_box
    from rfx.core.yee import init_materials

    grid = Grid(freq_max=16e9, domain=(24e-3, 20e-3, 16e-3), dx=2e-3,
                cpml_layers=5, cpml_axes="z")
    assert (grid.pad_x_lo, grid.pad_x_hi) == (0, 0)
    mats = init_materials(tuple(grid.shape))
    box = (0, 3, 4, 7, 6, 9)
    spec = DesignBoxSpec(bounds=box, eps_r=jnp.ones((3, 3, 3), jnp.float32) * 2.0)
    kw = dict(grid=grid, materials=mats, dt=grid.dt, use_cpml=True, use_upml=False,
              use_debye=False, use_lorentz=False, use_kerr=False, aniso_eps=None,
              aniso_inv_eps=None, stencil_order=2, bloch=None,
              sheet_impedance=None, cell_metas=())
    with pytest.raises(ValueError, match="CPML absorber"):
        _resolve_design_box(spec, cpml_axes="xyz", **kw)
    # The grid's own axes: x carries no absorber, the box is accepted.
    _resolve_design_box(spec, cpml_axes="z", **kw)
