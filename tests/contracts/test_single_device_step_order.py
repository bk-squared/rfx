"""Conductor-bearing observations pin the single-device step's physical order."""
from dataclasses import MISSING, fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.pec import apply_pec_faces, apply_pec_occupancy
from rfx.core.drives import StepDrives, drive_layout, inject_drives
from rfx.core.yee import curl_h, init_materials, init_state, update_e, update_h
from rfx.grid import Grid
from rfx.model.materials import with_components
from rfx.model.occupancy import with_edge_keep
from rfx.lumped import LumpedRLCSpec, build_rlc_meta, init_rlc_state, update_rlc_element
from rfx.materials.thin_conductor import (
    SheetImpedanceCtx, apply_sheet_impedance_e, sheet_update_coeffs,
)
from rfx.simulation import _StepContext, make_core_step


def _occupancy(shape):
    return jnp.zeros(shape).at[1:3, 1:3, 1:3].set(.5)


def _scene(kind):
    grid = Grid(freq_max=10e9, domain=(.005, .005, .005), dx=.001, cpml_layers=0)
    required = {f.name: False for f in fields(_StepContext)
                if f.default is MISSING and f.default_factory is MISSING}
    required.update(grid=grid, materials=with_components(init_materials(grid.shape), grid,
                                              periodic=(False, False, False)),
                    dt=grid.dt, dx=grid.dx, periodic=(False, False, False),
                    pec_axes='xyz', stencil_order=2, use_pec_faces=True)
    cell = (0, 2, 2) if kind != 'occupancy' else (2, 2, 2)
    ctx = _StepContext(**required, pec_faces_frozen=frozenset({'x_lo'}),
                       drives=StepDrives(drive_layout([(*cell, 'ez')], jnp.float32), None))
    initial = init_state(grid.shape)
    initial = initial._replace(ez=initial.ez.at[cell].set(.7))
    carry = {'fdtd': initial}
    if kind == 'occupancy':
        # The step multiplies by the factor the model publishes before the loop.
        ctx = replace(ctx, use_pec_occupancy=True, materials=with_edge_keep(
            ctx.materials, _occupancy(grid.shape), dtype=jnp.float32))
    elif kind == 'sheet':
        mask = jnp.zeros(grid.shape, dtype=bool).at[0:4, 2, 1:4].set(True)
        sheet = SheetImpedanceCtx(mask, jnp.zeros_like(mask), mask, mask * 2.)
        ctx = replace(ctx, use_sheet_impedance=True, sheet_impedance=sheet)
    else:
        spec = LumpedRLCSpec(R=500., L=1e-9, C=.2e-12,
                             position=(0., .002, .002), component='ez')
        meta = build_rlc_meta(grid, spec, ctx.materials)
        ctx = replace(ctx, use_lumped_rlc=True, rlc_meta=(meta,),
                      update_rlc_element=update_rlc_element)
        carry['rlc_states'] = [init_rlc_state()]
    return ctx, carry


def _reference(ctx, carry, source, sheet_coeffs):
    # Today's single-device order. rfx issue 1373 tracks the one order for
    # all lanes; this independent straight-line test is where it will show.
    old = carry['fdtd']
    st = update_h(old, ctx.materials, ctx.dt, ctx.dx, periodic=ctx.periodic)
    st = update_e(st, ctx.materials, ctx.dt, ctx.dx, periodic=ctx.periodic)
    st = apply_pec_faces(st, ctx.pec_faces_frozen)
    if ctx.use_pec_occupancy:
        # Independent of the published array: the cell occupancy through the
        # in-place 1 - M operator (0.5 is dyadic, so the two forms agree bit for bit).
        st = apply_pec_occupancy(st, _occupancy(st.ex.shape), ctx.periodic)
    if ctx.use_sheet_impedance:
        curls = curl_h(st.hx, st.hy, st.hz, ctx.dx, ctx.periodic, 2, None)
        st = apply_sheet_impedance_e(st, (old.ex, old.ey, old.ez), curls,
                                    ctx.sheet_impedance, sheet_coeffs)
    rlc = ()
    if ctx.use_lumped_rlc:
        meta = ctx.rlc_meta[0]
        st, next_rlc = update_rlc_element(st, carry['rlc_states'][0], meta,
                                         old.ez[meta.i, meta.j, meta.k])
        rlc = tuple(next_rlc)
    before = st.ez
    st = inject_drives(st, ctx.drives.electric, source)
    return (st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, before, *rlc)


def _check(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_order_checks_are_live():
    with pytest.raises(AssertionError):
        _check((np.array([1.]),), (np.array([0.]),))


@pytest.mark.parametrize('kind', ['occupancy', 'sheet', 'rlc'])
def test_conductors_and_sources_match_independent_single_device_step(kind):
    from rfx.stepping import HookPoint
    ctx, carry = _scene(kind)
    def before(frame):
        frame.carry = {**frame.carry, 'before': frame.st.ez}
    def end(frame):
        frame.extras['before'] = frame.carry['before']
    core = make_core_step(ctx, hooks={HookPoint.BEFORE_SOURCES: (before,),
                                      HookPoint.STEP_END: (end,)})
    coeffs = (sheet_update_coeffs(ctx.sheet_impedance.sigma_sheet, ctx.materials, ctx.dt)
              if ctx.use_sheet_impedance else None)
    source = jnp.array([2.], dtype=jnp.float32)
    def actual(carry):
        new, _, extras = core(carry, jnp.int32(0), source, jnp.zeros(0))
        st = new['fdtd']
        rlc = tuple(new['rlc_states'][0]) if ctx.use_lumped_rlc else ()
        return (st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, extras['before'], *rlc)
    _check(jax.jit(actual)(carry), jax.jit(lambda c: _reference(ctx, c, source, coeffs))(carry))


def _graded_scene(kind):
    from rfx.nonuniform import make_nonuniform_grid
    from rfx.stepping.graded import GradedStepContext, graded_probe_table

    grid = make_nonuniform_grid(
        (.005, .005), np.full(5, .001), .001, cpml_layers=0,
        dy_profile=np.array([.001, .0009, .0008, .0009, .001]))
    shape = (grid.nx, grid.ny, grid.nz)
    materials = with_components(init_materials(shape), grid,
                                periodic=(False, False, False))
    cell = (0, 2, 2) if kind != 'occupancy' else (2, 2, 2)
    initial = init_state(shape)
    initial = initial._replace(ez=initial.ez.at[cell].set(.7))
    carry = {'fdtd': initial}
    from rfx.core.yee import CurlBoundary
    values = {f.name: False if f.name.startswith('use_') else None
              for f in fields(GradedStepContext)}
    values.update(
        grid=grid, dt=grid.dt, inv_dy=grid.inv_dy,
        inv_dz=grid.inv_dz, inv_dy_h=grid.inv_dy_h,
        inv_dz_h=grid.inv_dz_h, _pec_faces_frozen=frozenset({'x_lo'}),
        _pmc_faces_frozen=frozenset(),
        curl_boundary=CurlBoundary(frozenset({'x_lo'}), frozenset()),
        rlc_metas=(),
        emit_time_series=True, waveguide_meta=(),
        waveguide_port_metrics=(), wp_meta=[], dft_meta=(), flux_meta=())
    sheet = coeffs = None
    if kind == 'occupancy':
        values['use_pec_occupancy'] = True
        materials = with_edge_keep(materials, _occupancy(shape), dtype=jnp.float32)
    elif kind == 'sheet':
        values['use_sheet_impedance'] = True
        mask = jnp.zeros(shape, dtype=bool).at[0:4, 2, 1:4].set(True)
        sheet = SheetImpedanceCtx(mask, jnp.zeros_like(mask), mask, mask * 2.)
        coeffs = sheet_update_coeffs(sheet.sigma_sheet, materials, grid.dt)
    else:
        values['use_lumped_rlc'] = True
        spec = LumpedRLCSpec(R=500., L=1e-9, C=.2e-12,
                             position=(0., .002, .002), component='ez')
        values['rlc_metas'] = (build_rlc_meta(grid, spec, materials),)
        carry['rlc_states'] = (init_rlc_state(),)
    invariants = dict(inv_dx=grid.inv_dx, inv_dx_h=grid.inv_dx_h, dx_arr=grid.dx_arr,
                      drives=drive_layout([(*cell, 'ez')], jnp.float32),
                      probe_table=graded_probe_table([], shape, dtype=initial.ex.dtype),
                      materials=materials, pec_edge_masks=None, cpml_inv_eps_r=None,
                      debye_coeffs=None, lorentz_coeffs=None, aniso_eps=None,
                      sheet_impedance=sheet, sheet_coeffs=coeffs, design_box_coeffs=None)
    return GradedStepContext(**values), carry, invariants


def _graded_reference(ctx, carry, source, invariants):
    from rfx.core.yee import curl_h_nu, update_e_nu, update_h_nu

    old = carry['fdtd']
    materials = invariants['materials']
    grid = ctx.grid
    st = update_h_nu(old, materials, ctx.dt, invariants['inv_dx_h'], grid.inv_dy_h,
                     grid.inv_dz_h, cell_sizes=(invariants['dx_arr'], grid.dy_arr, grid.dz))
    st = update_e_nu(st, materials, ctx.dt, invariants['inv_dx'], grid.inv_dy,
                     grid.inv_dz, boundary=ctx.curl_boundary)
    st = apply_pec_faces(st, ctx._pec_faces_frozen)
    if ctx.use_pec_occupancy:
        st = apply_pec_occupancy(st, _occupancy(st.ex.shape), (False, False, False))
    if ctx.use_sheet_impedance:
        curls = curl_h_nu(st.hx, st.hy, st.hz, invariants['inv_dx'], grid.inv_dy,
                          grid.inv_dz, boundary=ctx.curl_boundary)
        st = apply_sheet_impedance_e(st, (old.ex, old.ey, old.ez), curls,
                                    invariants['sheet_impedance'], invariants['sheet_coeffs'])
    rlc = ()
    if ctx.use_lumped_rlc:
        meta = ctx.rlc_metas[0]
        st, next_rlc = update_rlc_element(st, carry['rlc_states'][0], meta,
                                         old.ez[meta.i, meta.j, meta.k])
        rlc = tuple(next_rlc)
    before = st.ez
    st = inject_drives(st, invariants['drives'], source)
    return (st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, before, *rlc)


def _graded_check(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_graded_order_checks_are_live():
    with pytest.raises(AssertionError):
        _graded_check((np.array([1.]),), (np.array([0.]),))


@pytest.mark.parametrize('kind', ['occupancy', 'sheet', 'rlc'])
def test_graded_conductors_and_sources_match_independent_step(kind):
    from rfx.stepping import HookPoint, make_graded_step

    ctx, carry, invariants = _graded_scene(kind)
    captured = {}

    def before(frame):
        captured['before'] = frame.st.ez

    step = make_graded_step(ctx, hooks={HookPoint.BEFORE_SOURCES: (before,)})
    source = jnp.array([2.], dtype=jnp.float32)

    def actual(carry):
        new, _ = step(carry, (jnp.int32(0), source), invariants)
        st = new['fdtd']
        rlc = tuple(new['rlc_states'][0]) if ctx.use_lumped_rlc else ()
        return (st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, captured['before'], *rlc)

    _graded_check(jax.jit(actual)(carry),
                  jax.jit(lambda c: _graded_reference(ctx, c, source, invariants))(carry))
