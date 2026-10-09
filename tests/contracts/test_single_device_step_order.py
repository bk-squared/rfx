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
from rfx.lumped import LumpedRLCSpec, build_rlc_meta, init_rlc_state, update_rlc_element
from rfx.materials.thin_conductor import (
    SheetImpedanceCtx, apply_sheet_impedance_e, sheet_update_coeffs,
)
from rfx.simulation import _StepContext, make_core_step


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
        occupancy = jnp.zeros(grid.shape).at[1:3, 1:3, 1:3].set(.5)
        ctx = replace(ctx, use_pec_occupancy=True, pec_occupancy=occupancy)
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
        st = apply_pec_occupancy(st, ctx.pec_occupancy, ctx.periodic)
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
