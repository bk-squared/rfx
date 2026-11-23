"""Drives, soft current sources and lumped RLC elements read the E update's operand.

Physics. At every E edge the step applies one coefficient
``Cb = dt/(eps0*eps_B) / (1 + sigma*dt/(2*eps0*eps_B))`` where ``eps_B`` is the
operand of the branch that steps the edge: the tensor diagonal ``tensor.eps``
(smoothed, full tensor, conformal ``eps/w``), the Stage-2 inverse
``u = tensor.inv_eps`` (``Cb = (dt*u/eps0) / (1 + sigma*dt*u/(2*eps0))``), or the
UPML form (cell-owned operands: the float32 diagonal with a tensor, else
``upml_eps``; the cell-owned conductivity ``upml_sigma``). A port drive and a soft
current source are currents in Ampere's law at their edge; a lumped RLC
element's implicit solve uses the edge's denominator ``D0 = 1/Cb``
(``eps0*eps/dt + sigma/2``, inverse form ``eps0/(dt*u) + sigma/2``). Both enter
the same equation as the curl, so both must use ``eps_B``.

Sign convention (``rfx/core/drives.py``, ``rfx/sources/port_drive.py``): a drive
adds ``table`` to E; the port table is ``Cb*sigma_port*unit_field*w``, the soft
table ``(1/dV)*(Cb*w)``; the electric update reads ``E += Cb*(curl H - J)``, so
``J = -table/Cb``. ``Cb`` and ``D0`` are positive gains.

Row form (the realized-operator rows of
``tests/unit/runners/test_realized_port_drives.py``, whose helpers are imported):
one public ``run()`` or ``forward()`` to its first E update; the value a builder
received is compared with a numpy float64 expression of the operand captured
where the step reads it (``core_step_invariants`` on the uniform lane, the
graded kernel's arguments, ``init_upml``'s ``cb`` under UPML), within relative
1e-6; the kernel's own increment for a unit curl at the judged edge is checked
against the same expression (kernel self-check); then the operand alone is
scaled 1.7x at the attach, and the builder must follow ("follows" step).
Expected values never call the code's coefficient helpers.
"""

from __future__ import annotations

import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
from rfx.core.yee import EPS_0
from tests.unit.runners.test_realized_port_drives import (
    _O2_RTOL, _O2ReferenceCaptured, _o2_cb, _o2_operands, _o2_options, _o2_scene)

AXES = ('ex', 'ey', 'ez')

_GRADED_X_PROFILE = np.array([.001] * 4 + [.0009, .0011] + [.001] * 6)


@pytest.fixture(autouse=True)
def _clear_eager_frequency_grid():
    yield
    from rfx.measurement.monitors import _eager_frequency_grid
    _eager_frequency_grid.cache_clear()


def _edge(cell):
    return tuple(int(v) for v in cell)


def _simulation_with(**override):
    """``rfx.Simulation`` with some constructor arguments replaced."""
    real = rfx.Simulation

    def make(*args, **kwargs):
        kwargs.update(override)
        return real(*args, **kwargs)
    return make


def _scene(operator, branch, *, graded=False, upml=False, radius=None, sigma=None):
    """O2's scene for ``operator``; a soft current source replaces O2's lumped port.

    ``sigma`` gives O2's eps_r 4 sphere that conductivity (S/m).
    """
    o2_branch = 'smooth' if branch in ('kottke', 'upml', 'upml_plain') else branch
    override = {}
    if graded:
        override['dx_profile'] = _GRADED_X_PROFILE
    if upml:
        override['boundary'] = 'upml'
    with pytest.MonkeyPatch.context() as patch:
        if override:
            patch.setattr(rfx, 'Simulation', _simulation_with(**override))
        sim = _o2_scene('lumped' if operator == 'soft' else operator, o2_branch)
    if sigma is not None:
        sim.add_material('dielectric', eps_r=4., sigma=sigma)
    if operator == 'soft':
        port = sim._ports.pop()
        sim.add_source(port.position, port.component, waveform=jnp.ones_like,
                       amplitude_kind='current')
    if radius is not None:
        port = sim._ports.pop()
        sim.add_port(port.position, port.component, waveform=jnp.ones_like,
                     extent=port.extent, radius=radius)
    return sim


def _entry_options(branch, entry, extra=None):
    if branch == 'kottke':
        options = dict(subpixel_smoothing='kottke_pec')
    elif branch == 'upml':
        options = dict(subpixel_smoothing=True)
    elif branch in ('occupancy', 'upml_plain'):
        options = {}
    else:
        options = dict(_o2_options(branch))
    if entry == 'forward':
        options.pop('conformal_pec', None)
        options['checkpoint'] = False
    else:
        options['compute_s_params'] = False
    options.update(n_steps=12, skip_preflight=True, **(extra or {}))
    return options


def _inverse(branch):
    return branch in ('kottke', 'occupancy')


def _upml(branch):
    return branch in ('upml', 'upml_plain')


def _operand(components, branch, axis):
    """The step's operand array: the inverse on Stage-2 lanes, UPML's cell-owned
    ``upml_eps`` under UPML without a tensor, else the diagonal."""
    if branch == 'upml_plain':
        return np.asarray(components.upml_eps[axis], dtype=np.float64)
    tensor = components.tensor
    return np.asarray(tensor.inv_eps[axis] if _inverse(branch) else tensor.eps[axis],
                      dtype=np.float64)


def _judge(seen, kind, branch, change):
    """The row's call: the first whose edge's operand differs from the plain edge mean."""
    c = seen['materials'].components
    calls = [call for call in seen['calls'] if call['kind'] == kind]
    assert calls, (kind, 'no builder call recorded', seen['calls'])
    if change is not None:
        return next(call for call in calls if (call['axis'], call['edge']) == change)
    for call in calls:
        value = _operand(c, branch, call['axis'])[call['edge']]
        eps = (1. / value if value > 0 else np.inf) if _inverse(branch) else value
        plain = float(np.asarray(c.eps_update[call['axis']], np.float64)[call['edge']])
        if np.isfinite(eps) and abs(eps - plain) > _O2_RTOL * abs(plain):
            return call
    return calls[0]


def _step_cb(seen, branch, axis, edge):
    """The step's Cb at the edge, in numpy float64 from the captured operand."""
    if _upml(branch):
        return float(np.asarray(getattr(seen['upml_coeffs'], 'cb_' + AXES[axis]))[edge])
    materials, dt = seen['materials'], seen['dt']
    if _inverse(branch):
        u = _operand(materials.components, branch, axis)[edge]
        s = float(np.asarray(materials.components.sigma_update[axis], np.float64)[edge])
        return float((dt * u / EPS_0) / (1. + s * dt * u / (2. * EPS_0)))
    return _o2_cb(materials, branch, axis, edge, dt)


def _reference(seen, branch, call):
    """Expected builder value: Cb times the drive's multiplier, or D0 = 1/Cb's operand form."""
    axis, edge = call['axis'], call['edge']
    if call['kind'] != 'rlc':
        return _step_cb(seen, branch, axis, edge) * call['multiplier']
    if _upml(branch):
        return 1. / _step_cb(seen, branch, axis, edge)
    materials, dt = seen['materials'], seen['dt']
    s = float(np.asarray(materials.components.sigma_update[axis], np.float64)[edge])
    if _inverse(branch):
        u = _operand(materials.components, branch, axis)[edge]
        return float(EPS_0 / (dt * u) + s / 2.)
    eps, sigma = _o2_operands(materials, branch, axis)
    return float(eps[edge] * EPS_0 / dt + sigma[edge] / 2.)


def _scale_on_attach(patch, branch, change, seen):
    """Scale only the step's operand at the judged edge, once, when it is attached.

    Under UPML without a tensor the operand is ``upml_eps``, scaled when the
    components first form.
    """
    from dataclasses import replace
    from rfx.model import materials as model
    original = model.with_components

    def realize(materials, grid, *args, **kwargs):
        previous = materials.components
        result = original(materials, grid, *args, **kwargs)
        tensor = result.components.tensor
        if change is not None and branch == 'upml_plain' and previous is None:
            axis, edge = change
            moved = list(result.components.upml_eps)
            moved[axis] = jnp.asarray(moved[axis]).at[edge].multiply(1.7)
            result = result._replace(components=replace(
                result.components, upml_eps=tuple(moved)))
            seen['scaled'] = True
        elif (change is not None and tensor is not None
                and (previous is None or previous.tensor is None)):
            axis, edge = change
            name = 'inv_eps' if _inverse(branch) else 'eps'
            moved = list(getattr(tensor, name))
            moved[axis] = jnp.asarray(moved[axis]).at[edge].multiply(1.7)
            result = result._replace(components=replace(
                result.components, tensor=replace(tensor, **{name: tuple(moved)})))
            seen['scaled'] = True
        return result
    patch.setattr(model, 'with_components', realize)


def _measure(make_scene, entry, branch, kind, *, lane='uniform', change=None, extra=None):
    """Run one public entry to its first E update; record builders, operand and kernel."""
    from rfx.sources import port_drive
    from rfx.model import source_coefficients
    from rfx.model import materials as model
    from rfx.stepping import uniform_setup, uniform_loop
    from rfx.stepping import yee as kernels
    from rfx.boundaries import upml as upml_module
    import rfx.lumped as lumped
    import rfx.model as model_package
    import rfx.runners.nonuniform as nu_runner

    sim = make_scene()
    seen = {'calls': [], 'soft_edges': []}
    original_drive = port_drive.port_drive_waveform
    original_denominator = lumped.edge_update_denominator
    original_table = source_coefficients.source_table
    original_uniform_table = source_coefficients.uniform_source_table
    original_graded_source = nu_runner.make_current_source
    original_invariants = uniform_setup.core_step_invariants
    original_kernel = uniform_loop._update_e_with_optional_dispersion
    original_graded_kernel = kernels.update_e_nu_aniso
    original_kernel_materials = model.kernel_materials
    original_upml_e = upml_module.apply_upml_e

    def drive(grid, cell, component, excitation, n_steps, materials, **kwargs):
        value = original_drive(grid, cell, component, excitation, n_steps, materials, **kwargs)
        seen['calls'].append(dict(kind='port', axis=AXES.index(component), edge=_edge(cell),
                                  value=float(np.asarray(value)[0]),
                                  multiplier=float(kwargs['sigma_port'] * kwargs['unit_field'])))
        return value

    def denominator(materials, cell, component, dt, *args, **kwargs):
        value = original_denominator(materials, cell, component, dt, *args, **kwargs)
        seen['calls'].append(dict(kind='rlc', axis=AXES.index(component), edge=_edge(cell),
                                  value=float(value), multiplier=1.))
        return value

    def uniform_table(grid, cell, component, waveform, n_steps, materials, kind=None, **kwargs):
        if kind == 'current':
            seen['soft_edges'].append((AXES.index(component), _edge(cell)))
        return original_uniform_table(grid, cell, component, waveform, n_steps, materials,
                                      kind, **kwargs)

    def graded_source(grid, position_ijk, component, *args, **kwargs):
        if kwargs.get('amplitude_kind') in (None, 'current'):
            seen['soft_edges'].append((AXES.index(component), _edge(position_ijk)))
        return original_graded_source(grid, position_ijk, component, *args, **kwargs)

    def table(samples, kind=None, *, native='raw', coefficient=None, volume=1.0):
        out = original_table(samples, kind, native=native, coefficient=coefficient, volume=volume)
        if kind == 'current' and native in ('raw', 'cb_over_dv'):
            axis, edge = seen['soft_edges'][sum(c['kind'] == 'soft' for c in seen['calls'])]
            seen['calls'].append(dict(kind='soft', axis=axis, edge=edge,
                                      value=float(np.asarray(out)[0]) * float(volume),
                                      multiplier=1.))
        return out

    def invariants(ctx, **kwargs):
        seen.update(materials=ctx.materials, dt=float(ctx.dt))
        return original_invariants(ctx, **kwargs)

    def kernel_materials(materials, **kwargs):
        if lane == 'graded':
            seen['materials'] = materials
        return original_kernel_materials(materials, **kwargs)

    def unit_curl(state, axis, edge):
        zeros = jnp.zeros_like(state.ez)
        curls = tuple(zeros.at[edge].set(1.) if a == axis else zeros for a in range(3))
        return state._replace(ex=zeros, ey=zeros, ez=zeros), curls

    def judged():
        call = _judge(seen, kind, branch, change)
        seen['call'] = call
        return call['axis'], call['edge']

    def kernel(state, materials, dt, dx, **kwargs):
        axis, edge = judged()
        state, curls = unit_curl(state, axis, edge)
        with pytest.MonkeyPatch.context() as curl_patch:
            curl_patch.setattr(kernels, 'curl_h', lambda *a, **k: curls)
            result = original_kernel(state, materials, dt, dx, **kwargs)[0]
        seen['increment'] = float(getattr(result, AXES[axis])[edge])
        raise _O2ReferenceCaptured

    def upml_e(state, coeffs, **kwargs):
        seen['upml_coeffs'] = coeffs
        axis, edge = judged()
        state, curls = unit_curl(state, axis, edge)
        with pytest.MonkeyPatch.context() as curl_patch:
            curl_patch.setattr(upml_module, 'curl_h', lambda *a, **k: curls)
            result = original_upml_e(state, coeffs, **kwargs)
        seen['increment'] = float(getattr(result, AXES[axis])[edge])
        raise _O2ReferenceCaptured

    def graded_kernel(state, materials, eps_ex, eps_ey, eps_ez, dt, *args, **kwargs):
        seen['dt'] = float(dt)
        seen['kernel_eps'] = (eps_ex, eps_ey, eps_ez)
        axis, edge = judged()
        state, curls = unit_curl(state, axis, edge)
        with pytest.MonkeyPatch.context() as curl_patch:
            curl_patch.setattr(kernels, 'curl_h_nu', lambda *a, **k: curls)
            result = original_graded_kernel(state, materials, eps_ex, eps_ey, eps_ez, dt,
                                            *args, **kwargs)
        seen['increment'] = float(getattr(result, AXES[axis])[edge])
        raise _O2ReferenceCaptured

    with pytest.MonkeyPatch.context() as patch, jax.disable_jit():
        _scale_on_attach(patch, branch, change, seen)
        patch.setattr(port_drive, 'port_drive_waveform', drive)
        patch.setattr(nu_runner, 'port_drive_waveform', drive)
        patch.setattr(lumped, 'edge_update_denominator', denominator)
        patch.setattr(source_coefficients, 'source_table', table)
        patch.setattr(model_package, 'source_table', table)
        patch.setattr(source_coefficients, 'uniform_source_table', uniform_table)
        patch.setattr(nu_runner, 'make_current_source', graded_source)
        patch.setattr(uniform_setup, 'core_step_invariants', invariants)
        patch.setattr(uniform_loop, 'core_step_invariants', invariants)
        patch.setattr(uniform_loop, '_update_e_with_optional_dispersion', kernel)
        patch.setattr(kernels, 'update_e_nu_aniso', graded_kernel)
        patch.setattr(model, 'kernel_materials', kernel_materials)
        patch.setattr(upml_module, 'apply_upml_e', upml_e)
        with pytest.raises(_O2ReferenceCaptured):
            getattr(sim, entry)(**_entry_options(branch, entry, extra))
    assert ('kernel_eps' in seen) == (lane == 'graded'), (lane, sorted(seen))
    call = seen['call']
    axis, edge = call['axis'], call['edge']
    cb = _step_cb(seen, branch, axis, edge)
    ref = _reference(seen, branch, call)
    c = seen['materials'].components
    return dict(kind=kind, branch=branch, entry=entry, axis=axis, edge=edge,
                sigma_mean=float(np.asarray(c.sigma_update[axis], np.float64)[edge]),
                sigma_owner=(float(np.asarray(c.upml_sigma[axis], np.float64)[edge])
                             if _upml(branch) else None),
                value=call['value'], ref=ref, rel=abs(call['value'] - ref) / abs(ref),
                increment=seen['increment'], step_cb=cb,
                operand=float(_operand(seen['materials'].components, branch, axis)[edge]),
                plain=float(np.asarray(seen['materials'].components.eps_update[axis],
                                       np.float64)[edge]),
                scaled=seen.get('scaled', False))


def _judge_row(make_scene, entry, branch, kind, *, lane='uniform', extra=None):
    """Value, kernel self-check and follows step of one row (relative 1e-6 each)."""
    row = _measure(make_scene, entry, branch, kind, lane=lane, extra=extra)
    np.testing.assert_allclose(row['increment'], row['step_cb'], rtol=_O2_RTOL, atol=0.,
                               err_msg=f'kernel self-check {row}')
    np.testing.assert_allclose(row['value'], row['ref'], rtol=_O2_RTOL, atol=0.,
                               err_msg=str(row))
    follow = _measure(make_scene, entry, branch, kind, lane=lane, extra=extra,
                      change=(row['axis'], row['edge']))
    assert follow['scaled'], ('the operand was never attached', follow)
    np.testing.assert_allclose(follow['operand'], 1.7 * row['operand'], rtol=_O2_RTOL)
    assert follow['plain'] == row['plain']
    assert follow['ref'] != row['ref']
    np.testing.assert_allclose(follow['increment'], follow['step_cb'], rtol=_O2_RTOL, atol=0.,
                               err_msg=f'kernel self-check {follow}')
    np.testing.assert_allclose(follow['value'], follow['ref'], rtol=_O2_RTOL, atol=0.,
                               err_msg=str(follow))
    print(f'drives-after-tensor row {lane} {entry} {kind}-{branch}: {row}; follows {follow}',
          file=sys.stderr)
    return row



@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('branch', ('smooth', 'full', 'kottke'))
def test_soft_current_source_reads_the_update_operand(entry, branch):
    """Soft current source Cb on O2's judged lumped edge, uniform lane.

    Quantity: the Cb in the table ``(1/dV)*(Cb*w)`` the builder stored (unit w).
    Formula: ``Cb = (dt/(eps0*e))/(1 + s*dt/(2*eps0*e))`` with ``e = tensor.eps``
    (smooth, full), or ``(dt*u/eps0)/(1 + s*dt*u/(2*eps0))`` with
    ``u = tensor.inv_eps`` (Kottke), ``s = sigma_update``, all read at
    ``core_step_invariants``. Equal on both sides: scene, edge, component, dt,
    entry. Sign: Cb > 0, the table is added to E.
    """
    _judge_row(lambda: _scene('soft', branch), entry, branch, 'soft')


@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_soft_current_source_on_a_conformal_pec_edge_is_refused(entry):
    """O2's conformal judged edge is a PEC edge: a current source there is refused
    before any step (existing text), so this branch is a refusal row."""
    from rfx.errors import RfxRefusal
    sim = _scene('soft', 'conformal')
    with pytest.raises(RfxRefusal, match=r"Soft source _ports\[0\] at .* is dead: its own E edge is PEC"):
        getattr(sim, entry)(**_entry_options('conformal', entry))



@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('operator,kind', (('lumped', 'port'), ('wire', 'port'),
                                           ('msl', 'port'), ('rlc', 'rlc')))
def test_kottke_drives_read_the_inverse_operand(entry, operator, kind):
    """Lumped, wire and MSL drive Cb and series RLC D0 on the Stage-2 inverse.

    Scene: O2's smooth scene with ``subpixel_smoothing='kottke_pec'``. Edge: the
    first builder edge where ``1/u`` differs from the plain edge mean.
    Formula: ``Cb = (dt*u/eps0)/(1 + s*dt*u/(2*eps0))`` times the port's
    ``sigma_port*unit_field``; ``D0 = eps0/(dt*u) + s/2``; ``u = tensor.inv_eps``,
    ``s = sigma_update`` at ``core_step_invariants``. The kernel self-check is
    ``update_e_aniso_inv``'s increment for a unit curl. Sign: Cb, D0 > 0.
    """
    _judge_row(lambda: _scene(operator, 'kottke'), entry, 'kottke', kind)



@pytest.mark.parametrize('branch', ('upml', 'upml_plain'),
                         ids=('smoothing_on', 'smoothing_off'))
@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('operator,kind', (('lumped', 'port'), ('soft', 'soft'), ('rlc', 'rlc')))
def test_upml_drives_read_the_upml_coefficient(entry, operator, kind, branch):
    """Lumped Cb, soft Cb and RLC D0 under UPML, smoothing on and off.

    Smoothing on (``subpixel_smoothing=True``): ``init_upml`` forms Cb from the
    attached tensor's float32 diagonal. Smoothing off: from the cell-owned
    ``components.upml_eps``, not the four-cell edge mean; on O2's judged edge
    ez(6, 6, 6) the owner cell is eps 4 and the mean 1.75. Both arms use the
    cell-owned conductivity ``upml_sigma``.
    Reference: ``ctx.upml_coeffs.cb_<c>[edge]`` (formed once by ``init_upml``)
    and ``1/that`` for D0. Kernel self-check: ``apply_upml_e``'s increment for a
    unit curl. Follows: the operand (``components.tensor.eps`` at the attach, or
    ``components.upml_eps`` where the components form) scaled 1.7x at the
    judged edge, before ``init_upml``. Equal on both sides: scene (O2's smooth
    scene with ``boundary='upml'``), interior edge, dt. Sign: Cb, D0 > 0.
    """
    _judge_row(lambda: _scene(operator, branch, upml=True), entry, branch, kind)


@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_upml_rlc_reads_the_cell_owned_conductivity(entry):
    """RLC D0 under UPML with smoothing on, on a lossy edge.

    O2's sphere carries 0.5 S/m, so at the judged edge ez(6, 6, 6) the owner
    cell's ``upml_sigma`` is 0.5 S/m and the four-cell mean ``sigma_update``
    0.125 S/m (one of the four cells is in the sphere). D0 must be ``1/Cb`` of
    ``init_upml``, which reads the owner cell's conductivity: the edge mean
    would move D0 by 0.1875 S/m, about 1e-2 of D0.
    """
    row = _judge_row(lambda: _scene('rlc', 'upml', upml=True, sigma=.5), entry, 'upml', 'rlc')
    assert row['edge'] == (6, 6, 6), row
    np.testing.assert_allclose(row['sigma_owner'], .5, rtol=1e-6)
    np.testing.assert_allclose(row['sigma_mean'], .125, rtol=1e-6)


def _upml_pad_rlc_scene(side):
    """O2's plain UPML rlc scene with the element moved onto an x pad edge.

    Returns the scene and the edge the element must realize: one cell inside
    the x-lo pad at x = -1 mm, or the first x-hi pad cell, whose position is
    computed from the grid's shape and pad counts.
    """
    sim = _scene('rlc', 'upml_plain', upml=True)
    spec = sim._lumped_rlc.pop()
    grid = sim._build_grid()
    _, j, k = (round(v / grid.dx) + pad for v, pad in
               zip(spec.position, (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)))
    if side == 'x_lo':
        i = grid.pad_x_lo - 1
    else:
        i = grid.shape[0] - grid.pad_x_hi
    x = (i - grid.pad_x_lo) * grid.dx
    sim.add_lumped_rlc((x, spec.position[1], spec.position[2]), spec.component,
                       R=spec.R, L=spec.L, C=spec.C, topology=spec.topology)
    return sim, (i, j, k)


@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('side', ('x_lo', 'x_hi'))
def test_upml_rlc_on_an_absorber_pad_edge_is_refused(entry, side):
    """A lumped RLC element on a UPML absorber pad edge is refused before any step.

    On a pad edge the UPML step's Cb includes the perpendicular absorber
    conductivity, while the element's D0 is the interior 1/Cb. Scene: O2's
    smooth scene with ``boundary='upml'``, no smoothing, the series element
    ez moved to x = -1 mm (edge (1, 6, 6), inside the two-cell x-lo pad) or
    to the first x-hi pad cell. The interior element stays admitted (the
    ``smoothing_off`` rows of ``test_upml_drives_read_the_upml_coefficient``).
    """
    from rfx.errors import RfxRefusal
    sim, edge = _upml_pad_rlc_scene(side)
    if side == 'x_lo':
        assert edge == (1, 6, 6), edge
    with pytest.raises(RfxRefusal, match=(
            re.escape(f'realizes edge {edge}, which lies inside a UPML absorber pad')
            + '.*' + re.escape('Place the element inside the physical domain, '
                               'or enlarge the domain.'))):
        getattr(sim, entry)(**_entry_options('upml_plain', entry))


@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('branch', ('smooth', 'full'))
@pytest.mark.parametrize('operator,kind', (('lumped', 'port'), ('wire', 'port'), ('msl', 'port'),
                                           ('rlc', 'rlc'), ('soft', 'soft')))
def test_graded_drives_read_the_update_operand(entry, branch, operator, kind):
    """Graded run/forward: drive Cb, soft Cb and RLC D0 against the graded kernel.

    Scene: O2's scene on a graded x profile. Operand: ``tensor.eps`` of the
    materials the graded setup hands to its kernel, ``sigma_update``, and the
    kernel's dt. Formula: ``Cb = (dt/(eps0*e))/(1 + s*dt/(2*eps0*e))`` (the
    soft source's host formula ``(dt/eps0)/(e + s*dt/(2*eps0))`` is the same
    quantity), ``D0 = eps0*e/dt + s/2``. Kernel self-check:
    ``update_e_nu_aniso``'s increment for a unit curl. Sign: Cb, D0 > 0.
    """
    _judge_row(lambda: _scene(operator, branch, graded=True), entry, branch, kind,
               lane='graded')


@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('branch,name', (('conformal', 'conformal'), ('kottke', 'kottke_pec')))
def test_graded_conformal_and_kottke_are_refused(entry, branch, name):
    """The graded lane refuses conformal PEC and the Stage-2 inverse before any
    drive is built; each is a refusal row."""
    from rfx.errors import RfxUnsupported
    sim = _scene('lumped', branch, graded=True)
    with pytest.raises((RfxUnsupported, ValueError, NotImplementedError), match=name):
        getattr(sim, entry)(**_entry_options(branch, entry))



_RADIUS_REFUSAL = (
    'wire-port radius material records are not implemented on uniform Yee with '
    'UPML/tensor or design-box updates; resolve the pin geometrically '
    '(for example a coax feed or a volume wire).')


@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_wire_radius_on_a_tensor_run_is_refused(entry):
    """The wire radius update reads no tensor: a radius on a smoothed run is refused."""
    import re
    sim = _scene('wire', 'smooth', radius=1e-4)
    with pytest.raises(NotImplementedError, match=re.escape(_RADIUS_REFUSAL)):
        getattr(sim, entry)(**_entry_options('smooth', entry))


def _occupancy_scene():
    sim = rfx.Simulation(freq_max=10e9, domain=(.012,) * 3, dx=.001, boundary='pec',
                         pec_occupancy_model='kottke')
    sim.add_source((.006, .006, .006), 'ez', waveform=jnp.ones_like, amplitude_kind='current')
    return sim


def test_occupancy_lane_soft_source_reads_the_occupancy_inverse():
    """forward() with ``pec_occupancy_model='kottke'`` and a partial occupancy
    (M = 0.5) on the four cells around a soft current source's ez edge.

    Formula: ``Cb = (dt*u/eps0)/(1 + s*dt*u/(2*eps0))`` with ``u`` the run's
    inverse (``aniso_inv_eps_run``, attached as ``tensor.inv_eps`` with the
    smooth flag). Lumped ports on this lane are not reachable: the port
    occupancy release keeps their edges free of design metal. Sign: Cb > 0.
    """
    grid = _occupancy_scene()._build_grid()
    occupancy = jnp.zeros(grid.shape, jnp.float32).at[5:7, 5:7, 5:7].set(.5)
    _judge_row(_occupancy_scene, 'forward', 'occupancy', 'soft',
               extra=dict(pec_occupancy_override=occupancy))



_FROZEN = (r"{entity} at \(.*0\.006.*\) \(ez\), grid index \(6, 6, 6\): "
           r"the smoothed inverse permittivity freezes this E edge \(inv < 1e-9\), so the "
           r"drive would inject nothing\. Move it off the metal, or cut a gap in the conductor\.")
_DEAD_SOFT = r"Soft source _ports\[0\] at .* is dead: its own E edge is PEC"


@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('kind', ('lumped', 'soft'))
def test_drive_on_a_kottke_frozen_edge_is_refused(entry, kind):
    """A drive whose edge the concrete Stage-2 inverse freezes (inv < 1e-9) is
    refused before the step: after the change its inverse-form Cb is 0, so it
    would inject nothing. A soft source inside metal may meet the dead-source
    refusal first; its case accepts whichever text fires first.
    """
    from rfx import Box
    from rfx.errors import RfxRefusal
    sim = rfx.Simulation(freq_max=10e9, domain=(.012,) * 3, dx=.001, boundary='pec')
    sim.add(Box((.003, .003, .003), (.009, .009, .009)), material='pec')
    if kind == 'lumped':
        sim.add_port((.006, .006, .006), 'ez', impedance=50., waveform=jnp.ones_like)
        pattern = _FROZEN.format(entity=r"Lumped port _ports\[0\]")
    else:
        sim.add_source((.006, .006, .006), 'ez', waveform=jnp.ones_like,
                       amplitude_kind='current')
        pattern = '(' + _FROZEN.format(entity=r"Soft source _ports\[0\]") + ')|(' + _DEAD_SOFT + ')'
    with pytest.raises(RfxRefusal, match=pattern):
        getattr(sim, entry)(**_entry_options('kottke', entry))



def _b2_scene(eps_r, *, precision=None):
    """B2's closed box: 12 mm PEC box, dx 1 mm, an eps_r slab from x = 6.3 mm."""
    from rfx import Box
    kwargs = dict(freq_max=10e9, domain=(.012,) * 3, dx=.001, boundary='pec', stencil_order=2)
    if precision is not None:
        kwargs['precision'] = precision
    sim = rfx.Simulation(**kwargs)
    sim.add_material('slab', eps_r=eps_r)
    sim.add(Box((.0063, 0., 0.), (.012, .012, .012)), material='slab')
    return sim


def _b2_wave():
    from rfx import GaussianPulse
    return GaussianPulse(f0=5e9, bandwidth=.8)


def _b2_declare(sim, kind, edge_mm, source_mm=None):
    position = tuple(v * .001 for v in edge_mm)
    if kind == 'lumped':
        sim.add_port(position, 'ez', impedance=50., waveform=_b2_wave())
    elif kind == 'soft':
        sim.add_source(position, 'ez', waveform=_b2_wave(), amplitude_kind='current')
    else:
        sim.add_lumped_rlc(position, 'ez', R=50., L=1e-9, topology='series')
        sim.add_source(tuple(v * .001 for v in source_mm), 'ez', waveform=_b2_wave(),
                       amplitude_kind='current')


def _inverse_denominator(inv, sigma, dt):
    return EPS_0 / (dt * inv) + sigma / 2.


def _inverse_denominator_eps_r(inv, sigma, dt):
    return (EPS_0 / dt) / inv + sigma / 2.


def _denominator(eps, sigma, dt):
    return eps * EPS_0 / dt + sigma / 2.


def _denominator_eps_r(eps, sigma, dt):
    return eps * (EPS_0 / dt) + sigma / 2.


@pytest.mark.parametrize('branch', ('smooth', 'kottke'))
@pytest.mark.parametrize('kind', ('lumped', 'soft', 'rlc'))
def test_drive_gradient_follows_the_step_operand(branch, kind):
    """d(drive Cb or element D0)/d(eps_r) equals d(step Cb or D0)/d(eps_r) at ez(6,6,6) mm.

    Scene: B2's box with ``forward()``, 2 steps, float32; the slab's eps_r is
    ``jnp.float32(4.0)``, traced. Drive side: the tracer the builder's reader
    returned (``e_update_coefficient_at`` for Cb, ``edge_update_denominator``
    for D0). Step side: the operand in ``ctx.materials`` at
    ``core_step_invariants`` through the kernel's coefficient function
    (``e_update_coeffs``; Stage-2 ``si_value_eps_r_grad(_e_update_coeffs_inv_si,
    _ca_cb_inv_eps_r_units, ...)``; D0 ``eps0*e/dt + s/2`` or ``eps0/(dt*u) + s/2``
    in the ``si_value_eps_r_grad`` spelling). Both derivatives by ``jax.jacrev``
    of one function returning both captures; the forward runs to completion.
    Premise: |d(step operand)/d(eps_r)| > 0.05 (``tensor.eps``, or ``1/u``).
    Bar: relative 1e-6. Not an O2-form row: no follows step.
    """
    from rfx.core.yee import (e_update_coeffs, si_value_eps_r_grad,
                              _e_update_coeffs_inv_si, _ca_cb_inv_eps_r_units)
    from rfx.model import materials as model
    from rfx.sources import port_drive
    from rfx.stepping import uniform_setup, uniform_loop
    import rfx.lumped as lumped

    edge = (6, 6, 6)
    original_port = port_drive.e_update_coefficient_at
    original_soft = model.e_update_coefficient_at
    original_denominator = lumped.edge_update_denominator
    original_invariants = uniform_setup.core_step_invariants

    def objective(eps_r):
        captured = {}

        def port_coefficient(materials, cell, component, dt, *args, **kwargs):
            value = original_port(materials, cell, component, dt, *args, **kwargs)
            if _edge(cell) == edge and component == 'ez':
                captured['drive'] = value
            return value

        def soft_coefficient(materials, cell, component, dt, *args, **kwargs):
            value = original_soft(materials, cell, component, dt, *args, **kwargs)
            if _edge(cell) == edge and component == 'ez':
                captured['drive'] = value
            return value

        def denominator(materials, cell, component, dt, *args, **kwargs):
            value = original_denominator(materials, cell, component, dt, *args, **kwargs)
            if _edge(cell) == edge and component == 'ez':
                captured['drive'] = value
            return value

        def invariants(ctx, **kwargs):
            c = ctx.materials.components
            sigma = c.sigma_update[2][edge]
            if branch == 'kottke':
                u = c.tensor.inv_eps[2][edge]
                captured['operand'] = 1. / u
                if kind == 'rlc':
                    captured['step'] = si_value_eps_r_grad(
                        _inverse_denominator, _inverse_denominator_eps_r, u, sigma, ctx.dt)
                else:
                    captured['step'] = si_value_eps_r_grad(
                        _e_update_coeffs_inv_si, _ca_cb_inv_eps_r_units, u, sigma, ctx.dt)[1]
            else:
                e = c.tensor.eps[2][edge]
                captured['operand'] = e
                captured['step'] = (si_value_eps_r_grad(_denominator, _denominator_eps_r,
                                                        e, sigma, ctx.dt)
                                    if kind == 'rlc' else e_update_coeffs(e, sigma, ctx.dt)[1])
            return original_invariants(ctx, **kwargs)

        sim = _b2_scene(eps_r)
        _b2_declare(sim, kind, edge, source_mm=(3, 6, 6))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(port_drive, 'e_update_coefficient_at', port_coefficient)
            patch.setattr(model, 'e_update_coefficient_at', soft_coefficient)
            patch.setattr(lumped, 'edge_update_denominator', denominator)
            patch.setattr(uniform_setup, 'core_step_invariants', invariants)
            patch.setattr(uniform_loop, 'core_step_invariants', invariants)
            sim.forward(n_steps=2, checkpoint=False, skip_preflight=True,
                        subpixel_smoothing='kottke_pec' if branch == 'kottke' else True)
        return captured['drive'], captured['step'], captured['operand']

    d_drive, d_step, d_operand = jax.jacrev(objective)(jnp.float32(4.0))
    d_drive, d_step, d_operand = (float(d_drive), float(d_step), float(d_operand))
    print(f'gradient row {kind}-{branch}: d_drive={d_drive!r} d_step={d_step!r} '
          f'd_operand={d_operand!r}', file=sys.stderr)
    assert abs(d_operand) > .05, ('premise: the step operand does not depend on eps_r', d_operand)
    assert abs(d_step) > 0.
    np.testing.assert_allclose(d_drive, d_step, rtol=_O2_RTOL, atol=0.)



def _pulse_numpy(t):
    """The declared GaussianPulse(f0=5 GHz, bandwidth=0.8), evaluated without rfx."""
    tau = 1.0 / (5e9 * .8 * np.pi)
    c = 3.0
    u = t / tau
    x = (t - c * tau) / tau
    w = (-2.0 * x * np.exp(-x * x)
         - np.exp(-c * c - u * u) * (2 * c + (4 * c * c - 2) * u - 4 * c * u * u
                                     - 4 * c * c * u * u * u))
    return np.where(t >= 0, w, 0.0)


_B2_EDGES = {'drive': (6, 6, 6), 'control': (3, 6, 6)}
_B2_RLC_SOURCE = {'drive': (3, 6, 6), 'control': (3, 9, 6)}


def _b2_record(kind, edge_name):
    """One B2 run: record E, H (and the RLC current) after every step and the
    realized coefficients where the kernel forms them; return the balance."""
    import rfx.boundaries.pmc as pmc
    import rfx.stepping.uniform_loop as loop
    import rfx.stepping.uniform_setup as setup
    import rfx.stepping.yee as yee

    sim = _b2_scene(4., precision='float64')
    edge = _B2_EDGES[edge_name]
    source_edge = _B2_RLC_SOURCE[edge_name] if kind == 'rlc' else edge
    _b2_declare(sim, kind, edge, source_mm=source_edge)

    def host(x):
        return np.array(x, dtype=np.float64, copy=True)

    saved = {'E': {}, 'H': {}, 'I': {}, 'pairs': {}, 'pair_counts': [0, 0, 0], 'ch_count': 0}
    original_invariants, original_factory = setup.core_step_invariants, loop.make_core_step
    original_e, original_h = yee.e_update_coeffs, pmc.magnetic_update_coefficients
    traced_axis = [0]

    def invariants(ctx, *args, **kwargs):
        assert 'ctx' not in saved, 'more than one physical record in this run'
        saved['ctx'] = ctx
        assert ctx.stencil_order == 2 and not ctx.use_cpml and not ctx.use_upml
        c = ctx.materials.components
        assert c.tensor is not None and c.tensor.eps is not None
        assert c.tensor.node_smoothing is None
        saved['eps_B'] = np.stack([host(a) for a in c.tensor.eps])
        saved['sigma'] = np.stack([host(a) for a in c.sigma_update])
        premise = {}
        for label, cell in _B2_EDGES.items():
            b = np.asarray(c.tensor.eps[2])[cell]
            d = np.asarray(c.eps_update[2])[cell]
            s = np.asarray(c.sigma_update[2])[cell]
            premise[label] = dict(eps_B=float(b), eps_D=float(d), sigma=float(s),
                                  eps_B_dtype=str(b.dtype), eps_D_dtype=str(d.dtype),
                                  relative_difference=float(abs(b - d) / d),
                                  equal_after_float64=bool(np.float64(b) == np.float64(d)))
        print(f'B2 premise {kind}-{edge_name}: {premise}', file=sys.stderr)
        assert premise['drive']['relative_difference'] > .05, ('premise at the drive edge', premise)
        assert premise['control']['equal_after_float64'], ('premise at the control edge', premise)
        saved['premise'] = premise
        return original_invariants(ctx, *args, **kwargs)

    def save_pair(axis, ca, cb, eps, sigma):
        values = tuple(host(x) for x in (ca, cb, eps, sigma))
        if axis in saved['pairs']:
            for before, after in zip(saved['pairs'][axis], values):
                np.testing.assert_array_equal(before, after)
        else:
            saved['pairs'][axis] = values
        saved['pair_counts'][axis] += 1

    def electric(eps, sigma, dt):
        axis = traced_axis[0] % 3
        traced_axis[0] += 1
        ca, cb = original_e(eps, sigma, dt)
        jax.debug.callback(lambda *v: save_pair(axis, *v), ca, cb, eps, sigma, ordered=True)
        return ca, cb

    def save_ch(*values):
        ch = np.stack([host(x) for x in values])
        if 'ch' in saved:
            np.testing.assert_array_equal(saved['ch'], ch)
        else:
            saved['ch'] = ch
        saved['ch_count'] += 1

    def magnetic(*args, **kwargs):
        ch = original_h(*args, **kwargs)
        jax.debug.callback(save_ch, *ch, ordered=True)
        return ch

    def save_fields(n, fields, currents, initial):
        n = int(n)
        if initial and n != 0:
            return
        assert n not in saved['E'], ('duplicate carry', n)
        assert all(np.asarray(a).dtype == np.float64 for a in fields)
        saved['E'][n] = np.stack([host(x) for x in fields[:3]])
        saved['H'][n] = np.stack([host(x) for x in fields[3:]])
        if currents:
            saved['I'][n] = float(currents[0])

    def record(carry, n, initial=False):
        state = carry['fdtd']
        fields = tuple(getattr(state, c) for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))
        currents = tuple(x.inductor_current for x in carry.get('rlc_states', ()))
        jax.debug.callback(lambda *args: save_fields(*args, initial), n, fields, currents,
                           ordered=True)

    def factory(*args, **kwargs):
        step = original_factory(*args, **kwargs)

        def wrapped(carry, n, *values):
            record(carry, n, initial=True)
            result = step(carry, n, *values)
            record(result[0], n + 1)
            return result
        return wrapped

    jax.clear_caches()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(setup, 'core_step_invariants', invariants)
        patch.setattr(loop, 'core_step_invariants', invariants)
        patch.setattr(loop, 'make_core_step', factory)
        patch.setattr(yee, 'e_update_coeffs', electric)
        patch.setattr(pmc, 'magnetic_update_coefficients', magnetic)
        sim.run(n_steps=400, subpixel_smoothing=True, skip_preflight=True)
        jax.effects_barrier()

    assert sorted(saved['E']) == list(range(401))
    assert saved['pair_counts'] == [400] * 3 and saved['ch_count'] == 400
    ctx = saved['ctx']
    dt, dx = float(ctx.dt), float(ctx.dx)
    dv = dx ** 3
    ca = np.stack([saved['pairs'][a][0] for a in range(3)])
    cb = np.stack([saved['pairs'][a][1] for a in range(3)])
    for a in range(3):
        np.testing.assert_array_equal(saved['pairs'][a][2], saved['eps_B'][a])
        np.testing.assert_array_equal(saved['pairs'][a][3], saved['sigma'][a])
    eps_eff = dt * (1 + ca) / (2 * cb)
    sigma_eff = (1 - ca) / cb
    ch = saved['ch']
    E = np.stack([saved['E'][n] for n in range(401)])
    H = np.stack([saved['H'][n] for n in range(401)])
    active_h = ch != 0
    assert np.all(H[:, ~active_h] == 0)
    mu_eff = np.zeros_like(ch)
    np.divide(dt, ch, out=mu_eff, where=active_h)
    W = .5 * dv * (np.sum(eps_eff * E[:400] ** 2, axis=(1, 2, 3, 4))
                   + np.sum(mu_eff * H[:400] * H[1:401], axis=(1, 2, 3, 4)))
    ebar = .5 * (E[1:400] + E[:399])
    times = np.arange(399, dtype=np.float32).astype(np.float64) * dt
    w = _pulse_numpy(times)
    J = -(dx / (50. * dx * dx)) * (1. / dx) * w if kind == 'lumped' else -w / dv
    q = dt * J * ebar[(slice(None), 2, *source_edge)] * dv
    res = np.diff(W) + q + dt * dv * np.sum(sigma_eff * ebar ** 2, axis=(1, 2, 3, 4))
    out = dict(kind=kind, edge=edge_name, premise=saved['premise'][edge_name])
    if kind == 'rlc':
        I = np.array([saved['I'][n] for n in range(401)], dtype=np.float64)
        i_avg = .5 * (I[:399] + I[1:400])
        W_L = .5 * 1e-9 * I ** 2
        p = dt * dx * ebar[(slice(None), 2, *edge)] * i_avg
        res = res + np.diff(W_L[:400]) + dt * 50. * i_avg ** 2
        c_q, c_p = np.linalg.lstsq(np.column_stack((q, p)), res, rcond=None)[0]
        out.update(k_q=float(1 - c_q), k_p=float(1 - c_p),
                   leftover=float(np.sum(abs(res - c_q * q - c_p * p)) / np.sum(abs(p))))
    else:
        out.update(k_hat=float(1 - np.sum(res * q) / np.sum(q ** 2)),
                   ratio=float(np.sum(abs(res)) / np.sum(abs(q))))
    print(f'B2 {kind}-{edge_name}: {out}', file=sys.stderr)
    return out


@pytest.mark.slow_physics
def test_drive_energy_balance_in_a_closed_box():
    """B2: the discrete energy balance of a closed lossless PEC box (float64).

    Quantity per step n = 0..398: ``res_n = W^(n+1) - W^n + q_n + dt*sum(sigma_eff*Ebar^2)*dV``
    with ``W^n = 1/2 sum eps0*eps_eff*(E^n)^2 dV + 1/2 sum mu0*mu_eff*H^(n-1/2)*H^(n+1/2) dV``,
    ``Ebar = (E^(n+1)+E^n)/2``, ``eps0*eps_eff = dt*(1+Ca)/(2*Cb)``,
    ``sigma_eff = (1-Ca)/Cb``, ``mu0*mu_eff = dt/ch``, all from the coefficients
    the kernel forms; ``q_n = dt*J^n*Ebar*dV`` with J from the declaration
    (``J = -table/Cb``: lumped ``-sigma_port*unit_field*w``, soft ``-w/dV``,
    ``t_n = float32(n)*dt``). Fit: ``k_hat = 1 - sum(res*q)/sum(q^2)``.
    RLC (series 50 ohm + 1 nH): ``res_n`` adds ``W_L^(n+1) - W_L^n + dt*R*I_avg^2``,
    element work ``p_n = dt*d*Ebar_e*I_avg``, joint least squares
    ``res = c_q*q + c_p*p``, ``k_p = 1 - c_p``. Conditions equal on both edges:
    scene, waveform, steps, precision. Bars (after the change): ``|k_hat - 1|``
    and ``sum|res|/sum|q|`` at most 1e-6 on the control and drive edges; RLC
    ``|k_p - 1|`` and ``sum|res - fit|/sum|p|`` at most 1e-6, and ``k_q`` equal
    to the soft-source factor at the same source edge within 1e-6.

    Marked slow_physics: six 400-step float64 runs, each recompiled and
    copying every field to the host after every step, a physics anchor of
    the weekly CPU lane; the coefficient rows above are the always-on gate.

    x64 is switched process-wide for this test only (restored in ``finally``),
    not by the scoped ``enable_x64()`` context, because ``jax.debug.callback``
    runs the recording on another thread, where the scoped flag is off and the
    float64 fields arrive float32.
    """
    previous = bool(jax.config.read('jax_enable_x64'))
    jax.clear_caches()
    jax.config.update('jax_enable_x64', True)
    try:
        results = {(kind, name): _b2_record(kind, name)
                   for kind in ('lumped', 'soft', 'rlc') for name in ('control', 'drive')}
    finally:
        jax.config.update('jax_enable_x64', previous)
        jax.clear_caches()
    for name in ('control', 'drive'):
        for kind in ('lumped', 'soft'):
            row = results[kind, name]
            assert abs(row['k_hat'] - 1) <= 1e-6, row
            assert row['ratio'] <= 1e-6, row
        row = results['rlc', name]
        assert abs(row['k_p'] - 1) <= 1e-6, row
        assert row['leftover'] <= 1e-6, row
    assert abs(results['rlc', 'drive']['k_q'] - results['soft', 'control']['k_hat']) <= 1e-6
