"""Port Cb and RLC D0 consume the stored, stamped E-update operands."""

from rfx.geometry.rasterize_grid import unpadded_high_walls
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core import yee
from rfx.core.yee import EPS_0
from rfx.lumped import edge_update_denominator
from rfx.model.materials import e_update_material_at, with_components
from rfx.sources.port_drive import PortSourceQueue, port_drive_waveform
from types import SimpleNamespace


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("periodic", [(False, False, False), (True, True, True)])
def test_drive_and_denominator_match_stored_update(component, periodic):
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    transverse = (axis + 1) % 3
    shape = (4, 4, 4)
    cell = (0, 0, 0) if periodic[0] else (2, 2, 2)
    slab = (jnp.indices(shape)[transverse] >= 2)
    eps = jnp.where(slab, 4., 1.)
    sigma = jnp.where(slab, .2, .0)
    stamp = jnp.zeros(shape).at[cell].set(3.)
    parts = tuple(stamp if a == axis else None for a in range(3))
    raw = yee.MaterialArrays(eps + stamp, sigma + stamp, jnp.ones(shape),
                             eps_r_lumped=parts, sigma_lumped=parts)
    mats = with_components(raw, None, periodic=periodic, high_wall=(False, False, False))
    grid = SimpleNamespace(dt=1e-12)
    # Independent grid-wide E-update coefficients; never the cell helper.
    cb = yee.e_component_coeffs(mats, grid.dt, periodic, high_wall=unpadded_high_walls(grid))[1][axis][cell]
    wave = port_drive_waveform(grid, cell, component, jnp.ones_like, 3, mats,
                               sigma_port=2., unit_field=4.)
    np.testing.assert_array_equal(wave, jnp.full(3, cb * 2. * 4.))
    e, s = e_update_material_at(mats, cell, component, high_wall=unpadded_high_walls(grid))
    assert float(e) != float(raw.eps_r[cell])
    expected = float(e) * EPS_0 / grid.dt + float(s) / 2.
    assert edge_update_denominator(mats, cell, component, grid.dt, as_float=True) == expected
    # Future rules may differ from today's mean: only the stored operand moves.
    changed = list(mats.components.eps_update)
    changed[axis] = changed[axis].at[cell].multiply(1.7)
    changed = mats._replace(components=replace(mats.components, eps_update=tuple(changed)))
    changed_cb = yee.e_component_coeffs(changed, grid.dt, periodic, high_wall=unpadded_high_walls(grid))[1][axis][cell]
    changed_wave = port_drive_waveform(grid, cell, component, jnp.ones_like, 3, changed,
                                       sigma_port=2., unit_field=4.)
    np.testing.assert_array_equal(changed_wave, jnp.full(3, changed_cb * 2. * 4.))
    assert not np.array_equal(wave, changed_wave)
    assert edge_update_denominator(changed, cell, component, grid.dt) != edge_update_denominator(mats, cell, component, grid.dt)


def test_queue_preserves_source_order_and_supplies_final_materials():
    queue = PortSourceQueue()
    queue.append(("first",))
    seen = []

    def build(label, *, materials):
        seen.append(materials)
        return [(label,), ("second edge",)]

    queue.defer(build, "port")
    queue.append(("last",))
    assert not seen
    final = object()
    assert queue.resolve(final) == [("first",), ("port",), ("second edge",), ("last",)]
    assert seen == [final]


def test_realized_edge_material_gradient_matches_legacy():
    def loss(scale, realized):
        eps = jnp.ones((4, 4, 4)).at[2:].set(4. * scale)
        mats = yee.MaterialArrays(eps, jnp.zeros_like(eps), jnp.ones_like(eps))
        if realized:
            mats = with_components(mats, None, periodic=(False,) * 3, high_wall=(False, False, False))
        grid = SimpleNamespace(dt=1e-12)
        wave = port_drive_waveform(grid, (2, 2, 2), "ez",
                                  jnp.ones_like, 2, mats, sigma_port=2., unit_field=4.)
        # This raw-array fixture has no wall face.
        return jnp.sum(wave) + edge_update_denominator(mats, (2, 2, 2), "ez", 1e-12, grid=grid)

    a = jax.value_and_grad(lambda x: loss(x, False))(jnp.float32(1.))
    b = jax.value_and_grad(lambda x: loss(x, True))(jnp.float32(1.))
    np.testing.assert_allclose(a, b, rtol=1e-4)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("kind", ["lumped", "wire", "msl"])
def test_main_paths_build_drives_and_rlc_after_realization(monkeypatch, graded, forward, kind):
    from rfx import Box, Simulation
    from rfx.model import materials as model
    from rfx.sources import port_drive
    from tests.contracts.test_measurement_plan import setup
    import rfx.lumped as lumped
    import rfx.runners.nonuniform as nu_runner

    kwargs = dict(freq_max=8e9, domain=(.012, .011, .012), dx=.001,
                  boundary="cpml", cpml_layers=2)
    if graded:
        kwargs["dx_profile"] = np.array([.001]*4 + [.0009, .0011] + [.001]*6)
    sim = Simulation(**kwargs)
    sim.add_material("slab", eps_r=4.)
    sim.add(Box((.005, .002, .002), (.010, .009, .010)), material="slab")
    if kind == "msl":
        sim.add(Box(((2 - .35) * .001, (2 - .35) * .001, .003),
                    ((10 + .35) * .001, (9 + .35) * .001, .003)), material="pec")
        # #1512: the strip starts at the port's grid node. Drawn from 2 mm it left a 3 mm
        # open stub whose effective quarter wave (11.9 GHz) is within 1.5x of 0..8 GHz.
        from rfx.preflight.line_port_coverage import port_node_coordinate
        node = port_node_coordinate(sim, (.005, .005, .003))
        sim.add(Box((node, (4 - .35) * .001, .005),
                    ((10 + .35) * .001, (6 + .35) * .001, .005)), material="pec")
        sim.add_msl_port(position=(.005, .005, .003), width=.002, height=.002,
                         direction="+x", impedance=50., waveform=jnp.ones_like, eps_r_sub=4.)
    else:
        sim.add_port((.005, .005, .005), "ez", waveform=jnp.ones_like,
                     extent=.002 if kind == "wire" else None)
        if kind == "lumped":
            # forward() S from one record needs isolated drives: the second port is
            # passive there; run() keeps both driven (its S driver isolates them)
            sim.add_port((.005, .005, .005), "ez", impedance=75., waveform=jnp.ones_like,
                         excite=not forward)
    for j, topology in enumerate(("series", "parallel")):
        sim.add_lumped_rlc((.005, .006 + .001*j, .005), "ez",
                           R=70., L=2e-9, C=.3e-12, topology=topology)
    original_realize = model.with_components
    original_drive = port_drive.port_drive_waveform
    original_denominator = lumped.edge_update_denominator
    calls = {"realize": 0, "drive": 0, "rlc": 0}

    def realize(*args, **kwargs):
        mats = original_realize(*args, **kwargs)
        calls["realize"] += 1
        # Model a later rule change without changing cells or any consumer.
        c = mats.components
        return mats._replace(components=replace(c, eps_update=tuple(e * 1.3 for e in c.eps_update)))

    def drive(grid, cell, component, excitation, n_steps, materials, **kwargs):
        assert materials.components is not None
        calls["drive"] += 1
        value = original_drive(grid, cell, component, excitation, n_steps, materials, **kwargs)
        axis = {"ex": 0, "ey": 1, "ez": 2}[component]
        cb = yee.e_component_coeffs(materials, grid.dt, high_wall=unpadded_high_walls(grid))[1][axis][tuple(cell)]
        np.testing.assert_array_equal(value, jnp.full(n_steps, cb * kwargs["sigma_port"] * kwargs["unit_field"]))
        return value

    def denominator(materials, cell, component, dt, *args, **kwargs):
        assert materials.components is not None
        calls["rlc"] += 1
        value = original_denominator(materials, cell, component, dt, *args, **kwargs)
        axis = {"ex": 0, "ey": 1, "ez": 2}[component]
        eps = materials.components.eps_update[axis][tuple(cell)]
        sigma = materials.components.sigma_update[axis][tuple(cell)]
        if kwargs.get("as_float"):
            assert value == float(eps) * EPS_0 / dt + float(sigma) / 2.
        else:
            np.testing.assert_array_equal(value, eps * EPS_0 / dt + sigma / 2.)
        return value

    monkeypatch.setattr(model, "with_components", realize)
    monkeypatch.setattr(port_drive, "port_drive_waveform", drive)
    monkeypatch.setattr(nu_runner, "port_drive_waveform", drive)
    monkeypatch.setattr(lumped, "edge_update_denominator", denominator)
    setup(sim, forward=forward)
    assert calls["realize"] == 1
    assert calls["drive"] > 0
    assert calls["rlc"] == 2


import sys

# #1961: references use stored operands, independently of every consumer helper.
# The public smooth, full-tensor and conformal paths all step with tensor.eps;
# the inverse-tensor kernel is outside this table.
def _o2_operands(materials, branch, axis):
    c = materials.components
    eps = c.eps_update if branch == "plain" else c.tensor.eps
    return (np.asarray(eps[axis], dtype=np.float64),
            np.asarray(c.sigma_update[axis], dtype=np.float64))


def _o2_cb(materials, branch, axis, edge, dt):
    eps, sigma = _o2_operands(materials, branch, axis)
    e, s = eps[edge], sigma[edge]
    return float(dt / (EPS_0 * e) / (1. + s * dt / (2. * EPS_0 * e)))


class _O2ReferenceCaptured(Exception):
    pass


def _o2_reference_self_check(monkeypatch, branch):
    """Pre-row audit: the declared branch must supply the design's operand."""
    from rfx import Simulation, Sphere
    from rfx.model import materials as model
    from rfx.stepping import uniform_loop, yee as kernels
    from rfx.core import node_tensor

    records = []
    for entry in ('run', 'forward'):
        sim = Simulation(freq_max=20e9, domain=(.012,) * 3, dx=.001,
                         boundary=_o2_boundary(branch), cpml_layers=2, snap='declared')
        sim.add_material('dielectric', eps_r=4.)
        sphere = Sphere((.00537, .00613, .00573), .0023)
        sim.add(sphere, material='pec' if branch == 'conformal' else 'dielectric')
        sim.add_source((.002, .002, .002), 'ez', waveform=jnp.ones_like,
                       amplitude_kind='field')
        options = dict(n_steps=1, skip_preflight=True)
        if branch in ('smooth', 'full'):
            options['subpixel_smoothing'] = True if branch == 'smooth' else 'full_tensor'
        if branch == 'conformal':
            options['conformal_pec'] = True
        if entry == 'forward':
            options.pop('conformal_pec', None)
            options['checkpoint'] = False
        raw = []
        original_realize = model.with_components
        original_update = uniform_loop._update_e_with_optional_dispersion
        original_node = node_tensor.add_node_increment
        off_diagonal = []

        def realize(*args, **kwargs):
            value = original_realize(*args, **kwargs)
            raw.append(value)
            return value

        def node(updated, *args, **kwargs):
            result = original_node(updated, *args, **kwargs)
            off_diagonal.append(tuple(np.asarray(b) - np.asarray(a)
                                      for a, b in zip(updated, result)))
            return result

        def update(state, materials, dt, dx, **kwargs):
            tensor = materials.components.tensor
            supplied_inv = kwargs.get('aniso_inv_eps')
            assert supplied_inv is None
            effective, sigma = _o2_operands(materials, branch, 2)
            kernel = 'update_e' if branch == 'plain' else 'update_e_aniso'
            cells = np.asarray(raw[-1].eps_r)
            sigma = np.asarray(materials.components.sigma_update[2], dtype=np.float64)
            eligible = np.zeros(effective.shape, dtype=bool)
            eligible[3:-3, 3:-3, 3:-3] = True
            eligible &= (effective != cells) & (sigma == 0) & np.isfinite(effective)
            if branch == 'conformal':
                weights = np.asarray(tensor.conformal_weights[2])
                eligible &= (weights > 0) & (weights < 1)
            assert eligible.any(), (branch, 'no discriminating edge')
            score = np.where(eligible, np.abs(effective - cells), -1.)
            edge = tuple(int(i) for i in np.unravel_index(np.argmax(score), score.shape))
            zeros = jnp.zeros_like(state.ez)
            curls = (zeros, zeros, zeros.at[edge].set(1.))
            state = state._replace(ex=zeros, ey=zeros, ez=zeros)
            with monkeypatch.context() as curl_patch:
                curl_patch.setattr(kernels, 'curl_h', lambda *a, **k: curls)
                result = original_update(state, materials, dt, dx, **kwargs)[0]
            eps = effective[edge]
            cb = _o2_cb(materials, branch, 2, edge, dt)
            value = float(result.ez[edge])
            off = []
            for arrays in off_diagonal:
                for component, array in zip(('ex', 'ey', 'ez'), arrays):
                    off.extend([component, [int(i) for i in index], float(array[tuple(index)])]
                               for index in np.argwhere(array != 0))
            record = dict(branch=branch, entry=entry, edge=edge, kernel=kernel,
                          eps=float(eps), cell=float(cells[edge]), increment=value,
                          cb_from_actual_operand=cb, rel=abs(value-cb)/abs(cb),
                          tensor_inv_present=tensor is not None and tensor.inv_eps is not None,
                          tensor_eps_present=tensor is not None and tensor.eps is not None,
                          off_diagonal_at_edge=(float(off_diagonal[0][2][edge]) if off_diagonal else 0.),
                          coupled=off)
            records.append(record)
            np.testing.assert_allclose(value, cb, rtol=1e-6, atol=0.)
            raise _O2ReferenceCaptured

        with monkeypatch.context() as patch, jax.disable_jit():
            patch.setattr(model, 'with_components', realize)
            patch.setattr(uniform_loop, '_update_e_with_optional_dispersion', update)
            patch.setattr(node_tensor, 'add_node_increment', node)
            with pytest.raises(_O2ReferenceCaptured):
                getattr(sim, entry)(**options)

    assert len(records) == 2


@pytest.mark.parametrize("branch", ("plain", "smooth", "full", "conformal"))
def test_o2_reference(monkeypatch, branch):
    _o2_reference_self_check(monkeypatch, branch)


_O2_BRANCHES = ('plain', 'smooth', 'full', 'conformal')
_O2_OPERATORS = ('lumped', 'wire', 'msl', 'rlc', 'waveguide', 'sheet', 'kerr')
_O2_ROWS = tuple((x, b) for x in _O2_OPERATORS for b in _O2_BRANCHES) + (
    ('waveguide_second_medium', 'plain'),)
_O2_RTOL = 1e-6

# Values agree only because section admission enforces a scalar medium.
NOTE = {
    ('waveguide', b): (
        f'{b}: step 1 omits the unequal-cell precondition because the section must '
        'be homogeneous. cfg.medium.eps_r stays fixed when only the update operand '
        'is scaled 1.7x; coefficient stays 215.330292, reference becomes 126.664876.')
    for b in ('plain', 'smooth', 'full')
}
_SHEET_REFUSAL = (
    'surface-impedance (surface_impedance_f0) sheets are not supported with '
    'subpixel_smoothing / conformal_pec: the sheet operator assumes the plain '
    'isotropic E update at its edges (Stage-1/2 conformal and Kottke lanes do not '
    'apply the sheet operator).')
_KERR_REFUSAL = (
    'subpixel_smoothing cannot combine with kerr_chi3: the Kerr E update does not '
    'read the permittivity tensor.')
REFUSED = {
    **{('sheet', b): _SHEET_REFUSAL for b in ('smooth', 'full', 'conformal')},
    **{('kerr', b): _KERR_REFUSAL for b in ('smooth', 'full')},
    ('waveguide_second_medium', 'plain'): (
        "Waveguide port 'waveguide_0': section cells ((5, 8), (0, 8), (0, 7)): "
        'permittivity (materials.eps_r)=np.float32(4.0) at cell (5, 3, 2); '
        'waveguide sections require one homogeneous, isotropic, lossless, '
        'non-dispersive medium'),
}
# The corrected waveguide aperture includes v_hi-1: its weight is 0.6.
# Every remaining combination either reaches an eligible edge or refuses by name.
UNREACHABLE = {}

# Measured |observed-reference|/|reference|, identical on run and forward.
_O2_XFAIL_REL = {
    ('lumped', 'smooth'): .192144888, ('lumped', 'full'): .174978157,
    ('lumped', 'conformal'): 84.8711297,
    ('wire', 'smooth'): .123831689, ('wire', 'full'): .112768240,
    ('wire', 'conformal'): 50.4323076,
    ('msl', 'smooth'): .258927820, ('msl', 'full'): .000376344724,
    ('msl', 'conformal'): 2.26802405,
    ('rlc', 'smooth'): .75, ('rlc', 'full'): .640101789,
    ('rlc', 'conformal'): .996277330,
    ('waveguide', 'conformal'): .666666605,
    ('sheet', 'plain'): .562499985,
    ('kerr', 'plain'): 1.285714286, ('kerr', 'conformal'): .0692840508,
}


# Edge and operand attribution from dump 457262ef53, measured on run and forward.
# component, index, operator epsilon, update epsilon, sigma, applied w, length cut.
_O2_XFAIL_AT_EDGE = {
    ('lumped', 'smooth'): ('ez', (6, 6, 6), 1.75, 1.0, 20.0, None, None),
    ('lumped', 'full'): ('ez', (6, 6, 6), 1.75, 1.0670069456100464, 20.0, None, None),
    ('lumped', 'conformal'): ('ez', (7, 6, 9), 1.0, 268.6243896484375, 20.0, 0.0037226700223982334, True),
    ('wire', 'smooth'): ('ez', (6, 6, 6), 1.75, 1.0, 40.0, None, None),
    ('wire', 'full'): ('ez', (6, 6, 6), 1.75, 1.0670069456100464, 40.0, None, None),
    ('wire', 'conformal'): ('ez', (7, 6, 9), 1.0, 268.6243896484375, 40.0, 0.0037226700223982334, True),
    ('msl', 'smooth'): ('ez', (6, 6, 6), 1.75, 1.0, 10.649312019348145, None, None),
    ('msl', 'full'): ('ez', (6, 6, 5), 1.0, 1.000807762145996, 10.649312019348145, None, None),
    ('msl', 'conformal'): ('ez', (7, 4, 8), 1.0, 5.8684492111206055, 10.649312019348145, 0.17040276527404785, True),
    ('rlc', 'smooth'): ('ez', (6, 6, 6), 1.75, 1.0, 0.0, None, None),
    ('rlc', 'full'): ('ez', (6, 6, 6), 1.75, 1.0670069456100464, 0.0, None, None),
    ('rlc', 'conformal'): ('ez', (7, 6, 9), 1.0, 268.6243896484375, 0.0, 0.0037226700223982334, True),
    ('waveguide', 'conformal'): ('ez', (6, 0, 6), 1.0, 1.6666666269302368, 0.0, 0.6000000238418579, True),
    ('sheet', 'plain'): ('ez', (6, 6, 6), 4.0, 1.75, 0.0, None, None),
    ('kerr', 'plain'): ('ez', (5, 7, 7), 4.0, 1.75, 0.0, None, None),
    ('kerr', 'conformal'): ('ez', (5, 5, 7), 4.0, 4.29776668548584, 0.0, 0.9307159185409546, False),
}


def _o2_parameter(operator, branch):
    marks = []
    if (operator, branch) in _O2_XFAIL_REL:
        component, edge, operand, effective, sigma, weight, cut = _O2_XFAIL_AT_EDGE[operator, branch]
        reads = ('components.eps_update' if operator in ('lumped', 'wire', 'msl', 'rlc') else
                 'cfg.medium.eps_r (resolved from materials.eps_r)' if operator == 'waveguide' else
                 'materials.eps_r')
        update = ('components.eps_update' if branch == 'plain' else 'components.tensor.eps')
        reads = f'{reads} = {operand:.12g}'
        update = f'{update} = {effective:.12g}'
        if operator in ('lumped', 'wire', 'msl', 'rlc', 'sheet'):
            sigma_read = 'materials.sigma' if operator == 'sheet' else 'components.sigma_update'
            reads += f' and {sigma_read} = {sigma:.12g}'
            update += f' and components.sigma_update = {sigma:.12g}'
        at = f'{component}({edge[0]},{edge[1]},{edge[2]})'
        if weight is not None:
            geometry = 'cut along its length' if cut else 'not cut along its length (SC clamp)'
            at += f', w={weight:.12g}, {geometry}'
        quantity = ('of the update denominator ε·ε0/dt + σ/2' if operator == 'rlc' else
                    'of x2' if operator == 'sheet' else 'of the ε argument' if operator == 'kerr' else
                    '|a−b|/|b| of the update coefficient Cb')
        rel = _O2_XFAIL_REL[operator, branch]
        marks.append(pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
            f'#1961: {operator} reads {reads}, the E update reads {update} at {at}; '
            f'relative difference {quantity} = {rel:.9g}')))
    return pytest.param(operator, branch, id=f'{operator}-{branch}', marks=marks)


def _o2_boundary(branch):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    return (BoundarySpec(x='cpml', y=Boundary('pec', 'pec', conformal=True), z='cpml')
            if branch == 'conformal' else 'cpml')


def _o2_options(branch):
    if branch in ('smooth', 'full'):
        return dict(subpixel_smoothing=True if branch == 'smooth' else 'full_tensor')
    return dict(conformal_pec=branch == 'conformal')


def _o2_scene(operator, branch, mixed_section=False):
    from rfx import Box, Simulation, Sphere
    from rfx.boundaries.spec import Boundary, BoundarySpec

    if operator == 'waveguide':
        sim = Simulation(freq_max=40e9, domain=(.018, .008, .0066), dx=.001,
                         cpml_layers=2, snap='declared',
                         boundary=BoundarySpec(x='cpml',
                             y='pec',
                             z=Boundary('pec', 'pec', conformal=branch == 'conformal')))
        # The section must stay homogeneous. The distant shape activates
        # smoothing without putting a second medium in the port section.
        sim.add_material('remote', eps_r=4.)
        sim.add(Sphere((.004 if mixed_section else .014, .004, .003), .0013), material='remote')
        sim.add_waveguide_port(.004, direction='+x', mode=(1, 0), mode_type='TE',
                               freqs=[30e9], f0=30e9, bandwidth=.5,
                               probe_offset=2, ref_offset=1)
        return sim
    sim = Simulation(freq_max=8e9 if operator == 'msl' else 20e9, domain=(.012,) * 3, dx=.001,
                     boundary=_o2_boundary(branch), cpml_layers=2, snap='declared')
    sim.add_material('dielectric', eps_r=4., **({'chi3': 1e-3} if operator == 'kerr' else {}))
    sphere = Sphere((.00537, .00613, .00573), .0023)
    sim.add(sphere, material='pec' if branch == 'conformal' else 'dielectric')
    if operator == 'kerr' and branch == 'conformal':
        sim.add(Box((.002, .002, .002), (.010, .010, .010)), material='dielectric')
    point = (.005, .006, .007) if branch == 'conformal' else (.004, .004, .004)
    if operator in ('lumped', 'wire'):
        sim.add_port(point, 'ez', waveform=jnp.ones_like,
                     extent=.002 if operator == 'wire' else None)
    elif operator == 'msl':
        x, y, z = point
        ground = z - .001
        lo, hi, half_width = ((.002, .010, .001) if branch == 'conformal' else
                              (.00165, .01035, .00135))
        sim.add(Box((lo, lo, ground), (hi, hi, ground)), material='pec')
        from rfx.preflight.line_port_coverage import port_node_coordinate
        node = port_node_coordinate(sim, (x, y, ground))
        sim.add(Box((node, y-half_width, z+.001), (hi, y+half_width, z+.001)), material='pec')
        sim.add_msl_port(position=(x, y, ground), width=.002, height=.002,
                         direction='+x', impedance=50., waveform=jnp.ones_like, eps_r_sub=4.)
    else:
        sim.add_source((.002, .002, .002), 'ez', waveform=jnp.ones_like, amplitude_kind='field')
        if operator == 'rlc':
            sim.add_lumped_rlc(point, 'ez', R=70., L=2e-9, C=.3e-12, topology='series')
        elif operator == 'sheet':
            sim.add_thin_conductor(Box((.004, .002, .002), (.004, .009, .009)),
                                   sigma_bulk=100., thickness=1e-5, surface_impedance_f0=5e9)
    return sim


def _o2_changed(materials, branch, axis, edge):
    c = materials.components
    old = c.eps_update if branch == 'plain' else c.tensor.eps
    moved = list(old)
    moved[axis] = moved[axis].at[edge].multiply(1.7)
    if branch == 'plain':
        c = replace(c, eps_update=tuple(moved))
    else:
        c = replace(c, tensor=replace(c.tensor, eps=tuple(moved)))
    return materials._replace(components=c)


def _o2_scale_on_realization(patch, branch, change, seen):
    """Move the operand once, as it first becomes available to consumers.

    Plain components form in the runner; tensor components attach through
    uniform_setup._build_step_setup before admit_setup resolves the section.
    Earlier drive calls have no tensor: moving eps_update there would change
    their operand, rather than the tensor operand owned by the E update.
    """
    from rfx.model import materials as model

    original = model.with_components

    def realize(materials, grid, *args, **kwargs):
        previous = materials.components
        result = original(materials, grid, *args, **kwargs)
        formed = (previous is None if branch == 'plain' else
                  (previous is None or previous.tensor is None)
                  and result.components.tensor is not None)
        if change is not None and formed:
            axis, edge = change
            result = _o2_changed(result, branch, axis, edge)
            eps, _ = _o2_operands(result, branch, axis)
            seen['intervention'] = dict(
                eps=float(eps[edge]), cell=float(result.eps_r[edge]))
        return result

    patch.setattr(model, 'with_components', realize)


def _o2_measure(operator, branch, entry, change=None, mixed_section=False):
    """Record measurement errors per row; never let setup decide an xfail."""
    seen = {}
    try:
        return _o2_measure_impl(operator, branch, entry, change, mixed_section, seen)
    except (ValueError, NotImplementedError) as exc:
        result = dict(seen.get('intervention', {}), refused=str(exc))
        if change is not None and operator == 'waveguide':
            result['note'] = (
                f"{branch}: refused when only the update's operand is scaled: {exc}. "
                'Agreement holds only because admission forbids the difference.')
        return result
    except (Exception, pytest.fail.Exception) as exc:
        return dict(error=f'{type(exc).__name__}: {exc}')


def _o2_measure_impl(operator, branch, entry, change, mixed_section, seen):
    """Public declarations and real consumer calls; stop before a second step.

    setup() captures the public runner's exact kwargs. Replaying those kwargs
    enters the shared uniform step, with the prepared product's final materials.
    No reference is formed from a consumer's coefficient helper.
    """
    from rfx.sources import port_drive
    from rfx.stepping import prepared, uniform_setup, uniform_loop, kerr, waveguide_source
    from rfx.materials import thin_conductor
    from tests.contracts.test_measurement_plan import setup
    import rfx.simulation as simulation
    import rfx.lumped as lumped

    sim = _o2_scene(operator, branch, mixed_section=mixed_section)
    calls = []
    original_advance = prepared.advance
    original_drive = port_drive.port_drive_waveform
    original_denominator = lumped.edge_update_denominator
    original_invariants = uniform_setup.core_step_invariants
    original_sheet = thin_conductor.sheet_update_coeffs
    original_waveguide = waveguide_source.apply_waveguide_port_e
    original_entry = getattr(sim, entry)

    def public(**kwargs):
        kwargs.update(_o2_options(branch))
        if entry == 'forward':
            kwargs.pop('conformal_pec', None)
        if entry == 'run':
            kwargs['compute_s_params'] = False
            kwargs.pop('s_param_freqs', None)
        return original_entry(**kwargs)

    def advance(product, *args, **kwargs):
        seen['prepared_materials'] = product.materials
        return original_advance(product, *args, **kwargs)

    def drive(grid, cell, component, excitation, n_steps, materials, **kwargs):
        value = original_drive(grid, cell, component, excitation, n_steps, materials, **kwargs)
        seen.setdefault('checks', []).append(('constant drive', bool(np.all(np.asarray(value) == value[0]))))
        calls.append(dict(axis=('ex', 'ey', 'ez').index(component), edge=tuple(cell),
                          value=float(value[0]), multiplier=float(kwargs['sigma_port'] * kwargs['unit_field'])))
        return value

    def denominator(materials, cell, component, dt, *args, **kwargs):
        value = original_denominator(materials, cell, component, dt, *args, **kwargs)
        calls.append(dict(axis=('ex', 'ey', 'ez').index(component), edge=tuple(cell), value=float(value)))
        return value

    def sheet(sigma_sheet, materials, dt):
        result = original_sheet(sigma_sheet, materials, dt)
        seen['sheet_a'] = np.asarray(result[0] if isinstance(result, tuple) else result.a)
        seen['sheet_sigma'] = np.asarray(sigma_sheet)
        return result

    def invariants(ctx, **kwargs):
        seen.update(materials=ctx.materials, dt=ctx.dt, dx=ctx.dx, ctx=ctx)
        result = original_invariants(ctx, **kwargs)
        if operator not in ('kerr', 'waveguide'):
            raise _O2ReferenceCaptured
        return result

    def nonlinear(state, e_prev, chi3, eps):
        seen['kerr_eps'] = np.asarray(eps)
        seen['chi3'] = np.asarray(chi3)
        raise _O2ReferenceCaptured

    def waveguide(state, cfg, step, dt, dx):
        # Observe the actual additive correction from zero to avoid subtractive
        # cancellation with the E curl. Use an existing nonzero incident sample.
        index = int(np.argmax(np.abs(np.asarray(cfg.h_inc_table))))
        seen.setdefault('checks', []).append(('incident samples', cfg.h_inc_table.size > 1))
        zero = state._replace(ex=jnp.zeros_like(state.ex), ey=jnp.zeros_like(state.ey), ez=jnp.zeros_like(state.ez))
        result = original_waveguide(zero, cfg, index, dt, dx)
        seen.update(wg_cfg=cfg, wg_field=np.asarray(getattr(result, cfg.e_v_component)),
                    wg_inc=float(cfg.src_amp * cfg.h_inc_table[index]))
        raise _O2ReferenceCaptured

    with pytest.MonkeyPatch.context() as patch, jax.disable_jit():
        patch.setattr(sim, entry, public)
        _o2_scale_on_realization(patch, branch, change, seen)
        patch.setattr(prepared, 'advance', advance)
        patch.setattr(port_drive, 'port_drive_waveform', drive)
        patch.setattr(lumped, 'edge_update_denominator', denominator)
        patch.setattr(thin_conductor, 'sheet_update_coeffs', sheet)
        patch.setattr(uniform_setup, 'core_step_invariants', invariants)
        patch.setattr(uniform_loop, 'core_step_invariants', invariants)
        patch.setattr(kerr, 'apply_kerr_ade', nonlinear)
        patch.setattr(waveguide_source, 'apply_waveguide_port_e', waveguide)
        captured = setup(sim, forward=entry == 'forward')
        grid = captured.pop('grid')
        try:
            simulation.run(grid, seen['prepared_materials'], 12, **captured)
        except _O2ReferenceCaptured:
            pass
        else:
            raise RuntimeError('first step was not captured')
    materials, dt = seen['materials'], seen['dt']
    cells = np.asarray(materials.eps_r)
    if operator in ('sheet', 'kerr', 'waveguide'):
        axis = 2
        eps, sigma = _o2_operands(materials, branch, axis)
        if operator == 'sheet':
            eligible = np.asarray(seen['ctx'].sheet_impedance.mask_ez).copy()
            end = seen['ctx'].sheet_impedance.end_ez
            if end is not None:
                eligible &= ~np.asarray(end)
        elif operator == 'kerr':
            eligible = seen['chi3'] != 0
        else:
            cfg = seen['wg_cfg']
            seen.setdefault('checks', []).append(('waveguide orientation',
                cfg.e_v_component == 'ez' and cfg.direction == '+x'))
            eligible = np.zeros(eps.shape, bool)
            eligible[cfg.x_index, cfg.u_lo:cfg.u_hi, cfg.v_lo:cfg.v_hi] = np.asarray(cfg.hy_profile) != 0
        if operator != 'waveguide' or branch == 'conformal':
            eligible &= eps != cells
        if branch == 'conformal':
            weights = np.asarray(materials.components.tensor.conformal_weights[axis])
            eligible &= (weights > 0) & (weights < 1)
        if not eligible.any():
            raise RuntimeError(f'{operator}-{branch}: no discriminating operator edge')
        edge = tuple(int(i) for i in np.argwhere(eligible)[0]) if change is None else change[1]
        if operator == 'sheet':
            value = -float(np.log(seen['sheet_a'][edge]))
            total_sigma = seen['sheet_sigma'][edge] + sigma[edge]
            ref = float(total_sigma * dt / (EPS_0 * eps[edge]))
        elif operator == 'kerr':
            value, ref = float(seen['kerr_eps'][edge]), float(eps[edge])
        else:
            u, v = edge[1]-cfg.u_lo, edge[2]-cfg.v_lo
            value = float(seen['wg_field'][edge] / (-seen['wg_inc'] * float(cfg.hy_profile[u, v])))
            ref = float(dt / (EPS_0 * eps[edge] * seen['dx']))
    else:
        eligible = []
        for call in calls:
            eps, sigma = _o2_operands(materials, branch, call['axis'])
            edge = call['edge']
            good = eps[edge] != cells[edge]
            if branch == 'conformal':
                w = float(materials.components.tensor.conformal_weights[call['axis']][edge])
                good &= 0 < w < 1
            if good:
                eligible.append(call)
        if not eligible:
            raise RuntimeError(f'{operator}-{branch}: no discriminating operator edge: {calls}')
        call = eligible[0] if change is None else next(c for c in eligible if (c['axis'], c['edge']) == change)
        axis, edge, value = call['axis'], call['edge'], call['value']
        eps, sigma = _o2_operands(materials, branch, axis)
        ref = (float(eps[edge] * EPS_0 / dt + sigma[edge] / 2.) if operator == 'rlc' else
               _o2_cb(materials, branch, axis, edge, dt) * call['multiplier'])
    return dict(operator=operator, branch=branch, entry=entry, axis=axis, edge=edge,
                value=value, ref=ref, rel=abs(value-ref)/abs(ref), eps=float(eps[edge]),
                cell=float(cells[edge]), sigma=float(sigma[edge]), dt=float(dt),
                checks=seen.get('checks', []))


@pytest.fixture(scope='module')
def _o2_observations():
    """Measure and record only; each row is judged in its test call."""
    observations = {}
    for operator, branch in _O2_ROWS:
        for entry in ('run', 'forward'):
            mixed = operator == 'waveguide_second_medium'
            row = _o2_measure('waveguide' if mixed else operator, branch, entry,
                              mixed_section=mixed)
            if row.get('rel', float('inf')) <= _O2_RTOL:
                row['follow'] = _o2_measure(operator, branch, entry, (row['axis'], row['edge']))
            observations[operator, branch, entry] = row
    return observations


@pytest.mark.parametrize('operator,branch', [_o2_parameter(*row) for row in _O2_ROWS])
@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_o2_operator_operand(_o2_observations, operator, branch, entry):
    row = _o2_observations[operator, branch, entry]
    _o2_assert_measurement(row)
    if (operator, branch) in REFUSED:
        assert row == dict(refused=REFUSED[operator, branch])
        return
    np.testing.assert_allclose(row['value'], row['ref'], rtol=_O2_RTOL, atol=0.,
                               err_msg=str(row))
    changed = row['follow']
    _o2_assert_measurement(changed)
    # First prove the intervention, including when admission refused the scene.
    np.testing.assert_allclose(changed['eps'], 1.7 * row['eps'], rtol=_O2_RTOL)
    assert changed['cell'] == row['cell']
    if 'refused' in changed:
        assert (operator, branch) in NOTE
        assert "refused when only the update's operand is scaled" in changed['note']
        assert changed['refused'] in changed['note']
        assert 'waveguide sections require one homogeneous, isotropic, lossless' in changed['refused']
        print(f'O2 step 2 {entry} {operator}-{branch}: {changed}', file=sys.stderr)
        return
    assert changed['ref'] != row['ref']
    if (operator, branch) in NOTE:
        assert changed['rel'] > _O2_RTOL
        np.testing.assert_allclose(changed['value'], row['value'], rtol=_O2_RTOL, atol=0.)
    else:
        np.testing.assert_allclose(changed['value'], changed['ref'], rtol=_O2_RTOL, atol=0.)
    print(f'O2 step 2 assertions passed in call: {entry} {operator}-{branch}; {changed}', file=sys.stderr)


def _o2_assert_measurement(row):
    if 'error' in row:
        pytest.fail(row['error'])
    for label, holds in row.get('checks', []):
        if not holds:
            pytest.fail(f'measurement precondition: {label}')


def test_o2_classifications(_o2_observations):
    for entry in ('run', 'forward'):
        rows = {key[:2]: value for key, value in _o2_observations.items() if key[2] == entry}
        assert set(rows) == set(_O2_ROWS)
        assert {key for key, row in rows.items() if 'refused' in row} == set(REFUSED)
        assert {key for key, row in rows.items() if 'unreachable' in row} == set(UNREACHABLE)
        assert {key for key, row in rows.items()
                if row.get('rel', float('inf')) <= _O2_RTOL
                and ('refused' in row['follow'] or row['follow'].get('rel', 0.) > _O2_RTOL)} == set(NOTE)


@pytest.mark.parametrize('branch', _O2_BRANCHES)
def test_o2_intervention_reaches_kernel(monkeypatch, branch):
    """Always exercise the intervention, even if every operator fails step 1."""
    from rfx.stepping import uniform_loop, yee as kernels

    row = _o2_measure('rlc', branch, 'run')
    _o2_assert_measurement(row)
    axis, edge = row['axis'], row['edge']
    seen = {}
    original_update = uniform_loop._update_e_with_optional_dispersion

    def update(state, materials, dt, dx, **kwargs):
        zeros = jnp.zeros_like(state.ez)
        curls = tuple(zeros.at[edge].set(1.) if a == axis else zeros for a in range(3))
        state = state._replace(ex=zeros, ey=zeros, ez=zeros)
        with monkeypatch.context() as patch:
            patch.setattr(kernels, 'curl_h', lambda *a, **k: curls)
            result = original_update(state, materials, dt, dx, **kwargs)[0]
        seen['increment'] = float(getattr(result, ('ex', 'ey', 'ez')[axis])[edge])
        raise _O2ReferenceCaptured

    sim = _o2_scene('rlc', branch)
    with monkeypatch.context() as patch, jax.disable_jit():
        _o2_scale_on_realization(patch, branch, (axis, edge), seen)
        patch.setattr(uniform_loop, '_update_e_with_optional_dispersion', update)
        with pytest.raises(_O2ReferenceCaptured):
            sim.run(n_steps=1, skip_preflight=True, **_o2_options(branch))
    intervention = seen['intervention']
    # Expected Cb comes from the unscaled scene and the requested 1.7x change,
    # independently of whether the realization hook actually moved anything.
    cb_scaled = 1. / (1.7 * row['eps'] * EPS_0 / row['dt'] + row['sigma'] / 2.)
    cb_unscaled = 1. / (row['eps'] * EPS_0 / row['dt'] + row['sigma'] / 2.)
    print(f'O2 kernel {branch}: increment={seen["increment"]:.12g}, '
          f'Cb scaled={cb_scaled:.12g}, Cb unscaled={cb_unscaled:.12g}', file=sys.stderr)
    np.testing.assert_allclose(seen['increment'], cb_scaled, rtol=_O2_RTOL, atol=0.)
    assert not np.isclose(seen['increment'], cb_unscaled, rtol=_O2_RTOL, atol=0.)
    np.testing.assert_allclose(intervention['eps'], 1.7 * row['eps'], rtol=_O2_RTOL)
