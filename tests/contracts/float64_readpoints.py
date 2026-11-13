"""Capture PR A's R1–R5 without executing a time step."""
import dataclasses
import functools
import sys

import jax
import numpy as np

from rfx import Box, GaussianPulse, Simulation, Sphere
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import drude_pole, lorentz_pole
from rfx.model.operands import realized_operands

SCENES = ('U-mat', 'G-x', 'G-yz', 'U-kerr', 'U-smooth', 'U-conformal', 'U-tfsf', 'G-tfsf')


def mm(values):
    return tuple(float(v) * 1e-3 for v in values)


def scene(label):
    """DESIGN A2d: off-cell census geometry, including its fixed magnetic box."""
    options = dict(freq_max=6e9, domain=mm((40, 30, 26)), dx=.5e-3,
                   cpml_layers=8, precision='float64')
    if label in ('G-x', 'G-tfsf'):
        options['dx_profile'] = np.r_[.5, np.linspace(.25, .75, 78), .5] * 1e-3
    if label == 'G-yz':
        options['dy_profile'] = np.r_[.5, np.linspace(.25, .75, 58), .5] * 1e-3
        options['dz_profile'] = np.r_[.5, np.linspace(.25, .75, 50), .5] * 1e-3
    sim = Simulation(**options)
    specs = [
        ('dielectric', dict(eps_r=4., sigma=.05), (12.35, 20.15), (8.35, 15.15)),
        ('Lorentz', dict(lorentz_poles=[lorentz_pole(2., 2*np.pi*4e9, 1e9)]), (20.15, 27.65), (8.35, 15.15)),
        ('Drude', dict(lorentz_poles=[drude_pole(2*np.pi*3e9, 2e9)]), (12.35, 20.15), (15.15, 21.65)),
        ('Debye', dict(debye_poles=[DebyePole(3., 1e-11)]), (20.15, 27.65), (15.15, 21.65)),
        ('magnetic', dict(mu_r=2.), (28.35, 33.65), (4.85, 11.65)),
    ]
    if label.endswith('tfsf') or label == 'U-src':
        specs = specs[:1]
    elif label == 'U-kerr':
        specs = [('kerr', dict(eps_r=2., chi3=2.2e-19), (28.35, 33.65), (19.35, 24.65))]
    elif label in ('U-smooth', 'U-conformal'):
        specs = []
    for name, kwargs, x, y in specs:
        sim.add_material(name, **kwargs)
        sim.add(Box(mm((x[0], y[0], 8.35)), mm((x[1], y[1], 17.65))), material=name)
    if label == 'U-smooth':
        sim.add_material('dielectric', eps_r=4., sigma=.05)
        sim.add(Sphere(mm((15.3, 12.2, 11.9)), 4.1e-3), material='dielectric')
    if label == 'U-conformal':
        sim.add(Sphere(mm((32.6, 15.3, 12.9)), 3.1e-3), material='pec')
    if label.endswith('tfsf'):
        # test_api.py normal-incidence declaration uses default CPML boundaries.
        sim.add_tfsf_source(f0=3e9, bandwidth=.5, amplitude=1., margin=3)
    else:
        sim.add_source(mm((9.85, 12.35, 10.85)), 'ez', waveform=GaussianPulse(f0=3e9))
    if label == 'U-src':
        sim.add_source(mm((15.35, 11.35, 12.85)), 'ez', waveform=GaussianPulse(f0=3e9))
    sim.add_probe(mm((30.35, 17.85, 14.35)), 'ez')
    return sim, dict(n_steps=200, subpixel_smoothing=label == 'U-smooth',
                     conformal_pec=label == 'U-conformal')


def leaves(value, name):
    if hasattr(value, 'dtype') and hasattr(value, 'shape'):
        yield name, value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from leaves(getattr(value, field.name), f'{name}.{field.name}')
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, f'{name}.{key}' if name else str(key))
    elif isinstance(value, (tuple, list)):
        for key, child in zip(getattr(value, '_fields', range(len(value))), value):
            yield from leaves(child, f'{name}.{key}')
    elif hasattr(value, '__dict__') and type(value).__module__.startswith('rfx'):
        yield from leaves(vars(value), name)


def rounded_through_float32(array):
    """P2 independently reads the low twelve significand bits, not an rfx helper."""
    if array.dtype.kind == 'c':
        array = np.concatenate((array.real.ravel(), array.imag.ravel()))
    with np.errstate(over='ignore', invalid='ignore'):
        narrow = array.astype(np.float32)
        return bool(np.array_equal(array, narrow.astype(array.dtype)) and
                    np.any(narrow.view(np.uint32) & np.uint32((1 << 12) - 1)))


class Captured(BaseException):
    """Exit setup before the scan can step."""


def capture(label, *, values=None):
    sim, options = scene(label)
    arrays, holders, builder, scan = [], {}, {}, {}
    body_samples, tables = {}, {}
    if values is not None:
        values.update(poles=[], sources=[], conformal=[])

    def keep(point, name, value):
        if hasattr(value, 'dtype') and np.dtype(value.dtype).kind in 'fc':
            arrays.append((point, name, value))

    def profile(frame, event, arg):
        if not frame.f_globals.get('__name__', '').startswith('rfx.'):
            return
        name, local = frame.f_code.co_name, frame.f_locals
        if values is not None:
            if event == 'call' and name == 'admit_setup':
                values.update(grid=local['grid'], materials=local['materials'], updates=local['updates'])
            if event == 'call' and name == 'build_cell_table':
                body_samples.clear()
            if event == 'return' and name == 'union_at':
                body_samples[tuple(local['half_axes'])] = np.asarray(arg)
            if event == 'return' and name == 'build_cell_table' and 'sdf' in local:
                tables[id(arg)] = dict(sdf=local['sdf'], widths=local['widths'],
                                      tol=local['tol'], midpoints=dict(body_samples))
            if event == 'return' and name == 'conformal_fractions_from_table':
                values['conformal'].append((tables[id(local['table'])], local['table'], arg))
            if event == 'return' and name == 'pole_component_weights' and arg is not None:
                values['poles'].append((np.asarray(local['mask']), local['cell_sizes'], arg))
        if event == 'call' and name == 'admit_setup':
            for record in realized_operands(local['grid'], local['materials'], local['updates'], local['localized']):
                keep('R1', record.name, record.array)
            for key, value in leaves(local['grid'], 'grid'):
                keep('R2', key, value)
        if event == 'return' and name == 'init_cpml' and arg is not None:
            for key, value in leaves(arg[0], 'cpml'):
                keep('R3', key, value)
        if event == 'return' and name == 'pole_component_weights' and arg is not None:
            builder[id(arg)] = arg
        if event == 'return' and name == '_realize_poles' and arg is not None:
            counts = {}
            for pole, weights in zip(local['poles'], local['weights']):
                family = 'Debye' if local['kind'] == 'Debye' else ('Drude' if pole.omega_0 == 0 else 'Lorentz')
                index = counts.get(family, 0)
                counts[family] = index + 1
                assert builder[id(weights)] is weights
                for axis, value in zip('xyz', weights):
                    keep('R5', f'pole edge weights, {family} {index}, component {axis}', value)
        if event == 'call' and name == 'recorded_scan':
            scan.update(local)
            body = local['f']
            while isinstance(body, functools.partial):
                for path, value in leaves(body.keywords, ''):
                    holders.setdefault(id(value), path)
                body = body.func
            for key, cell in zip(body.__code__.co_freevars, body.__closure__ or ()):
                for path, value in leaves(cell.cell_contents, key):
                    holders.setdefault(id(value), path)
            ancestor = frame.f_back
            while ancestor:
                if ancestor.f_globals.get('__name__', '').startswith('rfx.'):
                    for key, value in leaves(ancestor.f_locals, ''):
                        holders.setdefault(id(value), key)
                ancestor = ancestor.f_back
            raise Captured()

    from unittest.mock import patch
    import contextlib
    import rfx.model.materials as material_module
    original_coefficient = material_module.e_update_coefficient_at

    def source_spy(*args, **kwargs):
        result = original_coefficient(*args, **kwargs)
        values["sources"].append((args[1], args[2], args[3], result))
        return result

    source_patch = (patch.object(material_module, "e_update_coefficient_at", source_spy)
                    if values is not None else contextlib.nullcontext())
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        try:
            with source_patch:
                sim.run(**options)
        except Captured:
            pass
    finally:
        sys.setprofile(previous)
    assert scan, f'{label}: no recorded_scan captured'
    originals = list(arrays)
    named = {}
    for _, name, value in originals:
        named.setdefault(id(value), name)
    for key, value in leaves(scan['init'], 'carry'):
        keep('R4', key, value)
    for key, value in leaves(scan['xs'], 'xs'):
        keep('R4', key, value)
    one = jax.tree_util.tree_map(lambda value: value[0], scan['xs'])
    closed = jax.make_jaxpr(scan['f'])(scan['init'], one)
    for var, value in zip(closed.jaxpr.constvars, closed.consts):
        a = np.asarray(value)
        if a.dtype.kind not in 'fc':
            continue
        if a.dtype in (np.dtype('float64'), np.dtype('complex128')) and not rounded_through_float32(a):
            continue  # A2e: passing R4 constants need no name.
        name = named.get(id(value))
        if name is None:
            for _, candidate, original in originals:
                if original.shape == a.shape and np.array_equal(a, np.asarray(original).astype(np.float64)):
                    name = f'widened {candidate}'
                    break
        if name is None:
            name = holders.get(id(value))
        if name is None:
            primitive = next(str(eqn.primitive) for eqn in closed.jaxpr.eqns
                             if any(operand is var for operand in eqn.invars))
            name = f'R4 {a.dtype} {a.shape} read by {primitive}'
        keep('R4', name, value)
    rows = []
    for point, name, value in arrays:
        a = np.asarray(value)
        p1 = a.dtype not in (np.dtype('float64'), np.dtype('complex128'))
        p2 = point in ('R4', 'R5') and rounded_through_float32(a)
        rows.append((point, name, str(a.dtype), tuple(a.shape), p1, p2))
    return rows
