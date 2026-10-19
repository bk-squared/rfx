"""Assembly/observable census for DESIGN.md, 2026-10-12; no product fixes.

No differences are classified here. Each entry uses a fresh public declaration.
Only declarations, seam capture, freezing, and comparison are retained here.
"""
from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import partial
import importlib
import inspect
import traceback
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Simulation, Box, PolylineWire, DebyePole, LorentzPole, drude_pole
from rfx.boundaries.spec import Boundary, BoundarySpec

# Every patched module/attribute is named here once. The contract passes this
# table to capture, so a pipeline restructure changes its integration seams here.
SEAMS = {
    'scan': ('jax.lax', 'scan'),
    'uniform': ('rfx.simulation', 'run'),
    'setup': ('rfx.stepping.uniform_loop', '_build_step_setup'),
    'core': ('rfx.stepping.uniform_loop', 'make_core_step'),
    'invariants': ('rfx.stepping.uniform_loop', 'core_step_invariants'),
    'graded': ('rfx.runners.nonuniform', 'run_nonuniform_path'),
    'graded_step': ('rfx.stepping.graded_loop', '_build_nu_scan'),
    'distributed': ('rfx.runners._distributed_nu_execute', 'execute_from_materials'),
    'distributed_step': ('rfx.runners.distributed_nu', 'run_nonuniform_distributed_pec'),
}

PATHS = ('uniform', 'graded', 'multi-device-graded', 'multi-device-uniform')
PROFILE = (1.2, 1.1, 1.0, .9, .8, .7, .6, .6, .7, .8, .9, 1.0, 1.1, .6)
SOURCE = (.011, .009, .0055)
PROBE = (.021, .012, .007)
BOX_LO = (.0143, .0072, .0033)
BOX_HI = (.0197, .0138, .0086)

# Contract addendum 11: common arrays always compare; these sets govern
# one-sided arrays only. Unknown one-sided roots require leader review.
MEASUREMENT_ROOTS = {
    'time_series', 'dft_planes', 'dft_time_records', 'flux_monitors',
    'ntff_data', 'ntff_box', 's_params', 'freqs', 'freqs_hz',
    'lumped_port_sparams', 'wire_port_sparams', 'sparam_time_records',
    'waveguide_ports', 'waveguide_sparams', 'waveguide_port_flux',
    'current_moment_data', 'current_moment_monitor', 'settling_witness',
}
PACKAGING_ROOTS = {'state', 'realized_geometry', 'diagnostics'}


def base_scene(path='uniform', *, source=True, **options):
    mesh = {'dz_profile': 1e-3 * np.array(PROFILE)} if path in ('graded', 'multi-device-graded') else {}
    sim = Simulation(freq_max=20e9, domain=options.pop('domain', (.030, .020, .012)), dx=options.pop('dx', .001),
                     cpml_layers=4, **mesh, **options)
    if source:
        sim.add_source(SOURCE, 'ez')
    sim.add_probe(PROBE, 'ez')
    return sim


def build_b_cpml(path):
    return base_scene(path, boundary='cpml')


def build_b_pecsides(path):
    return base_scene(path, boundary=BoundarySpec(x='cpml', y='pec', z='pec'))


def build_b_pmcpair(path):
    return base_scene(path, boundary=BoundarySpec(x='cpml', y='pmc', z='pec'))


def build_b_pmcone(path):
    return base_scene(path, boundary=BoundarySpec(x='cpml', y=Boundary(lo='pmc', hi='cpml'), z='cpml'))


def build_b_periodic(path):
    return base_scene(path, boundary=BoundarySpec(x='cpml', y='periodic', z='cpml'))


def build_b_upml(path):
    return base_scene(path, boundary='upml')


def build_b_asym(path):
    return base_scene(path, boundary=BoundarySpec(x=Boundary(lo='cpml', hi='pec'), y='cpml', z='pec'))


def build_b_closed(path):
    return base_scene(path, boundary='pec')


def build_s_field(path):
    return base_scene(path, source=False).add_source(SOURCE, 'ez', amplitude_kind='field')


def build_s_polarized(path):
    return base_scene(path, source=False).add_polarized_source(SOURCE, polarization='slant45')


def build_pw_a(path):
    return base_scene(path, source=False).add_tfsf_source()


def build_pw_b(path):
    # tests/contracts/test_tfsf_boundary_override.py:45-50: methodB requires oblique incidence.
    return base_scene(path, source=False).add_tfsf_source(method='methodB', angle_deg=20, polarization='ez')


def build_pw_oblique(path):
    return base_scene(path, source=False).add_tfsf_source(angle_deg=20)


def build_pw_closed(path):
    return base_scene(path, source=False).add_tfsf_source(closed_box=True)


def build_p_lumped(path):
    return base_scene(path, source=False).add_port(SOURCE, 'ez', impedance=50.)


def build_p_lumped2(path):
    return build_p_lumped(path).add_port((.022, .009, .0055), 'ez', impedance=50., excite=False)


def build_p_wire(path):
    return base_scene(path).add_port((.008, .012, .003), 'ez', impedance=50., extent=.003)


def build_p_coax(path):
    # Arguments from tests/contracts/test_line_stub_refusal_1512.py:184.
    return base_scene(path).add_coaxial_port((.006, .006, .008), face='bottom', pin_length=.004)


def build_p_msl(path):
    # Layout: tests/unit/preflight/test_line_stub.py:278-289, strip starts at
    # the port plane and reaches the domain end, over substrate and ground.
    # Relative to base: x domain 30 -> 40 mm; substrate/ground added; strip
    # x=6..40 mm, y=10..14 mm, z=3 mm. Base source/probe retained.
    sim = base_scene(path, domain=(.040, .020, .012))
    sim.add_material('sub', eps_r=2.2)
    sim.add(Box((0., 0., 0.), (.040, .020, .003)), material='sub')
    sim.add(Box((0., 0., 0.), (.040, .020, 0.)), material='pec')
    sim.add(Box((.006, .010, .003), (.040, .014, .003)), material='pec')
    sim.add_msl_port(position=(.006, .012, 0.), width=.004, height=.003,
                     direction='+x', impedance=50., eps_r_sub=2.2,
                     n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    return sim


def build_p_waveguide(path):
    # Replaces boundary with y/z PEC; full aperture (default ranges).
    return base_scene(path, source=False, boundary=BoundarySpec(x='cpml', y='pec', z='pec')).add_waveguide_port(
        .006, f0=10e9, freqs=np.array([10e9]), probe_offset=3, ref_offset=2)


def build_p_floquet(path):
    # Port arguments imitate tests/contracts/boundary_cases.py:28.
    # Replaces boundary with x/y periodic, as required by a z-normal Floquet port.
    return base_scene(path, boundary=BoundarySpec(x='periodic', y='periodic', z='cpml')).add_floquet_port(
        .003, freqs=np.array([10e9]), f0=10e9)


def dielectric(path, **properties):
    sim = base_scene(path)
    sim.add_material('box', eps_r=4., **properties)
    return sim.add(Box(BOX_LO, BOX_HI), material='box')


def build_m_diel(path):
    return dielectric(path)


def build_m_lossy(path):
    return dielectric(path, sigma=.5)


def build_m_mu(path):
    return dielectric(path, mu_r=2.)


def build_m_debye(path):
    return dielectric(path, debye_poles=[DebyePole(delta_eps=1., tau=1e-11)])


def build_m_lorentz(path):
    omega = 2 * np.pi * 8e9
    return dielectric(path, lorentz_poles=[LorentzPole(omega_0=omega, delta=.15 * omega, kappa=.4 * omega**2)])


def build_m_drude(path):
    # Public Drude form: lorentz_poles=[drude_pole(omega_p, gamma)].
    # Imitates tests/contracts/path_equivalence/builders.py:74.
    return dielectric(path, lorentz_poles=[drude_pole(2 * np.pi * 5e9, 1e9)])


def build_c_pecbox(path):
    return base_scene(path).add(Box(BOX_LO, BOX_HI), material='pec')


def build_c_sheet(path):
    return base_scene(path).add_thin_conductor(Box((.014, .006, .004), (.020, .014, .004)))


def build_c_film(path):
    return base_scene(path).add_thin_conductor(Box((.014, .006, .004), (.020, .014, .004)), sigma_bulk=1e5)


def build_c_wire(path):
    # Public thin-wire form: add(PolylineWire(..., radius=0), material='pec').
    # Imitates tests/contracts/test_realized_geometry_record.py:330.
    return base_scene(path).add(PolylineWire(((.014, .012, .004), (.019, .012, .004)), radius=0.), material='pec')


def build_c_pinned(path):
    return base_scene(path).add_pinned_sheet(plane_index=8, i_range=(18, 24), j_range=(10, 18))


def build_c_smooth(path):
    # Subpixel smoothing is an entry keyword (A6), not a constructor option.
    return dielectric(path)


def build_r_rlc(path):
    return base_scene(path).add_lumped_rlc((.017, .010, .006), 'ez', R=50., L=1e-9, C=1e-12, topology='series')


def build_o_dft(path):
    return base_scene(path).add_dft_plane_probe(axis='x', coordinate=.020, freqs=np.array([10e9]), name='census')


def build_o_flux(path):
    return base_scene(path).add_flux_monitor(axis='x', coordinate=.020, freqs=np.array([10e9]), name='census')


def build_o_ntff(path):
    return base_scene(path).add_ntff_box((.006, .004, .002), (.025, .016, .010), freqs=np.array([10e9]))


def build_o_vector(path):
    return base_scene(path).add_vector_probe(PROBE)


def build_o_cm(path):
    # rfx/current_moments.py:1014-1032: k*max_offset <= .5, including slab
    # thickness. At 10 GHz use 1 mm blocks and a 3 mm slab (z=4..7 mm).
    return base_scene(path).add_current_moment_monitor((.006, .004, .004), (.025, .016, .007), .001, np.array([10e9]))


BUILDERS = {
    'B-cpml': build_b_cpml, 'B-pecsides': build_b_pecsides,
    'B-pmcpair': build_b_pmcpair, 'B-pmcone': build_b_pmcone,
    'B-periodic': build_b_periodic, 'B-upml': build_b_upml,
    'B-asym': build_b_asym, 'B-closed': build_b_closed,
    'S-field': build_s_field, 'S-polarized': build_s_polarized,
    'PW-A': build_pw_a, 'PW-B': build_pw_b, 'PW-oblique': build_pw_oblique,
    'PW-closed': build_pw_closed, 'P-lumped': build_p_lumped,
    'P-lumped2': build_p_lumped2, 'P-wire': build_p_wire,
    'P-coax': build_p_coax, 'P-msl': build_p_msl,
    'P-waveguide': build_p_waveguide, 'P-floquet': build_p_floquet,
    'M-diel': build_m_diel, 'M-lossy': build_m_lossy, 'M-mu': build_m_mu,
    'M-debye': build_m_debye, 'M-lorentz': build_m_lorentz,
    'M-drude': build_m_drude, 'C-pecbox': build_c_pecbox,
    'C-sheet': build_c_sheet, 'C-film': build_c_film, 'C-wire': build_c_wire,
    'C-pinned': build_c_pinned, 'C-smooth': build_c_smooth,
    'R-rlc': build_r_rlc, 'O-dft': build_o_dft, 'O-flux': build_o_flux,
    'O-ntff': build_o_ntff, 'O-vector': build_o_vector, 'O-cm': build_o_cm,
}


@dataclass(frozen=True)
class FrozenArray:
    dtype: str
    shape: tuple
    data: bytes

    def array(self):
        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape)


def freeze(value, _active=None):
    """Freeze recursively; unsupported or cyclic values remain explicit markers."""
    if isinstance(value, jax.core.Tracer):
        # Observe the primal through JAX's public callback API. The one-value
        # sweep invokes this once with its element 0; AD supplies its primal.
        # No tracer internals, replayed builder, or reconstructed model is used.
        observed = []
        jax.debug.callback(lambda item: observed.append(freeze(np.asarray(item))), value, ordered=True)
        jax.effects_barrier()
        return observed[0] if len(observed) == 1 else 'UNFROZEN:' + type(value).__name__
    if isinstance(value, Enum):
        return (type(value).__name__, value.name)
    if isinstance(value, jax.Device):
        return ('device', value.platform, value.id)
    if isinstance(value, np.generic):
        array = np.asarray(value)
        return FrozenArray(array.dtype.str, array.shape, array.tobytes())
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    if isinstance(value, (np.ndarray, jax.Array)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            return 'UNFROZEN:' + type(value).__name__
        return FrozenArray(array.dtype.str, array.shape, array.tobytes())
    active = set() if _active is None else _active
    if id(value) in active:
        return 'UNFROZEN:' + type(value).__name__
    active.add(id(value))
    try:
        if isinstance(value, (set, frozenset)):
            return ('set', tuple(sorted((freeze(item, active) for item in value), key=repr)))
        if isinstance(value, slice):
            return ('slice', freeze(value.start, active), freeze(value.stop, active),
                    freeze(value.step, active))
        if isinstance(value, partial):
            return ('partial', freeze(value.func, active), freeze(value.args, active),
                    freeze(value.keywords, active))
        if is_dataclass(value) and not isinstance(value, type):
            return {'__class__': type(value).__module__ + '.' + type(value).__qualname__,
                    **{f.name: freeze(getattr(value, f.name), active) for f in fields(value)}}
        if isinstance(value, tuple) and hasattr(value, '_fields'):
            return {'__class__': type(value).__module__ + '.' + type(value).__qualname__,
                    **{name: freeze(getattr(value, name), active) for name in value._fields}}
        if isinstance(value, dict):
            return {key: freeze(item, active) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(freeze(item, active) for item in value)
        if callable(value):
            if not hasattr(value, '__qualname__') or not hasattr(value, '__module__'):
                return 'UNFROZEN:' + type(value).__name__
            cells = []
            for cell in getattr(value, '__closure__', None) or ():
                try:
                    cells.append(freeze(cell.cell_contents, active))
                except ValueError:
                    cells.append('UNFROZEN:cell')
            return {'module': value.__module__, 'qualified_name': value.__qualname__,
                    'owner_class': (type(value.__self__).__module__ + '.' + type(value.__self__).__qualname__
                                    if inspect.ismethod(value) else None),
                    'closure': tuple(cells),
                    'defaults': freeze(getattr(value, '__defaults__', None), active),
                    'keyword_defaults': freeze(getattr(value, '__kwdefaults__', None), active)}
        try:
            attributes = vars(value)
        except TypeError:
            return 'UNFROZEN:' + type(value).__name__
        return {'class': type(value).__name__, 'vars': freeze(attributes, active)}
    finally:
        active.remove(id(value))


def array_difference(left, right):
    """One array is one leaf; byte-wise element count includes signed zeros."""
    a, b = left.array(), right.array()
    same_layout = left.dtype == right.dtype and left.shape == right.shape
    equal = same_layout and left.data == right.data
    result = {'dtype': [left.dtype, right.dtype], 'shape': [list(left.shape), list(right.shape)],
              'bytes_equal': equal, 'differing_elements': None,
              'max_abs_difference': None, 'peak': None}
    if same_layout:
        av = np.frombuffer(left.data, np.uint8).reshape(a.size, a.dtype.itemsize)
        bv = np.frombuffer(right.data, np.uint8).reshape(b.size, b.dtype.itemsize)
        result['differing_elements'] = int(np.count_nonzero(np.any(av != bv, axis=1)))
    if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
        kind = np.complex128 if a.dtype.kind == 'c' or b.dtype.kind == 'c' else np.float64
        aa, bb = a.astype(kind), b.astype(kind)
        with np.errstate(all='ignore'):
            result['peak'] = float(max(np.max(np.abs(aa), initial=0), np.max(np.abs(bb), initial=0)))
            if a.shape == b.shape:
                result['max_abs_difference'] = float(np.max(np.abs(aa - bb), initial=0))
    return result


def compare(left, right, path=''):
    """Return every differing leaf. UNFROZEN markers never compare equal."""
    rows = []
    def marker(value):
        return isinstance(value, str) and value.startswith('UNFROZEN:')
    if marker(left) or marker(right):
        return [{'path': path, 'run': repr(left), 'forward': repr(right), 'unfrozen': True}]
    if isinstance(left, FrozenArray) and isinstance(right, FrozenArray):
        diff = array_difference(left, right)
        return [] if diff['bytes_equal'] else [dict(path=path, **diff)]
    if type(left) is not type(right):
        return [{'path': path, 'run': describe(left), 'forward': describe(right)}]
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys(), key=repr):
            child = f'{path}/{key}'
            if key not in left or key not in right:
                rows.extend(missing_leaves(right[key] if key not in left else left[key], child,
                                           'forward' if key not in left else 'run'))
            else:
                rows.extend(compare(left[key], right[key], child))
    elif isinstance(left, (tuple, list)):
        for i in range(max(len(left), len(right))):
            child = f'{path}/{i}'
            if i >= len(left) or i >= len(right):
                rows.extend(missing_leaves(right[i] if i >= len(left) else left[i], child,
                                           'forward' if i >= len(left) else 'run'))
            else:
                rows.extend(compare(left[i], right[i], child))
    elif left != right:
        rows.append({'path': path, 'run': describe(left), 'forward': describe(right)})
    return rows


def describe(value):
    if isinstance(value, FrozenArray):
        return {'dtype': value.dtype, 'shape': list(value.shape), 'bytes': len(value.data)}
    if isinstance(value, (dict, list, tuple)):
        return {'type': type(value).__name__, 'length': len(value)}
    return repr(value)


def missing_leaves(value, path, side):
    if isinstance(value, dict) and value:
        return [row for key, item in value.items() for row in missing_leaves(item, f'{path}/{key}', side)]
    if isinstance(value, (tuple, list)) and value:
        return [row for i, item in enumerate(value) for row in missing_leaves(item, f'{path}/{i}', side)]
    return [{'path': path, 'present_only': side, 'value': describe(value)}]


class Captured(Exception):
    """The requested assembly was captured without advancing a field step."""


class CaptureFailure(RuntimeError):
    """A measurement failed (distinct from a product admission refusal)."""


def invoke(sim, entry, path, n_steps, entry_options=None):
    kwargs = {'n_steps': n_steps, 'skip_preflight': True}
    if entry == 'forward':
        kwargs['checkpoint'] = False
    if path.startswith('multi-device'):
        devices = jax.devices('cpu')
        if len(devices) < 2:
            pytest.skip('Need >=2 JAX devices. Set XLA_FLAGS=--xla_force_host_platform_device_count=2 before importing jax.')
        kwargs['devices'] = devices[:2]
        if entry == 'forward':
            kwargs['distributed'] = True
    kwargs.update(entry_options or {})
    return getattr(sim, entry)(**kwargs)


def exception_record(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    frame = frames[-1]
    return {'outcome': 'raised', 'type': type(exc).__name__, 'message': str(exc),
            'raising_site': f'{frame.filename}:{frame.lineno}'}


def bound_arguments(fun, args, kwargs):
    bound = inspect.signature(fun).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def seam(key, seams=SEAMS):
    module, attribute = seams[key]
    owner = importlib.import_module(module)
    parts = attribute.split('.')
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def capture(builder, entry, path, level, n_steps=150, *, entry_options=None, seams=SEAMS):
    """Capture assembly from a fresh declaration without executing a field step."""
    captured, sites = {}, {}
    def original(key):
        owner, name = seam(key, seams)
        return getattr(owner, name)
    def install(stack, key, replacement):
        owner, name = seam(key, seams)
        stack.enter_context(patch.object(owner, name, replacement))
    def remember_site(key):
        frame = inspect.currentframe().f_back.f_back
        sites[key] = f'{frame.f_code.co_filename}:{frame.f_lineno}'
    def stop_at(fun, key):
        def stopped(*args, **kwargs):
            remember_site(key)
            captured[key] = freeze(bound_arguments(fun, args, kwargs))
            raise Captured
        return stopped
    scan = original('scan')
    def forbid_step(*args, **kwargs):
        if path == 'uniform' and level == 'b' and 'ctx' in captured:
            # Addendum 12: the drive table lives in xs, outside the ctx.
            scan_inputs = bound_arguments(scan, args, kwargs)
            captured['xs'] = freeze(scan_inputs['xs'])
            captured['init'] = freeze(scan_inputs['init'])
            remember_site('xs')
            raise Captured
        raise CaptureFailure('L1 reached jax.lax.scan before its capture seam')
    try:
        sim = builder(path)
        with ExitStack() as stack:
            install(stack, 'scan', forbid_step)
            if path == 'multi-device-graded':
                # A5: run may fall back to one device before the shared seam.
                install(stack, 'graded', stop_at(original('graded'), 'fallback_arguments'))
            if level == 'a':
                key = 'uniform' if path == 'uniform' else 'graded' if path == 'graded' else 'distributed'
                install(stack, key, stop_at(original(key), 'arguments'))
            elif path == 'uniform':
                build_setup, build_core = original('setup'), original('core')
                invariants = original('invariants')
                def setup(*args, **kwargs):
                    result = build_setup(*args, **kwargs)
                    captured['carry_init'] = freeze(result.carry_init)
                    remember_site('carry_init')
                    return result
                def core(ctx, *args, **kwargs):
                    remember_site('ctx')
                    captured['ctx'] = freeze(ctx)
                    passed = bound_arguments(build_core, (ctx, *args), kwargs).get('invariants')
                    captured['invariants'] = freeze(invariants(ctx) if passed is None else passed)
                    return build_core(ctx, *args, **kwargs)
                install(stack, 'setup', setup)
                install(stack, 'core', core)
            elif path == 'graded':
                build_scan = original('graded_step')
                def build(*args, **kwargs):
                    remember_site('arguments')
                    captured['arguments'] = freeze(bound_arguments(build_scan, args, kwargs))
                    captured['result'] = freeze(build_scan(*args, **kwargs))
                    raise Captured
                install(stack, 'graded_step', build)
            else:
                install(stack, 'distributed_step', stop_at(original('distributed_step'), 'arguments'))
            invoke(sim, entry, path, n_steps, entry_options)
        raise CaptureFailure('entry returned without reaching the requested L1 seam')
    except Captured:
        return {'outcome': 'fallback' if 'fallback_arguments' in captured else 'captured',
                'value': captured, 'seam_sites': sites}
    except CaptureFailure:
        raise
    except Exception as exc:
        return exception_record(exc)


def observables(result):
    """Collect arrays and per-attribute read errors without changing admission."""
    found = {}
    raises = {}
    active = set()
    def read_attribute(owner, name, path):
        try:
            value = getattr(owner, name)
        except Exception as exc:
            raises[path] = f'raises: {type(exc).__name__}: {exc}'
        else:
            walk(value, path)

    def walk(value, path):
        if isinstance(value, (np.ndarray, jax.Array)):
            found[path] = freeze(value)
            return
        if value is None or isinstance(value, (str, bytes, bool, int, float, complex, np.generic)) or callable(value):
            return
        if id(value) in active:
            return
        active.add(id(value))
        try:
            if isinstance(value, tuple) and hasattr(value, '_fields'):
                names = value._fields
            elif is_dataclass(value):
                names = tuple(f.name for f in fields(value))
            else:
                names = ()
            if names:
                for name in names:
                    read_attribute(value, name, f'{path}/{name}' if path else str(name))
            if isinstance(value, dict):
                items = value.items()
            elif isinstance(value, (tuple, list)) and not names:
                items = enumerate(value)
            elif hasattr(value, '__dict__') and not names:
                for name in vars(value):
                    read_attribute(value, name, f'{path}/{name}' if path else str(name))
                items = ()
            else:
                items = ()
            for name, item in items:
                walk(item, f'{path}/{name}' if path else str(name))
            for name in dir(type(value)):
                if isinstance(inspect.getattr_static(type(value), name), property) and name not in names:
                    read_attribute(value, name, f'{path}/{name}' if path else str(name))
        finally:
            active.remove(id(value))
    # Enumerate the result's full public attribute surface, including class
    # attributes and computed properties. Reading each attribute is isolated.
    for name in dir(result):
        if not name.startswith('_'):
            read_attribute(result, name, name)
    return {'arrays': found, 'raises': raises}


def run_l2(builder, entry, path, n_steps=150, *, entry_options=None):
    """Admission covers only the entry call; collection happens after both calls."""
    try:
        sim = builder(path)
    except Exception as exc:
        return {'outcome': 'not_called', 'construction': exception_record(exc)}
    try:
        result = invoke(sim, entry, path, n_steps, entry_options)
    except Exception as exc:
        return exception_record(exc)
    return {'outcome': 'admitted', 'result': result}


def equal_request_pair(declaration, path, direction, *, level=None, n_steps=150,
                       entry_options=None, seams=SEAMS):
    """The same S request on both entries: (a) run's default, (b) none.

    (a) forward receives exactly the bins run's default returned. When
    run's default returns no bins (it refused before running, or it was
    admitted and returned no S), forward receives run's declared request:
    the default grid of the built simulation's freq_max. L1 has no result,
    so run's bins come from a one-step run of the same declaration. A
    declaration with no impedance port makes no S request on run, so
    forward gets none.
    (b) run(compute_s_params=False) against forward's default.
    """
    from rfx.measurement import default_frequency_grid
    assert direction in ('a', 'b')
    options = dict(entry_options or {})
    def call(entry, requested):
        if level is not None:
            return capture(BUILDERS[declaration], entry, path, level,
                           n_steps=n_steps, entry_options=requested, seams=seams)
        return run_l2(BUILDERS[declaration], entry, path, n_steps=n_steps,
                      entry_options=requested)
    run = call('run', options if direction == 'a' else
               {**options, 'compute_s_params': False})
    if direction == 'a':
        sim = BUILDERS[declaration](path)
        if any(port.impedance != 0 for port in sim._ports):
            returned = run if level is None else run_l2(BUILDERS[declaration], 'run', path,
                                                         n_steps=1, entry_options=options)
            bins = returned['result'].freqs if returned['outcome'] == 'admitted' else None
            if bins is None:
                bins = default_frequency_grid(sim._freq_max, 50, host='graded' in path)
            options = {**options, 'port_s11_freqs': bins}
    forward = call('forward', options)
    return {'run': run, 'forward': forward}


def compare_l2(left, right, *, left_raises=None, right_raises=None):
    common = sorted(left.keys() & right.keys())
    compared = {}
    for name in common:
        diff = array_difference(left[name], right[name])
        peak, delta = diff['peak'], diff['max_abs_difference']
        if peak is None or delta is None:
            ulps = None
        else:
            with np.errstate(all='ignore'):
                ulps = float(delta / float(np.spacing(np.float32(peak))))
        compared[name] = dict(diff, float32_peak_ulps=ulps)
    left_raises, right_raises = left_raises or {}, right_raises or {}
    run_only = sorted(left.keys() - right.keys())
    forward_only = sorted(right.keys() - left.keys())
    one_sided = {name: {'run': 'array' if name in left else left_raises.get(name, 'absent'),
                        'forward': 'array' if name in right else right_raises.get(name, 'absent')}
                 for name in run_only + forward_only}
    return {'observables': compared, 'run_only': run_only, 'forward_only': forward_only,
            'observable_errors': {'run': left_raises, 'forward': right_raises},
            'one_sided': one_sided}



DRIVER_OPTIONS = {
    'return_state', 'checkpoint', 'checkpoint_segments', 'gradient',
    'report_every', 'snapshot', 'emit_time_series', 'checkpoint_every',
    'n_warmup', 'gather_final_state', 'lane', 'diagnostics',
}


def normalize_inputs(value, path, location=()):
    """Only C2(b,c) and the path-specific addendum normalizations."""
    if isinstance(value, dict):
        return {key: normalize_inputs(item, path, location + (key,))
                for key, item in value.items()
                # grid_key caches ids of grid arrays, not physical inputs.
                if key != 'grid_key' and not (
                    path == 'multi-device-graded' and location == ('arguments',)
                    and key in DRIVER_OPTIONS)}
    if isinstance(value, (tuple, list)):
        return type(value)(normalize_inputs(item, path, location + (i,))
                           for i, item in enumerate(value))
    return value


def l2_differences(comparison):
    """All common arrays compare; only one-sided packaging is excluded."""
    failures = [dict(path=name, **diff) for name, diff in comparison['observables'].items()
                if not diff['bytes_equal']]
    for name, sides in comparison['one_sided'].items():
        root = name.split('/')[0]
        if root in MEASUREMENT_ROOTS:
            failures.append(dict(path=name, **sides))
        elif root not in PACKAGING_ROOTS:
            # A known-row xfail must never hide an unclassified root.
            raise CaptureFailure(f'unclassified one-sided observable: {name}')
    return failures


def _assert_same_witness_value(left, right, where):
    """Bit-for-bit (NaN-equal) through the nested dicts and lists a witness holds."""
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys(), where
        for key in left:
            _assert_same_witness_value(left[key], right[key], f'{where}/{key}')
    elif isinstance(left, (list, tuple)) and not np.isscalar(left) and any(
            isinstance(item, (dict, list, tuple)) for item in left):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), where
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_same_witness_value(a, b, f'{where}/{index}')
    else:
        np.testing.assert_array_equal(left, right, err_msg=where)


def port_drive_witness_leaves(sim, run, forward):
    """Check separate S drives before excluding their eight run-only leaves.

    Decision C: forward judges its declared drive's port record; run also
    judges the passive port's separate drive. A witness has no drive-id field:
    attribution rests on the driver's loop order (rfx/probes/sparam_driver.py
    drives the impedance ports in registration order, one per_drive entry each).
    At the census's 150 steps the two entries are byte-identical (undetermined),
    so the driven/passive distinction is pinned at a determined record length
    in test_settling_witness_shared.py.
    """
    ports = [(i, port) for i, port in enumerate(sim._ports) if port.impedance != 0]
    port_ids = [f'port:{i}:mode:0' for i, _ in ports]
    for result in (run, forward):
        assert [record.id for record in result.sparam_time_records
                if record.kind == 'lumped'] == port_ids
    actual, expected = run.settling_witness, forward.settling_witness
    run_rows = [(i, row) for i, row in enumerate(actual['runs'])
                if row.get('route') == 's_channels']
    forward_rows = [row for row in expected['runs'] if row.get('route') == 's_channels']
    assert len(run_rows) == len(forward_rows) == 1
    row_index, run_row = run_rows[0]
    forward_row = forward_rows[0]
    assert 'per_drive' not in expected
    # (d) Top-level record selection, each side by its definition: run's S
    # comes from its separate drives, so its main-record port records are
    # skipped as unread; forward scores them (decision C).
    assert actual.get('run_records_scored') is True and 'run_records_scored' not in expected
    assert actual['scored_port_record_ids'] == [] and expected['scored_port_record_ids'] == port_ids
    assert [(row['id'], row['reason']) for row in actual['skipped_port_records']] == [
        (port_id, 'S not read') for port_id in port_ids]
    assert expected['skipped_port_records'] == []
    assert all('per_drive' not in row for row in expected['runs'])
    leaves = []
    # The top level repeats per_drive only when the S row is the worst row.
    places = ((('settling_witness', actual),) if 'per_drive' in actual else ()) + (
        (f'settling_witness/runs/{row_index}', run_row),)
    for prefix, witness in places:
        # (a) One witness per impedance port, indexed in declaration order.
        assert len(witness['per_drive']) == len(ports), prefix
        for column, ((_, port), port_id, drive) in enumerate(
                zip(ports, port_ids, witness['per_drive'], strict=True)):
            location = f'{prefix}/per_drive/{column}'
            assert drive['route'] == 's_channels', (location, port_id)
            assert drive['scored_port_record_ids'] == port_ids, (location, port_id)
            if port.excite:
                # (b) Compare the port-record row, not the probe aggregate.
                shared = drive.keys() & forward_row.keys()
                assert {'db', 'status', 'error_per_bin', 'share_per_bin'} <= shared, (location, shared)
                for key in shared:
                    _assert_same_witness_value(drive[key], forward_row[key],
                                               f'{location}/{key}: {port_id}')
            else:
                # (c) This ordinal is the passive port's run-only drive.
                assert np.isnan(np.asarray(forward.s_params)[:, column]).all()
            # Both stored representations must retain the same drive fields.
            if witness is run_row and 'per_drive' in actual:
                assert drive.keys() == actual['per_drive'][column].keys(), location
                for key in drive:
                    _assert_same_witness_value(drive[key], actual['per_drive'][column][key],
                                               f'{location}/{key}')
            leaves.extend(f'{location}/{key}' for key in ('error_per_bin', 'share_per_bin'))
    return tuple(leaves)


def port_s_pair(declaration, path, *, n_steps=150):
    """Equal bin requests; run is the independent reference for measured columns."""
    sim = BUILDERS[declaration](path)
    # One record gives one driven column; independent drive solves make
    # columns of non-excited ports run-only BY CONSTRUCTION.
    RUN_ONLY_PORT_COLUMNS = tuple(k for k, port in enumerate(
        pe for pe in sim._ports if pe.impedance != 0) if not port.excite)
    run = invoke(sim, 'run', path, n_steps)
    if run.s_params is None:
        # Explicit graded lumped S is currently refused by the public API.
        with pytest.raises(NotImplementedError, match='Single-cell lumped-port S-parameters require the uniform reference lane'):
            invoke(sim, 'run', path, 1, dict(compute_s_params=True,
                    s_param_freqs=np.array([2e9, 5e9, 8e9])))
        return None
    forward = invoke(BUILDERS[declaration](path), 'forward', path, n_steps,
                     dict(port_s11_freqs=np.asarray(run.freqs)))
    expected, actual = np.asarray(run.s_params), np.asarray(forward.s_params)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(forward.freqs, run.freqs)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=0,
                               atol=1e-4 * np.max(np.abs(expected)))
    for column in RUN_ONLY_PORT_COLUMNS:
        assert np.isnan(actual[:, column]).all()
    return {'run': {'outcome': 'admitted', 'result': run},
            'forward': {'outcome': 'admitted', 'result': forward}}
