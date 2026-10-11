"""Assembly/observable census for DESIGN.md, 2026-10-12; no product fixes.

No differences are classified here. Each entry uses a fresh public declaration.
Run the separate driver for records; importing this module runs no simulations.
"""
from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import partial
import gc
import importlib
import inspect
import time
import traceback
from unittest.mock import patch

import jax
import numpy as np

from rfx import Simulation, Box, PolylineWire, DebyePole, LorentzPole, drude_pole
from rfx.boundaries.spec import Boundary, BoundarySpec

PATHS = ('uniform', 'graded', 'multi-device-graded', 'multi-device-uniform')
PROFILE = (1.2, 1.1, 1.0, .9, .8, .7, .6, .6, .7, .8, .9, 1.0, 1.1, .6)
SOURCE = (.011, .009, .0055)
PROBE = (.021, .012, .007)
BOX_LO = (.0143, .0072, .0033)
BOX_HI = (.0197, .0138, .0086)


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
    # DESIGN requests a constructor option. No such option exists at the base:
    # rfx/api/__init__.py:314-347 rejects it. The run-only option is at
    # rfx/api/_execute.py:3949; substituting that would change the entry call.
    # Committed test tests/unit/geometry/test_subpixel_nonuniform.py:66 uses
    # sim.run(..., subpixel_smoothing=True), not a constructor option.
    # Preserve this public construction attempt and record its refusal.
    sim = base_scene(path, subpixel_smoothing=True)
    sim.add_material('box', eps_r=4.)
    return sim.add(Box(BOX_LO, BOX_HI), material='box')


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
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    if isinstance(value, np.generic):
        return value.item()
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
            return {f.name: freeze(getattr(value, f.name), active) for f in fields(value)}
        if isinstance(value, tuple) and hasattr(value, '_fields'):
            return {name: freeze(getattr(value, name), active) for name in value._fields}
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


def invoke(sim, entry, path, n_steps):
    kwargs = {'n_steps': n_steps, 'skip_preflight': True}
    if entry == 'forward':
        kwargs['checkpoint'] = False
    if path.startswith('multi-device'):
        devices = jax.devices('cpu')
        if len(devices) != 2:
            raise CaptureFailure(f'exactly two CPU devices required; observed {len(devices)}')
        kwargs['devices'] = devices[:2]
        if entry == 'forward':
            kwargs['distributed'] = True
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


def capture(builder, entry, path, level, n_steps=150):
    """L1a and L1b are separate executions, each with a fresh Simulation."""
    kernel = importlib.import_module('rfx.simulation')
    nu = importlib.import_module('rfx.nonuniform')
    runner = importlib.import_module('rfx.runners.nonuniform')
    distributed = importlib.import_module('rfx.runners.distributed_nu')
    from rfx.api._execute import _ExecuteMixin
    captured = {}
    sites = {}
    def remember_site(key):
        frame = inspect.currentframe().f_back.f_back
        sites[key] = f'{frame.f_code.co_filename}:{frame.f_lineno}'
    def stop_at(fun, key):
        def stopped(*args, **kwargs):
            remember_site(key)
            captured[key] = freeze(bound_arguments(fun, args, kwargs))
            raise Captured
        return stopped
    def forbidden(*args, **kwargs):
        raise CaptureFailure('L1 reached jax.lax.scan')
    try:
        sim = builder(path)
        with ExitStack() as stack:
            stack.enter_context(patch.object(jax.lax, 'scan', forbidden))
            if level == 'a':
                owner, name = ((kernel, 'run') if path == 'uniform' else
                               (runner, 'run_nonuniform_path') if path == 'graded' else
                               (_ExecuteMixin, '_execute_distributed_nonuniform_from_materials'))
                stack.enter_context(patch.object(owner, name, stop_at(getattr(owner, name), 'arguments')))
            elif path == 'uniform':
                original_setup = kernel._build_step_setup
                def setup(*args, **kwargs):
                    result = original_setup(*args, **kwargs)
                    captured['carry_init'] = freeze(result.carry_init)
                    remember_site('carry_init')
                    return result
                def core(ctx, *args, **kwargs):
                    remember_site('ctx')
                    captured['ctx'] = freeze(ctx)
                    captured['invariants'] = freeze(kernel.core_step_invariants(ctx))
                    raise Captured
                stack.enter_context(patch.object(kernel, '_build_step_setup', setup))
                stack.enter_context(patch.object(kernel, 'make_core_step', core))
            elif path == 'graded':
                original = nu._build_nu_scan
                def build(*args, **kwargs):
                    remember_site('arguments')
                    captured['arguments'] = freeze(bound_arguments(original, args, kwargs))
                    captured['result'] = freeze(original(*args, **kwargs))
                    raise Captured
                stack.enter_context(patch.object(nu, '_build_nu_scan', build))
            else:
                name = 'run_nonuniform_distributed_pec'
                stack.enter_context(patch.object(distributed, name, stop_at(getattr(distributed, name), 'arguments')))
            invoke(sim, entry, path, n_steps)
        raise CaptureFailure('entry returned without reaching the requested L1 seam')
    except Captured:
        return {'outcome': 'captured', 'value': captured, 'seam_sites': sites}
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


def run_l2(builder, entry, path, n_steps=150):
    """Admission covers only the entry call; collection happens after both calls."""
    try:
        sim = builder(path)
    except Exception as exc:
        return {'outcome': 'not_called', 'construction': exception_record(exc)}
    try:
        result = invoke(sim, entry, path, n_steps)
    except Exception as exc:
        return exception_record(exc)
    return {'outcome': 'admitted', 'result': result}


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


def census_row(declaration, path, *, unavailable_captures=None):
    """Return JSON-compatible record; the external driver persists one per row."""
    started = time.monotonic()
    builder = BUILDERS[declaration]
    record = {'declaration': declaration, 'path': path, 'n_steps': 150, 'l1a': {}, 'l1b': {}}
    try:
        if path != 'multi-device-uniform':
            for level in ('a', 'b'):
                captures = {entry: (unavailable_captures[entry] if unavailable_captures and entry in unavailable_captures
                                    else capture(builder, entry, path, level)) for entry in ('run', 'forward')}
                record['l1' + level] = {entry: {key: value for key, value in cap.items() if key != 'value'}
                                        for entry, cap in captures.items()}
                if all(cap['outcome'] == 'captured' for cap in captures.values()):
                    record['l1' + level]['diffs'] = compare(captures['run']['value'], captures['forward']['value'])
                else:
                    record['l1' + level]['diffs'] = None
                del captures
        results = {entry: run_l2(builder, entry, path) for entry in ('run', 'forward')}
        record['admission'] = {entry: {key: value for key, value in result.items() if key != 'result'}
                               for entry, result in results.items()}
        if path != 'multi-device-uniform' and all(result['outcome'] == 'admitted' for result in results.values()):
            values = {entry: observables(result['result']) for entry, result in results.items()}
            record['l2'] = compare_l2(values['run']['arrays'], values['forward']['arrays'],
                                      left_raises=values['run']['raises'],
                                      right_raises=values['forward']['raises'])
        else:
            record['l2'] = None
        return record
    finally:
        jax.clear_caches()
        gc.collect()
        record['wall_seconds'] = time.monotonic() - started


# Part B keeps its row set separate from the Part A declarations and driver.
REBUILT_IDS = ('P-waveguide', 'PW-B', 'C-smooth', 'P-msl', 'O-cm', 'P-floquet')
E2_IDS = ('M-diel', 'M-lossy', 'M-mu', 'M-debye', 'M-lorentz', 'M-drude',
          'C-pecbox', 'C-sheet', 'C-film', 'C-wire', 'C-pinned', 'M-overlap', 'M-oriented')
E3_IDS = ('plain', 'B-pecsides', 'C-sheet', 'P-lumped', 'O-dft', 'O-flux-fallback')
E4_IDS = ('plain', 'B-pecsides', 'C-sheet', 'P-lumped')
PORT_TABLES = ('sources', 'probes', 'lumped_port_sparams', 'wire_port_sparams')


def build_b_sweep(declaration):
    """One-value box.eps_r sweep; public declaration methods only."""
    sim = (build_b_pecsides('uniform') if declaration == 'B-pecsides' else
           build_c_sheet('uniform') if declaration == 'C-sheet' else
           build_p_lumped('uniform') if declaration == 'P-lumped' else
           build_o_dft('uniform') if declaration == 'O-dft' else
           build_o_flux('uniform') if declaration == 'O-flux-fallback' else base_scene())
    sim.add_material('box', eps_r=4.)
    return sim.add(Box(BOX_LO, BOX_HI), material='box')


def build_b_raster(declaration, mesh):
    # Start from the same public declarations. The grid is an explicit input
    # of the measured functions; it is shared by identity between both calls.
    path = 'graded' if mesh == 'dual' else 'uniform'
    if declaration in BUILDERS:
        return BUILDERS[declaration](path)
    sim = base_scene(path)
    sim.add_material('box', eps_r=4.)
    if declaration == 'M-overlap':
        # Ordered replacement: tests/unit/materials/test_later_entry_replaces_poles.py:146.
        sim.add(Box(BOX_LO, BOX_HI), material='box')
        sim.add_material('later', eps_r=7., sigma=.25)
        return sim.add(Box((.017, .009, .005), (.022, .016, .009)), material='later')
    # CAD-style solid: tests/unit/geometry/test_oriented_box.py:35-42.
    from rfx import OrientedBox
    q = np.array([[1, 2, 2], [2, 1, -2], [-2, 2, -1]], dtype=np.float64) / 3
    return sim.add(OrientedBox((.017, .011, .006), (.006, .005, .004), q.T), material='box')


def build_b_topology(declaration):
    from rfx.topology import TopologyDesignRegion
    sim = (build_b_pecsides('uniform') if declaration == 'B-pecsides' else
           build_c_sheet('uniform') if declaration == 'C-sheet' else
           build_p_lumped('uniform') if declaration == 'P-lumped' else base_scene())
    sim.add_material('design', eps_r=4.)
    # Public region call: tests/contracts/path_equivalence/test_source_on_pec.py:239-243.
    region = TopologyDesignRegion(corner_lo=(.014, .007, .003), corner_hi=(.019, .012, .006),
                                  material_bg='air', material_fg='design')
    return sim, region


def build_b_fit(declaration):
    from rfx.material_fit import DebyeFitResult, LorentzFitResult
    # Public factory layout: tests/unit/autodiff/test_differentiable_material_fit_inputs.py:61-84.
    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = build_p_lumped('uniform')
        sim.add_material('box', eps_r=eps_inf, debye_poles=debye_poles, lorentz_poles=lorentz_poles)
        return sim.add(Box(BOX_LO, BOX_HI), material='box')
    if declaration == 'M-debye':
        guess = DebyeFitResult(4., [DebyePole(delta_eps=1., tau=1e-11)], 0.)
    elif declaration == 'M-lorentz':
        omega = 2 * np.pi * 8e9
        guess = LorentzFitResult(4., [LorentzPole(omega_0=omega, delta=.15 * omega, kappa=.4 * omega**2)], 0.)
    else:
        guess = LorentzFitResult(4., [drude_pole(2 * np.pi * 5e9, 1e9)], 0.)
    return factory, guess


def build_b_rcs(declaration, *, vacuum=False):
    wave, scatterer = declaration.split('__')
    sim = base_scene(source=False).add_tfsf_source(angle_deg=20 if wave == 'PW-oblique' else 0)
    if not vacuum:
        if scatterer == 'M-diel':
            sim.add_material('box', eps_r=4.)
            sim.add(Box(BOX_LO, BOX_HI), material='box')
        elif scatterer == 'C-pecbox':
            sim.add(Box(BOX_LO, BOX_HI), material='pec')
        else:
            sim.add_thin_conductor(Box((.014, .006, .004), (.020, .014, .004)))
    return sim


def _site(frame):
    return f'{frame.f_code.co_filename}:{frame.f_lineno}'


def _public_call(call):
    try:
        result = call()
    except (Captured, CaptureFailure):
        raise
    except Exception as exc:
        return _b_exception_record(exc), None
    return {'outcome': 'admitted'}, result


def _b_exception_record(exc):
    frame = traceback.extract_tb(exc.__traceback__)[-1]
    if frame.filename == __file__:
        raise CaptureFailure(f'{type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}') from exc
    return exception_record(exc)


def _scan_forbidden(*args, **kwargs):
    raise CaptureFailure('L1 reached jax.lax.scan')


def _strip_values(capture_result):
    return {k: v for k, v in capture_result.items() if k not in ('value', 'raw', 'observed')}


def _l1_pair(left, right):
    return {'run': _strip_values(left), 'forward': _strip_values(right),
            'diffs': compare(left['value'], right['value'])
            if 'value' in left and 'value' in right else None}


def _observable_pair(left, right):
    return compare_l2(left['arrays'], right['arrays'], left_raises=left['raises'], right_raises=right['raises'])


def _returned_inventory(value, path=''):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {k: v for name, child in value.items()
                for k, v in _returned_inventory(child, f'{path}/{name}').items()}
    if isinstance(value, (tuple, list)):
        return {k: v for i, child in enumerate(value)
                for k, v in _returned_inventory(child, f'{path}/{i}').items()}
    return {path: describe(value)}


def _result_observables(result):
    return observables(result)


def capture_b_e1(declaration, distributed_side, n_steps=150):
    """Every call is forwarded, including port-reference and fallback calls."""
    kernel = importlib.import_module('rfx.simulation')
    dist = importlib.import_module('rfx.runners.distributed_v2')
    original_run, original_dist, original_owner = kernel.run, dist.run_distributed, dist.Slab.owner
    values, sites, routes, active = [], [], [], []
    def scalar(*args, **kwargs):
        bound = bound_arguments(original_run, args, kwargs)
        values.append(freeze({key: bound[key] for key in PORT_TABLES}))
        sites.append(_site(inspect.currentframe().f_back))
        routes.append('uniform fallback' if distributed_side else 'uniform')
        return original_run(*args, **kwargs)
    def distributed(*args, **kwargs):
        active.append(False)
        try:
            return original_dist(*args, **kwargs)
        finally:
            active.pop()
    def owner(*args, **kwargs):
        frame = inspect.currentframe().f_back
        if active and not active[-1] and frame.f_code.co_name == 'run_distributed':
            active[-1] = True
            values.append(freeze({key: frame.f_locals[key] for key in PORT_TABLES if key in frame.f_locals}))
            sites.append(_site(frame))
            routes.append('distributed_v2 before slab sharding')
        return original_owner(*args, **kwargs)
    with patch.object(kernel, 'run', scalar), patch.object(dist, 'run_distributed', distributed), patch.object(dist.Slab, 'owner', owner):
        admission, _ = _public_call(lambda: invoke(BUILDERS[declaration]('uniform'), 'run',
                                                    'multi-device-uniform' if distributed_side else 'uniform', n_steps))
    if admission['outcome'] == 'admitted' and not values:
        raise CaptureFailure('E1 returned without source/probe table seam')
    return dict(admission, **({'value': values} if values else {}), seam_sites=sites, routes=routes)


def capture_b_e2(declaration, mesh):
    from dataclasses import replace
    from rfx.grid import Grid
    from rfx.geometry import rasterize_grid as raster
    from rfx.model.materials import assemble_cells
    from rfx.geometry.smoothing import continued_conductor_shape
    sim = build_b_raster(declaration, mesh)
    grid = (sim._build_nonuniform_grid() if mesh == 'dual' else
            Grid(freq_max=20e9, domain=(.030, .020, .012), dx=.0005, cpml_layers=4))
    if mesh == 'dual':
        # Exact dual-average coordinate rule: rfx/runners/nonuniform.py:70-84.
        nodes = raster.coords_from_nonuniform_grid(grid)
        widths = (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)
        axes = [np.asarray(x) + np.asarray(d) / 2 for x, d in zip(nodes[:3], widths)]
        for axis in axes:
            axis[-1] = axis[-2]
        geometry = [replace(entry, shape=continued_conductor_shape(sim, grid, entry.shape, entry=entry, unextendable=[]))
                    if sim._resolve_material(entry.material_name).sigma >= sim._PEC_SIGMA_THRESHOLD else entry
                    for entry in sim._geometry]
    else:
        # Fine coordinates already represent centres; committed test:
        # tests/unit/geometry/test_oriented_box.py:131-136.
        nodes = raster.coords_from_uniform_grid(grid)
        axes = [np.asarray(x) + grid.dx / 2 for x in nodes[:3]]
        geometry = sim._geometry
    coords = raster.GridCoords(*axes, grid.shape)
    captures = {}
    for side in ('run', 'forward'):
        sheets, wires, sheet_specs, masks, entries = [], [], [], [], []
        sites = []
        def call():
            if side == 'run':
                sites.append(f'{__file__}:{inspect.currentframe().f_lineno + 1}')
                out = assemble_cells(sim, grid, pec_sheets=sheets, pec_wires=wires, sheet_specs=sheet_specs,
                                     geometry_masks=masks, assembly_entries=entries)
                names = ('materials', 'debye_spec', 'lorentz_spec', 'pec_mask', 'pec_shapes', 'boundary_pec_shapes', 'kerr_chi3')
            else:
                sites.append(f'{__file__}:{inspect.currentframe().f_lineno + 1}')
                out = raster.rasterize_geometry(geometry, sim._resolve_material, coords, centres=coords,
                    pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD, sheets=sheets, wires=wires,
                    geometry_masks=masks, assembly_entries=entries)
                names = ('materials', 'debye_spec', 'lorentz_spec', 'pec_mask', 'pec_shapes', 'kerr_chi3')
            value = dict(zip(names, out))
            value.update(pec_sheets=sheets, pec_wires=wires, geometry_masks=masks, assembly_entries=entries)
            if side == 'run':
                value['sheet_specs'] = sheet_specs
            return freeze(value)
        with patch.object(jax.lax, 'scan', _scan_forbidden):
            admission, value = _public_call(call)
        captures[side] = dict(admission, **({'value': value} if value is not None else {}),
                              seam_sites=sites,
                              seam_definitions=['rfx/model/materials.py:41' if side == 'run' else 'rfx/geometry/rasterize_grid.py:1025'])
    return captures


def capture_b_e3(declaration, sweep_side, level, n_steps=150):
    if declaration == 'P-lumped':
        return capture_b_e3_multi(declaration, sweep_side, level, n_steps)
    kernel = importlib.import_module('rfx.simulation')
    sweep = importlib.import_module('rfx.vmap_sweep')
    original_setup, original_core, original_run = kernel._build_step_setup, kernel.make_core_step, kernel.run
    original_batch, original_scan = sweep._build_batched_materials, sweep._build_vmap_scan_fn
    original_fallback = sweep._sequential_fallback
    value, sites, details = {}, {}, {'lane': 'batched' if sweep_side else 'uniform'}
    def fallback(*args, **kwargs):
        details['lane'] = 'sequential fallback'
        return original_fallback(*args, **kwargs)
    def run(*args, **kwargs):
        bound = bound_arguments(original_run, args, kwargs)
        value.update(materials_element0=freeze(bound['materials']),
                     sources=freeze(bound['sources']), probes=freeze(bound['probes']),
                     dft_probes=freeze(bound['dft_planes']))
        sites['tables'] = _site(inspect.currentframe().f_back)
        raise Captured
    def batch(*args, **kwargs):
        out = original_batch(*args, **kwargs)
        value['materials_element0'] = freeze(jax.tree.map(lambda a: a[0], out))
        sites['materials_element0'] = _site(inspect.currentframe().f_back)
        return out
    def scan_builder(*args, **kwargs):
        bound = bound_arguments(original_scan, args, kwargs)
        for name in ('sources', 'probes', 'dft_probes', 'j_source_meta'):
            value[name] = freeze(bound[name])
        sites['tables'] = _site(inspect.currentframe().f_back)
        return original_scan(*args, **kwargs)
    def setup(*args, **kwargs):
        out = original_setup(*args, **kwargs)
        value['carry_init'] = freeze(out.carry_init)
        sites['carry_init'] = _site(inspect.currentframe().f_back)
        return out
    def core(ctx, *args, **kwargs):
        frame = inspect.currentframe().f_back
        if level == 'b':
            value['ctx'] = freeze(ctx)
            value['invariants'] = freeze(kernel.core_step_invariants(ctx))
            sites['ctx'] = _site(frame)
        if frame.f_code.co_name != 'run_one':
            raise Captured
        if level == 'a':
            value['prepared_j_waveforms'] = freeze(frame.f_locals['prepared_j_waveforms'])
            raise Captured
        # run_one builds init_carry AFTER make_core_step. Forward only this
        # builder; the scan patch captures its exact carry then raises.
        return original_core(ctx, *args, **kwargs)
    def scan(*args, **kwargs):
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == 'run_one' and 'ctx' in value:
            value['carry_init'] = freeze(args[1])
            sites['carry_init'] = _site(frame)
            raise Captured
        return _scan_forbidden(*args, **kwargs)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(jax.lax, 'scan', scan))
            stack.enter_context(patch.object(sweep, '_sequential_fallback', fallback))
            if level == 'a':
                stack.enter_context(patch.object(kernel, 'run', run))
                if sweep_side:
                    stack.enter_context(patch.object(sweep, '_build_batched_materials', batch))
                    stack.enter_context(patch.object(sweep, '_build_vmap_scan_fn', scan_builder))
                    stack.enter_context(patch.object(kernel, 'make_core_step', core))
            else:
                stack.enter_context(patch.object(kernel, '_build_step_setup', setup))
                stack.enter_context(patch.object(kernel, 'make_core_step', core))
            sim = build_b_sweep(declaration)
            if sweep_side:
                # tests/contracts/path_equivalence/test_source_on_pec.py:229.
                sweep.vmap_material_sweep(sim, 'box.eps_r', [4.], n_steps=n_steps)
            else:
                sim.run(n_steps=n_steps, skip_preflight=True)
        raise CaptureFailure('E3 returned without requested capture')
    except Captured:
        return dict(outcome='captured', value=value, seam_sites=sites, **details)
    except CaptureFailure:
        raise
    except Exception as exc:
        return dict(_b_exception_record(exc), **details)


def capture_b_e3_multi(declaration, sweep_side, level, n_steps):
    """The lumped-port fallback calls run twice; forward and record both."""
    kernel = importlib.import_module('rfx.simulation')
    from rfx.vmap_sweep import vmap_material_sweep
    original_run, original_setup, original_core = kernel.run, kernel._build_step_setup, kernel.make_core_step
    values, sites, carry = [], [], {}
    def run(*args, **kwargs):
        bound = bound_arguments(original_run, args, kwargs)
        values.append(freeze(dict(materials_element0=bound['materials'], sources=bound['sources'],
                                  probes=bound['probes'], dft_probes=bound['dft_planes'])))
        sites.append(_site(inspect.currentframe().f_back))
        return original_run(*args, **kwargs)
    def setup(*args, **kwargs):
        out = original_setup(*args, **kwargs)
        carry['value'] = freeze(out.carry_init)
        carry['site'] = _site(inspect.currentframe().f_back)
        return out
    def core(ctx, *args, **kwargs):
        values.append(dict(ctx=freeze(ctx), invariants=freeze(kernel.core_step_invariants(ctx)),
                           carry_init=carry['value']))
        sites.append(_site(inspect.currentframe().f_back))
        return original_core(ctx, *args, **kwargs)
    with ExitStack() as stack:
        if level == 'a':
            stack.enter_context(patch.object(kernel, 'run', run))
        else:
            stack.enter_context(patch.object(kernel, '_build_step_setup', setup))
            stack.enter_context(patch.object(kernel, 'make_core_step', core))
        admission, _ = _public_call(lambda: vmap_material_sweep(build_b_sweep(declaration), 'box.eps_r', [4.], n_steps=n_steps)
            if sweep_side else build_b_sweep(declaration).run(n_steps=n_steps, skip_preflight=True))
    if admission['outcome'] == 'admitted' and not values:
        raise CaptureFailure('E3 multi-call fallback returned without seam')
    return dict(admission, **({'value': values} if values else {}), seam_sites=sites,
                lane='sequential fallback' if sweep_side else 'uniform', forwarding=True)


def capture_b_e4(declaration, topology_side, exact=None):
    kernel = importlib.import_module('rfx.simulation')
    from rfx.api._execute import _ExecuteMixin
    from rfx.topology import topology_optimize
    original, original_forward = kernel.run, _ExecuteMixin._forward_from_materials
    value, inputs, sites = {}, {}, {}
    def forward(*args, **kwargs):
        bound = bound_arguments(original_forward, args, kwargs)
        inputs.update(eps=freeze(bound['materials'].eps_r), n_steps=bound['n_steps'], checkpoint=bound['checkpoint'])
        return original_forward(*args, **kwargs)
    def run(*args, **kwargs):
        value['arguments'] = freeze(bound_arguments(original, args, kwargs))
        sites['arguments'] = _site(inspect.currentframe().f_back)
        raise Captured
    try:
        sim, region = build_b_topology(declaration)
        with patch.object(jax.lax, 'scan', _scan_forbidden), patch.object(kernel, 'run', run):
            if topology_side:
                with patch.object(_ExecuteMixin, '_forward_from_materials', forward):
                    topology_optimize(sim, region, lambda result: -jax.numpy.sum(result.time_series**2),
                        n_iterations=1, init_density=jax.numpy.full((6, 6, 4), .5), verbose=False, skip_preflight=True)
            else:
                sim.forward(eps_override=jax.numpy.asarray(exact['eps'].array()), n_steps=exact['n_steps'],
                            checkpoint=exact['checkpoint'], skip_preflight=True)
        raise CaptureFailure('E4 returned without one forward evaluation')
    except Captured:
        return dict(outcome='captured', value=value, raw=inputs, seam_sites=sites)
    except CaptureFailure:
        raise
    except Exception as exc:
        return _b_exception_record(exc)


def _fit_call(declaration):
    from rfx.differentiable_material_fit import differentiable_material_fit
    factory, guess = build_b_fit(declaration)
    return differentiable_material_fit(factory, np.zeros((1, 1, 1), complex), np.array([10e9]),
        n_debye_poles=1 if declaration == 'M-debye' else 0,
        n_lorentz_poles=0 if declaration == 'M-debye' else 1,
        initial_guess=guess, n_iterations=1, learning_rate=0., checkpoint=True, verbose=False)


def _fixed_fit_scene(declaration):
    factory, guess = build_b_fit(declaration)
    return factory(guess.eps_inf, guess.poles if declaration == 'M-debye' else [],
                   [] if declaration == 'M-debye' else guess.poles)


def capture_b_e5(declaration, fit_side, level, *, reference_steps=150, fit_steps_override=None, fields_run=False):
    """Fit's initial evaluation. A step override must be leader-selected."""
    from rfx.grid import Grid
    kernel = importlib.import_module('rfx.simulation')
    original_run, original_setup = kernel.run, kernel._build_step_setup
    value, sites, observed = {}, {}, {}
    def run(*args, **kwargs):
        bound = bound_arguments(original_run, args, kwargs)
        value['arguments'] = freeze(bound)
        sites['arguments'] = _site(inspect.currentframe().f_back)
        if not fields_run:
            raise Captured
        result = original_run(*args, **kwargs)
        observed.update(observables(result))
        return result
    def setup(*args, **kwargs):
        out = original_setup(*args, **kwargs)
        value['carry_init'] = freeze(out.carry_init)
        sites['carry_init'] = _site(inspect.currentframe().f_back)
        return out
    def core(ctx, *args, **kwargs):
        value['ctx'] = freeze(ctx)
        value['invariants'] = freeze(kernel.core_step_invariants(ctx))
        sites['ctx'] = _site(inspect.currentframe().f_back)
        raise Captured
    try:
        with ExitStack() as stack:
            if fit_side and fit_steps_override is not None:
                stack.enter_context(patch.object(Grid, 'num_timesteps', lambda *a, **kw: fit_steps_override))
            if not fields_run:
                stack.enter_context(patch.object(jax.lax, 'scan', _scan_forbidden))
            if level == 'a' or fields_run:
                stack.enter_context(patch.object(kernel, 'run', run))
            else:
                stack.enter_context(patch.object(kernel, '_build_step_setup', setup))
                stack.enter_context(patch.object(kernel, 'make_core_step', core))
            admission, _ = _public_call(lambda: _fit_call(declaration) if fit_side else
                _fixed_fit_scene(declaration).run(n_steps=reference_steps, skip_preflight=True))
        if not fields_run and admission['outcome'] == 'admitted':
            raise CaptureFailure('E5 returned without evaluation seam')
        return dict(admission, **({'value': value} if value else {}), seam_sites=sites, observed=observed)
    except Captured:
        return dict(outcome='captured', value=value, seam_sites=sites)


def capture_b_e6(declaration, rcs_side, n_steps=150):
    """DESIGN correction: forward every seam call, match by call order."""
    kernel = importlib.import_module('rfx.simulation')
    rcs = importlib.import_module('rfx.rcs')
    values, observed, sites, admissions, ordinals = {}, {}, [], [], []
    reference_ordinal = 0
    original = kernel.run
    def run(*args, **kwargs):
        ordinal = len(values) if rcs_side else reference_ordinal
        values[str(ordinal)] = freeze({'arguments': bound_arguments(original, args, kwargs)})
        ordinals.append(ordinal)
        sites.append(_site(inspect.currentframe().f_back))
        result = original(*args, **kwargs)
        observed[str(ordinal)] = _result_observables(result)
        return result
    if rcs_side:
        # Caller-built MaterialArrays from the same cell assembler used by
        # the main pipeline. A later run() admission refusal must not prevent
        # calling compute_rcs(), nor become a refusal attributed to that entry.
        from rfx.grid import Grid
        from rfx.model.materials import assemble_cells
        try:
            sim = build_b_rcs(declaration)
            grid = Grid(freq_max=20e9, domain=(.030, .020, .012), dx=.001, cpml_layers=4)
            materials = assemble_cells(sim, grid, pec_sheets=[], pec_wires=[])[0]
        except Exception as exc:
            return dict(outcome='not_called', construction=exception_record(exc), seam_sites=[])
        with patch.object(rcs, 'run', run):
            # Observation arguments: tests/unit/grid/test_x64_scan_carry_dtypes.py:227-229.
            admission, _ = _public_call(lambda: rcs.compute_rcs(grid, materials, n_steps=n_steps,
                f0=10e9, cpml_layers=4, theta_inc=20 if declaration.startswith('PW-oblique') else 0,
                freqs=np.array([10e9]), theta_obs=np.array([np.pi / 2]), phi_obs=np.array([np.pi]),
                subtract_incident_reference=True))
    else:
        with patch.object(kernel, 'run', run):
            for reference_ordinal, vacuum in enumerate((False, True)):
                outcome, _ = _public_call(lambda: build_b_rcs(declaration, vacuum=vacuum).run(
                    n_steps=n_steps, skip_preflight=True))
                admissions.append(dict(outcome, ordinal=reference_ordinal, role='vacuum' if vacuum else 'scatterer'))
        admission = ({'outcome': 'admitted'} if all(a['outcome'] == 'admitted' for a in admissions)
                     else next(a for a in admissions if a['outcome'] != 'admitted'))
    return dict(admission, **({'value': values} if values else {}), observed=observed, seam_sites=sites,
                call_ordinals=ordinals, call_admission=admissions)


def census_b_row(entry, declaration, *, e5_mode=None):
    """JSON record; run=main reference, forward=Part B entry (no classification)."""
    started = time.monotonic()
    record = dict(entry=entry, declaration=declaration, sides={'run': 'main pipeline', 'forward': entry},
                  l1a={}, l1b={}, l2=None)
    try:
        if entry == 'E1':
            left, right = capture_b_e1(declaration, False), capture_b_e1(declaration, True)
            record['l1a'] = _l1_pair(left, right)
            record['admission'] = {side: {k: v for k, v in cap.items() if k in ('outcome', 'type', 'message', 'raising_site')}
                                   for side, cap in (('run', left), ('forward', right))}
        elif entry == 'E2':
            decl, mesh = declaration.rsplit('__', 1)
            caps = capture_b_e2(decl, mesh)
            record['l1a'] = _l1_pair(caps['run'], caps['forward'])
            record['admission'] = {side: {k: v for k, v in cap.items() if k in ('outcome', 'type', 'message', 'raising_site')}
                                   for side, cap in caps.items()}
            inventories = {side: _returned_inventory(cap.get('value')) for side, cap in caps.items()}
            record['returned_inventory'] = inventories
            record['one_side_only'] = {
                side: {name: value for name, value in inventories[side].items() if name not in inventories[other]}
                for side, other in (('run', 'forward'), ('forward', 'run'))}
        elif entry == 'E3':
            for level in ('a', 'b'):
                record['l1' + level] = _l1_pair(capture_b_e3(declaration, False, level), capture_b_e3(declaration, True, level))
            from rfx.vmap_sweep import vmap_material_sweep
            admission_l, result_l = _public_call(lambda: build_b_sweep(declaration).run(n_steps=150, skip_preflight=True))
            admission_r, result_r = _public_call(lambda: vmap_material_sweep(build_b_sweep(declaration), 'box.eps_r', [4.], n_steps=150))
            record['admission'] = {'run': admission_l, 'forward': admission_r}
            if result_l is not None and result_r is not None:
                left, right = observables(result_l), observables(result_r)
                # Only field observables carry the sweep axis. Parameter values
                # remain a one-sided returned observable, without slicing them.
                for name, value in list(right['arrays'].items()):
                    if name == 'time_series' or name.startswith('dft_planes/'):
                        right['arrays'][name] = freeze(value.array()[0])
                record['l2'] = _observable_pair(left, right)
        elif entry == 'E4':
            right = capture_b_e4(declaration, True)
            left = (capture_b_e4(declaration, False, right['raw']) if 'raw' in right else
                    {'outcome': 'not_called', 'reason': 'topology did not produce its exact eps input'})
            record['l1a'] = _l1_pair(left, right)
            record['admission'] = {side: _strip_values(cap) for side, cap in (('run', left), ('forward', right))}
            record['l2_not_run'] = 'E4 specifies one assembly evaluation, interrupted before optimizer update'
            if 'raw' in right:
                record['n_steps'] = right['raw']['n_steps']
                record['checkpoint'] = right['raw']['checkpoint']
        elif entry == 'E5':
            if e5_mode not in ('native', 'override150', 'no-l2', 'pending'):
                raise CaptureFailure('E5 step-count rule awaits leader selection')
            reference_steps = 150
            override = 150 if e5_mode == 'override150' else None
            if e5_mode == 'native':
                reference_steps = _fixed_fit_scene(declaration)._build_grid().num_timesteps(num_periods=20.)
            for level in ('a', 'b'):
                left = capture_b_e5(declaration, False, level, reference_steps=reference_steps)
                right = capture_b_e5(declaration, True, level, fit_steps_override=override)
                record['l1' + level] = _l1_pair(left, right)
                if level == 'a' and 'value' in right:
                    record['fit_n_steps'] = right['value']['arguments']['n_steps']
            record['step_count_rule'] = e5_mode
            record['reference_n_steps'] = reference_steps
            if e5_mode in ('no-l2', 'pending'):
                record['admission'] = {'run': {'outcome': 'not_measured'}, 'forward': {'outcome': 'not_measured'}}
                record['l2_not_run'] = ('fit has no public n_steps parameter; leader selected no L2' if e5_mode == 'no-l2' else
                    'Pending leader step-count rule: fit fixes num_periods=20 internally; Part A L2 specifies 150 steps. No substitute chosen.')
                record['incomplete'] = e5_mode == 'pending'
            else:
                left = capture_b_e5(declaration, False, 'a', reference_steps=reference_steps, fields_run=True)
                right = capture_b_e5(declaration, True, 'a', fit_steps_override=override, fields_run=True)
                record['admission'] = {side: {k: v for k, v in cap.items() if k in ('outcome', 'type', 'message', 'raising_site')}
                                       for side, cap in (('run', left), ('forward', right))}
                if left['outcome'] == right['outcome'] == 'admitted':
                    record['l2'] = _observable_pair(left['observed'], right['observed'])
        elif entry == 'E6':
            left, right = capture_b_e6(declaration, False), capture_b_e6(declaration, True)
            common = sorted(left.get('value', {}).keys() & right.get('value', {}).keys())
            paired = [{**cap, 'value': {key: cap['value'][key] for key in common}} for cap in (left, right)] if common else [
                {k: v for k, v in cap.items() if k != 'value'} for cap in (left, right)]
            record['l1a'] = _l1_pair(*paired)
            record['l1a']['matched_ordinals'] = common
            record['l1a']['unmatched_calls'] = {side: sorted(cap.get('value', {}).keys() - set(common))
                                              for side, cap in (('run', left), ('forward', right))}
            record['admission'] = {side: {k: v for k, v in cap.items() if k in ('outcome', 'type', 'message', 'raising_site', 'construction', 'call_admission')}
                                   for side, cap in (('run', left), ('forward', right))}
            matched_results = sorted(left.get('observed', {}).keys() & right.get('observed', {}).keys())
            record['l2_matched_ordinals'] = matched_results
            # Each main reference invocation has its own admission. A refused
            # scatterer call must not suppress a returned vacuum pair.
            if right['outcome'] == 'admitted' and matched_results:
                la = {f'{i}/{name}': arr for i, obs in left['observed'].items() for name, arr in obs['arrays'].items()}
                ra = {f'{i}/{name}': arr for i, obs in right['observed'].items() for name, arr in obs['arrays'].items()}
                le = {f'{i}/{name}': err for i, obs in left['observed'].items() for name, err in obs['raises'].items()}
                re = {f'{i}/{name}': err for i, obs in right['observed'].items() for name, err in obs['raises'].items()}
                record['l2'] = compare_l2(la, ra, left_raises=le, right_raises=re)
        else:
            raise ValueError(f'Part B entry not implemented: {entry}')
        return record
    finally:
        jax.clear_caches()
        gc.collect()
        record['wall_seconds'] = time.monotonic() - started
