"""Assembly/observable census for DESIGN.md, 2026-10-12; no product fixes.

No differences are classified here. Each entry uses a fresh public declaration.
Run the separate driver for records; importing this module runs no simulations.
"""
from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
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
    sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
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
    # Other accepted method: methodB; tests/contracts/test_tfsf_boundary_override.py:45.
    return base_scene(path, source=False).add_tfsf_source(method='methodB')


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
    # Substrate/strip and port arguments imitate tests/contracts/test_grouped_drives.py:204-209;
    # the base scene's domain and default waveform are retained.
    sim = base_scene(path)
    sim.add_material('sub', eps_r=2.2)
    sim.add(Box((0., 0., 0.), (.024, .020, .003)), material='sub')
    sim.add(Box((0., .010, .003), (.024, .014, .003)), material='pec')
    sim.add_msl_port(position=(.006, .012, 0.), width=.004, height=.003,
                     direction='+x', impedance=50., eps_r_sub=2.2,
                     n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    return sim


def build_p_waveguide(path):
    # Replaces boundary with y/z PEC; full aperture (default ranges).
    return base_scene(path, boundary=BoundarySpec(x='cpml', y='pec', z='pec')).add_waveguide_port(
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
    return base_scene(path).add_current_moment_monitor((.006, .004, .002), (.025, .016, .010), .004, np.array([10e9]))


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
    """Collect array-valued attributes, recursively naming nested result data."""
    found = {}
    active = set()
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
                items = ((key, getattr(value, key)) for key in value._fields)
            elif is_dataclass(value):
                items = ((f.name, getattr(value, f.name)) for f in fields(value))
            elif isinstance(value, dict):
                items = value.items()
            elif isinstance(value, (tuple, list)):
                items = enumerate(value)
            elif hasattr(value, '__dict__'):
                items = vars(value).items()
            else:
                return
            for name, item in items:
                walk(item, f'{path}/{name}' if path else str(name))
        finally:
            active.remove(id(value))
    walk(result, '')
    # NamedTuple result APIs may also expose computed array properties.
    for name in dir(type(result)):
        if isinstance(getattr(type(result), name), property):
            walk(getattr(result, name), name)
    return found


def run_l2(builder, entry, path, n_steps=150):
    try:
        result = invoke(builder(path), entry, path, n_steps)
        values = observables(result)  # np.asarray synchronizes every captured array.
        return {'outcome': 'admitted', 'observables': values}
    except Exception as exc:
        return exception_record(exc)


def compare_l2(left, right):
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
    return {'observables': compared, 'run_only': sorted(left.keys() - right.keys()),
            'forward_only': sorted(right.keys() - left.keys())}


def census_row(declaration, path):
    """Return JSON-compatible record; the external driver persists one per row."""
    started = time.monotonic()
    builder = BUILDERS[declaration]
    record = {'declaration': declaration, 'path': path, 'n_steps': 150, 'l1a': {}, 'l1b': {}}
    try:
        if path != 'multi-device-uniform':
            for level in ('a', 'b'):
                captures = {entry: capture(builder, entry, path, level) for entry in ('run', 'forward')}
                record['l1' + level] = {entry: {key: value for key, value in cap.items() if key != 'value'}
                                        for entry, cap in captures.items()}
                if all(cap['outcome'] == 'captured' for cap in captures.values()):
                    record['l1' + level]['diffs'] = compare(captures['run']['value'], captures['forward']['value'])
                else:
                    record['l1' + level]['diffs'] = None
                del captures
        results = {entry: run_l2(builder, entry, path) for entry in ('run', 'forward')}
        record['admission'] = {entry: {key: value for key, value in result.items() if key != 'observables'}
                               for entry, result in results.items()}
        if path != 'multi-device-uniform' and all(result['outcome'] == 'admitted' for result in results.values()):
            record['l2'] = compare_l2(results['run']['observables'], results['forward']['observables'])
        else:
            record['l2'] = None
        return record
    finally:
        jax.clear_caches()
        gc.collect()
        record['wall_seconds'] = time.monotonic() - started
