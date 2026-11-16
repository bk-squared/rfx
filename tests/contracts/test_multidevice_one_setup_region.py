"""One setup region for the multi-device lanes.

(a) No compile-time context or setup region is opened inside the multi-device lane
modules except the allowed sites named below: setup runs in the one region the
public entries open.

(b) Under a caller's ``jax.jit`` every host product is concrete, the advance is
staged, and every returned array equals the eager call of the same declaration,
bit for bit, except the leaves in the two named lists. The eager call is the
reference. A row runs in this process when it has enough CPU devices, otherwise
in a child process. The graded wire-port ``run(devices=, report_every=)`` rows
always run in a child process, so a compiler abort fails the row instead of
ending pytest.
"""
import ast
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.distributed

ROOT = Path(__file__).resolve().parents[2]
N_STEPS = 32
REPORT_EVERY = 4
CHILD_TIMEOUT = 600

REGION_FILES = ('rfx/stepping/slab*.py', 'rfx/stepping/uniform_slab_ports.py',
                'rfx/model/distributed_ports.py', 'rfx/runners/_distributed_*.py',
                'rfx/runners/distributed_nu.py', 'rfx/runners/distributed_v2.py')
CONTEXT_CALLS = ('ensure_compile_time_eval', 'declaration_setup', 'compile_time_switch')
ALLOWED_CONTEXT_SITES = {
    ('rfx/stepping/slab_setup.py', '_custom_rules_evaluate_concretely', 'ensure_compile_time_eval'):
        'the probe asks whether a custom_jvp rule returns a concrete value under '
        'ensure_compile_time_eval (it does not inside jax.vmap); the question needs the context',
}

TRACED_RECORD_NOT_JUDGED = ('settling_witness.*_per_bin', 'settling_witness.pole_contributions.*')
TRACED_RECORD_NOT_JUDGED_REASON = ('a traced record is not judged, and the run says so in an '
                                   'AccuracyWarning')
X64_OFF_RETURNS_32_BIT = {
    'grid.dx_arr_f64': 'one float32 ULP of |eager|',
    'grid.dy_arr_f64': 'one float32 ULP of |eager|',
    'grid.dz_f64': 'one float32 ULP of |eager|',
    'current_moment_monitor.centres': 'one float32 ULP of |eager|',
    'current_moment_monitor.block_key': 'equal values',
}
X64_OFF_RETURNS_32_BIT_REASON = ('JAX with x64 off returns a host float64/int64 array through a '
                                 'traced function as float32/int32')

GRADED_SCENES = ('point_source', 'lumped_port', 'wire_port', 'dft_plane', 'flux_monitor',
                 'current_moment')
GRADED_ENTRIES = ('forward', 'run', 'report_every')
UNIFORM_SCENES = ('dft_plane', 'flux_monitor', 'current_moment')
ABORT_ROWS = {('wire_port', 'graded', 'report_every')}
ROWS = ([(kind, 'graded', entry) for kind in GRADED_SCENES for entry in GRADED_ENTRIES]
        + [(kind, 'uniform', 'report_every') for kind in UNIFORM_SCENES])


def _context_calls(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                if name.name in CONTEXT_CALLS:
                    aliases[name.asname or name.name] = name.name
    found = []

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = scope + (node.name,)
        if isinstance(node, ast.Call):
            func = node.func
            kind = None
            if isinstance(func, ast.Attribute) and func.attr in CONTEXT_CALLS:
                kind = func.attr
            elif isinstance(func, ast.Name) and func.id in aliases:
                kind = aliases[func.id]
            if kind is not None:
                found.append(('.'.join(scope) or '<module>', kind, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return found


def test_no_setup_context_outside_the_region():
    files = sorted({p for pattern in REGION_FILES for p in ROOT.glob(pattern)})
    named = {p.relative_to(ROOT).as_posix() for p in files}
    assert {'rfx/stepping/slab_setup.py', 'rfx/model/distributed_ports.py',
            'rfx/runners/distributed_nu.py', 'rfx/runners/distributed_v2.py'} <= named, named
    sites = {}
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        for function, kind, line in _context_calls(path):
            sites.setdefault((rel, function.split('.')[-1], kind), []).append(line)
    unexpected = {site: lines for site, lines in sites.items() if site not in ALLOWED_CONTEXT_SITES}
    assert not unexpected, (
        'a compile-time context or setup region opened inside a multi-device lane module; '
        f'setup belongs in the one region the entries open: {unexpected}')
    for site, reason in ALLOWED_CONTEXT_SITES.items():
        assert len(sites.get(site, ())) == 1, (
            f'allowed site {site} ({reason}) is expected exactly once, found {sites.get(site)}')


def scene(kind, mesh, count):
    from rfx import GaussianPulse, Simulation
    from tests.contracts.test_multidevice_setup_boundary import scene as boundary_scene
    if kind in ('dft_plane', 'flux_monitor', 'current_moment'):
        if mesh == 'graded':
            from tests.unit.runners._graded_records_scene import radiating_model
            sim, _, _ = radiating_model('cpml', count, grading='z')
        else:
            sim = Simulation(freq_max=15e9, domain=(.015, .008, .008), dx=.001,
                             boundary='cpml', cpml_layers=2)
            sim.add_source((.0051, .0032, .0043), 'ez', amplitude_kind='field',
                           waveform=GaussianPulse(f0=7e9, bandwidth=.8))
        sim.add_probe((.006, .004, .004), 'ez')
        if kind == 'dft_plane':
            sim.add_dft_plane_probe(axis='y', coordinate=.004, freqs=[3e9, 7e9])
        elif kind == 'flux_monitor':
            sim.add_flux_monitor(axis='y', coordinate=.004, freqs=[3e9, 7e9])
        else:
            sim.add_current_moment_monitor((.002, .001, .001), (.012, .007, .007),
                                           block_size=.002, freqs=[3e9, 7e9])
        return sim
    if kind == 'wire_port':
        assert mesh == 'graded'
        sim = Simulation(freq_max=10e9, domain=(.016, .006, .006), dx=.001, boundary='pec',
                         dz_profile=np.array([.001] * 3 + [.0005] * 6))
        sim.add_port((.008, .003, .003), 'ez', impedance=50, extent=.002)
        sim.add_probe((.008, .003, .003), 'ez')
        return sim
    return boundary_scene('pec_box' if kind == 'point_source' else kind, mesh)


def returned_arrays(value, path=''):
    """Array leaves keyed by field name, or by insertion position for geometry masks."""
    from rfx.grid import Grid
    if isinstance(value, (jax.Array, jax.core.Tracer, np.ndarray)):
        return {path: value}
    if isinstance(value, Grid):
        value = vars(value)
    elif is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in fields(value)}
    elif hasattr(value, '_asdict'):
        value = value._asdict()
    if isinstance(value, dict):
        items = enumerate(value.values()) if path.endswith('geometry_masks') else value.items()
    elif isinstance(value, (tuple, list)):
        items = enumerate(value)
    else:
        return {}
    result = {}
    for key, item in items:
        result.update(returned_arrays(item, f'{path}.{key}' if path else str(key)))
    return result


@contextmanager
def observed_setup():
    """Record each host product's tracer leaves and whether the advance was staged."""
    from rfx import current_moments
    from rfx.measurement import flux_weights, monitors, setup
    from rfx.model import distributed_ports, source_coefficients
    from rfx.stepping import sharded
    events = dict(host=[], advance=[])

    def host_product(name, original):
        def wrapper(*args, **kwargs):
            value = original(*args, **kwargs)
            traced = sorted(p for p, a in returned_arrays(value).items()
                            if isinstance(a, jax.core.Tracer))
            events['host'].append(dict(name=name, traced=traced))
            return value
        return wrapper

    advance = sharded.recorded_sharded_call

    def staged(*args, **kwargs):
        value = advance(*args, **kwargs)
        events['advance'].append([isinstance(a, jax.core.Tracer)
                                  for a in returned_arrays(value).values()])
        return value

    with ExitStack() as stack:
        for owner, name in ((distributed_ports, 'assemble_graded_ports'),
                            (source_coefficients, 'make_current_source'),
                            (source_coefficients, 'current_source_samples'),
                            (monitors, 'default_frequency_grid'), (setup, 'build_planes'),
                            (flux_weights, 'build_flux_monitor_cfgs'),
                            (current_moments, 'monitor_for_simulation')):
            stack.enter_context(patch.object(owner, name, host_product(name, getattr(owner, name))))
        stack.enter_context(patch.object(sharded, 'recorded_sharded_call', staged))
        yield events


def _traced_record_not_judged(path):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in TRACED_RECORD_NOT_JUDGED)


def _leaf_rule(path, eager, actual):
    """None when the leaf passes; otherwise what differs."""
    if eager.shape != actual.shape:
        return dict(shape=[list(eager.shape), list(actual.shape)])
    if path in X64_OFF_RETURNS_32_BIT:
        if X64_OFF_RETURNS_32_BIT[path] == 'equal values':
            return None if np.array_equal(eager, actual) else dict(dtype=[str(eager.dtype), str(actual.dtype)])
        e = eager.astype(np.float64)
        bound = np.spacing(np.abs(eager).astype(np.float32)).astype(np.float64)
        excess = np.abs(actual.astype(np.float64) - e) - bound
        if not eager.size or np.all(excess <= 0):
            return None
        return dict(dtype=[str(eager.dtype), str(actual.dtype)], max_excess=float(np.max(excess)))
    if eager.dtype == actual.dtype and eager.tobytes() == actual.tobytes():
        return None
    detail = dict(dtype=[str(eager.dtype), str(actual.dtype)])
    if eager.dtype == actual.dtype and np.issubdtype(eager.dtype, np.number) and eager.size:
        detail['max_abs_diff'] = float(np.max(np.abs(eager.astype(np.complex128)
                                                      - actual.astype(np.complex128))))
    return detail


def measure(kind, mesh, entry, count):
    """Eager call, then a fresh declaration under jax.jit; returns what differs."""
    devices = jax.devices('cpu')[:count]
    assert len(devices) == count
    options = dict(n_steps=N_STEPS, skip_preflight=True, devices=devices)
    if entry == 'report_every':
        options['report_every'] = REPORT_EVERY

    def invoke(sim):
        if entry == 'forward':
            return returned_arrays(sim.forward(distributed=True, **options))
        return returned_arrays(sim.run(compute_s_params=kind in ('lumped_port', 'wire_port'), **options))

    row = dict(kind=kind, mesh=mesh, entry=entry, devices=count)
    with observed_setup() as eager_events:
        sim = scene(kind, mesh, count)
        eager = {p: np.array(a, copy=True) for p, a in jax.block_until_ready(invoke(sim)).items()}
    print('C4ROW-EAGER ' + json.dumps(dict(row, arrays=len(eager))), flush=True)
    sim = scene(kind, mesh, count)
    jax.clear_caches()
    with observed_setup() as jit_events:
        try:
            actual = jax.block_until_ready(jax.jit(lambda: invoke(sim))())
        except Exception as exc:
            row['jit_error'] = f'{type(exc).__name__}: {exc}'
            return row
    row.update(
        jit_error=None,
        tracer_leaves=sorted(p for p, a in actual.items() if isinstance(a, jax.core.Tracer)),
        eager_host=sorted({e['name'] for e in eager_events['host']}),
        jit_host=sorted({e['name'] for e in jit_events['host']}),
        traced_host={e['name']: e['traced'] for e in jit_events['host'] if e['traced']},
        eager_advance_traced=any(any(a) for a in eager_events['advance']),
        jit_advance=[bool(a) and all(a) for a in jit_events['advance']],
        only_eager=sorted(eager.keys() - actual.keys()),
        only_jit=sorted(actual.keys() - eager.keys()))
    actual = {p: np.asarray(a) for p, a in actual.items() if not isinstance(a, jax.core.Tracer)}
    compared, differ = 0, {}
    for path in sorted(eager.keys() & actual.keys()):
        if _traced_record_not_judged(path):
            continue
        compared += 1
        detail = _leaf_rule(path, eager[path], actual[path])
        if detail is not None:
            differ[path] = detail
    row.update(compared=compared, differ=differ)
    return row


def check(row):
    name = f"{row['kind']}-{row['mesh']}-{row['entry']}-{row['devices']}"
    assert row['jit_error'] is None, f'{name}: the call under a caller\'s jax.jit raised: {row["jit_error"]}'
    assert not row['tracer_leaves'], f'{name}: returned Tracers {row["tracer_leaves"]}'
    assert row['jit_host'] and row['jit_host'] == row['eager_host'], (
        f'{name}: host products under jit {row["jit_host"]} differ from eager {row["eager_host"]}')
    assert not row['traced_host'], f'{name}: host products hold Tracers under jit: {row["traced_host"]}'
    assert not row['eager_advance_traced'], f'{name}: the eager advance returned Tracers'
    assert row['jit_advance'] and all(row['jit_advance']), (
        f'{name}: the advance ran at trace time under jit: {row["jit_advance"]}')
    unmatched = [p for p in row['only_eager'] + row['only_jit'] if not _traced_record_not_judged(p)]
    assert not unmatched, (f'{name}: eager and jit return different leaves {unmatched}; only '
                           f'TRACED_RECORD_NOT_JUDGED ({TRACED_RECORD_NOT_JUDGED_REASON}) may differ')
    assert row['compared'] > 0, f'{name}: nothing compared'
    x64 = {p: d for p, d in row['differ'].items() if p in X64_OFF_RETURNS_32_BIT}
    assert not x64, (f'{name}: X64_OFF_RETURNS_32_BIT ({X64_OFF_RETURNS_32_BIT_REASON}) allows '
                     f'one float32 ULP of |eager| (block_key: equal values); exceeded: {x64}')
    other = {p: d for p, d in row['differ'].items() if p not in X64_OFF_RETURNS_32_BIT}
    assert not other, (f'{name}: not bitwise equal to the eager call; excluded only '
                       f'TRACED_RECORD_NOT_JUDGED ({TRACED_RECORD_NOT_JUDGED_REASON}) and '
                       f'X64_OFF_RETURNS_32_BIT: {other}')


def child_row(kind, mesh, entry, count):
    print('C4ROW ' + json.dumps(measure(kind, mesh, entry, count)), flush=True)


def run_in_child(kind, mesh, entry, count):
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=3', PYTHONDONTWRITEBYTECODE='1')
    code = ('from tests.contracts import test_multidevice_one_setup_region as c; '
            f'c.child_row(*{(kind, mesh, entry, count)!r})')
    child = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT, text=True,
                           capture_output=True, timeout=CHILD_TIMEOUT)
    tail = (child.stdout + child.stderr)[-6000:]
    assert child.returncode == 0, (
        f'{kind}-{mesh}-{entry}-{count}: child exit {child.returncode} '
        f'(signal {-child.returncode if child.returncode < 0 else None})\n{tail}')
    rows = [json.loads(line[len('C4ROW '):]) for line in child.stdout.splitlines()
            if line.startswith('C4ROW ')]
    assert len(rows) == 1, tail
    return rows[0]


@pytest.mark.parametrize('count', [2, 3])
@pytest.mark.parametrize('kind,mesh,entry', ROWS)
def test_caller_jit_equals_eager(kind, mesh, entry, count):
    if (kind, mesh, entry) in ABORT_ROWS or len(jax.devices('cpu')) < count:
        row = run_in_child(kind, mesh, entry, count)
    else:
        row = measure(kind, mesh, entry, count)
        jax.clear_caches()
    check(row)
