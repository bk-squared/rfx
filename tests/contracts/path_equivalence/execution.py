"""Run S0 declarations and retain every record-level failure and timing."""
from dataclasses import fields, is_dataclass
import functools
import sys
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from rfx import _realized
from rfx.runners import _admission as admission

from .builders import BASE_ROWS, build, point
from .comparison import compare


@functools.lru_cache(maxsize=None)
def solve(row, lane, graded, steps, dt):
    started = time.perf_counter()
    sim = build(row, lane, graded=graded, dt=dt)
    grid = sim._build_realized_grid()
    geometry = sim.realized_geometry()
    kwargs = dict(n_steps=steps, skip_preflight=True)
    forward = lane.startswith('fwd_')
    if forward:
        run = sim.forward
        kwargs['checkpoint'] = False
        if row[0] == '_ports' and row[1] in ('lumped_port', 'passive_port', 'wire_port'):
            from .builders import FREQS
            kwargs['port_s11_freqs'] = FREQS
    else:
        run = sim.run
        kwargs['compute_s_params'] = cell_ports = row == ('_ports', 'wire_port')
        if cell_ports:
            from .builders import FREQS
            kwargs['s_param_freqs'] = FREQS
        if lane == 'run_distributed':
            devices = jax.devices('cpu')
            assert len(devices) == 2, 'S0 requires exactly two CPU devices'
            kwargs['devices'] = devices
    objective = gradient = None
    # Source normalization capture supports only a single declared port.
    # S0 observes materials and leaves every production source spec unchanged.
    with patch.object(_realized, 'sources', lambda grid, materials, specs, site, **kw: specs):
        with _realized.capture() as capture:
            if forward:
                eps = jnp.asarray(geometry.materials.eps_r)
                mask = eps != 1.0
                box = None
                if row[0] == '_current_moments':
                    # This monitor explicitly supports a bounded material
                    # design, and refuses a whole-grid traced override.
                    box = (point(4, 2, 2), point(6, 4, 3))
                    index = (lambda p: sim._pos_to_nu_index(grid, p)) if sim._uses_nonuniform_mesh else grid.position_to_index
                    lo, hi = map(index, box)
                    window = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
                    box_eps = eps[window]

                def loss(parameter):
                    design = jnp.where(mask, eps * parameter, eps)
                    if box is None:
                        out = sim.forward(eps_override=design, **kwargs)
                    else:
                        out = sim.forward(design_box=box, design_eps_override=jnp.where(
                            box_eps != 1.0, box_eps * parameter, box_eps), **kwargs)
                    return jnp.sum(out.time_series ** 2), out

                objective, gradient, result = jax.jvp(
                    loss, (jnp.float32(1.0),), (jnp.float32(1.0),), has_aux=True)
                jax.block_until_ready(gradient)
            else:
                result = run(**kwargs)
            jax.block_until_ready(result.time_series)
    assert capture.lane == lane, f're-routed: requested {lane}, observed {capture.lane}'
    assert result.time_series.shape == (steps, 2), f'probe record shape: {result.time_series.shape}'
    assert result.time_series.dtype == jnp.float32, f'probe dtype: {result.time_series.dtype}'
    return dict(grid=grid, geometry=geometry, result=result, capture=capture,
                objective=objective, gradient=gradient, elapsed=time.perf_counter()-started)


def _comparison(a, b, name, kind, report):
    try:
        if isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
            assert a == b, f'{name}: {a!r} vs {b!r}'
        else:
            compare(a, b, record=name, kind=kind, measurements=report['measurements'])
    except AssertionError as exc:
        report['failures'].append(str(exc))


def _tree(a, b, name, kind, report):
    if a is None and b is None:
        return  # Optional fields inside a present record.
    if a is None or b is None:
        _comparison(a, b, name, kind, report)
        return

    def container(value):
        if is_dataclass(value):
            return {f.name: getattr(value, f.name) for f in fields(value)}
        if hasattr(value, '_asdict'):
            return value._asdict()
        return value

    a, b = container(a), container(b)
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                report['failures'].append(f"{name}.{key}: record missing on {'A' if key not in a else 'B'}")
            else:
                _tree(a[key], b[key], f'{name}.{key}', kind, report)
    elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        for i in range(max(len(a), len(b))):
            if i >= len(a) or i >= len(b):
                report['failures'].append(f"{name}.{i}: record missing on {'A' if i >= len(a) else 'B'}")
            else:
                _tree(a[i], b[i], f'{name}.{i}', kind, report)
    elif isinstance(a, (dict, tuple, list)) or isinstance(b, (dict, tuple, list)):
        report['failures'].append(f'{name}: record structure differs: {type(a).__name__} vs {type(b).__name__}')
    else:
        _comparison(a, b, name, kind, report)


def _kernel_materials(capture):
    # H's consumption-site dump contains the actual cell material arrays.
    # Dispersive E updates have no _realized electric hook; H still consumes
    # and records their common MaterialArrays (including eps and sigma).
    magnetic = [r for r in capture.records if 'mu_h' in r]
    assert magnetic, 'kernel magnetic material record missing'
    owned = sorted((r for r in capture.records if 'owned_start' in r),
                   key=lambda r: int(r['owned_start']))
    if not owned:
        return magnetic[0]['materials']._asdict()
    end, chunks = 0, []
    for record in owned:
        start, count = int(record['owned_start']), int(record['owned_count'])
        assert start == end and count > 0, 'kernel slab coverage differs'
        end += count
        # E creates a special x-lo EPS/SIGMA view but leaves mu untouched.
        # Match H by all unmodified owned material rows, not a rank guess.
        candidates = [r for r in magnetic if all(
            np.array_equal(getattr(r['materials'], field)[1:1+count],
                           getattr(record['materials'], field)[1:1+count])
            for field in ('eps_r', 'sigma', 'mu_r'))]
        assert candidates, 'kernel H slab record missing'
        chunks.append((count, candidates[0]['materials']))
    output = {}
    for field in magnetic[0]['materials']._fields:
        values = [getattr(m, field) for _, m in chunks]
        if all(v is None for v in values):
            output[field] = None
        elif isinstance(values[0], tuple):
            output[field] = tuple(np.concatenate([v[a][1:1+n] for (n, _), v in zip(chunks, values)])
                                  for a in range(len(values[0])))
        else:
            output[field] = np.concatenate([v[1:1+n] for (n, _), v in zip(chunks, values)])
    return output


OBSERVERS = {
    '_dft_planes': 'dft_planes', '_flux_monitors': 'flux_monitors',
    '_ntff': 'ntff_data', '_current_moments': 'current_moment_data',
}
RECORDS = ('time_series', 'sparam_time_records', 'dft_planes', 'flux_monitors',
           'ntff_data', 'current_moment_data', 'lumped_port_sparams', 'wire_port_sparams', 's_params')


def _samples(value, row):
    if value is None:
        return None
    labels = ('V', 'I', 'V_port', 'V_ref') if row[1] == 'wire_port' else ('V', 'I', 'V_ref')
    return tuple({labels[i] if i < len(labels) else f'column_{i}': samples[:, i]
                  for i in range(samples.shape[1])} for samples in value)


def _port_dft(value, name):
    if value is None:
        return None
    labels = (('V', 'I', 'V_inc', 'V_port', 'V_ref') if name == 'wire_port_sparams'
              else ('V', 'I', 'V_ref'))
    return tuple({labels[i] if i < len(labels) else f'channel_{i}': channel
                  for i, channel in enumerate(accumulators)} for _, accumulators in value)


def _comparison_canary():
    # m1: disabling comparison must redden the actual matrix, including cells
    # which already have an expected geometry-record finding.
    try:
        compare(np.array([1.], dtype=np.float32), np.array([2.], dtype=np.float32),
                record='canary', kind='step', measurements=[])
    except AssertionError:
        return
    raise RuntimeError('S0 comparison disabled: changed record accepted')


def _entry_refusal(sim, cell):
    # A selector (e.g. a profile) normally routes away from the lane whose
    # refusal is under test. Select that explicit lane after the real routing
    # validation; retain all of its real preparation and admission checks.
    dispatch = sim._dispatch_plan
    scan = jax.lax.scan

    def select(**kwargs):
        return dispatch(**kwargs)._replace(lane=cell.refused)

    def before_scan(*args, **kwargs):
        caller = sys._getframe(1).f_code.co_filename.replace('\\', '/')
        if any(part in caller for part in ('rfx/simulation.py', 'rfx/nonuniform.py', 'rfx/runners/', 'rfx/progress.py')):
            raise RuntimeError('refusal reached the first kernel scan')
        return scan(*args, **kwargs)

    kwargs = dict(n_steps=cell.steps, skip_preflight=True)
    run = sim.forward if cell.refused.startswith('fwd_') else sim.run
    if cell.refused.startswith('fwd_'):
        kwargs['checkpoint'] = False
    else:
        kwargs['compute_s_params'] = False
    if cell.refused == 'run_distributed':
        kwargs['devices'] = jax.devices('cpu')[:2]
    with patch.object(sim, '_dispatch_plan', select), patch.object(jax.lax, 'scan', before_scan):
        try:
            run(**kwargs)
        except (NotImplementedError, ValueError) as exc:
            return str(exc)
    raise AssertionError(f'{cell.refused} entry did not refuse before stepping')


def execute(cell):
    started = time.perf_counter()
    report = dict(cell=cell.id, failures=[], measurements=[])
    try:
        _comparison_canary()
        if not cell.equivalence:
            # Exercise the production refusal gate before any time-step scan.
            try:
                sim = build(cell.row, cell.refused, graded=cell.graded)
            except (ValueError, NotImplementedError) as exc:
                targeted = (cell.row[0] in ('_dt_pin', '_dt_min_cell') and 'dt= pins' in str(exc)) or (cell.row[0] == '_floquet_ports' and 'Floquet ports do not support' in str(exc))
                if not targeted:
                    raise
                report['refusal'] = str(exc)
                return report
            assert admission.DETECTORS[cell.row](sim), f'builder did not activate {cell.row}'
            assert cell.row in admission.refused(sim, cell.refused), f'{cell.refused} failed to refuse {cell.row}'
            try:
                with patch.object(jax.lax, 'scan', side_effect=AssertionError('refusal reached a scan')):
                    admission.admit(sim, cell.refused)
            except NotImplementedError as exc:
                report['refusal'] = str(exc)
                if cell.steps:
                    report['entry_refusal'] = _entry_refusal(sim, cell)
                return report
            raise AssertionError(f'{cell.refused} failed to refuse {cell.row} before stepping')
        row = ('_materials', 'eps') if cell.row in BASE_ROWS else cell.row
        # The graded pair takes its first lane's computed time step. Other
        # pairs use the uniform lane's computed time step as declared.
        sim_a = build(row, cell.a, graded=cell.graded)
        dt = float(sim_a._build_realized_grid().dt)
        assert admission.DETECTORS[cell.row](sim_a), f'builder did not activate {cell.row}'
        a = solve(row, cell.a, cell.graded, cell.steps, None)
        pin_b = cell.graded or cell.b == 'run_nonuniform' or cell.b == 'fwd_nonuniform'
        b = solve(row, cell.b, cell.graded, cell.steps, dt if pin_b else None)
        _comparison(a['grid'].dt, b['grid'].dt, 'dt', 'exact', report)
        _comparison(a['result'].dt, b['result'].dt, 'result.dt', 'exact', report)
        _tree(a['geometry'].nodes, b['geometry'].nodes, 'nodes', 'exact', report)
        _tree(a['geometry'], b['geometry'], 'geometry', 'exact', report)
        try:
            _tree(_kernel_materials(a['capture']), _kernel_materials(b['capture']), 'kernel', 'exact', report)
        except AssertionError as exc:
            report['failures'].append(f'kernel: {exc}')
        for name in RECORDS:
            x, y = getattr(a['result'], name, None), getattr(b['result'], name, None)
            x = None if isinstance(x, (tuple, list, dict)) and not x else x
            y = None if isinstance(y, (tuple, list, dict)) and not y else y
            required = (name == 'time_series' or OBSERVERS.get(cell.row[0]) == name
                        or (cell.row[0] == '_ports' and cell.row[1] in ('lumped_port', 'passive_port', 'wire_port')
                            and name == 'sparam_time_records'))
            if x is None and y is None and not required:
                continue
            if name == 'sparam_time_records':
                x, y = _samples(x, cell.row), _samples(y, cell.row)
            elif name in ('wire_port_sparams', 'lumped_port_sparams'):
                x, y = _port_dft(x, name), _port_dft(y, name)
            if x is None or y is None:
                _comparison(x, y, name, 'step' if name in ('time_series', 'sparam_time_records') else 'accumulated', report)
            else:
                _tree(x, y, name, 'step' if name in ('time_series', 'sparam_time_records') else 'accumulated', report)
        if cell.a.startswith('fwd_'):
            for name in ('objective', 'gradient'):
                _comparison(a[name], b[name], name, 'accumulated', report)
    except Exception as exc:
        import traceback
        report['failures'].append(f'{type(exc).__name__}: {exc}')
        report['traceback'] = traceback.format_exc()
    finally:
        report['seconds'] = time.perf_counter()-started
    return report
