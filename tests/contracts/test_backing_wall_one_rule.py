"""1549: independently declared backing faces and run/forward kernel identity."""
import rfx.stepping.uniform_loop as uniform_loop
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
import gc
import inspect
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.depths import Kind
from rfx.boundaries.model import realize
from rfx.boundaries.spec import Boundary, BoundarySpec

# Classes W/R remain separate rows so a leader can explicitly defer either.
NAMED_EXCEPTIONS = {}
CASES = ('absorbing', 'mixed', 'pmc', 'periodic', 'asymmetric',
         'plane', 'closed_box', 'W-guide')
PATHS = ('uniform', 'graded', 'distributed-uniform', 'distributed-graded')
ALL_FACES = frozenset(f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi'))
# Entry-specific observer lists are not boundary operators. These are the
# exclusions from the survey's broad kernel-context capture, by exact name.
MONITOR_EXCLUSIONS = ('prb_meta', 'dft_meta', 'flux_meta', 'waveguide_meta',
                      'wire_sparam_meta', 'wire_refplane_meta', 'lumped_sparam_meta')
BOUNDARY_INPUTS = ('periodic', 'pec_axes', 'pec_faces_frozen', 'pmc_faces_frozen',
                  'curl_boundary', 'cpml_axes', 'cpml_params', 'upml_coeffs',
                  'pec_edge_masks', 'pec_static_edge_masks',
                  'conformal_weights', 'sheet_impedance', 'bloch', 'tfsf_cfg',
                  'use_cpml', 'use_upml', 'use_pec_faces', 'use_pmc_faces',
                  'use_pec_edges', 'use_pec_occupancy', 'use_sheet_impedance')


class Captured(Exception):
    """The factory is built; no field step may execute."""


def model(case, path='uniform'):
    boundary = {
        'absorbing': 'cpml',
        'mixed': BoundarySpec(x='cpml', y='pec', z='pec'),
        'pmc': BoundarySpec(x='cpml', y='pmc', z='pec'),
        'periodic': BoundarySpec(x='cpml', y='periodic', z='pec'),
        'asymmetric': BoundarySpec(x=Boundary(lo='cpml', hi='pec'), y='pec', z='pec'),
        'plane': 'cpml', 'closed_box': 'cpml',
        'W-guide': BoundarySpec(x='cpml', y='pec', z='pec'),
    }[case]
    mesh = {'dz_profile': np.full(10, .001)} if path.endswith('graded') else {}
    sim = Simulation(20e9, (.04, .02, .01), dx=.001, cpml_layers=4,
                     boundary=boundary, **mesh)
    if case in ('plane', 'closed_box'):
        sim.add_tfsf_source(f0=10e9, margin=3, closed_box=case == 'closed_box')
    elif case == 'W-guide':
        sim.add_waveguide_port(.01, f0=10e9, freqs=np.array([10e9]),
                              probe_offset=2, ref_offset=1)
    else:
        sim.add_source((.01, .01, .005), 'ez', amplitude_kind='field')
    sim.add_probe((.025, .01, .005), 'ez')
    return sim


def expected_faces(sim, grid, case, path):
    record = realize(sim.boundary_model(), grid)
    periodic = {a for a, yes in zip('xyz', sim._periodic_flags()) if yes}
    # The plane-wave declaration chooses a wrapped transverse pair on the
    # uniform path; graded paths install no periodic stencil (Addendum 5e).
    if case == 'plane' and not path.endswith('graded'):
        periodic.update('yz')
    return frozenset(f.name for f in record.faces
                     if f.kind in (Kind.PEC, Kind.ABSORBER) and f.name[0] not in periodic)


def freeze(value):
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).copy()
    if is_dataclass(value):
        return {f.name: freeze(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return {k: freeze(v) for k, v in value._asdict().items()}
    if isinstance(value, dict):
        return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    return value


def assert_equal(left, right, path='boundary'):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray), path
        assert left.dtype == right.dtype, f'{path}: dtype {left.dtype} != {right.dtype}'
        assert np.array_equal(left, right), f'{path}: arrays differ (shape {left.shape} / {right.shape})'
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), f'{path}: keys differ'
        for key in left:
            assert_equal(left[key], right[key], f'{path}/{key}')
    elif isinstance(left, tuple):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f'{path}/{index}')
    else:
        assert left == right, f'{path}: {left!r} != {right!r}'


def closure_values(fun):
    values = {}
    for name, cell in zip(fun.__code__.co_freevars, fun.__closure__ or ()):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            pass  # inactive branch's uninitialized cell
    return values


def stage(fun, name):
    """The nested function ``name`` reached through ``fun``'s closures: the stage the step itself runs."""
    seen, todo = set(), [fun]
    while todo:
        f = todo.pop()
        if id(f) in seen or not hasattr(f, '__code__'):
            continue
        seen.add(id(f))
        if f.__name__ == name:
            return f
        for value in closure_values(f).values():
            items = value.values() if isinstance(value, dict) else value if isinstance(value, (tuple, list)) else (value,)
            todo.extend(item for item in items if callable(item))
    raise AssertionError(f'no stage {name!r} under {fun.__name__}')


def capture(sim, entry, path, invoke=None):
    import rfx.simulation as kernel
    import rfx.nonuniform as nu
    captured = {}
    setup_original = uniform_loop._build_step_setup
    nu_original = nu._build_nu_scan
    jit_original = jax.jit
    grid_original = sim._build_nonuniform_grid
    from rfx.runners import _distributed_nu_wrapped as wrapped
    wrapped_original = wrapped.wrap

    def built_grid(*args, **kwargs):
        grid = grid_original(*args, **kwargs)
        captured['grid'] = grid
        return grid

    def setup(*args, **kwargs):
        result = setup_original(*args, **kwargs)
        captured['psi'] = freeze(result.carry_init.get('cpml'))
        bound = inspect.signature(setup_original).bind(*args, **kwargs)
        bound.apply_defaults()
        captured['sheets'] = freeze(bound.arguments['pec_sheets'])
        return result

    def factory(ctx, *args, **kwargs):
        captured.update(grid=ctx.grid, faces=ctx.pec_faces_frozen, backend='uniform')
        # The conformal weights the step reads ride on the realized material
        # model's tensor operands (M4b), not on the context itself.
        tensor = ctx.materials.components.tensor
        held = {'conformal_weights': tensor.conformal_weights if tensor else None}
        captured['boundary'] = {name: freeze(held[name] if name in held else getattr(ctx, name))
                                for name in BOUNDARY_INPUTS}
        # Capture the shared factory's derived absorber/material boundary
        # coefficients too; this builds invariants, never advances fields.
        invariants = uniform_loop.core_step_invariants(ctx)
        captured['invariants'] = {k: freeze(v) for k, v in invariants.items() if k != 'ctx'}
        raise Captured

    def nu_factory(*args, **kwargs):
        result = nu_original(*args, **kwargs)
        # The graded step is composed from stages (S3-2 stage B); read the set its wall stage applies.
        wall_stage = closure_values(stage(result.step_fn_inv, 'pec'))
        captured.update(grid=args[0], faces=wall_stage['ctx']._pec_faces_frozen, backend='graded')
        raise Captured

    def wrapped_factory(*args, **kwargs):
        # The wrapped multi-device step (#1776, plain CPML since #1402 item 5) rebuilds
        # the one-device graded step per slab from the context wrap() closes over.
        # Build that step from that context and read the faces its wall stage applies.
        from rfx.stepping.graded import make_graded_step
        advance = wrapped_original(*args, **kwargs)
        seen, todo, contexts = set(), [advance], []
        while todo:
            f = todo.pop()
            if id(f) in seen or not hasattr(f, '__code__'):
                continue
            seen.add(id(f))
            values = closure_values(f)
            if 'context' in values:
                contexts.append(values['context'])
            todo.extend(v for v in values.values() if callable(v))
        assert len({id(c) for c in contexts}) == 1, contexts
        wall_stage = closure_values(stage(make_graded_step(contexts[0]), 'pec'))
        assert 'grid' in captured
        captured.update(faces=wall_stage['ctx']._pec_faces_frozen, backend='distributed-graded')
        raise Captured

    def distributed_factory(fun=None, *args, **kwargs):
        # Both distributed runners build their complete step closure before
        # handing the driver to jit. Stop at that builder boundary, before
        # even tracing a field update. The recorded free variable is the
        # actual pec_faces argument of the CPML step's wall application.
        if fun is not None and fun.__module__ == 'rfx.runners.distributed_v2' and fun.__name__ in ('_run_cpml', '_run_pec'):
            driver = closure_values(fun)
            step = driver['step_fn_cpml' if fun.__name__ == '_run_cpml' else 'step_fn_pec']
            bound = closure_values(step)
            frame = inspect.currentframe().f_back
            captured.update(grid=frame.f_locals['grid'],
                            faces=bound['e_wall_faces'] if fun.__name__ == '_run_cpml' else bound['_pec_faces_frozen'],
                            backend='distributed-uniform')
            raise Captured
        if fun is not None and fun.__module__ == 'rfx.runners.distributed_nu' and fun.__name__ == 'run_fn':
            frame = inspect.currentframe().f_back
            # The step passes the runner's e_wall_faces (#1221 B4). Resolve it inside
            # the unmodified wall factory and stop at its shard-map builder;
            # no state is read and no field operation is traced or executed.
            from rfx.runners.distributed_nu import _apply_pec_face_nu_shmap
            from rfx.runners import _distributed_common as walls
            assert 'grid' in captured
            captured['backend'] = 'distributed-graded'

            def wall_factory(body, **kwargs):
                captured['faces'] = closure_values(body)['pec_faces']
                raise Captured

            with patch.object(walls, 'rank_shard_map', wall_factory):
                _apply_pec_face_nu_shmap(None, frame.f_locals['mesh'],
                    frame.f_locals['n_devices'], frame.f_locals['nx_local'],
                    pad_x=frame.f_locals['pad_x'], pec_faces=frame.f_locals['e_wall_faces'], ranks=None)
            raise AssertionError('distributed wall factory was not observed')
        return jit_original(fun, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('backing-wall factory judge reached a field scan')

    with ExitStack() as stack:
        for obj, name, replacement in ((sim, '_build_nonuniform_grid', built_grid),
                                       (uniform_loop, '_build_step_setup', setup),
                                       (uniform_loop, 'make_core_step', factory),
                                       (nu, '_build_nu_scan', nu_factory),
                                       (wrapped, 'wrap', wrapped_factory),
                                       (jax, 'jit', distributed_factory),
                                       (jax.lax, 'scan', forbidden)):
            stack.enter_context(patch.object(obj, name, replacement))
        with pytest.raises(Captured):
            if invoke is not None:
                invoke()
            else:
                kw = {'n_steps': 1, 'skip_preflight': True}
                if entry == 'forward':
                    kw['checkpoint'] = False
                if path.startswith('distributed'):
                    assert len(jax.devices('cpu')) >= 2
                    kw['devices'] = jax.devices('cpu')[:2]
                    if entry == 'forward':
                        kw['distributed'] = True
                getattr(sim, entry)(**kw)
    return captured


def refusal(case, entry, path):
    if case == 'closed_box' and path != 'uniform':
        return 'uniform, single-device'
    if case == 'periodic' and path.startswith('distributed'):
        return 'periodic'
    if case in ('plane', 'closed_box') and path == 'distributed-graded':
        return 'TFSF'
    if case == 'W-guide' and path == 'distributed-graded' and entry == 'forward':
        return 'Waveguide|waveguide'
    return None


@pytest.fixture(autouse=True)
def release_builds():
    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('path', PATHS)
def test_wall_matrix(case, entry, path):
    reason = refusal(case, entry, path)
    if reason:
        with pytest.raises((ValueError, NotImplementedError), match=reason):
            capture(model(case, path), entry, path)
        return
    sim = model(case, path)
    got = capture(sim, entry, path)
    expected = expected_faces(sim, got['grid'], case, path)
    assert got['faces'] == expected, f'{case}/{entry}/{path}: actual walls {sorted(got["faces"])} != declared {sorted(expected)}'


@pytest.mark.parametrize('case', (*CASES, 'tube'))
def test_uniform_boundary_inputs_equal(case):
    pair = [capture(tube() if case == 'tube' else model(case), entry, 'uniform')
            for entry in ('run', 'forward')]
    for name in ('faces', 'boundary', 'psi', 'invariants', 'sheets'):
        assert_equal(pair[0][name], pair[1][name], name)


def tube():
    """Issue 1549's 28-line model, unchanged geometry and 20 layers."""
    dx, nx, layers = 25e-6, 2, 20
    sim = Simulation(freq_max=10e9, domain=(nx*dx, 8*dx, 11*dx), dx=dx,
                     cpml_layers=layers, boundary=BoundarySpec(x='cpml', y='pec', z='pec'))
    for y in (2*dx, 5*dx):
        sim.add(Box((0, y, 2*dx), (nx*dx, y, 7*dx)), material='pec')
    for z in (2*dx, 7*dx):
        sim.add(Box((0, 2*dx, z), (nx*dx, 5*dx, z)), material='pec')
    sim.add_source((1.23*dx, 3*dx, 5*dx), 'ez', amplitude_kind='field',
                   waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    sim.add_probe((dx, 3*dx, 5*dx), 'ez')
    sim.add_probe(((nx+layers)*dx, 4*dx, 4*dx), 'ez')
    return sim


def test_issue_probe_records_equal():
    records = [np.asarray(getattr(tube(), entry)(n_steps=600, skip_preflight=True,
                                               checkpoint=False).time_series)
               for entry in ('run', 'forward')]
    assert records[0].dtype == records[1].dtype
    # run() against forward() is the same physics under another trace: a per-step record may differ by the
    # cross-trace bar (9 float32 ULP at the record's peak), not more. That bar alone does not see the defect: over
    # 600 steps the unbacked forward() differed by 8.3e-07 against a record peak of 1.40 (bar 1.07e-06), all of it
    # on the second probe, which sits on the x_hi conducting plane.
    from tests.contracts.path_equivalence.comparison import compare
    compare(records[0], records[1], record='1549 probes, run vs forward', kind='step', measurements=[])
    # The witness: tangential E on the conducting plane is exactly zero on every entry (nonzero on the base tree's
    # forward()). The 7000-step pulse test below is the other witness (post-pulse peak 9.8e5 on the base tree).
    for entry, record in zip(('run', 'forward'), records):
        assert np.count_nonzero(record[:, 1]) == 0, f'{entry}: Ez on the x_hi conducting plane is not zero'


def test_issue_pulse_never_grows():
    res = tube().forward(n_steps=7000, skip_preflight=True, checkpoint=False)
    a = np.asarray(res.time_series)
    peak_end = int(np.ceil(4/(5e9*1.6*np.pi*res.dt)))
    peak = np.max(np.abs(a[:peak_end]))
    after = np.max(np.abs(a[peak_end:]))
    assert np.isfinite(a).all()
    assert after <= peak, f'1549 post-pulse peak {after} exceeds pulse peak {peak}; window ends {peak_end}'


@pytest.mark.parametrize('normalize,n_modes', [(False, 1), (True, 1), ('flux', 1),
                                              (False, 2), ('flux', 2)])
def test_class_W_calculator(normalize, n_modes):
    sim = Simulation(30e9, (.04, .02, .01), dx=.001, cpml_layers=4,
                     boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'})
    for position, direction in ((.01, '+x'), (.03, '-x')):
        sim.add_waveguide_port(position, direction=direction, f0=20e9,
                              freqs=np.array([20e9]), n_modes=n_modes,
                              probe_offset=2, ref_offset=1)
    got = capture(sim, 'calculator', 'uniform', invoke=lambda: sim.compute_waveguide_s_matrix(
        n_steps=1, normalize=normalize))
    assert got['faces'] == expected_faces(sim, got['grid'], 'W-guide', 'uniform')


def test_class_R_oblique_calculator():
    from rfx.core.yee import init_materials
    from rfx.rcs import compute_rcs
    sim = Simulation(15e9, (.06, .08, .006), dx=.002, cpml_layers=6, boundary='cpml')
    grid = sim._build_grid()
    got = capture(sim, 'calculator', 'uniform', invoke=lambda: compute_rcs(
        grid, init_materials(grid.shape), n_steps=1, f0=5e9, bandwidth=.5,
        theta_inc=20., polarization='ez', theta_obs=np.array([np.pi/2]),
        phi_obs=np.array([np.pi]), freqs=np.array([5e9]), cpml_layers=6, tfsf_margin=6))
    record = realize(sim.boundary_model(), got['grid'])
    expected = frozenset(f.name for f in record.faces if f.kind in (Kind.PEC, Kind.ABSORBER)
                         and f.name[0] != 'z')  # Method B wraps only z.
    assert got['faces'] == expected
