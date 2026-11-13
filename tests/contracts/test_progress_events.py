"""Progress data: per-path accounting and raw-byte result identity (#2417).

Always-on CPU budget: estimate <120 s including the transport unit file.
Compiled multi-lane/measurement cases are slow; cluster A records durations.
"""
from dataclasses import fields, is_dataclass
import hashlib
import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.progress import events
from rfx.stepping.events import scope


def digest(result):
    arrays = []
    def visit(value):
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, 'dtype') and hasattr(value, 'shape'):
            a = np.asarray(value)
            arrays.append((str(a.dtype), a.shape, hashlib.sha256(a.tobytes()).hexdigest()))
        else:
            leaves = jax.tree_util.tree_leaves(value)
            if len(leaves) != 1 or leaves[0] is not value:
                for leaf in leaves:
                    visit(leaf)
    visit(result)
    assert arrays, 'the identity oracle must observe arrays'
    return arrays


def check_phases(rows, path, passes=1, deferred=False):
    plans = [r for r in rows if r['kind'] == 'plan']
    assert plans and plans[0]['phases'] == ['assemble', 'compile', 'step', 'post-process']
    assert len({r['run_id'] for r in rows}) == 1
    assert {r['path'] for r in rows} == {path}
    assert sum(len(r['passes']) for r in plans) == (passes if passes > 1 else 0)
    for index in range(1, passes + 1):
        batch = [r for r in rows if r['kind'] != 'plan' and r.get('pass_index', 1) == index]
        for phase in ['assemble', 'compile', 'step', 'post-process']:
            phase_rows = [r for r in batch if r['phase'] == phase]
            if phase == 'compile' or deferred:
                assert len(phase_rows) == 1, (phase, phase_rows)
                assert phase_rows[0]['kind'] == 'not_reported'
                assert phase_rows[0]['reported'] is False
                assert phase_rows[0]['reason_code'] == ('compile_not_separated' if phase == 'compile' else 'path_not_wired')
            else:
                assert phase_rows[0]['kind'] == 'phase_start', phase_rows
                assert phase_rows[-1]['kind'] == 'phase_end', phase_rows
                assert sum(r['kind'] == 'phase_start' for r in phase_rows) == 1
                assert sum(r['kind'] == 'phase_end' for r in phase_rows) == 1
    for row in rows:
        assert {'v', 't', 'kind', 'entry', 'path'} <= row.keys()
        assert row['v'] == 1
        wire = 'RFX_EVENT ' + json.dumps(row, separators=(',', ':'), ensure_ascii=True) + '\n'
        assert len(wire.encode()) <= 500
        if row['kind'] in ('progress', 'phase_end'):
            assert 'label' not in row and 'pass_label' not in row


def small_sim(graded=False):
    kw = {'dz_profile': [1e-3] * 3 + [.5e-3] * 6} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.006, .006, .006), dx=.001,
                     boundary='cpml', cpml_layers=2, **kw)
    sim.add_source((.003, .003, .003), 'ez', amplitude_kind='current',
                   waveform=lambda t: jnp.exp(-((t / 1e-11 - 3) / 2) ** 2))
    sim.add_probe((.004, .003, .003), 'ez')
    return sim


@pytest.mark.parametrize('graded', [False, True])
def test_run_default_identity_and_ticks(graded, capsys):
    sim = small_sim(graded)
    kw = dict(n_steps=200, compute_s_params=False, skip_preflight=True)
    reference = digest(sim.run(**kw))
    capsys.readouterr()
    rows = []
    with events(rows.append):
        actual = digest(sim.run(**kw))
    assert actual == reference, 'R1: events changed the production record'
    assert '[PROGRESS]' not in capsys.readouterr().out
    check_phases(rows, 'graded' if graded else 'uniform')
    starts = [r for r in rows if r['kind'] == 'phase_start' and r['phase'] == 'step']
    if graded:
        assert starts[0]['ticks'] is False
        assert not any(r['kind'] == 'progress' for r in rows)
    else:
        assert any(r['kind'] == 'progress' for r in rows)
    assert next(r for r in rows if r['kind'] == 'phase_end' and r['phase'] == 'step')['stop'] == 'steps_done'


@pytest.mark.parametrize('n_steps', [37, 101, 650])
def test_run_pec_derived_chunk_identity(n_steps, monkeypatch):
    monkeypatch.delenv('RFX_EVENTS', raising=False)
    sim = Simulation(freq_max=10e9, domain=(.00703, .00602, .00501), dx=.001,
                     boundary='pec', cpml_layers=0)
    sim.add_source((.0021, .0018, .0022), 'ez', amplitude_kind='current',
                   waveform=lambda t: jnp.exp(-((t / 1e-11 - 3) / 2) ** 2))
    sim.add_probe((.0048, .0039, .0037), 'ez')
    sim.add_flux_monitor(axis='x', coordinate=.0048,
                         freqs=jnp.array([3e9, 5e9]), name='flux')
    kw = dict(n_steps=n_steps, compute_s_params=False, skip_preflight=True)
    reference = sim.run(**kw)
    rows = []
    with events(rows.append):
        actual = sim.run(**kw)
    assert any(row['kind'] == 'progress' for row in rows)
    assert 'flux' in reference.flux_monitors and 'flux' in actual.flux_monitors
    assert digest(actual) == digest(reference), 'R1: events changed the PEC record'


@pytest.mark.slow
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('decay', [False, True])
def test_user_cadence_identity_and_unchanged_text(graded, decay, capsys, monkeypatch):
    # Freeze the existing human reporter clock, leaving all solver computations real.
    monkeypatch.setattr('rfx.stepping.progress.time.perf_counter', lambda: 100.0)
    sim = small_sim(graded)
    kw = dict(n_steps=40, report_every=10, report_label='\ud55c' * 200,
              compute_s_params=False, skip_preflight=True)
    if decay:
        kw.update(until_decay=1e-12, decay_max_steps=40, decay_min_steps=0, decay_check_interval=10)
    reference = digest(sim.run(**kw))
    text_off = [x for x in capsys.readouterr().out.splitlines() if '[PROGRESS]' in x or '[STOP]' in x]
    rows = []
    with events(rows.append):
        actual = digest(sim.run(**kw))
    text_on = [x for x in capsys.readouterr().out.splitlines() if '[PROGRESS]' in x or '[STOP]' in x]
    assert actual == reference
    assert text_on == text_off
    check_phases(rows, 'graded' if graded else 'uniform')
    end = next(r for r in rows if r['kind'] == 'phase_end' and r['phase'] == 'step')
    assert end['stop'] in (['threshold', 'float32_floor', 'cap'] if decay else ['steps_done'])
    if decay:
        assert any(f': {end["stop"]} |' in line for line in text_on)
        tick = next(r for r in rows if r['kind'] == 'progress')
        assert tick['total_is_cap'] is True
        assert {'criterion', 'criterion_value', 'criterion_threshold'} <= tick.keys()


@pytest.mark.slow
def test_uniform_loaded_carry_events_only_identity():
    from tests.unit.runners.test_run_progress_reporting import _loaded_run
    reference_hashes, reference_arrays = _loaded_run(n_steps=400)
    rows = []
    with events(rows.append), scope('run', 'uniform'):
        actual_hashes, actual_arrays = _loaded_run(n_steps=400)
    assert actual_hashes == reference_hashes, 'R1: events-only changed all-six-field/carry digests'
    assert digest(actual_arrays) == digest(reference_arrays), 'R1: events-only chunk carry changed fields/probes/DFT/snapshots/flux'
    check_phases(rows, 'uniform')


@pytest.mark.slow
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('wire', [False, True])
def test_run_drive_passes(graded, wire):
    from tests.unit.runners.test_silent_drop_warnings import _two_wire_sim
    sim = _two_wire_sim(graded)
    if not wire:
        # Build the same two-port declaration with lumped, one-cell ports.
        sim = Simulation(freq_max=5e9, domain=(.02, .02, .01), dx=.001,
                         boundary='pec', cpml_layers=0,
                         **({'dz_profile': np.full(10, .001)} if graded else {}))
        for i, x in enumerate((.006, .014)):
            sim.add_port((x, .01, .002), 'ez', impedance=50., excite=i == 0)
    kw = dict(n_steps=200, compute_s_params=True, s_param_freqs=np.array([2e9, 3e9]), skip_preflight=True)
    reference = digest(sim.run(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.run(**kw))
    assert actual == reference
    check_phases(rows, 'graded' if graded else 'uniform', passes=3)
    assert rows[0]['passes'] == ['main', 'p1', 'p2']
    assert len([r for r in rows if r['kind'] == 'phase_end' and r['phase'] == 'step']) == 3


@pytest.mark.slow
@pytest.mark.parametrize('path', ['multi-device', 'subgridded'])
def test_deferred_run_paths_never_silently_drop_step(path):
    from tests.unit.runners.test_silent_drop_warnings import _distributed_sim, _subgrid_sim
    sim = _distributed_sim() if path == 'multi-device' else _subgrid_sim()
    kw = dict(n_steps=8, compute_s_params=False, skip_preflight=True)
    if path == 'multi-device':
        assert len(jax.devices('cpu')) >= 2
        kw['devices'] = jax.devices('cpu')[:2]
    reference = digest(sim.run(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.run(**kw))
    assert actual == reference
    check_phases(rows, path, deferred=True)


@pytest.mark.slow
@pytest.mark.parametrize('graded', [False, True])
def test_forward_eager_deferred_identity(graded):
    sim = small_sim(graded)
    kw = dict(n_steps=20, skip_preflight=True)
    reference = digest(sim.forward(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.forward(**kw))
    assert actual == reference
    check_phases(rows, 'graded' if graded else 'uniform', deferred=True)


@pytest.mark.slow
@pytest.mark.parametrize('transform', ['jit', 'grad', 'vmap'])
@pytest.mark.parametrize('graded', [False, True])
def test_traced_forward_once_per_entry_path(transform, graded, monkeypatch):
    from rfx.stepping import events as transport
    monkeypatch.setattr(transport, '_TRACED', set())
    sim = small_sim(graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    def objective(scale):
        result = sim.forward(eps_override=eps * scale, n_steps=8, skip_preflight=True)
        return jnp.sum(result.time_series ** 2)
    fn = {'jit': jax.jit, 'grad': jax.grad, 'vmap': jax.vmap}[transform](objective)
    arg = jnp.array([1., 1.1]) if transform == 'vmap' else jnp.array(1.)
    reference = np.asarray(fn(arg))
    # A fresh transformation forces trace-time execution after the reference.
    fn = {'jit': jax.jit, 'grad': jax.grad, 'vmap': jax.vmap}[transform](lambda p: objective(p))
    rows = []
    with events(rows.append):
        actual = np.asarray(fn(arg))
        fn(arg)
    np.testing.assert_array_equal(actual, reference)
    refused = [r for r in rows if r['kind'] == 'not_reported']
    assert [(r['phase'], r['reason_code']) for r in refused] == [
        ('assemble', 'traced'), ('compile', 'compile_not_separated'),
        ('step', 'traced'), ('post-process', 'traced')]
    assert len([r for r in rows if r['kind'] == 'plan']) == 1


@pytest.mark.slow
@pytest.mark.parametrize('normalize', [False, True, 'flux'])
def test_waveguide_drive_passes(normalize):
    from tests.unit.sparams.test_waveguide_port_reference_sims import _tiny_two_port
    sim = _tiny_two_port()
    kw = dict(n_steps=200, normalize=normalize, skip_preflight=True)
    reference = digest(sim.compute_waveguide_s_matrix(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.compute_waveguide_s_matrix(**kw))
    assert actual == reference
    check_phases(rows, 'uniform', passes=4 if normalize else 2)
    assert {r['entry'] for r in rows} == {'compute_waveguide_s_matrix'}


@pytest.mark.slow
def test_msl_drive_passes():
    from tests.unit.runners.test_run_progress_reporting import _msl_thru
    sim = _msl_thru()
    kw = dict(n_steps=200, freqs=np.linspace(2e9, 18e9, 3), report_every=50)
    reference = digest(sim.compute_msl_s_matrix(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.compute_msl_s_matrix(**kw))
    assert actual == reference
    check_phases(rows, 'uniform', passes=2)
    assert {r['entry'] for r in rows} == {'compute_msl_s_matrix'}


@pytest.mark.slow
@pytest.mark.parametrize('option', [{'checkpoint': True}, {'checkpoint_every': 4}, {'n_warmup': 4}])
def test_graded_incompatible_chunk_options_keep_scan(option):
    # These knobs are exercised through the production runner shared by forward.
    from rfx.runners.nonuniform import run_nonuniform_path
    sim = small_sim(True)
    kw = dict(n_steps=20, **option)
    reference = digest(run_nonuniform_path(sim, **kw))
    rows = []
    with events(rows.append), scope('run', 'graded'):
        actual = digest(run_nonuniform_path(sim, **kw))
    assert actual == reference
    check_phases(rows, 'graded')
    assert next(r for r in rows if r['kind'] == 'phase_start' and r['phase'] == 'step')['ticks'] is False


@pytest.mark.slow
@pytest.mark.parametrize('normalize', [False, 'flux'])
def test_waveguide_multimode_drive_passes(normalize):
    from tests.unit.sparams.test_waveguide_port_reference_sims import _tiny_two_port
    sim = _tiny_two_port(n_modes=2)
    kw = dict(n_steps=200, normalize=normalize, skip_preflight=True)
    reference = digest(sim.compute_waveguide_s_matrix(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.compute_waveguide_s_matrix(**kw))
    assert actual == reference
    check_phases(rows, 'uniform', passes=8 if normalize else 4)


@pytest.mark.slow
@pytest.mark.parametrize('normalize', [True, 'flux'])
def test_graded_waveguide_does_not_derive_chunks(normalize):
    from tests.unit.sparams.test_waveguide_nu_flux import _make_wr90_nu_sim_with_slab
    sim = _make_wr90_nu_sim_with_slab()
    kw = dict(n_steps=200, normalize=normalize, skip_preflight=True)
    reference = digest(sim.compute_waveguide_s_matrix(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.compute_waveguide_s_matrix(**kw))
    assert actual == reference
    check_phases(rows, 'graded', passes=4)
    assert all(r['ticks'] is False for r in rows if r['kind'] == 'phase_start' and r['phase'] == 'step')


@pytest.mark.slow
def test_msl_material_override_inherits_calculator_scope():
    from tests.unit.runners.test_run_progress_reporting import _msl_thru
    sim = _msl_thru()
    eps = jnp.ones(sim._build_grid().shape, dtype=jnp.float32)
    kw = dict(n_steps=200, freqs=np.linspace(2e9, 18e9, 3), eps_override=eps)
    reference = digest(sim.compute_msl_s_matrix(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.compute_msl_s_matrix(**kw))
    assert actual == reference
    check_phases(rows, 'uniform', passes=2)
    assert {r['entry'] for r in rows} == {'compute_msl_s_matrix'}


@pytest.mark.slow
def test_mixed_s_matrix_drive_passes():
    from tests.unit.sparams.test_mixed_port_sparam import _base_sim, _add_msl, _add_feed
    sim, yc = _base_sim()
    _add_feed(sim, yc)
    _add_msl(sim, yc, n_probe_offset=10, n_probe_spacing=4)
    kw = dict(n_steps=200, freqs=np.array([2e9, 3e9]), skip_preflight=True)
    reference = digest(sim.compute_mixed_s_matrix(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.compute_mixed_s_matrix(**kw))
    assert actual == reference
    check_phases(rows, 'uniform', passes=2)


@pytest.mark.slow
@pytest.mark.parametrize('entry', ['compute_coaxial_line_reflection', 'compute_coaxial_two_port', 'compute_coax_msl_transition'])
def test_other_calculators_explicit_refusal_and_identity(entry):
    if entry == 'compute_coax_msl_transition':
        from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs
        sim, kw = build_instrument_junction(), instrument_kwargs(200)
    else:
        from tests.unit.sparams.test_coax_two_port_smatrix import _sim
        sim, kw = _sim(), dict(n_steps=200, freqs=np.array([6e9, 8e9, 10e9]))
    call = getattr(sim, entry)
    reference = digest(call(**kw))
    rows = []
    with events(rows.append):
        actual = digest(call(**kw))
    assert actual == reference
    check_phases(rows, 'uniform', passes=1 if entry == 'compute_coaxial_line_reflection' else 2, deferred=True)


@pytest.mark.slow
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_distributed_both_meshes_deferred(graded, entry):
    sim = Simulation(freq_max=15e9, domain=(.024, .012, .012), dx=.001, boundary='pec',
                     **({'dz_profile': np.full(12, .001)} if graded else {}))
    sim.add_source((.006, .006, .006), 'ez', amplitude_kind='field')
    sim.add_probe((.012, .006, .006), 'ez')
    assert len(jax.devices('cpu')) >= 2
    kw = dict(n_steps=8, devices=jax.devices('cpu')[:2], skip_preflight=True)
    kw.update(distributed=True) if entry == 'forward' else kw.update(compute_s_params=False)
    call = getattr(sim, entry)
    reference = digest(call(**kw))
    rows = []
    with events(rows.append):
        actual = digest(call(**kw))
    assert actual == reference
    check_phases(rows, 'multi-device', deferred=True)


@pytest.mark.slow
@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('entry', ['run', 'until_identified', 'forward'])
def test_ringdown_and_identification_accounting(lane, entry):
    from rfx.ringdown import RingdownSpec
    from tests.unit.sparams.test_ringdown_run import _box
    from tests.unit.sparams.test_ringdown_forward import _kw, _run_kw
    sim = _box(lane)
    kw = dict(n_steps=600, ringdown=RingdownSpec(), skip_preflight=True)
    if entry == 'forward':
        kw.update(_kw(lane))
        call = sim.forward
    else:
        kw.update(_run_kw(lane), compute_s_params=True)
        if entry == 'until_identified':
            kw['until_identified'] = True
        call = sim.run
    reference = digest(call(**kw))
    rows = []
    with events(rows.append):
        actual = digest(call(**kw))
    assert actual == reference
    check_phases(rows, lane, deferred=entry == 'forward')
    if entry != 'forward':
        assert any(r.get('extractor') == 'ringdown' for r in rows)
    if entry == 'until_identified':
        assert next(r for r in rows if r['kind'] == 'phase_end' and r['phase'] == 'step')['stop'] in ['threshold', 'cap']


@pytest.mark.slow
@pytest.mark.parametrize('option', ['checkpoint_segments', 'adjoint'])
def test_uniform_incompatible_chunk_options_keep_scan(option):
    if option == 'adjoint':
        from tests.unit.autodiff.test_reciprocity_adjoint import fixture
        from tests.unit.autodiff.test_design_box_tape import BOX_LO, BOX_HI
        sim, eps = fixture(precision='float32')
        kw = dict(design_box=(BOX_LO, BOX_HI), design_eps_override=eps, gradient='adjoint')
    else:
        sim, kw = small_sim(), dict(checkpoint_segments=4)
    kw.update(n_steps=20, skip_preflight=True)
    reference = digest(sim.forward(**kw))
    rows = []
    # Exercise the shared inner loop as an inherited calculator pass. Top-level
    # forward's explicit refusal is separately locked above.
    with events(rows.append), scope('compute_waveguide_s_matrix', 'uniform'):
        actual = digest(sim.forward(**kw))
    assert actual == reference
    check_phases(rows, 'uniform')
    assert next(r for r in rows if r['kind'] == 'phase_start' and r['phase'] == 'step')['ticks'] is False


@pytest.mark.slow
@pytest.mark.parametrize('wire', [False, True])
def test_diagnostic_eager_extractors_refuse_explicitly(wire):
    from rfx.core.yee import init_materials
    from rfx.sources.sources import LumpedPort, WirePort
    from rfx.probes.probes import extract_s_matrix, extract_s_matrix_wire
    from tests.unit.sparams.test_sparam_driver_matches_eager import _eager_grid, _waveform
    grid = _eager_grid()
    materials = init_materials(grid.shape)
    ctor, call = (WirePort, extract_s_matrix_wire) if wire else (LumpedPort, extract_s_matrix)
    geometry = ({'start': (.012, .006, .006), 'end': (.012, .006, .008)} if wire
                else {'position': (.012, .006, .006)})
    ports = [ctor(**geometry, component='ez', impedance=50., excitation=_waveform())]
    args = (grid, materials, ports, np.array([3e9, 5e9]), 20)
    reference = digest(call(*args, boundary='cpml', cpml_axes='xyz'))
    rows = []
    with events(rows.append):
        actual = digest(call(*args, boundary='cpml', cpml_axes='xyz'))
    assert actual == reference
    check_phases(rows, 'uniform', deferred=True)


@pytest.mark.slow
def test_environment_switch_on_public_run_subprocess(monkeypatch):
    import subprocess
    import sys
    program = '''
from tests.contracts.test_progress_events import small_sim
small_sim().run(n_steps=200, compute_s_params=False, skip_preflight=True)
'''
    monkeypatch.setenv('RFX_EVENTS', 'stderr')
    completed = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, check=True)
    rows = [json.loads(s[10:]) for s in completed.stderr.splitlines() if s.startswith('RFX_EVENT ')]
    check_phases(rows, 'uniform')
    assert '[PROGRESS]' not in completed.stdout
    monkeypatch.setenv('RFX_EVENTS', 'bogus')
    refused = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
    assert refused.returncode != 0 and "RFX_EVENTS must be 'stderr'" in refused.stderr


@pytest.mark.slow
@pytest.mark.parametrize('graded', [False, True])
def test_implicit_s_refusal_does_not_plan_unused_drives(graded):
    sim = small_sim(graded)
    for x in (.002, .004):
        sim.add_port((x, .002, .002), 'ez', impedance=50., excite=False)
    kw = dict(n_steps=200, skip_preflight=True)
    reference = digest(sim.run(**kw))
    rows = []
    with events(rows.append):
        actual = digest(sim.run(**kw))
    assert actual == reference
    check_phases(rows, 'graded' if graded else 'uniform')
    assert rows[0]['passes'] == []
