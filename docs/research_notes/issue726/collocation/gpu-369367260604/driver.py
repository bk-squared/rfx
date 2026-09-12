"""Run existing MSL consumer gates sequentially after spatial collocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(os.environ['RFX_OUT'])
CASES = {
    'axis': 'tests/unit/ports/test_msl_port_axis_generality.py::test_y_directed_thru_reproduces_the_x_directed_thru',
    'ad': 'tests/unit/autodiff/test_msl_ad_fd_converged.py::test_msl_ad_fd_converged_tight',
    'coupon': 'tests/unit/autodiff/test_msl_sparam_ad.py::test_compute_msl_s_matrix_end_to_end_matches_historical_base',
    'nu': 'tests/locks/test_msl_nu_sparam_gate.py::test_nu_msl_patch_s11_passive_and_edge_fed_match',
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def receipt():
    import jax
    import numpy
    import scipy
    import rfx
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    assert sha == os.environ['RFX_SHA'], (sha, os.environ['RFX_SHA'])
    assert not dirty, dirty
    assert Path(rfx.__file__).resolve().is_relative_to(Path.cwd())
    assert jax.default_backend() == 'gpu', jax.devices()
    return dict(source_sha=sha, source_dirty=False, jax=jax.__version__,
                numpy=numpy.__version__, scipy=scipy.__version__,
                devices=str(jax.devices()), cases=CASES,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def worker(name):
    import jax
    import numpy as np
    import pytest
    from rfx import Simulation

    case_out = OUT / name
    case_out.mkdir(exist_ok=False)
    original = Simulation.compute_msl_s_matrix
    records = []
    import rfx.simulation as core
    original_init = core.init_state
    init_records = []
    def capture_init(*args, **kwargs):
        state = original_init(*args, **kwargs)
        init_records.append({key: {'dtype': str(getattr(state,key).dtype), 'shape': list(getattr(state,key).shape)} for key in ('ex','ey','ez','hx','hy','hz')})
        write_json(case_out/'field-initializations.json', init_records)
        return state
    core.init_state = capture_init

    def capture(sim, *args, **kwargs):
        call_id = len(records)
        record = {'call': call_id, 'field_precision': str(sim._precision), 'num_periods': kwargs.get('num_periods'),
                  'n_freqs': kwargs.get('n_freqs'),
                  'has_eps_override': kwargs.get('eps_override') is not None,
                  'checkpoint_segments': kwargs.get('checkpoint_segments')}
        records.append(record)
        # The production dump is on the untraced extraction path. Do not
        # introduce host reads or dumping into the differentiated function.
        if not any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(kwargs.get('eps_override'))):
            kwargs['raw_3probe_dump_path'] = str(case_out / f'call-{call_id}-raw-vi.npz')
        result = original(sim, *args, **kwargs)
        if any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(result.S)):
            record['traced'] = True
        else:
            arrays = {}
            for key in ['S', 'S_raw', 'freqs', 'Z0', 'beta', 'settling_db',
                        'reliable', 'passivity_correction', 'cond_a']:
                value = getattr(result, key, None)
                if value is not None:
                    arrays[key] = np.asarray(value)
            np.savez_compressed(case_out / f'call-{call_id}-result.npz', **arrays)
            record['traced'] = False
            record['array_shapes'] = {k: list(v.shape) for k, v in arrays.items()}
            record['array_dtypes'] = {k: str(v.dtype) for k, v in arrays.items()}
        write_json(case_out / 'calls.json', records)
        return result

    import rfx.api._sparams as extraction
    original_waves = extraction.msl_solve_s_from_waves
    def capture_waves(wave_a, wave_b):
        result = original_waves(wave_a, wave_b)
        if not any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(wave_a)):
            cid = records[-1]['call']
            np.savez_compressed(case_out/f'call-{cid}-waves.npz',
                                a=np.asarray(wave_a), b=np.asarray(wave_b),
                                S=np.asarray(result[0]), cond_a=np.asarray(result[1]))
        return result
    extraction.msl_solve_s_from_waves = capture_waves
    Simulation.compute_msl_s_matrix = capture
    start = time.monotonic()
    rc = int(pytest.main([CASES[name], '-o', 'addopts=', '-vv', '-s',
                          '-p', 'no:cacheprovider', '--maxfail=1',
                          '--junitxml=' + str(case_out / 'junit.xml')]))
    write_json(case_out / 'outcome.json', {'exit_code': rc,
                                          'wall_s': time.monotonic() - start})
    return rc


def run():
    assert not (OUT / 'environment.json').exists(), 'refusing existing output'
    info = receipt()
    write_json(OUT / 'environment.json', info)
    print(json.dumps(info, indent=2), flush=True)
    outcomes = []
    for name in CASES:
        start = time.monotonic()
        print('START_CASE', name, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
        with (OUT / f'{name}.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', __file__, 'worker', name],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            rc = process.wait()
        outcomes.append({'case': name, 'exit_code': rc, 'wall_s': time.monotonic() - start})
        write_json(OUT / 'outcomes.json', outcomes)
        print('END_CASE', json.dumps(outcomes[-1]), flush=True)
    return 0 if all(item["exit_code"] == 0 for item in outcomes) else 1


if __name__ == '__main__':
    raise SystemExit(run() if sys.argv[1] == 'run' else worker(sys.argv[2]))
