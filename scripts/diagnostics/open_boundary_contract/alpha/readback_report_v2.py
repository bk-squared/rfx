import json
from pathlib import Path
import numpy as np

OUT = Path('/root/workspace/bk-workspace/.801-measure/alpha')
FACTORS = [('f001', .01), ('f025', .25), ('f1', 1.), ('f4', 4.)]

def read(p):
    return json.loads(p.read_text())

def table(headers, rows):
    def cell(x):
        return format(x, '.17g') if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, r)) + ' |' for r in rows]) + '\n'

def main():
    failed = read(OUT/'cv20_dry_f0/status.json')
    assert failed['completed'] == 0 and failed['exception_type'] == 'FloatingPointError'
    assert failed['exception'] == 'invalid value encountered in divide'
    lines = ['Readback: `rfx.simulation.make_core_step(ctx).cpml_params`; face: `x_lo`.',
             'Time steps: 0. Float arrays: float32. Indices: 0-based.',
             'Profile expression: `(0.05 * (1.0 - rho)) * factor`.',
             'Coefficient evaluation: `numpy.errstate(divide="raise", invalid="raise")`.',
             'A2 = 0; requested factor = 0; replacement factor = 0.01.',
             '```text', failed['exception_type'] + ': ' + failed['exception'], '```', '']
    verification = dict(A1=1, A2=0, zero_error={k:failed[k] for k in ('exception_type','exception','traceback')},
                        factors=[f for _,f in FACTORS], lanes={})
    for lane in ('cv20', 'patch'):
        baseline = np.load(OUT/f'{lane}_dry_unpatched/cpml_received_00.npz')
        equal_rows, changes = [], []
        record = {}
        for label, factor in FACTORS:
            folder = OUT/f'{lane}_dry_{label}'
            status = read(folder/('status.json' if lane=='cv20' else 'result.json'))
            assert status['completed'] == 1 and status['timestepping_calls'] == 0
            with np.load(folder/'cpml_received_00.npz') as z:
                identity = {k:int(z[k].dtype==baseline[k].dtype and z[k].shape==baseline[k].shape and z[k].tobytes()==baseline[k].tobytes()) for k in z.files}
                if factor == 1:
                    assert all(identity.values()), identity
                for k in z.files:
                    assert np.isfinite(z[k]).all()
                    if k.endswith(('_sigma','_kappa')):
                        assert identity[k] == 1
                rho = 1-np.arange(len(z['x_lo_alpha']), dtype=np.float64)/(len(z['x_lo_alpha'])-1)
                expected = ((.05*(1-rho))*factor).astype(np.float32)
                assert z['x_lo_alpha'].tobytes() == expected.tobytes()
                changed = {field:int(np.count_nonzero(z['x_lo_'+field] != baseline['x_lo_'+field])) for field in ('sigma','alpha','kappa','b','c')}
                if factor != 1:
                    assert changed['alpha'] > 0 and changed['b'] > 0 and changed['c'] > 0
                record[label] = dict(factor=factor, baseline_bit_identity=identity, changed_x_lo=changed,
                                     alpha_expected_bit_identity=1, finite_arrays=30)
                changes.append([factor]+[changed[k] for k in ('sigma','alpha','kappa','b','c')])
                lines += [f'{lane}, factor = {factor:.17g}', '',
                          table(['index','sigma','alpha','kappa','b','c'],
                                [[i]+[z['x_lo_'+k][i] for k in ('sigma','alpha','kappa','b','c')] for i in range(len(expected))]),'']
                if factor == 1:
                    equal_rows = [[face]+[identity[face+'_'+f] for f in ('sigma','alpha','kappa','b','c')]
                                  for face in ('x_lo','x_hi','y_lo','y_hi','z_lo','z_hi')]
        lines += [f'{lane}: factor 1 / unpatched bit identity (0/1)', '',
                  table(['face','sigma','alpha','kappa','b','c'], equal_rows), '',
                  f'{lane}: changed entries / unpatched, x_lo', '',
                  table(['factor','sigma','alpha','kappa','b','c'],changes), '']
        verification['lanes'][lane] = record
    lines += ['A1 = 1; A2 = 0; replacement factor = 0.01.', '']
    with (OUT/'readback_verification.json').open('x') as f:
        json.dump(verification,f,indent=2)
        f.write('\n')
    with (OUT/'readback.md').open('x') as f:
        f.write('\n'.join(lines))
    print(json.dumps(dict(A1=1,A2=0,factors=verification['factors'],bit_identical_factor1_arrays=60,timesteps=0)))

if __name__ == '__main__':
    main()
