from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
OUT = Path('/root/workspace/bk-workspace/.801-measure/msl_inset')
OLD = OUT.parent / 'msl'
SRC = OUT.parent / 'src-main'

def mutation_guard(event, args):
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, int) or not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return
        p = Path(os.fsdecode(path)).resolve()
        if not p.is_relative_to(OUT) or p.exists():
            raise PermissionError(f'NEW_FILES_ONLY: {event} {p}')
    elif event == 'os.mkdir':
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(OUT) or p.exists():
            raise PermissionError(f'NEW_DIRECTORIES_ONLY: {event} {p}')
    elif event in {'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink', 'os.chmod', 'os.chown', 'os.truncate'}:
        raise PermissionError(f'NO_MUTATION: {event} {args}')

sys.addaudithook(mutation_guard)
import numpy as np
from render_edges import fmt, table

FIXTURES = ('cv06b', 'cv20')
VARIANTS = ('baseline', 'continued', 'inset1')

def compact_record(path):
    raw = json.loads(Path(path).read_text())
    record = {k: v for k, v in raw.items() if k not in ('pec_sheets', 'pec_wires', 'source_specs', 'probe_specs')}
    record['source_x_indices_received'] = sorted({s['i'] for s in raw['source_specs']})
    record['probe_x_indices_received'] = sorted({s['i'] for s in raw['probe_specs']})
    record['record_file'] = str(Path(path).relative_to(OUT.parent))
    return record

# Numerical reduction functions copied from the previous two-arm measurement.
old_source = (OLD / 'reduce.py').read_text()
old_ast = ast.parse(old_source)
for name in ('read', 'write', 'module', 'stats', 'reduce_fixture'):
    node = next(n for n in old_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
    copied = ast.get_source_segment(old_source, node)
    if name == 'reduce_fixture':
        copied = copied[:copied.index('    if len(data)==2:')]
        copied = copied.replace('p=OUT/fixture/variant', "p=(OUT if variant == 'inset1' else OLD)/fixture/variant")
        copied += '    return result, data\n'
        copied = copied.replace('def reduce_fixture(', 'def source_reduce_fixture(', 1)
    exec(compile(copied, str(OLD / 'reduce.py') + '::copy::' + name, 'exec'), globals())

def reduce_fixture(fixture):
    result, data = source_reduce_fixture(fixture)
    result['comparisons'] = {}
    for reference in ('baseline', 'continued'):
        f, s, ds = data['inset1']
        fr, sr, dr = data[reference]
        assert np.array_equal(f, fr)
        c = {'frequency_vectors_equal': 1, 'deltas': {}, 'arrays': {}}
        for i in range(2):
            for j in range(2):
                phase = np.angle(s[i,j] * np.conj(sr[i,j]), deg=True)
                phase[(abs(s[i,j]) == 0) | (abs(sr[i,j]) == 0)] = np.nan
                c['deltas'][f'S{i+1}{j+1}'] = {
                    'linear magnitude': stats(abs(s[i,j]) - abs(sr[i,j])),
                    'dB': stats(20*np.log10(abs(s[i,j])) - 20*np.log10(abs(sr[i,j]))),
                    'phase (deg)': stats(phase),
                }
        with np.load(OUT/fixture/'inset1/assembly_received_00.npz') as inz, np.load(OLD/fixture/reference/'assembly_received_00.npz') as refz:
            for field in inz.files:
                c['arrays'][field + '_changed_entries'] = int(np.count_nonzero(inz[field] != refz[field]))
        result['comparisons']['inset1-vs-' + reference] = c
    for variant, row in result['variants'].items():
        if fixture == 'cv20':
            row['analytic_beta']['signed_deviation_stats_pct'] = stats(row['analytic_beta']['signed_deviation_pct'])
    write(OUT/(fixture + '_reduced.json'), result)
    print(json.dumps({fixture: {v: d['status']['completed'] for v,d in result['variants'].items()}}))

def render_table():
    lines = [
        'S = public result.S; S_raw = public result.S_raw. Column power = sum over output ports of magnitude squared.',
        'Delta = inset1 minus reference. Phase delta = arg(S_inset1 * conj(S_reference)), degrees.',
        'Ring-down = 10 log10(last-10%-mean(Ez squared) / peak(Ez squared)), maximum over the 10 point probes, per drive.',
        'S arithmetic: complex128. S_raw arithmetic: stored dtype. Passivity correction: public diagnostic, linear amplitude units.',
        '',
    ]
    for fixture in FIXTURES:
        r = read(OUT/(fixture + '_reduced.json'))
        d = r['variants']
        lines += [f'**{fixture}**', '']
        metrics = [
            ('frequency bins', lambda a: a['status']['frequencies']),
            ('time-stepping calls', lambda a: a['status']['timestepping_calls']),
            ('ring-down drive 0 (dB)', lambda a: a['settling_db'][0]),
            ('ring-down drive 1 (dB)', lambda a: a['settling_db'][1]),
            ('recomputed ring-down drive 0 (dB)', lambda a: a['ringdown_recomputed'][0]['worst_db']),
            ('recomputed ring-down drive 1 (dB)', lambda a: a['ringdown_recomputed'][1]['worst_db']),
            ('max abs(S12-S21)', lambda a: a['max_reciprocity']),
            ('max raw abs(S12-S21)', lambda a: a['max_reciprocity_raw']),
            ('max column power, raw', lambda a: a['max_column_power_raw']),
            ('max column power, corrected', lambda a: a['max_column_power']),
            ('max passivity correction', lambda a: a['max_passivity_correction']),
        ]
        lines += [table(['quantity', *VARIANTS], [[name] + [fn(d[v]) for v in VARIANTS] for name,fn in metrics])]
        for label, c in r['comparisons'].items():
            lines += [f'**{fixture}: {label}**', '']
            rows = []
            for entry, units in c['deltas'].items():
                for unit, st in units.items():
                    rows.append([entry, unit, st['count'], st['finite_count'], st['max_abs'], st['mean_abs'],
                                 st['min_signed'], st['max_signed'], st['mean_signed']])
            lines += [table(['S entry','delta unit','bins','finite bins','max abs delta','mean abs delta','min signed delta','max signed delta','mean signed delta'], rows)]
            lines += [table(['array', 'changed entries'], list(c['arrays'].items()))]
        if fixture == 'cv06b':
            lines += ['Notch: S21; 3-point parabola in log magnitude using spectral_features.refined_extremum(transform="log").', '']
            metrics = [
                ('sampled frequency (GHz)', lambda a: a['notch']['bin_f']/1e9),
                ('3-point frequency (GHz)', lambda a: a['notch']['refined_f']/1e9),
                ('sampled depth (dB)', lambda a: a['notch']['sampled_depth_db']),
                ('3-point depth (dB)', lambda a: a['notch']['parabolic_depth_db']),
                ('3-point shift (bins)', lambda a: a['notch']['sub_bin_shift']),
            ]
            lines += [table(['quantity', *VARIANTS], [[name]+[fn(d[v]) for v in VARIANTS] for name,fn in metrics])]
        else:
            a = {v: d[v]['analytic_beta'] for v in VARIANTS}
            lines += [
                'Beta: 20_msl_phase_referee.py::_analytic_beta_witness; eps_eff: _hammerstad_jensen_eps_eff; height: _h_dielectric_under_strip.',
                'Signed deviation (%) = 100 * (Re(beta_fitted) / beta_HJ - 1); gated band 3.0–4.5 GHz.',
                '',
                table(['quantity', *VARIANTS], [[name] + [a[v][key]*scale for v in VARIANTS] for name,key,scale in [
                    ('width (µm)','width_m',1e6), ('height (µm)','height_m',1e6),
                    ('eps_r','eps_r',1), ('eps_eff','eps_eff',1), ('c0 (m/s)','c0_m_per_s',1)]]),
                table(['quantity', *VARIANTS], [[label] + [a[v]['signed_deviation_stats_pct'][key] for v in VARIANTS] for label,key in [
                    ('gated bins','count'), ('signed beta deviation min (%)','min_signed'),
                    ('signed beta deviation max (%)','max_signed'), ('signed beta deviation mean (%)','mean_signed')]])
            ]
            for label,key in [('fitted beta (rad/m)','beta_fitted_rad_per_m'), ('signed beta deviation (%)','signed_deviation_pct'), ('angle(S21) (deg)','phase_s21_deg')]:
                rows = [[i, f/1e9, a['baseline']['beta_hj_rad_per_m'][n]] + [a[v][key][n] for v in VARIANTS]
                        for n,(i,f) in enumerate(zip(a['baseline']['gated_bin_indices'],a['baseline']['freqs_hz']))]
                lines += [label, '', table(['bin','f (GHz)','HJ beta (rad/m)', *VARIANTS],rows)]
            lines += [table(['analytic witness exception count', *VARIANTS],
                            [['RuntimeError', *[int(a[v]['exception'] is not None) for v in VARIANTS]]])]
    with (OUT/'TABLE.md').open('x') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('item', choices=(*FIXTURES, 'table'))
    args = ap.parse_args()
    if args.item == 'table':
        render_table()
    else:
        reduce_fixture(args.item)

