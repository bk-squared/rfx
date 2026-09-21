import json
from pathlib import Path
import sys

OUT = Path('/root/workspace/bk-workspace/.801-measure/msl_inset')

def compact_record(path):
    raw = json.loads(Path(path).read_text())
    record = {k:v for k,v in raw.items() if k not in ('pec_sheets','pec_wires','source_specs','probe_specs')}
    record['source_x_indices_received'] = sorted({s['i'] for s in raw['source_specs']})
    record['probe_x_indices_received'] = sorted({s['i'] for s in raw['probe_specs']})
    record['record_file'] = str(Path(path).relative_to(OUT))
    return record

def fmt(x):
    if x is None:
        return 'NA'
    if isinstance(x, float):
        return f'{x:.12g}'
    return str(x)

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in rows]) + '\n'

def render_record(label, d):
    lines = [f'**{label}**', '',
             f"grid = {d['grid_shape']}; dx = {d['dx_m'] * 1e6:.12g} µm; dt = {d['dt_s']:.12g} s; steps = {d['n_steps']}.",
             f"x absorber indices = [0, {d['face_pads']['x_lo'] - 1}], [{d['grid_shape'][0] - d['face_pads']['x_hi']}, {d['grid_shape'][0] - 1}].",
             f"source x indices received = {d['source_x_indices_received']}; point-probe x indices received = {d['probe_x_indices_received']}.",
             f"pec_mask present = {d['pec_mask_present']}; pec_edge_masks present = {d['pec_edge_masks_present']}.",
             f"trace declared bounds (m) = {d['trace_bounds_m']}; realized z indices = {d['realized_trace_z_indices']}.",
             f"readback = `{d['record_file']}`.", '']
    lines += [table(['port', 'source x index', 'reference x index', 'probe x indices', 'source x (mm)', 'reference x (mm)'],
                    [[p['name'],p['source_x_index'],p['reference_x_index'],str(p['probe_x_indices']),
                      p['source_x_m']*1e3,p['reference_x_m']*1e3] for p in d['ports']])]
    for p in d['planes']:
        lines += [f"z index = {p['z_index']}; z = {p['z_m']*1e6:.12g} µm; y index = {p['y_index']}; y = {p['y_m']*1e6:.12g} µm; eps_r sample z index = {p['eps_below_z_index']}; z = {p['eps_below_z_m']*1e6:.12g} µm.", '',
                  table(['edge', 'first x', 'last x', 'centre count', 'centre x-lo count', 'centre x-hi count', 'width x-lo count', 'width x-hi count'],
                        [[a]+[p['edge_counts'][a][key] for key in ('first_x_index','last_x_index','centre_row_total',
                          'centre_row_x_lo','centre_row_x_hi','trace_width_x_lo','trace_width_x_hi')] for a in 'xy'])]
        rows=[]
        for w in p['windows'].values():
            rows += [[x, xx*1e3, ex, ey, pm, eps] for x,xx,ex,ey,pm,eps in
                     zip(w['x_indices'],w['x_m'],w['pec_edge_x'],w['pec_edge_y'],w['pec_mask'],w['eps_r_below'])]
        lines += [table(['x index', 'x (mm)', 'Ex PEC', 'Ey PEC', 'pec_mask', 'eps_r below'],rows)]
    lines += [table(['array', 'total flags'], list(d['totals'].items()))]
    return '\n'.join(lines)

def main():
    lines = ['Readback: `rfx.simulation.run`; time-stepping calls = 0. Indices are zero-based.',
             'Trace-edge occupancy = Ex PEC OR Ey PEC. A1 = occupancy 0 at x indices 0 and nx-1; occupancy 1 at all other x absorber indices, on every recorded trace plane.', '']
    for fixture in ('cv06b', 'cv20'):
        for folder in sorted(OUT.glob(fixture + '_dry_*')):
            path = folder / 'inset1/assembly_received_00.json'
            status = json.loads((folder / 'inset1/status.json').read_text())
            if path.exists():
                d = compact_record(path)
                lines += [f"A1 = {status.get('A1')}; time-stepping calls = {status['timestepping_calls']}.", '',
                          render_record(folder.name, d)]
            else:
                lines += [json.dumps(status), '']
    with (OUT / 'EDGES_inset1.md').open('x') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    main()
