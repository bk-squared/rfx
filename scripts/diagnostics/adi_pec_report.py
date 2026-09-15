"""Pack the ADI sweep's raw JSON cases and plot the independent probe traces.

Usage: python scripts/diagnostics/adi_pec_report.py RAW_DIR OUTPUT_DIR
The output NPZ retains BOTH the transmitted and source-location probes.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

raw, out = map(Path, sys.argv[1:])
rows, traces = [], {}
for path in sorted([*raw.glob('3d_*.json'), *raw.glob('2d_tmz_*.json')]):
    row = json.loads(path.read_text())
    if not isinstance(row, dict) or 'traces' not in row:
        continue
    key = f'{row["mode"]}_{row["kind"]}_{row["factor"]:g}_{row["steps"]}'
    traces[key] = np.asarray(row.pop('traces'), dtype=np.float32)
    row['trace_key'] = key
    rows.append(row)
if not rows:
    rows = json.loads((raw / 'metrics.json').read_text())
    with np.load(raw / 'traces.npz') as stored:
        traces = {key: stored[key] for key in stored.files}
expected = {
    f'{mode}_{kind}_{factor:g}_{steps}'
    for steps, factors in ((200, (.5, 1, 1.5, 2, 3, 5)),
                           (800, (1, 1.25, 1.5, 1.75, 2)), (4000, (.5, 1)))
    for mode in ('3d', '2d_tmz')
    for kind in ('none', 'sheet', 'volume1', 'volume3')
    for factor in factors
}
if set(traces) != expected:
    raise ValueError('Report requires the complete 104-case battery; output not written')
out.mkdir(exist_ok=True, parents=True)
(out / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
np.savez_compressed(out / 'traces.npz', **traces)
colors = dict(none='black', sheet='#d55e00', volume1='#0072b2', volume3='#009e73')
fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout='constrained')
for i, mode in enumerate(('3d', '2d_tmz')):
    for j, (factor, steps) in enumerate(((5, 200), (1, 4000))):
        ax = axes[i, j]
        for kind, color in colors.items():
            key = f'{mode}_{kind}_{factor}_{steps}'
            row = next(r for r in rows if r['trace_key'] == key)
            vals = traces[key][:, 0]
            ax.semilogy(np.arange(steps) * row['dt'] * 1e9,
                        np.maximum(np.abs(vals), 1e-12), lw=.85,
                        color=color, label=kind)
        ax.set(xlabel='Time (ns)', ylabel='|Ez| at far probe', ylim=(1e-7, 1e35 if j == 0 else 1e21))
        ax.text(.02, .97, f'{mode}, factor {factor}, {steps} steps',
                transform=ax.transAxes, va='top')
        ax.grid(alpha=.2)
axes[0, 0].legend(loc='lower right')
fig.savefig(out / 'probe_growth.png', dpi=150)
plt.close(fig)
# Independent witness at the source-side live probe (not an inside-PEC zero).
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), layout='constrained')
for ax, mode in zip(axes, ('3d', '2d_tmz')):
    for kind, color in colors.items():
        key = f'{mode}_{kind}_1_4000'
        row = next(r for r in rows if r['trace_key'] == key)
        ax.semilogy(np.arange(4000) * row['dt'] * 1e9,
                    np.maximum(np.abs(traces[key][:, 1]), 1e-12), lw=.8,
                    color=color, label=kind)
    ax.set(xlabel='Time (ns)', ylabel='|Ez| at source-side probe', ylim=(1e-7, 1e22))
    ax.text(.02,.97, f'{mode}, factor 1', transform=ax.transAxes, va='top')
    ax.grid(alpha=.2)
axes[0].legend(loc='lower right')
fig.savefig(out / 'source_side_growth.png', dpi=150)
print(f'Packed {len(rows)} cases and {sum(a.shape[0] for a in traces.values())} two-probe samples')
