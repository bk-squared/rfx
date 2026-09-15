#!/usr/bin/env python3
"""Plot fresh short captures and the historical power decomposition offline."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), constrained_layout=True)
    for column, name in enumerate(('coarse', 'fine')):
        capture = report['live'][name]
        bins = capture['bins']
        frequencies = np.array([b['frequency_hz'] for b in bins]) / 1e9
        magnitudes = np.array([b['s_abs'] for b in bins])
        ax = axes[0, column]
        ax.plot(frequencies, magnitudes[:, 0, 0], 'o-', label='Left reflection')
        ax.plot(frequencies, magnitudes[:, 1, 1], 's-', label='Right reflection')
        ax.plot(frequencies, magnitudes[:, 1, 0], 'k:', label='Both transmissions = 0')
        ax.axhline(1, color='.6', lw=.7)
        ax.set(title=f'{name.capitalize()}: every measured bin', xlabel='Frequency (GHz)', ylabel='|S|', ylim=(-.04, 1.14))
        ax.legend(fontsize=8)
        with np.load(capture['records']['path']) as traces:
            ax = axes[1, column]
            data = json.loads(Path(capture['input']['path']).read_text())
            for drive, color in ((0, 'C0'), (1, 'C1')):
                for field, style in (('v_ref_t', '-'), ('i_ref_t', '--')):
                    y = traces[f'drive{drive}_port{drive}_{field}']
                    block = max(1, len(y) // 90)
                    starts = np.arange(0, len(y), block)
                    envelope = np.array([np.abs(y[i:i+block]).max() for i in starts])
                    envelope_db = 20 * np.log10(np.maximum(envelope / np.abs(y).max(), 1e-15))
                    label = ('Left' if drive == 0 else 'Right') + (' V' if field[0] == 'v' else ' I')
                    ax.plot((starts + block/2) * data['dt'] * 1e9, envelope_db,
                            color=color, linestyle=style, label=label)
            ax.set(title=f'{name.capitalize()}: reference V/I time-record envelopes',
                   xlabel='Time (ns)', ylabel='20 log10(block peak / record peak)', ylim=(-125, 3))
            ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 2]
    historical = report['historical']['node_volume']['bins']
    frequencies = np.array([b['frequency_hz'] for b in historical]) / 1e9
    reflection = np.array([b['left_reflection_power'] for b in historical])
    false_transmission = np.array([b['left_reported_transmission_power'] for b in historical])
    ax.stackplot(frequencies, reflection, false_transmission, colors=['C0', 'C3'], alpha=.65,
                 labels=['Historical reflection power', 'Same-cavity power falsely called S21'])
    ax.plot(frequencies, [b['column_power'][0] for b in report['live']['coarse']['bins']],
            'k-o', ms=4, label='Fresh repaired left column')
    ax.axhspan(2.25, 3, color='.4', alpha=.13, label='Unchanged (2.25, 3] interval')
    ax.set(title='Historical bridge: double counting', xlabel='Frequency (GHz)',
           ylabel='Column power', ylim=(0, 3.1))
    ax.legend(fontsize=7, loc='upper left')

    ax = axes[1, 2]
    for name, style in (('coarse', 'o-'), ('fine', 's-')):
        capture = report['live'][name]
        old = json.loads(Path(capture['committed_reference']['path']).read_text())
        reference = np.array(old['s_real']) + 1j*np.array(old['s_imag'])
        fresh = np.array([b['s_real'] for b in capture['bins']]) + 1j*np.array([b['s_imag'] for b in capture['bins']])
        delta = np.max(np.abs(fresh.transpose(1, 2, 0) - reference), axis=(0, 1))
        ax.semilogy(np.array(old['frequencies_hz'])/1e9, np.maximum(delta, 1e-12), style, label=name)
    ax.set(title='Fresh vs archived S: cross-host differences',
           xlabel='Frequency (GHz)', ylabel='Maximum complex |ΔS| in each bin')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.title.set_fontsize(11)
    fig.suptitle('Short fixture adjudication — VESSL ' + report['provenance']['run_id'], fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=135)
    plt.close(fig)


if __name__ == '__main__':
    main()
