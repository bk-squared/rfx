"""Engineering figures from stored records; explicit --write, no simulations."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from _visual_data import lens_mirror, filter_mirror, dbi, plane_cut, read_json, load_record, write_manifest, LENS_LABELS, lens_label, FILTER_MESHES, amplitude_db


def mask(ax, channel=None):
    regions = [(10, 10.6, -15, '//', '0.85')] if channel == 0 else [(8.2, 9, -25, '\\\\', '0.92'), (11.6, 12.4, -25, '\\\\', '0.92')]
    if channel is None:
        mask(ax, 0)
    for lo, hi, limit, hatch, color in regions:
        ax.fill_between([lo, hi], limit, 5, facecolor=color, edgecolor='0.65', hatch=hatch, linewidth=.5)
    ax.set(xlim=(8.2, 12.4), ylim=(-65, 5), xlabel='Frequency (GHz)', ylabel=r'$|S|$ (dB)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mesh-record', type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if not args.write:
        parser.error('--write required')
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.out / '.mplconfig'))
    from ieee_style import use_pubstyle, plt
    use_pubstyle()
    plt.rcParams['savefig.bbox'] = None  # Preserve the required 7.16-inch canvas.
    a = load_record(args.record)
    kind = 'lens' if a['eps'].ndim == 4 else 'filter'
    best = int(np.argmin(a['objective']))
    manifest = dict(record=str(args.record), kind=kind, best_iteration=int(a['iteration'][best]), final_iteration=int(a['iteration'][-1]), outputs=[], missing=[], baseline_keys=[], baselines_plotted=[], display_conventions=dict(style='SciencePlots science/ieee/no-latex; canonical ieee_style.py rcParams', width_inches=7.16, png_dpi=400, inspection_dpi=150, titles=False, layout='constrained; fixed canvas', colormap='viridis', permittivity_limits=[1, 2.7 if kind == 'lens' else 10], interpolation='nearest', line_styles=['solid', 'dashed', 'dash-dot', 'dotted'], markers=['o', 's', '^', 'D'], best='earliest objective minimum, black star', final='best iterate (earliest objective minimum), the declared reported design', dbi_zero='negative infinity; negative input rejected', mask='gray hatched forbidden regions above specification limit', slices='layers 0,4,9; z=55.5,67.5,82.5 mm', cuts='signed full polar range; opposite phi branch for negative theta; row zero exactly zero', rlw='stored norm and per-FD-pixel relative errors; floor 1e-12 for plotting', fd='judged pixels, all three step sizes; h=.05 judgement', mesh_order_mm=[1.5, 1, .75] if kind == 'lens' else [.635, .423333333, .3175]))
    manifest['display_conventions']['matplotlib_rcparams'] = {k: str(v) for k, v in plt.rcParams.items()}
    def save(fig, name):
        start = time.perf_counter()
        for ext in ('pdf', 'png'):
            p = args.out / f'{kind}_{name}.{ext}'
            fig.savefig(p, dpi=400, bbox_inches=None)
            manifest['outputs'].append(dict(path=p.name, bytes=p.stat().st_size, width_inches=7.16, height_inches=float(fig.get_size_inches()[1]), **({'resolution': [int(v*400) for v in fig.get_size_inches()]} if ext == 'png' else {})))
        if shutil.which('pdftoppm'):
            subprocess.run(['pdftoppm', '-r', '150', '-png', '-singlefile', str(args.out / f'{kind}_{name}.pdf'), str(args.out / f'{kind}_{name}_inspection')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        plt.close(fig)
        print(name, round(time.perf_counter() - start, 2), 's', flush=True)
    def sub(n=1, height=3):
        return plt.subplots(1, n, figsize=(7.16, height), constrained_layout=True, squeeze=False)
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    if kind == 'lens':
        e = lens_mirror(a['eps'][best])
        fig, axes = sub(3, 2.8)
        for ax, k in zip(axes[0], [0, 4, 9]):
            im = ax.imshow(e[:, :, k].T, origin='lower', extent=(-45, 45, -45, 45), vmin=1, vmax=2.7, interpolation='nearest')
            ax.set(xlabel='x (mm)', ylabel='y (mm)')
            ax.text(.04, .95, f'z = {55.5 + 3*k:g} mm', transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
        fig.colorbar(im, ax=axes.ravel().tolist(), label=r'$\epsilon_r$ (1)', shrink=.7)
        save(fig, 'slices')
        with np.load(args.record / 'baselines.npz') as b:
            manifest['baseline_keys'] = b.files
            uniform = read_json(args.record / 'baselines.json')['best_uniform']
            curves = [(k, b[k]) for k in ['no_lens', 'grin', uniform]] + [('designed lens', a['response'][best])]
        manifest['baselines_plotted'] = [x[0] for x in curves[:-1]]
        fi = int(np.argmin(abs(a['freqs_hz'] - 1e10)))
        fig, axes = sub(2, 3.3)
        for ax, plane in zip(axes[0], ['E', 'H']):
            for j, (label, d) in enumerate(curves):
                theta, values = plane_cut(d[fi], plane)
                ax.plot(theta, dbi(values), styles[j], label=lens_label(label))
            ax.axhline(dbi(4*np.pi*.09**2/(299792458/1e10)**2), color='k', ls=(0, (5, 2, 1, 2)), label='aperture bound 4πA/λ²')
            ax.set(xlabel=f'{plane}-plane θ (deg)', ylabel='Directivity (dBi)', xlim=(-180, 180), ylim=(-30, 25))
        axes[0, 0].legend(fontsize=6.5)
        save(fig, 'cuts')
        fig, axes = sub(height=3.2)
        ax = axes[0, 0]
        for j, f in enumerate(a['freqs_hz']):
            line, = ax.plot(a['iteration'], dbi(a['response'][:, j, 0].mean(axis=-1)), linestyle='-', marker=markers[j], markevery=20, label=f'{f/1e9:g} GHz')
            ax.plot(a['iteration'][best], dbi(a['response'][best, j, 0].mean()), 'k*', ms=10, zorder=10, label='Best' if j == 0 else None)
            gp = args.record / 'grin/iterations.npz'
            if gp.exists():
                with np.load(gp) as g:
                    ax.plot(g['iteration'], dbi(g['response'][:, j, 0].mean(axis=-1)), '--', color=line.get_color(), marker=markers[j], markevery=25, label=f'{f/1e9:g} GHz GRIN start')
        if not gp.exists():
            manifest['missing'].append(str(gp))
        ax.set(xlabel='Iteration (1)', ylabel='Boresight directivity (dBi)')
        ax.legend(ncol=2)
        save(fig, 'iterations')
        mp = args.mesh_record / 'mesh_trend.json' if args.mesh_record else args.record / 'mesh_trend.json'
        if mp.exists():
            fig, axes = sub(3, 3.3)
            for j, ax in enumerate(axes[0]):
                for k, (label, row) in enumerate(read_json(mp).items()):
                    ax.plot([1.5, 1, .75], np.asarray(row['boresight_dbi'])[:, j], styles[k], marker=markers[k], label=lens_label(label))
                ax.set(xlabel='Mesh spacing (mm)', ylabel='Boresight directivity (dBi)')
                ax.text(.95, .08, f'{a["freqs_hz"][j]/1e9:g} GHz', transform=ax.transAxes, ha='right', va='bottom')
            axes[0, 0].legend(fontsize=6)
            save(fig, 'mesh')
        else:
            manifest['missing'].append(str(mp))
    else:
        fig, axes = sub(2, 2.7)
        for ax, k, label in zip(axes[0], [0, best], ['start', f'designed (iteration {a["iteration"][best]})']):
            im = ax.imshow(filter_mirror(a['eps'][k]).T, origin='lower', extent=(59.69, 140.97, 0, 22.86), vmin=1, vmax=10, interpolation='nearest')
            ax.set(xlabel='x (mm)', ylabel='y (mm)')
            ax.text(.02, .95, label, transform=ax.transAxes, va='top', color='white')
        fig.colorbar(im, ax=axes.ravel().tolist(), label=r'$\epsilon_r$ (1)', shrink=.7)
        save(fig, 'maps')
        fig, axes = sub(2, 3.1)
        for c, ax in enumerate(axes[0]):
            mask(ax, c)
            for k, style, label in [(0, '--', 'start'), (best, '-', f'designed (iteration {a["iteration"][best]})')]:
                ax.plot(a['freqs_hz']/1e9, 2*dbi(abs(a['response'][k, c, 0])), style, label=label)
            ax.set_ylabel(f'$|S_{{{c+1}1}}|$ (dB)')
            ax.legend()
        save(fig, 'sparams')
        fig, axes = sub()
        ax = axes[0, 0]
        ax.plot(a['iteration'], a['objective'], 'k-')
        ax.plot(a['iteration'][best], a['objective'][best], 'k*', ms=9, label='Best')
        ax.legend()
        ax.set(xlabel='Iteration (1)', ylabel='Objective (dB²)')
        save(fig, 'objective')
        fig, axes = sub()
        ax = axes[0, 0]
        for k, style, label in [(0, '--', 'start'), (best, '-', f'designed (iteration {a["iteration"][best]})')]:
            s = a['response'][k]
            ax.plot(a['freqs_hz']/1e9, abs(s[0, 0])**2 + abs(s[1, 0])**2 - 1, style, label=label)
        ax.axhline(0, color='gray', ls=':')
        ax.set(xlabel='Frequency (GHz)', ylabel=r'$|S_{11}|^2+|S_{21}|^2-1$ (1)')
        ax.legend()
        save(fig, 'power')
        bp = args.record / 'baselines.npz'
        if bp.exists():
            with np.load(bp) as b:
                manifest['baseline_keys'] = b.files
        mesh_root = args.mesh_record or args.record
        mp = mesh_root / 'mesh_trend.json'
        if mp.exists():
            fig, axes = sub(2, 3.1)
            for c, ax in enumerate(axes[0]):
                mask(ax, c)
                for j, (dx, label) in enumerate(FILTER_MESHES):
                    source = mesh_root / f'mesh_{dx}' / 'resolve.npz'
                    with np.load(source, allow_pickle=False) as resolved:
                        ax.plot(a['freqs_hz']/1e9, amplitude_db(resolved['reported'][c, 0]), styles[j], label=label)
                ax.set_ylabel(f'$|S_{{{c+1}1}}|$ (dB)')
                ax.legend(fontsize=6.5)
            manifest['mesh_sources'] = [str(mesh_root / f'mesh_{dx}' / 'resolve.npz') for dx, _ in FILTER_MESHES]
            save(fig, 'mesh')
        else:
            manifest['missing'].append(str(mp))
    fig, axes = sub(2, 3.5)
    fp = args.record / 'fd_float32.json'
    if fp.exists():
        rows = read_json(fp)['judgement']['rows']
        for j, (name, row) in enumerate(rows.items()):
            if row['judged']:
                pairs = sorted((float(k), v) for k, v in row['rel_by_step'].items())
                axes[0, 0].semilogy(*np.array(pairs).T, styles[j], marker=markers[j], label=f'Pixel {j+1}')
        axes[0, 0].legend(fontsize=6)
    else:
        manifest['missing'].append(str(fp))
    axes[0, 0].set(xlabel=r'FD step in $\epsilon_r$ (1)', ylabel='AD–FD relative error (1)')
    for j, state in enumerate(['start', 'reported']):
        rp = args.record / f'rlw_{state}.json'
        if rp.exists():
            r = read_json(rp)
            vals = {'norm': r['worst'], **r['per_fd_pixel']}
            axes[0, 1].semilogy(range(len(vals)), np.maximum(list(vals.values()), 1e-12), styles[j], marker=markers[j], label='start' if state == 'start' else ('designed lens' if kind == 'lens' else f'designed (iteration {a["iteration"][best]})'))
            axes[0, 1].set_xticks(range(len(vals)), ['Norm'] + [f'Pixel {k+1}' for k in range(len(vals)-1)])
            manifest.setdefault('rlw_pixel_labels', {})[state] = list(vals)
        else:
            manifest['missing'].append(str(rp))
    axes[0, 1].set(xlabel='Record-length comparison (1)', ylabel='Gradient relative change (1)')
    axes[0, 1].legend()
    for ax in axes[0]:
        ax.axhline(.05, color='k', ls=':', label='0.05')
    save(fig, 'gradients')
    write_manifest(args.out, f'{kind}_figures_manifest.json', manifest)


if __name__ == '__main__':
    main()
