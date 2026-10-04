"""Sequential stored-array PyVista films; no solves or synthetic response frames."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from _visual_data import lens_mirror, filter_mirror, dbi, load_record, write_manifest, read_json, final_hold_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mesh-record', type=Path)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--preview', action='store_true', help='Render first and last only; do not encode')
    parser.add_argument('--max-render-minutes', type=float, default=40)
    args = parser.parse_args()
    if not args.write:
        parser.error('--write required')
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VTK_SMP_MAX_THREADS', 'LP_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['MPLCONFIGDIR'] = str(args.out / '.mplconfig')
    os.environ['MESA_SHADER_CACHE_DIR'] = str(args.out / '.mesa_cache')
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkMultiThreader
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from figures_lens_filter import mask
    import sys
    if sys.platform == 'darwin':
        from _showcase_mesa import install
        install()
    vtkMultiThreader.SetGlobalMaximumNumberOfThreads(1)
    vtkMultiThreader.SetGlobalDefaultNumberOfThreads(1)
    a = load_record(args.record)
    kind = 'lens' if a['eps'].ndim == 4 else 'filter'
    stage = args.out / f'{kind}_frames'
    stage.mkdir(exist_ok=True)
    p = pv.Plotter(off_screen=True, window_size=(1920, 1080), lighting='none')
    p.set_background('#081019', top='#26394a')
    p.enable_anti_aliasing('fxaa')
    p.renderer.SetUseDepthPeeling(False)
    p.renderer.SetUseOIT(True)
    for pos, color, strength in [((180, -250, 400), '#fff1d9', 1.3), ((-200, 80, 200), '#a3caff', .8)]:
        p.add_light(pv.Light(position=pos, focal_point=(0, 0, 50), color=color, intensity=strength, light_type='scene light'))
    amber = LinearSegmentedColormap.from_list('material', ['#d0e8ef', '#ffa329'])
    def metal(bounds, opacity=1):
        p.add_mesh(pv.Box(bounds=bounds).triangulate(), color='#9ba9b5', opacity=opacity, pbr=True, metallic=.9, roughness=.43)
    if kind == 'lens':
        metal((-15, 15, -15, 15, -.6, 0))
        grooves = [pv.Line((-15, y, .03), (15, y, .03)) for y in np.linspace(-15, 15, 65)]
        p.add_mesh(pv.merge(grooves), color='#e2e6ec', opacity=.16, line_width=.5, lighting=False)
        p.add_mesh(pv.Cylinder(center=(0, 0, 9), direction=(1, 0, 0), radius=.8, height=15), color='#cc8851', pbr=True, metallic=.8, roughness=.3)
        p.add_mesh(pv.Line((0, 0, 9), (0, 0, 185)), color='#ffdc92', line_width=1, opacity=.7)
        grid = np.stack(np.meshgrid(np.arange(30)*3-43.5, np.arange(30)*3-43.5, np.arange(10)*3+55.5, indexing='ij'), axis=-1).reshape(-1, 3)
        dimensions = (3, 3, 3)
        upper = 2.7
        theta = np.linspace(1e-4, np.pi-1e-4, 73)
        theta[0] = 0
        th, ph = np.meshgrid(theta, np.linspace(0, 2*np.pi, 73), indexing='ij')
        directions = np.stack((np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)), axis=-1)
        fi = int(np.argmin(abs(a['freqs_hz']-1e10)))
        scale = 175 / float(a['response'][:, fi].max())
        p.camera.position = (220, -320, 390)
        p.camera.focal_point = (0, 0, 90)
        p.camera.up = (0, 0, 1)
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 78
    else:
        metal((0, 200.66, 0, 22.86, -.7, 0), .55)
        metal((0, 200.66, -.7, 0, 0, 10.16), .25)
        metal((0, 200.66, 22.86, 23.56, 0, 10.16), .25)
        grid = np.stack(np.meshgrid(np.arange(32)*2.54+60.96, np.arange(9)*2.54+1.27, [5.08], indexing='ij'), axis=-1).reshape(-1, 3)
        dimensions = (2.54, 2.54, 10.16)
        upper = 10
        scale = None
        p.camera.position = (235, -175, 215)
        p.camera.focal_point = (100.33, 11.43, 0)
        p.camera.up = (0, 0, 1)
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 76
    p.camera_position = (p.camera.position, p.camera.focal_point, (0, 0, 1))
    text = p.add_text('', position=(75, 65), font_size=22, font='arial', color='#eef2f6')
    note = p.add_text('', position=(75, 30), font_size=14, font='arial', color='#9ca3aa')
    mesh_path = (args.mesh_record or args.record) / 'mesh_trend.json'
    mesh_root = mesh_path.parent
    final_note = ''
    if mesh_path.exists() and (mesh_root / 'eligibility.json').exists() and (mesh_root / 'stage.json').exists():
        final_note = final_hold_text(kind, read_json(mesh_path), a['freqs_hz'],
                                     read_json(mesh_root / 'eligibility.json'),
                                     read_json(mesh_root / 'stage.json').get('best_iteration'),
                                     int(a['iteration'][int(np.argmin(a['objective']))]))
    hud_strings = {}
    voxel_actor = None
    lobe_actor = None
    def render(i, final_hold=False):
        nonlocal voxel_actor, lobe_actor
        begin = time.perf_counter()
        note.SetInput(final_note if final_hold else '')
        if voxel_actor is not None:
            p.remove_actor(voxel_actor, render=False)
        eps = (lens_mirror(a['eps'][i]) if kind == 'lens' else filter_mirror(a['eps'][i])).ravel()
        keep = eps > 1.05
        cube = pv.Cube(x_length=dimensions[0], y_length=dimensions[1], z_length=dimensions[2])
        centers = grid[keep]
        nv = cube.n_points
        points = (cube.points[None]+centers[:, None]).reshape(-1, 3)
        faces = np.tile(cube.faces.reshape(-1, 5), (len(centers), 1, 1))
        faces[:, :, 1:] += np.arange(len(centers))[:, None, None]*nv
        mesh = pv.PolyData(points, faces.ravel())
        mesh['eps'] = np.repeat(eps[keep], nv)
        if len(centers):
            voxel_actor = p.add_mesh(mesh, scalars='eps', cmap=amber, clim=(1, upper), opacity=(255*np.linspace(.08, .9, 256)).astype(np.uint8), show_scalar_bar=False, pbr=True, metallic=.05, roughness=.3, render=False)
        if kind == 'lens':
            if lobe_actor is not None:
                p.remove_actor(lobe_actor, render=False)
            d = a['response'][i, fi]
            pts = directions*d[:, :, None]*scale + np.array([0, 0, 9])
            lobe = pv.StructuredGrid(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2])
            lobe['dbi'] = dbi(d).ravel(order='F')
            lobe_actor = p.add_mesh(lobe, scalars='dbi', cmap='plasma', clim=(-10, 21), opacity=.24, show_scalar_bar=False, smooth_shading=True, ambient=.65, render=False)
            text.SetInput(f'iteration {a["iteration"][i]}\nboresight directivity {dbi(d[0].mean()):.1f} dBi')
        else:
            text.SetInput(f'iteration {a["iteration"][i]}')
        hud_strings[int(a["iteration"][i])] = text.GetInput()
        p.reset_camera_clipping_range()
        p.render()
        frame = Image.fromarray(p.screenshot(return_img=True))
        if kind == 'filter':
            fig, ax = plt.subplots(figsize=(7.2, 3.15), dpi=100, layout='constrained')
            mask(ax)
            for c, style in [(0, '-'), (1, '--')]:
                ax.plot(a['freqs_hz']/1e9, 2*dbi(abs(a['response'][i, c, 0])), style, label=f'$|S_{{{c+1}1}}|$')
            ax.legend(loc='lower left', ncol=2)
            fig.canvas.draw()
            inset = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB')
            frame.paste(inset, (1160, 40))
            plt.close(fig)
        path = stage / (f'iterate_{i:04d}_hold.png' if final_hold else f'iterate_{i:04d}.png')
        frame.save(path)
        return path, time.perf_counter()-begin
    manifest = dict(record=str(args.record), kind=kind, display_conventions=dict(resolution=[1920, 1080], social_resolution=[1080, 1350], fps=30, duration_s=9, endpoint_holds_s=1, temporal='nearest stored iterate; no interpolation; repeated frames hardlinked', background=['#081019', '#26394a'], camera=list(p.camera.position), focal_point=list(p.camera.focal_point), camera_up=[0, 0, 1], parallel_scale=p.camera.parallel_scale, lights='positions (180,-250,400),(-200,80,200); colors #fff1d9/#a3caff; intensities 1.3/.8', antialias='FXAA; OIT alpha blending; no depth peeling', metal='gray #9ba9b5; metallic .9; roughness .43; lens plate thickness .6 mm display only; 65 decorative grooves', dipole='point source represented by x rod length 15 mm radius .8 mm; copper #cc8851', guide='upper broad wall removed; bottom opacity .55; side opacity .25; display wall thickness .7 mm', voxel='full physical pixel dimensions; eps<=1.05 omitted; pale #d0e8ef to amber #ffa329; opacity .08 to .9; metallic .05 roughness .3', lobe='10 GHz radius proportional to linear D; plasma dBi coloring [-10,21]; opacity .24; source origin; smooth shading; ambient .65', lobe_mm_per_linear_D=scale, boresight_ray='source to z=185 mm; 1 px; #ffdc92 opacity .7', hud='lower left (75,65), Arial 22, #eef2f6; values from stored design-mesh response', filter_inset='720x315 at (1160,40); white matplotlib panel; S11 solid/S21 dashed; -65..5 dB; gray hatched forbidden masks', social='landscape fitted 1080x608 centered on solid #081019 portrait canvas', encoding='libx264 CRF19 medium; yuv420p; 1 thread; faststart'), timings=[], outputs=[], subsampled=False)
    started = time.perf_counter()
    try:
        first, seconds = render(0)
        max_unique = 2 if args.preview else max(2, min(len(a['iteration']), int(args.max_render_minutes*60/max(seconds*1.5, .1))))
        # The film ends on the reported design: the best iterate (earliest
        # objective minimum), never on a later, worse one.
        best = int(np.argmin(a['objective']))
        chosen = np.unique(np.rint(np.linspace(0, best, max_unique)).astype(int))
        manifest['final_iterate'] = int(a['iteration'][best])
        manifest['subsampled'] = len(chosen) < len(a['iteration'])
        manifest['stored_iterates'] = len(a['iteration'])
        manifest['rendered_iterates'] = len(chosen)
        cache = {0: first}
        elapsed = {0: seconds}
        for i in chosen[1:]:
            path, sec = render(int(i))
            cache[int(i)] = path
            elapsed[int(i)] = sec
            print(f'{kind} iterate {i}: {sec:.3f}s', flush=True)
        hold_path, hold_seconds = render(int(chosen[-1]), final_hold=True)
        manifest['final_hold_render_seconds'] = hold_seconds
        manifest['hud_strings'] = hud_strings
        manifest['final_hold_text'] = final_note
        schedule = np.r_[np.zeros(30, dtype=int), np.rint(np.linspace(0, len(chosen)-1, 210)).astype(int), np.full(30, len(chosen)-1)]
        seen = set()
        for frame, ci in enumerate(schedule):
            i = int(chosen[ci])
            dest = stage / f'frame_{frame:04d}.png'
            if dest.exists():
                dest.unlink()
            os.link(hold_path if frame >= 240 else cache[i], dest)
            manifest['timings'].append(dict(frame=frame, iteration=int(a['iteration'][i]), render_seconds=elapsed[i] if i not in seen else 0, reused=i in seen))
            seen.add(i)
        final = Image.open(hold_path)
        final.save(args.out / f'{kind}_poster.png')
        social = Image.new('RGB', (1080, 1350), '#081019')
        social.paste(final.resize((1080, 608), Image.Resampling.LANCZOS), (0, 371))
        social.save(args.out / f'{kind}_social_poster.png')
    finally:
        window = p.render_window
        p.close()
        if hasattr(window, 'release_cgl'):
            window.release_cgl()
        manifest['render_wall_seconds'] = time.perf_counter()-started
        write_manifest(args.out, f'{kind}_film_manifest.json', manifest)
    if args.preview:
        return
    for suffix, vf in [('', 'null'), ('_social', 'scale=1080:608:flags=lanczos,pad=1080:1350:0:371:color=0x081019')]:
        target = args.out / f'{kind}{suffix}.mp4'
        subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-v', 'error', '-threads', '1', '-framerate', '30', '-i', str(stage/'frame_%04d.png'), '-vf', vf, '-filter_threads', '1', '-c:v', 'libx264', '-threads', '1', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
        manifest['outputs'].append(dict(path=target.name, bytes=target.stat().st_size, duration_s=9))
    write_manifest(args.out, f'{kind}_film_manifest.json', manifest)


if __name__ == '__main__':
    main()
