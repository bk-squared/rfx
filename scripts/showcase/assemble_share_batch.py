#!/usr/bin/env python3
"""Assemble the local f3a80981ec4b share batch; never upload anything.

Run with --archive-root, --assets-dir, --lens-film-dir and --fan-iterations. All output is restricted
to .local_checks. Old assets are downloaded serially with curl and
accepted only after matching both the old manifest's size and SHA-256.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / '.local_checks'
OLD = 'https://remilab.cnu.ac.kr/share/7c02ad43c580/'
RUNS = {
    'grin-lens': ('lens', '20261004T182324Z-b184fc8e-main', '20261004T205610Z-b184fc8e-resolve', '4705f69e'),
    # The fan-beam objective on the same lens (amendment 7), computed on a later rfx main.
    'grin-lens-fan': ('lens', '20261010T162447Z-dbf41692-main-fan', '20261010T170039Z-dbf41692-resolve-fan', '9ed83eb8'),
}
#: The waveguide filter waits for visuals in the site tone (rfx-lead, 2026-10-15); its page is on the
#: branch showcase/site-filter-held.
#: The lens page shows the narrated film (rfx-research ``showcase/film``, composed at 9b87e97ce by remi
#: job showcase-lens-final-compose-h2zrb) and three of its output frames: published name -> name in --lens-film-dir.
LENS_FILM = {'grin_lens_film.mp4': 'lens_film.mp4', 'grin_lens_film_poster.png': 'frame_2291.png',
             'grin_lens_start.png': 'frame_1920.png', 'grin_lens_fan.png': 'frame_2760.png'}
LENS_FILM_SHA256 = '8b87491c6d68ca14940a1484e1d466336f9df823140bd9fec6967e7ac61bfc97'


def digest(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def fetch(name, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                    '--max-time', '60', OLD + name, '--output', str(target)], check=True)


def assemble(args):
    output = args.output_dir.resolve()
    if not output.is_relative_to(LOCAL.resolve()):
        raise ValueError('output must be under .local_checks')
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous manifest before changing its inputs.
    (output / 'manifest.json').unlink(missing_ok=True)
    cache = LOCAL / 'share/7c02ad43c580'
    fetch('manifest.json', cache / 'manifest.json')
    old = json.loads((cache / 'manifest.json').read_text())
    if old['schema'] != 'rfx-showcase-share/1':
        raise ValueError('unexpected old manifest schema')
    manifest = copy.deepcopy(old)
    manifest['created'] = args.created
    for folder, case in old['cases'].items():
        for name, expected in case['files'].items():
            target = output / folder / name
            if not target.is_file() or digest(target) != expected:
                fetch(f'{folder}/{name}', target)
            if digest(target) != expected:
                raise ValueError(f'{folder}/{name}: size or SHA-256 mismatch')
    added = []
    for folder, (short, main, resolve, solver) in RUNS.items():
        # The rfx main commit whose solver code the record's branch commit carries.
        commit = subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', solver + '^{commit}'], text=True).strip()
        dest = output / folder
        dest.mkdir(exist_ok=True)
        runs = args.archive_root / f'20261004-showcase-{short}' / 'runs'
        sources = {name: runs / main / name for name in
                   ('result.json', 'iterations.npz', 'fd_float32.json', 'rlw_start.json', 'rlw_reported.json')}
        sources.update({name: runs / resolve / name for name in ('mesh_trend.json', 'eligibility.json')})
        if folder == 'grin-lens':
            sources.update({f'{short}_{name}.png': args.assets_dir / 'figures' / f'{short}_{name}.png'
                            for name in ('cuts', 'mesh', 'slices', 'iterations')})
            sources.update({name: args.lens_film_dir / source for name, source in LENS_FILM.items()})
        else:
            # The fan run's iterates (the film's fan counters) are not in the archive: 32 MB, kept on NFS.
            sources['iterations.npz'] = args.fan_iterations
            # Its central-difference check passed on the float64 repeat (the float32 ladder is kept too).
            sources['fd_float64.json'] = runs / main / 'fd_float64.json'
        for name, source in sources.items():
            shutil.copyfile(source, dest / name)
        if folder == 'grin-lens' and digest(dest / 'grin_lens_film.mp4')['sha256'] != LENS_FILM_SHA256:
            raise ValueError('grin_lens_film.mp4 is not the composed final film')
        files = list(sources)
        manifest['cases'][folder] = {'rfx_commit': commit,
                                     'files': {name: digest(dest / name) for name in sorted(files)}}
        added.extend(f'{folder}/{name}' for name in sorted(files))
    manifest['supersedes'] = {
        'batch': '7c02ad43c580', 'added': added, 'replaced': [],
        'note': 'every file of 7c02ad43c580 byte-identical; one new case (issue 1359): the printable lens with its '
                'narrated film and three of its frames (grin-lens/), and the fan-beam design on the same block '
                '(grin-lens-fan/, computed on a later rfx main)',
    }
    fetch('index.html', cache / 'index.html')
    index = (cache / 'index.html').read_text()
    rows = ''.join(f'<li><a href="{name}">{name}</a></li>' for name in added)
    section = '<section><h2>New cases: ' + ', '.join(RUNS) + '</h2><ul>' + rows + '</ul></section>'
    (output / 'index.html').write_text(index.replace('</main>', section + '</main>'))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{output}: {sum(len(c["files"]) for c in manifest["cases"].values())} files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True,
                        help='records directory holding 20261004-showcase-lens/runs')
    parser.add_argument('--assets-dir', type=Path, required=True,
                        help='directory with figures/ (lens_{cuts,mesh,slices,iterations}.png)')
    parser.add_argument('--lens-film-dir', type=Path, required=True,
                        help='directory with lens_film.mp4 and output frames 1920, 2291 and 2760 as PNG')
    parser.add_argument('--fan-iterations', type=Path, required=True,
                        help='iterations.npz of the fan main run 20261010T162447Z-dbf41692-main-fan')
    parser.add_argument('--created', required=True, help='upload date, YYYY-MM-DD')
    parser.add_argument('--output-dir', type=Path, default=LOCAL / 'share/f3a80981ec4b')
    assemble(parser.parse_args())


if __name__ == '__main__':
    main()
