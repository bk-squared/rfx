#!/usr/bin/env python3
"""Assemble the local 52f0a897c27d share batch; never upload anything.

Run with --lens-film and --created. All output is restricted
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
OLD_BATCH = 'ca3dd98f4722'
OLD = f'https://remilab.cnu.ac.kr/share/{OLD_BATCH}/'
NEW = '52f0a897c27d'
FILM = 'grin-lens/grin_lens_film.mp4'
# rfx-research showcase 355c82ebf, remi job showcase-lens-renarration-compose-kl98m.
LENS_FILM_SHA256 = '46f36d8c94ae4610643ed62201deef26f8e9574c1313a76037ff7d4619048b63'


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
    cache = LOCAL / 'share' / OLD_BATCH
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
    if digest(args.lens_film)['sha256'] != LENS_FILM_SHA256:
        raise ValueError('lens film: SHA-256 differs from the approved re-narration')
    target = output / FILM
    shutil.copyfile(args.lens_film, target)
    if digest(target)['sha256'] != LENS_FILM_SHA256:
        raise ValueError('lens film: copied SHA-256 differs from the approved re-narration')
    if 'grin_lens_film.mp4' not in old['cases']['grin-lens']['files']:
        raise ValueError('lens film: the old batch has no film to replace')
    manifest['cases']['grin-lens']['files']['grin_lens_film.mp4'] = digest(target)
    manifest['supersedes'] = {
        'batch': OLD_BATCH, 'added': [], 'replaced': [FILM],
        'note': "the lens film re-narrated per the PI's 2026-10-18 direction "
                '(what is optimised, by which variables, the gradient that makes it possible, '
                'how the result is checked; no comparison with another lens); '
                'every other file byte-identical',
    }
    fetch('index.html', cache / 'index.html')
    index = (cache / 'index.html').read_text()
    section = f'<section><h2>Replaced lens film</h2><ul><li><a href="{FILM}">{FILM}</a></li></ul></section>'
    if index.count('</main>') != 1:
        raise ValueError('old index.html has no single </main> to extend')
    (output / 'index.html').write_text(index.replace('</main>', section + '</main>'))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{output}: {sum(len(c["files"]) for c in manifest["cases"].values())} files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lens-film', type=Path, required=True,
                        help='approved re-narrated lens film (SHA-256 checked)')
    parser.add_argument('--created', required=True, help='upload date, YYYY-MM-DD')
    parser.add_argument('--output-dir', type=Path, default=LOCAL / 'share' / NEW)
    assemble(parser.parse_args())


if __name__ == '__main__':
    main()
