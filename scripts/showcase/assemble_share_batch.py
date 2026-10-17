#!/usr/bin/env python3
"""Assemble the local ca3dd98f4722 share batch; never upload anything.

Run with --archive-root, --assets-dir, --filter-composed-dir. All output is restricted
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
OLD = 'https://remilab.cnu.ac.kr/share/f3a80981ec4b/'
NEW = 'ca3dd98f4722'
RUNS = {
    'waveguide-filter': ('filter', '20261006T162805Z-b8119151-main',
                         '20261006T191443Z-b8119151-resolve', '9ed83eb8'),
}
FILTER_COMPOSED = (
    'waveguide_filter.mp4', 'waveguide_filter_social.mp4', 'waveguide_filter_site.png',
    'waveguide_filter_social.png', 'waveguide_filter_hero_4k.png', 'waveguide_filter_poster.png',
)


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
    cache = LOCAL / 'share/f3a80981ec4b'
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
        sources.update({f'filter_{name}.png': args.assets_dir / 'figures' / f'filter_{name}.png'
                        for name in ('mesh', 'sparams', 'maps', 'objective')})
        checksums = {}
        for line in (args.filter_composed_dir / 'SHA256SUMS').read_text().splitlines():
            checksum, name = line.split(maxsplit=1)
            checksums[name.lstrip('*')] = checksum
        sources.update({name: args.filter_composed_dir / name for name in FILTER_COMPOSED})
        for name, source in sources.items():
            if name in FILTER_COMPOSED and digest(source)['sha256'] != checksums[name]:
                raise ValueError(f'{name}: SHA-256 differs from composed SHA256SUMS')
            shutil.copyfile(source, dest / name)
            if name in FILTER_COMPOSED and digest(dest / name)['sha256'] != checksums[name]:
                raise ValueError(f'{name}: copied SHA-256 differs from composed SHA256SUMS')
        files = list(sources)
        manifest['cases'][folder] = {'rfx_commit': commit,
                                     'files': {name: digest(dest / name) for name in sorted(files)}}
        added.extend(f'{folder}/{name}' for name in sorted(files))
    manifest['supersedes'] = {
        'batch': 'f3a80981ec4b', 'added': added, 'replaced': [],
        'note': 'every file of f3a80981ec4b byte-identical; one new case, the printed waveguide filter in the site tone',
    }
    fetch('index.html', cache / 'index.html')
    index = (cache / 'index.html').read_text()
    rows = ''.join(f'<li><a href="{name}">{name}</a></li>' for name in added)
    section = '<section><h2>New cases: ' + ', '.join(RUNS) + '</h2><ul>' + rows + '</ul></section>'
    if index.count('</main>') != 1:
        raise ValueError('old index.html has no single </main> to extend')
    (output / 'index.html').write_text(index.replace('</main>', section + '</main>'))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{output}: {sum(len(c["files"]) for c in manifest["cases"].values())} files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True,
                        help='records directory holding 20261004-showcase-filter/runs')
    parser.add_argument('--assets-dir', type=Path, required=True,
                        help='directory with figures/ (filter_{mesh,sparams,maps,objective}.png)')
    parser.add_argument('--filter-composed-dir', type=Path, required=True,
                        help='directory with the six composed filter outputs and SHA256SUMS')
    parser.add_argument('--created', required=True, help='upload date, YYYY-MM-DD')
    parser.add_argument('--output-dir', type=Path, default=LOCAL / 'share' / NEW)
    assemble(parser.parse_args())


if __name__ == '__main__':
    main()
