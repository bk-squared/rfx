"""Freeze the S1 reference before source changes, from fa3fcc73f only."""

from pathlib import Path
import subprocess

import numpy as np

from rfx.geometry.conformal_cells import compute_conformal_fractions
from tests._cell_table_scenes import SCENES, cell_table_scene


def main():
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    assert head == 'fa3fcc73fd8c2d83e2dc52f3240f8f2ea9feec33', head
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', 'rfx/'], cwd=root)
    arrays = {}
    for name in SCENES:
        grid, shapes = cell_table_scene(name)
        for rule, kwargs in (('sc', {}), ('raw', {'clamp': False}),
                             ('floor', {'conformal_pec': 'cut_cell_floor'})):
            result = compute_conformal_fractions(grid, shapes, **kwargs)
            for kind, values in zip(('edges', 'faces'), result):
                for axis, value in enumerate(values):
                    arrays[f'{name}_{rule}_{kind}_{axis}'] = np.asarray(value)
    target = Path(__file__).with_name('cell_table_fa3fcc73f.npz')
    np.savez_compressed(target, **arrays)
    print(f'{target.name}: {target.stat().st_size} bytes; {len(arrays)} arrays; HEAD={head}')


if __name__ == '__main__':
    main()
