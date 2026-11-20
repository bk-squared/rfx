"""Freeze off-node Box operands and capture calls from commit dea200684."""

from pathlib import Path
import subprocess

import numpy as np

from rfx.geometry.conformal_cells import compute_conformal_fractions
from tests._cell_table_capture import record_cell_table_capture
from tests._cell_table_scenes import cell_table_scene


def main():
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    assert head == 'dea200684c816e6dac508c4ebe3e001ecd9e92c1', head
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', 'rfx/'], cwd=root)
    arrays, captures = {}, {}
    for name in ('box_off',):
        grid, shapes = cell_table_scene(name)
        for rule, kwargs in (('sc', {}), ('raw', {'clamp': False}),
                             ('floor', {'conformal_pec': 'cut_cell_floor'})):
            with record_cell_table_capture(grid, shapes) as calls:
                result = compute_conformal_fractions(grid, pec_shapes=shapes, **kwargs, field_dtype=np.float32)
            for kind, values in zip(('edges', 'faces'), result):
                for axis, value in enumerate(values):
                    arrays[f'{name}_{rule}_{kind}_{axis}'] = np.asarray(value)
            captures[f'{name}_{rule}_call_count'] = np.asarray(len(calls))
            for index, call in enumerate(calls):
                for argument, value in call.items():
                    captures[f'{name}_{rule}_{index}_{argument}'] = value
    for filename, values in (('cell_table_box_off_dea200684.npz', arrays),
                             ('cell_table_capture_box_off_dea200684.npz', captures)):
        target = Path(__file__).with_name(filename)
        np.savez_compressed(target, **values)
        print(f'{target.name}: {target.stat().st_size} bytes; {len(values)} arrays; HEAD={head}')


if __name__ == '__main__':
    main()
