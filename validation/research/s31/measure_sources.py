"""S3-1: unchanged boards, warmed marginal wall ms/step, one arm/process."""
import argparse
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board', choices=['msl', 'wire'], required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert jax.__version__ == '0.10.2'
    assert jax.default_backend() == 'gpu' and 'A6000' in jax.devices()[0].device_kind
    sys.path.insert(0, str(a.inputs))
    if a.board == 'wire':
        import d1_scene as scene
        from A3c import freeze_d1_scene
        # Read the frozen tutorial beside the copied scene inputs.
        scene.TUTORIAL = a.inputs / 'nonuniform_patch_demo.py'
        freeze_d1_scene.apply(scene)
        sim, metadata = scene.build_arm('V6', .00025)
        from A3d.arm_scene import configure
        configure(sim, metadata, 'G-wire', scene)
    else:
        from p3_throughput_model import make_model
        sim, metadata = make_model(json.loads((a.inputs/'params.json').read_text()), port='msl')
    sim._snap = 'declared'
    grid = sim._build_nonuniform_grid()
    record = dict(board=a.board, shape=grid.shape, cells=int(np.prod(grid.shape)),
                  jax=jax.__version__, device=jax.devices()[0].device_kind,
                  n1=200, n2=1200, repeats=3, measurements=[], interpretation='lead fills')
    if a.board == 'wire':
        expected = json.loads((a.inputs/'A3c/expected_cells.json').read_text())['V6_0.25mm']
        record['expected'] = expected
        assert list(grid.shape) == expected['padded_shape']
        assert record['cells'] == expected['cells']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.out.write_text(json.dumps(record, indent=2, default=str)+'\n')
    def timed(n):
        start = time.perf_counter()
        r = sim.run(n_steps=n, compute_s_params=False)
        for v in jax.tree.leaves(r):
            if hasattr(v, 'block_until_ready'):
                v.block_until_ready()
        return time.perf_counter()-start
    save()
    record['warmup'] = [timed(200), timed(1200)]
    save()
    for _ in range(3):
        short, long = timed(200), timed(1200)
        record['measurements'].append(dict(short_s=short, long_s=long,
                                            marginal_ms=(long-short)))
        save()
    vals = [v['marginal_ms'] for v in record['measurements']]
    record.update(marginal_ms=float(np.median(vals)), spread_ms=float(np.ptp(vals)), status='completed')
    save()
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
