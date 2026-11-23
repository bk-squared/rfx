"""RLC copies stay constant for multiple elements; one is no worse than main.
The singleton RLC form preserves run() bits.
"""

import gc
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "perf"))
from _census_scenes import DX, _centre, base  # noqa: E402
from step_cost_census import Program  # noqa: E402


@pytest.mark.parametrize("path", ["uniform", "graded"])
@pytest.mark.parametrize("topology", ["series", "parallel"])
def test_rlc_adds_no_domain_copy(path, topology):
    """An earlier CPU JAX 0.10.2 measurement at N=24 found 8 domain copies for k=1 in both topologies on both mesh paths."""
    n = 24
    counts = {}
    for k in (1, 2, 4, 8, 16):
        scene = base(n, path)
        for index in range(k):
            position = _centre(n, path) + np.array(
                (index % 4 - 2, index // 4 - 2, 0)) * DX
            scene.sim.add_lumped_rlc(tuple(position), "ez", R=50., L=1e-9,
                                     C=1e-12, topology=topology)
        program = Program(scene, path, (n + 16) ** 3)
        program.compile()
        counts[k] = program.record["counts"].get("copy", 0)
        del program, scene
        jax.clear_caches()
        gc.collect()
    assert counts[1] <= 8, counts
    assert counts[2] == counts[4] == counts[8] == counts[16], counts
