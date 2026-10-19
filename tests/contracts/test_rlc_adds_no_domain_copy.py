"""RLC copies stay constant for multiple elements; one is no worse than main."""

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
    # Small CPU compile, with all elements interior and on distinct ez edges.
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
        # Program uses the census opcode counter and its domain-size threshold.
        counts[k] = program.record["counts"].get("copy", 0)
        del program, scene
        jax.clear_caches()
        gc.collect()
    # CPU JAX 0.10.2, origin/main 4e54aa7c, measured at this N=24:
    # k=1 has 8 domain copies in both topologies on both mesh paths.
    # Leader decision: retain the singleton form that preserves run() bits.
    assert counts[1] <= 8, counts
    assert counts[2] == counts[4] == counts[8] == counts[16], counts
