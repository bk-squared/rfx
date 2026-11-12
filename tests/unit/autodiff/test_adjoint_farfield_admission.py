"""NTFF/H admission boundaries, compensation, and unchanged default digests."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import tarfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement.farfield import NTFFBox
from rfx.stepping.adjoint_monitors import monitor_frequencies
from tests._x64_compat import enable_x64
from tests.unit.autodiff.test_adjoint_farfield import scene, collocation_mode
from tests.unit.autodiff.test_design_box_tape import BOX_LO, BOX_HI


def context(box):
    return SimpleNamespace(dft_meta=(), use_ntff=True, ntff=box,
                           grid=SimpleNamespace(shape=(20,)*3, cpml_layers=2),
                           use_cpml=True, dt=1.)


@pytest.mark.parametrize("bins", [[], [0.], [-.1], [.5], [.1, .1], [np.nan]])
def test_ntff_only_bad_bins(bins):
    ctx = context(NTFFBox(4, 14, 4, 14, 4, 14, np.array(bins), face_centre=True))
    with pytest.raises(ValueError, match="distinct positive sub-Nyquist"):
        monitor_frequencies(ctx)


def test_ntff_halo_and_identical_plane_bins():
    from rfx.measurement.plan import field_channel
    ctx = context(NTFFBox(4, 14, 4, 14, 4, 14, np.array([.1, .2]), face_centre=True))
    np.testing.assert_array_equal(monitor_frequencies(ctx), [.1, .2])
    ctx.ntff = ctx.ntff._replace(i_lo=3)
    with pytest.raises(NotImplementedError, match="interior monitor"):
        monitor_frequencies(ctx)
    ctx.ntff = ctx.ntff._replace(face_centre=False)
    monitor_frequencies(ctx)  # node layout has no backward halo
    ctx.dft_meta = ((field_channel("hy"), 0, 10, np.array([.1]), (4, 8, 4, 8)),)
    with pytest.raises(NotImplementedError, match="identical monitor"):
        monitor_frequencies(ctx)
    ctx.dft_meta = ((field_channel("hy"), 0, 10, np.array([.1, .2]), (4, 8, 4, 8)),)
    monitor_frequencies(ctx)


@pytest.mark.slow
def test_compensation_objective_zero_and_adjoint_carry(record_property):
    from unittest.mock import patch
    seen = []
    # Inspect the production second scan without changing its carry or sources.
    from rfx.stepping import uniform_setup
    original = uniform_setup.make_core_step
    def make(ctx, *args, **kwargs):
        core = original(ctx, *args, **kwargs)
        def step(carry, *row):
            if ctx.drives is None:
                seen.append((ctx.use_ntff, "ntff" in carry, "dft_planes" in carry))
            return core(carry, *row)
        return step
    with enable_x64(), collocation_mode("face_centre"), patch.object(uniform_setup, "make_core_step", make):
        sim, eps = scene("ntff", "float64", False)
        def loss(e):
            result = sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=e,
                                 n_steps=256, gradient="adjoint", skip_preflight=True)
            return sum(jnp.real(jnp.sum(a)) for a in result.ntff_data[6:])
        grad = jax.jit(jax.grad(loss))(eps)
        np.testing.assert_array_equal(grad, jnp.zeros_like(eps))
        assert seen and all(row == (False, False, False) for row in seen)
        record_property("adjoint_carry", seen)


@pytest.mark.slow
def test_g5_default_digests_against_main(tmp_path, record_property):
    """Run the same independent scene in clean subprocesses at main and HEAD."""
    root = Path(__file__).resolve().parents[3]
    baseline = tmp_path / "main"
    baseline.mkdir()
    archive = tmp_path / "main.tar"
    with archive.open("wb") as out:
        subprocess.run(["git", "archive", "cd5bec868"], cwd=root, stdout=out, check=True)
    with tarfile.open(archive) as packed:
        packed.extractall(baseline, filter="data")
    program = Path(__file__).with_name("_adjoint_default_digest.py").read_text()
    digests = []
    start = time.monotonic()
    for tree in (baseline, root):
        env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1"}
        output = subprocess.check_output([sys.executable, "-c", program], cwd=tree, env=env, text=True)
        digests.append(json.loads(output.strip().splitlines()[-1]))
    record_property("digests_main", json.dumps(digests[0], sort_keys=True))
    record_property("digests_head", json.dumps(digests[1], sort_keys=True))
    record_property("seconds", time.monotonic()-start)
    assert digests[0].keys() == {"fields", "probes", "dft", "ntff"}
    assert digests[1] == digests[0], "G5 default path SHA-256 changed"
