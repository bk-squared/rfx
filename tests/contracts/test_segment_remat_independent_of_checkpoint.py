"""Segment remat is independent of the per-step checkpoint flag (#1789)."""
import ast
import inspect
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement.farfield import compute_far_field
from tests.contracts.test_ad_memory_tape_terms import _scene


def _compile(n, *, box, checkpoint, k=None, graded=False, observations=False):
    """CPU compiler oracle; deliberately independent of estimator helpers."""
    sim, grid, corners, shape = _scene(graded=graded)
    if observations:
        sim.add_probe((.021, .012, .012), "ez")
    knobs = {"checkpoint_every" if graded else "checkpoint_segments": k} if k else {}

    def objective(p):
        arr = jnp.ones(shape if box else grid.shape, jnp.float32)
        overrides = (dict(design_box=corners, design_eps_override=arr * p[0],
                          design_sigma_override=arr * p[1] * 1e-3) if box else
                     dict(eps_override=arr * p[0], sigma_override=arr * p[1] * 1e-3))
        result = sim.forward(n_steps=n, checkpoint=checkpoint, skip_preflight=True,
                             **overrides, **knobs)
        if graded:
            value = jnp.sum(result.time_series ** 2)
        else:
            ff = compute_far_field(result.ntff_data, result.ntff_box, grid,
                                   jnp.array([jnp.pi / 2]), jnp.array([jnp.pi]))
            value = jnp.sum(jnp.abs(ff.E_theta) ** 2)
        return value, result.time_series

    with jax.default_device(jax.devices("cpu")[0]):
        params = jnp.array([4. if box else 1., .2 if box else 0.], jnp.float32)
        # Exclude auxiliary observations from the memory oracle, matching #1786.
        executable = jax.jit(jax.value_and_grad(lambda p: objective(p)[0])).lower(params).compile()
        memory = executable.memory_analysis()
    compiled = (memory.argument_size_in_bytes + memory.output_size_in_bytes
                + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
    print("SEGMENT_REMAT_ROW " + json.dumps(dict(n=n, box=box, checkpoint=checkpoint,
          k=k, graded=graded, compiled_bytes=compiled)), file=sys.stderr, flush=True)
    return compiled, executable, params, objective


def _bit_identical(a, b):
    for left, right in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))
        assert left.shape == right.shape and left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()


@pytest.mark.parametrize("path", ("uniform", "graded",
    pytest.param("distributed-graded", marks=pytest.mark.source_scan), "distributed-uniform"))
def test_segment_remat_independent_of_checkpoint(path):
    if path == "uniform":
        rows = []
        for box in (False, True):
            baseline = _compile(800, box=box, checkpoint=False)[0]
            for k in (4, 8):
                off = _compile(800, box=box, checkpoint=False, k=k)[0]
                on = _compile(800, box=box, checkpoint=True, k=k)[0]
                rows.append((box, k, baseline, off, on))
                jax.clear_caches()
        # Collect every memory row before failing, so a mutation gives all numbers.
        for box, k, baseline, off, on in rows:
            assert abs(off - on) <= .01 * on, (box, k, baseline, off, on)
            assert max(off, on) < baseline, (box, k, baseline, off, on)
        for box in (False, True):
            results = []
            for checkpoint in (False, True):
                _, executable, params, objective = _compile(64, box=box, checkpoint=checkpoint,
                                                           k=8, observations=True)
                results.append((executable(params), jax.jit(objective)(params)[1]))
            _bit_identical(*results)
            assert np.any(np.asarray(results[0][0][1]) != 0)
            assert np.any(np.asarray(results[0][1]) != 0)
            print("SEGMENT_REMAT_BITS " + json.dumps(dict(box=box, value=True,
                  gradient=True, time_series=True)), file=sys.stderr, flush=True)
            jax.clear_caches()
    elif path == "graded":
        off = _compile(800, box=True, checkpoint=False, k=40, graded=True)[0]
        on = _compile(800, box=True, checkpoint=True, k=40, graded=True)[0]
        assert abs(off - on) <= .01 * on, (off, on)
        # Boundary storage can outweigh a small box's window-tape saving.
        # This row pins flag independence, not an unconditional memory saving.
        sim, _, _, _ = _scene(graded=True)
        with pytest.raises(NotImplementedError, match="uniform single-device"):
            sim.forward(n_steps=8, checkpoint=False, checkpoint_segments=4, skip_preflight=True)
        jax.clear_caches()
    elif path == "distributed-graded":
        # Static path proof, not a claim of measured multi-device memory.
        from rfx.runners import distributed_nu
        from rfx.stepping import remat
        runner = ast.parse(inspect.getsource(distributed_nu))
        segmented = [node for node in ast.walk(runner) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == "use_segmented"]
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "segmented_scan" for branch in segmented
                   for statement in branch.body for node in ast.walk(statement))
        condition = next(node.value for node in ast.walk(runner) if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "use_segmented" for t in node.targets))
        assert "checkpoint_every" in {node.id for node in ast.walk(condition) if isinstance(node, ast.Name)}
        assert "checkpoint" not in {node.id for node in ast.walk(condition) if isinstance(node, ast.Name)}
        helper = ast.parse(inspect.getsource(remat.segmented_scan))
        assignment = next(node for node in helper.body[0].body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "body" for t in node.targets))
        assert isinstance(assignment.value, ast.Call)
        assert ast.unparse(assignment.value.func) == "jax.checkpoint"
        assert ast.unparse(assignment.value.args[0]) == "segment_body"
    else:
        from rfx.api import Simulation
        sim = Simulation(freq_max=7.5e9, domain=(.024,) * 3, dx=.0015,
                         boundary="cpml", cpml_layers=4)
        sim.add_source((.003, .012, .012), "ez")
        for knobs in ({"checkpoint": True}, {"checkpoint_every": 4}, {"checkpoint_segments": 4}):
            with pytest.raises(NotImplementedError, match="checkpoint"):
                sim.forward(n_steps=8, distributed=True, devices=jax.devices("cpu"),
                            skip_preflight=True, **knobs)
