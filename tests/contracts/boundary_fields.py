"""Field responses from the entry point's own two-step scan (B1)."""

from contextlib import contextmanager
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from tests.contracts.boundary_cases import build


COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")


def is_fields(value):
    return all(hasattr(value, c) for c in COMPONENTS) or (
        isinstance(value, tuple) and len(value) == 6
        and all(getattr(v, "ndim", None) == 3 for v in value)
        and len({v.shape for v in value}) == 1)


def component(state, ci):
    return getattr(state, COMPONENTS[ci]) if hasattr(state, "ex") else state[ci]


def is_aux(value):
    return hasattr(value, "_fields") and any(n.startswith("psi_") for n in value._fields)


def field_nodes(tree):
    return [v for v in jax.tree_util.tree_leaves(tree, is_leaf=is_fields) if is_fields(v)]


def seed_fields(state, variant):
    shape = component(state, 0).shape
    xyz = np.indices(shape, dtype=np.float32)
    pattern = 1 + .01 * xyz[0] + .013 * xyz[1] + .017 * xyz[2]
    values = []
    for ci, name in enumerate(COMPONENTS):
        scale = 1.0 if ci < 3 else 1 / 376.730313668
        value = pattern * scale * (ci + 1)
        if variant == 1:
            value = (2 + .02 * xyz[(ci + 1) % 3] ** 2) * scale
        if 2 <= variant < 8:
            axis, side = divmod(variant - 2, 2)
            index = [slice(None)] * 3
            index[axis] = 0 if side == 0 else -1
            if ci % 3 != axis:
                value = value.copy()
                value[tuple(index)] += 5 * scale
        if variant in (8, 9):
            value = np.zeros(shape, np.float32)
            if ci >= 3:
                for axis in range(3):
                    if ci % 3 != axis:
                        index = [slice(None)] * 3
                        index[axis] = 0 if variant == 8 else -2
                        value[tuple(index)] += scale
        if variant == 11:
            value = np.full(shape, 1 if ci < 3 else 0, np.float32)
        values.append(jnp.asarray(value, dtype=component(state, ci).dtype))
    return state._replace(**dict(zip(COMPONENTS, values))) if hasattr(state, "ex") else tuple(values)


def seed_carry(carry, variant, physical_shape=None):
    def seed(value):
        if is_fields(value):
            shape = component(value, 0).shape
            if physical_shape is not None and shape != physical_shape:
                from rfx.core.yee import FDTDState
                zeros = jnp.zeros(physical_shape, component(value, 0).dtype)
                physical = seed_fields(FDTDState(*(zeros for _ in range(6)), jnp.array(0)), variant)
                per_rank = shape[0] // 2 - 2
                padded_n = per_rank * 2
                arrays = []
                for i in range(6):
                    padded = jnp.pad(component(physical, i), ((1, padded_n - physical_shape[0] + 1), (0, 0), (0, 0)))
                    arrays.append(jnp.concatenate([padded[r * per_rank:r * per_rank + per_rank + 2] for r in range(2)]))
                return value._replace(**dict(zip(COMPONENTS, arrays)))
            return seed_fields(value, variant)
        if is_aux(value):
            return type(value)(*(jnp.full_like(v, 0 if variant in (8, 9, 10, 11) else .01) for v in value))
        return value
    return jax.tree_util.tree_map(seed, carry,
                                  is_leaf=lambda v: is_fields(v) or is_aux(v))


@contextmanager
def observe_scans(records, physical_shape=None):
    """Seed scan carries; observe numeric fields independently of wall names."""
    original = jax.lax.scan
    depth = 0

    def scan(function, initial, xs=None, *args, **kwargs):
        nonlocal depth
        if depth or not field_nodes(initial):
            return original(function, initial, xs, *args, **kwargs)
        depth += 1
        try:
            seeded = [seed_carry(initial, v, physical_shape) for v in range(12)]
            batched = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *seeded)
            result, observations = jax.vmap(lambda c: original(function, c, xs, *args, **kwargs))(batched)
            try:
                rank = jax.lax.axis_index("devices")
            except NameError:
                rank = jnp.array(-1)
            auxiliary = [v for v in jax.tree_util.tree_leaves(result, is_leaf=is_aux) if is_aux(v)]
            # Batched field tuples have ndim=4, unlike the unbatched predicate.
            leaves, structure = jax.tree_util.tree_flatten(initial, is_leaf=is_fields)
            result_leaves = structure.flatten_up_to(result)
            for before, state in zip(leaves, result_leaves):
                if not is_fields(before):
                    continue
                layout = "full-grid"
                if physical_shape is not None and component(before, 0).shape != physical_shape:
                    slab_shape = (2 * ((physical_shape[0] + 1) // 2 + 2), *physical_shape[1:])
                    assert component(before, 0).shape == slab_shape, "Unrecognized distributed scan layout"
                    layout = "two-ghosted-slabs"
                def save(rank, arrays, aux, layout=layout):
                    records.append(dict(rank=int(rank), fields=np.stack([np.asarray(a) for a in arrays], axis=1),
                                        layout=layout,
                                        psi=[{k: np.asarray(v) for k, v in a._asdict().items()} for a in aux]))
                jax.debug.callback(save, rank, tuple(component(state, i) for i in range(6)), auxiliary)
            return jax.tree_util.tree_map(lambda a: a[0], (result, observations))
        finally:
            depth -= 1

    with patch.object(jax.lax, "scan", scan):
        yield


def execute(sim, entry):
    if entry == "forward":
        return sim.forward(n_steps=2, checkpoint=False)
    if entry == "sweep":
        from rfx.vmap_sweep import vmap_material_sweep
        return vmap_material_sweep(sim, "eps_r", [1.0], n_steps=2)
    kwargs = dict(n_steps=2, compute_s_params=False)
    if entry == "wire-fast":
        kwargs.update(compute_s_params=True, s_param_freqs=np.array([10e9]), s_param_n_steps=2)
    if entry == "distributed":
        assert len(jax.devices("cpu")) >= 2
        kwargs["devices"] = jax.devices("cpu")[:2]
    return sim.run(**kwargs)


def measure(case, entry):
    sim, declared = build(case, "run" if entry == "gpu-query" else entry)
    grid = sim._build_nonuniform_grid() if entry == "nonuniform" else sim._build_grid()
    records = []
    fast_calls = []
    with observe_scans(records, grid.shape if entry == "distributed" else None):
        if entry == "gpu-query":
            import rfx.simulation as simulation
            original_fast = simulation.update_he_fast
            def observed_fast(*args, **kwargs):
                fast_calls.append(1)
                return original_fast(*args, **kwargs)
            with patch.object(jax, "default_backend", lambda: "gpu"), patch.object(simulation, "update_he_fast", observed_fast):
                result = execute(sim, "run")
        elif entry == "adi":
            import rfx.adi as adi
            original_adi = adi.run_adi_3d
            def observed_adi(*args, **kwargs):
                result = original_adi(*args, **kwargs)
                jax.block_until_ready(result[0])
                jax.effects_barrier()
                reference_args = list(args)
                reference_args[7] = jnp.zeros_like(reference_args[7])
                reference = original_adi(*reference_args, **kwargs)
                jax.block_until_ready(reference[0])
                jax.effects_barrier()
                return result
            with patch.object(adi, "run_adi_3d", observed_adi):
                result = execute(sim, entry)
        else:
            result = execute(sim, entry)
        jax.block_until_ready(result.time_series)
        jax.effects_barrier()
    for item in records:
        item["fast_trace_calls"] = len(fast_calls)
    return sim, grid, declared, records


def measured_fields(records, grid, entry):
    """Select the recorded full grid or gather its two ghosted slabs."""
    if entry == "distributed":
        assert len(records) == 1 and records[0]["rank"] == -1
        record = records[0]
        fields = record["fields"]
        if record["layout"] == "two-ghosted-slabs":
            slab_shape = (2 * ((grid.shape[0] + 1) // 2 + 2), *grid.shape[1:])
            assert fields.shape[-3:] == slab_shape, "Recorded slabs do not match the grid"
            stride = fields.shape[2] // 2
            fields = np.concatenate([fields[:, :, r * stride + 1:(r + 1) * stride - 1]
                                     for r in range(2)], axis=2)[:, :, :grid.shape[0]]
        else:
            assert record["layout"] == "full-grid", "Unrecognized distributed record layout"
        reference = None
    else:
        matching = [r for r in records if r["fields"].shape[-3:] == grid.shape]
        assert matching, "No scan with the declared grid shape"
        record = matching[0]
        fields = record["fields"]
        reference = matching[1]["fields"] if entry == "adi" else None
    assert fields.shape[-3:] == grid.shape, (
        f"Selected fields {fields.shape[-3:]} != grid {grid.shape}")
    if reference is not None:
        assert reference.shape[-3:] == grid.shape, "Reference fields do not match the grid"
    return fields, record["psi"], reference
