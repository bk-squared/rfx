# T1 → core: `classify_pec_entry` reintroduces the outer-jit tracer bug

Owner of the file: whoever owns `rfx/geometry/rasterize_grid.py` (core
realization). Group T1 owns only `tests/unit/{autodiff,geometry,boundaries}`,
so the patch below is written here rather than applied.

## Fact

`forward()` wrapped in an outer `jax.jit` raises
`TracerBoolConversionError` for **any** simulation that has a PEC volume:

```
rfx/api/_compile.py:275: in _assemble_materials
    cells, sheet, wire = classify_pec_entry(...)
rfx/geometry/rasterize_grid.py:602: in classify_pec_entry
    if not traced and not bool(jnp.any(mask)):
jax.errors.TracerBoolConversionError
```

Red test: `tests/unit/autodiff/test_forward_outer_jit_traceable.py::
test_real_interior_pec_under_outer_jit_matches_eager`
(VESSL run 369367259135, checkout commit 299b0f9f).

## Cause

`_is_traced_coords(coords)` asks whether the GRID coordinates are tracers.
Under an outer `jax.jit` the coordinates are concrete host float64 arrays,
so `traced` is False — but `pec_volume_cell_mask` builds the mask with
`jnp` operations, and inside an active trace `jnp.asarray(<numpy>)` and the
boolean ops on it produce a `DynamicJaxprTracer`, not a concrete array.
`bool(jnp.any(mask))` then raises. Reproduced directly:

```python
ce = centres_from_uniform_grid(sim._build_grid())     # concrete float64
jax.jit(lambda d: (print(type(pec_volume_cell_mask(box, ce))), d))(jnp.ones(3))
# -> jax._src.interpreters.partial_eval.DynamicJaxprTracer
```

This is the same defect `rfx/api/_compile.py:240-248` documents and works
around one frame up ("this replaces a later `bool(jnp.any(pec_mask))`,
which … raises TracerBoolConversionError when the whole forward() is
wrapped in an outer `jax.jit`"). The zero-cell refusal put it back inside
`classify_pec_entry`. The same `bool(jnp.any(...))` sits in the non-Box
branch (`rasterize_grid.py:641`).

## Patch (recommended)

Decide emptiness on the HOST, from the concrete corner/centre arithmetic,
never from the jnp mask. Both sites:

```python
    mask = pec_volume_cell_mask(shape, centres)
    if not traced and not _volume_is_empty_host(shape, centres):
        ...
    if not traced and _volume_is_empty_host(shape, centres):
        _refuse_zero_cells(shape, name, "PEC volume")
```

with

```python
def _volume_is_empty_host(shape, centres) -> bool:
    """Zero-cell test on host numpy (§1.5 refusal).

    Must not go through the jnp mask: inside an outer ``jax.jit`` a jnp
    array built from concrete numpy is still a tracer, and ``bool()`` on
    it raises (the #642-class defect `_compile.py` documents).
    """
    lo = getattr(shape, "corner_lo", None)
    hi = getattr(shape, "corner_hi", None)
    cx = np.asarray(centres.x, dtype=np.float64)
    cy = np.asarray(centres.y, dtype=np.float64)
    cz = np.asarray(centres.z, dtype=np.float64)
    if lo is not None and hi is not None:
        return not (np.any((cx >= lo[0]) & (cx < hi[0]))
                    and np.any((cy >= lo[1]) & (cy < hi[1]))
                    and np.any((cz >= lo[2]) & (cz < hi[2])))
    return not bool(np.any(np.asarray(shape.mask_on_coords(cx, cy, cz))))
```

(The non-Box branch already runs on host coordinates, so `np.asarray` on
its result is safe; only the jnp round-trip has to go.)

## Falsifier

`tests/unit/autodiff/test_forward_outer_jit_traceable.py` in full —
`test_real_interior_pec_under_outer_jit_matches_eager` must pass and
`test_a_pec_shape_that_realizes_nothing_is_an_error_not_a_silent_none`
must still see the `ZERO cells` refusal on the eager path.
