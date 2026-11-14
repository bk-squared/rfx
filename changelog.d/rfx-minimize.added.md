### Added — `rfx.minimize` optimises a few scalar parameters with Adam, projected Adam or MMA (#PR)

- `rfx.minimize(fun, x0, bounds=..., method=...)` takes a JAX-traceable objective of a 1-D parameter vector. `bounds` are SciPy-style `(min, max)` pairs, one per parameter.
- `method="projected_adam"` steps on the box variable and clips to the box. `method="mma"` is the method of moving asymptotes.
- With `worst_case=True`, MMA minimises the largest entry of a nonnegative vector objective, such as the reflectance at several frequencies.
- `MinimizeResult` reports the evaluated solution, its value, the parameter and objective histories, and the solve counts.
- `MMASubproblemCapWarning` names each MMA step taken from a subproblem that reached a solver cap; the result counts them.
- `examples/inverse_design/multilayer_ar_coating.py` runs through `rfx.minimize`. `--compare-methods --mesh fine` compares the methods.
- `optimize()` is unchanged.
