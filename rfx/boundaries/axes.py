"""Resolve runner absorber axes against the grid's allocation."""


def resolve_cpml_axes(grid, cpml_axes: str | None = None) -> str:
    """Default to the grid; explicit subsets preserve feature boundary overrides.

    Uniform grids retain their declaration. NonUniformGrid represents it with
    per-face pads, so only axes with an allocated absorbing face are active.
    Legacy grid-like objects without per-face pads expose a whole-axis budget.
    Apply the grid's own periodic/2-D filtering to explicit values too.
    """
    grid_axes = getattr(grid, "cpml_axes", None)
    if grid_axes is None:
        def pad(axis, side):
            value = getattr(grid, f"pad_{axis}_{side}", None)
            return grid.cpml_layers if value is None else value

        grid_axes = "".join(
            axis for axis in "xyz" if pad(axis, "lo") + pad(axis, "hi") > 0)
    if cpml_axes is None:
        return grid_axes
    ignored = set(getattr(grid, "periodic_axes", ""))
    if getattr(grid, "is_2d", False):
        ignored.add("z")
    requested = set(cpml_axes) - ignored
    if not requested <= set(grid_axes):
        raise ValueError(
            f"Runner cpml_axes={cpml_axes!r} names axes outside the grid's "
            f"cpml_axes={grid_axes!r}. The grid's declaration is authoritative: "
            "omit cpml_axes, or build the grid with the axes you want. "
            "Explicit subsets of the grid's axes are accepted."
        )
    return "".join(axis for axis in "xyz" if axis in requested)
