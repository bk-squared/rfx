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


def drop_periodic_axes(cpml_axes: str, periodic) -> str:
    """Absorber axes left after removing the periodic ones (no CPML/PEC there)."""
    return "".join(a for a, p in zip("xyz", periodic) if a in cpml_axes and not p)


def padded_axes(grid, cpml_axes: str) -> str:
    """The requested axes that carry an absorber pad on at least one face.

    Axes whose lo+hi pad is zero are fully closed; the apply path's
    ``state.e*[:, :, :n]`` slices would clip to the short axis and break the
    broadcast against the ``(cpml_layers,)`` profile, so they are dropped and
    the no-op branch passes psi through unchanged.
    """
    return "".join(a for a in "xyz" if a in cpml_axes
                   and getattr(grid, f"pad_{a}_lo") + getattr(grid, f"pad_{a}_hi") > 0)


def hi_face_is_wall(grid, axis: str) -> bool:
    """Whether a grid's high face on ``axis`` is an electric wall.

    True when the axis does not absorb at all or the face is declared PEC.
    """
    pec_faces = getattr(grid, "pec_faces", set()) or set()
    return axis not in getattr(grid, "cpml_axes", "") or f"{axis}_hi" in pec_faces
