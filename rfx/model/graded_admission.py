"""Admission of the graded-mesh declarations of ``Simulation``.

One place that reads what the caller declared about a graded mesh, before any
grid exists: the domain extents a cell-size profile implies, the pinned time
step and its declared smallest cell, and the concrete nominal profile of a
traced axis. ``Simulation.__init__`` calls :func:`admit_graded_profiles` and
keeps what it returns; the messages are the ones the constructor raised.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from rfx.core.jax_utils import is_tracer

_AXES = ("x", "y", "z")
#: The nominal profile describes the declared domain to this relative figure.
NOMINAL_SUM_RTOL = 1e-9


class NominalProfile(NamedTuple):
    """Concrete interior widths and an optional half-open deformation window.

    Only cells in ``traced_cells=(i0, i1)`` may move from ``cells``. Indices
    are Python ints, with ``0 <= i0 < i1 <= len(cells)``.
    """

    cells: object
    traced_cells: tuple[int, int] | None = None


def admit_graded_profiles(domain, dx_profile, dy_profile, dz_profile, *,
                          dt=None, dt_min_cell=None, nominal_profiles=None):
    """Return ``(domain, nominal_cells)`` for the declared graded mesh.

    ``domain`` is the declared extent with every CONCRETE profile's sum put in
    its axis; a traced profile needs the caller's concrete extent. The dt pin
    is refused without a profile and ``dt_min_cell`` without a pin.
    Nominal widths are float64 arrays; windowed axes retain their
    :class:`NominalProfile` declaration so admission is idempotent.
    """
    # Synthesize domain extents from axis profiles before validation.
    # dx_profile / dy_profile set x / y; dz_profile sets z.
    # Tracer-valued profiles require the caller to supply a concrete
    # domain extent for that axis since the profile sum cannot be
    # host-coerced during tracing.
    if dx_profile is not None:
        if is_tracer(dx_profile):
            if len(domain) < 1 or domain[0] <= 0:
                raise ValueError(
                    "dx_profile is a JAX tracer; provide a concrete "
                    "domain[0] (profile sum cannot be host-coerced)."
                )
        else:
            domain = (float(np.sum(dx_profile)),
                      domain[1] if len(domain) >= 2 else 0.0,
                      domain[2] if len(domain) >= 3 else 0.0)
    if dy_profile is not None:
        if is_tracer(dy_profile):
            if len(domain) < 2 or domain[1] <= 0:
                raise ValueError(
                    "dy_profile is a JAX tracer; provide a concrete "
                    "domain[1] (profile sum cannot be host-coerced)."
                )
        else:
            domain = (domain[0],
                      float(np.sum(dy_profile)),
                      domain[2] if len(domain) >= 3 else 0.0)
    if dz_profile is not None:
        if any(d <= 0 for d in domain[:2]):
            raise ValueError(f"domain x/y must be positive, got {domain}")
        if is_tracer(dz_profile):
            if len(domain) < 3 or domain[2] <= 0:
                raise ValueError(
                    "dz_profile is a JAX tracer; provide a concrete "
                    "domain=(..., ..., dz_total) extent (the profile "
                    "sum cannot be host-coerced during tracing)."
                )
        else:
            dz_total = float(np.sum(dz_profile))
            if len(domain) < 3 or domain[2] <= 0:
                domain = (domain[0], domain[1], dz_total)
    elif any(d <= 0 for d in domain):
        raise ValueError(f"domain dimensions must be positive, got {domain}")

    # Pinned time step (non-uniform lane only); see the ``dt`` parameter.
    if dt is not None and dz_profile is None and dx_profile is None \
            and dy_profile is None:
        raise ValueError(
            "dt= pins the non-uniform lane's time step and needs at least "
            "one of dz_profile / dx_profile / dy_profile; the uniform "
            "lane derives its step in Grid.courant_dt."
        )
    if dt is None and dt_min_cell is not None:
        raise ValueError("dt_min_cell= has no effect without dt=")

    nominal = admit_nominal_profiles(
        nominal_profiles, (dx_profile, dy_profile, dz_profile), domain)
    for axis, window in (admit_traced_windows(nominal_profiles) or {}).items():
        nominal[axis] = NominalProfile(nominal[axis], window)
    return domain, nominal


def admit_nominal_profiles(nominal_profiles, profiles, domain=None):
    """Validate ``nominal_profiles=`` and return it as float64 arrays.

    ``profiles`` is ``(dx_profile, dy_profile, dz_profile)``; ``domain`` the
    declared extents, an entry of ``None`` (or no ``domain``) skipping that
    axis's sum check. Returns ``{axis: float64 interior widths}``, or ``None``
    when nothing is declared.

    A traced axis has no host node line, so a coordinate on it is resolved on
    the concrete profile the caller declares here: the profile at the primal
    point of the derivative. Nothing is read off the tracer. Each refusal is a
    ``ValueError`` naming the axis.
    """
    if nominal_profiles is None:
        return None
    if not isinstance(nominal_profiles, Mapping):
        raise ValueError(
            "nominal_profiles= must be a dict mapping 'x' / 'y' / 'z' to the "
            f"concrete cell widths of a traced profile, got "
            f"{type(nominal_profiles).__name__}.")
    out = {}
    for key, value in nominal_profiles.items():
        if key not in _AXES:
            raise ValueError(
                f"nominal_profiles key {key!r} is not an axis; use 'x', 'y' "
                "or 'z'.")
        profile = profiles[_AXES.index(key)]
        if profile is None or not is_tracer(profile):
            raise ValueError(
                f"nominal_profiles[{key!r}] is declared but the {key} "
                "profile is not traced: its node line is already concrete, "
                "so the declaration would have no effect.")
        if isinstance(value, NominalProfile):
            value = value.cells
        if is_tracer(value):
            raise ValueError(
                f"nominal_profiles[{key!r}] is a JAX tracer; declare the "
                f"concrete {key} cell widths at the primal point.")
        widths = np.array(value, dtype=np.float64)
        n = int(np.shape(profile)[0])
        if widths.ndim != 1 or widths.shape[0] != n:
            raise ValueError(
                f"nominal_profiles[{key!r}] has shape {widths.shape}; the "
                f"traced d{key}_profile has {n} cells.")
        if not (np.all(np.isfinite(widths)) and np.all(widths > 0)):
            raise ValueError(
                f"nominal_profiles[{key!r}] must hold finite positive cell "
                "widths.")
        extent = None if domain is None else domain[_AXES.index(key)]
        if extent is not None:
            total = float(np.sum(widths))
            if abs(total - float(extent)) > NOMINAL_SUM_RTOL * abs(float(extent)):
                raise ValueError(
                    f"nominal_profiles[{key!r}] sums to {total!r} m but the "
                    f"declared domain {key} extent is {float(extent)!r} m; "
                    "the nominal profile must describe that same domain.")
        out[key] = widths
    return out or None


def admit_traced_windows(nominal_profiles):
    """Return declared windows, or None, after nominal widths are admitted.

    Widths are not re-checked here; their admitted length is the traced
    profile length. Each malformed window raises ValueError naming its axis.
    """
    out = {}
    for axis, value in (nominal_profiles or {}).items():
        if not isinstance(value, NominalProfile) or value.traced_cells is None:
            continue
        window = value.traced_cells
        if (not isinstance(window, tuple) or len(window) != 2
                or any(type(i) is not int for i in window)):
            raise ValueError(
                f"nominal_profiles[{axis!r}].traced_cells must be a 2-tuple "
                "of Python ints (not bool).")
        i0, i1 = window
        n = len(value.cells)
        if not 0 <= i0 < i1 <= n:
            raise ValueError(
                f"nominal_profiles[{axis!r}].traced_cells={window!r} must "
                f"satisfy 0 <= i0 < i1 <= {n}.")
        out[axis] = window
    return out or None


def nominal_cells(grid, axis, padded: bool = False):
    """The caller's declared nominal widths of a traced ``axis``, or None.

    The one reader of ``NonUniformGrid.nominal_cells``: None when the axis is
    not traced or no ``nominal_profiles=`` entry was declared for it (read the
    grid's own widths there). ``padded=False`` gives the interior widths as
    declared; ``padded=True`` gives float64 widths with the SAME indexing and
    length as the grid's own width array on that axis (``grid.dx_arr`` /
    ``dy_arr`` / ``dz``: CPML pad cells of the boundary width and the trailing
    bounding cell), built by the grid builder's own padding.
    """
    from rfx.model.graded_grid import (
        _append_bounding_node, _axis_name, _pad_profile)
    ax, _n, pad_lo, pad_hi = grid._axis_layout(axis)
    if not is_tracer(grid._axis_store(ax)):
        return None
    widths = (getattr(grid, "nominal_cells", None) or {}).get(_axis_name(ax))
    if widths is None:
        return None
    widths = np.asarray(widths, dtype=np.float64)
    if not padded:
        return widths
    return np.asarray(_append_bounding_node(_pad_profile(widths, pad_lo, pad_hi)),
                      dtype=np.float64)


def traced_cell_mask(grid, axis, padded: bool = True):
    """Boolean deformation flags for a traced axis, or None without a window.

    Interior indices are half-open. Padding copies the corresponding edge
    flag; the trailing bounding cell copies the last interior flag too.
    """
    from rfx.model.graded_grid import _axis_name

    ax, n, pad_lo, pad_hi = grid._axis_layout(axis)
    if not is_tracer(grid._axis_store(ax)):
        return None
    window = (getattr(grid, "traced_cells", None) or {}).get(_axis_name(ax))
    if window is None:
        return None
    interior = np.zeros(n - pad_lo - pad_hi - 1, dtype=bool)
    interior[window[0]:window[1]] = True
    if not padded:
        return interior
    return np.pad(interior, (pad_lo, pad_hi + 1), mode="edge")
