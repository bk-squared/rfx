"""Admission of the graded-mesh declarations of ``Simulation``.

One place that reads what the caller declared about a graded mesh, before any
grid exists: the domain extents a cell-size profile implies, the pinned time
step and its declared smallest cell, and the concrete nominal profile of a
traced axis. ``Simulation.__init__`` calls :func:`admit_graded_profiles` and
keeps what it returns; the messages are the ones the constructor raised.
``NOMINAL_SUM_RTOL`` is the relative tolerance within which a nominal profile's cell widths must sum to the declared domain extent.
"""
from __future__ import annotations

from rfx.errors import AccuracyWarning, RfxRefusal, RfxWarning
from rfx.model.realization_verdict import legacy_mismatch
import warnings
from collections.abc import Mapping
from typing import NamedTuple

import jax
import numpy as np

from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer

_AXES = ("x", "y", "z")
NOMINAL_SUM_RTOL = 1e-9


TRACED_DISTRIBUTED_MESH_REFUSAL = (
    "A graded multi-device solve with traced mesh metrics cannot build "
    "its host-owned slabs. Use a concrete mesh and time step built "
    "outside jit/vmap/grad.")
DISTRIBUTED_GRADING_LIMIT = 5.0
DISTRIBUTED_GRADING_CAUTION = 3.0


def concrete_distributed_x_cells(grid):
    """Admit host-owned slab metrics before any NumPy conversion or stepping."""
    names = ("dx_arr", "dy_arr", "dz", "inv_dx", "inv_dx_h", "inv_dy",
             "inv_dy_h", "inv_dz", "inv_dz_h", "dt")
    if any(is_tracer(getattr(grid, name, None)) for name in names):
        raise TracedStructureRefusal(TRACED_DISTRIBUTED_MESH_REFUSAL)
    return np.asarray(grid.dx_arr, dtype=np.float64)


def admit_distributed_grading(profiles):
    """Admit the declared cell-size profiles of a graded multi-device solve.

    ``profiles`` is ``(dx_profile, dy_profile, dz_profile)``. A profile that
    holds a JAX tracer is refused with :class:`TracedStructureRefusal` before
    any host read. Otherwise the grading ratio is the largest max/min over the
    given non-empty profiles with a positive minimum (1.0 when none): above
    :data:`DISTRIBUTED_GRADING_LIMIT` it raises ``ValueError``, above
    :data:`DISTRIBUTED_GRADING_CAUTION` it warns with ``AccuracyWarning``.
    ``forward(distributed=True)`` and ``run(devices=...)`` both call it, with
    or without preflight. Returns the ratio.
    """
    if any(is_tracer(leaf) for profile in profiles
           for leaf in jax.tree_util.tree_leaves(profile)):
        raise TracedStructureRefusal(TRACED_DISTRIBUTED_MESH_REFUSAL)
    ratio = 1.0
    for profile in profiles:
        if profile is not None and len(profile) > 0:
            cells = np.asarray(profile, dtype=np.float64)
            if float(cells.min()) > 0.0:
                ratio = max(ratio, float(cells.max()) / float(cells.min()))
    if ratio > DISTRIBUTED_GRADING_LIMIT:
        raise ValueError(
            f"grading_ratio={ratio:.2f} exceeds {DISTRIBUTED_GRADING_LIMIT}; "
            "a graded multi-device solve (forward(distributed=True) or "
            "run(devices=...)) requires a max/min cell ratio <= "
            f"{DISTRIBUTED_GRADING_LIMIT} for its shared time step and x-face "
            "CPML calibration. Reduce the cell-size variation or run on one "
            "device.")
    if ratio > DISTRIBUTED_GRADING_CAUTION:
        from rfx._warning_origin import user_stacklevel
        warnings.warn(
            f"Distributed + non-uniform grading ratio {ratio:.2f}"
            ":1 exceeds the 3:1 stability caution threshold. "
            "Monitor for numerical dispersion / late-time drift.", AccuracyWarning,
            stacklevel=user_stacklevel(),
        )
    return ratio


class NominalProfile(NamedTuple):
    """Concrete interior widths and optional half-open deformation windows.

    Only cells in ``traced_cells=(i0, i1)`` may move from ``cells``. Indices
    are Python ints, with ``0 <= i0 < i1 <= len(cells)``. A tuple of such
    windows, ``((i0, i1), (j0, j1), ...)``, declares their union: cells on
    both sides of a centred feed move while the feed's own cells stay fixed.
    """

    cells: object
    traced_cells: tuple | None = None


class DeclaredZWarning(RfxWarning):
    """The caller explicitly keeps a declared z different from the solved length."""


DECLARED_Z_RTOL = 0.01


def judge_declared_z(declared_z, dz_total, snap="strict", *, stacklevel=2):
    """Refuse (``snap="strict"``) or warn (``"declared"``) when a concrete z
    profile's sum, the length the grid solves, misses the declared z by more
    than :data:`DECLARED_Z_RTOL`. Every reader of the declared extent
    (periodic z, flux regions, thin conductors, RCS) would read a length the
    grid does not solve. ``stacklevel`` names the caller's line."""
    rel = abs(dz_total - declared_z) / declared_z
    if not legacy_mismatch(rel, DECLARED_Z_RTOL):
        return
    message = (
        f"Declared domain z = {declared_z:.12g} m differs from "
        f"dz_profile sum (solved length) = {dz_total:.12g} m "
        f"(rel = {rel:.12g}, greater than {DECLARED_Z_RTOL}). "
        "Declare z equal to the profile sum, give 0 for z to "
        'take it from the profile, or use snap="declared".')
    if snap == "declared":
        warnings.warn(message, DeclaredZWarning, stacklevel=stacklevel)
        from rfx.model.realization_verdict import legacy_outcome
        return legacy_outcome("legacy:declared_z", rel, DECLARED_Z_RTOL,
                              message, "declared_z", "judge_declared_z")
    else:
        raise ValueError(message)


def admit_graded_profiles(domain, dx_profile, dy_profile, dz_profile, *,
                          dt=None, dt_min_cell=None, nominal_profiles=None, snap="strict",
                          multiband_ratio_cap=None, inplane_ratio_cap=None):
    """Return ``(domain, nominal_cells)`` for the declared graded mesh.

    Concrete x/y profiles set their extents. A concrete z profile checks a
    positive declared z, or supplies it when absent; a traced profile needs
    the caller's concrete extent. The dt pin
    is checked by the grid builder; ``dt_min_cell`` needs a pin.
    Nominal widths are float64 arrays; windowed axes retain their
    :class:`NominalProfile` declaration so admission is idempotent.
    """
    declared_domain = getattr(domain, "declared", domain)
    early_outcome = None
    if dx_profile is not None:
        if is_tracer(dx_profile):
            if len(domain) < 1 or domain[0] <= 0:
                raise RfxRefusal(
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
                raise RfxRefusal(
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
                raise RfxRefusal(
                    "dz_profile is a JAX tracer; provide a concrete "
                    "domain=(..., ..., dz_total) extent (the profile "
                    "sum cannot be host-coerced during tracing)."
                )
        else:
            dz_total = float(np.sum(dz_profile))
            if len(domain) < 3 or domain[2] <= 0:
                domain = (domain[0], domain[1], dz_total)
            else:
                early_outcome = judge_declared_z(domain[2], dz_total, snap, stacklevel=4)
    elif any(d <= 0 for d in domain):
        raise ValueError(f"domain dimensions must be positive, got {domain}")

    if dt is None and dt_min_cell is not None:
        raise ValueError("dt_min_cell= has no effect without dt=")

    nominal = admit_nominal_profiles(
        nominal_profiles, (dx_profile, dy_profile, dz_profile), domain)
    for axis, window in (admit_traced_windows(nominal_profiles) or {}).items():
        nominal[axis] = NominalProfile(nominal[axis], window)
    if multiband_ratio_cap is not None and inplane_ratio_cap is not None:
        warn_abrupt_grading(dx_profile, dy_profile, dz_profile,
                            multiband_ratio_cap, inplane_ratio_cap)
    from rfx.model.realization_reference import retain_declared_domain
    result = retain_declared_domain(domain, list(declared_domain) if early_outcome else declared_domain)
    if early_outcome is not None:
        from rfx.model.realization_verdict import remember_legacy
        remember_legacy(result, "z", early_outcome)
    return result, nominal


def warn_abrupt_grading(dx_profile, dy_profile, dz_profile,
                        multiband_ratio_cap, inplane_ratio_cap):
    """Warn when adjacent cells of a concrete profile differ by more than its axis cap.

    ``multiband_ratio_cap`` applies to ``dz_profile``, ``inplane_ratio_cap`` to
    ``dx_profile`` and ``dy_profile``. The warning points at the line that built
    the ``Simulation`` (it is raised through ``admit_graded_profiles``).
    """
    for _axis_name, _profile, _cap, _scope in (
            ("dz_profile", dz_profile, multiband_ratio_cap,
             "the validated multi-band grading cap"),
            ("dx_profile", dx_profile, inplane_ratio_cap,
             "the validated in-plane grading cap"),
            ("dy_profile", dy_profile, inplane_ratio_cap,
             "the validated in-plane grading cap")):
        if (_profile is not None and not is_tracer(_profile)
                and len(_profile) > 1):
            import warnings as _w
            _p = np.asarray(_profile, dtype=float)
            ratios = _p[1:] / _p[:-1]
            max_ratio = float(np.max(np.maximum(ratios, 1.0 / ratios)))
            if max_ratio > _cap + 1e-6:
                _w.warn(
                    f"{_axis_name} has max adjacent cell ratio "
                    f"{max_ratio:.3f} (> {_cap:g}, {_scope} — "
                    "docs/guides/support_matrix.md, "
                    "'Multi-band graded mesh'). This may cause "
                    "numerical reflections. Use "
                    f"rfx.smooth_grading({_axis_name}) to fix.", AccuracyWarning,
                    stacklevel=4,
                )


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
            raise RfxRefusal(
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
        windows = (window if isinstance(window, tuple) and window
                   and all(isinstance(w, tuple) for w in window) else (window,))
        n = len(value.cells)
        for w in windows:
            if (not isinstance(w, tuple) or len(w) != 2
                    or any(type(i) is not int for i in w)):
                raise RfxRefusal(
                    f"nominal_profiles[{axis!r}].traced_cells must be a 2-tuple "
                    "of Python ints (not bool), or a non-empty tuple of them.")
            if not 0 <= w[0] < w[1] <= n:
                raise ValueError(
                    f"nominal_profiles[{axis!r}].traced_cells={window!r} must "
                    f"satisfy 0 <= i0 < i1 <= {n} in every window.")
        out[axis] = window
    return out or None


def window_mask(window, n):
    """Interior flags of a declared window or tuple of windows (their union)."""
    windows = window if isinstance(window[0], tuple) else (window,)
    interior = np.zeros(n, dtype=bool)
    for i0, i1 in windows:
        interior[i0:i1] = True
    return interior


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
    return np.asarray(_append_bounding_node(_pad_profile(widths, pad_lo, pad_hi, dtype=np.float64)),
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
    duplicate = int(_axis_name(ax) not in getattr(grid, "periodic_axes", ""))
    interior = window_mask(window, n - pad_lo - pad_hi - duplicate)
    if not padded:
        return interior
    return np.pad(interior, (pad_lo, pad_hi + duplicate), mode="edge")


def host_cells(grid, axis, *, store=False):
    """Host widths in the precision previously read by a metric consumer.

    By default this is the concrete float64 spine (``grid.cells``);
    ``store=True`` preserves the solver's stored widths (``dx_arr``,
    ``dy_arr``, ``dz``). A traced axis with a declared deformation window
    returns its padded nominal widths in that same precision; consumers
    refuse reads within the window, and waveguide admission checks its
    total so downstream physical coordinates remain fixed. A traced axis
    without a window returns the traced widths unchanged, as before.
    """
    index = "xyz".index(axis) if isinstance(axis, str) else int(axis)
    stored = (getattr(grid, "dx_arr", None), getattr(grid, "dy_arr", None),
              getattr(grid, "dz", None))[index]
    if hasattr(grid, "is_traced") and grid.is_traced(axis):
        widths = nominal_cells(grid, axis, padded=True)
        if widths is None or traced_cell_mask(grid, axis) is None:
            return stored if store else grid.cells(axis)
        return np.asarray(widths, dtype=getattr(stored, "dtype", np.float64) if store else np.float64)
    if store and stored is not None and not np.isscalar(stored):
        return np.asarray(stored)
    return np.asarray(grid.cells(axis), dtype=np.float64)


def refuse_window_cells(grid, axis, selection, feature):
    """Refuse a host metric that needs a cell in a declared deformation window."""
    if not hasattr(grid, "is_traced") or not grid.is_traced(axis):
        return
    mask = traced_cell_mask(grid, axis)
    if mask is None or np.any(mask[selection]):
        raise TracedStructureRefusal(
            f"{feature}: nominal_profiles= on axis {axis!r} must keep every "
            "cell read by this host metric outside traced_cells.")
