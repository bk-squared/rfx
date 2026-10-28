"""Admission of the graded-mesh declarations of ``Simulation``.

One place that reads what the caller declared about a graded mesh, before any
grid exists: the domain extents a cell-size profile implies, the pinned time
step and its declared smallest cell, and the concrete nominal profile of a
traced axis. ``Simulation.__init__`` calls :func:`admit_graded_profiles` and
keeps what it returns; the messages are the ones the constructor raised.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer

_AXES = ("x", "y", "z")
#: The nominal profile describes the declared domain to this relative figure.
NOMINAL_SUM_RTOL = 1e-9


def concrete_distributed_x_cells(grid):
    """Admit host-owned slab metrics before any NumPy conversion or stepping."""
    names = ("dx_arr", "dy_arr", "dz", "inv_dx", "inv_dx_h", "inv_dy",
             "inv_dy_h", "inv_dz", "inv_dz_h", "dt")
    if any(is_tracer(getattr(grid, name, None)) for name in names):
        raise TracedStructureRefusal(
            "A graded multi-device solve with traced mesh metrics cannot build "
            "its host-owned slabs. Use a concrete mesh and time step built "
            "outside jit/vmap/grad.")
    return np.asarray(grid.dx_arr, dtype=np.float64)


class NominalProfile(NamedTuple):
    """Concrete interior widths and optional half-open deformation windows.

    Only cells in ``traced_cells=(i0, i1)`` may move from ``cells``. Indices
    are Python ints, with ``0 <= i0 < i1 <= len(cells)``. A tuple of such
    windows, ``((i0, i1), (j0, j1), ...)``, declares their union: cells on
    both sides of a centred feed move while the feed's own cells stay fixed.
    """

    cells: object
    traced_cells: tuple | None = None


class DeclaredZWarning(UserWarning):
    """The caller explicitly keeps a declared z different from the solved length."""


#: A concrete z profile may miss a positive declared z by this fraction (#1941).
DECLARED_Z_RTOL = 0.01


def judge_declared_z(declared_z, dz_total, snap="strict", *, stacklevel=2):
    """Refuse (``snap="strict"``) or warn (``"declared"``) when a concrete z
    profile's sum, the length the grid solves, misses the declared z by more
    than :data:`DECLARED_Z_RTOL`. Every reader of the declared extent
    (periodic z, flux regions, thin conductors, RCS) would read a length the
    grid does not solve. ``stacklevel`` names the caller's line."""
    rel = abs(dz_total - declared_z) / declared_z
    if rel <= DECLARED_Z_RTOL:
        return
    message = (
        f"Declared domain z = {declared_z:.12g} m differs from "
        f"dz_profile sum (solved length) = {dz_total:.12g} m "
        f"(rel = {rel:.12g}, greater than {DECLARED_Z_RTOL}). "
        "Declare z equal to the profile sum, give 0 for z to "
        'take it from the profile, or use snap="declared".')
    if snap == "declared":
        warnings.warn(message, DeclaredZWarning, stacklevel=stacklevel)
    else:
        raise ValueError(message)


def admit_graded_profiles(domain, dx_profile, dy_profile, dz_profile, *,
                          dt=None, dt_min_cell=None, nominal_profiles=None, snap="strict",
                          multiband_ratio_cap=None, inplane_ratio_cap=None):
    """Return ``(domain, nominal_cells)`` for the declared graded mesh.

    Concrete x/y profiles set their extents. A concrete z profile checks a
    positive declared z, or supplies it when absent; a traced profile needs
    the caller's concrete extent. The dt pin
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
            else:
                judge_declared_z(domain[2], dz_total, snap, stacklevel=4)
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
    if multiband_ratio_cap is not None and inplane_ratio_cap is not None:
        warn_abrupt_grading(dx_profile, dy_profile, dz_profile,
                            multiband_ratio_cap, inplane_ratio_cap)
    return domain, nominal


def warn_abrupt_grading(dx_profile, dy_profile, dz_profile,
                        multiband_ratio_cap, inplane_ratio_cap):
    """Warn when adjacent cells of a concrete profile differ by more than its axis cap.

    ``multiband_ratio_cap`` applies to ``dz_profile``, ``inplane_ratio_cap`` to
    ``dx_profile`` and ``dy_profile``. The warning points at the line that built
    the ``Simulation`` (it is raised through ``admit_graded_profiles``).
    """
    # P2: Warn on abrupt grading in a user-supplied profile. This ran
    # for dz only, so an in-plane profile with a 2:1 or 4:1 jump was
    # accepted in silence (#743) — the axes differ in which physics
    # they carry, not in whether an abrupt ratio reflects.
    # Tracer profiles skip the warning — adjacent ratios can't be
    # computed host-side during tracing.
    # Threshold, PER AXIS. dz: 1.3 -> 1.4 (SPEC-01 WP6, #780).
    # PROVENANCE for that move: 1.3 was `smooth_grading`'s own per-step
    # default with no measurement behind it, so ratios in (1.3, 1.4]
    # warned without evidence that they cost anything. The multi-band
    # witness battery (validation/research/multiband_nu/,
    # pre-declaration note
    # docs/design_notes/20260829_spec01_multiband_predeclaration.md)
    # measures the r = 1.4 transition directly: reflection inside the
    # exact discrete chain-model window (-54 dB class at 30
    # cells/wavelength; the class scales as (dz/lambda)^2), round-trip
    # amplitude asymmetry under the 3e-4 floor, 1e6-step energy bounded
    # at the float-accumulation class, and 2nd-order supraconvergence
    # preserved on a fixture whose graded axis carries ~90 % of the
    # error budget.
    # dx/dy STAY AT 1.3: every witness in that battery grades z and
    # holds the transverse mesh uniform (the harness takes a scalar
    # transverse cell), so there is no in-plane provenance for moving
    # an in-plane lock — SPEC-00 0.2-4. The 2026-08-29 adversarial
    # review found this axis overclaim in the docs; it lived here too.
    # Beyond the cap this warning still fires, and preflight adds the
    # richer advisory nu_grading_ratio_beyond_validated_cap with the
    # accuracy class on both sides.
    for _axis_name, _profile, _cap, _scope in (
            ("dz_profile", dz_profile, multiband_ratio_cap,
             "the validated multi-band grading cap"),
            ("dx_profile", dx_profile, inplane_ratio_cap,
             "the in-plane grading threshold — the validated "
             f"{multiband_ratio_cap:g} multi-band cap is a "
             "z-axis envelope and no witness grades an in-plane axis"),
            ("dy_profile", dy_profile, inplane_ratio_cap,
             "the in-plane grading threshold — the validated "
             f"{multiband_ratio_cap:g} multi-band cap is a "
             "z-axis envelope and no witness grades an in-plane axis")):
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
                    f"rfx.smooth_grading({_axis_name}) to fix.",
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
        windows = (window if isinstance(window, tuple) and window
                   and all(isinstance(w, tuple) for w in window) else (window,))
        n = len(value.cells)
        for w in windows:
            if (not isinstance(w, tuple) or len(w) != 2
                    or any(type(i) is not int for i in w)):
                raise ValueError(
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
    duplicate = int(_axis_name(ax) not in getattr(grid, "periodic_axes", ""))
    interior = window_mask(window, n - pad_lo - pad_hi - duplicate)
    if not padded:
        return interior
    return np.pad(interior, (pad_lo, pad_hi + duplicate), mode="edge")
