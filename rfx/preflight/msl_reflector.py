"""The microstrip downstream-reflector scan, out of ``rfx.preflight.msl``.

Moved verbatim (2026-10-06) to bring ``rfx/preflight/msl.py`` back under its
line count in ``scripts/ci/file_size_baseline.json``. ``rfx.preflight.msl``
re-exports the name, so every existing import and the facade
``rfx.api._preflight`` keep resolving it there; ``msl_probe_clearance_for_port``
reads it as a global of that module.
"""

from __future__ import annotations

import numpy as np


def msl_nearest_downstream_reflector(
    geometry,
    *,
    x_probe: float,
    x_feed: float,
    y_feed: float,
    w_trace: float,
    dx: float,
    domain_y: float,
    direction: str,
    resolve_material=None,
    thin_conductors=(),
    pec_sigma_threshold: float = 1e6,
    signed_front_distance: bool = False,
    width_cell: float | None = None,
    ground_plane: float | None = None,
    ground_cell: float | None = None,
    port_position=None,
    grid=None,
):
    """Distance from ``x_probe`` to the nearest downstream conductor edge.

    Walks every registered CONDUCTOR and returns
    ``(distance_m, label, unevaluated)`` for the nearest one at or beyond
    ``x_probe`` along the propagation direction — ``(inf, None,
    unevaluated)`` when nothing qualifies. ``unevaluated`` is the list of
    conductors this scan could NOT place (one string each); it is what
    lets the caller distinguish "nothing is nearby" from "I could not
    look" (issue #685).

    By default a probe inside a candidate has zero distance, preserving
    the auto-placement contract. ``signed_front_distance=True`` instead
    returns its signed distance to the candidate's entering boundary;
    this is needed when reporting how far an observation has passed it.

    What counts as a conductor (issue #685). This used to be
    ``isinstance(shape, Box) and str(material_name).lower() == "pec"``,
    which was blind to two whole classes:

    * **PEC-PROMOTED materials** — a conductor registered with
      ``sigma >= 1e6`` under any other name. That is the common case for
      imported CAD, where every conductor may be called ``"metal"``. Pass
      ``resolve_material`` (normally ``sim._resolve_material``) and the
      test becomes the same ``sigma >= pec_sigma_threshold`` rule the
      assembler itself uses to build ``pec_mask``. Without it the legacy
      name test is kept so old callers do not change behaviour.
    * **non-``Box`` shapes** — ``Sheet``, ``MeshShape`` (#358), CSG
      results. Any shape with a ``bounding_box()`` is now placed by that
      box; a bounding box OVERSTATES a non-convex outline, which is the
      conservative direction for a clearance advisory (it can only bring
      the reported reflector nearer, never push it away). A shape with no
      usable bounding box goes to ``unevaluated`` instead of being
      skipped in silence.

    ``thin_conductors`` (normally ``sim._thin_conductors``) are scanned
    too: a thin PEC sheet, a DC-fold lossy sheet and a
    ``surface_impedance_f0`` sheet are all metal to a wave on the line,
    and since #677 the f0 one is in neither ``pec_mask`` nor
    ``materials.sigma``, so nothing else would ever see it.

    Observed consequence of the old scan: on a board whose reflectors were
    all thin conductors, it found none. Probe 0 then sat well inside this
    repo's own downstream rule (``msl_min_probe_clearance``) and its
    upstream rule (``5*h_sub``), and two further probes landed physically
    on metal -- with nothing warned. The old scan read only volumetric
    ``geometry``, so a board built from sheets was invisible to it.

    ``dx`` is the local rasterization resolution at the feed along
    propagation; ``width_cell`` is the smaller resolution at the two trace
    edges on its width axis. Legacy callers that supply only ``dx`` retain
    their cubic tolerance.

    Axis generality (issue #661): the parameter names are the ``"+x"``-frame
    names. ``x_probe`` / ``x_feed`` are coordinates on the PROPAGATION axis,
    ``y_feed`` is the trace centreline on the WIDTH axis and ``domain_y``
    that axis's domain extent — the two axes are resolved from ``direction``
    via :func:`~rfx.sources.msl_port.msl_axis_roles`, so a ``"+y"`` port
    compares box y-extents along the feed axis and box x-extents across the
    trace width. Callers pass coordinates already projected onto those roles.

    Exclusions (both are #469-arc corrections to the pre-existing
    heuristic):

    * **the line being measured** — any trace-width box
      (|y-extent − w_trace| ≤ dx, y-range containing the feed centreline)
      whose x-range contains the FEED plane. This covers the through-line
      AND a port's own feed trace; the old rule (x-extent ≥ 80 % of an
      inter-port-extent estimate) missed the latter for '-x' ports —
      measured d=0 false positive against the port's OWN output feed
      (validation/crossval/07_sheen_lpf.py known-residual note) — and its
      '+x' extent estimate evaluated to the CPML thickness instead of the
      far-wall coordinate (latent arithmetic bug, now moot: the estimate
      is gone). A same-width series element that does NOT contain the
      feed plane is a genuine discontinuity and is still counted.
    * **ground conductors** entirely at or below ``ground_plane`` on the
      substrate-normal axis, independent of their width. If that reference
      is unavailable, retain the legacy width ≥ 80 % of domain heuristic.
    """
    from rfx.geometry.csg import Box as _Box
    from rfx.sources.msl_port import _MSL_AXIS_INDEX, msl_axis_roles

    if width_cell is None:
        width_cell = dx
    _prop_ax, _width_ax, _n_ax, sign = msl_axis_roles(direction)
    # Production callers hand over the port's position (and the grid); the
    # ground reference is its coordinate on the port's normal axis.
    if port_position is not None and ground_plane is None:
        ground_plane = float(port_position[_MSL_AXIS_INDEX[_n_ax]])
        if grid is not None and ground_cell is None:
            from rfx.preflight._common import local_cell
            ground_cell = local_cell(grid, _n_ax, ground_plane)
    _ip = _MSL_AXIS_INDEX[_prop_ax]
    _iw = _MSL_AXIS_INDEX[_width_ax]
    _in = _MSL_AXIS_INDEX[_n_ax]
    nearest_d = float("inf")
    nearest_label = None
    unevaluated: list[str] = []

    def _is_conductor(material_name) -> bool:
        if resolve_material is None:
            # Legacy name test, kept for callers that cannot resolve.
            return str(material_name).lower() == "pec"
        try:
            mat = resolve_material(material_name)
        except Exception:
            unevaluated.append(
                f"geometry entry with material {material_name!r}: the "
                f"material could not be resolved, so its conductivity is "
                f"unknown")
            return False
        sigma = getattr(mat, "sigma", None)
        if sigma is None:
            return str(material_name).lower() == "pec"
        try:
            return float(sigma) >= float(pec_sigma_threshold)
        except (TypeError, ValueError):
            # A traced sigma (material-as-design-variable) cannot be
            # compared host-side.
            unevaluated.append(
                f"geometry entry with material {material_name!r}: sigma is "
                f"not a concrete number (traced design variable), so PEC "
                f"promotion cannot be decided host-side")
            return False

    def _bounds(shape, what: str):
        lo = getattr(shape, "corner_lo", None)
        hi = getattr(shape, "corner_hi", None)
        if lo is not None and hi is not None:
            return lo, hi
        bbox = getattr(shape, "bounding_box", None)
        if bbox is None:
            unevaluated.append(
                f"{what} ({type(shape).__name__}): no corner_lo/corner_hi "
                f"and no bounding_box(), so it cannot be placed on the line")
            return None, None
        try:
            lo, hi = bbox()
        except Exception as exc:
            unevaluated.append(
                f"{what} ({type(shape).__name__}): bounding_box() raised "
                f"{type(exc).__name__}, so it cannot be placed on the line")
            return None, None
        return lo, hi

    # (shape, label_prefix, is_bbox_derived) for every registered conductor.
    candidates: list = []
    for _gi, ge in enumerate(geometry):
        shape = getattr(ge, "shape", None)
        mat = getattr(ge, "material_name", "")
        if shape is None or not _is_conductor(mat):
            continue
        candidates.append((shape, f"conductor '{mat}'",
                           not isinstance(shape, _Box)))
    for _ti, tc in enumerate(thin_conductors or ()):
        tshape = getattr(tc, "shape", None)
        if tshape is None:
            continue
        candidates.append((tshape, f"thin_conductor[{_ti}]",
                           not isinstance(tshape, _Box)))

    for shape, _what, _from_bbox in candidates:
        lo, hi = _bounds(shape, _what)
        if lo is None or hi is None:
            continue
        try:
            lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
            valid_bounds = (lo.shape == hi.shape == (3,)
                            and np.isfinite(lo).all() and np.isfinite(hi).all()
                            and np.all(hi >= lo))
        except (TypeError, ValueError):
            valid_bounds = False
        if not valid_bounds:
            unevaluated.append(
                f"{_what} ({type(shape).__name__}): bounds are not finite, "
                "ordered three-dimensional coordinates")
            continue
        # "x" = propagation axis, "y" = trace-width axis (issue #661).
        box_x_lo, box_x_hi = float(lo[_ip]), float(hi[_ip])
        box_y_lo, box_y_hi = float(lo[_iw]), float(hi[_iw])
        box_y_extent = box_y_hi - box_y_lo
        # Skip the line being measured (see docstring).
        if (
            abs(box_y_extent - w_trace) <= width_cell
            and box_y_lo - width_cell <= y_feed <= box_y_hi + width_cell
            and box_x_lo - dx <= x_feed <= box_x_hi + dx
        ):
            continue
        # The port reference, not lateral domain coverage, identifies ground.
        # Use the upper bound so metal rising into the substrate still counts.
        # Half a normal-axis cell of slack: a ground whose top sits a rounding
        # error or a rasterization residual above the reference is still the
        # ground, not a conductor containing the feed.
        if ground_plane is not None:
            if float(hi[_in]) <= ground_plane + 0.5 * (
                    dx if ground_cell is None else ground_cell):
                continue
        elif box_y_extent >= 0.8 * domain_y:
            continue
        # Distance from x_probe to the nearest edge of this box,
        # measured ALONG the propagation direction.
        if sign > 0:
            if box_x_lo > x_probe:
                d = box_x_lo - x_probe
            elif box_x_hi < x_probe:
                continue  # behind the probe
            else:
                d = box_x_lo - x_probe if signed_front_distance else 0.0
        else:
            if box_x_hi < x_probe:
                d = x_probe - box_x_hi
            elif box_x_lo > x_probe:
                continue
            else:
                d = x_probe - box_x_hi if signed_front_distance else 0.0
        if d < nearest_d:
            nearest_d = d
            _how = " (bounding box)" if _from_bbox else ""
            nearest_label = (
                f"{_what}{_how} at {_prop_ax}∈[{box_x_lo*1e3:.2f},"
                f"{box_x_hi*1e3:.2f}]mm "
                f"{_width_ax}∈[{box_y_lo*1e3:.2f},{box_y_hi*1e3:.2f}]mm"
            )
    return nearest_d, nearest_label, unevaluated
