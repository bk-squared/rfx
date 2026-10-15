"""Band-scoped admission for open signal tails behind line ports (#1512)."""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys

import numpy as np

from rfx.preflight.msl_codes import msl_diagnostic, msl_text, stub_diagnostic, msl_error
from rfx.geometry.csg import Box, declared_bounds
from rfx.geometry.port_termination import conductor_entries


@dataclass(frozen=True)
class LineStubFinding:
    collection: str
    port_index: int
    port_name: str
    axis: str
    endpoint_m: float
    overhang_m: float
    eps_eff: float
    frequency_hz: float
    declared_overhang_m: float = 0.0
    port_node_m: float = 0.0
    substrate_eps_r: float = 1.0
    declared_eps_r_sub: float | None = None
    end_extension_m: float = 0.0

    @property
    def effective_length_m(self):
        """Realized node-to-node length plus the open-end extension."""
        return self.overhang_m + self.end_extension_m

    @property
    def message(self):
        return stub_message(self)


def _permittivity(sim, port, point, grid, coords, cache, axis=None):
    """Material owning the sampled stub node, using the assembler's masks.

    Sample halfway along the stub and halfway through the substrate (MSL) or
    on the mid-annulus ring around the pin (coax, ``axis`` given). Later
    dielectric entries overwrite earlier ones, just as in assemble_cells.
    PEC does not overwrite dielectric epsilon. No material means fallback.
    """
    from rfx.geometry.rasterize_grid import (
        _material_cell_mask, centres_from_nonuniform_grid, centres_from_uniform_grid)
    explicit = getattr(port, "eps_r_sub", None)
    result = None
    points = [point]
    if axis is not None:
        # Coax: the field lives between pin and shield, and the pin-centre cell
        # is metal. Sample the mid-annulus ring; later samples do not override.
        radius = (port.pin_radius + port.outer_radius) / 2
        points = []
        for a in (a for a in range(3) if a != axis):
            for offset in (-radius, radius):
                points.append([x + offset * (b == a) for b, x in enumerate(point)])
    samples = [tuple(int(np.argmin(np.abs(np.asarray(nodes) - x)))
                     for nodes, x in zip(coords[:3], p)) for p in points]
    for entry in sim._geometry:
        material = sim._resolve_material(entry.material_name)
        if material.sigma >= sim._PEC_SIGMA_THRESHOLD:
            continue
        key = ('dielectric', id(entry))
        if key not in cache:
            nonuniform = hasattr(grid, 'dx_arr')
            centres = (centres_from_nonuniform_grid(grid, coords) if nonuniform
                       else centres_from_uniform_grid(grid))
            cache[key] = np.asarray(_material_cell_mask(
                entry.shape, coords, centres, grid=None if nonuniform else grid))
        if any(cache[key][sample] for sample in samples):
            result = float(material.eps_r)
    return result if result is not None else float(explicit if explicit is not None else 1.)


def _intervals(sim, grid, shape, coords, axis, lower, upper, cache):
    """Axial support in the signal aperture, retaining gaps in a shape."""
    from rfx.geometry.smoothing import _declared_conductor_lattice
    bounds = declared_bounds(shape)
    nodes = (coords.x, coords.y, coords.z)
    intervals = []
    key = id(shape)
    if key not in cache:
        try:
            cache[key] = _declared_conductor_lattice(sim, grid, shape, coords)
        except NotImplementedError as exc:
            # One shape the lattice cannot place must not switch the check off
            # for the conductors it can: skip this shape and say so.
            cache[key] = []
            cache.setdefault("uninspectable", []).append(f"{type(shape).__name__}: {exc}")
    for mask, cell_axes in cache[key]:
        indices = []
        for a, values in enumerate(nodes):
            values = np.asarray(values)
            end = (np.r_[values[1:], values[-1] + values[-1] - values[-2]]
                   if cell_axes[a] else values)
            indices.append(np.arange(len(values)) if a == axis else
                           np.flatnonzero((end >= lower[a]) & (values <= upper[a])))
        if not all(len(i) for i in indices):
            continue
        occupied = np.asarray(mask)[np.ix_(*indices)].any(
            axis=tuple(a for a in range(3) if a != axis))
        starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
        for first, last in zip(starts, ends):
            end = min(last + int(cell_axes[axis]), len(nodes[axis]) - 1)
            lo, hi = float(nodes[axis][first]), float(nodes[axis][end])
            declared_lo, declared_hi = (bounds[0][axis], bounds[1][axis]) if bounds else (lo, hi)
            intervals.append((lo, hi, declared_lo, declared_hi))
    return intervals


def open_end_extension(width, height, eps_eff):
    """Hammerstad open-end length extension of a microstrip (declared w, h).

    The fringing field at an open end makes the stub electrically longer than
    its metal; without it a short stub's notch is placed too high (#1512).
    """
    u = width / height
    return (0.412 * height * (eps_eff + 0.3) * (u + 0.264)
            / ((eps_eff - 0.258) * (u + 0.8)))


def line_stub_findings(sim, grid=None, uninspectable=None) -> list[LineStubFinding]:
    """Find attached straight signal tails ending at/before the domain face.

    MSL directions point into the device; a coax face supplies the equivalent
    direction. Ground/outer conductors do not intersect the signal aperture.
    Port-generated coax launch metal is deliberately excluded: its source gap
    and internal termination belong to the coax solver, not declared geometry.
    ``terminates=()`` is respected through production conductor continuation;
    a tail continued into the absorber is not an internal open end.
    """
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid)
    from rfx.sources.coaxial_port import _FACE_CONFIG
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

    if not (sim._msl_ports or sim._coaxial_ports):
        return []
    entries = list(conductor_entries(sim))
    if not entries and not getattr(sim, "_pinned_sheets", ()):
        return []  # Generated coax metal is not a declared signal tail.
    grid = sim._build_realized_grid() if grid is None else grid
    coords = (coords_from_nonuniform_grid(grid) if hasattr(grid, "dx_arr")
              else coords_from_uniform_grid(grid))
    from rfx.geometry.smoothing import continued_conductor_shape
    cache = {}
    conductors = []
    for _, entry in entries:
        try:
            conductors.append(continued_conductor_shape(
                sim, grid, entry.shape, entry=entry, unextendable=[]))
        except NotImplementedError as exc:
            cache.setdefault("uninspectable", []).append(
                f"{type(entry.shape).__name__}: {exc}")
    nodes = (coords.x, coords.y, coords.z)
    for sheet in getattr(sim, "_pinned_sheets", ()):
        tangents = [a for a in range(3) if a != sheet.normal_axis]
        lower, upper = [0.] * 3, [0.] * 3
        ranges = {sheet.normal_axis: (sheet.plane_index, sheet.plane_index),
                  tangents[0]: sheet.i_range, tangents[1]: sheet.j_range}
        for a, (lo, hi) in ranges.items():
            pad = int(getattr(grid, f"pad_{'xyz'[a]}_lo", 0))
            lower[a], upper[a] = float(nodes[a][lo + pad]), float(nodes[a][hi + pad])
        conductors.append(Box(tuple(lower), tuple(upper)))
    findings = []
    for collection in ("_msl_ports", "_coaxial_ports"):
        for index, port in enumerate(getattr(sim, collection)):
            point = list(map(float, port.position))
            lower, upper = point.copy(), point.copy()
            if collection == "_msl_ports":
                axis = "xyz".index(port.direction[-1])
                sign = 1 if port.direction[0] == "+" else -1
                width_axis = 1 - axis
                lower[width_axis] -= port.width / 2
                upper[width_axis] += port.width / 2
                from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry
                span = msl_cross_section_span(grid, msl_port_from_entry(port))
                lower[2] = upper[2] = float(nodes[2][span["n_hi"]])
                point[2] += port.height / 2
            else:
                letter, sign, _ = _FACE_CONFIG[port.face]
                axis = "xyz".index(letter)
                # The pin centre, never the annular outer conductor.
            signal_shapes = conductors
            if collection == "_coaxial_ports":
                # A solid ground wall through the pin centre is a junction
                # short, not a continuation of the line's centre conductor.
                signal_shapes = []
                for shape in conductors:
                    bounds = declared_bounds(shape)
                    if bounds is not None and all(
                            point[a] - port.pin_radius <= bounds[0][a]
                            and bounds[1][a] <= point[a] + port.pin_radius
                            for a in range(3) if a != axis):
                        signal_shapes.append(shape)
            declared_plane = float(port.position[axis])
            from rfx.sources.msl_port import _msl_position_to_index
            port_index = (span["i_feed"] if collection == "_msl_ports" else
                          _msl_position_to_index(grid, port.position)[axis])
            plane = float(nodes[axis][port_index])
            domain = float(sim._unresolved_domain[axis])
            tol = 32 * np.finfo(float).eps * max(domain, abs(plane), 1e-12)
            # Sweep separate rays across the signal aperture. Projecting all
            # its metal onto one axis would join an unrelated neighbouring
            # strip to this one across a transverse gap.
            rays = [(lower, upper)]
            if collection == "_msl_ports":
                samples = [float(nodes[width_axis][w]) for w in span["width_nodes"]]
                rays = []
                for sample in samples:
                    lo, hi = lower.copy(), upper.copy()
                    lo[width_axis] = hi[width_axis] = sample
                    rays.append((lo, hi))
            endpoints = []
            for lower_ray, upper_ray in rays:
                intervals = sorted(interval for shape in signal_shapes
                                   for interval in _intervals(
                                       sim, grid, shape, coords, axis,
                                       lower_ray, upper_ray, cache))
                merged = []
                for lo, hi, declared_lo, declared_hi in intervals:
                    if merged and lo <= merged[-1][1] + tol:
                        merged[-1][1] = max(merged[-1][1], hi)
                        merged[-1][2] = min(merged[-1][2], declared_lo)
                        merged[-1][3] = max(merged[-1][3], declared_hi)
                    else:
                        merged.append([lo, hi, declared_lo, declared_hi])
                for lo, hi, declared_lo, declared_hi in merged:
                    if not lo - tol <= plane <= hi + tol:
                        continue
                    end = lo if sign > 0 else hi
                    declared_end = declared_lo if sign > 0 else declared_hi
                    declared_length = max(0.0, (declared_plane - declared_end) * sign)
                    length = max(0.0, (plane - end) * sign)
                    at_lo, at_hi = abs(end) <= tol, abs(end - domain) <= tol
                    if ((at_lo and not getattr(grid, f"pad_{'xyz'[axis]}_lo", 0))
                            or (at_hi and not getattr(grid, f"pad_{'xyz'[axis]}_hi", 0))):
                        continue  # A closed/periodic face is not an absorber entrance.
                    if length > tol and -tol <= end <= domain + tol:
                        endpoints.append((length, end, declared_length))
            if endpoints:
                length, end, declared_length = max(endpoints)
                point[axis] = (plane + end) / 2
                substrate_eps = _permittivity(
                    sim, port, point, grid, coords, cache,
                    axis if collection == "_coaxial_ports" else None)
                eps = substrate_eps
                # Coax: no microstrip open-end formula applies to a pin tail, so
                # the extension stays 0. ASSUMPTION, unverified: its end effect is small.
                extension = 0.0
                if collection == "_msl_ports":
                    eps = hammerstad_jensen_z0_eps_eff(
                        port.width, port.height, eps)[1]
                    extension = open_end_extension(port.width, port.height, eps)
                findings.append(LineStubFinding(
                    collection, index, getattr(port, "name", f"coaxial_{index}"),
                    "xyz"[axis], end, length, float(eps),
                    299792458.0 / (4 * (length + extension) * math.sqrt(eps)),
                    declared_length, plane,
                    substrate_eps, getattr(port, "eps_r_sub", None), extension))
    if uninspectable is not None:
        uninspectable.extend(dict.fromkeys(cache.get("uninspectable", [])))
    return findings


@dataclass(frozen=True)
class _ReadScope:
    simulation: object
    band: tuple[float, float]


def _active_band(sim):
    """Outermost live entry owns the band, without a wrapper or retained frame.

    Each synchronous entry holds its admission result in `_line_stub_scope`.
    Reading only live ancestor frames makes normal/exceptional return end the
    scope automatically, and keeps unrelated simulations and threads isolated.
    Preflight's read-only shallow copy shares the geometry/port containers.
    """
    frame = sys._getframe(1)
    band = None
    try:
        while frame is not None:
            scope = frame.f_locals.get('_line_stub_scope')
            if isinstance(scope, _ReadScope) and (scope.simulation is sim or all(
                    hasattr(sim, name) and getattr(scope.simulation, name, None) is getattr(sim, name)
                    for name in ('_geometry', '_msl_ports', '_coaxial_ports'))):
                band = scope.band
            frame = frame.f_back
        return band
    finally:
        del frame


def read_band(sim, freqs=None):
    """The read interval, not the Gaussian pulse's centre/bandwidth."""
    current = _active_band(sim)
    if current is not None:
        return current
    if freqs is None:
        # MSL and coax entries have no frequency-set field. Their calculators
        # own the requested frequencies; run/forward receive theirs explicitly.
        return 0.0, float(sim._freq_max)
    from rfx._diagnostic_transport import diagnostic_refusal
    try:
        values = np.asarray(freqs, dtype=float)
    except Exception as exc:
        raise diagnostic_refusal(ValueError("#1512: line-port read frequencies must be concrete")) from exc
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise diagnostic_refusal(
            ValueError("#1512: line-port read frequencies must be finite, nonnegative and nonempty")
        )
    return float(values.min()), float(values.max())


def resonant_odd_orders(finding, band):
    """Inclusive odd-order index interval; no enumeration cap can drop a mode."""
    lo, hi = band[0] / 1.5 / finding.frequency_hz, 1.5 * band[1] / finding.frequency_hz
    # Expand by roundoff only, so exactly-on-boundary cases remain inclusive.
    tol = 16 * np.finfo(float).eps * max(1., abs(lo), abs(hi))
    first = max(1, math.ceil((lo - tol + 1) / 2))
    last = math.floor((hi + tol + 1) / 2)
    return (2 * first - 1, 2 * last - 1) if first <= last else None


def stub_message(finding, band=None):
    return stub_diagnostic(finding, band).message


def _uninspectable_warning(skipped):
    from rfx.preflight._common import PreflightWarning
    return PreflightWarning(
        msl_diagnostic(
            "msl.line_stub_inspection_unavailable",
            msl_text(
                "line_stub_inspection_unavailable",
                shape_count=len(skipped),
                shapes="; ".join(skipped),
            ),
            source="line_stub_findings",
        ),
        code="line_stub_inspection_unavailable",
        source="line_stub_findings",
    )


def _warn_uninspectable(skipped):
    """Solve-entry warning attributed to the first frame outside the rfx package.

    Entries differ in depth (forward stages its declarations in one more frame),
    so the stack level is counted, not fixed; no entry needs a line for it.
    """
    import os
    import warnings
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    level, frame = 1, sys._getframe(0)
    try:
        while frame.f_back is not None and (
                os.path.abspath(frame.f_code.co_filename).startswith(package)
                or "jax" + os.sep in frame.f_code.co_filename
                or frame.f_code.co_filename.startswith("<")):
            frame, level = frame.f_back, level + 1
    finally:
        del frame
    warning = _uninspectable_warning(skipped)
    warnings.warn(
        warning,
        stacklevel=level,
    )


def require_no_resonant_line_stub(sim, freqs=None):
    """Unconditional admission, including skip_preflight and internal callers."""
    if not (sim._msl_ports or sim._coaxial_ports):
        return
    import jax
    with jax.ensure_compile_time_eval():
        band = read_band(sim, freqs)
        skipped = []
        try:
            findings = line_stub_findings(sim, uninspectable=skipped)
        except NotImplementedError as exc:
            # Unsupported inspection leaves admission to the owning lane, but not silently.
            findings = []
            skipped.append(str(exc))
        except ValueError as exc:
            msl_error(
                msl_diagnostic(
                    "msl.line_stub_realization",
                    msl_text(
                        "line_stub_realization",
                        detail=exc,
                    ),
                    source="line_stub_findings",
                ),
                error=exc,
            )
            raise
        if skipped:
            _warn_uninspectable(skipped)
        for finding in findings:
            if resonant_odd_orders(finding, band) is not None:
                raise msl_error(
                    stub_diagnostic(
                        finding,
                        band,
                        severity="refusal",
                    )
                )


def line_stub_admission(sim, freqs=None):
    """Store the returned scope in the entry body; never wrap a warning emitter."""
    if not (sim._msl_ports or sim._coaxial_ports):
        return None
    band = read_band(sim, freqs)
    if _active_band(sim) is None:
        require_no_resonant_line_stub(sim, band)
    return _ReadScope(sim, band)


def preflight_line_stubs(sim, warn):
    """Advisory report; the solve's guard owns refusal and its exact read band."""
    if not (sim._msl_ports or sim._coaxial_ports):
        return
    from rfx.preflight._common import PreflightErrorWarning, PreflightWarning
    band = read_band(sim)
    skipped = []
    try:
        findings = line_stub_findings(sim, uninspectable=skipped)
    except NotImplementedError as exc:
        findings = []
        skipped.append(str(exc))
    except ValueError as exc:
        # Preserve the blocking realization error without aborting later checks.
        # Solve admission still raises it unconditionally, even with preflight off.
        warn.warn(
            PreflightErrorWarning(
                msl_diagnostic(
                    "msl.line_stub_realization",
                    msl_text(
                        "line_stub_realization",
                        detail=str(exc),
                    ),
                    source="line_stub_findings",
                ),
                code="line_stub_realization",
                source="line_stub_findings",
            ),
            stacklevel=3,
        )
        return
    if skipped:
        warn.warn(_uninspectable_warning(skipped), stacklevel=3)
    for finding in findings:
        warn.warn(
            PreflightWarning(
                stub_diagnostic(finding, band),
                code="line_stub_behind_port",
                severity="warning",
                source="line_stub_findings",
                loc=f"{finding.collection}[{finding.port_index}]",
            ),
            stacklevel=3,
        )
