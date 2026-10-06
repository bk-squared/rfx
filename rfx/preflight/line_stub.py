"""Band-scoped admission for open signal tails behind line ports (#1512)."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import math

import numpy as np

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

    @property
    def message(self):
        return (
            f"Line port {self.port_name!r} has an open stub behind the port "
            f"({self.overhang_m * 1e3:.6g} mm), ending inside the domain or "
            "at the absorber entrance. It shorts the port near "
            f"f = c/(4*L*sqrt(eps_eff)) = {self.frequency_hz / 1e9:.6g} GHz "
            f"(eps_eff={self.eps_eff:.6g}). Start the strip at the port plane."
        )


def _permittivity(sim, port, point):
    explicit = getattr(port, "eps_r_sub", None)
    if explicit is not None:
        return float(explicit)
    result = 1.0
    for entry in sim._geometry:
        bounds = declared_bounds(entry.shape)
        if bounds is not None and all(lo <= x <= hi for lo, x, hi in
                                      zip(bounds[0], point, bounds[1])):
            material = sim._resolve_material(entry.material_name)
            if material.sigma < sim._PEC_SIGMA_THRESHOLD:
                result = float(material.eps_r)
    return result


def _intervals(sim, grid, shape, coords, axis, lower, upper):
    """Axial support in the signal aperture, retaining gaps in a shape."""
    from rfx.geometry.smoothing import _declared_conductor_lattice
    bounds = declared_bounds(shape)
    nodes = (coords.x, coords.y, coords.z)
    if isinstance(shape, Box):
        tol = 32 * np.finfo(float).eps * max(
            *(abs(v) for b in bounds for v in b), 1e-12)
        if all(bounds[0][a] <= upper[a] + tol and bounds[1][a] >= lower[a] - tol
               for a in range(3) if a != axis):
            return [(float(bounds[0][axis]), float(bounds[1][axis]))]
        return []
    intervals = []
    for mask, cell_axes in _declared_conductor_lattice(sim, grid, shape, coords):
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
            intervals.append((float(nodes[axis][first]), float(nodes[axis][end])))
    return intervals


def line_stub_findings(sim, grid=None) -> list[LineStubFinding]:
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
    conductors = [continued_conductor_shape(sim, grid, entry.shape, entry=entry,
                                            unextendable=[])
                  for _, entry in entries]
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
                lower[2] = upper[2] = point[2] + port.height
                point[2] += port.height / 2
                eps = _permittivity(sim, port, point)
                eps = getattr(sim, "_msl_auto_probe_spacing", {}).get(
                    port.name, hammerstad_jensen_z0_eps_eff(
                        port.width, port.height, eps)[1])
            else:
                letter, sign, _ = _FACE_CONFIG[port.face]
                axis = "xyz".index(letter)
                # The pin centre, never the annular outer conductor.
                eps = _permittivity(sim, port, point)
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
            plane = float(port.position[axis])
            domain = float(sim._unresolved_domain[axis])
            tol = 32 * np.finfo(float).eps * max(domain, abs(plane), 1e-12)
            # Sweep separate rays across the signal aperture. Projecting all
            # its metal onto one axis would join an unrelated neighbouring
            # strip to this one across a transverse gap.
            rays = [(lower, upper)]
            if collection == "_msl_ports":
                cuts = {lower[width_axis], upper[width_axis]}
                for shape in conductors:
                    bounds = declared_bounds(shape)
                    if bounds is not None:
                        cuts.update(float(b[width_axis]) for b in bounds
                                    if lower[width_axis] <= b[width_axis]
                                    <= upper[width_axis])
                cuts = sorted(cuts)
                samples = cuts + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]
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
                                       lower_ray, upper_ray))
                merged = []
                for lo, hi in intervals:
                    if merged and lo <= merged[-1][1] + tol:
                        merged[-1][1] = max(merged[-1][1], hi)
                    else:
                        merged.append([lo, hi])
                for lo, hi in merged:
                    if not lo - tol <= plane <= hi + tol:
                        continue
                    end = lo if sign > 0 else hi
                    length = (plane - end) * sign
                    at_lo, at_hi = abs(end) <= tol, abs(end - domain) <= tol
                    if ((at_lo and not getattr(grid, f"pad_{'xyz'[axis]}_lo", 0))
                            or (at_hi and not getattr(grid, f"pad_{'xyz'[axis]}_hi", 0))):
                        continue  # A closed/periodic face is not an absorber entrance.
                    if length > tol and -tol <= end <= domain + tol:
                        endpoints.append((length, end))
            if endpoints:
                length, end = max(endpoints)
                findings.append(LineStubFinding(
                    collection, index, getattr(port, "name", f"coaxial_{index}"),
                    "xyz"[axis], end, length, float(eps),
                    299792458.0 / (4 * length * math.sqrt(eps))))
    return findings


# The calculator's requested band must survive its internal run()/forward()
# calls, which record probe planes without passing S-frequency arguments.
_READ_BAND = ContextVar("line_stub_read_band", default=None)


def read_band(sim, freqs=None):
    """The read interval, not the Gaussian pulse's centre/bandwidth."""
    if freqs is None:
        current = _READ_BAND.get()
        if current is not None and current[0] is sim:
            return current[1]
        sets = [getattr(p, "freqs", None)
                for p in (*sim._msl_ports, *sim._coaxial_ports)]
        sets = [np.asarray(f, dtype=float).ravel() for f in sets if f is not None]
        if not sets:
            return 0.0, float(sim._freq_max)
        freqs = np.concatenate(sets)
    try:
        values = np.asarray(freqs, dtype=float)
    except Exception as exc:
        raise ValueError("#1512: line-port read frequencies must be concrete") from exc
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("#1512: line-port read frequencies must be finite, nonnegative and nonempty")
    return float(values.min()), float(values.max())


def resonant_odd_orders(finding, band):
    """Inclusive odd-order index interval; no enumeration cap can drop a mode."""
    lo, hi = band[0] / 1.5 / finding.frequency_hz, 1.5 * band[1] / finding.frequency_hz
    # Expand by roundoff only, so exactly-on-boundary cases remain inclusive.
    tol = 16 * np.finfo(float).eps * max(1., abs(lo), abs(hi))
    first = max(1, math.ceil((lo - tol + 1) / 2))
    last = math.floor((hi + tol + 1) / 2)
    return (2 * first - 1, 2 * last - 1) if first <= last else None


def stub_message(finding, band):
    fq = finding.frequency_hz / 1e9
    orders = resonant_odd_orders(finding, band)
    if orders is None:
        relation = "outside the refusal interval for the band you read"
        frequencies = f"{fq:.6g}, {3*fq:.6g}, {5*fq:.6g}, ... GHz (odd multiples)"
    else:
        first, last = orders
        relation = "inside/near the band you read"
        frequencies = (f"{first*fq:.6g} GHz (order {first})" if first == last else
                       f"{first*fq:.6g}..{last*fq:.6g} GHz (odd orders {first}..{last})")
    return (
        f"The strip continues {finding.overhang_m*1e3:.6g} mm behind the port "
        f"and ends there; it is an open stub that shorts the port near {fq:.6g} GHz "
        f"(quarter wave); stub frequencies {frequencies}, {relation}. "
        f"Read band {band[0]/1e9:.6g}..{band[1]/1e9:.6g} GHz; "
        f"eps_eff={finding.eps_eff:.6g}; port {finding.port_name!r}. "
        "Fix: start the strip at the port plane (#1512)."
    )


def require_no_resonant_line_stub(sim, freqs=None):
    """Unconditional admission, including skip_preflight and internal callers."""
    if not (sim._msl_ports or sim._coaxial_ports):
        return
    import jax
    with jax.ensure_compile_time_eval():
        band = read_band(sim, freqs)
        for finding in line_stub_findings(sim):
            if resonant_odd_orders(finding, band) is not None:
                raise ValueError(stub_message(finding, band))


def line_stub_guard(frequency_argument):
    """One-line entry-point wrapper; preserves the signature and restores scope."""
    def decorate(function):
        @wraps(function)
        def guarded(self, *args, **kwargs):
            if not (self._msl_ports or self._coaxial_ports):
                return function(self, *args, **kwargs)
            band = read_band(self, kwargs.get(frequency_argument))
            current = _READ_BAND.get()
            if current is None or current[0] is not self or current[1] != band:
                require_no_resonant_line_stub(self, band)
            token = _READ_BAND.set((self, band))
            try:
                return function(self, *args, **kwargs)
            finally:
                _READ_BAND.reset(token)
        return guarded
    return decorate


def preflight_line_stubs(sim, warn):
    """Advisory report; the solve's guard owns refusal and its exact read band."""
    if not (sim._msl_ports or sim._coaxial_ports):
        return
    from rfx.preflight._common import PreflightWarning
    band = read_band(sim)
    for finding in line_stub_findings(sim):
        warn.warn(PreflightWarning(
            stub_message(finding, band), code="line_stub_behind_port",
            source="line_stub_findings", loc=f"{finding.collection}[{finding.port_index}]"),
            stacklevel=3)
