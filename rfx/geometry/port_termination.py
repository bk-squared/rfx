"""Declared port terminations and exact contact diagnostics."""
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True, eq=False)
class ConductorReference:
    """A registered conductor, independent of its current list position."""
    collection: str
    entry: object


def conductor_entries(sim):
    for entry in sim._geometry:
        if sim._resolve_material(entry.material_name).sigma >= sim._PEC_SIGMA_THRESHOLD:
            yield ConductorReference("_geometry", entry), entry
    for entry in sim._thin_conductors:
        yield ConductorReference("_thin_conductors", entry), entry


def resolve_terminates(sim, value, *, port):
    """Resolve explicit names at registration, identity before equality."""
    if value is None:
        return ()
    candidates = list(conductor_entries(sim))
    resolved = []
    for item in value if isinstance(value, (list, tuple)) else [value]:
        reference = None
        if isinstance(item, ConductorReference):
            reference = next((ref for ref, entry in candidates
                              if ref.collection == item.collection and entry is item.entry), None)
        elif isinstance(item, Integral) and not isinstance(item, bool):
            reference = next((ref for ref, _ in candidates
                              if ref.collection == "_geometry" and 0 <= item < len(sim._geometry)
                              and ref.entry is sim._geometry[item]), None)
        else:
            reference = next((ref for ref, entry in candidates if entry.shape is item), None)
            if reference is None:
                for ref, entry in candidates:
                    try:
                        equal = bool(entry.shape == item)
                    except (TypeError, ValueError):
                        equal = False
                    if equal:
                        reference = ref
                        break
        if reference is None:
            raise ValueError(f"{port}: unknown conductor in terminates={item!r}")
        if not any(ref.entry is reference.entry for ref in resolved):
            resolved.append(reference)
    return tuple(resolved)


def registered_ports(sim):
    for collection in ("_ports", "_msl_ports", "_coaxial_ports"):
        for index, port in enumerate(getattr(sim, collection, ())):
            if collection == "_ports" and port.impedance <= 0:
                continue
            yield collection, index, port


def _realization_key(sim, grid):
    """What an MSL default depends on: this grid and this conductor list."""
    pads = tuple(int(getattr(grid, f"pad_{a}_{s}", 0)) for a in "xyz" for s in ("lo", "hi"))
    return (id(grid), tuple(int(n) for n in grid.shape), pads,
            tuple(id(entry) for _, entry in conductor_entries(sim)))


def port_termination_references(sim, collection, port, grid):
    """Resolve an MSL default on this realization; retain explicit names."""
    if collection == "_msl_ports" and port.terminates is None:
        # Memoised per realization: the helper that asks is called once per
        # conductor, and resolving the default rasterizes every conductor, so
        # without this an assembly with an MSL port was quadratic in the
        # conductor count (review of PR #1178: 8.3 s at 20 conductors).
        realization = _realization_key(sim, grid)
        memo = sim.__dict__.get("_msl_default_memo")
        if memo is None or memo["realization"] != realization:
            memo = sim.__dict__["_msl_default_memo"] = {
                "realization": realization, "ports": {}}
        if id(port) not in memo["ports"]:
            memo["ports"][id(port)] = default_msl_terminates(
                sim, grid, position=port.position, width=port.width,
                height=port.height, direction=port.direction)
        return memo["ports"][id(port)]
    return tuple(ref for ref in port.terminates
                 if any(entry is ref.entry for entry in getattr(sim, ref.collection)))


def held_conductor_entries(sim, grid):
    entries = []
    for collection, _, port in registered_ports(sim):
        for ref in port_termination_references(sim, collection, port, grid):
            if not any(entry is ref.entry for entry in entries):
                entries.append(ref.entry)
    return entries


def lattice_intersects_aperture(lattice, nodes, lower, upper):
    """Occupied lattice support intersects the closed aperture, without cell slack."""
    # A footprint includes the segments between adjacent occupied nodes.
    # Requiring BOTH endpoints preserves gaps and a sheet's zero-thickness
    # normal plane; it does not grow a node by a contact-search radius.
    support = list(lattice)
    footprints = [(mask, axes) for mask, axes in lattice if not any(axes)]
    for axis in range(3):
        for mask, cell_axes in list(footprints):
            if cell_axes[axis]:
                continue
            adjacent = np.zeros_like(mask)
            lo, hi = [slice(None)]*3, [slice(None)]*3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            adjacent[tuple(lo)] = mask[tuple(lo)] & mask[tuple(hi)]
            if adjacent.any():
                axes = tuple(True if a == axis else cell_axes[a] for a in range(3))
                support.append((adjacent, axes))
                footprints.append((adjacent, axes))
    for mask, cell_axes in support:
        indices = []
        for axis, values in enumerate(nodes):
            line = np.asarray(values)
            high = (np.r_[line[1:], line[-1] + line[-1]-line[-2]]
                    if cell_axes[axis] else line)
            # Roundoff only, in the node array's precision; no geometric slack.
            eps = 8*np.finfo(line.dtype).eps*max(np.max(np.abs(line)), np.min(np.diff(line)))
            indices.append(np.flatnonzero((high >= lower[axis]-eps)
                                          & (line <= upper[axis]+eps)))
        if all(len(i) for i in indices) and mask[np.ix_(*indices)].any():
            return True
    return False


def default_msl_terminates(sim, grid, *, position, width, height, direction):
    """Name conductors exactly incident to the realized MSL signal aperture."""
    candidates = list(conductor_entries(sim))
    if not candidates:
        return ()
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid)
    from rfx.geometry.smoothing import _declared_conductor_lattice
    from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry

    coords = (coords_from_nonuniform_grid(grid) if hasattr(grid, "dx_arr")
              else coords_from_uniform_grid(grid))
    nodes = (coords.x, coords.y, coords.z)
    port = msl_port_from_entry(SimpleNamespace(
        position=position, width=width, height=height, direction=direction,
        impedance=50.0, waveform=None))
    span = msl_cross_section_span(grid, port)
    indices = [0, 0, 0]
    indices[span["prop_idx"]] = span["i_feed"]
    indices[span["width_idx"]] = span["w_lo"]
    indices[span["normal_idx"]] = span["n_hi"]
    lower = [float(nodes[axis][index]) for axis, index in enumerate(indices)]
    upper = lower.copy()
    transverse = span["width_idx"]
    upper[transverse] = float(nodes[transverse][span["w_hi"]])
    references = []
    for ref, entry in candidates:
        try:
            lattice = _declared_conductor_lattice(sim, grid, entry.shape, coords)
        except (ValueError, TypeError, IndexError, NotImplementedError):
            # A refused entry has no realized footprint; its own preflight
            # reports the refusal against that entry.
            continue
        if lattice_intersects_aperture(lattice, nodes, lower, upper):
            references.append(ref)
    return tuple(references)


def port_terminal_points(collection, port, grid, nodes):
    """The two physical terminal points, for diagnostics only."""
    if collection == "_msl_ports":
        return tuple(port.position), (*port.position[:2], port.position[2] + port.height)
    if hasattr(grid, "dx_arr"):
        from rfx.nonuniform import position_to_index
        def index_of(point):
            return position_to_index(grid, point)
    else:
        index_of = grid.position_to_index
    if collection == "_coaxial_ports":
        from types import SimpleNamespace
        from rfx.sources.coaxial_port import _coaxial_port_geometry
        tip = _coaxial_port_geometry(SimpleNamespace(position_to_index=index_of), port)[4]
        return tuple(port.position), tuple(tip)
    axis = "xyz".index(port.component[-1])
    if port.extent is None:
        index = index_of(port.position)
        start = tuple(float(nodes[a][index[a]]) for a in range(3))
        end = list(start)
        end[axis] = float(nodes[axis][min(index[axis]+1, len(nodes[axis])-1)])
    else:
        start = tuple(port.position)
        end = list(start)
        end[axis] += port.extent
    return start, tuple(end)
