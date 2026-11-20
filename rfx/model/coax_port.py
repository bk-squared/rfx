"""Uniform coax port assembly, bore ownership and deferred TEM injection."""
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from rfx.errors import RfxUnsupported
from rfx.model.coax_line import build_coax_port_line, coax_port_layout


@dataclass(frozen=True)
class CoaxBoreRelease:
    """User metal removed before the port adds its own pin and shell."""

    cells: object
    edges: tuple
    occupancy_cells: object
    layout: object


def radial_mask(grid, port, radius):
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * float(grid.dx)
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * float(grid.dx)
    return ((x[:, None] - port.position[0]) ** 2
            + (y[None, :] - port.position[1]) ** 2) < radius ** 2


def clear_coax_bore(root, bore, layout, *, edge_bore=None):
    """Clear user cells and already realized edges before the stamp's union."""
    cells = np.zeros(root.grid.shape, bool) if root.pec_cells is None else np.asarray(root.pec_cells)
    edges = root.edges
    edge_bore = bore if edge_bore is None else edge_bore
    cleared = CoaxBoreRelease(np.argwhere(cells & bore),
        tuple(np.argwhere(np.asarray(e) & edge_bore) for e in edges), np.argwhere(bore), layout)
    root = replace(root, pec_cells=cells & ~bore,
                   pec_edges=tuple(np.asarray(e) & ~edge_bore for e in edges),
                   coax_bore_release=cleared)
    return root


def stamp_coax_pin(grid, port, layout):
    """Only the pin continues in front; cell faces terminate at the tip node."""
    count = int(round(float(port.pin_length) / float(grid.dx)))
    direction = -1 if port.face == 'top' else 1
    tip = layout.reference_index + direction * count
    if not 0 <= tip < grid.shape[2]:
        raise RfxUnsupported('add_coaxial_port(): pin_length leaves the domain')
    cells = np.zeros(grid.shape, bool)
    lo, hi = sorted((layout.reference_index, tip))
    cells[:, :, lo:hi] = radial_mask(grid, port, port.pin_radius)[:, :, None]
    return cells


def _refuse_bore_materials(sim, root, bore):
    for entry in sim._geometry:
        material = sim._resolve_material(entry.material_name)
        if not (material.debye_poles or material.lorentz_poles or material.chi3):
            continue
        if np.any(np.asarray(entry.shape.mask(root.grid)) & bore):
            raise RfxUnsupported('add_coaxial_port(): dispersive or Kerr material inside the bore')


def _refuse_front_pec(sim, root, port, layout):
    from rfx.boundaries.pec import realized_pec_edge_masks
    front = np.zeros(root.grid.shape, bool)
    ref = layout.reference_index
    window = slice(None, ref) if port.face == 'top' else slice(ref, None)
    front[:, :, window] = radial_mask(root.grid, port, port.outer_radius)[:, :, None]
    allowed_cells = np.zeros(root.grid.shape, bool)
    allowed_edges = [np.zeros(root.grid.shape, bool) for _ in range(3)]
    pin = stamp_coax_pin(root.grid, port, layout)
    pin_edges = realized_pec_edge_masks(pin, periodic=root.periodic)
    for key, cells, sheet, wire, shape in root.assembly_entries:
        if any(key == id(ref.entry) for ref in port.terminates):
            owned = realized_pec_edge_masks(cells, sheets=() if sheet is None else (sheet,),
                wires=() if wire is None else (wire,), periodic=root.periodic)
            if any(np.any(e & p) for e, p in zip(owned, pin_edges)):
                if cells is not None:
                    allowed_cells |= np.asarray(cells)
                allowed_edges = [a | e for a, e in zip(allowed_edges, owned)]
    if ((root.pec_cells is not None and np.any(np.asarray(root.pec_cells) & front & ~allowed_cells))
            or any(np.any(np.asarray(e) & front & ~a) for e, a in zip(root.edges, allowed_edges))):
        raise RfxUnsupported('add_coaxial_port(): user PEC in front of the reference plane inside the bore; name the pin contact in terminates=')


def deferred_coax_sources(*, spec, magnetic_sources, materials):
    """Resolve both drive families at the same queue seam as other ports."""
    magnetic_sources.extend(spec.magnetic_sources)
    return list(spec.electric_sources)


def assemble_coax_port(sim, grid, materials, root, sources, mag_sources, *, n_steps,
                       occupancy=None):
    """Cut the user bore, stamp line and protruding pin, and queue the TEM pair."""
    if not sim._coaxial_ports:
        return materials, root, occupancy
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.model.conductors import coax_kernel_conductors
    port = sim._coaxial_ports[0]
    pads = tuple(getattr(grid, f'pad_{a}_{s}') for a in 'xyz' for s in ('lo', 'hi'))
    layout = coax_port_layout(grid, port, absorber_pads=pads, ntff=sim._ntff)
    bore = np.zeros(grid.shape, bool)
    bore[:, :, layout.z_lo_index:layout.z_hi_index + 1] = radial_mask(grid, port, port.outer_radius)[:, :, None]
    edge_bore = bore.copy()
    edge_bore[:, :, layout.z_hi_index + 1] = radial_mask(grid, port, port.outer_radius)
    _refuse_bore_materials(sim, root, bore)
    _refuse_front_pec(sim, root, port, layout)
    root = clear_coax_bore(root, bore, layout, edge_bore=edge_bore)
    layout, line = build_coax_port_line(grid, materials, port, absorber_pads=pads,
        ntff=sim._ntff, n_steps=n_steps)
    pin = stamp_coax_pin(grid, port, layout)
    if pin.any():
        from rfx.model.coax import StampedEntity
        from rfx.geometry.csg import Cylinder
        center = list(port.position)
        center[2] += (-1 if port.face == 'top' else 1) * port.pin_length / 2
        line.conductor_entities.append(StampedEntity('coaxial_port[0]/pin_extension',
            'coaxial_port[0]', 'volume', pin,
            Cylinder(tuple(center), port.pin_radius, port.pin_length, axis='z'),
            (0., port.pin_radius), (0., port.pin_radius)))
    cells = line.pec_cells | pin
    edges = realized_pec_edge_masks(cells, periodic=root.periodic)
    released = np.argwhere(bore | cells)
    root = replace(root, released_cells=np.unique(np.concatenate((root.released_cells, released)), axis=0))
    if occupancy is not None:
        occupancy = jnp.where(jnp.asarray(bore | cells), 0., occupancy)
    root = coax_kernel_conductors(sim, grid, line.materials, cells, edges,
        line.conductor_entities, root=root, skip_preflight=True)
    sources.defer(deferred_coax_sources, spec=line.ends[0].source_spec, magnetic_sources=mag_sources)
    return line.materials, root, occupancy
