"""Shield a declared line end using the existing node-sampled short disc."""
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from rfx.geometry.csg import Cylinder
from rfx.model.coax import StampedEntity
from rfx.model.coax_stamp import PTFE_EPS_R, stamp_coaxial_short_plane


def shield_coax_end(grid, materials, cells, entities, *, end, center_xy,
                    outer_radius, z_lo_index, z_hi_index):
    """Trim only the pin, restore its fill, and retain the cap as an entity."""
    gap = max(2, int(np.ceil(float(outer_radius) / float(grid.dx))))
    top = end.port.face == 'top'
    cap_index = z_hi_index if top else z_lo_index
    window = (slice(cap_index - gap, cap_index + 1) if top
              else slice(cap_index, cap_index + gap + 1))
    pin = next(e for e in entities if e.entity_id.endswith('/pin'))
    removed = np.zeros(grid.shape, bool)
    removed[:, :, window] = np.asarray(pin.cells)[:, :, window]
    cells = np.asarray(cells) & ~removed
    eps = np.where(removed, PTFE_EPS_R, np.asarray(materials.eps_r))
    sigma = np.where(removed, 0., np.asarray(materials.sigma))
    materials = materials._replace(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sigma))
    trimmed = np.asarray(pin.cells) & ~removed
    occupied = np.where(trimmed)[2]
    lo, hi = int(occupied.min()), int(occupied.max())
    center = (*center_xy, ((lo + hi) / 2 - grid.pad_z_lo) * grid.dx)
    shape = Cylinder(center, pin.declared_radii_m[1], (hi - lo + 1) * grid.dx, axis='z')
    entities = [replace(e, cells=trimmed, shape=shape) if e is pin else e for e in entities]
    materials, cap = stamp_coaxial_short_plane(grid, materials,
        center_xy=center_xy, z_index=cap_index, outer_radius=outer_radius)
    center = (*center_xy, (cap_index - grid.pad_z_lo) * grid.dx)
    realized_radius = float(np.sqrt(cap.sum() * grid.dx ** 2 / np.pi))
    entities.append(StampedEntity(f'{pin.port_id}/cap/{cap_index}', pin.port_id,
        'volume', cap, Cylinder(center, outer_radius, grid.dx, axis='z'),
        (0., realized_radius), (0., outer_radius)))
    return materials, cells | cap, entities
