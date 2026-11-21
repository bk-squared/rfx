"""Inputs a uniform coax line cannot preserve or measure independently."""
import numpy as np

from rfx.errors import RfxUnsupported


def refuse_other_s_drives(sim):
    """The coax wave fit reads a record driven by this port alone."""
    if sim._coaxial_ports and sim._ports:
        names = [f'_ports[{i}] ({"add_source/add_polarized_source" if p.is_soft_source else "add_port"}, impedance={p.impedance})'
                 for i, p in enumerate(sim._ports)]
        raise RfxUnsupported('add_coaxial_port(): S request refuses other excitations: ' + '; '.join(names))


def refuse_off_record_steps(sim, steps):
    if sim._coaxial_ports and steps is not None:
        raise RfxUnsupported(f'add_coaxial_port(): s_param_n_steps={steps} is unsupported; S11 reads the main record set by n_steps or until_decay')


def refuse_traced_materials(materials):
    from rfx.core.jax_utils import is_tracer
    import jax
    for name in materials._fields:
        if any(is_tracer(value) for value in jax.tree_util.tree_leaves(getattr(materials, name))):
            raise RfxUnsupported(f'add_coaxial_port(): traced material {name} is unsupported, including material assembly under jax.jit; use concrete material inputs')


def refuse_bore_materials(sim, root, materials, bore, edge_bore):
    from rfx.core.yee import lumped_components
    from rfx._internal._grid_metric import field_index
    from rfx.model.design_admission import sheet_edges
    refuse_traced_materials(materials)
    region = bore | np.logical_or.reduce(edge_bore)
    for entry in sim._geometry:
        material = sim._resolve_material(entry.material_name)
        if (material.debye_poles or material.lorentz_poles or material.chi3) and np.any(np.asarray(entry.shape.mask(root.grid)) & region):
            raise RfxUnsupported('add_coaxial_port(): dispersive or Kerr material inside the bore')
    magnetic = np.asarray(materials.mu_r)
    hits = (magnetic != 1) & region
    if np.any(hits):
        raise RfxUnsupported(f'add_coaxial_port(): mu_r != 1 inside the bore ({np.count_nonzero(hits)} cells, values {np.unique(magnetic[hits]).tolist()})')
    for i, part in enumerate(lumped_components(materials.sigma_film)):
        if part is not None and np.any((np.asarray(part) != 0) & region):
            raise RfxUnsupported(f'add_coaxial_port(): sigma_film component {i} inside the bore')
    for index, component, edges in sheet_edges(root, sim):
        if len(edges) and np.any(region[tuple(edges.T)]):
            raise RfxUnsupported(f'add_coaxial_port(): surface_impedance_f0 sheet {index} intersects the bore or reference aperture ({component})')
    for index, spec in enumerate(sim._lumped_rlc):
        edge = field_index(root.grid, spec.position, spec.component)
        if region[edge]:
            raise RfxUnsupported(f'add_coaxial_port(): lumped RLC stamp {index} inside the bore at {edge}')


def refuse_design_box(design, root):
    from rfx.model.design_admission import cells_in_box
    if design is not None:
        hits = cells_in_box(root.released_cells, design.bounds, root.grid.shape)
        if hits:
            raise RfxUnsupported(f'add_coaxial_port(): design_box {design.bounds} intersects {len(hits)} port-released cells')
