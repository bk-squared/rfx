"""Admission of edge-bound declarations to occupancy designs."""
from dataclasses import replace

import numpy as np

from rfx._grid_metric import field_index
from rfx.diagnostic_records import Diagnostic


def box_cells(bounds, shape):
    """Cells a design box writes: [lo, hi) per axis (``pec_occupancy_box_keep``).

    An E edge is reached by the box iff one of its four incident cells is in
    this set (keep is the product over those cells; permittivity averages them).
    """
    return tuple((int(bounds[2*a]), min(int(bounds[2*a+1]), shape[a])) for a in range(3))


def cells_in_box(cells, bounds, shape):
    window = box_cells(bounds, shape)
    return {tuple(map(int, c)) for c in cells
            if all(window[a][0] <= c[a] < window[a][1] for a in range(3))}


def sheet_edges(conductors, sim, sheet_operator=None):
    """Attribute the realized operator edges to their declared f0 sheets."""
    from rfx.model.conductors import SheetConductors
    indices = [i for i, tc in enumerate(sim._thin_conductors)
               if not tc.is_pec and tc.surface_impedance_f0 is not None]
    for index, spec in zip(indices, conductors.sheet_impedance, strict=True):
        ctx = SheetConductors((spec,), conductors.periodic).sheet_context(conductors.pec_edges)
        for component in ('ex', 'ey', 'ez'):
            mask = np.asarray(getattr(ctx, 'mask_' + component), dtype=bool)
            if sheet_operator is not None:
                mask = mask & np.asarray(getattr(sheet_operator, 'mask_' + component))
            yield index, component, np.argwhere(mask)


def publish_declarations(conductors, sim, design, sheet_operator=None, *, occupancy=None):
    from rfx.model.conductors import publish_port_release
    for index, spec in enumerate(sim._lumped_rlc):
        # Both metadata builders use this exact field-index resolver.
        edge = field_index(conductors.grid, spec.position, spec.component)
        conductors = publish_port_release(conductors, (edge,), component=spec.component,
            entity_id=f'lumped_rlc[{index}]', kind='lumped_rlc')
    if design is None or occupancy is not None:
        for index, component, edges in sheet_edges(conductors, sim, sheet_operator):
            conductors = publish_port_release(conductors, edges, component=component,
                entity_id=f'thin_conductor[{index}]', kind='f0_sheet')
    return conductors


def source_findings(sim, conductors, occupancy, design):
    """Geometry/presence only: identical for concrete and traced occupancies."""
    from rfx.model.conductors import publish_port_release
    sources = [(i, entry) for i, entry in enumerate(sim._ports) if entry.impedance == 0]
    if not sources:
        return ()
    explanation = ('its drive is weighted by the design metal on its edge and is '
                   'silent where that metal is complete')
    findings = []
    if occupancy is not None:
        names = ', '.join(f'source[{i}]' for i, _ in sources)
        findings.append(Diagnostic(code='source_under_design_occupancy', severity='advisory',
            subject=names, message=f'{names} under whole-domain design occupancy: '
            "each source's drive is weighted by the design metal on its edge and is "
            'silent where that metal is complete', values={'sources': names},
            source='design_admission'))
    if design is None:
        return tuple(findings)
    bounds = design.bounds
    for index, source in sources:
        edge = tuple(map(int, field_index(conductors.grid, source.position, source.component)))
        empty = replace(conductors, released_port_edges=(), _released_port_details=(),
                        released_cells=np.empty((0, 3), dtype=np.int32))
        incident = publish_port_release(empty, (edge,), component=source.component,
                                       entity_id=f'source[{index}]', kind='source')
        if cells_in_box(incident.released_cells, bounds, conductors.grid.shape):
            findings.append(Diagnostic(code='source_in_design_region', severity='advisory',
                subject=f'source[{index}]', message=f'source[{index}] lies inside design '
                f'region {bounds}; {explanation}', values={'component': source.component,
                'edge_index': str(edge), 'box_cells': str(bounds)}, source='design_admission'))
    return tuple(findings)


def with_design_diagnostics(result, conductors):
    from rfx._diagnostic_transport import result_with_diagnostics
    return result_with_diagnostics(result, conductors.design_findings)
