"""Global graded port assembly before slab staging."""
import jax
import jax.numpy as jnp
import numpy as np

from rfx import _grid_metric
from rfx.model import conductors as conductor_model
from rfx.model import source_coefficients as sc
from rfx.model.graded_ports import assemble_graded_ports
from rfx.model.overrides import apply_material_overrides
from rfx.sources.port_drive import PortSourceQueue


def assemble(sim, grid, materials, conductors, cells, sheets, wires, *,
             eps_override, sigma_override, pec_mask_override, pec_occupancy_override, debye_spec,
             lorentz_spec, n_steps, lane, gather, diagnostics, s_param_freqs, mesh):
    """Apply overrides, stamp loads, release edges, then resolve ordered drives."""
    from rfx.api._source_semantics import needs_scale
    from rfx.model.source_coefficients import make_current_source, current_source_samples, current_source_volume
    from rfx.runners.nonuniform import _port_drive_source
    from rfx.runners._admission import admit
    from rfx.simulation import SourceSpec
    from rfx.stepping.slab_wrap import select_step

    ports_present = any(p.impedance > 0 for p in sim._ports)
    call = "run(devices=...)" if lane == "run_distributed_nu" else "forward(distributed=True)"
    off = "omit devices=" if lane == "run_distributed_nu" else "omit distributed=True and devices="
    if ports_present:
        if any(p.extent is not None and p.reference_plane_cells is not None for p in sim._ports):
            raise NotImplementedError(f'Wire ports with reference_plane_cells are not supported by {call} on a graded mesh; the V_ref path needs pre-injection recordings. Use one device ({off}).')
        if select_step(cpml=bool(grid.cpml_layers), debye=debye_spec is not None,
                       lorentz=lorentz_spec is not None, pmc=bool(sim._boundary_spec.pmc_faces())) != 'wrapped':
            raise NotImplementedError('Graded multi-device ports require the wrapped step; PMC port faces are not supported.')
    admit(sim, lane, diagnostics=diagnostics)
    overridden = eps_override is not None or sigma_override is not None
    # Current-source pole terms are independent of the override, as before.
    in_program = mesh.devices.size < 2
    drive_model = sc.dispersive_drive_model(materials, debye_spec, lorentz_spec, grid,
                                          in_program=in_program)
    if overridden:
        materials = apply_material_overrides(materials, eps_override=eps_override,
            sigma_override=sigma_override, judge=(lorentz_spec, grid, debye_spec), in_program=in_program)
    if pec_mask_override is not None:
        cells = pec_mask_override if cells is None else cells | pec_mask_override
    conductors = conductor_model.kernel_conductors(sim, grid, materials, cells,
        sheets, wires, periodic=(False,) * 3, root=conductors)
    # The distributed lane is outside declaration_setup (_dispatch.py), so an
    # outer jit would trace the port's host cell sizes; traced inputs stay recorded.
    with jax.ensure_compile_time_eval():
        ports = assemble_graded_ports(sim, grid, materials, materials, conductors,
            pec_edge_masks=conductors.pec_edges, sizing_n=n_steps,
            lane='fwd_nonuniform' if lane == 'fwd_distributed_nu' else 'run_nonuniform',
            s_param_freqs=s_param_freqs)
    materials, conductors = ports.materials, ports.conductors
    drives = {}
    for index, drive in ports.drives:
        drives.setdefault(index, []).append(drive)
    if overridden:
        # A1 is declared only for edges with no pole contribution, including
        # a passive receiving port whose load changes the local coefficient.
        for spec in ports.port_specs:
            for cell in spec['live_cells']:
                axis = ('ex', 'ey', 'ez').index(spec['component'])
                window = tuple(slice(i if a == axis else max(0, i - 1), i + 1)
                               for a, i in enumerate(cell))
                for pole_spec in (debye_spec, lorentz_spec):
                    if pole_spec is None:
                        continue
                    poles, masks = pole_spec
                    masks = masks if isinstance(masks, (tuple, list)) else [masks] * len(poles)
                    if any(m is None or np.any(np.asarray(m)[window]) for m in masks):
                        raise NotImplementedError('Graded multi-device port overrides on Debye/Lorentz live edges are not measured by the step2-graded-port-s design (D3b).')
    # Drive samples and scales are host constants, as declaration_setup makes
    # them on one device: a user jit must not stage them (rfx #1840).
    with jax.ensure_compile_time_eval():
        queue, material_drive = PortSourceQueue(), []
        for index, pe in enumerate(sim._ports):
            if pe.impedance == 0:
                cell = _grid_metric.field_index(grid, pe.position, pe.component)
                if overridden and not needs_scale(pe.amplitude_kind, 'cb_over_dv'):
                    volume, _ = current_source_volume(grid, cell, pe.component)
                    queue.append((*cell, pe.component, current_source_samples(grid, pe.waveform, n_steps)))
                    material_drive.append((volume, sc.debye_pole_term(drive_model, cell, pe.component, grid.dt)))
                elif ports_present and not overridden:
                    queue.defer(make_current_source, grid, cell, pe.component, pe.waveform,
                        n_steps, amplitude_kind=pe.amplitude_kind)
                    material_drive.append(None)
                else:
                    queue.append(make_current_source(grid, cell, pe.component, pe.waveform,
                        n_steps, drive_model, amplitude_kind=pe.amplitude_kind))
                    material_drive.append(None)
            for cell, component, waveform, sigma_port, unit_field in drives.get(index, ()):
                if overridden:
                    queue.append((*cell, component, current_source_samples(grid, waveform, n_steps)))
                    material_drive.append((None, 0., sigma_port, unit_field))
                else:
                    queue.defer(_port_drive_source, grid, cell, component, waveform, n_steps,
                        sigma_port=sigma_port, unit_field=unit_field)
                    material_drive.append(None)
        if ports_present and not overridden:
            from rfx.model.materials import with_components
            final_drive_model = with_components(materials, grid, periodic=(False,) * 3,
                debye_spec=debye_spec, lorentz_spec=lorentz_spec)
        else:
            final_drive_model = materials
        sources = [SourceSpec(int(i), int(j), int(k), component, jnp.asarray(waveform))
                   for i, j, k, component, waveform in queue.resolve(final_drive_model)]
    if ports_present:
        conductors, pec_occupancy_override = conductor_model.prepare_port_occupancy(
            conductors, (), pec_occupancy_override, None, sim=sim)
    conductors, record = conductor_model.at_kernel(sim, conductors,
        lane=lane, pec_edges=conductors.pec_edges)
    return materials, cells, conductors.pec_edges, record, sources, tuple(material_drive), ports.port_specs, pec_occupancy_override


def refuse_unsupported_distributed_features(sim, **kwargs):
    """Reuse common refusals with the graded lane's passive-port admission.

    This validation-only view retains every port's geometry and load. The
    original declaration, including excite=False, goes to model assembly.
    Uniform callers use the original guard without a view.
    """
    from copy import copy
    from dataclasses import replace
    from rfx.runners.distributed_v2 import refuse_unsupported_distributed_features as refuse
    if sim._uses_nonuniform_mesh and any(p.impedance > 0 and not p.excite for p in sim._ports):
        view = copy(sim)
        view._ports = [replace(p, excite=True) if p.impedance > 0 and not p.excite else p
                       for p in sim._ports]
        refuse(view, **kwargs)
    else:
        refuse(sim, **kwargs)
