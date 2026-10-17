"""Adapt the graded distributed entry to per-slab setup and stepping."""
import jax
import numpy as np
from jax.sharding import Mesh

from rfx.nonuniform import NonUniformGrid
from rfx.stepping.rank import mesh_ranks
from rfx.stepping.slab import Slab
from rfx.stepping.slab_setup import build_setup, finish as split
from rfx.stepping.slab_wrap import wrap, reduce, run_steps


def run_wrapped(sg, materials, pec_mask, n_steps, sources, probes, *,
                devices, occupancy, checkpoint_every, n_warmup,
                emit_time_series, gather_final_state, material_drive,
                debye=None, lorentz=None, cpml=False, cpml_state=None):
    """Prepare local invariants and call the stepping-owned scan driver."""
    devices = jax.devices()[:sg.n_devices] if devices is None else devices
    if len(devices) != sg.n_devices:
        raise ValueError(f'need {sg.n_devices} devices, got {len(devices)}')
    mesh = Mesh(np.asarray(devices), ('x',))
    layout = Slab.from_grid(sg)
    grid = sg.global_grid or NonUniformGrid(
        sg.nx, sg.ny, sg.nz, float(sg.dx_padded[0]), float(sg.dy_cells[0]),
        np.asarray(sg.dx_padded[:sg.nx]), np.asarray(sg.dy_cells), np.asarray(sg.dz_cells),
        sg.dt, 0, np.asarray(sg.inv_dx_global[:sg.nx]), sg.inv_dy, sg.inv_dz,
        np.asarray(sg.inv_dx_h_global[:sg.nx]), sg.inv_dy_h, sg.inv_dz_h)
    setup = build_setup(grid, materials, layout, mesh, n_steps, sources, probes,
        pec_mask=pec_mask, occupancy=occupancy, debye=debye, lorentz=lorentz,
        cpml=cpml, cpml_state=cpml_state, emit_time_series=emit_time_series,
        material_drive=material_drive)
    invariants, carry = split(setup.invariants, setup.carry_init, layout, mesh,
                               waveforms=setup.src_waveforms, dt=grid.dt)
    waves = invariants['drives'].waves
    invariants['drives'] = invariants['drives']._replace(waves=None)
    advance = wrap(setup.step_fn_inv, layout, mesh, table=setup.invariants['probe_table'])
    from rfx.runners import distributed_nu as nu
    final, records = run_steps(carry, invariants, waves, mesh_ranks(mesh), advance,
        n_steps=n_steps, n_warmup=n_warmup, checkpoint_every=checkpoint_every,
        compile_loop=nu.jax.jit, scan_loop=nu.lax.scan)
    ts, state = reduce(records, final['fdtd'], layout, gather=gather_final_state)
    cpml_state = None
    if cpml:
        from rfx.stepping.slab_cpml import export_state
        cpml_state = export_state(final['cpml'], setup.step_fn_inv.context.cpml_grid, layout,
                                  native=True)
    return dict(time_series=ts if emit_time_series else None, final_state=state,
                final_state_sharded=final['fdtd'], cpml_state_sharded=cpml_state,
                debye_state_sharded=final.get('debye'), lorentz_state_sharded=final.get('lorentz'))
