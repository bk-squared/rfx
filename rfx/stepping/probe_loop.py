"""Adapt eager port-probe records and drives to the uniform step builder."""
from __future__ import annotations

import jax.numpy as jnp

from rfx.core.drives import StepDrives, drive_layout
from .sequence import HookPoint


def make_probe_step(grid, materials, ports, driven, *, use_cpml,
                    cpml_params, cpml_axes, debye, lorentz, pec_edge_masks,
                    curl_boundary, wire=False):
    """Keep eager port source arithmetic and probe objects around a uniform step."""
    from rfx.simulation import _StepContext, make_core_step
    from rfx.boundaries.cpml import apply_cpml_h, apply_cpml_e
    from rfx.probes import probes as observers
    from rfx.sources.sources import _wire_port_live_cells
    from rfx.sources.port_drive import stamped_drive, port_drive_waveform

    port = ports[driven]
    if wire:
        cells, live, _ = _wire_port_live_cells(grid, port, pec_edge_masks)
        cells = [cell for cell, enabled in zip(cells, live) if enabled]
        counts = [_wire_port_live_cells(grid, p, pec_edge_masks)[2] for p in ports]
    else:
        cells = [grid.position_to_index(port.position)]
    layout = drive_layout([(*cell, port.component) for cell in cells], jnp.float32)

    def source_values(t):
        return jnp.stack([port_drive_waveform(
            grid, cell, port.component, port.excitation, None, materials,
            sigma_port=stamped_drive(port, cell)[0],
            unit_field=stamped_drive(port, cell)[1], time=t) for cell in cells])

    def before_sources(frame):
        # Probe updates belong to this frame, never to the caller's input dict.
        frame.carry = dict(frame.carry)
        update = (observers.update_wire_drive_ref_probe if wire
                  else observers.update_lumped_drive_ref_probe)
        kwargs = {"pec_edge_masks": pec_edge_masks} if wire else {}
        frame.carry["sprobes"] = tuple(update(probe, frame.st, grid, p, grid.dt, **kwargs)
                                       for probe, p in zip(frame.carry["sprobes"], ports))

    def after_sources(frame):
        update = observers.update_wire_sparam_probe if wire else observers.update_sparam_probe
        frame.carry["sprobes"] = tuple(update(
            probe, frame.st, grid, p, grid.dt,
            **({"n_live": counts[i], "pec_edge_masks": pec_edge_masks} if wire else {}))
            for i, (probe, p) in enumerate(zip(frame.carry["sprobes"], ports)))

    def step_end(frame):
        frame.extras["sprobes"] = frame.carry["sprobes"]

    ctx = _StepContext(
        grid=grid, materials=materials, dt=grid.dt, dx=float(grid.cells('x')[0]),
        periodic=(False, False, False), pec_axes="xyz", stencil_order=2,
        use_fast_he=False, use_upml=False, use_cpml=use_cpml, use_tfsf=False,
        use_debye=debye is not None, use_lorentz=lorentz is not None,
        use_ntff=False, use_dft_planes=False, use_flux_monitors=False,
        use_waveguide_ports=False, use_pec_faces=True, use_pmc_faces=False,
        use_aniso_inv=False, aniso_inv_eps_smooth=False,
        use_pec_edges=pec_edge_masks is not None, use_pec_occupancy=False,
        use_conformal=False, use_wire_sparams=False, use_lumped_sparams=False,
        use_lumped_rlc=False, use_kerr=False, use_snapshot=False,
        use_monitor=False, use_flux_window=False,
        drives=StepDrives(layout, None), pec_edge_masks=pec_edge_masks,
        pec_faces_frozen=frozenset(f"{a}_{s}" for a in "xyz" for s in ("lo", "hi")),
        cpml_params=cpml_params, cpml_axes=cpml_axes,
        apply_cpml_h=apply_cpml_h, apply_cpml_e=apply_cpml_e,
        debye_coeffs=debye[0] if debye is not None else None,
        lorentz_coeffs=lorentz[0] if lorentz is not None else None,
        curl_boundary=curl_boundary,
    )
    return make_core_step(ctx, hooks={
        HookPoint.BEFORE_SOURCES: (before_sources,),
        HookPoint.AFTER_SOURCES: (after_sources,),
        HookPoint.STEP_END: (step_end,),
    }), source_values
