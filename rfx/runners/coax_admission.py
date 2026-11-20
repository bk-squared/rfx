"""S1 admission for the uniform one-device coax model and drive."""
from rfx.errors import RfxUnsupported


def admit_coax_port(sim, lane, *, run_args=None, grid=None):
    if not sim._coaxial_ports:
        return
    args = run_args or {}
    if lane not in {'run_uniform', 'fwd_uniform'}:
        raise RfxUnsupported(f'{lane}: add_coaxial_port() is supported only on run_uniform and fwd_uniform in S1')
    gates = (
        (sim._precision != 'float32', "precision='float32'"),
        (sim._stencil_order != 2, 'stencil_order=2'),
        (sim._mode != '3d', "mode='3d'"),
        (bool(sim._periodic_axes), 'no periodic axis'),
        (sim._boundary != 'cpml' or sim._cpml_layers <= 0 or any(
            getattr(sim._boundary_spec, a).lo != 'cpml' or getattr(sim._boundary_spec, a).hi != 'cpml'
            for a in 'xyz'), 'CPML on all six faces'),
        (sim._tfsf is not None, 'no TF/SF'),
        (len(sim._coaxial_ports) != 1, 'exactly one coax port'),
        (any(p.face not in ('top', 'bottom') for p in sim._coaxial_ports), "face='top' or 'bottom'"),
        (bool(sim._msl_ports or sim._waveguide_ports or sim._floquet_ports
              or any(p.impedance > 0 for p in sim._ports)), 'no other S-port family'),
        (bool(args.get('conformal_pec') or sim._boundary_spec.conformal_faces()), 'conformal_pec is unsupported'),
        (bool(args.get('subpixel_smoothing')), 'subpixel_smoothing is unsupported'),
        (args.get('eps_override') is not None, 'eps_override is unsupported'),
        (args.get('sigma_override') is not None, 'sigma_override is unsupported'),
    )
    for failed, name in gates:
        if failed:
            raise RfxUnsupported(f'{lane}: add_coaxial_port() requires {name} (S1)')
    if grid is not None:
        from rfx.model.coax_line import coax_port_layout
        pads = tuple(getattr(grid, f'pad_{a}_{s}') for a in 'xyz' for s in ('lo', 'hi'))
        coax_port_layout(grid, sim._coaxial_ports[0], absorber_pads=pads, ntff=sim._ntff)


def entry_lane(sim, *, forward=False, distributed=False):
    prefix = 'fwd' if forward else 'run'
    if distributed:
        return prefix + ('_distributed_nu' if sim._uses_nonuniform_mesh else '_distributed')
    if sim._refinement is not None:
        return 'run_subgridded'
    return prefix + ('_nonuniform' if sim._uses_nonuniform_mesh else '_uniform')
