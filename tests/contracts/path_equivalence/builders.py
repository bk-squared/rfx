"""Explicit minimal declarations on one asymmetric CPML dielectric box."""
import numpy as np
import jax.numpy as jnp

from rfx import Box, DebyePole, Simulation, PolylineWire, drude_pole, lorentz_pole
from rfx.boundaries.spec import Boundary, BoundarySpec

DX = 1 / 1024
DOMAIN = tuple(DX * v for v in (10.3, 7.2, 6.4))
FREQS = np.array([3e9, 5e9], dtype=np.float32)


def point(*values):
    return tuple(DX * v for v in values)


def waveform(t):
    return jnp.cos(t * 2e10)


def boundary():
    return BoundarySpec(x=Boundary(lo='cpml', hi='cpml', lo_thickness=1, hi_thickness=2),
                        y='cpml', z='cpml')


def _nothing(sim):
    return None


def _material(**params):
    def add(sim):
        sim.add_material('feature', **params)
        sim.add(Box(point(3.2, 2.1, 1.3), point(6.4, 4.3, 3.2)), material='feature')
    return {}, add


def _add(method, *args, **kwargs):
    return {}, lambda sim: getattr(sim, method)(*args, **kwargs)


def _microstrip(sim):
    # Minimal physical prerequisites of an MSL declaration: both longitudinal
    # conductor surfaces exist at its declared ground and trace heights.
    sim.add(Box(point(1, 1, 1), point(9, 6, 1)), material='pec')
    sim.add(Box(point(1, 2, 2), point(9, 4, 2)), material='pec')
    sim.add_msl_port(position=point(3, 3, 1), width=2*DX, height=DX,
                     direction='+x', impedance=50., waveform=waveform)


# A row is deliberately explicit: extending admission cannot inherit a no-op.
BUILDERS = {
    ('_freq_max', ''): ({}, _nothing),
    ('_domain', ''): ({}, _nothing),
    ('_dx', ''): ({}, _nothing),
    ('_dt_pin', ''): ({}, _nothing),
    ('_dt_min_cell', ''): ({'dt_min_cell': DX}, _nothing),
    ('_precision', ''): ({'precision': 'mixed'}, _nothing),
    ('_solver', ''): ({'solver': 'adi', 'boundary': 'pec'}, _nothing),
    ('_adi_cfl_factor', ''): ({'adi_cfl_factor': 1.1}, _nothing),
    ('_stencil_order', ''): ({'stencil_order': 4}, _nothing),
    ('_mode', ''): ({'mode': '2d_tmz'}, _nothing),
    ('_materials', 'eps'): ({}, _nothing),
    ('_materials', 'sigma'): _material(eps_r=2.5, sigma=0.5),
    ('_materials', 'mu'): _material(eps_r=2.5, mu_r=2.0),
    ('_materials', 'debye'): _material(eps_r=2.5, debye_poles=[DebyePole(0.5, 1e-11)]),
    ('_materials', 'lorentz'): _material(eps_r=2.5, lorentz_poles=[lorentz_pole(0.5, 2*np.pi*5e9, 1e9)]),
    ('_materials', 'drude'): _material(eps_r=2.5, lorentz_poles=[drude_pole(2*np.pi*5e9, 1e9)]),
    ('_materials', 'kerr'): _material(eps_r=2.5, chi3=1e-3),
    ('_geometry', 'pec_volume'): _add('add', Box(point(5.2, 2.1, 2.2), point(6.3, 4.2, 3.3)), material='pec'),
    ('_geometry', 'pec_sheet'): _add('add', Box(point(5, 2, 1), point(5, 4, 4)), material='pec'),
    ('_geometry', 'pec_wire'): _add('add', PolylineWire((point(5, 3, 1), point(5, 3, 4)), radius=0), material='pec'),
    ('_thin_conductors', 'lossy_sheet'): _add('add_thin_conductor', Box(point(5, 2, 1), point(5, 4, 4)), sigma_bulk=1e3, thickness=DX/10),
    ('_thin_conductors', 'pec_sheet'): _add('add_thin_conductor', Box(point(5, 2, 1), point(5, 4, 4))),
    ('_thin_conductors', 'surface_impedance'): _add('add_thin_conductor', Box(point(5, 2, 1), point(5, 4, 4)), sigma_bulk=1e3, thickness=DX/10, surface_impedance_f0=5e9),
    ('_pinned_sheets', 'pec_sheet'): _add('add_pinned_sheet', plane_index=6, i_range=(3, 5), j_range=(3, 6), normal_axis=0),
    ('_ports', 'source'): ({}, _nothing),
    ('_ports', 'amplitude_kind'): ({}, _nothing),
    ('_ports', 'lumped_port'): _add('add_port', point(5, 3, 3), 'ez', impedance=50., waveform=waveform),
    ('_ports', 'passive_port'): _add('add_port', point(5, 3, 3), 'ez', impedance=50., excite=False),
    ('_ports', 'wire_port'): _add('add_port', point(5, 3, 2), 'ez', impedance=50., extent=2*DX, waveform=waveform),
    ('_msl_ports', 'msl_port'): ({}, _microstrip),
    ('_waveguide_ports', 'waveguide_port'): _add('add_waveguide_port', 3*DX, direction='+x', mode=(1, 0), mode_type='TE', freqs=FREQS, f0=5e9, bandwidth=0.5, probe_offset=2, ref_offset=1),
    ('_coaxial_ports', 'coax_port'): _add('add_coaxial_port', point(5, 3, 0), face='bottom', pin_length=3*DX),
    ('_floquet_ports', 'floquet_port'): _add('add_floquet_port', 3*DX, axis='z', f0=5e9),
    ('_floquet_ports', 'scan_angle'): _add('add_floquet_port', 3*DX, axis='z', f0=5e9, scan_theta=30.),
    ('_lumped_rlc', 'R'): _add('add_lumped_rlc', point(5, 3, 3), 'ez', R=10.),
    ('_lumped_rlc', 'series_RL'): _add('add_lumped_rlc', point(5, 3, 3), 'ez', R=10., L=1e-9),
    ('_tfsf', 'plane_wave'): _add('add_tfsf_source', f0=5e9, bandwidth=0.5, margin=1),
    ('_refinement', 'slab'): _add('add_refinement', z_range=(0., 4*DX), ratio=2),
    ('_refinement', 'relaxed_validation'): _add('add_refinement', z_range=(0., 4*DX), ratio=2, validation='research'),
    ('_boundary', 'cpml'): ({}, _nothing),
    ('_boundary', 'upml'): ({'boundary': BoundarySpec(x=Boundary(lo='upml', hi='upml', lo_thickness=1, hi_thickness=2), y='upml', z='upml')}, _nothing),
    ('_pec_faces', 'pec_face'): ({'boundary': BoundarySpec(x=Boundary(lo='pec', hi='cpml'), y='cpml', z='cpml')}, _nothing),
    ('_boundary_spec', 'pmc_face'): ({'boundary': BoundarySpec(x=Boundary(lo='pmc', hi='cpml'), y='cpml', z='cpml')}, _nothing),
    ('_boundary_spec', 'conformal'): ({'boundary': BoundarySpec(x=Boundary(lo='pec', hi='pec', conformal=True), y='cpml', z='cpml')}, _nothing),
    ('_boundary_spec', 'conformal_s_matrix'): ({'boundary': BoundarySpec(x=Boundary(lo='pec', hi='pec', conformal=True), y='cpml', z='cpml')}, _nothing),
    ('_boundary_spec', 'absorbing_lid'): ({'boundary': BoundarySpec(x='pec', y='pec', z=Boundary(lo='pec', hi='cpml'))}, _nothing),
    ('_periodic_axes', 'periodic'): ({'boundary': BoundarySpec(x='periodic', y='cpml', z=Boundary(lo='pec', hi='cpml')), 'domain': point(11, 7.2, 6.4)}, _nothing),
    ('_cpml_layers', 'layers'): ({}, _nothing),
    ('_cpml_kappa_max', 'kappa'): ({'cpml_kappa_max': 3.0}, _nothing),
    ('_interface_eps', 'dual_average'): ({'interface_eps': 'dual_average'}, _nothing),
    ('_dx_profile', 'graded'): ({}, _nothing),
    ('_dy_profile', 'graded'): ({}, _nothing),
    ('_dz_profile', 'graded'): ({}, _nothing),
    ('_probes', 'probe'): ({}, _nothing),
    ('_dft_planes', 'dft_plane'): _add('add_dft_plane_probe', axis='x', coordinate=4*DX, freqs=FREQS),
    ('_flux_monitors', 'flux'): _add('add_flux_monitor', axis='x', coordinate=4*DX, freqs=FREQS),
    ('_ntff', 'ntff_box'): _add('add_ntff_box', point(1, 1, 1), point(9, 6, 5), freqs=FREQS),
    ('_current_moments', 'block_moments'): _add('add_current_moment_monitor', point(1, 1, 1), point(9, 6, 5), block_size=2*DX, freqs=FREQS),
}

BASE_ROWS = {('_freq_max', ''), ('_domain', ''), ('_dx', ''), ('_materials', 'eps'),
             ('_ports', 'source'), ('_boundary', 'cpml'), ('_cpml_layers', 'layers'), ('_probes', 'probe'),
             ('_dx_profile', 'graded'), ('_dy_profile', 'graded'), ('_dz_profile', 'graded')}


def build(row, lane, *, graded=False, dt=None):
    ctor, add = BUILDERS[row]
    kwargs = dict(freq_max=15e9, domain=DOMAIN, dx=DX, boundary=boundary(), cpml_layers=2)
    kwargs.update(ctor)
    if row[0] == '_waveguide_ports':
        # Modal aperture dimensions are declarations, not inferred rounded
        # box lengths. Put both transverse walls exactly on grid nodes.
        kwargs['domain'] = point(10.3, 8, 7)
    nu = lane in ('run_nonuniform', 'fwd_nonuniform') or graded
    if nu or row[0] in ('_dx_profile', '_dy_profile', '_dz_profile'):
        for axis, length in zip('xyz', kwargs['domain']):
            profile = np.full(int(np.ceil(length / DX)), DX)
            if graded:
                profile[3:5] = (DX*0.9, DX*1.1)
            kwargs[f'd{axis}_profile'] = profile
        if dt is not None:
            kwargs['dt'] = dt
    if row[0] in ('_dt_pin', '_dt_min_cell') and 'dt' not in kwargs:
        kwargs['dt'] = dt if dt is not None else 1e-12
    if row[0] == '_floquet_ports':
        kwargs['domain'] = point(11, 8, 6.4)
        kwargs['boundary'] = BoundarySpec(x='periodic', y='periodic', z=Boundary(lo='pec', hi='cpml'))
    if row[0] == '_waveguide_ports':
        # State the guide walls explicitly: uniform otherwise supplies its
        # own transverse PEC layout, while NU keeps the declared absorbers.
        kwargs['boundary'] = BoundarySpec(
            x=Boundary(lo='cpml', hi='cpml', lo_thickness=1, hi_thickness=2),
            y='pec', z='pec')
    sim = Simulation(**kwargs)
    sim.add_material('block', eps_r=2.5)
    sim.add(Box(point(3.2, 2.1, 1.3), point(6.4, 4.3, 3.2)), material='block')
    # add_source is a zero-impedance _PortEntry internally. Driven port,
    # waveguide and TFSF declarations provide their own excitation and must
    # not also carry that entry (their public APIs reject the combination).
    own_drive = row in (('_ports', 'lumped_port'), ('_ports', 'wire_port'),
                       ('_boundary_spec', 'conformal_s_matrix')) or row[0] in (
        '_tfsf', '_waveguide_ports', '_msl_ports', '_floquet_ports', '_coaxial_ports')
    if not own_drive:
        sim.add_source(point(2.1, 2.3, 2.2), 'ez', amplitude_kind='current' if row == ('_ports', 'amplitude_kind') else 'field', waveform=waveform)
    for p in ((4.1, 2.3, 2.2), (7.1, 3.2, 2.4)):
        sim.add_probe(point(*p), 'ez')
    add(sim)
    if row == ('_refinement', 'relaxed_validation'):
        _material(eps_r=2.5, debye_poles=[DebyePole(0.5, 1e-11)])[1](sim)
    if row == ('_boundary_spec', 'conformal_s_matrix'):
        sim.add_port(point(5, 3, 3), 'ez', impedance=50., waveform=waveform)
    return sim
