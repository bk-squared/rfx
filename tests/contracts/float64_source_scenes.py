"""Declarations and read-only observations for B-src's value rows."""
from contextlib import contextmanager
from collections import defaultdict
import sys

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, GaussianPulse, PolylineWire, Simulation, Sphere


POINTS = ((.0081, .0083, .0087), (.0162, .0141, .0153))


def scene(label, *, precision='float64'):
    options = dict(freq_max=15e9, domain=(.024,)*3, dx=.001,
                   boundary='cpml', cpml_layers=8, precision=precision)
    if label.startswith('G-'):
        options['dz_profile'] = np.r_[np.full(8, .001), np.full(10, .0008), np.full(8, .001)]
    if label == 'occupancy':
        options['pec_occupancy_model'] = 'kottke'
    if label in ('pin', 'port', 'floquet'):
        from rfx.grid import Grid
        options['dt'] = float(.5*Grid.courant_dt(.001, 3))
    sim = Simulation(**options)
    run = dict(n_steps=2)
    if label in ('U-tfsf', 'G-tfsf', 'closed', 'methodB'):
        source = dict(f0=10e9, bandwidth=.8, margin=4)
        if label == 'closed':
            source.update(closed_box=True, theta_deg=45, polarization="s", max_component=40)
        if label == 'methodB':
            source.update(angle_deg=30, method='methodB')
        sim.add_tfsf_source(**source)
    elif label == 'floquet':
        sim.add_floquet_port(.008, axis='z', scan_theta=0, polarization='te', f0=10e9, bandwidth=.8)
    elif label == 'port':
        sim.add_port((.010, .012, .012), 'ez', impedance=50,
                     waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        run['compute_s_params'] = False
    else:
        sim.add_source((.010, .012, .012), 'ez',
                       waveform=GaussianPulse(f0=10e9, bandwidth=.8), amplitude_kind='field')
    if label in ('U-stair', 'G-stair', 'inverse', 'occupancy', 'sphere'):
        sim.add_material('dielectric', eps_r=3.7)
        body = (PolylineWire(POINTS, radius=.0021) if label.endswith('stair')
                else Sphere((.0121, .0123, .0127), radius=.0061))
        sim.add(body, material='dielectric')
        if label != 'occupancy':
            run['subpixel_smoothing'] = 'kottke_pec' if label == 'inverse' else True
    if label in ('pin', 'port', 'floquet'):
        run['conformal_pec'] = 'cut_cell_floor'
    if label == 'decay':
        run = dict(until_decay=0, decay_max_steps=2, decay_min_steps=2, decay_check_interval=2)
    sim.add_probe((.015, .012, .012), 'ez')
    return sim, run


def waveguide(*, precision='float64', conformal=False):
    from rfx.boundaries.spec import Boundary
    transverse = Boundary('pec', 'pec', conformal=True) if conformal else 'pec'
    sim = Simulation(freq_max=12e9, domain=(.09, .04, .02), dx=.005,
                     boundary={'x': 'cpml', 'y': transverse, 'z': transverse},
                     cpml_layers=4, precision=precision)
    for position, direction, name in ((.015, '+x', 'left'), (.065, '-x', 'right')):
        sim.add_waveguide_port(position, direction=direction, n_modes=1,
                              freqs=jnp.array([8e9, 9e9], dtype=jnp.float32),
                              f0=8.5e9, bandwidth=.5, name=name)
    if conformal:
        sim.add(Sphere((.0403, .0207, .0109), radius=.0061), material='pec')
    else:
        sim.add_material('fill', eps_r=2.2)
        sim.add(Box((-.1,)*3, (.2,)*3), material='fill')
    sim.add_probe((.045, .02, .01), 'ez')
    return sim


@contextmanager
def observe():
    """Read returns/locals without changing any numerical implementation."""
    rows = defaultdict(list)
    names = {'init_tfsf', 'init_tfsf_methodB', 'init_plane_wave', '_cpml_profile',
             '_gather_table', '_homogeneous_reference_materials', '_to_jax',
             'compute_inv_eps_tensor_diag', 'select_occupancy_operator',
             'kottke_inv_eps_from_occupancy', 'uniform_source_table',
             'port_drive_waveform', 'make_msl_port_sources_jm', 'drive_columns',
             'drive_table', 'run', 'run_until_decay', 'occupancy_from_design',
             'topology_optimize', 'build_cell_table', 'union_at',
             'conformal_fractions_from_table', 'default_conformal_weights'}
    previous = sys.getprofile()
    samples, tables = {}, {}

    def profile(frame, event, value):
        name, local = frame.f_code.co_name, frame.f_locals
        if event == 'call' and name == 'build_cell_table':
            samples.clear()
        if event == 'return' and name == 'union_at':
            samples[tuple(local['half_axes'])] = np.asarray(value)
        if event == 'return' and name == 'build_cell_table' and 'sdf' in local:
            tables[id(value)] = dict(sdf=local['sdf'], widths=local['widths'],
                                     tol=local['tol'], midpoints=dict(samples),
                                     carried=tuple(shape for shape, _ in local['carried']),
                                     grid=local['grid'], lines=local['lines'],
                                     sample_lines=local['sample_lines'])
        if event == 'return' and name == 'conformal_fractions_from_table':
            rows['conformal_reference_inputs'].append((tables[id(local['table'])], local['table'], value))
        if event == 'return' and frame.f_code.co_name in names and frame.f_globals.get('__name__', '').startswith('rfx.'):
            rows[frame.f_code.co_name].append((dict(frame.f_locals), value))
    try:
        sys.setprofile(profile)
        yield rows
    finally:
        sys.setprofile(previous)


def capture(label):
    with observe() as rows:
        if label in ('waveguide', 'conformal-guide'):
            sim = waveguide(conformal=label == 'conformal-guide')
            result = sim.compute_waveguide_s_matrix(n_steps=2, normalize=True)
        else:
            sim, options = scene(label)
            if label == 'occupancy':
                grid = sim._build_realized_grid()
                occupancy = jnp.asarray(np.random.default_rng(2613).uniform(0, 1, grid.shape), dtype=jnp.float64)
                result = sim.forward(n_steps=2, checkpoint=False, pec_occupancy_override=occupancy)
            else:
                result = sim.run(**options)
        jax.block_until_ready(result)
    return rows
