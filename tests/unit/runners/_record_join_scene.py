"""Public record layouts, with deterministic carry data independent of stepping."""
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rfx import Simulation, SnapshotSpec
from rfx.runners._distributed_records import SlabRecords
from tests._multi_device_scene import host_devices

KINDS = ('planes', 'flux', 'ntff', 'moments', 'snapshots', 'time_planes')


def layout_for(kind, n=2, *, size=1, nx=17, single_frequency=False):
    # Same CPML/source construction as test_distributed_record_planes.model;
    # enough transverse cells to measure the second-axis chunk split directly.
    # Four CPML cells and the endpoint node are included in the requested nx.
    sim = Simulation(freq_max=15e9, domain=((nx - 5)*1e-3, 12e-3, 80e-3), dx=1e-3,
                     boundary='cpml', cpml_layers=2, precision='float32')
    sim.add_source((5e-3, 5e-3, 4e-3), 'ez', amplitude_kind='field')
    grid = sim._build_grid()
    assert grid.shape[0] == nx
    freqs = np.linspace(3e9, 7e9, 1 if single_frequency else 40*size)
    if kind in ('planes', 'time_planes'):
        sim.add_dft_plane_probe(axis='y', coordinate=4e-3, component='ez', freqs=freqs,
                                region=(2, nx - 2, 1, 79), name='record')
        if kind == 'time_planes':
            sim._time_recorded_dft_planes = {'record'}
    elif kind == 'flux':
        sim.add_flux_monitor(axis='y', coordinate=4e-3, freqs=freqs, name='record')
    elif kind == 'ntff':
        sim.add_ntff_box((3e-3, 2e-3, 2e-3), ((nx - 7)*1e-3, 10e-3, 78e-3), freqs=freqs)
    elif kind == 'moments':
        sim.add_current_moment_monitor((3e-3, 2e-3, 2e-3), ((nx - 7)*1e-3, 10e-3, 6e-3),
                                       block_size=2e-3, freqs=freqs)
    spec = SnapshotSpec(interval=1, components=('ex', 'ez', 'hy')) if kind == 'snapshots' else None
    mesh = Mesh(np.asarray(host_devices(n)), ('x',))
    layouts = SlabRecords(sim, grid, (nx + n - 1)//n, mesh, np.float32, 40*size, spec)
    return layouts.layouts[kind]


def filled(layout, *, zeros=False):
    """Negative zeros in both complex components, padded tails, and finite noise."""
    rng = np.random.default_rng(2608)
    def fill(a):
        host = np.empty(a.shape, a.dtype)
        if np.iscomplexobj(host):
            host.real = rng.normal(size=a.shape).astype(np.float32)
            host.imag = rng.normal(size=a.shape).astype(np.float32)
            host.real.reshape(-1)[::7] = -0.0
            host.imag.reshape(-1)[::11] = -0.0
        else:
            host[:] = rng.normal(size=a.shape).astype(a.dtype)
            host.reshape(-1)[::7] = -0.0
        if zeros:
            host.real[...] = -0.0
            if np.iscomplexobj(host):
                host.imag[...] = -0.0
        return jax.device_put(host, NamedSharding(layout.mesh, P('x')))
    return jax.tree.map(fill, layout.initial())
