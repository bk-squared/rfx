"""The asymmetric one-cell slab and off-plane pulse/probe for tracker 1589."""
import jax
import numpy as np
from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources import GaussianPulse
from rfx.core.yee import EPS_0
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import drude_pole


def geometry(face):
    """Permute x's reference scene; z uses a 12 x 24 x 24 box, always Ez."""
    permutation = {'x': (0, 1, 2), 'y': (1, 0, 2), 'z': (2, 1, 0)}[face[0]]
    def permute(values):
        return tuple(values[i] for i in permutation)
    return (permute((24, 24, 12)), permute((6, 5, 4)),
            permute((16, 15, 7)), permutation.index(1))


def build(owner='plain', face='x_lo', nu=False):
    dx=1e-3
    dimensions, source, probe, transverse = geometry(face)
    axis,side=face.split('_')
    faces={a:'pec' for a in 'xyz'}
    faces[axis]=Boundary(lo='cpml' if side=='lo' else 'pec', hi='cpml' if side=='hi' else 'pec', lo_thickness=6 if side=="lo" else None,hi_thickness=10 if side=="hi" else None)
    opts={f'd{a}_profile':np.full(n,dx) for a,n in zip('xyz',dimensions)} if nu else {}
    sim=Simulation(freq_max=20e9,domain=tuple(n*dx for n in dimensions),dx=dx,boundary=BoundarySpec(**faces),cpml_layers=10,**opts)
    dt=sim._build_realized_grid().dt
    props=dict(eps_r=4. if owner=='dielectric' else 2. if owner in ('debye','mixed') else 1.)
    if owner in ('plain','lorentz'): props['sigma']=4*EPS_0/dt
    if owner=='dielectric': props['sigma']=0.7*2*EPS_0/dt
    if owner in ('debye','mixed'): props['debye_poles']=[DebyePole(delta_eps=50.,tau=5e-12)]
    if owner in ('lorentz','mixed'): props['lorentz_poles']=[drude_pole(2*np.pi*50e9,1e10)]
    sim.add_material('slab',**props)
    lo=[0.,0.,0.];hi=[n*dx for n in dimensions]
    lo[transverse]=11*dx;hi[transverse]=12*dx
    # Pole masks keep declared occupancy; crossing the physical face alone
    # does not populate the pad. Extend the slab through the complete pad.
    normal = 'xyz'.index(axis)
    if side == 'lo':
        lo[normal] = -12 * dx
    else:
        hi[normal] += 12 * dx
    sim.add(Box(tuple(lo),tuple(hi)),material='slab')
    sim.add_source(position=tuple(p*dx for p in source),component='ez',waveform=GaussianPulse(f0=10e9,bandwidth=0.5),amplitude_kind='field')
    sim.add_probe(position=tuple(p*dx for p in probe),component='ez')
    grid = sim._build_realized_grid()
    assembled = sim._assemble_materials_nu(grid) if nu else sim._assemble_materials(grid)
    materials, debye_spec, lorentz_spec = assembled[:3]
    pad = [slice(None)] * 3
    pad[normal] = slice(0, 6) if side == 'lo' else slice(-10, None)
    pad = tuple(pad)
    if owner in ('plain', 'dielectric', 'lorentz'):
        assert np.any(np.asarray(materials.sigma)[pad] > 0), 'loss must reach the pad'
    for spec in (debye_spec, lorentz_spec):
        if spec is not None:
            masks = spec[1] if isinstance(spec[1], (tuple, list)) else [spec[1]]
            for mask in masks:
                occupied = np.asarray(mask)[pad]
                assert np.all(np.any(occupied, axis=tuple(i for i in range(3) if i != normal))), \
                    'each pole must occupy every layer of the tested pad'
    return sim


def probe_loop(sim, n_steps, face):
    """Exercise the eager extractor's step with the same point field drive."""
    import jax.numpy as jnp
    from rfx.core.yee import init_state
    from rfx.boundaries.cpml import init_cpml
    from rfx.materials.debye import init_debye
    from rfx.materials.lorentz import init_lorentz
    from rfx.probes.probes import init_sparam_probe, _port_curl_boundary
    from rfx.sources.sources import LumpedPort
    from rfx.stepping.probe_loop import make_probe_step
    from rfx._grid_metric import field_index

    grid = sim._build_grid()
    mats, ds, ls, *_ = sim._assemble_materials(grid)
    # Realize the same component operands the public run hands to the step.
    from rfx.model.materials import with_components
    mats = with_components(mats, grid, periodic=(False, False, False), debye_spec=ds, lorentz_spec=ls)
    db = init_debye(*ds[:1], mats, grid.dt, mask=ds[1]) if ds else None
    lr = init_lorentz(*ls[:1], mats, grid.dt, mask=ls[1]) if ls else None
    params, psi = init_cpml(grid)
    pulse = GaussianPulse(f0=10e9, bandwidth=0.5)
    _, source, probe, _ = geometry(face)
    port = LumpedPort(tuple(p*1e-3 for p in source), 'ez', 50., pulse)
    core, _ = make_probe_step(
        grid, mats, [port], 0, use_cpml=True, cpml_params=params,
        cpml_axes='xyz', debye=db, lorentz=lr, pec_edge_masks=None,
        curl_boundary=_port_curl_boundary(grid, (False,) * 3))
    carry = dict(fdtd=init_state(grid.shape), cpml=psi,
                 sprobes=(init_sparam_probe(grid, port, jnp.array([10e9]),
                                           dft_total_steps=n_steps),))
    if db:
        carry['debye'] = db[1]
    if lr:
        carry['lorentz'] = lr[1]
    cell = field_index(grid, tuple(p*1e-3 for p in probe), 'ez')

    # Eager SParamProbe carries a static component name; close it over the
    # scan while retaining every numerical leaf and the actual probe hooks.
    leaves, tree = jax.tree.flatten(carry)
    dynamic = [i for i, leaf in enumerate(leaves) if isinstance(leaf, (jax.Array, np.ndarray))]

    def unpack(values):
        full = list(leaves)
        for i, value in zip(dynamic, values):
            full[i] = value
        return jax.tree.unflatten(tree, full)

    def step(values, k):
        carry, _, extras = core(unpack(values), k, jnp.reshape(pulse(k * grid.dt), (1,)), jnp.zeros(0))
        carry["sprobes"] = extras["sprobes"]
        new_leaves = jax.tree.leaves(carry)
        return tuple(new_leaves[i] for i in dynamic), carry['fdtd'].ez[cell]

    initial = tuple(leaves[i] for i in dynamic)
    return jax.jit(lambda c: jax.lax.scan(step, c, jnp.arange(n_steps))[1])(initial)[:, None]


def solve(owner, face, lane, n_steps=6000):
    sim = build(owner, face, nu='graded' in lane or lane == 'distributed_forward')
    kwargs = dict(n_steps=n_steps, skip_preflight=True)
    if lane == 'vmap':
        from rfx.vmap_sweep import vmap_material_sweep
        eps = 4. if owner == 'dielectric' else 1.
        return np.asarray(vmap_material_sweep(sim, 'slab.eps_r', np.array([eps]),
                                    n_steps=n_steps).time_series)[0]
    if lane == 'probe_loop':
        return np.asarray(probe_loop(sim, n_steps, face))
    if 'distributed' in lane:
        assert len(jax.devices('cpu')) >= 2, 'requires two CPU devices'
        kwargs['devices'] = jax.devices('cpu')[:2]
        if 'forward' in lane:
            kwargs['distributed'] = True
    if 'forward' in lane:
        kwargs['checkpoint'] = False
    out = (sim.forward if 'forward' in lane else sim.run)(**kwargs)
    return np.asarray(out.time_series)
