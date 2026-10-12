"""Global single-device CPML is the independent oracle for slab emulation.

Base 7e6c7c7d0 layout: _distributed_common.split_array_x delegates to
rfx/stepping/slab.py:63-89: g=1, nx_per=ceil(Nx/D), pad_x=D*nx_per-Nx,
L=nx_per+2*g. The static x frame is (g,g+nx_per-pad_x) on every rank.
Uniform E ghosts are exchanged at distributed_v2.py:1298 (end of step),
before the next H CPML at :1219; H ghosts at :1225 precede E CPML :1241.
NU exchanges E at distributed_nu.py:2261, H at :2211, before CPML H :2197
and E :2230 respectively. Only tangential E/H components cross x seams: E receives right ghosts and
H receives left ghosts (_distributed_common.py:404-450).

Face/wall selection: simulation.py:1244 resolves walls; :1413 initializes
profiles with the declared walls. make_core_step (:1954) composes uniform.py;
uniform.py:166,295 calls CPML with axes and the default ALL_FACES (no faces
keyword). PEC and zero-depth profiles are identity. This test follows that
selection, including identity-profile rows, without imposing extra walls.
Materials use model.materials.with_components; electric coefficient helpers
are those used by stepping/uniform.py:299 and stepping/absorber.py:20-27.

Reference values come ONLY from H then E on global fields and global psi
initialized by init_cpml. Local placement/exchanges below do not produce
reference values. Fields are reset by the same deterministic seeded rule
before each step; psi recurs for 50 steps. Milestones 1/10/50 are recorded,
and all steps are checked. Identity psi rows start at +0 as required by R9.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries import cpml
from rfx.boundaries._cpml_slab import apply_ordered, store_psi
from rfx.boundaries.electric_coefficient import material_loss_operands, in_loop_loss, dispersive_curl
from rfx.core.yee import CurlBoundary, init_state, init_materials
from rfx.grid import Grid
from rfx.materials.debye import DebyePole, init_debye
from rfx.model.materials import with_components

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
CASES = [(d, case, medium) for d in (1, 2, 3)
         for case in ('asymmetric', 'pec', 'zero') for medium in ('lossy', 'debye')]
CASES += [(3, case, medium) for case in ('short_last', 'lo_touches_hi') for medium in ('lossy', 'debye')]
CASES += [(3, case, 'lossy') for case in ('one_row_uniform', 'one_row_graded')]


def _bytes(got, want, context):
    np.testing.assert_array_equal(np.asarray(got).view(np.uint32),
                                  np.asarray(want).view(np.uint32), err_msg=str(context))


def _cut(a, rank, per, *, edge=False):
    """Independent placement; sentinel rows make accidental reads observable."""
    a = np.asarray(a)
    indices = np.arange(rank * per - 1, (rank + 1) * per + 1)
    out = a[np.clip(indices, 0, a.shape[0]-1)].copy()
    if not edge:
        out[(indices < 0) | (indices >= a.shape[0])] = np.float32(.375)
    return jnp.asarray(out)


def _exchange(states, components, per, *, receive_left):
    """Copy only the live neighbor rows, using local states, not the oracle."""
    old = list(states)
    for rank, state in enumerate(old):
        changes = {}
        for name in components:
            a = getattr(state, name)
            if receive_left and rank:
                a = a.at[0].set(getattr(old[rank-1], name)[per])
            if not receive_left and rank + 1 < len(old):
                a = a.at[per+1].set(getattr(old[rank+1], name)[1])
            changes[name] = a
        states[rank] = state._replace(**changes)


@pytest.mark.parametrize('rank,lo,forward', [(0, True, True), (2, False, False)])
def test_frame_neighbor_operand_matches_global(rank, lo, forward):
    """R10: far window ends use physical neighbors, even at static frame ends.

    Nx=22, D=3 gives nx_per=8, pad_x=2 and frame=(1,7). A depth-six
    x-lo window ends at frame hi on rank 0; a depth-six x-hi window starts
    at frame lo on rank 2. E's forward and H's backward operands are read
    from global arrays by the single-device helper to form the oracle.
    _cut includes the current right E / left H ghost from that same field.
    """
    rng = np.random.default_rng(95210)
    global_field = jnp.asarray(rng.uniform(.25, 1.25, (22, 3, 4)), jnp.float32)
    local_field = _cut(global_field, rank, 8)
    frame = {"x": (1, 7)}
    with jax.disable_jit():
        expected = cpml.slab_neighbor(global_field, 0, 6, lo, forward=forward,
                                      frame=None)
        actual = cpml.slab_neighbor(local_field, 0, 6, lo, forward=forward,
                                    frame=frame["x"])
    _bytes(actual, expected, ('R10 operand', rank, 'x_lo' if lo else 'x_hi'))


def _scene(case, medium):
    nx, ny, nz = (22 if case in ('short_last', 'lo_touches_hi') else 23), 16, 20
    depths = dict(zip(cpml.ALL_FACES, (2, 3, 4, 5, 6, 7)))
    walls = frozenset({'y_hi'}) if case == 'pec' else frozenset()
    if walls:
        depths['y_hi'] = 0
    if case == 'zero':
        depths['x_hi'] = 0
    if case in ('short_last', 'lo_touches_hi'):
        depths.update(x_lo=8 if case == 'short_last' else 6, x_hi=2)
    budget = max(depths.values())
    shape = (nx, ny, nz)
    domain = tuple((size - 1 - depths[a+'_lo'] - depths[a+'_hi']) * .001
                   for size, a in zip(shape, 'xyz'))
    if case.startswith('one_row_'):
        from rfx import Simulation
        widths = dict(dz_profile=np.array([1., .9, 1.1, 1.]) * .001) if case.endswith('graded') else {}
        sim = Simulation(freq_max=15e9, domain=(.004,) * 3, dx=.001,
                         boundary='cpml', cpml_layers=1, **widths)
        grid = sim._build_nonuniform_grid() if widths else sim._build_grid()
        nx, ny, nz = shape = (7, 7, 7)
        depths = dict.fromkeys(cpml.ALL_FACES, 1)
    else:
        grid = Grid(freq_max=15e9, domain=domain, dx=.001, cpml_layers=budget,
                    face_layers=depths, pec_faces=walls)
    assert grid.shape == shape
    params, psi = cpml.init_cpml(grid, kappa_max=3.0)
    rng = np.random.default_rng(952)
    base = init_state(shape)._replace(**{
        name: jnp.asarray(rng.uniform(.25, 1.25, shape), jnp.float32) for name in FIELDS})
    inc = {name: jnp.asarray(rng.uniform(-.002, .002, shape), jnp.float32) for name in FIELDS}
    ix, iy, iz = np.indices(shape)
    block = (ix >= nx - max(depths['x_hi'], 1) - 2) & (iy <= depths['y_lo'] + 1) & (iz >= 5) & (iz < nz-5)
    raw = init_materials(shape)._replace(eps_r=jnp.where(block, 3., 1.),
               sigma=jnp.where(block, .02 if medium == 'lossy' else 0., 0.),
               mu_r=jnp.where(block, 1.7, 1.))
    poles = [DebyePole(delta_eps=1.3, tau=1e-11)]
    mats = with_components(raw, grid, periodic=(False,)*3,
             debye_spec=(poles, jnp.asarray(block)) if medium == 'debye' else None)
    debye = init_debye(poles, mats, float(grid.dt), jnp.asarray(block))[0] if medium == 'debye' else None
    loss = in_loop_loss(material_loss_operands(mats), grid.dt) if debye is None else None
    curl = dispersive_curl(debye, None, grid.dt)
    ek = dict(materials=mats, inv_eps_r_update=tuple(1/e for e in mats.components.eps_update),
              e_loss=loss, e_curl_coeff=curl, boundary=CurlBoundary(pec_faces=walls))
    return grid, params, psi, base, inc, mats, ek


@pytest.mark.parametrize('devices,case,medium', CASES)
def test_frame_owned_matches_global(devices, case, medium, monkeypatch, record_property):
    grid, params, initial_psi, base, inc, mats, ek = _scene(case, medium)
    per = (grid.nx + devices-1)//devices
    pad = devices*per-grid.nx
    frame = {"x": (1, 1+per-pad)}
    if case.startswith('one_row_'):
        # R12: the last frame owns one row; its backward operand must retain
        # the exchanged predecessor outside that frame. Global is the oracle.
        with jax.disable_jit():
            expected = cpml.slab_neighbor(base.hz, 0, 1, False, False, frame=None)
            actual = cpml.slab_neighbor(_cut(base.hz, devices-1, per), 0, 1,
                                        False, False, frame=frame["x"])
        _bytes(actual, expected, (case, 'R12 last-rank x_hi operand'))
    max_ulp = {}
    finite_counts = {}
    # Instrument raw psi and every pre-selection corrected slab. The oracle
    # is evaluated before installing these observers. No reference arithmetic
    # is reconstructed by the observers or placement/exchange helpers.
    for compiled in (False, True):
        with jax.disable_jit(not compiled):
            reference = base
            reference_psi = initial_psi
            locals_ = [base._replace(**{f: _cut(getattr(base,f),r,per) for f in FIELDS}) for r in range(devices)]
            local_base = list(locals_)
            local_inc = [{f:_cut(inc[f],r,per) for f in FIELDS} for r in range(devices)]
            psis = []
            local_eks, local_hks = [], []
            for rank in range(devices):
                psis.append(initial_psi._replace(**{name: (value if name.split('_')[-1][0]=='x'
                    else jnp.zeros((per+2,)+value.shape[1:],value.dtype))
                    for name,value in zip(initial_psi._fields,initial_psi)}))
                material = SimpleNamespace(eps_r=_cut(mats.eps_r,rank,per,edge=True),
                    sigma=_cut(mats.sigma,rank,per,edge=True))
                local_eks.append(dict(materials=material,
                    inv_eps_r_update=tuple(_cut(v,rank,per,edge=True) for v in ek['inv_eps_r_update']),
                    e_loss=None if ek['e_loss'] is None else tuple(_cut(v,rank,per,edge=True) for v in ek['e_loss']),
                    e_curl_coeff=None if ek['e_curl_coeff'] is None else tuple(_cut(v,rank,per,edge=True) for v in ek['e_curl_coeff']),
                    boundary=ek['boundary']))
                local_hks.append(dict(materials=SimpleNamespace(mu_r=material.eps_r,
                    components=SimpleNamespace(mu_update=tuple(_cut(v,rank,per,edge=True)
                        for v in mats.components.mu_update)))))
            def global_step(st, aux):
                h, hp = cpml.apply_cpml_h(st, params, aux, grid, materials=mats)
                return cpml.apply_cpml_e(h, params, hp, grid, **ek)
            global_call = jax.jit(global_step) if compiled else global_step
            local_calls = []
            for rank in range(devices):
                own = {'x_lo':jnp.asarray(rank == 0), 'x_hi':jnp.asarray(rank == devices-1)}
                def make_call(apply, kwargs, own):
                    def call(st, aux):
                        flags = []
                        def observed_psi(old, new, window, owner):
                            if owner is not None:
                                flags.append(jnp.all(jnp.isfinite(new)))
                            return store_psi(old, new, window, owner)
                        def observed_ordered(field, terms):
                            for term in terms:
                                sl,value=term[:2]
                                if len(term)==3 and term[2] is not None:
                                    flags.append(jnp.all(jnp.isfinite((field[sl]+value).astype(field.dtype))))
                                field=apply_ordered(field,[term])
                            return field
                        with monkeypatch.context() as patch:
                            patch.setattr(cpml,'store_psi',observed_psi)
                            patch.setattr(cpml,'apply_ordered',observed_ordered)
                            state, psi=apply(st,params,aux,grid,frame=frame,owned=own,**kwargs)
                        return state,psi,jnp.stack(flags)
                    return jax.jit(call) if compiled else call
                local_calls.append((make_call(cpml.apply_cpml_h,local_hks[rank],own),
                                    make_call(cpml.apply_cpml_e,local_eks[rank],own)))
            peaks = {f:0. for f in FIELDS}
            errors = {f:0. for f in FIELDS}
            count = 0
            for step in range(1,51):
                reference = base._replace(**{f:getattr(base,f)+step*inc[f] for f in FIELDS})
                for rank in range(devices):
                    locals_[rank] = local_base[rank]._replace(**{
                        f:getattr(local_base[rank],f)+step*local_inc[rank][f] for f in FIELDS})
                _exchange(locals_,('ey','ez'),per,receive_left=False)
                reference,reference_psi=global_call(reference,reference_psi)
                for rank in range(devices):
                    locals_[rank],psis[rank],flags=local_calls[rank][0](locals_[rank],psis[rank])
                    assert np.asarray(flags).all(), (rank,step,'H unselected branch')
                    count+=flags.size
                _exchange(locals_,('hy','hz'),per,receive_left=True)
                for rank in range(devices):
                    locals_[rank],psis[rank],flags=local_calls[rank][1](locals_[rank],psis[rank])
                    assert np.asarray(flags).all(), (rank,step,'E unselected branch')
                    count+=flags.size
                    start=rank*per
                    stop=min(start+per,grid.nx)
                    owned_rows=slice(1,1+stop-start)
                    for f in FIELDS:
                        a=np.asarray(getattr(locals_[rank],f)[owned_rows])
                        b=np.asarray(getattr(reference,f)[start:stop])
                        if not compiled:
                            _bytes(a,b,(case,medium,rank,step,f))
                        peaks[f]=max(peaks[f],float(np.max(np.abs(b))))
                        errors[f]=max(errors[f],float(np.max(np.abs(a.astype(np.float64)-b))))
                    for name in initial_psi._fields:
                        face=name.split('_')[-1]
                        if face[0]=='x':
                            if rank != (0 if face.endswith('lo') else devices-1):
                                _bytes(getattr(psis[rank],name),getattr(initial_psi,name),(rank,step,name,'nonowner'))
                                continue
                            a=getattr(psis[rank],name)
                            b=getattr(reference_psi,name)
                        else:
                            a=getattr(psis[rank],name)[owned_rows]
                            b=getattr(reference_psi,name)[start:stop]
                        if not compiled:
                            _bytes(a,b,(case,medium,rank,step,name))
                if step in (1,10,50):
                    record_property(f'{"jit" if compiled else "plain"}_step_{step}', 'completed')
            label='jit' if compiled else 'plain'
            max_ulp[label]={f:errors[f]/float(np.spacing(np.float32(peaks[f]))) for f in FIELDS}
            finite_counts[label]=int(count)
    record_property('field_peak_ulp',max_ulp)
    record_property('finite_branch_values_checked',finite_counts)
    print(f'{devices}/{case}/{medium}: ULP={max_ulp}, finite_checks={finite_counts}')
