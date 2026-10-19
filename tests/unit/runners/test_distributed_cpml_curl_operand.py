"""Scene T: inspect the actual CPML E call before any field step runs."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.stepping import graded_loop
from rfx.boundaries import cpml
from rfx.runners import _distributed_nu_wrapped as runner
from rfx.stepping import graded
from rfx.stepping.sequence import Kernel
from tests._distributed_dispersive_cpml import record, scene

pytestmark = pytest.mark.distributed


class BeforeStepping(Exception):
    pass


def boundary_kernel(ctx, monkeypatch):
    """Expose the production boundary operation, leaving all arithmetic intact."""
    compose, saved = graded.compose, {}
    def capture(kernels, hooks):
        saved['boundary'] = kernels[Kernel.E_BOUNDARY]
        return compose(kernels, hooks)
    with monkeypatch.context() as patch:
        patch.setattr(graded, 'compose', capture)
        graded.make_graded_step(ctx)
    return saved['boundary']


def frame(invariants, carry):
    return graded.GradedFrame(carry, (jnp.int32(0), jnp.zeros(0)), invariants,
        st=carry['fdtd'], cpml_new=carry['cpml'], materials=invariants['materials'],
        debye_coeffs=invariants['debye_coeffs'], lorentz_coeffs=invariants['lorentz_coeffs'],
        _cpml_inv_eps_r=invariants['cpml_inv_eps_r'])


@pytest.mark.parametrize('kind', ['debye', 'mixed', 'lorentz'])
@pytest.mark.parametrize('devices', [1, 2])
def test_wrapped_cpml_receives_one_device_curl(kind, devices, monkeypatch):
    sim = scene(kind, cells=False)
    grid = sim._build_nonuniform_grid()
    drawn = np.asarray(sim._assemble_materials_nu(grid)[0].eps_r)
    assert np.all(drawn[:, grid.ny//2, grid.nz//2] == 4.), 'Scene T must fill all x cells'
    canonical, staged = {}, {}
    build = graded_loop._build_nu_scan
    def capture_one(*args, **kwargs):
        canonical['setup'] = build(*args, **kwargs)
        raise BeforeStepping
    with monkeypatch.context() as patch:
        patch.setattr(graded_loop, '_build_nu_scan', capture_one)
        with pytest.raises(BeforeStepping):
            sim.forward(n_steps=2, checkpoint=False, skip_preflight=True)
    one = canonical['setup']
    expected = {}
    def inspect_one(state, profiles, psi, grid, axes='xyz', **kwargs):
        # This is what apply_cpml_e_step actually passes to its kernel.
        expected['curl'] = (None if kwargs['e_curl_coeff'] is None else
                            tuple(np.asarray(a) for a in kwargs['e_curl_coeff']))
        raise BeforeStepping
    with monkeypatch.context() as patch:
        patch.setattr(cpml, 'apply_cpml_e', inspect_one)
        with pytest.raises(BeforeStepping):
            boundary_kernel(one.step_fn_inv.context, patch)(frame(one.invariants, one.carry_init))

    build_local, finish = runner.build_setup, runner.split
    def capture_local(*args, **kwargs):
        setup = build_local(*args, **kwargs)
        staged['ctx'] = setup.step_fn_inv.context
        return setup
    def capture_slabs(*args, **kwargs):
        staged['invariants'], staged['carry'] = finish(*args, **kwargs)
        staged['layout'] = args[2]
        raise BeforeStepping
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'build_setup', capture_local)
        patch.setattr(runner, 'split', capture_slabs)
        with pytest.raises(BeforeStepping):
            sim.forward(n_steps=2, checkpoint=False, skip_preflight=True, distributed=True,
                        devices=jax.devices('cpu')[:devices])

    layout = staged['layout']
    ctx = replace(staged['ctx'], x_face_frame={'x': (
        layout.ghost_width, layout.ghost_width + layout.nx_per_rank - layout.pad_x)})
    boundary = boundary_kernel(ctx, monkeypatch)
    compared = []
    for rank in range(devices):
        def local(a):
            if not hasattr(a, 'ndim') or a.ndim == 0:
                return a
            return next(s.data for s in a.addressable_shards
                        if s.device == jax.devices('cpu')[rank])
        inv = jax.tree.map(local, {key: staged['invariants'][key] for key in (
            'materials', 'debye_coeffs', 'lorentz_coeffs', 'cpml_inv_eps_r', 'e_loss', 'cpml_x', 'owns_x')})
        inv['owns_x'] = jax.tree.map(lambda a: a[0], inv['owns_x'])
        if inv['e_loss'] is not None:
            inv['e_loss'] = (*inv['e_loss'], False)
        carry = jax.tree.map(local, {key: staged['carry'][key] for key in ('fdtd', 'cpml')})
        def inspect_slab(state, profiles, psi, metadata, **kwargs):
            got, want = kwargs['e_curl_coeff'], expected['curl']
            for face in sorted(kwargs['faces']):
                if not bool(kwargs['owned'].get(face, True)):
                    continue
                axis = 'xyz'.index(face[0])
                depth = getattr(grid, 'pad_' + face)
                window = [slice(None)] * 3
                lo, hi = kwargs['frame'].get(face[0], (0, state.ex.shape[axis]))
                window[axis] = slice(lo, lo + depth) if face.endswith('lo') else slice(hi - depth, hi)
                x0, x1, _ = window[0].indices(state.ex.shape[0])
                x0 = max(x0, layout.ghost_width)
                x1 = min(x1, layout.ghost_width + min(layout.nx_per_rank, layout.nx-rank*layout.nx_per_rank))
                assert x1 > x0, (face, rank, 'empty owned window')
                window[0] = slice(x0, x1)
                whole = list(window)
                whole[0] = slice(rank*layout.nx_per_rank+x0-layout.ghost_width,
                                 rank*layout.nx_per_rank+x1-layout.ghost_width)
                for component in range(3):
                    if component == axis:
                        continue
                    row = dict(entry='curl_operand', kind=kind, devices=devices, rank=rank,
                               face=face, component=('ex', 'ey', 'ez')[component])
                    if want is None:
                        record(dict(row, expected=None, actual=None if got is None else 'array'))
                        assert got is None, f"e_curl_coeff: Lorentz expects None: {row}"
                    else:
                        reference = want[component][tuple(whole)]
                        row.update(expected_min=float(reference.min()), expected_max=float(reference.max()),
                                   shape=list(reference.shape))
                        if got is None:
                            record(dict(row, actual=None))
                        assert got is not None, f"e_curl_coeff missing: actual=None, {row}"
                        actual = np.asarray(got[component])[tuple(window)]
                        delta = float(np.abs(actual.astype(np.float64)-reference).max())
                        record(dict(row, actual_min=float(actual.min()), actual_max=float(actual.max()), max_abs=delta))
                        assert actual.dtype == reference.dtype and actual.shape == reference.shape, row
                        assert actual.tobytes() == reference.tobytes(), f"e_curl_coeff bytes: max_abs={delta}, {row}"
                    compared.append((rank, face, component))
            raise BeforeStepping
        with monkeypatch.context() as patch:
            patch.setattr(cpml, 'apply_cpml_e', inspect_slab)
            with pytest.raises(BeforeStepping):
                boundary(frame(inv, carry))
    assert len(compared) == (12 if devices == 1 else 20)
