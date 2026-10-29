"""Shared operand witnesses for the kept Lorentz CPML tests."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.stepping import graded
from rfx.stepping.sequence import Kernel
from tests._multi_device_scene import host_devices


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


def _lorentz_scene(name):
    from rfx import Box
    from tests._distributed_dispersive_cpml import scene
    sim = scene('lorentz', cells=False, sigma=0. if name == 'lorentz' else .5)
    if name == 'lorentz_lossy_yz':
        sim.add(Box((.009, -1., -1.), (.015, 1., 1.)), material='m')
    return sim


def _loss_at_kernel(name, monkeypatch):
    """Pin the operands received by the real E boundary, before stepping."""
    from dataclasses import replace
    import jax
    from rfx.stepping import graded_loop
    from rfx.boundaries import cpml
    from rfx.runners import _distributed_nu_wrapped as runner
    from tests._distributed_dispersive_cpml import record
    from tests.unit.runners.test_distributed_nu_wrapped_step import assert_same_tree
    sim, saved, expected = _lorentz_scene(name), {}, {}
    build = graded_loop._build_nu_scan
    def capture_one(*args, **kwargs):
        saved['one'] = build(*args, **kwargs)
        raise BeforeStepping
    with monkeypatch.context() as patch:
        patch.setattr(graded_loop, '_build_nu_scan', capture_one)
        with pytest.raises(BeforeStepping):
            sim.forward(n_steps=2, checkpoint=False, skip_preflight=True)
    one = saved['one']
    def inspect_one(*args, **kwargs):
        for key in ('inv_eps_r_update', 'e_loss'):
            expected[key] = tuple(np.asarray(a) for a in kwargs[key])
        raise BeforeStepping
    with monkeypatch.context() as patch:
        patch.setattr(cpml, 'apply_cpml_e', inspect_one)
        with pytest.raises(BeforeStepping):
            boundary_kernel(one.step_fn_inv.context, patch)(frame(one.invariants, one.carry_init))
    build_local, finish = runner.build_setup, runner.split
    def capture_local(*args, **kwargs):
        setup = build_local(*args, **kwargs)
        saved['context'] = setup.step_fn_inv.context
        return setup
    def capture_slabs(*args, **kwargs):
        saved['invariants'], saved['carry'] = finish(*args, **kwargs)
        saved['layout'] = args[2]
        raise BeforeStepping
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'build_setup', capture_local)
        patch.setattr(runner, 'split', capture_slabs)
        with pytest.raises(BeforeStepping):
            sim.forward(n_steps=2, checkpoint=False, skip_preflight=True,
                        distributed=True, devices=host_devices(1))
    layout = saved['layout']
    def local(a):
        return a if not hasattr(a, 'ndim') or a.ndim == 0 else a.addressable_shards[0].data
    inv = jax.tree.map(local, {key: saved['invariants'][key] for key in (
        'materials', 'debye_coeffs', 'lorentz_coeffs', 'cpml_inv_eps_r', 'e_loss', 'cpml_x', 'owns_x')})
    inv['owns_x'] = jax.tree.map(lambda a: a[0], inv['owns_x'])
    inv['e_loss'] = (*inv['e_loss'], False)
    carry = jax.tree.map(local, {key: saved['carry'][key] for key in ('fdtd', 'cpml')})
    owned = slice(layout.ghost_width, layout.ghost_width + layout.nx)
    def inspect_slab(*args, **kwargs):
        for key, want in expected.items():
            actual = kwargs[key]
            row = dict(entry='lorentz_cpml_operand', kind=name, devices=1, operand=key,
                       expected_min=[float(a.min()) for a in want],
                       expected_max=[float(a.max()) for a in want],
                       actual=None if actual is None else 'array')
            if actual is not None:
                row['max_abs'] = max(float(np.abs(np.asarray(b)[owned].astype(np.float64)-a).max())
                                     for a, b in zip(want, actual))
            record(row)
            assert actual is not None, f"kernel {key} missing: actual=None, {row}"
            assert_same_tree(want, actual, 'kernel ' + key, owned=owned)
        raise BeforeStepping
    context = replace(saved['context'], x_face_frame={'x': (
        layout.ghost_width, layout.ghost_width + layout.nx)})
    with monkeypatch.context() as patch:
        patch.setattr(cpml, 'apply_cpml_e', inspect_slab)
        with pytest.raises(BeforeStepping):
            boundary_kernel(context, patch)(frame(inv, carry))
