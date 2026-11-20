"""Explicit opt-in defect injection for the blocked-fit mutation witnesses."""

import importlib
import inspect
import os


def pytest_configure(config):
    mutation = os.environ.get('RFX_BLOCKED_MUTATION')
    if not mutation:
        return
    from rfx.ringdown import _blocked as blocked
    planning = importlib.import_module('rfx.ringdown._pencil')
    from rfx.measurement import matrix_pencil as pencil

    if mutation in ('projection_dropped', 'partial_skipped', 'padded_tb_rows_dropped'):
        module = blocked
        source = inspect.getsource(module)
        if mutation == 'projection_dropped':
            old = '    _, q2 = _project(tangents[0] * g[None, :], s0, dt, mask, blocks)'
            new = '''    from rfx.ringdown._jax import _basis
    b, t = _basis(s0, y.shape[0], dt)
    c0 = _tri(blocks[0], blocks[2]) / blocks[5][:, None] * mask[:, None]
    d = (t[:, None] * b * mask[None, :])[:, None, :] * (c0.T*g[:, None])[None, :, :]
    ghat = d.reshape(-1, s0.size)
    q2 = tangents[0] * g[None, :]'''
            source = source.replace(old, new)
        elif mutation == 'partial_skipped':
            source = source.replace('(length + size - 1) // size', 'length // size')
        else:
            source = source.replace('initial = initial.at[k:2*k, k:2*k].set(unit)', 'initial = initial')
    elif mutation in ('host_partial_skipped', 'host_end_reference_dropped'):
        module = pencil
        source = inspect.getsource(module)
        if mutation == 'host_partial_skipped':
            source = source.replace('range(0, length, size)', 'range(0, length - length % size, size)')
        else:
            source = source.replace('(t - end)[:, None] * s[None, growing]', 't[:, None] * s[None, growing]')
    else:
        module = planning
        source = inspect.getsource(module)
        if mutation == 'static_stage2_dropped':
            source = source.replace('r=capacity,', 'r=None,')
        elif mutation == 'static_capacity_capped':
            source = source.replace('capacity = min(Nd - columns, columns * C)',
                                    'capacity = min(rd.HARMINV_MAX_MODES, Nd - columns, columns * C)')
        else:
            raise ValueError(mutation)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    if module is planning:
        from rfx import ringdown as rd
        rd._check_run_identification_memory = planning._check_run_identification_memory
