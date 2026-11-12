"""P1-P3: the same pole's gradient from the 12 ns and 24 ns sheet-Rs records."""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import ringdown as rd
from tests._x64_compat import enable_x64
from tests.oracle._sheet_rs_pole_scene import BINS, RS0, SIGMA, scene


@pytest.fixture(scope='module')
def longer_record_poles():
    # Share exactly one forward and one vmapped pullback per record across P1-P3.
    # Form each pole objective's cotangent separately: putting failed selectors
    # in one vector objective would mix their NaNs into the selected pole's VJP.
    with enable_x64():
        start = time.monotonic()
        records, gradients, walls = [], [], []
        for record_s in (12e-9, 24e-9):
            record_start = time.monotonic()
            _, grid, _ = scene(h=1e-3)
            n_steps = round(record_s / grid.dt)

            def one(Rs):
                sim, traced_grid, spec = scene(h=1e-3, Rs=Rs)
                result = sim.forward(
                    sigma_override=jnp.full(traced_grid.shape, SIGMA), n_steps=n_steps,
                    port_s11_freqs=BINS, ringdown=spec, skip_preflight=True).ringdown
                return result, result

            output, pullback, result = jax.vjp(one, jnp.float64(RS0), has_aux=True)

            def cotangent(target):
                _, pole_pullback = jax.vjp(lambda r: r.pole(target)[1], output)
                return pole_pullback(jnp.float64(1.))[0]

            cotangents = jax.vmap(cotangent)(jnp.asarray([8e9, 9e9, 8.12e9]))
            slopes, = jax.vmap(pullback)(cotangents)
            jax.block_until_ready(slopes)
            records.append(result)
            gradients.append(np.asarray(slopes))
            walls.append(time.monotonic() - record_start)
        print('longer-record pole ' + json.dumps({
            'record_wall_s': walls, 'wall_s': time.monotonic() - start,
            'gradients': [g.tolist() for g in gradients]}, sort_keys=True))
        yield tuple(records), gradients


def test_p1_longer_record_selected(longer_record_poles):
    records, gradients = longer_record_poles
    assert not any(isinstance(leaf, jax.core.Tracer)
                   for leaf in jax.tree_util.tree_leaves(records))
    g, g_other = (values[0] for values in gradients)
    assert np.isfinite([g, g_other]).all()
    baseline = rd.gradient_witness(g, g_other, against='longer_record')
    w = rd.gradient_witness(g, g_other, against='longer_record',
                            pole_target=8e9, ringdown=records)
    assert w.judged is True and w.ok is True
    assert np.float64(w.value).tobytes() == np.float64(baseline.value).tobytes()
    assert w.value <= w.bar == 1e-2
    # Independently formatted from the design's measured start values.
    selected = f'f={7.996492e9:.6g} Hz, 1/Q={3.8458e-4:.4g}'
    assert w.note == (f'selected at pole_target={8e9:.6g} Hz: '
                      f"grad's record ({12:.4g} ns): {selected}; "
                      f"grad_other's record ({24:.4g} ns): {selected}; record ratio {2:.3g}")


def test_p2_longer_record_neither_selected(longer_record_poles):
    records, gradients = longer_record_poles
    g, g_other = (values[1] for values in gradients)
    assert np.isnan([g, g_other]).all()
    w = rd.gradient_witness(g, g_other, against='longer_record',
                            pole_target=9e9, ringdown=records)
    assert w.judged is False and w.ok is False and np.isnan(w.value)
    assert w.note == ("cannot judge: grad's record (12 ns): none within delta (radius 166.7 MHz); "
                      "grad_other's record (24 ns): none within delta (radius 83.33 MHz)")


def test_p3_longer_record_only_short_selected(longer_record_poles):
    records, gradients = longer_record_poles
    g, g_other = (values[2] for values in gradients)
    assert np.isfinite(g) and np.isnan(g_other)
    w = rd.gradient_witness(g, g_other, against='longer_record',
                            pole_target=8.12e9, ringdown=records)
    assert w.judged is False and w.ok is False and np.isnan(w.value)
    assert w.note == "cannot judge: grad_other's record (24 ns): none within delta (radius 83.33 MHz)"
