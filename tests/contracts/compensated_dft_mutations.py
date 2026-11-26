"""Same-process mutations M1-M5; explicitly collected only when requested."""
import json
import sys
from types import FunctionType

import jax
import jax.numpy as jnp
import pytest

from rfx.measurement import dft, probes
from tests.contracts.test_compensated_dft import (
    LAST_ROWS, check_p1, check_p2, check_p3, check_p4, check_p5a, check_p5b,
)


def _patch_add(monkeypatch, replacement):
    monkeypatch.setattr(dft, 'compensated_add', replacement)
    monkeypatch.setattr(probes, 'compensated_add', replacement)


def _judge(check, mutation):
    jax.clear_caches()
    red = False
    try:
        with jax.default_device(jax.devices('cpu')[0]):
            check()
    except AssertionError:
        red = True
    pin = {'check_p5a': 'P5a', 'check_p5b': 'P5b'}.get(check.__name__, check.__name__.replace('check_p', 'P'))
    row = dict(pin=pin, mutation=mutation, e_over_u=LAST_ROWS[pin]['e_over_u'], red=red)
    print('MUTROW ' + json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
    assert red, row


@pytest.mark.parametrize('check', [check_p1, check_p2])
def test_m1(monkeypatch, check):
    _patch_add(monkeypatch, lambda total, compensation, term: (total + term, compensation))
    _judge(check, 'M1')


def test_m2(monkeypatch):
    from rfx.probes import probes as public_probes
    import rfx.probes as exports
    def plain(probe, state, dt):
        value = getattr(state, probe.component)[probe.index]
        phase = probes._dft_phase(state.step-1, probe.freqs, dt, probes.field_channel(probe.component)) / dt
        weight = probes._dft_window_weight(state.step, probe.total_steps, probe.window, probe.window_alpha)
        return probe._replace(accumulator=probe.accumulator + value * phase * dt * weight)
    for module in (probes, public_probes, exports):
        monkeypatch.setattr(module, 'update_dft_probe', plain)
    _judge(check_p3, 'M2')


def test_m3(monkeypatch):
    original = dft._replay.__wrapped__
    namespace = dict(original.__globals__, compensated_add=lambda total, compensation, term: (total + term, compensation))
    replay = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    replay.__kwdefaults__ = original.__kwdefaults__
    monkeypatch.setattr(dft, '_replay', jax.jit(replay, static_argnames=('traced', 'offset', 'window', 'alpha')))
    _judge(check_p4, 'M3')


def test_m4(monkeypatch):
    original = dft.compensated_add
    _patch_add(monkeypatch, lambda total, compensation, term: original(total, compensation, jax.lax.stop_gradient(term)))
    _judge(check_p5a, 'M4')


def test_m5(monkeypatch):
    original = dft.compensated_add
    def changed(total, compensation, term):
        new_total, new_compensation = original(total, compensation, term)
        return new_total, new_compensation + 2. ** -12 * term
    _patch_add(monkeypatch, changed)
    _judge(check_p5b, 'M5')
