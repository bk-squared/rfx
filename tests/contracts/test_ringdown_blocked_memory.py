"""Both public entries refuse the static host stage-2 bound before any field step."""

import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.measurement import matrix_pencil as pencil
from tests.unit.sparams.test_ringdown_run import _box, FREQS


def test_run_and_forward_refuse_stage2_before_first_step(monkeypatch):
    """A captured lane callable must never execute; both refusals preserve identical detail."""
    monkeypatch.setattr(pencil, '_RESIDUE_BLOCK_TARGET_BYTES', 2**40)
    n_win, nd, channels = 300, 300, 2
    columns = int(.33*nd)
    rows = nd-columns
    capacity = min(rows, columns*channels)
    stage1 = 1.87*16*(3*rows*columns*channels + rows*capacity
                      + 2*capacity*columns*channels + 4*capacity**2)
    width = capacity+channels
    stage2 = 1.71*16*5*(n_win*width + width**2)
    assert stage2 > stage1
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', (stage1+stage2)/2)
    original_run, original_forward = rd.RingdownRun.run, rd.RingdownForward.run
    steps = []

    def forbidden(**kwargs):
        steps.append(kwargs)
        pytest.fail('field lane reached before static stage-2 refusal')

    def wrapped_run(plan, call):
        return original_run(plan, forbidden)

    def wrapped_forward(plan, call):
        return original_forward(plan, forbidden)

    monkeypatch.setattr(rd.RingdownRun, 'run', wrapped_run)
    monkeypatch.setattr(rd.RingdownForward, 'run', wrapped_forward)
    errors = []
    for entry in ('run', 'forward'):
        sim = _box('uniform')
        kwargs = dict(n_steps=600, skip_preflight=True, ringdown=rd.RingdownSpec())
        with pytest.raises(rd.IdentificationMemoryRefused) as caught:
            if entry == 'run':
                sim.run(compute_s_params=True, s_param_freqs=FREQS, **kwargs)
            else:
                sim.forward(port_s11_freqs=FREQS, **kwargs)
        errors.append(caught.value)
    assert not steps, 'no field step may execute'
    assert type(errors[0]) is type(errors[1])
    assert str(errors[0]) == str(errors[1]), 'run and forward refusal messages differ'
    for error in errors:
        assert error.stage == 2 and error.r == capacity
        assert error.needed_bytes == stage2, (error.needed_bytes, stage2)
        assert (error.n_win, error.Nd, error.C) == (n_win, nd, channels)
        assert np.isfinite(error.needed_bytes)
