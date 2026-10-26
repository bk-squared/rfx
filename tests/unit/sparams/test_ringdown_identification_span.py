"""The identification span bounds cost without changing the short-record path."""
import numpy as np
import pytest

from rfx import ringdown as rd
from rfx._ringdown_pencil import identification_span
from rfx.sparams._tail_witness import tail_share_witness


@pytest.mark.parametrize('n', [199, 200, 1000, 32832])
def test_identification_span_is_the_whole_record_below_cap(n):
    dt, freq_max = 9.68540033721751e-13, 15e9
    assert ((n + 1)//2) * dt <= 1024 / (4 * freq_max)
    assert identification_span(n, dt, freq_max) == n


def test_four_n2_record_identifies_at_duration_cap_and_after_source_end(monkeypatch):
    n, dt, freq_max = 4 * 32832, 9.68540033721751e-13, 15e9
    factors, old_count = rd.decimation_plan(n//2, dt, freq_max)
    assert factors == (8, 2) and old_count > 1024
    # Addendum 6: T_cap=17.066666666666667 ns; floor(T_cap/dt)=17621.
    # The realized plan retains ceil((ceil(17621/8)-20)/2)-20=1072.
    half = 17621
    expected_span = 2 * half
    assert half * dt <= 1024 / (4 * freq_max) < (half + 1) * dt
    source_end = n - expected_span + 123
    models = []
    identify = rd.identify

    def capture(*args, **kwargs):
        model = identify(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(rd, 'identify', capture)
    y = np.exp((-1e6 + 2j*np.pi*10e9) * np.arange(n) * dt)
    tail_share_witness([('synthetic', y)], dt, source_end, [10e9], freq_max=freq_max)
    assert len(models) == 2
    assert models[0].n_decimated == 1072
    assert models[0].n_window == half
    assert models[0].n_window * dt <= 1024 / (4 * freq_max)
    assert all(model.n_ref >= source_end for model in models), 'identification precedes source end'
    assert models[1].n_window == round(.9 * half)


def test_ringdown_w2_and_early_start_use_the_same_span():
    n, dt, freq_max = 8192, 1., .2
    # T_cap=1024/(4*.2)=1280, h=1280, m=2560; no decimation.
    y = np.exp((-.0001 + 2j*np.pi*.1) * np.arange(n))
    w2 = rd.two_window_witness(y, dt, [.1], n, n//2, freq_max=freq_max)
    we = rd.early_start_witness(y, dt, [.1], n, .5, freq_max=freq_max)
    assert w2.model.n_ref == n - 1280
    assert w2.model_short.n_ref == n - 1280
    assert w2.model_short.n_window == 1024  # round(.9*2560)-1280
    assert we.model_long.n_ref == n - 1920


@pytest.mark.parametrize('start_today', [4096, 8100])
def test_until_identified_main_start_uses_span_and_todays_floor(monkeypatch, start_today):
    seen = []

    def capture(series, dt, start, stop, **kw):
        seen.append((start, stop))
        raise ValueError('window captured before pencil')

    monkeypatch.setattr(rd, 'identify', capture)
    rd._judge_record(np.ones((8192, 1)), 1., [.1], 8192, start_today,
                     rd.RingdownSpec(), ref_hz=.2, observable=lambda x: x)
    assert seen == [(max(start_today, 8192 - 1280), 8192)]


@pytest.mark.parametrize('n, expected', [(2558, 2558), (2559, 2559), (2560, 2560),
                                         (2561, 2560), (8192, 2560)])
def test_duration_boundary_includes_the_cap_and_ceils_odd_halves(n, expected):
    # dt=1, f_read=.2: T_cap=1280; ceil(n/2), then h=1280 if cut.
    assert identification_span(n, 1., .2) == expected


@pytest.mark.parametrize('caller, expected', [
    ('witness_early', [(1279, 2558), (639, 2558)]),
    ('witness_short', [(1279, 2558), (1279, 2430)]),
    ('w2', [(1279, 2558), (1279, 2302)]),
    ('we', [(1279, 2558), (640, 2558)]),
    ('judge', [(1279, 2558), (640, 2558)]),
])
def test_duration_pin_preserves_each_callers_main_geometry(monkeypatch, caller, expected):
    """Addendum 6: an undecimated half less than one period below the cap.

    Hand geometry on main: n=2558, half=1279, T_cap=1280, period=5.
    Witness early uses 1279//2=639; WE uses round(2558/4)=640.
    W2 ends at round(.9*2558)=2302; witness short at 1279+1151=2430.
    The m3 sample cap gives m=2048; m4 gives m=2550. Both move main.
    """
    n, dt, f_read = 2558, 1., .2
    assert 0 < 1280 - 1279 * dt < 1 / f_read
    assert rd.decimation_plan(1279, dt, f_read) == ((), 1279)
    seen = []

    def capture(series, step, start, stop, **kw):
        seen.append((start, stop))
        assert kw['freq_max'] == f_read
        # Geometry only: a one-pole model avoids a pencil in every caller.
        channels = 1 if np.ndim(series) == 1 else np.shape(series)[1]
        return rd.RingdownModel(
            s=np.array([-.01 + 2j*np.pi*.1]), c=np.ones((1, channels)),
            n_ref=start, dt=step, freq_max=f_read, guard=.9, factors=(),
            D=1, rank=1, n_window=stop-start, n_decimated=stop-start,
            n_growing_discarded=0, n_invalid=0, n_guard_dropped=0,
            singular_values=np.ones(1), amplitude=np.ones(1),
            s_growing=np.empty(0, complex), growing_amplitude=np.empty(0),
            window_rms=np.ones(channels))

    monkeypatch.setattr(rd, 'identify', capture)
    y = np.exp((-.001 + 2j*np.pi*.1) * np.arange(n))[:, None]
    if caller.startswith('witness'):
        # Highest read bin, rather than freq_max=.15, defines f_read=.2.
        source_end = 0 if caller == 'witness_early' else 1000
        tail_share_witness([('pin', y)], dt, source_end, [.2], freq_max=.15)
    elif caller == 'w2':
        rd.two_window_witness(y, dt, [.2], n, 1279, freq_max=f_read)
    elif caller == 'we':
        rd.early_start_witness(y, dt, [.2], n, .5, freq_max=f_read)
    else:
        rd._judge_record(y, dt, [.2], n, 1279, rd.RingdownSpec(),
                         ref_hz=f_read, observable=lambda a: a)
    assert seen == expected


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_duration_pin_preserves_run_main_we_and_w2_plan(lane):
    from tests.unit.sparams.test_ringdown_run import _box

    sim = _box(lane)
    grid = sim._build_grid() if lane == 'uniform' else sim._build_nonuniform_grid()
    dt = float(grid.dt)
    # f_read*dt=.2: T_cap=1280*dt, half=1279, margin=dt<5*dt.
    f_read = .2 / dt
    assert 0 < 1024 / (4*f_read) - 1279*dt < 1/f_read
    assert rd.decimation_plan(1279, dt, f_read) == ((), 1279)
    plan = rd.RingdownRun(sim, rd.RingdownSpec(), lane=lane, n_steps=2558,
                          grid=grid, bins=[f_read])
    # Main's round(.5*n), round(.9*n), round(.25*n), computed by hand.
    assert (plan.n_start, plan.n_split, plan.n_long) == (1279, 2302, 640)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_duration_pin_preserves_forward_host_windows(monkeypatch, lane):
    import jax
    import jax.numpy as jnp

    from rfx import _ringdown_jax as rj
    from tests.unit.sparams.test_ringdown_run import _box

    sim = _box(lane)
    grid = sim._build_grid() if lane == 'uniform' else sim._build_nonuniform_grid()
    dt = float(grid.dt)
    f_read = .2 / dt
    assert 0 < 1024 / (4*f_read) - 1279*dt < 1/f_read
    assert rd.decimation_plan(1279, dt, f_read) == ((), 1279)
    eps = jnp.full(tuple(grid.shape), 2.2, jnp.float32)
    seen = []
    host_poles = rj.host_poles

    def capture(identify_window, window, k_max, step, freqs):
        seen.append(window.shape[0])
        return host_poles(identify_window, window, k_max, step, freqs)

    monkeypatch.setattr(rj, 'host_poles', capture)
    # Trace shapes only: no time stepping or host pencil is executed.
    jax.eval_shape(lambda p: sim.forward(
        n_steps=2558, skip_preflight=True, eps_override=eps*p,
        ringdown=rd.RingdownSpec(), port_s11_freqs=np.array([f_read])).ringdown,
        jnp.float32(1.))
    # Main [1279,2558), WE [640,2558): main's unchanged host windows.
    assert seen == [1279, 1918]
