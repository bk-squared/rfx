"""R4: E end stays fixed; the known H tail delays the host energy stop."""
from types import SimpleNamespace

import numpy as np

from rfx.measurement import decay
from rfx.measurement.record_end import drive_tail_share


def test_energy_stop_waits_for_known_drive(monkeypatch):
    import rfx.model.waveguide_modes as modes

    interval, length = 20, 800
    e = np.zeros(length)
    e[:40] = 1.
    h = e.copy()
    h[40:600] = .02
    cfg = SimpleNamespace(src_amp=1., e_inc_table=e, h_inc_table=h,
                          dt=1., freqs=np.array([0., .001]))
    monkeypatch.setattr(modes, 'incident_tables_uncut', lambda port: (e, h))
    monkeypatch.setattr(decay, 'boundary_energy', lambda *a, **kw: lambda *a: 0.)
    grid = SimpleNamespace(dt=1.)
    n_d = next(n for n in range(interval, length + 1, interval)
               if np.all(drive_tail_share([e, h], 1., n, cfg.freqs) <= 1e-2))
    assert n_d > 40
    assert decay.drive_end(grid, length, waveguide_ports=[cfg]) == 40
    rule, _ = decay.make_decay_stop(
        grid, None, thr=.1, min_steps=40, interval=interval, consecutive=2,
        criterion=True, max_steps=length, side_step=None, waveguide_ports=[cfg])
    assert rule.n_d == n_d
    for n in range(interval, n_d, interval):
        assert rule.check(n, lambda: 1. if n <= 40 else .001) is None, (
            f"R4: decay stop fired at {n} before known-drive ready step {n_d}")
    assert rule.check(n_d, lambda: .001) == n_d
    capped, _ = decay.make_decay_stop(
        grid, None, thr=.1, min_steps=40, interval=interval, consecutive=2,
        criterion=True, max_steps=100, side_step=None, waveguide_ports=[cfg])
    for n in range(interval, 101, interval):
        assert capped.check(n, lambda: 1. if n <= 40 else .001) is None
    diagnostic = decay.finish_decay(capped, 100)
    assert diagnostic.values['criterion'] == 'cap'
    assert f"known tail exceeded 1e-2 of the in-record drive until step {n_d}" in diagnostic.message
