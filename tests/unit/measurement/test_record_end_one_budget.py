"""The record check judges one budget, pole tail plus known drive, and says what to do (rfx-lead L:d09556).

until_decay stops on field energy, no earlier than the step where the known drive's share is 1e-2; it does not test
the pole share (#2030 owns that in the loop). So a run it stops with s_poles = s_drive = 6e-3 must read fail in its
record check, and the warning must name the remedy: a longer fixed record, or until_identified once #2030 admits
waveguide ports.
"""
import numpy as np
import pytest

from rfx.sparams._settling_warning import _warn_if_ringdown_truncated
from rfx.sparams._tail_witness import tail_share_witness

DT, F, N = 1e-12, 1e10, 2000
SHARE = 6e-3


def _damped(n, share):
    """exp(j w n dt) * rho**n with rho**N chosen so the on-resonance tail / in-record DFT is ``share``:
    tail = (rho z)^N / (1 - rho z), record = (1 - (rho z)^N) / (1 - rho z), so share = rho^N / (1 - rho^N)."""
    rho = (share / (1 + share)) ** (1 / N)
    k = np.arange(n)
    return rho ** k * np.exp(2j * np.pi * F * DT * k)


def test_pole_plus_drive_over_budget_reads_fail_and_names_the_remedy():
    records = [("port0/v_probe_t", _damped(N, SHARE)[:, None])]
    alone = tail_share_witness(records, DT, 0, [F], freq_max=2 * F)
    both = tail_share_witness(records, DT, 0, [F], freq_max=2 * F,
                              known_drives=[_damped(8 * N, SHARE).real], n_record=N)
    assert alone.status == 'pass' and alone.share_per_bin[0] == pytest.approx(SHARE, rel=.05), (
        f"pole tail alone {alone.status} {alone.share_per_bin}")
    assert both.drive_share_per_bin[0] == pytest.approx(SHARE, rel=.05), both.drive_share_per_bin
    assert both.status == 'fail' and both.share_per_bin[0] > 1e-2, (
        f"one budget: {both.status} {both.share_per_bin}, drive {both.drive_share_per_bin}")

    row = {**both._asdict()}
    with pytest.warns(UserWarning) as caught:
        _warn_if_ringdown_truncated([both.db], (), witnesses=[row], n_steps=N,
                                    drive_labels=("S-channel",), consequence="S-parameter truncation is suspect")
    text = str(caught[0].message)
    assert 'NOT PASSED' in text and 'known drive beyond the record' in text, text
    # What to do: a longer fixed record; until_decay does not judge the budget; until_identified once #2030 admits ports.
    assert 'longer fixed record' in text and 'until_decay' in text and 'until_identified' in text and '#2030' in text, text


def _step_port(n, **change):
    """The step fixture's vacuum port near cutoff (a/45), its applied tables cut to n as the port builds them."""
    from types import SimpleNamespace
    import jax.numpy as jnp
    from rfx.model import waveguide_modes as wm
    fc, dt, dx, f0, bw = 6555808844.538607, 9.68540033721751e-13, .02286/45, 7.45e9, .6
    p = dict(f_c=fc, medium=wm.PortMedium(), dt=dt, dx=dx, t0=5/(f0*bw), tau=np.sqrt(2)/(f0*bw), f0=f0,
             waveform="modulated_gaussian", mode_profile="discrete", mode_type="TE")
    e, h = wm._incident_tables(**p, dft_total_steps=n, field_dtype=jnp.float32)
    zeros = np.zeros(n, np.float32)
    fields = dict(f_cutoff=fc, medium=p['medium'], dt=dt, dx=dx, src_t0=p['t0'], src_tau=p['tau'], src_fcen=f0,
                  waveform=p['waveform'], mode_profile='discrete', mode_type='TE', src_amp=1., port_name='p1',
                  e_inc_table=np.asarray(e), h_inc_table=np.asarray(h), v_inc_t=np.asarray(e), v_probe_t=zeros,
                  v_ref_t=zeros, i_probe_t=zeros, i_ref_t=zeros, n_steps_recorded=n, source_x_m=0.,
                  reference_x_m=.01, probe_x_m=.02, normal_axis='x', freqs=np.array([7e9]))
    return SimpleNamespace(**{**fields, **change})


def test_record_check_reads_the_applied_drive_beyond_the_record(monkeypatch):
    """Wiring (review P2): the port's record check gets the uncut tables, and they are the applied ones."""
    import rfx.measurement.settling as settling
    from rfx.measurement.record_end import drive_tail_share
    from rfx.model.waveguide_modes import incident_tables_uncut

    n = 2000
    cfg = _step_port(n)
    e, h = incident_tables_uncut(cfg)
    assert len(h) > n and np.array_equal(h[:n].astype(np.float32), cfg.h_inc_table), 'uncut H prefix is the applied H'
    seen = {}
    monkeypatch.setattr(settling, "settling_db_from_named_records", lambda named, **kw: seen.update(kw) or 0.)
    settling.settling_db_from_port_records((cfg,))
    drives = seen['known_drives']
    assert seen['n_record'] == n and len(drives) == 2 and all(len(d) > n for d in drives), seen
    assert drive_tail_share(drives, cfg.dt, n, cfg.freqs)[0] > 1e-2, 'a 2000-step record near cutoff leaves drive'

    # A config whose timing was changed without rebuilding its tables (waveguide.py copies port 0's t0 and tau):
    # the record check must read cannot judge, not a drive the port never applied (review P2).
    stale = _step_port(n, src_t0=cfg.src_t0*1.01)
    settling.settling_db_from_port_records((stale,))
    assert isinstance(seen['known_drives'], ValueError), seen['known_drives']
    witness = tail_share_witness([("port0/v_probe_t", _damped(N, SHARE)[:, None])], DT, 0, [F], freq_max=2*F,
                                 known_drives=seen['known_drives'], n_record=N)
    assert witness.status == 'undetermined' and 'cannot be judged' in witness.reason, witness
