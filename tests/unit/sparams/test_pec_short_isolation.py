"""Exact disconnected-network coverage, not reflection-accuracy qualification.

The advisory fixture's old (2.25, 3] power target is retired; see T12 in
docs/design_notes/931_migration. A full PEC plate separates the Maxwell
update into two uncoupled regions. With zero initial fields, the undriven
region stays exactly zero. This needs neither a fitted tolerance nor decay.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests._pec_short_advisory_fixture import build


@pytest.mark.slow
@pytest.mark.parametrize("band,dx,cpml,periods", [
    ((4e9, 6e9), 2e-3, 8, 30),
    ((5e9, 7e9), 1e-3, 10, 40),
], ids=["coarse", "fine"])
def test_pec_short_isolates_two_observable_ports(monkeypatch, band, dx, cpml, periods):
    import rfx.simulation as solver
    from rfx.sources.waveguide_port import extract_waveguide_port_waves

    records, waves = {}, []
    original_run = solver.run

    def capture(*args, **kwargs):
        result = original_run(*args, **kwargs)
        drive = len(waves)
        driven = result.waveguide_ports[drive]
        a, b = extract_waveguide_port_waves(driven)
        waves.append((np.asarray(a), np.asarray(b)))
        for port, cfg in enumerate(result.waveguide_ports):
            for name in ("v_ref_t", "i_ref_t", "v_probe_t", "i_probe_t"):
                records[f"drive{drive}_port{port}_{name}"] = np.asarray(getattr(cfg, name))
        return result

    monkeypatch.setattr(solver, "run", capture)
    frequencies = np.linspace(*band, 6)
    result = build(frequencies, dx, cpml).compute_waveguide_s_matrix(
        normalize=False, num_periods=periods)
    s = np.asarray(result.s_params)
    report = dict(frequencies_hz=frequencies.tolist(), dx=dx, cpml=cpml,
                  num_periods=periods, s_real=s.real.tolist(), s_imag=s.imag.tolist(),
                  column_power=np.sum(np.abs(s)**2, axis=0).tolist(),
                  settling_db=np.asarray(result.settling_db).tolist(),
                  trace_peaks={key: float(np.max(np.abs(v))) for key, v in records.items()},
                  driven_a_abs=[np.abs(a).tolist() for a, b in waves],
                  driven_b_abs=[np.abs(b).tolist() for a, b in waves])
    print(json.dumps(report, indent=2))
    if directory := os.environ.get("RFX_SHORT_EVIDENCE_DIR"):
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        label = "coarse" if dx == 2e-3 else "fine"
        (out / (label + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(out / (label + "-records.npz"), **records)
    assert len(waves) == 2
    assert np.isfinite(s).all()
    np.testing.assert_array_equal(s[[0, 1], [1, 0], :], 0)
    assert np.all(np.abs(s[[0, 1], [0, 1], :]) > 0)
    for drive, (a, b) in enumerate(waves):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.all(np.abs(a) > 0) and np.all(np.abs(b) > 0)
        for port in range(2):
            for name in ("v_ref_t", "i_ref_t", "v_probe_t", "i_probe_t"):
                trace = records[f"drive{drive}_port{port}_{name}"]
                assert np.isfinite(trace).all()
                if drive == port:
                    assert np.max(np.abs(trace)) > 0, "drive must be observable"
                else:
                    np.testing.assert_array_equal(trace, 0)
