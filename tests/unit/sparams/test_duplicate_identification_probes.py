"""Identify duplicates after positions resolve to component-owned cells."""
import sys
import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx import ringdown as rd
from rfx.ringdown import RingdownForward, RingdownRun, RingdownSpec


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("planner", [RingdownRun, RingdownForward])
def test_duplicate_identification_probe_indices_warn_once(lane, planner):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                     boundary="pec",
                     **({"dz_profile": np.full(8, .001)} if lane == "graded" else {}))
    grid = sim._build_nonuniform_grid() if lane == "graded" else sim._build_grid()
    probes = (((.003, .003, .003), "ez"),
              ((.00301, .003, .003), "ez"),
              ((.003, .003, .003), "ex"),
              ((.003, .003, .003), "ez"))
    with pytest.warns(UserWarning, match=r"Duplicate.*indices \[\[0, 1, 3\]\]") as records:
        plan = planner(sim, RingdownSpec(identification_probes=probes),
                       lane=lane, n_steps=20, grid=grid)
    assert len(records) == 1
    assert len(plan.identification_keys) == 4
    assert plan.identification_keys[0] == plan.identification_keys[1]
    assert plan.identification_keys[0] != plan.identification_keys[2]
    assert records[0].filename == __file__


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_duplicate_warning_once_per_early_stop_run(lane, monkeypatch, capsys):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                     boundary="pec",
                     **({"dz_profile": np.full(8, .001)} if lane == "graded" else {}))
    sim.add_port((.004, .004, .002), "ez", impedance=50., extent=.002,
                 waveform=lambda t: 0. * t)
    probe = ((.003, .003, .003), "ez")
    spec = RingdownSpec(identification_probes=(probe, probe))
    grid = sim._build_realized_grid()
    dt = float(grid.dt)
    distance = np.sqrt(3) * .008
    tau = 2 * distance / 299792458.
    ready = 1
    while (ready - round(ready / 2)) * dt < 2.05 * tau:
        ready += 1
    plans = []
    first_checks = []
    original = RingdownRun.__init__

    def counted(self, *args, **kwargs):
        plans.append(kwargs["n_steps"])
        original(self, *args, **kwargs)

    def stop_after_check(self, count, peek):
        if not self.schedule.due(count):
            return False
        first_checks.append(self.first_check)
        self._check(count, peek)
        return True

    monkeypatch.setattr(RingdownRun, "__init__", counted)
    monkeypatch.setattr(rd, "STOP_CHUNK", ready)
    monkeypatch.setattr(rd.RingdownStop, "after_chunk", stop_after_check)
    for run in range(2):
        plans.clear()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            result = sim.run(n_steps=2 * ready, ringdown=spec, until_identified=True,
                             skip_preflight=True)
        stopped = first_checks[-1]
        with capsys.disabled():
            print(f'12u.1 {lane} run={run + 1}: dt={dt:.17g}; tau={tau:.17g}; '
                  f'T_ready={ready}; first_check={stopped}; plans={plans}; '
                  f'record_length={len(result.time_series)}', file=sys.stderr, flush=True)
        duplicates = [r for r in records if "Duplicate ring-down" in str(r.message)]
        assert len(duplicates) == 1
        assert duplicates[0].filename == __file__
        assert plans == [2 * ready, stopped, stopped]
        assert len(result.time_series) == stopped
