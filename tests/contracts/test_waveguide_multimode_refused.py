"""A waveguide port with several modes is refused on run() and forward() (#2070).

An empty WR-90 section: a = 22.86 mm, b = 10.16 mm, dx = 1.27 mm (18 x 8
whole cells), 101.6 mm long, PEC walls, CPML on x. A port declared with
n_modes=2 compiles to a list of per-mode configurations that only
compute_waveguide_s_matrix() drives, one run per port and mode. The single-run
assembly of run() and forward() took that list for one configuration and
raised AttributeError; admission now refuses it before the first time step.

Physical anchor: none. This file pins a refusal and the uniform calculator's
S shape, not a number a user receives. Imitates
tests/contracts/test_admission_refusal_order.py (message fragments written
here, never read from the admission tables) and the census's one-step entries.
"""

import traceback
from pathlib import Path

import jax
import numpy as np
import pytest

from rfx import Simulation

FREQS = np.array([8e9, 10e9, 12e9])


def _section(*, graded=False, n_modes=2, two_ports=False):
    sim = Simulation(12e9, (.1016, .02286, .01016), dx=.00127,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8,
                     **({"dx_profile": np.full(80, .00127)} if graded else {}))
    sim.add_waveguide_port(.02, direction="+x", name="launch", f0=10e9,
                           freqs=FREQS, n_modes=n_modes)
    if two_ports:
        sim.add_waveguide_port(.08, direction="-x", name="receive", f0=10e9,
                               freqs=FREQS, n_modes=n_modes)
    return sim


def _call(sim, entry, skip_preflight):
    options = {"n_steps": 1, "skip_preflight": skip_preflight}
    if entry == "forward":
        options["checkpoint"] = False
    return getattr(sim, entry)(**options)


@pytest.mark.parametrize("skip_preflight", (True, False), ids=("no-preflight", "preflight"))
@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
def test_two_mode_port_refused_before_the_first_step(monkeypatch, graded, entry, skip_preflight):
    sim = _section(graded=graded)

    def forbidden(*args, **kwargs):
        pytest.fail("a two-mode waveguide port reached lax.scan")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError) as caught:
        _call(sim, entry, skip_preflight)
    message = str(caught.value)
    assert "a waveguide port with several modes (add_waveguide_port(n_modes>1))" in message
    assert "compute_waveguide_s_matrix()" in message
    assert "refused before the first time step" in message
    raising = traceback.extract_tb(caught.value.__traceback__)[-1].filename
    assert Path(raising).parts[-3:] == ("rfx", "runners", "_admission.py"), raising


def test_topology_optimize_refuses_a_two_mode_port(monkeypatch):
    from rfx.topology import TopologyDesignRegion, topology_optimize
    sim = _section()
    sim.add_material("design", eps_r=2.)
    region = TopologyDesignRegion(corner_lo=(.05, .00508, .00254),
                                  corner_hi=(.05254, .00762, .00508),
                                  material_bg="air", material_fg="design")

    def forbidden(*args, **kwargs):
        pytest.fail("a two-mode waveguide port reached lax.scan")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError) as caught:
        topology_optimize(sim, region, lambda result: 0., n_iterations=1,
                          verbose=False, skip_preflight=True)
    assert "a waveguide port with several modes" in str(caught.value)


@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
def test_one_mode_port_is_admitted(graded, entry):
    result = _call(_section(graded=graded, n_modes=1), entry, True)
    assert result is not None


def test_uniform_calculator_still_measures_both_modes_of_both_ports():
    result = _section(two_ports=True).compute_waveguide_s_matrix(n_steps=2)
    assert np.shape(result.s_params) == (4, 4, len(FREQS))
