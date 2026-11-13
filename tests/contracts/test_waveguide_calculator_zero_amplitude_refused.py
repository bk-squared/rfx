"""The full S calculator needs a nonzero declared drive on every port (#2715)."""

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.errors import RfxRefusal


DX = .02286 / 9
NORMALIZE = {"uniform": (False, True, "flux"), "graded-x": (True, "flux")}


def _guide(mesh, amplitude):
    profile = np.array([DX] * 11 + [.9 * DX, 1.1 * DX] + [DX] * 11)
    sim = Simulation(
        freq_max=12e9, domain=(24 * DX, 9 * DX, 4 * DX), dx=DX,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=2,
        **({"dx_profile": profile} if mesh == "graded-x" else {}),
    )
    options = dict(freqs=np.array([9e9, 10e9]), f0=9.5e9,
                   ref_offset=1, probe_offset=2)
    sim.add_waveguide_port(6 * DX, direction="+x", name="port 1",
                           amplitude=1.0, **options)
    sim.add_waveguide_port(18 * DX, direction="-x", name="port 2",
                           amplitude=amplitude, **options)
    return sim


class ReachedFieldLoop(Exception):
    pass


def _patch_loops(monkeypatch):
    def reached(*args, **kwargs):
        raise ReachedFieldLoop("field time loop reached")

    for name in ("scan", "fori_loop", "while_loop"):
        monkeypatch.setattr(jax.lax, name, reached)


@pytest.mark.parametrize("mesh", NORMALIZE)
def test_zero_amplitude_refused_before_field_loop(monkeypatch, mesh):
    sim = _guide(mesh, 0.0)
    _patch_loops(monkeypatch)
    for normalize in NORMALIZE[mesh]:
        with pytest.raises(RfxRefusal) as caught:
            sim.compute_waveguide_s_matrix(
                n_steps=2, normalize=normalize, skip_preflight=True)
        assert str(caught.value) == (
            "Waveguide port(s) 'port 2' have amplitude 0.0. "
            "The calculator drives each port in turn, "
            "so every waveguide port needs a nonzero amplitude."
        )


@pytest.mark.parametrize("mesh", NORMALIZE)
def test_nonzero_amplitudes_reach_field_loop(monkeypatch, mesh):
    sim = _guide(mesh, 1.0)
    _patch_loops(monkeypatch)
    for normalize in NORMALIZE[mesh]:
        with pytest.raises(ReachedFieldLoop, match="field time loop reached"):
            sim.compute_waveguide_s_matrix(
                n_steps=2, normalize=normalize, skip_preflight=True)
