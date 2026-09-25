"""Both coaxial line lanes put the line in open space: the run absorbs on x, y and z.

A coaxial line whose conductors end without a shield lets field out of the
annulus: the open termination does, and so do the pin and shell ending one cell
past each matched feed. When the lanes absorbed on z only, their lateral pads
were vacuum backed by the PEC grid faces, a closed metal can around the line,
and the can held that field between the shell and its walls. The open end
coupled it back, so the open's |S11| rose above 1 and grew with the record:
1.018 at 12 line traversals and 1.046 at 24 at 9 annulus cells, and 1.0015 at
both with the lateral pads absorbing (issue 1218; the records are in rfx-archive,
``rfx/records/20260924-coax-closed-can/open_closed_can_arms.json``).

``compute_coaxial_line_reflection`` and ``compute_coaxial_two_port`` now hand
the runner ``cpml_axes="xyz"``, refuse any other value, and refuse a board with
a face that has no absorber. No FDTD runs here: a spy stops each lane at the
runner call, which keeps these checks in the fast lane. The physics the change
buys is held by ``tests/oracle/test_coax_open_end_settles.py``.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

LANES = ("compute_coaxial_line_reflection", "compute_coaxial_two_port")
DX = (2.055e-3 - 0.635e-3) / 4.0          # four annulus cells


class _ReachedTheRunner(Exception):
    """Raised by the spy once it has the runner's arguments."""


def _sim(lane: str, boundary="cpml") -> Simulation:
    lz = 0.020 if lane == "compute_coaxial_line_reflection" else 0.012
    sim = Simulation(freq_max=40e9, domain=(0.008, 0.008, lz), boundary=boundary, dx=DX)
    sim.add_coaxial_port((0.004, 0.004, lz / 2.0), face="top", pin_length=5e-3,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    return sim


def _call(sim: Simulation, lane: str, **kw):
    if lane == "compute_coaxial_line_reflection":
        return sim.compute_coaxial_line_reflection(
            termination="open", n_steps=8, freqs=np.array([8e9]), probe_count=3, **kw)
    return sim.compute_coaxial_two_port(
        n_steps=8, freqs=np.array([8e9]), probe_count=3, probe_start_cells=4,
        probe_spacing_cells=2, **kw)


def _runner_arguments(monkeypatch, sim: Simulation, lane: str, **kw) -> dict:
    """The keyword arguments the lane hands ``rfx.simulation.run``.

    The lanes import ``run`` inside the function, so patching the module
    attribute reaches the call; the spy raises before any step.
    """
    import rfx.simulation

    seen: dict = {}

    def _spy(grid, materials, n_steps, **kwargs):
        seen.update(kwargs)
        seen["grid"] = grid
        raise _ReachedTheRunner

    monkeypatch.setattr(rfx.simulation, "run", _spy)
    with pytest.raises(_ReachedTheRunner):
        _call(sim, lane, **kw)
    return seen


@pytest.mark.parametrize("lane", LANES)
def test_the_runner_absorbs_on_every_axis(monkeypatch, lane):
    seen = _runner_arguments(monkeypatch, _sim(lane), lane)
    assert seen["boundary"] == "cpml"
    assert seen["cpml_axes"] == "xyz", (
        f"{lane} handed the runner cpml_axes={seen['cpml_axes']!r}: the lateral pads "
        "are then vacuum backed by PEC grid faces, a closed can around the line "
        "(issue 1218)")
    grid = seen["grid"]
    pads = [int(getattr(grid, f"pad_{ax}_{side}")) for ax in "xy" for side in ("lo", "hi")]
    assert min(pads) > 0, f"{lane}: a lateral face has no absorber pad ({pads})"


@pytest.mark.parametrize("lane", LANES)
def test_the_lane_defaults_to_absorbing_on_every_axis(lane):
    default = inspect.signature(getattr(Simulation, lane)).parameters["cpml_axes"].default
    assert default == "xyz", f"{lane} defaults to cpml_axes={default!r}"


@pytest.mark.parametrize("axes", ["z", "xy", "x", ""])
@pytest.mark.parametrize("lane", LANES)
def test_absorbing_on_fewer_axes_is_refused(monkeypatch, lane, axes):
    import rfx.simulation

    def _must_not_run(*a, **k):
        raise AssertionError(f"{lane} reached the runner with cpml_axes={axes!r}")

    monkeypatch.setattr(rfx.simulation, "run", _must_not_run)
    with pytest.raises(ValueError, match=r"accepts only cpml_axes='xyz'.*issue 1218"):
        _call(_sim(lane), lane, cpml_axes=axes)


@pytest.mark.parametrize("face", ["x", "y", "z"])
@pytest.mark.parametrize("lane", LANES)
def test_a_face_without_an_absorber_is_refused(monkeypatch, lane, face):
    """A CPML face of zero thickness allocates no pad: the grid face behind it is
    a bare PEC wall, and on x or y that rebuilds the can on one side."""
    import rfx.simulation

    def _must_not_run(*a, **k):
        raise AssertionError(f"{lane} reached the runner with no {face}-hi absorber")

    monkeypatch.setattr(rfx.simulation, "run", _must_not_run)
    axes = {ax: "cpml" for ax in "xyz"}
    axes[face] = Boundary(lo="cpml", hi="cpml", hi_thickness=0)
    with pytest.raises(ValueError, match="positive CPML thickness on all six faces"):
        _call(_sim(lane, boundary=BoundarySpec(**axes)), lane)
