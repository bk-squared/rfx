"""The waveguide S-parameter lane hands the solver the same permittivity the
runners do, pad included (issue #1066).

``rfx/sparams/waveguide.py`` used to rebuild ``shape_eps_pairs`` from
``sim._geometry`` at both its smoothing sites, under a comment saying it
mirrored ``rfx/runners/uniform.py``. Since #1043 stage B the runner sites build
theirs through ``rfx.geometry.smoothing.smoothed_shape_pairs``, which continues
a dielectric reaching a CPML/UPML face out through that pad. The lane did not,
so such a structure was solved with ``eps_r = 1`` in its own absorber -- the
#831 end facet -- and only a comment said so.

What is asserted here is the array THE LANE ITSELF hands down, captured by
wrapping ``rfx.geometry.smoothing.compute_smoothed_eps`` on the module the lane
imports it from at call time and stopping the call there. An earlier version of
this file called ``smoothed_shape_pairs`` twice and compared the two results:
neither side was the lane, so restoring the defect in
``rfx/sparams/waveguide.py`` left it green (review of PR #1131, F1).

Measured build-only on ``origin/main`` before the fold, WR-90 at dx = 1.27 mm
with an ``eps_r = 4`` slab running into the x-hi face: the lane handed the
solver ``eps_xx = 1.0`` down all 8 CPML cells where the runner handed 4.0,
1539 of 13167 cells differing, max |diff| 3.0. After the fold the two arrays
are bit-identical.

Nothing here time-steps: the capture raises out of the lane as soon as the
permittivity exists, before the scan. These are statements about the array the
solver is handed, which is what the defect was about; they are not
S-parameter claims.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

A_WG, B_WG, DX = 22.86e-3, 10.16e-3, 1.27e-3
LX = 60 * DX
FREQS = jnp.asarray([8.5e9, 10.0e9, 11.5e9])
CPML_CELLS = 8
#: Enough to build and reach the smoothing site; the capture aborts before the
#: scan, so the value only has to pass the lane's own argument checks.
N_STEPS = 10


class _StopAfterCapture(BaseException):
    """Raised inside the spy to end the call once the array exists.

    ``BaseException`` rather than ``Exception`` on purpose: a broad
    ``except Exception`` anywhere between here and the entry point would
    swallow the abort and turn a captured array into a silent pass.
    """


def _guide(with_slab: bool, *, slab_reaches_pad: bool = True):
    """WR-90, CPML on x, PEC y/z walls, two ports. The only pads are on x."""
    sim = Simulation(
        freq_max=12e9, domain=(LX, A_WG, B_WG), dx=DX, cpml_layers=CPML_CELLS,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
    )
    if with_slab:
        sim.add_material("slab", eps_r=4.0)
        # Reaching the x-hi FACE is the whole point: that is the port-side
        # absorber. The no-pad variant stops two cells short of it.
        x_hi = LX if slab_reaches_pad else LX - 2 * DX
        sim.add(Box((LX / 2.0, 0.0, 0.0), (x_hi, A_WG, B_WG)), material="slab")
    for pos, direction, name in ((6 * DX, "+x", "left"),
                                 (LX - 6 * DX, "-x", "right")):
        sim.add_waveguide_port(pos, direction=direction, freqs=FREQS,
                               f0=1e10, name=name)
    return sim


def _capture(call, monkeypatch, *, attr: str = "compute_smoothed_eps",
             stop_after: int = 1) -> dict:
    """Record what ``call`` handed to ``rfx.geometry.smoothing.<attr>``.

    The lane and the runners both import these functions INSIDE their
    smoothing block, so patching the attribute on ``rfx.geometry.smoothing``
    reaches the real call rather than a re-export. This is the pattern
    ``tests/unit/boundaries/test_boundary_touching_dielectric_pad_continuation
    .py`` uses for the runner; the difference is that nothing here runs the
    scan -- the spy raises once it has the calls it was asked for.

    ``stop_after`` is 1 for the Stage-1 site, which smooths once, and 2 for
    the Stage-2 ``kottke_pec`` site, which builds a device tensor and then a
    vacuum reference tensor. Returns ``{"calls": [...], "stopped": bool}``;
    each call carries its positional ``args``, its ``kwargs`` and its ``out``.
    """
    from rfx.geometry import smoothing as _smoothing

    captured: dict = {"calls": [], "stopped": False}
    real = getattr(_smoothing, attr)

    def _spy(*args, **kwargs):
        out = real(*args, **kwargs)
        captured["calls"].append({
            "args": args,
            "kwargs": kwargs,
            "out": tuple(np.asarray(c) for c in out),
        })
        if len(captured["calls"]) >= stop_after:
            raise _StopAfterCapture
        return out

    monkeypatch.setattr(_smoothing, attr, _spy)
    try:
        call()
    except _StopAfterCapture:
        # Recorded rather than assumed: if any layer between here and the
        # entry point swallowed the abort, the call would have gone on to
        # time-step and this flag would stay False.
        captured["stopped"] = True
    finally:
        # Two captures run in one test, so the spy must come off before the
        # next setattr -- otherwise the second spy wraps the first, the first
        # raises inside it, and the second capture comes back empty.
        monkeypatch.undo()
    return captured


def _first(captured) -> dict:
    """The one captured call, with the grid it was given.

    An empty capture comes back as ``{"stopped": False}`` so the caller's own
    "this fixture is not on the smoothed lane" assertion is what fires, rather
    than an ``IndexError`` here.
    """
    if not captured["calls"]:
        return {"stopped": False}
    call = captured["calls"][0]
    return {"aniso_eps": call["out"], "grid": call["args"][0],
            "shapes": call["args"][1], "stopped": captured["stopped"]}


def _lane_capture(sim, monkeypatch) -> dict:
    return _first(_capture(
        lambda: sim.compute_waveguide_s_matrix(subpixel_smoothing=True,
                                               n_steps=N_STEPS),
        monkeypatch,
    ))


def _runner_capture(sim, monkeypatch) -> dict:
    return _first(_capture(
        lambda: sim.run(n_steps=N_STEPS, subpixel_smoothing=True,
                        skip_preflight=True),
        monkeypatch,
    ))


def _pad_column(captured) -> np.ndarray:
    """eps_xx down the x-hi pad on the guide centre line."""
    eps = captured["aniso_eps"][0]
    pad = int(captured["grid"].face_pads[1])
    assert pad == CPML_CELLS, pad
    jm, km = eps.shape[1] // 2, eps.shape[2] // 2
    return eps[eps.shape[0] - pad:, jm, km].astype(float)


def test_the_lane_calls_the_runners_one_implementation() -> None:
    """A source check, cheap and exact, and NOT the gate.

    It pins the shape of the call site; the pad tests below pin the array. It
    is kept because it names the thing that drifted -- a local rebuild under a
    comment claiming to mirror the runner -- but on its own it cannot tell a
    call whose result is used from one whose result is discarded.
    """
    import inspect

    from rfx.sparams import waveguide as lane

    src = inspect.getsource(lane.compute_waveguide_s_matrix)
    assert src.count("smoothed_shape_pairs(self, grid)") == 2, (
        "both smoothing sites must go through the shared helper")
    assert "for entry in self._geometry" not in src, (
        "a local shape_eps_pairs construction is back in the lane; it is the "
        "duplicate that drifted from the runners and became #1066"
    )


def test_the_pad_the_lane_hands_the_solver_carries_the_dielectric(
        monkeypatch) -> None:
    """The #831 end facet, read off the lane's own array.

    Red before the fold: every cell of the x-hi pad read 1.0 in what the lane
    passed down.
    """
    sim = _guide(with_slab=True)
    lane = _lane_capture(sim, monkeypatch)
    assert lane.get("stopped"), (
        "the lane never called compute_smoothed_eps -- this fixture is not on "
        "the smoothed lane and proves nothing about it")

    pad_column = _pad_column(lane)
    np.testing.assert_allclose(pad_column, 4.0, rtol=0, atol=1e-12)
    assert not np.any(np.isclose(pad_column, 1.0)), (
        f"the x-hi pad the LANE hands the solver is back to vacuum: "
        f"{pad_column!r} -- that is the #831 facet this test exists to keep "
        "closed"
    )


@pytest.mark.parametrize("reaches_pad", [True, False],
                         ids=["reaches_the_pad", "stops_short_of_the_pad"])
def test_lane_and_runner_hand_down_the_same_array(reaches_pad: bool,
                                                  monkeypatch) -> None:
    """Bit identity between what the lane passes to the solver and what the
    runner passes, for the same declared geometry.

    Two fresh sims from one factory rather than one sim used twice: what has
    to agree is the description, not a cached array.

    The ``reaches_the_pad`` case is the defect -- before the fold the lane
    handed 1.0 and the runner 4.0 down all 8 CPML cells. The ``stops_short``
    case is the control: a geometry touching no pad was already identical and
    must stay so, which is what says the fold changed the pad and nothing else.
    """
    lane = _lane_capture(_guide(True, slab_reaches_pad=reaches_pad),
                         monkeypatch)
    runner = _runner_capture(_guide(True, slab_reaches_pad=reaches_pad),
                             monkeypatch)
    assert lane.get("stopped") and runner.get("stopped"), (
        sorted(lane), sorted(runner))

    for axis, (lane_c, runner_c) in enumerate(zip(lane["aniso_eps"],
                                                  runner["aniso_eps"])):
        assert lane_c.shape == runner_c.shape, (axis, lane_c.shape,
                                                runner_c.shape)
        np.testing.assert_array_equal(
            lane_c, runner_c,
            err_msg=f"lane and runner disagree on component {axis}")


def test_a_guide_with_no_dielectric_is_unaffected(monkeypatch) -> None:
    """The empty reference run passes ``dielectric_shapes=[]`` and cannot carry
    a facet; the fold must not invent pairs for it.

    Captured one step earlier than the tests above, at ``smoothed_shape_pairs``
    itself: with no pairs the lane never reaches ``compute_smoothed_eps``, so a
    spy there would see nothing and the call would go on to time-step.
    """
    from rfx.geometry import smoothing as _smoothing

    captured: dict = {}
    real = _smoothing.smoothed_shape_pairs

    def _spy(sim, grid):
        captured["pairs"], captured["unextendable"] = real(sim, grid)
        raise _StopAfterCapture

    monkeypatch.setattr(_smoothing, "smoothed_shape_pairs", _spy)
    sim = _guide(with_slab=False)
    try:
        sim.compute_waveguide_s_matrix(subpixel_smoothing=True,
                                       n_steps=N_STEPS)
    except _StopAfterCapture:
        captured["stopped"] = True
    finally:
        monkeypatch.undo()

    assert captured.get("stopped"), (
        "the lane never called smoothed_shape_pairs on an empty guide")
    assert captured["pairs"] == [], captured["pairs"]
    assert captured["unextendable"] == [], captured["unextendable"]


def test_the_kottke_pec_route_carries_the_dielectric_into_the_pad(
        monkeypatch) -> None:
    """The Stage-2 site, read off its own tensor (review of PR #1131, F5).

    ``subpixel_smoothing="kottke_pec"`` does not reach
    ``compute_smoothed_eps`` at all: it feeds ``smoothed_shape_pairs`` into
    ``compute_inv_eps_tensor_diag`` instead. Every test above is blind to it,
    so restoring the #1066 defect at that site alone left them green while the
    lane handed vacuum down the whole pad on this route.

    Two calls are captured because the site makes two: the device tensor, and
    then a vacuum reference tensor that passes ``dielectric_shapes=[]`` and by
    construction cannot carry a facet. What is asserted is the array rather
    than the shape list -- ``1/4`` in the pad is the same statement about the
    solver's input that the Stage-1 tests make with ``4.0``, and it does not
    re-test ``smoothed_shape_pairs`` against its own convention.
    """
    sim = _guide(with_slab=True)
    captured = _capture(
        lambda: sim.compute_waveguide_s_matrix(
            subpixel_smoothing="kottke_pec", n_steps=N_STEPS),
        monkeypatch, attr="compute_inv_eps_tensor_diag", stop_after=2,
    )
    assert captured["stopped"], (
        "the lane never built two inverse-eps tensors -- this fixture is not "
        "on the kottke_pec route and proves nothing about it")
    assert len(captured["calls"]) == 2, len(captured["calls"])

    device = [c for c in captured["calls"]
              if c["kwargs"].get("dielectric_shapes")]
    reference = [c for c in captured["calls"]
                 if not c["kwargs"].get("dielectric_shapes")]
    assert len(device) == 1 and len(reference) == 1, (
        [sorted(c["kwargs"]) for c in captured["calls"]])

    grid = device[0]["args"][0]
    pad = int(grid.face_pads[1])
    assert pad == CPML_CELLS, pad

    inv_xx = device[0]["out"][0]
    jm, km = inv_xx.shape[1] // 2, inv_xx.shape[2] // 2
    pad_column = inv_xx[inv_xx.shape[0] - pad:, jm, km].astype(float)
    np.testing.assert_allclose(pad_column, 0.25, rtol=0, atol=1e-12)
    assert not np.any(np.isclose(pad_column, 1.0)), (
        f"the x-hi pad the kottke_pec route hands the solver is back to "
        f"vacuum: {pad_column!r} -- inverse permittivity, so 1.0 is eps_r = 1 "
        "and 0.25 is the eps_r = 4 slab continued through its own absorber"
    )

    # The reference run is vacuum on purpose; if it ever stopped being so,
    # device minus reference would cancel the slab instead of the guide.
    ref_xx = reference[0]["out"][0]
    ref_column = ref_xx[ref_xx.shape[0] - pad:, jm, km].astype(float)
    np.testing.assert_allclose(ref_column, 1.0, rtol=0, atol=1e-12)
