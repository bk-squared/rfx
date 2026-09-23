"""Nothing a user can reach points at the removed single-plane coaxial lane.

``Simulation.compute_coaxial_s_matrix`` measured V/I on one plane of a coax
stub closed in a PEC box, with the outer-conductor wall drawn one cell inside
the declared radius; its own docstring recorded |S11| > 1 on a lossless short.
It was removed in #1212 together with the geometry helper it was built on
(``setup_coaxial_port``) and the termination builders only it consumed. A
coaxial port's S-parameters now come from ``compute_coaxial_line_reflection``
(one port against a short, open or resistive load) and
``compute_coaxial_two_port`` (through line).

The one invariant pinned here: after the removal, no runtime message and no
public attribute sends a user to the removed API.

* The errors ``run()`` and ``forward()`` raise on a coaxial port, and the
  S-parameter preflight's refusal of ``run(compute_s_params=True)`` and
  ``forward(port_s11_freqs=...)``, name the remaining lanes and neither the
  removed lane nor the removed ``rfx.sources.coaxial_port`` helpers.
* Asking the preflight for the removed calculator by name is refused with
  #1212 and the replacements, not accepted and not "unknown calculator".
* No public attribute of ``Simulation``, ``rfx``, ``rfx.api``,
  ``rfx.sources`` or ``rfx.sources.coaxial_port`` carries a removed name.
* Under ``rfx/``, every source line that names ``compute_coaxial_s_matrix``
  also names #1212, so a message that still routes users there fails.

``preflight_sparameters(calculator="coaxial")`` used to mirror the removed
lane's checks, so it passed a PEC box and several ports -- setups both
remaining lanes refuse before they build a grid. The parity tests below run
the preflight AND both lanes on the same setups: the preflight must refuse
exactly where both lanes refuse, and pass where both accept. The lanes are
called for real; only their grid build is replaced by a sentinel, so
"accepted" means every guard passed and no FDTD runs. (The two
``calculator="coaxial"`` checks moved here from the deleted
``tests/unit/sparams/test_coaxial_s_matrix.py``.)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import rfx
import rfx.api
import rfx.sources
import rfx.sources.coaxial_port
from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

RFX_ROOT = Path(rfx.__file__).resolve().parent
REMOVED_LANE = "compute_coaxial_s_matrix"
VALIDATED_LANES = ("compute_coaxial_line_reflection", "compute_coaxial_two_port")

REMOVED_NAMES = (
    (Simulation, (REMOVED_LANE, "add_coaxial_matched_load",
                  "add_coaxial_open_termination", "add_coaxial_pec_end_cap")),
    (rfx, ("CoaxialSMatrixResult",)),
    (rfx.api, ("CoaxialSMatrixResult",)),
    (rfx.sources, ("setup_coaxial_port", "add_coaxial_matched_termination",
                   "add_coaxial_open_termination", "add_coaxial_pec_end_cap",
                   "make_coaxial_port_source")),
    (rfx.sources.coaxial_port, ("setup_coaxial_port",
                                "add_coaxial_matched_termination",
                                "add_coaxial_open_termination",
                                "add_coaxial_pec_end_cap",
                                "make_coaxial_port_source")),
)


ONE_PORT = ((0.020, "top"),)
THREE_PORTS = ((0.010, "top"), (0.020, "top"), (0.030, "top"))


def _coax_sim(boundary="cpml", ports=ONE_PORT, **kwargs) -> Simulation:
    """The line-reflection lane's own test geometry; variations by argument."""
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40.0e9,
                     boundary=boundary, **kwargs)
    for z, face in ports:
        sim.add_coaxial_port((0.004, 0.004, z), face=face, pin_length=5.0e-3,
                             waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


def _one_coax_port_sim() -> Simulation:
    return _coax_sim()


def test_nothing_points_users_at_the_removed_coaxial_lane():
    for entry, call in (("run", lambda sim: sim.run(n_steps=1)),
                        ("forward", lambda sim: sim.forward(n_steps=1))):
        with pytest.raises(NotImplementedError) as exc:
            call(_one_coax_port_sim())
        text = str(exc.value)
        for gone in (REMOVED_LANE, "rfx.sources.coaxial_port"):
            assert gone not in text, (
                f"{entry}() on a coaxial port still points at {gone}: {text}")
        for lane in VALIDATED_LANES:
            assert lane in text, f"{entry}() does not name {lane}: {text}"

    for calculator in ("run", "forward"):
        errors = _one_coax_port_sim().preflight_sparameters(
            calculator=calculator).errors
        assert errors, (
            f"preflight_sparameters(calculator={calculator!r}) accepted a "
            "coaxial-port simulation; it must refuse and name the coaxial lanes")
        text = " ".join(str(e) for e in errors)
        assert REMOVED_LANE not in text, (
            f"the {calculator!r} refusal still sends users to the removed "
            f"lane: {text}")
        for lane in VALIDATED_LANES:
            assert lane in text, (
                f"the {calculator!r} refusal does not name {lane}: {text}")

    with pytest.raises(ValueError, match="#1212") as exc:
        _one_coax_port_sim().preflight_sparameters(calculator=REMOVED_LANE)
    for lane in VALIDATED_LANES:
        assert lane in str(exc.value), str(exc.value)

    for owner, names in REMOVED_NAMES:
        still_there = [n for n in names if hasattr(owner, n)]
        assert not still_there, (
            f"{getattr(owner, '__name__', owner)} still exposes {still_there}, "
            "removed in #1212")

    offenders = []
    for path in sorted(RFX_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if REMOVED_LANE in line and "1212" not in line:
                offenders.append(
                    f"{path.relative_to(RFX_ROOT.parent)}:{lineno}: "
                    f"{line.strip()}")
    assert not offenders, (
        "lines under rfx/ name the removed lane without naming its removal "
        "(#1212):\n  " + "\n  ".join(offenders))


class _ReachedTheSolve(Exception):
    """Raised by the patched ``_build_grid``: the lane passed every guard."""


def _lane_verdict(sim: Simulation, lane: str, monkeypatch) -> str:
    def _stop(self, *args, **kwargs):
        raise _ReachedTheSolve

    monkeypatch.setattr(Simulation, "_build_grid", _stop)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(sim, lane)(n_steps=1, n_freqs=1)
    except _ReachedTheSolve:
        return "accepted"
    except (ValueError, NotImplementedError) as exc:
        return f"refused: {exc}"
    finally:
        monkeypatch.undo()
    raise AssertionError(f"{lane} returned without building a grid")


def _with(sim: Simulation, register) -> Simulation:
    register(sim)
    return sim


# (setup, the lane requirement the preflight must name)
REFUSED_BY_BOTH_LANES = {
    "pec-one-port": (lambda: _coax_sim(boundary="pec"),
                     ["boundary='cpml'"]),
    "cpml-three-ports": (lambda: _coax_sim(ports=THREE_PORTS),
                         ["exactly one add_coaxial_port() (3 registered)"]),
    "pec-three-ports": (lambda: _coax_sim(boundary="pec", ports=THREE_PORTS),
                        ["boundary='cpml'", "exactly one add_coaxial_port()"]),
    "bottom-face": (lambda: _coax_sim(ports=((0.020, "bottom"),)),
                    ["face='top'"]),
    "pec-z-face": (lambda: _coax_sim(boundary=BoundarySpec(
        x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"))),
                   ["both z faces"]),
    "pec-side-walls": (lambda: _coax_sim(boundary=BoundarySpec(
        x="pec", y="pec", z="cpml")),
                       ["all six boundary faces"]),
    "graded-z": (lambda: _coax_sim(dz_profile=np.full(40, 1.0e-3)),
                 ["uniform grid"]),
    "registered-geometry": (lambda: _with(_coax_sim(), lambda s: (
        s.add_material("fill", eps_r=2.0),
        s.add(Box((0.001, 0.001, 0.010), (0.002, 0.002, 0.011)),
              material="fill"))),
                            ["registered geometry"]),
    "registered-probe": (lambda: _with(_coax_sim(), lambda s: s.add_probe(
        (0.004, 0.004, 0.020))),
                         ["probes"]),
    "lumped-port-family": (lambda: _with(_coax_sim(), lambda s: s.add_port(
        (0.002, 0.002, 0.010), "ez", impedance=50.0)),
                           ["add_port"]),
}


@pytest.mark.parametrize("case", sorted(REFUSED_BY_BOTH_LANES))
def test_coaxial_calculator_refuses_what_both_lanes_refuse(case, monkeypatch):
    build, requirements = REFUSED_BY_BOTH_LANES[case]
    for lane in VALIDATED_LANES:
        verdict = _lane_verdict(build(), lane, monkeypatch)
        assert verdict.startswith("refused"), (
            f"{case}: {lane} accepts this setup, so it is not a refusal case "
            f"to pin ({verdict})")
    errors = build().preflight_sparameters(calculator="coaxial").errors
    assert errors, (
        f"{case}: calculator='coaxial' passed a setup both coaxial lanes "
        "refuse before any solve")
    text = " ".join(str(e) for e in errors)
    for requirement in requirements:
        assert requirement in text, (
            f"{case}: the refusal does not name the failed requirement "
            f"{requirement!r}: {text}")


def test_coaxial_calculator_accepts_what_both_lanes_accept(monkeypatch):
    for lane in VALIDATED_LANES:
        verdict = _lane_verdict(_one_coax_port_sim(), lane, monkeypatch)
        assert verdict == "accepted", f"{lane}: {verdict}"
    assert len(_one_coax_port_sim().preflight_sparameters(
        calculator="coaxial")) == 0


def test_coaxial_calculator_reports_missing_coaxial_ports():
    issues = _coax_sim(ports=()).preflight_sparameters(calculator="coaxial")
    assert any("No coaxial ports" in str(issue) for issue in issues)
