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

* The S-parameter preflight's refusal of ``run(compute_s_params=True)`` and
  ``forward(port_s11_freqs=...)`` on a coaxial-port simulation names the
  remaining lanes and not the removed one.
* Asking the preflight for the removed calculator by name is refused with
  #1212 and the replacements, not accepted and not "unknown calculator".
* No public attribute of ``Simulation``, ``rfx``, ``rfx.api``,
  ``rfx.sources`` or ``rfx.sources.coaxial_port`` carries a removed name.
* Under ``rfx/``, every source line that names ``compute_coaxial_s_matrix``
  also names #1212, so a message that still routes users there fails.

The two ``calculator="coaxial"`` checks below moved here from the deleted
``tests/unit/sparams/test_coaxial_s_matrix.py``; the fixture is the one the
line-reflection lane's own tests use, so a pass means a setup that lane runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import rfx
import rfx.api
import rfx.sources
import rfx.sources.coaxial_port
from rfx import GaussianPulse, Simulation

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


def _one_coax_port_sim() -> Simulation:
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40.0e9,
                     boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


def test_nothing_points_users_at_the_removed_coaxial_lane():
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


def test_coaxial_calculator_passes_a_single_coaxial_port():
    assert len(_one_coax_port_sim().preflight_sparameters(
        calculator="coaxial")) == 0


def test_coaxial_calculator_reports_missing_coaxial_ports():
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40.0e9,
                     boundary="cpml")
    issues = sim.preflight_sparameters(calculator="coaxial")
    assert any("No coaxial ports" in str(issue) for issue in issues)
