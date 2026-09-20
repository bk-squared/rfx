"""P1.6 ``source_decoupled`` -- which faces count as a reflector (issue #1075).

``_validate_cfg_source_on_reflector_plane`` (``rfx/preflight/sources.py``) has
had its four-way component rule since it was written, and no behavioural test:
the only committed witness was one snapshot fixture, built on a per-face
``BoundarySpec``. That hid a hole. The check walked
``Simulation._pec_faces``, which is populated ONLY by an explicit
``pec_faces=`` kwarg or by a per-face ``BoundarySpec``. The scalar
``boundary="pec"`` -- a PEC box on all six faces, realized as
``apply_pec(axes=pec_axes)`` -- leaves that set empty, so the loop body never
ran and the advisory was silently skipped on the commonest way of asking for a
PEC wall (measured: "All checks passed", see
``scripts/diagnostics/issue1075_source_on_face.py`` arm A).

This file gates both halves: the face set (the #1075 fix) and the component
rule it has always had, so a future narrowing of either fails here rather than
in a snapshot diff. The message assertions are deliberately about the LANE
SPLIT and not about the whole string -- the exact text is pinned by
``tests/locks/test_preflight_split_snapshot.py``, and duplicating it here would
make one wording change red in two places for no extra information.
"""

from __future__ import annotations

import warnings

import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

DX = 1e-3
DOMAIN = (0.02, 0.015, 0.015)
CY = DOMAIN[1] / 2.0
CZ = DOMAIN[2] / 2.0


def _codes(sim) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    return sorted(i.code for i in report.issues if i.code)


def _issue(sim, code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    hits = [i for i in report.issues if i.code == code]
    assert len(hits) == 1, f"expected exactly one {code}, got {len(hits)}"
    # PreflightIssue subclasses str -- the issue IS its message.
    return str(hits[0])


def _sim(boundary, position, component, **kw):
    with warnings.catch_warnings():
        # The #571 amplitude_kind DeprecationWarning and the pec_faces= one are
        # not what this file measures.
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX,
                         boundary=boundary, **kw)
        sim.add_source(position, component, amplitude_kind="field")
    return sim


# ---------------------------------------------------------------------------
# The #1075 gap: scalar boundary="pec"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("face,position,component", [
    ("x_lo", (0.0, CY, CZ), "ez"),
    ("x_lo", (0.0, CY, CZ), "ey"),
    ("x_hi", (DOMAIN[0], CY, CZ), "ez"),
    ("y_lo", (0.01, 0.0, CZ), "ez"),
    ("z_lo", (0.01, CY, 0.0), "ex"),
    ("z_hi", (0.01, CY, DOMAIN[2]), "ex"),
])
def test_tangential_e_on_whole_boundary_pec_face_fires(face, position,
                                                       component):
    """All six faces of a scalar ``boundary="pec"`` box are reflectors.

    RED before #1075 on every row: the advisory was skipped entirely, so
    ``_codes`` came back without ``source_decoupled``.
    """
    sim = _sim("pec", position, component, cpml_layers=0)
    assert "source_decoupled" in _codes(sim)
    assert face in _issue(sim, "source_decoupled")


def test_normal_e_on_whole_boundary_pec_face_is_silent():
    """Normal E is the legitimate way to drive a PEC mirror.

    ``ex`` on the ``x_lo`` plane is NORMAL to that face, so the component rule
    must stay quiet even though the face is now in the reflector set. This is
    the test that would catch the #1075 fix having widened the RULE rather
    than the FACE SET.
    """
    sim = _sim("pec", (0.0, CY, CZ), "ex", cpml_layers=0)
    assert "source_decoupled" not in _codes(sim)


def test_source_one_cell_inside_whole_boundary_pec_is_silent():
    """One cell off the wall is the remedy the advisory recommends."""
    sim = _sim("pec", (DX, CY, CZ), "ez", cpml_layers=0)
    assert "source_decoupled" not in _codes(sim)


def test_periodic_axis_is_not_a_reflector():
    """A ``periodic`` face must not be read as PEC.

    The fix derives the face set from ``Simulation._boundary_spec``, whose
    periodic axes carry the ``periodic`` token, so they are excluded by
    construction rather than by a special case. ``ez`` at ``x = 0`` with x
    periodic is a wrap-around plane, not a wall.
    """
    sim = _sim(BoundarySpec(x="periodic", y="pec", z="pec"),
               (0.0, CY, CZ), "ez", cpml_layers=0)
    assert "source_decoupled" not in _codes(sim)


def test_strict_preflight_raises_on_whole_boundary_pec_face():
    """``preflight(strict=True)`` escalates every issue, this one included."""
    sim = _sim("pec", (0.0, CY, CZ), "ez", cpml_layers=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError,
                           match="sits on the PEC x_lo plane"):
            sim.preflight(strict=True)


# ---------------------------------------------------------------------------
# Regressions: the two spellings that already worked must keep working
# ---------------------------------------------------------------------------

def test_pec_faces_kwarg_still_fires():
    """The deprecated ``pec_faces=`` kwarg path, unchanged by #1075."""
    sim = _sim("cpml", (0.01, CY, 0.0), "ex",
               cpml_layers=6, pec_faces={"z_lo"})
    assert "source_decoupled" in _codes(sim)


def test_per_face_boundary_spec_still_fires():
    """The per-face ``BoundarySpec`` path, unchanged by #1075."""
    sim = _sim(BoundarySpec(x="cpml", y="cpml",
                            z=Boundary(lo="pec", hi="cpml")),
               (0.01, CY, 0.0), "ex", cpml_layers=6)
    assert "source_decoupled" in _codes(sim)


def test_two_spellings_of_one_wall_agree():
    """The same physical wall, asked for both ways, renders the same advice.

    This is the invariant #1075 is really about: a caller should not have to
    know which constructor argument they used to find out that their source is
    on a conductor. Only the face NAME may differ, and here it does not.
    """
    scalar = _issue(_sim("pec", (0.0, CY, CZ), "ez", cpml_layers=0),
                    "source_decoupled")
    spec = _issue(
        _sim(BoundarySpec(x=Boundary(lo="pec", hi="pec"), y="pec", z="pec"),
             (0.0, CY, CZ), "ez", cpml_layers=0),
        "source_decoupled")
    assert scalar == spec


# ---------------------------------------------------------------------------
# The message has to be true on BOTH lanes (#1075)
# ---------------------------------------------------------------------------

def test_message_states_the_lane_split():
    """It used to say "silently discarded", which is false on one lane.

    Measured (``scripts/diagnostics/issue1075_source_on_face.py``): the
    distributed lanes return a probe peak of exactly 0 for this placement --
    they apply the PEC face after injection since #1041/#1055 -- while the
    single-device lane, which applies it before injection, returns 8.92558e5
    against 4.00560e6 for the same source one cell inside. Both halves have to
    be in the text, or the advisory sends a reader looking for a zero they
    will not find.
    """
    msg = _issue(_sim("pec", (0.0, CY, CZ), "ez", cpml_layers=0),
                 "source_decoupled")
    assert "distributed" in msg
    assert "single-device" in msg
    assert "exactly zero" in msg
    assert "numerically inconsistent" in msg
    assert "silently discarded" not in msg
    # The remedy survives the rewrite.
    assert "offset" in msg and "one cell" in msg
