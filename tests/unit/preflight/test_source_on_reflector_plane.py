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

import numpy as np
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


@pytest.mark.parametrize("face,position,component", [
    ("x_lo", (0.0, CY, CZ), "ez"),
    ("x_hi", (DOMAIN[0], CY, CZ), "ey"),
    ("y_lo", (0.01, 0.0, CZ), "ez"),
    ("y_hi", (0.01, DOMAIN[1], CZ), "ex"),
    ("z_lo", (0.01, CY, 0.0), "ex"),
    ("z_hi", (0.01, CY, DOMAIN[2]), "ey"),
])
def test_tangential_e_on_magnetic_plane_reports_coupling(face, position, component):
    """The advisory distinguishes propagation on the plane from into the volume."""
    sim = _sim(BoundarySpec.uniform("pmc"), position, component)
    msg = _issue(sim, "source_decoupled")
    assert f"at {position} m (component={component})" in msg
    assert f"sits on the magnetic-wall plane {face}" in msg
    assert "On a single-device Yee run the wall is solved half a cell inside this face" in msg
    assert "E nodes on the plane form a sheet coupled only to itself" in msg
    assert "a line drawn entirely in the plane (a one-cell-wide model) carries its wave" in msg
    assert "nothing launched here reaches the volume off the plane" in msg
    assert "including the half of a line that the plane cuts along its centre" in msg
    assert (
        "The distributed lanes do not realise a magnetic wall: with no absorbing "
        "face the plane is shorted, and with absorbing faces the cells next to "
        "it absorb, so a source one cell off reaches the volume 65–75 dB low; "
        "use a single-device run."
    ) in msg
    assert "To radiate into the volume, place the source one cell (1mm) off the plane" in msg
    assert "no wave radiates" not in msg
    assert "silent zero field" not in msg
    assert "#1221" not in msg


def test_adi_tangential_e_on_magnetic_plane_reports_electric_wall():
    """ADI solves a face declared magnetic as an electric wall at any source offset."""
    position = (0.0, CY, CZ)
    sim = _sim(BoundarySpec.uniform("pmc"), position, "ez", solver="adi")
    assert _issue(sim, "source_decoupled") == (
        f"Source/port at {position} m (component=ez) sits on the "
        "magnetic-wall plane x_lo. solver='adi' does not realise a magnetic wall: "
        "it solves this face as an electric wall wherever the source sits. "
        "Use solver='yee' for a magnetic wall."
    )


def _graded_magnetic_face_sim(axis, side, on_plane):
    profile = np.array([1e-3] * 6 + [0.25e-3] * 8)
    if side == "lo":
        profile = profile[::-1]
    ax_i = "xyz".index(axis)
    domain = [0.012, 0.012, 0.012]
    domain[ax_i] = float(profile.sum())
    boundaries = dict.fromkeys("xyz", "pec")
    boundaries[axis] = Boundary(**{side: "pmc", "hi" if side == "lo" else "lo": "pec"})
    position = [0.006, 0.006, 0.006]
    position[ax_i] = 0.0 if side == "lo" else domain[ax_i]
    if not on_plane:
        position[ax_i] += 0.25e-3 if side == "lo" else -0.25e-3
    sim = Simulation(
        freq_max=20e9, domain=tuple(domain), dx=DX,
        boundary=BoundarySpec(**boundaries), **{f"d{axis}_profile": profile},
    )
    sim.add_source(tuple(position), "ey" if axis == "x" else "ex",
                   amplitude_kind="field")
    return sim


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("side", ["lo", "hi"])
def test_source_one_fine_cell_inside_graded_magnetic_face_is_silent(axis, side):
    """A source 0.25 mm inside a fine face is in the volume, not on the face."""
    sim = _graded_magnetic_face_sim(axis, side, on_plane=False)
    assert "source_decoupled" not in _codes(sim)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("side", ["lo", "hi"])
def test_source_on_graded_magnetic_face_quotes_adjacent_cell(axis, side):
    """The fine-face source remedy is 0.25 mm on every graded axis and side."""
    sim = _graded_magnetic_face_sim(axis, side, on_plane=True)
    msg = _issue(sim, "source_decoupled")
    assert f"sits on the magnetic-wall plane {axis}_{side}" in msg
    assert "one cell (250µm) off the plane" in msg


def test_source_one_cell_inside_magnetic_plane_is_silent():
    """The volume-launch remedy clears the source-placement advisory."""
    sim = _sim(BoundarySpec.uniform("pmc"), (DX, CY, CZ), "ez")
    assert "source_decoupled" not in _codes(sim)


def test_normal_e_on_magnetic_plane_keeps_its_symmetry_advice():
    """Normal E retains its existing mirror-symmetry advisory."""
    sim = _sim(BoundarySpec.uniform("pmc"), (0.0, CY, CZ), "ex")
    msg = _issue(sim, "source_decoupled")
    assert f"at {(0.0, CY, CZ)} m (component=ex)" in msg
    assert "sits on the magnetic-wall plane x_lo and drives the NORMAL E component" in msg
    assert (
        "PMC imposes odd symmetry on normal E (it must be zero at the plane), "
        "so the source fights the mirror image."
    ) in msg
    assert "tangential E source offset by one cell (1mm) off the plane" in msg
