"""Build-time bars for the missing conductor loss of f0 films."""
import re

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.errors import AccuracyWarning, RfxRefusal
from rfx.preflight._impl import run_preflight_gate


MM = 1e-3
CODE = "f0_sheet_backed_gap"


def _sheet(sim, axis, position, *, lo=None, hi=None):
    lo = list(lo if lo is not None else (2*MM, 2*MM, 2*MM))
    hi = list(hi if hi is not None else (8*MM, 8*MM, 8*MM))
    lo[axis] = hi[axis] = position
    sim.add_thin_conductor(Box(tuple(lo), tuple(hi)), sigma_bulk=5.8e5,
                           thickness=0.2*MM, surface_impedance_f0=8e9)


def _cavity(axis, *, graded=False):
    domain = (30*MM, 12*MM, 24*MM)
    profiles = {}
    if graded:
        profiles = dict(dy_profile=np.array([1, 1, 2, 2, 2, 2, 1, 1])*MM,
                        dz_profile=np.array([1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1])*MM)
    sim = Simulation(freq_max=10e9, domain=domain, dx=MM, boundary="pec",
                     snap="declared", **profiles)
    for side, position in (("lo", MM), ("hi", domain[axis] - MM)):
        _sheet(sim, axis, position, lo=(0, 0, 0), hi=domain)
    return sim


@pytest.mark.parametrize("axis", [1, 0], ids=["y-wall-pair", "x-wall-pair"])
def test_s1_cavity_fires(axis):
    hits = _cavity(axis).preflight().by_code(CODE)
    assert len(hits) == 2
    for index, hit in enumerate(hits):
        assert f"thin_conductor[{index}]" in str(hit)
        assert "case G" in str(hit)
        assert min(float(v) for v in re.findall(r"distance ([0-9.e+-]+) m", str(hit))) == pytest.approx(MM)
        assert "E normal to the sheet" in str(hit)
        assert "add_wall_loss(faces=...) or add_wall_loss(shape=...)" in str(hit)
        assert hit.severity == "warning" and hit.scope == "object"
        assert hit.subjects
    with pytest.warns(AccuracyWarning, match="case G"):
        run_preflight_gate(hits, context="test")


def test_s1_coincident_box_face_fires():
    sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                     boundary="cpml", cpml_layers=2, snap="declared")
    sim.add(Box((2*MM, 2*MM, 2*MM), (8*MM, 8*MM, 4*MM)), material="pec")
    # Keep all 84 Box-face edges inside the sheet, away from its boundary rows.
    _sheet(sim, 2, 4*MM, lo=(MM, MM, 4*MM), hi=(9*MM, 9*MM, 4*MM))
    hit, = sim.preflight().by_code(CODE)
    assert "case C: 84 PEC-owned tangential edges" in str(hit)
    assert "sheet's loss is absent where PEC owns its edges" in str(hit)


@pytest.mark.parametrize("scene", ["patch-over-ground", "free-standing", "one-pec-face"])
def test_s2_open_side_quiet(scene):
    boundary = "cpml"
    if scene == "one-pec-face":
        boundary = BoundarySpec(x=Boundary("cpml", "cpml"),
                                y=Boundary("cpml", "cpml"),
                                z=Boundary("pec", "cpml"))
    sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                     boundary=boundary, cpml_layers=2, snap="declared")
    if scene == "patch-over-ground":
        sim.add(Box((MM, MM, 2*MM), (9*MM, 9*MM, 2*MM)), material="pec")
    _sheet(sim, 2, MM if scene == "one-pec-face" else 5*MM)
    assert sim.preflight().by_code(CODE) == []


def test_s2_leontovich_transmission_quiet():
    domain = (10*MM, 2*MM, 2*MM)
    sim = Simulation(freq_max=12e9, domain=domain, dx=0.5*MM,
                     boundary=BoundarySpec(x="cpml", y="pmc", z="pec"),
                     cpml_layers=2)
    _sheet(sim, 0, 5*MM, lo=(0, 0, 0), hi=domain)
    assert sim.preflight().by_code(CODE) == []


def test_s2_perpendicular_pec_box_walls_quiet():
    sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                     boundary="cpml", cpml_layers=2, snap="declared")
    for low, high in ((1, 3), (7, 9)):
        sim.add(Box((low*MM, MM, MM), (high*MM, 9*MM, 9*MM)), material="pec")
    _sheet(sim, 2, 5*MM, lo=(3*MM, 2*MM, 5*MM), hi=(7*MM, 8*MM, 5*MM))
    assert sim.preflight().by_code(CODE) == []


def test_s3_graded_y_cavity_fires():
    hits = _cavity(1, graded=True).preflight().by_code(CODE)
    assert len(hits) == 2
    assert all("case G" in str(hit) for hit in hits)
    for hit in hits:
        distances = sorted(float(v) for v in re.findall(r"distance ([0-9.e+-]+) m", str(hit)))
        assert distances == pytest.approx([1e-3, 10e-3])


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("position", [0, 0.2*MM, MM, 10*MM],
                         ids=["z-lo", "snaps-to-z-lo", "one-cell-inside", "z-hi"])
def test_s4_pec_domain_face_admission(graded, entry, position, monkeypatch):
    profiles = {"dz_profile": np.array([1, 1, 2, 2, 2, 1, 1])*MM} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                     boundary="pec", snap="declared", **profiles)
    _sheet(sim, 2, position)

    class Admitted(Exception):
        pass

    from rfx.model import thin_conductors
    original = thin_conductors.fold_thin_conductor

    def build_only(*args, **kwargs):
        original(*args, **kwargs)
        raise Admitted

    monkeypatch.setattr(thin_conductors, "fold_thin_conductor", build_only)
    kwargs = {"compute_s_params": False} if entry == "run" else {}
    call = lambda: getattr(sim, entry)(n_steps=1, skip_preflight=True, **kwargs)
    if position == MM:
        with pytest.raises(Admitted):
            call()
    else:
        with pytest.raises(RfxRefusal) as caught:
            call()
        face = "z_hi" if position == 10*MM else "z_lo"
        assert str(caught.value) == (
            f"thin_conductor[0] ({face}): an f0 surface-impedance sheet on a PEC domain face (or an absorber face's outer node plane) "
            "carries no current (the face's PEC owns its edges), so its loss would "
            "be dropped; for loss on that wall use add_wall_loss(faces=...) "
            "or add_wall_loss(shape=...).")


def test_other_f0_planes_are_conductors():
    sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                     boundary="cpml", cpml_layers=2, snap="declared")
    for position in (3, 5, 7):
        _sheet(sim, 2, position*MM)
    hit, = sim.preflight().by_code(CODE)
    assert "thin_conductor[1]" in str(hit) and "case G" in str(hit)
    assert str(hit).count("distance 0.002 m") == 2


def test_periodic_seam_stops_scan():
    sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                     boundary=BoundarySpec(x=Boundary("cpml", "cpml"),
                                           y=Boundary("cpml", "cpml"),
                                           z=Boundary("periodic", "periodic")),
                     cpml_layers=2, snap="declared")
    _sheet(sim, 2, 5*MM)
    sim.add(Box((2*MM, 2*MM, 3*MM), (8*MM, 8*MM, 3*MM)), material="pec")
    assert sim.preflight().by_code(CODE) == []


@pytest.mark.parametrize("kind", ["pec-sheet", "pec-volume", "dc-film", "lossy-volume"])
def test_conductor_set(kind):
    sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                     boundary="cpml", cpml_layers=2, snap="declared")
    _sheet(sim, 2, 5*MM)
    if kind == "lossy-volume":
        sim.add_material("lossy", eps_r=1.0, sigma=1e3)
    for low, high in ((2, 3), (7, 8)):
        shape = Box((2*MM, 2*MM, low*MM),
                    (8*MM, 8*MM, (low if kind in ("pec-sheet", "dc-film") else high)*MM))
        if kind == "dc-film":
            sim.add_thin_conductor(shape, sigma_bulk=1e3, thickness=0.1*MM)
        else:
            sim.add(shape, material="lossy" if kind == "lossy-volume" else "pec")
    hits = sim.preflight().by_code(CODE)
    if kind in ("pec-sheet", "pec-volume"):
        assert len(hits) == 1 and "case G" in str(hits[0])
    else:
        assert hits == []


def test_conductors_must_share_a_footprint_column():
    sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                     boundary="cpml", cpml_layers=2, snap="declared")
    _sheet(sim, 2, 5*MM)
    sim.add(Box((2*MM, 2*MM, 3*MM), (4*MM, 8*MM, 3*MM)), material="pec")
    sim.add(Box((6*MM, 2*MM, 7*MM), (8*MM, 8*MM, 7*MM)), material="pec")
    assert sim.preflight().by_code(CODE) == []


def test_traced_f0_conductivity_still_warns_without_stepping():
    import jax
    import jax.numpy as jnp
    observed = []

    def build(sigma):
        sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                         boundary="pec", snap="declared")
        sim.add_thin_conductor(Box((2*MM, 2*MM, 5*MM), (8*MM, 8*MM, 5*MM)),
                               sigma_bulk=sigma, thickness=0.2*MM, surface_impedance_f0=8e9)
        observed.extend(sim.preflight().by_code(CODE))
        return sigma

    jax.grad(build)(jnp.float32(5.8e5))
    hit, = observed
    assert "case G" in str(hit)
    assert "thin_conductor[0]" in str(hit)


def test_traced_mesh_skips_without_a_concrete_footprint():
    import jax
    import jax.numpy as jnp

    observed = []

    def build(profile):
        sim = Simulation(freq_max=10e9, domain=(10*MM,)*3, dx=MM,
                         boundary="pec", snap="declared", dy_profile=profile,
                         nominal_profiles={"y": np.full(10, MM)},
                         dt=1e-13, dt_min_cell=MM/2)
        _sheet(sim, 1, 5*MM)
        report = sim.preflight()
        observed.append((sim._campaign_ctx().error, report.by_code(CODE)))
        return jnp.sum(profile)

    jax.jvp(build, (jnp.full(10, MM),), (jnp.ones(10),))
    assert observed == [("traced-mesh", [])]
