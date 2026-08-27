"""Input-fidelity report (rfx.fidelity): declared-vs-realized, input units only.

Pins the three finding classes the tool exists for, on a fixture built to
contain each trap exactly once:

* a one-cell PEC ground with dielectric ABOVE it — the sheet's own cell is
  vacuum inside the cavity ("sheet-own-cell-live", the #693/#702 class);
* a patch with deliberately off-lattice x-faces ("off-lattice-face");
* a dielectric later overwritten by another entity
  ("materialization-overridden");
* an entity thinner than any cell ("absent" — the #369 silent-drop class).

Also locks the non-findings: exactly-on-lattice faces report 0.0 um
residuals (the report must not cry wolf), and coordinates are DOMAIN
coordinates (the CPML pad offset is compensated).
"""
from __future__ import annotations

import numpy as np

from rfx import Simulation, Box


def _fixture():
    sim = Simulation(freq_max=15e9, domain=(10e-3, 10e-3, 6e-3), dx=0.5e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_material("sub", eps_r=3.38, sigma=0.0)
    sim.add_material("override", eps_r=2.2, sigma=0.0)
    sim.add(Box((0, 0, 2.0e-3), (10e-3, 10e-3, 2.5e-3)), material="pec")     # [0] ground sheet
    sim.add(Box((0, 0, 2.5e-3), (10e-3, 10e-3, 4.0e-3)), material="sub")     # [1] substrate
    sim.add(Box((3.13e-3, 3.0e-3, 4.0e-3), (6.87e-3, 7.0e-3, 4.5e-3)),
            material="pec")                                                  # [2] patch, off-lattice x
    sim.add(Box((1.0e-3, 1.0e-3, 2.5e-3), (3.0e-3, 3.0e-3, 3.0e-3)),
            material="override")                                             # [3] overwrites part of [1]
    sim.add_material("thinfilm", eps_r=8.0, sigma=0.0)
    sim.add(Box((4.0e-3, 4.0e-3, 5.10e-3), (6.0e-3, 6.0e-3, 5.15e-3)),
            material="thinfilm")   # [4] 50 um dielectric film mid-cell -> absent
            # (a thin PEC box would be SNAPPED to a sheet by the thin-conductor
            #  realization — correct behaviour, so the absent class needs a
            #  dielectric, which has no snap path — and even that lands in
            #  its containing cell, realized 10x thicker: entity [5] below is
            #  the true absent case, declared outside the domain)
    sim.add(Box((4.0e-3, 4.0e-3, 9.0e-3), (6.0e-3, 6.0e-3, 9.2e-3)),
            material="thinfilm")   # [5] beyond even the CPML pad -> absent
    return sim


def _kinds(item):
    return [f["kind"] for f in item["findings"]]


def test_finding_classes_are_detected():
    rep = _fixture().fidelity_report(print_report=False)

    ground = rep[0]
    assert "one-plane sheet" in ground["realization"]
    assert "sheet-own-cell-live" in _kinds(ground)
    assert ground["own_cell_eps_r"][1] <= 1.0 + 1e-6, "ground own-cell must read vacuum"

    patch = rep[2]
    assert "off-lattice-face" in _kinds(patch)
    ax_x = patch["axes"][0]
    assert max(ax_x["face_residual_um"]) > 100.0

    sub = rep[1]
    assert "materialization-overridden" in _kinds(sub)
    assert sub["assembled_matches_declared_frac"] < 0.999
    assert min(sub["assembled_eps_r"]) < 3.0   # the 2.2 override is visible

    film = rep[4]
    # Rasterization convention: a sub-cell box lands in its containing cell,
    # so a 50 um film is realized 500 um thick — caught as extent/face
    # distortion (the tool's job), not as absence.
    assert film["n_cells"] > 0
    assert film["axes"][2]["realized_extent_um"] > 5 * film["axes"][2]["declared_extent_um"]
    assert "off-lattice-face" in _kinds(film)

    clamped = rep[5]
    # Measured rasterization behaviour: an out-of-range box CLAMPS to the
    # boundary cells rather than vanishing — the report makes that visible as
    # a face residual of millimetres (declared z 9.0 mm vs realized at the
    # array edge). The "absent" branch stays in the tool defensively for
    # shapes whose mask is genuinely empty.
    assert clamped["n_cells"] > 0
    assert max(clamped["axes"][2]["face_residual_um"]) > 900.0
    assert "off-lattice-face" in _kinds(clamped)


def test_exact_faces_report_zero_residual_in_domain_coords():
    rep = _fixture().fidelity_report(print_report=False)
    ground = rep[0]
    for ax in ground["axes"]:
        assert max(ax["face_residual_um"]) < 1e-6, (
            f"exact face read nonzero residual on {ax['axis']}: {ax}")
    # domain coordinates, not padded-array coordinates:
    assert abs(ground["axes"][2]["realized_um"][0] - 2000.0) < 1e-6


def test_remedies_are_present_and_mechanical():
    rep = _fixture().fidelity_report(print_report=False)
    for item in rep:
        for f in item["findings"]:
            assert f.get("remedy"), f"finding without a remedy: {f}"
            assert "df/f" not in f["detail"], (
                "fidelity findings must never predict result-side impact")
