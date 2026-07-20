"""Bundle C.2 — NTFFBox.from_grid and per-face CPML origin asymmetry.

These tests pin the 2026-04 per-face refactor: per-face CPML thickness now threads
through NTFF box construction so NTFF integration surfaces sit outside the
CPML-active region on each face independently, not behind a scalar
cpml_layers origin.
"""
from __future__ import annotations

import jax.numpy as jnp

from rfx import Simulation
from rfx.grid import Grid
from rfx.farfield import NTFFBox


def _make_grid(*, cpml_layers=8, face_layers=None, domain=(0.04, 0.04, 0.04)):
    return Grid(
        freq_max=5e9,
        domain=domain,
        dx=1e-3,
        cpml_layers=cpml_layers,
        face_layers=face_layers,
    )


def test_symmetric_bit_identity_ntffbox_from_grid():
    """Symmetric face_layers -> from_grid produces cpml_lo_* == cpml_hi_*
    == cpml_layers on every face. Refactor is a no-op on the common case."""
    grid = _make_grid(cpml_layers=8)  # face_layers defaults to {f: 8 for f in faces}
    box = NTFFBox.from_grid(
        grid,
        i_lo=10, i_hi=grid.nx - 10,
        j_lo=10, j_hi=grid.ny - 10,
        k_lo=10, k_hi=grid.nz - 10,
        freqs=jnp.array([3e9], dtype=jnp.float32),
    )
    for field in ("cpml_lo_x", "cpml_hi_x", "cpml_lo_y", "cpml_hi_y",
                  "cpml_lo_z", "cpml_hi_z"):
        assert getattr(box, field) == 8, f"{field} = {getattr(box, field)} != 8"


def test_asymmetric_integer_motion_under_from_grid():
    """Asymmetric face_layers (z_lo=4, z_hi=16) - from_grid stores the
    per-face values verbatim so downstream NTFF-offset arithmetic can
    place the box at lo+offset / nz-hi-offset instead of cpml+offset
    symmetrically."""
    grid = _make_grid(
        cpml_layers=16,
        face_layers={"z_lo": 4, "z_hi": 16, "x_lo": 16, "x_hi": 16,
                     "y_lo": 16, "y_hi": 16},
    )
    box = NTFFBox.from_grid(
        grid,
        i_lo=5, i_hi=grid.nx - 5,
        j_lo=5, j_hi=grid.ny - 5,
        k_lo=5, k_hi=grid.nz - 5,
        freqs=jnp.array([3e9], dtype=jnp.float32),
    )
    assert box.cpml_lo_z == 4
    assert box.cpml_hi_z == 16
    assert box.cpml_lo_x == 16
    assert box.cpml_hi_x == 16


def _preflight_codes(corner_lo, corner_hi, *, cpml_layers=8, dx=2e-3,
                     domain=(0.120, 0.120, 0.120), freq=10e9):
    """Build a real Simulation with a CPML boundary and an NTFF box, run the
    production preflight, and return the set of issue codes it emitted."""
    sim = Simulation(freq_max=freq, domain=domain, dx=dx,
                     boundary="cpml", cpml_layers=cpml_layers)
    sim.add_source((domain[0] / 2, domain[1] / 2, domain[2] / 2), "ez")
    sim.add_ntff_box(corner_lo, corner_hi, freqs=(freq,))
    report = sim.preflight()
    return {getattr(i, "code", None) for i in report}


def test_ntff_box_outside_cpml_has_no_absorber_overlap():
    """Positive control: an NTFF box wholly inside the interior must NOT
    trip the production absorber-overlap validator.

    Exercises the real ``_validate_cfg_ntff_absorber_overlap`` reached by
    ``sim.preflight()`` (rfx/api/_preflight.py) — the production code that
    guards against an NTFF integration surface sitting in active CPML."""
    # cpml thickness = 8 * 2 mm = 16 mm; box spans [30, 90] mm on every axis.
    codes = _preflight_codes((0.030, 0.030, 0.030), (0.090, 0.090, 0.090))
    assert "absorber_overlap" not in codes, (
        f"clean interior box wrongly flagged as absorber overlap; codes={codes}"
    )


def test_ntff_box_inside_cpml_trips_absorber_overlap_lo_and_hi():
    """Negative control that MUST fail if the production placement guard is
    removed: an NTFF box whose lo (or hi) corner is inside the 16 mm CPML
    region must trip ``absorber_overlap`` via the real preflight path.

    This replaces a prior tautology (``k_lo = z_lo + offset; assert
    k_lo >= z_lo``) that re-asserted the test's own arithmetic and could not
    detect a real inside-CPML placement bug."""
    # lo corner x = 5 mm < 16 mm CPML -> extends into the lo-face absorber.
    codes_lo = _preflight_codes((0.005, 0.030, 0.030), (0.090, 0.090, 0.090))
    assert "absorber_overlap" in codes_lo, (
        f"lo-face inside-CPML NTFF box was NOT flagged; codes={codes_lo}"
    )
    # hi corner z = 118 mm > 120 - 16 = 104 mm -> extends into the hi-face
    # absorber. Verifies the guard is two-sided, not just lo-face.
    codes_hi = _preflight_codes((0.030, 0.030, 0.030), (0.090, 0.090, 0.118))
    assert "absorber_overlap" in codes_hi, (
        f"hi-face inside-CPML NTFF box was NOT flagged; codes={codes_hi}"
    )


def test_ntffbox_from_grid_places_faces_outside_per_face_active_cpml():
    """Per-face bookkeeping check on the real ``NTFFBox.from_grid`` output:
    the constructed box's stored per-face CPML layer counts must leave the
    requested integration indices outside the active absorber on each face,
    read from the box's OWN production-populated fields (not recomputed here).

    Asymmetric face_layers (z_lo=4, z_hi=16) means the box may legally sit
    closer to the thin lo face than the thick hi face."""
    grid = _make_grid(
        cpml_layers=16,
        face_layers={"z_lo": 4, "z_hi": 16, "x_lo": 16, "x_hi": 16,
                     "y_lo": 16, "y_hi": 16},
    )
    k_lo, k_hi = 6, grid.nz - 18
    box = NTFFBox.from_grid(
        grid,
        i_lo=18, i_hi=grid.nx - 18,
        j_lo=18, j_hi=grid.ny - 18,
        k_lo=k_lo, k_hi=k_hi,
        freqs=jnp.array([3e9], dtype=jnp.float32),
    )
    # The box stores the per-face active-CPML thickness verbatim from the
    # grid; the requested indices must clear it on each z face independently.
    assert k_lo >= box.cpml_lo_z, (
        f"k_lo={k_lo} sits inside active lo-z CPML (={box.cpml_lo_z})"
    )
    assert k_hi <= grid.nz - box.cpml_hi_z, (
        f"k_hi={k_hi} sits inside active hi-z CPML (={box.cpml_hi_z})"
    )
    # Asymmetry is real: the thin lo face (4) admits a closer box than the
    # thick hi face (16) — a genuine per-face property, not tautological.
    assert box.cpml_lo_z == 4
    assert box.cpml_hi_z == 16
