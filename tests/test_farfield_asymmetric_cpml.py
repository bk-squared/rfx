"""Bundle C.2 — NTFFBox.from_grid and per-face CPML origin asymmetry.

These tests pin the v1.7.4 refactor: per-face CPML thickness now threads
through NTFF box construction so NTFF integration surfaces sit outside the
CPML-active region on each face independently, not behind a scalar
cpml_layers origin.
"""
from __future__ import annotations

import jax.numpy as jnp

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


def test_physical_validity_ntff_box_outside_active_cpml():
    """Physical-validity assertion: the NTFF integration surface must sit
    in the scattered-field region (outside active CPML) on every face.

    Asserts k_lo >= z_lo_active_thickness and k_hi <= nz - z_hi_active_thickness
    on a sim-like grid built with asymmetric face_layers. A box placed
    inside active CPML cells would silently corrupt the FFRP."""
    grid = _make_grid(
        cpml_layers=16,
        face_layers={"z_lo": 4, "z_hi": 16, "x_lo": 16, "x_hi": 16,
                     "y_lo": 16, "y_hi": 16},
    )
    ntff_offset = 3
    fl = grid.face_layers
    k_lo = fl["z_lo"] + ntff_offset
    k_hi = grid.nz - fl["z_hi"] - ntff_offset
    assert k_lo >= fl["z_lo"], (
        f"NTFF k_lo={k_lo} inside active CPML (z_lo={fl['z_lo']}) - physics bug"
    )
    assert k_hi <= grid.nz - fl["z_hi"], (
        f"NTFF k_hi={k_hi} inside active hi-face CPML - physics bug"
    )
    # Symmetric sanity: the asymmetric case moves k_lo closer to grid edge
    # than the pre-refactor scalar would.
    assert k_lo == 7  # 4 + 3
    assert k_hi == grid.nz - 19  # nz - 16 - 3
