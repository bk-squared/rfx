"""Drawn extent equals realized extent, measured against an analytic oracle.

Every other check of the #931 lattice ownership contract asks rfx where it
put the metal and compares that against the rule.  Those checks share a
premise with the thing they test: if ``realized_pec_edge_masks`` and the
assertion helper both moved the same way, they would agree and both be
wrong.  That is exactly the failure #931 exists to close -- the pre-contract
asserts counted masked node planes, so a body realized one cell short passed
every one of them for months.

This file closes the loop with an oracle that never touches rfx's
realization code: the closed-form TE101 resonance of a rectangular PEC
cavity,

    f_101 = (c / 2) * sqrt((1/a)^2 + (1/d)^2)

computed from the DRAWN geometry.  A PEC slab across the floor of a box
shortens the cavity, so the resonance moves; where the wall actually lands
is measured by the peak of the FDTD spectrum and compared against the
number the drawn dimensions predict.  If a conductor realizes one cell away
from where it was declared, the measured peak misses the analytic one by
roughly one cell in ``d``, which on this mesh is about 5 %.

Provenance: written by the independent fresh-eyes reviewer of the #931
branch (2026-09-08) as a witness rather than a test, because that review was
read-only.  Reproduced here and promoted, with its measured baseline:

    case                          branch      origin/main
    empty box, machinery control  -0.050 %    -0.050 %
    volume slab z 0..4 mm         -0.078 %    -5.024 %
    volume slab z 0..6 mm         -0.119 %    -5.668 %
    sheet at z = 4 mm             -0.078 %    -0.078 %
    volume slab z 26..30 mm       -0.078 %    -0.078 %

The empty box carries the numerical dispersion floor, so the residual on
every closed case is that floor and not a placement error.  On ``main`` the
two VOLUME cases sit a full cell short at the hi face -- the missing far
wall -- while the SHEET case is already right, which is the whole diagnosis
in one table: the old rule was a correct sheet rule attached to a volume
primitive.

The runner is ~6 s for all five cases, so this stays in the default lane.
The standalone script, with the ``main`` column, is
``scripts/diagnostics/lattice_ownership_witness/cavity_witness.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation

C0 = 299792458.0
DX = 2e-3
AX, AY, AZ = 40e-3, 20e-3, 30e-3

# Numerical dispersion on this mesh, measured on the empty box: -0.050 %.
# The gate is a few times that, tight enough that a one-cell placement error
# (about 5 % here, 60x the gate) cannot hide inside it.
MAX_ABS_ERR_PCT = 0.15


def _f_te101(a: float, d: float) -> float:
    """Analytic TE101 of a rectangular PEC cavity. No rfx involved."""
    return 0.5 * C0 * np.sqrt((1.0 / a) ** 2 + (1.0 / d) ** 2)


def _peak_freq(build) -> float:
    """Drive an off-centre Ey pulse, read the spectral peak of the ring-down."""
    sim = Simulation(freq_max=12e9, domain=(AX, AY, AZ), dx=DX, boundary="pec")
    build(sim)
    sim.add_source((9e-3, 10e-3, 11e-3), "ey", amplitude_kind="field")
    sim.add_probe((13e-3, 10e-3, 17e-3), "ey")
    result = sim.run(n_steps=6000, skip_preflight=True, compute_s_params=False)
    trace = np.asarray(result.time_series)[:, 0].astype(np.float64)
    dt = float(sim._build_grid().dt)
    n_fft = 1 << 16
    spectrum = np.abs(np.fft.rfft(trace * np.hanning(trace.size), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt)
    band = (freqs > 3e9) & (freqs < 11e9)
    return float(freqs[band][np.argmax(spectrum[band])])


# label -> (builder, the cavity height the DRAWN geometry implies)
CASES = {
    "empty_box": (lambda s: None, AZ),
    "volume_slab_floor_4mm": (
        lambda s: s.add(Box((0, 0, 0), (AX, AY, 4e-3)), material="pec"),
        AZ - 4e-3,
    ),
    "volume_slab_floor_6mm": (
        lambda s: s.add(Box((0, 0, 0), (AX, AY, 6e-3)), material="pec"),
        AZ - 6e-3,
    ),
    "sheet_at_4mm": (
        lambda s: s.add_thin_conductor(Box((0, 0, 4e-3), (AX, AY, 4e-3))),
        AZ - 4e-3,
    ),
    "volume_slab_ceiling_26mm": (
        lambda s: s.add(Box((0, 0, 26e-3), (AX, AY, AZ)), material="pec"),
        26e-3,
    ),
}


@pytest.fixture(scope="module")
def measured() -> dict[str, float]:
    return {name: _peak_freq(build) for name, (build, _d) in CASES.items()}


@pytest.mark.parametrize("name", list(CASES))
def test_realized_cavity_matches_the_drawn_geometry(name, measured):
    """The resonance the drawn dimensions predict is the one that is measured."""
    _build, d_drawn = CASES[name]
    f_analytic = _f_te101(AX, d_drawn)
    err_pct = 100.0 * (measured[name] - f_analytic) / f_analytic
    assert abs(err_pct) < MAX_ABS_ERR_PCT, (
        f"{name}: drawn cavity height {d_drawn * 1e3:.1f} mm predicts "
        f"{f_analytic / 1e9:.4f} GHz, measured {measured[name] / 1e9:.4f} GHz "
        f"({err_pct:+.3f} %, gate {MAX_ABS_ERR_PCT} %). One cell of DX = "
        f"{DX * 1e3:.0f} mm is worth about 5 % here, so an error this size "
        f"is a conductor landing off its declared plane, not dispersion. "
        f"Before #931 this case read -5.024 % (4 mm slab) / -5.668 % (6 mm) "
        f"because a volume's far face was never a wall."
    )


def test_a_sheet_and_a_volume_on_the_same_plane_resonate_identically(measured):
    """§1.3 and §1.2 must agree where they describe the same wall.

    A zero-thickness conductor at z = 4 mm and a volume filling 0..4 mm both
    put a tangential wall on the z = 4 mm node plane. The cavity above them
    is the same cavity, so the two declarations must be indistinguishable to
    the physics. Measured: both 6.8708 GHz.
    """
    assert measured["sheet_at_4mm"] == pytest.approx(
        measured["volume_slab_floor_4mm"], rel=1e-6), (
        "a sheet and a volume bounding the same cavity disagree: "
        f"sheet {measured['sheet_at_4mm'] / 1e9:.4f} GHz vs volume "
        f"{measured['volume_slab_floor_4mm'] / 1e9:.4f} GHz")


def test_the_rule_does_not_depend_on_which_end_the_slab_is_drawn_from(measured):
    """A 4 mm slab on the floor and a 4 mm slab on the ceiling leave the same
    26 mm cavity, so they must resonate alike. This is the falsifier for a
    rule that treats the lo and hi faces differently -- which is precisely
    what the pre-#931 rule did, and why it only ever lost the far face.
    """
    assert measured["volume_slab_ceiling_26mm"] == pytest.approx(
        measured["volume_slab_floor_4mm"], rel=1e-6), (
        "a slab drawn from the ceiling and one drawn from the floor bound "
        "different cavities: "
        f"ceiling {measured['volume_slab_ceiling_26mm'] / 1e9:.4f} GHz vs "
        f"floor {measured['volume_slab_floor_4mm'] / 1e9:.4f} GHz")
