from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.geometry.csg import Box
from tests._realized_geometry import domain_wall_positions

DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
BAND_HZ = (5.0e9, 7.0e9)
N_FREQS = 6

# #931: the anchor's mesh is DECLARED, not auto-derived. The short below is a
# metal block — a total reflector, so a VOLUME (design note §1.5) — and a
# volume has to be an integer number of cells thick or the contract refuses it.
# With the auto-derived dx (0.00214137 m at freq_max = 7 GHz) the old 2 mm
# thickness was 0.93 of a cell and the drawn faces sat nowhere near a node, so
# the realized reflector depended on where 0.085 m happened to fall between two
# cell centres. LIVE_DX = 2 mm divides the domain (60 x 20 x 10 cells), both
# port planes (5, 45) and the short's faces (42, 43) exactly, and is within 7 %
# of the mesh the auto rule chose, so the anchor's gates keep their scale.
LIVE_DX = 0.002
# Short at 42 cells; one cell thick, i.e. a filled slab with walls on BOTH
# drawn faces (§1.2). It was 0.085 m against the old mesh, which is 42.5 cells
# — half a cell off the node line, and the half-cell that made the drawing
# ambiguous. The reflecting (near) face moves 1 mm; |S11| for a total
# reflector is a magnitude, so the gates below are unchanged quantities.
PEC_SHORT_X = 0.084
PEC_SHORT_THICKNESS = LIVE_DX


def _live_build_sim(freqs_hz, *, pec_short_x=None):
    """Two-port WR-style guide; optional full-cross-section PEC short.

    Compact local builder (mirrors the validation battery's ``_build_sim``) so
    this live anchor is self-contained.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=max(float(freqs[-1]), f0),
        domain=DOMAIN,
        dx=LIVE_DX,
        boundary="cpml",
        cpml_layers=10,
    )
    if pec_short_x is not None:
        # A VOLUME, RESOLVED. The short used to be drawn 2 mm thick on a mesh
        # this builder let rfx choose (~2.14 mm here) — i.e. THINNER than
        # one cell, which the lattice ownership contract refuses outright
        # (#931 §1.5) rather than realizing to whatever the raster happened
        # to give. The live mesh is now LIVE_DX = 2 mm (above), so the short
        # is exactly one cell drawn on node planes from the same front face:
        # the declaration says what the lattice can build.
        #
        # Two earlier readings of this anchor on the merged tree, both now
        # ATTRIBUTED (2026-09-07, on the WR-90 waveguide-port case's
        # identical set-up, per-bin dumps + port time records):
        #
        # 1. On the AUTO mesh (2.1414 mm) the one-cell volume read |S11| =
        #    [0.9663, 0.9572, 0.9741, 0.9827, 0.9847, 0.9836]. That mesh
        #    makes the guide ceil(20 / 2.1414) = 10 cells = 21.41 mm tall,
        #    and a plug drawn to DOMAIN[2] = 20 mm rounds its top face to
        #    the nearest node (#931 §1.1) at 19.27 mm: a 2.14 mm vacuum slot
        #    under the top wall, a parallel-plate line that carries Ez past
        #    the "short". cv11's 1 mm slot of the same origin was the whole
        #    0.0146 -> 0.0560 step (|S21| 0.22-0.33 behind the plug; closing
        #    it returns [0.9980, 1.0019], equal to the pre-change baseline).
        #    Not the lane: with the plug at the wall the lane seals the guide
        #    exactly (the record behind it is identically zero).
        # 2. Declared as a zero-thickness Box at pec_short_x on that same
        #    mesh the anchor read [1.2128, 0.7139, 0.8416, 0.9628, 1.0023,
        #    1.4836]. A sheet's footprint is sampled CLOSED at nodes, so on
        #    the auto mesh it stops one node short of the wall on BOTH
        #    transverse axes: an L-shaped zero-thickness slot, a resonator,
        #    in a 40-period record. The lane DOES apply sheets — a sheet
        #    drawn to the realized walls reads the >= 0.99 class on cv11.
        #
        # So the short is drawn to the grid's REALIZED walls (identical to
        # DOMAIN at LIVE_DX = 2 mm, which is on-lattice, and the reason the
        # tests-crossval re-run at LIVE_DX passed), and the build-time test
        # below asserts each face is a wall across the WHOLE cross-section.
        _grid = sim._build_grid()
        y_wall = domain_wall_positions(_grid, 1)[1]
        z_wall = domain_wall_positions(_grid, 2)[1]
        sim.add(
            Box((pec_short_x, 0.0, 0.0),
                (pec_short_x + PEC_SHORT_THICKNESS, y_wall, z_wall)),
            material="pec",
        )
    port_freqs = jnp.asarray(freqs)
    for x, direction, name in ((PORT_LEFT_X, "+x", "left"),
                               (PORT_RIGHT_X, "-x", "right")):
        sim.add_waveguide_port(
            x, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=port_freqs, f0=f0, bandwidth=bandwidth,
            waveform="modulated_gaussian", n_modes=1, name=name,
        )
    return sim


def _s_matrix(sim, *, normalize, num_periods=40):
    result = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=normalize)
    s = np.asarray(result.s_params)
    idx = {name: i for i, name in enumerate(result.port_names)}
    return s, np.asarray(result.freqs), idx


def _assert_cpml(sim):
    # Cheap constructor guard (echoes the boundary kwarg). NOTE (issue #395):
    # the empty-guide |S11|≈0 in test_live_empty_guide_s21_anchor is NOT an
    # absorbing-boundary witness — on the flux path device==reference makes it
    # ~0 by construction regardless of CPML quality. The real waveguide-lane
    # PML-reflection gate is the single-run test_matched_load_s11_empty_waveguide
    # in the validation battery.
    assert sim._boundary == "cpml", (
        f"live anchor requires a CPML (absorbing) boundary, got {sim._boundary!r}"
    )


def test_live_pec_short_s11_anchor():
    """LIVE compute_waveguide_s_matrix: PEC-short total reflection, |S11|≈1.

    The primary regression witness. Non-trivial (|S11|=1, NOT 0). A real
    extractor regression (ghost-cell contamination, wrong modal V/I integral)
    drops |S11| below the Meep-class 0.99 gate — exactly what the frozen replay
    cannot see.
    """
    freqs = np.linspace(*BAND_HZ, N_FREQS)
    sim = _live_build_sim(freqs, pec_short_x=PEC_SHORT_X)
    _assert_cpml(sim)
    s, _, idx = _s_matrix(sim, normalize=False)
    s11 = np.abs(s[idx["left"], idx["left"], :])
    s21 = np.abs(s[idx["right"], idx["left"], :])
    print(f"\n[live pec-short] |S11|={np.array2string(s11, precision=4)}")
    # NOTE: |S21| is NOT asserted here. With normalize=False (single-run wave
    # decomposition) the off-diagonal S21 is convention-dependent — the source
    # spectrum is not cancelled without the two-run normalization, so the raw
    # right-port ratio is ~1 even behind the short. PEC-short's validated,
    # Meep-class quantity is |S11| (battery test_pec_short_s11_magnitude); the
    # live transmission/S21 path is checked separately on the empty guide with
    # normalize='flux'. (R5: the |S21|≈0 expectation was an extraction-convention
    # misdiagnosis, surfaced here; not chased — R2.)
    print(f"[live pec-short] |S21|(normalize=False, NOT asserted)={np.array2string(s21, precision=4)}")
    assert s11.min() >= 0.99, (
        f"LIVE PEC-short min|S11|={s11.min():.4f} < 0.99 — compute_waveguide_s_matrix "
        f"regression (the frozen broad-E5 replay would NOT catch this)"
    )
    # 1.03 matches the battery's validated near-cutoff ceiling (the 5 GHz bin at
    # f/fc=1.33 carries a small over-unity discrete-Yee Z_TE residual).
    assert s11.max() < 1.03, f"LIVE PEC-short max|S11|={s11.max():.4f} non-passive"
    # Gate-tightness witness (non-vacuous): the live healthy values sit close to
    # the 0.99 floor, so the gate catches a regression of ~1%, not a slack one.
    # This is what makes the LIVE anchor discriminating where the frozen replay
    # (a fixed JSON answer key, blind to the live extractor) is not.
    assert s11.min() - 0.99 < 0.02, (
        f"PEC-short gate is slack: healthy min|S11|={s11.min():.4f} is >0.02 above "
        f"the 0.99 floor, so a real regression could hide under it"
    )


def test_live_empty_guide_s21_anchor():
    """LIVE compute_waveguide_s_matrix: empty matched guide transmits, |S21|≈1.

    Secondary witness covering the TRANSMISSION / S21 extraction path (PEC-short
    checks only reflection). ``normalize='flux'`` (documented-convergent), plus a
    live passivity check |S11|²+|S21|² ≤ 1. Empty-guide |S11|≈0 is used only as a
    sanity bound here, NOT as the S11 regression witness (that is PEC-short).
    """
    freqs = np.linspace(*BAND_HZ, N_FREQS)
    sim = _live_build_sim(freqs)
    _assert_cpml(sim)
    s, _, idx = _s_matrix(sim, normalize="flux")
    s11 = np.abs(s[idx["left"], idx["left"], :])
    s21 = np.abs(s[idx["right"], idx["left"], :])
    power = s11**2 + s21**2
    print(f"\n[live empty] |S21|={np.array2string(s21, precision=4)}  (ideal 1)")
    print(f"[live empty] |S11|={np.array2string(s11, precision=4)}  passivity={np.array2string(power, precision=4)}")
    # Tight transmission witness (measured ~0.999; the battery's matched-load
    # gates are ratcheted to their values, so this is too — 0.98 keeps ~2%
    # cross-machine float margin, far tighter than the prior slack 0.9).
    assert s21.min() >= 0.98, (
        f"LIVE empty-guide min|S21|={s21.min():.4f} < 0.98 — transmission "
        f"extraction regression in compute_waveguide_s_matrix"
    )
    # By-construction determinism bound (NOT a CPML witness — issue #395).
    # On the flux path the empty-guide diagonal is
    # |S11|^2 = |F_ref - F_dev| / |F_ref| and the empty device run equals the
    # vacuum reference run bit-for-bit, so |S11| ~ 0 REGARDLESS of boundary
    # quality — a crippled CPML would not move it. This asserts only that the
    # two-run flux cancellation is deterministic. The real waveguide-lane
    # PML-reflection gate is the single-run (normalize=False)
    # test_matched_load_s11_empty_waveguide in the validation battery, which
    # trebles when the CPML is crippled.
    assert s11.max() < 0.05, (
        f"LIVE empty-guide max|S11|={s11.max():.4f} >= 0.05 — the two-run flux "
        f"cancellation is not deterministic (plumbing regression); this is NOT "
        f"a CPML-absorption check (see battery test_matched_load_s11_empty_waveguide)"
    )
    assert power.max() <= 1.05, (
        f"LIVE empty-guide max(|S11|²+|S21|²)={power.max():.4f} > 1.05 — "
        f"non-passive (energy-injection) extractor bug"
    )
