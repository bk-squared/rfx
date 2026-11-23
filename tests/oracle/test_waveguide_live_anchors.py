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

LIVE_DX = 0.002
PEC_SHORT_X = 0.084
PEC_SHORT_THICKNESS = LIVE_DX


def _live_build_sim(freqs_hz, *, pec_short_x=None):
    """Two-port WR-style guide; optional full-cross-section PEC short.

    Compact local builder (mirrors the validation battery's ``_build_sim``) so
    this live anchor is self-contained.
    The lattice ownership contract refuses a volume thinner than one cell.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    assert all(np.isclose(value / LIVE_DX, round(value / LIVE_DX), rtol=0, atol=1e-12) for value in (*DOMAIN, PORT_LEFT_X, PORT_RIGHT_X, PEC_SHORT_X, PEC_SHORT_X + PEC_SHORT_THICKNESS)), (
        'LIVE_DX must divide the domain, port planes and short faces exactly to preserve the declared lattice geometry.')
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=max(float(freqs[-1]), f0),
        domain=DOMAIN,
        dx=LIVE_DX,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
        cpml_layers=10,
    )
    if pec_short_x is not None:
        _grid = sim._build_grid()
        y_wall = domain_wall_positions(_grid, 1)[1]
        z_wall = domain_wall_positions(_grid, 2)[1]
        sim.add(
            Box((pec_short_x, 0.0, 0.0),
                (pec_short_x + PEC_SHORT_THICKNESS, y_wall, z_wall)),
            material="pec",
        )
    port_freqs = jnp.asarray(freqs)
    SHORT_TO_RIGHT_PORT_CELLS = 4
    right_x = (PORT_RIGHT_X if pec_short_x is None
               else pec_short_x + PEC_SHORT_THICKNESS + SHORT_TO_RIGHT_PORT_CELLS * LIVE_DX)
    for x, direction, name in ((PORT_LEFT_X, "+x", "left"),
                               (right_x, "-x", "right")):
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
    assert sim._boundary == "cpml", (
        f"live anchor requires a CPML (absorbing) boundary, got {sim._boundary!r}"
    )


def test_live_pec_short_s11_anchor():
    """LIVE compute_waveguide_s_matrix: PEC-short total reflection, |S11|≈1.

    The primary regression witness. Non-trivial (|S11|=1, NOT 0). A real
    extractor regression (ghost-cell contamination, wrong modal V/I integral)
    drops |S11| below the Meep-class 0.99 gate — exactly what the frozen replay
    cannot see.
    With normalize=False (single-run wave decomposition), the off-diagonal S21 is convention-dependent because the source spectrum is not cancelled without the two-run normalization.
    """
    freqs = np.linspace(*BAND_HZ, N_FREQS)
    sim = _live_build_sim(freqs, pec_short_x=PEC_SHORT_X)
    _assert_cpml(sim)
    s, _, idx = _s_matrix(sim, normalize=False)
    s11 = np.abs(s[idx["left"], idx["left"], :])
    s21 = np.abs(s[idx["right"], idx["left"], :])
    print(f"\n[live pec-short] |S11|={np.array2string(s11, precision=4)}")
    print(f"[live pec-short] |S21|(normalize=False, NOT asserted)={np.array2string(s21, precision=4)}")
    assert s11.min() >= 0.99, (
        f"LIVE PEC-short min|S11|={s11.min():.4f} < 0.99 — compute_waveguide_s_matrix "
        f"regression (the frozen broad-E5 replay would NOT catch this)"
    )
    assert s11.max() < 1.03, f"LIVE PEC-short max|S11|={s11.max():.4f} non-passive"
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
    The 0.98 gate keeps approximately 2% cross-machine float margin.
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
    assert s21.min() >= 0.98, (
        f"LIVE empty-guide min|S21|={s21.min():.4f} < 0.98 — transmission "
        f"extraction regression in compute_waveguide_s_matrix"
    )
    assert s11.max() < 0.05, (
        f"LIVE empty-guide max|S11|={s11.max():.4f} >= 0.05 — the two-run flux "
        f"cancellation is not deterministic (plumbing regression); this is NOT "
        f"a CPML-absorption check (see battery test_matched_load_s11_empty_waveguide)"
    )
    assert power.max() <= 1.05, (
        f"LIVE empty-guide max(|S11|²+|S21|²)={power.max():.4f} > 1.05 — "
        f"non-passive (energy-injection) extractor bug"
    )
