"""Regression tests for multi-port wire-port S-matrix extraction.

These tests exercise the off-diagonal S21 / direction-aware wave
decomposition added alongside crossval 13. Before this change, rfx's
non-uniform wire-port S-matrix only filled diagonal entries
(`S[j,j,:]`), so any 2-port filter crossval was impossible.

Strategy: build a minimal cavity / short waveguide with a purely
reactive load between two wire ports. Verify:

  1. **Matched line test** — two ports at opposite ends of a short
     straight "MSL", port 1 excited, port 2 passive with 50 Ω
     termination. Expected: |S11| < 0.5 (some reflection because the
     dielectric in CPML isn't perfectly matched, but clearly not a
     Fabry-Perot comb) and |S21| has a monotonic low-pass rolloff
     through the source band.

  2. **Envelope lock on a known-wrong S column** (#313 / #318) — NOT a
     passivity check, and not expected behaviour. The driven column does
     not track the load. See ``test_two_port_s_envelope_on_matched_line``.

  3. **Direction is inert** — ``direction`` must not change the S matrix
     at all (issue #673); it stays on the port spec only for the
     reference-plane path.

These are BASIC regression tests, not full validation — the
claims-bearing MSL notch validation lives in
`validation/crossval/06b_msl_notch_filter_uniform.py` (`add_msl_port`;
the nonuniform wire-port crossval lane cv06 was retired as
artifact-anchored, issue #339).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.auto_config import smooth_grading


def _build_line(with_port2_excite: bool = False, with_direction: bool = True):
    """Build a straight lossless 50 Ω-ish "MSL" stub for a 2-port test.

    Geometry: 0.5 mm wide PEC line on a 0.25 mm FR4-like substrate,
    20 mm long, with an infinite PEC ground plane (pec_faces={z_lo}).
    Two vertical wire ports at each x end span the substrate
    thickness. Port 1 is excited; port 2 is passive (matched load).
    """
    dx = 0.5e-3
    substrate_thickness = 0.25e-3   # 0.25 mm
    dz_sub = substrate_thickness / 3
    n_air = 6
    dz = np.concatenate([np.full(3, dz_sub), np.full(n_air, dx)])
    dz_profile = smooth_grading(dz, max_ratio=1.3)

    dom_x = 20e-3
    dom_y = 6e-3
    port_margin = 4e-3     # 4 mm > cpml thickness (8 * 0.5mm)

    sim = Simulation(
        freq_max=8e9,
        domain=(dom_x, dom_y, 0),
        dx=dx,
        dz_profile=dz_profile,
        boundary="cpml",
        cpml_layers=8,
        pec_faces={"z_lo"},
    )
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, substrate_thickness)), material="sub")

    line_w = 0.5e-3
    line_y_lo = dom_y / 2 - line_w / 2
    line_y_hi = dom_y / 2 + line_w / 2
    sim.add(Box((0, line_y_lo, substrate_thickness),
                (dom_x, line_y_hi, substrate_thickness + dz_sub)),
            material="pec")

    pulse = GaussianPulse(f0=4e9, bandwidth=1.0)

    # Port 1 — excited, at x = port_margin
    direction_lo = "-x" if with_direction else None
    sim.add_port(
        position=(port_margin, dom_y / 2, 0.0),
        component="ez",
        impedance=50.0,
        extent=substrate_thickness,
        waveform=pulse,
        direction=direction_lo,
    )

    # Port 2 — passive, at x = dom_x − port_margin
    direction_hi = "+x" if with_direction else None
    sim.add_port(
        position=(dom_x - port_margin, dom_y / 2, 0.0),
        component="ez",
        impedance=50.0,
        extent=substrate_thickness,
        excite=with_port2_excite,
        direction=direction_hi,
    )
    return sim


def test_two_port_s_matrix_has_nonzero_s21():
    """With port 1 excited and port 2 passive matched, the S-matrix
    column 0 must have a non-zero S[1, 0] (transmission). This is the
    bare-minimum fix validation — crossval 13 before this change got
    S21 = 0 identically because only diagonal entries were filled.

    The VALUE moved with issue #673's wave-split convention fix: max|S21|
    on this fixture was 0.56519 before and is 2.62928 now. The threshold
    below is a floor-of-the-numerical-noise check, not an accuracy gate —
    the NU off-diagonal normalization is known-wrong (#308 receive channel,
    #318 per-cell Z0; see ``test_two_port_s_envelope_on_matched_line``), so
    the magnitude carries no physical meaning.
    """
    sim = _build_line()
    freqs = jnp.linspace(1e9, 8e9, 101)
    result = sim.run(
        n_steps=4000,
        compute_s_params=True,
        s_param_freqs=freqs,
    )
    S = np.asarray(result.s_params)
    assert S.shape == (2, 2, len(freqs))
    S21 = S[1, 0, :]
    # Before the fix this was identically zero; after the fix the
    # magnitude must be clearly above the numerical floor.
    assert np.max(np.abs(S21)) > 1e-3, (
        f"S21 all zero (max={np.max(np.abs(S21)):.2e}) — off-diagonal "
        f"wave-decomposition fix did not take effect"
    )


def test_two_port_s_envelope_on_matched_line():
    """Envelope lock on a KNOWN-WRONG S column. NOT a passivity gate, and
    NOT a statement that these values are expected behaviour.

    This test used to assert ``max(|S11|^2 + |S21|^2) < 1.5`` and call it
    "relaxed passivity". That framing was wrong twice over, and so was the
    first replacement for it. What is actually true:

    1. **The old gate passed because S11 was reported as its RECIPROCAL.**
       The NU wave split branched on the port's ``direction``, and the "-x"
       branch (which port 1 of this fixture uses) was the exact reciprocal
       of the correct one. Reciprocating a >1 number lands under 1, which is
       what kept the gate green. Fixed in #673.

    2. **The DIAGONAL was fixed by issue #764; the off-diagonal remains
       known-wrong.** HISTORICAL PROVENANCE for the old envelope (28.511,
       |S11| up to 4.648): the driven diagonal did not track the load at
       all — a matched ``R_L = Z0 = 50`` (``Gamma = 0``) read
       ``S11 = +0.35426`` and a PEC short (``Gamma = -1``) read
       ``+0.26780``; one geometry with only ``excite`` flipped read
       -0.600000 passive against +0.999670-0.022145j driven (n_live = 4).
       Root cause was the #313 / #318 frame mismatch: V and I sampled at
       ONE cell measure the per-cell ``Z0/n_live`` while the reflection
       formula references the whole-port ``Z0``.  #764 replaced the driven
       NU diagonal with the whole-port reflection
       ``S_kk = (V_port - Z0*I)/(V_port + Z0*I)`` (whole-gap line-integral
       V against the physically realized series Z0), validated on the
       clean-gap falsifier battery
       (validation/research/issue764_wireport_norm_falsifiers.py: matched
       |S11| 0.001-0.0125, short -1, open +1, load-law slope 0.9996).  On
       THIS fixture |S11| now reads in [0.115, 0.728] — bounded and
       load-tracking — but the stub-line fixture itself is not a
       calibrated oracle, so the diagonal stays inside the envelope lock
       below rather than getting its own physics gate here.

    3. **The NU off-diagonal is wrong in its own right.** It uses the
       receive channel issue #308 removed and the full Z0 instead of the
       per-cell ``Z0/n_live`` (#318). Measured against the validated uniform
       multi-port extractor on an IDENTICAL grid, ``S21_NU/S21_uniform`` is
       -1.000+0.006j at ``n_live = 2`` and 0.62 at ``n_live = 4`` — an
       n_live-dependent normalization error, not a sign. Fixing it needs
       n_live threaded into the NU wire-port spec.

    What #673 DID fix is that ``direction`` no longer changes the answer.
    The PASSIVE diagonal is lane-identical and hits its closed form; that is
    pinned in ``tests/test_nu_wire_port_lane_parity.py``. Nothing about the
    DRIVEN column below is validated.

    So: this asserts finiteness and freezes the measured envelope, purely so
    the numbers cannot drift unnoticed. RE-PINNED 2026-08-29 for issue #764
    (the old 28.511 envelope was a pin on the known-wrong per-cell driven
    diagonal; see the provenance in item 2 above). Measured on this branch
    (4000 steps, 2-6 GHz, 51 bins):

        max(|S11|^2 + |S21|^2) = 4.84444
        |S11| in [0.11471, 0.72794],  |S21| in [0.33882, 2.17686]

    The |S21| class is unchanged (the #308/#318 off-diagonal defect is
    untouched by #764). The value still depends on the V/I sampling
    ordering, which is OPEN as issue #683. Any correct fix to the NU
    off-diagonal (#313 / #318) or to #683 will red this test; that is the
    point. Re-derive and re-document when it happens — do not "restore"
    the number.
    """
    sim = _build_line()
    freqs = jnp.linspace(2e9, 6e9, 51)
    result = sim.run(
        n_steps=4000,
        compute_s_params=True,
        s_param_freqs=freqs,
    )
    S = np.asarray(result.s_params)
    S11 = S[0, 0, :]; S21 = S[1, 0, :]
    assert np.all(np.isfinite(S)), "non-finite entry in the S matrix"
    p_total = np.abs(S11) ** 2 + np.abs(S21) ** 2
    measured = 4.84444
    assert abs(float(np.max(p_total)) / measured - 1.0) < 0.05, (
        f"max(|S11|²+|S21|²) = {float(np.max(p_total)):.3f} moved "
        f"off the recorded {measured} envelope. This is an envelope lock on "
        f"a KNOWN-WRONG column (#313/#318, and #683 for the sampling "
        f"ordering), not a physics gate — read the docstring before "
        f"re-pinning it."
    )


def test_direction_does_not_change_the_s_matrix():
    """``direction`` must be inert in the wave decomposition (issue #673).

    This assertion is unchanged — explicit vs auto-detected ``direction``
    must give the same S matrix — but what it guards changed. It used to
    read as "auto-detection picks the direction I would have passed"; it
    now pins that NO choice of ``direction`` can reach the (a, b) split at
    all. ``direction`` is still carried on the port spec because the
    reference-plane path (``add_port(reference_plane_cells=...)``) needs it
    for the outboard sign.

    Before #673 the "-x"/"-y" branch returned the reciprocal of the
    "+x"/"+y" one, so this test passed only because ``_auto_direction``
    happened to pick the same string the fixture passed explicitly.
    """
    sim_explicit = _build_line(with_direction=True)
    sim_auto = _build_line(with_direction=False)
    freqs = jnp.linspace(3e9, 5e9, 21)

    r_exp = sim_explicit.run(
        n_steps=3000, compute_s_params=True, s_param_freqs=freqs)
    r_auto = sim_auto.run(
        n_steps=3000, compute_s_params=True, s_param_freqs=freqs)

    S_exp = np.asarray(r_exp.s_params)
    S_auto = np.asarray(r_auto.s_params)
    np.testing.assert_allclose(S_exp, S_auto, rtol=1e-5, atol=1e-8)
