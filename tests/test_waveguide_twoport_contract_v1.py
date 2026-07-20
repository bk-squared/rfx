"""Regression tests for the probe-aware normalized two-port contract.

What each test actually binds (corrected 2026-07-20, issue #395 — the
earlier "locks the core physical properties" framing overstated the
empty-guide cases, which are two-run identities):
- empty guide: S11==0 / |S21|==1 hold *by construction* (device run IS the
  reference run), so ``test_normalized_twoport_empty_is_extraction_identity``
  is an extraction-algebra / determinism tripwire, NOT a physics gate.
- dielectric obstacle (device != reference): reciprocity, sub-unity
  transmission and non-zero reflection are real, falsifiable physics —
  ``test_normalized_twoport_dielectric_binds_reciprocity_and_transmission``.
- PEC short reflects strongly without column-power blow-up (a real gate:
  device differs from reference).
- reference-plane knobs preserve S-parameter MAGNITUDES on a reflecting
  DUT (pure phase de-embedding). The de-embed PHASE-rotation sign/magnitude
  correctness is gated separately by
  ``tests/test_waveguide_phase_gate.py::test_deembed_step_sign_rotation_correct``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.geometry.csg import Box


FREQS = np.linspace(4.5e9, 8.0e9, 20)
F0 = float(FREQS.mean())
BW = max(0.2, min(0.8, (FREQS[-1] - FREQS[0]) / max(F0, 1.0)))


def _build_twoport_sim(*, kind: str, left_ref=None, right_ref=None):
    sim = Simulation(
        freq_max=max(float(FREQS[-1]), F0),
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
    )
    if kind == "pec_short":
        sim.add_material("pec_like", eps_r=1.0, sigma=1e10)
        sim.add(Box((0.05, 0.0, 0.0), (0.055, 0.04, 0.02)), material="pec_like")
    elif kind == "dielectric":
        # eps_r=4 centred slab: a genuine reflecting DUT (device != reference)
        # so the normalized S-matrix carries real reflection + sub-unity
        # transmission — used by the binding reciprocity / ref-plane tests.
        sim.add_material("diel", eps_r=4.0, sigma=0.0)
        sim.add(Box((0.05, 0.0, 0.0), (0.07, 0.04, 0.02)), material="diel")
    elif kind == "empty":
        pass
    else:
        raise ValueError(kind)

    sim.add_waveguide_port(
        0.01,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(FREQS),
        f0=F0,
        bandwidth=BW,
        ref_offset=3,
        probe_offset=15,
        name="left",
        reference_plane=left_ref,
    )
    sim.add_waveguide_port(
        0.09,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(FREQS),
        f0=F0,
        bandwidth=BW,
        ref_offset=3,
        probe_offset=15,
        name="right",
        reference_plane=right_ref,
    )
    return sim


def test_normalized_twoport_empty_is_extraction_identity():
    """Empty guide: S11==0 / |S21|==1 hold BY CONSTRUCTION (not physics).

    Issue #395: on an empty guide the device run *is* the reference run, so
    ``S11 = (b_dev - b_ref)/a_inc`` is identically 0 and ``S21 = b_dev/b_ref``
    is identically 1 for bit-identical runs — independent of PML/extractor
    physics. This test is an extraction-algebra / determinism tripwire: it
    fails only if the two-run subtraction/normalization plumbing breaks. The
    tight tolerances (1e-4, far below the old 0.15 physics-looking gate)
    reflect that the values are exact identities, not measured quantities.
    The real physics lives in
    ``test_normalized_twoport_dielectric_binds_reciprocity_and_transmission``.
    """
    sim = _build_twoport_sim(kind="empty")
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    s = np.asarray(result.s_params)

    # Identity, not physics: device==reference => exact 0 / exact 1.
    assert float(np.max(np.abs(s[0, 0, :]))) < 1e-4, (
        "empty-guide S11 is not the expected two-run identity 0 — the "
        "diagonal subtraction plumbing regressed"
    )
    assert float(np.max(np.abs(np.abs(s[1, 0, :]) - 1.0))) < 1e-4, (
        "empty-guide |S21| is not the expected two-run identity 1 — the "
        "off-diagonal normalization plumbing regressed"
    )


def test_normalized_twoport_dielectric_binds_reciprocity_and_transmission():
    """Dielectric obstacle (device != reference): real, falsifiable physics.

    Issue #395 replacement for the vacuous empty-guide contract. An eps_r=4
    slab reflects and attenuates, so the normalized S-matrix carries genuine
    reflection and sub-unity transmission that a physics regression can move.
    Measured 2026-07-20 (num_periods=40, 20 freqs): mean|S11|=0.509,
    mean|S21|=0.824, mean reciprocity error 0.0000.

    Passivity is NOT asserted here: ``normalize=True`` inflates |S11| on a
    reflector (column power ~1.21 on this obstacle, a documented extraction
    limitation). The honest passivity gate is the flux-path
    ``test_tight_passivity_reflecting_dut`` in the validation battery.
    """
    sim = _build_twoport_sim(kind="dielectric")
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    s = np.asarray(result.s_params)

    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    s12 = np.abs(s[0, 1, :])
    recip = np.abs(s21 - s12) / np.maximum(np.maximum(s21, s12), 1e-12)

    # Non-vacuity witness: device must differ from reference.
    assert float(np.mean(s11)) > 0.20, (
        f"obstacle not reflecting (mean|S11|={np.mean(s11):.3f}<=0.20) — the "
        "gate would collapse to the empty-guide identity"
    )
    # Physics: obstacle attenuates transmission below unity.
    assert 0.5 < float(np.mean(s21)) < 0.98, (
        f"mean|S21|={np.mean(s21):.3f} outside the reflecting-DUT band "
        "(0.5, 0.98) — an eps_r=4 step must attenuate but still transmit"
    )
    # Physics: Lorentz reciprocity S21==S12 on the projected TE10 subspace.
    assert float(np.mean(recip)) < 1e-3, (
        f"reciprocity error mean={np.mean(recip):.5f} >= 1e-3 — extractor "
        "drive/receive asymmetry regression"
    )


def test_normalized_twoport_pec_short_is_strongly_reflective():
    sim = _build_twoport_sim(kind="pec_short")
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    s = np.asarray(result.s_params)
    column_power = np.sum(np.abs(s) ** 2, axis=0)
    recip = np.abs(np.abs(s[1, 0, :]) - np.abs(s[0, 1, :])) / np.maximum(
        np.maximum(np.abs(s[1, 0, :]), np.abs(s[0, 1, :])),
        1e-12,
    )

    assert 0.85 < float(np.mean(np.abs(s[0, 0, :]))) < 1.05
    assert float(np.mean(np.abs(s[1, 0, :]))) < 0.10
    assert float(np.mean(column_power)) < 1.10
    assert float(np.mean(recip)) < 1e-3


def test_normalized_twoport_reference_plane_magnitude_invariant_on_reflecting_dut():
    """Reference-plane shift preserves |S| on a REFLECTING DUT (binding).

    Issue #395: the pre-#395 test ran on an empty guide, where S11==0 and the
    same ref_shift cancels in numerator and denominator of every two-run
    ratio, so a wrong-magnitude reference-plane implementation still passed.

    On a reflecting eps_r=4 DUT the diagonal S11 is non-zero, so the test
    actually exercises the de-embed path. A pure phase de-embed must leave
    the S-parameter MAGNITUDES unchanged (probe 2026-07-20: max|dmag|=0.0
    for S11 and S21) while legitimately rotating the S11 PHASE (up to ~4.4
    rad). A ref-plane bug that applied a real gain/attenuation instead of a
    unit-modulus phase would change |S| and trip this gate.

    Scope split (do not duplicate): the PHASE-rotation sign/magnitude
    correctness is gated by
    ``tests/test_waveguide_phase_gate.py::test_deembed_step_sign_rotation_correct``;
    here we bind only that the shift is a pure phase (magnitude-preserving)
    and that the transmission (off-diagonal) is fully invariant.
    """
    base = _build_twoport_sim(kind="dielectric").compute_waveguide_s_matrix(
        num_periods=40, normalize=True
    )
    shifted = _build_twoport_sim(
        kind="dielectric", left_ref=0.02, right_ref=0.08
    ).compute_waveguide_s_matrix(num_periods=40, normalize=True)

    s_base = np.asarray(base.s_params)
    s_shift = np.asarray(shifted.s_params)

    # Non-vacuity witness: the DUT reflects, so the diagonal is genuinely
    # exercised (the empty-guide S11==0 made the old test trivial).
    assert float(np.max(np.abs(s_base[0, 0, :]))) > 0.20, (
        "reference DUT is not reflecting — ref-plane magnitude invariance "
        "would be vacuous on the S11 diagonal"
    )
    # Binding: pure phase de-embed => |S| unchanged for the whole matrix.
    assert np.allclose(np.abs(s_base), np.abs(s_shift), rtol=1e-3, atol=1e-4), (
        "reference-plane shift changed |S| on a reflecting DUT — the "
        "de-embed is not a pure phase (wrong-magnitude ref-plane bug)"
    )
    # Binding: the transmission (off-diagonal) is fully complex-invariant —
    # its exp(+jbeta d) / exp(+jbeta d) cancels in b_dev/b_ref exactly.
    assert np.allclose(s_base[1, 0, :], s_shift[1, 0, :], rtol=1e-3, atol=1e-4), (
        "reference-plane shift moved the complex S21 — the off-diagonal "
        "ref_shift factors must cancel between device and reference runs"
    )
    # Witness that this is not merely re-checking an all-equal matrix: the
    # diagonal PHASE *does* rotate under the shift (that is the de-embed).
    diag_phase_shift = float(
        np.max(np.abs(np.angle(s_shift[0, 0, :]) - np.angle(s_base[0, 0, :])))
    )
    assert diag_phase_shift > 0.1, (
        f"S11 phase did not rotate under the reference-plane shift "
        f"(max dphase={diag_phase_shift:.4f} rad) — the ref-plane knob is a "
        "no-op, so the magnitude-invariance check above is not exercising it"
    )

