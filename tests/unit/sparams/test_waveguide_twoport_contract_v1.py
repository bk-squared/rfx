"""Regression tests for the probe-aware normalized two-port contract.

What each test actually binds (corrected 2026-07-20, issue #395 — the
earlier "locks the core physical properties" framing overstated the
empty-guide cases, which are two-run identities):
- empty guide: S11==0 / S21==single-run S21 hold by construction (device IS the
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
  ``tests/unit/sparams/test_waveguide_phase_gate.py::test_deembed_step_sign_rotation_correct``.
The port's numerical TE10 cutoff is what the far-port advisory uses.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.geometry.csg import Box


FREQS = np.linspace(4.5e9, 8.0e9, 20)
F0 = float(FREQS.mean())
BW = max(0.2, min(0.8, (FREQS[-1] - FREQS[0]) / max(F0, 1.0)))

C0 = 299_792_458.0
DOMAIN = (0.12, 0.04, 0.02)
FC_TE10_NUMERICAL = 3.476e9
_DX = C0 / max(float(FREQS[-1]), F0) / 20.0
_LAMBDA_G_LOW = (C0 / float(FREQS[0])) / np.sqrt(
    1.0 - (FC_TE10_NUMERICAL / float(FREQS[0])) ** 2)
CPML_LAYERS = int(np.ceil(0.75 * _LAMBDA_G_LOW / _DX))


def _build_twoport_sim(*, kind: str, left_ref=None, right_ref=None):
    """The domain lengths written as ``n_y * dx`` and ``n_z * dx`` are whole numbers of cells of dx, so their solved lengths equal their declared lengths."""
    dx = 0.75e-3 if kind == "dielectric" else (_DX if kind == "empty" else None)
    n_y, n_z = (54, 27) if kind == "dielectric" else (22, 11)
    sim = Simulation(
        freq_max=max(float(FREQS[-1]), F0),
        domain=(DOMAIN[0], n_y * dx, n_z * dx) if dx is not None else DOMAIN, dx=dx,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
        cpml_layers=CPML_LAYERS,
    )
    if kind == "pec_short":
        sim.add_material("pec_like", eps_r=1.0, sigma=1e10)
        sim.add(Box((0.05, 0.0, 0.0), (0.055, 0.04, 0.02)), material="pec_like")
    elif kind == "dielectric":
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
    """Empty-guide subtraction is exact; transmission keeps propagation."""
    sim = _build_twoport_sim(kind="empty")
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    single = sim.compute_waveguide_s_matrix(num_periods=40, normalize=False)
    s = np.asarray(result.s_params)
    s_single = np.asarray(single.s_params)

    assert np.array_equal(s[0, 0], np.zeros_like(s[0, 0]))
    for i, j in ((1, 0), (0, 1)):
        assert np.max(np.abs(s[i, j] - s_single[i, j])) <= (
            1e-5 * np.max(np.abs(s_single[i, j]))
        )


def test_normalized_twoport_dielectric_binds_reciprocity_and_transmission():
    """Dielectric obstacle (device != reference): real, falsifiable physics.

    Issue #395 replacement for the vacuous empty-guide contract. An eps_r=4
    slab reflects and attenuates, so the normalized S-matrix carries genuine
    reflection and sub-unity transmission that a physics regression can move.
    Measured 2026-07-20 (num_periods=40, 20 freqs): mean|S11|=0.509,
    mean|S21|=0.824, mean reciprocity error 0.0000.

    Passivity was NOT asserted here, on the stated grounds that ``normalize=True``
    inflates |S11| on a reflector — "column power ~1.21 on this obstacle, a
    documented extraction limitation". **That attribution was wrong and is
    withdrawn (#496).** Holding dx, the DUT, the ports and the reference planes
    fixed and raising only ``cpml_layers`` (CPML is padding OUTSIDE the user
    domain, so the interior is bit-identical — verified: grid shape minus
    2*cpml is a constant 66 on every axis), the max column power measures

        0.179 lambda_g (10 cells, as committed)  1.2113
        0.500 lambda_g (28)                      1.0329
        0.750 lambda_g (42, now)                 1.0128

    i.e. 94% of the excess was the thin far-port absorber, not the extractor.
    The same 1.2113 recurs at num_periods=80, so it is not record truncation
    either. An earlier probe that grew the domain alongside the absorber could
    not have said this: it moved port-to-absorber standoff in lockstep with
    absorber depth, and rfx documents port-to-CPML proximity as an |S11|-inflating
    mechanism in its own right.

    Passivity IS asserted now, on the MAX rather than the mean — see
    ``test_normalized_twoport_pec_short_is_strongly_reflective`` for why the mean
    was the wrong statistic. The tighter flux-path gate remains
    ``test_tight_passivity_reflecting_dut`` in the validation battery.
    The 1.05 maximum-column-power bound leaves margin for the float32 normalized envelope without admitting the 1.2113 an earlier thin absorber produced.
    """
    sim = _build_twoport_sim(kind="dielectric")
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    s = np.asarray(result.s_params)

    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    s12 = np.abs(s[0, 1, :])
    recip = np.abs(s21 - s12) / np.maximum(np.maximum(s21, s12), 1e-12)

    assert float(np.mean(s11)) > 0.20, (
        f"obstacle not reflecting (mean|S11|={np.mean(s11):.3f}<=0.20) — the "
        "gate would collapse to the empty-guide identity"
    )
    assert 0.5 < float(np.mean(s21)) < 0.98, (
        f"mean|S21|={np.mean(s21):.3f} outside the reflecting-DUT band "
        "(0.5, 0.98) — an eps_r=4 step must attenuate but still transmit"
    )
    colpow = np.sum(np.abs(s) ** 2, axis=0)
    assert float(np.max(colpow)) < 1.05, (
        f"max column power {np.max(colpow):.4f} on a LOSSLESS slab — at the "
        f"committed 0.179-lambda_g absorber this read 1.2113 and the extractor "
        f"emitted an unheeded 'S-parameters are UNRELIABLE' warning (#496)"
    )
    assert float(np.mean(recip)) < 1e-3, (
        f"reciprocity error mean={np.mean(recip):.5f} >= 1e-3 — extractor "
        "drive/receive asymmetry regression"
    )


def test_normalized_twoport_pec_short_is_strongly_reflective():
    """The (0.98, 1.005) mean-|S11| band and 1.05 maximum-column-power bound leave margin around the 0.9998 mean |S11| and 1.0208 maximum column power measured earlier with the 0.75-lambda_g absorber."""
    sim = _build_twoport_sim(kind="pec_short")
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    s = np.asarray(result.s_params)
    column_power = np.sum(np.abs(s) ** 2, axis=0)
    recip = np.abs(np.abs(s[1, 0, :]) - np.abs(s[0, 1, :])) / np.maximum(
        np.maximum(np.abs(s[1, 0, :]), np.abs(s[0, 1, :])),
        1e-12,
    )

    mean_s11 = float(np.mean(np.abs(s[0, 0, :])))
    assert 0.98 < mean_s11 < 1.005, (
        f"mean|S11|={mean_s11:.4f} on a LOSSLESS total reflector. Above unity is "
        f"unphysical; at the committed 0.179-lambda_g absorber this read 1.0252 "
        f"and passed only because the band was 1.05 wide (#496)"
    )
    assert float(np.mean(np.abs(s[1, 0, :]))) < 0.10
    assert float(np.mean(column_power)) < 1.10
    assert float(np.max(column_power)) < 1.05, (
        f"max column power {np.max(column_power):.4f} — the mean-only gate this "
        f"replaces passed at max 1.5343 (#496)"
    )
    assert float(np.mean(recip)) < 1e-3


def test_normalized_twoport_reference_plane_magnitude_invariant_on_reflecting_dut():
    """A plane shift preserves magnitudes and rotates one scattering matrix.

    Transmission uses driven-port reference incidence, retaining propagation;
    its doubled phase rotation equals the sum of the reflection rotations.
    """
    base = _build_twoport_sim(kind="dielectric").compute_waveguide_s_matrix(
        num_periods=40, normalize=True
    )
    shifted = _build_twoport_sim(
        kind="dielectric", left_ref=0.02, right_ref=0.08
    ).compute_waveguide_s_matrix(num_periods=40, normalize=True)

    s_base = np.asarray(base.s_params)
    s_shift = np.asarray(shifted.s_params)

    assert float(np.max(np.abs(s_base[0, 0, :]))) > 0.20, (
        "reference DUT is not reflecting — ref-plane magnitude invariance "
        "would be vacuous on the S11 diagonal"
    )
    assert np.allclose(np.abs(s_base), np.abs(s_shift), rtol=1e-3, atol=1e-4), (
        "reference-plane shift changed |S| on a reflecting DUT — the "
        "de-embed is not a pure phase (wrong-magnitude ref-plane bug)"
    )
    rotations = np.angle(s_shift * np.conj(s_base))
    residual = np.angle(np.exp(1j * (
        2 * rotations[1, 0] - rotations[0, 0] - rotations[1, 1]
    ))) / 2
    assert np.max(np.abs(residual)) < 1e-4
    diag_phase_shift = float(
        np.max(np.abs(np.angle(s_shift[0, 0, :]) - np.angle(s_base[0, 0, :])))
    )
    assert diag_phase_shift > 0.1, (
        f"S11 phase did not rotate under the reference-plane shift "
        f"(max dphase={diag_phase_shift:.4f} rad) — the ref-plane knob is a "
        "no-op, so the magnitude-invariance check above is not exercising it"
    )

