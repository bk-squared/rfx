"""Tests for the DEPRECATED ``compute_coaxial_s_matrix`` API.

``compute_coaxial_s_matrix`` (single-plane V/I in a closed PEC box) is
deprecated — it emits a ``DeprecationWarning`` and reports non-physical
|S11| > 1 on lossless shorts (measured 1.02–1.16 on the PEC-short fixture
here). The validated replacement is ``compute_coaxial_line_reflection``
(coax-line method; see ``tests/test_coaxial_line_calibration.py``). These
tests are retained as plumbing/regression guards for the deprecated path
until its removal. They focus on:

- API/schema contract (shape, types, status field)
- Family-routing rejection (mixing port families)
- Closed-form ``Z_TEM`` consistency between the helper and the result
- Plane-source amplitude scale invariance (same S-matrix at two field_scale
  values)
- Validation contract on the returned ``CoaxialSMatrixResult``

Calibrated physics gates live with the validated coax-line method
(``tests/test_coaxial_line_calibration.py``), NOT here — wiring physics
gates onto the deprecated single-plane path would mark it as
physics-validated by test-pass-alone, which the physics-validation
evidence rule forbids.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import (
    CoaxialSMatrixResult,
    Simulation,
    coaxial_tem_characteristic_impedance,
    validate_port_smatrix,
)
from rfx.sources.coaxial_port import (
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
)


def _make_one_port_sim() -> Simulation:
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
    return sim


def test_compute_coaxial_s_matrix_returns_canonical_result_shape():
    sim = _make_one_port_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)
    assert isinstance(res, CoaxialSMatrixResult)
    assert res.s_params.shape == (1, 1, 3)
    assert res.freqs.shape == (3,)
    assert res.port_names == ("coax_0",)
    assert res.port_faces == ("top",)
    assert res.reference_planes.shape == (1,)
    assert res.z_tem_ohm.shape == (1, 3)
    assert res.voltages.shape == (1, 1, 3)
    assert res.currents.shape == (1, 1, 3)
    assert res.status in {"passed", "degraded"}


def test_z_tem_matches_closed_form_helper():
    sim = _make_one_port_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)
    z_expected = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS, SMA_OUTER_RADIUS, PTFE_EPS_R
    )
    np.testing.assert_allclose(
        res.z_tem_ohm,
        np.full_like(res.z_tem_ohm, z_expected, dtype=np.complex128),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_field_scale_does_not_change_s_matrix_amplitudes():
    sim = _make_one_port_sim()
    res_lo = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3, field_scale=1.0e3)
    sim2 = _make_one_port_sim()
    res_hi = sim2.compute_coaxial_s_matrix(n_steps=200, n_freqs=3, field_scale=1.0e5)
    # The plane-source amplitude is a pure linear scale on both V and I, so
    # the power-wave ratio b/a is invariant under field_scale up to the
    # float32 DFT-accumulator round-off floor.
    np.testing.assert_allclose(
        res_lo.s_params, res_hi.s_params, rtol=1.0e-4, atol=1.0e-7
    )


def test_returns_finite_complex_smatrix_with_reasonable_magnitude():
    sim = _make_one_port_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)
    assert np.all(np.isfinite(res.s_params))
    # Smoke envelope (W3.5, tightened 5.0 → 1.5 with measured margins):
    # measured max|S| = 0.879 on this fixture (2026-07-03). NOT a passivity
    # gate (≤1) — the deprecated single-plane V/I method non-physically
    # exceeds 1 on lossless shorts (see the DeprecationWarning), so 1.5 is
    # the envelope that keeps headroom over the measured values across all
    # three smoke fixtures (0.88 / 1.16 / 1.14) while still catching the
    # plumbing blowups that the old 5.0 bound would have let through.
    assert float(np.max(np.abs(res.s_params))) < 1.5


def test_validation_helper_normalizes_coaxial_result():
    sim = _make_one_port_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)
    report = validate_port_smatrix(
        res,
        port_names=res.port_names,
        check_passivity=False,
        check_reciprocity=False,
    )
    assert report.n_ports == 1
    assert report.n_freqs == 3
    assert "max_column_power" not in report.metrics or np.isfinite(
        report.metrics.get("max_column_power", 0.0)
    )


def test_compute_coaxial_rejects_no_coaxial_ports():
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    with pytest.raises(ValueError, match="No coaxial ports"):
        sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)


def test_compute_coaxial_rejects_mixed_port_families():
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
    sim.add_msl_port(
        (0.005, 0.010, 0.001),
        width=0.6e-3,
        height=0.254e-3,
        direction="+x",
    )
    with pytest.raises(NotImplementedError, match="add_coaxial_port"):
        sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)


def test_preflight_sparameters_routes_coaxial_calculator():
    sim = _make_one_port_sim()
    issues = sim.preflight_sparameters(calculator="coaxial")
    assert issues == []
    issues_alias = sim.preflight_sparameters(calculator="compute_coaxial_s_matrix")
    assert issues_alias == []


def test_preflight_sparameters_rejects_coaxial_when_no_ports():
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    issues = sim.preflight_sparameters(calculator="coaxial")
    assert any("No coaxial ports" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------
#
# After M73 (TFSF restructure + setup_coaxial_port wiring + V/I direction
# convention), the PEC-short calibration target passes at the resolved
# fixture. These tests pin down the regression surface: nominal |S11| ≈ 1
# at the test fixture, robustness across pin_length, and phase rotation
# vs pin_length matching the analytic 2β·d expectation up to discretisation.


def _make_pec_short_sim() -> Simulation:
    """Coaxial port whose centre pin reaches the cavity floor (PEC short).

    ``freq_max=20 GHz`` gives ``dx ≈ 0.75 mm`` so the SMA PTFE annulus
    (0.71 mm wide between the 0.635 mm pin and the 1.345 mm shell-inner
    edge) is resolved by ~1 cell; at the previous ``freq_max=10 GHz``
    setting the PTFE region was 0.47 cells wide and could not support a
    discrete coax TEM mode regardless of source quality (see the
    20260510_coaxial_tfsf_session_progress handover doc).
    """
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(
        freq_max=20.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port(
        position=(0.010, 0.010, 0.015),
        face="top",
        pin_length=15.0e-3,  # extends from gap (z=15mm) to cavity floor (z=0)
        waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8),
    )
    return sim


def test_pec_short_yields_full_reflection_calibration_target():
    sim = _make_pec_short_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11 = np.abs(res.s_params[0, 0, :])
    assert float(np.min(s11)) >= 0.9, (
        f"PEC-short |S11| should be near 1, got {s11.tolist()}"
    )


def test_pec_short_smoke_returns_finite_values():
    """Mechanical smoke check that the PEC-short geometry runs end-to-end.

    Catches plumbing regressions (DFT-plane scope, V/I extraction,
    power-wave decomposition) that would not show up as a magnitude-band
    failure but would still break the pipeline.
    """
    sim = _make_pec_short_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11 = res.s_params[0, 0, :]
    assert np.all(np.isfinite(s11))
    mag = np.abs(s11)
    # W3.5 smoke envelope: measured max|S11| = 1.158 on this fixture
    # (2026-07-03; >1 is the documented non-physical behaviour of the
    # deprecated single-plane V/I method on a lossless short). 1.5 keeps
    # ~30% headroom while catching plumbing blowups.
    assert float(np.max(mag)) < 1.5
    assert float(np.max(mag)) > 0.0


@pytest.mark.parametrize("pin_length_mm", [13.0, 14.0, 15.0, 17.0])
def test_pec_short_calibration_holds_across_pin_lengths(pin_length_mm):
    """PEC-short |S11| stays near 1 across a sweep of pin_length values.

    The pin tip terminates at the cavity floor (z=0), and the V/I plane
    sits at the pin centre, so different pin_lengths put the V/I plane
    at different physical distances from the short. A clean lossless
    reflection must hold |S11| ≥ 0.85 regardless of pin_length; |Γ| = 1
    analytically, the 0.85 floor allows for the ~1-cell PTFE annulus
    discretisation at the fixture's freq_max=20 GHz.
    """
    from rfx.sources.sources import GaussianPulse

    pin_length = pin_length_mm * 1.0e-3
    sim = Simulation(
        freq_max=20.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port(
        position=(0.010, 0.010, 0.018),
        face="top",
        pin_length=pin_length,
        waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8),
    )
    res = sim.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11 = np.abs(res.s_params[0, 0, :])
    assert float(np.min(s11)) >= 0.85, (
        f"PEC-short |S11| should be near 1 at pin_length={pin_length_mm}mm; "
        f"got {s11.tolist()}"
    )


@pytest.mark.parametrize(
    "face,gap_z",
    [
        ("top", 0.015),     # gap at z=15 mm, pin extends -z to floor at z=0
        ("bottom", 0.005),  # gap at z= 5 mm, pin extends +z to ceiling at z=20 mm
    ],
)
def test_pec_short_calibration_holds_across_faces(face, gap_z):
    """PEC-short |S11| stays near 1 for both face='top' and face='bottom'.

    The two faces are mirror geometries: face='top' emits a wave in -z and
    reads V/I with the port-direction-aware ``current_sign = -1``;
    face='bottom' emits +z with ``current_sign = +1``. The calibration
    must reproduce the same |S11| ≈ 1 in both cases — a one-sided
    asymmetry would indicate a residual convention bug in the port-
    direction handling of either the source or the V/I extractor.
    """
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(
        freq_max=20.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port(
        position=(0.010, 0.010, gap_z),
        face=face,
        pin_length=15.0e-3,  # pin reaches the opposite PEC wall
        waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8),
    )
    res = sim.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11 = np.abs(res.s_params[0, 0, :])
    assert float(np.min(s11)) >= 0.9, (
        f"PEC-short |S11| should be near 1 at face={face!r}; got {s11.tolist()}"
    )


def test_open_termination_helpers_run_without_error():
    """Smoke check that pin-retract + PEC end-cap helpers run end-to-end.

    A clean Γ ≈ +1 open-circuit termination at this fixture's
    resolution is not yet achievable: shallow pin retraction (1–3 cells)
    leaves a below-cutoff PTFE-shell waveguide region too short to fully
    confine the wave, so the dominant reflection comes from the cavity
    floor PEC and looks short-like (phase ≈ 180°). Adding a PEC end-cap
    closes the shell and isolates the line from the cavity, but the
    resulting closed below-cutoff cup has discrete-grid resonances that
    drag |S11| below 1. Testing the resulting calibration target needs
    a finer grid AND a redesigned fixture; we keep this smoke test so
    the helpers don't regress structurally while that work is staged.
    """
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(
        freq_max=20.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port(
        position=(0.010, 0.010, 0.018),
        face="top",
        pin_length=15.0e-3,
        waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8),
    )
    sim.add_coaxial_open_termination(pin_retract_cells=3)
    sim.add_coaxial_pec_end_cap()
    res = sim.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11 = res.s_params[0, 0, :]
    assert np.all(np.isfinite(s11))
    # W3.5 smoke envelope: measured max|S11| = 1.144 on this fixture
    # (2026-07-03) — same 1.5 envelope rationale as the other smoke gates.
    assert float(np.max(np.abs(s11))) < 1.5
    # The result must differ from a vanilla (no-helper) PEC short — if it
    # didn't, the helpers would be no-ops.
    sim_short = Simulation(
        freq_max=20.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim_short.add_coaxial_port(
        position=(0.010, 0.010, 0.018),
        face="top",
        pin_length=15.0e-3,
        waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8),
    )
    res_short = sim_short.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    delta_max = float(np.max(np.abs(s11 - res_short.s_params[0, 0, :])))
    assert delta_max > 0.05, (
        f"open-termination helpers should perturb |S11| relative to a "
        f"plain PEC short; got max |Δ| = {delta_max}"
    )


def test_matched_load_absorbs_at_design_band():
    """Distributed annular Z₀ termination absorbs at high-freq design band.

    A single-cell matched load via ``add_coaxial_matched_load`` is
    electrically thin compared to the source pulse's longest wavelengths,
    so it does not match perfectly at low frequency — this is a known
    physical limit of a thin lossy slab, not a calibration bug.
    Stacking several slabs in series across multiple Yee cells gives an
    increasingly good match at high frequencies. We assert that the
    upper-band ``|S11|`` is markedly below the open-circuit baseline,
    which is the minimum signature that the load is absorbing rather
    than reflecting the incident wave.
    """
    from rfx.sources.sources import GaussianPulse
    from rfx.sources.coaxial_port import (
        SMA_OUTER_RADIUS,
        SMA_PIN_RADIUS,
        PTFE_EPS_R,
        coaxial_tem_characteristic_impedance,
    )

    z_tem = float(
        coaxial_tem_characteristic_impedance(
            SMA_PIN_RADIUS, SMA_OUTER_RADIUS, PTFE_EPS_R
        )
    )

    def _build_sim():
        sim = Simulation(
            freq_max=20.0e9,
            domain=(0.020, 0.020, 0.020),
            boundary="cpml",
        )
        sim.add_coaxial_port(
            position=(0.010, 0.010, 0.018),
            face="top",
            pin_length=12.0e-3,
            waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8),
        )
        return sim

    # Open-circuit baseline (no load placed) for the same geometry.
    sim_open = _build_sim()
    res_open = sim_open.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11_open = np.abs(res_open.s_params[0, 0, :])

    # Stacked five-cell parallel absorber: each slab carries 5·Z₀ so the
    # parallel combination looks like Z₀.
    sim_load = _build_sim()
    for offset in range(1, 6):
        sim_load.add_coaxial_matched_load(
            target_impedance=5.0 * z_tem, axial_offset_cells=offset
        )
    res_load = sim_load.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11_load = np.abs(res_load.s_params[0, 0, :])

    # Upper-band gates: at the auto-freq tail (15.5–20 GHz, last two bins
    # for freq_max=20 GHz), the loaded line should reflect markedly less
    # than the open baseline. Both gates are conservative — the
    # five-cell absorber typically gives ≤ 0.25 at these frequencies vs
    # ≥ 0.95 for the open termination.
    assert float(s11_load[-1]) <= 0.5, (
        f"matched-load |S11| at 20 GHz should be ≤ 0.5; got {s11_load.tolist()}"
    )
    assert float(s11_load[-1]) < 0.6 * float(s11_open[-1]), (
        f"matched-load |S11| should be markedly below open baseline; "
        f"load={s11_load.tolist()}, open={s11_open.tolist()}"
    )


def test_pec_short_phase_rotates_with_pin_length():
    """Pin-length sweep changes V/I-plane to short distance → S11 phase rotates.

    The S11 phase at a given frequency is approximately π − 2β·d where d
    is the V/I-plane to PEC-short distance. Doubling the pin span doubles
    d (V/I plane sits at pin centre), so the S11 phase difference should
    be ~2β·Δd at each frequency. We do not require analytic agreement at
    this discretisation; we only require the phases to be **distinct**
    across pin_length, which is the minimum sanity check that the
    S-matrix carries phase information at all.
    """
    from rfx.sources.sources import GaussianPulse

    phases_at_11ghz = []
    for pin_mm in (12.0, 14.0, 16.0):
        sim = Simulation(
            freq_max=20.0e9,
            domain=(0.020, 0.020, 0.020),
            boundary="pec",
        )
        sim.add_coaxial_port(
            position=(0.010, 0.010, 0.018),
            face="top",
            pin_length=pin_mm * 1.0e-3,
            waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8),
        )
        res = sim.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
        # Auto-freqs at freq_max=20 GHz puts the third bin at 11 GHz.
        phases_at_11ghz.append(np.angle(res.s_params[0, 0, 2]))

    phases_at_11ghz = np.asarray(phases_at_11ghz)
    diffs = np.abs(np.diff(phases_at_11ghz))
    # Phase shifts on the unit circle wrap, so we just require strictly
    # nonzero differences (well above DFT-floor noise).
    assert np.all(diffs > 1e-3), (
        f"S11 phase at 11 GHz should differ across pin_length sweep; "
        f"got phases (rad) {phases_at_11ghz.tolist()}"
    )
