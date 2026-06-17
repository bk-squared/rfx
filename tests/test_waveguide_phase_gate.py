"""T2.1 — Convention-INDEPENDENT waveguide phase gate.

The headline broad-E5/E4 S-parameter gate was magnitude-only (framework audit
2026-06-16, finding #7: phase-blind). This battery adds a phase witness that is
**approximately independent of any cross-solver phase convention** — so it
sidesteps the cv11 143° saga (Meep/OpenEMS/Palace disagree 100°+ on absolute
phase) while still catching a wrong measured-propagation-phase (a mis-selected
V/I component, a gross modal-impedance error). A SEPARATE de-embed test below
(`test_deembed_step_sign_rotation_correct`) covers the historical
``_shift_modal_waves`` ``step_sign`` −direction bug, which this monitor-offset
witness does NOT exercise (it runs at ``ref_shift = probe_shift = 0``).

Mechanism (approximately convention-free): the port records modal V/I at two
physical planes — reference @ ``reference_x_m`` and probe @ ``probe_x_m``. The
local-incident wave at each plane is

    incident(plane) = 0.5 * (V(plane) ± z_mode * I(plane))

and ``z_mode`` (which internally calls ``_compute_beta``) is IDENTICAL at both
planes, so in the ratio ``incident_probe / incident_ref`` it cancels **to first
order in the residual reflection**. On this guide the reference-plane reflection
``|b/a|`` is ~6–15% (CPML floor, weaker near the band edges), so the cancellation
is approximate, not exact: a gross ``z_mode`` error (e.g. 2×) still shifts the
measured phase by a few degrees and is caught — which is GOOD (the gate is not a
rubber stamp) but means the measured leg is NOT strictly β-free. The PREDICTED
phase uses an INDEPENDENT continuous β = √(k² − k_c²) computed in numpy float64 —
never rfx's ``_compute_beta`` / ``_shift_modal_waves``. The predictor being an
independent source (not rfx's β) + the slope-tracking assertion (recovers the
geometric plane separation) are what make this a real catch rather than a
tautology.

R5 evidence + tolerance derivation: docs/research_notes/20260616_t2.1_phase_witness_r5.md
(masked healthy residual ~3.8° both directions; a mis-selected-component
extractor bug gave 58–180° — a >10× gate window).
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.waveguide_port import (
    _extract_port_waves_from_time_series,
    extract_waveguide_port_waves,
    _compute_beta,
    C0_LOCAL,
)

# Canonical 40 mm x 20 mm guide (matches the validation battery), but with the
# port moved interior so the reference/probe planes sit OUTSIDE the CPML, and
# the band kept single-mode (< 0.9·fc_TE01 = 6.745 GHz) — both per the preflight
# advisories the R5 prototype surfaced. fc_TE10 = c/(2·0.04) = 3.75 GHz.
DOMAIN = (0.12, 0.04, 0.02)
F_CUTOFF_HZ = 3.75e9
BAND_HZ = (4.2e9, 6.5e9)
N_FREQS = 12

# Gate tolerance: the masked (propagating-band) healthy residual is ~3.8° both
# directions (R5); broken cases are 58–180°. 6° is a ~1.6× margin over health and
# far below any gross phase error. (The 5.1° figure in early notes was UNmasked,
# i.e. included the near-cutoff weak-signal edge bins the mask drops.)
PHASE_TOL_DEG = 6.0
# Weak-signal mask: gate only bins with incident magnitude >= this fraction of
# the band peak (drops the near-cutoff / band-edge bins where S/N collapses).
# Mirrors cv11's |S_ref| FP-null mask and compare_sparameter_reference's
# min_phase_mag.
MAG_MASK_FRAC = 0.05


def _beta_continuous_np(freqs_hz: np.ndarray) -> np.ndarray:
    """INDEPENDENT analytic continuous β in numpy float64 (no rfx involvement)."""
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)
    k = omega / float(C0_LOCAL)
    kc = 2.0 * np.pi * float(F_CUTOFF_HZ) / float(C0_LOCAL)
    return np.sqrt(np.maximum(k * k - kc * kc, 0.0))


def _measure_phase(direction: str, port_x: float):
    """Run an empty matched guide and return the convention-free phase data.

    Returns a dict with measured Δφ (probe vs ref, from the FDTD fields with NO
    de-embed), the independent predicted Δφ, β_np, the local plane separation,
    and the incident magnitude mask.
    """
    freqs = np.linspace(BAND_HZ[0], BAND_HZ[1], N_FREQS)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=float(freqs[-1]), domain=DOMAIN, boundary="cpml", cpml_layers=10,
    )
    sim.add_waveguide_port(
        port_x, direction=direction, mode=(1, 0), mode_type="TE",
        freqs=np.asarray(freqs), f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian", n_modes=1, name="p",
    )
    result = sim.run(num_periods=40, compute_s_params=False)
    cfg = result.waveguide_ports["p"]

    # MEASURED: local-incident wave at the two planes, NO de-embed (ref_shift =
    # probe_shift = 0). _extract_port_waves_from_time_series returns the
    # direction-resolved (incident, outgoing); z_mode cancels in the ratio.
    inc_ref, _ = _extract_port_waves_from_time_series(cfg, cfg.v_ref_t, cfg.i_ref_t)
    inc_probe, _ = _extract_port_waves_from_time_series(cfg, cfg.v_probe_t, cfg.i_probe_t)
    inc_ref = np.asarray(inc_ref)
    inc_probe = np.asarray(inc_probe)
    measured = np.angle(inc_probe / np.where(np.abs(inc_ref) > 0, inc_ref, 1.0))

    step_sign = 1 if direction.startswith("+") else -1
    dx_planes = float(cfg.probe_x_m) - float(cfg.reference_x_m)
    dx_local = step_sign * dx_planes  # local propagation distance, ref -> probe

    beta_np = _beta_continuous_np(freqs)
    predicted = np.angle(np.exp(1j * (-beta_np * dx_local)))  # wrapped

    mag = np.abs(inc_ref)
    mask = mag >= MAG_MASK_FRAC * float(mag.max())
    return {
        "freqs": freqs, "measured": measured, "predicted": predicted,
        "beta_np": beta_np, "dx_local": dx_local, "mask": mask, "mag": mag,
    }


def _wrap(x):
    return np.angle(np.exp(1j * x))


@pytest.mark.parametrize("direction,port_x", [("+x", 0.025), ("-x", 0.095)])
def test_monitor_offset_phase_tracks_independent_beta(direction, port_x):
    """Measured propagation phase matches an INDEPENDENT analytic β per-freq.

    Convention-free (z_mode cancels in the probe/ref ratio); exercises both the
    "+x" and "-x" (step_sign) ports.
    """
    d = _measure_phase(direction, port_x)
    resid_deg = np.degrees(np.abs(_wrap(d["measured"] - d["predicted"])))
    masked = resid_deg[d["mask"]]
    print(f"\n[phase-gate {direction}] gated bins={d['mask'].sum()}/{len(d['mask'])}"
          f"  max|resid|={masked.max():.3f}°  (tol {PHASE_TOL_DEG}°)")
    assert d["mask"].sum() >= 5, "too few propagating-band bins survived the mask"
    assert masked.max() <= PHASE_TOL_DEG, (
        f"{direction} phase residual {masked.max():.2f}° exceeds {PHASE_TOL_DEG}° "
        f"— extractor phase / de-embed regression"
    )


@pytest.mark.parametrize("direction,port_x", [("+x", 0.025), ("-x", 0.095)])
def test_phase_slope_recovers_known_plane_separation(direction, port_x):
    """Anti-tautology: measured Δφ vs independent β has slope = −(plane sep).

    A value hardwired to the predictor cannot recover the geometrically-known
    plane separation from the FDTD fields; recovering it (high R²) proves the
    measured leg is the physical propagation, not an echo of the predictor.
    """
    d = _measure_phase(direction, port_x)
    m = d["mask"]
    beta = d["beta_np"][m]
    meas = np.unwrap(d["measured"][m])
    # Linear fit meas = slope·beta + c ; slope should equal -dx_local.
    slope, intercept = np.polyfit(beta, meas, 1)
    ss_res = np.sum((meas - (slope * beta + intercept)) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(f"\n[slope {direction}] fit slope={slope*1e3:.4f} mm  "
          f"expected={-d['dx_local']*1e3:.4f} mm  R²={r2:.6f}")
    # R² ≥ 0.998 (observed ~0.9991) — 0.999 sat on a knife-edge cross-machine.
    assert r2 >= 0.998, f"phase-vs-β fit not linear (R²={r2:.5f})"
    # Slope recovers the physical plane separation to within 5 % (observed ~3.7 %
    # deficit). The deficit is SYSTEMATIC, not noise: the predictor is continuous
    # β while the FDTD propagates the Yee-discrete β, plus the first-order-only
    # z_mode cancellation under ~6–15 % residual reflection. A wrong-sign / wrong-β
    # extractor would be 100 %+ off, so 5 % still strongly discriminates.
    assert abs(slope - (-d["dx_local"])) <= 0.05 * abs(d["dx_local"]), (
        f"recovered plane sep {slope*1e3:.3f} mm != known {-d['dx_local']*1e3:.3f} mm"
    )


def test_perturbed_predictor_fails_the_gate():
    """Falsifier: a 10 % β error must break the gate (non-vacuous).

    Proves the predicted leg is load-bearing. The healthy residual is
    single-signed (measured lags β slightly), so the effect is ASYMMETRIC: a
    −10 % perturbation ADDS to the residual (~12°, robustly fails) while a +10 %
    perturbation partially CANCELS (~6.7°, clears the 6° gate by only ~0.7°). We
    require BOTH the worst-sign to fail by a comfortable margin AND each sign to
    measurably move the residual, so the assertion does not hinge on the fragile
    +10 % arm. Honest statement: the gate robustly catches a 10 % β error in the
    adding direction; one-sided sensitivity is ~10 %.
    """
    d = _measure_phase("+x", 0.025)
    healthy = float(np.degrees(np.abs(_wrap(d["measured"] - d["predicted"])))[d["mask"]].max())
    per_sign = {}
    for scale in (1.10, 0.90):
        predicted_pert = _wrap(-scale * d["beta_np"] * d["dx_local"])
        resid_deg = np.degrees(np.abs(_wrap(d["measured"] - predicted_pert)))
        per_sign[scale] = float(resid_deg[d["mask"]].max())
    worst = max(per_sign.values())
    print(f"\n[perturbed β] healthy={healthy:.3f}°  +10%={per_sign[1.10]:.3f}°  "
          f"-10%={per_sign[0.90]:.3f}°  (gate {PHASE_TOL_DEG}°)")
    # Worst-sign must fail with margin (robust, not knife-edge).
    assert worst > PHASE_TOL_DEG + 2.0, (
        "a 10% β perturbation did not robustly exceed the gate — too loose"
    )
    # Both signs must measurably worsen the residual (the predictor is load-bearing).
    assert min(per_sign.values()) > healthy + 1.0, (
        "a 10% β perturbation barely moved the residual — gate not β-sensitive"
    )


@pytest.mark.parametrize("direction,port_x", [("+x", 0.025), ("-x", 0.095)])
def test_deembed_step_sign_rotation_correct(direction, port_x):
    """Covers the historical ``_shift_modal_waves`` ``step_sign`` −direction bug.

    The monitor-offset witness above runs at ``ref_shift=0`` and never touches
    the de-embed path, so it cannot catch the 2026-04-22 bug where
    ``_shift_modal_waves`` ignored ``step_sign`` and applied the "+x" rotation to
    "-x" ports (cv11 phase offset never closed). This test exercises that path
    directly: de-embedding the incident wave by Δ rotates its phase by exactly
    ``-β·(step_sign·Δ)``. We compare the rfx de-embed rotation to an INDEPENDENT
    numpy-double β·Δ with the CORRECT sign; a reintroduced step_sign bug flips the
    sign for "-x", giving a ``2·β·Δ`` (~50°) error that this gate catches.

    The de-embed rotation is a pure analytic ``exp(-jβ·shift)`` multiply (no
    reflection contamination), so the healthy residual is just the
    continuous-vs-Yee β gap (<1°); the tolerance is 3°.
    """
    freqs = np.linspace(BAND_HZ[0], BAND_HZ[1], N_FREQS)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=float(freqs[-1]), domain=DOMAIN, boundary="cpml", cpml_layers=10,
    )
    sim.add_waveguide_port(
        port_x, direction=direction, mode=(1, 0), mode_type="TE",
        freqs=np.asarray(freqs), f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian", n_modes=1, name="p",
    )
    cfg = sim.run(num_periods=40, compute_s_params=False).waveguide_ports["p"]

    shift_m = 0.004  # 4 mm de-embed; β·Δ ~ 20–30° (no 2π wrap)
    a0, _ = extract_waveguide_port_waves(cfg, ref_shift=0.0)
    a1, _ = extract_waveguide_port_waves(cfg, ref_shift=shift_m)
    a0 = np.asarray(a0)
    a1 = np.asarray(a1)
    measured_rot = np.angle(a1 / np.where(np.abs(a0) > 0, a0, 1.0))

    step_sign = 1 if direction.startswith("+") else -1
    beta_np = _beta_continuous_np(freqs)
    predicted_rot = _wrap(-beta_np * (step_sign * shift_m))  # CORRECT-sign prediction

    mag = np.abs(a0)
    mask = mag >= MAG_MASK_FRAC * float(mag.max())
    resid_deg = np.degrees(np.abs(_wrap(measured_rot - predicted_rot)))
    # Witness that a step_sign bug WOULD be caught: the wrong-sign prediction.
    wrong_sign = _wrap(-beta_np * (-step_sign * shift_m))
    wrong_resid = np.degrees(np.abs(_wrap(measured_rot - wrong_sign)))
    print(f"\n[de-embed {direction}] max|resid correct-sign|={resid_deg[mask].max():.3f}°  "
          f"max|resid wrong-sign|={wrong_resid[mask].max():.3f}° (bug would be caught)")
    assert resid_deg[mask].max() <= 3.0, (
        f"{direction} de-embed rotation {resid_deg[mask].max():.2f}° != analytic "
        f"-β·(step_sign·Δ) — _shift_modal_waves step_sign regression"
    )
    # Sign-discrimination witness: the opposite-sign rotation (what a step_sign
    # bug produces on a "-x" port) is ~2·β·Δ (~50°) away — far outside the 3° gate,
    # so the bug cannot hide. Holds for both directions (the rotation is signed).
    assert wrong_resid[mask].max() > 10.0, (
        "wrong-sign de-embed not distinguishable — gate would miss a step_sign bug"
    )
