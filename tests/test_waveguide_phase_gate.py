"""T2.1 — Convention-INDEPENDENT waveguide phase gate.

The headline broad-E5/E4 S-parameter gate was magnitude-only (framework audit
2026-06-16, finding #7: phase-blind). This battery adds a phase witness that is
**independent of any cross-solver phase convention** — so it sidesteps the cv11
143° saga (Meep/OpenEMS/Palace disagree 100°+ on absolute phase) while still
catching a wrong-phase extractor (wrong-sign de-embed, the ``step_sign``
−direction bug).

Mechanism (convention-free): the waveguide port records the modal V/I at two
physical planes — reference @ ``reference_x_m`` and probe @ ``probe_x_m``. The
local-incident wave at each plane is

    incident(plane) = 0.5 * (V(plane) ± z_mode * I(plane))

and ``z_mode`` (which internally calls ``_compute_beta``) is IDENTICAL at both
planes, so in the ratio ``incident_probe / incident_ref`` it cancels exactly,
leaving only the physical FDTD propagation ``exp(-j·β_phys·Δx)``. The MEASURED
phase therefore never touches rfx's β. The PREDICTED phase uses an INDEPENDENT
continuous β = √(k² − k_c²) computed in numpy float64 — never rfx's
``_compute_beta`` / ``_shift_modal_waves``. If the two β were the same source,
the error would cancel and the test would pass anything (a tautology); the
independent predictor + the slope-tracking assertion below are what make this a
real catch.

R5 evidence + tolerance derivation: docs/research_notes/20260616_t2.1_phase_witness_r5.md
(healthy residual ≤ 5.1° both directions; a mis-selected-component extractor bug
gave 58–180° — a 10–30× gate window).
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.waveguide_port import (
    _extract_port_waves_from_time_series,
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
    assert r2 >= 0.999, f"phase-vs-β fit not linear (R²={r2:.5f})"
    # slope recovers the physical plane separation to within 5 %.
    assert abs(slope - (-d["dx_local"])) <= 0.05 * abs(d["dx_local"]), (
        f"recovered plane sep {slope*1e3:.3f} mm != known {-d['dx_local']*1e3:.3f} mm"
    )


def test_perturbed_predictor_fails_the_gate():
    """Falsifier: a 10 % β error must break the gate.

    Proves the predicted leg is load-bearing and the tolerance is tight — a gate
    that a 10 %-wrong β still passes would be vacuous. The healthy residual is
    single-signed (measured lags β slightly), so a +10 % perturbation partially
    cancels it (~6.7°) while a −10 % perturbation adds (~12°); we test BOTH signs
    and require the worst to exceed the gate, which is the robust statement of
    "the gate catches a 10 % β error".
    """
    d = _measure_phase("+x", 0.025)
    worst = 0.0
    for scale in (1.10, 0.90):
        predicted_pert = _wrap(-scale * d["beta_np"] * d["dx_local"])
        resid_deg = np.degrees(np.abs(_wrap(d["measured"] - predicted_pert)))
        worst = max(worst, float(resid_deg[d["mask"]].max()))
    print(f"\n[perturbed ±10% β] worst max|resid|={worst:.3f}° (must exceed {PHASE_TOL_DEG}°)")
    assert worst > PHASE_TOL_DEG, (
        "a 10% β perturbation stayed within tolerance — gate is too loose to "
        "be a real phase check"
    )
