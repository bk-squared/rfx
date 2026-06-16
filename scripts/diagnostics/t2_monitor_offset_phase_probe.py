"""T2.1 R5 prototype — convention-INDEPENDENT monitor-offset phase witness.

Goal: validate, on real rfx FDTD output, that the *physical* propagation
phase between the port's two recorded planes (reference @ ``reference_x_m``
and probe @ ``probe_x_m``) tracks the analytic guided β — WITHOUT calling
rfx's own ``_compute_beta`` / ``_shift_modal_waves`` on the measured leg.

Why this is not a tautology (critic C2):
  forward(plane) = 0.5 * (V(plane) + z_mode * I(plane))
The modal impedance ``z_mode`` (which internally calls ``_compute_beta``)
is IDENTICAL at both planes (same freqs / f_cutoff / dt / dx), so in the
RATIO ``forward_probe / forward_ref`` it cancels exactly, leaving only the
physical FDTD propagation ``exp(-j*beta_phys*Δx)``. The measured leg is
therefore convention-free. The predicted leg uses an INDEPENDENT continuous
β = sqrt(k² - kc²) computed in numpy float64 — never rfx's β.

This is an R5 inspection script (full per-frequency dump), not a committed
gate. It sets up the numbers (continuous-vs-discrete β gap, noise floor)
that T2.4 will turn into a physics-derived tolerance.

Run: python scripts/diagnostics/t2_monitor_offset_phase_probe.py
"""

from __future__ import annotations

import numpy as np

from rfx.api import Simulation
from rfx.sources.waveguide_port import (
    _extract_port_waves_from_time_series,
    _compute_beta,
    C0_LOCAL,
)

# Canonical 40 mm x 20 mm guide (matches the validation battery).
# Port moved interior (x=0.025 -> ref/probe > CPML edge ~18.7 mm) and band
# kept single-mode (< 0.9*fc_TE01 = 6.745 GHz) per the preflight advisories
# the first run surfaced. fc_TE10 = c/(2*0.04) = 3.75 GHz.
DOMAIN = (0.12, 0.04, 0.02)
PORT_X = 0.025
F_CUTOFF_HZ = 3.75e9  # TE10 cutoff = c / (2*0.04)
BAND_HZ = (4.2e9, 6.5e9)


def _build_single_port_sim(freqs_hz, direction: str, port_x: float):
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=max(float(freqs[-1]), f0),
        domain=DOMAIN,
        boundary="cpml",
        cpml_layers=10,
    )
    sim.add_waveguide_port(
        port_x,
        direction=direction,
        mode=(1, 0),
        mode_type="TE",
        freqs=np.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform="modulated_gaussian",
        n_modes=1,
        name="p",
    )
    return sim, freqs


def _beta_continuous_np(freqs_hz: np.ndarray) -> np.ndarray:
    """INDEPENDENT analytic continuous β in numpy float64 (no rfx)."""
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)
    k = omega / float(C0_LOCAL)
    kc = 2.0 * np.pi * float(F_CUTOFF_HZ) / float(C0_LOCAL)
    beta_sq = k * k - kc * kc
    return np.sqrt(np.maximum(beta_sq, 0.0))  # propagating band only


def probe(direction: str, port_x: float, label: str):
    freqs = np.linspace(BAND_HZ[0], BAND_HZ[1], 12)
    sim, freqs = _build_single_port_sim(freqs, direction, port_x)
    result = sim.run(num_periods=40, compute_s_params=False, skip_preflight=False)
    cfg = result.waveguide_ports["p"]

    # --- MEASURED leg (convention-free): the LOCAL-INCIDENT (source-launched)
    #     wave at the two recorded planes, NO de-embed (ref_shift =
    #     probe_shift = 0 -> z_mode cancels in the probe/ref ratio).
    #     _extract_port_waves_from_time_series returns (incident, outgoing)
    #     already direction-resolved: incident = global +x for a "+x" port,
    #     global -x for a "-x" port. Differencing incident@probe vs
    #     incident@ref isolates the physical FDTD propagation phase.
    inc_ref, _ = _extract_port_waves_from_time_series(cfg, cfg.v_ref_t, cfg.i_ref_t)
    inc_probe, _ = _extract_port_waves_from_time_series(cfg, cfg.v_probe_t, cfg.i_probe_t)
    fwd_ref = np.asarray(inc_ref)
    fwd_probe = np.asarray(inc_probe)

    step_sign = 1 if direction.startswith("+") else -1

    dx_planes = float(cfg.probe_x_m) - float(cfg.reference_x_m)  # global metres
    # Measured phase advance of the forward wave from ref->probe plane.
    ratio = fwd_probe / np.where(np.abs(fwd_ref) > 0, fwd_ref, 1.0)
    measured_dphi = np.angle(ratio)  # radians, in (-pi, pi]

    # --- PREDICTED leg: independent continuous β (numpy f64).
    beta_np = _beta_continuous_np(freqs)
    # A +x forward wave ~ exp(-j beta x): moving +Δx advances phase by -beta*Δx.
    # local propagation distance = step_sign * dx_planes.
    predicted_dphi = -beta_np * (step_sign * dx_planes)
    # wrap predicted into (-pi, pi] for fair comparison
    predicted_wrapped = np.angle(np.exp(1j * predicted_dphi))

    # --- rfx Yee-DISCRETE β (for the continuous-vs-discrete gap; NOT used to
    #     gate, only to size the T2.4 tolerance term).
    beta_yee = np.asarray(
        _compute_beta(freqs, F_CUTOFF_HZ, dt=float(cfg.dt), dx=float(cfg.dx))
    ).real
    predicted_yee = np.angle(np.exp(1j * (-beta_yee * (step_sign * dx_planes))))

    resid_cont = np.angle(np.exp(1j * (measured_dphi - predicted_wrapped)))
    resid_yee = np.angle(np.exp(1j * (measured_dphi - predicted_yee)))

    print(f"\n{'='*78}\n[{label}] direction={direction}  port_x={port_x}")
    print(f"  reference_x_m={cfg.reference_x_m:.6f}  probe_x_m={cfg.probe_x_m:.6f}"
          f"  Δx={dx_planes*1e3:.4f} mm  dx={float(cfg.dx)*1e3:.4f} mm  step_sign={step_sign}")
    print(f"  {'f[GHz]':>7} {'|fwd_ref|':>10} {'βΔx[°]':>9} {'meas[°]':>9}"
          f" {'pred_c[°]':>9} {'pred_yee[°]':>11} {'resΔc[°]':>9} {'resΔyee[°]':>10}")
    deg = np.degrees
    for i, f in enumerate(freqs):
        print(f"  {f/1e9:7.3f} {abs(fwd_ref[i]):10.3e} "
              f"{deg(beta_np[i]*dx_planes):9.3f} {deg(measured_dphi[i]):9.3f} "
              f"{deg(predicted_wrapped[i]):9.3f} {deg(predicted_yee[i]):11.3f} "
              f"{deg(resid_cont[i]):9.4f} {deg(resid_yee[i]):10.4f}")
    print(f"  --> max|resid vs continuous β| = {deg(np.max(np.abs(resid_cont))):.4f}°")
    print(f"  --> max|resid vs Yee-discrete β| = {deg(np.max(np.abs(resid_yee))):.4f}°")
    print(f"  --> continuous-vs-discrete β gap  = "
          f"{deg(np.max(np.abs(predicted_wrapped - predicted_yee))):.4f}° (sets T2.4 tol term)")
    return {
        "max_resid_continuous_deg": float(deg(np.max(np.abs(resid_cont)))),
        "max_resid_yee_deg": float(deg(np.max(np.abs(resid_yee)))),
        "dx_planes_mm": dx_planes * 1e3,
    }


if __name__ == "__main__":
    print("T2.1 monitor-offset phase witness — R5 prototype")
    probe("+x", PORT_X, "FORWARD (+x) port")
    probe("-x", DOMAIN[0] - PORT_X, "REVERSE (-x) port — step_sign bug witness")
