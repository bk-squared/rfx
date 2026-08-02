"""Issue #490 Lane 3 -- group delay near cutoff, gated against L_eff / v_g(f).

PRE-DECLARATION (frozen BEFORE the first measurement run -- do not edit this
section after running; append dated notes below it instead, per the workspace
R2/R3 process rule and the recent same-class incident on this session's other
lanes).

Fixture (purpose-built minimal straight guide, imitating the existing WR-band
broad-E5 fixtures' recipe in ``run_waveguide_band_broad_e5_flux_sweep.py``):

- Cross-section: WR-340 standard, a=86.36 mm, b=43.18 mm -> fc_TE10 = c/(2a)
  = 1.7357 GHz. Chosen because the SAME cross-section's coarse dx=1500 um
  mesh was already measured this session at ~170 s wall-clock/case on this
  CPU-only host (no GPU available) -- the cheapest of the 5 committed WR
  bands in absolute cell count, which is what makes 3 fresh runs (baseline +
  settling-check + domain-invariance) affordable in one session.
- EMPTY guide (no dielectric slab). Group delay from waveguide dispersion
  alone is the textbook oracle tau_g(f) = L_eff / v_g(f); a slab's Airy
  multi-reflection phase would entangle a second physical mechanism into the
  same gate, which the #490 spec's own formula (L/v_g) does not ask for.
- Domain (baseline): 0.560 x 0.08636 x 0.04318 m. Ports at x=0.100 m (+x) and
  x=0.460 m (-x), TE10, ``normalize='flux'`` (documented-convergent recipe,
  matches every other committed waveguide envelope in this repo).
- Reference planes: x=0.120 m and x=0.440 m -> L_eff = 0.320 m (the physical
  length the group delay is defined over -- NOT the full domain or the
  port-to-port span).
- CPML: 24 layers (matches the committed WR-band fixtures' convention, for
  comparability). This is EXPECTED to trip the same thin-absorber-vs-guide-
  wavelength preflight advisory the committed WR-340 magnitude fixture
  already carries (buffer 100 mm vs required >=0.5*lambda_g) -- quoted
  verbatim in the report, not silently suppressed. It is a pre-existing
  modeling choice shared with the rest of the committed evidence base, not a
  new problem introduced by this fixture.
- Frequency band: 2.0-2.6 GHz, 11 points -> f/fc in [1.152, 1.498].
  fc-margin choice: 1.152 (15.2% above cutoff) is NOT asymptotically close --
  chosen deliberately so the run settles in minutes on CPU rather than hours
  (group velocity collapses approaching cutoff, so does settling time). This
  keeps the campaign honest rather than silently sampling a convenient band;
  the true near-singularity behaviour (f/fc -> 1) is explicitly OUT of scope
  here and is named as such in the report. Even at this margin the analytic
  tau_g spans 1.434-2.148 ns (a 1.50x ratio) -- a real curvature to gate
  against, not a flat line.

Measurement:
- tau_g_measured(f) = -d(phase(S21))/d(omega), phase = np.unwrap(np.angle(S21))
  across the 11-point band (unwrapping is a documented no-op here: adjacent-
  bin phase steps are ~36 deg, far below the 180 deg wrap threshold, verified
  in the report -- included anyway because Lane 1's convention note requires
  a stated unwrapping rule, and because the falsifier below needs an
  UNWRAPPED baseline to meaningfully corrupt).
- Central difference (2nd-order accurate, O(domega^2) truncation error) at
  the 9 interior nodes; the 2 endpoints use a one-sided (forward/backward,
  O(domega), 1st-order) difference and are reported but NOT gated (stated
  explicitly, not silently dropped).
- Settling witness: ``compute_waveguide_s_matrix`` does not expose a built-in
  energy-based ``settling_db`` for the waveguide port family (unlike
  ``compute_msl_s_matrix``). This lane's settling evidence is therefore a
  RECORD-LENGTH CONVERGENCE check: the same case is run at num_periods=60 and
  num_periods=120; if tau_g(f) barely moves, the num_periods=60 record was
  already settled (an under-settled truncated transient would move
  materially when the record doubles). This is a deliberate, honest
  substitute for the MSL-style dB witness, not an oversight.

Qualitative expectations (declared BEFORE running; a failure here means STOP
and report, not tune):
1. tau_g(f) decreases monotonically with frequency (v_g increases away from
   cutoff).
2. The measured curve is smooth -- no branch jumps after unwrapping.
3. tau_g(f=2.0 GHz) / tau_g(f=2.6 GHz) is close to the analytic ratio 1.498
   (order-of-magnitude and sign check, not a tight gate).
4. num_periods=60 and num_periods=120 agree to within a small fraction of the
   analytic curvature (the settling witness).

Stop rule: if (1)-(3) fail on the num_periods=60 run, STOP this lane, dump the
full per-bin phase/tau_g trace, and report -- do not adjust the band or
domain to make it pass.

--- end of frozen pre-declaration ---

POST-RUN CORRECTION (2026-08-02): the no-op claim is false -- the absolute
phase (-387 deg across the band) wraps once even though per-bin steps are
~36 deg; the skip_unwrap falsifier demonstrates the consequence at 8.35 ns;
'verified in the report' referenced no artifact and is retracted. Measured
directly: the raw (wrapped) angle(S21) crosses the +-180 deg boundary
exactly once across the 11-point band, and ``np.unwrap`` applies a 360 deg
correction there -- unwrapping is load-bearing for this fixture, not a
no-op. This correction is appended per the #535 rule (do not edit the frozen
block above; append instead) after an adversarial review of PR #536 caught
the false claim.

HONESTY NOTE ON THE FREEZE ITSELF: unlike Lane 1 (whose phase envelope
landed in an earlier commit before its falsifier/domain-invariance work, so
git history plus background-job timestamps corroborate the stated order of
operations), this file's pre-declaration above and its 3 measurement runs'
results were committed together in ONE commit. Git history alone cannot
prove the pre-declaration was actually written before the runs executed --
that ordering rests on the author's account, not on an independently
checkable artifact. Stated plainly per the same adversarial review, not
hidden.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

C0 = 299_792_458.0
A_M = 86.36e-3
B_M = 43.18e-3
FC_TE10_HZ = C0 / (2 * A_M)
BAND_HZ = (2.0e9, 2.6e9)
N_FREQS = 11
CPML_LAYERS = 24
DX_M = 1500e-6

PORT_LEFT_X_M = 0.100
PORT_RIGHT_X_M = 0.460
REF_LEFT_X_M = 0.120
REF_RIGHT_X_M = 0.440
DOMAIN_X_M = 0.560
L_EFF_M = REF_RIGHT_X_M - REF_LEFT_X_M

OUT_DIR = REPO / ".omx/physics-gate/2026-08-02-waveguide-wr340-near-cutoff-group-delay"


def analytic_tau_g(freqs_hz: np.ndarray, L_eff: float = L_EFF_M) -> np.ndarray:
    """tau_g(f) = L_eff / v_g(f), v_g = c*sqrt(1-(fc/f)^2) for TE10."""
    ratio = FC_TE10_HZ / freqs_hz
    vg = C0 * np.sqrt(1.0 - ratio ** 2)
    return L_eff / vg


def build_sim(*, domain_x_m: float, port_left_x_m: float, port_right_x_m: float,
              ref_left_x_m: float, ref_right_x_m: float, freqs: np.ndarray):
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(domain_x_m, A_M, B_M),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=CPML_LAYERS, dx=DX_M,
    )
    pf = jnp.asarray(freqs)
    f0 = float(np.mean(freqs))
    bandwidth = (freqs[-1] - freqs[0]) / f0 * 1.3  # comfortable margin over the band
    sim.add_waveguide_port(
        port_left_x_m, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=ref_left_x_m, name="left",
    )
    sim.add_waveguide_port(
        port_right_x_m, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=ref_right_x_m, name="right",
    )
    return sim


def run_case(tag: str, *, num_periods: float, domain_x_m: float,
             port_left_x_m: float, port_right_x_m: float,
             ref_left_x_m: float, ref_right_x_m: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freqs = np.linspace(*BAND_HZ, N_FREQS)
    t0 = time.time()
    sim = build_sim(
        domain_x_m=domain_x_m, port_left_x_m=port_left_x_m,
        port_right_x_m=port_right_x_m, ref_left_x_m=ref_left_x_m,
        ref_right_x_m=ref_right_x_m, freqs=freqs,
    )
    result = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize="flux")
    dt = time.time() - t0
    s = np.asarray(result.s_params)
    pi = {n: i for i, n in enumerate(result.port_names)}
    s21 = s[pi["right"], pi["left"], :]
    s11 = s[pi["left"], pi["left"], :]
    fr = np.asarray(result.freqs)
    out_path = OUT_DIR / f"{tag}.npz"
    np.savez(out_path, freqs_hz=fr, s11=s11, s21=s21,
              num_periods=num_periods, domain_x_m=domain_x_m,
              port_left_x_m=port_left_x_m, port_right_x_m=port_right_x_m,
              ref_left_x_m=ref_left_x_m, ref_right_x_m=ref_right_x_m,
              l_eff_m=ref_right_x_m - ref_left_x_m, wallclock_s=dt)
    print(f"[{tag}] wallclock={dt:.1f}s -> {out_path}")
    return out_path


CASES = {
    "baseline_np60": dict(
        num_periods=60.0, domain_x_m=DOMAIN_X_M,
        port_left_x_m=PORT_LEFT_X_M, port_right_x_m=PORT_RIGHT_X_M,
        ref_left_x_m=REF_LEFT_X_M, ref_right_x_m=REF_RIGHT_X_M,
    ),
    "settling_np120": dict(
        num_periods=120.0, domain_x_m=DOMAIN_X_M,
        port_left_x_m=PORT_LEFT_X_M, port_right_x_m=PORT_RIGHT_X_M,
        ref_left_x_m=REF_LEFT_X_M, ref_right_x_m=REF_RIGHT_X_M,
    ),
    # Domain grown by +100mm total (+50mm buffer each side beyond the ports);
    # port-to-reference-plane offsets (20mm) and L_eff (0.320m) held fixed.
    "grown_domain_np60": dict(
        num_periods=60.0, domain_x_m=DOMAIN_X_M + 0.100,
        port_left_x_m=PORT_LEFT_X_M + 0.050, port_right_x_m=PORT_RIGHT_X_M + 0.050,
        ref_left_x_m=REF_LEFT_X_M + 0.050, ref_right_x_m=REF_RIGHT_X_M + 0.050,
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=sorted(CASES.keys()))
    args = p.parse_args()
    run_case(args.case, **CASES[args.case])


if __name__ == "__main__":
    main()
