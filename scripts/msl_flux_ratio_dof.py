"""#525: what does R = Re(V·conj(I)) / flux_spectrum actually constrain?

#525 asked for a correctly constructed flux oracle for the MSL modal V·I pair.
This script is the artifact behind closing it with a NEGATIVE result: no oracle
of this form exists, because R constrains strictly less than the S-matrix
depends on.

THE ARGUMENT (algebra + production code, independent of any measurement)
-----------------------------------------------------------------------
R is ONE real constraint: the product Re(V I*) against the guided power.
The S-matrix is assembled from the pair SEPARATELY, against a FIXED reference:

    a = 0.5 * (v0 + z0_hj * i_f)          rfx/api/_sparams.py:2497-2499
    b = 0.5 * (v0 - z0_hj * i_f)
    S = msl_solve_s_from_waves(...)        rfx/api/_sparams.py:2560, 2591
    over the same wave pairs recorded at    rfx/api/_sparams.py:2533-2540

and so does the separately reported characteristic impedance:

    z0 = (alpha - gamma) / (i1 + eps)      rfx/probes/msl_wave_decomp.py:566
    "Characteristic impedance extracted via 3-probe de-embedding"
                                           rfx/api/_spec.py:1284-1286

Modulo the unobservable common phase, (V, I) carries three meaningful real
degrees of freedom — |V|, |I|, arg(V/I). R pins one. Its invariance group at
fixed flux is two-dimensional. So R is NECESSARY BUT NOT SUFFICIENT: it moves
under one-sided defects (it caught #511, going 0.881 -> 1.006 on the V-span bug
— see the msl_modal_voltage docstring at _sparams.py:125-127) but cannot
certify the pair.

Note what is NOT claimed: R is not "blind" as a class. Any V-only or I-only
error moves it proportionally. It is blind only along the product-preserving
direction, and a purely algebraic demonstration of that — rescaling V by a and
I by 1/a — is a TAUTOLOGY, not a measurement: Re(aV·conj(I/a)) = Re(V·conj(I))
identically, for any fixture, healthy or broken. An earlier version of this
script presented that rescale as a pre-declared falsifier. It is not one; the
outcome was unreachable.

WHAT THIS SCRIPT MEASURES INSTEAD
---------------------------------
A real, physically caused excursion in the ratio direction, and whether R
registers it. The line is not perfectly matched, so |V|/|I| varies strongly with
position while the guided power — and therefore R — does not. If R tracks the
ratio, the DOF argument is wrong. If R stays flat through a large ratio swing,
the argument is demonstrated on healthy data with no identity doing the work.

PRE-DECLARED
  D1 |V|/|I| varies by >= 1.5x across the plane set while R stays within 1%
     => R does not register real ratio-direction variation. Close #525 as
        "no oracle of this form exists"; a sufficient oracle must additionally
        constrain the ratio, e.g. (alpha-gamma)/I against a closed form on a
        settled, matched line.
  D2 R tracks |V|/|I|
     => NOT "the DOF argument is wrong". That argument is algebra (R is a
        function of Re(V I*) and the flux alone) plus a code fact (S is
        assembled from the pair separately), and no field measurement can
        falsify either. Reaching D2 would mean the EXTRACTOR or the FLUX
        MONITOR is defective — a useful check in its own right. What the sweep
        contributes to the DOF claim is MAGNITUDE, not proof.

RESULT (2026-08-01, this script, dx = h_sub/3 = 84.67 um, 8000 steps, CPU)

    settling witness (peak-of-tail / peak Ez^2, 95% tail) = -99.9 dB   PASS

      x_mm     R@3.0G    R@4.5G   |V/I|@3.0G   |V/I|@4.5G
      3.50    1.00671   1.00732        42.36        57.34
      5.41    1.00851   1.00906        36.81        45.16
      7.32    1.00905   1.00905        33.03        35.69
      9.23    1.00857   1.00854        30.98        29.93
     10.50    1.00710   1.00716        30.79        28.97
     (12 planes; monotone at 4.5 GHz, monotone at 3.0 GHz except the last
      step, 30.78 -> 30.79)

    3.0 GHz : |V|/|I| swings 30.78 -> 42.36 ohm (1.38x), R varies 0.232%
    4.5 GHz : |V|/|I| swings 28.97 -> 57.34 ohm (1.98x), R varies 0.194%

  Each frequency is paired with ITS OWN R. (An earlier version maximised the
  swing and the R variation over frequency independently and printed them in one
  sentence, reporting a pair that never occurred.)

  WHAT IS CONSERVED, measured (all ptp/mean, one unit — an earlier version
  mixed "1.212x" with "6.47%" and made |V| look like the smaller variation):

    3.0 GHz : |V| 19.6%  |I| 12.5%  |V||I| 6.5%  |  Re(V I*) 0.158%  flux 0.223%
    4.5 GHz : |V| 37.5%  |I| 30.0%  |V||I| 9.0%  |  Re(V I*) 0.153%  flux 0.203%

  A clean monotone story: the individual magnitudes swing hardest, their product
  less (they anti-correlate), and the real part is conserved. |V||I| is NOT
  conserved because the standing wave moves the V-I phase, so the power factor
  varies while the real power does not.

  So R is flat because BOTH its numerator and its denominator are conserved
  along a lossless line — not because "the flux changes and R does not", which
  an earlier version claimed and the data above refutes. Stated correctly the
  demonstration is sharper: THE PLANE SWEEP IS A PHYSICAL TRAVERSAL OF R's
  INVARIANCE SET. Every plane carries the same guided power, so every plane has
  the same Re(V I*), while |V|/|I| spans 1.98x. Nature hands over a
  one-parameter family of (V, I) pairs with constant product and a factor-2
  range of ratios, and R cannot tell them apart. What is measured is the SIZE of
  the blind direction on a real fixture: 98% of ratio range maps to 0.194% of R,
  a sensitivity gap of ~506x.

  D1 CONFIRMED at 4.5 GHz.

  THE PRODUCTION EXTRACTOR on the same planes (extract_msl_nprobe, called
  directly — not a hand-rolled fit, and not a citation of someone's /tmp log):

    3.0 GHz : Z0 = 44.01 +0.35j ohm  (vs HJ 47.89, -8.1%)  |S11| = 0.1780
              eps_eff_fit = 3.0320 (HJ 2.8694)   residual = 2.3e-15
    4.5 GHz : Z0 = 44.01 +0.59j ohm  (-8.1%)               |S11| = 0.2081
              eps_eff_fit = 3.0345                residual = 2.9e-16

  Z0 is constant to the printed digits across frequency, its reactive part is
  ~0.5-1.3% of the real part, and it sits 8.1% below the closed form — inside
  the ">5% expected" Yee-staircase envelope preflight warns about for a
  3-substrate-cell mesh (_sparams.py:2479 records ~20-27% for this class). The
  configuration is HEALTHY. R = 1.008 on it is correct and unremarkable; the
  point is that R would read the same either way.

  WHY THE PRODUCTION PATH AND NOT A HAND-ROLLED FIT. An earlier version fitted
  with beta PINNED to HJ's eps_eff, got Z0 = 45.2 ohm (-5.6%), found a free-beta
  fit disagreeing by ~3%, and recorded that as an unresolvable "method spread".
  It was resolvable, three ways: the free fit has 9-21% lower residual; it makes
  Z0 flatter plane-to-plane (0.29% vs 0.47%); and — decisively — PRODUCTION DOES
  NOT PIN BETA. msl_wave_decomp.py:559-561 calls _estimate_beta, whose docstring
  reads "Robust beta estimate: residual scan around the analytic guess +
  refine"; HJ is only the scan anchor beta0. The production eps_eff lands at
  3.032/3.034, not 2.869. Reporting -5.6% as the extractor's behaviour answered
  a different question than the sentence around it asked. Recording a decidable
  question as undecidable is itself an inaccuracy (PR #531 review).

  (A draft of this paragraph added "and its residual is 2e-15 against the pinned
  fit's 3e-3 — twelve orders". That compared an ABSOLUTE L2 norm in volts —
  msl_wave_decomp.py:425 returns sqrt(sum(|v-pred|^2)) — against a RELATIVE
  norm(.)/norm(V). With ||V|| ~ 7.7e-13 V the "twelve orders" was the
  normalisation. Like for like the scan gains 9.0% / 21.1%, which the sentence
  above already said. Deleted.)

  The |V|/|I| ladder is the standing-wave envelope, NOT Z0, and is not an
  extractor-error measurement.

SETTLING IS PART OF THE RESULT. An earlier version of this script ran 4000
steps inside warnings.simplefilter("ignore"), which suppressed the extractor's
own ring-down witness. That record ended at FULL amplitude (end/peak Ez^2
= -0.0 dB) and its numbers were truncation artifacts that reversed at converged
run length: |V|/|I| at x=5mm/4.5GHz moved 31.19 -> 47.71 ohm (+53%) between 4000
and 8000 steps, and a forward/backward fit gave eps_eff < 1 — impossible for a
bound mode, which is itself the tell. Nothing here is suppressed; the witness is
printed and asserted before any ratio is reported.
"""

import sys
import warnings

sys.path.insert(0, "/root/workspace/byungkwan-workspace/research/rfx")

import numpy as np  # noqa: E402

EPS_R, H_SUB, W = 3.66, 254e-6, 600e-6
DX = H_SUB / 3.0
L, PM, FMAX = 10e-3, 2e-3, 5e9
N_STEPS = 8000
SETTLING_DB = -40.0


def main() -> int:
    import jax.numpy as jnp
    import rfx
    from rfx import GaussianPulse
    from rfx.api import Simulation
    from rfx.api._sparams import msl_modal_voltage
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box
    from rfx.probes.probes import flux_spectrum
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.sources.msl_port import MSLPort, _msl_yz_cells, msl_loop_current

    print(f"rfx.__file__ = {rfx.__file__}")
    if "/root/workspace/byungkwan-workspace/research/rfx" not in rfx.__file__:
        print("FATAL: imported rfx is not this checkout")
        return 2

    lx, ly, lz = L + 2 * PM, W + 2 * (2 * H_SUB + 8 * DX), H_SUB + 1.5e-3
    sim = Simulation(freq_max=FMAX, domain=(lx, ly, lz), dx=DX, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("sub", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (lx, ly, H_SUB)), material="sub")
    yc = ly / 2
    sim.add(Box((0, yc - W / 2, H_SUB), (lx, yc + W / 2, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(PM, yc, 0.0), width=W, height=H_SUB, direction="+x",
                     impedance=50.0, excite=True,
                     waveform=GaussianPulse(f0=FMAX / 2, bandwidth=0.8))
    sim.add_msl_port(position=(PM + L, yc, 0.0), width=W, height=H_SUB,
                     direction="-x", impedance=50.0, excite=False)

    freqs = np.array([3.0e9, 4.5e9])
    fr = jnp.asarray(freqs)
    xs = [PM + 1.5e-3 + i * (7.0e-3 / 11.0) for i in range(12)]
    for p, x in enumerate(xs):
        for c in ("ez", "hy", "hz"):
            sim.add_dft_plane_probe(axis="x", coordinate=float(x), component=c,
                                    freqs=fr, name=f"{c}{p}")
        sim.add_flux_monitor(axis="x", coordinate=float(x), freqs=fr, name=f"F{p}")
    sim.add_probe(position=(PM + 5e-3, yc, H_SUB * 0.5), component="ez")

    print("\n--- PREFLIGHT (verbatim; part of the result) ---")
    sim.preflight()
    print("--- end preflight ---\n")

    # NOT suppressed: the run's own warnings ARE the settling evidence.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.run(n_steps=N_STEPS)
    for w in caught:
        print(f"[RUN WARNING] {str(w.message)[:400]}")

    ts = np.asarray(res.time_series)[:, 0]
    tail = ts[int(0.95 * len(ts)):]
    # max-of-tail, not mean-of-tail: the repo convention, and the one the label
    # names. mean is ~5.6 dB more lenient here (PR #531 review).
    settle = 10 * np.log10(max(float(np.max(tail ** 2)), 1e-300)
                           / max(float(np.max(ts ** 2)), 1e-300))
    print(f"\nsettling witness (peak-of-tail/peak Ez^2, 95% tail): {settle:.1f} dB "
          f"[{'PASS' if settle <= SETTLING_DB else 'FAIL'}, need <= {SETTLING_DB}]")
    if settle > SETTLING_DB:
        print("ABORT: record is not settled. Every ratio below would be a")
        print("truncation artifact. Raise N_STEPS or use until_decay.")
        return 1

    grid = sim._build_grid()
    dt = float(grid.dt)
    mp = MSLPort(feed_x=PM, y_lo=yc - W / 2, y_hi=yc + W / 2, z_lo=0.0, z_hi=H_SUB,
                 direction="+x", impedance=50.0, excitation=None)
    cells = _msl_yz_cells(grid, mp)
    js, ks = sorted({c[1] for c in cells}), sorted({c[2] for c in cells})
    jlo, jhi, klo = js[0], js[-1], ks[0]
    jc = (jlo + jhi) // 2
    pm_mask = np.asarray(sim._assemble_materials(grid)[3])
    kp = np.where(pm_mask[cells[0][0], jc, ks[-1]:])[0]
    ktr = int(ks[-1] + int(kp.min()))

    hs = np.exp(1j * 2 * np.pi * freqs * dt * 0.5).astype(np.complex64)
    ny, nz = np.asarray(res.dft_planes["ez0"].accumulator).shape[1:]
    dza, dya = np.full(nz, DX), np.full(ny, DX)
    z0_hj, eps_eff = hammerstad_jensen_z0_eps_eff(W, H_SUB, EPS_R)
    print(f"trace node (rasterized) = {ktr}")
    print(f"Hammerstad-Jensen: Z0 = {float(z0_hj):.3f} ohm, "
          f"eps_eff = {float(eps_eff):.5f}\n")

    print(f"{'x_mm':>6} " + "".join(f"{'R@%.1fG' % (f/1e9):>10}" for f in freqs)
          + "".join(f"{'|V/I|@%.1fG' % (f/1e9):>13}" for f in freqs))
    Rs, ZZ, VV, II, FF, Vcpx, Icpx = [], [], [], [], [], [], []
    for p, x in enumerate(xs):
        ez = jnp.asarray(res.dft_planes[f"ez{p}"].accumulator)
        hy = jnp.asarray(res.dft_planes[f"hy{p}"].accumulator) * hs[:, None, None]
        hz = jnp.asarray(res.dft_planes[f"hz{p}"].accumulator) * hs[:, None, None]
        V = np.asarray(msl_modal_voltage(ez, j_centre=jc, k_lo=klo, k_hi=ktr,
                                         dz_arr=dza))
        I = np.asarray(msl_loop_current(hy, hz, j_lo=jlo, j_hi=jhi,
                                        k_trace_lo=ktr, k_trace_hi=ktr,
                                        dy_arr=dya, dz_arr=dza, direction="+x"))
        F = np.asarray(flux_spectrum(res.flux_monitors[f"F{p}"]))
        R = np.real(V * np.conj(I)) / F
        Z = np.abs(V) / np.abs(I)
        Rs.append(R); ZZ.append(Z)
        VV.append(np.abs(V)); II.append(np.abs(I)); FF.append(F)
        Vcpx.append(V); Icpx.append(I)
        print(f"{x*1e3:6.2f} " + "".join(f"{r:10.5f}" for r in R)
              + "".join(f"{z:13.2f}" for z in Z))

    Rs, ZZ = np.array(Rs), np.array(ZZ)
    VV, II, FF = np.array(VV), np.array(II), np.array(FF)
    xs_a = np.array(xs)

    # --- what is conserved and what is not -------------------------------
    # R is flat because BOTH its numerator and denominator are conserved on a
    # lossless line, not because "the flux changes and R doesn't". Saying so
    # requires measuring it (PR #531 review, MAJOR 2).
    print()
    def sp(a):
        return float(np.ptp(a) / np.mean(a) * 100)

    for k, f in enumerate(freqs):
        # Re(V conj(I)) — the ACTUAL numerator of R, and the conserved one.
        # |V|*|I| is NOT it and varies several percent: the standing wave moves
        # the V-I phase angle, so the power factor changes while the real power
        # does not. Printing the magnitude product next to a claim about the
        # real part invites exactly the wrong reading (PR #531 review class).
        rp = np.array([float(np.real(Vcpx[j][k] * np.conj(Icpx[j][k])))
                       for j in range(len(xs))])
        mp = VV[:, k] * II[:, k]
        # One unit (ptp/mean %) across the whole row: mixing "1.212x" with
        # "6.47%" invited the reading that |V| varies LESS than |V||I|. It
        # varies more (PR #531 review).
        print(f"{f/1e9:.1f} GHz conservation (all ptp/mean): "
              f"|V| {sp(VV[:,k]):.1f}%  |I| {sp(II[:,k]):.1f}%  "
              f"|V||I| {sp(mp):.1f}% [NOT conserved — power factor varies]  |  "
              f"Re(V I*) {sp(rp):.3f}%  flux {sp(FF[:,k]):.3f}%")

    # --- Z0 from the PRODUCTION extractor, not a hand-rolled fit ----------
    # An earlier version fitted V(x) = a e^{-jbx} + g e^{+jbx} with beta PINNED
    # to the Hammerstad-Jensen eps_eff, and recorded the disagreement with a
    # free-beta fit as an unresolvable "method spread". It is not unresolvable:
    # production SCANS beta (msl_wave_decomp.py:559-561 -> _estimate_beta,
    # "residual scan around the analytic guess + refine"), using HJ only as the
    # anchor beta0. Pinning answers a different question than the one the
    # surrounding sentence asks, so this calls the production path itself
    # (PR #531 review, the seventh error of the same family).
    from rfx.probes.msl_wave_decomp import extract_msl_nprobe

    v_arr = jnp.asarray(np.stack([np.asarray(v) for v in Vcpx], axis=1))  # (nf, N)
    x_arr = jnp.asarray(xs_a)
    i1_arr = jnp.asarray(np.asarray(Icpx[0]))                             # at probe 0
    beta0 = jnp.asarray(2 * np.pi * freqs * float(np.sqrt(eps_eff)) / 2.99792458e8)
    dec = extract_msl_nprobe(v_arr, x_arr, i1_arr, beta0, z0_hj=float(z0_hj))
    z0_n = np.asarray(dec["z0"])
    beta_n = np.asarray(dec["beta"])
    print()
    for k, f in enumerate(freqs):
        ee_fit = float((np.real(beta_n[k]) * 2.99792458e8 / (2 * np.pi * f)) ** 2)
        # Re AND Im, never abs(): a magnitude cannot show a sign flip or a large
        # reactive part, and this repo carries an MSL Z0 sign-normalisation
        # history (_sparams.py:2471-2479, the dir_sign block).
        print(f"{f/1e9:.1f} GHz production extract_msl_nprobe: "
              f"Z0 = {z0_n[k].real:.2f} {z0_n[k].imag:+.2f}j ohm "
              f"(vs HJ {float(z0_hj):.2f}, {(z0_n[k].real-float(z0_hj))/float(z0_hj)*100:+.1f}%)  "
              f"|S11| = {abs(np.asarray(dec['s11'])[k]):.4f}  "
              f"eps_eff_fit = {ee_fit:.4f} (HJ {float(eps_eff):.4f})  "
              f"resid = {float(np.asarray(dec['residual'])[k]):.2e} V "
              f"({float(np.asarray(dec['residual'])[k]) / float(np.linalg.norm(np.asarray(v_arr)[k])):.2e} relative)")

    # --- verdict: pair each frequency with ITSELF -------------------------
    # The previous version maximised the ratio swing and the R variation over
    # frequency INDEPENDENTLY and printed them in one sentence, so it reported
    # a (swing, variation) pair that never occurred (PR #531 review, MAJOR 1).
    print()
    swings = np.array([ZZ[:, k].max() / ZZ[:, k].min() for k in range(len(freqs))])
    spreads = np.array([np.ptp(Rs[:, k]) / np.mean(Rs[:, k]) for k in range(len(freqs))])
    kw = int(np.argmax(swings))
    for k, f in enumerate(freqs):
        print(f"{f/1e9:.1f} GHz : |V|/|I| swings {ZZ[:,k].min():.2f} -> {ZZ[:,k].max():.2f} ohm "
              f"({swings[k]:.2f}x) while R varies {spreads[k]*100:.3f}% "
              f"(R in [{Rs[:,k].min():.5f}, {Rs[:,k].max():.5f}])")
    print()
    if swings[kw] >= 1.5 and spreads[kw] <= 0.01:
        print(f"D1 at {freqs[kw]/1e9:.1f} GHz (the largest swing, paired with ITS OWN R):")
        print("    the plane sweep is a physical traversal of R's invariance set —")
        print("    every plane carries the same guided power, so Re(V I*) and the flux")
        print(f"    are both conserved, while |V|/|I| spans {swings[kw]:.2f}x. R moves")
        print(f"    {spreads[kw]*100:.3f}%, i.e. {swings[kw]-1:.0%} of ratio range maps to")
        print(f"    {spreads[kw]*100:.3f}% of R: a sensitivity gap of ~{(swings[kw]-1)/spreads[kw]:.0f}x.")
        print("    R constrains the product; the S assembly and the reported Z0 use the")
        print("    pair separately. No oracle of this form exists.")
    else:
        print(f"D2 at {freqs[kw]/1e9:.1f} GHz: swing {swings[kw]:.2f}x, R variation "
              f"{spreads[kw]*100:.3f}% — R tracked the ratio. Since the DOF argument is")
        print("    algebra plus a code fact, no field measurement can falsify it; reaching")
        print("    here means the EXTRACTOR or the FLUX MONITOR is defective. Investigate")
        print("    those, not the argument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
