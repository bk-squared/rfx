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

  WHAT IS CONSERVED, measured:

    3.0 GHz : |V| 1.212x  |I| 1.136x  |  Re(V I*) 0.158%  flux 0.223%
    4.5 GHz : |V| 1.447x  |I| 1.368x  |  Re(V I*) 0.153%  flux 0.203%
                                         [|V||I| 6.5% / 9.0% — NOT conserved:
                                          the standing wave moves the V-I phase,
                                          so the power factor varies while the
                                          real power does not]

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

  TWO-WAVE FIT on the same planes (computed here, not cited — a committed record
  must not point at a /tmp log, which is the #520 failure):

    3.0 GHz : Z0 = (alpha-gamma)/I = 45.21 ohm, plane spread 0.47%,
              |Gamma| = 0.189, fit residual 3.3e-03, vs HJ 47.89 ohm (-5.6%)
    4.5 GHz : Z0 = 45.16 ohm, spread 0.94%, |Gamma| = 0.215,
              residual 5.1e-03, vs HJ (-5.7%)

  Z0 is constant along the line to ~0.5-0.9% and sits 5.6-5.7% below the closed
  form — inside the ">5% expected" Yee-staircase envelope preflight warns about
  for a 3-substrate-cell mesh. The configuration is HEALTHY; R = 1.008 on it is
  correct and unremarkable. That is the point: R would read the same either way.

  METHOD NOTE, because the number is fit-dependent: this fit PINS beta to the
  Hammerstad-Jensen eps_eff (2.869). The PR #531 reviewer, FITTING beta on the
  same plane set, gets eps_eff ~ 3.035 and Z0 = 43.86/43.82 ohm (-8.5% vs HJ).
  Both land inside the staircase envelope, but they differ by ~3%, which bounds
  what this fit can resolve. Neither is quoted as the true Z0.

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
        print(f"{f/1e9:.1f} GHz conservation: |V| {VV[:,k].max()/VV[:,k].min():.3f}x  "
              f"|I| {II[:,k].max()/II[:,k].min():.3f}x  |  "
              f"Re(V I*) ptp/mean {sp(rp):.3f}%  flux ptp/mean {sp(FF[:,k]):.3f}%  "
              f"[|V||I| {sp(mp):.2f}% — NOT conserved, power factor varies]")

    # --- Z0 from a two-wave fit, computed HERE rather than cited ----------
    # V(x) = a e^{-j b x} + g e^{+j b x};  Z0 = (alpha_x - gamma_x)/I(x).
    # Self-contained so the record does not point at a /tmp log (#520).
    print()
    for k, f in enumerate(freqs):
        beta = 2 * np.pi * f * float(np.sqrt(eps_eff)) / 2.99792458e8
        ez_ph = np.array([np.exp(-1j * beta * xx) for xx in xs_a])
        M = np.stack([ez_ph, np.conj(ez_ph)], axis=1)
        Vc = np.array([np.asarray(v)[k] for v in Vcpx])
        Ic = np.array([np.asarray(i)[k] for i in Icpx])
        (a_w, g_w), *_ = np.linalg.lstsq(M, Vc, rcond=None)
        alpha_x, gamma_x = a_w * ez_ph, g_w * np.conj(ez_ph)
        z0_x = (alpha_x - gamma_x) / Ic
        resid = float(np.linalg.norm(M @ np.array([a_w, g_w]) - Vc) / np.linalg.norm(Vc))
        print(f"{f/1e9:.1f} GHz two-wave fit: Z0 = (alpha-gamma)/I = "
              f"{np.abs(z0_x).mean():.2f} ohm (spread {np.ptp(np.abs(z0_x))/np.abs(z0_x).mean()*100:.2f}%), "
              f"|Gamma| = {abs(g_w/a_w):.4f}, fit resid = {resid:.2e}, "
              f"vs HJ {float(z0_hj):.2f} ohm ({(np.abs(z0_x).mean()-float(z0_hj))/float(z0_hj)*100:+.1f}%)")

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
