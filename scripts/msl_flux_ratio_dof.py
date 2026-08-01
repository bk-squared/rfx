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
    S = B · A^-1  over the same pairs      rfx/api/_sparams.py:2533-2540

and so does the separately reported characteristic impedance:

    z0 = (alpha - gamma) / (i1 + eps)      rfx/probes/msl_wave_decomp.py:565
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
     => the DOF argument is wrong; reopen.

RESULT (2026-08-01, this script, dx = h_sub/3 = 84.67 um, 8000 steps, CPU)

    settling witness (end/peak Ez^2, 95% tail) = -105.4 dB   PASS

      x_mm     R@3.0G    R@4.5G   |V/I|@3.0G   |V/I|@4.5G
      3.50    1.00671   1.00732        42.36        57.34
      5.41    1.00851   1.00906        36.81        45.16
      7.32    1.00905   1.00905        33.03        35.69
      9.23    1.00857   1.00854        30.98        29.93
     10.50    1.00710   1.00716        30.79        28.97
     (12 planes total; the full ladder is monotone at both frequencies)

    3.0 GHz : |V|/|I| swings 30.78 -> 42.36 ohm (1.38x), R varies 0.232%
    4.5 GHz : |V|/|I| swings 28.97 -> 57.34 ohm (1.98x), R varies 0.194%

  D1 CONFIRMED. A real, physically caused 1.98x excursion along the ratio
  direction — the standing-wave envelope of an imperfectly matched line — moves
  R by 0.194%. Nothing algebraic is doing the work here: |V|, |I| and the flux
  all change substantially from plane to plane, and R is flat because the guided
  power is what it tracks. R would have to be ~10x more sensitive than this
  before it could resolve the ratio at all.

  The |V|/|I| ladder above is the standing-wave envelope, NOT Z0 — on an
  unmatched line |V|/|I| oscillates about Z0 rather than equalling it, so none
  of these numbers is an impedance measurement and none should be quoted as
  extractor error.

  NOT MEASURED HERE, attributed: the PR #531 reviewer independently reports that
  on this same settled fixture the extracted Z0 = (alpha-gamma)/I comes out
  ~44.1 ohm, constant to ~0.2% across 12 planes, i.e. -7.9% vs Hammerstad-Jensen
  and inside the >5% Yee-staircase envelope preflight warns about for a
  3-substrate-cell mesh (their /tmp/pr531_z0_8k.log). I have not re-derived it.
  If it holds, the configuration is HEALTHY and R = 1.008 on it is correct and
  unremarkable — which is the point: R would read the same either way.

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
    settle = 10 * np.log10(max(float(np.mean(tail ** 2)), 1e-300)
                           / max(float(np.max(ts ** 2)), 1e-300))
    print(f"\nsettling witness (end/peak Ez^2, 95% tail): {settle:.1f} dB "
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
    Rs, ZZ = [], []
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
        Rs.append(R)
        ZZ.append(Z)
        print(f"{x*1e3:6.2f} " + "".join(f"{r:10.5f}" for r in R)
              + "".join(f"{z:13.2f}" for z in Z))

    Rs, ZZ = np.array(Rs), np.array(ZZ)
    print()
    for k, f in enumerate(freqs):
        r_spread = float(np.ptp(Rs[:, k]) / np.mean(Rs[:, k]))
        z_swing = float(ZZ[:, k].max() / ZZ[:, k].min())
        print(f"{f/1e9:.1f} GHz : |V|/|I| swings {ZZ[:,k].min():.2f} -> "
              f"{ZZ[:,k].max():.2f} ohm ({z_swing:.2f}x) while R varies "
              f"{r_spread*100:.3f}% (R in [{Rs[:,k].min():.5f}, {Rs[:,k].max():.5f}])")

    worst_z = float(max(ZZ[:, k].max() / ZZ[:, k].min() for k in range(len(freqs))))
    worst_r = float(max(np.ptp(Rs[:, k]) / np.mean(Rs[:, k]) for k in range(len(freqs))))
    print()
    if worst_z >= 1.5 and worst_r <= 0.01:
        print(f"D1: a real {worst_z:.2f}x excursion in the RATIO direction moves R by")
        print(f"    only {worst_r*100:.3f}%. R constrains the product, not the pair.")
        print("    No oracle of this form exists; a sufficient one must also")
        print("    constrain the ratio — e.g. (alpha-gamma)/I vs a closed form,")
        print("    on a settled and matched line.")
    else:
        print(f"D2: ratio swing {worst_z:.2f}x, R variation {worst_r*100:.3f}% — "
              "the pre-declared separation did not appear; re-examine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
