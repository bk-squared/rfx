"""#525 decisive check: is R = Re(V·conj(I))/flux BLIND to the extractor's Z0?

The workflow's three refutation lenses all reached the same structural claim:
R is a transverse-profile consistency check whose numerator and denominator are
built from the SAME plane's accumulators, so it is invariant under the
power-preserving rescaling (V, I) -> (aV, I/a) while Z0 = V/I moves by a^2.
If true, R cannot certify the quantity the MSL extractor exists to produce, and
#525 must be closed with that finding rather than with a ratio.

That is algebra, but the lenses also made an EMPIRICAL claim I have not verified:
that on this fixture the extracted Z0 sits 24-32% from Hammerstad-Jensen while R
sits flat at ~1.01. This script tests both on ONE run so the pairing is not in
doubt, using the PRODUCTION trace-node anchoring rather than the round(h/dx)
proxy the design script used (that proxy IS the #511 defect; it is only safe here
because dx = h_sub/3 divides exactly).

PRE-DECLARED
  B1 R stays within 0.5% under the degenerate rescale while Z0 moves by a^2
     => R is structurally blind to the impedance direction. #525's ratio cannot
        be an extractor oracle; close it with that, and point any future oracle
        at Z0-vs-closed-form instead.
  B2 R moves with a
     => the invariance argument is wrong and the ratio does carry impedance
        information; re-open the design.
  Separately reported (not pre-declared, observational): |V|/|I| against the
  Hammerstad-Jensen closed form, and R itself.

RESULT (2026-08-01, this script, dx = h_sub/3 = 84.67 um, 4000 steps, CPU):

    max |dR| under (V, I) -> (1.3 V, I/1.3)  =  0.000e+00 %
    Z0 change under the same rescale         =  1.6900x  (+69.0%)

  B1 CONFIRMED, and not merely "within tolerance" — the change is EXACTLY zero.
  R is algebraically independent of the rescale: Re(aV * conj(I/a)) = Re(V I*).

    plane   f(GHz)      R      |V|/|I|   vs HJ 47.895 ohm
     5 mm    3.00   1.00881    36.16       -24.5%
     5 mm    4.50   1.01005    31.19       -34.9%
     7 mm    3.00   1.00899    33.77       -29.5%
     9 mm    3.00   1.00817    32.08       -33.0%
     9 mm    4.50   1.00935    30.73       -35.8%

  R sits flat at 1.008-1.010 across every plane and frequency while |V|/|I| is
  24-36% below the closed form. R is a "good" number on a configuration it
  cannot speak for.

CAVEAT, stated because the number invites the wrong reading: |V|/|I| at a plane
is NOT Z0 unless the line is matched. On a line carrying any standing wave it
oscillates about Z0, so the 24-36% figure is NOT an extractor-error measurement
and must not be quoted as one. Its variation ALONG x is the informative part:
36.16 -> 32.08 ohm at 3.0 GHz over 4 mm (-11.3%) where a matched line would be
constant. That is an independent witness of the standing wave which the #525
workflow's refutation lenses found by a different route (local beta from
unwrapped arg V(x) swinging 4.4x and implying eps_eff > eps_r, impossible for a
bound mode).

WHAT THIS SCRIPT IS FOR: it is the artifact behind closing #525. It is a
demonstrator, not a gate — it asserts nothing and is not wired into CI.
"""

import sys
import warnings

sys.path.insert(0, "/root/workspace/byungkwan-workspace/research/rfx")

import numpy as np  # noqa: E402

EPS_R, H_SUB, W = 3.66, 254e-6, 600e-6
DX = H_SUB / 3.0
L, PM, FMAX = 10e-3, 2e-3, 5e9


def main():
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
    assert "/root/workspace/byungkwan-workspace/research/rfx" in rfx.__file__

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

    freqs = np.linspace(3.0e9, 4.5e9, 4)
    fr = jnp.asarray(freqs)
    xs = [PM + 3e-3, PM + 5e-3, PM + 7e-3]
    for p, x in enumerate(xs):
        for c in ("ez", "hy", "hz"):
            sim.add_dft_plane_probe(axis="x", coordinate=x, component=c,
                                    freqs=fr, name=f"{c}{p}")
        sim.add_flux_monitor(axis="x", coordinate=x, freqs=fr, name=f"full{p}")

    print("\n--- PREFLIGHT (verbatim; part of the result) ---")
    sim.preflight()
    print("--- end preflight ---\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(n_steps=4000)

    grid = sim._build_grid()
    dt = float(grid.dt)
    mp = MSLPort(feed_x=PM, y_lo=yc - W / 2, y_hi=yc + W / 2, z_lo=0.0, z_hi=H_SUB,
                 direction="+x", impedance=50.0, excitation=None)
    cells = _msl_yz_cells(grid, mp)
    js = sorted({c[1] for c in cells})
    ks = sorted({c[2] for c in cells})
    jlo, jhi, klo = js[0], js[-1], ks[0]
    jc = (jlo + jhi) // 2
    # PRODUCTION anchoring: the rasterized trace node, not round(h_sub/dx).
    pm_mask = np.asarray(sim._assemble_materials(grid)[3])
    kp = np.where(pm_mask[cells[0][0], jc, ks[-1]:])[0]
    ktr = int(ks[-1] + int(kp.min()))
    print(f"trace node (rasterized) = {ktr}   round(h_sub/dx) proxy = "
          f"{int(round(H_SUB / DX))}   {'AGREE' if ktr == int(round(H_SUB/DX)) else 'DIFFER'}")

    hs = np.exp(1j * 2 * np.pi * freqs * dt * 0.5).astype(np.complex64)
    ny, nz = np.asarray(res.dft_planes["ez0"].accumulator).shape[1:]
    dza, dya = np.full(nz, DX), np.full(ny, DX)

    z0_hj, eps_eff = hammerstad_jensen_z0_eps_eff(W, H_SUB, EPS_R)
    print(f"Hammerstad-Jensen: Z0 = {float(z0_hj):.3f} ohm, eps_eff = {float(eps_eff):.5f}\n")

    A = 1.3  # the degenerate rescale: V -> A*V, I -> I/A  (power preserved)
    print(f"{'x_mm':>6} {'f_GHz':>6} {'R':>9} {'R_scaled':>9} {'dR%':>8} "
          f"{'Z0_ext':>8} {'Z0_scaled':>10} {'Z0 vs HJ':>10}")
    dR_max = 0.0
    for p, x in enumerate(xs):
        ez = jnp.asarray(res.dft_planes[f"ez{p}"].accumulator)
        hy = jnp.asarray(res.dft_planes[f"hy{p}"].accumulator) * hs[:, None, None]
        hz = jnp.asarray(res.dft_planes[f"hz{p}"].accumulator) * hs[:, None, None]
        V = np.asarray(msl_modal_voltage(ez, j_centre=jc, k_lo=klo, k_hi=ktr,
                                         dz_arr=dza))
        I = np.asarray(msl_loop_current(hy, hz, j_lo=jlo, j_hi=jhi,
                                        k_trace_lo=ktr, k_trace_hi=ktr,
                                        dy_arr=dya, dz_arr=dza, direction="+x"))
        F = np.asarray(flux_spectrum(res.flux_monitors[f"full{p}"]))
        R = np.real(V * np.conj(I)) / F
        # the degenerate transform, applied to the SAME measured phasors
        Rs = np.real((A * V) * np.conj(I / A)) / F
        z0 = np.abs(V) / np.abs(I)
        z0s = np.abs(A * V) / np.abs(I / A)
        for k, f in enumerate(freqs):
            d = abs(Rs[k] - R[k]) / abs(R[k]) * 100
            dR_max = max(dR_max, d)
            print(f"{x*1e3:6.1f} {f/1e9:6.2f} {R[k]:9.5f} {Rs[k]:9.5f} {d:8.2e} "
                  f"{z0[k]:8.2f} {z0s[k]:10.2f} "
                  f"{(z0[k]-float(z0_hj))/float(z0_hj)*100:+9.1f}%")

    print(f"\nmax |dR| under the (V,I) -> ({A}V, I/{A}) rescale : {dR_max:.3e} %")
    print(f"Z0 change under the same rescale                  : {A**2:.4f}x "
          f"({(A**2-1)*100:+.1f}%)")
    print()
    if dR_max < 0.5:
        print("B1: R is STRUCTURALLY BLIND to the impedance direction. A rescale that")
        print("    moves Z0 by 69% leaves R unchanged to machine precision. R cannot")
        print("    certify the MSL extractor's Z0; it can only catch ONE-SIDED defects")
        print("    (which is exactly what #511 was). Close #525 on this finding.")
    else:
        print("B2: R moved with the rescale — the invariance argument is WRONG.")


if __name__ == "__main__":
    main()
