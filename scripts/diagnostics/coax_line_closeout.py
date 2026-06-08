"""Coax line redesign — HIGH-FREQ CLOSEOUT.

Reuses the validated geometry/source from coax_line_twoplane_proto (long coax +
matched feed, conductors NOT into PML). Replaces the crude beta-grid-scan with a
closed-form matrix-pencil estimate of the complex propagation constant gamma over
ALL equally-spaced planes:
    V(z-D) + V(z+D) = 2 cosh(gamma D) V(z)
    p = LS estimate of 2 cosh(gamma D);  gamma = arccosh(p/2)/D
Then forward/backward amplitudes by lstsq, Gamma_load at the DUT.

Diagnostics dumped to localize the >=8 GHz degradation:
  * |V(z)| per freq            (SNR / standing-wave pattern)
  * recurrence residual         (single-mode 2-wave cleanliness)
  * gamma (beta/beta_an, alpha) (dispersion / loss)
  * |Gamma|, ang(Gamma), fit_resid
Usage: python coax_line_closeout.py <dut> <n_steps>   (dut in short/open/matched)
"""
import sys; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np
import scripts.diagnostics.coax_line_twoplane_proto as P

EPSR = P.EPSR; C0 = P.C0


def matrix_pencil_gamma(z_phys, V):
    """Closed-form complex gamma from equally-spaced samples via the
    2cosh recurrence (LS over all interior planes). Returns (gamma, rec_resid)."""
    D = z_phys[1] - z_phys[0]
    num = 0.0 + 0j; den = 0.0
    rec_res_num = 0.0; rec_res_den = 0.0
    for k in range(1, len(V) - 1):
        num += (V[k-1] + V[k+1]) * np.conj(V[k])
        den += (np.abs(V[k])**2)
    p = num / den                      # ~ 2 cosh(gamma D)
    gammaD = np.arccosh(p / 2.0 + 0j)  # complex
    gamma = gammaD / D
    # enforce beta = Im>0 (forward = -z = e^{+j beta z}); keep alpha = Re
    if np.imag(gamma) < 0:
        gamma = np.conj(gamma)         # flip branch (beta -> +)
    # recurrence residual with the LS p
    for k in range(1, len(V) - 1):
        rec_res_num += np.abs((V[k-1] + V[k+1]) - p * V[k])**2
        rec_res_den += np.abs(p * V[k])**2
    rec_resid = np.sqrt(rec_res_num / (rec_res_den + 1e-30))
    return gamma, rec_resid


def extract(grid, probes_z, Vz, z_dut, fi, beta_an):
    dz = float(grid.dx)
    z = np.array([(p - grid.pad_z_lo) * dz for p in probes_z], np.float64)
    V = np.array([Vz[p][fi] for p in probes_z], np.complex128)
    z0 = z.mean()
    gamma, rec_resid = matrix_pencil_gamma(z - z0, V)
    Phi = np.stack([np.exp(+gamma * (z - z0)), np.exp(-gamma * (z - z0))], axis=1)
    AB, *_ = np.linalg.lstsq(Phi, V, rcond=None)
    A, B = AB
    fit_resid = np.linalg.norm(Phi @ AB - V) / (np.linalg.norm(V) + 1e-30)
    zd = (z_dut - grid.pad_z_lo) * dz - z0
    Vinc = A * np.exp(+gamma * zd); Vref = B * np.exp(-gamma * zd)
    return gamma, Vref / Vinc, rec_resid, fit_resid, np.abs(V)


if __name__ == "__main__":
    dut = sys.argv[1] if len(sys.argv) > 1 else "short"
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    f = np.asarray(P.FREQS)
    grid, pz, Vz, z_dut, Z0 = P.run_dut(dut, n_steps=ns)
    print(f"##### DUT={dut} n_steps={ns} probes_z={pz} z_dut={z_dut} dx={grid.dx*1e3:.3f}mm #####")
    print(" f[GHz]  beta/beta_an  alpha*dz  rec_resid  fit_resid   |Gamma|  ang(Gamma)")
    for fi in range(len(f)):
        beta_an = 2 * np.pi * f[fi] * np.sqrt(EPSR) / C0
        g, G, rr, fr, Vmag = extract(grid, pz, Vz, z_dut, fi, beta_an)
        print(f" {f[fi]/1e9:5.1f}    {np.imag(g)/beta_an:6.3f}    {np.real(g)*grid.dx:6.3f}   "
              f"{rr:7.4f}   {fr:7.4f}    {abs(G):6.3f}   {np.degrees(np.angle(G)):7.1f}")
    print("\n |V(z)| per freq (SNR / standing-wave pattern):")
    print(" z_idx " + "  ".join(f"{x/1e9:4.0f}GHz" for x in f))
    for p in pz:
        print(f" {p:4d}  " + "  ".join(f"{abs(Vz[p][fi]):.2e}" for fi in range(len(f))))
