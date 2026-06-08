"""Coaxial broad-E5 redesign PROTOTYPE (Step C): proper TEM-line + two/three-plane.

Refuted in B0-3: the |S11|>1 bias is NOT the extractor (its math is exact for a
clean lossless TEM field). Root cause = closed-PEC-box + 5mm stub topology.

This prototype builds a REAL coax transmission line:
  * long coax (pin/PTFE/shell co-extensive) running INTO a z-CPML feed (absorbed)
  * PEC short at the DUT end
  * TFSF TEM source near the feed, launching toward the DUT
  * >=3 equally-spaced probe planes in the clean line region

Extraction is Z0-FREE and beta-SELF-MEASURED from the modal voltage V(z)=int E_r dr:
  3 equally spaced planes => cos(beta*Delta) = (V1+V3)/(2 V2)
  decompose V(z)=A e^{+j beta z} + B e^{-j beta z}  (forward = -z toward load)
  Gamma_load = V_bwd(z_dut)/V_fwd(z_dut)
Validation targets (no tolerance fudging): short -> |Gamma|=1, angle ~180 (Gamma=-1).
"""
import sys; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.geometry.csg import Cylinder
from rfx.probes.probes import init_dft_plane_probe
from rfx.simulation import run as _run
from rfx.sources.coaxial_port import (
    build_coaxial_tem_plane_source_specs,
    coaxial_tem_reference_plane_vi_from_cartesian_plane as extract_vi,
    PEC_SIGMA, PTFE_EPS_R, SMA_PIN_RADIUS, SMA_OUTER_RADIUS,
)

A_IN, B_OUT, EPSR = SMA_PIN_RADIUS, SMA_OUTER_RADIUS, PTFE_EPS_R
C0 = 299792458.0
CX = CY = 0.010
DOMAIN = (0.020, 0.020, 0.058)
FREQS = jnp.asarray([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])


def stamp_long_coax(grid, materials, z_lo_idx, z_hi_idx):
    """Stamp pin(PEC)/PTFE/shell(PEC) co-extensive over z-index [z_lo,z_hi]."""
    dz = float(grid.dx)
    z_lo = (z_lo_idx - grid.pad_z_lo) * dz
    z_hi = (z_hi_idx - grid.pad_z_lo) * dz
    zc = 0.5 * (z_lo + z_hi); H = (z_hi - z_lo) + 2 * dz
    center = (CX, CY, zc)
    shell_th = min(dz, 0.5 * (B_OUT - A_IN))
    shell_inner = B_OUT - shell_th
    outer = Cylinder(center=center, radius=B_OUT, height=H, axis="z").mask(grid)
    s_in = Cylinder(center=center, radius=shell_inner, height=H, axis="z").mask(grid)
    pin = Cylinder(center=center, radius=A_IN, height=H, axis="z").mask(grid)
    eps = np.array(materials.eps_r); sig = np.array(materials.sigma)
    shell = outer & ~s_in
    eps = np.where(shell, 1.0, eps); sig = np.where(shell, PEC_SIGMA, sig)
    ptfe = s_in & ~pin
    eps = np.where(ptfe, PTFE_EPS_R, eps); sig = np.where(ptfe, 0.0, sig)
    eps = np.where(pin, 1.0, eps); sig = np.where(pin, PEC_SIGMA, sig)
    return materials._replace(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sig)), shell_inner


def stamp_short(grid, materials, z_idx):
    """PEC disk across pin..shell at z_idx (short between conductors)."""
    eps = np.array(materials.eps_r); sig = np.array(materials.sigma)
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            if np.hypot(x - CX, y - CY) <= B_OUT:
                sig[i, j, z_idx] = PEC_SIGMA; eps[i, j, z_idx] = 1.0
    return materials._replace(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sig))


def stamp_matched(grid, materials, z_idx, Z_target, shell_inner):
    eps = np.array(materials.eps_r); sig = np.array(materials.sigma)
    dz = float(grid.dx)
    sigma_load = float(np.log(shell_inner / A_IN)) / (2 * np.pi * dz * Z_target)
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            r = np.hypot(x - CX, y - CY)
            if A_IN <= r <= shell_inner and sig[i, j, z_idx] < 0.5 * PEC_SIGMA:
                sig[i, j, z_idx] = sigma_load; eps[i, j, z_idx] = 1.0
    return materials._replace(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sig))


def voltage_at_plane(grid, port, ex, ey):
    u = (np.arange(grid.nx) - grid.pad_x_lo) * grid.dx
    v = (np.arange(grid.ny) - grid.pad_y_lo) * grid.dx
    # only voltage is needed; pass H=E as dummy (current unused here)
    res = extract_vi(u, v, np.asarray(ex, np.complex128), np.asarray(ey, np.complex128),
                     np.asarray(ex, np.complex128), np.asarray(ey, np.complex128),
                     center_u_m=CX, center_v_m=CY, inner_radius=A_IN, outer_radius=B_OUT, eps_r=EPSR)
    return np.asarray(res.vi.voltage, np.complex128)


def run_dut(dut, n_steps=2000, field_scale=1e4, waveform=None, cpml_axes="z", domain=None,
            freq_max=20.0e9):
    global DOMAIN
    if domain is not None:
        DOMAIN = domain
    # Feed = matched resistor; conductors STOP before the top z-CPML (running PEC
    # into CPML is unstable, verified: max|E|->3.5e14->NaN). Vacuum gap + z-CPML
    # above the coax absorbs any residual; the matched feed resistor provides the
    # loss that makes the DFT converge and lets two-plane handle residual mismatch.
    sim = Simulation(domain=DOMAIN, freq_max=freq_max, boundary="cpml")
    grid = sim._build_grid()
    nz = grid.shape[2]
    z_dut = 20
    z_hi_coax = nz - grid.pad_z_lo - 2       # coax top ends ~2 cells below top PML
    z_feed = z_hi_coax - 1                    # matched feed resistor slice
    z_src = z_hi_coax - 3                     # source below the feed resistor
    # source descriptor: pin_center at z_src
    wf = waveform if waveform is not None else GaussianPulse(f0=8.0e9, bandwidth=1.2)
    sim.add_coaxial_port((CX, CY, (z_src - grid.pad_z_lo) * grid.dx), face="top",
                         pin_length=grid.dx, waveform=wf)
    port = sim._coaxial_ports[0]
    materials, _, _ = sim._build_materials(grid)
    materials, shell_inner = stamp_long_coax(grid, materials, z_dut, z_hi_coax)
    Z0_an = (np.sqrt(4e-7*np.pi/(EPSR*8.8541878128e-12))/(2*np.pi))*np.log(B_OUT/A_IN)
    materials = stamp_matched(grid, materials, z_feed, Z0_an, shell_inner)  # feed term.
    if dut == "short":
        materials = stamp_short(grid, materials, z_dut)
    elif dut == "matched":
        materials = stamp_matched(grid, materials, z_dut, Z0_an, shell_inner)
    elif dut == "open":
        pass  # pin already ends; leave annulus open (handled by not stamping)
    spec = build_coaxial_tem_plane_source_specs(grid=grid, port=port, n_steps=n_steps,
                                                field_scale=field_scale, magnetic_ratio=1.0)
    probes_z = [z_dut + 8 + 4 * k for k in range(12)]   # dense, equally spaced (Δ=4)
    probes_z = [z for z in probes_z if z < z_src - 4]    # keep clear of source near-field
    planes = []
    for z in probes_z:
        for c in ("ex", "ey"):
            planes.append(init_dft_plane_probe(axis=2, index=z, component=c, freqs=FREQS,
                          grid_shape=grid.shape, dft_total_steps=n_steps))
    res = _run(grid, materials, n_steps, boundary="cpml", cpml_axes=cpml_axes,
               sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
               dft_planes=planes, return_state=False)
    Vz = {}
    for zi, z in enumerate(probes_z):
        ex = res.dft_planes[zi*2+0].accumulator; ey = res.dft_planes[zi*2+1].accumulator
        Vz[z] = voltage_at_plane(grid, port, ex, ey)
    return grid, probes_z, Vz, z_dut, Z0_an


def fit_gamma(grid, probes_z, Vz, z_dut, fi, beta_an):
    """Least-squares complex-gamma fit V(z)=A e^{+gamma z}+B e^{-gamma z} over all
    clean planes. A-term = incident (toward load, -z), B-term = reflected.
    Returns (beta, alpha, Gamma_load, rel_resid)."""
    dz = float(grid.dx)
    z = np.array([(p - grid.pad_z_lo) * dz for p in probes_z], np.float64)
    V = np.array([Vz[p][fi] for p in probes_z], np.complex128)
    z0 = z.mean()                                   # center for conditioning
    best = None
    for bm in np.linspace(0.6, 1.7, 111):
        beta = bm * beta_an
        for aD in np.linspace(0.0, 0.30, 16):       # mild loss per plane spacing
            alpha = aD / (4 * dz)
            g = alpha + 1j * beta
            Phi = np.stack([np.exp(+g * (z - z0)), np.exp(-g * (z - z0))], axis=1)
            AB, *_ = np.linalg.lstsq(Phi, V, rcond=None)
            resid = np.linalg.norm(Phi @ AB - V) / (np.linalg.norm(V) + 1e-30)
            if best is None or resid < best[0]:
                best = (resid, beta, alpha, AB)
    resid, beta, alpha, AB = best
    A, B = AB
    zd = (z_dut - grid.pad_z_lo) * dz - z0
    g = alpha + 1j * beta
    Vinc = A * np.exp(+g * zd); Vref = B * np.exp(-g * zd)
    return beta, alpha, Vref / Vinc, resid


if __name__ == "__main__":
    f = np.asarray(FREQS)
    for dut in ("short", "matched", "open"):
        grid, pz, Vz, z_dut, Z0 = run_dut(dut)
        print(f"\n##### DUT={dut}  probes_z={pz}  z_dut={z_dut}  Z0_an={Z0:.2f}  dx={grid.dx*1e3:.3f}mm #####")
        print(" f[GHz]  beta/beta_an  alpha*dz   |Gamma|   ang(Gamma)   fit_resid")
        for fi in range(len(f)):
            beta_an = 2 * np.pi * f[fi] * np.sqrt(EPSR) / C0
            beta, alpha, G, resid = fit_gamma(grid, pz, Vz, z_dut, fi, beta_an)
            print(f" {f[fi]/1e9:5.1f}    {beta/beta_an:6.3f}     {alpha*grid.dx:6.3f}   "
                  f"{abs(G):6.3f}   {np.degrees(np.angle(G)):7.1f}    {resid:7.4f}")
