"""Phase 2: centered slab in over-moded WR-90, multi-mode vs per-mode Airy.

Centered full-cross-section dielectric slab preserves modal structure:
TE10->TE10 and TE20->TE20 each follow analytic Airy with their own beta;
TE10<->TE20 coupling = 0 by symmetry. Non-degenerate (device != empty ref).
Compares normalize=False / True against per-mode Airy.
"""
import sys; sys.path.insert(0, "/root/workspace/bk-workspace/rfx-ref")
import numpy as np
import jax.numpy as jnp
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

C0 = 299_792_458.0; ETA0 = 376.730313668
A = 22.86e-3; B = 10.16e-3
FC10 = C0/(2*A); FC20 = C0/A
FREQS = np.linspace(13.5e9, 14.5e9, 6)
F0 = float(FREQS.mean()); BW = 0.3
DX = 0.5e-3; CPML = 24
DOMAIN_X = 200e-3
PL = 40e-3; PR = 160e-3; RL = 50e-3; RR = 150e-3
SLAB_EPS = 2.0; SL = 8e-3

def airy_mode(f, eps_r, L, fc_v):
    # per-mode Airy: modal cutoff fc_v in vacuum, fc_d in dielectric
    fc_d = fc_v / np.sqrt(eps_r)
    zv = ETA0/np.sqrt(1-(fc_v/f)**2)
    zd = (ETA0/np.sqrt(eps_r))/np.sqrt(1-(fc_d/f)**2)
    rho = (zd-zv)/(zd+zv); tau = 2*zd/(zd+zv); taub = 2*zv/(zd+zv)
    bd = (2*np.pi*f*np.sqrt(eps_r)/C0)*np.sqrt(1-(fc_d/f)**2)
    d = bd*L; e2 = np.exp(-2j*d)
    s11 = rho*(1-e2)/(1-rho*rho*e2)
    s21 = tau*taub*np.exp(-1j*d)/(1-rho*rho*e2)
    return s11, s21

def run(norm):
    sim = Simulation(
        freq_max=float(FREQS[-1])*1.1, domain=(DOMAIN_X, A, B),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec")),
        cpml_layers=CPML, dx=DX)
    c = 0.5*(PL+PR)
    sim.add_material("slab", eps_r=SLAB_EPS, sigma=0.0)
    sim.add(Box((c-0.5*SL,0,0),(c+0.5*SL,A,B)), material="slab")
    pf = jnp.asarray(FREQS)
    sim.add_waveguide_port(PL, direction="+x", mode=(1,0), mode_type="TE",
        freqs=pf, f0=F0, bandwidth=BW, waveform="modulated_gaussian",
        reference_plane=RL, name="left", n_modes=2)
    sim.add_waveguide_port(PR, direction="-x", mode=(1,0), mode_type="TE",
        freqs=pf, f0=F0, bandwidth=BW, waveform="modulated_gaussian",
        reference_plane=RR, name="right", n_modes=2)
    r = sim.compute_waveguide_s_matrix(num_periods=80, normalize=norm)
    return np.asarray(r.s_params), list(r.port_names)

# per-mode Airy references
beta10 = (2*np.pi*FREQS/C0)*np.sqrt(1-(FC10/FREQS)**2)
beta20 = (2*np.pi*FREQS/C0)*np.sqrt(1-(FC20/FREQS)**2)
c = 0.5*(PL+PR); d_left = c - 0.5*SL - RL
s11_10, s21_10 = airy_mode(FREQS, SLAB_EPS, SL, FC10)
s11_20, s21_20 = airy_mode(FREQS, SLAB_EPS, SL, FC20)
ref = {
    "S11_TE10": np.abs(s11_10), "S21_TE10": np.abs(s21_10),
    "S11_TE20": np.abs(s11_20), "S21_TE20": np.abs(s21_20),
}
print(f"per-mode Airy |S| (mid f={FREQS[3]/1e9:.2f}GHz):")
for k,v in ref.items(): print(f"  {k}: {v[3]:.4f}")

for norm in [False, True, "flux"]:
    s, names = run(norm)
    idx = {n:i for i,n in enumerate(names)}
    L10 = idx["left_mode0_TE10"]; L20 = idx["left_mode1_TE20"]
    R10 = idx["right_mode0_TE10"]; R20 = idx["right_mode1_TE20"]
    mid = 3
    # device S vs Airy: S11_TE10 = S[L10,L10], S21_TE10 = S[R10,L10]
    d_s11_10 = np.abs(np.abs(s[L10,L10,:]) - np.abs(s11_10))
    d_s21_10 = np.abs(np.abs(s[R10,L10,:]) - np.abs(s21_10))
    d_s11_20 = np.abs(np.abs(s[L20,L20,:]) - np.abs(s11_20))
    d_s21_20 = np.abs(np.abs(s[R20,L20,:]) - np.abs(s21_20))
    cross = max(np.abs(s[L20,L10,:]).max(), np.abs(s[R20,L10,:]).max(),
                np.abs(s[L10,L20,:]).max(), np.abs(s[R10,L20,:]).max())
    print(f"\n=== normalize={norm} ===")
    print(f"  TE10: |S11| diff {d_s11_10.max():.4f}  |S21| diff {d_s21_10.max():.4f}")
    print(f"  TE20: |S11| diff {d_s11_20.max():.4f}  |S21| diff {d_s21_20.max():.4f}")
    print(f"  cross-mode coupling max (should be ~0): {cross:.4f}")
    print(f"  rfx |S11|_TE10={np.abs(s[L10,L10,mid]):.3f} (Airy {ref['S11_TE10'][mid]:.3f}), "
          f"|S11|_TE20={np.abs(s[L20,L20,mid]):.3f} (Airy {ref['S11_TE20'][mid]:.3f})")
