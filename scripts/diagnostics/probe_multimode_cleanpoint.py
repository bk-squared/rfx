"""Clean-point multimode flux discriminator.

narrow-b guide (a=22.86mm, b=4mm) widens the 2-mode window to
13.1-19.67 GHz (TE20..TE30). Operating at 15.5-18.5 GHz puts TE20 at
1.18-1.41x cutoff (comfortable, not near-cutoff) and L=3mm keeps both
modes' |S11| moderate (~0.37 / ~0.55, no Fabry-Perot null).

If flux per-mode |S11| diff vs Airy is ~0.01 here (single-mode quality),
the earlier 0.08 was purely operating-point (null + near-cutoff FDTD
limits, not multimode-specific). If still ~0.08, it is a multimode
defect in extract_multimode_s_matrix_flux to fix.
"""
import sys; sys.path.insert(0, "/root/workspace/bk-workspace/rfx-ref")
import numpy as np
import jax.numpy as jnp
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

C0 = 299_792_458.0; ETA0 = 376.730313668
A = 22.86e-3; B = 4.0e-3
FC10 = C0/(2*A); FC20 = C0/A
FREQS = np.linspace(15.5e9, 18.5e9, 7)
F0 = float(FREQS.mean()); BW = 0.3
DX = 0.25e-3; CPML = 24
DOMAIN_X = 200e-3
PL = 40e-3; PR = 160e-3; RL = 50e-3; RR = 150e-3
SLAB_EPS = 2.0; SL = 3e-3

def airy_mode(f, eps_r, L, fc_v):
    fc_d = fc_v/np.sqrt(eps_r)
    zv = ETA0/np.sqrt(1-(fc_v/f)**2)
    zd = (ETA0/np.sqrt(eps_r))/np.sqrt(1-(fc_d/f)**2)
    rho=(zd-zv)/(zd+zv); tau=2*zd/(zd+zv); taub=2*zv/(zd+zv)
    bd=(2*np.pi*f*np.sqrt(eps_r)/C0)*np.sqrt(1-(fc_d/f)**2)
    d=bd*L; e2=np.exp(-2j*d)
    return rho*(1-e2)/(1-rho*rho*e2), tau*taub*np.exp(-1j*d)/(1-rho*rho*e2)

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
    r = sim.compute_waveguide_s_matrix(num_periods=100, normalize=norm)
    return np.asarray(r.s_params), list(r.port_names)

s11_10,s21_10 = airy_mode(FREQS,SLAB_EPS,SL,FC10)
s11_20,s21_20 = airy_mode(FREQS,SLAB_EPS,SL,FC20)
print(f"narrow-b a={A*1e3}mm b={B*1e3}mm, band {FREQS[0]/1e9}-{FREQS[-1]/1e9} GHz")
print(f"TE20 cutoff ratio {FREQS[0]/FC20:.2f}-{FREQS[-1]/FC20:.2f}, L={SL*1e3}mm")
print(f"Airy |S11|: TE10 [{np.abs(s11_10).min():.3f},{np.abs(s11_10).max():.3f}] "
      f"TE20 [{np.abs(s11_20).min():.3f},{np.abs(s11_20).max():.3f}]")

for norm in ["flux", False]:
    s,names = run(norm)
    idx={n:i for i,n in enumerate(names)}
    L10=idx["left_mode0_TE10"]; L20=idx["left_mode1_TE20"]
    R10=idx["right_mode0_TE10"]; R20=idx["right_mode1_TE20"]
    d11_10=np.abs(np.abs(s[L10,L10,:])-np.abs(s11_10)).max()
    d21_10=np.abs(np.abs(s[R10,L10,:])-np.abs(s21_10)).max()
    d11_20=np.abs(np.abs(s[L20,L20,:])-np.abs(s11_20)).max()
    d21_20=np.abs(np.abs(s[R20,L20,:])-np.abs(s21_20)).max()
    cross=max(np.abs(s[L20,L10,:]).max(),np.abs(s[R20,L10,:]).max(),
              np.abs(s[L10,L20,:]).max(),np.abs(s[R10,L20,:]).max())
    print(f"\n=== normalize={norm} ===")
    print(f"  TE10: |S11| diff {d11_10:.4f}  |S21| diff {d21_10:.4f}")
    print(f"  TE20: |S11| diff {d11_20:.4f}  |S21| diff {d21_20:.4f}")
    print(f"  cross-mode max: {cross:.4f}")
