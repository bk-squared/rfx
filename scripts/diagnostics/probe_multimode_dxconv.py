"""FDTD cross-mode dx-convergence: confirm the floor is a convergent Yee
stagger limit (should halve with dx, matching the profile cross-overlap).

Empty (eps_r=1) over-moded narrow-b guide, flux multimode, drive both
modes both ports. Report cross-mode max |S| at dx in {0.5,0.25,0.125}mm.
If it ~halves (0.04->0.02->0.01), the cross-mode floor is the first-order
convergent E/H half-cell stagger (not a fixed defect).
"""
import sys; sys.path.insert(0, "/root/workspace/bk-workspace/rfx-ref")
import numpy as np
import jax.numpy as jnp
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

C0 = 299_792_458.0
A = 22.86e-3; B = 4.0e-3
FREQS = np.linspace(15.5e9, 18.5e9, 5)
F0 = float(FREQS.mean()); BW = 0.3
DOMAIN_X = 200e-3
PL = 40e-3; PR = 160e-3; RL = 50e-3; RR = 150e-3

def run(DX):
    sim = Simulation(
        freq_max=float(FREQS[-1])*1.1, domain=(DOMAIN_X, A, B),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec")),
        cpml_layers=24, dx=DX)
    # eps_r=1 vacuum box = empty (device-free), exercises same path
    c = 0.5*(PL+PR)
    sim.add_material("vac", eps_r=1.0, sigma=0.0)
    sim.add(Box((c-1.5e-3,0,0),(c+1.5e-3,A,B)), material="vac")
    pf = jnp.asarray(FREQS)
    sim.add_waveguide_port(PL, direction="+x", mode=(1,0), mode_type="TE",
        freqs=pf, f0=F0, bandwidth=BW, waveform="modulated_gaussian",
        reference_plane=RL, name="left", n_modes=2)
    sim.add_waveguide_port(PR, direction="-x", mode=(1,0), mode_type="TE",
        freqs=pf, f0=F0, bandwidth=BW, waveform="modulated_gaussian",
        reference_plane=RR, name="right", n_modes=2)
    r = sim.compute_waveguide_s_matrix(num_periods=80, normalize="flux")
    s = np.asarray(r.s_params); names=list(r.port_names)
    idx={n:i for i,n in enumerate(names)}
    L10=idx["left_mode0_TE10"]; L20=idx["left_mode1_TE20"]
    R10=idx["right_mode0_TE10"]; R20=idx["right_mode1_TE20"]
    cross=max(np.abs(s[L20,L10,:]).max(), np.abs(s[R20,L10,:]).max(),
              np.abs(s[L10,L20,:]).max(), np.abs(s[R10,L20,:]).max())
    return cross

for DX in (0.5e-3, 0.25e-3, 0.125e-3):
    cross = run(DX)
    print(f"dx={DX*1e6:.0f}um  cross-mode max |S| = {cross:.4f}", flush=True)
