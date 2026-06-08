"""Coaxial broad-E5 Phase 1: quantitative termination gates vs ANALYTIC Gamma.
Existing tests only smoke/qualitatively check matched/open. This builds the real
quantitative gate: drive compute_coaxial_s_matrix with each canonical termination
and compare |S11| to the exact analytic reflection magnitude across a band.
  short (pin->floor):        Gamma=-1  -> |S11|=1
  open  (pin retracted):     Gamma=+1  -> |S11|=1
  matched (annular Z0 load):  Gamma=0   -> |S11|=0
No external solver needed (analytic truth). SMA coax, 20mm PEC cube, freq_max=20GHz
(dx~0.75mm, ~1 PTFE-annulus cell) per the calibrated PEC-short fixture.
"""
import sys; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.api import Simulation

DOM = (0.020, 0.020, 0.020); POS = (0.010, 0.010, 0.015)
FREQS = jnp.asarray(np.linspace(2.0e9, 10.0e9, 9))
NS, NF = 400, 9

def base():
    return Simulation(domain=DOM, freq_max=20.0e9, boundary="pec")

def run(tag, build):
    sim = base(); build(sim)
    res = sim.compute_coaxial_s_matrix(n_steps=NS, freqs=FREQS)
    return np.abs(np.asarray(res.s_params)[0, 0, :])

# short: pin extends to floor (z=0 from gap z=15mm)
s11_short = run("short", lambda s: s.add_coaxial_port(POS, face="top", pin_length=15.0e-3))
# open: floating pin + retract (PTFE-filled circular WG below cutoff -> Gamma~+1)
def _open(s): s.add_coaxial_port(POS, face="top", pin_length=5.0e-3); s.add_coaxial_open_termination(pin_retract_cells=2)
s11_open = run("open", _open)
# matched: floating pin + annular Z0 load -> Gamma~0
def _matched(s): s.add_coaxial_port(POS, face="top", pin_length=5.0e-3); s.add_coaxial_matched_load()
s11_matched = run("matched", _matched)

f = np.asarray(FREQS)
print("GHz   |S11|short(→1)  |S11|open(→1)  |S11|matched(→0)")
for k in range(NF):
    print(f"{f[k]/1e9:4.1f}    {s11_short[k]:.3f}          {s11_open[k]:.3f}         {s11_matched[k]:.3f}")
print()
print(f"short  : mean={s11_short.mean():.3f}  min={s11_short.min():.3f}  (analytic |Gamma|=1; gate |S11|>=0.9)")
print(f"open   : mean={s11_open.mean():.3f}  min={s11_open.min():.3f}  (analytic |Gamma|=1)")
print(f"matched: mean={s11_matched.mean():.3f}  max={s11_matched.max():.3f}  (analytic |Gamma|=0; lossy-line approx, improves w/ freq)")
print(f"contrast: matched < open everywhere? {bool(np.all(s11_matched < s11_open))}")
