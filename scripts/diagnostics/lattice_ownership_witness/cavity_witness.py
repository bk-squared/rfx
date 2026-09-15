"""Independent physics witness for the #931 volume/sheet realization rule.

Oracle: the analytic TE101 resonance of a rectangular PEC box,
f = (c/2) sqrt((1/a)^2 + (1/d)^2), computed from the DRAWN geometry.
Nothing in rfx's own realization code is consulted to form the prediction.
"""
import warnings, numpy as np
warnings.simplefilter("ignore")
from rfx import Simulation, Box

C0 = 299792458.0
DX = 2e-3
AX, AY, AZ = 40e-3, 20e-3, 30e-3


def f_te101(a, d):
    return 0.5 * C0 * np.sqrt((1.0 / a) ** 2 + (1.0 / d) ** 2)


def peak_freq(build, n_steps=6000, probe=(13e-3, 10e-3, 17e-3),
              src=(9e-3, 10e-3, 11e-3)):
    sim = Simulation(freq_max=12e9, domain=(AX, AY, AZ), dx=DX, boundary="pec")
    build(sim)
    sim.add_source(src, "ey", amplitude_kind="field")
    sim.add_probe(probe, "ey")
    r = sim.run(n_steps=n_steps, skip_preflight=True, compute_s_params=False)
    ts = np.asarray(r.time_series)[:, 0].astype(np.float64)
    dt = float(sim._build_grid().dt)
    n = 1 << 16
    sp = np.abs(np.fft.rfft(ts * np.hanning(ts.size), n=n))
    fr = np.fft.rfftfreq(n, dt)
    band = (fr > 3e9) & (fr < 11e9)
    k = np.argmax(sp[band])
    return float(fr[band][k]), float(sp[band].max()), ts


cases = {}
cases["A empty box  d=30mm"] = (lambda s: None, AZ)
cases["B volume slab z 0->4mm"] = (
    lambda s: s.add(Box((0, 0, 0), (AX, AY, 4e-3)), material="pec"), AZ - 4e-3)
cases["C volume slab z 0->6mm"] = (
    lambda s: s.add(Box((0, 0, 0), (AX, AY, 6e-3)), material="pec"), AZ - 6e-3)
cases["D sheet at z=4mm"] = (
    lambda s: s.add_thin_conductor(Box((0, 0, 4e-3), (AX, AY, 4e-3))), AZ - 4e-3)
cases["E volume slab z 26->30mm"] = (
    lambda s: s.add(Box((0, 0, 26e-3), (AX, AY, AZ)), material="pec"), 26e-3)

print(f"{'case':30s} {'f_meas GHz':>11s} {'f_analytic':>11s} {'err %':>8s}  d_drawn")
for name, (build, d) in cases.items():
    f, amp, ts = peak_freq(build)
    fa = f_te101(AX, d)
    print(f"{name:30s} {f/1e9:11.4f} {fa/1e9:11.4f} {100*(f-fa)/fa:8.3f}  {d*1e3:.1f} mm")
