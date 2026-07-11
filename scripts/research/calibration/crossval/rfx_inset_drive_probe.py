"""Port-DRIVEN interior probe: does energy delivered through the msl feed
reach the 9.044 GHz patch mode in rfx?

Runs the ported inset sim (msl port excited) with an Ez probe INSIDE the
patch; spectra + harminv of the interior field decide:
  - interior peak at ~9.04 but no S11 dip  -> extraction-side problem
  - interior peak only at ~10.4-10.6       -> feed genuinely bypasses TM010
"""
import ast, os, sys, time
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(_DIR, "rfx_patch_inset_xband.py")).read()
tree = ast.parse(src)
tree.body = [n for n in tree.body if not (isinstance(n, ast.If)
             and getattr(getattr(n.test, 'left', None), 'id', '') == '__name__')]
ns = {"__file__": os.path.join(_DIR, "rfx_patch_inset_xband.py")}
exec(compile(tree, "m", "exec"), ns)

from rfx.harminv import harminv

inset = float(sys.argv[1]) if len(sys.argv) > 1 else 2.4
sim = ns["build_sim"](inset * 1e-3)   # msl port INCLUDED and excited
XPL, XPH = ns["X_PATCH_LO"], ns["X_PATCH_HI"]
zmid = ns["Z_SUB_LO"] + (ns["Z_SUB_HI"] - ns["Z_SUB_LO"]) / 2
# interior probe: off-centre in x and y (hits TM010 antinode region)
sim.add_probe(position=(XPL + 0.75 * (XPH - XPL),
                        ns["Y_C"] + 0.25 * ns["W"], zmid), component="ez")

print(f"port-driven interior probe, inset={inset}mm")
t0 = time.time()
res = sim.run(num_periods=200)
ts = np.asarray(res.time_series)[:, 0]
dt = float(getattr(res, "dt", 0) or sim._build_grid().dt)

# spectrum of the driven interior field
n = len(ts)
win = np.hanning(n)
F = np.fft.rfft(ts * win)
fax = np.fft.rfftfreq(n, dt)
m = (fax > 7e9) & (fax < 12e9)
mag = np.abs(F[m]); fb = fax[m]
order = np.argsort(mag)[::-1][:6]
print("top spectral peaks of interior Ez under PORT drive:")
for i in sorted(order, key=lambda k: fb[k]):
    print(f"  {fb[i]/1e9:.3f} GHz  amp={mag[i]:.3e}")

# ring-down harminv on the tail
n0 = int(n * 0.5)
modes = harminv(ts[n0:], dt, f_min=7e9, f_max=12e9)
print("harminv (tail):")
for mo in modes:
    print(f"  {mo.freq/1e9:.4f} GHz  Q={mo.Q:.1f}  amp={abs(mo.amplitude):.3e}")
print(f"({time.time()-t0:.0f}s)")
