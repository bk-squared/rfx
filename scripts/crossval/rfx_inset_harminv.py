"""Port-independent resonance of the rfx INSET-fed patch (Harminv ring-down).

Decides the rfx dip-offset root cause: if the inset geometry's ring-down
resonance is ~9.3 GHz (like the edge-fed issue80 patch / openEMS / Palace),
the offset lives in the msl feed/extraction; if it is ~10.1 GHz, the rfx
field solution itself shifts for this geometry.
"""
import ast, os, sys, time
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(_DIR, "rfx_patch_inset_xband.py")).read()
tree = ast.parse(src)
tree.body = [n for n in tree.body if not (isinstance(n, ast.If)
             and getattr(getattr(n.test, 'left', None), 'id', '') == '__name__')]
ns = {"__file__": "rfx_patch_inset_xband.py"}
exec(compile(tree, "m", "exec"), ns)

from rfx import Box, Simulation
from rfx.sources import GaussianPulse
from rfx.harminv import harminv

DX = ns["DX"]; DOM = (ns["DOM_X"], ns["DOM_Y"], ns["DOM_Z"])
inset = float(sys.argv[1]) if len(sys.argv) > 1 else 2.4

# Build the SAME geometry but WITHOUT the msl port: bare source + probe.
sim = Simulation(freq_max=15e9, domain=DOM, dx=DX, cpml_layers=8, boundary="cpml")
sim.add_material("ro4003c", eps_r=ns["EPS_R"], sigma=0.0)
sim.add(Box((0, 0, ns["Z_GND"]), (DOM[0], DOM[1], ns["Z_GND"] + DX)), material="pec")
sim.add(Box((0, 0, ns["Z_SUB_LO"]), (DOM[0], DOM[1], ns["Z_SUB_HI"])), material="ro4003c")
yf_lo, yf_hi = ns["Y_C"] - ns["W_MSL"]/2, ns["Y_C"] + ns["W_MSL"]/2
yp_lo, yp_hi = ns["Y_C"] - ns["W"]/2, ns["Y_C"] + ns["W"]/2
yn_lo, yn_hi = yf_lo - ns["NOTCH_GAP"], yf_hi + ns["NOTCH_GAP"]
XPL, XPH = ns["X_PATCH_LO"], ns["X_PATCH_HI"]
x_conn = XPL + inset*1e-3
ZL, ZH = ns["Z_MET_LO"], ns["Z_MET_HI"]
sim.add(Box((0, yf_lo, ZL), (x_conn, yf_hi, ZH)), material="pec")       # feed strip
sim.add(Box((x_conn, yp_lo, ZL), (XPH, yp_hi, ZH)), material="pec")     # body
sim.add(Box((XPL, yp_lo, ZL), (x_conn, yn_lo, ZH)), material="pec")     # flank lo
sim.add(Box((XPL, yn_hi, ZL), (x_conn, yp_hi, ZH)), material="pec")     # flank hi

# source under the patch interior (off-centre to hit TM010), probe elsewhere
src_x = XPL + 0.75*(XPH-XPL); src_y = ns["Y_C"] + 0.25*ns["W"]
prb_x = XPL + 0.35*(XPH-XPL); prb_y = ns["Y_C"] - 0.30*ns["W"]
zmid = ns["Z_SUB_LO"] + (ns["Z_SUB_HI"]-ns["Z_SUB_LO"])/2
sim.add_source(position=(src_x, src_y, zmid), component="ez",
               waveform=GaussianPulse(f0=9.5e9, bandwidth=0.8))
sim.add_probe(position=(prb_x, prb_y, zmid), component="ez")

print(f"inset={inset}mm  harminv ring-down...")
t0 = time.time()
res = sim.run(num_periods=120)
ts = np.asarray(res.time_series)[:, 0]
dt = float(getattr(res, "dt", 0) or sim._build_grid().dt)
# drop the driven interval, analyze the ring-down tail
n0 = int(len(ts)*0.35)
modes = harminv(ts[n0:], dt, f_min=7e9, f_max=12e9)
print("modes (f GHz, Q, amp):")
try:
    for mo in modes:
        print(f"  {mo.freq/1e9:.4f}  Q={mo.Q:.1f}  amp={abs(mo.amplitude):.3e}")
except Exception:
    print(modes)
print(f"({time.time()-t0:.0f}s)")
