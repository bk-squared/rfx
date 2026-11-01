"""For every solve lane a user can reach: is a DECLARED PEC SHEET realized,
or is the lane refused by name?  Anything else is silence."""
import warnings, traceback, numpy as np
warnings.simplefilter("ignore")
from rfx import Simulation, Box

PLANE = 5e-3
ON  = (5e-3, 5e-3, PLANE)     # an Ex edge inside the sheet
OFF = (5e-3, 5e-3, 4e-3)      # the same component one plane below

def mk(**kw):
    sim = Simulation(freq_max=15e9, domain=(10e-3,10e-3,8e-3), dx=1e-3,
                     boundary="pec", **kw)
    sim.add(Box((2e-3,2e-3,2e-3),(4e-3,4e-3,3e-3)), material="pec")
    sim.add_thin_conductor(Box((3e-3,2e-3,PLANE),(8e-3,8e-3,PLANE)))
    sim.add_source((5e-3,5e-3,3e-3),"ez",amplitude_kind="field")
    sim.add_probe(ON,"ex"); sim.add_probe(OFF,"ex")
    return sim

def verdict(name, fn):
    try:
        on, off = fn()
    except NotImplementedError as e:
        print(f"  {name:34s} REFUSED (NotImplementedError): {str(e)[:90]}"); return
    except Exception as e:
        print(f"  {name:34s} RAISED {type(e).__name__}: {str(e)[:90]}"); return
    if off <= 1e-9:
        print(f"  {name:34s} INCONCLUSIVE control={off:.3e} (model did not run)"); return
    tag = "realized" if on == 0.0 else "*** SHEET DROPPED (SILENT) ***"
    print(f"  {name:34s} on_sheet={on:.3e}  control={off:.3e}   {tag}")

def peaks(r):
    ts=np.asarray(r.time_series)
    return float(np.max(np.abs(ts[:,0]))), float(np.max(np.abs(ts[:,1])))

N=60
verdict("run() uniform",         lambda: peaks(mk().run(n_steps=N, skip_preflight=True, compute_s_params=False)))
verdict("forward()",             lambda: peaks(mk().forward(n_steps=N, skip_preflight=True)))
verdict("run() subpixel=True",   lambda: peaks(mk().run(n_steps=N, skip_preflight=True, compute_s_params=False, subpixel_smoothing=True)))
verdict("run() kottke_pec",      lambda: peaks(mk().run(n_steps=N, skip_preflight=True, compute_s_params=False, subpixel_smoothing="kottke_pec")))
verdict("run() conformal_pec",   lambda: peaks(mk().run(n_steps=N, skip_preflight=True, compute_s_params=False, conformal_pec=True)))
verdict("solver='adi'",          lambda: peaks(mk(solver="adi").run(n_steps=N, skip_preflight=True, compute_s_params=False)))
verdict("stencil_order=4",       lambda: peaks(mk(stencil_order=4).run(n_steps=N, skip_preflight=True, compute_s_params=False)))
verdict("precision=float64",     lambda: peaks(mk(precision="float64").run(n_steps=N, skip_preflight=True, compute_s_params=False)))

def nu():
    prof = np.full(8, 1e-3)
    s = Simulation(freq_max=15e9, domain=(10e-3,10e-3,8e-3), dx=1e-3, boundary="pec",
                   dz_profile=prof)
    s.add(Box((2e-3,2e-3,2e-3),(4e-3,4e-3,3e-3)), material="pec")
    s.add_thin_conductor(Box((3e-3,2e-3,PLANE),(8e-3,8e-3,PLANE)))
    s.add_source((5e-3,5e-3,3e-3),"ez",amplitude_kind="field")
    s.add_probe(ON,"ex"); s.add_probe(OFF,"ex")
    return peaks(s.run(n_steps=N, skip_preflight=True, compute_s_params=False))
verdict("non-uniform (dz_profile)", nu)

def vm():
    from rfx.vmap_sweep import vmap_material_sweep
    return peaks(vmap_material_sweep(mk(), "eps_r", [1.0,1.0], n_steps=N))
verdict("vmap_material_sweep", vm)

def twod():
    s = Simulation(freq_max=15e9, domain=(10e-3,10e-3,1e-3), dx=1e-3,
                   boundary="pec", mode="2d")
    s.add_thin_conductor(Box((3e-3,2e-3,0.0),(8e-3,8e-3,0.0)))
    s.add_source((5e-3,5e-3,0.0),"ez",amplitude_kind="field")
    s.add_probe((5e-3,5e-3,0.0),"ez"); s.add_probe((1e-3,1e-3,0.0),"ez")
    r=s.run(n_steps=N, skip_preflight=True, compute_s_params=False)
    return peaks(r)
verdict("2-D lane (mode='2d')", twod)
