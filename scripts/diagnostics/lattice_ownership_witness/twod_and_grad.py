import warnings, numpy as np, jax, jax.numpy as jnp
warnings.simplefilter("ignore")
from rfx import Simulation, Box

print("=== 2-D lane, field level ===")
for m in ("2d_tmz","2d_tez"):
    for comp, probe_on, probe_off in (("ez",(6e-3,6e-3,0.0),(1e-3,6e-3,0.0)),
                                      ("ex",(6e-3,6e-3,0.0),(1e-3,6e-3,0.0))):
        try:
            s=Simulation(freq_max=15e9, domain=(12e-3,12e-3,1e-3), dx=1e-3, boundary="pec", mode=m)
            s.add_thin_conductor(Box((3e-3,3e-3,0.0),(9e-3,9e-3,0.0)))
            s.add_source((2e-3,6e-3,0.0),comp,amplitude_kind="field")
            s.add_probe(probe_on,comp); s.add_probe(probe_off,comp)
            r=s.run(n_steps=80, skip_preflight=True, compute_s_params=False)
            ts=np.asarray(r.time_series)
            on=float(np.max(np.abs(ts[:,0]))); off=float(np.max(np.abs(ts[:,1])))
            print(f"  {m} {comp}: inside-sheet={on:.3e}  outside={off:.3e}")
        except Exception as e:
            print(f"  {m} {comp}: {type(e).__name__}: {str(e)[:80]}")

print()
print("=== differentiable path: gradient through the soft PEC rule ===")
from rfx.boundaries.pec import apply_pec_occupancy, realized_pec_edge_masks, SheetSpec
shape=(6,6,6)
class St:
    pass
import collections
FD = collections.namedtuple("FD","ex ey ez")
def loss(occ):
    st = FD(ex=jnp.ones(shape), ey=jnp.ones(shape), ez=jnp.ones(shape))
    st = st._replace if False else st
    class S(tuple): pass
    # use a real namedtuple with _replace
    out = apply_pec_occupancy(FD(jnp.ones(shape), jnp.ones(shape), jnp.ones(shape)), occ)
    return jnp.sum(out.ex**2 + out.ey**2 + out.ez**2)
occ0 = jnp.full(shape, 0.3)
g = jax.grad(loss)(occ0)
print("  loss:", float(loss(occ0)))
print("  grad finite:", bool(jnp.all(jnp.isfinite(g))), " grad nonzero cells:", int(jnp.sum(jnp.abs(g)>0)), "/", g.size)
print("  grad range:", float(jnp.min(g)), float(jnp.max(g)))
# gradient at binary occupancy (the clip boundary)
for v in (0.0, 1.0):
    gb = jax.grad(loss)(jnp.full(shape, v))
    print(f"  occ={v}: grad nonzero {int(jnp.sum(jnp.abs(gb)>0))}/{gb.size}  max|g| {float(jnp.max(jnp.abs(gb))):.4g}")

print()
print("=== soft == hard at binary, with a body on face 0 and n-1 and a periodic seam ===")
for per in ((False,False,False),(True,False,False)):
    for desc, sl in (("interior", (slice(2,4),slice(2,4),slice(2,4))),
                     ("on face 0", (slice(0,2),slice(2,4),slice(2,4))),
                     ("on face n-1",(slice(4,6),slice(2,4),slice(2,4))),
                     ("split seam", None)):
        C=np.zeros(shape,dtype=bool)
        if sl is None: C[0,2:4,2:4]=True; C[5,2:4,2:4]=True
        else: C[sl]=True
        hard = realized_pec_edge_masks(jnp.asarray(C), periodic=per)
        st = FD(jnp.ones(shape), jnp.ones(shape), jnp.ones(shape))
        soft = apply_pec_occupancy(st, jnp.asarray(C, dtype=jnp.float32), periodic=per)
        ok=True
        for c,(h,f) in enumerate(zip(hard,(soft.ex,soft.ey,soft.ez))):
            expect = 1.0-np.asarray(h,dtype=np.float32)
            if not np.array_equal(expect, np.asarray(f)): ok=False
        print(f"  periodic={per[0]} {desc:12s} soft==hard: {ok}")
