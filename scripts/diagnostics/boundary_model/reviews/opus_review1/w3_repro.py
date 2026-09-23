"""Reviewer check: where does the W3 PMC-plate-line closed form come from?

Arm A: forward() as on main (pec_axes='xyz' on an absorber-free spec).
Arm B: same, but rfx.simulation.apply_pec restricted to the PEC axis z
       (i.e. no electric wall laid over the PMC faces x, y) -- what B2 would do
       while keeping today's half-cell-inside magnetic wall.
Arm B-probe: arm B plus two Ez probes at x = 2 mm: one on the port's node line
       (y = 0, the y_lo face node plane) and one on y = 1 mm (the y_hi face node
       plane), to see whether the two node lines are coupled at all.
"""
import sys
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/root/workspace/bk-workspace/.boundary-model/B0")
import rfx.simulation as S
from rfx.boundaries.pec import apply_pec as _orig_apply_pec
import reference_known_load_line as W3

ETA0 = W3.ETA0
DX = W3.DX


def s11(kind, r, patch):
    if patch:
        S.apply_pec = lambda st, axes="xyz": _orig_apply_pec(
            st, axes="".join(a for a in axes if a == "z"))
    else:
        S.apply_pec = _orig_apply_pec
    try:
        sim = W3._build(kind, r)
        res = sim.forward(port_s11_freqs=jnp.asarray(W3.FREQS_HZ),
                          num_periods=20.0, skip_preflight=True)
        return np.abs(np.asarray(res.s_params).reshape(-1))
    finally:
        S.apply_pec = _orig_apply_pec


def probes(patch):
    if patch:
        S.apply_pec = lambda st, axes="xyz": _orig_apply_pec(
            st, axes="".join(a for a in axes if a == "z"))
    try:
        sim = W3._build("lumped", 1.0)
        sim.add_probe(position=(2 * DX, 0.0, 0.0), component="ez")
        sim.add_probe(position=(2 * DX, DX, 0.0), component="ez")
        res = sim.forward(num_periods=20.0, skip_preflight=True)
        ts = np.asarray(res.time_series)
        return ts
    finally:
        S.apply_pec = _orig_apply_pec


if __name__ == "__main__":
    print("freqs GHz", W3.FREQS_HZ / 1e9)
    for patch in (False, True):
        for kind in ("lumped", "wire"):
            for r in (0.5, 1.0, 2.0):
                g = abs((r - 1) / (r + 1))
                print(f"patch={patch} {kind:6s} R/Zc={r}: closed {g:.4f} "
                      f"|S11| {np.round(s11(kind, r, patch), 4)}")
    for patch in (False, True):
        ts = probes(patch)
        print(f"patch={patch} probe time_series shape {ts.shape}")
        print("  max|Ez| on y=0 node line (x=2mm):", float(np.max(np.abs(ts[:, 0]))))
        print("  max|Ez| on y=1mm node line (x=2mm):", float(np.max(np.abs(ts[:, 1]))))
