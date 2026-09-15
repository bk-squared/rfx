"""No-FDTD check: does PR #1077's fixture + mutation reproduce the reviewed arm exactly?"""
import sys
import numpy as np
sys.path.insert(0, "/root/801r_tree_pr1077")
sys.path.insert(0, "/root/801r_tree_pr1077/tests/oracle")
from test_lossless_open_domain_ringdown_does_not_grow import (  # noqa: E402
    _build, _legacy_volume_edge_masks)
from rfx.boundaries import pec  # noqa: E402
from rfx.geometry.rasterize_grid import coords_from_uniform_grid  # noqa: E402

sim = _build()
grid = sim._build_grid()
eps = np.asarray(sim._assemble_materials(grid)[0].eps_r, float)
cm = np.asarray(sim.conductor_mask(grid), bool)
print("grid", cm.shape, "conductor cells", int(cm.sum()),
      "cell k-planes", np.flatnonzero(cm.any(axis=(0, 1))).tolist())

def counts(tag):
    mx, my, mz = pec.realized_pec_edge_masks(cm, periodic=(False, False, False))
    mx, my, mz = np.asarray(mx), np.asarray(my), np.asarray(mz)
    tot = int(mx.sum()) + int(my.sum()) + int(mz.sum())
    print("%-10s Mx %6d  My %6d  Mz %6d  total %6d" % (tag, mx.sum(), my.sum(), mz.sum(), tot))
    return np.stack([mx, my, mz])

shipped = counts("shipped")
orig = pec._volume_edge_masks
pec._volume_edge_masks = _legacy_volume_edge_masks
mutated = counts("mutated")
pec._volume_edge_masks = orig

ref = np.load("/root/801r_out/801r_mat_pre931.npz")     # my reviewed pre-#931 set
refm = np.stack([np.asarray(ref[k], bool) for k in ("mx", "my", "mz")])
print("mutated vs reviewed pre-#931 set: XOR =", int((mutated ^ refm).sum()))
print("eps vs reviewed:", "max|diff| %.3e" % np.abs(eps - np.asarray(ref["eps"])).max())

c = coords_from_uniform_grid(grid); z = np.asarray(c.z, float)
mx, my = shipped[0], shipped[1]
ic, jc = 133, 93
walls = np.flatnonzero(mx[ic, jc, :] | my[ic, jc, :])
print("shipped walls_um", [round(float(z[k] * 1e6), 2) for k in walls])
