"""Reviewer check (D1 vs committed tests): the #1205 test
tests/unit/boundaries/test_magnetic_wall_faces_not_shorted.py::
test_wire_port_on_a_magnetic_wall_plane_reads_its_load, run as committed
(arm 'main') and with D1's odd-image magnetic wall monkeypatched in
(arm 'image'): tangential H of the x and y magnetic faces no longer zeroed;
the backward difference along x and y uses the odd image at both ends; the
port's Ampere loop reads the image at index 0 on x and y.
Closed form |S11| = 1/3 for R = Zc/2 and 2 Zc; the test's bound is 0.05.
"""
import sys
import warnings
import numpy as np
import jax.numpy as jnp

arm = sys.argv[1]
warnings.simplefilter("ignore")

if arm == "image":
    import rfx.core.yee as Y
    import rfx.boundaries.pmc as P
    import rfx.probes.probes as PR
    ORIG = Y._diff_bwd_o
    MAG = (0, 1)   # x and y faces are magnetic in the fixture

    def diff_bwd(arr, axis, periodic, order, bloch=None):
        out = ORIG(arr, axis, periodic, order, bloch)
        if axis in MAG and not periodic[axis]:
            lo = [slice(None)] * arr.ndim; lo[axis] = 0
            hi = [slice(None)] * arr.ndim; hi[axis] = -1
            h2 = [slice(None)] * arr.ndim; h2[axis] = -2
            out = out.at[tuple(lo)].set(2.0 * arr[tuple(lo)])
            out = out.at[tuple(hi)].set(-2.0 * arr[tuple(h2)])
        return out
    Y._diff_bwd_o = diff_bwd
    P.apply_pmc_faces = lambda st, faces: st
    ORIG_BWD_H = PR._bwd_h

    def bwd_h(h, idx, axis, periodic=(False, False, False)):
        if axis in MAG and int(idx[axis]) == 0 and h.shape[axis] > 1:
            return -h[idx]
        return ORIG_BWD_H(h, idx, axis, periodic)
    PR._bwd_h = bwd_h

sys.path.insert(0, "main/tests/unit/boundaries")
import test_magnetic_wall_faces_not_shorted as T  # noqa: E402

for r in (0.5, 2.0):
    gamma = abs((r - 1.0) / (r + 1.0))
    s_fwd = np.abs(np.asarray(T._line(r).forward(
        port_s11_freqs=jnp.asarray(T.FREQS_HZ), num_periods=20.0,
        skip_preflight=True).s_params).reshape(-1))
    s_run = np.abs(np.asarray(T._line(r).run(
        compute_s_params=True, s_param_freqs=T.FREQS_HZ, num_periods=20.0,
        skip_preflight=True).s_params).reshape(-1))
    ok = np.all(np.abs(s_fwd - gamma) < 0.05) and np.all(np.abs(s_run - gamma) < 0.05)
    print(f"{arm} R/Zc={r}: |S11| forward {np.round(s_fwd, 4)} run {np.round(s_run, 4)} "
          f"at {T.FREQS_HZ/1e9} GHz; closed form {gamma:.4f}; test passes: {ok}")
