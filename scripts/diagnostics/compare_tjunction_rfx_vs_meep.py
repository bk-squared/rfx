"""Compare the rfx non-flux T-junction |S| against the external MEEP flux
reference (scripts/diagnostics/meep_tjunction_reference.py drives 0/1/2).

Both on the SAME 11-point grid over 5-7 GHz (single-mode TE10 band). Reports:
 - element-wise ||S_rfx| - |S_meep|| (max + matrix), vs the 0.05 envelope tol,
 - MEEP-internal reciprocity |S_ij|~|S_ji| and passivity sum_i|S_ij|^2<=1,
 - rfx-internal reciprocity/passivity.
Writes scripts/diagnostics/_artifacts/tjunction_rfx_vs_meep_comparison.json.
"""
import sys, os, json
sys.path.insert(0, "/root/workspace/bk-workspace/rfx-ref")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix)

ART = "/root/workspace/bk-workspace/rfx-ref/scripts/diagnostics/_artifacts"
FREQS = np.linspace(5.0e9, 7.0e9, 11)
TOL = 0.05


def rfx_oracle():
    dx = 0.002; nc = 10
    grid = Grid(freq_max=10e9, domain=(0.12, 0.12, 0.02), dx=dx, cpml_layers=nc, cpml_axes="xy")
    materials = init_materials(grid.shape)
    for box in (Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.04,0.12,0.02)),
                Box((0.08,0.08,0),(0.12,0.12,0.02))):
        materials = materials._replace(sigma=jnp.where(box.mask(grid), 1e10, materials.sigma))
    freqs = jnp.asarray(FREQS)
    n_steps = grid.num_timesteps(num_periods=30)
    padx, pady, padz = grid.axis_pads
    xs = (padx+int(round(0.04/dx)), padx+int(round(0.08/dx))+1)
    ys = (pady+int(round(0.04/dx)), pady+int(round(0.08/dx))+1)
    zs = (padz, grid.nz-padz)
    left = WaveguidePort(x_index=nc+5, y_slice=ys, z_slice=zs, a=0.04, b=0.02, mode=(1,0),
                         mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
    right = WaveguidePort(x_index=grid.nx-nc-6, y_slice=ys, z_slice=zs, a=0.04, b=0.02, mode=(1,0),
                          mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
    top = WaveguidePort(x_index=grid.ny-nc-6, y_slice=None, z_slice=None, a=0.04, b=0.02, mode=(1,0),
                        mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
    cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=3, probe_offset=15,
                                dft_total_steps=n_steps) for p in (left, right, top)]
    S = np.asarray(extract_waveguide_s_matrix(grid, materials, cfgs, n_steps,
                   boundary="cpml", cpml_axes="xy", pec_axes="z"))
    return np.abs(S)  # (3,3,nf)


def meep_matrix():
    M = np.zeros((3, 3, len(FREQS)))
    for d in range(3):
        z = np.load(os.path.join(ART, f"meep_tjunction_drive{d}.npz"))
        col = np.abs(z["col"])           # (3, nf_meep)
        fm = z["freqs_hz"]
        for j in range(3):
            M[j, d] = np.interp(FREQS, fm, col[j])
    return M


def stats(tag, A):
    cps = np.sum(A**2, axis=0)  # (3,nf) column power sums
    rec = [float(np.mean(np.abs(A[i,j]-A[j,i]))) for (i,j) in ((1,0),(2,0),(2,1))]
    print(f" {tag}: passivity max col-sum={cps.max():.3f} (band-mean {np.mean(cps,axis=1)})")
    print(f" {tag}: reciprocity |S_ij|-|S_ji| mean={[f'{r:.3f}' for r in rec]}")
    return dict(passivity_max=float(cps.max()), reciprocity=rec)


def main():
    print("[rfx] computing non-flux oracle on 11-pt grid ...")
    R = rfx_oracle()
    M = meep_matrix()
    print("\n=== |S| band-mean (rfx oracle) ===\n", np.array2string(np.mean(R,axis=2), precision=3))
    print("\n=== |S| band-mean (MEEP ref) ===\n", np.array2string(np.mean(M,axis=2), precision=3))
    d = np.abs(R - M)
    print("\n=== element-wise ||S_rfx|-|S_meep|| band-mean ===\n", np.array2string(np.mean(d,axis=2), precision=3))
    print(f"\n MAX element deviation over all freqs = {d.max():.3f}  (tol {TOL})  "
          f"{'PASS' if d.max()<=TOL else 'FAIL'}")
    print(f" band-mean MAX element deviation     = {np.mean(d,axis=2).max():.3f}")
    clean = slice(2, 9)  # drop near-cutoff band edges (keep 5.4-6.6 GHz)
    print(f" CLEAN-BAND (5.4-6.6 GHz) MAX element deviation = {d[:,:,clean].max():.3f}  "
          f"{'PASS' if d[:,:,clean].max()<=TOL else 'FAIL'}")
    print(f"   per-freq MAX dev across band: {np.array2string(d.max(axis=(0,1)), precision=3)}")
    print("\n-- internal physics --")
    sr = stats("rfx ", R); sm = stats("meep", M)
    out = dict(freqs_hz=FREQS.tolist(), tol=TOL,
               max_abs_dev=float(d.max()), bandmean_max_abs_dev=float(np.mean(d,axis=2).max()),
               rfx=sr, meep=sm,
               rfx_S_bandmean=np.mean(R,axis=2).tolist(), meep_S_bandmean=np.mean(M,axis=2).tolist())
    with open(os.path.join(ART, "tjunction_rfx_vs_meep_comparison.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n wrote {ART}/tjunction_rfx_vs_meep_comparison.json")


if __name__ == "__main__":
    main()
