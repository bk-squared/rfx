"""Verify the per-port matched-guide flux-reference fix makes the T-junction
flux S passive across the broad single-mode band (vs the vacuum-ref version
which gave col-sums 2.4-3.2)."""
import sys; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix_flux)

dx = 0.002; nc = 10
BAND = np.linspace(5.0e9, 7.0e9, 11)   # broad single-mode TE10 (fc=3.75, TE20=7.5)
grid = Grid(freq_max=10e9, domain=(0.12, 0.12, 0.02), dx=dx, cpml_layers=nc, cpml_axes="xy")

def with_pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes:
        m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m

# device T-junction
dev = with_pec([Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.04,0.12,0.02)),
                Box((0.08,0.08,0),(0.12,0.12,0.02))])
# per-port matched straight guides: horizontal (arms 0,1), vertical (arm 2)
mat_h = with_pec([Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.12,0.12,0.02))])
mat_v = with_pec([Box((0,0,0),(0.04,0.12,0.02)), Box((0.08,0,0),(0.12,0.12,0.02))])
ref_pp = [mat_h, mat_h, mat_v]
vac = init_materials(grid.shape)

freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=30)
px, py, pz = grid.axis_pads
xs = (px+int(round(0.04/dx)), px+int(round(0.08/dx))+1)
ys = (py+int(round(0.04/dx)), py+int(round(0.08/dx))+1)
zs = (pz, grid.nz-pz)
left = WaveguidePort(x_index=nc+5, y_slice=ys, z_slice=zs, a=0.04, b=0.02, mode=(1,0),
                     mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=grid.nx-nc-6, y_slice=ys, z_slice=zs, a=0.04, b=0.02, mode=(1,0),
                      mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
top = WaveguidePort(x_index=grid.ny-nc-6, y_slice=None, z_slice=None, a=0.04, b=0.02, mode=(1,0),
                    mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=3, probe_offset=15,
                            dft_total_steps=n_steps) for p in (left, right, top)]

print("[run] flux T-junction with per-port matched-guide reference (the fix) ...")
S = np.abs(np.asarray(extract_waveguide_s_matrix_flux(
    grid, dev, vac, cfgs, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z",
    ref_materials_per_port=ref_pp)))
cps = np.sum(S**2, axis=0)  # (3, nf)
print(" per-freq max col-sum (passivity, lossless PEC must be ~<=1):")
print("  " + " ".join(f"{BAND[k]/1e9:.1f}G:{cps[:,k].max():.3f}" for k in range(len(BAND))))
print(f" passivity MAX over band = {cps.max():.3f}  {'PASS<=1.10' if cps.max()<=1.10 else 'FAIL'}")
rec = max(np.mean(np.abs(S[i,j]-S[j,i])) for (i,j) in ((1,0),(2,0),(2,1)))
print(f" reciprocity = {rec:.3f}")
print(f" |S| band-mean:\n{np.array2string(np.mean(S,axis=2), precision=3)}")
