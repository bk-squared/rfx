"""Coaxial V/I localization — Layer 1 (FDTD), probe V/I(z) along the axis.

Layer 0 proved the extractor MATH is exact for a clean lossless TEM field
(|S11|=1, ang(V/I)=+-90 on any grid). So a real |S11|>1 must mean the FIELD
reaching the extractor is not a clean lossless 1-D standing wave.

Two structural facts found by reading the code:
  * compute_coaxial_s_matrix hardcodes boundary="pec"  (closed lossless box)
  * setup_coaxial_port stamps the coax only over height=pin_length
    -> a ~5mm coax STUB opening into a closed PEC box, not a line.

Test: run the production closed-PEC "clean short" once, but place DFT plane
probes at a SWEEP of z-indices. Extract V/I at each plane and print ang(V/I).
On a lossless transmission line ang(V/I) must be +-90 at EVERY plane. If it
wanders (and depends on n_steps), the field is a closed-box standing field,
i.e. the topology — not the extractor — is the defect.
"""
import sys; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.probes.probes import init_dft_plane_probe
from rfx.simulation import run as _run
from rfx.sources.coaxial_port import (
    build_coaxial_tem_plane_source_specs, extract_coaxial_plane_vi_from_dft,
    setup_coaxial_port, add_coaxial_pec_end_cap, _coaxial_port_geometry,
)

DOM = (0.020, 0.020, 0.020); POS = (0.010, 0.010, 0.015)
FREQS = jnp.asarray([3.0e9, 5.0e9, 7.0e9])
Z_TEM = (376.730313668/np.sqrt(2.1)/(2*np.pi))*np.log(2.055/0.635)


def build():
    sim = Simulation(domain=DOM, freq_max=20.0e9, boundary="cpml")  # boundary IGNORED by runner
    sim.add_coaxial_port(POS, face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8))
    sim.add_coaxial_pec_end_cap()
    return sim


def run_zsweep(n_steps, z_indices):
    sim = build()
    port = sim._coaxial_ports[0]
    grid = sim._build_grid()
    materials, _, _ = sim._build_materials(grid)
    materials = setup_coaxial_port(grid, port, materials)
    for cap_idx, off in sim._coaxial_pec_end_caps:
        materials = add_coaxial_pec_end_cap(grid, sim._coaxial_ports[cap_idx], materials, axial_offset_cells=off)
    spec = build_coaxial_tem_plane_source_specs(grid=grid, port=port, n_steps=n_steps,
                                                field_scale=1.0e4, magnetic_ratio=1.0)
    planes = []
    for z in z_indices:
        for comp in ("ex", "ey", "hx", "hy"):
            planes.append(init_dft_plane_probe(axis=2, index=int(z), component=comp,
                          freqs=FREQS, grid_shape=grid.shape, dft_total_steps=n_steps))
    result = _run(grid, materials, int(n_steps), boundary="pec",
                  sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
                  dft_planes=planes, return_state=False)
    out = {}
    for zi, z in enumerate(z_indices):
        grp = result.dft_planes[zi*4:zi*4+4]
        cm = {p.component: np.asarray(p.accumulator, dtype=np.complex128) for p in grp}
        vi = extract_coaxial_plane_vi_from_dft(grid=grid, port=port, plane_axial_index=int(z),
              ex_dft=cm["ex"], ey_dft=cm["ey"], hx_dft=cm["hx"], hy_dft=cm["hy"])
        out[z] = (np.asarray(vi.vi.voltage), np.asarray(vi.vi.current))
    return grid, out


# pin_center index & structure extent
sim = build(); g = sim._build_grid()
_,_,_,pc,tip,gap = _coaxial_port_geometry(g, sim._coaxial_ports[0])
zc = int(g.position_to_index(pc)[2]); zt = int(g.position_to_index(tip)[2]); zg = int(gap[2])
print(f"dx={g.dx*1e3:.4f}mm  Z_TEM={Z_TEM:.3f}  pin_center z_idx={zc} pin_tip z_idx={zt} gap z_idx={zg}  pad_z_lo={g.pad_z_lo}")
print(f"coax body z range (pin_length=5mm) ~ indices [{zt},{zg}]  (structure is only this tall)\n")

z_idx = list(range(zt-3, zg+6))
for ns in (600, 1600):
    grid, out = run_zsweep(ns, z_idx)
    print(f"===== n_steps={ns}  ang(V/I) [deg] per z-plane (lossless line MUST be +-90 everywhere) =====")
    print("z_idx  " + "  ".join(f"{f/1e9:.0f}GHz:ang(|V/I|/Z0)" for f in np.asarray(FREQS)))
    for z in z_idx:
        V, I = out[z]
        cells = []
        for k in range(len(FREQS)):
            zin = V[k]/I[k]
            cells.append(f"{np.degrees(np.angle(zin)):6.1f}({abs(zin)/Z_TEM:4.2f})")
        marker = " <-pin_center" if z == zc else (" <-gap/ref" if z == zg else "")
        print(f" {z:3d}   " + "  ".join(cells) + marker)
    print()
