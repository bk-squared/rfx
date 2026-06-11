"""Issue #48 — text/visual diagnostic for the three NU+PEC hypotheses.

Prints (and optionally renders) the actual pec_mask / eps_r / source
placement on both a uniform and a NU patch-antenna sim. Lets us see
where the thin PEC sheets and feed point LAND after rasterisation,
not just where we asked them to go.

  H1 — does pec_mask on NU rasterise ground/patch at the intended z?
  H2 — does pos_to_nu_index place the source on the expected z-cell?
  H3 — does the cell-size jump across the metal (dz_sub vs ambient dx)
        create suspicious geometry (e.g. patch spanning multiple
        z-cells because a graded-cell edge shifts)?

Run locally: python scripts/issue48_pec_mask_diagnostic.py
"""

from __future__ import annotations

import math
import os
import numpy as np

from rfx import Simulation, Box
from rfx.auto_config import smooth_grading
from rfx.runners.nonuniform import assemble_materials_nu, pos_to_nu_index
from rfx.sources.sources import GaussianPulse


F_DESIGN = 2.4e9


def _common_geom():
    eps_r_fr4 = 4.3
    h_sub = 1.5e-3
    W, L = 38.0e-3, 29.5e-3
    gx, gy = 60.0e-3, 55.0e-3
    air_above, air_below = 25.0e-3, 12.0e-3
    probe_inset = 8.0e-3
    dom_x = gx + 20e-3
    dom_y = gy + 20e-3
    return dict(
        eps_r_fr4=eps_r_fr4, h_sub=h_sub, W=W, L=L, gx=gx, gy=gy,
        air_above=air_above, air_below=air_below, probe_inset=probe_inset,
        dom_x=dom_x, dom_y=dom_y,
        gx_lo=(dom_x - gx) / 2, gy_lo=(dom_y - gy) / 2,
        px_lo=dom_x / 2 - L / 2, py_lo=dom_y / 2 - W / 2,
        feed_x=(dom_x / 2 - L / 2) + probe_inset, feed_y=dom_y / 2,
    )


def _build_nu(G):
    dx = 1e-3
    n_cpml = 8
    n_sub = 6; dz_sub = G["h_sub"] / n_sub
    n_below = int(math.ceil(G["air_below"] / dx))
    n_above = int(math.ceil(G["air_above"] / dx))
    dz = np.asarray(smooth_grading(np.concatenate([
        np.full(n_below, dx), np.full(n_sub, dz_sub), np.full(n_above, dx)
    ])), dtype=np.float64)
    sim = Simulation(freq_max=4e9, domain=(G["dom_x"], G["dom_y"], 0),
                     dx=dx, dz_profile=dz, boundary="cpml", cpml_layers=n_cpml)
    sim.add_material("fr4", eps_r=G["eps_r_fr4"])
    z_gnd_lo = G["air_below"] - dz_sub
    z_sub_hi = G["air_below"] + G["h_sub"]
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dz_sub
    src_z = G["air_below"] + dz_sub * 2.5
    sim.add(Box((G["gx_lo"], G["gy_lo"], z_gnd_lo),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], G["air_below"])),
            material="pec")
    sim.add(Box((G["gx_lo"], G["gy_lo"], G["air_below"]),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], z_sub_hi)),
            material="fr4")
    sim.add(Box((G["px_lo"], G["py_lo"], z_patch_lo),
                (G["px_lo"] + G["L"], G["py_lo"] + G["W"], z_patch_hi)),
            material="pec")
    sim.add_source(position=(G["feed_x"], G["feed_y"], src_z), component="ez",
                   waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2))
    return sim, (z_gnd_lo, G["air_below"], z_sub_hi, z_patch_hi, src_z)


def _build_uniform(G, dx):
    n_cpml = 8
    dom_z = G["air_above"] + G["air_below"] + G["h_sub"]
    sim = Simulation(freq_max=4e9, domain=(G["dom_x"], G["dom_y"], dom_z),
                     dx=dx, boundary="cpml", cpml_layers=n_cpml)
    sim.add_material("fr4", eps_r=G["eps_r_fr4"])
    z_gnd_lo = G["air_below"] - dx
    z_sub_hi = G["air_below"] + G["h_sub"]
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dx
    src_z = G["air_below"] + dx * 0.5
    sim.add(Box((G["gx_lo"], G["gy_lo"], z_gnd_lo),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], G["air_below"])),
            material="pec")
    sim.add(Box((G["gx_lo"], G["gy_lo"], G["air_below"]),
                (G["gx_lo"] + G["gx"], G["gy_lo"] + G["gy"], z_sub_hi)),
            material="fr4")
    sim.add(Box((G["px_lo"], G["py_lo"], z_patch_lo),
                (G["px_lo"] + G["L"], G["py_lo"] + G["W"], z_patch_hi)),
            material="pec")
    sim.add_source(position=(G["feed_x"], G["feed_y"], src_z), component="ez",
                   waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2))
    return sim, (z_gnd_lo, G["air_below"], z_sub_hi, z_patch_hi, src_z)


def _report(label, sim, expected):
    print("\n" + "=" * 70)
    print(f"# {label}")
    print("=" * 70)
    z_gnd_lo, z_gnd_hi, z_sub_hi, z_patch_hi, src_z = expected
    is_nu = sim._dz_profile is not None
    if is_nu:
        grid = sim._build_nonuniform_grid()
        dz = np.asarray(grid.dz)
        z_edges = np.concatenate([[0.0], np.cumsum(dz)])
        z_edges = z_edges - z_edges[grid.cpml_layers]
        z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
        materials, _, _, pec_mask = assemble_materials_nu(sim, grid)
        feed_idx = pos_to_nu_index(grid, sim._ports[0].position)
    else:
        grid = sim._build_grid()
        dz = np.full(grid.nz, grid.dx)
        z_edges = (np.arange(grid.nz + 1) - grid.cpml_layers) * grid.dx
        z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
        # Materials assembled during run; get via _assemble_materials
        mat, _, _, pec_mask, _, _ = sim._assemble_materials(grid)
        materials = mat
        # Uniform pos_to_index
        feed_idx = grid.position_to_index(sim._ports[0].position)

    # Column (ix_feed, iy_feed) for patch region
    i_f, j_f = int(feed_idx[0]), int(feed_idx[1])
    # Find a PEC-rich column: centre of patch
    i_ctr, j_ctr = grid.nx // 2, grid.ny // 2
    print(f"grid: nx={grid.nx}, ny={grid.ny}, nz={grid.nz}, cpml={grid.cpml_layers}")
    print(f"dz range: min={dz.min()*1e3:.3f} mm, max={dz.max()*1e3:.3f} mm  "
          f"(max ratio = {(dz[1:]/dz[:-1]).max():.2f})")

    print(f"\n## H1 / H3 — pec_mask + eps along centre column (i={i_ctr}, j={j_ctr})")
    print(f"{'k':>3} {'z (mm)':>8} {'dz (mm)':>8} {'eps_r':>6} {'PEC?':>5} {'annotation':<28}")
    def _annotate(z):
        tol = 1e-4
        tags = []
        if abs(z - z_gnd_lo) < tol: tags.append("gnd_lo")
        if abs(z - z_gnd_hi) < tol: tags.append("gnd_hi/sub_lo")
        if abs(z - z_sub_hi) < tol: tags.append("sub_hi/patch_lo")
        if abs(z - z_patch_hi) < tol: tags.append("patch_hi")
        if abs(z - src_z) < tol: tags.append("src_z")
        return ",".join(tags)
    # Focus on 10 cells around the substrate region
    k_sub = int(np.argmin(np.abs(z_centres - (z_gnd_hi + z_sub_hi) / 2)))
    kmin = max(0, k_sub - 6); kmax = min(grid.nz, k_sub + 10)
    for k in range(kmin, kmax):
        z_c = z_centres[k] * 1e3
        pec_flag = bool(pec_mask[i_ctr, j_ctr, k])
        eps = float(np.asarray(materials.eps_r)[i_ctr, j_ctr, k])
        print(f"{k:>3} {z_c:>8.3f} {dz[k]*1e3:>8.3f} {eps:>6.2f} "
              f"{'yes' if pec_flag else '.':>5} {_annotate(z_centres[k]):<28}")

    print(f"\n## H2 — source placement")
    print(f"requested src_z  = {src_z*1e3:.4f} mm")
    if is_nu:
        k_src = int(feed_idx[2])
        z_cell_lo = z_edges[k_src] * 1e3
        z_cell_hi = z_edges[k_src + 1] * 1e3
        z_cell_c = z_centres[k_src] * 1e3
        print(f"pos_to_nu_index → (i={feed_idx[0]}, j={feed_idx[1]}, k={k_src})")
        print(f"  that cell spans z ∈ [{z_cell_lo:.3f}, {z_cell_hi:.3f}] mm "
              f"(centre {z_cell_c:.3f})")
    else:
        k_src = feed_idx[2]
        z_cell_c = z_centres[k_src] * 1e3
        print(f"position_to_index → k={k_src}, z={z_cell_c:.3f} mm")

    # PEC cell count along the centre column
    pec_cells_col = int(np.sum(pec_mask[i_ctr, j_ctr, :]))
    print(f"\n## H1 summary — PEC cells along centre column: {pec_cells_col}  "
          f"(expected = 2: 1 ground + 1 patch)")


def main():
    G = _common_geom()
    sim_nu, exp_nu = _build_nu(G)
    sim_un, exp_un = _build_uniform(G, 0.5e-3)
    _report("UNIFORM dx=0.5mm (broadside OK)", sim_un, exp_un)
    _report("NU dz_sub=0.25mm (grazing, BROKEN)", sim_nu, exp_nu)


if __name__ == "__main__":
    main()
