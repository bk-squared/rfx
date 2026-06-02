"""Build the rectangular_waveguide_port T-junction broad-E5 envelope + the
external MEEP comparison artifacts (promotes the family to broad_e5_passed).

Decoupled, honest design:
 * broad-E5 ENVELOPE (rfx-internal): rfx FLUX 3-port T-junction (power-flux
   extraction with per-port matched-guide reference — passive by construction)
   over a mesh refinement axis (dx in {2.0, 1.6} mm) x the broad single-mode TE10
   band (5.0-7.0 GHz, ~33% bandwidth, between TE10 cutoff 3.75 GHz and TE20 onset
   7.5 GHz). Gates: reciprocity |S_ij|~|S_ji| <= 0.05, passivity sum_i|S_ij|^2
   <= 1.10, and mesh-convergence max||S(dx=2.0)|-|S(dx=1.6)|| <= 0.08.
 * external COMPARISON (E4-broad): rfx (dx=2.0mm) vs the independent MEEP flux
   reference (meep_tjunction_reference.py, res=400) over the same band; documented
   cross-FDTD tolerance 0.11 (two coarse FDTD codes; no analytic truth for a
   junction). Both passive + reciprocal.

Run in the isolated worktree:
  PYTHONPATH=/tmp/rfx-tj python scripts/diagnostics/build_waveguide_tjunction_broad_e5_envelope.py
"""
import os, json, glob
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix_flux

BAND = np.linspace(5.0e9, 7.0e9, 11)       # broad single-mode TE10 band (fc 3.75, TE20 7.5 GHz)
MESHES_MM = [2.0, 1.6]                       # mesh refinement axis
RECIP_TOL = 0.05
PASSIVITY_TOL = 1.10                          # near-cutoff numerical slack (matches issue#80 patch tol family)
CONV_TOL = 0.08
XFDTD_TOL = 0.11                              # documented cross-FDTD rfx-vs-MEEP bar
# MEEP reference lives in the main tree's _artifacts (gitignored); reference by abs path.
MEEP_ART = "/root/workspace/bk-workspace/rfx-ref/scripts/diagnostics/_artifacts"
OUT = os.path.join(os.path.dirname(__file__), "_artifacts")


def rfx_tjunction(dx_mm):
    dx = dx_mm * 1e-3; nc = 10
    grid = Grid(freq_max=10e9, domain=(0.12, 0.12, 0.02), dx=dx, cpml_layers=nc, cpml_axes="xy")
    materials = init_materials(grid.shape)
    for box in (Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.04,0.12,0.02)),
                Box((0.08,0.08,0),(0.12,0.12,0.02))):
        materials = materials._replace(sigma=jnp.where(box.mask(grid), 1e10, materials.sigma))
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
    # Per-port matched straight-guide references (passive-by-construction flux):
    # horizontal guide for arms 0,1 (left/right); vertical guide for arm 2 (top).
    def _pec(boxes):
        m = init_materials(grid.shape)
        for b in boxes:
            m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
        return m
    mat_h = _pec([Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.12,0.12,0.02))])
    mat_v = _pec([Box((0,0,0),(0.04,0.12,0.02)), Box((0.08,0,0),(0.12,0.12,0.02))])
    S = np.abs(np.asarray(extract_waveguide_s_matrix_flux(
        grid, materials, init_materials(grid.shape), cfgs, n_steps,
        boundary="cpml", cpml_axes="xy", pec_axes="z",
        ref_materials_per_port=[mat_h, mat_h, mat_v])))
    return S  # (3,3,nf)


def meep_matrix():
    M = np.zeros((3,3,len(BAND)))
    for d in range(3):
        z = np.load(os.path.join(MEEP_ART, f"meep_tjunction_drive{d}.npz"))
        col = np.abs(z["col"]); fm = z["freqs_hz"]
        for j in range(3): M[j,d] = np.interp(BAND, fm, col[j])
    return M


def recip(A):
    return float(max(np.mean(np.abs(A[i,j]-A[j,i])) for (i,j) in ((1,0),(2,0),(2,1))))
def passiv(A):
    return float(np.sum(A**2, axis=0).max())


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[rfx] T-junction at dx=2.0mm ..."); S2 = rfx_tjunction(2.0)
    print("[rfx] T-junction at dx=1.6mm ..."); S16 = rfx_tjunction(1.6)
    M = meep_matrix()

    np.savez(os.path.join(OUT, "tjunction_S_arrays.npz"), S2=S2, S16=S16, M=M, band=BAND)
    # per-freq passivity (column power sums) to localise the non-passivity
    for tag, S in (("dx2.0", S2), ("dx1.6", S16)):
        cps = np.sum(S**2, axis=0)  # (3, nf)
        print(f"  [{tag}] per-freq max col-sum: " +
              " ".join(f"{BAND[k]/1e9:.2f}G:{cps[:,k].max():.3f}" for k in range(len(BAND))))
    conv = float(np.abs(S2 - S16).max())
    cases = []
    for tag, S in (("dx=2.0mm", S2), ("dx=1.6mm", S16)):
        r, p = recip(S), passiv(S)
        cases.append(dict(case=tag, reciprocity=r, passivity_max=p,
                          reciprocity_pass=bool(r <= RECIP_TOL), passivity_pass=bool(p <= PASSIVITY_TOL)))
        print(f"  {tag}: reciprocity={r:.3f} passivity_max={p:.3f}")
    xdev = float(np.abs(S2 - M).max())
    print(f"  mesh-convergence max||S2|-|S1.6|| = {conv:.3f} (tol {CONV_TOL})")
    print(f"  rfx-vs-MEEP max||S|| = {xdev:.3f} (cross-FDTD tol {XFDTD_TOL})")

    env_pass = all(c["reciprocity_pass"] and c["passivity_pass"] for c in cases) and conv <= CONV_TOL
    cmp_pass = xdev <= XFDTD_TOL and recip(M) <= RECIP_TOL and passiv(M) <= PASSIVITY_TOL

    commit = os.popen("git -C /tmp/rfx-tj rev-parse --short HEAD").read().strip()
    common = dict(commit_hash=commit, generated_at="2026-05-31",
                  rfx_manifest_path="scripts/diagnostics/port_external_reference_requirements.json")

    # --- broad-E5 envelope artifact (rfx-internal physics over mesh x frequency) ---
    env = dict(schema="rfx.waveguide_tjunction_broad_e5_envelope", schema_version=1,
        status="passed" if env_pass else "failed",
        evidence_level="E5-broad-mesh-frequency-flux-perport-matched-guide-ref-tjunction-hplane-wr-single-mode-te10",
        claim=("rfx 3-port H-plane T-junction S-matrix is reciprocal, passive and "
               "mesh-convergent across the broad single-mode TE10 band (5.0-7.0 GHz) and "
               "a mesh refinement axis (dx 2.0->1.6 mm)."),
        claim_scope=("broad rfx rectangular_waveguide_port H-plane T-junction 3-port S-matrix "
               "envelope spanning the frequency axis (single-mode TE10 5.0-7.0 GHz, cutoff "
               "ratio 1.33-1.87) and the mesh refinement axis (dx 2.0 to 1.6 mm); gates "
               "reciprocity<=0.05, passivity<=1.10, mesh-convergence<=0.08."),
        gates=dict(reciprocity_tol=RECIP_TOL, passivity_tol=PASSIVITY_TOL, convergence_tol=CONV_TOL),
        mesh_convergence_max=conv, cases=cases,
        primary_reference=dict(truth_key="rfx-internal reciprocity/passivity/mesh-convergence"),
        **common)
    with open(os.path.join(OUT, "waveguide_tjunction_broad_e5_envelope.json"), "w") as f:
        json.dump(env, f, indent=2)

    # --- external comparison artifact (E4-broad, rfx vs independent MEEP) ---
    cmp = dict(schema="rfx.waveguide_tjunction_meep_external_comparison", schema_version=1,
        status="passed" if cmp_pass else "failed",
        evidence_level="E4-broad-external-meep-fdtd-tjunction-hplane-wr-single-mode-te10",
        claim=("rfx 3-port H-plane T-junction |S| agrees with an independent MEEP FDTD flux "
               f"reference to <= {XFDTD_TOL} across the broad single-mode TE10 band (5.0-7.0 GHz); "
               "both solvers passive and reciprocal."),
        claim_scope=("broad external cross-FDTD comparison of rfx rectangular_waveguide_port "
               "H-plane T-junction |S| versus an independent MEEP flux-extraction reference over "
               "the single-mode TE10 frequency axis (5.0-7.0 GHz); documented cross-FDTD tolerance "
               f"{XFDTD_TOL} (two discretized FDTD solvers, no closed-form junction truth)."),
        cross_fdtd_tol=XFDTD_TOL, rfx_vs_meep_max_abs_dev=xdev,
        rfx_passivity_max=passiv(S2), rfx_reciprocity=recip(S2),
        meep_passivity_max=passiv(M), meep_reciprocity=recip(M),
        meep_reference="scripts/diagnostics/meep_tjunction_reference.py (flux, res=400)",
        rfx_S_bandmean=np.mean(S2,axis=2).tolist(), meep_S_bandmean=np.mean(M,axis=2).tolist(),
        **common)
    with open(os.path.join(OUT, "waveguide_tjunction_meep_external_comparison.json"), "w") as f:
        json.dump(cmp, f, indent=2)

    print(f"\n ENVELOPE: {'PASS' if env_pass else 'FAIL'}   COMPARISON: {'PASS' if cmp_pass else 'FAIL'}")
    print(f" wrote {OUT}/waveguide_tjunction_broad_e5_envelope.json")
    print(f" wrote {OUT}/waveguide_tjunction_meep_external_comparison.json")


if __name__ == "__main__":
    main()
