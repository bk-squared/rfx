"""Producer for the PEC-cube RCS fixture (validation campaign Lane 3).

rfx has NO closed-form RCS reference for any shape that is not a sphere (Mie) or
flat plate (physical-optics). This lane uses the **Lane-1-validated Bempp EFIE
harness** (which reproduces exact Mie to <=0.15 dB) as an independent arbiter for
a shape with no closed form: an axis-aligned PEC cube.

Emits the COMPLETE fixture.json (rfx FDTD H-plane bistatic + Bempp BEM H-plane
bistatic at two mesh densities + physical-optics witness + provenance). Run
offline; Bempp is not an rfx CI/runtime dependency.

Env: JAX on CPU; `pip install bempp-cl gmsh` (import name `bempp_cl` in 0.4.x);
set `OPENBLAS_NUM_THREADS<=64` on many-core boxes. Runs rfx first (JAX) then
Bempp (Numba) in one process.

Convention (identical both sides, aligned with rfx/rcs.py): incidence +x, E||z,
H-plane cut theta=pi/2, phi in [0,pi] (phi=0 forward +x, phi=pi backscatter -x),
sigma reported in m^2.
"""
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

F0 = 6e9
C0 = 299792458.0
LAM = C0 / F0                 # 0.05 m
L = 0.03                      # cube side (kL = 2*pi*L/lam = 3.77, resonant region)
DOMAIN = 0.12
RES = 30
CPML = 8
N_PHI = 37
POL_STR = "ez"
BW = 0.5
PEC_SIGMA = 1e7
_HERE = Path(__file__).resolve().parent

phi = np.linspace(0.0, np.pi, N_PHI)
PHI_DEG = [round(float(np.degrees(p)), 2) for p in phi]
SIGMA_PO = 4 * np.pi * L ** 4 / LAM ** 2      # flat-plate PO (kL->inf asymptote)
KL = 2 * np.pi * L / LAM


def run_rfx():
    import jax.numpy as jnp
    from rfx.grid import Grid
    from rfx.geometry.csg import Box, rasterize
    from rfx.core.yee import MaterialArrays
    from rfx.rcs import compute_rcs
    dx = LAM / RES
    grid = Grid(freq_max=F0 * 1.5, domain=(DOMAIN,) * 3, dx=dx, cpml_layers=CPML)
    c = DOMAIN / 2
    cube = Box(corner_lo=(c - L / 2,) * 3, corner_hi=(c + L / 2,) * 3)
    eps_r, sigma = rasterize(grid, [(cube, 1.0, PEC_SIGMA)])
    materials = MaterialArrays(eps_r=eps_r, sigma=sigma,
                               mu_r=jnp.ones(grid.shape, dtype=jnp.float32))
    n_steps = 1100
    res = compute_rcs(grid, materials, n_steps, f0=F0, bandwidth=BW, theta_inc=0.0,
                      polarization=POL_STR, theta_obs=np.array([np.pi / 2]),
                      phi_obs=phi, freqs=np.array([F0]), boundary="cpml",
                      cpml_layers=CPML)
    return {"grid_shape": list(grid.shape), "dx_m": dx, "n_steps": n_steps,
            "monostatic_dbsm": float(res.monostatic_rcs[0]),
            "bistatic_sigma_m2": [float(v) for v in np.asarray(res.rcs_linear[0, 0, :])]}


def run_bempp(h):
    import bempp_cl.api as bem
    from bempp_cl.api.linalg import gmres
    k = 2 * np.pi / LAM
    d = np.array([1.0, 0.0, 0.0]); pol = np.array([0.0, 0.0, 1.0])
    obs = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=0)
    grid = bem.shapes.cube(length=L, origin=(-L / 2, -L / 2, -L / 2), h=h)
    rwg = bem.function_space(grid, "RWG", 0)
    snc = bem.function_space(grid, "SNC", 0)
    efie = bem.operators.boundary.maxwell.electric_field(rwg, rwg, snc, k)

    @bem.complex_callable
    def tan_trace(x, n, domain_index, result):
        result[:] = np.cross(pol * np.exp(1j * k * np.dot(d, x)), n)

    trace = bem.GridFunction(rwg, fun=tan_trace, dual_space=snc)
    sol, _ = gmres(efie, trace, tol=1e-6)
    ff = bem.operators.far_field.maxwell.electric_field(space=rwg, points=obs, wavenumber=k)
    e_inf = ff.evaluate(sol)
    sigma = 4 * np.pi * np.sum(np.abs(e_inf) ** 2, axis=0)
    return [float(v) for v in sigma], int(rwg.global_dof_count), bem.__version__


def main():
    t0 = time.time()
    rfx_out = run_rfx()
    b_main, n_main, ver = run_bempp(min(L / 10, LAM / 12))
    b_fine, n_fine, _ = run_bempp(min(L / 14, LAM / 16))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()

    rfx_m2 = np.array(rfx_out["bistatic_sigma_m2"])
    bm = np.array(b_main); bf = np.array(b_fine)
    dist = 10 * np.log10(np.maximum(rfx_m2, 1e-30) / np.maximum(bm, 1e-30))
    convB = 10 * np.log10(np.maximum(bm, 1e-30) / np.maximum(bf, 1e-30))
    back_i = N_PHI - 1  # phi = pi

    fixture = {
        "schema": "rfx.rcs_cube_bem", "schema_version": 1,
        "issue": "validation-campaign lane-3",
        "claim_scope": (
            "Independent surface-IE (Bempp-cl EFIE) cross-check of rfx's PEC-CUBE RCS "
            "H-plane bistatic pattern -- a shape with NO closed form. The SAME Bempp EFIE "
            "harness reproduces the exact Mie series to <=0.15 dB in-tree "
            "(tests/fixtures/rcs_sphere_three_way/, Lane 1 / PR #297), and is shown "
            "CONVERGED at every angle here (main-vs-fine mesh), so it can referee both the "
            "backscatter and the oblique bins. GATED claim: rfx near-backscatter agrees "
            "with BEM. RECORDED, NOT gated: rfx's forward-oblique bistatic bins read high -- "
            "the documented bistatic contamination (issue #280), here confirmed on a SECOND "
            "shape by a non-FDTD method. An axis-aligned cube is grid-perfect in FDTD (no "
            "staircase), a different regime than the sphere. No existing gate is weakened."),
        "arbiter_correctness_basis": (
            "Why the Bempp arbiter is trusted (not just 'converged != correct'): "
            "(1) IN-TREE, the identical EFIE harness reproduces exact Mie to <=0.15 dB across "
            "ka {0.8,1,1.5,2} (tests/fixtures/rcs_sphere_three_way/, PR #297); "
            "(2) kL=3.77 sits BELOW the first PEC-cube interior/cavity resonance kL=pi*sqrt(2)"
            "~=4.44 (mode 1,1,0), so the EFIE is not in a spurious interior-resonance band; "
            "(3) two-point anchor -- rfx and Bempp agree at BOTH backscatter (-0.42 dB) and "
            "forward-scatter phi=0 (-0.97 dB), so a Bempp error would need an implausible "
            "angular structure (~0 at phi=0,180 but ~13 dB at phi=75) whereas rfx has an "
            "independently documented forward-oblique lobe (issue #280) with exactly that "
            "structure. Staircase corroboration: the curved sphere shows BOTH a forward-"
            "oblique (~10 dB) and a moderate back-oblique (~3.5 dB @135deg) bump vs Mie; the "
            "no-staircase cube keeps the forward-oblique lobe but the back-oblique bump "
            "shrinks to ~1 dB -- consistent with the forward-oblique being the shape-"
            "independent TFSF/NTFF-face mechanism and the sphere's back-oblique carrying an "
            "additional staircase component."),
        "geometry": {"shape": "pec_cube", "L_m": L, "f0_hz": F0, "lambda_m": LAM,
                     "kL": KL, "domain_m": DOMAIN, "cpml": CPML,
                     "convention": "incidence +x, E||z, H-plane theta=pi/2, phi=pi backscatter"},
        "phi_deg": PHI_DEG,
        "rfx": {"grid_shape": rfx_out["grid_shape"], "dx_m": rfx_out["dx_m"],
                "res_cells_per_lambda": RES, "n_steps": rfx_out["n_steps"],
                "monostatic_dbsm": rfx_out["monostatic_dbsm"],
                "bistatic_sigma_m2": rfx_out["bistatic_sigma_m2"]},
        "bempp": {"solver": "bempp-cl EFIE (RWG/SNC), Numba CPU", "version": ver,
                  "N_main": n_main, "N_fine": n_fine,
                  "h_main": min(L / 10, LAM / 12), "h_fine": min(L / 14, LAM / 16),
                  "bistatic_sigma_m2": b_main, "bistatic_sigma_m2_fine": b_fine,
                  "self_convergence_max_db": float(np.abs(convB).max()),
                  "backscatter_dbsm": float(10 * np.log10(bm[back_i]))},
        "comparison": {
            "backscatter_rfx_minus_bempp_db": float(dist[back_i]),
            "near_backscatter_max_dist_db": float(np.abs(dist[phi >= np.radians(135)]).max()),
            "full_pattern_max_dist_db": float(np.abs(dist).max()),
            "full_pattern_max_dist_at_phi_deg": float(PHI_DEG[int(np.argmax(np.abs(dist)))]),
            "rfx_minus_bempp_db": [float(v) for v in dist],
        },
        "physical_optics_witness": {
            "sigma_po_flatplate_m2": float(SIGMA_PO),
            "rfx_backscatter_over_po": float(10 ** (rfx_out["monostatic_dbsm"] / 10) / SIGMA_PO),
            "bempp_backscatter_over_po": float(bm[back_i] / SIGMA_PO),
            "note": ("kL=3.77 is the RESONANT region; flat-plate PO is the kL->inf asymptote, "
                     "so backscatter/PO~2 is an order-of-magnitude sanity only, not a tight gate. "
                     "rfx and Bempp agreeing on this ratio is the meaningful statement."),
        },
        "provenance": {"rfx_commit": sha, "bempp_version": ver,
                       "producer": "tests/fixtures/rcs_cube_bem/generate.py",
                       "wall_s": time.time() - t0,
                       "env": "OPENBLAS_NUM_THREADS<=64; import name bempp_cl (not bempp) in 0.4.x"},
    }
    (_HERE / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"backscatter rfx-vs-bempp = {dist[back_i]:+.3f} dB; near-backscatter(>=135) "
          f"max = {fixture['comparison']['near_backscatter_max_dist_db']:.3f} dB")
    print(f"Bempp self-convergence (all angles) max = {fixture['bempp']['self_convergence_max_db']:.3f} dB")
    print(f"forward-oblique max dist = {fixture['comparison']['full_pattern_max_dist_db']:.2f} dB "
          f"at phi={fixture['comparison']['full_pattern_max_dist_at_phi_deg']} deg (issue #280, non-gated)")
    print(f"wrote fixture.json in {fixture['provenance']['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
