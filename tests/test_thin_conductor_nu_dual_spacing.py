"""Non-uniform thin-conductor sheets must realize their specified sheet
resistance on a GRADED node, not a cell-ratio-scaled one (issue #669 review).

A lossy thin conductor folds into ``materials.sigma`` at one E node along its
normal. The NU E update divides the curl at node ``k`` by
``inv_d_e[k] = 2/(d[k-1]+d[k])`` (``rfx/nonuniform.py``), so multiplying the
discrete Ampere law at that node through by the DUAL spacing
``dual[k] = (d[k-1]+d[k])/2`` turns the loss term into a surface current
``sigma_eff * dual[k] * E``: the realized sheet conductance is
``sigma_eff * dual``, never ``sigma_eff * d[k]``.

Both folds — the legacy #373 DC one (``sigma_bulk*t/d_norm``) and the #669
Leontovich one (``1/(Rs0*d_norm)``) — divided by the PRIMAL cell ``d[k]``
instead, which is right only where the two adjacent cells are equal: every
uniform mesh, and every NU node away from a grading step. ON a step the sheet
realized ``Rs * d[k]/dual``, silently.

Measured on the #669 parallel-plate alpha oracle (b = 5 mm, sigma_bulk = 1e4,
Rs0 = 1.98692 ohm/sq, f0 = 10 GHz, alpha_analytic = 1.05482 Np/m), attenuation
relative to the mesh-matched case, before the fix -> after:

    sheet on a 0.25/0.50 mm node (dual 0.375, primal 0.500)   1.2021 -> 0.9984
    sheet on a 1.00/0.25 mm node (dual 0.625, primal 0.250)   0.6214 -> 1.0005

both of which must be 1.000 for a mesh-independent sheet.

Attribution of the small residuals (all measured, this fixture, JAX CPU
float32, 2026-08-19) — the mesh perturbation itself, NOT the fold:

    profile (dz in mm)                          alpha    /alpha(ref)
    ref     [0.5]x12                            0.85777    1.0000
    A       [0.5]x10+[0.25,0.25]+[0.5]          0.85639    0.9984
    A-mesh  [0.5]x9+[0.25,0.25]+[0.5]x2         0.85643    0.9984   <- A's mesh,
                                                                       sheet OFF
                                                                       the step
    B       [0.5]x9+[1.0]+[0.25,0.25]           0.83297    0.9711
    B-ctrl  [0.5]x10+[0.25]x4                   0.83258    0.9706   <- B's split
                                                                       backing
                                                                       stub,
                                                                       sheet OFF
                                                                       the step
    B-mesh  [0.5]x8+[1.0]+[0.5,0.5]             0.85756    0.9998   <- B's coarse
                                                                       guide cell
                                                                       alone

A lands on its mesh-only control to 5 significant figures, and B lands on its
stub-matched control to 0.05%: splitting the 0.5 mm shorted stub behind a plate
into two cells gives that stub an interior E node it did not have and moves
alpha by 2.9% on its own. So each contrast is gated against the control that
shares its mesh and differs ONLY in whether the sheet node's two adjacent
cells are equal — that is the "locally uniform" comparison, and it isolates the
fold from the mesh.

These gates do NOT re-open the O3 envelope: alpha here is ~0.858 Np/m against
1.055 analytic on every profile, which is the R2-STOPped one-cell sheet-dynamics
deficit recorded in tests/test_leontovich_alpha_oracle.py. What is gated here is
mesh INVARIANCE of the realized sheet, which is independent of that offset.
"""

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.materials.thin_conductor import ThinConductor, leontovich_rs
from rfx.nonuniform import e_node_dual_spacings
from rfx.runners.nonuniform import (assemble_materials_nu,
                                    build_nonuniform_grid)

from tests.test_leontovich_alpha_oracle import (
    ABSORBER_N, ABSORBER_SIGMA_MAX, ABSORBER_X0, DOMAIN, DX, F0, FIT_X,
    SIGMA_BULK, SRC_X, THICKNESS, Z_SHEET_HI, Z_SHEET_LO, _fit_alpha,
)

MM = 1e-3

# dz profiles (mm). Every one totals 6.000 mm and keeps interior E nodes on
# z = 0.5 mm (lo plate), 3.0 mm (DFT plane) and 5.5 mm (hi plate); they differ
# only in the cells adjacent to the HI plate and in how the 0.5 mm shorted
# backing stub behind it is meshed.
PROFILES = {
    # matched: both plates sit on nodes with equal neighbours
    "ref":     [0.5] * 12,
    # hi plate on a 0.25/0.50 mm node (d[k-1]=0.25, d[k]=0.50); stub = 1 cell,
    # same as ref, so ref IS its locally-uniform control
    "a_step":  [0.5] * 10 + [0.25, 0.25] + [0.5],
    # hi plate on a 1.00/0.25 mm node (d[k-1]=1.00, d[k]=0.25); stub = 2 cells
    "b_step":  [0.5] * 9 + [1.0] + [0.25, 0.25],
    # b_step's locally-uniform control: same 2-cell stub, hi plate node has
    # EQUAL neighbours (0.25/0.25), so primal == dual and the fold is exact
    # there under either normalization
    "b_ctrl":  [0.5] * 10 + [0.25] * 4,
}

# (case, control, expected dual/primal at the hi plate)
INVARIANCE_CASES = (
    ("a_step", "ref", 0.375 / 0.500),
    ("b_step", "b_ctrl", 0.625 / 0.250),
)

RATIO_GATE = (0.98, 1.02)

# same physical duration on every mesh (the oracle's 4000 steps at dx = 0.5 mm)
T_PHYS = 4000 * (0.99 * DX / (299792458.0 * np.sqrt(3.0)))


# ---------------------------------------------------------------------------
# Pure algebra — no FDTD
# ---------------------------------------------------------------------------

def test_dual_spacing_is_the_e_update_metric():
    """``e_node_dual_spacings`` must be the reciprocal of the metric the NU E
    update actually divides by, and must return a uniform profile's cell size
    bit-exactly (so uniform-profile results cannot move)."""
    from rfx.nonuniform import _profile_to_inv_arrays

    for prof in ([0.5e-3] * 6,
                 [0.25e-3] * 3 + [1.0e-3] * 4,
                 [0.5e-3, 0.5e-3, 0.25e-3, 1.5e-3, 0.75e-3, 0.75e-3]):
        arr = np.asarray(prof, dtype=np.float64)
        inv_d_e, _ = _profile_to_inv_arrays(arr)
        dual = np.asarray(e_node_dual_spacings(jnp.asarray(arr,
                                                           dtype=jnp.float32)))
        np.testing.assert_allclose(dual * np.asarray(inv_d_e), 1.0, rtol=1e-6)
        # explicit form: dual[0] = d[0]; dual[k] = (d[k-1]+d[k])/2
        expect = np.concatenate([arr[:1], 0.5 * (arr[:-1] + arr[1:])])
        np.testing.assert_allclose(dual, expect, rtol=1e-6)

    uniform = jnp.full(7, 0.5e-3, dtype=jnp.float32)
    assert np.array_equal(np.asarray(e_node_dual_spacings(uniform)),
                          np.asarray(uniform))


def _graded_sheet_sigma(zc, **tc_kwargs):
    """Assemble the #373 graded fixture and return
    (sigma at the sheet node, primal cell, dual spacing)."""
    dx = 0.5e-3
    dz = [0.5e-3] * 8 + [1.5e-3] * 8
    L = 24 * dx
    sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                     boundary="cpml", cpml_layers=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((6 * dx, 6 * dx, zc), (18 * dx, 18 * dx, zc)), **tc_kwargs)
        grid = build_nonuniform_grid(
            sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
            sim._dz_profile, dx_profile=sim._dx_profile,
            dy_profile=sim._dy_profile,
            pec_faces=sim._boundary_spec.pec_faces(),
            pmc_faces=sim._boundary_spec.pmc_faces(),
            cpml_axes="xyz")
        sigma = np.asarray(assemble_materials_nu(sim, grid)[0].sigma)
    nz = np.argwhere(sigma > 0)
    assert len(nz) > 0
    ks = sorted({int(i[2]) for i in nz})
    assert len(ks) == 1, ks
    k = ks[0]
    primal = float(np.asarray(grid.dz)[k])
    dual = float(np.asarray(e_node_dual_spacings(grid.dz))[k])
    return float(sigma[tuple(nz[0])]), primal, dual


def test_dc_fold_uses_dual_spacing_at_a_grading_transition():
    """The LEGACY #373 DC fold is corrected too: on a 0.5/1.5 mm transition
    node the sheet realizes R_s = 1/(sigma_bulk*t) against the DUAL spacing.

    Before this fix it divided by the primal cell, so a DC thin conductor
    sitting on a grading step realized R_s * d[k]/dual — here 1.5x its
    specified sheet resistance. Correcting it CHANGES those numbers, which is
    the point: they were wrong.
    """
    sigma_bulk, t = 1.0e3, 35e-6
    rs_spec = 1.0 / (sigma_bulk * t)

    # matched node (deep in the coarse region): primal == dual, unchanged
    sig, primal, dual = _graded_sheet_sigma(8.0e-3, sigma_bulk=sigma_bulk,
                                            thickness=t)
    assert abs(dual / primal - 1.0) < 1e-6
    assert abs(1.0 / (sig * dual) / rs_spec - 1.0) < 1e-4

    # ON the transition: dual = 1.0 mm, primal = 1.5 mm
    sig, primal, dual = _graded_sheet_sigma(4.0e-3, sigma_bulk=sigma_bulk,
                                            thickness=t)
    assert abs(primal / dual - 1.5) < 1e-3, (primal, dual)
    assert abs(1.0 / (sig * dual) / rs_spec - 1.0) < 1e-4, (
        f"DC sheet realizes R_s = {1.0 / (sig * dual):.4f} against the dual "
        f"spacing, specified {rs_spec:.4f}")
    # and the primal normalization is NOT what was used (guard against the
    # assertion above going blind if the fixture stops being graded)
    assert abs(1.0 / (sig * primal) / rs_spec - 1.0) > 0.3


def test_leontovich_fold_uses_dual_spacing_at_a_grading_transition():
    """#669 f0 mode on the same transition node: sigma_eff*Rs0*dual == 1."""
    rs0 = float(leontovich_rs(F0, SIGMA_BULK))
    sig, primal, dual = _graded_sheet_sigma(
        4.0e-3, sigma_bulk=SIGMA_BULK, thickness=THICKNESS,
        surface_impedance_f0=F0)
    assert abs(primal / dual - 1.5) < 1e-3, (primal, dual)
    assert abs(sig * rs0 * dual - 1.0) < 1e-5, sig * rs0 * dual
    assert abs(sig * rs0 * primal - 1.0) > 0.3


def test_graded_fold_stays_on_the_ad_path():
    """The dual-spacing fold keeps the sigma_bulk*t DoF differentiable at a
    transition node: d/dt sum(sigma) == n_cells * sigma_bulk / dual."""
    import jax

    dx = 0.5e-3
    dz = [0.5e-3] * 8 + [1.5e-3] * 8
    L = 24 * dx
    sigma_bulk, t0 = 1.0e3, 35e-6
    sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                     boundary="cpml", cpml_layers=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((6 * dx, 6 * dx, 4.0e-3), (18 * dx, 18 * dx, 4.0e-3)),
            sigma_bulk=sigma_bulk, thickness=t0)
        grid = build_nonuniform_grid(
            sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
            sim._dz_profile, dx_profile=sim._dx_profile,
            dy_profile=sim._dy_profile,
            pec_faces=sim._boundary_spec.pec_faces(),
            pmc_faces=sim._boundary_spec.pmc_faces(),
            cpml_axes="xyz")
    shape = sim._thin_conductors[0].shape
    sigma0 = np.asarray(assemble_materials_nu(sim, grid)[0].sigma)
    n_cells = int((sigma0 > 0).sum())
    k = int(np.argwhere(sigma0 > 0)[0][2])
    dual = float(np.asarray(e_node_dual_spacings(grid.dz))[k])
    g_analytic = n_cells * sigma_bulk / dual

    def loss(thickness):
        sim._thin_conductors[0] = ThinConductor(
            shape=shape, sigma_bulk=sigma_bulk, thickness=thickness)
        return jnp.sum(assemble_materials_nu(sim, grid)[0].sigma)

    g = float(jax.grad(loss)(t0))
    sim._thin_conductors[0] = ThinConductor(
        shape=shape, sigma_bulk=sigma_bulk, thickness=t0)
    assert np.isfinite(g) and g != 0.0
    assert abs(g - g_analytic) / g_analytic < 1e-3, (g, g_analytic)


def test_preflight_advises_on_a_sheet_landing_on_a_grading_step():
    """Advisory pin: a LOSSY sheet on a node whose adjacent cells differ by
    more than 10% is announced (naming both cells and the dual spacing), and
    stays silent on every matched case — the silent case is exactly what hid
    the primal/dual confusion.
    """
    dx = 0.5e-3
    L = 24 * dx
    graded = [0.5e-3] * 8 + [1.5e-3] * 8

    def _msgs(dz, zc, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(
                freq_max=10e9,
                domain=(L, L, 0.0 if dz is not None else 8.0e-3),
                dx=dx, dz_profile=dz, boundary="cpml", cpml_layers=6)
            sim.add_thin_conductor(
                Box((6 * dx, 6 * dx, zc), (18 * dx, 18 * dx, zc)), **kw)
            return " ".join(sim.preflight())

    lossy = dict(sigma_bulk=1.0e3, thickness=35e-6)
    f0mode = dict(sigma_bulk=SIGMA_BULK, thickness=THICKNESS,
                  surface_impedance_f0=F0)

    # fires: sheet ON the 0.5/1.5 mm step, both fold modes
    for kw in (lossy, f0mode):
        fired = _msgs(graded, 4.0e-3, **kw)
        assert "adjacent cells differ" in fired, fired
        assert "DUAL spacing" in fired, fired
        assert "500µm below" in fired and "1.5mm above" in fired, fired

    # quiet: matched node on the same graded mesh; uniform profile; uniform
    # lane (no profile at all); and a PEC sheet, which folds no sigma
    assert "adjacent cells differ" not in _msgs(graded, 8.0e-3, **lossy)
    assert "adjacent cells differ" not in _msgs([0.5e-3] * 16, 4.0e-3, **lossy)
    assert "adjacent cells differ" not in _msgs(None, 4.0e-3, **lossy)
    assert "adjacent cells differ" not in _msgs(
        graded, 4.0e-3, sigma_bulk=5.8e7, thickness=35e-6)


# ---------------------------------------------------------------------------
# FDTD: attenuation must not depend on where the sheet sits in the grading
# ---------------------------------------------------------------------------

def _default_sheet(zs):
    """The committed sheet shape: a zero-thickness Box spanning the guide."""
    return Box((0.0, 0.0, zs), (DOMAIN[0], DOMAIN[1], zs))


def _build_guide_nu(profile_mm, sheet_shape=None):
    """``sheet_shape`` (issue #674) is a ``zs -> Shape`` factory; the default
    reproduces the committed Box guide exactly."""
    sheet_shape = _default_sheet if sheet_shape is None else sheet_shape
    prof = np.asarray(profile_mm, dtype=float) * MM
    assert abs(prof.sum() - DOMAIN[2]) < 1e-12, prof.sum()
    sim = Simulation(
        freq_max=10e9, domain=(DOMAIN[0], DOMAIN[1], 0.0), dx=DX,
        dz_profile=prof,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="pec"),
                              y="pmc", z="pec"),
        cpml_layers=10,
    )
    for i in range(ABSORBER_N):
        s = ABSORBER_SIGMA_MAX * ((i + 0.5) / ABSORBER_N) ** 2
        x0 = ABSORBER_X0 + i * DX
        sim.add_material(f"abs{i}", eps_r=1.0, sigma=s)
        sim.add(Box((x0, 0.0, 0.0), (x0 + DX, DOMAIN[1], DOMAIN[2])),
                material=f"abs{i}")
    for zs in (Z_SHEET_LO, Z_SHEET_HI):
        sim.add_thin_conductor(
            sheet_shape(zs),
            sigma_bulk=SIGMA_BULK, thickness=THICKNESS,
            surface_impedance_f0=F0)
    for k in range(9):
        sim.add_source((SRC_X, 0.001, 1.0e-3 + k * DX), "ez",
                       waveform=GaussianPulse(f0=F0, bandwidth=0.5),
                       amplitude_kind="field")
    sim.add_dft_plane_probe(axis="z", coordinate=3.0e-3, component="ez",
                            freqs=jnp.asarray([F0]), name="midplane")
    sim.add_probe((0.060, 0.001, 3.0e-3), "ez")
    return sim


def _nu_grid_of(sim):
    return build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
        sim._dz_profile, dx_profile=sim._dx_profile,
        dy_profile=sim._dy_profile,
        pec_faces=sim._boundary_spec.pec_faces(),
        pmc_faces=sim._boundary_spec.pmc_faces(),
        cpml_axes="xyz")


_cache = {}


def _run_nu_guide(name, sheet_shape=None, tag="box"):
    """One guide run on PROFILES[name]; cached across the tests below.

    ``sheet_shape``/``tag`` (issue #674) swap in a non-Box sheet of the same
    footprint; ``tag`` keys the cache so the two variants never collide.
    """
    key = (name, tag)
    if key in _cache:
        return _cache[key]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim = _build_guide_nu(PROFILES[name], sheet_shape=sheet_shape)
        grid = _nu_grid_of(sim)
        n_steps = int(np.ceil(T_PHYS / float(grid.dt)))
        result = sim.run(n_steps=n_steps, compute_s_params=False)
    acc = np.asarray(result.dft_planes["midplane"].accumulator)
    j = int(round(0.001 / DX)) + grid.pad_y_lo
    i0 = int(round(FIT_X[0] / DX)) + grid.pad_x_lo
    i1 = int(round(FIT_X[1] / DX)) + grid.pad_x_lo
    xs = (np.arange(i0, i1 + 1) - grid.pad_x_lo) * DX
    alpha, resid = _fit_alpha(xs, np.abs(acc[0, i0:i1 + 1, j]))
    ts = np.abs(np.asarray(result.time_series)[:, 0])
    tail = ts[int(0.95 * len(ts)):].max()
    # sheet-node geometry actually realized
    sigma = np.asarray(assemble_materials_nu(sim, grid)[0].sigma)
    primal = np.asarray(grid.dz)
    dual = np.asarray(e_node_dual_spacings(grid.dz))
    rs0 = float(leontovich_rs(F0, SIGMA_BULK))
    # the graded absorber tops out at 2 S/m; a plate's folded sigma_eff is
    # 1/(Rs0*dual) ~ 800-1350 S/m, so 100 S/m separates them with 8x margin
    sheet_ks = sorted({int(k) for k in np.argwhere(sigma > 100.0)[:, 2]})
    assert len(sheet_ks) == 2, (sheet_ks, float(sigma.max()))
    out = {
        "alpha": alpha,
        "resid": resid,
        "settle_db": float(20 * np.log10(max(tail, 1e-300) / ts.max())),
        "n_steps": n_steps,
        "sheets": [(k, float(primal[k]), float(dual[k]),
                    float(sigma[:, :, k].max()) * rs0 * float(dual[k]))
                   for k in sheet_ks],
        "warnings": [str(w.message) for w in rec],
    }
    _cache[key] = out
    return out


@pytest.mark.slow_physics
@pytest.mark.parametrize("case,control,dual_over_primal", INVARIANCE_CASES)
def test_alpha_invariant_to_sheet_node_grading(case, control,
                                               dual_over_primal):
    """The realized sheet must not depend on the grading at its own node:
    alpha(sheet ON a transition) / alpha(same mesh, sheet OFF it) == 1 within
    [0.98, 1.02]. Pre-fix this read 1.2021 and 0.6402 respectively."""
    got = _run_nu_guide(case)
    ref = _run_nu_guide(control)

    # the fixture must still be discriminating: the hi plate really sits on a
    # step, and the control's plates really do not
    hi_case = got["sheets"][-1]
    assert abs(hi_case[2] / hi_case[1] - dual_over_primal) < 1e-3, hi_case
    for k, prim, dual, _prod in ref["sheets"]:
        assert abs(dual / prim - 1.0) < 1e-6, (k, prim, dual)

    # per-node algebra (R5: the metric must reflect the realized sheet)
    for _k, _p, _d, prod in got["sheets"] + ref["sheets"]:
        assert abs(prod - 1.0) < 1e-5, prod

    # run witnesses
    for out in (got, ref):
        assert out["resid"] < 0.02, out["resid"]
        assert out["settle_db"] < -40.0, out["settle_db"]
        assert not any("PreflightError" in w for w in out["warnings"])

    ratio = got["alpha"] / ref["alpha"]
    lo, hi = RATIO_GATE
    assert lo <= ratio <= hi, (
        f"{case}: alpha {got['alpha']:.5f} vs locally-uniform control "
        f"{control} {ref['alpha']:.5f} -> ratio {ratio:.4f} outside "
        f"[{lo}, {hi}]")
