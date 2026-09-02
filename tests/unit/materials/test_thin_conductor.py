"""Tests for thin conductor subcell model.

Validates:
1. Effective conductivity computation
2. Selective application via shape mask
3. Integration with Simulation API
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.geometry.csg import Box
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor


def test_thin_conductor_sigma_eff():
    """σ_eff = σ_bulk · (t / Δx) should be correct for lossy conductors.

    For PEC-level conductors (σ_eff >= 1e6), the code routes to pec_mask
    instead of setting sigma. Test both cases.
    """
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))
    dx = grid.dx

    # Case 1: Lossy conductor (σ_eff below PEC threshold)
    sigma_bulk_lossy = 1e4
    thickness = 35e-6
    expected_sigma_eff = sigma_bulk_lossy * (thickness / dx)

    shape_box = Box((0.005, 0.005, 0.0), (0.015, 0.015, 0.001))
    tc_lossy = ThinConductor(shape=shape_box, sigma_bulk=sigma_bulk_lossy, thickness=thickness)

    materials = init_materials(grid.shape)
    materials, pec_mask = apply_thin_conductor(grid, tc_lossy, materials)

    mask = shape_box.mask(grid)
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        sigma_val = float(materials.sigma[i, j, k])
        assert abs(sigma_val - expected_sigma_eff) / max(expected_sigma_eff, 1e-30) < 0.01, \
            f"Lossy: σ_eff={sigma_val:.4e}, expected={expected_sigma_eff:.4e}"

    # Case 2: PEC conductor (copper → σ_eff > 1e6 → pec_mask)
    sigma_bulk_pec = 5.8e7  # copper
    tc_pec = ThinConductor(shape=shape_box, sigma_bulk=sigma_bulk_pec, thickness=thickness)
    assert tc_pec.is_pec, "Copper 35µm should be PEC"

    materials2 = init_materials(grid.shape)
    materials2, pec_mask2 = apply_thin_conductor(grid, tc_pec, materials2)

    if pec_mask2 is not None and len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        assert bool(pec_mask2[i, j, k]), "PEC conductor should set pec_mask"

    # Outside should be unaffected
    outside_idx = np.argwhere(~np.array(mask))
    if len(outside_idx) > 0:
        i, j, k = outside_idx[0]
        assert float(materials.sigma[i, j, k]) == 0.0


def test_thin_conductor_preserves_outside():
    """Thin conductor should not modify material outside its shape."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))

    # Pre-fill with some background
    materials = init_materials(grid.shape)
    materials = materials._replace(
        eps_r=jnp.full(grid.shape, 4.4, dtype=jnp.float32),
        sigma=jnp.full(grid.shape, 0.025, dtype=jnp.float32),
    )

    shape_box = Box((0.005, 0.005, 0.0), (0.01, 0.01, 0.001))
    # Use lossy conductor (below PEC threshold) so sigma is modified, not pec_mask
    tc = ThinConductor(shape=shape_box, sigma_bulk=1e4, thickness=35e-6, eps_r=1.0)
    materials, _ = apply_thin_conductor(grid, tc, materials)

    mask = shape_box.mask(grid)

    # Inside: eps_r should be 1.0 (conductor), sigma should be thin-conductor value
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        assert float(materials.eps_r[i, j, k]) == 1.0

    # Outside: eps_r should still be 4.4, sigma should still be 0.025
    outside_idx = np.argwhere(~np.array(mask))
    if len(outside_idx) > 0:
        i, j, k = outside_idx[len(outside_idx) // 2]
        assert abs(float(materials.eps_r[i, j, k]) - 4.4) < 1e-4
        assert abs(float(materials.sigma[i, j, k]) - 0.025) < 1e-4


def test_thin_conductor_api_integration():
    """ThinConductor works through the Simulation API.

    Copper (5.8e7 S/m, 35um) exceeds PEC threshold → routed to pec_mask.
    """
    from rfx.api import Simulation

    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.002), boundary="pec")
    sim.add_material("substrate", eps_r=4.4, sigma=0.025)
    sim.add(Box((0, 0, 0), (0.02, 0.02, 0.001)), material="substrate")
    sim.add_thin_conductor(
        Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001)),
        sigma_bulk=5.8e7,
        thickness=35e-6,
    )

    # Should build without error
    grid = sim._build_grid()
    materials, debye, lorentz, pec_mask, pec_shapes, *_ = sim._assemble_materials(grid)

    # Copper thin conductor is PEC-level → check pec_mask, not sigma
    tc_box = Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001))
    mask = tc_box.mask(grid)
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0 and pec_mask is not None:
        i, j, k = inside_idx[len(inside_idx) // 2]
        assert bool(pec_mask[i, j, k]), \
            "Copper thin conductor should be in pec_mask"

    print(f"\nThin conductor API integration: OK, grid={grid.shape}")


def test_thin_conductor_nonuniform_reflects_like_box():
    """#369: a PEC thin conductor on the non-uniform (dz_profile) path must
    reflect/resonate identically to a geometry PEC Box at the same cell.

    Before the fix, ``assemble_materials_nu`` skipped ALL thin conductors
    ("needs a uniform Grid"), so an ``add_thin_conductor`` PEC sheet on a
    dz_profile grid was silently dropped — no pec_mask, no reflection: the
    box-PEC cavity rang while the thin-sheet cavity decayed. This is an
    end-to-end behavioural lock through the real NU run path.
    """
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    dx = 1.5e-3
    N = 30
    L = N * dx
    dz = [dx] * N
    zc = 15 * dx + 0.5 * dx

    def late_energy(kind):
        sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                         boundary="cpml", cpml_layers=8)
        px_lo, px_hi = L / 2 - 6 * dx, L / 2 + 6 * dx
        py_lo, py_hi = L / 2 - 4 * dx, L / 2 + 4 * dx
        if kind == "box":
            sim.add(Box((px_lo, py_lo, 15 * dx), (px_hi, py_hi, 16 * dx)),
                    material="pec")
        else:
            sim.add_thin_conductor(Box((px_lo, py_lo, zc), (px_hi, py_hi, zc)),
                                   sigma_bulk=5.8e7, thickness=35e-6)
        sim.add_source(position=(px_lo + 3 * dx, L / 2, 10 * dx),
                       component="ez",
                       waveform=GaussianPulse(f0=6e9, bandwidth=1.0))
        sim.add_probe(position=(px_lo + 8 * dx, L / 2 + 2 * dx, 10 * dx),
                      component="ez")
        ts = np.asarray(sim.run(n_steps=250, skip_preflight=True).time_series).ravel()
        return float(np.max(np.abs(ts[-50:])))

    box_e = late_energy("box")
    thin_e = late_energy("thin")
    assert box_e > 0.0
    # Identical nominal patch cell → identical late-time cavity energy.
    # Pre-fix this ratio was <= 0.37 (thin dropped); post-fix it is ~1.0.
    assert abs(box_e - thin_e) / box_e < 0.1, (
        f"#369 regression: PEC thin sheet must ring like a box PEC on the NU "
        f"path (box={box_e:.3e}, thin={thin_e:.3e}, ratio={thin_e / box_e:.3f})")

    # R5: don't trust the aggregate energy alone — assert cell-identity on the
    # assembled pec_mask itself, which a small cell mismatch can't hide behind a
    # resonance that happens not to move. On a UNIFORM-valued NU profile the thin
    # sheet and the matching-thickness box coincide cell-for-cell. (#371, resolved
    # as not-a-placement-bug: even on a GENUINELY graded dz_profile a thin sheet
    # and a genuinely MATCHING-THICKNESS (sub-cell) box still select the same
    # z-layer, because both take the argmin-nearest-NODE thin branch. Only a
    # >=1-cell VOLUME box — a physically different object — lands a layer away.
    # See test_thin_conductor_graded_matches_matching_thickness_box_not_onecell.)
    from rfx.runners.nonuniform import (build_nonuniform_grid,
                                        assemble_materials_nu)

    def nu_pec_mask(kind):
        sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                         boundary="cpml", cpml_layers=8)
        px_lo, px_hi = L / 2 - 6 * dx, L / 2 + 6 * dx
        py_lo, py_hi = L / 2 - 4 * dx, L / 2 + 4 * dx
        if kind == "box":
            sim.add(Box((px_lo, py_lo, 15 * dx), (px_hi, py_hi, 16 * dx)),
                    material="pec")
        else:
            sim.add_thin_conductor(Box((px_lo, py_lo, zc), (px_hi, py_hi, zc)),
                                   sigma_bulk=5.8e7, thickness=35e-6)
        grid = build_nonuniform_grid(
            sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
            sim._dz_profile, dx_profile=sim._dx_profile,
            dy_profile=sim._dy_profile,
            pec_faces=(sim._boundary_spec.pec_faces()
                       if sim._boundary_spec is not None else None),
            pmc_faces=(sim._boundary_spec.pmc_faces()
                       if sim._boundary_spec is not None else None),
            cpml_axes="".join(a for a in "xyz"
                              if a not in (sim._periodic_axes or "")),
        )
        return np.asarray(assemble_materials_nu(sim, grid)[3])

    box_mask = nu_pec_mask("box")
    thin_mask = nu_pec_mask("thin")
    assert thin_mask is not None and int(thin_mask.sum()) > 0, \
        "#369: PEC thin conductor must produce a non-empty pec_mask on the NU path"
    assert np.array_equal(box_mask, thin_mask), \
        ("on a uniform NU profile a PEC thin sheet and a matching-thickness box "
         "must select identical pec_mask cells")


def test_thin_conductor_lossy_nonuniform_runs_no_skip_warn():
    """#373: a lossy (non-PEC) thin conductor now RUNS end-to-end on the NU
    (dz_profile) path (folded into sigma), no longer warning-and-skipping.
    (Superseded the earlier #370 warn-and-skip lock — end-to-end coverage
    complementing the unit-level test_lossy_thin_conductor_nonuniform_uses_local_dz.)
    """
    import warnings
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    dx = 1.5e-3
    N = 24
    L = N * dx
    dz = [dx] * N
    sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                     boundary="cpml", cpml_layers=6)
    # sigma_eff below the PEC threshold → lossy branch.
    sim.add_thin_conductor(Box((9 * dx, 9 * dx, 15.5 * dx),
                               (15 * dx, 15 * dx, 15.5 * dx)),
                           sigma_bulk=1.0e3, thickness=35e-6)
    sim.add_source(position=(6 * dx, L / 2, 10 * dx), component="ez",
                   waveform=GaussianPulse(f0=6e9, bandwidth=1.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = sim.run(n_steps=10, skip_preflight=True)
    assert res is not None, "lossy thin conductor on NU must run end-to-end"
    assert not any("not yet supported" in str(wi.message).lower()
                   and "thin conductor" in str(wi.message).lower() for wi in w), \
        "#373: lossy thin conductor on NU must no longer warn-and-skip"


def test_thin_conductor_graded_matches_matching_thickness_box_not_onecell():
    """#371: on a GENUINELY graded dz_profile, a PEC thin sheet selects the same
    z-layer as a genuinely MATCHING-THICKNESS (sub-cell) box — because both take
    the argmin thin branch, which realizes the conductor at the nearest E-NODE
    (apply_pec_mask zeros collocated tangential Ex/Ey there). A one-cell VOLUME
    box is a different object and legitimately lands one layer away; the
    sheet's layer is the one whose NODE is NEAREST z0 (minimum realized-plane
    error), which the one-cell box is NOT required to match.

    Wording note (#562 / #568 item 3): this said "cell CENTRE" until the NU
    coordinates became node-based. The coords the argmin runs over are nodes,
    and nodes are where tangential Ex/Ey actually sit, so the #371 closure this
    file records is STRENGTHENED by the correction — the minimized quantity is
    now the distance to the plane the conductor is really realized on.
    """
    import numpy as np
    from rfx.api import Simulation
    from rfx.runners.nonuniform import (build_nonuniform_grid,
                                        assemble_materials_nu)
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    from rfx.geometry.csg import Box

    dx = 0.5e-3
    dz = [0.5e-3] * 8 + [1.5e-3] * 8          # genuinely graded: fine then coarse
    L = 24 * dx
    px = (L / 2 - 6 * dx, L / 2 + 6 * dx)
    py = (L / 2 - 4 * dx, L / 2 + 4 * dx)
    z_req = 4.1e-3
    t = 35e-6

    def nu_pec_z(add_fn):
        sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                         boundary="cpml", cpml_layers=8)
        add_fn(sim)
        grid = build_nonuniform_grid(
            sim._freq_max, sim._domain, sim._dx, sim._cpml_layers, sim._dz_profile,
            dx_profile=sim._dx_profile, dy_profile=sim._dy_profile,
            pec_faces=(sim._boundary_spec.pec_faces() if sim._boundary_spec else None),
            pmc_faces=(sim._boundary_spec.pmc_faces() if sim._boundary_spec else None),
            cpml_axes="".join(a for a in "xyz" if a not in (sim._periodic_axes or "")))
        coords = coords_from_nonuniform_grid(grid)
        pm = np.asarray(assemble_materials_nu(sim, grid)[3])
        zc = np.asarray(coords.z)
        zl = sorted(set(np.argwhere(pm)[:, 2].tolist()))
        return zl, zc

    sheet_z, zc = nu_pec_z(lambda s: s.add_thin_conductor(
        Box((px[0], py[0], z_req), (px[1], py[1], z_req)),
        sigma_bulk=5.8e7, thickness=t))
    mbox_z, _ = nu_pec_z(lambda s: s.add(
        Box((px[0], py[0], z_req - t / 2), (px[1], py[1], z_req + t / 2)),
        material="pec"))
    onecell_z, _ = nu_pec_z(lambda s: s.add(
        Box((px[0], py[0], 4.0e-3), (px[1], py[1], 5.5e-3)), material="pec"))

    # R5 witness: realized plane (selected E-node) vs requested z0.
    k = sheet_z[0]
    sheet_err = abs(float(zc[k]) - z_req)
    nearest = int(np.argmin(np.abs(zc - z_req)))
    assert sheet_z == [nearest], (
        f"#371: thin sheet must land on the NEAREST-NODE layer {nearest} "
        f"(realized {float(zc[nearest])*1e3:.3f}mm); got {sheet_z}")
    assert sheet_z == mbox_z, (
        f"#371: sheet must equal a genuinely matching-thickness box "
        f"(sheet={sheet_z}, matching-box={mbox_z})")
    # sanity: the nearest-node layer really is the min-error placement
    assert sheet_err <= abs(float(zc[onecell_z[0]]) - z_req) + 1e-12
    # A one-cell VOLUME box is a physically different object (1-cell slab vs a
    # zero-thickness sheet); on this graded profile it lands one layer away
    # (onecell_z=[16] vs sheet_z=[15]). That divergence is expected and correct,
    # not a placement bug — it is exactly what #371 originally mis-read as one.


def test_lossy_thin_conductor_nonuniform_uses_local_dz():
    """#373: a LOSSY (non-PEC) thin conductor on the non-uniform (dz_profile)
    path folds into sigma using the LOCAL cell size normal to the sheet, not a
    uniform grid.dx — so the sheet resistance R_s = 1/(sigma_bulk*t) is
    preserved on a graded mesh. Before this, lossy sheets warned and were
    silently skipped on the NU path.
    """
    import warnings
    from rfx.api import Simulation
    from rfx.runners.nonuniform import (build_nonuniform_grid,
                                        assemble_materials_nu)

    dx = 0.5e-3
    dz = [0.5e-3] * 8 + [1.5e-3] * 8       # graded: coarse region has dz=3*dx
    L = 24 * dx
    zc = 8.0e-3                            # deep in the coarse (1.5mm) region
    sigma_bulk, t = 1.0e3, 35e-6          # lossy: sigma_eff << PEC threshold

    sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                     boundary="cpml", cpml_layers=6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim.add_thin_conductor(Box((6 * dx, 6 * dx, zc), (18 * dx, 18 * dx, zc)),
                               sigma_bulk=sigma_bulk, thickness=t)
        grid = build_nonuniform_grid(
            sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
            sim._dz_profile, dx_profile=sim._dx_profile,
            dy_profile=sim._dy_profile,
            pec_faces=(sim._boundary_spec.pec_faces()
                       if sim._boundary_spec is not None else None),
            pmc_faces=(sim._boundary_spec.pmc_faces()
                       if sim._boundary_spec is not None else None),
            cpml_axes="".join(a for a in "xyz"
                              if a not in (sim._periodic_axes or "")))
        sigma = np.asarray(assemble_materials_nu(sim, grid)[0].sigma)

    assert not any("not yet supported" in str(wi.message).lower() for wi in w), \
        "#373: lossy thin conductor on NU must no longer warn-and-skip"

    nz = np.argwhere(sigma > 0)
    assert len(nz) > 0, "#373: lossy thin conductor produced no sigma cells"
    k = int(nz[0][2])
    dz_local = float(np.asarray(grid.dz)[k])
    assert abs(dz_local - 1.5e-3) < 1e-9, "sheet must land in the coarse region"
    sigma_cell = float(sigma[tuple(nz[len(nz) // 2])])

    # Uses LOCAL dz (1.5mm), NOT the uniform grid.dx (0.5mm, which would be 3x).
    sigma_local = sigma_bulk * t / dz_local
    sigma_uniform_bug = sigma_bulk * t / dx
    assert abs(sigma_cell - sigma_local) / sigma_local < 1e-4, \
        f"sigma_eff must use local dz ({sigma_local:.3e}), got {sigma_cell:.3e}"
    assert abs(sigma_cell - sigma_uniform_bug) / sigma_uniform_bug > 0.5, \
        "sigma_eff must NOT use the uniform grid.dx (the pre-fix bug)"

    # Sheet resistance preserved: R_s = 1/(sigma_eff * dz_local) = 1/(sigma_bulk*t).
    R_s = 1.0 / (sigma_cell * dz_local)
    R_s_true = 1.0 / (sigma_bulk * t)
    assert abs(R_s - R_s_true) / R_s_true < 1e-3, \
        f"sheet resistance not preserved: {R_s:.3f} vs {R_s_true:.3f}"


def test_lossy_thin_conductor_nonuniform_ad_gate():
    """#373 AD gate (hard constraint): the lossy σ fold must stay on the
    gradient path. Differentiate through assemble_materials_nu w.r.t. the
    sheet thickness (thickness avoids the is_pec Python-bool routing on
    sigma_bulk) and check the gradient is finite, non-zero, and matches the
    closed form d/dt Σσ = n_cells · sigma_bulk / dz_local.
    """
    import jax
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.materials.thin_conductor import ThinConductor
    from rfx.runners.nonuniform import (build_nonuniform_grid,
                                        assemble_materials_nu)

    dx = 0.5e-3
    dz = [0.5e-3] * 8 + [1.5e-3] * 8
    L = 24 * dx
    zc = 8.0e-3                      # coarse region, dz_local = 1.5mm
    sigma_bulk, t0 = 1.0e3, 35e-6

    sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                     boundary="cpml", cpml_layers=6)
    sim.add_thin_conductor(Box((6 * dx, 6 * dx, zc), (18 * dx, 18 * dx, zc)),
                           sigma_bulk=sigma_bulk, thickness=t0)
    shape = sim._thin_conductors[0].shape
    grid = build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers, sim._dz_profile,
        dx_profile=sim._dx_profile, dy_profile=sim._dy_profile,
        pec_faces=(sim._boundary_spec.pec_faces()
                   if sim._boundary_spec is not None else None),
        pmc_faces=(sim._boundary_spec.pmc_faces()
                   if sim._boundary_spec is not None else None),
        cpml_axes="".join(a for a in "xyz"
                          if a not in (sim._periodic_axes or "")))

    # Closed form computed from the CONCRETE state first (before grad mutates
    # the conductor): Σσ = n_cells · sigma_bulk · t / dz_local
    # ⇒ dΣσ/dt = n_cells · sigma_bulk / dz_local.
    sigma0 = np.asarray(assemble_materials_nu(sim, grid)[0].sigma)
    n_cells = int((sigma0 > 0).sum())
    k = int(np.argwhere(sigma0 > 0)[0][2])
    dz_local = float(np.asarray(grid.dz)[k])
    g_analytic = n_cells * sigma_bulk / dz_local

    def loss(thickness):
        sim._thin_conductors[0] = ThinConductor(
            shape=shape, sigma_bulk=sigma_bulk, thickness=thickness)
        return jnp.sum(assemble_materials_nu(sim, grid)[0].sigma)

    g = float(jax.grad(loss)(t0))
    # restore the concrete conductor so the mutated (traced) one does not leak.
    sim._thin_conductors[0] = ThinConductor(
        shape=shape, sigma_bulk=sigma_bulk, thickness=t0)

    assert np.isfinite(g), "#373: gradient through the lossy σ fold is non-finite"
    assert abs(g) > 0, "#373: gradient through the lossy σ fold is zero (fold off the AD path)"
    assert abs(g - g_analytic) / g_analytic < 1e-3, \
        f"#373: grad {g:.4e} != closed form {g_analytic:.4e}"


def test_is_thin_classification_uses_local_cell_width():
    """#374: Box.mask_on_coords classifies thin-vs-volume by the LOCAL cell
    width (min of the two neighbouring centre spacings), not the single forward
    spacing. Bit-identical on a uniform axis; on a graded axis it no longer
    over-counts the cell size at a fine/coarse transition (which collapsed a
    >1-fine-cell shape onto a single argmin cell).
    """
    import jax.numpy as jnp
    from rfx.geometry.csg import Box

    # Coordinates are host float64, matching the exact node spine production
    # lanes supply since the #802/#807 fix. The old jnp.asarray route stored
    # centres in float32 (x64=0), where 7.5e-3 is one ulp BELOW the real face
    # value — the former ==2 result relied on the face rounding the same way
    # (an exact-tie accident, the very defect class #802 removed). With exact
    # coordinates the half-open [lo, hi) convention gives 2 by arithmetic.
    zero = np.zeros(1)

    def n_cells(coords, lo, hi):
        m = np.asarray(Box((0, 0, lo), (0, 0, hi)).mask_on_coords(zero, zero, coords))
        return int(m.sum())

    # --- Uniform axis: unchanged (thin <= 1 cell, volume otherwise) ---
    xu = np.arange(20) * 1e-3 + 0.5e-3   # 1mm cells, centres at k+0.5 (float64)
    assert n_cells(xu, 5e-3, 5.2e-3) == 1              # sub-cell sheet → 1
    assert n_cells(xu, 5e-3, 7.5e-3) == 2              # 2.5mm span → volume, 2 centres

    # --- Graded axis: fine 0.5mm (cells 0-7) then coarse 1.5mm (cells 8-15) ---
    dz = np.array([0.5e-3] * 8 + [1.5e-3] * 8)
    edges = np.concatenate([[0.0], np.cumsum(dz)])
    zc = 0.5 * (edges[:-1] + edges[1:])   # host float64, production route
    zc_np = np.asarray(zc)

    # A genuine sub-cell sheet still resolves to exactly one cell (the nearest).
    z0 = 2.75e-3                                       # a fine-region centre
    assert n_cells(zc, z0, z0) == 1
    k = int(np.argmin(np.abs(zc_np - z0)))
    m = np.asarray(Box((0, 0, z0), (0, 0, z0)).mask_on_coords(zero, zero, zc))
    assert int(np.argwhere(m)[0][2]) == k             # z-index lands on nearest cell

    # The local cell width at a fine-region centre is the fine size (0.5mm), so a
    # 1.2mm-extent box there spans multiple fine cells (volume), not one.
    assert n_cells(zc, 2.0e-3, 3.2e-3) >= 2


# ---------------------------------------------------------------------------
# Leontovich surface-impedance mode (issue #669)
# ---------------------------------------------------------------------------

def _nu_graded_grid(sim):
    """Build the graded NU grid the #373 fixtures use (dz 0.5mm x8 + 1.5mm x8)."""
    from rfx.runners.nonuniform import build_nonuniform_grid
    return build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
        sim._dz_profile, dx_profile=sim._dx_profile,
        dy_profile=sim._dy_profile,
        pec_faces=(sim._boundary_spec.pec_faces()
                   if sim._boundary_spec is not None else None),
        pmc_faces=(sim._boundary_spec.pmc_faces()
                   if sim._boundary_spec is not None else None),
        cpml_axes="".join(a for a in "xyz"
                          if a not in (sim._periodic_axes or "")))


def _nu_graded_sim(zc=8.0e-3, **tc_kwargs):
    """Sim + graded-dz thin conductor, reusing the #373 fixture geometry.

    ``dz = [0.5 mm] x8 + [1.5 mm] x8`` puts interior E nodes at 0, 0.5, ...,
    4.0 mm then 5.5, 7.0, 8.5, ... mm. The default ``zc = 8.0 mm`` lands the
    sheet on node 8.5 mm, deep in the coarse region, where the two adjacent
    cells are EQUAL (dual spacing == primal cell == 1.5 mm). ``zc = 4.0 mm``
    lands it exactly ON the grading transition, where they differ
    (d_below = 0.5 mm, d_above = 1.5 mm, dual = 1.0 mm) — the case that
    discriminates the two normalizations.
    """
    import warnings
    from rfx.api import Simulation

    dx = 0.5e-3
    dz = [0.5e-3] * 8 + [1.5e-3] * 8
    L = 24 * dx
    sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                     boundary="cpml", cpml_layers=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((6 * dx, 6 * dx, zc), (18 * dx, 18 * dx, zc)), **tc_kwargs)
    return sim


def test_leontovich_rs_pure_function_o1a():
    """O1a: leontovich_rs(10 GHz, copper) equals the hand-computed constant.

    Hand computation (x64): sqrt(pi * 1e10 * 1.25663706212e-6 / 5.8e7)
    = sqrt(6.8066237248e-4) = 2.6089507e-2 ohm/sq (the contract's pinned
    "2.609e-2" stated to 8 significant figures — 4 significant figures
    cannot carry an rtol 1e-6 comparison). And it must equal the inline
    formula to rtol 1e-12 (same algebra, x64).
    """
    from tests._x64_compat import enable_x64
    from rfx.core.yee import MU_0
    from rfx.materials.thin_conductor import leontovich_rs

    with enable_x64():
        rs = float(leontovich_rs(jnp.float64(10e9), jnp.float64(5.8e7)))
        assert abs(rs - 2.6089507e-2) / 2.6089507e-2 < 1e-6, rs
        inline = float(jnp.sqrt(jnp.pi * jnp.float64(10e9) * MU_0
                                / jnp.float64(5.8e7)))
        assert abs(rs - inline) / inline < 1e-12


def test_leontovich_dc_limit_equivalence_o1b():
    """O1b (the issue's DC gate, pure algebra): for (sigma_dc, t) chosen so
    that 1/(sigma_dc*t) == Rs0, the legacy DC fold's sigma_eff equals the
    f0-mode realized sheet conductivity on the same grid to rtol 1e-9, on
    BOTH lanes.

    #677 retarget, equivalent discriminant: the f0 sheet no longer rides
    ``materials.sigma`` — it is emitted as a ``SheetImpedanceSpec`` whose
    ``sigma_sheet`` is exactly the per-node quantity the old fold wrote
    into the arrays, so the identity under test (realized sheet
    conductance == DC sheet conductance when 1/(sigma_dc*t) == Rs0) is
    asserted on ``spec.sigma_sheet`` instead of ``materials.sigma``.
    """
    from tests._x64_compat import enable_x64
    from rfx.core.yee import init_materials
    from rfx.materials.thin_conductor import leontovich_rs
    from rfx.runners.nonuniform import assemble_materials_nu

    f0, sigma_bulk = 10e9, 1e4
    with enable_x64():
        rs0 = float(leontovich_rs(jnp.float64(f0), jnp.float64(sigma_bulk)))
        t = 35e-6
        sigma_dc = 1.0 / (rs0 * t)         # 1/(sigma_dc*t) == Rs0 exactly

        # -- uniform lane --
        grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))
        shape = Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001))
        got = {}
        for name, tc in (
            ("dc", ThinConductor(shape=shape, sigma_bulk=sigma_dc,
                                 thickness=t)),
            ("f0", ThinConductor(shape=shape, sigma_bulk=sigma_bulk,
                                 thickness=t, surface_impedance_f0=f0)),
        ):
            specs = []
            mats, _ = apply_thin_conductor(grid, tc,
                                           init_materials(grid.shape), None,
                                           sheet_specs=specs)
            if name == "dc":
                arr_dc = float(jnp.max(mats.sigma))
                # rtol 1e-9 is the ALGEBRA gate; the DC fold's array write
                # goes through the float32 material arrays, so the pure-x64
                # DC algebra is the comparand and the array is pinned to it
                # at float32 resolution (same split as the NU block below).
                got[name] = sigma_dc * t / grid.dx
                assert arr_dc > 0
                assert abs(arr_dc - got[name]) / got[name] < 1e-6
            else:
                # #677: f0 emits a spec and leaves the arrays sheet-free
                assert float(jnp.max(mats.sigma)) == 0.0
                assert len(specs) == 1
                got[name] = float(jnp.max(specs[0].sigma_sheet))
        assert got["dc"] > 0
        assert abs(got["f0"] - got["dc"]) / got["dc"] < 1e-9, got

        # -- non-uniform lane (graded dz, coarse-region sheet) --
        # The NU assembly's arrays are float32 by design (PROMOTE never
        # PIN governs the solver, not the assembler), so the rtol 1e-9
        # equivalence — a pure-algebra gate — is asserted on the two fold
        # FORMULAS evaluated in x64 with the REAL local d_norm taken from
        # the NU grid; the assembled f32 array is then pinned to that
        # algebra at float32 resolution. Comparing two f32 arrays at 1e-9
        # would be impossible arithmetic, not a stronger gate.
        got_nu = {}
        for name, kw in (
            ("dc", dict(sigma_bulk=sigma_dc, thickness=t)),
            ("f0", dict(sigma_bulk=sigma_bulk, thickness=t,
                        surface_impedance_f0=f0)),
        ):
            sim = _nu_graded_sim(**kw)
            grid_nu = _nu_graded_grid(sim)
            specs = []
            mats_nu = assemble_materials_nu(sim, grid_nu,
                                            sheet_specs=specs)[0]
            if name == "dc":
                sig = np.asarray(mats_nu.sigma)
            else:
                # #677: sheet-free arrays + one emitted spec
                assert float(np.asarray(mats_nu.sigma).max()) == 0.0
                assert len(specs) == 1
                sig = np.asarray(specs[0].sigma_sheet)
            nz = np.argwhere(sig > 0)
            assert len(nz) > 0
            dz_local = float(np.asarray(grid_nu.dz)[int(nz[0][2])])
            got_nu[name] = (float(sig.max()), dz_local)
        assert got_nu["dc"][1] == got_nu["f0"][1]          # same sheet layer
        dz_local = got_nu["dc"][1]
        alg_dc = sigma_dc * t / dz_local                    # x64 algebra
        alg_f0 = 1.0 / (rs0 * dz_local)
        assert abs(alg_f0 - alg_dc) / alg_dc < 1e-9, (alg_dc, alg_f0)
        assert abs(got_nu["dc"][0] - alg_dc) / alg_dc < 1e-6
        assert abs(got_nu["f0"][0] - alg_f0) / alg_f0 < 1e-6


def test_leontovich_algebraic_realization_o2():
    """O2: per NODE, sigma_eff * Rs0 * d_dual == 1 within rel err < 1e-6
    (float32), on the uniform grid, on a matched-node NU fixture, AND on a
    sheet sitting ON a grading transition.

    ``d_dual`` is the E-node DUAL spacing ``1/inv_d_e``, i.e.
    ``(d[k-1]+d[k])/2`` — the length the node's sigma actually acts over,
    because the NU E update divides the curl at node k by
    ``inv_d_e[k] = 2/(d[k-1]+d[k])``. It equals the primal cell ``d[k]``
    wherever the two adjacent cells are equal, which is why the first two
    blocks below cannot tell the two normalizations apart; the third block
    puts the sheet on a 0.5/1.5 mm transition, where they differ by 1.5x,
    and asserts BOTH that the dual product is 1 and that the primal product
    is not — so this gate can no longer pass while the fold divides by the
    primal cell (it did before the #669 review: the sheet then realized
    Rs*d[k]/dual, measured as a 1.2021 / 0.6214 attenuation ratio against
    the matched-mesh case in tests/unit/materials/test_thin_conductor_nu_dual_spacing.py).
    """
    from rfx.core.yee import init_materials
    from rfx.materials.thin_conductor import leontovich_rs
    from rfx.nonuniform import e_node_dual_spacings
    from rfx.runners.nonuniform import assemble_materials_nu

    f0, sigma_bulk = 10e9, 1e4
    rs0 = float(leontovich_rs(f0, sigma_bulk))

    # uniform (primal == dual by construction)
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))
    shape = Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001))
    tc = ThinConductor(shape=shape, sigma_bulk=sigma_bulk, thickness=35e-6,
                       surface_impedance_f0=f0)
    specs = []
    mats, _ = apply_thin_conductor(grid, tc, init_materials(grid.shape), None,
                                   sheet_specs=specs)
    # #677: arrays stay sheet-free; the realized per-node sheet conductivity
    # lives on the emitted spec.
    assert float(np.asarray(mats.sigma).max()) == 0.0
    assert len(specs) == 1
    sig = np.asarray(specs[0].sigma_sheet)
    cells = sig[sig > 0]
    assert cells.size > 0
    np.testing.assert_allclose(cells * rs0 * grid.dx, 1.0, rtol=1e-6)

    # graded NU (dz = [0.5mm]*8 + [1.5mm]*8): matched node, then transition
    for zc, expect_split in ((8.0e-3, False), (4.0e-3, True)):
        sim = _nu_graded_sim(zc=zc, sigma_bulk=sigma_bulk, thickness=35e-6,
                             surface_impedance_f0=f0)
        grid_nu = _nu_graded_grid(sim)
        specs = []
        mats_nu = assemble_materials_nu(sim, grid_nu, sheet_specs=specs)[0]
        assert float(np.asarray(mats_nu.sigma).max()) == 0.0  # sheet-free
        assert len(specs) == 1
        sig = np.asarray(specs[0].sigma_sheet)
        nz = np.argwhere(sig > 0)
        assert len(nz) > 0
        primal = np.asarray(grid_nu.dz)
        dual = np.asarray(e_node_dual_spacings(grid_nu.dz))
        # the dual spacing must be the reciprocal of the metric the E update
        # actually uses — pinned so the two cannot drift apart
        np.testing.assert_allclose(
            dual * np.asarray(grid_nu.inv_dz), 1.0, rtol=1e-6)
        ks = sorted({int(i[2]) for i in nz})
        assert len(ks) == 1, ks
        k = ks[0]
        if expect_split:
            assert abs(dual[k] / primal[k] - 1.0) > 0.3, (
                f"transition fixture is not discriminating: dual "
                f"{dual[k]:.4e} vs primal {primal[k]:.4e}")
        for idx in nz:
            prod = sig[tuple(idx)] * rs0 * dual[idx[2]]
            assert abs(prod - 1.0) < 1e-6, (idx, prod)
        if expect_split:
            prod_primal = float(sig[tuple(nz[0])]) * rs0 * primal[k]
            assert abs(prod_primal - 1.0) > 0.3, (
                f"fold normalized by the PRIMAL cell at a grading "
                f"transition (product {prod_primal:.5f})")


def test_leontovich_routing_no_pec_cells_both_lanes():
    """Routing table: a METAL (sigma_bulk >= 1e6) with surface_impedance_f0
    set contributes NO pec_mask cells and flows down the lossy sigma fold —
    on both lanes. Folded sigma_eff can legally exceed the 1e6 routing
    threshold (spec-level predicate, never applied to assembled arrays)."""
    import warnings
    from rfx.api import Simulation
    from rfx.runners.nonuniform import assemble_materials_nu

    # uniform lane
    sim = Simulation(freq_max=10e9, domain=(6e-3, 6e-3, 3e-3), dx=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((1e-3, 1e-3, 1e-3), (5e-3, 5e-3, 1e-3)),
            sigma_bulk=5.8e7, surface_impedance_f0=10e9)
    specs_u = []
    mats, _, _, pec_mask, *_ = sim._assemble_materials(
        sim._build_grid(), sheet_specs=specs_u)
    assert pec_mask is None or int(np.asarray(pec_mask).sum()) == 0
    # #677: metal-with-f0 flows down the SHEET route (spec emitted, arrays
    # sheet-free), never the PEC route — sigma_sheet can legally exceed the
    # 1e6 routing threshold (spec-level predicate only).
    assert float(np.asarray(mats.sigma).max()) == 0.0
    assert len(specs_u) == 1
    assert (np.asarray(specs_u[0].sigma_sheet) > 0).sum() > 0

    # NU lane
    sim_nu = _nu_graded_sim(sigma_bulk=5.8e7, surface_impedance_f0=10e9)
    grid_nu = _nu_graded_grid(sim_nu)
    specs_nu = []
    mats_nu, _, _, pec_nu = assemble_materials_nu(
        sim_nu, grid_nu, sheet_specs=specs_nu)
    assert pec_nu is None or int(np.asarray(pec_nu).sum()) == 0
    assert float(np.asarray(mats_nu.sigma).max()) == 0.0
    assert len(specs_nu) == 1
    assert (np.asarray(specs_nu[0].sigma_sheet) > 0).sum() > 0


def test_leontovich_ad_gates_o5():
    """O5 (x64 scoped per-test, never module-level): closed-form gradients
    through the fold — d(sigma_eff)/d(f0) == -sigma_eff/(2*f0) and
    d(sigma_eff)/d(sigma_bulk) == +sigma_eff/(2*sigma_bulk), rel err < 1e-5;
    plus the is_pec-order pin: a traced sigma_bulk with f0 set must not
    raise on is_pec evaluation (None-check short-circuits the float
    compare)."""
    import jax
    from tests._x64_compat import enable_x64
    from rfx.core.yee import init_materials

    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))
    shape = Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001))

    def sig_sum(f0, sb):
        tc = ThinConductor(shape=shape, sigma_bulk=sb, thickness=35e-6,
                           surface_impedance_f0=f0)
        # is_pec-order pin: evaluating the routing predicate on a TRACED
        # sigma_bulk must not raise while f0 is set.
        assert tc.is_pec is False
        specs = []
        mats, _ = apply_thin_conductor(grid, tc,
                                       init_materials(grid.shape), None,
                                       sheet_specs=specs)
        # #677: gradients flow through the emitted spec's sigma_sheet (the
        # realized sheet quantity); the arrays are sheet-free by design.
        return jnp.sum(specs[0].sigma_sheet)

    with enable_x64():
        f0, sb = jnp.float64(10e9), jnp.float64(1e4)
        s0 = float(sig_sum(f0, sb))
        g_f0 = float(jax.grad(sig_sum, argnums=0)(f0, sb))
        g_sb = float(jax.grad(sig_sum, argnums=1)(f0, sb))
        assert abs(g_f0 - (-s0 / (2 * 10e9))) / abs(s0 / (2 * 10e9)) < 1e-5
        assert abs(g_sb - (s0 / (2 * 1e4))) / (s0 / (2 * 1e4)) < 1e-5


def test_leontovich_add_time_validation():
    """Concrete-value ValueErrors at add time, plus the two STRUCTURAL
    refusals that survived #674 (which lifted the Box-only restriction): a
    shape that cannot rasterize itself, and one that cannot say where its
    normal is. The rasterized refusals (a body with HEIGHT, a sheet that
    folds zero cells) need the grid and so raise at build time — NU
    defensive mirror below; full non-Box contract in
    tests/unit/materials/test_thin_conductor_nonbox_sheet.py."""
    import pytest
    from rfx.api import Simulation
    from rfx.geometry.csg import Sphere
    from rfx.runners.nonuniform import assemble_materials_nu

    sim = Simulation(freq_max=10e9, domain=(6e-3, 6e-3, 3e-3), dx=1e-3)
    sheet = Box((1e-3, 1e-3, 1e-3), (5e-3, 5e-3, 1e-3))
    with pytest.raises(ValueError, match="positive frequency"):
        sim.add_thin_conductor(sheet, surface_impedance_f0=0.0)
    with pytest.raises(ValueError, match="sigma_bulk"):
        sim.add_thin_conductor(sheet, sigma_bulk=-1.0,
                               surface_impedance_f0=1e9)

    class _NoMask:
        def bounding_box(self):
            return (0.0, 0.0, 1e-3), (5e-3, 5e-3, 1e-3)

    class _NoBounds:
        def mask_on_coords(self, x, y, z):
            return sheet.mask_on_coords(x, y, z)

    with pytest.raises(ValueError, match="mask_on_coords"):
        sim.add_thin_conductor(_NoMask(), surface_impedance_f0=1e9)
    with pytest.raises(ValueError, match="bounding box"):
        sim.add_thin_conductor(_NoBounds(), surface_impedance_f0=1e9)

    # NU lane defensive mirror: bypass add_thin_conductor entirely. A sphere
    # is a legal shape CLASS now, so it clears the add-time checks and is
    # refused where the grid can see it is not a sheet.
    sim_nu = _nu_graded_sim(sigma_bulk=1e4, thickness=35e-6)
    sim_nu._thin_conductors[0] = ThinConductor(
        shape=Sphere((6e-3, 6e-3, 8e-3), 1e-3), sigma_bulk=1e4,
        thickness=35e-6, surface_impedance_f0=10e9)
    grid_nu = _nu_graded_grid(sim_nu)
    with pytest.raises(ValueError, match="cell layers along its normal"):
        assemble_materials_nu(sim_nu, grid_nu)
