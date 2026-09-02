"""The node-thin surface-impedance (Leontovich, ``surface_impedance_f0``) sheet:
operator, identities, lane fences, stacked-layer gap veto, non-Box shapes.

One file for the #677 sheet realization (tier 3b of the 2026-09 test-corpus
reorganisation, see ``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``).
Sections, each formerly its own file:

1. **Default-off identity and vmap parity (issue #669, retargeted by #677)** —
   was ``test_leontovich_sheet_identity.py``. O6: ``surface_impedance_f0``
   unset must be BYTE-IDENTICAL to the pre-#669 behaviour (SHA-256 over
   dtype + shape + raw bytes of the assembled ``eps_r`` / ``sigma`` /
   ``mu_r`` / ``pec_mask`` for a PEC-metal and a DC-lossy thin-conductor
   fixture, kwarg absent vs ``None``) with a NEGATIVE CONTROL: f0 set on the
   same fixture leaves the sigma digest unchanged (sheet-free assembly),
   registers exactly one live ``SheetImpedanceSpec``, de-PECs the sheet and
   no longer overwrites ``eps_r``. O7: one f0-mode case through the
   ``vmap_sweep`` batched material build vs the serial assembly.
2. **#677 node-thin sheet operator — unit + limit gates** — was
   ``test_sheet_impedance_operator.py``. Design B (exponential stepping):
   ``E^{n+1} = A*E^n + B*curlH`` with ``A = exp(-x2)``,
   ``B = -expm1(-x2)/sigma_tot``, ``x2 = sigma_tot*dt/(eps0*eps_r)``,
   ``sigma_tot = sigma_bg + G/d_dual``. Gates: G3a Rs0 = 1e12 recovers the
   sheet-absent fields to rtol 1e-5 (x64-scoped); G3b Rs0 = 1e-6 matches the
   PEC-mask resonance to <= 1e-4 relative; G3c per masked edge
   ``x2 * Rs0 * eps0 * eps_r * d_dual / dt == 1`` to rtol 1e-6 on uniform AND
   on the 0.5/1.5 mm NU grading-transition node (primal-spacing product
   asserted != 1); G4 footprint identity with ``apply_pec_mask``'s tangential
   edge masks (uniform, NU, patterned); G5 default-off run byte identity;
   G8 NU reference-strip negative control; G9 this module's share of the
   lane fences (GPU fast path, both distributed runners, UPML, dispersive
   overlap, crossing normals).
3. **#677 G9 lane fences pinned THROUGH each lane's entry point** — was
   ``test_sheet_lane_fences.py``. A lane that ignores the operator ctx does
   not produce a wrong sheet — it produces NO sheet, silently (#369 class).
   Every test enters through the lane's PUBLIC entry point with a live f0
   sheet and ``_fence`` asserts both the lane-naming message and the raising
   ``(module, function)`` frame. ``FENCE_REGISTRY`` is the single inventory
   of every #677 fence; ``test_every_fence_in_the_source_is_pinned``
   re-derives the inventory from the ``rfx/`` source by AST and fails on any
   unregistered or stale fence.
4. **#690 two same-normal f0 sheets on ADJACENT layers must not load the
   gap** — was ``test_sheet_stacked_adjacent_gap.py``. ORACLE 1
   (hand-enumerated edge counts: ``|mask_ex| = |mask_ey| = |F|*n``,
   ``|mask_ez| = 0``); ORACLE 2 (field witness: the vacuum gap edge rings,
   peak > 1e-5 against a 3.9e6 separation); negative controls; and the
   veto's exception for a LENGTH-1 axis (2-D ``mode="2d_tmz"`` sheets keep
   their out-of-plane component, shadow like PEC to 2 %), but NOT for a
   periodic axis.
5. **Surface-impedance sheets on NON-Box shapes (issue #674)** — was
   ``test_thin_conductor_nonbox_sheet.py``: O674-1 bit-identical fold for a
   Box and an independently implemented mask shape (uniform and NU on a
   grading step); O674-2 patterned sheet folds on occupied cells only;
   O674-3 bodies with height and zero-cell sheets fail LOUD on both lanes;
   O674-4 the ``thin_conductor_graded_node`` advisory follows a non-Box
   sheet; O674-5 design-IR round-trip / refusal; O674-6 the real CAD
   ``MeshShape`` path; O674-7 the occupancy guard reads CONCRETE masks only;
   O674-8 the batched build; plus the #671 transition-node oracle re-run
   with a non-Box sheet (``slow_physics``).

Every assertion, tolerance, fixture value, marker and parametrisation of the
absorbed files is kept verbatim (the identical ``_sha`` helper is defined
once; the cross-file imports between the absorbed files became in-module
references).
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import os
import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from rfx import Box, DebyePole, GaussianPulse, Simulation
from rfx.boundaries.pec import apply_pec_mask, tangential_edge_masks
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import EPS_0, init_state, init_materials
from rfx.geometry.csg import Cylinder, Sphere
from rfx.materials.thin_conductor import (
    SheetImpedanceSpec,
    ThinConductor,
    apply_thin_conductor,
    build_sheet_impedance_ctx,
    leontovich_rs,
    refuse_f0_sheets,
    sheet_update_coeffs,
)
from rfx.runners.nonuniform import assemble_materials_nu, build_nonuniform_grid

from tests.unit.materials.test_thin_conductor_nu_dual_spacing import INVARIANCE_CASES


# ===========================================================================
# formerly tests/unit/materials/test_sheet_impedance.py
# ===========================================================================

def _sha(*arrays) -> str:
    """SHA-256 over the raw bytes of *arrays*, dtype and shape included."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


_SHEET = (1e-3, 1e-3, 1e-3, 5e-3, 5e-3)  # x0 y0 z x1 y1 (zero-thickness in z)


def _fixture_sim(kind: str, **tc_kwargs):
    """Committed fixture: 6x6x3 mm box at dx = 1 mm with one thin sheet.

    kind='pec'  -> metal defaults (sigma_bulk = 5.8e7, PEC routing)
    kind='dc'   -> sub-threshold lossy DC fold (sigma_bulk = 1e4)
    """
    sim = Simulation(freq_max=10e9, domain=(6e-3, 6e-3, 3e-3), dx=1e-3)
    x0, y0, z, x1, y1 = _SHEET
    base = dict(sigma_bulk=5.8e7) if kind == "pec" else dict(sigma_bulk=1e4)
    base.update(tc_kwargs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim.add_thin_conductor(Box((x0, y0, z), (x1, y1, z)), **base)
    return sim, [str(w.message) for w in rec]


def _digests(sim):
    specs = []
    mats, _, _, pec_mask, *_ = sim._assemble_materials(
        sim._build_grid(), sheet_specs=specs)
    if pec_mask is None:
        pec_mask = np.zeros((0,), dtype=np.bool_)
    return {
        "eps_r": _sha(mats.eps_r),
        "sigma": _sha(mats.sigma),
        "mu_r": _sha(mats.mu_r),
        "pec_mask": _sha(pec_mask),
    }, mats, pec_mask, specs


def test_default_off_identity_and_negative_control_o6():
    for kind in ("pec", "dc"):
        sim_absent, warns_absent = _fixture_sim(kind)
        sim_default, warns_default = _fixture_sim(
            kind, surface_impedance_f0=None)
        d_absent, _, _, specs_absent = _digests(sim_absent)
        d_default, _, _, _ = _digests(sim_default)
        assert specs_absent == []   # no phantom sheet specs off-mode
        assert d_absent == d_default, (
            f"{kind}: passing surface_impedance_f0=None changed the "
            f"assembled arrays — default is not byte-off")

        # default path emits no NEW warnings: identical warning sets with
        # the kwarg absent vs default-passed, and the f0-mode add-time
        # warning ("Leontovich surface-resistance sheet with Rs0 = ...")
        # never fires. (The #504 PEC warning legitimately MENTIONS the
        # escape hatch — that is contract-mandated wording, not a new
        # warning event.)
        assert warns_absent == warns_default
        for w in warns_absent:
            assert "Leontovich surface-resistance sheet" not in w, w

        # NEGATIVE CONTROL (#677 retarget): the harness must be able to
        # move. f0 set on the same fixture now leaves the assembled ARRAYS
        # alone (sigma digest UNCHANGED for the pec fixture, whose sheet
        # was previously in pec_mask, and sheet-free for the dc fixture)
        # and instead registers exactly one live SheetImpedanceSpec; the
        # sheet contributes zero pec_mask bits and eps_r is untouched.
        sim_on, warns_on = _fixture_sim(kind, surface_impedance_f0=10e9)
        d_on, mats_on, pec_on, specs_on = _digests(sim_on)
        assert d_on["sigma"] == _sha(jnp.zeros_like(mats_on.sigma)), (
            f"{kind}: f0 mode wrote into materials.sigma — the #677 "
            f"node-thin realization must not fold the sheet into arrays")
        assert d_on["eps_r"] == _sha(jnp.ones_like(mats_on.eps_r)), (
            f"{kind}: f0 mode overwrote eps_r (removed by #677)")
        assert int(np.asarray(pec_on).sum()) == 0, (
            f"{kind}: f0-mode sheet still contributed pec_mask bits")
        assert any("Leontovich" in w for w in warns_on)

        # liveness: the emitted spec carries the sheet — nonzero
        # sigma_sheet exactly on the sheet plane
        assert len(specs_on) == 1, specs_on
        sig = np.asarray(specs_on[0].sigma_sheet)
        grid = sim_on._build_grid()
        k = int(round(_SHEET[2] / grid.dx)) + grid.pad_z_lo
        plane = sig[:, :, k]
        assert plane.max() > 0.0, f"{kind}: sheet plane has no sigma_sheet"
        assert (np.asarray(specs_on[0].mask).any(axis=(0, 1)).nonzero()[0]
                == np.array([k])).all(), f"{kind}: sheet mask off-plane"


def test_vmap_parity_o7():
    """Batched (vmap_sweep) vs serial builds of an f0-mode sheet: sigma
    slices exactly equal — #677 retarget: both builds are SHEET-FREE (the
    sheet is an operator ctx now), the serial build registers the spec,
    and a sim carrying an f0 sheet is fast-path INELIGIBLE (sequential
    fallback applies the ctx per swept value)."""
    from rfx.vmap_sweep import _build_batched_materials

    def make(eps_val):
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                         boundary="cpml", cpml_layers=6, dx=0.002)
        sim.add_material("substrate", eps_r=eps_val)
        sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(
                Box((0.0, 0.008, 0.008), (0.02, 0.012, 0.008)),
                sigma_bulk=5.8e7, surface_impedance_f0=3e9)
        sim.add_source((0.01, 0.01, 0.01), "ez",
                       waveform=GaussianPulse(f0=3e9),
                       amplitude_kind="field")
        return sim

    from rfx.vmap_sweep import _build_full_scan_fn

    eps_values = np.array([2.0, 6.0])
    sim = make(4.0)
    grid = sim._build_grid()
    base, *_ = sim._assemble_materials(grid)
    batched = _build_batched_materials(
        sim, grid, base, "substrate.eps_r", jnp.asarray(eps_values))

    assert batched.sigma.shape[0] == 2
    for idx, eps_val in enumerate(eps_values):
        specs = []
        serial, *_ = make(float(eps_val))._assemble_materials(
            grid, sheet_specs=specs)
        assert np.array_equal(np.asarray(batched.sigma[idx]),
                              np.asarray(serial.sigma)), (
            f"sigma mismatch at eps_r={eps_val}")
        # #677: the arrays are sheet-free on BOTH builds; the sheet lives
        # on the serial build's emitted spec
        assert float(np.asarray(serial.sigma).max()) == 0.0
        assert len(specs) == 1
        assert float(np.asarray(specs[0].sigma_sheet).max()) > 0.0

    # and the fast path refuses to run a sheet-bearing sim silently:
    # _build_full_scan_fn returns (None, None) -> sequential fallback,
    # which applies the operator ctx per value via Simulation.run().
    run_one_fn, dft_names = _build_full_scan_fn(
        sim, grid, base, 16, debye_spec=None, lorentz_spec=None,
        pec_mask=None)
    assert run_one_fn is None and dft_names is None


# ===========================================================================
# formerly tests/unit/materials/test_sheet_impedance.py
# ===========================================================================

F0 = 10e9
MU_0 = 4e-7 * np.pi


def _sigma_bulk_for_rs0(rs0, f0=F0):
    """Invert Rs0 = sqrt(pi*f0*mu0/sigma_bulk)."""
    return float(np.pi * f0 * MU_0 / rs0 ** 2)


def _cavity(precision=None, **tc_kwargs):
    kw = {} if precision is None else {"precision": precision}
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     boundary="pec", **kw)
    if tc_kwargs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(
                Box((4e-3, 4e-3, 10e-3), (16e-3, 16e-3, 10e-3)), **tc_kwargs)
    sim.add_source((10e-3, 10e-3, 5e-3), "ex",
                   waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    sim.add_probe((10e-3, 10e-3, 15e-3), "ex")
    return sim


# ---------------------------------------------------------------------------
# G3a — free-space identity (stencil-drift falsifier)
# ---------------------------------------------------------------------------

def test_g3a_huge_rs0_recovers_sheet_absent_fields():
    from tests._x64_compat import enable_x64

    with enable_x64():
        r0 = _cavity(precision="float64").run(n_steps=400,
                                              skip_preflight=True)
        ts0 = np.asarray(r0.time_series)
        sb = _sigma_bulk_for_rs0(1e12)
        r1 = _cavity(precision="float64", sigma_bulk=sb,
                     surface_impedance_f0=F0).run(n_steps=400,
                                                  skip_preflight=True)
        ts1 = np.asarray(r1.time_series)
    rel = np.abs(ts1 - ts0).max() / np.abs(ts0).max()
    assert rel < 1e-5, rel


# ---------------------------------------------------------------------------
# G3b — PEC recovery of resonance positions at tiny Rs0
# ---------------------------------------------------------------------------

def _peak_freq(ts, dt, band=(4e9, 12e9)):
    """Largest-|X| rFFT bin in ``band`` with parabolic interpolation (same
    chain both runs, so truncation bias cancels in the comparison; the band
    excludes the near-DC window-leakage lobe)."""
    w = np.hanning(len(ts))
    X = np.abs(np.fft.rfft(ts * w))
    freqs = np.fft.rfftfreq(len(ts), dt)
    sel = np.nonzero((freqs >= band[0]) & (freqs <= band[1]))[0]
    k = int(sel[np.argmax(X[sel])])
    a, b, c = np.log(X[k - 1]), np.log(X[k]), np.log(X[k + 1])
    delta = 0.5 * (a - c) / (a - 2 * b + c)
    return float(freqs[k] + delta * (freqs[1] - freqs[0]))


def test_g3b_tiny_rs0_matches_pec_resonance():
    n = 1500
    sim_pec = _cavity(sigma_bulk=5.8e7)          # PEC routing (f0 absent)
    r_pec = sim_pec.run(n_steps=n, skip_preflight=True)
    sb = _sigma_bulk_for_rs0(1e-6)
    sim_f0 = _cavity(sigma_bulk=sb, surface_impedance_f0=F0)
    r_f0 = sim_f0.run(n_steps=n, skip_preflight=True)
    dt = sim_pec._build_grid().dt
    f_pec = _peak_freq(np.asarray(r_pec.time_series)[:, 0], dt)
    f_f0 = _peak_freq(np.asarray(r_f0.time_series)[:, 0], dt)
    assert abs(f_f0 - f_pec) / f_pec <= 1e-4, (f_pec, f_f0)


# ---------------------------------------------------------------------------
# G3c — coefficient algebra at masked edges (uniform + NU transition node)
# ---------------------------------------------------------------------------

def test_g3c_x2_algebra_uniform_and_nu_transition():
    from rfx.grid import Grid
    rs0 = float(leontovich_rs(F0, 1e4))

    # uniform
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))
    shape = Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001))
    tc = ThinConductor(shape=shape, sigma_bulk=1e4, thickness=35e-6,
                       surface_impedance_f0=F0)
    specs = []
    mats, _ = apply_thin_conductor(grid, tc, init_materials(grid.shape),
                                   None, sheet_specs=specs)
    spec = specs[0]
    m = np.asarray(spec.mask)
    sigma_tot = np.asarray(mats.sigma) + np.asarray(spec.sigma_sheet)
    x2 = sigma_tot * grid.dt / (EPS_0 * np.asarray(mats.eps_r))
    prod = x2[m] * rs0 * EPS_0 * np.asarray(mats.eps_r)[m] * grid.dx / grid.dt
    np.testing.assert_allclose(prod, 1.0, rtol=1e-6)

    # NU 0.5/1.5 mm grading-transition node
    from rfx.runners.nonuniform import assemble_materials_nu
    from rfx.nonuniform import e_node_dual_spacings
    from tests.unit.materials.test_thin_conductor import _nu_graded_grid, _nu_graded_sim
    sim = _nu_graded_sim(zc=4.0e-3, sigma_bulk=1e4, thickness=35e-6,
                         surface_impedance_f0=F0)
    grid_nu = _nu_graded_grid(sim)
    specs = []
    mats_nu = assemble_materials_nu(sim, grid_nu, sheet_specs=specs)[0]
    spec = specs[0]
    m = np.asarray(spec.mask)
    ks = sorted({int(k) for k in np.argwhere(m)[:, 2]})
    assert len(ks) == 1
    k = ks[0]
    dual = float(np.asarray(e_node_dual_spacings(grid_nu.dz))[k])
    primal = float(np.asarray(grid_nu.dz)[k])
    assert abs(primal / dual - 1.0) > 0.3   # primal-product tooth kept
    sigma_tot = np.asarray(mats_nu.sigma) + np.asarray(spec.sigma_sheet)
    x2 = sigma_tot * grid_nu.dt / (EPS_0 * np.asarray(mats_nu.eps_r))
    prod = (x2[m] * rs0 * EPS_0 * np.asarray(mats_nu.eps_r)[m] * dual
            / grid_nu.dt)
    np.testing.assert_allclose(prod, 1.0, rtol=1e-6)
    prod_primal = (x2[m] * rs0 * EPS_0 * np.asarray(mats_nu.eps_r)[m]
                   * primal / grid_nu.dt)
    assert np.all(np.abs(prod_primal - 1.0) > 0.3)


# ---------------------------------------------------------------------------
# G4 — footprint identity with apply_pec_mask
# ---------------------------------------------------------------------------

def _masks_for_fixtures():
    """Cell masks: uniform Box sheet, NU graded sheet, patterned #674."""
    from rfx.grid import Grid
    out = []
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))
    out.append(("uniform-box", Box((0.005, 0.005, 0.001),
                                   (0.015, 0.015, 0.001)).mask(grid)))
    from tests.unit.materials.test_thin_conductor import _nu_graded_grid, _nu_graded_sim
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    sim = _nu_graded_sim(zc=4.0e-3, sigma_bulk=1e4, thickness=35e-6,
                         surface_impedance_f0=F0)
    grid_nu = _nu_graded_grid(sim)
    coords = coords_from_nonuniform_grid(grid_nu)
    out.append(("nu-graded", sim._thin_conductors[0].shape.mask_on_coords(
        coords.x, coords.y, coords.z)))
    holed = _planar_sheet(U_Z, U_FOOT, hole=U_HOLE)
    sim_u = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.002), dx=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim_u.add_thin_conductor(holed, sigma_bulk=1e4, thickness=35e-6,
                                 surface_impedance_f0=F0)
    specs_u = []
    sim_u._assemble_materials(sim_u._build_grid(), sheet_specs=specs_u)
    out.append(("patterned", specs_u[0].mask))
    return out


def test_g4_footprint_identity_with_pec_mask():
    for name, mask in _masks_for_fixtures():
        mask = jnp.asarray(mask)
        assert bool(jnp.any(mask)), name
        mex, mey, mez = tangential_edge_masks(mask)
        # exact equality with what apply_pec_mask zeroes: run it on a
        # state of ones — zeros land exactly on the tangential edge sets
        st = init_state(mask.shape)
        ones = jnp.ones(mask.shape, jnp.float32)
        st = st._replace(ex=ones, ey=ones, ez=ones)
        out = apply_pec_mask(st, mask)
        np.testing.assert_array_equal(np.asarray(out.ex) == 0.0,
                                      np.asarray(mex), err_msg=name)
        np.testing.assert_array_equal(np.asarray(out.ey) == 0.0,
                                      np.asarray(mey), err_msg=name)
        np.testing.assert_array_equal(np.asarray(out.ez) == 0.0,
                                      np.asarray(mez), err_msg=name)
        # one-layer sheet: the normal-component edge set is all-False
        one_layer_axes = [
            a for a in range(3)
            if int(np.count_nonzero(np.asarray(jnp.any(
                mask, axis=tuple(b for b in range(3) if b != a))))) == 1]
        for a in one_layer_axes:
            assert not bool(jnp.any((mex, mey, mez)[a])), (name, a)
        # and the assembled ctx carries exactly these masks (no PEC given)
        from rfx.materials.thin_conductor import SheetImpedanceSpec
        ctx = build_sheet_impedance_ctx([SheetImpedanceSpec(
            mask=mask, normal_axis=(one_layer_axes[0]
                                    if one_layer_axes else 2),
            g_sheet=1.0, sigma_sheet=jnp.where(mask, 1.0, 0.0))])
        np.testing.assert_array_equal(np.asarray(ctx.mask_ex),
                                      np.asarray(mex))
        np.testing.assert_array_equal(np.asarray(ctx.mask_ey),
                                      np.asarray(mey))
        np.testing.assert_array_equal(np.asarray(ctx.mask_ez),
                                      np.asarray(mez))


def test_g4_pec_owned_edges_are_excluded_from_the_ctx():
    from rfx.materials.thin_conductor import SheetImpedanceSpec
    rng = np.random.default_rng(677)
    mask = jnp.asarray(rng.random((8, 8, 8)) < 0.3)
    pec = jnp.asarray(rng.random((8, 8, 8)) < 0.3)
    ctx = build_sheet_impedance_ctx(
        [SheetImpedanceSpec(mask=mask, normal_axis=2, g_sheet=1.0,
                            sigma_sheet=jnp.where(mask, 1.0, 0.0))],
        pec_mask=pec)
    pex, pey, pez = tangential_edge_masks(pec)
    assert not bool(jnp.any(ctx.mask_ex & pex))
    assert not bool(jnp.any(ctx.mask_ey & pey))
    assert not bool(jnp.any(ctx.mask_ez & pez))


# ---------------------------------------------------------------------------
# G5 — default-off run byte identity (kwarg absent vs None)
# ---------------------------------------------------------------------------

def test_g5_default_off_run_byte_identity():
    import hashlib

    def digest(ts):
        h = hashlib.sha256()
        a = np.ascontiguousarray(np.asarray(ts))
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
        return h.hexdigest()

    for kind_kw in (dict(sigma_bulk=5.8e7),                 # PEC routing
                    dict(sigma_bulk=1e4, thickness=35e-6)):  # DC fold
        r_absent = _cavity(**kind_kw).run(n_steps=200, skip_preflight=True)
        r_none = _cavity(surface_impedance_f0=None, **kind_kw).run(
            n_steps=200, skip_preflight=True)
        assert digest(r_absent.time_series) == digest(r_none.time_series)


# ---------------------------------------------------------------------------
# G8 — NU reference-strip negative control
# ---------------------------------------------------------------------------

def _nu_probe_sim(with_sheet: bool):
    dz = [0.5e-3] * 16
    sim = Simulation(freq_max=10e9, domain=(8e-3, 8e-3, 0), dx=0.5e-3,
                     dz_profile=dz, boundary="cpml", cpml_layers=6)
    if with_sheet:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(
                Box((2e-3, 2e-3, 4e-3), (6e-3, 6e-3, 4e-3)),
                sigma_bulk=1e4, surface_impedance_f0=F0)
    sim.add_source((4e-3, 4e-3, 2e-3), "ex",
                   waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    sim.add_probe((4e-3, 4e-3, 6e-3), "ex")
    return sim


def test_g8_reference_strip_negative_control():
    from rfx.runners.nonuniform import run_nonuniform_path

    n = 300
    # (a) sheet-bearing device: the reference-configured run DIFFERS with
    # vs without the explicit ctx strip — i.e. sigma_override alone does
    # NOT strip the sheet, the strip parameter does.
    sim = _nu_probe_sim(with_sheet=True)
    ts = {}
    for strip in (False, True):
        r = run_nonuniform_path(sim, n_steps=n,
                                strip_sheet_impedance=strip)
        ts[strip] = np.asarray(r.time_series)
    assert not np.array_equal(ts[False], ts[True]), (
        "stripping the sheet ctx changed nothing — the reference would "
        "still carry the sheet")

    # (b) sheet-free control: the strip plumbing is a byte-level no-op.
    sim0 = _nu_probe_sim(with_sheet=False)
    r0 = run_nonuniform_path(sim0, n_steps=n, strip_sheet_impedance=False)
    r1 = run_nonuniform_path(sim0, n_steps=n, strip_sheet_impedance=True)
    assert np.asarray(r0.time_series).tobytes() == \
        np.asarray(r1.time_series).tobytes()


# ---------------------------------------------------------------------------
# G9 — lane fences
# ---------------------------------------------------------------------------

def _sheet_sim(**sim_kw):
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     **sim_kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((4e-3, 4e-3, 10e-3), (16e-3, 16e-3, 10e-3)),
            sigma_bulk=1e4, surface_impedance_f0=F0)
    sim.add_source((10e-3, 10e-3, 5e-3), "ex",
                   waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    return sim


def test_g9_fast_path_excludes_sheets(monkeypatch):
    """GPU baked fast path: pretend the backend is GPU; a sheet-free PEC
    cavity takes the fast path (control — the probe observes it), a
    sheet-bearing one must NOT."""
    import jax
    import rfx.simulation as sim_mod

    class _Boom(RuntimeError):
        pass

    def boom(*a, **k):
        raise _Boom("update_he_fast called")

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(sim_mod, "update_he_fast", boom)

    # control: eligible sim reaches the fast path -> probe fires
    ctrl = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                      boundary="pec")
    ctrl.add_source((10e-3, 10e-3, 5e-3), "ex",
                    waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    with pytest.raises(_Boom):
        ctrl.run(n_steps=8, skip_preflight=True)

    # sheet-bearing sim: _fast_eligible excludes it -> normal path, no Boom
    _sheet_sim(boundary="pec").run(n_steps=8, skip_preflight=True)


def test_g9_distributed_runners_refuse():
    from rfx.runners.distributed import run_distributed as run_v1
    from rfx.runners.distributed_v2 import run_distributed as run_v2

    sim = _sheet_sim(boundary="pec")
    with pytest.raises(ValueError, match="not supported on the distributed"):
        run_v1(sim, n_steps=8)
    with pytest.raises(ValueError, match="not supported on the distributed"):
        run_v2(sim, n_steps=8)


def test_g9_refusal_helper_message_names_the_lane():
    """UNIT test of the helper's message formatting ONLY.

    It proves nothing about whether any lane calls the helper: delete
    ``refuse_f0_sheets(self._thin_conductors, "ADI run()")`` from
    ``rfx/api/_execute.py`` and this test stays green (measured
    2026-08-19) while ``sim.run(solver="adi")`` stops refusing. The
    call-site proofs live in ``tests/unit/materials/test_sheet_impedance.py``.
    """
    tc = ThinConductor(shape=Box((0, 0, 0), (1e-3, 1e-3, 0)),
                       sigma_bulk=1e4, thickness=35e-6,
                       surface_impedance_f0=F0)
    with pytest.raises(ValueError, match="subgridded"):
        refuse_f0_sheets([tc], "subgridded (SBP-SAT) run()")
    with pytest.raises(ValueError, match="ADI"):
        refuse_f0_sheets([tc], "ADI run()")
    refuse_f0_sheets([], "anything")   # no sheets -> no raise


def test_g9_crossing_normals_refuse():
    from rfx.materials.thin_conductor import SheetImpedanceSpec

    m_z = jnp.zeros((6, 6, 6), bool).at[:, :, 3].set(True)
    m_x = jnp.zeros((6, 6, 6), bool).at[3, :, :].set(True)
    specs = [
        SheetImpedanceSpec(mask=m_z, normal_axis=2, g_sheet=1.0,
                           sigma_sheet=jnp.where(m_z, 1.0, 0.0)),
        SheetImpedanceSpec(mask=m_x, normal_axis=0, g_sheet=1.0,
                           sigma_sheet=jnp.where(m_x, 1.0, 0.0)),
    ]
    with pytest.raises(ValueError, match="DIFFERENT normal axes"):
        build_sheet_impedance_ctx(specs)
    # non-overlapping different normals are fine
    m_x2 = jnp.zeros((6, 6, 6), bool).at[3, :, :2].set(True)
    ctx = build_sheet_impedance_ctx([
        specs[0],
        SheetImpedanceSpec(mask=m_x2, normal_axis=0, g_sheet=1.0,
                           sigma_sheet=jnp.where(m_x2, 1.0, 0.0))])
    assert ctx is not None


def test_g9_dispersive_overlap_refuses_uniform_and_nu():
    # uniform
    sim = _sheet_sim(boundary="pec")
    from rfx import DebyePole
    sim.add_material("water", eps_r=4.0,
                     debye_poles=[DebyePole(delta_eps=2.0, tau=8.3e-12)])
    sim.add(Box((2e-3, 2e-3, 2e-3), (6e-3, 6e-3, 6e-3)), material="water")
    with pytest.raises(ValueError, match="dispersive"):
        sim.run(n_steps=8, skip_preflight=True)

    # NU
    dz = [0.5e-3] * 16
    sim_nu = Simulation(freq_max=10e9, domain=(8e-3, 8e-3, 0), dx=0.5e-3,
                        dz_profile=dz, boundary="cpml", cpml_layers=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim_nu.add_thin_conductor(
            Box((2e-3, 2e-3, 4e-3), (6e-3, 6e-3, 4e-3)),
            sigma_bulk=1e4, surface_impedance_f0=F0)
    from rfx import DebyePole
    sim_nu.add_material("water", eps_r=4.0,
                        debye_poles=[DebyePole(delta_eps=2.0, tau=8.3e-12)])
    sim_nu.add(Box((1e-3, 1e-3, 1e-3), (3e-3, 3e-3, 3e-3)),
               material="water")
    sim_nu.add_source((4e-3, 4e-3, 2e-3), "ex",
                      waveform=GaussianPulse(f0=5e9),
                      amplitude_kind="field")
    with pytest.raises(ValueError, match="dispersive"):
        sim_nu.run(n_steps=8, skip_preflight=True)


def test_g9_upml_refuses():
    sim = _sheet_sim(boundary="upml")
    with pytest.raises(ValueError, match="upml"):
        sim.run(n_steps=8, skip_preflight=True)


# ===========================================================================
# formerly tests/unit/materials/test_sheet_impedance.py
# ===========================================================================

F0 = 10e9
_SHEET_BOX = Box((4e-3, 4e-3, 10e-3), (16e-3, 16e-3, 10e-3))


# ---------------------------------------------------------------------------
# Fixtures: one f0 sheet, per-lane geometry
# ---------------------------------------------------------------------------

def _sheet(sim, box=_SHEET_BOX):
    """Register one Leontovich sheet (the add-time advisory is not the
    subject here — the lane refusal is)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(box, sigma_bulk=1e4, surface_impedance_f0=F0)
    return sim


def _cube(**sim_kw):
    """20 mm PEC cube, one z-normal sheet at mid-height, one field source."""
    sim_kw.setdefault("boundary", "pec")
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     **sim_kw)
    _sheet(sim)
    sim.add_source((10e-3, 10e-3, 5e-3), "ex",
                   waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    sim.add_probe((10e-3, 10e-3, 14e-3), "ex")
    return sim


def _nu_cube(**sim_kw):
    """Same fixture on the non-uniform lane (uniform dz_profile — the
    fences under test do not depend on the grading ratio)."""
    sim_kw.setdefault("boundary", "pec")
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.0), dx=1e-3,
                     dz_profile=[1e-3] * 20, **sim_kw)
    _sheet(sim)
    sim.add_source((10e-3, 10e-3, 5e-3), "ex",
                   waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    sim.add_probe((10e-3, 10e-3, 14e-3), "ex")
    return sim


def _design_cube():
    """Port-fed PEC cube for the optimize / gradient-check / topology lanes."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     boundary="pec")
    _sheet(sim)
    sim.add_port((5e-3, 10e-3, 10e-3), "ez")
    sim.add_probe((14e-3, 10e-3, 10e-3), "ez")
    return sim


def _wr90(n_modes=1):
    """Coarse WR-90 two-port with a sheet across the guide."""
    sim = Simulation(
        freq_max=8e9, domain=(0.06, 0.02286, 0.01016), dx=3e-3,
        boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=6)
    _sheet(sim, Box((0.02, 0.0, 0.005), (0.04, 0.02286, 0.005)))
    freqs = jnp.linspace(6e9, 7e9, 3)
    for x, d, name in ((0.009, "+x", "l"), (0.051, "-x", "r")):
        sim.add_waveguide_port(x, direction=d, mode=(1, 0), mode_type="TE",
                               freqs=freqs, f0=6.5e9, bandwidth=0.4,
                               name=name, n_modes=n_modes)
    return sim


def _mixed_probe_fed_msl():
    """Probe-fed MSL board for the mixed (wire + MSL) lane. Ladder geometry
    copied from the working fixture in tests/unit/sparams/test_mixed_port_sparam.py so
    the earlier port-geometry guards pass and the #677 fence is reached."""
    eps_r, h_sub, w_trace, dx = 3.66, 254e-6, 600e-6, 80e-6
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(freq_max=5e9, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("sub", eps_r=eps_r)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, h_sub)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - w_trace / 2, h_sub),
                (lx, y_c + w_trace / 2, h_sub + dx)), material="pec")
    _sheet(sim, Box((3e-3, 1e-3, 5e-4), (5e-3, 2e-3, 5e-4)))
    sim.add_port(position=(2e-3, y_c, 0.0), component="ez", impedance=50.0,
                 extent=h_sub)
    sim.add_msl_port(position=(5.5e-3, y_c, 0.0), width=w_trace, height=h_sub,
                     direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
                     n_probe_offset=10, n_probe_spacing=4)
    return sim


# ---------------------------------------------------------------------------
# The call-site assertion
# ---------------------------------------------------------------------------

def _fence(entry, *, where, match):
    """Run ``entry`` and prove the #677 refusal came FROM ``where``.

    ``where`` is ``(module basename, enclosing function name)`` of the call
    site under test. The frame check is the load-bearing half: matching the
    message alone would also pass if the helper were called from anywhere
    else, which is exactly the hole this file closes.
    """
    with pytest.raises(ValueError, match=match) as ei:
        entry()
    frames = [(os.path.basename(str(t.path)), t.name) for t in ei.traceback]
    assert where in frames, (
        f"the refusal did not pass through {where} — the call site may have "
        f"been dropped and a different fence caught it. frames={frames}")
    return ei


# ---------------------------------------------------------------------------
# refuse_f0_sheets() call sites
# ---------------------------------------------------------------------------

def test_fence_adi_run():
    """Honest scope note for BOTH ADI tests, measured 2026-08-19: with this
    call site deleted the ADI lane still refuses, from a pre-existing guard
    that predates #677 — ``_preflight._validate_adi_configuration`` rejects
    ANY registered thin conductor
    (``solver='adi' does not support thin-conductor corrections yet``). So
    ADI's exposure to a silently sheet-free run was lower than the other
    lanes' even before #677; what these two tests pin is that the #677
    fence is wired and names the right lane, not that it is ADI's only
    net."""
    _fence(lambda: _cube(solver="adi").run(n_steps=4, skip_preflight=True),
           where=("_execute.py", "run"),
           match=r"on the ADI run\(\) lane")


def test_fence_adi_forward():
    """See the scope note on ``test_fence_adi_run``: ADI carries a second,
    older thin-conductor guard behind this fence on both entry points."""
    _fence(lambda: _cube(solver="adi").forward(n_steps=4, skip_preflight=True),
           where=("_execute.py", "_forward_from_materials"),
           match=r"on the ADI forward lane")


def test_fence_subgridded_run():
    def go():
        sim = _cube()
        sim.add_refinement((8e-3, 12e-3), ratio=2)
        sim.run(n_steps=4, skip_preflight=True)
    _fence(go, where=("_execute.py", "run"),
           match=r"on the subgridded \(SBP-SAT\) run\(\) lane")


def test_fence_distributed_multidevice_run():
    """Two device handles select the lane; the fence fires before the runner
    is imported, so this needs no real second device (and therefore no
    process-global ``XLA_FLAGS`` device-count flip at import time)."""
    d = jax.devices()[0]
    _fence(lambda: _cube().run(n_steps=4, skip_preflight=True, devices=[d, d]),
           where=("_execute.py", "run"),
           match=r"on the distributed multi-device run\(\) lane")


def test_fence_distributed_nonuniform_forward():
    def go():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nu_cube().forward(n_steps=4, skip_preflight=True,
                               distributed=True)
    _fence(go, where=("_execute.py", "forward"),
           match=r"on the distributed non-uniform forward\(\) lane")


def test_fence_msl_junction_mixed_s_matrix():
    def go():
        _mixed_probe_fed_msl().compute_mixed_s_matrix(
            freqs=np.linspace(1e9, 4e9, 3), num_periods=1.0,
            skip_preflight=True)
    _fence(go, where=("_sparams.py", "compute_mixed_s_matrix"),
           match=r"on the MSL junction S-parameter lane")


def test_fence_optimize_uniform():
    from rfx.optimize import DesignRegion, optimize

    def go():
        region = DesignRegion(corner_lo=(7e-3, 7e-3, 7e-3),
                              corner_hi=(12e-3, 12e-3, 12e-3),
                              eps_range=(1.0, 4.4))
        optimize(_design_cube(), region,
                 lambda r: -jnp.sum(r.time_series ** 2),
                 n_iters=1, lr=0.01, verbose=False, n_steps=4,
                 skip_preflight=True)
    _fence(go, where=("optimize.py", "optimize"),
           match=r"on the optimize\(\) design lane")


def test_fence_optimize_nonuniform():
    from rfx.optimize import DesignRegion, optimize

    def go():
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.0), dx=1e-3,
                         dz_profile=[1e-3] * 20, boundary="pec")
        _sheet(sim)
        sim.add_port((5e-3, 10e-3, 10e-3), "ez")
        sim.add_probe((14e-3, 10e-3, 10e-3), "ez")
        region = DesignRegion(corner_lo=(7e-3, 7e-3, 7e-3),
                              corner_hi=(12e-3, 12e-3, 12e-3),
                              eps_range=(1.0, 4.4))
        optimize(sim, region, lambda r: -jnp.sum(r.time_series ** 2),
                 n_iters=1, lr=0.01, verbose=False, n_steps=4,
                 skip_preflight=True)
    _fence(go, where=("optimize.py", "optimize"),
           match=r"on the optimize\(\) non-uniform design lane")


def test_fence_gradient_check():
    from rfx.optimize import gradient_check

    def go():
        sim = _design_cube()
        dp = jnp.zeros(sim._build_grid().shape, dtype=jnp.float32)
        gradient_check(sim, dp, lambda r: jnp.sum(r.time_series ** 2),
                       eps=1e-2, n_steps=4)
    _fence(go, where=("optimize.py", "gradient_check"),
           match=r"on the gradient-check lane")


def test_fence_topology_optimize():
    """The fence sits ABOVE ``topology_optimize``'s optional ``import optax``
    precisely so this test binds in an environment without the
    ``[optimization]`` extra — the default CI image."""
    from rfx.topology import TopologyDesignRegion, topology_optimize

    def go():
        region = TopologyDesignRegion(corner_lo=(7e-3, 7e-3, 7e-3),
                                      corner_hi=(12e-3, 12e-3, 12e-3),
                                      material_bg="air", material_fg="fr4",
                                      beta_projection=1.0)
        topology_optimize(_design_cube(), region,
                          lambda r: -jnp.sum(r.time_series ** 2),
                          n_iterations=1, learning_rate=0.05,
                          beta_schedule=[(0, 1.0)], verbose=False,
                          skip_preflight=True)
    _fence(go, where=("topology.py", "topology_optimize"),
           match=r"on the topology-optimization lane")


def test_fence_differentiable_material_fit():
    from rfx.differentiable_material_fit import differentiable_material_fit

    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = Simulation(freq_max=5e9, domain=(0.024, 0.009, 0.009))
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
        sim.add(Box((0.010, 0.0, 0.0), (0.016, 0.009, 0.009)), material="dut")
        _sheet(sim, Box((0.004, 0.002, 0.004), (0.008, 0.007, 0.004)))
        sim.add_port((0.003, 0.0045, 0.0045), "ez",
                     waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
        sim.add_probe((0.020, 0.0045, 0.0045), component="ez")
        sim.add_probe((0.007, 0.0045, 0.0045), component="ez")
        return sim

    def go():
        differentiable_material_fit(
            factory, np.zeros((1, 1, 3), complex),
            np.linspace(2.0e9, 4.0e9, 3), n_debye_poles=1,
            n_iterations=1, learning_rate=0.0, verbose=False)
    # the fence lives in the fit's inner ``forward(p)`` closure
    _fence(go, where=("differentiable_material_fit.py", "forward"),
           match=r"on the differentiable material fit lane")


# ---------------------------------------------------------------------------
# Inline "(#677 v1)" fences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{"subpixel_smoothing": True},
                                {"conformal_pec": True}])
def test_fence_uniform_run_subpixel_or_conformal(kw):
    _fence(lambda: _cube().run(n_steps=4, skip_preflight=True, **kw),
           where=("uniform.py", "run_uniform"),
           match=r"not supported with subpixel_smoothing / conformal_pec")


def test_fence_nonuniform_run_anisotropic_eps():
    """The NU fence is conditioned on ``aniso_eps`` actually being built,
    which needs registered geometry — ``subpixel_smoothing=True`` on a
    geometry-free model leaves the plain isotropic update the sheet
    operator is correct against, and is deliberately NOT refused."""
    def go():
        sim = _nu_cube()
        sim.add_material("diel", eps_r=4.0)
        sim.add(Box((2e-3, 2e-3, 2e-3), (6e-3, 6e-3, 6e-3)), material="diel")
        sim.run(n_steps=4, skip_preflight=True, subpixel_smoothing=True)
    _fence(go, where=("nonuniform.py", "run_nonuniform_path"),
           match=r"anisotropic permittivity")


def test_fence_forward_upml():
    """#679: ``run()``'s UPML refusal lives in ``run_uniform``, which
    ``forward()`` never enters — so the ``forward()`` / ``eps_override``
    channel needs its OWN copy, and before #679 it had none: the same sim
    raised on ``run()`` and silently simulated a sheet over UPML's
    split-field E update on ``forward()``. The frame is the load-bearing
    half here, exactly as for the dispersive pair below."""
    def go():
        _cube(boundary="upml", cpml_layers=6).forward(
            n_steps=4, skip_preflight=True)
    _fence(go, where=("_execute.py", "forward"),
           match=r"boundary='upml' on the uniform forward\(\) / eps_override")


def test_fence_forward_dispersive_overlap():
    """``run()``'s dispersive fence lives in ``run_uniform``; ``forward()``
    carries its OWN copy in ``_execute.forward``, so the message match is
    not enough to tell them apart — the frame is."""
    def go():
        sim = _cube()
        sim.add_material("water", eps_r=4.0,
                         debye_poles=[DebyePole(delta_eps=2.0, tau=8.3e-12)])
        sim.add(Box((2e-3, 2e-3, 2e-3), (6e-3, 6e-3, 6e-3)), material="water")
        sim.forward(n_steps=4, skip_preflight=True)
    _fence(go, where=("_execute.py", "forward"),
           match=r"dispersive \(Debye/Lorentz\) materials")


def test_fence_waveguide_s_matrix_subpixel():
    def go():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _wr90().compute_waveguide_s_matrix(n_steps=4,
                                               subpixel_smoothing=True)
    _fence(go, where=("_sparams.py", "compute_waveguide_s_matrix"),
           match=r"on the waveguide S-matrix lane")


def test_fence_waveguide_s_matrix_multimode():
    def go():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _wr90(n_modes=2).compute_waveguide_s_matrix(n_steps=4)
    _fence(go, where=("_sparams.py", "compute_waveguide_s_matrix"),
           match=r"on the multimode waveguide S-matrix path")


# ---------------------------------------------------------------------------
# vmap sweep: not a refusal — a fallback that must still APPLY the sheet
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vmap_sweep_fallback_still_applies_the_sheet():
    """``_build_full_scan_fn`` returns ``(None, None)`` for a sheet-bearing
    sim (pinned at unit level in test_leontovich_sheet_identity.py) so
    ``vmap_material_sweep`` takes the sequential fallback. Completing is not
    the property that matters — a sheet-free sweep would complete too. The
    witness is that the swept fields DIFFER from the same sweep with the
    sheet removed, i.e. the fallback really applied the operator ctx."""
    from rfx.vmap_sweep import vmap_material_sweep

    def build(with_sheet):
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                         boundary="pec")
        sim.add_material("substrate", eps_r=4.0)
        sim.add(Box((2e-3, 2e-3, 2e-3), (18e-3, 18e-3, 9e-3)),
                material="substrate")
        if with_sheet:
            _sheet(sim, Box((4e-3, 4e-3, 12e-3), (16e-3, 16e-3, 12e-3)))
        sim.add_source((10e-3, 10e-3, 5e-3), "ex",
                       waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
        sim.add_probe((10e-3, 10e-3, 16e-3), "ex")
        return sim

    values = np.array([2.0, 6.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = vmap_material_sweep(build(True), "substrate.eps_r", values,
                                 n_steps=200)
        off = vmap_material_sweep(build(False), "substrate.eps_r", values,
                                  n_steps=200)

    ts_on = np.asarray(on.time_series)
    ts_off = np.asarray(off.time_series)
    assert ts_on.shape == ts_off.shape
    assert np.all(np.isfinite(ts_on)), "sheet-bearing sweep went non-finite"
    scale = np.abs(ts_off).max()
    assert scale > 0.0, "sheet-free control recorded no field — bad fixture"
    rel = np.abs(ts_on - ts_off).max() / scale
    assert rel > 1e-3, (
        f"sheet-on and sheet-free sweeps agree to {rel:.2e} — the sequential "
        f"fallback appears to have simulated a sheet-FREE model")


# ---------------------------------------------------------------------------
# Inventory guard: no fence may exist without a named pinning test
# ---------------------------------------------------------------------------

#: ``(rel path, enclosing function, discriminant)`` -> ``(test module, test)``
#:
#: ``discriminant`` is the lane string for ``refuse_f0_sheets`` sites, and a
#: distinctive message fragment for the inline ``(#677 v1)`` fences.
FENCE_REGISTRY: dict[tuple[str, str, str], tuple[str, str]] = {
    # --- refuse_f0_sheets(...) call sites ---
    ("rfx/api/_execute.py", "_forward_from_materials", "ADI forward"):
        (__name__, "test_fence_adi_forward"),
    ("rfx/api/_execute.py", "forward", "distributed non-uniform forward()"):
        (__name__, "test_fence_distributed_nonuniform_forward"),
    ("rfx/api/_execute.py", "run", "distributed multi-device run()"):
        (__name__, "test_fence_distributed_multidevice_run"),
    ("rfx/api/_execute.py", "run", "ADI run()"):
        (__name__, "test_fence_adi_run"),
    ("rfx/api/_execute.py", "run", "subgridded (SBP-SAT) run()"):
        (__name__, "test_fence_subgridded_run"),
    ("rfx/api/_sparams.py", "compute_mixed_s_matrix",
     "MSL junction S-parameter"):
        (__name__, "test_fence_msl_junction_mixed_s_matrix"),
    ("rfx/differentiable_material_fit.py", "forward",
     "differentiable material fit"):
        (__name__, "test_fence_differentiable_material_fit"),
    ("rfx/optimize.py", "optimize", "optimize() non-uniform design"):
        (__name__, "test_fence_optimize_nonuniform"),
    ("rfx/optimize.py", "optimize", "optimize() design"):
        (__name__, "test_fence_optimize_uniform"),
    ("rfx/optimize.py", "gradient_check", "gradient-check"):
        (__name__, "test_fence_gradient_check"),
    ("rfx/topology.py", "topology_optimize", "topology-optimization"):
        (__name__, "test_fence_topology_optimize"),
    ("rfx/runners/distributed.py", "run_distributed", "distributed (v1) runner"):
        ("tests.unit.materials.test_sheet_impedance",
         "test_g9_distributed_runners_refuse"),
    ("rfx/runners/distributed_v2.py", "run_distributed",
     "distributed (v2) runner"):
        ("tests.unit.materials.test_sheet_impedance",
         "test_g9_distributed_runners_refuse"),

    # --- inline "(#677 v1)" fences ---
    ("rfx/api/_execute.py", "forward",
     "boundary='upml' on the uniform forward()"):
        (__name__, "test_fence_forward_upml"),
    ("rfx/api/_execute.py", "forward", "dispersive (Debye/Lorentz)"):
        (__name__, "test_fence_forward_dispersive_overlap"),
    ("rfx/api/_sparams.py", "compute_waveguide_s_matrix",
     "waveguide S-matrix lane"):
        (__name__, "test_fence_waveguide_s_matrix_subpixel"),
    ("rfx/api/_sparams.py", "compute_waveguide_s_matrix",
     "multimode waveguide S-matrix path"):
        (__name__, "test_fence_waveguide_s_matrix_multimode"),
    ("rfx/runners/uniform.py", "run_uniform", "boundary='upml'"):
        ("tests.unit.materials.test_sheet_impedance", "test_g9_upml_refuses"),
    ("rfx/runners/uniform.py", "run_uniform",
     "subpixel_smoothing / conformal_pec"):
        (__name__, "test_fence_uniform_run_subpixel_or_conformal"),
    ("rfx/runners/uniform.py", "run_uniform", "dispersive (Debye/Lorentz)"):
        ("tests.unit.materials.test_sheet_impedance",
         "test_g9_dispersive_overlap_refuses_uniform_and_nu"),
    ("rfx/runners/nonuniform.py", "run_nonuniform_path",
     "dispersive (Debye/Lorentz)"):
        ("tests.unit.materials.test_sheet_impedance",
         "test_g9_dispersive_overlap_refuses_uniform_and_nu"),
    ("rfx/runners/nonuniform.py", "run_nonuniform_path",
     "anisotropic permittivity"):
        (__name__, "test_fence_nonuniform_run_anisotropic_eps"),

    # --- has_f0_sheets(...) eligibility fences (no raise: a fallback) ---
    ("rfx/vmap_sweep.py", "_build_full_scan_fn", "has_f0_sheets"):
        ("tests.unit.materials.test_sheet_impedance",
         "test_vmap_parity_o7"),
}

#: The GPU baked fast path excludes sheets through a ctx flag rather than a
#: call the AST scan can see (``rfx/simulation.py``'s ``_fast_eligible``
#: reads ``not _ctx["use_sheet_impedance"]``), so it is registered by hand.
_MANUAL_FENCES: dict[tuple[str, str, str], tuple[str, str]] = {
    ("rfx/simulation.py", "run", "use_sheet_impedance"):
        ("tests.unit.materials.test_sheet_impedance",
         "test_g9_fast_path_excludes_sheets"),
}

#: ``has_f0_sheets(...)`` call sites in ``rfx/`` that are NOT lane fences.
#:
#: The AST scan treats every ``has_f0_sheets`` call outside
#: ``thin_conductor.py`` as a lane-eligibility fence, because that is what
#: every such call was when the scan was written. A call that only ASKS the
#: question — to size something, to label an artifact — refuses nothing and
#: has no lane to test, so demanding a FENCE_REGISTRY row for it would mean
#: registering a refusal that does not exist. Each row here names the call
#: site and why it is not a fence; the guard still binds on every other
#: call, and a row that stops matching the source is reported as stale, so
#: this list cannot silently outlive its call site.
_NON_FENCE_F0_QUERIES: dict[tuple[str, str, str], str] = {
    ("rfx/api/__init__.py", "_ad_memory_static_accounting", "has_f0_sheets"):
        "#696 memory ACCOUNTING: adds the sheet operator's three edge masks "
        "+ sigma_sheet to the forward working-set estimate. It changes a "
        "reported number, never whether the lane runs.",
}

_RFX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "rfx")


def _enclosing_function(func_spans, lineno):
    cands = [f for f in func_spans if f[0] <= lineno <= f[1]]
    cands.sort(key=lambda f: f[1] - f[0])
    return cands[0][2] if cands else "<module>"


def _scan_source_fences():
    """Re-derive the fence inventory from the ``rfx/`` source by AST."""
    found: dict[tuple[str, str, str], str] = {}
    for dirpath, _dirs, files in os.walk(_RFX_ROOT):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, os.path.dirname(_RFX_ROOT))
            rel = rel.replace(os.sep, "/")
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            if "677" not in src and "f0_sheets" not in src:
                continue
            tree = ast.parse(src, filename=path)
            spans = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for node in ast.walk(tree):
                # refuse_f0_sheets(conductors, "<lane>")
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id.lstrip("_").startswith("refuse_f0")
                        and len(node.args) == 2
                        and isinstance(node.args[1], ast.Constant)):
                    key = (rel, _enclosing_function(spans, node.lineno),
                           node.args[1].value)
                    found[key] = "refuse_f0_sheets"
                # has_f0_sheets(...) used as a lane eligibility fence
                elif (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "has_f0_sheets"
                        and rel != "rfx/materials/thin_conductor.py"):
                    key = (rel, _enclosing_function(spans, node.lineno),
                           "has_f0_sheets")
                    found[key] = "has_f0_sheets"
                # raise ValueError("... (#677 v1) ...")
                elif isinstance(node, ast.Raise):
                    exc = node.exc
                    if not (isinstance(exc, ast.Call) and exc.args
                            and isinstance(exc.args[0], ast.Constant)
                            and isinstance(exc.args[0].value, str)):
                        continue
                    msg = exc.args[0].value
                    if "(#677 v1)" not in msg:
                        continue
                    func = _enclosing_function(spans, node.lineno)
                    disc = [k for (_r, _f, k) in FENCE_REGISTRY
                            if _r == rel and _f == func and k in msg]
                    key = (rel, func, disc[0] if disc else msg)
                    found[key] = "inline"
    return found


def test_every_fence_in_the_source_is_pinned():
    """Every #677 lane fence in ``rfx/`` is named by ``FENCE_REGISTRY``.

    This is the anti-regression that closes the CLASS: adding a new fence
    without a lane-level test is red here, and so is deleting a fence whose
    lane is still registered (the registry would name a call site the
    source no longer has).
    """
    found = _scan_source_fences()
    registered = set(FENCE_REGISTRY)
    non_fence = set(_NON_FENCE_F0_QUERIES)
    missing = sorted(set(found) - registered - non_fence)
    stale = sorted(registered - set(found))
    # A non-fence row that no longer matches any call site is stale too —
    # otherwise the exemption list would outlive the code it excuses.
    stale += sorted(non_fence - set(found))
    assert not missing, (
        "these #677 lane fences exist in rfx/ but no test is registered for "
        f"them — add a lane-level test and a FENCE_REGISTRY row: {missing}")
    assert not stale, (
        "FENCE_REGISTRY names call sites that are no longer in rfx/ — a "
        f"fence was moved or deleted: {stale}")


def test_every_registered_pinning_test_exists():
    """A registry row is only worth something if the test it names is real —
    a rename must not silently orphan a fence."""
    orphans = []
    for key, (mod_name, test_name) in {**FENCE_REGISTRY,
                                       **_MANUAL_FENCES}.items():
        mod = (importlib.import_module(mod_name)
               if mod_name != __name__ else globals())
        have = (test_name in mod if isinstance(mod, dict)
                else hasattr(mod, test_name))
        if not have:
            orphans.append((key, mod_name, test_name))
    assert not orphans, f"registered pinning tests that do not exist: {orphans}"


# ===========================================================================
# formerly tests/unit/materials/test_sheet_impedance.py
# ===========================================================================

F0 = 10e9
DX = 0.25e-3
NX, NY, NZ = 6, 6, 8
FOOT = range(1, 5)          # 4 cells wide in both in-plane directions


def _spec(xs, ys, zs, normal, g=1.0):
    m = np.zeros((NX, NY, NZ), bool)
    m[np.ix_(list(xs), list(ys), list(zs))] = True
    m = jnp.asarray(m)
    return SheetImpedanceSpec(mask=m, normal_axis=normal, g_sheet=g,
                              sigma_sheet=jnp.where(m, g / DX, 0.0))


def _counts(ctx):
    return (int(jnp.sum(ctx.mask_ex)),
            int(jnp.sum(ctx.mask_ey)),
            int(jnp.sum(ctx.mask_ez)))


def test_adjacent_same_normal_sheets_leave_the_gap_edge_unloaded():
    """ORACLE 1: |mask_ex| = |mask_ey| = |F|*n, |mask_ez| = 0."""
    ctx = build_sheet_impedance_ctx([
        _spec(FOOT, FOOT, [3], 2),
        _spec(FOOT, FOOT, [4], 2, g=2.0),
    ])
    assert _counts(ctx) == (32, 32, 0), _counts(ctx)

    # the in-plane sets are exactly the union of each film's own classification
    from rfx.boundaries.pec import tangential_edge_masks
    a = _spec(FOOT, FOOT, [3], 2)
    b = _spec(FOOT, FOOT, [4], 2, g=2.0)
    ax, ay, _ = tangential_edge_masks(a.mask)
    bx, by, _ = tangential_edge_masks(b.mask)
    assert bool(jnp.all(ctx.mask_ex == (ax | bx)))
    assert bool(jnp.all(ctx.mask_ey == (ay | by)))

    # sigma is still the per-node sum; the two films occupy disjoint cells,
    # so no mixing happens and each layer keeps its own value.
    sig = np.asarray(ctx.sigma_sheet)
    assert np.isclose(sig[2, 2, 3], 1.0 / DX)
    assert np.isclose(sig[2, 2, 4], 2.0 / DX)


def test_deeper_same_normal_stack_leaves_every_gap_edge_unloaded():
    """Three adjacent layers: |mask_ez| stays 0, in-plane scales with n."""
    ctx = build_sheet_impedance_ctx([
        _spec(FOOT, FOOT, [3], 2),
        _spec(FOOT, FOOT, [4], 2),
        _spec(FOOT, FOOT, [5], 2),
    ])
    assert _counts(ctx) == (48, 48, 0), _counts(ctx)


def test_non_adjacent_stack_is_unchanged_negative_control():
    """The pre-fix code already got this right; it must not move."""
    ctx = build_sheet_impedance_ctx([
        _spec(FOOT, FOOT, [2], 2),
        _spec(FOOT, FOOT, [5], 2, g=2.0),
    ])
    assert _counts(ctx) == (32, 32, 0), _counts(ctx)


def test_single_sheet_ctx_is_unchanged_negative_control():
    ctx = build_sheet_impedance_ctx([_spec(FOOT, FOOT, [3], 2)])
    assert _counts(ctx) == (16, 16, 0), _counts(ctx)


def test_coincident_same_normal_sheets_still_add_conductance():
    """Parallel sheet admittances on one node: 1/Rs_tot = 1/Rs_1 + 1/Rs_2."""
    ctx = build_sheet_impedance_ctx([
        _spec(FOOT, FOOT, [3], 2, g=1.0),
        _spec(FOOT, FOOT, [3], 2, g=2.0),
    ])
    assert _counts(ctx) == (16, 16, 0), _counts(ctx)
    assert np.isclose(float(np.asarray(ctx.sigma_sheet)[2, 2, 3]), 3.0 / DX)


def test_coplanar_abutting_one_cell_strips_keep_both_in_plane_components():
    """A patterned plane drawn as abutting 1-cell boxes must classify the
    same as the single Box that covers them. The literal per-spec-OR
    alternative to the #690 fix drops mask_ex here (measured 8 -> 0)."""
    ctx = build_sheet_impedance_ctx([
        _spec([3], FOOT, [3], 2),
        _spec([4], FOOT, [3], 2),
    ])
    one_box = build_sheet_impedance_ctx([_spec([3, 4], FOOT, [3], 2)])
    assert _counts(ctx) == _counts(one_box), (_counts(ctx), _counts(one_box))
    assert bool(jnp.all(ctx.mask_ex == one_box.mask_ex))
    assert bool(jnp.all(ctx.mask_ey == one_box.mask_ey))
    assert bool(jnp.all(ctx.mask_ez == one_box.mask_ez))


def _stacked_gap_sim():
    sim = Simulation(freq_max=20e9, domain=(4e-3, 4e-3, 4e-3), dx=DX,
                     boundary="pec")
    for zc in (2.0e-3, 2.25e-3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(
                Box((1e-3, 1e-3, zc), (3e-3, 3e-3, zc)),
                sigma_bulk=5.8e7, surface_impedance_f0=F0)
    # Source OFF the films and OFF the probed edge — see the module
    # docstring: an Ez source on the gap edge writes it directly and hides
    # the very loss under test.
    sim.add_source((2e-3, 2e-3, 1.0e-3), "ex",
                   waveform=GaussianPulse(f0=F0), amplitude_kind="field")
    sim.add_probe((2e-3, 2e-3, 2.125e-3), "ez")
    return sim


def test_gap_between_stacked_films_rings_instead_of_being_clamped():
    """ORACLE 2 — end-to-end field witness. Preflight on this fixture reads,
    verbatim: ``  [PREFLIGHT] All checks passed (NTFF advisory tier; the
    PEC-overlap error check runs on forward()/preflight()).``"""
    res = _stacked_gap_sim().run(n_steps=1500)
    ts = np.abs(np.asarray(res.time_series).ravel())
    peak = float(ts.max())
    # 4eb7fa4 measured 1.038701e-09 here; fixed measures 4.037870e-03.
    assert peak > 1e-5, peak


# ---------------------------------------------------------------------------
# The veto's exception: a LENGTH-1 axis is not a normal direction.
#
# The #690 veto above zeroes component ``c`` wherever no covering sheet has
# ``c`` tangential. On a 2-D grid (nz == 1) ``sheet_normal_axis`` calls every
# flat patch z-normal, because z IS its thinnest bounding-box axis — there is
# no other z extent available. The veto then zeroed ``mask_ez``, and in
# ``mode="2d_tmz"`` Ez is the only live E component, so the sheet became
# bit-identically inert: an f0 copper patch behaved exactly like vacuum.
#
# ``tangential_edge_masks`` already keeps the wrap on a length-1 axis for
# exactly this reason (#689) — the veto ran after it and threw the result
# away. Measured on the fixture below (20x20x1, dx = 1 mm, 6x6-cell copper
# patch, 400 steps):
#
#     ctx nnz (ex, ey, ez)          (36, 36,  0)  ->  (36, 36, 36)
#     G4 PEC/f0 identity, 2-D mask   False        ->  True
#     peak|Ez| at the patch node    7.536002e-02  ->  8.477738e-13
#     f0/none, patch node            1.000000     ->  0.000000
#     f0/none, shadow probe          1.000000     ->  0.051592
#     pec/none, shadow probe         0.051638 (reference, unchanged)
#
# The shadow ratio is the physics check, not just a mask count: at 15 GHz
# copper's Rs is 0.032 ohm/square, so a copper film must shadow like a PEC
# plate. Fixed f0 and PEC agree to 0.09%; the inert sheet is off by 19x.
#
# Preflight for the f0 and none fixtures, verbatim:
#   ``  [PREFLIGHT] All checks passed (NTFF advisory tier; the PEC-overlap
#   error check runs on forward()/preflight()).``
# and for the PEC reference, whose first probe sits inside the plate on
# purpose (that IS the measurement — Ez must be zero there):
#   ``  [PREFLIGHT] Port/source at (0.011, 0.011, 0) is inside PEC geometry
#   'pec'. Field will be zero. Move source outside PEC.``
#
# The exception is the DEGENERATE axis only, NOT ``periodic[c]``: see
# test_periodic_seam_gap_edge_stays_vetoed.
# ---------------------------------------------------------------------------

TMZ_SHAPE = (20, 20, 1)
TMZ_PATCH = Box((0.008, 0.008, 0.0), (0.014, 0.014, 0.001))


def _flat_spec(shape, xs, ys, normal, g=1.0):
    m = np.zeros(shape, bool)
    m[np.ix_(list(xs), list(ys), [0])] = True
    m = jnp.asarray(m)
    return SheetImpedanceSpec(mask=m, normal_axis=normal, g_sheet=g,
                              sigma_sheet=jnp.where(m, g / DX, 0.0))


def test_two_d_sheet_keeps_its_out_of_plane_component():
    """nz == 1: the veto must not fire on z. Without the exception the
    only live 2d_tmz component is zeroed and the sheet is inert."""
    spec = _flat_spec(TMZ_SHAPE, range(8, 14), range(8, 14), 2)
    ctx = build_sheet_impedance_ctx([spec], periodic=(False, False, True))
    assert _counts(ctx) == (36, 36, 36), _counts(ctx)

    # ...and it holds on the DEFAULT flags too, which is what the callers
    # that never pass ``periodic`` (forward(), the NU runner, the eager
    # S-parameter re-runs) get.
    ctx_def = build_sheet_impedance_ctx([spec])
    assert _counts(ctx_def) == (36, 36, 36), _counts(ctx_def)


def test_two_d_sheet_keeps_the_g4_pec_footprint_identity():
    """#677 G4 on a 2-D mask: the f0 footprint is what apply_pec_mask zeroes."""
    from rfx.boundaries.pec import apply_pec_mask
    from rfx.core.yee import init_state

    spec = _flat_spec(TMZ_SHAPE, range(8, 14), range(8, 14), 2)
    per = (False, False, True)
    ctx = build_sheet_impedance_ctx([spec], periodic=per)
    st = init_state(TMZ_SHAPE)
    ones = jnp.ones(TMZ_SHAPE, jnp.float32)
    out = apply_pec_mask(st._replace(ex=ones, ey=ones, ez=ones),
                         spec.mask, per)
    for zeroed, m in ((out.ex, ctx.mask_ex), (out.ey, ctx.mask_ey),
                      (out.ez, ctx.mask_ez)):
        np.testing.assert_array_equal(np.asarray(zeroed) == 0.0,
                                      np.asarray(m))


def test_periodic_seam_gap_edge_stays_vetoed():
    """The exception is the degenerate axis, NOT a periodic axis.

    Two one-layer z-normal films on cells 0 and n-1 of a z-PERIODIC domain
    are two films with the seam between them, exactly the #690 geometry —
    ``tangential_edge_masks`` fuses them through the wrap (32 Ez edges) and
    the veto must still remove them. Broadening the exception to
    ``union.shape[c] == 1 or periodic[c]`` reds this test with ez = 32.
    """
    from rfx.boundaries.pec import tangential_edge_masks
    per = (False, False, True)
    a = _spec(FOOT, FOOT, [0], 2)
    b = _spec(FOOT, FOOT, [NZ - 1], 2)
    fused = tangential_edge_masks(a.mask | b.mask, per)
    assert int(jnp.sum(fused[2])) == 32, int(jnp.sum(fused[2]))
    ctx = build_sheet_impedance_ctx([a, b], periodic=per)
    assert _counts(ctx) == (32, 32, 0), _counts(ctx)


def _tmz_sim(kind):
    sim = Simulation(freq_max=30e9, domain=(0.02, 0.02, 0.001), dx=1e-3,
                     boundary="pec", mode="2d_tmz")
    if kind == "pec":
        sim.add(TMZ_PATCH, material="pec")
    elif kind == "f0":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(TMZ_PATCH, sigma_bulk=5.8e7,
                                   surface_impedance_f0=15e9)
    sim.add_source((0.004, 0.010, 0), "ez",
                   waveform=GaussianPulse(f0=15e9), amplitude_kind="field")
    sim.add_probe((0.011, 0.011, 0), "ez")   # inside the patch footprint
    sim.add_probe((0.017, 0.010, 0), "ez")   # shadow, behind the patch
    return sim


def test_two_d_f0_patch_shadows_like_pec_end_to_end():
    """Physics witness: copper at 15 GHz must shadow like PEC. Preflight
    for each fixture is quoted verbatim in the section comment above."""
    peaks = {}
    for kind in ("none", "f0", "pec"):
        res = _tmz_sim(kind).run(n_steps=400, skip_preflight=(kind == "pec"))
        ts = np.abs(np.asarray(res.time_series))
        peaks[kind] = (float(ts[:, 0].max()), float(ts[:, 1].max()))

    node_ratio = peaks["f0"][0] / peaks["none"][0]
    shadow_f0 = peaks["f0"][1] / peaks["none"][1]
    shadow_pec = peaks["pec"][1] / peaks["none"][1]
    # inert sheet measured 1.000000 / 1.000000; fixed 1.1e-11 / 0.051592
    assert node_ratio < 1e-6, (node_ratio, peaks)
    assert abs(shadow_f0 - shadow_pec) / shadow_pec < 0.02, \
        (shadow_f0, shadow_pec, peaks)


# ===========================================================================
# formerly tests/unit/materials/test_sheet_impedance.py
# ===========================================================================

F0 = 10e9
SIGMA_BULK = 1e4
THICKNESS = 35e-6


# ---------------------------------------------------------------------------
# Test shapes: a planar sheet with an optional rectangular hole, implemented
# from the Shape protocol (mask_on_coords + bounding_box) WITHOUT deriving
# from Box — the point of O674-1 is that two independent occupancy
# implementations reach the same folded sigma, not that Box calls itself.
# ---------------------------------------------------------------------------

class PlanarSheet:
    """Flat sheet on ``axis = coord``, footprint ``[lo, hi)``, optional hole.

    Conventions match the primitives deliberately: half-open ``[lo, hi)`` on
    the in-plane axes (:class:`rfx.geometry.csg.Box`'s volume rule) and the
    single nearest node on the normal axis (Box's thin-sheet rule), so an
    equivalent Box and this shape must rasterize to the same cells.
    """

    def __init__(self, axis, coord, plane_lo, plane_hi, hole=None):
        self.axis = int(axis)
        self.coord = float(coord)
        self.plane_lo = tuple(float(v) for v in plane_lo)   # (a0, a1) lows
        self.plane_hi = tuple(float(v) for v in plane_hi)
        self.hole = None if hole is None else (
            tuple(float(v) for v in hole[0]), tuple(float(v) for v in hole[1]))

    @property
    def _plane_axes(self):
        return tuple(a for a in range(3) if a != self.axis)

    def bounding_box(self):
        lo, hi = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        lo[self.axis] = hi[self.axis] = self.coord
        for i, a in enumerate(self._plane_axes):
            lo[a], hi[a] = self.plane_lo[i], self.plane_hi[i]
        return tuple(lo), tuple(hi)

    def mask_on_coords(self, x, y, z):
        # Comparisons run in the COORDINATE dtype, exactly as the primitives
        # do, so a node sitting on a knife edge lands on the same side for
        # both: what this module tests is the FOLD, not float32 boundary
        # conventions (which the Box docstring covers at length).
        coords = [jnp.asarray(c).ravel() for c in (x, y, z)]
        per_axis = []
        for a in range(3):
            c = coords[a]
            if a == self.axis:
                m = jnp.zeros(c.shape, dtype=bool).at[
                    jnp.argmin(jnp.abs(c - self.coord))].set(True)
            else:
                i = self._plane_axes.index(a)
                m = (c >= self.plane_lo[i]) & (c < self.plane_hi[i])
            per_axis.append(m)
        out = (per_axis[0][:, None, None] & per_axis[1][None, :, None]
               & per_axis[2][None, None, :])
        if self.hole is not None:
            holes = []
            for a in range(3):
                if a == self.axis:
                    holes.append(jnp.ones(coords[a].shape, dtype=bool))
                else:
                    i = self._plane_axes.index(a)
                    holes.append((coords[a] >= self.hole[0][i])
                                 & (coords[a] < self.hole[1][i]))
            out = out & ~(holes[0][:, None, None] & holes[1][None, :, None]
                          & holes[2][None, None, :])
        return out

    def mask(self, grid):
        from rfx.geometry.csg import _grid_coords
        return self.mask_on_coords(*_grid_coords(grid))


class MaskOnlyShape:
    """A shape that can rasterize but cannot say where its normal is."""

    def __init__(self, inner):
        self._inner = inner

    def mask_on_coords(self, x, y, z):
        return self._inner.mask_on_coords(x, y, z)

    def mask(self, grid):
        return self._inner.mask(grid)


class BoundsOnlyShape:
    """A shape with a bounding box and no way to rasterize itself."""

    def __init__(self, lo, hi):
        self._lo, self._hi = tuple(lo), tuple(hi)

    def bounding_box(self):
        return self._lo, self._hi


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

# uniform fixture: 20x20x3 mm at dx = 1 mm, sheet on the z = 1 mm node plane
U_DX = 1e-3
U_DOMAIN = (0.02, 0.02, 0.003)
U_Z = 1e-3
U_FOOT = ((5e-3, 5e-3), (15e-3, 15e-3))          # [lo, hi) in x and y
U_HOLE = ((8e-3, 8e-3), (12e-3, 12e-3))

# NU fixture: dz = [0.5 mm]x8 + [1.5 mm]x8, sheet ON the 4.0 mm step node
NU_DX = 0.5e-3
NU_DZ = [0.5e-3] * 8 + [1.5e-3] * 8
NU_L = 24 * NU_DX
NU_Z = 4.0e-3
NU_FOOT = ((6 * NU_DX, 6 * NU_DX), (18 * NU_DX, 18 * NU_DX))
NU_HOLE = ((10 * NU_DX, 10 * NU_DX), (14 * NU_DX, 14 * NU_DX))


def _uniform_sigma(shape, **kw):
    sim = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(shape, sigma_bulk=SIGMA_BULK,
                               thickness=THICKNESS, **kw)
    grid = sim._build_grid()
    specs = []
    mats, _, _, pec_mask, *_ = sim._assemble_materials(
        grid, sheet_specs=specs)
    if kw.get("surface_impedance_f0") is not None:
        # #677: the f0 sheet is emitted as a SheetImpedanceSpec instead of
        # a materials.sigma fold; sigma_sheet is the SAME per-node realized
        # quantity every assertion in this module was written against, so
        # the helpers return it in f0 mode (and assert the arrays stayed
        # sheet-free).
        assert float(np.asarray(mats.sigma).max()) == 0.0
        assert len(specs) == 1
        return np.asarray(specs[0].sigma_sheet), pec_mask, grid, sim
    return np.asarray(mats.sigma), pec_mask, grid, sim


def _nu_grid(sim):
    return build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
        sim._dz_profile, dx_profile=sim._dx_profile,
        dy_profile=sim._dy_profile,
        pec_faces=sim._boundary_spec.pec_faces(),
        pmc_faces=sim._boundary_spec.pmc_faces(),
        cpml_axes="xyz")


def _nu_sigma(shape, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0),
                         dx=NU_DX, dz_profile=NU_DZ, boundary="cpml",
                         cpml_layers=6)
        sim.add_thin_conductor(shape, sigma_bulk=SIGMA_BULK,
                               thickness=THICKNESS, **kw)
        grid = _nu_grid(sim)
        specs = []
        mats, _, _, pec_mask = assemble_materials_nu(
            sim, grid, sheet_specs=specs)
    if kw.get("surface_impedance_f0") is not None:
        # #677 retarget — see _uniform_sigma
        assert float(np.asarray(mats.sigma).max()) == 0.0
        assert len(specs) == 1
        return np.asarray(specs[0].sigma_sheet), pec_mask, grid, sim
    return np.asarray(mats.sigma), pec_mask, grid, sim


def _box_sheet(z, foot):
    (x0, y0), (x1, y1) = foot
    return Box((x0, y0, z), (x1, y1, z))


def _planar_sheet(z, foot, hole=None):
    (x0, y0), (x1, y1) = foot
    return PlanarSheet(2, z, (x0, y0), (x1, y1), hole=hole)


# ---------------------------------------------------------------------------
# O674-1: Box vs an equivalent mask shape — bit-identical fold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_box_and_equivalent_mask_shape_fold_bit_identically(lane):
    """Same occupancy, different shape class -> identical sigma digest.

    The NU leg puts the sheet on the 0.5/1.5 mm grading step, so the fold's
    dual-spacing normalization (#671) is what is being reproduced, not a
    uniform-mesh coincidence.
    """
    if lane == "uniform":
        z, foot, cell, run = U_Z, U_FOOT, U_DX, _uniform_sigma
    else:
        z, foot, cell, run = NU_Z, NU_FOOT, NU_DX, _nu_sigma

    kw = dict(surface_impedance_f0=F0)
    sig_box, pec_box, grid, _ = run(_box_sheet(z, foot), **kw)
    sig_msk, pec_msk, _, _ = run(_planar_sheet(z, foot), **kw)

    # the two occupancies must agree cell-for-cell first: if they do not, the
    # digest below would be reporting a geometry difference, not a fold one
    n_box = int((sig_box > 0).sum())
    assert n_box > 0, "fixture folded no cells"
    np.testing.assert_array_equal(sig_box > 0, sig_msk > 0)
    assert _sha(sig_box) == _sha(sig_msk), (
        f"{lane}: non-Box sheet folded a different sigma "
        f"(max {sig_box.max():.6g} vs {sig_msk.max():.6g})")

    # neither contributes PEC bits, and the realized sheet is the requested one
    for pec in (pec_box, pec_msk):
        assert pec is None or int(np.asarray(pec).sum()) == 0
    rs0 = float(leontovich_rs(F0, SIGMA_BULK))
    if lane == "uniform":
        d_norm = np.full(1, grid.dx)
        ks = [0]
    else:
        from rfx.nonuniform import e_node_dual_spacings
        ks = sorted({int(k) for k in np.argwhere(sig_msk > 0)[:, 2]})
        assert len(ks) == 1, ks
        d_norm = np.asarray(e_node_dual_spacings(grid.dz))[ks]
        primal = np.asarray(grid.dz)[ks[0]]
        assert abs(primal / d_norm[0] - 1.5) < 1e-3, (primal, d_norm)
    prod = float(sig_msk.max()) * rs0 * float(d_norm[0])
    assert abs(prod - 1.0) < 1e-5, f"{lane}: sigma_eff*Rs0*d_norm = {prod}"

    # negative control: a digest-equality gate that cannot fail is not a gate.
    # One cell of footprint shift must move it.
    (fx, fy), hi_corner = foot
    shifted = _planar_sheet(z, ((fx + cell, fy), hi_corner))
    sig_shift, _, _, _ = run(shifted, **kw)
    assert _sha(sig_shift) != _sha(sig_box), (
        f"{lane}: shifting the sheet one cell did not change the digest")
    assert int((sig_shift > 0).sum()) < n_box


def test_dc_fold_also_accepts_a_mask_shape_on_the_uniform_lane():
    """The legacy DC fold was never Box-only on the uniform lane (it reads
    ``shape.mask``); pin that #674 did not change it. The NU DC path keeps its
    documented warn-and-skip for non-Box shapes."""
    sig_box, _, _, _ = _uniform_sigma(_box_sheet(U_Z, U_FOOT))
    sig_msk, _, _, _ = _uniform_sigma(_planar_sheet(U_Z, U_FOOT))
    assert _sha(sig_box) == _sha(sig_msk)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0), dx=NU_DX,
                         dz_profile=NU_DZ, boundary="cpml", cpml_layers=6)
        sim.add_thin_conductor(_planar_sheet(NU_Z, NU_FOOT),
                               sigma_bulk=SIGMA_BULK, thickness=THICKNESS)
        mats_nu, *_ = assemble_materials_nu(sim, _nu_grid(sim))
    assert any("non-Box shape is not yet supported" in str(w.message)
               for w in rec), [str(w.message) for w in rec]
    assert float(np.asarray(mats_nu.sigma).max()) == 0.0


# ---------------------------------------------------------------------------
# O674-2: patterned sheet — the hole stays untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_patterned_sheet_folds_only_occupied_cells(lane):
    if lane == "uniform":
        z, foot, hole, run = U_Z, U_FOOT, U_HOLE, _uniform_sigma
    else:
        z, foot, hole, run = NU_Z, NU_FOOT, NU_HOLE, _nu_sigma

    solid = _planar_sheet(z, foot)
    holed = _planar_sheet(z, foot, hole=hole)
    sig_solid, _, grid, _ = run(solid, surface_impedance_f0=F0)
    sig_holed, _, _, _ = run(holed, surface_impedance_f0=F0)

    if lane == "uniform":
        m_solid = np.asarray(solid.mask(grid))
        m_holed = np.asarray(holed.mask(grid))
    else:
        from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
        coords = coords_from_nonuniform_grid(grid)
        m_solid = np.asarray(solid.mask_on_coords(
            coords.x, coords.y, coords.z))
        m_holed = np.asarray(holed.mask_on_coords(
            coords.x, coords.y, coords.z))

    n_hole = int((m_solid & ~m_holed).sum())
    assert n_hole > 0, "clearance hole rasterized to nothing — gate is blind"
    assert int(m_holed.sum()) > 0

    # per cell: folded exactly on the occupied set, and NOT in the hole
    np.testing.assert_array_equal(sig_holed > 0, m_holed)
    assert float(sig_holed[m_solid & ~m_holed].max(initial=0.0)) == 0.0, (
        "clearance-hole cells carry sheet conductivity")
    # the cells that remain carry exactly the same sigma as in the solid sheet
    np.testing.assert_array_equal(sig_holed[m_holed], sig_solid[m_holed])
    # ... and the background is untouched
    assert float(sig_holed[~m_holed].max(initial=0.0)) == 0.0


# ---------------------------------------------------------------------------
# O674-3: what still fails loud
# ---------------------------------------------------------------------------

def test_body_with_height_refused_on_both_lanes():
    """A 3-D shape is not a sheet: it rasterizes to more than one layer along
    its normal, and folding it per cell would multiply the sheet conductance
    by the layer count. Refused at build time on both lanes."""
    ball = Sphere((10e-3, 10e-3, 1.5e-3), 1.2e-3)
    with pytest.raises(ValueError, match="cell layers along its normal"):
        _uniform_sigma(ball, surface_impedance_f0=F0)
    ball_nu = Sphere((6e-3, 6e-3, 4.0e-3), 1.2e-3)
    with pytest.raises(ValueError, match="cell layers along its normal"):
        _nu_sigma(ball_nu, surface_impedance_f0=F0)


def test_sheet_that_rasterizes_to_nothing_is_refused_not_vaporized():
    """The #369 class, reachable by a non-Box sheet a Box could not reach: a
    footprint that falls entirely between node planes folds zero cells. It
    must raise, never silently vanish."""
    # footprint strictly inside one cell in x: [5.2, 5.8) mm on a 1 mm grid
    ghost = PlanarSheet(2, U_Z, (5.2e-3, 5e-3), (5.8e-3, 15e-3))
    with pytest.raises(ValueError, match="ZERO cells"):
        _uniform_sigma(ghost, surface_impedance_f0=F0)


def test_shape_without_a_mask_or_bounds_is_refused_at_add_time():
    sim = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
    with pytest.raises(ValueError, match="mask_on_coords"):
        sim.add_thin_conductor(
            BoundsOnlyShape((0, 0, U_Z), (1e-3, 1e-3, U_Z)),
            surface_impedance_f0=F0)
    with pytest.raises(ValueError, match="bounding box"):
        sim.add_thin_conductor(
            MaskOnlyShape(_planar_sheet(U_Z, U_FOOT)),
            surface_impedance_f0=F0)


def test_nu_defensive_refusal_for_a_bounds_less_sheet():
    """Built outside ``add_thin_conductor`` (which is where the add-time check
    lives), a bounds-less f0 sheet must still fail loud on the NU lane rather
    than inherit the DC path's warn-and-skip."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0), dx=NU_DX,
                         dz_profile=NU_DZ, boundary="cpml", cpml_layers=6)
        sim.add_thin_conductor(_box_sheet(NU_Z, NU_FOOT),
                               sigma_bulk=SIGMA_BULK, thickness=THICKNESS)
        grid = _nu_grid(sim)
    sim._thin_conductors[0] = ThinConductor(
        shape=MaskOnlyShape(_planar_sheet(NU_Z, NU_FOOT)),
        sigma_bulk=SIGMA_BULK, thickness=THICKNESS, surface_impedance_f0=F0)
    with pytest.raises(ValueError, match="refusing to skip"):
        assemble_materials_nu(sim, grid)


# ---------------------------------------------------------------------------
# O674-4: the graded-node advisory follows a non-Box sheet
# ---------------------------------------------------------------------------

def _preflight_text(shape, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0), dx=NU_DX,
                         dz_profile=NU_DZ, boundary="cpml", cpml_layers=6)
        sim.add_thin_conductor(shape, sigma_bulk=SIGMA_BULK,
                               thickness=THICKNESS, **kw)
        return " ".join(sim.preflight())


def test_graded_node_advisory_follows_a_nonbox_sheet():
    """Fires for a plain non-Box sheet on the step, fires for a PATTERNED one
    whose bounding-box centre falls in its clearance hole (a single-point
    probe reads "no sheet" there), and stays quiet on a matched node."""
    on_step = _preflight_text(_planar_sheet(NU_Z, NU_FOOT),
                              surface_impedance_f0=F0)
    assert "adjacent cells differ" in on_step, on_step
    assert "DUAL spacing" in on_step, on_step
    assert "500µm below" in on_step and "1.5mm above" in on_step, on_step

    # bounding-box centre is inside the hole by construction
    centre_hole = _planar_sheet(NU_Z, NU_FOOT, hole=NU_HOLE)
    lo, hi = centre_hole.bounding_box()
    mid = [0.5 * (lo[a] + hi[a]) for a in range(3)]
    assert not bool(np.asarray(centre_hole.mask_on_coords(
        np.array([mid[0]]), np.array([mid[1]]), np.array([mid[2]]))).any())
    patterned = _preflight_text(centre_hole, surface_impedance_f0=F0)
    assert "adjacent cells differ" in patterned, patterned

    # quiet: same sheets on a locally uniform node (z = 8 mm, deep in the
    # 1.5 mm region)
    for shape in (_planar_sheet(8.0e-3, NU_FOOT),
                  _planar_sheet(8.0e-3, NU_FOOT, hole=NU_HOLE)):
        quiet = _preflight_text(shape, surface_impedance_f0=F0)
        assert "adjacent cells differ" not in quiet, quiet


# ---------------------------------------------------------------------------
# O674-5: design-IR contract
# ---------------------------------------------------------------------------

def test_design_ir_records_a_registered_nonbox_sheet_and_refuses_the_rest():
    """The EXISTING shape codec decides: a registered primitive round-trips
    with its f0 field, an unregistered shape class is refused loudly (never
    degraded to a bounding box)."""
    from rfx.interop import design_to_dict, simulation_from_design
    from rfx.interop._errors import UnsupportedDesignFeature

    # height well under one cell so the disc is a SHEET: only the node plane
    # at its centre is contained (|h| <= height/2 = 50 um vs dx = 1 mm)
    disc = Cylinder(center=(10e-3, 10e-3, U_Z), radius=4e-3, height=1e-4,
                    axis="z")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
        sim.add_thin_conductor(disc, sigma_bulk=SIGMA_BULK,
                               thickness=THICKNESS, surface_impedance_f0=F0)
        doc = design_to_dict(sim)
        back = simulation_from_design(doc)
    tc = back._thin_conductors[0]
    assert tc.shape == disc
    assert float(tc.surface_impedance_f0) == F0
    sig_a, _, _, _ = _uniform_sigma(disc, surface_impedance_f0=F0)
    specs_b = []
    mats_b = back._assemble_materials(
        back._build_grid(), sheet_specs=specs_b)[0]
    # #677: the round-tripped sim emits the identical sheet spec (arrays
    # stay sheet-free on both sides)
    assert float(np.asarray(mats_b.sigma).max()) == 0.0
    assert len(specs_b) == 1
    assert _sha(sig_a) == _sha(np.asarray(specs_b[0].sigma_sheet))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim2 = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
        sim2.add_thin_conductor(_planar_sheet(U_Z, U_FOOT),
                                sigma_bulk=SIGMA_BULK, thickness=THICKNESS,
                                surface_impedance_f0=F0)
    with pytest.raises(UnsupportedDesignFeature, match="PlanarSheet"):
        design_to_dict(sim2)


# ---------------------------------------------------------------------------
# The real CAD path: MeshShape (issue #358) carrying a surface-impedance sheet
# ---------------------------------------------------------------------------

def _mesh_slab(x0, x1, y0, y1, z0, z1, extra=()):
    """Watertight slab (or union of disjoint slabs) as a MeshShape."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")          # trimesh.contains needs it
    from rfx.geometry.mesh_import import MeshShape

    def _box(a0, a1, b0, b1, c0, c1):
        m = trimesh.creation.box(
            extents=(a1 - a0, b1 - b0, c1 - c0))
        m.apply_translation(((a0 + a1) / 2, (b0 + b1) / 2, (c0 + c1) / 2))
        return m

    parts = [_box(x0, x1, y0, y1, z0, z1)]
    parts += [_box(*p) for p in extra]
    mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    return MeshShape(mesh)


def test_mesh_shape_sheet_folds_bit_identically_to_its_box():
    """An imported CAD slab and the Box it stands for fold the same sigma.

    Bounds are chosen OFF the node planes (4.6 .. 14.6 mm on a 1 mm grid) so
    the mesh's closed containment test and Box's half-open ``[lo, hi)`` rule
    select the same nodes 5..14 mm — the comparison is of the FOLD, not of two
    boundary conventions.
    """
    slab = _mesh_slab(4.6e-3, 14.6e-3, 4.6e-3, 14.6e-3,
                      U_Z - 1e-4, U_Z + 1e-4)
    sig_mesh, pec_mesh, grid, _ = _uniform_sigma(slab,
                                                 surface_impedance_f0=F0)
    sig_box, _, _, _ = _uniform_sigma(_box_sheet(U_Z, U_FOOT),
                                      surface_impedance_f0=F0)
    assert int((sig_mesh > 0).sum()) == 100, int((sig_mesh > 0).sum())
    np.testing.assert_array_equal(sig_mesh > 0, sig_box > 0)
    assert _sha(sig_mesh) == _sha(sig_box)
    assert pec_mesh is None or int(np.asarray(pec_mesh).sum()) == 0


def test_mesh_shape_patterned_sheet_leaves_its_clearance_hole_alone():
    """A ground plane with a clearance hole, as it arrives from CAD: four
    disjoint bars around an opening. The fold touches the metal only."""
    x0, x1, y0, y1 = 4.6e-3, 14.6e-3, 4.6e-3, 14.6e-3
    h0, h1 = 7.6e-3, 11.6e-3            # clearance opening
    z0, z1 = U_Z - 1e-4, U_Z + 1e-4
    frame = _mesh_slab(
        x0, x1, y0, h0, z0, z1,
        extra=((x0, x1, h1, y1, z0, z1),
               (x0, h0, h0, h1, z0, z1),
               (h1, x1, h0, h1, z0, z1)))
    sig, _, grid, _ = _uniform_sigma(frame, surface_impedance_f0=F0)
    mask = np.asarray(frame.mask(grid))

    solid = _mesh_slab(x0, x1, y0, y1, z0, z1)
    hole = np.asarray(solid.mask(grid)) & ~mask
    assert int(hole.sum()) == 16, int(hole.sum())     # 4x4 opening cells
    np.testing.assert_array_equal(sig > 0, mask)
    assert float(sig[hole].max(initial=0.0)) == 0.0

    rs0 = float(leontovich_rs(F0, SIGMA_BULK))
    prod = np.asarray(sig)[mask] * rs0 * grid.dx
    np.testing.assert_allclose(prod, 1.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# The #671 transition-node FDTD oracle, re-run with a NON-Box sheet
# ---------------------------------------------------------------------------

def _guide_planar_sheet(zs):
    """The oracle guide's plate, as a non-Box mask shape of equal footprint."""
    from tests.oracle.test_leontovich_alpha_oracle import DOMAIN as _D
    return PlanarSheet(2, zs, (0.0, 0.0), (_D[0], _D[1]))


@pytest.mark.slow_physics
@pytest.mark.parametrize("case,control,dual_over_primal",
                         [pytest.param(*c, id=c[0]) for c in INVARIANCE_CASES])
def test_alpha_invariance_transfers_to_a_nonbox_sheet(case, control,
                                                      dual_over_primal):
    """#671's oracle with the plates expressed as non-Box mask shapes.

    The gate is the same one: attenuation on a mesh where the sheet sits ON a
    grading step, over the locally-uniform control that shares that mesh, must
    be 1 within [0.98, 1.02] — a mesh-independent sheet has no other option.
    Because the two shapes rasterize the same cells, the folded sigma is
    bit-identical and so is alpha; both are asserted, so a divergence names
    itself (occupancy vs fold) instead of arriving as a drifted ratio.
    """
    from tests.unit.materials.test_thin_conductor_nu_dual_spacing import (
        RATIO_GATE, _run_nu_guide)
    got = _run_nu_guide(case, sheet_shape=_guide_planar_sheet, tag="planar")
    ref = _run_nu_guide(control, sheet_shape=_guide_planar_sheet,
                        tag="planar")

    hi_case = got["sheets"][-1]
    assert abs(hi_case[2] / hi_case[1] - dual_over_primal) < 1e-3, hi_case
    for _k, prim, dual, _prod in ref["sheets"]:
        assert abs(dual / prim - 1.0) < 1e-6, (prim, dual)
    for _k, _p, _d, prod in got["sheets"] + ref["sheets"]:
        assert abs(prod - 1.0) < 1e-5, prod
    for out in (got, ref):
        assert out["resid"] < 0.02, out["resid"]
        assert out["settle_db"] < -40.0, out["settle_db"]
        assert not any("PreflightError" in w for w in out["warnings"])

    ratio = got["alpha"] / ref["alpha"]
    lo, hi = RATIO_GATE
    assert lo <= ratio <= hi, (
        f"{case} (non-Box sheet): alpha {got['alpha']:.5f} vs control "
        f"{control} {ref['alpha']:.5f} -> ratio {ratio:.4f} outside "
        f"[{lo}, {hi}]")

    # ... and it is the SAME number the Box sheet gives on the same mesh
    for name, out in ((case, got), (control, ref)):
        box = _run_nu_guide(name)
        assert out["alpha"] == box["alpha"], (
            f"{name}: non-Box sheet alpha {out['alpha']:.9f} != Box "
            f"{box['alpha']:.9f} on identical occupancy")


def test_occupancy_guard_does_not_break_the_traced_mesh_path():
    """The #674 guard reads CONCRETE occupancy; on the differentiable-mesh
    path the mask is a tracer and the guard must step aside rather than raise
    a ConcretizationTypeError. Closed form: sigma_eff = 1/(Rs0*d_norm) scales
    as 1/scale, so d(sum sigma)/d(scale) = -sum/scale.
    """
    import jax

    base = jnp.asarray(NU_DZ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0), dx=NU_DX,
                         dz_profile=base, boundary="cpml", cpml_layers=6)
        sim.add_thin_conductor(_box_sheet(NU_Z, NU_FOOT),
                               sigma_bulk=SIGMA_BULK, thickness=THICKNESS,
                               surface_impedance_f0=F0)

    def loss(scale):
        sim._dz_profile = base * scale
        specs = []
        assemble_materials_nu(sim, _nu_grid(sim), sheet_specs=specs)
        # #677: the sheet quantity (and its mesh derivative) lives on the
        # emitted spec now; the occupancy guard must still skip the traced
        # mask instead of raising ConcretizationTypeError.
        return jnp.sum(specs[0].sigma_sheet)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(loss(1.0))
        grad = float(jax.grad(loss)(1.0))
    sim._dz_profile = base
    assert value > 0.0
    assert abs(grad + value) / value < 1e-5, (grad, -value)


def test_vmap_batched_build_folds_a_nonbox_sheet_identically():
    """The batched (``vmap_sweep``) material build re-applies the same shared
    fold, so a non-Box sheet must land on the batched slices exactly as it
    lands on the serial ones (#669's O7, re-run off the Box)."""
    from rfx.vmap_sweep import _build_batched_materials

    def make(eps_val):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
            sim.add_material("substrate", eps_r=eps_val)
            sim.add(Box((0.0, 0.0, 0.0), (0.02, 0.02, 1e-3)),
                    material="substrate")
            sim.add_thin_conductor(_planar_sheet(U_Z, U_FOOT, hole=U_HOLE),
                                   sigma_bulk=SIGMA_BULK,
                                   thickness=THICKNESS,
                                   surface_impedance_f0=F0)
        return sim

    eps_values = np.array([2.0, 6.0])
    sim = make(4.0)
    grid = sim._build_grid()
    base, *_ = sim._assemble_materials(grid)
    batched = _build_batched_materials(
        sim, grid, base, "substrate.eps_r", jnp.asarray(eps_values))
    assert batched.sigma.shape[0] == 2
    for idx, eps_val in enumerate(eps_values):
        specs = []
        serial, *_ = make(float(eps_val))._assemble_materials(
            grid, sheet_specs=specs)
        assert np.array_equal(np.asarray(batched.sigma[idx]),
                              np.asarray(serial.sigma))
        # #677: both builds are sheet-free in sigma; the non-Box sheet
        # lands on the emitted spec exactly as on the serial build (the
        # sweep itself takes the sequential fallback which applies it).
        assert float(np.asarray(serial.sigma).max()) == 0.0
        assert len(specs) == 1
        assert float(np.asarray(specs[0].sigma_sheet).max()) > 0.0
