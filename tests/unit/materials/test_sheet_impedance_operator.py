"""#677 node-thin surface-impedance sheet operator — unit + limit gates.

Design B (exponential stepping): the f0 sheet is realized per step at the
apply_pec_mask tangential edge set as ``E^{n+1} = A*E^n + B*curlH`` with
``A = exp(-x2)``, ``B = -expm1(-x2)/sigma_tot``,
``x2 = sigma_tot*dt/(eps0*eps_r)``, ``sigma_tot = sigma_bg + G/d_dual``.

Gates in this module (numbers per the #677 implementation contract):

G3a  Rs0 = 1e12 (G -> 0): sheet-on fields == sheet-absent fields to rtol
     1e-5 — the stencil-identity falsifier for operator/kernel curl drift
     (x64-scoped per-test; float32 coefficient rounding grows secularly in
     a resonant cavity and is precision noise, not stencil drift).
G3b  Rs0 = 1e-6: resonances match the PEC-mask realization to <= 1e-4
     relative.
G3c  per masked edge, x2 * Rs0 * eps0 * eps_r * d_dual / dt == 1 to rtol
     1e-6, on uniform AND on the 0.5/1.5 mm NU grading-transition node,
     with the primal-spacing product asserted != 1 (O2 anti-regression
     teeth ported, not dropped).
G4   footprint identity: the ctx's tangential edge masks equal
     apply_pec_mask's masks for the same cell mask, exact boolean
     equality, on uniform, NU and patterned (#674) shapes; the
     normal-component edge set is all-False for a one-layer sheet.
G5   default-off run byte identity: kwarg-absent vs
     surface_impedance_f0=None produce byte-identical run outputs on PEC
     and DC-fold fixtures (assembly digests are pinned in
     tests/unit/materials/test_leontovich_sheet_identity.py).
G8   reference-strip negative control: the NU reference run's explicit
     ``strip_sheet_impedance`` changes S/observables on a sheet-bearing
     device and is a byte-level no-op on a sheet-free control.
G9   lane fences, THIS module's share only: the GPU baked fast path
     excludes sheets; the two distributed runners, UPML, the two
     dispersive-overlap lanes and the crossing-normal builder refuse
     loudly. Each of those enters through its own lane (or, for the
     builder, is a pure unit).

     The other G9 fences named in the #677 commit — ADI run/forward,
     subgridded, distributed multi-device, NU-forward, MSL, MSL-junction,
     waveguide subpixel + multimode, optimize (both meshes),
     gradient-check, topology, material-fit, NU anisotropic-eps, uniform
     subpixel/conformal, forward-dispersive — are pinned in
     ``tests/unit/materials/test_sheet_lane_fences.py``, which enters each lane's public
     entry point AND asserts the raising frame, so a dropped call site
     reds. That file's ``FENCE_REGISTRY`` is the single inventory of every
     #677 fence and names the pinning test for each, including the ones
     that live here; its ``test_every_fence_in_the_source_is_pinned``
     fails if a fence exists with no registered test.
"""

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation
from rfx.core.yee import EPS_0
from rfx.materials.thin_conductor import (
    ThinConductor,
    apply_thin_conductor,
    build_sheet_impedance_ctx,
    leontovich_rs,
    refuse_f0_sheets,
    sheet_update_coeffs,
)
from rfx.boundaries.pec import apply_pec_mask, tangential_edge_masks
from rfx.core.yee import init_state, init_materials

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
    from tests.unit.materials.test_thin_conductor_nonbox_sheet import (
        _planar_sheet, U_Z, U_FOOT, U_HOLE)
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
    call-site proofs live in ``tests/unit/materials/test_sheet_lane_fences.py``.
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
