"""Finite-region ``add_flux_monitor(size=...)`` on the non-uniform (graded) mesh.

Audit task (c) / B1. Until this landed, the NU runner raised
``NotImplementedError`` for any finite-region flux monitor
(``rfx/runners/nonuniform.py`` guard). B1 wires the physical ``size=``/``center=``
into a CELL window against the graded cumulative cell edges via the fresh
``_nu_flux_tangential_bounds`` helper (waveguide path left byte-identical — #889
edits it concurrently). These tests pin, from strongest to most physical:

  T1  the guard is gone and a graded-mesh finite monitor returns a finite,
      non-zero spectrum (guards against silent re-introduction of the raise).
  T2  RELATIVE oracle: the finite monitor equals the full-plane NU integrand
      summed over the SAME lo:hi window to machine precision (x64-scoped).
  T3  WINDOW oracle (reviewer defect P1): the selected window and its
      cumulative dA are compared against a span recomputed from the realized
      edge array — the pad offset and the node->cell (#868) decision that T2 is
      structurally blind to (it applies the same possibly off-by-one window on
      both sides). Sub-check (e) is the one that leaves the window entirely: it
      compares each REQUESTED endpoint against the edge the runner actually
      chose for it, PER ENDPOINT, within that endpoint's own half-cell argmin
      bound (review2 F4 + fix2b verify nit 2 — (a)-(d) all read the monitor's
      own lo/hi, so none of them can see a snap to the wrong edge; the earlier
      size-only form of (e) could not see a one-cell error at ONE endpoint,
      because that error lands exactly ON a whole-cell tolerance).
  T4  GENERALITY / anti-hardcode: the SAME physical size/center on two DIFFERENT
      gradings must yield DIFFERENT lo/hi and DIFFERENT dA (the monitor tracks
      the mesh), while the recovered physical power agrees across gradings within
      a MEASURED ENVELOPE whose shape ((k*d_max)^2 * (1+Q/P)) is derived and
      computed at runtime from the fixture — never a pinned literal — but whose
      O(1) prefactor is empirical, NOT a derived error bound (review2 F6; Yee
      dispersion carries 1/12). A monitor returning identical dA regardless of
      grading FAILS. Carries the open-CPML ring-down settling witness.
  T5  CLAMPING is documented and non-silent (review2 F7 + fix2b verify nit 1):
      an oversize or off-edge window is CLAMPED to the interior — the helper
      never claimed correctly that it "never clamps" — and that clamp emits a
      UserWarning. (d) pins the case the old size-difference PROXY missed: a
      clamp SMALLER than one cell, which only the per-endpoint clamp test sees.
      (e) pins the documented silent case (a clamp of half an end cell or less,
      indistinguishable from ordinary snapping); (c) pins that an
      exactly-representable window does not warn.

Every tolerance printed in-assertion is ``f(dA, dt, dx/dy/dz, eps, freq)``.
x64 is scoped per-test via ``jax.enable_x64`` (never a module-level
``jax.config.update`` — repo rule; it flips at collection and reds every
same-process pytest-split shard).
"""
from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from rfx import Simulation, flux_spectrum
from rfx.sources.sources import ModulatedGaussian
from rfx.nonuniform import interior_cells

try:  # jax >= 0.8.0
    from jax import enable_x64
except ImportError:  # older jax
    from tests._x64_compat import enable_x64

C0 = 2.998e8


# --------------------------------------------------------------------------
# independent (test-side) re-derivation of the intended CELL window and dA,
# read straight off the realized graded grid — NOT via the runner helper, so
# it is a genuine oracle for the runner helper.
# --------------------------------------------------------------------------
def _interior_edges(d_full, pad_lo, pad_hi):
    return np.insert(np.cumsum(np.asarray(interior_cells(d_full, pad_lo, pad_hi))), 0, 0.0)


def _expected_cell_span(edges, pad_lo, center, size):
    """(lo, hi) half-open CELL slice the finite monitor MUST select.

    Mirrors the physics contract (argmin against cumulative interior edges,
    then NODE->CELL narrowing, #868) but written independently here.
    """
    lo_phys = center - size / 2.0
    hi_phys = center + size / 2.0
    lo_local = int(np.argmin(np.abs(edges - lo_phys)))
    hi_local = int(np.argmin(np.abs(edges - hi_phys)))
    # NODE span (lo, hi+1) -> CELL span (lo, hi): the hi_local-lo_local cells
    # bracketed by the two chosen edge nodes.
    return (lo_local + pad_lo, hi_local + pad_lo), (lo_local, hi_local)


def _grid_axis(grid, axis):
    if axis == 0:
        return np.asarray(grid.dx_arr), grid.pad_x_lo, grid.pad_x_hi, grid.nx
    if axis == 1:
        return np.asarray(grid.dy_arr), grid.pad_y_lo, grid.pad_y_hi, grid.ny
    return np.asarray(grid.dz), grid.pad_z_lo, grid.pad_z_hi, grid.nz


def _build_open_graded_sim(dz_profile, *, dy_profile=None, nx=44, dx=0.5e-3,
                           f0=9e9, add_probe=False):
    """Open-CPML graded sim: ez source; monitor plane is x-normal so the
    graded z axis (and optionally graded y) is TANGENTIAL to the plane and its
    per-cell dA varies with the grading."""
    dz = np.asarray(dz_profile, dtype=float)
    Lz = float(np.sum(dz))
    ny = 28 if dy_profile is None else len(dy_profile)
    Ly = ny * dx if dy_profile is None else float(np.sum(dy_profile))
    Lx = nx * dx
    kwargs = dict(freq_max=0.5 * C0 / dx, domain=(Lx, Ly, Lz), dx=dx,
                  dz_profile=dz, cpml_layers=8, boundary="cpml")
    if dy_profile is not None:
        kwargs["dy_profile"] = np.asarray(dy_profile, dtype=float)
    sim = Simulation(**kwargs)
    sim.add_source((Lx * 0.30, Ly * 0.5, Lz * 0.5), "ez",
                   waveform=ModulatedGaussian(f0=f0, bandwidth=0.6, amplitude=1.0))
    if add_probe:
        sim.add_probe((Lx * 0.72, Ly * 0.5, Lz * 0.5), "ez")
    grid = sim._build_nonuniform_grid()
    return sim, grid, (Lx, Ly, Lz)


# --------------------------------------------------------------------------
# T1 — the guard is gone; a graded finite monitor returns a finite non-zero
#      spectrum. (Fail-before: origin/main raised NotImplementedError here.)
# --------------------------------------------------------------------------
def test_nu_finite_flux_guard_removed_returns_spectrum():
    dz = np.array([0.25e-3] * 8 + [0.5e-3] * 8, dtype=float)
    sim, grid, (Lx, Ly, Lz) = _build_open_graded_sim(dz)
    freqs = jnp.asarray(np.linspace(5e9, 18e9, 6))
    sim.add_flux_monitor(axis="x", coordinate=Lx * 0.6, freqs=freqs,
                         size=(Ly * 0.5, Lz * 0.5),
                         center=(Ly * 0.5, Lz * 0.5), name="fin")
    # Would raise NotImplementedError on origin/main; must run now.
    res = sim.run(n_steps=400, compute_s_params=False)
    mon = res.flux_monitors["fin"]
    spec = np.asarray(flux_spectrum(mon))
    assert np.all(np.isfinite(spec)), f"finite-region NU flux spectrum not finite: {spec}"
    assert float(np.max(np.abs(spec))) > 0.0, (
        "finite-region NU flux spectrum is identically zero — the DFT never "
        "accumulated (field/monitor mismatch)")
    # It selected a strict sub-window of the plane (fewer cells than full).
    assert (mon.hi1 - mon.lo1) < grid.ny and (mon.hi2 - mon.lo2) < grid.nz, (
        f"finite window ({mon.lo1}:{mon.hi1}, {mon.lo2}:{mon.hi2}) is not a "
        f"strict sub-region of the ({grid.ny},{grid.nz}) plane")


# --------------------------------------------------------------------------
# T2 — RELATIVE oracle: finite == full-plane restricted to the same window,
#      to machine precision. Proves B1 adds only a window, not new field math.
# --------------------------------------------------------------------------
def _restrict_full_to_window(mon_full, mon_sub):
    e1 = np.asarray(mon_full.e1_dft); e2 = np.asarray(mon_full.e2_dft)
    h1 = np.asarray(mon_full.h1_dft); h2 = np.asarray(mon_full.h2_dft)
    integrand = e1 * np.conj(h2) - e2 * np.conj(h1)
    dA = np.asarray(mon_full.dA)  # (ny_plane, nz_plane) per-cell for NU
    win_i = integrand[:, mon_sub.lo1:mon_sub.hi1, mon_sub.lo2:mon_sub.hi2]
    win_A = dA[mon_sub.lo1:mon_sub.hi1, mon_sub.lo2:mon_sub.hi2]
    return np.real(np.sum(win_i * win_A[None], axis=(-2, -1)))


def test_nu_finite_flux_bitexact_restriction():
    with enable_x64(True):
        dz = np.array([0.3e-3] * 6 + [0.55e-3] * 8, dtype=float)
        sim, grid, (Lx, Ly, Lz) = _build_open_graded_sim(dz)
        freqs = jnp.asarray(np.linspace(5e9, 18e9, 8))
        cx = Lx * 0.6
        sim.add_flux_monitor(axis="x", coordinate=cx, freqs=freqs, name="full")
        sim.add_flux_monitor(axis="x", coordinate=cx, freqs=freqs,
                             size=(Ly * 0.5, Lz * 0.5),
                             center=(Ly * 0.5, Lz * 0.5), name="fin")
        res = sim.run(n_steps=1200, compute_s_params=False)
        mon_full = res.flux_monitors["full"]
        mon_fin = res.flux_monitors["fin"]

        # Witness: DFT accumulator well above noise.
        acc = float(np.max(np.abs(np.asarray(mon_full.e2_dft))))
        assert acc > 1e-25, f"flux DFT at noise floor ({acc:.3e}); setup degenerate"

        fin = np.asarray(flux_spectrum(mon_fin))
        ref = _restrict_full_to_window(mon_full, mon_fin)
        scale = max(np.max(np.abs(fin)), np.max(np.abs(ref)), 1e-300)
        reldev = float(np.max(np.abs(fin - ref) / scale))
        print(f"\n[T2 bit-exact] window=({mon_fin.lo1}:{mon_fin.hi1},"
              f"{mon_fin.lo2}:{mon_fin.hi2}) max reldev={reldev:.3e}")
        # Machine-precision gate; ceiling generous vs measured ~1e-15 but far
        # below any real bookkeeping bug.
        assert reldev < 1e-11, (
            f"finite NU flux deviates from the full-plane integrand over the "
            f"SAME window by {reldev:.3e} (>1e-11) — a real finite-region "
            "bookkeeping bug, not the expected machine-eps match")


# --------------------------------------------------------------------------
# T3 — ABSOLUTE aperture-cell / cumulative-dA oracle (reviewer defect P1).
#      Catches a node/cell off-by-one that T2 (self-referential window) cannot.
# --------------------------------------------------------------------------
def test_nu_finite_flux_absolute_aperture_and_dA():
    dz = np.array([0.25e-3] * 8 + [0.5e-3] * 8, dtype=float)
    sim, grid, (Lx, Ly, Lz) = _build_open_graded_sim(dz)
    freqs = jnp.asarray(np.linspace(5e9, 18e9, 4))
    size_y, size_z = Ly * 0.5, Lz * 0.5
    cy, cz = Ly * 0.5, Lz * 0.5
    sim.add_flux_monitor(axis="x", coordinate=Lx * 0.6, freqs=freqs,
                         size=(size_y, size_z), center=(cy, cz), name="fin")
    res = sim.run(n_steps=300, compute_s_params=False)
    mon = res.flux_monitors["fin"]

    # Independent expectation from the realized graded edges (axis y=1, z=2).
    dy_full, pad_y_lo, pad_y_hi, _ = _grid_axis(grid, 1)
    dz_full, pad_z_lo, pad_z_hi, _ = _grid_axis(grid, 2)
    edges_y = _interior_edges(dy_full, pad_y_lo, pad_y_hi)
    edges_z = _interior_edges(dz_full, pad_z_lo, pad_z_hi)
    (yl, yh), (yll, yhl) = _expected_cell_span(edges_y, pad_y_lo, cy, size_y)
    (zl, zh), (zll, zhl) = _expected_cell_span(edges_z, pad_z_lo, cz, size_z)

    # (a) exact index match.
    assert (mon.lo1, mon.hi1) == (yl, yh), (
        f"y window {mon.lo1}:{mon.hi1} != independently-derived {yl}:{yh}")
    assert (mon.lo2, mon.hi2) == (zl, zh), (
        f"z window {mon.lo2}:{mon.hi2} != independently-derived {zl}:{zh}")

    # (b) cell COUNT == hi_local - lo_local (the node->cell contract, #868).
    n_y, n_z = yhl - yll, zhl - zll
    assert (mon.hi1 - mon.lo1) == n_y and (mon.hi2 - mon.lo2) == n_z, (
        f"selected cell counts ({mon.hi1-mon.lo1},{mon.hi2-mon.lo2}) != the "
        f"aperture cell counts ({n_y},{n_z}) — a node/cell off-by-one")

    # (c) cumulative dA == sum of the INTENDED cells' local face areas, computed
    #     independently from the realized grid (NOT from the monitor's own dA).
    exp_dA = float(np.sum(np.outer(dy_full[yl:yh], dz_full[zl:zh])))
    got_dA = float(np.sum(np.asarray(mon.dA)))
    reldA = abs(got_dA - exp_dA) / max(exp_dA, 1e-300)
    print(f"\n[T3 absolute] y {mon.lo1}:{mon.hi1} z {mon.lo2}:{mon.hi2} "
          f"n=({n_y},{n_z}) dA got={got_dA:.6e} exp={exp_dA:.6e} rel={reldA:.2e}")
    # float32 store of the cell sizes -> ~1e-6 relative; gate derived from that.
    tol_dA = 1e-4  # >> f32 eps*n_cells, << one coarse-cell area fraction
    assert reldA < tol_dA, (
        f"cumulative dA {got_dA:.6e} != intended {exp_dA:.6e} (rel {reldA:.2e} "
        f">{tol_dA}) — the finite window does not integrate the intended cells")

    # (d) node->cell APPLIED, not skipped: the physical aperture the selected
    #     z-cells span equals edges[hi_local]-edges[lo_local]; the WRONG (raw
    #     node-span) choice would append one extra coarse cell and overshoot.
    span_selected = float(edges_z[zhl] - edges_z[zll])
    span_if_offbyone = float(edges_z[min(zhl + 1, len(edges_z) - 1)] - edges_z[zll])
    dA_z_selected = float(np.sum(dz_full[zl:zh]))
    # Exactly equal in exact arithmetic (the selected cells ARE interior[zll:zhl]);
    # the only spread is float32 cumulative-sum roundoff, ~ n_cells * eps_f32.
    tol_span = 1e-4 * span_selected   # >> n*eps_f32 (~1e-6), << one coarse cell (~0.2)
    assert abs(dA_z_selected - span_selected) < tol_span, (
        f"selected z-cell widths {dA_z_selected:.9e} != physical aperture "
        f"{span_selected:.9e} (tol {tol_span:.2e}) — node->cell (#868) not "
        "applied correctly (an off-by-one would append one whole coarse cell)")
    assert span_if_offbyone > span_selected, (
        "graded fixture must make the off-by-one detectable (extra coarse cell "
        "strictly enlarges the aperture); choose a coarser tail")

    # (e) REQUESTED vs SELECTED, PER ENDPOINT — the only check here that reads
    #     the REQUESTED center/size at all. (a)-(d) all compare against a
    #     window. The runner resolves each requested endpoint by argmin against
    #     the cumulative interior edges, and the nearest edge to a point is at
    #     most half a local cell away, so each endpoint carries its OWN bound:
    #       |edges[lo_local] - (c - s/2)| <= max(d[lo_local-1], d[lo_local])/2
    #     and likewise at hi_local. Both the request and the bound come from the
    #     realized grid — no literal.
    #     WHY PER ENDPOINT (fix2b, verify nit 2): the previous form compared the
    #     SUMMED extent against max(d) over the window. A one-cell error at ONE
    #     endpoint moves the summed extent by exactly one cell, i.e. exactly ONTO
    #     that tolerance, and passed under `<=`. Split across the two endpoints
    #     the same error is one WHOLE cell against a HALF-cell bound, so it
    #     fails. What (e) now catches: a two-cell-or-larger mis-snap of either
    #     endpoint; a ONE-cell mis-snap only when the request lies on an edge or
    #     the neighbouring cells differ by < 2x (at a 2:1 transition a request
    #     0.1-0.49 cell into the last fine cell can still pass the
    #     max(neighbours)/2 bound - measured on this fixture's z axis); and any
    #     clamp that moves an endpoint more than half a cell.
    #     What it still cannot catch: a request landing exactly mid-cell, where
    #     both neighbouring edges are equidistant and either is a correct argmin.
    for label, d_full_, edges_, pad_lo_, pad_hi_, lo, hi, c_req, s_req in (
            ("y", dy_full, edges_y, pad_y_lo, pad_y_hi, mon.lo1, mon.hi1, cy, size_y),
            ("z", dz_full, edges_z, pad_z_lo, pad_z_hi, mon.lo2, mon.hi2, cz, size_z)):
        int_d = np.asarray(interior_cells(d_full_, pad_lo_, pad_hi_))
        n_int = int(len(int_d))
        for end, node, requested in (("lo", lo - pad_lo_, c_req - s_req / 2.0),
                                     ("hi", hi - pad_lo_, c_req + s_req / 2.0)):
            # half the larger of the (at most two) interior cells that the
            # chosen edge node separates: the argmin bound at THIS endpoint.
            neigh = ([float(int_d[node - 1])] if node > 0 else []) + \
                    ([float(int_d[node])] if node < n_int else [])
            tol_end = max(neigh) / 2.0
            err = abs(float(edges_[node]) - requested)
            print(f"[T3 requested] {label}.{end}: requested={requested:.6e} "
                  f"edge={float(edges_[node]):.6e} err={err:.3e} "
                  f"half_cell_bound={tol_end:.3e}")
            assert err <= tol_end, (
                f"{label} {end} endpoint: the runner chose interior edge node "
                f"{node} at {float(edges_[node]):.6e} m but {requested:.6e} m "
                f"was REQUESTED — off by {err:.3e} m, beyond this endpoint's "
                f"half-cell argmin bound {tol_end:.3e} m: that endpoint did not "
                "snap to the nearest cumulative edge (a one-cell mis-snap, or a "
                "clamp)")
        selected = float(np.sum(d_full_[lo:hi]))
        print(f"[T3 requested] {label}: requested_size={s_req:.6e} "
              f"selected_size={selected:.6e} err={abs(selected - s_req):.3e}")


# --------------------------------------------------------------------------
# T4 — GENERALITY: two gradings -> different lo/hi and dA; recovered physical
#      power agrees within a runtime-DERIVED bound. + ring-down settling.
# --------------------------------------------------------------------------
def _settling_db(time_series):
    """End-vs-peak ring-down witness on the AC (time-varying) field only.

    A hard ez source injects net charge, so the probe carries a STATIC offset
    that never radiates away (measured ~ -20 dB plateau). The flux monitor's
    DFT is evaluated at analysis frequencies >= 5 GHz, which integrate ONLY the
    time-varying field and reject that DC offset identically. The settling
    witness must therefore track the AC content, isolated here by the first
    time-difference d/dt (DC-blind): a constant offset -> 0, radiating field ->
    its true decay. This is the field the recovered-power DFT actually sees.
    """
    ac = np.abs(np.diff(np.asarray(time_series).ravel().astype(np.float64)))
    peak = float(np.max(ac))
    tail = float(np.max(ac[int(len(ac) * 0.9):]))
    return 20.0 * math.log10(max(tail, 1e-300) / max(peak, 1e-300))


def _react_ratio(mon):
    """Per-frequency |Im S| / |Re S| of the plane flux integral: the local
    reactive-to-real ratio, measured from the monitor's own DFT (used to size
    the near-field cross-mesh bound; large in the near field)."""
    e1 = np.asarray(mon.e1_dft); e2 = np.asarray(mon.e2_dft)
    h1 = np.asarray(mon.h1_dft); h2 = np.asarray(mon.h2_dft)
    g = e1 * np.conj(h2) - e2 * np.conj(h1)
    S = np.sum(g * np.asarray(mon.dA)[None], axis=(-2, -1))
    return np.abs(np.imag(S)) / (np.abs(np.real(S)) + 1e-300)


def test_nu_finite_flux_generality_two_gradings():
    # SAME physical domain (same Lz, same buffers, same source/monitor
    # positions); the two runs differ ONLY in how the MIDDLE 2.5 mm is meshed:
    # 10 cells of 0.25 mm vs 5 cells of 0.50 mm. Uniform buffers (>=8 cells)
    # against both z-absorber faces keep the CPML runway uniform (the
    # graded-cell-in-absorber PML-breakdown class the preflight flags). The
    # graded z axis is TANGENTIAL to the x-normal monitor plane, so a hardcoded
    # cubic-cell (dz==dx) monitor would misweight it.
    d0 = 0.35e-3
    dz_fine = np.array([d0] * 8 + [0.25e-3] * 10 + [d0] * 8, dtype=float)
    dz_coarse = np.array([d0] * 8 + [0.50e-3] * 5 + [d0] * 8, dtype=float)
    assert abs(float(np.sum(dz_fine)) - float(np.sum(dz_coarse))) < 1e-12, \
        "fixtures must share the SAME domain (only the mesh density differs)"

    freqs_np = np.linspace(5e9, 16e9, 6)
    freqs = jnp.asarray(freqs_np)
    size_z = 2.6e-3   # covers the whole graded middle + a little buffer
    recovered, windows, dA_arrs, react = [], [], [], []
    settle = []
    ap_area = []   # independently-computed physical aperture area per run
    Lx = Ly = None
    x_src = x_mon = None
    for tag, dz in [("fine", dz_fine), ("coarse", dz_coarse)]:
        sim, grid, (Lx, Ly, Lzr) = _build_open_graded_sim(dz, add_probe=True)
        size_y = Ly * 0.6
        cy, cz = Ly * 0.5, Lzr * 0.5
        cx = Lx * 0.62
        x_src, x_mon = Lx * 0.30, cx
        sim.add_flux_monitor(axis="x", coordinate=cx, freqs=freqs,
                             size=(size_y, size_z), center=(cy, cz), name="fin")
        res = sim.run(n_steps=3000, compute_s_params=False)
        mon = res.flux_monitors["fin"]
        recovered.append(np.real(np.asarray(flux_spectrum(mon))))
        windows.append((mon.lo1, mon.hi1, mon.lo2, mon.hi2))
        dA_arrs.append(np.asarray(mon.dA))
        react.append(_react_ratio(mon))
        settle.append(_settling_db(res.time_series))
        # independent physical aperture area from the realized graded edges:
        # (sum of selected dy)(sum of selected dz), NOT the monitor's own dA.
        dy_full, ply, phy, _ = _grid_axis(grid, 1)
        dz_full, plz, phz, _ = _grid_axis(grid, 2)
        y_ext = float(np.sum(dy_full[mon.lo1:mon.hi1]))
        z_ext = float(np.sum(dz_full[mon.lo2:mon.hi2]))
        ap_area.append((y_ext, z_ext, y_ext * z_ext))

    dA_sums = [float(np.sum(a)) for a in dA_arrs]
    ncells = [(w[1] - w[0], w[3] - w[2]) for w in windows]
    print(f"\n[T4 generality] windows={windows} ncells={ncells} "
          f"dA_sums={dA_sums} settle_dB={[round(s,1) for s in settle]}")
    print(f"  aperture (y_ext,z_ext,area): {ap_area}")

    # (a) the z-cell COUNT tracks the mesh: the finer middle mesh selects MORE
    #     cells over the SAME physical aperture. (Cell count is grading-DEPENDENT.)
    nz_fine, nz_coarse = ncells[0][1], ncells[1][1]
    assert nz_fine > nz_coarse, (
        f"finer grading did not select more aperture cells ({nz_fine} vs "
        f"{nz_coarse}) — the window is not tracking the realized mesh")

    # (b) the per-cell dA face weights are grading-DEPENDENT (different arrays),
    #     not a single pinned value.
    assert dA_arrs[0].shape != dA_arrs[1].shape, (
        "per-cell dA arrays have identical shape across two different mesh "
        "densities — the weights are not tracking the realized mesh")

    # (c) THE anti-hardcode killer: dA_sum equals the grading-INVARIANT physical
    #     aperture area for BOTH gradings (to float32), even though the cell
    #     counts differ. A cubic-cell (dz==dx) hardcode would instead give
    #     dA_sum = dx^2 * n_cells, which differs between the two gradings and
    #     does NOT equal the physical area on the graded axis — so it CANNOT
    #     pass this pair of checks.
    for (tag, dAs, (yE, zE, area)) in zip(("fine", "coarse"), dA_sums, ap_area):
        rel_area = abs(dAs - area) / max(area, 1e-300)
        assert rel_area < 1e-4, (
            f"[{tag}] dA_sum {dAs:.6e} != physical aperture area {area:.6e} "
            f"(rel {rel_area:.2e}) — the face weights are not the realized dz")
    rel_area_cross = abs(dA_sums[0] - dA_sums[1]) / max(dA_sums[0], 1e-300)
    # both equal the same physical area to within the sub-cell snapping of the
    # two meshes (< one coarse cell of z-extent over the full aperture). The two
    # apertures share the same y window, so the RELATIVE area difference equals
    # the relative z-extent difference and one coarse cell of z-snap is the
    # DIMENSIONLESS fraction max(dz_coarse)/z_ext. (Until the review2 F5 fix this
    # line also multiplied by y_ext, so the "fraction" carried units of metres.)
    coarse_cell_frac = float(np.max(dz_coarse)) / ap_area[1][1]
    assert rel_area_cross < coarse_cell_frac, (
        f"aperture areas disagree across gradings ({dA_sums}) by more than one "
        f"coarse cell ({coarse_cell_frac:.2e}) — snapping/weighting inconsistent")

    # (d) ring-down settling witness (claims-bearing power): AC field decayed
    #     well below the source peak (open-CPML domain, DC offset removed).
    for tag, s in zip(("fine", "coarse"), settle):
        assert s < -40.0, (
            f"[{tag}] end-of-run AC envelope {s:.1f} dB of peak (> -40 dB) — "
            "the DFT window closed on an un-settled field; recovered power is "
            "truncation-contaminated (repo ring-down rule)")

    # (e) recovered physical power agrees across gradings within a MEASURED
    #     ENVELOPE (review2 F6: this is NOT a derivation — do not call it one).
    #     The cross-mesh difference is a near-field observable: a small REAL
    #     flux extracted from a large reactive field, so the resolution error is
    #     amplified by the local reactive/real ratio Q/P (measured above). The
    #     SHAPE is derived and computed at runtime from the fixture: the
    #     (k*d_max)^2 Yee dispersion scaling times the measured (1+Q/P)
    #     amplification — no pinned literal, no fixture-specific number. The
    #     PREFACTOR 3.0 is NOT derived: Yee phase-velocity error carries 1/12,
    #     and 3.0 is an empirical O(1) envelope covering the two tangential
    #     quadrature directions and the face-snapping difference between the
    #     meshes. It is an envelope that still bites (measured 2026-09-05 on this
    #     fixture: propagating-bin rel = [0.033, 0.029, 0.024, 0.029] against
    #     bound = [0.039, 0.055, 0.076, 0.094], 15 % margin at the tightest bin),
    #     so a violation means "worse than this mesh pair has ever measured", NOT
    #     "worse than theory allows".
    #     Restrict the comparison to the PROPAGATING bins (monitor >= 0.2*lambda
    #     downstream, k*L_path >= 2*pi*0.2) — a physical near/far criterion, not a
    #     hand-picked bin list; the deep-near-field bins carry no meaningful net
    #     real power to referee.
    p1, p2 = recovered[0], recovered[1]
    scale = np.maximum(np.abs(p1), np.abs(p2)) + 1e-300
    rel = np.abs(p1 - p2) / scale
    k = 2 * math.pi * freqs_np / C0
    L_path = abs(x_mon - x_src)
    dmax = float(max(np.max(dz_fine), np.max(dz_coarse)))
    qp = np.maximum(react[0], react[1])
    # measured envelope (see (e)): derived shape, empirical O(1) prefactor.
    envelope_prefactor = 3.0
    bound = envelope_prefactor * (k * dmax) ** 2 * (1.0 + qp)
    propagating = (k * L_path) >= (2 * math.pi * 0.2)
    print(f"[T4 power] L_path={L_path*1e3:.2f}mm dmax={dmax*1e3:.2f}mm")
    print(f"  freqGHz={np.round(freqs_np/1e9,1)} rel={rel.round(3)} "
          f"Q/P={qp.round(2)} bound={bound.round(3)} prop={propagating}")
    assert np.any(propagating), "no propagating bin — enlarge the x-domain"
    viol = propagating & (rel > bound + 1e-9)
    assert not np.any(viol), (
        f"recovered power disagrees across gradings beyond the MEASURED "
        f"near-field envelope at propagating bins: rel={rel[propagating]} vs "
        f"envelope={bound[propagating]} — a weight bug or under-resolved "
        "fixture (the envelope's shape is derived, its prefactor empirical)")


# --------------------------------------------------------------------------
# T5 — CLAMPING is documented and non-silent (review2 F7). The helper does NOT
#      "never clamp": a window reaching outside the interior snaps to the end
#      edges and the monitor then integrates the CLAMPED extent. Pin the
#      documented behaviour and the warning that makes it non-silent, against
#      an expectation computed here from the realized interior edges.
# --------------------------------------------------------------------------
def test_nu_finite_flux_oversize_window_clamps_and_warns():
    import warnings as _warnings
    from rfx.runners.nonuniform import _nu_flux_tangential_bounds

    pad = 2
    d_full = np.concatenate([np.full(pad, 0.5e-3), np.full(20, 0.5e-3),
                             np.full(pad, 0.5e-3)])
    interior = np.asarray(interior_cells(d_full, pad, pad))
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    axis_len = float(edges[-1])
    n_int = int(len(interior))

    # (a) size far larger than the axis -> the WHOLE interior, and a warning.
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        lo, hi = _nu_flux_tangential_bounds(d_full, pad, pad,
                                            axis_len / 2, 3.0 * axis_len)
    assert (lo, hi) == (pad, pad + n_int), (
        f"oversize window resolved to ({lo},{hi}); the documented behaviour is "
        f"the clamped intersection with the interior ({pad},{pad + n_int})")
    assert any(issubclass(c.category, UserWarning) and "CLAMPED" in str(c.message)
               for c in caught), (
        "an oversize window clamped SILENTLY — the review2 F7 warning is gone; "
        f"caught={[str(c.message) for c in caught]}")

    # (b) off-edge window: the requested span reaches below the interior, so
    #     the low face clamps to edge 0 and the delivered extent is the clamped
    #     intersection [max(c-s/2, 0), min(c+s/2, axis_len)] — computed here
    #     from the request and the realized axis length, not written out as a
    #     number (fix2b, verify nit 4: the old form asserted a literal 4.0e-3).
    c_b, s_b = 1.0e-3, 6.0e-3
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        lo, hi = _nu_flux_tangential_bounds(d_full, pad, pad, c_b, s_b)
    delivered = float(np.sum(interior[lo - pad:hi - pad]))
    expected_b = min(c_b + s_b / 2.0, axis_len) - max(c_b - s_b / 2.0, 0.0)
    assert abs(delivered - expected_b) < 1e-12, (
        f"off-edge window delivered {delivered:.6e} m, expected the clamped "
        f"intersection {expected_b:.6e} m")
    assert any("CLAMPED" in str(c.message) for c in caught), (
        "an off-edge window clamped SILENTLY")

    # (c) an ordinary in-interior request must resolve exactly and NOT warn
    #     (the warning fires on clamping, not on ordinary snapping). Endpoints
    #     are taken FROM the realized edge array, so the request is exactly
    #     representable on this mesh whatever interior_cells returns.
    mid = n_int // 2
    lo_i, hi_i = mid - 4, mid + 4
    c_in = 0.5 * (float(edges[lo_i]) + float(edges[hi_i]))
    size_in = float(edges[hi_i]) - float(edges[lo_i])
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        lo, hi = _nu_flux_tangential_bounds(d_full, pad, pad, c_in, size_in)
    assert (lo, hi) == (pad + lo_i, pad + hi_i), (
        f"exactly-representable window resolved to ({lo},{hi}), expected "
        f"({pad + lo_i},{pad + hi_i})")
    assert abs(float(np.sum(interior[lo - pad:hi - pad])) - size_in) < 1e-12
    assert not [c for c in caught if "CLAMPED" in str(c.message)], (
        f"an exactly-representable window raised the clamp warning: "
        f"{[str(c.message) for c in caught]}")

    # (d) SUB-BOUND CLAMP — the case the old size-difference PROXY missed
    #     (fix2b, verify nit 1: the verifier measured requested [-0.5, 3.5] mm
    #     on this axis clamping to 3.5 mm with warned=False). The low face is
    #     requested three quarters of an END CELL below the interior; the high
    #     face is an exact interior edge. The clamp therefore shortens the
    #     extent by LESS than one cell, so |realized - size| never exceeds the
    #     largest cell touching the window and the size-difference test alone
    #     stays silent. Only the PER-ENDPOINT clamp test (endpoint outside the
    #     interior by more than half the end cell) can see it. Every quantity
    #     below is read off the realized interior profile.
    k_hi = n_int // 2 - 2
    lo_req_d = -0.75 * float(interior[0])
    hi_req_d = float(edges[k_hi])
    s_d = hi_req_d - lo_req_d
    c_d = 0.5 * (hi_req_d + lo_req_d)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        lo, hi = _nu_flux_tangential_bounds(d_full, pad, pad, c_d, s_d)
    assert (lo, hi) == (pad, pad + k_hi), (
        f"sub-bound clamp resolved to ({lo},{hi}), expected the clamped "
        f"({pad},{pad + k_hi})")
    delivered_d = float(np.sum(interior[lo - pad:hi - pad]))
    size_gap = abs(delivered_d - s_d)
    proxy_bound = float(np.max(interior[max(lo - pad - 1, 0):hi - pad + 1]))
    # fixture guard: this case MUST stay invisible to the size-difference test,
    # otherwise it is not testing the per-endpoint detector.
    assert size_gap < proxy_bound, (
        f"fixture no longer exercises a SUB-bound clamp: size gap {size_gap:.3e} "
        f"m >= the {proxy_bound:.3e} m size-difference bound, so the old proxy "
        "would have caught it too — shrink the requested overhang")
    assert any(issubclass(c.category, UserWarning) and "CLAMPED" in str(c.message)
               for c in caught), (
        f"a clamp of {abs(lo_req_d):.3e} m (< one {proxy_bound:.3e} m cell) was "
        f"SILENT — the per-endpoint clamp test is gone and only the size-"
        f"difference proxy is left; caught={[str(c.message) for c in caught]}")

    # (e) the DOCUMENTED silent case: a clamp of a QUARTER end cell, i.e. less
    #     than the half-cell threshold. An endpoint anywhere in the interior may
    #     move that far by ordinary snapping, so such a clamp is not
    #     distinguishable from snapping and the docstring says it stays silent.
    #     Pin that, so the threshold cannot drift unnoticed in either direction.
    lo_req_e = -0.25 * float(interior[0])
    hi_req_e = float(edges[k_hi])
    s_e = hi_req_e - lo_req_e
    c_e = 0.5 * (hi_req_e + lo_req_e)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        lo, hi = _nu_flux_tangential_bounds(d_full, pad, pad, c_e, s_e)
    assert (lo, hi) == (pad, pad + k_hi), (
        f"quarter-cell clamp resolved to ({lo},{hi}), expected "
        f"({pad},{pad + k_hi})")
    assert not [c for c in caught if "CLAMPED" in str(c.message)], (
        f"a clamp of {abs(lo_req_e):.3e} m (below the "
        f"{float(interior[0]) / 2.0:.3e} m half-end-cell threshold the "
        f"docstring documents as silent) warned: "
        f"{[str(c.message) for c in caught]}")
