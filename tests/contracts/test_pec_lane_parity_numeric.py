"""One geometry, every solver lane, one answer (#931 §1.7).

Stage B left a gap that no unit test could see: ``conductor_mask()``,
``fidelity_report()`` and the slice viewer all showed a declared PEC
SHEET, and no solver lane applied it — a sheet owns no cell, and every
solve entry took only ``pec_mask``. A simulation with a sheet-declared
ground plane ran as if the metal were not there.

So this file drives the actual lanes and compares fields. ADI's unstable
interior-PEC projection is refused; its contract below pins that refusal
and checks that declared sheets and wires still reach the shared owner.

Battery, on a 20 mm PEC-walled box at dx = 1 mm with an Ez source below
the conductor and a probe above it:

* ``sheet``   — a zero-thickness PEC Box at z = 10 mm (§1.5);
* ``thin``    — the same rectangle through ``add_thin_conductor`` (PEC);
* ``volume``  — a 1-cell PEC Box from z = 10 mm to 11 mm;
* ``none``    — no conductor, the control.

``sheet`` and ``thin`` are the SAME object by the contract (G4 by
construction) and must agree to the bit on every lane. ``volume`` is a
different object — it shorts the normal edge between its two faces — and
must differ from ``sheet`` while still differing from ``none``.

The distributed lanes were absent from this battery because they refused
every kind of declared PEC. #1053 gave the shard_map lane
(``rfx/runners/distributed_v2.py``) a realized-PEC mask stage, so ``volume``
now has a row there; ``sheet`` and ``thin`` own no cell on that lane and its
narrowed refusal is asserted in the same row.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx import Box, Simulation

DX = 1e-3
DOMAIN = (20e-3, 20e-3, 20e-3)
Z_PLANE = 10e-3
SRC = (10e-3, 10e-3, 5e-3)
PROBE = (10e-3, 10e-3, 15e-3)
FOOT_LO = (4e-3, 4e-3)
FOOT_HI = (16e-3, 16e-3)

KINDS = ("none", "sheet", "thin", "volume")


def _build(kind, **sim_kw):
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec",
                     **sim_kw)
    lo = (FOOT_LO[0], FOOT_LO[1], Z_PLANE)
    hi = (FOOT_HI[0], FOOT_HI[1], Z_PLANE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "sheet":
            sim.add(Box(lo, hi), material="pec")
        elif kind == "thin":
            sim.add_thin_conductor(Box(lo, hi), sigma_bulk=5.8e7,
                                   thickness=1e-6)
        elif kind == "volume":
            sim.add(Box(lo, (hi[0], hi[1], Z_PLANE + DX)), material="pec")
    sim.add_source(position=SRC, component="ez")
    sim.add_probe(position=PROBE, component="ez")
    return sim


def _peak(sim, **run_kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(n_steps=200, skip_preflight=True, **run_kw)
    return float(np.max(np.abs(np.asarray(res.time_series)[:, 0])))


def _traces_uniform():
    return {k: _peak(_build(k)) for k in KINDS}


def test_the_conductor_reaches_the_uniform_lane_at_all():
    """The stage-B gap: a declared sheet must change the field."""
    peaks = _traces_uniform()
    assert peaks["sheet"] != pytest.approx(peaks["none"], rel=1e-6), (
        "a declared PEC sheet did not reach the solver at all: "
        f"{peaks}")
    assert peaks["volume"] != pytest.approx(peaks["none"], rel=1e-6)


def test_sheet_and_pec_thin_conductor_are_the_same_object_on_the_uniform_lane():
    """G4 by construction (§1.3), measured on fields rather than masks."""
    peaks = _traces_uniform()
    assert peaks["sheet"] == pytest.approx(peaks["thin"], rel=0, abs=0), peaks


def test_a_one_cell_volume_is_not_the_same_object_as_a_sheet():
    """A volume shorts the normal edge between its two faces; a sheet does
    not (#690). They must not be silently equated."""
    peaks = _traces_uniform()
    assert peaks["volume"] != pytest.approx(peaks["sheet"], rel=1e-9), peaks


def test_the_nonuniform_lane_agrees_with_the_uniform_lane():
    """Same declarations on a uniform-valued NU mesh, same realization.

    The NU lane assembles, rasterizes and realizes through different code
    (``runners/nonuniform.py`` / ``rfx/nonuniform.py``); a uniform dz
    profile makes the two meshes the same lattice, so the CLASSIFICATION
    must agree even though the steppers differ numerically.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.runners.nonuniform import assemble_materials_nu

    for kind in ("sheet", "thin", "volume"):
        uni = _build(kind)
        grid_u = uni._build_grid()
        sheets_u: list = []
        _m, _d, _l, pec_u, *_ = uni._assemble_materials(
            grid_u, pec_sheets=sheets_u)

        nu = _build(kind, dz_profile=np.full(grid_u.nz - 1, DX))
        grid_n = nu._build_nonuniform_grid()
        sheets_n: list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _mn, _dn, _ln, pec_n = assemble_materials_nu(
                nu, grid_n, pec_sheets=sheets_n)

        assert len(sheets_u) == len(sheets_n), kind
        if pec_u is None:
            assert pec_n is None, kind
        else:
            assert np.array_equal(np.asarray(pec_u), np.asarray(pec_n)), kind

        planes_u = realized_wall_planes(
            realized_pec_edge_masks(pec_u, sheets=tuple(sheets_u)), 2)
        planes_n = realized_wall_planes(
            realized_pec_edge_masks(pec_n, sheets=tuple(sheets_n)), 2)
        assert planes_u == planes_n, (kind, planes_u, planes_n)
        if kind == "volume":
            assert len(planes_u) == 2, (kind, planes_u)
        else:
            assert len(planes_u) == 1, (kind, planes_u)


def test_the_vmap_sweep_fast_path_realizes_the_same_conductor():
    """The batched lane builds its own scan body; it must see the sheet."""
    from rfx.vmap_sweep import vmap_material_sweep

    for kind in ("sheet", "volume"):
        sim = _build(kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            swept = vmap_material_sweep(sim, "eps_r", [1.0, 1.0], n_steps=200)
        ts = np.asarray(swept.time_series)
        batched = float(np.max(np.abs(ts[0, :, 0])))
        single = _peak(_build(kind))
        assert batched == pytest.approx(single, rel=2e-3), (kind, batched, single)


#: The shard_map distributed lane against the uniform lane, relative on the
#: probe peak. Measured on this fixture at 2 virtual CPU devices (2026-09-15,
#: jax 0.10.2, float32, 200 steps): the empty-domain LANE FLOOR is 1.472e-06
#: and the ``volume`` row is 1.035e-05, so the gate carries ~190x headroom
#: over the number it gates. It is the same 2e-3 the vmap row above uses, and
#: for the same reason: these lanes run different kernels and different
#: float32 fusions over the same declaration, so the contract is "same
#: conductor", not "same arithmetic". The defect it has to reject is the one
#: this lane shipped until #1053 — the body absent altogether, which moves
#: the peak by 3.3e-01 relative (``volume`` against ``none``, measured on the
#: uniform lane), 166x the gate.
V2_LANE_REL_GATE = 2e-3


def test_the_shmap_distributed_lane_realizes_the_same_conductor():
    """#1053: ``sim.run(devices=...)`` realizes a declared PEC VOLUME.

    Until #1053 this lane assembled ``pec_mask``, dropped it, and refused
    rather than run a board without its metal. It now shards the mask and
    applies it in both step bodies at the #1041 ordering, so the volume row
    joins the battery. The seam matters on this fixture: at 2 devices the
    21-node domain pads to 22 and splits at x = 11 mm, and the conductor's
    4–16 mm footprint straddles that.

    A sheet and a sub-cell wire own no cell, the mask is the lane's only
    carrier, and nothing else there realizes them — so those stay refused,
    and this row asserts the refusal rather than leaving it to a unit test.
    A silent drop on either half is exactly what this file exists to catch.
    """
    import jax

    if jax.device_count() < 2:
        pytest.skip("needs 2 devices; see tests/unit/runners/"
                    "test_device_count_sentinel.py, which FAILS rather than "
                    "skips when the environment provides fewer")
    devs = jax.devices()[:2]

    single = _peak(_build("volume"))
    distributed = _peak(_build("volume"), devices=devs)
    assert distributed == pytest.approx(single, rel=V2_LANE_REL_GATE), (
        f"the shard_map lane reads {distributed:.8e} where the uniform lane "
        f"reads {single:.8e} for the same declared PEC volume")

    # teeth: the conductor must move THIS lane's own trace, so a lane that
    # dropped the body could not pass the comparison by arithmetic.
    empty = _peak(_build("none"), devices=devs)
    assert distributed != pytest.approx(empty, rel=100 * V2_LANE_REL_GATE), (
        f"the declared volume did not change the shard_map lane's trace: "
        f"{distributed:.8e} with the conductor, {empty:.8e} without")

    for kind in ("sheet", "thin"):
        with pytest.raises(NotImplementedError, match="SHEETS"):
            _peak(_build(kind), devices=devs)


def _adi_conductor(kind, mode="3d", **kwargs):
    """The measured slab classes, with an in-plane slab on TMz."""
    from rfx import PolylineWire

    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec",
                     solver="adi", mode=mode, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "wire":
            sim.add(PolylineWire(((10e-3, 10e-3, 8e-3),
                                  (10e-3, 10e-3, 14e-3)), radius=0.2e-3),
                    material="pec")
        else:
            thickness = {"sheet": 0, "volume": DX, "thick_volume": 3 * DX}[kind]
            if mode == "3d":
                lo, hi = (4e-3, 4e-3, 10e-3), (16e-3, 16e-3, 10e-3 + thickness)
            else:
                lo, hi = (4e-3, 10e-3, 0), (16e-3, 10e-3 + thickness, 20e-3)
            sim.add(Box(lo, hi), material="pec")
    sim.add_source(position=SRC if mode == "3d" else (10e-3, 5e-3, 0),
                   component="ez")
    sim.add_probe(position=PROBE if mode == "3d" else (10e-3, 15e-3, 0),
                  component="ez")
    return sim


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("kind", ["sheet", "volume", "thick_volume"])
@pytest.mark.parametrize("entrypoint", ["run", "forward"])
def test_adi_default_refuses_interior_pec_even_without_preflight(mode, kind, entrypoint):
    """The shipped factor must refuse before returning the measured overflow.

    A finite 200-step factor-1 trace was not a stability bound: longer
    measurements also grew at factor 1. No supported interior-PEC ADI
    factor is inferred from these short conductor-realization witnesses.
    """
    sim = _adi_conductor(kind, mode)
    assert sim._adi_cfl_factor == 5.0
    with pytest.raises(ValueError, match="adi_interior_pec_unsupported"):
        getattr(sim, entrypoint)(n_steps=4, skip_preflight=True)


@pytest.mark.parametrize("kind", ["sheet", "wire"])
@pytest.mark.parametrize("entrypoint", ["run", "forward"])
def test_adi_refusal_preserves_sheet_and_wire_forwarding(monkeypatch, kind, entrypoint):
    """A refusal must not conceal a return of #931's dropped-conductor bug.

    Both spies call production code: the declaration reaches the ADI
    helper, its shared owner realizes nonempty edges, and the solver then
    refuses. Neither an early declaration-only rejection nor dropping the
    sheet/wire arguments satisfies this contract.
    """
    import rfx.boundaries.pec as pec

    sim = _adi_conductor(kind)
    original_lane = sim._run_adi_from_materials
    original_owner = pec.realized_pec_edge_masks
    lane_calls = []
    owner_calls = []
    in_lane = False

    def owner_spy(*args, **kwargs):
        masks = original_owner(*args, **kwargs)
        if in_lane:
            owner_calls.append(masks)
        return masks

    def lane_spy(*args, **kwargs):
        nonlocal in_lane
        lane_calls.append(kwargs)
        in_lane = True
        try:
            return original_lane(*args, **kwargs)
        finally:
            in_lane = False

    monkeypatch.setattr(pec, "realized_pec_edge_masks", owner_spy)
    monkeypatch.setattr(sim, "_run_adi_from_materials", lane_spy)
    with pytest.raises(ValueError, match="adi_interior_pec_unsupported"):
        getattr(sim, entrypoint)(n_steps=4, skip_preflight=True)

    assert len(lane_calls) == 1
    assert lane_calls[0]["pec_mask"] is None
    assert len(lane_calls[0]["pec_sheets" if kind == "sheet" else "pec_wires"]) == 1
    assert len(owner_calls) == 1
    assert any(np.any(np.asarray(mask)) for mask in owner_calls[0])


def test_the_adi_forward_lane_refuses_the_thin_conductor_it_cannot_realize():
    """``add_thin_conductor`` has no carrier on the ADI lane, so both entry
    points refuse it by name rather than solving a board without its metal."""
    sim = _build("thin", solver="adi")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="thin-conductor"):
            sim.run(n_steps=4, skip_preflight=True)
        with pytest.raises(ValueError, match="thin-conductor"):
            _build("thin", solver="adi").forward(n_steps=4, skip_preflight=True)


def test_the_subgridded_lane_refuses_a_sheet_it_cannot_realize():
    """No silent drop: the SBP-SAT lane applies PEC from cell masks on two
    grids, so a sheet has no carrier there."""
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add(Box((FOOT_LO[0], FOOT_LO[1], Z_PLANE),
                    (FOOT_HI[0], FOOT_HI[1], Z_PLANE)), material="pec")
        sim.add_source(position=SRC, component="ez")
        sim.add_refinement(z_range=(6e-3, 14e-3), ratio=2)
        with pytest.raises(NotImplementedError, match="PEC sheets"):
            sim.run(n_steps=4, skip_preflight=True)


# ---------------------------------------------------------------------------
# the lossy (f0) sheet ctx on the forward lane (§1.7 one spelling, §1.9)
# ---------------------------------------------------------------------------

F0_DX = 2e-3
F0_DOM = (12e-3, 12e-3, 12e-3)


def test_the_forward_lossy_sheet_ctx_knows_about_a_pec_wire():
    """The lossy operator REPLACES the E update at its edges, so an edge a
    PEC conductor owns has to be removed from its ctx first.

    ``forward()`` asked "is there any PEC?" of ``pec_mask`` and the sheet list
    only. A filament owns no cell and is not a sheet, so a model whose only
    conductor is a wire reached the operator with ``pec_edge_masks=None`` and
    the operator wrote field back onto the wire's own PEC edge: measured, an
    Ex probe ON the filament read 0 through ``run()`` and 2.44e-6 through
    ``forward()``.
    """
    from rfx import PolylineWire

    def _build_wire():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(freq_max=15e9, domain=F0_DOM, dx=F0_DX,
                             boundary="pec")
            sim.add(PolylineWire(((4e-3, 6e-3, 6e-3), (8e-3, 6e-3, 6e-3)),
                                 radius=0.2e-3), material="pec")
            sim.add_thin_conductor(
                Box((2e-3, 2e-3, 6e-3), (10e-3, 10e-3, 6e-3)),
                sigma_bulk=5.8e7, surface_impedance_f0=10e9)
            sim.add_source(position=(4e-3, 4e-3, 2e-3), component="ez",
                           amplitude_kind="field")
            sim.add_probe(position=(5e-3, 6e-3, 6e-3), component="ex")
        return sim

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_ts = np.asarray(
            _build_wire().run(n_steps=60, skip_preflight=True).time_series)
        fwd_ts = np.asarray(_build_wire().forward(n_steps=60).time_series)
    run_peak = float(np.max(np.abs(run_ts[:, 0])))
    fwd_peak = float(np.max(np.abs(fwd_ts[:, 0])))
    assert run_peak == 0.0, run_peak
    assert fwd_peak == 0.0, (
        f"forward() left {fwd_peak:g} on a PEC filament's own edge: the lossy "
        "sheet ctx was built without the wires")


def test_the_forward_lossy_sheet_ctx_uses_the_runs_periodic_flags():
    """#689 on the OUTER call: the builder realizes the f0 footprint's own
    edges, and on a periodic axis the seam edge (node n-1 to node 0) is
    inside the sheet. ``forward()`` passed the flags to its PEC realization
    and not to the ctx, so the seam carried no loss — measured, 35 loaded Ex
    edges through ``run()`` against 30 through ``forward()`` on the same
    x-periodic board.
    """
    import rfx.materials.thin_conductor as _tc
    from rfx.boundaries.spec import Boundary, BoundarySpec

    def _build_periodic():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(
                freq_max=15e9, domain=F0_DOM, dx=F0_DX,
                boundary=BoundarySpec(
                    x=Boundary(lo="periodic", hi="periodic"),
                    y=Boundary(lo="pec", hi="pec"),
                    z=Boundary(lo="pec", hi="pec")))
            sim.add_thin_conductor(
                Box((0.0, 2e-3, 6e-3), (12e-3, 10e-3, 6e-3)),
                sigma_bulk=5.8e7, surface_impedance_f0=10e9)
            sim.add_source(position=(4e-3, 4e-3, 2e-3), component="ez",
                           amplitude_kind="field")
            sim.add_probe(position=(4e-3, 6e-3, 6e-3), component="ex")
        return sim

    seen: list = []
    _orig = _tc.build_sheet_impedance_ctx

    def _record(*args, **kwargs):
        ctx = _orig(*args, **kwargs)
        seen.append(ctx)
        return ctx

    _tc.build_sheet_impedance_ctx = _record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_res = _build_periodic().run(n_steps=60, skip_preflight=True)
            run_ctx = seen[-1]
            fwd_res = _build_periodic().forward(n_steps=60)
            fwd_ctx = seen[-1]
    finally:
        _tc.build_sheet_impedance_ctx = _orig

    for comp in ("mask_ex", "mask_ey", "mask_ez"):
        run_m = np.asarray(getattr(run_ctx, comp), dtype=bool)
        fwd_m = np.asarray(getattr(fwd_ctx, comp), dtype=bool)
        assert np.array_equal(run_m, fwd_m), (
            f"{comp}: run() loads {int(run_m.sum())} f0 edges, forward() "
            f"{int(fwd_m.sum())} — the ctx was built with different #689 flags")
    # the seam edge really is in the sheet, so the comparison has teeth
    seam = np.asarray(run_ctx.mask_ex, dtype=bool)[-1]
    assert seam.any(), "no seam edge in the loaded set — the pin is vacuous"

    run_peak = float(np.max(np.abs(np.asarray(run_res.time_series)[:, 0])))
    fwd_peak = float(np.max(np.abs(np.asarray(fwd_res.time_series)[:, 0])))
    assert fwd_peak == pytest.approx(run_peak, rel=1e-6), (run_peak, fwd_peak)
