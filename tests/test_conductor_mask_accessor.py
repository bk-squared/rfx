"""Issue #695 — one accessor for the WHOLE conductor footprint.

Since #677 a ``surface_impedance_f0`` thin conductor is a node-thin
per-step operator: it appears in NEITHER ``pec_mask`` NOR
``materials.sigma``.  A conductor-connectivity check written the obvious
way (``pec_mask | (sigma > 1e3)``) therefore finds nothing and reports a
healthy model as disconnected.  ``Simulation.conductor_mask()`` /
``rfx.materials.thin_conductor.conductor_footprint`` is the one spelling
that covers PEC cells, sigma-promoted cells AND f0 sheet cells.

Binding note: every assertion below is written so that DELETING the sheet
term from ``conductor_footprint`` (the ``for m in sheet_masks`` loop)
turns it red — see the module's mutation record in the PR body.

NU fixtures here use a GRADED dz_profile with a grading ratio different
from the other axes; a uniform-valued profile exercises the NU code path
without ever exercising its metrics.
"""

import numpy as np
import pytest

from rfx import Simulation
from rfx.geometry.csg import Box
from rfx.materials.thin_conductor import (
    CONDUCTOR_SIGMA_THRESHOLD,
    conductor_footprint,
)


F0 = 10e9
SHEET_LO = (1e-3, 1e-3, 1.5e-3)
SHEET_HI = (5e-3, 5e-3, 1.5e-3)


def _sheet_sim(**kw):
    sim = Simulation(freq_max=F0, domain=(6e-3, 6e-3, 3e-3), dx=0.25e-3,
                     boundary="cpml", cpml_layers=6, **kw)
    with pytest.warns(UserWarning, match="Leontovich"):
        sim.add_thin_conductor(
            Box(SHEET_LO, SHEET_HI),
            sigma_bulk=5.8e7, thickness=35e-6, surface_impedance_f0=F0,
        )
    return sim


def _graded_dz(n=20, dz0=0.08e-3, ratio_step=1.06):
    """Graded z profile spanning ~3 mm in 20 cells.

    The ratio is deliberately unlike any x/y grading and unlike 1.0: a
    UNIFORM-valued dz_profile takes the NU code path without exercising a
    single NU metric (primal == dual everywhere), which is how NU defects
    hide in this suite.
    """
    return dz0 * ratio_step ** np.arange(n, dtype=float)


def test_naive_spelling_finds_nothing_but_accessor_finds_the_sheet():
    """The measured #695 symptom, pinned in both directions."""
    sim = _sheet_sim()
    grid = sim._build_grid()
    specs: list = []
    mats, _, _, pec_mask, _, _, _ = sim._assemble_materials(
        grid, sheet_specs=specs)

    # The assembled arrays really are sheet-free (this is #677 working).
    assert pec_mask is None
    assert int((np.asarray(mats.sigma) > CONDUCTOR_SIGMA_THRESHOLD).sum()) == 0
    assert len(specs) == 1

    naive = np.zeros(grid.shape, bool)
    naive |= np.asarray(mats.sigma) > CONDUCTOR_SIGMA_THRESHOLD
    assert int(naive.sum()) == 0, "naive spelling should find no metal"

    full = np.asarray(sim.conductor_mask())
    assert full.shape == tuple(grid.shape)
    n_sheet = int(np.asarray(specs[0].mask).sum())
    assert n_sheet > 0
    assert int(full.sum()) == n_sheet, (
        "conductor_mask must contain exactly the sheet's cell footprint "
        f"({n_sheet} cells); got {int(full.sum())}")
    np.testing.assert_array_equal(full, np.asarray(specs[0].mask))


def test_accessor_unions_pec_and_sigma_and_sheet():
    """All three contributions present at once, counted exactly."""
    sim = _sheet_sim()
    sim.add_material("copper_bulk", eps_r=1.0, sigma=5.8e7)
    sim.add(Box((1e-3, 1e-3, 0.5e-3), (2e-3, 2e-3, 0.75e-3)),
                     material="copper_bulk")
    sim.add_material("lossy", eps_r=4.0, sigma=1e4)
    sim.add(Box((3e-3, 3e-3, 0.5e-3), (4e-3, 4e-3, 0.75e-3)),
                     material="lossy")
    grid = sim._build_grid()
    specs: list = []
    mats, _, _, pec_mask, _, _, _ = sim._assemble_materials(
        grid, sheet_specs=specs)

    pec = np.asarray(pec_mask)
    sig = np.asarray(mats.sigma) > CONDUCTOR_SIGMA_THRESHOLD
    sheet = np.asarray(specs[0].mask)
    expected = pec | sig | sheet
    assert int(pec.sum()) > 0
    assert int(sig.sum()) > 0
    assert int(sheet.sum()) > 0
    # The sheet contributes cells NO other term has — this is what makes
    # the assertion below bind on the sheet term.
    assert int((sheet & ~(pec | sig)).sum()) > 0

    got = np.asarray(sim.conductor_mask())
    np.testing.assert_array_equal(got, expected)
    assert int(got.sum()) == int(expected.sum())
    # And the naive spelling misses exactly the sheet-only cells.
    assert int(got.sum()) - int((pec | sig).sum()) == \
        int((sheet & ~(pec | sig)).sum())


def test_accessor_uses_the_nonuniform_grid_on_a_graded_mesh():
    """On a graded-dz sim the accessor rasterizes on the NU grid."""
    dz = _graded_dz()
    sim = _sheet_sim(dz_profile=dz)
    nu = sim._build_nonuniform_grid()
    uni = sim._build_grid()
    assert tuple(nu.shape) != tuple(uni.shape), (
        "fixture is not exercising the NU/uniform shape difference")
    # The profile really is graded (a uniform-valued profile would take
    # the NU code path without exercising any NU metric).
    assert float(np.max(dz) / np.min(dz)) > 1.5

    got = np.asarray(sim.conductor_mask())
    assert got.shape == tuple(nu.shape)
    assert int(got.sum()) > 0, (
        "the f0 sheet must be visible on the NU lane too")


def test_conductor_footprint_refuses_an_empty_call():
    with pytest.raises(ValueError, match="nothing to union"):
        conductor_footprint()
    seeded = np.asarray(conductor_footprint(shape=(2, 3, 4)))
    assert seeded.shape == (2, 3, 4)
    assert not seeded.any()


def test_conductor_footprint_sheet_term_is_load_bearing():
    """Direct unit pin on the sheet term of the shared helper."""
    pec = np.zeros((4, 4, 4), bool)
    sigma = np.zeros((4, 4, 4), float)
    sheet = np.zeros((4, 4, 4), bool)
    sheet[1:3, 1:3, 2] = True
    got = np.asarray(conductor_footprint(pec_mask=pec, sigma=sigma,
                                         sheet_masks=[sheet]))
    assert int(got.sum()) == 4
    np.testing.assert_array_equal(got, sheet)


def test_refplane_call_site_sees_the_sheet():
    """rfx's own reference-plane trace scan uses the full footprint.

    ``build_wire_refplane_specs`` BFSes a conducting cross-section. Handed
    the bare ``pec_mask`` (None for a sheet-traced board) it raised "no
    conductor" on a healthy model; ``_refplane_conductor_mask`` is what
    the run path now hands it.
    """
    from rfx.api._execute import _refplane_conductor_mask
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx

    sim = _sheet_sim()
    grid = sim._build_grid()
    specs: list = []
    _mats, _, _, pec_mask, _, _, _ = sim._assemble_materials(
        grid, sheet_specs=specs)
    ctx = build_sheet_impedance_ctx(specs, pec_mask=pec_mask)
    assert ctx is not None
    assert pec_mask is None, "fixture must have no PEC at all"

    got = _refplane_conductor_mask(pec_mask, ctx)
    assert got is not None, "the sheet-traced board must not read as no-metal"
    got = np.asarray(got)
    assert int(got.sum()) == int(np.asarray(specs[0].mask).sum())
    # No sheet ctx -> unchanged passthrough (no behaviour change for the
    # PEC-traced boards this path was written for).
    assert _refplane_conductor_mask(pec_mask, None) is pec_mask


# ---------------------------------------------------------------------------
# #695 through the PUBLIC path.
#
# ``test_refplane_call_site_sees_the_sheet`` above calls
# ``_refplane_conductor_mask`` directly, so it pins the HELPER and not the
# wiring: reverting ``rfx/api/_execute.py``'s call site to ``pec_mask=pec_mask``
# left it green while the reported field failure came straight back. The tests
# below enter through ``Simulation.run(compute_s_params=True)`` and through the
# per-drive call the production S-matrix driver makes, i.e. through the line
# that was changed.
#
# The board is the committed refplane thru fixture
# (tests/locks/test_refplane_port_waves.py) with ONE substitution: the PEC signal
# trace becomes a ``surface_impedance_f0`` sheet at the same z. Everything
# else — dx, domain, port positions, N — is byte-identical, which is what
# lets the plane geometry below be compared against that file's hand-derived
# Phase-0 indices.
# ---------------------------------------------------------------------------

_RP_DX = 0.5e-3
_RP_DOMAIN = (0.032, 0.020, 0.010)
_RP_H = 1.0e-3
_RP_W = 5.0e-3
_RP_X1, _RP_X2 = 0.008, 0.024
_RP_Y_MID = _RP_DOMAIN[1] / 2
_RP_FREQS = np.array([4e9, 5e9])
_RP_N = 3


def _refplane_thru(trace: str):
    """The committed thru with a ``sheet`` / ``pec`` / ``none`` signal trace.

    ``none`` is the in-test CONTROL: with no metal at all the plane V/I
    method genuinely cannot work and must say so. It proves the raise the
    other two avoid is live and reachable on this exact fixture, so their
    passing is the sheet being SEEN and not the check being absent.
    """
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(
        freq_max=10e9, domain=_RP_DOMAIN, dx=_RP_DX,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=8,
    )
    lo = (_RP_X1 - _RP_DX, _RP_Y_MID - _RP_W / 2, _RP_H)
    hi = (_RP_X2 + _RP_DX, _RP_Y_MID + _RP_W / 2, _RP_H)
    if trace == "sheet":
        with pytest.warns(UserWarning, match="Leontovich"):
            sim.add_thin_conductor(
                Box(lo, hi), sigma_bulk=5.8e7, thickness=35e-6,
                surface_impedance_f0=5e9)
    elif trace == "pec":
        sim.add(Box(lo, (hi[0], hi[1], hi[2] + _RP_DX)), material="pec")
    elif trace != "none":
        raise AssertionError(trace)
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    for pos, direction in (((_RP_X1, _RP_Y_MID, 0.0), "-x"),
                           ((_RP_X2, _RP_Y_MID, 0.0), "+x")):
        sim.add_port(position=pos, component="ez", impedance=50.0,
                     extent=_RP_H, waveform=pulse, direction=direction,
                     reference_plane_cells=_RP_N)
    return sim


def _run_thru(trace: str):
    return _refplane_thru(trace).run(
        n_steps=12, compute_s_params=True, s_param_freqs=_RP_FREQS,
        s_param_n_steps=12, skip_preflight=True)


def test_public_run_on_a_sheet_traced_refplane_board():
    """``run(compute_s_params=True)`` must not call a sheet trace 'no metal'.

    This is the reported #695 failure, entered the way a user hits it. The
    fix is ``_execute.py``'s ``pec_mask=_refplane_conductor_mask(pec_mask,
    sheet_impedance)``; with that argument reverted to ``pec_mask`` this
    run raises ``ValueError: reference_plane_cells: the simulation has no
    conductor at the reference planes``.
    """
    res = _run_thru("sheet")
    S = np.asarray(res.s_params)
    assert S.shape == (2, 2, len(_RP_FREQS))
    assert np.all(np.isfinite(S)), S


def test_public_run_pec_trace_unchanged_and_no_trace_still_refuses():
    """The two controls that make the test above mean something.

    ``pec``: the boards this path was written for are untouched.
    ``none``: with no conductor anywhere the refusal still fires, so the
    sheet board's success is the sheet being found — not a check that
    quietly stopped running.
    """
    S = np.asarray(_run_thru("pec").s_params)
    assert S.shape == (2, 2, len(_RP_FREQS))
    assert np.all(np.isfinite(S)), S

    with pytest.raises(ValueError, match="no conductor at the reference"):
        _run_thru("none")


def test_driver_drive_pass_registers_planes_on_the_sheet_trace():
    """The per-drive call the S-matrix driver makes registers FOUR planes,
    and their Ampere loops hug the SHEET trace.

    The leg/span indices asserted here are the hand-derived Phase-0 values
    pinned for the PEC thru in
    ``tests/locks/test_refplane_port_waves.py::
    test_refplane_registers_two_planes_per_port_with_phase0_geometry``.
    They can only come out equal if the cross-section BFS found the sheet
    at the same cells the PEC box occupies — a fallback or a partial mask
    would move them.
    """
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx

    sim = _refplane_thru("sheet")
    grid = sim._build_grid()
    specs: list = []
    mats, dsp, lsp, pec_mask, _, _, _ = sim._assemble_materials(
        grid, sheet_specs=specs)
    assert pec_mask is None, "fixture must carry NO pec_mask at all"
    ctx = build_sheet_impedance_ctx(specs, pec_mask=pec_mask)
    assert ctx is not None

    raw = sim._forward_from_materials(
        grid, mats, dsp, lsp, n_steps=8, checkpoint=False, pec_mask=pec_mask,
        port_s11_freqs=_RP_FREQS, _sparam_drive_idx=0,
        _return_raw_port_sparams=True, sheet_impedance=ctx)

    rp = raw["wire_refplane"]
    assert rp is not None and len(rp) == 4, (
        f"expected 2 ports x 2 planes, got {0 if rp is None else len(rp)}")
    by_key = {(s.port_index, s.plane_slot): s for s, _ in rp}
    assert set(by_key) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert {k: by_key[k].plane_index for k in by_key} == {
        (0, 0): 27, (0, 1): 30, (1, 0): 53, (1, 1): 50}
    for key, spec in by_key.items():
        # Trace bbox -> Ampere loop, byte-equal to the PEC thru's pins.
        assert (spec.u_lo_leg, spec.u_hi_leg) == (22, 33), key
        assert (spec.v_lo_leg, spec.v_hi_leg) == (1, 3), key
        assert (spec.u_span_lo, spec.u_span_hi) == (23, 34), key
        assert (spec.v_span_lo, spec.v_span_hi) == (2, 4), key
