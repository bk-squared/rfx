"""Replay gate for the NU waveguide boundary-vs-local-cell beta table.

The nonuniform waveguide S-matrix lane evaluates beta for the reference-plane
shift at the grid's BOUNDARY cell (``NonUniformGrid.dx``), not the cell the
plane sits in. ``scripts/diagnostics/waveguide_nu_beta_cell_size_envelope.py``
evaluates the production ``_compute_beta`` / ``_compute_mode_impedance`` at
both cell sizes of the committed NU AD fixture (1.5 mm boundary, 0.75 mm
fine) over 8--12 GHz and writes
``tests/fixtures/waveguide_nu_beta_cell_size_envelope.json``. This module
replays that JSON:

1. against the closed form -- the Yee correction is second order,
   ``delta_beta = s_x^3 (dx_b^2 - dx_l^2)/24`` with the next arcsin-series
   term as the tolerance, and ``Z_TE`` is cell-size independent because the
   discrete ``sin(beta*dx/2)`` equals ``s_x*dx/2`` by construction;
2. against the LIVE functions and the LIVE fixture (no FDTD: the port config
   builder only), so the committed numbers cannot drift from the code;
3. for the headline the support matrix quotes: under 1 degree over a 20 mm
   plane offset, and both fixture spans in uniform 1.5 mm cells.

Pure arithmetic, no time stepping, no settling witness applies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tests._x64_compat import enable_x64

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling fixture module

FIXTURE = REPO / "tests" / "fixtures" / "waveguide_nu_beta_cell_size_envelope.json"

C0 = 299_792_458.0


@pytest.fixture(scope="module")
def envelope() -> dict:
    return json.loads(FIXTURE.read_text())


def _closed_form(f_hz, f_cutoff_hz, dt_s, dx_b, dx_l):
    omega = 2.0 * np.pi * f_hz
    kc = 2.0 * np.pi * f_cutoff_hz / C0
    s_t_over_c = np.sin(omega * 0.5 * dt_s) / (C0 * 0.5 * dt_s)
    s_x = np.sqrt(s_t_over_c ** 2 - kc ** 2)
    lead = s_x ** 3 * (dx_b ** 2 - dx_l ** 2) / 24.0
    nxt = 3.0 * s_x ** 5 * (dx_b ** 4 - dx_l ** 4) / 640.0
    return s_x, lead, nxt


def test_fixture_shape_and_provenance(envelope):
    assert envelope["no_fdtd_run"] is True
    assert envelope["settling_db"] is None
    assert envelope["producer"] == "scripts/diagnostics/waveguide_nu_beta_cell_size_envelope.py"
    assert envelope["fixture"].startswith("tests/test_waveguide_nu_flux_ad.py")
    rows = envelope["rows"]
    f = np.array([r["f_hz"] for r in rows])
    assert f.min() <= 8e9 and f.max() >= 12e9, f
    assert len(rows) >= 5
    assert envelope["beta_inputs"]["dx_boundary_m"] == pytest.approx(1.5e-3)
    assert envelope["beta_inputs"]["dx_local_m"] == pytest.approx(0.75e-3)
    assert envelope["grid"]["boundary_dx_m"] == pytest.approx(1.5e-3)


def test_delta_beta_matches_yee_second_order_closed_form(envelope):
    """(beta dx)^2/24 at the two cell sizes explains the whole difference."""
    inp = envelope["beta_inputs"]
    for r in envelope["rows"]:
        s_x, lead, nxt = _closed_form(
            r["f_hz"], inp["f_cutoff_hz"], inp["dt_s"], inp["dx_boundary_m"], inp["dx_local_m"])
        assert s_x == pytest.approx(r["s_x_rad_per_m"], rel=1e-12)
        assert lead == pytest.approx(r["delta_beta_closed_form_rad_per_m"], rel=1e-12)
        assert nxt == pytest.approx(r["delta_beta_next_order_rad_per_m"], rel=1e-12)
        d = r["delta_beta_rad_per_m"]
        assert d > 0.0, r  # the coarser cell always carries the larger beta
        # Leading order within the next-order term (x1.5 margin) ...
        assert abs(d - lead) <= 1.5 * nxt + 1e-9, (r["f_hz"], d, lead, nxt)
        # ... and the next-order term closes the residual to a few percent of itself.
        assert abs(d - lead - nxt) <= 0.05 * nxt + 1e-6, (r["f_hz"], d - lead - nxt, nxt)
        # The recorded beta values are the discrete relation, not the continuum
        # one: the coarser cell carries the larger beta. (No ordering against the
        # continuum value is asserted -- the temporal term lowers s_x below k_x,
        # so at this fixture's dt both discrete betas sit BELOW the continuum
        # one, the 1.5 mm value closer to it than the 0.75 mm value.)
        assert r["beta_boundary_cell_rad_per_m"] > r["beta_local_cell_rad_per_m"]
        assert r["beta_continuous_rad_per_m"] > 0.0
        # Phase columns are delta_beta times the recorded offsets.
        for key, off in (
            ("delta_phi_over_applied_shift_deg", inp["applied_shift_m"]),
            ("delta_phi_over_port_to_reference_offset_deg", inp["port_to_reference_offset_m"]),
            ("delta_phi_over_nominal_20mm_deg", inp["nominal_offset_m"]),
        ):
            assert r[key] == pytest.approx(np.degrees(d * off), rel=1e-12, abs=1e-15)


def test_mode_impedance_is_cell_size_independent(envelope):
    """Discrete Z_TE = mu0 dx sin(w dt/2)/(dt sin(beta dx/2)); sin(beta dx/2) = s_x dx/2 cancels dx."""
    for r in envelope["rows"]:
        assert r["z_te_boundary_cell_ohm"] > 0.0
        assert r["z_te_rel_diff"] <= 1e-12, r
        assert abs(r["z_te_boundary_cell_ohm"] - r["z_te_local_cell_ohm"]) <= 1e-9 * r["z_te_boundary_cell_ohm"]
    assert envelope["headline"]["max_z_te_rel_diff"] <= 1e-12


def test_effect_is_resolved_at_the_production_float32_precision(envelope):
    """The lane runs float32; the recorded float32 column must still show the effect."""
    for r in envelope["rows"]:
        d32 = r["delta_beta_float32_rad_per_m"]
        d64 = r["delta_beta_rad_per_m"]
        # float32 beta ~ 1e2 rad/m carries ~1e-5 rad/m per ulp; a few ulp per
        # operand is far below the smallest recorded difference (~0.08 rad/m).
        assert abs(d32 - d64) <= 1e-3, (r["f_hz"], d32, d64)
        assert d32 > 0.5 * d64


def test_live_functions_reproduce_the_committed_beta(envelope):
    """_compute_beta / _compute_mode_impedance at the JSON's inputs give the JSON's values."""
    from rfx.sources.waveguide_port import _compute_beta, _compute_mode_impedance
    inp = envelope["beta_inputs"]
    with enable_x64():
        for r in envelope["rows"]:
            f = jnp.asarray([r["f_hz"]], dtype=jnp.float64)
            bb = float(np.real(np.asarray(_compute_beta(f, inp["f_cutoff_hz"], dt=inp["dt_s"], dx=inp["dx_boundary_m"])))[0])
            bl = float(np.real(np.asarray(_compute_beta(f, inp["f_cutoff_hz"], dt=inp["dt_s"], dx=inp["dx_local_m"])))[0])
            bc = float(np.real(np.asarray(_compute_beta(f, inp["f_cutoff_hz"])))[0])
            zb = float(np.real(np.asarray(_compute_mode_impedance(f, inp["f_cutoff_hz"], "TE", dt=inp["dt_s"], dx=inp["dx_boundary_m"])))[0])
            zl = float(np.real(np.asarray(_compute_mode_impedance(f, inp["f_cutoff_hz"], "TE", dt=inp["dt_s"], dx=inp["dx_local_m"])))[0])
            assert bb == pytest.approx(r["beta_boundary_cell_rad_per_m"], rel=1e-9)
            assert bl == pytest.approx(r["beta_local_cell_rad_per_m"], rel=1e-9)
            assert bc == pytest.approx(r["beta_continuous_rad_per_m"], rel=1e-9)
            assert zb == pytest.approx(r["z_te_boundary_cell_ohm"], rel=1e-9)
            assert zl == pytest.approx(r["z_te_local_cell_ohm"], rel=1e-9)


def test_live_fixture_reproduces_the_committed_inputs(envelope):
    """The JSON's cutoff, dt, boundary dx and plane positions are the fixture's own (no FDTD)."""
    from test_waveguide_nu_flux_ad import NUM_PERIODS, _FREQS, _wr90_nu_sim
    from rfx.api._sparams import _assert_nu_shift_span_in_one_grading_zone
    from rfx.runners.nonuniform import _build_waveguide_port_config_nu
    from rfx.sources.waveguide_port import waveguide_plane_positions

    sim, domain_x = _wr90_nu_sim()
    grid = sim._build_nonuniform_grid()
    assert float(grid.dx) == pytest.approx(envelope["grid"]["boundary_dx_m"], rel=1e-12)
    assert float(grid.dt) == pytest.approx(envelope["grid"]["dt_s"], rel=1e-9)
    assert float(domain_x) == pytest.approx(envelope["grid"]["domain_x_m"], rel=1e-9)
    n_steps = int(np.ceil(NUM_PERIODS / float(sim._freq_max) / float(grid.dt)))
    by_name = {p["name"]: p for p in envelope["ports"]}
    for entry in sim._waveguide_ports:
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, jnp.asarray(_FREQS), n_steps)
        planes = waveguide_plane_positions(cfg)
        rec = by_name[entry.name]
        assert float(cfg.f_cutoff) == pytest.approx(envelope["beta_inputs"]["f_cutoff_hz"], rel=1e-9)
        assert float(cfg.dx) == pytest.approx(rec["cfg_dx_m"], rel=1e-12)
        assert float(cfg.dt) == pytest.approx(rec["cfg_dt_s"], rel=1e-9)
        assert planes["source"] == pytest.approx(rec["port_plane_m"], abs=1e-9)
        assert planes["reference"] == pytest.approx(rec["modal_record_plane_m"], abs=1e-9)
        desired = float(entry.reference_plane)
        axis, lo, hi, sizes = _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired, entry.name)
        assert axis == rec["span_axis"] == "x"
        np.testing.assert_allclose(sizes, rec["cells_crossed_m"], rtol=1e-12)


def test_headline_envelope_the_support_matrix_quotes(envelope):
    """Under 1 degree over 20 mm; both fixture spans in uniform 1.5 mm cells."""
    head = envelope["headline"]
    assert 0.0 < head["max_delta_phi_over_nominal_20mm_deg"] < 1.0, head
    assert head["max_delta_phi_over_applied_shift_deg"] < 0.05, head
    assert head["both_spans_in_one_cell_size"] is True
    for p in envelope["ports"]:
        assert p["distinct_cell_sizes_crossed_m"] == pytest.approx([1.5e-3])
        assert len(p["cells_crossed_m"]) >= 3, p
    # The fixture's graded block starts at x = 0.030 m; the coarse blocks are
    # 0.030 m long, and both ports keep their planes inside them.
    assert envelope["grid"]["first_non_coarse_cell_starts_x_m"] == pytest.approx(0.030, abs=1e-9)
    by_f = {round(r["f_hz"] / 1e9, 2): r for r in envelope["rows"]}
    for ghz, lo, hi in ((8.0, 0.05, 0.15), (10.0, 0.25, 0.45), (12.0, 0.7, 0.95)):
        assert lo < by_f[ghz]["delta_phi_over_nominal_20mm_deg"] < hi, (ghz, by_f[ghz])
