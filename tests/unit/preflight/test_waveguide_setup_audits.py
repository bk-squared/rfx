"""Falsifiers for the two waveguide S-parameter setup audits (post-v1.8 item 2).

``docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md``
sections 2-1 and 2-4. Both checks speak in INPUT units — times, lengths,
indices, and the knob value that changes them — so every number a message
prints can be recomputed here from the fixture's declared dimensions and the
grid the runner builds, with no FDTD anywhere in this file.

The independent oracles used below:

* the TE10 discrete cutoff of an ``N``-cell PEC guide,
  ``(c / 2a) * sinc(pi / 2N)``, computed from ``A_M`` and the rung's ``N``
  rather than read back off the port config (measured agreement with
  ``WaveguidePortConfig.f_cutoff``: 2.5e-14 relative);
* ``far_path``, ``v_g``, ``tau_far`` and ``T`` recomputed from the fixture
  constants and ``Grid.dt`` / ``Grid.num_timesteps``;
* the mirror sums recomputed from the runner's OWN plane arithmetic
  (``_build_waveguide_port_config`` for ``ref_x`` / ``probe_x``,
  ``apply_waveguide_port_e`` / ``apply_waveguide_port_h`` for the two source
  planes), never from a literal typed into this file.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._preflight import WAVEGUIDE_DEFAULT_NUM_PERIODS

from tests import _waveguide_chain_battery_fixture as F


C0 = 299_792_458.0
RUNG_M = F.DX_LADDER[2]          # a / 36, the fine rung of the battery ladder
RUNG_N = F.N_LADDER[2]


def _sinc(x: float) -> float:
    return 1.0 if x == 0.0 else math.sin(x) / x


def _analytic_te10_cutoff_hz(n_cells: int) -> float:
    """Discrete TE10 cutoff of an ``n_cells``-wide PEC guide of width ``A_M``."""
    return (C0 / (2.0 * F.A_M)) * _sinc(math.pi / (2 * n_cells))


def _built(dut: str = "thru", dx: float = RUNG_M, num_periods: float = 20.0):
    """``(sim, grid, cfgs, n_steps)`` through the RUNNER's own builders."""
    sim = F.build_simulation(dut, dx)
    grid = sim._build_grid()
    n_steps = int(grid.num_timesteps(num_periods))
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(F.FREQS), n_steps)
            for e in sim._waveguide_ports]
    return sim, grid, cfgs, n_steps


def _collect(sim, method: str, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        getattr(sim, method)(warnings, **kwargs)
    return [w.message for w in rec]


def _codes(msgs):
    return [getattr(m, "code", None) for m in msgs]


def _number_after(text: str, marker: str) -> float:
    """First float printed after ``marker`` in a check message."""
    tail = text.split(marker, 1)[1]
    return float(re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", tail).group(0))


# ---------------------------------------------------------------------------
# (a) record length vs the far-boundary round trip
# ---------------------------------------------------------------------------

def test_analytic_cutoff_is_an_independent_oracle_for_the_port_config():
    """The sinc cutoff this file computes IS the port's own discrete cutoff.

    Everything below leans on it, so it is asserted first: if this drifts, the
    tau_far numbers checked later are being compared against the wrong band
    edge, not against a wrong check.
    """
    _sim, _grid, cfgs, _n = _built()
    for cfg in cfgs:
        assert float(cfg.f_cutoff) == pytest.approx(
            _analytic_te10_cutoff_hz(RUNG_N), rel=1e-6)


def test_record_shorter_than_one_round_trip_is_error_severity():
    num_periods = 10.0
    sim, grid, cfgs, n_steps = _built(num_periods=num_periods)

    # --- hand computation, from the fixture's declared dimensions ---------
    f_min = float(min(F.FREQS))
    f_c = _analytic_te10_cutoff_hz(RUNG_N)
    v_g = C0 * math.sqrt(1.0 - (f_c / f_min) ** 2)
    pad_m = int(grid.pad_x_hi) * float(grid.dx)
    far_path = (F.DOMAIN_X_M - F.PORT_LEFT_X_M) + pad_m
    tau_far = 2.0 * far_path / v_g
    T = n_steps * float(grid.dt)
    ratio = T / tau_far
    assert ratio < 1.0, ratio          # the regime this test is about

    msgs = _collect(sim, "_validate_cfg_record_vs_far_boundary",
                    freqs=F.FREQS, num_periods=num_periods,
                    grid=grid, cfgs=cfgs, n_steps=n_steps)
    hits = [m for m in msgs
            if m.code == "record_shorter_than_far_boundary_round_trip"]
    assert len(hits) == 2, _codes(msgs)      # one per port
    assert all(m.severity == "error" for m in hits)

    left = next(m for m in hits if m.loc == "waveguide_port[0]")
    text = str(left)
    assert "ends BEFORE the far-boundary round trip" in text
    assert _number_after(text, "T/tau_far = ") == pytest.approx(ratio, rel=1e-4)
    assert _number_after(text, "far_path = ") == pytest.approx(
        far_path * 1e3, rel=1e-4)
    assert _number_after(text, "tau_far = 2 x far_path / v_g = ") == pytest.approx(
        tau_far * 1e9, rel=1e-4)
    assert _number_after(text, "T = ") == pytest.approx(T * 1e9, rel=1e-4)
    assert _number_after(text, "v_g(f_min)/c = ") == pytest.approx(
        v_g / C0, rel=1e-4)
    assert _number_after(text, "f_min = ") == pytest.approx(f_min / 1e9, rel=1e-4)
    assert _number_after(text, "cutoff f_c = ") == pytest.approx(f_c / 1e9, rel=1e-4)
    assert _number_after(text, "absorber pad ") == pytest.approx(
        pad_m * 1e3, rel=1e-4)
    # The remedy is a knob value, in the knob's own units.
    required = int(math.ceil(3.0 * tau_far * float(grid.freq_max)))
    assert f"num_periods >= {required} makes T/tau_far >= 3" in text
    # ... and it is the value that actually clears the threshold.
    assert (int(grid.num_timesteps(required)) * float(grid.dt)) / tau_far >= 3.0

    # The printed values ARE the formatted hand computation, digit for digit.
    assert f"far_path = {far_path * 1e3:.4g} mm" in text
    assert f"T/tau_far = {ratio:.4g}" in text


def test_record_three_round_trips_long_draws_no_finding():
    """The other half of the falsifier: the same fixture, a longer record."""
    num_periods = 80.0
    sim, grid, cfgs, n_steps = _built(num_periods=num_periods)
    msgs = _collect(sim, "_validate_cfg_record_vs_far_boundary",
                    freqs=F.FREQS, num_periods=num_periods,
                    grid=grid, cfgs=cfgs, n_steps=n_steps)
    assert "record_shorter_than_far_boundary_round_trip" not in _codes(msgs)


def test_band_below_the_port_cutoff_reports_that_and_no_ratio():
    sim, grid, cfgs, n_steps = _built()
    f_c = _analytic_te10_cutoff_hz(RUNG_N)
    below = np.linspace(0.70 * f_c, 0.95 * f_c, 5)
    msgs = _collect(sim, "_validate_cfg_record_vs_far_boundary",
                    freqs=below, num_periods=20.0,
                    grid=grid, cfgs=cfgs, n_steps=n_steps)
    codes = _codes(msgs)
    assert codes.count("record_far_boundary_band_below_cutoff") == 2, codes
    assert "record_shorter_than_far_boundary_round_trip" not in codes
    text = str(msgs[0])
    assert "NO T/tau_far ratio is reported" in text
    assert _number_after(text, "f_min = ") == pytest.approx(
        float(below.min()) / 1e9, rel=1e-4)


def test_unreadable_launch_direction_is_skipped_with_a_note():
    """A port whose direction cannot be read never raises; the surviving
    port's message carries the one-line note naming it."""
    sim, grid, cfgs, n_steps = _built(num_periods=10.0)
    broken = [cfgs[0], cfgs[1]._replace(direction="?")]
    msgs = _collect(sim, "_validate_cfg_record_vs_far_boundary",
                    freqs=F.FREQS, num_periods=10.0,
                    grid=grid, cfgs=broken, n_steps=n_steps)
    hits = [m for m in msgs
            if m.code == "record_shorter_than_far_boundary_round_trip"]
    assert len(hits) == 1 and hits[0].loc == "waveguide_port[0]"
    assert "waveguide_port[1]" in str(hits[0])
    assert "launch direction could not be read" in str(hits[0])


# ---------------------------------------------------------------------------
# (b) port index mirror covariance
# ---------------------------------------------------------------------------

def _runner_plane_sums(grid, cfgs):
    """The four index sums, from the runner's own plane arithmetic.

    ``apply_waveguide_port_e`` puts the ``-`` port's E correction at
    ``x_index + 1`` and ``apply_waveguide_port_h`` puts the ``+`` port's H
    correction at ``x_index - 1``; ``ref_x`` / ``probe_x`` come off the config
    already signed by direction.
    """
    plus = next(c for c in cfgs if str(c.direction).startswith("+"))
    minus = next(c for c in cfgs if str(c.direction).startswith("-"))
    return {
        "source E plane": int(plus.x_index) + (int(minus.x_index) + 1),
        "source H plane": (int(plus.x_index) - 1) + int(minus.x_index),
        "reference probe plane": int(plus.ref_x) + int(minus.ref_x),
        "measurement probe plane": int(plus.probe_x) + int(minus.probe_x),
    }


def test_mirror_audit_reports_only_the_known_e_plane_offset():
    sim, grid, cfgs, n_steps = _built()
    n_axis = int(grid.shape[0])
    sums = _runner_plane_sums(grid, cfgs)

    # The layout IS mirror-symmetric: the primal planes sit on n_axis - 1 and
    # the dual (H) plane on n_axis - 2, because a dual node at (j+0.5)*d
    # mirrors to j' = n_axis - 2 - j. Only the E plane carries the shipped +1.
    assert sums["source E plane"] == n_axis
    assert sums["source H plane"] == n_axis - 2
    assert sums["reference probe plane"] == n_axis - 1
    assert sums["measurement probe plane"] == n_axis - 1

    msgs = _collect(sim, "_validate_cfg_port_index_mirror_covariance",
                    freqs=F.FREQS, grid=grid, cfgs=cfgs, n_steps=n_steps)
    codes = _codes(msgs)
    assert "port_index_mirror_asymmetry" not in codes, [str(m) for m in msgs]
    info = [m for m in msgs
            if m.code == "port_index_mirror_known_e_plane_offset"]
    assert len(info) == 1
    assert info[0].severity == "info"
    text = str(info[0])
    assert "source E plane" in text
    assert f"sum = {n_axis} on n_axis = {n_axis}" in text
    assert f"covariant sum for a primal plane is {n_axis - 1}" in text
    assert "x_index + 1" in text


def test_one_port_moved_one_cell_fires_the_asymmetry_warning():
    sim = F.build_simulation("thru", RUNG_M)
    grid = sim._build_grid()
    n_steps = int(grid.num_timesteps(20.0))
    moved = dataclasses.replace(
        sim._waveguide_ports[1],
        x_position=F.PORT_RIGHT_X_M + RUNG_M,
    )
    sim._waveguide_ports[1] = moved
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(F.FREQS), n_steps)
            for e in sim._waveguide_ports]

    msgs = _collect(sim, "_validate_cfg_port_index_mirror_covariance",
                    freqs=F.FREQS, grid=grid, cfgs=cfgs, n_steps=n_steps)
    hits = [m for m in msgs if m.code == "port_index_mirror_asymmetry"]
    assert hits, _codes(msgs)
    named = " | ".join(str(m) for m in hits)
    for plane in ("reference probe plane", "measurement probe plane"):
        assert plane in named
    n_axis = int(grid.shape[0])
    sums = _runner_plane_sums(grid, cfgs)
    # The moved port shifts every primal plane by one cell, so their sums are
    # one ABOVE the covariant value; the message must print both.
    assert sums["reference probe plane"] == n_axis
    hit = next(m for m in hits if "reference probe plane" in str(m))
    assert f"sum = {n_axis}" in str(hit)
    assert f"covariant sum is {n_axis - 1}" in str(hit)


def test_single_port_axis_is_silent():
    sim = Simulation(freq_max=F.FREQ_MAX_HZ,
                     domain=(F.DOMAIN_X_M, F.A_M, F.B_M),
                     dx=F.DX_COARSE, boundary="cpml", cpml_layers=8)
    sim.add_waveguide_port(F.PORT_LEFT_X_M, direction="+x", mode=(1, 0),
                           mode_type="TE", freqs=jnp.asarray(F.FREQS),
                           f0=F.F0_HZ, bandwidth=F.BANDWIDTH)
    grid = sim._build_grid()
    n_steps = int(grid.num_timesteps(20.0))
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(F.FREQS), n_steps)
            for e in sim._waveguide_ports]
    msgs = _collect(sim, "_validate_cfg_port_index_mirror_covariance",
                    freqs=F.FREQS, grid=grid, cfgs=cfgs, n_steps=n_steps)
    assert msgs == []


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_preflight_sparameters_waveguide_carries_the_audits():
    """The audits reach the report, advisory, with .ok still True.

    The T/tau_far ~ 1.06 quoted here is at ``WAVEGUIDE_DEFAULT_NUM_PERIODS``
    (20.0), which is what ``preflight_sparameters`` evaluates at because it is
    ``compute_waveguide_s_matrix``'s own default. It is NOT the fixture's own
    record: ``F.NUM_PERIODS`` is 40, which puts the same rung at T/tau_far
    ~ 2.12 — still a warning, since the threshold is 3.
    """
    sim = F.build_simulation("thru", RUNG_M)
    report = sim.preflight_sparameters(calculator="waveguide")
    codes = [i.code for i in report]
    assert "record_shorter_than_far_boundary_round_trip" in codes
    assert "port_index_mirror_known_e_plane_offset" in codes
    assert report.ok and not report.errors
    # The three tiers partition the report and .warnings excludes info.
    assert len(report.errors) + len(report.warnings) + len(report.infos) == len(report)
    assert [i.code for i in report.infos] == [
        "port_index_mirror_known_e_plane_offset"]
    assert all(i.code == "record_shorter_than_far_boundary_round_trip"
               for i in report.warnings)
    # The fixture's own record is longer than the audit's evaluation default,
    # and still short of the threshold — stated as a number, not an adjective.
    grid = sim._build_grid()
    assert F.NUM_PERIODS == 40.0
    assert (int(grid.num_timesteps(F.NUM_PERIODS)) * float(grid.dt)) / (
        int(grid.num_timesteps(WAVEGUIDE_DEFAULT_NUM_PERIODS))
        * float(grid.dt)) == pytest.approx(2.0, rel=2e-3)


# ---------------------------------------------------------------------------
# strict=True escalates errors only (review of the item-2 PR)
# ---------------------------------------------------------------------------

def test_strict_does_not_raise_on_a_healthy_two_port_guide():
    """The falsifier for the review's HIGH finding.

    Every healthy two-port guide carries the informational E-plane note, so a
    strict gate keyed on emptiness would raise on every correct waveguide
    setup. This one has advisories and an info note and no error.
    """
    sim = F.build_simulation("thru", F.DX_LADDER[0])
    report = sim.preflight_sparameters(calculator="waveguide", strict=True)
    assert not report.errors
    assert report.warnings and report.infos      # not a vacuous pass
    assert report.ok


def test_strict_raises_when_an_error_severity_finding_is_present():
    """The other half: a record shorter than one round trip still stops it.

    A thicker absorber pushes the far wall out without lengthening the record,
    so T/tau_far falls below 1 at the same evaluation default.
    """
    sim = F.build_simulation("thru", F.DX_LADDER[0], cpml_layers=30)
    with pytest.raises(ValueError) as excinfo:
        sim.preflight_sparameters(calculator="waveguide", strict=True)
    text = str(excinfo.value)
    assert "error-severity issue(s)" in text
    assert "record ends BEFORE the far-boundary round trip" in text
    # The info note is returned, counted as advisory, and NOT escalated.
    assert "advisory/informational finding(s)" in text
    assert "port_index_mirror_known_e_plane_offset" not in text


# ---------------------------------------------------------------------------
# a builder exception is reported, never re-raised
# ---------------------------------------------------------------------------

def test_builder_exception_skips_the_audits_and_says_so(monkeypatch):
    sim = F.build_simulation("thru", F.DX_LADDER[0])

    def _boom(*_a, **_k):
        raise RuntimeError("mode solve exploded")

    monkeypatch.setattr(type(sim), "_build_waveguide_port_config",
                        _boom, raising=True)
    report = sim.preflight_sparameters(calculator="waveguide")
    skipped = [i for i in report if i.code == "waveguide_setup_audit_skipped"]
    assert len(skipped) == 1, [i.code for i in report]
    assert skipped[0].severity == "warning"
    assert "mode solve exploded" in str(skipped[0])
    assert "UNCHECKED" in str(skipped[0])
    # Skipped, not half-run: neither audit reported anything else, and the
    # call returned instead of raising.
    assert [i.code for i in report] == ["waveguide_setup_audit_skipped"]
    assert report.ok


def test_default_num_periods_mirrors_the_live_extractor_signature():
    live = inspect.signature(
        Simulation.compute_waveguide_s_matrix).parameters["num_periods"].default
    assert WAVEGUIDE_DEFAULT_NUM_PERIODS == live
