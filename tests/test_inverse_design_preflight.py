"""Tests for inverse-design preflight checks (issue #30).

Covers the four new checks:
  1. Tightened minimum-resolution thresholds (5 cells PEC / 10 cells dielectric)
  2. NTFF ↔ PEC overlap (hard error)
  3. NTFF near-field gap < λ/4 (warning)
  4. AD memory estimate via ``sim.estimate_ad_memory`` / ``check_ad_memory``
"""

from __future__ import annotations

import warnings

import pytest

from rfx.api import Simulation, AD_MemoryEstimate
from rfx.geometry.csg import Box


C = 2.998e8


def _sane_sim(freq_max=5e9, domain=(0.04, 0.04, 0.04)):
    sim = Simulation(freq_max=freq_max, domain=domain, boundary="cpml",
                     cpml_layers=8, dx=C / freq_max / 20.0)
    sim.add_port((domain[0] / 2, domain[1] / 2, domain[2] / 2), "ez")
    sim.add_probe((domain[0] / 2 + 2e-3, domain[1] / 2, domain[2] / 2), "ez")
    return sim


def test_ntff_pec_overlap_rejects():
    """NTFF face crossing a PEC bbox must raise an error."""
    sim = _sane_sim()
    # PEC waveguide-like wall spanning z = 10..40 mm, large xy extent
    sim.add(Box((0.005, 0.005, 0.010), (0.035, 0.035, 0.040)),
                     material="pec")
    # NTFF z_lo=24mm sits strictly inside the PEC z-range (10..40mm)
    sim.add_ntff_box(
        corner_lo=(0.010, 0.010, 0.024),
        corner_hi=(0.030, 0.030, 0.035),
        n_freqs=4,
    )
    # Either raises on strict=True or shows an ERROR entry in the report
    with pytest.raises(ValueError, match="NTFF face"):
        sim.preflight(strict=True)

    issues = sim.preflight(strict=False)
    assert any("NTFF face" in s and "PEC" in s for s in issues), issues


def test_ntff_near_field_gap_warns():
    """NTFF box 2 cells above a source should warn about λ/4 gap."""
    freq_max = 5e9
    dx = C / freq_max / 20.0  # 3 mm
    sim = Simulation(freq_max=freq_max, domain=(0.04, 0.04, 0.04),
                     boundary="cpml", cpml_layers=8, dx=dx)
    src_z = 0.020
    sim.add_port((0.020, 0.020, src_z), "ez")
    sim.add_probe((0.025, 0.020, src_z), "ez")
    # NTFF z_lo only 2 cells above the source — well below λ/4
    sim.add_ntff_box(
        corner_lo=(0.010, 0.010, src_z + 2 * dx),
        corner_hi=(0.030, 0.030, src_z + 3 * dx),
        n_freqs=4,
    )
    issues = sim.preflight(strict=False)
    assert any("λ/4" in s or "lambda/4" in s.lower() or "near-field" in s
               for s in issues), (
        "Expected a λ/4 near-field warning, got: " + "\n".join(issues)
    )


def test_ad_memory_estimate_reasonable():
    """40×40×40 @ 3GHz: estimate must be > 0 and < 10GB (checkpointed).
    At n_steps=10000 the non-checkpointed value must be much larger.
    """
    freq_max = 3e9
    dx = C / freq_max / 20.0
    # Use a small physical domain to land at ~40 cells
    extent = 40 * dx
    sim = Simulation(freq_max=freq_max, domain=(extent, extent, extent),
                     boundary="cpml", cpml_layers=6, dx=dx)
    sim.add_port((extent / 2, extent / 2, extent / 2), "ez")
    sim.add_probe((extent / 2 + dx, extent / 2, extent / 2), "ez")

    est_small = sim.estimate_ad_memory(500, available_memory_gb=40.0)
    assert isinstance(est_small, AD_MemoryEstimate)
    assert est_small.forward_gb > 0
    assert 0 < est_small.ad_checkpointed_gb < 10.0
    # checkpoint ≪ full when n_steps is large
    est_big = sim.estimate_ad_memory(10000, available_memory_gb=40.0)
    assert est_big.ad_full_gb > 5 * est_big.ad_checkpointed_gb


def test_preflight_passes_on_clean_setup():
    """A sane patch-antenna-ish sim triggers no ERROR / NTFF warnings.

    The original 5 GHz / 60 mm-cube version pre-2026-05-07 was not
    actually sane: the GP was sub-cell (2 mm at dx≈3 mm = 0.7 cells),
    the NTFF box reached into the 24 mm-thick x-CPML, and after the
    NTFF λ/2 near-field guardrail landed (NTFF face must be ≥ λ/2
    from any source), the 60 mm-cube domain at 5 GHz left no room
    for a sane NTFF box at all (λ/2 = 30 mm, full domain = 60 mm).
    Bumped to 10 GHz / 60 mm cube so λ/2 = 15 mm and a real
    1-cell-margin NTFF box fits clear of CPML and clear of the
    near field on every face.
    """
    freq_max = 10e9
    dx = C / freq_max / 20.0   # 1.5 mm
    extent = 0.060             # 60 mm cube; CPML 8·dx = 12 mm, so the
                                # CPML-free interior is [12, 48] mm.
    sim = Simulation(freq_max=freq_max, domain=(extent, extent, extent),
                     boundary="cpml", cpml_layers=8, dx=dx)
    # Ground plane well resolved (xy = 24 cells, z = 8 cells PEC volume).
    sim.add(Box((0.012, 0.012, 0.012), (0.048, 0.048, 0.024)),
                     material="pec")
    # Source 2 cells above GP, clear of CPML on every face.
    sim.add_port((0.030, 0.030, 0.027), "ez")
    sim.add_probe((0.032, 0.030, 0.027), "ez")
    # NTFF box: λ_min = 30 mm at 10 GHz → λ/2 = 15 mm.  Every face
    # sits ≥ 17 mm from the source, well above both λ/4 and λ/2.
    sim.add_ntff_box(
        corner_lo=(0.013, 0.013, 0.044),
        corner_hi=(0.047, 0.047, 0.047),
        n_freqs=4,
    )
    issues = sim.preflight(strict=False)
    # Filter to errors/NTFF/resolution issues that this test guards.
    bad = [s for s in issues
           if s.startswith("ERROR:")
           or "NTFF face" in s
           or "λ/4" in s
           or "λ/2" in s]
    assert not bad, "Clean setup should not trip new checks: " + "\n".join(bad)


def test_waveguide_resolution_warns():
    """Dielectric block whose ``cells_per_λ_eff`` is below the P1.5
    threshold (10) should warn — the rule fires on per-axis effective
    wavelength, not on per-dimension cell count.

    Original WR-90 setup (eps_r=2.2 / freq_max=10 GHz / dx=2 mm) gave
    ~10.1 cells/λ_eff, marginally above the threshold; the warning was
    silent and the assertion never tested the rule. Bumped eps_r to 3.0
    so cells_per_λ_eff ≈ 8.7 and P1.5 actually fires.
    """
    freq_max = 10e9
    dx = 2e-3
    sim = Simulation(freq_max=freq_max, domain=(0.05, 0.05, 0.05),
                     boundary="cpml", cpml_layers=6, dx=dx)
    sim.add_material("diel", eps_r=3.0)
    sim.add(Box((0.010, 0.010, 0.010),
                         (0.010 + 0.01016, 0.010 + 0.02286, 0.040)),
                     material="diel")
    sim.add_port((0.015, 0.020, 0.025), "ez")
    sim.add_probe((0.018, 0.020, 0.025), "ez")
    issues = sim.preflight(strict=False)
    # P1.5 wording was reworded from "under-resolved" to "Need ≥10 for
    # phase-accurate propagation"; pin the substantive content
    # (dielectric + cells/λ_eff threshold) rather than the old wording.
    assert any(
        "dielectric" in s
        and ("cells per λ_eff" in s or "phase-accurate" in s)
        for s in issues
    ), "Expected a dielectric resolution warning: " + "\n".join(issues)


# ---------------------------------------------------------------------------
# Issue #303: run() advisory tier + honest summary + probe exclusion
# ---------------------------------------------------------------------------

def _pec_overlap_sim():
    """The test_ntff_pec_overlap_rejects config, reused for tier tests."""
    sim = _sane_sim()
    sim.add(Box((0.005, 0.005, 0.010), (0.035, 0.035, 0.040)),
                     material="pec")
    sim.add_ntff_box(
        corner_lo=(0.010, 0.010, 0.024),
        corner_hi=(0.030, 0.030, 0.035),
        n_freqs=4,
    )
    return sim


def test_advisory_tier_keeps_gap_warning_drops_pec_error():
    """check_ntff="advisory" = λ/4 advisories WITHOUT the PEC hard error.

    This is run()'s tier (issue #303): the near-field advisory is
    physics-relevant to any far-field computation, while the PEC-overlap
    structural error stays an inverse-design (forward/optimize/explicit
    preflight) gate.
    """
    # PEC-overlap config: full tier errors, advisory tier must not.
    issues_adv = _pec_overlap_sim().preflight(check_ntff="advisory")
    assert not any("PEC" in s and "NTFF face" in s for s in issues_adv), (
        "advisory tier must not run the PEC-overlap error check: "
        + "\n".join(issues_adv))

    # λ/4-gap config (from test_ntff_near_field_gap_warns): advisory tier
    # must still surface the near-field advisory.
    freq_max = 5e9
    dx = C / freq_max / 20.0
    sim = Simulation(freq_max=freq_max, domain=(0.04, 0.04, 0.04),
                     boundary="cpml", cpml_layers=8, dx=dx)
    src_z = 0.020
    sim.add_port((0.020, 0.020, src_z), "ez")
    sim.add_ntff_box(
        corner_lo=(0.010, 0.010, src_z + 2 * dx),
        corner_hi=(0.030, 0.030, src_z + 3 * dx),
        n_freqs=4,
    )
    issues = sim.preflight(check_ntff="advisory")
    assert any("near-field" in s or "λ/4" in s for s in issues), (
        "advisory tier must keep the λ/4 gap advisory: " + "\n".join(issues))


def test_run_auto_preflight_surfaces_ntff_advisories():
    """run() now WARNS on NTFF near-field configs instead of silence (#303).

    Previously run() skipped the NTFF family (check_ntff=False) and printed
    "All checks passed", contradicting explicit sim.preflight() on the same
    configuration. The advisory must be a non-blocking UserWarning: the run
    completes.
    """
    freq_max = 5e9
    dx = C / freq_max / 20.0
    sim = Simulation(freq_max=freq_max, domain=(0.04, 0.04, 0.04),
                     boundary="cpml", cpml_layers=8, dx=dx)
    src_z = 0.020
    sim.add_port((0.020, 0.020, src_z), "ez")
    sim.add_ntff_box(
        corner_lo=(0.010, 0.010, src_z + 2 * dx),
        corner_hi=(0.030, 0.030, src_z + 3 * dx),
        n_freqs=4,
    )
    with pytest.warns(UserWarning, match=r"\[run\] preflight found"):
        result = sim.run(n_steps=3)
    assert result is not None  # non-blocking: the run completed


def test_run_does_not_hard_fail_on_ntff_pec_overlap():
    """The PEC-overlap ERROR stays off run()'s tier (historical contract)."""
    sim = _pec_overlap_sim()
    # Must not raise PreflightConfigError/ValueError from preflight; other
    # advisories (if any) surface as UserWarning, which we tolerate here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=3)
    assert result is not None


def test_probe_on_ntff_face_is_not_a_culprit():
    """Passive DFT probes are excluded from the near-field culprits (#303).

    A probe ON the box face previously produced a 0.00mm 'radiating/
    scattering structure' advisory. Config keeps every face >= λ/2 from the
    SOURCE (x_lo gap 35mm > λ/2 = 30mm; other faces tangentially out of
    scope), so with probes excluded the report must carry no NTFF finding.
    """
    freq_max = 5e9
    domain = (0.10, 0.10, 0.10)
    sim = Simulation(freq_max=freq_max, domain=domain, boundary="cpml",
                     cpml_layers=8, dx=C / freq_max / 20.0)
    sim.add_port((0.010, 0.060, 0.060), "ez")
    sim.add_probe((0.045, 0.060, 0.060), "ez")  # exactly on the x_lo face
    sim.add_ntff_box(
        corner_lo=(0.045, 0.045, 0.045),
        corner_hi=(0.075, 0.075, 0.075),
        n_freqs=4,
    )
    issues = sim.preflight()
    ntff_issues = [s for s in issues if "NTFF face" in s]
    assert ntff_issues == [], (
        "probe-on-face must not be an NTFF culprit: " + "\n".join(ntff_issues))


def test_zero_issue_summary_line_is_honest(capsys):
    """The all-clear line states which NTFF tier actually ran (#303)."""
    def _clean_sim():
        from rfx.sources.sources import GaussianPulse
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02),
                         dx=0.02 / 15, boundary="cpml", cpml_layers=6)
        sim.add_port(position=(0.0093, 0.0093, 0.0093), component="ez",
                     impedance=50.0,
                     waveform=GaussianPulse(f0=5e9, bandwidth=0.9),
                     extent=0.004)
        return sim

    assert _clean_sim().preflight() == []
    assert "All checks passed." in capsys.readouterr().out

    assert _clean_sim().preflight(check_ntff="advisory") == []
    assert "NTFF advisory tier" in capsys.readouterr().out

    assert _clean_sim().preflight(check_ntff=False) == []
    assert "NTFF checks skipped" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Issue #314: wire-port MIDPOINT V/I probe cell inside PEC
# ---------------------------------------------------------------------------

def _microstrip_sim(port_extent_m: float,
                    trace_z_lo_m: float = 1.0e-3,
                    trace_z_hi_m: float = 1.5e-3) -> Simulation:
    """Minimal air-microstrip: PEC trace at z in [1.0, 1.5] mm, dx=0.5 mm.

    A vertical ez wire port from z=0 rasterizes (production
    ``_wire_port_cells``) to cells of 0.5 mm with sample centers at
    z = 0.25/0.75/1.25/... mm; extent 1.5 mm -> 4 cells, midpoint cell
    center z = 1.25 mm INSIDE the default trace (the measured #314
    corruption case); extent 1.0 mm -> 3 cells, midpoint center z =
    0.75 mm in the substrate gap BUT the TOP cell (center 1.25 mm) is
    inside the trace — the #313 dead-extent-cell geometry (issue #319).
    ``trace_z_lo_m``/``trace_z_hi_m`` move/thicken the trace for the
    clean and both-dead variants.
    """
    sim = Simulation(freq_max=10e9, domain=(0.016, 0.010, 0.006),
                     boundary="cpml", cpml_layers=6, dx=0.5e-3)
    sim.add(Box((0.004, 0.003, trace_z_lo_m), (0.012, 0.007, trace_z_hi_m)),
            material="pec")
    sim.add_port(position=(0.008, 0.005, 0.0), component="ez",
                 impedance=50.0, extent=port_extent_m)
    return sim


def _by_code(issues, code):
    return [s for s in issues if getattr(s, "code", None) == code]


def test_wire_midpoint_in_pec_warns():
    issues = _microstrip_sim(1.5e-3).preflight()
    hits = [s for s in issues if "MIDPOINT" in s and "PEC" in s]
    assert hits, (
        "extent=1.5mm puts the midpoint probe cell (z=1.25mm) inside the "
        "trace [1.0,1.5]mm — the #314 warning must fire: "
        + "\n".join(issues))


def test_wire_midpoint_in_gap_does_not_warn():
    """extent=1.0mm keeps the MIDPOINT (z=0.75mm) in the gap — no #314
    warning. (The top extent cell IS dead on this geometry; that is the
    separate #319 advisory, covered below.)"""
    issues = _microstrip_sim(1.0e-3).preflight()
    hits = [s for s in issues if "MIDPOINT" in s]
    assert hits == [], (
        "extent=1.0mm keeps the midpoint (z=0.75mm) in the gap — no #314 "
        "warning expected: " + "\n".join(hits))


# ---------------------------------------------------------------------------
# Issue #319: non-midpoint wire-port extent cells inside PEC (dead cells)
# ---------------------------------------------------------------------------

def test_wire_dead_extent_cell_warns_with_live_count():
    """The #313 geometry: extent=1.0mm -> 3 cells (centers 0.25/0.75/1.25
    mm); the TOP cell sits inside the trace [1.0,1.5]mm while the midpoint
    (0.75mm) is clean. The #319 dead-extent advisory must fire with
    n_live/n = 2/3, the post-#318-fix semantics (dead cells EXCLUDED from
    the resistance distribution / drive / normalization), and the
    historical pre-fix Z0*(n_live/n) = 33.3-ohm citation; the #314
    midpoint warning must NOT fire."""
    issues = _microstrip_sim(1.0e-3).preflight()
    dead = _by_code(issues, "wire_port_dead_extent_cells")
    assert len(dead) == 1, (
        "expected exactly one dead-extent advisory, got: "
        + "\n".join(str(s) for s in issues))
    assert "2/3" in dead[0] and "excluded" in dead[0], (
        "dead-extent advisory must report n_live/n = 2/3 and the "
        f"post-#318 excluded-from-distribution semantics: {dead[0]}")
    assert "33.3" in dead[0], (
        "dead-extent advisory must keep the historical pre-fix "
        f"Z0*(n_live/n) = 33.3-ohm citation (issue #313): {dead[0]}")
    assert not _by_code(issues, "wire_port_midpoint_in_pec"), (
        "midpoint is clean on this geometry — #314 must stay silent")


def test_wire_midpoint_only_dead_keeps_midpoint_warning_only():
    """Regression (#314) + documented #319/#318 behavior: when ONLY the
    midpoint cell is dead (extent=1.5mm -> 4 cells, only z=1.25mm inside
    the trace), the strong midpoint warning fires and the dead-extent
    advisory (reserved for NON-midpoint dead cells) does not. (The #318
    live-cell split still excludes that dead midpoint from sigma/drive —
    the advisory split here is about which WARNING fires.)"""
    issues = _microstrip_sim(1.5e-3).preflight()
    assert _by_code(issues, "wire_port_midpoint_in_pec"), (
        "#314 midpoint warning must still fire: "
        + "\n".join(str(s) for s in issues))
    assert not _by_code(issues, "wire_port_dead_extent_cells"), (
        "only the midpoint cell is dead — the dead-extent advisory is "
        "reserved for non-midpoint dead cells")


def test_wire_port_all_cells_live_is_silent():
    """Trace raised to [1.5,2.0]mm: all 3 cell centers (0.25/0.75/1.25mm)
    are live — neither wire-port-in-PEC warning fires."""
    issues = _microstrip_sim(1.0e-3, trace_z_lo_m=1.5e-3,
                             trace_z_hi_m=2.0e-3).preflight()
    hits = (_by_code(issues, "wire_port_midpoint_in_pec")
            + _by_code(issues, "wire_port_dead_extent_cells"))
    assert hits == [], (
        "fully-live wire port must be silent: "
        + "\n".join(str(s) for s in hits))


def test_wire_midpoint_and_other_cell_dead_fires_both():
    """Documented #319/#318 behavior for the combined case: a thick trace
    [0.5,1.5]mm with extent=1.5mm kills cells 0.75mm AND 1.25mm (the
    midpoint) -> BOTH warnings fire, and the dead-extent live count
    includes the midpoint cell (n_live/n = 2/4; historical pre-#318-fix
    termination citation ~25.0 ohm), because the live split does not care
    which cell holds the probe."""
    issues = _microstrip_sim(1.5e-3, trace_z_lo_m=0.5e-3,
                             trace_z_hi_m=1.5e-3).preflight()
    assert _by_code(issues, "wire_port_midpoint_in_pec"), (
        "midpoint (z=1.25mm) is inside the thick trace — #314 must fire: "
        + "\n".join(str(s) for s in issues))
    dead = _by_code(issues, "wire_port_dead_extent_cells")
    assert len(dead) == 1, (
        "non-midpoint cell (z=0.75mm) is also dead — #319 must fire: "
        + "\n".join(str(s) for s in issues))
    assert "2/4" in dead[0] and "25.0" in dead[0], (
        "dead-extent live count must include the dead midpoint cell "
        f"(n_live/n = 2/4; historical pre-fix citation 25.0 ohm): {dead[0]}")
