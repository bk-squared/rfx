"""Advisory coverage: sub-wavelength ground plane under an NTFF fixture (#334).

A patch on a 60×55 mm ground plane is 0.48λ × 0.44λ at 2.4 GHz: its far-field
pattern is ground-plane-edge-diffraction-dominated (broadside dip, off-axis
side peaks). That is expected physics — fine for a resonance fixture, wrong to
read as a solver defect in a pattern fixture. These tests pin the
``ntff_small_ground_plane`` preflight advisory that names the mechanism:

- fires ONCE on the cv05-class small-GP patch stack, anchored on the ground
  plane (never the intentionally sub-wavelength patch element);
- stays silent on the canonical >=1.4λ ground plane, without an NTFF box,
  for a sheet with no source over it, and for a thick PEC volume;
- wording is advisory: says "expected physics", never says "error";
- fires in BOTH preflight tiers (full and run()'s "advisory" tier, #303).

Geometry constants mirror the cv05 patch (L=29.5, W=38, h=1.5 mm) and the
2026-07-12 far-field arc's two ground planes (60×55 mm vs 180×180 mm class).
"""

from __future__ import annotations

from rfx import Box, Simulation


FREQ_MAX = 3e9
NTFF_FREQS = (2.4e9,)      # λ = 124.9 mm; advisory λ-reference = max(freqs)
LAM = 3e8 / NTFF_FREQS[0]

PATCH_L = 29.5e-3
PATCH_W = 38.0e-3
H_SUB = 1.5e-3
GP_T = 1.0e-3              # sheet-like ground plane thickness


def _patch_sim(gp_x: float, gp_y: float, *, with_ntff: bool = True,
               source_over_gp: bool = True) -> Simulation:
    """Patch-over-ground-plane stack centred in a CPML domain."""
    margin = 0.04
    dom = (gp_x + 2 * margin, gp_y + 2 * margin, 0.05)
    cx, cy = dom[0] / 2, dom[1] / 2
    z_gp_lo, z_gp_hi = 0.010, 0.010 + GP_T
    z_patch = z_gp_hi + H_SUB

    sim = Simulation(freq_max=FREQ_MAX, domain=dom, boundary="cpml",
                     cpml_layers=4, dx=2.5e-3)
    # ground plane (sheet) + patch element (sheet, smaller footprint)
    sim.add(Box((cx - gp_x / 2, cy - gp_y / 2, z_gp_lo),
                (cx + gp_x / 2, cy + gp_y / 2, z_gp_hi)), material="pec")
    sim.add(Box((cx - PATCH_L / 2, cy - PATCH_W / 2, z_patch),
                (cx + PATCH_L / 2, cy + PATCH_W / 2, z_patch)),
            material="pec")
    # feed between GP and patch: inside the patch footprint (over the GP), or
    # laterally clear of every sheet for the "no radiator over it" case
    if source_over_gp:
        src_xy = (cx - PATCH_L / 2 + 8e-3, cy)
    else:
        src_xy = (cx - gp_x / 2 - 0.02, cy)
    sim.add_port((src_xy[0], src_xy[1], z_gp_hi + H_SUB / 2), "ez")
    if with_ntff:
        sim.add_ntff_box(
            corner_lo=(0.012, 0.012, 0.004),
            corner_hi=(dom[0] - 0.012, dom[1] - 0.012, dom[2] - 0.012),
            freqs=NTFF_FREQS,
        )
    return sim


def _gp_issues(report):
    return [i for i in report
            if getattr(i, "code", None) == "ntff_small_ground_plane"]


def test_small_gp_pattern_fixture_fires_once_on_the_ground_plane():
    """cv05-class 60×55 mm GP (0.48λ × 0.44λ at 2.4 GHz) fires exactly once,
    anchored on the GROUND PLANE dims — not on the sub-λ patch element."""
    report = _patch_sim(60e-3, 55e-3).preflight()
    issues = _gp_issues(report)
    assert len(issues) == 1, (
        f"expected exactly one small-GP advisory, got {issues!r}")
    msg = str(issues[0])
    assert "60.0mm × 55.0mm" in msg, msg   # the GP, largest qualifying sheet
    assert "38.0mm" not in msg, (
        "advisory anchored on the patch element, not the ground plane: " + msg)
    assert "0.48λ × 0.44λ" in msg, msg


def test_wording_is_advisory_expected_physics_never_error():
    """Deliberately-small-GP designs are legitimate: the advisory must read as
    expected physics with an interpretation hint, and must never say 'error'."""
    report = _patch_sim(60e-3, 55e-3).preflight()
    (issue,) = _gp_issues(report)
    msg = str(issue)
    assert "expected physics" in msg, msg
    assert "edge diffraction" in msg, msg
    assert "intentional" in msg, msg           # the legitimate-design branch
    assert "error" not in msg.lower(), msg
    assert getattr(issue, "severity", None) == "warning"
    # a warning-only report must not block the errors-only launch gate
    report.raise_for_failure()


def test_canonical_large_gp_stays_silent():
    """The canonical >=1.4λ pattern fixture (180×180 mm class GP at 2.4 GHz)
    must NOT fire — even though its patch element is itself sub-wavelength."""
    report = _patch_sim(200e-3, 200e-3).preflight()
    assert _gp_issues(report) == [], list(report)


def test_no_ntff_box_no_fire():
    """Resonance/impedance use of the SAME small-GP stack is fine: without an
    NTFF box there is no pattern to advise about."""
    report = _patch_sim(60e-3, 55e-3, with_ntff=False).preflight()
    assert _gp_issues(report) == [], list(report)


def test_sheet_without_a_source_over_it_no_fire():
    """A sub-λ PEC sheet with no radiator over it (scattering-target class)
    must not be called a ground plane."""
    report = _patch_sim(60e-3, 55e-3, source_over_gp=False).preflight()
    assert _gp_issues(report) == [], list(report)


def test_thick_pec_volume_no_fire():
    """A 36×36×12 mm PEC VOLUME (0.4λ thick at 10 GHz) is not a sheet — the
    clean inverse-design preflight fixture class must stay clean."""
    sim = Simulation(freq_max=10e9, domain=(0.060, 0.060, 0.060),
                     boundary="cpml", cpml_layers=8, dx=1.5e-3)
    sim.add(Box((0.012, 0.012, 0.012), (0.048, 0.048, 0.024)),
            material="pec")
    sim.add_port((0.030, 0.030, 0.027), "ez")
    sim.add_ntff_box(corner_lo=(0.013, 0.013, 0.044),
                     corner_hi=(0.047, 0.047, 0.047), n_freqs=4)
    report = sim.preflight()
    assert _gp_issues(report) == [], list(report)


def test_advisory_tier_fires_too():
    """run()'s preflight tier (check_ntff="advisory", #303) must surface the
    advisory: run() is the entry point that actually computes patterns."""
    report = _patch_sim(60e-3, 55e-3).preflight(check_ntff="advisory")
    assert len(_gp_issues(report)) == 1, list(report)
