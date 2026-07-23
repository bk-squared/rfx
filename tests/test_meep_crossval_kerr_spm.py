"""Cross-validation: rfx vs Meep for Kerr χ³ self-phase modulation (SPM), issue #446.

Independent external-FDTD confirmation of rfx's absolute-magnitude Kerr-SPM oracle
(tests/test_kerr_spm_absolute_oracle.py, which gives ratio 0.955±0.03 vs the analytic textbook
Δn=(3/8)χ³A²). Here Meep measures the SAME observable with the SAME comparator:
  - χ³-only matched region (index=1 background ⇒ linear impedance-matched, single traveling wave)
  - true-CW source (mp.ContinuousSource) ⇒ ⟨E²⟩=A²/2 well-defined at steady state
  - k_x from the DFT phase SLOPE over the steady window; ⟨A(x)²⟩ = mean of per-point squared
    amplitude (the phase slope integrates local intensity — same rigorous comparator as rfx)
  - ratio = (k − k0)/[(3/8)·χ³·⟨A²⟩·k0]   [Meep units c=1, index=1 ⇒ n0=1]

Result (Meep 1.34.0): ratio 0.988±0.006, positive sign, R²≈1.0 — an INDEPENDENT FDTD code agreeing
with the analytic textbook to ~1%, and with rfx (0.955) to ~3%. Two comparator bugs were found and
fixed while building this (the historical "bug is in the extractor" pattern): (1) Meep's
get_dft_array normalization biased A by 2.51× ⇒ 6.3× in the ratio — fixed by doing the DFT ourselves
in numpy from recorded Ex(t); (2) coarse probe spacing aliased np.unwrap (k·Δz>π) — fixed with Δz<π/k.

Reproduce-gate (docs/research_notes/experiments/meep_thg_reproduce_gate.py): Meep's own THG tutorial —
ω power 0.04%, amplitude-Kerr F³ equivalence <1%, Born (χ³)² slope 1.965≈2 — validates Meep's χ³ wiring.

Requires: meep (conda-forge pymeep, numpy<2). Skipped via importorskip when Meep is absent (the normal
rfx CI has no Meep). Runs in any Meep-enabled env — matching the existing tests/test_meep_crossval*.py
precedent. Validated locally against Meep 1.34.0 in an isolated micromamba env (3 passed). Runtime ~3s
(1D Meep). Reproduce-gate + SPM harness: docs/research_notes/experiments/meep_{thg_reproduce_gate,kerr_spm}.py.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

FCEN = 1 / 3.0                 # λ = 3 (index=1), Meep units c=1
K0 = 2 * np.pi * FCEN
RES = 25
DPML = 2.0
SZ = 48.0
CHI3_X0, CHI3_X1 = 6.0, 42.0
PROBE_X0, PROBE_X1 = 12.0, 36.0
T_FILL = 140.0
T_MEAS = 60.0
RFX_RATIO = 0.955             # tests/test_kerr_spm_absolute_oracle.py (committed rfx reference)


def _meep_spm(chi3, amp=1.0):
    """(k_x, ⟨A²⟩, R²) for a CW wave in a χ³-only matched slab — Meep. DFT done in numpy (not
    get_dft_array) so the amplitude normalization is unambiguous."""
    mp = pytest.importorskip("meep", reason="Meep required for rfx-vs-Meep Kerr SPM crossval")
    geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, CHI3_X1 - CHI3_X0),
                     center=mp.Vector3(0, 0, (CHI3_X0 + CHI3_X1) / 2 - SZ / 2),
                     material=mp.Medium(index=1, chi3=chi3))]
    src = [mp.Source(mp.ContinuousSource(FCEN, width=15.0), component=mp.Ex,
                     center=mp.Vector3(0, 0, -0.5 * SZ + DPML - 0.5), amplitude=amp)]
    sim = mp.Simulation(cell_size=mp.Vector3(0, 0, SZ), geometry=geom, sources=src,
                        boundary_layers=[mp.PML(DPML)], resolution=RES, dimensions=1)
    sim.run(until=T_FILL)
    # probe spacing must keep k·Δz < π or np.unwrap aliases; span 24=8λ, 49 probes ⇒ Δz=0.5 ⇒ ~1.05 rad
    zc = np.linspace(PROBE_X0, PROBE_X1, 49)
    pts = [mp.Vector3(0, 0, z - SZ / 2) for z in zc]
    dt = sim.fields.dt
    rec, tstamp = [], []

    def cb(s):
        tstamp.append(s.meep_time())
        rec.append([s.get_field_point(mp.Ex, p).real for p in pts])
    sim.run(mp.at_every(dt, cb), until=T_MEAS)
    rec = np.asarray(rec)
    t = np.asarray(tstamp)
    ph = np.sum(rec * (np.exp(-1j * 2 * np.pi * FCEN * t) * dt)[:, None], axis=0)
    T = t[-1] - t[0]
    phi = np.unwrap(np.angle(ph))
    slope, icpt = np.polyfit(zc, phi, 1)
    r2 = 1.0 - np.var(phi - (slope * zc + icpt)) / np.var(phi)
    a_local = 2.0 * np.abs(ph) / T
    return abs(slope), float(np.mean(a_local ** 2)), r2


@pytest.fixture(scope="module")
def meep_spm():
    pytest.importorskip("meep", reason="Meep required for rfx-vs-Meep Kerr SPM crossval")
    kx0, _, r2_0 = _meep_spm(0.0)
    pts = {}
    for chi3 in (0.05, 0.20):
        kx, mean_a2, r2 = _meep_spm(chi3)
        dkx_txt = (3.0 / 8.0) * chi3 * mean_a2 * K0
        pts[chi3] = {"kx": kx, "dkx": kx - kx0, "ratio": (kx - kx0) / dkx_txt, "r2": r2}
    return {"kx0": kx0, "r2_0": r2_0, "pts": pts}


def test_meep_baseline_is_clean_traveling_wave(meep_spm):
    """χ³=0 baseline: a single clean traveling wave at the (near-continuum) Yee wavenumber — the
    prerequisite for a trustworthy phase-slope SPM measurement."""
    assert meep_spm["r2_0"] > 0.999, f"baseline phase not linear: R²={meep_spm['r2_0']:.5f}"
    assert abs(meep_spm["kx0"] / K0 - 1) < 0.01, (
        f"baseline kx0={meep_spm['kx0']:.4f} not near continuum k0={K0:.4f} (75 cells/λ ⇒ tiny Yee)")


def test_meep_kerr_spm_matches_textbook(meep_spm):
    """Meep — an INDEPENDENT FDTD code — reproduces the analytic Δk=(3/8)χ³⟨A²⟩k0 to within ±12%
    (measured ≈0.99), with the correct positive (self-focusing) sign and a clean phase fit."""
    for chi3, p in meep_spm["pts"].items():
        assert p["dkx"] > 0, f"χ³={chi3}: Δk={p['dkx']:.4f} not positive (self-focusing) in Meep"
        assert p["r2"] > 0.999, f"χ³={chi3}: phase not linear (R²={p['r2']:.5f})"
        assert 0.88 <= p["ratio"] <= 1.12, (
            f"χ³={chi3}: Meep SPM ratio {p['ratio']:.3f} outside [0.88,1.12] vs textbook")


def test_meep_agrees_with_rfx(meep_spm):
    """The two independent FDTD codes agree: Meep's ratio matches rfx's committed 0.955 to within
    0.10 — a genuine cross-code confirmation of the D-based Kerr operator's SPM magnitude, not just
    rfx-vs-its-own-analytic."""
    meep_ratio = float(np.mean([p["ratio"] for p in meep_spm["pts"].values()]))
    assert abs(meep_ratio - RFX_RATIO) < 0.10, (
        f"Meep ratio {meep_ratio:.3f} vs rfx {RFX_RATIO} differ by "
        f"{abs(meep_ratio - RFX_RATIO):.3f} (>0.10) — codes disagree, investigate")
