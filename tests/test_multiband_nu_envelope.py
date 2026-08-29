"""SPEC-01 WP6 — multi-band graded-mesh envelope regression tests (#780).

Packages the multiband witness battery
(``validation/research/multiband_nu/``, design note
``docs/design_notes/20260829_spec01_multiband_predeclaration.md``) as
regression gates:

* FAST lane (default addopts, ~2 min): reduced witness arms — exact-conservation
  lock of the Remis dual-cell energy (f64), a reduced-step F-S1 envelope
  arm (f32), one W2 single-transition reflection arm at the cap ratio,
  the W5 multiband AD check, the W4R3 F-S4 z-dominance gates (analytic)
  and reduced ladder, the W4R3 grading-side revert-proof (which must
  FIRE), and the WP6 preflight advisory contract.
* SLOW/GPU lane: the full pre-declared arms — 1e6-step 1D energy audits
  (``slow_physics``), the 3D P-B 1e6-step audit (``gpu``), and the full
  four-scale W4R3 supraconvergence ladder (``slow_physics``).

Every threshold below is either the pre-declared falsifier window
(F-S1/F-S2 from the design note, frozen before measurement) or the
witness-validity gate the note declares (1e-12 f64 conservation; the
W4R3 fixture gates G1/G2/G3) — no threshold is tuned to a measured
number.

F-S4 is gated on the W4R3 z-dominant analytic cavity, NOT on the earlier
W4R2 fixture: the adversarial review found that W4R2 carries ~1 % of its
error on the graded axis, so its order gate could not have failed for a
grading reason (design-note sections WP6R.2 / WP6R.3).
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from validation.research.multiband_nu import fixtures as fx
from validation.research.multiband_nu.harness import (
    build_pec_fixture, run_energy_audit, te10_blob_ex, random_blob_3d)
from validation.research.multiband_nu.remis_energy import (
    adjointness_residual)
from validation.research.multiband_nu.w1_energy_drift import evaluate_fs1


# ---------------------------------------------------------------------------
# fast lane
# ---------------------------------------------------------------------------

def _reduced_pa(r: float, variant: str = "abrupt") -> np.ndarray:
    return fx.pa_profile(r, variant, n_fine=12, n_coarse=9)


def test_remis_energy_witness_validity_f64():
    """The note's §2.3 witness-validity gates + gate-2 revert-proof:
    SBP adjointness < 1e-12 (host f64), then the full revert-proof
    (baseline f64 conservation < 1e-12 AND both deliberate defect
    injections break it) in a JAX_ENABLE_X64 subprocess — the exact
    committed ``validation/research/multiband_nu/revert_proof.py`` run
    (x64 is required for the kernel-realized coefficients; without it
    the functional carries the documented ~5e-8 f32-coefficient
    residual, not a conservation defect)."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    prof = _reduced_pa(1.4)
    grid, _mats = build_pec_fixture(prof, (fx.A_X, fx.B_Y), fx.DXY)
    assert adjointness_residual(grid, seed=0) < 1e-12
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, JAX_ENABLE_X64="1",
               PYTHONPATH=str(root))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # the script writes its JSON to a relative path; run it in a
        # scratch cwd so the committed evidence file is not rewritten
        (Path(td) / "validation/research/multiband_nu/results").mkdir(
            parents=True)
        subprocess.run(
            [sys.executable, "-m",
             "validation.research.multiband_nu.revert_proof"],
            cwd=td, env=env, check=True)
        out = json.loads(
            (Path(td) / "validation/research/multiband_nu/results/"
             "revert_proof.json").read_text())
    assert out["pass"], out
    assert out["baseline_drift"] < 1e-12, out


def test_fs1_envelope_reduced_f32():
    """Reduced F-S1 arm: P-A(r=1.4 abrupt), 2e4 steps, float32. The
    pre-declared envelope (K·u·sqrt(n), K=20) and growth-trend judge
    from the note §2.4 must not fire."""
    prof = _reduced_pa(1.4)
    grid, mats = build_pec_fixture(prof, (fx.A_X, fx.B_Y), fx.DXY)
    init = {"ex": te10_blob_ex(grid)}
    steps, energies = run_energy_audit(grid, mats, init, 20000,
                                       sample_every=500)
    verdict = evaluate_fs1(steps, energies)
    assert not verdict["fs1_fired"], verdict


def test_fs2_single_transition_at_cap():
    """One W2 arm at the envelope cap (r=1.4 abrupt): gated reflection
    against the frozen chain-model window max(3*R_model, 3e-5) — the
    exact pre-declared F-S2 judge."""
    from validation.research.multiband_nu.w2_w3_reflection import (
        run_probe, w2_arm)
    prof_b = fx.uniform_reference_profile()
    grid_b, trace_b = run_probe(prof_b, fx.K_PRB, 800, fx.K_SRC)
    res = w2_arm(1.4, "abrupt", trace_b, grid_b)
    assert res["fs2_fired"] is False, res


def test_w5_multiband_ad_consistency():
    """F-S5: jax.grad of sum(ts^2) w.r.t. the multiband w5_profile vs
    central FD, dominant cells within the existing NU AD convention
    (15 % rel, f32 — test_nonuniform_gradient.py convention)."""
    from validation.research.multiband_nu.w5_ad_consistency import measure
    res = measure("regression")
    assert not res["fs5_fired"], res


@functools.lru_cache(maxsize=None)
def _w4r3_arm(scale, multiband, defect=False):
    """One W4R3 arm, cached so the fast lane runs each arm exactly once."""
    from validation.research.multiband_nu.w4r3_zdominant_cavity import measure
    return measure(scale, multiband, defect=defect)


def _w4r3_rows(scales, defect=False):
    return {(mb, s): _w4r3_arm(s, mb, defect and mb)
            for s in scales for mb in (False, True)}


def test_fs4_fixture_is_z_dominant():
    """BL2 gate (note WP6R.3): the F-S4 fixture's error budget must be
    carried by the GRADED axis, or its order gate cannot fail for a
    grading reason. Pure analytic decomposition, no FDTD — G1 (the z
    share of the modelled error budget) >= 0.80 in both arms at every
    scale, and G2 (the grading-SPECIFIC share) >= 0.20 at every scale."""
    from validation.research.multiband_nu import w4r3_zdominant_cavity as W
    from validation.research.multiband_nu.analytic_dispersion import decompose
    W.assert_planes_realizable()
    for s in W.SCALES:
        d = {}
        for mb in (False, True):
            prof = W.mb_profile(s) if mb else W.uc_profile(s)
            grid, _m = build_pec_fixture(prof, (W.A_X, W.B_Y), W.DXY0 * s)
            d[mb] = decompose(prof, W.DXY0 * s, W.A_X, float(grid.dt),
                              W.M_X, W.P_Z, W.L_Z)
            assert d[mb]["z_fraction"] >= W.G1_Z_FRACTION_MIN, (s, mb, d[mb])
        share = abs(d[True]["e_z"] - d[False]["e_z"]) / abs(d[True]["e_total"])
        assert share >= W.G2_GRADING_SHARE_MIN, (s, share)


def test_fs4_supraconvergence_reduced():
    """F-S4 fast gate: the W4R3 ladder on the three coarse scales, judged
    by the committed judge — the frozen W4R.3 rule (fixture gate
    p_uc in [1.7, 2.6]; fires iff p_mb < 1.5 or p_mb < p_uc - 0.4;
    anomaly iff p_mb > p_uc + 0.4) plus the W4R3 fixture-validity
    gates."""
    from validation.research.multiband_nu.w4r3_zdominant_cavity import judge
    scales = (0.5, 1.0, 2.0)
    out = judge(_w4r3_rows(scales), scales=scales)
    for g in ("g1_pass", "g2_pass", "g3_pass"):
        assert out["fixture_gates"][g], out["fixture_gates"]
    assert out["fs4_fired"] is False, out["verdict"]
    assert not out["anomaly_a4"], out["verdict"]


def test_fs4_grading_defect_fires():
    """Revert-proof for the ORDER witness (note WP6R.4): with the
    CORE-C2-class metric defect on ONE multiband transition node — an
    error that is identically null on a uniform mesh, i.e. purely
    grading-side — the committed judge must FIRE. A witness that cannot
    detect a corrupted transition coefficient cannot carry an order
    claim."""
    from validation.research.multiband_nu.w4r3_zdominant_cavity import judge
    scales = (0.5, 1.0, 2.0)
    out = judge(_w4r3_rows(scales, defect=True), defect=True, scales=scales)
    assert out["fs4_fired"] is True, out["verdict"]


def _grading_advisories(dz, cpml=0, boundary="pec"):
    from rfx import Simulation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9,
                         domain=(10e-3, 10e-3, float(np.sum(dz))),
                         dx=1e-3, boundary=boundary, cpml_layers=cpml,
                         dz_profile=np.asarray(dz, float))
        sim.add_source((5e-3, 5e-3, float(np.sum(dz)) / 2), "ez",
                       amplitude_kind="current")
        issues = sim.preflight()
    return [getattr(i, "code", "") for i in
            (getattr(issues, "issues", issues) or [])
            if "grading" in str(getattr(i, "code", ""))]


def test_preflight_multiband_within_cap_is_clean():
    """WP6 allow-side: a small-large-small-large profile with every
    adjacent ratio <= 1.4 draws NO grading advisory and NO constructor
    ratio warning."""
    mm = 1e-3
    dz = np.array([1, 1, 1, 1.4, 1.4, 1.4, 1, 1, 1, 1.4, 1.4, 1, 1, 1]) * mm
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from rfx import Simulation
        Simulation(freq_max=10e9, domain=(10e-3, 10e-3, float(dz.sum())),
                   dx=1e-3, boundary="pec", dz_profile=dz)
        ctor = [x for x in w if "adjacent cell ratio" in str(x.message)]
    assert not ctor, [str(x.message) for x in ctor]
    assert _grading_advisories(dz) == []


def test_preflight_beyond_cap_advisory():
    mm = 1e-3
    dz = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1]) * mm
    codes = _grading_advisories(dz)
    assert "nu_grading_ratio_beyond_validated_cap" in codes, codes


def test_preflight_grading_reaches_absorber_advisory():
    mm = 1e-3
    graded_to_face = np.array(
        [1, 1, 1, 1, 1, 1.4, 1.4, 1.4, 1.4, 1.3, 1.2, 1.1, 1.0]) * mm
    codes = _grading_advisories(graded_to_face, cpml=4, boundary="cpml")
    assert "nu_grading_reaches_absorber" in codes, codes
    uniform_runways = np.array(
        [1, 1, 1, 1, 1, 1.4, 1.4, 1.4, 1, 1, 1, 1, 1]) * mm
    codes = _grading_advisories(uniform_runways, cpml=4, boundary="cpml")
    assert codes == [], codes


def test_preflight_absorber_runway_exactly_compliant():
    """The absorber-runway advisory must match its own REMEDY ("keep at
    least `layers` uniform interior cells against that face"): a profile
    with EXACTLY that many uniform cells is compliant and must be clean;
    one cell short must fire. Locks the off-by-one found in review
    (the check read p[:layers+1], i.e. layers+1 uniform cells)."""
    mm = 1e-3
    exact = np.array([1, 1, 1, 1, 1.4, 1.4, 1.4, 1.4, 1, 1, 1, 1]) * mm
    assert _grading_advisories(exact, cpml=4, boundary="cpml") == []
    one_short = np.array([1, 1, 1, 1.4, 1.4, 1.4, 1.4, 1.4, 1, 1, 1, 1]) * mm
    assert "nu_grading_reaches_absorber" in _grading_advisories(
        one_short, cpml=4, boundary="cpml")


# ---------------------------------------------------------------------------
# slow / gpu lane — the full pre-declared arms
# ---------------------------------------------------------------------------

@pytest.mark.slow_physics
@pytest.mark.parametrize("r,variant", [(1.4, "abrupt"), (1.4, "smooth"),
                                       (1.1, "abrupt"), (1.2, "abrupt")])
def test_fs1_full_1e6_1d(r, variant):
    """Full pre-declared F-S1 1D arms: 1e6 steps, float32, P-A."""
    from validation.research.multiband_nu.w1_energy_drift import run_pa_arm
    res = run_pa_arm(r, variant, 1_000_000, 5000)
    assert not res["fs1_fired"], res


@pytest.mark.gpu
@pytest.mark.parametrize("r", [1.4, 2.0])
def test_fs1_full_1e6_3d_pb(r):
    """Full pre-declared F-S1 3D arm (P-B, 1e6 steps) — GPU-scale;
    the committed GPU-run evidence lives in
    validation/research/multiband_nu/results/w1_pb_full_gpu.json
    (VESSL 369367256892)."""
    from validation.research.multiband_nu.w1_energy_drift import run_pb_arm
    res = run_pb_arm(r, 1_000_000, 5000, claim=True)
    assert not res["fs1_fired"], res


@pytest.mark.slow_physics
def test_fs4_supraconvergence_full():
    """The full pre-declared W4R3 ladder, all four scales.

    Runs the committed script's own ``main`` (which applies the frozen
    W4R.3 judge and the W4R3 fixture gates) in a scratch cwd, so the
    committed evidence file ``results/w4r3_zdominant_cavity.json`` is
    never rewritten by a test run."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "validation/research/multiband_nu/results").mkdir(
            parents=True)
        subprocess.run(
            [sys.executable, "-m",
             "validation.research.multiband_nu.w4r3_zdominant_cavity"],
            cwd=td, env=env, check=True)
        out = json.loads(
            (Path(td) / "validation/research/multiband_nu/results/"
             "w4r3_zdominant_cavity.json").read_text())
    for g in ("g1_pass", "g2_pass", "g3_pass"):
        assert out["fixture_gates"][g], out["fixture_gates"]
    assert out["fs4_fired"] is False, out["verdict"]
    assert not out.get("anomaly_a4"), out["verdict"]
