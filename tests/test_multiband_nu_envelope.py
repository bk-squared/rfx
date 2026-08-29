"""SPEC-01 WP6 — multi-band graded-mesh envelope regression tests (#780).

Packages the multiband witness battery
(``validation/research/multiband_nu/``, design note
``docs/design_notes/20260829_spec01_multiband_predeclaration.md``) as
regression gates:

* FAST lane (default addopts): reduced witness arms — exact-conservation
  lock of the Remis dual-cell energy (f64), a reduced-step F-S1 envelope
  arm (f32), one W2 single-transition reflection arm at the cap ratio,
  the W5 multiband AD check, and the WP6 preflight advisory contract.
* SLOW/GPU lane: the full pre-declared arms — 1e6-step 1D energy audits
  (``slow_physics``), the 3D P-B 1e6-step audit (``gpu``), and the W4R
  supraconvergence ladder (``slow_physics``).

Every threshold below is either the pre-declared falsifier window
(F-S1/F-S2 from the design note, frozen before measurement) or the
witness-validity gate the note declares (1e-12 f64 conservation) — no
threshold is tuned to the measured phase-1/W4R numbers.
"""

from __future__ import annotations

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


def _w4r2_fit(scales):
    """Run W4R2 arms at the given scales and apply the frozen F-S4
    judge (note sections W4R.3 / W4R2)."""
    from validation.research.multiband_nu.w4r2_analytic_cavity import (
        E_FLOOR_HZ, measure)
    orders = {}
    for mb in (False, True):
        pts = []
        for s in scales:
            e = measure(s, mb)
            assert e["valid"], e
            if e["err_hz"] >= E_FLOOR_HZ:
                pts.append((s, e["err_hz"]))
        assert len(pts) >= 3, pts
        h = np.log10([p[0] for p in pts])
        er = np.log10([p[1] for p in pts])
        orders[mb] = float(np.polyfit(h, er, 1)[0])
    return orders[False], orders[True]


def test_fs4_supraconvergence_reduced():
    """F-S4 fast gate: the W4R2 analytic-cavity ladder on the three
    coarse scales (~10 s). Frozen judge: p_uc in [1.7, 2.6];
    fires iff p_mb < 1.5 or p_mb < p_uc - 0.4; anomaly iff
    p_mb > p_uc + 0.4 (evidence run: p_uc = p_mb = 1.95,
    results/w4r2_analytic_cavity.json)."""
    p_uc, p_mb = _w4r2_fit((0.5, 1.0, 2.0))
    assert 1.7 <= p_uc <= 2.6, (p_uc, p_mb)
    assert not (p_mb < 1.5 or p_mb < p_uc - 0.4), (p_uc, p_mb)
    assert not p_mb > p_uc + 0.4, (p_uc, p_mb)


@pytest.mark.slow_physics
def test_fs4_supraconvergence_full():
    """The full pre-declared W4R2 ladder including s = 0.25 (~1 min).

    Runs the committed script's own ``main`` (which applies the frozen
    W4R.3 judge) in a scratch cwd, so the committed evidence file
    ``results/w4r2_analytic_cavity.json`` is never rewritten by a test
    run."""
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
             "validation.research.multiband_nu.w4r2_analytic_cavity"],
            cwd=td, env=env, check=True)
        out = json.loads(
            (Path(td) / "validation/research/multiband_nu/results/"
             "w4r2_analytic_cavity.json").read_text())
    assert out["fs4_fired"] is False, out["verdict"]
    assert not out.get("anomaly_a4"), out["verdict"]
