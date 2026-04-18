"""Bundle C.4 — opt-in V173-A bit-identity gate (release-tag-time).

This test is gated by ``@pytest.mark.slow_physics``. Default CI
(``pytest`` with the project's ``-m 'not gpu and not slow'``) does NOT
collect it. Run explicitly before pushing a release tag:

    pytest -m slow_physics tests/test_v173a_physics_equivalence_slow.py

It runs the V173-A patch-antenna harness in-process and asserts the
(f_res, |S11| dip) pair matches the committed baseline
``tests/data/v173a_pre_t7_phase2_baseline.json`` exactly (Δ = 0.0).

JAX/XLA-upgrade policy: if a JAX / XLA / CUDA upgrade moves the baseline
by an O(1e-14) numerical perturbation, rebump the baseline in a dedicated
``chore(baseline): rebump V173-A for JAX 0.X.Y`` commit. The commit
message must state the upgrade version and the pre/post delta must be
bounded and physically explainable (CPML dispersion + Kahan-compensated
float32 accumulation reorder is the expected source).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tests" / "data" / "v173a_pre_t7_phase2_baseline.json"
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.mark.slow_physics
def test_v173a_baseline_bit_identity():
    """Re-run the V173-A harness on current HEAD and assert Δf_res and
    Δ|S11| dip are both exactly zero relative to the committed pre-T7
    Phase 2 baseline."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import v173a_physics_equivalence as v173a  # type: ignore[import-not-found]

    baseline = json.loads(BASELINE_PATH.read_text())

    sim_a = v173a._build_sim()
    f_res = v173a._extract_harminv_f_res(sim_a, n_steps=3000)

    sim_b = v173a._build_sim()
    s11_dip_db, _ = v173a._extract_s11_dip_from_probe(sim_b, n_steps=3000)

    delta_f_hz = abs(f_res - baseline["f_res_hz"])
    delta_s11_db = abs(s11_dip_db - baseline["s11_dip_db"])

    assert delta_f_hz == 0.0, (
        f"V173-A Δf_res = {delta_f_hz} Hz ≠ 0. "
        f"current={f_res} Hz, baseline={baseline['f_res_hz']} Hz. "
        f"Rebump tests/data/v173a_pre_t7_phase2_baseline.json only if "
        f"the drift is explainable (JAX/XLA upgrade, CPML reorder); "
        f"otherwise investigate the offending commit."
    )
    assert delta_s11_db == 0.0, (
        f"V173-A Δ|S11| dip = {delta_s11_db} dB ≠ 0. "
        f"current={s11_dip_db} dB, baseline={baseline['s11_dip_db']} dB."
    )
