"""Regression tests for example 9's supported-safe workflow and physics anchor."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rfx import DesignRegion, maximize_transmitted_energy


def _load_example9_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "09_agent_friendly_workflow.py"
    spec = importlib.util.spec_from_file_location("example09_agent_friendly_workflow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example9_physics_anchor_prefers_air_matched_slab():
    """Uniform-slab sweep should peak at eps_r=1.0 (matched air case)."""
    ex9 = _load_example9_module()

    sweep_eps = [1.0, 2.0, 3.0, 4.0]
    energies = [ex9.run_uniform_slab_energy(eps)[0] for eps in sweep_eps]

    best_idx = int(np.argmax(energies))
    assert sweep_eps[best_idx] == 1.0, (
        f"Expected matched air slab to win, got eps_r={sweep_eps[best_idx]} "
        f"with energies={energies}"
    )


def test_example9_strict_optimize_improves_over_midpoint_uniform_baseline():
    """Strict optimization should beat the midpoint uniform slab baseline."""
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import importlib.util, json
from pathlib import Path
path = Path('examples/09_agent_friendly_workflow.py')
spec = importlib.util.spec_from_file_location('ex9', path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.N_STEPS = 100
mid_energy, _ = mod.run_uniform_slab_energy(2.5)
sim = mod.build_supported_lane_sim()
region = mod.DesignRegion(corner_lo=mod.REGION_LO, corner_hi=mod.REGION_HI, eps_range=(1.0, 4.0))
obj = mod.maximize_transmitted_energy(output_probe_idx=1)
result = mod.optimize(sim, region, obj, n_iters=10, lr=0.1, n_steps=mod.N_STEPS, preflight_mode='strict', verbose=False)
payload = {'mid_energy': mid_energy, 'final_energy': float(-result.loss_history[-1])}
print(json.dumps(payload))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["final_energy"] > payload["mid_energy"] * 1.5, (
        f"Expected optimized design to materially beat midpoint baseline: "
        f"final={payload['final_energy']:.6e}, midpoint={payload['mid_energy']:.6e}"
    )


def test_example9_preflight_and_forward_validation_are_clean():
    """Canonical workflow should start from a clean preflight and real forward signal."""
    ex9 = _load_example9_module()
    ex9.N_STEPS = 100

    sim = ex9.build_supported_lane_sim()

    region = DesignRegion(
        corner_lo=ex9.REGION_LO,
        corner_hi=ex9.REGION_HI,
        eps_range=(1.0, 4.0),
    )
    obj = maximize_transmitted_energy(output_probe_idx=1)
    report = sim.preflight_optimize(region, obj, n_steps=ex9.N_STEPS)

    assert report.ok
    assert report.strict_ok
    assert not report.issues

    mid_energy, midpoint_trace = ex9.run_uniform_slab_energy(2.5)
    assert mid_energy > 0.0
    assert np.max(np.abs(midpoint_trace[:, 1])) > 0.0


def test_example9_main_writes_png(monkeypatch):
    """The canonical example entry point should run end-to-end and save a figure."""
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "examples" / "09_agent_friendly_workflow.png"
    if out_path.exists():
        out_path.unlink()
    code = """
import importlib.util
from pathlib import Path
path = Path('examples/09_agent_friendly_workflow.py')
spec = importlib.util.spec_from_file_location('ex9', path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.N_STEPS = 100
mod.main()
"""
    monkeypatch.chdir(repo_root)
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
