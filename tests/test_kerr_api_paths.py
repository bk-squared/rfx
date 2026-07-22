"""Kerr χ³ on the public API paths: run() WIRES it, forward() GUARDS against silently
dropping it (#403 blind spot — the emergent-propagation Kerr path was untested end-to-end).

Two gaps this closes:

1. ``test_nonlinear.py`` validates the single-cell ADE update formula
   (E^{n+1}=E^n/(1+factor)) and energy-boundedness via the MANUAL update loop, but nothing
   asserted that the public ``Simulation.run()`` API actually threads χ³ through to
   ``apply_kerr_ade`` (simulation.py). If run() silently dropped Kerr (as forward() did),
   every existing test would still pass. This gates that wiring.

2. ``forward()`` (the differentiable path) discarded ``kerr_chi3`` from
   ``_assemble_materials`` — a Kerr material gave LINEAR physics and a LINEAR gradient with
   no error (a silent-wrong footgun for differentiable nonlinear design). Now guarded.

WHY a wiring/monotonicity/stability oracle and NOT a quantitative SPM(Δφ) match: the naive
pulsed-transmission-phase comparator is run-length-artifact-dominated (∠ flips sign between
NP=30 and NP=40) AND at gateable χ³ the ADE per-step factor (dt/ε0)·χ³·E²≈0.4 is STRONGLY
nonlinear (perturbative Δn=χ³⟨E²⟩/2n0 breaks down). A tight quantitative Kerr oracle needs a
CW steady-state intensity comparator in the weak regime — a dedicated session (issue filed).
This oracle instead gates the ROBUST observable: the probe field-difference norm ||E(χ³)−E(0)||
is run-length-robust (0.8426 vs 0.8415 @ NP=30/40), domain-size INVARIANT (0.8426 across
0.5/0.6 m × 0.06/0.10 m), null-controlled (χ³=0 ⇒ exactly 0), and monotonic in χ³
(0.70/0.78/0.84 @ χ³=1/2/4). Harness: docs/research_notes/experiments/kerr_spm_*.py
"""
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.geometry import Box

F0 = 5e9
DX = 0.002
EPS_R = 2.0
DOMAIN = (0.50, 0.06, 0.006)
X_IFACE, X_SLAB_END = 0.12, 0.27
XPROBES = (0.16, 0.18, 0.20, 0.22, 0.33)


# --------------------------------------------------------------------------- #
# FAST: forward() must fail loud on a Kerr material (no silent linear physics).
# --------------------------------------------------------------------------- #
def _kerr_sim(chi3):
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml",
                     cpml_layers=10, mode="3d")
    sim.add_material("kerr", eps_r=EPS_R, chi3=chi3)
    sim.add(Box((X_IFACE, -1, -1), (X_SLAB_END, 1, 1)), material="kerr")
    sim.add_tfsf_source(f0=F0, bandwidth=0.3, amplitude=1.0, polarization="ez",
                        direction="+x", waveform="modulated_gaussian")
    for xp in XPROBES:
        sim.add_probe((xp, DOMAIN[1] / 2, DOMAIN[2] / 2), component="ez")
    return sim


def test_forward_rejects_kerr_material():
    """forward() with a χ³ material raises (rather than silently running LINEAR physics)."""
    sim = _kerr_sim(0.1)
    with pytest.raises(NotImplementedError, match="Kerr"):
        sim.forward(n_steps=50, skip_preflight=True)


def test_forward_guard_message_points_to_run():
    """The guard tells the caller to use run() — the path that does thread χ³."""
    sim = _kerr_sim(0.1)
    with pytest.raises(NotImplementedError, match="run"):
        sim.forward(n_steps=50, skip_preflight=True)


def test_forward_without_kerr_is_unaffected():
    """A χ³=0 (linear) material does NOT trip the guard — the forward path still runs."""
    sim = _kerr_sim(0.0)  # chi3=0 ⇒ not a Kerr material
    res = sim.forward(n_steps=50, skip_preflight=True)
    assert res.time_series is not None and np.all(np.isfinite(res.time_series))


# --------------------------------------------------------------------------- #
# SLOW: run() actually applies Kerr, monotonically and stably.
# --------------------------------------------------------------------------- #
def _run_probe_series(chi3):
    return np.asarray(_kerr_sim(chi3).run(num_periods=30, skip_preflight=True).time_series)


@pytest.fixture(scope="module")
def kerr_run_diffs():
    base = _run_probe_series(0.0)
    denom = float(np.linalg.norm(base))
    diffs = {}
    series = {0.0: base}
    for chi3 in (1.0, 2.0, 4.0):
        s = _run_probe_series(chi3)
        series[chi3] = s
        diffs[chi3] = float(np.linalg.norm(s - base)) / denom
    return {"diffs": diffs, "series": series}


@pytest.mark.slow
def test_run_wires_kerr(kerr_run_diffs):
    """run() with χ³>0 measurably changes the field vs χ³=0 — the public API threads Kerr.
    DISCRIMINATING: if run() dropped χ³ (like forward() did), every diff would be ~0."""
    diffs = kerr_run_diffs["diffs"]
    for chi3, d in diffs.items():
        assert d > 0.3, f"χ³={chi3} barely changed the field (||Δ||/||E||={d:.4f}) — Kerr not wired?"


@pytest.mark.slow
def test_run_kerr_monotonic_in_chi3(kerr_run_diffs):
    """Stronger χ³ ⇒ larger field change (monotone). A wrong-sign or clipped coupling breaks this."""
    d = kerr_run_diffs["diffs"]
    assert d[4.0] > d[2.0] > d[1.0], f"non-monotone in χ³: {d}"


@pytest.mark.slow
def test_run_kerr_stable_and_bounded(kerr_run_diffs):
    """The nonlinear run stays finite and bounded (no ADE blow-up); the change is physical,
    not a divergence (||Δ||/||E|| well below the ~√2 two-uncorrelated-signals ceiling)."""
    for chi3, s in kerr_run_diffs["series"].items():
        assert np.all(np.isfinite(s)), f"non-finite field at χ³={chi3}"
    assert kerr_run_diffs["diffs"][4.0] < 1.2, "field change looks like a blow-up, not Kerr"
