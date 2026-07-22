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

The run() oracle gates the REACTIVE Kerr (#437: apply_kerr_ade is now ε_eff=ε_r+χ³|E|²
increment-scaling, lossless — NOT the pre-#437 dissipative absorber). Robust, discriminating
observables: χ³=0 byte-identical to a linear run; the probe field-difference ‖E(χ³)−E(0)‖ is
monotone in χ³ (0.25/0.46/0.88 @ χ³=0.5/1/2) and finite; and — the key reactive signature —
the peak field amplitude is PRESERVED (0.93–1.14× across χ³, a lossless index change), which
the pre-#437 dissipative operator would have driven well below 1. The exact quantitative SPM
magnitude (matching k0·L·χ³⟨E²⟩/(2n0) to a few %) still needs a clean CW ⟨E²⟩ comparator; the
operator's correctness against analytic ε_eff is pinned in 1D by
docs/research_notes/experiments/kerr_reactive_1d_verify.py.
Harness: docs/research_notes/experiments/kerr_spm_*.py, kerr_operator_decider.py.
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


CHI3S = (0.5, 1.0, 2.0)   # moderate reactive regime (χ³=4 is strongly nonlinear)


@pytest.fixture(scope="module")
def kerr_run_diffs():
    base = _run_probe_series(0.0)
    denom = float(np.linalg.norm(base))
    peak0 = float(np.max(np.abs(base)))
    diffs, peaks, series = {}, {}, {0.0: base}
    for chi3 in CHI3S:
        s = _run_probe_series(chi3)
        series[chi3] = s
        diffs[chi3] = float(np.linalg.norm(s - base)) / denom
        peaks[chi3] = float(np.max(np.abs(s))) / peak0
    return {"diffs": diffs, "peaks": peaks, "series": series}


@pytest.mark.slow
def test_run_wires_kerr_byte_identity_and_change(kerr_run_diffs):
    """χ³=0 is byte-identical to a truly-linear (no-Kerr) run, and χ³>0 measurably changes the
    field — the public run() API threads χ³. If run() dropped it (like forward() did), the
    χ³>0 diffs would be ~0."""
    # byte-identity: a chi3=0 Kerr material == the same geometry with a plain linear material
    lin = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    lin.add_material("lin", eps_r=EPS_R)
    lin.add(Box((X_IFACE, -1, -1), (X_SLAB_END, 1, 1)), material="lin")
    lin.add_tfsf_source(f0=F0, bandwidth=0.3, amplitude=1.0, polarization="ez",
                        direction="+x", waveform="modulated_gaussian")
    for xp in XPROBES:
        lin.add_probe((xp, DOMAIN[1] / 2, DOMAIN[2] / 2), component="ez")
    lin_ts = np.asarray(lin.run(num_periods=30, skip_preflight=True).time_series)
    assert float(np.max(np.abs(kerr_run_diffs["series"][0.0] - lin_ts))) == 0.0, \
        "chi3=0 must be byte-identical to a linear run"
    for chi3, d in kerr_run_diffs["diffs"].items():
        assert d > 0.15, f"χ³={chi3} barely changed the field (||Δ||={d:.4f}) — Kerr not wired?"


@pytest.mark.slow
def test_run_kerr_monotonic_in_chi3(kerr_run_diffs):
    """Stronger χ³ ⇒ larger field change (monotone). A wrong-sign/clipped coupling breaks this."""
    d = kerr_run_diffs["diffs"]
    assert d[2.0] > d[1.0] > d[0.5], f"non-monotone in χ³: {d}"


@pytest.mark.slow
def test_run_kerr_reactive_lossless_and_stable(kerr_run_diffs):
    """The KEY #437 signature: the reactive Kerr PRESERVES the field amplitude (a lossless index
    change), unlike the pre-#437 dissipative absorber which drove |E| well below 1. Peak-amplitude
    ratio stays near 1 across χ³, and every run is finite."""
    for chi3, s in kerr_run_diffs["series"].items():
        assert np.all(np.isfinite(s)), f"non-finite field at χ³={chi3}"
    for chi3, r in kerr_run_diffs["peaks"].items():
        assert 0.7 < r < 1.5, (
            f"χ³={chi3}: peak|E| ratio {r:.3f} — reactive Kerr must preserve amplitude "
            "(a dissipative absorber would push it far below 1)")
