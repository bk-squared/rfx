"""dz-only dispatch honesty across every public S-matrix entry point (#811).

``compute_waveguide_s_matrix`` gated its non-uniform lane on ``dx_profile``
/ ``dy_profile`` only, so a simulation whose ONLY profile was ``dz_profile``
was silently solved on the uniform grid built from the scalar ``dx`` —
while ``preflight()`` described the graded mesh the solve never used. The
six sibling S-matrix dispatches all tested dz. This file locks the CLASS:

- every public ``compute_*`` S-matrix entry point has a declared dz-only
  contract row, and the enumeration fails when a new entry point ships
  without one;
- the fixed waveguide dispatch actually enters the NU lane on dz-only
  (sentinel witness, no FDTD) and fails loud on the default
  ``normalize=False`` instead of returning uniform-mesh numbers;
- the already-correct sibling gates are locked against regression;
- ``run()`` / ``forward()`` lane selection includes dz (was already
  correct — locked);
- an AST scan holds the whole dispatch family to "a predicate that tests
  dx_profile and dy_profile must test dz_profile too";
- a slow_physics falsifier asserts two genuinely DIFFERENT graded z meshes
  change the answer (a uniform-valued profile tests plumbing, never NU
  metrics — rfx-known-issues lesson).

No module-level jax x64 config here (process-global; contaminates the
shard), and no FDTD outside the slow_physics falsifier.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from rfx import Box, GaussianPulse
from rfx.api import Simulation
from rfx.boundaries.spec import BoundarySpec, Boundary

_REPO = Path(__file__).resolve().parents[1]

# Genuinely graded z meshes (adjacent ratios <= 1.4). Both sum EXACTLY to
# the WR-90 narrow wall b = 10.16 mm; their minimum cells differ, so their
# dt differ (~1.55x) — bit-identical S across them means dz never reached
# the solve.
_DZ_WR90_A = np.concatenate([
    np.full(10, 0.40e-3), np.full(3, 0.52e-3),
    np.full(2, 0.70e-3), np.full(4, 0.80e-3),
])
_DZ_WR90_C = np.concatenate([
    np.full(6, 0.80e-3), np.full(4, 0.62e-3), np.full(4, 0.72e-3),
])

_A_WG = 0.02286
_B_WG = 0.01016


# ---------------------------------------------------------------------------
# The contract. Keys must enumerate every public compute_* entry point on
# Simulation; values state what a dz-ONLY graded mesh must do there.
#   "nu-lane" — dispatches to that family's non-uniform lane
#   "raises"  — refuses loudly, naming the profile restriction
# ---------------------------------------------------------------------------
DZ_ONLY_CONTRACT = {
    "compute_waveguide_s_matrix": "nu-lane",   # THE #811 fix
    "compute_msl_s_matrix": "nu-lane",         # dz-aware since the NU MSL lane
    "compute_mixed_s_matrix": "raises",
    "compute_coaxial_s_matrix": "raises",
    "compute_coaxial_line_reflection": "raises",
    "compute_coaxial_two_port": "raises",
    "compute_coax_msl_transition": "raises",
}


def test_every_compute_entry_point_has_a_dz_only_contract_row():
    surface = sorted(
        n for n in dir(Simulation)
        if n.startswith("compute_") and callable(getattr(Simulation, n))
    )
    missing = sorted(set(surface) - set(DZ_ONLY_CONTRACT))
    stale = sorted(set(DZ_ONLY_CONTRACT) - set(surface))
    assert not missing, (
        f"public compute_* entry points without a dz-only contract row: "
        f"{missing}. Decide the dz-only behaviour and add a row + test."
    )
    assert not stale, f"contract rows for entry points that no longer exist: {stale}"


# ---------------------------------------------------------------------------
# Waveguide: the fixed dispatch.
# ---------------------------------------------------------------------------

def _dz_only_wg(dz_profile: np.ndarray) -> Simulation:
    sim = Simulation(
        freq_max=12.4e9,
        domain=(0.06, _A_WG, _B_WG),
        dx=1e-3,
        dz_profile=dz_profile,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )
    for x_position, direction, name in ((0.012, "+x", "wg1"),
                                        (0.048, "-x", "wg2")):
        sim.add_waveguide_port(
            x_position, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=np.linspace(8.2e9, 12.4e9, 3), f0=10.3e9, bandwidth=0.5,
            name=name)
    return sim


def test_waveguide_dz_only_dispatches_to_the_nonuniform_lane(monkeypatch):
    """Sentinel witness, no FDTD: a dz-only sim must ENTER the NU lane.

    Pre-#811 this test fails with the sentinel never raised: the call
    completed on the uniform lane and returned uniform-mesh numbers."""
    def boom(self, **kw):
        raise RuntimeError("NU-LANE-ENTERED")

    monkeypatch.setattr(Simulation, "_compute_waveguide_s_matrix_nu", boom)
    sim = _dz_only_wg(_DZ_WR90_A)
    with pytest.raises(RuntimeError, match="NU-LANE-ENTERED"):
        with pytest.warns(Warning):
            # the NU lane's far-port absorber advisory fires first (thin
            # absorber vs lambda_g on this deliberately small fixture) —
            # asserted so the witness does not silently swallow it
            sim.compute_waveguide_s_matrix(n_steps=1, normalize="flux")


def test_waveguide_dz_only_default_normalize_fails_loud():
    """dz-only + the DEFAULT normalize=False must raise the NU fence — the
    pre-#811 behaviour was to silently return uniform-grid numbers. The
    fence message must name dz_profile so the remedy is actionable."""
    sim = _dz_only_wg(_DZ_WR90_A)
    with pytest.raises(NotImplementedError, match="dz_profile"):
        sim.compute_waveguide_s_matrix(n_steps=1)


# ---------------------------------------------------------------------------
# Siblings: already dz-aware — locked against regression.
# ---------------------------------------------------------------------------

def test_msl_dz_only_reaches_the_nu_lane():
    """A dz-only graded mesh must get PAST the MSL NU fence (laplace feed).
    This minimal geometry has no PEC trace, so reaching the extractor's
    trace-PEC RuntimeError proves the dispatch took the NU lane."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.006, 0.002),
        dx=0.5e-3,
        dz_profile=np.array([0.40e-3, 0.45e-3, 0.55e-3, 0.60e-3]),
        boundary="pec",
    )
    sim.add_msl_port(position=(0.004, 0.003, 0.0), width=0.5e-3,
                     height=0.5e-3, direction="+x", mode="laplace")
    with pytest.raises(RuntimeError, match="no PEC trace conductor"):
        sim.compute_msl_s_matrix(n_steps=1)


def test_mixed_dz_only_raises():
    _DX = 0.25e-3
    _H_SUB = 0.5e-3
    _W_TRACE = 0.74e-3
    lx, ly, lz = 8e-3, 4e-3, 2.5e-3
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        dz_profile=np.full(10, lz / 10),
    )
    sim.add_material("sub", eps_r=4.3)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + _DX)), material="pec")
    sim.add_msl_port(position=(5.5e-3, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5))
    sim.add_port(position=(2e-3, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_H_SUB)
    with pytest.raises(NotImplementedError, match="uniform mesh"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


def test_coaxial_s_matrix_dz_only_raises():
    sim = Simulation(freq_max=26e9, domain=(0.020, 0.020, 0.020),
                     boundary="pec", dz_profile=np.full(20, 1e-3))
    sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
    with pytest.raises(NotImplementedError, match="uniform Yee lane only"):
        sim.compute_coaxial_s_matrix(n_steps=1, n_freqs=1)


def test_coaxial_line_reflection_dz_only_raises():
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40e9,
                     boundary="cpml", dz_profile=np.full(40, 1e-3))
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5e-3,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    with pytest.raises(ValueError, match="dz_profile"):
        sim.compute_coaxial_line_reflection(termination="short", n_steps=1,
                                            n_freqs=1)


def test_coaxial_two_port_dz_only_raises():
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40e9,
                     boundary="cpml", dz_profile=np.full(40, 1e-3))
    for z, face in ((0.005, "bottom"), (0.035, "top")):
        sim.add_coaxial_port((0.004, 0.004, z), face=face, pin_length=5e-3,
                             waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    with pytest.raises(ValueError, match="dz_profile"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_coax_msl_transition_dz_only_raises():
    sim = Simulation(domain=(0.010, 0.010, 0.006), freq_max=20e9,
                     boundary="cpml", dz_profile=np.full(12, 0.5e-3))
    sim.add_coaxial_port((0.005, 0.005, 0.005), face="top", pin_length=2e-3,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    sim.add_msl_port(position=(0.008, 0.005, 0.0), width=0.5e-3,
                     height=0.5e-3, direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    with pytest.raises(ValueError, match="dz_profile"):
        sim.compute_coax_msl_transition(junction_x=0.005, n_steps=1)


# ---------------------------------------------------------------------------
# run()/forward() lane selection (was already dz-aware — locked).
# ---------------------------------------------------------------------------

def test_run_and_forward_dz_only_pick_nonuniform_lanes():
    sim = Simulation(freq_max=12.4e9, domain=(0.02, 0.01, _B_WG), dx=1e-3,
                     dz_profile=_DZ_WR90_C, boundary="cpml", cpml_layers=4)
    plan_run = sim._dispatch_plan(mode="run", n_steps=8, num_periods=20.0)
    plan_fwd = sim._dispatch_plan(mode="forward", n_steps=8, num_periods=20.0)
    assert plan_run.lane == "run_nonuniform", plan_run
    assert plan_fwd.lane == "fwd_nonuniform", plan_fwd


# ---------------------------------------------------------------------------
# Source-level class lock: a dispatch predicate that tests dx_profile AND
# dy_profile must test dz_profile too. rfx/api/__init__.py is excluded on
# purpose: its ADI guard tests dx/dy only because ADI's z-graded (ZCZ) lane
# ACCEPTS dz_profile — a deliberate per-axis check, not this defect class.
# ---------------------------------------------------------------------------
_DISPATCH_FILES = (
    "rfx/api/_sparams.py",
    "rfx/api/_preflight.py",
    "rfx/api/_execute.py",
    "rfx/api/_compile.py",
    "rfx/optimize.py",
    "rfx/visualize.py",
)


def test_no_transverse_only_dispatch_predicate_remains():
    offenders = []
    for rel in _DISPATCH_FILES:
        src = (_REPO / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp):
                continue
            seg = ast.get_source_segment(src, node) or ""
            if ("_dx_profile is not None" in seg
                    and "_dy_profile is not None" in seg
                    and "_dz_profile" not in seg):
                offenders.append(f"{rel}:{node.lineno}: {' '.join(seg.split())}")
    assert not offenders, (
        "profile predicates that test dx_profile and dy_profile but not "
        "dz_profile — the #811 defect class:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Physics falsifier (FDTD; slow lane): two genuinely different graded z
# meshes must change the answer. Never replace these with uniform-valued
# profiles — primal==dual grids hide exactly the defect this exists to
# catch.
# ---------------------------------------------------------------------------

@pytest.mark.slow_physics
def test_two_different_z_meshes_change_the_answer():
    import warnings

    results = {}
    for label, dz in (("A", _DZ_WR90_A), ("C", _DZ_WR90_C)):
        sim = _dz_only_wg(dz)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_waveguide_s_matrix(num_periods=2,
                                                 normalize="flux")
        results[label] = np.asarray(res.s_params)

    sa, sc = results["A"], results["C"]
    # R5-style dump: the full per-bin trace, not a bare verdict.
    print("\n[dz-dispatch] per-bin |S| for two graded z meshes:")
    for label, s in results.items():
        print(f"  mesh {label}: |S11|={np.abs(s[0, 0, :])} "
              f"|S21|={np.abs(s[1, 0, :])}")
    print(f"  max|dS| = {np.max(np.abs(sa - sc)):.6e}")

    assert not np.array_equal(sa, sc), (
        "two genuinely different z meshes returned BIT-IDENTICAL "
        "S-parameters — dz_profile is not reaching the solve (#811 class)"
    )
