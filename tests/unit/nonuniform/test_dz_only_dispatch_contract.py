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
  metrics).

No module-level jax x64 config here (process-global; contaminates the
shard), and no FDTD outside the slow_physics tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from rfx import Box, GaussianPulse
from rfx.api import Simulation
from rfx.boundaries.spec import BoundarySpec, Boundary

_REPO = Path(__file__).resolve().parents[3]

_DZ_WR90_A = np.concatenate([
    np.full(10, 0.40e-3), np.full(3, 0.52e-3),
    np.full(2, 0.70e-3), np.full(4, 0.80e-3),
])
_DZ_WR90_C = np.concatenate([
    np.full(6, 0.80e-3), np.full(4, 0.62e-3), np.full(4, 0.72e-3),
])

_A_WG = 0.02286
_B_WG = 0.01016


DZ_ONLY_CONTRACT = {
    "compute_waveguide_s_matrix": "nu-lane",
    "compute_msl_s_matrix": "nu-lane",
    "compute_mixed_s_matrix": "raises",
    "compute_coaxial_line_reflection": "raises",
    "compute_coaxial_two_port": "raises",
    "compute_coax_msl_transition": "raises",
    "compute_s_matrix": "delegates",
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
    completed on the uniform lane and returned uniform-mesh numbers.
    The thin absorber relative to the guide wavelength on the small fixture causes the far-port advisory before the NU-lane sentinel."""
    def boom(self, **kw):
        raise RuntimeError("NU-LANE-ENTERED")

    monkeypatch.setattr(Simulation, "_compute_waveguide_s_matrix_nu", boom)
    sim = _dz_only_wg(_DZ_WR90_A)
    with pytest.raises(RuntimeError, match="NU-LANE-ENTERED"):
        with pytest.warns(Warning):
            sim.compute_waveguide_s_matrix(n_steps=1, normalize="flux")


def test_waveguide_dz_only_default_normalize_fails_loud():
    """dz-only + the DEFAULT normalize=False must raise the NU fence — the
    pre-#811 behaviour was to silently return uniform-grid numbers. The
    fence message must name dz_profile so the remedy is actionable."""
    sim = _dz_only_wg(_DZ_WR90_A)
    with pytest.raises(NotImplementedError, match="dz_profile"):
        sim.compute_waveguide_s_matrix(n_steps=1)



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
    with pytest.raises(RuntimeError,
                       match="no realized PEC trace conductor"):
        sim.compute_msl_s_matrix(n_steps=1)


def test_a_sheet_declared_msl_trace_is_found_by_the_detector():
    """Positive control for the negative test above (#931 §1.9).

    The witness in ``test_msl_dz_only_reaches_the_nu_lane`` is a RAISE, so
    on its own it is satisfied by a detector that finds NOTHING ever. Until
    #931 that was a live risk in the other direction: the detector scanned
    the primal-cell ``pec_mask`` column above the substrate, and a sheet
    owns no cell, so declaring the trace as a foil — the declaration the
    contract asks for — turned every working MSL fixture into that same
    RuntimeError. The detector reads realized WALL PLANES now, so a sheet
    trace answers with its single plane and a volume trace with its two.
    Build-time only: no solve.
    """
    from rfx.probes.msl_wave_decomp import realized_trace_planes_on_column
    from tests._realized_geometry import node_index, realized

    h_sub, w_trace, dx = 0.5e-3, 1.0e-3, 0.25e-3
    lx, ly, lz = 4e-3, 4e-3, 2e-3
    y_c = ly / 2.0

    def _sim(sheet: bool):
        s = Simulation(freq_max=10e9, domain=(lx, ly, lz), dx=dx,
                       boundary="pec")
        s.add_material("sub", eps_r=4.3)
        s.add(Box((0.0, 0.0, 0.0), (lx, ly, h_sub)), material="sub")
        z_hi = h_sub if sheet else h_sub + dx
        s.add(Box((0.0, y_c - w_trace / 2, h_sub),
                  (lx, y_c + w_trace / 2, z_hi)), material="pec")
        return s

    for sheet, expect_span in ((True, 0), (False, 1)):
        rz = realized(_sim(sheet))
        assert len(rz.sheets) == (1 if sheet else 0)
        i = node_index(rz.grid, 0, lx / 2)
        j = node_index(rz.grid, 1, y_c)
        k_sub = node_index(rz.grid, 2, h_sub)
        lo, hi = realized_trace_planes_on_column(
            rz.edge_masks, 2, (i, j), k_from=k_sub)
        assert lo == k_sub, (
            f"sheet={sheet}: the trace must be found at the plane it is "
            f"drawn on ({k_sub}), got {lo}")
        assert hi - lo == expect_span, (
            "a sheet is ONE plane; a one-cell volume is two")


def test_mixed_dz_only_raises():
    """The MSL trace is foil, so its sheet declaration is a zero-extent axis on the substrate-top node plane."""
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
    sim.add(Box((0.0, y_c - (1 + .35) * _DX, _H_SUB),
                (lx, y_c + (1 + .35) * _DX, _H_SUB)), material="pec")
    sim.add_msl_port(position=(5.5e-3, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5))
    sim.add_port(position=(2e-3, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_H_SUB, terminates=1)
    with pytest.raises(NotImplementedError, match="uniform mesh"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


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


def test_compute_s_matrix_dz_only_is_exactly_the_delegates_answer(monkeypatch):
    """The #980 dispatcher must add no mesh behaviour of its own.

    Both directions, because "delegates" is only an honest contract row if it
    is unfalsifiable in neither:

    * an "nu-lane" delegate must still REACH the NU lane through the
      dispatcher (same sentinel as the direct waveguide test above), and
    * a "raises" delegate must still raise ITS OWN fence, with its own
      message naming dz_profile -- not a generic dispatcher-level refusal
      that would hide which restriction was hit.
    """
    def boom(self, **kw):
        raise RuntimeError("NU-LANE-ENTERED")

    monkeypatch.setattr(Simulation, "_compute_waveguide_s_matrix_nu", boom)
    sim = _dz_only_wg(_DZ_WR90_A)
    with pytest.raises(RuntimeError, match="NU-LANE-ENTERED"):
        with pytest.warns(Warning):
            sim.compute_s_matrix(n_steps=1, normalize="flux")

    coax = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40e9,
                      boundary="cpml", dz_profile=np.full(40, 1e-3))
    coax.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5e-3,
                          waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    with pytest.raises(ValueError, match="dz_profile"):
        coax.compute_s_matrix(method="compute_coaxial_two_port", n_steps=1,
                              n_freqs=1)


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



def test_run_and_forward_dz_only_pick_nonuniform_lanes():
    sim = Simulation(freq_max=12.4e9, domain=(0.02, 0.01, _B_WG), dx=1e-3,
                     dz_profile=_DZ_WR90_C, boundary="cpml", cpml_layers=4)
    plan_run = sim._dispatch_plan(mode="run", n_steps=8, num_periods=20.0)
    plan_fwd = sim._dispatch_plan(mode="forward", n_steps=8, num_periods=20.0)
    assert plan_run.lane == "run_nonuniform", plan_run
    assert plan_fwd.lane == "fwd_nonuniform", plan_fwd


_DISPATCH_FILES = (
    "rfx/api/_sparams.py",
    *sorted(str(p.relative_to(_REPO)) for p in (_REPO / "rfx/sparams").glob("*.py")),
    "rfx/measurement/sparam_assembly.py",
    "rfx/measurement/sparam_advisories.py",
    "rfx/api/_preflight.py",
    *sorted(str(p.relative_to(_REPO)) for p in (_REPO / "rfx/preflight").glob("*.py")),
    "rfx/api/_execute.py",
    "rfx/api/_compile.py",
    "rfx/optimize.py",
    "rfx/visualize.py",
)


@pytest.mark.source_scan
def test_no_transverse_only_dispatch_predicate_remains():
    """Package-wide scanning prevents vacuous coverage when a calculator or preflight predicate lives outside the API facade."""
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



def test_two_different_z_meshes_change_the_answer():
    import warnings

    assert (all(np.isclose(np.sum(dz), _B_WG, rtol=0, atol=1e-14)
                and np.ptp(dz) > 0
                and np.max(np.maximum(dz[1:] / dz[:-1], dz[:-1] / dz[1:])) <= 1.4
                for dz in (_DZ_WR90_A, _DZ_WR90_C))
            and np.min(_DZ_WR90_A) != np.min(_DZ_WR90_C)), (
        'The WR-90 witness needs equal wall spans, distinct minima and genuinely graded cells so uniform primal/dual coincidence cannot mask a discarded z profile.')
    results = {}
    for label, dz in (("A", _DZ_WR90_A), ("C", _DZ_WR90_C)):
        sim = _dz_only_wg(dz)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_waveguide_s_matrix(num_periods=2,
                                                 normalize="flux")
        results[label] = np.asarray(res.s_params)

    sa, sc = results["A"], results["C"]
    print("\n[dz-dispatch] per-bin |S| for two graded z meshes:")
    for label, s in results.items():
        print(f"  mesh {label}: |S11|={np.abs(s[0, 0, :])} "
              f"|S21|={np.abs(s[1, 0, :])}")
    print(f"  max|dS| = {np.max(np.abs(sa - sc)):.6e}")

    assert not np.array_equal(sa, sc), (
        "two genuinely different z meshes returned BIT-IDENTICAL "
        "S-parameters — dz_profile is not reaching the solve (#811 class)"
    )


def test_dz_only_short_run_fires_the_settling_witness():
    """A truncated dz-only run reports an unavailable post-source witness."""
    sim = _dz_only_wg(_DZ_WR90_A)
    with pytest.warns(UserWarning, match="ring-down settling witness NOT PASSED"):
        res = sim.compute_waveguide_s_matrix(num_periods=2, normalize="flux")
    assert res.settling_db is not None, (
        "NU lane returned settling_db=None — the #827 witness gap is back")
    sdb = np.asarray(res.settling_db, dtype=float)
    assert sdb.shape == (2,)
    print(f"\n[dz-settling] settling_db per drive = {sdb} dB")
    assert np.isnan(sdb).all(), sdb
    assert len(res.settling_witness) == 2
    for detail in res.settling_witness:
        assert detail["status"] == "undetermined"
        assert "source end is unavailable" in detail["reason"]
