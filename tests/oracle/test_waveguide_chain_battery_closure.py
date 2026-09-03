"""Interior power-closure witness for the WR-90 chain battery (v1.8 WP3).

Plan: ``docs/design_notes/v18_waveguide_s_chain_plan.md``, "WP3 —
power-closure witnesses", item 2. Contract: ``docs/design_notes/
chain_closure_contract.md``, criterion 2 ("power closure ``|1 − Σ_i |S_ij|²| ≤
tol_c`` on a lossless DUT"). Fixture geometry, ladder, drive and absorber:
``tests/_waveguide_chain_battery_fixture.py`` (WP2, unchanged here).

WHY THIS TEST EXISTS. ``physics_gates[*].power_closure_gate`` in
``tests/fixtures/waveguide_chain_battery/fixture.json`` reads
``"report-only (WP3)"``: the battery measures ``1 − |S11|² − |S21|²`` but
nothing checks that the number means what it says. It cannot check itself.
The flux lane builds |S|² from Poynting integrals taken at the two PORT PROBE
planes (``rfx/sources/waveguide_port.py::extract_waveguide_s_matrix_flux``,
``|S_ii|² = |F_ref − F_dev| / |F_ref|``, ``|S_ji|² = |F_dev[j]| / |F_ref[i]|``),
so ``1 − Σ|S|²`` is algebraically ``(F_dev[in] − F_dev[out]) / F_ref[in]``
evaluated at those planes. The port column power reuses the same integrals.
Port closure and the S-matrix are therefore ONE witness, not two.

THE INDEPENDENT WITNESS. The same power balance, evaluated at two planes the
S-extraction never touches: two full-plane ``add_flux_monitor`` planes placed
INSIDE the guide, one between the left port probe plane and the slab, one
between the slab and the right port probe plane. Route A and route B share the
lossless guide between them, so they must agree; what they do not share is the
plane index.

INDEPENDENCE SCOPE — ONE axis, the plane index (memory:
``feedback_agreement_is_not_independence``). An earlier draft of this docstring
claimed a second independent axis, the area weighting, and that claim was wrong:
both routes integrate with the SAME uniform ``dx²`` weight over the same NONZERO
transverse support on this fixture. ``extract_waveguide_s_matrix_flux``'s
``_make_flux_monitors`` (``rfx/sources/waveguide_port.py``) passes
``d1 = d2 = cfg.dx`` with ``lo/hi`` from the port's ``u/v`` span and never passes
``cfg.aperture_dA``; the public ``add_flux_monitor`` path
(``rfx/runners/uniform.py``) passes ``d1 = d2 = grid.dx`` over the full grid
plane. ``dA = 6.4516003e-06`` for both.

The two INDEX ranges are not identical: route B spans ``u[0, 10)``, ``v[0, 5)``
(the whole grid plane) while the port lane spans the guide's cells,
``u[0, 9)``, ``v[0, 4)`` — the port aperture is a cell span, not the node span
that produced it (issue #868). That difference contributes nothing, measured
rather than argued: on the port plane the row ``u = 9`` and the column
``v = 4`` carry ``Re(Ey·Hz* − Ez·Hy*)·dA = 0.000e+00`` at every one of the 17
bins, because both tangential E components are exactly zero there (the PEC wall
node, and the unused outermost slot), against a full-plane flux of 1.2e-22 at
the band centre. ``aperture_dA`` serves the modal V/I integral and reaches
neither monitor.

So route B calls the same ``rfx/probes/probes.py::flux_spectrum`` kernel with the
same weights at a different plane.

NOT caught, each checked rather than asserted:
  * a shared Poynting kernel defect that scales every plane by the same factor,
    and any area-weighting error — both cancel in the two ratios alike;
  * the reference-plane de-embedding. Both routes are MAGNITUDE-only. In
    ``extract_waveguide_s_matrix_flux`` the shift enters through ``ref_shifts``
    at ``rfx/sources/waveguide_port.py:2149`` and ``:2191``, and both uses feed
    ``extract_waveguide_port_waves`` whose result reaches only
    ``phase = jnp.angle(ratio)``; the magnitude is ``sqrt(P_num / safe_P_inc)``,
    built from flux alone, so ``|S|`` and ``closure_S = 1 − |S11|² − |S21|²`` do
    not depend on the shift by construction. Measured on this fixture, with a
    positive control that the shift really acted: moving the left reference
    plane five coarse cells (0.02032 → 0.03302 m, 12.7 mm) swings ``∠S11`` by
    277.3°, while ``closure_S`` moves 1.09e-07 and ``|S|`` moves 7.0e-08 —
    against a 0.02 gate and a committed ``max|closure_S| = 9.033264e-05``. The
    one-cell port aperture cutoff error (issue #868) therefore cannot reach this
    witness.

CAUGHT: any failure of power transport in the guide between the two plane pairs
— a lossy or reflecting feature between them, an absorber reaching in, energy
appearing where none is fed. A wrong plane INDEX is caught only when the plane
mis-snaps into the slab or the absorber: in a lossless source-free region the
closure residual is plane-invariant, which the artifact's own numbers show
(port planes k=15/33 give max|closure_S| = 9.033e-05, interior planes k=18/30
give max|closure_M| = 6.887e-05, agreeing to 2.146e-05 for a three-cell move).

Nor does it bound the far absorber. Both routes divide by a reference run's net
flux, so a reflection there biases ``F_ref`` and the flux lane's ``F_ref[i]``
by the same factor and cancels; a travelling backward wave carries the same
power at every plane, so the empty-guide transport check below cannot see it
either. The absorber is bounded by the battery's own thru gates, not here.

PRE-DECLARATION (fixed before the first run of this measurement; §-numbers
refer to ``docs/design_notes/waveguide_chain_battery_predeclaration.md``):

* Fixture: the WP2 ``slab`` DUT — lossless ``eps_r = 4`` filling the full
  WR-90 cross-section, ``x ∈ [0.05588, 0.06604) m`` — at the COARSE rung
  ``dx = 2.54 mm`` (chosen for CPU wall time; the mid and fine rungs are the
  same construction and are not measured here).
* Route A: ``compute_waveguide_s_matrix(normalize="flux", num_periods=40)`` on
  the unmodified two-port fixture; per bin
  ``closure_S(f) = 1 − |S11(f)|² − |S21(f)|²`` from column 0 (left drive).
* Route B: two x-normal full-plane monitors at ``18·dx_coarse = 0.04572 m``
  (``guide_in``) and ``30·dx_coarse = 0.07620 m`` (``guide_out``) — 4 coarse cells
  outside the nearer slab face, 6 either side of the slab centre, and strictly
  between the port probe planes (0.03810 / 0.08382 m) and the slab. Two
  single-port runs of the SAME builder at the same rung, drive and record
  length: the ``slab`` device run and a ``thru`` reference run.
  ``closure_M(f) = [F_dev(guide_in,f) − F_dev(guide_out,f)] / F_ref(guide_in,f)``.
  ``F_ref(guide_in)`` is the incident power at that plane, the same role
  ``F_ref[i]`` plays in the flux lane's own ratios.
* GATE: ``max_f |closure_S(f) − closure_M(f)| ≤ 0.02`` — the column-power
  tolerance the plan's WP3 Falsifier names.
* Branches, fixed before the run: (i) ≤ 0.02 → pass, numbers recorded in the
  fixture README; (ii) > 0.02 → the tolerance is NOT widened, the test is
  ``xfail(strict=True)`` carrying the measured number and the PR body names
  which route is suspect with the evidence; (iii) anything outside both — a
  monitor reading exactly zero, a non-positive reference flux, a settling
  witness above −40 dB — is reported as NON-CLOSING and no verdict is drawn.
* Recorded with every number (R5 + repo rule 10): preflight findings verbatim
  for all three solves, ``settling_db`` per drive, the per-bin dumps of
  ``F_dev(guide_in)``, ``F_dev(guide_out)``, ``F_ref(guide_in)``, ``|S11|²``, ``|S21|²``,
  ``closure_S``, ``closure_M`` and their difference.

MEASURED (one run, branch (i)): ``max_f |closure_S − closure_M| = 2.146e-05``
at 8.60 GHz, against the 0.02 gate; band centre 3.23e-06. Artifact
``tests/fixtures/waveguide_chain_battery/closure_witness.json``, key
``max_abs_diff``. Read honestly: BOTH routes put the closure residual at
~1e-05 (``max|closure_S| = 9.03e-05``, ``max|closure_M| = 6.89e-05``), i.e. at
the float32 field-noise floor of this rung, so the measurement bounds the
disagreement rather than resolving a physical closure defect. The gate is
still load-bearing — a 5 % power-balance error makes it red (the refute
below) — and the agreement is not host-side cancellation: re-summing the same
accumulators in float64 moves ``closure_M`` by 3.08e-07.

Layers, mirroring ``tests/test_waveguide_chain_battery.py``:

* REPLAY (fast, no FDTD) — reads
  ``tests/fixtures/waveguide_chain_battery/closure_witness.json`` and
  re-asserts the gate, the plane geometry, the settling witness and the
  recorded verdict.
* LIVE (also fast: 13.5 s wall for all three solves on this box's CPU, under
  the contract's 30 s line) — re-measures and re-asserts the gate against
  physics rather than against the artifact.

Must-not list honoured: ``normalize=True`` never enters; nothing from
``rfx/probes/refplane.py`` is imported; ``fixture.json`` and every WP2 gate
are untouched; no committed tolerance is moved.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import flux_spectrum
from rfx.sources.waveguide_port import settling_db_from_port_records

from tests import _waveguide_chain_battery_fixture as F
from tests import _waveguide_chain_battery_gates as G

REPO = Path(__file__).resolve().parents[2]
WITNESS = REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "closure_witness.json"

SCHEMA = "rfx.waveguide_chain_battery_closure"
SCHEMA_VERSION = 1

# --- pre-declared placement, in coarse cells (the builder's own integers) ---
K_MON_IN = 18       # 0.04572 m: between the left probe plane (k=15) and the slab (k=22)
K_MON_OUT = 30      # 0.07620 m: between the slab (k=26) and the right probe plane (k=33)
MON_IN_X_M = K_MON_IN * F.DX_COARSE
MON_OUT_X_M = K_MON_OUT * F.DX_COARSE
MON_IN_NAME = "guide_in"
MON_OUT_NAME = "guide_out"

RUNG = "coarse"
RUNG_DX_M = F.DX_COARSE
DUT = "slab"
REFERENCE_DUT = "thru"

# Plan WP3, "Falsifier": interior-monitor closure and port closure agree
# within 0.02, the column-power tolerance. Not derived from an envelope and
# not moved: a larger disagreement means one of the two routes is wrong and is
# reported as such.
CLOSURE_ABS_DIFF_GATE = 0.02
SETTLING_DB_MAX = -40.0

# Non-vacuity: an empty guide would make the closure identity trivially true
# at both plane pairs (#395, the contract's "never an empty guide"). The slab
# reflects; the fixture's own gate is max|S11| > 0.20 on the reflecting DUTs.
NON_VACUITY_MIN_S11 = 0.20


# ---------------------------------------------------------------------------
# the gate's own arithmetic — ONE copy
# ---------------------------------------------------------------------------

def interior_closure(f_in, f_out, f_ref) -> np.ndarray:
    """Route B per bin: ``[F_dev(in) − F_dev(out)] / F_ref(in)``."""
    return (np.asarray(f_in, dtype=float) - np.asarray(f_out, dtype=float)) \
        / np.asarray(f_ref, dtype=float)


def route_disagreement(closure_s, closure_m) -> float:
    """The gated quantity: ``max_f |closure_S − closure_M|``.

    The gate test, the live test and the cheap refute all call THIS, so the
    refute exercises the gate's own expression instead of re-deriving a
    parallel one (memory: ``feedback_gate_can_bind_artifact``, "a falsifier
    that re-derives instead of calling the gate's expression proves nothing").
    """
    return float(np.max(np.abs(np.asarray(closure_s, dtype=float)
                               - np.asarray(closure_m, dtype=float))))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def build_monitor_simulation(dut: str, dx: float = RUNG_DX_M):
    """The WP2 fixture for ``dut`` with the right port dropped and the two
    interior flux planes registered.

    ``Simulation.run()`` accepts exactly one waveguide port
    (``rfx/runners/uniform.py``, "supports only a single waveguide port"), and
    the flux-lane extractor drives one port at a time with every other port at
    ``src_amp = 0`` — a passive port injects nothing (the ``src_amp`` scaling
    in ``apply_waveguide_port_e`` / ``_h``) and only records. Dropping the
    right entry is therefore field-identical to the extractor's left-drive run;
    ``test_monitor_simulation_matches_the_battery_fixture`` proves it on the
    realized grid and the port config rather than asserting it in prose.
    """
    sim = F.build_simulation(dut, dx)
    assert sim._waveguide_ports[0].name == F.PORT_NAMES[0]
    del sim._waveguide_ports[1]
    freqs = jnp.asarray(F.FREQS)
    sim.add_flux_monitor(axis="x", coordinate=MON_IN_X_M, freqs=freqs, name=MON_IN_NAME)
    sim.add_flux_monitor(axis="x", coordinate=MON_OUT_X_M, freqs=freqs, name=MON_OUT_NAME)
    return sim


def _preflight_findings(sim) -> list[dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    return [{"code": getattr(i, "code", "uncoded"),
             "severity": getattr(i, "severity", "warning"),
             "message": str(i)} for i in report]


def _dedupe_warnings(wlist) -> list[dict]:
    seen: dict[str, int] = {}
    for w in wlist:
        key = f"{w.category.__name__}: {w.message}"
        seen[key] = seen.get(key, 0) + 1
    return [{"message": k, "count": n} for k, n in seen.items()]


def _run_monitor_solve(dut: str, dx: float = RUNG_DX_M) -> dict:
    """One single-port solve of ``dut`` with the interior monitors, recorded
    with its preflight findings, warnings, settling witness and per-bin flux."""
    sim = build_monitor_simulation(dut, dx)
    findings = _preflight_findings(sim)
    grid = sim._build_grid()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        t0 = time.time()
        res = sim.run(num_periods=F.NUM_PERIODS)
        flux = {name: np.asarray(flux_spectrum(res.flux_monitors[name]), dtype=float)
                for name in (MON_IN_NAME, MON_OUT_NAME)}
        wall = time.time() - t0
    # Same accumulators, float64 host-side summation (``flux_spectrum``'s own
    # sanctioned #304 remedy). Diagnostic only — the gate reads the default
    # path, which is the path the flux lane itself uses.
    flux_f64 = {name: np.asarray(flux_spectrum(res.flux_monitors[name], exact_f64=True),
                                 dtype=float)
                for name in (MON_IN_NAME, MON_OUT_NAME)}
    settling = float(settling_db_from_port_records(list(res.waveguide_ports.values())))
    return {
        "dut": dut,
        "dx_m": float(dx),
        "n_steps": int(grid.num_timesteps(F.NUM_PERIODS)),
        "grid_shape": [int(x) for x in grid.shape],
        "cpml_layers": int(sim._cpml_layers),
        "preflight": findings,
        "warnings": _dedupe_warnings(wl),
        "settling_db": settling,
        "flux": {k: [float(x) for x in v] for k, v in flux.items()},
        "flux_exact_f64": {k: [float(x) for x in v] for k, v in flux_f64.items()},
        "wall_time_s": float(wall),
    }


def _run_flux_lane(dx: float = RUNG_DX_M) -> dict:
    """Route A: the battery's own two-port flux-lane S-matrix at this rung."""
    sim = F.build_simulation(DUT, dx)
    findings = _preflight_findings(sim)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        t0 = time.time()
        res = sim.compute_waveguide_s_matrix(num_periods=F.NUM_PERIODS, normalize="flux")
        S = np.asarray(res.s_params).astype(np.complex128)
        wall = time.time() - t0
    settling = np.asarray(res.settling_db, dtype=float)
    return {
        "dx_m": float(dx),
        "preflight": findings,
        "warnings": _dedupe_warnings(wl),
        "settling_db": {name: float(settling[i]) for i, name in enumerate(F.PORT_NAMES)},
        "reference_planes_m": [float(x) for x in np.asarray(res.reference_planes)],
        # One copy of the (port, port, freq) index convention: the battery's
        # own writer/replay helper, not a hand copy of it here.
        "s_params": G.s_to_json(S),
        "wall_time_s": float(wall),
    }


INDEPENDENCE_SCOPE = (
    "independent in the PLANE INDEX ONLY; both routes integrate the same NONZERO transverse "
    "support with the same uniform dA = dx^2 (dA = 6.4516003e-06) through "
    "rfx/probes/probes.py::flux_spectrum, so a shared-kernel defect or an area-weighting "
    "error cancels in both ratios; the port's aperture_dA reaches neither monitor. The two "
    "windows are not the same INDEX range: route B's full-plane monitor spans u[0,10) "
    "v[0,5) while the port lane's spans the guide's cells u[0,9) v[0,4) (issue #868 — the "
    "port aperture is a cell span, not the node span). Measured on this fixture rather "
    "than argued: on the port plane the row u=9 and the column v=4 carry Re(Ey.Hz* - "
    "Ez.Hy*).dA = 0.000e+00 at every one of the 17 bins (both tangential E components are "
    "exactly 0 there — the PEC wall node and the unused outer slot), against a full-plane "
    "flux of 1.2e-22 at the band centre, so the extra index range contributes nothing and "
    "the shared-support argument above is unchanged. "
    "NOT caught either: the reference-plane de-embedding — both routes are magnitude-only, "
    "|S| = sqrt(P_num/P_inc) is built from flux alone and ref_shifts reaches only "
    "jnp.angle(ratio), so moving the left reference plane five coarse cells (12.7 mm) swings "
    "angle(S11) by 277.3 deg while closure_S moves 1.09e-07 and |S| moves 7.0e-08, against a "
    "0.02 gate and a committed max|closure_S| = 9.033264e-05. CAUGHT: any failure of power "
    "transport in the guide between the two plane pairs; a wrong plane index only when it "
    "mis-snaps into the slab or the absorber, since the closure residual is plane-invariant "
    "in a lossless source-free region"
)


def measure_closure_witness(dx: float = RUNG_DX_M) -> dict:
    """Run route A and route B once and return the full record.

    ONE measurement (R2, rfx-tightened): the three solves below are the single
    pre-declared attempt. Nothing here re-runs to obtain a different number.
    """
    t0 = time.time()
    lane = _run_flux_lane(dx)
    device = _run_monitor_solve(DUT, dx)
    reference = _run_monitor_solve(REFERENCE_DUT, dx)

    S = G.s_from_json(lane["s_params"])
    s11, s21 = S[0, 0], S[1, 0]
    closure_s = 1.0 - np.abs(s11) ** 2 - np.abs(s21) ** 2

    f_in = np.asarray(device["flux"][MON_IN_NAME], dtype=float)
    f_out = np.asarray(device["flux"][MON_OUT_NAME], dtype=float)
    f_ref = np.asarray(reference["flux"][MON_IN_NAME], dtype=float)
    closure_m = interior_closure(f_in, f_out, f_ref)

    diff = np.abs(closure_s - closure_m)
    worst = int(np.argmax(diff))
    freqs = np.asarray(F.FREQS, dtype=float)
    centre = int(F.BAND_CENTRE_BIN)

    verdict = ("pass" if route_disagreement(closure_s, closure_m) <= CLOSURE_ABS_DIFF_GATE
               else "fail")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "plan": "docs/design_notes/v18_waveguide_s_chain_plan.md",
        "contract": "docs/design_notes/chain_closure_contract.md",
        "predeclaration": "docs/design_notes/waveguide_chain_battery_predeclaration.md",
        "builder": "tests/_waveguide_chain_battery_fixture.py",
        "test": "tests/oracle/test_waveguide_chain_battery_closure.py",
        "readme": "tests/fixtures/waveguide_chain_battery/README.md",
        "declaration": {
            "gate_abs_diff": CLOSURE_ABS_DIFF_GATE,
            "gate_source": ("plan WP3 Falsifier: interior-monitor closure and port "
                            "closure agree within 0.02, the column-power tolerance"),
            "route_a": ("1 - |S11|^2 - |S21|^2 from "
                        "compute_waveguide_s_matrix(normalize='flux') on the two-port "
                        "fixture, column 0 (left drive)"),
            "route_b": ("[F_dev(guide_in) - F_dev(guide_out)] / F_ref(guide_in) from "
                        "full-plane add_flux_monitor planes inside the guide, device "
                        "run on the slab and reference run on the thru"),
            "monitor_x_m": {MON_IN_NAME: MON_IN_X_M, MON_OUT_NAME: MON_OUT_X_M},
            "monitor_coarse_cells": {MON_IN_NAME: K_MON_IN, MON_OUT_NAME: K_MON_OUT},
            "independence_scope": INDEPENDENCE_SCOPE,
            "settling_db_max": SETTLING_DB_MAX,
            "non_vacuity_min_s11": NON_VACUITY_MIN_S11,
        },
        "rung": RUNG,
        "dx_m": float(dx),
        "dut": DUT,
        "reference_dut": REFERENCE_DUT,
        "num_periods": float(F.NUM_PERIODS),
        "band_centre_bin": centre,
        "freqs_hz": [float(x) for x in freqs],
        "flux_lane": lane,
        "device_run": device,
        "reference_run": reference,
        "s11_mag2_per_bin": [float(x) for x in np.abs(s11) ** 2],
        "s21_mag2_per_bin": [float(x) for x in np.abs(s21) ** 2],
        "closure_s_per_bin": [float(x) for x in closure_s],
        "closure_m_per_bin": [float(x) for x in closure_m],
        "abs_diff_per_bin": [float(x) for x in diff],
        "max_abs_diff": float(diff.max()),
        "worst_bin_index": worst,
        "worst_bin_hz": float(freqs[worst]),
        "closure_s_at_worst": float(closure_s[worst]),
        "closure_m_at_worst": float(closure_m[worst]),
        "closure_s_at_centre": float(closure_s[centre]),
        "closure_m_at_centre": float(closure_m[centre]),
        "abs_diff_at_centre": float(diff[centre]),
        "non_vacuity_max_s11": float(np.abs(s11).max()),
        "verdict": verdict,
        "wall_time_s": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# REPLAY layer (fast, no FDTD)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def witness() -> dict:
    if not WITNESS.exists():  # pragma: no cover - the artifact is committed
        pytest.fail(f"missing closure witness artifact {WITNESS.relative_to(REPO)}; "
                    "regenerate with scripts/diagnostics/"
                    "waveguide_chain_battery_closure_measure.py")
    return json.loads(WITNESS.read_text())


def test_witness_declaration_matches_this_module(witness):
    """The artifact was measured under the constants this file declares."""
    d = witness["declaration"]
    assert witness["schema"] == SCHEMA
    assert witness["schema_version"] == SCHEMA_VERSION
    assert d["gate_abs_diff"] == CLOSURE_ABS_DIFF_GATE
    assert d["monitor_x_m"][MON_IN_NAME] == pytest.approx(MON_IN_X_M, abs=1e-12)
    assert d["monitor_x_m"][MON_OUT_NAME] == pytest.approx(MON_OUT_X_M, abs=1e-12)
    assert d["monitor_coarse_cells"] == {MON_IN_NAME: K_MON_IN, MON_OUT_NAME: K_MON_OUT}
    # The scope sentence is a claim about what this witness can and cannot catch,
    # so it is gated like a number: the frozen artifact and this module must agree.
    assert d["independence_scope"] == INDEPENDENCE_SCOPE
    assert witness["dut"] == DUT and witness["reference_dut"] == REFERENCE_DUT
    assert witness["dx_m"] == pytest.approx(RUNG_DX_M, abs=1e-15)
    assert witness["num_periods"] == pytest.approx(F.NUM_PERIODS)
    assert witness["freqs_hz"] == pytest.approx(list(np.asarray(F.FREQS, dtype=float)))


def test_monitor_planes_sit_between_the_port_probes_and_the_slab():
    """Geometry guard, no FDTD: each monitor is strictly inside the guide,
    between its port's probe plane and the nearer slab face. A monitor on the
    wrong side of either would measure a different power balance and the
    comparison would be meaningless rather than wrong."""
    probe_left, probe_right = F.PROBE_LEFT_M, F.PROBE_RIGHT_M
    slab_lo, slab_hi = F.SLAB_X_M
    assert probe_left < MON_IN_X_M < slab_lo, (probe_left, MON_IN_X_M, slab_lo)
    assert slab_hi < MON_OUT_X_M < probe_right, (slab_hi, MON_OUT_X_M, probe_right)
    # Equal clearance either side of the slab, so an evanescent residue at one
    # face cannot bias one monitor more than the other.
    assert (slab_lo - MON_IN_X_M) == pytest.approx(MON_OUT_X_M - slab_hi, abs=1e-12)
    # And both monitors are downstream of the left source plane.
    assert F.PORT_LEFT_X_M < MON_IN_X_M


def test_monitor_simulation_matches_the_battery_fixture():
    """Dropping the right port leaves the left drive bit-identical.

    Realized grid, time step, absorber and the left port's compiled config
    (including the pre-filtered injection tables) must equal the two-port
    fixture's. This is the guard behind ``build_monitor_simulation``'s claim
    that route B solves the same field problem as the flux lane's left drive.
    """
    two_port = F.build_simulation(DUT, RUNG_DX_M)
    one_port = build_monitor_simulation(DUT, RUNG_DX_M)
    g2, g1 = two_port._build_grid(), one_port._build_grid()
    assert g1.shape == g2.shape
    assert float(g1.dt) == float(g2.dt)
    assert one_port._cpml_layers == two_port._cpml_layers
    assert len(one_port._waveguide_ports) == 1
    n_steps = int(g2.num_timesteps(F.NUM_PERIODS))
    freqs = jnp.asarray(F.FREQS)
    cfg2 = two_port._build_waveguide_port_config(
        two_port._waveguide_ports[0], g2, freqs, n_steps)
    cfg1 = one_port._build_waveguide_port_config(
        one_port._waveguide_ports[0], g1, freqs, n_steps)
    for field in ("x_index", "ref_x", "probe_x", "src_amp", "src_t0", "src_tau",
                  "src_fcen", "f_cutoff", "waveform", "direction"):
        assert getattr(cfg1, field) == getattr(cfg2, field), field
    assert np.array_equal(np.asarray(cfg1.e_inc_table), np.asarray(cfg2.e_inc_table))
    assert np.array_equal(np.asarray(cfg1.h_inc_table), np.asarray(cfg2.h_inc_table))


def test_recorded_solves_are_non_degenerate(witness):
    """Branch (iii) of the pre-declaration: the numbers only carry a verdict
    if every solve was healthy. A monitor reading exactly zero, a non-positive
    reference flux or a settling witness above −40 dB is reported here rather
    than absorbed into the closure number."""
    for key in ("device_run", "reference_run"):
        run = witness[key]
        assert run["settling_db"] <= SETTLING_DB_MAX, (key, run["settling_db"])
        for name in (MON_IN_NAME, MON_OUT_NAME):
            arr = np.asarray(run["flux"][name], dtype=float)
            assert np.all(np.isfinite(arr)), (key, name)
            assert not np.any(arr == 0.0), (key, name, "degenerate flux record")
    ref_in = np.asarray(witness["reference_run"]["flux"][MON_IN_NAME], dtype=float)
    assert np.all(ref_in > 0.0), ref_in
    dev = witness["device_run"]["flux"]
    assert np.all(np.asarray(dev[MON_IN_NAME], dtype=float) > 0.0)
    assert np.all(np.asarray(dev[MON_OUT_NAME], dtype=float) > 0.0)
    for port, db in witness["flux_lane"]["settling_db"].items():
        assert db <= SETTLING_DB_MAX, (port, db)
    assert witness["non_vacuity_max_s11"] > NON_VACUITY_MIN_S11


def test_recorded_preflight_matches_the_battery_record(witness):
    """The three solves ran on the geometry WP2 measured: same preflight
    codes as ``fixture.json``'s ``("slab", "coarse")`` and ``("thru",
    "coarse")`` entries. Preflight output is part of the result — a changed
    finding set means a changed fixture, not a changed number."""
    slab_codes = sorted(i["code"] for i in witness["flux_lane"]["preflight"])
    assert slab_codes == ["lossless_q", "mesh_resolution", "mesh_resolution",
                          "mesh_resolution"]
    assert sorted(i["code"] for i in witness["device_run"]["preflight"]) == slab_codes
    assert [i["code"] for i in witness["reference_run"]["preflight"]] == []
    joined = " ".join(i["message"] for i in witness["flux_lane"]["preflight"])
    assert "5.1 cells per" in joined and "lossless" in joined


def test_closure_routes_agree_within_the_column_power_tolerance(witness):
    """THE GATE. ``max_f |closure_S − closure_M| ≤ 0.02``.

    Never widened. If this goes red the tolerance stays and the measurement is
    re-read: the PR body must name which route is suspect and why.
    """
    s = np.asarray(witness["closure_s_per_bin"], dtype=float)
    m = np.asarray(witness["closure_m_per_bin"], dtype=float)
    measured = route_disagreement(s, m)
    assert measured == pytest.approx(witness["max_abs_diff"], rel=1e-9, abs=1e-12)
    worst = int(np.argmax(np.abs(s - m)))
    assert measured <= CLOSURE_ABS_DIFF_GATE, (
        f"interior-monitor closure and port closure disagree by "
        f"{measured:.6g} > {CLOSURE_ABS_DIFF_GATE} at "
        f"{witness['freqs_hz'][worst] / 1e9:.2f} GHz "
        f"(port route {s[worst]:.6g}, interior route {m[worst]:.6g})")
    assert witness["verdict"] == "pass"


def test_recorded_closure_columns_are_recomputed_from_the_stored_intermediates(witness):
    """R5: the headline is re-derived from the dumped per-bin intermediates,
    so a hand-edited headline cannot survive. Both routes are rebuilt from the
    raw fluxes and the raw S entries, not read off the summary."""
    s11m2 = np.asarray(witness["s11_mag2_per_bin"], dtype=float)
    s21m2 = np.asarray(witness["s21_mag2_per_bin"], dtype=float)
    S = G.s_from_json(witness["flux_lane"]["s_params"])
    s11, s21 = S[0, 0], S[1, 0]
    assert s11m2 == pytest.approx(np.abs(s11) ** 2, rel=1e-9, abs=1e-12)
    assert s21m2 == pytest.approx(np.abs(s21) ** 2, rel=1e-9, abs=1e-12)
    assert np.asarray(witness["closure_s_per_bin"]) == pytest.approx(
        1.0 - s11m2 - s21m2, rel=1e-9, abs=1e-12)

    f_in = np.asarray(witness["device_run"]["flux"][MON_IN_NAME], dtype=float)
    f_out = np.asarray(witness["device_run"]["flux"][MON_OUT_NAME], dtype=float)
    f_ref = np.asarray(witness["reference_run"]["flux"][MON_IN_NAME], dtype=float)
    assert np.asarray(witness["closure_m_per_bin"]) == pytest.approx(
        interior_closure(f_in, f_out, f_ref), rel=1e-9, abs=1e-12)


def test_interior_planes_reproduce_the_port_magnitudes(witness):
    """Stronger than the closure identity, from the same measurement: the
    interior planes reproduce ``|S11|²`` and ``|S21|²`` themselves, not only
    their sum.

    ``1 − F_dev(in)/F_ref(in)`` is the reflected power fraction and
    ``F_dev(out)/F_ref(in)`` the transmitted one — the flux lane's own two
    ratios, read at planes the extractor never samples. A compensating pair of
    errors could leave the closure sum intact while both magnitudes were
    wrong; this separates them. Same declared 0.02, no new tolerance.
    """
    f_in = np.asarray(witness["device_run"]["flux"][MON_IN_NAME], dtype=float)
    f_out = np.asarray(witness["device_run"]["flux"][MON_OUT_NAME], dtype=float)
    f_ref = np.asarray(witness["reference_run"]["flux"][MON_IN_NAME], dtype=float)
    s11m2 = np.asarray(witness["s11_mag2_per_bin"], dtype=float)
    s21m2 = np.asarray(witness["s21_mag2_per_bin"], dtype=float)
    assert np.max(np.abs((1.0 - f_in / f_ref) - s11m2)) <= CLOSURE_ABS_DIFF_GATE
    assert np.max(np.abs((f_out / f_ref) - s21m2)) <= CLOSURE_ABS_DIFF_GATE


def test_the_empty_guide_transports_power_between_the_two_monitor_planes(witness):
    """Placement guard on the reference run: with no slab, the two monitor
    planes must read the same net power. A monitor snapped into the absorber,
    onto the source plane or outside the guide would break this before it
    broke the closure comparison. Same declared 0.02."""
    ref = witness["reference_run"]["flux"]
    ratio = (np.asarray(ref[MON_OUT_NAME], dtype=float)
             / np.asarray(ref[MON_IN_NAME], dtype=float))
    assert np.max(np.abs(ratio - 1.0)) <= CLOSURE_ABS_DIFF_GATE, ratio


def test_float64_resummation_of_the_same_accumulators_keeps_the_verdict(witness):
    """The agreement is not a float32 summation artefact.

    ``flux_spectrum(exact_f64=True)`` re-sums the SAME complex64 accumulators
    in float64 — the sanctioned remedy for the subnormal-flush failure mode
    where a physically tiny flux returns exactly 0.0 (issue #304). The stored
    interior fluxes are ~1e-24 W, far above the float32 minimum normal, and
    the re-summed closure must still pass the declared gate.
    """
    dev, ref = witness["device_run"], witness["reference_run"]
    f_in = np.asarray(dev["flux_exact_f64"][MON_IN_NAME], dtype=float)
    f_out = np.asarray(dev["flux_exact_f64"][MON_OUT_NAME], dtype=float)
    f_ref = np.asarray(ref["flux_exact_f64"][MON_IN_NAME], dtype=float)
    closure_m64 = interior_closure(f_in, f_out, f_ref)
    s = np.asarray(witness["closure_s_per_bin"], dtype=float)
    assert route_disagreement(s, closure_m64) <= CLOSURE_ABS_DIFF_GATE
    # And the two summations agree with each other far inside the gate, which
    # is what says the 2.1e-05 route difference is physics-level float32
    # noise in the fields rather than host-side cancellation.
    closure_m32 = np.asarray(witness["closure_m_per_bin"], dtype=float)
    assert route_disagreement(closure_m32, closure_m64) <= CLOSURE_ABS_DIFF_GATE


def test_a_perturbed_interior_flux_makes_the_gate_red(witness):
    """Cheap refute, run on a copy of the artifact: scale the device run's
    outgoing interior flux by 1.05 and the gate must fail. A gate that cannot
    be made red by a 5 % power-balance error is not measuring the balance."""
    f_in = np.asarray(witness["device_run"]["flux"][MON_IN_NAME], dtype=float)
    f_out = np.asarray(witness["device_run"]["flux"][MON_OUT_NAME], dtype=float) * 1.05
    f_ref = np.asarray(witness["reference_run"]["flux"][MON_IN_NAME], dtype=float)
    perturbed = interior_closure(f_in, f_out, f_ref)
    s = np.asarray(witness["closure_s_per_bin"], dtype=float)
    assert route_disagreement(s, perturbed) > CLOSURE_ABS_DIFF_GATE


# ---------------------------------------------------------------------------
# LIVE layer (slow: three FDTD solves at the coarse rung)
# ---------------------------------------------------------------------------

def test_live_closure_routes_agree_within_the_column_power_tolerance(witness):
    """Re-measure both routes and re-assert the gate against physics.

    Lane placement (contract criterion 3, "fast lane when ≤ 30 s"): the three
    coarse-rung solves — the flux lane's 2 × n_ports runs plus the two
    single-port monitor runs, 713 steps on an 83 × 10 × 5 grid — measured
    13.5 s wall on this box's CPU (jax 0.6.2), so this stays in the fast lane
    and carries no ``slow`` marker. XLA compilation dominates that time, not
    the stepping. If a slower runner pushes it past 30 s, move THIS test to
    the slow lane; the replay layer above keeps the gate in the fast lane at
    zero FDTD cost.
    """
    live = measure_closure_witness()
    s = np.asarray(live["closure_s_per_bin"], dtype=float)
    m = np.asarray(live["closure_m_per_bin"], dtype=float)
    measured = route_disagreement(s, m)
    worst = int(np.argmax(np.abs(s - m)))
    assert measured <= CLOSURE_ABS_DIFF_GATE, (
        f"live interior-vs-port closure disagreement {measured:.6g} > "
        f"{CLOSURE_ABS_DIFF_GATE} at {live['freqs_hz'][worst] / 1e9:.2f} GHz")
    # And the committed artifact still describes this box's physics. This is a
    # coherence check, not a reproduction gate: it must fire when the witness
    # stops describing the run at all, and must not fire on run-to-run float32
    # noise. The bound is the gate itself divided by 20 (0.02 / 20 = 1e-3): a
    # drift that large is a twentieth of the disagreement the gate allows, so it
    # cannot hide a real change, while sitting far above the 5e-6 cross-backend
    # |S| envelope the battery measured. Tightening it to that envelope would
    # make this assert a second, undeclared reproduction gate.
    assert live["max_abs_diff"] == pytest.approx(
        witness["max_abs_diff"], abs=1e-3), (live["max_abs_diff"],
                                             witness["max_abs_diff"])
