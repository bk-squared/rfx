"""Lumped/wire-port V-I extraction + Z0-normalization validation battery.

Validation-campaign lane: locks the POST-PROCESSING layer of the lumped/wire
port family — the V/I DFT accumulation inside the production scan
(``rfx/simulation.py``), the shared pure wave decomposers
(``rfx/probes/probes.py::decompose_lumped_s_matrix`` /
``decompose_wire_s_matrix`` / ``extract_lumped_s11``), the production scan
driver (``rfx/probes/sparam_driver.py``), and the Z0 normalization — via the
NULL-INPUT-ISOLATION method: a 2-port THRU (two matched ports joined by a
uniform line) has a known trivial answer, so every deviation isolates an
extraction/calibration property independent of any DUT physics. The core
FDTD update is already validated elsewhere; this battery does not re-test it.

Honest posture: changes/relaxes NO gate outside this module. The
``tests/test_twoport_wire_port.py`` floor gate (max|S21| > 1e-3), the
``tests/test_sparam_driver_matches_eager.py`` driver-vs-eager atol-2e-3
locks, and the ``tests/test_run_forward_s11_contract.py`` magnitude-only
CPML atol-2e-3 contract are all untouched. This module's own thru gates
were re-baselined 2026-07-10 in the same PR as the issue #308
receive-wave fix — exactly the tripwire protocol the original gates
prescribed (fail LOUDLY on the convention change, re-measure in the same
PR). The S11 floor/alive gates, the run<->forward cross-check, and the
algebraic-identity test are byte-untouched by the re-baseline.

Measured baseline (R5 measure-before-gate; re-baselined 2026-07-10 for
the issue #308 receive-wave fix and re-measured the same day for the
falsifier-driven receive-sign amendment b_recv = (V - Z0*I)/(2*sqrt(Z0)),
fresh fixture reruns on this CPU box, x64 OFF — complex64 accumulators,
the documented envelope):

THRU fixture (wire 2-port, 16 mm air microstrip w/h=5 over a pec_faces
ground, Zc ~ 50 ohm, driver path of PR #258; 9 bins 3-7 GHz, 4000 steps,
~70 s -> slow_physics):
  - max in-band |S11| = 0.12962 (mean 0.0759), max |S22| = 0.13156 —
    bit-identical to the pre-#308 baseline (the fix touches receive-side
    off-diagonals only; diagonals are byte-frozen). The reflection
    channel is ALIVE, not a dead readout: on the same fixture, measured
    Z_in = 44.0-1.9j ohm at 5 GHz predicts |S11| = 0.06674 via
    (Zin-50)/(Zin+50) vs 0.06676 reported (2e-5 agreement), and
    mismatching port 2 to 12.5 ohm moved |S11| from the 0.035-0.13
    floor to 0.20-0.42 in the expected direction (Z0 witness, reported
    in the lane, not re-run here).
  - |S21| = 0.523601..0.667776 across 3-7 GHz. HISTORY: before the
    issue #308 fix (landed in this PR) the shipped receive-side b-wave
    b_i = (-V - Z0_i*I)/(2*sqrt(Z0_i)) structurally cancelled the
    arriving wave against the matched receive cell's local Ohm's law
    -V = +Z0_cell*I (measured -V/I = 16.6672+0.0002j ohm vs
    Z0/n_cells = 16.667 ohm), so a matched thru read |S21| =
    0.0025-0.0046 near-null, family-wide (lumped + wire). #308
    role-selects the receive b-wave to the orthogonal channel
    (V - Z0*I) (sign pinned by the DC falsifier, phase item below; the
    first-cut sign (-V + Z0*I) was amended in the same PR — |S21| is
    sign-invariant, these magnitudes did not move), recovering the
    transmission channel. HONESTY LABEL — REGRESSION LOCK, NOT
    validated physics: the recovered channel is correct (mismatch
    witness responds per-bin; b-wave voltage-dominated at the old
    residual scale 0.38-0.89%) but its MAGNITUDE is unvalidated. The
    extractor-independent flux referee (flux monitors bracketing the
    line, 2026-07-10 falsifier battery) measured a raw transmitted
    power fraction of 0.959-0.998 across 3-7 GHz vs port-based
    |S21|^2 = 0.274-0.446 — per-bin ratio (flux fraction)/|S21|^2 =
    2.237..3.541 (mean 2.911), i.e. implied flux-true |S21| =
    0.971-0.997 (lossless closure gap 0.002-0.040), consistent with the
    physics-true expectation |S21| >= ~0.9 for this near-lossless thru.
    The port-based magnitude deficit is a confirmed drive-side
    common-mode scale bias kappa(f) = 1.49..1.86 (frequency-DEPENDENT)
    entering a_j via the source-cell V/I accounting; it is invisible to
    S11. The flux-vs-port transmitted-power delta is an OPEN item
    (recorded per-bin above; see the PR body / follow-up issue) — do
    NOT cite the 0.52-0.67 band as thru-transmission physics.
  - S21 signed phase deviation vs the analytic line delay
    exp(-j*2*pi*f*L/c): -0.754891..-0.335259 rad across 3-7 GHz — a
    smooth delay-like excess (~16.6 ps feed-post group delay on top of
    the 53.4 ps line delay), no pi offset. The DC-limit sign is
    RESOLVED by the same-PR amendment (2026-07-10): the first-cut #308
    sign (-V + Z0*I) measured S21(DC) -> -1 on the low-frequency
    falsifier (0.5-2 GHz; the pi sat in the raw cross-port phasors,
    arg(V2/V1) ~= pi - beta*L, from the source-driven cell field sense
    in V = -E*dx, while both port diagonals were internally
    sign-consistent); the amended receive wave b_i =
    (V - Z0*I)/(2*sqrt(Z0)) — a global -1 of the first cut — measures
    S21(DC) -> +1: at 0.5 GHz S21 = +0.57116-0.11059j (arg -0.191 rad,
    heading to 0 as f -> 0; re-verified on this exact geometry in the
    amendment rerun). |S21|, reciprocity, the flux comparisons and both
    diagonals are sign-invariant and did not move (bit-identical to the
    81c1983 rerun).
  - reciprocity max|S21-S12| = 1.043e-2 absolute (rel 0.016254) on the
    recovered O(0.52-0.67) magnitudes — now a meaningful symmetry check
    on a live channel (pre-#308 it was vacuous at the near-null scale).
  - passivity max singular value over the 9 per-freq 2x2 slices =
    0.687410 (margin 0.31 to the physical bound 1.0). Still NOT
    transmission validation (kappa open item deflates the port-based
    magnitudes); kept as a strict sub-unity energy-sanity lock.

RUN<->FORWARD cross-check fixture (1-port wire, CPML — byte-for-byte the
``_wire_sim`` of tests/test_run_forward_s11_contract.py, which this
COMPLEMENTS: that committed test gates |S11| magnitude at atol 2e-3; this
battery adds (a) both return SHAPES pinned as an intended contract and
(b) a COMPLEX-value delta gate at the measured float32 envelope):
  - run(compute_s_params=True) returns shape (1, 1, 7) complex64 — the
    full-matrix rank, S[receive, drive, freq], even for 1 port;
    forward(port_s11_freqs=) returns shape (7,) complex64 — a 1-D
    diagonal via extract_lumped_s11. The rank difference is INTENDED
    behavior, pinned here as a contract: PR #258 deliberately kept the
    1-port wire run() path on the JIT main-scan fast-path (preserving
    the run/forward PEC contract) while multi-port sets go through the
    production scan driver; run() reports the S-matrix convention,
    forward() the AD-friendly per-port diagonal. Do not "fix" one to
    match the other.
  - measured max complex |S_run - S_fwd| = 4.27e-7 all-band (2.15e-7
    in-band) vs the committed 2e-3 magnitude gate — the committed band is
    honest with ~4 orders of margin. Small deltas here are the measured
    float32 conditioning envelope (PR #258 finding: the decompose
    diagonal is algebraically identical to extract_lumped_s11) — gate
    them, don't chase them.

Direction sensitivity of the S21 phase gate (claim VERIFIED, in-test):
the gate is on the SIGNED, per-bin wrapped deviation arg(S21) -
(-2*pi*f*L/c), so it constrains sign AND magnitude. It is NOT
conjugation-invariant: conjugating S21 (the 5/5-recurring comparator bug
class, W3.4) flips arg(S21), and on the measured amended-sign data moves
8 of 9 bins outside the band (measured conj devs +2.347..-0.834 rad vs
band [-1.1, -0.1]; only the 7 GHz bin remains inside). The phase test
asserts this discrimination live on the measured data (conj(S21) must
violate the band), so the gate cannot silently degrade into a
conjugation-blind |dev| check. A flip back to the first-cut receive
sign shifts every bin by pi (to ~+2.39..+2.81 rad) and also fails the
band — the DC-witnessed sign is locked, not merely a convention
envelope.

Preflight (quoted verbatim per feedback_never_ignore_preflight; the thru
fixture asserts and prints this at fixture setup):
  "pec_faces={z_lo} creates an INFINITE PEC boundary AND the geometry
  contains finite PEC objects. For antennas or finite-GP structures, the
  pec_faces boundary makes the ground plane cover the entire domain face,
  which changes the physics (cavity vs radiating antenna). If you need a
  finite ground plane, remove pec_faces and use an explicit PEC Box
  instead."
(advisory-only; intended here — the infinite ground plane IS the
microstrip return). The 1-port CPML fixture measures preflight clean
("All checks passed"), asserted in the fast suite. The environment-level
JAX float64->float32 truncation UserWarning is the documented
x64-off accumulator envelope; x64 is deliberately never flipped at module
level in this file.

Fixture-authoring lesson baked into the thru geometry (2026-07-10 lane):
Box PEC rasterization is lower-inclusive/upper-EXCLUSIVE, so the trace
box must overhang BOTH port columns by >= 1 cell — a box ending exactly
at the port-2 x coordinate leaves that column with no PEC overhead and
produces a silently dead thru. Also: wire extent=1.0 mm at dx=0.5 mm
rasterizes to n_cells=3 (endpoint-inclusive), which enters the
Z0_cell = Z0/n_cells off-diagonal normalization.

No network, no external solver; deterministic (fixed geometry, fixed step
counts, rect DFT window). Tighten gates only with a fresh measured
baseline; a failure here marks an extraction/normalization regression (or
an intentional upstream decomposer-convention change, which must update
the measured provenance here in the same PR).
"""

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.probes import (
    decompose_lumped_s_matrix,
    decompose_wire_s_matrix,
    extract_lumped_s11,
)
from rfx.sources.sources import GaussianPulse

C0_M_PER_S = 299792458.0

# ===========================================================================
# THRU fixture geometry (all lengths integer multiples of dx = 0.5 mm)
# ===========================================================================
_THRU_DX_M = 0.5e-3
_THRU_DOMAIN_M = (0.032, 0.020, 0.010)
_THRU_FREQ_MAX_HZ = 10e9
_THRU_CPML_LAYERS = 8
_THRU_H_M = 1.0e-3        # trace height above ground (2 cells)
_THRU_W_M = 5.0e-3        # trace width (10 cells) -> air microstrip Zc ~ 50 ohm
_THRU_X1_M = 0.008        # port 1 x
_THRU_X2_M = 0.024        # port 2 x
_THRU_L_M = _THRU_X2_M - _THRU_X1_M   # 16 mm port-to-port
_THRU_Y_MID_M = _THRU_DOMAIN_M[1] / 2
_THRU_N_STEPS = 4000
_THRU_FREQS_HZ = np.linspace(3e9, 7e9, 9)   # in-band of f0=5 GHz, bw=0.8

_PEC_FACES_ADVISORY_SNIPPET = (
    "pec_faces={z_lo} creates an INFINITE PEC boundary AND the geometry "
    "contains finite PEC objects."
)

# ===========================================================================
# Gate constants (R5: every gate = measured value + honest margin)
# ===========================================================================
# Measured max in-band |S11| = 0.12962, |S22| = 0.13156 (mean |S11| 0.0759).
# Gate 0.30 (~2.3x measured max) on BOTH diagonals. The floor is a real
# physical-match reading (Zin witness agrees to 2e-5; Z0-mismatch witness
# moves it as predicted), so a break here means the reflection channel or
# the Z0 normalization moved.
_THRU_S11_FLOOR_MAX = 0.30
# Two-sided (review finding): a DEAD diagonal channel reads ~0, which would
# sail under the upper bound. The measured per-diagonal maxima are
# 0.1296/0.1316 (~6.5x above this lower bound), so requiring max|Sii| > 0.02
# makes the in-test liveness of both diagonals explicit rather than relying
# on the lane's out-of-test Zin/mismatch witnesses.
_THRU_S11_ALIVE_MIN = 0.02

# Measured |S21| = 0.523601..0.667776 across 3-7 GHz (post-#308 fix,
# fresh rerun at 81c1983). REGRESSION LOCK on the recovered channel, NOT
# physics: the flux-vs-port transmitted-power delta is OPEN (flux referee
# measured per-bin (flux fraction)/|S21|^2 = 2.237..3.541; implied
# flux-true |S21| = 0.971-0.997 — module docstring; see the PR body /
# follow-up issue). Band [0.35, 0.85]: lower ~0.67x measured min (a
# collapse back toward the pre-#308 near-null 0.0025-0.0046 fails
# loudly), upper ~1.27x measured max and strictly < 1 (an over-unity
# extraction artefact also fails). Still never weaker than the committed
# max|S21| > 1e-3 floor of test_twoport_wire_port.py.
_THRU_S21_BAND = (0.35, 0.85)

# Measured signed per-bin phase deviation arg(S21) - (-2*pi*f*L/c),
# wrapped: -0.754891..-0.335259 rad (amended receive sign, 2026-07-10
# rerun). Band [-1.1, -0.1] rad (margins ~0.35/0.24 rad; both edges
# well inside (-pi, pi) so wrapped values stay comparable). HONESTY
# LABEL — REGRESSION LOCK on the deviation VALUES (flux-vs-port
# magnitude delta OPEN, kappa — module docstring), but the overall SIGN
# is physics-anchored by the DC witness: the low-f falsifier measured
# S21(DC) -> +1 under this sign (dev -0.024 rad at 0.5 GHz, tracking
# the analytic delay); the first-cut sign measured -1 and was amended
# in the same PR. The deviation is a smooth ~16.6 ps feed-post
# group-delay excess, physical, not a convention artefact. Signed on
# purpose: conjugation moves 8/9 measured bins out of band (verified
# live in the test; conj dev is NOT -dev, the analytic reference phase
# differs per bin — measured conj devs +2.347..-0.834 with only the
# 7 GHz bin inside), and a sign flip back moves all 9 bins (by pi).
_THRU_PHASE_DEV_BAND_RAD = (-1.1, -0.1)

# Measured reciprocity max|S21 - S12| = 1.043e-2 (rel 0.016254) on the
# recovered O(0.52-0.67) magnitudes. Gates ~4.8x / ~6x measured. The abs
# gate is scaled to the live channel (pre-#308 it was 5e-4 on the
# near-null scale); the rel gate survives unchanged.
_THRU_RECIP_ABS_MAX = 5.0e-2
_THRU_RECIP_REL_MAX = 0.10

# Measured passivity: max singular value over the 9 per-freq 2x2 slices
# = 0.687410 (post-#308). Gate 0.90 (~1.31x measured, margin 0.21) —
# strictly below the physical bound 1.0, and BELOW the Frobenius
# dominance bound sqrt(2*0.30^2 + 2*0.85^2) ~= 1.275 implied by the
# S11/S21 gates, so this gate is independently bindable. Energy-sanity
# lock; NOT transmission validation (kappa open item, module docstring).
_THRU_MAX_SINGULAR_VALUE = 0.90

# ===========================================================================
# run<->forward cross-check constants
# ===========================================================================
_XCHK_F0_HZ = 5e9
_XCHK_FREQS_HZ = np.array([1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]) * 1e9
# Measured max complex |S_run - S_fwd| = 4.27e-7 over ALL 7 bins (CPML,
# well-conditioned) — but that is SINGLE-MACHINE provenance at float32
# ulp scale. Gate 5e-5 (~100x measured; review finding, the v173a
# cross-machine-float lesson) — still 40x tighter than the committed
# magnitude-only atol-2e-3 gate it complements, and still fails on any
# pure-phase divergence that gate cannot see.
_XCHK_COMPLEX_DELTA_MAX = 5.0e-5

# Algebraic-identity lock (no FDTD): extract_lumped_s11 vs the decompose
# diagonals on synthetic well-conditioned V/I. The formulas are
# algebraically identical (PR #258 finding: divergence is float32
# conditioning, not formula); in complex64 they may differ only by
# rounding order. Gate 1e-5 relative (float32 eps ~1.2e-7, x ~100 margin).
_IDENTITY_REL_MAX = 1.0e-5


# ===========================================================================
# Fixtures
# ===========================================================================
def _build_thru() -> Simulation:
    """Wire 2-port air-microstrip THRU (2026-07-10 lane M1 fixture, exact).

    Both ports carry excite=True + the same waveform: the production scan
    driver drives each eligible port BY INDEX regardless of ``pe.excite``
    (one at a time; others are matched loads), and an excite=False port
    stores waveform=None, which cannot be driven. 2-port wire set ->
    run(compute_s_params=True) routes through the PR #258 production scan
    driver, the extraction path under test.
    """
    sim = Simulation(
        freq_max=_THRU_FREQ_MAX_HZ,
        domain=_THRU_DOMAIN_M,
        dx=_THRU_DX_M,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=_THRU_CPML_LAYERS,
    )
    # PEC trace one cell thick on top of the wire-port spans. The x-extent
    # overhangs each port column by one cell — Box rasterization is
    # lower-inclusive/upper-EXCLUSIVE, and a box ending exactly at the
    # port-2 x leaves that column without PEC overhead (silently dead thru;
    # module docstring, fixture-authoring lesson).
    sim.add(
        Box((_THRU_X1_M - _THRU_DX_M, _THRU_Y_MID_M - _THRU_W_M / 2, _THRU_H_M),
            (_THRU_X2_M + _THRU_DX_M, _THRU_Y_MID_M + _THRU_W_M / 2,
             _THRU_H_M + _THRU_DX_M)),
        material="pec",
    )
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    sim.add_port(position=(_THRU_X1_M, _THRU_Y_MID_M, 0.0), component="ez",
                 impedance=50.0, extent=_THRU_H_M, waveform=pulse,
                 direction="-x")
    sim.add_port(position=(_THRU_X2_M, _THRU_Y_MID_M, 0.0), component="ez",
                 impedance=50.0, extent=_THRU_H_M, waveform=pulse,
                 direction="+x")
    return sim


@pytest.fixture(scope="module")
def thru_smatrix():
    """Run the THRU once (~70 s); quote preflight verbatim; return S(2,2,9)."""
    sim = _build_thru()
    issues = [str(i) for i in sim.preflight()]
    # Quote every preflight message verbatim BEFORE reporting numbers
    # (feedback_never_ignore_preflight).
    for msg in issues:
        print(f"\n[thru battery] preflight (verbatim): {msg}")
    # Exactly the one known advisory (intended: the infinite ground plane
    # IS the microstrip return). Anything else = fixture drift, stop.
    assert len(issues) == 1 and _PEC_FACES_ADVISORY_SNIPPET in issues[0], (
        f"thru fixture preflight drifted from the measured baseline: {issues}")

    result = sim.run(n_steps=_THRU_N_STEPS, compute_s_params=True,
                     s_param_freqs=_THRU_FREQS_HZ)
    S = np.asarray(result.s_params).astype(np.complex128)
    assert S.shape == (2, 2, len(_THRU_FREQS_HZ)), (
        f"driver S-matrix shape {S.shape}, expected (2, 2, 9)")
    assert np.all(np.isfinite(S)), "thru S-matrix contains non-finite entries"
    with np.printoptions(precision=4, suppress=False):
        print(f"[thru battery] |S11|={np.abs(S[0, 0])}")
        print(f"[thru battery] |S21|={np.abs(S[1, 0])}")
    return S


@pytest.fixture(scope="module")
def crosscheck():
    """1-port wire CPML fixture of test_run_forward_s11_contract.py, run
    through BOTH estimators; returns raw (S_run, S_fwd) complex arrays."""

    def _wire_sim():
        sim = Simulation(
            freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
            boundary="cpml", cpml_layers=6,
        )
        sim.add_port(
            position=(0.0093, 0.0093, 0.0093), component="ez", impedance=50.0,
            waveform=GaussianPulse(f0=_XCHK_F0_HZ, bandwidth=0.9),
            extent=0.004,
        )
        return sim

    sim_r = _wire_sim()
    issues = sim_r.preflight()
    for msg in issues:
        print(f"\n[crosscheck] preflight (verbatim): {msg}")
    # Measured baseline: this fixture preflights CLEAN. Gate on
    # error-severity only (review finding): a future advisory-only
    # validator flagging this vanilla fixture should be PRINTED verbatim
    # above, not fail the whole cross-check module fixture.
    errors = [str(i) for i in issues
              if getattr(i, "severity", "error") == "error"]
    assert errors == [], (
        f"1-port cross-check fixture has error-severity preflight "
        f"findings: {errors}")

    r = sim_r.run(n_steps=2000, compute_s_params=True,
                  s_param_freqs=_XCHK_FREQS_HZ)
    fr = _wire_sim().forward(port_s11_freqs=_XCHK_FREQS_HZ)
    return np.asarray(r.s_params), np.asarray(fr.s_params)


# ===========================================================================
# FAST battery (default suite): decomposer identity + run<->forward contract
# ===========================================================================

def test_extract_lumped_s11_is_the_decompose_diagonal():
    """Algebraic-identity lock: three shipped S11 formulas agree (no FDTD).

    extract_lumped_s11 (S11 = (V + Z0*I)/(V - Z0*I)), the
    decompose_lumped_s_matrix diagonal (b/a wave form) and the
    decompose_wire_s_matrix diagonal (Zin = -V/I input-impedance form) are
    algebraically the same map; PR #258 proved observed run/forward
    divergence is float32 CONDITIONING, not formula. This pins the
    identity on synthetic well-conditioned phasors so a formula edit in
    any one of the three fails loudly. This locks the FORMULA identity
    only; the two ENTRY-POINT implementations are cross-checked end-to-end
    in test_run_forward_complex_values_agree_on_cpml.
    """
    rng = np.random.default_rng(20260710)
    n_ports, n_freqs, z0 = 2, 11, 50.0
    # Well-conditioned: V, Z0*I both O(Z0), away from the a=0 guard.
    v = (rng.normal(1.0, 0.3, (n_ports, n_ports, n_freqs))
         + 1j * rng.normal(0.0, 0.3, (n_ports, n_ports, n_freqs))) * z0
    i = (rng.normal(3.0, 0.3, (n_ports, n_ports, n_freqs))
         + 1j * rng.normal(0.0, 0.3, (n_ports, n_ports, n_freqs)))

    s_lumped = np.asarray(decompose_lumped_s_matrix(v, i, [z0, z0]))
    s_wire = np.asarray(decompose_wire_s_matrix(v, i, [z0, z0], [3, 3]))
    for p in range(n_ports):
        s_ref = np.asarray(extract_lumped_s11(v[p, p], i[p, p], z0=z0))
        for name, s_diag in (("lumped", s_lumped[p, p]),
                             ("wire", s_wire[p, p])):
            rel = np.max(np.abs(s_diag - s_ref) / np.abs(s_ref))
            assert rel < _IDENTITY_REL_MAX, (
                f"decompose_{name} diagonal (port {p}) deviates from "
                f"extract_lumped_s11 by rel {rel:.2e} "
                f"(gate {_IDENTITY_REL_MAX}) — the three shipped S11 "
                f"formulas are no longer the same map")


def test_run_forward_shapes_are_the_intended_ranks(crosscheck):
    """Rank contract: run() (1, 1, n_freqs) full-matrix vs forward() (n_freqs,).

    The rank difference is INTENDED behavior, pinned here on purpose
    (rfx-known-issues 'Added 2026-06-21' item 3 asked for a contract test;
    this is it, together with the value gate below): run() always reports
    the S-matrix convention S[receive, drive, freq] — (1, 1, n_freqs) even
    for one port — while forward() returns the AD-friendly per-port
    diagonal, (n_freqs,) for one port, via extract_lumped_s11. PR #258
    deliberately kept the 1-port wire run() path on the JIT main-scan
    fast-path to preserve the run/forward PEC contract; do NOT re-unify
    the ranks (or the paths) without a superseding decision.
    """
    S_run, S_fwd = crosscheck
    n_freqs = len(_XCHK_FREQS_HZ)
    assert S_run.shape == (1, 1, n_freqs), (
        f"run() 1-port S-matrix rank changed: {S_run.shape}, contract is "
        f"(1, 1, {n_freqs}) — full-matrix convention")
    assert S_fwd.shape == (n_freqs,), (
        f"forward() 1-port s_params rank changed: {S_fwd.shape}, contract "
        f"is ({n_freqs},) — 1-D per-port diagonal")
    assert np.iscomplexobj(S_run) and np.iscomplexobj(S_fwd), (
        "s_params must stay complex (magnitude-only returns would break "
        "phase-consuming consumers)")


def test_run_forward_complex_values_agree_on_cpml(crosscheck):
    """COMPLEX-value cross-check at the measured float32 envelope.

    Complements tests/test_run_forward_s11_contract.py::
    test_run_forward_s11_agree_on_well_conditioned_cpml (same fixture,
    magnitude-only, atol 2e-3 — untouched): measured max complex delta is
    4.27e-7 over all 7 bins, gated at 5e-5 (~100x, cross-machine float32
    headroom), which also catches a pure-PHASE divergence the magnitude
    gate cannot see. NOT a tautology (review-verified): run() uses the
    inline decomposition on the runners/uniform.py + rfx/simulation.py
    scan path while forward() uses extract_lumped_s11 in
    rfx/api/_execute.py — distinct code sites compiled as different XLA
    graphs — so this gate catches a regression in EITHER entry-point
    implementation (DFT accumulation, port eligibility, freq handling),
    which the pure-formula identity test cannot. Small deltas at this
    scale are the measured float32 conditioning envelope (PR #258: the
    formulas are algebraically identical) — if this fails marginally,
    re-measure the envelope before touching anything; if it fails
    grossly, one of the two extraction paths regressed.
    """
    S_run, S_fwd = crosscheck
    delta = np.abs(S_run.reshape(-1).astype(np.complex128)
                   - S_fwd.reshape(-1).astype(np.complex128))
    print(f"\n[crosscheck] max complex |S_run - S_fwd| = {delta.max():.3e} "
          f"(measured 4.27e-7, gate {_XCHK_COMPLEX_DELTA_MAX:.0e})")
    assert delta.max() < _XCHK_COMPLEX_DELTA_MAX, (
        f"run() vs forward() complex S11 delta {delta.max():.3e} exceeds "
        f"{_XCHK_COMPLEX_DELTA_MAX:.0e} (measured envelope 4.27e-7) on a "
        f"well-conditioned CPML port")


# ===========================================================================
# slow_physics battery (opt-in: -m slow_physics): the 2-port THRU locks
# ===========================================================================

@pytest.mark.slow_physics
def test_thru_s11_floor(thru_smatrix):
    """Matched-thru reflection floor: max in-band |S11|, |S22| < 0.30.

    Measured 0.12962 / 0.13156 (mean |S11| 0.0759). The floor is a REAL
    match reading of the ~50-ohm air-microstrip line against Z0=50 (Zin
    witness agreement 2e-5; Z0-mismatch witness responds as predicted), so
    a break means the diagonal V-I extraction or Z0 normalization moved —
    not the FDTD core.
    """
    s11 = np.abs(thru_smatrix[0, 0])
    s22 = np.abs(thru_smatrix[1, 1])
    worst = max(s11.max(), s22.max())
    assert worst < _THRU_S11_FLOOR_MAX, (
        f"thru diagonal floor broke: max(|S11|, |S22|) = {worst:.4f} "
        f"(measured 0.130/0.132, gate {_THRU_S11_FLOOR_MAX})")
    # Two-sided liveness (review finding): a dead diagonal reads ~0 and
    # would pass the upper bound. Measured maxima 0.1296/0.1316.
    assert s11.max() > _THRU_S11_ALIVE_MIN and s22.max() > _THRU_S11_ALIVE_MIN, (
        f"thru diagonal channel reads dead: max|S11|={s11.max():.4f}, "
        f"max|S22|={s22.max():.4f} (measured 0.130/0.132, alive floor "
        f"{_THRU_S11_ALIVE_MIN})")


@pytest.mark.slow_physics
def test_thru_s21_band_locks_shipped_decomposer_envelope(thru_smatrix):
    """|S21| stays in [0.35, 0.85] — REGRESSION LOCK, NOT validated physics.

    Measured 0.523601..0.667776 post-#308 (the receive-wave fix in this
    PR recovered the transmission channel from the pre-fix structural
    near-null 0.0025-0.0046 — mechanism history in the module docstring).
    HONESTY LABEL: regression lock only; the flux-vs-port transmitted-
    power delta is OPEN — the extractor-independent flux referee measured
    per-bin (flux fraction)/|S21|^2 = 2.237..3.541 (raw flux transmitted
    fraction 0.959-0.998 vs |S21|^2 = 0.274-0.446; implied flux-true
    |S21| = 0.971-0.997), a confirmed frequency-dependent drive-side
    scale bias kappa(f) = 1.49..1.86 (see PR body / follow-up issue). Do
    not cite this band as thru-transmission physics; when the kappa item
    lands, |S21| moves toward 0.97-1.0 and this fails LOUDLY —
    re-baseline in the same PR, do not widen the band to keep both
    behaviors green. The lower edge also catches a collapse back to the
    pre-#308 near-null (dead channel class); the upper edge stays
    strictly < 1 (over-unity extraction artefact class).
    """
    s21 = np.abs(thru_smatrix[1, 0])
    lo, hi = _THRU_S21_BAND
    # Per-bin lower edge (review finding): max() would let 8/9 dead bins
    # slip through. Measured per-bin min 0.5236 (~1.5x this floor).
    assert s21.min() > lo, (
        f"|S21| collapsed below the post-#308 envelope: per-bin min "
        f"{s21.min():.4f} <= {lo} (measured min 0.5236) — the recovered "
        f"channel died in at least one bin (dead probe / dead thru / "
        f"receive-sign regression class)")
    assert s21.max() < hi, (
        f"|S21| = {s21.max():.4f} left the measured envelope (max 0.6678, "
        f"band hi {hi}). If this is the drive-side kappa scale-bias fix "
        f"landing (flux-true |S21| = 0.971-0.997), re-baseline this "
        f"battery in the same PR")


@pytest.mark.slow_physics
def test_thru_s21_phase_band_is_sign_sensitive(thru_smatrix):
    """Signed S21 phase-deviation band + live conjugation discrimination.

    dev(f) = wrap(arg S21 - (-2*pi*f*L/c)) with the analytic ideal-thru
    delay for the 16 mm air line (DFT kernel exp(-j*2*pi*f*t) => e^{+jwt}
    phasors, outgoing wave e^{-j*beta*x}). Measured dev (amended receive
    sign) = -0.754891..-0.335259 rad; band [-1.1, -0.1] rad — a smooth
    ~16.6 ps feed-post group-delay excess over the 53.4 ps line delay,
    no pi offset. HONESTY LABEL — REGRESSION LOCK on the deviation
    values (flux-vs-port magnitude delta OPEN, kappa — module
    docstring); the overall SIGN is physics-anchored by the DC witness:
    the low-f falsifier measured S21(DC) -> +1 under the amended sign
    (V - Z0*I) (dev -0.024 rad at 0.5 GHz); the first-cut sign measured
    -1 and was amended in this same PR. Sign AND magnitude are gated (per-bin, signed) — verified
    NOT conjugation-invariant: the test also asserts conj(S21) violates
    the band (on the measured data 8/9 bins leave it; conj dev =
    +2.347..-0.834 rad, only the 7 GHz bin stays inside), so the
    W3.4-class conjugation bug cannot pass; a flip back to the first-cut
    receive sign shifts every bin by pi (~+2.39..+2.81 rad) and fails
    too.
    """
    s21 = thru_smatrix[1, 0]
    expected = np.exp(-1j * 2 * np.pi * _THRU_FREQS_HZ * _THRU_L_M / C0_M_PER_S)
    dev = np.angle(s21 / expected)              # wrapped signed deviation
    lo, hi = _THRU_PHASE_DEV_BAND_RAD
    print(f"\n[thru battery] signed phase dev (rad): {np.round(dev, 3)}")
    assert np.all((dev > lo) & (dev < hi)), (
        f"S21 signed phase deviation left [{lo}, {hi}] rad "
        f"(measured -0.755..-0.335 under the amended receive sign): "
        f"dev = {np.round(dev, 3)}. A receive-sign regression back to "
        f"the first-cut convention shifts this by pi; any deliberate "
        f"sign decision MUST re-baseline this battery in the same PR")

    # Live sign-discrimination witness: a conjugated S21 must FAIL the
    # same band, otherwise this gate has degraded into a |dev| check.
    dev_conj = np.angle(np.conj(s21) / expected)
    assert not np.all((dev_conj > lo) & (dev_conj < hi)), (
        "conj(S21) also satisfies the signed phase band — the gate lost "
        "its direction sensitivity (conjugation-blind)")


@pytest.mark.slow_physics
def test_thru_reciprocity(thru_smatrix):
    """Decomposer-symmetry lock: max|S21 - S12| small on the live channel.

    Measured post-#308: 1.043e-2 absolute (rel 0.016254) on the recovered
    O(0.52-0.67) magnitudes; gates 5e-2 / 0.10. The abs gate is scaled to
    the live channel (pre-#308 it was 5e-4 on the near-null residual —
    per-channel scale, not a weakening; the REL gate, the scale-free
    check, is unchanged and the measured rel barely moved: 0.0162 ->
    0.016254). A break catches an asymmetric edit to the shared
    decomposers or per-port Z0/n_cells bookkeeping. NOT transmission
    validation while the kappa magnitude item is open (module docstring).
    """
    s21 = thru_smatrix[1, 0]
    s12 = thru_smatrix[0, 1]
    abs_dev = np.abs(s21 - s12)
    rel_dev = abs_dev / np.maximum(np.abs(s21), np.abs(s12))
    assert abs_dev.max() < _THRU_RECIP_ABS_MAX, (
        f"reciprocity |S21-S12| = {abs_dev.max():.2e} "
        f"(measured 1.04e-2, gate {_THRU_RECIP_ABS_MAX})")
    assert rel_dev.max() < _THRU_RECIP_REL_MAX, (
        f"reciprocity rel dev = {rel_dev.max():.4f} "
        f"(measured 0.016254, gate {_THRU_RECIP_REL_MAX})")


@pytest.mark.slow_physics
def test_thru_passivity_singular_values(thru_smatrix):
    """Energy sanity: max singular value of every per-freq 2x2 slice < 0.9.

    Measured post-#308: 0.687410 (margin 0.31 to the physical bound 1.0,
    ~1.31x measured headroom to the gate). A strict-passivity regression
    lock that catches an over-unity extraction artefact on either channel
    — and, per the kappa open item (module docstring), a drive-side
    scale-bias fix moving |S21| to the flux-implied 0.971-0.997 would
    push the singular value past 0.9 and fail LOUDLY here too, forcing a
    re-baseline in the same PR. Do not cite this as transmission
    evidence while the port-based magnitude is unvalidated.
    """
    sv_max = max(
        np.linalg.svd(thru_smatrix[:, :, k], compute_uv=False)[0]
        for k in range(thru_smatrix.shape[2])
    )
    assert sv_max < _THRU_MAX_SINGULAR_VALUE, (
        f"max singular value {sv_max:.4f} (measured 0.6874 post-#308, "
        f"gate {_THRU_MAX_SINGULAR_VALUE}) — extraction produced "
        f"over-envelope energy on the thru (or the kappa scale-bias fix "
        f"landed: re-baseline this battery in the same PR)")
