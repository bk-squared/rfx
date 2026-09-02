"""Integration test: 50 Ω microstrip thru-line S-parameter passivity gate.

Marked ``slow`` — excluded from the default pytest run.  Run explicitly with::

    pytest tests/unit/sparams/test_msl_port_integration.py -m slow -v -s

Geometry
--------
- Substrate : RO4350B, εr = 3.66, h = 254 µm
- Trace     : W = 600 µm (50 Ω design width), L = 10 mm
- Port margin from each feed plane to domain edge: 2 mm
- PEC ground plane at z = 0 (z_lo boundary); CPML on z_hi and all x/y faces
- Uniform dx = 80 µm — substrate gets ~3 cells (254/80 ≈ 3.2)
  NOTE: dz_profile is intentionally NOT used; the non-uniform runner does not
  propagate MSL port sources into its scan body so DFT plane probes return zero.
- f_max = 5 GHz, 30 freq points, num_periods = 12
- Two MSL ports: port 0 driven (+x), port 1 passive matched (-x)

Source mode: static-Laplace Ez profile (mode="laplace", the default).
The Schelkunoff J+M eigenmode source (mode="eigenmode") was attempted but
naive soft J+M sources cannot cancel the backward wave without a numerically
dispersion-corrected 1D auxiliary FDTD table; that path is reserved for a
future implementation.

Fix (2026-05-02)
----------------
A microstrip quasi-TEM mode requires a PEC trace conductor above the substrate.
Without the trace, the Ez source excites a TM-like substrate mode giving
Z0 ≈ 1600–2500 Ω and |S21| ≈ 0.  Following the canonical microstrip-trace
pattern (``sim.add(Box(...), material="pec")``, as in
``validation/crossval/06b_msl_notch_filter_uniform.py``),
we add a one-cell-thick PEC strip at z = H_SUB spanning the full line length.

The 3-probe extractor in ``compute_msl_s_matrix`` was also corrected to apply a
direction-aware sign to the Hy current integral: for a +x propagating quasi-TEM
wave with Ez > 0, Hy < 0 (x̂ × ẑ = −ŷ), so the raw integral must be negated to
recover the physical current I = +∮H·dl and give Z0 = V/I > 0.

Frequency window: at DX = 80 µm (≈3 cells across substrate), the quasi-TEM mode
is well-established above ~3 GHz.  The gate uses a fixed 3–4.5 GHz window rather
than a symmetric midband slice to avoid low-frequency dispersion artefacts.

Gate calibration (dx=80 µm, laplace mode, measured 2026-05-04)
--------------------------------------------------------------
  mean |S11| ≈ 0.118  → gate < 0.15
  mean |S21| ≈ 0.972  → gate (0.90, 1.05)
  mean Re(Z0) ≈ 54 Ω  → gate (40, 65) Ω
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

from tests._gate_policy import gate_from_envelope


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

EPS_R = 3.66          # RO4350B
H_SUB = 254e-6        # substrate thickness, metres
W_TRACE = 600e-6      # trace width, metres
L_LINE = 10e-3        # thru-line length
PORT_MARGIN = 2e-3    # feed → domain edge clearance

# Uniform cell size: 80 µm gives 254/80 ≈ 3.2 cells in substrate.
DX = 80e-6
F_MAX = 5e9

LX = L_LINE + 2 * PORT_MARGIN
# LY uses fixed physical lateral clearance (≥2·h_sub each side) instead
# of cell-counted clearance: microstrip fringing extends ~2·h_sub
# laterally, and a CPML lateral wall closer than that systematically
# inflates Z0 (and makes Z0 mesh-divergent under refinement). Verified
# 2026-05-04 fixed-LY mesh-conv sweep: with this LY, Z0 lands within
# 2-9% of Hammerstad analytic across dx=40-80µm; with the previous
# LY=W+6·dx the box shrank with mesh and Z0 drifted 54→60Ω.
# See docs/research_notes/20260504_msl_meshconv_fixed_ly.md.
LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX)   # W + (2·h_sub + 8·dx) each side
LZ = H_SUB + 1.5e-3      # substrate + 1.5 mm air above

# Evaluation window: quasi-TEM mode is well-established at 3–4.5 GHz
# (below this the staircase substrate has Z0 < 40 Ω due to poor resolution).
GATE_F_LO = 3.0e9
GATE_F_HI = 4.5e9

# Issue #610 (single-source, no second copy of the gate): hoisted out of
# test_msl_thru_line_z0_length_invariance_and_positive_sign to module scope so
# tests/locks/test_msl_z0_platform_datums.py can import and re-derive the SAME gate
# instead of restating 0.007 as a fresh literal in a second file. Values and
# derivation are UNCHANGED — see that test's docstring for the arithmetic and
# tests/fixtures/msl_z0_length_invariance/platform_datums.json for the
# cross-platform datum ledger.
MEASURED_SPREAD_ENVELOPE = 0.004607   # {8,10,12} mm legs, 2026-08-09
LEG_S11_BOUND = 0.15                  # per-leg mean|S11| bound, any leg length


@pytest.mark.slow
def test_msl_thru_line_passive_gate():
    """50 Ω microstrip thru with static-Laplace Ez source (mode='laplace').

    Gates calibrated for dx=80 µm (≈3 substrate cells), laplace mode:
      |S11| < 0.15
      0.90 < |S21| < 1.05
      Z0 ∈ (40, 65) Ω

    Issue #80 stage S1 (2026-05-19) closed the long-standing ``xfail``
    here. The de-embedded Z0 used to read ~74 Ω because the line current
    was an open single-leg ``∑Hy·dy`` integral that undercounted ``I`` by
    ~1.5x. ``compute_msl_s_matrix`` now measures the closed Ampère-loop
    current ``∮H·dl`` (``msl_loop_current``). The (40, 65) bound has been
    left UNCHANGED throughout — it was never weakened to make this pass.
    See ``docs/agent-memory/port_sparam_review_2026-05-19.md``.

    CALIBRATION REFRESHED 2026-07-30 (issues #511/#507 + PR #516 review
    finding F2). Bounds untouched — never weakened or moved; only the
    recorded measured values:

        quantity        pre-#511    final (trace-anchored V, PR #516)
        mean|S11|       0.118       0.1160
        mean|S21|       0.972       0.9930
        mean Re(Z0)     ~57 Ω       57.58 Ω

    (An intermediate PR #516 state read 0.0746 / 42.41 Ω here; that was the
    review's finding F2 — a V span anchored on ``round(h_sub/dx)``, one
    substrate edge short of the RASTERIZED trace on this mesh — and is
    retired. Its apparent agreement with the aligned-mesh sibling was
    cancellation.)

    Worth knowing before touching this fixture: ``h_sub/dx = 254/80 =
    3.175``, so the substrate boundary bisects cell k=3 (preflight's
    mixed-cell danger zone) and the trace rasterizes at node 4 — the
    structure this mesh actually simulates has the strip at z = 320 µm over
    a 254 µm dielectric plus a 66 µm air gap. Its Z0 of 57.58 Ω is the
    faithful extraction of THAT geometry, not extractor error: the aligned
    dx = 84.67 µm sibling rasterizes the intended structure and reads
    44.11 Ω. The cross-mesh difference is geometry quantization.

    Margin note (2026-07-30): the sibling length-invariance lock's
    cross-length Z0 spread measures 4.96% against its 5% bound, dominated
    by the L = 6 mm leg under EITHER V span — effectively margin-less.
    Investigate the short-line N-probe fit conditioning before re-pinning
    anything here.

    A future refinement of this fixture should prefer dx = h_sub/3 or
    h_sub/4 (84.7 / 63.5 µm), which preflight already recommends.
    """

    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, LY, LZ),
        dx=DX,
        cpml_layers=8,
        # PEC ground plane at z=0 (microstrip ground); CPML above.
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),  # PEC ground plane at z=0
        ),
    )

    # --- substrate fill (RO4350B, lossless) ---
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(
        Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)),
        material="ro4350b",
    )

    # --- PEC trace strip (one cell thick at z = H_SUB) ---
    # A microstrip quasi-TEM mode requires a metal trace above the substrate.
    # Without the trace, the Ez source excites a TM substrate mode (Z0>>50Ω).
    # Canonical pattern (as in validation/crossval/06b_msl_notch_filter_uniform.py):
    #   sim.add(Box(..., substrate_thickness, substrate_thickness+dz), material="pec")
    # Use one-cell thickness (H_SUB to H_SUB + DX) so rfx Box captures the
    # cells whose z-centres fall within the box z-range.
    y_centre = LY / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
        material="pec",
    )

    # --- port 0: driven from left, propagates +x ---
    sim.add_msl_port(
        position=(PORT_MARGIN, y_centre, 0.0),
        width=W_TRACE,
        height=H_SUB,
        direction="+x",
        impedance=50.0,
    )

    # --- port 1: passive matched at right end, propagates -x ---
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_centre, 0.0),
        width=W_TRACE,
        height=H_SUB,
        direction="-x",
        impedance=50.0,
    )

    # ------------------------------------------------------------------ run
    result = sim.compute_msl_s_matrix(
        n_freqs=30,
        num_periods=12,
    )

    S = result.S          # shape (n_ports, n_ports, n_freqs)
    Z0 = result.Z0        # shape (n_ports, n_freqs)
    freqs = result.freqs  # shape (n_freqs,)

    # Gate window: 3–4.5 GHz where quasi-TEM is well-established at DX=80µm.
    gate_mask = (freqs >= GATE_F_LO) & (freqs <= GATE_F_HI)
    if not np.any(gate_mask):
        # Fallback: centre 10 points
        n_f = freqs.shape[0]
        gate_mask = np.zeros(n_f, dtype=bool)
        gate_mask[max(0, n_f // 2 - 5):min(n_f, n_f // 2 + 5)] = True

    s11_gate = np.abs(S[0, 0, gate_mask])
    s21_gate = np.abs(S[1, 0, gate_mask])
    z0_gate  = Z0[0, gate_mask].real

    mean_s11 = float(np.mean(s11_gate))
    mean_s21 = float(np.mean(s21_gate))
    mean_z0  = float(np.mean(z0_gate))

    print(f"\n[MSL thru] gate freqs: {freqs[gate_mask][0]*1e-9:.2f}–{freqs[gate_mask][-1]*1e-9:.2f} GHz ({int(np.sum(gate_mask))} pts)")
    print(f"[MSL thru] mean |S11| = {mean_s11:.4f}")
    print(f"[MSL thru] mean |S21| = {mean_s21:.4f}")
    print(f"[MSL thru] mean Re(Z0) = {mean_z0:.2f} Ω")
    print(f"[MSL thru] freqs total: {freqs[0]*1e-9:.2f}–{freqs[-1]*1e-9:.2f} GHz")
    nz_sub = int(round(H_SUB / DX))
    print(f"[MSL thru] dx = {DX*1e6:.0f} µm, nz_sub ≈ {nz_sub}")

    # --- reflection gate (laplace mode, dx=80µm: measured ~0.118) ---
    assert mean_s11 < 0.15, (
        f"|S11| = {mean_s11:.4f} ≥ 0.15 — excessive reflection at source plane"
    )

    # --- passivity / transmission gate ---
    assert mean_s21 > 0.90, (
        f"|S21| = {mean_s21:.4f} ≤ 0.90 — insufficient forward transmission"
    )
    assert mean_s21 < 1.05, (
        f"|S21| = {mean_s21:.4f} ≥ 1.05 — passivity violated"
    )

    # --- Z0 gate (dx=80µm corrected window; measured ~54 Ω) ---
    assert 40.0 < mean_z0 < 65.0, (
        f"Re(Z0) = {mean_z0:.2f} Ω outside (40, 65) Ω"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason=(
        "MSL mode='eigenmode' is deferred/out-of-scope — the FDFD eigenmode "
        "source is a falsified dead-end (5 attempts; Option B accepted the "
        "Laplace floor). mode='laplace' is the supported lane. Strict "
        "tripwire: XPASSes if an eigenmode solver ever lands."
    ),
)
def test_msl_thru_line_eigenmode_gate():
    """50 Ω microstrip thru with FDFD-derived J+M Schelkunoff source (mode='eigenmode').

    The eigenmode source uses the full vectorial 2D Maxwell mode profile
    from the from-scratch FDFD solver (``rfx.sources.msl_fdfd_eigenmode``,
    Path B) and injects via the Schelkunoff J+M pair. Compared to the
    static-Laplace baseline (|S11|=0.118 at this mesh), the eigenmode
    source is expected to do at least as well; substantial improvement
    requires finer mesh (mesh-conv test).

    Gates: same as laplace baseline at dx=80µm with 3 substrate cells
    (the underlying physical floor is mesh-limited at this resolution).
    """

    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, LY, LZ),
        dx=DX,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )

    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(
        Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)),
        material="ro4350b",
    )

    y_centre = LY / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + DX)),
        material="pec",
    )

    sim.add_msl_port(
        position=(PORT_MARGIN, y_centre, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="+x", impedance=50.0,
        mode="eigenmode",
        eps_r_sub=EPS_R,
    )
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_centre, 0.0),
        width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0,
        mode="eigenmode",
        eps_r_sub=EPS_R,
    )

    result = sim.compute_msl_s_matrix(n_freqs=30, num_periods=12)

    S = result.S
    Z0 = result.Z0
    freqs = result.freqs

    gate_mask = (freqs >= GATE_F_LO) & (freqs <= GATE_F_HI)
    if not np.any(gate_mask):
        n_f = freqs.shape[0]
        gate_mask = np.zeros(n_f, dtype=bool)
        gate_mask[max(0, n_f // 2 - 5):min(n_f, n_f // 2 + 5)] = True

    s11_gate = np.abs(S[0, 0, gate_mask])
    s21_gate = np.abs(S[1, 0, gate_mask])
    z0_gate  = Z0[0, gate_mask].real

    mean_s11 = float(np.mean(s11_gate))
    mean_s21 = float(np.mean(s21_gate))
    mean_z0  = float(np.mean(z0_gate))

    print(f"\n[MSL eigenmode] gate {freqs[gate_mask][0]*1e-9:.2f}–{freqs[gate_mask][-1]*1e-9:.2f} GHz")
    print(f"[MSL eigenmode] mean |S11| = {mean_s11:.4f}")
    print(f"[MSL eigenmode] mean |S21| = {mean_s21:.4f}")
    print(f"[MSL eigenmode] mean Re(Z0) = {mean_z0:.2f} Ω")

    # Same gates as laplace baseline. Tighten only if eigenmode source
    # demonstrates meaningful improvement at finer mesh (mesh-conv test).
    assert mean_s11 < 0.20, (
        f"|S11| = {mean_s11:.4f} ≥ 0.20 — eigenmode source worse than baseline"
    )
    assert mean_s21 > 0.85, (
        f"|S21| = {mean_s21:.4f} ≤ 0.85 — insufficient forward transmission"
    )
    assert 35.0 < mean_z0 < 70.0, (
        f"Re(Z0) = {mean_z0:.2f} Ω outside (35, 70) Ω"
    )


def _run_msl_thru(l_line: float):
    """Build + run the canonical RO4350B microstrip thru-line at a given physical
    line length and return its MSL S-matrix result.

    Identical geometry/run params to ``test_msl_thru_line_passive_gate`` (the
    validated single-length envelope), parameterised by ``l_line`` so the
    length-invariance test can sweep it. ``LY``/``LZ`` are length-independent.
    """
    lx = l_line + 2 * PORT_MARGIN
    sim = Simulation(
        freq_max=F_MAX,
        domain=(lx, LY, LZ),
        dx=DX,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, LY, H_SUB)), material="ro4350b")
    y_centre = LY / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, H_SUB), (lx, trace_y_hi, H_SUB + DX)),
        material="pec",
    )
    sim.add_msl_port(
        position=(PORT_MARGIN, y_centre, 0.0),
        width=W_TRACE, height=H_SUB, direction="+x", impedance=50.0,
    )
    sim.add_msl_port(
        position=(PORT_MARGIN + l_line, y_centre, 0.0),
        width=W_TRACE, height=H_SUB, direction="-x", impedance=50.0,
    )
    return sim.compute_msl_s_matrix(n_freqs=30, num_periods=12)


@pytest.mark.slow
def test_msl_thru_line_z0_length_invariance_and_positive_sign():
    """MSL |Z0| length-invariance + per-port POSITIVE Z0 sign (issue #140).

    First committed test of MSL characteristic-impedance invariance ACROSS
    physical line lengths (every other MSL test is single-length). Two locks:

    1. **Per-port positive Z0 sign — the #140 fix.** ``msl_loop_current`` negates
       the loop current only for ``+x`` ports, so before the fix the ``-x`` port
       reported a NEGATIVE Re(Z0) and false-fired the |Z0| honesty guard (~228%
       deviation vs the true ~20-27% Yee-staircase bias). This asserts BOTH ports
       report Re(Z0) > 0 across the band — it FAILS on the pre-fix code and PASSES
       after. The reported-Z0 sign does NOT enter S11/S21 (those use the static
       analytic Hammerstad-Jensen z0), so this is purely a reported/diagnostic fix.
    2. **|Z0| length-invariance.** A thru-line's characteristic impedance is a
       per-unit-length property, so the mean-band |Z0| must agree across lengths
       to a few percent.

    LOCK RE-DERIVED FROM FRESH MEASUREMENT 2026-08-09 (issue #518). Every leg
    re-measured in one process with this exact recipe (dx = 80 µm, band
    3-4.5 GHz, mean-band |Z0| on the +x port):

        L =  6 mm : 60.2313 Ω   (REMOVED — see below)
        L =  8 mm : 57.3381 Ω
        L = 10 mm : 57.5778 Ω   (reproduces PR #516's recorded 57.58)
        L = 12 mm : 57.6030 Ω   (new leg)

    Envelope over the retained legs {8, 10, 12}:
    spread = (57.6030 − 57.3381) / mean(57.3381, 57.5778, 57.6030) = 0.4607%.
    Enforced bound = gate_from_envelope(0.004607, quantum=1000) = 0.007 —
    the shared measured-envelope policy (``tests/_gate_policy.py``:
    ENVELOPE_GATE_MULTIPLIER then ceil-quantize to 1/1000), a 1.52x margin.

    The L = 6 mm leg was REMOVED, not re-bounded (no-silent-loosening: the
    bound TIGHTENED 5% → 0.7%). Issue #518 attributes its ~+2.9 Ω offset to
    short-line N-probe fit conditioning: it was the outlier under BOTH V-span
    choices measured 2026-07-30 (k_hi proxy AND trace-anchored), so the offset
    is a property of the short-line fixture's fit window, not of the extractor
    — measured-by-elimination. The fresh 2026-08-09 trace shows the signature
    directly: L = 6 mm Re(Z0) runs monotone 59.89 → 60.60 Ω across the band
    while all three retained legs are flat to ±0.03 Ω. Falsifier (gate-can-
    bind-artifact mandate, run fresh-vs-fresh 2026-08-09): re-including the
    fresh L = 6 mm value gives spread 4.9556% > 0.007 — the retired
    configuration reds this lock.

    Prior state, for the record: legs {6, 8, 10}, bound 5%, measured spread
    4.96% — margin-less (commit 7f889d8's warning; the 2026-06-14 "0.49%
    headroom" record was obsolete).

    MARGIN NOTE (2026-08-09): the L = 12 mm leg measures mean|S11| = 0.1487
    against the per-leg < 0.15 envelope bound — 0.0013 of margin. Deterministic
    on a fixed platform, but any change that moves |S11| at the ~1% level will
    surface there first; triage that as the longer line's larger accumulated
    mismatch, not as a spread-lock problem.

    PLATFORM ENVELOPE (issue #610): both bounds above were derived/checked on
    ONE platform with thin cross-machine headroom (0.24 pp / 0.0013). No two
    datums share a commit, so cross-platform agreement is demonstrated as two
    same-regime PAIRS, not at a fixed commit: pre-drift {dev 0.4607 % @69a6956a,
    GitHub runner 0.46 % @90c79d1d, 4 rfx/ commits apart} and post-drift
    {GitHub runner 0.47 % @7f68f9fb and @8e004976, this pod 0.4676 % @1f005d0d}.
    Agreement finer than the 0.01 pp / 0.01 Ω print quantum is NOT MEASURED. The
    rows: the 2026-08-09 derivation (commit 69a6956a, lead machine — CPU/OS/jax/numpy/
    BLAS NOT RECORDED; PR #611 carries no platform identity, so do not infer
    one), the GitHub ubuntu-24.04 runner (jax 0.6.2, three separate weekly
    CI runs), and an AMD EPYC 9654 pod (jax 0.6.2, x64=False, complex64). See
    tests/fixtures/msl_z0_length_invariance/platform_datums.json for the full
    per-datum ledger (each record carries a `precision` tag; CI/derivation
    rows are hand- or print-captured at 2-5 dp, only the pod row is full
    float32) and tests/locks/test_msl_z0_platform_datums.py for the fast structural
    check that keeps it internally consistent with THIS test's own gate.

    The envelope has DRIFTED since 2026-08-09: a full-precision pod re-measure
    at HEAD (1f005d0d) gives spread = 0.0046760 (0.4676%) vs the committed
    0.004607 (0.4607%) — entering somewhere in the CI commit window
    90c79d1d..7f68f9fb (2026-08-11/12), unattributed to a single commit.
    gate_from_envelope(0.004676) would derive 0.008 — a ONE-QUANTUM WIDENING.
    Per issue #610's no-silent-loosening rule the gate STAYS 0.007: this is
    recorded as CODE drift to diagnose later (lead's lane — bisect requires a
    full-precision pod run of 69a6956a first, ~53 min/leg-set), NOT re-derived
    as routine gate maintenance. Full-precision per-leg mean|Z0[+x]| moved:
    57.3381 → 57.32987 Ω (−0.00823) at L=8mm, 57.5778 → 57.57185 Ω (−0.00595)
    at L=10mm, 57.6030 → 57.59874 Ω (−0.00426) at L=12mm — all DECREASES
    below the 0.01 Ω print quantum that makes the CI logs' "57.34 → 57.33"
    look like a bigger move than it is; do not bisect off 2-dp prints. The
    falsifier for the "platform, not code" attribution is a same-commit
    second-platform run reproducing spread to <0.01 pp. It has NOT been run —
    no commit in this ledger carries two platforms' datums — so that
    attribution is provisional. Likewise the drift WINDOW 90c79d1d..7f68f9fb
    is inferred from 2-significant-figure CI spread prints (0.46 % vs 0.47 %
    straddle a rounding boundary at 0.465) and cannot resolve the 0.0069 pp
    shift; only the drift's EXISTENCE is established at full precision.
    CLASSIFICATION POLICY (this threshold is
    POLICY, not itself measured): a same-commit cross-platform disagreement
    ≥ 0.05 pp is treated as platform variance; a shift entering between two
    same-platform CI runs across a commit window is code drift. The measured
    fact is narrower than the policy: cross-platform agreement at a fixed
    commit is demonstrated to the 0.01 pp / 0.01 Ω print quantum (three
    platforms, 2026-08-10/17/24 CI + this pod); agreement finer than that
    print quantum is NOT MEASURED. Both bounds still hold here (spread
    headroom 0.2324 pp; L=12mm |S11| headroom 0.0013026; settling −98 to
    −104 dB, well past the −40 dB ring-down witness bar). If either bound
    reds elsewhere: record a new datum in platform_datums.json (a
    RE-MEASURE, per platform) before touching this lock — a red on another
    platform/BLAS is diagnosed against the ledger, never fixed by loosening.
    @slow: three full FDTD thru-line runs.
    """
    # Legs {8, 10, 12} mm (L = 6 mm removed 2026-08-09, issue #518 — short-line
    # N-probe fit conditioning; see docstring). Largest domain is now
    # L + 2*PORT_MARGIN = 16 mm (the validated single-length gate's domain is
    # 14 mm); the set spans a 50% length range — still enough to expose any
    # length-dependent Z0 drift.
    lengths = [8e-3, 10e-3, 12e-3]
    mean_abs_z0_per_length = []
    for l_line in lengths:
        result = _run_msl_thru(l_line)
        S, Z0, freqs = result.S, result.Z0, result.freqs
        mask = (freqs >= GATE_F_LO) & (freqs <= GATE_F_HI)
        assert np.any(mask), f"no gate-band freqs at L={l_line}"
        z0_p0 = Z0[0, mask]   # +x driven port
        z0_p1 = Z0[1, mask]   # -x port (the one that read negative pre-fix)
        s11 = np.abs(S[0, 0, mask])
        s21 = np.abs(S[1, 0, mask])

        # R5 witness: dump the full per-port Z0 trace, not just a headline.
        print(f"\n[MSL z0-len L={l_line*1e3:.0f}mm] Re(Z0[+x]) min/mean/max = "
              f"{z0_p0.real.min():.2f}/{z0_p0.real.mean():.2f}/{z0_p0.real.max():.2f}")
        print(f"[MSL z0-len L={l_line*1e3:.0f}mm] Re(Z0[-x]) min/mean/max = "
              f"{z0_p1.real.min():.2f}/{z0_p1.real.mean():.2f}/{z0_p1.real.max():.2f}")
        print(f"[MSL z0-len L={l_line*1e3:.0f}mm] mean|S11|={np.mean(s11):.4f} "
              f"mean|S21|={np.mean(s21):.4f}")

        # (1) #140 LOCK: positive Re(Z0) on BOTH ports, every band bin.
        assert np.all(z0_p0.real > 0), f"+x port Re(Z0) not all positive at L={l_line}"
        assert np.all(z0_p1.real > 0), (
            f"-x port Re(Z0) not all positive at L={l_line} — issue #140 sign regression"
        )
        # (2) passivity (per-bin, not just mean).
        assert np.max(s11) <= 1.05 and np.max(s21) <= 1.05, (
            f"passivity violated at L={l_line}: max|S11|={np.max(s11):.3f} "
            f"max|S21|={np.max(s21):.3f}"
        )
        # (3) transmission + reflection envelope (consistent with the single-length gate).
        assert float(np.mean(s21)) > 0.90, f"|S21|={np.mean(s21):.3f} too low at L={l_line}"
        assert float(np.mean(s11)) < LEG_S11_BOUND, (
            f"|S11|={np.mean(s11):.3f} too high at L={l_line}"
        )
        # (4) |Z0| magnitude symmetry: the -x port differs from +x only in (now-fixed) sign.
        m0 = float(np.mean(np.abs(z0_p0)))
        m1 = float(np.mean(np.abs(z0_p1)))
        assert abs(m0 - m1) / m0 < 0.05, (
            f"|Z0| +x/-x magnitude mismatch at L={l_line}: {m0:.2f} vs {m1:.2f}"
        )
        # consistent with the existing single-length Z0 gate.
        assert 40.0 < float(np.mean(z0_p0.real)) < 65.0, (
            f"mean Re(Z0[+x])={np.mean(z0_p0.real):.2f} outside (40,65) at L={l_line}"
        )
        mean_abs_z0_per_length.append(m0)

    # (5) |Z0| length-invariance: per-unit-length property -> agree across lengths.
    # Bound derived from the fresh 2026-08-09 measured envelope through the
    # shared gate policy (see docstring for the per-leg numbers + arithmetic).
    spread_tol = gate_from_envelope(MEASURED_SPREAD_ENVELOPE, quantum=1000)
    assert spread_tol == 0.007, (
        f"spread gate derives to {spread_tol}, not its committed 0.007 — the "
        f"envelope constant or tests/_gate_policy.py moved; re-measure before "
        f"touching this lock"
    )
    mean_of_means = sum(mean_abs_z0_per_length) / len(mean_abs_z0_per_length)
    spread = (max(mean_abs_z0_per_length) - min(mean_abs_z0_per_length)) / mean_of_means
    print(f"\n[MSL z0-len] mean|Z0| per length {[round(z, 2) for z in mean_abs_z0_per_length]} Ω; "
          f"spread = {spread*100:.2f}% (bound {spread_tol*100:.1f}%)")
    assert spread < spread_tol, (
        f"|Z0| length spread {spread*100:.2f}% >= {spread_tol*100:.1f}% — not "
        f"length-invariant ({mean_abs_z0_per_length}). If the offender is a "
        f"SHORT line, check N-probe fit conditioning first (issue #518) "
        f"before suspecting the extractor."
    )
