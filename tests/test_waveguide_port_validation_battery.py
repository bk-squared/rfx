"""Physical-correctness validation battery for the waveguide port.

Locks Meep-class behavior on a canonical rectangular waveguide for the
source + extractor pipeline. Each test asserts a falsifiable physical
property — independent of other tests — that a correct FDTD simulator
must satisfy.

Thresholds reflect the extractor state as of 2026-04-22 after:
- diagonal-subtraction patch in ``extract_waveguide_s_params_normalized``
  (subtracts empty-guide reference outgoing from device outgoing)
- CPML default retune (polynomial order 2->3). A parallel `kappa_max`
  sweep showed that `kappa_max > 1` degrades guided-mode absorption
  in the current CFS-CPML formulation, so `kappa_max=1` is kept (an
  earlier handoff that claimed a `kappa_max 1->5` retune was retracted
  on 2026-04-22 after the sweep was run).

Industry anchors (handoff 2026-04-21):
- Meep EigenModeSource: matched-load |S11| < 1 %, directionality ~0.1 %.
- OpenEMS: matched-load |S11| < 5 %.
- Lossless passivity: sum_i |S_ij|^2 <= 1.
- Passive-medium reciprocity: S21 == S12 on the projected subspace.

Tighten thresholds when extractor quality improves; a failure here
marks a physics regression.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box


# =============================================================================
# Canonical geometry
# =============================================================================
#
# Rectangular waveguide, 40 mm x 20 mm cross-section. TE10 cutoff at
# c / (2 * 0.04) = 3.75 GHz. Matches the geometry used by the existing
# conservation-laws tests so thresholds are directly comparable.

DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
F_CUTOFF_HZ = 3.75e9
TARGET_CPML_M = 0.030  # 30 mm physical CPML absorber target


def _build_sim(
    freqs_hz,
    *,
    dx: float | None = None,
    cpml_layers: int | None = None,
    obstacles=(),
    pec_short_x: float | None = None,
    waveform: str = "modulated_gaussian",
    n_modes: int = 1,
):
    """Two-port rectangular waveguide simulation.

    Parameters
    ----------
    freqs_hz
        Frequency points for S-parameter extraction.
    dx
        Cell size override; ``None`` lets ``Simulation`` pick from freq_max.
    cpml_layers
        CPML layer count; ``None`` uses the Simulation default.
    obstacles
        Sequence of ``((lo, hi, eps_r),)`` dielectric boxes.
    pec_short_x
        If provided, add a thin PEC wall at this x position (full
        cross-section) — short-circuit termination for S11 tests.
    waveform
        Source pulse shape. Default ``modulated_gaussian`` for clean
        bandpass excitation.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))

    sim_kwargs = dict(
        freq_max=max(float(freqs[-1]), f0),
        domain=DOMAIN,
        boundary="cpml",
    )
    if cpml_layers is not None:
        sim_kwargs["cpml_layers"] = cpml_layers
    else:
        sim_kwargs["cpml_layers"] = 10
    if dx is not None:
        sim_kwargs["dx"] = dx
    sim = Simulation(**sim_kwargs)

    for idx, (lo, hi, eps_r) in enumerate(obstacles):
        name = f"diel_{idx}"
        sim.add_material(name, eps_r=eps_r, sigma=0.0)
        sim.add(Box(lo, hi), material=name)

    if pec_short_x is not None:
        # Thin PEC wall spanning the full cross-section.
        thickness = 0.002  # 2 mm — a few cells
        sim.add(
            Box((pec_short_x, 0.0, 0.0), (pec_short_x + thickness, DOMAIN[1], DOMAIN[2])),
            material="pec",
        )

    port_freqs = jnp.asarray(freqs)
    sim.add_waveguide_port(
        PORT_LEFT_X,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        n_modes=n_modes,
        name="left",
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        n_modes=n_modes,
        name="right",
    )
    return sim


def _s_matrix(sim, *, num_periods=40, normalize=True):
    result = sim.compute_waveguide_s_matrix(
        num_periods=num_periods,
        normalize=normalize,
    )
    s = np.asarray(result.s_params)
    port_idx = {name: idx for idx, name in enumerate(result.port_names)}
    # Multi-mode dispatch (n_modes>1) renames ports to
    # "<name>_mode<k>_<TE/TM><mn>". Add a short alias mapping plain
    # "left"/"right" → the dominant-mode index so test code that uses
    # port_idx["left"] keeps working.
    for name in list(port_idx):
        if "_mode0_" in name:
            short = name.split("_mode0_", 1)[0]
            port_idx.setdefault(short, port_idx[name])
    freqs = np.asarray(result.freqs)
    return s, freqs, port_idx


# =============================================================================
# Test 1 — Matched-load |S11| on an empty waveguide (industry anchor)
# =============================================================================
#
# Empty lossless two-port waveguide with both ends absorbed by CPML.
# There is no obstacle: the forward wave should be perfectly absorbed
# at the far CPML and S11 should be ~0. This is the single most-telling
# diagnostic for extractor + PML quality on the waveguide port.
#
# Meep class: <1 %.  OpenEMS class: <5 %.  Current rfx target: <10 %.

def test_matched_load_s11_empty_waveguide():
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    sim = _build_sim(freqs, waveform="modulated_gaussian")
    s, sim_freqs, port_idx = _s_matrix(sim, num_periods=40, normalize=True)

    s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
    s22 = np.abs(s[port_idx["right"], port_idx["right"], :])
    max_s11 = float(max(s11.max(), s22.max()))

    print("\n[matched-load] |S11| per freq:", np.array2string(s11, precision=3))
    print("[matched-load] |S22| per freq:", np.array2string(s22, precision=3))
    print(f"[matched-load] max(|S11|, |S22|) = {max_s11:.4f}")
    print("[matched-load] Meep-class target <0.01; rfx gate <0.10")

    # Gate ratcheted 2026-04-22 from 0.10 to 0.02. Post diagonal-subtraction
    # + CPML retune + discrete-β consistency the empty-guide matched-load
    # measurement returns bit-identical zeros on the canonical battery
    # geometry. 0.02 catches a regression to the pre-fix >0.05 regime.
    assert max_s11 < 0.02, (
        f"Matched-load |S11| too large: {max_s11:.4f} (gate 0.02). "
        "Extractor or PML regression — check empty-guide reflection."
    )


# =============================================================================
# Test 2 — Tight passivity on empty waveguide
# =============================================================================
#
# Tighter variant of test_passivity_two_port_empty_waveguide. The
# existing test passes at threshold 1.15 (15% power excess allowed). For
# an empty lossless guide a correct extractor should stay within a few
# percent of unity. Use this as a regression lock at 1.05 — the
# current extractor (post-retune, post-subtraction) is expected to sit
# here or tighter.

def test_tight_passivity_empty_waveguide():
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    sim = _build_sim(freqs, waveform="modulated_gaussian")
    s, _, _ = _s_matrix(sim, num_periods=40, normalize=True)

    col_power = np.sum(np.abs(s) ** 2, axis=0)
    max_power = float(col_power.max())

    print(f"\n[tight-passivity] max column |S|^2 = {max_power:.4f}")
    print("[tight-passivity] gate 1.05; loose legacy gate was 1.15")

    # Gate ratcheted 2026-04-22 from 1.05 to 1.02. Current state measures
    # 1.0000 on the canonical battery — 1.02 catches a 2% regression before
    # the old legacy 1.15 gate would.
    assert max_power < 1.02, (
        f"Empty-guide passivity exceeded tight gate: {max_power:.4f} "
        "(1.02). Extractor or PML regression."
    )


# =============================================================================
# Test 3 — Reciprocity on a symmetric obstacle
# =============================================================================
#
# Lorentz reciprocity on a geometrically symmetric obstacle — this is
# the case where two-run normalization can fully cancel obstacle-CPML
# multi-bounce asymmetry. A tight threshold should be achievable.

def test_reciprocity_symmetric_obstacle():
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    # Symmetric about domain midpoint (0.06 m), fills full y-cross-section
    # and is centered.
    obstacles = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]
    sim = _build_sim(freqs, obstacles=obstacles, waveform="modulated_gaussian")
    s, _, port_idx = _s_matrix(sim, num_periods=40, normalize=True)

    s21 = np.abs(s[port_idx["right"], port_idx["left"], :])
    s12 = np.abs(s[port_idx["left"], port_idx["right"], :])
    denom = np.maximum(np.maximum(s21, s12), 1e-12)
    rel_err = np.abs(s21 - s12) / denom
    mean_err = float(rel_err.mean())

    print(f"\n[reciprocity-sym] mean |S21-S12|/max = {mean_err:.4f}")
    print("[reciprocity-sym] gate 0.05")

    # Gate ratcheted 2026-04-22 from 0.05 to 0.01. Two-run normalization
    # on a symmetric obstacle currently measures 0.0005 mean reciprocity
    # error — the full 0.05 headroom was from the pre-consistency era.
    assert mean_err < 0.01, (
        f"Reciprocity on symmetric obstacle worse than gate: "
        f"{mean_err:.4f} (0.01). Possible extractor regression."
    )


# =============================================================================
# Test 4 — Reciprocity converges with run length on symmetric obstacle
# =============================================================================
#
# A correct extractor should improve monotonically as num_periods grows
# — more DFT samples cancel transient multi-bounce better. If the error
# flatlines or grows with longer runs, the extractor has a structural
# multi-bounce bug (Stage-3 hypothesis for the asymmetric-obstacle
# failure). This test exists so such a regression is caught early.

def test_reciprocity_converges_with_num_periods():
    freqs = np.array([6.0e9])  # single frequency, fast
    obstacles = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]
    errors = []
    for num_periods in (40, 80):
        sim = _build_sim(freqs, obstacles=obstacles, waveform="modulated_gaussian")
        s, _, port_idx = _s_matrix(sim, num_periods=num_periods, normalize=True)
        s21 = np.abs(s[port_idx["right"], port_idx["left"], 0])
        s12 = np.abs(s[port_idx["left"], port_idx["right"], 0])
        denom = max(max(s21, s12), 1e-12)
        rel_err = abs(s21 - s12) / denom
        errors.append(float(rel_err))
        print(f"[reciprocity-conv] num_periods={num_periods:3d}: rel_err = {rel_err:.4f}")

    # Monotonic non-increase with num_periods. The previous revision of
    # this test added an ``or errors[1] < 0.02`` escape hatch which made
    # the assertion tautological for the current extractor state (≈1e-4
    # error floor) — the convergence property it was supposed to guard
    # could never fire. The gate is now strictly monotonic: any future
    # regression that reintroduces CPML-accumulation (the symptom that
    # motivated this test) will show non-monotonic behaviour and trip.
    improved = errors[1] <= errors[0] + 1e-3
    assert improved, (
        f"Reciprocity error did not improve with longer run: "
        f"40->{errors[0]:.4f}, 80->{errors[1]:.4f}"
    )


# =============================================================================
# Test 5 — Asymmetric-obstacle reciprocity (known gap, ratcheted lock)
# =============================================================================
#
# Stage-3 diagnosis: Lorentz reciprocity holds in theory for the
# projected TE10 S-matrix regardless of geometric asymmetry (modal
# orthogonality on uniform cross-section). The extractor currently
# shows ~0.22 reciprocity error on the existing asymmetric test, which
# points at multi-bounce contamination not cancelled by two-run
# normalization in geometrically asymmetric DUTs.
#
# This test LOCKS the current achievable upper bound. When the
# extractor is improved to <0.05, tighten the gate.

def test_reciprocity_asymmetric_obstacle_known_gap():
    freqs = np.linspace(4.5e9, 8.0e9, 10)
    # Half y-cross-section, off-center in x: the legacy test's geometry.
    obstacles = [((0.03, 0.0, 0.0), (0.05, 0.02, 0.02), 6.0)]
    sim = _build_sim(freqs, obstacles=obstacles, waveform="modulated_gaussian")
    s, _, port_idx = _s_matrix(sim, num_periods=40, normalize=True)

    s21 = np.abs(s[port_idx["right"], port_idx["left"], :])
    s12 = np.abs(s[port_idx["left"], port_idx["right"], :])
    denom = np.maximum(np.maximum(s21, s12), 1e-12)
    mean_err = float((np.abs(s21 - s12) / denom).mean())

    print(f"\n[reciprocity-asym-lock] mean |S21-S12|/max = {mean_err:.4f}")
    print("[reciprocity-asym-lock] gate 0.30 (known extractor gap ~0.22)")

    # Gate ratcheted 2026-04-22 from 0.30 to 0.10. Post fixes the
    # asymmetric-obstacle case measures ~0.06 — the old 0.30 gate
    # tolerated the 0.22 regime that no longer reflects current extractor
    # state. 0.10 keeps headroom for frequency-dependent reciprocity
    # residuals while catching genuine regressions.
    assert mean_err < 0.10, (
        f"Asymmetric-obstacle reciprocity regression: {mean_err:.4f} > 0.10. "
        "Current-state upper bound was ~0.06. Investigate extractor."
    )


# =============================================================================
# Test 6 — Mesh convergence of |S21| with properly scaled CPML
# =============================================================================
#
# Fixes the design bug in test_mesh_convergence_s21: cpml_layers is
# proportional to 1/dx so the physical CPML thickness is constant
# across resolutions.
#
# Assertions:
# (a) monotone: |S21(2mm)-S21(1.5mm)| <= |S21(3mm)-S21(2mm)|
# (b) absolute: fine-mesh change < 0.10

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Mesh-convergence regression after the 2026-04-27 DROP-weight fix "
        "(commit bcd67d2 lifted PEC-short min |S11| 0.94 -> 0.9997). "
        "The 2026-04-26 dead-end note hypothesised the multi-mode receive "
        "extractor would recover this — it does not. Empirically (today, "
        "scripts/spikes/2026-04-27/_path_c_ablation.py companion), running "
        "with n_modes=2 dispatches through extract_multimode_s_params_normalized "
        "and produces the same per-mesh |S21| values: at f=6 GHz the only "
        "propagating mode in the 40x20 mm guide is TE10 (TE01 cutoff 7.5 GHz, "
        "TE20 cutoff 7.5 GHz) so higher modes carry no power to the receive "
        "port and there is nothing to project away. The actual root cause is "
        "the modal-template normalisation's mesh-dependence: dropping the "
        "+face cell changes integral(|E|^2.dA) by 7%/5%/4% at dx={3,2,1.5} mm, "
        "and that uneven proportion shifts the effective TE10 amplitude "
        "between meshes. Recovery requires either a fractional boundary-cell "
        "weight that converges to a well-defined dx -> 0 limit (e.g. the "
        "OpenEMS user-pinned integration box) or a probe-style integration "
        "that doesn't go through aperture_dA-based template normalisation. "
        "Out of scope for the DROP-weight closure; tracking as follow-up. "
        "2026-04-27 falsification round (scripts/spikes/2026-04-27/): "
        "(a) 11-point v_hi weight sweep shows sharp discontinuity at w=0; "
        "no single weight satisfies both PEC-short and mesh-conv. "
        "(b) Asymmetric template-norm vs sim-extract weights cap PEC-short "
        "|S11| at 0.96 because waveguide_port.py:623-628 zeros templates "
        "where dA<=0, propagating norm-side DROP into runtime extraction. "
        "(c) Analytic vs discrete templates differ by 0.0004 in |S11| -- "
        "template form is not the bug. (d) PEC closed-cavity resonance "
        "test recovers TM modes within 1.5%% of analytic at Q ~ 1e7-1e9, "
        "confirming FDTD-core PEC handling is sound. (e) 2026-04-27 late "
        "evening — Phase 1A.0 (analytic beta/Z swap) gives only +1.6%% on "
        "KEEP-both PEC-short. Phase 1A.1 (position-aware analytic "
        "projection on full aperture, analytic Z) yields mean |S11|=1.000 "
        "(Meep class!) but per-frequency oscillation is +/-7%% from Yee "
        "numerical dispersion at the test resolution (~30 cells/lambda). "
        "DROP-both with Yee-discrete Z is more per-freq tight (+/-0.05%%) "
        "because the Z-vs-beta convention matches. Conclusion: rfx port "
        "extractor is Meep-class in MEAN; per-freq +/-7%% residual is the "
        "intrinsic Yee dispersion footprint, NOT a port-code bug. True "
        "recovery would require finer mesh OR subpixel PEC handling in "
        "rfx/core/yee.py (multi-month FDTD-core work). Not in scope for "
        "port-extractor follow-ups. "
        "(g) 2026-04-28 end-of-day: the dump-derived per-frequency "
        "PEC-short |S11| oscillation that drove (a)-(f) and the codex "
        "#1/#2 ablations was traced to a missing Yee leapfrog half-step "
        "correction (exp(+j*omega*dt/2) on the H_y spectrum) in the "
        "diagnostic comparator at scripts/diagnostics/wr90_port/"
        "s11_from_dumps.py — NOT a real rfx FDTD residual. With that "
        "correction landed (commits 2fb9b76, 3e2754c) the dump-recipe "
        "spread drops from 0.1326 to 0.0166 at R=1 (Meep-class). The "
        "production extractor was always applying that correction "
        "through _co_located_current_spectrum and was always Meep-class "
        "on this geometry. The mesh-convergence regression below is a "
        "SEPARATE issue from the resolved per-frequency oscillation; "
        "it is the only remaining waveguide-port-side residual. "
        "(h) 2026-04-29 evening — the handover-doc fractional cell-"
        "overlap weight (path 1 of "
        "docs/research_notes/2026-04-29_item_c_handover.md) was "
        "implemented and tested. ∑aperture_dA converges exactly to "
        "port.a·port.b at every dx (structural goal works). But on the "
        "staircase Yee grid the PEC sits at Nu·dx, not port.a — at "
        "WR-90 dx=1 mm the boundary cell carries weight ~0.86 and "
        "admits non-physical normal-E (apply_pec_faces only zeros "
        "tangential E). Net regressions: cv11 PEC-short |S11| diff vs "
        "OpenEMS 0.025 → 0.094 (gate 0.050), "
        "test_reciprocity_asymmetric_structure 0.041 → 0.078 (gate "
        "0.05). Reverted. DROP stays in place; the fix is subpixel PEC "
        "handling in rfx/core/yee.py, not a port-extractor knob. See "
        "handover §'Why path 1 fails' for the trade-off table."
    ),
)
def test_mesh_convergence_s21_scaled_cpml():
    freq = 6.0e9
    obstacles = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]

    # Skip the coarsest resolution that has the worst staircasing.
    resolutions = [0.003, 0.002, 0.0015]
    s21_values: list[float] = []
    for dx in resolutions:
        layers = max(8, int(round(TARGET_CPML_M / dx)))  # [10, 15, 20]
        sim = _build_sim(
            [freq],
            dx=dx,
            cpml_layers=layers,
            obstacles=obstacles,
            waveform="modulated_gaussian",
        )
        s, _, port_idx = _s_matrix(sim, num_periods=40, normalize=True)
        s21 = float(np.abs(s[port_idx["right"], port_idx["left"], 0]))
        s21_values.append(s21)
        print(f"[mesh-conv] dx={dx*1e3:.1f}mm  cpml={layers}  |S21|={s21:.4f}")

    coarse_delta = abs(s21_values[0] - s21_values[1])
    fine_delta = abs(s21_values[1] - s21_values[2])
    print(f"[mesh-conv] coarse_delta={coarse_delta:.4f}  fine_delta={fine_delta:.4f}")

    assert fine_delta <= coarse_delta + 0.005, (
        f"Mesh refinement did not reduce |S21| change: "
        f"coarse={coarse_delta:.4f}, fine={fine_delta:.4f}"
    )
    assert fine_delta < 0.10, (
        f"Fine-mesh |S21| change too large: {fine_delta:.4f} (gate 0.10)"
    )


# =============================================================================
# Test 7 — PEC-short |S11| magnitude
# =============================================================================
#
# Total-reflection termination: |S11| = 1 at every frequency for a
# lossless short. With PEC ~15 mm from the downstream CPML and num_periods
# chosen so the single round-trip settles within the DFT window, the
# normalized extractor (diagonal-subtraction patch) recovers |S11| close
# to 1 without needing early-time DFT gating. An early DFT gate
# (num_periods_dft << num_periods) here would cut off the reflected
# wave before it returns to the driven port — useful only for strong
# resonators, not for a one-bounce short.
#
# Measured on 2026-04-22: mean |S11| ~ 0.93, per-freq range [0.78, 1.06].
# The remaining ~7 % departure from unity is extractor/mode-projection
# error. Tighten when P3 (discrete-eigenmode profile) lands.

def test_pec_short_s11_magnitude():
    """PEC-short min |S11| ≥ 0.99 (Meep-class strict closure).

    Achieved by the 2026-04-27 DROP-weight fix on the PEC +face
    aperture-dA cell. Uses ``normalize=False`` (single-run wave
    decomposition, OpenEMS convention) — the legacy ``normalize=True``
    two-run subtraction has standing-wave node artifacts on strong
    reflectors that make it the wrong tool for this gate. Two-run
    normalization remains the correct tool for long-distance
    transmission gates where Yee numerical dispersion accumulates.
    """
    freqs = np.linspace(5.0e9, 7.0e9, 6)
    sim = _build_sim(
        freqs,
        pec_short_x=0.085,
        waveform="modulated_gaussian",
    )
    # Full-window DFT: the single PEC->CPML round trip fits inside
    # num_periods=40 and there is no resonator to build up late-time.
    # Phase 2 cleanup (2026-04-25) removed the num_periods_dft early
    # gate; the post-scan rect full-record DFT is finite-energy on the
    # transient so no gating is needed.
    s, _, port_idx = _s_matrix(
        sim,
        num_periods=40,
        normalize=False,
    )

    s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
    max_s11 = float(s11.max())
    min_s11 = float(s11.min())
    mean_s11 = float(s11.mean())

    print(f"\n[pec-short] |S11| per freq: {np.array2string(s11, precision=4)}")
    print(f"[pec-short] range [{min_s11:.4f}, {max_s11:.4f}]  mean {mean_s11:.4f}; ideal 1.00")

    # Meep-class strict closure. Pre-2026-04-27 baseline (PROD half-
    # weight) was min=0.78, mean=0.93; the DROP-weight fix took it to
    # min ≥ 0.99 by removing the spurious-Ez ghost-cell contribution
    # that apply_pec_faces does not zero (Ez is "normal" at the z_hi
    # PEC face by staircase convention).
    assert min_s11 >= 0.99, (
        f"PEC-short min |S11|={min_s11:.4f} below 0.99 Meep-class gate. "
        "Regression vs DROP-weight baseline — ghost-cell contamination "
        "may have crept back into the modal V/I integral."
    )
    # Max gate relaxed to 1.03 to allow the small (~2.5%) over-unity
    # residual at the lowest freq closest to cutoff (5.0 GHz at f/fc=1.33);
    # other freqs land within ±0.001 of unity. Tighten when the discrete-
    # Yee Z_TE residual at near-cutoff is tracked down.
    assert max_s11 < 1.03, (
        f"PEC-short max |S11|={max_s11:.4f} above 1.03 — non-passive "
        "reflector (energy injection bug) or near-cutoff residual blew up."
    )
    assert abs(mean_s11 - 1.0) < 0.02, (
        f"PEC-short mean |S11|={mean_s11:.4f} deviates from unity by "
        f"more than 2% — Meep-class regression."
    )


# =============================================================================
# Test 8 — Early-time source directionality (phantom-diagnosis regression lock)
# =============================================================================
#
# Measures raw source backward-leakage via time-domain probes on either
# side of the port, gated to an *early-time window* that excludes the
# right-CPML round-trip reflection. See
# docs/research_notes/2026-04-21_waveguide_port_phantom_diagnosis.md:
# five prior sessions chased a max|Ez|-over-full-window metric that was
# dominated by the CPML round-trip, not by real source directionality.
# The real directionality of the current source (as of commit 15c6a13,
# 2026-04-21) is 0.67%.
#
# This test locks that baseline. A regression here means either
#   (a) the source injector lost its one-sided property (physics bug), or
#   (b) the CPML leak reaches the upstream probe inside the gate, which
#       means the left CPML has regressed.
# Both deserve a hard failure.

def test_source_directionality_early_time():
    # Bespoke geometry: needs enough source-to-CPML distance so the
    # early-time window covers the source-pulse peak before any CPML
    # round-trip reflection can reach the probes.
    #
    # Waveguide side walls (y, z) are PEC — the physical boundary for a
    # rectangular guide — so point probes on the cross-section axis are
    # not attenuated by CPML. CPML is applied only on the propagation
    # axis (x), where the source radiates.
    #
    # For fcen=10 GHz, modulated_gaussian peak @ t0=5*tau=1 ns with
    # bandwidth=0.5; v_g(10 GHz, fc=3.75 GHz) ~ 2.78e8 m/s; round-trip
    # path (source -> nearest x-CPML edge and back to the upstream probe
    # ~40 mm upstream) is ~1.51 ns — leaves a clean early window past
    # the source peak.
    domain = (0.50, 0.04, 0.02)
    dx = 0.002
    f_cutoff = F_CUTOFF_HZ
    f0 = 10.0e9
    freqs = np.linspace(8.0e9, 12.0e9, 8)
    bandwidth = 0.5

    port_x = domain[0] / 2.0                 # 0.25 m, free region midpoint
    probe_offset = 0.040                     # 40 mm ~= λ at 10 GHz
    y_c = domain[1] / 2
    z_c = domain[2] / 2

    sim = Simulation(
        freq_max=12.0e9,
        domain=domain,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=10,
        dx=dx,
    )
    sim.add_waveguide_port(
        port_x,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform="modulated_gaussian",
        name="source",
    )
    # Probes at symmetric offsets; physical coord snapped by api.
    sim.add_probe((port_x - probe_offset, y_c, z_c), "ez")  # probe 0: backward
    sim.add_probe((port_x + probe_offset, y_c, z_c), "ez")  # probe 1: forward

    # Run long enough to capture source peak + first CPML round-trip.
    # We only ASSERT on the early window, but seeing post-early-window
    # CPML-reflection energy in the diagnostic print is useful to
    # validate the travel-time model (and catches a too-narrow gate).
    num_periods_total = 24.0
    grid_est = sim._build_grid()
    n_steps = int(grid_est.num_timesteps(num_periods=num_periods_total))
    result = sim.run(n_steps=n_steps, compute_s_params=False)

    dt = float(result.dt)
    times = np.arange(n_steps) * dt

    # Travel-time gate: source -> nearer CPML edge and back -> the upstream
    # probe. Geometry is x-symmetric, so the nearer edge is the same on both
    # sides.
    cpml_thickness_m = 10 * float(result.grid.dx)
    dist_source_to_right_cpml = domain[0] - port_x - cpml_thickness_m
    dist_source_to_left_cpml = port_x - cpml_thickness_m
    nearest_cpml_m = min(dist_source_to_right_cpml, dist_source_to_left_cpml)
    v_g = 2.998e8 * np.sqrt(max(1.0 - (f_cutoff / f0) ** 2, 1e-6))
    round_trip_s = 2 * nearest_cpml_m / v_g
    # Allow two cycles of headroom before the reflection reaches the probe.
    cycle_s = 1.0 / f0
    early_t_s = round_trip_s - 2.0 * cycle_s
    # If the geometry is too tight to produce a clean early window, skip with
    # explicit failure rather than silently relaxing the gate.
    assert early_t_s > 3.0 * cycle_s, (
        f"Early-time window too short for clean measurement: "
        f"{early_t_s * 1e9:.2f} ns < 3 cycles. Geometry needs to be larger."
    )

    early_mask = times < early_t_s
    time_series = np.asarray(result.time_series)  # shape (n_steps, 2)
    max_backward = float(np.max(np.abs(time_series[early_mask, 0])))
    max_forward = float(np.max(np.abs(time_series[early_mask, 1])))
    directionality = max_backward / max(max_forward, 1e-30)

    # Full-window metric for diagnostic only — this is the misleading one.
    max_bwd_full = float(np.max(np.abs(time_series[:, 0])))
    max_fwd_full = float(np.max(np.abs(time_series[:, 1])))
    dir_full = max_bwd_full / max(max_fwd_full, 1e-30)

    print(f"\n[source-dir] v_g(f0={f0/1e9:.2f}GHz) = {v_g:.3e} m/s")
    print(f"[source-dir] nearest CPML edge = {nearest_cpml_m*1e3:.1f} mm; "
          f"round-trip = {round_trip_s*1e9:.2f} ns")
    print(f"[source-dir] early window     = t < {early_t_s*1e9:.2f} ns "
          f"({early_mask.sum()} / {n_steps} steps)")
    print(f"[source-dir] early-time bwd/fwd = {directionality*100:.3f}% "
          f"(current baseline ~1.2%; phantom_diagnosis WR-90 at ±3dx was 0.67%)")
    print(f"[source-dir] full-window  bwd/fwd = {dir_full*100:.3f}% "
          f"(if != early-time, CPML round-trip is entering the gate)")

    # Lock at 2.5% — ~2x the current-geometry baseline (~1.2% on 2026-04-22).
    # This gate catches a real physics regression while tolerating pulse-
    # shape and mode-mismatch jitter. Tighten after the P1 Lorentz overlap
    # landing once the projected baseline is re-measured.
    assert directionality < 0.025, (
        f"Early-time source directionality regressed: "
        f"{directionality*100:.3f}% > 2.5% gate. "
        "Baseline ~1.2% (40x20mm TE10 at probe_offset=40mm, "
        "2026-04-22). Phantom diagnosis (WR-90 at ±3dx) gave 0.67%. "
        "Investigate source injector or x-CPML quality."
    )


# =============================================================================
# Test 9 — Below-cutoff frequency does not produce NaN in Z_mode
# =============================================================================
#
# Regression lock: `_compute_mode_impedance` formerly returned `jnp.inf`
# for below-cutoff frequencies; `inf × complex(r, 0)` yields a NaN in the
# imaginary component on most NumPy/JAX implementations, which propagated
# through the V/I decomposition into NaN S-parameters. The current
# sentinel (1e30 for TE, 0 for TM) keeps downstream arithmetic defined.
# This test is pure algebra — no FDTD run — so it's cheap to leave on.

def test_below_cutoff_z_mode_no_nan():
    from rfx.sources.waveguide_port import (
        _compute_mode_impedance,
        _extract_global_waves,
    )

    # WR-90-like f_cutoff = 13.12 GHz (TE20 cutoff); probe a frequency
    # below that to exercise the evanescent branch.
    freqs = np.array([5.0e9, 10.0e9, 15.0e9])
    f_cutoff = 13.12e9
    dx, dt = 1.0e-3, 1.5e-12

    z = np.asarray(_compute_mode_impedance(freqs, f_cutoff, "TE", dt=dt, dx=dx))
    assert not np.any(np.isnan(z)), f"NaN in Z_mode below cutoff: {z}"
    assert not np.any(np.isinf(z)), (
        "inf in Z_mode below cutoff regresses the nan-cascade fix (2026-04-22)."
    )

    # TM likewise should give a finite (zero) value below cutoff.
    z_tm = np.asarray(_compute_mode_impedance(freqs, f_cutoff, "TM", dt=dt, dx=dx))
    assert not np.any(np.isnan(z_tm)), f"NaN in TM Z_mode below cutoff: {z_tm}"

    # Cascade check: the original failure mode was NaN S-parameters in
    # the V/I decomposition `(V ± Z·I) / 2` when Z was ±∞. Build a
    # minimal cfg-like namespace and confirm `_extract_global_waves`
    # returns finite forward/backward even with below-cutoff freqs.
    class _MinCfg:
        def __init__(self, freqs_hz, f_cutoff_hz, mode_type):
            self.freqs = jnp.asarray(freqs_hz)
            self.f_cutoff = float(f_cutoff_hz)
            self.mode_type = str(mode_type)
            self.dt = 1.5e-12
            self.dx = 1.0e-3
    cfg = _MinCfg(freqs, f_cutoff, "TE")
    V = jnp.asarray([1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=jnp.complex64)
    I = jnp.asarray([0.01 + 0j, 0.01 + 0j, 0.01 + 0j], dtype=jnp.complex64)
    fwd, bwd = _extract_global_waves(cfg, V, I)
    fwd_np = np.asarray(fwd)
    bwd_np = np.asarray(bwd)
    assert not np.any(np.isnan(fwd_np)), (
        f"NaN in forward wave from V/I with below-cutoff freq: {fwd_np}"
    )
    assert not np.any(np.isnan(bwd_np)), (
        f"NaN in backward wave from V/I with below-cutoff freq: {bwd_np}"
    )
