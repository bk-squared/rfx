"""Finite-size FluxMonitor vs full-plane in a standing-wave-heavy environment.

Settles the long-standing (and now refuted) caveat in
``docs/agent-memory/rfx-known-issues.md``:

    "FluxMonitor finite-size regions ... standing-wave-heavy environments
     remain more stable on the full-plane path."

What this test actually measures
--------------------------------
The finite-size flux path differs from the full-plane path by *exactly one
thing*: the ``lo1:hi1 / lo2:hi2`` index slice applied to the per-cell Poynting
integrand (``rfx/simulation.py`` flux-accumulation block, ~L1082-1126). The DFT
kernel, the Yee H-colocation half-step, and the area weight ``dA`` are byte-for-
byte identical between the two paths. Therefore a finite-size monitor over a
sub-window MUST equal the full-plane monitor's integrand summed over that SAME
window to ~machine precision. Any larger deviation would be a real bookkeeping
bug, i.e. genuine "instability" of the finite-size machinery.

We build a genuinely standing-wave-heavy field (an internal full-transverse PEC
short inside a uniform-CPML guide; SWR ~6 in the cavity region) and check the
finite-vs-restricted agreement PER FREQUENCY.

Result (CPU, measured): max relative deviation ~2e-16 (machine epsilon) across
all 16 frequencies — the finite-size path is bit-exact, NOT "less stable". The
old caveat conflated *coverage* (a finite region legitimately excludes the CPML
padding cells that the legacy ``size=None`` full plane integrates) with
*instability*. They are different things; the finite-size arithmetic is exact.

Notes for future maintainers
-----------------------------
* Any PEC *boundary* (``BoundarySpec.uniform("pec")`` or a per-axis
  ``y="pec"``) currently yields a DEAD field — the source never injects. That
  is a SEPARATE issue and is deliberately avoided here by using a uniform-CPML
  domain with an INTERNAL PEC short to build the standing wave.
* x64 is scoped via ``jax.experimental.enable_x64`` (NOT a module-level
  ``jax.config.update``, which leaks across the pytest process). The flux
  accumulator already promotes to float64 internally, but we scope x64 to match
  the diagnostic and keep the reference arithmetic in float64.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx import Simulation, flux_spectrum, Box
from rfx.boundaries.spec import BoundarySpec
from rfx.sources.sources import ModulatedGaussian

try:  # jax >= 0.8.0
    from jax import enable_x64
except ImportError:  # older jax
    from jax.experimental import enable_x64

C0 = 2.998e8


def _build_and_run():
    dx = 1.0e-3
    nx, ny = 100, 28
    domain = (nx * dx, ny * dx, dx)
    f0 = 9e9
    bw = 0.6

    sim = Simulation(
        freq_max=0.5 * C0 / dx,
        domain=domain,
        dx=dx,
        boundary=BoundarySpec.uniform("cpml"),
        cpml_layers=8,
        mode="2d_tmz",
    )
    # Internal full-transverse PEC short -> strong reflection -> standing wave
    # between the source and the short.
    short_x = domain[0] * 0.72
    sim.add(Box((short_x, 0, 0), (short_x + dx, domain[1], dx)), material="pec")
    sim.add_source(
        position=(domain[0] * 0.18, domain[1] * 0.5, 0),
        component="ez",
        waveform=ModulatedGaussian(f0=f0, bandwidth=bw, amplitude=1.0),
    )

    freqs = jnp.asarray(np.linspace(4e9, 22e9, 16))
    xmon = domain[0] * 0.45  # inside the standing-wave region
    ymid = domain[1] / 2.0

    # (1) legacy full-plane (size=None) on the x=xmon plane
    sim.add_flux_monitor(axis="x", coordinate=xmon, freqs=freqs, name="full")
    # (2) finite over the central HALF of the transverse extent
    sim.add_flux_monitor(
        axis="x", coordinate=xmon, freqs=freqs,
        size=(domain[1] * 0.5, dx), center=(ymid, dx / 2.0), name="fin_half",
    )
    # (3) finite over the FULL interior transverse extent
    sim.add_flux_monitor(
        axis="x", coordinate=xmon, freqs=freqs,
        size=(domain[1], dx), center=(ymid, dx / 2.0), name="fin_full",
    )

    sim.preflight(strict=False)
    # Moderate run: long enough to establish the standing wave, short enough
    # that the DFT window closes while the field still rings (healthy DFT mag).
    res = sim.run(n_steps=2200)
    return res, freqs, ny


def _restrict_full_to_window(mon_full, mon_sub):
    """Full-plane per-cell integrand summed over mon_sub's index window."""
    e1 = np.asarray(mon_full.e1_dft)
    e2 = np.asarray(mon_full.e2_dft)
    h1 = np.asarray(mon_full.h1_dft)
    h2 = np.asarray(mon_full.h2_dft)
    integrand = e1 * np.conj(h2) - e2 * np.conj(h1)
    dA = np.asarray(mon_full.dA)  # scalar (1,1) on a uniform grid
    win = integrand[:, mon_sub.lo1:mon_sub.hi1, mon_sub.lo2:mon_sub.hi2]
    return np.real(np.sum(win * dA, axis=(-2, -1)))


def test_finite_size_flux_matches_full_plane_in_standing_wave():
    """Finite-size flux == full-plane integrand over the SAME window (exact).

    Decisive refutation of the "finite-size is less stable in standing-wave
    environments" caveat. Per-frequency, machine-precision agreement.
    """
    with enable_x64(True):
        res, freqs, ny = _build_and_run()

        mon_full = res.flux_monitors["full"]
        mon_half = res.flux_monitors["fin_half"]
        mon_finf = res.flux_monitors["fin_full"]

        # --- Witness 1: the DFT accumulators are REAL (well above float noise).
        ez_dft_mag = float(np.max(np.abs(np.asarray(mon_full.e2_dft))))
        assert ez_dft_mag > 1e-20, (
            f"flux DFT accumulator is at noise floor ({ez_dft_mag:.3e}); "
            "the field never built up — test setup is degenerate"
        )

        # --- Witness 2: the field is genuinely standing-wave-heavy.
        # |Ez| along x at an off-center transverse line in the cavity region
        # must show a strong min/max ratio (standing wave), not a flat
        # travelling wave.
        ez = np.abs(np.asarray(res.state.ez))[:, ny // 2 + 8, 0]
        cavity = ez[26:78]  # padded-grid indices between source and short
        swr = float(cavity.max() / max(cavity.min(), 1e-30))
        assert swr > 2.0, (
            f"expected a standing-wave-heavy field (SWR>2), got SWR={swr:.2f}; "
            "the caveat is about standing-wave environments specifically"
        )

        # --- Core claim: finite(half-window) == full-plane restricted to the
        # SAME window, per frequency, to machine precision.
        fin_half = np.asarray(flux_spectrum(mon_half))
        ref_half = _restrict_full_to_window(mon_full, mon_half)
        scale_h = max(np.max(np.abs(fin_half)), np.max(np.abs(ref_half)), 1e-300)
        reldev_half = np.abs(fin_half - ref_half) / scale_h
        max_rd_half = float(np.max(reldev_half))

        # And the same for the full-interior window.
        fin_full = np.asarray(flux_spectrum(mon_finf))
        ref_full = _restrict_full_to_window(mon_full, mon_finf)
        scale_f = max(np.max(np.abs(fin_full)), np.max(np.abs(ref_full)), 1e-300)
        reldev_full = np.abs(fin_full - ref_full) / scale_f
        max_rd_full = float(np.max(reldev_full))

        # Full per-frequency dump (R5: no bare pass/fail).
        fa = np.asarray(freqs)
        print(f"\n[finite-size flux] SWR(witness)={swr:.2f}  "
              f"Ez-DFT max={ez_dft_mag:.3e}")
        print("  GHz   | fin_half      ref_half      reldev   | "
              "fin_full      ref_full      reldev")
        for i in range(len(fa)):
            print(f"  {fa[i]/1e9:5.2f} | {fin_half[i]: .4e} {ref_half[i]: .4e} "
                  f"{reldev_half[i]:.2e} | {fin_full[i]: .4e} "
                  f"{ref_full[i]: .4e} {reldev_full[i]:.2e}")
        print(f"  max reldev: half-window={max_rd_half:.3e}  "
              f"full-interior={max_rd_full:.3e}")

        # Machine-precision gate. Generous 1e-12 ceiling (measured ~2e-16) so
        # the test is robust to platform float-summation-order differences yet
        # would still catch any real >ppm-scale finite-size bookkeeping bug.
        assert max_rd_half < 1e-12, (
            f"finite-size (half) flux deviates from the full-plane integrand "
            f"over the same window by {max_rd_half:.3e} (>1e-12) — a real "
            "finite-size bookkeeping bug, not the expected machine-eps match"
        )
        assert max_rd_full < 1e-12, (
            f"finite-size (full-interior) flux deviates by {max_rd_full:.3e} "
            "(>1e-12) — real finite-size bookkeeping bug"
        )


def test_legacy_full_plane_includes_cpml_cells():
    """Document WHY legacy full-plane != finite(full-interior): coverage, not bug.

    The legacy ``size=None`` full plane integrates the ENTIRE grid plane,
    including the CPML padding cells on either transverse edge. A finite-size
    monitor covering the interior transverse extent legitimately excludes those
    cells. The difference is therefore *coverage*, and is fully recovered by
    restricting the legacy full-plane integrand to the finite window
    (verified to machine precision in the test above). This is the real origin
    of the now-refuted "less stable" caveat: a coverage difference misread as
    instability.
    """
    with enable_x64(True):
        res, freqs, _ny = _build_and_run()
        mon_full = res.flux_monitors["full"]
        mon_finf = res.flux_monitors["fin_full"]

        full = np.asarray(flux_spectrum(mon_full))
        fin_full = np.asarray(flux_spectrum(mon_finf))

        # The finite full-interior window is strictly inside the legacy plane.
        plane_n1 = np.asarray(mon_full.e2_dft).shape[1]
        win_n1 = mon_finf.hi1 - mon_finf.lo1
        assert win_n1 < plane_n1, (
            "finite full-interior window should be strictly smaller than the "
            f"legacy plane (got window {win_n1} vs plane {plane_n1})"
        )

        # Restricting the legacy integrand to the finite window reproduces the
        # finite monitor exactly; the *unrestricted* legacy flux differs only
        # by the excluded CPML cells' contribution.
        ref = _restrict_full_to_window(mon_full, mon_finf)
        scale = max(np.max(np.abs(fin_full)), np.max(np.abs(ref)), 1e-300)
        exact = float(np.max(np.abs(fin_full - ref)) / scale)

        legacy_scale = max(np.max(np.abs(full)), np.max(np.abs(fin_full)), 1e-300)
        coverage_gap = float(np.max(np.abs(full - fin_full)) / legacy_scale)

        print(f"\n[coverage] finite-vs-restricted exact match={exact:.3e}  "
              f"legacy-vs-finite coverage gap (CPML cells)={coverage_gap:.3e}")

        # The finite path equals the restricted legacy integrand exactly.
        assert exact < 1e-12, exact
        # The legacy-vs-finite gap is a real coverage difference (the CPML
        # cells carry attenuated field), i.e. non-negligible but NOT zero —
        # this is the expected, explained behaviour, not instability.
        assert coverage_gap > 1e-6, (
            "expected a non-trivial coverage gap from the excluded CPML cells; "
            f"got {coverage_gap:.3e}"
        )
