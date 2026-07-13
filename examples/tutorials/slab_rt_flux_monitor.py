"""Broadband R(f)/T(f) of a dielectric slab — the recommended flux-monitor path.

This is the runnable companion to ``docs/agent/recipe-rt-measurement.mdx``. It
demonstrates the *correct* way to measure broadband reflection and transmission
spectra in rfx:

    add_flux_monitor  +  two-run reference subtraction (field-level)

and deliberately avoids the documented footgun (``add_probe`` time series +
``np.fft.rfft``), which suffers a resolution-independent window-bias for
dispersive media.

Why two runs?
  Run 1 (reference, no slab) gives the incident flux. Run 2 (sample, with slab)
  gives the total flux. For *reflection* the scattered field is recovered by
  subtracting the reference E/H DFT accumulators from the sample ones *before*
  computing the Poynting flux (flux is bilinear in E and H, so subtracting the
  already-computed fluxes would leave cross terms — see recipe detail (a)).

The slab here is dispersionless (eps_r = 4), so an analytic Fresnel
transfer-matrix result is available as a sanity witness. The grid is kept coarse
for speed; halving ``dx`` reduces the error toward the analytic curve. Preflight
will flag the under-resolution and the lossless slab — both are expected here:
this is a fast method demo, we measure R/T (not Q), and a lossless slab is what
matches the lossless analytic Fresnel reference.

Public docs:
  - docs/agent/recipe-rt-measurement.mdx   (canonical two-run pattern)
  - docs/public/guide/probes-sparams.mdx   (flux monitors as non-port observables)
  - docs/public/api/results-observables.mdx (add_flux_monitor -> Result.flux_monitors)

Run:  python examples/slab_rt_flux_monitor.py
"""

import numpy as np

from rfx import Box, Simulation
from rfx.probes.probes import flux_spectrum

C0 = 299_792_458.0

# ----- problem definition (kept small/fast; refine dx for accuracy) -----
EPS_SLAB = 4.0
D_SLAB = 10.0e-3  # 10 mm slab
DX = 1.0e-3  # 1 mm cells (coarse demo)
FREQ_MAX = 16e9
F0 = 9e9
BW = 0.9

DOM_X = 200e-3  # propagation axis -> 200 cells in x
DOM_Y = 8e-3  # transverse (TFSF forces the transverse axes periodic)

FREQS_RT = np.linspace(4e9, 14e9, 21)  # R/T evaluation band

# slab centred in x; nudge the upper x-corner down by dx/2 so an inclusive-bounds
# Box maps to exactly D_SLAB of interior cells (recipe detail (d)).
CENTER_X = DOM_X / 2.0
SLAB_X_LO = CENTER_X - D_SLAB / 2.0
SLAB_X_HI = CENTER_X + D_SLAB / 2.0 - DX / 2.0

# flux planes: reflection monitor before the slab, transmission monitor after it
REFL_X = CENTER_X - 40e-3
TRANS_X = CENTER_X + 40e-3
Y_CENTER = DOM_Y / 2.0


def build_sim(with_slab: bool) -> Simulation:
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=(DOM_X, DOM_Y, DX),
        dx=DX,
        boundary="cpml",
        cpml_layers=10,
        mode="2d_tmz",
    )
    if with_slab:
        sim.add_material("slab", eps_r=EPS_SLAB)
        # The material Box must span the full transverse extent (incl. CPML
        # padding): with TFSF the transverse axes are periodic, so any cell
        # outside the material mask breaks the plane-wave assumption (detail (c)).
        sim.add(Box((SLAB_X_LO, -1.0, -1.0), (SLAB_X_HI, 1.0, 1.0)), material="slab")
    sim.add_tfsf_source(f0=F0, bandwidth=BW, polarization="ez", direction="+x")
    sim.add_flux_monitor(axis="x", coordinate=REFL_X, freqs=FREQS_RT, name="refl")
    sim.add_flux_monitor(axis="x", coordinate=TRANS_X, freqs=FREQS_RT, name="trans")
    return sim


def run_one(with_slab: bool):
    sim = build_sim(with_slab)
    # run() auto-runs preflight; advisories are printed, never suppressed.
    return sim.run(
        n_steps=4000,
        until_decay=1e-3,  # dispersionless slab -> 1e-3 is sufficient
        decay_monitor_component="ez",
        decay_monitor_position=(TRANS_X, Y_CENTER, 0.0),
    )


def fresnel_slab_rt(freqs, eps_r, d):
    """Analytic lossless-slab |r|^2, |t|^2 via the 1-D transfer matrix."""
    n = np.sqrt(eps_r)
    ra = np.zeros_like(freqs)
    ta = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
        delta = 2 * np.pi * f * n * d / C0
        cd, sd = np.cos(delta), np.sin(delta)
        m00, m01, m10, m11 = cd, 1j * sd / n, 1j * n * sd, cd
        r = (m00 + m01 - m10 - m11) / (m00 + m01 + m10 + m11)
        t = 2.0 / (m00 + m01 + m10 + m11)
        ra[i] = abs(r) ** 2
        ta[i] = abs(t) ** 2
    return ra, ta


def main():
    print("RUN 1: reference (no slab)")
    res_ref = run_one(with_slab=False)
    ref_refl_fm = res_ref.flux_monitors["refl"]
    ref_trans_flux = np.asarray(flux_spectrum(res_ref.flux_monitors["trans"]))

    print("RUN 2: sample (with slab)")
    res_slab = run_one(with_slab=True)
    slab_refl_fm = res_slab.flux_monitors["refl"]
    slab_trans_flux = np.asarray(flux_spectrum(res_slab.flux_monitors["trans"]))

    # Field-level subtraction for the reflected (scattered) flux — recipe detail (a).
    # FluxMonitor is a NamedTuple; the accumulated DFT fields are e1_dft, e2_dft,
    # h1_dft, h2_dft.
    scat_refl_fm = slab_refl_fm._replace(
        e1_dft=slab_refl_fm.e1_dft - ref_refl_fm.e1_dft,
        e2_dft=slab_refl_fm.e2_dft - ref_refl_fm.e2_dft,
        h1_dft=slab_refl_fm.h1_dft - ref_refl_fm.h1_dft,
        h2_dft=slab_refl_fm.h2_dft - ref_refl_fm.h2_dft,
    )
    scat_refl_flux = np.asarray(flux_spectrum(scat_refl_fm))

    transmittance = slab_trans_flux / ref_trans_flux
    reflectance = -scat_refl_flux / ref_trans_flux  # reflection travels -x

    r_an, t_an = fresnel_slab_rt(FREQS_RT, EPS_SLAB, D_SLAB)

    print("\n  f(GHz)    R       T      R+T   |   R_an    T_an")
    for i, f in enumerate(FREQS_RT):
        print(
            f"  {f / 1e9:5.1f}  {reflectance[i]:6.3f}  {transmittance[i]:6.3f}  "
            f"{reflectance[i] + transmittance[i]:5.3f}  |  {r_an[i]:6.3f}  {t_an[i]:6.3f}"
        )

    finite = np.all(np.isfinite(reflectance)) and np.all(np.isfinite(transmittance))
    print(f"\n  R+T max = {np.max(reflectance + transmittance):.3f} (energy: expect <= ~1)")
    print(f"  mean |T - T_analytic| = {np.mean(np.abs(transmittance - t_an)):.3f}")
    print(f"  mean |R - R_analytic| = {np.mean(np.abs(reflectance - r_an)):.3f}")
    print(f"  all finite: {finite}")


if __name__ == "__main__":
    main()
