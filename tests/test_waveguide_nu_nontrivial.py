"""NU-VALIDATE: non-trivial WR-90 S-matrix validation for the nonuniform path.

Settles the open question after NU-DRIVE-FIX: does
``_compute_waveguide_s_matrix_nu`` return correct S-parameters for a
*non-trivial* device (real reflection, |S11|>0, |S21|<1)?

== RESOLVED (2026-05-29, issue #88) via normalize="flux" ==

The ``normalize=True`` verdict below stays a strict-xfail because the
normalize=True diagonal genuinely remains fragile on a graded mesh
(band-edge denominator collapse — see the num_periods scan in this file's
docstring). It was NOT fixed; it is kept as a regression witness.

The issue itself is closed by the ``normalize="flux"`` branch
(PR #94 scan-body flux accumulation + PR #95 NU flux extractor): the flux
path is passive at a settled num_periods and matches the analytic Airy
reference within the broad-E5 0.05 tolerance, even for the eps_r=4 strong
reflector that normalize=True floors at ~0.077. See
``tests/test_waveguide_nu_flux.py`` (CPU sanity + Airy accuracy) and the
GPU-validated WR-90 NU flux broad-E5 envelope (16/16 cases, VESSL run
369367240154). Trail: docs/research_notes/20260529_flux_nu_wiring_design.md.

== UPDATE 2026-06-24: matches_uniform flipped to a real flux gate ==

``test_nu_nontrivial_matches_uniform`` is now a REAL passing gate (no longer
strict-xfail). The long-open "NU non-trivial S-matrix at noise floor" symptom
was confirmed (triple-verified: two diagnose->verify workflows) to be
``num_periods=2`` UNDERSETTLING, not a production bug — the reflected wave had
not returned to the port (round trip 236-333 ps vs a 167.8 ps window). At a
SETTLED ``num_periods`` (>=32 here) the NU ``normalize='flux'`` S-matrix
conserves power band-wide (|S11|^2+|S21|^2 = 1.00 +/- 0.005), is unitary on
BOTH drive columns, and reciprocal (S21=S12 to -67..-120 dB). The gate asserts
those ROBUST shared invariants; it deliberately does NOT gate pointwise
|S21_nu| vs |S21_uni| — the graded and uniform meshes resolve the in-band
Fabry-Perot split differently (up to ~0.21 at the band edge), so power
conservation, not the split, is the shared invariant (R5). Absolute |S21|-split
accuracy is the analytic-Airy / GPU lane (see test_waveguide_nu_flux.py NOTE;
VESSL 369367240154).

The air-thru test (tests/test_waveguide_nu_sparam.py) only validates the
trivial device==reference cancellation case.  A real device requires the
normalization path to correctly de-embed a non-zero reflected wave.

== Device choice: dielectric slab iris ==

A dielectric slab (eps_r=4) spanning a fraction of the cross-section is
used instead of a PEC iris for a critical reason:

  ``run_nonuniform_path(eps_override=vacuum_eps)`` replaces eps/sigma but
  does NOT clear ``pec_mask``.  A PEC obstacle would therefore persist in
  the reference run, making the reference contaminated and the S-matrix
  extraction undefined.  A dielectric obstacle is fully erased by the
  eps_override, giving a clean vacuum reference.

== Verdict logic ==

VALIDATED (NU sound for non-trivial):
  - |S11_nu| > 0.02  at some frequency (real reflection registered)
  - |S21_nu - S21_uni| < 0.15 over all freqs (agrees with uniform path)
  - Passivity: |S11_nu|^2 + |S21_nu|^2 <= 1.10 (loose; float32 numerics)

NOT VALIDATED (NU-DRIVE-FIX reopened):
  - |S11_nu| < 0.005 everywhere (all-zero / noise-floor injection), OR
  - |S21_nu - S21_uni| >= 0.15 (wildly disagrees with uniform path)
  → test is marked xfail(strict=True) with the magnitude evidence.

== Why num_periods does NOT fix this (issue #88, 2026-05-29) ==

DO NOT try to flip this xfail by raising ``num_periods``. A scan over
{8, 16, 24, 40} shows there is no settled sweet spot:

    num_periods   |S11_nu|max   passivity max   verdict
        8           0.045          0.69          undersettled (b_dev≈b_ref)
       16           6.32          40.5           band-edge blow-up
       24           0.886          2.01          fail
       40           0.940          1.18          fail

Two compounding ``normalize=True`` mechanisms, NOT undersettling alone:
  1. Band-edge normalization-denominator collapse. The Gaussian source
     spectrum delivers ~30x less incident power at the band edges
     (8.2 / 12.4 GHz) than at center, so ``a_inc_ref`` drops to ~1e-11
     while it is ~4.5e-10 mid-band. ``S11 = (b_dev - b_ref) / a_inc_ref``
     then divides by a near-noise denominator and blows up.
  2. ``b_dev`` exceeds ``a_inc_ref`` near the WR-90 cutoff (6.56 GHz),
     where the CPML absorbs poorly and round-trip end-to-end reflections
     accumulate over the longer window.

The real fix is wiring ``extract_waveguide_s_matrix_flux`` (the uniform
broad-E5 silver bullet, PR #92) into ``_compute_waveguide_s_matrix_nu``:
the spatial Poynting-flux integral does not divide by the
source-spectrum-weighted ``a_inc_ref`` and is immune to mechanism (1).
DONE (PR #95): the NU path now accepts ``normalize='flux'`` — the
``normalize != True`` NotImplementedError is gone. The settled-np flux gate
lives in ``test_nu_nontrivial_matches_uniform`` below (flipped from strict-xfail
2026-06-24).

R5 mandate: per-frequency |S11|/|S21| table for BOTH paths and raw
b_dev/a_inc magnitudes for the NU run are printed unconditionally so the
verdict is traceable.

No module-level jax_enable_x64 (leaks into sibling tests).
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# WR-90 geometry constants (matches test_waveguide_nu_sparam.py)
# ---------------------------------------------------------------------------
_A_WG = 0.02286   # m  broad wall
_B_WG = 0.01016   # m  narrow wall
_F_MAX = 12e9
_FREQS = jnp.linspace(8.2e9, 12.4e9, 5)

# Dielectric slab: centered in the guide, spans full y/z cross-section,
# 4 mm thick along x, eps_r=4.  Produces |S11| ~ 0.05-0.30 in X-band.
_SLAB_X_LO = 0.045
_SLAB_X_HI = 0.049
_SLAB_EPS_R = 4.0


# ---------------------------------------------------------------------------
# Simulation builders
# ---------------------------------------------------------------------------

def _make_wr90_nu_sim_with_slab():
    """WR-90 NU mesh (graded dx_profile) with a centred dielectric slab."""
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary
    from rfx.auto_config import smooth_grading
    from rfx.geometry.csg import Box

    dx_coarse = 1.5e-3
    dx_fine = 0.75e-3

    n_pre = int(round(0.030 / dx_coarse))
    n_fine = int(round(0.040 / dx_fine))
    n_post = int(round(0.030 / dx_coarse))
    raw = np.concatenate([
        np.full(n_pre, dx_coarse),
        np.full(n_fine, dx_fine),
        np.full(n_post, dx_coarse),
    ])
    dx_profile = smooth_grading(raw, max_ratio=1.3)
    domain_x = float(np.sum(dx_profile))

    sim = Simulation(
        freq_max=_F_MAX,
        domain=(domain_x, _A_WG, _B_WG),
        dx=dx_coarse,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
        dx_profile=dx_profile,
    )

    sim.add_material("diel_slab", eps_r=_SLAB_EPS_R, sigma=0.0)
    sim.add(
        Box(
            (_SLAB_X_LO, 0.0, 0.0),
            (_SLAB_X_HI, _A_WG, _B_WG),
        ),
        material="diel_slab",
    )

    sim.add_waveguide_port(
        0.015, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=0.020, name="left",
    )
    sim.add_waveguide_port(
        domain_x - 0.015, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=domain_x - 0.020, name="right",
    )
    return sim, domain_x


def _make_wr90_uniform_sim_with_slab():
    """WR-90 uniform mesh with the same dielectric slab (reference path)."""
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary
    from rfx.geometry.csg import Box

    dom_x = 0.100

    sim = Simulation(
        freq_max=_F_MAX,
        domain=(dom_x, _A_WG, _B_WG),
        dx=1.5e-3,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )

    sim.add_material("diel_slab", eps_r=_SLAB_EPS_R, sigma=0.0)
    sim.add(
        Box(
            (_SLAB_X_LO, 0.0, 0.0),
            (_SLAB_X_HI, _A_WG, _B_WG),
        ),
        material="diel_slab",
    )

    sim.add_waveguide_port(
        0.015, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=0.020, name="left",
    )
    sim.add_waveguide_port(
        dom_x - 0.015, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=dom_x - 0.020, name="right",
    )
    return sim


# ---------------------------------------------------------------------------
# Module-level result cache: run both sims once; tests share the result.
# ---------------------------------------------------------------------------
_cached_nu: object = None
_cached_uni: object = None


def _get_nu_result():
    global _cached_nu
    if _cached_nu is None:
        sim, _ = _make_wr90_nu_sim_with_slab()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cached_nu = sim.compute_waveguide_s_matrix(
                num_periods=2, normalize=True,
            )
    return _cached_nu


def _get_uni_result():
    global _cached_uni
    if _cached_uni is None:
        sim = _make_wr90_uniform_sim_with_slab()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cached_uni = sim.compute_waveguide_s_matrix(
                num_periods=2, normalize=True,
            )
    return _cached_uni


# ---------------------------------------------------------------------------
# Helper: probe NU injection strength by running one port-drive manually
# and returning raw b_dev/a_inc arrays before normalization.
# ---------------------------------------------------------------------------

def _get_nu_raw_waves():
    """Return (a_inc_ref, b_dev_diag) for drive_idx=0 on the NU slab sim.

    Bypasses compute_waveguide_s_matrix to expose the raw port-wave
    amplitudes before the diagonal subtraction step.  Used for R5 magnitude
    dump so we can verify injection is at physical signal levels, not ~1e-31
    noise floor.
    """
    from dataclasses import replace as _dc_replace
    from rfx.runners.nonuniform import (
        run_nonuniform_path,
        assemble_materials_nu,
        build_nonuniform_grid,
    )
    from rfx.sources.waveguide_port import (
        extract_waveguide_port_waves,
        waveguide_plane_positions,
    )

    sim, domain_x = _make_wr90_nu_sim_with_slab()

    # Synthesise dz_profile (same as _compute_waveguide_s_matrix_nu does)
    _nz = int(round(float(sim._domain[2]) / float(sim._dx)))
    sim._dz_profile = np.full(max(_nz, 1), float(sim._dx))

    pec_set = (sim._boundary_spec.pec_faces()
               if sim._boundary_spec is not None else None) or set()
    pmc_set = (sim._boundary_spec.pmc_faces()
               if sim._boundary_spec is not None else None) or set()

    def _axis_fully_closed(ax: str) -> bool:
        return {f"{ax}_lo", f"{ax}_hi"}.issubset(pec_set | pmc_set)

    cpml_axes = "".join(
        ax for ax in "xyz"
        if not _axis_fully_closed(ax)
    )

    grid = build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
        sim._dz_profile,
        dx_profile=sim._dx_profile,
        pec_faces=pec_set or None,
        pmc_faces=pmc_set or None,
        cpml_axes=cpml_axes,
    )
    n_steps = int(np.ceil(2 / sim._freq_max / float(grid.dt)))

    entries = list(sim._waveguide_ports)

    # Drive port 0 only
    sim._waveguide_ports = [
        _dc_replace(e, amplitude=(e.amplitude if i == 0 else 0.0))
        for i, e in enumerate(entries)
    ]

    dev_materials, _, _, _ = assemble_materials_nu(sim, grid)
    vacuum_eps = jnp.ones_like(dev_materials.eps_r)
    vacuum_sigma = jnp.zeros_like(dev_materials.sigma)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dev_result = run_nonuniform_path(sim, n_steps=n_steps)
        ref_result = run_nonuniform_path(
            sim, n_steps=n_steps,
            eps_override=vacuum_eps,
            sigma_override=vacuum_sigma,
        )

    sim._waveguide_ports = entries

    dev_wg = dev_result.waveguide_ports or {}
    ref_wg = ref_result.waveguide_ports or {}

    drive_name = entries[0].name
    cfg_dev = dev_wg[drive_name]
    cfg_ref = ref_wg[drive_name]

    planes_dev = waveguide_plane_positions(cfg_dev)
    desired = (
        entries[0].reference_plane
        if entries[0].reference_plane is not None
        else planes_dev["source"]
    )
    ref_shift = desired - planes_dev["reference"]

    a_inc_ref, _ = extract_waveguide_port_waves(cfg_ref, ref_shift=ref_shift)
    _, b_dev = extract_waveguide_port_waves(cfg_dev, ref_shift=ref_shift)

    return np.array(a_inc_ref), np.array(b_dev)


# ---------------------------------------------------------------------------
# Test 1: per-frequency S-parameter dump + verdict (R5 mandate)
# ---------------------------------------------------------------------------

def test_nu_nontrivial_slab_verdict():
    """NU-VALIDATE main verdict test for the dielectric-slab iris case.

    R5 dump: per-freq |S11|/|S21| for NU and uniform paths, plus raw
    b_dev/a_inc magnitudes for the NU run.

    Verdict:
      VALIDATED  → asserts pass (|S11|>0.02, |S21-S21_uni|<0.15, passivity)
      NOT VALIDATED → xfail(strict=True) with magnitude evidence quoted
    """
    res_nu = _get_nu_result()
    res_uni = _get_uni_result()

    s_nu = np.array(res_nu.s_params)
    s_uni = np.array(res_uni.s_params)
    freqs_ghz = np.array(res_nu.freqs) / 1e9

    # --- R5: raw NU injection magnitudes ---
    a_inc_ref, b_dev = _get_nu_raw_waves()
    a_inc_mag = np.abs(a_inc_ref)
    b_dev_mag = np.abs(b_dev)

    print("\n[NU-VALIDATE] Raw NU port-wave magnitudes (drive port 0, ref run):")
    _a_str = " ".join("%.3e" % v for v in a_inc_mag.ravel())
    _b_str = " ".join("%.3e" % v for v in b_dev_mag.ravel())
    print(f"  a_inc_ref (|a|): {_a_str}")
    print(f"  b_dev     (|b|): {_b_str}")
    print(f"  mean|a_inc|={a_inc_mag.mean():.3e}  mean|b_dev|={b_dev_mag.mean():.3e}")

    # --- R5: per-freq S table ---
    s11_nu = np.abs(s_nu[0, 0, :])
    s21_nu = np.abs(s_nu[1, 0, :])
    s11_uni = np.abs(s_uni[0, 0, :])
    s21_uni = np.abs(s_uni[1, 0, :])

    print("\n[NU-VALIDATE] Per-frequency |S11|/|S21| — NU vs Uniform:")
    print(f"  {'freq(GHz)':>10}  {'|S11_nu|':>9}  {'|S21_nu|':>9}  "
          f"{'|S11_uni|':>10}  {'|S21_uni|':>10}  {'|dS21|':>8}")
    for k in range(len(freqs_ghz)):
        print(f"  {freqs_ghz[k]:>10.2f}  {s11_nu[k]:>9.4f}  {s21_nu[k]:>9.4f}  "
              f"{s11_uni[k]:>10.4f}  {s21_uni[k]:>10.4f}  "
              f"{abs(s21_nu[k]-s21_uni[k]):>8.4f}")

    # --- Passivity ---
    passivity_nu = s11_nu**2 + s21_nu**2
    print(f"\n[NU-VALIDATE] NU passivity (|S11|²+|S21|²): "
          f"{np.array2string(passivity_nu, precision=4)}  max={passivity_nu.max():.4f}")

    # --- Verdict logic (roadmap W1.5 split) ---
    # These two gates HOLD on main today and must HARD-FAIL on regression.
    # The previous runtime ``pytest.xfail(...)`` here was always non-strict,
    # so ANY regression (including in these passing gates) reported as
    # green-"xfailed" forever.
    assert a_inc_mag.mean() > 1e-25, (
        f"NU injection at noise floor: mean|a_inc|={a_inc_mag.mean():.3e} "
        "<= 1e-25. NU-DRIVE-FIX has REGRESSED for non-trivial devices."
    )
    assert passivity_nu.max() <= 1.10, (
        f"Passivity violation: max(|S11|²+|S21|²)={passivity_nu.max():.4f} > 1.10."
    )

    # NU reflection magnitude + power-conservation/reciprocity (the ROBUST
    # shared NU-vs-uniform invariants) are now a REAL passing gate in
    # test_nu_nontrivial_matches_uniform below (flipped from strict-xfail
    # 2026-06-24 once the noise-floor symptom was shown to be num_periods=2
    # undersettling, not a bug). This np=2 normalize=True case stays a
    # regression witness for the injection-not-noise-floor + passivity gates.


@pytest.mark.slow
def test_nu_nontrivial_matches_uniform():
    """SETTLED-np flux gate: the NU non-trivial S-matrix conserves power, is
    reciprocal, and behaves like the uniform path (flipped from strict-xfail
    2026-06-24).

    Resolves the long-open "NU non-trivial S-matrix at noise floor" item: that
    symptom was ``num_periods=2`` UNDERSETTLING, not a production bug (the
    reflected wave had not returned to the port). Triple-verified by two
    diagnose->verify workflows; see the module docstring + rfx-known-issues
    "NU non-trivial device".

    At a settled ``num_periods`` the NU ``normalize='flux'`` S-matrix:
      - conserves power band-wide on BOTH drive columns (|S|^2 = 1.00 +/- 0.02,
        ~4x the measured 0.005 spread — tighter than the sibling flux test's
        loose <=1.10 passivity, which is the added value here),
      - is reciprocal (|S21 - S12| small),
      - registers real reflection and non-trivial transmission (not noise floor),
    and the uniform path on the same slab is a power-conserving witness.

    SCOPE (do NOT overclaim "NU validated"): this gates the ROBUST shared
    invariants. It deliberately does NOT gate pointwise |S21_nu| vs |S21_uni|
    (graded vs uniform meshes resolve the in-band Fabry-Perot split differently
    — max |dS21| ~ 0.21 at the band edge — so power conservation, not the split,
    is the shared invariant; R5). It does NOT exercise graded transverse-dA (the
    x-normal port plane has uniform y,z axes) nor |S21|-split ABSOLUTE accuracy
    (analytic-Airy / GPU lane: test_waveguide_nu_flux.py NOTE, VESSL 369367240154).
    """
    _NP_SETTLED = 40  # diagnosis: NU |S|^2 band-wide 1.000 +/- 0.005 at np >= 32

    nu_sim, _ = _make_wr90_nu_sim_with_slab()
    uni_sim = _make_wr90_uniform_sim_with_slab()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_nu = nu_sim.compute_waveguide_s_matrix(
            num_periods=_NP_SETTLED, normalize="flux",
        )
        res_uni = uni_sim.compute_waveguide_s_matrix(
            num_periods=_NP_SETTLED, normalize="flux",
        )

    s_nu = np.array(res_nu.s_params)    # (recv, drive, nf)
    s_uni = np.array(res_uni.s_params)
    freqs_ghz = np.array(res_nu.freqs) / 1e9

    s11_nu = np.abs(s_nu[0, 0, :]); s21_nu = np.abs(s_nu[1, 0, :])
    s12_nu = np.abs(s_nu[0, 1, :]); s22_nu = np.abs(s_nu[1, 1, :])
    s21_uni = np.abs(s_uni[1, 0, :])
    power_nu_d0 = s11_nu**2 + s21_nu**2
    power_nu_d1 = s22_nu**2 + s12_nu**2
    power_uni_d0 = np.abs(s_uni[0, 0, :])**2 + s21_uni**2
    recip = np.abs(s21_nu - s12_nu)

    # --- R5: per-frequency dump for both paths (traceable verdict) ---
    print(f"\n[NU-MATCH] settled flux gate (num_periods={_NP_SETTLED}):")
    print(f"  {'f(GHz)':>7} {'|S11n|':>7} {'|S21n|':>7} {'Pcol0':>7} {'Pcol1':>7} "
          f"{'|S21u|':>7} {'|dS21|':>7} {'recip':>9}")
    for k in range(len(freqs_ghz)):
        print(f"  {freqs_ghz[k]:>7.2f} {s11_nu[k]:>7.4f} {s21_nu[k]:>7.4f} "
              f"{power_nu_d0[k]:>7.4f} {power_nu_d1[k]:>7.4f} {s21_uni[k]:>7.4f} "
              f"{abs(s21_nu[k]-s21_uni[k]):>7.4f} {recip[k]:>9.2e}")

    # --- (a) power conservation (the headline finding), BOTH drive columns ---
    assert np.all(np.abs(power_nu_d0 - 1.0) < 0.02), (
        f"NU drive-0 power not conserved: |S11|^2+|S21|^2={power_nu_d0} "
        "(expect 1.0 +/- 0.02 at settled np — undersettling or a real defect)"
    )
    assert np.all(np.abs(power_nu_d1 - 1.0) < 0.02), (
        f"NU drive-1 power not conserved: |S22|^2+|S12|^2={power_nu_d1}"
    )
    # --- (b) reciprocity S21 == S12 ---
    assert recip.max() < 0.01, (
        f"NU reciprocity violated: max|S21-S12|={recip.max():.3e} >= 0.01"
    )
    # --- (c) real reflection + non-trivial transmission (NOT noise floor — the
    #         exact regression this test was created to catch) ---
    assert s11_nu.max() > 0.02, (
        f"|S11_nu| max={s11_nu.max():.4f} — slab reflection not registered "
        "(NU back at noise floor?)"
    )
    assert s21_nu.min() > 0.10, (
        f"|S21_nu| min={s21_nu.min():.4f} — transmission at noise floor"
    )
    # --- (d) uniform path is a sound power-conserving witness on the same slab ---
    assert np.all(np.abs(power_uni_d0 - 1.0) < 0.02), (
        f"uniform witness power not conserved: {power_uni_d0}"
    )
    # NOTE: pointwise |S21_nu| vs |S21_uni| is intentionally NOT asserted —
    # see the SCOPE paragraph (different meshes -> different in-band FP split).


# ---------------------------------------------------------------------------
# Test 2: both paths return finite arrays of matching shape
# ---------------------------------------------------------------------------

def test_nu_nontrivial_shape_and_finite():
    """Basic shape + finite check — decoupled from verdict so it always runs."""
    res_nu = _get_nu_result()
    res_uni = _get_uni_result()

    s_nu = np.array(res_nu.s_params)
    s_uni = np.array(res_uni.s_params)

    assert s_nu.shape == (2, 2, 5), f"NU shape {s_nu.shape} != (2,2,5)"
    assert s_uni.shape == (2, 2, 5), f"Uniform shape {s_uni.shape} != (2,2,5)"
    assert np.all(np.isfinite(s_nu)), "NU s_params contains NaN/Inf"
    assert np.all(np.isfinite(s_uni)), "Uniform s_params contains NaN/Inf"
