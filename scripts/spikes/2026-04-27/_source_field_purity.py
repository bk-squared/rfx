"""Phase 0 of port-extractor redesign — source-purity diagnostic.

Question: how much of the ~7% slab S11 floor comes from source-side
modal impurity vs extractor-side projection mismatch?

Test: empty waveguide, dump E_z(y,z) at port plane via dft_plane_probe;
fit to analytic TE10 mode shape sin(πy/a)·1; report complex-difference
|E_sim − E_func_analytic| after best amplitude/phase match.

Decision rule:
  max |Δ_normalised| < 0.01  → source is clean; Phase 1B not needed,
                                Phase 1A (extractor-only redesign) suffices.
  max |Δ_normalised| ≥ 0.01  → source has discrete modal mismatch;
                                Phase 1B (rebuild source with continuous
                                weighting) is required to reach Meep class.

Geometry: WR-90 (a=22.86 mm, b=10.16 mm), single port, no obstacles,
no PEC short. Probe at the port plane.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp
import importlib.util


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    cv = _load_cv11()
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS,
        dx=cv.DX_M,
    )
    port_freqs = jnp.asarray(freqs)
    sim.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050,
        name="left",
    )

    # DFT plane probe several cells DOWNSTREAM of the source so we
    # measure forward-propagating mode (CPML absorbs at the far end,
    # so no reflection contaminates).
    probe_x = cv.PORT_LEFT_X + 0.020  # 20 mm downstream
    sim.add_dft_plane_probe(
        axis="x", coordinate=probe_x, component="ez",
        freqs=port_freqs, name="ez_dn",
    )

    result = sim.run(num_periods=cv.NUM_PERIODS_LONG, compute_s_params=False)
    probe = result.dft_planes["ez_dn"]
    e_z = np.asarray(probe.accumulator)  # (n_freqs, ny, nz)
    print(f"E_z accumulator shape: {e_z.shape}")
    n_freqs, ny, nz = e_z.shape

    # CRITIC FIX (2026-04-27): rfx uses uniform cubic cells dx=DX_M=0.001 m.
    # Per rfx/grid.py:146-149 (+1 fence-post): ny = ceil(L/dx) + 1, so for
    # L=22.86 mm and dx=1 mm, ny=24 and effective aperture = (ny-1)*dx = 23 mm.
    # PEC nodes are at j=0 (y=0) and j=ny-1=23 (y=23*dx=23 mm). Both are
    # physical boundaries; yee.py:362-367 is correct.
    a_phys = cv.A_WG  # WR-90 22.86 mm (physical waveguide width)
    b_phys = cv.B_WG  # WR-90 10.16 mm
    dy = cv.DX_M
    dz = cv.DX_M
    a = (ny - 1) * dy  # effective aperture rfx solves on (23 mm)
    b = (nz - 1) * dz  # effective height (11 mm)
    print(f"  rfx-effective aperture: a={a*1e3:.3f} mm (vs WR-90 {a_phys*1e3:.3f}), b={b*1e3:.3f} mm (vs WR-90 {b_phys*1e3:.3f})")
    print(f"  geometry quantization: ({(a-a_phys)/a_phys*100:+.2f}%, {(b-b_phys)/b_phys*100:+.2f}%)")
    # Yee staggering for E_z: at integer y, integer x, half-integer z.
    # So along y the sample positions are y = j·dy (NOT (j+0.5)·dy).
    # PEC at y=0 forces E_z=0 at j=0; PEC at y=ny·dy forces E_z=0 at j=ny-1
    # if ny·dy = a (aperture-fitting). Here ny=24, dy=A_WG/24, so the
    # last sample j=23 is at y=23·dy = 21.91 mm, not at the boundary y=a.
    # Hmm — that's weird. Let's compute both conventions and report both.
    y_int = np.arange(ny) * dy
    y_centers = (np.arange(ny) + 0.5) * dy
    z_centers = (np.arange(nz) + 0.5) * dz
    print(f"  y_int (j·dy) range: [{y_int[0]*1e3:.3f}, {y_int[-1]*1e3:.3f}] mm")
    print(f"  y_centers range: [{y_centers[0]*1e3:.3f}, {y_centers[-1]*1e3:.3f}] mm")

    # Analytic TE10 mode shape (m=1, n=0): E_z = sin(πy/a). Uniform in z.
    # Try BOTH y conventions.
    e_z_analytic = np.sin(np.pi * y_int / a)[:, None] * np.ones(nz)[None, :]

    # Also compute the DISCRETE-Yee TE10 template (the one rfx's source
    # imprints). If sim matches discrete template within < 1%, the source
    # is "clean" in the discrete world — and the residual vs analytic
    # template above is the discrete-vs-continuous shape gap, which means
    # Phase 1A (analytic-only extractor) will NOT close the loop without
    # also rebuilding the source (Phase 1B).
    from rfx.sources.waveguide_port import _discrete_te_mode_profiles
    u_widths = np.full(ny, dy)
    v_widths = np.full(nz, dz)
    aperture_dA_full = (u_widths[:, None] * v_widths[None, :])
    ey_d, ez_d, hy_d, hz_d, _kc = _discrete_te_mode_profiles(
        a, b, 1, 0, u_widths, v_widths, aperture_dA=aperture_dA_full,
        h_offset=(0.5, 0.5),
    )
    # Use Ez component as the discrete template shape (already normalised).
    e_z_discrete = ez_d  # shape (ny, nz)

    print(f"\nFrequency-by-frequency complex-diff metrics:")
    print(f"{'f_GHz':>7} {'A_match':>10} {'phase_deg':>11} "
          f"{'|Δ|_max':>10} {'|Δ|_mean':>10} {'|Δ|_max/|A|':>14}")
    print("-" * 78)
    rows = []
    for f_idx in range(n_freqs):
        f = float(probe.freqs[f_idx])
        slice_ = e_z[f_idx]  # (ny, nz)
        # Best (complex amplitude, phase) match by overlap:
        # A = <E_sim, E_an> / <E_an, E_an>
        num = np.sum(slice_ * np.conj(e_z_analytic))
        den = np.sum(np.abs(e_z_analytic) ** 2)
        A = num / den if den > 0 else 0.0 + 0.0j
        residual = slice_ - A * e_z_analytic
        # Normalised by analytic peak amplitude (after multiplying by |A|)
        peak = np.max(np.abs(A * e_z_analytic))
        max_abs = float(np.max(np.abs(residual)))
        mean_abs = float(np.mean(np.abs(residual)))
        rel = max_abs / peak if peak > 0 else float("nan")
        print(f"{f/1e9:>7.2f} {abs(A):>10.4e} {np.degrees(np.angle(A)):>11.2f} "
              f"{max_abs:>10.4e} {mean_abs:>10.4e} {rel:>14.4f}")
        rows.append((f, abs(A), max_abs, mean_abs, rel))

    rels = [r[4] for r in rows if np.isfinite(r[4])]
    print(f"\n=== Verdict (vs ANALYTIC sin(πy/a)) ===")
    print(f"max |Δ|/|A_match| across {len(rels)} freqs: {max(rels):.4f}")
    print(f"mean |Δ|/|A_match|:                        {np.mean(rels):.4f}")

    # Same comparison vs DISCRETE-Yee template
    print(f"\n=== Verdict (vs DISCRETE-Yee TE10) ===")
    print(f"{'f_GHz':>7} {'A_match':>10} {'|Δ|_max/|peak|':>16}")
    print("-" * 40)
    rels_disc = []
    for f_idx in range(n_freqs):
        slice_ = e_z[f_idx]
        num = np.sum(slice_ * np.conj(e_z_discrete))
        den = np.sum(np.abs(e_z_discrete) ** 2)
        A = num / den if den > 0 else 0.0 + 0.0j
        residual = slice_ - A * e_z_discrete
        peak = np.max(np.abs(A * e_z_discrete))
        rel = float(np.max(np.abs(residual))) / peak if peak > 0 else float("nan")
        rels_disc.append(rel)
        print(f"{float(probe.freqs[f_idx])/1e9:>7.2f} {abs(A):>10.4e} {rel:>16.4f}")
    print(f"\nmax |Δ|/|peak| vs discrete: {max(rels_disc):.4f}")
    print(f"mean |Δ|/|peak| vs discrete: {np.mean(rels_disc):.4f}")

    print(f"\n=== Decision ===")
    if max(rels_disc) < 0.01:
        print("Sim matches DISCRETE template < 1% (source is discrete-Yee-clean).")
        print(f"vs ANALYTIC residual {max(rels):.4f} is the discrete-vs-continuous gap.")
        print("=> Phase 1A (analytic extractor only) CANNOT close past this gap.")
        print("=> Phase 1B (rebuild source with analytic weighting) is REQUIRED.")
    elif max(rels) < 0.01:
        print("Sim matches ANALYTIC < 1% (source is continuous-pure).")
        print("=> Phase 1A alone suffices.")
    else:
        print("Sim doesn't match either template within 1%.")
        print("=> Source may have noise/higher-mode content; investigate further.")

    # Detail: per-cell residual at one frequency (10 GHz) along y at z=b/2
    f_target_idx = int(np.argmin(np.abs(np.asarray(probe.freqs) - 10.0e9)))
    z_target_idx = nz // 2
    print(f"\n[detail at f={float(probe.freqs[f_target_idx])/1e9:.2f} GHz, z_idx={z_target_idx}, z={z_centers[z_target_idx]*1e3:.2f} mm]")
    A_f = np.sum(e_z[f_target_idx] * np.conj(e_z_analytic)) / np.sum(np.abs(e_z_analytic) ** 2)
    print(f"{'y_idx':>6} {'y_mm':>7} {'|sim|':>11} {'|analytic·A|':>14} {'|residual|':>11} {'rel%':>7}")
    print("-" * 62)
    for j in range(ny):
        sim_val = e_z[f_target_idx, j, z_target_idx]
        ana_val = A_f * e_z_analytic[j, z_target_idx]
        res = sim_val - ana_val
        peak = abs(A_f) * 1.0  # max of sin(πy/a) is 1
        rel = abs(res) / peak * 100 if peak > 0 else float("nan")
        print(f"{j:>6d} {y_centers[j]*1e3:>7.2f} {abs(sim_val):>11.4e} "
              f"{abs(ana_val):>14.4e} {abs(res):>11.4e} {rel:>7.2f}")


if __name__ == "__main__":
    main()
