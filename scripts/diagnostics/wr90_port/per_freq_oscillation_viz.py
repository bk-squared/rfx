"""Per-frequency |S11| oscillation visualization.

Goal: show concretely what "per-frequency oscillation" means in S-parameter
terms. For a PEC-short termination, |S11(f)| MUST be 1.0 at every frequency
(perfect reflector — all incident power returns). Any departure from a flat
unity line is the bug we're chasing.

Three traces on one plot:
  (A) Canonical DROP-both extractor (normalize=False, single-run wave
      decomposition). Mean ~0.999, per-freq spread ~0.05%. Production state.
  (B) Phase 1A.1 fix: KEEP-both + analytic Pozar template + analytic Z_TE +
      position-aware projection + Mechanism C (H * exp(+jβ·dx/2)). Mean
      ~1.000 but per-freq spread ~3-7% (the unidentified 6% residual).
  (C) Ideal: flat 1.0.

Output: PNG + a small HTML report, both for /share upload.
"""
from __future__ import annotations
import os
import sys
import importlib.util
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def _load_cv11():
    cv11_path = REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_canonical_drop_both(cv):
    """Production path: compute_waveguide_s_matrix + normalize=False."""
    f, s11, _ = cv.run_rfx_pec_short()
    return np.asarray(f), np.abs(np.asarray(s11))


def run_phase1a1_keep_both(cv, dx_m=1e-3):
    """KEEP-both + analytic + position-aware + E/H phase align (Mechanism C)."""
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    pec_short_x = cv.PEC_SHORT_X  # Meep/OpenEMS-aligned canonical (+45 mm OE)
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS,
        dx=dx_m,
    )
    sim.add(
        Box((pec_short_x, 0.0, 0.0),
            (pec_short_x + 2 * dx_m, cv.DOMAIN_Y, cv.DOMAIN_Z)),
        material="pec",
    )
    pf = jnp.asarray(freqs)
    sim.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050,
        name="left",
    )
    for comp in ("ez", "hy"):
        sim.add_dft_plane_probe(
            axis="x", coordinate=cv.PORT_LEFT_X, component=comp,
            freqs=pf, name=f"{comp}_port",
        )
    res = sim.run(num_periods=200, compute_s_params=False)

    e_z = np.transpose(np.asarray(res.dft_planes["ez_port"].accumulator), (1, 2, 0))
    h_y = np.transpose(np.asarray(res.dft_planes["hy_port"].accumulator), (1, 2, 0))
    ny, nz, n_freqs = e_z.shape
    a = (ny - 1) * dx_m
    y_int = np.arange(ny) * dx_m
    e_func = np.sin(np.pi * y_int / a)[:, None] * np.ones(nz)[None, :]
    h_func = e_func.copy()
    dA = dx_m * dx_m * np.ones((ny, nz))

    V = np.zeros(n_freqs, dtype=complex)
    I = np.zeros(n_freqs, dtype=complex)
    for k in range(n_freqs):
        V[k] = np.sum(e_z[:, :, k] * e_func * dA)
        I[k] = np.sum(h_y[:, :, k] * h_func * dA)

    f_hz = np.asarray(freqs)
    omega = 2 * np.pi * f_hz
    C0 = 2.998e8
    MU_0 = 1.2566370614e-6
    f_c = C0 / (2 * a)
    k0 = omega / C0
    kc = 2 * np.pi * f_c / C0
    beta = np.sqrt(np.maximum(k0**2 - kc**2, 0.0) + 0j)
    Z = omega * MU_0 / beta

    # Mechanism C: spatial half-cell phase between staggered E and H
    I_aligned = I * np.exp(+1j * beta * dx_m / 2)
    a_fwd = 0.5 * (V + I_aligned * Z)
    a_ref = V - a_fwd
    s11 = np.abs(a_ref / a_fwd)
    return f_hz, s11


def make_plot(f_hz, s11_canonical, s11_phase1a1, png_path):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)

    f_ghz = f_hz / 1e9

    ax = axes[0]
    ax.axhline(1.0, color="k", lw=1, ls="--", label="Ideal PEC-short |S11|=1.000", zorder=1)
    ax.plot(f_ghz, s11_canonical, "o-", color="#2a7", lw=1.6, ms=4,
            label=f"(A) Canonical DROP-both — mean={s11_canonical.mean():.4f}, "
                  f"spread={s11_canonical.max()-s11_canonical.min():.4f}")
    ax.plot(f_ghz, s11_phase1a1, "s-", color="#c52", lw=1.6, ms=4,
            label=f"(B) Phase 1A.1 (KEEP+E/H align) — mean={s11_phase1a1.mean():.4f}, "
                  f"spread={s11_phase1a1.max()-s11_phase1a1.min():.4f}")
    ax.set_ylabel("|S11(f)|", fontsize=11)
    ax.set_title("WR-90 PEC-short |S11(f)|: per-frequency oscillation\n"
                 "(Goal: flat unity. Departure = port-extractor bug.)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=1, ls="--", zorder=1)
    ax.plot(f_ghz, s11_canonical - 1.0, "o-", color="#2a7", lw=1.6, ms=4,
            label="(A) DROP-both: ~±0.05% (production)")
    ax.plot(f_ghz, s11_phase1a1 - 1.0, "s-", color="#c52", lw=1.6, ms=4,
            label="(B) Phase 1A.1: ~±3-7% (unidentified residual)")
    ax.set_xlabel("Frequency [GHz]", fontsize=11)
    ax.set_ylabel("|S11(f)| − 1.0", fontsize=11)
    ax.set_title("Departure from ideal — this IS the per-freq oscillation", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {png_path}")


def make_html(png_filename, summary_rows, html_path):
    rows_html = "\n".join(
        f"<tr><td>{name}</td><td>{mean:.5f}</td><td>{minv:.5f}</td>"
        f"<td>{maxv:.5f}</td><td>{maxv-minv:.5f}</td></tr>"
        for name, mean, minv, maxv in summary_rows
    )
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>WR-90 PEC-short per-frequency oscillation (rfx, 2026-04-28)</title>
<link rel="stylesheet" href="https://remilab.cnu.ac.kr/share/assets/remilab-base.css"/>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.55}}
h1,h2{{border-bottom:1px solid #ddd;padding-bottom:.25rem}}
table{{border-collapse:collapse;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:.35rem .7rem;text-align:right}}
th{{background:#f4f4f4}}
td:first-child{{text-align:left;font-family:ui-monospace,Menlo,monospace}}
img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
code{{background:#f4f4f4;padding:1px 5px;border-radius:3px}}
.callout{{background:#fff8e1;border-left:4px solid #f0b400;padding:.6rem 1rem;margin:1rem 0}}
</style></head><body>
<h1>What is the "per-frequency oscillation" exactly?</h1>
<p><b>Setup:</b> WR-90 rectangular waveguide (a=22.86mm, b=10.16mm), 8.2–12.4 GHz, 21 frequency points.
A PEC short is placed near the receive port. <b>Physically</b>, all incident TE10 power must return:
<code>|S11(f)| = 1.000</code> exactly, at every frequency. Any departure from a flat unity line
is a discrete-extractor artefact — the bug we are tracking.</p>

<div class="callout">
<b>Definition.</b> "Per-frequency oscillation" = the ripple of <code>|S11(f)|</code> over the
21-point sweep around its mean. Mean tells you how well the V/I integrals are <em>scaled</em>
on average; spread (max−min) tells you what's <em>frequency-dependent</em> and therefore not
absorbed by a simple normalization constant.
</div>

<h2>The plot</h2>
<img src="{png_filename}" alt="WR-90 PEC-short |S11(f)|"/>

<h2>The numbers</h2>
<table>
<tr><th>Configuration</th><th>mean |S11|</th><th>min</th><th>max</th><th>spread (max−min)</th></tr>
{rows_html}
</table>

<h2>Reading the plot</h2>
<ul>
<li><b>(A) Canonical DROP-both</b> (current production, <code>normalize=False</code>): mean
slightly below 1 (~0.999), but the <em>frequency-dependent</em> ripple is sub-1e-3. As an
S-parameter measurement, this is the right tradeoff at typical resolutions — it's Meep-class.</li>
<li><b>(B) Phase 1A.1</b> (KEEP-both + analytic Pozar TE10 templates + analytic Z_TE +
position-aware projection + Mechanism C, the half-cell E/H phase fix
<code>H·exp(+jβ·dx/2)</code>): mean is essentially 1.000, but per-frequency spread blows up
to ±3–7%. This is the unexplained residual. <b>From an S-parameter standpoint, this means
each frequency point sees a different effective port impedance / reference plane shift.</b></li>
<li><b>Ideal</b> (dashed black, |S11|=1): what OpenEMS RectWGPort produces (±0% spread).</li>
</ul>

<h2>Why it matters</h2>
<p>The mean offset can hide behind a global normalization. The <em>oscillation</em> cannot —
it implies one of:
(i) frequency-dependent error in V or I from the V/I overlap integrals (template shape,
aperture weighting, field staggering),
(ii) frequency-dependent error in the wave-decomposition impedance Z(f) used to split V/I
into forward/reflected,
(iii) a frequency-dependent reference-plane offset between V and I samples.</p>

<p>Six mechanisms have been tested and rejected (see commit chain
<code>ecdd845..a005524</code>): discrete vs analytic templates, analytic vs Yee Z, Yee
dispersion (predicts dx² scaling, observed dx¹), h_offset roll asymmetry, source/extractor
weighting symmetry, E/H DFT time-stagger correction. Mechanism C is real and accounts for
half the residual; the remaining ~6% is local-hypothesis-exhausted.</p>

<h2>Next step</h2>
<p>Cell-by-cell field comparison vs OpenEMS RectWGPort using HDF5 dumps of E_z and H_y at
the port plane. Whatever discrete computation step diverges first — that's the residual.</p>

<p style="color:#666;font-size:.9em;margin-top:2rem">
rfx commit chain: <code>ecdd845..a005524</code>. Generated 2026-04-28 by
<code>scripts/diagnostics/wr90_port/per_freq_oscillation_viz.py</code>.</p>
</body></html>
"""
    html_path.write_text(html)
    print(f"[html] saved {html_path}")


def main():
    cv = _load_cv11()

    print("[run A] canonical DROP-both (production)...", flush=True)
    f_hz, s11_a = run_canonical_drop_both(cv)

    print("[run B] Phase 1A.1 KEEP-both + Mechanism C...", flush=True)
    _, s11_b = run_phase1a1_keep_both(cv, dx_m=1e-3)

    summary = [
        ("(A) DROP-both canonical [production]",
         float(s11_a.mean()), float(s11_a.min()), float(s11_a.max())),
        ("(B) Phase 1A.1 KEEP+E/H align",
         float(s11_b.mean()), float(s11_b.min()), float(s11_b.max())),
        ("Ideal (PEC-short)", 1.0, 1.0, 1.0),
    ]

    print("\n=== Summary ===")
    print(f"{'config':<42}{'mean':>10}{'min':>10}{'max':>10}{'spread':>10}")
    for name, mean, mn, mx in summary:
        print(f"{name:<42}{mean:>10.5f}{mn:>10.5f}{mx:>10.5f}{mx-mn:>10.5f}")

    np.savez(OUT / "data.npz", f_hz=f_hz, s11_canonical=s11_a, s11_phase1a1=s11_b)
    png = OUT / "wr90_pec_short_per_freq_oscillation.png"
    html = OUT / "index.html"
    make_plot(f_hz, s11_a, s11_b, png)
    make_html(png.name, summary, html)
    print(f"\nartifacts in {OUT}")


if __name__ == "__main__":
    main()
