"""#636 M2 — the issue's unspent 2026-08-12 pre-declared factorial (ONE attempt).

Declared in docs/design_notes/i636_cpml_pole_pad_predeclaration.md (commit
841dcc2) BEFORE this script was first run. Running this battery consumes
the single physics attempt; its verdict is final for this session — no
tuning afterwards.

Battery (fixed knobs: dx=1mm, domain 45x39x12 mm, freq_max=7.5e9,
GaussianPulse(f0=3e9, bw=0.8) ez source at (15,13,5) mm, cpml_layers=12,
60,000 steps, float32 unless stated, skip_preflight=True [research script],
subpixel_smoothing=False):

  ON  = pole masks replicated into the pad like the statics (piggyback on
        extend_cpml_pad_materials, #627a fallback included) AND the CFS
        alpha rule alpha_max = 1.2*2*pi*f_top*eps0 (f_top = 7.5e9)
        monkeypatched into _cpml_profile (same (1-rho) grading).
  OFF = identical (alpha rule included), poles NOT extended.

  C1 both-face  eps_inf=4  Lorentz Q60   slab (0,0,3)-(45,39,7) mm
  C2 lo-only    eps_inf=4  Lorentz Q60   slab (0,0,3)-(30,26,7) mm
  C3 both-face  eps_inf=1  Lorentz Q5    (omega0=2pi*3e9, delta=omega0/10)
  C4 both-face  eps_inf=1  Drude         (omega_p=2pi*3e9, gamma=omega_p/100)
  C1-ON float64 control (precision="float64")

Observable: pad probes (absolute physical coordinates; pads live at
x<0 / x>44mm, y<0 / y>38mm): ez at face pads x-lo/x-hi/y-lo/y-hi, depths
2, 6, 10 cells, transverse mid, z=5mm; ez+ex+ey at the four x-y corner
pads, depth (6,6), z=5mm. Envelope = max|value| over each 200-step window
across (a) face probes, (b) corner probes, (c) all. Growth rate g =
least-squares slope of ln(envelope) vs step over the LAST 50% of windows.
Finiteness flag. A cell whose last-window envelope is below 1e-20 x its
peak envelope counts as fully decayed (reported g is then floor noise and
the cell counts as g <= 0). Free discriminators: face-vs-corner split,
final-state |E| localization, FFT peak of the longest-growing pad probe.

Vacuum floor: pure vacuum, same domain/layers/source, probe (36,19,5) mm,
4,000 steps; floor_dB = 20*log10(max|ez| over last 1000 steps / peak).
Compare shipped alpha vs rule alpha; degradation = floor_rule -
floor_shipped (positive = worse).

Two-sided falsifier F2 (verbatim from the note): fix viable ONLY if every
ON cell (C1..C4 + float64 control) has g <= 0 and stays finite AND vacuum
floor degradation <= 3 dB. Otherwise STOP to guards-only (preflight
advisory + re-baselined lock), no tuning.

Run:  .venv/bin/python validation/research/cpml_pole_pad/factorial_636.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

EPS_0 = 8.8541878128e-12

DX = 1e-3
NA, NB, NZ = 45, 39, 12
F0 = 3e9
W0 = 2 * np.pi * F0
FREQ_MAX = 2.5 * F0
LAYERS = 12
STEPS = 60000
WINDOW = 200
ALPHA_RULE = 1.2 * 2 * np.pi * FREQ_MAX * EPS_0  # ~0.5007 S/m


def patch_alpha(alpha_max):
    """Monkeypatch the alpha literal in _cpml_profile (0.05 shipped)."""
    import rfx.boundaries.cpml as cpml
    import numpy as _np

    if not hasattr(patch_alpha, "_orig"):
        patch_alpha._orig = cpml._cpml_profile

    orig = patch_alpha._orig

    def patched(n_layers, dt, dx, order=3, kappa_max=1.0, R_asymptotic=1e-15):
        p = orig(n_layers, dt, dx, order=order, kappa_max=kappa_max,
                 R_asymptotic=R_asymptotic)
        # Rebuild alpha/b/c with the requested alpha_max, float64, same
        # formulas as the shipped function.
        import jax.numpy as jnp
        sigma = _np.asarray(p.sigma, dtype=_np.float64)
        kappa = _np.asarray(p.kappa, dtype=_np.float64)
        rho = 1.0 - _np.arange(n_layers, dtype=_np.float64) / max(n_layers - 1, 1)
        alpha = alpha_max * (1.0 - rho)
        denom = sigma * kappa + kappa ** 2 * alpha
        b = _np.exp(-(sigma / kappa + alpha) * float(dt) / EPS_0)
        c = _np.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)
        return cpml.CPMLParams(
            sigma=jnp.asarray(sigma, dtype=jnp.float32),
            kappa=jnp.asarray(kappa, dtype=jnp.float32),
            alpha=jnp.asarray(alpha, dtype=jnp.float32),
            b=jnp.asarray(b, dtype=jnp.float32),
            c=jnp.asarray(c, dtype=jnp.float32),
        )

    cpml._cpml_profile = patched
    # init_cpml resolves _cpml_profile as a module global at call time.


def unpatch_alpha():
    import rfx.boundaries.cpml as cpml
    if hasattr(patch_alpha, "_orig"):
        cpml._cpml_profile = patch_alpha._orig


def make_pole_extended_class():
    import jax.numpy as jnp
    from rfx import Simulation
    from rfx.geometry.rasterize_grid import extend_cpml_pad_materials

    class PoleExtendedSim(Simulation):
        """Lorentz/Debye pole masks replicated into the CPML pads the same
        way the statics are (incl. the #627a hi-face fallback), by
        piggybacking on extend_cpml_pad_materials with mask+1 as a fake
        eps array. Identical to the premise-stage scout harness."""

        def _assemble_materials(self, grid, **kw):
            out = super()._assemble_materials(grid, **kw)
            materials, debye_spec, lorentz_spec, *rest = out

            def ext_masks(spec):
                if spec is None:
                    return None
                poles, masks = spec
                plx, phx = grid.pad_x_lo, grid.pad_x_hi
                ply, phy = grid.pad_y_lo, grid.pad_y_hi
                plz, phz = grid.pad_z_lo, grid.pad_z_hi
                new_masks = []
                for m in masks:
                    fake_eps = m.astype(jnp.float32) + 1.0
                    z = jnp.zeros_like(fake_eps)
                    o = jnp.ones_like(fake_eps)
                    e, _, _ = extend_cpml_pad_materials(
                        fake_eps, z, o, plx, phx, ply, phy, plz, phz)
                    new_masks.append(e > 1.5)
                return (poles, new_masks)

            return (materials, ext_masks(debye_spec),
                    ext_masks(lorentz_spec), *rest)

    return PoleExtendedSim


def pad_probe_positions():
    """(label, kind, position, component) — absolute physical coords."""
    z = 5.0 * DX
    probes = []
    for d in (2, 6, 10):
        probes.append((f"xlo_d{d}", "face", (-d * DX, (NB // 2) * DX, z), "ez"))
        probes.append((f"xhi_d{d}", "face", ((NA - 1 + d) * DX, (NB // 2) * DX, z), "ez"))
        probes.append((f"ylo_d{d}", "face", ((NA // 2) * DX, -d * DX, z), "ez"))
        probes.append((f"yhi_d{d}", "face", ((NA // 2) * DX, (NB - 1 + d) * DX, z), "ez"))
    corners = [("c_ll", (-6 * DX, -6 * DX)),
               ("c_hl", ((NA - 1 + 6) * DX, -6 * DX)),
               ("c_lh", (-6 * DX, (NB - 1 + 6) * DX)),
               ("c_hh", ((NA - 1 + 6) * DX, (NB - 1 + 6) * DX))]
    for name, (x, y) in corners:
        for comp in ("ez", "ex", "ey"):
            probes.append((f"{name}_{comp}", "corner", (x, y, z), comp))
    return probes


def build_sim(cls, cell, precision="float32"):
    from rfx import Box, GaussianPulse
    from rfx.materials.lorentz import LorentzPole, drude_pole

    sim = cls(freq_max=FREQ_MAX, domain=(NA * DX, NB * DX, NZ * DX),
              dx=DX, boundary="cpml", cpml_layers=LAYERS,
              precision=precision)
    if cell in ("C1", "C2"):
        sim.add_material("slab", eps_r=4.0,
                         lorentz_poles=[LorentzPole(omega_0=W0,
                                                    delta=W0 / 120.0,
                                                    kappa=3.0 * W0 ** 2)])
    elif cell == "C3":
        sim.add_material("slab", eps_r=1.0,
                         lorentz_poles=[LorentzPole(omega_0=W0,
                                                    delta=W0 / 10.0,
                                                    kappa=3.0 * W0 ** 2)])
    elif cell == "C4":
        sim.add_material("slab", eps_r=1.0,
                         lorentz_poles=[drude_pole(omega_p=W0,
                                                   gamma=W0 / 100.0)])
    else:
        raise ValueError(cell)

    if cell == "C2":
        sim.add(Box((0.0, 0.0, 3 * DX), (30 * DX, 26 * DX, 7 * DX)),
                material="slab")
    else:
        sim.add(Box((0.0, 0.0, 3 * DX), (NA * DX, NB * DX, 7 * DX)),
                material="slab")

    sim.add_source((NA * DX / 3, NB * DX / 3, 5.0 * DX), "ez",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8),
                   amplitude_kind="field")
    probes = pad_probe_positions()
    for _, _, pos, comp in probes:
        sim.add_probe(pos, comp)
    return sim, probes


def analyze(ts, probes, dt):
    """ts: (n_steps, n_probes). Returns per-kind growth rates + extras."""
    ts = np.asarray(ts, dtype=np.float64)
    n = ts.shape[0]
    nw = n // WINDOW
    kinds = {"face": [i for i, p in enumerate(probes) if p[1] == "face"],
             "corner": [i for i, p in enumerate(probes) if p[1] == "corner"],
             "all": list(range(len(probes)))}
    out = {}
    for kind, idx in kinds.items():
        env = np.abs(ts[:nw * WINDOW, idx]).reshape(nw, WINDOW, len(idx)).max(axis=(1, 2))
        peak = float(env.max())
        lastw = float(env[-1])
        tail = env[nw // 2:]
        steps = (np.arange(nw)[nw // 2:] + 0.5) * WINDOW
        lg = np.log(np.maximum(tail, 1e-300))
        A = np.vstack([steps, np.ones_like(steps)]).T
        slope = float(np.linalg.lstsq(A, lg, rcond=None)[0][0])
        decayed = bool(lastw < 1e-20 * max(peak, 1e-300))
        out[kind] = {"g_per_step": slope, "peak": peak, "last_window": lastw,
                     "decayed_to_floor": decayed}
    out["finite"] = bool(np.isfinite(ts).all())
    # FFT of the strongest-growth pad probe's tail (free diagnostic)
    pi = int(np.argmax(np.abs(ts[-WINDOW:]).max(axis=0)))
    tail_sig = ts[n // 2:, pi]
    if np.isfinite(tail_sig).all() and np.abs(tail_sig).max() > 0:
        sp = np.abs(np.fft.rfft(tail_sig * np.hanning(len(tail_sig))))
        fr = np.fft.rfftfreq(len(tail_sig), dt)
        out["fft_peak_hz"] = float(fr[int(np.argmax(sp))])
        out["fft_probe"] = probes[pi][0]
    return out


def localization(state, grid, cell_label):
    """|E| mass split interior / face-pad / corner-pad (x-y pads)."""
    ez = np.abs(np.asarray(state.ez, dtype=np.float64))
    ex = np.abs(np.asarray(state.ex, dtype=np.float64))
    ey = np.abs(np.asarray(state.ey, dtype=np.float64))
    e = ez + ex + ey
    nx, ny, nz = e.shape
    plx, phx = grid.pad_x_lo, grid.pad_x_hi
    ply, phy = grid.pad_y_lo, grid.pad_y_hi
    pad_x = np.zeros((nx, ny), dtype=bool)
    pad_x[:plx, :] = True
    pad_x[nx - phx:, :] = True
    pad_y = np.zeros((nx, ny), dtype=bool)
    pad_y[:, :ply] = True
    pad_y[:, ny - phy:] = True
    corner = pad_x & pad_y
    face = (pad_x | pad_y) & ~corner
    interior = ~(pad_x | pad_y)
    e2 = e.sum(axis=2)
    tot = e2.sum() + 1e-300
    imax = np.unravel_index(np.argmax(e2), e2.shape)
    return {"interior": float(e2[interior].sum() / tot),
            "face": float(e2[face].sum() / tot),
            "corner": float(e2[corner].sum() / tot),
            "argmax_ij": [int(imax[0]), int(imax[1])],
            "shape": [nx, ny, nz]}


def run_cell(cell, on, precision="float32"):
    from rfx import Simulation
    cls = make_pole_extended_class() if on else Simulation
    sim, probes = build_sim(cls, cell, precision)
    grid = sim._build_grid()
    _, debye_spec, lorentz_spec, *_ = sim._assemble_materials(grid)
    spec = lorentz_spec if lorentz_spec is not None else debye_spec
    _, masks = spec
    pole = np.asarray(masks[0])
    plx, phx = grid.pad_x_lo, grid.pad_x_hi
    ply, phy = grid.pad_y_lo, grid.pad_y_hi
    witness = {"pad_pole_cells": {
        "x_lo": int(pole[:plx].sum()), "x_hi": int(pole[-phx:].sum()),
        "y_lo": int(pole[:, :ply].sum()), "y_hi": int(pole[:, -phy:].sum())}}
    t0 = time.time()
    result = sim.run(n_steps=STEPS, compute_s_params=False,
                     skip_preflight=True, subpixel_smoothing=False)
    wall = time.time() - t0
    ts = np.asarray(result.time_series)
    rep = analyze(ts, probes, float(grid.dt))
    rep.update(witness)
    rep["wall_s"] = round(wall, 1)
    rep["localization"] = localization(result.state, grid,
                                       f"{cell}-{'ON' if on else 'OFF'}")
    return rep


def vacuum_floor(alpha_label, alpha_max):
    from rfx import Simulation, GaussianPulse
    unpatch_alpha()
    if alpha_max is not None:
        patch_alpha(alpha_max)
    sim = Simulation(freq_max=FREQ_MAX, domain=(NA * DX, NB * DX, NZ * DX),
                     dx=DX, boundary="cpml", cpml_layers=LAYERS)
    sim.add_source((NA * DX / 3, NB * DX / 3, 5.0 * DX), "ez",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8),
                   amplitude_kind="field")
    sim.add_probe((36 * DX, 19 * DX, 5.0 * DX), "ez")
    result = sim.run(n_steps=4000, compute_s_params=False,
                     skip_preflight=True, subpixel_smoothing=False)
    ts = np.abs(np.asarray(result.time_series, dtype=np.float64)).ravel()
    floor_db = 20 * np.log10(max(ts[-1000:].max(), 1e-300) / ts.max())
    unpatch_alpha()
    return float(floor_db)


RESULT_JSON = "validation/research/cpml_pole_pad/factorial_636_result.json"
F64_JSON = "validation/research/cpml_pole_pad/factorial_636_f64.json"


def main_f64():
    """C1-ON float64 control. Run in a DEDICATED process with
    JAX_ENABLE_X64=1 (precision='float64' fields need x64 enabled; the
    battery process stays x64-off so the float32 lanes are untouched)."""
    import jax
    assert jax.config.read("jax_enable_x64"), "run with JAX_ENABLE_X64=1"
    patch_alpha(ALPHA_RULE)
    try:
        rep = run_cell("C1", True, precision="float64")
    finally:
        unpatch_alpha()
    print(f"[C1-ON-f64] finite={rep['finite']} "
          f"g_all={rep['all']['g_per_step']:+.3e} "
          f"g_face={rep['face']['g_per_step']:+.3e} "
          f"g_corner={rep['corner']['g_per_step']:+.3e} "
          f"loc={rep['localization']} fft={rep.get('fft_peak_hz')} "
          f"wall={rep['wall_s']}s", flush=True)
    with open(F64_JSON, "w") as f:
        json.dump({"C1-ON-f64": rep}, f, indent=1)
    print(f"wrote {F64_JSON}")


def main_verdict():
    with open(RESULT_JSON) as f:
        report = json.load(f)
    with open(F64_JSON) as f:
        report.update(json.load(f))
    cells = ["C1", "C2", "C3", "C4"]
    on_tags = [f"{c}-ON" for c in cells] + ["C1-ON-f64"]
    all_stable = all(
        report[t]["finite"]
        and (report[t]["all"]["g_per_step"] <= 0.0
             or report[t]["all"]["decayed_to_floor"])
        for t in on_tags)
    degr = report["vacuum_floor_db"]["degradation_db"]
    vac_ok = degr <= 3.0
    verdict = "FIX-VIABLE" if (all_stable and vac_ok) else "GUARDS-ONLY"
    report["verdict"] = verdict
    print(f"VERDICT (two-sided falsifier F2, as declared): {verdict} "
          f"(all_ON_stable={all_stable}, vacuum_ok={vac_ok}, "
          f"degradation={degr:+.1f} dB)")
    with open(RESULT_JSON, "w") as f:
        json.dump(report, f, indent=1)


def main():
    report = {"steps": STEPS, "layers": LAYERS,
              "alpha_rule_S_per_m": ALPHA_RULE}

    print(f"alpha rule = {ALPHA_RULE:.4f} S/m (shipped literal 0.05)")

    # Vacuum floor first (short runs)
    f_ship = vacuum_floor("shipped", None)
    f_rule = vacuum_floor("rule", ALPHA_RULE)
    degr = f_rule - f_ship
    report["vacuum_floor_db"] = {"shipped": f_ship, "rule": f_rule,
                                 "degradation_db": degr}
    print(f"[vacuum] floor shipped = {f_ship:.1f} dB, rule = {f_rule:.1f} dB, "
          f"degradation = {degr:+.1f} dB", flush=True)

    cells = ["C1", "C2", "C3", "C4"]
    patch_alpha(ALPHA_RULE)
    try:
        for cell in cells:
            for on in (True, False):
                tag = f"{cell}-{'ON' if on else 'OFF'}"
                rep = run_cell(cell, on)
                report[tag] = rep
                print(f"[{tag}] finite={rep['finite']} "
                      f"g_all={rep['all']['g_per_step']:+.3e} "
                      f"g_face={rep['face']['g_per_step']:+.3e} "
                      f"g_corner={rep['corner']['g_per_step']:+.3e} "
                      f"decayed={rep['all']['decayed_to_floor']} "
                      f"pad_pole={rep['pad_pole_cells']} "
                      f"loc={rep['localization']} "
                      f"fft={rep.get('fft_peak_hz')} wall={rep['wall_s']}s",
                      flush=True)
    finally:
        unpatch_alpha()

    with open(RESULT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {RESULT_JSON} (run 'f64' then 'verdict' to finish)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "battery"
    if mode == "battery":
        main()
    elif mode == "f64":
        main_f64()
    elif mode == "verdict":
        main_verdict()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
