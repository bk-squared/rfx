"""#636 M3 — growing-mode localization addendum (non-gating diagnostic).

Declared in docs/design_notes/i636_cpml_pole_pad_predeclaration.md (M3
section) BEFORE first run. Characterizes the KNOWN-unstable
configuration (the premise/scout config: C1 material+geometry, 8 CPML
layers, SHIPPED alpha 0.05, poles extended into the pad), 20,000 steps.
No verdict, no fix decision, no tuning depends on this measurement.

Prediction (informational, from the note): growing mode concentrated in
the x/y pads at slab z-levels; spectrum peaks near the eps(omega)=0
polariton band edge (3.97 GHz for this material).

Run:  .venv/bin/python validation/research/cpml_pole_pad/localize_636.py
"""

from __future__ import annotations

import json
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "factorial_636", os.path.join(_here, "factorial_636.py"))
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)


def main():
    # Shipped alpha (no patch), 8 layers, 20k steps — the premise config.
    fa.unpatch_alpha()
    fa.LAYERS = 8
    fa.STEPS = 20000

    # Harness fix (8-layer pads): the battery's probe depths (2, 6, 10)
    # assume 12-layer pads; depth 10 does not exist at 8 layers. Use
    # depths 2, 5, 7 and corner depth (5, 5) — same observable.
    DX, NA, NB = fa.DX, fa.NA, fa.NB

    def pad_probe_positions_8():
        z = 5.0 * DX
        probes = []
        for d in (2, 5, 7):
            probes.append((f"xlo_d{d}", "face", (-d * DX, (NB // 2) * DX, z), "ez"))
            probes.append((f"xhi_d{d}", "face", ((NA - 1 + d) * DX, (NB // 2) * DX, z), "ez"))
            probes.append((f"ylo_d{d}", "face", ((NA // 2) * DX, -d * DX, z), "ez"))
            probes.append((f"yhi_d{d}", "face", ((NA // 2) * DX, (NB - 1 + d) * DX, z), "ez"))
        corners = [("c_ll", (-5 * DX, -5 * DX)),
                   ("c_hl", ((NA - 1 + 5) * DX, -5 * DX)),
                   ("c_lh", (-5 * DX, (NB - 1 + 5) * DX)),
                   ("c_hh", ((NA - 1 + 5) * DX, (NB - 1 + 5) * DX))]
        for name, (x, y) in corners:
            for comp in ("ez", "ex", "ey"):
                probes.append((f"{name}_{comp}", "corner", (x, y, z), comp))
        return probes

    fa.pad_probe_positions = pad_probe_positions_8

    cls = fa.make_pole_extended_class()
    sim, probes = fa.build_sim(cls, "C1")
    grid = sim._build_grid()
    result = sim.run(n_steps=fa.STEPS, compute_s_params=False,
                     skip_preflight=True, subpixel_smoothing=False)
    ts = np.asarray(result.time_series)
    rep = fa.analyze(ts, probes, float(grid.dt))
    rep["localization"] = fa.localization(result.state, grid, "M3")

    # z-resolved witness: |E| mass per z-plane, split interior vs x/y pads.
    ez = np.abs(np.asarray(result.state.ez, dtype=np.float64))
    ex = np.abs(np.asarray(result.state.ex, dtype=np.float64))
    ey = np.abs(np.asarray(result.state.ey, dtype=np.float64))
    e = ez + ex + ey
    nx, ny, nz = e.shape
    plx, phx = grid.pad_x_lo, grid.pad_x_hi
    ply, phy = grid.pad_y_lo, grid.pad_y_hi
    plz = grid.pad_z_lo
    pad_xy = np.zeros((nx, ny), dtype=bool)
    pad_xy[:plx, :] = True
    pad_xy[nx - phx:, :] = True
    pad_xy[:, :ply] = True
    pad_xy[:, ny - phy:] = True
    prof_pad = e[pad_xy, :].sum(axis=0)
    prof_int = e[~pad_xy, :].sum(axis=0)
    rep["z_profile"] = {
        "pad_xy": [float(v) for v in prof_pad],
        "interior": [float(v) for v in prof_int],
        "pad_z_lo": int(plz),
        "slab_z_cells_interior_frame": [3, 6],
        "argmax_pad_z": int(np.argmax(prof_pad)),
    }

    print(f"[M3 C1-ext shipped-alpha L8 20k] finite={rep['finite']} "
          f"g_all={rep['all']['g_per_step']:+.3e} "
          f"g_face={rep['face']['g_per_step']:+.3e} "
          f"g_corner={rep['corner']['g_per_step']:+.3e} "
          f"loc={rep['localization']} "
          f"fft={rep.get('fft_peak_hz')} probe={rep.get('fft_probe')}")
    print(f"[M3] pad z-profile argmax at k={rep['z_profile']['argmax_pad_z']} "
          f"(pad_z_lo={plz}, slab occupies k={plz + 3}..{plz + 6}); "
          f"pad/interior mass = "
          f"{prof_pad.sum() / max(prof_int.sum(), 1e-300):.3f}")
    out = os.path.join(_here, "localize_636_result.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
