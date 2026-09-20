import json

import numpy as np

D = "scripts/diagnostics/_artifacts/cv03_seam_facet/"
prof = json.load(open(D + "profile.json"))["stages"]["profile"]["arms"]
print("== sigma at the FIRST pad cell (hi face, adjacent to the seam) ==")
for n in (20, 40, 60):
    u = prof[f"upml_{n}"]
    sE = np.asarray(u["sigma_E_centre_row"], dtype=float)
    phx = u["pad_x_hi"]
    first = sE[-phx]
    smax = sE.max()
    print(f"  upml N={n}: sigma_first={first:.4f} S/m  sigma_max={smax:.1f}"
          f"  first/max={first / smax:.3e}")
    c = prof[f"cpml_{n}"]
    sc = np.asarray(c["sigma"], dtype=float)
    print(f"  cpml N={n}: sigma_first={sc[-1]:.4f} S/m (profile is lo-face "
          f"order; [-1] is the interior edge)  sigma_max={sc.max():.1f}")
print("\n== ratios ==")
for fam in ("upml", "cpml"):
    v = []
    for n in (20, 40, 60):
        a = prof[f"{fam}_{n}"]
        if fam == "upml":
            sE = np.asarray(a["sigma_E_centre_row"], dtype=float)
            v.append(sE[-a["pad_x_hi"]])
        else:
            v.append(np.asarray(a["sigma"], dtype=float)[-2])
    print(f"  {fam}: {v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f}"
          f"  20/40={v[0] / v[1]:.3f}  40/60={v[1] / v[2]:.3f}"
          f"  20/60={v[0] / v[2]:.3f}   (N^-3 predicts 8 / 3.375 / 27)")

print("\n== exit codes and G1/G2 from the sweep arms ==")
sw = json.load(open(D + "sweep.json"))["stages"]["sweep"]
for k, v in sw.items():
    f = np.asarray(v["freqs_c_over_a"])
    mask = np.asarray(v["band_mask"], dtype=bool)
    nr = np.asarray(v["n_eff_rfx"])
    na = np.asarray(v["n_eff_analytic"])
    dev = np.max(np.abs(nr[mask] / na[mask] - 1.0)) * 100
    res = np.max(np.asarray(v["two_wave_rel_residual"])[mask])
    print(f"  {k}: exit={v['exit_code']}  G1 band-max dev={dev:.3f}% "
          f"(gate 2.0) resid={res:.4f} (gate 0.05)  "
          f"G2 band-mean T={v['T_rfx_band_mean']:.4f} (gate 0.95-1.05) "
          f"-> {'PASS' if 0.95 <= v['T_rfx_band_mean'] <= 1.05 else 'FAIL'}")

print("\n== band-mean T, B1 vs B2 ==")
bp = json.load(open(D + "bprime.json"))["stages"]["bprime"]
for n in (20, 40, 60):
    print(f"  N={n}: B1 guidecont T={bp[f'guidecont_upml_{n}']['T_rfx_band_mean']:.4f}"
          f"  B2 nosub T={bp[f'nosub_upml_{n}']['T_rfx_band_mean']:.4f}"
          f"  committed T={sw[f'sweep_upml_{n}']['T_rfx_band_mean']:.4f}")
