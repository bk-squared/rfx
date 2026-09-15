import json
import sys

import numpy as np

D = "scripts/diagnostics/_artifacts/cv03_far_end_return_831/"
stage = sys.argv[1] if len(sys.argv) > 1 else "sweep"
sw = json.load(open(D + f"{stage}.json"))["stages"][stage]
for k, v in sw.items():
    if "ez_abs_carrier_full_x" not in v:
        print(f"--- {k}: {v.get('status')}")
        continue
    ez = np.array(v["ez_abs_carrier_full_x"])
    nx = len(ez)
    plx, phx = v["pad_x_lo"], v["pad_x_hi"]
    seam_hi = nx - phx - 1
    inside = ez[plx:nx - phx]
    pad = ez[seam_hi:]
    print(f"--- {k}  nx={nx} pad={plx}/{phx} seam_hi_idx={seam_hi} "
          f"|B/A|={v['b_over_a_carrier_bin']:.4f} SWR={v['swr_fit_window']:.3f} "
          f"T={v['T_rfx_band_mean']:.4f} settling={v['settling_db']:.2f}dB")
    print(f"    interior |Ez| max={inside.max():.4e}  at seam={ez[seam_hi]:.4e}"
          f"  seam/max={ez[seam_hi] / inside.max():.4f}")
    depths = sorted({0, 1, 2, 5, 10, min(20, len(pad) - 1),
                     len(pad) // 2, len(pad) - 2, len(pad) - 1})
    print("    pad |Ez| by depth (cells):",
          {int(i): f"{pad[i]:.3e}" for i in depths})
    print(f"    A(1)={pad[1] / pad[0]:.4f} A(5)={pad[5] / pad[0]:.4f} "
          f"A(10)={pad[10] / pad[0]:.4f} A(end-1)={pad[-2] / pad[0]:.3e}")
