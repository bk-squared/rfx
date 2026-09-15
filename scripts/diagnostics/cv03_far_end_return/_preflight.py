import json
import sys

D = "scripts/diagnostics/_artifacts/cv03_far_end_return_831/"
stage, key = sys.argv[1], sys.argv[2]
v = json.load(open(D + f"{stage}.json"))["stages"][stage][key]
print("=== preflight stdout ===")
print(v["preflight_stdout"].strip())
print("=== preflight warnings ===")
for w in v["preflight_warnings"]:
    print("-", w)
print("=== run warnings ===")
for w in v["run_warnings"]:
    print("-", w)
print("=== settling ===", v["settling_db"])
print(json.dumps(v["settling_witness"], indent=2)[:900])
print("=== fit window ===", v["fit_window_meep_a"], v["fit_window_rfx_m"],
      v["fit_window_grid_index"], v["fit_window_n_samples"], "samples")
