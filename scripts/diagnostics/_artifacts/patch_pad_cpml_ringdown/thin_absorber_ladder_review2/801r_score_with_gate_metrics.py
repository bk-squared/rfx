"""Score PR #1077's OWN metric functions on the series I measured in the review."""
import sys
import numpy as np
sys.path.insert(0, "/root/801r_tree_pr1077/tests/oracle")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "gate", "/root/801r_tree_pr1077/tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py")
gate = importlib.util.module_from_spec(spec)
sys.modules["gate"] = gate
spec.loader.exec_module(gate)

ARMS = [("shipped rule (a3e4dba4 == main)", "/root/801r_out/801r_cpu_a3e4dba4_ts.npz"),
        ("MUTATED pre-#931 rule (main)", "/root/801r_out/801r_cpu_main_oldrule_ts.npz"),
        ("overhang cleared only", "/root/801r_out/801r_cpu_noverhang_ts.npz")]
print("bars: settling <= %.0f dB, worst rate < %.1e" % (gate.SETTLING_DB_BAR, gate.MAX_LOG_RATE_PER_STEP))
full = {}
for lab, f in ARMS:
    ts = np.asarray(np.load(f)["ts"])
    r = gate._late_time_log_rate_per_step(ts)
    s = gate._settling_db(ts)
    full[lab] = (r, s, ts)
    green = (s <= gate.SETTLING_DB_BAR) and (max(r) < gate.MAX_LOG_RATE_PER_STEP)
    red = (min(r) > 0.0) and (s > gate.SETTLING_DB_BAR)
    print("%-34s settling %8.2f  worst %+9.3e  best %+9.3e  -> GREEN=%-5s RED=%-5s"
          % (lab, s, max(r), min(r), green, red))

print()
print("Would a REDUCED record work?  (truncating the SAME n=4 series)")
print("%-10s %-22s %-22s %s" % ("periods", "shipped settling/rate", "mutated settling/rate", "gate/falsifier fire?"))
for periods in (40, 60, 80, 100, 120, 150):
    n = int(round(26659 * periods / 150.0))
    row = []
    for lab in ("shipped rule (a3e4dba4 == main)", "MUTATED pre-#931 rule (main)"):
        ts = full[lab][2][:n]
        r = gate._late_time_log_rate_per_step(ts)
        s = gate._settling_db(ts)
        row.append((s, max(r), min(r)))
    g = (row[0][0] <= gate.SETTLING_DB_BAR) and (row[0][1] < gate.MAX_LOG_RATE_PER_STEP)
    rd = (row[1][2] > 0.0) and (row[1][0] > gate.SETTLING_DB_BAR)
    print("%-10s %8.2f dB %+9.3e   %8.2f dB %+9.3e   green=%-5s red=%-5s  (%d steps)"
          % (periods, row[0][0], row[0][1], row[1][0], row[1][1], g, rd, n))
