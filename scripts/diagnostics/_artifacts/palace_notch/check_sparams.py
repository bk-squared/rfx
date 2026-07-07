#!/usr/bin/env python3
"""Palace port-S.csv gate/summary for the cv06b notch referee (WP 1-B).

Usage:
  python3 check_sparams.py postpro/notch_probe_4090/port-S.csv --gate     # passivity gate (exit 3 on fail)
  python3 check_sparams.py postpro/notch_full_4090/port-S.csv --summary   # notch verdict summary
"""
import csv
import math
import sys


def load(path):
    rows = list(csv.reader(open(path)))
    hdr = [h.strip() for h in rows[0]]

    def col(sub):
        return next(i for i, h in enumerate(hdr) if sub in h)

    fi, s11i, s21i = col("f (GHz)"), col("|S[1][1]|"), col("|S[2][1]|")
    data = []
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        f = float(r[fi])
        s11 = 10 ** (float(r[s11i]) / 20)
        s21 = 10 ** (float(r[s21i]) / 20)
        data.append((f, s11, s21))
    return data


def main():
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--summary"
    data = load(path)
    worst = max(s1 * s1 + s2 * s2 for _, s1, s2 in data)
    if mode == "--gate":
        for f, s1, s2 in data:
            print("  f=%.2f GHz |S11|=%.4f |S21|=%.4f sum=%.4f" % (f, s1, s2, s1 * s1 + s2 * s2))
        print("PROBE max(|S11|^2+|S21|^2) = %.4f" % worst)
        if worst > 1.02:
            print("REFEREE-INVALID: probe violates passivity (>1.02) -- STOP per WP 1-B falsifier")
            sys.exit(3)
        print("PASSIVITY GATE PASS")
        return
    fn, s11n, s21n = min(data, key=lambda t: t[2])
    print("FULL: n=%d bins, max(|S11|^2+|S21|^2)=%.4f" % (len(data), worst))
    print("NOTCH (argmin |S21|): f=%.4f GHz, |S21|=%.5f (%.1f dB), |S11|=%.4f"
          % (fn, s21n, 20 * math.log10(max(s21n, 1e-12)), s11n))
    print("REFERENCE POINTS: analytic 3.69 | rfx 3.6273 | openEMS 3.4286 GHz")


if __name__ == "__main__":
    main()
