#!/usr/bin/env python3
"""Does matching the feed remove the resonance pull it causes?

THE QUESTION. The edge-fed fixture attaches a 50 ohm microstrip directly to a
patch radiating edge whose resistance at resonance is ~465 ohm (Balanis two-slot
model; the repo's own witness records 230-470 ohm for this geometry). A 51 ohm
line on a 465 ohm edge is not a feed, it is a heavy load: |Gamma| = 0.80. The
attached feed then carries the mode -- measured modal |Ez| under the feed
divided by under the patch is 1.00 attached, 0.03 removed -- and pulls the
resonance by -6.9%.

An INSET is the textbook cure: moving the attachment point into the patch, with
etched notches flanking the line, drops the edge resistance at the attachment
toward 50 ohm. The reference geometry this fixture is compared against uses
inset depth 2.40 mm and notch gap 0.90 mm each side, and the fixture has
neither.

MODE IDENTIFICATION -- READ THIS BEFORE CHANGING THE READOUT.
This patch has W = 10.129 mm > L = 8.595 mm, so the orthogonal TM001 sits BELOW
TM010 in frequency, and on the feed-removed arm it rings ~2.7x LOUDER (measured
parity census, feed-removed: TM001 7.6092 a=7.5e-4, TM010 8.7673 a=2.8e-4).
Two consequences, both of which this file got wrong before 2026-08-29:

  * Amplitude rank picks TM001 on the ruler arm and TM010 on the fed arms, and
    differencing those two fabricates a ~+7% "pull" out of nothing.  Selection
    is therefore by MEASURED SPATIAL PARITY, never by amplitude rank.
  * Two probes placed off BOTH symmetry planes cannot separate TM010 from
    TM001: their sign ratio reads the PRODUCT px*py, which is -1 for both.  It
    takes a four-probe mirror quad to read the two parities independently.

So every arm here reads a quad A(-x,-y) B(+x,-y) C(-x,+y) D(+x,+y), labels each
pole by parity, and among the TM010-parity poles picks the one whose four
magnitudes are most nearly equal (the patch TM010 is |Ez| ~ |cos(pi x'/L)|,
independent of y, so a clean patch mode reads the same magnitude at all four;
a feed-loaded hybrid does not).  Amplitude is never used to choose across modes.

PRE-DECLARED READING (unchanged from the original run):
  If the pull is caused by the IMPEDANCE MISMATCH at the attachment, then
  deepening the inset toward its matched depth must MONOTONICALLY shrink the
  pull, and at the reference's 2.40 mm the resonance must sit materially closer
  to the feed-removed value than the 0 mm case does.
    SUPPORTED  if |f(inset) - f_isolated| decreases monotonically with depth and
               at 2.40 mm is below half the 0 mm gap.
    REFUTED    if the pull is flat in inset depth, or grows -- then the pull is
               not the mismatch but the mere presence of attached metal, and an
               inset would not rescue the comparison.
  Either way the isolated arm (no feed at all) is the ruler, measured here, not
  taken from another run. Any arm whose settling witness ends above -40 dB is
  not read, and any arm with no TM010-parity pole is not read.

RECORDED VERDICT (2026-08-30). Source: docs/agent-memory/rfx-known-issues.md,
"Added 2026-08-30", section 1 plus the same-day tip-fixed follow-up (the
agent-memory file is local to the primary checkout); artifact root
scripts/diagnostics/_artifacts/patch_close_20260830/ (local, not committed).

  Default sweep (port held 5 mm from the domain face, so the open stub grows
  with the inset), N = 260 periods, per-depth NOTCH RULERS against the FED
  arms, every arm settled below -40 dB (rulers -78.2 / -75.4 / -64.8 / -48.5
  dB, fed -98.6 / -95.4 / -90.3 / -77.7 dB; the 2.4 mm ruler at N = 200 ended
  at -38.93 dB and was NOT READ, N was raised instead):

    inset mm   notch ruler   fed (GHz)   geometry   attachment    total
    0.00       8.75124       8.16121     +0.000     -6.742       -6.742
    0.80       8.93429       8.13141     +2.092     -8.987       -7.083
    1.60       9.06825       8.06673     +3.622    -11.044       -7.822
    2.40       9.08793       7.96350     +3.847    -12.373       -9.001

  |att(2.4)| / |att(0)| = 1.84 and the attachment column grows at every
  step, so the pre-declared survival rule fails.  REFUTED: matching the feed
  (deepening the inset) does not remove the attachment pull.  Control:
  notch-ruler(0 mm) = plain ruler bit for bit.  (inset_decomposition/
  decomposition_final.json, inset_decomposition.png, VESSL 369367257183 /
  185 / 188.)

  The depth trend in that column is a LINE-LENGTH trend, not an inset trend:
  with zero inset and no notch, lengthening the port-to-patch line by the
  same +4 / +8 / +12 cells the sweep adds moves the lower TM010-parity
  branch -2.61 / -5.34 / -8.23 %, which over-explains the growth.  The
  inset-specific part is not separable from the default sweep.

  --tip-fixed follow-up (VESSL 369367257262, N = 260, fed arms settled
  -84 .. -107 dB): attachment -6.74 / -6.22 / -5.54 / -3.63 % at 0 / 0.8 /
  1.6 / 2.4 mm.  Once the stub length is held the pull SHRINKS monotonically,
  but |att(2.4)| = 3.63 % is not below half of |att(0)| (0.5 x 6.74 =
  3.37 %): against the rule above the tip-fixed data is monotone but not
  halved -- borderline, not a pass.  At 2.4 mm the notch-geometry term
  (+3.85 %) and the attachment term (-3.63 %) cancel to +0.08 %; that is a
  coincidence of this h/4 fixture, not a rule to build a gate on.
  (inset_tipfixed/tipfixed_fits.json, tipfixed_decomposition.png.)

  Open, as a gate-design fact: every fed arm carries TWO TM010-parity poles
  (8.16 + 10.46 GHz at 0 mm; 7.96 + 9.61 at 2.4 mm) and the uniformity
  margin between them is 0.006 at 2.4 mm; the pick flips under domain
  padding.  A selector with that margin is one padding change away from a
  different answer.

Run: python scripts/diagnostics/patch_inset_match_vs_pull.py [--depths 0,0.8,1.6,2.4]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from rfx import Box, Simulation
from rfx.harminv import harminv
from rfx.sources import GaussianPulse

EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3
L = 8.595e-3
W_MSL = 1.8e-3
NOTCH_G = 0.90e-3          # the reference geometry's gap, each side
PORT_MARGIN = 5.0e-3
Z_GND = 4e-3
FEED_LEN = 8.0e-3
DOM_X, DOM_Y, DOM_Z = 29.747e-3, 18.130e-3, 12.787e-3
N_SUB = 4
DX = H_SUB / N_SUB
NUM_PERIODS = 120.0
BAR_DB = -40.0


def build(inset_m: float | None, fed: bool = True, tip_fixed: bool = False):
    """inset_m None -> plain patch, no feed (the depth-independent ruler).

    tip_fixed=True (fed arms only): translate the trace's open end AND the MSL
    port by +inset along x, so the trace length (open end -> attachment point)
    stays PORT_MARGIN + FEED_LEN = 13.18 mm at every depth.  The default sweep
    keeps the open end at x = 0, so the stub lengthens by the inset depth and
    the 2026-08-30 B1 result (open-stub reactive load, ~ -2.2 %/mm) is folded
    into the "attachment" column.  With the tip fixed, the attachment column
    isolates whatever the inset does that is NOT stub length.

    fed=False with an inset depth cuts the SAME notches but omits the feed
    line and the MSL port, driving the cavity with the ruler's own interior
    Ez dipole instead.  That separates the two things the fed sweep moves
    together: removing patch metal (which lowers every mode) and attaching
    the 50 ohm line (the pull this file is about).  With it,
        attach pull(d) = f_fed(d) / f_notch_ruler(d) - 1
    is the attachment alone, and f_notch_ruler(0) must reproduce the plain
    ruler exactly -- a built-in control, since inset 0 cuts no notches.
    """
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z), dx=DX,
                     cpml_layers=8, boundary="cpml")
    z_gnd_hi = Z_GND + DX
    z_sub_lo, z_sub_hi = z_gnd_hi, z_gnd_hi + H_SUB
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + DX
    x_p0 = PORT_MARGIN + FEED_LEN
    y_c = DOM_Y / 2.0
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, Z_GND), (DOM_X, DOM_Y, z_gnd_hi)), material="pec")
    sim.add(Box((0, 0, z_sub_lo), (DOM_X, DOM_Y, z_sub_hi)), material="ro4003c")

    if inset_m is None:
        # ruler: patch alone, soft interior source off both symmetry planes
        sim.add(Box((x_p0, y_c - W / 2, z_tr_lo),
                    (x_p0 + L, y_c + W / 2, z_tr_hi)), material="pec")
        sim.add_source(position=(x_p0 + 0.31 * L, y_c - 0.27 * W,
                                 0.5 * (z_sub_lo + z_sub_hi)),
                       component="ez", amplitude_kind="field",
                       waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6))
    else:
        # feed line runs to x_p0 + inset; the patch is cut back around it by the
        # notch gaps so the line reaches an interior point, not the whole edge.
        x_tip = x_p0 + inset_m
        x_open = inset_m if (fed and tip_fixed) else 0.0   # trace open end
        if fed:
            sim.add(Box((x_open, y_c - W_MSL / 2, z_tr_lo),
                        (x_tip, y_c + W_MSL / 2, z_tr_hi)), material="pec")
        if inset_m <= 0:
            sim.add(Box((x_p0, y_c - W / 2, z_tr_lo),
                        (x_p0 + L, y_c + W / 2, z_tr_hi)), material="pec")
        else:
            # patch minus the two notch slots: three boxes, so the line enters a
            # channel of width W_MSL + 2*NOTCH_G and touches only at x_tip.
            y_lo_hi = y_c - W_MSL / 2 - NOTCH_G
            y_hi_lo = y_c + W_MSL / 2 + NOTCH_G
            sim.add(Box((x_p0, y_c - W / 2, z_tr_lo),
                        (x_p0 + L, y_lo_hi, z_tr_hi)), material="pec")
            sim.add(Box((x_p0, y_hi_lo, z_tr_lo),
                        (x_p0 + L, y_c + W / 2, z_tr_hi)), material="pec")
            sim.add(Box((x_tip, y_lo_hi, z_tr_lo),
                        (x_p0 + L, y_hi_lo, z_tr_hi)), material="pec")
        if fed:
            sim.add_msl_port(position=(PORT_MARGIN + x_open, y_c, z_sub_lo), width=W_MSL,
                             height=H_SUB, direction="+x", impedance=50.0,
                             waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6))
        else:
            # identical drive to the plain ruler, so notches are the ONLY delta
            sim.add_source(position=(x_p0 + 0.31 * L, y_c - 0.27 * W,
                                     0.5 * (z_sub_lo + z_sub_hi)),
                           component="ez", amplitude_kind="field",
                           waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6))
    # PARITY QUAD: A(-x,-y) B(+x,-y) C(-x,+y) D(+x,+y), mid-substrate.  y at
    # +-0.26*W = +-2.63 mm sits OUTSIDE the +-1.80 mm notch channel, so every
    # probe is under patch metal at every inset depth swept here.
    x_c = x_p0 + 0.5 * L
    z_m = 0.5 * (z_sub_lo + z_sub_hi)
    quad = [(x_c - 0.30 * L, y_c - 0.26 * W),
            (x_c + 0.30 * L, y_c - 0.26 * W),
            (x_c - 0.30 * L, y_c + 0.26 * W),
            (x_c + 0.30 * L, y_c + 0.26 * W)]
    for (xq, yq) in quad:
        sim.add_probe(position=(xq, yq, z_m), component="ez")
    return sim, quad


def joint_amplitudes(sig4, poles, dt):
    """Least-squares complex amplitudes of every pole in EVERY probe at once.

    Projecting onto one pole at a time leaks neighbouring modes into the answer
    (the decaying exponentials are not orthogonal on a finite window), which
    shows up as unequal |a| at four symmetric probes.  Solving for all poles
    together removes that.  The series are real, so each pole enters with its
    conjugate partner.
    """
    n = np.arange(len(sig4[0]))
    cols = [np.exp(s_k * dt) ** n for s_k in poles]
    cols += [np.exp(np.conjugate(s_k) * dt) ** n for s_k in poles]
    Z = np.column_stack(cols)
    out = []
    for y in sig4:
        coef, *_ = np.linalg.lstsq(Z, y.astype(complex), rcond=None)
        out.append(coef[:len(poles)])
    return np.array(out)          # (4 probes, n_poles)


def classify(sig4, dt, f_lo=6e9, f_hi=14e9):
    """Modes labelled by MEASURED spatial parity across the probe quad.

    A(-x,-y) B(+x,-y) C(-x,+y) D(+x,+y).  x-parity is read from the two
    independent mirror pairs A|B and C|D, y-parity from A|C and B|D; both
    pairs of a given axis must agree in sign or the label is withheld.
    """
    seen = []
    for i in range(4):
        for m in harminv(sig4[i], dt, f_lo, f_hi):
            if m.Q > 2 and abs(m.amplitude) > 1e-9:
                if not any(abs(m.freq - g.freq) / m.freq < 5e-3 for g in seen):
                    seen.append(m)
    seen.sort(key=lambda m: m.freq)
    poles = [-m.decay + 2j * math.pi * m.freq for m in seen]
    if not poles:
        return []
    A = joint_amplitudes(sig4, poles, dt)
    rows = []
    for k, m in enumerate(seen):
        a = A[:, k]
        mag = np.abs(a)
        if mag.min() <= 0:
            continue
        cx1 = (a[0] * a[1].conjugate()).real / (mag[0] * mag[1])
        cx2 = (a[2] * a[3].conjugate()).real / (mag[2] * mag[3])
        cy1 = (a[0] * a[2].conjugate()).real / (mag[0] * mag[2])
        cy2 = (a[1] * a[3].conjugate()).real / (mag[1] * mag[3])
        ok = (np.sign(cx1) == np.sign(cx2)) and (np.sign(cy1) == np.sign(cy2))
        px, py = 0.5 * (cx1 + cx2), 0.5 * (cy1 + cy2)
        if not ok:
            lab = "AMBIGUOUS(pairs disagree)"
        else:
            lab = ("TM010" if px < 0 < py else
                   "TM001" if py < 0 < px else
                   "TM011" if px < 0 and py < 0 else "TM000/other")
        rows.append(dict(f_ghz=m.freq / 1e9, Q=m.Q, amp=float(np.mean(mag)),
                         px=float(px), py=float(py), label=lab,
                         cx=[float(cx1), float(cx2)], cy=[float(cy1), float(cy2)],
                         mag=[float(v) for v in mag],
                         uniformity=float(mag.min() / mag.max())))
    return rows


def run(tag, sim, quad, num_periods=None):
    adv = [str(a) for a in sim.preflight()]
    print(f"[{tag}] probe quad (mm) "
          f"{[(round(a*1e3, 4), round(b*1e3, 4)) for a, b in quad]}", flush=True)
    print(f"[{tag}] preflight ({len(adv)}), quoted verbatim:", flush=True)
    for a in adv:
        print(f"   ! {a}", flush=True)
    res = sim.run(num_periods=NUM_PERIODS if num_periods is None else num_periods)
    ts = np.asarray(res.time_series)
    # time_series is (n_steps, n_probes) -- ts[0] would be one TIME STEP, not
    # one probe.  That slip read a 2-sample series and reported "0.0 dB /
    # spectrum []" for every arm of the 2026-08-29 run.
    assert ts.ndim == 2 and ts.shape[1] == 4, f"unexpected time_series {ts.shape}"
    dt = float(res.dt)
    env = np.abs(ts[:, 0])
    end_db = 20 * math.log10(max(float(np.max(env[int(len(env) * .95):])), 1e-300)
                             / max(float(np.max(env)), 1e-300))
    i0 = int(ts.shape[0] * 0.3)
    rows = classify([np.asarray(ts[i0:, i], dtype=float) for i in range(4)], dt)
    print(f"[{tag}] settling {end_db:6.2f} dB (bar {BAR_DB})  n_steps {ts.shape[0]}",
          flush=True)
    print(f"[{tag}] parity census (f/Q/amp/px/py/label):", flush=True)
    for r in rows:
        print(f"    {r['f_ghz']:8.4f} Q{r['Q']:6.1f} a{r['amp']:.3g} "
              f"px{r['px']:+.3f}(AB{r['cx'][0]:+.2f} CD{r['cx'][1]:+.2f}) "
              f"py{r['py']:+.3f}(AC{r['cy'][0]:+.2f} BD{r['cy'][1]:+.2f}) "
              f"u{r['uniformity']:.3f}  {r['label']}", flush=True)
    # SELECTION BY MODE SHAPE, NEVER BY AMPLITUDE RANK (see module docstring).
    tm010 = [r for r in rows if r["label"] == "TM010"
             and abs(r["px"]) >= 0.7 and abs(r["py"]) >= 0.7]
    pick = max(tm010, key=lambda r: r["uniformity"]) if tm010 else None
    if len(tm010) > 1:
        print(f"[{tag}] {len(tm010)} TM010-parity modes; shape scores "
              f"{[(round(r['f_ghz'], 4), round(r['uniformity'], 3)) for r in tm010]}"
              f" -> picked {pick['f_ghz']:.4f}", flush=True)
    rec = dict(tag=tag, settling_db=end_db, settled=bool(end_db < BAR_DB),
               f=None if pick is None else pick["f_ghz"],
               Q=None if pick is None else pick["Q"],
               uniformity=None if pick is None else pick["uniformity"],
               census=rows)
    print("[RESULT] " + json.dumps(rec), flush=True)
    _persist(rec, adv, num_periods)
    return rec


def _persist(rec, adv, num_periods):
    """Append the record to a JSONL the moment it exists, then mirror the
    whole artifact directory to NFS.  Two 8.5 h runs died in 2026-08 because
    the save came after an optional stage; nothing here is optional."""
    import datetime
    import os
    import shutil
    path = os.environ.get("PATCH_ARTIFACT_JSONL")
    if not path:
        return
    row = dict(ts=datetime.datetime.utcnow().isoformat() + "Z",
               num_periods=num_periods, preflight=adv, rec=rec)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush(); os.fsync(fh.fileno())
    mirror = os.environ.get("PATCH_ARTIFACT_MIRROR")
    if mirror:
        os.makedirs(mirror, exist_ok=True)
        shutil.copy2(path, os.path.join(mirror, os.path.basename(path)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", default="0,0.8,1.6,2.4",
                   help="inset depths in mm")
    # The notch RULER arms are driven by a weak interior dipole and their TM001 Q
    # climbs with notch depth (measured 40.0 / 44.6 / 53.8 at 0 / 0.8 / 1.6 mm),
    # so a run length that settles the shallow arms truncates the deep ones:
    # settling_dB ~ -1832/Q at num_periods=120 (three arms, product constant to
    # 0.6%).  Lengthen the ruler arms; never lower the -40 dB bar.
    p.add_argument("--ruler-periods", type=float, default=NUM_PERIODS,
                   help="run length for the feed-removed arms")
    p.add_argument("--fed-periods", type=float, default=NUM_PERIODS,
                   help="run length for the fed arms")
    p.add_argument("--arms", default="all", choices=("all", "notch", "fed"),
                   help="restrict to one arm class (the other is read from a "
                        "previous log by hand; the CONTROL at 0 mm still runs)")
    p.add_argument("--tip-fixed", action="store_true",
                   help="fed arms: move the trace open end and the port with the "
                        "notch so the stub length stays 13.18 mm (see build())")
    a = p.parse_args()
    depths = [float(v) * 1e-3 for v in a.depths.split(",")]

    print(f"=== RULER: plain patch, no feed, no notches "
          f"(num_periods={a.ruler_periods:g}) ===")
    sim, quad = build(None)
    plain = run("isolated", sim, quad, a.ruler_periods)
    print()

    fed, notch = {}, {}
    for d in depths:
        if a.arms in ("all", "notch"):
            sim, quad = build(d, fed=False)
            notch[d] = run(f"notch-ruler {d*1e3:.2f}mm", sim, quad, a.ruler_periods)
            print()
        if a.arms in ("all", "fed"):
            sim, quad = build(d, fed=True, tip_fixed=a.tip_fixed)
            _tag = f"inset-tipfixed {d*1e3:.2f}mm" if a.tip_fixed else f"inset {d*1e3:.2f}mm"
            fed[d] = run(_tag, sim, quad, a.fed_periods)
            print()
    if a.arms != "all":
        print(f"=== arms={a.arms}: decomposition needs both classes, not printed ===")
        return 0

    print("=== READ: the fed sweep decomposed ===")
    if not plain["settled"] or plain["f"] is None:
        print("  NOT READ: the plain ruler did not settle or found no TM010-parity mode.")
        return 0
    f0 = plain["f"]
    print(f"  plain ruler (no feed, no notches): {f0:.4f} GHz  (u={plain['uniformity']:.3f})")
    # Control: at inset 0 no notches are cut, so the notch ruler MUST reproduce
    # the plain ruler.  A mismatch means the two build paths differ elsewhere.
    if 0.0 in notch and notch[0.0]["f"] is not None:
        dev = (notch[0.0]["f"] - f0) / f0 * 100
        print(f"  CONTROL notch-ruler(0 mm) = {notch[0.0]['f']:.4f} GHz "
              f"({dev:+.4f}% vs plain ruler; must be ~0)")
    print()
    hdr = (f"  {'depth':>7} {'notch ruler':>12} {'fed':>10} "
           f"{'geometry':>10} {'attachment':>12} {'total':>9}")
    print(hdr)
    rows = []
    for d in depths:
        n, f = notch[d], fed[d]
        if not (n["settled"] and f["settled"]) or n["f"] is None or f["f"] is None:
            print(f"  {d*1e3:7.2f} {'not read':>12}")
            continue
        geo = (n["f"] - f0) / f0 * 100          # notches alone, feed absent
        att = (f["f"] - n["f"]) / n["f"] * 100  # attaching the line, notches held
        tot = (f["f"] - f0) / f0 * 100
        rows.append((d, geo, att, tot))
        print(f"  {d*1e3:7.2f} {n['f']:12.4f} {f['f']:10.4f} "
              f"{geo:+9.2f}% {att:+11.2f}% {tot:+8.2f}%")
    if len(rows) >= 2:
        att = [abs(r[2]) for r in rows]
        mono = all(att[i] >= att[i + 1] for i in range(len(att) - 1))
        halved = att[-1] < 0.5 * att[0]
        print(f"\n  ATTACHMENT pull alone, monotonically shrinking: {mono}")
        print(f"  deepest inset below half the 0 mm attachment pull: {halved}")
        print("  ->", "SUPPORTED: the pull is the impedance mismatch"
              if (mono and halved) else
              "REFUTED: matching does not remove the attachment pull")
        print("\n  (the earlier single-ruler read conflated the geometry column"
              " with the attachment column; both are printed above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
