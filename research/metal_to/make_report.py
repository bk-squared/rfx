"""Build the self-contained HTML report for the metal-TO Phase-0/1 campaign.

Reads the committed JSON/NPZ artifacts, renders design/field maps as embedded
PNGs, emits inline-SVG charts (hover tooltips, light+dark), writes report.html.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
VES = HERE / "out_vessl"

# --- palette (validated: see dataviz scripts/validate_palette.js) ------------
S1L, S2L, S3L = "#2a78d6", "#eb6834", "#1baf7a"
S1D, S2D, S3D = "#3987e5", "#d95926", "#199e70"


def png_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def design_map(arr, title, cmap="gray_r"):
    fig, ax = plt.subplots(figsize=(1.9, 4.6))
    ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#9a9a93"); sp.set_linewidth(0.8)
    return png_uri(fig)


def field_map(ez, title, ref=None):
    """|Ez| in dB relative to a common reference so the two panels compare."""
    ref = float(np.max(ez)) if ref is None else ref
    db = 20.0 * np.log10(np.maximum(ez, 1e-30) / ref)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    im = ax.imshow(db.T, origin="lower", cmap="inferno", aspect="auto",
                   vmin=-45, vmax=0, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, color="#8a8880", pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015, ticks=[0, -15, -30, -45])
    cb.ax.tick_params(labelsize=7, colors="#8a8880", length=2)
    cb.outline.set_visible(False)
    return png_uri(fig)


# ---------------------------------------------------------------- data load
pk = json.loads((OUT / "probe01_kottke.json").read_text())
pl = json.loads((OUT / "probe01_legacy.json").read_text())

runs = {}
for arm in ("B", "C"):
    for init in ("uniform", "low", "stub"):
        f = VES / f"phase1c_{arm}_{init}" / f"phase1c_{init}_i150_{arm}.json"
        if f.exists():
            runs[(arm, init)] = json.loads(f.read_text())

npz = {}
for k in runs:
    arm, init = k
    npz[k] = np.load(VES / f"phase1c_{arm}_{init}" / f"phase1c_{init}_i150_{arm}_final.npz")

seed = np.load(OUT / "phase1b_stub_i1_C_final.npz")

# tier-1 / 1b cross-validation (independent hard-PEC geometry + extractor)
XV = VES / "xval1"
xv = {f.stem: json.loads(f.read_text()) for f in XV.glob("*.json")}
_xf = np.array(xv["empty"]["freqs_GHz"]); _xe = np.array(xv["empty"]["s21_db"])
_xit = int(np.argmin(abs(_xf - 6.0))); _xpb = abs(_xf - 6.0) > 0.9


def xv_metrics(key):
    t = np.array(xv[key]["s21_db"]) - _xe
    return dict(notch=float(t[_xit]), pb=float(np.mean(t[_xpb])),
                contrast=float(np.mean(t[_xpb]) - t[_xit]),
                fmin=xv[key]["f_min_GHz"], dmin=xv[key]["depth_min_db"],
                t=[float(v) for v in t])


_sweep = sorted((xv[k]["stub_mm"], k) for k in xv if k.startswith("oracle_"))
_cal_key = max(_sweep, key=lambda sk: xv_metrics(sk[1])["contrast"])[1]
CAL = xv_metrics(_cal_key); CAL["stub_mm"] = xv[_cal_key]["stub_mm"]
XVB = xv_metrics("B_stub"); XVC = xv_metrics("C_low"); XVO = xv_metrics("oracle")
ORACLE = runs[("B", "stub")]["oracle_contrast_db"]
win = runs[("B", "stub")]
freqs = win["freqs_GHz"]


# ------------------------------------------------------------- svg helpers
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def line_chart(series, xdom, ydom, xlab, ylab, w=680, h=300, xticks=None,
               yticks=None, vline=None, vlab=""):
    """series: list of dict(name, slot, pts=[(x,y,tip)])"""
    ml, mr, mt, mb = 58, 96, 14, 42
    pw, ph = w - ml - mr, h - mt - mb
    x0, x1 = xdom
    y0, y1 = ydom

    def X(v):
        return ml + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return mt + (1 - (v - y0) / (y1 - y0)) * ph

    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(ylab)} vs {esc(xlab)}">']
    for ty in (yticks or []):
        p.append(f'<line class="grid" x1="{ml}" y1="{Y(ty):.1f}" x2="{ml+pw}" y2="{Y(ty):.1f}"/>')
        p.append(f'<text class="tick" x="{ml-9}" y="{Y(ty)+4:.1f}" text-anchor="end">{ty:g}</text>')
    for tx in (xticks or []):
        p.append(f'<text class="tick" x="{X(tx):.1f}" y="{mt+ph+18}" text-anchor="middle">{tx:g}</text>')
    p.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')
    if vline is not None:
        p.append(f'<line class="vline" x1="{X(vline):.1f}" y1="{mt}" x2="{X(vline):.1f}" y2="{mt+ph}"/>')
        p.append(f'<text class="vlab" x="{X(vline)+6:.1f}" y="{mt+12}">{esc(vlab)}</text>')
    for s in series:
        d = " ".join(("M" if i == 0 else "L") + f"{X(x):.1f},{Y(y):.1f}"
                     for i, (x, y, _) in enumerate(s["pts"]))
        p.append(f'<path class="ln s{s["slot"]}" d="{d}"/>')
    for s in series:
        for (x, y, tip) in s["pts"]:
            p.append(f'<circle class="pt s{s["slot"]}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" '
                     f'data-tip="{esc(tip)}"/>')
    # direct labels at the right edge, de-collided vertically (>=15 px apart)
    lab = sorted(((Y(s["pts"][-1][1]), X(s["pts"][-1][0]), s) for s in series),
                 key=lambda t: t[0])
    placed = []
    for ly, lx, s in lab:
        while any(abs(ly - q) < 15 for q in placed):
            ly += 15
        placed.append(ly)
        p.append(f'<text class="dlab s{s["slot"]}t" x="{lx+10:.1f}" y="{ly+4:.1f}">{esc(s["name"])}</text>')
    p.append(f'<text class="axlab" x="{ml+pw/2:.0f}" y="{h-6}" text-anchor="middle">{esc(xlab)}</text>')
    p.append(f'<text class="axlab" transform="translate(14,{mt+ph/2:.0f}) rotate(-90)" text-anchor="middle">{esc(ylab)}</text>')
    p.append("</svg>")
    return "\n".join(p)


def grouped_bars(groups, series_names, values, ydom, ylab, ref=None, reflab="",
                 w=680, h=320, notes=None):
    """values[g][s] = float or None"""
    ml, mr, mt, mb = 58, 20, 16, 54
    pw, ph = w - ml - mr, h - mt - mb
    y0, y1 = ydom

    def Y(v):
        return mt + (1 - (v - y0) / (y1 - y0)) * ph

    gw = pw / len(groups)
    bw = min(46, gw / (len(series_names) + 1.2))
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(ylab)}">']
    for ty in range(int(y0), int(y1) + 1, 5):
        p.append(f'<line class="grid" x1="{ml}" y1="{Y(ty):.1f}" x2="{ml+pw}" y2="{Y(ty):.1f}"/>')
        p.append(f'<text class="tick" x="{ml-9}" y="{Y(ty)+4:.1f}" text-anchor="end">{ty}</text>')
    p.append(f'<line class="axis" x1="{ml}" y1="{Y(0):.1f}" x2="{ml+pw}" y2="{Y(0):.1f}"/>')
    if ref is not None:
        p.append(f'<line class="ref" x1="{ml}" y1="{Y(ref):.1f}" x2="{ml+pw}" y2="{Y(ref):.1f}"/>')
        p.append(f'<text class="reflab" x="{ml+6}" y="{Y(ref)-8:.1f}">{esc(reflab)}</text>')
    for gi, g in enumerate(groups):
        gx = ml + gi * gw + gw / 2
        for si, sn in enumerate(series_names):
            v = values[gi][si]
            bx = gx + (si - (len(series_names) - 1) / 2) * (bw + 3) - bw / 2
            if v is None:
                p.append(f'<rect class="nodata" x="{bx:.1f}" y="{Y(0)-26:.1f}" width="{bw:.1f}" '
                         f'height="24" rx="4" data-tip="{esc(sn)} / {esc(g)}: 데이터 없음 — 하강 도중 반복 120회에서 인프라 사유로 중단(J=0.211)"/>')
                p.append(f'<text class="nodlab" x="{bx+bw/2:.1f}" y="{Y(0)-32:.1f}" text-anchor="middle">미완</text>')
                continue
            top, bot = (Y(v), Y(0)) if v >= 0 else (Y(0), Y(v))
            p.append(f'<rect class="bar s{si+1}b" x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                     f'height="{max(2, bot-top):.1f}" rx="4" '
                     f'data-tip="{esc(sn)} / {esc(g)}: {v:+.1f} dB"/>')
            vy = top - 7 if v >= 0 else bot + 15
            p.append(f'<text class="blab" x="{bx+bw/2:.1f}" y="{vy:.1f}" text-anchor="middle">{v:+.1f}</text>')
        p.append(f'<text class="tick" x="{gx:.1f}" y="{mt+ph+20}" text-anchor="middle">{esc(g)}</text>')
    p.append(f'<text class="axlab" transform="translate(14,{mt+ph/2:.0f}) rotate(-90)" text-anchor="middle">{esc(ylab)}</text>')
    if notes:
        p.append(f'<text class="note" x="{ml}" y="{h-8}">{esc(notes)}</text>')
    p.append("</svg>")
    return "\n".join(p)


# ------------------------------------------------------------------ charts
lv_k = [(r["level"], r["s21_ft"] ** 2,
         f'Kottke · 점유율 {r["level"]:.1f}: J={r["s21_ft"]**2:.4f}, R={r["R"]:.2f}, T={r["T"]:.2f}, A={r["A"]:+.2f}')
        for r in pk["rows"]]
lv_l = [(r["level"], r["s21_ft"] ** 2,
         f'레거시 · 점유율 {r["level"]:.1f}: J={r["s21_ft"]**2:.4f}, R={r["R"]:.2f}, T={r["T"]:.2f}, A={r["A"]:+.2f}')
        for r in pl["rows"]]
chart_probe = line_chart(
    [dict(name="Kottke (운영)", slot=1, pts=lv_l and lv_k),
     dict(name="레거시 감쇠", slot=2, pts=lv_l)],
    (0, 1), (0, 0.05), "점유율  (0 = 금속 없음 → 1 = 완전한 스텁)",
    "J = |S21(6 GHz)|²",
    xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    yticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05])

groups = ["uniform ρ≈0.5", "low ρ≈0.25", "λ/4-stub seed"]
snames = ["B — damped conductive gray", "C — legacy damping"]
vals = []
for init in ("uniform", "low", "stub"):
    row = []
    for arm in ("B", "C"):
        r = runs.get((arm, init))
        row.append(None if r is None else r["contrast_db"])
    vals.append(row)
chart_bars = grouped_bars(
    groups, snames, vals, (-10, 25), "통과대역↔노치 대비도 (dB)",
    ref=ORACLE, reflab=f"해석적 λ/4 스텁 — {ORACLE:.1f} dB",
    notes="이진화 설계 · 하드 PEC 평가기 · 30주기 창 — 모든 실행에 동일 적용")

bandsets = [("B 스텁씨앗(우승)", 1, ("B", "stub")),
            ("C 저밀도(씨앗 없음)", 2, ("C", "low")),
            ("B 균일(갇힘)", 3, ("B", "uniform"))]
series = []
for name, slot, key in bandsets:
    t = runs[key]["t_hard_band"]
    pts = [(f, 20 * np.log10(max(v, 1e-6)),
            f"{name} · {f:.2f} GHz: {20*np.log10(max(v,1e-6)):.1f} dB")
           for f, v in zip(freqs, t)]
    series.append(dict(name=name, slot=slot, pts=pts))
# --- cross-validation charts (independent hard-PEC path) ---
_sw = [(xv[k]["stub_mm"], xv_metrics(k)["contrast"]) for _, k in _sweep]
_sw.sort()
chart_sweep = line_chart(
    [dict(name="고전 스텁 (메시 보정 스윕)", slot=2,
          pts=[(L, c, f"스텁 {L:.2f} mm: 대비도 {c:+.1f} dB") for L, c in _sw])],
    (5.6, 7.6), (-10, 35), "고전 스텁 길이 (mm)", "통과대역↔노치 대비도 (dB)",
    yticks=[-10, 0, 10, 20, 30], xticks=[5.8, 6.4, 7.0, 7.37])

chart_xband = line_chart(
    [dict(name="C 자유형(씨앗 없음)", slot=1,
          pts=[(f, v, f"자유형 · {f:.2f} GHz: {v:.1f} dB") for f, v in zip(_xf, XVC["t"])]),
     dict(name=f"고전 보정 {CAL['stub_mm']:.2f} mm", slot=2,
          pts=[(f, v, f"보정 고전 · {f:.2f} GHz: {v:.1f} dB") for f, v in zip(_xf, CAL["t"])]),
     dict(name="고전 해석적 7.37 mm", slot=3,
          pts=[(f, v, f"해석적 · {f:.2f} GHz: {v:.1f} dB") for f, v in zip(_xf, XVO["t"])])],
    (4.5, 8.5), (-40, 6), "주파수 (GHz)", "빈 선로 대비 |S21| (dB)",
    xticks=[4.5, 5.5, 6.5, 7.5, 8.5], yticks=[-40, -30, -20, -10, 0],
    vline=6.0, vlab="목표 6.0 GHz")

chart_band = line_chart(series, (4.5, 8.5), (-32, 6),
                        "주파수 (GHz)", "정규화 |S21| (dB)",
                        xticks=[4.5, 5.5, 6.5, 7.5, 8.5],
                        yticks=[-30, -20, -10, 0], vline=6.0, vlab="목표 6.0 GHz")

# --------------------------------------------------------------- image maps
maps = [
    (design_map(seed["hard"], ""), "λ/4 스텁 씨앗",
     "해석적 λ/4 스텁 · 폭 1셀"),
    (design_map(npz[("B", "stub")]["hard"], ""), "B — 우승",
     "넓힌 저-Z₀ 스텁 + 접합부 발"),
    (design_map(npz[("C", "low")]["hard"], ""), "C — 씨앗 없음",
     "랜덤 저밀도에서 스스로 찾은 형상"),
    (design_map(npz[("B", "uniform")]["hard"], ""), "B — 갇힘",
     "중간 밀도 시작 · 끝내 못 빠져나옴"),
]
ez_ref = max(float(np.max(npz[("B", "stub")]["ez"])), float(np.max(npz[("B", "uniform")]["ez"])))
ez_win = field_map(npz[("B", "stub")]["ez"], "우승 설계 (B · 스텁 씨앗) — 6.0 GHz |Ez| (dB, 공통 기준)", ez_ref)
ez_trap = field_map(npz[("B", "uniform")]["ez"], "갇힌 설계 (B · 균일 초기값) — 6.0 GHz |Ez| (dB, 공통 기준)", ez_ref)

# ------------------------------------------------------------------- table
rows_tbl = []
for init in ("stub", "low", "uniform"):
    for arm in ("B", "C"):
        r = runs.get((arm, init))
        label = f"{arm} · {init}"
        if r is None:
            rows_tbl.append(f"<tr><td>{label}</td><td colspan='5' class='na'>반복 120회에서 인프라 사유로 중단 (J = 0.211, 하강 진행 중) — 재실행 대상</td></tr>")
            continue
        cls = " class='win'" if (arm, init) == ("B", "stub") else ""
        rows_tbl.append(
            f"<tr{cls}>"
            f"<td>{label}</td>"
            f"<td class='num'>{r['contrast_db']:+.1f}</td>"
            f"<td class='num'>{r['J_hard_ft_db']:.1f}</td>"
            f"<td class='num'>{r['t_notch']:.3f}</td>"
            f"<td class='num'>{r['t_pb']:.3f}</td>"
            f"<td class='num'>{r['fill_hard']:.2f}</td>"
            "</tr>")
table_html = "\n".join(rows_tbl)

W = win
tiles = [
    (f"{XVC['contrast']:+.1f} dB", "자유형 설계 (씨앗 없음)", "독립 형상·독립 추출기 검증"),
    (f"{CAL['contrast']:+.1f} dB", "메시 보정 고전 스텁", f"{CAL['stub_mm']:.2f} mm · 같은 조건"),
    (f"{XVC['contrast']-CAL['contrast']:+.1f} dB", "차이 — 동등, 우세 아님", "최초 보고 2.2배는 철회"),
    (f"{XVC['dmin']:.1f} dB", "6 GHz 노치 깊이", "정착 −119 dB · 신뢰 82/82"),
]
tiles_html = "\n".join(
    f'<div class="tile"><div class="tval">{esc(v)}</div><div class="tlab">{esc(l)}</div>'
    f'<div class="tsub">{esc(s)}</div></div>' for v, l, s in tiles)

maps_html = "\n".join(
    f'<figure class="mapfig"><img src="{u}" alt="{esc(cap)}"/>'
    f'<figcaption><b>{esc(cap)}</b><span>{esc(sub)}</span></figcaption></figure>'
    for u, cap, sub in maps)

HTML = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rfx 자유형 금속 위상최적화 — 교차검증까지</title>
<style>
:root {{
  color-scheme: light;
  --bg:#f6f6f4; --surface:#fcfcfb; --line:#e2e1dc; --grid:#eceae4;
  --ink:#0b0b0b; --ink2:#52514e; --ink3:#7c7a73;
  --s1:{S1L}; --s2:{S2L}; --s3:{S3L};
  --accent:#2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg:#121211; --surface:#1a1a19; --line:#2e2e2b; --grid:#262623;
    --ink:#ffffff; --ink2:#c3c2b7; --ink3:#93928a;
    --s1:{S1D}; --s2:{S2D}; --s3:{S3D}; --accent:{S1D};
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg:#121211; --surface:#1a1a19; --line:#2e2e2b; --grid:#262623;
  --ink:#ffffff; --ink2:#c3c2b7; --ink3:#93928a;
  --s1:{S1D}; --s2:{S2D}; --s3:{S3D}; --accent:{S1D};
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI","Apple SD Gothic Neo","Noto Sans KR",sans-serif;
 -webkit-font-smoothing:antialiased;word-break:keep-all;overflow-wrap:anywhere}}
.wrap{{max-width:920px;margin:0 auto;padding:44px 22px 80px}}
header h1{{font-size:1.72rem;line-height:1.3;margin:0 0 8px;letter-spacing:-.02em}}
header p.sub{{color:var(--ink2);margin:0 0 6px;font-size:1.02rem}}
header p.meta{{color:var(--ink3);margin:0;font-size:.83rem;font-variant-numeric:tabular-nums}}
h2{{font-size:1.17rem;margin:46px 0 6px;letter-spacing:-.01em}}
h2 .n{{color:var(--ink3);font-weight:600;margin-right:.5em;font-size:.92rem}}
p{{color:var(--ink2);margin:.7em 0}}
p strong,li strong{{color:var(--ink);font-weight:650}}
.lede{{color:var(--ink);font-size:1.03rem}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:26px 0 8px}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 16px 14px}}
.tval{{font-size:1.85rem;font-weight:680;letter-spacing:-.02em;font-variant-numeric:tabular-nums;color:var(--ink)}}
.tile:first-child .tval{{color:var(--accent)}}
.tlab{{font-size:.9rem;color:var(--ink);margin-top:2px}}
.tsub{{font-size:.79rem;color:var(--ink3);margin-top:3px}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px 16px 10px;margin:16px 0}}
.card h3{{margin:0 0 2px;font-size:.98rem}}
.card p.cap{{margin:0 0 10px;font-size:.85rem;color:var(--ink3)}}
.chart{{width:100%;height:auto;display:block;overflow:visible}}
.chart .grid{{stroke:var(--grid);stroke-width:1}}
.chart .axis{{stroke:var(--line);stroke-width:1.5}}
.chart .tick{{fill:var(--ink3);font-size:11px;font-variant-numeric:tabular-nums}}
.chart .axlab{{fill:var(--ink2);font-size:11.5px}}
.chart .note{{fill:var(--ink3);font-size:10.5px}}
.chart .ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.chart .pt{{stroke:var(--surface);stroke-width:2;cursor:pointer}}
.chart .dlab{{font-size:11.5px;font-weight:600}}
.chart .s1{{stroke:var(--s1)}} .chart circle.s1{{fill:var(--s1)}}
.chart .s2{{stroke:var(--s2)}} .chart circle.s2{{fill:var(--s2)}}
.chart .s3{{stroke:var(--s3)}} .chart circle.s3{{fill:var(--s3)}}
.chart .s1t{{fill:var(--s1)}} .chart .s2t{{fill:var(--s2)}} .chart .s3t{{fill:var(--s3)}}
.chart .bar{{stroke:var(--surface);stroke-width:2;cursor:pointer}}
.chart .s1b{{fill:var(--s1)}} .chart .s2b{{fill:var(--s2)}}
.chart .blab{{fill:var(--ink2);font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;paint-order:stroke;stroke:var(--surface);stroke-width:3px;stroke-linejoin:round}}
.chart .nodata{{fill:var(--grid)}}
.chart .nodlab{{fill:var(--ink3);font-size:10.5px}}
.chart .ref{{stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:5 4}}
.chart .reflab{{fill:var(--ink3);font-size:11px;paint-order:stroke;stroke:var(--surface);stroke-width:3px;stroke-linejoin:round}}
.chart .vline{{stroke:var(--ink3);stroke-width:1.2;stroke-dasharray:4 4}}
.chart .vlab{{fill:var(--ink3);font-size:10.5px}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 4px;font-size:.83rem;color:var(--ink2)}}
.legend i{{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:6px;vertical-align:-1px}}
.maps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:14px 0 4px}}
.mapfig{{margin:0;text-align:center}}
.mapfig img{{width:100%;max-width:150px;height:auto}}
.mapfig figcaption{{font-size:.79rem;color:var(--ink3);margin-top:6px;line-height:1.45}}
.mapfig figcaption b{{display:block;color:var(--ink);font-size:.85rem;font-weight:620}}
.fields{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin:14px 0 4px}}
.fields img{{width:100%;height:auto;border-radius:8px}}
table{{width:100%;border-collapse:collapse;margin:14px 0 6px;font-size:.87rem}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
th{{color:var(--ink3);font-weight:600;font-size:.79rem;letter-spacing:.02em}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}}
td.na{{color:var(--ink3);font-style:italic}}
tr.win td{{background:color-mix(in srgb, var(--accent) 9%, transparent)}}
ul{{color:var(--ink2);padding-left:1.15em}} li{{margin:.42em 0}}
.callout{{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
 border-radius:10px;padding:14px 16px;margin:18px 0}}
.callout p{{margin:0;color:var(--ink)}}
footer{{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);
 color:var(--ink3);font-size:.8rem;line-height:1.7}}
code{{background:var(--grid);padding:.1em .38em;border-radius:4px;font-size:.87em;color:var(--ink2)}}
#tip{{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
 background:var(--ink);color:var(--bg);padding:6px 9px;border-radius:7px;font-size:.79rem;
 max-width:280px;z-index:9;box-shadow:0 4px 14px rgba(0,0,0,.22)}}
</style></head>
<body>
<div class="wrap">
<header>
  <h1>자유형 금속 위상최적화가 고전 스텁과 동등한 성능에 도달했습니다</h1>
  <p class="sub">rfx 미분가능 3-D FDTD · 2 256셀 자유형 금속 밀도로 6 GHz 노치 필터 설계 — 독립 경로 교차검증으로 최초 결론을 정정한 기록</p>
  <p class="meta">2026-08-25 실험 · 2026-08-26 교차검증 정정 · 브랜치 <code>feat/metal-topology-ramp</code> · VESSL RTX 4090 잡 21건 · 최종 판정은 <b>독립 하드 PEC 형상 + 독립 추출기</b> 기준</p>
</header>

<div class="tiles">{tiles_html}</div>

<div class="callout"><p><b>정정 기록입니다.</b> 8월 25일에는 자유형 설계가 고전 λ/4 스텁의 2.2배 대비도를 냈다고 보고했습니다. 이튿날 독립 경로로 교차검증하니 그 마진의 대부분이 두 가지 낙관 편향이었고, 공정하게 다시 재면 <b>자유형 설계는 고전 스텁과 동등</b>합니다(30.5 대 31.6 dB). 넘어서지는 못했습니다. 확정된 T-MTT 논문이 유전체 테이퍼에 대해 쓴 문장 — “기울기는 고전 합성법의 성능을 회복하지, 능가하지 않는다” — 이 자유형 금속에도 그대로 적용됩니다.</p></div>

<h2><span class="n">01</span>왜 어려웠나 — 회색 금속 통과 구간을 실측하다</h2>
<p class="lede">밀도 기반 위상최적화는 “금속 없음(0)”에서 “금속 있음(1)”까지 중간 밀도를 반드시 지나갑니다. 그 구간에서 목적함수가 어떻게 생겼는지를 먼저 측정했습니다. 검증된 노치 지오메트리에서 스텁을 해석적 λ/4 길이로 <b>고정</b>하고 점유율만 0→1로 올린 결과입니다.</p>
<div class="card">
  <h3>점유율 스윕 목적함수 지형</h3>
  <p class="cap">낮을수록 좋음 · 두 곡선 모두 같은 지오메트리·같은 창·같은 추출기</p>
  <div class="legend"><span><i style="background:var(--s1)"></i>Kottke (운영 경로)</span><span><i style="background:var(--s2)"></i>레거시 감쇠 경로</span></div>
  {chart_probe}
</div>
<p><b>운영 경로(Kottke)의 회색 금속은 “손실 매질”이 아니라 무손실 고유전율 <i>공진기</i>였습니다.</b> 점유율이 오르는 동안 이 가짜 공진이 대역을 가로질러 내려오면서 레벨 0.7에 <b>가짜 국소 최소</b>, 0.8에 <b>장벽</b>을 만든 뒤에야 진짜 노치에 도달합니다. 최적화기가 여기 갇히면 약한 해에 주저앉습니다. 반대로 레거시 경로는 경로의 90 %가 <b>평탄</b>해서(회색 스텁이 선로에 거의 보이지 않음) 기울기 신호 자체가 소멸합니다.</p>
<p>이 측정이 처방을 바꿨습니다. 밀도→점유율 곡선만 재조정하는 방식으로는 두 병리 모두 못 고칩니다. 대신 <b>중간 밀도에 유한한 전도도를 부여해 형성 중인 가짜 공진기를 감쇠</b>시키고, 밀도 1 부근에서만 진짜 PEC로 접는 방식(“전도성 회색”, 아래 B안)을 설계했습니다.</p>

<h2><span class="n">02</span>캠페인 결과 — 시작점이 성패를 갈랐다</h2>
<p>동일 예산(150회 반복)·동일 목적함수로 두 경로 × 세 시작점을 돌렸습니다. 목적함수는 단순 억제가 아니라 <b>선택도</b>입니다: 빈 선로 기준으로 정규화한 6 GHz 억제와 통과대역 보존을 함께 봅니다(1차 시도에서 단일 주파수 목적함수가 “전 대역을 막는 금속 벽”이라는 퇴화해를 낳았기 때문입니다).</p>
<div class="card">
  <h3>이진화 설계의 통과대역↔노치 대비도</h3>
  <p class="cap">높을수록 좋음 · 점선은 해석적 λ/4 스텁(오라클)을 같은 평가기로 잰 값</p>
  <div class="legend"><span><i style="background:var(--s1)"></i>B — 전도성 회색(감쇠)</span><span><i style="background:var(--s2)"></i>C — 레거시 감쇠</span></div>
  {chart_bars}
</div>
<table>
  <thead><tr><th>실행</th><th style="text-align:right">대비도 (dB)</th><th style="text-align:right">6 GHz 깊이 (dB)</th>
  <th style="text-align:right">노치 투과</th><th style="text-align:right">통과대역 투과</th><th style="text-align:right">채움률</th></tr></thead>
  <tbody>{table_html}</tbody>
</table>
<ul>
  <li><b>균일 초기값은 두 경로 모두 실패했습니다</b> — 01절에서 실측한 통과 장벽이 실제 장애물임을 확인해 줍니다.</li>
  <li><b>B(전도성 회색)는 저밀도 시작에서 빈 영역으로 붕괴</b>합니다. 회색의 흡수 자체가 통과대역 벌점을 키우니 “금속을 아예 지우는” 방향이 국소적으로 유리해지기 때문입니다 — 감쇠 세기와 통과대역 가중치의 상호작용이 B안의 약점입니다.</li>
  <li><b>씨앗을 주면 B가 최고 성적</b>을 냅니다. 프로브가 예측한 대로, 올바른 골짜기에 놓이면 매끈한 기울기가 그때부터 위력을 발휘합니다.</li>
  <li><b>C(레거시)는 씨앗 없이 저밀도에서 +18.9 dB를 스스로 찾았습니다</b> — 씨앗 정제가 아니라 진짜 자유형 발견도 성립한다는 뜻입니다.</li>
</ul>

<h2><span class="n">03</span>우승 설계 — 최적화기가 스텁을 넓혔다</h2>
<p>결과가 해석 가능하다는 점이 중요합니다. 최적화기는 씨앗으로 준 1셀 폭 스텁을 <b>폭 5~6셀의 광폭 스텁</b>으로 키웠습니다. 낮은 특성 임피던스 → 더 깊고 날카로운 노치라는 고전 이론과 정확히 일치하는 방향이며, 접합부에 작은 발까지 붙였습니다.</p>
<div class="card">
  <h3>이진화된 금속 배치</h3>
  <p class="cap">검정 = 금속 · 각 지도는 3 mm × 12 mm 설계 영역(24 × 94셀)</p>
  <div class="maps">{maps_html}</div>
  <p class="cap" style="margin-top:10px">각 지도의 <b>아래 모서리</b>가 선로와 만나는 접합부입니다 · 위로 갈수록 선로에서 멀어집니다.</p>
</div>
<div class="card">
  <h3>하드 PEC 대역 응답</h3>
  <p class="cap">빈 선로 기준 정규화 · 낮을수록 강한 억제</p>
  <div class="legend"><span><i style="background:var(--s1)"></i>B 스텁씨앗(우승)</span><span><i style="background:var(--s2)"></i>C 저밀도(씨앗 없음)</span><span><i style="background:var(--s3)"></i>B 균일(갇힘)</span></div>
  {chart_band}
</div>
<p>우승 설계의 응답은 목표 주파수에 정확히 앉은 깨끗한 V자입니다. 갇힌 설계(B 균일)는 대비 자체가 뒤집혀 통과대역이 노치보다 더 막히는, 필터라 부를 수 없는 상태입니다.</p>

<h2><span class="n">04</span>장 분포 — 무엇이 달랐나</h2>
<p>6 GHz에서 기판 표면의 전기장 세기입니다. 우승 설계는 스텁이 공진하며 <b>선로 하류로 가는 에너지를 끊습니다</b>. 갇힌 설계는 금속이 영역 전반에 흩어져 선로를 그냥 어지럽힐 뿐 선택적 공진을 만들지 못합니다.</p>
<div class="card">
  <div class="fields">
    <img src="{ez_win}" alt="우승 설계의 6 GHz 표면 전기장 분포">
    <img src="{ez_trap}" alt="갇힌 설계의 6 GHz 표면 전기장 분포">
  </div>
</div>

<h2><span class="n">05</span>교차검증 — 결론이 뒤집힌 지점</h2>
<p class="lede">여기까지의 수치는 모두 <b>최적화에 쓴 것과 같은 연산자</b>로 잰 값이고, 비교 대상인 고전 스텁은 <b>해석식이 준 길이 그대로</b>였습니다. 두 가지 모두 유리한 쪽으로 기울 수 있어서, 설계를 실제 PEC 상자로 다시 세워 독립 추출기로 풀고(형상·추출 경로 교체), 고전 스텁에도 같은 메시에서 길이를 맞출 기회를 줬습니다.</p>
<div class="card">
  <h3>고전 스텁에 같은 메시 보정을 허용하면</h3>
  <p class="cap">해석적 길이 7.37 mm는 이 메시에서 노치를 5.70 GHz에 놓습니다 · 목표에 앉히는 길이는 7.00 mm</p>
  {chart_sweep}
</div>
<div class="card">
  <h3>독립 경로 대역 응답</h3>
  <p class="cap">실제 PEC 상자 형상 · 독립 추출기 · 80주기 창(정착 −119 dB, 신뢰 82/82)</p>
  <div class="legend"><span><i style="background:var(--s1)"></i>C 자유형(씨앗 없음)</span><span><i style="background:var(--s2)"></i>고전 보정 {CAL['stub_mm']:.2f} mm</span><span><i style="background:var(--s3)"></i>고전 해석적 7.37 mm</span></div>
  {chart_xband}
</div>
<table>
  <thead><tr><th>설계</th><th style="text-align:right">대비도 (dB)</th><th style="text-align:right">6 GHz 억제</th><th style="text-align:right">통과대역</th><th style="text-align:right">자기 최소점</th></tr></thead>
  <tbody>
    <tr><td>고전 스텁 · 메시 보정 {CAL['stub_mm']:.2f} mm</td><td class="num">{CAL['contrast']:+.1f}</td><td class="num">{CAL['notch']:.1f}</td><td class="num">{CAL['pb']:.1f}</td><td class="num">{CAL['dmin']:.1f} dB @ {CAL['fmin']:.2f} GHz</td></tr>
    <tr class="win"><td>C 자유형 · 씨앗 없음</td><td class="num">{XVC['contrast']:+.1f}</td><td class="num">{XVC['notch']:.1f}</td><td class="num">{XVC['pb']:.1f}</td><td class="num">{XVC['dmin']:.1f} dB @ {XVC['fmin']:.2f} GHz</td></tr>
    <tr><td>B 스텁씨앗</td><td class="num">{XVB['contrast']:+.1f}</td><td class="num">{XVB['notch']:.1f}</td><td class="num">{XVB['pb']:.1f}</td><td class="num">{XVB['dmin']:.1f} dB @ {XVB['fmin']:.2f} GHz</td></tr>
    <tr><td>고전 스텁 · 해석적 7.37 mm</td><td class="num">{XVO['contrast']:+.1f}</td><td class="num">{XVO['notch']:.1f}</td><td class="num">{XVO['pb']:.1f}</td><td class="num">{XVO['dmin']:.1f} dB @ {XVO['fmin']:.2f} GHz</td></tr>
  </tbody>
</table>
<p>두 가지 낙관 편향이 잡혔습니다. <b>첫째, 기준선이 보정되지 않았습니다</b> — 해석적 길이는 이 메시에서 노치를 5.70 GHz에 놓기 때문에, 처음 잰 마진의 상당 부분은 “기울기가 이겼다”가 아니라 “고전 설계가 빗나갔다”였습니다. 같은 메시를 주면 고전 스텁은 목표에서 {CAL['notch']:.1f} dB를 냅니다. <b>둘째, 같은 연산자로 평가했습니다</b> — 실제 형상으로 다시 풀자 두 자유형 설계의 순위가 뒤집혔습니다. 씨앗을 준 설계는 폭이 좁은 고Q 스텁이라 하드 형상에서 공진이 6.10 GHz로 밀렸고, 씨앗 없이 찾은 분산형 설계는 6.00 GHz에 그대로 앉았습니다.</p>
<p><b>남는 결론은 이렇습니다.</b> 전송선 이론을 전혀 주지 않고 랜덤 저밀도에서 출발한 자유형 탐색이 <b>사람이 유도한 고전 설계와 같은 수준</b>에 도달했습니다({XVC['contrast']:+.1f} 대 {CAL['contrast']:+.1f} dB). 확정 논문이 자유형 이진 금속을 “주요 미해결 한계”로 적어 둔 것을 생각하면 이것만으로도 선을 하나 넘은 것이지만, <b>능가했다는 주장은 철회</b>합니다.</p>
<p>덧붙일 관점 하나: 단일 주파수 노치는 유효 자유도가 하나뿐이고 닫힌 해가 존재하는 문제입니다. 2 256개 변수 탐색이 λ/4 스텁을 이길 수 없는 게 오히려 정상이고, 할 수 있는 최선은 그것을 <i>찾아내는</i> 것입니다. 기울기의 우위를 보이려면 닫힌 해가 없는 문제 — 다중 대역, 비대칭 제약, 면적·형상 제약이 걸린 배치 — 로 옮겨야 합니다.</p>

<h2><span class="n">06</span>한계 — 다음 단계에서 닫아야 할 것</h2>
<ul>
  <li><b>통과대역 삽입손실.</b> 독립 경로에서 잰 통과대역 평균은 {XVC['pb']:.1f} dB입니다(처음 보고한 −8 dB는 의도적으로 발산된 평면 추출기에서 나온 값이라 신뢰할 수 없습니다). 실용 필터의 −1 dB급과는 여전히 거리가 있어, 통과대역 가중치를 올려 스윕하는 것이 다음 실험입니다.</li>
  <li><b>비교 설계가 문제를 고르는 방식.</b> 닫힌 해가 있는 문제에서는 동등이 최선입니다. 다중 대역·비대칭·면적 제약처럼 고전 합성법이 답을 주지 못하는 문제로 옮겨야 기울기의 실질 이득이 드러납니다.</li>
  <li><b>메쉬 전이성 미검증.</b> 방금 T-MTT 논문에 “표준 설계 절차”로 써넣은 2단계 검사(2배 세분화 메쉬 재평가 → 실패 시 프로덕션 해상도에서 재최적화)를 이 결과에 그대로 적용해야 합니다.</li>
  <li><b>추출기 절대 스케일</b>은 교차검증 경로에서 해소됐습니다(빈 선로가 0.00 dB). 다만 최적화 루프 안의 수치는 여전히 정규화 비교로만 유효합니다.</li>
  <li><b>AD-vs-FD 기울기 검증은 방법론적으로 미결.</b> 합성 목적함수에서 유한차분 잡음이 커서 단순 2점 차분으로는 판정되지 않습니다. 방향미분 리처드슨 설계가 필요합니다. 하강 증거(매끈한 100배 감소)는 기울기가 <i>유용</i>함을 말하지만 <i>정확도 수치</i>로 인용할 수는 없습니다.</li>
  <li>지오메트리 1종, 잡음 시드 1종. C-균일 잡 하나는 하강 도중 인프라 사유로 중단되어 재실행 대상입니다.</li>
</ul>
<p>다음 단계는 닫힌 해가 없는 설계 문제로의 이동, 통과대역 가중치 스윕, 메시 전이성 게이트(필터 반경을 셀이 아닌 mm로 고정하는 수정 포함), 외부 솔버(openEMS) 2단계 교차검증, 그리고 이 캠페인을 근거 자료로 삼은 <b>rfx 코어 패치 제안</b>입니다. 코어 패치의 1순위는 위상최적화 API가 금속을 전경 물질로 쓸 때 <b>기본값이 여전히 옛 감쇠 경로</b>라는 점입니다.</p>

<footer>
  코드·결과·판정 노트: <code>bk-squared/rfx</code> 브랜치 <code>feat/metal-topology-ramp</code>,
  <code>research/metal_to/</code> — 프로브 판정 <code>NOTE_probe01_verdict.md</code>,
  단계별 판정 <code>NOTE_phase1a_verdict.md</code> · <code>NOTE_phase1c_verdict.md</code>.<br>
  실행 환경: VESSL <code>remilab-c0</code> RTX 4090 · 954k셀 격자 · 반복당 20–24초 · 150회 실행 1건당 60–90분.
  모든 판정 수치는 이진화(밀도&gt;0.5) 후 하드 PEC Kottke 평가기 · 30주기 창 · 9주파수 대역에서 재측정한 값입니다.
</footer>
</div>
<div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.addEventListener('mouseover',e=>{{
  const t=e.target.closest('[data-tip]');
  if(!t){{tip.style.opacity=0;return;}}
  tip.textContent=t.getAttribute('data-tip');tip.style.opacity=1;
}});
document.addEventListener('mousemove',e=>{{
  if(tip.style.opacity!=='1')return;
  const r=tip.getBoundingClientRect();
  let x=e.clientX+14,y=e.clientY-r.height-10;
  if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;
  if(y<8)y=e.clientY+18;
  tip.style.left=x+'px';tip.style.top=y+'px';
}});
</script>
</body></html>
"""

(HERE / "report.html").write_text(HTML, encoding="utf-8")
print(f"wrote {HERE/'report.html'}  ({len(HTML)/1024:.0f} KB)")
