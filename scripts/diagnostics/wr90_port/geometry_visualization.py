"""Generate geometry diagram for the two PEC-short runs (OpenEMS-canonical
+45 mm vs rfx-canonical +55 mm) and produce a wave-clean compliant HTML
report that supersedes the earlier per-freq oscillation page.

Includes the existing per-freq oscillation panel (loaded from out/data.npz)
and a new geometry diagram panel explaining the 10 mm placement difference.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).parent / "out"


def make_geometry_diagram(png_path: Path) -> None:
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.5, 6.4),
                                         gridspec_kw=dict(height_ratios=[1, 1]))

    # Domain extents in OpenEMS frame
    x_min, x_max = -100.0, +100.0
    pml_mm = 20.0
    src_x = -60.0
    mon_l = -50.0
    mon_r = +50.0
    pec_canonical = +45.0   # Meep + OpenEMS + (post-fix) rfx
    pec_rfx_old = +55.0     # rfx pre-fix (PORT_RIGHT_X − 5 mm)
    pec_thick = 2.0

    # ---- Panel 1 (top): BEFORE — rfx convention drift ----
    track_h = 0.8
    y_oe = 1.0
    y_rfx = -0.1

    for ax in (ax_top, ax_bot):
        for y in (y_oe, y_rfx):
            ax.add_patch(Rectangle((x_min, y), pml_mm, track_h,
                                   color="#e0e0e0", lw=0))
            ax.add_patch(Rectangle((x_max - pml_mm, y), pml_mm, track_h,
                                   color="#e0e0e0", lw=0))
            ax.add_patch(Rectangle((x_min, y), x_max - x_min, track_h,
                                   fill=False, edgecolor="#888", lw=0.7))

        # Source + measurement planes (same in all configs)
        for label, x_pos, color, ls in [
            ("source\n−60", src_x, "#c52", "-"),
            ("mon_left\n−50", mon_l, "#27a", "--"),
            ("mon_right\n+50", mon_r, "#27a", "--"),
        ]:
            for y in (y_oe, y_rfx):
                ax.plot([x_pos, x_pos], [y, y + track_h], color=color,
                        lw=2, ls=ls)
            ax.text(x_pos, y_oe + track_h + 0.05, label, ha="center",
                    va="bottom", fontsize=7.5, color=color)

        ax.text(x_min + 2, y_oe + track_h / 2,
                "Meep / OpenEMS", va="center", ha="left", fontsize=9,
                weight="bold", color="#333")
        ax.text(x_min + 2, y_rfx + track_h / 2,
                "rfx", va="center", ha="left", fontsize=9, weight="bold",
                color="#333")

    # Top panel: BEFORE
    ax_top.add_patch(Rectangle((pec_canonical, y_oe), pec_thick, track_h,
                               color="#222", lw=0))
    ax_top.text(pec_canonical + pec_thick / 2, y_oe - 0.15,
                f"PEC +{pec_canonical:.0f}",
                ha="center", va="top", fontsize=8, weight="bold", color="#222")
    ax_top.add_patch(Rectangle((pec_rfx_old, y_rfx), pec_thick, track_h,
                               color="#a02", lw=0))
    ax_top.text(pec_rfx_old + pec_thick / 2, y_rfx - 0.15,
                f"PEC +{pec_rfx_old:.0f} (rfx drift)",
                ha="center", va="top", fontsize=8, weight="bold", color="#a02")
    ax_top.annotate("", xy=(pec_rfx_old, y_rfx + track_h - 0.05),
                    xytext=(pec_canonical, y_rfx + track_h - 0.05),
                    arrowprops=dict(arrowstyle="<->", color="#a02", lw=1.6))
    ax_top.text((pec_canonical + pec_rfx_old) / 2, y_oe - 0.65,
                "10 mm convention drift\n(rfx anchored to PORT_RIGHT_X +60, "
                "not mon_right +50)",
                ha="center", fontsize=9, color="#a02", weight="bold")
    ax_top.set_title("BEFORE — rfx drift: same boilerplate, "
                     "different anchor plane",
                     fontsize=11, color="#a02")

    # Bottom panel: AFTER (unified)
    for y in (y_oe, y_rfx):
        ax_bot.add_patch(Rectangle((pec_canonical, y), pec_thick, track_h,
                                   color="#222", lw=0))
        ax_bot.text(pec_canonical + pec_thick / 2, y - 0.15,
                    f"PEC +{pec_canonical:.0f}",
                    ha="center", va="top", fontsize=8, weight="bold",
                    color="#222")
    ax_bot.text((x_min + x_max) / 2, y_oe - 0.65,
                "AFTER (today's fix in crossval/11.PEC_SHORT_X) — "
                "all three identical",
                ha="center", fontsize=10, color="#0a6", weight="bold")
    ax_bot.set_title("AFTER — unified canonical geometry: rfx == Meep == OpenEMS",
                     fontsize=11, color="#0a6")

    for ax in (ax_top, ax_bot):
        ax.set_xlim(x_min - 5, x_max + 5)
        ax.set_ylim(-1.0, y_oe + track_h + 0.6)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.25)
    ax_bot.set_xlabel("x [mm]  (OpenEMS frame; rfx_x = openems_x + 100)",
                      fontsize=10)

    # Legend (custom) on bottom panel
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#c52", lw=2, label="excitation / port primitive"),
        Line2D([0], [0], color="#27a", lw=2, ls="--", label="measurement / reporting plane"),
        Rectangle((0, 0), 1, 1, color="#222", label="PEC short (canonical +45 mm)"),
        Rectangle((0, 0), 1, 1, color="#e0e0e0", label="CPML / PML"),
    ]
    ax_bot.legend(handles=handles, loc="lower right", fontsize=8,
                  framealpha=0.95)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[geom] {png_path}")


def write_wave_clean_html(html_path: Path, geom_png: str, osc_png: str,
                          osc_summary: list[tuple[str, float, float, float]]) -> None:
    rows = "\n".join(
        f'<tr><td><code>{name}</code></td><td><code>{mean:.5f}</code></td>'
        f'<td><code>{minv:.5f}</code></td><td><code>{maxv:.5f}</code></td>'
        f'<td><code>{maxv-minv:.5f}</code></td></tr>'
        for name, mean, minv, maxv in osc_summary
    )
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="robots" content="noindex, nofollow">
  <title>WR-90 port-extractor 진단 — REMI Lab</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://remilab.cnu.ac.kr/share/assets/remilab-base.css">
</head>
<body>
  <div class="container">
    <h1>WR-90 port-extractor 진단: per-frequency oscillation + geometry 비교</h1>
    <p><small>REMI Lab — 2026-04-28 — rfx Phase 1A.1 follow-up</small></p>

    <h2>1. Per-frequency oscillation 이 정확하게 무엇인가</h2>
    <div class="card">
      <p>WR-90 도파관(a=22.86mm, b=10.16mm), 8.2–12.4 GHz, 21점. PEC short 종단:
      입사 모든 전력이 반사 → 모든 f에서 <code>|S11(f)| = 1.000</code> 이어야 함.
      이 평탄 라인에서의 ripple = "per-frequency oscillation".</p>

      <div class="alert alert-info">
        <div class="alert-title">왜 mean이 아니라 spread가 중요한가</div>
        <div class="alert-body">mean offset은 단일 normalization 상수로 흡수 가능.
        spread (max − min)는 freq-dependent → 단일 상수로는 못 잡음.
        반드시 (i) V/I overlap integral, (ii) wave-decomposition Z(f), (iii)
        E vs H reference-plane offset 중 하나에 freq-dependent 오차가 있다는 뜻.</div>
      </div>

      <p><img src="{osc_png}" alt="WR-90 PEC-short |S11(f)|"></p>

      <table>
        <thead>
          <tr><th>configuration</th><th>mean |S11|</th><th>min</th><th>max</th><th>spread</th></tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>

      <p><strong>(A) DROP-both canonical</strong>
      <span class="badge badge-pass">production</span>:
      mean이 1에서 살짝 어긋나지만 frequency-dependent ripple은 sub-1e-3.
      Meep-class.</p>
      <p><strong>(B) Phase 1A.1 KEEP+E/H align</strong>
      <span class="badge badge-warn">unidentified residual</span>:
      mean은 정확히 1이지만 spread가 ±3–7%로 폭발. 이게 미해결 6%.</p>
    </div>

    <h2>2. Geometry 정합 — convention drift 발견 + 수정</h2>
    <div class="card">
      <p>이전 진단에서 rfx와 OpenEMS의 PEC short 위치가 10 mm 어긋남이 드러남.
      Meep와 OpenEMS는 <code>PEC = mon_right − 5 mm</code>로 <code>+45 mm</code>에 둠.
      rfx는 같은 boilerplate를 쓰되 anchor가 <em>port primitive 평면</em>
      (<code>PORT_RIGHT_X = +60 mm OE</code>)이라 <code>+55 mm</code>에 들어가 있었음.
      측정/보고 평면(<code>±50 mm</code>)은 두 쪽 모두 동일.</p>

      <div class="alert alert-warn">
        <div class="alert-title">의도된 설계 차이가 아니라 convention 충돌</div>
        <div class="alert-body">두 reference 모두 동일한 표현 "right port의 5 mm 앞"을
        쓰지만, OpenEMS+Meep는 "right port" = <em>측정 평면</em> = <code>+50 mm</code>,
        rfx는 "right port" = <em>port primitive 평면</em>(셀 중심 좌우대칭으로 잡힌
        <code>+60 mm</code>)으로 해석 → 결과적으로 같은 단어가 다른 평면을 가리킴.
        <br><br>
        <strong>오늘 수정</strong>: <code>examples/crossval/11_waveguide_port_wr90.py</code>에
        명시적으로 <code>MON_RIGHT_X = 0.150</code>, <code>PEC_SHORT_X = MON_RIGHT_X − 0.005
        = 0.145 m</code> 상수 도입. <code>run_rfx_pec_short</code>가 이 상수를 사용.
        rfx · Meep · OpenEMS 모두 <code>+45 mm OE</code>로 정합.</div>
      </div>

      <p><img src="{geom_png}" alt="WR-90 geometry: before (drift) vs after (unified)"></p>

      <p>위 그림 위쪽 패널은 수정 전 상태, 아래쪽은 수정 후 통일된 canonical.
      <span class="badge badge-pass">unified</span>
      세 시뮬레이터의 cell, source, mon, PEC short, PML 두께, fwidth, freq grid가
      모두 동일.</p>

      <h3>수정 후 |S11| 재측정 (DROP-both, Phase 1A.1)</h3>
      <p>Geometry를 정합해도 Phase 1A.1의 ±6% per-freq oscillation은 거의 동일.
      이는 oscillation이 PEC 위치-독립이며 진짜로 extractor/source-side 알고리즘
      bug라는 것을 의미함. 위 §1 표가 수정 후 값임 (DROP-both spread 4.5e-4,
      Phase 1A.1 spread 6.1e-2).</p>
    </div>

    <h2>3. VESSL job (정합 후)</h2>
    <div class="card">
      <p>이전에 띄운 두 job(<code>369367235573</code>, <code>369367235574</code>)은
      모두 terminate. 통일된 canonical geometry 단일 job으로 재제출:</p>

      <table>
        <thead>
          <tr><th>job</th><th>geometry</th><th>상태</th></tr>
        </thead>
        <tbody>
          <tr><td><code>369367235573</code></td>
              <td>OpenEMS canonical (PEC +45)</td>
              <td><span class="badge badge-fail">terminated</span></td></tr>
          <tr><td><code>369367235574</code></td>
              <td>rfx-aligned (PEC +55)</td>
              <td><span class="badge badge-fail">terminated (was wrong)</span></td></tr>
          <tr><td><code>369367235575</code></td>
              <td>unified canonical (PEC +45, == rfx PEC_SHORT_X)</td>
              <td><span class="badge badge-pass">running</span></td></tr>
        </tbody>
      </table>

      <p>R=1, 2, 4 PEC-short device runs + raw Yee staggered E/H frequency-domain
      HDF5 dumps at source plane(−60 mm)와 mon_left plane(−50 mm).
      예상 wall ~2–5 h. URL:
      <code>https://vessl.ai/remilab/runs/byungkwan/369367235575</code></p>
    </div>

    <h2>4. 다음 단계</h2>
    <div class="card">
      <p>Job 369367235575 완료 후:</p>
      <ol>
        <li><code>vessl run download</code> 또는 mwe FS에서 dump 회수</li>
        <li><code>compare</code> 모드로 R=1, 2, 4 실행 (geometry 통일되어 단일 비교)</li>
        <li>peak <code>|E_z|</code>, <code>|H_y|</code>, <code>|V|</code>,
            <code>|I|</code>, <code>arg(V)</code> per-freq 비교 → 첫 divergence step 식별</li>
        <li>해당 step에 표적 fix 설계</li>
        <li>(선택) 같은 geometry로 Meep field-dump 추가하여 triple cross-validation</li>
      </ol>
    </div>

    <div class="footer">REMI Lab — Generated by AI Agent · rfx commit chain ecdd845..a005524</div>
  </div>
</body>
</html>
"""
    html_path.write_text(html)
    print(f"[html] {html_path}")


def main() -> None:
    geom_png = OUT / "wr90_geometry_comparison.png"
    make_geometry_diagram(geom_png)

    data = np.load(OUT / "data.npz")
    s11_a = np.asarray(data["s11_canonical"])
    s11_b = np.asarray(data["s11_phase1a1"])
    summary = [
        ("(A) DROP-both canonical [production]",
         float(s11_a.mean()), float(s11_a.min()), float(s11_a.max())),
        ("(B) Phase 1A.1 KEEP+E/H align",
         float(s11_b.mean()), float(s11_b.min()), float(s11_b.max())),
        ("Ideal (PEC-short)", 1.0, 1.0, 1.0),
    ]
    write_wave_clean_html(
        OUT / "index.html",
        geom_png="wr90_geometry_comparison.png",
        osc_png="wr90_pec_short_per_freq_oscillation.png",
        osc_summary=summary,
    )


if __name__ == "__main__":
    main()
