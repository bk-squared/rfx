# X-band Inset-Fed Patch — 제작 · CST 교차검증 · VNA 측정 Design Doc

**작성**: Prof. BK / REMI Lab (2026-07-07) · **담당 학생**: TBD
**목적**: 아래 patch 안테나 하나를 (1) CST로 재현(4번째 솔버), (2) 제작, (3) VNA로 측정한다.
이 결과는 단순 실습이 아니라 **미분가능 FDTD(rfx) 캘리브레이션 연구의 실측 앵커**다 — 우리는 이미 rfx(FDTD) + openEMS(FDTD) + Palace(FEM) 3개 솔버로 이 설계를 교차검증했고, 남은 열린 질문들을 **너의 CST와 측정이 판정**한다.

---

## 1. 설계 (모든 치수 mm)

**기판**: Rogers **RO4003C**, 두께 **h = 0.787** (31 mil), εr(공칭) = 3.38, tanδ ≈ 0.0027 @10 GHz.
동박 1 oz (35 µm), 양면. **아래면 전체 = 접지** (에칭 없음).

**상면 금속 (단층)**:

```
             y ↑
              │      ┌────────────────────────────┐  ─┐
              │      │                            │   │
              │  ┌───┘ ← notch slot (g=0.90)      │   │ W = 10.129
 feed line    │  │                                │   │  (y 방향)
 w = 1.80 ────┼──┤    patch body                  │   │
 (50 Ω)       │  │                                │   │
              │  └───┐ ← notch slot (g=0.90)      │   │
              │      │                            │   │
              │      └────────────────────────────┘  ─┘
              └──|—————|———————— L = 8.595 ————————|→ x
              inset d = 2.40      (x 방향 = 공진 방향)
```

| 파라미터 | 값 | 비고 |
|---|---|---|
| Patch L (공진, x) | **8.595** | 급전 방향 |
| Patch W (y) | **10.129** | |
| 50Ω 급전선 폭 w | **1.80** | |
| Inset 깊이 d | **2.40** | 급전선이 patch 안으로 관통 |
| Notch 갭 g | **0.90** (양쪽) | 에칭 슬롯, 급전선 양옆 |
| 급전선 길이 | **≥ 12** (보드 에지→patch) | SMA까지 직선 |
| 보드 크기 | **≥ 31 × 31** (권장 35×35) | patch 주변 ≥10 여유 |
| 커넥터 | SMA **end-launch** (edge mount) | 급전선 끝, 보드 에지 |

주의: patch·급전선 위 **솔더마스크 없음**(bare copper 또는 ENIG). 실크는 금속에서 떨어뜨릴 것.

## 2. 시뮬레이션 예측 (3-솔버 교차검증 완료, 2026-07-07)

**Level-1 — 차폐(전면 PEC 박스) 공진** (형상 검증용 기준; 박스 = x 30.595 × y 30.129 × z 10.787, 접지=바닥면, 급전선은 x=0 벽에 단락):

| 솔버 | 기본 모드 | 2차 모드 |
|---|---|---|
| rfx (Harminv) | 9.131 GHz | 10.764 |
| openEMS (ring-down) | 9.194 GHz | 10.799 |
| Palace (FEM eigenmode) | 9.199 GHz | 10.797/10.806 |

**Level-2 — 방사 S11** (개방 경계):

| 솔버 | dip | 깊이 |
|---|---|---|
| rfx (197 µm) | **9.250 GHz** | −7.0 dB |
| openEMS | **9.262 GHz** | −9.5 dB |
| Palace (driven) | 9.05 GHz* | −13 dB* |

\* Palace는 흡수경계가 좁고 1차라 dip이 끌려 내려간 상태(차폐 eigen은 9.199로 합의) — 이것도 네가 판정할 항목.

**예상 측정값**: 공진 dip **9.2–9.3 GHz**, 깊이 **−7 ~ −13 dB** (중간 정합; εr·에칭 공차에 민감).

비교용 데이터(우리 세 곡선): `/root/workspace/lab-shared/rfx-patch-crossval/` — rfx/openEMS JSON(freqs_hz + 복소 S11), Palace CSV, 종합 플롯 PNG.

시뮬레이션 재현(3-솔버 전부, 단계별 명령·함정 포함): 같은 디렉토리의 `REPRODUCE.md` — 스크립트 일체는 `repro-kit/` 또는 공개 브랜치 `git clone -b research/calibration-inverse https://github.com/bk-squared/rfx.git`.

## 3. 과제 A — CST 재현 (제작 전 필수)

1. **Shielded eigenmode** (Level-1): 위 차폐 박스 치수 **그대로** (배경=PEC, 유전체 무손실 εr=3.38). Eigenmode solver, 7–12 GHz 모드.
   **판정**: 기본 모드가 9.13–9.20 GHz 대역에 들어오는가? → 형상 입력이 우리와 등가라는 확인. 안 맞으면 형상부터 대조(치수표 §1).
2. **방사 S11** (Level-2): open boundary(λ/4 이상 여유), waveguide port 또는 discrete port(50Ω)를 급전선 끝에. 7–12 GHz, adaptive mesh 수렴까지.
   **산출**: S11 Touchstone(.s1p) + dip 주파수/깊이. 우리 3곡선과 겹쳐 그릴 것.
3. (선택) tanδ=0.0027, 구리 손실 포함 버전 — 깊이가 어디로 가는지.

## 4. 과제 B — 제작

- **재료**: RO4003C 0.787mm (랩 재고 확인; 없으면 JLCPCB Rogers 옵션 또는 국내 업체).
- 단면 패턴(상면) + 하면 전체 접지. 도금 스루홀 불필요(급전은 마이크로스트립).
- SMA end-launch 납땜: 중심핀↔급전선, 플랜지/GND legs↔하면 접지. **납땜 최소·짧게**(기생 인덕턴스).
- 에칭 공차 목표 ±50 µm (L 방향 ±50 µm ≈ 공진 ∓0.5%).

## 5. 과제 C — VNA 측정

1. **SOLT 캘** (케이블 끝 = 기준면), 7–12 GHz, ≥801 포인트, IF BW ≤1 kHz.
2. 보드는 **폼/흡수체 위**, 금속 책상·손에서 λ 이상 이격 (방사 구조!).
3. `.s1p` 저장 (복소, RI 포맷 선호). 사진 포함(셋업·보드 근접).
4. 플롯: 측정 vs CST vs 우리 3곡선 (하나의 |S11| dB 그래프).

## 6. 이 측정이 판정하는 열린 질문 (연구 기여 지점)

1. **dip 깊이**: −7(rfx) vs −9.5(openEMS) vs −13(Palace) — 실제는?
2. **Palace의 9.05 vs FDTD의 9.25**: 흡수경계 처리 차이 가설 검증.
3. 측정 dip이 9.2–9.3을 벗어나면 → **εr/치수 공차의 지문** — 그대로 다음 단계 입력이 된다:
4. **캘리브레이션 실험(후속)**: 네 `.s1p`를 rfx의 미분가능 캘리브레이션 파이프라인(`calibrate_material_s11` + identifiability 분석)에 넣어 **이 보드의 실제 εr(f)/tanδ를 gradient로 역추정**한다. 이게 이 과제의 최종 목적.

## 7. 민감도 참고표 (측정 해석용)

| 요인 | 변화 | 공진 이동 |
|---|---|---|
| εr +0.05 | (공차 상한) | ≈ −0.7 % |
| L +50 µm | (과에칭 반대) | ≈ −0.5 % |
| h +5 % | | < ±0.3 % |
| inset d ±0.1 | | 깊이 변화 주도(주파수는 소폭) |
| SMA/납땜 | | 기준면 회전 + 리플; dip 위치엔 소폭 |

## 8. 산출물 체크리스트

- [x] CST shielded eigenmode 결과 (모드 표) ← 260712 완료
- [x] CST 방사 S11 `.s1p` + dip 값 ← 260712 완료 (양 포트)
- [ ] 제작 보드 사진 (상/하면, 커넥터)
- [ ] 측정 `.s1p` + 셋업 사진
- [ ] 종합 비교 플롯 1장 (측정+CST+rfx+openEMS+Palace)
- [ ] 한 페이지 메모: dip 주파수/깊이 표 + 예상-실측 차이에 대한 본인 해석

**질문/데이터 제출**: Prof. BK. 파일은 `/root/workspace/lab-shared/rfx-patch-crossval/student/` 아래에.

---

## 9. ★결과 갱신 (260712, 학생 HW — CST 4번째 솔버 완료)★

**Level-1 차폐 eigenmode**: CST **9.221 GHz** (2차 10.777) — 4-솔버 스프레드 1.0%로 형상 등가 확인.
단 adaptive pass 5에서 아직 +0.6%/pass 상승 중 + CST만 금속 t=35µm 모델링(나머지는 시트/1셀) →
9.13–9.20 원 판정대역보다 0.2% 위는 이 두 계통차 안. **보너스**: CST 모드 6.45/6.73/6.81 GHz가
rfx below-window 라인(6.627/6.791)을 실모드로 corroborate (verdicts.json level1 갱신).

**Level-2 방사 S11** (t=35µm, lossy tanδ 0.0027): discrete port **9.12 GHz / −20.15 dB**,
waveguide port **9.33 GHz / −12.09 dB** — 둘 다 passive (max|S11| 0.99).

**§6 열린 질문 판정** (verdicts.json `feed-model-dominance` 신규 행):
- **dip 깊이 질문 → 판정됨**: 동일 CST·동일 형상·포트만 교체 = +210 MHz / +8 dB. 깊이는
  솔버가 아니라 **급전 모델의 관측량** — 급전 유형 간 비교·게이트 금지.
- **Palace 9.05 vs FDTD 9.25 → 재구성**: lumped 급전(Palace LumpedPort, CST discrete)
  9.05–9.12 vs guided 급전(rfx msl, openEMS MSL, CST waveguide) 9.25–9.33.
  ABC 가설은 부차적 — **급전 모델이 지배항**. guided 클러스터 내부 곡선 일치
  mean|Δ|S11|| 0.5–0.9 dB (8–11.5 GHz).
- **실측 비교 지침**: SMA end-launch는 물리적 guided 급전 → 제작 보드 측정은
  **guided 클러스터(9.25–9.33)와 비교**할 것 (SMA 기준면 디임베드 후).

FFRP(방사패턴)도 제출됨: E/H-plane, broadside ~7 dBi, F/B ~15 dB — 양 포트 모델에서
패턴 거의 동일(패턴은 모드가 결정, 급전 민감도 낮음 — 이론과 정합).
데이터: `student/Radiation_pattern/` + s1p 사본 `evidence/cst_s11_*_hw260712.s1p`.
