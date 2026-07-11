# 재현 가이드 — X-band inset-fed patch 3-솔버 crossval

학생용. 이 문서만 보고 rfx / openEMS / Palace 세 솔버의 시뮬레이션을 그대로 재현해서
`DESIGN_DOC_patch_student.md`의 기준 수치와 비교할 수 있게 쓰였다.

- 설계/기하 파라미터·과제 정의: `DESIGN_DOC_patch_student.md` (랩 공유본: `/root/workspace/lab-shared/rfx-patch-crossval/DESIGN_DOC.md`)
- 기준 수치·판정 근거: `verdicts.json` + `evidence/` (이 디렉토리)
- 이 레인의 성격: **research-only** — 검증 증거를 주장하지 않는다 (`README.md` 참조)

## 0. 코드 받기

두 방법 중 하나. 공개 GitHub 브랜치가 정본이다.

```bash
# 방법 A (권장): 공개 브랜치 클론
git clone -b research/calibration-inverse https://github.com/bk-squared/rfx.git
cd rfx/scripts/research/calibration/crossval

# 방법 B: 랩 NFS 스냅샷 (편의용 — repo가 정본, 이쪽은 뒤처질 수 있음)
ls /root/workspace/lab-shared/rfx-patch-crossval/repro-kit/
```

공통 환경: Python 3.10, numpy 2.x, **JAX x64 OFF** (rfx 코어는 설계상 float32/complex64 —
`JAX_ENABLE_X64`를 켜면 오히려 크래시함).

## 1. 재현 대상 — 두 레벨

같은 공통 박스(x [-12, 18.595] / y ±15.06 / z [0, 10.787] mm, 접지 = z_lo PEC 경계면,
RO4003C εr=3.38 h=0.787 mm, patch L8.595×W10.129 mm, inset 2.4 mm)에서:

| 레벨 | 관측량 | rfx | openEMS | Palace | 스프레드 |
|---|---|---|---|---|---|
| **1. 차폐 공진** (전면 PEC 박스) | 기본모드 f₀ | **9.131 GHz** (Harminv) | **9.194 GHz** (ring-down) | **9.199 GHz** (eigenmode) | 0.7% |
| **2. 방사 S11** (개방 경계) | dip 주파수 | **9.250 GHz** (−7.04 dB, dx=197 µm) | **9.2625 GHz** (−9.46 dB) | 9.05 GHz (−12.99 dB, caveat 참조) | — |

읽는 법 (중요):
- **dip "주파수"만 비교 대상**이다. dip **깊이(-dB)는 솔버 간 비교 불가** — 손실 모델이
  통일되어 있지 않다 (FDTD 두 개는 무손실 관례, 실보드 tanδ≈0.0027). 그리고 dip 주파수도
  mesh-limited + 불안정 argmin이라 **절대 합격/불합격 게이트로 쓰지 말 것**
  (`tests/test_issue80_patch_s11_regression.py`의 문서화된 한계).
- Palace driven 9.05 GHz는 타이트한 1차 흡수경계가 dip을 끌어내린 값 — 같은 Palace의
  차폐 eigenmode는 9.199 GHz로 합의쪽이다.
- 재현 성공의 기준: 레벨-1 수치가 표의 ±1% 이내, 레벨-2 곡선 모양(단일 null, 대역외 평탄)이
  `evidence/` 곡선과 눈으로 겹치면 성공.

## 2. rfx 레그 (FDTD, JAX)

```bash
pip install -e .        # 클론 루트에서 (또는 pip install rfx-fdtd)
cd scripts/research/calibration/crossval

# 레벨 2: 방사 S11 곡선 (기본 dx=197um, 7-12 GHz 201점)
python rfx_patch_xband_canonical_frame.py --mode s11

# 레벨 1: 차폐 공진 (Harminv ring-down)
python rfx_patch_xband_canonical_frame.py --mode shielded-harminv

# (참고) 개방 프레임 Harminv
python rfx_patch_xband_canonical_frame.py --mode harminv
```

- 출력: `out_canonical_frame/*.json` (freqs_hz + 복소 S11 / Harminv 모드 리스트).
- CLI: `--dx-um` (기본 197), `--num-periods` (기본 200), `--nfreq` (기본 201), `--output`.
- **GPU 권장**: dx=197 µm, 200 period면 CPU에서 매우 오래 걸린다. 레인에 커밋된
  VESSL YAML을 쓰면 된다 (`run:` 블록의 `cd` 한 줄만 본인 클론 경로로 수정 —
  `/root/workspace`가 NFS 마운트라 클론이 잡에 그대로 보임, push 불필요):

```bash
vessl run create -f scripts/research/calibration/crossval/gpu/canonical-frame.vessl.yaml
# cluster remilab-c0 / preset gpu-rtx4090 / image nvcr.io/nvidia/jax:24.10-py3
```

- 함정: preflight 경고가 나오면 **그대로 기록**할 것 (경고 포함이 결과다). dx를 줄이면
  (98.5 µm) max|S11|=1.36 비수동 아티팩트가 알려져 있음 — OPEN 이슈, verdicts.json 참조.

## 3. openEMS 레그 (FDTD, 독립 구현)

openEMS 0.0.35 + python 바인딩. numpy 2.x용 `np.float=float` shim은 **스크립트 안에
이미 들어 있음** (37행) — 따로 할 것 없다.

```bash
# 레벨 2: 방사 S11 (박스/메시를 rfx 프레임에 맞춘 재현)
python openems_patch_inset_xband.py --inset-mm 2.4 --margin-y 10

# 레벨 1: 차폐 ring-down (S11은 정의 안 됨 — Harminv 모드만 유효)
python openems_patch_inset_xband.py --inset-mm 2.4 --margin-y 10 --shielded

# 죽은/이전 런의 포트 덤프만 후처리 (FDTD 재실행 없이)
python openems_patch_inset_xband.py --inset-mm 2.4 --margin-y 10 --calc-only
```

- 출력: `out_xband/openems_xband_inset_2.4mm*.json` — `s11_passive` PASS/FAIL 포함
  (1.05 per-bin slack; `--shielded`는 S11이 정의 안 되므로 null).
- 메시 수렴 확인은 `--res-scale 0.5` (셀 반감; 9.347 GHz로 위쪽 이동 확인됨).
- 함정: 포트 전압 파일(`port_ut_*`) 직접 읽을 땐 `np.loadtxt(..., comments='%')`.
  CPU 경합 시 3.7~17 MC/s로 편차 큼 — 수십 분 걸릴 수 있다.

## 4. Palace 레그 (FEM, 3번째 심판)

랩 공유 spack 빌드가 NFS에 있다 (직접 빌드 불필요):

```bash
PALACE=/root/workspace/bk-workspace/_spack/opt/spack/linux-zen3/palace-0.16.0-pbs4byncyg5yfp42z5axhsv4csx342tm/bin/palace

cd palace_patch
python mesh_patch.py            # Gmsh OCC 메시 생성 (~10k 노드 / 54k tets, msh v2.2)

# root 컨테이너에서 OpenMPI 5는 아래 env 두 개가 필요
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

$PALACE -np 16 patch_s11.json            # 레벨 2: driven S11 (8-12 GHz)
$PALACE -np 16 patch_eigen_shielded.json # 레벨 1: 차폐 eigenmode
```

- 출력: `postpro/patch_s11/port-S.csv`, `postpro/patch_eigen_shielded/eig.csv`.
- 함정 (빌드를 다시 할 경우에만): cmake ≥3.24 필요(spack이 처리, external cmake 금지),
  MFEM의 Gmsh 리더가 msh 4.1을 거부 → `mesh_patch.py`가 이미
  `removeDuplicateNodes()` + `Mesh.MshFileVersion=2.2`로 우회해 둠.
- 손실 관례: `patch_s11.json`은 LossTan=0.0 (FDTD 무손실 레그와 통일, #273 correction).
  eigen config 두 개는 기록된 런이 LossTan=1e-3이었음(주석 참조) — 재실행하면 0.0으로.

## 5. 비교·제출

```bash
# 세 곡선 오버레이 (rfx JSON + openEMS JSON + Palace CSV).
# 인자 없이 돌리면 커밋된 evidence/ 기준곡선으로 blessed 플롯을 재현하고,
# --rfx/--openems/--palace 로 본인 결과 파일을 끼워 넣어 비교한다.
python plot_3solver_s11.py
python plot_3solver_s11.py --rfx out_canonical_frame/rfx_cf_s11_dx197um.json \
    --openems out_xband/openems_xband_inset_2.4mm.json \
    --palace palace_patch/postpro/patch_s11/port-S.csv
```

(주의: `compare_patch_s11.py`는 X-band용이 아니라 **cv05 2.4 GHz FR4 patch** 레인의
비교기다 — 헷갈리지 말 것.)

- 기준 데이터(우리 세 곡선): `/root/workspace/lab-shared/rfx-patch-crossval/`
  (rfx/openEMS JSON, Palace CSV, 종합 플롯 PNG). **경로는 반드시
  `/root/workspace/lab-shared/...` 실경로로** — `/lab-shared` 심링크는 컨테이너에 따라 없다.
- 제출: 결과 JSON/CSV + 곡선 플롯 + 실행 로그(preflight 경고 포함)를
  `/root/workspace/lab-shared/rfx-patch-crossval/student/` 아래 본인 이름 디렉토리로.
- CST/실측(VNA) 과제는 `DESIGN_DOC_patch_student.md`의 Task 브리프를 따른다. 측정
  S11은 이후 `rfx.differentiable_material_fit.calibrate_material_s11`(브랜치의
  `fit_nuisance=True` 경로)로 기판 εr/tanδ 역추정에 들어간다 — 그게 이 실습의 연구 목적.

## 6. 하지 말 것 / 알려진 함정 요약

| 함정 | 처방 |
|---|---|
| JAX_ENABLE_X64=true | 금지 — rfx 코어 f32 설계, scan carry dtype 크래시 |
| dip 주파수를 게이트로 | 금지 — mesh-limited + 불안정 argmin (issue80 문서화) |
| dip 깊이 솔버 간 비교 | 금지 — 손실 모델 미통일 (무손실 관례 vs 실보드 tanδ 0.0027) |
| Meep을 3번째 솔버로 | 기각됨 — serial 빌드 해상도 한계로 KILL (verdicts.json) |
| 1-포트 라인을 CPML에 박은 thru | 기각됨(PR#272 오배선) — 정식은 `scripts/diagnostics/msl_thru_mesh_convergence.py` (양단 msl 포트) |
| `/lab-shared` 경로로 문서화 | 금지 — 심링크 부재 시 로컬 유령 디렉토리 생성됨; NFS 실경로 사용 |
| openEMS `--shielded`의 S11 | 무의미 (닫힌 박스) — Harminv ring-down 모드만 사용 |
