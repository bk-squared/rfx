# rfx Agent Memory — Durable Knowledge Index

## Read first

- `public-site-sync-checklist.md`
- `nu_known_limits.md` — non-uniform mesh path: soft-PEC and mesh-as-design-variable gradient sentinels
- `nu_memory_efficient_stack.md` — A+C+segmented path decisions and the right `checkpoint_every` for FDTD inverse design

## Session Handover (2026-04-12)

### GPU Validation Results

#### v1.5.0 (VESSL #369367233081, RTX 4090) — 2026-04-12

| Phase | Result |
|-------|--------|
| Fast CI (non-slow) | **719 passed, 3 failed**, 1 skipped (78 min) |
| Gradient + optimization | **43 passed** (2:51) |
| Non-uniform mesh | **31 passed** (0:34) |
| Crossval 12 (patch antenna) | **Harminv PASS** (1.59% error), OpenEMS skipped (no CSXCAD) |
| Crossval 14 (inverse design) | **PASS** (loss 2.25x, εr 3.49→3.23) |

#### 3 Remaining Failures (all experimental lane, unchanged from v1.4.0)

| Test | Error | Lane |
|------|-------|------|
| `test_ris::test_ris_sweep_capacitance` | ValueError: Floquet unsupported | Experimental (Floquet) |
| `test_ris::test_ris_sweep_angle` | ValueError: Floquet unsupported | Experimental (Floquet) |
| `test_sbp_sat_alpha::test_init_subgrid_3d_default_tau` | 3D subgrid | Experimental (SBP-SAT) |

**Verdict**: Reference lane is clean. All 3 failures are in experimental lanes.
v1.4.0 → v1.5.0: **9 failures → 3 failures** (6 fixed: gradient NameError, optimize contract, Fresnel, TFSF oblique, topology gradient).

### Test Infrastructure (2026-04-12)

- Added `pytest.mark.gpu` marker to 28 test files (214 tests)
- GPU validation YAML: `pytest -m gpu` + crossval only (~30 min vs 78 min)
- GitHub CI: `pytest -m "not gpu"` (570 code tests, CPU-only)
- VESSL: `scripts/vessl_v150_gpu_validation.yaml`

---

### Repo Cleanup (2026-04-10)

6 commits pushed to main:

1. `bad2446` — fix: pyproject.toml explicit package discovery (forge_example 제외)
2. `3f7091a` — chore: 56 stale example files 삭제 (01-09, 40_*, 50_*, crossval 01-23)
3. `ac1b8c2` — feat: 6 new crossval diagnostics
4. `2d39cf7` — docs: support matrix + reference-lane contract
5. `130798f` — docs: lane annotations (README, public docs, agent overview)
6. `1d81190` — chore: gitignore forge_example/

---

### Crossval Examples 현황 (2026-04-12 정리)

| # | File | Reference | Validates | Status |
|---|------|-----------|-----------|--------|
| 01 | `01_waveguide_bend.py` | Meep | T(f), self-T, \|rfx-Meep\| < 0.10 | A (valid) |
| 02 | `02_ring_resonator.py` | Meep | Harminv mode freqs, mean error < 5% | A (PASS/FAIL added) |
| 03 | `03_straight_waveguide_flux.py` | Meep | Flux shape correlation > 0.90 | A (PASS/FAIL added) |
| 04 | `04_multilayer_fresnel.py` | Analytic + Meep | TFSF R/T, mean error < 5% | A (valid) |
| 05 | `05_patch_antenna.py` | OpenEMS + Analytic | Patch resonance, rfx vs OpenEMS < 20% | A (valid) |
| 06 | `06_msl_notch_filter.py` | OpenEMS ref | MSL notch freq < 15% | B (weak threshold) |
| 07 | `07_inverse_design_demo.py` | None (self-test) | Gradient pipeline converges | B (demo) |

**Deleted (2026-04-12):** 8 scripts removed — diagnostic artifacts (01a, 02, 03, 04, 08a) and broken self-tests (24 GPR, 25 horn, 26 BPF) that had no external reference and meaningless thresholds.

**Rule:** Crossval scripts MUST compare against an external reference (Meep, OpenEMS, analytic) with quantitative PASS/FAIL thresholds < 10%. Self-tests and diagnostics do not belong in `crossval/`.

---

### 남은 작업 (우선순위 순)

#### P0 — Remaining Test Failures (3, all experimental lane)
- [ ] `test_ris` Floquet ×2 — Floquet port single-device 제약 (설계 제약, 문서화됨)
- [ ] `test_sbp_sat_alpha` 3D — SBP-SAT 3D 미검증 (experimental)

#### P1 — Known Issues
- [ ] `test_floquet::test_unit_cell_with_floquet` — pre-existing NaN (Floquet+NU 비호환)
- [ ] Far-field dS per-face for non-uniform z — audit item #9

#### P2 — Quick Wins
- [ ] PyPI 배포 확인 (v1.5.0 버전 범프 완료, 배포 필요)
- [ ] Crossval 24-26 GPU 검증 (GPR, horn, coupled-line BPF)

#### P3 — Advanced (deferred)
- [ ] Auto-subgrid (AMR indicator → subgrid)
- [ ] Level-set topology optimization
- [ ] Neural surrogate pipeline (사용자 요청 시)

---

### 핵심 설계 원칙

1. **Physical absolute coordinates**: Probe/port/source 위치는 항상 물리 절대 좌표
2. **Axis-aware formulas**: `d_parallel/(Z0·d_perp1·d_perp2)` — cubic cell 가정 금지
3. **Duck typing for grid types**: `getattr(grid, 'dy', dx)` 패턴
4. **JIT safety**: Cell sizes는 Python float로 추출 → NamedTuple
5. **Evidence before defaults**: 파라미터 변경은 sweep 실측 후 결정

---

### Support Lane Model (2026-04-10 도입)

| Lane | Status | Docs |
|------|--------|------|
| Uniform Cartesian Yee | **Reference** (claims-bearing) | `docs/guides/reference_lane_contract.md` |
| Non-uniform graded-z | Shadow | `docs/guides/support_matrix.md` |
| Distributed multi-GPU | Experimental | `docs/guides/support_matrix.md` |
| SBP-SAT subgridding | Experimental | `docs/guides/support_matrix.md` |
| Floquet/Bloch | Experimental | `docs/guides/support_matrix.md` |

---

### Previous Session (2026-04-07) — Archived

<details>
<summary>Session 2 summary (click to expand)</summary>

#### Completed
1. Non-uniform runner 15% error 해결 (center probe → edge probe)
2. Port sigma fix (anisotropic cell): `d_parallel/(Z0·d_perp1·d_perp2)`
3. CPML axis-aware refactor: `CPMLAxisParams` 4-profile
4. Port/Source sigma 통합: shared helper
5. Geometry rasterization 통합: `rfx/geometry/rasterize.py`
6. Physical absolute coordinates 통일
7. PR #24 close

#### VESSL Runs
- 369367232429: CPML reflectivity sweep — ALL PASS
- 369367232419: Physics validation — edge probe 일치

#### Axis-Dependent Audit (12건)
- #1-8, 10-12: Fixed
- #9: Far-field dS — Pending

</details>
