# RFX 공개 문서 개편 계획: 시뮬레이션을 보여주고, 재현하고, API로 연결하기

작성일: 2026-09-27 · 상태: 제안 / 구현 전

조사 기준: RFX `a9054338616b91bb1364699882c2530fa29a0048`, 공개 사이트 실제 HTTP 응답과
데스크톱·모바일 화면, 기존 예제·API 도구, 공유 발표 자료, 현재 배포 저장소와 런타임.
이 문서는 문서 구조와 실행 순서를 제안한다. 새 시뮬레이션 결과나 정확도를 검증한 기록은 아니다.

## 1. 결정 제안

RFX 사이트를 **시각적 사례로 가능성을 발견하고 → 작은 예제를 실행하고 → 정확한 API 계약을 읽는**
동선으로 재구성한다. 기존 Astro/Starlight와 RFX 문서 원본을 유지한다.
사람용 설명과 LLM용 문서를 별도로 집필하지 않고 같은 버전의 코드·예제·지원 정보에서 생성한다.

첫 공개 묶음은 유전체 슬래브, WR-90 도파관/테이퍼, 마이크로스트립 노치 필터를 우선한다.
안테나 방사·빔 조향·산란을 후속 묶음으로 확장한다. 기존 패치 진단 갤러리는 대표 성공 사례로 쓰지 않는다.
내부 발표는 시각화와 설명 방식의 참고 자료로 사용하고, PEFM 연구 과제의 목표·성과·운영 문맥은 분리한다.

완료 기준은 페이지 수가 아니라 다음 세 가지다.

1. 첫 방문자가 대표 그림과 영상만으로 시뮬레이션 대상과 결과를 구분할 수 있다.
2. 사례 페이지의 실행 코드·그림·수치가 동일한 모델과 명시된 코드 버전을 가리킨다.
3. 사람과 에이전트가 같은 API signature, 지원 조건, 한계, 실행 예제를 찾는다.

## 2. 현재 구성의 장점과 실제 한계

기존 기반은 재사용할 가치가 있다. `docs/public/` 원본, 배포 exporter, API inventory,
페이지 코드 블록 실행기, gallery manifest, 예제 학습 순서, 지원 행렬이 이미 있다.
슬래브와 도파관 상세 페이지에는 GIF도 있다. 애니메이션이 전혀 없는 사이트는 아니다.

| 관찰 | 독자·유지보수에 미치는 영향 | 보완 방향 |
|---|---|---|
| 라이브 첫 화면과 Examples 허브 본문에 이미지·영상이 각각 0개 | 기능 표와 코드부터 읽어야 시뮬레이션의 모습을 상상할 수 있음 | 결과 중심 첫 화면, 시각적 사례 카드, 실행 예제로 직결 |
| 갤러리 3종은 모두 `1.6.5 / 1eb551b` 산출물 | 현재 문서와 과거 계산 결과의 적용 버전이 다름 | 과거 결과 표시는 유지하고 대표 사례를 목표 버전에서 재생성 |
| 패치 갤러리는 preflight를 통과하지 못한 포트 설정이며, 장 그림과 포트 결과도 서로 다른 실행 | 그림을 한 모델의 안테나 성능으로 연결할 수 없음 | 진단 자료로 보존, 새 대표 패치는 독립 재현·검증 후 등록 |
| 슬래브 갤러리 실행 설명에 저수준 Yee 업데이트와 생략된 루프가 남아 있음 | 입문자가 공개 `Simulation` API보다 내부 구현부터 만나게 됨 | 기존 `slab_rt_flux_monitor.py` 같은 고수준 예제에서 본문 코드 생성 |
| `site_map.json`은 guide 24개만 주 탐색 대상으로 허용하고 배포 쪽도 탐색 구조를 가짐 | API·Examples·Gallery의 위치가 여러 곳에서 결정됨 | 한 route registry에서 sidebar·허브·검색·LLM 색인을 파생 |
| API inventory 검사가 타입·기본값·반환값·단위까지 보장하지 않음 | 존재하는 함수도 잘못 호출하거나 반환 배열을 오해할 수 있음 | 구조적 signature 생성 + 지원 의미에 대한 소규모 수기 보완 |
| `docs/api/`는 generated/ignored인데 exporter는 tracked 파일만 복사 | pdoc를 생성해도 배포 산출물에 들어가지 않는 경로 | 고정 SHA로 생성한 API bundle을 명시적인 build 입력으로 수용 |
| `/rfx/llms.txt`, `/rfx/llms-full.txt`, `/rfx/api/inventory.json`은 조사 시 404 | 안정적인 기계용 진입점이 없음 | 작고 버전이 명확한 색인·Markdown·JSON 제공 |
| 설치는 PyPI, 본문은 개발 중 변경과 과거 결과를 함께 설명 | 설치된 패키지와 읽는 문서의 일치 여부를 알기 어려움 | stable/dev 분리, 태그·소스 SHA·빌드 식별자 노출 |
| README·landing·quickstart가 비슷한 코드를 별도로 유지 | 어느 한쪽만 수정되어 예제가 달라짐 | 하나의 실행 파일에서 필요한 구간을 생성 |
| 배포 CI가 RFX의 움직이는 기본 브랜치를 조회 | 같은 배포 커밋의 재검사 결과가 외부 변경에 좌우됨 | 소스 SHA를 고정한 bundle과 명시적 갱신 흐름 |

배포 저장소 최신화와 RFX 문서 최신화는 별개였다. 운영 checkout은 최신 GitOps main이지만,
마지막 RFX snapshot 갱신은 9월 6일 `b81134e`이며 그 commit은 RFX 원본 `6975228c`를 가리킨다.
9월 23일의 웹 build가 9월 27일 RFX 코드·문서를 반영한다는 뜻은 아니다.

근거 원본:

- [공개 문서 소유권과 제외 정책](../guides/public_docs_architecture.md)
- [현재 landing](../public/index.mdx), [Examples](../public/examples/index.mdx),
  [Gallery](../public/gallery/index.mdx), [패치 진단의 범위](../public/gallery/patch_antenna.mdx)
- [갤러리 생성기](../../scripts/precompute_gallery_artifacts.py),
  [route 검사](../../scripts/check_public_docs_manifest.py),
  [배포 exporter](../../scripts/export_public_docs_to_gitops.py)
- [문서 일괄 갱신 #1171](https://github.com/bk-squared/rfx/issues/1171),
  [예제 신뢰성 #737](https://github.com/bk-squared/rfx/issues/737),
  [공개 예제의 모델·배열 접근성 #1271](https://github.com/bk-squared/rfx/issues/1271)

조사 당시 RFX 최신 GitHub release는 v1.8.0이고 개발 브랜치의 패키지 version도 1.8.0이다.
따라서 package version만 찍어서는 release와 개발 코드를 구분할 수 없다.
문서 내용 불일치를 모두 런타임 결함으로 분류하지 않는다. 실행 실패와 잘못된 결과는 별도 재현이 필요하다.

## 3. 정보 구조와 URL

기존 URL을 우선 유지하고 화면의 탐색 순서를 바꾼다. 새 slug는 kebab-case로 통일한다.
이름을 바꾸는 기존 underscore 경로에는 redirect를 두고 실제 응답과 목적지를 검사한다.

| 주 메뉴 | 독자의 질문 | 내용과 연결 |
|---|---|---|
| Explore | 무엇을 계산할 수 있나? | `/rfx/` 대표 시연, `/rfx/gallery/` 구조·물리 현상별 사례 |
| Learn | 처음부터 어떻게 실행하나? | 설치, 첫 파동, 슬래브, 도파관, 포트, 안테나 순의 짧은 manual |
| Examples | 내 문제와 비슷한 코드는? | `/rfx/examples/`, 검색 가능한 실행 예제, 코드·데이터 다운로드 |
| API | 정확히 무엇을 호출하나? | `/rfx/api/` 작업별 요약과 생성된 symbol 상세 |
| Accuracy & limits | 결과를 어느 범위까지 믿나? | 지원 행렬, 현재 한계, 수렴·참조 비교, 버전별 근거 |

기여자 작업 절차·CI 운영 문서는 GitHub의 `docs/agent/`에 계속 둔다.
이 디렉터리를 웹에 통째로 게시하는 것은 기존 제외 정책과 맞지 않는다.
LLM용 공개 문서는 **RFX를 사용하는 사람의 API 문서**에서 파생하며, 저장소를 운영하는 지침과 구분한다.

기존 Guide는 `Getting started → Model → Run → Measure → Design`으로 정리한다.
예제는 "13개를 모두 순서대로 읽기"와 "안테나/회로/파동 중 내 작업부터 찾기"를 함께 지원한다.
API 함수 수와 이슈 번호가 주 탐색 분류가 되어서는 안 된다.

## 4. 사람이 읽는 문서: 화면과 사례의 구체적인 구성

### 첫 화면

1. 한 문장 소개와 대표 시뮬레이션 poster. 첫 화면에서 재생 버튼과 `예제 실행` 링크가 보인다.
2. `파동·재료`, `전송선·필터`, `안테나·산란`, `미분·설계` 네 카드.
3. 대표 사례 3개: 구조 그림 + 관측량 이름 + 짧은 영상 + 한 문장으로 설명한 결과.
4. 설치/첫 실행. 긴 최소 예제 전체는 quickstart로 이동한다.
5. 적용 버전과 지원 범위 요약, 정확도 근거와 상세 API로 가는 링크.

공개 사이트의 기본 언어는 현재처럼 영어를 유지하는 안을 기본값으로 둔다.
한국어 발표 대본은 설명 구성에 활용한다. 전면 이중 언어화는 별도 유지보수 범위다.

### 사례 페이지 공통 순서

| 순서 | 보이는 내용 | 독자가 알아야 하는 것 |
|---|---|---|
| 1 | 결과 poster와 10–20초 정도의 짧은 시연 | 무엇이 움직이고, 어떤 질문에 답하는가 |
| 2 | 구조·재료·급전·경계 도식 | 실제로 어떤 모델을 계산했는가 |
| 3 | 장과 측정 곡선의 동시 표시 | 장의 변화가 R/T, S, 방사 패턴에 어떻게 나타나는가 |
| 4 | `Run this example` | 고정 버전, 명령, 입력, 출력 파일, 필요한 자원 |
| 5 | 바꿔 볼 변수 1–2개 | 값의 의미, 유효 범위, 새 계산이 필요한 경우 |
| 6 | 정확도와 한계 | 해당 결과의 수렴·참조·지원 조건과 진단 상태 |
| 7 | API 상세와 관련 예제 | 같은 모델을 확장하기 위한 구체적 호출 |

개념 설명은 화면 곁의 짧은 문장으로 쓴다. 상세 검증표·재현 정보는 펼칠 수 있지만,
`diagnostic`, 적용 버전, 결과를 바꾸는 주요 한계는 접힌 영역에 숨기지 않는다.

### 애니메이션의 종류를 구분한다

- **시간 영역 실행**: 저장된 FDTD snapshot, 실제 시간·장 성분·좌표·색상 단위를 표시한다.
- **정상상태 위상 재생**: DFT phasor를 위상에 따라 재구성한 영상임을 표시한다.
  이를 펄스의 전파·도달 시간 증거로 사용하지 않는다.
- **설계 이력**: 실제 계산된 iteration과 같은 iteration의 형상·목적함수·S 곡선을 연결한다.
  중간 frame 보간을 추가했다면 보간임을 밝힌다.
- **미리 계산된 sweep**: frequency/parameter slider는 저장된 표본을 선택한다.
  웹에서 새 전자기 계산이 실행되는 것처럼 표현하지 않는다.

장 색상 범위는 비교하는 frame 사이에서 고정한다. E/H 성분, 선형/로그 척도,
정규화 기준, 단위를 표시한다. 최소한 곡선 CSV/Touchstone과 주요 scalar를 다운로드할 수 있게 한다.
스크린샷 속 수치만 남기지 않는다.

영상은 MP4/WebM과 정적 poster를 제공하고, GIF는 기존 URL 호환과 작은 보조 영상에 한정한다.
재생·정지·키보드 조작, reduced-motion의 정적 대체, 색각을 고려한 색상표와 설명을 제공한다.
초기 화면은 poster만 읽고 영상은 재생 시 로드한다. 제안 예산은 poster당 200 KB 이내,
짧은 영상 1–3 MB 목표이며, 초과하면 길이·해상도·frame rate를 조정한다.
과학적 가독성을 해치는 압축까지 강제하지 않는다.

## 5. 대표 사례와 기존 자료의 활용 순서

아래는 **구현 후보 순서**다. 기존 영상의 존재가 현재 버전의 정확도를 뜻하지 않는다.

| 우선순위 | 사례 | 보여줄 장면 | 재사용 가능한 출발점 | 공개 전 확인 |
|---|---|---|---|---|
| 1 | 유전체 슬래브 반사·투과 | 입사/반사/투과 펄스와 R/T 곡선 | `examples/tutorials/slab_rt_flux_monitor.py`, 기존 Fresnel gallery | 같은 설정의 장·flux·참조, 현재 버전 재실행 |
| 1 | WR-90 전파와 유전체 테이퍼 | TE10 장, taper 변화, 반사 감소 | 기존 waveguide gallery, 발표의 `taper_descent.mp4`·`taper_traj.mp4` | empty guide와 taper를 다른 모델로 명시, 목적함수·최종 RF 결과 연결 |
| 1 | 마이크로스트립 노치 필터 | stub 길이 변화에 따른 notch 이동, 장·전류 | 발표의 `notch_path.mp4`·`notch_phase.mp4` | 공개 builder, port/calibration, 위상 재생 여부, 현재 소스와 배열 확보 |
| 2 | 패치 안테나 | 구조 → 급전 → 근접장 → 방사 패턴 | `patch_antenna_demo.py`, `antenna_farfield_pattern.py`, 기존 발표 장 영상 | 갤러리 진단 bundle과 분리, 현재 지원 모델의 S·장·원거리장 범위 확인 |
| 2 | 유전체 superstrate 빔 조향 | 형상/유전율 변화와 목표각 방사 패턴 | 발표의 `steer_descent.mp4`·`steer_traj.mp4` | directivity/gain 구분, 동일 설계 재실행, gradient record-length witness |
| 2 | 산란과 RCS | 입사장/산란장 구분, 각도별 응답 | `examples/tutorials/rcs_scattering.py` | incident subtraction, 정규화와 기준 면, 지원 형상·대역 |
| 3 | 다층 반사 방지 코팅 | 층별 반사 간섭과 band objective | `examples/inverse_design/multilayer_ar_coating.py`, 공유 AR 자료 | 과거 보고서의 솔버 간 수치·결론 불일치 해소 전 성능 비교 재사용 금지 |
| 3 | 금속 topology 설계 | iteration별 도체 형상과 S 곡선 | 공유 topology 보고서 | 고전 stub보다 우수하다는 철회된 주장 재사용 금지; 현재 공개 지원 범위 확인 |

발표 자산은 `notch/taper/steer` 6종 MP4와 생성 스크립트·일부 원시 배열까지 확인했다.
공유 viewer의 이 영상들은 조사 시 HTTP 200이었다. 따라서 영상 제작 기법은 재사용 가능하다.
논문 시점의 수치를 현재 코드의 성능으로 바꾸어 말하는 일은 재검증 후에만 가능하다.

선별 자산을 찾기 위한 출처 지도(이 링크들은 영구 문서 자산으로 채택한 URL이 아니다):

| 사례 | 확인한 공유 영상 | 현재 저장소의 대응 소스 |
|---|---|---|
| 도파관 테이퍼 | [trajectory](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/taper_traj.mp4), [descent](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/taper_descent.mp4) | [waveguide_dielectric_taper.py](../../validation/tmtt_paper/waveguide_dielectric_taper.py) |
| 빔 조향 | [trajectory](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/steer_traj.mp4), [descent](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/steer_descent.mp4) | [beam_steering_superstrate.py](../../validation/tmtt_paper/beam_steering_superstrate.py) |
| 노치 필터 | [path](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/notch_path.mp4), [phase](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/notch_phase.mp4) | [msl_stub_notch_tuning.py](../../validation/tmtt_paper/msl_stub_notch_tuning.py) |

대응 소스가 존재한다는 뜻이며 공유 영상과 current source의 결과 동일성을 확인한 것은 아니다.
발표의 영상 생성 코드는 `figures/make_traj_animations.py`(테이퍼·빔 조향),
`figures/make_animations.py`(노치)다. 테이퍼·빔 조향은 내부 발표 원본에서 iteration별
`snapshots.npz`의 존재를 확인했다. 노치는 영상 생성 코드를 확인했지만 공개할 원시 배열 bundle의
완전성은 확인하지 않았다. 내부 경로·검토 기록은 별도 비공개 조사 기록에 두고,
공개 예제에는 선별한 배열과 그 생성 명령만 제공한다.
노치 path의 coarse 구간은 설명용 보간이며 phase 영상과 최종 spectrum은 stub 길이도 다르다.
빔 조향 영상은 441개 latent 변수의 궤적이고 gradient는 `∂L/∂latent`이다.
현재 API tutorial로 편집할 때 이 차이를 캡션·manifest에 남기거나 한 설정에서 다시 생성한다.

PEFM과의 경계:

- 공통 RFX 시뮬레이션, 재현 가능한 코드, 물리 설명만 RFX 사례로 편집한다.
- 과제 소개, 연차 목표, 내부 일정, 참여자·기관별 맥락, 미공개 후속 연구는 옮기지 않는다.
- 과거 논문 그림은 명시된 시점의 기록으로 유지할 수 있다. 현재 문서의 권장 설정과 섞지 않는다.
- 원본 공유 링크와 내부 발표는 출처 목록에 남기되, 사이트의 영구 자산은 검토한 공개용 bundle로
  새로 만든다. 임시 `/share/<id>/` URL에 장기 운영을 의존하지 않는다.

## 6. 예제·미디어를 하나의 모델에서 생성하기

새로운 예제 framework를 만들기보다 기존 builder, `Result.snapshots`/`snapshot_axes`,
artifact exporter, gallery manifest를 확장한다. `#1271`의 공개 builder/배열 문제와 먼저 조율한다.

목표 흐름:

```text
고정 코드 SHA + 공개 example builder + config
  → model / preflight
  → 계산 결과와 관측량 배열
  → 검증 근거와 범위
  → plot / video / poster / downloadable data
  → 사례 페이지 / 예제 카드 / API recipe / README 일부
```

장·S-parameter·gradient가 여러 실행을 필요로 하면 각각의 run identity와 공통 geometry/config hash를
기록한다. 서로 다른 실행을 하나인 것처럼 포장하지 않는다.
문서 생성기가 물리 검증을 새로 판정하지 않고, 검증 절차의 결과를 인용하도록 한다.

기존 `rfx-gallery-manifest-v2`와 `provenance`를 재사용하면서 다음 정보의 누락을 보완한다.
아래는 필드 설계 제안이며 구현된 schema는 아니다.

| 정보 | 의미 |
|---|---|
| `example_id`, `source_path`, `source_sha`, `config_hash` | 실행 모델과 코드 출처 |
| `package_version`, `docs_channel`, `environment` | release/dev, 주요 dependency·실행 환경 |
| `model`, `observable`, `units`, `axes` | 구조, 관측량, 차원·단위·좌표 |
| `execution_status` | setup-only, completed, failed 등의 실행 상태 |
| `evidence_scope`, `evidence_refs`, `limitations` | 어떤 정확도 주장을 어디까지 뒷받침하는가 |
| `code_compatibility`, `verified_source_sha` | 해당 버전에서 재실행했는가; 과거 결과 그대로인가 |
| `media_kind`, `run_refs`, `assets[].sha256` | transient/phasor/optimization 구분, 실행 연결, 파일 검증 |

하나의 `validated: true`로 실행 성공·물리 정확도·버전 호환성을 합치지 않는다.
실험 원자료와 큰 NPZ·로그는 기존 연구 archive에 보존하고, 공개에 필요한 최소 결과만 공개 artifact
저장소에 둔다. 코드 저장소에는 생성 코드·설정·작은 manifest 중심으로 남긴다.
다운로드 데이터와 영상은 immutable version/hash URL로 제공하고 checksum을 검사한다.

## 7. LLM-compatible API documentation

### 제공할 산출물

다음 경로는 모두 **제안**이다. 현재 존재 여부와 혼동하지 않도록 구현 PR에서 route를 확정한다.

| 산출물 | 역할 |
|---|---|
| `/rfx/llms.txt` | 제품·버전·지원 범위, 주제별 작은 Markdown 링크 색인 |
| `/rfx/api/.../index.md` | symbol/작업 단위의 정적 Markdown, HTML과 동일한 내용 |
| `/rfx/api/inventory.json` | schema version, fully-qualified symbol, signature, status, 상세 URL |
| `/rfx/examples/.../index.md` | 실행 가능한 recipe, code URL, 결과의 의미와 한계 |
| `/rfx/docs-manifest.json` | RFX SHA, docs SHA/channel, dependency lock hash, artifact 식별자 |

`llms-full.txt`는 우선순위를 낮춘다. 자동 생성한 범위 제한 bundle이 필요할 때 제공하고,
모든 문서를 거대한 한 파일로 합치는 것을 기본 접근법으로 삼지 않는다.
`llms.txt`는 탐색을 돕는 공개 제안 형식이며, 특정 모델의 수집·정확한 사용을 보장하지 않는다.
[공식 제안](https://llmstxt.org/)에 맞춰 페이지에 Markdown alternate와 llms describedby 링크를 둔다.

### 각 API 항목의 계약

signature와 타입만으로는 EM 코드를 올바르게 만들기 어렵다. 주요 API에는 다음이 함께 필요하다.

- 정확한 import 경로, positional/keyword-only, 기본값, 타입, 반환 객체의 field.
- 길이·주파수·시간·전류 등의 단위, 배열 shape와 축 순서, dtype/complex convention.
- 좌표 기준, 포트 방향·기준면·기준 임피던스와 정규화 의미.
- 지원 runner/mesh/boundary 조합, 미지원 시의 예외와 warning.
- AD 가능한 입력, static 입력, traced 입력의 제한, 값과 gradient의 서로 다른 검증 조건.
- 최소 실행 예제, 결과 읽는 방법, 관련 지원 항목·테스트·현재 한계.
- 추가/변경/폐기 버전과 해당 문서의 소스 SHA.

`Simulation`, geometry/material/source/port, run/forward, Result, S-parameter, far-field,
AD를 우선한다. import 가능한 내부 함수 전체를 지원 API로 승격하지 않는다.

### 생성과 유지보수

1. 기존 `scripts/check_api_reference.py`와 inventory를 확장한다. structural signature는 코드에서
   추출하고, 의미 정보는 canonical docstring과 기존 지원 문서에서 가져온다.
   `docs/guides/support_matrix.json`, `sparameter_support_matrix.json`의 기존 식별자·shape·지원 범위를
   연결한다. 같은 의미를 새로운 수기 지원표에 다시 적지 않는다.
2. 단위·shape·지원 조건은 자동 추측하지 않는다. 빠진 핵심 의미는 원본에 한 번 보완한다.
3. [pdoc](https://pdoc.dev/docs/pdoc.html)의 기존 생성 경로는 deep reference로 유지한다.
   ignored generated output을 exporter가 우연히 발견하게 두지 않고, 고정 SHA에서 만든 bundle을
   별도 입력으로 전달하고 허용 경로·checksum을 검사한다.
4. JSON·Markdown·HTML이 동일 inventory와 의미 원본을 읽는다. 제2의 수기 API 목록을 만들지 않는다.
5. MDX의 카드·aside·링크·코드를 의미 있게 변환한다. 태그를 무조건 삭제해 경고나 설명을 잃지 않는다.
6. 잘못된 symbol/signature/link/schema는 빠른 검사에서 잡고, 과학적 지원 주장은 별도 근거로 유지한다.

사용성 확인은 새 context의 에이전트에게 이 문서만 주고 수행한다.
기존 `scripts/diagnostics/blind_docs_test/`의 평가 틀을 재사용하되 오래된 과제와 정답 범위를 먼저 갱신한다.
슬래브 R/T, 적절한 포트 선택, 반환 배열 해석, 지원하지 않는 조합 판별, AD 제한 설명의
다섯 과제를 평가한다. 존재하지 않는 API와 잘못된 단위 사용은 0건이어야 한다.
실행 확인과 물리 정확도 확인을 각각 기록한다. 모델의 자유 응답을 그대로 필수 CI pass/fail로 쓰지 않는다.

## 8. README와 문서가 낡는 문제

README는 제품 소개, 대표 poster 한 장, 설치, 가장 작은 실행 코드, 문서/예제/API/한계 링크로 줄인다.
기능별 지원 표, 하드웨어별 속도 표, 길게 변하는 validation 상태를 중복해서 싣지 않는다.
대표 그림은 정적인 결과이며 클릭하면 버전과 근거가 있는 사례 페이지로 간다.

| 자주 낡는 항목 | 정본 | README·웹에서의 사용 |
|---|---|---|
| 설치·최소 코드 | 실행 가능한 canonical starter | 지정 구간 자동 추출, CPU 실행 검사 |
| symbol/signature | 소스 코드와 docstring | API inventory/HTML/Markdown 자동 생성 |
| 지원/제한 | 지원 행렬·known limitations | 요약과 링크, 독립된 수기 복제 금지 |
| 성능·정확도 수치 | 특정 코드·환경·검증 기록 | 버전별 사례에서 설명, README는 해당 페이지 링크 |
| 예제 목록·탐색 | source route/example registry | Examples/Sidebar/llms 색인 생성 |
| 변경 내역 | changelog fragments → release assembly | 버전별 release notes로 연결 |

기존 README는 146행으로 무조건 길이를 줄이는 문제가 아니다. 별도 유지되는 quickstart와
특정 GPU 처리량 등 **자주 변하는 사실의 복제**를 줄이는 것이 목적이다.

버전 정책:

- `/rfx/`는 검증된 stable 문서로 안내한다. `/rfx/dev/`는 개발 SHA를 명확히 표시한다.
- `/rfx/versions/<tag>/`에 release별 문서를 보존하고 버전 선택기를 제공한다.
- 현재 절대 `/rfx/...` 링크는 선택한 channel/tag 안에서 해석되도록 생성 시 변환한다.
  API·예제·support·미디어 링크가 다른 버전으로 새지 않는지 검사하고, 코드 링크는 해당 tag/SHA로 고정한다.
  여러 버전을 함께 검색할 때 결과에 버전을 표시하며 기본 검색 범위는 현재 선택한 버전이다.
- 처음 stable 묶음은 그 release tag에서 실제로 build/test한다. 현재 main 문서에 v1.8.0 표지만 붙이지 않는다.
- 코드에 없는 과거 원시 결과는 새로 검증했다고 쓰지 않는다. 필요하면 historical 페이지로 남긴다.
- 페이지 footer의 날짜만으로 freshness를 주장하지 않는다. source SHA와 확인한 example/API 버전을 표시한다.

개발 중인 prose mismatch를 매 PR의 merge blocker로 만드는 것은 기존 `#1171` 정책과 충돌한다.
개발 prose는 batch에 모으되, 깨진 생성 API 계약·필수 파일·공개된 잘못된 결과는 구분해서 처리한다.
release 전에 문서 batch, 예제 재실행, 필요한 수치 재검증을 끝낸다.

## 9. 배포와 운영

소유권은 그대로 유지한다. RFX가 내용·예제·계약을, infra가 rendering·route·배포를 소유한다.
운영 checkout과 NFS의 다른 세션 수정 사항은 문서 작업으로 덮어쓰지 않는다.

조사 시 운영 GitOps SHA는 `4dc53602a64c3b082c4169934064cbe2d08493cb`였다.
현재 배포에는 `deploy-public-family.sh`의 별도 candidate build와 dist 교환·이전 산출물 보존이 있다.
과거 문서의 "컨테이너를 재생성하면 빌드된다" 절차 대신 현행 family-site 배포 절차를 따른다.

계획할 흐름:

```text
RFX release tag 또는 명시한 dev SHA
  → 문서·API·예제 metadata 검증
  → checksum을 포함한 public docs bundle
  → 그 SHA와 bundle을 고정한 GitOps 변경
  → Astro build + family renderer
  → candidate HTTP/화면/asset 검사
  → 기존 원자적 activation
  → 공개 HTTP·본문·media checksum·version manifest 확인
```

RFX 변경이 있을 때 matching GitOps snapshot을 만드는 자동화는 review 가능한 PR/bundle까지 수행한다.
배포 저장소 CI는 매번 최신 RFX main을 가져오지 않고 선언된 SHA를 검사한다.
일일 drift 검사는 "새 원본이 있다"를 알리고, 기존 버전 산출물이 재현되지 않는 문제와 구분한다.

주의할 실제 rendering 경계:

- family renderer가 Astro의 CSS link/class/일부 inline style을 정리한다.
  Astro preview만 확인해서 새 visualization component가 운영에서도 동작한다고 판단하지 않는다.
- CSS·JS·poster·video·Markdown·JSON의 보존과 MIME를 최종 renderer 이후 검사한다.
- 같은 최종 산출물에서 HTML·Markdown·API inventory·llms·검색의 내부 링크가 선택 버전을 유지하고,
  코드·지원 근거 링크가 해당 tag/SHA를 가리키는지도 검사한다.
- 새 artifact 입력에도 public allowlist, symlink/경로 이탈 거부, checksum 검사를 적용한다.
  `docs/agent/`, private notes, 임시 출력이 bundle에 포함되면 출판 검사를 실패시킨다.
- shared site의 `/`, `/rfx/`, `/creative_engineering_design/`를 함께 확인한다.
- 이전 dist와 artifact를 유지한다. 문제 시 이전 bundle로 되돌리고 HTTP와 checksum을 재확인한다.
- 다른 프로젝트의 family shell을 바꾸는 PR은 RFX 콘텐츠 변경과 따로 검토한다.

## 10. 작업 순서와 완료 조건

계획은 한 이슈/PR씩 진행한다. 독립적인 읽기·검토는 병렬화하되 공개 예제·검증 갱신을
다른 solver 작업과 한 PR로 섞지 않는다. 아래 단계는 기존 `#1171`, `#737`, `#1271`과 연결하며
이 계획 자체가 중복 이슈 생성을 요구하지 않는다.

| 단계 | 결과물 | 완료 조건 |
|---|---|---|
| 0 — 이번 조사 | 현재 구조·배포·시각 자료·제약과 이 계획 | 실제 사이트·소스·운영 상태 확인; 새 물리 주장 없음 |
| 1 — 출처와 버전 | source SHA/bundle manifest, stable/dev 설계, generated API export 경로 | 같은 SHA로 같은 route·inventory가 생성되고 과거 artifact를 현재 것으로 표시하지 않음 |
| 2 — 대표 예제 묶음 | 슬래브, 도파관/테이퍼, 노치의 공개 builder·데이터·poster·영상 | clean checkout에서 실행; 모델·곡선·영상 연결; 주장별 근거와 지원 범위 확인 |
| 3 — 사람용 manual | 첫 화면, 사례 template, Learn/Examples 탐색, 시각화 component | 작은/큰 화면·키보드·정지 화면·최종 family renderer 확인; 기존 URL 유지/redirect |
| 4 — API와 LLM | signature/semantic reference, Markdown/JSON/llms 색인 | 링크·schema·signature 검사, 다섯 사용자 과제에서 API/단위 오류 없음 |
| 5 — release 문서 갱신 | #1171 batch, README 생성 구간, 릴리스별 snapshot | 목표 release 환경에서 예제·코드 블록 확인, 필요한 물리 재검증, 공개 산출물 readback |
| 6 — 확장 | 패치·빔 조향·RCS·재료 최적화 | 단계 2와 동일한 기준을 충족하는 사례부터 추가 |

단계 2의 GPU 작업량은 현재 자료의 builder/data 준비 상태를 확인한 뒤 산정한다.
이 계획을 위해 GPU 실행은 제출하지 않았다. 향후 제출마다 현행 자원 선택 규칙과 live preset 비교를 적용한다.

### 검사 비용에 따른 분리

- 변경 PR: schema·route·링크·API 구조 검사, 바뀐 짧은 예제의 CPU 실행, 웹 build/component 검사.
- release: 전체 public 코드 블록과 canonical examples 실행, current numerical claim에 필요한 검증.
- 주기적 운영: 외부 링크·미디어 접근·배포 SHA·원본 대비 drift 확인.

`PASS`의 의미를 검사별로 적는다. schema 통과나 코드를 실행했다는 사실을 물리 검증으로 승격하지 않는다.
support/정확도 tolerance는 문서 개편을 위해 바꾸지 않는다.

## 11. 선택하지 않는 접근

| 대안 | 선택하지 않는 이유 |
|---|---|
| 새 문서 framework로 전면 이관 | 문제의 핵심은 출처·예제·버전·export이며, 현재 도구에서 개선 가능 |
| LLM용 문서를 별도로 수기 집필 | 사람용 API와 곧 달라지는 두 번째 정본이 됨 |
| `docs/agent/` 전체 웹 공개 | end-user API와 contributor 운영 절차가 다르고 기존 제외 정책에 어긋남 |
| 발표자료를 그대로 복사 | 과거 수치, 진단·성공 혼합, 내부 프로젝트 맥락까지 따라옴 |
| 매 문서 build에서 모든 FDTD 재계산 | build 비용·재현성이 나빠지고 기존 검증 범위를 불필요하게 재개함 |
| README에 최신 성능·지원 상태를 계속 추가 | 동일 사실의 수기 복제가 늘어남 |
| 브라우저에서 바로 GPU 시뮬레이션 | 이번 공개 문서 목표에 불필요한 실행 서비스·운영 범위가 추가됨 |

## 12. 이번에 확인한 범위

완료: 원본 규칙·작업 절차·문서 정책 읽기, 원본/배포/발표 자산 읽기 전용 조사,
공개 HTTP 확인, 데스크톱/390px 모바일 화면 확인(가로 overflow 없음), 기존 route/API/gallery 검사.
기존 route 검사는 24개 slug, gallery 검사는 3개 manifest를 통과했다.
API inventory 검사도 통과했으며, Simulation의 public method 49개는 모두 docstring을 갖고 있다.
따라서 API 문서 작업의 출발점은 전면 재집필보다 기존 내용의 구조화와 배포 연결이다.
이 결과는 현재 검사 도구의 범위 안에서의 성공이다.

미수행: 전체 예제와 public 코드 블록 재실행, 새 FDTD/GPU 실행, 과거 수치 재검증,
새 페이지·생성 pipeline 구현, 배포·push·PR·이슈 변경.
따라서 이 문서의 미디어/버전/API 구조는 구현 계획이며 이미 제공 중인 기능으로 읽으면 안 된다.
