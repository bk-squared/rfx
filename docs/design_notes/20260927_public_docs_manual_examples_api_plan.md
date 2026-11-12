# RFX public documentation redesign plan: show simulations, reproduce them, and connect them to the API

Date: 2026-09-27 · Status: proposal / not implemented

Survey baseline: RFX `a9054338616b91bb1364699882c2530fa29a0048`, actual public-site HTTP responses and
desktop/mobile views, existing examples and API tools, shared presentations, and the deployment repository and runtime.
This document proposes a documentation structure and work sequence. It does not validate new simulation results or accuracy.

## 1. Proposed decision

Organize the RFX site around **discovering capabilities through visual examples → running a small example → reading precise API contracts**.
Keep the existing Astro/Starlight framework and RFX documentation sources.
Generate human and LLM documentation from the same version of the code, examples and support information, without separate authorship.

Prioritize a dielectric slab, WR-90 waveguide/taper, and microstrip notch filter for the first public collection.
Add antenna radiation, beam steering and scattering later. Do not use the existing diagnostic patch gallery as a representative success case.
Use internal presentations for visualization and exposition ideas; keep PEFM research goals, results and operational context separate.

Completion has three criteria, independent of page count.

1. A first-time visitor can distinguish the simulated object and its results from representative images and videos.
2. A case page's runnable code, figures and numbers refer to the same model and an explicit code version.
3. Humans and agents find the same API signature, support conditions, limitations and runnable examples.

## 2. Strengths and limitations of the current structure

The existing foundation is reusable: `docs/public/` sources, the deployment exporter, API inventory,
page code-block runner, gallery manifests, example learning sequence and support matrices already exist.
Slab and waveguide detail pages also contain GIFs. The site already has some animation.

| Observation | Effect on readers and maintenance | Proposed improvement |
|---|---|---|
| The live landing page and Examples hub body each contain 0 images or videos | Readers must start with feature tables and code to picture a simulation | Lead with results, visual case cards and direct links to runnable examples |
| All 3 galleries were generated with `1.6.5 / 1eb551b` | Current documentation and historical results apply to different versions | Retain historical labels and regenerate representative cases at the target version |
| The patch gallery's port setup fails preflight, and its field images and port results come from different runs | The images cannot be presented as the antenna performance of one model | Preserve it as diagnostic material; register a new representative patch after independent reproduction and validation |
| The slab gallery instructions still contain low-level Yee updates and an omitted loop | Beginners encounter internals before the public `Simulation` API | Generate page code from a high-level example such as `slab_rt_flux_monitor.py` |
| `site_map.json` admits only 24 guide pages as primary navigation, and deployment also defines navigation | API, Examples and Gallery locations are decided in multiple places | Derive sidebars, hubs, search and LLM indexes from one route registry |
| API inventory checks do not guarantee types, defaults, returns or units | Readers can call an existing function incorrectly or misinterpret returned arrays | Generate structural signatures and add small manual annotations for support semantics |
| `docs/api/` is generated/ignored, while the exporter copies only tracked files | Generated pdoc output does not enter the deployment artifacts | Accept an API bundle generated at a fixed SHA as an explicit build input |
| `/rfx/llms.txt`, `/rfx/llms-full.txt`, `/rfx/api/inventory.json` returned 404 during the survey | No stable machine-readable entry point exists | Provide small indexes, Markdown and JSON with explicit versions |
| Installation uses PyPI while the text mixes development changes and historical results | Readers cannot easily match installed packages to documentation | Separate stable/dev and expose tags, source SHAs and build identifiers |
| README, landing and quickstart maintain similar code separately | Updating only one makes the examples diverge | Generate the needed excerpts from one executable file |
| Deployment CI queries RFX's moving default branch | External changes affect reruns of the same deployment commit | Use bundles with fixed source SHAs and an explicit update process |

Deployment-repository freshness and RFX-documentation freshness were separate. The production checkout was at the latest GitOps main,
but its last RFX snapshot update was on month 9, day 6, at `b81134e`, which points to RFX source `6975228c`.
A web build on month 9, day 23 does not imply that it includes RFX code and documentation from month 9, day 27.

Sources:

- [Public documentation ownership and exclusions](../guides/public_docs_architecture.md)
- [Current landing page](../public/index.mdx), [Examples](../public/examples/index.mdx),
  [Gallery](../public/gallery/index.mdx), and the patch diagnostic scope (gallery/patch_antenna.mdx, removed in 2.0 and absent from the repository)
- [Gallery generator](../../scripts/precompute_gallery_artifacts.py),
  [route check](../../scripts/check_public_docs_manifest.py),
  [deployment exporter](../../scripts/export_public_docs_to_gitops.py)
- [Documentation batch update #1171](https://github.com/bk-squared/rfx/issues/1171),
  [example fidelity #737](https://github.com/bk-squared/rfx/issues/737),
  [public example model and array access #1271](https://github.com/bk-squared/rfx/issues/1271)

At the time of the survey, the latest RFX GitHub release was v1.8.0 and the development branch's package version was also 1.8.0.
Printing the package version alone therefore cannot distinguish released and development code.
Documentation mismatches are not all runtime defects. Execution failures and incorrect results need separate reproductions.

## 3. Information architecture and URLs

Prefer retaining existing URLs while changing navigation order. Use kebab-case for new slugs.
Redirect renamed underscore paths and check the actual responses and destinations.

| Main menu | Reader's question | Content and links |
|---|---|---|
| Explore | What can I simulate? | Representative demonstrations at `/rfx/`, cases by structure and phenomenon at `/rfx/gallery/` |
| Learn | How do I start? | A short manual covering installation, first waves, slabs, waveguides, ports and antennas |
| Examples | Which code resembles my problem? | `/rfx/examples/`, searchable runnable examples, code and data downloads |
| API | What exactly do I call? | Task summaries and generated symbol details at `/rfx/api/` |
| Accuracy & limits | Within what scope can I trust the result? | Support matrices, current limitations, convergence/reference comparisons and versioned evidence |

Keep contributor procedures and CI operations documentation in GitHub's `docs/agent/`.
Publishing that whole directory on the website would conflict with the existing exclusion policy.
Derive public LLM documentation from **API documentation for RFX users**, separately from repository operations instructions.

Organize the existing Guide as `Getting started → Model → Run → Measure → Design`.
Support both reading all 13 examples in sequence and finding a task in antennas, circuits or waves.
API function counts and issue numbers should not define primary navigation.

## 4. Human documentation: screen and case layouts

### Landing page

1. A one-sentence introduction and representative simulation poster, with a visible play button and `Run example` link.
2. Four cards: `Waves and materials`, `Transmission lines and filters`, `Antennas and scattering`, `Differentiation and design`.
3. 3 representative cases: a structure image, observable name, short video and one-sentence result.
4. Installation and first run. Put the complete minimal example in quickstart.
5. Applicable version and support summary, with links to accuracy evidence and detailed API documentation.

The default proposal retains English as the public site's primary language.
Use Korean presentation scripts to inform the exposition. Full bilingual coverage is a separate maintenance scope.

### Common case-page sequence

| Order | Visible content | What the reader needs to know |
|---|---|---|
| 1 | Result poster and a short demonstration of about 10–20 seconds | What moves and which question it answers |
| 2 | Structure, materials, feed and boundary diagram | The model actually simulated |
| 3 | Fields and measurement curves shown together | How field changes appear in R/T, S and radiation patterns |
| 4 | `Run this example` | Fixed version, commands, inputs, output files and required resources |
| 5 | 1–2 variables to change | Meaning, valid range and when a new computation is required |
| 6 | Accuracy and limitations | Convergence, references, support conditions and diagnostic status of this result |
| 7 | API details and related examples | Specific calls for extending the same model |

Explain concepts in short sentences beside the visualization. Detailed validation tables and reproduction information may be collapsible,
but keep `diagnostic`, the applicable version and major result-affecting limitations visible.

### Distinguish animation types

- **Time-domain run**: show saved FDTD snapshots with actual time, field component, coordinates and color units.
- **Steady-state phase playback**: label the video as a reconstruction from DFT phasors over phase.
  Do not use it as evidence of pulse propagation or arrival times.
- **Design history**: connect actual computed iterations to their geometry, objective and S curves.
  Label any interpolated intermediate frames.
- **Precomputed sweep**: frequency/parameter sliders select saved samples.
  Do not imply that the website performs new electromagnetic computations.

Fix field color ranges across compared frames. Label E/H components, linear/log scales,
normalization and units. Provide at least curve CSV/Touchstone files and key scalars for download.
Do not leave numbers available only in screenshots.

Provide MP4/WebM videos and static posters; limit GIFs to existing URL compatibility and small supplementary animations.
Provide play/pause and keyboard controls, static reduced-motion alternatives, accessible color maps and descriptions.
Load only posters initially and videos on playback. The proposed budget is at most 200 KB per poster,
with a 1–3 MB target for short videos; adjust duration, resolution or frame rate when exceeded.
Do not require compression that harms scientific readability.

## 5. Representative cases and reuse priorities

The following is a **candidate implementation order**. An existing video does not establish accuracy for the current version.

| Priority | Case | Scene to show | Reusable starting point | Checks before publication |
|---|---|---|---|---|
| 1 | Dielectric slab reflection/transmission | Incident/reflected/transmitted pulses and R/T curves | `examples/tutorials/slab_rt_flux_monitor.py`, existing Fresnel gallery | Fields, flux and reference at the same settings; rerun on the current version |
| 1 | WR-90 propagation and dielectric taper | TE10 fields, taper changes and reduced reflection | Existing waveguide gallery, presentation `taper_descent.mp4` and `taper_traj.mp4` | Identify the empty guide and taper as different models; connect objective and final RF results |
| 1 | Microstrip notch filter | Notch movement with stub length, fields and currents | Presentation `notch_path.mp4` and `notch_phase.mp4` | Public builder, port/calibration, phase-playback status, current source and arrays |
| 2 | Patch antenna | Structure → feed → near field → radiation pattern | `patch_antenna_demo.py`, `antenna_farfield_pattern.py`, existing presentation field videos | Separate from the diagnostic gallery bundle; check supported S, field and far-field scope |
| 2 | Dielectric superstrate beam steering | Geometry/permittivity changes and radiation at the target angle | Presentation `steer_descent.mp4` and `steer_traj.mp4` | Distinguish directivity/gain; rerun the same design; gradient record-length witness |
| 2 | Scattering and RCS | Incident/scattered fields and angular response | `examples/tutorials/rcs_scattering.py` | Incident subtraction, normalization and reference planes, supported shapes and bands |
| 3 | Multilayer antireflection coating | Interference between layer reflections and band objective | `examples/inverse_design/multilayer_ar_coating.py`, shared AR material | Do not reuse performance comparisons until discrepancies in historical solver numbers and conclusions are resolved |
| 3 | Metal topology design | Conductor geometry and S curves per iteration | Shared topology report | Do not reuse the withdrawn superiority claim over a classical stub; check current public support scope |

The survey confirmed 6 `notch/taper/steer` MP4s, generation scripts and some raw arrays.
These videos returned HTTP 200 in the shared viewer. Their video-production techniques are reusable.
Presenting paper-era numbers as current-code performance requires revalidation.

Source map for locating selected assets (these URLs have not been adopted as permanent documentation assets):

| Case | Confirmed shared videos | Corresponding source in the current repository |
|---|---|---|
| Waveguide taper | [trajectory](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/taper_traj.mp4), [descent](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/taper_descent.mp4) | [waveguide_dielectric_taper.py](../../validation/tmtt_paper/waveguide_dielectric_taper.py) |
| Beam steering | [trajectory](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/steer_traj.mp4), [descent](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/steer_descent.mp4) | [beam_steering_superstrate.py](../../validation/tmtt_paper/beam_steering_superstrate.py) |
| Notch filter | [path](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/notch_path.mp4), [phase](https://remilab.cnu.ac.kr/share/6ba431aac4d9/viewer/assets/notch_phase.mp4) | [msl_stub_notch_tuning.py](../../validation/tmtt_paper/msl_stub_notch_tuning.py) |

The corresponding source exists; result identity between the shared videos and current source was not checked.
The presentation's video-generation code is `figures/make_traj_animations.py` (taper and beam steering) and
`figures/make_animations.py` (notch). For taper and beam steering, the internal presentation sources contain per-iteration
`snapshots.npz`. The notch video-generation code was checked, but completeness of the raw-array bundle
to be published was not checked. Keep internal paths and review records in a separate private survey record;
provide only selected arrays and their generation commands with public examples.
The coarse part of the notch path is illustrative interpolation; its phase video and final spectrum also use different stub lengths.
The beam-steering video traces 441 latent variables, with gradient `∂L/∂latent`.
Retain these distinctions in captions/manifests when making an API tutorial, or regenerate from one configuration.

Boundary with PEFM:

- Adapt only shared RFX simulations, reproducible code and physical explanations into RFX cases.
- Omit project introductions, annual goals, internal schedules, participant/institution context and unpublished follow-up research.
- Historical paper figures may remain as records of their stated time. Keep them distinct from current recommended settings.
- Retain original shared links and internal presentations in source lists, but create permanent site assets from reviewed public bundles.
  Do not depend on temporary `/share/<id>/` URLs for long-term operation.

## 6. Generate examples and media from one model

Extend existing builders, `Result.snapshots`/`snapshot_axes`,
artifact exporters and gallery manifests. Coordinate first with `#1271` on public builders and arrays.

Target flow:

```text
고정 코드 SHA + 공개 example builder + config
  → model / preflight
  → 계산 결과와 관측량 배열
  → 검증 근거와 범위
  → plot / video / poster / downloadable data
  → 사례 페이지 / 예제 카드 / API recipe / README 일부
```

If fields, S-parameters and gradients require multiple runs, record each run identity and their shared geometry/config hash.
Do not present separate runs as one.
The documentation generator cites validation results; it does not make new physical-validation judgments.

Reuse `rfx-gallery-manifest-v2` and `provenance`, filling gaps in the following information.
These are proposed fields, not an implemented schema.

| Information | Meaning |
|---|---|
| `example_id`, `source_path`, `source_sha`, `config_hash` | Executed model and code provenance |
| `package_version`, `docs_channel`, `environment` | Release/dev, major dependencies and execution environment |
| `model`, `observable`, `units`, `axes` | Structure, observable, dimensions, units and coordinates |
| `execution_status` | Execution state such as setup-only, completed or failed |
| `evidence_scope`, `evidence_refs`, `limitations` | Which accuracy claims the evidence supports and to what extent |
| `code_compatibility`, `verified_source_sha` | Whether rerun at that version or retained as a historical result |
| `media_kind`, `run_refs`, `assets[].sha256` | Transient/phasor/optimization distinction, run links and file verification |

Do not combine execution success, physical accuracy and version compatibility into one `validated: true` field.
Preserve raw experimental data and large NPZ/log files in the existing research archive; put only the minimum public results
in the public artifact repository. Keep generation code, configurations and small manifests in the code repository.
Serve downloadable data and videos at immutable version/hash URLs and check their checksums.

## 7. LLM-compatible API documentation

### Deliverables

All paths below are **proposals**. Finalize routes in implementation PRs so they are not confused with existing features.

| Deliverable | Role |
|---|---|
| `/rfx/llms.txt` | Product, version, support scope and a small topical index of Markdown links |
| `/rfx/api/.../index.md` | Static Markdown per symbol/task, with the same content as HTML |
| `/rfx/api/inventory.json` | Schema version, fully-qualified symbol, signature, status and detail URL |
| `/rfx/examples/.../index.md` | Runnable recipe, code URL, result interpretation and limitations |
| `/rfx/docs-manifest.json` | RFX SHA, docs SHA/channel, dependency lock hash and artifact identifiers |

Give `llms-full.txt` lower priority. Provide a generated bundle with bounded scope when needed;
a single giant file containing all documentation is not the default approach.
`llms.txt` is a public proposal for navigation; it does not guarantee ingestion or correct use by a particular model.
Following the [official proposal](https://llmstxt.org/), add Markdown alternate and llms describedby links to pages.

### Contract for each API entry

Signatures and types alone are insufficient for correct EM code. Major APIs also need:

- Exact import paths, positional/keyword-only arguments, defaults, types and returned-object fields.
- Units for length, frequency, time and current; array shapes and axis order; dtype/complex conventions.
- Coordinate conventions, port directions, reference planes, reference impedances and normalization semantics.
- Supported runner/mesh/boundary combinations, and exceptions or warnings for unsupported ones.
- Differentiable inputs, static inputs, traced-input restrictions, and separate validation conditions for values and gradients.
- Minimal runnable examples, result interpretation, related support entries/tests and current limitations.
- Introduction/change/deprecation versions and the documentation's source SHA.

`Simulation`, geometry/material/source/port, run/forward, Result, S-parameter, far-field,
Prioritize AD alongside these. Do not promote every importable internal function to a supported API.

### Generation and maintenance

1. Extend `scripts/check_api_reference.py` and the inventory. Extract structural signatures from code,
   and semantics from canonical docstrings and existing support documentation.
   Connect existing identifiers, shapes and support scopes in `docs/guides/support_matrix.json` and `sparameter_support_matrix.json`.
   Do not repeat the same semantics in a new manual support table.
2. Do not infer units, shapes or support conditions automatically. Fill missing essential semantics once at the source.
3. Retain the existing [pdoc](https://pdoc.dev/docs/pdoc.html) generation path for deep reference.
   Explicitly supply the exporter with a bundle generated at a fixed SHA,
   checking allowed paths and checksums instead of relying on discovery of ignored generated output.
4. JSON, Markdown and HTML read the same inventory and semantic sources. Do not create a 2nd manual API list.
5. Convert MDX cards, asides, links and code meaningfully. Blind tag removal must not discard warnings or explanations.
6. Catch incorrect symbols, signatures, links and schemas with fast checks; retain separate evidence for scientific support claims.

Check usability by giving only this documentation to an agent in a fresh context.
Reuse the evaluation framework in `scripts/diagnostics/blind_docs_test/`, first updating old tasks and expected-answer scope.
Evaluate five tasks: slab R/T, suitable port selection, returned-array interpretation, unsupported-combination detection,
and AD-limit explanation. There must be 0 uses of nonexistent APIs or incorrect units.
Record execution checks and physical-accuracy checks separately. Do not use a model's free-form answers directly as required CI pass/fail.

## 8. README and documentation drift

Limit the README to a product introduction, one representative poster, installation, minimal runnable code and documentation/example/API/limitations links.
Do not duplicate per-feature support tables, per-hardware speed tables or lengthy changing validation status.
The representative image is a static result linked to a case page with its version and evidence.

| Frequently stale item | Canonical source | README/web use |
|---|---|---|
| Installation/minimal code | Executable canonical starter | Automatically extract designated sections and check CPU execution |
| Symbol/signature | Source code and docstrings | Generate API inventory/HTML/Markdown |
| Support/limitations | Support matrices and known limitations | Summaries and links; no independent manual copies |
| Performance/accuracy numbers | Specific code, environment and validation records | Explain in versioned cases; README links to those pages |
| Example list/navigation | Source route/example registry | Generate Examples/Sidebar/llms indexes |
| Changes | Changelog fragments → release assembly | Link to versioned release notes |

The existing README has 146 lines; unconditional shortening is not the goal. Reduce duplication of frequently changing facts,
including the separately maintained quickstart and specific GPU throughput figures.

Version policy:

- `/rfx/` leads to checked stable documentation. `/rfx/dev/` clearly identifies its development SHA.
- Preserve release documentation at `/rfx/versions/<tag>/` and provide a version selector.
- Transform current absolute `/rfx/...` links during generation so they resolve within the selected channel/tag.
  Check that API, example, support and media links stay in that version; pin code links to its tag/SHA.
  Show versions when searching across versions; default search scope is the currently selected version.
- Build/test the first stable bundle at the actual release tag. Do not merely label current main documentation v1.8.0.
- Do not claim new validation for historical raw results absent from the code. Retain historical pages when needed.
- A footer date alone does not establish freshness. Show the source SHA and checked example/API versions.

Making development prose mismatches merge blockers on every PR conflicts with the existing `#1171` policy.
Collect development prose changes in batches; handle broken generated API contracts, required files and published incorrect results separately.
Finish the documentation batch, example reruns and necessary numerical revalidation before release.

## 9. Deployment and operations

Retain ownership: RFX owns content, examples and contracts; infra owns rendering, routes and deployment.
Documentation work must not overwrite other sessions' changes in production checkouts or NFS.

The production GitOps SHA during the survey was `4dc53602a64c3b082c4169934064cbe2d08493cb`.
Current deployment uses a separate candidate build, dist swap and previous-artifact preservation in `deploy-public-family.sh`.
Follow the current family-site deployment procedure instead of the older container-recreation build instructions.

Planned flow:

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

Automation responding to RFX changes creates a matching, reviewable GitOps snapshot PR/bundle.
Deployment-repository CI checks the declared SHA instead of fetching the latest RFX main each time.
Daily drift checks report new source separately from failure to reproduce an existing version's artifacts.

Rendering boundaries to check:

- The family renderer removes Astro CSS links, classes and some inline styles.
  Astro preview alone cannot establish that a new visualization component works in production.
- Check CSS, JS, posters, videos, Markdown and JSON preservation and MIME types after the final renderer.
- In that same final output, check that HTML, Markdown, API inventory, llms and search links retain the selected version,
  and that code and support-evidence links point to its tag/SHA.
- Apply the public allowlist, symlink/path-traversal rejection and checksum checks to new artifact inputs too.
  Fail publication checks if bundles include `docs/agent/`, private notes or temporary output.
- Check the shared site's `/`, `/rfx/` and `/creative_engineering_design/` together.
- Retain the previous dist and artifacts. On failure, restore the previous bundle and recheck HTTP and checksums.
- Review PRs changing other projects' family shell separately from RFX content changes.

## 10. Work sequence and completion criteria

Proceed one issue/PR at a time. Independent reading and review may run in parallel; public-example and validation updates
must not share a PR with unrelated solver work. Connect the stages below to existing `#1171`, `#737` and `#1271`;
this plan does not require duplicate issues.

| Stage | Deliverable | Completion criteria |
|---|---|---|
| 0 — This survey | Current structure, deployment, visuals, constraints and this plan | Check actual site, source and operations; no new physics claims |
| 1 — Provenance and versions | Source SHA/bundle manifest, stable/dev design, generated API export path | Same SHA generates the same routes/inventory; historical artifacts are not labeled current |
| 2 — Representative examples | Public builders, data, posters and videos for slab, waveguide/taper and notch | Run from a clean checkout; connect model, curves and video; check each claim's evidence and support scope |
| 3 — Human manual | Landing page, case template, Learn/Examples navigation, visualization components | Check small/large screens, keyboard, static views and final family renderer; preserve/redirect existing URLs |
| 4 — API and LLM | Signature/semantic reference, Markdown/JSON/llms indexes | Link/schema/signature checks; no API/unit errors in the five user tasks |
| 5 — Release documentation update | #1171 batch, generated README sections, release snapshots | Check examples/code blocks in the target release environment, revalidate physics where needed, read back public artifacts |
| 6 — Expansion | Patch, beam steering, RCS and material optimization | Add cases as they meet stage 2's criteria |

Estimate stage 2 GPU work after checking current builder/data readiness.
No GPU jobs were submitted for this plan. Apply current resource-selection rules and live-preset comparisons to each future submission.

### Separate checks by cost

- Change PR: schema, route, link and API-structure checks; CPU execution of changed short examples; web build/component checks.
- Release: run all public code blocks and canonical examples; validate current numerical claims as needed.
- Periodic operations: check external links, media access, deployment SHA and source drift.

State what `PASS` means for each check. Schema validity or successful execution does not establish physical validation.
Do not change support/accuracy tolerances for this documentation redesign.

## 11. Approaches not selected

| Alternative | Reason not selected |
|---|---|
| Full migration to a new documentation framework | Provenance, examples, versions and export are the core problems and can be improved with current tools |
| Separately authored LLM documentation | Creates another canonical source that soon diverges from the human API documentation |
| Publish all of `docs/agent/` on the web | End-user API documentation and contributor operations differ; publication conflicts with the exclusion policy |
| Copy presentations unchanged | Carries over historical numbers, mixed diagnostic/success cases and internal project context |
| Recompute all FDTD on every documentation build | Increases build cost, weakens reproducibility and unnecessarily reopens existing validation scope |
| Keep adding current performance/support status to README | Increases manual duplication of the same facts |
| Run GPU simulations directly from the browser | Adds an execution service and operations outside this documentation goal |

## 12. Scope checked in this survey

Completed: reading source rules, procedures and documentation policy; read-only surveys of source, deployment and presentation assets;
public HTTP checks; desktop/390px mobile checks (no horizontal overflow); existing route/API/gallery checks.
The existing route check passed for 24 slugs and the gallery check for 3 manifests.
The API inventory check also passed; all 49 public Simulation methods have docstrings.
API documentation work can therefore start by structuring existing content and connecting deployment.
These results establish success only within the current tools' check scope.

Not performed: rerunning all examples/public code blocks, new FDTD/GPU runs, revalidating historical numbers,
implementing new pages/generation pipelines, deployment, pushes, PRs or issue changes.
The media/version/API structure in this document is an implementation plan, not a claim that these features already exist.
