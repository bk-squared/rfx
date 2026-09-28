# Pre-declaration: a path admits only the inputs it declares it carries (2026-09-27)

Status: pre-declaration. Decided by the NU leader; the change of method was approved by the PI on 2026-09-27. This note is committed as the first commit of its PR, before any code, and the judges below are frozen from that commit.

## 1. What a user got, and why it kept happening

A `Simulation` is a declaration: materials, conductors, ports, sources, boundaries, a mesh and observers. Nine time-stepping paths run it:
- `run()` on a uniform, graded, subgridded, ADI or multi-device model;
- `forward()` on a uniform, graded or distributed graded model;
- `forward()` on an ADI model.

Each path reads only part of the declaration. An input a path does not read is solved as if it had not been declared, and nothing in the result says so:
- a Kerr block comes back linear on a graded `run()` (#1309);
- ADI solves μr = 4 as vacuum (#1308);
- a Floquet port on a graded mesh launches nothing (#1312);
- the subgridded lane solves magnetic x walls as electric walls, after passing production validation (#1311).

This lane has 24 such issues: #1221, #1257 and #1266, and 21 more that the path survey (#1338) and the reviews filed in the three days before this note (#1285–#1342). Fourteen share one cause (#1285, #1286, #1290, #1299, #1300, #1305, #1308–#1313, #1339, #1340): admission is a denylist. Each path lists what it refuses, and anything not listed passes through. Every new input therefore opens a silent drop on every path that was written before it, until someone finds it. This note reverses that. A path declares what it carries, and it refuses everything else before the first step. That is the PI's rule of 2026-09-24 ("a physics input a path does not implement is refused"), enforced by construction instead of case by case.

## 2. Decisions

1. **One declaration in `rfx/`.** A new module (suggested `rfx/api/_admission.py`) holds two things:
   - `DETECTORS`: for every input row of the #1338 table (an `(attribute, feature)` pair of the physics, observer or setting class), a predicate `active(sim) -> bool` that says whether the declaration switches that input on. For settings, active means a non-default value, e.g. `precision != 'float32'` or `mode != '3d'`.
   - `ADMITS`: for every lane, the set of input rows it admits. That is the rows whose cell on that lane is `carries` or `ignorable`; an ignorable cell is a setting that belongs to another solver, such as `adi_cfl_factor` on a Yee lane.

   Bookkeeping rows get no detector. The test table (`tests/contracts/path_disposition.py`) keeps its notes and its `wrong=` references. Its cell kinds must agree with `ADMITS`, and a contract test fails on any disagreement. There is one list of what a lane admits, and it lives in the code that enforces it.

   A detector reads only the static declaration. It never reads a traced array and never runs inside `jit`. An input that can arrive traced, such as a design-region ε override, is judged by what the model declares, not by its value.
2. **Admission at the point where the path is known.** After `_dispatch_plan` picks the lane (both `run` and `forward` modes), `active − ADMITS[lane]` must be empty, or the call raises before any kernel is built. The lane checked is the lane that actually runs, after any fallback: a waveguide port or a TFSF source with `devices=` runs on one device today, and it is judged there. The same check runs at every entry that reaches a kernel without `_dispatch_plan`:
   - `run_nonuniform_path` (`rfx/runners/nonuniform.py`);
   - a direct `run_uniform` (`rfx/runners/uniform.py`);
   - a direct `distributed_v2.run_distributed`;
   - the ADI route inside `_forward_from_materials` (column `fwd_adi`).
3. **One message, true by construction.** The error lists each refused input and names the lanes whose table cell for it is `carries`. Example: "Kerr χ³ is not carried by the graded run() lane; it is carried by: uniform run(), uniform forward(), run(devices=...)". No hand-written remedy can point to a second refusal, because the remedy is derived from the same table.
4. **Existing specific refusals stay.** #1240, #1297 and #1241 give specific reasons, and they still run. Admission is the backstop for everything they do not name. If an existing test pins a message that now comes from admission instead, the PR lists it. It does not silently rewrite it.
5. **What this PR closes, and what it does not.**
   - Every table cell that is `refuses(wrong=#N)` is a lane that drops the input today. On the #1338 head (59a86a3d) there are 50 such cells, under 12 issues. Each becomes a real refusal. Its strict xfail turns into an XPASS, and the PR converts it to a plain `refuses` in the same commit.
   - Issues whose every cell is one of the 50: #1285, #1286, #1308, #1309, #1310, #1311, #1312, #1313, #1339, #1340. Each closes as a refusal, but only after the PR reads the issue and finds no part of it outside the table.
   - Drop cells convert, but the issue stays open:
     - #1221, which also has nine cells where the lane carries the boundary but solves it in the wrong place;
     - #1299, whose calculator part remains.
   - Not closed here:
     - the calculator paths (#1290, #1299, #1300). They need their columns classified first, and a follow-up extends the same mechanism to them;
     - #1305, where the `devices=` fallback overrides an explicit setting. That is a fallback rule, not an input a lane lacks;
     - every cell where a lane carries an input but gets it wrong: #1221, #1266, #1302, #1303 in the table. That is the model-assembly and convention work, next note.
6. **Refusal is the minimum, not the goal.** Where a refused combination turns out to be needed (an example, the gallery, a paper), carrying it is a separate feature issue. It is not a reason to weaken admission.

## 3. Judges (frozen)

- **J1, nothing carried changes.** Every `carries` and `ignorable` cell of the #1338 cells file still passes. The locks under `tests/locks/` and `tests/contracts/test_example_fidelity_contract.py` pass unchanged. Admission only checks; it adds no numerics. A model whose inputs are all admitted gives a bit-identical result.
- **J2, the drops become refusals.** All 50 `refuses(wrong=#N)` cells raise the admission error before any kernel scan, with the input and the lane named. No other strict xfail changes state. The `carries(wrong=#N)` cells stay strict xfails.
- **J2b, the detectors see what the table builds.** For every cell of the cells file, the detector of that row fires on the model the cell builds. On the cells file's base model, no detector fires. This makes a detector that returns False forever fail a test.
- **J3, the examples tell us which ones were silently wrong.** For every script under `examples/` and `scripts/precompute_gallery_artifacts.py`, build its `Simulation` and run admission only. List every one that is now refused. A refused example was computing a silently wrong result before this PR. Stop and report before changing any example.
- **J4, the gate can fail.** These mutations must each turn the named tests red; all three are reported in the PR body:
  - (a) Admission disabled (the check returns empty): the converted cells.
  - (b) A lane's `ADMITS` gains an input it drops, with the admission call kept: that input's cell on that lane.
  - (c) A new `Simulation` attribute, or a new row, with no detector: the contract test.
  - (d) One detector made to return False, with the admission call kept: J2b and that row's converted cells.
- **J5, cost.** Admission adds no measurable time to a run: one host-side pass over the declaration. The PR states the time on the cells file.

## 4. Out of scope

- Classifying the seven calculator columns and extending admission to them (follow-up).
- One shared model assembly and one set of conventions (classes B and C: #1221, #1257 (closed), #1266, #1293, #1295 (PR #1341), #1298, #1302, #1303, #1306, #1342). That is the next design note.
- The NU kernel performance work for the uniform-kernel retirement rule.
