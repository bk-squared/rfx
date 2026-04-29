# WR-90 waveguide-port field comparison harness

Cell-by-cell triple cross-validation between **rfx**, **OpenEMS**, and
**Meep** at WR-90 port planes. Used to localise the residual
per-frequency `|S11|` oscillation in rfx PEC-short reflection that
patch-level fixes have failed to close after 7+ sessions (see status
in `docs/agent-memory/rfx-known-issues.md`).

## Why this lives here, not in `scripts/spikes/<date>/`

`scripts/spikes/<date>/` is reserved for dated experimental ablations
that may be deleted later. This harness is the **stable** field-level
diagnostic surface: every architectural candidate after 2026-04-28
must report numbers from these scripts before being considered viable
or refuted. Promoting it out of `spikes/` makes that contract explicit.

## Tools

| script | role |
|---|---|
| `s11_from_dumps.py` | Apples-to-apples `|S11(f)|` from independent simulator dumps. Identical V/I projection (TE10 modal sin(πy/a) overlap) on rfx, OpenEMS, and (optionally) Meep dump fields at the same `mon_left` plane. |
| `dump_compare_openems_vs_rfx.py` | Cell-by-cell field comparison at `source_plane` and `mon_left_plane`. Inspects HDF5 schema, plots per-frequency E_z and H_y on a shared mesh, and reports the divergence step (E sample / H sample / V overlap / I overlap). |
| `field_shape_compare.py` | TE10 mode-shape projection at the port plane. Quantifies how closely the simulated field matches the analytic continuous-coordinate `sin(πy/a)` template, separated from the V/I post-processing. |
| `per_freq_oscillation_viz.py` | Frequency-by-frequency visualisation of the rfx vs reference `|S11|` oscillation. Used for at-a-glance regression sanity checks. |
| `geometry_visualization.py` | Static HTML/image report of the rfx vs OpenEMS vs Meep canonical geometry overlay (port positions, mon planes, PEC short). |

All five scripts treat `examples/crossval/11_waveguide_port_wr90.py`
as the source-of-truth for canonical constants (`A_WG`, `B_WG`,
`PORT_LEFT_X`, `MON_LEFT_X`, `MON_RIGHT_X`, `PEC_SHORT_X`, `FREQS_HZ`,
`CPML_LAYERS`, `NUM_PERIODS_LONG`). Drift between the comparator and
crossval/11 is therefore impossible by construction — change the
crossval/11 constants and the comparator follows.

## Reference dumps

OpenEMS HDF5 reference dumps are produced by the companion script in
the `microwave-energy` repo:

```bash
python /root/workspace/byungkwan-workspace/research/microwave-energy/openems_simulation/wr90_sparam_reference.py \
  --resolutions 1 \
  --geometries pec_short \
  --output /tmp/openems_wr90/openems_r1_pec_short.json \
  --dump-port-fields \
  --dump-output-dir /tmp/openems_wr90/dumps_r1 \
  --threads 4
```

The `--dump-port-fields` flag tells OpenEMS to emit
`{E,H}field_{source_plane,mon_left_plane}.h5` per resolution under the
`--dump-output-dir`. `s11_from_dumps.py` and
`dump_compare_openems_vs_rfx.py` consume that directory directly via
their `--openems-dump-dir` arg. (Matching Meep dumps come from
`microwave-energy/meep_simulation/wr90_sparam_reference.py` with the
same dump flag; they are optional.)

## Running

Stable apples-to-apples `|S11|` recipe at R=1 (1 mm cells, ~30 cells/λ):

```bash
python scripts/diagnostics/wr90_port/s11_from_dumps.py \
  --openems-dump-dir /tmp/openems_wr90/dumps_r1 \
  --R 1
```

Cell-by-cell field comparison at the same plane:

```bash
python scripts/diagnostics/wr90_port/dump_compare_openems_vs_rfx.py \
  --compare /tmp/openems_wr90/dumps_r1 \
  --R 1
```

Outputs land under `out/` and `out_compare/` (gitignored — see
`scripts/spikes/2026-04-28/.gitignore` for the pattern; mirror it here
when adding new output directories).

## 2026-04-28 reference baseline

Apples-to-apples dump-derived `|S11|` spread at the canonical
`mon_left` plane, R=1, with the cell-centred TE10 V/I recipe:

| simulator | min | mean | max | spread | client-side half-step? |
|---|---:|---:|---:|---:|---|
| OpenEMS | 0.9966 | 0.9996 | 1.0003 | 0.0036 | **NO** — applied internally by the OpenEMS dump writer |
| Meep    | 0.9960 | 1.0000 | 1.0112 | 0.0152 | **NO** — applied internally by the Meep dump writer |
| rfx     | 0.9916 | 0.9995 | 1.0082 | 0.0166 | **YES** — `s11_from_dumps.py` applies it on the rfx path |

Per-tool convention is verified by
``scripts/diagnostics/wr90_port/cross_tool_half_step_audit.py`` at R=1:
adding the `exp(+jω·dt/2)` correction client-side leaves OpenEMS and
Meep essentially unchanged or marginally worse (0.0036 → 0.1245 on
OE, 0.0152 → 0.0182 on Meep), while it drops rfx 0.1326 → 0.0166.
That asymmetry is the entire reason `s11_from_dumps.py` applies the
correction only on the rfx path and not on the reference paths.

The rfx number used to read 0.1326 spread before the comparator
fix landed: the dump-derived recipe was missing the same correction
that the production extractor `compute_waveguide_s_matrix` always
applied through `rfx/sources/waveguide_port.py::_co_located_current_spectrum`.
The full diagnostic chain that traced this lives in commit
`2fb9b76` and the four scripts named `pec_short_position_sweep`,
`boundary_cell_trim`, `h_normal_average_test`, and
`production_vs_raw_same_sim`. The historical 0.1326 number
appearing in earlier research notes and codex attempts under
`scripts/spikes/2026-04-28/refuted_codex_archive/` is a comparator
artefact, not a real rfx FDTD residual.

## Investigation pointers

- Full 2026-04-28 codex investigation:
  `docs/research_notes/2026-04-28_codex_arch_attempts.md`
- Cross-session candidate list and refutation log: memory
  `project_wr90_architectural_candidates.md`
- Known-issue entry with all closed sub-bullets:
  `docs/agent-memory/rfx-known-issues.md` (PEC-short item)
- Archived refuted source/probe code:
  `scripts/spikes/2026-04-28/refuted_codex_archive/`

## Adding a new architectural candidate

1. Implement the candidate in a feature branch (or as a spike under
   `scripts/spikes/<today>/`).
2. Reuse this harness — do **not** spin up a parallel comparator. If
   you need a new dispatch knob (e.g. `mode_profile=`), add it to the
   relevant comparator script with a sensible default that preserves
   the canonical baseline.
3. Report numbers via `s11_from_dumps.py` at R=1 and R=2 against
   OpenEMS and Meep on the same `mon_left` plane.
4. Land the verdict in
   `docs/agent-memory/rfx-known-issues.md` (PEC-short entry) and
   memory `project_wr90_architectural_candidates.md` whether the
   candidate passes or fails.
