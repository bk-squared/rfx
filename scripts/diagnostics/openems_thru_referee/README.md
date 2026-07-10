# openEMS external thru referee — issue #313 final gate

Independent cross-solver anchor for the reference-plane port-wave architecture
(PR #320) on the canonical 16 mm air-microstrip thru
(`tests/test_refplane_port_waves.py` fixture geometry, coordinate-identical
model; the one structural difference is the port family — rfx vertical wire
point-feeds vs openEMS 3 mm MSL launch spans — stated below where it matters).

Run: VESSL `369367246600` (2026-07-10, remilab-c0, source-built openEMS per the
`vessl_crossval_external.yaml` bootstrap; solve time 8.1 s at dx=0.5 mm).
Full log: `docs/research_notes/vessl_logs/openems_thru_referee_369367246600.log`
(local-only). Result: `results/thru_openems_dx500um.json` (committed; note: the
final closing brace was restored during log extraction — the container stream
dropped the last line at stop; all numeric content is verbatim).

## Comparison, 3–7 GHz (17 openEMS points / 9 rfx bins)

| Quantity | rfx reference-plane path (PR #320) | openEMS (this run) | distance |
|---|---|---|---|
| \|S21\| band | 0.983 – 0.998 (falling with f) | 0.973 – 1.034 (falling with f) | rfx band sits inside the openEMS envelope; band means 0.990 vs 0.9965 (0.7%) |
| \|S11\| band | 0.034 – 0.130 | 0.030 – 0.160 | same mild-mismatch class, both converging near 6–7 GHz |
| Port/line impedance | two-plane Zc 47.9 – 48.6 Ω | port Z0 mean 55.5 Ω | different definitions and planes (line invariant vs MSL-port V/I); same ~50 Ω class |
| Group delay | ~70 ps (full 16 mm span + vertical feed posts) | 39.6 – 42.0 ps | not comparable as configured: openEMS de-embeds to MSL reference planes inside the span (~12 mm effective at c ≈ 41 ps) and has no vertical posts; a matched-span rerun would be needed for a delay comparison |

The pre-#320 port-cell extraction read \|S21\| = 0.52 – 0.67 on the same rfx
fixture — 33–47% from both the rfx plane path and this referee.

openEMS values above unity at the low bins (max 1.034) are within that model's
own coarse-mesh MSL-port normalization envelope at dx=0.5 mm; a finer-mesh
rerun (`--dx-um` flag) is the first step if that envelope ever matters.

This record is a bracket, not a judgment: it states where the two solvers land
on the same geometry class. No committed rfx gate cites these numbers; the
battery's gates remain anchored to the in-repo closed-box flux referee.

Note: the VESSL YAML for this lane is local-only (repo-wide `**/vessl*.yaml`
ignore; maintained harnesses are the tracked exceptions). It is a verbatim
derivative of `scripts/vessl_crossval_external.yaml` with the rfx-runtime
stage removed; the README above records every deviation, so the lane is
reproducible from the template + this document.
