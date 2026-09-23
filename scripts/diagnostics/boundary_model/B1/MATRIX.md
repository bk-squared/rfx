# B1 boundary measurements

The 24 × 20 × 16 mm boxes have the twelve B0 face declarations and source/port registrations.
Each entry point advances twelve seeded E/H and auxiliary-state patterns for two steps at dx = 1 mm.
The JSON records E/H zero planes in metres, auxiliary decay, constant-field attenuation, end coupling and periods; the comparisons use the declared boundary descriptor.

Kernel base: `3247dc0efdf91c571cdfecb043a9a47479c2dcd1`. B0: `798ec64e`.
E = tangential E zero; H = tangential H zero; A = absorber response; W = coupled ends. B0 letters refer to calls; B1 letters refer to fields.
Codes: a magnetic→electric; b1 dead face node; b2 shorted face node; c electric face absorbing; d periodic→wall/absorber; e period; f backing; g absorber on reflector; h magnetic plane. Supplemental codes: PEC (electric zero absent), feature (absorber requirement), absorber_type (constant-field loss without CPML auxiliary decay).

| Declaration | run | wire-fast | forward | sweep | nonuniform | subgridded | distributed | adi | gpu-query |
|---|---|---|---|---|---|---|---|---|---|
| pec | pass | pass | pass | pass | pass | REFUSED | pass | pass | pass |
| cpml | pass | pass | f | pass | pass | REFUSED | f | absorber_type | pass |
| upml | pass | pass | f | pass | REFUSED | REFUSED | REFUSED | REFUSED | pass |
| pmc-pec | b1,h | b1,h | b1,h | REFUSED | b1,h | REFUSED | b1,b2,h | a,b1 | b1,h |
| pmc-cpml | b1,h | b1,h | b1,f,h | REFUSED | b1,h | REFUSED | b1,f,g,h | REFUSED | b1,h |
| pec-zlo | pass | pass | f | pass | pass | REFUSED | PEC,c,f | REFUSED | pass |
| periodic-xy | e | REFUSED | e,f | e | d | REFUSED | REFUSED | REFUSED | e |
| tfsf | f,feature | REFUSED | f,feature | f,feature | pass | REFUSED | f,feature | REFUSED | f,feature |
| waveguide-cpml | f,feature | REFUSED | f,feature | f,feature | pass | REFUSED | f,feature | REFUSED | f,feature |
| waveguide-pmc | b1,f,h | REFUSED | b1,f,h | b1,f,h | b1,h | REFUSED | b1,f,h | REFUSED | b1,f,h |
| waveguide-pec | f | REFUSED | f | f | pass | REFUSED | f | REFUSED | f |
| floquet | f,feature | REFUSED | f,feature | f,feature | REFUSED | REFUSED | REFUSED | REFUSED | f,feature |

## B0 class → B1 field class

- pec / subgridded: MEASURED -> REFUSED.
- pmc-pec / run: x_lo EH -> H; x_hi EH -> H.
- pmc-pec / wire-fast: x_lo EH -> H; x_hi EH -> H.
- pmc-pec / forward: x_lo EH -> H; x_hi EH -> H.
- pmc-pec / sweep: MEASURED -> REFUSED.
- pmc-pec / nonuniform: x_lo EH -> H; x_hi EH -> H.
- pmc-pec / subgridded: MEASURED -> REFUSED.
- pmc-pec / gpu-query: x_lo E -> H; x_hi E -> H.
- pmc-cpml / run: x_lo EH -> H; x_hi EH -> H.
- pmc-cpml / wire-fast: x_lo EH -> H; x_hi EH -> H.
- pmc-cpml / sweep: MEASURED -> REFUSED.
- pmc-cpml / nonuniform: x_lo EH -> H; x_hi EH -> H.
- pmc-cpml / gpu-query: x_lo EH -> H; x_hi EH -> H.
- tfsf / distributed: x_lo EA -> A; x_hi EA -> A.
- waveguide-pmc / run: y_lo EH -> H; y_hi EH -> H; z_lo EH -> H; z_hi EH -> H.
- waveguide-pmc / forward: y_lo EH -> H; y_hi EH -> H; z_lo EH -> H; z_hi EH -> H.
- waveguide-pmc / sweep: y_lo EH -> H; y_hi EH -> H; z_lo EH -> H; z_hi EH -> H.
- waveguide-pmc / nonuniform: y_lo EH -> H; y_hi EH -> H; z_lo EH -> H; z_hi EH -> H.
- waveguide-pmc / distributed: y_lo EH -> H; y_hi EH -> H; z_lo EH -> H; z_hi EH -> H.
- waveguide-pmc / gpu-query: y_lo EH -> H; y_hi EH -> H; z_lo EH -> H; z_hi EH -> H.

## Stopped comparisons

- Subgrid B0 comparison: trace_matrix.py uses z=(6,10) mm; selected B0 trace_matrix_subgrid.py uses z=(0,12) mm. B1 retains the required trace_matrix.py build; all twelve centered-slab declarations refuse.
- Forward dielectric smoothing: forward() has no subpixel_smoothing argument.

The first distributed scratch pass gathered one ghosted concatenation as a single slab; the selected records use two slabs, strip one ghost row per side per slab, and trim the high alignment pad. No first-pass distributed classification is in this baseline.
The selected ADI records compare constant-field amplitudes with the same two-step implicit scan at zero conductivity. Earlier scratch classifications used a unit-amplitude reference and are excluded. A held-zero E or H plane is excluded from the coupled-end classification.

Conclusion: leader fills.
