# B1 boundary measurements

The 24 × 20 × 16 mm boxes have the twelve B0 face declarations and source/port registrations.
Each entry point advances twelve seeded E/H and auxiliary-state patterns for two steps at dx = 1 mm.
The JSON records E/H zero planes in metres, auxiliary decay, constant-field attenuation, end coupling and periods; the comparisons use the declared boundary descriptor.

Kernel base: `cca8ee5bae9f43b93423962e1888c0829b21c8a3`. B0: `798ec64e`.
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

- pec / subgridded: MEASURED -> REFUSED; harness change: B0 refinement z=(0,12) mm; B1 z=(6,10) mm; not a change on main.
- pmc-pec / run: x_lo EH -> H; x_hi EH -> H.
- pmc-pec / wire-fast: x_lo EH -> H; x_hi EH -> H.
- pmc-pec / forward: x_lo EH -> H; x_hi EH -> H.
- pmc-pec / sweep: MEASURED -> REFUSED.
- pmc-pec / nonuniform: x_lo EH -> H; x_hi EH -> H.
- pmc-pec / subgridded: MEASURED -> REFUSED; harness change: B0 refinement z=(0,12) mm; B1 z=(6,10) mm; not a change on main.
- pmc-pec / gpu-query: x_lo E -> H; x_hi E -> H.
- pmc-cpml / run: x_lo EH -> H; x_hi EH -> H.
- pmc-cpml / wire-fast: x_lo EH -> H; x_hi EH -> H.
- pmc-cpml / sweep: MEASURED -> REFUSED.
- pmc-cpml / nonuniform: x_lo EH -> H; x_hi EH -> H.
- pmc-cpml / gpu-query: x_lo EH -> H; x_hi EH -> H.
- tfsf / distributed: HARNESS CORRECTION (Addendum 4): use the recorded full grid without slab stripping; B1 x_lo/x_hi E backings absent -> present; (x_lo,f)/(x_hi,f) removed; not a change on main.
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

What the fields show on main 76f68f9f (after #1205 and #1213). No uniform-lane entry point zeroes tangential E on a declared magnetic face any more. Every lane that accepts a magnetic face still zeroes tangential H half a cell inside it (h) and leaves the face-node plane cut off from the interior (b1). Two lanes still put an electric operation on it: the distributed lane shorts the face-node plane (b2, pmc-pec) or absorbs on the face (g, pmc-cpml), and ADI solves it as an electric wall (a). The vmap sweep refuses the magnetic x-face declarations but accepts a waveguide run with magnetic transverse faces (b1, f, h). A periodic axis is one cell longer than declared on every lane that accepts it (e); the graded lane solves periodic faces as walls (d). forward() puts no electric wall behind its absorbers while run() does (f); on the distributed lane a declared electric face absorbs when the other faces absorb (c, pec-zlo). TFSF, waveguide and Floquet runs rewrite the declared transverse faces on the uniform lanes (feature), as B0 found. Subgrid: with the contract's centred refinement slab every declaration is refused; with B0's own build (z = 0-12 mm, Addendum 1) pec and pmc-pec run and ten declarations are refused, and the classifier does not yet read the fine-grid fields, so the subgrid lane is classified in B4. Addendum 2 re-ran every cell on 76f68f9f: no class, plane, period or refusal changed.
