# Boundary-site inventory

This is a code inventory; no effect on results.

Source: `798ec64e5cda057318664bc431e528819f9e371a`. 1518 executable sites across 79 files.

One site is one file, line, symbol; comments and docstrings are excluded. Argument declarations are DECLARE, bare forwarding is PASS, flag transformations are DERIVE, other reads are DECIDE, wall/absorber calls are APPLY. `init_*` calls are DERIVE. Mask calls are retained as a superset; the row notes that array-edge coverage depends on the mask. The CSV retains the enclosing function and source line. Shared distributed helpers are listed once under distributed-v1; their callers occur in their own lanes.

| Lane | DECLARE | DERIVE | APPLY | DECIDE | PASS | Total |
|---|---:|---:|---:|---:|---:|---:|
| uniform-scan | 15 | 29 | 8 | 24 | 70 | 146 |
| fast-path | 0 | 0 | 1 | 1 | 1 | 3 |
| forward | 0 | 8 | 0 | 7 | 17 | 32 |
| nonuniform | 13 | 33 | 4 | 24 | 53 | 127 |
| vmap-sweep | 3 | 8 | 4 | 8 | 13 | 36 |
| subgridded | 1 | 20 | 20 | 76 | 31 | 148 |
| distributed-v1 | 9 | 1 | 17 | 14 | 4 | 45 |
| distributed-v2 | 3 | 7 | 6 | 9 | 13 | 38 |
| distributed-nu | 6 | 22 | 6 | 7 | 14 | 55 |
| adi | 0 | 0 | 6 | 0 | 0 | 6 |
| tfsf-aux | 0 | 0 | 4 | 1 | 0 | 5 |
| waveguide-lane | 15 | 7 | 0 | 20 | 46 | 88 |
| rcs | 0 | 0 | 0 | 0 | 4 | 4 |
| floquet | 1 | 1 | 0 | 5 | 0 | 7 |
| probes-reference | 2 | 2 | 6 | 0 | 4 | 14 |
| preflight | 8 | 16 | 0 | 46 | 45 | 115 |
| materials | 3 | 0 | 0 | 2 | 2 | 7 |
| ports | 4 | 9 | 0 | 39 | 6 | 58 |
| probes | 5 | 2 | 0 | 2 | 15 | 24 |
| farfield | 0 | 0 | 0 | 0 | 1 | 1 |
| io | 0 | 0 | 0 | 4 | 0 | 4 |
| other | 108 | 87 | 10 | 163 | 187 | 555 |

## APPLY sites

| Site | Symbol | Lane | Applied operation |
|---|---|---|---|
| `rfx/adi.py:272` | _apply_pec_2d | adi | Ez zeroed at all four array edges; optional conductor mask. |
| `rfx/adi.py:327` | _apply_pec_2d | adi | Ez zeroed at all four array edges; optional conductor mask. |
| `rfx/adi.py:859` | _apply_pec_3d | adi | Tangential E zeroed at all six array faces; optional conductor edge masks. |
| `rfx/adi.py:864` | _apply_pec_3d | adi | Tangential E zeroed at all six array faces; optional conductor edge masks. |
| `rfx/adi.py:881` | _apply_pec_3d | adi | Tangential E zeroed at all six array faces; optional conductor edge masks. |
| `rfx/adi.py:886` | _apply_pec_3d | adi | Tangential E zeroed at all six array faces; optional conductor edge masks. |
| `rfx/core/yee.py:184` | periodic | other | `_diff_fwd_o`: `if periodic[axis]:` |
| `rfx/core/yee.py:186` | bloch | other | `_diff_fwd_o`: `if bloch is not None:` |
| `rfx/core/yee.py:187` | bloch | other | `_diff_fwd_o`: `nxt = nxt * bloch[axis]` |
| `rfx/core/yee.py:191` | periodic | other | `_diff_fwd_o`: `if periodic[axis]:` |
| `rfx/core/yee.py:209` | periodic | other | `_diff_bwd_o`: `if periodic[axis]:` |
| `rfx/core/yee.py:211` | bloch | other | `_diff_bwd_o`: `if bloch is not None:` |
| `rfx/core/yee.py:212` | bloch | other | `_diff_bwd_o`: `prv = prv * bloch[axis].conjugate()` |
| `rfx/core/yee.py:216` | periodic | other | `_diff_bwd_o`: `if periodic[axis]:` |
| `rfx/core/yee.py:248` | bloch | other | `update_h`: `if bloch is not None and so != 2:` |
| `rfx/core/yee.py:460` | bloch | other | `update_e`: `if bloch is not None and so != 2:` |
| `rfx/nonuniform.py:2537` | apply_cpml_h | nonuniform | CPML H curl correction on selected axes with per-face profiles. |
| `rfx/nonuniform.py:2544` | apply_pmc_faces | nonuniform | Tangential H at low index 0 / high index -2. |
| `rfx/nonuniform.py:2596` | apply_cpml_e | nonuniform | CPML E curl correction on selected axes with per-face profiles. |
| `rfx/nonuniform.py:2602` | apply_pec | nonuniform | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/probes/probes.py:1607` | apply_cpml_h | probes-reference | CPML H curl correction on selected axes with per-face profiles. |
| `rfx/probes/probes.py:1621` | apply_cpml_e | probes-reference | CPML E curl correction on selected axes with per-face profiles. |
| `rfx/probes/probes.py:1624` | apply_pec | probes-reference | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/probes/probes.py:1947` | apply_cpml_h | probes-reference | CPML H curl correction on selected axes with per-face profiles. |
| `rfx/probes/probes.py:1961` | apply_cpml_e | probes-reference | CPML E curl correction on selected axes with per-face profiles. |
| `rfx/probes/probes.py:1964` | apply_pec | probes-reference | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/runners/_distributed_common.py:518` | pmc_faces | distributed-v1 | `apply_pmc_face_shmap`: `if not pmc_faces:` |
| `rfx/runners/distributed.py:559` | pad_x | distributed-v1 | `_apply_pec_local`: `last_real = nx_local_with_ghost - 1 - ghost - pad_x  # last real cell index` |
| `rfx/runners/distributed.py:584` | pmc_faces | distributed-v1 | `_apply_pmc_local`: `if not pmc_faces:` |
| `rfx/runners/distributed.py:596` | pmc_faces | distributed-v1 | `_apply_pmc_local`: `if "y_lo" in pmc_faces:` |
| `rfx/runners/distributed.py:599` | pmc_faces | distributed-v1 | `_apply_pmc_local`: `if "y_hi" in pmc_faces:` |
| `rfx/runners/distributed.py:602` | pmc_faces | distributed-v1 | `_apply_pmc_local`: `if "z_lo" in pmc_faces:` |
| `rfx/runners/distributed.py:605` | pmc_faces | distributed-v1 | `_apply_pmc_local`: `if "z_hi" in pmc_faces:` |
| `rfx/runners/distributed.py:611` | pad_x | distributed-v1 | `_apply_pmc_local`: `last_real = nx_local_with_ghost - 1 - ghost - pad_x` |
| `rfx/runners/distributed.py:614` | pmc_faces | distributed-v1 | `_apply_pmc_local`: `if "x_lo" in pmc_faces:` |
| `rfx/runners/distributed.py:619` | pmc_faces | distributed-v1 | `_apply_pmc_local`: `if "x_hi" in pmc_faces:` |
| `rfx/runners/distributed.py:753` | pad_x | distributed-v1 | `_apply_cpml_e_distributed`: `x_hi_edge = g + pad_x` |
| `rfx/runners/distributed.py:1028` | pad_x | distributed-v1 | `_apply_cpml_h_distributed`: `x_hi_edge = g + pad_x` |
| `rfx/runners/distributed.py:1553` | _apply_cpml_h_distributed | distributed-v1 | CPML H correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed.py:1571` | _apply_pmc_local | distributed-v1 | Tangential H on requested PMC faces, with rank-local x indexing. |
| `rfx/runners/distributed.py:1587` | _apply_cpml_e_distributed | distributed-v1 | CPML E correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed.py:1699` | _apply_pmc_local | distributed-v1 | Tangential H on requested PMC faces, with rank-local x indexing. |
| `rfx/runners/distributed.py:1733` | _apply_pec_local | distributed-v1 | Tangential E at physical x end-rank faces and local y/z faces in the pmap lane. |
| `rfx/runners/distributed_nu.py:1245` | pad_x | distributed-nu | `_apply_cpml_e_local_nu`: `x_hi_edge = g + pad_x` |
| `rfx/runners/distributed_nu.py:1497` | pad_x | distributed-nu | `_apply_cpml_h_local_nu`: `x_hi_edge = g + pad_x` |
| `rfx/runners/distributed_nu.py:2369` | _apply_cpml_h_local_nu | distributed-nu | CPML H correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed_nu.py:2437` | _apply_cpml_e_local_nu | distributed-nu | CPML E correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed_nu.py:2494` | _apply_cpml_h_shmap | distributed-nu | CPML H correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed_nu.py:2524` | _apply_cpml_e_shmap | distributed-nu | CPML E correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed_v2.py:259` | _apply_cpml_e_distributed | distributed-v2 | CPML E correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed_v2.py:353` | _apply_cpml_h_distributed | distributed-v2 | CPML H correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed_v2.py:1285` | _apply_cpml_h_shmap | distributed-v2 | CPML H correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed_v2.py:1312` | _apply_cpml_e_shmap | distributed-v2 | CPML E correction in distributed slabs, with end-rank selection on x. |
| `rfx/runners/distributed_v2.py:1338` | apply_pec_mask_shmap | distributed-v2 | PEC edge-mask E zeroing under shard_map, including any mask/outer-edge intersection. |
| `rfx/runners/distributed_v2.py:1453` | apply_pec_mask_shmap | distributed-v2 | PEC edge-mask E zeroing under shard_map, including any mask/outer-edge intersection. |
| `rfx/simulation.py:1718` | apply_upml_h | uniform-scan | UPML H update with component damping coefficients. |
| `rfx/simulation.py:1729` | apply_cpml_h | uniform-scan | CPML H curl correction on selected axes with per-face profiles. |
| `rfx/simulation.py:1767` | apply_pmc_faces | uniform-scan | Tangential H at low index 0 / high index -2. |
| `rfx/simulation.py:1802` | apply_upml_e | uniform-scan | UPML E update with component damping coefficients. |
| `rfx/simulation.py:1846` | apply_cpml_e | uniform-scan | CPML E curl correction on selected axes with per-face profiles. |
| `rfx/simulation.py:1863` | pec_axes | uniform-scan | `core_step`: `if pec_axes:` |
| `rfx/simulation.py:1864` | apply_pec | uniform-scan | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/simulation.py:1866` | apply_pec_faces | uniform-scan | Tangential E at specified node-array faces. |
| `rfx/simulation.py:2554` | precompute_coeffs | fast-path | Bake tangential-E zeros in Ca/Cb at both array ends for pec_axes. |
| `rfx/sources/tfsf_2d.py:413` | _apply_cpml_h | tfsf-aux | CPML H correction in TFSF auxiliary grid. |
| `rfx/sources/tfsf_2d.py:438` | _apply_cpml_e | tfsf-aux | CPML E correction in TFSF auxiliary grid. |
| `rfx/sources/tfsf_2d.py:488` | _apply_cpml_h | tfsf-aux | CPML H correction in TFSF auxiliary grid. |
| `rfx/sources/tfsf_2d.py:516` | _apply_cpml_e | tfsf-aux | CPML E correction in TFSF auxiliary grid. |
| `rfx/subgridding/disjoint_3d.py:309` | apply_pec | subgridded | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/subgridding/disjoint_3d.py:355` | apply_pec | subgridded | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/subgridding/disjoint_3d.py:748` | apply_pec | subgridded | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/subgridding/jit_runner.py:1852` | apply_cpml_h | subgridded | CPML H curl correction on selected axes with per-face profiles. |
| `rfx/subgridding/jit_runner.py:1979` | apply_cpml_e | subgridded | CPML E curl correction on selected axes with per-face profiles. |
| `rfx/subgridding/jit_runner.py:1981` | apply_pec | subgridded | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/subgridding/jit_runner.py:1989` | apply_pec_mask | subgridded | E on realized PEC mask edges; array-edge intersection depends on input. |
| `rfx/subgridding/jit_runner.py:2012` | apply_pec_mask | subgridded | E on realized PEC mask edges; array-edge intersection depends on input. |
| `rfx/subgridding/jit_runner.py:2014` | apply_pec_faces | subgridded | Tangential E at specified node-array faces. |
| `rfx/subgridding/jit_runner.py:2144` | apply_pec | subgridded | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/subgridding/jit_runner.py:2153` | apply_pec_faces | subgridded | Tangential E at specified node-array faces. |
| `rfx/subgridding/runner.py:116` | apply_cpml_h | subgridded | CPML H curl correction on selected axes with per-face profiles. |
| `rfx/subgridding/runner.py:127` | apply_cpml_e | subgridded | CPML E curl correction on selected axes with per-face profiles. |
| `rfx/subgridding/runner.py:129` | apply_pec | subgridded | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/subgridding/runner.py:137` | apply_pec_mask | subgridded | E on realized PEC mask edges; array-edge intersection depends on input. |
| `rfx/subgridding/runner.py:148` | apply_pec_mask | subgridded | E on realized PEC mask edges; array-edge intersection depends on input. |
| `rfx/subgridding/sbp_sat_2d.py:216` | _apply_pec_2d | subgridded | Ez zeroed at all four array edges; optional conductor mask. |
| `rfx/subgridding/sbp_sat_2d.py:223` | _apply_pec_2d | subgridded | Ez zeroed at all four array edges; optional conductor mask. |
| `rfx/subgridding/sbp_sat_3d.py:180` | apply_pec | subgridded | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |
| `rfx/subgridding/sbp_sat_3d.py:188` | apply_pec_mask | subgridded | E on realized PEC mask edges; array-edge intersection depends on input. |
| `rfx/vmap_sweep.py:615` | apply_cpml_h | vmap-sweep | CPML H curl correction on selected axes with per-face profiles. |
| `rfx/vmap_sweep.py:622` | apply_cpml_e | vmap-sweep | CPML E curl correction on selected axes with per-face profiles. |
| `rfx/vmap_sweep.py:627` | pec_axes | vmap-sweep | `step_fn`: `if pec_axes:` |
| `rfx/vmap_sweep.py:628` | apply_pec | vmap-sweep | Tangential E at both node-array ends on named axes; z also zeros high ghost Ez. |

Conclusion: leader fills.
