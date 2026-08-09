# Palace setups — X-band inset-fed patch (four-solver cross-validation)

The Palace (AWS open-source FEM) leg of the four-solver patch study whose
complete record is [`docs/crossval/patch_xband_4solver.md`](../../../docs/crossval/patch_xband_4solver.md).
Copied verbatim from the campaign branch
(`research/calibration-inverse`, `scripts/research/calibration/crossval/palace_patch/`)
so the paper's `validation/` tree carries every solver's setup.

| File | Purpose |
|---|---|
| `mesh_patch.py` | builds the Gmsh mesh of the patch (shielded and radiating variants) |
| `patch_eigen_shielded.json` | **shielded eigenmode** config — the claims-bearing leg (9.199 GHz, agrees with rfx/openEMS/CST to 1.0 %) |
| `patch_eigen.json` | open-box eigenmode variant |
| `patch_s11.json` | driven-port |S11| config — **excluded from radiating comparisons** (first-order absorbing boundary on a tight box biases the dip low; see the record, §4) |

Palace version used in the campaign: 0.16.0. Raw outputs, the falsification
ledger, and the reproduction guide live on the campaign branch under
`scripts/research/calibration/crossval/`.
