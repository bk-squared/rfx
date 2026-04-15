# Deferred — distributed NonUniformGrid + CPML / dispersion — 2026-04-15

## Status

Deferred out of the `nonuniform-completion` branch. Tracked here so the
scope doesn't get lost.

## What landed in Phase B (this branch)

`rfx/runners/distributed_nu.py` (new) + edits to `rfx/runners/distributed_v2.py`
and `rfx/api.py`. Enables multi-device `shard_map` FDTD with a
`NonUniformGrid` for the vacuum / PEC case.

Tests: `tests/test_distributed_nu_smoke.py`,
`tests/test_distributed_nu_kernel.py` (4/4 pass under
`XLA_FLAGS=--xla_force_host_platform_device_count=2`).

## What is NOT supported on the distributed + NU path

The runner-level guardrail in `rfx/runners/distributed_v2.py` and the
API-level guard in `rfx/api.py` raise `NotImplementedError` /
`ValueError` for:

1. **CPML on NU + distributed.** Reason: existing
   `_apply_cpml_*_distributed` kernels hard-code scalar `grid.dx` in y/z
   face curls. Reusing them for an anisotropic NU grid would be silently
   incorrect on y/z. A correct `_apply_cpml_*_distributed_nu` needs
   per-axis `dx`/`dy`/`dz` plumbing and a NU-aware `CPMLAxisParams`
   path. ~2–3 days of focused work, independently verifiable.

2. **Debye / Lorentz dispersion on NU + distributed.** Reason: the
   single-device NU runner (`rfx/nonuniform.py::_update_e_nu_dispersive`)
   does the ADE math correctly, but porting it to the distributed path
   means adding an `_update_e_local_nu_with_dispersion` kernel and
   threading ADE coefficient sharding through `shard_map` in_specs.
   ~1–2 days.

## Why deferred

The NonUniformGrid completion target (patch antenna / microstrip CPW /
single-cavity workloads, 0.4–1.3 M cells after graded-mesh reduction)
fits on a single A6000 48 GB with substantial headroom. Multi-GPU +
NonUniformGrid + CPML only becomes load-bearing for multi-element
arrays, dense via grids, or broadband time-stepping — all workloads
outside the current target.

Step 4 (capability coverage: DFT plane probe, lumped RLC, NTFF,
waveguide port on the NU path) is the actual blocker for the target
workloads and takes priority over distributed extensions.

## How to pick this up later

Branch off `main` (or wherever Phase B landed) as
`distributed-nu-cpml-dispersion`. Two independent sub-branches are
cleaner than one:

- `distributed-nu-cpml`: per-axis CPML apply kernels in
  `rfx/runners/distributed_nu.py` (mirror the existing
  `_apply_cpml_*_distributed` but take `CPMLAxisParams` and per-axis
  dx/dy/dz). Remove the runtime NotImplementedError in
  `distributed_v2.py` for `use_cpml and is_nu`. Add `tests/
  test_distributed_nu_cpml.py::test_graded_x_cpml_reflection_matches_single`.

- `distributed-nu-dispersive`: port `_update_e_nu_dispersive` to
  `_update_e_local_nu_with_dispersion`. Extend `_update_e_shmap` NU
  branch to thread Debye/Lorentz ADE coeffs. Add
  `tests/test_distributed_nu_dispersive.py`.

Both sub-branches must keep the uniform distributed path bit-identical
(`tests/test_distributed.py` 30/30 green).

## Reference commit

Phase B landing commit on `nonuniform-completion` branch: to be filled
in after the partial-Phase-B commit lands.
