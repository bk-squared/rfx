# Hybrid adjoint custom_vjp crosswalk

Status: planning support doc

## Brownfield touchpoints

Primary touchpoints for the project:

- `rfx/simulation.py`
- `rfx/core/yee.py`
- `rfx/boundaries/cpml.py`
- `rfx/materials/debye.py`
- `rfx/materials/lorentz.py`
- `rfx/optimize.py`
- `rfx/topology.py`

## Phase 1A seam target

The planning target is not “all FDTD math,” but the **canonical scan/step seam** in `simulation.py`.

Phase 1A should answer:

- what is the canonical scan body?
- what must be replayed?
- what state is residual vs recomputed?
- where can an experimental hybrid boundary live without polluting all run paths?

## Phase 1B correctness target

The first correctness target is:

- uniform
- lossless
- PEC-only
- simple scalar objective on `time_series`
- pure AD vs hybrid gradient comparison

## Explicit fallback boundary

Unsupported physics in early phases must route to pure AD rather than silently entering the hybrid lane.
