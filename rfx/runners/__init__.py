"""Runner modules for the Simulation run paths.

Distributed lanes (#1038 leg 6, PI decision 2026-09-15). ``distributed_v2.py``
(``shard_map``) is the production distributed runner: ``Simulation.run(devices=...)``
dispatches there for uniform AND non-uniform grids, and it is the single
development trunk for distributed work.

``rfx.runners.distributed`` (v1, ``jax.pmap``) is the legacy lane. It is NOT
merged into v2 -- the two are not bit-identical (max |delta| 2.794e-09 on a
9.4145e-03 peak) and they disagree on the odd-``nx`` rule (v1 refuses, v2 pads.)
It is kept as an internal dependency: ``distributed_v2`` imports twelve domain
splitting / CPML names from it (including ``_split_materials``, which
``rfx/api/_execute.py`` also imports directly) and delegates to its
``run_distributed`` as the ``n_devices == 1`` fast path. It stays importable by
full module path -- ``from rfx.runners.distributed import run_distributed`` --
and is deliberately NOT exported from this package, so the package-level name
``run_distributed`` no longer resolves to a runner that ``sim.run()`` does not
use. Migration: ``Simulation.run(devices=[...])``, or the full module path.
"""

from rfx.runners.uniform import run_uniform
from rfx.runners.nonuniform import run_nonuniform_path
from rfx.runners.subgridded import run_subgridded_path

__all__ = ["run_uniform", "run_nonuniform_path", "run_subgridded_path"]
