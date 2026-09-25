"""Runner modules for the Simulation run paths.

Distributed lanes (#1038 leg 6, PI decision 2026-09-15). ``distributed_v2.py``
(``shard_map``) is the production distributed runner: ``Simulation.run(devices=...)``
dispatches there for uniform AND non-uniform grids, and it is the single
development trunk for distributed work.

``rfx.runners.distributed`` was the v1 ``jax.pmap`` runner, removed in #1296.
Its helpers live in ``_distributed_common``; the module remains for one release
as a re-export whose ``run_distributed`` raises and names the replacement,
``Simulation.run(devices=[...])`` or ``rfx.runners.distributed_v2.run_distributed``.
Neither ``run_distributed`` is exported from this package.
"""

from rfx.runners.uniform import run_uniform
from rfx.runners.nonuniform import run_nonuniform_path
from rfx.runners.subgridded import run_subgridded_path

__all__ = ["run_uniform", "run_nonuniform_path", "run_subgridded_path"]
