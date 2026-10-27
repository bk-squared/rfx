"""Non-uniform Yee grid FDTD runner.

Supports spatially-varying dx, dy, dz profiles. The most common use
is dz-graded meshes for thin-substrate structures where z-resolution
must be fine near the substrate but coarse in the air region; dx/dy
graded meshes additionally allow fine cells near metal edges
(fringing-field physics in patch antennas, microstrip filters, etc.)
without paying the cost of a uniform fine mesh everywhere.

Back-compat: `make_nonuniform_grid(domain_xy, dz_profile, dx, ...)`
with a scalar `dx` still produces a uniform-xy grid; only `dz` is
graded. Pass `dx_profile=` / `dy_profile=` to enable per-cell x/y
spacing.

Uses update_h_nu / update_e_nu from core/yee.py with pre-computed
per-axis inverse spacing arrays. Fully JIT-compiled via jax.lax.scan.
"""

from __future__ import annotations

from rfx.measurement import assemble_graded_result as _assemble_nu_result

from rfx.model.graded_grid import (  # noqa: F401 -- moved; re-exported
    C0,
    NonUniformGrid,
)
from rfx.model import (  # noqa: F401 -- moved; re-exported
    e_node_dual_spacing_at,
    make_band_profile,
    make_current_source,
    make_nonuniform_grid,
    make_z_profile,
    position_to_index,
)


from rfx.stepping import run_nonuniform as run_nonuniform  # noqa: E402
from rfx.stepping import run_nonuniform_until_decay as run_nonuniform_until_decay  # noqa: E402
