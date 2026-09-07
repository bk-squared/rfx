"""rfx.fdfd -- differentiable frequency-domain solvers.

Two layers:

* :mod:`rfx.fdfd.linear_solve` -- ``sparse_solve``: a host-factorised sparse
  direct solve that JAX can differentiate in forward AND reverse mode through
  the implicit-function rule (one extra transposed solve per gradient, no
  time history). This is the reusable engine.
* :mod:`rfx.fdfd.hplane` -- a 2-D H-plane TE_n0 Helmholtz solver with exact
  discrete transparent ports, matching the independent referee node-for-node
  and differentiable in frequency, in a per-node permittivity map and in
  the iris aperture widths (body-fitted grid: edges stay on nodes, the
  nodes move with the width, so the shape derivative is smooth).

Both require x64.
"""
from rfx.fdfd.linear_solve import sparse_solve, sparse_matvec, clear_factor_cache
from rfx.fdfd.hplane import (
    HPlaneSpec, HPlaneModel, build, assemble, solve, s_params,
    richardson_first_order,
)

__all__ = [
    "sparse_solve", "sparse_matvec", "clear_factor_cache",
    "HPlaneSpec", "HPlaneModel", "build", "assemble", "solve", "s_params",
    "richardson_first_order",
]
