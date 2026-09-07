"""rfx.fdfd -- differentiable frequency-domain solvers.

Layers, bottom up (all require x64):

* :mod:`rfx.fdfd.linear_solve` -- ``sparse_solve``: a host-factorised sparse
  direct solve that JAX differentiates in forward AND reverse mode through
  the implicit-function rule (one extra transposed solve per gradient, no
  time history). Accepts one or many right-hand sides per factorisation.
* :mod:`rfx.fdfd.hplane` -- 2-D H-plane TE_n0 Helmholtz solver with exact
  discrete transparent ports, matching the independent referee node-for-node
  and differentiable in frequency, in a per-node permittivity map and in
  the iris aperture widths (body-fitted grid: edges stay on nodes, the
  nodes move with the width, so the shape derivative is smooth).
* :mod:`rfx.fdfd.yee3d` -- 3-D vector Yee FDFD (curl-curl, complex eps per
  cell, PEC masks, PML by complex stretching, nonuniform traced grid steps)
  with a TE10 waveguide port for validation against the 2-D solver.
* :mod:`rfx.fdfd.ports3d` -- lumped ports, N-port S/Z matrices from one
  factorisation, renormalisation.
* :mod:`rfx.fdfd.conductor` -- Leontovich surface-impedance conductors on
  cell masks or on the outer walls, differentiable in conductivity.
* :mod:`rfx.fdfd.deembed` -- jnp network conversions, the de-embedding
  functions of :mod:`rfx.deembed`, open-short de-embedding and inductor
  L/Q metrics, all differentiable.
* :mod:`rfx.fdfd.gds` -- GDS + layer-stack import, parametric spirals,
  edge-snapped tensor-grid planning and area-exact rasterisation (host
  numpy; the grid lines it returns are what the solver differentiates).
"""
from rfx.fdfd import conductor, deembed, gds, hplane, linear_solve, ports3d, yee3d
from rfx.fdfd.linear_solve import sparse_solve, sparse_matvec, clear_factor_cache
from rfx.fdfd.hplane import (
    HPlaneSpec, HPlaneModel, build, assemble, solve, s_params,
    richardson_first_order,
)
from rfx.fdfd.yee3d import Yee3DSpec, Yee3DModel, BoundaryTerms
from rfx.fdfd.ports3d import LumpedElement, s_matrix, z_matrix
from rfx.fdfd.conductor import Conductor
from rfx.fdfd.gds import LayerStack, rect_spiral, octagonal_spiral, mesh_lines, rasterise

__all__ = [
    "conductor", "deembed", "gds", "hplane", "linear_solve", "ports3d", "yee3d",
    "sparse_solve", "sparse_matvec", "clear_factor_cache",
    "HPlaneSpec", "HPlaneModel", "build", "assemble", "solve", "s_params",
    "richardson_first_order",
    "Yee3DSpec", "Yee3DModel", "BoundaryTerms",
    "LumpedElement", "s_matrix", "z_matrix",
    "Conductor",
    "LayerStack", "rect_spiral", "octagonal_spiral", "mesh_lines", "rasterise",
]
