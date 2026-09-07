"""Lumped ports, lumped loads and the N-port S-matrix for the 3-D Yee FDFD.

A lumped element is a box of E edges of ONE component ``c``: along ``c`` the
edges span the gap between two conductors (``n_gap`` edges in series), on
the two other axes the box selects ``n_col`` parallel columns. Let ``l_e``
be the primal length of edge ``e``, ``A_e`` its dual-face area
(``dual_a * dual_b`` at its node indices), ``L = sum of l_e along one
column`` the gap length and ``A = sum of A_e over the columns`` the port
cross-section.

Internal impedance / lumped load. The dual-face Ampere loop of edge ``e``
reads ``oint H dl = j omega eps E_e A_e + I_e``; an element of impedance
``Z`` across the gap carries ``I = (sum_e E_e l_e) / Z`` along ``c``
(series division of ``Z`` along the gap, parallel division across the
columns in proportion to ``A_e``), so the per-edge current density is
``J_e = E_e L / (Z A)``. Moved to the left-hand side of
``curl curl E - k0^2 eps E = -j omega mu0 J`` this is the diagonal term

    A[e, e] += j omega mu0 * L / (Z A)        (an equivalent conductivity
                                              sigma_eff = L / (Z A) = l_e / (Z_e A_e)),

the familiar "resistor across a Yee edge" formula ``R = l / (sigma A)``
with the series / parallel bookkeeping for multi-edge elements. A port is
this element with ``Z = Z0`` (its internal impedance) plus the excitation.

Excitation. An impressed current ``I_src`` through the port: ``J_e =
I_src / A`` on every port edge (Norton source in parallel with ``Z0``).

Port voltage and current. ``V = -sum_e E_e l_e A_e / A`` (the voltage of the
``+c`` conductor relative to the ``-c`` one, averaged over the columns
with their dual areas; for one column exactly ``-sum E dl``). ``I`` is the
discrete Ampere loop of ``H`` around the port: ``I = sum_e (l_e / L) A_e
(Ch H)_e``, the total current crossing the port cross-section (averaged
along the gap), i.e. the current entering the ``+c`` conductor = the
current INTO the network. It contains the source, the shunt ``Z0`` and the
displacement current of the gap, exactly as KCL at the terminal demands.

Waves. ``a = (V + Z0 I) / (2 sqrt Z0)``, ``b = (V - Z0 I) / (2 sqrt Z0)``,
so ``|a|^2 - |b|^2 = Re(V I*)`` is the power into the network. For a
single-edge port ``Re(V I*) = -l A Re(E J_src*) - sigma_eff l A |E|^2``
is EXACTLY the source power minus the shunt power of the discrete system
(the ``j omega eps`` term drops out), so with the discrete Poynting
identity of the Yee grid the S-matrix of a lossless structure is unitary
to LU roundoff and that of a lossy one is strictly passive. For
multi-edge ports ``V`` and ``I`` are averages and the identity holds up to
the non-uniformity of ``E`` and ``curl H`` over the element.

S-matrix. Port ``p`` is excited with ``I_src = 1`` while every other port
is only loaded by its ``Z0``; all excitations are one block right-hand
side of one LU factorisation (:func:`rfx.fdfd.linear_solve.sparse_solve`).
With ``A[q, p] = a_q`` and ``B[q, p] = b_q`` under excitation ``p``,
``S = B A^{-1}`` (the generalised definition, exact even when the unexcited
ports' ``a_q`` are not zero). Everything is ``jax.numpy`` complex128 on the
differentiable path: impedances, steps, permittivity and the traced parts
of the boundary terms can all carry gradients.

Scope fence. Ports must lie outside any PML (their geometry uses the real
steps); the reference plane is the port itself (no de-embedding); the
port discontinuity (the gap capacitance, the lumped feed edge's own
inductance) is part of the network, which is why a matched TEM section
between two lumped ports does not give exactly ``|S21| = 1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rfx.fdfd import yee3d as y

__all__ = ["LumpedElement", "element_edges", "element_geometry", "lumped_terms",
           "current_source", "port_voltage", "port_current", "s_matrix", "z_matrix",
           "renormalize", "PortSolution"]


@dataclass(frozen=True)
class LumpedElement:
    """Edge box of component ``axis``: edge-array indices ``lo <= (i, j, k) < hi``.
    Along ``axis`` the range is the gap (series), on the other two axes the
    parallel columns. A single edge is ``hi = lo + (1, 1, 1)``."""
    axis: int
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    @staticmethod
    def single(axis: int, i: int, j: int, k: int) -> "LumpedElement":
        return LumpedElement(axis, (i, j, k), (i + 1, j + 1, k + 1))


def element_edges(model: y.Yee3DModel, el: LumpedElement) -> np.ndarray:
    """Flat edge ids of the element (static), in C order of the box."""
    if el.axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    shp = y.edge_shapes(*model.shape)[el.axis]
    for d in range(3):
        if not (0 <= el.lo[d] < el.hi[d] <= shp[d]):
            raise ValueError(f"element box {el.lo}..{el.hi} outside the edge array {shp}")
    idx = np.arange(int(np.prod(shp))).reshape(shp) + model.edge_offsets[el.axis]
    box = idx[el.lo[0]:el.hi[0], el.lo[1]:el.hi[1], el.lo[2]:el.hi[2]]
    return box.ravel()


def _edge_ijk(model: y.Yee3DModel, el: LumpedElement):
    shp = y.edge_shapes(*model.shape)[el.axis]
    ijk = np.indices(shp)[:, el.lo[0]:el.hi[0], el.lo[1]:el.hi[1], el.lo[2]:el.hi[2]]
    return ijk.reshape(3, -1)


def element_geometry(model: y.Yee3DModel, el: LumpedElement, dx, dy, dz):
    """``(l_e, A_e, L, A)`` of the element: per-edge primal length and dual
    area (real, unstretched steps), the gap length and the cross-section."""
    steps = [jnp.asarray(d, dtype=jnp.float64) for d in (dx, dy, dz)]
    duals = [jnp.concatenate([0.5 * d[:1], 0.5 * (d[:-1] + d[1:]), 0.5 * d[-1:]]) for d in steps]
    c = el.axis
    a, b = (c + 1) % 3, (c + 2) % 3
    ijk = _edge_ijk(model, el)
    length = steps[c][ijk[c]]
    area = duals[a][ijk[a]] * duals[b][ijk[b]]
    n_gap = el.hi[c] - el.lo[c]
    n_col = len(length) // n_gap
    gap_len = jnp.sum(length) / n_col
    cross = jnp.sum(area) / n_gap
    return length, area, gap_len, cross


def lumped_terms(model: y.Yee3DModel, freq, elements: Sequence[LumpedElement],
                 impedances, dx, dy, dz) -> y.BoundaryTerms:
    """Diagonal admittance ``j omega mu0 L / (Z A)`` of each element (traced
    impedances) as :class:`rfx.fdfd.yee3d.BoundaryTerms`."""
    omega = 2.0 * jnp.pi * jnp.asarray(freq, dtype=jnp.float64)
    diag = jnp.zeros(model.n_edges, jnp.complex128)
    for el, z in zip(elements, impedances):
        ids = element_edges(model, el)
        _, _, gap_len, cross = element_geometry(model, el, dx, dy, dz)
        z = jnp.asarray(z, dtype=jnp.complex128)
        diag = diag.at[ids].add(1j * omega * y.MU0 * gap_len / (z * cross))
    return y.BoundaryTerms(diag_add=diag)


def current_source(model: y.Yee3DModel, el: LumpedElement, dx, dy, dz, amplitude=1.0):
    """(Jx, Jy, Jz) of an impressed current ``amplitude`` (A) through the
    element along ``+axis``: ``J = amplitude / A`` on its edges."""
    _, _, _, cross = element_geometry(model, el, dx, dy, dz)
    ids = element_edges(model, el)
    j = jnp.zeros(model.n_edges, jnp.complex128).at[ids].set(
        jnp.asarray(amplitude, jnp.complex128) / cross)
    return y.split_edges(model, j)


def port_voltage(model: y.Yee3DModel, el: LumpedElement, e_flat, dx, dy, dz):
    """``V = -sum_e E_e l_e A_e / A`` (column-averaged ``-int E dl``);
    ``e_flat`` is ``(n_edges,)`` or ``(n_edges, m)``."""
    length, area, _, cross = element_geometry(model, el, dx, dy, dz)
    ids = element_edges(model, el)
    w = (length * area / cross).astype(jnp.complex128)
    return -jnp.tensordot(w, e_flat[ids], axes=(0, 0))


def port_current(model: y.Yee3DModel, el: LumpedElement, curl_h_flat, dx, dy, dz):
    """``I = sum_e (l_e / L) A_e (Ch H)_e``: the Ampere loop of ``H`` around
    the port cross-section, averaged along the gap. ``curl_h_flat`` is the
    output of :func:`rfx.fdfd.yee3d.curl_h`."""
    length, area, gap_len, _ = element_geometry(model, el, dx, dy, dz)
    ids = element_edges(model, el)
    w = (length * area / gap_len).astype(jnp.complex128)
    return jnp.tensordot(w, curl_h_flat[ids], axes=(0, 0))


@dataclass(frozen=True)
class PortSolution:
    """Fields and port quantities of an N-port solve: ``e`` = (Ex, Ey, Ez)
    with a trailing excitation axis, ``v`` / ``i`` = ``(n_ports, n_ports)``
    voltage / current at port q under excitation p, ``a`` / ``b`` the wave
    matrices, ``s`` the S-matrix."""
    e: tuple
    v: jax.Array
    i: jax.Array
    a: jax.Array
    b: jax.Array
    s: jax.Array


def s_matrix(model: y.Yee3DModel, freq, eps_r, dx, dy, dz, ports: Sequence[LumpedElement],
             z0, pec=None, loads: Sequence[LumpedElement] = (), load_impedances=(),
             terms: y.BoundaryTerms | None = None, return_solution: bool = False):
    """N-port S-matrix with lumped ports of internal impedances ``z0``
    (scalar or one per port, traced), optional lumped ``loads`` with
    ``load_impedances`` (traced) and optional extra boundary ``terms``
    (e.g. a surface-impedance conductor). One LU factorisation for all
    ports. Returns ``S`` ``(n, n)`` or a :class:`PortSolution`."""
    n = len(ports)
    if n == 0:
        raise ValueError("need at least one port")
    z0 = jnp.broadcast_to(jnp.asarray(z0, dtype=jnp.complex128), (n,))
    if len(loads) != len(load_impedances):
        raise ValueError("loads and load_impedances must have the same length")
    elements = list(ports) + list(loads)
    impedances = [z0[p] for p in range(n)] + [jnp.asarray(z, jnp.complex128) for z in load_impedances]
    all_terms = y.merge_terms(terms, lumped_terms(model, freq, elements, impedances, dx, dy, dz))
    sources = [current_source(model, p, dx, dy, dz) for p in ports]
    e = y.solve(model, freq, eps_r, dx, dy, dz, sources, pec, all_terms)
    e_flat = jnp.concatenate([jnp.asarray(c).reshape(-1, n) for c in e])
    h = y.h_from_e(model, freq, e, dx, dy, dz)
    ch = y.curl_h(model, freq, h, dx, dy, dz, all_terms)
    v = jnp.stack([port_voltage(model, p, e_flat, dx, dy, dz) for p in ports])   # (q, p)
    i = jnp.stack([port_current(model, p, ch, dx, dy, dz) for p in ports])
    root = jnp.sqrt(z0)[:, None]
    a = (v + z0[:, None] * i) / (2.0 * root)
    b = (v - z0[:, None] * i) / (2.0 * root)
    s = jnp.linalg.solve(a.T, b.T).T
    if return_solution:
        return PortSolution(e=e, v=v, i=i, a=a, b=b, s=s)
    return s


def z_matrix(sol: PortSolution) -> jax.Array:
    """Impedance matrix ``Z = V I^{-1}`` of the network beyond the ports; it
    does not depend on the ports' internal impedances (a consistency check
    of the readout)."""
    return jnp.linalg.solve(sol.i.T, sol.v.T).T


def renormalize(z: jax.Array, z0) -> jax.Array:
    """S-matrix of the network with impedance matrix ``z`` referenced to the
    (real, per-port) impedances ``z0``: ``S = F (Z - Z0)(Z + Z0)^{-1} F^{-1}``
    with ``F = diag(1 / sqrt z0)``."""
    n = z.shape[0]
    z0 = jnp.broadcast_to(jnp.asarray(z0, dtype=jnp.complex128), (n,))
    f = jnp.diag(1.0 / jnp.sqrt(z0))
    z0m = jnp.diag(z0)
    core = jnp.linalg.solve((z + z0m).T, (z - z0m).T).T
    return f @ core @ jnp.linalg.inv(f)
