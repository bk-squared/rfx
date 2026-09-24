"""The in-loop block current moments, against what the lattice already says.

A current element pulsed once in empty space is the simplest structure whose
current is known in closed form: on every edge but the one the source writes
to, the electric field obeys the vacuum Ampere update exactly, so the current
``curl_h H - eps0 dE/dt`` there is zero by construction of the update; at the
source edge it is the impressed current and nothing else. Both statements are
checked here, on two meshes and through both runners.

The same monitor is then checked against a second route written in this file:
the same current read after the run out of recorded frequency-domain planes
(DFT plane probes), ``curl(H_dft) - j w~ eps0 E_dft``. The two routes share
the fields and nothing else — one reduces the current inside the time loop,
the other builds it per edge from the planes and sums it into blocks here —
and they are exactly equal up to where each stamps its time and the field
left standing when the record stops. That equality holds in a dielectric, at
a lossy edge and at a port's feed as well as in vacuum, so it is checked on a
loaded board too.

The far field is checked end to end on a strip dipole against the repository's
own Huygens-box (NTFF) transform on the same run, and its gradient against a
central difference in float64.

Two meshes, because one of them cannot see half the defects. On a uniform
mesh the dual spacing IS the primal cell, bit for bit, so no check anywhere
can tell a dual-for-primal swap from the right metric; and the vacuum
identity multiplies a zero current by the edge volume, so it cannot see a
wrong volume at all. The graded-z fixture below has primal/dual ratios from
0.870 to 1.130 across its slab, and both defects are loud there.

Vocabulary: a ``mutation`` is a defect put back on purpose with every helper
call left in place; the ``vacuum residual`` is the largest block moment
outside the source's block, over the source block's.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from rfx.core.yee import EPS_0, init_materials
from rfx.grid import Grid
from rfx.simulation import ProbeSpec, SourceSpec, run
from rfx.current_moments import (
    NONUNIFORM_PLANE_STAMP_STEPS,
    UNIFORM_PLANE_STAMP_STEPS,
    accumulate_current_moments,
    block_far_field_jax,
    block_far_field_np,
    build_current_moment_monitor,
    current_moment_far_field,
    current_moment_monitor_from_grid,
    e_dual_spacings,
    end_of_record_moments,
    init_current_moment_data,
    moments_to_PQT,
    plane_stamp_steps,
    slab_e_snapshot,
    to_post_processing_convention,
)
from rfx.probes.probes import init_dft_plane_probe
from tests._x64_compat import enable_x64

REPO_ROOT = Path(__file__).resolve().parents[3]
C0 = 299_792_458.0
ETA0 = float(np.sqrt(1.25663706212e-6 / EPS_0))


# ---------------------------------------------------------------------------
# The second route, written here: the same current from recorded DFT planes
# ---------------------------------------------------------------------------

def _midpoint_span(cells, k):
    """Distance between the centres of the two cells node ``k`` sits between.

    On a graded axis this is what the E update divides a curl by, and it is
    the side of the dual face the current crosses. Written from the cell
    widths alone, so nothing here shares a metric helper with the monitor.
    """
    return float(cells[0]) if k == 0 else 0.5 * (float(cells[k - 1])
                                                 + float(cells[k]))


def _h_planes_needed(e_planes, ez_planes):
    """The H planes the E curl over the slab reads.

    The Ex/Ey curl at node plane k differences Hx/Hy at k and k-1 and reads
    Hz at k; the Ez curl at half plane k reads Hx and Hy at k only.
    """
    hxy = set(e_planes) | {k - 1 for k in e_planes} | set(ez_planes)
    return {"hx": sorted(hxy), "hy": sorted(hxy), "hz": sorted(set(e_planes))}


def _plane_specs(e_planes, ez_planes):
    need = _h_planes_needed(e_planes, ez_planes)
    return ([("ex", k) for k in e_planes] + [("ey", k) for k in e_planes]
            + [("ez", k) for k in ez_planes]
            + [("hx", k) for k in need["hx"]] + [("hy", k) for k in need["hy"]]
            + [("hz", k) for k in need["hz"]])


def _plane_route_edges(planes, end_state, *, nodes, cells, window, freqs, dt,
                       n_steps, stamp):
    """Every E edge of the slab with its current moment, from the planes.

    Frequency-domain Ampere on the lattice, per edge::

        J_post = curl(H_dft) - j w~ eps0 E_dft exp(-j w dt/2),
        j w~   = (2j/dt) sin(w dt/2)

    The in-loop monitor sums ``J^{n+1/2} = curl H^{n+1/2}
    - eps0 (E^{n+1} - E^n)/dt`` with the phase ``exp(-j w (n+1/2) dt)``.
    Summing that difference of electric fields by parts over a record of N
    steps that starts from rest leaves ``2j sin(w dt/2)`` times the DFT of
    E plus the field standing at the end, ``E^N exp(-j w (N+1/2) dt)``; a
    plane probe that stamps the sample of step n at ``(n + s) dt`` makes
    ``curl(H_dft)`` carry ``exp(-j w (s - 1/2) dt)`` against the half-step
    stamp. So, exactly::

        A = exp(+j w dt (s - 1/2)) J_post - eps0 E^N exp(-j w (N + 1/2) dt)

    which is what this returns as ``mom``, times each edge's control volume
    (its own primal cell times the midpoint span on each of the other two
    axes). ``post`` is the same before the end term is taken off.
    """
    cx, cy, cz = (np.asarray(c, dtype=np.float64) for c in cells)
    nx_, ny_, nz_ = (np.asarray(n, dtype=np.float64) for n in nodes)
    (i0, i1), (j0, j1), (k0, k1) = window
    ii, jj = np.arange(i0, i1), np.arange(j0, j1)
    II, JJ = np.meshgrid(ii, jj, indexing="ij")
    sx = np.array([_midpoint_span(cx, i) for i in ii])[:, None]
    sy = np.array([_midpoint_span(cy, j) for j in jj])[None, :]
    freqs = np.asarray(freqs, dtype=np.float64)
    w = 2.0 * np.pi * freqs[:, None, None]
    jw = (2j / dt) * np.sin(w * dt / 2.0)
    half = np.exp(-1j * w * dt / 2.0)
    to_loop = np.exp(1j * w * dt * (float(stamp) - 0.5))
    end_phase = np.exp(-1j * w * (float(n_steps) + 0.5) * dt)

    def P(comp, k, di=0, dj=0):
        a = np.asarray(planes[(comp, int(k))], dtype=np.complex128)
        return a[:, i0 - di:i1 - di, j0 - dj:j1 - dj]

    def end(comp, k):
        a = np.asarray(getattr(end_state, comp), dtype=np.float64)
        return a[i0:i1, j0:j1, int(k)][None]

    out = {"pos": [], "comp": [], "idx": [], "vol": [], "mom": [],
           "post": []}

    def add(comp, k, curl, e_name, pos, vol):
        post = to_loop * (curl - jw * EPS_0 * P(e_name, k) * half)
        a = post - EPS_0 * end(e_name, k) * end_phase
        n = II.size
        out["pos"].append(np.stack(pos, axis=-1).reshape(n, 3))
        out["comp"].append(np.full(n, comp))
        out["idx"].append(np.stack([II.ravel(), JJ.ravel(),
                                    np.full(n, int(k))], axis=-1))
        out["vol"].append(vol.reshape(n))
        out["mom"].append((a * vol[None]).reshape(freqs.size, n))
        out["post"].append((post * vol[None]).reshape(freqs.size, n))

    ones = np.ones((ii.size, jj.size))
    for k in range(k0, k1 + 1):
        sz = _midpoint_span(cz, k)
        # ex(i+1/2, j, k): dHz/dy - dHy/dz
        curl = ((P("hz", k) - P("hz", k, dj=1)) / sy
                - (P("hy", k) - P("hy", k - 1)) / sz)
        add(0, k, curl, "ex",
            (nx_[II] + 0.5 * cx[II], ny_[JJ], nz_[k] * ones),
            cx[ii][:, None] * sy * sz * ones)
        # ey(i, j+1/2, k): dHx/dz - dHz/dx
        curl = ((P("hx", k) - P("hx", k - 1)) / sz
                - (P("hz", k) - P("hz", k, di=1)) / sx)
        add(1, k, curl, "ey",
            (nx_[II], ny_[JJ] + 0.5 * cy[JJ], nz_[k] * ones),
            sx * cy[jj][None, :] * sz * ones)
    for k in range(k0, k1):
        # ez(i, j, k+1/2): dHy/dx - dHx/dy
        curl = ((P("hy", k) - P("hy", k, di=1)) / sx
                - (P("hx", k) - P("hx", k, dj=1)) / sy)
        add(2, k, curl, "ez",
            (nx_[II], ny_[JJ], (nz_[k] + 0.5 * cz[k]) * ones),
            sx * sy * cz[k] * ones)
    return {key: np.concatenate(v, axis=-1 if key in ("mom", "post") else 0)
            for key, v in out.items()}


def _plane_route_blocks(edges, *, i_lo, j_lo, block_cells, off_cells=0):
    """Blocks of the edge table and their P, Q, T, summed edge by edge.

    A block is every edge whose in-plane index falls in one square of
    ``block_cells`` cells, through the whole slab thickness. Its centre is
    the plain mean of the edge positions in it. ``P = sum p``,
    ``Q[a, c] = sum (r - c)_a p_c``, ``T[a, b, c] = sum (r - c)_a (r - c)_b
    p_c`` with ``p`` pointing along the edge's own axis.
    """
    d, off = int(block_cells), int(off_cells)
    key = np.stack([(edges["idx"][:, 0] - i_lo + off) // d,
                    (edges["idx"][:, 1] - j_lo + off) // d], axis=1)
    uniq, gid = np.unique(key, axis=0, return_inverse=True)
    gid = np.asarray(gid).ravel()
    ng = uniq.shape[0]
    counts = np.bincount(gid, minlength=ng)
    centres = np.stack([np.bincount(gid, weights=edges["pos"][:, a],
                                    minlength=ng) for a in range(3)],
                       axis=1) / counts[:, None]
    delta = edges["pos"] - centres[gid]
    comp = edges["comp"]

    def moments(mom):
        nf = mom.shape[0]
        P = np.zeros((nf, ng, 3), dtype=np.complex128)
        Q = np.zeros((nf, ng, 3, 3), dtype=np.complex128)
        T = np.zeros((nf, ng, 3, 3, 3), dtype=np.complex128)
        for c in range(3):
            sel = comp == c
            g, m, dl = gid[sel], mom[:, sel], delta[sel]
            for f in range(nf):
                np.add.at(P[f, :, c], g, m[f])
                for a in range(3):
                    np.add.at(Q[f, :, a, c], g, dl[:, a] * m[f])
                    for b in range(3):
                        np.add.at(T[f, :, a, b, c], g,
                                  dl[:, a] * dl[:, b] * m[f])
        return P, Q, T

    P, Q, T = moments(edges["mom"])
    scale = moments(edges["post"])
    return {"key": uniq, "centres": centres, "P": P, "Q": Q, "T": T,
            "scale": dict(zip("PQT", scale)), "gid": gid}


def _rel_l2(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _disagreement(acc, monitor, blocks, orders=("P", "Q", "T")):
    """Worst difference, over frequency and order, monitor vs plane route.

    Measured against the moments before the end term is taken off. With a
    pulse that leaves a static charge behind, the field standing at the end
    is up to 22 times the in-loop moment at 10 GHz on these fixtures, and the
    plane route builds the in-loop moment as the difference of the two, so
    its float32 round-off is set by the larger of them.
    """
    got = dict(zip(("P", "Q", "T"), moments_to_PQT(acc, monitor)))
    return max(float(np.linalg.norm(got[name][f] - blocks[name][f])
                     / np.linalg.norm(blocks["scale"][name][f]))
               for name in orders for f in range(blocks["P"].shape[0]))


# ---------------------------------------------------------------------------
# Two fixtures: one uniform, one graded in z
# ---------------------------------------------------------------------------

DX = 2.0e-3
N_STEPS = 260
FREQS = np.array([5.0e9, 7.5e9, 1.0e10])
SLAB_HALF = 4          # in-plane half width of the slab, in cells
BLOCK_CELLS = 3

# The vacuum residual in float32 on THESE slabs, measured on the unmutated
# runs (CPU, JAX 0.6.2): 1.7e-06 on the uniform fixture and 1.7e-08 on the
# graded one; the floors below are those rounded up. It is float32 round-off
# on the cancellation between ``curl_h H`` and ``eps0 dE/dt``, scaled by the
# local field, so it belongs to the fixture and to where the slab sits. The
# bar is ten times the floor measured here and is not a bar any other
# fixture inherits.
VACUUM_FLOOR = {"uniform": 1.8e-06, "graded": 5.0e-08}
VACUUM_BAR = {kind: 10.0 * v for kind, v in VACUUM_FLOOR.items()}

# In-loop moments against the plane route: the two are the same numbers up
# to float32 arithmetic. What sets the floor is the phase: both sides form
# exp(-j w t) from a float32 time, whose last bit is worth eps32 * w * t, and
# t runs to the record length. That is 3.8e-06 at 10 GHz on the uniform
# fixture (a 1.0 ns record) and 8e-07 on the graded one (0.23 ns); measured
# (CPU, JAX 0.6.2): 5.3e-06 and 2.2e-06 worst over P, Q, T and frequency,
# loaded or not. The bar is ten times the worst; the smallest monitor defect
# in the mutation table below reads 3.2e-02. A one-step stamp error is
# |exp(-j w dt) - 1| = 3.6e-02 of the in-loop moment here, at least 1.6e-03
# on this scale, and a sign or metric defect is order one.
EQUALITY_BAR = 5e-5


def _gaussian(n_steps, dt):
    t = np.arange(n_steps) * dt
    t0, tau = 40.0 * dt, 12.0 * dt
    return np.exp(-((t - t0) ** 2) / (2.0 * tau ** 2)).astype(np.float32)


def _loaded(materials, ci, cj, k_nodes):
    """A lossy dielectric block beside the source, through the slab.

    Its faces are where the edge-averaged material rule (the mean of the four
    cells incident to an edge) takes effect, and its conductivity puts a loss
    current on every edge inside it: the identity must hold there too. It
    runs through the whole slab thickness, so on the graded fixture its Ex
    and Ey edges sit where the z primal and dual widths differ.
    """
    eps = np.asarray(materials.eps_r).copy()
    sig = np.asarray(materials.sigma).copy()
    box = (slice(ci + 1, ci + 4), slice(cj - 2, cj + 2),
           slice(k_nodes[0], k_nodes[1] + 1))
    eps[box] = 4.0
    sig[box] = 0.05
    return materials._replace(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sig))


def _uniform_case(monitor_kwargs=None, n_steps=N_STEPS, with_planes=True,
                  block_cells=BLOCK_CELLS, probes=(), runner=None,
                  loaded=False):
    """Uniform ``Grid`` + CPML, a soft Ez element at the centre."""
    grid = Grid(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 2.4e-2), dx=DX,
                cpml_layers=6)
    materials = init_materials(grid.shape)
    nx, ny, nz = grid.shape
    ci, cj, ck = nx // 2, ny // 2, nz // 2
    wave = jnp.asarray(_gaussian(n_steps, grid.dt))
    source = SourceSpec(i=ci, j=cj, k=ck, component="ez", waveform=wave)

    i_range = (ci - SLAB_HALF, ci + SLAB_HALF + 1)
    j_range = (cj - SLAB_HALF, cj + SLAB_HALF + 1)
    k_nodes = (ck - 1, ck + 2)
    if loaded:
        materials = _loaded(materials, ci, cj, k_nodes)
    nodes = [(np.arange(n) - pad) * DX for n, pad in
             ((nx, grid.pad_x_lo), (ny, grid.pad_y_lo), (nz, grid.pad_z_lo))]
    cells = [np.full(n, DX) for n in (nx, ny, nz)]

    monitor = build_current_moment_monitor(
        node_x=nodes[0], node_y=nodes[1], node_z=nodes[2],
        cell_x=cells[0], cell_y=cells[1], cell_z=cells[2],
        i_range=i_range, j_range=j_range, k_node_range=k_nodes,
        block_cells=block_cells, freqs=FREQS,
        pads=(grid.pad_x_lo, grid.pad_x_hi, grid.pad_y_lo, grid.pad_y_hi,
              grid.pad_z_lo, grid.pad_z_hi),
        **(monitor_kwargs or {}))

    specs = (_plane_specs(range(k_nodes[0], k_nodes[1] + 1),
                          range(k_nodes[0], k_nodes[1]))
             if with_planes else [])
    planes = [init_dft_plane_probe(
        axis=2, index=k, component=comp,
        freqs=jnp.asarray(FREQS, dtype=jnp.float32), grid_shape=grid.shape)
        for comp, k in specs]

    runner = runner or run
    res = runner(grid, materials, n_steps, boundary="cpml", sources=[source],
                 probes=list(probes), dft_planes=planes,
                 current_moments=monitor)
    return dict(kind="uniform", grid=grid, materials=materials, source=source,
                monitor=monitor, specs=specs, i_range=i_range,
                j_range=j_range, k_nodes=k_nodes, nodes=nodes, cells=cells,
                src_idx=(ci, cj, ck), n_steps=n_steps, wave=np.asarray(wave),
                result=res, acc=np.asarray(res.current_moment_data[0]),
                state=res.state, time_series=np.asarray(res.time_series),
                dft_planes=res.dft_planes or (),
                stamp=UNIFORM_PLANE_STAMP_STEPS, block_cells=block_cells)


GRADED_N_STEPS = 320
GRADED_SLAB_K = (11, 24)          # straddles both grading ramps
GRADED_SRC_K = 17                 # inside the fine band


def _graded_grid():
    """A z mesh graded by the repo's own smoothing, 2 mm down to 0.381 mm."""
    from rfx.auto_config import smooth_grading
    from rfx.nonuniform import make_nonuniform_grid
    raw = np.concatenate([np.full(3, 2e-3), np.full(4, 0.381e-3),
                          np.full(3, 2e-3)])
    dz = smooth_grading(raw, max_ratio=1.3)
    return make_nonuniform_grid((2.4e-2, 2.4e-2), dz, DX, cpml_layers=6)


def _graded_case(monitor_kwargs=None, n_steps=GRADED_N_STEPS,
                 with_planes=True, block_cells=BLOCK_CELLS, runner=None,
                 loaded=False):
    """The same element on a z-graded mesh, through the graded-mesh runner."""
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    from rfx.nonuniform import run_nonuniform

    grid = _graded_grid()
    coords = coords_from_nonuniform_grid(grid)
    nodes = [np.asarray(coords.x, float), np.asarray(coords.y, float),
             np.asarray(coords.z, float)]
    cells = [np.asarray(grid.dx_arr, float), np.asarray(grid.dy_arr, float),
             np.asarray(grid.dz, float)]
    nx, ny, _nz = grid.shape
    ci, cj = nx // 2, ny // 2
    ck = GRADED_SRC_K
    materials = init_materials(grid.shape)
    if loaded:
        materials = _loaded(materials, ci, cj, GRADED_SLAB_K)
    wave = jnp.asarray(_gaussian(n_steps, grid.dt))

    i_range = (ci - SLAB_HALF, ci + SLAB_HALF + 1)
    j_range = (cj - SLAB_HALF, cj + SLAB_HALF + 1)
    k_nodes = GRADED_SLAB_K
    monitor = build_current_moment_monitor(
        node_x=nodes[0], node_y=nodes[1], node_z=nodes[2],
        cell_x=cells[0], cell_y=cells[1], cell_z=cells[2],
        i_range=i_range, j_range=j_range, k_node_range=k_nodes,
        block_cells=block_cells, freqs=FREQS,
        pads=(grid.pad_x_lo, grid.pad_x_hi, grid.pad_y_lo, grid.pad_y_hi,
              grid.pad_z_lo, grid.pad_z_hi),
        **(monitor_kwargs or {}))

    specs = (_plane_specs(range(k_nodes[0], k_nodes[1] + 1),
                          range(k_nodes[0], k_nodes[1]))
             if with_planes else [])
    planes = [init_dft_plane_probe(
        axis=2, index=k, component=comp,
        freqs=jnp.asarray(FREQS, dtype=jnp.float32), grid_shape=grid.shape)
        for comp, k in specs]

    runner = runner or run_nonuniform
    res = runner(grid, materials, n_steps,
                 sources=[(ci, cj, ck, "ez", wave)],
                 dft_planes=planes or None,
                 current_moments=monitor)
    return dict(kind="graded", grid=grid, materials=materials, monitor=monitor,
                specs=specs, i_range=i_range, j_range=j_range,
                k_nodes=k_nodes, nodes=nodes, cells=cells,
                src_idx=(ci, cj, ck), n_steps=n_steps, wave=np.asarray(wave),
                result=res, acc=np.asarray(res["current_moment_data"][0]),
                state=res["state"],
                time_series=np.asarray(res["time_series"]),
                dft_planes=res.get("dft_planes") or (),
                stamp=NONUNIFORM_PLANE_STAMP_STEPS, block_cells=block_cells)


CASE_BUILDERS = {"uniform": _uniform_case, "graded": _graded_case}


@functools.lru_cache(maxsize=None)
def _reference(kind, loaded=False):
    """The unmutated run with every plane recorded (shared, read-only)."""
    return CASE_BUILDERS[kind](loaded=loaded)


def _route_blocks(case, stamp=None, off_cells=0):
    """The plane route's blocks for a case that recorded its planes."""
    planes = {spec: np.asarray(p.accumulator)
              for spec, p in zip(case["specs"], case["dft_planes"])}
    edges = _plane_route_edges(
        planes, case["state"], nodes=case["nodes"], cells=case["cells"],
        window=(case["i_range"], case["j_range"], case["k_nodes"]),
        freqs=FREQS, dt=float(case["grid"].dt), n_steps=case["n_steps"],
        stamp=case["stamp"] if stamp is None else stamp)
    return edges, _plane_route_blocks(
        edges, i_lo=case["i_range"][0], j_lo=case["j_range"][0],
        block_cells=case["block_cells"], off_cells=off_cells)


@functools.lru_cache(maxsize=None)
def _reference_blocks(kind, loaded=False):
    return _route_blocks(_reference(kind, loaded))


# ---------------------------------------------------------------------------
# Shared readings
# ---------------------------------------------------------------------------

def _source_block(case):
    m = case["monitor"]
    ci, cj, _ck = case["src_idx"]
    seg = np.asarray(m.seg).reshape(m.i_hi - m.i_lo, m.j_hi - m.j_lo)
    return int(seg[ci - m.i_lo, cj - m.j_lo])


def _vacuum_residual(case, acc=None):
    """Largest block moment outside the source's, over the source block's."""
    acc = case["acc"] if acc is None else acc
    g = _source_block(case)
    P = acc[..., 0]
    return float((np.delete(np.abs(P), g, axis=1).max(axis=(1, 2))
                  / np.abs(P[:, g, :]).max(axis=1)).max())


def _analytic_source_moment(case):
    """``P`` the soft source alone puts into its block, in closed form.

    Both low-level runners inject a precomputed waveform straight into the
    field: after the vacuum E update ``E_lin = E^n + (dt/eps0) curl_h H`` the
    step writes ``E^{n+1} = E_lin + w_n``. Put that into the lattice current,

        J = curl_h H - eps0 (E^{n+1} - E^n)/dt
          = curl_h H - eps0 (dt/eps0 curl_h H + w_n)/dt
          = -eps0 w_n / dt,

    so the edge carries no curl term at all — the impressed current is the
    injected field increment and nothing else. Its moment is that times the
    edge's control volume, and the accumulator holds the DFT of it at the
    half-step stamp:

        P = -eps0 V sum_n w_n exp(-j w (n + 1/2) dt).

    ``V`` is built from the fixture's own cell arrays, never read out of the
    monitor under test.
    """
    ci, cj, ck = case["src_idx"]
    cx, cy, cz = (np.asarray(c, dtype=np.float64) for c in case["cells"])
    vol = _midpoint_span(cx, ci) * _midpoint_span(cy, cj) * cz[ck]
    dt = float(case["grid"].dt)
    n = np.arange(case["n_steps"])
    phase = np.exp(-2j * np.pi * FREQS[:, None] * (n[None, :] + 0.5) * dt)
    return -EPS_0 * vol * (phase * case["wave"][None, :]).sum(axis=1), vol


def _vacuum_against_the_closed_form(case, acc=None):
    """Largest non-source block moment over the source's ANALYTIC moment.

    The vacuum residual divides by the source block as the run measured it,
    which collapses when a defect empties the source block. This one divides
    by the closed form instead, so "the vacuum edges are still exactly zero"
    and "the feed current disappeared" read as two separate facts.
    """
    acc = case["acc"] if acc is None else acc
    g = _source_block(case)
    analytic, _vol = _analytic_source_moment(case)
    P = acc[..., 0]
    return float((np.delete(np.abs(P), g, axis=1).max(axis=(1, 2))
                  / np.abs(analytic)).max())


# ---------------------------------------------------------------------------
# 1. The edge set, the volumes, the block map and the centres
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["uniform", "graded"])
def test_geometry_matches_the_plane_route(kind):
    """Same edges, same volumes, same blocks, same centres as the plane route.

    The monitor builds its slab from the grid arrays before the run; the
    plane route above builds its edge table from the cell widths alone. The
    weights of slot 0 are the edge volumes.
    """
    case = _reference(kind)
    m = case["monitor"]
    edges, blocks = _reference_blocks(kind)
    assert edges["pos"].shape[0] == m.n_edges
    assert blocks["key"].shape[0] == m.n_blocks
    np.testing.assert_array_equal(blocks["key"], m.block_key)
    centres_err = float(np.max(np.abs(np.asarray(m.centres)
                                      - blocks["centres"])))
    assert centres_err < 1e-12, centres_err

    w = [np.asarray(getattr(m, f"w_e{c}"))[0] for c in "xyz"]
    seg = np.asarray(m.seg).reshape(w[0].shape[0], w[0].shape[1])
    worst = 0.0
    for c in range(3):
        sel = edges["comp"] == c
        idx = edges["idx"][sel]
        iis = idx[:, 0] - case["i_range"][0]
        jjs = idx[:, 1] - case["j_range"][0]
        kks = idx[:, 2] - case["k_nodes"][0]
        got = w[c][iis, jjs, kks]
        want = edges["vol"][sel]
        worst = max(worst, float(np.max(np.abs(got - want) / want)))
        np.testing.assert_array_equal(seg[iis, jjs], blocks["gid"][sel])
    assert worst < 1e-6, worst


def test_graded_edge_volumes_are_primal_times_midpoint_spans():
    """On the graded axis the volume is not the primal cell cubed.

    An edge's control volume is the primal cell on its OWN axis times the
    midpoint-to-midpoint distance on each of the other two. On the graded
    fixture those differ by up to 13 %, so this is the check that can see a
    primal-for-dual swap in the volume.
    """
    case = _reference("graded")
    m = case["monitor"]
    cx, cy, cz = (np.asarray(c, dtype=np.float64) for c in case["cells"])
    ii = np.arange(m.i_lo, m.i_hi)
    jj = np.arange(m.j_lo, m.j_hi)
    kk = np.arange(m.k_lo, m.k_hi + 1)
    kz = np.arange(m.k_lo, m.k_hi)
    sx = np.array([_midpoint_span(cx, i) for i in ii])
    sy = np.array([_midpoint_span(cy, j) for j in jj])
    sz = np.array([_midpoint_span(cz, k) for k in kk])
    want = {
        "x": cx[ii][:, None, None] * sy[None, :, None] * sz[None, None, :],
        "y": sx[:, None, None] * cy[jj][None, :, None] * sz[None, None, :],
        # the z-edge volume's third factor is its own primal cell, not a span
        "z": sx[:, None, None] * sy[None, :, None] * cz[kz][None, None, :],
    }
    ratio = cz[m.k_lo:m.k_hi + 1] / sz
    assert ratio.min() < 0.9 and ratio.max() > 1.1, "this axis is not graded"
    worst = max(float(np.max(np.abs(np.asarray(getattr(m, f"w_e{c}"))[0]
                                    - want[c]) / want[c])) for c in "xyz")
    assert worst < 1e-6, worst


def test_realized_window_is_the_nearest_node_to_each_corner():
    """Metres to indices: the nearest node at each corner.

    The tutorial patch's board edges land halfway between two nodes, which is
    the case where a rule can go either way, so the corners here are placed
    halfway on purpose. Which node is nearer is then decided by the last bit
    of the two distances, so the rule is stated here on the grid's own node
    line: ``argmin |node - corner|`` at both ends, widened by the margin,
    half-open at the top.
    """
    grid = Grid(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 2.4e-2), dx=DX,
                cpml_layers=6)
    nx, ny, nz = grid.shape
    nodes = [(np.arange(n) - pad) * DX for n, pad in
             ((nx, grid.pad_x_lo), (ny, grid.pad_y_lo), (nz, grid.pad_z_lo))]
    lo_m = float(nodes[0][10]) + 0.5 * DX       # exactly between two nodes
    hi_m = float(nodes[0][14]) + 0.5 * DX
    monitor = current_moment_monitor_from_grid(
        grid, corner_lo=(lo_m, lo_m, float(nodes[2][11])),
        corner_hi=(hi_m, hi_m, float(nodes[2][13])),
        block_size=3 * DX, freqs=FREQS, margin_cells=(2, 2, 0))
    near_lo = int(np.argmin(np.abs(nodes[0] - lo_m)))
    near_hi = int(np.argmin(np.abs(nodes[0] - hi_m)))
    assert near_lo in (10, 11) and near_hi in (14, 15)
    want = (near_lo - 2, near_hi + 2 + 1)
    assert (monitor.i_lo, monitor.i_hi) == want
    assert (monitor.j_lo, monitor.j_hi) == want
    assert (monitor.k_lo, monitor.k_hi) == (11, 13)
    n_planes = monitor.k_hi - monitor.k_lo + 1
    edges = ((monitor.i_hi - monitor.i_lo) * (monitor.j_hi - monitor.j_lo)
             * (2 * n_planes + n_planes - 1))
    assert monitor.n_edges == edges
    assert monitor.block_cells == 3


# ---------------------------------------------------------------------------
# 2. The vacuum identity and the closed form for the source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["uniform", "graded"])
def test_vacuum_identity_source_block_and_the_rest(kind):
    """Only the source block carries current, and it carries the analytic one.

    The closed form exercises the ez edge's volume only (dual_x * dual_y *
    primal_z), so on the graded fixture it sees the z PRIMAL width and not the
    z DUAL span that enters the ex and ey volumes; the volume test above is
    the general witness for those.
    """
    case = _reference(kind)
    g = _source_block(case)
    residual = _vacuum_residual(case)
    analytic, _vol = _analytic_source_moment(case)
    got = case["acc"][..., 0][:, g, 2]
    err = np.abs(got - analytic) / np.abs(analytic)
    assert residual <= VACUUM_BAR[kind], residual
    assert err.max() <= 1e-4, err


# ---------------------------------------------------------------------------
# 3. Equality with the plane route
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loaded", [False, True], ids=["vacuum", "loaded"])
@pytest.mark.parametrize("kind", ["uniform", "graded"])
def test_equality_with_the_plane_route(kind, loaded):
    """The in-loop moments and the plane route's are the same numbers.

    They differ by where the current is stamped and by the field left
    standing at the end of the record, both exact and both written into the
    route above; what is left is the order the float32 sums run in. The
    loaded run adds a lossy dielectric block, so polarization and loss
    current and the edge-averaged material faces enter both sides.
    """
    case = _reference(kind, loaded)
    _edges, blocks = _reference_blocks(kind, loaded)
    worst = _disagreement(case["acc"], case["monitor"], blocks)
    assert worst <= EQUALITY_BAR, worst


@pytest.mark.parametrize("kind", ["uniform", "graded"])
def test_the_modules_own_conversion_matches_the_plane_route(kind):
    """``end_of_record_moments`` and ``to_post_processing_convention``.

    These two are the module's public spelling of the same algebra the route
    above writes out; mapping the accumulator through them has to land on
    the route's own post-processing moments — ``curl(H_dft) - j w~ eps0
    E_dft`` summed into blocks straight from the planes, with no end term
    taken off and none of the module's put back.
    """
    case = _reference(kind)
    _edges, blocks = _reference_blocks(kind)
    m = case["monitor"]
    dt = float(case["grid"].dt)
    end = end_of_record_moments(m, case["state"], case["n_steps"], dt)
    mine = to_post_processing_convention(case["acc"], end, dt, FREQS,
                                         plane_stamp_steps=case["stamp"])
    w = 2.0 * np.pi * FREQS
    # The route's ``post`` blocks carry exp(+j w dt (s - 1/2)); take it off.
    rephase = np.exp(-1j * w * dt * (case["stamp"] - 0.5))
    got = dict(zip("PQT", moments_to_PQT(mine, m)))
    worst = 0.0
    for name in "PQT":
        for f in range(FREQS.size):
            want = rephase[f] * blocks["scale"][name][f]
            worst = max(worst, _rel_l2(got[name][f], want))
    assert worst <= EQUALITY_BAR, worst


def test_the_two_runners_stamp_dft_planes_one_step_apart():
    """The plane probes of the two runners write different times on one state.

    Both runners sample the plane probes at the same slot — electric field at
    ``(n+1) dt``, magnetic at ``(n+1/2) dt`` — and ``rfx/simulation.py``
    stamps it ``st.step * dt`` while ``rfx/nonuniform.py`` stamps it
    ``step_idx * dt``, one whole timestep earlier. A magnitude spectrum cannot
    see it; a current, which subtracts a scaled electric field from a
    magnetic curl, carries it in full. This pins each lane to its constant
    and the size of the difference to ``|exp(-j w dt) - 1|``.
    """
    uni = _reference("uniform")
    gra = _reference("graded")
    assert plane_stamp_steps(uni["grid"]) == UNIFORM_PLANE_STAMP_STEPS == 1.0
    assert plane_stamp_steps(gra["grid"]) == NONUNIFORM_PLANE_STAMP_STEPS == 0.0
    _e, wrong = _route_blocks(gra, stamp=UNIFORM_PLANE_STAMP_STEPS)
    a_wrong = _disagreement(gra["acc"], gra["monitor"], wrong, orders=("P",))
    one_step = float(np.max(np.abs(
        np.exp(-2j * np.pi * FREQS * float(gra["grid"].dt)) - 1.0)))
    assert 0.5 * one_step <= a_wrong <= 2.0 * one_step, (a_wrong, one_step)


# ---------------------------------------------------------------------------
# 4. The far field
# ---------------------------------------------------------------------------

def _direct_dipole_sum(theta, phi, pos, p, k):
    """E_theta, E_phi of point current moments, summed one by one.

    ``E = -j k eta / (4 pi) sum_i p_i exp(+j k rhat.r_i)``, projected on
    theta-hat and phi-hat — the electric-current half of the radiation
    integral in ``rfx.farfield``'s convention (1/r omitted).
    """
    out_t = np.zeros((theta.size, phi.size), dtype=np.complex128)
    out_p = np.zeros_like(out_t)
    for a, th in enumerate(theta):
        for b, ph in enumerate(phi):
            r = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                          np.cos(th)])
            t_hat = np.array([np.cos(th) * np.cos(ph),
                              np.cos(th) * np.sin(ph), -np.sin(th)])
            p_hat = np.array([-np.sin(ph), np.cos(ph), 0.0])
            n_vec = (p * np.exp(1j * k * (pos @ r))[:, None]).sum(axis=0)
            pref = -1j * k * ETA0 / (4.0 * np.pi)
            out_t[a, b] = pref * (n_vec @ t_hat)
            out_p[a, b] = pref * (n_vec @ p_hat)
    return out_t, out_p


def test_block_far_field_expansion_against_a_direct_sum():
    """The block expansion is the Taylor series of the phase, to its order.

    A cluster of random axis-aligned current elements in a cube of side s,
    summed one by one, against the cluster reduced to one block at its
    centre. What the block drops is the next term of the phase, so the error
    of order L falls as (k s)^(L+1); a slope that is not L+1 means the
    expansion is not the expansion. The NumPy and JAX spellings are the same
    expression.
    """
    rng = np.random.default_rng(20260921)
    n, side = 60, 0.01
    pos = (rng.random((n, 3)) - 0.5) * side
    comp = rng.integers(0, 3, size=n)
    amp = rng.normal(size=n) + 1j * rng.normal(size=n)
    p = np.zeros((n, 3), dtype=np.complex128)
    p[np.arange(n), comp] = amp
    centre = pos.mean(axis=0)
    d = pos - centre
    P = p.sum(axis=0)[None]
    Q = np.einsum("na,nc->ac", d, p)[None]
    T = np.einsum("na,nb,nc->abc", d, d, p)[None]
    theta = np.linspace(0.0, np.pi, 13)
    phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ks = np.array([0.05, 0.1, 0.2, 0.4])
    for order in (0, 1, 2):
        errs = []
        for ks_i in ks:
            k = ks_i / side
            ref = _direct_dipole_sum(theta, phi, pos, p, k)
            got = block_far_field_np(theta, phi, centre[None], P, Q, T, k,
                                     order)
            errs.append(_rel_l2(np.concatenate([g.ravel() for g in got]),
                                np.concatenate([r.ravel() for r in ref])))
        slope = float(np.polyfit(np.log(ks), np.log(errs), 1)[0])
        assert abs(slope - (order + 1)) <= 0.3, (order, slope, errs)
    k = 0.2 / side
    a = block_far_field_np(theta, phi, centre[None], P, Q, T, k, 2)
    b = block_far_field_jax(theta, phi, centre[None], P, Q, T, k, 2)
    for x, y in zip(a, b):
        assert _rel_l2(np.asarray(y), x) < 1e-5


# The strip dipole below, against the face-centre NTFF box on the same run
# (CPU, JAX 0.6.2, 1 mm cells, 2 mm blocks through the 4 mm slab, k*|delta|
# 0.28 / 0.38 / 0.48, order 2). The comparison reads only what a common time
# origin cannot change: the two routes stamp their fields at times whose
# offset has not been established, so a complex distance would mix a
# per-frequency phase into the verdict. Measured at 6 / 8 / 10 GHz: the
# largest difference of the two patterns, each in dB below its own peak, over
# the directions where the box's pattern is within 20 dB of its peak,
# 0.0054 / 0.0103 / 0.0221 dB; directivity 0.0006 / 0.0031 / 0.0082 dB apart;
# radiated power 0.0026 / 0.0068 / 0.0158 dB apart. With 4 mm blocks
# (k*|delta| up to 0.67, now refused) the same run read 0.0061 / 0.0128 /
# 0.0278 dB on the pattern. Bars: twice the worst measured value of each,
# rounded up to two digits. P alone (order 0) on the same accumulator reads
# 0.107 / 0.211 / 0.369 dB on the pattern and must stay above the pattern bar
# at every frequency.
DIPOLE_SHAPE_FLOOR_DB = -20.0
DIPOLE_SHAPE_BAR_DB = 0.045
DIPOLE_DIRECTIVITY_BAR_DB = 0.017
DIPOLE_POWER_BAR_DB = 0.032
DIPOLE_FREQS = np.array([6e9, 8e9, 10e9])


@functools.lru_cache(maxsize=None)
def _strip_dipole_run():
    """A 14 mm PEC strip dipole along x, fed at a one-cell gap, in vacuum.

    The currents that radiate are the conductor's own surface current (the
    curl of H at the shorted edges) and the feed's; the slab holds all of it.
    """
    from rfx import Box, Simulation
    L, lx, ly, lz = 14e-3, 30e-3, 20e-3, 20e-3
    xc, yc, zc = lx / 2, ly / 2, lz / 2
    sim = Simulation(freq_max=12e9, domain=(lx, ly, lz), dx=1e-3,
                     boundary="cpml", cpml_layers=8)
    sim.add(Box((xc - L / 2, yc, zc), (xc - 1e-3, yc + 1e-3, zc)),
            material="pec")
    sim.add(Box((xc + 1e-3, yc, zc), (xc + L / 2, yc + 1e-3, zc)),
            material="pec")
    sim.add_source((xc, yc, zc), "ex")
    sim.add_ntff_box((4e-3, 4e-3, 4e-3), (lx - 4e-3, ly - 4e-3, lz - 4e-3),
                     freqs=DIPOLE_FREQS)
    sim.add_current_moment_monitor(
        (xc - L / 2 - 2e-3, yc - 1e-3, zc - 2e-3),
        (xc + L / 2 + 2e-3, yc + 3e-3, zc + 2e-3),
        block_size=2e-3, freqs=DIPOLE_FREQS)
    return sim.run(n_steps=1200, skip_preflight=True)


def test_far_field_matches_the_ntff_box_on_a_strip_dipole():
    """The pattern from the moments is the Huygens box's, on the same run.

    ``current_moment_far_field`` returns the repository's ``FarFieldResult``,
    so ``directivity``, ``radiation_pattern`` and ``axial_ratio`` take it
    unchanged.
    """
    from rfx import (axial_ratio, compute_far_field, directivity,
                     radiation_pattern)
    res = _strip_dipole_run()
    theta = np.linspace(0.0, np.pi, 37)
    phi = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    box = compute_far_field(res.ntff_data, res.ntff_box, res.grid, theta, phi)
    mon = current_moment_far_field(res, theta, phi)
    assert mon.E_theta.shape == box.E_theta.shape == (3, 37, 36)
    np.testing.assert_allclose(mon.freqs, DIPOLE_FREQS, rtol=1e-7)

    def shape_db(ff):
        """Largest dB difference above the floor, per frequency."""
        a, b = radiation_pattern(ff), radiation_pattern(box)
        return [float(np.max(np.abs(a[f] - b[f])[b[f] >= DIPOLE_SHAPE_FLOOR_DB]))
                for f in range(3)]

    def radiated(ff):
        power = np.abs(ff.E_theta) ** 2 + np.abs(ff.E_phi) ** 2
        th, ph = np.asarray(ff.theta), np.asarray(ff.phi)
        w = (np.sin(th) * np.gradient(th))[None, :, None] * np.gradient(ph)
        return np.sum(power * w, axis=(1, 2))

    shape = shape_db(mon)
    assert max(shape) <= DIPOLE_SHAPE_BAR_DB, shape
    d_db = np.abs(directivity(mon) - directivity(box))
    assert d_db.max() <= DIPOLE_DIRECTIVITY_BAR_DB, d_db
    p_db = np.abs(10.0 * np.log10(radiated(mon) / radiated(box)))
    assert p_db.max() <= DIPOLE_POWER_BAR_DB, p_db
    assert np.isfinite(axial_ratio(mon)).all()

    # Negative control on the same accumulator: the total moment P alone.
    m = res.current_moment_monitor
    p_only = res._replace(
        current_moment_data=(np.asarray(res.current_moment_data[0])[..., :1],),
        current_moment_monitor=m._replace(order=0))
    control = shape_db(current_moment_far_field(p_only, theta, phi))
    assert min(control) > DIPOLE_SHAPE_BAR_DB, control


def test_far_field_refuses_a_result_without_moments():
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                     cpml_layers=6, boundary="cpml")
    sim.add_source((1.2e-2, 1.2e-2, 1.2e-2), "ez")
    res = sim.run(n_steps=4, skip_preflight=True)
    with pytest.raises(ValueError, match="add_current_moment_monitor"):
        current_moment_far_field(res, np.array([0.5]), np.array([0.0]))


# ---------------------------------------------------------------------------
# 5. The gradient through the monitor
# ---------------------------------------------------------------------------

GRAD_FREQS = np.array([4e9, 5e9])     # k*|delta| 0.39 / 0.49
GRAD_N_STEPS = 220


GRAD_BOX = ((7e-3, 5e-3, 5e-3), (8e-3, 7e-3, 7e-3))   # 2 x 3 x 3 cells


def _grad_problem():
    """A soft Ex element beside a dielectric block whose eps_r is the knob.

    The block is a ``design_box``: a traced whole-grid ``eps_override`` could
    put material anywhere and is refused while the monitor is on, a design
    box is checked against the slab.
    """
    from rfx import Simulation
    sim = Simulation(freq_max=12e9, domain=(12e-3, 12e-3, 12e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=6, precision="float64")
    sim.add_source((6e-3, 6e-3, 6e-3), "ex")
    sim.add_current_moment_monitor((2e-3, 2e-3, 2e-3), (10e-3, 10e-3, 10e-3),
                                   block_size=4e-3, freqs=GRAD_FREQS)
    theta, phi = np.array([0.3, 1.2]), np.array([0.4])

    def objective(eps_r):
        box_eps = jnp.ones((2, 3, 3), dtype=jnp.float64) * eps_r
        fr = sim.forward(design_box=GRAD_BOX, design_eps_override=box_eps,
                         n_steps=GRAD_N_STEPS, checkpoint=False,
                         skip_preflight=True)
        ff = current_moment_far_field(fr, theta, phi)
        return jnp.sum(jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2)

    return objective


def test_gradient_through_the_monitor_matches_a_central_difference():
    """``jax.grad`` of a far-field power through the monitor, float64.

    The power radiated into two directions at two frequencies, as a function
    of the permittivity of a small block beside the source, differentiated
    under ``jax.jit`` (the way ``optimize(jit=True)`` compiles a step).
    Measured (CPU, at eps_r = 2, 4 / 5 GHz): the AD gradient and the central
    difference agree to 1.4e-9 relative at a step of 1e-4, 1.4e-7 at 1e-3 and
    1.4e-5 at 1e-2 — the difference falls as the square of the step, so it is
    the difference quotient's own truncation. The jitted value is the plain
    call's (2.3e-16 relative).
    """
    with enable_x64():
        objective = _grad_problem()
        a0, h = 2.0, 1e-4
        value, grad = jax.jit(jax.value_and_grad(objective))(jnp.float64(a0))
        fd = (float(objective(jnp.float64(a0 + h)))
              - float(objective(jnp.float64(a0 - h)))) / (2.0 * h)
        assert float(value) > 0.0 and fd != 0.0
        assert abs(float(grad) - fd) <= 1e-7 * abs(fd), (float(grad), fd)
        plain = float(objective(jnp.float64(a0)))
        assert abs(plain - float(value)) <= 1e-12 * plain


@pytest.mark.slow
def test_gradient_without_jit_is_the_jitted_one():
    """``jax.grad`` outside ``jax.jit``: the monitor's frequencies are then
    concrete while the accumulator is traced (10 s on CPU, op by op)."""
    with enable_x64():
        objective = _grad_problem()
        grad = float(jax.grad(objective)(jnp.float64(2.0)))
        grad_j = float(jax.jit(jax.grad(objective))(jnp.float64(2.0)))
        assert abs(grad - grad_j) <= 1e-12 * abs(grad_j), (grad, grad_j)


# ---------------------------------------------------------------------------
# 6. The monitor leaves the solve alone, and the two runners agree
# ---------------------------------------------------------------------------

def test_monitor_does_not_move_the_field_by_one_bit():
    """Turning the monitor on must not change the solve at all."""
    grid = Grid(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 2.4e-2), dx=DX,
                cpml_layers=6)
    nx, ny, nz = grid.shape
    ci, cj, ck = nx // 2, ny // 2, nz // 2
    probes = [ProbeSpec(i=ci + 3, j=cj + 2, k=ck, component="ez")]
    on = _uniform_case(with_planes=False, probes=probes)
    off = run(grid, init_materials(grid.shape), N_STEPS, boundary="cpml",
              sources=[on["source"]], probes=probes)
    a = np.asarray(off.time_series)
    b = on["time_series"]
    assert np.array_equal(a, b), float(np.max(np.abs(a - b)))
    for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
        assert np.array_equal(np.asarray(getattr(off.state, comp)),
                              np.asarray(getattr(on["state"], comp))), comp


def test_nonuniform_lane_matches_the_uniform_lane():
    """The same slab on the same (uniform) mesh, through the other runner."""
    from rfx.nonuniform import make_nonuniform_grid, run_nonuniform

    uni = _uniform_case(with_planes=False)
    grid = uni["grid"]
    nz_phys = grid.nz - grid.pad_z_lo - grid.pad_z_hi - 1
    nu_grid = make_nonuniform_grid(
        (grid.domain[0], grid.domain[1]), np.full(nz_phys, DX), DX,
        cpml_layers=grid.cpml_layers)
    assert tuple(nu_grid.shape) == tuple(grid.shape)
    src = uni["source"]
    nu = run_nonuniform(nu_grid, init_materials(nu_grid.shape),
                        uni["n_steps"],
                        sources=[(src.i, src.j, src.k, src.component,
                                  src.waveform)],
                        current_moments=uni["monitor"])
    err = _rel_l2(np.asarray(nu["current_moment_data"][0]), uni["acc"])
    assert err < 1e-4, err


# ---------------------------------------------------------------------------
# 7. Through the public API: a port, a substrate, both lanes
# ---------------------------------------------------------------------------

API_FREQS = np.array([5e9, 7e9])      # k*|delta| up to 0.47 / 0.49


def _api_board(lane):
    """A 50 ohm wire port across a lossy substrate, through ``Simulation``.

    The graded lane grades z under the port; its port drive is built from
    the materials after the port's own load is stamped. Every H and E plane
    the route needs is recorded with ``add_dft_plane_probe``.
    """
    from rfx import Box, Simulation
    if lane == "graded":
        # nodes 0, 1, 2, 3.0, 3.8, 4.4, 4.9, 5.4, 6.0, 6.8 ... mm
        dz = np.array([1.0, 1.0, 1.0, 0.8, 0.6, 0.5, 0.5, 0.6, 0.8, 1.0, 1.0,
                       1.0]) * 1e-3
        kw, lz, top = {"dz_profile": dz}, float(dz.sum()), 4.9e-3
    else:
        kw, lz, top = {}, 10e-3, 5.0e-3
    sim = Simulation(freq_max=12e9, domain=(16e-3, 16e-3, lz), dx=1e-3,
                     boundary="cpml", cpml_layers=6, **kw)
    sim.add_material("sub", eps_r=3.38, sigma=0.02)
    sim.add(Box((3e-3, 3e-3, 3e-3), (13e-3, 13e-3, top)), material="sub")
    sim.add_port((8e-3, 8e-3, 3e-3), "ez", impedance=50.0, extent=top - 3e-3)
    sim.add_current_moment_monitor((1e-3, 1e-3, 2.0e-3), (15e-3, 15e-3, 6e-3),
                                   block_size=4e-3, freqs=API_FREQS)
    grid = (sim._build_nonuniform_grid() if lane == "graded"
            else sim._build_grid())
    probe_monitor = _realized(sim, grid)
    k0, k1 = probe_monitor.k_lo, probe_monitor.k_hi
    specs = _plane_specs(range(k0, k1 + 1), range(k0, k1))
    nodes_z = _node_line(grid, 2)
    for comp, k in specs:
        sim.add_dft_plane_probe(axis="z", coordinate=float(nodes_z[k]),
                                component=comp, freqs=API_FREQS,
                                name=f"{comp}{k}")
    return sim, grid, specs


def _realized(sim, grid):
    from rfx.current_moments import monitor_for_simulation
    return monitor_for_simulation(sim, grid)


def _cells(grid, axis):
    """Primal cell widths, the float64 copy where the grid keeps one."""
    for name in (("dx_arr_f64", "dy_arr_f64", "dz_f64")[axis],
                 ("dx_arr", "dy_arr", "dz")[axis]):
        arr = getattr(grid, name, None)
        if arr is not None:
            return np.asarray(arr, dtype=np.float64)
    return np.asarray(grid.cells(axis), dtype=np.float64)


def _node_line(grid, axis):
    """Node positions from the cell widths, zero at the inner absorber edge."""
    cells = _cells(grid, axis)
    pad = int((grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[axis])
    edges = np.concatenate([[0.0], np.cumsum(cells)])
    return edges[:cells.size] - edges[pad]


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_the_public_run_matches_the_plane_route_with_a_port(lane):
    """``Simulation.run`` with a wire port in a substrate, on both lanes.

    The monitor is realized against the grid the run builds; the planes are
    the run's own ``Result.dft_planes``. Port load, substrate polarization
    and loss current and the feed all enter both sides.
    """
    sim, grid, specs = _api_board(lane)
    res = sim.run(n_steps=500, skip_preflight=True)
    m = res.current_moment_monitor
    assert m is not None and res.current_moment_data is not None
    planes = {spec: np.asarray(res.dft_planes[f"{spec[0]}{spec[1]}"]
                               .accumulator) for spec in specs}
    stamp = 1.0 if lane == "uniform" else 0.0
    edges = _plane_route_edges(
        planes, res.state, nodes=[_node_line(grid, a) for a in range(3)],
        cells=[_cells(grid, a) for a in range(3)],
        window=((m.i_lo, m.i_hi), (m.j_lo, m.j_hi), (m.k_lo, m.k_hi)),
        freqs=np.asarray(m.freqs, dtype=np.float64), dt=float(grid.dt),
        n_steps=500, stamp=stamp)
    blocks = _plane_route_blocks(edges, i_lo=m.i_lo, j_lo=m.j_lo,
                                 block_cells=m.block_cells)
    assert float(np.max(np.abs(np.asarray(m.centres)
                               - blocks["centres"]))) < 1e-12
    worst = _disagreement(np.asarray(res.current_moment_data[0]), m, blocks)
    assert worst <= EQUALITY_BAR, worst


def _ringdown_box(lane):
    """A lossy dielectric block in open space, fed by a 50 ohm wire port."""
    from rfx import Box, GaussianPulse, Simulation
    mm = 1e-3
    if lane == "graded":
        # nodes 0, 1, 2, 2.8, 3.4, 4.8, 6, 7, 8 mm
        dz = np.array([1.0, 1.0, 0.8, 0.6, 1.4, 1.2, 1.0, 1.0]) * mm
        kw, lz, top = {"dz_profile": dz}, float(dz.sum()), 3.4 * mm
        pos, ext = (6 * mm, 6 * mm, 2.8 * mm), 0.6 * mm
    else:
        kw, lz, top = {}, 8 * mm, 4 * mm
        pos, ext = (6 * mm, 6 * mm, 2 * mm), 1.0 * mm
    sim = Simulation(freq_max=20e9, domain=(12 * mm, 12 * mm, lz),
                     dx=1.0 * mm, boundary="cpml", cpml_layers=6, **kw)
    sim.add_material("fill", eps_r=2.2, sigma=0.0127)
    sim.add(Box((3 * mm, 3 * mm, 2 * mm), (9 * mm, 9 * mm, top)),
            material="fill")
    sim.add_port(position=pos, component="ez", impedance=50.0, extent=ext,
                 waveform=GaussianPulse(f0=13.0e9, bandwidth=0.8, cutoff=4.5))
    sim.add_current_moment_monitor(
        (1 * mm, 1 * mm, 1 * mm), (11 * mm, 11 * mm, 6 * mm),
        block_size=2 * mm, freqs=np.array([6e9, 8e9]))   # k*|delta| <= 0.49
    return sim


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_ringdown_completion_returns_the_same_moments(lane):
    """``run(ringdown=RingdownSpec())`` adds probes and completes S after the
    run; the moments it returns are the plain run's. On the graded lane the
    completion assembles S through the lane's result assembly with a
    stand-in set-up that carries no monitor.

    Bit for bit on CPU. On a GPU the added probes compile to a different
    program and the float32 sums can round differently in the last digits
    (RTX 3090, JAX 0.6.2, on an earlier closed-box version of this fixture:
    0.06 ULP uniform, 1.0 ULP graded), so the comparison is the repository's
    cross-trace rule, the one ``run(ringdown=)``'s own W0 witness reads: at
    most 9 ULP, counted at the array's peak in its own real dtype."""
    from rfx.ringdown import RingdownSpec
    n = 800 if lane == "uniform" else 1200
    kw = dict(n_steps=n, compute_s_params=True,
              s_param_freqs=np.array([10e9, 12.4e9, 14e9]),
              skip_preflight=True)
    plain = _ringdown_box(lane).run(**kw)
    done = _ringdown_box(lane).run(ringdown=RingdownSpec(), **kw)
    assert done.ringdown is not None
    a = np.asarray(plain.current_moment_data[0])
    b = np.asarray(done.current_moment_data[0])
    peak = np.abs(a).max()
    assert peak > 0.0
    ulps = float(np.abs(b - a).max() / np.spacing(peak.astype(a.real.dtype)))
    assert ulps <= 9.0, ulps


# ---------------------------------------------------------------------------
# 8. Mutations of the monitor
# ---------------------------------------------------------------------------

def _mutations(cells, block_cells=BLOCK_CELLS, cell=DX):
    """Every monitor mutation, with all the helper calls left in place.

    Each entry goes through the SAME builder with one derived quantity
    replaced by a wrong one, so the reduction, the block map, the segment sum
    and the phase accumulation all still run.

    The declared colour is per fixture: on a uniform mesh the dual spacing IS
    the primal cell, bit for bit, so the last two rows are no-ops there and
    are recorded GREEN rather than skipped.
    """
    dual = [e_dual_spacings(c) for c in cells]
    signs = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    signs[1] = -signs[1]
    return {
        "curl sign flipped (dHy/dz in Jx)": (
            {"uniform": "RED", "graded": "RED"},
            dict(curl_signs=tuple(signs))),
        "stamped at n*dt": (
            {"uniform": "RED", "graded": "RED"},
            dict(half_step=0.0)),
        "block centre shifted half a block": (
            {"uniform": "RED", "graded": "RED"},
            dict(_centre_shift_x=0.5 * block_cells * cell)),
        "edge volume: primal instead of dual on z": (
            {"uniform": "GREEN", "graded": "RED"},
            dict(volume_spacings=(dual[0], dual[1], np.asarray(cells[2])))),
        "curl divided by primal instead of dual on z": (
            {"uniform": "GREEN", "graded": "RED"},
            dict(curl_spacings=(dual[0], dual[1], np.asarray(cells[2])))),
    }


# The first two run in the pull-request lane; the rest are the same check
# on further defects and run with the slow tests.
MUTATION_NAMES = [
    name if i < 2 else pytest.param(name, marks=pytest.mark.slow)
    for i, name in enumerate(_mutations([np.ones(3)] * 3))]


@pytest.mark.parametrize("kind", ["uniform", "graded"])
@pytest.mark.parametrize("name", MUTATION_NAMES)
def test_monitor_mutations_are_caught(kind, name):
    """Each defect, put back, against the plane route on the loaded board.

    Loaded, because a volume error on an edge that carries no current is no
    error at all: in vacuum only the source's Ez edge carries current, and
    its volume has no z dual in it. The lossy block puts current on Ex and Ey
    edges across the graded z cells. The monitor does not touch the fields,
    so the reference run's planes are the plane route for the mutated run
    too.
    """
    ref = _reference(kind, True)
    _edges, blocks = _reference_blocks(kind, True)
    expect, kwargs = _mutations(ref["cells"], ref["block_cells"])[name]
    kwargs = dict(kwargs)
    shift = kwargs.pop("_centre_shift_x", None)
    if shift is not None:
        centres = np.asarray(ref["monitor"].centres).copy()
        centres[:, 0] += shift
        kwargs["centres_override"] = centres
    case = CASE_BUILDERS[kind](monitor_kwargs=kwargs, with_planes=False,
                               loaded=True)
    worst = _disagreement(case["acc"], case["monitor"], blocks)
    red = worst > EQUALITY_BAR
    assert ("RED" if red else "GREEN") == expect[kind], worst


# ---------------------------------------------------------------------------
# 9. Mutations of the snapshot slot, in the runners themselves
# ---------------------------------------------------------------------------

def _load_mutated(path: Path, name: str, replacements):
    """Import a copy of a runner module with the given text edits applied."""
    src = path.read_text()
    for old, new in replacements:
        assert src.count(old) == 1, (path.name, src.count(old), old[:60])
        src = src.replace(old, new)
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


UNIFORM_SRC = REPO_ROOT / "rfx" / "simulation.py"
GRADED_SRC = REPO_ROOT / "rfx" / "nonuniform.py"

_SNAP_UNI = ("        if ctx.use_current_moments:\n"
             "            e_prev_slab = _slab_e_snapshot(st, ctx.current_moments)\n")
_ACC_UNI = ("        if ctx.use_current_moments:\n"
            "            cm_new = ctx.accumulate_current_moments(\n"
            "                carry[\"current_moments\"], st, e_prev_slab,\n"
            "                ctx.current_moments, dt, step_idx)\n")
_SRCLOOP_UNI = "        # Soft sources — cast source value to field dtype"
_RLC_UNI = "        # Lumped RLC ADE update (after E update + boundaries, before sources)"

_SNAP_NU = ("        if use_current_moments:\n"
            "            e_prev_slab = _slab_e_snapshot(st, current_moments)\n")
_ACC_NU = ("        new_cm = None\n"
           "        if use_current_moments:\n"
           "            new_cm = _accumulate_cm(\n"
           "                carry[\"current_moments\"], st, e_prev_slab, current_moments,\n"
           "                dt, step_idx)\n")
_SRCLOOP_NU = "        # Sources (point sources + wire port excitation)"
_RLC_NU = "        # Lumped RLC ADE update (after E update + boundaries, before sources)"


def _runner_mutations(kind):
    """Three ways of getting the snapshot slot wrong, in the runner's own text.

    ``after the E update`` and ``after the source loop`` both break the vacuum
    identity, because the difference of the two electric fields no longer
    cancels the curl. ``accumulate before the source loop`` does NOT: every
    vacuum edge still reads exactly zero, and the only thing that changes is
    that the impressed feed current disappears. That is the one a vacuum
    witness cannot see, so the closed-form check is what has to catch it.
    """
    if kind == "uniform":
        snap, acc, srcloop, rlc = _SNAP_UNI, _ACC_UNI, _SRCLOOP_UNI, _RLC_UNI
        path, name = UNIFORM_SRC, "_mutated_rfx_simulation"
        entry = "run"
    else:
        snap, acc, srcloop, rlc = _SNAP_NU, _ACC_NU, _SRCLOOP_NU, _RLC_NU
        path, name = GRADED_SRC, "_mutated_rfx_nonuniform"
        entry = "run_nonuniform"
    return {
        "snapshot taken after the E update": (
            path, name + "_a", entry, [(snap, ""), (rlc, snap + rlc)]),
        "snapshot taken after the source loop": (
            path, name + "_b", entry, [(snap, ""), (acc, snap + acc)]),
        "accumulated before the source loop": (
            path, name + "_c", entry, [(acc, ""), (srcloop, acc + srcloop)]),
    }


@pytest.mark.parametrize("kind", ["uniform", "graded"])
@pytest.mark.parametrize("mut", [
    "snapshot taken after the E update",
    pytest.param("snapshot taken after the source loop",
                 marks=pytest.mark.slow),
    pytest.param("accumulated before the source loop",
                 marks=pytest.mark.slow)])
def test_snapshot_slot_mutations_in_the_runner(kind, mut):
    """Move the snapshot or the accumulate call in a copy of the runner.

    Not a switch turned off: every helper call stays, the monitor is still
    built, reduced and phased, and only the STEP at which the electric field
    is read moves.
    """
    path, modname, entry, edits = _runner_mutations(kind)[mut]
    mod = _load_mutated(path, modname, edits)
    try:
        case = CASE_BUILDERS[kind](with_planes=False,
                                   runner=getattr(mod, entry))
    finally:
        sys.modules.pop(modname, None)
    vac = _vacuum_against_the_closed_form(case)
    analytic, _vol = _analytic_source_moment(case)
    g = _source_block(case)
    src_err = float(np.max(np.abs(case["acc"][..., 0][:, g, 2] - analytic)
                           / np.abs(analytic)))
    vac_red = vac > VACUUM_BAR[kind]
    src_red = src_err > 1e-4
    if mut == "accumulated before the source loop":
        # Every edge, source included, reads exactly zero: the electric field
        # this reads is the one BEFORE the step's injection, so the impressed
        # current never enters and the vacuum identity is satisfied trivially.
        assert not vac_red, vac
        assert src_red, src_err
    else:
        assert vac_red, vac
        assert src_red, src_err


@pytest.mark.parametrize("kind", ["uniform", "graded"])
def test_the_snapshot_is_the_previous_steps_accumulated_field(kind):
    """Step n's post-accumulate slab E is step n+1's ``e_prev_slab``.

    Written with public outputs only: the difference between an N-step and an
    (N-1)-step accumulator has to be the single step the monitor would build
    from the (N-1)-step run's FINAL state as ``e_prev`` and the N-step run's
    final state as the post-update one. Move the snapshot anywhere else in
    the step and this stops holding.
    """
    # Read while the pulse is still in the domain: by step 120 the CPML has
    # absorbed it and both accumulators' last increment is exactly zero in
    # float32, which makes the comparison 0/0 rather than a check.
    n = 60
    long = CASE_BUILDERS[kind](with_planes=False, n_steps=n)
    short = CASE_BUILDERS[kind](with_planes=False, n_steps=n - 1)
    m = long["monitor"]
    dt = float(long["grid"].dt)
    one, _comp = accumulate_current_moments(
        init_current_moment_data(m), long["state"],
        slab_e_snapshot(short["state"], m), m, dt, n - 1)
    # The two runs use different source waveform tables (they are cut to
    # their own length), so compare only over the steps they share.
    assert np.array_equal(long["wave"][:n - 1], short["wave"])
    got = long["acc"] - short["acc"]
    one = np.asarray(one)
    assert np.linalg.norm(one) > 0.0, "the rebuilt step is identically zero"
    # The two accumulators are float32 sums of ~30 comparable increments, so
    # their difference carries both totals' round-off, not just the step's.
    err = _rel_l2(got, one)
    assert err < 1e-4, err


# ---------------------------------------------------------------------------
# 10. Fences
# ---------------------------------------------------------------------------

def test_refuses_a_slab_inside_the_absorber():
    grid = Grid(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 2.4e-2), dx=DX,
                cpml_layers=6)
    with pytest.raises(ValueError, match="absorber"):
        current_moment_monitor_from_grid(
            grid, corner_lo=(-8e-3, -8e-3, 2e-3), corner_hi=(8e-3, 8e-3, 6e-3),
            block_size=6e-3, freqs=FREQS)


def test_refuses_a_periodic_declaration():
    """The refusal fires on the flags the CALLER passes.

    The uniform runner passes the run's effective ones, and the graded-mesh
    runner passes the DECLARED axes because its own stepper installs no
    periodic boundary at all. This covers the builder; the test below covers
    the graded lane's choice of what to hand it.
    """
    grid = Grid(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 2.4e-2), dx=DX,
                cpml_layers=6)
    with pytest.raises(NotImplementedError, match="periodic"):
        current_moment_monitor_from_grid(
            grid, corner_lo=(4e-3, 4e-3, 4e-3),
            corner_hi=(1.6e-2, 1.6e-2, 1.2e-2),
            block_size=6e-3, freqs=FREQS, periodic=(True, False, False))


def test_graded_lane_hands_the_monitor_the_declared_periodic_axes():
    from rfx import Simulation

    dz = np.full(10, DX)
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, float(dz.sum())),
                     dx=DX, dz_profile=dz, cpml_layers=6, boundary="cpml")
    sim.set_periodic_axes("x")
    sim.add_source(position=(1.2e-2, 1.2e-2, 1.0e-2), component="ez")
    sim.add_current_moment_monitor(
        corner_lo=(0.6e-2, 0.6e-2, 0.8e-2), corner_hi=(1.8e-2, 1.8e-2, 1.2e-2),
        block_size=6e-3, freqs=FREQS)
    with pytest.raises(NotImplementedError,
                       match="current-moment monitor.*periodic"):
        sim.run(n_steps=4, skip_preflight=True)


def test_refuses_a_slab_that_starts_at_index_zero():
    with pytest.raises(ValueError, match="index 1 or above"):
        build_current_moment_monitor(
            node_x=np.arange(10) * DX, node_y=np.arange(10) * DX,
            node_z=np.arange(10) * DX,
            cell_x=np.full(10, DX), cell_y=np.full(10, DX),
            cell_z=np.full(10, DX),
            i_range=(0, 4), j_range=(1, 4), k_node_range=(1, 3),
            block_cells=2, freqs=FREQS)


def test_refuses_a_graded_in_plane_mesh():
    from rfx.nonuniform import make_nonuniform_grid
    # dx_profile's first and last cell must equal the boundary dx (the CPML
    # profile is calibrated on it), so the grading sits in the middle.
    dx_prof = np.concatenate([np.full(4, DX), np.full(4, 0.5 * DX),
                              np.full(4, DX)])
    grid = make_nonuniform_grid((2.4e-2, 2.4e-2), np.full(12, DX), DX,
                                cpml_layers=6, dx_profile=dx_prof)
    with pytest.raises(NotImplementedError, match="uniform in-plane mesh"):
        current_moment_monitor_from_grid(
            grid, corner_lo=(4e-3, 4e-3, 4e-3),
            corner_hi=(1.4e-2, 1.4e-2, 1.0e-2), block_size=6e-3, freqs=FREQS)


def test_refuses_a_traced_mesh():
    """A mesh that is a design variable would move the baked-in weights."""
    from rfx import Simulation

    def build(delta):
        dz = jnp.full((10,), DX).at[4].add(delta).at[5].add(-delta)
        sim = Simulation(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 10 * DX),
                         dx=DX, dz_profile=dz, cpml_layers=6, boundary="cpml",
                         dt=0.5 * DX / C0 / np.sqrt(3.0), dt_min_cell=0.8 * DX)
        sim.add_source(position=(1.2e-2, 1.2e-2, 1.0e-2), component="ez")
        sim.add_current_moment_monitor(
            corner_lo=(0.6e-2, 0.6e-2, 0.8e-2),
            corner_hi=(1.8e-2, 1.8e-2, 1.2e-2), block_size=6e-3, freqs=FREQS)
        fr = sim.forward(n_steps=4, skip_preflight=True)
        return jnp.sum(fr.time_series)

    with pytest.raises(NotImplementedError, match="traced mesh profile"):
        jax.grad(build)(0.0)


def test_refuses_a_tfsf_source():
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                     cpml_layers=6, boundary="cpml")
    sim.add_tfsf_source(f0=6e9, bandwidth=0.5, margin=3)
    sim.add_current_moment_monitor(
        corner_lo=(0.6e-2, 0.6e-2, 0.6e-2), corner_hi=(1.8e-2, 1.8e-2, 1.8e-2),
        block_size=6e-3, freqs=FREQS)
    with pytest.raises(NotImplementedError,
                       match="current-moment monitor and a TFSF"):
        sim.run(n_steps=4, skip_preflight=True)


def test_refuses_the_fourth_order_stencil():
    """With ``stencil_order=4`` the E update differences H over the wider
    stencil, so the monitor's second-order curl would leave a current on
    every vacuum edge: 2.5-2.9 % of the source block's moment in its
    neighbours in a vacuum PEC box, against 1e-8..2e-7 at order 2."""
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                     boundary="pec", stencil_order=4)
    sim.add_source((1.2e-2, 1.2e-2, 1.2e-2), "ez", amplitude_kind="current")
    sim.add_current_moment_monitor(
        corner_lo=(0.6e-2, 0.6e-2, 0.6e-2), corner_hi=(1.8e-2, 1.8e-2, 1.8e-2),
        block_size=6e-3, freqs=FREQS)
    with pytest.raises(NotImplementedError, match="stencil_order=4"):
        sim.run(n_steps=4, skip_preflight=True)


@pytest.mark.parametrize("order", [0, 1])
def test_declaration_accepts_order_two_only(order):
    """Below order 2 the block centre's half-cell offset from the x-directed
    edges is left in the pattern (order 1) or Q is dropped (order 0)."""
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                     cpml_layers=6, boundary="cpml")
    with pytest.raises(ValueError, match="only order=2"):
        sim.add_current_moment_monitor(
            corner_lo=(6e-3,) * 3, corner_hi=(18e-3,) * 3, block_size=6e-3,
            freqs=FREQS, order=order)


# The Simulation-level models below test refusals, lanes and weight dtypes,
# not patterns, so their frequency list does not weaken them. It is 2 and
# 3 GHz so that their 6 mm blocks through a 12 mm slab (k*|delta| 1.46 at
# 10 GHz) stay inside the block-extent bound (0.44 at 3 GHz).
GUARD_FREQS = np.array([2e9, 3e9])


# The model the guard tests below vary: 2 mm cells, a 24 mm cube of vacuum
# in CPML, a soft Ez source at the centre, and a slab from 6 to 18 mm on
# every axis, i.e. node indices 9..15 with a 6-cell absorber. Its interior
# (the window without the outermost layer) is edge index 10..14 in-plane and
# node planes 10..14; a cell must sit at index 10..13 on every axis for all
# twelve of its edges to be inside.
def _guard_sim(**kw):
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                     cpml_layers=6, **({"boundary": "cpml"} | kw))
    sim.add_source((1.2e-2, 1.2e-2, 1.2e-2), "ez", amplitude_kind="current")
    sim.add_current_moment_monitor(
        corner_lo=(6e-3,) * 3, corner_hi=(18e-3,) * 3, block_size=6e-3,
        freqs=GUARD_FREQS)
    return sim


def _guard_runs(sim, **kw):
    return sim.run(n_steps=4, skip_preflight=True, **kw)


def _block(sim, lo_mm, hi_mm, material="diel"):
    from rfx import Box
    if material == "diel" and "diel" not in sim._materials:
        sim.add_material("diel", eps_r=3.0)
    sim.add(Box(tuple(v * 1e-3 for v in lo_mm), tuple(v * 1e-3 for v in hi_mm)),
            material=material)
    return sim


def test_refuses_a_magnetic_material():
    """``mu_r != 1`` radiates through magnetization current, which the
    electric-current identity does not contain — even inside the slab."""
    sim = _guard_sim()
    sim.add_material("ferrite", eps_r=1.0, mu_r=2.0)
    _block(sim, (10, 10, 10), (14, 14, 14), material="ferrite")
    with pytest.raises(NotImplementedError, match="mu_r != 1"):
        _guard_runs(sim)


def test_refuses_a_magnetic_override_the_model_does_not_declare():
    """The check reads the permeability the run uses, not the declared
    material list: a concrete ``mu_r_override`` on ``forward()`` is seen."""
    sim = _guard_sim()
    mu = np.ones(sim._build_grid().shape, dtype=np.float32)
    mu[12, 12, 12] = 1.5
    with pytest.raises(NotImplementedError, match="mu_r != 1"):
        sim.forward(mu_r_override=jnp.asarray(mu), n_steps=4,
                    skip_preflight=True)


def test_refuses_conducting_domain_faces():
    sim = _guard_sim(boundary="pec")
    with pytest.raises(NotImplementedError, match="PEC domain face"):
        _guard_runs(sim)


@pytest.mark.parametrize("case", [
    "dielectric outside", "dielectric on the outermost layer",
    "conductor sheet crossing the slab", "source outside", "port outside",
    "lumped element outside", "dispersive material outside"])
def test_refuses_a_radiator_the_slab_does_not_hold(case):
    """Current outside the slab radiates and never enters the pattern."""
    from rfx import Box, DebyePole
    sim = _guard_sim()
    if case == "dielectric outside":
        _block(sim, (2, 2, 2), (4, 4, 4))
    elif case == "dielectric on the outermost layer":
        # cells 9..10 on x: cell 9's edges lie on the slab's first layer
        _block(sim, (6, 10, 10), (10, 14, 14))
    elif case == "conductor sheet crossing the slab":
        sim.add(Box((8e-3, 8e-3, 12e-3), (20e-3, 16e-3, 12e-3)), material="pec")
    elif case == "source outside":
        sim.add_source((3e-3, 12e-3, 12e-3), "ez", amplitude_kind="current")
    elif case == "port outside":
        sim.add_port((20e-3, 12e-3, 10e-3), "ez", impedance=50.0)
    elif case == "lumped element outside":
        sim.add_lumped_rlc((12e-3, 20e-3, 12e-3), "ez", R=50.0)
    else:
        sim.add_material("debye", eps_r=1.0,
                         debye_poles=[DebyePole(delta_eps=1.5, tau=1e-11)])
        _block(sim, (2, 2, 2), (4, 4, 4), material="debye")
    with pytest.raises(NotImplementedError, match="outside the slab's interior"):
        _guard_runs(sim)


def test_accepts_the_same_structures_one_cell_further_in():
    """The positive half of the guard: a dielectric block whose cells are all
    at interior indices, a conductor sheet, a port and a lumped element inside
    the slab run and return the moments."""
    from rfx import Box
    sim = _guard_sim()
    _block(sim, (8, 10, 10), (12, 14, 14))
    sim.add(Box((8e-3, 8e-3, 14e-3), (16e-3, 16e-3, 14e-3)), material="pec")
    sim.add_port((14e-3, 12e-3, 10e-3), "ez", impedance=50.0)
    sim.add_lumped_rlc((10e-3, 14e-3, 12e-3), "ez", R=50.0)
    res = _guard_runs(sim)
    assert res.current_moment_data is not None


def test_refuses_a_traced_whole_grid_override():
    """A traced ``eps_override`` could put material anywhere; a design box is
    the way to hand a design variable in (the gradient test above uses one)."""
    sim = _guard_sim()
    shape = sim._build_grid().shape

    def f(a):
        eps = jnp.ones(shape) + a * jnp.zeros(shape).at[12, 12, 12].set(1.0)
        return jnp.sum(sim.forward(eps_override=eps, n_steps=4,
                                   skip_preflight=True).time_series)

    with pytest.raises(NotImplementedError, match="traced whole-grid"):
        jax.grad(f)(1.0)


def test_refuses_a_design_box_outside_the_slab():
    sim = _guard_sim()

    def f(a):
        return jnp.sum(sim.forward(
            design_box=((2e-3, 2e-3, 2e-3), (4e-3, 4e-3, 4e-3)),
            design_eps_override=jnp.ones((2, 2, 2)) * a, n_steps=4,
            skip_preflight=True).time_series)

    with pytest.raises(NotImplementedError, match="design region"):
        jax.grad(f)(2.0)


# The block-extent bound (rfx.current_moments.MAX_K_OFFSET = 0.5): k at the
# highest monitored frequency times the largest distance from a block's centre
# to an edge in it, slab thickness included, must not exceed 0.5. The
# expansion is second order; a board sets the bound because in each block
# column the patch current and its ground return nearly cancel in total moment
# P, so the board radiates through the first moment Q and the dropped term is
# second order in k|delta| relative to what radiates (third order for a dipole
# or wire, where P dominates).
#
# "shape" is the largest dB difference of two patterns, each normalized to its
# own peak, within 20 dB of the reference's peak; dD directivity.
#
# Tutorial patch (examples/tutorials/patch_antenna_demo.py), 2 mm cells,
# 2.0 / 2.4 / 2.8 / 3.4 / 4.0 GHz: one run with one-cell blocks, coarser
# blocks (4-28 mm) built exactly from those moments by the parallel-axis
# shift, shape against the one-cell pattern (the one-cell pattern itself is
# 0.03-0.15 dB off the NTFF box):
#
#   k|delta|    points  shape max dB (where)          |dD| max  >0.2 dB
#   0.10-0.21      6    0.021 (8 mm, 2.0 GHz)           0.003     0
#   0.26-0.46      8    0.162 (12 mm, 2.8 GHz)          0.019     0
#   0.54-0.68      7    0.308 (16 mm, 2.8 GHz; 0.203    0.042     4
#                       at 0.54)
#   0.76-0.96      8    0.715 (28 mm, 2.4 GHz)          0.088     5
#   1.12-1.60      6    1.955 (28 mm, 4.0 GHz)          0.189     6
#
# One block against the direct sum of its current (the test below repeats
# the first two rows):
#
#   current in the block                         k|delta| 0.5   0.7    1.0
#   a sheet over its opposite, 1/12 of the side  0.187          0.368  0.748
#   a square loop (P = 0)                        0.180          0.354  0.726
#   a uniform sheet (P dominates)                0.002          0.007  0.029
#
# Strip dipoles and wires, 1 mm cells, 6 / 8 / 10 GHz, 3000 steps, against
# the NTFF box on the same run (x dipole 14 mm, 1-14 mm blocks; x wire 8+20
# mm, 1-12 mm; z dipoles 6-20 mm and z wires 4+10 / 8+20 mm, 2-4 mm):
#
#   k|delta|    points  shape max dB (where)          |dD| max  |dP| max
#   0.26-0.47     11    0.114 (x wire, 1 mm blocks)     0.015     0.023
#   0.53-0.74     10    0.115 (x wire, 4 mm blocks)     0.017     0.022
#   0.84-0.98      8    0.185 (x dipole, 10 mm blocks)  0.016     0.019
#   1.06-1.25      9    0.420 (x dipole, 10 mm, 1.23)   0.030     0.023
#   1.27-1.96     11    0.903 (x wire, 12 mm, 1.41)     0.087     0.104
#   2.04-3.39      5    6.670 (z wire 8+20, 3.39)       0.351     2.087
#
# Dropped from that ladder: the same wires in a 46 mm domain, whose far end
# lay outside the NTFF box, so the box itself missed current. The tutorial
# patch at 1 mm, 8 mm blocks, 50 frequencies to 4 GHz (k|delta| 0.46) stays
# accepted.


def _block_vs_direct_sum(pos, current, k_offset):
    """Shape dB of one block's second-order pattern against the direct sum of
    its point currents, with k chosen so that k * |delta|max = k_offset."""
    from rfx import radiation_pattern
    from rfx.farfield import FarFieldResult
    th = np.linspace(0.0, np.pi, 61)
    ph = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    centre = pos.mean(axis=0)
    d = pos - centre
    k = k_offset / float(np.sqrt((d ** 2).sum(axis=1)).max())
    st, ct, sp, cp = np.sin(th)[:, None], np.cos(th)[:, None], np.sin(ph), np.cos(ph)
    one = np.ones_like(st * cp)
    r_hat = np.stack([st * cp, st * sp, ct * one], -1).reshape(-1, 3)
    t_hat = np.stack([ct * cp, ct * sp, -st * one], -1).reshape(-1, 3)
    p_hat = np.stack([-sp * one, cp * one, 0.0 * one], -1).reshape(-1, 3)
    n_vec = np.exp(1j * k * (r_hat @ pos.T)) @ current
    ref = FarFieldResult(E_theta=(n_vec * t_hat).sum(1).reshape(1, th.size, ph.size),
                         E_phi=(n_vec * p_hat).sum(1).reshape(1, th.size, ph.size),
                         theta=th, phi=ph, freqs=np.array([1.0]))
    P = current.sum(0)
    Q = np.einsum("na,nb->ab", d, current)
    T = np.einsum("na,nb,nc->abc", d, d, current)
    a, b = block_far_field_np(th, ph, centre[None], P[None], Q[None], T[None], k, 2)
    got = ref._replace(E_theta=np.asarray(a)[None], E_phi=np.asarray(b)[None])
    ra, rb = radiation_pattern(got)[0], radiation_pattern(ref)[0]
    return float(np.max(np.abs(ra - rb)[rb >= -20.0]))


def test_the_block_extent_bound_holds_where_the_total_moment_cancels():
    """At k*|delta| = MAX_K_OFFSET, a block whose total moment cancels (a
    current sheet over its opposite, as a patch column over its ground; a
    loop) still reads within 0.2 dB of its direct sum. Measured: 0.187 and
    0.180 dB at 0.5; 0.748 and 0.726 dB at 1.0."""
    from rfx.current_moments import MAX_K_OFFSET
    g = np.linspace(-0.5, 0.5, 41)[:-1] + 1.0 / 80.0
    X, Y = (v.ravel() for v in np.meshgrid(g, g, indexing="ij"))
    h = 1.0 / 12.0
    pair = np.concatenate([np.stack([X, Y, np.full(X.size, h / 2)], 1),
                           np.stack([X, Y, np.full(X.size, -h / 2)], 1)])
    pair_j = np.zeros(pair.shape, complex)
    pair_j[:X.size, 0], pair_j[X.size:, 0] = 1.0, -1.0
    s = np.linspace(-0.5, 0.5, 200)
    z = 0.0 * s
    sides = [(np.c_[s, z - 0.5, z], (1, 0, 0)), (np.c_[z + 0.5, s, z], (0, 1, 0)),
             (np.c_[-s, z + 0.5, z], (-1, 0, 0)), (np.c_[z - 0.5, -s, z], (0, -1, 0))]
    loop = np.concatenate([p for p, _ in sides])
    loop_j = np.concatenate([np.tile(v, (s.size, 1)) for _, v in sides]).astype(complex)
    for pos, cur in ((pair, pair_j), (loop, loop_j)):
        assert _block_vs_direct_sum(pos, cur, MAX_K_OFFSET) <= 0.2
        # the same block at twice the bound is well past it
        assert _block_vs_direct_sum(pos, cur, 2.0 * MAX_K_OFFSET) > 0.5


def test_refuses_blocks_too_large_for_the_frequency():
    """The guard model's slab is 6 mm blocks through 12 mm of thickness:
    k*|delta| = 1.46 at 10 GHz and 0.73 at 5 GHz are refused, 0.44 at 3 GHz
    is not."""
    from rfx import Simulation
    for freqs, refused in ((FREQS, True), (np.array([5e9]), True),
                           (GUARD_FREQS, False)):
        sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                         cpml_layers=6, boundary="cpml")
        sim.add_source((1.2e-2,) * 3, "ez", amplitude_kind="current")
        sim.add_current_moment_monitor(
            corner_lo=(6e-3,) * 3, corner_hi=(18e-3,) * 3, block_size=6e-3,
            freqs=freqs)
        if refused:
            with pytest.raises(NotImplementedError, match="blocks are too large"):
                _guard_runs(sim)
        else:
            assert _guard_runs(sim).current_moment_data is not None


def test_refuses_a_slab_too_thick_even_with_small_blocks():
    """The slab thickness is part of every block: 1 mm blocks do not help a
    20 mm-thick slab at 10 GHz (k*|delta| about 2.1)."""
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=1e-3,
                     cpml_layers=6, boundary="cpml")
    sim.add_source((1.2e-2,) * 3, "ez", amplitude_kind="current")
    sim.add_current_moment_monitor(
        corner_lo=(10e-3, 10e-3, 2e-3), corner_hi=(14e-3, 14e-3, 22e-3),
        block_size=1e-3, freqs=np.array([10e9]))
    with pytest.raises(NotImplementedError, match="thinner slab"):
        _guard_runs(sim)


@pytest.mark.parametrize("block_size", [0.0, -2e-3])
def test_declaration_refuses_a_non_positive_block_size(block_size):
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                     cpml_layers=6, boundary="cpml")
    with pytest.raises(ValueError, match="must be positive"):
        sim.add_current_moment_monitor(
            corner_lo=(6e-3,) * 3, corner_hi=(18e-3,) * 3,
            block_size=block_size, freqs=GUARD_FREQS)


def _pec_box(sim):
    from rfx import Box
    sim.add(Box((10e-3, 10e-3, 14e-3), (14e-3, 14e-3, 16e-3)), material="pec")
    return sim


def test_refuses_kottke_pec_and_accepts_the_other_conductor_updates():
    """``subpixel_smoothing="kottke_pec"`` zeroes H inside the cells it
    treats as PEC, a magnetic surface current the monitor never reads; the
    staircase and the E-side ``conformal_pec=True`` updates leave H alone."""
    with pytest.raises(NotImplementedError, match="zeroes H"):
        _guard_runs(_pec_box(_guard_sim()), subpixel_smoothing="kottke_pec")
    for kw in ({}, {"conformal_pec": True}):
        assert _guard_runs(_pec_box(_guard_sim()), **kw).current_moment_data is not None


def test_refuses_the_kottke_occupancy_lane(monkeypatch):
    sim = _guard_sim()
    occ = np.zeros(sim._build_grid().shape, dtype=np.float32)
    occ[12, 12, 12] = 1.0
    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "1")
    with pytest.raises(NotImplementedError, match="zeroes H"):
        sim.forward(pec_occupancy_override=jnp.asarray(occ), n_steps=4,
                    skip_preflight=True)


def _topology(region_lo_mm, region_hi_mm, material_fg):
    from rfx.topology import TopologyDesignRegion, topology_optimize
    sim = _guard_sim()
    sim.add_material("diel6", eps_r=6.0)
    region = TopologyDesignRegion(
        corner_lo=tuple(v * 1e-3 for v in region_lo_mm),
        corner_hi=tuple(v * 1e-3 for v in region_hi_mm),
        material_bg="air", material_fg=material_fg)
    theta, phi = np.array([0.5, 1.5]), np.array([0.0, 1.0])

    def objective(result):
        ff = current_moment_far_field(result, theta, phi)
        return -jnp.sum(jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2)

    return topology_optimize(sim, region, objective, n_iterations=1,
                             verbose=False, skip_preflight=True)


@pytest.mark.parametrize("material_fg", ["diel6", "pec"])
def test_topology_design_region_outside_the_slab_is_refused(material_fg):
    """``topology_optimize`` hands the solve whole-grid traced design arrays;
    it gives the guard the design region's bounds instead."""
    with pytest.raises(NotImplementedError, match="topology design region"):
        _topology((18, 8, 8), (22, 16, 16), material_fg)


@pytest.mark.slow
def test_topology_design_region_inside_the_slab_runs():
    assert _topology((10, 10, 10), (14, 14, 14), "diel6") is not None


# One refused and one accepted model per branch of the guard. Each helper
# puts the thing either outside the slab (refused) or inside it (accepted).
def _surface_impedance_sheet(sim, inside):
    from rfx import Box
    z = 12e-3 if inside else 20e-3
    sim.add_thin_conductor(Box((10e-3, 10e-3, z), (14e-3, 14e-3, z)),
                           sigma_bulk=1e5, thickness=1e-5,
                           surface_impedance_f0=4e9)


def _kerr(sim, inside):
    sim.add_material("kerr", eps_r=1.0, chi3=1e-2)
    lo, hi = ((10, 10, 10), (14, 14, 14)) if inside else ((2, 2, 2), (4, 4, 4))
    _block(sim, lo, hi, material="kerr")


def _pec_wire(sim, inside):
    from rfx.geometry.csg import PolylineWire
    y = 12e-3 if inside else 3e-3
    sim.add(PolylineWire(points=((10e-3, y, 12e-3), (14e-3, y, 12e-3)),
                         radius=1e-4), material="pec")


GUARD_BRANCHES = {
    "surface-impedance sheet": (_surface_impedance_sheet, "surface-impedance sheet"),
    "Kerr material": (_kerr, "Kerr material"),
    "PEC wire": (_pec_wire, "a conductor"),
}


@pytest.mark.parametrize("inside", [False, True], ids=["outside", "inside"])
@pytest.mark.parametrize("branch", list(GUARD_BRANCHES))
def test_guard_branch_refuses_outside_and_accepts_inside(branch, inside):
    add, message = GUARD_BRANCHES[branch]
    sim = _guard_sim()
    add(sim, inside)
    if inside:
        assert _guard_runs(sim).current_moment_data is not None
    else:
        with pytest.raises(NotImplementedError, match=message):
            _guard_runs(sim)


def test_refuses_a_pmc_face_and_accepts_absorbing_faces():
    from rfx.boundaries.spec import Boundary, BoundarySpec
    spec = BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                        y=Boundary(lo="cpml", hi="cpml"),
                        z=Boundary(lo="pmc", hi="cpml"))
    with pytest.raises(NotImplementedError, match="PMC face"):
        _guard_runs(_guard_sim(boundary=spec))
    assert _guard_runs(_guard_sim()).current_moment_data is not None


def test_refuses_a_plane_port_and_accepts_a_lumped_port():
    from rfx import Simulation

    def bare():
        sim = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                         cpml_layers=6, boundary="cpml")
        sim.add_current_moment_monitor(
            corner_lo=(6e-3,) * 3, corner_hi=(18e-3,) * 3, block_size=6e-3,
            freqs=GUARD_FREQS)
        return sim

    sim = bare()
    sim.add_waveguide_port(4e-3, y_range=(4e-3, 20e-3), z_range=(8e-3, 16e-3),
                           freqs=GUARD_FREQS)
    with pytest.raises(NotImplementedError, match="waveguide port"):
        _guard_runs(sim)
    sim = bare()
    sim.add_port((12e-3, 12e-3, 10e-3), "ez", impedance=50.0)
    assert _guard_runs(sim).current_moment_data is not None


def _forward_with(sim, **kw):
    return sim.forward(n_steps=4, skip_preflight=True, **kw)


@pytest.mark.parametrize("inside", [False, True], ids=["outside", "inside"])
@pytest.mark.parametrize("override", ["pec_mask_override", "pec_occupancy_override"])
def test_concrete_pec_overrides_are_measured(override, inside):
    """A concrete ``pec_mask_override`` / ``pec_occupancy_override`` is part
    of the conductor the run solves, and the guard reads it."""
    sim = _guard_sim()
    shape = sim._build_grid().shape
    cell = (12, 12, 12) if inside else (5, 12, 12)
    if override == "pec_mask_override":
        arr = np.zeros(shape, dtype=bool)
        arr[cell] = True
    else:
        arr = np.zeros(shape, dtype=np.float32)
        arr[cell] = 1.0
    if inside:
        assert _forward_with(sim, **{override: jnp.asarray(arr)}).current_moment_data is not None
    else:
        with pytest.raises(NotImplementedError, match="a conductor"):
            _forward_with(sim, **{override: jnp.asarray(arr)})


@pytest.mark.parametrize("inside", [False, True], ids=["outside", "inside"])
def test_design_occupancy_box_is_checked_by_its_bounds(inside):
    sim = _guard_sim()
    lo, hi = ((10e-3, 10e-3, 10e-3), (12e-3, 12e-3, 12e-3)) if inside else \
        ((2e-3, 2e-3, 2e-3), (4e-3, 4e-3, 4e-3))

    def f(a):
        return jnp.sum(_forward_with(
            sim, design_box=(lo, hi),
            design_occupancy_override=jnp.ones((2, 2, 2)) * a,
            checkpoint=False).current_moment_data[0].real)

    if inside:
        assert np.isfinite(float(jax.grad(f)(0.5)))
    else:
        with pytest.raises(NotImplementedError, match="design occupancy box"):
            jax.grad(f)(0.5)


def test_declaration_rejects_an_unknown_keyword():
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 2.4e-2), dx=DX,
                     cpml_layers=6, boundary="cpml")
    with pytest.raises(TypeError):
        sim.add_current_moment_monitor(
            corner_lo=(4e-3, 4e-3, 4e-3), corner_hi=(1.6e-2, 1.6e-2, 1.2e-2),
            block_size=6e-3, freqs=FREQS, curl_signs=(1,) * 6)


# ---------------------------------------------------------------------------
# 11. The lanes that do not accumulate the monitor refuse it
# ---------------------------------------------------------------------------

def _sim_with_monitor(**kw):
    from rfx import Simulation
    sim = Simulation(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 2.4e-2), dx=DX,
                     cpml_layers=6, boundary="cpml", **kw)
    sim.add_source(position=(1.2e-2, 1.2e-2, 1.2e-2), component="ez")
    sim.add_current_moment_monitor(
        corner_lo=(0.6e-2, 0.6e-2, 0.6e-2), corner_hi=(1.8e-2, 1.8e-2, 1.8e-2),
        block_size=6e-3, freqs=GUARD_FREQS)
    return sim


@pytest.mark.parametrize("lane", ["subgridded", "disjoint", "distributed",
                                  "distributed_v2", "vmap"])
def test_lanes_that_do_not_accumulate_it_refuse_it(lane):
    """A declared monitor must not come back silently empty.

    Five entry points never read the declaration; each says so. The two
    distributed runners refuse AFTER their single-device fallbacks, because
    those return through ``sim.run()``, which does accumulate it.
    """
    sim = _sim_with_monitor()
    with pytest.raises(NotImplementedError, match="add_current_moment_monitor"):
        if lane == "subgridded":
            from rfx.runners.subgridded import run_subgridded_path
            run_subgridded_path(sim, None, None, None, 4)
        elif lane == "disjoint":
            from rfx.runners.disjoint import run_disjoint_stage2_path
            run_disjoint_stage2_path(sim, None, 4)
        elif lane == "distributed":
            from rfx.runners.distributed import run_distributed
            run_distributed(sim, n_steps=4)
        elif lane == "distributed_v2":
            from rfx.runners.distributed_v2 import run_distributed
            run_distributed(sim, n_steps=4)
        else:
            from rfx.vmap_sweep import vmap_material_sweep
            vmap_material_sweep(sim, "eps_r", [1.0, 2.0], n_steps=4)


def test_a_multi_device_run_refuses_it():
    """``run(devices=...)`` across two devices goes to the sharded runner."""
    if len(jax.devices()) < 2:
        pytest.skip("needs two JAX devices (the repo conftest makes two CPUs)")
    sim = _sim_with_monitor()
    # The runner's own refusal, before any step: the backstop in run() would
    # also raise, but only after the whole run.
    with pytest.raises(NotImplementedError,
                       match="not supported on the distributed \\(v2\\) runner"):
        sim.run(n_steps=4, devices=jax.devices()[:2], skip_preflight=True)


def test_the_distributed_nu_forward_lane_refuses_it():
    """Reached through the private entry, with no devices needed.

    The refusal sits beside this lane's existing ``add_flux_monitor()`` and
    ``add_dft_plane_probe()`` refusals and fires before any sharding.
    """
    dz = np.full(10, DX)
    sim = _sim_with_monitor(dz_profile=dz)
    with pytest.raises(NotImplementedError, match="add_current_moment_monitor"):
        sim._forward_distributed_nonuniform_from_materials(n_steps=4)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_the_adi_lane_refuses_it(entry):
    """The ADI update is not the Yee Ampere step, so the identity the monitor
    rests on does not hold there at all."""
    sim = _sim_with_monitor(solver="adi")
    # The lane's own refusal, before any step (the backstop would raise only
    # after the whole run, with a different message).
    with pytest.raises(NotImplementedError,
                       match="not supported on the ADI lane"):
        if entry == "run":
            sim.run(n_steps=4)
        else:
            sim.forward(n_steps=4)


def test_a_declared_monitor_cannot_come_back_empty_from_any_lane():
    """The backstop that does not need the list of lanes.

    Per-lane refusals are only as complete as the last runner someone added.
    The check where ``run()`` and ``forward()`` hand their result back
    refuses an empty field whatever produced it; here a lane is faked by
    emptying the field of a real result, which is what an unlisted runner
    would return.
    """
    from rfx.current_moments import require_accumulated_current_moments

    sim = _sim_with_monitor()
    res = sim.run(n_steps=4)
    assert res.current_moment_data is not None
    require_accumulated_current_moments(sim, res, "run")        # filled: silent

    emptied = res._replace(current_moment_data=None)
    with pytest.raises(NotImplementedError, match="returned no block current"):
        require_accumulated_current_moments(sim, emptied, "run")

    # No monitor declared: an empty field is the normal case and stays silent.
    from rfx import Simulation
    plain = Simulation(freq_max=1.2e10, domain=(2.4e-2,) * 3, dx=DX,
                       cpml_layers=6, boundary="cpml")
    require_accumulated_current_moments(plain, emptied, "run")


def test_every_user_facing_return_carries_the_backstop():
    """Static half of the same guarantee: each place ``run()`` / ``forward()``
    hands a result back calls the backstop, so a new return site cannot skip
    it without this count changing."""
    import re
    text = (REPO_ROOT / "rfx/api/_execute.py").read_text()
    returns = len(re.findall(
        r'_warn_if_nonfinite_result\(_res, context="(?:run|forward)"\)', text))
    guards = len(re.findall(
        r'require_accumulated_current_moments\(self, _res, "(?:run|forward)"\)',
        text))
    assert returns == guards == 6, (returns, guards)


def test_the_traceable_moment_unpack_matches_the_host_one():
    """``moments_to_PQT_jax`` is what a differentiable objective reads the
    accumulator through; the host ``moments_to_PQT`` is what the NumPy
    pattern is built with. Same index order, slot for slot, on an accumulator
    with no symmetry to hide a transposed axis behind."""
    from rfx.current_moments import moments_to_PQT_jax

    case = _reference("uniform")
    m = case["monitor"]
    rng = np.random.default_rng(7)
    shape = np.asarray(case["acc"]).shape
    acc = (rng.standard_normal(shape)
           + 1j * rng.standard_normal(shape)).astype(np.complex64)
    host = moments_to_PQT(acc, m)
    traced = moments_to_PQT_jax(acc, m)
    for name, a, b in zip("PQT", host, traced):
        assert (a is None) == (b is None), name
        if a is not None:
            assert np.asarray(b).shape == np.asarray(a).shape, name
            np.testing.assert_array_equal(np.asarray(b), np.asarray(a),
                                          err_msg=name)


# ---------------------------------------------------------------------------
# 12. The weights are stored at the field's precision, not pinned
# ---------------------------------------------------------------------------

def test_weight_dtype_follows_the_runs_precision():
    """float32 fields keep float32 weights; float64 fields get float64 ones.

    The weights multiply the current inside the spatial reduction, so a
    float32 weight quantizes the block moments at float32 however wide the
    accumulator is; with the weights pinned to float32 the float64 gradient
    check in this file misses its bar. The float32 branch is pinned too:
    that is the path every existing run takes.
    """
    from rfx import Simulation
    from rfx.current_moments import monitor_for_simulation, weight_dtype_for

    def _sim(precision):
        s = Simulation(freq_max=1.2e10, domain=(2.4e-2, 2.4e-2, 2.4e-2),
                       dx=DX, cpml_layers=6, boundary="cpml",
                       precision=precision)
        s.add_current_moment_monitor(
            corner_lo=(4e-3, 4e-3, 4e-3), corner_hi=(1.6e-2, 1.6e-2, 1.2e-2),
            block_size=6e-3, freqs=GUARD_FREQS)
        return s

    s32 = _sim("float32")
    assert weight_dtype_for(s32) == np.float32
    m32 = monitor_for_simulation(s32, s32._build_grid())
    assert m32.w_ex.dtype == jnp.float32
    assert m32.inv_dx_e.dtype == jnp.float32

    s64 = _sim("float64")
    assert weight_dtype_for(s64) == np.float64
    with enable_x64():
        m64 = monitor_for_simulation(s64, s64._build_grid())
        assert m64.w_ex.dtype == jnp.float64, m64.w_ex.dtype
        assert m64.inv_dx_e.dtype == jnp.float64, m64.inv_dx_e.dtype
        # Same numbers, wider storage, normalised by the largest weight in
        # the slot: the first-order slots change sign, so an element-wise
        # ratio blows up where a weight passes through zero.
        a = np.asarray(m64.w_ex, dtype=np.float64)
        b = np.asarray(m32.w_ex, dtype=np.float64)
        scale = np.abs(a).max(axis=(1, 2, 3), keepdims=True)
        assert float(np.max(np.abs(a - b) / scale)) < 1e-6
