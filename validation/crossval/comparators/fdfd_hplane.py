"""Independent 2-D H-plane FDFD comparator for inductive-iris structures.

Solves the scalar Helmholtz BVP for E_y(x,z) on a finite-difference grid. It
shares NO formulation with mode matching: the port condition is the EXACT
discrete transparent condition (a one-sided ghost relation built from the
discrete transverse eigenbasis and the discrete longitudinal propagation
constant), so an empty guide reflects at machine precision and the only error
left is the interior/staircase discretization, which is Richardson-extrapolated.

    (Dxx + Dzz + k^2) E = 0,   E = 0 on PEC (walls + irises)
    ghost at z=0 :  E_{-1}    = Q E_0 + (e^{+g1 h} - e^{-g1 h}) phi1
    ghost at z=L :  E_{nz+1}  = Q E_{nz}
    Q = sum_n e^{-gt_n h} phi_n phi_n^T h,  cosh(gt_n h) = 1 - h^2 (lam_n + k^2)/2

    S11 = <E_0, phi1> - 1 ,  S21 = <E_nz, phi1> e^{+gt_1 nz h}
    (MAGNITUDES ONLY are validated. The phases are referenced to the two
    domain ends, so S11's phase rotates with the margin -- measured: the
    complex S11 moves by 1.16 between margin 1 and 20 while |S11| is
    invariant. Do not consume the phases without re-deriving the planes.)

ORIGIN. Ported from the solver written by the independent reviewer of PR #480
(crossval case 18). That solver is the artifact that caught a 4-6x envelope error
the mode-matching oracle could not have caught -- because the mode-matching
oracle was RIGHT and the defect was in the FDTD setup geometry, which only a
same-geometry independent solve exposes.

CONDITIONS OF USE, carried from the handoff note (its own instruction was
verbatim; the measured record in condition 2 is updated to this case's
five-iris numbers, and that substitution is disclosed here rather than
silent). These are not footnotes:

 1. 2-D H-PLANE ONLY. Scalar Helmholtz for E_y(x,z), TE_n0, no variation along b.
    Valid for inductive irises, H-plane septa, width steps, H-plane bends. NOT
    valid for capacitive/E-plane obstacles, posts, or anything varying along b --
    pointed at those it returns a confident wrong number with clean unitarity,
    which is the worst failure mode. Hard scope fence.
 2. NOT A SINGLE-RUN ORACLE. It converges FIRST order (node-Dirichlet staircase,
    the same convention as rfx, which is why it reproduces rfx under both fin
    conventions). One number at one mesh is useless: Richardson over >= 2
    levels is mandatory, and the original review used three levels and
    confirmed the two extrapolation estimates agreed to ~1e-3 before trusting
    either. See the case fixture's fdfd_formulation_independent block for this
    case's measured per-level record.
 3. REQUIRES GRID-EXACT GEOMETRY AT EVERY REFINEMENT LEVEL. If one level snaps a
    dimension, the O(h) coefficient changes between levels and Richardson
    silently degrades: plausible number, no warning. `solve` asserts exactness
    per level rather than assuming it; callers should express geometry in BASE
    CELLS and refine by INTEGER factors, which makes exactness structural.
 4. SELF-TESTS ARE GATES, NOT OPTIONAL CHECKS. The original had a real bug -- a
    missing `/h` in the discrete propagation constant -- and the empty-guide
    transparency test is what caught it (|S11| came back 1.0 instead of 1e-14).
    The handoff's spec: empty-guide |S11| <= 1e-12 and |S21| = 1 to 1e-12
    (measured headroom: 5e-14 at r=1), and unitarity on EVERY evaluation.
    `self_test` runs both and RAISES; callers doing sweeps must keep the
    per-evaluation unitarity check as well, as the case script does.
    BUT `self_test`'s `unitarity` is a ROUNDOFF REALIZATION, not a property of
    the method: on the gated configuration it moves 1.25 decades across the
    four mathematically equivalent `permc_spec` orderings through `splu`, and
    1.88 through `spsolve` (#884). Gate
    `refined_unitarity` instead -- iterative refinement with an exactly
    accumulated residual leaves ~5e-14, which is the discretization's own
    unitarity and is build-independent. `self_test`'s number is still worth
    recording; it is not worth comparing across builds.
 5. FORMULATION-INDEPENDENT, NOT TOTALLY INDEPENDENT. It shares one element with
    mode matching: the TE_n0 modal basis at the ports. But that is evaluated in
    uniform guide far from the discontinuities, where the expansion is exact
    rather than an approximation, so it is not a shared APPROXIMATION. The
    aperture treatment, the discretization and the solve are entirely different.
 6. COST. The DtN blocks are dense nxi x nxi; memory ~ nx^2. Comfortable to
    nx ~ 500. Comparator, not a sweep engine.
 7. DIAGNOSTIC, NEVER CLAIMS-BEARING. It exercises no rfx code path and is not
    differentiable -- the source of its comparator value and also the reason it
    can never substitute for an rfx test.

Zero rfx dependency by design: numpy and scipy only.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

C0 = 299792458.0


def discrete_gamma(lam, k, h):
    """Discrete longitudinal propagation constant, outgoing/decaying branch."""
    mu = np.asarray(1.0 - h * h * (lam + k * k) / 2.0, dtype=complex)
    g = np.arccosh(mu)
    g = np.where(np.real(g) < 0, -g, g)
    prop = np.abs(np.imag(mu)) < 1e-14
    gp = 1j * np.arccos(np.clip(np.real(mu), -1.0, 1.0))
    g = np.where(prop & (np.real(mu) <= 1.0), gp, g)
    return g / h


def _assemble(a, freq, base_cells, refinement, apertures_cells, cavities_cells,
              thickness_cells, margin_cells, empty=False):
    """Assemble (A, rhs) and the port context for one solve.

    Split out of `solve` (unchanged, line for line) only so that the refined
    unitarity witness can reuse the SAME matrix and the SAME factorization
    rather than re-deriving either. `solve` remains the one entry point that
    calls the linear solver, so nothing about the committed curves moves.
    """
    r = int(refinement)
    assert r >= 1 and r == refinement, ("refinement must be a positive integer "
                                        "or condition 3 is violated", refinement)
    nx = base_cells * r
    h = a / nx
    k = 2 * np.pi * freq / C0

    tc = thickness_cells * r + 1          # metal NODES; electrical = tc - 1
    span = len(apertures_cells) * (tc - 1) + sum(c * r for c in cavities_cells)
    nz = span + 2 * margin_cells * r
    ix = np.arange(1, nx)
    nxi, nzi = len(ix), nz + 1

    metal = np.zeros((nxi, nzi), dtype=bool)
    if not empty:
        z = margin_cells * r
        for i, d_c in enumerate(apertures_cells):
            # condition 3: exact at THIS level, asserted not assumed
            assert (nx - d_c * r) % 2 == 0, (
                "aperture is not realizable symmetrically at this refinement; "
                "the geometry would be re-snapped and Richardson would degrade",
                d_c, r)
            fc = (nx - d_c * r) // 2
            # bounding zeroed planes at ix = fc and ix = nx - fc, so the
            # electrical aperture is exactly (nx - 2*fc) = d_c*r cells at THIS
            # level, with d_c*r - 1 open interior nodes -- node-for-node the rfx
            # raster at r = 1. An earlier revision used strict inequalities,
            # which left the planes d_c*r + 2 cells apart at every level: a
            # first-order bias that Richardson cancelled (the extrapolated
            # numbers were right) but that made this docstring's convention
            # claim false at finite h. Found by an independent port review.
            assert nx - 2 * fc == d_c * r
            metal[(ix <= fc) | (ix >= nx - fc), z:z + tc] = True
            if i < len(cavities_cells):
                z += (tc - 1) + cavities_cells[i] * r
        assert z + tc - 1 == margin_cells * r + span, ("span mismatch", z, span)

    ns = np.arange(1, nx)
    phi = np.sqrt(2.0 / a) * np.sin(np.pi * np.outer(ns, ix) / nx)
    lam = (2 * np.cos(np.pi * ns / nx) - 2) / h ** 2
    gt = discrete_gamma(lam, k, h)
    Q = (phi.T * np.exp(-gt * h)) @ phi * h

    Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nxi, nxi)) / h ** 2
    Dzz = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nzi, nzi)) / h ** 2
    N = nxi * nzi
    A = (sp.kron(Dxx, sp.eye(nzi)) + sp.kron(sp.eye(nxi), Dzz)
         + k * k * sp.eye(N))
    P0 = sp.csr_matrix(([1.0], ([0], [0])), shape=(nzi, nzi))
    PL = sp.csr_matrix(([1.0], ([nzi - 1], [nzi - 1])), shape=(nzi, nzi))
    Qs = sp.csr_matrix(Q)
    A = A + (sp.kron(Qs, P0) + sp.kron(Qs, PL)) / h ** 2

    rhs = np.zeros((nxi, nzi), dtype=complex)
    rhs[:, 0] = -(np.exp(gt[0] * h) - np.exp(-gt[0] * h)) * phi[0] / h ** 2

    A = A.tolil()
    midx = np.where(metal.reshape(-1))[0]
    for row in midx:
        A.rows[row] = [row]
        A.data[row] = [1.0]
    rhs = rhs.reshape(-1)
    rhs[midx] = 0.0

    ctx = dict(phi=phi, gt=gt, nz=nz, nxi=nxi, nzi=nzi, h=h,
               info=dict(nx=nx, nz=nz, unknowns=N, h=h, metal_nodes_z=tc))
    return A.tocsc(), rhs, ctx


def _ports(x, ctx):
    """(S11, S21) from a solution vector, exactly as `solve` reads them off."""
    E = x.reshape(ctx["nxi"], ctx["nzi"])
    phi, gt, h, nz = ctx["phi"], ctx["gt"], ctx["h"], ctx["nz"]
    s11 = np.sum(phi[0] * E[:, 0]) * h - 1.0
    s21 = np.sum(phi[0] * E[:, -1]) * h * np.exp(gt[0] * nz * h)
    return complex(s11), complex(s21)


def _unitarity(x, ctx):
    """The lossless two-port power identity |S11|^2 + |S21|^2 - 1."""
    s11, s21 = _ports(x, ctx)
    return abs(abs(s11) ** 2 + abs(s21) ** 2 - 1.0)


def solve(a, freq, base_cells, refinement, apertures_cells, cavities_cells,
          thickness_cells, margin_cells, empty=False):
    """One solve. Geometry is given in BASE cells; refinement is an integer.

    apertures_cells / cavities_cells / thickness_cells are ELECTRICAL, i.e. the
    distance between bounding zeroed node planes, matching the convention the
    rfx rasterizer realizes.
    """
    A, rhs, ctx = _assemble(a, freq, base_cells, refinement, apertures_cells,
                            cavities_cells, thickness_cells, margin_cells,
                            empty=empty)
    x = spl.spsolve(A, rhs)
    s11, s21 = _ports(x, ctx)
    return s11, s21, ctx["info"]


def self_test(a, freq, base_cells, refinement, apertures_cells, cavities_cells,
              thickness_cells, margin_cells, empty_tol=1e-12, unitary_tol=1e-6):
    """Condition 4: GATES. Raises rather than returning a verdict."""
    e11, e21, info = solve(a, freq, base_cells, refinement, apertures_cells,
                           cavities_cells, thickness_cells, margin_cells,
                           empty=True)
    s11 = e11
    if abs(s11) > empty_tol:
        raise AssertionError(
            f"empty-guide |S11| = {abs(s11):.3e} > {empty_tol:g}: the discrete "
            "transparent condition is not transparent. This is the gate that "
            "caught a missing /h in the propagation constant.")
    if abs(abs(e21) - 1.0) > empty_tol:
        raise AssertionError(f"empty-guide |S21| = {abs(e21):.12f}, expected 1")
    f11, f21, _ = solve(a, freq, base_cells, refinement, apertures_cells,
                        cavities_cells, thickness_cells, margin_cells)
    u = abs(abs(f11) ** 2 + abs(f21) ** 2 - 1.0)
    if u > unitary_tol:
        raise AssertionError(f"lossless unitarity violated by {u:.3e}")
    return dict(empty_s11=abs(e11), empty_s21=abs(e21), unitarity=u, **info)


def _exact_residual(a_csr, x, b):
    """r = b - A x, each row's inner product accumulated EXACTLY.

    The products are rounded once (they are ordinary float64 multiplies); the
    summation over a row is exact (`math.fsum`). That matters because the
    residual of a cond ~ 1e12 system is five decades of cancellation: evaluated
    in plain float64 the residual is itself dominated by roundoff, and
    refinement then converges to the accumulation's floor rather than to the
    solution. This is the standard requirement for iterative refinement to
    recover more than backward stability (Wilkinson); here it is what makes the
    refined unitarity a build-independent number instead of a second sample of
    the factorization's luck.
    """
    ip = a_csr.indptr.tolist()
    prod = a_csr.data * x[a_csr.indices]         # rounded products, vectorized
    pr, pi = prod.real.tolist(), prod.imag.tolist()
    fsum = math.fsum
    ar = [fsum(pr[ip[i]:ip[i + 1]]) for i in range(len(ip) - 1)]
    ai = [fsum(pi[ip[i]:ip[i + 1]]) for i in range(len(ip) - 1)]
    return b - (np.asarray(ar) + 1j * np.asarray(ai))


def refined_unitarity(a, freq, base_cells, refinement, apertures_cells,
                      cavities_cells, thickness_cells, margin_cells, steps=2,
                      permc_spec=None):
    """The unitarity of the METHOD, separated from the factorization's roundoff.

    Issue #884. `self_test`'s `unitarity` is the lossless power identity
    evaluated on `spsolve`'s answer. On this problem (cond_1(A) ~ 1e12,
    backward error ~ 3 eps) that number is dominated by LU roundoff, so it has
    no build-independent value: the four `permc_spec` orderings -- which solve
    the SAME system and are mathematically identical -- spread it over 1.25
    decades on one machine through `splu` (1.88 through `spsolve`), wider than
    the 1.03-decade Python 3.10 -> 3.11 gap that was reported as a regression.

    Iterative refinement on the same LU factor with an exactly accumulated
    residual removes that: what is left, ~5e-14, is the discretization's own
    unitarity, at the arithmetic floor of the two length-(nx-1) modal inner
    products the witness is built from, and it is stable across orderings,
    library versions and platforms.

    Returns `unitarity_raw` (the quantity `self_test` records),
    `unitarity_refined` (min over the refinement steps -- refinement stagnates
    at the residual-evaluation floor and then oscillates within it, so the
    minimum is the converged value and the rule is fixed here rather than
    chosen per run) and the per-step list. Cost is one `splu` plus `steps`
    triangular solves and residuals; no second assembly and no second
    factorization.
    """
    steps = int(steps)
    assert steps >= 1, ("the refined witness needs at least one refinement "
                        "step; there is nothing to take a minimum over", steps)
    A, rhs, ctx = _assemble(a, freq, base_cells, refinement, apertures_cells,
                            cavities_cells, thickness_cells, margin_cells)
    lu = spl.splu(A, permc_spec=permc_spec)
    x = lu.solve(rhs)
    u_raw = _unitarity(x, ctx)
    a_csr = A.tocsr()
    per_step = []
    for _ in range(steps):
        x = x + lu.solve(_exact_residual(a_csr, x, rhs))
        per_step.append(_unitarity(x, ctx))
    return dict(unitarity_raw=u_raw, unitarity_refined=min(per_step),
                unitarity_steps=tuple(per_step), **ctx["info"])


def richardson_first_order(value_coarse, r_coarse, value_fine, r_fine):
    """Condition 2. h ∝ 1/r for integer refinement, so f(r) = f_exact + c/r."""
    assert r_fine > r_coarse >= 1
    return (r_fine * value_fine - r_coarse * value_coarse) / (r_fine - r_coarse)
