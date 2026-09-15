"""rfx.fdfd.yee3d: operator assembly against scipy, PML reflection of the
TE10 mode, the inductive iris against the 2-D H-plane solver, energy
conservation, and the gradients w.r.t. permittivity and a grid step."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from rfx.fdfd import hplane
from rfx.fdfd import yee3d as y
from tests._x64_compat import enable_x64

A_WR90 = 22.86e-3
F0 = 10e9
NX, NY = 24, 3                 # 24 cells across a (same as the hplane gate), a few y cells
H = A_WR90 / NX
NPML = 10
NZ = 56
K_SRC, K_REF1, Z0, K_REF2 = 13, 20, 30, 41   # iris on z nodes 30, 31 (cell 30)
APERTURE = 12                  # base cells, hplane convention


def _fd4(f, x0, d):
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def _guide(npml=NPML, nz=NZ):
    spec = y.Yee3DSpec(nx=NX, ny=NY, nz=nz, pml=(0, 0, 0, 0, npml, npml))
    m = y.build(spec)
    return m, jnp.full(NX, H), jnp.full(NY, H), jnp.full(nz, H)


def _iris_pec(spec, aperture=APERTURE, z0=Z0):
    """Grid-aligned inductive iris, one cell thick: the same node-Dirichlet
    staircase as hplane.build (metal nodes ix <= fc and ix >= nx - fc)."""
    fc = (NX - aperture) // 2
    cells = np.zeros((NX, NY, spec.nz), dtype=bool)
    cells[:fc, :, z0] = True
    cells[NX - fc:, :, z0] = True
    return y.pec_edges_from_cells(spec, cells)


_CACHE: dict = {}


def _iris_s_params():
    """|S11|, |S21| of the iris (memoised: G2 and G3 read the same solve)."""
    if "iris" not in _CACHE:
        m, dx, dy, dz = _guide()
        pec = _iris_pec(m.spec)
        s11, s21 = y.te10_s_params(m, F0, None, dx, dy, dz, K_SRC, K_REF1, K_REF2, pec=pec)
        _CACHE["iris"] = (complex(s11), complex(s21))
    return _CACHE["iris"]


def test_operator_is_the_product_of_the_two_curls():
    """Independent reference: scipy builds Ch @ Ce - k0^2 diag(eps) from the
    same curl entries; the entry-wise product map must reproduce it, the
    outer walls must be identity rows, and the edge permittivity must be the
    average of the cells sharing the edge."""
    with enable_x64():
        spec = y.Yee3DSpec(nx=5, ny=4, nz=6, pml=(1, 0, 0, 1, 2, 2), pml_kappa_max=2.0)
        m = y.build(spec)
        rng = np.random.default_rng(0)
        dx, dy, dz = (1e-3 * (1 + 0.3 * rng.random(n)) for n in (5, 4, 6))
        eps = 1 + rng.random((5, 4, 6)) + 0.1j * rng.random((5, 4, 6))
        ce, ch, omega = y._curl_values(m, F0, dx, dy, dz)
        Ce = sp.coo_matrix((np.asarray(ce), (m.ce_rows, m.ce_cols)), shape=(m.n_faces, m.n_edges))
        Ch = sp.coo_matrix((np.asarray(ch), (m.ch_rows, m.ch_cols)), shape=(m.n_edges, m.n_faces))
        k0 = float(omega) / y.C0
        eps_e = np.asarray(y._eps_on_edges(m, eps))
        ref = (Ch.tocsr() @ Ce.tocsr() - k0 ** 2 * sp.diags(eps_e)).tolil()
        ref[m.wall, :] = 0
        ref[:, m.wall] = 0
        for i in np.where(m.wall)[0]:
            ref[i, i] = 1.0
        data, rhs = y.assemble(m, F0, eps, dx, dy, dz, y.te10_current(m, dx, 3))
        a_mat = sp.coo_matrix((np.asarray(data), (m.rows, m.cols)), shape=(m.n_edges,) * 2)
        gap = abs(a_mat.tocsr() - ref.tocsr()).max()
        assert gap < 1e-12 * abs(a_mat).max()          # measured 2.7e-16 relative
        # PML entries are complex (stretch on), interior entries real
        assert np.max(np.abs(np.imag(np.asarray(ce)))) > 0
        ex, _, _ = y.split_edges(m, jnp.asarray(eps_e))
        i, j, k = 2, 2, 3
        want = 0.25 * (eps[i, j - 1, k - 1] + eps[i, j, k - 1] + eps[i, j - 1, k] + eps[i, j, k])
        assert abs(complex(ex[i, j, k]) - want) < 1e-15
        # rhs = -j omega mu0 J with the wall rows zeroed
        assert float(jnp.max(jnp.abs(rhs[m.wall]))) == 0.0
        phi, _ = y._te10_profile(dx)
        want_rhs = float(omega) * y.MU0 * float(jnp.max(phi))
        assert abs(float(jnp.max(jnp.abs(rhs))) - want_rhs) < 1e-9 * want_rhs
        # PEC mask: an interior metal cell makes exactly its 12 edges PEC
        cells = np.zeros((5, 4, 6), dtype=bool)
        cells[2, 1, 3] = True
        pec = y.pec_edges_from_cells(spec, cells)
        assert [int(p.sum()) for p in pec] == [4, 4, 4]
        assert pec[0][2, 1:3, 3:5].all() and pec[1][2:4, 1, 3:5].all() and pec[2][2:4, 1:3, 3].all()


def test_empty_guide_pml_reflection_and_poynting_power():
    """G1. A TE10 current sheet in the empty guide; the modal amplitudes on
    the planes between the source and the +z PML are split into a forward
    and a backward wave with the grid's own discrete beta (fit residual
    ~2e-14, so the split is exact). |Gamma_PML| <= 1e-3 with 10 cells and
    smaller with 16 (measured 1.9e-5 and 3.6e-6). The field is a pure Ey
    (Ex, Ez ~1e-15), and the Poynting power carried by (E, H) equals the
    discrete modal power |A|^2 (a b / 4) sin(beta h) / (omega mu0 h)."""
    with enable_x64():
        beta = y.te10_discrete_beta(F0, A_WR90, H, H)
        f = np.exp(-1j * beta * H)
        gammas = []
        for npml in (10, 16):
            m, dx, dy, dz = _guide(npml=npml, nz=40 + 2 * npml)
            nz = m.shape[2]
            k_src = npml + 3
            e = y.solve(m, F0, None, dx, dy, dz, y.te10_current(m, dx, k_src))
            assert float(jnp.max(jnp.abs(e[0]))) < 1e-12
            assert float(jnp.max(jnp.abs(e[2]))) < 1e-12
            ks = np.arange(k_src + 4, nz - npml - 1)
            amps = jnp.stack([y.te10_amplitude(e[1], dx, int(k)) for k in ks])
            a_f, a_b = y.wave_split(amps, f, ks)
            fit = a_f * f ** ks + a_b * f ** (-ks)
            assert float(jnp.max(jnp.abs(amps - fit))) < 1e-10 * abs(complex(a_f))
            gammas.append(abs(complex(a_b / a_f)))
            if npml == 10:
                hx, hy, hz = y.h_from_e(m, F0, e, dx, dy, dz)
                k = k_src + 8
                ey_mid = 0.5 * (e[1][:, :, k] + e[1][:, :, k + 1])
                _, w = y._te10_profile(dx)
                p_num = 0.5 * jnp.real(jnp.sum(-ey_mid * jnp.conj(hx[:, :, k]) * w[:, None] * dy[None, :]))
                omega = 2 * np.pi * F0
                b = NY * H
                p_mode = abs(complex(a_f)) ** 2 * (A_WR90 * b / 4) * np.sin(beta * H) / (omega * y.MU0 * H)
                # measured 3.8e-10 relative (the discrete identity is exact;
                # the 2e-5 backward wave's cross terms cancel in the power)
                assert abs(float(p_num) - p_mode) < 1e-8 * p_mode
                assert float(jnp.max(jnp.abs(hy))) < 1e-12 * float(jnp.max(jnp.abs(hx)))
        assert gammas[0] <= 1e-3, gammas
        assert gammas[1] < gammas[0], gammas


def test_inductive_iris_matches_hplane_and_conserves_energy():
    """G2 + G3. The one-cell inductive iris on the same transverse grid as
    hplane (24 cells across a, node-Dirichlet metal at the same nodes): the
    y-invariant Ey field reduces the curl-curl stencil to the 5-point
    Laplacian, so the only differences are PML vs exact DtN ports and the
    normalisation-run readout. Measured gaps 3.0e-5 (|S11|) and 1.4e-5
    (|S21|); |S11|^2 + |S21|^2 = 1 + 6.4e-5."""
    with enable_x64():
        s11, s21 = _iris_s_params()
        hm = hplane.build(hplane.HPlaneSpec(a=A_WR90, base_cells=NX, apertures_cells=(APERTURE,),
                                            margin_cells=8))
        r11, r21 = hplane.solve(hm, F0)
        gap11 = abs(abs(s11) - abs(complex(r11)))
        gap21 = abs(abs(s21) - abs(complex(r21)))
        assert gap11 < 5e-3, gap11
        assert gap21 < 5e-3, gap21
        assert abs(s11) > 0.5                      # a real obstacle, not a trivially empty guide
        assert abs(abs(s11) ** 2 + abs(s21) ** 2 - 1.0) < 1e-2


def test_gradients_wrt_eps_cell_and_dz_step_match_fd4_jvp_and_jit():
    """G4. d|S11|^2/d eps_r of one interior cell and d|S11|^2/d dz of one
    step between the reference plane and the iris, reverse mode vs FD4.
    FD steps (1e-2 on eps, 0.02 h on dz) move the objective by ~1e-5 and
    ~1e-4, far above the ~1e-9 LU noise floor. Measured relative agreement
    2.2e-11 (eps) and 3.6e-10 (dz); jvp vs grad 1e-12; jit exact."""
    with enable_x64():
        m, dx, dy, dz = _guide()
        pec = _iris_pec(m.spec)
        eps0 = jnp.ones((NX, NY, NZ), jnp.complex128)

        def obj(eps, dzv):
            s11, _ = y.te10_s_params(m, F0, eps, dx, dy, dzv, K_SRC, K_REF1, K_REF2, pec=pec)
            return jnp.abs(s11) ** 2

        g_eps, g_dz = jax.grad(obj, argnums=(0, 1))(eps0, dz)
        ci = (12, 1, Z0 + 4)
        fd = _fd4(lambda t: obj(eps0.at[ci].add(t), dz), 0.0, 1e-2)
        assert abs(float(jnp.real(g_eps[ci])) - float(fd)) < 1e-4 * abs(float(fd))
        kz = 25
        fd_z = _fd4(lambda t: obj(eps0, dz.at[kz].add(t)), 0.0, 0.02 * H)
        assert abs(float(g_dz[kz]) - float(fd_z)) < 1e-4 * abs(float(fd_z))
        assert abs(float(fd_z)) * 0.02 * H > 1e-6        # the step moves the objective
        _, jv = jax.jvp(lambda d: obj(eps0, d), (dz,), (jnp.zeros(NZ).at[kz].set(1.0),))
        assert abs(float(jv) - float(g_dz[kz])) < 1e-9 * abs(float(jv))
        v_jit = jax.jit(obj)(eps0, dz)
        assert abs(float(v_jit) - float(obj(eps0, dz))) < 1e-12
        g_jit = jax.jit(jax.grad(obj))(eps0, dz)
        assert float(jnp.max(jnp.abs(g_jit - g_eps))) < 1e-12
        # PEC cells (iris metal) carry no permittivity sensitivity
        fc = (NX - APERTURE) // 2
        assert float(jnp.max(jnp.abs(g_eps[:fc, :, Z0]))) == 0.0


def test_lossy_dielectric_slab_attenuates_and_its_loss_gradient_matches_fd4():
    """G5. A 5-cell slab eps_r = 2 - j sigma/(omega eps0) across the guide:
    |S21| falls monotonically with sigma, and d|S21|^2/d(eps'') of one slab
    cell (reverse mode, JAX conj convention: dL/dIm = -Im grad) matches FD4
    along the imaginary axis (FD step 1e-2 on eps''). Measured |S21| =
    0.896, 0.466, 0.095 for sigma = 0, 1, 5 S/m; gradient rel 8.5e-14."""
    with enable_x64():
        m, dx, dy, dz = _guide()
        omega = 2 * np.pi * F0

        def slab_eps(sigma):
            eps = jnp.ones((NX, NY, NZ), jnp.complex128)
            return eps.at[:, :, Z0 - 2:Z0 + 3].set(2.0 - 1j * sigma / (omega * y.EPS0))

        s21s = [abs(complex(y.te10_s_params(m, F0, slab_eps(s), dx, dy, dz,
                                            K_SRC, K_REF1, K_REF2)[1])) for s in (0.0, 1.0, 5.0)]
        assert s21s[0] > s21s[1] > s21s[2], s21s
        assert s21s[2] < 0.5 * s21s[0], s21s

        def obj(eps):
            _, s21 = y.te10_s_params(m, F0, eps, dx, dy, dz, K_SRC, K_REF1, K_REF2)
            return jnp.abs(s21) ** 2

        eps1 = slab_eps(1.0)
        g = jax.grad(obj)(eps1)
        ci = (12, 1, Z0)
        d_im = -float(jnp.imag(g[ci]))
        assert np.isfinite(d_im)
        fd = _fd4(lambda t: obj(eps1.at[ci].add(1j * t)), 0.0, 1e-2)
        assert abs(d_im - float(fd)) < 1e-4 * abs(float(fd)), (d_im, float(fd))


def test_block_sources_and_input_checks():
    """Two current sheets solved as one (n, 2) block equal two single solves;
    malformed inputs are rejected."""
    with enable_x64():
        m, dx, dy, dz = _guide(npml=4, nz=20)
        j1 = y.te10_current(m, dx, 7)
        j2 = y.te10_current(m, dx, 12)
        e1 = y.solve(m, F0, None, dx, dy, dz, j1)
        e2 = y.solve(m, F0, None, dx, dy, dz, j2)
        eb = y.solve(m, F0, None, dx, dy, dz, [j1, j2])
        assert eb[1].shape == e1[1].shape + (2,)
        assert float(jnp.max(jnp.abs(eb[1][..., 0] - e1[1]))) < 1e-12 * float(jnp.max(jnp.abs(e1[1])))
        assert float(jnp.max(jnp.abs(eb[1][..., 1] - e2[1]))) < 1e-12 * float(jnp.max(jnp.abs(e2[1])))
        with pytest.raises(ValueError, match="eps_r"):
            y.assemble(m, F0, jnp.ones((NX, NY, 3)), dx, dy, dz, j1)
        with pytest.raises(ValueError, match="pec"):
            y.assemble(m, F0, None, dx, dy, dz, j1, pec=(np.zeros((2, 2, 2), bool),) * 3)
        with pytest.raises(ValueError, match="step array"):
            y.assemble(m, F0, None, dx, dy, dz[:-1], j1)
        with pytest.raises(ValueError, match="PML"):
            y.build(y.Yee3DSpec(nx=4, ny=4, nz=6, pml=(0, 0, 0, 0, 3, 3)))


# ---------------------------------------------------------------------------
# Regression: the rectangular curls must keep their tail rows.
# ---------------------------------------------------------------------------

def test_rectangular_curl_operators_keep_every_row_against_scipy():
    """``Ch`` maps faces -> edges and is RECTANGULAR (n_edges > n_faces), so
    a square scatter of its entries silently drops every row with edge id >=
    n_faces (the last Ez edges) -- the defect found in the spiral track,
    where a lumped port sitting on those edges read a wrong current. Gate:
    ``h_from_e`` and ``curl_h`` against an explicit scipy.sparse assembly of
    the same Ce / Ch entries (the reference applies the ``terms.ch_scale``
    multiplier itself), single vector and (n, 2) block, with the z-PML on so
    the curl entries are complex.

    On this 4 x 5 x 6 grid: 523 edges, 434 faces, 1736 Ch entries of which
    290 (89 distinct rows) are in the dropped range and carry 43 % (single)
    / 45 % (block) of the reference's norm -- a truncated result misses the
    largest reference entry outright, so the tolerance below does not have
    to be tight to catch it. Measured worst relative disagreement 2.5e-16
    (summation order only); gate 1e-12."""
    with enable_x64():
        spec = y.Yee3DSpec(nx=4, ny=5, nz=6, pml=(0, 0, 0, 0, 2, 2), pml_kappa_max=2.0)
        m = y.build(spec)
        assert m.n_edges > m.n_faces
        tail = m.ch_rows >= m.n_faces
        assert int(np.sum(tail)) > 0 and len(np.unique(m.ch_rows[tail])) > 0
        rng = np.random.default_rng(11)
        dx, dy, dz = (1e-4 * (1 + 0.4 * rng.random(n)) for n in (4, 5, 6))
        f0 = 3e9
        scale = 0.5 + rng.random(len(m.ch_rows))        # a non-trivial ch_scale
        terms = y.BoundaryTerms(ch_scale=jnp.asarray(scale))
        ce, ch, omega = y._curl_values(m, f0, dx, dy, dz)
        assert np.max(np.abs(np.imag(np.asarray(ch)))) > 0      # PML stretch is live
        ce_ref = sp.coo_matrix((np.asarray(ce), (m.ce_rows, m.ce_cols)),
                               shape=(m.n_faces, m.n_edges)).tocsr()
        ch_ref = sp.coo_matrix((np.asarray(ch) * scale, (m.ch_rows, m.ch_cols)),
                               shape=(m.n_edges, m.n_faces)).tocsr()
        for ncol in (None, 2):
            shape = (m.n_edges,) if ncol is None else (m.n_edges, ncol)
            e_flat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            e = y.split_edges(m, jnp.asarray(e_flat))
            # H = -curl E / (j omega mu0) on the FACES
            h = y.h_from_e(m, f0, e, dx, dy, dz)
            h_flat = np.concatenate([np.asarray(a).reshape(-1, *shape[1:]) for a in h])
            h_want = (ce_ref @ e_flat) / (-1j * complex(omega) * y.MU0)
            assert h_flat.shape == (m.n_faces,) + shape[1:]
            assert np.max(np.abs(h_flat - h_want)) <= 1e-12 * np.max(np.abs(h_want))
            # Ch H on the EDGES, with the ch_scale hook
            got = np.asarray(y.curl_h(m, f0, h, dx, dy, dz, terms))
            want = ch_ref @ h_want
            assert got.shape == shape
            assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
            # the tail rows are populated, in both the reference and the result
            assert np.linalg.norm(want[m.n_faces:]) > 0.3 * np.linalg.norm(want)
            assert np.linalg.norm(got[m.n_faces:]) > 0.3 * np.linalg.norm(got)
            # and the truncated (buggy) answer is nowhere near: > 50 % off
            trunc = np.zeros_like(want)
            trunc[:m.n_faces] = want[:m.n_faces]
            assert np.max(np.abs(trunc - want)) > 0.5 * np.max(np.abs(want))
            # without terms the scale drops out exactly
            got0 = np.asarray(y.curl_h(m, f0, h, dx, dy, dz))
            want0 = sp.coo_matrix((np.asarray(ch), (m.ch_rows, m.ch_cols)),
                                  shape=(m.n_edges, m.n_faces)).tocsr() @ h_want
            assert np.max(np.abs(got0 - want0)) <= 1e-12 * np.max(np.abs(want0))


def _eps_on_edges_independent(nx, ny, nz, eps):
    """Independent re-derivation of the edge permittivity: the arithmetic mean
    of the (up to four) cells sharing an edge, boundary indices clamped.

    Explicit per-edge index clamping, against the solver's pad-and-slice sum
    in ``yee3d._eps_on_edges`` -- a different code path for the same rule."""
    out = []
    for c, shape in enumerate(y.edge_shapes(nx, ny, nz)):
        a, b = (c + 1) % 3, (c + 2) % 3
        lim = (nx, ny, nz)
        arr = np.zeros(shape, np.complex128)
        for ijk in np.ndindex(*shape):
            acc = 0.0 + 0j
            for da in (-1, 0):
                for db in (-1, 0):
                    cell = list(ijk)
                    cell[a] = min(max(ijk[a] + da, 0), lim[a] - 1)
                    cell[b] = min(max(ijk[b] + db, 0), lim[b] - 1)
                    acc += eps[tuple(cell)]
            arr[ijk] = 0.25 * acc
        out.append(arr.ravel())
    return np.concatenate(out)


def test_assembled_operator_equals_ch_ce_minus_k0sq_eps():
    """The assembled ``A`` must BE the composition of the two curls the field
    post-processing uses: on every free edge

        A = (Ch * ch_scale) @ Ce - k0^2 diag(eps_edge) + diag(diag_add),

    with ``Ce``/``Ch`` re-assembled with scipy (they are checked entry by
    entry against the same reference in the test above), ``eps_edge``
    re-derived independently by per-edge index clamping, and PEC rows
    (outer wall plus an interior iris) reduced to a unit diagonal. This is
    what makes the rectangular-curl regression below a statement about the
    system that is actually solved, not only about a helper: the SAME
    ``ch_scale``-weighted ``Ch`` entries that ``curl_h`` is checked against
    are shown to be the ones ``A`` is built from.

    Increment over ``test_operator_is_the_product_of_the_two_curls`` above,
    which checks the vacuum-hook case: the ``BoundaryTerms`` hooks
    (``ch_scale``, ``diag_add``) are live here, interior PEC edges are in the
    mask as well as the wall, and the edge permittivity is re-derived for
    every edge instead of spot-checked at one.

    Grid 5 x 3 x 7 with x- and y-PML (so the stretch is complex in two
    directions), graded random steps, non-uniform complex eps_r, a complex
    ``ch_scale`` and a non-zero ``diag_add``: 472 edges, 386 faces, 284 wall
    plus 28 iris edges PEC, 168 free. Measured max|A - A_ref| / max|A_ref|
    over the free block = 1.3e-16 (7.5e-9 absolute on a 5.6e7 peak); gate
    1e-12. The curl-curl part carries the operator there (peak 5.6e7 against
    k0^2 = 6.3e4), the independent eps map agrees with the solver's to 0.0
    (same arithmetic, different indexing), and every PEC row is exactly
    (1 on the diagonal, 0 elsewhere)."""
    with enable_x64():
        nx, ny, nz = 5, 3, 7
        spec = y.Yee3DSpec(nx=nx, ny=ny, nz=nz, pml=(2, 2, 1, 1, 0, 0), pml_kappa_max=3.0)
        m = y.build(spec)
        rng = np.random.default_rng(5)
        dx, dy, dz = (2e-4 * (1 + 0.5 * rng.random(n)) for n in (nx, ny, nz))
        f0 = 12e9
        eps_r = (1.0 + 3.0 * rng.random((nx, ny, nz))
                 - 1j * rng.random((nx, ny, nz)))
        scale = 0.3 + rng.random(len(m.ch_rows)) + 0.2j * rng.random(len(m.ch_rows))
        diag_add = 1e3 * (rng.standard_normal(m.n_edges) + 1j * rng.standard_normal(m.n_edges))
        cells = np.zeros((nx, ny, nz), bool)
        cells[1, :, 3] = True                      # an interior iris -> extra PEC edges
        pec = y.pec_edges_from_cells(spec, cells)
        terms = y.BoundaryTerms(ch_scale=jnp.asarray(scale), diag_add=jnp.asarray(diag_add))

        ce, ch, omega = y._curl_values(m, f0, dx, dy, dz)
        ce_ref = sp.coo_matrix((np.asarray(ce), (m.ce_rows, m.ce_cols)),
                               shape=(m.n_faces, m.n_edges)).tocsr()
        ch_ref = sp.coo_matrix((np.asarray(ch) * scale, (m.ch_rows, m.ch_cols)),
                               shape=(m.n_edges, m.n_faces)).tocsr()
        eps_e = _eps_on_edges_independent(nx, ny, nz, eps_r)
        assert np.max(np.abs(eps_e - np.asarray(y._eps_on_edges(m, jnp.asarray(eps_r))))) < 1e-14
        k0 = float(omega) / y.C0
        a_ref = (ch_ref @ ce_ref - k0 ** 2 * sp.diags(eps_e)
                 + sp.diags(diag_add)).tocsr()

        srcs = y.split_edges(m, jnp.zeros(m.n_edges, jnp.complex128))
        data, _ = y.assemble(m, f0, jnp.asarray(eps_r), dx, dy, dz, srcs, pec, terms)
        a = sp.coo_matrix((np.asarray(data), (m.rows, m.cols)),
                          shape=(m.n_edges, m.n_edges)).tocsr()

        # free = not on the outer wall and not on the iris (built here, not
        # read back from the solver's own mask)
        pec_flat = np.concatenate([np.asarray(p).ravel() for p in pec])
        free = ~(m.wall | pec_flat)
        assert 0 < int(free.sum()) < m.n_edges and int(pec_flat.sum()) > 0
        proj = sp.diags(free.astype(float))
        ref_block = proj @ a_ref @ proj
        err = abs(proj @ (a - a_ref) @ proj).max()
        assert err <= 1e-12 * abs(ref_block).max(), (err, abs(ref_block).max())
        # the curl-curl part is what carries the operator: it is not a tiny
        # correction on top of the k0^2 term
        assert abs(proj @ (ch_ref @ ce_ref) @ proj).max() > 10 * k0 ** 2
        # PEC rows: unit diagonal, nothing else
        dense_pec = np.asarray(a[pec_flat].todense())
        assert np.allclose(dense_pec.sum(axis=1), 1.0, atol=1e-12)
        assert np.allclose(np.abs(dense_pec).sum(axis=1), 1.0, atol=1e-12)


if __name__ == "__main__":
    t = time.time()
    pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    print(f"{time.time() - t:.1f} s")
