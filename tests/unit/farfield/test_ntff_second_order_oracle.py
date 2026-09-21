"""The Huygens-box far field must halve its error four times over when the
surface mesh is halved twice — second order in the cell size.

A handful of point dipoles sit inside a closed box in free space. Their exact
fields are known everywhere, so the exact radiated far field is known too.
This test writes those exact fields into the six Yee field arrays, each
component at the position and at the time the FDTD solver would really hold
it (E at ``(n+1)*dt``, H half a step earlier at ``(n+1/2)*dt``), steps the
production ``accumulate_ntff`` exactly the way a runner's scan body does, and
sends the result through the production ``compute_far_field``. Nothing here
re-implements the transform; the only thing supplied is the physics.

What that catches: the tangential H stored for a face sits half a cell OFF
the face along its normal, and the two tangential E components straddle the
face cell along different edges. Taking all four where the lattice stores
them and pricing them at the cell's lower corner is a left-endpoint rectangle
rule — first order, and on a coarse surface it moves sidelobes by dB. Moving
each component to the centre of its face cell first, and integrating at that
centre, is the midpoint rule.

Measured on this oracle (0.7-wavelength box, 3 GHz, three electric and one
magnetic dipole; complex relative L2 error over 25x16 directions, both
far-field components):

    surface sampling      lambda/10   lambda/20   lambda/40   ratios
    face-centre (now)      1.479e-02   3.691e-03   9.225e-04  4.01, 4.00
    legacy node rule       3.174e-01   1.599e-01   8.011e-02  1.99, 2.00

and on a mesh whose z cells stretch smoothly by 3x across the axis, the same
box and the same dipoles:

    face-centre (now)      1.663e-02   4.789e-03   1.339e-03  3.47, 3.58
    legacy node rule       3.205e-01   1.621e-01   8.165e-02  1.98, 1.99

At lambda/20 the midpoint rule's own quadrature error on the radiation
phase, (k*d)^2/24, is 4.1e-3 — so 3.7e-3 is at the floor of what this rule
can do on that mesh, not a residual defect.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import FDTDState
from rfx.farfield import (
    ETA_0,
    NTFFBox,
    accumulate_ntff,
    compute_far_field,
    init_ntff_data,
    with_face_centre_collocation,
)
from rfx.grid import C0

from tests._x64_compat import enable_x64

FREQ = 3.0e9
LAMBDA = C0 / FREQ
K0 = 2.0 * np.pi / LAMBDA
OMEGA = 2.0 * np.pi * FREQ
STEPS_PER_PERIOD = 16


# ---------------------------------------------------------------------------
# Closed-form fields of a set of point dipoles (self-contained)
# ---------------------------------------------------------------------------
#
# Time convention exp(+j*omega*t), so a wave leaving the source carries
# exp(-j*k*R). For an electric current moment p (A*m) and a magnetic current
# moment m (V*m) at the same point, with u = R/|R|, P = u u^T, X = [u]_cross:
#
#     E = Ge . p + Bx . m           H = -Bx . p + (Ge/eta^2) . m
#     Ge = A_R P + A_T (P - I)      Bx = B X
#     A_R = eta/(2 pi R^2) (1 + 1/(jkR)) exp(-jkR)
#     A_T = j eta k/(4 pi R) (1 + 1/(jkR) - 1/(kR)^2) exp(-jkR)
#     B   = j k/(4 pi R) (1 + 1/(jkR)) exp(-jkR)
#
# The magnetic-dipole entries follow by duality (E -> H, H -> -E,
# eta -> 1/eta). Far from the sources only the 1/R pieces survive, which is
# the closed-form far field written out in _dipole_far_field below — the
# point-source case of the same radiation integral rfx/farfield.py evaluates.


def _dipole_fields(r_obs, r_src, moments_e, moments_m, k=K0, eta=ETA_0):
    """Exact E and H at ``r_obs`` (n, 3) of the dipoles at ``r_src`` (m, 3)."""
    r_obs = np.asarray(r_obs, dtype=np.float64).reshape(-1, 3)
    r_src = np.asarray(r_src, dtype=np.float64).reshape(-1, 3)
    d = r_obs[:, None, :] - r_src[None, :, :]
    R = np.linalg.norm(d, axis=-1)
    if np.any(R <= 0.0):
        raise ValueError("a sample point coincides with a dipole")
    u = d / R[..., None]

    jkR = 1j * k * R
    expo = np.exp(-jkR)
    A_R = eta / (2.0 * np.pi * R ** 2) * (1.0 + 1.0 / jkR) * expo
    A_T = (1j * eta * k / (4.0 * np.pi * R)
           * (1.0 + 1.0 / jkR - 1.0 / (k * R) ** 2) * expo)
    B = 1j * k / (4.0 * np.pi * R) * (1.0 + 1.0 / jkR) * expo

    P = u[..., :, None] * u[..., None, :]
    eye = np.eye(3)[None, None, :, :]
    Ge = A_R[..., None, None] * P + A_T[..., None, None] * (P - eye)
    zero = np.zeros_like(u[..., 0])
    X = np.stack([
        np.stack([zero, -u[..., 2], u[..., 1]], axis=-1),
        np.stack([u[..., 2], zero, -u[..., 0]], axis=-1),
        np.stack([-u[..., 1], u[..., 0], zero], axis=-1),
    ], axis=-2)
    Bx = B[..., None, None] * X.astype(np.complex128)

    p = np.asarray(moments_e, dtype=np.complex128)
    q = np.asarray(moments_m, dtype=np.complex128)
    E = np.einsum("nmij,mj->ni", Ge, p) + np.einsum("nmij,mj->ni", Bx, q)
    H = (-np.einsum("nmij,mj->ni", Bx, p)
         + np.einsum("nmij,mj->ni", Ge, q) / eta ** 2)
    return E, H


def _dipole_far_field(theta, phi, r_src, moments_e, moments_m, k=K0, eta=ETA_0):
    """Exact far field, shaped and phased exactly like ``compute_far_field``."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    sth, cth = np.sin(TH), np.cos(TH)
    sph, cph = np.sin(PH), np.cos(PH)
    r_hat = np.stack([sth * cph, sth * sph, cth], axis=-1).reshape(-1, 3)
    th_hat = np.stack([cth * cph, cth * sph, -sth], axis=-1).reshape(-1, 3)
    ph_hat = np.stack([-sph, cph, np.zeros_like(sth)], axis=-1).reshape(-1, 3)

    r_src = np.asarray(r_src, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(moments_e, dtype=np.complex128)
    q = np.asarray(moments_m, dtype=np.complex128)
    phase = np.exp(1j * k * (r_hat @ r_src.T))
    N, L = phase @ p, phase @ q
    N_th = np.sum(N * th_hat, axis=1)
    N_ph = np.sum(N * ph_hat, axis=1)
    L_th = np.sum(L * th_hat, axis=1)
    L_ph = np.sum(L * ph_hat, axis=1)
    jk4pi = 1j * k / (4.0 * np.pi)
    E_theta = -jk4pi * (L_ph + eta * N_th)
    E_phi = jk4pi * (L_th - eta * N_ph)
    return E_theta.reshape(TH.shape), E_phi.reshape(TH.shape)


# ---------------------------------------------------------------------------
# Grid stubs: only what the transform reads off a grid
# ---------------------------------------------------------------------------

class _UniformGrid:
    """Cubic cells, no CPML padding, so node ``i`` sits at ``i*dx``."""

    def __init__(self, dx, n):
        self.dx = float(dx)
        self.cpml_layers = 0
        self.nx = self.ny = self.nz = int(n)


class _GradedGrid:
    """Per-cell widths on every axis (the ``dx_arr``/``dy_arr``/``dz`` path)."""

    def __init__(self, dx_arr, dy_arr, dz):
        self.dx_arr = np.asarray(dx_arr, dtype=np.float64)
        self.dy_arr = np.asarray(dy_arr, dtype=np.float64)
        self.dz = np.asarray(dz, dtype=np.float64)
        self.dx = float(self.dx_arr[0])
        self.dy = float(self.dy_arr[0])
        self.cpml_layers = 0
        self.nx = len(self.dx_arr)
        self.ny = len(self.dy_arr)
        self.nz = len(self.dz)


def _nodes(widths):
    """Node coordinates: node ``i`` bounds cell ``i`` from below."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(widths, dtype=np.float64))])


# ---------------------------------------------------------------------------
# The oracle itself
# ---------------------------------------------------------------------------

def _dipole_set(centre):
    """Three electric and one magnetic dipole, well inside the box."""
    c = np.asarray(centre, dtype=np.float64)
    r_src = c + LAMBDA * np.array([
        [+0.066, -0.042, +0.030],
        [-0.054, +0.072, -0.036],
        [+0.018, +0.024, +0.078],
        [-0.030, -0.018, -0.066],
    ])
    moments_e = np.array([
        [1.0, 0.3j, -0.2],
        [0.2, -0.5, 0.8j],
        [0.0, 0.0, 0.0],
        [-0.4j, 0.25, 0.15],
    ], dtype=np.complex128)
    moments_m = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.4, 0.1, -0.3],
        [0.0, 0.0, 0.0],
    ], dtype=np.complex128) * ETA_0
    return r_src, moments_e, moments_m


def _yee_phasors(dx_w, dy_w, dz_w, r_src, moments_e, moments_m):
    """Complex phasor of each Yee component at the position it really occupies.

    Returns a dict of six (nx, ny, nz) complex arrays. ``ex`` lives at the
    x cell centre and on the y/z nodes, ``hx`` on the x node and at the y/z
    cell centres, and so on round the lattice.
    """
    xn, yn, zn = _nodes(dx_w), _nodes(dy_w), _nodes(dz_w)
    nx, ny, nz = len(dx_w), len(dy_w), len(dz_w)
    xn, yn, zn = xn[:nx + 1], yn[:ny + 1], zn[:nz + 1]
    xc = 0.5 * (xn[:-1] + xn[1:])
    yc = 0.5 * (yn[:-1] + yn[1:])
    zc = 0.5 * (zn[:-1] + zn[1:])
    xn, yn, zn = xn[:nx], yn[:ny], zn[:nz]

    layout = {
        "ex": (xc, yn, zn, "E"), "ey": (xn, yc, zn, "E"), "ez": (xn, yn, zc, "E"),
        "hx": (xn, yc, zc, "H"), "hy": (xc, yn, zc, "H"), "hz": (xc, yc, zn, "H"),
    }
    out = {}
    for comp, (ax, ay, az, kind) in layout.items():
        X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        E, H = _dipole_fields(pts, r_src, moments_e, moments_m)
        vec = E if kind == "E" else H
        out[comp] = vec[:, "xyz".index(comp[1])].reshape(nx, ny, nz)
    return out


def _accumulate_oracle(box, phasors, *, dt, n_steps,
                       e_time_offset=1.0, field_dtype=jnp.float32):
    """Step the production accumulator over ``n_steps`` of exact fields.

    ``e_time_offset`` is the multiple of ``dt`` at which the E field handed to
    the accumulator is sampled; 1.0 is the truth (the state is post-E-update).
    """
    data = init_ntff_data(box, field_dtype=field_dtype)
    acc = jax.jit(lambda d, s, n: accumulate_ntff(d, s, box, dt, n))
    ph = {k: jnp.asarray(v) for k, v in phasors.items()}
    for n in range(n_steps):
        t_e = (n + e_time_offset) * dt
        t_h = (n + 0.5) * dt
        rot_e = np.exp(1j * OMEGA * t_e)
        rot_h = np.exp(1j * OMEGA * t_h)
        state = FDTDState(
            ex=jnp.real(ph["ex"] * rot_e).astype(field_dtype),
            ey=jnp.real(ph["ey"] * rot_e).astype(field_dtype),
            ez=jnp.real(ph["ez"] * rot_e).astype(field_dtype),
            hx=jnp.real(ph["hx"] * rot_h).astype(field_dtype),
            hy=jnp.real(ph["hy"] * rot_h).astype(field_dtype),
            hz=jnp.real(ph["hz"] * rot_h).astype(field_dtype),
            step=jnp.asarray(n, dtype=jnp.int32),
        )
        data = acc(data, state, jnp.asarray(n, dtype=jnp.int32))
    return data


def _relative_error(dx_w, dy_w, dz_w, grid, *, collocation="face_centre",
                    margin=3, e_time_offset=1.0, theta=None, phi=None):
    """Complex relative L2 error of the transformed far field, one mesh.

    Builds the box ``margin`` cells inside the array on every axis, fills the
    lattice with the exact dipole fields, runs the production accumulate +
    transform, and compares with the closed-form far field.
    """
    nx, ny, nz = len(dx_w), len(dy_w), len(dz_w)
    i_lo, i_hi = margin, nx - margin
    j_lo, j_hi = margin, ny - margin
    k_lo, k_hi = margin, nz - margin

    xn, yn, zn = _nodes(dx_w), _nodes(dy_w), _nodes(dz_w)
    centre = np.array([0.5 * (xn[i_lo] + xn[i_hi]),
                       0.5 * (yn[j_lo] + yn[j_hi]),
                       0.5 * (zn[k_lo] + zn[k_hi])])
    r_src, me, mm = _dipole_set(centre)

    box = NTFFBox(i_lo=i_lo, i_hi=i_hi, j_lo=j_lo, j_hi=j_hi,
                  k_lo=k_lo, k_hi=k_hi,
                  freqs=jnp.asarray([FREQ], dtype=jnp.float32))
    if collocation == "face_centre":
        box = with_face_centre_collocation(box, grid)

    phasors = _yee_phasors(dx_w, dy_w, dz_w, r_src, me, mm)
    dt = 1.0 / (FREQ * STEPS_PER_PERIOD)
    n_steps = STEPS_PER_PERIOD
    data = _accumulate_oracle(box, phasors, dt=dt, n_steps=n_steps,
                              e_time_offset=e_time_offset)

    if theta is None:
        theta = np.linspace(0.05, np.pi - 0.05, 25)
    if phi is None:
        phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ff = compute_far_field(data, box, grid, theta, phi)

    # A running DFT of a real single-tone field over an integer number of
    # periods returns (total time / 2) times the phasor, exactly.
    scale = 0.5 * n_steps * dt
    got = np.stack([np.asarray(ff.E_theta[0]) / scale,
                    np.asarray(ff.E_phi[0]) / scale])
    ref_th, ref_ph = _dipole_far_field(theta, phi, r_src, me, mm)
    ref = np.stack([ref_th, ref_ph])
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def _uniform_case(cells_per_lambda, box_lambda=0.7, margin=3, **kw):
    dx = LAMBDA / cells_per_lambda
    n_box = int(round(box_lambda * cells_per_lambda))
    n = n_box + 2 * margin
    widths = np.full(n, dx)
    grid = _UniformGrid(dx, n)
    return _relative_error(widths, widths, widths, grid, margin=margin, **kw)


# ---------------------------------------------------------------------------
# T1 — end-to-end second order through the production functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cells_per_lambda", [10, 20, 40])
def test_second_order_oracle_error_bound(cells_per_lambda):
    """Each mesh must clear the error it was measured at, with headroom."""
    bound = {10: 2.0e-2, 20: 5.0e-3, 40: 1.2e-3}[cells_per_lambda]
    err = _uniform_case(cells_per_lambda)
    assert err < bound, (
        f"NTFF far field off by {err:.3e} at lambda/{cells_per_lambda} "
        f"surface sampling (bound {bound:.1e})")


def test_second_order_oracle_convergence_rate():
    """Halving the surface mesh must cut the error by at least 3.5x."""
    errs = [_uniform_case(c) for c in (10, 20, 40)]
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(r >= 3.5 for r in ratios), (
        "NTFF surface integral is not second order: errors "
        f"{['%.3e' % e for e in errs]}, ratios {['%.2f' % r for r in ratios]}")
    assert errs[1] <= 5.0e-3, (
        f"NTFF far field off by {errs[1]:.3e} at lambda/20 (bound 5.0e-3)")


# ---------------------------------------------------------------------------
# T2 — the same on a graded mesh (dx_arr / dy_arr / dz path)
# ---------------------------------------------------------------------------

def _graded_widths(n, total, end_ratio=3.0):
    """Smoothly stretching cells spanning ``total``, widest/narrowest = 3.

    The cell-to-cell ratio is whatever spreads ``end_ratio`` over ``n``
    cells — 1.10 at the coarse mesh, 1.03 at the fine one, both well inside
    the 1.3 a solver would tolerate — so refining the mesh keeps the same
    physical extent and the same shape of grading.
    """
    r = end_ratio ** (1.0 / (n - 1))
    w = r ** np.arange(n, dtype=np.float64)
    return w * (total / w.sum())


def _graded_case(cells_per_lambda, box_lambda=0.7, margin=3):
    dx = LAMBDA / cells_per_lambda
    n = int(round(box_lambda * cells_per_lambda)) + 2 * margin
    uniform = np.full(n, dx)
    graded = _graded_widths(n, n * dx)
    grid = _GradedGrid(uniform, uniform, graded)
    return _relative_error(uniform, uniform, graded, grid, margin=margin)


@pytest.mark.parametrize("cells_per_lambda,bound", [(20, 7.0e-3), (40, 2.0e-3)])
def test_second_order_oracle_graded_axis(cells_per_lambda, bound):
    """A z-graded mesh must stay accurate and keep improving with refinement.

    The z cells stretch by 1.15 per cell over the top two thirds of the axis,
    so the two cells straddling a z face differ in width and the half-cell
    interpolation of the tangential H has to use their real widths.
    """
    err = _graded_case(cells_per_lambda)
    assert err < bound, (
        f"graded-mesh NTFF far field off by {err:.3e} at lambda/"
        f"{cells_per_lambda}-class sampling (bound {bound:.1e})")


def test_graded_axis_improves_with_refinement():
    coarse = _graded_case(20)
    fine = _graded_case(40)
    assert fine < coarse / 2.0, (
        f"graded-mesh error did not improve with refinement: {coarse:.3e} -> "
        f"{fine:.3e}")


# ---------------------------------------------------------------------------
# T4 — old dumps are read with the geometry they were accumulated with
# ---------------------------------------------------------------------------

def test_node_collocation_reproduces_legacy_far_field():
    """A box left at ``collocation="node"`` integrates exactly as before.

    Accumulators dumped by an earlier run were filled with the four
    components taken where the lattice stores them. Reading them back must
    put the samples on the lower-corner nodes again, not on the face-cell
    centres — otherwise every saved dump silently changes meaning.
    """
    dx = LAMBDA / 10
    n = 8 + 6
    widths = np.full(n, dx)
    grid = _UniformGrid(dx, n)
    margin = 3
    box = NTFFBox(i_lo=margin, i_hi=n - margin, j_lo=margin, j_hi=n - margin,
                  k_lo=margin, k_hi=n - margin,
                  freqs=jnp.asarray([FREQ], dtype=jnp.float32))
    assert box.collocation == "node"

    xn = _nodes(widths)
    centre = np.full(3, 0.5 * (xn[margin] + xn[n - margin]))
    r_src, me, mm = _dipole_set(centre)
    phasors = _yee_phasors(widths, widths, widths, r_src, me, mm)
    dt = 1.0 / (FREQ * STEPS_PER_PERIOD)
    data = _accumulate_oracle(box, phasors, dt=dt,
                              n_steps=STEPS_PER_PERIOD)

    theta = np.linspace(0.05, np.pi - 0.05, 9)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ff = compute_far_field(data, box, grid, theta, phi)

    # Reference: the legacy rule written out here — every stored component
    # read at its own lattice index, every sample priced at the cell's
    # lower-corner node, weighted by dx*dy.
    ref_th, ref_ph = _legacy_transform(data, box, widths, theta, phi)
    assert np.allclose(np.asarray(ff.E_theta[0]), ref_th, rtol=1e-12, atol=0.0)
    assert np.allclose(np.asarray(ff.E_phi[0]), ref_ph, rtol=1e-12, atol=0.0)


def _legacy_transform(data, box, widths, theta, phi):
    """The pre-second-order surface integral, written out independently."""
    dx = float(widths[0])
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    sth, cth = np.sin(TH), np.cos(TH)
    sph, cph = np.sin(PH), np.cos(PH)
    r_hat = np.stack([sth * cph, sth * sph, cth], axis=-1).reshape(-1, 3)
    th_hat = np.stack([cth * cph, cth * sph, -sth], axis=-1).reshape(-1, 3)
    ph_hat = np.stack([-sph, cph, np.zeros_like(sth)], axis=-1).reshape(-1, 3)

    i0, i1 = box.i_lo, box.i_hi
    j0, j1 = box.j_lo, box.j_hi
    k0, k1 = box.k_lo, box.k_hi
    N = np.zeros((r_hat.shape[0], 3), dtype=np.complex128)
    L = np.zeros_like(N)
    faces = [
        (np.asarray(data.x_lo[0], dtype=np.complex128), 0, -1, i0, (j0, j1), (k0, k1)),
        (np.asarray(data.x_hi[0], dtype=np.complex128), 0, +1, i1, (j0, j1), (k0, k1)),
        (np.asarray(data.y_lo[0], dtype=np.complex128), 1, -1, j0, (i0, i1), (k0, k1)),
        (np.asarray(data.y_hi[0], dtype=np.complex128), 1, +1, j1, (i0, i1), (k0, k1)),
        (np.asarray(data.z_lo[0], dtype=np.complex128), 2, -1, k0, (i0, i1), (j0, j1)),
        (np.asarray(data.z_hi[0], dtype=np.complex128), 2, +1, k1, (i0, i1), (j0, j1)),
    ]
    for f, axis, sign, idx, r1, r2 in faces:
        a = np.arange(r1[0], r1[1]) * dx
        b = np.arange(r2[0], r2[1]) * dx
        A, B = np.meshgrid(a, b, indexing="ij")
        fixed = np.full_like(A, idx * dx)
        if axis == 0:
            pos = np.stack([fixed, A, B], axis=-1)
        elif axis == 1:
            pos = np.stack([A, fixed, B], axis=-1)
        else:
            pos = np.stack([A, B, fixed], axis=-1)
        pos = pos.reshape(-1, 3)
        f = f.reshape(-1, 4)
        f0, f1_, f2, f3 = (f[:, i] for i in range(4))
        z = np.zeros_like(f0)
        s = sign
        if axis == 0:
            J = np.stack([z, -s * f3, s * f2], axis=-1)
            M = np.stack([z, s * f1_, -s * f0], axis=-1)
        elif axis == 1:
            J = np.stack([s * f3, z, -s * f2], axis=-1)
            M = np.stack([-s * f1_, z, s * f0], axis=-1)
        else:
            J = np.stack([-s * f3, s * f2, z], axis=-1)
            M = np.stack([s * f1_, -s * f0, z], axis=-1)
        phase = np.exp(1j * K0 * (r_hat @ pos.T))
        N += (phase @ J) * dx * dx
        L += (phase @ M) * dx * dx

    jk4pi = 1j * K0 / (4.0 * np.pi)
    E_th = -jk4pi * (np.sum(L * ph_hat, axis=1) + ETA_0 * np.sum(N * th_hat, axis=1))
    E_ph = jk4pi * (np.sum(L * th_hat, axis=1) - ETA_0 * np.sum(N * ph_hat, axis=1))
    return E_th.reshape(TH.shape), E_ph.reshape(TH.shape)


# ---------------------------------------------------------------------------
# A2 — a box with no room for the half-cell averages is refused loudly
# ---------------------------------------------------------------------------

def test_box_touching_the_array_boundary_is_refused():
    """Reading half a cell outside the box needs a cell of margin."""
    n = 12
    box = NTFFBox(i_lo=0, i_hi=n - 1, j_lo=2, j_hi=n - 2, k_lo=2, k_hi=n - 2,
                  freqs=jnp.asarray([FREQ], dtype=jnp.float32),
                  collocation="face_centre")
    zeros = jnp.zeros((n, n, n), dtype=jnp.float32)
    state = FDTDState(ex=zeros, ey=zeros, ez=zeros, hx=zeros, hy=zeros,
                      hz=zeros, step=jnp.asarray(0, dtype=jnp.int32))
    data = init_ntff_data(box)
    with pytest.raises(ValueError, match="no room for the face-centre"):
        accumulate_ntff(data, state, box, 1e-12, jnp.asarray(0, jnp.int32))


# ---------------------------------------------------------------------------
# T5 — the scan carry still closes under every supported dtype
# ---------------------------------------------------------------------------

def _scan_once(field_dtype, accum_dtype, complex_fields=False):
    n = 12
    box = NTFFBox(i_lo=2, i_hi=n - 2, j_lo=2, j_hi=n - 2, k_lo=2, k_hi=n - 2,
                  freqs=jnp.asarray([FREQ], dtype=jnp.float32),
                  collocation="face_centre", w_x_lo=0.4, w_x_hi=0.6)
    data = init_ntff_data(box, field_dtype=field_dtype)
    arr = jnp.ones((n, n, n), dtype=field_dtype)
    state = FDTDState(ex=arr, ey=arr, ez=arr, hx=arr, hy=arr, hz=arr,
                      step=jnp.asarray(0, dtype=jnp.int32))
    dt = np.float64(1e-12)

    def body(carry, n_idx):
        return accumulate_ntff(carry, state, box, dt, n_idx), None

    out, _ = jax.lax.scan(body, data, jnp.arange(4, dtype=jnp.int32))
    assert out.x_lo.dtype == accum_dtype
    assert np.all(np.isfinite(np.asarray(out.x_lo)))


def test_scan_carry_closes_float32():
    _scan_once(jnp.float32, jnp.complex64)


def test_scan_carry_closes_mixed_precision():
    _scan_once(jnp.float16, jnp.complex64)


def test_scan_carry_closes_complex_bloch_fields():
    _scan_once(jnp.complex64, jnp.complex64)


def test_scan_carry_closes_float64():
    with enable_x64():
        _scan_once(jnp.float64, jnp.complex128)
