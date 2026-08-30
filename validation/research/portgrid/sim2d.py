"""M1 two-region 2-D TEz subgridding prototype (JAX, single lax.scan).

Implements the subgridding scheme of arXiv:1606.08761 (IEEE TAP 65(2):751,
2017) for one rectangular fine island embedded in a coarse host:

* unmodified Yee updates strictly inside each grid;
* interface tangential-E update per eq. (61) — coarse-side half-cell plus
  fine-side averaged half-cell (eps_hat / r, sigma_hat / r per eq. (58));
* fine interface tangential E replicated from coarse per eq. (55);
* coarse hanging H replaced by the fine average per eq. (56) (appears as the
  segment-mean of the fine boundary-row Hz in the interface update);
* corners of the island need no special treatment (paper Sec. IV);
* one global dt for both regions (0.99 x fine CFL in all fixtures);
* per-step conserved energy from the staggered storage function (25) with
  region-wise half-cell areas.

State is a pytree of per-region arrays; the whole run is one ``lax.scan``
with static shapes and no conditionals (SPEC-02 §3).  Everything is float64;
callers must be inside a scoped ``enable_x64`` context (or a script that
enables x64 at startup).

Field layout (SI units, physical absolute coordinates):
  coarse Ex[i, j]: i in [0, nx), j in [0, ny]   edge centers (i+1/2, j)
  coarse Ey[i, j]: i in [0, nx], j in [0, ny)   edge centers (i, j+1/2)
  coarse Hz[i, j]: i in [0, nx), j in [0, ny)   cell centers (i+1/2, j+1/2)
  fine arrays identical with (nfx, nfy) = (r * island width, r * island height).

The island covers coarse cells [i0, i1) x [j0, j1).  Coarse samples strictly
inside the island are masked to zero and excluded from the energy sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C0 = 1.0 / np.sqrt(EPS0 * MU0)


@dataclass
class TwoRegionSpec:
    """Static description of the two-region fixture (all SI units)."""

    nx: int
    ny: int
    dx: float
    dy: float
    i0: int
    i1: int
    j0: int
    j1: int
    r: int
    dt: float
    # material maps (vacuum by default); fine eps per fine x-edge/y-edge
    eps_fx: np.ndarray | None = None  # (nfx, nfy+1)
    eps_fy: np.ndarray | None = None  # (nfx+1, nfy)
    # source: magnetic current on coarse Hz cells (single cell, or a mask)
    src_ij: tuple[int, int] = (0, 0)
    src_mask: np.ndarray | None = None  # (nx, ny); overrides src_ij when given
    probe_ij: tuple[int, int] = (0, 0)  # coarse Hz probe
    extra: dict = field(default_factory=dict)

    @property
    def nfx(self) -> int:
        return (self.i1 - self.i0) * self.r

    @property
    def nfy(self) -> int:
        return (self.j1 - self.j0) * self.r


def fine_cfl_dt(spec: TwoRegionSpec) -> float:
    dxf, dyf = spec.dx / spec.r, spec.dy / spec.r
    return np.sqrt(EPS0 * MU0) / np.sqrt(1.0 / dxf**2 + 1.0 / dyf**2)


def _masks(spec: TwoRegionSpec):
    """Precompute update masks and energy area-weights (NumPy, static)."""
    nx, ny, i0, i1, j0, j1, r = spec.nx, spec.ny, spec.i0, spec.i1, spec.j0, spec.j1, spec.r
    if not (1 <= i0 < i1 <= nx - 1 and 1 <= j0 < j1 <= ny - 1):
        # A fine island touching the domain boundary would alias the interface
        # update onto PEC rows/columns (measured: a full-height island acts as
        # a near-total barrier).  Strictly interior islands only.
        raise ValueError(
            f"fine island [{i0},{i1})x[{j0},{j1}) must be strictly interior to "
            f"the {nx}x{ny} coarse grid (>=1 cell of coarse host on every side)")
    dx, dy = spec.dx, spec.dy
    dxf, dyf = dx / r, dy / r

    # Hz: update outside the island only.
    hz_mask = np.ones((nx, ny))
    hz_mask[i0:i1, j0:j1] = 0.0

    # Ex standard-update mask: not domain PEC (j=0, ny), not strictly inside
    # the island (i in [i0,i1), j in (j0, j1)), not on the interface rows.
    ex_std = np.ones((nx, ny + 1))
    ex_std[:, 0] = 0.0
    ex_std[:, ny] = 0.0
    ex_std[i0:i1, j0:j1 + 1] = 0.0
    # Ex interface masks (south row j0, north row j1 within island span)
    ex_ifc_s = np.zeros((nx, ny + 1))
    ex_ifc_s[i0:i1, j0] = 1.0
    ex_ifc_n = np.zeros((nx, ny + 1))
    ex_ifc_n[i0:i1, j1] = 1.0

    # Ey standard-update mask.
    ey_std = np.ones((nx + 1, ny))
    ey_std[0, :] = 0.0
    ey_std[nx, :] = 0.0
    ey_std[i0:i1 + 1, j0:j1] = 0.0
    ey_ifc_w = np.zeros((nx + 1, ny))
    ey_ifc_w[i0, j0:j1] = 1.0
    ey_ifc_e = np.zeros((nx + 1, ny))
    ey_ifc_e[i1, j0:j1] = 1.0

    # ---- energy area weights (paper eq. (25), region-wise) ----
    # Coarse Ex: full dy except half at domain walls and at the coarse side of
    # the interface rows; zero strictly inside the island.
    w_ex = np.full((nx, ny + 1), dx * dy)
    w_ex[:, 0] = dx * dy / 2.0
    w_ex[:, ny] = dx * dy / 2.0
    w_ex[i0:i1, j0] = dx * dy / 2.0
    w_ex[i0:i1, j1] = dx * dy / 2.0
    w_ex[i0:i1, j0 + 1:j1] = 0.0
    w_ey = np.full((nx + 1, ny), dx * dy)
    w_ey[0, :] = dx * dy / 2.0
    w_ey[nx, :] = dx * dy / 2.0
    w_ey[i0, j0:j1] = dx * dy / 2.0
    w_ey[i1, j0:j1] = dx * dy / 2.0
    w_ey[i0 + 1:i1, j0:j1] = 0.0
    w_hz = dx * dy * hz_mask.copy()

    nfx, nfy = spec.nfx, spec.nfy
    w_fex = np.full((nfx, nfy + 1), dxf * dyf)
    w_fex[:, 0] = dxf * dyf / 2.0
    w_fex[:, nfy] = dxf * dyf / 2.0
    w_fey = np.full((nfx + 1, nfy), dxf * dyf)
    w_fey[0, :] = dxf * dyf / 2.0
    w_fey[nfx, :] = dxf * dyf / 2.0
    w_fhz = np.full((nfx, nfy), dxf * dyf)

    return dict(
        hz_mask=hz_mask, ex_std=ex_std, ey_std=ey_std,
        ex_ifc_s=ex_ifc_s, ex_ifc_n=ex_ifc_n, ey_ifc_w=ey_ifc_w, ey_ifc_e=ey_ifc_e,
        w_ex=w_ex, w_ey=w_ey, w_hz=w_hz, w_fex=w_fex, w_fey=w_fey, w_fhz=w_fhz,
    )


def _interface_coeffs(spec: TwoRegionSpec, eps_fx, eps_fy,
                      sigma_fx=None, sigma_fy=None,
                      eps_cx=None, eps_cy=None, sigma_cx=None, sigma_cy=None):
    """Per-edge (ca, cb) for the four interface updates, eq. (61).

    For an interface tangential-E edge with coarse half-cell material
    (eps_c, sigma_c) and fine-side averaged material (eps_hat, sigma_hat)
    (eq. (58) segment means of the fine boundary-row per-edge values), the
    update is

      E^{n+1} = ca * E^n + cb * (H_plus - H_minus)

    with H_plus/H_minus the coarse Hz and the segment-mean of the fine
    boundary-row Hz on the appropriate sides, and

      Dp = (eps_c + eps_hat/r)/dt + (sigma_c + sigma_hat/r)/2
      Dm = (eps_c + eps_hat/r)/dt - (sigma_c + sigma_hat/r)/2
      ca = Dm / Dp ,  cb = (2 / delta_n) / Dp

    where delta_n is the coarse cell size normal to the interface.
    Defaults (sigma = 0, vacuum coarse host) reproduce the lossless
    coefficients exactly (Dm = Dp => ca = 1.0).
    """
    r, dt = spec.r, spec.dt
    i0, i1, j0, j1 = spec.i0, spec.i1, spec.j0, spec.j1
    mi = i1 - i0
    mj = j1 - j0

    if sigma_fx is None:
        sigma_fx = jnp.zeros_like(eps_fx)
    if sigma_fy is None:
        sigma_fy = jnp.zeros_like(eps_fy)

    def coeffs(eps_c, sig_c, eps_hat, sig_hat, delta_n):
        dp = (eps_c + eps_hat / r) / dt + (sig_c + sig_hat / r) / 2.0
        dm = (eps_c + eps_hat / r) / dt - (sig_c + sig_hat / r) / 2.0
        ca = dm / dp
        cb = (2.0 / delta_n) / dp
        return jnp.asarray(ca), jnp.asarray(cb)

    def seg(v):  # eq. (58): r-segment mean along the boundary row/col
        return v.reshape(-1, r).mean(axis=1)

    # coarse host material AT the interface edges (vacuum lossless default)
    def cvals(arr, default, sl):
        if arr is None:
            n = mi if sl[0] == "x" else mj
            return jnp.full((n,), default)
        if sl[0] == "x":
            return jnp.asarray(arr)[i0:i1, sl[1]]
        return jnp.asarray(arr)[sl[1], j0:j1]

    ec_s = cvals(eps_cx, EPS0, ("x", j0));  sc_s = cvals(sigma_cx, 0.0, ("x", j0))
    ec_n = cvals(eps_cx, EPS0, ("x", j1));  sc_n = cvals(sigma_cx, 0.0, ("x", j1))
    ec_w = cvals(eps_cy, EPS0, ("y", i0));  sc_w = cvals(sigma_cy, 0.0, ("y", i0))
    ec_e = cvals(eps_cy, EPS0, ("y", i1));  sc_e = cvals(sigma_cy, 0.0, ("y", i1))

    ca_s, cb_s = coeffs(ec_s, sc_s, seg(eps_fx[:, 0]), seg(sigma_fx[:, 0]), spec.dy)
    ca_n, cb_n = coeffs(ec_n, sc_n, seg(eps_fx[:, -1]), seg(sigma_fx[:, -1]), spec.dy)
    ca_w, cb_w = coeffs(ec_w, sc_w, seg(eps_fy[0, :]), seg(sigma_fy[0, :]), spec.dx)
    ca_e, cb_e = coeffs(ec_e, sc_e, seg(eps_fy[-1, :]), seg(sigma_fy[-1, :]), spec.dx)
    return dict(s=(ca_s, cb_s), n=(ca_n, cb_n), w=(ca_w, cb_w), e=(ca_e, cb_e))


def _lossy_e_coeffs(eps, sigma, dt):
    """Standard lossy-Yee E coefficients: ca = (eps/dt - sig/2)/(eps/dt + sig/2),
    cb = (dt/eps)/(1 + sig*dt/(2*eps)).  sigma = 0 gives ca = 1.0 and
    cb = dt/eps exactly (float division by 1.0 is exact)."""
    x = sigma * dt / (2.0 * eps)
    ca = (1.0 - x) / (1.0 + x)
    cb = (dt / eps) / (1.0 + x)
    return ca, cb


def make_stepper(spec: TwoRegionSpec, eps_fx=None, eps_fy=None, *,
                 sigma_fx=None, sigma_fy=None,
                 eps_cx=None, eps_cy=None, sigma_cx=None, sigma_cy=None):
    """Return (step_fn, init_state, aux) for the two-region fixture.

    step_fn(state, src_val) -> (state, energy) advances one leapfrog step:
    H update (+ magnetic-current source on one coarse cell), energy sample
    (staggered storage (25) at time n), then E update (interior + interface
    (61) + replication (55)).

    Optional per-edge material maps (all default to the vacuum/lossless
    values, reproducing the original coefficients exactly):
      sigma_fx (nfx, nfy+1), sigma_fy (nfx+1, nfy)   fine conductivity
      eps_cx (nx, ny+1), eps_cy (nx+1, ny)           coarse host permittivity
      sigma_cx (nx, ny+1), sigma_cy (nx+1, ny)       coarse host conductivity
    The interface update takes the eq. (58) segment means of the fine
    boundary-row eps/sigma and the coarse host values at the interface edges
    (full eq. (61) coefficients).
    """
    nx, ny, r = spec.nx, spec.ny, spec.r
    i0, i1, j0, j1 = spec.i0, spec.i1, spec.j0, spec.j1
    nfx, nfy = spec.nfx, spec.nfy
    dx, dy, dt = spec.dx, spec.dy, spec.dt
    dxf, dyf = dx / r, dy / r

    if eps_fx is None:
        eps_fx = np.full((nfx, nfy + 1), EPS0)
    if eps_fy is None:
        eps_fy = np.full((nfx + 1, nfy), EPS0)
    eps_fx = jnp.asarray(eps_fx, dtype=jnp.float64)
    eps_fy = jnp.asarray(eps_fy, dtype=jnp.float64)
    if sigma_fx is not None:
        sigma_fx = jnp.asarray(sigma_fx, dtype=jnp.float64)
    if sigma_fy is not None:
        sigma_fy = jnp.asarray(sigma_fy, dtype=jnp.float64)

    m = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in _masks(spec).items()}
    ifc = _interface_coeffs(spec, eps_fx, eps_fy, sigma_fx, sigma_fy,
                            eps_cx, eps_cy, sigma_cx, sigma_cy)

    ch = dt / MU0                       # H update factor (vacuum mu)
    ce_c = dt / EPS0                    # coarse E factor (vacuum host)
    # generalized coarse/fine E coefficients (identical to the scalars above
    # when every optional map is None)
    have_mats = any(v is not None for v in
                    (sigma_fx, sigma_fy, eps_cx, eps_cy, sigma_cx, sigma_cy))
    if have_mats:
        _ecx = jnp.full((nx, ny + 1), EPS0) if eps_cx is None else jnp.asarray(eps_cx, dtype=jnp.float64)
        _ecy = jnp.full((nx + 1, ny), EPS0) if eps_cy is None else jnp.asarray(eps_cy, dtype=jnp.float64)
        _scx = jnp.zeros((nx, ny + 1)) if sigma_cx is None else jnp.asarray(sigma_cx, dtype=jnp.float64)
        _scy = jnp.zeros((nx + 1, ny)) if sigma_cy is None else jnp.asarray(sigma_cy, dtype=jnp.float64)
        _sfx = jnp.zeros((nfx, nfy + 1)) if sigma_fx is None else sigma_fx
        _sfy = jnp.zeros((nfx + 1, nfy)) if sigma_fy is None else sigma_fy
        ca_cx, cb_cx = _lossy_e_coeffs(_ecx, _scx, dt)
        ca_cy, cb_cy = _lossy_e_coeffs(_ecy, _scy, dt)
        ca_fx, cb_fx = _lossy_e_coeffs(eps_fx, _sfx, dt)
        ca_fy, cb_fy = _lossy_e_coeffs(eps_fy, _sfy, dt)
    pi, pj = spec.probe_ij
    if spec.src_mask is not None:
        src_mask = jnp.asarray(spec.src_mask, dtype=jnp.float64)
    else:
        sm = np.zeros((nx, ny))
        sm[spec.src_ij] = 1.0
        src_mask = jnp.asarray(sm)

    def seg_mean(v):
        return v.reshape(-1, r).mean(axis=1)

    def step(state, src_val):
        ex, ey, hz, fex, fey, fhz = (
            state["ex"], state["ey"], state["hz"],
            state["fex"], state["fey"], state["fhz"],
        )

        # ---- H update: n-1/2 -> n+1/2 (unmodified Yee, masked) ----
        curl_e = (ex[:, 1:] - ex[:, :-1]) / dy - (ey[1:, :] - ey[:-1, :]) / dx
        hz_new = hz + ch * m["hz_mask"] * curl_e
        hz_new = hz_new + src_val * src_mask     # magnetic-current source
        fcurl_e = (fex[:, 1:] - fex[:, :-1]) / dyf - (fey[1:, :] - fey[:-1, :]) / dxf
        fhz_new = fhz + ch * fcurl_e

        # ---- energy at time n (staggered storage, eq. (25)) ----
        e_wx = _ecx if have_mats else EPS0
        e_wy = _ecy if have_mats else EPS0
        energy = (
            0.5 * jnp.sum(e_wx * m["w_ex"] * ex * ex)
            + 0.5 * jnp.sum(e_wy * m["w_ey"] * ey * ey)
            + 0.5 * MU0 * jnp.sum(m["w_hz"] * hz * hz_new)
            + 0.5 * jnp.sum(m["w_fex"] * eps_fx * fex * fex)
            + 0.5 * jnp.sum(m["w_fey"] * eps_fy * fey * fey)
            + 0.5 * MU0 * jnp.sum(m["w_fhz"] * fhz * fhz_new)
        )

        # ---- E update: n -> n+1 ----
        # coarse standard interior (padded difference; masks zero the rest)
        dhz_y = jnp.pad(hz_new, ((0, 0), (1, 1)))  # Hz[i, j-1/2] above/below Ex[i, j]
        dhz_x = jnp.pad(hz_new, ((1, 1), (0, 0)))
        if have_mats:
            ex_new = ex + m["ex_std"] * (
                (ca_cx - 1.0) * ex + cb_cx * (dhz_y[:, 1:] - dhz_y[:, :-1]) / dy)
            ey_new = ey + m["ey_std"] * (
                (ca_cy - 1.0) * ey + cb_cy * (dhz_x[:-1, :] - dhz_x[1:, :]) / dx)
            fex_new = fex.at[:, 1:-1].set(
                ca_fx[:, 1:-1] * fex[:, 1:-1]
                + cb_fx[:, 1:-1] * (fhz_new[:, 1:] - fhz_new[:, :-1]) / dyf)
            fey_new = fey.at[1:-1, :].set(
                ca_fy[1:-1, :] * fey[1:-1, :]
                + cb_fy[1:-1, :] * (fhz_new[:-1, :] - fhz_new[1:, :]) / dxf)
        else:
            ex_new = ex + ce_c * m["ex_std"] * (dhz_y[:, 1:] - dhz_y[:, :-1]) / dy
            ey_new = ey + ce_c * m["ey_std"] * (dhz_x[:-1, :] - dhz_x[1:, :]) / dx

            # fine standard interior
            fex_new = fex.at[:, 1:-1].set(
                fex[:, 1:-1]
                + (dt / eps_fx[:, 1:-1]) * (fhz_new[:, 1:] - fhz_new[:, :-1]) / dyf
            )
            fey_new = fey.at[1:-1, :].set(
                fey[1:-1, :]
                + (dt / eps_fy[1:-1, :]) * (fhz_new[:-1, :] - fhz_new[1:, :]) / dxf
            )

        # ---- interface updates, eq. (61):  E' = ca E + cb (H_plus - H_minus)
        # south (island south face): H_plus = fine mean above, H_minus = coarse below
        ca, cb = ifc["s"]
        e_s = ca * ex[i0:i1, j0] + cb * (seg_mean(fhz_new[:, 0]) - hz_new[i0:i1, j0 - 1])
        ex_new = ex_new.at[i0:i1, j0].set(e_s)
        # north: H_plus = coarse above, H_minus = fine mean below
        ca, cb = ifc["n"]
        e_n = ca * ex[i0:i1, j1] + cb * (hz_new[i0:i1, j1] - seg_mean(fhz_new[:, -1]))
        ex_new = ex_new.at[i0:i1, j1].set(e_n)
        # west (Ey): E' = ca E + cb (H_west - H_east); H_west coarse, H_east fine mean
        ca, cb = ifc["w"]
        e_w = ca * ey[i0, j0:j1] + cb * (hz_new[i0 - 1, j0:j1] - seg_mean(fhz_new[0, :]))
        ey_new = ey_new.at[i0, j0:j1].set(e_w)
        # east: H_west fine mean, H_east coarse
        ca, cb = ifc["e"]
        e_e = ca * ey[i1, j0:j1] + cb * (seg_mean(fhz_new[-1, :]) - hz_new[i1, j0:j1])
        ey_new = ey_new.at[i1, j0:j1].set(e_e)

        # ---- replication (55): fine interface tangential E from coarse ----
        fex_new = fex_new.at[:, 0].set(jnp.repeat(e_s, r))
        fex_new = fex_new.at[:, -1].set(jnp.repeat(e_n, r))
        fey_new = fey_new.at[0, :].set(jnp.repeat(e_w, r))
        fey_new = fey_new.at[-1, :].set(jnp.repeat(e_e, r))

        new_state = dict(ex=ex_new, ey=ey_new, hz=hz_new,
                         fex=fex_new, fey=fey_new, fhz=fhz_new)
        return new_state, (energy, hz_new[pi, pj])

    def init_state():
        z = partial(jnp.zeros, dtype=jnp.float64)
        return dict(
            ex=z((nx, ny + 1)), ey=z((nx + 1, ny)), hz=z((nx, ny)),
            fex=z((nfx, nfy + 1)), fey=z((nfx + 1, nfy)), fhz=z((nfx, nfy)),
        )

    return step, init_state, dict(masks=m, ifc=ifc, eps_fx=eps_fx, eps_fy=eps_fy)


def run_scan(step, state, waveform):
    """Single lax.scan over the whole run; returns (state, energies, probe)."""
    state, (energies, probe) = jax.lax.scan(step, state, waveform)
    return state, energies, probe


def make_uniform_stepper(nx: int, ny: int, dx: float, dy: float, dt: float,
                         src_mask: np.ndarray, probe_ij: tuple[int, int]):
    """Reference: plain uniform coarse Yee grid, PEC box, vacuum (no island)."""
    src = jnp.asarray(src_mask, dtype=jnp.float64)
    ch = dt / MU0
    ce = dt / EPS0
    pi, pj = probe_ij

    def step(state, src_val):
        ex, ey, hz = state["ex"], state["ey"], state["hz"]
        curl_e = (ex[:, 1:] - ex[:, :-1]) / dy - (ey[1:, :] - ey[:-1, :]) / dx
        hz_new = hz + ch * curl_e + src_val * src
        ex_new = ex.at[:, 1:-1].set(ex[:, 1:-1] + ce * (hz_new[:, 1:] - hz_new[:, :-1]) / dy)
        ey_new = ey.at[1:-1, :].set(ey[1:-1, :] + ce * (hz_new[:-1, :] - hz_new[1:, :]) / dx)
        energy = (
            0.5 * EPS0 * dx * dy * (jnp.sum(ex * ex) - 0.5 * jnp.sum(ex[:, 0] ** 2)
                                    - 0.5 * jnp.sum(ex[:, -1] ** 2))
            + 0.5 * EPS0 * dx * dy * (jnp.sum(ey * ey) - 0.5 * jnp.sum(ey[0, :] ** 2)
                                      - 0.5 * jnp.sum(ey[-1, :] ** 2))
            + 0.5 * MU0 * dx * dy * jnp.sum(hz * hz_new)
        )
        return dict(ex=ex_new, ey=ey_new, hz=hz_new), (energy, hz_new[pi, pj])

    def init_state():
        z = partial(jnp.zeros, dtype=jnp.float64)
        return dict(ex=z((nx, ny + 1)), ey=z((nx + 1, ny)), hz=z((nx, ny)))

    return step, init_state


# ---------------------------------------------------------------------------
# Terminated (PML) steppers for the F-M1b retry fixture (paper Fig. 8 class).
# Separate builders on purpose: the M1a-verified PEC steppers above are not
# touched.  Split-field Berenger PML graded along x only (PEC walls in y),
# outer wall PEC; Hz = Hzx + Hzy inside these steppers only.
# ---------------------------------------------------------------------------

def pml_profiles(nx: int, dx: float, npml: int, m_pml: float = 3.0,
                 r0: float = 1e-5):
    """x-graded PML conductivity profiles (SI) for a grid with nx cells.

    Returns (sig_e, sig_h): sig_e on the Ey planes x = i*dx (shape nx+1),
    sig_h on the Hz planes x = (i+1/2)*dx (shape nx).  Polynomial grading of
    order m_pml over npml cells at both ends, design reflection r0:
    sigma_max = -(m+1) * eps0 * c * ln(r0) / (2 * d),  d = npml*dx.
    The matched magnetic conductivity is sigma* = sig_h * mu0/eps0 (applied
    inside the steppers via x = sig*dt/(2*eps0) for both species).
    """
    d = npml * dx
    smax = -(m_pml + 1.0) * EPS0 * C0 * np.log(r0) / (2.0 * d)

    def depth(pos):  # pos in cells from the left domain edge
        dl = np.maximum(0.0, npml - pos) / npml
        dr = np.maximum(0.0, pos - (nx - npml)) / npml
        return np.maximum(dl, dr)

    sig_e = smax * depth(np.arange(nx + 1, dtype=float)) ** m_pml
    sig_h = smax * depth(np.arange(nx, dtype=float) + 0.5) ** m_pml
    return sig_e, sig_h


def disk_sigma_maps(nx: int, ny: int, dx: float, dy: float,
                    origin: tuple[float, float],
                    centers: list[tuple[float, float]], radius: float,
                    sigma: float):
    """Per-edge conductivity maps for circular rods (SI absolute coords).

    Edge centers: Ex[i,j] at (origin_x+(i+1/2)dx, origin_y+j*dy);
    Ey[i,j] at (origin_x+i*dx, origin_y+(j+1/2)dy).  An edge whose center
    lies inside any disk gets `sigma`.
    Returns (sig_x (nx,ny+1), sig_y (nx+1,ny)).
    """
    ox, oy = origin
    xs_x = ox + (np.arange(nx) + 0.5) * dx
    ys_x = oy + np.arange(ny + 1) * dy
    xs_y = ox + np.arange(nx + 1) * dx
    ys_y = oy + (np.arange(ny) + 0.5) * dy
    sig_x = np.zeros((nx, ny + 1))
    sig_y = np.zeros((nx + 1, ny))
    for cx, cy in centers:
        sig_x[(xs_x[:, None] - cx) ** 2 + (ys_x[None, :] - cy) ** 2 <= radius**2] = sigma
        sig_y[(xs_y[:, None] - cx) ** 2 + (ys_y[None, :] - cy) ** 2 <= radius**2] = sigma
    return sig_x, sig_y


def make_stepper_pml(spec: TwoRegionSpec, *, src_col: int, probe_col: int,
                     npml: int = 15, m_pml: float = 3.0, r0: float = 1e-5,
                     eps_fx=None, eps_fy=None, sigma_fx=None, sigma_fy=None,
                     probe_full: bool = False):
    """Two-region stepper with x-PML termination, Jy column source, and an Ey
    probe on ``probe_col`` (the retry fixture).  Coarse host is vacuum outside
    the PML strips.  step(state, src_val) -> (state, probe_val).

    ``probe_full`` selects the PROJECTION applied to the probe column, and it
    is a physics choice, not a convenience (retry pre-declaration Correction
    R3):

    * ``False`` (default, unchanged): the y-MEAN of Ey over the column.  With
      PEC plates at y = 0, H this is the TEM (n = 0) modal amplitude alone --
      the projection integral (1/H)∫Ey dy annihilates every cos(nπy/H), n >= 1.
      A y-uniform source launches only TEM, but an interface or a scatterer
      converts TEM into higher-order content, so an |S11| built on this probe
      is a TEM->TEM reflection coefficient and is BLIND to mode-converted
      reflected energy.
    * ``True``: the whole Ey column, shape (ny,), so the caller can apply any
      projection (y-mean, a point row, a modal overlap) to the SAME run.

    The default keeps every previously committed measurement bit-identical.
    """
    nx, ny, r = spec.nx, spec.ny, spec.r
    i0, i1, j0, j1 = spec.i0, spec.i1, spec.j0, spec.j1
    nfx, nfy = spec.nfx, spec.nfy
    dx, dy, dt = spec.dx, spec.dy, spec.dt
    dxf, dyf = dx / r, dy / r
    if not (npml < i0 and i1 < nx - npml):
        raise ValueError("fine island must not overlap the PML strips")
    if not (npml < src_col < nx - npml and npml < probe_col < nx - npml):
        raise ValueError("source/probe columns must be outside the PML strips")

    if eps_fx is None:
        eps_fx = np.full((nfx, nfy + 1), EPS0)
    if eps_fy is None:
        eps_fy = np.full((nfx + 1, nfy), EPS0)
    eps_fx = jnp.asarray(eps_fx, dtype=jnp.float64)
    eps_fy = jnp.asarray(eps_fy, dtype=jnp.float64)
    sfx = jnp.zeros((nfx, nfy + 1)) if sigma_fx is None else jnp.asarray(sigma_fx, dtype=jnp.float64)
    sfy = jnp.zeros((nfx + 1, nfy)) if sigma_fy is None else jnp.asarray(sigma_fy, dtype=jnp.float64)

    m = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in _masks(spec).items()}
    ifc = _interface_coeffs(spec, eps_fx, eps_fy, sfx, sfy)

    sig_e, sig_h = pml_profiles(nx, dx, npml, m_pml, r0)
    xh = jnp.asarray(sig_h) * dt / (2.0 * EPS0)          # matched sigma*
    da_h = ((1.0 - xh) / (1.0 + xh))[:, None]
    db_h = ((dt / MU0) / (1.0 + xh))[:, None]
    xe = jnp.asarray(sig_e) * dt / (2.0 * EPS0)
    ca_ey = ((1.0 - xe) / (1.0 + xe))[:, None]
    cb_ey = ((dt / EPS0) / (1.0 + xe))[:, None]
    ce_c = dt / EPS0
    ch = dt / MU0
    ca_fx, cb_fx = _lossy_e_coeffs(eps_fx, sfx, dt)
    ca_fy, cb_fy = _lossy_e_coeffs(eps_fy, sfy, dt)

    def seg_mean(v):
        return v.reshape(-1, r).mean(axis=1)

    def step(state, src_val):
        ex, ey, hzx, hzy = state["ex"], state["ey"], state["hzx"], state["hzy"]
        fex, fey, fhz = state["fex"], state["fey"], state["fhz"]

        # ---- H update (split field; island hole masked) ----
        hzx_new = da_h * hzx - m["hz_mask"] * db_h * (ey[1:, :] - ey[:-1, :]) / dx
        hzy_new = hzy + m["hz_mask"] * ch * (ex[:, 1:] - ex[:, :-1]) / dy
        hz_new = hzx_new + hzy_new
        fcurl_e = (fex[:, 1:] - fex[:, :-1]) / dyf - (fey[1:, :] - fey[:-1, :]) / dxf
        fhz_new = fhz + ch * fcurl_e

        # ---- E update ----
        dhz_y = jnp.pad(hz_new, ((0, 0), (1, 1)))
        dhz_x = jnp.pad(hz_new, ((1, 1), (0, 0)))
        ex_new = jnp.where(
            m["ex_std"] > 0,
            ex + ce_c * (dhz_y[:, 1:] - dhz_y[:, :-1]) / dy, ex)
        ey_new = jnp.where(
            m["ey_std"] > 0,
            ca_ey * ey + cb_ey * (dhz_x[:-1, :] - dhz_x[1:, :]) / dx, ey)
        ey_new = ey_new.at[src_col, :].add(src_val)   # Jy line source

        fex_new = fex.at[:, 1:-1].set(
            ca_fx[:, 1:-1] * fex[:, 1:-1]
            + cb_fx[:, 1:-1] * (fhz_new[:, 1:] - fhz_new[:, :-1]) / dyf)
        fey_new = fey.at[1:-1, :].set(
            ca_fy[1:-1, :] * fey[1:-1, :]
            + cb_fy[1:-1, :] * (fhz_new[:-1, :] - fhz_new[1:, :]) / dxf)

        # ---- interface updates, eq. (61) ----
        ca, cb = ifc["s"]
        e_s = ca * ex[i0:i1, j0] + cb * (seg_mean(fhz_new[:, 0]) - hz_new[i0:i1, j0 - 1])
        ex_new = ex_new.at[i0:i1, j0].set(e_s)
        ca, cb = ifc["n"]
        e_n = ca * ex[i0:i1, j1] + cb * (hz_new[i0:i1, j1] - seg_mean(fhz_new[:, -1]))
        ex_new = ex_new.at[i0:i1, j1].set(e_n)
        ca, cb = ifc["w"]
        e_w = ca * ey[i0, j0:j1] + cb * (hz_new[i0 - 1, j0:j1] - seg_mean(fhz_new[0, :]))
        ey_new = ey_new.at[i0, j0:j1].set(e_w)
        ca, cb = ifc["e"]
        e_e = ca * ey[i1, j0:j1] + cb * (seg_mean(fhz_new[-1, :]) - hz_new[i1, j0:j1])
        ey_new = ey_new.at[i1, j0:j1].set(e_e)

        # ---- replication (55) ----
        fex_new = fex_new.at[:, 0].set(jnp.repeat(e_s, r))
        fex_new = fex_new.at[:, -1].set(jnp.repeat(e_n, r))
        fey_new = fey_new.at[0, :].set(jnp.repeat(e_w, r))
        fey_new = fey_new.at[-1, :].set(jnp.repeat(e_e, r))

        new_state = dict(ex=ex_new, ey=ey_new, hzx=hzx_new, hzy=hzy_new,
                         fex=fex_new, fey=fey_new, fhz=fhz_new)
        probe = (ey_new[probe_col, :] if probe_full
                 else jnp.mean(ey_new[probe_col, :]))
        return new_state, probe

    def init_state():
        z = partial(jnp.zeros, dtype=jnp.float64)
        return dict(
            ex=z((nx, ny + 1)), ey=z((nx + 1, ny)),
            hzx=z((nx, ny)), hzy=z((nx, ny)),
            fex=z((nfx, nfy + 1)), fey=z((nfx + 1, nfy)), fhz=z((nfx, nfy)),
        )

    return step, init_state, dict(masks=m, ifc=ifc, sig_e=sig_e, sig_h=sig_h)


def make_uniform_pml(nx: int, ny: int, dx: float, dy: float, dt: float, *,
                     src_col: int, probe_col: int, npml: int = 15,
                     m_pml: float = 3.0, r0: float = 1e-5,
                     eps_x=None, eps_y=None, sigma_x=None, sigma_y=None,
                     probe_full: bool = False):
    """Uniform-grid stepper with x-PML, Jy column source, an Ey probe on
    ``probe_col``, optional per-edge materials (rods).  PEC at y walls and
    behind the PML.  step(state, src_val) -> (state, probe_val).

    ``probe_full`` has the same meaning as in :func:`make_stepper_pml`:
    False (default) returns the y-MEAN of the column -- the TEM-only
    projection -- and True returns the whole (ny,) column so the caller can
    project it itself.  Default keeps prior measurements bit-identical."""
    if eps_x is None:
        eps_x = np.full((nx, ny + 1), EPS0)
    if eps_y is None:
        eps_y = np.full((nx + 1, ny), EPS0)
    eps_x = jnp.asarray(eps_x, dtype=jnp.float64)
    eps_y = jnp.asarray(eps_y, dtype=jnp.float64)
    sx = jnp.zeros((nx, ny + 1)) if sigma_x is None else jnp.asarray(sigma_x, dtype=jnp.float64)
    sy = jnp.zeros((nx + 1, ny)) if sigma_y is None else jnp.asarray(sigma_y, dtype=jnp.float64)

    sig_e, sig_h = pml_profiles(nx, dx, npml, m_pml, r0)
    xh = jnp.asarray(sig_h) * dt / (2.0 * EPS0)
    da_h = ((1.0 - xh) / (1.0 + xh))[:, None]
    db_h = ((dt / MU0) / (1.0 + xh))[:, None]
    # Ey: PML sigma_x adds to any material sigma on the same edges
    sy_tot = sy + jnp.asarray(sig_e)[:, None]
    ca_x, cb_x = _lossy_e_coeffs(eps_x, sx, dt)
    ca_y, cb_y = _lossy_e_coeffs(eps_y, sy_tot, dt)
    ch = dt / MU0

    def step(state, src_val):
        ex, ey, hzx, hzy = state["ex"], state["ey"], state["hzx"], state["hzy"]
        hzx_new = da_h * hzx - db_h * (ey[1:, :] - ey[:-1, :]) / dx
        hzy_new = hzy + ch * (ex[:, 1:] - ex[:, :-1]) / dy
        hz_new = hzx_new + hzy_new
        ex_new = ex.at[:, 1:-1].set(
            ca_x[:, 1:-1] * ex[:, 1:-1]
            + cb_x[:, 1:-1] * (hz_new[:, 1:] - hz_new[:, :-1]) / dy)
        ey_new = ey.at[1:-1, :].set(
            ca_y[1:-1, :] * ey[1:-1, :]
            + cb_y[1:-1, :] * (hz_new[:-1, :] - hz_new[1:, :]) / dx)
        ey_new = ey_new.at[src_col, :].add(src_val)
        probe = (ey_new[probe_col, :] if probe_full
                 else jnp.mean(ey_new[probe_col, :]))
        return (dict(ex=ex_new, ey=ey_new, hzx=hzx_new, hzy=hzy_new), probe)

    def init_state():
        z = partial(jnp.zeros, dtype=jnp.float64)
        return dict(ex=z((nx, ny + 1)), ey=z((nx + 1, ny)),
                    hzx=z((nx, ny)), hzy=z((nx, ny)))

    return step, init_state, dict(sig_e=sig_e, sig_h=sig_h)


def gaussian_modulated(n_steps: int, dt: float, f0: float, hwhm_bw: float,
                       amplitude: float = 1.0) -> np.ndarray:
    """Modulated Gaussian with COMPACT support (identically zero after cutoff).

    hwhm_bw is the half-width-at-half-maximum bandwidth of the spectrum
    (paper Sec. V-A: f0 = 3.75 GHz, HWHM = 0.74 GHz).  For a Gaussian
    envelope exp(-t^2 / (2 tau^2)) the spectral HWHM satisfies
    2*pi*f_hwhm = sqrt(2 ln 2) / tau.
    """
    tau = np.sqrt(2.0 * np.log(2.0)) / (2.0 * np.pi * hwhm_bw)
    t_c = 6.0 * tau
    t = np.arange(n_steps) * dt
    env = np.exp(-((t - t_c) ** 2) / (2.0 * tau**2))
    w = amplitude * env * np.sin(2.0 * np.pi * f0 * (t - t_c))
    w[t > 2.0 * t_c] = 0.0  # compact support: exactly zero after 12 tau
    return w
