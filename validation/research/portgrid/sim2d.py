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


def _interface_coeffs(spec: TwoRegionSpec, eps_fx, eps_fy):
    """Per-edge (ca, cb) for the four interface updates, eq. (61).

    For an interface tangential-E edge with coarse half-cell material
    (eps_c, sigma_c) and fine-side averaged material (eps_hat, sigma_hat)
    (eq. (58)), the update is

      E^{n+1} = ca * E^n + cb * (H_plus - H_minus)

    with H_plus/H_minus the coarse Hz and the segment-mean of the fine
    boundary-row Hz on the appropriate sides, and

      Dp = (eps_c + eps_hat/r)/dt + (sigma_c + sigma_hat/r)/2
      Dm = (eps_c + eps_hat/r)/dt - (sigma_c + sigma_hat/r)/2
      ca = Dm / Dp ,  cb = (2 / delta_n) / Dp

    where delta_n is the coarse cell size normal to the interface.
    Lossless here (sigma = 0 in all M1 fixtures); eps maps may vary.
    """
    r, dt = spec.r, spec.dt
    i0, i1, j0, j1 = spec.i0, spec.i1, spec.j0, spec.j1
    mi = i1 - i0
    mj = j1 - j0

    def coeffs(eps_c, eps_hat, delta_n):
        dp = (eps_c + eps_hat / r) / dt
        ca = ((eps_c + eps_hat / r) / dt) / dp  # = 1 lossless; kept for structure
        cb = (2.0 / delta_n) / dp
        return jnp.asarray(ca), jnp.asarray(cb)

    # eps_hat per coarse interface edge = mean of the r fine-edge values on
    # the boundary row/col (eq. (58)).
    eps_hat_s = eps_fx[:, 0].reshape(mi, r).mean(axis=1)
    eps_hat_n = eps_fx[:, -1].reshape(mi, r).mean(axis=1)
    eps_hat_w = eps_fy[0, :].reshape(mj, r).mean(axis=1)
    eps_hat_e = eps_fy[-1, :].reshape(mj, r).mean(axis=1)

    eps_c = EPS0  # coarse host is vacuum in all M1 fixtures
    ca_s, cb_s = coeffs(eps_c, eps_hat_s, spec.dy)
    ca_n, cb_n = coeffs(eps_c, eps_hat_n, spec.dy)
    ca_w, cb_w = coeffs(eps_c, eps_hat_w, spec.dx)
    ca_e, cb_e = coeffs(eps_c, eps_hat_e, spec.dx)
    return dict(s=(ca_s, cb_s), n=(ca_n, cb_n), w=(ca_w, cb_w), e=(ca_e, cb_e))


def make_stepper(spec: TwoRegionSpec, eps_fx=None, eps_fy=None):
    """Return (step_fn, init_state, aux) for the two-region fixture.

    step_fn(state, src_val) -> (state, energy) advances one leapfrog step:
    H update (+ magnetic-current source on one coarse cell), energy sample
    (staggered storage (25) at time n), then E update (interior + interface
    (61) + replication (55)).
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

    m = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in _masks(spec).items()}
    ifc = _interface_coeffs(spec, eps_fx, eps_fy)

    ch = dt / MU0                       # H update factor (vacuum mu)
    ce_c = dt / EPS0                    # coarse E factor (vacuum host)
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
        energy = (
            0.5 * EPS0 * jnp.sum(m["w_ex"] * ex * ex)
            + 0.5 * EPS0 * jnp.sum(m["w_ey"] * ey * ey)
            + 0.5 * MU0 * jnp.sum(m["w_hz"] * hz * hz_new)
            + 0.5 * jnp.sum(m["w_fex"] * eps_fx * fex * fex)
            + 0.5 * jnp.sum(m["w_fey"] * eps_fy * fey * fey)
            + 0.5 * MU0 * jnp.sum(m["w_fhz"] * fhz * fhz_new)
        )

        # ---- E update: n -> n+1 ----
        # coarse standard interior (padded difference; masks zero the rest)
        dhz_y = jnp.pad(hz_new, ((0, 0), (1, 1)))  # Hz[i, j-1/2] above/below Ex[i, j]
        ex_new = ex + ce_c * m["ex_std"] * (dhz_y[:, 1:] - dhz_y[:, :-1]) / dy
        dhz_x = jnp.pad(hz_new, ((1, 1), (0, 0)))
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
