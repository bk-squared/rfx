"""Printable GRIN lens: stages governed by the 20261004 pre-declaration.

Free indices run outward from the axis in x/y and upward from the lens bottom.
The physical domain is 126 x 126 x 120 mm; CPML is outside that domain.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lens_filter_common as common

CASE = "lens"
MESHES = (1.5e-3, 1e-3, .75e-3)
DESIGN_DX = 1e-3
SMOKE_DX = 3e-3
SHAPE = (15, 15, 10)
FREQS = np.array([9.5e9, 10e9, 10.5e9])
FD_PIXELS = {"x1.5_y1.5_layer5": (0, 0, 4), "x22.5_y1.5_layer5": (7, 0, 4),
             "x40.5_y1.5_layer5": (13, 0, 4)}
UPPER = 2.7
ITERATIONS = 150
LR = (.10, .01)


def expand_pixels(free, dx, xp=np):
    """Half-open 3 mm voxels; the two central voxels are mirrored partners."""
    n = common.integral_cells(.003, dx)
    full = xp.concatenate((free[::-1, :, :], free), axis=0)
    full = xp.concatenate((full[:, ::-1, :], full), axis=1)
    for axis in range(3):
        full = xp.repeat(full, n, axis=axis)
    return full


def textbook_grin(r):
    n = np.sqrt(2.7) - (np.sqrt(.045 ** 2 + np.asarray(r) ** 2) - .045) / .030
    return np.clip(n ** 2, 1., 2.7)


def starts():
    xy = (np.arange(15) + .5) * .003
    r = np.hypot(xy[:, None], xy[None, :])
    return {"primary": np.full(SHAPE, 1.85, np.float32),
            "grin": np.broadcast_to(textbook_grin(r)[:, :, None], SHAPE).astype(np.float32)}


def baselines():
    return {"no_lens": np.ones(SHAPE, np.float32),
            **{f"uniform_{e:g}": np.full(SHAPE, e, np.float32) for e in (1., 1.85, 2.7)},
            "grin": starts()["grin"]}


def loss(response, objective="boresight"):
    if objective == "fan":
        return fan_loss(response)
    if objective == "wide":
        return wide_loss(response)
    if objective != "boresight":
        return mask_loss(response, objective)

    import jax.numpy as jnp
    return -jnp.mean(jnp.log(jnp.maximum(response[:, 0, :].mean(axis=1), 1e-30)))


def summary(response, objective="boresight"):
    if objective == "fan":
        return fan_summary(response)
    if objective == "wide":
        return wide_summary(response)
    if objective != "boresight":
        return mask_summary(response, objective)

    d = np.asarray(response)
    bore = 10 * np.log10(np.maximum(d[:, 0, :].mean(axis=1), 1e-30))
    bound = 10 * np.log10(4 * np.pi * .09 ** 2 / (299792458. / FREQS) ** 2)
    return {"boresight_dbi": bore, "mean_boresight_dbi": float(bore.mean()),
            "aperture_bound_dbi": bound, "above_aperture_bound": bore > bound}


THETA = np.r_[0., np.linspace(1e-4, np.pi - 1e-4, 73)[1:]]
PHI = np.linspace(0., 2 * np.pi, 73)
# Regions by the grid's nominal angles (multiples of 2.5 degrees): the row at
# nominal 12.5 degrees sits 5e-5 degrees above it and belongs to C (Amendment 5
# clarification).
NOMINAL_DEG = 2.5 * np.arange(73)
REGIONS = {"S": NOMINAL_DEG >= 25, "C": NOMINAL_DEG <= 12.5, "O": NOMINAL_DEG >= 32.5}
# Sector half-angle of objective W. Amendment 6 declares 30 degrees with its two
# judged levels; another half-angle needs its own amendment (the judged levels
# below belong to 30 degrees only).
WIDE_HALF_DEG = float(os.environ.get("RFX_DESIGN_SECTOR_DEG", "30"))
if WIDE_HALF_DEG % 2.5 or not 10 <= WIDE_HALF_DEG <= 60:
    raise ValueError("RFX_DESIGN_SECTOR_DEG must be a multiple of 2.5 between 10 and 60")
WIDE_REGION = NOMINAL_DEG <= WIDE_HALF_DEG
WIDE_OUTSIDE = NOMINAL_DEG >= WIDE_HALF_DEG + 30
FAN_ROWS = np.flatnonzero(NOMINAL_DEG <= 30)
FAN_COLS = np.array([17, 18, 19, 53, 54, 55])


def objective_spec(objective):
    if objective == "boresight":
        return {"objective": objective}
    if objective == "fan":
        return {"objective": objective,
                "objective_words": "Fan beam (flat over ±30° in one plane, narrow in the other)",
                "tau_db": .2, "judged": {"ripple_max_db": 2., "level_min_dbi": 11.,
                                         "narrow_hpbw_max_deg": 23.},
                "linear_floor": 1e-30,
                "regions": {"S_N": {"predicate": "nominal theta <= 30 degrees; phi in 85, 90, 95, 265, 270, 275 degrees",
                                      "theta_indices": FAN_ROWS, "theta_rad": THETA[FAN_ROWS],
                                      "phi_indices": FAN_COLS, "directions": 78}},
                "phi_endpoint": "exclude column 72 (2pi); retain column 0; solver quadrature unchanged"}
    if objective == "wide":
        region = WIDE_REGION
        return {"objective": objective, "objective_words": f"Wide flat-top beam (±{WIDE_HALF_DEG:g}°)",
                "sector_half_deg": WIDE_HALF_DEG,
                "tau_db": .5, "judged": {"ripple_max_db": 2., "level_min_dbi": 7.5},
                "linear_floor": 1e-30,
                "regions": {"C_W": {"predicate": f"nominal theta <= {WIDE_HALF_DEG:g} degrees",
                                      "theta_indices": np.flatnonzero(region),
                                      "theta_rad": THETA[region], "phi_indices": np.arange(72),
                                      "directions": int(region.sum() * 72)}},
                "phi_endpoint": "exclude column 72 (2pi); retain column 0; solver quadrature unchanged"}
    if objective not in ("mask", "flattop"):
        raise ValueError(f"unknown lens objective {objective}")
    names = ("S",) if objective == "mask" else ("C", "O")
    return {"objective": objective, "objective_words": (
        "Pencil beam under a flat sidelobe mask" if objective == "mask" else "Flat-top beam"),
        "mask_dbi": ({"boresight_min": 18., "outer_max": -2.} if objective == "mask"
                     else {"window_min": 13., "window_max": 15., "outer_max": 0.}),
        "aim_dbi": ({"boresight_min": 19., "outer_max": -3.} if objective == "mask"
                    else {"window_min": 13.5, "window_max": 14.5, "outer_max": -1.}),
        "delta_db": .1, "progress_epsilon_db2": 1e-3, "linear_floor": 1e-30,
        "regions": {name: {"predicate": {"S": "nominal theta >= 25 degrees", "C": "nominal theta <= 12.5 degrees",
                                         "O": "nominal theta >= 32.5 degrees"}[name], "theta_indices": np.flatnonzero(REGIONS[name]),
                            "theta_rad": THETA[REGIONS[name]], "phi_indices": np.arange(72),
                            "directions": int(REGIONS[name].sum() * 72)} for name in names},
        "phi_endpoint": "exclude column 72 (2pi); retain column 0; solver quadrature unchanged"}


def wide_loss(response):
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    d = 10 * jnp.log10(jnp.maximum(response[:, WIDE_REGION, :72], 1e-30))
    tau = jnp.asarray(.5, dtype=d.dtype)
    m = -tau * (logsumexp(-d / tau, axis=(1, 2)) - jnp.log(jnp.asarray(int(WIDE_REGION.sum()) * 72, dtype=d.dtype)))
    return -jnp.mean(m)


def wide_summary(response):
    spec = objective_spec("wide")
    linear = np.asarray(response)[:, :, :72]
    d = 10 * np.log10(np.maximum(linear, spec["linear_floor"]))
    sector = d[:, WIDE_REGION, :]
    lo, hi = sector.min(axis=(1, 2)), sector.max(axis=(1, 2))
    # Shift by the hard minimum: every exponential argument is nonpositive.
    m = lo - spec["tau_db"] * np.log(np.exp(-(sector - lo[:, None, None]) /
                                            spec["tau_db"]).mean(axis=(1, 2)))
    outside = d[:, WIDE_OUTSIDE, :].max(axis=(1, 2))
    # Plain Riemann sum on the driver's grid, with sin(theta) weights and
    # the duplicated phi endpoint excluded from numerator and denominator.
    power = linear * np.sin(THETA)[None, :, None]
    fraction = power[:, WIDE_REGION, :].sum(axis=(1, 2)) / power.sum(axis=(1, 2))
    angles = np.rad2deg(THETA)
    rolloff = np.full((len(d), 2), np.nan)
    for f in range(len(d)):
        for plane, phi in enumerate((0, 18)):
            cut = d[f, :, phi]
            crossings = []
            start = 1
            for drop in (3., 15.):
                threshold = lo[f] - drop
                below = np.flatnonzero(cut[start:] < threshold) + start
                if not len(below):
                    break
                i = int(below[0])
                crossings.append(angles[i - 1] + (threshold - cut[i - 1]) *
                                 (angles[i] - angles[i - 1]) / (cut[i] - cut[i - 1]))
                # Both thresholds can cross within the same grid interval.
                start = i
            if len(crossings) == 2:
                rolloff[f, plane] = crossings[1] - crossings[0]
    return {"sector_min_dbi": lo, "sector_max_dbi": hi, "ripple_db": hi - lo,
            "smooth_min_dbi": m, "outside_max_dbi": outside,
            "outside_minus_sector_min_db": outside - lo,
            "boresight_dbi": summary(response)["boresight_dbi"],
            "cone_power_fraction": fraction, "rolloff_deg": rolloff,
            "meets_ripple": hi - lo <= spec["judged"]["ripple_max_db"],
            "meets_level": lo >= spec["judged"]["level_min_dbi"],
            "objective": float(-m.mean())}


def fan_loss(response):
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    d = 10 * jnp.log10(jnp.maximum(response[:, FAN_ROWS][:, :, FAN_COLS], 1e-30))
    tau = jnp.asarray(.2, dtype=d.dtype)
    log_count = jnp.log(jnp.asarray(78, dtype=d.dtype))
    smin = -tau * (logsumexp(-d / tau, axis=(1, 2)) - log_count)
    smax = tau * (logsumexp(d / tau, axis=(1, 2)) - log_count)
    return jnp.mean(smax - 2 * smin)


def fan_summary(response):
    spec = objective_spec("fan")
    d = 10 * np.log10(np.maximum(np.asarray(response, dtype=np.float64), spec["linear_floor"]))
    strip = d[:, FAN_ROWS][:, :, FAN_COLS]
    lo, hi = strip.min(axis=(1, 2)), strip.max(axis=(1, 2))
    tau = spec["tau_db"]
    smin = lo - tau * np.log(np.exp(-(strip - lo[:, None, None]) / tau).mean(axis=(1, 2)))
    smax = hi + tau * np.log(np.exp((strip - hi[:, None, None]) / tau).mean(axis=(1, 2)))
    bore = d[:, 0, 0]
    crossings = np.full((len(d), 4), np.nan)
    for f in range(len(d)):
        for side, phi in enumerate((0, 36, 18, 54)):
            cut = d[f, :, phi]
            threshold = bore[f] - 3.
            below = np.flatnonzero(cut < threshold)
            if len(below):
                i = int(below[0])
                crossings[f, side] = NOMINAL_DEG[i - 1] + 2.5 * (cut[i - 1] - threshold) / (cut[i - 1] - cut[i])
    narrow = crossings[:, 0] + crossings[:, 1]
    return {"strip_min_dbi": lo, "strip_max_dbi": hi, "ripple_db": hi - lo,
            "smooth_min_dbi": smin, "smooth_max_dbi": smax, "boresight_dbi": bore,
            "narrow_hpbw_deg": narrow, "wide_width_deg": crossings[:, 2] + crossings[:, 3],
            "outside_max_dbi": d[:, NOMINAL_DEG >= 60, :72].max(axis=(1, 2)),
            "meets_ripple": hi - lo <= spec["judged"]["ripple_max_db"],
            "meets_level": lo >= spec["judged"]["level_min_dbi"],
            "meets_narrow": narrow <= spec["judged"]["narrow_hpbw_max_deg"],
            "objective": float(np.mean(smax - 2 * smin))}


def mask_loss(response, objective):
    import jax.numpy as jnp
    objective_spec(objective)
    d = 10 * jnp.log10(jnp.maximum(response[:, :, :72], 1e-30))
    def q(region, offset):
        v = jnp.maximum(d[:, REGIONS[region], :] + offset, 0)
        return jnp.sqrt(jnp.mean(v ** 4, axis=(1, 2)) + .1 ** 4) - .1 ** 2
    if objective == "mask":
        bore = 10 * jnp.log10(jnp.maximum(response[:, 0, :].mean(axis=1), 1e-30))
        return jnp.mean(jnp.maximum(19 - bore, 0) ** 2 + q("S", 3))
    c = d[:, REGIONS["C"], :]
    return jnp.mean(jnp.mean(jnp.maximum(13.5 - c, 0) ** 2 +
                             jnp.maximum(c - 14.5, 0) ** 2, axis=(1, 2)) + q("O", 1))


def mask_summary(response, objective):
    spec = objective_spec(objective)
    result = summary(response)
    if objective == "flattop":
        for key in ("aperture_bound_dbi", "above_aperture_bound"):
            result.pop(key)
    d = 10 * np.log10(np.maximum(np.asarray(response)[:, :, :72], 1e-30))
    region = REGIONS["S" if objective == "mask" else "O"]
    outer = d[:, region, :]
    index = np.argmax(outer.reshape(len(d), -1), axis=1)
    ti, pi = np.unravel_index(index, outer.shape[1:])
    worst = outer.reshape(len(d), -1)[np.arange(len(d)), index]
    met = worst <= spec["mask_dbi"]["outer_max"]
    result.update(outer_max_dbi=worst, outer_max_theta_rad=THETA[region][ti],
                  outer_max_phi_rad=PHI[pi])
    if objective == "mask":
        met &= result["boresight_dbi"] >= 18
    else:
        c = d[:, REGIONS["C"], :]
        lo, hi = c.min(axis=(1, 2)), c.max(axis=(1, 2))
        result.update(window_min_dbi=lo, window_max_dbi=hi)
        met &= (lo >= 13) & (hi <= 15)
    result.update(mask_met_per_frequency=met, mask_met=bool(np.all(met)))
    return result


def with_objective(objective):
    """Bind a case locally: no module-global objective can leak between runs."""
    from types import SimpleNamespace
    from functools import partial
    spec = objective_spec(objective)
    class ObjectiveModel(Model):
        def evaluate(self, e, n_steps, normalize="flux"):
            response, row = super().evaluate(e, n_steps, normalize)
            if objective != "boresight":
                for key in summary(response):
                    row.pop(key)
                row.update(summary(response, objective))
            return response, row
    return SimpleNamespace(**{**vars(sys.modules[__name__]), "OBJECTIVE": objective,
                              "objective_metadata": spec, "Model": ObjectiveModel,
                              "loss": partial(loss, objective=objective),
                              "summary": partial(summary, objective=objective)})


class Model(common.ModelBase):
    def __init__(self, dx=DESIGN_DX, precision="float32", smoke=False):
        from rfx import Simulation
        from rfx.geometry.csg import Box
        from validation.crossval.comparators import realized_conductors as RC
        self.dx, self.precision, self.smoke = dx, precision, smoke
        self.freqs = FREQS
        self.sim = Simulation(freq_max=12.6e9, domain=(.126, .126, .120), dx=dx,
                              cpml_layers=common.integral_cells(.015, dx), precision=precision)
        self.sim.add_source((.063, .063, .027), "ex", amplitude_kind="current")
        self.probes = ((.063, .063, .087), (.1065, .1065, .087),
                       (.063, .063, .1095), (.078, .063, .021))
        for p in self.probes:
            self.sim.add_probe(p, "ex")
        self.sim.add(Box((.048, .048, .018), (.078, .078, .018)), material="pec")
        self.sim.add_ntff_box(corner_lo=(.003, .003, .003),
                              corner_hi=(.123, .123, .117), freqs=list(FREQS))
        self.grid = self.sim._build_grid()
        # Every geometric face, including all internal printable-pixel faces.
        faces = (np.r_[0., .003, .123, .126, .048, .078, .018 + .003 * np.arange(31)],
                 np.r_[0., .003, .123, .126, .048, .078, .018 + .003 * np.arange(31)],
                 np.r_[0., .003, .018, .117, .120, .072 + .003 * np.arange(11)])
        for axis, coords in enumerate(faces):
            for coordinate in coords:
                common.assert_node(self.grid, axis, coordinate)
        self.lo = self.grid.position_to_index((.018, .018, .072))
        self.hi = self.grid.position_to_index((.108, .108, .102))
        realized = (np.asarray(self.hi) - np.asarray(self.lo)) * dx
        if not np.allclose(realized, (.09, .09, .03), atol=1e-12, rtol=0):
            raise ValueError(f"realized lens region {realized}")
        self.plate = RC.assert_wall_planes(self.sim, 2, [.018], at=(.063, .063),
                                           label="reflector", grid=self.grid, tol_m=1e-12)
        self.default_steps = common.segmented_steps(self.grid.num_timesteps(num_periods=60))
        if smoke:
            self.default_steps = 200
        self._functions = {}

    def response_fn(self, n_steps, normalize="flux"):
        import jax
        import jax.numpy as jnp
        from rfx.measurement.farfield import compute_far_field_jax
        from rfx.farfield import make_ntff_box
        if n_steps in self._functions:
            return self._functions[n_steps]
        dtype = jnp.float64 if self.precision == "float64" else jnp.float32
        box = make_ntff_box(self.grid, *self.sim._ntff)
        box = box._replace(freqs=np.asarray(box.freqs))
        # Beam module's 73 x 73 grid and rectangular quadrature. Evaluate
        # boresight separately at exactly theta=0 rather than claiming 1e-4 is 0.
        th = jnp.asarray(np.linspace(1e-4, np.pi - 1e-4, 73), dtype=dtype)
        ph = jnp.asarray(np.linspace(0., 2 * np.pi, 73), dtype=dtype)
        w = jnp.sin(th)[:, None] * jnp.gradient(th)[:, None] * jnp.gradient(ph)[None, :]

        # Design-box formulation (#1179): reverse mode stores box-shaped
        # fields per step instead of grid-shaped ones; the same gradient
        # (docstring of Simulation.forward). Corners are inclusive cells.
        box_lo = tuple(float(v) for v in (.018, .018, .072))
        box_hi = tuple(float(v - self.dx) for v in (.108, .108, .102))

        def run(e):
            cells = expand_pixels(jnp.asarray(e, dtype=dtype), self.dx, jnp)
            return self.sim.forward(design_box=(box_lo, box_hi), design_eps_override=cells,
                                    n_steps=n_steps, checkpoint=False, skip_preflight=True)

        # One theta row at a time: the transform otherwise materializes every
        # (direction x NTFF surface point x frequency) phase factor at once,
        # about 11 GB at 1.0 mm. Rematerialized in the backward pass.
        @jax.checkpoint
        def row_power(ntff_data, t):
            ff = compute_far_field_jax(ntff_data, box, self.grid, t[None], ph)
            return ((jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2) * 1e27)[:, 0, :]

        def response(e):
            r = run(e)
            p = jnp.moveaxis(jax.lax.map(lambda t: row_power(r.ntff_data, t), th), 0, 1)
            prad = jnp.sum(p * w[None, :, :], axis=(1, 2))
            bore = compute_far_field_jax(r.ntff_data, box, self.grid, jnp.zeros(1, dtype=dtype), ph)
            p0 = (jnp.abs(bore.E_theta) ** 2 + jnp.abs(bore.E_phi) ** 2) * 1e27
            d = 4 * jnp.pi * p / jnp.maximum(prad[:, None, None], 1e-30)
            return d.at[:, :1, :].set(4 * jnp.pi * p0 / jnp.maximum(prad[:, None, None], 1e-30))

        response.run = run
        self._functions[n_steps] = response
        return response

    def evaluate(self, e, n_steps, normalize="flux"):
        from rfx.sparams._tail_witness import tail_share_witness
        fn = self.response_fn(n_steps)
        response = np.asarray(fn(e))
        # A separate eager record retains four concrete probe traces for the
        # host-only pole identification; no witness is inferred from a JIT trace.
        r = fn.run(e)
        traces = np.asarray(r.time_series)
        w = tail_share_witness(tuple((f"probe{i}", traces[:, i]) for i in range(4)),
                               self.grid.dt, r.settling_source_end_index, FREQS,
                               freq_max=12.6e9)
        return response, {"settling": w._asdict(), "settled": w.status == "pass",
                          "field_dtype": str(r.ntff_data.x_lo.dtype), **summary(response)}

    def describe(self):
        return {**super().describe(), "reflector": self.plate,
                "probe_declared_m": self.probes,
                "probe_realized_m": [(np.asarray(self.grid.position_to_index(p)) -
                                      np.asarray(self.grid.axis_pads)) * self.dx for p in self.probes],
                "theta_rad": np.r_[0., np.linspace(1e-4, np.pi - 1e-4, 73)[1:]],
                "phi_rad": np.linspace(0., 2 * np.pi, 73),
                "quadrature_theta_rad": np.linspace(1e-4, np.pi - 1e-4, 73),
                "ntff_corners_m": ((.003, .003, .003), (.123, .123, .117))}


if __name__ == "__main__":
    raise SystemExit(common.main(sys.modules[__name__]))
