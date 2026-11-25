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
MESHES = (1e-3, .75e-3, .5e-3)
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


FACES = (np.r_[0., .003, .123, .126, .048, .078, .018 + .003 * np.arange(31)],
         np.r_[0., .003, .123, .126, .048, .078, .018 + .003 * np.arange(31)],
         np.r_[0., .003, .018, .117, .120, .072 + .003 * np.arange(11)])


def graded_profiles(dx):
    from rfx.model.graded_profile import make_band_profile
    spans = ((.048 - .65 * dx, .048 + .35 * dx),
             (.078 - .35 * dx, .078 + .65 * dx))
    result = {}
    for axis, faces in zip("xy", FACES[:2]):
        lines = sorted(float(v) for v in faces
                       if all(abs(v - e) > 1e-12 for e in (.048, .078)))
        lines = sorted(set(lines + [v for span in spans for v in span]))
        lines = [v for i, v in enumerate(lines) if i == 0 or v - lines[i - 1] > 1e-12]
        protected = [any(abs(a - lo) <= 1e-12 and abs(b - hi) <= 1e-12
                         for lo, hi in spans) for a, b in zip(lines[:-1], lines[1:])]
        result[f"d{axis}_profile"] = make_band_profile(
            lines, [dx] * (len(lines) - 1), protected=protected,
            max_ratio=1.3, boundary_cell=dx)
    return result


def pixel_weights(nodes, faces, boundary):
    widths = np.diff(nodes)
    overlap = np.maximum(0., np.minimum(nodes[1:, None], faces[None, 1:]) -
                         np.maximum(nodes[:-1, None], faces[None, :-1]))
    weights = overlap / widths[:, None]
    weights[abs(weights) < 1e-12] = 0.
    if boundary == "nearest":
        pixels = np.searchsorted(faces, (nodes[:-1] + nodes[1:]) / 2, side="right") - 1
        weights = np.eye(len(faces) - 1)[pixels]
    return weights


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


def summary(response, objective="boresight", freqs=FREQS):
    if objective == "fan":
        return fan_summary(response)
    if objective == "wide":
        return wide_summary(response, freqs)
    if objective != "boresight":
        return mask_summary(response, objective, freqs)

    d = np.asarray(response)
    bore = 10 * np.log10(np.maximum(d[:, 0, :].mean(axis=1), 1e-30))
    bound = 10 * np.log10(4 * np.pi * .09 ** 2 / (299792458. / np.asarray(freqs)) ** 2)
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


def wide_summary(response, freqs=FREQS):
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
            "boresight_dbi": summary(response, freqs=freqs)["boresight_dbi"],
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


def mask_summary(response, objective, freqs=FREQS):
    spec = objective_spec(objective)
    result = summary(response, freqs=freqs)
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
                for key in summary(response, freqs=self.freqs):
                    row.pop(key)
                row.update(summary(np.asarray(response, dtype=np.float64) if self.band else response,
                                   objective, self.freqs))
            return response, row
    return SimpleNamespace(**{**vars(sys.modules[__name__]), "OBJECTIVE": objective,
                              "objective_metadata": spec, "Model": ObjectiveModel,
                              "loss": partial(loss, objective=objective),
                              "summary": partial(summary, objective=objective)})


class Model(common.ModelBase):
    def __init__(self, dx=DESIGN_DX, precision="float32", smoke=False, *,
                 mesh="graded", boundary=None, freqs=None):
        from rfx import Grid, Simulation
        from rfx.geometry.csg import Box
        from validation.crossval.comparators import realized_conductors as RC
        self.dx, self.precision, self.smoke = dx, precision, smoke
        self.mesh, self.boundary = mesh, boundary
        if mesh not in ("graded", "uniform") or boundary not in (None, "nearest", "arith", "harm"):
            raise ValueError("invalid mesh or boundary")
        self.band = freqs is not None
        self.freqs = FREQS if freqs is None else np.asarray(freqs, dtype=np.float64)
        self.design_indices = [int(np.argmin(abs(self.freqs - f))) for f in FREQS]
        if any(abs(self.freqs[i] - f) > 1e3 for i, f in zip(self.design_indices, FREQS)):
            raise ValueError("resolve frequencies must include the three design bins")
        self.sim = Simulation(freq_max=12.6e9, domain=(.126, .126, .120), dx=dx,
                              cpml_layers=common.integral_cells(.015, dx), precision=precision,
                              snap="strict", **(graded_profiles(dx) if mesh == "graded" else {}))
        self.sim.add_source((.063, .063, .027), "ex", amplitude_kind="current")
        self.probes = ((.063, .063, .087), (.1065, .1065, .087),
                       (.063, .063, .1095), (.078, .063, .021))
        for p in self.probes:
            self.sim.add_probe(p, "ex")
        self.sim.add(Box((.048, .048, .018), (.078, .078, .018)), material="pec")
        self.sim.add_ntff_box(corner_lo=(.003, .003, .003),
                              corner_hi=(.123, .123, .117), freqs=list(self.freqs))
        if mesh == "graded":
            from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
            self.grid = self.sim._build_nonuniform_grid()
            for name in ("dx_arr_f64", "dy_arr_f64", "dz_f64"):
                spine = getattr(self.grid, name)
                if spine is None or np.asarray(spine).dtype != np.float64:
                    raise ValueError(f"missing float64 cell spine: {name}")
            coords = coords_from_nonuniform_grid(self.grid)
            self.nodes = tuple(np.asarray(v, dtype=np.float64) for v in (coords.x, coords.y, coords.z))
            for axis, faces in enumerate(FACES):
                for v in faces:
                    on = np.min(abs(self.nodes[axis] - v)) <= 1e-12
                    edge = axis < 2 and any(abs(v - e) <= 1e-12 for e in (.048, .078))
                    if on == edge:
                        raise ValueError(f"axis {axis}, face {v}: node alignment")
            plane = self.index((.063, .063, .018))[2]
            normal = np.diff(self.nodes[2][plane - 1:plane + 2])
            for line in self.nodes[:2]:
                for edge in (.048, .078):
                    k = np.searchsorted(line, edge, side="right") - 1
                    width = line[k + 1] - line[k]
                    if abs(width / dx - 1) > 1e-9 or np.any(abs(normal / width - 1) > 1e-9):
                        raise ValueError("plate edge cell or normal ratio")
        else:
            self.grid = self.sim._build_grid()
            self.nodes = tuple(np.asarray([self.grid.node_of(a, i) for i in range(n)], dtype=np.float64)
                               for a, n in enumerate(self.grid.shape))
            for axis, faces in enumerate(FACES):
                for v in faces:
                    common.assert_node(self.grid, axis, v)
        self.lo = self.index((.018, .018, .072))
        self.hi = self.index((.108, .108, .102))
        realized = np.array([line[b] - line[a] for line, a, b in zip(self.nodes, self.lo, self.hi)])
        if not np.allclose(realized, (.09, .09, .03), atol=1e-12, rtol=0):
            raise ValueError(f"realized lens region {realized}")
        if mesh == "graded":
            counts = {1e-3: (92, 92, 30), .75e-3: (122, 122, 40), .5e-3: (182, 182, 60)}
            if dx not in counts or tuple(b - a for a, b in zip(self.lo, self.hi)) != counts[dx]:
                raise ValueError("unexpected graded design box cell counts")
        self.box = (tuple(line[a] for line, a in zip(self.nodes, self.lo)),
                    tuple(line[b - 1] for line, b in zip(self.nodes, self.hi)))
        self.weights = tuple(pixel_weights(line[a:b + 1], faces, boundary)
                             for line, a, b, faces in zip(
                                 self.nodes, self.lo, self.hi,
                                 (.018 + .003 * np.arange(31), .018 + .003 * np.arange(31),
                                  .072 + .003 * np.arange(11))))
        self.plate = RC.assert_wall_planes(self.sim, 2, [.018], at=(.063, .063),
                                           label="reflector", grid=self.grid, tol_m=1e-12)
        uniform = Grid(freq_max=12.6e9, domain=(.126, .126, .120), dx=dx,
                       cpml_layers=common.integral_cells(.015, dx))
        self.default_steps = common.segmented_steps(
            int(np.ceil(uniform.num_timesteps(num_periods=60) * uniform.dt / self.grid.dt))
            if mesh == "graded" else self.grid.num_timesteps(num_periods=60))
        if smoke:
            self.default_steps = 200
        self._functions = {}

    def index(self, position):
        return tuple(int(np.argmin(abs(line - v))) for line, v in zip(self.nodes, position))

    def cells(self, free, xp=np):
        if self.mesh == "uniform":
            return expand_pixels(free, self.dx, xp)
        if self.boundary is None:
            raise ValueError("graded lens mapping requires --boundary")
        full = xp.concatenate((free[::-1, :, :], free), axis=0)
        full = xp.concatenate((full[:, ::-1, :], full), axis=1)
        wx, wy, wz = (xp.asarray(w, dtype=full.dtype) for w in self.weights)
        values = 1 / full if self.boundary == "harm" else full
        cells = xp.einsum("ip,jq,kr,pqr->ijk", wx, wy, wz, values, optimize=True)
        return 1 / cells if self.boundary == "harm" else cells

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
        box_lo, box_hi = self.box

        def run(e):
            cells = self.cells(jnp.asarray(e, dtype=dtype), jnp)
            return self.sim.forward(design_box=(box_lo, box_hi), design_eps_override=cells,
                                    n_steps=n_steps, checkpoint=False, skip_preflight=True)

        # One theta row at a time: the transform otherwise materializes every
        # (direction x NTFF surface point x frequency) phase factor at once,
        # about 11 GB at 1.0 mm. Rematerialized in the backward pass.
        @jax.checkpoint
        def row_power(ntff_data, t):
            ff = compute_far_field_jax(ntff_data, box, self.grid, t[None], ph)
            return ((jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2) * 1e27)[:, 0, :]

        def transform(r):
            p = jnp.moveaxis(jax.lax.map(lambda t: row_power(r.ntff_data, t), th), 0, 1)
            prad = jnp.sum(p * w[None, :, :], axis=(1, 2))
            bore = compute_far_field_jax(r.ntff_data, box, self.grid, jnp.zeros(1, dtype=dtype), ph)
            p0 = (jnp.abs(bore.E_theta) ** 2 + jnp.abs(bore.E_phi) ** 2) * 1e27
            d = 4 * jnp.pi * p / jnp.maximum(prad[:, None, None], 1e-30)
            return d.at[:, :1, :].set(4 * jnp.pi * p0 / jnp.maximum(prad[:, None, None], 1e-30))

        def response(e):
            return transform(run(e))

        response.run = run
        response.transform = transform
        self._functions[n_steps] = response
        return response

    def evaluate(self, e, n_steps, normalize="flux"):
        from rfx.sparams._tail_witness import tail_share_witness
        fn = self.response_fn(n_steps)
        response = None if self.band else np.asarray(fn(e))
        r = fn.run(e)
        if self.band:
            response = np.asarray(fn.transform(r), dtype=np.float64)
        traces = np.asarray(r.time_series)
        w = tail_share_witness(tuple((f"probe{i}", traces[:, i]) for i in range(4)),
                               self.grid.dt, r.settling_source_end_index, self.freqs,
                                freq_max=12.6e9, echo_bound=r.settling_echo_bound)
        return response, {"settling": w._asdict(), "settled": w.status == "pass",
                          "field_dtype": str(r.ntff_data.x_lo.dtype), **summary(response, freqs=self.freqs)}

    def preflight(self):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = self.sim.preflight()
        self.preflight_record = {"verdict": "PASS" if report.ok else "FAIL",
                                 "findings": report.to_dict()["issues"],
                                 "text": report.format(),
                                 "warnings": [{"message": str(w.message),
                                               "severity": getattr(w.message, "severity", "warning")}
                                              for w in caught]}
        messages = [str(i) for i in report] + [f"warning: {w.message}" for w in caught]
        print("PREFLIGHT BEGIN", *messages, "PREFLIGHT END", sep="\n", flush=True)
        return messages

    def describe(self):
        g = self.grid
        description = {"dx_m": self.dx, "grid_shape": g.shape, "grid_points": int(np.prod(g.shape)),
                       "cells": int(np.prod(np.asarray(g.shape) - 1)), "dt_s": g.dt,
                       "cpml_layers": g.pad_x_lo, "cpml_thickness_m": -self.nodes[0][0],
                       "n_steps": self.default_steps,
                       "checkpoint_segments": common.sqrt_segments(self.default_steps),
                       "design_lo": self.lo, "design_hi_exclusive": self.hi,
                       "design_lo_m": [line[a] for line, a in zip(self.nodes, self.lo)],
                       "design_hi_m": [line[b] for line, b in zip(self.nodes, self.hi)],
                       "freqs_hz": self.freqs, "precision": self.precision, "smoke": self.smoke,
                       "mesh": self.mesh, "boundary": self.boundary}
        return {**description, "reflector": self.plate,
                "probe_declared_m": self.probes,
                "probe_realized_m": [[line[i] for line, i in zip(self.nodes, self.index(p))] for p in self.probes],
                "theta_rad": np.r_[0., np.linspace(1e-4, np.pi - 1e-4, 73)[1:]],
                "phi_rad": np.linspace(0., 2 * np.pi, 73),
                "quadrature_theta_rad": np.linspace(1e-4, np.pi - 1e-4, 73),
                "ntff_corners_m": ((.003, .003, .003), (.123, .123, .117))}


if __name__ == "__main__":
    raise SystemExit(common.main(sys.modules[__name__]))
