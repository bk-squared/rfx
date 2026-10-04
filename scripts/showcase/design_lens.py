"""Printable GRIN lens: stages governed by the 20261004 pre-declaration.

Free indices run outward from the axis in x/y and upward from the lens bottom.
The physical domain is 126 x 126 x 120 mm; CPML is outside that domain.
"""
from __future__ import annotations

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


def loss(response):
    import jax.numpy as jnp
    return -jnp.mean(jnp.log(jnp.maximum(response[:, 0, :].mean(axis=1), 1e-30)))


def summary(response):
    d = np.asarray(response)
    bore = 10 * np.log10(np.maximum(d[:, 0, :].mean(axis=1), 1e-30))
    bound = 10 * np.log10(4 * np.pi * .09 ** 2 / (299792458. / FREQS) ** 2)
    return {"boresight_dbi": bore, "mean_boresight_dbi": float(bore.mean()),
            "aperture_bound_dbi": bound, "above_aperture_bound": bore > bound}


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
        from rfx.farfield import compute_far_field_jax, make_ntff_box
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
