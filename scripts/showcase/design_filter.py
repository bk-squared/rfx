"""WR-90 printable insert, with Amendment 1 pixel and record conventions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lens_filter_common as common

CASE = "filter"
MESHES = tuple(.02286 / n for n in (36, 54, 72))
DESIGN_DX = MESHES[0]
SMOKE_DX = .00254
SHAPE = (32, 5)
FREQS = np.linspace(8.2e9, 12.4e9, 85)
FD_PIXELS = {f"pixel_{i}_centre": (i - 1, 4) for i in (6, 16, 26)}
UPPER = 10.
ITERATIONS = 200
LR = (.15, .015)
PASS = (FREQS >= 10e9) & (FREQS <= 10.6e9)
STOP = (FREQS <= 9e9) | (FREQS >= 11.6e9)


def expand_pixels(free, dx, xp=np):
    n = common.integral_cells(.00254, dx)
    # Free columns are lower wall -> centre; centre appears only once.
    full = xp.concatenate((free, free[:, -2::-1]), axis=1)
    full = xp.repeat(xp.repeat(full, n, axis=0), n, axis=1)
    return xp.repeat(full[:, :, None], common.integral_cells(.01016, dx), axis=2)


def starts():
    # The pre-declaration leaves the material between the S2 slabs unspecified.
    # Keep the S1 background (1.5); this convention is recorded, not inferred
    # from a run. All six named full-width pixels are exactly eps_r=6.
    s2 = np.full(SHAPE, 1.5, np.float32)
    for first in (6, 16, 26):
        s2[first - 1:first + 1, :] = 6.
    return {"S1": np.full(SHAPE, 1.5, np.float32), "S2": s2}


def baselines():
    return {"empty": np.ones(SHAPE, np.float32), "S2": starts()["S2"]}


def mask_objective(s11, s21, xp=np):
    reflection = 20 * xp.log10(xp.maximum(xp.abs(s11), 1e-6))
    transmission = 20 * xp.log10(xp.maximum(xp.abs(s21), 1e-6))
    return xp.mean(xp.maximum(reflection[PASS] + 17, 0) ** 2) + xp.mean(
        xp.maximum(transmission[STOP] + 27, 0) ** 2)


def loss(response):
    import jax.numpy as jnp
    return mask_objective(response[0, 0], response[1, 0], jnp)


def summary(response):
    s = np.asarray(response)
    db = 20 * np.log10(np.maximum(np.abs(s), 1e-6))
    # Edges are first and last compliant read bins, as declared (also retain
    # all compliant bins so disconnected intervals remain visible).
    compliant = FREQS[db[0, 0] <= -15]
    power = np.abs(s[0, 0]) ** 2 + np.abs(s[1, 0]) ** 2
    return {"S11_db": db[0, 0], "S21_db": db[1, 0],
            "passband_edges_hz": [float(compliant[0]), float(compliant[-1])] if len(compliant) else None,
            "compliant_reflection_bins_hz": compliant,
            "worst_passband_S11_db": float(db[0, 0, PASS].max()),
            "worst_stopband_S21_db": float(db[1, 0, STOP].max()),
            "inside_mask": bool(np.all(db[0, 0, PASS] <= -15) and np.all(db[1, 0, STOP] <= -25)),
            "power_balance": power, "power_balance_passed": bool(np.all((power >= .98) & (power <= 1.01)))}


class Model(common.ModelBase):
    def __init__(self, dx=DESIGN_DX, precision="float32", smoke=False):
        from rfx import Simulation
        from rfx.boundaries.spec import Boundary, BoundarySpec
        from validation.crossval.comparators import realized_conductors as RC
        self.dx, self.precision, self.smoke = dx, precision, smoke
        self.freqs = FREQS
        boundary = BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                y=Boundary(lo="pec", hi="pec"), z=Boundary(lo="pec", hi="pec"))
        self.sim = Simulation(freq_max=13.02e9, domain=(.20066, .02286, .01016), dx=dx,
                              boundary=boundary, cpml_layers=common.integral_cells(.04572, dx),
                              precision=precision)
        # Amendment 2: 48 / 64 cells at a/36 (30.48 / 40.64 mm), on a node
        # line at a/36, a/54 and a/72 alike.
        for x, direction, ref, name in ((.03048, "+x", .04064, "left"),
                                         (.17018, "-x", .16002, "right")):
            self.sim.add_waveguide_port(x, direction=direction, reference_plane=ref, name=name,
                                        freqs=FREQS, f0=float(FREQS.mean()), bandwidth=.45,
                                        waveform="modulated_gaussian")
        self.grid = self.sim._build_grid()
        RC.assert_no_conductor(self.sim, label="WR-90 filter (domain PEC)")
        built = [(self.grid.shape[ax] - 1 - self.grid.face_pads[2 * ax]
                  - self.grid.face_pads[2 * ax + 1]) * dx for ax in (1, 2)]
        if not np.allclose(built, (.02286, .01016), rtol=0, atol=1e-12):
            raise ValueError(f"realized guide {built} is not 22.86 x 10.16 mm")
        self.walls = built
        for axis, coords in ((0, np.r_[.03048, .04064, .16002, .17018,
                                       .05969 + np.arange(33) * .00254]),
                             (1, np.arange(10) * .00254), (2, (0., .01016))):
            if not smoke:
                for coordinate in coords:
                    common.assert_node(self.grid, axis, coordinate)
        self.lo = self.grid.position_to_index((.05969, 0., 0.))
        self.hi = tuple(a + b for a, b in zip(self.lo,
                         (common.integral_cells(.08128, dx), common.integral_cells(.02286, dx),
                          common.integral_cells(.01016, dx))))
        self.default_steps = common.segmented_steps(self.grid.num_timesteps(num_periods=240))
        if smoke:
            self.default_steps = 240
        self._functions = {}

    def response_fn(self, n_steps, normalize="flux"):
        import jax.numpy as jnp
        key = n_steps, normalize
        if key in self._functions:
            return self._functions[key]
        dtype = jnp.float64 if self.precision == "float64" else jnp.float32
        slices = tuple(slice(a, b) for a, b in zip(self.lo, self.hi))

        def run(e):
            block = expand_pixels(jnp.asarray(e, dtype=dtype), self.dx, jnp)
            # Include transverse boundary nodes (the last cell's material);
            # the PEC domain constraints remain independent of this override.
            block = jnp.pad(block, ((0, 0), (0, 1), (0, 1)), mode="edge")
            sl = (slices[0], slice(self.lo[1], self.hi[1] + 1), slice(self.lo[2], self.hi[2] + 1))
            eps = jnp.ones(self.grid.shape, dtype=dtype).at[sl].set(block)
            return self.sim.compute_waveguide_s_matrix(eps_override=eps, n_steps=n_steps,
                        normalize=normalize, checkpoint_segments=common.sqrt_segments(n_steps))

        def response(e):
            return run(e).s_params

        response.run = run
        self._functions[key] = response
        return response

    def evaluate(self, e, n_steps, normalize="flux"):
        r = self.response_fn(n_steps, normalize).run(e)
        response = np.asarray(r.s_params)
        details = r.settling_witness
        def passed(row):
            return (row["status"] == "pass" and np.all(np.asarray(row["share_per_bin"]) <= .01)
                    and all(passed(run) for run in row.get("runs", ())))
        settled = bool(details) and all(passed(row) for row in details)
        metrics = summary(response)
        return response, {"settling": details, "settled": settled, "normalize": normalize,
                          "mesh_number_eligible": settled and metrics["power_balance_passed"],
                          "reference_planes_m": r.reference_planes, **metrics}

    def describe(self):
        declared = np.array([.03048, .17018, .04064, .16002])  # Amendment 2
        realized = [(self.grid.position_to_index((float(x), 0., 0.))[0] - self.grid.pad_x_lo)
                    * self.dx for x in declared]
        return {**super().describe(), "realized_guide_m": self.walls,
                "port_then_reference_declared_x_m": declared,
                "port_then_reference_realized_x_m": realized, "S2_background_eps": 1.5,
                "TE20_advisory_disposition": "Expected: even-y, z-uniform insert; TE20 is odd in y. "
                "TE30 is cut off in the empty guide; nominal insert-to-reference gap is 19.05 mm."}


if __name__ == "__main__":
    raise SystemExit(common.main(sys.modules[__name__]))
