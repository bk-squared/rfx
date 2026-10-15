"""Lumped voltage-current (V-I) port S-parameter gradient check.

Paper version and main
----------------------
The number in the T-MTT paper (the directional derivative of |S11(3 GHz)|^2
along the gradient agrees with a central finite difference to 0.2 %) was
produced at the tagged release ``paper-tmtt-2026`` (2026-08-10), on a
different scene: a one-cell 50 ohm lumped port alone in an open CPML box with a
4 x 3 x 2-cell permittivity block beside it.  To reproduce it::

    git checkout paper-tmtt-2026
    python validation/tmtt_paper/lumped_port_gradient_check.py

(re-run 2026-10-11: |S11|^2 = 0.927, directional agreement 0.006 %).

On main that scene no longer tests anything.  Three corrections of the port
and material model (2026-09-23..25) made the lone one-cell port reflect
|S11|^2 = 0.99997, so the objective barely depends on the block and a float32
finite difference cannot resolve its change.  This file therefore runs a
different scene on main.

The scene on main
-----------------
A lumped Ez port feeds a one-cell parallel-plate TEM line (PEC plates, PMC
side walls; the line of the lumped/wire port battery,
``scripts/diagnostics/lumped_wire_chain_battery_measure.py``), 30 mm long at
dx = 0.5 mm, ended in a parallel resistor R = 2 Zc, with the port referenced
to the line's Zc (= eta0).  The design variables are the permittivity
entries of 24 cells on one wall row of the line (cells [12:36, 0, 0], 5.5 to
17.5 mm from the port node), all 2.2 at the start.  Each E edge takes the mean
of its four incident cells, so the line's edges see 2.2 on the y_lo wall and
1.6 on the y_hi wall, with one-cell ramps at the section ends.  The objective is the mean of |S11|^2 over 4, 4.5, 5, 5.5 and
6 GHz through the differentiable ``forward(port_s11_freqs=...)``.  The port
and the load sit on the y_lo magnetic face, and rfx solves an element on a
magnetic face as the element of the full symmetric object (rfx issue 1221,
decision A, the convention preflight states): against the declared Zc and
2 Zc the line looks like Zc/2.  On that reading the air line follows the
closed form of a Zc/2 line (length fitted once, 30.84 mm) within 0.0012 in
|S11|^2, and this scene, with the section taken as permittivity 1.9 (the
mean over the two Ez rows), within 0.009 (rfx-archive
records/20261011-examples-recheck/lumped_gradient_main/MIRROR-CLOSED-FORM.md).
That is context only: the check compares rfx's gradient with rfx's own
finite difference.

PMC-plane convention: the four side faces (x and y) are PMC. On this
single-device uniform run the magnetic wall is the declared face itself (H
taken as its odd image there, ``rfx/boundaries/pmc.py``), and the port and
load on the y_lo face are solved as the elements of the mirrored object, as
stated above.

What is checked, with the bars declared before the run: G1 the port couples
(mean |S11|^2 between 0.02 and 0.5); G2 the float64 finite difference resolves
at least 1e4 ULPs; G3 the directional derivative agrees within 1e-2 and is flat
over the step sweep; G4 per cell, vector relative error <= 5e-2 and signs
agree; G5 the gradient from an 80-period record equals the 40-period one
within 1e-2 of its peak.  On main (Mac CPU, 2026-10-12): mean |S11|^2 0.2516;
directional agreement 4.6e-6; per-cell vector agreement 4.9e-6; the 80-period
gradient differs by at most 4.4e-7 against a 6.0e-3 peak.  All gates pass.

Run (CPU, about 20 minutes, most of it the float64 finite differences):
    JAX_PLATFORMS=cpu python validation/tmtt_paper/lumped_port_gradient_check.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import rfx
from rfx import GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

DX = 0.5e-3
L_LINE = 30e-3
PORT_NODE = 1
ZC = getattr(rfx, "eta0", 376.730313668)
FREQS = np.asarray([4.0, 4.5, 5.0, 5.5, 6.0]) * 1e9
CELLS = np.arange(12, 36)
THETA0 = 2.2
R_STEPS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4)
MIN_ULPS = 1e4


def build(precision="float32"):
    """Self-contained lumped/res_double layout from the canonical battery."""
    n_l = int(round(L_LINE / DX))
    n_nodes = n_l + 3
    sim = Simulation(
        freq_max=10e9, domain=((n_nodes - 1) * DX, DX, DX), dx=DX,
        precision=precision,
        boundary=BoundarySpec(x=Boundary(lo="pmc", hi="pmc"),
                              y=Boundary(lo="pmc", hi="pmc"),
                              z=Boundary(lo="pec", hi="pec")),
    )
    sim.add_port(position=(PORT_NODE * DX, 0, 0), component="ez",
                 impedance=ZC, waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    sim.add_lumped_rlc(position=((n_nodes - 2) * DX, 0, 0), component="ez",
                       R=2 * ZC, topology="parallel")
    return sim


def own_eps(sim, dtype):
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    return jnp.asarray(np.asarray(materials.eps_r), dtype=dtype)


def make_eps(base, theta):
    return base.at[12:36, 0, 0].set(theta)


def verify_eps(base, theta):
    """Read back the exact override array supplied to forward."""
    actual = np.asarray(make_eps(base, theta))
    mask = np.zeros(actual.shape, dtype=bool)
    mask[12:36, 0, 0] = True
    if not (mask.sum() == 24 and np.all(actual[mask] == actual.dtype.type(THETA0))
            and np.all(actual[~mask] == 1)):
        raise RuntimeError("The forward override does not contain exactly the declared 24 cells")
    return actual


def functions(sim, base, periods):
    def s11(theta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.forward(eps_override=make_eps(base, theta),
                                 port_s11_freqs=jnp.asarray(FREQS, dtype=base.dtype),
                                 num_periods=periods, skip_preflight=True)
        return result.s_params.reshape(-1)

    def loss(theta):
        return jnp.mean(jnp.abs(s11(theta)) ** 2)
    return s11, loss


def relative(numerator, denominator):
    return float(numerator / denominator) if denominator != 0 else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("validation/tmtt_paper/_out/lumped_port_gradient_check"))
    args = parser.parse_args()
    started = time.perf_counter()
    print(f"rfx.__file__ = {rfx.__file__}", flush=True)
    root = Path(__file__).resolve().parents[2]
    if not Path(rfx.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"rfx was not imported from worktree {root}")
    if jax.config.x64_enabled:
        raise RuntimeError("Production AD requires default float32 mode; x64 must be scoped")
    sim = build()
    grid = sim._build_grid()
    port = grid.position_to_index(sim._ports[0].position)
    load = grid.position_to_index(sim._lumped_rlc[0].position)
    if tuple(grid.shape) != (63, 2, 2) or port != (1, 0, 0) or load != (61, 0, 0):
        raise RuntimeError("Canonical channel grid, port, or load did not realize as declared")
    # Design cells 12..35: first node x=6 mm, 5.5 mm after the port node.
    ranges = np.column_stack((CELLS, CELLS + 1)) * grid.dx
    lengths = [ranges[0, 0] - port[0] * grid.dx,
               ranges[-1, 1] - ranges[0, 0],
               load[0] * grid.dx - ranges[-1, 1]]
    base = own_eps(sim, jnp.float32)
    theta = jnp.full((24,), THETA0, dtype=jnp.float32)
    eps32 = verify_eps(base, theta)
    record = dict(rfx_file=rfx.__file__, jax_version=jax.__version__,
                  devices=[str(d) for d in jax.devices()],
                  grid_shape=list(grid.shape), dx_m=grid.dx, dt_s=grid.dt,
                  port_node=list(port), load_node=list(load), zc_ohm=ZC,
                  load_ohm=2 * ZC, theta0=THETA0, periods=[40, 80],
                  frequencies_hz=FREQS.tolist(), design_array_indices=CELLS.tolist(),
                  design_x_ranges_mm=(ranges * 1e3).tolist(),
                  section_lengths_mm=(np.asarray(lengths) * 1e3).tolist(),
                  eps_override_float32=eps32.tolist(), eps_readback_verified=True)
    print(f"grid={grid.shape} dx={grid.dx:.17g} m dt={grid.dt:.17g} s", flush=True)
    print(f"port={port} load={load} Zc={ZC:.17g} ohm R={2*ZC:.17g} ohm", flush=True)
    print(f"theta0={THETA0} periods=40,80 frequencies_hz={FREQS.tolist()}", flush=True)
    print("eps readback: exactly 24 design entries at theta0; every other entry 1.0", flush=True)
    for cell, bounds in zip(CELLS, ranges * 1e3):
        print(f"design cell {cell}: x=[{bounds[0]:.9g}, {bounds[1]:.9g}] mm", flush=True)
    print(f"realized section lengths mm={record['section_lengths_mm']}", flush=True)
    s11, loss = functions(sim, base, 40)
    measured = np.asarray(s11(theta))
    record["s11"] = []
    for freq, meas in zip(FREQS, measured):
        row = dict(frequency_hz=float(freq), rfx_real=float(meas.real),
                   rfx_imag=float(meas.imag), rfx_power=float(abs(meas)**2))
        record["s11"].append(row)
        print(f"S11 {freq/1e9:.1f} GHz rfx={meas.real:+.12e}{meas.imag:+.12e}j "
              f"power={row['rfx_power']:.12e}", flush=True)
    value, grad = jax.value_and_grad(loss)(theta)
    if value.dtype != jnp.float32 or grad.dtype != jnp.float32:
        raise RuntimeError("Production objective/gradient did not run in float32")
    value = float(value)
    g = np.asarray(grad, dtype=np.float64)
    norm = np.linalg.norm(g)
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("Cannot form the declared direction along the AD gradient")
    direction = g / norm
    dd_ad = float(g @ direction)
    print(f"objective float32(40)={value:.17g}; directional AD={dd_ad:.17g}", flush=True)
    for cell, derivative in zip(CELLS, g):
        print(f"AD cell {cell}: {derivative:+.17e}", flush=True)

    with jax.enable_x64():
        sim64 = build("float64")
        grid64 = sim64._build_grid()
        if (tuple(grid64.shape), grid64.dx, grid64.dt) != (tuple(grid.shape), grid.dx, grid.dt):
            raise RuntimeError("Float64 grid shape, dx, or dt differs from float32")
        base64 = own_eps(sim64, jnp.float64)
        theta64 = jnp.full((24,), THETA0, dtype=jnp.float64)
        record["eps_override_float64"] = verify_eps(base64, theta64).tolist()
        _, loss64 = functions(sim64, base64, 40)

        def referee(t):
            result = loss64(t)
            if result.dtype != jnp.float64:
                raise RuntimeError(f"Float64 referee loss has dtype {result.dtype}")
            return float(result)

        def difference(vector, h):
            plus = referee(theta64 + h * jnp.asarray(vector))
            minus = referee(theta64 - h * jnp.asarray(vector))
            span = abs(plus - minus) / np.spacing(np.float64(abs(0.5 * (plus + minus))))
            return dict(h=h, f_plus=plus, f_minus=minus,
                        derivative=(plus - minus) / (2 * h), ulp_span=float(span),
                        resolution="PASS" if np.isfinite(span) and span >= MIN_ULPS else "CANNOT-JUDGE")

        value64 = referee(theta64)
        sweep = []
        print("step sweep: r h f_plus f_minus dd_AD dd_FD relative_error ULP_span resolution", flush=True)
        for r in R_STEPS:
            row = difference(direction, r * THETA0)
            row.update(r=r, dd_ad=dd_ad,
                       relative_error=relative(abs(dd_ad-row["derivative"]), abs(row["derivative"])))
            sweep.append(row)
            print(f"{r:.9g} {row['h']:.9g} {row['f_plus']:.17g} {row['f_minus']:.17g} "
                  f"{dd_ad:+.12e} {row['derivative']:+.12e} {row['relative_error']:.12e} "
                  f"{row['ulp_span']:.12e} {row['resolution']}", flush=True)
        cell_fd = []
        print("per-cell FD: cell h f_plus f_minus AD FD relative_error ULP_span resolution", flush=True)
        for i, cell in enumerate(CELLS):
            row = difference(np.eye(24)[i], 1e-3 * THETA0)
            row.update(cell=int(cell), ad=float(g[i]),
                       relative_error=relative(abs(g[i]-row["derivative"]), abs(row["derivative"])))
            cell_fd.append(row)
            print(f"{cell} {row['h']:.9g} {row['f_plus']:.17g} {row['f_minus']:.17g} "
                  f"{g[i]:+.12e} {row['derivative']:+.12e} {row['relative_error']:.12e} "
                  f"{row['ulp_span']:.12e} {row['resolution']}", flush=True)

    _, loss80 = functions(sim, base, 80)
    value80, grad80 = jax.value_and_grad(loss80)(theta)
    value80 = float(value80)
    g80 = np.asarray(grad80, dtype=np.float64)
    print("record length: cell AD40 AD80 absolute_difference", flush=True)
    for cell, a, b in zip(CELLS, g, g80):
        print(f"{cell} {a:+.17e} {b:+.17e} {abs(b-a):.17e}", flush=True)
    fd = np.asarray([row["derivative"] for row in cell_fd])
    vec_rel = relative(np.linalg.norm(g-fd), np.linalg.norm(fd))
    significant = np.abs(fd) >= 1e-2 * np.max(np.abs(fd))
    sign_matches = np.sign(g) == np.sign(fd)
    max_delta = float(np.max(np.abs(g80-g)))
    grad_limit = float(1e-2 * np.max(np.abs(g)))
    objective_rel = relative(abs(value80-value), abs(value))
    precision_rel = relative(abs(value-value64), abs(value64))
    declared = next(row for row in sweep if row["r"] == 1e-3)
    plateau = [row for row in sweep if row["r"] in (1e-2, 3e-3, 1e-3)]
    gates = {}

    def gate(name, threshold, metrics, passed, resolved=True):
        status = "CANNOT-JUDGE" if not resolved else ("PASS" if passed else "FAIL")
        gates[name] = dict(threshold=threshold, metrics=metrics, status=status)
        print(f"{name} {status}: threshold={threshold}; measured={json.dumps(metrics)}", flush=True)

    gate("G1", "0.02 <= mean |S11|^2 <= 0.5", dict(objective=value), 0.02 <= value <= 0.5)
    gate("G2", "declared directional FD ULP span >= 1e4", dict(ulp_span=declared["ulp_span"]),
         True, declared["resolution"] == "PASS")
    gate("G3", "directional relative error <= 1e-2 at r=1e-2,3e-3,1e-3",
         dict(declared_relative_error=declared["relative_error"],
              plateau_relative_errors=[row["relative_error"] for row in plateau]),
         all(row["relative_error"] <= 1e-2 for row in plateau),
         all(row["resolution"] == "PASS" for row in plateau))
    gate("G4", "vector relative error <= 5e-2; signs agree wherever |FD| >= 1e-2 * max|FD|; FD spans >= 1e4",
         dict(vector_relative_error=vec_rel, significant_threshold=float(1e-2*np.max(np.abs(fd))),
              significant_cells=CELLS[significant].tolist(),
              sign_mismatch_cells=CELLS[significant & ~sign_matches].tolist(),
              min_ulp_span=min(row["ulp_span"] for row in cell_fd)),
         vec_rel <= 5e-2 and bool(np.all(sign_matches[significant])),
         all(row["resolution"] == "PASS" for row in cell_fd))
    gate("G5", "max |g80-g40| <= 1e-2 * max|g40|; |L80-L40|/|L40| <= 1e-3",
         dict(max_gradient_difference=max_delta, gradient_difference_limit=grad_limit,
              objective40=value, objective80=value80, objective_relative_change=objective_rel),
         max_delta <= grad_limit and objective_rel <= 1e-3)
    print(f"forward precision: float32={value:.17g} float64={value64:.17g} "
          f"absolute_difference={abs(value-value64):.17g} relative_difference={precision_rel:.17g} (information only)", flush=True)
    overall = "PASS" if all(gate["status"] == "PASS" for gate in gates.values()) else "FAIL"
    record.update(objective_float32=value, objective_float64=value64,
                  precision_absolute_difference=abs(value-value64), precision_relative_difference=precision_rel,
                  ad_gradient=g.tolist(), direction=direction.tolist(), directional_ad=dd_ad,
                  step_sweep=sweep, per_cell_fd=cell_fd, objective80=value80,
                  ad_gradient80=g80.tolist(), record_gradient_absolute_difference=np.abs(g80-g).tolist(),
                  float64_grid_matches=True, float64_loss_dtype="float64",
                  gates=gates, overall=overall, wall_seconds=time.perf_counter()-started)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "verdict.json"
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"OVERALL {overall}", flush=True)
    print(f"wall_seconds={record['wall_seconds']:.6f}", flush=True)
    print(f"wrote {output}", flush=True)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
