"""Native CPML vacuum dipole waveform witness: run the fixture, then judge it.

Standalone runner for the Experiment 68/69 fixture (docs/design_notes/
2026-09-13_distributed_cpml_vacuum_gpu68.md, ..._cpml_stagger_gpu69.md, and
the requirement text in ..._2026-09-12_distributed_cpml_waveform_protocol.md,
all on research branch accel/pod-distributed-usable-20260912).  It builds the
fixture through the PUBLIC ``rfx.Simulation`` API only, records the probe
traces, and evaluates them against an analytic continuum dipole reference.

PROVENANCE.  Every function between the ``--- BEGIN`` / ``--- END`` markers
below is copied verbatim (modulo this module's constants) from that research
branch's ``scripts/diagnostics/distributed_cpml_dipole_witness.py`` at commit
7b6c33d5 -- ``reference()``, ``check_traces()`` and the moment/broadside
helpers they call.  Do not "improve" them: their numbers are what Experiment
68 (far-Hy full residual 0.13937570732, FAIL) and Experiment 69 (0.000388194-
010609, PASS) were measured with, and a changed comparator would make those
two receipts incomparable.  What is NOT copied is the pod checker's
``_load_inputs`` fixture-admission path: that authenticated a recorded
evidence directory by manifest SHA256, and this script runs the solver itself,
so it re-derives the same realization facts from the built grid instead.

SCOPE.  The four fixed channels measure TOTAL waveform disagreement with a
specified continuum dipole.  Bulk discretization, finite-cell source
representation, source sampling and boundary effects all sit inside that
total.  Passing does NOT separate those contributions, does not certify a CPML
reflection coefficient, and does not qualify other meshes, angles or
materials.  The ordinary causal derivatives omit the distributional onset
impulse of the initial current jump.  No amplitude or time fitting is done and
no window is selected from the output.

The recorded source table is regenerated assembly evidence (the same
boundary-selected builder the solver calls), not an in-scan injection capture;
that limitation is inherited from the pod protocol and is not weakened here.

Exit codes (repo convention): 0 = every fixed requirement passes, 1 = a
requirement failed, 2 = the fixture or inputs were not admissible.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np

EPS_0: Final = 8.8541878128e-12
MU_0: Final = 1.25663706212e-6
DX: Final = 1 / 512
DT: Final = 3.723779042052746e-12
STEPS: Final = 1536
SHAPE: Final = (96, 72, 60)
CPML_LAYERS: Final = 12
TAU: Final = 1 / (math.pi * 2.5e9 * .5)
CUTOFF: Final = 4.5
CHANNELS: Final = ("middle_ez", "middle_hy", "far_ez", "far_hy")
COLUMNS: Final = (8, 10, 14, 16)
RADII_CELLS: Final = (23., 23.5, 50., 50.5)
FULL_TOL: Final = .05
LATE_TOL: Final = .01
SOURCE_TOL: Final = 1e-5

# The Experiment 68/69 fixture, spelled exactly as examples/distributed_-
# verified.py's ``fixture_spec("main", "cpml", "vacuum")`` spells it on the
# research branch. These are the literal doubles, not derived quantities.
DOMAIN: Final = (0.138671875, 0.091796875, 0.068359375)
SOURCE_POSITION: Final = (0.021484375, 0.0390625, 0.03125)
PROBE_POSITIONS: Final = ((0.025390625, 0.0390625, 0.03125),
                          (0.06640625, 0.0390625, 0.03125),
                          (0.119140625, 0.0390625, 0.03125))
FIELDS: Final = ("ex", "ey", "ez", "hx", "hy", "hz")
WAVEFORM: Final = dict(f0=2.5e9, bandwidth=0.5, amplitude=1.0, cutoff=4.5)
EXPECTED_SOURCE_INDEX: Final = (23, 32, 28)
EXPECTED_PROBE_INDICES: Final = ((25, 32, 28), (46, 32, 28), (73, 32, 28))


# --- BEGIN copied from research 7b6c33d5 scripts/diagnostics/distributed_cpml_dipole_witness.py

class Terms(NamedTuple):
    e_near: np.ndarray
    e_induction: np.ndarray
    e_radiation: np.ndarray
    h_induction: np.ndarray
    h_radiation: np.ndarray

    @property
    def electric(self) -> np.ndarray:
        return self.e_near + self.e_induction + self.e_radiation

    @property
    def magnetic(self) -> np.ndarray:
        return self.h_induction + self.h_radiation


class Reference(NamedTuple):
    values: np.ndarray
    times: np.ndarray
    late: np.ndarray
    source: np.ndarray


def medium(epsilon_r: float) -> tuple[float, float]:
    """Return epsilon and speed only for the declared homogeneous media."""
    if epsilon_r not in (1., 4.):
        raise ValueError("unsupported_homogeneous_medium")
    epsilon = epsilon_r * EPS_0
    return epsilon, 1 / math.sqrt(epsilon * MU_0)


def source_waveform(times: np.ndarray) -> np.ndarray:
    """Prescribed differentiated Gaussian sampled on its source clock."""
    u = (np.asarray(times, dtype=np.float64) - CUTOFF * TAU) / TAU
    return -2 * u * np.exp(-u * u)


def moments(times: np.ndarray, epsilon_r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal p, p-dot and ordinary p-double-dot in physical source time."""
    epsilon, _ = medium(epsilon_r)
    times = np.asarray(times, dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("nonfinite_source_times")
    shifted = times - DT / 2
    u = (shifted - CUTOFF * TAU) / TAU
    g = np.exp(-u * u)
    g0 = math.exp(-((-DT / 2 - CUTOFF * TAU) / TAU)**2)
    strength = epsilon * DX**3 / DT
    active = times > 0
    p = np.where(active, -strength * TAU * (g - g0), 0.)
    rate = np.where(active, -strength * source_waveform(shifted), 0.)
    acceleration = np.where(active, -strength * (4 * u * u - 2) * g / TAU, 0.)
    return p, rate, acceleration


def broadside(displacement_x: float, times: np.ndarray, epsilon_r: float) -> Terms:
    """All broadside terms; signed x displacement fixes magnetic direction."""
    if not math.isfinite(displacement_x) or displacement_x == 0:
        raise ValueError("invalid_broadside_displacement")
    epsilon, speed = medium(epsilon_r)
    radius = abs(displacement_x)
    p, rate, acceleration = moments(np.asarray(times) - radius / speed, epsilon_r)
    electric_factor = -1 / (4 * math.pi * epsilon)
    magnetic_factor = math.copysign(1., displacement_x) / (4 * math.pi)
    return Terms(
        electric_factor * p / radius**3,
        electric_factor * rate / (speed * radius**2),
        electric_factor * acceleration / (speed**2 * radius),
        magnetic_factor * rate / radius**2,
        magnetic_factor * acceleration / (speed * radius),
    )


def reference(epsilon_r: float) -> Reference:
    """Generate fixed-position, staggered-time reference without fitting."""
    _, speed = medium(epsilon_r)
    frames = np.arange(STEPS, dtype=np.float64)
    electric_times, magnetic_times = (frames + 1) * DT, (frames + .5) * DT
    times = np.column_stack((electric_times, magnetic_times, electric_times, magnetic_times))
    values = np.empty((STEPS, 4), dtype=np.float64)
    for j, cells in enumerate(RADII_CELLS):
        terms = broadside(cells * DX, times[:, j], epsilon_r)
        values[:, j] = terms.electric if j % 2 == 0 else terms.magnetic
    late_time = 2 * CUTOFF * TAU + DT / 2 + RADII_CELLS[-1] * DX / speed + 4 * DT
    return Reference(values, times, times >= late_time, source_waveform(frames * DT))


def check_traces(trace: np.ndarray, source: np.ndarray, epsilon_r: float) -> dict:
    """Check all fixed samples; caller must authenticate fixture metadata."""
    if (trace.shape != (STEPS, 18) or source.shape != (STEPS,)
            or trace.dtype != np.float32 or source.dtype != np.float32
            or not np.isfinite(trace).all() or not np.isfinite(source).all()):
        raise ValueError("invalid_complete_float32_trace_or_source")
    expected = reference(epsilon_r)
    source_error = float(np.max(np.abs(source.astype(np.float64) - expected.source))
                         / np.max(np.abs(expected.source)))
    channels = {}
    for j, (name, column) in enumerate(zip(CHANNELS, COLUMNS, strict=True)):
        target = expected.values[:, j]
        scale = float(np.max(np.abs(target)))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("unresolved_reference_channel")
        actual = trace[:, column].astype(np.float64)
        residual = actual - target
        full_error = float(np.max(np.abs(residual)) / scale)
        late_error = float(np.max(np.abs(residual[expected.late[:, j]])) / scale)
        channels[name] = dict(
            full_relative_max=full_error, late_relative_max=late_error,
            full_pass=full_error <= FULL_TOL, late_pass=late_error <= LATE_TOL,
            reference_peak=scale, late_start_frame=int(np.flatnonzero(expected.late[:, j])[0]),
            reference=target.tolist(), observed=actual.tolist(), residual=residual.tolist(),
        )
    epsilon, _ = medium(epsilon_r)
    return dict(
        schema="distributed-cpml-dipole/v1",
        passed=source_error <= SOURCE_TOL and all(
            value["full_pass"] and value["late_pass"] for value in channels.values()),
        channels=channels, source_relative_peak_error=source_error,
        source_pass=source_error <= SOURCE_TOL,
        deposited_moment_end=-epsilon * DX**3 * float(np.sum(source, dtype=np.float64)),
        smooth_moment_end=float(moments(np.array([STEPS * DT]), epsilon_r)[0][0]),
        source_reference=expected.source.tolist(), source_observed=source.tolist(),
        time_e=expected.times[:, 0].tolist(), time_h=expected.times[:, 1].tolist(),
        epsilon_r=epsilon_r, full_tolerance=FULL_TOL, late_tolerance=LATE_TOL,
        source_tolerance=SOURCE_TOL, bulk_error_certified=False,
        cpml_reflection_coefficient_certified=False, onset_impulse_resolved=False,
        scope="fixed_four_channel_total_waveform_requirements",
    )

# --- END copied from research 7b6c33d5


def build_simulation() -> Any:
    """Build the Experiment 68/69 native CPML vacuum dipole fixture.

    Public API only: no runner internals, no replacement source, no snapshot
    request (traces only -- full float32 histories are ~14.2 GiB and are not
    part of this requirement).
    """
    from rfx import GaussianPulse, Simulation

    sim = Simulation(domain=list(DOMAIN), dx=DX, freq_max=5e9, boundary="cpml",
                     cpml_layers=CPML_LAYERS, cpml_kappa_max=1.0,
                     precision="float32", solver="yee", stencil_order=2)
    # Vacuum: no material, no Box. The dipole is the existing POSITIVE Ez
    # field increment (amplitude_kind="field"), not a unit current.
    sim.add_source(SOURCE_POSITION, "ez",
                   waveform=GaussianPulse(**WAVEFORM), amplitude_kind="field")
    for position in PROBE_POSITIONS:
        for component in FIELDS:
            sim.add_probe(position, component)
    return sim


def admit_realization(sim: Any) -> dict:
    """Re-derive, from the built grid, the facts the pod manifest authenticated.

    Replaces the copied checker's ``_load_inputs``: this script runs the solver
    rather than reading a signed evidence directory, so the fixture identity is
    established from the realized grid, materials and raster instead of a
    manifest SHA256. Any mismatch is exit 2, never a relaxed comparison.
    """
    grid = sim._build_grid()
    if tuple(grid.shape) != SHAPE:
        raise ValueError(f"realized_shape_mismatch:{tuple(grid.shape)}")
    if float(grid.dx) != DX or float(grid.dt) != DT:
        raise ValueError(f"realized_mesh_mismatch:dx={grid.dx!r},dt={grid.dt!r}")
    if tuple(grid.face_pads) != (CPML_LAYERS,) * 6 or float(grid.kappa_max) != 1.0:
        raise ValueError(f"realized_cpml_mismatch:{tuple(grid.face_pads)}")
    assembled = sim._assemble_materials(grid)
    materials = assembled[0]
    for name, value in (("eps_r", 1.0), ("sigma", 0.0), ("mu_r", 1.0)):
        array = np.asarray(getattr(materials, name))
        if (array.dtype != np.float32 or array.shape != SHAPE
                or not np.all(array == np.float32(value))):
            raise ValueError("unsupported_homogeneous_material:" + name)
    # Grid carries no coordinate arrays, so the raster convention the pod
    # checker verified via grid_{x,y,z}.npy (x_i = (i - n_pad) * dx) is checked
    # here as the half-open raster rule position_to_index obeys, on both the
    # source and all three probes -- the same expectation
    # examples/distributed_verified.py asserts.
    source_index = tuple(grid.position_to_index(SOURCE_POSITION))
    if source_index != EXPECTED_SOURCE_INDEX:
        raise ValueError(f"source_raster_mismatch:{source_index}")
    probe_indices = tuple(tuple(grid.position_to_index(p)) for p in PROBE_POSITIONS)
    if probe_indices != EXPECTED_PROBE_INDICES:
        raise ValueError(f"probe_raster_mismatch:{probe_indices}")
    for position, index in ((SOURCE_POSITION, source_index),
                            *zip(PROBE_POSITIONS, probe_indices, strict=True)):
        literal = tuple(int(p / DX) + CPML_LAYERS for p in position)
        if literal != index:
            raise ValueError(f"unsupported_grid_coordinates:{position}!={index}")
    return dict(shape=list(SHAPE), dx=DX, dt=DT,
                face_cpml_pads=[CPML_LAYERS] * 6,
                source_index=list(source_index),
                probe_indices=[list(index) for index in probe_indices],
                material="vacuum", eps_r=1.0, sigma=0.0, mu_r=1.0,
                steps=STEPS, cpml_layers=CPML_LAYERS, cpml_kappa_max=1.0,
                precision="float32", solver="yee", stencil_order=2)


def assembly_source_table(sim: Any) -> np.ndarray:
    """Regenerate the source table through the same boundary-selected builder.

    Assembly evidence, NOT an in-scan injection capture -- the pod protocol's
    own caveat, kept verbatim. The source requirement below therefore checks
    the table the solver is handed, not the increments it actually applied.
    """
    from rfx import GaussianPulse
    from rfx.simulation import make_j_source

    grid = sim._build_grid()
    materials = sim._assemble_materials(grid)[0]
    built = make_j_source(grid, SOURCE_POSITION, "ez", GaussianPulse(**WAVEFORM),
                          STEPS, materials=materials, amplitude_kind="field")
    table = np.asarray(built.waveform, dtype=np.float32)
    if table.shape != (STEPS,):
        raise ValueError(f"source_table_shape:{table.shape}")
    return table


def run_fixture(sim: Any) -> tuple[np.ndarray, dict]:
    """One public ``Simulation.run``; return the (1536, 18) float32 trace."""
    import jax

    result = sim.run(n_steps=STEPS, compute_s_params=False)
    trace = np.asarray(result.time_series)
    if trace.shape != (STEPS, 18):
        raise ValueError(f"trace_shape_mismatch:{trace.shape}")
    trace = trace.astype(np.float32, copy=False)
    if int(result.state.step) != STEPS:
        raise ValueError(f"final_step_mismatch:{int(result.state.step)}")
    if getattr(result, "snapshots", None) is not None:
        raise ValueError("unexpected_full_field_capture")
    runtime = dict(
        jax_version=jax.__version__,
        devices=[dict(id=d.id, kind=d.device_kind, platform=d.platform)
                 for d in jax.devices()],
        python=platform.python_version(), machine=platform.machine(),
        realized_dt=float(result.dt),
        live_e=bool(np.any(np.asarray(result.state.ez))),
    )
    return trace, runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON verdict path; must not already exist")
    parser.add_argument("--trace-output", type=Path, default=None,
                        help="optional .npy for the raw (1536, 18) float32 trace")
    parser.add_argument("--label", default="",
                        help="free-text arm label recorded in the verdict")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output_exists")
        import rfx

        sim = build_simulation()
        realization = admit_realization(sim)
        source = assembly_source_table(sim)
        trace, runtime = run_fixture(sim)
        if args.trace_output is not None:
            with args.trace_output.open("xb") as stream:
                np.save(stream, trace)
        report = check_traces(trace, source, 1.0)
        report["realization"] = realization
        report["runtime"] = runtime
        report["label"] = args.label
        report["rfx_version"] = getattr(rfx, "__version__", "unknown")
        report["source_table_provenance"] = (
            "regenerated_assembly_evidence_not_in_scan_injection_capture")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({
            "label": args.label,
            "passed": report["passed"],
            "source_pass": report["source_pass"],
            "full_tolerance": FULL_TOL,
            "late_tolerance": LATE_TOL,
            "channels": {name: {key: channel[key] for key in
                                ("full_relative_max", "late_relative_max",
                                 "full_pass", "late_pass")}
                         for name, channel in report["channels"].items()},
        }, indent=2))
        return 0 if report["passed"] else 1
    except (ValueError, KeyError, TypeError, OSError, ImportError,
            AttributeError) as exc:
        print(json.dumps({"status": "invalid", "label": args.label,
                          "error": f"{type(exc).__name__}: {exc}"}),
              file=sys.stdout)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
