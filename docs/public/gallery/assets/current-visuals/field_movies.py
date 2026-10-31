"""Render actual RFX field snapshots for three introductory gallery examples.

Run from a checkout with ``python -m pip install -e .`` and ffmpeg installed::

    JAX_PLATFORMS=cpu PYTHONPATH=. python examples/visualization/field_movies.py --case all

These finite-duration, single-mesh demonstrations are qualitative teaching
examples, not accuracy, absorption, frequency, Q, or performance benchmarks.
Each model uses the public Simulation API; every displayed position and frame
time comes from Result.snapshot_axes. No phasor replay or interpolated frames.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import rfx
from rfx import Box, GaussianPulse, ModulatedGaussian, Simulation, SnapshotSpec

C0 = 299_792_458.0
DOMAIN = (0.080, 0.060, 0.001)
DX = 0.001
CORE_COMMIT = "a9054338616b91bb1364699882c2530fa29a0048"
CASES = {
    "boundary-reflection": {"variants": ("cpml", "pec"), "steps": 640, "interval": 4,
                            "title": "Let a pulse leave, or reflect it back"},
    "cavity-standing-wave": {"variants": ("cavity",), "steps": 1600, "interval": 8,
                             "title": "A pulse leaves a standing wave in a closed cavity"},
    "dielectric-interface": {"variants": ("vacuum", "dielectric"), "steps": 640, "interval": 4,
                             "title": "Watch a pulse meet a dielectric interface"},
}
LABELS = {"cpml": "Open domain · CPML", "pec": "Closed domain · PEC",
          "cavity": "Closed rectangular cavity · PEC", "vacuum": "Vacuum reference",
          "dielectric": "Dielectric half-space · εᵣ = 4"}


def source_settings(variant: str) -> tuple:
    """Return physical source position, central frequency and bandwidth."""
    if variant == "cavity":
        f0 = 0.5 * C0 * np.sqrt(DOMAIN[0] ** -2 + DOMAIN[1] ** -2)
        return (0.040, 0.030, 0.0), float(f0), 0.35
    return (0.020, 0.030, 0.0), 6.0e9, 0.8


def build_simulation(variant: str) -> Simulation:
    """Build one public example without solving; also used by fidelity checks."""
    if variant not in LABELS:
        raise ValueError(f"Unknown variant: {variant}")
    boundary = "pec" if variant in ("pec", "cavity") else "cpml"
    sim = Simulation(freq_max=12.0e9, domain=DOMAIN, dx=DX,
                     boundary=boundary, cpml_layers=16, mode="2d_tmz")
    if variant == "dielectric":
        sim.add_material("dielectric", eps_r=4.0)
        # Extend the half-space through the absorbing layers. The interface
        # lies on the declared 40 mm grid plane; there is no rear slab face.
        sim.add(Box((0.040, -1.0, -1.0), (1.0, 1.0, 1.0)), material="dielectric")
    position, f0, bandwidth = source_settings(variant)
    waveform_type = ModulatedGaussian if variant == "cavity" else GaussianPulse
    sim.add_source(position, "ez", waveform=waveform_type(
        f0=f0, bandwidth=bandwidth, amplitude=1.0e-8, cutoff=3.0),
        amplitude_kind="current")
    sim.add_probe((0.055, 0.030, 0.0), "ez")
    return sim


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_core_commit() -> str | None:
    """Name the original solver commit only when the loaded code matches it."""
    repo = Path(rfx.__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        return None
    check = subprocess.run(["git", "diff", "--quiet", CORE_COMMIT, "--", "rfx"], cwd=repo)
    return CORE_COMMIT if check.returncode == 0 else None


def calculate(case: str, output: Path) -> tuple[dict, dict]:
    """Run the models once, retaining the exact field samples and preflight."""
    config = CASES[case]
    arrays, runs = {}, []
    for variant in config["variants"]:
        print(f"\n{case}: {variant}", flush=True)
        sim = build_simulation(variant)
        preflight = io.StringIO()
        with redirect_stdout(preflight):
            report = sim.preflight()
        print(preflight.getvalue(), end="", flush=True)
        if report.errors:
            raise RuntimeError("Preflight errors: refusing to render this model")
        result = sim.run(n_steps=config["steps"], compute_s_params=False,
                         snapshot=SnapshotSpec(interval=config["interval"],
                                               components=("ez",),
                                               slice_axis=2, slice_index=0))
        axes = result.snapshot_axes["ez"]
        fields = np.asarray(result.snapshots["ez"])
        x, y = np.asarray(axes.coords["x"]), np.asarray(axes.coords["y"])
        # Keep the physical interior for display; the API includes CPML pads.
        xi = np.flatnonzero((x >= -1e-12) & (x <= DOMAIN[0] + 1e-12))
        yi = np.flatnonzero((y >= -1e-12) & (y <= DOMAIN[1] + 1e-12))
        fields = fields[:, xi][:, :, yi]
        x, y = x[xi], y[yi]
        trace = np.asarray(result.time_series)[:, 0]
        assert axes.dims == ("frame", "x", "y")
        assert fields.shape == (len(axes.times_s), len(x), len(y))
        assert np.isfinite(fields).all() and np.isfinite(trace).all()
        assert np.max(np.abs(fields)) > 0
        np.testing.assert_allclose(x[[0, -1]], [0, DOMAIN[0]], atol=1e-12)
        np.testing.assert_allclose(y[[0, -1]], [0, DOMAIN[1]], atol=1e-12)
        np.testing.assert_array_equal(axes.steps, np.arange(
            config["interval"], config["steps"] + 1, config["interval"]))
        arrays[f"{variant}_ez_V_per_m"] = fields
        arrays[f"{variant}_probe_ez_V_per_m"] = trace
        arrays[f"{variant}_probe_times_s"] = np.arange(1, len(trace) + 1) * result.dt
        arrays[f"{variant}_frame_times_s"] = np.asarray(axes.times_s)
        arrays[f"{variant}_frame_steps"] = np.asarray(axes.steps)
        arrays[f"{variant}_x_m"], arrays[f"{variant}_y_m"] = x, y
        position, f0, bandwidth = source_settings(variant)
        runs.append({"variant": variant, "boundary": "pec" if variant in ("pec", "cavity") else "cpml",
                     "domain_m": list(DOMAIN), "realized_display_span_m":
                     {"x": [float(x[0]), float(x[-1])], "y": [float(y[0]), float(y[-1])]},
                     "mode": "2d_tmz", "dx_m": DX, "dt_s": float(result.dt),
                     "allocated_grid_shape": list(result.grid.shape),
                     "n_steps": config["steps"], "snapshot_interval": config["interval"],
                     "source": {"position_m": list(position), "component": "ez",
                                "waveform": "ModulatedGaussian" if variant == "cavity" else "GaussianPulse",
                                "f0_Hz": f0, "bandwidth": bandwidth, "cutoff": 3.0,
                                "amplitude_A": 1e-8, "amplitude_kind": "current"},
                     "probe_position_m": [0.055, 0.030, 0.0],
                     "snapshot_slice_coord_m": float(axes.slice_coord),
                     "snapshot_dims": list(axes.dims),
                     "preflight_stdout": preflight.getvalue(),
                     "preflight_codes": [finding.code for finding in report]})
    np.savez_compressed(output / f"{case}.npz", **arrays)
    metadata = {"schema_version": 1, "case": case, "title": config["title"],
                "media_kind": "time-domain-fdtd-snapshots",
                "core_source_commit": verified_core_commit(),
                "generator": "examples/visualization/field_movies.py",
                "generator_sha256": sha256(Path(__file__)),
                "environment": {"python": platform.python_version(), "jax": jax.__version__,
                                "numpy": np.__version__, "backend": jax.default_backend()},
                "evidence_status": "qualitative single-mesh teaching example",
                "limits": ["No mesh-refinement or run-length accuracy claim",
                           "No measured reflection, transmission, absorption, resonance frequency, or Q",
                           "2D TMz model; this is not a finite-height 3D device",
                           "CPML is a finite numerical absorber and may reflect residual fields"],
                "runs": runs}
    return arrays, metadata


def render(case: str, output: Path, arrays: dict, metadata: dict) -> None:
    """Create a movie and a poster from saved samples, without modifying data."""
    variants = CASES[case]["variants"]
    frames = [arrays[f"{v}_ez_V_per_m"] for v in variants]
    times = [arrays[f"{v}_frame_times_s"] for v in variants]
    for t in times[1:]:
        np.testing.assert_allclose(t, times[0], rtol=1e-12)
    # A shared, fixed linear scale across all frames AND compared models.
    # Only the tiny current-injection neighborhood is excluded when finding
    # the scale. Its pixels remain in the data and can saturate in the image.
    reference = 0.0
    for variant, field in zip(variants, frames):
        x, y = arrays[f"{variant}_x_m"], arrays[f"{variant}_y_m"]
        position = source_settings(variant)[0]
        outside_source = (x[:, None] - position[0]) ** 2 + (y[None, :] - position[1]) ** 2 >= 0.004 ** 2
        reference = max(reference, float(np.max(np.abs(field[:, outside_source]))))
    n = len(variants)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(10.0, 7.0), layout="constrained", facecolor="#f8fafc")
    gs = fig.add_gridspec(2, n, height_ratios=(3, 1.15))
    field_axes, artists = [], []
    for col, (variant, field) in enumerate(zip(variants, frames)):
        ax = fig.add_subplot(gs[0, col])
        x, y = arrays[f"{variant}_x_m"], arrays[f"{variant}_y_m"]
        im = ax.pcolormesh(x * 1e3, y * 1e3, (field[0] / reference).T,
                           cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=LABELS[variant],
               xlim=(0, DOMAIN[0] * 1e3), ylim=(0, DOMAIN[1] * 1e3), aspect="equal")
        sx, sy, _ = source_settings(variant)[0]
        ax.plot(sx * 1e3, sy * 1e3, marker="+", color="#0f172a", ms=8, label="current source")
        ax.plot(55, 30, marker="o", mfc="none", mec="#14532d", ms=6, label="probe")
        if variant == "dielectric":
            ax.axvline(40, color="#334155", ls="--", lw=1.2)
            ax.text(58, 56, "εᵣ = 4", color="#0f172a", ha="center")
            ax.text(20, 56, "vacuum", color="#0f172a", ha="center")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
        artists.append(im)
        field_axes.append(ax)
    cb = fig.colorbar(artists[0], ax=field_axes, fraction=0.027, pad=0.015)
    cb.set_label("Eᶻ / Eref (fixed linear scale)")
    trace_ax = fig.add_subplot(gs[1, :])
    colors = ("#0369a1", "#b45309")
    for variant, color in zip(variants, colors):
        trace_ax.plot(arrays[f"{variant}_probe_times_s"] * 1e9,
                      arrays[f"{variant}_probe_ez_V_per_m"] / reference,
                      label=LABELS[variant], color=color, lw=1.2)
    trace_ax.set(xlabel="Simulation time (ns)", ylabel="Probe Eᶻ / Eref",
                 xlim=(0, times[0][-1] * 1e9))
    trace_ax.grid(alpha=0.16)
    trace_ax.legend(loc="upper right", fontsize=8)
    marker = trace_ax.axvline(times[0][0] * 1e9, color="#0f172a", lw=1)
    clock = fig.suptitle("", fontsize=15, fontweight="semibold")
    fig.supxlabel("Actual FDTD samples · 2D TMz · qualitative example\n"
                  "Fixed scale excludes 4 mm source neighborhood; source pixels may saturate.", fontsize=9)

    def frame(index):
        for artist, field in zip(artists, frames):
            artist.set_array((field[index] / reference).T)
        marker.set_xdata([times[0][index] * 1e9] * 2)
        clock.set_text(f"{CASES[case]['title']}\nt = {times[0][index] * 1e9:.3f} ns")
        return artists + [marker, clock]

    # Posters show a visible field rather than the nearly-zero first frame.
    poster_index = int(len(times[0]) * (0.7 if case == "cavity-standing-wave" else 0.20))
    frame(poster_index)
    fig.savefig(output / f"{case}.png", dpi=120, facecolor=fig.get_facecolor())
    movie = animation.FuncAnimation(fig, frame, frames=len(times[0]), interval=60, blit=False)
    movie.save(output / f"{case}.mp4", writer=animation.FFMpegWriter(
        fps=15, codec="libx264", bitrate=550,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2"]), dpi=100)
    plt.close(fig)
    # Probe CSV has physical units and every recorded time step, not movie samples.
    columns, names = [], []
    for variant in variants:
        columns.extend((arrays[f"{variant}_probe_times_s"], arrays[f"{variant}_probe_ez_V_per_m"]))
        names.extend((f"{variant}_time_s", f"{variant}_probe_ez_V_per_m"))
    np.savetxt(output / f"{case}.csv", np.column_stack(columns), delimiter=",",
               header=",".join(names), comments="")
    metadata["display"] = {"component": "ez", "normalization_V_per_m": reference,
                           "scale": "linear -1 to +1 shared across all frames and variants",
                           "normalization_excludes_source_radius_m": 0.004,
                           "source_pixels_may_saturate": True, "fps": 15,
                           "frame_count": len(times[0]), "poster_frame_index": poster_index,
                           "poster_time_s": float(times[0][poster_index]),
                           "frame_interpolation": False}
    metadata["artifacts"] = [{"file": f"{case}.{ext}", "sha256": sha256(output / f"{case}.{ext}"),
                              "bytes": (output / f"{case}.{ext}").stat().st_size}
                             for ext in ("mp4", "png", "csv", "npz")]
    (output / f"{case}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {case}: {len(times[0])} actual frames, Eref={reference:.6g} V/m", flush=True)


def publish(case: str, output: Path, destination: Path) -> None:
    """Publish a small genuine-frame subset; full arrays stay in output."""
    if output.resolve() == destination.resolve():
        raise ValueError("Full output and publication directory must differ")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), destination / "field_movies.py")
    metadata = json.loads((output / f"{case}.json").read_text())
    with np.load(output / f"{case}.npz", allow_pickle=False) as archive:
        selected = {key: value[::16] if (key.endswith("_ez_V_per_m") and "probe" not in key)
                    or "_frame_" in key else value for key, value in archive.items()}
    np.savez_compressed(destination / f"{case}.npz", **selected)
    for extension in ("mp4", "png", "csv"):
        shutil.copyfile(output / f"{case}.{extension}", destination / f"{case}.{extension}")
    metadata["download"] = {"snapshot_frame_stride": 16,
                            "description": "Every 16th actual movie frame, full spatial resolution; all probe samples",
                            "full_arrays": "Regenerate with the script; full arrays remain in --output"}
    metadata["artifacts"] = [{"file": f"{case}.{ext}", "sha256": sha256(destination / f"{case}.{ext}"),
                              "bytes": (destination / f"{case}.{ext}").stat().st_size}
                             for ext in ("mp4", "png", "csv", "npz")]
    for artifact in metadata["artifacts"]:
        if artifact["bytes"] > 1_000_000:
            raise RuntimeError(f"Publication artifact exceeds budget: {artifact['file']}")
    (destination / f"{case}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("field-movies"))
    parser.add_argument("--publish-to", type=Path, help="copy media plus sparse NPZ under 1 MB per file")
    parser.add_argument("--render-only", action="store_true", help="reuse NPZ + JSON without solving")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode:
        raise RuntimeError("ffmpeg is required for MP4 generation")
    for case in CASES if args.case == "all" else (args.case,):
        if args.render_only:
            with np.load(args.output / f"{case}.npz", allow_pickle=False) as archive:
                arrays = dict(archive)
            metadata = json.loads((args.output / f"{case}.json").read_text())
        else:
            arrays, metadata = calculate(case, args.output)
            # Preserve metadata before rendering so a rendering retry needs no solve.
            (args.output / f"{case}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        render(case, args.output, arrays, metadata)
        if args.publish_to:
            publish(case, args.output, args.publish_to)


if __name__ == "__main__":
    main()
