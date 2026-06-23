"""Config-driven command-line interface for rfx.

Subcommands
-----------
``rfx run <config.yaml> [--output result.h5] [--num-periods N] [--compute-s-params]``
    Build a simulation from YAML, run it, and save the result dataset.
``rfx plot <result.h5> [--quantity s_params|time_series] [--out fig.png]``
    Plot S-parameters or probe time series from a saved result.
``rfx export-geometry <config.yaml> --output geo.json``
    Build the simulation and export its geometry as JSON.

Uses only the standard-library :mod:`argparse` (no extra CLI deps). Bad
config surfaces a clear, loud error and a non-zero exit code — never a
silent pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(args) -> int:
    from rfx.config import run_and_save

    run_kwargs: dict = {}
    if args.num_periods is not None:
        run_kwargs["num_periods"] = args.num_periods
    if args.compute_s_params:
        run_kwargs["compute_s_params"] = True

    result = run_and_save(args.config, args.output, **run_kwargs)
    print(f"Ran simulation from {args.config}")
    if getattr(result, "s_params", None) is not None:
        print(f"  S-parameters: shape {tuple(result.s_params.shape)}")
    if getattr(result, "time_series", None) is not None:
        import numpy as np
        print(f"  time series:  shape {tuple(np.asarray(result.time_series).shape)}")
    print(f"  saved -> {args.output}")
    return 0


def _cmd_plot(args) -> int:
    import h5py
    import numpy as np

    path = Path(args.result)
    if not path.exists():
        raise FileNotFoundError(f"result file not found: {path}")

    from rfx.visualize import plot_s_params, plot_time_series

    with h5py.File(path, "r") as f:
        if args.quantity == "s_params":
            if "output/s_params" not in f:
                raise KeyError(
                    f"{path} has no 'output/s_params' dataset; run with "
                    f"compute_s_params enabled or use --quantity time_series."
                )
            s_params = np.asarray(f["output/s_params"])
            if "output/freqs" not in f:
                raise KeyError(f"{path} has no 'output/freqs' dataset")
            freqs = np.asarray(f["output/freqs"])
            fig = plot_s_params(s_params, freqs)
        elif args.quantity == "time_series":
            if "output/time_series" not in f:
                raise KeyError(
                    f"{path} has no 'output/time_series' dataset; add probes "
                    f"to the config or use --quantity s_params."
                )
            ts = np.asarray(f["output/time_series"])
            dt = None
            if "input" in f and "dt" in f["input"].attrs:
                dt = float(f["input"].attrs["dt"])
            if dt is None:
                raise KeyError(
                    f"{path} has no input/dt attribute; time-series plotting "
                    f"needs the timestep. Re-run so the dataset stores config."
                )
            fig = plot_time_series(ts, dt)
        else:  # pragma: no cover - argparse choices guard this
            raise ValueError(f"unknown quantity {args.quantity!r}")

    out = args.out or f"{path.stem}_{args.quantity}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {args.quantity} plot -> {out}")
    return 0


def _cmd_export_geometry(args) -> int:
    from rfx.config import simulation_from_yaml
    from rfx.io import export_geometry_json

    sim = simulation_from_yaml(args.config)
    export_geometry_json(args.output, sim)
    print(f"Exported geometry from {args.config} -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfx",
        description="Config-driven CLI for the rfx FDTD simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a simulation from a YAML config")
    p_run.add_argument("config", help="path to the YAML config file")
    p_run.add_argument(
        "--output", "-o", default="result.h5",
        help="output HDF5 dataset path (default: result.h5)",
    )
    p_run.add_argument(
        "--num-periods", type=float, default=None,
        help="override the number of periods at freq_max for the run length",
    )
    p_run.add_argument(
        "--compute-s-params", action="store_true",
        help="force S-parameter computation (overrides the config)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_plot = sub.add_parser("plot", help="plot a saved result HDF5 file")
    p_plot.add_argument("result", help="path to the result .h5 file")
    p_plot.add_argument(
        "--quantity", choices=("s_params", "time_series"), default="s_params",
        help="which quantity to plot (default: s_params)",
    )
    p_plot.add_argument(
        "--out", default=None,
        help="output figure path (default: <result>_<quantity>.png)",
    )
    p_plot.set_defaults(func=_cmd_plot)

    p_geo = sub.add_parser(
        "export-geometry", help="export simulation geometry as JSON"
    )
    p_geo.add_argument("config", help="path to the YAML config file")
    p_geo.add_argument(
        "--output", "-o", required=True, help="output JSON path"
    )
    p_geo.set_defaults(func=_cmd_export_geometry)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        FileNotFoundError, KeyError, ValueError, TypeError, NotImplementedError
    ) as exc:
        # Fail loud with a clean one-line error rather than a traceback dump.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
