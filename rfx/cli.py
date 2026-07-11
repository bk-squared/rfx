"""Config-driven command-line interface for rfx.

Subcommands
-----------
``rfx run <config.yaml> [--output result.h5] [--num-periods N] [--compute-s-params]``
    Build a simulation from YAML, run it, and save the result dataset.
``rfx plot <result.h5> [--quantity s_params|time_series] [--out fig.png]``
    Plot S-parameters or probe time series from a saved result.
``rfx export-geometry <config.yaml> --output geo.json``
    Build the simulation and export its geometry as JSON.
``rfx experiment validate|run|submit|status|cancel ...``
    Operate strict, versioned AI-native experiments in isolated CPU workers.

Uses only the standard-library :mod:`argparse` (no extra CLI deps). Bad
config surfaces a clear, loud error and a non-zero exit code — never a
silent pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_experiment_document(path: str) -> dict:
    import json

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"experiment spec not found: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"experiment spec {source} must contain a JSON object")
    return value


def _experiment_record_json(service, record) -> str:
    import json

    return json.dumps(
        {
            "id": record.id,
            "state": record.state,
            "spec_sha256": record.spec_sha256,
            "compiled_sha256": record.compiled_sha256,
            "pid": record.pid,
            "cancel_requested": record.cancel_requested,
            "artifact_sha256": record.artifact_sha256,
            "artifact_path": record.artifact_path,
            "error": record.error,
            "events": [
                {
                    "sequence": event.sequence,
                    "type": event.event_type,
                    "state": event.state,
                    "payload": event.payload,
                    "created_at": event.created_at,
                }
                for event in service.repository.list_events(record.id)
            ],
        },
        sort_keys=True,
        indent=2,
    )


def _cmd_experiment_validate(args) -> int:
    import json

    from rfx.experiments import compile_experiment

    compiled = compile_experiment(_load_experiment_document(args.spec))
    preflight = compiled.preflight()
    print(
        json.dumps(
            {
                "spec_sha256": compiled.spec.sha256,
                "compiled_sha256": compiled.sha256,
                "preflight": preflight,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if preflight["ok"] else 1


def _cmd_experiment_run(args) -> int:
    from rfx.experiments import ExperimentService

    service = ExperimentService(args.workspace)
    record = service.run_sync(_load_experiment_document(args.spec))
    print(_experiment_record_json(service, record))
    return 0 if record.state == "succeeded" else 1


def _cmd_experiment_submit(args) -> int:
    from rfx.experiments import ExperimentService

    service = ExperimentService(args.workspace)
    record = service.submit(_load_experiment_document(args.spec))
    service.start(record.id)
    print(_experiment_record_json(service, service.get(record.id)))
    return 0


def _cmd_experiment_status(args) -> int:
    from rfx.experiments import ExperimentService

    service = ExperimentService(args.workspace)
    print(_experiment_record_json(service, service.get(args.run_id)))
    return 0


def _cmd_experiment_cancel(args) -> int:
    from rfx.experiments import ExperimentService

    service = ExperimentService(args.workspace)
    record = service.cancel(args.run_id)
    print(_experiment_record_json(service, record))
    return 0


def _cmd_experiment_bundle(args) -> int:
    from rfx.experiments import ExperimentService, export_replay_bundle

    service = ExperimentService(args.workspace)
    path = export_replay_bundle(service, args.run_id, destination=args.output)
    print(str(path))
    return 0


def _cmd_experiment_replay(args) -> int:
    import json

    from rfx.experiments import replay_bundle

    report = replay_bundle(args.bundle, args.workspace)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["passed"] else 1


def _cmd_experiment_compare(args) -> int:
    import json

    from rfx.experiments import ExperimentService, compare_sparameter_runs

    service = ExperimentService(args.workspace)
    report = compare_sparameter_runs(service, args.run_ids)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


def _cmd_studio(args) -> int:
    from rfx.studio.cli import main as studio_main

    forwarded = [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--workspace",
        args.workspace,
    ]
    if args.no_browser:
        forwarded.append("--no-browser")
    if args.auth_token_file:
        forwarded.extend(["--auth-token-file", args.auth_token_file])
    for origin in args.allowed_origin:
        forwarded.extend(["--allowed-origin", origin])
    if args.tls_terminated:
        forwarded.append("--tls-terminated")
    return studio_main(forwarded)


def _cmd_workspace_migrate(args) -> int:
    import json

    from rfx.operations import migrate_workspace

    print(json.dumps(migrate_workspace(args.workspace), sort_keys=True, indent=2))
    return 0


def _cmd_workspace_backup(args) -> int:
    from rfx.operations import backup_workspace

    print(str(backup_workspace(args.workspace, args.output)))
    return 0


def _cmd_workspace_restore(args) -> int:
    import json

    from rfx.operations import restore_workspace

    print(
        json.dumps(
            restore_workspace(args.backup, args.workspace), sort_keys=True, indent=2
        )
    )
    return 0


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
        "--output",
        "-o",
        default="result.h5",
        help="output HDF5 dataset path (default: result.h5)",
    )
    p_run.add_argument(
        "--num-periods",
        type=float,
        default=None,
        help="override the number of periods at freq_max for the run length",
    )
    p_run.add_argument(
        "--compute-s-params",
        action="store_true",
        help="force S-parameter computation (overrides the config)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_plot = sub.add_parser("plot", help="plot a saved result HDF5 file")
    p_plot.add_argument("result", help="path to the result .h5 file")
    p_plot.add_argument(
        "--quantity",
        choices=("s_params", "time_series"),
        default="s_params",
        help="which quantity to plot (default: s_params)",
    )
    p_plot.add_argument(
        "--out",
        default=None,
        help="output figure path (default: <result>_<quantity>.png)",
    )
    p_plot.set_defaults(func=_cmd_plot)

    p_geo = sub.add_parser("export-geometry", help="export simulation geometry as JSON")
    p_geo.add_argument("config", help="path to the YAML config file")
    p_geo.add_argument("--output", "-o", required=True, help="output JSON path")
    p_geo.set_defaults(func=_cmd_export_geometry)

    p_studio = sub.add_parser("studio", help="launch the local-first rfx Studio")
    p_studio.add_argument("--host", default="127.0.0.1")
    p_studio.add_argument("--port", type=int, default=8765)
    p_studio.add_argument("--workspace", default=".rfx-studio")
    p_studio.add_argument("--no-browser", action="store_true")
    p_studio.add_argument("--auth-token-file")
    p_studio.add_argument("--allowed-origin", action="append", default=[])
    p_studio.add_argument("--tls-terminated", action="store_true")
    p_studio.set_defaults(func=_cmd_studio)

    p_workspace = sub.add_parser(
        "workspace", help="migrate, backup, and restore a Studio workspace"
    )
    workspace_sub = p_workspace.add_subparsers(
        dest="workspace_command", required=True
    )
    p_workspace_migrate = workspace_sub.add_parser(
        "migrate", help="apply idempotent schema migrations"
    )
    p_workspace_migrate.add_argument("--workspace", default=".rfx-studio")
    p_workspace_migrate.set_defaults(func=_cmd_workspace_migrate)
    p_workspace_backup = workspace_sub.add_parser(
        "backup", help="create a checksummed workspace backup"
    )
    p_workspace_backup.add_argument("--workspace", default=".rfx-studio")
    p_workspace_backup.add_argument("--output", "-o", required=True)
    p_workspace_backup.set_defaults(func=_cmd_workspace_backup)
    p_workspace_restore = workspace_sub.add_parser(
        "restore", help="verify and atomically restore a workspace backup"
    )
    p_workspace_restore.add_argument("backup")
    p_workspace_restore.add_argument("--workspace", default=".rfx-studio")
    p_workspace_restore.set_defaults(func=_cmd_workspace_restore)

    p_experiment = sub.add_parser(
        "experiment",
        help="validate and run versioned AI-native experiments",
    )
    experiment_sub = p_experiment.add_subparsers(
        dest="experiment_command", required=True
    )

    p_experiment_validate = experiment_sub.add_parser(
        "validate", help="compile and preflight a versioned JSON spec"
    )
    p_experiment_validate.add_argument("spec", help="path to experiment JSON")
    p_experiment_validate.set_defaults(func=_cmd_experiment_validate)

    def add_workspace_argument(command_parser) -> None:
        command_parser.add_argument(
            "--workspace",
            default=".rfx-experiments",
            help="durable run/artifact workspace (default: .rfx-experiments)",
        )

    p_experiment_run = experiment_sub.add_parser(
        "run", help="run synchronously in an isolated CPU worker"
    )
    p_experiment_run.add_argument("spec", help="path to experiment JSON")
    add_workspace_argument(p_experiment_run)
    p_experiment_run.set_defaults(func=_cmd_experiment_run)

    p_experiment_submit = experiment_sub.add_parser(
        "submit", help="start an isolated CPU worker and return immediately"
    )
    p_experiment_submit.add_argument("spec", help="path to experiment JSON")
    add_workspace_argument(p_experiment_submit)
    p_experiment_submit.set_defaults(func=_cmd_experiment_submit)

    p_experiment_status = experiment_sub.add_parser(
        "status", help="show durable run state and events"
    )
    p_experiment_status.add_argument("run_id")
    add_workspace_argument(p_experiment_status)
    p_experiment_status.set_defaults(func=_cmd_experiment_status)

    p_experiment_cancel = experiment_sub.add_parser(
        "cancel", help="request cancellation and stop a matching worker"
    )
    p_experiment_cancel.add_argument("run_id")
    add_workspace_argument(p_experiment_cancel)
    p_experiment_cancel.set_defaults(func=_cmd_experiment_cancel)

    p_experiment_bundle = experiment_sub.add_parser(
        "bundle", help="export a checksummed replay bundle for a succeeded run"
    )
    p_experiment_bundle.add_argument("run_id")
    p_experiment_bundle.add_argument("--output", "-o", required=True)
    add_workspace_argument(p_experiment_bundle)
    p_experiment_bundle.set_defaults(func=_cmd_experiment_bundle)

    p_experiment_replay = experiment_sub.add_parser(
        "replay", help="verify and replay a bundle against declared tolerances"
    )
    p_experiment_replay.add_argument("bundle")
    add_workspace_argument(p_experiment_replay)
    p_experiment_replay.set_defaults(func=_cmd_experiment_replay)

    p_experiment_compare = experiment_sub.add_parser(
        "compare", help="compare cited S-parameter metrics across runs"
    )
    p_experiment_compare.add_argument("run_ids", nargs="+")
    add_workspace_argument(p_experiment_compare)
    p_experiment_compare.set_defaults(func=_cmd_experiment_compare)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        NotImplementedError,
    ) as exc:
        # Fail loud with a clean one-line error rather than a traceback dump.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
