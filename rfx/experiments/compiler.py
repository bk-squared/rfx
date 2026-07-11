"""Deterministic ExperimentSpec -> native rfx config compiler."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
from typing import Any, Mapping

from rfx.config import execution_to_run_kwargs, simulation_from_dict

from .spec import ExperimentSpec


COMPILED_SCHEMA_VERSION = "rfx-compiled-experiment/v1"


@dataclass(frozen=True)
class CompiledExperiment:
    spec: ExperimentSpec
    config: dict[str, Any]
    generated_python: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPILED_SCHEMA_VERSION,
            "spec_sha256": self.spec.sha256,
            "compiled_sha256": self.sha256,
            "config": self.config,
        }

    def build_simulation(self):
        return simulation_from_dict(self.config)

    def run_kwargs(self) -> dict[str, Any]:
        return execution_to_run_kwargs(self.config["execution"])

    def preflight(self) -> dict[str, Any]:
        # Legacy preflight checks also print human advisories. The experiment
        # boundary returns a pure machine artifact; callers can render issues
        # from the structured records without contaminating JSON stdout.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            report = self.build_simulation().preflight(strict=False)
        return report.to_dict()


def compile_experiment(document: ExperimentSpec | Mapping[str, Any]):
    # v2 is the canonical Studio/MCP contract. Keep the implemented v1 patch
    # proof available as a compatibility lane while callers migrate.
    from .canonical import (
        CANONICAL_SCHEMA_VERSION,
        CanonicalExperimentSpec,
        compile_canonical_experiment,
    )

    if isinstance(document, CanonicalExperimentSpec) or (
        isinstance(document, Mapping)
        and document.get("schema_version") == CANONICAL_SCHEMA_VERSION
    ):
        return compile_canonical_experiment(document)
    spec = (
        document
        if isinstance(document, ExperimentSpec)
        else ExperimentSpec.from_dict(document)
    )
    config = _compile_patch_antenna(spec)
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    generated = _generated_python(config, spec.sha256, digest)
    return CompiledExperiment(
        spec=spec, config=config, generated_python=generated, sha256=digest
    )


def _compile_patch_antenna(spec: ExperimentSpec) -> dict[str, Any]:
    model = spec.model
    domain_x, domain_y, _ = model.domain_m
    center_x = domain_x / 2.0
    center_y = domain_y / 2.0
    margin = (model.cpml_layers + 1) * model.cell_size_m
    z_ground_lo = model.stack_base_z_m if model.stack_base_z_m is not None else margin
    z_ground_hi = z_ground_lo + model.ground.thickness_m
    z_substrate_hi = z_ground_hi + model.substrate.thickness_m
    z_patch_hi = z_substrate_hi + model.patch.thickness_m

    def bounds(
        size: tuple[float, float], z_lo: float, z_hi: float
    ) -> list[list[float]]:
        return [
            [center_x - size[0] / 2.0, center_y - size[1] / 2.0, z_lo],
            [center_x + size[0] / 2.0, center_y + size[1] / 2.0, z_hi],
        ]

    patch_x_lo = center_x - model.patch.size_m[0] / 2.0
    port_clearance = min(model.substrate.thickness_m * 0.1, model.cell_size_m * 0.1)
    port_z = z_ground_hi + port_clearance
    port_extent = model.substrate.thickness_m - 2.0 * port_clearance

    sweep = model.frequency_sweep
    return {
        "frequency": {"freq_max": sweep.stop_hz},
        "domain": {
            "x": model.domain_m[0],
            "y": model.domain_m[1],
            "z": model.domain_m[2],
        },
        "boundary": "cpml",
        "cpml_layers": model.cpml_layers,
        "dx": model.cell_size_m,
        "mode": "3d",
        "precision": spec.execution.precision,
        "materials": {
            "experiment_substrate": {
                "eps_r": model.substrate.relative_permittivity,
                "sigma": model.substrate.conductivity_s_per_m,
            }
        },
        "geometry": [
            {
                "shape": "box",
                "bounds": bounds(model.ground.size_m, z_ground_lo, z_ground_hi),
                "material": "pec",
            },
            {
                "shape": "box",
                "bounds": bounds(model.substrate.size_m, z_ground_hi, z_substrate_hi),
                "material": "experiment_substrate",
            },
            {
                "shape": "box",
                "bounds": bounds(model.patch.size_m, z_substrate_hi, z_patch_hi),
                "material": "pec",
            },
        ],
        "sources": [
            {
                "type": "port",
                "position": [patch_x_lo + model.feed.inset_m, center_y, port_z],
                "component": "ez",
                "impedance": model.feed.impedance_ohm,
                "extent": port_extent,
                "waveform": {
                    "type": "gaussian_pulse",
                    "f0": model.design_frequency_hz,
                    "bandwidth": model.feed.pulse_bandwidth,
                    "amplitude": 1.0,
                },
            }
        ],
        "probes": [],
        "execution": {
            "n_steps": spec.execution.n_steps,
            "compute_s_params": True,
            "s_param_freq_start": sweep.start_hz,
            "s_param_freq_end": sweep.stop_hz,
            "s_param_n_freqs": sweep.points,
            "s_param_n_steps": spec.execution.s_param_n_steps,
        },
    }


def _generated_python(
    config: dict[str, Any], spec_digest: str, compiled_digest: str
) -> str:
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    encoded = json.dumps(canonical, ensure_ascii=True)
    return (
        "# Generated by rfx.experiments; do not edit.\n"
        f"# spec_sha256={spec_digest}\n"
        f"# compiled_sha256={compiled_digest}\n"
        "import json\n"
        "import os\n\n"
        "os.environ['JAX_PLATFORMS'] = 'cpu'\n"
        "os.environ['JAX_PLATFORM_NAME'] = 'cpu'\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
        "os.environ['ROCR_VISIBLE_DEVICES'] = ''\n\n"
        "from rfx.config import execution_to_run_kwargs, simulation_from_dict\n\n"
        f"CONFIG_JSON = {encoded}\n"
        "CONFIG = json.loads(CONFIG_JSON)\n\n"
        "def build_simulation():\n"
        "    return simulation_from_dict(CONFIG)\n\n"
        "def run():\n"
        "    simulation = build_simulation()\n"
        "    return simulation.run(**execution_to_run_kwargs(CONFIG['execution']))\n"
    )
