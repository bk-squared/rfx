"""Build an rfx :class:`~rfx.api.Simulation` from a config dict / YAML file.

This is a thin declarative front-end over the public builder API. It calls
``Simulation(...)``, ``add_material``, ``add`` (geometry), ``add_port`` /
``add_source``, and ``add_probe`` in that order, converting host-side
values (lists -> tuples, freq ranges -> arrays) before they reach the
builder. No physics lives here.

MVP scope: uniform-grid 3D microstrip / patch. Everything outside that
(waveguide / coaxial / Floquet ports, nonuniform / subgridding, DFT planes,
non-box shapes) raises a clear :class:`NotImplementedError` at the relevant
key so the boundary is obvious.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rfx.api import Simulation

from ._shapes import shape_from_config
from ._waveforms import waveform_from_config

# Top-level config keys the loader understands. An unknown top-level key is an
# error (fail loud) rather than a silent typo (e.g. ``boundry``).
_KNOWN_TOP_KEYS = {
    "frequency",
    "domain",
    "boundary",
    "cpml_layers",
    "dx",
    "mode",
    "precision",
    "materials",
    "geometry",
    "sources",
    "probes",
    "execution",
}

# Source ``type`` values the loader supports today. ``port`` -> add_port
# (lumped, with resistive termination + excitation); ``source`` -> add_source
# (soft point source, no impedance load). Other physical port types are
# explicitly deferred.
_DEFERRED_SOURCE_TYPES = {
    "waveguide": "add_waveguide_port",
    "coaxial": "add_coaxial_port",
    "floquet": "add_floquet_port",
    "msl": "add_msl_port",
    "tfsf": "add_tfsf_source",
}


def _require(cfg: dict, key: str, ctx: str):
    if key not in cfg:
        raise KeyError(f"{ctx} is missing required key {key!r}")
    return cfg[key]


def _as_xyz(value, ctx: str) -> tuple[float, float, float]:
    """Coerce a 3-element list/tuple to a float ``(x, y, z)`` tuple."""
    try:
        seq = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ctx} must be a 3-number list, got {value!r}") from exc
    if len(seq) != 3:
        raise ValueError(f"{ctx} must have exactly 3 numbers, got {value!r}")
    return seq


def _build_simulation_kwargs(cfg: dict) -> dict:
    """Translate the top-level config block into ``Simulation(...)`` kwargs."""
    unknown = set(cfg) - _KNOWN_TOP_KEYS
    if unknown:
        raise KeyError(
            f"Unknown top-level config key(s) {sorted(unknown)}; "
            f"supported: {sorted(_KNOWN_TOP_KEYS)}"
        )

    frequency = _require(cfg, "frequency", "config")
    if not isinstance(frequency, dict) or "freq_max" not in frequency:
        raise KeyError("config 'frequency' must be a mapping with key 'freq_max'")
    _freq_unknown = set(frequency) - {"freq_max"}
    if _freq_unknown:
        raise KeyError(
            f"frequency: unknown key(s) {sorted(_freq_unknown)}; "
            f"only 'freq_max' is accepted here"
        )
    freq_max = float(frequency["freq_max"])

    domain_cfg = _require(cfg, "domain", "config")
    if isinstance(domain_cfg, dict):
        _dom_unknown = set(domain_cfg) - {"x", "y", "z"}
        if _dom_unknown:
            raise KeyError(
                f"domain: unknown key(s) {sorted(_dom_unknown)}; "
                f"allowed: 'x', 'y', 'z'"
            )
        for _axis in ("x", "y", "z"):
            if _axis not in domain_cfg:
                raise KeyError(
                    f"domain: missing required axis {_axis!r}; "
                    f"got keys {sorted(domain_cfg)}"
                )
        domain = _as_xyz(
            [domain_cfg["x"], domain_cfg["y"], domain_cfg["z"]],
            "config 'domain'",
        )
    else:
        domain = _as_xyz(domain_cfg, "config 'domain'")

    kwargs: dict = {"freq_max": freq_max, "domain": domain}

    if "boundary" in cfg:
        boundary = cfg["boundary"]
        if not isinstance(boundary, str):
            raise NotImplementedError(
                "config 'boundary' supports only the scalar strings "
                "'pec', 'cpml', or 'upml'; BoundarySpec / per-face boundaries "
                "must be built via the Python API."
            )
        kwargs["boundary"] = boundary
    if "cpml_layers" in cfg:
        kwargs["cpml_layers"] = int(cfg["cpml_layers"])
    if "dx" in cfg and cfg["dx"] is not None:
        kwargs["dx"] = float(cfg["dx"])
    if "mode" in cfg:
        kwargs["mode"] = str(cfg["mode"])
    if "precision" in cfg:
        kwargs["precision"] = str(cfg["precision"])

    return kwargs


def _add_materials(sim: Simulation, materials_cfg) -> None:
    if materials_cfg is None:
        return
    if not isinstance(materials_cfg, dict):
        raise TypeError(
            f"config 'materials' must be a mapping name->props, got "
            f"{type(materials_cfg).__name__}"
        )
    allowed = {"eps_r", "sigma", "mu_r", "chi3"}
    for name, props in materials_cfg.items():
        props = props or {}
        if not isinstance(props, dict):
            raise TypeError(
                f"material {name!r} props must be a mapping, got "
                f"{type(props).__name__}"
            )
        unknown = set(props) - allowed
        if unknown:
            raise KeyError(
                f"material {name!r} has unsupported key(s) {sorted(unknown)}; "
                f"allowed: {sorted(allowed)}"
            )
        sim.add_material(str(name), **{k: float(v) for k, v in props.items()})


def _add_geometry(sim: Simulation, geometry_cfg) -> None:
    if geometry_cfg is None:
        return
    if not isinstance(geometry_cfg, list):
        raise TypeError(
            f"config 'geometry' must be a list of shapes, got "
            f"{type(geometry_cfg).__name__}"
        )
    for i, entry in enumerate(geometry_cfg):
        ctx = f"geometry[{i}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{ctx} must be a mapping, got {type(entry).__name__}")
        material = _require(entry, "material", ctx)
        shape = shape_from_config(entry)
        sim.add(shape, material=str(material))


def _add_sources(sim: Simulation, sources_cfg) -> None:
    if sources_cfg is None:
        return
    if not isinstance(sources_cfg, list):
        raise TypeError(
            f"config 'sources' must be a list, got {type(sources_cfg).__name__}"
        )
    for i, entry in enumerate(sources_cfg):
        ctx = f"sources[{i}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{ctx} must be a mapping, got {type(entry).__name__}")
        stype = entry.get("type", "port")
        if stype in _DEFERRED_SOURCE_TYPES:
            raise NotImplementedError(
                f"{ctx}: source type {stype!r} is not supported by the config "
                f"CLI v1. Use the Python API "
                f"({_DEFERRED_SOURCE_TYPES[stype]}) for it."
            )
        if stype not in ("port", "source"):
            raise NotImplementedError(
                f"{ctx}: unknown source type {stype!r}. Supported: "
                f"'port' (lumped) and 'source' (soft point source)."
            )

        position = _as_xyz(_require(entry, "position", ctx), f"{ctx} 'position'")
        component = str(entry.get("component", "ez"))
        waveform_cfg = entry.get("waveform")
        waveform = waveform_from_config(waveform_cfg) if waveform_cfg else None

        if stype == "source":
            extra = set(entry) - {"type", "position", "component", "waveform"}
            if extra:
                raise KeyError(
                    f"{ctx}: soft source got unsupported key(s) {sorted(extra)}"
                )
            sim.add_source(position, component, waveform=waveform)
            continue

        # Lumped port.
        extra = set(entry) - {
            "type", "position", "component", "waveform",
            "impedance", "extent", "excite", "direction",
        }
        if extra:
            raise KeyError(
                f"{ctx}: lumped port got unsupported key(s) {sorted(extra)}"
            )
        port_kwargs: dict = {"impedance": float(entry.get("impedance", 50.0))}
        if "extent" in entry and entry["extent"] is not None:
            port_kwargs["extent"] = float(entry["extent"])
        if "excite" in entry:
            port_kwargs["excite"] = bool(entry["excite"])
        if "direction" in entry and entry["direction"] is not None:
            port_kwargs["direction"] = str(entry["direction"])
        if waveform is not None:
            port_kwargs["waveform"] = waveform
        sim.add_port(position, component, **port_kwargs)


def _add_probes(sim: Simulation, probes_cfg) -> None:
    if probes_cfg is None:
        return
    if not isinstance(probes_cfg, list):
        raise TypeError(
            f"config 'probes' must be a list, got {type(probes_cfg).__name__}"
        )
    for i, entry in enumerate(probes_cfg):
        ctx = f"probes[{i}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{ctx} must be a mapping, got {type(entry).__name__}")
        extra = set(entry) - {"position", "component"}
        if extra:
            raise KeyError(f"{ctx}: probe got unsupported key(s) {sorted(extra)}")
        position = _as_xyz(_require(entry, "position", ctx), f"{ctx} 'position'")
        component = str(entry.get("component", "ez"))
        sim.add_probe(position, component)


def execution_to_run_kwargs(execution_cfg) -> dict:
    """Translate the ``execution`` block into ``Simulation.run`` kwargs.

    The runner uses ``s_param_freqs`` (an explicit frequency array), not the
    ``s_param_freq_start/end/n_freqs`` triple from the config sketch, so the
    range is materialised into a numpy array host-side here.

    ``n_steps`` and ``num_periods`` are mutually exclusive override levers in
    :meth:`~rfx.api.Simulation.run`: when ``n_steps`` is given it wins and
    ``num_periods`` is ignored by the runner; when only ``num_periods`` is
    given the runner auto-computes the step count.  If the YAML provides
    both, both flow through and the runner's precedence (``n_steps`` first)
    governs.
    """
    if execution_cfg is None:
        return {}
    if not isinstance(execution_cfg, dict):
        raise TypeError(
            f"config 'execution' must be a mapping, got "
            f"{type(execution_cfg).__name__}"
        )
    allowed = {
        "n_steps", "num_periods", "compute_s_params",
        "s_param_freq_start", "s_param_freq_end", "s_param_n_freqs",
        "s_param_n_steps",
    }
    unknown = set(execution_cfg) - allowed
    if unknown:
        raise KeyError(
            f"config 'execution' has unsupported key(s) {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )

    run_kwargs: dict = {}
    if "n_steps" in execution_cfg and execution_cfg["n_steps"] is not None:
        run_kwargs["n_steps"] = int(execution_cfg["n_steps"])
    if "num_periods" in execution_cfg:
        run_kwargs["num_periods"] = float(execution_cfg["num_periods"])
    if "compute_s_params" in execution_cfg:
        run_kwargs["compute_s_params"] = bool(execution_cfg["compute_s_params"])
    if "s_param_n_steps" in execution_cfg and execution_cfg["s_param_n_steps"] is not None:
        run_kwargs["s_param_n_steps"] = int(execution_cfg["s_param_n_steps"])

    start = execution_cfg.get("s_param_freq_start")
    end = execution_cfg.get("s_param_freq_end")
    n_freqs = execution_cfg.get("s_param_n_freqs")
    has_any = any(v is not None for v in (start, end, n_freqs))
    if has_any:
        missing = [
            k for k, v in (
                ("s_param_freq_start", start),
                ("s_param_freq_end", end),
                ("s_param_n_freqs", n_freqs),
            ) if v is None
        ]
        if missing:
            raise KeyError(
                "execution S-parameter frequency sweep requires all of "
                "s_param_freq_start / s_param_freq_end / s_param_n_freqs; "
                f"missing {missing}"
            )
        run_kwargs["s_param_freqs"] = np.linspace(
            float(start), float(end), int(n_freqs)
        )
    return run_kwargs


def simulation_from_dict(cfg: dict) -> Simulation:
    """Build a :class:`~rfx.api.Simulation` from a config ``dict``.

    The ``execution`` block is ignored here (it only affects ``run``); use
    :func:`execution_to_run_kwargs` or :func:`~rfx.config.runner.run_and_save`
    to consume it.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be a mapping, got {type(cfg).__name__}")

    sim = Simulation(**_build_simulation_kwargs(cfg))
    _add_materials(sim, cfg.get("materials"))
    _add_geometry(sim, cfg.get("geometry"))
    _add_sources(sim, cfg.get("sources"))
    _add_probes(sim, cfg.get("probes"))
    return sim


def simulation_from_yaml(path) -> Simulation:
    """Load a YAML file and build a :class:`~rfx.api.Simulation` from it."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"config file {path} is empty")
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config file {path} must contain a top-level mapping, got "
            f"{type(cfg).__name__}"
        )
    return simulation_from_dict(cfg)
