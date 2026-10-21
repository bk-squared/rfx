"""Step A read ratchet: the union of models P and W on every execution lane.

Model P holds admitted static eps/sigma/mu and Debye/Lorentz/Drude blocks,
a probe, one excited lumped port, DFT/flux/NTFF observers,
series-RL element and lossy thin sheet. Refused kinds are omitted explicitly
by the disposition table. Model W has only one wire port and one probe. Both use the default S request.
Material blocks use separate names; neither model has a plain source.

Per-lane contents (P; W), measured on 8f992ab4e:
  run_uniform: P=_materials/eps, _materials/sigma, _materials/mu, _materials/debye, _materials/lorentz, _materials/drude, _ports/lumped_port, _probes/probe, _dft_planes/dft_plane, _flux_monitors/flux, _ntff/ntff_box, _lumped_rlc/series_RL, _thin_conductors/lossy_sheet; W=_ports/wire_port, _probes/probe
  run_nonuniform: P=_materials/eps, _materials/sigma, _materials/mu, _materials/debye, _materials/lorentz, _materials/drude, _ports/lumped_port, _probes/probe, _dft_planes/dft_plane, _flux_monitors/flux, _ntff/ntff_box, _lumped_rlc/series_RL, _thin_conductors/lossy_sheet; W=_ports/wire_port, _probes/probe
  run_subgridded: P=_materials/eps, _materials/sigma, _materials/mu, _ports/lumped_port, _probes/probe; W=_ports/wire_port, _probes/probe
    P omitted: _materials/debye: production refused (#1465); earlier envelope: production validation; research/off drop it, see _refinement 'relaxed_validation' (dispersive_or_nonlinear_material); _materials/lorentz: production refused (#1465); earlier envelope: production validation; research/off drop it, see _refinement 'relaxed_validation' (dispersive_or_nonlinear_material); _materials/drude: production refused (#1465); earlier envelope: production validation; research/off drop it, see _refinement 'relaxed_validation' (dispersive_or_nonlinear_material); _dft_planes/dft_plane: production refused (#1465); earlier envelope: production validation (dft_plane_unvalidated); _flux_monitors/flux: refused by lane admission; before it, result.flux_monitors came back None (#1313); _ntff/ntff_box: its x and y faces are always PEC (only a z lid may absorb); _lumped_rlc/series_RL: production refused (#1465); earlier envelope: production validation; research/off drop it, see _refinement 'relaxed_validation' (rlc_unvalidated); _thin_conductors/lossy_sheet: refused by lane admission; before it, it was dropped although production validation passes (#1311)
  run_adi: P=_probes/probe; W=_probes/probe
    P omitted: _materials/eps: ADI refuses a material interface (#1373); a homogeneous fill is carried (LANE_GATES); _materials/sigma: ADI refuses a material interface (#1373); a homogeneous fill is carried (LANE_GATES); _materials/mu: refused by lane admission; before it, the ADI kernel took ε and σ only (#1308); _materials/debye: ADI refuses dispersive materials; _materials/lorentz: ADI refuses dispersive materials; _materials/drude: ADI refuses dispersive materials; _ports/lumped_port: ADI refuses every port but add_source() soft sources; _dft_planes/dft_plane: ADI refuses DFT planes; _flux_monitors/flux: refused by lane admission; before it, result.flux_monitors came back None (#1313); _ntff/ntff_box: ADI refuses NTFF; _lumped_rlc/series_RL: ADI refuses lumped RLC; _thin_conductors/lossy_sheet: ADI refuses thin-conductor corrections
    W omitted: _ports/wire_port: ADI refuses every port but add_source() soft sources
  run_distributed: P=_materials/eps, _materials/sigma, _materials/mu, _materials/debye, _materials/lorentz, _materials/drude, _ports/lumped_port, _probes/probe, _dft_planes/dft_plane, _flux_monitors/flux, _ntff/ntff_box, _thin_conductors/lossy_sheet; W=_ports/wire_port, _probes/probe
    P omitted: _lumped_rlc/series_RL: lumped RLC refused (#1239)
  run_distributed_nu: P=_materials/eps, _materials/sigma, _materials/mu, _materials/debye, _materials/lorentz, _materials/drude, _ports/lumped_port, _probes/probe, _dft_planes/dft_plane, _flux_monitors/flux, _ntff/ntff_box, _thin_conductors/lossy_sheet; W=_ports/wire_port, _probes/probe
    P omitted: _lumped_rlc/series_RL: lumped RLC refused (#1239)
  fwd_distributed: P=_materials/eps, _materials/sigma, _materials/mu, _materials/debye, _materials/lorentz, _materials/drude, _ports/lumped_port, _probes/probe, _dft_planes/dft_plane, _flux_monitors/flux, _ntff/ntff_box, _thin_conductors/lossy_sheet; W=_ports/wire_port, _probes/probe
    P omitted: _lumped_rlc/series_RL: lumped RLC refused (#1239)
  fwd_uniform: P=_materials/eps, _materials/sigma, _materials/mu, _materials/debye, _materials/lorentz, _materials/drude, _ports/lumped_port, _probes/probe, _dft_planes/dft_plane, _flux_monitors/flux, _ntff/ntff_box, _lumped_rlc/series_RL, _thin_conductors/lossy_sheet; W=_ports/wire_port, _probes/probe
  fwd_nonuniform: P=_materials/eps, _materials/sigma, _materials/mu, _materials/debye, _materials/lorentz, _materials/drude, _ports/lumped_port, _probes/probe, _dft_planes/dft_plane, _flux_monitors/flux, _ntff/ntff_box, _lumped_rlc/series_RL, _thin_conductors/lossy_sheet; W=_ports/wire_port, _probes/probe
  fwd_distributed_nu: P=_materials/eps, _materials/sigma, _materials/mu, _materials/debye, _materials/lorentz, _materials/drude, _ports/lumped_port, _probes/probe, _dft_planes/dft_plane, _flux_monitors/flux, _ntff/ntff_box, _thin_conductors/lossy_sheet; W=_ports/wire_port, _probes/probe
    P omitted: _lumped_rlc/series_RL: lumped RLC refused
  fwd_adi: P=_probes/probe; W=_probes/probe
    P omitted: _materials/eps: ADI refuses a material interface (#1373); a homogeneous fill is carried (LANE_GATES); _materials/sigma: ADI refuses a material interface (#1373); a homogeneous fill is carried (LANE_GATES); _materials/mu: refused by lane admission; before it, the ADI kernel took ε and σ only (#1308); _materials/debye: ADI refuses dispersive materials; _materials/lorentz: ADI refuses dispersive materials; _materials/drude: ADI refuses dispersive materials; _ports/lumped_port: ADI refuses every port but add_source() soft sources; _dft_planes/dft_plane: ADI refuses DFT planes; _flux_monitors/flux: refused by lane admission; before it, forward() returned flux_monitors=None; _ntff/ntff_box: ADI refuses NTFF; _lumped_rlc/series_RL: ADI refuses lumped RLC; _thin_conductors/lossy_sheet: ADI refuses thin-conductor corrections
    W omitted: _ports/wire_port: ADI refuses every port but add_source() soft sources
"""

import collections
import io
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np
import pytest

from tests.contracts import path_disposition as T
from tests.contracts.declaration_reads import Instrument
from tests.unit.runners import test_path_disposition_cells as B

BASELINE = Path(__file__).with_name("fixtures") / "declaration_reads_baseline.json"
DESIGN = "rfx-archive rfx/plans/acceleration/20261021-declaration-stepA-SPEC.md §A1"
FEATURES = (
    ("_materials", "eps"),
    ("_materials", "sigma"),
    ("_materials", "mu"),
    ("_materials", "debye"),
    ("_materials", "lorentz"),
    ("_materials", "drude"),
    ("_ports", "lumped_port"),
    ("_ports", "wire_port"),
    ("_probes", "probe"),
    ("_dft_planes", "dft_plane"),
    ("_flux_monitors", "flux"),
    ("_ntff", "ntff_box"),
    ("_lumped_rlc", "series_RL"),
    ("_thin_conductors", "lossy_sheet"),
)
CONSUMPTION_PREFIXES = (
    "rfx.model.",
    "rfx.geometry.",
    "rfx.boundaries.",
    "rfx.materials.",
)
CONSUMPTION_MODULES = {
    "rfx.grid",
    "rfx.realized_geometry",
    "rfx.realized_placement",
    "rfx._periodic",
    "rfx._grid_metric",
    "rfx.runners._admission",
    "rfx.measurement.setup",
    "rfx.api",
}


def combined_model(lane, model):
    """P combines admitted declarations; W isolates the wire-port convention."""
    sim = B._base(
        lane,
        source=None,
        **(
            {"boundary": "cpml"}
            if model == "P" and lane not in B.ADI and lane != "run_subgridded"
            else {}
        ),
    )
    contents, omitted = [], {}
    features = (
        FEATURES if model == "P" else (("_ports", "wire_port"), ("_probes", "probe"))
    )
    for attr, feature in features:
        if model == "P" and feature == "wire_port":
            continue
        cell = T.cell(attr, feature, lane)
        if cell.kind in (T.REFUSES, T.NOT_REACHABLE) or cell.wrong:
            omitted[f"{attr}/{feature}"] = cell.note or cell.raises or cell.wrong
            continue
        other = B.FEATURES[attr, feature].build(lane, True)
        if attr == "_materials":
            name = f"block_{feature}"
            sim._materials[name] = other._materials["block"]
            from dataclasses import replace

            sim._geometry.extend(
                replace(g, material_name=name) for g in other._geometry
            )
        elif attr == "_ports":
            sim._ports.extend(p for p in other._ports if p.impedance > 0)
        elif attr == "_probes":
            pass  # The common base probe is already present.
        elif attr == "_ntff":
            sim._ntff = other._ntff
        else:
            getattr(sim, attr).extend(getattr(other, attr))
        contents.append(f"{attr}/{feature}")
    return sim, contents, omitted


def measure(lane):
    triples, models = {}, {}
    started = time.monotonic()
    for model in ("P", "W"):
        sim, contents, omitted = combined_model(lane, model)
        sink = io.StringIO()
        instrument = Instrument(sink, {"lane": lane, "model": model})
        instrument.install(sim)
        kwargs = dict(n_steps=40, skip_preflight=False)
        if "distributed" in lane:
            devices = jax.devices("cpu")
            if len(devices) < 2:
                pytest.skip("two host CPU devices required")
            kwargs["devices"] = devices[:2]
        with instrument.watch(), jax.default_device(jax.devices("cpu")[0]):
            if lane.startswith("run_"):
                result = sim.run(**kwargs)
            else:
                result = sim.forward(
                    **kwargs,
                    checkpoint=False,
                    **({"distributed": True} if "distributed" in lane else {}),
                )
            jax.block_until_ready(result.time_series)
        instrument.active = False
        models[model] = dict(contents=contents, omitted=omitted)
        for line in sink.getvalue().splitlines():
            row = json.loads(line)
            reader = row["reader"]["module"]
            if reader in CONSUMPTION_MODULES or reader.startswith(CONSUMPTION_PREFIXES):
                continue
            key = (lane, row["attr"], reader)
            phase = (
                "before-kernel-scan"
                if row["phase"] == "before-kernel-scan"
                else "after-kernel-scan"
            )
            triples.setdefault(key, set()).add(phase)
    rows = [
        dict(lane=k[0], attribute=k[1], reader=k[2], phases=sorted(v))
        for k, v in sorted(triples.items())
    ]
    return rows, dict(
        models=models,
        seconds=time.monotonic() - started,
        histogram=dict(sorted(collections.Counter(k[2] for k in triples).items())),
    )


def compare(rows, baseline):
    def keyed(values):
        return {(r["lane"], r["attribute"], r["reader"]): r["phases"] for r in values}

    actual, expected = keyed(rows), keyed(baseline)
    added, removed = actual.keys() - expected.keys(), expected.keys() - actual.keys()
    messages = [
        f"new declaration read: lane={l}, attribute={a}, reader={r}; {DESIGN}"
        for l, a, r in sorted(added)
    ]
    messages += [
        f"remove it from the baseline: lane={l}, attribute={a}, reader={r}"
        for l, a, r in sorted(removed)
    ]
    messages += [
        f"declaration read phase changed: {k}: {expected[k]} -> {actual[k]}; {DESIGN}"
        for k in actual.keys() & expected.keys()
        if actual[k] != expected[k]
    ]
    assert not messages, "\n".join(messages)


@pytest.mark.parametrize("lane", T.LANES)
def test_declaration_reads_ratchet(lane):
    rows, details = measure(lane)
    print(
        f"{lane}: {len(rows)} outside declaration triples; {details['seconds']:.3f}s",
        file=sys.stderr,
    )
    baseline = json.loads(BASELINE.read_text())["triples"]
    compare(rows, [r for r in baseline if r["lane"] == lane])
