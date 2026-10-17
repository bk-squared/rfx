"""Compile-only saved-tape slopes against JAX memory_analysis (#1786)."""
import json
import sys
import time
import math
from collections import Counter

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.api._ad_memory_types import ADMemoryUnavailable
from rfx.farfield import compute_far_field
from rfx.geometry import Sphere


def _scene(graded=False, precision="float32"):
    profile = {"dz_profile": np.array([.0015] + [.0014, .0016] * 7 + [.0015])} if graded else {}
    sim = Simulation(freq_max=7.5e9, domain=(.024,) * 3, dx=.0015,
                     boundary="cpml", cpml_layers=4, precision=precision, **profile)
    if graded:
        sim.add_source((.003, .012, .012), "ez")
        sim.add_probe((.021, .012, .012), "ez")
    else:
        sim.add_tfsf_source(f0=5.25e9, bandwidth=1., polarization="ez",
                            direction="+x", margin=2, closed_box=True)
    sim.add(Sphere(center=(.012,) * 3, radius=.003), material="pec")
    if not graded:
        sim.add_ntff_box((.0005,) * 3, (.024,) * 3, freqs=np.array([5.25e9]))
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    assert max(grid.shape) <= 25
    box = ((.0075,) * 3, (.0165,) * 3)
    bounds = sim._design_box_bounds_from_corners(grid, box)
    shape = tuple(bounds[2 * a + 1] - bounds[2 * a] for a in range(3))
    return sim, grid, box, shape


def _measure(n, *, box, checkpoint, k=None, occupancy=False, graded=False, precision="float32"):
    sim, grid, corners, shape = _scene(graded, precision)
    knobs = {"checkpoint_every" if graded else "checkpoint_segments": k} if k else {}

    def objective(p):
        arr = jnp.ones(shape if box else grid.shape, jnp.float32)
        if occupancy:
            overrides = dict(design_box=corners, design_occupancy_override=arr * p[0])
        elif box:
            overrides = dict(design_box=corners, design_eps_override=arr * p[0],
                             design_sigma_override=arr * p[1] * 1e-3)
        else:
            overrides = dict(eps_override=arr * p[0], sigma_override=arr * p[1] * 1e-3)
        result = sim.forward(n_steps=n, checkpoint=checkpoint, skip_preflight=True,
                             **overrides, **knobs)
        if graded:
            return jnp.sum(result.time_series ** 2)
        ff = compute_far_field(result.ntff_data, result.ntff_box, grid,
                              jnp.array([jnp.pi / 2]), jnp.array([jnp.pi]))
        return jnp.sum(jnp.abs(ff.E_theta) ** 2)

    params = jnp.array([.4 if occupancy else 4. if box else 1., .2 if box else 0.], jnp.float32)
    with jax.default_device(jax.devices("cpu")[0]):
        executable = jax.jit(jax.value_and_grad(objective)).lower(params).compile()
        memory = executable.memory_analysis()
    compiled = (memory.argument_size_in_bytes + memory.output_size_in_bytes
                + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
    estimate = sim.estimate_ad_memory(n, checkpoint=checkpoint,
                                     design_box=corners if box else None,
                                     design="occupancy" if occupancy else "permittivity", **knobs)
    if k and not checkpoint:
        assert "rfx#1789" not in estimate.warning
        assert not estimate.tape_is_bound
        report = sim.explain_ad_memory(n, checkpoint=checkpoint,
                                      design_box=corners if box else None,
                                      design="occupancy" if occupancy else "permittivity", **knobs)
        components = {part.name: part for part in report.components}
        boundary = components["segmented_boundary_field_tape"]
        assert boundary.memory_gb * 1e9 == pytest.approx(
            estimate.ad_segmented_active_segments * estimate.carry_bytes)
        assert "full_reverse_field_tape" not in components
    estimated = (estimate.ad_segmented_gb if k else estimate.ad_full_gb) * 1e9
    row = dict(n=n, box=box, checkpoint=checkpoint, K=k, occupancy=occupancy,
               graded=graded, precision=precision, compiled_bytes=compiled, estimate_bytes=estimated)
    print("AD_TAPE_ROW " + json.dumps(row), file=sys.stderr, flush=True)
    return row


def _assert_slope(label, estimated, compiled):
    ratio = estimated / compiled
    print("AD_TAPE_B1 " + json.dumps(dict(row=label, est_slope=estimated,
                                        compiled_slope=compiled, ratio=ratio)),
          file=sys.stderr, flush=True)
    assert .999 <= ratio <= 1.6, f"B1 {label}: est slope={estimated}, compiled slope={compiled}, ratio={ratio}"


# Prior mixed box measurement, n=800/1600: estimated 17,496 B/step,
# compiled 12,320 B/step. Mixed whole-grid tape and c0 remain uncalibrated.
@pytest.mark.parametrize("precision", ("mixed", "float64"))
@pytest.mark.parametrize("box", (False, True))
def test_uncalibrated_precision_refused(box, precision):
    sim, _, corners, _ = _scene(precision=precision)
    options = dict(design_box=corners if box else None)
    with pytest.raises(ADMemoryUnavailable, match="fixed term and tape are calibrated on float32 only"):
        sim.estimate_ad_memory(800, **options)
    with pytest.raises(ADMemoryUnavailable, match="fixed term and tape are calibrated on float32 only"):
        sim.plan_ad_memory(800, available_memory_gb=1., **options)


def _scan_carries(jaxpr):
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "scan":
            start, count = eqn.params["num_consts"], eqn.params["num_carry"]
            yield eqn.params["length"], [v.aval for v in eqn.invars[start:start + count]]
        for value in eqn.params.values():
            if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                yield from _scan_carries(value)


def _carry_scene(feature, graded):
    from rfx import Box
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.materials.debye import DebyePole
    from rfx.materials.lorentz import lorentz_pole
    profile = {"dz_profile": np.array([.002] + [.0018, .0022] * 3 + [.002])} if graded else {}
    boundary = (BoundarySpec(x="cpml", y="periodic", z="cpml") if feature == "periodic"
                else BoundarySpec(x=Boundary(lo="pec", hi="cpml"), y="cpml", z="cpml")
                if feature == "pec-face" else BoundarySpec(x="cpml", y="pec", z="pec")
                if feature == "waveguide" else "cpml")
    sim = Simulation(freq_max=10e9, domain=(.016,) * 3, dx=.002,
                     boundary=boundary, cpml_layers=2, **profile)
    if feature in ("tfsf", "method-b"):
        sim.add_tfsf_source(f0=5e9, polarization="ez", direction="+x", margin=2,
                            angle_deg=20 if feature == "method-b" else 0,
                            method="methodB" if feature == "method-b" else "bloch")
    elif feature == "waveguide":
        sim.add_waveguide_port(direction="+x", x_position=.004,
                               y_range=(0., .016), z_range=(0., .016),
                               probe_offset=2, ref_offset=1, mode_profile="analytic")
    else:
        sim.add_source((.002, .008, .008), "ez", amplitude_kind="current")
    sim.add_probe((.014, .008, .008), "ez")
    if feature in ("debye", "lorentz"):
        poles = {"debye_poles": [DebyePole(.2, 1e-10)]} if feature == "debye" else {
            "lorentz_poles": [lorentz_pole(.2, 2*np.pi*1e9, 1e8)]}
        sim.add_material("pole", eps_r=2., **poles)
        sim.add(Box((.006,) * 3, (.010,) * 3), material="pole")
    if feature == "ports-rlc":
        sim.add_lumped_rlc((.008, .008, .008), R=500., C=1e-13)
        sim.add_port((.004, .004, .004), "ez", extent=.004)
        sim.add_port((.012, .012, .008), "ez")
    if feature in ("monitors", "ntff"):
        sim.add_ntff_box((.0005,) * 3, (.016,) * 3, freqs=np.array([5e9]))
    if feature in ("monitors", "dft-plane"):
        sim.add_dft_plane_probe(axis="x", coordinate=.008, freqs=np.array([5e9]))
    if feature == "flux":
        sim.add_flux_monitor(axis="x", coordinate=.008, freqs=np.array([5e9]))
    return sim


@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
@pytest.mark.parametrize("feature", ("core", "tfsf", "method-b", "periodic", "pec-face",
                                     "debye", "lorentz", "ports-rlc", "monitors", "ntff",
                                     "dft-plane", "flux", "waveguide"))
def test_carry_contract(feature, graded):
    sim = _carry_scene(feature, graded)
    grid = sim._build_realized_grid()
    inputs = (jax.ShapeDtypeStruct(grid.shape, jnp.float32),) * 2
    def observations(eps, sigma, n_steps=4):
        result = sim.forward(n_steps=n_steps, checkpoint=False, skip_preflight=True,
                             eps_override=eps, sigma_override=sigma,
                             checkpoint_every=None, checkpoint_segments=None, n_warmup=0)
        return result.time_series, result.ntff_data
    # Trace exactly the estimator's request, including abstract overrides.
    try:
        traced = jax.make_jaxpr(observations)(*inputs)
    except Exception as cause:
        with pytest.raises(ADMemoryUnavailable, match="carry could not be read") as error:
            sim.estimate_ad_memory(4, checkpoint=False)
        assert type(error.value.__cause__) is type(cause)
        assert str(cause).splitlines()[0] in str(error.value)
        print("AD_CARRY_REFUSAL " + json.dumps(dict(feature=feature, graded=graded,
              exception=type(cause).__name__, reason=str(cause))), file=sys.stderr, flush=True)
        return
    # The oracle imports no estimator helper.
    _, carry = max(_scan_carries(traced), key=lambda row: len(row[1]))
    estimate = sim.estimate_ad_memory(4, checkpoint=False)
    records = (*estimate.saved, *estimate.excluded_linear)
    oracle = Counter((tuple(a.shape), str(a.dtype)) for a in carry)
    model = Counter((tuple(a["shape"]), a["dtype"]) for a in records)
    oracle_bytes = sum(math.prod(a.shape) * a.dtype.itemsize for a in carry)
    model_bytes = sum(a["bytes"] for a in records)
    print("AD_CARRY_CONTRACT " + json.dumps(dict(feature=feature, graded=graded,
          modelled=estimate.modelled, oracle_bytes=oracle_bytes, model_bytes=model_bytes,
          missing=str(oracle-model), extra=str(model-oracle), saved=estimate.saved,
          excluded_linear=estimate.excluded_linear)), file=sys.stderr, flush=True)
    assert estimate.modelled, estimate.unmodelled_features
    assert model == oracle, f"carry {feature}, graded={graded}: missing={oracle-model}, extra={model-oracle}"
    assert model_bytes == oracle_bytes
    if feature == "flux" and graded:
        assert all(text in estimate.warning for text in
                   ("flux monitors", "graded", "AttributeError", "position_to_index"))
        assert not estimate.excluded_linear
        rows = []
        for n in (800, 1600):
            def objective(eps, sigma):
                return jnp.sum(observations(eps, sigma, n_steps=n)[0] ** 2)
            with jax.default_device(jax.devices("cpu")[0]):
                executable = jax.jit(jax.value_and_grad(objective, argnums=(0, 1))).lower(*inputs).compile()
                memory = executable.memory_analysis()
            compiled = (memory.argument_size_in_bytes + memory.output_size_in_bytes
                        + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
            estimate = sim.estimate_ad_memory(n, checkpoint=False)
            row = dict(n=n, checkpoint=False, grid=list(grid.shape),
                       estimate_bytes=estimate.ad_full_gb * 1e9, compiled_bytes=compiled,
                       argument=memory.argument_size_in_bytes, output=memory.output_size_in_bytes,
                       temp=memory.temp_size_in_bytes, alias=memory.alias_size_in_bytes,
                       warning=estimate.warning)
            rows.append(row)
            print("AD_GRADED_FLUX " + json.dumps(row), file=sys.stderr, flush=True)
        est_slope = (rows[1]["estimate_bytes"] - rows[0]["estimate_bytes"]) / 800
        compiled_slope = (rows[1]["compiled_bytes"] - rows[0]["compiled_bytes"]) / 800
        ratio = est_slope / compiled_slope
        print("AD_TAPE_B1 " + json.dumps(dict(row="graded flux", n1=800, n2=1600,
              est_slope=est_slope, compiled_slope=compiled_slope, ratio=ratio)), file=sys.stderr, flush=True)
        assert ratio >= .999, f"graded flux slope: est={est_slope}, compiled={compiled_slope}, ratio={ratio}"
        return
    # Independent feature initializers pin the exclusion, not just the
    # partition: dropping NTFF must fail even though all leaves stay saved.
    expected = []
    if sim._ntff is not None:
        from rfx.farfield import make_ntff_box, init_ntff_data
        lo, hi, freqs = sim._ntff
        expected.extend(jax.tree_util.tree_leaves(jax.eval_shape(
            lambda: init_ntff_data(make_ntff_box(grid, lo, hi, freqs), field_dtype=jnp.float32))))
    if sim._dft_planes:
        from rfx.measurement.setup import build_planes
        expected.extend(jax.tree_util.tree_leaves(jax.eval_shape(
            lambda: tuple(p.accumulator for p in build_planes(sim, grid, 4)))))
    if sim._flux_monitors:
        from rfx.measurement.flux_weights import build_flux_monitor_cfgs
        expected.extend(jax.tree_util.tree_leaves(jax.eval_shape(lambda: tuple(
            (m.e1_dft, m.e2_dft, m.h1_dft, m.h2_dft)
            for m in build_flux_monitor_cfgs(sim, grid, 4, periodic=sim._periodic_flags())))))
    excluded = Counter((tuple(r["shape"]), r["dtype"]) for r in estimate.excluded_linear)
    assert excluded == Counter((tuple(a.shape), str(a.dtype)) for a in expected), "declared linear accumulators must be excluded exactly (including NTFF)"


def test_ad_memory_tape_terms():
    started = time.monotonic()
    pairs = {}
    cases = []
    for box in (False, True):
        cases.extend(dict(box=box, checkpoint=checkpoint, k=k)
                     for checkpoint, k in ((False, None), (True, 8), (True, 16), (False, 8)))
    cases.extend((dict(box=True, checkpoint=False, occupancy=True),
                  dict(box=True, checkpoint=True), dict(box=False, checkpoint=True),
                  dict(box=True, checkpoint=True, k=40, graded=True),
                  dict(box=True, checkpoint=False, k=40, graded=True)))
    for case in cases:
        label = json.dumps(case, sort_keys=True)
        a, b = (_measure(n, **case) for n in (800, 1600))
        est_slope = (b["estimate_bytes"] - a["estimate_bytes"]) / 800
        compiled_slope = (b["compiled_bytes"] - a["compiled_bytes"]) / 800
        _assert_slope(label, est_slope, compiled_slope)
        if case.get("k") in (8, 16) and case["checkpoint"]:
            # Remove the measured live-segment growth from each executable
            # before comparing K intercepts: this isolates boundary storage.
            pairs[case["box"], case["k"]] = (
                a["estimate_bytes"] - 800 * est_slope,
                a["compiled_bytes"] - 800 * compiled_slope)
        jax.clear_caches()
    for box in (False, True):
        a, b = pairs[box, 8], pairs[box, 16]
        _assert_slope(f"boundary box={box}, K=8..16", (b[0] - a[0]) / 8, (b[1] - a[1]) / 8)
    print(f"AD_TAPE_RUNTIME {time.monotonic() - started:.6f}", file=sys.stderr, flush=True)


@pytest.mark.parametrize("segments,record_chosen", ((16, True), (1, False)),
                         ids=("record-chosen", "line-carried"))
def test_closed_box_record_memory_slope(segments, record_chosen):
    """The resident record counts only when it replaces more saved line bytes."""
    from rfx.api._ad_memory_model import face_record_pays
    from rfx.sources.tfsf import tfsf_from_declaration
    sim, grid, _, _ = _scene()
    rows = []
    for n in (800, 1600):
        cfg, _ = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=n)
        assert face_record_pays(cfg, n, checkpoint=True, checkpoint_segments=segments) == record_chosen
        rows.append(_measure(n, box=False, checkpoint=True, k=segments))
        jax.clear_caches()
    _assert_slope(f"closed box record chosen={record_chosen}",
                  (rows[1]["estimate_bytes"] - rows[0]["estimate_bytes"]) / 800,
                  (rows[1]["compiled_bytes"] - rows[0]["compiled_bytes"]) / 800)
