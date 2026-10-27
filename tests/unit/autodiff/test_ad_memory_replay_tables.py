"""Physical anchor: realized forward pullback leaves, not estimator arithmetic."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation


def _scene(graded):
    profile = {"dz_profile": np.full(8, .002)} if graded else {}
    return Simulation(freq_max=15e9, domain=(.016,) * 3, dx=.002,
                      boundary={"x": "cpml", "y": "pec", "z": "pec"},
                      cpml_layers=2, **profile)


def _tables(pullback, n_steps, n_bins):
    return [leaf for leaf in jax.tree.leaves(pullback)
            if hasattr(leaf, "nbytes") and jnp.iscomplexobj(leaf)
            and leaf.shape == ((n_steps + 255) // 256, 256, n_bins)]


@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
@pytest.mark.parametrize("n_steps,n_bins", ((8, 2), (257, 3)))
@pytest.mark.parametrize("segmented", (False, True))
def test_waveguide_replay_bytes_match_pullback(graded, n_steps, n_bins, segmented):
    sim = _scene(graded)
    sim.add_waveguide_port(.004, direction="+x", freqs=jnp.linspace(10e9, 14e9, n_bins),
                          ref_offset=1, probe_offset=2, mode_profile="analytic", name="port")
    knobs = ({"checkpoint_every": (4, 2)} if graded else {"checkpoint_segments": n_steps}) if segmented else {}
    with jax.default_device(jax.devices("cpu")[0]):
        eps = jnp.ones(sim._build_realized_grid().shape, jnp.float32)

        def objective(e):
            port = sim.forward(n_steps=n_steps, eps_override=e, skip_preflight=True,
                               **knobs).waveguide_sparams["port"]
            return jnp.sum(jnp.abs(port.s11) ** 2 + jnp.abs(port.s21) ** 2)

        _, pullback = jax.vjp(objective, eps)
        tables = _tables(pullback, n_steps, n_bins)
        measured_bytes = sum(int(leaf.nbytes) for leaf in tables)
        estimate = sim.estimate_ad_memory(n_steps, **knobs)
        assert len(tables) == 4
        assert estimate.replay_table_bytes == measured_bytes
        assert dict(estimate.replay_table_bytes_by_kind) == {"waveguide_port": measured_bytes}
        base = estimate.ad_fixed_gb + estimate.ntff_dft_gb
        assert estimate.ad_full_gb == pytest.approx(
            base + (n_steps * estimate.tape_step_bytes + measured_bytes) / 1e9)
        if segmented:
            from rfx.api._ad_memory_model import segment_counts
            count, live = segment_counts(n_steps, 0, knobs.get("checkpoint_every"), knobs.get("checkpoint_segments"))
            assert estimate.ad_segmented_gb == pytest.approx(
                base + (count * estimate.carry_bytes + live * estimate.tape_step_bytes + measured_bytes) / 1e9)
        report = sim.explain_ad_memory(n_steps, **knobs)
        component = next(c for c in report.components if c.name == "waveguide_port_replay_tables")
        assert round(component.memory_gb * 1e9) == measured_bytes
        assert sum(c.memory_gb for c in report.components) == pytest.approx(report.selected_memory_gb)
    jax.clear_caches()


@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
def test_replay_uses_realized_shared_port_bins(graded):
    sim = _scene(graded)
    for name, bins in (("first", 2), ("second", 5)):
        sim.add_waveguide_port(.004, direction="+x", freqs=jnp.linspace(10e9, 14e9, bins),
                              ref_offset=1, probe_offset=2, mode_profile="analytic", name=name)
    realized_bins = []
    with jax.default_device(jax.devices("cpu")[0]):
        def objective(e):
            result = sim.forward(n_steps=8, eps_override=e, skip_preflight=True)
            realized_bins.extend(cfg.freqs.size for cfg in result.waveguide_ports.values())
            return sum(jnp.sum(jnp.abs(p.s11) ** 2 + jnp.abs(p.s21) ** 2)
                       for p in result.waveguide_sparams.values())

        _, pullback = jax.vjp(objective, jnp.ones(sim._build_realized_grid().shape, jnp.float32))
        tables = _tables(pullback, 8, 2)
        assert realized_bins == [2, 2]  # The solve shares the first explicit frequency grid.
        assert len(tables) == 8
        measured_bytes = sum(int(leaf.nbytes) for leaf in tables)
        assert sim.estimate_ad_memory(8).replay_table_bytes == measured_bytes
        for options in ({}, {"max_levels": 2}) if graded else ({},):
            plan = sim.plan_ad_memory(8, available_memory_gb=1e-6, **options)
            estimate = plan.selected_estimate
            assert estimate.replay_table_bytes == measured_bytes
            from rfx.api._ad_memory_model import segment_counts
            count, live = segment_counts(8, 0, estimate.checkpoint_every, estimate.checkpoint_segments)
            assert estimate.ad_segmented_gb == pytest.approx(
                estimate.ad_fixed_gb + estimate.ntff_dft_gb
                + (count * estimate.carry_bytes + live * estimate.tape_step_bytes + measured_bytes) / 1e9)
    jax.clear_caches()


@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
@pytest.mark.parametrize("kind", ("lumped", "wire", "dft_plane", "flux", "point", "tfsf"))
def test_other_forward_measurements_have_no_replay_tables(kind, graded, monkeypatch):
    from rfx.measurement import dft

    def unexpected_replay(*args, **kwargs):
        pytest.fail(f"{kind} unexpectedly reached stored-record transform")

    monkeypatch.setattr(dft, "transform", unexpected_replay)
    profile = {"dz_profile": np.full(8, .002)} if graded else {}
    sim = Simulation(freq_max=15e9, domain=(.016,) * 3, dx=.002,
                     boundary="cpml", cpml_layers=2, **profile)
    bins = jnp.array([10e9, 14e9])
    knobs = {}
    if kind in ("lumped", "wire"):
        sim.add_port((.008,) * 3, "ez", **({"extent": .004} if kind == "wire" else {}))
        knobs["port_s11_freqs"] = bins
    elif kind == "tfsf":
        sim.add_tfsf_source(f0=10e9, direction="+x", polarization="ez", margin=2)
    else:
        sim.add_source((.004,) * 3, "ez")
    sim.add_probe((.010,) * 3, "ez")
    if kind == "dft_plane":
        sim.add_dft_plane_probe(axis="x", coordinate=.008, freqs=bins, name="plane")
    elif kind == "flux":
        sim.add_flux_monitor(axis="x", coordinate=.008, freqs=bins, name="flux")

    with jax.default_device(jax.devices("cpu")[0]):
        def objective(e):
            result = sim.forward(n_steps=8, eps_override=e, skip_preflight=True, **knobs)
            if kind in ("lumped", "wire"):
                values = result.s_params
                assert values is not None
            elif kind == "dft_plane":
                values = result.dft_planes["plane"].accumulator
            elif kind == "flux":
                from rfx.probes.probes import flux_spectrum
                values = flux_spectrum(result.flux_monitors["flux"])
            else:
                assert result.s_params is None
                values = result.time_series
            return jnp.sum(jnp.abs(values) ** 2)

        _, pullback = jax.vjp(objective, jnp.ones(sim._build_realized_grid().shape, jnp.float32))
        assert _tables(pullback, 8, 2) == []
        estimate = sim.estimate_ad_memory(8)
        assert estimate.replay_table_bytes == 0
        assert estimate.replay_table_bytes_by_kind == ()
    jax.clear_caches()


@pytest.mark.parametrize("graded", (False, True), ids=("uniform", "graded"))
def test_coaxial_spectra_not_reachable_from_forward(graded):
    sim = _scene(graded)
    sim.add_coaxial_port((.008,) * 3)
    with pytest.raises(NotImplementedError, match="not wired into Simulation.forward"):
        sim.forward(n_steps=8, skip_preflight=True)
    with pytest.raises(ValueError, match="not wired into Simulation.forward"):
        sim.estimate_ad_memory(8)
