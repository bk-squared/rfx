"""The default plane-wave pulse leaves an empty TF/SF box."""

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec


def test_default_plane_wave_until_decay(record_property):
    """An empty TF/SF box holds no field once its pulse has left.

    A jump-free start leaves 1.8e-13 of the peak energy in the #1812 scene;
    the bare differentiated Gaussian instead excites a persistent line mode.
    """
    sim = Simulation(
        freq_max=15e9, domain=(0.036, 0.024, 0.003), dx=1e-3, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z="periodic"),
    )
    sim.add_tfsf_source(f0=8e9, bandwidth=0.5, margin=4,
                        polarization="ez", direction="+x")
    sim.add_probe((0.0185, 0.0125, 0.0015), component="ez")
    result = sim.run(until_decay=1e-3, decay_max_steps=3000)
    stop, = [d for d in result.diagnostics if d.code == "until_decay.stop"]
    values = dict(stop.values)
    record_property("decay_stop", values)
    print("T1 decay stop:", values)
    assert len(result.time_series) < 3000, values
    assert len(result.time_series) <= 1000, values
    assert values["criterion"] == "float32_floor", values
