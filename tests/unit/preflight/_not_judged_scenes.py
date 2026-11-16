"""Concrete P4 scenes, importable by the freezer on the unmodified base tree."""
from rfx import Box, Simulation


SCENES = ("vacuum_without_source", "coarse_dielectric", "thin_absorber_wire_port")


def build_scene(name):
    sim = Simulation(domain=(.012, .012, .012), dx=.001,
                     freq_max=10e9, boundary="cpml", cpml_layers=4)
    if name == "vacuum_without_source":
        return sim
    if name == "coarse_dielectric":
        sim.add_material("dielectric", eps_r=16.)
        sim.add(Box((.003, .003, .003), (.009, .009, .009)), material="dielectric")
        return sim
    if name == "thin_absorber_wire_port":
        sim.add_port((.004, .006, .006), "ez")
        sim.add_probe((.008, .006, .006), "ez")
        return sim
    raise ValueError(name)


def issue_rows(report):
    # All public issue fields, in emission order. No implementation helper from
    # _not_judged participates in either the baseline or the comparison.
    return [issue.to_dict() for issue in report]
