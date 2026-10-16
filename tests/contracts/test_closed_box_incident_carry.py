"""A longer incident record must not enlarge the saved field-scan carry."""
from rfx.api._ad_carry import read_carry
from tests.contracts.test_ad_memory_tape_terms import _scene


def test_closed_box_carry_shapes_do_not_depend_on_step_bound():
    sim, grid, _, _ = _scene()
    shapes = []
    for n_steps in (8, 16):
        saved, _, _ = read_carry(
            sim, grid, n_steps, design_box=None, design="permittivity",
            shape=grid.shape, checkpoint=True, checkpoint_every=None,
            checkpoint_segments=None, n_warmup=0)
        assert saved
        shapes.append([(tuple(leaf["shape"]), leaf["dtype"]) for leaf in saved])
    assert shapes[0] == shapes[1], "closed-box incident line leaked into the field carry"


def test_only_a_fixed_step_run_builds_a_face_record(monkeypatch):
    """A fixed-step run reads a precomputed face record; until_decay keeps the line.

    A record over until_decay's whole step bound would cost (bound) x (line
    nodes), and the line grows with the bound too: an early-stopping oblique
    box at the default bound took 50 times longer (#1814).
    """
    import rfx.stepping.incident_line as line
    from rfx import Simulation
    calls = []
    record = line.record_face_values

    def counted(cfg, *args):
        calls.append(cfg.n_steps)
        return record(cfg, *args)

    monkeypatch.setattr(line, "record_face_values", counted)
    sim = Simulation(freq_max=15e9, domain=(.036, .024, .024), dx=1e-3,
                     boundary="cpml", cpml_layers=8)
    sim.add_tfsf_source(f0=8e9, margin=4, polarization="ez", direction="+x",
                        closed_box=True)
    sim.add_probe((.0215, .0135, .0105), "ez")
    sim.run(n_steps=8, skip_preflight=True)
    sim.run(until_decay=1e-3, decay_max_steps=100, skip_preflight=True)
    assert calls == [8]
