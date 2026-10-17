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


def test_only_a_paying_forward_builds_a_face_record(monkeypatch):
    """A paying checkpointed forward builds one face record; until_decay keeps the line.

    Segments save one line per boundary whatever ``checkpoint`` says (#1815),
    so eight segments without checkpoint pay here too (8 x 1,392 B of lines
    against a 5,696 B record). A record over until_decay's whole step bound
    would cost (bound) x (line nodes), and the line grows with the bound too:
    an early-stopping oblique box at the default bound took 50 times longer
    (#1814).
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
    sim.forward(n_steps=8, checkpoint=True, skip_preflight=True)
    sim.forward(n_steps=8, checkpoint=False, checkpoint_segments=8, skip_preflight=True)
    sim.run(until_decay=1e-3, decay_max_steps=100, skip_preflight=True)
    assert calls == [8, 8]


def test_nonpaying_requests_build_no_face_record(monkeypatch):
    """Run, no checkpoint, and one boundary cannot repay this oblique record."""
    import rfx.stepping.incident_line as line
    from rfx import Simulation
    calls = []
    record = line.record_face_values

    def counted(cfg, *args):
        calls.append(cfg.n_steps)
        return record(cfg, *args)

    monkeypatch.setattr(line, "record_face_values", counted)
    sim = Simulation(freq_max=15e9, domain=(.025,) * 3, dx=1e-3,
                     boundary="cpml", cpml_layers=8)
    sim.add_tfsf_source(f0=8e9, margin=4, polarization="s", direction="+x",
                        closed_box=True, theta_deg=30., phi_deg=0.)
    sim.add_probe((.0155, .0135, .0125), "ez")
    sim.run(n_steps=128, skip_preflight=True)
    assert calls == [], "run must carry the line"
    sim.forward(n_steps=128, checkpoint=False, skip_preflight=True)
    assert calls == [], "without checkpoints no saved line repays the record"
    sim.forward(n_steps=128, checkpoint=True, checkpoint_segments=1, skip_preflight=True)
    assert calls == [], "this face record exceeds one saved line"


def test_face_record_overrun_reads_nan():
    import jax.numpy as jnp
    import numpy as np
    from rfx.sources.tfsf import tfsf_from_declaration
    from rfx.sources.tfsf_discrete_plane_wave import IncidentStep, table_operands
    from rfx.stepping.incident_line import prepare_tfsf_for_scan
    sim, grid, _, _ = _scene()
    pair = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=8, record_faces=True)
    cfg, _ = prepare_tfsf_for_scan(pair, grid.dx, grid.dt)
    for magnetic, tables in ((True, cfg.tables_h), (False, cfg.tables_e)):
        assert bool(jnp.all(jnp.isfinite(table_operands(cfg, IncidentStep(jnp.int32(7)), tables[0], 0, magnetic))))
        values = table_operands(cfg, IncidentStep(jnp.int32(8)), tables[0], 0, magnetic)
        assert np.isnan(np.asarray(values)).all()
