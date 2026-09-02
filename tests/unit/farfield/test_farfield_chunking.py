"""compute_far_field_jax must chunk its direction grid without changing the
answer (issue #727).

The transform materializes a (n_freqs, n_directions, n_cells) phase array
per face. On a board-scale box with a full sphere that reached 214 GB and
killed a run AFTER an 8-hour solve. The sum over surface cells is
independent per direction, so splitting theta is exact — this test is what
makes "exact" a checked claim rather than an argument, and it also pins the
memory bound as a real behaviour (a chunk actually happens when the budget
is small).
"""
from __future__ import annotations

import numpy as np

from rfx import Box, Simulation, compute_far_field_jax
from rfx.sources import GaussianPulse


def _run():
    sim = Simulation(freq_max=20e9, domain=(6e-3, 6e-3, 6e-3), dx=300e-6,
                     boundary="cpml", cpml_layers=6)
    sim.add(Box((2.7e-3, 2.7e-3, 2.7e-3), (3.3e-3, 3.3e-3, 3.3e-3)),
            material="pec")
    sim.add_source(position=(3e-3, 3e-3, 2.4e-3), component="ez",
                   amplitude_kind="current",
                   waveform=GaussianPulse(f0=12e9, bandwidth=8e9))
    sim.add_ntff_box((1.5e-3, 1.5e-3, 1.5e-3), (4.5e-3, 4.5e-3, 4.5e-3),
                     freqs=[10e9, 12e9, 14e9])
    return sim.run(n_steps=300)


def test_chunked_matches_single_shot_bitwise():
    res = _run()
    th = np.radians(np.linspace(0.0, 180.0, 37))
    ph = np.radians(np.linspace(0.0, 350.0, 12))

    whole = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid,
                                  th, ph, max_phase_bytes=float("inf"))
    # a budget small enough to force several passes
    chunked = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid,
                                    th, ph, max_phase_bytes=1e4)

    assert np.asarray(whole.E_theta).shape == np.asarray(chunked.E_theta).shape
    assert np.array_equal(np.asarray(whole.E_theta), np.asarray(chunked.E_theta)), (
        "chunking changed E_theta: max |delta| = "
        f"{np.max(np.abs(np.asarray(whole.E_theta) - np.asarray(chunked.E_theta))):.3e}")
    assert np.array_equal(np.asarray(whole.E_phi), np.asarray(chunked.E_phi))
    assert np.array_equal(np.asarray(whole.theta), np.asarray(chunked.theta))


def test_default_budget_leaves_small_problems_single_shot():
    """The bound must not chunk problems that fit — chunking is for the case
    that would otherwise die, not a change of default behaviour."""
    res = _run()
    th = np.radians(np.linspace(0.0, 180.0, 19))
    ph = np.radians(np.linspace(0.0, 350.0, 8))
    a = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid, th, ph)
    b = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid, th, ph,
                              max_phase_bytes=float("inf"))
    assert np.array_equal(np.asarray(a.E_theta), np.asarray(b.E_theta))


def test_default_budget_chunks_a_large_problem(monkeypatch):
    """The sizing DECISION must work, not just the split.

    Two tests have now been written for this and neither caught the real
    bug: the first forced a tiny max_phase_bytes (exercising the split,
    never the decision), the second compared chunked against unchunked
    output — which is equal whether or not a split happened. The actual
    defect was that the sizing scanned for NTFFData attributes starting
    with "J", a name no field has, so n_cells was always 0 and the
    DEFAULT path never chunked; a board pattern run then died on the
    214 GB allocation this function exists to prevent, after an 8.5-hour
    solve.

    So observe the decision directly: count how many times the function
    re-enters itself. One call means no split.
    """
    res = _run()
    th = np.radians(np.linspace(0.0, 180.0, 61))
    ph = np.radians(np.linspace(0.0, 350.0, 24))

    import rfx.farfield as _ff
    real = _ff.compute_far_field_jax
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(_ff, "compute_far_field_jax", counting)

    n_cells = max(int(getattr(res.ntff_data, f).shape[1])
                  * int(getattr(res.ntff_data, f).shape[2])
                  for f in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"))
    n_freqs = int(np.asarray(res.ntff_box.freqs).shape[0])
    per_theta = n_freqs * len(ph) * n_cells * 16.0
    assert per_theta > 0

    budget = per_theta * 5          # fits five directions, not sixty-one
    calls["n"] = 0
    chunked = counting(res.ntff_data, res.ntff_box, res.grid, th, ph,
                       max_phase_bytes=budget)
    n_passes = calls["n"] - 1       # the outer call plus one per chunk
    assert n_passes >= 2, (
        f"a {budget / 1e9:.2f} GB budget against {per_theta / 1e9:.3f} GB per "
        f"direction and {len(th)} directions must split, but the function "
        f"re-entered itself {n_passes} time(s) — the sizing did not fire")

    whole = real(res.ntff_data, res.ntff_box, res.grid, th, ph,
                 max_phase_bytes=float("inf"))
    assert np.array_equal(np.asarray(whole.E_theta),
                          np.asarray(chunked.E_theta))


def test_chunking_block_is_code_not_docstring():
    """The sizing block must be executable, not swallowed by a string literal.

    A docstring edit once moved the whole chunking block inside a second
    triple-quoted string. It parsed, every existing test still passed (they
    forced a tiny budget or compared chunked-vs-unchunked output, both of
    which are satisfied when nothing chunks), and a board pattern run died on
    the same 214 GB allocation the block exists to prevent — after an 8.5-hour
    solve. This pins the shape directly: the sizing symbols belong to the code
    object, not to __doc__.
    """
    import rfx.farfield as _ff

    doc = _ff.compute_far_field_jax.__doc__ or ""
    assert "max_phase_bytes" in doc, "the budget parameter must stay documented"
    for leaked in ("n_th_total", "jnp.concatenate", "for _name in"):
        assert leaked not in doc, (
            f"{leaked!r} appears in the docstring — the chunking block was "
            "captured by a string literal and is dead code"
        )

    names = _ff.compute_far_field_jax.__code__.co_names
    consts = _ff.compute_far_field_jax.__code__.co_consts
    assert "isfinite" in names, "the budget guard is not in the compiled body"
    flat = [x for c in consts for x in (c if isinstance(c, tuple) else (c,))]
    assert "x_lo" in flat, (
        "the face-name scan is not in the compiled body — sizing cannot run"
    )
