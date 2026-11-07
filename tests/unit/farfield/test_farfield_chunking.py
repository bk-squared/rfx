"""compute_far_field_jax must chunk its direction grid without changing the
answer (issue #727).

The transform materializes a (n_freqs, n_directions, n_cells) phase array
per face. On a board-scale box with a full sphere that reached 214 GB and
killed a run AFTER an 8-hour solve. The sum over surface cells is
independent per direction, so splitting theta changes no term of any
direction's sum. Split and unsplit calls compile the final projection onto
E_theta, E_phi differently (fused inside the chunk map, op by op without
it), so the two agree to rounding, not bitwise: each component's largest
difference is held within 1e-4 of the field peak (the larger of max |E_theta|
and max |E_phi| of the unsplit result), the scale the cross-trace bar uses
for a summed quantity. This file also pins the memory bound as a real
behaviour (a chunk actually happens when the budget is small).
"""
from __future__ import annotations

import numpy as np

from rfx import Box, Simulation, compute_far_field_jax
from rfx.sources import GaussianPulse


_REL_TOL = 1e-4


def _assert_fields_match(whole, chunked):
    """|whole - chunked| <= 1e-4 x field peak of whole, E_theta and E_phi."""
    w_th, c_th = np.asarray(whole.E_theta), np.asarray(chunked.E_theta)
    w_ph, c_ph = np.asarray(whole.E_phi), np.asarray(chunked.E_phi)
    assert w_th.shape == c_th.shape and w_ph.shape == c_ph.shape
    scale = max(float(np.max(np.abs(w_th))), float(np.max(np.abs(w_ph))))
    assert scale > 0.0
    for name, w, c in (("E_theta", w_th, c_th), ("E_phi", w_ph, c_ph)):
        ratio = float(np.max(np.abs(w - c))) / scale
        assert ratio <= _REL_TOL, (
            f"chunking changed {name}: max |delta| / field peak = "
            f"{ratio:.3e} > {_REL_TOL:.0e}")


def _run():
    sim = Simulation(freq_max=20e9, domain=(6e-3, 6e-3, 6e-3), dx=300e-6,
                     boundary="cpml", cpml_layers=6)
    # a solid scatterer: a VOLUME (2 cells per axis, node to node), left one.
    # #931 §1.2 gives it walls on BOTH bounding planes of every axis, so its
    # absolute scattered field moved; this file's gate is chunked == whole
    # to rounding (1e-4 of the field peak) on the same run, which does not
    # read that. Same cube as test_ntff_box_nu_pads.
    sim.add(Box((2.7e-3, 2.7e-3, 2.7e-3), (3.3e-3, 3.3e-3, 3.3e-3)),
            material="pec")
    sim.add_source(position=(3e-3, 3e-3, 2.4e-3), component="ez",
                   amplitude_kind="current",
                   waveform=GaussianPulse(f0=12e9, bandwidth=0.8))
    sim.add_ntff_box((1.5e-3, 1.5e-3, 1.5e-3), (4.5e-3, 4.5e-3, 4.5e-3),
                     freqs=[10e9, 12e9, 14e9])
    return sim.run(n_steps=300)


def test_chunked_matches_single_shot_to_rounding():
    """A budget that forces several passes gives the unsplit answer to
    rounding at the field peak; theta is passed through, so it stays bitwise."""
    res = _run()
    th = np.radians(np.linspace(0.0, 180.0, 37))
    ph = np.radians(np.linspace(0.0, 350.0, 12))

    whole = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid,
                                  th, ph, max_phase_bytes=float("inf"))
    # a budget small enough to force several passes
    chunked = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid,
                                    th, ph, max_phase_bytes=1e4)

    _assert_fields_match(whole, chunked)
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


def test_default_budget_chunks_a_large_problem():
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

    Observe the realized scan length and theta chunk width in the traced
    program: a mapped split no longer re-enters the public function. The
    split result then matches the unsplit one to rounding at the field peak
    (the projection is compiled differently on the two paths; see the module
    docstring).
    """
    res = _run()
    th = np.radians(np.linspace(0.0, 180.0, 61))
    ph = np.radians(np.linspace(0.0, 350.0, 24))

    import jax
    import rfx.measurement.farfield as _ff
    real = _ff.compute_far_field_jax

    n_cells = max(int(getattr(res.ntff_data, f).shape[1])
                  * int(getattr(res.ntff_data, f).shape[2])
                  for f in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"))
    n_freqs = int(np.asarray(res.ntff_box.freqs).shape[0])
    per_theta = n_freqs * len(ph) * n_cells * 16.0
    assert per_theta > 0

    budget = per_theta * 5          # fits five directions, not sixty-one
    expected_width = max(1, int(budget // per_theta))
    expected_length = (len(th) + expected_width - 1) // expected_width
    traced = jax.make_jaxpr(
        lambda theta: real(res.ntff_data, res.ntff_box, res.grid, theta, ph,
                           max_phase_bytes=budget)
    )(th)

    def scan_equations(program):
        if hasattr(program, "jaxpr"):
            yield from scan_equations(program.jaxpr)
        elif hasattr(program, "eqns"):
            for equation in program.eqns:
                if equation.primitive.name == "scan":
                    yield equation
                yield from scan_equations(equation.params)
        elif isinstance(program, dict):
            for value in program.values():
                yield from scan_equations(value)
        elif isinstance(program, (tuple, list)):
            for value in program:
                yield from scan_equations(value)

    def direction_chunk_maps(program):
        """The scans whose scanned operand is built from theta.

        The faces are also a scan (inside each chunk, or alone when the
        budget does not split); its scanned operands are the face records,
        not theta. The direction-chunk map is the scan whose scanned
        operand descends from the traced theta input.
        """
        top = program.jaxpr
        theta_var = top.invars[0]
        producer = {}
        for equation in top.eqns:
            for out in equation.outvars:
                producer[out] = equation

        def from_theta(var, seen):
            if var is theta_var:
                return True
            equation = producer.get(var)
            if equation is None or id(equation) in seen:
                return False
            seen.add(id(equation))
            return any(from_theta(v, seen) for v in equation.invars
                       if not hasattr(v, "val"))

        found = []
        for scan in scan_equations(program):
            first_x = scan.params["num_consts"] + scan.params["num_carry"]
            if any(from_theta(v, set()) for v in scan.invars[first_x:]
                   if not hasattr(v, "val")):
                found.append(scan)
        return found

    scans = direction_chunk_maps(traced)
    assert len(scans) == 1, "expected one direction-chunk map"
    scan = scans[0]
    assert scan.params["length"] == expected_length, (
        f"realized {scan.params['length']} chunks; expected {expected_length}")
    theta_operand = scan.invars[scan.params["num_consts"] + scan.params["num_carry"]]
    chunk_shape = theta_operand.aval.shape[1:]  # remove the scan iteration axis
    assert chunk_shape == (expected_width,), (
        f"realized theta chunk shape {chunk_shape}; expected {(expected_width,)}")

    # A budget that does not split theta enters no direction-chunk map.
    unsplit = jax.make_jaxpr(
        lambda theta: real(res.ntff_data, res.ntff_box, res.grid, theta, ph,
                           max_phase_bytes=float("inf"))
    )(th)
    assert direction_chunk_maps(unsplit) == [], (
        "an unsplit budget entered a direction-chunk map")
    chunked = real(res.ntff_data, res.ntff_box, res.grid, th, ph,
                   max_phase_bytes=budget)

    whole = real(res.ntff_data, res.ntff_box, res.grid, th, ph,
                 max_phase_bytes=float("inf"))
    _assert_fields_match(whole, chunked)


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
    import rfx.measurement.farfield as _ff

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
