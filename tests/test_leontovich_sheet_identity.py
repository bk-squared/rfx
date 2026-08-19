"""Default-off identity and vmap parity for the Leontovich sheet (issue #669).

O6: ``surface_impedance_f0`` unset must be BYTE-IDENTICAL to the pre-#669
behaviour — SHA-256 over dtype + shape + raw bytes of the assembled
``eps_r`` / ``sigma`` / ``mu_r`` / ``pec_mask`` for a PEC-metal thin
conductor fixture AND a DC-lossy thin conductor fixture, built with the
kwarg absent vs default-passed (``None``). A digest-equality gate can be
green while measuring nothing (feedback: a gate can bind an artifact), so
the NEGATIVE CONTROL is part of the gate: setting ``surface_impedance_f0``
on the same fixture MUST change the sigma digest, MUST contribute zero
pec_mask bits for the sheet, and the sheet plane must be LIVE (nonzero
sigma there, array non-constant).

O7: one f0-mode case through the ``rfx.vmap_sweep`` batched material build
vs the serial per-value assembly — sigma slices exactly equal (the batched
path re-applies the same shared ``apply_thin_conductor``; #669 made ZERO
edits to ``rfx/vmap_sweep.py``).

Digest harness follows tests/test_run_progress_reporting.py (PR #668).
"""

import hashlib
import warnings

import numpy as np
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation


def _sha(*arrays) -> str:
    """SHA-256 over the raw bytes of *arrays*, dtype and shape included."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


_SHEET = (1e-3, 1e-3, 1e-3, 5e-3, 5e-3)  # x0 y0 z x1 y1 (zero-thickness in z)


def _fixture_sim(kind: str, **tc_kwargs):
    """Committed fixture: 6x6x3 mm box at dx = 1 mm with one thin sheet.

    kind='pec'  -> metal defaults (sigma_bulk = 5.8e7, PEC routing)
    kind='dc'   -> sub-threshold lossy DC fold (sigma_bulk = 1e4)
    """
    sim = Simulation(freq_max=10e9, domain=(6e-3, 6e-3, 3e-3), dx=1e-3)
    x0, y0, z, x1, y1 = _SHEET
    base = dict(sigma_bulk=5.8e7) if kind == "pec" else dict(sigma_bulk=1e4)
    base.update(tc_kwargs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim.add_thin_conductor(Box((x0, y0, z), (x1, y1, z)), **base)
    return sim, [str(w.message) for w in rec]


def _digests(sim):
    mats, _, _, pec_mask, *_ = sim._assemble_materials(sim._build_grid())
    if pec_mask is None:
        pec_mask = np.zeros((0,), dtype=np.bool_)
    return {
        "eps_r": _sha(mats.eps_r),
        "sigma": _sha(mats.sigma),
        "mu_r": _sha(mats.mu_r),
        "pec_mask": _sha(pec_mask),
    }, mats, pec_mask


def test_default_off_identity_and_negative_control_o6():
    for kind in ("pec", "dc"):
        sim_absent, warns_absent = _fixture_sim(kind)
        sim_default, warns_default = _fixture_sim(
            kind, surface_impedance_f0=None)
        d_absent, _, _ = _digests(sim_absent)
        d_default, _, _ = _digests(sim_default)
        assert d_absent == d_default, (
            f"{kind}: passing surface_impedance_f0=None changed the "
            f"assembled arrays — default is not byte-off")

        # default path emits no NEW warnings: identical warning sets with
        # the kwarg absent vs default-passed, and the f0-mode add-time
        # warning ("Leontovich surface-resistance sheet with Rs0 = ...")
        # never fires. (The #504 PEC warning legitimately MENTIONS the
        # escape hatch — that is contract-mandated wording, not a new
        # warning event.)
        assert warns_absent == warns_default
        for w in warns_absent:
            assert "Leontovich surface-resistance sheet" not in w, w

        # NEGATIVE CONTROL: the harness must be able to move. f0 set on the
        # same fixture changes the sigma digest and de-PECs the sheet.
        sim_on, warns_on = _fixture_sim(kind, surface_impedance_f0=10e9)
        d_on, mats_on, pec_on = _digests(sim_on)
        assert d_on["sigma"] != d_absent["sigma"], (
            f"{kind}: f0 mode did not change sigma — identity gate is blind")
        assert int(np.asarray(pec_on).sum()) == 0, (
            f"{kind}: f0-mode sheet still contributed pec_mask bits")
        assert any("Leontovich" in w for w in warns_on)

        # liveness: sigma nonzero and non-constant at the sheet plane
        sig = np.asarray(mats_on.sigma)
        grid = sim_on._build_grid()
        k = int(round(_SHEET[2] / grid.dx)) + grid.pad_z_lo
        plane = sig[:, :, k]
        assert plane.max() > 0.0, f"{kind}: sheet plane has no sigma"
        assert sig.max() != sig.min(), f"{kind}: sigma array is constant"


def test_vmap_parity_o7():
    """Batched (vmap_sweep) vs serial builds of an f0-mode sheet: sigma
    slices exactly equal — the batched path runs the same shared fold."""
    from rfx.vmap_sweep import _build_batched_materials

    def make(eps_val):
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                         boundary="cpml", cpml_layers=6, dx=0.002)
        sim.add_material("substrate", eps_r=eps_val)
        sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(
                Box((0.0, 0.008, 0.008), (0.02, 0.012, 0.008)),
                sigma_bulk=5.8e7, surface_impedance_f0=3e9)
        sim.add_source((0.01, 0.01, 0.01), "ez",
                       waveform=GaussianPulse(f0=3e9),
                       amplitude_kind="field")
        return sim

    eps_values = np.array([2.0, 6.0])
    sim = make(4.0)
    grid = sim._build_grid()
    base, *_ = sim._assemble_materials(grid)
    batched = _build_batched_materials(
        sim, grid, base, "substrate.eps_r", jnp.asarray(eps_values))

    assert batched.sigma.shape[0] == 2
    for idx, eps_val in enumerate(eps_values):
        serial, *_ = make(float(eps_val))._assemble_materials(grid)
        assert np.array_equal(np.asarray(batched.sigma[idx]),
                              np.asarray(serial.sigma)), (
            f"sigma mismatch at eps_r={eps_val}")
        # and the sheet is live in both
        assert float(np.asarray(serial.sigma).max()) > 0.0
