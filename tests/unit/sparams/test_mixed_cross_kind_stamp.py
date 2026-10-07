"""S between a wire port and a microstrip port carries one time stamp.

A wire port's V and I and a microstrip port's E and H planes are records of
different kinds. Until S2 M2 the wire port's were stamped n dt and the
planes' (n+1) dt, so S12 and S21 of ``compute_mixed_s_matrix`` carried a
phase error of -/+ omega dt (1.614e-13 s on this fixture: 4.1e-3 rad at
4 GHz; magnitudes equal to 1e-6). M2 corrected it (pre-declaration,
decision record "M2 after two reviews", item 1).

The check: every spectrum the S assembly receives, and the S it assembles, equal
what a NumPy float64 DFT of the run's own time records gives with E at
(n+1) dt and H at (n+1/2) dt -- the wire port's V/I record and the
microstrip port's modal V and loop I projected from the recorded planes.
Bar: 1e-4 of peak (a quantity summed over the record); the fixture's five
bins, 1 .. 4 GHz, all lie within 40 dB of the spectral peak.
"""
import numpy as np

from tests.locks.test_sparams_split_bit_identity import _mixed_result


def _host(record, freqs, dt, offset):
    n = np.arange(record.shape[0], dtype=np.float64)
    return (np.exp(-2j * np.pi * freqs[:, None] * (n + offset) * dt)
            @ np.asarray(record, dtype=np.complex128)) * dt


def _capture(monkeypatch):
    import rfx.sources.waveguide_port as waveguide_port
    import rfx.sparams.mixed as mixed
    seen = {"runs": []}
    settle, assemble = (waveguide_port.settling_db_from_named_records,
                        mixed._assemble_mixed_power_wave_s)

    def settling(channels, **kw):
        seen["runs"].append((dict(channels), float(kw["dt"]),
                             np.asarray(kw["freqs"], dtype=np.float64)))
        return settle(channels, **kw)

    def assembly(v_lw, i_lw, v0_msl, i_msl, *args, **kw):
        seen["spectra"] = tuple(np.array(a) for a in (v_lw, i_lw, v0_msl, i_msl))
        seen["v_ref"] = np.array(kw["v_ref_lw"])
        seen["rest"] = (args, kw)
        out = assemble(v_lw, i_lw, v0_msl, i_msl, *args, **kw)
        seen["S"] = np.array(out[0])
        return out

    monkeypatch.setattr(waveguide_port, "settling_db_from_named_records", settling)
    monkeypatch.setattr(mixed, "_assemble_mixed_power_wave_s", assembly)
    return seen, assemble


def test_cross_kind_s_is_the_float64_dft_of_both_records_at_the_physical_stamps(monkeypatch):
    seen, assemble = _capture(monkeypatch)
    _mixed_result()
    # The power-wave S as assembled from the spectra. (The returned S then has
    # its off-diagonal taken through the flux channel, which changes its
    # magnitude and its phase by up to 3.4e-3 rad on this fixture; that step
    # reads no time stamp -- main -> M2 the returned and the assembled S turn
    # by the same amount at every bin.)
    S = seen["S"]
    assert len(seen["runs"]) == 2  # one driven run per port
    host = [np.zeros_like(a, dtype=np.complex128) for a in seen["spectra"]]
    v_ref = np.zeros_like(seen["v_ref"], dtype=np.complex128)
    for run, (channels, dt, freqs) in enumerate(seen["runs"]):
        wire, msl = np.asarray(channels["wire0/V_I"]), np.asarray(channels["msl0/V_I"])
        host[0][run, 0] = _host(wire[:, 0], freqs, dt, 1.0)
        host[1][run, 0] = _host(wire[:, 1], freqs, dt, 0.5)
        v_ref[run, 0] = _host(wire[:, 3], freqs, dt, 1.0)
        host[2][run, 0] = _host(msl[:, 0], freqs, dt, 1.0)
        host[3][run, 0] = _host(msl[:, -1], freqs, dt, 0.5)
    for name, got, want in zip(("wire V", "wire I", "microstrip V", "microstrip I"),
                               seen["spectra"], host):
        peak = np.max(np.abs(want))
        worst = float(np.max(np.abs(got - want)) / peak)
        print(f"\n[{name}] |kernel - float64 host DFT| / peak = {worst:.3g}")
        assert worst <= 1e-4, (name, worst)
    args, kw = seen["rest"]
    reference, _ = assemble(*host, *args, **{**kw, "v_ref_lw": v_ref})
    reference = np.asarray(reference)
    peak = np.max(np.abs(reference))
    for i, j in ((0, 1), (1, 0)):
        worst = float(np.max(np.abs(S[i, j] - reference[i, j])) / peak)
        turn = float(np.max(np.abs(np.angle(S[i, j] / reference[i, j]))))
        print(f"\n[S{i + 1}{j + 1}] |S - S_host| / peak = {worst:.3g}; phase {turn:.3g} rad")
        assert worst <= 1e-4, ((i, j), worst)
