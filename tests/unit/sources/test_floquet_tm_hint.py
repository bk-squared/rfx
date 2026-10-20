"""Pin the Floquet TM refusal and its closed-box polarization guidance."""

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.tfsf_discrete_plane_wave import incidence


def test_floquet_tm_refusal_text():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02),
                     dx=0.001, boundary="cpml", cpml_layers=4)
    with pytest.raises(NotImplementedError) as exc:
        sim.add_floquet_port(0.01, polarization="tm")
    assert str(exc.value) == (
        "add_floquet_port(polarization='tm'): TM Floquet S-parameter extraction is not "
        "implemented — the extractor is hardwired to the TE (Ex,Hy) pair and TE wave "
        "impedance, so a TM drive yields wrong S-parameters. Use polarization='te'. "
        "(For an oblique TM, p-polarized, plane wave without S-parameters on a finite structure, use "
        "add_tfsf_source(closed_box=True, theta_deg=..., polarization='p'); "
        "the open plane wave is TE at oblique incidence.)"
    )


def test_closed_box_p_is_in_incidence_plane():
    info = incidence(theta_deg=30, polarization="p")
    k, e = info["k_hat"], info["e_hat"]
    np.testing.assert_allclose(e @ np.cross([1., 0., 0.], k),
                               0., atol=1e-12, rtol=0)
    assert np.linalg.norm(np.cross(e, k)) > 0.99


def test_closed_box_s_is_normal_to_incidence_plane():
    info = incidence(theta_deg=30, polarization="s")
    u_te = np.cross([1., 0., 0.], info["k_hat"])
    u_te /= np.linalg.norm(u_te)
    np.testing.assert_allclose(abs(info["e_hat"] @ u_te),
                               1., atol=1e-12, rtol=0)
