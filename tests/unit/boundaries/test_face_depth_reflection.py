"""An 8/16-cell declaration must produce the same two absorbers on both paths.

The 1-D TEM witness expresses x faces only: its transverse periodic (uniform)
or PMC/PEC (NU) termination cannot express y PEC plus a 12-cell z absorber.
It uses production Yee curls and CPML with default kappa and asymptotic R.
"""
import numpy as np

from tests._absorber_witness import plane_wave_reflection


def test_asymmetric_face_reflection_uniform_and_graded():
    freqs = np.array([2, 4, 6, 8, 10]) * 1e9
    returns = {}
    for nonuniform in (False, True):
        returns[nonuniform] = np.array([
            plane_wave_reflection(
                16, None, face, freqs, face_layers=(8, 16),
                nonuniform=nonuniform, near=30, far=120, reference_distance=260,
            ) for face in ("lo", "hi")
        ])
    uniform, graded = returns[False], returns[True]
    # Measured float32 cross-path floor: max |delta R| = 6.531e-7 over
    # 2--10 GHz. Near -100 dB this is up to 0.574 dB; compare amplitude
    # with a 1e-6 absolute floor, not a relative error on near-zero echoes.
    np.testing.assert_allclose(10 ** (graded / 20), 10 ** (uniform / 20),
                               rtol=0, atol=1e-6,
                               err_msg="per-face reflection differs between uniform and graded paths")
    uniform_margin = uniform[0] - uniform[1]
    graded_margin = graded[0] - graded[1]
    # Uniform measured advantage: 35.96--43.51 dB. Pin an independent
    # lower bound so two paths making the same depth mistake cannot pass.
    assert np.all(uniform_margin > 35), uniform_margin
    assert np.all(graded_margin >= uniform_margin - 1.0), (
        "16-layer face lost the uniform path's depth advantage (1 dB tolerance)",
        uniform_margin, graded_margin,
    )
