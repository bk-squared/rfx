"""TE10 normal-H interface oracle: continuous ABCD, aligned asymmetric slab.

Two meshes (1 and .5 mm): the third requires minutes of CPU time. The
declared second-order window [1.7, 2.4] means an error ratio [2**1.7, 2**2.4].
The 2 mm guide height reduces work; TE10 is constant in that direction.
All 21 bins, both complex S entries, and both value witnesses are judged.
"""
import numpy as np
import pytest
from tests._x64_compat import enable_x64

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec


def _reference(freqs):
    # exp(+i omega t); beta=sqrt(k0**2*eps*mu-(pi/a)**2), Z~mu/beta.
    # M_i=[[cos(beta*l), i*Z*sin(beta*l)], [i*sin(beta*l)/Z, cos(beta*l)]].
    # S11=(A+B/Z0-C*Z0-D)/(A+B/Z0+C*Z0+D); S21=2/denominator.
    k0 = 2 * np.pi * freqs / 299792458.
    beta0 = np.sqrt(k0**2 - (np.pi / .024)**2)
    z0 = 1 / beta0
    matrix = np.broadcast_to(np.eye(2, dtype=complex), (len(freqs), 2, 2)).copy()
    for mu, length in ((1., .016), (4., .010), (1., .030)):
        beta = np.sqrt(k0**2 * mu - (np.pi / .024)**2)
        z = mu / beta
        layer = np.empty_like(matrix)
        layer[:, 0, 0] = layer[:, 1, 1] = np.cos(beta * length)
        layer[:, 0, 1] = 1j * z * np.sin(beta * length)
        layer[:, 1, 0] = 1j * np.sin(beta * length) / z
        matrix = matrix @ layer
    a, b, c, d = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 1, 0], matrix[:, 1, 1]
    denominator = a + b / z0 + c * z0 + d
    return np.array([(a + b / z0 - c * z0 - d) / denominator, 2 / denominator])


@pytest.fixture
def spectral_precision():
    with enable_x64():
        yield


def test_normal_h_mu_slab_has_second_order_complex_s(spectral_precision):
    freqs = np.linspace(8e9, 10e9, 21)
    expected = _reference(freqs)
    errors = []
    for h in (.001, .0005):
        sim = Simulation(
            freq_max=11e9, domain=(.024, .002, .096), dx=h,
            cpml_layers=round(.048 / h),
            boundary=BoundarySpec(
                x=Boundary('pec', 'pec'), y=Boundary('pec', 'pec'),
                z=Boundary('cpml', 'cpml', lo_thickness=round(.040 / h),
                           hi_thickness=round(.048 / h))))
        sim.add_material('slab', eps_r=1., mu_r=4.)
        sim.add(Box((0., 0., .036), (.024, .002, .046)), material='slab')
        for position, reference, direction in ((.016, .020, '+z'), (.080, .076, '-z')):
            sim.add_waveguide_port(
                position, direction=direction, mode=(1, 0), mode_type='TE',
                freqs=freqs, f0=9e9, bandwidth=.4, waveform='modulated_gaussian',
                ref_offset=round(.002 / h), probe_offset=round(.004 / h),
                reference_plane=reference)
        result = sim.compute_waveguide_s_matrix(num_periods=60, normalize='flux')
        assert all(w['status'] == 'pass' for w in result.settling_witness)
        measured = np.asarray(result.s_params)[[0, 1], 0, :]
        errors.append(float(np.max(np.abs(measured - expected))))
    ratio = errors[0] / errors[1]
    print(f"normal-H errors={errors}, ratio={ratio}, slope={np.log2(ratio)}")
    assert 2**1.7 <= ratio <= 2**2.4, (errors, ratio, np.log2(ratio))
