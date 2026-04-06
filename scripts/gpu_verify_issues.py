"""GPU verification for issues #13, #17 + gradient + non-uniform convergence."""

import numpy as np
import sys
import traceback


def test_optimize_memory():
    """#13: optimize() with n_steps=1000 must fit in 24GB."""
    print("=== #13: optimize() memory test ===")
    from rfx import Simulation, Box, GaussianPulse
    from rfx.optimize import optimize, DesignRegion

    sim = Simulation(freq_max=5e9, domain=(0.03, 0.02, 0.02), boundary="pec", dx=0.002)
    sim.add_port((0.005, 0.01, 0.01), "ez", impedance=50,
                 waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.025, 0.01, 0.01), "ez")

    region = DesignRegion(corner_lo=(0.01, 0.005, 0.005),
                          corner_hi=(0.02, 0.015, 0.015),
                          eps_range=(1.0, 4.4))

    def obj(result):
        import jax.numpy as jnp
        ts = result.time_series
        if ts.ndim == 2:
            ts = ts[:, 0]
        return jnp.sum(ts ** 2)

    try:
        opt = optimize(sim, region, obj, n_iters=3, lr=0.01, n_steps=1000, verbose=True)
        print(f"PASS: loss history = {opt.loss_history}")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_topology_gradient():
    """#13: topology conductor occupancy — loss must change over 5 iterations."""
    print("\n=== #13: topology conductor occupancy ===")
    from rfx import Simulation, Box, GaussianPulse

    sim = Simulation(freq_max=5e9, domain=(0.03, 0.02, 0.02), boundary="pec", dx=0.002)
    sim.add_port((0.005, 0.01, 0.01), "ez", impedance=50,
                 waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.025, 0.01, 0.01), "ez")

    try:
        from rfx.topology import topology_optimize, TopologyDesignRegion
        region = TopologyDesignRegion(
            corner_lo=(0.01, 0.005, 0.005),
            corner_hi=(0.02, 0.015, 0.015),
            material_bg="air", material_fg="copper",
        )

        def obj(result):
            import jax.numpy as jnp
            ts = result.time_series
            if ts.ndim == 2:
                ts = ts[:, 0]
            return -jnp.sum(ts ** 2)

        topo = topology_optimize(sim, region, obj, n_iterations=5, verbose=True)
        losses = topo.loss_history if hasattr(topo, "loss_history") else topo.history
        print(f"Loss history: {losses}")
        if len(losses) >= 2 and losses[-1] != losses[0]:
            print("PASS: loss changed (gradient is non-zero)")
            return True
        else:
            print(f"FAIL: loss flat ({losses[0]:.4e} -> {losses[-1]:.4e})")
            return False
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_ntff_power():
    """#17: NTFF Kahan summation — far-field power must be > 1e-10."""
    print("\n=== #17: NTFF far-field power ===")
    from rfx import Simulation, compute_far_field
    from rfx.sources.sources import ModulatedGaussian
    from rfx.grid import C0

    # NTFF box must be INSIDE the interior, NOT overlapping CPML.
    # CPML = 10 layers × 2mm = 20mm. NTFF margin must be > 20mm from edge.
    # Use larger domain so there's room for CPML + gap + NTFF + source.
    dom = 0.12  # 120mm domain
    cpml_n = 10
    dx_val = 0.002
    cpml_thick = cpml_n * dx_val  # 20mm

    sim = Simulation(freq_max=8e9, domain=(dom, dom, dom),
                     boundary="cpml", cpml_layers=cpml_n, dx=dx_val)
    # CW source: continuous sinusoidal at exactly 5 GHz.
    # DFT at 5 GHz will accumulate linearly with time → large values.
    from rfx.sources.sources import CWSource
    sim.add_source((dom * 0.4, dom * 0.5, dom * 0.5), "ez",
                   waveform=CWSource(f0=5e9, amplitude=1.0))
    sim.add_probe((dom * 0.4, dom * 0.5, dom * 0.5), "ez")

    # NTFF box: 5mm inside the CPML inner boundary
    ntff_margin = cpml_thick + 5e-3  # 25mm from domain edge
    sim.add_ntff_box(
        (ntff_margin, ntff_margin, ntff_margin),
        (dom - ntff_margin, dom - ntff_margin, dom - ntff_margin),
        np.array([5e9]),
    )

    grid = sim._build_grid()
    n_steps = int(np.ceil(15e-9 / grid.dt))
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

    result = sim.run(n_steps=n_steps)

    theta = np.linspace(0.1, np.pi - 0.1, 5)
    phi = np.array([0.0, np.pi / 2])

    try:
        ff = compute_far_field(result.ntff_data, result.ntff_box, grid, theta, phi)
        power = float(np.max(np.abs(ff.E_theta) ** 2 + np.abs(ff.E_phi) ** 2))
        print(f"Max far-field power: {power:.4e}")
        if power > 1e-10:
            print("PASS: far-field power is physically meaningful")
            return True
        else:
            print(f"FAIL: far-field power {power:.2e} is near zero")
            return False
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = {}
    results["optimize_memory"] = test_optimize_memory()
    results["topology_gradient"] = test_topology_gradient()
    results["ntff_power"] = test_ntff_power()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    total = sum(results.values())
    print(f"\n{total}/{len(results)} passed")

    if not all(results.values()):
        sys.exit(1)
