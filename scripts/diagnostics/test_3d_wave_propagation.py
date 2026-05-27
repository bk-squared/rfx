import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rfx.subgridding.sbp_sat_3d_production import (
    init_nonsplit_subgrid_3d,
    step_sbp_sat_nonsplit_3d,
)
from rfx.subgridding.sbp_sat_3d import _update_h_only, _update_e_only

jax.config.update("jax_enable_x64", True)

def main():
    print("=================================================================")
    print("🚀 Running 3D SBP-SAT Subgridding Physical Cross-Validation")
    print("=================================================================")

    steps = 200
    ratio = 3
    shape_c = (16, 16, 16)
    dx_c = 0.05
    dx_f = dx_c / ratio
    
    config, state = init_nonsplit_subgrid_3d(
        shape_c=shape_c,
        dx_c=dx_c,
        fine_region=(5, 11, 5, 11, 5, 11),
        ratio=ratio,
        courant=0.3,
        tau=0.25,
    )
    
    dt = config.dt
    print(f"Coarse shape: {shape_c}, dx: {dx_c}m")
    print(f"Fine region (coarse indices): 5 to 11 (fine size: {config.nx_f}x{config.ny_f}x{config.nz_f})")
    print(f"dx_f: {dx_f:.6f}m, dt: {dt:.6e}s")

    # Define Gaussian pulse source
    t = np.arange(steps) * dt
    t0 = 35.0 * dt
    spread = 8.0 * dt
    pulse = np.exp(-0.5 * ((t - t0) / spread) ** 2, dtype=np.float32)

    # Source positions
    src_c = (2, 8, 8)               # Injected in the coarse grid (before subgrid)
    src_f_ref = (src_c[0]*ratio + ratio//2, src_c[1]*ratio + ratio//2, src_c[2]*ratio + ratio//2)
    
    # -----------------------------------------------------------------
    # 1. Run Uniform Coarse Solver
    # -----------------------------------------------------------------
    print("\nRunning Uniform Coarse Solver...")
    ex_c = jnp.zeros(shape_c, dtype=jnp.float32)
    ey_c = jnp.zeros(shape_c, dtype=jnp.float32)
    ez_c = jnp.zeros(shape_c, dtype=jnp.float32)
    hx_c = jnp.zeros(shape_c, dtype=jnp.float32)
    hy_c = jnp.zeros(shape_c, dtype=jnp.float32)
    hz_c = jnp.zeros(shape_c, dtype=jnp.float32)
    
    probe_coarse = []
    
    for s in range(steps):
        ez_c = ez_c.at[src_c].set(ez_c[src_c] + pulse[s])
        hx_c, hy_c, hz_c = _update_h_only(ex_c, ey_c, ez_c, hx_c, hy_c, hz_c, dt, dx_c)
        ex_c, ey_c, ez_c = _update_e_only(ex_c, ey_c, ez_c, hx_c, hy_c, hz_c, dt, dx_c)
        # Probe at center of coarse grid (8, 8, 8)
        probe_coarse.append(float(ez_c[8, 8, 8]))
        
    probe_coarse = np.array(probe_coarse)

    # -----------------------------------------------------------------
    # 2. Run High-Resolution Uniform Fine Reference Solver
    # -----------------------------------------------------------------
    print("Running Uniform Fine Reference Solver...")
    shape_f_ref = (shape_c[0]*ratio, shape_c[1]*ratio, shape_c[2]*ratio)
    ex_f = jnp.zeros(shape_f_ref, dtype=jnp.float32)
    ey_f = jnp.zeros(shape_f_ref, dtype=jnp.float32)
    ez_f = jnp.zeros(shape_f_ref, dtype=jnp.float32)
    hx_f = jnp.zeros(shape_f_ref, dtype=jnp.float32)
    hy_f = jnp.zeros(shape_f_ref, dtype=jnp.float32)
    hz_f = jnp.zeros(shape_f_ref, dtype=jnp.float32)
    
    probe_ref = []
    
    # Scale point source amplitude by ratio^2 for 2D cross-section area equivalence
    pulse_f = pulse * (ratio ** 2)
    
    for s in range(steps):
        ez_f = ez_f.at[src_f_ref].set(ez_f[src_f_ref] + pulse_f[s])
        hx_f, hy_f, hz_f = _update_h_only(ex_f, ey_f, ez_f, hx_f, hy_f, hz_f, dt, dx_f)
        ex_f, ey_f, ez_f = _update_e_only(ex_f, ey_f, ez_f, hx_f, hy_f, hz_f, dt, dx_f)
        # Probe at center of fine reference grid (24, 24, 24)
        probe_ref.append(float(ez_f[24, 24, 24]))
        
    probe_ref = np.array(probe_ref)

    # Define coarse and reference parameters for analysis
    peak_ref = np.max(probe_ref)
    arrival_ref = np.argmax(probe_ref)
    peak_coarse = np.max(probe_coarse)
    arrival_coarse = np.argmax(probe_coarse)
    rmse_coarse = np.sqrt(np.mean((probe_coarse - probe_ref) ** 2))

    # -----------------------------------------------------------------
    # 3. Run Our 3D SBP-SAT Overlapping Subgrid Solver (Sweep tau)
    # -----------------------------------------------------------------
    for tau_val in [0.25, 0.5, 1.0, 2.0]:
        print(f"Running 3D SBP-SAT Subgrid Solver (tau = {tau_val})...")
        config_sweep, state_sweep = init_nonsplit_subgrid_3d(
            shape_c=shape_c,
            dx_c=dx_c,
            fine_region=(5, 11, 5, 11, 5, 11),
            ratio=ratio,
            courant=0.3,
            tau=tau_val,
        )
        
        probe_subgrid = []
        
        @jax.jit
        def step_jit(s):
            return step_sbp_sat_nonsplit_3d(s, config_sweep)
            
        for s in range(steps):
            # Excite source in coarse grid (before subgrid)
            state_sweep = state_sweep._replace(
                ez_c=state_sweep.ez_c.at[src_c].set(state_sweep.ez_c[src_c] + pulse[s])
            )
            state_sweep = step_jit(state_sweep)
            # Probe at center of fine subgrid region (9, 9, 9 inside fine shape (18, 18, 18))
            probe_subgrid.append(float(state_sweep.ez_f[9, 9, 9]))
            
        probe_subgrid = np.array(probe_subgrid)

        # -----------------------------------------------------------------
        # 4. Quantitative Analysis
        # -----------------------------------------------------------------
        peak_subgrid = np.max(probe_subgrid)
        arrival_subgrid = np.argmax(probe_subgrid)
        rmse_subgrid = np.sqrt(np.mean((probe_subgrid - probe_ref) ** 2))
        
        print(f"  [tau = {tau_val}] Peak Amp: {peak_subgrid:.6f}, Arrival: {arrival_subgrid}, RMSE: {rmse_subgrid:.6e}")
        
    print("\n========================= ANALYSIS RESULTS =========================")
    print(f"High-Res Ref Peak   : Amplitude = {peak_ref:.6f}, Arrival Step = {arrival_ref}")
    print(f"Uniform Coarse Peak : Amplitude = {peak_coarse:.6f}, Arrival Step = {arrival_coarse}")
    print(f"Uniform Coarse RMSE against Ref: {rmse_coarse:.6e}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
