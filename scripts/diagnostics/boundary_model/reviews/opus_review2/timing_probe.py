import time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import wall_eigs2 as W
C0=W.C0
L=30e-3; nc=30; dx=L/nc; n=nc+1; dt=0.5*dx/C0
f=W.uniform_map(n,dx,dt,np.ones(n))
t=time.time(); J=np.asarray(jax.jit(jax.jacfwd(f))(jnp.zeros(2*n))); print("jac", time.time()-t, flush=True)
t=time.time(); lam=np.linalg.eigvals(J); print("eig", time.time()-t, flush=True)
