"""Diagnostic spin-up: flux, energy, zonal energies, high-k fraction vs time (used to find the bracket bug and to choose regimes).
Usage: python experiments/diag_spinup.py --res 64 --alpha 0.2 [--nu 1e-3 --dt 0.0025 --centered --unmod --T 400 --every 5]
"""
import sys, time, numpy as np, argparse
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp
from sens.mhw_system import MHWSystem
ap=argparse.ArgumentParser(); ap.add_argument('--res',type=int,default=64); ap.add_argument('--alpha',type=float,default=0.2)
ap.add_argument('--nu',type=float,default=1e-3); ap.add_argument('--diffop',type=int,default=6); ap.add_argument('--dt',type=float,default=0.0025)
ap.add_argument('--centered',action='store_true'); ap.add_argument('--unmod',action='store_true'); ap.add_argument('--T',type=float,default=150); ap.add_argument('--every',type=float,default=5)
ap.add_argument('--seed',type=int,default=0)
a=ap.parse_args()
S=MHWSystem(res=a.res, alpha=a.alpha, nu=a.nu, diffop=a.diffop, dt=a.dt, arakawa=not a.centered, modified=not a.unmod)
u=S.init_state(a.seed); nst=int(round(a.every/a.dt))
@jax.jit
def chunk(u):
    u = jax.lax.scan(lambda c,_: (S.step(c,a.alpha),None), u, None, length=nst)[0]
    ph = S.phi_hat(u[0]); phi = jnp.real(jnp.fft.ifft2(ph))
    zonal = jnp.mean(phi,axis=1); vz = jnp.gradient(zonal, S.L/a.res); Ez = 0.5*jnp.mean(vz**2)
    nz = jnp.mean(u[1],axis=1); Enz = 0.5*jnp.mean(nz**2)
    # spectrum: fraction of |n_hat|^2 in |k|>0.75 kmax
    nh=jnp.abs(jnp.fft.fft2(u[1]))**2; kmax=jnp.max(jnp.abs(S.gp['KX'])); hi=jnp.sum(jnp.where(jnp.sqrt(S.gp['KSQ'])>0.75*kmax, nh, 0.))/jnp.sum(nh)
    return u, (S.flux(u), S.energy(u), Ez, Enz, jnp.max(jnp.abs(u[0])), jnp.max(jnp.abs(u[1])), hi)
t0=time.time(); print(S.tag(), flush=True)
for k in range(int(a.T/a.every)):
    u, r = chunk(u); r=[float(x) for x in r]; t=(k+1)*a.every
    print(f"t={t:6.1f} flux={r[0]:.4e} E={r[1]:.4e} Ezon={r[2]:.3e} Enzon={r[3]:.3e} max|w|={r[4]:.3e} max|n|={r[5]:.3e} hi-k frac={r[6]:.3e} ({time.time()-t0:.0f}s)", flush=True)
    if not np.all(np.isfinite(r)): print("NONFINITE at t≈",t); break
