"""Task 2 extra: lambda_1 (M=1, T given) at a given resolution/alpha/nu, archive box.
Usage: python experiments/task2_lambda1_resolution.py <res> <alpha> <T> [nu]
"""
import sys, time, json, numpy as np; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
from sens.mhw_system import MHWSystem, get_spun_state
from sens.lyapunov import lyapunov_spectrum, exponents_from_logR, batched_se
from sens.hw_linear import gamma_max
res=int(sys.argv[1]); alpha=float(sys.argv[2]); T=float(sys.argv[3]); nu=float(sys.argv[4]) if len(sys.argv)>4 else 1e-3
S=MHWSystem(res=res,alpha=alpha,nu=nu); t=time.time(); u0,fl,en=get_spun_state(S,0,300.0)
print(f"{S.tag()}: spin-up {time.time()-t:.0f}s; flux last100 {fl[-100:].mean():.3f}+-{fl[-100:].std():.3f} E {en[-100:].mean():.2f}", flush=True)
# high-k fraction diagnostic
import jax.numpy as jnp
nh=jnp.abs(jnp.fft.fft2(u0[1]))**2; kmax=jnp.max(jnp.abs(S.gp['KX'])); hi=float(jnp.sum(jnp.where(jnp.sqrt(S.gp['KSQ'])>0.75*kmax, nh, 0.))/jnp.sum(nh))
t=time.time(); nR=int(T/(40*S.dt))
logR,_=lyapunov_spectrum(S,u0,alpha,M=1,renorm_every=40,n_renorm=nR,key=jax.random.PRNGKey(0))
lam,se=exponents_from_logR(logR,S.dt,40,discard=nR//10); seb=batched_se(logR,S.dt,40,discard=nR//10)
print(f"res {res} alpha {alpha} nu {nu}: lambda1={lam[0]:.4f} +- {se[0]:.4f} (batched {seb[0]:.4f}) T={T} hi-k frac={hi:.3e} ({time.time()-t:.0f}s)", flush=True)
json.dump(dict(res=res,alpha=alpha,nu=nu,T=T,lambda1=float(lam[0]),se=float(se[0]),se_batched=float(seb[0]),hi_k_frac=hi,
  flux_mean=float(fl[-100:].mean()),flux_std=float(fl[-100:].std()),energy=float(en[-100:].mean()),gamma_max=gamma_max(res,64.0,alpha,1.0,nu,6)[0]),
  open(f"results/sens/lambda1_{S.tag()}.json","w"),indent=1)
