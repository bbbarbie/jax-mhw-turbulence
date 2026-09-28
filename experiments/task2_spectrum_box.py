"""Task 2 extra: Lyapunov spectrum (M exponents, T) for a given box size L.
Usage: python experiments/task2_spectrum_box.py <res> <alpha> <nu> <L> <M> <T>
"""
import sys, time, json, numpy as np; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp
from sens.mhw_system import MHWSystem, get_spun_state
from sens.lyapunov import lyapunov_spectrum, exponents_from_logR, batched_se, kaplan_yorke
from sens.hw_linear import gamma_max
res=int(sys.argv[1]); alpha=float(sys.argv[2]); nu=float(sys.argv[3]); L=float(sys.argv[4]); M=int(sys.argv[5]); T=float(sys.argv[6])
S=MHWSystem(res=res,alpha=alpha,nu=nu,L=L); t=time.time(); u0,fl,en=get_spun_state(S,0,300.0)
nh=jnp.abs(jnp.fft.fft2(u0[1]))**2; kmax=jnp.max(jnp.abs(S.gp['KX'])); hi=float(jnp.sum(jnp.where(jnp.sqrt(S.gp['KSQ'])>0.75*kmax, nh, 0.))/jnp.sum(nh))
print(f"{S.tag()}: spin-up {time.time()-t:.0f}s; flux last100 {fl[-100:].mean():.4f}+-{fl[-100:].std():.4f} E {en[-100:].mean():.3f} hi-k {hi:.2e}; flux by 50s: {np.round([fl[i:i+50].mean() for i in range(0,300,50)],4).tolist()}", flush=True)
t=time.time(); nR=int(T/(40*S.dt))
logR,_=lyapunov_spectrum(S,u0,alpha,M=M,renorm_every=40,n_renorm=nR,key=jax.random.PRNGKey(0))
lam,se=exponents_from_logR(logR,S.dt,40,discard=nR//10); seb=batched_se(logR,S.dt,40,discard=nR//10)
Np=int((lam>0).sum()); g=gamma_max(res,L,alpha,1.0,nu,6)[0]
print(f"L={L} res={res} nu={nu} alpha={alpha} M={M} T={T}: N+={Np} lambda1={lam[0]:.4f}+-{seb[0]:.4f} lam_M={lam[-1]:.4f} sum={lam.sum():.3f} D_KY={kaplan_yorke(lam) if lam[-1]<0 else None} gamma_max={g:.4f} ({time.time()-t:.0f}s)", flush=True)
print("lam:",np.round(lam,4).tolist())
json.dump(dict(params=dict(res=res,alpha=alpha,nu=nu,L=L,diffop=6,dt=S.dt),M=M,T=T,lam=lam.tolist(),se=se.tolist(),se_batched=seb.tolist(),N_plus=Np,sum_M=float(lam.sum()),
  kaplan_yorke=(kaplan_yorke(lam) if lam[-1]<0 else None),gamma_max=g,flux_mean=float(fl[-100:].mean()),flux_std=float(fl[-100:].std()),energy=float(en[-100:].mean()),hi_k_frac=hi,wall_s=time.time()-t),
  open(f"results/sens/spectrum_{S.tag()}_M{M}_T{T:g}.json","w"), indent=1)
