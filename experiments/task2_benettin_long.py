"""Task 2 extra: Benettin lambda_1 over a long window t in [300, 300+T] for several seeds (64^2).
Usage: python experiments/task2_benettin_long.py <alpha> <T> <seed,seed,...>
"""
import sys, time, json, numpy as np; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
from sens.mhw_system import MHWSystem, get_spun_state
from sens.lyapunov import lyapunov_spectrum, exponents_from_logR, batched_se
alpha=float(sys.argv[1]); T=float(sys.argv[2]); seeds=[int(s) for s in sys.argv[3].split(',')]
S=MHWSystem(res=64,alpha=alpha); out={}
for sd in seeds:
    u0,fl,en=get_spun_state(S,sd,300.0); t=time.time(); nR=int(T/(40*S.dt))
    logR,_=lyapunov_spectrum(S,u0,alpha,M=1,renorm_every=40,n_renorm=nR,key=jax.random.PRNGKey(sd))
    x=logR[:,0]/(40*S.dt); cum=np.cumsum(x)/np.arange(1,nR+1)
    win=[float(x[i*400:(i+1)*400].mean()) for i in range(nR//400)]   # per-100-t.u. windows
    lam,se=exponents_from_logR(logR,S.dt,40,discard=0); seb=batched_se(logR,S.dt,40,discard=0)
    out[sd]=dict(lambda1=float(lam[0]),se=float(se[0]),se_batched=float(seb[0]),per100=win,running=cum[::40].tolist())
    print(f"alpha {alpha} seed {sd}: lambda1 over t in [300,{300+T:g}] = {lam[0]:.4f} +- {seb[0]:.4f}; per-100-t.u. windows: {np.round(win,3).tolist()} ({time.time()-t:.0f}s)", flush=True)
json.dump(out, open(f"results/sens/benettin_long_a{alpha:g}_res64_T{T:g}.json","w"), indent=1)
