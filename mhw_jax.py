import time
import argparse
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. Backend Configuration
# -----------------------------------------------------------------------------
ENABLE_JAX = True

try:
    if not ENABLE_JAX:
        raise ImportError("JAX disabled by user config.")
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, value_and_grad, lax
    from jax import config

    # Enable float64 for precision
    config.update("jax_enable_x64", True)

    xp = jnp
    print(f"Running with JAX backend on {jax.devices()[0]}")

    def scan_wrapper(f, init, xs):
        return jax.lax.scan(f, init, xs)

except ImportError:
    import numpy as jnp
    xp = np
    print("Running with standard NUMPY backend (No JIT, No AD)")

    # Dummy decorators / Fallbacks
    def jit(fun, static_argnames=None): return fun
    def grad(fun): raise NotImplementedError("AD requires JAX")
    def value_and_grad(fun): raise NotImplementedError("AD requires JAX")
    class lax:
        @staticmethod
        def cond(pred, true_fun, false_fun, operand):
            return true_fun(operand) if pred else false_fun(operand)

    def scan_wrapper(f, init, xs):
        carrier = init
        outputs = []
        for x in xs:
            carrier, out = f(carrier, x)
            outputs.append(out)
        return carrier, xp.stack(outputs)

# Import h5py for output
try:
    import h5py
except ImportError:
    print("Warning: h5py not installed. Output will fail.")

# -----------------------------------------------------------------------------
# 2. Physics & Grid Setup
# -----------------------------------------------------------------------------
class MHWParams:
    def __init__(self, nx=256, ny=256, Lx=64.0, Ly=64.0,
                 alpha=0.1, kappa=1.0, diffw=1e-4, diffn=1e-4, diffop=4,
                 dt=0.005, modified=True, arakawa=True):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.alpha = alpha
        self.kappa = kappa
        self.diffw = diffw
        self.diffn = diffn
        self.diffop = diffop
        self.dt = dt
        self.modified = modified
        self.arakawa = arakawa

        self.dx = Lx / nx
        self.dy = Ly / ny

        # Spectral Grid
        kx = 2 * np.pi * np.fft.fftfreq(nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, self.dy)
        self.KX, self.KY = xp.meshgrid(kx, ky, indexing='ij')
        self.KSQ = self.KX**2 + self.KY**2

        # Inverse Laplacian mask
        self.inv_ksq = xp.where(self.KSQ != 0, 1.0 / self.KSQ, 0.0)

        # Hyperdiffusion Integrating Factors
        self.damp_w = xp.exp(-self.diffw * (self.KSQ**(self.diffop/2)) * dt)
        self.damp_n = xp.exp(-self.diffn * (self.KSQ**(self.diffop/2)) * dt)

    def print_config(self):
        print("\n" + "="*40)
        print("       MHW MODEL CONFIGURATION")
        print("="*40)
        print(f"Domain:      {self.Lx} x {self.Ly}")
        print(f"Resolution:  {self.nx} x {self.ny} (dx={self.dx:.4f}, dy={self.dy:.4f})")
        print(f"Time Step:   dt = {self.dt}")
        print("-" * 40)
        print(f"Physics:     Alpha = {self.alpha}")
        print(f"             Kappa = {self.kappa}")
        print("-" * 40)
        print(f"Dissipation: order = {self.diffop}")
        print(f"             nu_w  = {self.diffw}")
        print(f"             nu_n  = {self.diffn}")
        print("-" * 40)
        print(f"Flags:       Modified HW = {self.modified}")
        print(f"             Arakawa     = {self.arakawa}")
        print("="*40 + "\n")

def get_jax_params(p):
    """Convert class to a dictionary of JAX-friendly arrays/scalars."""
    return {
        'dx': p.dx, 'dy': p.dy, 'dt': p.dt,
        'inv_ksq': p.inv_ksq,
        'KX': p.KX, 'KY': p.KY, 'KSQ': p.KSQ,
        'damp_w': p.damp_w, 'damp_n': p.damp_n,
        'is_modified': 1.0 if p.modified else 0.0,
        'is_arakawa': 1.0 if p.arakawa else 0.0
    }

# -----------------------------------------------------------------------------
# 3. Numerical Operators
# -----------------------------------------------------------------------------
@jit
def arakawa(f, g, dx, dy):
    """Arakawa Bracket using optimized slicing."""
    def pad_periodic(arr):
        return jnp.pad(arr, ((1,1), (1,1)), mode='wrap') if ENABLE_JAX else np.pad(arr, ((1,1), (1,1)), mode='wrap')

    fp = pad_periodic(f)
    gp = pad_periodic(g)

    # Access neighbors via slicing
    f1 = fp[2:, 1:-1]; f2 = fp[0:-2, 1:-1]
    f3 = fp[1:-1, 2:]; f4 = fp[1:-1, 0:-2]
    f5 = fp[2:, 2:];   f6 = fp[0:-2, 0:-2]
    f7 = fp[2:, 0:-2]; f8 = fp[0:-2, 2:]

    g1 = gp[2:, 1:-1]; g2 = gp[0:-2, 1:-1]
    g3 = gp[1:-1, 2:]; g4 = gp[1:-1, 0:-2]
    g5 = gp[2:, 2:];   g6 = gp[0:-2, 0:-2]
    g7 = gp[2:, 0:-2]; g8 = gp[0:-2, 2:]

    j1 = (f1 - f2) * (g3 - g4) - (f3 - f4) * (g1 - g2)
    j2 = f1 * (g5 - g7) - f2 * (g8 - g6) - f3 * (g5 - g8) + f4 * (g7 - g6)
    j3 = f5 * (g3 - g1) - f6 * (g4 - g2) - f7 * (g4 - g1) + f8 * (g3 - g2)

    return (j1 + j2 + j3) / (12.0 * dx * dy)

@jit
def poisson_simple(f, g, dx, dy):
    fx = (xp.roll(f, -1, axis=0) - xp.roll(f, 1, axis=0)) / (2*dx)
    fy = (xp.roll(f, -1, axis=1) - xp.roll(f, 1, axis=1)) / (2*dy)
    gx = (xp.roll(g, -1, axis=0) - xp.roll(g, 1, axis=0)) / (2*dx)
    gy = (xp.roll(g, -1, axis=1) - xp.roll(g, 1, axis=1)) / (2*dy)
    return fx * gy - fy * gx

@jit
def compute_bracket(phi, x, params):
    return lax.cond(params['is_arakawa'] > 0.5,
                    lambda _: arakawa(phi, x, params['dx'], params['dy']),
                    lambda _: poisson_simple(phi, x, params['dx'], params['dy']),
                    operand=None)

@jit
def rhs_hw(state_spec, grid_params, phys_params):
    w_hat, n_hat = state_spec
    alpha, kappa = phys_params

    phi_hat = -w_hat * grid_params['inv_ksq']

    w = xp.real(xp.fft.ifft2(w_hat))
    n = xp.real(xp.fft.ifft2(n_hat))
    phi = xp.real(xp.fft.ifft2(phi_hat))

    brack_phi_w = compute_bracket(phi, w, grid_params)
    brack_phi_n = compute_bracket(phi, n, grid_params)

    coupling = phi - n
    zonal_mean = xp.mean(coupling, axis=1, keepdims=True)
    coupling = coupling - zonal_mean * grid_params['is_modified']

    c_term = alpha * coupling
    phi_y = xp.real(xp.fft.ifft2(1j * grid_params['KY'] * phi_hat))
    k_term = -kappa * phi_y

    dw_dt = -brack_phi_w + c_term
    dn_dt = -brack_phi_n + c_term + k_term

    return xp.fft.fft2(dw_dt), xp.fft.fft2(dn_dt)

# -----------------------------------------------------------------------------
# 4. Steppers
# -----------------------------------------------------------------------------
@jit
def step_rk4(state_spec, params_packed):
    grid_params, phys_params = params_packed
    w_hat, n_hat = state_spec
    dt = grid_params['dt']

    k1_w, k1_n = rhs_hw((w_hat, n_hat), grid_params, phys_params)
    k2_w, k2_n = rhs_hw((w_hat + 0.5*dt*k1_w, n_hat + 0.5*dt*k1_n), grid_params, phys_params)
    k3_w, k3_n = rhs_hw((w_hat + 0.5*dt*k2_w, n_hat + 0.5*dt*k2_n), grid_params, phys_params)
    k4_w, k4_n = rhs_hw((w_hat + dt*k3_w, n_hat + dt*k3_n), grid_params, phys_params)

    w_new = w_hat + (dt/6.0) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
    n_new = n_hat + (dt/6.0) * (k1_n + 2*k2_n + 2*k3_n + k4_n)

    w_new = w_new * grid_params['damp_w']
    n_new = n_new * grid_params['damp_n']
    return (w_new, n_new)

@jit
def step_leapfrog(state_spec, params_packed):
    grid_params, phys_params = params_packed
    w_hat, n_hat = state_spec
    dt = grid_params['dt']

    dw, dn = rhs_hw((w_hat, n_hat), grid_params, phys_params)
    w_pred = w_hat + dt * dw
    n_pred = n_hat + dt * dn
    dw_c, dn_c = rhs_hw((w_pred, n_pred), grid_params, phys_params)

    w_new = w_hat + 0.5 * dt * (dw + dw_c)
    n_new = n_hat + 0.5 * dt * (dn + dn_c)

    w_new = w_new * grid_params['damp_w']
    n_new = n_new * grid_params['damp_n']
    return (w_new, n_new)

# -----------------------------------------------------------------------------
# 5. Production Run with HDF5 Output
# -----------------------------------------------------------------------------
def run_production_io(p_obj, w0, n0, nframes, steps_per_frame, solver_name="leapfrog", filename="mhw_out.h5"):
    """
    nframes: Total number of output frames
    steps_per_frame (nts): Number of time steps between frames
    """
    total_steps = nframes * steps_per_frame
    print(f"\n--- Starting Production Run (HDF5 Output) ---")
    print(f"Frames: {nframes}, Steps/Frame: {steps_per_frame}, Total Steps: {total_steps}")
    print(f"Solver: {solver_name}")

    # 1. Setup JIT functions
    grid_params = get_jax_params(p_obj)
    phys_params = (p_obj.alpha, p_obj.kappa)

    if solver_name.lower() == "rk4":
        stepper = step_rk4
    else:
        stepper = step_leapfrog

    # Compile the "Chunk" stepper
    @jit
    def step_chunk(carry, _):
        def body(c, __):
            params_packed = (grid_params, phys_params)
            return stepper(c, params_packed), None

        # Run 'steps_per_frame' steps
        final_state, _ = scan_wrapper(body, carry, xp.arange(steps_per_frame))

        # Diagnostics
        w_curr, n_curr = final_state
        energy = xp.sum(xp.abs(n_curr)**2) / (p_obj.nx * p_obj.ny)**2
        return final_state, energy

    # 2. Initialize State
    w_hat = xp.fft.fft2(w0)
    n_hat = xp.fft.fft2(n0)
    current_state = (w_hat, n_hat)

    # 3. Initialize HDF5 File
    # Total snapshots = nframes + 1 (including t=0)
    num_snaps = nframes + 1

    with h5py.File(filename, "w") as f:
        # Save Parameters
        grp_param = f.create_group("parameters")
        grp_param.attrs["nx"] = p_obj.nx
        grp_param.attrs["ny"] = p_obj.ny
        grp_param.attrs["dt"] = p_obj.dt
        grp_param.attrs["alpha"] = p_obj.alpha
        grp_param.attrs["kappa"] = p_obj.kappa
        grp_param.attrs["solver"] = solver_name

        # Create Datasets
        dset_w = f.create_dataset("vorticity", (num_snaps, p_obj.nx, p_obj.ny), dtype='f4')
        dset_n = f.create_dataset("density", (num_snaps, p_obj.nx, p_obj.ny), dtype='f4')
        dset_phi = f.create_dataset("potential", (num_snaps, p_obj.nx, p_obj.ny), dtype='f4')
        dset_t = f.create_dataset("time", (num_snaps,), dtype='f4')
        dset_e = f.create_dataset("energy", (num_snaps,), dtype='f4')

        # 4. Save Initial Condition (Frame 0)
        print(f"Writing Frame 0/{nframes}...", end="\r")
        w_real = np.array(w0)
        n_real = np.array(n0)

        if ENABLE_JAX:
            phi_hat = -w_hat * grid_params['inv_ksq']
            phi_real = np.array(xp.real(xp.fft.ifft2(phi_hat)))
        else:
            phi_hat = -w_hat * grid_params['inv_ksq']
            phi_real = np.real(np.fft.ifft2(phi_hat))

        dset_w[0] = w_real
        dset_n[0] = n_real
        dset_phi[0] = phi_real
        dset_t[0] = 0.0

        # 5. Time Loop
        t_start = time.time()

        for frame in range(1, num_snaps):
            # Run JAX chunk
            current_state, energy = step_chunk(current_state, None)

            if ENABLE_JAX:
                current_state[0].block_until_ready()

            # Retrieve data
            w_hat_curr, n_hat_curr = current_state

            w_out = np.array(xp.real(xp.fft.ifft2(w_hat_curr)))
            n_out = np.array(xp.real(xp.fft.ifft2(n_hat_curr)))

            phi_hat_curr = -w_hat_curr * grid_params['inv_ksq']
            phi_out = np.array(xp.real(xp.fft.ifft2(phi_hat_curr)))

            # Write
            dset_w[frame] = w_out
            dset_n[frame] = n_out
            dset_phi[frame] = phi_out
            dset_t[frame] = frame * steps_per_frame * p_obj.dt
            dset_e[frame] = float(energy)

            print(f"Writing Frame {frame}/{nframes} | Energy: {energy:.4e}", end="\r")

    t_total = time.time() - t_start
    print(f"\nFinished. Saved to {filename}")
    print(f"Total time: {t_total:.2f}s | Speed: {total_steps/t_total:.0f} steps/s")


# -----------------------------------------------------------------------------
# 6. Auto-Differentiation Test
# -----------------------------------------------------------------------------
def run_ad_optimization(p_obj, w0, n0, n_steps, solver="leapfrog"):
    print("\n--- Running Auto-Differentiation Test ---")
    print(f"Solver: {solver} | Target: Turbulent Flux")

    grid_params = get_jax_params(p_obj)
    w_hat_init = xp.fft.fft2(w0)
    n_hat_init = xp.fft.fft2(n0)

    if solver.lower() == "rk4":
        stepper = step_rk4
    else:
        stepper = step_leapfrog

    def target_function(opt_params):
        # opt_params is [alpha, kappa] (Tracer)
        alpha_in = opt_params[0]
        kappa_in = opt_params[1]
        phys_params = (alpha_in, kappa_in)

        init_state = (w_hat_init, n_hat_init)

        def body(carrier, _):
            params_packed = (grid_params, phys_params)
            next_state = stepper(carrier, params_packed)

            w_c, n_c = next_state
            # Metric: Kinetic Energy
            phi_hat = -w_c * grid_params['inv_ksq']
            phi_y_hat = 1j * grid_params['KY'] * phi_hat
            phi_y = xp.real(xp.fft.ifft2(phi_y_hat))
            n_real = xp.real(xp.fft.ifft2(n_c))
            # Turbulent Flux: Gamma = -kappa * n * dphi/dy
            flux = -kappa_in * n_real * phi_y
            mean_flux = xp.mean(flux)

            return next_state, mean_flux

        final_state, metrics = scan_wrapper(body, init_state, xp.arange(n_steps))

        return xp.sum(metrics)

    initial_params = xp.array([p_obj.alpha, p_obj.kappa])

    t0 = time.time()
    val, gradients = value_and_grad(target_function)(initial_params)
    t1 = time.time()

    print(f"Objective Value: {val:.4e}")
    print(f"Gradient w.r.t [alpha, kappa]: {gradients}")
    print(f"Time: {t1-t0:.4f}s")

# -----------------------------------------------------------------------------
# 7. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Updated Arguments
    parser.add_argument("--nframes", type=int, default=20, help="Number of output frames")
    parser.add_argument("--nts", type=int, default=100, help="Steps between frames (nfdump)")

    parser.add_argument("--solver", type=str, default="leapfrog", choices=["leapfrog", "rk4"],
                        help="Solver method (default: leapfrog)")

    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--res", type=int, default=128, help="Grid Resolution")
    parser.add_argument("--test_ad", action="store_true", help="Run AD Test")
    parser.add_argument("--out", type=str, default="mhw_out.h5", help="Output filename")
    args = parser.parse_args()

    # 1. Initialize Params
    params = MHWParams(nx=args.res, ny=args.res, dt=args.dt)

    params.print_config()

    # 2. Initial Conditions
    x = np.linspace(0, params.Lx, args.res)
    y = np.linspace(0, params.Ly, args.res)
    X, Y = np.meshgrid(x, y)

    #w0 = np.sin(2*np.pi*X/params.Lx) * np.cos(2*np.pi*Y/params.Ly) + 0.1*np.random.randn(args.res, args.res)
    # random noise
    w0 = 1.e-4 * (np.random.rand(args.res, args.res) - 0.5)
    n0 = w0.copy()

    w0_jax = xp.array(w0)
    n0_jax = xp.array(n0)

    # 3. Select Mode
    if args.test_ad:
        run_ad_optimization(params, w0_jax, n0_jax, n_steps=5000, solver=args.solver)
    else:
        run_production_io(params, w0_jax, n0_jax,
                          nframes=args.nframes,
                          steps_per_frame=args.nts,
                          solver_name=args.solver,
                          filename=args.out)