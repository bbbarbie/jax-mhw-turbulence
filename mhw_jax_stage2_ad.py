"""Stage 2 detached-spin-up HW/MHW JAX solver.

This later implementation preserves the baseline numerical model while adding
a workflow intended for future statistical-sensitivity experiments: forward
spin-up is run outside the differentiated window, the nonlinear state can be
saved or loaded as NPZ, the AD objective is the mean turbulent flux over the
averaging window, and compact CSV records can be written for postprocessing.
It also includes finite-value checks and separate spin-up and averaging
durations.

Use this file as the recommended starting point for future detached-state AD
experiments. The historical accumulated-objective sweep remains reproducible
with ``mhw_jax.py``.
"""

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
        dset_flux = f.create_dataset("flux", (num_snaps,), dtype='f4')

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

        phi_y0 = np.real(
            np.fft.ifft2(
                1j * np.array(grid_params['KY']) * np.array(phi_hat)
            )
        )

        flux0 = -p_obj.kappa * n_real * phi_y0

        dset_flux[0] = float(np.mean(flux0))

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

            phi_y_out = np.real(
                np.fft.ifft2(
                    1j * np.array(grid_params['KY']) * np.array(phi_hat_curr)
                )
            )

            flux_out = -p_obj.kappa * n_out * phi_y_out

            # Write
            dset_w[frame] = w_out
            dset_n[frame] = n_out
            dset_phi[frame] = phi_out
            dset_t[frame] = frame * steps_per_frame * p_obj.dt
            dset_e[frame] = float(energy)
            dset_flux[frame] = float(np.mean(flux_out))

            print(f"Writing Frame {frame}/{nframes} | Energy: {energy:.4e}", end="\r")

    t_total = time.time() - t_start
    print(f"\nFinished. Saved to {filename}")
    print(f"Total time: {t_total:.2f}s | Speed: {total_steps/t_total:.0f} steps/s")


# -----------------------------------------------------------------------------
# 6. Detached Nonlinear-State Auto-Differentiation
# -----------------------------------------------------------------------------
def select_stepper(solver):
    """Return the requested time-step function."""
    if solver.lower() == "rk4":
        return step_rk4
    return step_leapfrog


def time_to_steps(duration, dt, name):
    """Convert a physical duration to an integer number of time steps."""
    if duration < 0:
        raise ValueError(f"{name} must be non-negative")

    steps = int(round(duration / dt))
    reconstructed = steps * dt

    if not np.isclose(reconstructed, duration, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{name}={duration} is not an integer multiple of dt={dt}. "
            f"Nearest representable duration is {reconstructed} ({steps} steps)."
        )
    return steps


def state_is_finite(state):
    """Return True only when both spectral state arrays contain finite values."""
    w_hat, n_hat = state
    finite = xp.logical_and(
        xp.all(xp.isfinite(w_hat)),
        xp.all(xp.isfinite(n_hat)),
    )
    return bool(np.asarray(jax.device_get(finite)))


def save_spectral_state(filename, state, p_obj, elapsed_time):
    """Save a reusable nonlinear state without reducing it to float32."""
    w_hat, n_hat = jax.device_get(state)
    np.savez_compressed(
        filename,
        w_hat=np.asarray(w_hat),
        n_hat=np.asarray(n_hat),
        nx=np.int64(p_obj.nx),
        ny=np.int64(p_obj.ny),
        dt=np.float64(p_obj.dt),
        alpha=np.float64(p_obj.alpha),
        kappa=np.float64(p_obj.kappa),
        diffw=np.float64(p_obj.diffw),
        diffn=np.float64(p_obj.diffn),
        diffop=np.int64(p_obj.diffop),
        modified=np.int64(p_obj.modified),
        arakawa=np.int64(p_obj.arakawa),
        elapsed_time=np.float64(elapsed_time),
    )
    print(f"Saved detached spin-up state: {filename}")


def load_spectral_state(filename, p_obj):
    """Load a saved spectral state and verify that its configuration matches."""
    with np.load(filename) as data:
        checks = {
            "nx": (int(data["nx"]), p_obj.nx),
            "ny": (int(data["ny"]), p_obj.ny),
            "dt": (float(data["dt"]), p_obj.dt),
            "alpha": (float(data["alpha"]), p_obj.alpha),
            "kappa": (float(data["kappa"]), p_obj.kappa),
            "diffw": (float(data["diffw"]), p_obj.diffw),
            "diffn": (float(data["diffn"]), p_obj.diffn),
            "diffop": (int(data["diffop"]), p_obj.diffop),
            "modified": (bool(data["modified"]), p_obj.modified),
            "arakawa": (bool(data["arakawa"]), p_obj.arakawa),
        }

        mismatches = []
        for name, (saved, current) in checks.items():
            if isinstance(saved, float):
                same = np.isclose(saved, current, rtol=0.0, atol=1e-14)
            else:
                same = saved == current
            if not same:
                mismatches.append(f"{name}: saved={saved}, current={current}")

        if mismatches:
            raise ValueError(
                "Saved state configuration does not match this run:\n  "
                + "\n  ".join(mismatches)
            )

        state = (
            xp.asarray(data["w_hat"]),
            xp.asarray(data["n_hat"]),
        )
        elapsed_time = float(data["elapsed_time"])

    if not state_is_finite(state):
        raise FloatingPointError("Loaded state contains NaN or Inf.")

    print(f"Loaded detached spin-up state: {filename}")
    print(f"Loaded state time: {elapsed_time:.6f}")
    return state, elapsed_time


def run_forward_spinup(p_obj, w0, n0, spinup_steps, solver="leapfrog"):
    """
    Run the spin-up as an ordinary forward simulation.

    This function is never differentiated. It returns only the final state, so
    no spin-up trajectory is retained for reverse-mode AD.
    """
    if spinup_steps < 0:
        raise ValueError("spinup_steps must be non-negative")

    grid_params = get_jax_params(p_obj)
    phys_params = (p_obj.alpha, p_obj.kappa)
    stepper = select_stepper(solver)

    initial_state = (xp.fft.fft2(w0), xp.fft.fft2(n0))

    if spinup_steps == 0:
        return initial_state

    @jit
    def spinup_scan(state):
        def body(carrier, _):
            next_state = stepper(carrier, (grid_params, phys_params))
            return next_state, None

        final_state, _ = scan_wrapper(
            body,
            state,
            xp.arange(spinup_steps),
        )
        return final_state

    final_state = spinup_scan(initial_state)
    if ENABLE_JAX:
        final_state[0].block_until_ready()
    return final_state


def run_ad_window(p_obj, spun_state, avg_steps, solver="leapfrog"):
    """
    Differentiate only through the averaging window.

    The nonlinear starting state is explicitly detached. The objective is the
    time mean of the spatially averaged turbulent flux.
    """
    if avg_steps <= 0:
        raise ValueError("avg_steps must be positive")
    if not ENABLE_JAX:
        raise RuntimeError("The AD test requires JAX.")

    grid_params = get_jax_params(p_obj)
    stepper = select_stepper(solver)
    kappa_fixed = xp.asarray(p_obj.kappa)

    # Critical separation: the gradient cannot propagate through spin-up.
    fixed_state = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        spun_state,
    )

    def objective(alpha_in):
        phys_params = (alpha_in, kappa_fixed)

        def body(carrier, _):
            next_state = stepper(carrier, (grid_params, phys_params))
            w_hat, n_hat = next_state

            phi_hat = -w_hat * grid_params["inv_ksq"]
            phi_y = xp.real(
                xp.fft.ifft2(1j * grid_params["KY"] * phi_hat)
            )
            n_real = xp.real(xp.fft.ifft2(n_hat))

            instantaneous_mean_flux = xp.mean(
                -kappa_fixed * n_real * phi_y
            )
            return next_state, instantaneous_mean_flux

        _, flux_series = scan_wrapper(
            body,
            fixed_state,
            xp.arange(avg_steps),
        )

        # Long-time quantity is an average, not a sum.
        return xp.mean(flux_series)

    alpha0 = xp.asarray(p_obj.alpha)
    mean_flux, grad_alpha = value_and_grad(objective)(alpha0)

    if ENABLE_JAX:
        mean_flux.block_until_ready()

    return mean_flux, grad_alpha


def append_result_csv(filename, row):
    """Append one AD result to a CSV file."""
    import csv

    fieldnames = [
        "resolution",
        "dt",
        "alpha",
        "kappa",
        "diffw",
        "diffn",
        "diffop",
        "solver",
        "seed",
        "spinup_time",
        "avg_time",
        "avg_steps",
        "mean_flux",
        "grad_alpha",
        "finite",
        "spinup_runtime_s",
        "ad_runtime_s",
    ]

    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended result: {filename}")


def run_nonlinear_state_ad_test(
    p_obj,
    w0,
    n0,
    spinup_time,
    avg_time,
    solver="leapfrog",
    seed=0,
    state_in=None,
    state_out=None,
    result_csv=None,
):
    """
    Run forward-only spin-up, detach the final state, and differentiate only
    through the requested averaging window.
    """
    spinup_steps = time_to_steps(spinup_time, p_obj.dt, "spinup_time")
    avg_steps = time_to_steps(avg_time, p_obj.dt, "avg_time")

    if avg_steps <= 0:
        raise ValueError("avg_time must contain at least one time step.")

    print("\n--- Detached Nonlinear-State Direct AD ---")
    print(f"Solver: {solver}")
    print(f"Requested spin-up time: {spinup_time:.6f}")
    print(f"Forward-only spin-up steps: {spinup_steps}")
    print(f"Requested averaging time: {avg_time:.6f}")
    print(f"Differentiated averaging steps: {avg_steps}")
    print("Objective: time-mean turbulent flux")
    print("Gradient: d(mean flux)/d(alpha), with kappa fixed")

    spinup_start = time.time()

    if state_in:
        spun_state, actual_spinup_time = load_spectral_state(
            state_in,
            p_obj,
        )
        spinup_runtime = time.time() - spinup_start

        if not np.isclose(
            actual_spinup_time,
            spinup_time,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Loaded state time is {actual_spinup_time}, "
                f"but --spinup_time is {spinup_time}."
            )
    else:
        spun_state = run_forward_spinup(
            p_obj,
            w0,
            n0,
            spinup_steps=spinup_steps,
            solver=solver,
        )
        actual_spinup_time = spinup_steps * p_obj.dt
        spinup_runtime = time.time() - spinup_start

    finite_spinup = state_is_finite(spun_state)

    print(f"Spin-up final time: {actual_spinup_time:.6f}")
    print(f"Spin-up state finite: {finite_spinup}")
    print(f"Spin-up/load runtime: {spinup_runtime:.4f} s")

    if not finite_spinup:
        raise FloatingPointError(
            "Spin-up produced NaN/Inf before the AD window. "
            "This state cannot be used for sensitivity analysis."
        )

    if state_out and not state_in:
        save_spectral_state(
            state_out,
            spun_state,
            p_obj,
            elapsed_time=actual_spinup_time,
        )

    ad_start = time.time()
    mean_flux, grad_alpha = run_ad_window(
        p_obj,
        spun_state,
        avg_steps=avg_steps,
        solver=solver,
    )
    ad_runtime = time.time() - ad_start

    mean_flux_value = float(jax.device_get(mean_flux))
    grad_alpha_value = float(jax.device_get(grad_alpha))
    finite_result = bool(
        np.isfinite(mean_flux_value)
        and np.isfinite(grad_alpha_value)
    )

    print("\n--- Result ---")
    print(f"Window start time: {actual_spinup_time:.6f}")
    print(f"Window end time: {actual_spinup_time + avg_time:.6f}")
    print(f"Mean flux: {mean_flux_value:.12e}")
    print(f"d(mean flux)/d(alpha): {grad_alpha_value:.12e}")
    print(f"Finite result: {finite_result}")
    print(f"AD-window runtime: {ad_runtime:.4f} s")

    result = {
        "resolution": p_obj.nx,
        "dt": p_obj.dt,
        "alpha": p_obj.alpha,
        "kappa": p_obj.kappa,
        "diffw": p_obj.diffw,
        "diffn": p_obj.diffn,
        "diffop": p_obj.diffop,
        "solver": solver,
        "seed": seed,
        "spinup_time": actual_spinup_time,
        "avg_time": avg_time,
        "avg_steps": avg_steps,
        "mean_flux": mean_flux_value,
        "grad_alpha": grad_alpha_value,
        "finite": finite_result,
        "spinup_runtime_s": spinup_runtime,
        "ad_runtime_s": ad_runtime,
    }

    if result_csv:
        append_result_csv(result_csv, result)

    return result


# -----------------------------------------------------------------------------
# 7. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Production run.
    parser.add_argument("--nframes", type=int, default=20)
    parser.add_argument("--nts", type=int, default=100)
    parser.add_argument("--out", type=str, default="mhw_out.h5")

    # Numerical setup. Defaults preserve the supplied Stage-1 code.
    parser.add_argument(
        "--solver",
        type=str,
        default="leapfrog",
        choices=["leapfrog", "rk4"],
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffw", type=float, default=1e-4)
    parser.add_argument("--diffn", type=float, default=1e-4)
    parser.add_argument("--diffop", type=int, default=4)

    # Detached AD mode, expressed in physical simulation time.
    parser.add_argument(
        "--test_ad",
        action="store_true",
        help="Run detached nonlinear-state direct AD.",
    )
    parser.add_argument(
        "--spinup_time",
        type=float,
        default=300.0,
        help="Forward-only simulation time before the AD window.",
    )
    parser.add_argument(
        "--avg_time",
        type=float,
        default=1.0,
        help="Physical duration of the differentiated averaging window.",
    )
    parser.add_argument(
        "--state_in",
        type=str,
        default=None,
        help="Load a previously saved spectral spin-up state (.npz).",
    )
    parser.add_argument(
        "--state_out",
        type=str,
        default=None,
        help="Save the forward-spun spectral state for reuse (.npz).",
    )
    parser.add_argument(
        "--result_csv",
        type=str,
        default=None,
        help="Append the AD result to this CSV file.",
    )

    args = parser.parse_args()

    params = MHWParams(
        nx=args.res,
        ny=args.res,
        dt=args.dt,
        alpha=args.alpha,
        kappa=args.kappa,
        diffw=args.diffw,
        diffn=args.diffn,
        diffop=args.diffop,
    )
    params.print_config()

    # Preserve the original Stage-1 initialization exactly.
    np.random.seed(args.seed)
    w0 = 1.e-4 * (
        np.random.rand(args.res, args.res) - 0.5
    )
    n0 = w0.copy()

    w0_jax = xp.asarray(w0)
    n0_jax = xp.asarray(n0)

    if args.test_ad:
        run_nonlinear_state_ad_test(
            params,
            w0_jax,
            n0_jax,
            spinup_time=args.spinup_time,
            avg_time=args.avg_time,
            solver=args.solver,
            seed=args.seed,
            state_in=args.state_in,
            state_out=args.state_out,
            result_csv=args.result_csv,
        )
    else:
        run_production_io(
            params,
            w0_jax,
            n0_jax,
            nframes=args.nframes,
            steps_per_frame=args.nts,
            solver_name=args.solver,
            filename=args.out,
        )
