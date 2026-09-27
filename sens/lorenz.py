"""Lorenz 63 as a discrete map (RK4) for validating NILSS."""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def lorenz_rhs(u, rho, sigma=10.0, beta=8.0 / 3.0):
    x, y, z = u[0], u[1], u[2]
    return jnp.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def make_lorenz_step(dt=0.01, sigma=10.0, beta=8.0 / 3.0):
    def step(u, rho):
        k1 = lorenz_rhs(u, rho, sigma, beta)
        k2 = lorenz_rhs(u + 0.5 * dt * k1, rho, sigma, beta)
        k3 = lorenz_rhs(u + 0.5 * dt * k2, rho, sigma, beta)
        k4 = lorenz_rhs(u + dt * k3, rho, sigma, beta)
        return u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return step


def qoi_z(u):
    return u[2]


def rollout_mean(step, u0, s, nsteps, qoi):
    """Forward-only time mean of qoi over nsteps (left-point rule)."""
    @jax.jit
    def run(u):
        def body(c, _):
            u, acc = c
            return (step(u, s), acc + qoi(u)), None
        (u, acc), _ = jax.lax.scan(body, (u, 0.0), None, length=nsteps)
        return u, acc / nsteps
    return run(u0)
