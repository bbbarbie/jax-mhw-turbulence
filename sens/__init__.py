"""Sensitivity-analysis layer on top of the archived MHW JAX solver.

Everything here differentiates the *discrete* map ``step_rk4`` from
``mhw_jax_stage2_ad.py`` with ``jax.jvp``.  No hand-written tangent equations.
"""
import jax

jax.config.update("jax_enable_x64", True)
