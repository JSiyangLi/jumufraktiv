# jumufraktiv/symbols.py

import sympy as sp


# ============================================================
# Global canonical symbols
# ============================================================

# transform variable (MGF / CGF domain)
t = sp.Symbol("t", real=True)

# latent variable (parameter space)
theta = sp.Symbol("theta", positive=True, real=True)

# posterior MGF variable
r = sp.Symbol("r", real=True)

# cdf evaluation point
u = sp.Symbol("u", real=True)

# ============================================================
# Parameter symbol factory (important for extensibility)
# ============================================================

def param(name: str):
    """
    Create a symbolic parameter safely.

    This avoids global clutter like alpha_sym, beta_sym, xi_sym everywhere.
    """
    return sp.Symbol(name, real=True, positive=True)