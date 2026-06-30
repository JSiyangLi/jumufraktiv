"""
Test script for custom prior, custom likelihood, and sequential updating
with likelihood changes.

Tests three scenarios:
1. Custom prior + named likelihood -> update to another named likelihood.
2. Custom prior + named likelihood -> update to custom likelihood.
3. Named prior + custom likelihood -> update to named likelihood.
"""

import numpy as np
import pandas as pd
import math
import sympy as sp
from MGFderivative_class import MGFDerivative
from like_stats.Poisson import readyPoisson, cPoisson
from like_stats.Gamma import readyGamma, cGamma


# ===== Helper functions for custom Gamma prior =====
def custom_gamma_mgf(t, alpha, beta):
    """MGF of Gamma(alpha, beta) with rate beta."""
    return (beta / (beta - t)) ** alpha

def custom_gamma_cgf(t, alpha, beta):
    """CGF of Gamma(alpha, beta) with rate beta."""
    return alpha * (math.log(beta) - math.log(beta - t))

def custom_gamma_pdf(theta, alpha, beta):
    """PDF of Gamma(alpha, beta) with rate beta."""
    return (beta**alpha / math.gamma(alpha)) * theta**(alpha-1) * math.exp(-beta * theta)

# ===== Helper functions for custom Gamma likelihood =====
def ready_gamma_custom(data, alpha_lik, **kwargs):
    """
    Custom ready function for Gamma likelihood with known shape alpha_lik.
    Returns a, b, log_c.
    """
    y = np.asarray(data)
    n = len(y)
    a = n * alpha_lik
    b = np.sum(y)
    # log_c = sum((alpha_lik-1)*log(y) - lgamma(alpha_lik))
    log_c = np.sum((alpha_lik - 1) * np.log(y) - math.lgamma(alpha_lik))
    return {'a': a, 'b': b, 'log_c': log_c}

def c_gamma_custom():
    """Custom c function for Gamma likelihood."""
    n = sp.Symbol('n', integer=True, positive=True)
    alpha = sp.Symbol('alpha_lik', positive=True, real=True)
    y = sp.IndexedBase('y')
    i = sp.Idx('i')
    # ∏ y_i^(alpha-1) / Γ(alpha)
    expr = sp.Product(y[i]**(alpha - 1) / sp.gamma(alpha), (i, 1, n))
    return expr


# ===== Test data =====
# For Poisson likelihood: counts
data_poisson = pd.DataFrame({'y': [1, 2, 3]})
scale_poisson = 1.0

# For Gamma likelihood: positive values
data_gamma = pd.DataFrame({'y': [0.5, 1.0, 1.5]})
shape_gamma_lik = 2.0   # known shape

# Prior hyperparameters for custom Gamma prior
alpha_prior = 2.0
beta_prior = 3.0

print("=" * 60)
print("Scenario 1: Custom prior + named Poisson -> update to named Gamma")
print("=" * 60)

# ---- Custom prior + Poisson (named) ----
deriv1 = MGFDerivative(
    prior='custom',
    data=data_poisson['y'],
    likelihood='poisson',
    method='jax',               # numeric derivative
    params=None,                # no params, embedded in functions
    prior_mgf_func=lambda t: custom_gamma_mgf(t, alpha_prior, beta_prior),
    prior_cgf_func=lambda t: custom_gamma_cgf(t, alpha_prior, beta_prior),
    prior_pdf_func=lambda theta: custom_gamma_pdf(theta, alpha_prior, beta_prior),
    prior_logpdf_func=lambda theta: np.log(custom_gamma_pdf(theta, alpha_prior, beta_prior)),
    scale=scale_poisson,
    log=True
)

# Compute evidence for first chunk
log_ev1, sign1 = deriv1.evidence()
print(f"Evidence 1 (Poisson) log = {log_ev1:.6f}, sign = {sign1}")

# ---- Update to Gamma likelihood (named) ----
new_data = pd.DataFrame({'y': [0.5, 1.0]})   # new data under Gamma
# Change likelihood to 'gamma' with known shape (shape=shape_gamma_lik)
# Note: Gamma likelihood requires shape parameter; we pass it via kwargs.
deriv2 = deriv1.update(
    new_data=new_data['y'],
    method='jax',
    log=True,
    likelihood='gamma',
    shape=shape_gamma_lik   # parameter for Gamma likelihood
)

log_ev2, sign2 = deriv2.evidence()
print(f"Evidence 2 (Gamma) log = {log_ev2:.6f}, sign = {sign2}")
print("A warning about likelihood change should appear above.\n")


print("=" * 60)
print("Scenario 2: Custom prior + named Poisson -> update to custom Gamma")
print("=" * 60)

# Re-use deriv1 (custom prior + Poisson)
# ---- Update to custom Gamma likelihood ----
new_data2 = pd.DataFrame({'y': [0.5, 1.0, 1.5]})
deriv3 = deriv1.update(
    new_data=new_data2['y'],
    method='jax',
    log=True,
    likelihood='custom',
    ready_func=lambda data, **kw: ready_gamma_custom(data, alpha_lik=shape_gamma_lik, **kw),
    c_func=c_gamma_custom
)

log_ev3, sign3 = deriv3.evidence()
print(f"Evidence 3 (custom Gamma) log = {log_ev3:.6f}, sign = {sign3}")
print("A warning about likelihood change should appear above.\n")


print("=" * 60)
print("Scenario 3: Named prior + custom Gamma likelihood -> update to named Poisson")
print("=" * 60)

# ---- Named Gamma prior + custom Gamma likelihood ----
deriv4 = MGFDerivative(
    prior='gamma',
    data=data_gamma['y'],
    likelihood='custom',
    method='jax',
    params={'alpha': alpha_prior, 'beta': beta_prior},   # prior hyperparameters
    ready_func=lambda data, **kw: ready_gamma_custom(data, alpha_lik=shape_gamma_lik, **kw),
    c_func=c_gamma_custom,
    log=True
)

log_ev4, sign4 = deriv4.evidence()
print(f"Evidence 4 (named Gamma prior + custom Gamma likelihood) log = {log_ev4:.6f}, sign = {sign4}")

# ---- Update to Poisson likelihood (named) ----
new_data3 = pd.DataFrame({'y': [1, 2]})
deriv5 = deriv4.update(
    new_data=new_data3['y'],
    method='jax',
    log=True,
    likelihood='poisson',
    scale=scale_poisson
)

log_ev5, sign5 = deriv5.evidence()
print(f"Evidence 5 (after update to Poisson) log = {log_ev5:.6f}, sign = {sign5}")
print("A warning about likelihood change should appear above.\n")


print("=" * 60)
print("All tests completed successfully.")
print("Check that warnings for likelihood changes were printed.")