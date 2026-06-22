"""
evidence.py

Compute the marginal likelihood (evidence) for Poisson likelihoods
using MGF marginalisation.

Imports:
    - mgfDerivative_integer from derivativeDispatch
    - readyPoisson, cPoisson from like_stats.Poisson
"""

import sympy as sp
import math
from derivativeDispatch import mgfDerivative_integer
from like_stats.Poisson import readyPoisson, cPoisson


def evidence(
    likelihood: str,
    prior: str,
    data,
    method: str = "symbolic",
    params: dict = None,
    simplify: bool = False,
    log: bool = True,
    **kwargs
):
    """
    Compute the marginal likelihood (evidence) for a given likelihood and prior.

    Parameters
    ----------
    likelihood : str
        Currently only 'poisson' is supported.
    prior : str
        'gamma' or 'pareto'.
    data : pandas DataFrame, Series, or array-like
        Observed counts (must be non‑negative).
    method : str, optional
        One of 'symbolic', 'bell', 'jax'. Default 'symbolic'.
    params : dict, optional
        Prior parameters. If method='symbolic' and params is None or empty,
        returns a symbolic expression in terms of prior parameters.
        If params is provided (numeric or symbolic), it is passed to
        mgfDerivative_integer.
    simplify : bool, optional
        If True, simplify the symbolic expression (only for 'symbolic' method).
    log : bool, optional
        If True and output is numeric, return (log_abs, sign) on log scale.
        If False, return ordinary value.
    **kwargs : additional arguments for readyPoisson (e.g., scale).

    Returns
    -------
    For numeric evaluation:
        - If log=True: tuple (log_abs, sign) where log_abs = log|evidence|.
        - If log=False: float (ordinary value).
    For symbolic evaluation (method='symbolic' and params is None/empty):
        - sympy.Expr : symbolic expression c * M^{(a)}(-b)
    """
    if likelihood.lower() != "poisson":
        raise NotImplementedError(...)

    stats = readyPoisson(data, **kwargs)
    a, b, log_c = stats['a'], stats['b'], stats['log_c']

    # Symbolic (un‑evaluated) mode
    if method.lower() == "symbolic" and (params is None or not params):
        deriv_expr = mgfDerivative_integer(
            order=int(a),
            prior=prior,
            method="symbolic",
            t=float(-b),
            params=None,
            simplify=simplify
        )
        return cPoisson() * deriv_expr

    # Numeric evaluation (any method)
    if params is None:
        raise ValueError("For numeric evaluation, params must be provided.")
    
    log_abs, sign = mgfDerivative_integer(
        order=int(a),
        prior=prior,
        method=method,
        t=float(-b),
        params=params,
        simplify=simplify,
        log=True
    )

    total_log_abs = log_c + log_abs
    if log:
        return total_log_abs, sign
    else:
        return math.exp(total_log_abs) * sign


# ===== Example usage =====
if __name__ == "__main__":
    import pandas as pd
    # Create sample data
    data = pd.DataFrame({'counts': [1, 2, 3, 4]})
    # Numeric evaluation with Gamma prior
    params_num = {'alpha': 2.0, 'beta': 3.0}
    res_num = evidence("poisson", "gamma", data, method="symbolic", params=params_num, scale=1.0)
    print("Numeric evidence (log scale):", res_num)

    # Symbolic expression (params not provided)
    res_sym = evidence("poisson", "gamma", data, method="symbolic", params=None, scale=1.0)
    print("Symbolic evidence expression:")
    sp.pprint(res_sym)