"""
MGFderivative_class.py

Defines a class MGFDerivative that encapsulates the computation of MGF derivatives
and marginal likelihoods (evidence) for various likelihoods.
"""

import math
import sympy as sp
import pandas as pd
import numpy as np
from derivativeDispatch import mgfDerivative

# ---- Import all ready* and c* functions ----
from like_stats.Poisson import readyPoisson, cPoisson
from like_stats.Gamma import readyGamma, cGamma

# Other likelihoods (to be implemented later)
def _not_implemented(*args, **kwargs):
    raise NotImplementedError("This likelihood's module has not been implemented yet.")


class MGFDerivative:
    """
    Represents a prior MGF derivative (integer or fractional) ready to be combined with data.

    The derivative order (a) and evaluation point (t = -b) are determined from the data
    via the sufficient statistics of the likelihood.

    Currently supports:
        - Poisson, Gamma, Laplace, normal, Rayleigh, Maxwell-Boltzmann,
          inverse gamma, Levy, Weibull, BurrXII, Pareto, Dagum, Gompertz

    Special route:
        - Weibull has a special derivative route that converts an n-th order derivative
          into n 1st-order partial derivatives.
    """

    # Registry: maps likelihood name -> (ready_func, c_func)
    _registry = {
        'poisson':           (readyPoisson, cPoisson),
        'gamma':             (readyGamma,   cGamma),
        'laplace':           (_not_implemented, _not_implemented),
        'normal':            (_not_implemented, _not_implemented),
        'rayleigh':          (_not_implemented, _not_implemented),
        'maxwell-boltzmann': (_not_implemented, _not_implemented),
        'inverse gamma':     (_not_implemented, _not_implemented),
        'levy':              (_not_implemented, _not_implemented),
        'weibull':           (_not_implemented, _not_implemented),  # special derivative route
        'burrxii':           (_not_implemented, _not_implemented),
        'pareto':            (_not_implemented, _not_implemented),
        'dagum':             (_not_implemented, _not_implemented),
        'gompertz':          (_not_implemented, _not_implemented),
    }

    # Likelihoods that use a special derivative computation (instead of standard mgfDerivative)
    _special_likelihoods = {'weibull'}

    # Keys that are meant for the likelihood's ready function (not for mgfDerivative)
    _ready_keys = {'scale', 'shape'}   # extend as new likelihoods are added

    def __init__(self, prior, data, likelihood='poisson', method='symbolic',
                 params=None, simplify=False, log=True, **kwargs):
        """
        Compute the MGF derivative for the given data and prior.

        Parameters
        ----------
        prior : str
            'gamma' or 'pareto'.
        data : pandas DataFrame, Series, or array‑like
            Observed data.
        likelihood : str, optional
            One of the supported likelihoods (default 'poisson').
        method : str, optional
            For integer order: 'symbolic', 'bell', 'jax'.
            For fractional order: 'scipy', 'mpmath', 'symbolic' (if order is fractional).
            Default 'symbolic'.
        params : dict or None
            Prior parameters. If None and method='symbolic', returns symbolic expression.
        simplify : bool, optional
            If True, simplify symbolic expressions.
        log : bool, optional
            If True, store derivative in log scale (numeric only).
        **kwargs : additional arguments passed to the likelihood's ready function
                   and/or to mgfDerivative.
            For Poisson: scale.
            For Gamma: shape.
            For Weibull: shape (rho) and possibly others.
            For mgfDerivative: integer_method, epsrel, dps, tol, etc.
        """
        self.prior = prior
        self.method = method
        self.simplify = simplify
        self.log = log
        self.likelihood = likelihood.lower()
        self.params = params
        self.data = data

        # ---- Separate kwargs for ready vs derivative ----
        self._ready_kwargs = {k: v for k, v in kwargs.items() if k in self._ready_keys}
        self._deriv_kwargs = {k: v for k, v in kwargs.items() if k not in self._ready_keys}

        # ---- Look up ready and c functions from registry ----
        if self.likelihood not in self._registry:
            raise ValueError(f"Unsupported likelihood: {likelihood}. "
                             f"Choose from {list(self._registry.keys())}")

        self.ready_func, self.c_func = self._registry[self.likelihood]

        # ---- Compute sufficient statistics ----
        # Pass only kwargs that are relevant to ready_func (but with **kwargs in ready, we can pass all)
        # However, to be safe, we pass only self._ready_kwargs (since ready functions ignore extra anyway)
        stats = self.ready_func(data, **self._ready_kwargs)
        self.a = stats['a']
        self.b = stats['b']
        self.log_c = stats['log_c']

        # ---- Compute derivative ----
        if self.likelihood in self._special_likelihoods:
            self._compute_derivative_special()
        else:
            self._compute_derivative_normal()

    def _compute_derivative_normal(self):
        """Standard derivative computation using mgfDerivative."""
        result = mgfDerivative(
            order=self.a,
            prior=self.prior,
            method=self.method,
            t=float(-self.b),
            params=self.params,
            simplify=self.simplify,
            log=self.log,
            **self._deriv_kwargs
        )
        self._store_result(result)

    def _compute_derivative_special(self):
        """
        Special derivative computation for Weibull.
        This converts an n-th order derivative into n 1st-order partial derivatives.
        """
        # Placeholder: to be implemented specifically for Weibull.
        # For now, we raise a clear error to remind the developer.
        raise NotImplementedError(
            f"Special derivative route for '{self.likelihood}' not yet implemented. "
            f"Please implement in {self.__class__.__name__}._compute_derivative_special()."
        )

    def _store_result(self, result):
        """Store the result from mgfDerivative."""
        if isinstance(result, sp.Expr):
            self._symbolic = True
            self._expr = result
            self._log_abs = None
            self._sign = None
            self._value = None
        else:
            self._symbolic = False
            if self.log:
                self._log_abs, self._sign = result
                self._value = None
            else:
                self._value = result
                self._log_abs = None
                self._sign = None

    # ---- Properties ----
    @property
    def is_symbolic(self):
        return self._symbolic

    @property
    def expr(self):
        if not self._symbolic:
            raise ValueError("This is a numeric result; use .log_abs / .value instead.")
        return self._expr

    @property
    def log_abs(self):
        if self._symbolic:
            raise ValueError("This is a symbolic result; use .expr instead.")
        if self._log_abs is None:
            if self._value is not None:
                if self._value == 0:
                    return -float('inf')
                return math.log(abs(self._value))
            else:
                raise ValueError("No log_abs available.")
        return self._log_abs

    @property
    def sign(self):
        if self._symbolic:
            raise ValueError("This is a symbolic result; use .expr instead.")
        if self._sign is None:
            if self._value is not None:
                return 1 if self._value > 0 else -1
            else:
                raise ValueError("No sign available.")
        return self._sign

    @property
    def value(self):
        """Ordinary‑scale value (if numeric)."""
        if self._symbolic:
            raise ValueError("This is a symbolic result; use .expr instead.")
        if self._value is None:
            if self._log_abs == -float('inf'):
                return 0.0
            return self.sign * math.exp(self._log_abs)
        return self._value

    # ---- Methods ----
    def to_ordinary(self):
        """Return a new MGFDerivative with ordinary scale (if numeric)."""
        if self._symbolic:
            return self
        return MGFDerivative(
            prior=self.prior,
            data=self.data,
            likelihood=self.likelihood,
            method=self.method,
            params=self.params,
            simplify=self.simplify,
            log=False,
            **self._deriv_kwargs
        )

    def to_log(self):
        """Return a new MGFDerivative in log scale (if numeric)."""
        if self._symbolic:
            return self
        return MGFDerivative(
            prior=self.prior,
            data=self.data,
            likelihood=self.likelihood,
            method=self.method,
            params=self.params,
            simplify=self.simplify,
            log=True,
            **self._deriv_kwargs
        )

    def evidence(self):
        """
        Return the marginal likelihood (evidence).

        For numeric: returns (log_abs, sign) if self.log=True, else ordinary float.
        For symbolic: returns sympy.Expr (c_expr * derivative_expr).
        """
        if self.is_symbolic:
            c_expr = self.c_func()
            return c_expr * self.expr
        else:
            total_log_abs = self.log_c + self.log_abs
            if self.log:
                return total_log_abs, self.sign
            else:
                return math.exp(total_log_abs) * self.sign