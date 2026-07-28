"""
MGFderivative_class.py

Defines a class MGFDerivative that encapsulates the computation of MGF derivatives
and marginal likelihoods (evidence) for various likelihoods and priors.

Supports sequential updating via the `update` method, using the posterior MGF
as the prior for the next chunk of data.

Priors are represented as mitMGFprior objects.
"""

import math
import traceback
from unittest import result
import sympy as sp
import numpy as np
import pandas as pd

from jumufraktiv.derivativeDispatch import mgfDerivative
from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.symbols import t, theta, r, u
from jumufraktiv.root_finding import solve_root

# ============================================================
# Likelihood registry
# ============================================================
# ---- Likelihood imports ----
from jumufraktiv.like_stats.Poisson import readyPoisson, cPoisson, bereitPoisson
from jumufraktiv.like_stats.Gamma import readyGamma, cGamma, bereitGamma
from jumufraktiv.like_stats.Laplace import readyLaplace, cLaplace, bereitLaplace
from jumufraktiv.like_stats.Normal import readyNormal, cNormal, bereitNormal
from jumufraktiv.like_stats.Rayleigh import readyRayleigh, cRayleigh, bereitRayleigh
from jumufraktiv.like_stats.MaxwellBoltzmann import readyMaxwellBoltzmann, cMaxwellBoltzmann, bereitMaxwellBoltzmann
from jumufraktiv.like_stats.InverseGamma import readyInverseGamma, cInverseGamma, bereitInverseGamma
from jumufraktiv.like_stats.Levy import readyLevy, cLevy, bereitLevy
from jumufraktiv.like_stats.Weibull import readyWeibull, cWeibull, bereitWeibull
from jumufraktiv.like_stats.BurrXII import readyBurrXII, cBurrXII, bereitBurrXII
from jumufraktiv.like_stats.Pareto import readyPareto, cPareto, bereitPareto
from jumufraktiv.like_stats.Dagum import readyDagum, cDagum, bereitDagum
from jumufraktiv.like_stats.Gompertz import readyGompertz, cGompertz, bereitGompertz
from jumufraktiv.like_stats.HalfNormal import readyHalfNormal, cHalfNormal, bereitHalfNormal


# ============================================================
# Likelihood registry
# ============================================================
LIKELIHOOD_REGISTRY = {
    'poisson': (readyPoisson, cPoisson, bereitPoisson),
    'gamma': (readyGamma, cGamma, bereitGamma),
    'laplace': (readyLaplace, cLaplace, bereitLaplace),
    'normal': (readyNormal, cNormal, bereitNormal),
    'rayleigh': (readyRayleigh, cRayleigh, bereitRayleigh),
    'maxwell-boltzmann': (readyMaxwellBoltzmann, cMaxwellBoltzmann, bereitMaxwellBoltzmann),
    'inverse gamma': (readyInverseGamma, cInverseGamma, bereitInverseGamma),
    'levy': (readyLevy, cLevy, bereitLevy),
    'weibull': (readyWeibull, cWeibull, bereitWeibull),
    'burrxii': (readyBurrXII, cBurrXII, bereitBurrXII),
    'pareto': (readyPareto, cPareto, bereitPareto),
    'dagum': (readyDagum, cDagum, bereitDagum),
    'gompertz': (readyGompertz, cGompertz, bereitGompertz),
    'halfnormal': (readyHalfNormal, cHalfNormal, bereitHalfNormal),
}


# ============================================================
# Core class
# ============================================================
class MGFDerivative:

    def __init__(
        self,
        prior,                  # mitMGFprior object ONLY
        data,
        likelihood='poisson',
        method='auto',
        simplify=False,
        log=True,
        **kwargs
    ):
        """
        prior must be a mitMGFprior instance.
        """
        # ----------------------------------------------------
        # PRIOR HANDLING
        # ----------------------------------------------------
        if not isinstance(prior, mitMGFprior):
            raise TypeError("prior must be a mitMGFprior object")

        self.prior = prior
        self.params = prior.params

        # ----------------------------------------------------
        # LIKELIHOOD
        # ----------------------------------------------------
        self.likelihood = likelihood.lower()
        self.data = data
        self.method = method
        self.simplify = simplify
        self.log = log

        if self.likelihood not in LIKELIHOOD_REGISTRY:
            raise ValueError(f"Unknown likelihood: {likelihood}")

        self.ready_func, self.c_func, self.bereit_func = LIKELIHOOD_REGISTRY[self.likelihood]

        # ----------------------------------------------------
        # Separate kwargs for ready vs derivative
        # ----------------------------------------------------
        _ready_keys = {
            'scale', 'shape', 'mean', 'location', 'rho',
            'known_shape', 'r', 's',
        }
        self._ready_kwargs = {k: v for k, v in kwargs.items() if k in _ready_keys}
        self._deriv_kwargs = {k: v for k, v in kwargs.items() if k not in _ready_keys}

        # ----------------------------------------------------
        # Sufficient statistics
        # ----------------------------------------------------
        stats = self.ready_func(data, **self._ready_kwargs)
        self.a = stats['a']
        self.b = stats['b']
        self.log_c = stats['log_c']

        # ----------------------------------------------------
        # Build derivative representation
        # ----------------------------------------------------
        self._build_derivative()

        # ----------------------------------------------------
        # Evaluate derivative at posterior point t=-b
        # ----------------------------------------------------
        self._compute()

    # ========================================================
    # CORE COMPUTATION, 3-layer design
    # ========================================================
    def _build_derivative(self):
        """
        Construct D_a(t)=M^(a)(t) without evaluating at t=-b.
        """
        self._deriv = mgfDerivative(
            order=self.a,
            prior=self.prior,
            method=self.method,
            t=None,
            u=None,
            simplify=self.simplify,
            complete=True,
            log=False,
            **self._deriv_kwargs
        )
        self._deriv_is_symbolic = isinstance(self._deriv, sp.Expr) # Do I have an unevaluated symbolic derivative function representation DM(t)? It tells you whether you can work with it symbolically.
    
    def _evaluate_derivative(self, t_value):
        if isinstance(self._deriv, sp.Basic):
            val = self._deriv.subs(t, t_value)
            if not val.free_symbols:
                numeric_val = float(val.evalf())
                if self.log:
                    # Return (log_abs, sign)
                    if abs(numeric_val) < 1e-300:
                        return (-float('inf'), 1)
                    return (np.log(abs(numeric_val)), 1 if numeric_val > 0 else -1)
                else:
                    # Return scalar
                    return numeric_val
            return val
        else:
            # Numeric function (callable)
            return self._deriv(t_value, **self._deriv_kwargs)
    
    def _compute(self):
        """
        Delegates ALL math to mgfDerivative,
        using mitMGFprior ONLY as input.
        """
        result = self._evaluate_derivative(-self.b)
        self._store_result(result)

    # ========================================================
    # RESULT STORAGE
    # ========================================================
    def _store_result(self, result):
        """
        Store the result of evaluating the derivative at t = -b.

        The derivative at -b is the normalising constant of the posterior.
        It must be positive (since it represents the marginal likelihood).
        If the sign is negative, an error is raised.
        """
        # ----------------------------------------------------
        # Symbolic state
        # ----------------------------------------------------
        if isinstance(result, sp.Expr):
            self._result_expr = result
            self._is_symbolic = True
            self.log_abs = None
            self._sign = None
            self.value = None
            return

        # ----------------------------------------------------
        # Numeric state
        # ----------------------------------------------------
        self._result_expr = None
        self._is_symbolic = False

        if self.log:
            # Expect (log_abs, sign)
            if not isinstance(result, tuple):
                raise TypeError(
                    "Expected (log_abs, sign) tuple when log=True."
                )
            log_abs, sign = result
            # Sign must be positive for the normalising constant
            if sign == -1:
                raise ValueError(
                    "Derivative at t=-b is negative. "
                    "This suggests a numerical issue or invalid likelihood/prior. "
                    "Posterior density cannot be negative."
                )
            self.log_abs = log_abs
            self._sign = sign
            self.value = None
        else:
            # Expect ordinary numeric value
            if isinstance(result, tuple):
                raise TypeError(
                    "Expected numeric value when log=False, "
                    "but received (log_abs, sign)."
                )
            value = float(result)
            if value < 0:
                raise ValueError(
                    "Derivative at t=-b is negative. "
                    "This suggests a numerical issue or invalid likelihood/prior. "
                    "Posterior density cannot be negative."
                )
            self.value = value
            self.log_abs = None
            self._sign = None

    @property
    def is_symbolic(self):
        return self._is_symbolic
    
    @property
    def value_numeric(self):
        if self._is_symbolic:
            raise ValueError("Result is symbolic")

        if self.log:
            return self._sign * np.exp(self.log_abs)

        return self.value
    
    @property
    def prior_has_iMGF(self) -> bool:
        return self.prior.has_iMGF()

    # ========================================================
    # EVIDENCE
    # ========================================================
    def evidence(self):
        """
        Return the marginal likelihood (evidence).

        If `self.is_symbolic` is True, returns a symbolic expression.

        Otherwise:
            - if self.log=True: returns (log_abs, sign)
            - if self.log=False: returns ordinary numeric value
        """

        if self._is_symbolic:
            return self.c_func() * self._result_expr

        else:
            if self.log:
                total_log_abs = self.log_c + self.log_abs
                return total_log_abs, self._sign

            else:
                return np.exp(self.log_c) * self.value

    # ========================================================
    # POSTERIOR DENSITY
    # ========================================================
        
    def post_density(self, theta_val=None, log=True):
        """
        Compute the posterior density (or log-density) at given θ.

        If `self._deriv_is_symbolic` is True:
            - If theta_val is None or a sympy Symbol: returns a symbolic expression.
            - If theta_val is numeric (scalar or array): evaluates the expression numerically.
            - Uses the prior's symbolic PDF if available.
        If `self._deriv_is_symbolic` is False: performs numeric evaluation (vectorized).

        Parameters
        ----------
        theta_val : scalar, array-like, or sympy Symbol, optional
            Evaluation point(s). If array, must be convertible to numpy array.
        log : bool, optional
            If True, return log-density; else ordinary density.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            If theta_val is scalar and numeric: returns scalar.
            If theta_val is array-like: returns array.
            If theta_val is symbolic or has free symbols: returns sympy.Expr.
        """
        # ---- Symbolic path ----
        if self._deriv_is_symbolic and (theta_val is None or isinstance(theta_val, sp.Symbol)):
            try:
                denom_expr = self._deriv.subs(t, -self.b)

                if theta_val is None or isinstance(theta_val, sp.Symbol):
                    theta_sym = theta if theta_val is None else theta_val
                else:
                    theta_sym = theta

                pdf_sym = self.prior.pdf_sym
                if pdf_sym is None:
                    raise ValueError("No symbolic PDF available for this prior.")
                if callable(pdf_sym):
                    pdf_sym = pdf_sym()
                if not isinstance(pdf_sym, sp.Expr):
                    raise TypeError("pdf_sym must be a SymPy expression.")

                log_prior = sp.log(pdf_sym)
                if self.params is not None:
                    subs_dict = {}
                    for sym in pdf_sym.free_symbols:
                        if sym.name in self.params:
                            subs_dict[sym] = self.params[sym.name]
                    if subs_dict:
                        pdf_sym = pdf_sym.subs(subs_dict)

                log_num = log_prior + self.a * sp.log(theta_sym) - self.b * theta_sym
                log_post = log_num - sp.log(denom_expr)

                # Handle numeric theta_val (scalar or array)
                if theta_val is not None and not isinstance(theta_val, sp.Symbol):
                    # Convert to array if not already
                    theta_arr = np.asarray(theta_val)
                    scalar_input = theta_arr.ndim == 0
                    if scalar_input:
                        theta_arr = np.array([theta_val])
                    batch = len(theta_arr)

                    # Pre-allocate results
                    results_log = np.zeros(batch)
                    results_sym = [None] * batch  # store expressions if any remain symbolic

                    for idx, t_val in enumerate(theta_arr):
                        evaluated = log_post.subs(theta_sym, t_val).evalf()
                        if evaluated.free_symbols:
                            # If any free symbols remain, we cannot fully numericize.
                            # For scalar, return expression; for array, raise error.
                            if batch == 1:
                                return evaluated if log else sp.exp(evaluated)
                            else:
                                # Store expression and continue; later we may raise or return mixed.
                                results_sym[idx] = evaluated
                        else:
                            results_log[idx] = float(evaluated)
                            results_sym[idx] = None  # numeric

                    # Check if any symbolic results remain
                    if any(r is not None for r in results_sym):
                        # For array input with mixed symbolic/numeric, we cannot return a uniform array.
                        # We'll raise an error to avoid confusion.
                        raise ValueError(
                            "Vectorized symbolic evaluation failed: some theta values "
                            "still have free symbols. Use scalar symbolic input."
                        )
                    else:
                        # All numeric
                        if log:
                            return results_log[0] if scalar_input else results_log
                        else:
                            dens = np.exp(results_log)
                            return float(dens[0]) if scalar_input else dens
                else:
                    # theta_val is None or Symbol: return expression
                    return log_post if log else sp.exp(log_post)

            except Exception as e:
                raise RuntimeError(f"Symbolic posterior density computation failed: {e}") from e

        # ---- Numeric path (vectorized) ----
        if theta_val is None:
            raise ValueError("For numeric evaluation, theta must be provided.")

        # Ensure theta_val is a numpy array
        if not isinstance(theta_val, np.ndarray):
            theta_arr = np.asarray(theta_val)
            scalar_input = theta_arr.ndim == 0
            if scalar_input:
                theta_arr = np.array([theta_val])
        else:
            theta_arr = theta_val
            scalar_input = theta_arr.ndim == 0
            if scalar_input:
                theta_arr = np.array([theta_val])

        # Get log prior density (vectorized)
        if self.prior.logpdf_func is not None:
            log_prior = self.prior.logpdf_func(theta_arr)
        elif self.prior.pdf_func is not None:
            prior_pdf = self.prior.pdf_func(theta_arr)
            if np.any(prior_pdf <= 0):
                raise ValueError("Prior PDF must be positive for all theta values.")
            log_prior = np.log(prior_pdf)
        else:
            raise ValueError("No numeric PDF function available for this prior.")

        log_num = log_prior + self.a * np.log(theta_arr) - self.b * theta_arr
        log_denom = self.log_abs if self.log else np.log(self.value)
        log_post = log_num - log_denom

        # Check numerical validity of log-posterior
        if np.any(np.isnan(log_post)):
            raise ValueError("NaN encountered in log-posterior.")
        if np.any(np.isposinf(log_post)):
            raise ValueError("+Inf encountered in log-posterior.")

        # Output: scalar or array
        if scalar_input:
            return float(log_post[0]) if log else float(np.exp(log_post[0]))
        else:
            return log_post if log else np.exp(log_post)
        
    # ========================================================
    # POSTERIOR CUMULATIVE DENSITY
    # ========================================================
    def post_cdf(self, u_val=None, log=True):
        """
        Compute the posterior CDF F(Θ ≤ u | y) (or log‑CDF) at threshold(s) u.

        If `self._is_symbolic` is True:
            - If u_val is None or a sympy Symbol: returns a symbolic expression.
            - If u_val is numeric (scalar or array): evaluates the expression numerically.
            - Requires the prior's symbolic incomplete MGF (imgf_sym).
        If `self._is_symbolic` is False: performs numeric evaluation using the prior's
        numeric imgf/logimgf functions. Supports tuple‑vectorisation: u can be an array.

        Parameters
        ----------
        u_val : scalar, array-like, or sympy Symbol, optional
            Upper limit(s) for the CDF. If array-like, returns array of CDFs.
        log : bool, optional
            If True, return log-CDF; else ordinary CDF.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            If u_val is scalar and numeric: scalar.
            If u_val is array-like: array.
            If u_val is symbolic: sympy.Expr.
        """
        # ---- Ensure iMGF support ----
        if not hasattr(self.prior, "has_iMGF") or not self.prior.has_iMGF():
            raise RuntimeError("Prior does not support incomplete MGF (iMGF).")

        # ---- Symbolic path ----
        if self._is_symbolic:
            try:
                # Build symbolic derivative of incomplete MGF once
                num_expr = mgfDerivative(
                    order=self.a,
                    prior=self.prior,
                    method="symbolic",
                    t=None,
                    simplify=self.simplify,
                    complete=False,
                    log=False,
                    **self._deriv_kwargs
                )
                # Evaluate at t = -b (fixed)
                num_expr = num_expr.subs(t_sym, -self.b)
                denom_expr = self._deriv.subs(t_sym, -self.b)
                log_cdf_expr = sp.log(num_expr) - sp.log(denom_expr)

                # Substitute known hyperparameters
                if self.params is not None:
                    subs_dict = {sym: self.params[sym.name]
                                for sym in log_cdf_expr.free_symbols
                                if sym.name in self.params}
                    if subs_dict:
                        log_cdf_expr = log_cdf_expr.subs(subs_dict)

                # Determine if u_val is symbolic or numeric
                if isinstance(u_val, sp.Symbol):
                    return log_cdf_expr if log else sp.exp(log_cdf_expr)
                elif u_val is None:
                    return log_cdf_expr if log else sp.exp(log_cdf_expr)
                else:
                    # Numeric u: vectorize evaluation
                    orig_shape = np.shape(u_val)
                    u_flat = np.asarray(u_val).ravel()
                    def eval_u(uu):
                        expr_i = log_cdf_expr.subs(u_sym, uu).evalf()
                        if expr_i.free_symbols:
                            raise ValueError(
                                "Symbolic expression still has free symbols after substituting u. "
                                "Cannot evaluate numerically."
                            )
                        return float(expr_i)
                    vec_eval = np.vectorize(eval_u)
                    results = vec_eval(u_flat).reshape(orig_shape)
                    if log:
                        return results.item() if np.ndim(results) == 0 else results
                    else:
                        return np.exp(results.item()) if np.ndim(results) == 0 else np.exp(results)

            except Exception as e:
                raise RuntimeError(f"Symbolic posterior CDF computation failed: {e}") from e

        # ---- Numeric path (vectorised with tuple‑vectorisation) ----
        if u_val is None:
            raise ValueError("For numeric evaluation, u must be provided.")

        # Store original shape
        orig_shape = np.shape(u_val)
        u_arr = np.asarray(u_val)
        scalar_input = u_arr.ndim == 0
        if scalar_input:
            u_arr = np.array([u_val])

        # Numerator: derivative of incomplete MGF at t = -b for all u values
        # mgfDerivative now supports tuple‑vectorisation: it will broadcast t=-self.b with u_arr.
        log_abs_num, sign_num = mgfDerivative(
            order=self.a,
            prior=self.prior,
            method=self.method,
            t=-self.b,
            simplify=self.simplify,
            complete=False,
            log=True,
            u=u_arr,
            **self._deriv_kwargs
        )

        # Denominator (already stored)
        if self.log:
            log_denom = self.log_abs
        else:
            log_denom = np.log(abs(self.value)) if self.value != 0 else -np.inf

        # Combine
        log_ratio = log_abs_num - log_denom
        sign_ratio = sign_num * (self._sign if self._sign is not None else 1.0)

        # Reshape to original shape
        log_ratio = log_ratio.reshape(orig_shape)
        sign_ratio = sign_ratio.reshape(orig_shape)

        # Consistency check: CDF must be non-negative
        if np.any(sign_ratio < 0):
            raise RuntimeError("Posterior CDF became negative – numerical issue.")

        if scalar_input:
            if log:
                return float(log_ratio.item())
            else:
                if log_ratio.item() == -float('inf'):
                    return 0.0
                return float(sign_ratio.item() * np.exp(log_ratio.item()))
        else:
            if log:
                return log_ratio
            else:
                result = sign_ratio * np.exp(log_ratio)
                result[log_ratio == -float('inf')] = 0.0
                return result

    # ========================================================
    # POSTERIOR PREDICTIVE
    # ========================================================
    def _post_predictive_symbolic_scalar(self, x, log=True, **kwargs):
        """
        Compute symbolic predictive density for a single scalar observation.
        This is the original scalar logic, kept unchanged.
        """
        if not self._is_symbolic:
            raise RuntimeError("This helper is only for symbolic posterior.")

        try:
            # Wrap scalar in a list for ready_func (expects array‑like)
            stats_new = self.ready_func([x], **kwargs)
            a_new = stats_new['a']
            b_new = stats_new['b']
            log_c_new = stats_new['log_c']

            combined_order = self.a + a_new
            combined_b = self.b + b_new

            num = mgfDerivative(
                order=combined_order,
                prior=self.prior,
                method="symbolic",
                t=-combined_b,
                simplify=self.simplify,
                complete=True,
                log=True
            )

            if isinstance(num, tuple):
                raise RuntimeError("Symbolic predictive unexpectedly received numeric derivative.")

            denom = self._evaluate_derivative(-self.b)

            log_pred = log_c_new + sp.log(num) - sp.log(denom)

            if self.params is not None:
                log_pred = log_pred.subs(
                    {sym: self.params[sym.name] for sym in log_pred.free_symbols if sym.name in self.params}
                )

            if log_pred.free_symbols:
                return log_pred if log else sp.exp(log_pred)
            return float(log_pred.evalf()) if log else float(sp.exp(log_pred).evalf())

        except Exception as e:
            raise RuntimeError(f"Symbolic predictive computation failed: {e}") from e


    def post_predictive(self, new_data, log=True, individual=True, **kwargs):
        """
        Compute the posterior predictive density (or log-density) for new data.

        Parameters
        ----------
        new_data : scalar, array-like, or sympy.Symbol
            New observation(s). If array-like, each element is treated separately.
        log : bool, optional
            If True, return log-density; else ordinary density.
        individual : bool, optional
            If True (default), return an array of densities for each new data point.
            If False, return the joint density (product of individual densities).
        **kwargs : additional arguments passed to the likelihood's `ready_func` or `bereit_func`.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            Depending on input and `individual` flag.
        """
        # ---- Symbolic new_data (single symbol) ----
        if isinstance(new_data, sp.Symbol):
            if not self._is_symbolic:
                raise ValueError("Cannot compute predictive density symbolically with a numeric posterior.")
            # Directly build symbolic expression for the joint density (same as helper, but with symbolic stats)
            a_new = sp.Symbol('a_new', real=True)
            b_new = sp.Symbol('b_new', real=True)
            log_c_new = sp.Symbol('log_c_new', real=True)
            combined_order = self.a + a_new
            combined_b = self.b + b_new
            num = mgfDerivative(
                order=combined_order,
                prior=self.prior,
                method="symbolic",
                t=-combined_b,
                simplify=self.simplify,
                complete=True,
                log=True
            )
            if isinstance(num, tuple):
                raise RuntimeError("Symbolic predictive unexpectedly received numeric derivative.")
            denom = self._evaluate_derivative(-self.b)
            log_pred = log_c_new + sp.log(num) - sp.log(denom)
            if self.params is not None:
                log_pred = log_pred.subs(
                    {sym: self.params[sym.name] for sym in log_pred.free_symbols if sym.name in self.params}
                )
            if log_pred.free_symbols:
                return log_pred if log else sp.exp(log_pred)
            return float(log_pred.evalf()) if log else float(sp.exp(log_pred).evalf())

        # ---- Symbolic posterior with numeric new_data ----
        if self._is_symbolic:
            # Flatten input
            new_data_arr = np.asarray(new_data).ravel()
            scalar_input = len(new_data_arr) == 1

            if individual:
                # Compute per‑element densities using the scalar helper
                results = []
                for x in new_data_arr:
                    res = self._post_predictive_symbolic_scalar(x, log=log, **kwargs)
                    results.append(res)
                # If any result is symbolic, return a list; otherwise convert to array
                if any(isinstance(r, sp.Expr) for r in results):
                    return results[0] if scalar_input else results
                else:
                    return float(results[0]) if scalar_input else np.array(results)
            else:
                # Joint density: aggregate stats and call helper once
                stats = self.ready_func(new_data_arr, **kwargs)  # returns summed stats
                return self._post_predictive_symbolic_scalar(new_data_arr, log=log, **kwargs)

        # ---- Numeric path (non‑symbolic posterior, vectorised) ----
        new_data_arr = np.asarray(new_data).ravel()
        scalar_input = len(new_data_arr) == 1

        if individual:
            stats = self.bereit_func(new_data_arr, **kwargs)
            a_vals = np.asarray(stats['a']).ravel()
            b_vals = np.asarray(stats['b']).ravel()
            log_c_vals = np.asarray(stats['log_c']).ravel()
        else:
            stats = self.ready_func(new_data_arr, **kwargs)
            a_vals = np.array([stats['a']])
            b_vals = np.array([stats['b']])
            log_c_vals = np.array([stats['log_c']])

        a_comb = self.a + a_vals
        b_comb = self.b + b_vals
        log_abs_num, sign_num = mgfDerivative(
            order=a_comb,
            prior=self.prior,
            method=self.method,
            t=-b_comb,
            simplify=self.simplify,
            complete=True,
            log=True,
            **self._deriv_kwargs
        )
        log_pred_vals = log_c_vals + log_abs_num - self.log_abs

        if individual:
            if scalar_input:
                if log:
                    return float(log_pred_vals[0])
                else:
                    sign_pred = sign_num[0] * (self._sign if self._sign is not None else 1)
                    return 0.0 if log_pred_vals[0] == -np.inf else sign_pred * np.exp(log_pred_vals[0])
            else:
                if log:
                    return log_pred_vals
                else:
                    sign_pred = sign_num * (self._sign if self._sign is not None else 1)
                    result = sign_pred * np.exp(log_pred_vals)
                    result[log_pred_vals == -np.inf] = 0.0
                    return result
        else:
            log_pred_joint = np.sum(log_pred_vals)
            if log:
                return log_pred_joint
            else:
                sign_prod = np.prod(sign_num) if not scalar_input else sign_num[0]
                sign_prod *= (self._sign if self._sign is not None else 1)
                return 0.0 if log_pred_joint == -np.inf else sign_prod * np.exp(log_pred_joint)

    # ========================================================
    # POSTERIOR MGF
    # ========================================================
    def post_mgf(self, r_val, log=True):
        """
        Compute the posterior moment-generating function (MGF) at given r.
        Supports scalar, array‑like, and symbolic `r_val`.

        Parameters
        ----------
        r_val : scalar, array-like, or sympy.Symbol
            Evaluation point(s). If array-like, must be convertible to NumPy array.
        log : bool, optional
            If True, return log-MGF; else ordinary MGF.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            If r_val is scalar numeric: returns scalar.
            If r_val is array-like: returns array.
            If r_val is symbolic or has free symbols: returns sympy.Expr.
        """
        # ---- Symbolic path (if derivative is symbolic) ----
        if self._deriv_is_symbolic:
            try:
                # Build symbolic expression using the canonical `r`
                num_expr = self._deriv.subs(t, r - self.b)
                denom_expr = self._deriv.subs(t, -self.b)
                log_ratio = sp.log(num_expr) - sp.log(denom_expr)

                # Substitute known numeric parameters
                if self.params is not None:
                    log_ratio = log_ratio.subs(
                        {sym: self.params[sym.name]
                        for sym in log_ratio.free_symbols
                        if sym.name in self.params}
                    )

                # ---- Handle different input types for r_val ----

                # Case 1: r_val is a SymPy symbol
                if isinstance(r_val, sp.Symbol):
                    if log_ratio.free_symbols:
                        return log_ratio if log else sp.exp(log_ratio)
                    val = float(log_ratio.evalf())
                    return val if log else np.exp(val)

                # Case 2: r_val is array-like (numeric)
                if hasattr(r_val, '__len__') and not isinstance(r_val, (str, bytes)):
                    free_after_params = log_ratio.free_symbols - {r}
                    if free_after_params:
                        raise RuntimeError(
                            "Cannot evaluate MGF numerically for array `r` because "
                            "hyperparameters are symbolic. Use numeric hyperparameters."
                        )

                    # Convert to flat array
                    r_arr = np.asarray(r_val).ravel()
                    scalar_input = len(r_arr) == 1
                    orig_shape = np.shape(r_val)

                    # Pre-allocate results (as objects to allow mixed types)
                    results = [None] * len(r_arr)

                    # ---- Loop over each r value (scalar evaluation) ----
                    for idx, ri in enumerate(r_arr):
                        expr_i = log_ratio.subs(r, ri)
                        if expr_i.free_symbols:
                            # If still symbolic, keep the expression
                            results[idx] = expr_i
                        else:
                            results[idx] = float(expr_i.evalf())

                    # ---- Determine return type ----
                    # If all results are numeric, convert to array
                    if all(isinstance(v, (float, int, np.floating)) for v in results):
                        val = np.array(results, dtype=float).reshape(orig_shape)
                        if scalar_input:
                            return float(val.item()) if log else np.exp(float(val.item()))
                        else:
                            return val if log else np.exp(val)
                    else:
                        # Mixed or all symbolic: return object array or list
                        # For scalar input, return the expression itself
                        if scalar_input:
                            return results[0] if log else sp.exp(results[0])
                        else:
                            # If log=False, exponentiate symbolic expressions
                            if log:
                                out = np.array(results, dtype=object).reshape(orig_shape)
                                return out
                            else:
                                out = [sp.exp(res) if isinstance(res, sp.Expr) else np.exp(res) for res in results]
                                return np.array(out, dtype=object).reshape(orig_shape)

                # Case 3: r_val is scalar numeric (int, float, or None)
                if r_val is not None:
                    log_ratio = log_ratio.subs(r, r_val)

                if log_ratio.free_symbols:
                    return log_ratio if log else sp.exp(log_ratio)

                val = float(log_ratio.evalf())
                return val if log else np.exp(val)

            except Exception as e:
                raise RuntimeError(f"Symbolic computation failed: {e}") from e

        # ---- Numeric path (if derivative is not symbolic) ----
        if r_val is None:
            raise ValueError("For numeric evaluation, r must be provided.")

        r_arr = np.asarray(r_val)
        scalar_input = r_arr.ndim == 0
        if scalar_input:
            r_arr = np.array([r_val])

        log_abs_num, sign_num = mgfDerivative(
            order=self.a,
            prior=self.prior,
            method=self.method,
            t=r_arr - self.b,
            u=None,
            simplify=self.simplify,
            complete=True,
            log=True,
            **self._deriv_kwargs
        )

        log_ratio = log_abs_num - self.log_abs
        sign_ratio = sign_num * (self._sign if self._sign is not None else 1)

        if scalar_input:
            if log:
                return float(log_ratio[0])
            else:
                if log_ratio[0] == -np.inf:
                    return 0.0
                return float(sign_ratio[0] * np.exp(log_ratio[0]))
        else:
            if log:
                return log_ratio
            else:
                result = sign_ratio * np.exp(log_ratio)
                result[log_ratio == -np.inf] = 0.0
                return result

    # ========================================================
    # POSTERIOR RAW MOMENT
    # ========================================================
    def post_raw_moment(self, q, numerator_method='auto', log=True):
        """
        Compute the posterior moment of order q.

        Parameters
        ----------
        q : scalar or array-like
            Moment order(s). If array-like, returns array of results.
        numerator_method : str, optional
            Method for derivative computation.
        log : bool, optional
            If True, return log of the moments; otherwise ordinary values.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            If q is scalar numeric: scalar.
            If q is array-like: np.ndarray.
            If q is symbolic and scalar: sympy.Expr.
        """
        # ---- Determine if q is array-like ----
        if hasattr(q, '__len__') and not isinstance(q, (str, bytes, sp.Basic)):
            q_arr = np.asarray(q)
            is_array = True
        else:
            is_array = False

        # ---- Warn if any high-order (not in {1,2,3,4}) ----
        if is_array:
            if any(qi not in (1, 2, 3, 4) for qi in q_arr):
                import warnings
                warnings.warn("computing high-order posterior moments can be very slow", RuntimeWarning)
        else:
            if q not in (1, 2, 3, 4):
                import warnings
                warnings.warn("computing high-order posterior moments can be very slow", RuntimeWarning)

        # ---- Symbolic path ----
        if self._is_symbolic:
            try:
                if is_array:
                    # Compute symbolic derivatives for all orders at once
                    deriv_exprs = mgfDerivative(
                        order=q_arr,
                        prior=self.prior,
                        method=numerator_method,
                        t=None,
                        simplify=self.simplify,
                        log=False,
                        complete=True
                    )
                    # deriv_exprs is a list of sympy.Expr (one per order)
                    log_ratios = []
                    for deriv_expr in deriv_exprs:
                        num_expr = deriv_expr.subs(t, -self.b)
                        denom_expr = self._evaluate_derivative(-self.b)
                        log_ratio = sp.log(num_expr) - sp.log(denom_expr)
                        if self.params is not None:
                            log_ratio = log_ratio.subs(
                                {sym: self.params[sym.name]
                                for sym in log_ratio.free_symbols
                                if sym.name in self.params}
                            )
                        log_ratios.append(log_ratio)

                    # Check if any free symbols remain
                    if any(r.free_symbols for r in log_ratios):
                        # Return list of expressions
                        return log_ratios if log else [sp.exp(r) for r in log_ratios]
                    else:
                        # All numeric
                        vals = [float(r.evalf()) for r in log_ratios]
                        return np.array(vals) if log else np.exp(vals)
                else:
                    # Scalar q
                    order = self.a + q
                    deriv_expr = mgfDerivative(
                        order=order,
                        prior=self.prior,
                        method=numerator_method,
                        t=None,
                        simplify=self.simplify,
                        log=False,
                        complete=True
                    )
                    num_expr = deriv_expr.subs(t, -self.b)
                    denom_expr = self._evaluate_derivative(-self.b)
                    log_ratio = sp.log(num_expr) - sp.log(denom_expr)
                    if self.params is not None:
                        log_ratio = log_ratio.subs(
                            {sym: self.params[sym.name]
                            for sym in log_ratio.free_symbols
                            if sym.name in self.params}
                        )
                    if log_ratio.free_symbols:
                        return log_ratio if log else sp.exp(log_ratio)
                    val = float(log_ratio.evalf())
                    return val if log else np.exp(val)

            except Exception as e:
                raise RuntimeError(f"Symbolic computation failed: {e}. Falling back to numeric.") from e

        # ---- Numeric path (vectorized) ----
        if is_array:
            orders = self.a + q_arr
        else:
            orders = self.a + q

        log_abs_num, sign_num = mgfDerivative(
            order=orders,
            prior=self.prior,
            method=numerator_method,
            t=-self.b,
            simplify=self.simplify,
            log=True,
            complete=True,
            **self._deriv_kwargs
        )

        log_ratio = log_abs_num - self.log_abs
        sign_ratio = sign_num * (self._sign if self._sign is not None else 1)

        if log:
            if is_array:
                return log_ratio
            else:
                return float(log_ratio)
        else:
            result = sign_ratio * np.exp(log_ratio)
            if is_array:
                result[log_ratio == -np.inf] = 0.0
                return result
            else:
                if log_ratio == -np.inf:
                    return 0.0
                return float(result)
    
    # ========================================================
    # POSTERIOR CENTRAL MOMENT
    # ======================================================== 
    def post_central_moment(self, order=None, log=True, numerator_method='auto'):
        """
        Compute central moments of order(s) 1, 2, 3, or 4.

        Parameters
        ----------
        order : int or list of ints, optional
            Central moment order(s). If None (default), computes all four (1,2,3,4).
            If an integer, returns a single result.
        log : bool, optional
            If True, return (log_abs, sign) for each central moment.
            If False, return the ordinary central moment (float or sympy.Expr).
        numerator_method : str, optional
            Method for computing the numerator derivative in raw moments.
            Passed to post_raw_moment.

        Returns
        -------
        If order is an integer:
            - If log=True: (log_abs, sign) where log_abs is float or sympy.Expr,
            sign is int or sympy.Expr.
            - If log=False: float or sympy.Expr (ordinary central moment).
        If order is None or a list:
            - A dictionary {order: result} where each result is as above.
        """
        # Determine which orders to compute
        if order is None:
            orders = [1, 2, 3, 4]
            single_order = False
        elif isinstance(order, int):
            orders = [order]
            single_order = True
        else:
            # Assume iterable of ints
            orders = list(order)
            single_order = False

        # Validate orders
        for o in orders:
            if o not in {1, 2, 3, 4}:
                raise ValueError(f"Order {o} is not supported. Must be 1, 2, 3, or 4.")

        # ---- Fetch all needed raw moments in one vectorized call ----
        max_order = max(orders)
        # We need raw moments up to max_order (including 0)
        q_all = list(range(0, max_order + 1))   # e.g., [0,1,2,3,4]
        raw_all = self.post_raw_moment(q_all, log=False, numerator_method=numerator_method)
        # raw_all is an array or list of raw moments for orders 0..max_order.
        # Ensure it's a list for indexing.
        if not isinstance(raw_all, (list, np.ndarray)):
            # If scalar? But q_all is array, so raw_all should be array.
            raw_all = [raw_all]
        raw = {i: raw_all[i] for i in range(max_order + 1)}

        # ---- Compute central moments for the requested orders ----
        results = {}
        for o in orders:
            # μ_o = Σ_{j=0}^o C(o, j) * μ'_j * (-μ_1)^{o-j}
            central = 0
            for j in range(0, o + 1):
                coeff = math.comb(o, j)
                term = coeff * raw[j] * ((-raw[1]) ** (o - j))
                central += term

            # For order 1, central moment is always 0
            if o == 1:
                central = 0

            # ---- Handle log vs ordinary for this order ----
            if log:
                if isinstance(central, (int, float)):
                    if central == 0:
                        result = (-float('inf'), 1)
                    else:
                        result = (np.log(abs(central)), 1 if central > 0 else -1)
                elif isinstance(central, sp.Expr):
                    if not central.free_symbols:
                        val = float(central.evalf())
                        if val == 0:
                            result = (-float('inf'), 1)
                        else:
                            result = (np.log(abs(val)), 1 if val > 0 else -1)
                    else:
                        result = (sp.log(sp.Abs(central)), sp.sign(central))
                else:
                    raise TypeError(f"Unexpected type for central moment: {type(central)}")
            else:
                result = central

            results[o] = result

        # ---- Return ----
        if single_order:
            return results[orders[0]]
        else:
            return results
     
      
    # ========================================================
    # POSTERIOR QUANTILE
    # ========================================================        
    # Inside the MGFDerivative class
    def post_quantile(
        self,
        p: float | np.ndarray,
        root_method: str = "auto",
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        maxiter: int = 50,
        tol: float = 1e-8,
        rel_tol: float = 1e-8,
        **kwargs
    ) -> np.ndarray | float:
        """
        Compute quantiles (inverse CDF) for given probabilities.

        Parameters
        ----------
        p : float or array-like
            Probabilities (must be in (0, 1)).
        root_method : str, optional
            Root-finding method. See solve_root() for options.
        lower, upper : array-like, optional
            Search interval bounds. If None, automatically expanded.
        x0 : array-like, optional
            Initial guess for Newton methods. If None, uses midpoint.
        maxiter, tol, rel_tol : passed to solve_root.
        **kwargs : additional arguments for solve_root.

        Returns
        -------
        scalar or np.ndarray
            Quantiles matching the shape of p.
        """

        p_arr = np.asarray(p)
        scalar_input = p_arr.ndim == 0
        if scalar_input:
            p_arr = np.array([p])

        if np.any((p_arr <= 0) | (p_arr >= 1)):
            raise ValueError("Probabilities must be strictly between 0 and 1.")

        def f(x):
            return self.post_cdf(x, log=False) - p_arr

        def df(x):
            return self.post_density(x, log=False)

        if lower is None or upper is None:
            lower_init = np.full_like(p_arr, 1e-6)
            upper_init = np.full_like(p_arr, 1.0 / 1e-6)
            lower, upper = self._expand_bracket(p_arr, f, lower_init, upper_init)

        if x0 is None:
            x0 = (lower + upper) / 2.0

        roots = solve_root(
            f=f,
            df=df,
            x0=x0,
            lower=lower,
            upper=upper,
            root_method=root_method,
            maxiter=maxiter,
            tol=tol,
            rel_tol=rel_tol,
            **kwargs
        )

        return float(roots[0]) if scalar_input else roots

    def _expand_bracket(self, p, f, lower, upper, max_expand=10):
        """
        Expand lower/upper arrays until f(lower) < 0 and f(upper) > 0 for all elements.
        Returns valid lower, upper arrays.
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        for _ in range(max_expand):
            f_low = f(lower)
            f_up = f(upper)
            need_lower = f_low >= 0
            need_upper = f_up <= 0
            lower = np.where(need_lower, lower * 0.5, lower)
            upper = np.where(need_upper, upper * 2.0, upper)
            if not np.any(need_lower | need_upper):
                break
        if np.any(f(lower) >= 0) or np.any(f(upper) <= 0):
            raise RuntimeError("Could not find valid brackets; provide explicit lower/upper.")
        return lower, upper
    
    # ========================================================
    # POSTERIOR SAMPLE
    # ========================================================     
    def post_sample(self, n: int | None = None, u: np.ndarray | None = None, root_method: str = "auto", **kwargs) -> np.ndarray:
        """
        Generate posterior samples using inverse transform sampling.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate. Ignored if u is provided.
        u : array-like, optional
            Uniform random numbers in (0,1). If provided, n is ignored.
        root_method : str, optional
            Passed to post_quantile.
        **kwargs : additional arguments passed to post_quantile (e.g., maxiter, tol).

        Returns
        -------
        np.ndarray
            Samples from the posterior distribution.
        """
        if u is None:
            if n is None:
                raise ValueError("Either n or u must be provided.")
            u = np.random.rand(n)
        else:
            u = np.asarray(u)
            if np.any((u <= 0) | (u >= 1)):
                raise ValueError("All uniform variates must be in (0, 1).")

        result = self.post_quantile(u, root_method=root_method, **kwargs)
        return np.asarray(result)
    
    # ========================================================
    # CENTRAL POSTERIOR CREDIBLE INTERVAL
    # ========================================================     
    def post_interval(
        self,
        level: float | np.ndarray = 0.95,
        root_method: str = "auto",
        **kwargs
    ) -> tuple | np.ndarray:
        """
        Compute central credible intervals (equal-tailed) for the posterior.

        Parameters
        ----------
        level : float or array-like, optional (default 0.95)
            Credible level(s). Must be in (0, 1).
        root_method : str, optional
            Root-finding method passed to post_quantile.
        **kwargs : additional arguments passed to post_quantile (e.g., maxiter, tol).

        Returns
        -------
        If level is scalar:
            (lower, upper) where lower and upper are floats.
        If level is array-like:
            np.ndarray of shape (len(level), 2) where each row is [lower, upper].
        """
        levels = np.asarray(level)
        scalar_input = levels.ndim == 0
        if scalar_input:
            levels = np.array([level])

        if np.any((levels <= 0) | (levels >= 1)):
            raise ValueError("Credible levels must be in (0, 1).")

        # Lower and upper probabilities for each level
        p_lower = (1 - levels) / 2.0
        p_upper = (1 + levels) / 2.0

        # Compute quantiles for all levels simultaneously
        p_all = np.concatenate([p_lower, p_upper])
        quantiles = self.post_quantile(p_all, root_method=root_method, **kwargs)

        # Split into lower and upper
        n = len(levels)
        lower = quantiles[:n]
        upper = quantiles[n:]

        if scalar_input:
            return float(lower[0]), float(upper[0])
        else:
            return np.column_stack([lower, upper])

    # ========================================================
    # SEQUENTIAL UPDATING
    # ========================================================
    def to_prior_object(self):
        """
        Convert current posterior into a mitMGFprior object.
        Tries to construct a symbolic prior first if possible.
        """

        if self._deriv_is_symbolic:
            try:

                # Get symbolic expressions from post_mgf and post_density
                mgf_sym_expr = self.post_mgf(r, log=False)
                mgf_sym_expr = mgf_sym_expr.subs(r, t) # posterior MGF of r becomes the prior MGF of t
                pdf_sym_expr = self.post_density(theta, log=False)   # use canonical theta

                # Ensure they are SymPy expressions
                if isinstance(mgf_sym_expr, sp.Expr) and isinstance(pdf_sym_expr, sp.Expr):
                    return mitMGFprior(
                        name="posterior_prior_symbolic",
                        mgf_sym=mgf_sym_expr,
                        pdf_sym=pdf_sym_expr,
                        params=self.params
                    ).as_mitMGFprior()
            except Exception as e:
                print("Symbolic construction failed:")
                import traceback
                traceback.print_exc()
                pass

        # ---- Backend (numeric) route ----
        def mgf_backend(t_val, xp=np, **params):
            return self.post_mgf(t_val, log=self.log)

        def pdf_backend(theta_val, xp=np, **params):
            return self.post_density(theta_val, log=self.log)

        return mitMGFprior(
            name="posterior_prior",
            mgf_backend=mgf_backend,
            pdf_backend=pdf_backend,
            params=self.params
        ).as_mitMGFprior()

    def update(self, new_data, **kwargs):
        """
        Sequential update returns a new MGFDerivative,
        using posterior mitMGFprior as prior.
        """
        # Extract known arguments
        method = kwargs.pop("method", self.method)
        likelihood = kwargs.pop("likelihood", self.likelihood)
        simplify = kwargs.pop("simplify", self.simplify)
        log = kwargs.pop("log", self.log)

        # Enforce symbolic restriction
        if method == 'symbolic' and not self._is_symbolic:
            raise ValueError(
                "Cannot use symbolic method for sequential update when the posterior derivative is numeric. "
                "The posterior prior is numeric and cannot be used symbolically. Choose a numeric method (jax, bell, scipy, mpmath)."
            )

        post_prior = self.to_prior_object()
        return MGFDerivative(
            prior=post_prior,
            data=new_data,
            likelihood=likelihood,
            method=method,
            simplify=simplify,
            log=log, # new object's requested state
            **kwargs
        )

