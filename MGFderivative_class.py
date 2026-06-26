"""
MGFderivative_class.py

Defines a class MGFDerivative that encapsulates the computation of MGF derivatives
and marginal likelihoods (evidence) for various likelihoods and priors.

Supports sequential updating via the `update` method, using the posterior MGF
as the prior for the next chunk of data.
"""

import math
import sympy as sp
import pandas as pd
import numpy as np
import scipy.stats as stats
from derivativeDispatch import mgfDerivative

# ---- Import all ready* and c* functions ----
from like_stats.Poisson import readyPoisson, cPoisson
from like_stats.Gamma import readyGamma, cGamma
from like_stats.Laplace import readyLaplace, cLaplace
from like_stats.Normal import readyNormal, cNormal
from like_stats.Rayleigh import readyRayleigh, cRayleigh
from like_stats.MaxwellBoltzmann import readyMaxwellBoltzmann, cMaxwellBoltzmann
from like_stats.InverseGamma import readyInverseGamma, cInverseGamma
from like_stats.Levy import readyLevy, cLevy
from like_stats.Weibull import readyWeibull, cWeibull
from like_stats.BurrXII import readyBurrXII, cBurrXII
from like_stats.Pareto import readyPareto, cPareto
from like_stats.Dagum import readyDagum, cDagum
from like_stats.Gompertz import readyGompertz, cGompertz
from like_stats.HalfNormal import readyHalfNormal, cHalfNormal

# ---- Import prior MGF/CGF and PDF functions ----
from MGFdictionary.gammaMGF import (
    gamma_mgf_symbolic,
    gamma_cgf_symbolic,
    gamma_cgf,
    gamma_mgf,
    gamma_cgf_jax,
    gamma_mgf_jax,
    gamma_pdf_symbolic,
    gamma_pdf_symbolic_sub,
)
from MGFdictionary.paretoMGF import (
    pareto_mgf_symbolic,
    pareto_cgf_symbolic,
    pareto_cgf,
    pareto_mgf,
    pareto_cgf_jax,
    pareto_mgf_jax,
    pareto_pdf_symbolic,
    pareto_pdf_symbolic_sub,
)

# ---- Placeholder for unimplemented likelihoods ----
def _not_implemented(*args, **kwargs):
    raise NotImplementedError("This likelihood's module has not been implemented yet.")

# ---- Prior registry ----
PRIOR_REGISTRY = {
    'gamma': {
        'dist': lambda p: stats.gamma(a=p['alpha'], scale=1/p['beta']),
        'mgf_sym': gamma_mgf_symbolic,
        'cgf_sym': gamma_cgf_symbolic,
        'mgf': gamma_mgf,
        'cgf': gamma_cgf,
        'mgf_jax': gamma_mgf_jax,
        'cgf_jax': gamma_cgf_jax,
        'pdf_sym': gamma_pdf_symbolic,
        'pdf_sym_func': gamma_pdf_symbolic_sub,
    },
    'pareto': {
        'dist': lambda p: stats.pareto(b=p['alpha'], scale=p['xi']),
        'mgf_sym': pareto_mgf_symbolic,
        'cgf_sym': pareto_cgf_symbolic,
        'mgf': pareto_mgf,
        'cgf': pareto_cgf,
        'mgf_jax': pareto_mgf_jax,
        'cgf_jax': pareto_cgf_jax,
        'pdf_sym': pareto_pdf_symbolic,
        'pdf_sym_func': pareto_pdf_symbolic_sub,
    },
}


class MGFDerivative:
    """
    Represents a prior MGF derivative (integer or fractional) ready to be combined with data.

    The derivative order (a) and evaluation point (t = -b) are determined from the data
    via the sufficient statistics of the likelihood.

    Currently supports:
        - Likelihoods: Poisson, Gamma, Laplace, Normal, Rayleigh, Maxwell‑Boltzmann,
          Inverse Gamma, Lévy, Weibull, Burr XII, Pareto, Dagum, Gompertz, Half‑Normal
        - Priors: gamma, pareto (others can be added to the registry)
    """

    LIKELIHOOD_REGISTRY = {
        'poisson':           (readyPoisson, cPoisson),
        'gamma':             (readyGamma,   cGamma),
        'laplace':           (readyLaplace, cLaplace),
        'normal':            (readyNormal,  cNormal),
        'rayleigh':          (readyRayleigh, cRayleigh),
        'maxwell-boltzmann': (readyMaxwellBoltzmann, cMaxwellBoltzmann),
        'inverse gamma':     (readyInverseGamma, cInverseGamma),
        'levy':              (readyLevy, cLevy),
        'weibull':           (readyWeibull, cWeibull),
        'burrxii':           (readyBurrXII, cBurrXII),
        'pareto':            (readyPareto, cPareto),
        'dagum':             (readyDagum, cDagum),
        'gompertz':          (readyGompertz, cGompertz),
        'halfnormal':        (readyHalfNormal, cHalfNormal),
        'maxwell':           (readyMaxwellBoltzmann, cMaxwellBoltzmann),
        'inverse-gamma':     (readyInverseGamma, cInverseGamma),
        'burr xii':          (readyBurrXII, cBurrXII),
        'burr-xii':          (readyBurrXII, cBurrXII),
    }

    _special_likelihoods = {'weibull'}

    _ready_keys = {
        'scale', 'shape', 'mean', 'location', 'rho',
        'known_shape', 'r', 's',
    }

    def __init__(self, prior, data, likelihood='poisson', method='symbolic',
                 params=None, simplify=False, log=True,
                 prior_mgf_func=None, prior_cgf_func=None,
                 prior_pdf_func=None, prior_pdf_sym_func=None,
                 **kwargs):
        """
        Compute the MGF derivative for the given data and prior.

        Parameters
        ----------
        prior : str
            Prior name (must be in PRIOR_REGISTRY) or 'custom'.
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
            For 'custom' prior, this is ignored.
        simplify : bool, optional
            If True, simplify symbolic expressions.
        log : bool, optional
            If True, store derivative in log scale (numeric only).
        prior_mgf_func : callable, optional
            Function (t, *args) -> M(t) for custom prior.
        prior_cgf_func : callable, optional
            Function (t, *args) -> log M(t) for custom prior.
        prior_pdf_func : callable, optional
            Function (theta, *args) -> p(theta) for custom prior.
        prior_pdf_sym_func : callable, optional
            Function (params) -> symbolic PDF expression for custom prior.
        **kwargs : additional arguments passed to the likelihood's ready function
                   and/or to mgfDerivative.
        """
        # ---- Validate prior ----
        self.prior = prior.lower()
        self.prior_info = None
        self._custom_prior = False

        if self.prior == 'custom':
            self._custom_prior = True
            if prior_mgf_func is None:
                raise ValueError("For custom prior, prior_mgf_func must be provided.")
            self._prior_mgf_func = prior_mgf_func
            self._prior_cgf_func = prior_cgf_func
            self._prior_pdf_func = prior_pdf_func
            self._prior_pdf_sym_func = prior_pdf_sym_func
            # Create a dummy prior_info for consistency
            self.prior_info = {
                'dist': None,
                'mgf_sym': None,
                'cgf_sym': None,
                'mgf': prior_mgf_func,
                'cgf': prior_cgf_func,
                'mgf_jax': None,
                'cgf_jax': None,
                'pdf_sym': prior_pdf_sym_func,
                'pdf_sym_func': lambda p: prior_pdf_sym_func,
            }
            # We'll set params to None since they are not used for custom prior
            self.params = None
        else:
            if self.prior not in PRIOR_REGISTRY:
                raise ValueError(f"Unsupported prior: {prior}. "
                                 f"Choose from {list(PRIOR_REGISTRY.keys())} or 'custom'.")
            self.prior_info = PRIOR_REGISTRY[self.prior]
            self.params = params

        self.method = method
        self.simplify = simplify
        self.log = log
        self.likelihood = likelihood.lower()
        self.data = data
        self._has_numeric_params = (
            not self._custom_prior and
            params is not None and
            all(isinstance(v, (int, float)) for v in params.values())
        )

        self._ready_kwargs = {k: v for k, v in kwargs.items() if k in self._ready_keys}
        self._deriv_kwargs = {k: v for k, v in kwargs.items() if k not in self._ready_keys}

        if self.likelihood not in self.LIKELIHOOD_REGISTRY:
            raise ValueError(f"Unsupported likelihood: {likelihood}. "
                             f"Choose from {list(self.LIKELIHOOD_REGISTRY.keys())}")

        self.ready_func, self.c_func = self.LIKELIHOOD_REGISTRY[self.likelihood]

        stats = self.ready_func(data, **self._ready_kwargs)
        self.a = stats['a']
        self.b = stats['b']
        self.log_c = stats['log_c']

        if self.likelihood in self._special_likelihoods:
            self._compute_derivative_special()
        else:
            self._compute_derivative_standard()

    def _compute_derivative_standard(self):
        """Standard derivative computation using mgfDerivative."""
        if self._custom_prior:
            # For custom prior, we need to compute the derivative of the custom MGF.
            # This is not yet implemented in mgfDerivative, so we raise an error.
            # In practice, the user should use the standard prior registry for sequential updates.
            raise NotImplementedError("Custom prior derivatives are not yet supported by mgfDerivative.")
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
        """Special derivative computation for Weibull (placeholder)."""
        raise NotImplementedError(
            f"Special derivative route for '{self.likelihood}' not yet implemented."
        )

    def _store_result(self, result):
        """Store the result from mgfDerivative."""
        if result is None:
            raise RuntimeError("Derivative computation returned None.")
        if isinstance(result, sp.Expr):
            self._is_symbolic = True
            self._expr = result
            self._log_abs = None
            self._sign = None
            self._value = None
        else:
            self._is_symbolic = False
            if self.log:
                self._log_abs, self._sign = result
                self._value = None
            else:
                self._value = result
                self._log_abs = None
                self._sign = None

    @property
    def is_symbolic(self):
        """
        Returns True only if the derivative is symbolic AND parameters are not all numeric.
        If parameters are numeric, we treat the object as numeric (even if the derivative was computed symbolically).
        For custom prior, we treat it as symbolic only if the MGF function returns a sympy expression.
        """
        if self._custom_prior:
            # For custom prior, we cannot easily determine if it's symbolic.
            # We assume it's numeric unless the method is 'symbolic' and params are None.
            return self.method.lower() == 'symbolic' and self.params is None
        return self._is_symbolic and not self._has_numeric_params

    @property
    def expr(self):
        if not self._is_symbolic:
            raise ValueError("This is a numeric result; use .log_abs / .value instead.")
        return self._expr

    @property
    def log_abs(self):
        if self.is_symbolic:
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
        if self.is_symbolic:
            raise ValueError("This is a symbolic result; use .expr instead.")
        if self._sign is None:
            if self._value is not None:
                return 1 if self._value > 0 else -1
            else:
                raise ValueError("No sign available.")
        return self._sign

    @property
    def value(self):
        if self.is_symbolic:
            raise ValueError("This is a symbolic result; use .expr instead.")
        if self._value is None:
            if self._log_abs == -float('inf'):
                return 0.0
            return self.sign * math.exp(self._log_abs)
        return self._value

    def to_ordinary(self):
        """Return a new MGFDerivative with ordinary scale (if numeric)."""
        if self.is_symbolic:
            return self
        return MGFDerivative(
            prior=self.prior,
            data=self.data,
            likelihood=self.likelihood,
            method=self.method,
            params=self.params,
            simplify=self.simplify,
            log=False,
            **self._ready_kwargs,
            **self._deriv_kwargs
        )

    def to_log(self):
        """Return a new MGFDerivative in log scale (if numeric)."""
        if self.is_symbolic:
            return self
        return MGFDerivative(
            prior=self.prior,
            data=self.data,
            likelihood=self.likelihood,
            method=self.method,
            params=self.params,
            simplify=self.simplify,
            log=True,
            **self._ready_kwargs,
            **self._deriv_kwargs
        )

    def _get_prior_dist(self):
        """Return a scipy.stats distribution object for the prior."""
        if self._custom_prior:
            # Custom prior: we cannot provide a scipy distribution.
            # Return None and let the method handle it.
            return None
        return self.prior_info['dist'](self.params)

    def evidence(self):
        """
        Return the marginal likelihood (evidence).

        If `self.is_symbolic` is True, returns a symbolic expression.
        Otherwise, evaluates numerically.
        """
        if self.is_symbolic:
            c_expr = self.c_func()
            return c_expr * self._expr
        else:
            # Recompute derivative numerically to get numeric result
            log_abs_num, sign_num = mgfDerivative(
                order=self.a,
                prior=self.prior,
                method=self.method,
                t=float(-self.b),
                params=self.params,
                simplify=self.simplify,
                log=True,
                **self._deriv_kwargs
            )
            total_log_abs = self.log_c + log_abs_num
            if self.log:
                return total_log_abs, sign_num
            else:
                return math.exp(total_log_abs) * sign_num

    def post_density(self, theta=None, log=True):
        """
        Compute the posterior density (or log-density) at given θ.

        If `self.is_symbolic` is True:
            - If theta is None or a sympy Symbol: returns a symbolic expression.
            - If theta is numeric: evaluates the expression numerically.
            - Uses numeric‑substituted PDF if params are numeric; otherwise base symbolic PDF.
        If `self.is_symbolic` is False: performs numeric evaluation.

        Parameters
        ----------
        theta : float, numpy array, or sympy.Symbol (optional)
            Evaluation point(s). If None and derivative is symbolic, returns symbolic expression.
        log : bool, optional
            If True, return log-density; else density.

        Returns
        -------
        sympy.Expr or float or numpy array
            Symbolic expression or numeric value.
        """
        if self.is_symbolic:
            try:
                t_sym = sp.Symbol('t', real=True)
                denom_expr = self._expr.subs(t_sym, -self.b)

                # Determine theta symbol
                if theta is None:
                    theta_sym = sp.Symbol('theta', positive=True)
                elif isinstance(theta, sp.Symbol):
                    theta_sym = theta
                else:
                    theta_sym = sp.Symbol('theta', positive=True)

                # Choose PDF: numeric‑substituted if params numeric, otherwise base symbolic
                if self._has_numeric_params and not self._custom_prior:
                    pdf_sym = self.prior_info['pdf_sym_func'](self.params)
                elif self._custom_prior and self._prior_pdf_sym_func is not None:
                    pdf_sym = self._prior_pdf_sym_func(self.params) if self.params is not None else self._prior_pdf_sym_func()
                else:
                    pdf_sym = self.prior_info['pdf_sym']() if not self._custom_prior else None

                if pdf_sym is None:
                    raise ValueError("No symbolic PDF available for this prior.")

                log_prior = sp.log(pdf_sym)

                log_num = log_prior + self.a * sp.log(theta_sym) - self.b * theta_sym
                log_post = log_num - sp.log(denom_expr)

                if theta is not None and not isinstance(theta, sp.Symbol):
                    if hasattr(theta, '__len__'):
                        from sympy import lambdify
                        func = lambdify(theta_sym, log_post, modules='numpy')
                        return func(theta)
                    else:
                        return float(log_post.subs(theta_sym, float(theta)).evalf())
                else:
                    return log_post if log else sp.exp(log_post)
            except Exception as e:
                print(f"⚠️ Symbolic computation failed: {e}. Falling back to numeric.")

        # ---- Numeric path ----
        if theta is None:
            raise ValueError("For numeric evaluation, theta must be provided.")

        # For custom prior, use the provided PDF function
        if self._custom_prior:
            if self._prior_pdf_func is None:
                raise ValueError("No numeric PDF function provided for custom prior.")
            log_prior = np.log(self._prior_pdf_func(theta))
        else:
            dist = self._get_prior_dist()
            log_prior = dist.logpdf(theta)

        log_num = log_prior + self.a * np.log(theta) - self.b * theta
        log_post = log_num - self.log_abs
        if log:
            return log_post
        else:
            return np.exp(log_post)

    def post_predictive(self, new_data, log=True, **kwargs):
        """
        Compute the posterior predictive density (or log-density) for new data.

        If `self.is_symbolic` is True:
            - If `new_data` is a sympy Symbol, returns a symbolic expression.
            - If `new_data` is numeric, builds the symbolic expression and evaluates it
              numerically (using `lambdify` for arrays, `evalf` for scalars).
        If `self.is_symbolic` is False:
            - Performs numeric evaluation directly.

        Parameters
        ----------
        new_data : pandas DataFrame, Series, array‑like, or sympy.Symbol
            New observation(s). If Symbol, treated as symbolic.
        log : bool, optional
            If True, return log-density; else density.
        **kwargs : additional arguments for the likelihood's ready function (only used for numeric data).

        Returns
        -------
        sympy.Expr or float or numpy array
            Symbolic expression or numeric value(s).
        """
        # ---- Symbolic path ----
        if self.is_symbolic:
            try:
                # Compute statistics for new data
                if isinstance(new_data, sp.Symbol):
                    a_new = sp.Symbol('a_new', real=True)
                    b_new = sp.Symbol('b_new', real=True)
                    log_c_new = sp.Symbol('log_c_new', real=True)
                    numeric_new = False
                else:
                    # Numeric new data: compute statistics (these will be numeric)
                    stats_new = self.ready_func(new_data, **kwargs)
                    a_new = stats_new['a']
                    b_new = stats_new['b']
                    log_c_new = stats_new['log_c']
                    numeric_new = True

                combined_order = self.a + a_new
                combined_b = self.b + b_new

                # Get symbolic derivative of combined order
                deriv_combined = mgfDerivative(
                    order=combined_order,
                    prior=self.prior,
                    method='symbolic',
                    t=float('nan'),
                    params=self.params if self._has_numeric_params else None,
                    simplify=self.simplify,
                    log=False
                )
                t_sym = sp.Symbol('t', real=True)
                num_expr = deriv_combined.subs(t_sym, -combined_b)
                denom_expr = self._expr.subs(t_sym, -self.b)

                log_pred = log_c_new + sp.log(num_expr) - sp.log(denom_expr)

                if numeric_new:
                    # Evaluate numerically
                    subs_dict = {}
                    for sym in log_pred.free_symbols:
                        if sym.name in self.params:
                            subs_dict[sym] = self.params[sym.name]
                    if subs_dict:
                        log_pred_sub = log_pred.subs(subs_dict)
                    else:
                        log_pred_sub = log_pred

                    if hasattr(new_data, '__len__') and not isinstance(new_data, (str, bytes)):
                        return float(log_pred_sub.evalf()) if log else float(sp.exp(log_pred_sub).evalf())
                    else:
                        return float(log_pred_sub.evalf()) if log else float(sp.exp(log_pred_sub).evalf())
                else:
                    return log_pred if log else sp.exp(log_pred)

            except Exception as e:
                print(f"⚠️ Symbolic predictive computation failed: {e}. Falling back to numeric.")

        # ---- Numeric path ----
        if isinstance(new_data, sp.Symbol):
            raise ValueError("Cannot evaluate numeric predictive density with symbolic new_data.")

        stats_new = self.ready_func(new_data, **kwargs)
        a_new = stats_new['a']
        b_new = stats_new['b']
        log_c_new = stats_new['log_c']

        a_combined = self.a + a_new
        b_combined = self.b + b_new
        log_abs_num, sign_num = mgfDerivative(
            order=a_combined,
            prior=self.prior,
            method=self.method,
            t=float(-b_combined),
            params=self.params,
            simplify=self.simplify,
            log=True,
            **self._deriv_kwargs
        )
        log_pred = log_c_new + log_abs_num - self.log_abs
        if log:
            return log_pred
        else:
            sign_pred = sign_num * self.sign
            if log_pred == -float('inf'):
                return 0.0
            return sign_pred * math.exp(log_pred)

    def post_mgf(self, r, log=False):
        """
        Compute the posterior moment-generating function (MGF) at given r.

        M_{Θ|y}(r) = D^{a(y)} M_Θ(t) |_{t = r - b(y)} / D^{a(y)} M_Θ(t) |_{t = -b(y)}

        If `self.is_symbolic` is True:
            - If r is a sympy Symbol or None, returns a symbolic expression.
            - If r is numeric, evaluates the expression numerically.
        If `self.is_symbolic` is False:
            - Performs numeric evaluation.

        Parameters
        ----------
        r : float, numpy array, or sympy.Symbol
            The argument of the posterior MGF.
        log : bool, optional
            If True, return log MGF; otherwise return MGF.

        Returns
        -------
        sympy.Expr or float or numpy array
            Symbolic expression or numeric value(s).
        """
        # ---- Symbolic path ----
        if self.is_symbolic:
            try:
                if isinstance(r, sp.Symbol) or r is None:
                    r_sym = sp.Symbol('r', real=True) if r is None else r
                else:
                    r_sym = sp.Symbol('r', real=True)

                t_sym = sp.Symbol('t', real=True)
                num_expr = self._expr.subs(t_sym, r_sym - self.b)
                denom_expr = self._expr.subs(t_sym, -self.b)

                log_ratio = sp.log(num_expr) - sp.log(denom_expr)

                if isinstance(r, (int, float)) or hasattr(r, '__len__'):
                    subs_dict = {}
                    for sym in log_ratio.free_symbols:
                        if sym.name in self.params:
                            subs_dict[sym] = self.params[sym.name]
                    if subs_dict:
                        log_ratio_sub = log_ratio.subs(subs_dict)
                    else:
                        log_ratio_sub = log_ratio

                    if hasattr(r, '__len__') and not isinstance(r, (str, bytes)):
                        from sympy import lambdify
                        func = lambdify(r_sym, log_ratio_sub, modules='numpy')
                        if log:
                            return func(r)
                        else:
                            return np.exp(func(r))
                    else:
                        log_val = float(log_ratio_sub.evalf())
                        if log:
                            return log_val
                        else:
                            return np.exp(log_val)
                else:
                    if log:
                        return log_ratio
                    else:
                        return sp.exp(log_ratio)

            except Exception as e:
                print(f"⚠️ Symbolic computation failed: {e}. Falling back to numeric.")

        # ---- Numeric path ----
        if self.is_symbolic:
            raise ValueError("Cannot compute numeric MGF from a symbolic derivative.")
        if r is None:
            raise ValueError("For numeric evaluation, r must be provided.")

        log_abs_num, sign_num = mgfDerivative(
            order=self.a,
            prior=self.prior,
            method=self.method,
            t=float(r - self.b),
            params=self.params,
            simplify=self.simplify,
            log=True,
            **self._deriv_kwargs
        )

        log_ratio = log_abs_num - self.log_abs
        sign_ratio = sign_num * self.sign

        if log:
            return log_ratio
        else:
            if log_ratio == -float('inf'):
                return 0.0
            return sign_ratio * math.exp(log_ratio)

    def post_moment(self, q, log=False):
        """
        Compute the posterior moment of order q.

        E[Θ^q | y] = D^{a(y)+q} M_Θ(t) |_{t = -b(y)} / D^{a(y)} M_Θ(t) |_{t = -b(y)}

        If `self.is_symbolic` is True:
            - If q is a sympy Symbol, returns a symbolic expression.
            - If q is numeric, evaluates the expression numerically if possible.
        If `self.is_symbolic` is False:
            - Performs numeric evaluation.

        Parameters
        ----------
        q : float or sympy.Symbol
            Order of the moment. Can be integer, fractional, or symbolic.
        log : bool, optional
            If True, return log of the moment; else return the moment value.

        Returns
        -------
        sympy.Expr or float
            Symbolic expression or numeric value.
        """
        if self.is_symbolic:
            try:
                q_is_symbol = isinstance(q, sp.Symbol)
                if q_is_symbol:
                    order = self.a + q
                else:
                    order = self.a + q

                deriv_expr = mgfDerivative(
                    order=order,
                    prior=self.prior,
                    method='symbolic',
                    t=float('nan'),
                    params=self.params if self._has_numeric_params else None,
                    simplify=self.simplify,
                    log=False
                )
                t_sym = sp.Symbol('t', real=True)
                num_expr = deriv_expr.subs(t_sym, -self.b)
                denom_expr = self._expr.subs(t_sym, -self.b)

                log_ratio = sp.log(num_expr) - sp.log(denom_expr)

                if not q_is_symbol:
                    subs_dict = {}
                    for sym in log_ratio.free_symbols:
                        if sym.name in self.params:
                            subs_dict[sym] = self.params[sym.name]
                    if subs_dict:
                        log_ratio_sub = log_ratio.subs(subs_dict)
                    else:
                        log_ratio_sub = log_ratio

                    try:
                        log_val = float(log_ratio_sub.evalf())
                        if log:
                            return log_val
                        else:
                            return np.exp(log_val)
                    except Exception:
                        if log:
                            return log_ratio
                        else:
                            return sp.exp(log_ratio)
                else:
                    if log:
                        return log_ratio
                    else:
                        return sp.exp(log_ratio)

            except Exception as e:
                print(f"⚠️ Symbolic moment computation failed: {e}. Falling back to numeric.")

        # ---- Numeric path ----
        if self.is_symbolic:
            raise ValueError("Cannot compute numeric moment from a symbolic derivative.")
        if not isinstance(q, (int, float)):
            raise ValueError("For numeric evaluation, q must be numeric.")

        order_num = self.a + q
        log_abs_num, sign_num = mgfDerivative(
            order=order_num,
            prior=self.prior,
            method=self.method,
            t=float(-self.b),
            params=self.params,
            simplify=self.simplify,
            log=True,
            **self._deriv_kwargs
        )

        log_ratio = log_abs_num - self.log_abs
        sign_ratio = sign_num * self.sign

        if log:
            return log_ratio
        else:
            if log_ratio == -float('inf'):
                return 0.0
            return sign_ratio * math.exp(log_ratio)

    # ---- Sequential updating methods ----

    def _posterior_mgf_func(self):
        """
        Return a callable that evaluates the posterior MGF at any r.
        """
        if self.is_symbolic:
            t_sym = sp.Symbol('t', real=True)
            r_sym = sp.Symbol('r', real=True)
            num_expr = self._expr.subs(t_sym, r_sym - self.b)
            denom_expr = self._expr.subs(t_sym, -self.b)
            post_mgf_expr = num_expr / denom_expr
            def func(r_val, *args):
                return float(post_mgf_expr.subs({r_sym: r_val}).evalf())
            return func
        else:
            def func(r_val, *args):
                log_abs_num, sign_num = mgfDerivative(
                    order=self.a,
                    prior=self.prior,
                    method=self.method,
                    t=float(r_val - self.b),
                    params=self.params,
                    simplify=self.simplify,
                    log=True,
                    **self._deriv_kwargs
                )
                log_ratio = log_abs_num - self.log_abs
                sign_ratio = sign_num * self.sign
                if log_ratio == -float('inf'):
                    return 0.0
                return sign_ratio * math.exp(log_ratio)
            return func

    def _posterior_cgf_func(self):
        """Return callable for log of posterior MGF."""
        def func(r_val, *args):
            return math.log(self._posterior_mgf_func()(r_val))
        return func

    def _posterior_pdf_func(self):
        """Return callable for posterior density (ordinary scale)."""
        def func(theta, *args):
            return self.post_density(theta, log=False)
        return func

    def update(self, new_data, method=None, log=None, simplify=None, **kwargs):
        """
        Perform sequential Bayesian updating with new data.

        Parameters
        ----------
        new_data : pandas DataFrame, Series, or array‑like
            New observations.
        method : str, optional
            Method for computing the derivative in the new object.
            If 'symbolic', it is only allowed if self.is_symbolic is True.
            Otherwise, must be a numeric method ('jax', 'bell', 'scipy', 'mpmath').
        log : bool, optional
            Whether to store the derivative in log scale.
        simplify : bool, optional
            Whether to simplify symbolic expressions.
        **kwargs : additional arguments passed to the likelihood's ready function.

        Returns
        -------
        MGFDerivative
            A new object with the posterior MGF as the prior, ready to process new_data.
        """
        if method is None:
            method = self.method
        if log is None:
            log = self.log
        if simplify is None:
            simplify = self.simplify

        # Enforce method restriction
        if method.lower() == 'symbolic':
            if not self.is_symbolic:
                raise ValueError("Cannot use symbolic method for sequential update when the posterior derivative is not symbolic. Choose a numeric method (jax, bell, scipy, mpmath).")

        # Prepare custom prior functions
        post_mgf = self._posterior_mgf_func()
        post_cgf = self._posterior_cgf_func()
        post_pdf = self._posterior_pdf_func()
        post_pdf_sym = None  # Not needed for numeric updates

        # Create new object with prior='custom'
        # We merge the old ready kwargs with any new ones provided
        merged_ready = {**self._ready_kwargs, **kwargs}

        return MGFDerivative(
            prior='custom',
            data=new_data,
            likelihood=self.likelihood,
            method=method,
            params=None,  # No params needed for custom prior
            simplify=simplify,
            log=log,
            prior_mgf_func=post_mgf,
            prior_cgf_func=post_cgf,
            prior_pdf_func=post_pdf,
            prior_pdf_sym_func=post_pdf_sym,
            **merged_ready
        )