"""
MGFderivative_class.py

Defines a class MGFDerivative that encapsulates the computation of MGF derivatives
and marginal likelihoods (evidence) for various likelihoods and priors.

Supports sequential updating via the `update` method, using the posterior MGF
as the prior for the next chunk of data.

Priors are represented as mitMGFprior objects.
"""

import math
import sympy as sp
import numpy as np
import pandas as pd

from jumufraktiv.derivativeDispatch import mgfDerivative
from mitMGFprior_class import mitMGFprior
from jumufraktiv.symbols import t, theta

# ============================================================
# Likelihood registry
# ============================================================
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


# ============================================================
# Likelihood registry
# ============================================================
LIKELIHOOD_REGISTRY = {
    'poisson': (readyPoisson, cPoisson),
    'gamma': (readyGamma, cGamma),
    'laplace': (readyLaplace, cLaplace),
    'normal': (readyNormal, cNormal),
    'rayleigh': (readyRayleigh, cRayleigh),
    'maxwell-boltzmann': (readyMaxwellBoltzmann, cMaxwellBoltzmann),
    'inverse gamma': (readyInverseGamma, cInverseGamma),
    'levy': (readyLevy, cLevy),
    'weibull': (readyWeibull, cWeibull),
    'burrxii': (readyBurrXII, cBurrXII),
    'pareto': (readyPareto, cPareto),
    'dagum': (readyDagum, cDagum),
    'gompertz': (readyGompertz, cGompertz),
    'halfnormal': (readyHalfNormal, cHalfNormal),
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
        method='symbolic',
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

        self.ready_func, self.c_func = LIKELIHOOD_REGISTRY[self.likelihood]

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
        # Compute derivative
        # ----------------------------------------------------
        self._compute()

    # ========================================================
    # CORE COMPUTATION
    # ========================================================
    def _compute(self):
        """
        Delegates ALL math to mgfDerivative,
        using mitMGFprior ONLY as input.
        """
        result = mgfDerivative(
            order=self.a,
            prior=self.prior,
            method=self.method,
            t=float(-self.b),
            simplify=self.simplify,
            log=self.log,
            **self._deriv_kwargs
        )
        self._store_result(result)

    # ========================================================
    # RESULT STORAGE
    # ========================================================
    def _store_result(self, result):
        if isinstance(result, sp.Expr):
            self._expr = result
            self._is_symbolic = True
            self._log_abs = None
            self._sign = None
        else:
            self._expr = None
            self._is_symbolic = False
            self._log_abs, self._sign = result

    @property
    def is_symbolic(self):
        return self._is_symbolic

    # ========================================================
    # EVIDENCE
    # ========================================================
    def evidence(self):
        """
        Return the marginal likelihood (evidence).

        If `self.is_symbolic` is True, returns a symbolic expression.
        Otherwise, returns numeric (log_abs, sign) or ordinary value.
        """
        if self.is_symbolic:
            return self.c_func() * self._expr
        else:
            total_log_abs = self.log_c + self._log_abs
            if self.log:
                return total_log_abs, self._sign
            return math.exp(total_log_abs) * self._sign

    # ========================================================
    # POSTERIOR DENSITY
    # ========================================================
    def post_density(self, theta_val=None, log=True):
        """
        Compute the posterior density (or log-density) at given θ.

        If `self.is_symbolic` is True:
            - If theta_val is None or a sympy Symbol: returns a symbolic expression.
            - If theta_val is numeric: evaluates the expression numerically.
            - Uses the prior's symbolic PDF if available.
        If `self.is_symbolic` is False: performs numeric evaluation.
        """
        if self.is_symbolic:
            try:
                denom_expr = self._expr.subs(t, -self.b)

                # Determine theta symbol
                if theta_val is None or isinstance(theta_val, sp.Symbol):
                    theta_sym = theta if theta_val is None else theta_val
                else:
                    theta_sym = theta

                # Get symbolic PDF from the prior
                pdf_sym = self.prior.pdf_sym
                if pdf_sym is None:
                    raise ValueError("No symbolic PDF available for this prior.")

                if callable(pdf_sym):
                    pdf_sym = pdf_sym()

                if not isinstance(pdf_sym, sp.Expr):
                    raise TypeError("pdf_sym must be a SymPy expression.")

                # Substitute numeric parameters if they exist
                if self.params is not None:
                    subs_dict = {}
                    for sym in pdf_sym.free_symbols:
                        if sym.name in self.params:
                            subs_dict[sym] = self.params[sym.name]
                    if subs_dict:
                        pdf_sym = pdf_sym.subs(subs_dict)

                log_prior = sp.log(pdf_sym)

                log_num = log_prior + self.a * sp.log(theta_sym) - self.b * theta_sym
                log_post = log_num - sp.log(denom_expr)

                if theta_val is not None and not isinstance(theta_val, sp.Symbol):
                    if hasattr(theta_val, '__len__'):
                        from sympy import lambdify
                        func = lambdify(theta_sym, log_post, modules='numpy')
                        return func(theta_val)
                    else:
                        return float(log_post.subs(theta_sym, float(theta_val)).evalf())
                else:
                    return log_post if log else sp.exp(log_post)

            except Exception as e:
                print(f"⚠️ Symbolic computation failed: {e}. Falling back to numeric.")

        # ---- Numeric path ----
        if theta_val is None:
            raise ValueError("For numeric evaluation, theta must be provided.")

        # Get log prior density from the prior object
        if self.prior.logpdf_func is not None:
            log_prior = self.prior.logpdf_func(theta_val)
        elif self.prior.pdf_func is not None:
            log_prior = np.log(self.prior.pdf_func(theta_val))
        else:
            raise ValueError("No numeric PDF function available for this prior.")

        log_num = log_prior + self.a * np.log(theta_val) - self.b * theta_val
        log_post = log_num - self.log_abs
        if log:
            return log_post
        else:
            return np.exp(log_post)

    # ========================================================
    # POSTERIOR PREDICTIVE
    # ========================================================
    def post_predictive(self, new_data, log=True, **kwargs):
        """
        Compute the posterior predictive density (or log-density) for new data.
        """
        # ---- Symbolic path ----
        if self.is_symbolic:
            try:
                if isinstance(new_data, sp.Symbol):
                    a_new = sp.Symbol('a_new', real=True)
                    b_new = sp.Symbol('b_new', real=True)
                    log_c_new = sp.Symbol('log_c_new', real=True)
                    numeric_new = False
                else:
                    stats_new = self.ready_func(new_data, **kwargs)
                    a_new = stats_new['a']
                    b_new = stats_new['b']
                    log_c_new = stats_new['log_c']
                    numeric_new = True

                combined_order = self.a + a_new
                combined_b = self.b + b_new

                deriv_combined = mgfDerivative(
                    order=combined_order,
                    prior=self.prior,
                    method='symbolic',
                    t=None,
                    simplify=self.simplify,
                    log=False
                )
                num_expr = deriv_combined.subs(t, -combined_b)
                denom_expr = self._expr.subs(t, -self.b)

                log_pred = log_c_new + sp.log(num_expr) - sp.log(denom_expr)

                if numeric_new:
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
            simplify=self.simplify,
            log=True,
            **self._deriv_kwargs
        )
        log_pred = log_c_new + log_abs_num - self.log_abs
        if log:
            return log_pred
        else:
            sign_pred = sign_num * self._sign if hasattr(self, '_sign') and self._sign is not None else sign_num
            if log_pred == -float('inf'):
                return 0.0
            return sign_pred * math.exp(log_pred)

    # ========================================================
    # POSTERIOR MGF
    # ========================================================
    def post_mgf(self, r, log=False):
        """
        Compute the posterior moment-generating function (MGF) at given r.
        """
        # ---- Symbolic path ----
        if self.is_symbolic:
            try:
                if isinstance(r, sp.Symbol) or r is None:
                    r_sym = sp.Symbol('r', real=True) if r is None else r
                else:
                    r_sym = sp.Symbol('r', real=True)

                num_expr = self._expr.subs(t, r_sym - self.b)
                denom_expr = self._expr.subs(t, -self.b)

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
            simplify=self.simplify,
            log=True,
            **self._deriv_kwargs
        )

        log_ratio = log_abs_num - self.log_abs
        sign_ratio = sign_num * self._sign if hasattr(self, '_sign') and self._sign is not None else sign_num

        if log:
            return log_ratio
        else:
            if log_ratio == -float('inf'):
                return 0.0
            return sign_ratio * math.exp(log_ratio)

    # ========================================================
    # POSTERIOR MOMENT
    # ========================================================
    def post_moment(self, q, log=False):
        """
        Compute the posterior moment of order q.
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
                    t=None,
                    simplify=self.simplify,
                    log=False
                )
                num_expr = deriv_expr.subs(t, -self.b)
                denom_expr = self._expr.subs(t, -self.b)

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
            simplify=self.simplify,
            log=True,
            **self._deriv_kwargs
        )

        log_ratio = log_abs_num - self.log_abs
        sign_ratio = sign_num * self._sign if hasattr(self, '_sign') and self._sign is not None else sign_num

        if log:
            return log_ratio
        else:
            if log_ratio == -float('inf'):
                return 0.0
            return sign_ratio * math.exp(log_ratio)

    # ========================================================
    # SEQUENTIAL UPDATING
    # ========================================================

    def to_prior_object(self):
        """
        Convert current posterior into a mitMGFprior object.
        """
        return mitMGFprior.as_(
            name="posterior_prior",
            mgf=lambda r: self.post_mgf(r, log=False),
            pdf=lambda theta: self.post_density(theta, log=False)
        )

    def update(self, new_data, **kwargs):
        """
        Sequential update returns a new MGFDerivative,
        using posterior mitMGFprior as prior.
        """
        post_prior = self.to_prior_object()

        return MGFDerivative(
            prior=post_prior,
            data=new_data,
            likelihood=kwargs.get("likelihood", self.likelihood),
            method=kwargs.get("method", self.method),
            simplify=kwargs.get("simplify", self.simplify),
            log=kwargs.get("log", self.log),
            **kwargs
        )