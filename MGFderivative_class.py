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
from jumufraktiv.symbols import t, theta

# ============================================================
# Likelihood registry
# ============================================================
from jumufraktiv.like_stats.Poisson import readyPoisson, cPoisson
from jumufraktiv.like_stats.Gamma import readyGamma, cGamma
from jumufraktiv.like_stats.Laplace import readyLaplace, cLaplace
from jumufraktiv.like_stats.Normal import readyNormal, cNormal
from jumufraktiv.like_stats.Rayleigh import readyRayleigh, cRayleigh
from jumufraktiv.like_stats.MaxwellBoltzmann import readyMaxwellBoltzmann, cMaxwellBoltzmann
from jumufraktiv.like_stats.InverseGamma import readyInverseGamma, cInverseGamma
from jumufraktiv.like_stats.Levy import readyLevy, cLevy
from jumufraktiv.like_stats.Weibull import readyWeibull, cWeibull
from jumufraktiv.like_stats.BurrXII import readyBurrXII, cBurrXII
from jumufraktiv.like_stats.Pareto import readyPareto, cPareto
from jumufraktiv.like_stats.Dagum import readyDagum, cDagum
from jumufraktiv.like_stats.Gompertz import readyGompertz, cGompertz
from jumufraktiv.like_stats.HalfNormal import readyHalfNormal, cHalfNormal


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
            simplify=self.simplify,
            log=False,
            **self._deriv_kwargs
        )
        self._deriv_is_symbolic = isinstance(self._deriv, sp.Expr) # is the derivative function representation symbolic? Can I represent D_a(t) symbolically?
    
    def _evaluate_derivative(self, t_value):
        if isinstance(self._deriv, sp.Basic):
            val = self._deriv.subs(t, t_value)
            if not val.free_symbols:
                numeric_val = float(val.evalf())
                if self.log:
                    # Return (log_abs, sign)
                    if abs(numeric_val) < 1e-300:
                        return (-float('inf'), 1)
                    return (math.log(abs(numeric_val)), 1 if numeric_val > 0 else -1)
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

        # ----------------------------------------------------
        # Symbolic state
        # ----------------------------------------------------
        if isinstance(result, sp.Expr):
            self._result_expr = result
            self._is_symbolic = True # Is D_a(-b) symbolic after evaluation? is the stored evaluated result symbolic?
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
            self.log_abs, self._sign = result
            self.value = None

        else:
            # Expect ordinary numeric value
            if isinstance(result, tuple):
                raise TypeError(
                    "Expected numeric value when log=False, "
                    "but received (log_abs, sign)."
                )

            self.value = float(result)
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
            return self._sign * math.exp(self.log_abs)

        return self.value

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
                return math.exp(self.log_c) * self.value

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
        if self._is_symbolic:
            try:
                denom_expr = self._evaluate_derivative(-self.b)

                if theta_val is None or isinstance(theta_val, sp.Symbol):
                    theta_sym = theta if theta_val is None else theta_val

                else:
                    theta_sym = theta

                pdf_sym = self.prior.pdf_sym

                if callable(pdf_sym):
                    pdf_sym = pdf_sym()

                log_prior = sp.log(pdf_sym)

                log_num = (
                    log_prior
                    + self.a * sp.log(theta_sym)
                    - self.b * theta_sym
                )

                log_post = sp.simplify(
                    log_num - sp.log(denom_expr)
                )

                # resolve theta if numeric
                if theta_val is not None and not isinstance(theta_val, sp.Symbol):

                    evaluated = log_post.subs(theta_sym, theta_val).evalf()

                    if evaluated.free_symbols:
                        return evaluated if log else sp.exp(evaluated)

                    return float(evaluated) if log else float(sp.exp(evaluated))

                else:
                    return log_post if log else sp.exp(log_post)

            except Exception as e:
                raise RuntimeError(
                    f"Symbolic posterior density computation failed: {e}"
                ) from e

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
        # reconstruct normalization constant

        if self.log:
            log_denom = self.log_abs
        else:
            log_denom = math.log(self.value)

        log_post = log_num - log_denom

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
        if self._is_symbolic:
            try:
                if isinstance(new_data, sp.Symbol):
                    a_new = sp.Symbol('a_new', real=True)
                    b_new = sp.Symbol('b_new', real=True)
                    log_c_new = sp.Symbol('log_c_new', real=True)
                else:
                    stats_new = self.ready_func(new_data, **kwargs)
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
                    log=True
                )

                if isinstance(num, tuple):
                    raise RuntimeError(
                        "Symbolic predictive unexpectedly received numeric derivative."
                    )

                denom = self._evaluate_derivative(-self.b)

                log_pred = (
                    log_c_new
                    + sp.log(num)
                    - sp.log(denom)
                )

                # final symbol resolution
                if log_pred.free_symbols:
                    subs_dict = {}
                    for sym in log_pred.free_symbols:
                        if sym.name in self.params:
                            subs_dict[sym] = self.params[sym.name]
                    log_pred = log_pred.subs(subs_dict)
    
                    return log_pred if log else sp.exp(log_pred)

                return float(log_pred.evalf()) if log else float(sp.exp(log_pred).evalf())

            except Exception as e:
                raise RuntimeError(
                    f"Symbolic predictive computation failed: {e}"
                ) from e

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
            t=-b_combined,
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
    def post_mgf(self, r, log=True):
        """
        Compute the posterior moment-generating function (MGF) at given r.
        """
        # ---- Symbolic path ----
        if self._is_symbolic:
            try:
                r_sym = sp.Symbol('r', real=True) if r is None else (
                    r if isinstance(r, sp.Symbol)
                    else sp.Symbol('r', real=True)
                )

                num_expr = self._evaluate_derivative(r_sym - self.b)
                denom_expr = self._evaluate_derivative(-self.b)

                log_ratio = sp.log(num_expr) - sp.log(denom_expr)

                # substitute known parameters
                if self.params is not None:
                    log_ratio = log_ratio.subs(
                        {
                            sym:self.params[sym.name]
                            for sym in log_ratio.free_symbols
                            if sym.name in self.params
                        }
                    )

                # symbol-numeric decision
                if log_ratio.free_symbols:
                    return log_ratio if log else sp.exp(log_ratio)

                # fully numeric
                if hasattr(r, '__len__') and not isinstance(r,(str,bytes)):
                    func = sp.lambdify(r_sym, log_ratio, modules="numpy")
                    val = func(r)
                else:
                    val = float(log_ratio.evalf())

                return val if log else np.exp(val)

            except Exception as e:
                raise RuntimeError(
                    f"Symbolic computation failed: {e}. Falling back to numeric."
                ) from e

        # ---- Numeric path ----
        if r is None:
            raise ValueError("For numeric evaluation, r must be provided.")

        log_abs_num, sign_num = mgfDerivative(
            order=self.a,
            prior=self.prior,
            method=self.method,
            t=r - self.b,
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
    def post_moment(self, q, numerator_method='auto', log=True):
        """
        Compute the posterior moment of order q.
        """
        if self._is_symbolic:
            try:
                order = self.a + q

                deriv_expr = mgfDerivative(
                    order=order,
                    prior=self.prior,
                    method=numerator_method,
                    t=None,
                    simplify=self.simplify,
                    log=False
                )

                num_expr = deriv_expr.subs(t, -self.b)
                denom_expr = self._evaluate_derivative(-self.b)

                log_ratio = sp.log(num_expr) - sp.log(denom_expr)

                # substitute known parameters
                if self.params is not None:
                    log_ratio = log_ratio.subs(
                        {
                            sym: self.params[sym.name]
                            for sym in log_ratio.free_symbols
                            if sym.name in self.params
                        }
                    )

                # symbol-numeric decision
                if log_ratio.free_symbols:
                    return log_ratio if log else sp.exp(log_ratio)

                # fully numeric
                val = float(log_ratio.evalf())
                return val if log else np.exp(val)

            except Exception as e:
                raise RuntimeError(
                    f"Symbolic computation failed: {e}. Falling back to numeric."
                ) from e

        # ---- Numeric path ----
        if self._is_symbolic:
            raise ValueError("Cannot compute numeric moment from a symbolic derivative.")
        if not isinstance(q, (int, float)):
            raise ValueError("For numeric evaluation, q must be numeric.")

        order_num = self.a + q
        log_abs_num, sign_num = mgfDerivative(
            order=order_num,
            prior=self.prior,
            method=numerator_method,
            t=-self.b,
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
        Tries to construct a symbolic prior first if possible.
        """
        # ---- Try symbolic route (if derivative is symbolic) ----
        print("self._is_symbolic =", self._is_symbolic)
        print("type(self._is_symbolic) =", type(self._is_symbolic))

        if self._is_symbolic:
            try:
                # Use 'r' as the MGF argument symbol (post_mgf expects a symbol)
                r_sym = sp.Symbol('r', real=True)

                # Get symbolic expressions from post_mgf and post_density
                mgf_sym_expr = self.post_mgf(r_sym, log=False)
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
        def mgf_backend(t_val, xp=math, **params):
            return self.post_mgf(t_val, log=self.log)

        def pdf_backend(theta_val, xp=math, **params):
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
        print("symbolic:", self._is_symbolic)
        return MGFDerivative(
            prior=post_prior,
            data=new_data,
            likelihood=likelihood,
            method=method,
            simplify=simplify,
            log=log, # new object's requested state
            **kwargs
        )

