"""
MGFderivative_class.py

Defines a class MGFDerivative that encapsulates the computation of MGF derivatives
and marginal likelihoods (evidence) for various likelihoods and priors.

Supports sequential updating via the `update` method, using the posterior MGF
as the prior for the next chunk of data.

Custom priors are supported via `prior='custom'` with either symbolic or numeric
functions. Symbolic requires `prior_mgf_sym`; numeric requires `prior_mgf_func`.

Custom likelihoods are supported via `likelihood='custom'`, requiring `ready_func`
and `c_func` to be provided.
"""

import math
import sympy as sp
import numpy as np
import pandas as pd

from derivativeDispatch import mgfDerivative
from mitMGFprior_class import mitMGFprior


# ============================================================
# Likelihood registry (UNCHANGED)
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
        prior,                  # <-- NOW: mitMGFprior object ONLY
        data,
        likelihood='poisson',
        method='symbolic',
        simplify=False,
        log=True,
        ready_func=None,
        c_func=None,
        **kwargs
    ):
        """
        prior must be a mitMGFprior instance.
        """

        # ----------------------------------------------------
        # PRIOR HANDLING (SIMPLIFIED)
        # ----------------------------------------------------
        if not isinstance(prior, mitMGFprior):
            raise TypeError("prior must be a mitMGFprior object")

        self.prior = prior
        self.prior_info = prior  # direct access
        self.params = prior.params

        # expose prior functions directly
        self.prior_mgf = prior.mgf
        self.prior_cgf = prior.cgf
        self.prior_pdf = prior.pdf_func
        self.prior_logpdf = prior.logpdf_func

        # symbolic if available
        self.prior_mgf_sym = prior.mgf_sym
        self.prior_cgf_sym = prior.cgf_sym

        # ----------------------------------------------------
        # likelihood
        # ----------------------------------------------------
        self.likelihood = likelihood.lower()
        self.data = data
        self.method = method
        self.simplify = simplify
        self.log = log

        if self.likelihood not in LIKELIHOOD_REGISTRY:
            raise ValueError(f"Unknown likelihood {likelihood}")

        self.ready_func, self.c_func = LIKELIHOOD_REGISTRY[self.likelihood]

        # ----------------------------------------------------
        # sufficient statistics
        # ----------------------------------------------------
        stats = self.ready_func(data, **kwargs)
        self.a = stats['a']
        self.b = stats['b']
        self.log_c = stats['log_c']

        # ----------------------------------------------------
        # compute derivative via existing engine
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
            params=self.params,
            simplify=self.simplify,
            log=self.log
        )

        self._store_result(result)

    # ========================================================
    # result storage
    # ========================================================
    def _store_result(self, result):
        if isinstance(result, sp.Expr):
            self._expr = result
            self._is_symbolic = True
        else:
            self._expr = None
            self._is_symbolic = False
            self._log_abs, self._sign = result

    # ========================================================
    # evidence
    # ========================================================
    def evidence(self):
        if self._is_symbolic:
            return self.c_func() * self._expr
        else:
            total = self.log_c + self._log_abs
            if self.log:
                return total, self._sign
            return math.exp(total) * self._sign

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

        # Get log prior density
        if self._custom_prior:
            if self._prior_logpdf_func is not None:
                log_prior = self._prior_logpdf_func(theta)
            elif self._prior_pdf_func is not None:
                log_prior = np.log(self._prior_pdf_func(theta))
            else:
                raise ValueError("No numeric PDF function provided for custom prior.")
        else:
            prior_info = self.prior_info
            if 'logpdf_func' in prior_info and prior_info['logpdf_func'] is not None:
                log_prior = prior_info['logpdf_func'](theta, **self.params)
            elif 'pdf_func' in prior_info and prior_info['pdf_func'] is not None:
                log_prior = np.log(prior_info['pdf_func'](theta, **self.params))
            elif prior_info['dist'] is not None:
                dist = prior_info['dist'](self.params)
                log_prior = dist.logpdf(theta)
            else:
                raise NotImplementedError("No numeric PDF function available for this prior. Please use custom prior with numeric PDF or use symbolic path.")

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

    def to_prior_object(self):
        """
        Convert current posterior into a mitMGFprior object.
        """

        return mitMGFprior(
            name="posterior_prior",
            mgf=self.post_mgf,
            cgf=lambda r: self.post_mgf(r, log=True),
            pdf_func=lambda theta: self.post_density(theta, log=False),
            logpdf_func=lambda theta: self.post_density(theta, log=True),
            mgf_sym=self.prior.mgf_sym,   # optional fallback
            pdf_sym=self.prior.pdf_sym,
            params=self.params
        ).as_mitMGFprior()
        
    def update(self, new_data, **kwargs):
        """
        Sequential update returns a new MGFDerivative,
        using posterior mitMGFprior as prior.
        """

        # 1. compute posterior object (unchanged logic inside class)
        post_prior = self.to_prior_object()

        # 2. return new inference problem
        return MGFDerivative(
            prior=post_prior,
            data=new_data,
            likelihood=kwargs.get("likelihood", self.likelihood),
            method=kwargs.get("method", self.method),
            simplify=kwargs.get("simplify", self.simplify),
            log=kwargs.get("log", self.log),
            **kwargs
        )