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
from MGFdictionary.heavisideMGF import (
    heaviside_mgf_symbolic,
    heaviside_cgf_symbolic,
    heaviside_cgf,
    heaviside_mgf,
    heaviside_cgf_jax,
    heaviside_mgf_jax,
    heaviside_pdf_symbolic,
    heaviside_pdf_symbolic_sub,
)
from MGFdictionary.uniformMGF import (
    uniform_mgf_symbolic,
    uniform_cgf_symbolic,
    uniform_cgf,
    uniform_mgf,
    uniform_cgf_jax,
    uniform_mgf_jax,
    uniform_pdf_symbolic,
    uniform_pdf_symbolic_sub,
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
    'heaviside': {
        'dist': None,
        'mgf_sym': heaviside_mgf_symbolic,
        'cgf_sym': heaviside_cgf_symbolic,
        'mgf': heaviside_mgf,
        'cgf': heaviside_cgf,
        'mgf_jax': heaviside_mgf_jax,
        'cgf_jax': heaviside_cgf_jax,
        'pdf_sym': heaviside_pdf_symbolic,
        'pdf_sym_func': heaviside_pdf_symbolic_sub,
        'pdf_func': lambda theta, k: 1.0 if theta >= k else 0.0,
        'logpdf_func': lambda theta, k: 0.0 if theta >= k else -np.inf,
    },
    'uniform': {
        'dist': lambda p: stats.uniform(loc=p['a'], scale=p['b'] - p['a']),
        'mgf_sym': uniform_mgf_symbolic,
        'cgf_sym': uniform_cgf_symbolic,
        'mgf': uniform_mgf,
        'cgf': uniform_cgf,
        'mgf_jax': uniform_mgf_jax,
        'cgf_jax': uniform_cgf_jax,
        'pdf_sym': uniform_pdf_symbolic,
        'pdf_sym_func': uniform_pdf_symbolic_sub,
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
        - Priors: gamma, pareto, heaviside, uniform (others can be added to the registry)
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

    def __init__(
        self,
        prior: str,
        data,
        likelihood: str = 'poisson',
        method: str = 'symbolic',
        params: dict = None,
        simplify: bool = False,
        log: bool = True,
        prior_mgf_func=None,
        prior_cgf_func=None,
        prior_pdf_func=None,
        prior_logpdf_func=None,
        prior_pdf_sym_func=None,
        prior_mgf_sym=None,
        prior_cgf_sym=None,
        ready_func=None,
        c_func=None,
        **kwargs
    ):
        """
        Compute the MGF derivative for the given data and prior.

        Parameters
        ----------
        prior : str
            Prior name (must be in PRIOR_REGISTRY) or 'custom'.
        data : pandas DataFrame, Series, or array‑like
            Observed data.
        likelihood : str, optional
            One of the supported likelihoods or 'custom'. Default 'poisson'.
        method : str, optional
            For integer order: 'symbolic', 'bell', 'jax'.
            For fractional order: 'scipy', 'mpmath', 'symbolic' (if order is fractional).
            Default 'symbolic'.
        params : dict or None
            Prior parameters. If None and method='symbolic', returns symbolic expression.
            For 'custom' prior, if provided, will be used for numeric substitution
            when method='symbolic'.
        simplify : bool, optional
            If True, simplify symbolic expressions.
        log : bool, optional
            If True, store derivative in log scale (numeric only).
        prior_mgf_func : callable, optional
            Function (t, *args) -> M(t) for custom prior (numeric).
        prior_cgf_func : callable, optional
            Function (t, *args) -> log M(t) for custom prior (numeric).
        prior_pdf_func : callable, optional
            Function (theta, *args) -> p(theta) for custom prior.
        prior_logpdf_func : callable, optional
            Function (theta, *args) -> log p(theta) for custom prior.
        prior_pdf_sym_func : callable, optional
            Function (params) -> symbolic PDF expression for custom prior.
        prior_mgf_sym : sympy.Expr or callable, optional
            Symbolic MGF expression (or function returning one) for custom prior with method='symbolic'.
        prior_cgf_sym : sympy.Expr or callable, optional
            Symbolic CGF expression (or function returning one) for custom prior with method='symbolic'.
        ready_func : callable, optional
            For custom likelihood: function (data, **kwargs) -> dict with 'a', 'b', 'log_c'.
        c_func : callable, optional
            For custom likelihood: function () -> symbolic expression for the normalising constant c(y).
        **kwargs : additional arguments passed to the likelihood's ready function
                   and/or to mgfDerivative.
            For Poisson: scale.
            For Gamma: shape.
            For Weibull: rho.
            For BurrXII: known_shape.
            For Dagum: r, s.
            For mgfDerivative: integer_method, epsrel, dps, tol, symbolic_timeout, etc.
        """
        # ---- Validate prior ----
        self.prior = prior.lower()
        self.prior_info = None
        self._custom_prior = False

        if self.prior == 'custom':
            self._custom_prior = True
            # Store custom functions
            self._prior_mgf_func = prior_mgf_func
            self._prior_cgf_func = prior_cgf_func
            self._prior_pdf_func = prior_pdf_func
            self._prior_logpdf_func = prior_logpdf_func
            self._prior_pdf_sym_func = prior_pdf_sym_func
            self._prior_mgf_sym = prior_mgf_sym
            self._prior_cgf_sym = prior_cgf_sym
            # Create a dummy prior_info for consistency
            self.prior_info = {
                'dist': None,
                'mgf_sym': prior_mgf_sym,
                'cgf_sym': prior_cgf_sym,
                'mgf': prior_mgf_func,
                'cgf': prior_cgf_func,
                'mgf_jax': None,
                'cgf_jax': None,
                'pdf_sym': prior_pdf_sym_func,
                'pdf_sym_func': lambda p: prior_pdf_sym_func,
                'pdf_func': prior_pdf_func,
                'logpdf_func': prior_logpdf_func,
            }
            # For custom prior, store params if provided (for numeric substitution)
            self.params = params
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
            params is not None and
            all(isinstance(v, (int, float)) for v in params.values())
        )

        # ---- Handle custom likelihood ----
        if self.likelihood == 'custom':
            if ready_func is None:
                raise ValueError("For custom likelihood, ready_func must be provided.")
            if c_func is None:
                raise ValueError("For custom likelihood, c_func must be provided.")
            self.ready_func = ready_func
            self.c_func = c_func
            # No registry lookup
        else:
            if self.likelihood not in self.LIKELIHOOD_REGISTRY:
                raise ValueError(f"Unsupported likelihood: {likelihood}. "
                                 f"Choose from {list(self.LIKELIHOOD_REGISTRY.keys())} or 'custom'.")
            self.ready_func, self.c_func = self.LIKELIHOOD_REGISTRY[self.likelihood]

        # ---- Separate kwargs for ready vs derivative ----
        # For custom likelihood, we don't filter based on _ready_keys; just pass all kwargs
        if self.likelihood == 'custom':
            self._ready_kwargs = kwargs  # all kwargs go to ready_func
            self._deriv_kwargs = {}       # no derivative-specific kwargs from this call
        else:
            self._ready_kwargs = {k: v for k, v in kwargs.items() if k in self._ready_keys}
            self._deriv_kwargs = {k: v for k, v in kwargs.items() if k not in self._ready_keys}

        # ---- Compute sufficient statistics ----
        stats = self.ready_func(data, **self._ready_kwargs)
        self.a = stats['a']
        self.b = stats['b']
        self.log_c = stats['log_c']

        # ---- Warn if order is integer but method is fractional ----
        if abs(self.a - round(self.a)) < 1e-12:
            if self.method.lower() in ('scipy', 'mpmath'):
                import warnings
                warnings.warn(
                    f"The derivative order a = {self.a} is integer. "
                    f"Methods 'scipy' and 'mpmath' are intended for fractional orders. "
                    f"Consider using method='symbolic', 'bell', or 'jax' for integer orders. "
                    f"Your current method '{self.method}' will be dispatched to the integer path.",
                    UserWarning
                )

        # ---- Compute derivative ----
        if self.likelihood in self._special_likelihoods:
            self._compute_derivative_special()
        else:
            self._compute_derivative_standard()

    def _compute_derivative_standard(self):
        """Standard derivative computation using mgfDerivative or custom prior."""
        if self._custom_prior:
            # ---- Custom prior ----
            if self.method.lower() == 'symbolic':
                # Require symbolic MGF
                if self._prior_mgf_sym is None:
                    raise ValueError(
                        "For method='symbolic' with custom prior, prior_mgf_sym must be provided."
                    )
                # Get symbolic MGF expression
                mgf_expr = self._prior_mgf_sym() if callable(self._prior_mgf_sym) else self._prior_mgf_sym
                # Ensure it's a SymPy expression
                if not isinstance(mgf_expr, sp.Expr):
                    raise TypeError("prior_mgf_sym must be a SymPy expression or a callable returning one.")
                # Differentiate with respect to t
                t_sym = sp.Symbol('t', real=True)
                deriv_expr = sp.diff(mgf_expr, t_sym, int(round(self.a)))
                # Evaluate at t = -b if it's numeric
                if not math.isnan(-self.b):
                    deriv_expr = deriv_expr.subs(t_sym, -self.b)
                # Substitute numeric parameters if provided (they may be in self.params)
                if self.params is not None:
                    for sym in deriv_expr.free_symbols:
                        if sym.name in self.params:
                            deriv_expr = deriv_expr.subs(sym, self.params[sym.name])
                if self.simplify:
                    deriv_expr = sp.simplify(deriv_expr)
                self._is_symbolic = True
                self._expr = deriv_expr
                self._log_abs = None
                self._sign = None
                self._value = None
                return

            elif self.method.lower() in ('jax', 'bell'):
                # Numeric route: require MGF function
                if self._prior_mgf_func is None:
                    raise ValueError(
                        f"For method='{self.method}' with custom prior, prior_mgf_func must be provided."
                    )
                # If bell and no CGF provided, derive from MGF (log)
                if self.method.lower() == 'bell' and self._prior_cgf_func is None:
                    # Define CGF as log of MGF
                    def cgf_from_mgf(t, *args):
                        return math.log(self._prior_mgf_func(t, *args))
                    self._prior_cgf_func = cgf_from_mgf

                # Use JAX for differentiation
                import jax
                import jax.numpy as jnp
                jax.config.update("jax_enable_x64", True)
                t_jax = jnp.array(-self.b, dtype=jnp.float64)

                if self.method.lower() == 'jax':
                    # Differentiate MGF directly
                    mgf_func = self._prior_mgf_func
                    f = mgf_func
                    for _ in range(int(round(self.a))):
                        f = jax.grad(f, argnums=0)
                    deriv_val = float(f(t_jax))
                    # Store numeric result
                    if abs(deriv_val) < 1e-300:
                        log_abs = -float('inf')
                        sign = 1
                    else:
                        log_abs = math.log(abs(deriv_val))
                        sign = 1 if deriv_val > 0 else -1
                    if self.log:
                        self._log_abs, self._sign = log_abs, sign
                        self._value = None
                    else:
                        self._value = deriv_val
                        self._log_abs, self._sign = None, None
                    self._is_symbolic = False
                    self._expr = None
                    return

                else:  # bell
                    # Differentiate CGF and use Bell polynomial
                    from logsum import bell_polynomial_log
                    cgf_func = self._prior_cgf_func
                    # Build derivative functions up to order a (integer)
                    order_int = int(round(self.a))
                    deriv_funcs = [cgf_func]
                    for _ in range(order_int):
                        deriv_funcs.append(jax.grad(deriv_funcs[-1], argnums=0))
                    kappa_log_abs = []
                    kappa_sign = []
                    for k in range(1, order_int + 1):
                        val = float(deriv_funcs[k](t_jax))
                        if abs(val) < 1e-300:
                            kappa_log_abs.append(-float('inf'))
                            kappa_sign.append(1)
                        else:
                            kappa_log_abs.append(math.log(abs(val)))
                            kappa_sign.append(1 if val > 0 else -1)
                    log_abs_B, sign_B = bell_polynomial_log(order_int, kappa_log_abs, kappa_sign)
                    cgf_t = float(cgf_func(t_jax))
                    deriv_log_abs = cgf_t + log_abs_B
                    deriv_sign = sign_B
                    if self.log:
                        self._log_abs, self._sign = deriv_log_abs, deriv_sign
                        self._value = None
                    else:
                        self._value = deriv_sign * math.exp(deriv_log_abs)
                        self._log_abs, self._sign = None, None
                    self._is_symbolic = False
                    self._expr = None
                    return

            else:
                raise ValueError(f"Unsupported method '{self.method}' for custom prior. "
                                 f"Choose 'symbolic', 'jax', or 'bell'.")

        # ---- Non-custom prior: use mgfDerivative ----
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
        For custom prior, we treat it as symbolic only if the method is 'symbolic' and params are None.
        """
        if self._custom_prior:
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
            return None
        dist_constructor = self.prior_info['dist']
        if dist_constructor is None:
            return None
        return dist_constructor(self.params)

    def evidence(self):
        """
        Return the marginal likelihood (evidence).

        If `self.is_symbolic` is True, returns a symbolic expression.
        Otherwise, uses the already‑computed numeric derivative (stored in `_log_abs` and `_sign`).
        """
        if self.is_symbolic:
            c_expr = self.c_func()
            return c_expr * self._expr
        else:
            # The numeric derivative must have been computed at instantiation
            if self._log_abs is None or self._sign is None:
                raise RuntimeError(
                    "Numeric derivative not available. "
                    "Make sure the derivative was computed during instantiation."
                )
            total_log_abs = self.log_c + self._log_abs
            if self.log:
                return total_log_abs, self._sign
            else:
                return math.exp(total_log_abs) * self._sign

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

    def update(self, new_data, method=None, log=None, simplify=None, params=None, likelihood=None, **kwargs):
        """
        Perform sequential Bayesian updating with new data.

        If `self.is_symbolic` is True and `method='symbolic'`, it uses the symbolic
        posterior MGF expression as the prior for the new data. If `params` is provided,
        it will be used for numeric substitution during derivative evaluation.
        Otherwise, it uses numeric methods (jax, bell, scipy, mpmath).

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
        params : dict, optional
            Numeric hyperparameters for the symbolic prior (only used if method='symbolic').
            If provided, the derivative expression will be evaluated numerically after
            symbolic differentiation.
        likelihood : str, optional
            Likelihood to use for the new data. If not provided, uses the same likelihood
            as the current object. If provided and differs from the current likelihood,
            a warning is issued.
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
        if likelihood is None:
            likelihood = self.likelihood

        # ---- Warn if likelihood changes ----
        if likelihood != self.likelihood:
            import warnings
            warnings.warn(
                f"Changing likelihood from '{self.likelihood}' to '{likelihood}' in sequential update. "
                "This means the new data will be processed under a different likelihood.",
                UserWarning
            )

        # ---- Symbolic sequential update ----
        if method.lower() == 'symbolic':
            if not self.is_symbolic:
                raise ValueError(
                    "Cannot use symbolic method for sequential update when the posterior derivative is not symbolic. "
                    "Choose a numeric method (jax, bell, scipy, mpmath)."
                )
            # Get symbolic posterior MGF expression using existing post_mgf method
            r_sym = sp.Symbol('r', real=True)
            post_mgf_expr = self.post_mgf(r_sym, log=False)
            # Substitute r with t to get the prior MGF as a function of t
            t_sym = sp.Symbol('t', real=True)
            prior_mgf_expr = post_mgf_expr.subs(r_sym, t_sym)
            prior_cgf_expr = sp.log(prior_mgf_expr)
            if simplify:
                prior_mgf_expr = sp.simplify(prior_mgf_expr)
                prior_cgf_expr = sp.simplify(prior_cgf_expr)

            # Merge ready kwargs
            merged_ready = {**self._ready_kwargs, **kwargs}

            # ---- Handle likelihood functions ----
            # If likelihood is 'custom', we need ready_func and c_func.
            # They may be provided in kwargs or from self.
            if likelihood == 'custom':
                if 'ready_func' not in merged_ready and 'ready_func' not in self.__dict__:
                    raise ValueError("For likelihood='custom', ready_func must be provided.")
                if 'c_func' not in merged_ready and 'c_func' not in self.__dict__:
                    raise ValueError("For likelihood='custom', c_func must be provided.")
                # Use user-provided if given, else fall back to self's
                if 'ready_func' not in merged_ready:
                    merged_ready['ready_func'] = self.ready_func
                if 'c_func' not in merged_ready:
                    merged_ready['c_func'] = self.c_func
            else:
                # For named likelihood, ignore any ready_func/c_func in kwargs (with warning)
                if 'ready_func' in merged_ready or 'c_func' in merged_ready:
                    import warnings
                    warnings.warn(
                        "ready_func and c_func are ignored when likelihood is not 'custom'.",
                    UserWarning
                    )
                    merged_ready.pop('ready_func', None)
                    merged_ready.pop('c_func', None)

            return MGFDerivative(
                prior='custom',
                data=new_data,
                likelihood=likelihood,
                method='symbolic',
                params=params,
                simplify=simplify,
                log=log,
                prior_mgf_sym=lambda: prior_mgf_expr,
                prior_cgf_sym=lambda: prior_cgf_expr,
                **merged_ready
            )

        # ---- Numeric sequential update ----
        # Create numeric lambdas using existing methods
        post_mgf = lambda r_val, *args: self.post_mgf(r_val, log=False)
        post_cgf = lambda r_val, *args: self.post_mgf(r_val, log=True)
        post_pdf = lambda theta, *args: self.post_density(theta, log=False)
        post_logpdf = lambda theta, *args: self.post_density(theta, log=True)

        # Merge ready kwargs
        merged_ready = {**self._ready_kwargs, **kwargs}

        # ---- Handle likelihood functions ----
        if likelihood == 'custom':
            if 'ready_func' not in merged_ready and 'ready_func' not in self.__dict__:
                raise ValueError("For likelihood='custom', ready_func must be provided.")
            if 'c_func' not in merged_ready and 'c_func' not in self.__dict__:
                raise ValueError("For likelihood='custom', c_func must be provided.")
            if 'ready_func' not in merged_ready:
                merged_ready['ready_func'] = self.ready_func
            if 'c_func' not in merged_ready:
                merged_ready['c_func'] = self.c_func
        else:
            if 'ready_func' in merged_ready or 'c_func' in merged_ready:
                import warnings
                warnings.warn(
                    "ready_func and c_func are ignored when likelihood is not 'custom'.",
                    UserWarning
                )
                merged_ready.pop('ready_func', None)
                merged_ready.pop('c_func', None)

        return MGFDerivative(
        prior='custom',
        data=new_data,
        likelihood=likelihood,
        method=method,
        params=None,           # numeric custom prior has embedded parameters
        simplify=simplify,
        log=log,
        prior_mgf_func=post_mgf,
        prior_cgf_func=post_cgf,
        prior_pdf_func=post_pdf,
        prior_logpdf_func=post_logpdf,
        **merged_ready
        )