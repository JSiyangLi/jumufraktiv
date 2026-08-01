"""
MGFDerivative_class.py

Defines the `MGFDerivative` class, which encapsulates the computation of
posterior distributions via MGF marginalisation. It computes the marginal
likelihood (evidence) and provides a wide range of inference methods:

- Posterior density (`post_density`)
- Cumulative distribution function (`post_cdf`)
- Quantiles (`post_quantile`)
- Moment‑generating function (`post_mgf`)
- Raw and central moments (`post_raw_moment`, `post_central_moment`)
- Credible intervals (`post_interval`)
- Posterior sampling (`post_sample`)
- Posterior predictive density (`post_predictive`)

The class supports both symbolic and numeric evaluation, respecting the
**symbol‑numeric principle**: the return type depends only on whether
unresolved symbols remain. Numeric methods are fully vectorised for array
inputs, following the **tuple‑vectorisation principle** for evaluation
points `(t, u)`.

Priors are represented as `mitMGFprior` objects. The class also supports
sequential Bayesian updating via the `update` method, which treats the
current posterior as the prior for a new dataset.

Likelihoods are registered with `ready_func` (aggregated sufficient
statistics) and `bereit_func` (per‑element statistics for vectorised
predictive evaluation).
"""

import difflib
import inspect
import math
import warnings

import numpy as np
import sympy as sp

from jumufraktiv.derivativeDispatch import (
    _check_moment_exists_at_origin,
    mgfDerivative,
    resolve_backend,
)
from jumufraktiv.like_stats.BurrXII import bereitBurrXII, cBurrXII, readyBurrXII
from jumufraktiv.like_stats.Dagum import bereitDagum, cDagum, readyDagum
from jumufraktiv.like_stats.Gamma import bereitGamma, cGamma, readyGamma
from jumufraktiv.like_stats.Gompertz import bereitGompertz, cGompertz, readyGompertz
from jumufraktiv.like_stats.HalfNormal import (
    bereitHalfNormal,
    cHalfNormal,
    readyHalfNormal,
)
from jumufraktiv.like_stats.InverseGamma import (
    bereitInverseGamma,
    cInverseGamma,
    readyInverseGamma,
)
from jumufraktiv.like_stats.Laplace import bereitLaplace, cLaplace, readyLaplace
from jumufraktiv.like_stats.Levy import bereitLevy, cLevy, readyLevy
from jumufraktiv.like_stats.MaxwellBoltzmann import (
    bereitMaxwellBoltzmann,
    cMaxwellBoltzmann,
    readyMaxwellBoltzmann,
)
from jumufraktiv.like_stats.Normal import bereitNormal, cNormal, readyNormal
from jumufraktiv.like_stats.Pareto import bereitPareto, cPareto, readyPareto

# ============================================================
# Likelihood registry
# ============================================================
# ---- Likelihood imports ----
from jumufraktiv.like_stats.Poisson import bereitPoisson, cPoisson, readyPoisson
from jumufraktiv.like_stats.Rayleigh import bereitRayleigh, cRayleigh, readyRayleigh
from jumufraktiv.like_stats.Weibull import bereitWeibull, cWeibull, readyWeibull
from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.root_finding import solve_root

# `q` is deliberately absent. The moment methods take a parameter named `q`
# that may itself be the canonical symbol -- the caller imports it and passes
# it in -- so every `q` in this file is that parameter. Importing the symbol
# here as well bound a module-level name nothing read, and left every moment
# method's signature shadowing it.
from jumufraktiv.symbols import r, t, theta, u

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
# Keyword-argument routing
# ============================================================
#: Keyword arguments understood by the derivative layer. These are forwarded to
#: :func:`~jumufraktiv.derivativeDispatch.mgfDerivative` and, through it, to the
#: selected backend. Kept explicit rather than inferred, because the backends
#: terminate in ``**kwargs`` and so cannot be introspected reliably.
#: Every name here is consumed by some backend. That is a property worth
#: stating, because the set had drifted: `use_interpolation` and `d_vec` tuned
#: the near-integer interpolation module PR 6b deleted, and `epsabs`, `epsrel`,
#: `limit`, `initial_L` and `max_L` tuned the adaptive quadrature PR 8 deleted.
#: Seven names the constructor accepted therefore reached no code at all -- the
#: caller who set `initial_L` to widen the truncation range got the default,
#: silently, which is the same failure mode as rejecting the option except
#: harder to notice.
DERIVATIVE_KWARGS = frozenset({
    # mgfDerivative's own named parameters that a caller may reasonably set
    "integer_method", "int_tol",
    # popped from **kwargs inside mgfDerivative
    "cgf_method", "symbolic_timeout",
    # fixed-grid fractional kernel
    "tol",
    # mpmath backend
    "dps", "use_tan",
})

#: Arguments this class supplies to ``mgfDerivative`` itself. A caller must not
#: be able to set them: ``complete`` in particular is ``True`` for the evidence
#: and ``False`` for the incomplete-MGF path behind ``post_cdf``, so accepting it
#: from the caller would either be ignored or raise "multiple values for
#: keyword argument". Enforced by a test.
_RESERVED_DERIVATIVE_KWARGS = frozenset({
    "order", "prior", "method", "t", "u", "simplify", "complete", "log",
})


def _likelihood_kwargs(ready_func) -> frozenset:
    """Return the keyword arguments a likelihood's ``ready`` function accepts.

    Derived from the function's signature rather than a hardcoded list, so it
    cannot drift out of step with the likelihood modules, and so each likelihood
    is checked against *its own* parameters instead of the union of everyone's.

    Parameters
    ----------
    ready_func : callable
        A ``readyX`` function from :mod:`jumufraktiv.like_stats`.

    Returns
    -------
    frozenset of str
        Named parameters other than ``data``. A trailing ``**kwargs`` on the
        function is deliberately ignored: every ``ready`` function has one, and
        it is what silently absorbed misspelled arguments.
    """
    parameters = inspect.signature(ready_func).parameters
    return frozenset(
        name
        for name, param in parameters.items()
        if name != "data"
        and param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
    )


def _split_kwargs(kwargs, ready_func, likelihood):
    """Route keyword arguments to the likelihood or the derivative layer.

    Parameters
    ----------
    kwargs : dict
        Extra keyword arguments given to :class:`MGFDerivative`.
    ready_func : callable
        The chosen likelihood's ``ready`` function.
    likelihood : str
        Name of the chosen likelihood, used in the error message.

    Returns
    -------
    tuple of dict
        ``(likelihood_kwargs, derivative_kwargs)``.

    Raises
    ------
    TypeError
        If any argument belongs to neither group.

    Notes
    -----
    The previous implementation matched against the *union* of every
    likelihood's parameter names and sent everything else to the derivative
    layer, where ``**kwargs`` absorbed it. Two mistakes were therefore silent
    and produced a confidently wrong number rather than an error:

    - a misspelling (``scal=2.0``) fell through to the derivative layer, so the
      likelihood used its default and the evidence was wrong by 0.92 nats;
    - a parameter valid for a *different* likelihood (``rho=`` on a Poisson) was
      forwarded into ``ready_func`` and swallowed by its ``**kwargs``.

    Both now raise.
    """
    accepted = _likelihood_kwargs(ready_func)

    likelihood_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    derivative_kwargs = {k: v for k, v in kwargs.items() if k in DERIVATIVE_KWARGS}

    unknown = set(kwargs) - accepted - DERIVATIVE_KWARGS
    if unknown:
        raise TypeError(_unknown_kwargs_message(unknown, accepted, likelihood))

    return likelihood_kwargs, derivative_kwargs


def _unknown_kwargs_message(unknown, accepted, likelihood):
    """Build an actionable message for unrecognised keyword arguments."""
    lines = []
    for name in sorted(unknown):
        suggestions = difflib.get_close_matches(
            name, sorted(accepted | DERIVATIVE_KWARGS), n=1, cutoff=0.7
        )
        if suggestions:
            lines.append(f"'{name}' (did you mean '{suggestions[0]}'?)")
        else:
            lines.append(f"'{name}'")

    accepted_text = ", ".join(sorted(accepted)) if accepted else "none"
    return (
        f"Unexpected keyword argument(s) for MGFDerivative: {', '.join(lines)}. "
        f"The '{likelihood}' likelihood accepts: {accepted_text}. "
        f"Derivative options are: {', '.join(sorted(DERIVATIVE_KWARGS))}."
    )


# ============================================================
# Core class
# ============================================================
class MGFDerivative:
    """
    Posterior distribution derived via MGF marginalisation.

    This class encapsulates a posterior distribution obtained from a prior
    (a `mitMGFprior` object) and a likelihood from the MGF‑marginalisable
    family. It computes the posterior normalising constant (evidence) and
    provides methods for density, CDF, quantiles, moments, and predictive
    inference.

    The core computation is based on fractional derivatives of the prior
    moment‑generating function (MGF), evaluated at the posterior location
    `t = -b`. The class stores either a symbolic expression or a numeric
    value for the normalising constant, depending on the chosen backend.

    Attributes
    ----------
    a : float
        Order of differentiation. Together with `b` this forms the pair
        `(a, b)`, which is jointly sufficient for `theta`; neither component is
        sufficient on its own.
    b : float
        Evaluation point, entering as `t = -b`. See `a`.
    log_c : float
        Log‑normalising constant from the likelihood.
    log : bool
        Whether the normalising constant is stored in log‑scale.
    prior : mitMGFprior
        The prior object used.
    method : str
        Derivative backend method.
    simplify : bool
        Whether to simplify symbolic expressions.

    Properties
    ----------
    is_symbolic : bool
        True if the normalising constant is a symbolic expression.
    value_numeric : float
        Numeric value of the normalising constant (if numeric).
    prior_has_iMGF : bool
        True if the prior supports the incomplete MGF (iMGF).

    Methods
    -------
    evidence()
        Return the marginal likelihood (evidence).
    post_density(theta_val, log=True)
        Evaluate the posterior density.
    post_cdf(u_val, log=True)
        Evaluate the posterior CDF.
    post_quantile(p, root_method='auto')
        Compute quantiles (inverse CDF).
    post_mgf(r_val, log=True)
        Evaluate the posterior MGF.
    post_raw_moment(q, numerator_method='auto', log=True)
        Compute raw moments.
    post_central_moment(order=None, log=True, numerator_method='auto')
        Compute central moments (1,2,3,4).
    post_interval(level=0.95, root_method='auto')
        Compute credible intervals.
    post_sample(n=None, u=None, root_method='auto')
        Generate posterior samples.
    post_predictive(new_data, log=True, individual=True)
        Compute posterior predictive density.
    to_prior_object()
        Convert the posterior to a prior object for sequential updating.
    update(new_data, **kwargs)
        Perform a sequential Bayesian update.

    Notes
    -----
    The class supports both symbolic and numeric evaluation, following the
    **symbol‑numeric principle**: the return type of methods depends on
    whether free symbols remain. Numeric methods are vectorised for array
    inputs (tuple‑vectorisation for `(t, u)` pairs).

    Examples
    --------
    >>> from jumufraktiv.mitMGFprior_class import mitMGFprior
    >>> from jumufraktiv.MGFDerivative_class import MGFDerivative
    >>> gamma_prior = mitMGFprior.from_registry('gamma', params={'alpha':2.0, 'beta':3.0})
    >>> deriv = MGFDerivative(gamma_prior, data=[1.0, 2.0], likelihood='poisson', scale=1.0)
    >>> log_ev, sign = deriv.evidence()
    >>> print(log_ev)
    """
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
        self._ready_kwargs, self._deriv_kwargs = _split_kwargs(
            kwargs, self.ready_func, self.likelihood
        )

        # ----------------------------------------------------
        # Sufficient statistics
        # ----------------------------------------------------
        stats = self.ready_func(data, **self._ready_kwargs)
        self.a = stats['a']
        self.b = stats['b']
        self.log_c = stats['log_c']

        # The posterior is evaluated at t = -b. When b == 0 that is the origin,
        # where D^a M(0) = E[Theta^a] and the moment must exist. Checked here
        # rather than only in mgfDerivative because _build_derivative passes
        # t=None, so the dispatcher never sees the concrete evaluation point.
        _check_moment_exists_at_origin(self.a, self.prior, -self.b)

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
    def _likelihood_arguments(self, **overrides):
        """The likelihood's known parameters, with any caller overrides applied.

        The parameters a likelihood needs -- ``mean`` for normal, ``shape`` for
        gamma, ``rho`` for weibull -- are supplied once at construction and
        stored in ``_ready_kwargs``. Anything that later calls ``ready_func`` or
        ``bereit_func`` needs them again, because those functions take them as
        required positional arguments.

        ``post_predictive`` used to forward only the *caller's* keyword
        arguments, so ten of the fourteen likelihoods raised
        ``TypeError: bereitNormal() missing 1 required positional argument:
        'mean'`` for a parameter the object was already holding. The four that
        worked are exactly the four with no known parameters, which is why the
        suite never saw it: every predictive test uses Poisson, whose one
        parameter has a default.

        Overrides are honoured rather than ignored, so a caller can still
        evaluate the predictive under a different known parameter than the one
        the posterior was built with -- passing ``rho`` for a new observation
        from a different Weibull shape, say.
        """
        merged = dict(self._ready_kwargs)
        merged.update(overrides)
        return merged

    def _build_derivative(self):
        """
        Construct a representation of ``D^a M(t)`` that is not yet tied to a `t`.

        Only the ``symbolic`` backend can produce such a representation: it
        differentiates the prior's MGF and hands back a `sympy.Expr` in `t`,
        which later methods substitute into as often as they like. Every
        numeric backend quadratures at a *particular* `t` and so has nothing to
        return until one is known.

        For those, this method stores a callable instead — a thunk that runs
        the same `mgfDerivative` call once an evaluation point arrives. The
        arrangement is what `_evaluate_derivative`'s callable branch has always
        expected; until now nothing produced a callable, so the branch was
        unreachable and every numeric backend raised
        ``ValueError: t must be provided`` during construction.

        Notes
        -----
        Which case applies is asked of `resolve_backend`, so this method does
        not need to know the backend matrix — that stays in the dispatch layer,
        as the layer rule requires.
        """
        _, resolved_method, _ = resolve_backend(
            self.a,
            self.method,
            self._deriv_kwargs.get("integer_method", "symbolic"),
            self._deriv_kwargs.get("int_tol", 1e-12),
            prior=self.prior,
        )

        if resolved_method == "symbolic":
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
        else:
            self._deriv = self._deferred_derivative()

        # Do I have an unevaluated symbolic derivative function representation
        # DM(t)? It tells you whether you can work with it symbolically.
        self._deriv_is_symbolic = isinstance(self._deriv, sp.Expr)

    def _deferred_derivative(self):
        """Return a thunk evaluating ``D^a M`` at a `t` supplied later.

        The thunk closes over the backend options, so callers pass only the
        evaluation point. Re-passing ``**self._deriv_kwargs`` at call time —
        which `_evaluate_derivative` used to do — would raise ``TypeError: got
        an unexpected keyword argument`` for any option the closure already
        holds.

        It is built with ``log=self.log`` rather than a fixed ``log=False``:
        `_store_result` requires a ``(log_abs, sign)`` tuple whenever
        ``self.log`` is set, and the log principle makes that choice the
        caller's alone.
        """
        options = dict(self._deriv_kwargs)

        def deferred(t_value, u=None, complete=True, log=None):
            return mgfDerivative(
                order=self.a,
                prior=self.prior,
                method=self.method,
                t=t_value,
                u=u,
                simplify=self.simplify,
                complete=complete,
                log=self.log if log is None else log,
                **options
            )

        return deferred

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
            # Deferred numeric backend: the thunk already holds the options.
            return self._deriv(t_value)


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
        Return the marginal likelihood (evidence) of the model.

        The marginal likelihood is the normalising constant of the posterior
        distribution, i.e., p(y) = ∫ p(y|θ) p(θ) dθ.

        Returns
        -------
        sympy.Expr or tuple or float
            - If the posterior is symbolic (`self._is_symbolic is True`):
                Returns a SymPy expression for the evidence.
            - If the posterior is numeric:
                - If `self.log is True`: returns a tuple `(log_abs, sign)` where
                `log_abs` is the natural logarithm of the absolute evidence,
                and `sign` is the sign (±1) of the evidence.
                - If `self.log is False`: returns the ordinary float value of the
                evidence.

        Notes
        -----
        The evidence is computed as:
            evidence = c_func() * D^a M(t) |_{t=-b}
        where `c_func` is the likelihood's normalising constant and the
        derivative is evaluated at the posterior mode location `t = -b`.

        Examples
        --------
        >>> evidence_log, sign = deriv.evidence()
        >>> print(f"log evidence = {evidence_log:.4f}, sign = {sign}")
        """

        if self._is_symbolic:
            # `sp.exp(self.log_c)`, not `self.c_func()`.
            #
            # `c_func()` returns the likelihood's normalising constant as a
            # general formula -- for Poisson,
            # `Product(s[i]**y[i]/factorial(y[i]), (i, 1, n))` -- written over
            # indexed symbols that are never bound to this object's data. The
            # symbolic evidence therefore came back carrying free `n`, `s` and
            # `y` and an unevaluated `Product`, which no substitution available
            # to the caller could resolve.
            #
            # `log_c` is the same constant already evaluated on the actual data
            # by the likelihood's `ready` function, so this returns an
            # expression whose only free symbols are the prior's
            # hyperparameters -- which the caller can substitute. It also makes
            # the symbolic and numeric branches agree on where the constant
            # comes from, which they did not before.
            return sp.exp(self.log_c) * self._result_expr

        if self.log:
            return self.log_c + self.log_abs, self._sign

        return np.exp(self.log_c) * self.value

    # ========================================================
    # POSTERIOR DENSITY
    # ========================================================
        
    def post_density(self, theta_val=None, log=True):
        """
        Compute the posterior density (or log-density) at given θ.

        This method respects the **symbol‑numeric principle**: the return type
        depends only on whether unresolved symbols remain, not on the path taken.

        - If `theta_val` is a SymPy symbol (canonically :data:`jumufraktiv.symbols.theta`),
        or if the derivative expression still contains free symbols, a symbolic
        expression is returned.
        - If `theta_val` is numeric (scalar or array), the expression is evaluated
        numerically, respecting vectorisation.

        Parameters
        ----------
        theta_val : scalar, array-like, or sympy Symbol, optional
            Evaluation point(s). If array-like, must be convertible to a NumPy array.
            If a SymPy Symbol, the canonical symbol :data:`jumufraktiv.symbols.theta`
            is typically used.
        log : bool, optional
            If True, return the log-density; otherwise the ordinary density.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            - Scalar float for scalar numeric input.
            - NumPy array for array‑like input.
            - SymPy expression if symbolic evaluation is requested or free symbols remain.

        Notes
        -----
        The posterior density is computed as:
            p(θ | y) = p(θ) * θ^a * exp(-b θ) / D^a M(t) |_{t=-b}
        where `D^a M` is the fractional derivative of the prior MGF.

        For vectorised symbolic input, all elements must evaluate to a pure numeric
        result; if any free symbols remain, a `ValueError` is raised. For scalar
        symbolic input, the expression is returned directly.

        Examples
        --------
        >>> # Numeric evaluation at a single point
        >>> deriv.post_density(0.2, log=False)
        1.234567e+00

        >>> # Vectorised evaluation on a grid
        >>> theta_grid = np.linspace(0.0, 1.0, 100)
        >>> log_dens = deriv.post_density(theta_grid, log=True)

        >>> # Symbolic evaluation using the canonical symbol
        >>> from jumufraktiv.symbols import theta
        >>> expr = deriv.post_density(theta, log=False)  # SymPy expression
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
                log_num = log_prior + self.a * sp.log(theta_sym) - self.b * theta_sym
                log_post = log_num - sp.log(denom_expr)

                # Substitute the known hyperparameters into the ASSEMBLED
                # expression. Two things were wrong before. The substitution was
                # applied to `pdf_sym` after `log_prior` had already been formed
                # from it, so it was computed and discarded; and even correctly
                # ordered it would have reached only the prior's density, while
                # the normalising constant `denom_expr` -- the derivative of the
                # prior MGF -- carries the same hyperparameters. Substituting
                # once, here, covers both. This mirrors `post_cdf`, which always
                # substituted into its assembled expression.
                if self.params is not None:
                    subs_dict = {sym: self.params[sym.name]
                                 for sym in log_post.free_symbols
                                 if sym.name in self.params}
                    if subs_dict:
                        log_post = log_post.subs(subs_dict)

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

        theta_arr = np.asarray(theta_arr, dtype=float)

        # ---- Domain guard ----------------------------------------------
        # `theta` is strictly positive, so the density is exactly 0 outside
        # the support. Answering that is better than computing `log(theta)`
        # for a non-positive `theta` and producing NaN, which is what happened:
        # `post_quantile` evaluates this as the derivative for its Newton step
        # and can propose a non-positive point, and the resulting
        # "invalid value encountered in log" is escalated to an exception by
        # the test suite's `filterwarnings = ["error"]` while ordinary users
        # see only a warning and a NaN. That difference between harness and
        # user is precisely the hazard CLAUDE.md records.
        in_support = theta_arr > 0.0
        theta_safe = np.where(in_support, theta_arr, 1.0)

        # Get log prior density (vectorized)
        if self.prior.logpdf_func is not None:
            log_prior = self.prior.logpdf_func(theta_safe)
        elif self.prior.pdf_func is not None:
            prior_pdf = self.prior.pdf_func(theta_safe)
            if np.any(prior_pdf <= 0):
                raise ValueError("Prior PDF must be positive for all theta values.")
            log_prior = np.log(prior_pdf)
        else:
            raise ValueError("No numeric PDF function available for this prior.")

        log_num = log_prior + self.a * np.log(theta_safe) - self.b * theta_safe
        log_num = np.where(in_support, log_num, -np.inf)
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

        This method respects the **symbol‑numeric principle**: the return type
        depends only on whether unresolved symbols remain.

        - If `u_val` is a SymPy symbol (canonically :data:`jumufraktiv.symbols.u`),
        or if the expression still contains free symbols, a symbolic expression
        is returned.
        - If `u_val` is numeric (scalar or array), the CDF is evaluated numerically,
        supporting tuple‑vectorisation: the evaluation point is the pair `(t, u)`,
        where `t = -self.b` is fixed and `u` is broadcast to match the input.

        Parameters
        ----------
        u_val : scalar, array-like, or sympy Symbol, optional
            Upper limit(s) for the CDF. If array-like, returns an array of CDFs.
            If a SymPy symbol, the canonical symbol :data:`jumufraktiv.symbols.u`
            is typically used.
        log : bool, optional
            If True, return the log-CDF; otherwise the ordinary CDF.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            - Scalar float for scalar numeric input.
            - NumPy array for array‑like input.
            - SymPy expression if symbolic evaluation is requested or free symbols remain.

        Raises
        ------
        RuntimeError
            If the prior does not support the incomplete MGF (iMGF) or if the
            CDF becomes negative (indicating a numerical issue).

        Notes
        -----
        The posterior CDF is computed as:
            F(u | y) = D^a M_inc(t; u) / D^a M(t) |_{t=-b}
        where `M_inc` is the incomplete MGF of the prior.

        Examples
        --------
        >>> # Numeric evaluation at a single point
        >>> deriv.post_cdf(0.2, log=False)
        0.8574

        >>> # Vectorised evaluation on a grid
        >>> u_grid = np.linspace(0.0, 1.0, 100)
        >>> log_cdf = deriv.post_cdf(u_grid, log=True)  # array of log-CDFs

        >>> # Symbolic evaluation using the canonical symbol
        >>> from jumufraktiv.symbols import u
        >>> expr = deriv.post_cdf(u, log=False)  # SymPy expression
        """
        # ---- Ensure iMGF support ----
        if not hasattr(self.prior, "has_iMGF") or not self.prior.has_iMGF():
            raise RuntimeError("Prior does not support incomplete MGF (iMGF).")

        # ---- Symbolic path ----
        #
        # This block used to be wrapped in `except Exception` and re-raised as
        # RuntimeError("Symbolic posterior CDF computation failed: ..."). That
        # wrapper is gone. It added nothing a caller could act on, and it was
        # actively harmful here: the block referenced two names, `t_sym` and
        # `u_sym`, that this module has never defined -- it imports `t` and `u`
        # -- so every call raised NameError and the wrapper reported it as a
        # failed computation rather than as the typo it was. A genuine failure
        # inside SymPy now reaches the caller with its own message and its own
        # traceback.
        if self._is_symbolic:
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
            num_expr = num_expr.subs(t, -self.b)
            denom_expr = self._deriv.subs(t, -self.b)
            log_cdf_expr = sp.log(num_expr) - sp.log(denom_expr)

            # Substitute known hyperparameters
            if self.params is not None:
                subs_dict = {sym: self.params[sym.name]
                            for sym in log_cdf_expr.free_symbols
                            if sym.name in self.params}
                if subs_dict:
                    log_cdf_expr = log_cdf_expr.subs(subs_dict)

            # Determine if u_val is symbolic or numeric
            if isinstance(u_val, sp.Symbol) or u_val is None:
                return log_cdf_expr if log else sp.exp(log_cdf_expr)

            # Numeric u: vectorise evaluation
            orig_shape = np.shape(u_val)
            u_flat = np.asarray(u_val).ravel()

            def eval_u(uu):
                expr_i = log_cdf_expr.subs(u, uu).evalf()
                if expr_i.free_symbols:
                    raise ValueError(
                        "Symbolic expression still has free symbols after "
                        f"substituting u: {sorted(map(str, expr_i.free_symbols))}. "
                        "Supply the remaining hyperparameters through the prior's "
                        "`params` so the expression can be evaluated numerically."
                    )
                return float(expr_i)

            results = np.vectorize(eval_u)(u_flat).reshape(orig_shape)
            if not log:
                results = np.exp(results)
            return results.item() if np.ndim(results) == 0 else results

        # ---- Numeric path (vectorised with tuple‑vectorisation) ----
        if u_val is None:
            raise ValueError("For numeric evaluation, u must be provided.")

        # Store original shape
        orig_shape = np.shape(u_val)
        u_arr = np.asarray(u_val, dtype=float)
        scalar_input = u_arr.ndim == 0
        if scalar_input:
            u_arr = np.array([float(u_val)])

        # ---- Domain guard ----------------------------------------------
        # `theta` is strictly positive, so F(u) = P(Theta <= u) is exactly 0
        # for u <= 0. That is a legitimate question with a known answer, not a
        # caller error, so it is answered rather than refused.
        #
        # It has to be answered *before* evaluation, not corrected afterwards.
        # Passing a non-positive u into the incomplete-MGF derivative below
        # either recursed until `RecursionError` (u = -1e-9) or returned a
        # log-CDF above zero -- a probability of 2.43 at u = -0.5 -- because
        # nothing downstream knows where the support starts.
        in_support = u_arr > 0.0
        if not np.any(in_support):
            log_ratio = np.full(u_arr.shape, -np.inf).reshape(orig_shape)
            if scalar_input:
                return float(log_ratio.item()) if log else 0.0
            return log_ratio if log else np.zeros(orig_shape)

        # Evaluate at a safe stand-in wherever u is out of support, then
        # overwrite. Substituting keeps one batched call rather than splitting
        # the array, which the tuple-vectorisation principle asks for.
        u_eval = np.where(in_support, u_arr, np.max(u_arr[in_support]))

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
            u=u_eval,
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

        # Replace the stand-in results with the exact answer below the support.
        # This happens before the sign check so that check sees only points
        # that were genuinely evaluated.
        log_ratio = np.where(in_support, log_ratio, -np.inf)
        sign_ratio = np.where(in_support, sign_ratio, 1.0)

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
            stats_new = self.ready_func([x], **self._likelihood_arguments(**kwargs))
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

        This method respects the **symbol‑numeric principle**: the return type
        depends only on whether unresolved symbols remain.

        - If `new_data` is a SymPy symbol, the posterior is symbolic, and a
        symbolic expression for the predictive density is returned.
        - If `new_data` is numeric (scalar or array), the predictive density
        is evaluated numerically, supporting vectorisation.

        The `individual` flag controls the aggregation:
        - `individual=True` (default): returns a density for each element of `new_data`
        (array or scalar).
        - `individual=False`: returns the joint density (product of individual
        densities) as a scalar.

        Parameters
        ----------
        new_data : scalar, array-like, or sympy.Symbol
            New observation(s). If array-like, each element is treated separately.
            If a SymPy symbol, the posterior must be symbolic (`self._is_symbolic` is True).
        log : bool, optional
            If True, return log-density; otherwise ordinary density.
        individual : bool, optional
            If True (default), return an array of densities for each new data point.
            If False, return the joint density (product of individual densities).
        **kwargs : additional arguments passed to the likelihood's `ready_func` or `bereit_func`.
            For example, `scale` for Poisson or `shape` for Gamma.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            - If `new_data` is a scalar and numeric: returns a Python float.
            - If `new_data` is array-like: returns a NumPy array of the same length.
            - If `new_data` is a SymPy symbol: returns a SymPy expression.

        Notes
        -----
        The posterior predictive density for a single new observation is:
            p(y_new | y) = c(y_new) * D^{a(y_new)} M_{post}(t) |_{t = -b(y_new)}
        where `M_{post}` is the posterior MGF, and `c`, `a`, `b` are the likelihood
        statistics.

        For a symbolic posterior, the computation uses the original scalar symbolic
        logic; for array `new_data` with `individual=True`, it loops over elements
        using a scalar helper (`_post_predictive_symbolic_scalar`).

        Examples
        --------
        >>> # Numeric predictive density for a single new observation
        >>> deriv.post_predictive(14, scale=125.76, log=False)
        0.01234

        >>> # Vectorised predictive masses for multiple y values
        >>> y_vals = np.arange(0, 51)
        >>> log_pred = deriv.post_predictive(y_vals, scale=125.76, log=True)
        >>> pred_masses = np.exp(log_pred)  # array of masses

        >>> # Joint predictive density for two new observations
        >>> joint_log = deriv.post_predictive([14, 15], scale=125.76, log=True, individual=False)

        >>> # Symbolic predictive mass for a new observation
        >>> from sympy import Symbol
        >>> y_sym = Symbol('y', integer=True, positive=True)
        >>> expr = deriv_sym.post_predictive(y_sym, scale=125.76, log=False)
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
                # Joint density: the helper aggregates the statistics itself.
                # An identical `ready_func` call used to sit here, its result
                # discarded on the next line; it is gone rather than merely
                # rewritten, since forwarding the stored parameters into a
                # computation nothing reads would only make the waste tidier.
                return self._post_predictive_symbolic_scalar(new_data_arr, log=log, **kwargs)

        # ---- Numeric path (non‑symbolic posterior, vectorised) ----
        new_data_arr = np.asarray(new_data).ravel()
        scalar_input = len(new_data_arr) == 1

        if individual:
            stats = self.bereit_func(
                new_data_arr, **self._likelihood_arguments(**kwargs)
            )
            a_vals = np.asarray(stats['a']).ravel()
            b_vals = np.asarray(stats['b']).ravel()
            log_c_vals = np.asarray(stats['log_c']).ravel()
        else:
            stats = self.ready_func(
                new_data_arr, **self._likelihood_arguments(**kwargs)
            )
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
        Compute the posterior moment‑generating function (MGF) at given r.

        This method respects the **symbol‑numeric principle**: the return type
        depends only on whether unresolved symbols remain.

        - If `r_val` is a SymPy symbol (canonically :data:`jumufraktiv.symbols.r`),
        or if the expression still contains free symbols, a symbolic expression
        is returned.
        - If `r_val` is numeric (scalar or array), the MGF is evaluated numerically,
        supporting vectorisation. The evaluation point is `r` (complete MGF only).

        Parameters
        ----------
        r_val : scalar, array-like, or sympy.Symbol
            Evaluation point(s). If array-like, must be convertible to a NumPy array.
            If a SymPy symbol, the canonical symbol :data:`jumufraktiv.symbols.r`
            is typically used.
        log : bool, optional
            If True, return the log-MGF; otherwise the ordinary MGF.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            - Scalar float for scalar numeric input.
            - NumPy array for array‑like input.
            - SymPy expression if symbolic evaluation is requested or free symbols remain.

        Notes
        -----
        The posterior MGF is computed as:
            M_{post}(r) = D^a M(t) / D^a M(t) |_{t = r - b}  /  (t = -b)
        where `D^a M` is the fractional derivative of the prior MGF.

        Examples
        --------
        >>> # Numeric evaluation at a single point
        >>> deriv.post_mgf(0.2, log=False)
        1.234567e+00

        >>> # Vectorised evaluation on a grid
        >>> r_grid = np.linspace(-1.0, 1.0, 100)
        >>> log_mgf = deriv.post_mgf(r_grid, log=True)  # array of log-MGFs

        >>> # Symbolic evaluation using the canonical symbol
        >>> from jumufraktiv.symbols import r
        >>> expr = deriv.post_mgf(r, log=False)  # SymPy expression
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
        Compute the posterior raw moment of order q.

        This method respects the **symbol‑numeric principle**: the return type
        depends only on whether unresolved symbols remain.

        - If `q` is a SymPy symbol (canonically :data:`jumufraktiv.symbols.q`),
        or if the expression still contains free symbols, a symbolic expression
        (or list of expressions for array `q`) is returned.
        - If `q` is numeric (scalar or array), the moment is evaluated numerically,
        supporting vectorisation.

        For integer orders 1–4, the computation is typically fast; for higher
        orders, a warning is emitted as the calculation may be slow.

        Parameters
        ----------
        q : scalar, array-like, or sympy.Symbol
            Moment order(s). If array-like, returns an array of results.
        numerator_method : str, optional
            Method for derivative computation (passed to `mgfDerivative`).
        log : bool, optional
            If True, return the log‑moment; otherwise the ordinary moment.

        Returns
        -------
        scalar, np.ndarray, or sympy.Expr
            - Scalar float for scalar numeric input.
            - NumPy array for array‑like input.
            - SymPy expression if symbolic evaluation is requested or free symbols remain.

        Notes
        -----
        The posterior moment is computed as:
            E[Θ^q | y] = D^{a+q} M(t) / D^a M(t) |_{t=-b}
        where `D^a M` is the fractional derivative of the prior MGF.

        Examples
        --------
        >>> # Numeric moment of order 2
        >>> deriv.post_raw_moment(2, log=False)
        0.1234

        >>> # Vectorised moments for multiple orders
        >>> q_vals = np.array([1, 2, 3, 4])
        >>> log_moments = deriv.post_raw_moment(q_vals, log=True)
        >>> print(log_moments)  # array of log moments

        >>> # Symbolic moment using the canonical symbol
        >>> from jumufraktiv.symbols import q as q_sym
        >>> expr = deriv.post_raw_moment(q_sym, log=False)  # SymPy expression
        """
        # ---- Determine if q is array-like ----
        if hasattr(q, '__len__') and not isinstance(q, (str, bytes, sp.Basic)):
            q_arr = np.asarray(q)
            is_array = True
        else:
            is_array = False

        # ---- Warn if any high-order (not in {1,2,3,4}) ----
        high_order = (
            any(qi not in (1, 2, 3, 4) for qi in q_arr)
            if is_array
            else q not in (1, 2, 3, 4)
        )
        if high_order:
            # stacklevel=2 so the warning names the caller's line rather than
            # this one, which is the only location a user can act on.
            warnings.warn(
                "computing high-order posterior moments can be very slow",
                RuntimeWarning,
                stacklevel=2,
            )

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
        orders = self.a + q_arr if is_array else self.a + q

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
        Compute central moments of the posterior distribution.

        Supported orders are 1 (mean), 2 (variance), 3 (skewness-related), and
        4 (kurtosis-related). The method respects the **symbol‑numeric principle**:
        if the raw moments are symbolic, the central moment will be a SymPy
        expression; otherwise it is numeric.

        Parameters
        ----------
        order : int or list of ints, optional
            Central moment order(s). If `None` (default), computes all four
            moments (1, 2, 3, 4). If an integer, returns a single result.
            If a list, returns a dictionary mapping each order to its result.
        log : bool, optional
            If True, return `(log_abs, sign)` for each central moment, where
            `log_abs` is `log(|central moment|)` and `sign` is ±1.
            If False, return the ordinary central moment (float or sympy.Expr).
        numerator_method : str, optional
            Method for computing the numerator derivative in raw moments.
            Passed to `post_raw_moment`.

        Returns
        -------
        If `order` is an integer:
            - If `log=True`: a tuple `(log_abs, sign)`.
            - If `log=False`: a float or `sympy.Expr`.
        If `order` is `None` or a list:
            - A dictionary `{order: result}` where each result is as above.

        Notes
        -----
        The central moment of order `k` is computed using the binomial expansion:
            μ_k = Σ_{j=0}^k C(k, j) * μ'_j * (-μ_1)^{k-j}
        where `μ'_j` are the raw moments. This method currently only supports
        orders 1, 2, 3, and 4.

        Examples
        --------
        >>> # Single central moment (variance)
        >>> log_abs, sign = deriv.post_central_moment(order=2, log=True)
        >>> print(f"log variance = {log_abs}, sign = {sign}")

        >>> # All four central moments (ordinary scale)
        >>> moments = deriv.post_central_moment(log=False)
        >>> print(moments[1])  # mean
        >>> print(moments[2])  # variance

        >>> # Symbolic central moment
        >>> from sympy import Symbol
        >>> # (assuming deriv_sym is a symbolic derivative object)
        >>> expr = deriv_sym.post_central_moment(order=2, log=False)  # SymPy expression
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

        This method numerically inverts the posterior CDF `F(u) = p` using
        vectorised root‑finding. It supports both scalar and array inputs for `p`,
        and automatically finds a valid bracketing interval if not supplied.

        Parameters
        ----------
        p : float or array-like
            Probabilities (must be strictly between 0 and 1). If array‑like,
            returns quantiles of the same shape.
        root_method : str, optional
            Root‑finding method passed to `solve_root`. Options include:
            - `"auto"` (default): tries JAX methods first, then NumPy fallbacks.
            - `"bisectioned-newton-np"`, `"newton-np"`, `"bisection-np"` (NumPy).
            - `"bisectioned-newton-jax"`, `"newton-jax"`, `"bisection-jax"` (JAX).
            See `jumufraktiv.root_finding.solve_root` for full list.
        lower, upper : array-like, optional
            Search interval bounds. If not provided, an automatic expansion
            from `1e-6` to `1e6` is performed until `CDF(lower) < p < CDF(upper)`.
        x0 : array-like, optional
            Initial guess for Newton‑based methods. If `None`, uses the midpoint
            of the bracket.
        maxiter : int, optional
            Maximum number of root‑finding iterations.
        tol : float, optional
            Absolute tolerance for `|CDF(x) - p|`.
        rel_tol : float, optional
            Relative tolerance for change in `x` (Newton methods only).
        **kwargs : additional keyword arguments passed to `solve_root`.

        Returns
        -------
        scalar or np.ndarray
            Quantiles such that `CDF(quantile) = p`. The shape matches `p`.

        Notes
        -----
        - This method is purely numeric and does not support symbolic evaluation.
        - The CDF and density functions are called on arrays, so the computation
        is fully vectorised over the elements of `p`.
        - If the automatic bracket expansion fails, you can provide explicit
        `lower` and `upper` bounds to improve robustness.

        Examples
        --------
        >>> # 95% quantile (single value)
        >>> q = deriv.post_quantile(0.95)
        >>> print(f"95% quantile = {q:.4f}")

        >>> # Multiple quantiles at once
        >>> p_vals = np.array([0.025, 0.5, 0.975])
        >>> quantiles = deriv.post_quantile(p_vals)
        >>> print(quantiles)  # array of three quantiles

        >>> # Using a specific root method and providing a bracket
        >>> q = deriv.post_quantile(0.5, root_method='bisection-np', lower=0.0, upper=1.0)
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
            # Start from the posterior's own scale, not from a fixed 1e-6.
            #
            # The old bracket ran 1e-6 to 1e6, and the lower end is not merely
            # wasteful -- it is outside the region where the incomplete-MGF
            # derivative is accurate. Measured against the exact Gamma(8, 6)
            # posterior of the canonical test problem, the numerator's log at
            # u = 1e-6 comes out as -65.17 where the true value is -110.41: a
            # 45-nat error, with the computed sign flipped negative, which then
            # trips the non-negativity guard in `post_cdf`. It is already wrong
            # by 17.8 nats at u = 1e-4, and only becomes trustworthy around
            # u = 1e-2. So every call failed before the root finder started.
            #
            # The likelihood's own statistics give a scale for free: the
            # posterior is proportional to `theta**a * exp(-b*theta)` times the
            # prior, whose density peaks at `a/b` and has mean `(a+1)/b`. That
            # is the right order of magnitude for the quantiles of interest and
            # sits well inside the accurate region, so the expansion below
            # reaches the root without ever probing where the CDF is unusable.
            scale = (self.a + 1.0) / self.b if self.b > 0 else 1.0
            lower_init = np.full_like(p_arr, scale * 0.5, dtype=float)
            upper_init = np.full_like(p_arr, scale * 2.0, dtype=float)
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

        # ---- Verify the root, and fall back to bisection if it is not one --
        #
        # `solve_root`'s "auto" mode tries methods in order and accepts the
        # first that does not raise -- but a non-finite or out-of-bracket
        # result is not an exception, so a diverged Newton step counts as
        # success. That is how `post_quantile(0.025)` came back as 7.7e+300
        # while `post_quantile` on the same probability inside an array came
        # back correct: different methods in the ordering won.
        #
        # A bracket is already in hand and the CDF is monotone, so bisection
        # cannot fail here. Verifying and falling back costs one cheap check on
        # the common path and removes the failure mode entirely.
        roots = np.asarray(roots, dtype=float)
        acceptable = np.isfinite(roots) & (roots >= lower) & (roots <= upper)
        if not np.all(acceptable):
            roots = np.asarray(
                solve_root(
                    f=f,
                    df=None,
                    x0=None,
                    lower=lower,
                    upper=upper,
                    root_method="bisection-np",
                    maxiter=maxiter,
                    tol=tol,
                    rel_tol=rel_tol,
                    **kwargs
                ),
                dtype=float,
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

        This method draws samples from the posterior distribution by applying
        the inverse CDF (`post_quantile`) to uniform random numbers. It is
        fully vectorised and can generate large samples efficiently.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate. Ignored if `u` is provided.
        u : array-like, optional
            Uniform random numbers in (0, 1). If provided, `n` is ignored.
            These are used as the probabilities for `post_quantile`.
        root_method : str, optional
            Root‑finding method passed to `post_quantile`. See
            `post_quantile` and `solve_root` for options.
        **kwargs : additional keyword arguments passed to `post_quantile`
            (e.g., `maxiter`, `tol`, `lower`, `upper`).

        Returns
        -------
        np.ndarray
            Samples from the posterior distribution. The shape is `(n,)` if
            `n` is provided, or the shape of `u` if `u` is provided.

        Notes
        -----
        The method uses the inverse transform:
            X = F^{-1}(U), where U ~ Uniform(0, 1).
        This is exact up to the numerical accuracy of the quantile computation.

        Examples
        --------
        >>> # Generate 1000 posterior samples
        >>> samples = deriv.post_sample(n=1000)

        >>> # Use custom uniform variates for reproducibility
        >>> rng = np.random.default_rng(42)
        >>> u = rng.random(500)
        >>> samples = deriv.post_sample(u=u)

        >>> # Control the root‑finding method and tolerance
        >>> samples = deriv.post_sample(n=100, root_method='bisection-np', tol=1e-10)
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
        Compute central (equal-tailed) credible intervals for the posterior.

        For a given credible level `α`, the interval is defined as:
            [F^{-1}( (1-α)/2 ),  F^{-1}( (1+α)/2 ) ]
        where `F` is the posterior CDF. This gives the central interval that
        contains the middle `α` fraction of the posterior mass.

        Parameters
        ----------
        level : float or array-like, optional (default 0.95)
            Credible level(s). Must be in (0, 1). If array‑like, returns
            intervals for each level.
        root_method : str, optional
            Root‑finding method passed to `post_quantile`. See `post_quantile`
            and `solve_root` for options.
        **kwargs : additional keyword arguments passed to `post_quantile`
            (e.g., `maxiter`, `tol`, `lower`, `upper`).

        Returns
        -------
        If `level` is scalar:
            tuple (lower, upper) where both are Python floats.
        If `level` is array‑like:
            np.ndarray of shape `(len(level), 2)` where each row is `[lower, upper]`.

        Notes
        -----
        - The method is fully vectorised: quantiles for all levels are computed
        in a single call to `post_quantile`.
        - The intervals are "equal-tailed", meaning the same probability mass
        is left in each tail.

        Examples
        --------
        >>> # 95% credible interval (single)
        >>> lower, upper = deriv.post_interval(level=0.95)
        >>> print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

        >>> # Multiple intervals at once
        >>> levels = np.array([0.68, 0.95, 0.99])
        >>> intervals = deriv.post_interval(level=levels)
        >>> for level, (l, u) in zip(levels, intervals):
        ...     print(f"{level*100:.0f}% CI: [{l:.4f}, {u:.4f}]")

        >>> # Using a specific root method
        >>> lower, upper = deriv.post_interval(level=0.9, root_method='bisection-np')
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
        Convert the current posterior distribution into a prior object.

        This method constructs a :class:`mitMGFprior` object that can be used
        as a prior in a subsequent sequential update. It first attempts to create
        a **symbolic** prior by extracting the symbolic MGF and PDF from the
        current posterior (via `post_mgf` and `post_density`). The posterior MGF
        is expressed in terms of the canonical variable `r` and then substituted
        with `t` to become a prior MGF. The posterior PDF is expressed in terms
        of the canonical `theta`.

        If symbolic construction fails (e.g., the posterior is numeric or the
        symbolic expression cannot be formed), it falls back to a **numeric**
        backend prior that wraps the numeric `post_mgf` and `post_density` methods.

        Returns
        -------
        mitMGFprior
            A prior object representing the posterior distribution. The prior's MGF
            and PDF are derived from the current posterior.

        Notes
        -----
        - This method uses the canonical symbols `r`, `t`, and `theta` from
        `jumufraktiv.symbols` for symbolic manipulation.
        - The symbolic route requires that the posterior derivative is symbolic
        (`self._deriv_is_symbolic is True`).
        - The numeric fallback works for any posterior, but the resulting prior
        cannot be used in symbolic computations.

        Examples
        --------
        >>> # Convert a numeric posterior to a prior for sequential updating
        >>> post_prior = deriv.to_prior_object()
        >>> new_deriv = post_prior.update(new_data, likelihood='poisson')
        """

        if self._deriv_is_symbolic:
            # No blanket `except Exception` here any more. It caught every
            # failure, printed "Symbolic construction failed:" with a traceback,
            # and fell through to the numeric route -- so a genuine bug in the
            # symbolic path surfaced as a silently different kind of prior,
            # announced only on stdout. A failure here now propagates.
            mgf_sym_expr = self.post_mgf(r, log=False)
            mgf_sym_expr = mgf_sym_expr.subs(r, t)  # posterior MGF of r becomes the prior MGF of t
            pdf_sym_expr = self.post_density(theta, log=False)  # use canonical theta

            if isinstance(mgf_sym_expr, sp.Expr) and isinstance(pdf_sym_expr, sp.Expr):
                return mitMGFprior(
                    name="posterior_prior_symbolic",
                    mgf_sym=mgf_sym_expr,
                    pdf_sym=pdf_sym_expr,
                    params=self.params
                ).as_mitMGFprior()

        # ---- Backend (numeric) route ----
        #
        # `log=False`, not `log=self.log`. These are the prior's *density* and
        # *MGF*; a container expecting p(theta) was being handed log p(theta)
        # whenever the posterior stored its constant in log scale, which is the
        # default. The result was a prior whose "density" went negative --
        # measured at theta = 1.0 it returned -3.14 where the density is 0.0432
        # -- and nothing downstream could detect that, because a log density is
        # a perfectly well-formed float.
        def mgf_backend(t_val, xp=np, **params):
            return self.post_mgf(t_val, log=False)

        def pdf_backend(theta_val, xp=np, **params):
            return self.post_density(theta_val, log=False)

        return mitMGFprior(
            name="posterior_prior",
            mgf_backend=mgf_backend,
            pdf_backend=pdf_backend,
            params=self.params
        ).as_mitMGFprior()

    def update(self, new_data, **kwargs):
        """
        Perform a sequential Bayesian update.

        This method treats the current posterior as the prior for a new dataset
        and returns a new :class:`MGFDerivative` object for the updated posterior.
        It uses `to_prior_object` to create a prior from the current posterior,
        then constructs a new derivative object with the new data.

        Parameters
        ----------
        new_data : array‑like
            New observations to condition on. Must be compatible with the likelihood's
            `ready_func` or `bereit_func`.
        **kwargs : additional keyword arguments
            - method : str, optional
                Derivative backend for the new object. Defaults to the current method.
                If 'symbolic' is chosen but the current posterior is numeric,
                an error is raised.
            - likelihood : str, optional
                Likelihood name. Defaults to the current likelihood.
            - simplify : bool, optional
                Whether to simplify symbolic expressions. Defaults to current setting.
            - log : bool, optional
                Whether to store the normalising constant in log‑scale.
                Defaults to current setting.
            - Other arguments are passed to the new `MGFDerivative` constructor.

        Returns
        -------
        MGFDerivative
            A new derivative object representing the posterior after conditioning
            on `new_data`.

        Raises
        ------
        ValueError
            If `method='symbolic'` is requested but the current posterior is
            numeric (i.e., `self._is_symbolic is False`). This is because the
            posterior prior would be numeric and cannot be used symbolically.

        Notes
        -----
        - This method is the primary way to perform sequential Bayesian updating.
        - The new object's `log` parameter can differ from the current one,
        allowing you to switch between log‑scale and ordinary‑scale storage
        at each update.

        Examples
        --------
        >>> # Sequential update: add two new observations
        >>> deriv2 = deriv.update(new_data=[5, 7], likelihood='poisson', scale=1.0)
        >>> # Check the updated evidence
        >>> log_ev, sign = deriv2.evidence()
        """
        # Extract known arguments
        method = kwargs.pop("method", self.method)
        likelihood = kwargs.pop("likelihood", self.likelihood)
        simplify = kwargs.pop("simplify", self.simplify)
        log = kwargs.pop("log", self.log)

        # ----------------------------------------------------
        # Sequential updating, and why it now works numerically
        # ----------------------------------------------------
        # This used to refuse every numeric posterior. `to_prior_object` builds
        # the updated prior from this one, and its numeric route returns a prior
        # carrying `pdf_backend`/`mgf_backend` and no `mgf_sym`/`cgf_sym` --
        # which no *differentiating* backend can consume: `bell` raises "Prior
        # does not provide a symbolic CGF", `jax` raises inside its tracer, and
        # the quadrature backends returned -inf because a blanket
        # `except Exception: return 0.0` turned the missing MGF into a zero at
        # every node. Since no symbolic backend serves fractional orders, that
        # ruled out sequential updating for fractional posteriors entirely.
        #
        # The direct-expectation route removes the obstacle rather than working
        # around it. It computes `E[theta^a e^{t theta}]` from the prior's
        # DENSITY and never touches its MGF, so the numeric prior above is
        # exactly what it needs.
        #
        # What the guard below tests is `self._deriv_is_symbolic` -- a property
        # of *this* posterior, not of the prior about to be built -- because
        # that is precisely what decides which kind of prior comes out:
        # symbolic in, symbolic prior out, which every backend can consume;
        # numeric in, density-only prior out, which only the expectation route
        # can. Testing it here avoids constructing the prior twice.
        #
        # Every DIFFERENTIATING backend needs the prior's symbolic MGF or CGF,
        # and the prior built from a numeric posterior has only a density. So
        # an explicit `method=` naming one of them still cannot update -- that
        # is a limit of those backends, not of updating, and it is reported
        # here rather than surfacing as "Prior does not provide a symbolic MGF"
        # from inside a quadrature or as a TracerArrayConversionError from JAX.
        #
        # `auto` and `expectation` both work, because the direct-expectation
        # route reads the density and never touches the MGF.
        requested = str(kwargs.get("method", method)).lower()
        differentiating = {"symbolic", "bell", "jax", "scipy", "mpmath"}
        if not self._deriv_is_symbolic and requested in differentiating:
            raise ValueError(
                f"method='{requested}' differentiates the prior's symbolic MGF "
                "or CGF, and the prior built from a numeric posterior has only "
                "a density. Use method='auto' or method='expectation', which "
                "compute the derivative as an expectation and need only the "
                "density, or build the original posterior with "
                "method='symbolic'."
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
