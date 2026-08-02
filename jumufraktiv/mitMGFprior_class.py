"""
mitMGFprior.py

Unified container for moment-generating function (MGF) priors.

This module defines the `mitMGFprior` class, which serves as a standardised
container for prior distributions in the MGF marginalisation framework.
It holds both symbolic and numeric representations of the prior MGF, CGF,
and PDF, along with JAX-compatible versions for fast computation.

Design philosophy:

- The registry (`PRIOR_REGISTRY`) provides fully-formed function bundles;
  the class only composes, stores, and exposes interfaces.

- Both symbolic and backend-based construction routes are supported.
- The class is dataclass-based for clarity and easy extension.

Key features:

- Supports **complete** and **incomplete** MGFs (iMGF) via the `imgf`, `logimgf`,
  `imgf_jax`, `logimgf_jax`, `imgf_sym`, and `logimgf_sym` attributes.

- Provides a `has_iMGF()` method to check if all iMGF components are present.
- Includes a factory method `from_registry` for automatic construction from
  the registry, and a manual compiler `as_mitMGFprior` for custom priors.

- Validation via `is_mitMGFprior` ensures a prior object is fully compiled.

Examples
--------
>>> # Build from registry
>>> gamma_prior = mitMGFprior.from_registry('gamma', params={'alpha':2.0, 'beta':3.0})
>>> gamma_prior.mgf(-1.0)  # numeric MGF
0.5625

>>> # Manual construction with symbolic expressions
>>> from jumufraktiv.symbols import t, theta
>>> prior = mitMGFprior(mgf_sym=(1 - t)**(-2), pdf_sym=theta*sp.exp(-theta))
>>> prior = prior.as_mitMGFprior()
>>> float(prior.mgf(-0.5))
0.4444444444444444
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import sympy as sp

from jumufraktiv.registry import PRIOR_REGISTRY, get_prior
from jumufraktiv.symbols import t, theta

# ============================================================
# Core container
# ============================================================

@dataclass
class mitMGFprior:
    """
    Container for a prior distribution's MGF, CGF, and PDF.

    This class holds both symbolic and numeric representations of a prior's
    moment-generating function (MGF), cumulant-generating function (CGF),
    and probability density function (PDF). It also supports JAX-compatible
    backends for high-performance computation and incomplete MGF (iMGF)
    functions for truncated distributions.

    Notes
    -----
    The class follows a two-step construction pattern:

    1. Create an instance with raw inputs (symbolic or backend).
    2. Call `as_mitMGFprior()` to compile and populate all functions.

    The registry route (`from_registry`) performs both steps automatically.

    The fields divide into three groups. The *raw* fields — `name`, `mgf_sym`,
    `pdf_sym`, `mgf_backend`, `pdf_backend`, `params` — are what a caller
    supplies. The two *domain* fields, `max_finite_moment` and
    `mgf_finite_below`, declare where the prior's moments and MGF exist. The
    rest are *compiled* and are `None` until construction fills them in. Both
    routes produce the six `is_mitMGFprior` requires — `mgf`, `cgf`, their
    `_jax` counterparts, `pdf_func` and `logpdf_func` — plus `cgf_sym`.

    One compiled field is set by the registry route only, and is `None` on a
    prior built by hand: `mgf_sym_out`, the MGF expression with `params`
    substituted. A hand-built prior keeps its own `mgf_sym` and needs no
    second copy.

    The incomplete MGF is a fourth group and behaves differently again.
    `imgf`, `logimgf`, their `_jax` counterparts and `imgf_sym`, `logimgf_sym`
    are *attached* by the registry route only for a prior that has one, rather
    than declared as fields and left `None`. They are therefore absent rather
    than empty, so test with `has_iMGF()` instead of reading an attribute that
    may not exist. All four registry priors have one; a hand-built prior has
    none, and loses `post_cdf`, `post_quantile`, `post_interval` and
    `post_sample` with it.

    Examples
    --------
    >>> # Manual symbolic prior
    >>> from jumufraktiv.symbols import t, theta
    >>> prior = mitMGFprior(mgf_sym=(1 - t)**(-2), pdf_sym=theta * sp.exp(-theta))
    >>> prior = prior.as_mitMGFprior()
    >>> prior.mgf(-0.5)
    0.4444444444444444

    >>> # Registry-based prior
    >>> gamma_prior = mitMGFprior.from_registry(
    ...     'gamma', params={'alpha':2.0, 'beta':3.0}
    ... )
    >>> gamma_prior.mgf(-1.0)
    0.5625
    """
    #: Name of the prior distribution. Set by `from_registry` to the registry
    #: key; `"custom"` for a prior built by hand.
    name: str = "custom"

    # -----------------------------
    # user inputs (raw layer)
    #
    # These carry `#:` comments rather than a NumPyDoc `Attributes` section in
    # the class docstring. Napoleon renders that section into `.. attribute::`
    # directives, which for a dataclass field collides with the one autodoc
    # already emits, and Sphinx warns about the duplicate.
    # -----------------------------

    # `as_mitMGFprior` has two modes and takes the pairs whole: either both
    # symbolic fields or both backend ones. Supplying one of a pair raises
    # rather than falling back, since a prior with an MGF and no density
    # cannot serve the default route and one with a density and no MGF cannot
    # serve any differentiating backend.

    #: Symbolic expression for the MGF, in the canonical symbol
    #: `jumufraktiv.symbols.t`. Required together with `pdf_sym` for the
    #: symbolic construction mode.
    mgf_sym: sp.Expr | None = None

    #: Symbolic expression for the density, in `jumufraktiv.symbols.theta`.
    #: Required together with `mgf_sym`. A density is needed in one form or
    #: the other because the default `auto` route computes `E[θ^a e^{tθ}]`
    #: directly and never reads the MGF.
    pdf_sym: sp.Expr | None = None

    #: Numeric MGF with signature ``(x, xp, **params)``, where `xp` is the
    #: array module to compute in — `numpy` or `jax.numpy`. Required together
    #: with `pdf_backend` for the backend construction mode.
    mgf_backend: Callable | None = None

    #: Numeric density, same signature as `mgf_backend` and required with it.
    pdf_backend: Callable | None = None

    #: Numeric hyperparameters, e.g. ``{'alpha': 2.0, 'beta': 3.0}``. Used two
    #: ways depending on how the prior was built: `from_registry` substitutes
    #: them into the prior module's symbolic expressions, while the backend
    #: route forwards them as ``**params`` to the callables supplied. Values
    #: must be finite real numbers; a free symbol is refused, because a prior
    #: carrying one produces `0` or `nan` from every derived quantity.
    params: dict[str, Any] | None = None

    #: Supremum of the orders `a` for which `E[Θ^a]` is finite, i.e. the moment
    #: domain. Only consulted when the evaluation point is `t = 0`, where
    #: `Dᵃ M(0) = E[Θ^a]` and the moment must exist. Everywhere else (`t < 0`)
    #: the exponential dominates any polynomial and no moment condition is
    #: needed — see "The operator" in CLAUDE.md.
    #:
    #: The bound is **strict**: order `a` is admissible iff `a < max_finite_moment`.
    #: Defaults to infinity, which is correct for any prior with all moments and
    #: is the safe default for a custom prior: it defers to the numerical result
    #: rather than pre-emptively rejecting.
    max_finite_moment: float = float("inf")

    #: Supremum of the `t` for which `M(t)` is finite — the MGF's radius of
    #: convergence, expressed as a right endpoint rather than a radius because
    #: the domain is one-sided here. `M(t)` is finite for every `t` strictly
    #: below it; whether the endpoint itself is attained varies by prior and is
    #: left to the prior's own arithmetic, which raises where it diverges.
    #:
    #: The package evaluates the prior at `t = -b(y) <= 0`, so this bound is
    #: never in the way of an ordinary posterior. It matters for `post_mgf`,
    #: whose evaluation point is `t = r - b` and therefore moves right as `r`
    #: grows: past the bound the posterior MGF does not exist, and an analytic
    #: expression evaluated there returns the value of the *formula* rather
    #: than of the MGF. `(beta/(beta-t))**alpha` at `t > beta` is positive
    #: whenever `alpha` is even, so the wrong answer looks like a right one.
    #:
    #: Defaults to infinity, the safe default for a custom prior in the same
    #: sense as `max_finite_moment`: it defers to the numerical result rather
    #: than pre-emptively rejecting.
    mgf_finite_below: float = float("inf")

    # -----------------------------
    # compiled outputs
    # -----------------------------
    mgf: Callable | None = None
    cgf: Callable | None = None

    mgf_jax: Callable | None = None
    cgf_jax: Callable | None = None

    pdf_func: Callable | None = None
    logpdf_func: Callable | None = None

    #: The MGF expression with `params` substituted, set by the registry route.
    #: `None` on a prior built by hand, which keeps its own `mgf_sym`.
    mgf_sym_out: Any = None

    #: `log M(t)`, set by both construction routes.
    cgf_sym: Any = None

    # ============================================================
    # USER ROUTE: manual construction compiler
    # ============================================================
    def as_mitMGFprior(self):
        """
        Compile the prior object from raw inputs into a fully functional prior.

        This method takes the raw symbolic or backend inputs stored in the instance
        (`mgf_sym`, `pdf_sym`, `mgf_backend`, `pdf_backend`) and compiles them into
        callable functions for MGF, CGF, PDF, and their JAX counterparts. It also
        populates the compiled attributes (`mgf`, `cgf`, `mgf_jax`, `cgf_jax`,
        `pdf_func`, `logpdf_func`).

        Two construction modes are supported:
            1. **Symbolic mode**: if both `mgf_sym` and `pdf_sym` are provided,
            they are lambdified to NumPy and JAX functions. The CGF is derived
            as `log(mgf_sym)`.
            2. **Backend mode**: if both `mgf_backend` and `pdf_backend` are provided,
            they are wrapped to accept a backend parameter (`xp=np` or `xp=jnp`),
            and the CGF and log-PDF are derived numerically.

        The method modifies the instance in place and returns it for chaining.

        Returns
        -------
        mitMGFprior
            The compiled prior object (self), with all callables populated.

        Raises
        ------
        ValueError
            If neither a valid symbolic pair nor a valid backend pair is provided,
            or if only one of a required pair is given.

        Notes
        -----

        - The symbolic mode requires both `mgf_sym` and `pdf_sym` to be SymPy
          expressions containing the canonical variable `t` (for MGF) and `theta`
          (for PDF).

        - The backend mode requires both `mgf_backend` and `pdf_backend` to be
          callables with signature `(x, xp, **params)` where `x` is the evaluation
          point and `xp` is either `numpy` or `jax.numpy`.

        - The `params` dictionary (if provided) is passed to the backend functions.

        Examples
        --------
        >>> # Symbolic mode
        >>> from jumufraktiv.symbols import t, theta
        >>> prior = mitMGFprior(
        ...     mgf_sym=(1 - t)**(-2),
        ...     pdf_sym=theta * sp.exp(-theta)
        ... )
        >>> prior = prior.as_mitMGFprior()
        >>> prior.mgf(-0.5)
        0.4444444444444444

        >>> # Backend mode
        >>> def mgf_backend(x, xp, **params):
        ...     return xp.exp(x)
        >>> def pdf_backend(x, xp, **params):
        ...     return xp.exp(-x)
        >>> prior = mitMGFprior(mgf_backend=mgf_backend, pdf_backend=pdf_backend)
        >>> prior = prior.as_mitMGFprior()
        >>> float(prior.mgf(0.0))
        1.0
        """
        # ----------------------------------------------------
        # CASE 1: symbolic input (both must be provided)
        # ----------------------------------------------------
        if self.mgf_sym is not None and self.pdf_sym is not None:

            self.cgf_sym = sp.log(self.mgf_sym)

            self.mgf = sp.lambdify(t, self.mgf_sym, modules="numpy")
            self.cgf = sp.lambdify(t, self.cgf_sym, modules="numpy")

            self.mgf_jax = sp.lambdify(t, self.mgf_sym, modules="jax")
            self.cgf_jax = sp.lambdify(t, self.cgf_sym, modules="jax")

            self.pdf_func = sp.lambdify(theta, self.pdf_sym, modules="numpy")
            self.logpdf_func = lambda x: np.log(self.pdf_func(x))

            return self

        # ----------------------------------------------------
        # CASE 2: backend input (both must be provided)
        # ----------------------------------------------------
        if self.mgf_backend is not None and self.pdf_backend is not None:

            params = self.params or {}

            def mgp_np(tval):
                return self.mgf_backend(tval, xp=np, **params)

            def mgf_jax_fn(tval):
                return self.mgf_backend(tval, xp=jnp, **params)

            self.mgf = mgp_np
            self.mgf_jax = mgf_jax_fn

            self.cgf = lambda tval: np.log(self.mgf(tval))
            self.cgf_jax = lambda tval: jnp.log(self.mgf_jax(tval))

            def pdf_math(x):
                return self.pdf_backend(x, xp=np, **params)

            self.pdf_func = pdf_math
            self.logpdf_func = lambda x: np.log(pdf_math(x))

            return self

        # ----------------------------------------------------
        # ERROR: missing required pairs
        # ----------------------------------------------------
        if self.mgf_sym is not None and self.pdf_sym is None:
            raise ValueError("Symbolic mode requires both mgf_sym and pdf_sym.")
        if self.pdf_sym is not None and self.mgf_sym is None:
            raise ValueError("Symbolic mode requires both mgf_sym and pdf_sym.")
        if self.mgf_backend is not None and self.pdf_backend is None:
            raise ValueError("Backend mode requires both mgf_backend and pdf_backend.")
        if self.pdf_backend is not None and self.mgf_backend is None:
            raise ValueError("Backend mode requires both mgf_backend and pdf_backend.")

        raise ValueError(
            "Must provide either (mgf_sym, pdf_sym) or (mgf_backend, pdf_backend)."
        )

    # ============================================================
    # REGISTRY ROUTE: automatic construction
    # ============================================================
    @classmethod
    def from_registry(cls, prior_name, params=None, simplify=False):
        """
        Build a fully compiled prior object from the registry.

        This factory method retrieves the prior specification from the global
        `PRIOR_REGISTRY` and constructs a `mitMGFprior` instance with all
        symbolic and numeric functions compiled. It automatically includes
        both complete and incomplete MGF (iMGF) functions if they are available
        in the registry.

        Parameters
        ----------
        prior_name : str
            Name of the prior distribution as registered in `PRIOR_REGISTRY`.
        params : dict, optional
            Numeric hyperparameters for the prior (e.g., `{'alpha':2.0, 'beta':3.0}`).
        simplify : bool, default False
            If True, simplify the symbolic expressions using SymPy.

        Returns
        -------
        mitMGFprior
            A fully compiled prior object with all callables populated.

        Raises
        ------
        ValueError
            If `prior_name` is not in the registry, or if the registry does not
            provide the required MGF and PDF functions.

        Notes
        -----

        - The registry entry must provide at least `mgf_sym`, `pdf_sym`, `mgf`,
          `cgf`, and `pdf_func`.

        - If iMGF functions (`imgf_sym`, `imgf`, `imgf_jax`, etc.) are present,
          they are also extracted and stored.

        - The method bypasses the manual `as_mitMGFprior` compiler and directly
          assigns the compiled functions to the object.

        Examples
        --------
        >>> gamma_prior = mitMGFprior.from_registry(
        ...     'gamma', params={'alpha':2.0, 'beta':3.0}
        ... )
        >>> gamma_prior.mgf(-1.0)
        0.5625

        >>> # With symbolic simplification
        >>> prior = mitMGFprior.from_registry(
        ...     'pareto', params={'alpha':0.5, 'xi':1.0}, simplify=True
        ... )
        """
        from jumufraktiv.registry import failed_prior_modules

        params = params or {}

        # ---------------------------------------------------------
        # Get the factory function and call it
        # ---------------------------------------------------------
        # `get_prior` initialises the registry, so it must be what the lookup
        # goes through: reading `PRIOR_REGISTRY` directly fails in a fresh
        # process unless some other registry function has already run, and
        # cannot tell a typo from an unpopulated registry. The name is only
        # read below to list what is available once the lookup has failed.
        try:
            factory = get_prior(prior_name)
        except KeyError as exc:
            message = (
                f"Unknown prior '{prior_name}'. Available: {sorted(PRIOR_REGISTRY)}"
            )
            failed = failed_prior_modules()
            if failed:
                details = "; ".join(
                    f"{module} ({type(err).__name__}: {err})"
                    for module, err in sorted(failed.items())
                )
                message += (
                    f". Note that {len(failed)} prior module(s) failed to import, "
                    f"so priors they define are missing from that list: {details}"
                )
            raise ValueError(message) from exc

        # Every registry factory freezes a SciPy distribution and evaluates its
        # density numerically, so each hyperparameter must be a finite number.
        # Sort the arguments into the three cases before raising, so a caller
        # who got two of them wrong hears about both.
        #
        # A SymPy object that carries no free symbol IS a number, and is
        # converted here rather than refused: `sp.Integer(2)` and `sp.Float(2.0)`
        # arise from ordinary SymPy arithmetic. Converting is what makes them
        # work everywhere -- passed through unconverted they build an
        # object-dtype array inside the Pareto factory and fail there with
        # "Cannot cast array data from dtype('O')", several frames from the
        # argument at fault.
        resolved: dict[str, float] = {}
        free_symbols: list[str] = []
        not_a_number: list[str] = []
        not_finite: list[str] = []

        for name, value in params.items():
            if isinstance(value, sp.Basic) and value.free_symbols:
                free_symbols.append(name)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                # `sp.zoo` (complex infinity) has no float at all, and neither
                # does a string. Both land here rather than in `not_finite`,
                # which would report them as numbers of the wrong size.
                not_a_number.append(f"{name} ({type(value).__name__})")
                continue
            if not math.isfinite(number):
                not_finite.append(name)
                continue
            resolved[name] = number

        if free_symbols:
            names = ", ".join(sorted(free_symbols))
            raise TypeError(
                f"from_registry() needs numeric hyperparameters; {names} "
                f"{'is' if len(free_symbols) == 1 else 'are'} symbolic. "
                "Registry priors compile a numeric density, which a free "
                "symbol cannot supply. For a prior whose hyperparameters stay "
                "free, build it directly -- mitMGFprior(mgf_sym=..., "
                "pdf_sym=..., params={}).as_mitMGFprior() -- and substitute "
                "the values later, or pass them through `params` to have them "
                "resolved."
            )

        if not_a_number:
            names = ", ".join(sorted(not_a_number))
            raise TypeError(
                f"from_registry() needs numeric hyperparameters; {names} "
                f"cannot be converted to a float."
            )

        if not_finite:
            names = ", ".join(sorted(not_finite))
            raise ValueError(
                f"from_registry() needs finite hyperparameters; {names} "
                f"{'is' if len(not_finite) == 1 else 'are'} not. An infinite "
                "or undefined hyperparameter builds a density that integrates "
                "to zero or to nan, and every quantity derived from it "
                "inherits that silently rather than raising."
            )

        spec = factory(resolved)  # <-- this is the make_prior_spec dict

        # ---------------------------------------------------------
        # Extract symbolic forms from the spec
        # ---------------------------------------------------------
        mgf_sym = spec.get("mgf_sym")
        pdf_sym = spec.get("pdf_sym")
        imgf_sym = spec.get("imgf_sym")

        if mgf_sym is None or pdf_sym is None:
            raise ValueError("Registry must contain mgf_sym and pdf_sym")

        if simplify:
            mgf_sym = sp.simplify(mgf_sym)
            pdf_sym = sp.simplify(pdf_sym)
            imgf_sym = sp.simplify(imgf_sym) if imgf_sym is not None else None

        cgf_sym = sp.log(mgf_sym)
        logimgf_sym = sp.log(imgf_sym) if imgf_sym is not None else None

        # ---------------------------------------------------------
        # Extract backend functions from the spec
        # ---------------------------------------------------------
        # Math backend
        mgf_math = spec.get("mgf")
        cgf_math = spec.get("cgf")
        pdf_math = spec.get("pdf_func")
        logpdf_math = spec.get("logpdf_func")
        imgf_math = spec.get("imgf")
        logimgf_math = spec.get("logimgf")

        # JAX backend (the spec already contains lambdified jax versions)
        mgf_jax = spec.get("mgf_jax")
        cgf_jax = spec.get("cgf_jax")
        imgf_jax = spec.get("imgf_jax")
        logimgf_jax = spec.get("logimgf_jax")

        if mgf_math is None or cgf_math is None or pdf_math is None:
            raise ValueError(
                "Registry must provide numeric MGF, CGF, and PDF functions."
            )

        # ---------------------------------------------------------
        # Build the object using the existing class
        # ---------------------------------------------------------
        obj = cls(
            name=prior_name,
            mgf_sym=mgf_sym,
            pdf_sym=pdf_sym,
            mgf_backend=None,   # not used
            pdf_backend=None,   # not used
            params=params,
        )

        # Directly assign the compiled functions (bypass as_mitMGFprior)
        obj.mgf = mgf_math
        obj.cgf = cgf_math
        obj.mgf_jax = mgf_jax
        obj.cgf_jax = cgf_jax
        obj.pdf_func = pdf_math
        obj.logpdf_func = logpdf_math
        obj.imgf = imgf_math
        obj.logimgf = logimgf_math
        obj.imgf_jax = imgf_jax
        obj.logimgf_jax = logimgf_jax

        # Store symbolic outputs
        obj.max_finite_moment = float(
            spec.get("max_finite_moment", float("inf"))
        )
        obj.mgf_finite_below = float(
            spec.get("mgf_finite_below", float("inf"))
        )

        obj.mgf_sym_out = mgf_sym
        obj.cgf_sym = cgf_sym
        obj.imgf_sym = imgf_sym
        obj.logimgf_sym = logimgf_sym

        return obj

    # ============================================================
    # VALIDATION
    # ============================================================
    @staticmethod
    def is_mitMGFprior(obj) -> bool:
        """
        Check if an object is a fully compiled mitMGFprior.
        Requires all six compiled functions to be present.
        """
        required_attrs = [
            "mgf", "cgf",
            "mgf_jax", "cgf_jax",
            "pdf_func", "logpdf_func"
        ]

        for attr in required_attrs:
            val = getattr(obj, attr, None)
            if not callable(val):
                return False
        return True

    # ============================================================
    # iMGF SUPPORT CHECK
    # ============================================================
    def has_iMGF(self) -> bool:
        """
        Check if this prior object has complete incomplete MGF (iMGF) support.

        Returns True only if all six iMGF-related functions are present:

        - imgf         (numeric ordinary)
        - logimgf      (numeric log)
        - imgf_jax     (JAX ordinary)
        - logimgf_jax  (JAX log)
        - imgf_sym     (symbolic ordinary)
        - logimgf_sym  (symbolic log)
        """
        required_attrs = [
            "imgf", "logimgf",
            "imgf_jax", "logimgf_jax",
            "imgf_sym", "logimgf_sym"
        ]
        return all(getattr(self, attr, None) is not None for attr in required_attrs)
