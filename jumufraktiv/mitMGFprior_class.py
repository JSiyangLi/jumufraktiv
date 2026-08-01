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
0.8888888889

>>> # Manual construction with symbolic expressions
>>> from jumufraktiv.symbols import t, theta
>>> prior = mitMGFprior(mgf_sym=(1 - t)**(-2), pdf_sym=theta*sp.exp(-theta))
>>> prior.as_mitMGFprior()
"""

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

    Attributes
    ----------
    name : str, default "custom"
        Name of the prior distribution.
    mgf_sym : sympy.Expr, optional
        Symbolic expression for the MGF.
    pdf_sym : sympy.Expr, optional
        Symbolic expression for the PDF.
    mgf_backend : callable, optional
        Numeric MGF function (e.g., from a custom implementation).
    pdf_backend : callable, optional
        Numeric PDF function.
    params : dict, optional
        Dictionary of numeric parameters (e.g., `{'alpha': 2.0, 'beta': 3.0}`).

    Compiled functions (populated by `as_mitMGFprior` or `from_registry`):
    - mgf, cgf : NumPy-based MGF and CGF functions.
    - mgf_jax, cgf_jax : JAX-based MGF and CGF functions.
    - pdf_func, logpdf_func : NumPy-based PDF and log-PDF functions.

    Incomplete MGF (iMGF) attributes (if supported):
    - imgf, logimgf : Numeric ordinary and log-scale iMGF.
    - imgf_jax, logimgf_jax : JAX versions.
    - imgf_sym, logimgf_sym : Symbolic expressions.

    Methods
    -------
    as_mitMGFprior()
        Compile the prior from symbolic or backend inputs.
    from_registry(cls, prior_name, params=None, simplify=False)
        Factory method to build a prior from the registry.
    is_mitMGFprior(obj)
        Static method to check if an object is a fully compiled prior.
    has_iMGF()
        Return True if all iMGF components are present.

    Notes
    -----
    The class follows a two-step construction pattern:
    1. Create an instance with raw inputs (symbolic or backend).
    2. Call `as_mitMGFprior()` to compile and populate all functions.

    The registry route (`from_registry`) performs both steps automatically.

    Examples
    --------
    >>> # Manual symbolic prior
    >>> from jumufraktiv.symbols import t, theta
    >>> prior = mitMGFprior(mgf_sym=(1 - t)**(-2), pdf_sym=theta * sp.exp(-theta))
    >>> prior = prior.as_mitMGFprior()
    >>> prior.mgf(-0.5)
    1.7777777778

    >>> # Registry-based prior
    >>> gamma_prior = mitMGFprior.from_registry('gamma', params={'alpha':2.0, 'beta':3.0})
    >>> gamma_prior.mgf(-1.0)
    0.8888888889
    """
    name: str = "custom"

    # -----------------------------
    # user inputs (raw layer)
    # -----------------------------
    mgf_sym: sp.Expr | None = None
    pdf_sym: sp.Expr | None = None

    mgf_backend: Callable | None = None
    pdf_backend: Callable | None = None

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

    # -----------------------------
    # compiled outputs
    # -----------------------------
    mgf: Callable | None = None
    cgf: Callable | None = None

    mgf_jax: Callable | None = None
    cgf_jax: Callable | None = None

    pdf_func: Callable | None = None
    logpdf_func: Callable | None = None

    mgf_sym_out: Any = None
    cgf_sym: Any = None
    pdf_sym_func: Any = None

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
        1.7777777778

        >>> # Backend mode
        >>> def mgf_backend(x, xp, **params):
        ...     return xp.exp(x)
        >>> def pdf_backend(x, xp, **params):
        ...     return xp.exp(-x)
        >>> prior = mitMGFprior(mgf_backend=mgf_backend, pdf_backend=pdf_backend)
        >>> prior = prior.as_mitMGFprior()
        >>> prior.mgf(0.0)
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

        raise ValueError("Must provide either (mgf_sym, pdf_sym) or (mgf_backend, pdf_backend).")

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
        >>> gamma_prior = mitMGFprior.from_registry('gamma', params={'alpha':2.0, 'beta':3.0})
        >>> gamma_prior.mgf(-1.0)
        0.8888888889

        >>> # With symbolic simplification
        >>> prior = mitMGFprior.from_registry('pareto', params={'alpha':0.5, 'xi':1.0}, simplify=True)
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
            message = f"Unknown prior '{prior_name}'. Available: {sorted(PRIOR_REGISTRY)}"
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
        spec = factory(params)  # <-- this is the make_prior_spec dict

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
            raise ValueError("Registry must provide numeric MGF, CGF, and PDF functions.")

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

        obj.mgf_sym_out = mgf_sym
        obj.cgf_sym = cgf_sym
        obj.pdf_sym_func = spec.get("pdf_sym_func")
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
