"""
mitMGFprior.py

Defines the mitMGFprior class to represent a prior distribution in the
MGF‑marginalisation framework. It stores all necessary functions:
MGF, CGF, PDF (symbolic and numeric), and scipy.stats distribution.

Priors are registered in a global registry. New priors can be added by
calling `register_prior` or by adding an entry to the registry.

The class provides:
    - `as_()`: A class method to convert a set of functions into a mitMGFprior object.
    - `is_()`: Check if a particular function is available.

Usage:
    from mitMGFprior import mitMGFprior

    custom_prior = mitMGFprior.as_(
        name='my_prior',
        mgf=lambda t, a, b: (b/(b-t))**a,
        cgf=lambda t, a, b: a*(math.log(b)-math.log(b-t)),
        pdf_func=lambda theta, a, b: ...,
        ...
    )
    if custom_prior.is_('mgf'):
        print("MGF is available")
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

# optional
import sympy as sp
import scipy.special as sc
import jax.scipy.special as jsc


# ============================================================
# Core class
# ============================================================

@dataclass
class mitMGFprior:
    name: str = "custom"

    # --- symbolic layer (optional) ---
    mgf_sym: Optional[sp.Expr] = None
    pdf_sym: Optional[sp.Expr] = None

    # --- backend layer (optional) ---
    mgf_backend: Optional[Callable] = None
    pdf_backend: Optional[Callable] = None

    params: Optional[Dict[str, Any]] = None

    # compiled outputs
    mgf: Optional[Callable] = None
    cgf: Optional[Callable] = None
    mgf_jax: Optional[Callable] = None
    cgf_jax: Optional[Callable] = None

    pdf_func: Optional[Callable] = None
    logpdf_func: Optional[Callable] = None

    mgf_sym_out: Any = None
    cgf_sym: Any = None
    pdf_sym_func: Any = None


    # ========================================================
    # Validation method
    # ========================================================
    @staticmethod
    def is_mitMGFprior(obj) -> bool:
        """
        Check whether an object is compatible with mitMGFprior.
        Accepts dict-like or object-like inputs.
        """

        required_any = [
            "mgf_sym", "mgf_backend"
        ]

        # allow dict or object
        def get(x, key):
            if isinstance(x, dict):
                return x.get(key, None)
            return getattr(x, key, None)

        has_symbolic = get(obj, "mgf_sym") is not None
        has_backend = get(obj, "mgf_backend") is not None

        return has_symbolic or has_backend


    # ========================================================
    # Build / compile method
    # ========================================================
    @classmethod
    def from_registry(cls, name: str, params: dict = None):
        from jumufraktiv.registry import get_prior

        spec_fn = get_prior(name)
        spec = spec_fn(params or {})

        obj = cls(
            name=name,
            mgf_sym=spec.get("mgf_sym"),
            pdf_sym=spec.get("pdf_sym"),

            # directly inject backend functions (NO recomputation)
            mgf=spec.get("mgf"),
            cgf=spec.get("cgf"),
            mgf_jax=spec.get("mgf_jax"),
            cgf_jax=spec.get("cgf_jax"),

            pdf_func=spec.get("pdf_func"),
            logpdf_func=spec.get("logpdf_func"),

            params=params,
        )

        return obj
    
    def as_mitMGFprior(self):
        """
        Compile all representations from available inputs.
        """

        # ----------------------------------------------------
        # CASE 1: symbolic input
        # ----------------------------------------------------
        if self.mgf_sym is not None:

            t = sp.Symbol("t")

            self.cgf_sym = sp.log(self.mgf_sym)

            self.mgf_sym_out = sp.simplify(self.mgf_sym)
            self.cgf_sym = sp.simplify(self.cgf_sym)

            # lambdify symbolic → math / numpy / jax
            self.mgf = sp.lambdify(t, self.mgf_sym, modules="math")
            self.cgf = sp.lambdify(t, self.cgf_sym, modules="math")

            self.mgf_jax = sp.lambdify(t, self.mgf_sym, modules="jax")
            self.cgf_jax = sp.lambdify(t, self.cgf_sym, modules="jax")

            # pdf if provided
            if self.pdf_sym is not None:
                x = sp.Symbol("x")
                self.pdf_sym_func = sp.lambdify(x, self.pdf_sym, modules="math")
                self.pdf_func = self.pdf_sym_func
                self.logpdf_func = lambda x: math.log(self.pdf_func(x))

            return self


        # ----------------------------------------------------
        # CASE 2: backend input
        # ----------------------------------------------------
        if self.mgf_backend is not None:

            params = self.params or {}

            def mgf_math(t):
                return self.mgf_backend(t, xp=math, special=sc, **params)

            def mgf_numpy(t):
                return self.mgf_backend(t, xp=np, special=sc, **params)

            def mgf_jax_fn(t):
                return self.mgf_backend(t, xp=jnp, special=jsc, **params)

            self.mgf = mgf_math
            self.mgf_jax = mgf_jax_fn

            self.cgf = lambda t: math.log(self.mgf(t))
            self.cgf_jax = lambda t: jnp.log(self.mgf_jax(t))

            if self.pdf_backend is not None:

                def pdf_math(x):
                    return self.pdf_backend(x, xp=math, special=sc, **params)

                self.pdf_func = pdf_math
                self.logpdf_func = lambda x: math.log(pdf_math(x))

            return self

        raise ValueError("Must provide either mgf_sym or mgf_backend")