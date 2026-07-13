"""
mitMGFprior.py

Unified container for moment generating function (MGF) priors.

Design philosophy:
- registry provides fully-formed function bundles (no recomputation here)
- class only composes / stores / exposes interfaces
- symbolic and backend constructions are supported
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from jumufraktiv.registry import PRIOR_REGISTRY
import sympy as sp

from jumufraktiv.registry import get_prior
from jumufraktiv.symbols import t, theta


# ============================================================
# Core container
# ============================================================

@dataclass
class mitMGFprior:
    name: str = "custom"

    # -----------------------------
    # user inputs (raw layer)
    # -----------------------------
    mgf_sym: Optional[sp.Expr] = None
    pdf_sym: Optional[sp.Expr] = None

    mgf_backend: Optional[Callable] = None
    pdf_backend: Optional[Callable] = None

    params: Optional[Dict[str, Any]] = None

    # -----------------------------
    # compiled outputs
    # -----------------------------
    mgf: Optional[Callable] = None
    cgf: Optional[Callable] = None

    mgf_jax: Optional[Callable] = None
    cgf_jax: Optional[Callable] = None

    pdf_func: Optional[Callable] = None
    logpdf_func: Optional[Callable] = None

    mgf_sym_out: Any = None
    cgf_sym: Any = None
    pdf_sym_func: Any = None

    # ============================================================
    # USER ROUTE: manual construction compiler
    # ============================================================
    def as_mitMGFprior(self):

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
        Build a fully backend-complete MGF prior object from registry.

        Guarantees:
            - symbolic MGF/PDF if available
            - JAX-compatible functions always created via lambdify
            - math backend always available
        """
        from jumufraktiv.registry import PRIOR_REGISTRY

        params = params or {}

        if prior_name not in PRIOR_REGISTRY:
            raise ValueError(f"Unknown prior '{prior_name}'")

        # ---------------------------------------------------------
        # Get the factory function and call it
        # ---------------------------------------------------------
        factory = PRIOR_REGISTRY[prior_name]
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
        for attr in required_attrs:
            if getattr(self, attr, None) is None:
                return False
        return True