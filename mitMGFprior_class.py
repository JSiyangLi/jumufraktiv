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
import sympy as sp
import numpy as np
import scipy.stats as stats
from typing import Callable, Optional, Any, Dict

# ---- Import MGFdictionary functions ----
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


class mitMGFprior:
    """
    Container for a prior distribution's functions.

    Attributes
    ----------
    name : str
        Prior name.
    dist : Optional[Callable]
        Constructor for a scipy.stats distribution (params dict -> rv_continuous).
    mgf_sym : Optional[Callable]
        Function returning symbolic MGF expression (no arguments).
    cgf_sym : Optional[Callable]
        Function returning symbolic CGF expression (no arguments).
    mgf : Optional[Callable]
        Numeric MGF function (t, *args).
    cgf : Optional[Callable]
        Numeric CGF function (t, *args).
    mgf_jax : Optional[Callable]
        JAX MGF function.
    cgf_jax : Optional[Callable]
        JAX CGF function.
    pdf_sym : Optional[Callable]
        Function returning symbolic PDF expression (no arguments).
    pdf_sym_func : Optional[Callable]
        Function (params dict) -> symbolic PDF with numeric params substituted.
    pdf_func : Optional[Callable]
        Numeric PDF function (theta, *args).
    logpdf_func : Optional[Callable]
        Numeric log‑PDF function (theta, *args).
    """
    def __init__(
        self,
        name: str,
        dist: Optional[Callable] = None,
        mgf_sym: Optional[Callable] = None,
        cgf_sym: Optional[Callable] = None,
        mgf: Optional[Callable] = None,
        cgf: Optional[Callable] = None,
        mgf_jax: Optional[Callable] = None,
        cgf_jax: Optional[Callable] = None,
        pdf_sym: Optional[Callable] = None,
        pdf_sym_func: Optional[Callable] = None,
        pdf_func: Optional[Callable] = None,
        logpdf_func: Optional[Callable] = None,
    ):
        self.name = name
        self.dist = dist
        self.mgf_sym = mgf_sym
        self.cgf_sym = cgf_sym
        self.mgf = mgf
        self.cgf = cgf
        self.mgf_jax = mgf_jax
        self.cgf_jax = cgf_jax
        self.pdf_sym = pdf_sym
        self.pdf_sym_func = pdf_sym_func
        self.pdf_func = pdf_func
        self.logpdf_func = logpdf_func

    @classmethod
    def as_(cls, **kwargs):
        """
        Convert a set of functions into a mitMGFprior object.

        This is analogous to R's `as.matrix()` or `as.vector()`: it takes
        loose functions and wraps them into a `mitMGFprior` instance.

        Parameters
        ----------
        **kwargs : keyword arguments
            Any of the attributes of mitMGFprior: name, dist, mgf_sym, cgf_sym,
            mgf, cgf, mgf_jax, cgf_jax, pdf_sym, pdf_sym_func, pdf_func, logpdf_func.

        Returns
        -------
        mitMGFprior
            A new instance with the provided functions.

        Example
        -------
        custom = mitMGFprior.as_(
            name='my_prior',
            mgf=lambda t, a, b: (b/(b-t))**a,
            cgf=lambda t, a, b: a*(math.log(b)-math.log(b-t)),
            pdf_func=lambda theta, a, b: (b**a / math.gamma(a)) * theta**(a-1) * math.exp(-b*theta),
        )
        """
        # Default name if not provided
        if 'name' not in kwargs:
            kwargs['name'] = 'custom'
        return cls(**kwargs)

    def is_(self, attr: str) -> bool:
        """
        Check if the prior has a given function attribute (i.e., it is not None).

        Parameters
        ----------
        attr : str
            Name of the attribute to check (e.g., 'mgf', 'cgf', 'pdf_func').

        Returns
        -------
        bool
            True if the attribute exists and is not None, False otherwise.
        """
        return hasattr(self, attr) and getattr(self, attr) is not None

    def get_dist(self, params: dict):
        """Return a scipy.stats distribution instance for given parameters."""
        if self.dist is None:
            raise ValueError(f"No scipy distribution available for prior '{self.name}'.")
        return self.dist(params)

    def __repr__(self):
        return f"mitMGFprior(name='{self.name}')"


# ---- Global registry of prior specifications ----
PRIOR_REGISTRY: Dict[str, dict] = {}

def register_prior(name: str, spec: dict):
    """
    Register a new prior.

    Parameters
    ----------
    name : str
        Prior name (key in registry).
    spec : dict
        Dictionary with keys: 'dist', 'mgf_sym', 'cgf_sym', 'mgf', 'cgf',
        'mgf_jax', 'cgf_jax', 'pdf_sym', 'pdf_sym_func', 'pdf_func', 'logpdf_func'.
        All keys are optional; missing ones default to None.
    """
    PRIOR_REGISTRY[name] = spec

def make_prior(name: str, **overrides) -> mitMGFprior:
    """
    Create a mitMGFprior instance by looking up the registry.

    Parameters
    ----------
    name : str
        Name of the prior (must be in PRIOR_REGISTRY).
    **overrides : dict
        Additional keyword arguments to override registry entries.

    Returns
    -------
    mitMGFprior
        The prior instance.
    """
    if name not in PRIOR_REGISTRY:
        raise ValueError(f"Unknown prior: {name}. Available: {list(PRIOR_REGISTRY.keys())}")
    spec = PRIOR_REGISTRY[name].copy()
    spec.update(overrides)
    return mitMGFprior(name=name, **spec)


# ---- Register built‑in priors ----
register_prior('gamma', {
    'dist': lambda p: stats.gamma(a=p['alpha'], scale=1/p['beta']),
    'mgf_sym': gamma_mgf_symbolic,
    'cgf_sym': gamma_cgf_symbolic,
    'mgf': gamma_mgf,
    'cgf': gamma_cgf,
    'mgf_jax': gamma_mgf_jax,
    'cgf_jax': gamma_cgf_jax,
    'pdf_sym': gamma_pdf_symbolic,
    'pdf_sym_func': gamma_pdf_symbolic_sub,
})

register_prior('pareto', {
    'dist': lambda p: stats.pareto(b=p['alpha'], scale=p['xi']),
    'mgf_sym': pareto_mgf_symbolic,
    'cgf_sym': pareto_cgf_symbolic,
    'mgf': pareto_mgf,
    'cgf': pareto_cgf,
    'mgf_jax': pareto_mgf_jax,
    'cgf_jax': pareto_cgf_jax,
    'pdf_sym': pareto_pdf_symbolic,
    'pdf_sym_func': pareto_pdf_symbolic_sub,
})

register_prior('heaviside', {
    'dist': None,  # improper, no scipy distribution
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
})

register_prior('uniform', {
    'dist': lambda p: stats.uniform(loc=p['a'], scale=p['b'] - p['a']),
    'mgf_sym': uniform_mgf_symbolic,
    'cgf_sym': uniform_cgf_symbolic,
    'mgf': uniform_mgf,
    'cgf': uniform_cgf,
    'mgf_jax': uniform_mgf_jax,
    'cgf_jax': uniform_cgf_jax,
    'pdf_sym': uniform_pdf_symbolic,
    'pdf_sym_func': uniform_pdf_symbolic_sub,
})