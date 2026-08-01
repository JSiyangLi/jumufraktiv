API Reference
=============

This section contains the complete API documentation for ``jumufraktiv``.

.. currentmodule:: jumufraktiv

Main Classes
------------

.. autoclass:: MGFDerivative
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mitMGFprior
   :members:
   :undoc-members:
   :show-inheritance:

Core Derivative Dispatch
------------------------

.. automodule:: derivativeDispatch
   :members:
   :undoc-members:

Root Finding
------------

.. automodule:: root_finding
   :members:
   :undoc-members:

Registry and Symbols
--------------------

.. automodule:: registry
   :members:
   :undoc-members:

.. automodule:: symbols
   :members:
   :undoc-members:

Symbolic Backends
-----------------

.. automodule:: symbolic_integerDeriv
   :members:
   :undoc-members:

.. automodule:: symbolic_fractionalDeriv
   :members:
   :undoc-members:

Numeric Backends
----------------

.. automodule:: numeric_integerDeriv_Bell
   :members:
   :undoc-members:

.. automodule:: numeric_integerDeriv_JAX
   :members:
   :undoc-members:

.. automodule:: numeric_fractionalDeriv_grid
   :members:
   :undoc-members:

.. automodule:: numeric_fractionalDeriv_mpmath
   :members:
   :undoc-members:

.. automodule:: numeric_expectation
   :members:
   :undoc-members:

Likelihood Statistics (``like_stats``)
--------------------------------------

Each distribution module provides:
- `ready<Distribution>`: aggregated sufficient statistics.
- `bereit<Distribution>`: per‑element statistics for vectorised predictive evaluation.
- `c<Distribution>`: symbolic normalising constant.

.. automodule:: like_stats.Poisson
   :members:
   :undoc-members:

.. automodule:: like_stats.Gamma
   :members:
   :undoc-members:

.. automodule:: like_stats.Laplace
   :members:
   :undoc-members:

.. automodule:: like_stats.Normal
   :members:
   :undoc-members:

.. automodule:: like_stats.Rayleigh
   :members:
   :undoc-members:

.. automodule:: like_stats.MaxwellBoltzmann
   :members:
   :undoc-members:

.. automodule:: like_stats.InverseGamma
   :members:
   :undoc-members:

.. automodule:: like_stats.Levy
   :members:
   :undoc-members:

.. automodule:: like_stats.Weibull
   :members:
   :undoc-members:

.. automodule:: like_stats.BurrXII
   :members:
   :undoc-members:

.. automodule:: like_stats.Pareto
   :members:
   :undoc-members:

.. automodule:: like_stats.Dagum
   :members:
   :undoc-members:

.. automodule:: like_stats.Gompertz
   :members:
   :undoc-members:

.. automodule:: like_stats.HalfNormal
   :members:
   :undoc-members:

Prior Dictionary (``MGFdictionary``)
------------------------------------

This subpackage imports all prior modules and registers them. It is not intended for direct use; use :func:`registry.get_prior` instead.

.. automodule:: MGFdictionary
   :members: