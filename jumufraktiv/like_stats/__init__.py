"""Likelihood statistics for the MGF-marginalisable family.

Each module here exports exactly three functions for one likelihood:

``readyX(data, **kwargs)``
    Statistics ``{'a', 'b', 'log_c'}`` aggregated over the whole sample.
``bereitX(data, **kwargs)``
    The same statistics per observation, used by the vectorised posterior
    predictive.
``cX()``
    The symbolic normalising constant.

These modules are pure functions of the data. They know nothing about priors
or derivatives, and must not import from the inference layer.

This file exists so that ``jumufraktiv.like_stats`` is an ordinary package
rather than a namespace package. It shipped correctly before, but only
incidentally: ``setuptools``'s ``find_packages`` did not list it, and the
fourteen modules reached the wheel by a route that declaring the package makes
explicit. It is deliberately empty of imports — importing the fourteen modules
here would make every one of them a cost of touching any one of them.
"""
