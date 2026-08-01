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
rather than a namespace package, which is what lets ``setuptools``'s
``find_packages`` list it. It is deliberately empty of imports — importing the
fourteen modules here would make every one of them a cost of touching any one
of them.
"""
