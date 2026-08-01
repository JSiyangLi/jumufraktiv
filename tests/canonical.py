"""The canonical test problem, in one place.

A Gamma prior against a Poisson likelihood is conjugate, so the posterior is a
Gamma with known parameters and every quantity the package computes has a
closed form to compare against. Most of the suite, and every docstring example,
is written against this one problem.

It lives in its own module because two conftest files need it: ``tests/``
supplies it to the suite, and the repository-root ``conftest.py`` supplies it to
the docstring examples, which are collected from ``jumufraktiv/`` and so never
see ``tests/conftest.py``. Both import from here rather than each carrying a
copy.
"""

#: Prior:      theta ~ Gamma(shape=ALPHA, rate=BETA)
#: Likelihood: y_i ~ Poisson(theta * s_i) with s_i = POISSON_SCALE
ALPHA = 2.0
BETA = 3.0
POISSON_DATA = [1, 2, 3]
POISSON_SCALE = 1.0

#: Posterior shape and rate implied by the values above:
#: theta | y ~ Gamma(shape=ALPHA + sum(y), rate=BETA + sum(s)).
POST_SHAPE = ALPHA + sum(POISSON_DATA)  # 8.0
POST_RATE = BETA + POISSON_SCALE * len(POISSON_DATA)  # 6.0
