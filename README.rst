===========
jumufraktiv
===========

Bayesian inference via fractional derivatives of prior moment-generating
functions.

.. image:: https://img.shields.io/pypi/pyversions/jumufraktiv.svg
   :target: https://pypi.org/project/jumufraktiv/
   :alt: Supported Python versions

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/JSiyangLi/jumufraktiv/blob/main/LICENSE
   :alt: MIT license

.. readme-body-start

Overview
========

``jumufraktiv`` computes model evidence, posterior densities, posterior
predictive densities, posterior moment-generating functions and posterior
moments by taking **fractional derivatives of the prior MGF**.

For a likelihood in the MGF-marginalisable family, the joint density factorises
as

.. math::

   L(\theta; y) = c(y)\, \theta^{a(y)} \exp\!\big(-b(y)\,\theta\big),

where :math:`\big(a(y), b(y)\big)` is **jointly** sufficient for :math:`\theta` —
neither component is sufficient on its own. The marginal likelihood is then a
derivative of the prior MGF :math:`M`, of order :math:`a(y)`, evaluated at
:math:`t = -b(y)`:

.. math::

   p(y) = c(y) \, \frac{\mathrm{d}^{a}}{\mathrm{d}t^{a}} M(t) \Big|_{t=-b}.

So the two components play different roles: :math:`a(y)` sets the order of
differentiation and :math:`b(y)` the point of evaluation. When :math:`a(y)` is
not an integer the derivative is fractional — the Liouville--Caputo definition of fractional derivatives are used extensively in this package. Everything else
(density, CDF, quantiles, moments, predictive, sequential updating) follows from
the same object.

The name of this package constitutes of two parts - "jumu", 矩母, means "moment-generating" in Chinese;
"fraktiv", a newly-invented word by authors of this package, stands for "fractional derivatives".
Together, "jumufraktiv" becomes "fractional derivatives of moment-generating (function)", the essential computation method of this package.
Moreover, the word "jumufraktiv" sounds foreign in many languages, just like fractional derivatives and MGFs sound foreign in the world of Bayesian computation.
In Chinese, although 矩母 is native, "fraktiv" is clearly a foreign word; in constrast, although "aktiv" is a native German word and "актив" is a Russian word, "jumu" and "Джуму" make this word sounds foreign in both languages; moreover, this word is clearly foreign in English.

Installation
============

.. code-block:: bash

   pip install jumufraktiv

To work from a checkout:

.. code-block:: bash

   git clone https://github.com/JSiyangLi/jumufraktiv.git
   cd jumufraktiv
   pip install -e ".[dev]"

Optional extras: ``docs`` (Sphinx), ``examples`` (notebook dependencies),
``dev`` (tests and linting).

None of the extras are needed for the built-in priors, which run on the
core dependencies alone.

Quick start
===========

.. code-block:: python

   from jumufraktiv import registry
   from jumufraktiv.MGFPrior_class import MGFPrior
   from jumufraktiv.MGFDerivative_class import MGFDerivative

   registry.initialize()

   prior = MGFPrior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})
   post = MGFDerivative(prior, data=[1, 2, 3], likelihood="poisson", scale=1.0)

   log_evidence = post.evidence()
   log_density = post.post_density(0.5)

Before you start
================

This package makes three assumptions. If any of them does not fit your problem,
another package will serve you better.

1. **Strictly positive parameters.** The parameter space may be the whole
   positive real line or a subset of it, but it must not include zero or
   negative values.

2. **A supported likelihood.** Fourteen are built in. The name in the first
   column is the string to pass as ``likelihood=``; the second column lists the
   known parameters that likelihood needs as keyword arguments.

   ======================= =================
   ``likelihood=``         known parameters
   ======================= =================
   ``"poisson"``           ``scale``
   ``"laplace"``           ``mean``
   ``"normal"``            ``mean``
   ``"halfnormal"``        none
   ``"rayleigh"``          none
   ``"maxwell-boltzmann"`` none
   ``"gamma"``             ``shape``
   ``"inverse gamma"``     ``shape``
   ``"levy"``              ``location``
   ``"weibull"``           ``rho``
   ``"burrxii"``           ``known_shape``
   ``"pareto"``            ``scale``
   ``"dagum"``             ``r``, ``s``
   ``"gompertz"``          ``scale``
   ======================= =================

   Adding a fifteenth means writing a module under ``jumufraktiv/like_stats/``
   that supplies :math:`a(y)`, :math:`b(y)` and :math:`\log c(y)`, and
   registering it — there is no runtime API for supplying those functions from
   caller code. Theorem 4.1 of the reference gives the test for whether a
   likelihood belongs to the family at all: it does if and only if it admits a
   gamma conjugate prior.

3. **A prior with a finite MGF.** Any prior works provided its MGF is finite on
   the negative half of the real line; improper priors whose MGF diverges there
   are not supported. Priors may come from the built-in dictionary
   (``gamma``, ``pareto``, ``uniform``, ``heaviside``) or be supplied directly
   as symbolic or callable MGF/PDF pairs.

Citation
========

This package implements the method of

   Li, S.-Y., van Dyk, D. A., & Autenrieth, M. *Using fractional derivatives to
   derive marginal densities.* [manuscript in preparation] (2026).
   `arXiv:2409.11167 <https://arxiv.org/abs/2409.11167>`_

Please cite that paper if you use ``jumufraktiv`` in published work. Machine-
readable metadata is in ``CITATION.cff``.

The derivative is the Liouville–Caputo fractional derivative with lower terminal
at :math:`-\infty`. That terminal is essential rather than conventional: it is
what gives :math:`D^{a} e^{t\theta} = \theta^{a} e^{t\theta}`, and hence
:math:`D^{a} M(t) = \mathbb{E}[\theta^{a} e^{t\theta}]`. It also means the
operator reads :math:`M` only on :math:`(-\infty, t]`, so the method works for
priors whose MGF exists only for :math:`t \le 0`, such as the Pareto.

Status
======

Alpha. The API is not yet stable and may change without a deprecation period
before version 1.0. See ``CHANGELOG.md`` for release notes.

License
=======

MIT. See ``LICENSE``.
