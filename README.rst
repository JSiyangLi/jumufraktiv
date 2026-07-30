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

so the marginal likelihood is a derivative of the prior MGF :math:`M` evaluated
at :math:`t = -b`:

.. math::

   p(y) = c(y) \, \frac{\mathrm{d}^{a}}{\mathrm{d}t^{a}} M(t) \Big|_{t=-b}.

When the sufficient statistic :math:`a(y)` is not an integer, the derivative is
fractional — hence the name. Everything else (density, CDF, quantiles, moments,
predictive, sequential updating) follows from the same object.

Installation
============

.. code-block:: bash

   pip install jumufraktiv

To work from a checkout:

.. code-block:: bash

   git clone https://github.com/JSiyangLi/jumufraktiv.git
   cd jumufraktiv
   pip install -e ".[dev]"

Optional extras: ``torch`` (PyTorch backend for the Pareto prior), ``docs``
(Sphinx), ``examples`` (notebook dependencies), ``dev`` (tests and linting).

.. note::

   The ``pareto`` and ``uniform`` priors currently require the ``torch`` extra.
   The Pareto module imports PyTorch while the prior dictionary is being loaded,
   and a failure there stops the priors after it from registering as well. The
   registry reports this as a warning, not an error, so the two priors are simply
   absent from ``registry.list_priors()``. Install with::

      pip install "jumufraktiv[torch]"

   if you need either of them. A future release will make the import lazy so
   that a missing extra costs only the Pareto Torch backend.

Quick start
===========

.. code-block:: python

   from jumufraktiv import registry
   from jumufraktiv.mitMGFprior_class import mitMGFprior
   from jumufraktiv.MGFDerivative_class import MGFDerivative

   registry.initialize()

   prior = mitMGFprior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})
   post = MGFDerivative(prior, data=[1, 2, 3], likelihood="poisson", scale=1.0)

   log_evidence, sign = post.evidence()
   log_density = post.post_density(0.5)

Before you start
================

This package makes three assumptions. If any of them does not fit your problem,
another package will serve you better.

1. **Strictly positive parameters.** The parameter space may be the whole
   positive real line or a subset of it, but it must not include zero or
   negative values.

2. **A supported likelihood.** Fourteen named likelihoods are built in:
   Poisson, Laplace (known mean), Normal (known mean), half-normal, Rayleigh,
   Maxwell-Boltzmann, Gamma (known shape), inverse-gamma (known shape),
   Lévy (known location), Weibull (known ``rho`` = shape/scale), Burr XII
   (known ``c``), Pareto (known scale), Dagum (known ``a`` and ``b``), and
   Gompertz (known scale). Custom likelihoods are supported by supplying your
   own ``a()``, ``b()`` and ``c()`` sufficient-statistic functions.

3. **A prior with a finite MGF.** Any prior works provided its MGF is finite on
   the negative half of the real line; improper priors whose MGF diverges there
   are not supported. Priors may come from the built-in dictionary
   (``gamma``, ``pareto``, ``uniform``, ``heaviside`` — see the note under
   *Installation* about ``pareto`` and ``uniform``) or be supplied directly as
   symbolic or callable MGF/PDF pairs.

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
