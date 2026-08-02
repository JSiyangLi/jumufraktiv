Tutorial
========

This page walks through the one calculation the package is built around, and
then through the quantities derived from it.

The object everything comes from
--------------------------------

For a likelihood in the MGF-marginalisable family the joint density factorises
as

.. math::

   L(\theta; y) = c(y)\, \theta^{a(y)} \exp\!\big(-b(y)\,\theta\big),

where :math:`\big(a(y), b(y)\big)` is **jointly** sufficient for
:math:`\theta`. Neither component is sufficient on its own, and they play
different roles: :math:`a(y)` is the order of differentiation and
:math:`b(y)` fixes the point of evaluation.

The marginal likelihood is then a derivative of the prior's moment-generating
function :math:`M`:

.. math::

   p(y) = c(y)\, \frac{\mathrm{d}^{a(y)}}{\mathrm{d}t^{a(y)}} M(t)
          \Big|_{t = -b(y)}.

Because :math:`a(y)` need not be a whole number, that derivative is in general
*fractional*, which is the package's reason to exist.

A first posterior
-----------------

.. code-block:: python

   from jumufraktiv import MGFDerivative, MGFPrior

   prior = MGFPrior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})
   post = MGFDerivative(prior, data=[1, 2, 3], likelihood="poisson", scale=1.0)

   log_evidence = post.evidence()

A Gamma prior against a Poisson likelihood is conjugate, so this particular
posterior is exactly ``Gamma(8, 6)`` and every answer below can be checked by
hand.

What you can ask of it
----------------------

.. code-block:: python

   post.post_density(0.5)          # posterior density (log by default)
   post.post_cdf(1.0)              # posterior CDF
   post.post_quantile(0.95)        # posterior quantile
   post.post_interval(level=0.95)  # equal-tailed credible interval
   post.post_sample(n=100)         # draws, by inverse transform
   post.post_raw_moment(2)         # E[theta^2]
   post.post_central_moment(2)     # the variance
   post.post_mgf(0.2)              # posterior MGF
   post.post_predictive(2)         # predictive mass for a new observation

Every one of these is derived from the same fractional derivative, which is
why they share a backend and a set of options.

Conditioning on more data
-------------------------

``update`` treats the current posterior as the prior for a new batch, so the
evidence factorises:

.. code-block:: python

   later = post.update(new_data=[5, 7], likelihood="poisson", scale=1.0)

   post.evidence() + later.evidence()   # equals the batch evidence for all five

Choosing a backend
------------------

``method="auto"`` is the default and computes the derivative as an
expectation, :math:`\mathrm{D}^a M(t) = E[\theta^a e^{t\theta}]`, whose
integrand is positive and therefore cannot cancel. It is not the fastest
route, but it is never catastrophically wrong, which is the property a default
needs. ``method="symbolic"``, ``"scipy"``, ``"mpmath"``, ``"bell"`` and
``"jax"`` are available when you know your problem suits them.
