Examples
========

Two worked notebooks ship with the repository, under ``notebooks/``. They are
run by hand rather than by the test suite, because one of them takes tens of
minutes; the suite checks statically that every cell parses, that none carries
stored output, and that none uses a retired spelling.

``examples.ipynb``
------------------

Compares backends against each other and against closed forms, for the
conjugate Gamma/Poisson problem and for a Gamma likelihood with an Exponential
prior. Read this one first.

``ParetoPumpFailureExample.ipynb``
----------------------------------

The pump-failure data of Gaver and O'Muircheartaigh (1987) with a diffuse
Pareto prior, following the paper. It covers the evidence against the paper's
analytic formula, the posterior density, CDF, credible intervals and sampling,
the posterior predictive, the posterior MGF and moments, and sequential
updating on a split of the data.

Two of its cells demonstrate a *refusal* on purpose --- a symbolic moment
order, and a negative moment order below the posterior's bound --- and catch
and print it rather than stopping the notebook.

Running them
------------

.. code-block:: bash

   pip install -e ".[examples]"
   jupyter lab notebooks/

The README's quick start is a shorter starting point, and it is executed by
the test suite on every push.
