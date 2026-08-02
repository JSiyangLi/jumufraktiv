Installation
============

``jumufraktiv`` requires Python 3.10 or newer.

From PyPI
---------

.. code-block:: bash

   pip install jumufraktiv

From source
-----------

.. code-block:: bash

   git clone https://github.com/JSiyangLi/jumufraktiv.git
   cd jumufraktiv
   pip install -e .

Optional extras
---------------

Three extras are declared, and each is genuinely optional --- the package
imports and computes without any of them.

``examples``
   ``matplotlib`` and ``jupyter``, for the two notebooks under ``notebooks/``.

``docs``
   ``sphinx`` and ``sphinx-rtd-theme``, for building this documentation.

``dev``
   ``pytest``, ``pytest-cov``, ``hypothesis``, ``ruff`` and ``docutils``, for
   the test suite and the linter.

.. code-block:: bash

   pip install -e ".[dev]"

Verifying the install
---------------------

.. code-block:: bash

   python -c "import jumufraktiv; print(jumufraktiv.__version__)"

The quick start in the README is executed by the test suite on every push, so
if it runs for you the installation is sound.
