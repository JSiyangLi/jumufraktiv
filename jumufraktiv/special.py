"""Numeric implementations of special functions SymPy emits and SciPy lacks.

`lambdify` compiles an expression against a stack of modules, and every name it
cannot resolve is left as a bare global — so the compiled function raises
``NameError`` on its first call rather than failing to build. This module
supplies the names that would otherwise do that, so the compiled path stays
available instead of falling through to symbolic substitution.

Notes
-----
The one entry so far is the generalised exponential integral, which the Pareto
prior's MGF is written with. Its absence was expensive rather than merely
inconvenient: with no compiled form, the fixed-grid fractional kernel evaluated
its integrand by ``expr.subs(...).evalf()`` at every node, which is **2840
substitutions for a single evaluation point** at roughly 306 microseconds each.

This module belongs to neither the prior layer nor the dispatch layer in the
sense the architecture rules police. It is not distribution-specific
knowledge — ``expint`` is a mathematical function, not a fact about Pareto —
and the dispatcher does not consult it directly, so no layer learns which
distribution it is handling by importing it.
"""

import mpmath as mp
import numpy as np


def expint(n, z):
    """The generalised exponential integral ``E_n(z)``, for real order.

    Parameters
    ----------
    n : float or array-like
        Order. Real, and not restricted to integers.
    z : float or array-like
        Argument. Broadcast against `n`.

    Returns
    -------
    float or numpy.ndarray
        ``E_n(z)``, scalar if both inputs are scalar.

    Notes
    -----
    Backed by :func:`mpmath.expint`, elementwise, at roughly 39 microseconds a
    point. Neither NumPy nor SciPy can serve this at real order:
    ``scipy.special.expn`` takes an integer order and **truncates a real one**,
    so ``expn(2.5, x)`` silently returns ``expn(2, x)``, and the identity
    ``E_n(z) = z^{n-1} Gamma(1-n, z)`` routes through an upper incomplete gamma
    of negative first argument, which ``scipy.special.gammaincc`` does not
    accept.

    The elementwise loop is the cost of correctness here, and it is still far
    cheaper than the alternative it replaces — symbolic substitution runs about
    eight times slower per node.
    """
    order, argument = np.broadcast_arrays(
        np.asarray(n, dtype=float), np.asarray(z, dtype=float)
    )

    flat_order = order.ravel()
    flat_argument = argument.ravel()
    out = np.empty(flat_order.size, dtype=float)

    # 30 digits internally, matching the incomplete-MGF helper: enough that the
    # float64 result is correctly rounded, cheap enough to pay per node.
    with mp.workdps(30):
        for index in range(flat_order.size):
            out[index] = float(
                mp.expint(
                    mp.mpf(float(flat_order[index])),
                    mp.mpf(float(flat_argument[index])),
                )
            )

    result = out.reshape(order.shape)
    return float(result) if result.ndim == 0 else result


#: Passed to :func:`sympy.lambdify` ahead of ``scipy`` and ``numpy``, so these
#: take precedence over anything of the same name in either.
LAMBDIFY_NAMESPACE = {"expint": expint}
