# mgf_derivative.py
import sympy as sp
import math
from derivativeDispatch import mgfDerivative_integer
from like_stats.Poisson import readyPoisson, cPoisson

class MGFDerivative:
    """
    Computes the MGF derivative for a given dataset and prior.
    The derivative order (a = sum y_i) and evaluation point (t = -b = -sum s_i)
    are determined from the data at instantiation.

    Usage:
        deriv = MGFDerivative(prior='gamma', data=data, params=params, method='jax', log=True)
        ev_log, sign = deriv.evidence()   # numeric mode
        # or
        deriv_sym = MGFDerivative(prior='gamma', data=data, method='symbolic', params=None)
        expr = deriv_sym.evidence()       # symbolic expression
    """
    def __init__(self, prior, data, method='symbolic', params=None,
                 simplify=False, log=True, **kwargs):
        self.prior = prior
        self.method = method
        self.simplify = simplify
        self.log = log
        self.params = params

        # ---- Compute sufficient statistics ----
        stats = readyPoisson(data, **kwargs)
        self.a = stats['a']
        self.b = stats['b']
        self.log_c = stats['log_c']

        # ---- Compute derivative ----
        # If method='symbolic' and params is None, we want symbolic expression.
        # Otherwise, numeric.
        if method.lower() == "symbolic" and params is None:
            # Symbolic derivative (prior parameters left as symbols)
            self._deriv_expr = mgfDerivative_integer(
                order=int(self.a),
                prior=self.prior,
                method="symbolic",
                t=float(-self.b),
                params=None,
                simplify=self.simplify,
                log=False
            )
            self._symbolic = True
            self._log_abs = None
            self._sign = None
            self._value = None
        else:
            # Numeric derivative
            if params is None:
                raise ValueError("For numeric evaluation, params must be provided.")
            log_abs, sign = mgfDerivative_integer(
                order=int(self.a),
                prior=self.prior,
                method=self.method,
                t=float(-self.b),
                params=self.params,
                simplify=self.simplify,
                log=True
            )
            self._symbolic = False
            self._log_abs = log_abs
            self._sign = sign
            self._value = None   # will be computed on demand

    # ---- Properties ----
    @property
    def is_symbolic(self):
        return self._symbolic

    @property
    def expr(self):
        if not self._symbolic:
            raise ValueError("This is a numeric result; use .log_abs / .value instead.")
        return self._deriv_expr

    @property
    def log_abs(self):
        if self._symbolic:
            raise ValueError("This is a symbolic result; use .expr instead.")
        return self._log_abs

    @property
    def sign(self):
        if self._symbolic:
            raise ValueError("This is a symbolic result; use .expr instead.")
        return self._sign

    @property
    def value(self):
        """Ordinary‑scale value of the derivative (numeric mode)."""
        if self._symbolic:
            raise ValueError("This is a symbolic result; use .expr instead.")
        if self._log_abs == -float('inf'):
            return 0.0
        return self.sign * math.exp(self._log_abs)

    # ---- Main method: evidence ----
    def evidence(self):
        """
        Return the marginal likelihood (evidence).

        For numeric: returns (log_abs, sign) if self.log=True, else ordinary float.
        For symbolic: returns sympy.Expr (c_expr * derivative_expr).
        """
        if self.is_symbolic:
            c_expr = cPoisson()
            return c_expr * self.expr
        else:
            total_log_abs = self.log_c + self.log_abs
            if self.log:
                return total_log_abs, self.sign
            else:
                return math.exp(total_log_abs) * self.sign