import math

def logplus(x: float, y: float) -> float:
    """
    Compute log(exp(x) + exp(y)) in a numerically stable way.
    Returns log(exp(x) + exp(y)).
    """
    if x > y:
        return x + math.log1p(math.exp(y - x))  # log1p(exp(d)) is more accurate for small d
    else:
        return y + math.log1p(math.exp(x - y))

def logminus(x: float, y: float) -> float:
    """
    Compute log(exp(x) - exp(y)) in a numerically stable way.
    Returns log(exp(x) - exp(y)). Returns NaN if x <= y.
    """
    if x > y:
        return x + math.log1p(-math.exp(y - x))  # log1p(-exp(d)) is stable when d is negative
    else:
        return float('nan')

def logplusvec(vals) -> float:
    """
    Compute log( sum(exp(vals)) ) for a sequence of log‑space values.
    Uses a numerically stable "log-sum-exp" trick.
    """
    # Start with the smallest possible log value (negative infinity)
    # Equivalent to -inf in Python: float('-inf')
    r = float('-inf')
    for x in vals:
        r = logplus(r, x)
    return r