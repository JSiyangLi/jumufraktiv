# backward compatibility alias during migration
import sys

sys.modules["mgf2post"] = sys.modules[__name__]

# jumufraktiv/__init__.py

from .MGFDerivative_class import MGFDerivative
from .mitMGFprior_class import mitMGFprior
from .derivativeDispatch import mgfDerivative, mgfDerivative_integer, mgfDerivative_fractional
# add other public imports as needed