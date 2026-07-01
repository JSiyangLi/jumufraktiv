# backward compatibility alias during migration
import sys

sys.modules["mgf2post"] = sys.modules[__name__]