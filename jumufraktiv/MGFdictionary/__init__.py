import pkgutil
import importlib
from pathlib import Path

package_dir = Path(__file__).resolve().parent

for module_info in pkgutil.iter_modules([str(package_dir)]):
    # skip non-prior modules
    if module_info.ispkg:
        continue

    name = module_info.name

    # optional filter: only load MGFs
    if "MGF" in name:
        importlib.import_module(f"{__name__}.{name}")