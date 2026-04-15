from __future__ import annotations

import sys
from importlib import util
from pathlib import Path


def ensure_text2cypher_package() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src" / "text2cypher"
    if not package_root.exists() and "text2cypher" in sys.modules:
        return

    spec = util.spec_from_file_location(
        "text2cypher",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError("Unable to load text2cypher package from the source tree")

    module = util.module_from_spec(spec)
    sys.modules["text2cypher"] = module
    spec.loader.exec_module(module)
