"""Import earlier sprints' code by folder, without package-name clashes.

Every sprint keeps its code in a package called `src`, so Sprint 4's can't
be imported as `src` from here. This loads it under another name:

    from src.deps import sprint
    gpt2 = sprint("sprint-04-gpt2").gpt2          # Sprint 4's src/gpt2.py

It's YOUR code from those sprints, imported live, not a copy: finish
Sprint 4 first, or these tests fail with its NotImplementedErrors.
"""

import importlib
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sprint(folder):
    name = "sprint_" + folder.replace("-", "_")
    if name not in sys.modules:
        src = os.path.join(REPO, folder, "src")
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(src, "__init__.py"), submodule_search_locations=[src]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)

    class _Submodules:
        def __getattr__(self, sub):
            return importlib.import_module(f"{name}.{sub}")

    return _Submodules()


def gpt2_dir():
    """Where Sprint 4's download put the real weights."""
    return os.path.join(REPO, "sprint-04-gpt2", "data", "gpt2")
