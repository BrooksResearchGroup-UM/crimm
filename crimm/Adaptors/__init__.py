# RDKitHetConverter needs rdkit, an optional extra. It is loaded on first
# access so that `import crimm.Adaptors` works without rdkit (#58).
_LAZY = {"RDKitHetConverter": "crimm.Adaptors.RDKitConverter"}

__all__ = ["RDKitHetConverter"]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
