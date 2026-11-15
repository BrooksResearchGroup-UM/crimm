# Deprecated alias: RDKitHetConverter lives in crimm.Adaptors.RDKitConverter. The
# alias is loaded on first access, so `import crimm.Adaptors` works without rdkit
# (#58), and it warns when used.
# TODO(#97): remove this shim when its deprecation period ends; the file is then empty.
_DEPRECATED = {"RDKitHetConverter": "crimm.Adaptors.RDKitConverter"}


def __getattr__(name):
    if name in _DEPRECATED:
        import importlib
        import warnings

        module = _DEPRECATED[name]
        warnings.warn(
            f"Importing {name} from crimm.Adaptors is deprecated and will be "
            f"removed in a future release; use `from {module} import {name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
