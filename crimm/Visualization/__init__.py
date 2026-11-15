# The backends need nglview or py3Dmol, both in the `viz` extra. Their names are
# loaded on first access, so `import crimm.Visualization` works without them (#58).
_LAZY = {
    "NGLStructure": ("NGLStructure", ".NGLVisualization"),
    "show_nglview": ("show_nglview", ".NGLVisualization"),
    "show_nglview_multiple": ("show_nglview_multiple", ".NGLVisualization"),
    "show_nglview_residue": ("show_nglview_residue", ".NGLVisualization"),
    "NGLView": ("View", ".NGLVisualization"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        attr, module = _LAZY[name]
        return getattr(importlib.import_module(module, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


# Each backend module defines show, show_multiple, show_residue and View. The public
# functions below look the backend up at every call, so set_backend() reaches callers
# that imported them earlier, and a backend is imported only when first used (#145).
_BACKEND_MODULES = {"ngl": ".NGLVisualization", "py3dmol": ".py3DmolVis"}
_BACKENDS = tuple(_BACKEND_MODULES)
_backend = "ngl"


def _backend_module():
    """Import and return the module of the active backend."""
    import importlib

    return importlib.import_module(_BACKEND_MODULES[_backend], __name__)


def set_backend(backend):
    """Switch the active visualization backend.

    Parameters
    ----------
    backend : {'ngl', 'py3dmol'}
    """
    global _backend
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose from {_BACKENDS}.")
    _backend = backend


def get_backend():
    """Return the name of the currently active visualization backend."""
    return _backend


def show(entity, **kwargs):
    """Show entity using the active backend (see :func:`set_backend`).

    Both backends accept ``width`` and ``height`` (pixels).
    """
    return _backend_module().show(entity, **kwargs)


def show_multiple(entity_list, **kwargs):
    """Show a list of entities using the active backend."""
    return _backend_module().show_multiple(entity_list, **kwargs)


def get_viewer(**kwargs):
    """Return a new View instance for the active backend."""
    return _backend_module().View(**kwargs)


def show_residue(residue, **kwargs):
    """Show a residue using the active backend.

    When the residue belongs to a polymer chain the whole chain is rendered as
    a subdued grey cartoon and the residue is highlighted as ball+stick.
    """
    return _backend_module().show_residue(residue, **kwargs)
