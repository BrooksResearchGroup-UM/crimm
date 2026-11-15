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


_BACKENDS = ("ngl", "py3dmol")
_backend = "ngl"


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
    if _backend == "ngl":
        from .NGLVisualization import show_nglview

        return show_nglview(entity, **kwargs)
    from .py3DmolVis import show_py3dmol

    return show_py3dmol(entity, **kwargs)


def show_multiple(entity_list, **kwargs):
    """Show a list of entities using the active backend."""
    if _backend == "ngl":
        from .NGLVisualization import show_nglview_multiple

        return show_nglview_multiple(entity_list, **kwargs)
    from .py3DmolVis import show_py3dmol_multiple

    return show_py3dmol_multiple(entity_list, **kwargs)


def get_viewer(**kwargs):
    """Return a new View instance for the active backend."""
    if _backend == "ngl":
        from .NGLVisualization import View as NGLView

        return NGLView(**kwargs)
    from .py3DmolVis import View as Py3DmolView

    return Py3DmolView(**kwargs)


def show_residue(residue, **kwargs):
    """Show a residue using the active backend.

    When the residue belongs to a polymer chain the whole chain is rendered as
    a subdued grey cartoon and the residue is highlighted as ball+stick.
    """
    if _backend == "ngl":
        from .NGLVisualization import show_nglview_residue

        return show_nglview_residue(residue, **kwargs)
    from .py3DmolVis import show_py3dmol_residue

    return show_py3dmol_residue(residue, **kwargs)
