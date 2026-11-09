"""Optional dependencies stay optional: `import crimm` needs only the core (#58).

Each check runs in a fresh interpreter, with an import hook that makes the
listed packages look uninstalled.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

OPTIONAL = ("rdkit", "nglview", "ipywidgets", "py3Dmol")
TESTS_DIR = Path(__file__).parent

BLOCKER = """
import importlib.abc, sys
BLOCKED = {blocked!r}
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{{name}} is blocked for this test")
sys.meta_path.insert(0, _Block())
sys.path.insert(0, {tests_dir!r})
"""


def run(code, blocked=()):
    """Run code in a fresh interpreter with `blocked` packages hidden."""
    script = BLOCKER.format(blocked=set(blocked), tests_dir=str(TESTS_DIR))
    script += textwrap.dedent(code)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_import_crimm_without_optional_packages():
    run(
        """
        import crimm, crimm.Adaptors, crimm.Modeller, crimm.Superimpose
        import crimm.Visualization
        """,
        blocked=OPTIONAL,
    )


def test_import_crimm_loads_no_optional_package():
    code = """
        import sys, crimm, crimm.Visualization, crimm.Adaptors
        print(sorted(m for m in OPTIONAL if m in sys.modules))
        """
    out = run(code.replace("OPTIONAL", repr(OPTIONAL)))
    assert out.strip() == "[]"


def test_ngl_show_without_nglview_raises_import_error():
    out = run(
        """
        from conftest import parse_structure
        from crimm import Visualization
        chain = parse_structure("1crn").models[0].chains[0]
        try:
            Visualization.show(chain)
        except ImportError as exc:
            print("ImportError:", exc)
        """,
        blocked=("nglview",),
    )
    assert "ImportError" in out and "nglview" in out


def test_py3dmol_backend_works_without_nglview():
    pytest.importorskip("py3Dmol")
    out = run(
        """
        from conftest import parse_structure
        from crimm import Visualization
        Visualization.set_backend("py3dmol")
        chain = parse_structure("1crn").models[0].chains[0]
        print(type(Visualization.show(chain)).__module__)
        """,
        blocked=("nglview", "ipywidgets"),
    )
    assert out.strip() == "py3Dmol"


def test_cgenff_loader_needs_rdkit_only_when_constructed():
    out = run(
        """
        from crimm.Modeller.TopoLoader import CGENFFTopologyLoader
        try:
            CGENFFTopologyLoader()
        except ImportError as exc:
            print("ImportError:", exc)
        """,
        blocked=("rdkit",),
    )
    assert "ImportError" in out and "rdkit" in out


def test_lazy_names_resolve_when_installed():
    pytest.importorskip("rdkit")
    pytest.importorskip("nglview")
    from crimm.Adaptors import RDKitHetConverter
    from crimm.Adaptors.RDKitConverter import RDKitHetConverter as direct
    from crimm.Visualization import NGLView, show_nglview
    from crimm.Visualization.NGLVisualization import View
    from crimm.Visualization.NGLVisualization import show_nglview as ngl_show

    assert RDKitHetConverter is direct
    assert (NGLView, show_nglview) == (View, ngl_show)


def test_unknown_names_still_raise_attribute_error():
    import crimm.Adaptors
    import crimm.Visualization

    with pytest.raises(AttributeError):
        crimm.Adaptors.NoSuchName  # noqa: B018
    with pytest.raises(AttributeError):
        crimm.Visualization.NoSuchName  # noqa: B018
