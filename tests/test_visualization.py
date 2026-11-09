"""Visualization: every show/highlight path follows set_backend (#140)."""

import inspect
import warnings

import pytest

nv = pytest.importorskip("nglview")
py3Dmol = pytest.importorskip("py3Dmol")

from crimm import Visualization  # noqa: E402
from crimm.Modeller.LoopBuilder import ChainLoopBuilder  # noqa: E402
from crimm.Superimpose import ChainSuperimposer  # noqa: E402
from crimm.Visualization import NGLVisualization, py3DmolVis  # noqa: E402

FIGURE = {"ngl": nv.NGLWidget, "py3dmol": py3Dmol.view}
VIEW = {"ngl": NGLVisualization.View, "py3dmol": py3DmolVis.View}
VIEW_METHODS = [
    "load_entity",
    "load_rdkit",
    "subdue_all_entities",
    "highlight_residues",
    "highlight_atoms",
    "highlight_chains",
    "add_arrow",
]


@pytest.fixture(params=["ngl", "py3dmol"])
def backend(request):
    previous = Visualization.get_backend()
    Visualization.set_backend(request.param)
    yield request.param
    Visualization.set_backend(previous)


@pytest.fixture
def chain(load_structure):
    return load_structure("1crn").models[0].chains[0]


@pytest.fixture
def loop_builder(chain):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder = ChainLoopBuilder(chain)
    builder.set_template_chain(chain.copy())
    return builder


def test_chain_superimposer_show_follows_backend(backend, chain):
    superimposer = ChainSuperimposer()
    superimposer.set_chains(chain, chain.copy())
    assert isinstance(superimposer.show(), FIGURE[backend])


def test_loop_builder_show_follows_backend(backend, loop_builder):
    assert isinstance(loop_builder.show(), FIGURE[backend])


def test_highlight_repaired_gaps_follows_backend(backend, loop_builder):
    loop_builder.repaired_gaps = [[10, 11, 12]]
    assert isinstance(loop_builder.highlight_repaired_gaps(), VIEW[backend])


def test_show_gaps_follows_backend_and_draws_arrows(backend, chain):
    for resseq in (20, 21):
        chain.detach_child((" ", resseq, " "))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder = ChainLoopBuilder(chain)
    assert builder.model_chain.gaps, "fixture should have a gap"
    view = builder.show_gaps()
    assert isinstance(view, VIEW[backend])
    if backend == "py3dmol":
        assert "addArrow" in view.view.startjs


@pytest.mark.parametrize("name", VIEW_METHODS)
def test_view_classes_share_methods_and_parameters(name):
    def params(cls):
        signature = inspect.signature(getattr(cls, name))
        return [
            p.name
            for p in signature.parameters.values()
            if p.kind is not inspect.Parameter.VAR_KEYWORD
        ]

    assert params(NGLVisualization.View) == params(py3DmolVis.View)


def test_size_reaches_the_ngl_widget(chain):
    Visualization.set_backend("ngl")
    widget = Visualization.show(chain, width=500, height=300)
    assert (widget.layout.width, widget.layout.height) == ("500px", "300px")
    viewer = Visualization.get_viewer(width=500)
    assert viewer.layout.width == "500px"


def test_py3dmol_warns_about_ignored_arguments(chain):
    view = py3DmolVis.View()
    view.load_entity(chain)
    with pytest.warns(UserWarning, match="colorScheme"):
        view.highlight_chains([chain], colorScheme="sstruc")


@pytest.mark.parametrize(
    "method, entities, message",
    [("highlight_atoms", "atoms", "Atoms"), ("highlight_chains", "chains", "Chains")],
)
def test_ngl_errors_name_what_was_passed(
    chain, load_structure, method, entities, message
):
    view = NGLVisualization.View()
    view.load_entity(chain)
    other = load_structure("1ubq").models[0].chains[0]
    arg = list(other.get_atoms())[:3] if entities == "atoms" else [other]
    with pytest.raises(ValueError, match=message):
        getattr(view, method)(arg)


@pytest.fixture
def two_conformers():
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=1)
    assert mol.GetNumConformers() == 2
    return mol


def test_sdf_has_one_record_per_conformer(two_conformers):
    from crimm.Visualization._common import rdkit_mol_to_sdf

    assert rdkit_mol_to_sdf(two_conformers).count("$$$$") == 2
    assert rdkit_mol_to_sdf(two_conformers, conf_id=1).count("$$$$") == 1


@pytest.mark.parametrize("cls", [NGLVisualization.View, py3DmolVis.View])
def test_load_rdkit_takes_conf_id(cls, two_conformers):
    cls().load_rdkit(two_conformers, conf_id=1)
