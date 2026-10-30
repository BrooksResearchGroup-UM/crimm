"""PSF and CRD output compared with reviewed reference files in ``tests/data/golden``.

A refactor that changes the output fails here and prints a diff. After an intended change,
run ``pytest tests/test_golden.py --update-golden`` and commit the regenerated files with a
``CHANGELOG.md`` entry; the changed bytes then show in the pull request.

Golden files must not encode known bugs. Each of these fixes adds its file:
the 1CRN PSF waits for #112, 3PTB for #46, 1BNA for #45.
"""

import difflib
import warnings

import numpy as np
import pytest
from conftest import DATA_DIR

from crimm.IO import get_crd_str, get_psf_str

GOLDEN_DIR = DATA_DIR / "golden"
PSF_IDS = ["1ubq", "2igd"]
CRD_IDS = ["1ubq", "1crn", "2igd"]

# TODO(#120): remove once the writer gives these sections a defined order and direction.
# Section marker -> atoms per term, terms per line.
UNORDERED_SECTIONS = {"!NTHETA": (3, 3), "!NPHI": (4, 2)}


def drop_title(text):
    """Remove the title lines, which carry a timestamp and the user name."""
    return "".join(
        line for line in text.splitlines(keepends=True) if not line.startswith("*")
    )


def sort_unordered_terms(psf_text):
    """Put the angle and dihedral sections in a defined order and direction."""
    lines = iter(psf_text.split("\n"))
    result = []
    for line in lines:
        result.append(line)
        shape = next(
            (shape for key, shape in UNORDERED_SECTIONS.items() if key in line), None
        )
        if shape is None:
            continue
        n_atoms, per_line = shape
        indices = []
        for body in lines:
            if not body.strip():
                break
            indices.extend(int(index) for index in body.split())
        terms = sorted(
            min(term, term[::-1])
            for term in (
                tuple(indices[i : i + n_atoms]) for i in range(0, len(indices), n_atoms)
            )
        )
        for i in range(0, len(terms), per_line):
            row = [index for term in terms[i : i + per_line] for index in term]
            result.append("".join(f"{index:>10}" for index in row))
        result.append("")
    return "\n".join(result)


def normalized_psf(model):
    return sort_unordered_terms(drop_title(get_psf_str(model)))


def normalized_crd(model):
    return drop_title(get_crd_str(model))


def assert_matches_golden(text, filename, request):
    path = GOLDEN_DIR / filename
    if request.config.getoption("--update-golden"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"{path} is missing; create it with --update-golden"
    expected = path.read_text()
    if text != expected:
        diff = difflib.unified_diff(
            expected.splitlines(),
            text.splitlines(),
            f"golden/{filename}",
            "current output",
            lineterm="",
        )
        shown = "\n".join(list(diff)[:20])
        pytest.fail(
            f"{filename} differs from its golden file. If the change is intended, run "
            f"pytest --update-golden and add a CHANGELOG entry.\n{shown}",
            pytrace=False,
        )


@pytest.fixture
def seeded_model(prepared_model):
    """Factory: fresh model whose water hydrogens are at reproducible positions.

    ``TopologyGenerator`` places them with the global NumPy generator, whose stream for a
    given seed does not change between NumPy versions.
    """

    def build(pdb_id):
        state = np.random.get_state()
        np.random.seed(0)
        try:
            return prepared_model(pdb_id)
        finally:
            np.random.set_state(state)

    return build


@pytest.mark.parametrize("pdb_id", PSF_IDS)
def test_psf_matches_golden(seeded_model, request, pdb_id):
    text = normalized_psf(seeded_model(pdb_id))
    assert_matches_golden(text, f"{pdb_id}.psf", request)


@pytest.mark.parametrize("pdb_id", CRD_IDS)
def test_crd_matches_golden(seeded_model, request, pdb_id):
    text = normalized_crd(seeded_model(pdb_id))
    assert_matches_golden(text, f"{pdb_id}.crd", request)


def test_title_is_the_only_part_dropped():
    text = "* title\n*\n   1 !NATOM\n  * not a title\n"
    assert drop_title(text) == "   1 !NATOM\n  * not a title\n"


def test_sorting_terms_ignores_order_and_direction():
    row = "".join(f"{index:>10}" for index in (7, 8, 9, 3, 2, 1, 4, 5, 6))
    text = f"   3 !NTHETA: angles\n{row}\n\n   0 !NPHI: dihedrals\n\n\n"
    expected = "".join(f"{index:>10}" for index in (1, 2, 3, 4, 5, 6, 7, 8, 9))
    assert sort_unordered_terms(text) == (
        f"   3 !NTHETA: angles\n{expected}\n\n   0 !NPHI: dihedrals\n\n\n"
    )


@pytest.mark.xfail(
    strict=True,
    reason="#120: angles and dihedrals are collected in sets, and DisorderedAtom hashes by "
    "address, so two builds of 2IGD in one process write them in a different order",
)
def test_psf_term_order_is_reproducible(prepared_model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = prepared_model("2igd")
        second = prepared_model("2igd")
    assert drop_title(get_psf_str(first)) == drop_title(get_psf_str(second))
