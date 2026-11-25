"""PSF and CRD output compared with reviewed reference files in ``tests/data/golden``.

A refactor that changes the output fails here and prints a diff. After an intended change,
run ``pytest tests/test_golden.py --update-golden`` and commit the regenerated files with a
``CHANGELOG.md`` entry; the changed bytes then show in the pull request.

Golden files must not encode known bugs. Each of these fixes adds its file:
the 1CRN PSF waits for #112, 3PTB for #46, 1BNA for #45.
"""

import difflib
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA_DIR

from crimm.IO import CRDParser, get_crd_str, get_psf_str

GOLDEN_DIR = DATA_DIR / "golden"
PSF_IDS = ["1ubq", "2igd"]
CRD_IDS = ["1ubq", "1crn", "2igd"]


def drop_title(text):
    """Remove the title lines, which carry a timestamp and the user name."""
    return "".join(
        line for line in text.splitlines(keepends=True) if not line.startswith("*")
    )


def normalized_psf(model):
    return drop_title(get_psf_str(model))


def normalized_crd(model):
    # Match PSF/CRD export order: the PSF writer assigns segment IDs in place.
    # TODO(#124): drop this call once the CRD writer no longer needs a PSF written first.
    get_psf_str(model)
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
def test_crd_matches_golden(seeded_model, request, tmp_path, pdb_id):
    text = normalized_crd(seeded_model(pdb_id))
    path = tmp_path / f"{pdb_id}.crd"
    path.write_text(text)
    reread = CRDParser(QUIET=True).get_structure(str(path))
    assert all(residue.segid.strip() for residue in reread.get_residues())
    assert_matches_golden(text, f"{pdb_id}.crd", request)


def test_title_is_the_only_part_dropped():
    text = "* title\n*\n   1 !NATOM\n  * not a title\n"
    assert drop_title(text) == "   1 !NATOM\n  * not a title\n"


def test_psf_term_order_is_reproducible(prepared_model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = prepared_model("2igd")
        second = prepared_model("2igd")
    assert drop_title(get_psf_str(first)) == drop_title(get_psf_str(second))


PSF_DIGEST_SCRIPT = """
import hashlib, sys, warnings
sys.path.insert(0, sys.argv[1])
from conftest import make_prepared_model
from crimm.IO import get_psf_str
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    text = get_psf_str(make_prepared_model(sys.argv[2]))
body = "".join(line for line in text.splitlines(True) if not line.startswith("*"))
print(hashlib.sha256(body.encode()).hexdigest())
"""


def psf_digest_in_subprocess(pdb_id, hash_seed):
    tests_dir = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(tests_dir.parent), env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, "-c", PSF_DIGEST_SCRIPT, str(tests_dir), pdb_id],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()[-1]


def test_psf_is_identical_across_hash_seeds():
    """String hashes, and with them set order, change with ``PYTHONHASHSEED`` (#120)."""
    digests = {seed: psf_digest_in_subprocess("1ubq", seed) for seed in (1, 2)}
    assert digests[1] == digests[2]


def psf_terms(text, marker, n_atoms):
    lines = iter(text.splitlines())
    for line in lines:
        if marker in line:
            break
    indices = []
    for line in lines:
        if not line.strip():
            break
        indices.extend(int(index) for index in line.split())
    return [tuple(indices[i : i + n_atoms]) for i in range(0, len(indices), n_atoms)]


@pytest.mark.parametrize("marker, n_atoms", [("!NTHETA", 3), ("!NPHI", 4)])
def test_psf_terms_are_sorted_and_directed(shared_prepared, marker, n_atoms):
    """Each angle or dihedral starts from its end with the lower serial, and the terms
    are written in ascending order of serials."""
    terms = psf_terms(get_psf_str(shared_prepared("1ubq")), marker, n_atoms)
    assert terms
    assert all(term <= term[::-1] for term in terms)
    assert terms == sorted(terms)
