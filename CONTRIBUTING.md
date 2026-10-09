# Contributing to crimm

For everyone who changes this repository, human or AI. AI agents get their working
instructions from [AGENTS.md](AGENTS.md); the rules here bind them too.

## Set up

```bash
conda env create -f environment-dev.yml      # creates `crimm-dev`
conda activate crimm-dev
pip install -e ".[all,test]" --no-deps
```

CHARMM, pyCHARMM and the CGenFF program are not installable from conda. Tests that need them
are skipped unless you set them up (`CRIMM_CGENFF_PATH` for CGenFF).

**CHARMM needs a licence.** CHARMM (and so pyCHARMM) is licensed software, and its source and
binaries are not available to contributors who do not hold a licence. You do not need it to
contribute: the default test run is offline and never touches it, `pycharmm` tests are skipped,
and CI cannot run them. If you change `crimm/Adaptors/pyCHARMMAdaptors.py` without access to
CHARMM, say so in the pull request so a maintainer who has it can run `pytest -m pycharmm`.
Never add CHARMM source, binaries or libraries to the repository or to an issue.

## Run the tests

```bash
pytest -q                      # offline and fast; the default
pytest -m network              # needs internet (RCSB, AlphaFold DB)
pytest -m "cgenff or slow"
pytest -m ""                   # everything
```

Markers: `network`, `pycharmm`, `cgenff`, `slow`, `gpu`. Every bug fix adds a test that fails
without it. The standards behind this are in
[docs/dev/BEST_PRACTICES.md](docs/dev/BEST_PRACTICES.md).

## Changing code

- One concern per commit and per pull request. A fix does not ride along with a rename.
- `import crimm` needs only the core dependencies. Import optional packages (rdkit, nglview,
  ipywidgets, py3Dmol, propka) inside the function that uses them, and raise an `ImportError`
  that names the extra.
- Library code logs or raises; it does not `print`. Do not use `assert` to validate input.
- Public API may break, but old import paths keep working with a `DeprecationWarning`.
- Solvation code (`Solvator.py`, `CrystalSDF.py`, PCA methods of `CoordManipulator.py`) is
  Stan's area; topology code is Truman's. Ask them to review changes there.
- A `TODO` in code carries an issue number.

## Issues and pull requests

- Work is tracked as GitHub issues, grouped in milestones that mirror the phases of
  [docs/dev/ROADMAP.md](docs/dev/ROADMAP.md). Use the issue forms.
- Open a pull request from a short-lived branch. It needs passing tests and a description of
  why the change is made. Squash-merge; delete the branch after.
- A correctness bug (wrong scientific output) blocks the next release.

## Credit and authorship

- Every commit carries its authors. An AI model that wrote a substantial part signs with a
  trailer naming the exact model: `Co-Authored-By: <model name and version> <vendor
  no-reply address>`.
- Source files carry no signatures: no author lines, "generated with" banners, model names or
  session notes in code, comments, docstrings or file names. `git log` and `git blame` are
  the record.
- Do not sign for code ported or derived from another project as if it were original.
- Never remove or rewrite someone else's credit.
