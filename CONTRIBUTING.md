# Contributing to crimm

For everyone who changes this repository, human or AI. AI agents get their working
instructions from [AGENTS.md](AGENTS.md); the rules here bind them too.

## Set up

```bash
conda env create -f environment-dev.yml      # creates `crimm-dev`
conda activate crimm-dev
pip install -e ".[all,test]" --no-deps
pre-commit install                           # ruff check and ruff format on every commit
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

`tests/test_golden.py` compares PSF and CRD output with reference files in
`tests/data/golden`. If you change the output on purpose, run
`pytest tests/test_golden.py --update-golden`, and commit the regenerated files in the same pull
request with a changelog fragment, so the reviewer sees exactly what moved.

## Lint

```bash
ruff check crimm tests         # must pass; configured in pyproject.toml
ruff check --fix crimm tests   # applies the safe automatic fixes
ruff format crimm tests        # CI fails if a file is left unformatted
```

`ruff` is in `environment-dev.yml`. If a rule is wrong for a line, prefer fixing the line; if
you must silence it, add `# noqa: <code>` with a reason.

A commit that only lints or reformats goes in its own commit and is listed in
[.git-blame-ignore-revs](.git-blame-ignore-revs), so `git blame` keeps each line's author. Run
`git config blame.ignoreRevsFile .git-blame-ignore-revs` once to make local blame use it.

## Changing code

- One concern per commit and per pull request. A fix does not ride along with a rename.
- A user-visible change adds a fragment to `changelog.d/` (one file per entry, named
  `<issue>.<type>.md`; see [changelog.d/README.md](changelog.d/README.md)). Do not edit
  `CHANGELOG.md` directly; towncrier writes the fragments into it at release.
- `import crimm` needs only the core dependencies. Import optional packages (rdkit, nglview,
  ipywidgets, py3Dmol, propka) inside the function that uses them, and raise an `ImportError`
  that names the extra.
- Library code logs or raises; it does not `print`. Do not use `assert` to validate input.
- Public API may break, but old import paths keep working with a `DeprecationWarning`.
- Solvation code (`Solvator.py`, `CrystalSDF.py`, PCA methods of `CoordManipulator.py`) is
  Stan's area; topology code is Truman's. Their review is not requested by default; every
  code change goes through the Macroscope review below.
- A `TODO` in code carries an issue number.

## Issues and pull requests

- Work is tracked as GitHub issues, grouped in milestones that mirror the phases of
  [docs/dev/ROADMAP.md](docs/dev/ROADMAP.md). Use the issue forms.
- **This project prefers small pull requests.** Each one has a single concern and can be read
  in one sitting; split a large change into a series. They are quicker to review, whether
  before or after the merge, and easier to revert.
- Open a pull request from a short-lived branch. It needs passing tests and a description of
  why the change is made. Squash-merge; delete the branch after. Exception: a pull request
  that adds a commit to `.git-blame-ignore-revs` is merged with a merge commit, because
  squashing rewrites the commit that file names.
- A correctness bug (wrong scientific output) blocks the next release.

### Review and merge

Adopted 2026-10-28, revised 2026-11-21 (Macroscope review, then made automatic). Reviewer time is limited, so every pull request that
changes code is first reviewed by [Macroscope](https://docs.macroscope.com), an AI reviewer
installed on this repository, and people review afterwards, in batches.

- **Macroscope reviews every pull request automatically**, Stan's files included: when it is
  opened and again on each push. The one exception is a pull request labelled
  `documentation`, which it skips; give that label only to pull requests that change nothing
  but documentation. To review a skipped pull request or re-run a review, comment
  `@macroscope-app review`.
- **Acting on its findings.** A finding with a suggested fix that is right is applied, with a
  test, in a commit that names the review; reply on its thread with the commit. A suggested
  fix that is wrong or incomplete gets a reply saying why, and what was done instead. A
  finding without a suggested fix, or one that needs a design decision, is left open for the
  owner with a proposed solution. A thread the owner has answered follows his answer.
- **The owner merges, people review later.** Once the `all-green` check passes and the owner
  has decided on the open findings, he (`@Truman-Xu`) merges the pull request and labels it
  `review-after-merge`. Branch protection still asks for one approval, so he merges with admin
  rights (`gh pr merge --admin`). Agents never merge. Merge rules that need less of him may
  come later.
- **Stan's files follow the same rules** (since 2026-11-21; before, they waited for his
  approval). Keep changes to them in their own pull request all the same.
- **Batches.** Before a review session, post the list of `review-after-merge` pull requests
  in merge order, with what to check in each and what can be skimmed. The reviewer removes
  the label when done.
- **Findings after merge** become an issue or a follow-up pull request. A merged change that
  produces wrong scientific output is reverted first.
- **Stack branches at most two deep.** A branch that does not need unmerged work starts from
  `master`.

## Credit and authorship

- Every commit carries its authors. An AI model that wrote a substantial part signs with a
  trailer naming the exact model: `Co-Authored-By: <model name and version> <vendor
  no-reply address>`.
- Source files carry no signatures: no author lines, "generated with" banners, model names or
  session notes in code, comments, docstrings or file names. `git log` and `git blame` are
  the record.
- Do not sign for code ported or derived from another project as if it were original.
- Never remove or rewrite someone else's credit.
