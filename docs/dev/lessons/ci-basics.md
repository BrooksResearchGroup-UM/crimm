# Lint and continuous integration: what they are and why each step exists

Written 2026-10-12 for the project owner, who had not set these up before. The workflow
described is [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml); the lint
configuration is in [`pyproject.toml`](../../../pyproject.toml) under `[tool.ruff]`.

## Part 1: ruff, a linter

A **linter** reads Python source without running it and reports likely mistakes, like a
spell-checker for code. In crimm it found, among others:

- imports nobody uses (the F401 rule);
- a name that is never defined: `topo` in `ArcLoopBuilder` (`crimm/Modeller/LoopBuilder.py`),
  a real bug that would raise `NameError`;
- variables assigned and never read (F841);
- imports in inconsistent order (I001).

**ruff** is one tool that replaces several older ones (pyflakes, flake8, isort, pyupgrade).
It is written in Rust, so it checks the whole repository in a fraction of a second, and it is
configured in `pyproject.toml`, next to pytest. It can fix many findings itself (`--fix`). It
also has a formatter (`ruff format`, comparable to Black) that rewrites layout; crimm does not
use that yet, because it would change most files.

Rules come in groups named by letter. crimm enables `F` (real errors), `E4/E7/E9` (the
pycodestyle checks that find real errors), `I` (import order), `B` (likely bugs) and `UP`
(outdated syntax). Rules that would flag hundreds of lines today (for example `B028`, a
missing `stacklevel` in 138 `warnings.warn` calls) are listed under `ignore` with a reason,
so the check can be strict from now on without a giant clean-up first. Start small, widen
later.

How you use it:

```bash
ruff check crimm tests          # report problems; exit code 0 means clean
ruff check --fix crimm tests    # apply the fixes ruff considers safe
```

Two habits: fix the line rather than silence the rule, and when you must silence one,
write `# noqa: F401` with a reason, or add a `per-file-ignores` entry with a comment.

## Part 2: continuous integration (CI)

### The problem

You change code, run the tests on your machine, and push. Three things can still go wrong:
you forgot to run them; they pass on your machine only (your Python, your NumPy, a file you
have but did not commit); or a collaborator breaks something you could not see. CI means a
server runs the checks for you, on a clean machine, every time code is proposed.
"Integration" is merging many people's changes; "continuous" means checking every change, not
only before a release.

### Vocabulary

- **GitHub Actions**: GitHub's CI service. Free for public repositories.
- **Workflow**: one YAML file in `.github/workflows/`. It says when to run and what to do.
- **Trigger** (`on:`): the event that starts it: a push, a pull request, a button, a schedule.
- **Job**: a group of steps that runs on one fresh virtual machine (the "runner",
  `ubuntu-latest`). Jobs run in parallel unless one `needs` another.
- **Step**: one command (`run:`) or one packaged action (`uses:`, e.g. `actions/checkout@v4`).
- **Matrix**: the same job over combinations (Python version x NumPy version). Each
  combination is its own machine and its own green or red result.
- **Required check**: a result GitHub insists is green before a pull request can merge.
  That is what turns CI from advice into a gate.

### What our workflow does, and why

1. **Triggers: pull requests to `master`, pushes to `master`, and a manual button.** A pull
   request is where a change is reviewed; testing it there finds problems before they reach
   `master`. The push trigger re-checks `master` after a merge.
2. **A fresh runner and a fresh checkout.** The runner starts empty. If the tests pass there,
   they do not depend on anything that exists only on your cluster account (the `crimm-dev`
   environment, `~/.local`, `CRIMM_CGENFF_PATH`). This is the main value of CI.
3. **Install the package the way a user does** (`pip install -e ".[all,test]"`). A dependency
   missing from `pyproject.toml` fails here instead of on a user's machine. `[all]` is used
   because `import crimm` still needs rdkit and nglview today (ROADMAP 1.5 will remove that).
4. **Lint is its own job and runs in seconds.** A typo is reported at once and a style
   failure does not hide a test failure.
5. **The linter version is pinned** (`ruff==0.16.10`). Otherwise a new ruff release can add
   rules and turn a green build red with no change from us. Upgrade on purpose.
6. **A test matrix.** `requires-python = ">=3.9"` is a promise to users; without testing 3.9
   to 3.13 the promise is unchecked. NumPy 1.x versus 2.x matters because NumPy 2 removed
   functions (crimm already hit `ndarray.ptp`). The NumPy line is installed first so that
   pip keeps it while resolving the rest. Python 3.13 with NumPy 1.x is excluded because
   NumPy 1.26 never shipped wheels for 3.13: that cell cannot exist.
7. **`fail-fast: false`.** One red cell must not cancel the others. The pattern (only 3.9
   fails? only NumPy 1?) is what tells you the cause.
8. **Offline tests only.** The default pytest markers skip `network`, `cgenff`, `pycharmm`,
   `slow` and `gpu`. Hosted runners have no CHARMM (licensed, never in CI) and no CGenFF, and
   network tests make CI flaky when RCSB is down. A maintainer runs those by hand before a
   release.
9. **`permissions: contents: read`, `concurrency`, `timeout-minutes`.** Least privilege: the
   workflow can read the repository and do nothing else. A newer push cancels the older run.
   A hung job is stopped instead of burning minutes.
10. **One summary job (`all-green`).** Matrix cells have many names, and the names change when
    the matrix changes. Branch protection needs one stable name, so a last job that `needs`
    all the others is the one marked as required.

### Reading a failed run

Open the pull request, click "Details" beside the red check, open the red step and read the
first error, not the last line. Reproduce it locally with the same Python version
(`conda create -n py39 python=3.9`). Then fix the code or the configuration. Do not re-run
until green, and do not delete the failing cell: a real failure gets a fix, or an issue plus
an `xfail(strict=True)` test, or a narrowed matrix with a comment saying why.

### What CI does not do for us yet

- It cannot run `pycharmm` tests (licensed software). A maintainer with a CHARMM build runs
  `pytest -m pycharmm` before releases and after any change to `pyCHARMMAdaptors.py`.
- It does not yet check that `import crimm` works with only the core dependencies. That
  check, which asserts rdkit, nglview, ipywidgets and py3Dmol are not imported, comes with
  ROADMAP 1.5.
- It does not build the documentation or publish to PyPI (Phases 4 and 5).

### Habits worth keeping

- Never merge a red pull request, and never "fix CI" by weakening a check without writing
  down why.
- Keep CI fast, minutes not hours. Slow CI gets bypassed.
- Treat workflow files like code: small commits, review, pinned versions.
- Branch protection (require `all-green`, no direct pushes to `master`) is what makes all of
  this enforceable. Set it after the first green run.
