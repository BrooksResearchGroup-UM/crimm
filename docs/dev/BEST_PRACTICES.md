# Good practices for crimm

Recommendations for keeping crimm a long-lived, high-quality open-source project. Each one
says what to do and why it matters here. They are targets, not a description of the current
state; [NOTES.md](NOTES.md) has that, and [ROADMAP.md](ROADMAP.md) schedules the work.

Where something is a suggestion to decide on and not a clear win, it says so.

## The short list

If only ten things happen, these give the most for the effort:

1. Every pull request runs tests and a linter in CI before merge.
2. Every bug fix comes with a test that fails without it.
3. Scientific output (PSF, CRD, charges, box sizes) is checked against reference files.
4. `import crimm` needs only the core dependencies; everything else is an extra.
5. One version number, in one place.
6. A changelog entry for every user-visible change.
7. No direct pushes to `master`; a second person reviews changes to the areas they own.
8. Library code logs or raises; it does not `print`.
9. A `TODO` in the code carries an issue number.
10. Nothing generated is committed (built docs, notebook outputs, logs).

## Code

### Style and static checks

- Use **ruff** for linting and formatting, configured in `pyproject.toml`, and run it through
  **pre-commit** so problems are caught before a commit. Start with the `F`, `E`, `B`, `I`
  and `UP` rule sets and widen later.
- Reformat the whole codebase once, in one commit with no other changes, and list that
  commit in `.git-blame-ignore-revs` so `git blame` stays useful.
- New modules use snake_case names. Existing CamelCase modules are renamed in the planned
  restructure, not piecemeal.

### Types and docstrings

- Type-hint every public function. Scientists read signatures to learn what a function
  takes; hints also let editors help them in notebooks.
- Use NumPy-style docstrings everywhere (Sphinx napoleon already renders them). State
  **units** for every physical quantity: Å, e, mol/L, degrees.
- Say in the docstring whether a function modifies its argument. Several do (`solvate`,
  `orient_coords`, topology generation), and with shared atom objects the effect reaches the
  parent structure.
- Put a short runnable example in the docstring of each main entry point, and run them with
  doctest in CI.

### Errors, warnings and logging

- Raise specific exceptions derived from one `CrimmError` base, so pipeline code can catch
  crimm failures without catching everything.
- Never use `assert` for input or state validation. It vanishes under `python -O`.
- Use `logging` for progress and diagnostics, `warnings.warn` only for things the caller
  should change (deprecated arguments, a coerced residue name). No `print` in library code;
  a library that prints cannot be silenced in a 5,000-structure batch job.
- Never catch bare `Exception` without re-raising or logging what was caught.

### State and side effects

- Avoid module-level mutable state. pyCHARMM forces one global PSF per process; keep the
  bookkeeping for that in one clearly named object and document that it is per-process.
- Do not write to the current working directory by default. Take an explicit path or use a
  temporary directory.
- Prefer returning a new object, or make in-place behaviour obvious in the name and
  docstring. Offer `inplace=False` where a copy is affordable.

### Optional dependencies

- Core dependencies are only what the parse, organize, topology, write path needs.
- Import an optional package inside the function or class that uses it. When it is missing,
  raise an `ImportError` that names the extra: `pip install "crimm[viz]"`.
- A CI job installs core only and asserts that optional packages are not in `sys.modules`
  after `import crimm`. Without this guard the problem comes back with the next convenience
  import.

### Numerics and science-specific code

- Give constants a name, a unit and a source. Keep them in one module.
- Any algorithm with randomness (Monte Carlo ion placement) accepts a seed or a
  `numpy.random.Generator`, so a prepared system can be reproduced exactly.
- Coordinate transforms must be proper rotations. Test that handedness is preserved; the
  mirrored-structure bug in the roadmap is what happens without that test.
- Use one matrix convention for transforms throughout and state it once.
- Compare floats with tolerances chosen for the quantity, and write the tolerance down.
- Watch memory scaling. Structures of 100,000 atoms and more are normal after solvation, so
  anything quadratic in atom count needs a reason.

## Testing

- **Layers.** Fast offline unit tests on small fixture structures run on every push.
  Integration tests needing the network, pyCHARMM or CGenFF are marked and run on a
  schedule. The PDB-wide benchmark is a third layer, run before releases; track its
  per-stage success rate over time, since that number is the best single measure of
  crimm's robustness.
- **Fixtures.** Keep a few small structures in the repository, chosen to cover the hard
  cases: altlocs, insertion codes, disulfides, missing loops, a ligand, a nucleic acid,
  a modified residue, more than 26 chains.
- **Golden files.** Check PSF and CRD output against stored references, ideally files
  produced by CHARMM itself for the same system. This is what makes large refactors safe.
- **Invariants.** Some properties should hold for any input and are cheap to test: total
  charge is an integer after topology generation, atom count matches the PSF, chirality is
  preserved by every transform, write-then-read returns the same structure.
- **Regression.** Every bug fix adds the test that would have caught it.
- **Matrix.** Test the oldest and newest supported Python and NumPy. NumPy 2 already broke
  one call in crimm unnoticed.
- **Coverage.** Measure it and stop it from falling; do not chase a number.
- **Notebooks.** Execute the tutorials in CI (`pytest --nbmake` or similar) so they cannot
  silently rot.

## Scientific validity and reproducibility

- State which force-field release the bundled toppar files come from, and record the source
  and date in `crimm/Data/toppar/README`. Users must cite the force field and need to know
  the version.
- Record provenance in output files: crimm version, force-field version and the options
  used. The PSF and CRD title lines are the natural place.
- Cite the method next to the implementation and in the docs: SPLIT and SLTCAP for ion
  counts, CGenFF, PropKa, AlphaFold DB for loop templates.
- When a bug could have produced wrong structures, say so plainly in the changelog and
  describe how users can check their existing systems. In scientific software this matters
  more than the fix itself.
- Add a `CITATION.cff` so users can cite crimm, and archive releases on Zenodo for a DOI.

## Packaging and release

- **One version source.** Either `setuptools-scm` (version from the git tag) or a single
  `__version__` read by `pyproject.toml`. Expose `crimm.__version__`.
- **Versioning policy.** Calendar versions are fine; write down what the parts mean and the
  compatibility promise. A workable one: deprecations last at least two releases, and
  removals are listed in the changelog one release ahead.
- **Changelog.** `CHANGELOG.md` in the Keep a Changelog layout, updated in the same pull
  request as the change.
- **Releases from CI.** A tag triggers build, test and upload to PyPI through trusted
  publishing, so no long-lived API token exists and a release never depends on one laptop.
- **Check the artifact.** In CI, build the wheel, install it into a clean environment and
  run the tests against the installed copy. This catches data files missing from the wheel.
- **Supported versions.** State the supported Python and NumPy range and drop old versions
  on a schedule (the scientific-Python SPEC 0 schedule is a reasonable default).
- **conda-forge.** Most of the audience uses conda, and rdkit, OpenMM and propka install
  most reliably from there. A feedstock is worth the setup.
- **Package size.** The toppar files are 11 MB. Keep an eye on wheel size and never let
  notebooks or built docs into the sdist.

## Documentation

- Organise by what the reader wants: a tutorial for newcomers, task-focused how-to guides
  (solvate a system, parameterize a ligand, build a loop), the API reference, and short
  explanations of design (the entity model, how topology is applied).
- Build the docs in CI from `master` and deploy from there. Do not commit build output.
- Render the tutorial notebooks into the docs site (`nbsphinx` or `myst-nb`), with outputs
  produced at build time, so the repository stores them without outputs.
- Keep the README short: what it is, install, one example, links. Test the README example.
- Document the visualization backends and what each needs, since that is a headline feature.

## Project management

### Repository

- Protect `master`: changes arrive by pull request, CI must pass, and one approval is
  required. With two maintainers this is light, and it is what keeps solvation changes in
  front of Stan and topology changes in front of Truman.
- `CODEOWNERS` to request those reviews automatically.
- Short-lived branches, deleted on merge. Clear out the roughly 35 old remote branches and
  keep one default branch.
- Squash-merge, with a commit message that explains why the change was made.

### Community files

- `CONTRIBUTING.md`: how to set up a development environment, run tests, and what a pull
  request needs. This is the single most useful file for a first outside contributor.
- `CODE_OF_CONDUCT.md`, `SECURITY.md` (how to report a problem privately), issue templates
  for bug reports (asking for the PDB ID, crimm version and traceback) and feature requests,
  and a pull-request template with a short checklist.

### Planning

- Track work as GitHub issues grouped into milestones that mirror the roadmap phases.
  Convert code TODOs to issues; a TODO without an issue number does not get merged.
- A few labels are enough: `bug`, `correctness` (wrong scientific output), `enhancement`,
  `refactor`, `docs`, `good first issue`, plus area labels.
- Treat `correctness` bugs as release blockers.

### Maintenance

- Dependabot or Renovate for GitHub Actions and pinned development tools.
- A scheduled CI run against the newest releases of numpy, scipy and biopython, so breakage
  is seen when it lands upstream and not when a user reports it. This matters most for
  Biopython, whose classes crimm subclasses.
- Write down who can publish to PyPI and make sure more than one person can.

## Licensing

- crimm is GPLv3. That means software that imports crimm and is distributed must be
  GPL-compatible, which can discourage use in commercial pipelines and by projects under
  permissive licences. Whether that is intended is a decision for the owner; it is worth
  making deliberately and stating in the README. Changing licence later needs the agreement
  of every contributor, so it is easier to settle while there are two.
- Confirm and document the redistribution terms of the bundled CHARMM toppar files.
- Check that the licence of each core dependency is compatible, and note that pyCHARMM and
  CHARMM, and the CGenFF program, have their own terms that users must meet separately.

## Working with AI coding tools

Part of this codebase was written with Copilot and Claude, and that will continue.

- Generated code gets the same review as any other, with extra attention to numerical
  details: sign conventions, matrix layout, units. The orientation bugs in the roadmap sit
  in code that reads plausibly and is well commented.
- Ask for tests in the same change, and read the tests first.
- Keep work orders for agents as GitHub issues, not loose Markdown files at the repo root.
- Do not commit agent scratch files or local tool settings; add them to `.gitignore`.
