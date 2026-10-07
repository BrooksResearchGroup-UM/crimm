# Instructions for AI agents

crimm prepares biomolecular structures for MD simulation and ML training (fetch, organize,
build loops, assign topology, solvate, write PSF/CRD). It must stay lightweight: heavy
dependencies are optional extras, imported lazily.

**Before doing anything, read [`docs/dev/AGENT_HANDOFF.md`](docs/dev/AGENT_HANDOFF.md)**
sections 0 to 5: protocol, the owner's standing instructions, environment facts, traps,
current state and claims. Work is tracked as GitHub issues and indexed in
[`docs/dev/ROADMAP.md`](docs/dev/ROADMAP.md); standards are in
[`docs/dev/BEST_PRACTICES.md`](docs/dev/BEST_PRACTICES.md); developer rules, which bind you
too, are in [`CONTRIBUTING.md`](CONTRIBUTING.md). When you finish, log what you did in the
handoff.

The rules that matter most, in case you read nothing else:

- Do not commit or push unless the owner asks. That permission does not carry over to a later
  session.
- Sign commits, never files: a `Co-Authored-By:` trailer naming your exact model; no author
  names, banners or session notes in any file.
- Offline default run: `pytest -p no:cacheprovider -q` (about 45 s warm). Other layers are
  selected by marker: `network`, `pycharmm`, `cgenff`, `slow`, `gpu`. Python and environment
  details are in the handoff, section 3.
- `import crimm` must not import rdkit, nglview, ipywidgets or py3Dmol. Import optional
  packages inside the function that uses them. No module-level `import crimm_ml` either.
- Public API may break, but old import paths keep working with a `DeprecationWarning`.
- `crimm/Modeller/Solvator.py`, `CrystalSDF.py` and the PCA methods in `CoordManipulator.py`
  are Stan's area: propose, and flag for his review.
- A bug you find but do not fix gets an `xfail(strict=True)` test with the reason and a
  GitHub issue. Remove the marker in the change that fixes it.
- One concern per change. A fix does not ride along with a rename.
- A `TODO` in code carries an issue number.
- Never take timings on the shared login node.
- Personal notes of the owner live in `CLAUDE.local.md`, which is not in the repository.
