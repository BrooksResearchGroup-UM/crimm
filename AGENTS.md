# Instructions for AI agents

crimm prepares biomolecular structures for MD simulation and ML training (fetch, organize,
build loops, assign topology, solvate, write PSF/CRD). It must stay lightweight: heavy
dependencies are optional extras, imported lazily.

**Before doing anything, read [`docs/dev/AGENT_HANDOFF.md`](docs/dev/AGENT_HANDOFF.md)**
sections 0 to 4: protocol, the owner's standing instructions, environment facts and traps.
Current state, claims and the session log are in the open GitHub issue labelled
`agent-log` (`gh issue list --label agent-log --state open`): read its description and
newest three comments. Work is tracked as GitHub issues and indexed in
[`docs/dev/ROADMAP.md`](docs/dev/ROADMAP.md); standards are in
[`docs/dev/BEST_PRACTICES.md`](docs/dev/BEST_PRACTICES.md); developer rules, which bind you
too, are in [`CONTRIBUTING.md`](CONTRIBUTING.md). When you finish, post what you did as a
comment on that issue and update its State (handoff, section 0). That needs no pull request.

The rules that matter most, in case you read nothing else:

- Do not commit or push unless the owner asks. That permission does not carry over to a later
  session.
- Sign commits, never files: a `Co-Authored-By:` trailer naming your exact model; no author
  names, banners or session notes in any file.
- Offline default run: `pytest -p no:cacheprovider -q` (about 45 s warm). Other layers are
  selected by marker: `network`, `pycharmm`, `cgenff`, `slow`, `gpu`. How the `crimm-dev`
  environment is built is in the handoff, section 3; paths and machine details of the owner's
  setup are in `CLAUDE.local.md`.
- `import crimm` must not import rdkit, nglview, ipywidgets or py3Dmol. Import optional
  packages inside the function that uses them. No module-level `import crimm_ml` either.
- Public API may break, but old import paths keep working with a `DeprecationWarning`.
- Macroscope reviews every pull request automatically, when it is opened and on each push,
  except pull requests labelled `documentation`: give that label to documentation-only pull
  requests and to no others. Apply the suggested fixes that are right, answer the ones that
  are not, and leave findings without a fix, or needing a design decision, to the owner with
  a proposed solution. The owner merges.
  Stan's files (`Solvator.py`, `CrystalSDF.py`, `CoordManipulator.py`) follow the same rules;
  his review is no longer requested. Details: CONTRIBUTING, "Review and merge".
- A bug you find but do not fix gets an `xfail(strict=True)` test with the reason and a
  GitHub issue. Remove the marker in the change that fixes it.
- One concern per change. A fix does not ride along with a rename.
- A `TODO` in code carries an issue number.
- Never take timings on the shared login node.
- Personal notes of the owner live in `CLAUDE.local.md`, which is not in the repository. Facts
  about the owner's machine and account, and about private downstream projects, go there, not
  in `docs/dev/`.
