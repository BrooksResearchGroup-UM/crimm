# AGENT_HANDOFF

Shared memory for AI agents working on crimm. Any model, any tool. Written for agents first;
humans read it second.

```yaml
schema: agent-handoff/1
repo: github.com/BrooksResearchGroup-UM/crimm
default_branch: master
last_updated: 2026-11-07
log: open issue labelled agent-log   # gh issue list --label agent-log --state open
```

## 0. Protocol

Read this section fully. Skim the rest for what your task touches.

This file holds what changes rarely: protocol, orientation, the owner's standing
instructions, the environment and gotchas. What changes every session (STATE, CLAIMS and
LOG) is in the open GitHub issue labelled `agent-log` (section 5), so that no session needs
a pull request just to record itself.

### On arrival

1. Read sections 1 to 4. They are short and change rarely.
2. Read the `agent-log` issue: its description (CLAIMS and STATE) and its newest three
   comments (LOG). `N=$(gh issue list --label agent-log --state open --json number -q
   '.[0].number')`, then `gh issue view $N` and `gh issue view $N --comments`.
3. Run `git status --short` and `git log --oneline -5`. If they disagree with STATE, trust
   git and correct STATE.
4. Check CLAIMS before editing a file. If another agent holds it, do not edit it.

### On departure

1. Post one LOG entry as a comment on the `agent-log` issue (template in the issue
   description): `gh issue comment $N --body-file <file>`.
2. Update STATE so it is true now. Re-read the description immediately before, edit it in
   place, and write it back with `gh issue edit $N --body-file <file>`. Do not append.
3. Release your CLAIMS in the same edit.
4. Move anything you learned that will still be true next month into FACTS or GOTCHAS of
   this file. That is a tracked change: include it in the pull request of your work, or open
   a small docs pull request.

Without network access or `gh`, write the entry and the new STATE to local files and post
them in your next session that has both.

When a milestone closes, the agent that closes it opens the next log issue (label
`agent-log`, title `Agent log: <phase>`) with the current STATE, links it from the old one,
then closes the old one and moves the pin.

### Writing rules

- **Tag every claim with how you know it.** Use exactly one of:
  - `[ran]` you executed it and saw the result this session
  - `[read]` you read the code or file; did not execute
  - `[told]` the owner said so
  - `[inferred]` your conclusion; say from what
- **Date everything** as `YYYY-MM-DD`. Never write "today", "recently" or "currently".
- **Cite locations** as `path:line` or `path::symbol`. Prefer the symbol; line numbers rot.
- **State what you did not check.** A missing check written down is worth more than a
  confident guess.
- **One fact, one place.** Link to `NOTES.md` or `ROADMAP.md` instead of copying from them.
- **Do not edit or delete another agent's entry.** If it is wrong, say so in your own
  comment: `SUPERSEDED YYYY-MM-DD by <agent>: <why>`, with a link to the comment.
- **Identify yourself** as `<model>/<tool>/<short task name>`, for example
  `claude-opus/claude-code/cleanup`, in the heading of your log comment. Comments appear
  under the GitHub account the agent runs as, which is usually a person's; the heading is
  what tells sessions apart.
- **The log is public**, like this file. Nothing about a particular machine or account, and
  nothing about private downstream projects; that goes in the owner's `CLAUDE.local.md`.
- Keep entries terse. No preamble, no summaries of summaries.

### What goes where

| Kind of information | Place |
| --- | --- |
| How the code is built, conventions, architecture | `docs/dev/NOTES.md` |
| Anything about crimm-ml, the optional ML package (its code, model, service, state) | `docs/dev/AGENT_HANDOFF.md` in the crimm-ml repository |
| Machine- and account-specific setup (paths, cluster, accounts), and private downstream projects | the owner's `CLAUDE.local.md` (not in the repository) |
| What work is planned, its order, its status | `docs/dev/ROADMAP.md` |
| Standards the work should meet | `docs/dev/BEST_PRACTICES.md` |
| Conflicts and issues met preparing particular PDB entries (clashes, patch choices) | `docs/dev/PDB_PREP_LOG.md` |
| Who is doing what now, current state, what was tried, what failed, decisions | the `agent-log` issue (section 5) |
| Questions an agent could not settle | an issue labelled `question` |
| Protocol, standing instructions, environment facts, traps | this file |

If something belongs in the first three, put it there and give a one-line pointer in your
log comment.

## 1. Orientation

- crimm prepares biomolecular structures for MD simulation and ML training. Published on
  PyPI. GPLv3. `[told 2026-10-06]`
- Owner: Truman Xu (`Truman-Xu`). Collaborator: Stan, who wrote the solvation code.
  `[told 2026-10-06]` Stan is Stanislav Cherepanov. `[inferred from pyproject.toml authors]`
- Design goal: versatile and lightweight, with rich Jupyter visualization. `[told 2026-10-06]`
  Consequence: heavy dependencies must stay optional and lazily imported.
- Pipeline: fetch mmCIF, `OrganizedModel`, loop build, `TopologyGenerator`, `Solvator`,
  write PSF/CRD, optional pyCHARMM load. Package map is in `NOTES.md`.
- **Optional ML library: crimm-ml.** RFdiffusion loop building lives in a separate repo,
  github.com/Truman-Xu/crimm-ml (import `crimm_ml`), installed as
  `pip install crimm[ml]`. It runs the model on ONNX Runtime without PyTorch, takes PDB lines
  and a contig string, and returns coordinates. It **depends on crimm**; crimm must import it
  only inside functions. The crimm side (`crimm/Modeller/RFLoopBuilder.py`, the `ml` extra)
  is ROADMAP "Parallel track: crimm-ml integration" and waits for "1.5 Make rdkit and
  nglview optional at import time". `[told 2026-10-06]` **Update `[told 2026-10-08]`:** the
  owner intends the RFdiffusion model to become part of `LoopBuilder`, so the separate
  `RFLoopBuilder.py` name here and in the contract below is the original placeholder plan;
  the layout is open (issue #101).
- **To pick up crimm-ml context:** in the crimm-ml repository, read `AGENTS.md`, then
  sections 0 to 4 of its `docs/dev/AGENT_HANDOFF.md` (same protocol as this file), its open
  `agent-log` issue and its `docs/dev/ROADMAP.md`. The owner's local checkout path is in `CLAUDE.local.md`. Do not copy
  its facts here; link to them.
- **Private downstream projects.** Projects that depend on crimm and are not public are
  described in the owner's `CLAUDE.local.md` (not in the repository): what they import from
  crimm and the reminders the owner asked for. Agents on the owner's machine load it
  automatically; treat its instructions like section 2.

## 2. Standing instructions from the owner

Follow these unless the owner says otherwise in your own session.

| Date | Instruction | Tag |
| --- | --- | --- |
| 2026-10-06 | Dev documents are tracked Markdown under `docs/dev/`. | `[told]` |
| 2026-11-07 | Agent activity (state, claims, session log) is not kept in a tracked file, because each update would need a pull request. It goes in the open `agent-log` issue; open questions are issues labelled `question` (section 5). | `[told]` |
| 2026-10-06 | Public API may break, but old import paths must keep working with `DeprecationWarning`. | `[told]` |
| 2026-10-06 | Resolve TODO comments before the refactor and restructure phases. SUPERSEDED 2026-10-28 by the row below. | `[told]` |
| 2026-10-28 | Pull requests are merged by the owner once CI is green and reviewed afterwards in batches (label `review-after-merge`), except changes to Stan's files, which wait for his approval. Small pull requests, one concern each. Rule: CONTRIBUTING, "Review and merge". | `[told]` |
| 2026-10-28 | Not every TODO comment has to be resolved before the refactor and restructure phases. Some are refactoring work themselves. Which phase a TODO belongs to is decided when its resolution starts, not in advance. Every TODO still carries its issue number. | `[told]` |
| 2026-10-30 | The project, not only Stan, prefers small pull requests (CONTRIBUTING, "Issues and pull requests"). Merging most of them first and reviewing later stays. The owner gives Stan a deadline with each review request. | `[told]` |
| 2026-11-02 | Design ideas for later improvements go in ROADMAP Phase 7 (planned features and robustness), after the minimizer, as problems surface. First: a simulation system object with a readiness check before writing files or handing the system to another program. The owner adds notes. | `[told]` |
| 2026-10-06 | `import-fix.md` (repo root) is folded into ROADMAP 1.5. Delete the file as the last step of that fix, not before. | `[told]` |
| 2026-10-06 | Commit only when asked; never push unless asked. The owner asked for the 2026-10-06 commits explicitly; that does not carry over to later sessions. | `[told]` for those commits, `[inferred]` as a standing rule |
| 2026-10-08 | CHARMM (and so pyCHARMM) is licensed and not accessible to developers without a licence. Keep CHARMM and pyCHARMM tooling and tests in the roadmap, gated by the `pycharmm` marker; document the licence limit. Never put CHARMM source, binaries or libraries in the repository or in issues. | `[told]` |
| 2026-10-08 | The owner compiles CHARMM locally, as part of the developer environment setup, in the next session (likely Monday 2026-10-12). Do not start the build or install anything for it before then. SUPERSEDED 2026-10-14: built (section 3). | `[told]` |
| 2026-10-14 | Stan agreed that his modules (`Solvator.py`, `CrystalSDF.py`, `CoordManipulator.py`) may be linted and formatted, with git blame kept to him: lint and format commits go in `.git-blame-ignore-revs`, and such a PR is merged with a merge commit. Other changes to his code still go to him for review. | `[told]` |
| 2026-10-14 | The default C-terminal patch `CT3` in `generate_model` is intended. Structure-specific problems it causes are logged per entry in `docs/dev/PDB_PREP_LOG.md`. | `[told]` |
| 2026-10-14 | Machine- and account-specific setup (paths, cluster, accounts) and private downstream projects go in the owner's `CLAUDE.local.md`, not in `docs/dev/`. Published docs describe the setup generically. | `[told]` |

Working agreements between agents (not owner instructions; change them by logging why):

- Changes to `crimm/Modeller/Solvator.py`, `crimm/Modeller/CrystalSDF.py` and the PCA
  methods in `CoordManipulator.py` are Stan's area. Propose; flag for his review.
- A bug you find but do not fix gets a test marked `xfail(strict=True)` with the reason, and
  a GitHub issue (ROADMAP indexes the issues). When you fix it, remove the marker in the same
  change.
- One concern per change. Do not mix a fix with a rename.
- **crimm-ml contract.** No module-level `import crimm_ml` anywhere in crimm. crimm-ml and
  the planned `RFLoopBuilder` use: `crimm.IO.PDBString.get_pdb_str`,
  `crimm.Modeller.ResidueTopologySet` (and `ResidueDefinition.create_residue`),
  `TopologyGenerator.apply_topo_def_on_residue`, `ResidueFixer.load_residue` /
  `build_missing_atoms`, `PolymerChain.gaps`, `.missing_res`, `.sort_residues`. If you rename
  or change one, keep the deprecated alias (as for any public API) and post a one-line
  pointer as a comment on crimm-ml's `agent-log` issue.
- **Private downstream contracts.** `CLAUDE.local.md` lists the crimm names that private
  projects import. The same rule applies: keep a deprecated alias and post a one-line
  pointer in that project's agent log (its `agent-log` issue). `[inferred: same rule as for
  crimm-ml]`
- Do not delete the branch `RDKConverter-Het` or rewrite `8075b75` without telling the owner;
  a downstream project depends on it (`CLAUDE.local.md`). `[inferred]` SUPERSEDED 2026-10-30:
  deleted at the owner's request, reimplemented on `fix/59-rdkit-plain-residue`.

## 3. Development environment

How the maintainers' `crimm-dev` environment is built. Paths, cluster details and accounts of
the owner's machine are in the owner's `CLAUDE.local.md`, not here.

### `crimm-dev` from `environment-dev.yml`

```bash
conda env create -f environment-dev.yml      # Python 3.12; crimm's core and optional deps,
                                             # crimm-ml's runtime deps, dev tools
conda activate crimm-dev
pip install -e ".[all,test]" --no-deps       # from the crimm checkout
python -m ipykernel install --user --name crimm-dev --display-name crimm-dev
```

Check it with `python -c "import crimm"`, `ruff check crimm tests` and
`pytest -p no:cacheprovider -q` (147 passed, 16 xfailed, 21 deselected on 2026-11-07, master
f9b3e99).

| Fact | Tag |
| --- | --- |
| `environment-dev.yml` must name `python-build` (the conda package), not `build`; the first `conda env create` failed on that. | `[ran 2026-10-07]` |
| The env provides `ruff`, `pre-commit`, `nox`, `pytest` and `gh`; lint is `ruff check crimm tests`. | `[ran 2026-10-12]` |
| The first `import crimm` on a cold network filesystem can take minutes (47 to 145 s seen). It is slow, not hung; give the first command a timeout of at least 300 s. | `[ran 2026-10-06]` |
| CGenFF is not installable from conda. The `cgenff` tests read the executable from `CRIMM_CGENFF_PATH`, and also need the network (G11). | `[ran 2026-10-06]` |
| crimm 2026.2.2 from PyPI, in a clean Python 3.11 venv: `pip install crimm` pulls nglview and ipywidgets, and `import crimm` fails with `ModuleNotFoundError: rdkit` (ROADMAP "1.5 Make rdkit and nglview optional at import time"). | `[ran 2026-10-06 by claude-opus/claude-code/crimm-ml]` |

### CHARMM and pyCHARMM in `crimm-dev` (2026-10-14)

CHARMM is licensed; this is for licence holders. What was done (the build flags and caveats are
in `NOTES.md`, "Building CHARMM for the `pycharmm` tests"):

1. Built CHARMM `master` (52.1, `c16e5816e`) as a CPU-only shared library with GCC 13.4, from
   a separate `git worktree` of the CHARMM checkout so its working branch stayed untouched.
   The build ran as a batch job on a compute node (about 2 minutes on 16 cores).
2. `pip install --no-deps -e <charmm>/tool/pycharmm` into `crimm-dev`.
3. `conda env config vars set -n crimm-dev CHARMM_LIB_DIR=<install>/lib`.
4. Added `"env": {"CHARMM_LIB_DIR": "<install>/lib"}` to the `crimm-dev` Jupyter kernel's
   `kernel.json`, because conda env vars apply only after `conda activate`.
5. Checked: `import pycharmm` in the activated env and in the kernel; `pytest -m pycharmm`
   gives 8 passed, 2 xfailed (#112).

| Fact | Tag |
| --- | --- |
| CHARMM's `configure` is a Python script that needs Python >= 3.7; run it with `crimm-dev`'s Python where the system's is older. | `[ran 2026-10-14]` |
| On clusters with environment modules, load modules before `set -e` in a batch script: a failing command inside the module init script can abort the job. | `[ran 2026-10-14]` |
| The editable pyCHARMM install and `CHARMM_LIB_DIR` point into the CHARMM worktree; removing it breaks both. | `[inferred: editable install]` |

## 4. Gotchas

Things that cost a previous agent time or would produce wrong results silently.

| ID | Trap | Tag |
| --- | --- | --- |
| G1 | `Solvator.solvate()` with defaults returns a **mirror image** of the solute (ROADMAP 1.1). Do not trust coordinates from it for cube, octa or rhdo boxes until fixed. | `[ran 2026-10-06 on 1UBQ]` |
| G2 | Atoms are shared between a model and its parent structure. Moving one moves the other. `solvate`, `orient_coords*` and `TopologyGenerator.generate*` all modify their argument in place. Build a fresh model per test. | `[read]`, `[ran]` |
| G3 | `OrganizedModel` properties are `protein`, `DNA`, `RNA` (upper case for nucleic acids), `ligand`, `ion`, `solvent`, `co_solvent`. `model.dna` raises `AttributeError`. | `[ran 2026-10-06]` |
| G4 | `OrganizedModel(...)` is offline by default. It only calls RCSB when `identify_ligands=True` or `fetch_web_data=True`. | `[read]` |
| G5 | `chain.total_charge` returns `None`, not a number, if any atom lacks `topo_definition`. | `[read]` |
| G6 | PSF and CRD title blocks contain a timestamp and the user name. Byte comparison of output must skip lines starting with `*`. | `[read]` |
| G7 | `id(obj)` comparisons across a call that frees objects give false matches. Hold references. This caused an order-dependent test failure. | `[ran 2026-10-06]` |
| G8 | `ndarray.ptp` does not exist on NumPy 2. Use `np.ptp(a, axis=...)`. | `[ran 2026-10-06]` |
| G9 | `fetch_rcsb(local_entry=...)` expects `<root>/<id[1:3]>/<id>.cif`, lower case, not a flat directory. | `[read]`, `[ran]` |
| G10 | `get_pdb_str` on a model with `TIP3` waters needs `convert_water=True` to be readable by `PDBParser`. | `[ran 2026-10-06]` |
| G11 | The CGenFF ligand path queries RCSB for the ligand's chemistry, so `cgenff` tests also need the network. | `[inferred: RDKitConverter posts to RCSB; test passed only with network available]` |
| G12 | Other sessions edit `docs/dev/*.md` concurrently. Re-read a file immediately before editing it, and make targeted edits, never whole-file rewrites. | `[ran 2026-10-06: ROADMAP.md and NOTES.md changed under an active session]` |
| G13 | ROADMAP section numbers are not stable identifiers. 1.3 to 1.5 were renumbered on 2026-10-06. Cite the section title with the number. | `[ran 2026-10-06]` |
| G15 | Two PSF or CRD files of one structure are not byte-identical: angle and dihedral order varies (#120), and water hydrogens use the unseeded global NumPy generator. Compare through `tests/test_golden.py::normalized_psf`, and seed before building the model, as `seeded_model` does. `shared_prepared` models were built unseeded. | `[ran 2026-10-30]` |
| G14 | The test suite lives only on `cleanup/phase0-tests-and-dev-docs` until it is merged. A branch cut from `master` has no `tests/test_*.py`, so `pytest` there collects nothing and reports no failures. SUPERSEDED 2026-11-02: merged with PR #108. | `[ran 2026-10-06]` |
| G17 | ruff `B007` (unused loop variable) only looks inside the loop body. A variable used after the loop, for example one that keeps the value at a `break`, is flagged too; renaming it raises `NameError`. Check uses after the loop before fixing (`TopoDefinitions._create_init_atom_coords`). | `[ran 2026-11-02]` |
| G16 | Segment IDs are assigned to the model by writing a PSF (`PSFWriter._assign_segids`, also via `get_psf_str`). A CRD written before that has blank segment IDs, which `CRDParser` cannot read (#124). Write the PSF first. | `[ran 2026-11-02]` |
| G18 | The remote has a branch named `docs`, so no branch named `docs/<anything>` can be pushed: GitHub rejects it with "directory file conflict". Use another prefix, for example `chore/`. | `[ran 2026-11-07]` |

## 5. State, claims, open questions and log

These change every session, so they are not kept in this file, which needs a pull request
for every change. Since 2026-11-07 `[told]`:

- **State, claims and the log** are in the open issue labelled `agent-log`, pinned on the
  issue list (Phase 1: #135). Its description holds Claims and State, rewritten in place;
  each comment is one session's log entry. Find it with
  `gh issue list --label agent-log --state open`.
- **Open questions** are issues labelled `question`.
- **The earlier log**, 2026-10-06 to 2026-11-02, with State, Claims and the open questions as
  they stood, is section 8 of this file at commit 36c9052:
  `git show 36c9052:docs/dev/AGENT_HANDOFF.md`.
