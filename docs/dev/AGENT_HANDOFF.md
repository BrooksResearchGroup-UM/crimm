# AGENT_HANDOFF

Shared memory for AI agents working on crimm. Any model, any tool. Written for agents first;
humans read it second.

```yaml
schema: agent-handoff/1
repo: github.com/BrooksResearchGroup-UM/crimm
default_branch: master
last_updated: 2026-10-06
```

## 0. Protocol

Read this section fully. Skim the rest for what your task touches.

### On arrival

1. Read sections 1 to 4. They are short and change rarely.
2. Read section 5 (STATE) and the newest three entries of section 8 (LOG).
3. Run `git status --short` and `git log --oneline -5`. If they disagree with STATE, trust
   git and correct STATE.
4. Check section 6 (CLAIMS) before editing a file. If another agent holds it, do not edit it.

### On departure

1. Append one entry to LOG (newest first, template in section 8).
2. Update STATE so it is true now. Rewrite it in place; do not append.
3. Release your CLAIMS.
4. Move anything you learned that will still be true next month into FACTS or GOTCHAS.

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
- **Do not delete another agent's entry.** If it is wrong, add `SUPERSEDED YYYY-MM-DD by
  <agent>: <why>` under it.
- **Identify yourself** as `<model>/<tool>/<short task name>`, for example
  `claude-opus/claude-code/cleanup`. You have no persistent identity; the task name is what
  lets others tell sessions apart.
- Keep entries terse. No preamble, no summaries of summaries.

### What goes where

| Kind of information | File |
| --- | --- |
| How the code is built, conventions, architecture | `docs/dev/NOTES.md` |
| Anything about crimm-ml, the optional ML package (its code, model, service, state) | `/home/ziqiaoxu/crimm-ml/docs/dev/AGENT_HANDOFF.md` |
| What work is planned, its order, its status | `docs/dev/ROADMAP.md` |
| Standards the work should meet | `docs/dev/BEST_PRACTICES.md` |
| Who is doing what now, what was tried, what failed, traps, decisions | this file |

If something belongs in the first three, put it there and log a one-line pointer here.

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
  `/home/ziqiaoxu/crimm-ml` (github.com/Truman-Xu/crimm-ml, import `crimm_ml`), installed as
  `pip install crimm[ml]`. It runs the model on ONNX Runtime without PyTorch, takes PDB lines
  and a contig string, and returns coordinates. It **depends on crimm**; crimm must import it
  only inside functions. The crimm side (`crimm/Modeller/RFLoopBuilder.py`, the `ml` extra)
  is ROADMAP "Parallel track: crimm-ml integration" and waits for "1.5 Make rdkit and
  nglview optional at import time". `[told 2026-10-06]`
- **To pick up crimm-ml context:** read `/home/ziqiaoxu/crimm-ml/AGENTS.md`, then sections 0
  to 5 of its `docs/dev/AGENT_HANDOFF.md` (same protocol as this file) and its
  `docs/dev/ROADMAP.md`. Do not copy its facts here; link to them.

## 2. Standing instructions from the owner

Follow these unless the owner says otherwise in your own session.

| Date | Instruction | Tag |
| --- | --- | --- |
| 2026-10-06 | Dev documents are tracked Markdown under `docs/dev/`. | `[told]` |
| 2026-10-06 | Public API may break, but old import paths must keep working with `DeprecationWarning`. | `[told]` |
| 2026-10-06 | Resolve TODO comments before the refactor and restructure phases. | `[told]` |
| 2026-10-06 | `import-fix.md` (repo root) is folded into ROADMAP 1.5. Delete the file as the last step of that fix, not before. | `[told]` |
| 2026-10-06 | Commit only when asked; never push unless asked. The owner asked for the 2026-10-06 commits explicitly; that does not carry over to later sessions. | `[told]` for those commits, `[inferred]` as a standing rule |

Working agreements between agents (not owner instructions; change them by logging why):

- Changes to `crimm/Modeller/Solvator.py`, `crimm/Modeller/CrystalSDF.py` and the PCA
  methods in `CoordManipulator.py` are Stan's area. Propose; flag for his review.
- A bug you find but do not fix gets a test marked `xfail(strict=True)` with the reason, and
  a row in ROADMAP Phase 1. When you fix it, remove the marker in the same change.
- One concern per change. Do not mix a fix with a rename.
- **crimm-ml contract.** No module-level `import crimm_ml` anywhere in crimm. crimm-ml and
  the planned `RFLoopBuilder` use: `crimm.IO.PDBString.get_pdb_str`,
  `crimm.Modeller.ResidueTopologySet` (and `ResidueDefinition.create_residue`),
  `TopologyGenerator.apply_topo_def_on_residue`, `ResidueFixer.load_residue` /
  `build_missing_atoms`, `PolymerChain.gaps`, `.missing_res`, `.sort_residues`. If you rename
  or change one, keep the deprecated alias (as for any public API) and log a one-line
  pointer in crimm-ml's AGENT_HANDOFF.

## 3. Environment facts

Machine: the owner's cluster login node (Rocky Linux 8, SLURM). Paths are specific to it.

| Fact | Tag |
| --- | --- |
| There is no `python` on `PATH`. Use `/home/ziqiaoxu/.conda/envs/torch210/bin/python` (Python 3.13, NumPy 2.2.6, Biopython 1.86, pytest 9.0.3). | `[ran 2026-10-06]` |
| In that env: rdkit, nglview, propka present. Missing: ruff, pytest-cov, pytest-xdist, openmm, pycharmm. pyflakes works as `python -m pyflakes crimm`. | `[ran 2026-10-06]` |
| No conda env under `~/.conda/envs` contains pycharmm. | `[ran 2026-10-06, searched site-packages names only]` |
| `import crimm` took 47 to 145 s on a cold filesystem cache and a few seconds when warm. Give the first command of a session a timeout of at least 300 s. It is slow, not hung. | `[ran 2026-10-06]` |
| The login node reaches `files.rcsb.org`. SLURM compute nodes do not resolve DNS. | `[ran 2026-10-06]` login; `[read tests/slurm/benchmark.log]` compute |
| CGenFF executable: `/export/apps/RockyOS8/cgenff/src/2026/silcsbio.2026.1-alpha/cgenff/cgenff`. Tests read it from `CRIMM_CGENFF_PATH`. | `[ran 2026-10-06]` |
| crimm 2026.2.2 from PyPI, in a clean Python 3.11 venv: `pip install crimm` pulls nglview and ipywidgets, and `import crimm` fails with `ModuleNotFoundError: rdkit` (ROADMAP "1.5 Make rdkit and nglview optional at import time"). | `[ran 2026-10-06 by claude-opus/claude-code/crimm-ml]` |

Commands that work `[ran 2026-10-06]`:

```bash
PY=/home/ziqiaoxu/.conda/envs/torch210/bin/python
$PY -B -m pytest -p no:cacheprovider -q                 # offline suite, about 45 s warm
$PY -B -m pytest -p no:cacheprovider -q -m network      # needs internet
CRIMM_CGENFF_PATH=<path above> $PY -B -m pytest -q -m "cgenff"
$PY -m pyflakes crimm
```

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
| G14 | The test suite lives only on `cleanup/phase0-tests-and-dev-docs` until it is merged. A branch cut from `master` has no `tests/test_*.py`, so `pytest` there collects nothing and reports no failures. | `[ran 2026-10-06]` |

## 5. State

Rewrite in place. True as of `2026-10-06`.

```yaml
branches:                          # all local, none pushed; all start at fbb165d (master)
  cleanup/phase0-tests-and-dev-docs:     # agent cleanup work; continue here
    - d10c62d                      # tests/, tests/data/, pyproject.toml pytest config
    - fa7338a                      # docs/dev/*.md
    - "later commits: handoff updates"
  feat/coordmanipulator-find-max-dim:    # the owner's solvation edits, kept separate
    - 44e9bf6                      # CoordManipulator.find_max_dim; Solvator TODO comment
  solvator-coor-orient-options:    # old working branch; still points at fa7338a.
                                   # Redundant now. Owner decides whether to delete it.
uncommitted_owner_work: none
uncommitted_agent_work: none
untracked_not_ours:
  - import-fix.md                  # work order, see standing instructions
  - tests/slurm/                   # owner's benchmark job files
  - .claude/
test_suite: {passed: 133, xfailed: 13, deselected: 3, date: 2026-10-06}
roadmap_done:
  - "Phase 0: offline pytest suite"
  - "Phase 0: markers"
roadmap_next:
  - "Phase 0: golden-file tests for PSF and CRD"
  - "Phase 0: ruff config"
  - "Phase 0: CI workflow"
  - "Phase 0: move tests/benchmark_pipeline.py to benchmarks/"
library_code_changed_by_agents: none
```

## 6. Claims

A claim says "I am editing this; stay out". Add a row when you start, remove it when you
stop. A claim older than 24 hours with no matching LOG entry is stale; take it and log that
you did.

| Path or area | Agent | Since | Purpose |
| --- | --- | --- | --- |
| (none) | | | |

## 7. Open questions

Things an agent could not settle. Answer one by replacing it with a dated FACT, GOTCHA or
ROADMAP row.

| ID | Question | Raised |
| --- | --- | --- |
| Q1 | Why does `PSFWriter` write 394 atoms and charge +4.5 per 1BNA strand when the model has 383 atoms and charge -12? Residue 1 matches; residues 2 to 12 each have one duplicated atom name. Not investigated further. | 2026-10-06 |
| Q2 | Is `ArcLoopBuilder` (`crimm/Modeller/LoopBuilder.py`) meant to be finished or removed? It references an undefined name `topo`. Owner decision. | 2026-10-06 |
| Q3 | Should a ligand without topology be dropped from both PSF and CRD, or kept in both? They disagree now. Owner decision. | 2026-10-06 |
| Q4 | Are `crimm/Utils/cuda_info.py` and `crimm/Data/probes/` still wanted in crimm? They look left over from a removed docking module. Owner decision. | 2026-10-06 |
| Q5 | Licence and provenance of the bundled CHARMM toppar files are undocumented. | 2026-10-06 |
| Q6 | No pyCHARMM is available to agents on this machine, so the `pycharmm` marker has no tests and the pyCHARMM adaptor is unverified by any agent. Where can it be run? | 2026-10-06 |
| Q7 | Which session owns the "Parallel track: crimm-ml integration" section of ROADMAP? It was added by another session on 2026-10-06; that session has not logged here. | 2026-10-06 **ANSWERED 2026-10-06 by claude-opus/claude-code/crimm-ml:** that session added it at the owner's request; see its LOG entry below. State for crimm-ml lives in crimm-ml's own AGENT_HANDOFF. |

## 8. Log

Newest first. Append only. Template:

```markdown
### YYYY-MM-DD <model>/<tool>/<task>

- asked: <what the owner asked for, one line>
- did: <what changed, with paths>
- verified: <what you ran and the result>
- not verified: <what you assumed or skipped>
- found: <bugs, surprises; point to ROADMAP rows or GOTCHAS>
- next: <the obvious next step, and anything blocking it>
```

### 2026-10-06 claude-opus/claude-code/cleanup (entry 5)

- asked: put the owner's uncommitted edits on their own branch, the cleanup work on
  another, and commit.
- did: created `feat/coordmanipulator-find-max-dim` at `fbb165d` and committed the owner's
  two files there (`44e9bf6`); created `cleanup/phase0-tests-and-dev-docs` at `fa7338a`
  and switched to it. Entry 4's branch description is superseded by State.
- verified: `git status` clean on both apart from the untracked files listed in State;
  `git log` on each `[ran]`.
- not verified: the test suite was not re-run on the feature branch. It does not contain
  `tests/`, so there is nothing to run there until the branches meet.
- found: G14.
- next: unchanged from entry 2.

### 2026-10-06 claude-opus/claude-code/cleanup (entry 4)

- asked: wrap up; update dev docs; track and commit the dev docs and the Phase 0 work.
- did: two local commits on `solvator-coor-orient-options`: `d10c62d` (test suite,
  fixtures, pytest config) and the one after it (`docs/dev/`). Updated NOTES (repository
  state now points here), ROADMAP (suggested pull requests), and this file's State and
  standing instructions.
- verified: offline suite re-run immediately before committing, 133 passed, 13 xfailed
  `[ran]`. Staged by explicit path; the owner's two modified files, `import-fix.md`,
  `tests/slurm/` and `.claude/` were left out `[ran: git status]`.
- not verified: nothing was pushed, so no CI or remote state was exercised.
- found: the commits sit on a feature branch named for solvation work, because that is
  what was checked out and switching would have carried the owner's uncommitted edits.
  They are independent of that work and can be cherry-picked onto their own branch.
- next: unchanged from entry 2 (golden-file tests; see Q1 first).

### 2026-10-06 claude-opus/claude-code/crimm-ml

- asked: split RFdiffusion loop building into an optional package (`crimm[ml]`); make it
  known to agents working on crimm and say how to pick up its context.
- did: in crimm, only docs: wrote `import-fix.md` (repo root; now folded into ROADMAP "1.5
  Make rdkit and nglview optional at import time"); added ROADMAP "Parallel track: crimm-ml
  integration" and `ml` to the Phase 4 extras row; in this file, added the crimm-ml rows to
  "What goes where", Orientation, the working agreements and Environment facts, and answered
  Q7. No crimm source, tests or `pyproject.toml` touched.
- verified: `pip install crimm` from PyPI then `import crimm` in a clean venv (fails on
  rdkit) `[ran]`. Every line number in `import-fix.md` re-read against `fbb165d` `[read]`;
  the PEP 562 trap in step 4 checked with a toy module `[ran]`.
- not verified: the planned `RFLoopBuilder` API against crimm after the cleanup refactors.
- found: nothing new in crimm beyond 1.5.
- next: once 1.5 lands, the crimm-ml track (RFLoopBuilder, `ml` extra). Whoever does it
  should claim `crimm/Modeller/RFLoopBuilder.py` here and log in both handoffs.

### 2026-10-06 claude-opus/claude-code/cleanup (entry 3)

- asked: start a shared document for AI agents under `docs/dev/`.
- did: created this file.
- verified: STATE against `git status`. `[ran]`
- not verified: whether other tools will discover this file. Nothing at the repo root
  points to it; agents that auto-read `AGENTS.md` or `CLAUDE.md` will not find it unless
  a pointer is added there.
- next: owner to decide on a root pointer file.

### 2026-10-06 claude-opus/claude-code/cleanup (entry 2)

- asked: do the first two Phase 0 roadmap items.
- did: added `tests/data/` (1UBQ, 1CRN, 2IGD, 3PTB, 1BNA from RCSB, unmodified),
  `tests/conftest.py`, nine `tests/test_*.py` modules; added pytest `addopts`, markers and a
  `test` extra to `pyproject.toml`; updated ROADMAP (two items done, new section "1.3 Bugs
  found by the new test suite", later sections renumbered) and NOTES (tests section).
- verified: default run 133 passed, 13 xfailed; `-m "network or cgenff"` 3 passed. `[ran]`
- not verified: Python versions other than 3.13; NumPy 1.x; any pyCHARMM path; test
  stability across many runs (ion placement is unseeded, assertions were written not to
  depend on it; the full suite was run twice, passing once after fixes).
- found: mirror-image solvation confirmed on 1UBQ (G1); five new bugs (ROADMAP 1.3); Q1, Q3.
- next: golden-file PSF/CRD tests. Strip title lines (G6). Do not make golden files from
  the DNA fixture until Q1 is resolved, or the wrong output becomes the reference.

### 2026-10-06 claude-opus/claude-code/cleanup (entry 1)

- asked: remember project facts; write dev notes, a roadmap and a good-practices guide.
- did: created `docs/dev/NOTES.md`, `ROADMAP.md`, `BEST_PRACTICES.md`.
- verified: `python -m pyflakes crimm`; counted 27 TODO comments with grep; ran each
  `CoordManipulator` orientation method on 100 synthetic point sets (octa flipped
  handedness in 51, stored matrix disagreed with applied transform for all PCA methods in
  100 of 100). `[ran]`
- not verified: most Phase 1 "Other bugs" rows are `[read]` only, except the NumPy `ptp`
  failure which was `[ran]`. Memory estimate for the N by N distance matrix is arithmetic,
  not measured.
- found: ROADMAP Phase 1.
- next: Phase 0.
