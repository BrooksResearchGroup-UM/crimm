# AGENT_HANDOFF

Shared memory for AI agents working on crimm. Any model, any tool. Written for agents first;
humans read it second.

```yaml
schema: agent-handoff/1
repo: github.com/BrooksResearchGroup-UM/crimm
default_branch: master
last_updated: 2026-10-08
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
| Anything about msld-prep, the MSLD ligand-preparation package that depends on crimm (its code, tests, state) | `/home/ziqiaoxu/msld-prep/docs/dev/AGENT_HANDOFF.md` |
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
  nglview optional at import time". `[told 2026-10-06]` **Update `[told 2026-10-08]`:** the
  owner intends the RFdiffusion model to become part of `LoopBuilder`, so the separate
  `RFLoopBuilder.py` name here and in the contract below is the original placeholder plan;
  the layout is open (issue #101).
- **To pick up crimm-ml context:** read `/home/ziqiaoxu/crimm-ml/AGENTS.md`, then sections 0
  to 5 of its `docs/dev/AGENT_HANDOFF.md` (same protocol as this file) and its
  `docs/dev/ROADMAP.md`. Do not copy its facts here; link to them.
- **Downstream project: msld-prep.** Prepares ligand series for multisite lambda dynamics
  (MSLD) in CHARMM. Separate repo at `/home/ziqiaoxu/msld-prep` (the old path
  `/home/ziqiaoxu/msld-prep-tools` is a symlink to it, published to cluster users; GitHub
  `Truman-Xu/msld-prep`, import `msldprep`, GPLv3, not pushed yet). It **depends on crimm**: it
  imports `crimm.IO.RTFParser.RTFParser` and, from `crimm.IO.PRMParser`, `angle_par`,
  `bond_par`, `categorize_lines`, `cmap_par`, `dihe_par`, `impr_par`, `nbfix_par`,
  `nonbond14_par`, `nonbond_par`, `parse_line_dict`, `ub_par`. `[read msld-prep, 2026-10-08]`
  Its working env is `crimm-dev`, so the crimm branch checked out here is what it sees. Its
  dependency is the latest PyPI release of crimm until the overhaul is done, then a pinned
  version. `[told 2026-10-08]` **It needs the parked hotfix `RDKConverter-Het`; see section 2.**
- **To pick up msld-prep context:** read `/home/ziqiaoxu/msld-prep/AGENTS.md`, then sections 0
  to 5 of its `docs/dev/AGENT_HANDOFF.md` (same protocol as this file). Do not copy its facts
  here; link to them. Do not edit msld-prep from a crimm session unless the owner says so.

## 2. Standing instructions from the owner

Follow these unless the owner says otherwise in your own session.

| Date | Instruction | Tag |
| --- | --- | --- |
| 2026-10-06 | Dev documents are tracked Markdown under `docs/dev/`. | `[told]` |
| 2026-10-06 | Public API may break, but old import paths must keep working with `DeprecationWarning`. | `[told]` |
| 2026-10-06 | Resolve TODO comments before the refactor and restructure phases. | `[told]` |
| 2026-10-06 | `import-fix.md` (repo root) is folded into ROADMAP 1.5. Delete the file as the last step of that fix, not before. | `[told]` |
| 2026-10-06 | Commit only when asked; never push unless asked. The owner asked for the 2026-10-06 commits explicitly; that does not carry over to later sessions. | `[told]` for those commits, `[inferred]` as a standing rule |

| 2026-10-08 | CHARMM (and so pyCHARMM) is licensed and not accessible to developers without a licence. Keep CHARMM and pyCHARMM tooling and tests in the roadmap, gated by the `pycharmm` marker; document the licence limit. Never put CHARMM source, binaries or libraries in the repository or in issues. | `[told]` |
| 2026-10-08 | The owner compiles CHARMM locally, as part of the developer environment setup, in the next session (likely Monday 2026-10-12). Do not start the build or install anything for it before then. | `[told]` |
| 2026-10-08 | **msld-prep needs the hotfix on branch `RDKConverter-Het` (`8075b75`, `RDKitConverter.py`).** The owner said the fix will not be published anytime soon. **Remind the owner that msld-prep needs it** when: the fix is merged into any branch, pushed, or released; someone opens an issue or pull request about `heterogen_to_rdkit` or `RDKitHetConverter`; or you pick the hotfix up yourself (its test should cover a plain `Residue` read from a CRD file, msld-prep's case). Say it in your reply, not only in a file. After the owner has been reminded, they update msld-prep's handoff (its Q12). | `[told]` |

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
- **msld-prep contract.** Keep the names listed in Orientation (`crimm.IO.RTFParser.RTFParser`
  and the `crimm.IO.PRMParser` functions). If you rename or change one, keep the deprecated
  alias and log a one-line pointer in msld-prep's AGENT_HANDOFF. `[inferred: same rule as
  for crimm-ml]`
- Do not delete the branch `RDKConverter-Het` or rewrite `8075b75` without telling the owner;
  msld-prep depends on it. `[inferred]`

## 3. Environment facts

Machine: the owner's cluster login node (Rocky Linux 8, SLURM). Paths are specific to it.

| Fact | Tag |
| --- | --- |
| There is no `python` on `PATH`. Use `/home/ziqiaoxu/.conda/envs/torch210/bin/python` (Python 3.13, NumPy 2.2.6, Biopython 1.86, pytest 9.0.3). | `[ran 2026-10-06]` |
| In that env: rdkit, nglview, propka present. Missing: ruff, pytest-cov, pytest-xdist, openmm, pycharmm. pyflakes works as `python -m pyflakes crimm`. | `[ran 2026-10-06]` |
| No conda env under `~/.conda/envs` contains pycharmm. | `[ran 2026-10-06, searched site-packages names only]` |
| `import crimm` took 47 to 145 s on a cold filesystem cache and a few seconds when warm. Give the first command of a session a timeout of at least 300 s. It is slow, not hung. | `[ran 2026-10-06]` |
| The login node reaches `files.rcsb.org`. SLURM compute nodes do not resolve DNS. | `[ran 2026-10-06]` login; `[read tests/slurm/benchmark.log]` compute |
| CHARMM material already on the cluster, found by listing directories only (nothing read or run): shared builds and docs under `/export/apps/RockyOS8/charmm/` (for example `c51a1`, `c51a2`, `August_26_2026`; `c50a1/lib/libcharmm.so` exists); environment modules `charmm/charmm/c49a1` to `c52a1` (default `c52a1`), whose modulefile activates a conda build environment under `/export/apps/RockyOS8/charmm/envs/`; a CHARMM source checkout (git, `configure`, `CMakeLists.txt`) at `~/charmm`, last changed 2026-03-15; Jupyter kernels `charmm` and `pycharmm` pointing at other people's conda envs. Whether any of these is usable from `crimm-dev` is open (ROADMAP "CHARMM and pyCHARMM"). | `[ran 2026-10-08, ls only]` |
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

Rewrite in place. True as of `2026-10-08`.

```yaml
branches:                          # all start at fbb165d; the cleanup branch is pushed (2026-10-08)
  cleanup/phase0-tests-and-dev-docs:     # agent cleanup work; continue here
    - d10c62d                      # tests/, tests/data/, pyproject.toml pytest config
    - fa7338a                      # docs/dev/*.md
    - "later commits: handoff updates; repo-setup files (2026-10-07)"
    - d90f94b                      # merge of origin/master (cf51387), 2026-10-08
    - "then three docs commits of 2026-10-08: CHARMM licence and plan; roadmap and issue
       index; handoff"
  feat/coordmanipulator-find-max-dim:    # the owner's solvation edits, kept separate
    - 44e9bf6                      # CoordManipulator.find_max_dim; Solvator TODO comment
  RDKConverter-Het:                # owner's hotfix, local only, not pushed. Cut from 72eef9f.
    - 8075b75                      # RDKitHetConverter accepts non-Heterogen; see LOG 2026-10-08.
                                   # Parked: needs a test and review before it joins anything.
                                   # NEEDED BY msld-prep: remind the owner when it is published
                                   # (section 2).
  solvator-coor-orient-options:    # old working branch; still points at fa7338a.
                                   # Redundant now. Owner decides whether to delete it.
env_rebuild: done                  # conda env `crimm-dev` (py3.12.15) built from
                                   # environment-dev.yml (built as `crimm-dev-new`, cloned to
                                   # `crimm-dev`, the clone verified, `crimm-dev-new` removed).
                                   # Old py3.8 env deleted by agent at owner's request 2026-10-08.
                                   # Jupyter kernel `crimm-dev` re-registered to the new env.
                                   # Offline suite 133 passed, 13 xfailed, 3 deselected, ran
                                   # 2026-10-08. The editable `crimm` install is in the user site
                                   # (`~/.local/lib/python3.12/site-packages`), not in the env.
base_is_behind_remote: false       # `origin/master` (cf51387) merged into the cleanup branch as
                                   # d90f94b on 2026-10-08, not pushed. Offline suite unchanged
                                   # after the merge (133 passed, 13 xfailed), so the PSF/CRD
                                   # xfails still hold on cf51387.
uncommitted_owner_work: none
uncommitted_agent_work: none      # all 2026-10-08 docs work is committed and pushed
untracked_not_ours:
  - import-fix.md                  # work order, see standing instructions
  # tests/slurm/ (owner's benchmark job files) and .claude/ are ignored through
  # .git/info/exclude (local, not in the repository) as of 2026-10-08.
local_branches_deleted: [cuda_util, list, mmcif-bug-fix, newGridGen, py3dmol]   # 2026-10-08, all merged into master
github:                            # 2026-10-08
  issues: "#39 to #107 (no #105), milestones 1 to 6 = Phases 0 to 5"
  labels_added: [correctness, refactor, agent-task, topology, solvation, io, visualization, loops, ml, infra]
  branch_protection: not_set
owner_reminders:                   # tell the owner at the start of the next session
  - "msld-prep needs the RDKConverter-Het hotfix (section 2); issue #59 tracks it"
test_suite: {passed: 133, xfailed: 13, deselected: 3, date: 2026-10-08}
roadmap_done:
  - "Phase 0: offline pytest suite"
  - "Phase 0: markers"
roadmap_next:
  - "Next session (likely Monday 2026-10-12): build CHARMM and pyCHARMM locally (owner), then the pycharmm tests"
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
| Q6 | No pyCHARMM is available to agents on this machine, so the `pycharmm` marker has no tests and the pyCHARMM adaptor is unverified by any agent. Where can it be run? | 2026-10-06 **PLAN 2026-10-08 `[told]`:** the owner builds CHARMM locally, likely Monday 2026-10-12; see ROADMAP "CHARMM and pyCHARMM". Stays open until `import pycharmm` works in `crimm-dev`. |
| Q7 | Which session owns the "Parallel track: crimm-ml integration" section of ROADMAP? It was added by another session on 2026-10-06; that session has not logged here. | 2026-10-06 **ANSWERED 2026-10-06 by claude-opus/claude-code/crimm-ml:** that session added it at the owner's request; see its LOG entry below. State for crimm-ml lives in crimm-ml's own AGENT_HANDOFF. |

## 8. Log

Newest first. Append only. Template:

```markdown
### YYYY-MM-DD <model>/<tool>/<task>

- asked:
- did:
- verified:
- not verified:
- found:
- next:
```

### 2026-10-08 claude-sonnet/claude-code/wrapup

- asked: end-of-day wrap-up; review the msld-prep session's edits to the dev docs; then
  commit, push, delete merged branches, ignore `tests/slurm/`, and update all md docs and logs.
- did: committed the day's docs in three commits (CHARMM licence and plan; roadmap issue
  index and the LoopBuilder decision; this handoff) and pushed
  `cleanup/phase0-tests-and-dev-docs`; deleted the local branches `cuda_util`, `list`,
  `mmcif-bug-fix`, `newGridGen`, `py3dmol` (all merged into master; their SHAs were noted
  before deleting, so `git branch <name> <sha>` restores one); added `tests/slurm/` to
  `.git/info/exclude`; corrected State (branches, uncommitted work, untracked files, test
  date, GitHub block, owner reminders); updated ROADMAP (header, new row for #59) and NOTES
  (snapshot note); added a one-line pointer to crimm-ml's handoff, as the contract requires.
  `[ran]`
- reviewed (msld-prep-intro entry): the imports it lists exist in `crimm/IO/RTFParser.py` and
  `crimm/IO/PRMParser.py` (the `*_par` names are namedtuples) and import together under
  `crimm-dev`; line 664 of `RDKitConverter.py` on this branch is the `pdbx_description`
  access it describes. `[ran]` / `[read]` msld-prep itself was not edited, and its notebook was
  not run.
- found: switching crimm branches changes what msld-prep sees (its G2); with this branch
  checked out its `align.ipynb` fails again. Remote branches were not touched: 22 of 36 are
  merged into `origin/master` (issue #95). Issues #60 to #62 (RTF parser) and #90 (renames)
  change names msld-prep imports; their bodies do not say so, because msld-prep is not public.
  Old issues #24 and #29 may overlap with #46 and #80; not reconciled.
- not done: branch protection; `xfail` reasons and TODO comments still lack issue numbers;
  ROADMAP is not yet an index of `#N`; crimm-ml's own ROADMAP still names `RFLoopBuilder`;
  `solvator-coor-orient-options` kept (owner decides); `RDKConverter-Het` kept and unpushed.
- next: Monday 2026-10-12, build CHARMM and pyCHARMM (issue #107); then link `xfail`/TODO to
  issues; then the chirality and PCA fixes (#43, #44) for Stan's review.

### 2026-10-08 claude-sonnet/claude-code/msld-prep-intro

- asked: by the owner, from a session in the msld-prep repo: record in crimm's dev docs that
  msld-prep exists and needs the parked hotfix `RDKConverter-Het`, so that agents here remind
  the owner when the fix is published. The owner allowed naming the project here now.
- did: docs only, in this file: a "What goes where" row, an Orientation bullet and a pickup
  bullet for msld-prep, an owner instruction (section 2) with the reminder rule, two
  working agreements (names msld-prep imports; keep the branch), a note on the
  `RDKConverter-Het` entry in State, and this entry. Also added the missing template and
  closing fence at the top of this section: the template had been replaced by the
  `hotfix-park` entry inside an unclosed code fence, so the whole log rendered as code. No
  entry was changed or removed. Nothing else in crimm was touched: no source, tests,
  configuration, branches or git state; nothing committed. `[ran]`
- why msld-prep needs the fix: an exploratory notebook in msld-prep calls
  `heterogen_to_rdkit(res, smiles=...)` on a `Residue` read from a CRD file. On
  `cleanup/phase0-tests-and-dev-docs` this raises `AttributeError: 'Residue' object has no
  attribute 'pdbx_description'` at `RDKitConverter.py:664`. `[ran 2026-10-08]` The second hunk
  of `8075b75` (the `hasattr` guard) is on that line `[read]`. I did not check out the branch
  or run the notebook with the fix. Whether the first hunk (warning instead of `TypeError`)
  is also needed is not known: the notebook was run only as far as that call.
- the owner's statement: the fix will not be published anytime soon. `[told 2026-10-08]`
- verified: `git show 8075b75` and the `msldprep` imports of crimm by grep `[read]`.
- not verified: the fix against msld-prep; whether the PyPI release of crimm has the same
  failure (not checked).
- found: the unclosed template fence (repaired above). Another session was editing this file
  at the same time (`CHARMM` rows, env notes); I re-read it just before the write and changed
  only the places listed above.
- next: when the fix is published, remind the owner (section 2). msld-prep's own handoff
  records the dependency in its Q12.

### 2026-10-08 claude-sonnet/claude-code/hotfix-park

- asked: commit the owner's uncommitted hotfix, park it, return to the cleanup branch, and
  report the conda env.
- did: committed the owner's edit to `crimm/Adaptors/RDKitConverter.py` as `8075b75` on the
  local branch `RDKConverter-Het` (cut from `72eef9f`, not pushed), then checked out
  `cleanup/phase0-tests-and-dev-docs`. The edit is the owner's; the agent only committed it.
  `[ran]`
- what the fix does: `RDKitHetConverter.__init__` warns instead of raising `TypeError` when
  the input is not a `Heterogen`; `heterogen_to_rdkit` guards `pdbx_description` with
  `hasattr`. `[read]` The owner's reason for the hotfix was not stated to the agent.
- known gaps (handle when the fix is picked up): no test covers `RDKitHetConverter` or
  `heterogen_to_rdkit` (grep of `tests/` found none); the `TypeError` contract was removed, so
  check callers that relied on it; a blank line at `RDKitConverter.py:351` has trailing
  whitespace; no GitHub issue exists for it yet.
  SUPERSEDED 2026-10-08 by claude-sonnet/claude-code/wrapup: the issue exists, #59.
- how to pick it up: `git checkout RDKConverter-Het`, add a test, then either merge it into
  the cleanup branch or cherry-pick `8075b75`. It is one concern, one commit, so either works.
  ROADMAP 1.5 (`import-fix.md`) does not edit `RDKitConverter.py` `[read]`, so the two should
  not conflict.
- env (follow-up, same day, at the owner's request): deleted the old py3.8 `crimm-dev`;
  cloned `crimm-dev-new` to `crimm-dev` so the name matches `environment-dev.yml` and
  CONTRIBUTING.md, then removed `crimm-dev-new`; re-registered kernel `crimm-dev`. `[ran]`
  The editable installs of `crimm` and `crimm-dock` in the old env pointed at source
  directories, which were not touched.
- merge: `git merge --no-ff origin/master` into `cleanup/phase0-tests-and-dev-docs`, no
  conflicts, `d90f94b`. Offline suite after the merge: 133 passed, 13 xfailed `[ran]`.
- gh (follow-up, at the owner's request): owner logged in as `Truman-Xu` (ADMIN on the repo,
  scopes `repo`, `read:org`, `admin:public_key`, `gist`). Created labels `correctness`,
  `refactor`, `agent-task`, `topology`, `solvation`, `io`, `visualization`, `loops`, `ml`
  (existing `documentation` used instead of `docs`), and milestones 1 to 6 = ROADMAP Phases
  0 to 5, titled as the ROADMAP headings. `[ran]` Milestone descriptions for Phases 2 to 5 are
  the agent's wording, not read from the sections.
- issues (same day, owner approved the list): created 68 issues #39 to #107 (draft id n is
  issue n+38; draft 67 was dropped at the owner's request, so #105 does not exist).
  Milestones 1 to 6 for Phases 0 to 5; backlog features #71 to #76 and the crimm-ml issues
  #101 to #104 have no milestone; #106 (toppar licence) is in Phase 5; #107 (CHARMM and
  pyCHARMM build and tests) is in Phase 0. Label `infra` added and put on #39 to #42. Stan
  is `@`-mentioned in #43, #44, #51, #52, #65, #66, #67, #85, #88. Draft-to-number map:
  scratchpad `issue_map.json` (not kept; the numbers above are enough). `[ran]`
- found: older open issues exist that may overlap: #24 ("Ligand topology not natively
  generated and not saved in psf") with #46, and #29 (CRD titles too long) with the title
  block duplication in #80. Not reconciled.
- owner decisions recorded: the RFdiffusion model is intended to become part of
  `LoopBuilder`, not a separate `RFLoopBuilder`; ROADMAP row "Keep RFLoopBuilder separate"
  deleted, note added to #101.
- docs: CHARMM licence and the local build plan (Monday 2026-10-12) are in ROADMAP "CHARMM and
  pyCHARMM", CONTRIBUTING, NOTES and this file (standing instructions, Q6, FACTS).
- not done: nothing pushed; branch protection not set; `env.yaml` and old remote branches
  not touched; docs edits of this session are uncommitted.

### 2026-10-07 claude-sonnet/claude-code/repo-setup

- asked: set up the repo for agents and humans (BEST_PRACTICES last two items); rebuild the
  dev environment.
- did: wrote `AGENTS.md`, `CLAUDE.md` (`@AGENTS.md`), `CONTRIBUTING.md`, issue forms, PR
  template and `CODEOWNERS` under `.github/` (Stan is `@stanislc`), `environment-dev.yml`.
  Committed in three commits and pushed. Added personal-file patterns (`CLAUDE.local.md`,
  `AGENTS.local.md`, `.claude/settings.local.json`, `.claude/commands/`) to
  `.git/info/exclude`.
  `[ran]`
- env: first `conda env create` failed (`python-build`, not `build`); fixed and restarted.
  The result was not checked before this entry was written. `[ran]`
- not done: `gh` install, labels, milestones, converting work orders to issues, branch
  protection, `env.yaml` removal, merging `origin/master` (1 commit behind), deleting the old
  env.
- resolves entry 3's open question: the root pointer is `AGENTS.md`.
- next: finish and test the env, swap it in, merge master, set up `gh`.

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
