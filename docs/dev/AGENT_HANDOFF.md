# AGENT_HANDOFF

Shared memory for AI agents working on crimm. Any model, any tool. Written for agents first;
humans read it second.

```yaml
schema: agent-handoff/1
repo: github.com/BrooksResearchGroup-UM/crimm
default_branch: master
last_updated: 2026-10-14
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
| Anything about crimm-ml, the optional ML package (its code, model, service, state) | `docs/dev/AGENT_HANDOFF.md` in the crimm-ml repository |
| Machine- and account-specific setup (paths, cluster, accounts), and private downstream projects | the owner's `CLAUDE.local.md` (not in the repository) |
| What work is planned, its order, its status | `docs/dev/ROADMAP.md` |
| Standards the work should meet | `docs/dev/BEST_PRACTICES.md` |
| Conflicts and issues met preparing particular PDB entries (clashes, patch choices) | `docs/dev/PDB_PREP_LOG.md` |
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
  sections 0 to 5 of its `docs/dev/AGENT_HANDOFF.md` (same protocol as this file) and its
  `docs/dev/ROADMAP.md`. The owner's local checkout path is in `CLAUDE.local.md`. Do not copy
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
| 2026-10-06 | Public API may break, but old import paths must keep working with `DeprecationWarning`. | `[told]` |
| 2026-10-06 | Resolve TODO comments before the refactor and restructure phases. | `[told]` |
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
  a row in ROADMAP Phase 1. When you fix it, remove the marker in the same change.
- One concern per change. Do not mix a fix with a rename.
- **crimm-ml contract.** No module-level `import crimm_ml` anywhere in crimm. crimm-ml and
  the planned `RFLoopBuilder` use: `crimm.IO.PDBString.get_pdb_str`,
  `crimm.Modeller.ResidueTopologySet` (and `ResidueDefinition.create_residue`),
  `TopologyGenerator.apply_topo_def_on_residue`, `ResidueFixer.load_residue` /
  `build_missing_atoms`, `PolymerChain.gaps`, `.missing_res`, `.sort_residues`. If you rename
  or change one, keep the deprecated alias (as for any public API) and log a one-line
  pointer in crimm-ml's AGENT_HANDOFF.
- **Private downstream contracts.** `CLAUDE.local.md` lists the crimm names that private
  projects import. The same rule applies: keep a deprecated alias and log a one-line
  pointer in that project's handoff. `[inferred: same rule as for crimm-ml]`
- Do not delete the branch `RDKConverter-Het` or rewrite `8075b75` without telling the owner;
  a downstream project depends on it (`CLAUDE.local.md`). `[inferred]`

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
`pytest -p no:cacheprovider -q` (136 passed, 15 xfailed, 13 deselected on 2026-10-14).

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
| G14 | The test suite lives only on `cleanup/phase0-tests-and-dev-docs` until it is merged. A branch cut from `master` has no `tests/test_*.py`, so `pytest` there collects nothing and reports no failures. | `[ran 2026-10-06]` |

## 5. State

Rewrite in place. True as of `2026-10-14`.

```yaml
branches:                          # all start at fbb165d
  cleanup/phase0-tests-and-dev-docs:     # agent cleanup work; continue here. On origin at the
                                   # commit that carries this State (pushed 2026-10-14; the
                                   # sandbox push worked this time).
    - "2026-10-06 to 10-12: tests, dev docs, repo setup, merge of origin/master (d90f94b),
       benchmark move, ruff, CI, ROADMAP index (e976f0b)"
    - "2026-10-14: b2cb021 TODO(#N); 2b019d3 lint + 85640ad format of Stan's files;
       0bf85e4 .git-blame-ignore-revs; 6a00566 CI NumPy pin; 3132e52 PDB_PREP_LOG;
       then the docs/handoff commit"
  feat/coordmanipulator-find-max-dim:    # the owner's solvation edits, kept separate
    - 44e9bf6                      # CoordManipulator.find_max_dim; Solvator TODO comment.
                                   # WILL CONFLICT with 85640ad (format of the same files);
                                   # its new TODO needs `TODO(#66)` when merged.
  RDKConverter-Het:                # owner's hotfix, local only, not pushed. Cut from 72eef9f.
    - 8075b75                      # RDKitHetConverter accepts non-Heterogen; see LOG 2026-10-08.
                                   # Parked: needs a test and review before it joins anything.
                                   # Needed by a private downstream project: CLAUDE.local.md.
  solvator-coor-orient-options:    # old working branch; still points at fa7338a.
                                   # Redundant now. Owner decides whether to delete it.
env: "crimm-dev from environment-dev.yml, plus pycharmm and CHARMM_LIB_DIR (section 3).
      Account-specific install details: CLAUDE.local.md"
uncommitted_owner_work: none
uncommitted_agent_work: none
worktrees: "one extra worktree, for the hotfix branch (CLAUDE.local.md)"
untracked_not_ours:
  - import-fix.md                  # work order, see standing instructions
github:
  issues: "#39 to #111 (no #105), milestones 1 to 6 = Phases 0 to 5. #109 to #111 (ruff
           follow-ups) filed 2026-10-14 in Phase 0"
  branch_protection: "master: all-green required (strict false), 1 approving review, force pushes disallowed, deletions disallowed, admins not enforced [ran 2026-10-12, gh api; set by the owner]"
  pull_requests: "#108 draft, cleanup/phase0-tests-and-dev-docs into master. Body updated
                  2026-10-14 (Closes #40, #41, #42; per-commit list; MERGE WITH A MERGE COMMIT,
                  not squash, because of .git-blame-ignore-revs). CI on e976f0b: all 9 cells
                  green, every cell's pytest line read 2026-10-14. CI on 3132e52 (run 38592786917):
                  all 9 cells green; each cell's NumPy matches its name (py3.12 numpy<2 now
                  1.26.4, 134 passed, 12 xfailed) [ran 2026-10-14]. GitHub links #40-#42."
owner_reminders:                   # tell the owner at the start of the next session
  - "Reminders for private downstream projects (hotfix #59): CLAUDE.local.md"
  - "Merge #108 with a merge commit (not squash)"
  - "feat/coordmanipulator-find-max-dim will conflict with the format of Stan's files; rebase it"
test_suite: {passed: 136, xfailed: 15, deselected: 13, date: 2026-10-14}   # offline default run
test_suite_pycharmm: {passed: 8, xfailed: 2, date: 2026-10-14}   # pytest -m pycharmm in crimm-dev
roadmap_done:
  - "Phase 0: offline pytest suite, markers, benchmark move, ruff (lint only), CI workflow"
  - "Phase 0: ROADMAP as index (2026-10-12); TODO(#N) in every TODO (2026-10-14)"
  - "Phase 0: Stan's files linted and formatted (2026-10-14)"
  - "Phase 0: #107 CHARMM build, pyCHARMM in crimm-dev, tests/test_pycharmm.py (2026-10-14);
     open in #107: how a maintainer runs them before a release"
roadmap_next:
  - "#112 disulfide dihedrals (topology, owner's area); its xfails are in test_topology.py and
     test_pycharmm.py"
  - "Issue numbers in the xfail reasons (11 reasons, none numbered)"
  - "Mark #108 ready for review once Stan has seen it"
  - "Phase 0: golden-file tests for PSF and CRD (#39; Q1 first)"
library_code_changed_by_agents: "2026-10-12 and 2026-10-14, behaviour-neutral only: lint fixes
  (2026-10-12; 2026-10-14 in Stan's files), ruff format of Stan's three files, TODO comment
  text. Suite unchanged."
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
| Q6 | No pyCHARMM is available to agents on this machine, so the `pycharmm` marker has no tests and the pyCHARMM adaptor is unverified by any agent. Where can it be run? | 2026-10-06 **PLAN 2026-10-08 `[told]`:** the owner builds CHARMM locally, likely Monday 2026-10-12; see ROADMAP "CHARMM and pyCHARMM". Stays open until `import pycharmm` works in `crimm-dev`. **ANSWERED 2026-10-14:** in `crimm-dev` (section 3). `load_model` was run on 1CRN `[ran]`; `tests/test_pycharmm.py` added the same day (8 passed, 2 xfailed for #112) `[ran]`. |
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

### 2026-10-14 claude-opus/claude-code/charmm-build

- asked: pick up from 2026-10-12; build CHARMM (#107) from the owner's CHARMM checkout, compilers from
  `module av`. Mid-session: issue numbers for the TODOs; lint and format Stan's files (he
  agreed) keeping blame with him; update PR #108; explain the ruff issue filing; start a log of
  structure-specific prep issues (first: CT3 on 1CRN).
- did: owner chose `origin/master` in a new worktree, minimal CPU build, agent configures and
  submits. Created a detached worktree of that checkout at `c16e5816e`, wrote and submitted the
  SLURM job (3 tries: modules vs `set -e`; Python 3.6 vs `configure`), installed pyCHARMM
  editable into `crimm-dev`, set `CHARMM_LIB_DIR` there (section 3). Commits, pushed: `b2cb021`
  (26 TODOs -> `TODO(#N)`, all to existing issues), `2b019d3` (lint of Stan's files, per-file
  ignores removed, unused `max_serial` removed), `85640ad` (`ruff format` of them),
  `0bf85e4` (`.git-blame-ignore-revs`, CONTRIBUTING: merge-commit exception to squash),
  `6a00566` (CI NumPy pin), `3132e52` (`docs/dev/PDB_PREP_LOG.md`), then this docs commit.
  PR #108 body rewritten. Filed #109, #110, #111. Deleted the two 2026-10-13 reminder memories
  after delivering them.
- verified `[ran]`: CHARMM build exit 0, 1 min 51 s on a compute node; `import pycharmm` in
  activated `crimm-dev`; `load_model` on 1CRN: atoms and charge agree with CHARMM (CT3: 651, 0;
  CTER: 646, -1); CTER energy 178.3 kcal/mol. Offline suite 133 passed, 13 xfailed after each
  library change; `ruff check crimm tests` clean; ASTs of Stan's files unchanged by the format
  except whitespace in six docstrings; blame with the ignore file gives Stan 1,367 of
  Solvator's lines (1,348 before today; 38 lines new from the formatter stay on the format
  commit). CI on `3132e52` green in all cells with the right NumPy.
- not verified: any `pycharmm` test (none written yet); `get_charmm_coord_dict` and the
  non-PSF/CRD load paths; GitHub's blame view with the ignore file; the cluster's shared
  CHARMM builds; whether 1UBQ, 2IGD, 3PTB also clash with CT3.
- found: CI cell py3.12 numpy<2 had run NumPy 2.5.4 since 2026-10-12 (fixed). Squash-merging
  #108 would break `.git-blame-ignore-revs`. `generate_model` default CT3 builds its methyl
  onto the 1CRN Arg10-Asn46 salt bridge (VDW 4.2e6); owner: CT3 is intended (section 2,
  PDB_PREP_LOG). `max_serial` in `_convert_existing_waters_to_tip3` suggests built water
  hydrogens were meant to get serial numbers (told Stan in #108). pyCHARMM `stop` dumps core.
- followed (same day, owner's request): added `CHARMM_LIB_DIR` to the `env` of
  the `crimm-dev` kernel's `kernel.json` (outside the repo); a notebook run on
  that kernel with the variable unset in the shell imports pyCHARMM `[ran]`. Wrote
  `tests/test_pycharmm.py` (10 tests, marker `pycharmm`, no module-level pyCHARMM import,
  skips without `CHARMM_LIB_DIR`): CHARMM's reading of crimm's PSF equals crimm's file; crimm's
  PSF and energy equal CHARMM's own generation (sequence, patches, DISU, crimm's CRD) for 1CRN
  (CTER), 1UBQ, 2IGD; `fetch_coords_from_charmm` follows a shift applied in CHARMM (first run of
  `get_charmm_coord_dict`). Found #112 (filed): disulfide cysteines get `N/C-CA-CB-SG` twice and
  lack `HB*-CB-SG-SG'`; same dihedral count, DIHE off by 1.10 kcal/mol on 1CRN; 3PTB has 24
  duplicates. Pinned by strict xfails (offline in `test_topology.py` for 1CRN and 3PTB; the two
  1CRN comparisons in `test_pycharmm.py`); with `--runxfail` they fail on `dihedrals` and
  `DIHE` only `[ran]`. Comparing PSFs as sets hid the duplicates at first; the tests use
  multisets.
- verified `[ran]`: `-m pycharmm` 8 passed, 2 xfailed (about 10 s); offline 136 passed,
  15 xfailed, 13 deselected; `-m pycharmm` without `CHARMM_LIB_DIR` 10 skipped; ruff clean.
- not verified: the pycharmm tests with solvent, ligands (CGenFF) or nucleic acids; the
  legacy `use_psf_crd=False` load path.
- followed (same day, owner's request): moved machine- and account-specific facts (paths,
  cluster, the CHARMM build location) and private downstream projects from this file to the
  owner's `CLAUDE.local.md` (new standing instruction, section 2). Section 3 now describes
  how `crimm-dev` is built from `environment-dev.yml` and what was done for CHARMM and
  pyCHARMM, generically. Older LOG entries were redacted in place at the owner's request (an
  exception to "do not delete another agent's entry"); one entry moved to
  `CLAUDE.local.md`, with a stub here. The old text stays in git history.
- claims: none held.
- next: State `roadmap_next`.

### 2026-10-12 claude-sonnet/claude-code/phase0-lint-ci

- asked: remind the owner of the skipped `git worktree` lesson (done; the owner then created
  a worktree for the hotfix branch); list the working items; then do ROADMAP Phase 0 items
  "move benchmark", "ruff", "CI workflow", and write a lesson on ruff and CI for the owner.
  The CHARMM and pyCHARMM build (#107) was postponed to 2026-10-13 by the owner.
- did (committed as 734ffdb, b340111 and the CI/lesson/handoff commit; push and draft PR: see below): `git mv tests/benchmark_pipeline.py benchmarks/`;
  ruff in three steps: `ruff check --select I --fix` (54 files, excluding Stan's three files),
  safe fixes (unused imports and variables, `super()`, open modes, `yield from`, f-string,
  UP rules; one loop variable renamed in `get_charmm_coord_dict` to fix B020), then
  `[tool.ruff]` in `pyproject.toml` (select F, E4, E7, E9, I, B, UP; documented ignores;
  per-file ignores for `__init__.py`, `LoopBuilder.py` and Stan's files);
  `.github/workflows/ci.yml` (jobs `lint`, `test` matrix Python 3.9 to 3.13 x NumPy `<2`/`>=2`
  minus 3.13+`<2`, `all-green`); `docs/dev/lessons/ci-basics.md`; updated CONTRIBUTING, NOTES,
  ROADMAP rows, this file.
- verified `[ran]`: offline suite 133 passed, 13 xfailed, 3 deselected before the work and after
  each step (about 50 s); `ruff check crimm tests` exits 0 (ruff 0.16.10, py39 target);
  `import crimm` and subpackages import; the workflow YAML parses; `pip install --dry-run
  --python-version 3.9/3.10/3.13 --only-binary` resolves numpy, scipy, pandas, biopython,
  rdkit, ipywidgets for the edge matrix cells (nglview, propka and py3Dmol are sdist-only
  and were not part of that check).
- not verified: the workflow has not run on GitHub; no test was run on Python 3.9 to 3.11 or
  NumPy 1.x (so a red cell there is expected information, not a surprise); the pyCHARMM path
  (`get_charmm_coord_dict` rename is behaviour-neutral by reading, not run); `actionlint` is
  not installed. Branch protection not set.
- left for Stan (findings still in place, covered by per-file ignores in `pyproject.toml`):
  `Solvator.py` (unused imports `Atom`, `get_bounding_cube_size`, unused `max_serial`,
  f-string without placeholders, unsorted imports), `CrystalSDF.py` and
  `CoordManipulator.py` (unsorted imports).
- found: the handoff State said the 2026-10-08 commits were local only; git shows the branch on
  origin at `e5b855a`. `LoopBuilder.py` still has the undefined `topo` (Q2), kept visible by a
  per-file ignore with a comment.
- followed (same day, at the owner's request): updated the path in the owner's
  `tests/slurm/submit_pipeline.sh`; made the three commits `734ffdb`, `b340111`, `8cd6204`;
  pushed `cleanup/phase0-tests-and-dev-docs` (e5b855a..8cd6204) and opened **draft PR #108**
  into `master`. First CI run `[ran 2026-10-12/13 UTC]`: `lint`, all nine `test` cells
  (Python 3.9 to 3.13, NumPy 1.26.4 and 2.5.4) and `all-green` passed. The NumPy 1 cell shows
  134 passed, 12 xfailed: the `box_dim` xfail is conditional on NumPy >= 2, as designed.
  Not checked: the log of every cell, only py3.9+NumPy 1 and py3.13+NumPy 2. Branch
  protection is not set.
  This LOG addition and the State edits above are uncommitted (a fourth commit was not
  authorised).
- next: see `roadmap_next` in State: run CI on GitHub (needs the owner's go-ahead to push and
  open a draft PR), then read each matrix cell; CHARMM build on 2026-10-13.

### 2026-10-12 claude-sonnet/claude-code/wrapup

- asked: end-of-day wrap-up; commit and push this handoff; set branch protection on `master`
  to require `all-green`; write a note for the owner's meeting with Stan (in the owner's private
  notes, not in the repository).
- did: committed and pushed the handoff (the commit that carries this entry); wrote the meeting
  note there. `git log @{u}..HEAD` was empty before this commit, so nothing else was
  unpushed. No branch was deleted: no local branch other than `master` is merged into it.
- found: **`master` is already protected** (the State said `not_set`, wrong): 1 approving review
  required, no required status checks, force pushes allowed, admins not enforced `[ran,
  gh api repos/BrooksResearchGroup-UM/crimm/branches/master/protection]`.
- NOT done: requiring `all-green`. My `gh api -X PUT .../branches/master/protection` was denied
  by the sandbox's permission classifier; I did not retry or work around it. A PUT replaces the
  whole protection object, so the command must restate the existing settings. The owner runs, from an ordinary terminal
  (`gh` is in `crimm-dev`):
  `gh api -X PUT repos/BrooksResearchGroup-UM/crimm/branches/master/protection --input protection.json`
  with `{"required_status_checks":{"strict":false,"contexts":["all-green"]},"enforce_admins":false,
  "required_pull_request_reviews":{"required_approving_review_count":1,"dismiss_stale_reviews":false,
  "require_code_owner_reviews":false,"require_last_push_approval":false},"restrictions":null,
  "allow_force_pushes":true,"allow_deletions":false}`. Consider `allow_force_pushes: false`
  (a decision for the owner; it was left as found). The check name `all-green` exists only
  because PR #108 ran it, so set this after that run, as now.
- out of date and fixed in this commit: State `branch_protection`, `roadmap_next`, the ROADMAP
  CI row. Still out of date (not touched): ROADMAP is not an index of `#N`; `xfail` reasons and
  TODOs lack issue numbers; no GitHub issue exists yet for the ruff follow-ups (ignored rules,
  `ruff format`, `pre-commit`) or for Stan's lint findings; #107 stays open for tomorrow.
- addendum (same evening): the owner applied the protection themselves: `all-green` required,
  force pushes now disallowed `[ran, gh api]`. ROADMAP is now an index: every item carries its
  issue number, plus an "Index of issues" table (uncommitted when this was written).
  Agent memory holds two start-of-session reminders for 2026-10-13 (CHARMM build; the
  housekeeping items still out of date: `xfail`/TODO issue numbers, missing issues for the ruff
  follow-ups and Stan's lint findings, unread CI logs and the `Closes` lines of #108).
- claims: none were held.
- next: 2026-10-13 CHARMM and pyCHARMM build (#107, owner); meeting with Stan about linting and
  formatting his three files; then read the remaining CI logs and make #108 ready.

### 2026-10-08 claude-sonnet/claude-code/wrapup

- asked: end-of-day wrap-up; review a downstream project's session edits to the dev docs; then
  commit, push, delete merged branches, ignore `tests/slurm/`, and update all md docs and logs.
- did: committed the day's docs in three commits (CHARMM licence and plan; roadmap issue
  index and the LoopBuilder decision; this handoff). `git push origin
  cleanup/phase0-tests-and-dev-docs` FAILED: `Host key verification failed` although
  `~/.ssh/known_hosts` holds a matching github.com key; `ssh -T git@github.com` fails the
  same way inside the agent's sandbox. A retry outside the sandbox was refused, so nothing
  was pushed. Cause not established. Deleted the local branches `cuda_util`, `list`,
  `mmcif-bug-fix`, `newGridGen`, `py3dmol` (all merged into master; their SHAs were noted
  before deleting, so `git branch <name> <sha>` restores one); added `tests/slurm/` to
  `.git/info/exclude`; corrected State (branches, uncommitted work, untracked files, test
  date, GitHub block, owner reminders); updated ROADMAP (header, new row for #59) and NOTES
  (snapshot note); added a one-line pointer to crimm-ml's handoff, as the contract requires.
  `[ran]`
- reviewed (the 2026-10-08 entry about a private downstream project, now in the owner's
  `CLAUDE.local.md`): the crimm imports it lists exist and import together under
  `crimm-dev` `[ran]`.
- found: switching crimm branches changes what the downstream project sees. Remote branches
  were not touched: 22 of 36 are merged into `origin/master` (issue #95). Issues #60 to #62
  (RTF parser) and #90 (renames) change names that project imports. Old issues #24 and #29
  may overlap with #46 and #80; not reconciled.
- not done: the push (the owner runs `git push origin cleanup/phase0-tests-and-dev-docs` from
  an ordinary terminal); the pointer entry in crimm-ml's handoff
  is uncommitted in that repo; branch protection; `xfail` reasons and TODO comments still lack issue numbers;
  ROADMAP is not yet an index of `#N`; crimm-ml's own ROADMAP still names `RFLoopBuilder`;
  `solvator-coor-orient-options` kept (owner decides); `RDKConverter-Het` kept and unpushed.
- next: Monday 2026-10-12, build CHARMM and pyCHARMM (issue #107); then link `xfail`/TODO to
  issues; then the chirality and PCA fixes (#43, #44) for Stan's review.

### 2026-10-08 claude-sonnet/claude-code/downstream-intro

- Moved to the owner's `CLAUDE.local.md` on 2026-10-14 (it concerns a private downstream
  project). It added the hotfix reminder rule and the downstream contract now kept there, and
  repaired an unclosed template fence in this log.

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
  The editable installs in the old env pointed at source
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
