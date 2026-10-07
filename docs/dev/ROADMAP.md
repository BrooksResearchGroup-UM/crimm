# crimm cleanup and overhaul roadmap

Written 2026-10-06 against commit `fbb165d`. Background is in [NOTES.md](NOTES.md); the
standards the work should meet are in [BEST_PRACTICES.md](BEST_PRACTICES.md).

## Ground rules

- **Order matters.** Tests and CI come first (Phase 0), because nothing after that can be
  proven safe without them. Bugs next, then TODOs, then refactor, then restructure.
- **Public API may change, with deprecation.** Old import paths and names keep working and
  emit `DeprecationWarning` for at least one release before removal.
- **Solvation changes go through Stan.** Items touching `Solvator.py`, `CrystalSDF.py` or the
  PCA orientation methods are marked (Stan).
- **One concern per pull request.** A bug fix does not ride along with a rename.

Sizes are rough: S is under a day, M is a few days, L is a week or more.

## Phase 0: Safety net

Nothing here changes behaviour.

| Item | Done when | Size |
| --- | --- | --- |
| **Done 2026-10-06.** Offline pytest suite with small fixture structures checked into `tests/data/` (1UBQ, 1CRN, 2IGD, 3PTB, 1BNA) | `pytest` passes with no network; covers parsing, organizing, topology generation, PSF/CRD round trip, solvation. Result: 133 passed, 13 expected failures, each pinning a bug listed in Phase 1 | L |
| **Done 2026-10-06.** Markers for `network`, `pycharmm`, `cgenff`, `slow` | Default `pytest` run skips them; each can be selected with `-m`. There are no `pycharmm` tests yet | S |
| **Done 2026-10-07.** Agent and contributor files: `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, issue forms, PR template, `CODEOWNERS`, `environment-dev.yml` | A fresh agent session finds the handoff; contributors can set up an environment from one file | S |
| Move work orders to GitHub issues; milestones mirror these phases; ROADMAP becomes an index linking `#N` | Every item below has an issue; needs `gh` set up (see BEST_PRACTICES, Project management) | M |
| Golden-file tests for PSF and CRD output | A refactor that changes one byte of output fails a test | M |
| `ruff` configured in `pyproject.toml`; current findings fixed or explicitly ignored | `ruff check crimm` is clean | S |
| CI workflow: lint and tests on Python 3.9 to 3.13, on both NumPy 1.x and 2.x | Required check on pull requests to `master` | M |
| Move `tests/benchmark_pipeline.py` to `benchmarks/` | `tests/` holds only pytest tests | S |

## Phase 1: Bugs

Found while exploring the code. The first two are confirmed by running the code; the rest
are from reading it and from `pyflakes`.

### 1.1 Solvation can mirror the structure (Stan) — highest priority

`CoordManipulator.orient_coords_octa` (`crimm/Modeller/CoordManipulator.py`) rotates with the
raw PCA axes from `np.linalg.svd` and never checks the determinant. When the determinant is
-1 the "rotation" is a reflection, so every chiral centre is inverted and L-amino acids
become D. `orient_coords_ortho` and `orient_coords_hexa` have the check; `octa` does not.

This is the **default path**: `Solvator.solvate()` defaults to `box_type='cube'` and
`orient_coords=True`, and cube, octa and rhdo boxes all select the `octa` orientation.

Evidence: on 100 random synthetic coordinate sets, `orient_coords_octa` flipped handedness
in 51; `ortho`, `hexa` and the original `orient_coords` flipped none. It is also confirmed end
to end: `Solvator(model).solvate()` with default arguments on 1UBQ returns the mirror image,
with every residue's N, C, CB arrangement around CA inverted
(`tests/test_solvator.py::test_solvation_preserves_chirality`).

- Fix: add the determinant check (flip one axis when `det < 0`).
- Test: signed volume of a chiral centre is unchanged by every orientation method.
- Follow-up: say in the release notes that structures solvated with cube, octa or rhdo
  boxes by affected versions may be mirror images, and how to check.
- Size: S for the fix, plus the release note.

### 1.2 PCA orientation stores a transform that does not match what it applied (Stan)

The three PCA methods compute `new = centered @ rotation` (row-vector convention) and store
the translation in row 3 of `op_mat`. `apply_coords` and `apply_entity` read `op_mat` as a
column-vector matrix. Calling `apply_coords` on the original coordinates after any PCA
orientation gave a different result from what was applied, in 100 of 100 trials. The
`apply_to_parent` argument is accepted and ignored by all three.

- Fix: one helper builds the 4x4 matrix in one convention; all orientation methods use it.
  This also removes about 100 lines of triplicated code.
- Test: `apply_coords(original)` equals the stored coordinates for every method.
- Size: S.

### 1.3 Bugs found by the new test suite

Each has a test marked `xfail(strict=True)`, so the suite fails as soon as the bug is fixed
and the marker is not removed.

| Where | Problem | Test | Size |
| --- | --- | --- | --- |
| `IO/PSFWriter.py`, DNA | For 1BNA the PSF has 394 atoms per strand where the model has 383 (one atom duplicated in 11 of 12 residues), and the PSF total charge is +9 where the model's is -24. The CRD has the model's count, so the pair cannot be loaded together. Cause not yet investigated. | `test_psf_crd.py::test_dna_psf_atom_count_matches_model`, `::test_dna_psf_charge_matches_model` | M |
| `IO/PSFWriter.py` and `IO/CRDWriter.py` | A ligand without topology (no CGenFF) is left out of the PSF but written to the CRD (3PTB: 3416 against 3425 atoms). Decide on one behaviour for both writers. | `test_psf_crd.py::test_psf_and_crd_agree_with_untyped_ligand` | S |
| `TopologyGenerator.load_psf_crd` | Terminal patch atoms (ACE and CT3 caps) come back without a `topo_definition`, so `total_charge` of a reloaded chain is `None`. | `test_psf_crd.py::test_load_psf_crd_restores_charges` | S |
| `PolymerChain.seq` | Returns `X` for every DNA residue; `can_seq` is correct. | `test_parsing.py::test_dna_present_sequence_matches_canonical` | S |
| `IO/PDBString.get_pdb_str` | With default options, 4-character `TIP3` residue names shift the chain column and crimm's own `PDBParser` cannot read the output. `convert_water=True` works. | `test_pdb_io.py::test_round_trip_after_topology_default_options` | S |

### 1.4 Other bugs

| Where | Problem | Fix | Size |
| --- | --- | --- | --- |
| `Modeller/LoopBuilder.py:659` (`ArcLoopBuilder.reconstruct_backbone`) | Undefined name `topo`; the method raises `NameError` whenever it runs. `best_backbone` and `min_clashes` in `build` are assigned and never used, so the builder looks unfinished. | Decide whether `ArcLoopBuilder` is finished or removed; if kept, pass in a topology set and add a test. | M |
| `Modeller/CoordManipulator.py:128` (`box_dim`) | `self.coords.ptp(0)`; `ndarray.ptp` was removed in NumPy 2. | `np.ptp(self.coords, axis=0)`. | S |
| `CoordManipulator.load_entity` | Builds the full N by N distance matrix to find the farthest pair. Memory is quadratic: about 20 GB at 50,000 atoms. Runs even when a PCA method is used and the result is never needed. | Farthest pair over convex-hull vertices (`scipy.spatial.ConvexHull`), computed lazily. | S |
| `StructEntities/TopoElements.py` (`Dihedral.angle`) | Returns `0.000` for any dihedral. | Implement (this is also TODO 5 below). | S |
| `CoordManipulator`, `Solvator`, `OrganizedModel`, `Atom` | `assert` used for runtime validation; stripped under `python -O`. | Raise real exceptions. | S |
| Version strings | `pyproject.toml` 2026.2, `docs/conf.py` 2026.1a, tag 2026.2.1, no `crimm.__version__`. | Single source (see Phase 4); in the meantime align them. | S |
| `crimm/Data/residues.xml` | Tracked, not packaged, not referenced. | Delete, or package and use it. | S |
| `pyflakes` findings | Unused imports and variables in `Solvator.py`, `StructureUtils.py`, `PropKaAdaptors.py`, `RDKitConverter.py`, `charmm_struct_prep.py`. | Remove. | S |

### 1.5 Make rdkit and nglview optional at import time

`import crimm` imports rdkit and nglview unconditionally, although rdkit is only an extra.
A core `pip install crimm` therefore cannot import the package without them, and nglview
plus ipywidgets are forced on headless installs. The full analysis is in `import-fix.md` at
the repo root (verified against `fbb165d`); the work order is repeated here so the roadmap
stands on its own.

1. `crimm/Modeller/TopoLoader.py`: remove the top-level
   `from crimm.Adaptors.RDKitConverter import ...` and import inside
   `CGENFFTopologyLoader.__init__` and `CGENFFTopologyLoader.generate`, the only users.
2. `crimm/Adaptors/__init__.py`: lazy re-export of `RDKitHetConverter` through a module
   `__getattr__` (PEP 562).
3. `crimm/Superimpose/ChainSuperimposer.py`: import `show_nglview_multiple` inside `show()`.
4. `crimm/Visualization/__init__.py`: lazy re-exports through `__getattr__`, **and** local
   imports inside `show`, `show_multiple`, `get_viewer` and `show_residue`. Both are needed:
   a module `__getattr__` is not consulted for names used inside the module's own functions.
5. `crimm/Visualization/NGLVisualization.py`: move `from rdkit import Chem` into
   `get_structure_string`.
6. `pyproject.toml`: move `nglview` and `ipywidgets` out of core into a new `viz` extra, add
   `py3Dmol` there (it is imported but declared nowhere), and spell the new names out in
   `all`. This changes what `pip install crimm` provides, so bump the version and tell
   notebook users in the README to install `crimm[viz]`.
7. CI: a core-install job that runs `import crimm` and asserts that none of `rdkit`,
   `nglview`, `ipywidgets`, `py3Dmol` are in `sys.modules`.

Done when:

- with only core dependencies, `python -c "import crimm"` succeeds and neither rdkit nor
  nglview is in `sys.modules`;
- `crimm.Visualization.show(entity)` without nglview still raises the existing
  `ImportError` that names nglview;
- `from crimm.Adaptors import RDKitHetConverter` and
  `from crimm.Visualization import show_nglview` still work under `[all]`;
- the tutorials run unchanged under `[all]`;
- the built wheel's core `Requires-Dist` no longer lists nglview or ipywidgets;
- **`import-fix.md` is deleted** as the last commit of this work, once the checks above pass.

Size: M.

## Phase 2: Resolve every TODO comment

There are 27 TODO comments in `crimm/` (26 committed, plus one added in the uncommitted
`Solvator.py` edit). Each gets one outcome: **fix** in this phase, or **issue** (remove the
comment and open a GitHub issue, because it is a feature or a design question). After this
phase a TODO in the code must carry an issue number.

### Correctness: fix

| # | Location | TODO | Action |
| --- | --- | --- | --- |
| 1 | `IO/RTFParser.py:362` | `quad_parser` is not correct for IMPR | Give IMPR its own parser; test against `prot.rtf` impropers. |
| 2 | `IO/RTFParser.py:160` | `DELETE` parser handles only atom type | Parse all `DELETE` forms used in the bundled toppar (ATOM, BOND, ANGL, DIHE, IMPR, IC). |
| 3 | `IO/RTFParser.py:265` | Keywords DIHE, ANGLE and PATCH are not handled | Parse them, or raise on encountering them instead of skipping silently. |
| 4 | `Utils/StructureUtils.py:211` | DNA and RNA are not distinguished; everything is RNA | Decide from residue names (DA/DC/DG/DT against A/C/G/U) with an O2' check as fallback. |
| 5 | `StructEntities/TopoElements.py:190` | Dihedral angle calculation not implemented | Implement; same as bug 1.4. |
| 6 | `StructEntities/TopoDefinitions.py:30` | Element guessed from the first letter of the atom name | Look the element up from the atom type's MASS entry in the RTF; fall back to the name only with a warning. |
| 7 | `Modeller/Solvator.py:170` | Atom serial numbers above 99,999 (Stan) | PDB output: hybrid-36 or wrap with a warning. CRD/PSF: confirm the extended format is used. Add a large-system test. |

### Cleanup: fix

| # | Location | TODO | Action |
| --- | --- | --- | --- |
| 8 | `Modeller/Solvator.py:22` | Gather constants in `Data/constants.py` (Stan) | Move physical constants; keep box-specific ones beside the solvator. |
| 9 | `Modeller/Solvator.py:1174` | Move ion-chain charge into the `Utils` charge function (Stan) | Fold `_get_ion_chain_charge` into `StructureUtils.get_charges`. |
| 10 | `IO/MMCIFParser.py:443` | Refactor nested loops | Extract per-chain and per-residue helpers; golden-file tests guard it. |
| 11 | `StructEntities/Chain.py:128` | Move disordered-reset to the structure builder | Move; keep a thin method on the chain. |
| 12 | `Modeller/TopoLoader.py:947` | Use logging for the skipped BLNK entry | Done as part of the logging work in Phase 3; remove the comment then. |
| 13 | `IO/RTFParser.py:91` | Tidy `quad_parser` loop | One-line comprehension; delete the commented-out line. |

### Documentation: fix

| # | Location | TODO | Action |
| --- | --- | --- | --- |
| 14 | `Modeller/TopoLoader.py:1585` | Add examples to the `coerce_resname` docstring | Add a doctest-style example. |
| 15 | `Superimpose/ChainSuperimposer.py:9` | Refactor and add docstring examples | Examples now; the refactor is covered by issue 23. |

### Topology design: issue, resolved in Phase 4

These five are one piece of work: patches and topology elements should come from the RTF
definitions instead of hand-written special cases.

| # | Location | TODO |
| --- | --- | --- |
| 16 | `Modeller/TopoLoader.py:559` | Full DISU patching from the patch definition |
| 17 | `Modeller/TopoLoader.py:740` | Read CMAP from the RTF |
| 18 | `Modeller/TopoLoader.py:1139` | Parse parameters from the CGenFF toppar block |
| 19 | `Modeller/TopoLoader.py:1211` | Build ligand topology elements from the RTF |
| 20 | `Modeller/TopoLoader.py:2456` | Rewrite `ResiduePatcher` and the topology definitions around `Topology` |
| 21 | `StructEntities/TopoDefinitions.py:348` | Add a type to identify heterogens |

### Features: issue

| # | Location | TODO |
| --- | --- | --- |
| 22 | `IO/PDBString.py:251` | CONECT records |
| 23 | `Superimpose/ChainSuperimposer.py:206` | `on_atoms` option for loop-edge superposition |
| 24 | `IO/CRDParser.py:110` | CRD header parser |
| 25 | `Visualization/NGLVisualization.py:105` | `add_representation` for a list of entities |
| 26 | `Visualization/NGLVisualization.py:204` | Colour specific residues in a cartoon |
| 27 | `StructEntities/Chain.py:192` | Het-flag check for chromophore residues in fluorescent proteins |

## Phase 3: Refactor

Internals only. Public names and behaviour stay the same; the golden-file tests prove it.

| Item | Detail | Size |
| --- | --- | --- |
| Logging | `logging.getLogger(__name__)` per module replaces about 60 `print` calls. One documented way to set verbosity replaces the `QUIET` keyword arguments (kept as deprecated aliases). | M |
| Exceptions | A small hierarchy rooted at `CrimmError` (topology, parsing, fetch, solvation). | S |
| HTTP | One helper for RCSB and AlphaFold requests with a sensible timeout (30 s, not 500), retries with backoff, and proxy handling. Replaces 10 call sites. | S |
| Duplication | Orientation methods (done in 1.2); the title and timestamp block repeated in `CRDWriter.py` and `PSFWriter.py`; ion name tables repeated inside `Solvator.py`. | M |
| Resources | `importlib.resources` to locate toppar files and `water_coords.npy`. | S |
| Global state | `LOADED_TOPOLOGY_TYPES` and the visualization backend become explicit objects or context managers. | M |
| Types and docstrings | Type hints on public functions; NumPy-style docstrings throughout; fix wrong annotations (for example `_find_transformation_operators -> None` returns a matrix). | L |
| Dead code | Commented-out `StructEntities/__init__.py`, unused attributes such as `_convex_hull`, leftover commented lines. | S |
| Reproducibility (Stan) | Monte Carlo ion placement takes a `seed` or `Generator`. | S |

## Phase 4: Restructure

Breaking changes, each with a compatibility shim.

| Item | Detail | Size |
| --- | --- | --- |
| Split `TopoLoader.py` | Its 11 classes move into a `topology` subpackage: topology containers, `ParameterLoader`, residue topology sets, CGenFF loading, `TopologyGenerator`, `ResiduePatcher`. `crimm.Modeller.TopoLoader` keeps re-exporting them. | L |
| Topology from definitions | TODOs 16 to 21: patches, CMAP and ligand topology derived from RTF data. | L |
| Split `Solvator.py` (Stan) | Crystal types and geometry, water-box construction, ion calculation and placement as separate modules; `Solvator` stays the entry point. | M |
| Public API | `__all__` in every package, lazy top-level imports, `crimm.__version__` from package metadata, entity classes exported from `crimm.StructEntities`. | M |
| Module names | snake_case module and package names; the CamelCase paths re-export with `DeprecationWarning`. Do this last and in one release, since it touches every import. | L |
| Extras | `viz`, `cheminformatics`, `protonation`, `openmm`, `dev`, `docs`, `all`; `ml` is added by the crimm-ml track below and stays out of `all`. | S |
| Orphans | Decide the future of `Utils/cuda_info.py` (445 lines, exported at top level, unused inside crimm) and `Data/probes` (994 lines). Both look left over from the removed docking module; they may belong in a separate package. | S |
| Docs build | Remove `docs/_build` from git; build and deploy in CI from `master`; retire the `docs` branch. Replace the placeholder `index.rst`. | M |
| Notebooks | Strip outputs on commit (`nbstripout`) and render them in the docs build instead; run them in CI. | M |
| Repository hygiene | Delete merged remote branches; remove the duplicate `main` branch. | S |

## Phase 5: Release

| Item | Size |
| --- | --- |
| `CHANGELOG.md` covering everything above, with a migration section for renamed imports and the `viz` extra | S |
| Deprecation schedule stating the release in which each shim is removed | S |
| Publish to PyPI from CI on a tag, using trusted publishing | S |
| Update the `Development Status` classifier | S |
| conda-forge feedstock | M |

## Parallel track: crimm-ml integration (RFdiffusion loop building)

RFdiffusion loop building is being split into its own distribution, `crimm-ml`
(github.com/Truman-Xu/crimm-ml, import `crimm_ml`), which depends on crimm and runs the
model on ONNX Runtime without PyTorch. `crimm-ml` takes PDB lines and a contig string and
returns coordinates; it knows nothing about crimm chains. The chain-facing glue (choosing
gaps, writing the contig, rebuilding full residues from the result) belongs in crimm, next
to the topology code it uses, so it can change with it.

This is not part of the cleanup phases and does not block them. It starts **after "1.5 Make rdkit and nglview
optional at import time" has landed**: `crimm[ml]` is pointless while `import crimm` still needs rdkit and nglview.

| Item | Done when | Size |
| --- | --- | --- |
| New `crimm/Modeller/RFLoopBuilder.py`, a placeholder lifted from the RFdiffusion fork's `Tests.ipynb` (cell 7 and the gap loop in cells 9 to 13): `build_residues(chain, built_res_coords)` (real residue names from `chain.missing_res`, `ResidueTopologySet('protein')`, `TopologyGenerator.apply_topo_def_on_residue`, `ResidueFixer.build_missing_atoms`), `insert_built_residues(chain, residues, inplace=False)`, and `fill_gaps(chain, model, gaps=None, build_terminals=False, T=50, num_designs=1, seed=None, progress=None)`. `crimm_ml` is imported only inside functions. Mark it a placeholder in the docstring: this is the surface expected to change. | Filling every internal gap of 5IEV chain A matches the notebook: real residue names, complete side chains, a continuous chain. `import crimm` does not import `crimm_ml`. | M |
| `ml = ["crimm-ml>=0.1"]` in `[project.optional-dependencies]`, kept **out of** `all`: crimm-ml depends on crimm, so `all` would pull in a package that depends back on crimm. | `pip install crimm[ml]` resolves in a clean venv from TestPyPI. | S |
| `ml` pytest marker (skipped by default, like `network`) for the `fill_gaps` test, which needs the 275 MB model; the model path comes from `CRIMM_ML_TEST_MODEL`. | Default `pytest` stays offline and fast; `pytest -m ml` runs it. | S |
| Document that crimm-ml needs Python 3.10 or newer (its oldest usable onnxruntime, 1.18, has no 3.9 wheel past 1.19), so `crimm[ml]` does not resolve on 3.9 even though crimm itself supports it. | README install section says so. | S |
| Keep `RFLoopBuilder` separate from `Modeller/LoopBuilder.py`. The existing module is the classical builder; `ArcLoopBuilder` there has its own open bug ("1.4 Other bugs"). | Two modules, no shared state. | S |

Once this lands, crimm-ml removes its deprecated copies of the old glue (`crimm_ml.crimm_utils`,
`crimm_ml.minimal_pipeline_loop_builder`) in its next release.

## Suggested first three pull requests

1. Chirality fix and transform-matrix fix in `CoordManipulator` (1.1, 1.2), with tests.
   Small, and the most serious correctness problem.
2. Rest of Phase 0: golden-file tests, ruff, CI workflow. The pytest suite, fixtures and
   markers are in (commit `d10c62d`).
3. Optional imports (1.5), ending with the deletion of `import-fix.md`.
