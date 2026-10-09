# crimm development notes

A working description of the codebase as it stands, for anyone about to change it.
Snapshot taken 2026-10-06 at commit `fbb165d` (branch `solvator-coor-orient-options`).
Companion documents: [ROADMAP.md](ROADMAP.md) and [BEST_PRACTICES.md](BEST_PRACTICES.md).

## Purpose and design goals

crimm prepares biomolecular structures for MD simulation and for ML model training.
Two goals decide most trade-offs:

- **Versatile and lightweight.** A scientist should be able to `pip install crimm` on a
  laptop, a cluster node or a container and get the core pipeline without a heavy stack.
- **Rich visualization in Jupyter.** Every structure entity can show itself in a notebook.

These pull in opposite directions (viewers are heavy), so the rule is: the core stays small,
and everything else is an optional extra that is imported only when used.

## Package map

| Package | What it holds | Notes |
| --- | --- | --- |
| `crimm/StructEntities` | `Structure`, `Model`, `OrganizedModel`, `Chain` family, `Residue`, `Atom`, topology elements and definitions | Subclasses of Biopython `Bio.PDB` entities. Biopython is a structural dependency, not a utility. |
| `crimm/IO` | Parsers and writers for mmCIF, PDB, CRD, PSF, RTF, PRM | `PSFWriter.py` is 1,402 lines. |
| `crimm/Modeller` | Topology generation, residue fixing, loop building, solvation, coordinate orientation, lone pairs | `TopoLoader.py` is 2,665 lines, `Solvator.py` 1,755. |
| `crimm/Adaptors` | Bridges to pyCHARMM, RDKit, PropKa, OpenMM, OpenBabel | Each one hard-imports its third-party package at module level. |
| `crimm/Visualization` | NGLView and py3Dmol backends behind `show()`, `set_backend()` | Backend choice is module-level global state. |
| `crimm/Superimpose` | `ChainSuperimposer` | Used by `Fetchers` and loop building. |
| `crimm/Fetchers.py` | RCSB and AlphaFold DB download | |
| `crimm/Utils` | Structure helpers, RCSB queries, `cuda_info` | `cuda_info` is exported from the top-level package but nothing in crimm uses it. |
| `crimm/Data` | Bundled CHARMM toppar files, element tables, `water_coords.npy`, probes | 11 MB, almost all toppar. |

The chain classes in `StructEntities/Chain.py` are: `BaseChain`, `Chain`, `PolymerChain`,
`Heterogens`, `Ligand`, `Macrolide`, `Oligosaccharide`, `Solvent`, `CoSolvent`, `Ion`,
`Glycosylation`, `NucleosidePhosphate`. `OrganizedModel` sorts chains into these types.

## The pipeline users run

1. `fetch_rcsb(..., organize=True)` downloads mmCIF and builds an `OrganizedModel`.
2. `ChainLoopBuilder` fills missing residues from AlphaFold templates.
3. `TopologyGenerator.generate(chain)` applies CHARMM36m residue definitions and patches.
   Ligands go through `CGENFFTopologyLoader`, which shells out to the `cgenff` executable.
4. `Solvator.solvate()` orients the solute, builds a water box, then `add_ions()`.
5. `write_psf` / `write_crd`, or load straight into pyCHARMM through `pyCHARMMAdaptors`.

`tests/benchmark_pipeline.py` runs stages 1 to 3 plus the pyCHARMM load across randomly
sampled PDB entries, one subprocess per structure.

## Who wrote what

By commit history (262 commits: Truman Xu 217, Stanislav Cherepanov 45):

- **Stan**: multi-crystal solvation and the SPLIT/SLTCAP ion methods in `Solvator.py`,
  `CrystalSDF.py`, `LonePairBuilder.py`, the PCA orientation methods in
  `CoordManipulator.py`, PSF/CRD reading and writing, offline `OrganizedModel`
  classification, and the 2026.2 topology hardening.
- **Truman**: the entity model, parsers, topology loader, loop builder, adaptors,
  visualization.

Changes to solvation should go to Stan for review.

## Packaging and release

- Build backend is setuptools via `pyproject.toml`. Versioning is calendar-based.
- **The version is written in several places and they disagree**: `pyproject.toml` says
  `2026.2`, `docs/conf.py` says `2026.1a`, the newest git tag is `2026.2.1`. There is no
  `crimm.__version__`.
- The classifier is still `Development Status :: 2 - Pre-Alpha`.
- `README.md` says Python 3.9 or newer; `env.yaml` asks for 3.10 or newer.
- Data files ship through `MANIFEST.in`. `crimm/Data/residues.xml` is tracked in git but is
  not in `MANIFEST.in`, and no Python file refers to it.
- `tests` is excluded from the built package.

## Dependencies

| Kind | Packages |
| --- | --- |
| Core (declared) | numpy, scipy, biopython, pandas, requests (with socks), nglview, ipywidgets |
| Extras (declared) | `protonation`: propka. `cheminformatics`: rdkit. `all`: both. |
| Used but not declared | py3Dmol |
| Install separately | pyCHARMM, OpenMM, OpenBabel, the `cgenff` executable |

`import crimm` currently imports **rdkit and nglview unconditionally**, even though rdkit is
only an extra. The two import chains and the fix are written up in `import-fix.md` at the
| Licensed, not obtainable without a licence | CHARMM and pyCHARMM (the owner told us on 2026-10-08), and the CGenFF program. Contributors without a licence cannot run the `pycharmm` and `cgenff` tests. |
repo root and scheduled in the roadmap (Phase 1).

There is no upper bound on numpy, and at least one call (`ndarray.ptp`) no longer exists in
NumPy 2.

## Tests and CI

- `tests/` holds an offline pytest suite (added 2026-10-06) built on five small mmCIF
  fixtures in `tests/data/`. Run it with `pytest`; it takes under a minute once crimm is
  imported. Known bugs are pinned with `xfail(strict=True)` tests.
- `tests/benchmark_pipeline.py` is not part of the suite. It needs network access and is
  meant for a SLURM job (`tests/slurm/`, untracked).
- The only GitHub workflow, `.github/workflows/static.yml`, publishes
  `docs/_build/html` from the `docs` branch to GitHub Pages. Nothing lints or tests code.
- Markers `network`, `pycharmm`, `cgenff`, `slow` and `gpu` are deselected by default.
  Select them with `-m`, for example `pytest -m network`, or `pytest -m ""` for everything.
  The `cgenff` tests read the executable path from `CRIMM_CGENFF_PATH`.

## Documentation

- Sphinx with autosummary and napoleon, `sphinx_rtd_theme`.
- `docs/_build` (12 MB of generated HTML) is committed. It is stale: it still documents a
  `crimm.fft_docking` module that no longer exists.
- `docs/index.rst` still contains the sphinx-quickstart placeholder paragraph, and
  `docs/usage.rst` is one install command.
- Ten tutorial notebooks under `tutorials/` take 25 MB because outputs are committed.
  The README lists `tutorials/` as the documentation.

## Conventions currently in the code

Described as they are, not as they should be. See the best-practices guide for targets.

- Module files are CamelCase (`TopoLoader.py`), named after their main class.
- User feedback is a mix of `print` (about 60 calls, 17 of them in `Solvator.py`) and
  `warnings.warn` (29 files). `logging` is not used anywhere in the package.
- Verbosity is controlled by `QUIET=False` keyword arguments threaded through
  `TopoLoader.py`.
- Runtime checks sometimes use `assert` (`CoordManipulator`, `Solvator`, `OrganizedModel`,
  `Atom`), which disappears under `python -O`.
- Every HTTP call uses `timeout=500` (one uses 1000) with no retry.
- Module-level global state: `LOADED_TOPOLOGY_TYPES` in `pyCHARMMAdaptors.py`, and
  `_backend` in `Visualization/__init__.py`.
- Data files are located with `os.path.dirname(Data.__file__)` arithmetic.
- Docstrings mix NumPy style and free-form text.
- `StructEntities/__init__.py` is entirely commented out, so entity classes must be imported
  from their modules.

## Things that will surprise you

- `Solvator.solvate()` **modifies the model in place** and reorients its coordinates by
  default (`orient_coords=True`). The default orientation for cube, octa and rhdo boxes is
  `CoordManipulator.orient_coords_octa`.
- Atoms are shared objects between a model and its parent structure, so moving a model's
  atoms moves the structure's too.
- `CoordManipulator` has two matrix conventions. `orient_coords` builds a column-vector
  4x4 matrix; the PCA variants store a row-vector one in the same attribute.
- `TopoElements.Dihedral.angle` is a stub that returns `0.000`.
- `CGENFFTopologyLoader` writes into the current working directory when no `save_path` is
  given.
- pyCHARMM holds one global PSF per process; the adaptor tracks what has been loaded in a
  module global, so state leaks between calls in one session.

## Repository state

The live record of branch, uncommitted work and who is editing what is the State section of
[AGENT_HANDOFF.md](AGENT_HANDOFF.md); it is not repeated here.

- `import-fix.md` at the repo root is an untracked work order, scheduled in the roadmap
  ("Make rdkit and nglview optional at import time") and deleted when that fix lands.
- `tests/slurm/` holds the owner's untracked benchmark job files.
- About 35 remote branches exist, most of them merged or abandoned. Both `main` and
  `master` exist on the remote; `master` is the default.

## Useful commands

```bash
# lint (pyflakes is installed in the torch210 conda env; ruff is not yet)
python -m pyflakes crimm

# list every TODO
grep -rn "TODO" --include="*.py" crimm

# pipeline benchmark, small run, no pyCHARMM
python tests/benchmark_pipeline.py --n 20 --no-charmm
```
