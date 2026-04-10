# crimm

**crimm** stands for **Chemistry with the ReInvented Macromolecular Mechanics**.

This is a Python toolkit for biomolecule structure preparation, designed to unify common modeling routines under one scriptable platform. While many tools exist for tasks like solvation, adding hydrogens, or building missing loops, they often lack the scriptability needed for high-throughput pipelines. crimm aims to fill this gap by providing intuitive Python APIs that integrate seamlessly with pyCHARMM, RDKit, and other computational chemistry tools.

## Why crimm?

- **Scriptable**: All preparation steps are Python functions—no clicking through web interfaces or writing shell scripts
- **Integrated**: Works natively with pyCHARMM for energy calculations and simulations
- **Accurate**: Uses mmCIF format to correctly identify chain types, detect missing loops, and handle biological assemblies
- **Extensible**: Adaptors connect crimm to RDKit, PropKa, OpenMM, and more

## Features

- Fetch structures from RCSB PDB or AlphaFold Database
- Organize mmCIF structures into protein, nucleic acid, ligand, solvent, and ion components
- Generate **CHARMM36m** topology for proteins and protein-containing systems
- Parameterize small-molecule ligands with **CGenFF** integration, including lone-pair-aware workflows
- Read, write, and reload native CHARMM PSF and CRD files
- Solvate in cubic or truncated octahedral water boxes
- Add monovalent ions at target concentrations (SPLIT, SLTCAP methods)
- Build missing loops from homology models
- Visualize structures in Jupyter notebooks with NGLView

## Installation

```bash
pip install crimm
```

Optional cheminformatics and protonation extras:

```bash
pip install "crimm[all]"
```

Requires Python >= 3.9. For a complete development environment:

```bash
conda env create -f env.yaml
```

> **Note**: pyCHARMM, OpenMM, and the external CGenFF executable must be installed separately when those workflows are needed.

## Quick Example

```python
from crimm.Fetchers import fetch_rcsb
from crimm.Modeller import TopologyGenerator
from crimm.Modeller.Solvator import Solvator
from crimm.IO import write_psf, write_crd

# Fetch and prepare structure
model = fetch_rcsb('1LSA', organize=True)
topo_gen = TopologyGenerator()
for chain in model.protein:
    topo_gen.generate(chain)

# Solvate and add 150 mM KCl
solvator = Solvator(model)
solvator.solvate(cutoff=10.0, box_type='octa')
solvator.add_ions(concentration=0.15, cation='POT', anion='CLA')

# Write CHARMM files
write_psf(model, 'system.psf')
write_crd(model, 'system.crd')
```

## Modules

| Module | Purpose |
|--------|---------|
| `Fetchers` | Download structures from RCSB PDB or AlphaFold |
| `Modeller` | Topology generation, PSF/CRD loading, solvation, loop building |
| `IO` | Read/write PDB, mmCIF, PSF, and CRD files |
| `Adaptors` | Connect to pyCHARMM, RDKit, PropKa |

## Documentation

See `tutorials/` for Jupyter notebooks on structure preparation, topology generation, PSF/CRD workflows, loop building, and more.

## License

GPLv3

## Links

- Repository: https://github.com/BrooksResearchGroup-UM/crimm
- Issues: https://github.com/BrooksResearchGroup-UM/crimm/issues
