# **crimm**
**crimm** stands for **Chemistry with the ReInvented Macromolecular Mechanics**. This project aims to integrate and supplement CHARMM with better object handle and APIs

## Why *"reinvent the wheel"*
This is a toolkit that is under active development, where many useful macromolecular modeling routines are selected to be reimplemented. This is an attempt to unify many macromolecular preparation/modeling routine under one platform while offering proper object handles and APIs in python. While currently, we aim to integrate with CHARMM and pyCHARMM, the broader goal is to provide highly usable, integratable, and scriptable python library/platform for simplifying any macromolecular modeling pipelines.

-----------------
## Installations
crimm can be installed by `pip install crimm`

crimm requires `python>=3.8`. The main dependencies are biopython, nglview, scipy, and requests. To use the adaptors, the respective packages need to be installed separately (e.g. pyCHARMM, rdkit, etc.)

If you are installing crimm on a fresh enviroment, it is recommended to use the `env.yaml` file. 

```conda env create -f env.yaml```

**Note**
1. `OpenMM` and `pyCHARMM` still need to be installed separately in this environment if you require these in your pipeline.

2. If you are using a centralized `Jupyterlab` installation and install the ipython kernel to it, the `nglview` version should match in both environment (`crimm` env and `jupyterlab` env). Otherwise the ipywidget for `nglview` could break. The required `nglview` version is currently 3.0.6

-----------------
## Base Library and Object Handles
This library is built upon the excellent Biopython library. The macromolecular entity representations are derived from Biopython's entity classes and follow the same hierarchy. As a result, the entities in this library remain fully compatible with all functions and routines provided in Biopython.
## Parser Module
New and improved mmCIF parser is implemented to allow accurate structure representations and more complete information.

## Looper Module
For a given PDB structure with gaps or missing loops in the chain, this module provides functions to query PDB for the residue sequence and fill in the gaps or missing loop regions with the residues coordinates from the homology models.

## Structure Alignment/Superposition

Structure Alignment utilizes Biopython's [Superimposer](http://biopython.org/DIST/docs/tutorial/Tutorial.html#sec241) tool. However, sequence alignment based on the canonical sequence will be performed prior to the superimposition to determine where the residues should be aligned for two polymer chains that are not identical in sequence identity.

## Visualization
[NGLView](http://nglviewer.org/nglview/latest/) is integrated for a direct visualization of structures for Jupyter Notebook/JupuyterLab.