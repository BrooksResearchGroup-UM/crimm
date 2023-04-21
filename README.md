# **CRIMM**
**CRIMM** stands for **Chemistry with the ReInvented Macromolecular Mechanics**. This project aims to integrate and supplement CHARMM with better object handle and APIs

## Why *"reinvent the wheel"*
This is a toolkit that is under active development, where many useful macromolecular modeling routines are selected to be reimplemented. This is an attempt to unify many macromolecular preparation/modeling routine under one platform while offering proper object handles and APIs in python. While currently, we aim to integrate with CHARMM and pyCHARMM, the broader goal is to provide highly usable, integratable, and scriptable python library/platform for simplifying any macromolecular modeling pipelines.

## Base Library and Object Handles
This library is built upon the excellent Biopython library. The macromolecular entity representations are derived from Biopython's entity classes and follow the same hierarchy. As a result, the entities in this library remain fully compatible with all functions and routines provided in Biopython.
## Parser Module
New and improved mmCIF parser is implemented to allow accurate structure representations and more complete information.

## Looper Module
For a given PDB structure with gaps or missing loops in the chain, this module provides functions to query PDB for the residue sequence and fill in the gaps or missing loop regions with the residues coordinates from the homology models.

## Structure Alignment/Superposition

## Visualization
