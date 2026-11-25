# Changelog

User-visible changes to crimm. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). crimm uses calendar versions
(`YYYY.N` and `YYYY.N.P`); the compatibility policy is still to be written (#97).

Releases up to 2026.2.1 are described on the
[GitHub releases page](https://github.com/BrooksResearchGroup-UM/crimm/releases).

## [Unreleased]

### Added

- `test` extra: `pip install crimm[test]` installs pytest for running the test suite.

### Changed

- `RDKitHetConverter.load_heterogen` warns instead of raising `TypeError` when given a residue
  that is not a `Heterogen`, such as one read from a CRD file.

### Removed

- `crimm.CUDAInfo`, `crimm.is_cuda_available` and the module `crimm.Utils.cuda_info`. Nothing in
  crimm used them, and no known downstream project imports them.

### Fixed

- `Solvator.solvate()` no longer returns a mirror image of the solute. The PCA orientation
  used for cube, octa and rhdo boxes (`CoordManipulator.orient_coords_octa`, the default)
  applied a reflection to about half of all inputs, which inverts every chiral centre
  (L-amino acids become D). Structures solvated by earlier versions in such a box, or with
  `orient_method="octa"`, with `orient_coords=True` may be mirror images. To check one, compute for each residue the sign of
  `dot(cross(N - CA, C - CA), CB - CA)`: it is positive for L-amino acids and negative for
  all of them in a mirrored structure. Ortho, tetra and hexa boxes with their default orientation
  were not affected.
- After a PCA orientation (`CoordManipulator.orient_coords_octa`, `orient_coords_ortho`,
  `orient_coords_hexa`), the stored matrix (`op_mat`, `get_transformation_matrix`) is the
  transform that was applied, so `apply_coords` and `apply_entity` reproduce it; they applied
  a different transform before. All orientation methods now share one column-vector
  convention, and the PCA methods round coordinates to 4 decimals like `orient_coords`.
- The PCA orientation methods honour `apply_to_parent=True`: the whole parent entity is
  moved, each atom once, as `orient_coords` already did. They ignored the argument before.
  `Solvator.solvate()` passes it when the model has a parent structure, so the other models
  of a multi-model structure now move with the solvated one for every box type.
- Renumbering a residue with alternative residue names (`DisorderedResidue`, e.g. a point
  mutation) now renumbers every alternative and the parent chain's lookup, not only the
  wrapper; `missing_res` no longer reports the new position as missing.
- `heterogen_to_rdkit` no longer raises `AttributeError` on a residue without
  `pdbx_description`, such as one read from a CRD file.
