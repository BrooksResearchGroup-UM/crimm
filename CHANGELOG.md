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

- `TopologyGenerator.load_psf_crd` now gives each protein residue the patched definition it
  was built with (ACE, CT3, NTER, GLYP, PROP, CTER and the other terminal patches, and DISU),
  read off the residue's atoms and the PSF charges and types; nucleic-acid residues get DEO5,
  DEOX, 5TER and 3PHO the same way. Patch atoms such as the ACE and CT3 caps no longer come
  back without a `topo_definition`, so `total_charge` of a reloaded chain is a number, and the
  bonds, angles, dihedrals and impropers of the patched residues are restored.
- Renumbering a residue with alternative residue names (`DisorderedResidue`, e.g. a point
  mutation) now renumbers every alternative and the parent chain's lookup, not only the
  wrapper; `missing_res` no longer reports the new position as missing.
- `heterogen_to_rdkit` no longer raises `AttributeError` on a residue without
  `pdbx_description`, such as one read from a CRD file.
