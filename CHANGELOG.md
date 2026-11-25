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

- Renumbering a residue with alternative residue names (`DisorderedResidue`, e.g. a point
  mutation) now renumbers every alternative and the parent chain's lookup, not only the
  wrapper; `missing_res` no longer reports the new position as missing.
- `heterogen_to_rdkit` no longer raises `AttributeError` on a residue without
  `pdbx_description`, such as one read from a CRD file.
- `PolymerChain.seq` gives the one-letter code of DNA residues (`DA`, `DC`, `DG`, `DT`) and
  of other nucleotides with names shorter than three characters, instead of `X`.
