# Test fixture structures

mmCIF files downloaded unmodified from `https://files.rcsb.org/download/<ID>.cif` on
2026-10-06. They are small on purpose so the default test run stays offline and fast.

| File | Entry | Why it is here |
| --- | --- | --- |
| `1ubq.cif` | Ubiquitin, 76 residues, 58 waters | Plain single-chain protein; the default fixture |
| `1crn.cif` | Crambin, 46 residues | Three disulfide bonds, no solvent |
| `2igd.cif` | Protein G domain, 61 residues, 106 waters | Alternate locations (37 disordered atoms) |
| `3ptb.cif` | Trypsin with benzamidine, 223 residues | Ligand (BEN), ion (CA), six disulfides |
| `1bna.cif` | B-DNA dodecamer, two strands | Nucleic acid |

To add a fixture, keep it under about 250 KB, add a row here, and add its expected values
to `EXPECTED` in `tests/conftest.py`.
