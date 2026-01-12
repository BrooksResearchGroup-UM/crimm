"""
Module for reading CHARMM PSF (Protein Structure File) format files.

This module provides functions to parse CHARMM PSF files and return
structured topology data that can be used for comparison or integration
with crimm structures.

Format specification (from CHARMM source io/psfres.F90):
- Extended format (EXT): I10 for integers, A8 for strings
- Standard format: I8 for integers, A4 for strings
- XPLOR format: Uses atom type names instead of parameter file indices
"""

from typing import List, Tuple, Dict, Any, NamedTuple
from dataclasses import dataclass, field


class PSFAtom(NamedTuple):
    """Atom record from PSF file."""
    serial: int
    segid: str
    resid: str
    resname: str
    atomname: str
    atomtype: str
    charge: float
    mass: float
    imove: int = 0


@dataclass
class PSFData:
    """Container for parsed PSF topology data.

    Attributes
    ----------
    title : List[str]
        Title lines from PSF file
    atoms : List[PSFAtom]
        Atom records
    bonds : List[Tuple[int, int]]
        Bond pairs (1-based atom indices)
    angles : List[Tuple[int, int, int]]
        Angle triplets (1-based atom indices)
    dihedrals : List[Tuple[int, int, int, int]]
        Dihedral quadruplets (1-based atom indices)
    impropers : List[Tuple[int, int, int, int]]
        Improper quadruplets (1-based atom indices)
    donors : List[Tuple[int, int]]
        H-bond donor pairs
    acceptors : List[Tuple[int, int]]
        H-bond acceptor pairs
    nonbonded : List[int]
        Non-bonded exclusion indices
    groups : List[Tuple[int, int, int]]
        Atom group definitions (pointer, type, move_flag)
    cmap : List[Tuple[int, ...]]
        CMAP cross-term definitions (8 atom indices each)
    lonepairs : List[Dict[str, Any]]
        Lone pair definitions
    extended : bool
        Whether file was in extended format
    xplor : bool
        Whether file was in XPLOR format
    has_cmap : bool
        Whether file contains CMAP terms
    """
    title: List[str] = field(default_factory=list)
    atoms: List[PSFAtom] = field(default_factory=list)
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    angles: List[Tuple[int, int, int]] = field(default_factory=list)
    dihedrals: List[Tuple[int, int, int, int]] = field(default_factory=list)
    impropers: List[Tuple[int, int, int, int]] = field(default_factory=list)
    donors: List[Tuple[int, int]] = field(default_factory=list)
    acceptors: List[Tuple[int, int]] = field(default_factory=list)
    nonbonded: List[int] = field(default_factory=list)
    groups: List[Tuple[int, int, int]] = field(default_factory=list)
    cmap: List[Tuple[int, ...]] = field(default_factory=list)
    lonepairs: List[Dict[str, Any]] = field(default_factory=list)
    extended: bool = False
    xplor: bool = False
    has_cmap: bool = False


class PSFReader:
    """Read CHARMM PSF files into structured data.

    Attributes
    ----------
    extended : bool
        Whether file uses extended format
    xplor : bool
        Whether file uses XPLOR format
    has_cmap : bool
        Whether file contains CMAP terms
    """

    def __init__(self):
        self.extended = False
        self.xplor = False
        self.has_cmap = False
        self._lines: List[str] = []
        self._line_idx: int = 0

    def read(self, filepath: str) -> PSFData:
        """Parse PSF file and return topology data.

        Parameters
        ----------
        filepath : str
            Path to PSF file

        Returns
        -------
        PSFData
            Parsed topology data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self._lines = [line.rstrip() for line in f.readlines()]

        self._line_idx = 0
        data = PSFData()

        # Parse header
        self._parse_header()
        data.extended = self.extended
        data.xplor = self.xplor
        data.has_cmap = self.has_cmap

        # Parse sections
        while self._line_idx < len(self._lines):
            line = self._current_line()

            if not line or line.startswith('!'):
                self._line_idx += 1
                continue

            if '!NTITLE' in line:
                data.title = self._parse_title(line)
            elif '!NATOM' in line:
                data.atoms = self._parse_atoms(line)
            elif '!NBOND' in line:
                data.bonds = self._parse_bonds(line)
            elif '!NTHETA' in line:
                data.angles = self._parse_angles(line)
            elif '!NPHI' in line:
                data.dihedrals = self._parse_dihedrals(line)
            elif '!NIMPHI' in line:
                data.impropers = self._parse_impropers(line)
            elif '!NDON' in line:
                data.donors = self._parse_pairs(line)
            elif '!NACC' in line:
                data.acceptors = self._parse_pairs(line)
            elif '!NNB' in line:
                data.nonbonded = self._parse_nonbonded(line)
            elif '!NGRP' in line:
                data.groups = self._parse_groups(line)
            elif '!NCRTERM' in line:
                data.cmap = self._parse_cmap(line)
            elif '!NUMLP' in line:
                data.lonepairs = self._parse_lonepairs(line)
            else:
                self._line_idx += 1

        return data

    def _current_line(self) -> str:
        """Get current line."""
        if self._line_idx < len(self._lines):
            return self._lines[self._line_idx]
        return ""

    def _next_line(self) -> str:
        """Advance and return next line."""
        self._line_idx += 1
        return self._current_line()

    def _parse_header(self) -> None:
        """Parse PSF header line to detect format flags."""
        line = self._current_line().upper()

        if not line.startswith('PSF'):
            raise ValueError("File does not appear to be a PSF file (missing PSF header)")

        self.extended = 'EXT' in line
        self.xplor = 'XPLOR' in line
        self.has_cmap = 'CMAP' in line

        self._line_idx += 1

    def _parse_count(self, line: str) -> int:
        """Parse count from section header line."""
        # Extract number before !LABEL
        parts = line.split('!')
        count_str = parts[0].strip()
        # Handle multiple numbers (e.g., NGRP NST2)
        nums = count_str.split()
        return int(nums[0]) if nums else 0

    def _parse_title(self, line: str) -> List[str]:
        """Parse NTITLE section."""
        count = self._parse_count(line)
        titles = []
        for _ in range(count):
            self._next_line()
            titles.append(self._current_line().strip())
        self._line_idx += 1
        return titles

    def _parse_atoms(self, line: str) -> List[PSFAtom]:
        """Parse NATOM section."""
        count = self._parse_count(line)
        atoms = []

        for _ in range(count):
            self._next_line()
            atom_line = self._current_line()
            atom = self._parse_atom_line(atom_line)
            atoms.append(atom)

        self._line_idx += 1
        return atoms

    def _parse_atom_line(self, line: str) -> PSFAtom:
        """Parse a single atom line."""
        # Split by whitespace for XPLOR format
        parts = line.split()

        if len(parts) < 9:
            raise ValueError(f"Invalid atom line: {line}")

        serial = int(parts[0])
        segid = parts[1]
        resid = parts[2]
        resname = parts[3]
        atomname = parts[4]
        atomtype = parts[5]
        charge = float(parts[6])
        mass = float(parts[7])
        imove = int(parts[8]) if len(parts) > 8 else 0

        return PSFAtom(
            serial=serial,
            segid=segid,
            resid=resid,
            resname=resname,
            atomname=atomname,
            atomtype=atomtype,
            charge=charge,
            mass=mass,
            imove=imove
        )

    def _read_integers(self, count: int, ints_per_element: int) -> List[int]:
        """Read specified number of integers from following lines."""
        total_ints = count * ints_per_element
        integers = []

        while len(integers) < total_ints:
            self._next_line()
            line = self._current_line()
            if not line or line.startswith('!') or '!' in line:
                break
            # Parse all integers from line
            nums = line.split()
            integers.extend(int(n) for n in nums)

        return integers[:total_ints]

    def _parse_bonds(self, line: str) -> List[Tuple[int, int]]:
        """Parse NBOND section."""
        count = self._parse_count(line)
        integers = self._read_integers(count, 2)

        bonds = []
        for i in range(0, len(integers), 2):
            bonds.append((integers[i], integers[i + 1]))

        self._line_idx += 1
        return bonds

    def _parse_angles(self, line: str) -> List[Tuple[int, int, int]]:
        """Parse NTHETA section."""
        count = self._parse_count(line)
        integers = self._read_integers(count, 3)

        angles = []
        for i in range(0, len(integers), 3):
            angles.append((integers[i], integers[i + 1], integers[i + 2]))

        self._line_idx += 1
        return angles

    def _parse_dihedrals(self, line: str) -> List[Tuple[int, int, int, int]]:
        """Parse NPHI section."""
        count = self._parse_count(line)
        integers = self._read_integers(count, 4)

        dihedrals = []
        for i in range(0, len(integers), 4):
            dihedrals.append((
                integers[i], integers[i + 1],
                integers[i + 2], integers[i + 3]
            ))

        self._line_idx += 1
        return dihedrals

    def _parse_impropers(self, line: str) -> List[Tuple[int, int, int, int]]:
        """Parse NIMPHI section."""
        count = self._parse_count(line)
        integers = self._read_integers(count, 4)

        impropers = []
        for i in range(0, len(integers), 4):
            impropers.append((
                integers[i], integers[i + 1],
                integers[i + 2], integers[i + 3]
            ))

        self._line_idx += 1
        return impropers

    def _parse_pairs(self, line: str) -> List[Tuple[int, int]]:
        """Parse NDON or NACC section (pairs of integers)."""
        count = self._parse_count(line)
        integers = self._read_integers(count, 2)

        pairs = []
        for i in range(0, len(integers), 2):
            pairs.append((integers[i], integers[i + 1]))

        self._line_idx += 1
        return pairs

    def _parse_nonbonded(self, line: str) -> List[int]:
        """Parse NNB section."""
        count = self._parse_count(line)

        # NNB contains exclusion list pointers (one per atom)
        integers = []
        while len(integers) < count:
            self._next_line()
            curr_line = self._current_line()
            if not curr_line or '!' in curr_line:
                break
            nums = curr_line.split()
            integers.extend(int(n) for n in nums)

        self._line_idx += 1
        return integers[:count]

    def _parse_groups(self, line: str) -> List[Tuple[int, int, int]]:
        """Parse NGRP NST2 section."""
        # Line format: NGRP NST2 !NGRP NST2
        parts = line.split('!')
        nums = parts[0].split()
        ngrp = int(nums[0]) if nums else 0

        integers = self._read_integers(ngrp, 3)

        groups = []
        for i in range(0, len(integers), 3):
            groups.append((integers[i], integers[i + 1], integers[i + 2]))

        self._line_idx += 1
        return groups

    def _parse_cmap(self, line: str) -> List[Tuple[int, ...]]:
        """Parse NCRTERM section (cross-map terms)."""
        count = self._parse_count(line)
        integers = self._read_integers(count, 8)

        cmaps = []
        for i in range(0, len(integers), 8):
            cmaps.append(tuple(integers[i:i + 8]))

        self._line_idx += 1
        return cmaps

    def _parse_lonepairs(self, line: str) -> List[Dict[str, Any]]:
        """Parse NUMLP NUMLPH section."""
        parts = line.split('!')
        nums = parts[0].split()
        nlp = int(nums[0]) if nums else 0

        lonepairs = []
        for _ in range(nlp):
            self._next_line()
            lp_line = self._current_line()
            parts = lp_line.split()
            if len(parts) >= 6:
                lonepairs.append({
                    'host': int(parts[0]),
                    'lp': int(parts[1]),
                    'type': parts[2],
                    'distance': float(parts[3]),
                    'angle': float(parts[4]),
                    'dihedral': float(parts[5])
                })

        self._line_idx += 1
        return lonepairs


# Convenience function

def read_psf(filepath: str) -> PSFData:
    """Read CHARMM PSF file and return parsed data.

    Parameters
    ----------
    filepath : str
        Path to PSF file

    Returns
    -------
    PSFData
        Parsed topology data
    """
    reader = PSFReader()
    return reader.read(filepath)


def compare_psf(psf1: PSFData, psf2: PSFData, verbose: bool = False) -> Dict[str, Any]:
    """Compare two PSF data structures.

    Parameters
    ----------
    psf1 : PSFData
        First PSF data
    psf2 : PSFData
        Second PSF data
    verbose : bool, default False
        Print detailed comparison results

    Returns
    -------
    Dict[str, Any]
        Comparison results with keys: 'equal', 'differences'
    """
    differences = []

    # Compare atom counts
    if len(psf1.atoms) != len(psf2.atoms):
        differences.append(f"Atom count: {len(psf1.atoms)} vs {len(psf2.atoms)}")

    # Compare atoms (if same count)
    if len(psf1.atoms) == len(psf2.atoms):
        for i, (a1, a2) in enumerate(zip(psf1.atoms, psf2.atoms)):
            if a1.atomname != a2.atomname:
                differences.append(f"Atom {i+1} name: {a1.atomname} vs {a2.atomname}")
            if a1.atomtype != a2.atomtype:
                differences.append(f"Atom {i+1} type: {a1.atomtype} vs {a2.atomtype}")
            if abs(a1.charge - a2.charge) > 1e-4:
                differences.append(f"Atom {i+1} charge: {a1.charge} vs {a2.charge}")
            if abs(a1.mass - a2.mass) > 1e-4:
                differences.append(f"Atom {i+1} mass: {a1.mass} vs {a2.mass}")

    # Compare topology counts
    counts = [
        ('bonds', len(psf1.bonds), len(psf2.bonds)),
        ('angles', len(psf1.angles), len(psf2.angles)),
        ('dihedrals', len(psf1.dihedrals), len(psf2.dihedrals)),
        ('impropers', len(psf1.impropers), len(psf2.impropers)),
        ('cmap', len(psf1.cmap), len(psf2.cmap)),
    ]

    for name, c1, c2 in counts:
        if c1 != c2:
            differences.append(f"{name.capitalize()} count: {c1} vs {c2}")

    # Compare bond sets (order-independent)
    bonds1 = set(tuple(sorted(b)) for b in psf1.bonds)
    bonds2 = set(tuple(sorted(b)) for b in psf2.bonds)
    if bonds1 != bonds2:
        missing1 = bonds2 - bonds1
        missing2 = bonds1 - bonds2
        if missing1:
            differences.append(f"Bonds in psf2 but not psf1: {len(missing1)}")
        if missing2:
            differences.append(f"Bonds in psf1 but not psf2: {len(missing2)}")

    result = {
        'equal': len(differences) == 0,
        'differences': differences
    }

    if verbose:
        if result['equal']:
            print("PSF files are equivalent")
        else:
            print("PSF files differ:")
            for diff in differences[:20]:  # Limit output
                print(f"  - {diff}")
            if len(differences) > 20:
                print(f"  ... and {len(differences) - 20} more differences")

    return result
