"""
Module for writing CHARMM PSF (Protein Structure File) format files.

This module provides functions to write molecular topology information from
crimm Model objects to CHARMM PSF format files. PSF files contain atomic
properties and connectivity information (bonds, angles, dihedrals, impropers,
CMAP terms, etc.).

Format specification (from CHARMM source io/psfres.F90):
- Extended format (EXT): I10 for integers, A8 for strings
- Standard format: I8 for integers, A4 for strings
- XPLOR format: Uses atom type names instead of parameter file indices
"""

from typing import Union, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from crimm.StructEntities import Model, Structure, Residue, Atom
from crimm.StructEntities.Chain import Chain
from crimm.StructEntities.TopoElements import Bond, Angle, Dihedral, Improper, CMap


@dataclass
class LonePairInfo:
    """Information about a lone pair for PSF output."""
    lp_atom: Atom
    host_atom: Atom
    distance: float = 0.0
    angle: float = 0.0
    dihedral: float = 0.0


class PSFWriter:
    """Write CHARMM PSF format files from crimm Model objects.

    Parameters
    ----------
    extended : bool, default True
        Use extended format (I10/A8) for large systems
    xplor : bool, default True
        Use XPLOR format (atom type names instead of indices)

    Attributes
    ----------
    extended : bool
        Whether extended format is used
    xplor : bool
        Whether XPLOR format is used
    """

    def __init__(self, extended: bool = True, xplor: bool = True):
        self.extended = extended
        self.xplor = xplor
        self._atom_map: Dict[Atom, int] = {}
        self._atoms: List[Atom] = []
        self._lonepairs: List[LonePairInfo] = []
        self._segid_map: Dict[Any, str] = {}  # Maps chain to assigned segid

    def write(self, model: Model, filepath: str, title: str = "") -> None:
        """Write PSF file for the given Model.

        Parameters
        ----------
        model : Model
            The Model object with topology to write
        filepath : str
            Path to output file
        title : str, default ""
            Title line(s) for the PSF header
        """
        psf_str = self.get_psf_string(model, title)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(psf_str)

    def _get_chains(self, entity: Union[Model, Chain]) -> List[Chain]:
        """Normalize input to a list of chains.

        Parameters
        ----------
        entity : Model or Chain
            Input entity

        Returns
        -------
        List[Chain]
            List of chains to process
        """
        if isinstance(entity, Chain):
            return [entity]
        return list(entity)

    def _get_topology(self, entity: Union[Model, Chain]):
        """Get topology container from entity.

        Parameters
        ----------
        entity : Model or Chain
            Input entity

        Returns
        -------
        TopologyElementContainer
            Topology containing bonds, angles, etc.
        """
        # Both Model and Chain have .topology attribute
        return entity.topology

    def get_psf_string(self, model: Union[Model, Chain], title: str = "") -> str:
        """Return PSF content as string.

        Parameters
        ----------
        model : Model or Chain
            The Model or Chain object with topology to write
        title : str, default ""
            Title line(s) for the PSF header

        Returns
        -------
        str
            PSF format string
        """
        # Reset state
        self._atom_map = {}
        self._atoms = []
        self._lonepairs = []
        self._segid_map = {}

        # Get topology container (Model has .topology, Chain has .topo_elements)
        topology = self._get_topology(model)
        if topology is None:
            entity_type = "Chain" if isinstance(model, Chain) else "Model"
            raise ValueError(
                f"{entity_type} has no topology. Generate topology first using "
                "TopologyGenerator.generate_model() or topo.generate()"
            )

        # Determine if CMAP terms present
        # First check topology.cmap, then fall back to extracting from residues
        # where ChainTopology.cmap may be None but residues have CMAP definitions
        cmap_terms = None
        if hasattr(topology, 'cmap') and topology.cmap is not None:
            cmap_terms = topology.cmap
        else:
            # Try to get CMAP from residues directly
            cmap_terms = self._get_cmaps_from_model(model)

        has_cmap = cmap_terms is not None and len(cmap_terms) > 0
        self._cmap_terms = cmap_terms  # Store for use in _write_cmap

        # Build atom index map (including lone pairs)
        self._build_atom_index_map(model)

        # Build sections
        sections = []
        sections.append(self._write_header(has_cmap))
        sections.append(self._write_title(title))
        sections.append(self._write_atoms(model))
        sections.append(self._write_bonds(topology))
        sections.append(self._write_angles(topology))
        sections.append(self._write_dihedrals(topology))
        sections.append(self._write_impropers(topology))
        sections.append(self._write_donors(model))
        sections.append(self._write_acceptors(model))
        sections.append(self._write_nonbonded())
        sections.append(self._write_groups(model))

        # Lone pair section (CHARMM always writes this, even if 0)
        # Must come before CMAP section
        sections.append(self._write_lonepairs())

        # CMAP section (if present) - always last
        if has_cmap:
            sections.append(self._write_cmap(topology))

        # Join sections with blank lines between them (CHARMM format)
        return "\n\n".join(sections) + "\n"

    def _assign_segids(self, model: Union[Model, Chain]) -> None:
        """Assign automatic segment IDs to chains without segids.

        Rules:
        - Protein: PRO{A,B,C,...}
        - DNA: DNA{A,B,C,...}
        - RNA: RNA{A,B,C,...}
        - Ligand/Other: LIG{A,B,C,...}

        Parameters
        ----------
        model : Model or Chain
            The Model or Chain object
        """
        # Track counts for each type to assign letters
        type_counts = {'PRO': 0, 'DNA': 0, 'RNA': 0, 'LIG': 0}
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        chains = [model] if isinstance(model, Chain) else model
        for chain in chains:
            # Check if chain already has segid set on its residues
            # Note: segid might be whitespace-only ('    '), which is truthy but empty
            has_segid = False
            for residue in chain:
                segid = getattr(residue, 'segid', None)
                if segid and segid.strip():  # Must have non-whitespace content
                    has_segid = True
                    break

            if has_segid:
                # Use existing segid
                continue

            # Determine chain type
            chain_type = getattr(chain, 'chain_type', None) or ''

            if 'polypeptide' in chain_type.lower():
                prefix = 'PRO'
            elif 'polydeoxyribonucleotide' in chain_type.lower():
                prefix = 'DNA'
            elif 'polyribonucleotide' in chain_type.lower():
                prefix = 'RNA'
            else:
                prefix = 'LIG'

            # Get the letter to use
            letter_idx = type_counts[prefix]
            if letter_idx < len(letters):
                letter = letters[letter_idx]
            else:
                letter = str(letter_idx)  # Fallback for >26 chains

            type_counts[prefix] += 1
            segid = f"{prefix}{letter}"

            # Store in map
            self._segid_map[chain] = segid

            # Also set on all residues of this chain for consistency
            for residue in chain:
                residue.segid = segid

    def _get_cmaps_from_model(self, model: Union[Model, Chain]) -> List[Tuple[Tuple[Atom, ...], Tuple[Atom, ...]]]:
        """Extract CMAP terms from residues in the model.

        Each residue may have a .cmap attribute containing CMAP definitions
        as tuples of (dihedral1_atom_names, dihedral2_atom_names).
        Atom names can have prefixes:
        - '-' : previous residue (e.g., '-C' for C atom in previous residue)
        - '+' : next residue (e.g., '+N' for N atom in next residue)

        Returns
        -------
        List of tuples of (dihedral1_atoms, dihedral2_atoms) where each
        dihedral is a tuple of 4 Atom objects.
        """
        cmaps = []
        for chain in self._get_chains(model):
            residues = list(chain.get_residues())
            for i, res in enumerate(residues):
                if not hasattr(res, 'cmap') or not res.cmap:
                    continue

                prev_res = residues[i - 1] if i > 0 else None
                next_res = residues[i + 1] if i < len(residues) - 1 else None

                for dihedral1_names, dihedral2_names in res.cmap:
                    try:
                        dihedral1_atoms = self._resolve_cmap_atoms(
                            dihedral1_names, res, prev_res, next_res
                        )
                        dihedral2_atoms = self._resolve_cmap_atoms(
                            dihedral2_names, res, prev_res, next_res
                        )
                        if dihedral1_atoms and dihedral2_atoms:
                            cmaps.append((dihedral1_atoms, dihedral2_atoms))
                    except Exception:
                        pass  # Skip CMAPs that can't be resolved

        return cmaps if cmaps else None

    def _resolve_cmap_atoms(
        self, atom_names: Tuple[str, ...], res: Residue,
        prev_res: Optional[Residue], next_res: Optional[Residue]
    ) -> Optional[Tuple[Atom, ...]]:
        """Resolve CMAP atom names to actual Atom objects.

        Parameters
        ----------
        atom_names : tuple of str
            Atom names, possibly with '-' or '+' prefixes
        res : Residue
            Current residue
        prev_res : Residue or None
            Previous residue in chain
        next_res : Residue or None
            Next residue in chain

        Returns
        -------
        Tuple of Atom objects, or None if resolution fails
        """
        atoms = []
        for name in atom_names:
            atom = None
            if name.startswith('-'):
                # Previous residue atom
                if prev_res is not None:
                    atom = self._get_atom_by_name(prev_res, name[1:])
            elif name.startswith('+'):
                # Next residue atom
                if next_res is not None:
                    atom = self._get_atom_by_name(next_res, name[1:])
            else:
                # Current residue atom
                atom = self._get_atom_by_name(res, name)

            if atom is None:
                return None
            atoms.append(atom)

        return tuple(atoms) if len(atoms) == len(atom_names) else None

    def _get_atom_by_name(self, residue: Residue, name: str) -> Optional[Atom]:
        """Get an atom from a residue by name."""
        for atom in residue.get_atoms():
            if atom.name == name:
                return atom
        return None

    def _build_atom_index_map(self, model: Union[Model, Chain]) -> None:
        """Build mapping from Atom objects to 1-based PSF indices.

        Also collects lone pair information for CGENFF ligands.

        Parameters
        ----------
        model : Model or Chain
            The Model or Chain object to index
        """
        # First assign automatic segids
        self._assign_segids(model)

        idx = 1
        # Normalize input: if Chain, wrap in list; if Model, iterate directly
        chains = [model] if isinstance(model, Chain) else model
        for chain in chains:
            for residue in chain:
                # Regular atoms
                for atom in residue.get_atoms():
                    self._atom_map[atom] = idx
                    self._atoms.append(atom)
                    idx += 1

                # Lone pairs (for CGENFF ligands)
                if hasattr(residue, 'lone_pair_dict') and residue.lone_pair_dict:
                    for lp_name, lp_atom in residue.lone_pair_dict.items():
                        self._atom_map[lp_atom] = idx
                        self._atoms.append(lp_atom)
                        idx += 1
                        # Track lone pair info for NUMLP section
                        # Find host atom from topology definition
                        if hasattr(residue, 'topo_definition') and residue.topo_definition:
                            lp_def = residue.topo_definition.get(lp_name)
                            if lp_def and hasattr(lp_def, 'lonepair_info'):
                                host_name = lp_def.lonepair_info.get('host')
                                if host_name and host_name in residue:
                                    self._lonepairs.append(LonePairInfo(
                                        lp_atom=lp_atom,
                                        host_atom=residue[host_name],
                                        distance=lp_def.lonepair_info.get('distance', 0.0),
                                        angle=lp_def.lonepair_info.get('angle', 0.0),
                                        dihedral=lp_def.lonepair_info.get('dihedral', 0.0)
                                    ))

    def _write_header(self, has_cmap: bool) -> str:
        """Write PSF header line with format keywords."""
        keywords = ["PSF"]
        if self.extended:
            keywords.append("EXT")
        if self.xplor:
            keywords.append("XPLOR")
        if has_cmap:
            keywords.append("CMAP")
        return " ".join(keywords)

    def _write_title(self, title: str) -> str:
        """Write NTITLE section.

        CHARMM format requires title lines to start with asterisk (*).
        """
        lines = []
        if title:
            title_lines = title.strip().split('\n')
        else:
            title_lines = ["REMARKS PSF file generated by crimm"]

        lines.append(self._format_section_header(len(title_lines), "NTITLE"))
        for line in title_lines:
            # Pad to 80 chars as CHARMM does
            padded_line = f"* {line}".ljust(80)
            lines.append(padded_line)
        return "\n".join(lines)

    def _write_atoms(self, model: Model) -> str:
        """Write NATOM section with atom properties."""
        lines = []
        lines.append(self._format_section_header(len(self._atoms), "NATOM"))

        for atom in self._atoms:
            serial = self._atom_map[atom]
            residue = atom.parent

            # Get residue/segment info
            if residue is not None:
                segid = residue.segid or ""
                resid = str(residue.id[1])
                resname = residue.resname
            else:
                segid = ""
                resid = "1"
                resname = "UNK"

            atomname = atom.name

            # Get topology info
            if atom.topo_definition is not None:
                atomtype = atom.topo_definition.atom_type
                charge = atom.topo_definition.charge
                mass = atom.topo_definition.mass
            else:
                atomtype = atom.element or "X"
                charge = 0.0
                mass = atom.mass if atom.mass else 0.0

            imove = 0  # Fixed atom flag (0 = free)

            lines.append(self._format_atom_line(
                serial, segid, resid, resname, atomname,
                atomtype, charge, mass, imove
            ))

        return "\n".join(lines)

    def _format_g14_6(self, value: float) -> str:
        """Format a float value in Fortran G14.6 style.

        G14.6 uses F notation for values that fit, E notation otherwise.
        Field width is 14, with 6 significant figures.
        """
        if value == 0.0:
            return f"{0.0:>14.6f}"

        abs_val = abs(value)
        # Use E notation for very small or very large values
        if abs_val < 0.01 or abs_val >= 1e6:
            # E notation: total width 14, 6 significant figures
            # Format: sign + 0. + 6 digits + E + sign + 2 digits = 13-14 chars
            return f"{value:>14.6E}"
        else:
            # F notation: total width 14, enough decimals for 6 sig figs
            return f"{value:>14.6f}"

    def _format_atom_line(
        self, serial: int, segid: str, resid: str, resname: str,
        atomname: str, atomtype: str, charge: float, mass: float, imove: int
    ) -> str:
        """Format a single atom line using Fortran-compatible G14.6 format."""
        charge_str = self._format_g14_6(charge)
        mass_str = self._format_g14_6(mass)

        if self.extended:
            # Extended XPLOR format: I10 A8 A8 A8 A8 A6 2G14.6 I8
            return (
                f"{serial:>10d} {segid:<8s} {resid:<8s} {resname:<8s} "
                f"{atomname:<8s} {atomtype:<6s}{charge_str}{mass_str}{imove:>8d}"
            )
        else:
            # Standard XPLOR format: I8 A4 A4 A4 A4 A4 2G14.6 I8
            return (
                f"{serial:>8d} {segid:<4s} {resid:<4s} {resname:<4s} "
                f"{atomname:<4s} {atomtype:<4s}{charge_str}{mass_str}{imove:>8d}"
            )

    def _write_bonds(self, topology) -> str:
        """Write NBOND section."""
        bonds = topology.bonds if topology.bonds else []
        indices = []
        for bond in bonds:
            a1, a2 = bond
            if a1 in self._atom_map and a2 in self._atom_map:
                indices.extend([self._atom_map[a1], self._atom_map[a2]])
        return self._format_index_section(indices, "NBOND: bonds", items_per_line=4)

    def _write_angles(self, topology) -> str:
        """Write NTHETA section."""
        angles = topology.angles if topology.angles else []
        indices = []
        for angle in angles:
            a1, a2, a3 = angle
            if all(a in self._atom_map for a in (a1, a2, a3)):
                indices.extend([
                    self._atom_map[a1],
                    self._atom_map[a2],
                    self._atom_map[a3]
                ])
        return self._format_index_section(indices, "NTHETA: angles", items_per_line=3)

    def _write_dihedrals(self, topology) -> str:
        """Write NPHI section."""
        dihedrals = topology.dihedrals if topology.dihedrals else []
        indices = []
        for dihe in dihedrals:
            atoms = list(dihe)
            if all(a in self._atom_map for a in atoms):
                indices.extend([self._atom_map[a] for a in atoms])
        return self._format_index_section(indices, "NPHI: dihedrals", items_per_line=2)

    def _write_impropers(self, topology) -> str:
        """Write NIMPHI section."""
        impropers = topology.impropers if topology.impropers else []
        indices = []
        for impr in impropers:
            atoms = list(impr)
            if all(a in self._atom_map for a in atoms):
                indices.extend([self._atom_map[a] for a in atoms])
        return self._format_index_section(indices, "NIMPHI: impropers", items_per_line=2)

    def _write_donors(self, model: Union[Model, Chain]) -> str:
        """Write NDON section.

        RTF format: DONOR HN N (hydrogen, heavy_atom)
        PSF format: (heavy_atom, hydrogen) - so we swap the order
        """
        indices = []
        for chain in self._get_chains(model):
            for residue in chain:
                if hasattr(residue, 'H_donors') and residue.H_donors:
                    for donor_pair in residue.H_donors:
                        if len(donor_pair) >= 2:
                            # RTF stores (hydrogen, heavy) but PSF writes (heavy, hydrogen)
                            first, second = donor_pair[0], donor_pair[1]

                            # Handle both string names and Atom objects
                            if isinstance(first, str):
                                # Tuple of atom names from RTF: (hydrogen, heavy)
                                h_name, d_name = first, second
                                if h_name in residue and d_name in residue:
                                    h_atom = residue[h_name]
                                    d_atom = residue[d_name]
                                else:
                                    continue
                            else:
                                # Atom objects directly
                                h_atom, d_atom = first, second

                            if d_atom in self._atom_map and h_atom in self._atom_map:
                                # Write as (heavy, hydrogen) for PSF format
                                indices.extend([
                                    self._atom_map[d_atom],
                                    self._atom_map[h_atom]
                                ])
        return self._format_index_section(indices, "NDON: donors", items_per_line=4)

    def _write_acceptors(self, model: Union[Model, Chain]) -> str:
        """Write NACC section.

        RTF format: ACCE O C (acceptor, antecedent)
        PSF format: (acceptor, antecedent) - same order
        """
        indices = []
        for chain in self._get_chains(model):
            for residue in chain:
                if hasattr(residue, 'H_acceptors') and residue.H_acceptors:
                    for acc_info in residue.H_acceptors:
                        # Handle both tuple of names and Atom objects
                        if isinstance(acc_info, (list, tuple)):
                            if len(acc_info) >= 2:
                                first, second = acc_info[0], acc_info[1]
                                if isinstance(first, str):
                                    # Tuple of atom names: (acceptor, antecedent)
                                    acc_name, ante_name = first, second
                                    if acc_name in residue and ante_name in residue:
                                        acc_atom = residue[acc_name]
                                        ante_atom = residue[ante_name]
                                    else:
                                        continue
                                else:
                                    # Atom objects directly
                                    acc_atom, ante_atom = first, second
                            elif len(acc_info) == 1:
                                # Single acceptor (no antecedent specified)
                                acc_name = acc_info[0]
                                if isinstance(acc_name, str):
                                    if acc_name in residue:
                                        acc_atom = residue[acc_name]
                                        ante_atom = None
                                    else:
                                        continue
                                else:
                                    acc_atom = acc_name
                                    ante_atom = None
                            else:
                                continue
                        else:
                            # Single Atom object
                            acc_atom = acc_info
                            ante_atom = None

                        if acc_atom in self._atom_map:
                            acc_idx = self._atom_map[acc_atom]
                            # Use antecedent index if available, otherwise 0
                            ante_idx = self._atom_map.get(ante_atom, 0) if ante_atom else 0
                            indices.extend([acc_idx, ante_idx])
        return self._format_index_section(indices, "NACC: acceptors", items_per_line=4)

    def _write_nonbonded(self) -> str:
        """Write NNB section (non-bonded exclusion list).

        PSF format for NNB section:
        - Header: NNB !NNB (number of explicit exclusion pairs)
        - INBLO array: one entry per atom (index of last exclusion)
        - IEXCL array: the actual exclusion atom indices (NNB entries)

        For simplicity, write NNB=0 (no explicit exclusions beyond bonded).
        The INBLO array still needs to be output (all zeros).
        """
        lines = []
        nnb = 0  # Number of explicit exclusion pairs (not atom count!)

        # Header: "       0 !NNB"
        lines.append(self._format_section_header(nnb, "NNB"))

        # CHARMM format requires blank line after NNB header before INBLO array
        lines.append("")

        # INBLO array: one entry per atom (all zeros = no exclusions)
        inblo = [0] * len(self._atoms)
        if inblo:
            lines.append(self._format_indices(inblo, items_per_line=8))

        return "\n".join(lines)

    def _write_groups(self, model: Union[Model, Chain]) -> str:
        """Write NGRP section (atom groups for charge computation).

        Groups are defined per residue from atom_groups.
        """
        lines = []
        groups = []
        atom_offset = 0

        for chain in self._get_chains(model):
            for residue in chain:
                if hasattr(residue, 'atom_groups') and residue.atom_groups:
                    for group in residue.atom_groups:
                        # Each group entry: (first_atom_in_group, group_type, move_flag)
                        if group:
                            first_atom = group[0]
                            if first_atom in self._atom_map:
                                groups.append((
                                    self._atom_map[first_atom] - 1,  # 0-based pointer
                                    1,  # Group type (1 for protein/standard groups)
                                    0   # Move flag (0 = free)
                                ))

        ngrp = len(groups)
        nst2 = 0  # Number of groups with ST2 flag

        # Header: NGRP NST2
        if self.extended:
            lines.append(f"{ngrp:>10d}{nst2:>10d} !NGRP NST2")
        else:
            lines.append(f"{ngrp:>8d}{nst2:>8d} !NGRP NST2")

        # Group data (3 integers per group, but 9 integers per line per CHARMM fmt04)
        indices = []
        for igpbs, igptyp, imoveg in groups:
            indices.extend([igpbs, igptyp, imoveg])

        if indices:
            lines.append(self._format_indices(indices, items_per_line=9))

        return "\n".join(lines)

    def _write_cmap(self, topology) -> str:
        """Write NCRTERM section (cross-map terms).

        Uses self._cmap_terms if available (from residue-level extraction),
        otherwise falls back to topology.cmap.
        """
        # Prefer stored CMAP terms from residue extraction
        if hasattr(self, '_cmap_terms') and self._cmap_terms:
            cmaps = self._cmap_terms
        elif hasattr(topology, 'cmap') and topology.cmap:
            cmaps = topology.cmap
        else:
            cmaps = []

        indices = []

        for cmap in cmaps:
            if isinstance(cmap, CMap):
                dihe1, dihe2 = cmap
                # Each CMAP has 8 atoms (2 dihedrals)
                atoms = list(dihe1) + list(dihe2)
            elif isinstance(cmap, tuple) and len(cmap) == 2:
                # Tuple of (dihedral1_atoms, dihedral2_atoms) from residue extraction
                dihe1, dihe2 = cmap
                atoms = list(dihe1) + list(dihe2)
            else:
                # Assume it's already a sequence of atoms
                atoms = list(cmap)

            if all(a in self._atom_map for a in atoms):
                indices.extend([self._atom_map[a] for a in atoms])

        # CMAP uses 8 atoms per entry, 1 entry per line (8 integers per line)
        return self._format_index_section(indices, "NCRTERM: cross-terms", items_per_line=8)

    def _write_lonepairs(self) -> str:
        """Write NUMLP NUMLPH section for lone pairs.

        CHARMM always writes this section, even when there are no lone pairs.
        """
        lines = []
        nlp = len(self._lonepairs)
        nlph = nlp  # Number of LP hosts

        if self.extended:
            lines.append(f"{nlp:>10d}{nlph:>10d} !NUMLP NUMLPH")
        else:
            lines.append(f"{nlp:>8d}{nlph:>8d} !NUMLP NUMLPH")

        # Lone pair host information (only if there are lone pairs)
        for lp_info in self._lonepairs:
            host_idx = self._atom_map.get(lp_info.host_atom, 0)
            lp_idx = self._atom_map.get(lp_info.lp_atom, 0)
            # Format: host_atom, lp_atom, type, distance, angle, dihedral
            if self.extended:
                lines.append(
                    f"{host_idx:>10d}{lp_idx:>10d}   F"
                    f"{lp_info.distance:>14.6f}{lp_info.angle:>14.6f}{lp_info.dihedral:>14.6f}"
                )
            else:
                lines.append(
                    f"{host_idx:>8d}{lp_idx:>8d}   F"
                    f"{lp_info.distance:>14.6f}{lp_info.angle:>14.6f}{lp_info.dihedral:>14.6f}"
                )

        return "\n".join(lines)

    def _format_section_header(self, count: int, label: str) -> str:
        """Format a section header line."""
        if self.extended:
            return f"{count:>10d} !{label}"
        else:
            return f"{count:>8d} !{label}"

    def _format_index_section(
        self, indices: List[int], label: str, items_per_line: int
    ) -> str:
        """Format a section with index data."""
        lines = []

        # Calculate count based on label
        # Each section stores data in a specific format:
        # - bonds, donors, acceptors: pairs (2 ints each)
        # - angles: triplets (3 ints each)
        # - dihedrals, impropers: quads (4 ints each)
        # - cross-terms (CMAP): 8 atoms each
        if "bonds" in label.lower():
            count = len(indices) // 2
        elif "angles" in label.lower():
            count = len(indices) // 3
        elif "dihedrals" in label.lower() or "impropers" in label.lower():
            count = len(indices) // 4
        elif "cross-terms" in label.lower():
            count = len(indices) // 8
        elif "donors" in label.lower() or "acceptors" in label.lower():
            count = len(indices) // 2  # Donors/acceptors are pairs (atom, hydrogen/antecedent)
        else:
            count = len(indices)

        lines.append(self._format_section_header(count, label))

        if indices:
            # Calculate actual integers per line (bonds: 4 pairs = 8 ints)
            if "bonds" in label.lower() or "donors" in label.lower() or "acceptors" in label.lower():
                ints_per_line = items_per_line * 2
            elif "angles" in label.lower():
                ints_per_line = items_per_line * 3
            elif "dihedrals" in label.lower() or "impropers" in label.lower():
                ints_per_line = items_per_line * 4
            elif "cross-terms" in label.lower():
                ints_per_line = 8  # 8 atoms per CMAP term
            else:
                ints_per_line = items_per_line

            lines.append(self._format_indices(indices, ints_per_line))

        return "\n".join(lines)

    def _format_indices(self, indices: List[int], items_per_line: int) -> str:
        """Format a list of indices into lines."""
        lines = []
        int_width = 10 if self.extended else 8

        for i in range(0, len(indices), items_per_line):
            chunk = indices[i:i + items_per_line]
            line = "".join(f"{idx:>{int_width}d}" for idx in chunk)
            lines.append(line)

        return "\n".join(lines)


# Convenience functions

def write_psf(
    model: Model,
    filepath: str,
    extended: bool = True,
    xplor: bool = True,
    title: str = ""
) -> None:
    """Write CHARMM PSF format file from a crimm Model.

    Parameters
    ----------
    model : Model
        The Model object with topology to write
    filepath : str
        Path to output file
    extended : bool, default True
        Use extended format (I10/A8) for large systems
    xplor : bool, default True
        Use XPLOR format (atom type names instead of indices)
    title : str, default ""
        Title line(s) for the PSF header
    """
    writer = PSFWriter(extended=extended, xplor=xplor)
    writer.write(model, filepath, title)


def get_psf_str(
    model: Model,
    extended: bool = True,
    xplor: bool = True,
    title: str = ""
) -> str:
    """Get CHARMM PSF format string from a crimm Model.

    Parameters
    ----------
    model : Model
        The Model object with topology to write
    extended : bool, default True
        Use extended format (I10/A8) for large systems
    xplor : bool, default True
        Use XPLOR format (atom type names instead of indices)
    title : str, default ""
        Title line(s) for the PSF header

    Returns
    -------
    str
        PSF format string
    """
    writer = PSFWriter(extended=extended, xplor=xplor)
    return writer.get_psf_string(model, title)
