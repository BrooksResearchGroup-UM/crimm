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

import warnings
from typing import Union, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.Residue import Residue
from crimm.StructEntities.Chain import BaseChain
from crimm.StructEntities.Model import Model
from crimm.StructEntities.TopoElements import CMap


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
    separate_crystal_segids : bool, default False
        If True, crystallographic waters/ions get separate segids (CRWT/CION)
        from generated ones (SOLV/IONS). If False (default), all waters use
        SOLV and all ions use IONS regardless of source.

    Attributes
    ----------
    extended : bool
        Whether extended format is used
    xplor : bool
        Whether XPLOR format is used
    separate_crystal_segids : bool
        Whether crystallographic chains get separate segids

    Examples
    --------
    Default behavior - all waters/ions share segids:

    >>> writer = PSFWriter()
    >>> writer.write(model, 'system.psf')  # Waters->SOLV, Ions->IONS

    Separate crystallographic chains:

    >>> writer = PSFWriter(separate_crystal_segids=True)
    >>> writer.write(model, 'system.psf')  # Crystal waters->CRWT, Generated->SOLV
    """

    def __init__(
        self,
        extended: bool = True,
        xplor: bool = True,
        separate_crystal_segids: bool = False
    ):
        self.extended = extended
        self.xplor = xplor
        self.separate_crystal_segids = separate_crystal_segids
        self._atom_map: Dict[Atom, int] = {}
        self._atoms: List[Atom] = []
        self._lonepairs: List[LonePairInfo] = []
        self._segid_map: Dict[Any, str] = {}  # Maps chain to assigned segid

    def validate_for_simulation(
        self, model: Union[Model, BaseChain], strict: bool = False
    ) -> List[str]:
        """Validate model topology before writing PSF.

        Performs CHARMM-style validation checks that would normally occur
        during PSF generation. Since loading a pre-built PSF skips these
        checks, we must validate before writing.

        Parameters
        ----------
        model : Model or BaseChain
            The entity to validate
        strict : bool, default False
            If True, raise ValueError for any issues.
            If False, issue warnings and return list of issues.

        Returns
        -------
        List[str]
            List of validation issues found (empty if valid)

        Raises
        ------
        ValueError
            If strict=True and validation issues are found
        """
        issues = []

        chains = self._get_chains(model)

        for chain in chains:
            chain_id = chain.id if hasattr(chain, 'id') else str(chain)

            # Check 1: Missing parameters (like CHARMM's "BOND NOT FOUND" etc.)
            if hasattr(chain, 'topology') and chain.topology is not None:
                topo = chain.topology
                if hasattr(topo, 'missing_param_dict') and topo.missing_param_dict:
                    for param_type, missing_list in topo.missing_param_dict.items():
                        if missing_list:
                            issues.append(
                                f"Chain {chain_id}: {len(missing_list)} {param_type} "
                                f"parameters not found"
                            )

            # Check 2: Atoms without topology definitions
            for residue in chain.get_residues():
                res_id = f"{residue.resname} {residue.id[1]}"
                res_def = getattr(residue, 'topo_definition', None)

                if res_def is None:
                    issues.append(
                        f"Chain {chain_id}, Residue {res_id}: "
                        f"No topology definition"
                    )
                    continue

                # Check 3: Atoms with missing or suspicious charges
                for atom in residue:
                    # ResidueDefinition uses __contains__ and __getitem__
                    if atom.name not in res_def:
                        issues.append(
                            f"Chain {chain_id}, Residue {res_id}, Atom {atom.name}: "
                            f"Not defined in topology"
                        )
                    else:
                        atom_def = res_def[atom.name]
                        if not hasattr(atom_def, 'charge') or atom_def.charge is None:
                            issues.append(
                                f"Chain {chain_id}, Residue {res_id}, Atom {atom.name}: "
                                f"No charge defined (will use 0.0)"
                            )

        # Report issues
        if issues:
            msg = (
                f"PSF validation found {len(issues)} issue(s) that may cause "
                f"incorrect simulation results:\n"
            )
            # Limit output to first 20 issues
            for issue in issues[:20]:
                msg += f"  ** WARNING ** {issue}\n"
            if len(issues) > 20:
                msg += f"  ... and {len(issues) - 20} more issues\n"

            if strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning)

        return issues

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

    def _get_chains(self, entity: Union[Model, BaseChain]) -> List[BaseChain]:
        """Normalize input to a list of chains.

        Parameters
        ----------
        entity : Model or BaseChain
            Input entity

        Returns
        -------
        List[BaseChain]
            List of chains to process
        """
        if isinstance(entity, BaseChain):
            return [entity]
        if isinstance(entity, Model):
            return list(entity.chains)
        if isinstance(entity, List):
            if all(isinstance(c, BaseChain) for c in entity):
                return entity
        raise ValueError(
            f"Input must be a Model or Chain object"
            f" or a list of Chain objects, but got {type(entity)}"
        )


    def _get_topology(self, entity: Union[Model, BaseChain]):
        """Get topology container from entity.

        Parameters
        ----------
        entity : Model or BaseChain
            Input entity

        Returns
        -------
        TopologyElementContainer or ModelTopology
            Topology containing bonds, angles, etc.
        """
        if isinstance(entity, BaseChain):
            return entity.topology

        # For Model, use existing ModelTopology if available
        # This avoids recreating topology and adding duplicate elements
        if hasattr(entity, 'topology') and entity.topology is not None:
            return entity.topology

        # Otherwise create new ModelTopology which properly handles:
        # - Disulfide bonds (DISU patch: removes HG1, changes SG type to SM)
        # - Inter-chain bonds
        # - Combined topology elements from all chains
        from crimm.Modeller.TopoLoader import ModelTopology
        return ModelTopology(entity)

    def _combine_chain_topologies(self, model: Model):
        """Combine topologies from all chains in the model.

        Parameters
        ----------
        model : Model
            Model containing multiple chains

        Returns
        -------
        CombinedTopology
            Object with combined bonds, angles, dihedrals, impropers, cmap
        """
        class CombinedTopology:
            """Simple container for combined topology elements."""
            def __init__(self):
                self.bonds = []
                self.angles = []
                self.dihedrals = []
                self.impropers = []
                self.cmap = []

        combined = CombinedTopology()

        for chain in model:
            chain_topo = getattr(chain, 'topology', None)
            if chain_topo is None:
                continue

            # Combine all topology elements
            if chain_topo.bonds:
                combined.bonds.extend(chain_topo.bonds)
            if chain_topo.angles:
                combined.angles.extend(chain_topo.angles)
            if chain_topo.dihedrals:
                combined.dihedrals.extend(chain_topo.dihedrals)
            if chain_topo.impropers:
                combined.impropers.extend(chain_topo.impropers)
            if hasattr(chain_topo, 'cmap') and chain_topo.cmap:
                combined.cmap.extend(chain_topo.cmap)

        return combined

    def get_psf_string(self, model: Union[Model, BaseChain], title: str = "") -> str:
        """Return PSF content as string.

        Parameters
        ----------
        model : Model or BaseChain
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

        # Validate topology before writing (CHARMM-style pre-generation checks)
        # This catches issues that CHARMM would normally find during PSF generation
        # but which are skipped when loading a pre-built PSF file
        self.validate_for_simulation(model, strict=False)

        # Get topology container (Chain has .topology, Model uses ModelTopology wrapper)
        topology = self._get_topology(model)
        if topology is None:
            entity_type = "Chain" if isinstance(model, BaseChain) else "Model"
            raise ValueError(
                f"{entity_type} has no topology. Generate topology first using "
                "TopologyGenerator.generate_model() or topo.generate()"
            )

        # Determine if CMAP terms present
        # First check topology.cmap, then fall back to extracting from residues
        # where ChainTopology.cmap may be None or empty but residues have CMAP definitions
        cmap_terms = None
        if hasattr(topology, 'cmap') and topology.cmap:  # Check for non-empty
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
        sections.append(self._write_title(title, model))
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

        # CMAP section - always write for append mode compatibility
        # (CHARMM expects NCRTERM section when CMAP flag is present)
        sections.append(self._write_cmap(topology))

        # Join sections with blank lines between them (CHARMM format)
        return "\n\n".join(sections) + "\n"

    def _assign_segids(self, model: Union[Model, BaseChain]) -> None:
        """Assign automatic segment IDs to chains without segids.

        Rules:
        - Protein: PRO{A,B,C,...}
        - DNA: DNA{A,B,C,...}
        - RNA: RNA{A,B,C,...}
        - Solvent: SOLV (single segment for all waters)
        - Ion: IONS (single segment for all ions)
        - Ligand/Other: LIG{A,B,C,...}

        Parameters
        ----------
        model : Model or BaseChain
            The Model or Chain object
        """
        # Track counts for each type to assign letters
        type_counts = {'PRO': 0, 'DNA': 0, 'RNA': 0, 'LIG': 0}
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        chains = [model] if isinstance(model, BaseChain) else model
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
            elif chain_type.lower() == 'solvent':
                # By default, all waters share SOLV segid
                # If separate_crystal_segids=True, crystallographic waters get CRWT
                if self.separate_crystal_segids and (
                    chain.source is None or chain.source.lower() != 'generated'
                ):
                    segid = 'CRWT'
                else:
                    segid = 'SOLV'
                self._segid_map[chain] = segid
                for residue in chain:
                    residue.segid = segid
                continue
            elif chain_type.lower() == 'ion':
                # By default, all ions share IONS segid
                # If separate_crystal_segids=True, crystallographic ions get CION
                if self.separate_crystal_segids and (
                    chain.source is None or chain.source.lower() != 'generated'
                ):
                    segid = 'CION'
                else:
                    segid = 'IONS'
                self._segid_map[chain] = segid
                for residue in chain:
                    residue.segid = segid
                continue
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

    def _get_cmaps_from_model(self, model: Union[Model, BaseChain]) -> List[Tuple[Tuple[Atom, ...], Tuple[Atom, ...]]]:
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
                    except (KeyError, AttributeError, IndexError) as e:
                        # CMAP resolution can fail at chain termini or with incomplete topology
                        warnings.warn(
                            f"Could not resolve CMAP for residue {res.resname} {res.id}: {e}. "
                            f"CMAP term will be omitted from PSF file.",
                            UserWarning
                        )

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

    def _build_atom_index_map(self, model: Union[Model, BaseChain]) -> None:
        """Build mapping from Atom objects to 1-based PSF indices.

        Also collects lone pair information for CGENFF ligands.

        Parameters
        ----------
        model : Model or BaseChain
            The Model or Chain object to index
        """
        # First assign automatic segids
        self._assign_segids(model)

        idx = 1
        # Normalize input: if Chain, wrap in list; if Model, iterate directly
        chains = [model] if isinstance(model, BaseChain) else model
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
        """Write PSF header line with format keywords.

        Note: CMAP flag is always included for compatibility with CHARMM's
        PSF append mode, which expects consistent format flags.
        """
        keywords = ["PSF"]
        if self.extended:
            keywords.append("EXT")
        if self.xplor:
            keywords.append("XPLOR")
        # Always include CMAP for append mode compatibility
        keywords.append("CMAP")
        return " ".join(keywords)

    def _write_title(self, title: str, model: Model = None) -> str:
        """Write NTITLE section.

        CHARMM format requires title lines to start with asterisk (*).
        If no title is provided and model is given, informative system
        info is extracted. Otherwise a basic default title is used.

        Parameters
        ----------
        title : str
            User-provided title text (can be multiline)
        model : Model, optional
            Model to extract system information from for auto-generated title
        """
        # Import system info generator from CRDWriter
        from crimm.IO.CRDWriter import _generate_system_info

        lines = []
        if title:
            title_lines = title.strip().split('\n')
        elif model is not None:
            # Auto-generate informative title from model
            title_lines = _generate_system_info(model)
        else:
            # Basic default title
            import datetime
            import getpass
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                user = getpass.getuser()
            except Exception:
                user = "unknown"
            title_lines = [
                "PSF file generated by crimm",
                f"Created: {timestamp} by {user}"
            ]

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
                if resname == 'HIS':
                    # Use residue definition resname for HIS variants
                    resname = residue.topo_definition.resname
            else:
                # Missing topology - use fallback values with warning
                warnings.warn(
                    f"Atom {atom.name} in residue {residue.resname} {residue.id} "
                    f"has no topology definition. Using fallback values: "
                    f"type={atom.element or 'X'}, charge=0.0, mass={atom.mass or 0.0}. "
                    f"Simulation results may be incorrect!",
                    UserWarning
                )
                atomtype = atom.element or "X"
                charge = 0.0
                mass = atom.mass if atom.mass else 0.0

            imove = 0  # Movement flag (0 = free to move, non-zero = constrained)

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

    def _write_donors(self, model: Union[Model, BaseChain]) -> str:
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

    def _write_acceptors(self, model: Union[Model, BaseChain]) -> str:
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
                            if ante_atom is not None:
                                ante_idx = self._atom_map.get(ante_atom, 0)  
                            else:
                                ante_idx = 0
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

    def _determine_group_type(self, residue) -> int:
        """Determine CHARMM group type based on residue charges.

        Returns:
            0: No charges (all atoms have zero partial charge, e.g., dummy atoms)
            1: Neutral (non-zero charges that sum to zero)
            2: Charged (non-zero net charge, e.g., ions, ASP, GLU, LYS, ARG)
        """
        has_any_charge = False
        total_charge = 0.0

        for atom in residue.get_atoms():
            if atom.topo_definition is not None:
                charge = atom.topo_definition.charge
                if abs(charge) > 1e-6:
                    has_any_charge = True
                total_charge += charge

        if not has_any_charge:
            return 0  # No charges (dummy atoms)
        elif abs(total_charge) > 0.01:
            return 2  # Charged group
        else:
            return 1  # Neutral group

    def _write_groups(self, model: Union[Model, BaseChain]) -> str:
        """Write NGRP section (atom groups for charge computation).

        Groups are defined per residue from atom_groups.
        Groups must be sorted by first atom index in ascending order.

        Group types (igptyp) per CHARMM documentation:
        - 0: No charges (all atoms have zero partial charge)
        - 1: Neutral (non-zero charges that sum to zero)
        - 2: Charged (non-zero net charge, e.g., ions, charged residues)
        - 3: ST2 (special ST2 water model)
        """
        lines = []
        groups = []

        for chain in self._get_chains(model):
            for residue in chain:
                # Determine group type based on residue's charges
                # Type 0: no charges (all atoms have zero partial charge, e.g., dummy atoms)
                # Type 1: neutral (non-zero charges that sum to zero)
                # Type 2: charged (non-zero net charge, e.g., ions, ASP, GLU, LYS, ARG)
                group_type = self._determine_group_type(residue)

                if hasattr(residue, 'atom_groups') and residue.atom_groups:
                    for group in residue.atom_groups:
                        # Each group entry: (first_atom_in_group, group_type, move_flag)
                        if group:
                            first_atom = group[0]
                            if first_atom in self._atom_map:
                                groups.append((
                                    self._atom_map[first_atom] - 1,  # 0-based pointer
                                    group_type,  # Group type (1=neutral, 2=charged)
                                    0   # Move flag (0 = free)
                                ))

        # CRITICAL: Groups must be sorted by first atom index (igpbs) in ascending order
        # CHARMM expects groups in order for proper charge neutrality calculations
        groups.sort(key=lambda x: x[0])

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
        else:
            # Empty section needs extra blank line for CHARMM append mode compatibility
            lines.append("")

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
    title: str = "",
    separate_crystal_segids: bool = False
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
    separate_crystal_segids : bool, default False
        If True, crystallographic waters/ions get separate segids (CRWT/CION)
        from generated ones (SOLV/IONS). If False (default), all waters use
        SOLV and all ions use IONS regardless of source.

    Examples
    --------
    >>> from crimm.IO import write_psf
    >>> write_psf(model, 'system.psf')  # All waters->SOLV, ions->IONS

    To separate crystallographic from generated chains:

    >>> write_psf(model, 'system.psf', separate_crystal_segids=True)
    """
    writer = PSFWriter(
        extended=extended,
        xplor=xplor,
        separate_crystal_segids=separate_crystal_segids
    )
    writer.write(model, filepath, title)


def get_psf_str(
    model: Model,
    extended: bool = True,
    xplor: bool = True,
    title: str = "",
    separate_crystal_segids: bool = False
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
    separate_crystal_segids : bool, default False
        If True, crystallographic waters/ions get separate segids (CRWT/CION)
        from generated ones (SOLV/IONS). If False (default), all waters use
        SOLV and all ions use IONS regardless of source.

    Returns
    -------
    str
        PSF format string
    """
    writer = PSFWriter(
        extended=extended,
        xplor=xplor,
        separate_crystal_segids=separate_crystal_segids
    )
    return writer.get_psf_string(model, title)


def validate_psf(
    model: Model,
    strict: bool = False
) -> List[str]:
    """Validate model topology before writing PSF.

    Performs CHARMM-style validation checks that would normally occur
    during PSF generation. Since loading a pre-built PSF skips these
    checks in CHARMM, we must validate before writing.

    This function checks for:
    - Missing force field parameters (bonds, angles, dihedrals, etc.)
    - Atoms without topology definitions
    - Missing charges

    Parameters
    ----------
    model : Model
        The Model object to validate
    strict : bool, default False
        If True, raise ValueError for any issues.
        If False, issue warnings and return list of issues.

    Returns
    -------
    List[str]
        List of validation issues found (empty if valid)

    Raises
    ------
    ValueError
        If strict=True and validation issues are found

    Examples
    --------
    >>> issues = validate_psf(model)
    >>> if issues:
    ...     print(f"Found {len(issues)} issues")
    ...
    >>> # Or strict mode to halt on errors
    >>> validate_psf(model, strict=True)
    """
    writer = PSFWriter()
    return writer.validate_for_simulation(model, strict=strict)
