"""
Module for writing CHARMM CARD coordinate files (CRD).

This module provides functions to write coordinate data from crimm entities
to CHARMM CRD format files. The CRD format stores atomic coordinates along
with residue and segment information.

Format specification (from CHARMM source io/coorio.F90):
- Extended format: 2I10, 2X,A8, 2X,A8, 3F20.10, 2X,A8, 2X,A8, F20.10
- Standard format: 2I5, 1X,A4, 1X,A4, 3F10.5, 1X,A4, 1X,A4, F10.5
"""

from typing import Union, List, Optional
from crimm.StructEntities import Model, Structure, Chain, Residue, Atom

# Extended format (EXT) - for systems with >99999 atoms
_EXT_HEADER_FORMAT = "{:>10d}  EXT\n"
_EXT_ATOM_FORMAT = (
    "{serial:>10d}{resno:>10d}  {resname:<8s}  {atomname:<8s}"
    "{x:>20.10f}{y:>20.10f}{z:>20.10f}  {segid:<8s}  {resid:<8s}{weight:>20.10f}\n"
)

# Standard format - for smaller systems
_STD_HEADER_FORMAT = "{:>5d}\n"
_STD_ATOM_FORMAT = (
    "{serial:>5d}{resno:>5d} {resname:<4s} {atomname:<4s}"
    "{x:>10.5f}{y:>10.5f}{z:>10.5f} {segid:<4s} {resid:<4s}{weight:>10.5f}\n"
)


def _get_atoms_from_entity(entity, include_lonepairs: bool = True) -> List[Atom]:
    """Get all atoms from an entity, optionally including lone pairs.

    Parameters
    ----------
    entity : Model, Structure, Chain, or Residue
        The entity to extract atoms from
    include_lonepairs : bool
        Whether to include lone pair pseudo-atoms for CGENFF ligands

    Returns
    -------
    List[Atom]
        List of atoms in order
    """
    atoms = []

    if hasattr(entity, 'get_atoms'):
        atoms.extend(list(entity.get_atoms()))
    elif isinstance(entity, Atom):
        atoms.append(entity)

    # Include lone pairs if present and requested
    if include_lonepairs:
        if hasattr(entity, 'get_residues'):
            for residue in entity.get_residues():
                if hasattr(residue, 'lone_pair_dict') and residue.lone_pair_dict:
                    atoms.extend(residue.lone_pair_dict.values())
        elif hasattr(entity, 'lone_pair_dict') and entity.lone_pair_dict:
            atoms.extend(entity.lone_pair_dict.values())

    return atoms


def _format_title_lines(title: str) -> str:
    """Format title lines with asterisk prefix.

    Parameters
    ----------
    title : str
        Title text (can be multiline)

    Returns
    -------
    str
        Formatted title lines with asterisk prefix
    """
    lines = []
    if title:
        for line in title.strip().split('\n'):
            lines.append(f"* {line}\n")
    lines.append("*\n")
    return "".join(lines)


def get_crd_str(
    entity: Union[Model, Structure, Chain, Residue],
    extended: bool = True,
    title: str = "",
    include_lonepairs: bool = True,
    reset_serial: bool = True
) -> str:
    """Get CHARMM CRD format string from a crimm entity.

    Parameters
    ----------
    entity : Model, Structure, Chain, or Residue
        The entity to write coordinates from
    extended : bool, default True
        Use extended format (I10/A8/F20.10) for large systems
    title : str, default ""
        Title line(s) to include in the header
    include_lonepairs : bool, default True
        Include lone pair pseudo-atoms for CGENFF ligands
    reset_serial : bool, default True
        Reset atom serial numbers starting from 1

    Returns
    -------
    str
        CRD format string
    """
    # Get atoms
    atoms = _get_atoms_from_entity(entity, include_lonepairs)
    natom = len(atoms)

    if natom == 0:
        raise ValueError("No atoms found in entity")

    # Select format based on extended flag and atom count
    if extended or natom >= 100000:
        header_fmt = _EXT_HEADER_FORMAT
        atom_fmt = _EXT_ATOM_FORMAT
        str_width = 8
    else:
        header_fmt = _STD_HEADER_FORMAT
        atom_fmt = _STD_ATOM_FORMAT
        str_width = 4

    # Build output
    lines = []

    # Title section
    lines.append(_format_title_lines(title))

    # Atom count header
    lines.append(header_fmt.format(natom))

    # Build global residue numbering map
    # RESNO in CRD format must be sequential across the entire system (1, 2, 3...)
    # while RESID is the segment-local residue identifier
    resno_map = {}  # Maps residue object to global resno
    global_resno = 1
    last_residue = None

    for atom in atoms:
        residue = atom.parent
        if residue is not last_residue and residue is not None:
            if residue not in resno_map:
                resno_map[residue] = global_resno
                global_resno += 1
            last_residue = residue

    # Atom lines
    for idx, atom in enumerate(atoms, start=1):
        serial = idx if reset_serial else (atom.serial_number or idx)

        # Get parent residue info
        residue = atom.parent
        if residue is not None:
            resname = residue.resname[:str_width]
            segid = (residue.segid or "")[:str_width]
            # Use global sequential RESNO for CHARMM compatibility
            resno = resno_map.get(residue, 1)
            # RESID is the segment-local residue ID as string
            resid = str(residue.id[1])[:str_width]
        else:
            resname = "UNK"[:str_width]
            segid = ""[:str_width]
            resno = 1
            resid = "1"[:str_width]

        atomname = atom.name[:str_width]

        # Get coordinates
        if atom.coord is None:
            raise ValueError(f"Atom {atom.name} has no coordinates")
        x, y, z = atom.coord

        # Weight/temperature factor
        weight = atom.bfactor if atom.bfactor is not None else 0.0

        lines.append(atom_fmt.format(
            serial=serial,
            resno=resno,
            resname=resname,
            atomname=atomname,
            x=x,
            y=y,
            z=z,
            segid=segid,
            resid=resid,
            weight=weight
        ))

    return "".join(lines)


def write_crd(
    entity: Union[Model, Structure, Chain, Residue],
    filepath: str,
    extended: bool = True,
    title: str = "",
    include_lonepairs: bool = True,
    reset_serial: bool = True
) -> None:
    """Write CHARMM CRD format file from a crimm entity.

    Parameters
    ----------
    entity : Model, Structure, Chain, or Residue
        The entity to write coordinates from
    filepath : str
        Path to output file
    extended : bool, default True
        Use extended format (I10/A8/F20.10) for large systems
    title : str, default ""
        Title line(s) to include in the header
    include_lonepairs : bool, default True
        Include lone pair pseudo-atoms for CGENFF ligands
    reset_serial : bool, default True
        Reset atom serial numbers starting from 1
    """
    crd_str = get_crd_str(
        entity,
        extended=extended,
        title=title,
        include_lonepairs=include_lonepairs,
        reset_serial=reset_serial
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(crd_str)
