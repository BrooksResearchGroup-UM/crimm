import warnings

# Exceptions and Warnings
from Bio import BiopythonWarning
# Allowed Elements
from Bio.Data.IUPACData import atom_weights
from crimm import StructEntities as Entities

_ATOM_FORMAT_STRING = (
    "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%s%6.2f      %4s%2s%2s\n"
)

def _get_atom_line_with_parent_info(atom: Entities.Atom):
    """Return the parent info of the atom (PRIVATE). Atom must have a parent residue."""
    residue = atom.parent
    resname = residue.resname
    segid = residue.segid
    hetfield, resseq, icode = residue.id
    if (chain:=residue.parent) is not None:
        chain_id = chain.get_id()
    else:
        chain_id = '_'
    return _get_atom_line(
        atom, hetfield, segid, atom.get_serial_number(),
        resname, resseq, icode, chain_id
    )

def _get_orphan_atom_line(atom: Entities.Atom):
    """Return the orphan atom line (PRIVATE). Dummy residue and chain info will be filled."""
    resname = 'DUM'
    segid = ' '
    hetfield, resseq, icode = ' ', 1, ' '
    chain_id = '_'
    return _get_atom_line(
        atom, hetfield, segid, atom.get_serial_number(),
        resname, resseq, icode, chain_id
    )

def _get_atom_line(
    atom,
    hetfield,
    segid,
    atom_number,
    resname,
    resseq,
    icode,
    chain_id,
    charge="  ",
):
    """Return an ATOM PDB string (PRIVATE)."""
    if hetfield != " ":
        record_type = "HETATM"
    else:
        record_type = "ATOM  "

    # Atom properties

    # Check if the atom serial number is an integer
    # Not always the case for structures built from
    # mmCIF files.
    try:
        atom_number = int(atom_number)
    except ValueError as exc:
        raise ValueError(
            f"{atom_number!r} is not a number."
            "Atom serial numbers must be numerical"
            " If you are converting from an mmCIF"
            " structure, try using"
            " preserve_atom_numbering=False"
        ) from exc

    if atom_number > 99999:
        raise ValueError(
            f"Atom serial number ('{atom_number}') exceeds PDB format limit."
        )

    # Check if the element is valid, unknown (X), or blank
    if atom.element:
        element = atom.element.strip().upper()
        if element.capitalize() not in atom_weights and element != "X":
            raise ValueError(f"Unrecognised element {atom.element}")
        element = element.rjust(2)
    else:
        element = "  "

    # Format atom name
    # Pad if:
    #     - smaller than 4 characters
    # AND - is not C, N, O, S, H, F, P, ..., one letter elements
    # AND - first character is NOT numeric (funky hydrogen naming rules)
    name = atom.fullname.strip()
    if len(name) < 4 and name[:1].isalpha() and len(element.strip()) < 2:
        name = " " + name

    altloc = atom.altloc
    x, y, z = atom.coord

    # Write PDB format line
    bfactor = atom.bfactor
    try:
        occupancy = f"{atom.occupancy:6.2f}"
    except (TypeError, ValueError):
        if atom.occupancy is None:
            occupancy = " " * 6
            warnings.warn(
                f"Missing occupancy in atom {atom.full_id!r} written as blank",
                BiopythonWarning,
            )
        else:
            raise ValueError(
                f"Invalid occupancy value: {atom.occupancy!r}"
            ) from None

    args = (
        record_type,
        atom_number,
        name,
        altloc,
        resname,
        chain_id,
        resseq,
        icode,
        x,
        y,
        z,
        occupancy,
        bfactor,
        segid,
        element,
        charge,
    )
    return _ATOM_FORMAT_STRING % args

##TODO: Add support for TER and END records
##TODO: Add support for CONECT records
def get_pdb_str(entity, reset_serial=True, include_alt=False):
    """Return the PDB string of the entity."""
    if reset_serial:
        entity.reset_atom_serial_numbers(include_alt=include_alt)

    if entity.level == 'A':
        if entity.parent is None:
            return _get_orphan_atom_line(entity)
        return _get_atom_line_with_parent_info(entity)

    pdb_str = ''
    for atom in entity.get_atoms(include_alt=include_alt):
        pdb_str += _get_atom_line_with_parent_info(atom)
    return pdb_str