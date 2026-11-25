import io
import pathlib
import warnings

from Bio.Data.PDBData import (
    nucleic_letters_3to1,
    nucleic_letters_3to1_extended,
    protein_letters_3to1,
    protein_letters_3to1_extended,
)
from Bio.PDB.PDBParser import PDBParser as _PDBParser

from crimm.Data.components_dict import CHARMM_PDB_ION_NAMES
from crimm.IO.StructureBuilder import StructureBuilder
from crimm.StructEntities.Chain import Chain, Heterogens, Ion, PolymerChain, Solvent
from crimm.Utils.StructureUtils import index_to_letters

protein_letters_3to1.update({"HSD": "H", "HSE": "H", "HSP": "H"})
_WATER_RESNAMES = ("HOH", "TIP3")


def check_chain_type(residues, resname_lookup, extended_lookup=None):
    for res in residues:
        if res.resname not in resname_lookup:
            # Check if the residue is a heterogen and if it is in the extended
            # lookup table, heterogen flag will be changed to 'H_{resname}'
            if extended_lookup and (res.resname in extended_lookup):
                hetflag, resseq, icode = res.get_id()
                hetflag = f"H_{res.resname}"
                res.id = (hetflag, resseq, icode)
                warnings.warn(f"Heterogen flag for {res.get_id()} corrected.")
                continue
            return False
    return True


def _wide_resnames(lines):
    """Residue names that fill columns 18-21 (TIP3, POPC), as CHARMM and VMD write them.

    Biopython reads the residue name from columns 18-20 only. Takes the file's lines
    and returns a dict keyed by (model index, chain ID, resseq, icode, the
    three-character name Biopython reads, is HETATM), following Biopython's model
    numbering. Biopython tells an ATOM residue from a
    HETATM residue with the same number apart by its hetero flag, so the key does too.
    """
    names = {}
    model_index = -1
    model_open = False
    for line in lines:
        record_type = line[0:6]
        if record_type == "MODEL ":
            model_index += 1
            model_open = True
        elif record_type == "ENDMDL":
            model_open = False
        elif record_type in ("END   ", "CONECT"):
            break
        elif record_type in ("ATOM  ", "HETATM"):
            if not model_open:
                model_index += 1
                model_open = True
            if len(line) < 27 or line[20] == " ":
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            key = (
                model_index,
                line[21],
                resseq,
                line[26],
                line[17:20].strip(),
                record_type == "HETATM",
            )
            names[key] = line[17:21].strip()
    return names


def _restore_wide_resnames(structure, lines):
    """Give residues their full 4-character names from columns 18-21 of the file."""
    names = _wide_resnames(lines)
    if not names:
        return
    for model_index, model in enumerate(structure):
        for chain in model:
            for res in list(chain):
                hetflag, resseq, icode = res.get_id()
                key = (
                    model_index,
                    chain.id,
                    resseq,
                    icode,
                    res.resname,
                    hetflag != " ",
                )
                resname = names.get(key)
                if resname is None or res.is_disordered() == 2:
                    continue
                if resname[:3] != res.resname:
                    continue
                res.resname = resname
                if not hetflag.startswith("H_"):
                    continue
                # A water read as H_TIP becomes W, unless a residue with that id (an HOH
                # of the same number) is already in the chain; then it keeps H_TIP3.
                for new_flag in ("W", f"H_{resname}"):
                    if new_flag == "W" and resname not in _WATER_RESNAMES:
                        continue
                    new_id = (new_flag, resseq, icode)
                    if not chain.has_id(new_id):
                        res.id = new_id
                        break


def separate_solvent(residues):
    het = []
    water = []
    for res in residues:
        if res.resname in _WATER_RESNAMES:
            water.append(res)
        else:
            het.append(res)
    return het, water


def find_chain_type(chain):
    residues = []
    het_residues = []
    for res in chain:
        hetflag, resseq, icode = res.get_id()
        if hetflag == " ":
            residues.append(res)
        else:
            het_residues.append(res)
    if residues:
        if check_chain_type(
            residues, protein_letters_3to1, protein_letters_3to1_extended
        ):
            return [("Polypeptide(L)", chain.id, residues)]
        elif check_chain_type(
            residues, nucleic_letters_3to1, nucleic_letters_3to1_extended
        ):
            return [("Polyribonucleotide", chain.id, residues)]
        elif check_chain_type(residues, CHARMM_PDB_ION_NAMES, None):
            return [("Ion", chain.id, residues)]
        else:
            warnings.warn(f"Chain type cannot be determined for {chain.get_full_id()}")
            return [("Chain", chain.id, residues)]
    else:
        het, water = separate_solvent(het_residues)
        return [("Heterogens", chain.id, het), ("Solvent", chain.id, water)]


def convert_chains(chains):
    organized_residues = []
    for chain in chains:
        organized_residues.extend(find_chain_type(chain))

    chain_sort_dict = {
        "Polypeptide(L)": 0,
        "Polyribonucleotide": 1,
        "Chain": 2,  # 'Chain' is a catch-all for 'unknown
        "Heterogens": 3,
        "Ion": 4,
        "Solvent": 5,
    }

    new_chains = []
    for chain_type, auth_chain_id, residues in sorted(
        organized_residues, key=lambda x: chain_sort_dict[x[0]]
    ):
        if len(residues) == 0:
            continue
        new_chain_id = index_to_letters(len(new_chains))
        if chain_type in ("Polypeptide(L)", "Polyribonucleotide"):
            new_chain = PolymerChain(
                chain_id=new_chain_id,
                entity_id=None,
                author_chain_id=auth_chain_id,
                chain_type=chain_type,
                known_sequence="",
                canon_sequence="",
                reported_res=[],
                reported_missing_res=[],
            )
        elif chain_type == "Chain":
            new_chain = Chain(
                chain_id=new_chain_id,
            )
        elif chain_type == "Heterogens":
            new_chain = Heterogens(new_chain_id)
        elif chain_type == "Ion":
            new_chain = Ion(new_chain_id)
        else:
            new_chain = Solvent(new_chain_id)
        for res in residues:
            res.detach_parent()
            if res.resname == "ILE" and "CD" in res:
                # ILE could have a CD atom (CHARMM convention)
                # that is not in the standard PDB format
                atom = res["CD"]
                res.detach_child("CD")
                atom.name = "CD1"
                atom.id = "CD1"
                atom.fullname = " CD1"
                res.add(atom)
            new_chain.add(res)
        new_chains.append(new_chain)
    return new_chains


class PDBParser(_PDBParser):
    """PDBParser that returns a Structure with determined chain types."""

    def __init__(
        self,
        first_model_only=True,
        include_solvent=True,
        strict_parser=True,
        get_header=False,
        QUIET=False,
    ):
        structure_builder = StructureBuilder()
        self.first_model_only = first_model_only
        self.include_solvent = include_solvent
        PERMISSIVE = not strict_parser
        super().__init__(PERMISSIVE, get_header, structure_builder, QUIET, is_pqr=False)

    def _get_structure(self, filepath, structure_id=None):
        """Return the structure contained in file."""
        if structure_id is None:
            structure_id = pathlib.Path(filepath).stem
        if hasattr(filepath, "read"):
            # An open handle (the SwissModel fetchers pass an io.StringIO) can be read
            # only once, so parse and scan the same text.
            text = filepath.read()
            structure = super().get_structure(structure_id, io.StringIO(text))
            lines = text.splitlines()
        else:
            structure = super().get_structure(structure_id, filepath)
            with open(filepath) as handle:
                lines = handle.readlines()
        _restore_wide_resnames(structure, lines)
        if self.first_model_only:
            structure.child_list = structure.child_list[:1]
            structure.child_dict = {c.id: c for c in structure.child_list}
        for model in structure:
            new_chains = convert_chains(model.child_list)
            if not self.include_solvent:
                new_chains = [c for c in new_chains if c.chain_type != "Solvent"]
            for chain in new_chains:
                chain.set_parent(model)
            model.child_list = new_chains
            model.child_dict = {c.id: c for c in new_chains}
        return structure

    def get_structure(self, filepath, structure_id=None):
        """Return the structure contained in file."""
        if self.QUIET:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._get_structure(filepath, structure_id)
        return self._get_structure(filepath, structure_id)
