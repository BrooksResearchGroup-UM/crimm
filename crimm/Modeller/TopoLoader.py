import warnings
from types import MappingProxyType
import pickle
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_1to3
from crimm import StructEntities as Entities
from crimm.IO.RTFParser import RTFParser

class TopologyLoader:
    """Class for loading topology definition to the residue and find any missing atoms."""
    def __init__(self, file_path=None, data_dict_path=None):
        self.rtf_version = None
        self.res_defs = None
        self.residues = None
        self.patches = None
        if data_dict_path is not None:
            with open(data_dict_path, 'rb') as f:
                data_dict = pickle.load(f)
                self.load_data_dict(data_dict)
        elif file_path is not None:
            rtf = RTFParser(file_path=file_path)
            self.load_data_dict(rtf.topo_dict, rtf.rtf_version)

    def load_data_dict(self, topo_data_dict, rtf_version=None):
        self.rtf_version = rtf_version
        topo_def_dict = {}
        for resname, res_topo_dict in topo_data_dict.items():
            topo_def_dict[resname] = Entities.ResidueDefinition(
                self.rtf_version, resname, res_topo_dict
            )

        if 'HIS' not in topo_def_dict and 'HSD' in topo_def_dict:
            # Map all histidines HIS to HSD
            topo_def_dict['HIS'] = topo_def_dict['HSD']
        
        self.patches = tuple(
            res_def for res_def in topo_def_dict.values() if res_def.is_patch
        )
        self.residues = tuple(
            res_def for res_def in topo_def_dict.values() if not res_def.is_patch
        )
        self.res_defs = MappingProxyType(topo_def_dict)

    def __repr__(self):
        return (
            f'<TopologyLoader Ver={self.rtf_version} '
            f'Contains {len(self.residues)} RESIDUE and '
            f'{len(self.patches)} PATCH definitions>'
        )

    def __getitem__(self, __key: 'str'):
        return self.res_defs[__key]
    
    def generate_residue_topology(
            self, residue: Entities.Residue, coerce: bool = False, QUIET=False
        ):
        """Load topology definition into the residue and find any missing atoms.
        Argument:
            residue: the Residue object whose topology is to be defined
            coerce: if True, try to coerce the modified residue name to the canonical name
                    and load the canonical residue topology definition
            QUIET: if True, suppress all warnings

        Return:
            True if the residue is defined, False otherwise"""
        
        if isinstance(residue, Entities.DisorderedResidue):
            return self.generate_residue_topology(
                residue.selected_child, coerce=coerce, QUIET=QUIET
            )

        if residue.resname not in self.res_defs:
            if not QUIET:
                warnings.warn(
                    f"Residue {residue.resname} is not defined in the topology file!"
                )
            if residue.topo_definition is not None:
                return True
            if not coerce:
                return False
            return self.coerce_resname(residue, QUIET=QUIET)

        if residue.topo_definition is not None and not QUIET:
            warnings.warn("Topology definition already exists! Overwriting...")
        
        res_definition = self.res_defs[residue.resname]
        residue.topo_definition = res_definition
        residue.total_charge = res_definition.total_charge
        residue.impropers = res_definition.impropers
        residue.cmap = res_definition.cmap
        residue.H_donors = res_definition.H_donors
        residue.H_acceptors = res_definition.H_acceptors
        residue.param_desc = res_definition.desc
        self._load_atom_groups(residue)
        residue.undefined_atoms = []
        for atom in residue:
            if atom.name not in res_definition:
                residue.undefined_atoms.append(atom)
                if not QUIET:
                    warnings.warn(
                        f"Atom {atom.name} is not defined in the topology file!"
                    )
        return True
    
    @staticmethod
    def _create_missing_atom(residue: Entities.Residue, atom_name: str):
        """Create and separate missing heavy atoms and missing hydrogen atom by atom name"""
        missing_atom = residue.topo_definition[atom_name].create_new_atom()
        missing_atom.set_parent(residue)
        if atom_name.startswith('H'):
            residue.missing_hydrogens[atom_name] = missing_atom
        else:
            residue.missing_atoms[atom_name] = missing_atom

    @staticmethod
    def _load_group_atom_topo_definition(
            residue: Entities.Residue, atom_name_list
        ) -> tuple:
        """Load topology definition to each atom in the residue and find any missing 
        atoms. 
        Argument:
            atom_name_list: list of atom names that are in the same group
        Return:
            atom_group: the Atom object in the group
        """
        atom_group = []
        for atom_name in atom_name_list:
            if atom_name not in residue:
                TopologyLoader._create_missing_atom(residue, atom_name)
                continue
            cur_atom = residue[atom_name]
            cur_atom.topo_definition = residue.topo_definition[atom_name]
            atom_group.append(cur_atom)
        return tuple(atom_group)

    @staticmethod
    def _load_atom_groups(residue: Entities.Residue):
        residue.atom_groups = []
        residue.missing_atoms, residue.missing_hydrogens = {},{}
        atom_group_lists = residue.topo_definition.atom_groups
        for atom_names in atom_group_lists:
            cur_group = TopologyLoader._load_group_atom_topo_definition(
                residue, atom_names
            )
            residue.atom_groups.append(cur_group)

    def coerce_resname(self, residue: Entities.Residue, QUIET=False):
        """Coerce the name of modified residue to reconstruct it as the canonical 
        one that it is based on. 

        Argument:
            residue: the residue whose name is to be coerced
        Return:
            True if the residue is a known modified residue and is successfully coerced
            False otherwise"""
        ## TODO: add examples to the docstring
        if residue.resname not in protein_letters_3to1_extended:
            return False
        # if the residue is a known modified residue,
        # coerce the residue name to reconstruct it as the
        # canonical one
        code = protein_letters_3to1_extended[residue.resname]
        new_resname = protein_letters_1to3[code]
        if not QUIET:
            warnings.warn(
                f'Coerced Residue {residue.resname} to {new_resname}'
            )
        residue.resname = new_resname
        _, resseq, icode = residue.id
        residue.id = (" ", resseq, icode)
        if hasattr(residue.parent, "reported_res"):
            # if the chain has reported_res attribute, update it
            # to avoid generating new gaps due to mismatch resnames
            residue.parent.reported_res[resseq-1] = (resseq, new_resname)
        return True

    ## TODO: get Bond, Angle, Dihedral, Improper, Cmap, and Nonbonded from the topology
    def generate_chain_topology(
            self, chain: Entities.Chain, coerce: bool = False, QUIET=False
        ):
        """Load topology definition into the chain and find any missing atoms.
        Argument:
            chain: the Chain object whose topology is to be defined
            coerce: if True, try to coerce the modified residue name to the canonical name
                    and load the canonical residue topology definition
            QUIET: if True, suppress all warnings
        """
        chain.undefined_res = []
        for residue in chain:
            is_defined = self.generate_residue_topology(residue, coerce=coerce, QUIET=QUIET)
            if not is_defined:
                chain.undefined_res.append(residue)
        if (n_undefined:=len(chain.undefined_res)) >0 and not QUIET:
            warnings.warn(
                f"{n_undefined} residues are not defined in the chain!"
            )