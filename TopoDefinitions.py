import warnings
from typing import List, Dict, Tuple
from collections import OrderedDict
import numpy as np
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from ICBuilder import recur_find_build_seq, find_coords_by_ic
from Atom import Atom

aa_3to1 = protein_letters_3to1_extended.copy()
aa_3to1.update({'HSE':'H', 'HSD':'H', 'HSP':'H'})

class AtomDefinition:
    def __init__(
        self, parent_def, name, atom_type, charge, mass,
        desc = None
    ):
        self.parent_def = parent_def
        self.name = name
        self.atom_type = atom_type
        self.is_donor = False
        self.is_acceptor = False
        self.charge = charge
        self.mass = mass
        self.desc = desc
        self.element = name[0]

    def __repr__(self):
        repr_str = f"<Atom Definition name={self.name} type={self.atom_type}>"
        return repr_str

    def create_new_atom(self, coords = None, serial_number = 0):
        """Create a new atom instance from the Atom definition. The default coordinates
        will be none if not specified."""
        
        return Atom(
            name = self.name, 
            coord=coords,
            bfactor=0.0, 
            occupancy=1.0, 
            altloc=' ',
            serial_number=serial_number,
            element=self.element,
            fullname=self.name
        )

## TODO: separate patch definition from residue definition
class ResidueDefinition:
    bond_order_dict = {'single':1, 'double':2, 'triple':3, 'aromatic':2}

    def __init__(
            self, file_source : str, resname : str, res_topo_dict: Dict
        ):
        self.file_source = file_source
        self.resname = resname
        self.is_modified = False
        self.is_patch : bool = None
        self.atom_groups = {}
        self.atom_dict = {}
        self.total_charge : float = None
        self.bonds : List= None
        self.impropers : List[Tuple] = None
        self.cmap : List[Tuple] = None
        self.H_donors = []
        self.H_acceptors = []
        self.desc : str = None
        self.ic = {}
        self.atom_lookup_dict : Dict = None
        self.standard_coord_dict = None
        self._standard_res = None
        self.load_topo_dict(res_topo_dict)
        self.assign_donor_acceptor()
        self.create_atom_lookup_dict()
        if not self.is_patch:
            self.build_standard_coord_from_ic_table()
            self._construct_standard_residue_from_ic()

    def __len__(self):
        """Return the number of atom definitions."""
        return len(self.atom_dict)

    def __repr__(self):
        if self.is_patch:
            def_type = "Patch"
        else:
            def_type = "Residue"
        if (code := aa_3to1.get(self.resname)) is not None:
            code_repr = f"code={code}"
        else:
            code_repr = ''
        return (
            f"<{def_type} Definition name={self.resname} "
            f"{code_repr} atoms={len(self)}>"
        )

    def __getitem__(self, id):
        """Return the child with given id."""
        return self.atom_dict[id]

    def __contains__(self, id):
        """Check if there is an atom element with the given atom name."""
        return id in self.atom_dict

    def __iter__(self):
        """Iterate over atom definitions."""
        yield from self.atom_dict.values()

    def get_atom_defs(self):
        return list(self.atom_dict.values())

    def load_topo_dict(self, res_topo_dict):
        for key, val in res_topo_dict.items():
            if key == 'atoms':
                self.process_atom_groups(val)
            else:
                setattr(self, key, val)

    def process_atom_groups(self, atom_dict):
        for i, group_def in atom_dict.items():
            cur_group = []
            for atom_name, atom_info in group_def.items():
                atom_def = AtomDefinition(
                    self, atom_name, **atom_info
                )
                cur_group.append(atom_name)
                self.atom_dict[atom_name] = atom_def
            self.atom_groups[i] = cur_group

    def assign_donor_acceptor(self):
        for hydrogen_name, donor_name in self.H_donors:
            atom_def = self.atom_dict[donor_name]
            atom_def.is_donor = True

        for entry in self.H_acceptors:
            if len(entry) == 2:
                acceptor_name, neighbor_name = entry
            else:
                acceptor_name = entry[0]
            atom_def = self.atom_dict[acceptor_name]
            atom_def.is_acceptor = True

    def create_atom_lookup_dict(self):
        self.atom_lookup_dict = {}
        for i, j, k, l in self.ic.keys():
            for atom_name in (i, l):
                if atom_name not in self.atom_lookup_dict:
                    self.atom_lookup_dict[atom_name] = []
                self.atom_lookup_dict[atom_name].append((i, j, k, l))

    def build_standard_coord_from_ic_table(self):
        """Build Cartesian coordinates from ic table loaded from residue topology
        definitions. The initial coordinates will be generated by placing 'N' on
        the origin and '-C' along the x-axis."""
        if self.resname in aa_3to1:
            base_ic1 = self.ic[('-C', 'N', 'CA', 'C')]
            base_ic2 = self.ic[('N', 'CA', 'C', '+N')]
        elif self.resname in nucleic_letters_3to1_extended:
            warnings.warn(
                'Nucleic Acids Topology has not been implemented!'
                'Standard coord build from IC table is skipped.'
            )
            return 
        else:
            warnings.warn(
                f'Unknown Residue Type {self.resname}! '
                'Standard coord build from IC table is skipped.'
            )
            return 
        
        r_CN = base_ic1['R(I-J)']
        t_CNCA = np.deg2rad(base_ic1['T(I-J-K)'])
        r_NCA = base_ic2['R(I-J)']

        x_N = np.cos(t_CNCA)*r_NCA
        y_N = np.sin(t_CNCA)*r_NCA

        self.standard_coord_dict = {
            '-C': np.array([r_CN,0,0], dtype=float),
            'N': np.array([0, 0, 0], dtype=float),
            'CA':np.array([x_N, y_N, 0])
        }

        lookup_dict = self.atom_lookup_dict
        all_atoms = [k for k in lookup_dict.keys() if k not in ('-C','CA','N')]
        running_dict = OrderedDict({k: lookup_dict[k] for k in all_atoms})

        build_seq = recur_find_build_seq(
            running_dict,
            all_atoms,
            build_seq = [],
            exclude_list = []
        )
        
        find_coords_by_ic(build_seq, self.ic, self.standard_coord_dict)
        
    def _construct_standard_residue_from_ic(self):
        """Create a standard residue from the internal coordinate definitions
        """
        if self.standard_coord_dict is None:
            warnings.warn(
                'No standard coordinates available. '
                f'Skipped construction of new residue: {self.resname}'
            )
            return
        from Residue import Residue
        self._standard_res = Residue((' ', 0, ' '), self.resname, segid = " ")
        for i, (atom_name, coords) in enumerate(self.standard_coord_dict.items()):
            if atom_name.startswith('-') or atom_name.startswith('+'):
                continue
            new_atom = self[atom_name].create_new_atom(
                coords=coords, serial_number=i+1
            )
            self._standard_res.add(new_atom)
    
    def create_residue(self, resseq = 0, icode = " ", segid = " "):
        """Create a new residue from the internal coordinate definitions
        
        Args:
            resseq: int, residue sequence number for the new residue. Default to 0
            icode: str, insertion code
            segid: str, segment id the new residue belongs to. Default to " " (empty)
            
        Return:
            Residue object with atoms whose names and coordinates are filled
        """
        if self._standard_res is None:
            warnings.warn(
                'No standard coordinates available. '
                f'Skipped construction of new residue: {self.resname}'
            )
            return
        new_res = self._standard_res.copy()
        new_res.id = (" ", int(resseq), str(icode))
        new_res.segid = str(segid)
        return new_res