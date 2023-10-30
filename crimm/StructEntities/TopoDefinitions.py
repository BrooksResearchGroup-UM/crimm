import warnings
from typing import List, Dict, Tuple
from collections import OrderedDict
import numpy as np
from Bio.Data.PDBData import protein_letters_3to1_extended
import crimm.StructEntities as Entities
from crimm.Modeller.TopoFixer import recur_find_build_seq, find_coords_by_ic

class AtomDefinition:
    """Atom definition class. This class is used to define the atom type and
    other properties of an atom. It is also used to create new atom instances."""
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
        return Entities.Atom(
            name = self.name,
            coord=coords,
            bfactor=0.0,
            occupancy=1.0,
            altloc=' ',
            serial_number=serial_number,
            element=self.element,
            fullname=self.name,
            topo_definition=self
        )

class ResidueDefinition:
    aa_3to1 = protein_letters_3to1_extended.copy()
    aa_3to1.update({'HSE':'H', 'HSD':'H', 'HSP':'H'})
    na_3to1 = {
        'GUA':'G', 'ADE':'A', 'CYT':'C', 'THY':'T', 'URA':'U'
    }
    na_1to3 = {
        'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'T': 'THY', 'U': 'URA',
    }
    
    bond_order_dict = {'single':1, 'double':2, 'triple':3, 'aromatic':2}

    def __init__(
            self, file_source : str, resname : str, res_topo_dict: Dict
        ):
        self.file_source = file_source
        self.resname = resname
        self.is_modified = False
        self.is_patch : bool = None
        self.atom_groups : List = None
        self.atom_dict = {}
        self.removed_atom_dict = {}
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
        self.patch_with = None

    def __len__(self):
        """Return the number of atom definitions."""
        return len(self.atom_dict)

    def __repr__(self):
        if self.patch_with:
            patched = f" Patched with {self.patch_with}"
        else:
            patched = ''
        if (code := self.aa_3to1.get(self.resname)) is not None:
            code_repr = f"code={code}"
        else:
            code_repr = ''
        return (
            f"<Residue Definition name={self.resname} "
            f"{code_repr} atoms={len(self)}{patched}>"
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
        self.atom_groups = []
        for group_def in atom_dict.values():
            cur_group = []
            for atom_name, atom_info in group_def.items():
                atom_def = AtomDefinition(
                    self, atom_name, **atom_info
                )
                cur_group.append(atom_name)
                self.atom_dict[atom_name] = atom_def
            self.atom_groups.append(tuple(cur_group))

    def assign_donor_acceptor(self):
        """Assign donor and acceptor properties to atoms."""
        for hydrogen_name, donor_name in self.H_donors:
            atom_def = self.atom_dict[donor_name]
            atom_def.is_donor = True

        for entry in self.H_acceptors:
            if len(entry) == 2:
                acceptor_name, neighbor_name = entry
            else:
                acceptor_name = entry[0]
            if acceptor_name not in self.atom_dict:
                if self.is_patch:
                    continue
                raise ValueError(
                    f"Atom {acceptor_name} not found in residue {self.resname}"
                )
            atom_def = self.atom_dict[acceptor_name]
            atom_def.is_acceptor = True

    def create_atom_lookup_dict(self):
        """Create a dictionary that maps atom names to the corresponding
        internal coordinates, by which the atom can be built."""
        self.atom_lookup_dict = {}
        atom_lookup_entries = []
        for (i, j, k, l), ic in self.ic.items():
            is_improper = int(ic['improper'])
            for atom_name in (i, l):
                if atom_name == 'BLNK':
                    continue
                atom_lookup_entries.append((is_improper, atom_name, (i, j, k, l)))
        # we need to sort the entries so that the non-improper entries are first
        for is_improper, atom_name, ic_key in sorted(atom_lookup_entries):
            if atom_name not in self.atom_lookup_dict:
                self.atom_lookup_dict[(atom_name)] = []
            self.atom_lookup_dict[atom_name].append(ic_key)

    def _is_ic_defined(self):
        """Check if the parameters for internal coordinates table are defined 
        for the residue."""
        if not self.ic:
            return False
        for key, ic_entries in self.ic.items():
            values = list(ic_entries.values())
            if 'BLNK' in key:
                if key.index('BLNK') == 0:
                    values = values[-2:]
                else:
                    values = values[:2]
            for value in values:
                if value is None:
                    return False
        return True
    
    def _create_init_atom_coords(self):
        """Find the atoms that can be used to build the initial coordinates for 
        residue with ic table filled."""
        for (a, b, c, d), base_ic1 in self.ic.items():
            if not base_ic1['improper']:
                break

        init_atoms = (a, b, c)
        r_ab = base_ic1.get('R(I-J)') #or base_ic1.get('R(I-K)')
        rad_abc = base_ic1.get('T(I-J-K)') #or base_ic1.get('T(I-K-J)')
        t_abc = np.deg2rad(rad_abc)
        r_bc = None
        for (i, j, k, l), ic_entry in self.ic.items():
            is_improper = ic_entry['improper']
            if ((i == b and j == c) or (j == b and i == c)) and not is_improper:
                r_bc = ic_entry['R(I-J)']
                break
            elif ((i == b and k == c) or (k == b and i == c)) and is_improper:
                r_bc = ic_entry['R(I-K)']
                break
        if r_bc is None:
            raise ValueError(
                f'Cannot find the bond between {(b,c)} for {self.resname}!'
            )

        x_N = np.cos(t_abc)*r_bc
        y_N = np.sin(t_abc)*r_bc

        init_coord_dict = {
            init_atoms[0]: np.array([r_ab,0,0], dtype=float),
            init_atoms[1]: np.array([0, 0, 0], dtype=float),
            init_atoms[2]: np.array([x_N, y_N, 0], dtype=float)
        }
        return init_coord_dict
    
    def build_standard_coord_from_ic_table(self, init_coord_dict=None):
        """Build Cartesian coordinates from ic table loaded from residue topology
        definitions. If no init_coord_dict provided, the initial coordinates will 
        be generated by placing 'N' on the origin and '-C' along the x-axis."""
        if not self._is_ic_defined():
            warnings.warn(
                f'IC table for {self.resname} is not fully defined! '
                'Standard coord build from IC table is skipped.'
            )
            return

        if init_coord_dict is None:
            standard_coord_dict = self._create_init_atom_coords()
        else:
            standard_coord_dict = init_coord_dict

        lookup_dict = self.atom_lookup_dict
 
        all_atoms = [k for k in lookup_dict.keys() if k not in standard_coord_dict]
        running_dict = OrderedDict({k: lookup_dict[k] for k in all_atoms})

        build_seq = recur_find_build_seq(
            running_dict,
            all_atoms,
            build_seq = [],
            exclude_list = []
        )

        find_coords_by_ic(build_seq, self.ic, standard_coord_dict)
        return standard_coord_dict

    def _construct_standard_residue_from_ic(self):
        """Create a standard residue from the internal coordinate definitions
        """
        if self.standard_coord_dict is None:
            self.standard_coord_dict = self.build_standard_coord_from_ic_table()
            if self.standard_coord_dict is None:
                warnings.warn(
                    'No standard coordinates available. '
                    f'Skipped construction of new residue: {self.resname}'
                )
                return

        self._standard_res = self.create_residue_from_coord_dict(
            self.standard_coord_dict
        )

    def create_residue_from_coord_dict(
            self, coord_dict, resseq = 0, icode = " ", segid = " "
        ):
        """Create a new residue from the atom coordinate dictionary.
        
        Args:
            coord_dict: dict, atom name: coordinates
            resseq: int, residue sequence number for the new residue. Default to 0
            icode: str, insertion code
            segid: str, segment id the new residue belongs to. Default to " " (empty)
            
        Return:
            Residue object with atoms whose names and coordinates are filled
        """
        
        if self.resname in self.na_3to1:
            resname = self.na_3to1[self.resname]
        else:
            resname = self.resname
        
        new_res = Entities.Residue(
            (' ', int(resseq), str(icode)),
            resname,
            segid = segid
        )
        for i, (atom_name, coords) in enumerate(coord_dict.items()):
            if atom_name.startswith('-') or atom_name.startswith('+'):
                continue
            new_atom = self[atom_name].create_new_atom(
                coords=coords, serial_number=i+1
            )
            new_res.add(new_atom)

        return new_res
    
    def create_residue(self, resseq = 0, icode = " ", segid = " "):
        """Create a new residue from the internal coordinate definitions
        
        Args:
            resseq: int, residue sequence number for the new residue. Default to 0
            icode: str, insertion code
            segid: str, segment id the new residue belongs to. Default to " " (empty)
            
        Return:
            Residue object with atoms whose names and coordinates are filled
        """
        if not self._is_ic_defined():
            warnings.warn(
                f'IC table for {self.resname} is not fully defined! '
                'Standard coord build from IC table is skipped.'
            )
            return
        
        if self._standard_res is None:
            self._construct_standard_residue_from_ic()
            if self._standard_res is None:
                return

        new_res = self._standard_res.copy()
        new_res.id = (" ", int(resseq), str(icode))
        new_res.segid = str(segid)
        return new_res


class PatchDefinition(ResidueDefinition):
    """Class Object for patch definition"""
    def __init__(self, file_source : str, resname : str, res_topo_dict: Dict):
        self.delete = None
        super().__init__(file_source, resname, res_topo_dict)
        
    def build_standard_coord_from_ic_table(self):
        raise NotImplementedError
    
    def _construct_standard_residue_from_ic(self):
        raise NotImplementedError
    
    def create_residue(self, resseq = 0, icode = " ", segid = " "):
        raise NotImplementedError

    def __repr__(self):
        return (
            f"<Patch Definition name={self.resname} "
            f"atoms={len(self)}>"
        )
        
