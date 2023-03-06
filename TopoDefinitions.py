from typing import List, Dict, Tuple
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

    def __repr__(self):
        repr_str = f"<Atom Definition name={self.name} type={self.atom_type}>"
        return repr_str

class ResidueDefinition:
    bond_order_dict = {'single':1, 'double':2, 'triple':3, 'aromatic':2}

    def __init__(self, file_source : str, resname : str, res_topo_dict: Dict):
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
        self.ic : List[Dict] = None
        self.atom_lookup_dict : Dict = None
        self.load_topo_dict(res_topo_dict)
        self.assign_donor_acceptor()
        self.create_atom_lookup_dict()

    def __len__(self):
        """Return the number of atom definitions."""
        return len(self.atom_dict)

    def __repr__(self):
        return f"<Residue Definition name={self.resname} atoms={len(self)}>"

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
