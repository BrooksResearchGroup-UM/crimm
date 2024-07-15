from Bio.Seq import Seq
from Bio.Data.PDBData import protein_letters_1to3, protein_letters_3to1
from crimm.Modeller import TopologyGenerator, ParameterLoader, ResidueTopologySet
from crimm.StructEntities import PolymerChain, ResidueDefinition

class SeqChainGenerator:
    """Generate a polymer chain from a sequence string"""
    def __init__(self) -> None:
        self.sequence = None
        self.chain_type = None
        self.chain_id = None
        self.topo_definitions = None
        self.params = None
        self._1to3 = None
        self._3to1 = None
        self._report_res = None
        self.resnames = None
        self.built_residues = None

    def _set_topo_definitions(self, definition_type: str):
        self.topo_definitions = ResidueTopologySet(definition_type)
        self.params = ParameterLoader(definition_type)
        self.params.fill_ic(self.topo_definitions, preserve=True)

    def set_chain_type(self, chain_type: str):
        chain_type = chain_type.lower()
        if chain_type in ('protein', 'peptide', 'polypeptide', 'polypeptide(l)'):
            chain_type = 'Polypeptide(L)'
            self._1to3 = protein_letters_1to3
            self._3to1 = protein_letters_3to1
            self._set_topo_definitions('protein')
        elif chain_type in ('rna', 'polyribonucleotide'):
            chain_type = 'Polyribonucleotide'
            self._3to1 = ResidueDefinition.na_3to1
            self._1to3 = ResidueDefinition.na_1to3
            self._set_topo_definitions('nucleic')
        elif chain_type in ('dna', 'polydeoxyribonucleotide'):
            chain_type = 'Polydeoxyribonucleotide'
            raise NotImplementedError('DNA is not supported yet.')
            self._3to1 = ResidueDefinition.na_3to1
            self._1to3 = ResidueDefinition.na_1to3
            self._set_topo_definitions('nucleic')
        else:
            raise ValueError(
                f'Invalid chain type \'{chain_type}\'! '
                f'Valid chain types are: Protein, RNA'
            )
        self.chain_type = chain_type

    def set_sequence(self, sequence: str, chain_type: str):
        self.set_chain_type(chain_type)
        self.resnames = []
        for one_letter in sequence:
            if one_letter not in self._1to3:
                raise ValueError(
                    f'Invalid one-letter code \'{one_letter}\' found '
                    f'in sequence: {sequence}'
                )
            self.resnames.append(self._1to3[one_letter])
        self.sequence = Seq(sequence)
        self._create_reported_residues()

    def set_three_letter_sequence(self, sequence3: str, chain_type: str):
        self.set_chain_type(chain_type)
        self.resnames = sequence3.split()
        sequence = ''
        for resname in self.resnames:
            if resname not in self._3to1:
                raise ValueError(
                    f'Invalid three-letter code \'{resname}\' found '
                    f'in sequence: {sequence3}'
                )
            sequence += self._3to1[resname]
        self.sequence = Seq(sequence)
        self._create_reported_residues()

    def _create_reported_residues(self):
        if self.chain_type == 'Polyribonucleotide' or 'Polydeoxyribonucleotide':
            # RNA and DNA are reported in one-letter code for PDBx format
            PDB_resnames = [self._3to1[resname] for resname in self.resnames]
        else:
            # Protein is reported in three-letter code for PDBx format
            PDB_resnames = self.resnames
        self._report_res = [
            (i+1, resname) for i, resname in enumerate(PDB_resnames)
        ]
        
    def _create_initial_residue(self):
        resname = self.resnames[0]
        res_def = self.topo_definitions[resname]
        res = res_def.create_residue(resseq = 1)
        return res
    
    def create_chain(self, chain_id: str = 'A'):
        self.chain_id = chain_id
        init_residue = self._create_initial_residue()
        self.built_residues = {1: init_residue}
        cur_coord_dict = self.topo_definitions[self.resnames[0]].standard_coord_dict
        for resseq, resname in enumerate(self.resnames[1:], start=2):
            last_res = self.built_residues[resseq-1]
            last_def = self.topo_definitions[last_res.resname]
            cur_def = self.topo_definitions[resname]
            if cur_def.standard_coord_dict is None:
                cur_def.standard_coord_dict = cur_def.build_standard_coord_from_ic_table()
            cur_coord_dict = self._create_neighbor_coord_dict(last_def, cur_def, cur_coord_dict)
            cur_res = cur_def.create_residue_from_coord_dict(
                cur_coord_dict, resseq = resseq
            )
            self.built_residues[resseq] = (cur_res)

        chain = PolymerChain(
            chain_id=self.chain_id,
            chain_type=self.chain_type,
            author_chain_id=self.chain_id,
            entity_id=1,
            canon_sequence=self.sequence,
            known_sequence=self.sequence,
            reported_res = self._report_res
        )

        for res in self.built_residues.values():
            chain.add(res)

        return chain

    def _find_nei_ic_entry(self, topo_definition):
        for atoms in topo_definition.ic:
            nei_atoms_next = []
            cur_atoms = []
            for i, atom in enumerate(atoms):
                if atom.startswith('+'):
                    nei_atoms_next.append((i, atom))
                else:
                    cur_atoms.append((i, atom))
            if len(nei_atoms_next) == 2:
                return cur_atoms, nei_atoms_next

    def _find_init_coord_for_nei(self, standard_coord_dict, atoms, nei_atoms):
        (i, a1), (j, a2) = atoms
        (k, a3), (l, a4) = nei_atoms
        if i < j < k < l:
            atom_names = (a2, a3, a4)
        elif l < k < j < i:
            atom_names = (a4, a3, a2)
        coords = (standard_coord_dict[atom_name] for atom_name in atom_names)
        nei_atom_names = []
        for atom_name in atom_names:
            if atom_name.startswith('+'):
                nei_atom_names.append(atom_name[1:])
            else:
                nei_atom_names.append('-'+atom_name)
        return dict(zip(nei_atom_names, coords))

    def _create_neighbor_coord_dict(self, last_def, cur_def, cur_coord_dict):
        atoms, nei_atoms = self._find_nei_ic_entry(last_def)
        init_coord_dict = self._find_init_coord_for_nei(cur_coord_dict, atoms, nei_atoms)
        new_coord_dict = cur_def.build_standard_coord_from_ic_table(
            init_coord_dict=init_coord_dict
        )
        return new_coord_dict
