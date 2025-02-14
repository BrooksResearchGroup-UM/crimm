import warnings
import requests
from copy import copy
import pandas as pd
from Bio.PDB.Selection import unfold_entities
# Nucleoside phosphates and phosphonates
from crimm.Data.components_dict import NUCLEOSIDE_PHOS, PDB_CHARMM_ION_NAMES 
from crimm.Fetchers import query_drugbank_info
from .Model import Model
from .Chain import (
    PolymerChain, Heterogens, Solvent, CoSolvent, Ion,
    Glycosylation, NucleosidePhosphate, Chain, Ligand
)
from crimm.Utils.StructureUtils import index_to_letters

class OrganizedModel(Model):
    """The OrganizedModel class represents a model in a structure with a specific 
    organization of chains. It is derived from the Model class and is used in the 
    Structure class to organize chains into categories. Require network access to
    fetch binding affinity and drugbank information.
    """
    chain_types = {
        'Polypeptide(L)': PolymerChain,
        'Polyribonucleotide': PolymerChain,
        'Polydeoxyribonucleotide': PolymerChain,
        'Solvent': Solvent,
        'Heterogens': Heterogens,
        'Ligand': Ligand,
        'CoSolvent': CoSolvent,
        'Ion': Ion,
        'Glycosylation': Glycosylation,
        'NucleosidePhosphate': NucleosidePhosphate,
        'UnknownType': Chain,
    }
    organized_chains = {
        'protein': 'Polypeptide(L)',
        'RNA': 'Polyribonucleotide',
        'DNA': 'Polydeoxyribonucleotide',
        'other_polymer': 'PolymerChain',
        'solvent': 'Solvent',
        'unknown_type': 'UnknownType',
        'phos_ligand': 'NucleosidePhosphate',
        'ligand': 'Ligand',
        'co_solvent': 'CoSolvent',
        'ion': 'Ion',
        'glycosylation': 'Glycosylation',
    }
    def __init__(self, entity, rename_ion=True, rename_solvent=True, make_copy=True):
        if entity.level == 'M':
            model = entity.copy()
        elif entity.level == 'S':
            model = entity.models[0].copy()
        else:
            raise ValueError(
                "Only Structure level (S) or Model level (M) entity is accepted for OrganizedModel!"
                f"{entity}  has level \"{entity.level}\"."
            )
        super().__init__(model.id)
        self.connect_atoms = model.connect_atoms
        self.connect_dict = model.connect_dict
        self.pdb_id = None
        self.rcsb_web_data = None
        # Binding Affinity Information
        self.binding_info = None
        # Biologically Interesting Molecules Information
        self.bio_mol_info = None
        # Names (3-letter code) for ligand with binding affinity information
        self.lig_names = set()
        # Names (descriptor string) for ligand that exists in 
        # Biologically Interesting Molecules databases
        self.bio_mol_names = set()
        
        if model.pdb_id is not None:
            self.pdb_id = model.pdb_id
            self._get_rcsb_web_data()
            self.binding_info = self._get_keyword_info('rcsb_binding_affinity')
            self.bio_mol_info = self._get_keyword_info('pdbx_molecule_features')
        elif model.parent.id is not None:
            self.pdb_id = model.parent.id
            self._get_rcsb_web_data()
            self.binding_info = self._get_keyword_info('rcsb_binding_affinity')
            self.bio_mol_info = self._get_keyword_info('pdbx_molecule_features')
        else:
            warnings.warn(
                "PDB ID not set for this model! "
                "Binding affinity information will not be fetched. Possible errors "
                "in ligand classification."
            )

        if self.binding_info is not None:
            self.lig_names = set(self.binding_info.comp_id)
        if self.bio_mol_info is not None:
            self.bio_mol_names = set(self.bio_mol_info.name)
        self.organize(model, make_copy=make_copy)
        if rename_solvent:
            self.rename_solvent()
        if rename_ion:
            self.rename_ion()

    def __getattr__(self, name):
        if name in self.organized_chains:
            chains = []
            for chain in self:
                if chain.chain_type == self.organized_chains[name]:
                    chains.append(chain)
            return chains
        return getattr(super(), name)

    def _get_rcsb_web_data(self):
        query_url = f'https://data.rcsb.org/rest/v1/core/entry/{self.pdb_id}'
        response = requests.get(query_url, timeout=500)
        if (code := response.status_code) != 200:
            warnings.warn(
                f"GET request on RCSB for \"{self.pdb_id}\" for binding affinity data "
                f"did not return valid result! \n[Status Code] {code}"
            )
            return
        self.rcsb_web_data = response.json()

    def _get_keyword_info(self, keyword):
        if self.rcsb_web_data is None or keyword not in self.rcsb_web_data:
            return
        return pd.DataFrame(self.rcsb_web_data[keyword])

    def is_glycosylation(self, residue):
        """Check if the heterogen is part of glycosylation (covalently bonded sugar)"""
        if 'covale' not in self.connect_atoms:
            return False
        for atom_pair in self.connect_atoms['covale']:
            if residue in unfold_entities(atom_pair, 'R'):
                return True
        return False

    def is_ligand(self, residue):
        """Check if the residue is a ligand."""
        if len(self.lig_names) > 0:
            return residue.resname in self.lig_names
        elif len(self.bio_mol_names) > 0:
            return residue.pdbx_description in self.bio_mol_names
        else:
            return query_drugbank_info(residue.resname) is not None
    
    def create_hetero_chain(
            self, chain_id, hetero_res_list, chain_type='Heterogens', 
            reset_resid=True
        ):
        """Create a new Heterogens chain from a list of heterogens."""
        all_description = []
        hetero_chain = self.chain_types[chain_type](chain_id)
        for i, res in enumerate(hetero_res_list, start=1):
            if reset_resid:
                het_flag, resseq, icode = res.id
                res.id = (het_flag, i, icode)
            if (
                hasattr(res, 'pdbx_description') and 
                res.pdbx_description not in all_description and
                res.pdbx_description is not None
            ):
                all_description.append(res.pdbx_description)
            hetero_chain.add(res)

        hetero_chain.pdbx_description = ', '.join(all_description)
        if chain_type == 'Solvent':
            hetero_chain.pdbx_description = 'water'
        return hetero_chain

    def determine_heterogen_type(self, heterogens):
        """Determine the type of heterogens in the model."""
        heterogen_type_dict = {
            'NucleosidePhosphate': [],
            'Glycosylation': [],
            'Ligand': [],
            'CoSolvent': [],
            'Ion': []
        }

        for res in heterogens:
            if len(res.resname) <= 2:
                heterogen_type_dict['Ion'].append(res)
            elif res.resname in NUCLEOSIDE_PHOS:
                heterogen_type_dict['NucleosidePhosphate'].append(res)
            elif self.is_glycosylation(res):
                heterogen_type_dict['Glycosylation'].append(res)
            elif self.is_ligand(res):
                heterogen_type_dict['Ligand'].append(res)
            else:
                heterogen_type_dict['CoSolvent'].append(res)
        return heterogen_type_dict
    
    def update(self):
        self.organize(self, make_copy=False)

    def organize(self, model: Model, make_copy=True):
        """Organize the chains in the model into categories."""
        undecided_heterogens = []
        all_chains = []
        solvent_entry = {'Solvent': []}
        for chain in model.chains:
            if make_copy:
                chain = chain.copy()
            if chain.chain_type == 'Solvent':
                solvent_entry['Solvent'].extend(chain.residues)
            elif chain.chain_type == 'Heterogens':
                undecided_heterogens.extend(chain.residues)
            else:
                all_chains.append(chain)

        heterogen_type_dict = self.determine_heterogen_type(undecided_heterogens)
        heterogen_type_dict.update(solvent_entry)

        for i, (chain_type, hetero_res_list) in enumerate(heterogen_type_dict.items()):
            if len(hetero_res_list) == 0:
                continue
            temp_id = index_to_letters(i)
            hetero_chain = self.create_hetero_chain(
                f'H{temp_id}', hetero_res_list, chain_type
            )
            all_chains.append(hetero_chain)

        for i, chain in enumerate(all_chains):
            chain.id = index_to_letters(i)
            self.add(chain)
            
    def rename_solvent(self):
        """Rename the oxygen atom in solvent to OH2."""
        for chain in self.solvent:
            for res in chain:
                if 'O' in res:
                    atom = res['O']
                    atom.rename('OH2')

    def rename_ion(self):
        """Rename the ions to CHARMM naming convention."""
        for chain in self.ion:
            for res in chain:
                charmm_name = PDB_CHARMM_ION_NAMES.get(
                        res.resname.upper(), res.resname
                    )
                if res.resname == charmm_name:
                    continue
                res.resname = charmm_name
                for atom in res:
                    # Rename the atom to CHARMM naming convention
                    # atom name should be the same as resname here
                    atom.rename(charmm_name)

    def __repr__(self):
        repr_str = f"<OrganizedModel model={self.pdb_id} "
        if len(self.protein) > 0:
            repr_str += f"Protein={len(self.protein)} "
        if len(self.RNA) > 0:
            repr_str += f"RNA={len(self.RNA)} "
        if len(self.DNA) > 0:
            repr_str += f"DNA={len(self.DNA)} "
        if len(self.other_polymer) > 0:
            repr_str += f"OtherPolymer={len(self.other_polymer)} "
        if len(self.phos_ligand) > 0:
            repr_str += f"NucleosidePhosphates/Phosphonate={len(self.phos_ligand)} "
        if len(self.glycosylation) > 0:
            repr_str += f"Glycosylation={len(self.glycosylation)} "
        if len(self.ligand) > 0:
            repr_str += f"Ligand={len(self.ligand)} "
        if len(self.co_solvent) > 0:
            repr_str += f"CoSolvent={len(self.co_solvent)} "
        if len(self.ion) > 0:
            repr_str += f"Ion={len(self.ion)} "
        if len(self.solvent) > 0:
            repr_str += f"Solvent={len(self.solvent)} "
        if len(self.unknown_type) > 0:
            repr_str += f"UnknownType={len(self.unknown_type)} "
        repr_str += ">"
        return repr_str

    def _ipython_display_(self):
        """Return the nglview interactive visualization window"""
        if len(self) != 0:
            from crimm.Visualization import show_nglview_multiple
            from IPython.display import display
            display(show_nglview_multiple(self.child_list))
        print(self.expanded_view())

    def create_new_model_from_list(self, entity_list):
        new_model = Model(self.pdb_id)
        chain_list = []
        for entity in entity_list:
            if entity.level == 'C':
                chain_list.append(entity)
            elif entity.level == 'R' and entity.parent not in chain_list:
                chain_list.append(entity.parent)
        for chain in chain_list:
            new_model.add(copy(chain))
        return new_model