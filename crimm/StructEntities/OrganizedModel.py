import warnings
import requests
from copy import copy
import pandas as pd
from Bio.PDB.Selection import unfold_entities
from crimm.Data.components_dict import NUCLEOSIDE_PHOS # Nucleoside phosphates and phosphonates
from crimm.Fetchers import query_drugbank_info
from .Model import Model
from .Chain import (
    PolymerChain, Heterogens, Solvent, CoSolvent, Ion,
    Glycosylation, NucleosidePhosphate, Chain
)

class OrganizedChainContainer(list):
    """The EntityTypeContainer class is a list of chains with the same chain type."""
    def __init__(self, chain_type):
        super().__init__()
        self.chain_type = chain_type
        
    def __repr__(self):
        return f"<{self.chain_type}={len(self)}>"
    
    def __str__(self):
        return f"{self.chain_type}={len(self)}"

class OrganizedModel:
    """The OrganizedModel class represents a model in a structure with a specific 
    organization of chains. It is derived from the Model class and is used in the 
    Structure class to organize chains into categories. Require network access to
    fetch binding affinity and drugbank information."""
    chain_types = {
        'Polypeptide(L)': PolymerChain,
        'Polyribonucleotide': PolymerChain,
        'Polydeoxyribonucleotide': PolymerChain,
        'Solvent': Solvent,
        'Heterogens': Heterogens,
        'CoSolvent': CoSolvent,
        'Ion': Ion,
        'Glycosylation': Glycosylation,
        'NucleosidePhosphate': NucleosidePhosphate,
        'UnknownType': Chain,
    }

    def __init__(self, entity):
        if entity.level == 'M':
            self._model = entity
        elif entity.level == 'S':
            self._model = entity.models[0]
        else:
            raise ValueError(
                "Only Structure level (S) or Model level (M) entity is accepted for OrganizedModel!"
                f"{entity}  has level \"{entity.level}\"."
            )
        self.pdb_id = None
        self.rcsb_web_data = None
        # Binding Affinity Information
        self.binding_info = None
        # Biologically Interesting Molecules Information
        self.bio_mol_info = None
        # Lists of Entity Classifications
        self.protein = OrganizedChainContainer('Polypeptide(L)')
        self.RNA = OrganizedChainContainer('Polyribonucleotide')
        self.DNA = OrganizedChainContainer('Polydeoxyribonucleotide')
        self.other_polymer = OrganizedChainContainer('UnknownType')
        self.phos_ligand = OrganizedChainContainer('NucleosidePhosphate')
        self.ligand = OrganizedChainContainer('Heterogens')
        self.co_solvent = OrganizedChainContainer('CoSolvent')
        self.ion = OrganizedChainContainer('Ion')
        self.solvent = OrganizedChainContainer('Solvent')
        self.glycosylation = OrganizedChainContainer('Glycosylation')
        self.unknown_type = OrganizedChainContainer('UnknownType')
        # Names (3-letter code) for ligand with binding affinity information
        self.lig_names = set()
        # Names (descriptor string) for ligand that exists in 
        # Biologically Interesting Molecules databases
        self.bio_mol_names = set()

        if self._model.parent.id is None:
            warnings.warn(
                "PDB ID not set for this model! "
                "Binding affinity information will not be fetched. Possible errors "
                "in ligand classification."
            )
        else:
            self.pdb_id = self._model.parent.id
            self._get_rcsb_web_data()
            self.binding_info = self._get_keyword_info('rcsb_binding_affinity')
            self.bio_mol_info = self._get_keyword_info('pdbx_molecule_features')
        if self.binding_info is not None:
            self.lig_names = set(self.binding_info.comp_id)
        if self.bio_mol_info is not None:
            self.bio_mol_names = set(self.bio_mol_info.name)
        self.organize()

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
        if 'covale' not in self._model.connect_atoms:
            return False
        for atom_pair in self._model.connect_atoms['covale']:
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

    def organize(self):
        """Organize the chains in the model into categories."""
        _temp_ligand = []
        for chain in self._model.chains:
            if chain.chain_type == 'Polypeptide(L)':
                self.protein.append(chain)
            elif chain.chain_type == 'Polyribonucleotide':
                self.RNA.append(chain)
            elif chain.chain_type == 'Polydeoxyribonucleotide':
                self.DNA.append(chain)
            elif isinstance(chain, PolymerChain):
                self.other_polymer.append(chain)
            elif chain.chain_type == 'Solvent':
                self.solvent.append(chain)
            elif chain.chain_type == 'Heterogens':
                _temp_ligand.extend(chain.residues)
            else:
                self.unknown_type.append(chain)

        for res in _temp_ligand:
            if len(res.resname) <= 2:
                self.ion.append(res)
            elif res.resname in NUCLEOSIDE_PHOS:
                self.phos_ligand.append(res)
            elif self.is_glycosylation(res):
                self.glycosylation.append(res)
            elif self.is_ligand(res):
                self.ligand.append(res)
            else:
                self.co_solvent.append(res)

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
        if len(self._model) == 0:
            return
        from crimm.Visualization import show_nglview_multiple
        from IPython.display import display
        display(show_nglview_multiple(self._model.child_list))
        print(repr(self))
        print(self._model.expanded_view())

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