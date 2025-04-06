import warnings
from copy import copy
import requests
import pandas as pd
from Bio.PDB.Selection import unfold_entities
# Nucleoside phosphates and phosphonates
from crimm.Data.components_dict import NUCLEOSIDE_PHOS, PDB_CHARMM_ION_NAMES 
from crimm.Fetchers import query_drugbank_info
from crimm.Utils.StructureUtils import index_to_letters, letters_to_index
from .Model import Model
from .Chain import (
    PolymerChain, Heterogens, Solvent, CoSolvent, Ion,
    Glycosylation, NucleosidePhosphate, Chain, Ligand, Oligosaccharide, Macrolide
)
from .Residue import Residue

class OrganizedModel(Model):
    """The OrganizedModel class represents a model in a structure with a specific 
    organization of chains. It is derived from the Model class and is used in the 
    Structure class to organize chains into categories. Require network access to
    fetch binding affinity and drugbank information. Alt locs will be removed and only 
    the selected alt loc atom in disordered residues will be kept in the model.
    The following chain types are supported:
    - protein: Polypeptide(L)
    - rna: Polyribonucleotide
    - dna: Polydeoxyribonucleotide
    - other_polymer: PolymerChain
    - solvent: Solvent
    - unknown_type: UnknownType
    - phos_ligand: NucleosidePhosphate
    - ligand: Ligand
    - co_solvent: CoSolvent
    - ion: Ion
    - glycosylation: Glycosylation
    - heterogens: Heterogens
    

    """
    chain_types = {
        'Polypeptide(L)': PolymerChain,
        'Polyribonucleotide': PolymerChain,
        'Polydeoxyribonucleotide': PolymerChain,
        'Oligosaccharide': Oligosaccharide,
        'Macrolide': Macrolide,
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
        'rna': 'Polyribonucleotide',
        'dna': 'Polydeoxyribonucleotide',
        'oligosaccharide': 'Oligosaccharide',
        'macrolide': 'Macrolide',
        'other_polymer': 'PolymerChain',
        'solvent': 'Solvent',
        'unknown_type': 'UnknownType',
        'phos_ligand': 'NucleosidePhosphate',
        'ligand': 'Ligand',
        'co_solvent': 'CoSolvent',
        'ion': 'Ion',
        'glycosylation': 'Glycosylation',
    }
    def __init__(
            self, entity, rename_charmm_ions=True, rename_solvent_oxygen=True,
            identify_ligands=False
        ):
        """Initialize the OrganizedModel object.
        Args:
            entity (Entity): The entity to be organized.
            rename_charmm_ions (bool): Whether to rename ions in the structure to 
            CHARMM ion name defined in water_ions.str
            rename_solvent_oxygen (bool): Whether to rename solvent oxygen to CHARMM 
            name "OH2" in the crystallographic water. Doing so will allow crimm to 
            generate topology definitions on the water
        """
        if entity.level == 'M':
            model = entity
            pdbx_description = entity.pdbx_description
        elif entity.level == 'S':
            model = entity.models[0]
            pdbx_description = entity.models[0].pdbx_description
        else:
            raise ValueError(
                "Only Structure level (S) or Model level (M) entity is accepted for OrganizedModel!"
                f"{entity}  has level \"{entity.level}\"."
            )
        super().__init__(model.id)
        self.pdbx_description = pdbx_description
        self.connect_dict = copy(model.connect_dict)
        ## keep original model's connect atoms for looking up covalent bonds
        ## for glycosylation
        self._ref_connect_atoms = model.connect_atoms
        self.identify_ligands = identify_ligands
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
        self.topology_loader = None
        
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
        self.organize(model, identify_ligands=self.identify_ligands)
        if rename_solvent_oxygen:
            self.rename_solvent_oxygen()
        if rename_charmm_ions:
            self.rename_charmm_ions()
        
        
    
    def filter(self, name):
        """Return the type of the chain with the given name."""
        name = name.rstrip('s').lower()
        if name in self.organized_chains:
            chains = []
            for chain in self:
                if chain.chain_type == self.organized_chains[name]:
                    chains.append(chain)
            return chains
        raise KeyError(f"Invalid chain type {name}! Available types are {self.organized_chains.keys()}.")

    @property
    def protein(self):
        return self.filter('protein')
    @property
    def RNA(self):
        return self.filter('rna')
    @property
    def DNA(self):
        return self.filter('dna')
    @property
    def oligosaccharide(self):
        return self.filter('oligosaccharide')
    @property
    def macrolide(self):
        return self.filter('macrolide')
    @property
    def other_polymer(self):
        return self.filter('other_polymer')
    @property
    def solvent(self):
        return self.filter('solvent')
    @property
    def unknown_type(self):
        return self.filter('unknown_type')
    @property
    def phos_ligand(self):
        return self.filter('phos_ligand')
    @property
    def ligand(self):
        return self.filter('ligand')
    @property
    def co_solvent(self):
        return self.filter('co_solvent')
    @property
    def ion(self):
        return self.filter('ion')
    @property
    def glycosylation(self):
        return self.filter('glycosylation')
    
    @property
    def non_solvent(self):
        """Return all chains that are not solvent chains."""
        non_solvent = []
        for chain in self:
            if chain.chain_type != 'Solvent':
                non_solvent.append(chain)
        return non_solvent
    
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
        if self._ref_connect_atoms is None:
            return False
        if 'covale' not in self._ref_connect_atoms:
            return False
        for atom_pair in self._ref_connect_atoms['covale']:
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
        resnames = set()
        for i, res in enumerate(hetero_res_list, start=1):
            description = None
            if hasattr(res, 'pdbx_description'):
                description = res.pdbx_description
            elif res.parent is not None:
                description  = res.parent.pdbx_description
            if description not in all_description and description is not None:
                all_description.append(description)
            res.detach_parent()
            if reset_resid:
                het_flag, resseq, icode = res.id
                res.id = (het_flag, i, icode)
            hetero_chain.add(res)
            resnames.add(res.resname)

        if chain_type == 'Solvent':
            hetero_chain.pdbx_description = 'water'
        elif len(all_description) > 0:
            hetero_chain.pdbx_description = ', '.join(all_description)
        hetero_chain.resnames = ', '.join(resnames)
        return hetero_chain

    def determine_heterogen_type(self, heterogens, identify_ligands=False):
        """Determine the type of heterogens in the model."""
        heterogen_type_dict = {
            'Ion': [],
            'Glycosylation': [],
            'Ligand': [],
            'NucleosidePhosphate': [],
            'CoSolvent': []   
        }
        
        for res in heterogens:
            if len(res.resname) <= 2:
                heterogen_type_dict['Ion'].append(res)
            elif self.is_glycosylation(res):
                heterogen_type_dict['Glycosylation'].append(res)
            elif not identify_ligands:
                # If the ligand is not guessed, all heterogens except ions and
                # identified glycosylation will be considered ligands
                heterogen_type_dict['Ligand'].append(res)
            elif res.resname in NUCLEOSIDE_PHOS:
                heterogen_type_dict['NucleosidePhosphate'].append(res)
            elif self.is_ligand(res):
                heterogen_type_dict['Ligand'].append(res)
            else:
                heterogen_type_dict['CoSolvent'].append(res)
        return heterogen_type_dict
    
    def update(self):
        self.organize(self, identify_ligands=self.identify_ligands)

    def organize(self, model: Model, identify_ligands=False):
        """Organize the chains in the model into categories."""
        undecided_heterogens = []
        all_chains = []
        solvent_entry = {'Solvent': []}
        chain_id_map = {}
        for chain in model.chains:
            if chain.chain_type == 'Solvent':
                solvent_entry['Solvent'].extend(chain.residues)
            elif chain.chain_type == 'Heterogens':
                undecided_heterogens.extend(chain.residues)
            else:
                chain_id_map[chain.id] = chain.id
                all_chains.append(chain)

        heterogen_type_dict = self.determine_heterogen_type(
            undecided_heterogens, identify_ligands=identify_ligands
        )
        heterogen_type_dict.update(solvent_entry)
        named_hetero_chains = {}
        for chain_type, hetero_res_list in heterogen_type_dict.items():
            named_hetero_chains[chain_type] = {}
            cur_chain_type = named_hetero_chains[chain_type]
            for res in hetero_res_list:
                temp_id = f'_{res.resname}'
                chain_id_map[res.parent.id] = temp_id
                if temp_id in cur_chain_type:
                    cur_chain_type[temp_id].append(res)
                else:
                    cur_chain_type[temp_id] = [res]
        for chain_type, resname_dict in named_hetero_chains.items():
            if len(resname_dict) == 0:
                continue
            for temp_id, hetero_res_list in resname_dict.items():
                hetero_chain = self.create_hetero_chain(
                    temp_id, hetero_res_list, chain_type
                )
                all_chains.append(hetero_chain)

        all_chains.sort(key=lambda x: x.id)
        temp_id_map = {}
        for i, chain in enumerate(all_chains):
            chain.detach_parent()
            temp_id = chain.id
            new_id = index_to_letters(i)
            temp_id_map[temp_id] = new_id
            chain.id = new_id
            self.add(chain)
        chain_id_map = {k: temp_id_map[v] for k, v in chain_id_map.items()}
        # update the connected atoms in the model level bonds (e.g. disulfide bonds)
        self._update_connect_dict_chain_id(chain_id_map)
        self.set_connect(self.connect_dict)
            
    def _update_connect_dict_chain_id(self, id_map: dict):
        """Update the chain ID in the connect_dict."""
        for entries in self.connect_dict.values():
            for atom_pair_dicts in entries:
                for atom_info in atom_pair_dicts:
                    if atom_info['chain'] in id_map:
                        atom_info['chain'] = id_map[atom_info['chain']]
                    else:
                        print(f"Chain {atom_info['chain']} not found in the model!")

    def rename_solvent_oxygen(self):
        """Rename oxygen atom in solvent to OH2."""
        for chain in self.solvent:
            for res in chain:
                if 'O' in res:
                    atom = res['O']
                    atom.rename('OH2')

    def replace_ion(self, res, new_ion_name):
        """Replace the ion residue by changing its name."""
        new_ion_name = new_ion_name.upper()
        if res.resname == new_ion_name:
            return
        het_flag, resseq, icode = res.id
        het_flag = f'H_{new_ion_name}'
        res.id = (het_flag, resseq, icode)
        res.resname = new_ion_name
        for atom in res:
            # Rename the atom. Atom name should be the same as resname here
            atom.rename(new_ion_name)

    def replace_ion_chain(self, ion_chain, new_ion_name):
        """Replace the ion chain by changing its name."""
        if ion_chain.chain_type != 'Ion':
            raise ValueError(
                f"Chain {ion_chain.id} is not an Ion chain!"
            )
        new_ion_name = new_ion_name.upper()
        for res in ion_chain:
            self.replace_ion(res, new_ion_name)
        ion_chain.resnames = new_ion_name
        ion_chain.pdbx_description = new_ion_name

    def rename_charmm_ions(self):
        """Rename the ions to CHARMM naming convention."""
        for chain in self.ion:
            for res in chain:
                charmm_name = PDB_CHARMM_ION_NAMES.get(
                        res.resname.upper(), res.resname
                    )
                if res.resname == 'ZN':
                    res.resname = 'ZN2'
                    continue
                elif res.resname == charmm_name:
                    continue
                self.replace_ion(res, charmm_name)

    def __repr__(self):
        repr_str = f"<OrganizedModel model={self.pdb_id} "
        for key, val in self.organized_chains.items():
            if len(self.filter(key)) > 0:
                repr_str += f"{val}={len(self.filter(key))} "
        repr_str += ">"
        return repr_str

    def _ipython_display_(self):
        """Return the nglview interactive visualization window"""
        if len(self) != 0:
            from crimm.Visualization import show_nglview_multiple
            from IPython.display import display
            display(show_nglview_multiple(self.child_list))
        print(self.expanded_view())
        
    def correct_chain_type(self, chain_id, chain_type):
        """Reassign the chain type of a chain in the model."""
        if (general_name:=chain_type.rstrip('s').lower()) in self.organized_chains:
            ## Use the actual chain class name if user provides the organized_chains name
            chain_type = self.organized_chains[general_name]
        if chain_type not in self.chain_types:
            raise ValueError(f"Invalid chain type {chain_type}!")
        old_chain = self[chain_id]
        if chain_type == old_chain.chain_type:
            return
        self.detach_child(chain_id)
        new_chain = self.create_hetero_chain(
            old_chain.id, old_chain.residues, chain_type
        )
        self.add(new_chain)

    def combine_chains_by_id(self, chain_ids, new_id=None):
        """Combine the chains in the model into a single chain."""
        if len(chain_ids) == 0:
            return
        chains = [self[chain_id] for chain_id in chain_ids]
        self.combine_chains(chains, new_id)


    def combine_chains(self, chains, new_id=None):
        """Combine a list of chains into a single chain."""
        if len(chains) == 0:
            return
        chain_types = set(c.chain_type for c in chains)
        if len(chain_types) > 1:
            raise ValueError(
                f"Cannot combine chains with different chain types! {chain_types}"
                "found in the chain list."
            )
        chain_type = chain_types.pop()
        if new_id is None:
            new_id = self[0].id
        new_chain = self.chain_types[chain_type](new_id)
        all_res = [res for chain in chains for res in chain]
        for i, res in enumerate(all_res, start=1):
            het_flag, resseq, icode = res.id
            res.id = (het_flag, i, icode)
            new_chain.add(res)

        for chain in chains:
            self.detach_child(chain.id)
        self.add(new_chain)

    def _add_ligand_residue(self, res, chain_id=None, description=None):
        """Add a ligand residue to the model."""
        self.sort_chains(reset_id=False)
        if description is not None:
            res.pdbx_description = description
        if chain_id in self:
            chain = self[chain_id]
            chain.add(res)
            all_descriptions = set(res.pdbx_description for res in chain if res.pdbx_description is not None)
            if len(all_descriptions) > 0:
                chain.pdbx_description = ', '.join(all_descriptions)
            else:
                chain.pdbx_description = None
            return

        if chain_id is None:
            chain_id = index_to_letters(letters_to_index(self.chains[-1].id)+1)
            assert chain_id not in self
        
        chain = self.chain_types['Ligand'](chain_id)
        chain.pdbx_description = res.pdbx_description
        chain.resnames = [res.resname]
        self.add(chain)
        chain.add(res)
        
    def _add_ligand_chain(self, chain, description=None):
        """Add a ligand chain to the model."""
        self.sort_chains(reset_id=False)
        if chain.id in self:
            new_id = index_to_letters(letters_to_index(self.chains[-1].id)+1)
            warnings.warn(
                f"Chain {chain.id} already exists in the model! "
                f"Ligand chain id will be renamed to {new_id}."
            )
            chain.id = new_id
        self.add(chain)
        if description is not None:
            chain.pdbx_description = description

    def add_ligand(self, ligand, chain_id=None, description=None):
        """Add a ligand to the model."""
        if isinstance(ligand, Residue):
            self._add_ligand_residue(ligand, chain_id, description)
        elif isinstance(ligand, Chain):
            ligand.id = chain_id
            self._add_ligand_chain(ligand, description)
        else:
            raise ValueError(f"Invalid ligand type {type(ligand)}!")