# Copyright (C) 2023, Truman Xu, Brooks Lab at the University of Michigan
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""Module containing the parser class for constructing structures from mmCIF 
files from PDB"""
import warnings
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from crimm.StructEntities.Atom import Atom
from crimm.IO.MMCIF2Dict import MMCIF2Dict
from crimm.IO.StructureBuilder import StructureBuilder
from crimm.StructEntities.Chain import (
    Chain, PolymerChain, Heterogens, Oligosaccharide, Solvent, Macrolide
)
from crimm.StructEntities.Model import Model
from crimm.Utils.StructureUtils import get_coords, index_to_letters
class MMCIFParser:
    """Parser class for standard mmCIF files from PDB"""
    def __init__(
            self,
            first_model_only = True,
            use_bio_assembly = True,
            include_solvent = True,
            include_hydrogens = False,
            strict_parser = True,
            QUIET = False
        ):
        self._structure_builder = StructureBuilder()
        self.QUIET = QUIET
        self.first_model_only = first_model_only
        self.use_bio_assembly = use_bio_assembly
        self.include_solvent = include_solvent
        self.include_hydrogens = include_hydrogens
        self.strict_parser = strict_parser
        self.cifdict = None
        self.model_template = None
        self.symmetry_ops = None

    @staticmethod
    def _cif_find_resolution(cifdict):
        """Find structure resolution information from the parsed mmCIF dictionary"""
        resolution_keys = [
            ("refine", "ls_d_res_high"),
            ("refine_hist", "d_res_high"),
            ("em_3d_reconstruction", "resolution"),
        ]
        for key, subkey in resolution_keys:
            resolution = cifdict.level_two_get(key, subkey)
            if resolution is not None and (value:=resolution[0]) is not None:
                return value

    @staticmethod
    def _cif_get_header(cifdict):
        """Get header information from the parsed mmCIF dictionary"""
        header = {
            "name": cifdict.get("data"),
            "keywords": cifdict.retrieve_single_value_dict("struct_keywords"),
            "citation": cifdict.get("citation"),
            "idcode": cifdict.retrieve_single_value_dict('struct'),
            "deposition_date": cifdict.level_two_get(
                "pdbx_database_status", "recvd_initial_deposition_date"
            )[0]
        }

        return header

    def _cif_find_symmetry_info(self):
        symmetry_info = self.cifdict.get('pdbx_struct_oper_list')
        if symmetry_info is None:
            return
        operation_names = symmetry_info['type']
        matrices = np.empty((len(operation_names), 3, 3))
        vectors = np.empty((len(operation_names), 3))
        for key, val in symmetry_info.items():
            if key.startswith('matrix'):
                idx_j, idx_k = int(key[-2])-1, int(key[-1])-1
                for idx_i, x in enumerate(val):
                    matrices[idx_i, idx_j, idx_k] = x
            if key.startswith('vector'):
                idx_j = int(key[-1])-1
                for idx_i, x in enumerate(val):
                    vectors[idx_i, idx_j] = x

        operation_dict = {}
        for operation, matrix, vector in zip(operation_names, matrices, vectors):
            if operation == 'identity operation':
                continue
            if operation not in operation_dict:
                operation_dict[operation] = []
            operation_dict[operation].append((matrix, vector))

        return operation_dict

    def get_structure(self, filepath, structure_id = None):
        """Return the structure.

        Arguments:
         :structure_id: string, the id that will be used for the structure
         :filepath: path to mmCIF file, OR an open text mode file handle

        """
        with warnings.catch_warnings():
            if self.QUIET:
                warnings.filterwarnings("ignore", category=PDBConstructionWarning)
            # mmCIF will be parsed into dictionary first and then namedtuples
            # to gather all the necessary info to construct the structure
            self.cifdict = MMCIF2Dict(filepath)
            self.model_template = self.create_model_template()
            # find crystal symmetry operation
            self.symmetry_ops = self._cif_find_symmetry_info()
            self._build_structure(structure_id)
            # set additional info on the structure
            structure_method = None
            struct_method_list = self.cifdict.level_two_get("exptl", "method")
            if struct_method_list is not None:
                structure_method = struct_method_list[0]
                
            self._structure_builder.set_structure_method(structure_method)
            resolution = self._cif_find_resolution(self.cifdict)
            self._structure_builder.set_resolution(resolution)
            header = self._cif_get_header(self.cifdict)
            self._structure_builder.set_header(header)

        return self._structure_builder.get_structure()

    def create_polymer_chain_dict(self):
        """Create a dictionary for the PolymerChain object from mmCIF. Empty 
        PolymerChain classes will be created, and information on entity id, 
        author chain id, reported sequence, and reported missing residues will 
        be assigned. Return a dictionary with empty PolymerChain objects keyed
        by entity id (int)"""
        zero_occupancy_residues = self.cifdict.create_namedtuples(
            'pdbx_unobs_or_zero_occ_residues'
        )
        missing_res_dict = dict()
        for res in zero_occupancy_residues:
            if res.auth_asym_id not in missing_res_dict:
                missing_res_dict[res.auth_asym_id] = []
            missing_res_dict[res.auth_asym_id].append(
                (res.label_seq_id, res.label_comp_id)
            )

        entity_poly = self.cifdict.create_namedtuples('entity_poly')
        entity_poly_dict = dict()
        for entity in entity_poly:
            auth_chain_ids = str(entity.pdbx_strand_id).split(',')
            for chain_id in auth_chain_ids:
                entity_poly_dict[chain_id] = entity

        entity_poly_seq = self.cifdict.create_namedtuples('entity_poly_seq')
        reported_res_dict = dict()
        for res in entity_poly_seq:
            if res.entity_id not in reported_res_dict:
                reported_res_dict[res.entity_id] = []
            reported_res_dict[res.entity_id].append((res.num, res.mon_id))

        polymer_dict = dict()
        for auth_chain_id, entity_info in entity_poly_dict.items():
            entity_id = entity_info.entity_id
            chain = PolymerChain(
                None,
                entity_id,
                auth_chain_id,
                entity_info.type,
                entity_info.pdbx_seq_one_letter_code,
                entity_info.pdbx_seq_one_letter_code_can,
                reported_res_dict[entity_info.entity_id],
                missing_res_dict.get(auth_chain_id)
            )
            polymer_dict[entity_id] = chain
        return polymer_dict

    def _set_chain_attr(self, entity, chain):
        """Set the info from mmcif as attribute for the chain"""
        for k, v in entity._asdict().items():
            if k != 'id' and k != 'type' and v is not None:
                setattr(chain, k, v)

    def create_model_template(self):
        """Set up empty templates for the chains in a template model. The chain 
        types are determined based on the entity type in mmCIF, and a model object
        containing the empty chains will be returned."""
        model_temp = Model('template')

        poly_dict = self.create_polymer_chain_dict()
        entity_info = self.cifdict.create_namedtuples('entity')

        for entity in entity_info:
            if entity.type == 'polymer':
                cur_chain = poly_dict[entity.id]
            elif entity.type == 'non-polymer':
                cur_chain = Heterogens(entity.id)
            elif entity.type == 'water':
                cur_chain = Solvent(entity.id)
            elif entity.type == 'branched':
                cur_chain = Oligosaccharide(entity.id)
            elif entity.type == 'macrolide':
                cur_chain = Macrolide(entity.id)
            elif not self.strict_parser:
                cur_chain = Chain(entity.id)
            else:
                raise TypeError(f'Unknown Chain Type: {entity.type}')
            self._set_chain_attr(entity, cur_chain)
            model_temp.add(cur_chain)
        return model_temp

    def create_atom_from_namedtuple(self, atom_entry):
        """Create an Atom object from the namedtuple of an "atom_site" entry
        from the mmCIF dict

        Arguments:
         :param atom_entry: A namedtuple created from mmCIF "atom_site" entries
                           with field names as attributes in the tuple
         :type atom_entry: namedtuple
        """
        coord = np.array(
                [atom_entry.Cartn_x, atom_entry.Cartn_y, atom_entry.Cartn_z]
        )
        altloc = atom_entry.label_alt_id
        if altloc is None:
            altloc = ' '

        atom = Atom(
            name = atom_entry.label_atom_id,
            fullname = atom_entry.label_atom_id,
            coord = coord,
            bfactor = atom_entry.B_iso_or_equiv,
            occupancy = atom_entry.occupancy,
            altloc = altloc,
            serial_number = atom_entry.id,
            element = atom_entry.type_symbol,
        )
        return atom

    @staticmethod
    def _assign_hetflag(fieldname, resname):
        if fieldname != "HETATM":
            return ' '
        if resname in ("HOH", "WAT"):
            return "W"
        return "H"

    def _create_residue_dict_entry(self, atom_entry, resseq):
        """Create a residue dictionary entry with empty atom list"""
        resname = str(atom_entry.label_comp_id)
        hetatm_flag = self._assign_hetflag(
            atom_entry.group_PDB, resname
        )
        icode = atom_entry.pdbx_PDB_ins_code
        if icode is None:
            icode = ' '
        res_id = (hetatm_flag, resseq, icode)
        res_dict_entry = {
            "resname": resname,
            "res_id": res_id,
            "author_seq_id": atom_entry.auth_seq_id,
            "atom_list": []
        }
        return res_dict_entry
        
    def create_atom_site_entry_dict(self):
        """Create a dictionary containing structured data from all "atom_site" 
        fields in mmCIF. Return a dictionary that contains four levels, which 
        are keyed by [model_id,[chain_id,[resseq]]].

        The dictionary adopt the strucuture as following.
        structure_dict : {
            model_dict[model_id] : {
                chain_dict[chain_id] : {
                    res_dict[resseq] : {
                        "resname": str,
                        "res_id": Tuple[hetflag: str, resseq: int, icode: str],
                        "atom_list": List[atoms: Atom]
                    }
                }
            }
        }
        """
        atom_site = self.cifdict.create_namedtuples('atom_site')
        model_dict = dict()

        for atom_entry in atom_site:
            if not self.include_hydrogens and atom_entry.type_symbol == 'H':
                continue
            model_num = atom_entry.pdbx_PDB_model_num
            if model_num not in model_dict:
                if self.first_model_only and len(model_dict) > 0:
                    break
                model_dict[model_num] = dict()
            entity_dict = model_dict[model_num]
            entity_id = atom_entry.label_entity_id
            if entity_id not in entity_dict:
                entity_dict[entity_id] = dict()
            chain_dict = entity_dict[entity_id]
            chain_id = atom_entry.label_asym_id
            if chain_id not in chain_dict:
                last_auth_seq = None
                het_chain_resseq = 0
                chain_dict[chain_id] = dict()
            res_dict = chain_dict[chain_id]

            resseq = atom_entry.label_seq_id
            auth_seq_id = atom_entry.auth_seq_id
            if resseq is None:
                # The residue sequence for HETATOM is not defined in mmCIF, 
                # we need to manually increment it from the last sequence number
                if auth_seq_id != last_auth_seq:
                    het_chain_resseq += 1
                    last_auth_seq = auth_seq_id
                resseq = het_chain_resseq

            if resseq not in res_dict:
                res_dict[resseq] = self._create_residue_dict_entry(atom_entry, resseq)
            atom = self.create_atom_from_namedtuple(atom_entry)
            res_dict[resseq]['atom_list'].append(atom)

        return model_dict

    def create_assembly_dict(self):
        """Return a dictionary of chains grouped by mmcif assembly ids under
        "pdbx_struct_assembly_gen". If no assembly info available, None is
        returned.
        """
        assemblies = dict()
        structure_id = self.cifdict['data']
        if structure_id.startswith('AF-'):
            # AF-xxxxx is AlphaFold Structure, which does not have assembly
            return

        if 'pdbx_struct_assembly_gen' not in self.cifdict:
            warnings.warn(
                f"No structure assembly info in {structure_id}"
            )
            return
        entries = self.cifdict.create_namedtuples('pdbx_struct_assembly_gen')
        if self.use_bio_assembly:
            entries = entries[:1]
        for entry in entries:
            chain_id_list = entry.asym_id_list.split(',')
            assemblies[entry.assembly_id] = chain_id_list

        return assemblies

    def _execute_symmetry_operations(self, model):
        if not self.symmetry_ops:
            return
        reference_model = model.copy()
        reference_model.detach_parent()

        for operation_name, operations in self.symmetry_ops.items():
            warnings.warn(
                f"{operation_name.upper()} performed as specified in mmCIF file."
            )
            for matrix, vector in operations:
                copy_model = reference_model.copy()
                copy_coords = get_coords(copy_model)
                new_coords = copy_coords @ matrix + vector
                for i, atom in enumerate(copy_model.get_atoms()):
                    atom.coord = new_coords[i]
                for i, chain in enumerate(copy_model):
                    chain.id = index_to_letters(len(model)+i)
                for chain in copy_model:
                    model.add(chain)
        model.reset_atom_serial_numbers()

    def _build_structure(self, structure_id):
        """build the structure with structure builder object and mmcif dict"""
        sb = self._structure_builder
        self.model_template = self.create_model_template()
        
        all_anisou = self.cifdict.create_namedtuples('atom_site_anisotrop')
        if structure_id is None:
            structure_id = self.cifdict['data']
        sb.init_structure(structure_id)
        sb.init_seg(" ")
        if 'cell' in self.cifdict:
            cell_info = self.cifdict.create_namedtuples('cell')[0]
            sb.structure.cell_info = cell_info._asdict()
        assembly_dict = self.create_assembly_dict()
        if assembly_dict is None:
            selected_chains = None
        else:
            sb.structure.assemblies = assembly_dict
            selected_chains = []
            for chain_list in assembly_dict.values():
                selected_chains.extend(chain_list)

        atom_site = self.create_atom_site_entry_dict()
        ##TODO: refactor these nested for loops
        for model_id, entity_dict in atom_site.items():
            model = Model(model_id)
            sb.structure.add(model)
            sb.model = model
            for entity_id, chain_dict in entity_dict.items():
                for chain_id, res_dict in chain_dict.items():
                    if (
                        selected_chains is not None
                    ) and (
                        chain_id not in selected_chains
                    ):
                        continue
                    chain = self.model_template[entity_id].copy()
                    if not self.include_solvent and isinstance(chain, Solvent):
                        continue
                    sb.model.add(chain)
                    sb.chain = chain
                    # This is the mmCIF label chain id
                    sb.chain.id = chain_id
                    for resseq, res_info in res_dict.items():
                        resname = res_info["resname"]
                        res_id = res_info["res_id"]
                        author_seq_id = res_info["author_seq_id"]
                        atoms = res_info["atom_list"]
                        sb.init_residue(resname, *res_id, author_seq_id=author_seq_id)
                        for atom in atoms:
                            sb.add_atom(atom, sb.residue)
                            if atom.orig_serial_number <= len(all_anisou):
                                anisou = all_anisou[atom.orig_serial_number-1]
                                u = (
                                    anisou.U11, anisou.U12, anisou.U13,
                                    anisou.U22, anisou.U23, anisou.U33,
                                )
                                atom.set_anisou(np.array(u, "f"))
                if isinstance(sb.chain, Heterogens):
                    sb.chain.update()
                if isinstance(sb.chain, PolymerChain):
                    sb.chain.reset_disordered_residues()
                    sb.chain.sort_residues()

            if self.use_bio_assembly:
                self._execute_symmetry_operations(sb.model)

        sb.structure.set_pdb_id(structure_id)
        self.add_cell_and_symmetry_info()
        self.add_connect_record()
        
    def add_cell_and_symmetry_info(self):
        cell = self.cifdict.create_namedtuples('cell', single_value=True)
        symmetry = self.cifdict.create_namedtuples('symmetry', single_value=True)

        if not (
            cell and symmetry and hasattr(symmetry, "space_group_name_H_M")
        ):
            return
        
        a = float(cell.length_a)
        b = float(cell.length_b)
        c = float(cell.length_c)
        alpha = float(cell.angle_alpha)
        beta = float(cell.angle_beta)
        gamma = float(cell.angle_gamma)
        cell_data = np.array((a, b, c, alpha, beta, gamma), "f")
        spacegroup = symmetry.space_group_name_H_M #Hermann-Mauguin space-group symbol
        self._structure_builder.set_symmetry(spacegroup, cell_data)

    def add_connect_record(self):
        if 'struct_conn' not in self.cifdict:
            return
        struct_conn = self.cifdict.create_namedtuples('struct_conn')
        label_dict = {
            'chain': 'ptnr{}_label_asym_id',
            'resname': 'ptnr{}_label_comp_id',
            'resseq': 'ptnr{}_label_seq_id',
            'atom_id': 'ptnr{}_label_atom_id',
            'altloc': 'pdbx_ptnr{}_label_alt_id'
        }

        conn_dict = {}
        for connect in struct_conn:
            conn_type = connect.conn_type_id
            if conn_type not in conn_dict:
                conn_dict[conn_type] = []
            cur_connect = []
            for i in (1,2):
                cur_label_dict = {k: v.format(i) for k, v in label_dict.items()}
                cur_connect.append({
                    k: getattr(connect, v) for k, v in cur_label_dict.items()
                })
            conn_dict[conn_type].append(tuple(cur_connect))
        self._structure_builder.set_connect(conn_dict)