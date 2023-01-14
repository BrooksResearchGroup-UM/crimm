import warnings
from itertools import zip_longest
import numpy
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Atom import Atom
from ChMMCIF2Dict import ChMMCIF2Dict
from ChmStructureBuilder import ChmStructureBuilder
from Chain import Chain, PolymerChain, Heterogens, Oligosaccharide, Solvent, Macrolide
from Model import Model


class ChMMCIFParser:
    def __init__(self, sturcture_builder = None, QUIET = False):
        if sturcture_builder is None:
            sturcture_builder = ChmStructureBuilder()
        self._structure_builder = sturcture_builder
        self.QUIET = QUIET
        self.cifdict = None
        self.header = None
        self.model_template = None

    def _get_header(self):
        resolution_keys = [
            ("refine", "ls_d_res_high"),
            ("refine_hist", "d_res_high"),
            ("em_3d_reconstruction", "resolution"),
        ]
        for key, subkey in resolution_keys:
            resolution = self.cifdict.level_two_get(key, subkey)
            if resolution is not None and resolution[0] is not None:
                break
                
        self.header = {
            "name": self.cifdict.get("data"),
            "keywords": self.cifdict.retrieve_single_value_dict("struct_keywords"),
            "citation": self.cifdict.get("citation"),
            "idcode": self.cifdict.retrieve_single_value_dict('struct'),
            "deposition_date": self.cifdict.level_two_get(
                "pdbx_database_status", "recvd_initial_deposition_date"
            )[0],
            "structure_method": self.cifdict.level_two_get(
                "exptl", "method"
            )[0],
            "resolution": resolution,
        }

        return self.header
    
    def get_structure(self, filename, structure_id = None):
        """Return the structure.

        Arguments:
         - structure_id - string, the id that will be used for the structure
         - filename - name of mmCIF file, OR an open text mode file handle

        """
        with warnings.catch_warnings():
            if self.QUIET:
                warnings.filterwarnings("ignore", category=PDBConstructionWarning)
            self.cifdict = ChMMCIF2Dict(filename)
            self.model_template = self.create_model_template()
            self._build_structure(structure_id)
            self._structure_builder.set_header(self._get_header())

        return self._structure_builder.get_structure()
    
    def create_polymer_chain_dict(self):
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
            auth_chain_ids = entity.pdbx_strand_id.split(',')
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

    def set_chain_attr(self, entity, chain):
        for k, v in entity._asdict().items():
            if k != 'id' and k != 'type' and v is not None:
                setattr(chain, k, v)

    def create_model_template(self, strict_parser = True):
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
            elif not strict_parser:
                cur_chain = Chain(entity.id)
            else:
                raise TypeError(f'Unknown Chain Type: {entity.type}')
            self.set_chain_attr(entity, cur_chain)
            model_temp.add(cur_chain)
        return model_temp

    def create_atom_from_namedtuple(self, atom_entry):
        """Create an Atom object from the namedtuple of an "atom_site" entry
        from the mmCIF dict

        :param atom_entry: A namedtuple created from mmCIF "atom_site" entries
                           with field names as attributes in the tuple
        :type atom_entry: namedtuple
        """
        coord = numpy.array(
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
            model_num = atom_entry.pdbx_PDB_model_num
            if model_num not in model_dict:
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



    def _build_structure(self, structure_id):
        sb = self._structure_builder
        self.model_template = self.create_model_template()
        all_anisou = self.cifdict.create_namedtuples('atom_site_anisotrop')
        if structure_id is None:
            structure_id = self.cifdict['data']
        sb.init_structure(structure_id)
        sb.init_seg(" ")

        atom_site = self.create_atom_site_entry_dict()

        for model_id, entity_dict in atom_site.items():
            model = Model(model_id)
            sb.structure.add(model)
            sb.model = model
            for entity_id, chain_dict in entity_dict.items():
                chain = self.model_template[entity_id].copy()
                model.add(chain)
                sb.chain = chain
                for chain_id, res_dict in chain_dict.items():
                    # This is the mmCIF label chain id
                    chain.id = chain_id
                    for resseq, res_info in res_dict.items():
                        resname = res_info["resname"]
                        res_id = res_info["res_id"]
                        atoms = res_info["atom_list"]
                        sb.init_residue(resname, *res_id)
                        for atom in atoms:
                            sb.add_atom(atom)
                            if atom.serial_number <= len(all_anisou):
                                anisou = all_anisou[atom.id-1]
                                u = (
                                    anisou.U11,
                                    anisou.U12,
                                    anisou.U13,
                                    anisou.U22,
                                    anisou.U23,
                                    anisou.U33,
                                )
                                sb.set_anisou(numpy.array(u, "f"))
                if isinstance(sb.chain, PolymerChain):
                    sb.chain.reset_disordered_residues()
                    sb.chain.update()

        self.add_cell_and_symmetry_info()
        
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
        cell_data = numpy.array((a, b, c, alpha, beta, gamma), "f")
        spacegroup = symmetry.space_group_name_H_M #Hermann-Mauguin space-group symbol
        self._structure_builder.set_symmetry(spacegroup, cell_data)