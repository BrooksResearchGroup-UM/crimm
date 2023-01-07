from Bio.PDB.MMCIFParser import MMCIFParser
from ChMMCIF2Dict import ChMMCIF2Dict
from ChmStructureBuilder import ChmStructureBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Chain import Chain, PolymerChain, Ligands, Saccharide, Solvent, Macrolide
from Model import Model
from itertools import zip_longest
import numpy
import warnings

class ChMMCIFParser:
    def __init__(self, sturcture_builder = None, QUIET = False):
        if sturcture_builder == None:
            sturcture_builder = ChmStructureBuilder()
        self._structure_builder = sturcture_builder
        self.QUIET = QUIET

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
            pchain = PolymerChain(
                entity_id,
                auth_chain_id,
                entity_info.type,
                entity_info.pdbx_seq_one_letter_code,
                entity_info.pdbx_seq_one_letter_code_can,
                reported_res_dict.get(entity_info.entity_id),
                missing_res_dict.get(chain_id)
            )
            polymer_dict[entity_id] = pchain
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
                cur_chain = Ligands(entity.id)
            elif entity.type == 'branched':
                cur_chain = Saccharide(entity.id)
            elif entity.type == 'water':
                cur_chain = Solvent(entity.id)
            elif entity.type == 'macrolide':
                cur_chain = Macrolide(entity.id)
            elif not strict_parser:
                cur_chain = Chain(entity.id)
            else:
                raise TypeError('Unknown Chain Type: {}'.format(entity.type))
            self.set_chain_attr(entity, cur_chain)
            model_temp.add(cur_chain)
        return model_temp

    @staticmethod
    def _assign_hetflag(fieldname, resname):
        if fieldname != "HETATM":
            return ' '
        if resname in ("HOH", "WAT"):
            return "W"
        return "H"

    def _build_structure(self, structure_id):
        sb = self._structure_builder
        model_template = self.create_model_template()

        atom_site = self.cifdict.create_namedtuples('atom_site')
        coords = self.cifdict.find_atom_coords()
        all_anisou = self.cifdict.create_namedtuples('atom_site_anisotrop')

        if structure_id is None:
            structure_id = self.cifdict['data']
        sb.init_structure(structure_id, model_template)
        sb.init_seg(" ")

        last_chain_id = None
        last_entity_id = None
        last_auth_seq = None
        het_chain_resseq = 0
        sb.chain = None
        current_resname = None
        current_res_id = None

        all_data = zip_longest(atom_site, coords, all_anisou)
        for i, (atom_site, coord, anisou) in enumerate(all_data):
            # set the line_counter for 'ATOM' lines only and not
            # as a global line counter found in the PDBParser()
            sb.set_line_counter(i)
            
            model_serial_id = atom_site.pdbx_PDB_model_num
            if model_serial_id not in sb.structure:
                # if serial changes, update it and start new model
                model = sb.init_model(model_serial_id)
                current_resname = None
                current_res_id = None
                het_chain_resseq = 0

            entity_id = atom_site.label_entity_id
            chain_id = atom_site.label_asym_id
            if entity_id != last_entity_id or \
                (isinstance(sb.chain, PolymerChain) and chain_id != last_chain_id ):
                last_chain_id = chain_id
                last_entity_id = entity_id
                if isinstance(sb.chain, PolymerChain):
                    sb.chain.reset_disordered_residues()
                    sb.chain.update()
                if chain_id not in model:
                    chain = sb.model_template[entity_id].copy()
                    chain.id = chain_id
                    model.add(chain)
                else:
                    chain = model[chain_id]
                sb.chain = chain

            resname = atom_site.label_comp_id
            hetatm_flag = self._assign_hetflag(atom_site.group_PDB, resname)
            auth_seq_id = atom_site.auth_seq_id
            auth_chain_id = atom_site.auth_asym_id
            # For heterogens that do not have resseq
            if atom_site.label_seq_id == None:
                # if the atom does not the same author seq id as the last atom,
                # they are in the separate molecules, and the resseq needs to increment
                if auth_seq_id != last_auth_seq or chain_id != last_chain_id:
                    last_chain_id = chain_id
                    het_chain_resseq += 1
                    int_resseq = het_chain_resseq
                    last_auth_seq = 1*auth_seq_id
            else:
                int_resseq = atom_site.label_seq_id
                
            if atom_site.pdbx_PDB_ins_code == None:
                icode = ' '
            else:
                icode = atom_site.pdbx_PDB_ins_code
            res_id = (hetatm_flag, int_resseq, icode)

            if res_id != current_res_id or current_resname != resname:
                current_res_id = res_id
                current_resname = resname
                sb.init_residue(resname, hetatm_flag, int_resseq, icode)

            # Reindex the atom serial number
            atom_serial = i+1
            if atom_site.label_alt_id == None:
                altloc = ' '
            else:
                # Convert any possible int to string to allow ord() when sorting
                # on the altlocs
                altloc = str(atom_site.label_alt_id)
            
            sb.init_atom(
                name = atom_site.label_atom_id,
                fullname = atom_site.label_atom_id,
                coord = coord,
                b_factor = atom_site.B_iso_or_equiv,
                occupancy = atom_site.occupancy,
                altloc = altloc,
                serial_number = atom_serial,
                element = atom_site.type_symbol,
            )
            
            if anisou:
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
        sb.set_symmetry(spacegroup, cell_data)