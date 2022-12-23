from Bio.PDB.MMCIFParser import MMCIFParser
from ChMMCIF2Dict import ChMMCIF2Dict
from ChmStructureBuilder import ChmStructureBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from itertools import zip_longest
import numpy
import warnings

class ChMMCIFParser:
    def __init__(self, structure_builder = None, QUIET = False):
        if structure_builder == None:
            self._structure_builder = ChmStructureBuilder()
        else:
            self._structure_builder = structure_builder
        self.QUIET = QUIET

    def _get_header(self):
        resolution_keys = [
            ("refine", "ls_d_res_high"),
            ("refine_hist", "d_res_high"),
            ("em_3d_reconstruction", "resolution"),
        ]
        for key, subkey in resolution_keys:
            resolution = self._mmcif_dict.level_two_get(key, subkey)
            if resolution is not None:
                break
                
        self.header = {
            "name": self._mmcif_dict.get("data"),
            "head": self._mmcif_dict.get("struct_keywords"),
            "idcode": self._mmcif_dict.get('struct'),
            "deposition_date": self._mmcif_dict.level_two_get(
                "pdbx_database_status", "recvd_initial_deposition_date"
            ),
            "structure_method": self._mmcif_dict.level_two_get(
                "exptl", "method"
            ),
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
            self._mmcif_dict = ChMMCIF2Dict(filename)
            self._build_structure(structure_id)
            self._structure_builder.set_header(self._get_header())

        return self._structure_builder.get_structure()
    
    @staticmethod
    def _assign_hetflag(fieldname, resname):
        if fieldname != "HETATM":
            return ' '
        if resname in ("HOH", "WAT"):
            return "W"
        return "H"
    
    def _build_structure(self, structure_id):

        all_atoms = self._mmcif_dict.create_namedtuple('atom_site')
        coords = self._mmcif_dict.find_atom_coords()
        all_anisou = self._mmcif_dict.create_namedtuple('atom_site_anisotrop')
        cell = self._mmcif_dict.create_namedtuple('cell')
        symmetry = self._mmcif_dict.create_namedtuple('symmetry')
        
        if structure_id is None:
            structure_id = self._mmcif_dict['data']
        self._structure_builder.init_structure(structure_id)
        self._structure_builder.init_seg(" ")

        current_model_id = -1
        current_serial_id = -1
        cur_id_diff = 0

        current_chain_id = None
        current_residue_id = None
        current_resname = None

        all_data = zip_longest(all_atoms, coords, all_anisou)
        for i, (atom_site, coord, anisou) in enumerate(all_data):
            # set the line_counter for 'ATOM' lines only and not
            # as a global line counter found in the PDBParser()
            self._structure_builder.set_line_counter(i)
            
            model_serial_id = atom_site.pdbx_PDB_model_num
            if current_serial_id != model_serial_id:
                # if serial changes, update it and start new model
                current_serial_id = model_serial_id
                current_model_id += 1
                self._structure_builder.init_model(current_model_id, current_serial_id)
                current_chain_id = None
                current_residue_id = None
                current_resname = None
            
            chainid = atom_site.label_asym_id
            if current_chain_id != chainid:
                self._structure_builder.reset_chain_disordered_res()
                current_chain_id = chainid
                self._structure_builder.init_chain(current_chain_id)
                current_residue_id = None
                current_resname = None
            
            resname = atom_site.label_comp_id
            hetatm_flag = self._assign_hetflag(atom_site.group_PDB, resname)
            cur_auth_seq = int(atom_site.auth_seq_id)
            if atom_site.label_seq_id == None:
                int_resseq = cur_id_diff + cur_auth_seq
            else:
                int_resseq = int(atom_site.label_seq_id)
                cur_id_diff = int_resseq - cur_auth_seq
            if atom_site.pdbx_PDB_ins_code == None:
                icode = ' '
            else:
                icode = atom_site.pdbx_PDB_ins_code
            res_id = (hetatm_flag, int_resseq, icode)
            if current_residue_id != res_id or current_resname != resname:
                current_residue_id = res_id
                current_resname = resname
                self._structure_builder.init_residue(resname, hetatm_flag, int_resseq, icode)
            
            # Reindex the atom serial number
            atom_serial = i+1
            if atom_site.label_alt_id == None:
                altloc = ' '
            else:
                altloc = atom_site.label_alt_id
            
            self._structure_builder.init_atom(
                name = atom_site.label_atom_id,
                fullname = atom_site.label_atom_id,
                coord = coord,
                b_factor = float(atom_site.B_iso_or_equiv),
                occupancy = float(atom_site.occupancy),
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
                self._structure_builder.set_anisou(numpy.array(u, "f"))
        
        if cell and symmetry and hasattr(symmetry, "space_group_name_H_M"):
            a = float(cell.length_a)
            b = float(cell.length_b)
            c = float(cell.length_c)
            alpha = float(cell.angle_alpha)
            beta = float(cell.angle_beta)
            gamma = float(cell.angle_gamma)
            cell_data = numpy.array((a, b, c, alpha, beta, gamma), "f")
            spacegroup = symmetry.space_group_name_H_M #Hermann-Mauguin space-group symbol
            self._structure_builder.set_symmetry(spacegroup, cell_data)