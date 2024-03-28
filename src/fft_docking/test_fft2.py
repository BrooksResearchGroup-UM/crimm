import pickle
import numpy as np
# from scipy.signal import correlate
from crimm import fft_docking
from crimm.Docking.GridGenerator import ReceptorGridGenerator, ProbeGridGenerator
from crimm.Data.probes.probes import create_new_probe_set
# from tqdm.contrib.concurrent import process_map
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BondType as rdBond
from rdkit.Geometry import Point3D
import time

PROBE_NAME = 'ethanol'
SPACING = 1.0
ROTATION_LEVEL = 2
N_TOP_POSES = 2000
N_THREADS = 12
REDUCE_SAMPLE_FACTOR = 10


with open('/home/truman/crimm/notebooks/2HYY.pkl', 'rb') as fh:
    new_chain = pickle.load(fh)

grid_gen = ReceptorGridGenerator(grid_spacing=SPACING, padding=8.0)
grid_gen.load_entity(new_chain, grid_shape='convex_hull')

vdw_attr_grid = grid_gen.get_attr_vdw_grid()
vdw_rep_grid = grid_gen.get_rep_vdw_grid()
elec_grid = grid_gen.get_elec_grid()

# rdbond_order_dict = {
#     'single': rdBond.SINGLE,
#     'double': rdBond.DOUBLE,
#     'triple': rdBond.TRIPLE,
#     'quadruple': rdBond.QUADRUPLE,
#     'aromatic': rdBond.AROMATIC,
# }
# def get_rdkit_bond_order(bo_name):
#     return rdbond_order_dict.get(bo_name, rdBond.OTHER)

# def create_rdkit_mol_from_probe(probe):
#     edmol = Chem.EditableMol(Chem.Mol())
#     rd_atoms = {}
#     conf = Chem.Conformer()
#     conf.Set3D(True)
#     for atom in probe.atoms:
#         rd_atom = Chem.Atom(atom.element.capitalize())
#         atom_idx = edmol.AddAtom(rd_atom)
#         rd_atoms[atom.name] = atom_idx
#         coord = Point3D(*atom.coord)
#         conf.SetAtomPosition(atom_idx, coord)
#     for bond in probe.bonds:
#         a1, a2 = bond[0].name, bond[1].name
#         idx1, idx2 = rd_atoms[a1], rd_atoms[a2]
#         bo = get_rdkit_bond_order(bond.type)
#         edmol.AddBond(idx1, idx2, bo)

#     mol = edmol.GetMol()
#     mol.AddConformer(conf)
#     AllChem.SanitizeMol(mol)
#     mol = AllChem.AddHs(mol, addCoords=True)
#     AllChem.SanitizeMol(mol)
#     AllChem.ComputeGasteigerCharges(mol)
#     Chem.AssignStereochemistryFrom3D(mol)
#     return mol

probe_set = create_new_probe_set()
probe = probe_set[PROBE_NAME]
grid_gen_probe = ProbeGridGenerator(
    grid_spacing=SPACING, rotation_search_level=ROTATION_LEVEL
)
grid_gen_probe.load_probe(probe)
probe_grids = grid_gen_probe.generate_grids()

kernels = probe_grids
signal = np.stack((
    grid_gen.get_elec_grid(),
    grid_gen.get_attr_vdw_grid(),
    grid_gen.get_rep_vdw_grid()
)).astype(np.float32)


time1 = time.time()
result = fft_docking.fft_correlate(signal, kernels, N_THREADS)
time2 = time.time()
print(round(time2 - time1, 3), 's')

# time1 = time.time()
# effective_grid_ids = grid_gen.coord_grid.grid_ids_in_box
# result = result.reshape((result.shape[0], -1))
# result = np.ascontiguousarray(result[:,effective_grid_ids], dtype=np.float32)
# top_scores2, pose_id, orientation_id = fft_correlate.rank_poses(
#     result, N_TOP_POSES, REDUCE_SAMPLE_FACTOR, N_THREADS
# )
# time2 = time.time()
# pose_id = effective_grid_ids[pose_id]

# print(round(time2 - time1, 3), 's')
# print(top_scores2[:20])
# print(pose_id[:20])
# print(orientation_id[:20])

# grid_gen.save_dx('output/test_all_grid.dx', result.flatten(), True)
# grid_gen.save_dx('output/test_all_grid2.dx', result.flatten(), False)

# selected_ori_coord = grid_gen_probe.rotated_coords[orientation_id]
# dists_to_recep_grid = grid_gen.bounding_box_grid.coords[pose_id]
# probe_origins = (selected_ori_coord.max(1) + selected_ori_coord.min(1))/2
# offsets = dists_to_recep_grid + probe_origins
# conf_coords = selected_ori_coord+offsets[:,np.newaxis,:]
# mol = create_rdkit_mol_from_probe(probe)

# for conf_coord in conf_coords:
#     conf = Chem.Conformer()
#     conf.Set3D(True)
#     for atom_idx, coord in enumerate(conf_coord):
#         coord = Point3D(*coord)
#         conf.SetAtomPosition(atom_idx, coord)
#     mol.AddConformer(conf, assignId=True)

# with AllChem.SDWriter(f'output/{probe.resname}_top_2000.sdf') as writer:
#     for conf in mol.GetConformers():
#         writer.write(mol, confId=conf.GetId())