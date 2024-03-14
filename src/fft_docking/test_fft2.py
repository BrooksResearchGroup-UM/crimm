import pickle
import numpy as np
# from scipy.signal import correlate
import fft_correlate
from crimm.GridGen.GridGenerator import ReceptorGridGenerator, ProbeGridGenerator
from crimm.Data.probes.probes import create_new_probe_set
# from tqdm.contrib.concurrent import process_map
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BondType as rdBond
from rdkit.Geometry import Point3D
import time

PROBE_NAME = 'benzaldehyde'
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
probe_grid_dim = np.array(grid_gen_probe.grids.shape[-3:])
roll_steps = np.ceil(probe_grid_dim/2).astype(int)
roll_x, roll_y, roll_z = roll_steps
assert roll_x == roll_y == roll_z

kernels = probe_grids
signal = np.stack((
    grid_gen.get_elec_grid(),
    grid_gen.get_attr_vdw_grid(),
    grid_gen.get_rep_vdw_grid()
)).astype(np.float32)

n_orientation, n_grids, x,y,z = kernels.shape
result = np.zeros((len(kernels), *signal.shape), dtype=np.float32)
result[:,:,:x,:y,:z] = kernels

fft_correlate.fft_correlate_batch(signal, result, N_THREADS)
total_ener = np.zeros((n_orientation, *signal.shape[-3:]), dtype=np.float32)

fft_correlate.sum_grids(result, roll_x, total_ener)
print(total_ener.shape)


# time1 = time.time()
# orientation_ids, grid_ids, top_ener = rank_results(total_ener.reshape(total_ener.shape[0],-1))[:10]
# time2 = time.time()
# print(round(time2 - time1, 3), 's')

# print(top_ener[:20])
# print(grid_ids[:20])
# print(orientation_ids[:20])

# time1 = time.time()
# top_scores, pose_id, orientation_id = fft_correlate.rank_scores(
#     total_ener, N_TOP_POSES, N_THREADS
# )
# time2 = time.time()

# print(round(time2 - time1, 3), 's')
# print(top_scores[:20])
# print(pose_id[:20])
# print(orientation_id[:20])


time1 = time.time()
effective_grid_ids = grid_gen.coord_grid.grid_ids_in_box
total_ener = total_ener.reshape((total_ener.shape[0], -1))
total_ener = np.ascontiguousarray(total_ener[:,effective_grid_ids], dtype=np.float32)
top_scores2, pose_id, orientation_id = fft_correlate.rank_poses(
    total_ener, N_TOP_POSES, REDUCE_SAMPLE_FACTOR, N_THREADS
)
time2 = time.time()
pose_id = effective_grid_ids[pose_id]

print(round(time2 - time1, 3), 's')
print(top_scores2[:20])
print(pose_id[:20])
print(orientation_id[:20])

# print(np.allclose(top_scores, top_scores, atol=1e-2))

# result = np.flip(result, axis=(-3,-2,-1))
# result = np.roll(result, roll_steps, axis=(-3,-2,-1))

# total_ener2 = result.sum(1)
# print(total_ener[:3])
# print(total_ener2[:3])
# print(np.allclose(total_ener, total_ener2, atol=1e-2))
# print(total_ener.shape)
grid_gen.save_dx('output/test_all_grid.dx', total_ener.sum(0).flatten(), True)
# grid_gen.save_dx('output/test_all_grid2.dx', total_ener2.sum(0).flatten(), False)
# scipy_result = []
# for kernel in kernels:
#     results = []
#     for s, k in zip(signal, kernel):
#         results.append(correlate(s, k, mode='same', method='fft'))
#     scipy_result.append(np.asarray(results))
# scipy_result = np.array(scipy_result)

# def scipy_correlate(kernel):
#     results = []
#     for s, k in zip(signal, kernel):
#         results.append(correlate(s, k, mode='same', method='fft'))
#     return np.asarray(results)

# result_scipy = process_map(scipy_correlate, kernels, max_workers=4, chunksize=1)
# result_scipy = np.array(result_scipy)
# scipy_total_ener = result_scipy.sum(1)
# print(scipy_total_ener.shape)
# grid_gen.save_dx('output/test_all_grid_scipy.dx', scipy_total_ener.sum(0).flatten(), False)

# print(np.allclose(result, result_scipy, atol=1e-2))
# print(np.max(np.abs(result - result_scipy)))
# print(np.min(result))
# print(np.min(result_scipy))

# print(np.argmin(result))
# print(np.argmin(result_scipy))

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