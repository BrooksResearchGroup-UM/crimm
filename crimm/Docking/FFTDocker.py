import numpy as np
# This is a C extension module compiled from src/fft_docking/py_bindings.c
from crimm import fft_docking
from .GridGenerator import ReceptorGridGenerator, ProbeGridGenerator
from crimm.Data.probes.probes import create_new_probe_set
from .GridShapes import CubeGrid, ConvexHullGrid, BoundingBoxGrid, TruncatedSphereGrid

class FFTDocker:
    def __init__(
            self, grid_spacing=1.0, receptor_padding=8.0,
            effective_grid_shape='convex_hull', rad_dielec_const=2.0,
            elec_rep_max=40, elec_attr_max=-20,
            vdw_rep_max=2.0, vdw_attr_max=-1.0, use_constant_dielectric=False,
            rotation_level=2, n_top_poses=2000, reduce_sample_factor=10,
            n_threads=None
        ):
        if n_threads is None:
            import multiprocessing
            n_threads = multiprocessing.cpu_count()
        self.n_threads = n_threads
        self.grid_spacing = grid_spacing
        self.receptor_padding = receptor_padding
        self.effective_grid_shape = effective_grid_shape
        self.rad_dielec_const = rad_dielec_const
        self.elec_rep_max = elec_rep_max
        self.elec_attr_max = elec_attr_max
        self.vdw_rep_max = vdw_rep_max
        self.vdw_attr_max = vdw_attr_max
        self.use_constant_dielectric = use_constant_dielectric
        self.rotation_level = rotation_level
        self.n_top_poses = n_top_poses
        self.reduce_sample_factor = reduce_sample_factor
        ## Grids to be generated
        self.probe_grids = None
        self.receptor_grids = None
        self.recep_gen = None
        self.probe_gen = None
        ## These are the outputs of the docking
        self.conf_coords = None
        self.pose_id = None
        self.orientation_id = None
        self.top_scores = None
        self.total_energy = None
        self.result = None

    def load_receptor(self, receptor):
        self.recep_gen = ReceptorGridGenerator(
            grid_spacing=self.grid_spacing,
            padding=self.receptor_padding,
            rad_dielec_const=self.rad_dielec_const,
            elec_rep_max=self.elec_rep_max,
            elec_attr_max=self.elec_attr_max,
            vdw_rep_max=self.vdw_rep_max,
            vdw_attr_max=self.vdw_attr_max,
            use_constant_dielectric=False
        )
        self.recep_gen.load_entity(receptor, grid_shape=self.effective_grid_shape)
        self.receptor_grids = self.recep_gen.get_potential_grids()


    def load_probe(self, probe):
        self.probe_gen = ProbeGridGenerator(
            grid_spacing=self.grid_spacing,
            rotation_search_level=self.rotation_level
        )
        self.probe_gen.load_probe(probe)
        self.probe_grids = self.probe_gen.get_param_grids()

    # TODO: Implement batch splitting for large number of poses
    def dock(self):
        if self.receptor_grids is None or self.probe_grids is None:
            raise ValueError('Receptor and Probe must be loaded before docking')

        self.result = fft_docking.fft_correlate(
            self.receptor_grids, self.probe_grids, self.n_threads
        )
        self.total_energy = fft_docking.sum_grids(self.result)

    def dock_single_pose(self, pose_coords):
        if self.receptor_grids is None or self.probe_grids is None:
            raise ValueError('Receptor and Probe must be loaded before docking')

        pose_grids = self.probe_gen.generate_grids_single_pose(pose_coords)
        result = fft_docking.fft_correlate(
            self.receptor_grids, np.expand_dims(pose_grids,0), self.n_threads
        )
        return np.squeeze(result)

    def rank_poses(self):
        self.top_scores, self.pose_id, self.orientation_id = fft_docking.rank_poses(
            self.total_energy, 
            top_n_poses=self.n_top_poses,
            sample_factor=self.reduce_sample_factor,
            n_threads=self.n_threads
        )
        self.conf_coords = self._get_conf_coords(self.pose_id, self.orientation_id)
        return self.top_scores, self.conf_coords

    def _get_conf_coords(self, pose_id, orientation_id):
        selected_ori_coord = self.probe_gen.rotated_coords[orientation_id]
        coord_grid = self.recep_gen.coord_grid
        if isinstance(coord_grid, CubeGrid):
            dists_to_recep_grid = coord_grid.coords[pose_id]
        else:
            dists_to_recep_grid = self.recep_gen.bounding_box_grid.coords[pose_id]
        probe_origins = (selected_ori_coord.max(1) + selected_ori_coord.min(1))/2
        offsets = dists_to_recep_grid + probe_origins
        conf_coords = selected_ori_coord+offsets[:,np.newaxis,:]

        return conf_coords