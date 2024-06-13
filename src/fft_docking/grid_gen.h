#ifndef GRID_GEN_H_
#define GRID_GEN_H_

typedef struct{
    int x, y, z;
} Dim3d;

typedef struct{
    float x, y, z;
} Vector3d;

typedef struct{
    float w, x, y, z;
} Quaternion;

// Calculate pairwise distance between grid points and coordinates
void calc_grid_coord_pairwise_dist(
    const float* grid_pos, const float* coords, const int N_coords, 
    const int N_grid_points, float* dists
);

// Generate protein grids
void gen_all_grids(
    const float* grid_pos, const float* coords, const float* charges, 
    const float* epsilons, const float* vdw_rs, const float cc_elec, 
    const float rad_dielec_const, const float elec_rep_max, 
    const float elec_attr_max, const float vwd_rep_max,
    const float vwd_attr_max, const int N_coords, const int N_grid_points,
    const int is_constant_dielectric,
    float* electrostat_grid, float* vdw_grid_attr, float* vdw_grid_rep
);

// Find the maximum pairwise distance within a set of coordinates
float get_max_pairwise_dist(const float* coords, const int N_coords);

// Batch rotate coordinates based on quaternions
void batch_quaternion_rotate(
    const float* coords, const int N_coords, const float* quats, 
    const int N_quats, float* coords_rotated
);

// Calculate vdw energy factors from epsilon and vdw r_min
void calc_vdw_energy_factors(
    const float* epsilons, const float* vdw_rs, const int N_coords, 
    float* vdw_attr_factors, float* vdw_rep_factors
);

// Rotate and generate ligand grids
void rotate_gen_lig_grids(
    const float grid_spacing, const float* charges,
    const float* vdw_attr_factors, const float* vdw_rep_factors,
    const float* coords, const int N_coords,
    const float* quats, const int N_quats, const int cube_dim, 
    float* rot_coords, float* elec_grids, 
    float* vdw_grids_attr, float* vdw_grids_rep
);

#endif // GRID_GEN_H_
