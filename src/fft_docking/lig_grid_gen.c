#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "grid_gen.h"

// Rotate a set of coordinates based on a quaternion
Vector3d quaternion_rotate(Quaternion q, Vector3d v){
    Vector3d dest;

    float qxsq = q.x * q.x;
    float qysq = q.y * q.y;
    float qzsq = q.z * q.z;

    float qxqy = q.x * q.y;
    float qxqz = q.x * q.z;
    float qyqz = q.y * q.z;
    float qwsq = q.w * q.w;

    dest.x = v.x * (qwsq + qxsq - qysq - qzsq) +
            2* (v.y * (qxqy - q.w*q.z) + v.z * (qxqz + q.w*q.y));

    dest.y = v.y * (qwsq - qxsq + qysq - qzsq) +
            2*(v.x * (qxqy + q.w*q.z) + v.z * (qyqz - q.w*q.x));

    dest.z = v.z * (qwsq - qxsq - qysq + qzsq) +
            2*(v.x * (qxqz - q.w*q.y) + v.y * (qyqz + q.w*q.x));

    return dest;
}

// Batch rotate coordinates based on quaternions
void batch_quaternion_rotate(
    const float* coords, const int N_coords, const float* quats, 
    const int N_quats, float* coords_rotated
){
    for (int i = 0; i < N_quats; i++){
        Quaternion q = {quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]};
        for (int j = 0; j < N_coords; j++){
            Vector3d v = {coords[j * 3], coords[j * 3 + 1], coords[j * 3 + 2]};
            Vector3d v_rotated = quaternion_rotate(q, v);
            coords_rotated[i * N_coords * 3 + j * 3] = v_rotated.x;
            coords_rotated[i * N_coords * 3 + j * 3 + 1] = v_rotated.y;
            coords_rotated[i * N_coords * 3 + j * 3 + 2] = v_rotated.z;
        }
    }
}

// Get the minimum coordinate of the grid
Vector3d get_min_coord(
    const float* grid_pos, const int N_grid_points
){  
    float diff;
    bool is_neg;
    Vector3d min_grid_coord = {grid_pos[0], grid_pos[1], grid_pos[2]};
    for (int i = 0; i < N_grid_points; i++){
        diff = grid_pos[i * 3] - min_grid_coord.x;
        is_neg = signbit(diff);
        min_grid_coord.x += is_neg * diff;
        diff = grid_pos[i * 3 + 1] - min_grid_coord.y;
        is_neg = signbit(diff);
        min_grid_coord.y += is_neg * diff;
        diff = grid_pos[i * 3 + 2] - min_grid_coord.z;
        is_neg = signbit(diff);
        min_grid_coord.z += is_neg * diff;
    }
    return min_grid_coord;
}

// Find the maximum pairwise distance within a set of coordinates
float get_max_pairwise_dist(const float* coords, const int N_coords) {
    float max_dist = 0.0;
    float diff;
    bool is_neg;
    for (int i = 0; i < N_coords; i++) {
        for (int j = i + 1; j < N_coords; j++) {
            float dx = coords[i * 3] - coords[j * 3];
            float dy = coords[i * 3 + 1] - coords[j * 3 + 1];
            float dz = coords[i * 3 + 2] - coords[j * 3 + 2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);
            diff = max_dist - dist;
            is_neg = signbit(diff);
            max_dist -= is_neg * diff;
        }
    }
    return max_dist;
}

// Generate ligand grid
void gen_lig_grid(
    const float grid_spacing, const float* charges,
    const float* vdw_attr_factors, const float* vdw_rep_factors,
    float* coords, const int N_coords, const int grid_dim, 
    float* elec_grid, float* vdw_grid_attr, float* vdw_grid_rep
){
    Vector3d min_grid_coord = get_min_coord(coords, N_coords);
    for (int i = 0; i < N_coords; i++){
        Vector3d coord = {coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]};
        // Distance between the ligand atom and the min grid coord
        Vector3d d_grid_coord = {
            coord.x - min_grid_coord.x,
            coord.y - min_grid_coord.y,
            coord.z - min_grid_coord.z
        };
        Dim3d grid_coord_idx = {
            (int) floorf(d_grid_coord.x / grid_spacing),
            (int) floorf(d_grid_coord.y / grid_spacing),
            (int) floorf(d_grid_coord.z / grid_spacing)
        };
        Vector3d grid_coord_frac = {
            d_grid_coord.x / grid_spacing - grid_coord_idx.x,
            d_grid_coord.y / grid_spacing - grid_coord_idx.y,
            d_grid_coord.z / grid_spacing - grid_coord_idx.z
        };
        float charge = charges[i];
        float vdw_attr_factor = vdw_attr_factors[i];
        float vdw_rep_factor = vdw_rep_factors[i];
        for (int x = 0; x < 2; x++){
            for (int y = 0; y < 2; y++){
                for (int z = 0; z < 2; z++){
                    int grid_idx = (
                        (grid_coord_idx.x + x) * grid_dim * grid_dim +
                        (grid_coord_idx.y + y) * grid_dim +
                        (grid_coord_idx.z + z)
                    );
                    float frac = (
                        (x == 0 ? 1.0 - grid_coord_frac.x : grid_coord_frac.x) *
                        (y == 0 ? 1.0 - grid_coord_frac.y : grid_coord_frac.y) *
                        (z == 0 ? 1.0 - grid_coord_frac.z : grid_coord_frac.z)
                    );
                    elec_grid[grid_idx] += charge * frac;
                    vdw_grid_attr[grid_idx] += vdw_attr_factor * frac;
                    vdw_grid_rep[grid_idx] += vdw_rep_factor * frac;
                }
            }
        }
    }
}

// Calculate vdw energy factors from epsilon and vdw r_min
void calc_vdw_energy_factors(
    const float* epsilons, const float* vdw_rs, const int N_coords, 
    float* vdw_attr_factors, float* vdw_rep_factors
){
    float epsilon_sqrt, r_min_3;
    for (int i = 0; i < N_coords; i++){
        float vdw_r = vdw_rs[i];
        float epsilon = epsilons[i];
        epsilon_sqrt = sqrtf(fabsf(epsilon));
        r_min_3 = powf(vdw_r, 3.0);
        vdw_attr_factors[i] = epsilon_sqrt * r_min_3;
        vdw_rep_factors[i] = epsilon_sqrt * r_min_3 * r_min_3;
    }
}

// Rotate and generate ligand grids
void rotate_gen_lig_grids(
    const float grid_spacing, const float* charges,
    const float* vdw_attr_factors, const float* vdw_rep_factors,
    const float* coords, const int N_coords,
    const float* quats, const int N_quats, const int cube_dim, 
    float* rot_coords, float* elec_grids, 
    float* vdw_grids_attr, float* vdw_grids_rep
){
    int N_grid_points = cube_dim * cube_dim * cube_dim;
    for (int i = 0; i < N_quats; i++){
        float* cur_coords = rot_coords + i * N_coords * 3;
        Quaternion q = {quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]};
        for (int j = 0; j < N_coords; j++){
            Vector3d v = {coords[j * 3], coords[j * 3 + 1], coords[j * 3 + 2]};
            Vector3d v_rotated = quaternion_rotate(q, v);
            cur_coords[j * 3] = v_rotated.x;
            cur_coords[j * 3 + 1] = v_rotated.y;
            cur_coords[j * 3 + 2] = v_rotated.z;
        }
        float* elec_grid = elec_grids + i * N_grid_points;
        float* vdw_grid_attr = vdw_grids_attr + i * N_grid_points;
        float* vdw_grid_rep = vdw_grids_rep + i * N_grid_points;
        gen_lig_grid(
            grid_spacing, charges, vdw_attr_factors, vdw_rep_factors, 
            cur_coords, N_coords, cube_dim, 
            elec_grid, vdw_grid_attr, vdw_grid_rep
        );
    }
}

