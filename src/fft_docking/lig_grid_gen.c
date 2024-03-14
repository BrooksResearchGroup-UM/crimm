#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#ifdef USE_DOUBLES // use -DUSE_DOUBLES=1 to compile with double precision floats
typedef double user_float_t;
#else
typedef float user_float_t;
#endif

typedef struct{
    int x, y, z;
} Dim3d;

typedef struct{
    user_float_t x, y, z;
} Vector3d;

typedef struct{
    user_float_t w, x, y, z;
} Quaternion;

Vector3d quaternion_rotate(Quaternion q, Vector3d v){
    Vector3d dest;

    user_float_t qxsq = q.x * q.x;
    user_float_t qysq = q.y * q.y;
    user_float_t qzsq = q.z * q.z;

    user_float_t qxqy = q.x * q.y;
    user_float_t qxqz = q.x * q.z;
    user_float_t qyqz = q.y * q.z;
    user_float_t qwsq = q.w * q.w;

    dest.x = v.x * (qwsq + qxsq - qysq - qzsq) +
            2* (v.y * (qxqy - q.w*q.z) + v.z * (qxqz + q.w*q.y));

    dest.y = v.y * (qwsq - qxsq + qysq - qzsq) +
            2*(v.x * (qxqy + q.w*q.z) + v.z * (qyqz - q.w*q.x));

    dest.z = v.z * (qwsq - qxsq - qysq + qzsq) +
            2*(v.x * (qxqz - q.w*q.y) + v.y * (qyqz + q.w*q.x));

    return dest;
}

void batch_quatornion_rotate(
    const user_float_t* quats, const user_float_t* coords, const int N_quats, 
    const int N_coords, user_float_t* coords_rotated
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

Vector3d get_min_coord(
    const user_float_t* grid_pos, const int N_grid_points
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

void gen_lig_grid(
    const user_float_t grid_spacing, const user_float_t* charges,
    const user_float_t* vdw_attr_factors, const user_float_t* vdw_rep_factors,
    user_float_t* coords, const int N_coords, const int grid_dim, 
    user_float_t* elec_grid, user_float_t* vdw_grid_attr, user_float_t* vdw_grid_rep
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
            (int) floor(d_grid_coord.x / grid_spacing),
            (int) floor(d_grid_coord.y / grid_spacing),
            (int) floor(d_grid_coord.z / grid_spacing)
        };
        Vector3d grid_coord_frac = {
            d_grid_coord.x / grid_spacing - grid_coord_idx.x,
            d_grid_coord.y / grid_spacing - grid_coord_idx.y,
            d_grid_coord.z / grid_spacing - grid_coord_idx.z
        };
        user_float_t charge = charges[i];
        user_float_t vdw_attr_factor = vdw_attr_factors[i];
        user_float_t vdw_rep_factor = vdw_rep_factors[i];
        for (int x = 0; x < 2; x++){
            for (int y = 0; y < 2; y++){
                for (int z = 0; z < 2; z++){
                    int grid_idx = (
                        (grid_coord_idx.x + x) * grid_dim * grid_dim +
                        (grid_coord_idx.y + y) * grid_dim +
                        (grid_coord_idx.z + z)
                    );
                    user_float_t frac = (
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

void calc_vdw_energy_factors(
    const user_float_t* epsilons, const user_float_t* vdw_rs, const int N_coords, 
    user_float_t* vdw_attr_factors, user_float_t* vdw_rep_factors
){
    user_float_t epsilon_sqrt, r_min_3;
    for (int i = 0; i < N_coords; i++){
        user_float_t vdw_r = vdw_rs[i];
        user_float_t epsilon = epsilons[i];
        epsilon_sqrt = sqrt(fabsf(epsilon));
        r_min_3 = pow(vdw_r, 3.0);
        vdw_attr_factors[i] = epsilon_sqrt * r_min_3;
        vdw_rep_factors[i] = epsilon_sqrt * pow(r_min_3, 2.0);
    }
}

void rotate_gen_lig_grids(
    const user_float_t grid_spacing, const user_float_t* charges,
    const user_float_t* vdw_attr_factors, const user_float_t* vdw_rep_factors,
    const user_float_t* coords, const int N_coords,
    const user_float_t* quats, const int N_quats, const int cube_dim, 
    user_float_t* rot_coords, user_float_t* elec_grids, 
    user_float_t* vdw_grids_attr, user_float_t* vdw_grids_rep
){
    int N_grid_points = cube_dim * cube_dim * cube_dim;
    for (int i = 0; i < N_quats; i++){
        user_float_t* cur_coords = rot_coords + i * N_coords * 3;
        Quaternion q = {quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]};
        for (int j = 0; j < N_coords; j++){
            Vector3d v = {coords[j * 3], coords[j * 3 + 1], coords[j * 3 + 2]};
            Vector3d v_rotated = quaternion_rotate(q, v);
            cur_coords[j * 3] = v_rotated.x;
            cur_coords[j * 3 + 1] = v_rotated.y;
            cur_coords[j * 3 + 2] = v_rotated.z;
        }
        user_float_t* elec_grid = elec_grids + i * N_grid_points;
        user_float_t* vdw_grid_attr = vdw_grids_attr + i * N_grid_points;
        user_float_t* vdw_grid_rep = vdw_grids_rep + i * N_grid_points;
        gen_lig_grid(
            grid_spacing, charges, vdw_attr_factors, vdw_rep_factors, 
            cur_coords, N_coords, cube_dim, 
            elec_grid, vdw_grid_attr, vdw_grid_rep
        );
    }
}