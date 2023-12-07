#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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

typedef struct{
    Vector3d min_coord;
    Vector3d ptp3d;
} BoxSpec;

typedef struct{
    Dim3d dim;
    int N_grid_points;
    Vector3d origin;
    user_float_t spacing;
    Vector3d* coords;
    user_float_t* lig_coords;
    user_float_t* elec_grid;
    user_float_t* vdw_grid_attr;
    user_float_t* vdw_grid_rep;
} Grid;

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

BoxSpec get_box_spec(const user_float_t* coords, const int N_coords){
    Vector3d max_vec = {0, 0, 0};
    Vector3d min_vec = {0, 0, 0};
    for (int i = 0; i < N_coords; i++){
        Vector3d coord = {coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]};
        if (coord.x > max_vec.x){
            max_vec.x = coord.x;
        }
        else if (coord.x < min_vec.x){
            min_vec.x = coord.x;
        }
        if (coord.y > max_vec.y){
            max_vec.y = coord.y;
        }
        else if (coord.y < min_vec.y){
            min_vec.y = coord.y;
        }
        if (coord.z > max_vec.z){
            max_vec.z = coord.z;
        }
        else if (coord.z < min_vec.z){
            min_vec.z = coord.z;
        }
    }
    Vector3d ptp3d = {
        max_vec.x - min_vec.x,
        max_vec.y - min_vec.y,
        max_vec.z - min_vec.z
    };
    BoxSpec box_spec = {min_vec, ptp3d};
    return box_spec;
}

Grid create_grid(
    const user_float_t spacing, const BoxSpec box_spec, 
    user_float_t* lig_coords
){
    Vector3d ptp3d = box_spec.ptp3d;
    Vector3d min_coord = box_spec.min_coord;
    Vector3d origin = {
        ptp3d.x / 2.0 + min_coord.x,
        ptp3d.y / 2.0 + min_coord.y,
        ptp3d.z / 2.0 + min_coord.z
    };

    Dim3d dim = {
        (int) ceil(ptp3d.x / spacing) + 1,
        (int) ceil(ptp3d.y / spacing) + 1,
        (int) ceil(ptp3d.z / spacing) + 1
    };
    int N_grid_points = dim.x * dim.y * dim.z;

    user_float_t len_x = dim.x * spacing;
    user_float_t len_y = dim.y * spacing;
    user_float_t len_z = dim.z * spacing;

    Grid grid;
    grid.dim = dim;
    grid.N_grid_points = N_grid_points;
    grid.origin = origin;
    grid.spacing = spacing;
    grid.coords = malloc(N_grid_points * sizeof(Vector3d));
    grid.lig_coords = lig_coords;
    grid.elec_grid = malloc(N_grid_points * sizeof(user_float_t));
    grid.vdw_grid_attr = malloc(N_grid_points * sizeof(user_float_t));
    grid.vdw_grid_rep = malloc(N_grid_points * sizeof(user_float_t));
    memset(grid.elec_grid, 0, N_grid_points * sizeof(user_float_t));
    memset(grid.vdw_grid_attr, 0, N_grid_points * sizeof(user_float_t));
    memset(grid.vdw_grid_rep, 0, N_grid_points * sizeof(user_float_t));

    for (int i = 0; i < N_grid_points; i++){
        Vector3d grid_coord = {
            origin.x - len_x / 2.0 + (i % dim.x) * spacing,
            origin.y - len_y / 2.0 + ((i / dim.x) % dim.y) * spacing,
            origin.z - len_z / 2.0 + (i / (dim.x * dim.y)) * spacing
        };
        grid.coords[i] = grid_coord;
    }
    return grid;
}

void dealloc_grid(Grid grid){
    free(grid.coords);
    free(grid.lig_coords);
    free(grid.elec_grid);
    free(grid.vdw_grid_attr);
    free(grid.vdw_grid_rep);
}

Grid gen_lig_grid(
    const user_float_t grid_spacing, const user_float_t* charges,
    const user_float_t* vdw_attr_factors, const user_float_t* vdw_rep_factors,
    user_float_t* coords, const int N_coords
){
    BoxSpec box_spec = get_box_spec(coords, N_coords);
    Vector3d min_grid_coord = box_spec.min_coord;
    Grid grid = create_grid(grid_spacing, box_spec, coords);
    
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
                        (grid_coord_idx.x + x) * grid.dim.y * grid.dim.z +
                        (grid_coord_idx.y + y) * grid.dim.z +
                        (grid_coord_idx.z + z)
                    );
                    user_float_t frac = (
                        (x == 0 ? 1.0 - grid_coord_frac.x : grid_coord_frac.x) *
                        (y == 0 ? 1.0 - grid_coord_frac.y : grid_coord_frac.y) *
                        (z == 0 ? 1.0 - grid_coord_frac.z : grid_coord_frac.z)
                    );
                    
                    grid.elec_grid[grid_idx] += charge * frac;
                    grid.vdw_grid_attr[grid_idx] += vdw_attr_factor * frac;
                    grid.vdw_grid_rep[grid_idx] += vdw_rep_factor * frac;
                }
            }
        }
    }
    return grid;
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

Grid* rotate_gen_lig_grids(
    const user_float_t grid_spacing, const user_float_t* charges,
    const user_float_t* vdw_attr_factors, const user_float_t* vdw_rep_factors,
    const user_float_t* coords, const int N_coords,
    const user_float_t* quats, const int N_quats
){
    Grid* grids = malloc(N_quats * sizeof(Grid));
    for (int i = 0; i < N_quats; i++){
        user_float_t* cur_coords = malloc(N_coords * 3 * sizeof(user_float_t));
        Quaternion q = {quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]};
        for (int j = 0; j < N_coords; j++){
            Vector3d v = {coords[j * 3], coords[j * 3 + 1], coords[j * 3 + 2]};
            Vector3d v_rotated = quaternion_rotate(q, v);
            cur_coords[j * 3] = v_rotated.x;
            cur_coords[j * 3 + 1] = v_rotated.y;
            cur_coords[j * 3 + 2] = v_rotated.z;
        }
        grids[i] = gen_lig_grid(
            grid_spacing, charges, vdw_attr_factors, vdw_rep_factors, 
            cur_coords, N_coords
        );
    }
    return grids;
}

Grid* rotate_gen_lig_grids_eps_rmin(
    const user_float_t grid_spacing, const user_float_t* charges,
    const user_float_t* epsilons, const user_float_t* vdw_rs,
    const user_float_t* coords, const int N_coords,
    const user_float_t* quats, const int N_quats
){
    user_float_t* vdw_attr_factors = malloc(N_coords * sizeof(user_float_t));
    user_float_t* vdw_rep_factors = malloc(N_coords * sizeof(user_float_t));
    calc_vdw_energy_factors(epsilons, vdw_rs, N_coords, vdw_attr_factors, vdw_rep_factors);
    Grid* grids = rotate_gen_lig_grids(
        grid_spacing, charges, vdw_attr_factors, vdw_rep_factors, 
        coords, N_coords, quats, N_quats
    );
    free(vdw_attr_factors);
    free(vdw_rep_factors);
    return grids;
}