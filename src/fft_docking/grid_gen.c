#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "grid_gen.h"

void calc_grid_coord_pairwise_dist(
    const float* grid_pos, const float* coords, const int N_coords, 
    const int N_grid_points, float* dists
){
    float dx, dy, dz;
    #pragma omp parallel for private(dx, dy, dz)
    for (int i = 0; i < N_grid_points; i++) {
        for (int j = 0; j < N_coords; j++) {
            // pairwise distances between the receptor atom coords 
            // and the grid points
            dx = grid_pos[i * 3] - coords[j * 3];
            dy = grid_pos[i * 3 + 1] - coords[j * 3 + 1];
            dz = grid_pos[i * 3 + 2] - coords[j * 3 + 2];
            dists[i * N_coords + j] = sqrtf(dx * dx + dy * dy + dz * dz);
        }
    }
}

float calc_point_elec_potential_radial(
    const float dist_sq, const float elec_const, const float charge, 
    const float rc_sq, const float alpha, const float elec_rep_max, 
    const float elec_attr_max
) {
    float cur_potential;
    // beyond cutoff
    if (dist_sq > rc_sq) {
        cur_potential = elec_const / dist_sq;
    // within cutoff
    } else {
        float alpha_tmp = alpha * dist_sq;
        // repulsive
        if (charge > 0) {
            cur_potential = elec_rep_max - alpha_tmp;
        // attractive
        } else {
            cur_potential = elec_attr_max + alpha_tmp;
        }
        // zero charge is very unlikely and will
        // turn out to be zero potential in the end
    }
    return cur_potential;
}

float calc_point_elec_potential_constant(
    const float dist_sq, const float elec_const, const float charge, 
    const float rc_sq, const float alpha, const float elec_rep_max, 
    const float elec_attr_max
) {
    float cur_potential, dist;
    dist = sqrtf(dist_sq);
    // beyond cutoff
    if (dist_sq > rc_sq) {
        cur_potential = elec_const / dist;
    // within cutoff
    } else {
        float alpha_tmp = alpha * dist;
        // repulsive
        if (charge > 0) {
            cur_potential = elec_rep_max - alpha_tmp;
        // attractive
        } else {
            cur_potential = elec_attr_max + alpha_tmp;
        }
        // zero charge is very unlikely and will
        // turn out to be zero potential in the end
    }
    return cur_potential;
}

void gen_all_grids(
    const float* grid_pos, const float* coords, const float* charges, 
    const float* epsilons, const float* vdw_rs, const float cc_elec, 
    const float rad_dielec_const, const float elec_rep_max, 
    const float elec_attr_max, const float vwd_rep_max,
    const float vwd_attr_max, const int N_coords, const int N_grid_points,
    const int use_constant_dielectric,
    float* electrostat_grid, float* vdw_grid_attr, float* vdw_grid_rep
) {
    float dx, dy, dz, dist_sq;
    float r_min, eps_sqrt, vdwconst_attr, vdwconst_rep, r6;
    float beta_attr, beta_rep;
    float rc_sq_vdw_attr, rc_sq_vdw_rep;
    float charge, elec_const, emax_tmp, alpha, rc_sq_elec, cur_elec_potential;
    float (*calc_point_elec_potential)(
        float, float, float, float, 
        float, float, float
    );

    // set the electrostatic potential function
    if (use_constant_dielectric) {
        calc_point_elec_potential = &calc_point_elec_potential_constant;
    } else {
        calc_point_elec_potential = &calc_point_elec_potential_radial;
    }

    #pragma omp parallel for private( \
        dx, dy, dz, dist_sq, r_min, eps_sqrt, vdwconst_attr, vdwconst_rep, r6, \
        beta_attr, beta_rep, rc_sq_vdw_attr, rc_sq_vdw_rep, charge, elec_const, \
        emax_tmp, alpha, rc_sq_elec, cur_elec_potential \
    )
    for (int i = 0; i < N_coords; i++) {
        // calculate vdw constants
        r_min = vdw_rs[i];
        eps_sqrt = sqrtf(fabsf(epsilons[i]));
        vdwconst_attr = 0.25 * fabsf(vwd_attr_max) / eps_sqrt;
        vdwconst_rep = 0.5 * fabsf(vwd_rep_max) / eps_sqrt;
        // vdw cutoff distance squared
        rc_sq_vdw_attr = r_min * powf(vdwconst_attr, -1.0 / 3.0);
        rc_sq_vdw_rep = r_min * powf(vdwconst_rep, -1.0 / 6.0);
        // we use half of the beta since the rc and dist are squared
        beta_attr = -12.0 * eps_sqrt / vwd_attr_max * vdwconst_attr;
        beta_rep = 12.0 * eps_sqrt / vwd_rep_max * vdwconst_rep;
        // calculate electrostatic constants
        charge = charges[i];
        elec_const = cc_elec * charge / rad_dielec_const;
        if (charge > 0) {
            // repulsive
            emax_tmp = elec_rep_max;
        } else {
            // attractive
            emax_tmp = elec_attr_max;
        }
        // elec cutoff distance squared
        rc_sq_elec = 2.0 * fabsf(elec_const / emax_tmp);
        alpha = fabsf(emax_tmp / (2.0 * rc_sq_elec));

        for (int j = 0; j < N_grid_points; j++) {
            // pairwise distances squared between the receptor atom coords 
            // and the grid points
            dx = grid_pos[j * 3] - coords[i * 3];
            dy = grid_pos[j * 3 + 1] - coords[i * 3 + 1];
            dz = grid_pos[j * 3 + 2] - coords[i * 3 + 2];
            dist_sq = dx * dx + dy * dy + dz * dz;

            r6 = powf(r_min / dist_sq, 3.0); // (sqrt(r_min)/r)^6
            
            // repulsive vdw grid
            if (dist_sq > rc_sq_vdw_rep) {
                // beyond cutoff
                #pragma omp atomic
                vdw_grid_rep[j] += eps_sqrt * r6 * r6;
            } else {
                // within cutoff
                #pragma omp atomic
                vdw_grid_rep[j] += vwd_rep_max * 
                (1.0 - 0.5 * powf((dist_sq / rc_sq_vdw_rep), beta_rep));
            }
            // attractive vdw grid
            if (dist_sq > rc_sq_vdw_attr) {
                #pragma omp atomic
                vdw_grid_attr[j] -= 2.0 * eps_sqrt * r6;
            } else {
                // within cutoff
                #pragma omp atomic
                vdw_grid_attr[j] += vwd_attr_max * 
                (1.0 - 0.5 * powf((dist_sq / rc_sq_vdw_attr), beta_attr));
            }
            // electrostatic grid
            cur_elec_potential = (*calc_point_elec_potential)(
                dist_sq, elec_const, charge, rc_sq_elec, alpha,
                elec_rep_max, elec_attr_max
            );
            #pragma omp atomic
            electrostat_grid[j] += cur_elec_potential;
        } // end grid points loop
    } // end coords loop
}
