#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// TODO: implement cdie on electrostatic potentials
#ifdef USE_DOUBLES // use -DUSE_DOUBLES=1 to compile with doubles
typedef double user_float_t;
#else
typedef float user_float_t;
#endif

void calc_pairwise_dist(
    const user_float_t* grid_pos, const user_float_t* coords, const int N_coords, 
    const int N_grid_points, user_float_t* dists
){
    user_float_t dx, dy, dz;
    #pragma omp parallel for private(dx, dy, dz)
    for (int i = 0; i < N_grid_points; i++) {
        for (int j = 0; j < N_coords; j++) {
            // pairwise distances between the receptor atom coords 
            // and the grid points
            dx = grid_pos[i * 3] - coords[j * 3];
            dy = grid_pos[i * 3 + 1] - coords[j * 3 + 1];
            dz = grid_pos[i * 3 + 2] - coords[j * 3 + 2];
            dists[i * N_coords + j] = sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
}

user_float_t calc_point_elec_potential(
    const user_float_t dist_sq, const user_float_t elec_const, const user_float_t charge, 
    const user_float_t rc_sq, const user_float_t alpha, const user_float_t elec_rep_max, 
    const user_float_t elec_attr_max
) {
    user_float_t cur_potential;
    // beyond cutoff
    if (dist_sq > rc_sq) {
        cur_potential = elec_const / dist_sq;
    // within cutoff
    } else {
        user_float_t alpha_tmp = alpha * dist_sq;
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
    const user_float_t* grid_pos, const user_float_t* coords, const user_float_t* charges, 
    const user_float_t* epsilons, const user_float_t* vdw_rs, const user_float_t cc_elec, 
    const user_float_t rad_dielec_const, const user_float_t elec_rep_max, 
    const user_float_t elec_attr_max, const user_float_t vwd_rep_max,
    const user_float_t vwd_attr_max, const int N_coords, const int N_grid_points,
    user_float_t* electrostat_grid, user_float_t* vdw_grid_attr, user_float_t* vdw_grid_rep
) {
    user_float_t dx, dy, dz, dist_sq;
    user_float_t r_min, eps_sqrt, vdwconst_attr, vdwconst_rep, r6;
    user_float_t beta_attr, beta_rep;
    user_float_t rc_sq_vdw_attr, rc_sq_vdw_rep;
    user_float_t charge, elec_const, emax_tmp, alpha, rc_sq_elec, cur_elec_potential;

    #pragma omp parallel for private( \
        dx, dy, dz, dist_sq, r_min, eps_sqrt, vdwconst_attr, vdwconst_rep, \
        rc_sq_vdw_attr, rc_sq_vdw_rep, beta_attr, beta_rep, elec_const, \
        emax_tmp, alpha, rc_sq_elec, cur_elec_potential, r6 \
    )
    for (int i = 0; i < N_coords; i++) {
        // calculate vdw constants
        r_min = vdw_rs[i];
        eps_sqrt = sqrt(fabs(epsilons[i]));
        vdwconst_attr = 0.5 * fabs(vwd_attr_max) / eps_sqrt;
        vdwconst_rep = 0.25 * fabs(vwd_rep_max) / eps_sqrt;
        // vdw cutoff distance squared
        rc_sq_vdw_attr = r_min * r_min * pow(vdwconst_attr, -1.0 / 3.0);
        rc_sq_vdw_rep = r_min * r_min * pow(vdwconst_rep, -1.0 / 6.0);
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
        rc_sq_elec = 2.0 * fabs(elec_const / emax_tmp);
        alpha = fabs(emax_tmp / (2.0 * rc_sq_elec));

        for (int j = 0; j < N_grid_points; j++) {
            // pairwise distances squared between the receptor atom coords 
            // and the grid points
            dx = grid_pos[j * 3] - coords[i * 3];
            dy = grid_pos[j * 3 + 1] - coords[i * 3 + 1];
            dz = grid_pos[j * 3 + 2] - coords[i * 3 + 2];
            dist_sq = dx * dx + dy * dy + dz * dz;
            // attractive vdw grid
            r6 = pow(r_min / dist_sq, 3.0); // (sqrt(r_min)/r)^6
            
            // repulsive vdw grid
            if (dist_sq > rc_sq_vdw_rep) {
                // beyond cutoff
                #pragma omp atomic
                vdw_grid_rep[j] += eps_sqrt * r6 * r6;
            } else {
                // within cutoff
                #pragma omp atomic
                vdw_grid_rep[j] += vwd_rep_max * 
                (1.0 - 0.5 * pow((dist_sq / rc_sq_vdw_rep), beta_rep));
            }
            // attractive vdw grid
            if (dist_sq > rc_sq_vdw_attr) {
                #pragma omp atomic
                vdw_grid_attr[j] -= 2.0 * eps_sqrt * r6;
            } else {
                // within cutoff
                #pragma omp atomic
                vdw_grid_attr[j] += vwd_attr_max * 
                (1.0 - 0.5 * pow((dist_sq / rc_sq_vdw_attr), beta_attr));
            }
            // electrostatic grid
            cur_elec_potential = calc_point_elec_potential(
                dist_sq, elec_const, charge, rc_sq_elec, alpha,
                elec_rep_max, elec_attr_max
            );
            #pragma omp atomic
            electrostat_grid[j] += cur_elec_potential;
        } // end grid points loop
    } // end coords loop
}