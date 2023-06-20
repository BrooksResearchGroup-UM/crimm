#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper function for checking CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
        __FILE__, __LINE__, result, cudaGetErrorString(result), #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void calc_pairwise_dist_kernel(
    double* grid_pos, double* coords, int N_COORDS, int N_GRID_POINTS, 
    double* dists
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
  
    if (i < N_GRID_POINTS && j < N_COORDS) {
        double dx, dy, dz;
        
        dx = grid_pos[i * 3] - coords[j * 3];
        dy = grid_pos[i * 3 + 1] - coords[j * 3 + 1];
        dz = grid_pos[i * 3 + 2] - coords[j * 3 + 2];
        
        dists[i * N_COORDS + j] = sqrt(dx * dx + dy * dy + dz * dz);
    }
}

__device__ double calc_point_elec_potential(
    double dist, double elec_const, double charge, double rc, double alpha,
    double elec_rep_max, double elec_attr_max
) {
    double cur_potential;
    
    if (dist > rc) {
        cur_potential = elec_const / (dist * dist);
    } else {
        double alpha_tmp = alpha * dist * dist;
        if (charge > 0) {
            cur_potential = elec_rep_max - alpha_tmp;
        } else {
            cur_potential = elec_attr_max + alpha_tmp;
        }
    }
    
    return cur_potential;
}

__global__ void gen_elec_grid_kernel(
    const double* dists, const double* charges, const double cc_elec,
    const double rad_dielec_const, const double elec_rep_max, 
    const double elec_attr_max,
    const int N_COORDS, const int N_GRID_POINTS, double* electrostat_grid
) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_GRID_POINTS) {
        double elec_const, rc, alpha;
        double emax_tmp;
        double cur_grid_val = 0.0;
        for (int j = 0; j < N_COORDS; j++) {
            double dist = dists[i * N_COORDS + j];

            elec_const = cc_elec * charges[j] / rad_dielec_const;

            if (charges[j] > 0) {
                emax_tmp = elec_rep_max;
            } else {
                emax_tmp = elec_attr_max;
            }
            rc = sqrt(2.0 * fabs(elec_const / emax_tmp));
            alpha = fabs(emax_tmp / (2.0 * rc * rc));

            double cur_potential = calc_point_elec_potential(
                dist, elec_const, charges[j], rc, alpha, 
                elec_rep_max, elec_attr_max
            );
            cur_grid_val += cur_potential;
        }
        electrostat_grid[i] = cur_grid_val;
    }
}

__device__ double calc_point_vdw_potential(
    double dist, double epsilon, double r_min, double probe_radius, 
    double vwd_softcore_max, double rc_vdw, double beta
) {
    double cur_potential;

    double r_min_over_dist = r_min / dist;
    if (dist > rc_vdw) {
        cur_potential = epsilon * (pow(r_min_over_dist, 12.0) - 2.0 * pow(r_min_over_dist, 6.0));
    } else {
        cur_potential = vwd_softcore_max * (1.0 - 0.5 * pow((dist / rc_vdw), beta));
    }

    return cur_potential;
}

__global__ void gen_vdw_grid_kernel(
    double* dists, double* epsilons, double* vdw_rs, double probe_radius, 
    double vwd_softcore_max, int N_COORDS, int N_GRID_POINTS, double* vdw_grid
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N_GRID_POINTS) {
        double r_mins, eps_sqrt, rc_vdw, beta;
        double cur_grid_val = 0.0;
        for (int j = 0; j < N_COORDS; j++) {
            double dist = dists[i * N_COORDS + j];

            r_mins = vdw_rs[j] + probe_radius;
            eps_sqrt = sqrt(fabs(epsilons[j]));
            rc_vdw = r_mins * pow(1.0 + sqrt(1.0 + 0.5 * fabs(vwd_softcore_max) / eps_sqrt), -1.0 / 6.0);
            beta = 24.0 * eps_sqrt / vwd_softcore_max * \
            (pow(1.0 + sqrt(1.0 + 0.5 * fabs(vwd_softcore_max) / eps_sqrt), 2.0)\
             - 1.0 - sqrt(1.0 + 0.5 * fabs(vwd_softcore_max) / eps_sqrt));

            double cur_potential = calc_point_vdw_potential(
                dist, epsilons[j], r_mins, probe_radius, 
                vwd_softcore_max, rc_vdw, beta
            );
            cur_grid_val += cur_potential;
        }
        vdw_grid[i] = cur_grid_val;
    }
}

extern "C"
void calc_pairwise_dist(
    double* host_grid_pos, double* host_coords, 
    const int N_GRID_POINTS, const int N_COORDS, double* host_dists
) {
    // Allocate memory on the device
    double* device_grid_pos;
    double* device_coords;
    double* device_dists;

    cudaMalloc((void**)&device_grid_pos, N_GRID_POINTS * 3 * sizeof(double));
    cudaMalloc((void**)&device_coords, N_COORDS * 3 * sizeof(double));
    cudaMalloc((void**)&device_dists, N_GRID_POINTS * N_COORDS * sizeof(double));

    // Copy data to the device
    cudaMemcpy(device_grid_pos, host_grid_pos, N_GRID_POINTS * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_coords, host_coords, N_COORDS * 3 * sizeof(double), cudaMemcpyHostToDevice);

    // Run the kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid(
        (N_GRID_POINTS + dimBlock.x - 1) / dimBlock.x,
        (N_COORDS + dimBlock.y - 1) / dimBlock.y
    );
    calc_pairwise_dist_kernel<<<dimGrid, dimBlock>>>(
        device_grid_pos, device_coords, N_GRID_POINTS, N_COORDS, device_dists
    );

    // Copy data back to the host
    cudaMemcpy(
        host_dists, device_dists, 
        N_GRID_POINTS * N_COORDS * sizeof(double), 
        cudaMemcpyDeviceToHost
    );

    // Free memory on the device
    cudaFree(device_grid_pos);
    cudaFree(device_coords);
    cudaFree(device_dists);
}

extern "C"
void gen_elec_grid(
    double* host_dists, double* host_charges, const double cc_elec, 
    const double rad_dielec_const, const double elec_rep_max, 
    const double elec_attr_max, 
    const int N_GRID_POINTS, const int N_COORDS, 
    double* host_electrostat_grid
) {
    // Allocate memory on the device
    double* device_dists;
    double* device_charges;
    double* device_electrostat_grid;

    cudaMalloc((void**)&device_dists, N_GRID_POINTS * N_COORDS * sizeof(double));
    cudaMalloc((void**)&device_charges, N_COORDS * sizeof(double));
    cudaMalloc((void**)&device_electrostat_grid, N_GRID_POINTS * sizeof(double));

    // Copy data to the device
    cudaMemcpy(device_dists, host_dists, N_GRID_POINTS * N_COORDS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_charges, host_charges, N_COORDS * sizeof(double), cudaMemcpyHostToDevice);

    // Run the kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid(
        (N_GRID_POINTS + dimBlock.x - 1) / dimBlock.x
        // (N_COORDS + dimBlock.y - 1) / dimBlock.y
    );
    gen_elec_grid_kernel<<<dimGrid, dimBlock>>>(
        device_dists, device_charges, cc_elec, 
        rad_dielec_const, elec_rep_max, elec_attr_max, 
        N_GRID_POINTS, N_COORDS, device_electrostat_grid
    );

    // Copy data back to the host
    cudaMemcpy(
        host_electrostat_grid, device_electrostat_grid, 
        N_GRID_POINTS * sizeof(double), cudaMemcpyDeviceToHost
    );

    // Free memory on the device
    cudaFree(device_dists);
    cudaFree(device_charges);
    cudaFree(device_electrostat_grid);
}

extern "C"
void gen_vdw_grid(
    double* host_dists, double* host_epsilons, double* host_vdw_rs, 
    const double probe_radius, const double vwd_softcore_max, 
    const int N_GRID_POINTS, const int N_COORDS, double* host_vdw_grid
) {
    // Allocate memory on the device
    double* device_dists;
    double* device_epsilons;
    double* device_vdw_rs;
    double* device_vdw_grid;

    cudaMalloc((void**)&device_dists, N_GRID_POINTS * N_COORDS * sizeof(double));
    cudaMalloc((void**)&device_epsilons, N_COORDS * sizeof(double));
    cudaMalloc((void**)&device_vdw_rs, N_COORDS * sizeof(double));
    cudaMalloc((void**)&device_vdw_grid, N_GRID_POINTS * sizeof(double));

    // Copy data to the device
    cudaMemcpy(device_dists, host_dists, N_GRID_POINTS * N_COORDS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_epsilons, host_epsilons, N_COORDS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vdw_rs, host_vdw_rs, N_COORDS * sizeof(double), cudaMemcpyHostToDevice);

    // Run the kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid(
        (N_GRID_POINTS + dimBlock.x - 1) / dimBlock.x 
        // (N_COORDS + dimBlock.y - 1) / dimBlock.y
    );
    gen_vdw_grid_kernel<<<dimGrid, dimBlock>>>(
        device_dists, device_epsilons, device_vdw_rs, 
        probe_radius, vwd_softcore_max, 
        N_GRID_POINTS, N_COORDS, device_vdw_grid
    );

    // Copy data back to the host
    cudaMemcpy(
        host_vdw_grid, device_vdw_grid, 
        N_GRID_POINTS * sizeof(double), cudaMemcpyDeviceToHost
    );

    // Free memory on the device
    cudaFree(device_dists);
    cudaFree(device_epsilons);
    cudaFree(device_vdw_rs);
    cudaFree(device_vdw_grid);
}