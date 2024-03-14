#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <fftw3.h>

#include "rank_poses.h"
#include "fft_correlate.h"

// Function to perform batch 3D FFT correlation
void fft_correlate_batch(
  PyArrayObject *recep_grid, PyArrayObject *result, int n_threads
) {
  // recep_grid: (n_grids, nx, ny, nz)
  // result: (n_orientations, n_grids, nx, ny, nz)

  // Get array dimensions
  int n_grids = PyArray_DIMS(recep_grid)[0];
  int n_orientations = PyArray_DIMS(result)[0];
  int nx = PyArray_DIMS(recep_grid)[1];
  int ny = PyArray_DIMS(recep_grid)[2];
  int nz = PyArray_DIMS(recep_grid)[3];
  int N_grid_points = nx * ny * nz;
  // Number FFT coefficients (only half of the array is needed due to symmetry)
  int N_fft_points = nx * ny * (nz / 2 + 1);

  // Get NumPy arrays data pointers
  float *recep_arr = (float *)PyArray_GETPTR1(recep_grid, 0);
  float *lig_arr = (float *)PyArray_GETPTR1(result, 0);

  // Allocate memory for FFTW plans and data
  fftwf_plan plan_fwd, plan_inv;
  fftwf_complex *fft_r, *fft_l;
  fft_r = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points);
  fft_l = (fftwf_complex *)fftwf_malloc(
    sizeof(fftwf_complex) * N_fft_points * n_orientations
  );

  // Create forward and inverse FFT plans for both arrays
  plan_fwd = fftwf_plan_dft_r2c_3d(nx, ny, nz, recep_arr, fft_r, FFTW_MEASURE);
  plan_inv = fftwf_plan_dft_c2r_3d(nx, ny, nz, fft_l, lig_arr, FFTW_MEASURE);

  float scale = 1.0 / N_grid_points;
  // Execute forward FFTs on both arrays
  for (int i = 0; i < n_grids; i++) {
    float *cur_recep = (float *)PyArray_GETPTR1(recep_grid, i);
    fftwf_execute_dft_r2c(plan_fwd, cur_recep, fft_r);
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_orientations; j++) {
      float *cur_lig = (float *)PyArray_GETPTR2(result, j, i);
      fftwf_complex *cur_fft_l = fft_l + j * N_fft_points;
      // The pointer to the padded ligand array is the same as the pointer 
      // to the correlation array
      fftwf_execute_dft_r2c(plan_fwd, cur_lig, cur_fft_l);
      // Perform element-wise complex conjugate multiplication
      for (int k = 0; k < N_fft_points; k++) {
        cur_fft_l[k] = conjf(fft_r[k]) * cur_fft_l[k] * scale;
      }
      // Execute inverse FFT on the product
      fftwf_execute_dft_c2r(plan_inv, cur_fft_l, cur_lig);
    }
  }

  // Clean up memory and plans
  fftwf_destroy_plan(plan_fwd);
  fftwf_destroy_plan(plan_inv);
  fftwf_free(fft_r);
  fftwf_free(fft_l);
}

// Function to sum all grids in the input array
// the output array will be flipped and rolled by roll_steps
// to correct for the index changes from fft_correlation
// roll step should be half of the probe grid dimension (ceiling of nx/2)
void flip_roll_and_sum(
  float *grids, float *result, int roll_steps, 
  int nx, int ny, int nz
){
  // Get array dimensions
  for (int x = nx - 1, new_x = 0; x >= 0; x--, new_x++) {
    for (int y = ny - 1, new_y = 0; y >= 0; y--, new_y++) {
        for (int z = nz - 1, new_z = 0; z >= 0; z--, new_z++) {
          int updated_idx = (new_x + roll_steps) % nx * ny * nz + 
            (new_y + roll_steps) % ny * nz + 
            (new_z + roll_steps) % nz;
          result[updated_idx] += grids[
            x * ny * nz + y * nz + z
          ];
        }
    }
  }
}

void sum_grids(
  PyArrayObject *grids, PyArrayObject *result, int roll_steps
) {
  // Get array dimensions
  int n_orientations = PyArray_DIMS(grids)[0];
  int n_grids = PyArray_DIMS(grids)[1];
  int nx = PyArray_DIMS(grids)[2];
  int ny = PyArray_DIMS(grids)[3];
  int nz = PyArray_DIMS(grids)[4];

  // Sum all grids
  #pragma omp parallel for
  for (int i = 0; i < n_orientations; i++) {
    float *cur_arr_result = (float *)PyArray_GETPTR1(result, i);
    for (int j = 0; j < n_grids; j++) {
      float *cur_arr_grids = (float *)PyArray_GETPTR2(grids, i, j);
      flip_roll_and_sum(
        cur_arr_grids, cur_arr_result, roll_steps, nx, ny, nz
      );
    }
  }
}

