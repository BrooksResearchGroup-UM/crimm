#ifndef FFT_CORRELATE_H
#define FFT_CORRELATE_H

// Function to fill the padded 3D array for ligand grid for FFT
void fill_padded_array(
  int x, int y, int z, 
  int pad_x, int pad_y, int pad_z,
  float *arr_l, float *padded_arr_l
);

// Function to perform 3D FFT correlation
void fft_correlate(
  float *recep_arr, float *lig_arr, int N_grids, 
  // Protein receptor grid dimensions
  int nx, int ny, int nz,
  // Ligand grid dimensions
  int nx_lig, int ny_lig, int nz_lig,
  int N_orientations, int N_threads, 
  float *result_arr
);

void sum_grids(
  float *grids, float *result, size_t N_orietations, int N_grids, size_t N_grid_points
);

#endif