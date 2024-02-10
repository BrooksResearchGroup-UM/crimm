#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>
#include <Python.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>

// Function to fill padded 3D array
void fill_pad_3d_array(
    int x, int y, int z,
    int pad_x, int pad_y, int pad_z,
    float *arr_l, float *padded_arr_l)
{
  for (int i = 0; i < x; i++)
  {
    for (int j = 0; j < y; j++)
    {
      float *padded_arr_l_ij = padded_arr_l + (pad_y * pad_z * i) + (pad_z * j);
      float *arr_l_ij = arr_l + (y * z * i) + z * j;
      memcpy(padded_arr_l_ij, arr_l_ij, z * sizeof(float));
    }
  }
}

// Function to fill the zero padded 4D array
void fill_pad_4d_array(
    int n_grids, int x, int y, int z,
    int pad_x, int pad_y, int pad_z,
    float *arr_l, float *padded_arr_l)
{
  int N_padded = pad_x * pad_y * pad_z;
  int N = x * y * z;
  for (int i = 0; i < n_grids; i++)
  {
    float *cur_padded_arr_l = padded_arr_l + (N_padded * i);
    float *cur_arr_l = arr_l + (N * i);
    fill_pad_3d_array(x, y, z, pad_x, pad_y, pad_z, cur_arr_l, cur_padded_arr_l);
  }
}

// Function to excute FFT correlation with created plans
void excute_fft_correlate(
    fftwf_plan plan_fwd, fftwf_plan plan_inv,
    fftwf_complex *fft_r, fftwf_complex *fft_l, fftwf_complex *fft_prod,
    float *arr_r, int n_orientations, int n_grids,
    int N_grid_points, int N_fft_points, float *corr)
{
  float scale = 1.0 / N_grid_points;
  // Execute forward FFTs on both arrays
  for (int i = 0; i < n_grids; i++)
  {
    float *cur_arr_r = arr_r + i * N_grid_points;
    fftwf_execute_dft_r2c(plan_fwd, cur_arr_r, fft_r);
    for (int j = 0; j < n_orientations; j++)
    {
      float *cur_corr = corr + (j * n_grids + i) * N_grid_points;
      // The pointer to the padded ligand array is the same as the pointer
      // to the correlation array
      fftwf_execute_dft_r2c(plan_fwd, cur_corr, fft_l);
      // Perform element-wise complex conjugate multiplication
      for (int k = 0; k < N_fft_points; k++)
      {
        fft_prod[k] = conjf(fft_r[k]) * fft_l[k] * scale;
      }
      // Execute inverse FFT on the product
      fftwf_execute_dft_c2r(plan_inv, fft_prod, cur_corr);
    }
  }
}

void excute_fft_correlate_batch(
    fftwf_plan plan_fwd_r, fftwf_plan plan_fwd_l, fftwf_plan plan_inv,
    fftwf_complex *fft_r, fftwf_complex *fft_l,
    float *arr_r, int n_orientations, int n_grids,
    int N_grid_points, int N_fft_points, float *corr)
{
  float scale = 1.0 / N_grid_points;
  // Execute forward FFTs on both arrays
  fftwf_execute_dft_r2c(plan_fwd_r, arr_r, fft_r);
  fftwf_execute_dft_r2c(plan_fwd_l, corr, fft_l);

  for (int i = 0; i < n_grids; i++){
    for (int j = 0; j < n_orientations; j++){
      for (int k = 0; k < N_fft_points; k++){
        int index_l = (j * n_grids + i) * N_fft_points + k;
        int index_r = i * N_fft_points + k;
        fft_l[index_l] = conjf(fft_r[index_r]) * fft_l[index_l] * scale;
      }
    }
  }
  fftwf_execute_dft_c2r(plan_inv, fft_l, corr);
}

// Function to perform 3D FFT correlation
void fft_correlate(
    PyArrayObject *recep_grid, PyArrayObject *lig_grid, PyArrayObject *result)
{
  // Check array dimensions and data type
  if (PyArray_NDIM(recep_grid) != 4 || PyArray_NDIM(lig_grid) != 4 ||
      PyArray_TYPE(recep_grid) != NPY_FLOAT32 || PyArray_TYPE(lig_grid) != NPY_FLOAT32)
  {
    PyErr_SetString(PyExc_TypeError, "Expected arrays of float32 with 4 dimensions");
    return;
  }

  int n_grids = PyArray_DIMS(recep_grid)[0];
  int n_grids_l = PyArray_DIMS(lig_grid)[0];
  int n_grids_result = PyArray_DIMS(result)[0];
  if (n_grids != n_grids_l || n_grids != n_grids_result)
  {
    PyErr_SetString(
        PyExc_TypeError,
        "Expected same number of grids for both receptor, ligand, and result arrays.");
    return;
  }

  // Get array dimensions
  int nx = PyArray_DIMS(recep_grid)[1];
  int ny = PyArray_DIMS(recep_grid)[2];
  int nz = PyArray_DIMS(recep_grid)[3];
  int N_grid_points = nx * ny * nz;
  // Number FFT coefficients (only half of the array is needed due to symmetry)
  int N_fft_points = nx * ny * (nz / 2 + 1);

  // Get lig array dimensions
  int nx_l = PyArray_DIMS(lig_grid)[1];
  int ny_l = PyArray_DIMS(lig_grid)[2];
  int nz_l = PyArray_DIMS(lig_grid)[3];

  // Get NumPy arrays data pointers
  float *arr_r = (float *)PyArray_DATA(recep_grid);
  float *arr_l = (float *)PyArray_DATA(lig_grid);
  float *corr = (float *)PyArray_DATA(result);

  // Allocate memory for the padded array
  // To save memory, we use corr as the padded array as it should be initialized to 0
  // and has the same shape. It will be overwritten in the excute_fft_correlate
  // after the inverse FFT to store the correlation result.
  fill_pad_4d_array(n_grids, nx_l, ny_l, nz_l, nx, ny, nz, arr_l, corr);

  // Allocate memory for FFTW plans and data
  fftwf_plan plan_fwd, plan_inv;
  fftwf_complex *fft_r, *fft_l, *fft_prod;
  fft_r = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points);
  fft_l = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points);
  fft_prod = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points);

  // Create forward and inverse FFT plans for both arrays
  plan_fwd = fftwf_plan_dft_r2c_3d(nx, ny, nz, arr_r, fft_r, FFTW_ESTIMATE);
  plan_inv = fftwf_plan_dft_c2r_3d(nx, ny, nz, fft_prod, corr, FFTW_ESTIMATE);

  // Execute forward FFTs on all grids
  excute_fft_correlate(
      plan_fwd, plan_inv,
      fft_r, fft_l, fft_prod, arr_r,
      1, n_grids, N_grid_points, N_fft_points, corr);

  // Clean up memory and plans
  fftwf_destroy_plan(plan_fwd);
  fftwf_destroy_plan(plan_inv);
  fftwf_free(fft_r);
  fftwf_free(fft_l);
  fftwf_free(fft_prod);
}

// Function to perform batch 3D FFT correlation
void fft_correlate_batch(
    PyArrayObject *recep_grid, PyArrayObject *lig_grid, PyArrayObject *result)
{
  // recep_grid: (n_grids, nx, ny, nz)
  // lig_grid: (n_orientations, n_grids, nx, ny, nz)
  // result: (n_orientations, n_grids, nx, ny, nz)

  printf("fft_correlate_batch\n");

  // Check array dimensions and data type
  if (PyArray_NDIM(recep_grid) != 4 || PyArray_TYPE(recep_grid) != NPY_FLOAT32)
  {
    PyErr_SetString(
        PyExc_TypeError, "Expected receptor arrays of float32 with 4 dimensions");
    return;
  }

  if (PyArray_NDIM(lig_grid) != 5 || PyArray_TYPE(lig_grid) != NPY_FLOAT32)
  {
    PyErr_SetString(
        PyExc_TypeError, "Expected ligand arrays of float32 with 5 dimensions");
    return;
  }

  if (PyArray_NDIM(result) != 5 || PyArray_TYPE(result) != NPY_FLOAT32)
  {
    PyErr_SetString(
        PyExc_TypeError, "Expected result arrays of float32 with 5 dimensions");
    return;
  }

  int n_grids = PyArray_DIMS(recep_grid)[0];
  int n_grids_l = PyArray_DIMS(lig_grid)[1];
  int n_grids_result = PyArray_DIMS(result)[1];
  if (n_grids != n_grids_l || n_grids != n_grids_result)
  {
    PyErr_SetString(
        PyExc_TypeError,
        "Expected same number of grids for both receptor, ligand, and result arrays.");
    return;
  }

  int n_orientations = PyArray_DIMS(lig_grid)[0];
  if (n_orientations != PyArray_DIMS(result)[0])
  {
    PyErr_SetString(
        PyExc_TypeError,
        "Expected same number of orientations for ligand and result arrays.");
    return;
  }

  // Get array dimensions
  int nx = PyArray_DIMS(recep_grid)[1];
  int ny = PyArray_DIMS(recep_grid)[2];
  int nz = PyArray_DIMS(recep_grid)[3];
  int N_grid_points = nx * ny * nz;
  // Number FFT coefficients (only half of the array is needed due to symmetry)
  int N_fft_points = nx * ny * (nz / 2 + 1);

  // Get lig array dimensions
  int nx_l = PyArray_DIMS(lig_grid)[2];
  int ny_l = PyArray_DIMS(lig_grid)[3];
  int nz_l = PyArray_DIMS(lig_grid)[4];

  // Get NumPy arrays data pointers
  float *arr_r = (float *)PyArray_DATA(recep_grid);
  float *arr_l = (float *)PyArray_DATA(lig_grid);
  float *corr = (float *)PyArray_DATA(result);

  // Allocate memory for the padded array
  // To save memory, we use corr as the padded array as it should be initialized to 0
  // and has the same shape. It will be overwritten in the excute_fft_correlate
  // after the inverse FFT to store the correlation result.
  fill_pad_4d_array(n_orientations * n_grids, nx_l, ny_l, nz_l, nx, ny, nz, arr_l, corr);

  // Allocate memory for FFTW plans and data
  fftwf_plan plan_fwd_r, plan_fwd_l, plan_inv;
  fftwf_complex *fft_r, *fft_l;
  fft_r = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points * n_grids);
  fft_l = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points * n_orientations * n_grids);

  // Create forward and inverse FFT plans for both arrays
  plan_fwd_r = fftwf_plan_many_dft_r2c(
      3, (int[]){nx, ny, nz}, n_grids, arr_r, NULL, 1, N_grid_points,
      fft_r, NULL, 1, N_fft_points,
      FFTW_ESTIMATE);

  int success = fftwf_init_threads();
  if (success == 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to initialize FFTW threads");
    return;
  }
  // int n_threads = fftwf_planner_nthreads();
  int n_threads = 4;
  fftwf_plan_with_nthreads(n_threads);
  printf("Number of threads: %d\n", n_threads);

  plan_fwd_l = fftwf_plan_many_dft_r2c(
      3, (int[]){nx, ny, nz}, n_orientations*n_grids, corr, NULL, 1, N_grid_points,
      fft_l, NULL, 1, N_fft_points,
      FFTW_ESTIMATE);
  plan_inv = fftwf_plan_many_dft_c2r(
      3, (int[]){nx, ny, nz}, n_orientations*n_grids, fft_l, NULL, 1, N_fft_points,
      corr, NULL, 1, N_grid_points,
      FFTW_ESTIMATE);

  // Execute forward FFTs on all grids
  excute_fft_correlate_batch(
      plan_fwd_r, plan_fwd_l, plan_inv,
      fft_r, fft_l, arr_r,
      n_orientations, n_grids, N_grid_points, N_fft_points, corr);

  // Clean up memory and plans
  fftwf_destroy_plan(plan_fwd_r);
  fftwf_destroy_plan(plan_fwd_l);
  fftwf_destroy_plan(plan_inv);
  fftwf_free(fft_r);
  fftwf_free(fft_l);
  fftwf_cleanup_threads();
}

// Wrapper function to be called from Python
static PyObject *py_fft_correlate(PyObject *self, PyObject *args)
{
  PyArrayObject *recep_grid, *lig_grid, *result;
  if (!PyArg_ParseTuple(
          args, "O!O!O!",
          &PyArray_Type, &recep_grid,
          &PyArray_Type, &lig_grid,
          &PyArray_Type, &result))
  {
    return NULL;
  }

  fft_correlate(recep_grid, lig_grid, result);

  Py_RETURN_NONE;
}

// Wrapper function to be called from Python
static PyObject *py_fft_correlate_batch(PyObject *self, PyObject *args)
{
  PyArrayObject *recep_grid, *lig_grid, *result;
  if (!PyArg_ParseTuple(
          args, "O!O!O!",
          &PyArray_Type, &recep_grid,
          &PyArray_Type, &lig_grid,
          &PyArray_Type, &result))
  {
    return NULL;
  }

  fft_correlate_batch(recep_grid, lig_grid, result);

  Py_RETURN_NONE;
}

// Method table
static PyMethodDef FftCorrelateMethods[] = {
    {"fft_correlate", py_fft_correlate, METH_VARARGS, "FFT Correlation"},
    {"fft_correlate_batch", py_fft_correlate_batch, METH_VARARGS, "FFT Correlation Batch"},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef fftcorrelatemodule = {
    PyModuleDef_HEAD_INIT,
    "fft_correlate", // name of module
    NULL,            // module documentation, may be NULL
    -1,              // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    FftCorrelateMethods};

// Initialization function
PyMODINIT_FUNC PyInit_fft_correlate(void)
{
  import_array();
  return PyModule_Create(&fftcorrelatemodule);
}